#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

# ---- silence annoying multi-process import warnings (jieba/pkg_resources) ----
import warnings
warnings.filterwarnings(
    "ignore",
    message=r"pkg_resources is deprecated as an API\..*",
    category=UserWarning,
)

import argparse
import json
import math
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

# ---- ensure repo root is importable (spawn subprocesses need this) ----
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ============== attack-side shared tokenizer (fork COW) ==============
_SHARED_TOKENIZER = None  # set in main() if attack_start_method == "fork"

# ============== detect-side shared tokenizer (fork COW) ==============
_DETECT_TOKENIZER = None  # set in main() if detect_start_method == "fork"
_DETECT_EXECUTORS: Dict[Tuple[Tuple[str, ...], int, str], ProcessPoolExecutor] = {}
_VERBOSE = False


# -----------------------------
# CSV helpers
# -----------------------------
def read_json(p: Path) -> Dict[str, Any]:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def parse_csv_list(s: str, cast=float) -> List[Any]:
    s = (s or "").strip()
    if not s:
        return []
    return [cast(x.strip()) for x in s.split(",") if x.strip()]

def pick_text_col(df: pd.DataFrame) -> str:
    candidates = [
        "text", "output_text", "generated_text", "gen_text",
        "completion", "continuation", "watermarked_text",
        "response", "output", "sample", "full_text",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    for c in df.columns:
        if df[c].dtype == object:
            return c
    raise ValueError(f"Cannot find a text column. columns={list(df.columns)}")

def find_prompt_col(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "prompt", "prompts", "prompt_text", "prompt_str", "prefix",
        "input", "input_text", "query", "context",
        "source", "instruction",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    return None

def add_pid_if_needed(df: pd.DataFrame, key: str) -> pd.DataFrame:
    if key == "_pid" and "_pid" not in df.columns:
        df["_pid"] = np.arange(len(df), dtype=np.int64)
    return df

def conservative_threshold(z_neg: np.ndarray, target_fpr: float) -> Tuple[float, float]:
    z = z_neg[np.isfinite(z_neg)]
    if z.size == 0:
        raise ValueError("Empty NEG stats after filtering NaNs.")
    z_sorted = np.sort(z)
    n = z_sorted.size
    q = 1.0 - float(target_fpr)
    idx = int(math.ceil(q * n) - 1)
    idx = max(0, min(idx, n - 1))
    thr = float(z_sorted[idx])
    achieved = float(np.mean(z > thr))
    return thr, achieved

def tpr_at_thr(z_pos: np.ndarray, thr: float) -> float:
    z = z_pos[np.isfinite(z_pos)]
    if z.size == 0:
        return float("nan")
    return float(np.mean(z > thr))


# -----------------------------
# Character attack (RandomAttack from repo root random_attack.py)
# -----------------------------
class _NullLogger:
    class _L:
        def info(self, *args, **kwargs):
            return
    def __init__(self) -> None:
        self.logger = self._L()

def ops_to_char_ops(ops: Sequence[str]) -> List[int]:
    # RandomAttack.char_attack1 uses:
    # 1 delete, 2 homo-replace, 3 insert zwsp/zwj, 4 swap, 5 typo
    m = {"delete": 1, "replace": 2, "homo": 2, "insert": 3, "swap": 4, "typo": 5}
    out: List[int] = []
    for op in ops:
        op = op.strip().lower()
        if not op:
            continue
        if op not in m:
            raise ValueError(f"Unknown op '{op}'. Supported: {sorted(m.keys())}")
        out.append(m[op])
    return out or [2]

# globals in attack workers
_G_ATTACKER = None
_G_CHAR_OPS: List[int] = []
_G_EDIT_RATIO: float = 0.0
_G_ATTACK_GENERATED_ONLY: bool = False
_G_SEED_BASE: int = 1234
_G_DETECTOR_CACHE: Dict[Tuple[Any, ...], Any] = {}

def _attack_worker_init(
    model_or_tok_path: str,
    char_ops: List[int],
    edit_ratio: float,
    attack_generated_only: bool,
    seed_base: int,
) -> None:
    """Initializer for attack worker processes."""
    global _G_ATTACKER, _G_CHAR_OPS, _G_EDIT_RATIO, _G_ATTACK_GENERATED_ONLY, _G_SEED_BASE
    global _SHARED_TOKENIZER

    # deterministic + explicit: RandomAttack is in repo-root random_attack.py
    from random_attack import RandomAttack
    from transformers import AutoTokenizer

    tok = _SHARED_TOKENIZER
    if tok is None:
        tok = AutoTokenizer.from_pretrained(model_or_tok_path, use_fast=True)

    _G_ATTACKER = RandomAttack(
        tokenizer=tok,
        logger=_NullLogger(),
        ref_model=None,
        device="cpu",
        wm_name="",
        ori_flag=False,
        wm_detector=None,
        ppl_checker=None,
        char_op=2,
        def_stl="",
    )

    _G_CHAR_OPS = list(char_ops)
    _G_EDIT_RATIO = float(edit_ratio)
    _G_ATTACK_GENERATED_ONLY = bool(attack_generated_only)
    _G_SEED_BASE = int(seed_base)

def _attack_one(idx: int, text: str, prompt: Optional[str]) -> Tuple[int, str]:
    global _G_ATTACKER, _G_CHAR_OPS, _G_EDIT_RATIO, _G_ATTACK_GENERATED_ONLY, _G_SEED_BASE
    assert _G_ATTACKER is not None

    random.seed(_G_SEED_BASE + idx)
    np.random.seed(_G_SEED_BASE + idx)

    base = text if isinstance(text, str) else ""
    if not base:
        return idx, base

    prefix = ""
    suffix = base
    if _G_ATTACK_GENERATED_ONLY and prompt and isinstance(prompt, str) and base.startswith(prompt):
        prefix = prompt
        suffix = base[len(prompt):]

    per_pass = float(_G_EDIT_RATIO) / max(1, len(_G_CHAR_OPS))
    cur = suffix
    for op in _G_CHAR_OPS:
        _G_ATTACKER.char_op = int(op)
        adv = _G_ATTACKER.get_adv(
            cur,
            atk_style="char",
            max_edit_rate=per_pass,
            atk_times=1,
            target_class=0,
        )
        cur = adv["sentence"] if isinstance(adv, dict) and "sentence" in adv else cur

    return idx, prefix + cur

def parallel_attack(
    texts: List[str],
    prompts: List[Optional[str]],
    *,
    model_or_tok_path: str,
    ops: List[str],
    edit_ratio: float,
    attack_generated_only: bool,
    seed_base: int,
    num_workers: int,
    start_method: str = "fork",
) -> List[str]:
    import multiprocessing as mp

    char_ops = ops_to_char_ops(ops)
    n = len(texts)
    out = [""] * n
    ctx = mp.get_context(start_method)

    with ProcessPoolExecutor(
        max_workers=num_workers,
        mp_context=ctx,
        initializer=_attack_worker_init,
        initargs=(model_or_tok_path, char_ops, edit_ratio, attack_generated_only, seed_base),
    ) as ex:
        futs = [ex.submit(_attack_one, i, t, p) for i, (t, p) in enumerate(zip(texts, prompts))]
        for fut in as_completed(futs):
            i, attacked = fut.result()
            out[i] = attacked
    return out


# -----------------------------
# Detection (KGW / ByteKGWv5)
# -----------------------------
class _LiteKGWDetector:
    """Lightweight KGW detector that only needs tokenizer and config (no model)."""

    def __init__(self, *, cfg_path: str, tokenizer, device: str) -> None:
        cfg = read_json(Path(cfg_path))
        self.gamma = float(cfg.get("gamma", 0.5))
        self.hash_key = int(cfg.get("hash_key", 15485863))
        self.prefix_length = int(cfg.get("prefix_length", 4))
        self.z_threshold = float(cfg.get("z_threshold", 4.0))
        self.f_scheme = str(cfg.get("f_scheme", "time"))
        self.window_scheme = str(cfg.get("window_scheme", "left"))
        self.vocab_size = len(tokenizer)
        self.device = torch.device(device)
        self.tokenizer = tokenizer

        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(self.hash_key)
        self.prf = torch.randperm(self.vocab_size, device=self.device, generator=self.rng)

    def _f(self, input_ids: torch.LongTensor) -> int:
        ids = input_ids
        if self.f_scheme == "time":
            time_result = 1
            for i in range(self.prefix_length):
                time_result *= ids[-1 - i].item()
            return int(self.prf[time_result % self.vocab_size].item())
        if self.f_scheme == "additive":
            additive_result = 0
            for i in range(self.prefix_length):
                additive_result += ids[-1 - i].item()
            return int(self.prf[additive_result % self.vocab_size].item())
        if self.f_scheme == "skip":
            return int(self.prf[ids[-self.prefix_length].item()].item())
        if self.f_scheme == "min":
            return min(int(self.prf[ids[-1 - i].item()].item()) for i in range(self.prefix_length))
        raise ValueError(f"Unknown f_scheme {self.f_scheme}")

    def _get_greenlist_ids_left(self, input_ids: torch.LongTensor) -> List[int]:
        self.rng.manual_seed((self.hash_key * self._f(input_ids)) % self.vocab_size)
        greenlist_size = int(self.vocab_size * self.gamma)
        vocab_permutation = torch.randperm(self.vocab_size, device=self.device, generator=self.rng)
        return vocab_permutation[:greenlist_size].tolist()

    def _get_greenlist_ids_self(self, input_ids: torch.LongTensor) -> List[int]:
        greenlist_size = int(self.vocab_size * self.gamma)
        greenlist_ids: List[int] = []
        f_x = self._f(input_ids)
        for k in range(self.vocab_size):
            h_k = f_x * int(self.prf[k].item())
            self.rng.manual_seed(h_k % self.vocab_size)
            vocab_permutation = torch.randperm(self.vocab_size, device=self.device, generator=self.rng)
            temp_greenlist_ids = vocab_permutation[:greenlist_size]
            if k in temp_greenlist_ids:
                greenlist_ids.append(int(k))
        return greenlist_ids

    def _compute_z(self, observed: int, total: int) -> float:
        expected = self.gamma
        numer = observed - expected * total
        denom = math.sqrt(max(total * expected * (1.0 - expected), 1e-12))
        return float(numer / denom)

    def score(self, text: str) -> float:
        ids = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(self.device)
        if len(ids) - self.prefix_length < 1:
            return 0.0
        green_count = 0
        for idx in range(self.prefix_length, len(ids)):
            cur_ids = ids[:idx]
            if self.window_scheme == "left":
                greenlist = self._get_greenlist_ids_left(cur_ids)
            else:
                greenlist = self._get_greenlist_ids_self(cur_ids)
            if int(ids[idx].item()) in greenlist:
                green_count += 1
        return self._compute_z(green_count, len(ids) - self.prefix_length)


class _LiteByteDetector:
    """Lightweight ByteKGWv5 detector that only needs tokenizer and config (no model)."""

    def __init__(
        self,
        *,
        cfg_path: str,
        tokenizer,
        device: str,
        byte_maxpos: Optional[int],
        pos_weights: Optional[np.ndarray],
        use_prefix_override: Optional[bool] = None,
    ) -> None:
        from MarkLLM.watermark.bytekgwV5.detector import ByteKGWv5Detector, ByteTreeDetectorConfig
        from MarkLLM.watermark.bytekgwV5.prf import BytePRF, PRFConfig
        from MarkLLM.watermark.bytekgwV5.token_bytes import TokenByteVocab

        cfg = read_json(Path(cfg_path))
        gamma = float(cfg.get("gamma", 0.5))
        hash_key = int(cfg.get("hash_key", 15485863))
        prefix_length = int(cfg.get("prefix_length", 4))
        z_threshold = float(cfg.get("z_threshold", 4.0))
        max_byte_pos_cfg = int(cfg.get("max_byte_pos", 64))
        max_byte_pos_use = int(byte_maxpos) if byte_maxpos is not None else max_byte_pos_cfg
        use_prefix_bytes_in_prf = bool(cfg.get("use_prefix_bytes_in_prf", False)) if use_prefix_override is None else bool(use_prefix_override)
        add_special_tokens = bool(cfg.get("add_special_tokens", True))

        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.add_special_tokens = add_special_tokens

        prf = BytePRF(PRFConfig(hash_key=hash_key, gamma=gamma), device=self.device)
        vocab = TokenByteVocab.from_tokenizer(self.tokenizer, skip_markers=True).to(self.device)

        det_cfg = ByteTreeDetectorConfig(
            prefix_length=prefix_length,
            gamma=gamma,
            z_threshold=z_threshold,
            max_byte_pos=max_byte_pos_use,
            use_prefix_bytes_in_prf=use_prefix_bytes_in_prf,
            pos_weights=None if pos_weights is None else [float(x) for x in pos_weights.tolist()],
        )
        self.detector = ByteKGWv5Detector(prf=prf, vocab=vocab, cfg=det_cfg, device=self.device)

    def score(self, text: str) -> float:
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=self.add_special_tokens,
        )
        input_ids = enc["input_ids"].to(self.device)
        ret = self.detector.detect(input_ids)
        z = ret["z"]
        if isinstance(z, torch.Tensor):
            return float(z[0].item() if z.dim() else z.item())
        return float(z)

def detect_worker(
    algo: str,
    texts: List[str],
    *,
    model_path: str,
    kgw_cfg_path: str,
    byte_cfg_path: str,
    device: str,
        dtype: str,
        delta: float,
        byte_maxpos: Optional[int],
        byte_weights: Optional[np.ndarray],
        byte_use_prefix: Optional[bool],
) -> List[float]:
    from transformers import AutoTokenizer
    global _DETECT_TOKENIZER, _G_DETECTOR_CACHE, _VERBOSE

    t_start = time.time()
    tok = _DETECT_TOKENIZER
    if tok is None:
        tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        if tok.pad_token_id is None and tok.eos_token_id is not None:
            tok.pad_token = tok.eos_token
        if _DETECT_TOKENIZER is None:
            _DETECT_TOKENIZER = tok
        if _VERBOSE:
            print(f"[detect_worker] loaded tokenizer {model_path} t={time.time() - t_start:.2f}s", flush=True)

    def weight_sig(arr: Optional[np.ndarray]) -> Any:
        if arr is None:
            return None
        return tuple(float(x) for x in arr.tolist())

    key: Tuple[Any, ...]
    if algo == "kgw":
        key = ("kgw", model_path, kgw_cfg_path, device)
    else:
        key = (
            "byte",
            model_path,
            byte_cfg_path,
            device,
            int(byte_maxpos) if byte_maxpos is not None else None,
            weight_sig(byte_weights),
            None if byte_use_prefix is None else bool(byte_use_prefix),
        )

    detector = _G_DETECTOR_CACHE.get(key)
    if detector is None:
        t_det_start = time.time()
        if algo == "kgw":
            detector = _LiteKGWDetector(cfg_path=kgw_cfg_path, tokenizer=tok, device=device)
        else:
            detector = _LiteByteDetector(
                cfg_path=byte_cfg_path,
                tokenizer=tok,
                device=device,
                byte_maxpos=byte_maxpos,
                pos_weights=byte_weights,
                use_prefix_override=byte_use_prefix,
            )
        _G_DETECTOR_CACHE[key] = detector
        if _VERBOSE:
            print(f"[detect_worker] built detector key={key} t={time.time() - t_det_start:.2f}s", flush=True)
    else:
        if _VERBOSE:
            print(f"[detect_worker] reuse detector key={key}", flush=True)

    t_score_start = time.time()
    out: List[float] = []
    for t in texts:
        out.append(detector.score(t))
    if _VERBOSE:
        print(f"[detect_worker] scored {len(texts)} texts in {time.time() - t_score_start:.2f}s (total {time.time() - t_start:.2f}s)", flush=True)
    return out

def detect_multi_gpu(
    algo: str,
    texts: List[str],
    *,
    model_path: str,
    kgw_cfg_path: str,
    byte_cfg_path: str,
    devices: List[str],
    workers: Optional[int],
    dtype: str,
    delta: float,
    byte_maxpos: Optional[int],
    byte_weights: Optional[np.ndarray],
    start_method: str = "spawn",
    label: str = "",
    byte_use_prefix: Optional[bool] = None,
) -> np.ndarray:
    import multiprocessing as mp
    ctx = mp.get_context(start_method)

    n = len(texts)
    dev_is_cpu = all(str(d).lower().startswith("cpu") for d in devices)
    if dev_is_cpu:
        workers_use = int(workers) if workers and workers > 0 else len(devices)
    else:
        # GPU: cap workers to device count to avoid spawning too many CUDA contexts
        w = int(workers) if workers and workers > 0 else len(devices)
        workers_use = min(len(devices), max(1, w))
    workers_use = max(1, workers_use)

    # round-robin assign worker -> device
    dev_cycle = [devices[i % len(devices)] for i in range(workers_use)]
    shard_idxs: List[List[int]] = [[] for _ in range(workers_use)]
    for i in range(n):
        shard_idxs[i % workers_use].append(i)

    zs = np.zeros(n, dtype=np.float32)

    key = (tuple(dev_cycle), workers_use, start_method)
    ex = _DETECT_EXECUTORS.get(key)
    if ex is None:
        ex = ProcessPoolExecutor(max_workers=workers_use, mp_context=ctx)
        _DETECT_EXECUTORS[key] = ex
        if _VERBOSE:
            shard_nonempty = len([s for s in shard_idxs if s])
            print(f"[detect_pool] create key={key} shards={shard_nonempty}", flush=True)
    else:
        if _VERBOSE:
            shard_nonempty = len([s for s in shard_idxs if s])
            print(f"[detect_pool] reuse key={key} shards={shard_nonempty}", flush=True)

    fut_to_idxs = {}
    for dev, idxs in zip(dev_cycle, shard_idxs):
        if not idxs:
            continue
        sub = [texts[i] for i in idxs]
        if _VERBOSE:
            print(f"[detect_submit] dev={dev} n={len(sub)}", flush=True)
        fut = ex.submit(
            detect_worker,
            algo,
            sub,
            model_path=model_path,
            kgw_cfg_path=kgw_cfg_path,
            byte_cfg_path=byte_cfg_path,
            device=dev,
            dtype=dtype,
            delta=delta,
            byte_maxpos=byte_maxpos,
            byte_weights=byte_weights,
            byte_use_prefix=byte_use_prefix,
        )
        fut_to_idxs[fut] = idxs

    pbar = None
    if tqdm is not None:
        pbar = tqdm(total=len(fut_to_idxs), desc=label or f"detect[{algo}]", leave=False)
    elif _VERBOSE:
        print(f"[detect_run] start label={label or algo} jobs={len(fut_to_idxs)}", flush=True)

    t_wait_start = time.time()
    for fut in as_completed(list(fut_to_idxs.keys())):
        idxs = fut_to_idxs[fut]
        vals = fut.result()
        for i, v in zip(idxs, vals):
            zs[i] = float(v)
        if pbar is not None:
            pbar.update(1)
        elif _VERBOSE:
            print(f"[detect_done] shard={len(idxs)} elapsed={time.time() - t_wait_start:.2f}s", flush=True)

    if pbar is not None:
        pbar.close()

    return zs


def shutdown_detect_executors():
    for ex in list(_DETECT_EXECUTORS.values()):
        try:
            ex.shutdown(wait=True, cancel_futures=True)
        except Exception:
            pass
    _DETECT_EXECUTORS.clear()


# -----------------------------
# Main pipeline
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head_dir", type=str, required=True)
    ap.add_argument("--all_dir", type=str, required=True)
    ap.add_argument("--all_weights", type=str, default="")
    ap.add_argument("--deltas", type=str, default="1,2,3,4,5")
    ap.add_argument("--fprs", type=str, default="0.01,0.05,0.10,0.20")
    ap.add_argument("--calib", type=str, default="attacked", choices=["clean", "attacked"])
    ap.add_argument("--edit_ratio", type=float, default=0.02)
    ap.add_argument("--ops", type=str, default="replace,delete,insert")
    ap.add_argument("--attack_generated_only", action="store_true")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--num_workers", type=int, default=max(1, (os.cpu_count() or 8) - 1))
    ap.add_argument("--devices", type=str, default="cuda:0")
    ap.add_argument("--devices_byte", type=str, default="", help="Optional override device list for byte detectors (comma-separated).")
    ap.add_argument("--devices_kgw", type=str, default="", help="Optional override device list for KGW detectors (comma-separated).")
    ap.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--attack_start_method", type=str, default="fork", choices=["fork", "spawn", "forkserver"])
    ap.add_argument("--detect_start_method", type=str, default="spawn", choices=["spawn", "fork", "forkserver"])
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    global _VERBOSE
    _VERBOSE = bool(args.verbose)

    head_dir = Path(args.head_dir)
    all_dir = Path(args.all_dir)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    deltas = [int(x) for x in parse_csv_list(args.deltas, cast=int)]
    fprs = [float(x) for x in parse_csv_list(args.fprs, cast=float)]
    ops = [x.strip() for x in args.ops.split(",") if x.strip()]
    devices = [x.strip() for x in args.devices.split(",") if x.strip()]
    devices_byte = [x.strip() for x in args.devices_byte.split(",") if x.strip()] if args.devices_byte else devices
    devices_kgw = [x.strip() for x in args.devices_kgw.split(",") if x.strip()] if args.devices_kgw else devices
    if not devices_byte:
        raise ValueError("No devices for byte detectors (check --devices or --devices_byte).")
    if not devices_kgw:
        raise ValueError("No devices for kgw detectors (check --devices or --devices_kgw).")

    # locate baseline hf_generate.csv
    hf_path = head_dir / "hf_generate.csv"
    if not hf_path.exists():
        hf_path = all_dir / "hf_generate.csv"
    if not hf_path.exists():
        raise FileNotFoundError("Cannot find hf_generate.csv in head_dir or all_dir")

    df_hf = pd.read_csv(hf_path)
    text_col_hf = pick_text_col(df_hf)
    prompt_col = find_prompt_col(df_hf)

    # NEG stats
    neg_head_path = head_dir / "neg_stats_bytekgw_head.csv"
    neg_all_path = all_dir / "neg_stats_bytekgw_all.csv"
    neg_kgw_path = head_dir / "neg_stats_kgw.csv"
    for p in [neg_head_path, neg_all_path, neg_kgw_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required NEG file: {p}")
    df_neg_head = pd.read_csv(neg_head_path)
    df_neg_all = pd.read_csv(neg_all_path)
    df_neg_kgw = pd.read_csv(neg_kgw_path)

    # alignment key
    if prompt_col is None:
        print("[WARN] hf_generate.csv has no recognizable prompt column; align by row index (_pid).")
        key = "_pid"
        df_hf = add_pid_if_needed(df_hf, key)
        df_neg_head = add_pid_if_needed(df_neg_head, key)
        df_neg_all = add_pid_if_needed(df_neg_all, key)
        df_neg_kgw = add_pid_if_needed(df_neg_kgw, key)
    else:
        key = prompt_col

    def pos_path(run_dir: Path, algo: str, d: int) -> Path:
        if algo == "bytekgw_head":
            return run_dir / f"bytekgw_head_delta{d}.csv"
        if algo == "bytekgw_all":
            p2 = run_dir / f"bytekgw_all_delta{d}_reweighted.csv"
            return p2 if p2.exists() else (run_dir / f"bytekgw_all_delta{d}.csv")
        if algo == "kgw":
            return run_dir / f"kgw_delta{d}.csv"
        raise ValueError(algo)

    # intersection of keys across all files
    common = set(df_hf[key].astype(str).tolist())
    common &= set(df_neg_head[key].astype(str).tolist())
    common &= set(df_neg_all[key].astype(str).tolist())
    common &= set(df_neg_kgw[key].astype(str).tolist())
    for d in deltas:
        for algo, run_dir in [("bytekgw_head", head_dir), ("bytekgw_all", all_dir), ("kgw", head_dir)]:
            p = pos_path(run_dir, algo, d)
            dfp = pd.read_csv(p)
            dfp = add_pid_if_needed(dfp, key)
            common &= set(dfp[key].astype(str).tolist())

    common = sorted(list(common))
    if args.limit and args.limit > 0:
        common = common[: args.limit]
    print(f"[DATA] common prompts = {len(common)}")

    def filt(df: pd.DataFrame) -> pd.DataFrame:
        df = add_pid_if_needed(df, key)
        return df[df[key].astype(str).isin(common)].reset_index(drop=True)

    df_hf = filt(df_hf)
    df_neg_head = filt(df_neg_head)
    df_neg_all = filt(df_neg_all)
    df_neg_kgw = filt(df_neg_kgw)

    # metadata: model + config paths
    meta_path = head_dir / "run_metadata.json"
    if not meta_path.exists():
        meta_path = all_dir / "run_metadata.json"
    meta = read_json(meta_path)
    model_path = meta.get("model", "")
    if not model_path:
        raise ValueError("run_metadata.json missing 'model'")

    byte_cfg_path = meta.get("bytekgw_config", "config/ByteKGWv5.json")
    kgw_cfg_path = meta.get("kgw_config", "config/KGW.json")
    byte_use_prefix_meta = bool(meta.get("use_prefix_bytes_in_prf", False))
    if not Path(byte_cfg_path).exists():
        byte_cfg_path = str(_REPO_ROOT / byte_cfg_path)
    if not Path(kgw_cfg_path).exists():
        kgw_cfg_path = str(_REPO_ROOT / kgw_cfg_path)

    # load allbyte weights (optional)
    byte_weights = None
    if args.all_weights:
        wj = read_json(Path(args.all_weights))
        arr = wj["weights"] if isinstance(wj, dict) and "weights" in wj else wj
        w = np.array(arr, dtype=np.float32)
        byte_weights = (w / (np.linalg.norm(w) + 1e-12)).astype(np.float32)

    # NEG texts/prompts
    neg_texts_clean = df_hf[text_col_hf].astype(str).tolist()
    prompt_col2 = find_prompt_col(df_hf)
    neg_prompts = df_hf[prompt_col2].astype(str).tolist() if prompt_col2 else [None] * len(df_hf)

    # preload tokenizer for fork COW
    global _SHARED_TOKENIZER
    if args.attack_start_method == "fork":
        from transformers import AutoTokenizer
        _SHARED_TOKENIZER = AutoTokenizer.from_pretrained(model_path, use_fast=True)

    # preload detect-side tokenizer for fork COW
    global _DETECT_TOKENIZER
    if args.detect_start_method == "fork":
        from transformers import AutoTokenizer
        _DETECT_TOKENIZER = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        if _DETECT_TOKENIZER.pad_token_id is None and _DETECT_TOKENIZER.eos_token_id is not None:
            _DETECT_TOKENIZER.pad_token = _DETECT_TOKENIZER.eos_token

    print(
        f"[ATTACK] edit_ratio={args.edit_ratio} ops={ops} generated_only={args.attack_generated_only} "
        f"workers={args.num_workers} start={args.attack_start_method}"
    )
    neg_texts_att = parallel_attack(
        neg_texts_clean,
        neg_prompts,
        model_or_tok_path=model_path,
        ops=ops,
        edit_ratio=args.edit_ratio,
        attack_generated_only=args.attack_generated_only,
        seed_base=args.seed,
        num_workers=args.num_workers,
        start_method=args.attack_start_method,
    )

    # helper: load POS texts + clean stat from CSV, then attack
    def load_pos(algo: str, d: int) -> Tuple[List[str], List[Optional[str]]]:
        run_dir = head_dir if algo in ("bytekgw_head", "kgw") else all_dir
        p = pos_path(run_dir, algo, d)
        dfp = filt(add_pid_if_needed(pd.read_csv(p), key))

        tcol = pick_text_col(dfp)
        pcol = find_prompt_col(dfp)
        texts = dfp[tcol].astype(str).tolist()
        prompts = dfp[pcol].astype(str).tolist() if pcol else [None] * len(dfp)
        return texts, prompts

    pos_texts: Dict[Tuple[str, int], List[str]] = {}
    pos_att_text: Dict[Tuple[str, int], List[str]] = {}

    for algo in ["bytekgw_head", "bytekgw_all", "kgw"]:
        for d in deltas:
            texts, prompts = load_pos(algo, d)
            pos_texts[(algo, d)] = texts
            pos_att_text[(algo, d)] = parallel_attack(
                texts,
                prompts,
                model_or_tok_path=model_path,
                ops=ops,
                edit_ratio=args.edit_ratio,
                attack_generated_only=args.attack_generated_only,
                seed_base=args.seed + 100000 + 1000 * d + (0 if algo == "bytekgw_head" else 1 if algo == "bytekgw_all" else 2),
                num_workers=args.num_workers,
                start_method=args.attack_start_method,
            )
            print(f"[ATTACK POS] {algo} delta={d} done ({len(texts)})")

    # evaluate
    rows = []
    print("\n==================== RESULTS (clean vs attacked) ====================\n")

    for algo in ["bytekgw_all", "bytekgw_head", "kgw"]:
        for d in deltas:
            delta = float(d)

            if algo == "bytekgw_head":
                byte_maxpos = 1
                det_algo = "byte"
                w_use = None
            elif algo == "bytekgw_all":
                byte_maxpos = 64
                det_algo = "byte"
                w_use = byte_weights
            else:
                byte_maxpos = None
                det_algo = "kgw"
                w_use = None
            det_devices = devices_byte if det_algo == "byte" else devices_kgw

            # NEG calibration z (always re-detect to avoid relying on stored CSV stats)
            neg_texts_for_calib = neg_texts_clean if args.calib == "clean" else neg_texts_att
            zneg = detect_multi_gpu(
                det_algo,
                neg_texts_for_calib,
                model_path=model_path,
                kgw_cfg_path=kgw_cfg_path,
                byte_cfg_path=byte_cfg_path,
                devices=det_devices,
                workers=args.num_workers,
                dtype=args.dtype,
                delta=delta,
                byte_maxpos=byte_maxpos,
                byte_weights=w_use,
                start_method=args.detect_start_method,
                label=f"NEG {algo} d{d} ({len(neg_texts_for_calib)})",
                byte_use_prefix=byte_use_prefix_meta if det_algo == "byte" else None,
            )

            # clean POS detect
            zpos_clean = detect_multi_gpu(
                det_algo,
                pos_texts[(algo, d)],
                model_path=model_path,
                kgw_cfg_path=kgw_cfg_path,
                byte_cfg_path=byte_cfg_path,
                devices=det_devices,
                workers=args.num_workers,
                dtype=args.dtype,
                delta=delta,
                byte_maxpos=byte_maxpos,
                byte_weights=w_use,
                start_method=args.detect_start_method,
                label=f"CLEAN {algo} d{d} ({len(pos_texts[(algo, d)])})",
                byte_use_prefix=byte_use_prefix_meta if det_algo == "byte" else None,
            )

            # attacked POS redetect
            zpos_att = detect_multi_gpu(
                det_algo,
                pos_att_text[(algo, d)],
                model_path=model_path,
                kgw_cfg_path=kgw_cfg_path,
                byte_cfg_path=byte_cfg_path,
                devices=det_devices,
                workers=args.num_workers,
                dtype=args.dtype,
                delta=delta,
                byte_maxpos=byte_maxpos,
                byte_weights=w_use,
                start_method=args.detect_start_method,
                label=f"ATT {algo} d{d} ({len(pos_att_text[(algo, d)])})",
                byte_use_prefix=byte_use_prefix_meta if det_algo == "byte" else None,
            )

            for tf in fprs:
                thr, achieved = conservative_threshold(zneg, tf)
                rows.append({
                    "algo": algo,
                    "delta": d,
                    "target_fpr": tf,
                    "thr": thr,
                    "achieved_fpr": achieved,
                    "tpr_clean": tpr_at_thr(zpos_clean, thr),
                    "tpr_attacked": tpr_at_thr(zpos_att, thr),
                })

    df_out = pd.DataFrame(rows)
    out_csv = out_dir / "summary.csv"
    df_out.to_csv(out_csv, index=False)
    print(f"[OK] wrote: {out_csv}")

    def pivot(col: str) -> pd.DataFrame:
        return df_out.pivot_table(index=["target_fpr", "delta"], columns="algo", values=col, aggfunc="first")

    print("\n=== TPR (CLEAN, freshly detected) ===")
    print(pivot("tpr_clean").round(3).to_string())
    print("\n=== TPR (ATTACKED, freshly detected) ===")
    print(pivot("tpr_attacked").round(3).to_string())
    print("\n======================================================================\n")


if __name__ == "__main__":
    try:
        main()
    finally:
        shutdown_detect_executors()
