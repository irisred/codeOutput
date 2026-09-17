#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trace ByteKGWv6 robustness under MIXED CHARACTER-LEVEL attacks (char-only),
and report three core robustness signals:

(1) PRF stability: green_flip_rate (uid fixed; fp_clean vs fp_attack)
(2) n-bytes aggregation stability: uid_match_rate_aligned (uid_clean vs uid_attack at aligned offsets)
(3) tokenization disruption: extra_token_count_in_traced (attack tokens whose start offset has no match in clean)

Also reports token_id_match_rate_aligned as a stronger baseline vs uid_match.

Multiprocessing (Python 3.12 safe): worker function is defined at module top-level.

Outputs:
  - out_csv: per-row per-ratio metrics
  - out_mean.csv: mean over rows grouped by attack_ratio
  - out_mean_by_style.csv: mean over rows grouped by (attack_ratio, attack_style_used)

Requires a FAST tokenizer (offset_mapping).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer
from tqdm import tqdm

# ------------------------------------
# imports from repo (robust to path)
# ------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from MarkLLM.watermark.bytekgwV6.token_bytes import TokenByteVocabV6  # type: ignore
    from MarkLLM.watermark.bytekgwV6.prf import RobustPartitioner  # type: ignore
except ModuleNotFoundError:
    from watermark.bytekgwV6.token_bytes import TokenByteVocabV6  # type: ignore
    from watermark.bytekgwV6.prf import RobustPartitioner  # type: ignore


# ------------------------------------
# helpers
# ------------------------------------
ASCII_POOL = (
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    " .,;:!?'-\"()[]{}"
)

ZWSP = "\u200b"  # zero-width space
ZWJ  = "\u200d"  # zero-width joiner

# small homoglyph map (ASCII -> visually similar Unicode)
HOMO = {
    "a": "а", "c": "с", "e": "е", "i": "і", "o": "о", "p": "р", "x": "х", "y": "у",
    "A": "Α", "B": "Β", "E": "Ε", "H": "Η", "I": "Ι", "K": "Κ", "M": "Μ", "N": "Ν",
    "O": "Ο", "P": "Ρ", "T": "Τ", "X": "Χ",
}

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _to_bytes(key) -> bytes:
    if isinstance(key, bytes):
        return key
    if isinstance(key, int):
        return int(key).to_bytes(8, "little", signed=False)
    if isinstance(key, str):
        s = key.strip()
        try:
            if s.startswith("0x"):
                return bytes.fromhex(s[2:])
            return bytes.fromhex(s)
        except Exception:
            return s.encode("utf-8")
    raise TypeError(f"Unsupported key type: {type(key)}")

def tail_str(s: str, n: int) -> str:
    return s[-n:] if n > 0 else ""

def tokenize_with_offsets(tok, text: str, add_special_tokens: bool):
    enc = tok(
        text,
        return_tensors="pt",
        add_special_tokens=add_special_tokens,
        return_offsets_mapping=True,
    )
    ids = enc["input_ids"][0]
    offsets = None
    if "offset_mapping" in enc:
        try:
            offsets = [tuple(map(int, x)) for x in enc["offset_mapping"][0].tolist()]
        except Exception:
            offsets = None
    return ids, offsets

def build_v6_partitioner(v6_cfg: Dict[str, Any]) -> RobustPartitioner:
    return RobustPartitioner(
        master_key=_to_bytes(v6_cfg.get("hash_key", 15485863)),
        m_bits=int(v6_cfg.get("m_bits", 256)),
        target_anchors=int(v6_cfg.get("target_anchors", 96)),
        k_choices=tuple(v6_cfg.get("k_choices", [4, 5, 6])),
        normalize_whitespace=bool(v6_cfg.get("normalize_whitespace", True)),
    )

# ------------------------------------
# char attack styles (char-only mix)
# ------------------------------------
def _rand_ascii(rng: np.random.RandomState) -> str:
    return ASCII_POOL[int(rng.randint(0, len(ASCII_POOL)))]

def _rand_zw(rng: np.random.RandomState) -> str:
    return ZWSP if float(rng.rand()) < 0.5 else ZWJ

def _pick_positions(rng: np.random.RandomState, n: int, k: int) -> List[int]:
    k = max(1, min(k, n))
    idxs = rng.choice(n, size=k, replace=False)
    return sorted(int(i) for i in idxs)

def attack_replace_X(segment: str, rng: np.random.RandomState, ratio: float) -> Tuple[str, Dict[str, int]]:
    n = len(segment)
    k = int(round(n * ratio))
    if n <= 0 or k <= 0:
        return segment, {"ops": 0, "replace": 0, "delete": 0, "insert": 0}
    idxs = _pick_positions(rng, n, k)
    arr = list(segment)
    for i in idxs:
        arr[i] = "X"
    return "".join(arr), {"ops": len(idxs), "replace": len(idxs), "delete": 0, "insert": 0}

def attack_ascii_replace(segment: str, rng: np.random.RandomState, ratio: float) -> Tuple[str, Dict[str, int]]:
    n = len(segment)
    k = int(round(n * ratio))
    if n <= 0 or k <= 0:
        return segment, {"ops": 0, "replace": 0, "delete": 0, "insert": 0}
    idxs = _pick_positions(rng, n, k)
    arr = list(segment)
    for i in idxs:
        arr[i] = _rand_ascii(rng)
    return "".join(arr), {"ops": len(idxs), "replace": len(idxs), "delete": 0, "insert": 0}

def attack_homo_replace(segment: str, rng: np.random.RandomState, ratio: float) -> Tuple[str, Dict[str, int]]:
    n = len(segment)
    k = int(round(n * ratio))
    if n <= 0 or k <= 0:
        return segment, {"ops": 0, "replace": 0, "delete": 0, "insert": 0}
    idxs = _pick_positions(rng, n, k)
    arr = list(segment)
    rep = 0
    for i in idxs:
        ch = arr[i]
        arr[i] = HOMO.get(ch, _rand_ascii(rng))
        rep += 1
    return "".join(arr), {"ops": rep, "replace": rep, "delete": 0, "insert": 0}

def attack_zw_insert(segment: str, rng: np.random.RandomState, ratio: float) -> Tuple[str, Dict[str, int]]:
    n = len(segment)
    k = int(round(n * ratio))
    if n <= 0 or k <= 0:
        return segment, {"ops": 0, "replace": 0, "delete": 0, "insert": 0}
    k = max(1, k)
    arr = list(segment)
    for _ in range(k):
        i = int(rng.randint(0, len(arr) + 1))
        arr.insert(i, _rand_zw(rng))
    return "".join(arr), {"ops": k, "replace": 0, "delete": 0, "insert": k}

def attack_mixed(segment: str, rng: np.random.RandomState, ratio: float) -> Tuple[str, Dict[str, int]]:
    """
    mixed replace/delete/insert:
      - replace: 70% homoglyph if possible else ascii
      - insert : 60% zw else ascii
      - delete : remove one char
    """
    n0 = len(segment)
    k = int(round(n0 * ratio))
    if n0 <= 0 or k <= 0:
        return segment, {"ops": 0, "replace": 0, "delete": 0, "insert": 0}

    arr = list(segment)
    cnt = {"ops": 0, "replace": 0, "delete": 0, "insert": 0}
    ops = ["replace", "delete", "insert"]

    for _ in range(k):
        op = ops[int(rng.randint(0, len(ops)))]
        if op == "replace":
            if not arr:
                continue
            i = int(rng.randint(0, len(arr)))
            ch = arr[i]
            if float(rng.rand()) < 0.7 and ch in HOMO:
                arr[i] = HOMO[ch]
            else:
                arr[i] = _rand_ascii(rng)
            cnt["replace"] += 1
            cnt["ops"] += 1

        elif op == "delete":
            if not arr:
                continue
            i = int(rng.randint(0, len(arr)))
            arr.pop(i)
            cnt["delete"] += 1
            cnt["ops"] += 1

        elif op == "insert":
            i = int(rng.randint(0, len(arr) + 1))
            if float(rng.rand()) < 0.6:
                arr.insert(i, _rand_zw(rng))
            else:
                arr.insert(i, _rand_ascii(rng))
            cnt["insert"] += 1
            cnt["ops"] += 1

    return "".join(arr), cnt

ATTACK_FUNCS = {
    "mixed": attack_mixed,
    "homo_replace": attack_homo_replace,
    "zw_insert": attack_zw_insert,
    "ascii_replace": attack_ascii_replace,
    "x_replace": attack_replace_X,
}

def apply_mixchar_attack(
    full_text: str,
    prompt: str,
    ratio: float,
    seed: int,
    styles: List[str],
    weights: Optional[List[float]],
) -> Dict[str, Any]:
    """
    Returns attacked_full_text + style + counts + realized rates.
    edit_ratio is based on attacked segment's original char length.
    """
    if ratio <= 0.0:
        return {
            "attacked_full_text": full_text,
            "attack_style_used": "none",
            "char_ops": 0,
            "char_replace": 0,
            "char_delete": 0,
            "char_insert": 0,
            "segment_char_len": 0,
            "actual_char_edit_rate": 0.0,
        }

    rng = np.random.RandomState(seed)

    # choose style
    if weights and len(weights) == len(styles):
        p = np.array(weights, dtype=float)
        p = p / p.sum()
        style = str(rng.choice(styles, p=p))
    else:
        style = str(rng.choice(styles))

    if style not in ATTACK_FUNCS:
        style = "mixed"

    # choose segment: continuation if possible
    use_cont = bool(prompt) and full_text.startswith(prompt)
    segment = full_text[len(prompt):] if use_cont else full_text
    L0 = len(segment)

    attacked_seg, cnt = ATTACK_FUNCS[style](segment, rng, ratio)

    attacked_full = (prompt + attacked_seg) if use_cont else attacked_seg
    ops = int(cnt.get("ops", 0))
    actual_rate = float(ops / max(1, L0)) if L0 > 0 else 0.0

    return {
        "attacked_full_text": attacked_full,
        "attack_style_used": style,
        "char_ops": ops,
        "char_replace": int(cnt.get("replace", 0)),
        "char_delete": int(cnt.get("delete", 0)),
        "char_insert": int(cnt.get("insert", 0)),
        "segment_char_len": int(L0),
        "actual_char_edit_rate": float(actual_rate),
    }

# ------------------------------------
# core (single row)
# ------------------------------------
def compute_row_summary(
    row: pd.Series,
    tok,
    vocab: TokenByteVocabV6,
    partitioner: RobustPartitioner,
    device: torch.device,
    seed_window_chars: int,
    n_bytes: int,
    attack_ratio: float,
    seed: int,
    max_tokens: int,
    stride: int,
    styles: List[str],
    weights: Optional[List[float]],
) -> Dict[str, Any]:
    full: str = row["full_text"]
    prompt: str = row.get("prompt_text", "") or ""

    atk = apply_mixchar_attack(
        full_text=full,
        prompt=prompt,
        ratio=attack_ratio,
        seed=seed,
        styles=styles,
        weights=weights,
    )
    attacked = atk["attacked_full_text"]

    ids_c, offs_c = tokenize_with_offsets(tok, full, add_special_tokens=True)
    ids_a, offs_a = tokenize_with_offsets(tok, attacked, add_special_tokens=True)
    if offs_c is None or offs_a is None:
        raise RuntimeError("Tokenizer offsets unavailable; use a fast tokenizer.")

    firstn_ids = vocab.first_n_id(device, n_bytes)  # [V]

    # start offset -> index mapping for clean
    clean_start2i: Dict[int, int] = {}
    for i, (s, _e) in enumerate(offs_c):
        clean_start2i[s] = i

    prompt_len_chars = len(prompt)

    # trace stats
    tail_eq_cnt = 0
    tail_den = 0

    flip_cnt = 0
    flip_den = 0

    extra_tok_cnt = 0
    traced = 0

    # NEW: alignment + uid match stats (what you asked for)
    aligned_cnt = 0
    uid_match_cnt = 0
    tid_match_cnt = 0

    for pos_a, (s_a, _e_a) in enumerate(offs_a):
        if traced >= max_tokens:
            break
        if s_a < prompt_len_chars:
            continue
        if (pos_a % max(1, stride)) != 0:
            continue

        traced += 1
        tok_id_a = int(ids_a[pos_a].item())
        uid_a = int(firstn_ids[tok_id_a].item())

        prefix_end = s_a
        prefix_a = attacked[:prefix_end]
        prefix_c = full[:min(prefix_end, len(full))]

        fp_a = partitioner.fingerprint(prefix_a, seed_window_chars)
        fp_c = partitioner.fingerprint(prefix_c, seed_window_chars)

        # tail match (attack touches fingerprint window)
        tail_a = tail_str(prefix_a, seed_window_chars)
        tail_c = tail_str(prefix_c, seed_window_chars)
        tail_eq_cnt += int(tail_a == tail_c)
        tail_den += 1

        # PRF stability: same uid, different fp
        is_green_a = int(partitioner.is_green(uid_a, fp_a))
        is_green_uid_under_clean_fp = int(partitioner.is_green(uid_a, fp_c))
        flip_cnt += int(is_green_a != is_green_uid_under_clean_fp)
        flip_den += 1

        # alignment / extra tokens + uid match (aggregation stability)
        pos_c = clean_start2i.get(s_a, -1)
        if pos_c == -1:
            extra_tok_cnt += 1
        else:
            aligned_cnt += 1
            tok_id_c = int(ids_c[pos_c].item())
            uid_c = int(firstn_ids[tok_id_c].item())

            if tok_id_c == tok_id_a:
                tid_match_cnt += 1
            if uid_c == uid_a:
                uid_match_cnt += 1

    tail_match_rate = float(tail_eq_cnt / max(1, tail_den))
    green_flip_rate = float(flip_cnt / max(1, flip_den))

    uid_match_rate = float(uid_match_cnt / max(1, aligned_cnt))
    tid_match_rate = float(tid_match_cnt / max(1, aligned_cnt))
    aligned_rate = float(aligned_cnt / max(1, traced))

    return {
        "row_idx": int(row.name),
        "attack_ratio": float(attack_ratio),
        "seed_used": int(seed),
        "attack_style_used": str(atk["attack_style_used"]),

        "seed_window_chars": int(seed_window_chars),
        "n_bytes": int(n_bytes),

        "clean_tokens": int(ids_c.numel()),
        "attacked_tokens": int(ids_a.numel()),

        # disruption + trace coverage
        "traced_tokens": int(traced),
        "extra_token_count_in_traced": int(extra_tok_cnt),

        # PRF-related
        "tail_match_rate_over_traced": float(tail_match_rate),
        "green_flip_rate_uid_cleanfp_vs_attackfp": float(green_flip_rate),

        # aggregation-related (NEW)
        "aligned_tokens": int(aligned_cnt),
        "aligned_rate_over_traced": float(aligned_rate),
        "uid_match_rate_aligned": float(uid_match_rate),
        "token_id_match_rate_aligned": float(tid_match_rate),

        # attack strength bookkeeping
        "char_ops": int(atk["char_ops"]),
        "char_replace": int(atk["char_replace"]),
        "char_delete": int(atk["char_delete"]),
        "char_insert": int(atk["char_insert"]),
        "segment_char_len": int(atk["segment_char_len"]),
        "actual_char_edit_rate": float(atk["actual_char_edit_rate"]),
    }


# ------------------------------------
# multiprocessing worker (TOP-LEVEL; pickleable)
# ------------------------------------
def process_chunk_worker(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    run_meta = load_json(payload["run_meta_path"])
    v6_cfg = load_json(payload["v6_config_path"])

    tok = AutoTokenizer.from_pretrained(run_meta["model"])
    device = torch.device(payload["device_str"])

    partitioner = build_v6_partitioner(v6_cfg)
    vocab = TokenByteVocabV6.from_tokenizer(tok, skip_markers=True).to(device)

    seed_window_chars = int(v6_cfg.get("seed_window_chars", 18))
    n_bytes = int(v6_cfg.get("n_bytes", 3))

    df = pd.read_csv(payload["input_csv"])

    ratios: List[float] = payload["ratios"]
    chunk_indices: List[int] = payload["chunk_indices"]
    attack_seed0 = int(payload["attack_seed"])
    max_tokens = int(payload["max_tokens"])
    stride = int(payload["stride"])
    styles: List[str] = payload["mix_styles"]
    weights: Optional[List[float]] = payload["mix_weights"]

    out: List[Dict[str, Any]] = []
    for r in ratios:
        for row_idx in chunk_indices:
            row = df.iloc[row_idx]
            seed = int(attack_seed0 + row_idx + int(float(r) * 1_000_000))
            out.append(
                compute_row_summary(
                    row=row,
                    tok=tok,
                    vocab=vocab,
                    partitioner=partitioner,
                    device=device,
                    seed_window_chars=seed_window_chars,
                    n_bytes=n_bytes,
                    attack_ratio=float(r),
                    seed=seed,
                    max_tokens=max_tokens,
                    stride=stride,
                    styles=styles,
                    weights=weights,
                )
            )
    return out


# ------------------------------------
# main
# ------------------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_meta", required=True)
    ap.add_argument("--v6_config", required=True)
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--attack_ratios", default="0,0.01,0.02,0.05,0.10")
    ap.add_argument("--attack_seed", type=int, default=0)
    ap.add_argument("--indices", default="all", help="comma-separated row indices or 'all'")
    ap.add_argument("--max_tokens", type=int, default=200)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--num_workers", type=int, default=1)

    ap.add_argument(
        "--mix_styles",
        default="mixed,homo_replace,zw_insert,ascii_replace,x_replace",
        help="comma-separated char styles: mixed,homo_replace,zw_insert,ascii_replace,x_replace",
    )
    ap.add_argument(
        "--mix_weights",
        default="",
        help="optional comma-separated weights matching mix_styles (e.g. 0.4,0.2,0.2,0.1,0.1)",
    )
    ap.add_argument("--out_csv", default="trace_v6_attack_grid_mixchar.csv")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    use_gpu = args.device.startswith("cuda") and torch.cuda.is_available()
    if use_gpu and args.num_workers > 1:
        raise ValueError("Multi-worker is CPU-only. Use --device cpu or --num_workers 1 for GPU.")
    device = torch.device(args.device if use_gpu else "cpu")

    run_meta = load_json(args.run_meta)
    v6_cfg = load_json(args.v6_config)
    tok = AutoTokenizer.from_pretrained(run_meta["model"])

    seed_window_chars = int(v6_cfg.get("seed_window_chars", 18))
    n_bytes = int(v6_cfg.get("n_bytes", 3))

    partitioner = build_v6_partitioner(v6_cfg)
    vocab = TokenByteVocabV6.from_tokenizer(tok, skip_markers=True).to(device)

    df = pd.read_csv(args.input_csv)
    if args.indices.strip().lower() == "all":
        wanted = list(range(len(df)))
    else:
        wanted = [int(x) for x in args.indices.split(",") if x.strip()]

    ratios = [float(x) for x in args.attack_ratios.split(",") if x.strip()]
    styles = [s.strip() for s in args.mix_styles.split(",") if s.strip()]
    weights = None
    if args.mix_weights.strip():
        weights = [float(x) for x in args.mix_weights.split(",") if x.strip()]
        if len(weights) != len(styles):
            raise ValueError("mix_weights length must match mix_styles length")

    rows_out: List[Dict[str, Any]] = []

    if args.num_workers == 1:
        for r in ratios:
            print(f"[attack_ratio={r}] processing {len(wanted)} rows...")
            for row_idx in tqdm(wanted, desc=f"r={r}", leave=False):
                row = df.iloc[row_idx]
                seed = int(args.attack_seed + row_idx + int(r * 1_000_000))
                rows_out.append(
                    compute_row_summary(
                        row=row,
                        tok=tok,
                        vocab=vocab,
                        partitioner=partitioner,
                        device=device,
                        seed_window_chars=seed_window_chars,
                        n_bytes=n_bytes,
                        attack_ratio=r,
                        seed=seed,
                        max_tokens=args.max_tokens,
                        stride=args.stride,
                        styles=styles,
                        weights=weights,
                    )
                )
    else:
        chunks = np.array_split(wanted, args.num_workers)
        payloads = []
        for chunk in chunks:
            payloads.append({
                "run_meta_path": args.run_meta,
                "v6_config_path": args.v6_config,
                "input_csv": args.input_csv,
                "device_str": "cpu",
                "ratios": ratios,
                "chunk_indices": chunk.tolist(),
                "attack_seed": args.attack_seed,
                "max_tokens": args.max_tokens,
                "stride": args.stride,
                "mix_styles": styles,
                "mix_weights": weights,
            })

        with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as ex:
            futures = [ex.submit(process_chunk_worker, p) for p in payloads]
            for fut in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="chunks"):
                rows_out.extend(fut.result())

    out_df = pd.DataFrame(rows_out)
    out_path = Path(args.out_csv)
    out_df.to_csv(out_path, index=False)

    mean_df = (
        out_df.groupby("attack_ratio")
        .agg(
            mean_tail_match=("tail_match_rate_over_traced", "mean"),
            mean_green_flip=("green_flip_rate_uid_cleanfp_vs_attackfp", "mean"),

            mean_uid_match=("uid_match_rate_aligned", "mean"),
            mean_tid_match=("token_id_match_rate_aligned", "mean"),
            mean_aligned_rate=("aligned_rate_over_traced", "mean"),

            mean_extra_tokens=("extra_token_count_in_traced", "mean"),
            mean_traced=("traced_tokens", "mean"),

            mean_char_ops=("char_ops", "mean"),
            mean_actual_char_rate=("actual_char_edit_rate", "mean"),
        )
        .reset_index()
    )
    mean_path = out_path.with_name(out_path.stem + "_mean.csv")
    mean_df.to_csv(mean_path, index=False)

    mean_style_df = (
        out_df.groupby(["attack_ratio", "attack_style_used"])
        .agg(
            mean_tail_match=("tail_match_rate_over_traced", "mean"),
            mean_green_flip=("green_flip_rate_uid_cleanfp_vs_attackfp", "mean"),

            mean_uid_match=("uid_match_rate_aligned", "mean"),
            mean_tid_match=("token_id_match_rate_aligned", "mean"),
            mean_aligned_rate=("aligned_rate_over_traced", "mean"),

            mean_extra_tokens=("extra_token_count_in_traced", "mean"),
            mean_traced=("traced_tokens", "mean"),

            mean_actual_char_rate=("actual_char_edit_rate", "mean"),
            n=("row_idx", "count"),
        )
        .reset_index()
    )
    mean_style_path = out_path.with_name(out_path.stem + "_mean_by_style.csv")
    mean_style_df.to_csv(mean_style_path, index=False)

    print(f"wrote per-row to {out_path} (rows={len(out_df)})")
    print(f"wrote mean-by-ratio to {mean_path}")
    print(f"wrote mean-by-style to {mean_style_path}")


if __name__ == "__main__":
    main()
