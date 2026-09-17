#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import hashlib
import importlib
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer


# ======================================================================================
# CSV / text helpers
# ======================================================================================
def read_csv_rows(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: str, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def key_prompt(s: str) -> str:
    return " ".join((s or "").strip().split())


def pick_text_fields(row: Dict[str, Any]) -> Tuple[str, str]:
    """
    Return (prompt, full_text).
    Assumes your csv uses prompt_text/full_text like your other scripts.
    """
    prompt = row.get("prompt_text", row.get("prompt", "")) or ""
    full = row.get("full_text", row.get("text", "")) or ""

    if not full:
        # fallback: prompt + completion
        p = row.get("prompt_text", row.get("prompt", "")) or ""
        c = row.get("completion_text", row.get("completion", "")) or ""
        full = p + c

    return prompt, full


def load_prompt_text_map(csv_path: str) -> Dict[str, Dict[str, str]]:
    """
    Return {prompt_key: {"prompt":..., "full":...}}
    """
    rows = read_csv_rows(csv_path)
    m: Dict[str, Dict[str, str]] = {}
    for r in rows:
        p, full = pick_text_fields(r)
        k = key_prompt(p)
        if not k or not full:
            continue
        m[k] = {"prompt": p, "full": full}
    return m


# ======================================================================================
# Deterministic RNG seed
# ======================================================================================
def stable_seed(base: int, *xs) -> int:
    h = hashlib.blake2b(digest_size=8)
    h.update(str(base).encode("utf-8"))
    for x in xs:
        h.update(b"|")
        h.update(str(x).encode("utf-8"))
    return int.from_bytes(h.digest(), "little") & 0x7FFFFFFF


# ======================================================================================
# Threshold selection / metrics
# ======================================================================================
def empirical_rate(zs: List[float], thr: float) -> float:
    if not zs:
        return 0.0
    return sum(1 for z in zs if z > thr) / float(len(zs))


def choose_threshold_at_most(zs: List[float], target_fpr: float) -> Tuple[float, float]:
    """
    Conservative: pick threshold so empirical FPR <= target_fpr and as large as possible.
    """
    if not zs:
        return (float("inf"), 0.0)

    uniq = sorted(set(zs))
    cands = [uniq[-1] + 1e-9] + uniq  # first => FPR=0
    best_thr = cands[0]
    best_fpr = empirical_rate(zs, best_thr)

    for thr in cands:
        fpr = empirical_rate(zs, thr)
        if fpr <= target_fpr:
            if (fpr > best_fpr + 1e-12) or (abs(fpr - best_fpr) <= 1e-12 and thr < best_thr):
                best_thr, best_fpr = thr, fpr

    return best_thr, best_fpr


# ======================================================================================
# Char-edit attack
# ======================================================================================
ASCII_POOL = (
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    " .,;:!?'-\"()[]{}"
)


def random_char(rng: random.Random) -> str:
    return ASCII_POOL[rng.randrange(len(ASCII_POOL))]


def charedit(s: str, edit_ratio: float, ops: List[str], rng: random.Random) -> Tuple[str, int]:
    """
    Apply approx round(len(s) * edit_ratio) edits on whole string.
    ops: replace/delete/insert
    """
    if not s:
        return s, 0

    ops = [o.strip().lower() for o in ops if o.strip()]
    if not ops:
        ops = ["replace", "delete", "insert"]

    n = len(s)
    k = int(round(n * float(edit_ratio)))
    if k <= 0:
        return s, 0

    arr = list(s)
    edits = 0

    for _ in range(k):
        if not arr:
            op = "insert"
        else:
            op = ops[rng.randrange(len(ops))]

        if op == "replace":
            i = rng.randrange(len(arr))
            arr[i] = random_char(rng)
            edits += 1
        elif op == "delete":
            i = rng.randrange(len(arr))
            del arr[i]
            edits += 1
        elif op == "insert":
            i = rng.randrange(len(arr) + 1)
            arr.insert(i, random_char(rng))
            edits += 1
        else:
            # fallback -> replace
            i = rng.randrange(len(arr))
            arr[i] = random_char(rng)
            edits += 1

    return "".join(arr), edits


def attack_text(prompt: str, full_text: str, *, generated_only: bool, edit_ratio: float, ops: List[str], rng: random.Random) -> Tuple[str, int]:
    """
    If generated_only=True, only edit the continuation part (full_text[len(prompt):]) if prefix matches.
    Otherwise edit the whole text.
    """
    if not generated_only:
        return charedit(full_text, edit_ratio, ops, rng)

    p = prompt or ""
    if p and full_text.startswith(p):
        gen = full_text[len(p):]
        gen2, n_ed = charedit(gen, edit_ratio, ops, rng)
        return p + gen2, n_ed

    # fallback: can't align prompt -> attack full
    return charedit(full_text, edit_ratio, ops, rng)


# ======================================================================================
# Algo wrappers
# ======================================================================================
@dataclass
class Algo:
    name: str
    score_fn: Any  # (text)->float


def l2_normalize(w: List[float]) -> List[float]:
    s = float(sum(x * x for x in w))
    if s <= 0:
        return w
    n = math.sqrt(s)
    return [x / n for x in w]


# ======================================================================================
# KGW import (robust)
# ======================================================================================
def _try_import(paths: List[str], attr: str):
    last_err = None
    for p in paths:
        try:
            m = importlib.import_module(p)
            if hasattr(m, attr):
                return getattr(m, attr), p
        except Exception as e:
            last_err = e
    raise ImportError(f"Cannot import {attr} from any of {paths}. Last error: {last_err}")


def build_kgw_score(kgw_config_path: str, model_path: str, device: torch.device) -> Algo:
    """
    Prefer MarkLLM.watermark.kgw.KGW if available.
    Uses dummy model; only tokenizer is needed for detection.
    """
    KGW, used_path = _try_import(
        [
            "MarkLLM.watermark.kgw",
            "MarkLLM.watermark.kgw.kgw",
            "MarkLLM.watermark.kgwV3",
            "MarkLLM.watermark.kgwv3",
        ],
        "KGW",
    )
    TransformersConfig, _ = _try_import(
        [
            "MarkLLM.utils.transformers_config",
        ],
        "TransformersConfig",
    )

    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token

    class _DummyModel:
        def eval(self): return self
        def to(self, *_a, **_k): return self

    dummy = _DummyModel()

    # compatible ctor
    try:
        tf_cfg = TransformersConfig(model=dummy, tokenizer=tok, device=str(device), gen_kwargs={})
    except TypeError:
        try:
            tf_cfg = TransformersConfig(dummy, tok)
        except TypeError:
            tf_cfg = TransformersConfig(model=dummy, tokenizer=tok)

    # patch required attrs (older versions)
    for k, v in {
        "model": dummy,
        "tokenizer": tok,
        "generation_tokenizer": tok,
        "detection_tokenizer": tok,
        "device": str(device),
        "gen_kwargs": {},
        "generation_kwargs": {},
        "vocab_size": int(len(tok)),
    }.items():
        if not hasattr(tf_cfg, k):
            try:
                setattr(tf_cfg, k, v)
            except Exception:
                pass

    kgw = KGW(kgw_config_path, tf_cfg)

    def score(text: str) -> float:
        out = kgw.detect_watermark(text, return_dict=True)
        # common keys: score or z
        if isinstance(out, dict):
            if "score" in out:
                return float(out["score"])
            if "z" in out:
                z = out["z"]
                if torch.is_tensor(z):
                    return float(z.detach().cpu().view(-1)[0].item())
                return float(z)
        raise RuntimeError(f"Unexpected KGW detect output: {type(out)} {out}")

    print(f"[KGW] imported KGW from {used_path}")
    return Algo("kgw", score)


# ======================================================================================
# ByteKGW detector-only (pos_weights supported)
# ======================================================================================
def build_bytekgw_detector_only(
    *,
    model_path: str,
    bytekgw_cfg_path: str,
    device: torch.device,
    max_byte_pos: int,
    use_prefix_bytes_in_prf: bool,
    pos_weights: Optional[List[float]],
    add_special_tokens_call: bool,
    name: str,
) -> Algo:
    """
    Build detector-only pipeline:
      TokenByteVocab + BytePRF + ByteTreeDetector(pos_weights)

    Important alignment:
      - head: add_special_tokens_call=False
      - all : add_special_tokens_call=True
    """
    # ---- imports (robust against PRFConfig name difference) ----
    token_bytes_mod = importlib.import_module("MarkLLM.watermark.bytekgwV5.token_bytes")
    TokenByteVocab = getattr(token_bytes_mod, "TokenByteVocab")

    prf_mod = importlib.import_module("MarkLLM.watermark.bytekgwV5.prf")
    BytePRF = getattr(prf_mod, "BytePRF")
    PRFConfig = getattr(prf_mod, "PRFConfig", None)
    if PRFConfig is None:
        # some repos may name it BytePRFConfig
        PRFConfig = getattr(prf_mod, "BytePRFConfig")

    det_mod = importlib.import_module("MarkLLM.watermark.bytekgwV5.detector")

    # detector class name may vary
    DetectorCls = getattr(det_mod, "ByteKGWv5Detector", None)
    if DetectorCls is None:
        DetectorCls = getattr(det_mod, "ByteTreeDetector", None)
    if DetectorCls is None:
        raise ImportError("Cannot find ByteKGWv5Detector/ByteTreeDetector in MarkLLM.watermark.bytekgwV5.detector")

    DetCfg = getattr(det_mod, "ByteTreeDetectorConfig", None)
    if DetCfg is None:
        raise ImportError("Cannot find ByteTreeDetectorConfig in MarkLLM.watermark.bytekgwV5.detector")

    # ---- tokenizer ----
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token

    # ---- read cfg ----
    cfg = json.load(open(bytekgw_cfg_path, "r", encoding="utf-8"))
    gamma = float(cfg.get("gamma", 0.5))
    hash_key = int(cfg.get("hash_key", 15485863))
    prefix_length = int(cfg.get("prefix_length", 4))
    z_threshold = float(cfg.get("z_threshold", 4.0))

    vocab = TokenByteVocab.from_tokenizer(tok, skip_markers=True).to(device)
    prf = BytePRF(PRFConfig(hash_key=hash_key, gamma=gamma), device=device)

    det_cfg = DetCfg(
        prefix_length=prefix_length,
        gamma=gamma,
        z_threshold=z_threshold,
        max_byte_pos=int(max_byte_pos),
        use_prefix_bytes_in_prf=bool(use_prefix_bytes_in_prf),
        pos_weights=pos_weights,
    )
    det = DetectorCls(prf=prf, vocab=vocab, cfg=det_cfg, device=device)

    @torch.inference_mode()
    def score(text: str) -> float:
        enc = tok(text, return_tensors="pt", add_special_tokens=add_special_tokens_call)
        ids = enc["input_ids"].to(device)
        out = det.detect(ids)
        z = out["z"]
        if torch.is_tensor(z):
            return float(z.detach().cpu().view(-1)[0].item())
        return float(z)

    return Algo(name, score)


# ======================================================================================
# main
# ======================================================================================
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--head_dir", required=True)
    ap.add_argument("--all_dir", required=True)

    ap.add_argument("--kgw_config", default="config/KGW.json")
    ap.add_argument("--bytekgw_config", default="config/ByteKGWv5.json")

    ap.add_argument("--all_weights", required=True, help="fit_pos_weights.json")
    ap.add_argument("--all_csv_suffix", default="", help="e.g. _reweighted")

    ap.add_argument("--deltas", default="1,2,3,4,5")
    ap.add_argument("--fprs", default="0.01,0.05,0.10,0.20")
    ap.add_argument("--thr_mode", default="conservative", choices=["conservative", "at_most"])

    ap.add_argument("--calib", default="attacked", choices=["clean", "attacked"],
                    help="calibrate thresholds using NEG clean or NEG attacked")
    ap.add_argument("--edit_ratio", type=float, default=0.02)
    ap.add_argument("--ops", type=str, default="replace,delete,insert")
    ap.add_argument("--attack_generated_only", action="store_true")

    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=1234)

    ap.add_argument("--no_normalize_weights", action="store_true")

    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    deltas = [int(x.strip()) for x in args.deltas.split(",") if x.strip()]
    fprs = [float(x.strip()) for x in args.fprs.split(",") if x.strip()]
    ops = [x.strip().lower() for x in args.ops.split(",") if x.strip()]

    device = torch.device(args.device)

    head_meta = json.load(open(os.path.join(args.head_dir, "run_metadata.json"), "r", encoding="utf-8"))
    all_meta = json.load(open(os.path.join(args.all_dir, "run_metadata.json"), "r", encoding="utf-8"))
    model_path = head_meta["model"]

    # ------------------------------------------------------------------
    # Alignment rules you verified:
    #   head detect: add_special_tokens=False
    #   all  detect: add_special_tokens=True
    # ------------------------------------------------------------------
    head_addsp_call = False
    all_addsp_call = True

    head_use_prefix = bool(head_meta.get("use_prefix_bytes_in_prf", True))
    all_use_prefix = bool(all_meta.get("use_prefix_bytes_in_prf", True))
    all_max_byte_pos = int(all_meta.get("bytekgw_all_max_byte_pos", all_meta.get("max_byte_pos", 64)))

    # load weights
    wj = json.load(open(args.all_weights, "r", encoding="utf-8"))
    weights = [float(x) for x in (wj.get("weights") or [])]
    if not weights:
        raise ValueError(f"weights empty in {args.all_weights}")

    if len(weights) < all_max_byte_pos:
        weights = weights + [0.0] * (all_max_byte_pos - len(weights))
    weights = weights[:all_max_byte_pos]

    if not args.no_normalize_weights:
        weights = l2_normalize(weights)

    print("=" * 120)
    print(f"[CFG] model={model_path} device={device}")
    print(f"[ALIGN] bytekgw_head add_special_tokens_call={head_addsp_call}")
    print(f"[ALIGN] bytekgw_all  add_special_tokens_call={all_addsp_call}")
    print(f"[BYTE] head use_prefix_bytes_in_prf={head_use_prefix} max_byte_pos=1")
    print(f"[BYTE] all  use_prefix_bytes_in_prf={all_use_prefix}  max_byte_pos={all_max_byte_pos} weights_len={len(weights)}")
    print(f"[ATTACK] edit_ratio={args.edit_ratio} ops={ops} generated_only={args.attack_generated_only} seed={args.seed}")
    print(f"[EVAL] deltas={deltas} fprs={fprs} thr_mode={args.thr_mode} calib={args.calib}")
    print("=" * 120)

    # Build algorithms
    algo_kgw = build_kgw_score(args.kgw_config, model_path, device)

    algo_head = build_bytekgw_detector_only(
        model_path=model_path,
        bytekgw_cfg_path=args.bytekgw_config,
        device=device,
        max_byte_pos=1,
        use_prefix_bytes_in_prf=head_use_prefix,
        pos_weights=None,
        add_special_tokens_call=head_addsp_call,
        name="bytekgw_head",
    )

    algo_all = build_bytekgw_detector_only(
        model_path=model_path,
        bytekgw_cfg_path=args.bytekgw_config,
        device=device,
        max_byte_pos=all_max_byte_pos,
        use_prefix_bytes_in_prf=all_use_prefix,
        pos_weights=weights,
        add_special_tokens_call=all_addsp_call,
        name="bytekgw_all_w",
    )

    algos = [algo_all, algo_head, algo_kgw]

    # Load NEG (hf_generate.csv)
    hf_path = os.path.join(args.head_dir, "hf_generate.csv")
    if not os.path.exists(hf_path):
        hf_path = os.path.join(args.all_dir, "hf_generate.csv")
    if not os.path.exists(hf_path):
        raise FileNotFoundError("Cannot find hf_generate.csv in head_dir or all_dir")
    neg_map = load_prompt_text_map(hf_path)

    # Load POS per algo & delta
    pos_kgw = {d: load_prompt_text_map(os.path.join(args.head_dir, f"kgw_delta{d}.csv")) for d in deltas}
    pos_head = {d: load_prompt_text_map(os.path.join(args.head_dir, f"bytekgw_head_delta{d}.csv")) for d in deltas}
    pos_all = {d: load_prompt_text_map(os.path.join(args.all_dir, f"bytekgw_all_delta{d}{args.all_csv_suffix}.csv")) for d in deltas}

    # Common prompts: NEG ∩ all POS (all algos) ∩ all deltas
    common = set(neg_map.keys())
    for d in deltas:
        common &= set(pos_kgw[d].keys())
        common &= set(pos_head[d].keys())
        common &= set(pos_all[d].keys())
    common = sorted(common)
    if not common:
        raise RuntimeError("No common prompts across NEG and all POS files.")
    print(f"[DATA] common prompts = {len(common)}")

    # Prepare NEG clean/attacked
    neg_clean = [neg_map[k]["full"] for k in common]
    neg_attacked = []
    neg_edits = []
    for k in common:
        rng = random.Random(stable_seed(args.seed, "NEG", k))
        attacked, n_ed = attack_text(
            neg_map[k]["prompt"], neg_map[k]["full"],
            generated_only=args.attack_generated_only,
            edit_ratio=args.edit_ratio,
            ops=ops,
            rng=rng,
        )
        neg_attacked.append(attacked)
        neg_edits.append(n_ed)
    mean_neg_ed = float(sum(neg_edits)) / float(len(neg_edits))

    calib_texts = neg_attacked if args.calib == "attacked" else neg_clean

    # Calibrate thresholds per algo on chosen NEG set
    thresholds: Dict[str, Dict[float, Tuple[float, float]]] = {a.name: {} for a in algos}
    neg_z_cache: Dict[str, List[float]] = {}

    for a in algos:
        zs = []
        for i, txt in enumerate(calib_texts, 1):
            zs.append(float(a.score_fn(txt)))
            if i % 50 == 0 or i == len(calib_texts):
                print(f"[NEG score] {a.name} {i}/{len(calib_texts)}")
        neg_z_cache[a.name] = zs

        for fpr in fprs:
            thr, ach = choose_threshold_at_most(zs, fpr)
            thresholds[a.name][fpr] = (thr, ach)

    # Evaluate POS clean & attacked
    rows_out: List[Dict[str, Any]] = []

    def eval_one_algo_on_delta(a: Algo, d: int, pos_map: Dict[str, Dict[str, str]]) -> Tuple[List[float], List[float], float]:
        pos_clean = [pos_map[k]["full"] for k in common]
        pos_att = []
        edits = 0.0
        for k in common:
            rng = random.Random(stable_seed(args.seed, a.name, "POS", d, k))
            attacked, n_ed = attack_text(
                pos_map[k]["prompt"], pos_map[k]["full"],
                generated_only=args.attack_generated_only,
                edit_ratio=args.edit_ratio,
                ops=ops,
                rng=rng,
            )
            pos_att.append(attacked)
            edits += float(n_ed)
        edits /= float(len(common))

        z_clean = []
        z_att = []

        for i, txt in enumerate(pos_clean, 1):
            z_clean.append(float(a.score_fn(txt)))
            if i % 50 == 0 or i == len(pos_clean):
                print(f"[POS clean] {a.name} d={d} {i}/{len(pos_clean)}")
        for i, txt in enumerate(pos_att, 1):
            z_att.append(float(a.score_fn(txt)))
            if i % 50 == 0 or i == len(pos_att):
                print(f"[POS att ] {a.name} d={d} {i}/{len(pos_att)}")

        return z_clean, z_att, edits

    for d in deltas:
        zc_all, za_all, ed_all = eval_one_algo_on_delta(algo_all, d, pos_all[d])
        zc_head, za_head, ed_head = eval_one_algo_on_delta(algo_head, d, pos_head[d])
        zc_kgw, za_kgw, ed_kgw = eval_one_algo_on_delta(algo_kgw, d, pos_kgw[d])

        per = {
            "bytekgw_all_w": (zc_all, za_all, ed_all),
            "bytekgw_head": (zc_head, za_head, ed_head),
            "kgw": (zc_kgw, za_kgw, ed_kgw),
        }

        for algo_name, (zc, za, ed) in per.items():
            for fpr in fprs:
                thr, ach = thresholds[algo_name][fpr]
                tpr_clean = empirical_rate(zc, thr)
                tpr_att = empirical_rate(za, thr)
                rows_out.append({
                    "algo": algo_name,
                    "delta": d,
                    "target_fpr": fpr,
                    "calib": args.calib,
                    "threshold": thr,
                    "neg_fpr_emp": ach,
                    "tpr_clean": tpr_clean,
                    "tpr_attacked": tpr_att,
                    "edit_ratio": args.edit_ratio,
                    "ops": ",".join(ops),
                    "attack_generated_only": bool(args.attack_generated_only),
                    "mean_neg_char_edits": mean_neg_ed,
                    "mean_pos_char_edits": ed,
                    "n_prompts": len(common),
                })

    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, "attack_char_3alg_summary.csv")
    write_csv(out_csv, list(rows_out[0].keys()), rows_out)
    print(f"[OK] wrote: {out_csv}")

    # -------------------------------------------------------------
    # Print pivots: clean / attacked / drop
    # -------------------------------------------------------------
    def build_pivot(rows: List[Dict[str, Any]], field: str) -> Dict[Tuple[float, int], Dict[str, float]]:
        pv: Dict[Tuple[float, int], Dict[str, float]] = {}
        for r in rows:
            key = (float(r["target_fpr"]), int(r["delta"]))
            pv.setdefault(key, {})[str(r["algo"])] = float(r[field])
        return pv

    pivot_clean = build_pivot(rows_out, "tpr_clean")
    pivot_att = build_pivot(rows_out, "tpr_attacked")

    def print_pivot(title: str, pv: Dict[Tuple[float, int], Dict[str, float]]) -> None:
        print(f"\n=== {title} (rows: fpr,delta; cols: algo) ===")
        for fpr in fprs:
            print(f"\n[target_fpr={fpr}]")
            for d in deltas:
                cols = pv.get((float(fpr), int(d)), {})
                print(
                    f"  delta={d}: "
                    f"bytekgw_all_w={cols.get('bytekgw_all_w', float('nan')):.3f}  "
                    f"bytekgw_head={cols.get('bytekgw_head', float('nan')):.3f}  "
                    f"kgw={cols.get('kgw', float('nan')):.3f}"
                )

    print_pivot("Clean TPR pivot", pivot_clean)
    print_pivot("Attacked TPR pivot", pivot_att)

    print("\n=== TPR drop (attacked - clean) ===")
    for fpr in fprs:
        print(f"\n[target_fpr={fpr}]")
        for d in deltas:
            c = pivot_clean.get((float(fpr), int(d)), {})
            a = pivot_att.get((float(fpr), int(d)), {})
            def dv(algo: str) -> float:
                if algo not in a or algo not in c:
                    return float("nan")
                return a[algo] - c[algo]
            print(
                f"  delta={d}: "
                f"bytekgw_all_w={dv('bytekgw_all_w'):.3f}  "
                f"bytekgw_head={dv('bytekgw_head'):.3f}  "
                f"kgw={dv('kgw'):.3f}"
            )

    # Print achieved FPR summary
    print("\n=== NEG calibration achieved_fpr (per algo) ===")
    for a in algos:
        print(f"\n[{a.name}] calib={args.calib}")
        for fpr in fprs:
            thr, ach = thresholds[a.name][fpr]
            print(f"  target_fpr={fpr:<5} thr={thr:.6g} achieved_fpr={ach:.3f}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
