#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import math
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# -----------------------------
# Small utils
# -----------------------------
def read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return list(r)


def write_csv(path: str, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def norm_prompt(s: str) -> str:
    return " ".join((s or "").strip().split())


def as_float(x: Any) -> float:
    return float(x)


def mean_std(xs: List[float]) -> Tuple[float, float]:
    if not xs:
        return float("nan"), float("nan")
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, 0.0
    v = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, math.sqrt(v)


def summary(xs: List[float]) -> str:
    if not xs:
        return "empty"
    s = sorted(xs)
    n = len(s)
    med = s[n // 2]
    return f"min={s[0]:.4f} med={med:.4f} max={s[-1]:.4f} uniq={len(set(xs))}"


def rate_above(xs: List[float], thr: float) -> float:
    if not xs:
        return 0.0
    return sum(1 for x in xs if x > thr) / float(len(xs))


def choose_threshold(zs: List[float], target_fpr: float, mode: str) -> float:
    """
    predict wm if stat > thr
    mode:
      - at_least: smallest thr with achieved_fpr >= target_fpr (practical when FPR is discrete)
      - conservative: prefer achieved_fpr <= target_fpr when tied
    """
    if not zs:
        return 0.0
    uniq = sorted(set(zs))
    cands = [uniq[-1] + 1e-9] + uniq + [uniq[0] - 1e-9]

    def fpr(thr: float) -> float:
        return sum(1 for z in zs if z > thr) / float(len(zs))

    if mode == "at_least":
        best = None
        for thr in cands:
            val = fpr(thr)
            if val + 1e-12 >= target_fpr:
                overshoot = val - target_fpr
                key = (overshoot, -thr)
                if best is None or key < best[0]:
                    best = (key, thr, val)
        if best is not None:
            return float(best[1])
        return float(min(cands))

    # conservative
    best_thr = cands[0]
    best_err = 1e18
    best_fpr = fpr(best_thr)
    for thr in cands:
        val = fpr(thr)
        err = abs(val - target_fpr)
        better = (err < best_err - 1e-12) or (
            abs(err - best_err) < 1e-12 and val <= target_fpr and best_fpr > target_fpr
        )
        if better:
            best_err, best_thr, best_fpr = err, thr, val
    return float(best_thr)


def parse_dtype(name: str) -> torch.dtype:
    name = name.lower().strip()
    if name in ("float16", "fp16", "half"):
        return torch.float16
    if name in ("bfloat16", "bf16"):
        return torch.bfloat16
    if name in ("float32", "fp32"):
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def ensure_pad_token(tok) -> None:
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token


def _tensor_to_float(x: Any) -> Optional[float]:
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if torch.is_tensor(x) and x.numel() == 1:
        return float(x.detach().cpu().item())
    return None


def extract_stat(det_out: Dict[str, Any]) -> float:
    # ByteKGWv5 -> z, KGW -> score
    if "z" in det_out:
        v = _tensor_to_float(det_out["z"])
        if v is not None:
            return v
    if "score" in det_out:
        v = _tensor_to_float(det_out["score"])
        if v is not None:
            return v
    for _, vv in det_out.items():
        v = _tensor_to_float(vv)
        if v is not None:
            return v
    raise KeyError(f"Cannot extract stat from detector keys={list(det_out.keys())}")


# -----------------------------
# Loading rows keyed by prompt
# -----------------------------
def load_hf_baseline_as_neg(hf_csv: str) -> Dict[str, Dict[str, Any]]:
    rows = read_csv(hf_csv)
    out: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        k = norm_prompt(r.get("prompt_text", ""))
        if not k:
            continue
        out[k] = {
            "prompt_text": r.get("prompt_text", ""),
            "full_text": r.get("full_text", ""),
            "ppl": float(r.get("ppl", "nan")),
        }
    return out


def load_pos_csv(pos_csv: str) -> Dict[str, Dict[str, Any]]:
    rows = read_csv(pos_csv)
    out: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        k = norm_prompt(r.get("prompt_text", ""))
        if not k:
            continue
        # stat_value is what your generator scripts write
        sv = r.get("stat_value", "")
        if sv == "":
            # fallback older fieldnames if any
            sv = r.get("z", "") or r.get("score", "")
        out[k] = {
            "stat": float(sv),
            "ppl": float(r.get("ppl", "nan")),
            "full_text": r.get("full_text", ""),
            "continuation_text": r.get("continuation_text", ""),
            "seed": r.get("seed", ""),
        }
    return out


def load_neg_stats_csv(path: str) -> Dict[str, float]:
    rows = read_csv(path)
    m: Dict[str, float] = {}
    for r in rows:
        k = norm_prompt(r.get("prompt_text", ""))
        if not k:
            continue
        m[k] = float(r.get("stat_value", "nan"))
    return m


# -----------------------------
# Compute missing NEG stats for ByteKGW(all)
# -----------------------------
def compute_bytekgw_all_neg_stats(
    *,
    base_neg: Dict[str, Dict[str, Any]],
    out_path: str,
    model_path: str,
    bytekgw_config_path: str,
    device: str,
    dtype: str,
    add_special_tokens: bool,
) -> Dict[str, float]:
    """
    Loads model+tokenizer once, instantiates ByteKGWv5(all-bytes), runs detect on NEG full_text.
    Writes neg_stats_bytekgw_all.csv.
    """
    print(f"[NEG] building ByteKGWv5(all) neg stats -> {out_path}")
    dt = parse_dtype(dtype)
    dev = torch.device(device)

    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    ensure_pad_token(tok)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dt, device_map=None).to(dev)
    model.eval()

    # Build TransformersConfig (robust)
    from MarkLLM.utils.transformers_config import TransformersConfig

    gen_kwargs = {"use_cache": True}
    if tok.pad_token_id is not None:
        gen_kwargs["pad_token_id"] = int(tok.pad_token_id)
    if tok.eos_token_id is not None:
        gen_kwargs["eos_token_id"] = int(tok.eos_token_id)

    try:
        tf_cfg = TransformersConfig(model=model, tokenizer=tok, device=device, gen_kwargs=gen_kwargs)
    except TypeError:
        try:
            tf_cfg = TransformersConfig(model, tok)
        except TypeError:
            tf_cfg = TransformersConfig(model=model, tokenizer=tok)

    # temp config overrides
    with open(bytekgw_config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg.update({
        "algorithm_name": "ByteKGWv5",
        "scheme": "byte_tree",
        "delta": 1.0,                 # detector doesn't really use delta, but config expects it
        "max_byte_pos": 64,           # all-bytes
        "detector_mode": "all_bytes", # if still present
        "use_prefix_bytes_in_prf": True,
        "add_special_tokens": bool(add_special_tokens),
        "use_torch_generator": False,
    })

    fd, tmp = tempfile.mkstemp(prefix="tmp_bytekgw_all_neg_", suffix=".json")
    os.close(fd)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    try:
        from MarkLLM.watermark.bytekgwV5 import ByteKGWv5
        alg = ByteKGWv5(tmp, tf_cfg)

        keys = sorted(base_neg.keys())
        stats: Dict[str, float] = {}
        rows_out: List[Dict[str, Any]] = []

        for i, k in enumerate(keys, 1):
            txt = base_neg[k]["full_text"]
            det = alg.detect_watermark(txt, return_dict=True, add_special_tokens=False)
            z = float(extract_stat(det))
            stats[k] = z
            rows_out.append({"prompt_text": base_neg[k]["prompt_text"], "stat_value": z})
            if i % 25 == 0 or i == len(keys):
                print(f"  -> {i}/{len(keys)}", flush=True)

        write_csv(out_path, ["prompt_text", "stat_value"], rows_out)
        return stats

    finally:
        try:
            os.remove(tmp)
        except Exception:
            pass


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head_dir", required=True, help="outputs/c4_samples_head_200 (has KGW+head + neg_stats)")
    ap.add_argument("--all_dir", required=True, help="outputs/c4_samples_bytekgw_all_200 (has all-bytes)")
    ap.add_argument("--deltas", default="1,2,3,4,5")
    ap.add_argument("--fprs", default="0.01,0.05,0.10,0.20")
    ap.add_argument("--thr_mode", default="at_least", choices=["at_least", "conservative"])

    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--no_model", action="store_true",
                    help="Do not load model. Requires neg_stats_bytekgw_all.csv already exists.")
    ap.add_argument("--write_summary_csv", action="store_true")

    args = ap.parse_args()

    deltas = [int(x.strip()) for x in args.deltas.split(",") if x.strip()]
    fprs = [float(x.strip()) for x in args.fprs.split(",") if x.strip()]

    head_dir = args.head_dir
    all_dir = args.all_dir

    # NEG baseline: prefer head_dir hf_generate.csv
    hf_head = os.path.join(head_dir, "hf_generate.csv")
    hf_all  = os.path.join(all_dir,  "hf_generate.csv")
    if not os.path.exists(hf_head) and not os.path.exists(hf_all):
        raise FileNotFoundError("Need hf_generate.csv in head_dir or all_dir")

    base_hf_csv = hf_head if os.path.exists(hf_head) else hf_all
    base_neg = load_hf_baseline_as_neg(base_hf_csv)

    # Check prompts overlap between runs (optional)
    neg_all = load_hf_baseline_as_neg(hf_all) if os.path.exists(hf_all) else {}
    if neg_all:
        common = set(base_neg.keys()) & set(neg_all.keys())
        if len(common) < min(len(base_neg), len(neg_all)):
            print(f"[WARN] head/all hf prompts not identical. Using baseline from: {base_hf_csv}")
            print(f"       baseline_n={len(base_neg)} all_n={len(neg_all)} common={len(common)}")
        # To be strict, we intersect if there is a mismatch
        if len(common) > 0 and len(common) < len(base_neg):
            base_neg = {k: base_neg[k] for k in sorted(common)}
            print(f"[INFO] using intersection prompts for fairness: n={len(base_neg)}")

    keys = sorted(base_neg.keys())
    n_neg = len(keys)
    if n_neg == 0:
        raise RuntimeError("No NEG prompts loaded")

    neg_ppl = [float(base_neg[k]["ppl"]) for k in keys]
    neg_ppl_m, neg_ppl_s = mean_std(neg_ppl)

    # Load existing NEG stats for KGW + head
    neg_stats_kgw_path = os.path.join(head_dir, "neg_stats_kgw.csv")
    neg_stats_head_path = os.path.join(head_dir, "neg_stats_bytekgw_head.csv")
    if not os.path.exists(neg_stats_kgw_path):
        raise FileNotFoundError(neg_stats_kgw_path)
    if not os.path.exists(neg_stats_head_path):
        raise FileNotFoundError(neg_stats_head_path)

    neg_kgw_map = load_neg_stats_csv(neg_stats_kgw_path)
    neg_head_map = load_neg_stats_csv(neg_stats_head_path)

    # ByteKGW all NEG stats (maybe missing)
    neg_all_path = os.path.join(all_dir, "neg_stats_bytekgw_all.csv")
    if os.path.exists(neg_all_path):
        neg_all_map = load_neg_stats_csv(neg_all_path)
    else:
        if args.no_model:
            raise RuntimeError(f"{neg_all_path} missing and --no_model set.")
        # Need model path + config path from all_dir run_metadata.json
        meta_path = os.path.join(all_dir, "run_metadata.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(meta_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        model_path = meta["model"]
        bytekgw_cfg = meta["bytekgw_config"]
        add_special_tokens = bool(meta.get("add_special_tokens", True))
        neg_all_map = compute_bytekgw_all_neg_stats(
            base_neg=base_neg,
            out_path=neg_all_path,
            model_path=model_path,
            bytekgw_config_path=bytekgw_cfg,
            device=args.device,
            dtype=args.dtype,
            add_special_tokens=add_special_tokens,
        )

    # Gather NEG stats aligned to keys
    def aligned_neg(map_: Dict[str, float], name: str) -> List[float]:
        miss = [k for k in keys if k not in map_]
        if miss:
            raise RuntimeError(f"[{name}] missing NEG stats for {len(miss)} prompts (e.g. {miss[0][:60]}...)")
        return [float(map_[k]) for k in keys]

    zneg_kgw  = aligned_neg(neg_kgw_map, "KGW")
    zneg_head = aligned_neg(neg_head_map, "ByteKGW-head")
    zneg_all  = aligned_neg(neg_all_map, "ByteKGW-all")

    # Load POS per delta
    pos_kgw: Dict[int, Tuple[List[float], List[float]]] = {}
    pos_head: Dict[int, Tuple[List[float], List[float]]] = {}
    pos_all: Dict[int, Tuple[List[float], List[float]]] = {}

    for d in deltas:
        kgw_csv = os.path.join(head_dir, f"kgw_delta{d}.csv")
        head_csv = os.path.join(head_dir, f"bytekgw_head_delta{d}.csv")
        all_csv = os.path.join(all_dir, f"bytekgw_all_delta{d}.csv")

        if not os.path.exists(kgw_csv):  raise FileNotFoundError(kgw_csv)
        if not os.path.exists(head_csv): raise FileNotFoundError(head_csv)
        if not os.path.exists(all_csv):  raise FileNotFoundError(all_csv)

        kgw_map = load_pos_csv(kgw_csv)
        head_map = load_pos_csv(head_csv)
        all_map = load_pos_csv(all_csv)

        def aligned_pos(map_: Dict[str, Dict[str, Any]], name: str) -> Tuple[List[float], List[float]]:
            miss = [k for k in keys if k not in map_]
            if miss:
                raise RuntimeError(f"[{name} delta={d}] missing POS rows for {len(miss)} prompts (e.g. {miss[0][:60]}...)")
            zs = [float(map_[k]["stat"]) for k in keys]
            ppls = [float(map_[k]["ppl"]) for k in keys]
            return zs, ppls

        pos_kgw[d]  = aligned_pos(kgw_map,  "KGW")
        pos_head[d] = aligned_pos(head_map, "ByteKGW-head")
        pos_all[d]  = aligned_pos(all_map,  "ByteKGW-all")

    # ---------------- report ----------------
    print("=" * 120)
    print(f"[NEG baseline] {base_hf_csv}")
    print(f"[n_neg] {n_neg} (FPR step={1.0/n_neg:.4f}) thr_mode={args.thr_mode}")
    print(f"[deltas] {deltas}")
    print(f"[fprs] {fprs}")
    print("=" * 120)

    print(f"\n[NEG PPL] HF mean={neg_ppl_m:.6f} std={neg_ppl_s:.6f}")
    print(f"[NEG stat] KGW          [{summary(zneg_kgw)}]")
    print(f"[NEG stat] ByteKGW(head) [{summary(zneg_head)}]")
    print(f"[NEG stat] ByteKGW(all)  [{summary(zneg_all)}]  (cached at: {neg_all_path})")

    for d in deltas:
        z, ppl = pos_kgw[d]
        m, s = mean_std(ppl)
        print(f"\n[POS delta={d}] KGW          PPL mean={m:.6f} std={s:.6f} gap_vs_NEG={m-neg_ppl_m:+.6f} stat[{summary(z)}]")
        z, ppl = pos_head[d]
        m, s = mean_std(ppl)
        print(f"[POS delta={d}] ByteKGW(head) PPL mean={m:.6f} std={s:.6f} gap_vs_NEG={m-neg_ppl_m:+.6f} stat[{summary(z)}]")
        z, ppl = pos_all[d]
        m, s = mean_std(ppl)
        print(f"[POS delta={d}] ByteKGW(all)  PPL mean={m:.6f} std={s:.6f} gap_vs_NEG={m-neg_ppl_m:+.6f} stat[{summary(z)}]")

    print("\n" + "=" * 120)
    print("[RESULT] predict watermarked if stat > thr (thr calibrated on SAME NEG HF set per algorithm)")
    print("=" * 120)

    summary_rows: List[Dict[str, Any]] = []
    algs = [
        ("KGW", zneg_kgw, pos_kgw),
        ("ByteKGWv5(head)", zneg_head, pos_head),
        ("ByteKGWv5(all)", zneg_all, pos_all),
    ]

    for fpr_t in fprs:
        print(f"\n--- target_fpr={fpr_t:.4f} ---")
        for alg_name, zneg, pos_map in algs:
            thr = choose_threshold(zneg, fpr_t, mode=args.thr_mode)
            achieved = rate_above(zneg, thr)
            line = f"{alg_name:16s} thr={thr:9.4f} achieved_FPR={achieved:6.3f}"
            print(line)
            for d in deltas:
                zpos, ppl = pos_map[d]
                tpr = rate_above(zpos, thr)
                pm, ps = mean_std(ppl)
                print(f"  - delta={d:>2d}  TPR={tpr:6.3f}  PPL_mean={pm:.4f}  PPL_gap={pm-neg_ppl_m:+.4f}")
                if args.write_summary_csv:
                    summary_rows.append({
                        "algo": alg_name,
                        "target_fpr": fpr_t,
                        "thr": thr,
                        "achieved_fpr": achieved,
                        "delta": d,
                        "tpr": tpr,
                        "ppl_mean": pm,
                        "ppl_std": ps,
                        "neg_ppl_mean": neg_ppl_m,
                        "neg_ppl_std": neg_ppl_s,
                        "ppl_gap_vs_neg": pm - neg_ppl_m,
                    })

    if args.write_summary_csv:
        out = os.path.join(head_dir, "eval_summary_3alg.csv")
        write_csv(
            out,
            [
                "algo","target_fpr","thr","achieved_fpr","delta","tpr",
                "ppl_mean","ppl_std","neg_ppl_mean","neg_ppl_std","ppl_gap_vs_neg"
            ],
            summary_rows,
        )
        print(f"\n[WROTE] {out}")


if __name__ == "__main__":
    main()
