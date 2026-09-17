#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
from typing import Dict, Iterable, List, Tuple, Optional

import numpy as np
import pandas as pd


def norm_prompt(s: str) -> str:
    """Normalize prompt for stable joins across csv files."""
    if s is None:
        return ""
    s = str(s).strip()
    # collapse whitespace
    s = re.sub(r"\s+", " ", s)
    return s


def pick_stat_series(df: pd.DataFrame) -> pd.Series:
    """
    Robustly pick a statistic column (watermark detection stat).
    Priority:
      1) stat_value
      2) z
      3) score
      4) first numeric-like column
    """
    for col in ["stat_value", "z", "score"]:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce")

    # fallback: find first column that looks numeric for most rows
    best_col = None
    best_valid = -1
    for col in df.columns:
        if col in ["prompt_text", "full_text", "continuation_text", "gen_params_json"]:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        valid = int(s.notna().sum())
        if valid > best_valid:
            best_valid = valid
            best_col = col
    if best_col is None:
        raise ValueError("Cannot find any numeric stat column in dataframe.")
    return pd.to_numeric(df[best_col], errors="coerce")


def choose_threshold(zs: np.ndarray, target_fpr: float, mode: str = "conservative") -> float:
    """
    Choose threshold thr such that P(z > thr) ~= target_fpr on NEG.
    mode:
      - conservative: minimize |fpr-target|, tie-break prefer fpr <= target (more conservative)
      - at_least: prefer fpr >= target with smallest overshoot; tie-break prefer larger thr
    """
    zs = np.asarray(zs, dtype=np.float64)
    zs = zs[np.isfinite(zs)]
    if zs.size == 0:
        raise ValueError("NEG stats empty after filtering NaNs.")

    uniq = np.unique(zs)
    eps = 1e-12
    cands = np.concatenate(([uniq.max() + eps], uniq, [uniq.min() - eps]))

    best_thr = None
    best_key = None

    for thr in cands:
        fpr = float(np.mean(zs > thr))

        if mode == "at_least":
            # want fpr >= target, minimal overshoot; if all < target, pick closest below
            overshoot = fpr - target_fpr
            if overshoot >= 0:
                key = (0, overshoot, -thr)  # prefer smaller overshoot, then larger thr
            else:
                key = (1, abs(overshoot), -thr)  # below target: penalize
        elif mode == "conservative":
            # minimize abs error; tie-break prefer fpr <= target; then larger thr
            err = abs(fpr - target_fpr)
            tie = 0 if fpr <= target_fpr else 1
            key = (err, tie, -thr)
        else:
            raise ValueError("mode must be 'conservative' or 'at_least'")

        if best_key is None or key < best_key:
            best_key = key
            best_thr = float(thr)

    return float(best_thr)


def load_baseline_neg(hf_csv: str) -> pd.DataFrame:
    """
    Load hf_generate.csv as NEG baseline prompts & ppl.
    Expect columns include prompt_text, ppl (full_text optional).
    """
    df = pd.read_csv(hf_csv)
    if "prompt_text" not in df.columns:
        raise ValueError(f"{hf_csv} missing prompt_text")
    df["k"] = df["prompt_text"].map(norm_prompt)
    if "ppl" not in df.columns:
        raise ValueError(f"{hf_csv} missing ppl")
    df["ppl_neg"] = pd.to_numeric(df["ppl"], errors="coerce")
    return df[["k", "prompt_text", "ppl_neg"]].drop_duplicates("k")


def load_neg_stats(path: str, label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "prompt_text" not in df.columns:
        raise ValueError(f"{path} missing prompt_text")
    df["k"] = df["prompt_text"].map(norm_prompt)
    stat = pick_stat_series(df)
    out = pd.DataFrame({"k": df["k"], f"neg_stat_{label}": stat})
    return out.drop_duplicates("k")


def load_pos_csv(path: str, label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "prompt_text" not in df.columns:
        raise ValueError(f"{path} missing prompt_text")
    df["k"] = df["prompt_text"].map(norm_prompt)

    stat = pick_stat_series(df)
    ppl = pd.to_numeric(df["ppl"], errors="coerce") if "ppl" in df.columns else pd.Series([np.nan] * len(df))
    # delta column may exist; else infer from filename
    delta = None
    if "delta" in df.columns:
        try:
            delta = float(df["delta"].iloc[0])
        except Exception:
            delta = None

    out = pd.DataFrame(
        {
            "k": df["k"],
            f"pos_stat_{label}": stat,
            f"pos_ppl_{label}": ppl,
        }
    ).drop_duplicates("k")
    if delta is not None:
        out["delta"] = delta
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head_dir", required=True, help="e.g. outputs/c4_samples_head_200")
    ap.add_argument("--all_dir", required=True, help="e.g. outputs/c4_samples_bytekgw_all_200")
    ap.add_argument("--deltas", default="1,2,3,4,5", help="comma list, e.g. 1,2,3,4,5")
    ap.add_argument("--fprs", default="0.01,0.05,0.10,0.20", help="comma list of target FPRs")
    ap.add_argument("--thr_mode", default="conservative", choices=["conservative", "at_least"])
    ap.add_argument("--out_csv", default="eval_3alg_from_existing.csv")
    args = ap.parse_args()

    deltas = [int(x) for x in args.deltas.split(",") if x.strip()]
    fprs = [float(x) for x in args.fprs.split(",") if x.strip()]

    # baseline NEG prompts & ppl
    hf_head = os.path.join(args.head_dir, "hf_generate.csv")
    hf_all = os.path.join(args.all_dir, "hf_generate.csv")
    hf_csv = hf_head if os.path.exists(hf_head) else hf_all
    if not os.path.exists(hf_csv):
        raise FileNotFoundError("Cannot find hf_generate.csv in head_dir or all_dir")
    neg_base = load_baseline_neg(hf_csv)

    # NEG stats for 3 algos
    neg_kgw = load_neg_stats(os.path.join(args.head_dir, "neg_stats_kgw.csv"), "kgw")
    neg_head = load_neg_stats(os.path.join(args.head_dir, "neg_stats_bytekgw_head.csv"), "head")
    neg_all = load_neg_stats(os.path.join(args.all_dir, "neg_stats_bytekgw_all.csv"), "all")

    # merge NEG info
    neg = neg_base.merge(neg_kgw, on="k", how="inner").merge(neg_head, on="k", how="inner").merge(neg_all, on="k", how="inner")
    if len(neg) == 0:
        raise ValueError("No common prompts across hf_generate and neg_stats files (join is empty).")

    neg_mean_ppl = float(np.nanmean(neg["ppl_neg"].to_numpy()))

    print(f"[INFO] Using baseline: {hf_csv}")
    print(f"[INFO] Common prompts for NEG: {len(neg)}")
    print(f"[INFO] NEG mean ppl: {neg_mean_ppl:.4f}")

    results: List[Dict] = []

    # Precompute thresholds per algo per target_fpr (calibrated on NEG)
    thr_map: Dict[Tuple[str, float], Tuple[float, float]] = {}  # (algo, target_fpr) -> (thr, achieved_fpr)
    for target_fpr in fprs:
        for algo, col in [("kgw", "neg_stat_kgw"), ("bytekgw_head", "neg_stat_head"), ("bytekgw_all", "neg_stat_all")]:
            zneg = neg[col].to_numpy(dtype=np.float64)
            thr = choose_threshold(zneg, target_fpr, mode=args.thr_mode)
            achieved = float(np.mean(zneg > thr))
            thr_map[(algo, target_fpr)] = (thr, achieved)

    # Evaluate POS per delta
    for d in deltas:
        pos_kgw_path = os.path.join(args.head_dir, f"kgw_delta{d}.csv")
        pos_head_path = os.path.join(args.head_dir, f"bytekgw_head_delta{d}.csv")
        pos_all_path = os.path.join(args.all_dir, f"bytekgw_all_delta{d}.csv")

        for p in [pos_kgw_path, pos_head_path, pos_all_path]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing POS csv: {p}")

        pos_kgw = load_pos_csv(pos_kgw_path, "kgw")
        pos_head = load_pos_csv(pos_head_path, "head")
        pos_all = load_pos_csv(pos_all_path, "all")

        # Merge POS with NEG baseline prompts intersection (important for fair ppl comparisons)
        pos = neg[["k", "ppl_neg"]].merge(pos_kgw, on="k", how="inner").merge(pos_head, on="k", how="inner").merge(pos_all, on="k", how="inner")
        if len(pos) == 0:
            raise ValueError(f"Join empty at delta={d}. Prompts mismatch across files?")

        # For each target fpr compute tpr, avg ppl
        for target_fpr in fprs:
            for algo, stat_col, ppl_col in [
                ("kgw", "pos_stat_kgw", "pos_ppl_kgw"),
                ("bytekgw_head", "pos_stat_head", "pos_ppl_head"),
                ("bytekgw_all", "pos_stat_all", "pos_ppl_all"),
            ]:
                thr, achieved_fpr = thr_map[(algo, target_fpr)]
                zpos = pos[stat_col].to_numpy(dtype=np.float64)
                tpr = float(np.mean(zpos > thr))

                ppl_pos = pos[ppl_col].to_numpy(dtype=np.float64)
                ppl_mean = float(np.nanmean(ppl_pos))
                ppl_gap = float(ppl_mean - np.nanmean(pos["ppl_neg"].to_numpy(dtype=np.float64)))

                results.append(
                    {
                        "target_fpr": target_fpr,
                        "algo": algo,
                        "delta": d,
                        "thr": thr,
                        "achieved_fpr": achieved_fpr,
                        "tpr": tpr,
                        "ppl_mean": ppl_mean,
                        "ppl_gap_vs_neg": ppl_gap,
                        "n_prompts": int(len(pos)),
                    }
                )

    out = pd.DataFrame(results).sort_values(["target_fpr", "algo", "delta"])
    out.to_csv(args.out_csv, index=False)
    print(f"[OK] Wrote: {args.out_csv}")

    # Also print a compact pivot for quick view
    try:
        piv = out.pivot_table(index=["target_fpr", "delta"], columns="algo", values="tpr")
        print("\n=== TPR pivot (rows: target_fpr,delta; cols: algo) ===")
        print(piv.to_string(float_format=lambda x: f"{x:.3f}"))
    except Exception:
        pass


if __name__ == "__main__":
    main()
