#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare TPR by PPL bins for ByteKGW head/all and KGW.

Thresholds are taken from NEG CSV stat_value columns (clean calibration).
Clean POS z uses stat_value from POS CSV; attacked POS z is recomputed with lightweight detectors.

Example:
  python analyze_ppl_bins.py \
    --head_dir outputs/c4_samples_head_200 \
    --all_dir outputs/c4_samples_bytekgw_all_200 \
    --model_path $(jq -r .model outputs/c4_samples_head_200/run_metadata.json) \
    --byte_cfg config/ByteKGWv5.json \
    --kgw_cfg config/KGW.json \
    --byte_weights_all outputs/c4_samples_bytekgw_all_200/fit_pos_weights.json \
    --delta 1 \
    --ppl_bins 2,2.5,3,3.5,4,4.5,5,6 \
    --fprs 0.01,0.05,0.10,0.20 \
    --edit_ratio 0.02 \
    --ops replace,delete,insert \
    --device_byte cpu \
    --device_kgw cuda:0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from attack_char_3alg_mp import (
    conservative_threshold,
    parallel_attack,
    pick_text_col,
    _LiteByteDetector,
    _LiteKGWDetector,
)


def read_json(p: Path):
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_list_csv(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def load_stat_col(df: pd.DataFrame) -> np.ndarray:
    for c in ["stat_value", "z", "score", "z_score"]:
        if c in df.columns:
            return df[c].to_numpy(dtype=np.float32)
    raise ValueError("No stat column found.")


def load_ppl_col(df: pd.DataFrame) -> np.ndarray:
    if "ppl" in df.columns:
        return df["ppl"].to_numpy(dtype=np.float32)
    raise ValueError("No ppl column in CSV.")


def bin_indices(ppl: np.ndarray, edges: List[float]) -> List[Tuple[float, float, np.ndarray]]:
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (ppl >= lo) & (ppl < hi)
        bins.append((lo, hi, np.nonzero(mask)[0]))
    return bins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head_dir", type=str, required=True)
    ap.add_argument("--all_dir", type=str, required=True)
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--byte_cfg", type=str, default="config/ByteKGWv5.json")
    ap.add_argument("--kgw_cfg", type=str, default="config/KGW.json")
    ap.add_argument("--byte_weights_all", type=str, default="")
    ap.add_argument("--byte_maxpos_head", type=int, default=1)
    ap.add_argument("--byte_maxpos_all", type=int, default=64)
    ap.add_argument("--deltas", type=str, default="1,2,3,4,5")
    ap.add_argument("--ppl_bins", type=str, default="2,2.5,3,3.5,4,4.5,5,6")
    ap.add_argument("--fprs", type=str, default="0.01,0.05,0.10,0.20")
    ap.add_argument("--ops", type=str, default="replace,delete,insert")
    ap.add_argument("--edit_ratio", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device_byte", type=str, default="cpu")
    ap.add_argument("--device_kgw", type=str, default="cuda:0")
    ap.add_argument("--num_workers", type=int, default=8)
    args = ap.parse_args()

    edges = parse_list_csv(args.ppl_bins)
    if len(edges) < 2:
        raise ValueError("ppl_bins must have at least two edges.")
    fprs = parse_list_csv(args.fprs)
    deltas = [int(d) for d in parse_list_csv(args.deltas)]
    ops = [x.strip() for x in args.ops.split(",") if x.strip()]

    head = Path(args.head_dir)
    all_dir = Path(args.all_dir)

    # load configs
    cfg_byte = read_json(Path(args.byte_cfg))
    use_prefix = bool(cfg_byte.get("use_prefix_bytes_in_prf", False))

    tok = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token

    # detectors
    byte_det_head = _LiteByteDetector(
        cfg_path=args.byte_cfg,
        tokenizer=tok,
        device=args.device_byte,
        byte_maxpos=args.byte_maxpos_head,
        pos_weights=None,
        use_prefix_override=use_prefix,
    )
    weights_all = None
    if args.byte_weights_all:
        wj = read_json(Path(args.byte_weights_all))
        arr = wj["weights"] if isinstance(wj, dict) and "weights" in wj else wj
        w = np.array(arr, dtype=np.float32)
        weights_all = (w / (np.linalg.norm(w) + 1e-12)).astype(np.float32)
    byte_det_all = _LiteByteDetector(
        cfg_path=args.byte_cfg,
        tokenizer=tok,
        device=args.device_byte,
        byte_maxpos=args.byte_maxpos_all,
        pos_weights=weights_all,
        use_prefix_override=use_prefix,
    )
    kgw_det = _LiteKGWDetector(cfg_path=args.kgw_cfg, tokenizer=tok, device=args.device_kgw)

    def attack(texts):
        return parallel_attack(
            texts,
            [None] * len(texts),
            model_or_tok_path=args.model_path,
            ops=ops,
            edit_ratio=args.edit_ratio,
            attack_generated_only=True,
            seed_base=args.seed,
            num_workers=min(args.num_workers, max(1, len(texts))),
            start_method="fork",
        )

    def process(algo: str):
        if algo == "byte_head":
            pos_paths = [head / f"bytekgw_head_delta{d}.csv" for d in deltas]
            neg_path = head / "neg_stats_bytekgw_head.csv"
            det = byte_det_head
        elif algo == "byte_all":
            pos_paths = []
            for d in deltas:
                p = all_dir / f"bytekgw_all_delta{d}.csv"
                if not p.exists():
                    p = all_dir / f"bytekgw_all_delta{d}_reweighted.csv"
                pos_paths.append(p)
            neg_path = all_dir / "neg_stats_bytekgw_all.csv"
            det = byte_det_all
        else:
            pos_paths = [head / f"kgw_delta{d}.csv" for d in deltas]
            neg_path = head / "neg_stats_kgw.csv"
            det = kgw_det

        df_pos = pd.concat([pd.read_csv(p) for p in pos_paths], ignore_index=True)
        text_col = pick_text_col(df_pos)
        z_clean = load_stat_col(df_pos)
        ppl = load_ppl_col(df_pos)
        texts = df_pos[text_col].astype(str).tolist()

        z_neg = pd.read_csv(neg_path)["stat_value"].to_numpy(dtype=np.float32)
        thr_map = {f: conservative_threshold(z_neg, f)[0] for f in fprs}

        # attack
        texts_att = attack(texts)
        z_att = np.array([det.score(t) for t in texts_att], dtype=np.float32)

        print(f"\n=== {algo} deltas={deltas} ===")
        for lo, hi, idxs in bin_indices(ppl, edges):
            if idxs.size == 0:
                continue
            zc = z_clean[idxs]
            za = z_att[idxs]
            print(f"PPL[{lo},{hi}) n={len(idxs)}")
            for f in fprs:
                thr = thr_map[f]
                tpr_c = float((zc > thr).mean())
                tpr_a = float((za > thr).mean())
                print(f"  fpr={f:.2f} thr={thr:.3f} tpr_clean={tpr_c:.3f} tpr_att={tpr_a:.3f}")

    for algo in ["byte_head", "byte_all", "kgw"]:
        process(algo)


if __name__ == "__main__":
    main()
