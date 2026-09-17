#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute two defense metrics from trace_v6_attack_process outputs:
- three-byte head stability: rate of same UID after attack (aligned tokens)
- PRF stability: rate of green decision not flipping (all traced tokens)

Usage:
TOKENIZERS_PARALLELISM=false /home/star/jf/python/stega/bin/python \
  scripts/analyze_v6_components.py \
  --trace_glob 'traces_v6_process/trace_row*_r0.020.tsv'
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--trace_glob",
        default="traces_v6_process/trace_row*_r*.tsv",
        help="glob to trace TSVs produced by trace_v6_attack_process.py",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    paths = sorted(glob.glob(args.trace_glob))
    if not paths:
        print(f"No trace files match {args.trace_glob}", file=sys.stderr)
        sys.exit(1)

    frames = []
    for p in paths:
        df = pd.read_csv(p, sep="\t")
        frames.append(df)

    df_all = pd.concat(frames, ignore_index=True)
    aligned = df_all[df_all["pos_c"] != -1]

    same_uid_rate = float((aligned["same_uid"] == 1).mean())
    no_clean_rate = float((df_all["no_clean_token_at_same_char_start"] == 1).mean())
    token_id_changed_rate = float(
        (aligned["token_id_changed_at_same_char_start"] == 1).mean()
    )
    green_flip_rate = float((df_all["green_flip_cleanfp_vs_attackfp"] == 1).mean())

    out = {
        "files": paths,
        "total_rows": int(len(df_all)),
        "aligned_rows": int(len(aligned)),
        # Three-byte head stability: UID unchanged when aligned
        "same_uid_rate_aligned": same_uid_rate,
        # How often attack created a new token at that char start
        "no_clean_token_rate": no_clean_rate,
        # Token id changed at same start (attack touched token front)
        "token_id_changed_rate_aligned": token_id_changed_rate,
        # PRF stability: green decision flip rate
        "green_flip_rate_all": green_flip_rate,
    }

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
