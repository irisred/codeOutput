#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sweep multiple attack ratios and styles (char/token) across the four algorithms,
using analyze_attack_tpr_by_bucket.py. Outputs one CSV per ratio.

Defaults:
  ratios: 0,0.01,0.02,0.05,0.10
  atk_styles: char,token
  out_root: outputs/wm_eval_attack_grid/

Example:
TOKENIZERS_PARALLELISM=false /home/star/jf/python/stega/bin/python scripts/run_attack_grid.py \
  --device cuda:0
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_meta", default="outputs/c4_samples_head_200/run_metadata.json")
    ap.add_argument("--v6_config", default="config/ByteKGWv6.json")
    ap.add_argument("--kgw_config", default="config/KGW.json")
    ap.add_argument("--dip_config", default="config/DIP.json")
    ap.add_argument("--unbiased_config", default="config/Unbiased.json")
    ap.add_argument("--clean_csv", default="outputs/wm_eval/clean/hf_generate.csv")
    ap.add_argument("--bucket_root", default="outputs/wm_eval")
    ap.add_argument("--fprs", default="0.01,0.05,0.1,0.2")
    ap.add_argument("--atk_styles", default="char,token", help="comma-separated styles (char,token,...)")
    ap.add_argument(
        "--ratios",
        default="0,0.01,0.02,0.05,0.10",
        help="comma-separated attack ratios to sweep",
    )
    ap.add_argument("--attack_seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out_root", default="outputs/wm_eval_attack_grid")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    ratios = [r for r in args.ratios.split(",") if r.strip()]

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for r in ratios:
        out_dir = out_root / f"r{r}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_csv = out_dir / "tpr_attack_by_bucket.csv"

        cmd = [
            sys.executable,
            "scripts/analyze_attack_tpr_by_bucket.py",
            "--run_meta",
            args.run_meta,
            "--v6_config",
            args.v6_config,
            "--kgw_config",
            args.kgw_config,
            "--dip_config",
            args.dip_config,
            "--unbiased_config",
            args.unbiased_config,
            "--clean_csv",
            args.clean_csv,
            "--bucket_root",
            args.bucket_root,
            "--fprs",
            args.fprs,
            "--attack_ratio",
            str(r),
            "--attack_seed",
            str(args.attack_seed),
            "--atk_styles",
            args.atk_styles,
            "--device",
            args.device,
            "--out_csv",
            str(out_csv),
        ]

        print("\n=== Running ratio", r, "styles", args.atk_styles, "->", out_csv)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
