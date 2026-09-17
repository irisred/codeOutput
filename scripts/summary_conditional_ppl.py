#!/usr/bin/env python3
"""
Summarize conditional PPL statistics for a batch of CSV files produced by
compute_conditional_ppl_csv.py. Outputs per-file average and count.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Summarize conditional PPL from CSV files.")
    ap.add_argument("files", nargs="+", help="CSV files with cond_ppl column.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    records: List[tuple[str, float, int]] = []

    for fname in args.files:
        path = Path(fname)
        if not path.exists():
            print(f"[warn] missing file: {fname}")
            continue
        ppl_vals: List[float] = []
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if "cond_ppl" not in reader.fieldnames:
                print(f"[warn] no cond_ppl column in {fname}, skipping")
                continue
            for row in reader:
                val = row.get("cond_ppl")
                if not val:
                    continue
                try:
                    ppl = float(val)
                    if ppl > 0:
                        ppl_vals.append(ppl)
                except Exception:
                    continue
        if not ppl_vals:
            print(f"[info] {fname} has zero valid cond_ppl entries.")
            continue
        avg = sum(ppl_vals) / len(ppl_vals)
        records.append((fname, avg, len(ppl_vals)))

    print("file,avg_cond_ppl,count")
    for fname, avg, count in records:
        print(f"{fname},{avg:.6f},{count}")


if __name__ == "__main__":
    main()
