#!/usr/bin/env python3
"""
Recompute KGW z-scores for CSV datasets using the official KGW detector.

This fixes earlier runs where Charm-style scores were written into KGW CSVs.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List

from tqdm import tqdm
from transformers import AutoTokenizer

from MarkLLM.watermark.kgw.kgw import KGW
from MarkLLM.utils.transformers_config import TransformersConfig


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Recompute KGW z-scores for CSV files.")
    ap.add_argument("files", nargs="+", help="CSV files to overwrite with fresh KGW z_score values.")
    ap.add_argument("--config", default="MarkLLM/config/KGW.json", help="KGW config JSON.")
    ap.add_argument("--model", default="../Meta-Llama-3-8B", help="Tokenizer/model path (tokenizer only is used).")
    ap.add_argument("--device", default="cuda:0", help="Device used by the detector (e.g., cuda:0).")
    return ap.parse_args()


def build_detector(model_name: str, cfg_path: str, device: str) -> KGW:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code="qwen" in model_name.lower())
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tf_cfg = TransformersConfig(model=None, tokenizer=tokenizer, device=device)
    detector = KGW(cfg_path, tf_cfg)
    return detector


def iter_rows(csv_path: Path) -> List[dict]:
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = [row for row in reader]
        fieldnames = reader.fieldnames or []
    return rows, fieldnames


def write_rows(csv_path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    tmp_path = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    tmp_path.replace(csv_path)


def recompute_file(detector: KGW, csv_path: Path) -> None:
    rows, fieldnames = iter_rows(csv_path)
    if not fieldnames:
        print(f"[warn] {csv_path} has no header; skipping.")
        return
    if "z_score" not in fieldnames:
        fieldnames.append("z_score")

    for row in tqdm(rows, desc=f"kgw-z {csv_path.name}"):
        text = (row.get("text") or "").strip()
        if not text:
            row["z_score"] = ""
            continue
        res = detector.detect_watermark(text, return_dict=True)
        row["z_score"] = f"{float(res.get('score', 0.0)):.6f}"

    write_rows(csv_path, rows, fieldnames)
    print(f"[done] updated {csv_path}")


def main() -> None:
    args = parse_args()
    detector = build_detector(args.model, args.config, args.device)
    for fname in args.files:
        path = Path(fname)
        if not path.exists():
            print(f"[warn] missing file: {fname}")
            continue
        recompute_file(detector, path)


if __name__ == "__main__":
    main()
