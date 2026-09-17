#!/usr/bin/env python3
"""Convert a CSV file with a `text` column into the JSON format expected by test_rand_attack."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List, Dict


def convert_csv(input_csv: Path, output_json: Path, text_column: str = "text", limit: int | None = None) -> None:
    if not input_csv.exists():
        raise SystemExit(f"[error] missing input CSV: {input_csv}")

    records: List[Dict[str, str]] = []
    with input_csv.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if text_column not in reader.fieldnames:
            raise SystemExit(f"[error] column '{text_column}' not found in {input_csv}")
        for idx, row in enumerate(reader):
            text = (row.get(text_column) or "").strip()
            if not text:
                continue
            records.append({"wm_text": text})
            if limit is not None and len(records) >= limit:
                break

    if not records:
        raise SystemExit(f"[error] no usable rows found in {input_csv}")

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)
    print(f"[done] wrote {len(records)} samples to {output_json}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Convert CSV with text column to WM JSON format.")
    ap.add_argument("input_csv", type=Path, help="Source CSV path.")
    ap.add_argument("output_json", type=Path, help="Destination JSON path.")
    ap.add_argument("--text-column", default="text", help="Column containing text (default: text).")
    ap.add_argument("--limit", type=int, default=None, help="Optional maximum number of samples.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    convert_csv(args.input_csv, args.output_json, text_column=args.text_column, limit=args.limit)


if __name__ == "__main__":
    main()
