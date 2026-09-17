#!/usr/bin/env python3
"""Recompute detector-aligned first/other-byte z-scores for CSV rows."""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
from pathlib import Path
from typing import List

from tqdm import tqdm
from transformers import AutoTokenizer

from MarkLLM.charm_v2.detector import CharmDetectorV2


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Update z_first_byte/z_other_byte using detector semantics.")
    ap.add_argument("files", nargs="+", help="CSV files to update in-place.")
    ap.add_argument("--config", default="MarkLLM/config/CharmKGW.json", help="Charm config JSON path.")
    ap.add_argument("--model", default="../Meta-Llama-3-8B", help="Tokenizer/model source.")
    ap.add_argument("--device", default="cpu", help="Device string for detector (CPU inside workers).")
    ap.add_argument(
        "--num-workers",
        type=int,
        default=max(1, mp.cpu_count() // 2),
        help="Parallel worker processes (set 1 to disable multiprocessing).",
    )
    ap.add_argument("--first-column", default="z_first_byte", help="Column name for first-byte z.")
    ap.add_argument("--other-column", default="z_other_byte", help="Column name for other-byte z.")
    return ap.parse_args()


def build_detector(config_path: Path, model_name: str, device: str) -> CharmDetectorV2:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    gamma = float(cfg.get("gamma", 0.5))
    hash_key = int(cfg.get("hash_key", 0))
    prefix_length = int(cfg.get("prefix_length", 1))
    charm_cfg = cfg.get("charm_cfg") or {}
    prefix_length = int(charm_cfg.get("prefix_length", prefix_length))
    first_w = float(cfg.get("first_byte_weight", charm_cfg.get("first_byte_weight", 1.0)))
    other_w = float(cfg.get("other_byte_weight", charm_cfg.get("other_byte_weight", 0.0)))
    z_threshold = float(cfg.get("z_threshold", 4.0))

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code="qwen" in model_name.lower())
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    return CharmDetectorV2(
        tokenizer=tokenizer,
        hash_key=hash_key,
        gamma=gamma,
        prefix_length=prefix_length,
        z_threshold=z_threshold,
        weight_first=first_w,
        weight_other=other_w,
    )


def _format_z(value: float) -> str:
    return "" if value is None or not math.isfinite(value) else f"{value:.6f}"


def _extract_first_other(res: dict) -> tuple[float | None, float | None]:
    bucket_stats = res.get("bucket_stats") or {}
    seg = bucket_stats.get("segmented_start") or {}
    nonseg = bucket_stats.get("nonsegmented_start") or {}
    return seg.get("z"), nonseg.get("z")


_WORKER_DETECTOR: CharmDetectorV2 | None = None
_WORKER_FIRST_COL: str = ""
_WORKER_OTHER_COL: str = ""


def _init_worker(cfg_path: str, model_name: str, device: str, first_col: str, other_col: str) -> None:
    global _WORKER_DETECTOR, _WORKER_FIRST_COL, _WORKER_OTHER_COL
    _WORKER_DETECTOR = build_detector(Path(cfg_path), model_name, device)
    _WORKER_FIRST_COL = first_col
    _WORKER_OTHER_COL = other_col


def _process_row(row: dict) -> dict:
    assert _WORKER_DETECTOR is not None
    text = (row.get("text") or "").strip()
    if not text:
        row[_WORKER_FIRST_COL] = ""
        row[_WORKER_OTHER_COL] = ""
        return row
    res = _WORKER_DETECTOR.detect(text, return_dict=True)
    z_first, z_other = _extract_first_other(res)
    row[_WORKER_FIRST_COL] = _format_z(z_first if z_first is not None else float("nan"))
    row[_WORKER_OTHER_COL] = _format_z(z_other if z_other is not None else float("nan"))
    return row


def process_file(
    detector: CharmDetectorV2 | None,
    csv_path: Path,
    *,
    first_column: str,
    other_column: str,
    num_workers: int,
    cfg_path: Path,
    model_name: str,
    device: str,
) -> None:
    rows: List[dict] = []
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        for row in reader:
            rows.append(row)
    if not rows:
        print(f"[warn] {csv_path} has no rows; skipping")
        return
    if first_column not in fieldnames:
        fieldnames.append(first_column)
    if other_column not in fieldnames:
        fieldnames.append(other_column)

    if num_workers <= 1:
        assert detector is not None
        for row in tqdm(rows, desc=f"first/other {csv_path.name}"):
            text = (row.get("text") or "").strip()
            if not text:
                row[first_column] = ""
                row[other_column] = ""
                continue
            res = detector.detect(text, return_dict=True)
            z_first, z_other = _extract_first_other(res)
            row[first_column] = _format_z(z_first if z_first is not None else float("nan"))
            row[other_column] = _format_z(z_other if z_other is not None else float("nan"))
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(
            processes=num_workers,
            initializer=_init_worker,
            initargs=(str(cfg_path), model_name, device, first_column, other_column),
        ) as pool:
            rows = list(
                tqdm(
                    pool.imap(_process_row, rows, chunksize=4),
                    total=len(rows),
                    desc=f"first/other {csv_path.name}",
                )
            )

    tmp_path = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    tmp_path.replace(csv_path)
    print(f"[done] updated {csv_path}")


def main() -> None:
    args = parse_args()
    detector: CharmDetectorV2 | None = None
    if args.num_workers <= 1:
        detector = build_detector(Path(args.config), args.model, args.device)
    for fname in args.files:
        csv_path = Path(fname)
        if not csv_path.exists():
            print(f"[warn] missing {fname}; skipping")
            continue
        process_file(
            detector,
            csv_path,
            first_column=args.first_column,
            other_column=args.other_column,
            num_workers=args.num_workers,
            cfg_path=Path(args.config),
            model_name=args.model,
            device=args.device,
        )


if __name__ == "__main__":
    main()
