#!/usr/bin/env python3
"""Add per-sample z-score for a specific K bucket (default K=121) into CSV files."""
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
    ap = argparse.ArgumentParser(description="Compute z-score for a specific K bucket per row.")
    ap.add_argument("files", nargs="+", help="CSV files to update in-place.")
    ap.add_argument("--config", default="MarkLLM/config/CharmKGW.json", help="Charm config JSON.")
    ap.add_argument("--model", default="../Meta-Llama-3-8B", help="Tokenizer source for detector.")
    ap.add_argument("--device", default="cpu", help="Device string for detector (CPU only internally).")
    ap.add_argument(
        "--num-workers",
        type=int,
        default=mp.cpu_count(),
        help="Parallel worker processes (default = CPU core count).",
    )
    ap.add_argument("--bucket", type=int, default=121, help="K bucket to compute z for (default 121).")
    ap.add_argument(
        "--column-name",
        default=None,
        help="Optional column name (default: z_bucket_<K>).",
    )
    return ap.parse_args()


def build_detector(config_path: Path, model_name: str, device: str) -> tuple[CharmDetectorV2, float]:
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

    detector = CharmDetectorV2(
        tokenizer=tokenizer,
        hash_key=hash_key,
        gamma=gamma,
        prefix_length=prefix_length,
        z_threshold=z_threshold,
        weight_first=first_w,
        weight_other=other_w,
    )
    return detector, gamma


def bucket_z(stats: dict | None, gamma: float) -> float:
    if not stats:
        return float("nan")
    count = float(stats.get("count", 0.0))
    hits = float(stats.get("hits", 0.0))
    if count <= 0.0 or not (0.0 < gamma < 1.0):
        return float("nan")
    denom = math.sqrt(count * gamma * (1.0 - gamma))
    if denom <= 0.0:
        return float("nan")
    return (hits - count * gamma) / denom


_WORKER_DETECTOR: CharmDetectorV2 | None = None
_WORKER_GAMMA: float = 0.0
_WORKER_BUCKET: int = 0
_WORKER_COLUMN: str = ""


def _init_worker(cfg_path: str, model_name: str, device: str, bucket: int, column_name: str) -> None:
    global _WORKER_DETECTOR, _WORKER_GAMMA, _WORKER_BUCKET, _WORKER_COLUMN
    detector, gamma = build_detector(Path(cfg_path), model_name, device)
    _WORKER_DETECTOR = detector
    _WORKER_GAMMA = gamma
    _WORKER_BUCKET = bucket
    _WORKER_COLUMN = column_name


def _process_row(row: dict) -> dict:
    assert _WORKER_DETECTOR is not None
    text = (row.get("text") or "").strip()
    if not text:
        row[_WORKER_COLUMN] = ""
        return row
    res = _WORKER_DETECTOR.detect(text, return_dict=True)
    stats = res.get("k_stats", {}).get(str(_WORKER_BUCKET)) or res.get("k_stats", {}).get(_WORKER_BUCKET)
    z = bucket_z(stats, _WORKER_GAMMA)
    row[_WORKER_COLUMN] = "" if not math.isfinite(z) else f"{z:.6f}"
    return row


def process_file(
    detector: CharmDetectorV2 | None,
    gamma: float,
    csv_path: Path,
    bucket: int,
    column_name: str,
    *,
    num_workers: int = 1,
    cfg_path: Path | None = None,
    model_name: str | None = None,
    device: str | None = None,
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
    if column_name not in fieldnames:
        fieldnames.append(column_name)

    if num_workers <= 1:
        assert detector is not None
        for row in tqdm(rows, desc=f"bucket {bucket} {csv_path.name}"):
            text = (row.get("text") or "").strip()
            if not text:
                row[column_name] = ""
                continue
            res = detector.detect(text, return_dict=True)
            stats = res.get("k_stats", {}).get(str(bucket)) or res.get("k_stats", {}).get(bucket)
            z = bucket_z(stats, gamma)
            row[column_name] = "" if not math.isfinite(z) else f"{z:.6f}"
    else:
        if cfg_path is None or model_name is None or device is None:
            raise ValueError("cfg_path/model/device required for multiprocessing mode.")
        ctx = mp.get_context("spawn")
        with ctx.Pool(
            processes=num_workers,
            initializer=_init_worker,
            initargs=(str(cfg_path), model_name, device, bucket, column_name),
        ) as pool:
            rows = list(
                tqdm(
                    pool.imap(_process_row, rows, chunksize=4),
                    total=len(rows),
                    desc=f"bucket {bucket} {csv_path.name}",
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
    column_name = args.column_name or f"z_bucket_{args.bucket}"
    detector: CharmDetectorV2 | None = None
    gamma = 0.0
    if args.num_workers <= 1:
        detector, gamma = build_detector(Path(args.config), args.model, args.device)
    for fname in args.files:
        path = Path(fname)
        if not path.exists():
            print(f"[warn] missing file: {fname}")
            continue
        process_file(
            detector,
            gamma,
            path,
            int(args.bucket),
            column_name,
            num_workers=args.num_workers,
            cfg_path=Path(args.config),
            model_name=args.model,
            device=args.device,
        )


if __name__ == "__main__":
    main()
