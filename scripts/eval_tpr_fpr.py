#!/usr/bin/env python3
"""
Compute detection threshold from a plain CSV at a target FPR,
then report TPR for other CSV files using CharmDetectorV2.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm import tqdm
from transformers import AutoTokenizer

from MarkLLM.charm_v2.detector import CharmDetectorV2


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Evaluate TPR @ FPR using Charm detector.")
    ap.add_argument("--plain", required=True, help="CSV file of plain (negative) samples.")
    ap.add_argument(
        "--files",
        nargs="+",
        required=True,
        help="CSV files (watermarked) to evaluate TPR against.",
    )
    ap.add_argument("--config", default="MarkLLM/config/CharmKGW.json", help="Charm config JSON.")
    ap.add_argument("--model", default="../Meta-Llama-3-8B", help="HF model/tokenizer for detector.")
    ap.add_argument("--prompts", default="data/prompts_c4.txt", help="Prompt file used during generation.")
    ap.add_argument("--prompt-trim", type=int, default=64, help="Prompt trim length used during generation.")
    ap.add_argument("--device", default="cpu", help="Device for tokenizer encoding (detector is CPU only).")
    ap.add_argument("--fpr", type=float, default=0.001, help="Target false-positive rate (e.g., 0.001 = 0.1%).")
    return ap.parse_args()


def load_prompts(path: Path, trim: int) -> List[str]:
    prompts: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            prompt = line[:trim] if (trim and trim > 0) else line
            prompts.append(prompt)
    if not prompts:
        raise SystemExit(f"No prompts found in {path}")
    return prompts


def build_detector(config_path: Path, tokenizer) -> CharmDetectorV2:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    gamma = float(cfg.get("gamma", 0.5))
    hash_key = int(cfg.get("hash_key", 0))
    prefix_length = int(cfg.get("prefix_length", 1))
    charm_cfg = cfg.get("charm_cfg") or {}
    prefix_length = int(charm_cfg.get("prefix_length", prefix_length))
    z_threshold = float(cfg.get("z_threshold", 4.0))
    first_w = float(cfg.get("first_byte_weight", charm_cfg.get("first_byte_weight", 1.0)))
    other_w = float(cfg.get("other_byte_weight", charm_cfg.get("other_byte_weight", 0.0)))

    return CharmDetectorV2(
        tokenizer=tokenizer,
        hash_key=hash_key,
        gamma=gamma,
        prefix_length=prefix_length,
        z_threshold=z_threshold,
        weight_first=first_w,
        weight_other=other_w,
    )


def iter_rows(csv_path: Path):
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "prompt_id" not in reader.fieldnames or "text" not in reader.fieldnames:
            raise SystemExit(f"CSV {csv_path} must contain prompt_id and text columns.")
        for row in reader:
            yield row


def compute_scores(detector: CharmDetectorV2, csv_path: Path, prompts: List[str]) -> List[float]:
    rows = list(iter_rows(csv_path))
    scores: List[float] = []
    for row in tqdm(rows, desc=f"Scoring {csv_path.name}"):
        try:
            prompt_idx = int(row.get("prompt_id", 0))
        except Exception:
            prompt_idx = 0
        text = (row.get("text") or "").strip()
        if not text:
            continue
        res = detector.detect(text, return_dict=True, verify=False)
        scores.append(float(res.get("score", 0.0)))
    return scores


def threshold_from_scores(scores: List[float], target_fpr: float) -> float:
    if not scores:
        raise ValueError("No scores provided for threshold calculation.")
    n = len(scores)
    allowed_fp = max(1, math.ceil(target_fpr * n))
    scores_sorted = sorted(scores)
    if allowed_fp >= n:
        return scores_sorted[0] - 1.0
    threshold = scores_sorted[-allowed_fp]
    return threshold


def main() -> None:
    args = parse_args()

    prompts = load_prompts(Path(args.prompts), args.prompt_trim)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code="qwen" in args.model.lower())
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    detector = build_detector(Path(args.config), tokenizer)

    plain_scores = compute_scores(detector, Path(args.plain), prompts)
    threshold = threshold_from_scores(plain_scores, args.fpr)
    plain_fpr = sum(score >= threshold for score in plain_scores) / len(plain_scores) if plain_scores else float("nan")

    print(f"Threshold (score >=) for FPR {args.fpr:.4f}: {threshold:.4f}")
    print(f"Empirical FPR on plain set: {plain_fpr:.6f} ({sum(score >= threshold for score in plain_scores)}/{len(plain_scores)})")
    print("\nfile,count,tpr")
    for fname in args.files:
        scores = compute_scores(detector, Path(fname), prompts)
        if not scores:
            print(f"{fname},0,0.0")
            continue
        hits = sum(score >= threshold for score in scores)
        tpr = hits / len(scores)
        print(f"{fname},{len(scores)},{tpr:.6f}")


if __name__ == "__main__":
    main()
