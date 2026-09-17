#!/usr/bin/env python3
"""Compute TPR at target FPR for KGW / Charm detectors, with on-disk caching."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from MarkLLM.charm_v2.charm_kgw import CharmKGW
from MarkLLM.utils.transformers_config import TransformersConfig
from MarkLLM.watermark.kgw.kgw import KGW


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Analyze detection TPR at desired FPR=10%")
    ap.add_argument("--plain-csv", required=True)
    ap.add_argument("--kgw-csvs", nargs="*", default=[])
    ap.add_argument("--charm-csvs", nargs="*", default=[])
    ap.add_argument("--model", default="../Meta-Llama-3-8B")
    ap.add_argument("--kgw-config", default="MarkLLM/config/KGW.json")
    ap.add_argument("--charm-config", default="MarkLLM/config/CharmKGW.json")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--fpr", type=float, default=0.10)
    ap.add_argument("--cache-file", default="saved_data/detect_cache.json")
    return ap.parse_args()


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_cache(path: Path) -> Dict[str, Dict[str, float]]:
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    return {"kgw": {}, "charm": {}}


def save_cache(path: Path, cache: Dict[str, Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(cache, handle)


def load_runtime(model: str, device: str) -> TransformersConfig:
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code="qwen" in model.lower())
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    torch_dtype = torch.float16 if device.startswith("cuda") else None
    model_obj = AutoModelForCausalLM.from_pretrained(
        model,
        torch_dtype=torch_dtype,
        trust_remote_code="qwen" in model.lower(),
    ).to(device)
    model_obj.eval()
    return TransformersConfig(model=model_obj, tokenizer=tokenizer, device=device)


def load_texts(csv_path: str) -> List[str]:
    rows: List[str] = []
    with Path(csv_path).open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            text = (row.get("text") or "").strip()
            if text:
                rows.append(text)
    return rows


def detect_scores(detector, key: str, texts: List[str], cache: Dict[str, Dict[str, float]]) -> List[float]:
    store = cache.setdefault(key, {})
    scores = []
    for text in texts:
        h = hash_text(text)
        if h not in store:
            res = detector.detect_watermark(text, return_dict=True)
            store[h] = float(res.get("score", 0.0))
        scores.append(store[h])
    return scores


def tau_at_fpr(scores: List[float], fpr: float) -> float:
    if not scores:
        return float("inf")
    sorted_scores = sorted(scores)
    idx = max(0, min(len(sorted_scores) - 1, int((1.0 - fpr) * len(sorted_scores))))
    return sorted_scores[idx]


def tpr(scores: List[float], tau: float) -> float:
    if not scores:
        return 0.0
    hits = sum(1 for s in scores if s > tau)
    return hits / len(scores)


def analyze(detector, name: str, plain_scores: List[float], wm_files: List[str], cache: Dict[str, Dict[str, float]], fpr: float) -> None:
    if not wm_files:
        return
    tau = tau_at_fpr(plain_scores, fpr)
    print(f"\n=== {name.upper()} @ FPR={fpr*100:.1f}% ===")
    print(f"tau={tau:.4f} (plain N={len(plain_scores)})")
    for file in wm_files:
        texts = load_texts(file)
        scores = detect_scores(detector, name, texts, cache)
        val = tpr(scores, tau)
        print(f"  {Path(file).name}: TPR={val*100:.2f}% (N={len(scores)})")


def main():
    args = parse_args()
    cache_file = Path(args.cache_file)
    cache = load_cache(cache_file)

    tf_cfg = load_runtime(args.model, args.device)
    charm = CharmKGW(args.charm_config, tf_cfg)
    kgw = KGW(args.kgw_config, tf_cfg)

    plain_texts = load_texts(args.plain_csv)
    plain_scores_charm = detect_scores(charm, "charm", plain_texts, cache)
    plain_scores_kgw = detect_scores(kgw, "kgw", plain_texts, cache)

    analyze(charm, "charm", plain_scores_charm, args.charm_csvs, cache, args.fpr)
    analyze(kgw, "kgw", plain_scores_kgw, args.kgw_csvs, cache, args.fpr)

    save_cache(cache_file, cache)


if __name__ == "__main__":
    main()
