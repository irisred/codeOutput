#!/usr/bin/env python3
"""
Augment watermark CSV files with detection z-scores:
- For KGW rows: add a single column "z_score".
- For Charm rows: add "z_first_byte" and "z_other_byte".
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
    ap = argparse.ArgumentParser(description="Add detection z-scores to CSV files.")
    ap.add_argument("files", nargs="+", help="Input CSV files to process.")
    ap.add_argument("--config", default="MarkLLM/config/CharmKGW.json", help="Charm config JSON path.")
    ap.add_argument("--model", default="../Meta-Llama-3-8B", help="Tokenizer source for prompts/detector.")
    ap.add_argument("--prompts", default="data/prompts_c4.txt", help="Prompt file used during generation.")
    ap.add_argument("--prompt-trim", type=int, default=64, help="Prompt trim length (0 = no trim).")
    ap.add_argument(
        "--output-suffix",
        default="_with_z.csv",
        help="Suffix to append to each output file (default: _with_z.csv).",
    )
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


def build_detector(config_path: Path, tokenizer) -> Tuple[CharmDetectorV2, int]:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    gamma = float(cfg.get("gamma", 0.5))
    hash_key = int(cfg.get("hash_key", 0))
    prefix_length = int(cfg.get("prefix_length", 1))
    charm_cfg = cfg.get("charm_cfg") or {}
    prefix_length = int(charm_cfg.get("prefix_length", prefix_length))

    weight_cfg = charm_cfg.get("first_other_weights", {})
    weight_first = float(weight_cfg.get("first", 1.0))
    weight_other = float(weight_cfg.get("other", 0.0))
    z_threshold = float(cfg.get("z_threshold", 4.0))

    detector = CharmDetectorV2(
        tokenizer=tokenizer,
        hash_key=hash_key,
        gamma=gamma,
        prefix_length=prefix_length,
        z_threshold=z_threshold,
        weight_first=weight_first,
        weight_other=weight_other,
    )

    k_start = len(detector._trie_next[0])
    return detector, k_start


def z_from_hits(hits: int, trials: int, gamma: float) -> float:
    if trials <= 0:
        return 0.0
    g = float(gamma)
    denom = math.sqrt(max(trials * g * (1.0 - g), 1e-12))
    return (hits - trials * g) / denom


def iterate_rows(csv_path: Path):
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "prompt_id" not in reader.fieldnames or "text" not in reader.fieldnames:
            raise SystemExit(f"CSV {csv_path} must contain prompt_id and text columns.")
        for row in reader:
            yield row


def compute_z_values(detector: CharmDetectorV2, k_start: int, prompt: str, text: str, method: str) -> Dict[str, float]:
    vb = detector._visible_bytes(text)
    L = len(vb)
    if L <= detector.prefix_length:
        if method == "kgw":
            return {"z_score": 0.0}
        return {"z_first_byte": 0.0, "z_other_byte": 0.0}

    K_all = detector._compute_k_per_byte(text)
    gamma = detector.gamma
    first_hits = first_trials = 0
    other_hits = other_trials = 0

    for t in range(detector.prefix_length, L):
        window = vb[t - detector.prefix_length : t]
        green = detector.prf.greenlist(window, gamma)
        b = int(vb[t])
        hit = b in green
        Ki = int(K_all[t]) if 0 <= t < len(K_all) else 0
        if Ki == k_start:
            first_trials += 1
            if hit:
                first_hits += 1
        else:
            other_trials += 1
            if hit:
                other_hits += 1

    if method == "kgw":
        total_hits = first_hits + other_hits
        total_trials = first_trials + other_trials
        return {"z_score": z_from_hits(total_hits, total_trials, gamma)}

    return {
        "z_first_byte": z_from_hits(first_hits, first_trials, gamma),
        "z_other_byte": z_from_hits(other_hits, other_trials, gamma),
    }


def process_file(
    csv_path: Path,
    prompts: List[str],
    detector: CharmDetectorV2,
    k_start: int,
    suffix: str,
) -> None:
    rows = list(iterate_rows(csv_path))
    if not rows:
        print(f"[warn] {csv_path} has no rows; skipping.")
        return

    method = rows[0].get("method", "").strip().lower()
    if method == "plain":
        method = "charm"
    if method not in {"kgw", "charm"}:
        print(f"[warn] {csv_path} method '{method}' not kgw/charm; skipping.")
        return

    updated_rows = []
    for row in tqdm(rows, desc=f"z-scores {csv_path.name}"):
        try:
            prompt_idx = int(row.get("prompt_id", 0))
        except Exception:
            prompt_idx = 0
        prompt = prompts[prompt_idx] if 0 <= prompt_idx < len(prompts) else prompts[0]
        text = (row.get("text") or "").strip()
        if not text:
            row.setdefault("z_score", "")
            row.setdefault("z_first_byte", "")
            row.setdefault("z_other_byte", "")
            updated_rows.append(row)
            continue
        z_values = compute_z_values(detector, k_start, prompt, text, method)
        for key, value in z_values.items():
            row[key] = f"{value:.6f}"
        updated_rows.append(row)

    output_path = csv_path.with_name(f"{csv_path.stem}{suffix}")
    fieldnames = list(updated_rows[0].keys())
    if method == "kgw" and "z_score" not in fieldnames:
        fieldnames.append("z_score")
    if method == "charm":
        if "z_first_byte" not in fieldnames:
            fieldnames.append("z_first_byte")
        if "z_other_byte" not in fieldnames:
            fieldnames.append("z_other_byte")

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in updated_rows:
            writer.writerow(row)
    print(f"[done] Wrote z-scores to {output_path}")


def main() -> None:
    args = parse_args()
    prompts = load_prompts(Path(args.prompts), args.prompt_trim)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code="qwen" in args.model.lower())
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    detector, k_start = build_detector(Path(args.config), tokenizer)

    for fname in args.files:
        csv_path = Path(fname)
        if not csv_path.exists():
            print(f"[warn] missing file {fname}, skipping.")
            continue
        process_file(csv_path, prompts, detector, k_start, args.output_suffix)


if __name__ == "__main__":
    main()
