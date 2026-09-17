"""
Compute KGW PRF green-set overlap before vs. after char attack.

For each sample and each token position (>= prefix_length):
  - Build greenlist on clean prefix and attacked prefix.
  - overlap = |G_clean ∩ G_att| / |G_clean|
  - IoU     = |G_clean ∩ G_att| / |G_clean ∪ G_att|

Outputs overall mean overlap/IoU and per-sample mean stats.

Example:
  TOKENIZERS_PARALLELISM=false /home/star/jf/python/stega/bin/python scripts/analyze_kgw_prf_overlap.py \
    --run_meta outputs/c4_samples_head_200/run_metadata.json \
    --kgw_config config/KGW.json \
    --input_csv outputs/v6_vs_kgw_gen/kgw_delta2.0.csv \
    --attack_ratio 0.02 \
    --n_samples 50 \
    --device cuda:0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from MarkLLM.watermark.kgw.kgw import KGWUtils  # type: ignore


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_meta", required=True)
    ap.add_argument("--kgw_config", required=True)
    ap.add_argument("--input_csv", required=True, help="CSV with prompt_text/full_text")
    ap.add_argument("--n_samples", type=int, default=None)
    ap.add_argument("--attack_ratio", type=float, default=0.02)
    ap.add_argument("--attack_seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--step_stride", type=int, default=1, help="skip characters to speed up (>=1)")
    return ap.parse_args()


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def apply_char_attack(text: str, prompt_text: str, ratio: float, *, seed: int = 0) -> str:
    if ratio <= 0.0:
        return text
    import random

    random.seed(seed)
    if text.startswith(prompt_text):
        prefix = prompt_text
        cont = text[len(prompt_text) :]
    else:
        prefix = ""
        cont = text
    chars = list(cont)
    L = len(chars)
    k = max(1, int(L * ratio)) if L > 0 else 0
    idxs = random.sample(range(L), k) if L > 0 else []
    for i in idxs:
        chars[i] = "X"
    return prefix + "".join(chars)


class SimpleKGWConfig:
    def __init__(self, cfg: Dict[str, Any], vocab_size: int, device: str) -> None:
        self.gamma = float(cfg.get("gamma", 0.5))
        self.delta = float(cfg.get("delta", 2.0))
        self.hash_key = int(cfg.get("hash_key", 15485863))
        self.z_threshold = float(cfg.get("z_threshold", 4.0))
        self.prefix_length = int(cfg.get("prefix_length", 4))
        self.f_scheme = cfg.get("f_scheme", "additive")
        self.window_scheme = cfg.get("window_scheme", "left")
        self.vocab_size = int(vocab_size)
        self.device = device
        self.gen_kwargs = {}
        self.generation_model = None
        self.generation_tokenizer = None


def main() -> None:
    args = parse_args()
    run_meta = load_json(args.run_meta)
    kgw_cfg = load_json(args.kgw_config)

    tok = AutoTokenizer.from_pretrained(run_meta["model"])
    kgw_config_obj = SimpleKGWConfig(kgw_cfg, vocab_size=len(tok), device=args.device)
    kgw_utils = KGWUtils(kgw_config_obj)

    df = pd.read_csv(args.input_csv)
    if args.n_samples:
        df = df.head(args.n_samples)

    overlaps: List[float] = []
    ious: List[float] = []
    per_sample_mean: List[float] = []

    for row_idx, row in tqdm(list(df.iterrows()), desc="Samples"):
        prompt = row.get("prompt_text", "")
        full = row["full_text"]
        attacked = apply_char_attack(full, prompt, args.attack_ratio, seed=args.attack_seed + row_idx)

        enc_prompt = tok(prompt, return_tensors="pt", add_special_tokens=False)
        prompt_len = enc_prompt["input_ids"][0].numel()

        ids_c = tok(full, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(args.device)
        ids_a = tok(attacked, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(args.device)

        L = min(ids_c.numel(), ids_a.numel())
        start = max(prompt_len, kgw_config_obj.prefix_length)
        sample_overlaps: List[float] = []
        sample_ious: List[float] = []

        for pos in range(start, L, max(1, args.step_stride)):
            prefix_c = ids_c[:pos]
            prefix_a = ids_a[:pos]
            gl_c = torch.tensor(kgw_utils.get_greenlist_ids(prefix_c), device="cpu")
            gl_a = torch.tensor(kgw_utils.get_greenlist_ids(prefix_a), device="cpu")
            if gl_c.numel() == 0:
                continue
            inter = len(np.intersect1d(gl_c.numpy(), gl_a.numpy()))
            union = len(np.union1d(gl_c.numpy(), gl_a.numpy()))
            overlap = inter / len(gl_c) if len(gl_c) > 0 else 0.0
            iou = inter / union if union > 0 else 0.0
            sample_overlaps.append(overlap)
            sample_ious.append(iou)

        if sample_overlaps:
            per_sample_mean.append(float(np.mean(sample_overlaps)))
            overlaps.extend(sample_overlaps)
            ious.extend(sample_ious)

    if overlaps:
        print(f"Mean overlap (|Gc∩Ga|/|Gc|): {np.mean(overlaps):.4f}")
        print(f"Mean IoU (|Gc∩Ga|/|Gc∪Ga|): {np.mean(ious):.4f}")
        print(f"Per-sample overlap mean: mean={np.mean(per_sample_mean):.4f} std={np.std(per_sample_mean):.4f}")
    else:
        print("No tokens processed.")


if __name__ == "__main__":
    main()
