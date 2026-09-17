"""
Quick check: delta=0 (i.e., plain HF generate) should reproduce hf_generate.csv rows.

This script reloads the model/tokenizer, re-generates a handful of prompts with the
same gen params + seeds as recorded in outputs/.../hf_generate.csv, and reports
whether the generated text matches the CSV.

Usage:
  TOKENIZERS_PARALLELISM=false python scripts/test_v6_delta0_vs_hf_generate.py \
    --csv outputs/c4_samples_head_200/hf_generate.csv \
    --run_meta outputs/c4_samples_head_200/run_metadata.json \
    --n 5 \
    --device cuda:0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="hf_generate.csv path")
    ap.add_argument("--run_meta", required=True, help="run_metadata.json path")
    ap.add_argument("--n", type=int, default=5, help="number of rows to verify")
    ap.add_argument("--device", default=None, help="override device (default uses run_meta.devices[0])")
    return ap.parse_args()


def load_model(model_path: str, device: str):
    tok = AutoTokenizer.from_pretrained(model_path)
    mdl = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16 if "cuda" in device else None)
    mdl = mdl.to(device)
    mdl.eval()
    return tok, mdl


def generate_one(tokenizer, model, prompt: str, gen_params: Dict[str, Any], seed: int, device: str) -> str:
    torch.manual_seed(int(seed))
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(int(seed))

    add_special_tokens = gen_params.pop("add_special_tokens", True)
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=add_special_tokens).to(device)
    output_ids = model.generate(**encoded, **gen_params)
    return tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]


def main():
    args = parse_args()
    df = pd.read_csv(args.csv)

    with open(args.run_meta, "r", encoding="utf-8") as f:
        meta = json.load(f)

    model_path = meta.get("model") or df["model_path"].iloc[0]
    device = args.device or (meta.get("devices") or ["cpu"])[0]

    tokenizer, model = load_model(model_path, device)

    rows = df.head(args.n)
    mismatches = 0
    for _, row in rows.iterrows():
        gen_params = json.loads(row["gen_params_json"])
        # ensure int types for ids
        for k in ("eos_token_id", "pad_token_id"):
            if k in gen_params and pd.notna(gen_params[k]):
                gen_params[k] = int(gen_params[k])
        prompt = row["prompt_text"]
        seed = int(row["seed"])
        got = generate_one(tokenizer, model, prompt, gen_params.copy(), seed, device)
        ref = row["full_text"]
        ok = got.strip() == ref.strip()
        if not ok:
            mismatches += 1
            print(f"[mismatch] prompt_id={row['prompt_id']} seed={seed}")
            print("  expected:", ref[:120].replace("\n", " "))
            print("  got     :", got[:120].replace("\n", " "))
    if mismatches == 0:
        print(f"All {len(rows)} checked rows match hf_generate.csv")
    else:
        print(f"{mismatches}/{len(rows)} rows mismatched")


if __name__ == "__main__":
    main()
