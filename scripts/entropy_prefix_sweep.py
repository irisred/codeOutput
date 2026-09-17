#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Estimate how much entropy is concentrated in short token-prefixes.
For a single prompt, generate multiple samples (default 10, max_new_tokens=32),
and for each decoding step compute entropy of:
  - full token distribution
  - prefix distributions for n=1..7 (token string UTF-8 bytes truncated to n)
Outputs:
  - per-step CSV: sample_id,step,n,entropy_prefix,entropy_full
  - summary CSV: n,mean_entropy_prefix,mean_entropy_full,mean_ratio

Example:
TOKENIZERS_PARALLELISM=false /home/star/jf/python/stega/bin/python \
  scripts/entropy_prefix_sweep.py \
  --run_meta outputs/c4_samples_head_200/run_metadata.json \
  --prompt_csv outputs/c4_samples_head_200/hf_generate.csv \
  --prompt_row 0 \
  --device cuda:0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_meta", required=True, help="JSON with model path (run_metadata.json)")
    ap.add_argument(
        "--prompt_csv",
        default="outputs/c4_samples_head_200/hf_generate.csv",
        help="CSV containing prompt_text column",
    )
    ap.add_argument("--prompt_row", type=int, default=0, help="Which row to pick the prompt from")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--output_dir", default="outputs/entropy_prefix_sweep")
    ap.add_argument(
        "--prefix_ns",
        default="1,2,3,4,5,6,7",
        help="comma-separated prefix lengths (bytes of token string)",
    )
    return ap.parse_args()


def softmax_entropy(probs: np.ndarray) -> float:
    probs = probs[probs > 0]
    if probs.size == 0:
        return 0.0
    return float(-np.sum(probs * np.log(probs)))


def main() -> None:
    args = parse_args()
    ns = [int(x) for x in args.prefix_ns.split(",") if x.strip()]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # load run meta + prompt
    with open(args.run_meta, "r", encoding="utf-8") as f:
        run_meta = json.load(f)
    model_path = run_meta["model"]

    df_prompt = pd.read_csv(args.prompt_csv)
    if "prompt_text" not in df_prompt.columns:
        raise ValueError(f"{args.prompt_csv} must contain prompt_text column")
    prompt_text = df_prompt.iloc[args.prompt_row]["prompt_text"]

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    use_gpu = args.device.startswith("cuda") and torch.cuda.is_available()
    load_dtype = torch.float16 if use_gpu else None
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=load_dtype,
    ).to(args.device if use_gpu else "cpu")
    model.eval()

    # precompute token -> prefix bytes for each n
    vocab_size = model.config.vocab_size
    token_strs: List[str] = tokenizer.convert_ids_to_tokens(list(range(vocab_size)))
    prefix_id_maps: Dict[int, np.ndarray] = {}
    for n in ns:
        prefixes: List[bytes] = []
        prefix_ids = np.zeros(vocab_size, dtype=np.int32)
        prefix_to_id: Dict[bytes, int] = {}
        for tid, tstr in enumerate(token_strs):
            b = tstr.encode("utf-8", errors="ignore")
            pref = b[:n]
            if pref not in prefix_to_id:
                prefix_to_id[pref] = len(prefixes)
                prefixes.append(pref)
            prefix_ids[tid] = prefix_to_id[pref]
        prefix_id_maps[n] = prefix_ids

    per_step_records = []

    gen_kwargs = dict(
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
        use_cache=True,
        output_scores=True,
        return_dict_in_generate=True,
    )

    with torch.no_grad():
        for sid in range(args.samples):
            inputs = tokenizer(prompt_text, return_tensors="pt").to(args.device)
            out = model.generate(**inputs, **gen_kwargs)
            scores = out.scores  # list len = max_new_tokens, each [1, vocab]
            for step, logits in enumerate(scores):
                probs = torch.softmax(logits[0], dim=-1).detach().cpu().numpy()
                H_full = softmax_entropy(probs)
                for n in ns:
                    pref_ids = prefix_id_maps[n]
                    prefix_probs = np.bincount(pref_ids, weights=probs, minlength=pref_ids.max() + 1)
                    H_pref = softmax_entropy(prefix_probs)
                    per_step_records.append(
                        {
                            "sample_id": sid,
                            "step": step,
                            "n": n,
                            "entropy_prefix": H_pref,
                            "entropy_full": H_full,
                        }
                    )

    df = pd.DataFrame(per_step_records)
    per_step_path = out_dir / "entropy_per_step.csv"
    df.to_csv(per_step_path, index=False)

    summary = (
        df.groupby("n")
        .agg(
            mean_entropy_prefix=("entropy_prefix", "mean"),
            mean_entropy_full=("entropy_full", "mean"),
        )
        .reset_index()
    )
    summary["mean_ratio"] = summary["mean_entropy_prefix"] / summary["mean_entropy_full"]
    summary_path = out_dir / "entropy_summary.csv"
    summary.to_csv(summary_path, index=False)

    print(f"wrote per-step to {per_step_path} (rows={len(df)})")
    print("summary:")
    print(summary)


if __name__ == "__main__":
    main()
