#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Add conditional PPL column to CSVs (in-place).
PPL is computed on continuation conditioned on prompt_text (labels for prompt tokens are ignored).

Usage example:
  TOKENIZERS_PARALLELISM=false /home/star/jf/python/stega/bin/python scripts/add_ppl_inplace.py \
    --run_meta outputs/c4_samples_head_200/run_metadata.json \
    --input_glob "outputs/wm_eval/**/*.csv" \
    --device cuda:0 \
    --batch_size 4
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import List, Dict

import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_meta", required=True)
    ap.add_argument("--input_glob", required=True, help='e.g. "outputs/wm_eval/**/*.csv"')
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch_size", type=int, default=4)
    return ap.parse_args()


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def cond_ppl_batch(model, tokenizer, prompts: List[str], conts: List[str], device: str, add_special_tokens: bool) -> List[float]:
    """
    Compute conditional perplexity for each (prompt, continuation) pair.
    Only continuation tokens are counted in loss (prompt tokens masked with -100).
    """
    texts = [p + c for p, c in zip(prompts, conts)]
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=False,
        add_special_tokens=add_special_tokens,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    # prompt lengths in tokens
    prompt_lens = []
    for p in prompts:
        pl = tokenizer(
            p,
            return_tensors="pt",
            add_special_tokens=add_special_tokens,
        )["input_ids"].shape[1]
        prompt_lens.append(pl)

    labels = input_ids.clone()
    for i, pl in enumerate(prompt_lens):
        labels[i, :pl] = -100  # ignore prompt tokens

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    loss_flat = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
    )
    loss_flat = loss_flat.view(shift_labels.shape)

    ppl_list: List[float] = []
    for i in range(input_ids.size(0)):
        mask = shift_labels[i] != -100
        if mask.sum() == 0:
            ppl_list.append(float("inf"))
            continue
        nll = loss_flat[i][mask].sum().item()
        avg_nll = nll / mask.sum().item()
        ppl_list.append(float(torch.exp(torch.tensor(avg_nll)).item()))
    return ppl_list


def main() -> None:
    args = parse_args()
    run_meta = load_json(args.run_meta)
    model_path = run_meta["model"]
    add_special_tokens = bool(run_meta.get("add_special_tokens", True))

    device = args.device
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device.startswith("cuda") else None,
    ).to(device)
    model.eval()

    paths = [Path(p) for p in glob.glob(args.input_glob, recursive=True)]
    for path in paths:
        df = pd.read_csv(path)
        if "ppl" in df.columns:
            continue
        if "prompt_text" not in df.columns:
            print(f"[skip] no prompt_text in {path}")
            continue
        if "continuation_text" in df.columns:
            conts = df["continuation_text"].fillna("").tolist()
        elif "full_text" in df.columns:
            conts = []
            pts = df["prompt_text"].fillna("").tolist()
            fulls = df["full_text"].fillna("").tolist()
            for p, f in zip(pts, fulls):
                conts.append(f[len(p):] if f.startswith(p) else f)
            df["continuation_text"] = conts
        else:
            print(f"[skip] no continuation/full_text in {path}")
            continue

        prompts = df["prompt_text"].fillna("").tolist()
        ppl_vals: List[float] = []
        bs = max(1, args.batch_size)
        for i in tqdm(range(0, len(df), bs), desc=f"ppl {path.name}", leave=False):
            batch_prompts = prompts[i:i+bs]
            batch_conts = conts[i:i+bs]
            ppl_vals.extend(cond_ppl_batch(model, tok, batch_prompts, batch_conts, device, add_special_tokens))

        df["ppl"] = ppl_vals
        df.to_csv(path, index=False)
        print(f"[done] wrote ppl to {path} (rows={len(df)})")


if __name__ == "__main__":
    main()
