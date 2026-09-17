#!/usr/bin/env python3
"""
Compute conditional perplexity (continuation given prompt) for every row
in a watermark CSV file and write the results back to disk.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import List

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compute conditional PPL for CSV samples.")
    ap.add_argument("csv", help="Input CSV file containing prompt_id and text columns.")
    ap.add_argument(
        "--prompts",
        default="data/prompts_c4.txt",
        help="Prompt file used during generation (one prompt per line).",
    )
    ap.add_argument(
        "--prompt-trim",
        type=int,
        default=64,
        help="Prompt trim length used during generation (0 = no trim).",
    )
    ap.add_argument(
        "--model",
        default="../Meta-Llama-3-8B",
        help="HF model path or identifier for computing PPL.",
    )
    ap.add_argument("--device", default="cuda:0", help="Device for the model (e.g., cuda:0 or cpu).")
    ap.add_argument(
        "--output",
        default=None,
        help="Output CSV path. Defaults to <input>_with_ppl.csv in the same directory.",
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


def conditional_ppl(model, tokenizer, prompt: str, full_text: str, device: str) -> float:
    full = tokenizer(full_text, return_tensors="pt", add_special_tokens=True).to(device)
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(device)["input_ids"]
    input_ids = full["input_ids"]
    attention_mask = full.get("attention_mask")

    labels = input_ids.clone()
    prompt_len = min(prompt_ids.shape[1], labels.shape[1])
    labels[:, :prompt_len] = -100

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
    return math.exp(loss.item())


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"Input CSV {csv_path} does not exist.")
    output_path = Path(args.output) if args.output else csv_path.with_name(f"{csv_path.stem}_with_ppl.csv")

    prompts = load_prompts(Path(args.prompts), args.prompt_trim)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code="qwen" in args.model.lower())
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    torch_dtype = torch.float16 if args.device.startswith("cuda") else None
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        trust_remote_code="qwen" in args.model.lower(),
    ).to(args.device)
    model.eval()

    rows = []
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "prompt_id" not in reader.fieldnames or "text" not in reader.fieldnames:
            raise SystemExit("CSV must contain prompt_id and text columns.")
        for row in reader:
            rows.append(row)

    for row in tqdm(rows, desc="Computing conditional PPL"):
        try:
            prompt_idx = int(row.get("prompt_id", 0))
        except Exception:
            prompt_idx = 0
        if not (0 <= prompt_idx < len(prompts)):
            prompt = prompts[0]
        else:
            prompt = prompts[prompt_idx]
        text = (row.get("text") or "").strip()
        if not text:
            row["cond_ppl"] = ""
            continue
        try:
            ppl = conditional_ppl(model, tokenizer, prompt, text, args.device)
        except Exception as exc:
            row["cond_ppl"] = ""
            print(f"[warn] Failed to compute PPL for prompt_id={prompt_idx}: {exc}")
            continue
        row["cond_ppl"] = f"{ppl:.6f}"

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(rows[0].keys()) if rows else []
        if "cond_ppl" not in fieldnames:
            fieldnames.append("cond_ppl")
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"[done] Wrote conditional PPL to {output_path}")


if __name__ == "__main__":
    main()
