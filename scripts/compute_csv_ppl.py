#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    ap = argparse.ArgumentParser(description="Compute average PPL for CSV files containing a text column.")
    ap.add_argument("files", nargs="+", help="CSV files to analyze")
    ap.add_argument("--model", default="../Meta-Llama-3-8B")
    ap.add_argument("--device", default="cuda:0")
    return ap.parse_args()


def calc_ppl(model, tokenizer, text, device):
    enc = tokenizer(text, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**enc, labels=enc["input_ids"])
        loss = outputs.loss
    return torch.exp(loss).item()


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code="qwen" in args.model.lower())
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float16 if args.device.startswith("cuda") else None
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        trust_remote_code="qwen" in args.model.lower(),
    ).to(args.device)
    model.eval()

    for file in args.files:
        path = Path(file)
        if not path.exists():
            print(f"[warn] missing {file}")
            continue
        ppl_values = []
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if "text" not in reader.fieldnames:
                raise SystemExit(f"CSV {file} lacks text column")
            for row in reader:
                text = (row.get("text") or "").strip()
                if not text:
                    continue
                ppl = calc_ppl(model, tokenizer, text, args.device)
                ppl_values.append(ppl)
        avg = sum(ppl_values) / len(ppl_values) if ppl_values else float("nan")
        print(f"{file}: avg_ppl={avg:.4f} (N={len(ppl_values)})")


if __name__ == "__main__":
    main()
