#!/usr/bin/env python
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from bytekgwV5 import ByteKGWv5, ByteKGWv5Config, GenerationConfigV5


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def first_mismatch(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return None if len(a) == len(b) else n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--do_sample", action="store_true")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--repetition_penalty", type=float, default=1.0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="float16", choices=["float16","bfloat16","float32"])
    ap.add_argument("--add_special_tokens", action="store_true")
    ap.add_argument("--sampling_dtype", default="model", choices=["model","float32"])
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, device_map=None).to(device)
    model.eval()

    gen = GenerationConfigV5(
        max_new_tokens=args.max_new_tokens,
        do_sample=bool(args.do_sample),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        top_k=int(args.top_k),
        repetition_penalty=float(args.repetition_penalty),
        eos_token_id=int(tok.eos_token_id) if tok.eos_token_id is not None else None,
        pad_token_id=int(tok.pad_token_id) if tok.pad_token_id is not None else (int(tok.eos_token_id) if tok.eos_token_id is not None else None),
        sampling_dtype=args.sampling_dtype,
        use_torch_generator=False,  # use global to match HF most often
        seed=args.seed,
    )

    wm = ByteKGWv5Config(
        gamma=0.5,
        delta=0.0,
        hash_key=15485863,
        prefix_length=4,
        sampling_scheme="token",
        enable_token_level_bias=False,
    )

    # HF baseline
    enc = tok(args.prompt, return_tensors="pt", add_special_tokens=args.add_special_tokens).to(device)
    set_seed(args.seed)
    out = model.generate(
        **enc,
        do_sample=bool(args.do_sample),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        top_k=int(args.top_k),
        repetition_penalty=float(args.repetition_penalty),
        max_new_tokens=int(args.max_new_tokens),
        pad_token_id=gen.pad_token_id,
        eos_token_id=gen.eos_token_id,
    )
    ids_hf = out[0].tolist()
    txt_hf = tok.decode(out[0], skip_special_tokens=False, clean_up_tokenization_spaces=False)

    # v5 token mode
    set_seed(args.seed)
    v5 = ByteKGWv5(model, tok, wm_cfg=wm, gen_cfg=gen, device=device)
    ids_v5 = v5.generate_ids(args.prompt, add_special_tokens=args.add_special_tokens, max_new_tokens=args.max_new_tokens).tolist()
    txt_v5 = tok.decode(torch.tensor(ids_v5), skip_special_tokens=False, clean_up_tokenization_spaces=False)

    m = first_mismatch(ids_hf, ids_v5)
    print("="*110)
    print(f"[gen] do_sample={args.do_sample} temp={args.temperature} top_p={args.top_p} top_k={args.top_k} rep_penalty={args.repetition_penalty}")
    print(f"[encode] add_special_tokens={args.add_special_tokens}")
    print(f"[sampling_dtype] {args.sampling_dtype}")
    print(f"[mismatch_index] {m}")
    print("-"*110)
    print("[HF]")
    print(txt_hf)
    print("-"*110)
    print("[v5 token-mode]")
    print(txt_v5)
    print("="*110)


if __name__ == "__main__":
    main()
