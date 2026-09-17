#!/usr/bin/env python
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from bytekgwV5 import ByteKGWv5, ByteKGWv5Config, GenerationConfigV5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="float16", choices=["float16","bfloat16","float32"])

    ap.add_argument("--scheme", default="byte_tree", choices=["token","byte_tree"])
    ap.add_argument("--delta", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--hash_key", type=int, default=15485863)
    ap.add_argument("--prefix_length", type=int, default=4)
    ap.add_argument("--max_byte_pos", type=int, default=16)

    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--do_sample", action="store_true")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--repetition_penalty", type=float, default=1.0)
    ap.add_argument("--add_special_tokens", action="store_true")

    ap.add_argument("--use_generator", action="store_true")
    ap.add_argument("--seed", type=int, default=1234)
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
        use_torch_generator=bool(args.use_generator),
        seed=int(args.seed),
    )

    wm = ByteKGWv5Config(
        gamma=float(args.gamma),
        delta=float(args.delta),
        hash_key=int(args.hash_key),
        prefix_length=int(args.prefix_length),
        sampling_scheme=str(args.scheme),
        max_byte_pos=int(args.max_byte_pos),
    )

    if not args.use_generator:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    v5 = ByteKGWv5(model, tok, wm_cfg=wm, gen_cfg=gen, device=device)
    txt = v5.generate_text(args.prompt, add_special_tokens=args.add_special_tokens, max_new_tokens=args.max_new_tokens)
    print(txt)


if __name__ == "__main__":
    main()
