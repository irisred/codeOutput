#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
import random
import inspect
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

from MarkLLM.watermark.kgw.kgw import KGW
from MarkLLM.watermark.bytekgwV5 import ByteKGWv5


# -----------------------------
# Utils
# -----------------------------
def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_dtype(name: str) -> torch.dtype:
    name = name.lower().strip()
    if name in ("fp16", "float16", "half"):
        return torch.float16
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def ensure_pad_token(tokenizer) -> None:
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token


def load_prompts(prompt: str, prompt_file: Optional[str]) -> List[str]:
    if prompt_file:
        with open(prompt_file, "r", encoding="utf-8") as f:
            ps = [ln.strip() for ln in f if ln.strip()]
        if not ps:
            raise ValueError("prompt_file is empty")
        return ps
    return [prompt]


def parse_fpr_list(s: str) -> List[float]:
    xs = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        x = float(part)
        if not (0.0 < x < 1.0):
            raise ValueError(f"FPR must be in (0,1), got {x}")
        xs.append(x)
    if not xs:
        raise ValueError("Empty --fpr_list")
    return xs


# -----------------------------
# PPL on continuation
# -----------------------------
@torch.no_grad()
def ppl_continuation(
    model,
    tokenizer,
    prompt: str,
    full_text: str,
    device: torch.device,
    *,
    add_special_tokens: bool,
) -> float:
    """
    PPL only on continuation (tokens after prompt).
    Robust-ish alignment: try strict prefix, else search near beginning.
    """
    model.eval()

    p_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=add_special_tokens)["input_ids"][0].to(device)
    x_ids = tokenizer(full_text, return_tensors="pt", add_special_tokens=add_special_tokens)["input_ids"][0].to(device)

    if x_ids.numel() < 2:
        return float("nan")

    Lp = int(p_ids.numel())

    # find prompt in x_ids near the beginning
    offset = 0
    if x_ids.numel() >= Lp and torch.equal(x_ids[:Lp], p_ids):
        offset = 0
    else:
        found = None
        max_start = min(32, int(x_ids.numel() - Lp))
        for s in range(max_start + 1):
            if torch.equal(x_ids[s:s + Lp], p_ids):
                found = s
                break
        offset = int(found) if found is not None else 0

    prompt_len = offset + Lp
    if x_ids.numel() <= prompt_len:
        return float("nan")

    input_ids = x_ids.unsqueeze(0)
    out = model(input_ids=input_ids, use_cache=False)

    logits = out.logits[:, :-1, :]
    labels = input_ids[:, 1:]

    start = max(prompt_len - 1, 0)  # label index aligned to first continuation token
    logits = logits[:, start:, :]
    labels = labels[:, start:]

    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        reduction="mean",
    )
    return float(torch.exp(loss).item())


# -----------------------------
# Robust calls (signature-adaptive)
# -----------------------------
def call_generate(alg, prompt: str, *, seed: int, add_special_tokens: bool, watermarked: bool) -> str:
    """
    ByteKGWv5: has generate_unwatermarked_text / generate_watermarked_text with optional seed.
    KGW: only generate_watermarked_text (we’ll temporarily set delta=0 for unwatermarked).
    """
    if (not watermarked) and hasattr(alg, "generate_unwatermarked_text"):
        fn = getattr(alg, "generate_unwatermarked_text")
    else:
        fn = getattr(alg, "generate_watermarked_text")

    sig = inspect.signature(fn)
    kwargs = {}
    if "seed" in sig.parameters:
        kwargs["seed"] = int(seed)
    if "add_special_tokens" in sig.parameters:
        kwargs["add_special_tokens"] = bool(add_special_tokens)
    if "skip_special_tokens" in sig.parameters:
        kwargs["skip_special_tokens"] = True
    return fn(prompt, **kwargs)


def call_detect(alg, text: str, *, add_special_tokens: bool) -> Dict[str, Any]:
    fn = getattr(alg, "detect_watermark")
    sig = inspect.signature(fn)
    kwargs = {}
    if "add_special_tokens" in sig.parameters:
        kwargs["add_special_tokens"] = bool(add_special_tokens)
    if "return_dict" in sig.parameters:
        kwargs["return_dict"] = True
    out = fn(text, **kwargs)
    if isinstance(out, dict):
        return out
    if isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], dict):
        return out[1]
    raise TypeError(f"Unexpected detect_watermark return type: {type(out)}")


# -----------------------------
# Threshold for a target FPR (strict rule: z > thr)
#   - if no ties: choose thr between k-th and (k+1)-th largest to make exact k FPs
#   - if ties: fall back to thr = k-th largest, report achieved_fpr
# -----------------------------
def threshold_for_target_fpr_strict_gt(z_neg: List[float], target_fpr: float) -> float:
    if not z_neg:
        return float("inf")
    n = len(z_neg)
    k = int(math.ceil(target_fpr * n))  # desired FP count (approx)
    k = max(0, min(k, n))

    zs = sorted(z_neg, reverse=True)  # desc

    if k == 0:
        # want 0 FP: set thr = max(z); with strict '>' gives FP=0
        return zs[0]

    if k >= n:
        # want all FP: set thr < min(z)
        return zs[-1] - 1e-6

    a = zs[k - 1]  # k-th largest (1-indexed)
    b = zs[k]      # (k+1)-th largest
    if a != b:
        # pick midpoint so exactly k values are > thr
        return 0.5 * (a + b)
    # tie: cannot hit exact k with deterministic threshold; choose boundary value
    return a


def rate_strict_gt(zs: List[float], thr: float) -> float:
    return sum(1 for z in zs if z > thr) / len(zs) if zs else float("nan")


# -----------------------------
# TransformersConfig adapter
# -----------------------------
@dataclass
class MiniTransformersConfig:
    model: Any
    tokenizer: Any
    vocab_size: int
    device: torch.device
    gen_kwargs: Dict[str, Any]


def build_tf_cfg(model, tokenizer, device: torch.device, gen_kwargs: Dict[str, Any]) -> MiniTransformersConfig:
    V = int(model.get_input_embeddings().weight.shape[0])
    return MiniTransformersConfig(
        model=model,
        tokenizer=tokenizer,
        vocab_size=V,
        device=device,
        gen_kwargs=gen_kwargs,
    )


# -----------------------------
# Eval one algorithm
# -----------------------------
@torch.no_grad()
def eval_algorithm(
    name: str,
    alg,
    *,
    z_key: str,
    prompts: List[str],
    n: int,
    base_seed: int,
    delta: float,
    fpr_targets: List[float],
    model,
    tokenizer,
    device: torch.device,
    add_special_tokens_prompt: bool,
) -> Dict[str, Any]:
    # set delta for watermark generation
    if hasattr(alg, "config") and hasattr(alg.config, "delta"):
        alg.config.delta = float(delta)

    z_neg: List[float] = []
    z_pos: List[float] = []
    ppl_neg: List[float] = []
    ppl_pos: List[float] = []

    for i in range(n):
        seed = int(base_seed + i)
        prompt = prompts[i % len(prompts)]

        # ---- negative (unwatermarked) ----
        if not hasattr(alg, "generate_unwatermarked_text"):
            # KGW: temporarily delta=0
            old_delta = float(getattr(alg.config, "delta", delta))
            try:
                alg.config.delta = 0.0
                set_all_seeds(seed)
                txt0 = call_generate(alg, prompt, seed=seed, add_special_tokens=add_special_tokens_prompt, watermarked=True)
            finally:
                alg.config.delta = old_delta
        else:
            set_all_seeds(seed)
            txt0 = call_generate(alg, prompt, seed=seed, add_special_tokens=add_special_tokens_prompt, watermarked=False)

        det0 = call_detect(alg, txt0, add_special_tokens=False)
        z0 = float(det0[z_key])
        p0 = ppl_continuation(model, tokenizer, prompt, txt0, device, add_special_tokens=add_special_tokens_prompt)
        z_neg.append(z0)
        ppl_neg.append(p0)

        # ---- positive (watermarked) ----
        if hasattr(alg, "config") and hasattr(alg.config, "delta"):
            alg.config.delta = float(delta)
        set_all_seeds(seed)
        txt1 = call_generate(alg, prompt, seed=seed, add_special_tokens=add_special_tokens_prompt, watermarked=True)

        det1 = call_detect(alg, txt1, add_special_tokens=False)
        z1 = float(det1[z_key])
        p1 = ppl_continuation(model, tokenizer, prompt, txt1, device, add_special_tokens=add_special_tokens_prompt)
        z_pos.append(z1)
        ppl_pos.append(p1)

    # ---- multi-FPR report ----
    rows = []
    for fpr_t in fpr_targets:
        thr = threshold_for_target_fpr_strict_gt(z_neg, fpr_t)
        fpr = rate_strict_gt(z_neg, thr)
        tpr = rate_strict_gt(z_pos, thr)
        rows.append((fpr_t, fpr, thr, tpr))

    return {
        "name": name,
        "n": n,
        "delta": float(delta),
        "ppl_neg_mean": float(sum(ppl_neg) / len(ppl_neg)),
        "ppl_pos_mean": float(sum(ppl_pos) / len(ppl_pos)),
        "z_neg": z_neg,
        "z_pos": z_pos,
        "rows": rows,  # (target_fpr, achieved_fpr, thr, tpr)
    }


def print_report(res: Dict[str, Any]) -> None:
    print(f"[{res['name']}] n={res['n']} delta={res['delta']}")
    print(f"  PPL mean (neg/unwm) = {res['ppl_neg_mean']:.4f}")
    print(f"  PPL mean (pos/wm)  = {res['ppl_pos_mean']:.4f}")
    print("  target_fpr  achieved_fpr  threshold(z)   tpr")
    for (tf, af, thr, tpr) in res["rows"]:
        print(f"   {tf:8.3f}     {af:8.3f}     {thr:10.4f}  {tpr:6.3f}")
    print(f"  (note) FPR resolution is 1/{res['n']}={1/res['n']:.3f}; ties may cause achieved_fpr != target_fpr.")


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, type=str)
    ap.add_argument("--device", default="cuda:0", type=str)
    ap.add_argument("--dtype", default="float16", type=str)

    ap.add_argument("--prompt", default="Hello, my name is", type=str)
    ap.add_argument("--prompt_file", default=None, type=str)

    ap.add_argument("--do_sample", action="store_true")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--repetition_penalty", type=float, default=1.0)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--add_special_tokens", action="store_true")

    ap.add_argument("--delta", type=float, default=1.0)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--fpr_list", type=str, default="0.02,0.05,0.10,0.20,0.30")

    ap.add_argument("--kgw_config", required=True, type=str)
    ap.add_argument("--bytekgw_config", required=True, type=str)

    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = parse_dtype(args.dtype)
    fpr_targets = parse_fpr_list(args.fpr_list)

    set_all_seeds(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    ensure_pad_token(tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map=None,
    ).to(device)
    model.eval()

    gen_kwargs = dict(
        do_sample=bool(args.do_sample),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        top_k=int(args.top_k),
        repetition_penalty=float(args.repetition_penalty),
        max_new_tokens=int(args.max_new_tokens),
        pad_token_id=int(tokenizer.pad_token_id) if tokenizer.pad_token_id is not None else None,
        eos_token_id=int(tokenizer.eos_token_id) if tokenizer.eos_token_id is not None else None,
        use_cache=True,
    )
    gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

    tf_cfg = build_tf_cfg(model, tokenizer, device, gen_kwargs)

    kgw = KGW(args.kgw_config, tf_cfg)
    bkgw = ByteKGWv5(args.bytekgw_config, tf_cfg)

    prompts = load_prompts(args.prompt, args.prompt_file)

    print("=" * 120)
    print(f"[model] {args.model}")
    print(f"[device] {device} dtype={dtype}")
    print(f"[eval] n={args.n} seed={args.seed} delta={args.delta}")
    print(f"[gen] do_sample={args.do_sample} temp={args.temperature} top_p={args.top_p} top_k={args.top_k} "
          f"rep_penalty={args.repetition_penalty} max_new_tokens={args.max_new_tokens} add_special_tokens={args.add_special_tokens}")
    print(f"[fpr_list] {fpr_targets}")
    print("=" * 120)

    # KGW detector z in "score"
    res_kgw = eval_algorithm(
        "KGW", kgw,
        z_key="score",
        prompts=prompts,
        n=args.n,
        base_seed=args.seed,
        delta=args.delta,
        fpr_targets=fpr_targets,
        model=model,
        tokenizer=tokenizer,
        device=device,
        add_special_tokens_prompt=bool(args.add_special_tokens),
    )

    # ByteKGWv5 detector z in "z"
    res_bkgw = eval_algorithm(
        "ByteKGWv5", bkgw,
        z_key="z",
        prompts=prompts,
        n=args.n,
        base_seed=args.seed,
        delta=args.delta,
        fpr_targets=fpr_targets,
        model=model,
        tokenizer=tokenizer,
        device=device,
        add_special_tokens_prompt=bool(args.add_special_tokens),
    )

    print_report(res_kgw)
    print("-" * 120)
    print_report(res_bkgw)
    print("=" * 120)


if __name__ == "__main__":
    main()
