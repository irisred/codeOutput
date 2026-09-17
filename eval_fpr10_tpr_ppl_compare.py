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
            prompts = [ln.strip() for ln in f if ln.strip()]
        if not prompts:
            raise ValueError("prompt_file is empty")
        return prompts
    return [prompt]


def threshold_for_fpr10_strict_gt(z_neg: List[float]) -> float:
    """
    n=10 时，FPR=10% 对应允许 1 个 FP。
    使用规则: predict = (z > thr)
    取 thr 为 “第二大值”，这样只有最大值会 > thr -> FP=1/10（若无并列）。
    """
    if not z_neg:
        return float("inf")
    zs = sorted(z_neg, reverse=True)
    if len(zs) == 1:
        return zs[0] - 1e-6
    return zs[1]  # second largest


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
    计算 continuation perplexity（prompt 之后的部分）。
    用 tokenizer 对 prompt 和 full_text 分别编码，尽量找 prompt 在 full_text token 序列中的对齐位置。
    """
    model.eval()

    p_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=add_special_tokens)["input_ids"][0].to(device)
    x_ids = tokenizer(full_text, return_tensors="pt", add_special_tokens=add_special_tokens)["input_ids"][0].to(device)

    if x_ids.numel() < 2:
        return float("nan")

    Lp = int(p_ids.numel())
    offset = 0

    # strict prefix
    if x_ids.numel() >= Lp and torch.equal(x_ids[:Lp], p_ids):
        offset = 0
    else:
        # find near beginning
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

    input_ids = x_ids.unsqueeze(0)  # [1,T]
    out = model(input_ids=input_ids, use_cache=False)
    logits = out.logits[:, :-1, :]      # predicts token t+1
    labels = input_ids[:, 1:]           # target token t+1

    # continuation starts at token index prompt_len, corresponding label index is prompt_len-1
    start = max(prompt_len - 1, 0)
    logits = logits[:, start:, :]
    labels = labels[:, start:]

    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        reduction="mean",
    )
    return float(torch.exp(loss).item())


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
# Robust call helpers (handle optional signature params)
# -----------------------------
def call_generate(alg, prompt: str, *, seed: int, add_special_tokens: bool, watermarked: bool) -> str:
    """
    调用算法的生成方法：优先使用 generate_unwatermarked_text（若存在且 watermarked=False），否则用 generate_watermarked_text。
    自动适配不同签名：seed / add_special_tokens / skip_special_tokens 等可选参数。
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
    """
    调用 detect_watermark，要求 return_dict=True（若支持）。
    自动适配 add_special_tokens / return_dict 参数。
    """
    fn = getattr(alg, "detect_watermark")
    sig = inspect.signature(fn)
    kwargs = {}
    if "add_special_tokens" in sig.parameters:
        kwargs["add_special_tokens"] = bool(add_special_tokens)
    if "return_dict" in sig.parameters:
        kwargs["return_dict"] = True
    out = fn(text, **kwargs)
    # 有些实现可能直接返回 dict；有些返回 tuple，这里尽量规整
    if isinstance(out, dict):
        return out
    if isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], dict):
        return out[1]
    raise TypeError(f"Unexpected detect_watermark return type: {type(out)}")


# -----------------------------
# Eval
# -----------------------------
@torch.no_grad()
def eval_one_algorithm(
    name: str,
    alg,
    *,
    z_key: str,
    prompts: List[str],
    n: int,
    base_seed: int,
    delta: float,
    target_fpr: float,
    model,
    tokenizer,
    device: torch.device,
    add_special_tokens_prompt: bool,
) -> Dict[str, Any]:
    """
    生成 n 个 negative (delta=0) + n 个 positive (delta=delta)，
    用 negative 标定 FPR=10% 阈值，再算 positive 的 TPR。
    计算两组 PPL 均值（continuation ppl）。
    """
    # set delta
    if hasattr(alg, "config") and hasattr(alg.config, "delta"):
        alg.config.delta = float(delta)

    z_neg, z_pos = [], []
    ppl_neg, ppl_pos = [], []

    for i in range(n):
        seed = int(base_seed + i)
        prompt = prompts[i % len(prompts)]

        # ---- negative: delta=0 ----
        # 对于没有 generate_unwatermarked_text 的算法（KGW），临时将 delta=0
        if not hasattr(alg, "generate_unwatermarked_text"):
            old_delta = float(getattr(alg.config, "delta", delta))
            try:
                alg.config.delta = 0.0
                set_all_seeds(seed)  # IMPORTANT: don't use generator kwarg; use global RNG
                txt0 = call_generate(alg, prompt, seed=seed, add_special_tokens=add_special_tokens_prompt, watermarked=True)
            finally:
                alg.config.delta = old_delta
        else:
            set_all_seeds(seed)
            txt0 = call_generate(alg, prompt, seed=seed, add_special_tokens=add_special_tokens_prompt, watermarked=False)

        det0 = call_detect(alg, txt0, add_special_tokens=False)  # detector 通常用 False 更一致
        z0 = float(det0[z_key])
        p0 = ppl_continuation(model, tokenizer, prompt, txt0, device, add_special_tokens=add_special_tokens_prompt)
        z_neg.append(z0)
        ppl_neg.append(p0)

        # ---- positive: delta=delta ----
        if hasattr(alg, "config") and hasattr(alg.config, "delta"):
            alg.config.delta = float(delta)
        set_all_seeds(seed)
        txt1 = call_generate(alg, prompt, seed=seed, add_special_tokens=add_special_tokens_prompt, watermarked=True)

        det1 = call_detect(alg, txt1, add_special_tokens=False)
        z1 = float(det1[z_key])
        p1 = ppl_continuation(model, tokenizer, prompt, txt1, device, add_special_tokens=add_special_tokens_prompt)
        z_pos.append(z1)
        ppl_pos.append(p1)

    # ---- calibrate threshold on negatives ----
    # n=10, target_fpr=0.1 -> allow 1 FP; use strict rule z > thr
    thr = threshold_for_fpr10_strict_gt(z_neg)
    fpr = sum(1 for z in z_neg if z > thr) / len(z_neg)
    tpr = sum(1 for z in z_pos if z > thr) / len(z_pos)

    return {
        "name": name,
        "delta": float(delta),
        "n": int(n),
        "target_fpr": float(target_fpr),
        "threshold": float(thr),
        "fpr": float(fpr),
        "tpr": float(tpr),
        "z_neg": z_neg,
        "z_pos": z_pos,
        "ppl_neg_mean": float(sum(ppl_neg) / len(ppl_neg)),
        "ppl_pos_mean": float(sum(ppl_pos) / len(ppl_pos)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--dtype", type=str, default="float16")

    ap.add_argument("--prompt", type=str, default="Hello, my name is")
    ap.add_argument("--prompt_file", type=str, default=None)

    # generation knobs
    ap.add_argument("--do_sample", action="store_true")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--repetition_penalty", type=float, default=1.0)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--add_special_tokens", action="store_true")

    # eval knobs
    ap.add_argument("--delta", type=float, default=1.0)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--target_fpr", type=float, default=0.10)

    # config paths
    ap.add_argument("--kgw_config", type=str, required=True)
    ap.add_argument("--bytekgw_config", type=str, required=True)
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = parse_dtype(args.dtype)

    set_all_seeds(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    ensure_pad_token(tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map=None,
    ).to(device)
    model.eval()

    # gen kwargs (NO generator kwarg, transformers may reject it)
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
    print(f"[eval] n={args.n} seed={args.seed} target_fpr={args.target_fpr} delta={args.delta}")
    print(f"[gen] do_sample={args.do_sample} temp={args.temperature} top_p={args.top_p} top_k={args.top_k} "
          f"rep_penalty={args.repetition_penalty} max_new_tokens={args.max_new_tokens} add_special_tokens={args.add_special_tokens}")
    print("=" * 120)

    # KGW detector z in key "score"
    res_kgw = eval_one_algorithm(
        "KGW", kgw,
        z_key="score",
        prompts=prompts,
        n=args.n,
        base_seed=args.seed,
        delta=args.delta,
        target_fpr=args.target_fpr,
        model=model,
        tokenizer=tokenizer,
        device=device,
        add_special_tokens_prompt=bool(args.add_special_tokens),
    )

    # ByteKGWv5 detector z in key "z"
    res_bkgw = eval_one_algorithm(
        "ByteKGWv5", bkgw,
        z_key="z",
        prompts=prompts,
        n=args.n,
        base_seed=args.seed,
        delta=args.delta,
        target_fpr=args.target_fpr,
        model=model,
        tokenizer=tokenizer,
        device=device,
        add_special_tokens_prompt=bool(args.add_special_tokens),
    )

    def show(r: Dict[str, Any]) -> None:
        print(f"[{r['name']}]")
        print(f"  threshold@FPR{r['target_fpr']:.2f} = {r['threshold']:.4f}   (rule: z > thr)")
        print(f"  FPR = {r['fpr']:.3f}   TPR = {r['tpr']:.3f}")
        print(f"  PPL mean (neg/unwm) = {r['ppl_neg_mean']:.4f}")
        print(f"  PPL mean (pos/wm)  = {r['ppl_pos_mean']:.4f}")
        print(f"  z_neg = {[round(x,4) for x in r['z_neg']]}")
        print(f"  z_pos = {[round(x,4) for x in r['z_pos']]}")

    show(res_kgw)
    print("-" * 120)
    show(res_bkgw)
    print("=" * 120)


if __name__ == "__main__":
    main()
