#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import gzip
import inspect
import json
import math
import os
import random
import re
import sys
import tempfile
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


# ------------------------------------------------------------
# Fixed defaults for "fairness"
# ------------------------------------------------------------
USE_PREFIX_BYTES_IN_PRF: bool = True              # 强制：生成/检测 PRF 使用 prefix-bytes（pos内）
BYTEKGW_HEAD_MAX_BYTE_POS: int = 1                # 首字节偏置
BYTEKGW_SCHEME: str = "byte_tree"                 # 你现在在用 byte_tree
CAP_CONT_TOKENS_FOR_EVAL: bool = True             # 检测+PPL都只用前 K 个 continuation token
PROMPT_WORDS: int = 20                            # C4 prompt 取前20词


# ------------------------------------------------------------
# Utils
# ------------------------------------------------------------
def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_dtype(name: str) -> torch.dtype:
    name = name.lower().strip()
    if name in ("float16", "fp16", "half"):
        return torch.float16
    if name in ("bfloat16", "bf16"):
        return torch.bfloat16
    if name in ("float32", "fp32"):
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def ensure_pad_token(tokenizer) -> None:
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_tmp_json(d: Dict[str, Any]) -> str:
    fd, path = tempfile.mkstemp(prefix="wm_cfg_", suffix=".json")
    os.close(fd)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)
    return path


def make_temp_cfg(base_path: str, overrides: Dict[str, Any]) -> str:
    d = read_json(base_path)
    d.update(overrides)
    return write_tmp_json(d)


def find_subsequence(haystack: List[int], needle: List[int]) -> Optional[int]:
    if not needle or len(needle) > len(haystack):
        return None
    m = len(needle)
    for i in range(len(haystack) - m + 1):
        if haystack[i:i + m] == needle:
            return i
    return None


def locate_prompt_end(full_ids: List[int], prompt_ids: List[int]) -> int:
    s = find_subsequence(full_ids, prompt_ids)
    if s is None:
        return len(prompt_ids)
    return s + len(prompt_ids)


def truncate_to_cont(full_ids: List[int], prompt_end: int, cap_cont_tokens: int) -> List[int]:
    keep = min(len(full_ids), prompt_end + int(cap_cont_tokens))
    return full_ids[:keep]


@torch.no_grad()
def ppl_continuation_from_ids(
    model,
    ids: List[int],
    prompt_end: int,
    device: torch.device,
) -> Tuple[float, int]:
    if len(ids) < 2:
        return float("inf"), 0

    input_ids = torch.tensor([ids], device=device, dtype=torch.long)
    attn = torch.ones_like(input_ids)

    out = model(input_ids=input_ids, attention_mask=attn)
    logits = out.logits  # [1,T,V]

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()

    nll = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
    ).view(1, -1)  # [1,T-1]

    start = max(prompt_end - 1, 0)
    if start >= nll.size(1):
        return float("inf"), 0

    cont_nll = nll[:, start:]
    mean_nll = cont_nll.mean().item()
    ppl = float(math.exp(min(50.0, mean_nll)))
    cont_len = max(len(ids) - prompt_end, 0)
    return ppl, cont_len


def _tensor_to_float(x: Any) -> Optional[float]:
    # 注意：bool 是 int 子类，必须排除
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if torch.is_tensor(x) and x.numel() == 1:
        return float(x.detach().cpu().item())
    return None


def extract_stat(det_out: Dict[str, Any]) -> float:
    """
    ByteKGWv5: det_out['z']
    KGW: det_out['score']
    """
    if not isinstance(det_out, dict):
        raise TypeError(f"Detector output must be dict, got {type(det_out)}")

    if "z" in det_out:
        fv = _tensor_to_float(det_out["z"])
        if fv is not None:
            return fv

    if "score" in det_out:
        fv = _tensor_to_float(det_out["score"])
        if fv is not None:
            return fv
        if isinstance(det_out["score"], dict):
            sc = det_out["score"]
            for k in ["z", "z_score", "zstat", "stat", "score"]:
                if k in sc:
                    fv2 = _tensor_to_float(sc[k])
                    if fv2 is not None:
                        return fv2

    # fallback: find any non-bool numeric
    for _, v in det_out.items():
        fv = _tensor_to_float(v)
        if fv is not None:
            return fv

    raise KeyError(f"Cannot extract stat from keys={list(det_out.keys())}")


def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())


def first_n_words(text: str, n: int) -> Optional[str]:
    text = normalize_ws(text)
    if not text:
        return None
    words = text.split(" ")
    if len(words) < n:
        return None
    prompt = " ".join(words[:n]).strip()
    # 末尾补一个空格，避免“续写半个词”的感觉
    return prompt + " "


# ------------------------------------------------------------
# Dataset loader (C4 json.gz)
# ------------------------------------------------------------
@dataclass
class PromptItem:
    prompt_id: int
    source_path: str
    source_line: int
    prompt_text: str
    source_head: str  # 原文前一小段便于追溯


def load_c4_prompts(paths: List[str], n_prompts: int, n_words: int) -> List[PromptItem]:
    out: List[PromptItem] = []
    pid = 0
    for p in paths:
        if not os.path.exists(p):
            continue
        with gzip.open(p, "rt", encoding="utf-8", errors="ignore") as f:
            for line_idx, line in enumerate(f):
                if pid >= n_prompts:
                    return out
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                text = obj.get("text", "")
                if not isinstance(text, str):
                    continue
                prompt = first_n_words(text, n_words)
                if not prompt:
                    continue
                head = normalize_ws(text)[:240]
                out.append(PromptItem(
                    prompt_id=pid,
                    source_path=p,
                    source_line=line_idx,
                    prompt_text=prompt,
                    source_head=head,
                ))
                pid += 1
                if pid >= n_prompts:
                    return out
    return out


# ------------------------------------------------------------
# MarkLLM TransformersConfig (robust + force gen_kwargs)
# ------------------------------------------------------------
def build_transformers_config(model, tokenizer, device_str: str, gen_kwargs: Dict[str, Any]):
    from MarkLLM.utils.transformers_config import TransformersConfig  # noqa

    sig = inspect.signature(TransformersConfig.__init__)
    params = sig.parameters

    kwargs = {}
    if "model" in params:
        kwargs["model"] = model
    elif "generation_model" in params:
        kwargs["generation_model"] = model

    if "tokenizer" in params:
        kwargs["tokenizer"] = tokenizer
    elif "generation_tokenizer" in params:
        kwargs["generation_tokenizer"] = tokenizer

    if "device" in params:
        kwargs["device"] = device_str
    elif "device_str" in params:
        kwargs["device_str"] = device_str

    if "gen_kwargs" in params:
        kwargs["gen_kwargs"] = gen_kwargs
    elif "generation_kwargs" in params:
        kwargs["generation_kwargs"] = gen_kwargs

    if "dtype" in params:
        kwargs["dtype"] = model.dtype

    try:
        tf = TransformersConfig(**kwargs)
    except TypeError:
        try:
            tf = TransformersConfig(model, tokenizer, device_str, gen_kwargs)
        except TypeError:
            tf = TransformersConfig(model, tokenizer)

    # 强制写回，避免 KGW 跑默认 max_length
    if hasattr(tf, "gen_kwargs"):
        tf.gen_kwargs = dict(gen_kwargs)
    if hasattr(tf, "generation_kwargs"):
        tf.generation_kwargs = dict(gen_kwargs)

    # 补齐字段
    if hasattr(tf, "generation_model") and getattr(tf, "generation_model", None) is None:
        tf.generation_model = model
    if hasattr(tf, "generation_tokenizer") and getattr(tf, "generation_tokenizer", None) is None:
        tf.generation_tokenizer = tokenizer
    if hasattr(tf, "tokenizer") and getattr(tf, "tokenizer", None) is None:
        tf.tokenizer = tokenizer
    if hasattr(tf, "model") and getattr(tf, "model", None) is None:
        tf.model = model

    return tf


# ------------------------------------------------------------
# Generation (HF + MarkLLM algorithms)
# ------------------------------------------------------------
@torch.no_grad()
def hf_generate_ids(
    model,
    tokenizer,
    prompt: str,
    device: torch.device,
    seed: int,
    gen_kwargs: Dict[str, Any],
    add_special_tokens: bool,
) -> Tuple[List[int], int]:
    set_all_seeds(seed)
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=bool(add_special_tokens))
    input_ids = enc["input_ids"].to(device)
    attn = enc.get("attention_mask", None)
    if attn is not None:
        attn = attn.to(device)

    out_ids = model.generate(input_ids=input_ids, attention_mask=attn, **gen_kwargs)
    if out_ids.dim() == 2:
        out_ids = out_ids[0]
    prompt_len = int(input_ids.shape[1])
    return out_ids.tolist(), prompt_len


def alg_generate_text(alg, prompt: str, seed: int) -> str:
    set_all_seeds(seed)
    return alg.generate_watermarked_text(prompt)


def tokenize_truncate_for_eval(
    tokenizer,
    full_text: str,
    prompt_text: str,
    add_special_tokens: bool,
    max_new_tokens: int,
) -> Tuple[List[int], int, str, str, bool]:
    """
    Return:
      trunc_ids, prompt_end, trunc_full_text, trunc_cont_text, prompt_found
    """
    prompt_ids = tokenizer(prompt_text, add_special_tokens=bool(add_special_tokens))["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=bool(add_special_tokens))["input_ids"]

    s = find_subsequence(full_ids, prompt_ids)
    prompt_found = s is not None
    prompt_end = (s + len(prompt_ids)) if s is not None else len(prompt_ids)

    if CAP_CONT_TOKENS_FOR_EVAL:
        trunc_ids = truncate_to_cont(full_ids, prompt_end, max_new_tokens)
    else:
        trunc_ids = full_ids

    trunc_full = tokenizer.decode(trunc_ids, skip_special_tokens=True)
    cont_ids = trunc_ids[prompt_end:]
    trunc_cont = tokenizer.decode(cont_ids, skip_special_tokens=True)

    return trunc_ids, prompt_end, trunc_full, trunc_cont, prompt_found


# ------------------------------------------------------------
# Output schema
# ------------------------------------------------------------
CSV_FIELDS = [
    "algo", "delta", "device",
    "model_path", "bytekgw_config", "kgw_config",
    "seed", "prompt_id",
    "c4_source_path", "c4_source_line",
    "prompt_text", "c4_source_head",
    "full_text", "continuation_text",
    "stat_name", "stat_value",
    "ppl",
    "prompt_tokens", "cont_tokens", "full_tokens",
    "prompt_found_in_full",
    "gen_params_json",
]


def row_base(
    *,
    algo: str,
    delta: float,
    device: str,
    model_path: str,
    bytekgw_config: str,
    kgw_config: str,
    seed: int,
    item: PromptItem,
    prompt_tokens: int,
    cont_tokens: int,
    full_tokens: int,
    prompt_found: bool,
    gen_params: Dict[str, Any],
    full_text: str,
    continuation_text: str,
    stat_name: str,
    stat_value: Any,
    ppl: float,
) -> Dict[str, Any]:
    return {
        "algo": algo,
        "delta": delta,
        "device": device,
        "model_path": model_path,
        "bytekgw_config": bytekgw_config,
        "kgw_config": kgw_config,
        "seed": seed,
        "prompt_id": item.prompt_id,
        "c4_source_path": item.source_path,
        "c4_source_line": item.source_line,
        "prompt_text": item.prompt_text,
        "c4_source_head": item.source_head,
        "full_text": full_text,
        "continuation_text": continuation_text,
        "stat_name": stat_name,
        "stat_value": stat_value,
        "ppl": ppl,
        "prompt_tokens": prompt_tokens,
        "cont_tokens": cont_tokens,
        "full_tokens": full_tokens,
        "prompt_found_in_full": bool(prompt_found),
        "gen_params_json": json.dumps(gen_params, ensure_ascii=False, sort_keys=True),
    }


# ------------------------------------------------------------
# Worker
# ------------------------------------------------------------
def worker_main(rank: int, device_str: str, items: List[PromptItem], args_dict: Dict[str, Any], q: mp.Queue):
    """
    Each worker loads model+tokenizer on its GPU and generates for assigned prompts.
    Returns dict[file_key -> rows]
    """
    try:
        torch.cuda.set_device(int(device_str.split(":")[1]))
    except Exception:
        pass

    device = torch.device(device_str)
    dtype = parse_dtype(args_dict["dtype"])

    model_path = args_dict["model"]
    add_special_tokens = bool(args_dict["add_special_tokens"])
    max_new_tokens = int(args_dict["max_new_tokens"])
    seed_base = int(args_dict["seed_base"])
    deltas = list(args_dict["deltas"])
    out_dir = args_dict["out_dir"]
    bytekgw_config = args_dict["bytekgw_config"]
    kgw_config = args_dict["kgw_config"]

    # gen kwargs统一
    gen_kwargs = dict(
        do_sample=bool(args_dict["do_sample"]),
        temperature=float(args_dict["temperature"]),
        top_p=float(args_dict["top_p"]),
        top_k=int(args_dict["top_k"]),
        repetition_penalty=float(args_dict["repetition_penalty"]),
        max_new_tokens=max_new_tokens,
        use_cache=True,
    )

    # load tokenizer/model
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    ensure_pad_token(tok)
    gen_kwargs["pad_token_id"] = int(tok.pad_token_id) if tok.pad_token_id is not None else None
    gen_kwargs["eos_token_id"] = int(tok.eos_token_id) if tok.eos_token_id is not None else None
    gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, device_map=None).to(device)
    model.eval()

    tf_cfg = build_transformers_config(model, tok, device_str, gen_kwargs)

    # import algorithms
    from MarkLLM.watermark.bytekgwV5 import ByteKGWv5
    try:
        from MarkLLM.watermark.kgw import KGW
    except Exception:
        from MarkLLM.watermark.kgw.kgw import KGW

    results: Dict[str, List[Dict[str, Any]]] = {}

    def add_row(file_key: str, row: Dict[str, Any]) -> None:
        results.setdefault(file_key, []).append(row)

    # ------------------------------
    # HF baseline (delta=0)
    # ------------------------------
    for idx, item in enumerate(items):
        seed = seed_base + item.prompt_id  # 固定：同 prompt_id 同 seed
        ids, prompt_len_tokens = hf_generate_ids(
            model, tok, item.prompt_text,
            device=device, seed=seed,
            gen_kwargs=gen_kwargs,
            add_special_tokens=add_special_tokens,
        )

        # truncate for eval
        if CAP_CONT_TOKENS_FOR_EVAL:
            trunc_ids = truncate_to_cont(ids, prompt_len_tokens, max_new_tokens)
        else:
            trunc_ids = ids

        full_text = tok.decode(trunc_ids, skip_special_tokens=True)
        cont_ids = trunc_ids[prompt_len_tokens:]
        continuation = tok.decode(cont_ids, skip_special_tokens=True)

        ppl, cont_len = ppl_continuation_from_ids(model, trunc_ids, prompt_len_tokens, device=device)

        row = row_base(
            algo="hf.generate",
            delta=0.0,
            device=device_str,
            model_path=model_path,
            bytekgw_config=bytekgw_config,
            kgw_config=kgw_config,
            seed=seed,
            item=item,
            prompt_tokens=prompt_len_tokens,
            cont_tokens=cont_len,
            full_tokens=len(trunc_ids),
            prompt_found=True,
            gen_params=gen_kwargs,
            full_text=full_text,
            continuation_text=continuation,
            stat_name="none",
            stat_value="",
            ppl=ppl,
        )
        add_row("hf_generate", row)

        if (idx + 1) % max(1, len(items)) == 0:
            print(f"[GPU {device_str}] HF baseline done ({idx+1}/{len(items)})", flush=True)

    # ------------------------------
    # Watermarked generations for deltas
    # ------------------------------
    for delta in deltas:
        # ByteKGWv5 head temp cfg
        tmp_b = make_temp_cfg(bytekgw_config, {
            "delta": float(delta),
            "scheme": BYTEKGW_SCHEME,
            "max_byte_pos": int(BYTEKGW_HEAD_MAX_BYTE_POS),
            "use_prefix_bytes_in_prf": True,
            "add_special_tokens": bool(add_special_tokens),
            "use_torch_generator": False,
            # 可选：兼容旧版config里还有 detector_mode
            "detector_mode": "all_bytes",
            "gen": dict(gen_kwargs),
        })

        # KGW temp cfg：强制 gen 参数（尤其 max_new_tokens）
        prompt_len = len(tok(items[0].prompt_text, add_special_tokens=add_special_tokens)["input_ids"]) if items else 0
        tmp_k = make_temp_cfg(kgw_config, {
            "delta": float(delta),
            "add_special_tokens": bool(add_special_tokens),
            "use_torch_generator": False,
            "gen": {
                **dict(gen_kwargs),
                "max_length": int(prompt_len + max_new_tokens),
            },
            "max_new_tokens": int(max_new_tokens),
            "max_length": int(prompt_len + max_new_tokens),
        })

        try:
            bytekgw = ByteKGWv5(tmp_b, tf_cfg)
            kgw = KGW(tmp_k, tf_cfg)

            # 额外保险：KGW 强制使用 gen_kwargs
            if hasattr(kgw, "config") and hasattr(kgw.config, "gen_kwargs"):
                kgw.config.gen_kwargs = dict(gen_kwargs)
                kgw.config.gen_kwargs.setdefault("max_length", int(prompt_len + max_new_tokens))

            # Generate per prompt
            for idx, item in enumerate(items):
                seed = seed_base + item.prompt_id

                # --- KGW ---
                txt_k = alg_generate_text(kgw, item.prompt_text, seed=seed)

                trunc_ids, prompt_end, trunc_full, trunc_cont, prompt_found = tokenize_truncate_for_eval(
                    tok, txt_k, item.prompt_text,
                    add_special_tokens=add_special_tokens,
                    max_new_tokens=max_new_tokens,
                )

                det_k = kgw.detect_watermark(trunc_full, return_dict=True, add_special_tokens=False)
                stat_k = extract_stat(det_k)

                ppl_k, cont_len_k = ppl_continuation_from_ids(model, trunc_ids, prompt_end, device=device)

                add_row(
                    f"kgw_delta{int(delta)}",
                    row_base(
                        algo="KGW",
                        delta=float(delta),
                        device=device_str,
                        model_path=model_path,
                        bytekgw_config=bytekgw_config,
                        kgw_config=kgw_config,
                        seed=seed,
                        item=item,
                        prompt_tokens=prompt_end,
                        cont_tokens=cont_len_k,
                        full_tokens=len(trunc_ids),
                        prompt_found=prompt_found,
                        gen_params=gen_kwargs,
                        full_text=trunc_full,
                        continuation_text=trunc_cont,
                        stat_name="score",
                        stat_value=stat_k,
                        ppl=ppl_k,
                    )
                )

                # --- ByteKGWv5 head ---
                txt_b = alg_generate_text(bytekgw, item.prompt_text, seed=seed)

                trunc_ids2, prompt_end2, trunc_full2, trunc_cont2, prompt_found2 = tokenize_truncate_for_eval(
                    tok, txt_b, item.prompt_text,
                    add_special_tokens=add_special_tokens,
                    max_new_tokens=max_new_tokens,
                )

                det_b = bytekgw.detect_watermark(trunc_full2, return_dict=True, add_special_tokens=False)
                stat_b = extract_stat(det_b)

                ppl_b, cont_len_b = ppl_continuation_from_ids(model, trunc_ids2, prompt_end2, device=device)

                add_row(
                    f"bytekgw_head_delta{int(delta)}",
                    row_base(
                        algo="ByteKGWv5(head)",
                        delta=float(delta),
                        device=device_str,
                        model_path=model_path,
                        bytekgw_config=bytekgw_config,
                        kgw_config=kgw_config,
                        seed=seed,
                        item=item,
                        prompt_tokens=prompt_end2,
                        cont_tokens=cont_len_b,
                        full_tokens=len(trunc_ids2),
                        prompt_found=prompt_found2,
                        gen_params=gen_kwargs,
                        full_text=trunc_full2,
                        continuation_text=trunc_cont2,
                        stat_name="z",
                        stat_value=stat_b,
                        ppl=ppl_b,
                    )
                )

                if (idx + 1) % max(1, len(items) // 2) == 0 or (idx + 1) == len(items):
                    print(f"[GPU {device_str}] delta={delta} done ({idx+1}/{len(items)})", flush=True)

        finally:
            # cleanup temp json
            for p in [tmp_b, tmp_k]:
                try:
                    os.remove(p)
                except Exception:
                    pass

    q.put(results)


# ------------------------------------------------------------
# CSV writer + main
# ------------------------------------------------------------
def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            # stringify None
            rr = {k: ("" if r.get(k) is None else r.get(k)) for k in CSV_FIELDS}
            w.writerow(rr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, type=str)

    ap.add_argument("--devices", default="cuda:0,cuda:1", type=str,
                    help="comma-separated, e.g. cuda:0,cuda:1")
    ap.add_argument("--dtype", default="float16", type=str)

    ap.add_argument("--bytekgw_config", required=True, type=str)
    ap.add_argument("--kgw_config", required=True, type=str)

    ap.add_argument("--c4_paths", type=str, default="dataset/c4/realnewslike/c4-train.00000-of-00512.json.gz",
                    help="comma-separated list of .json.gz files (C4).")
    ap.add_argument("--n_prompts", type=int, default=10)
    ap.add_argument("--prompt_words", type=int, default=PROMPT_WORDS)

    ap.add_argument("--add_special_tokens", action="store_true")

    # generation knobs
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--do_sample", action="store_true")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--repetition_penalty", type=float, default=1.0)

    ap.add_argument("--seed_base", type=int, default=1234)
    ap.add_argument("--deltas", type=str, default="1,2,3,4,5")

    ap.add_argument("--out_dir", type=str, default="generated_samples_c4")

    args = ap.parse_args()

    devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    if len(devices) < 1:
        raise ValueError("Need at least 1 device")

    c4_paths = [p.strip() for p in args.c4_paths.split(",") if p.strip()]
    deltas = [float(x.strip()) for x in args.deltas.split(",") if x.strip()]

    prompts = load_c4_prompts(c4_paths, args.n_prompts, args.prompt_words)
    if len(prompts) < args.n_prompts:
        print(f"WARNING: only loaded {len(prompts)}/{args.n_prompts} prompts from {c4_paths}", file=sys.stderr)

    # split prompts across devices
    buckets: List[List[PromptItem]] = [[] for _ in devices]
    for it in prompts:
        buckets[it.prompt_id % len(devices)].append(it)

    os.makedirs(args.out_dir, exist_ok=True)
    meta_path = os.path.join(args.out_dir, "run_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": args.model,
            "devices": devices,
            "dtype": args.dtype,
            "bytekgw_config": args.bytekgw_config,
            "kgw_config": args.kgw_config,
            "c4_paths": c4_paths,
            "n_prompts": args.n_prompts,
            "prompt_words": args.prompt_words,
            "add_special_tokens": bool(args.add_special_tokens),
            "gen": {
                "do_sample": bool(args.do_sample),
                "temperature": float(args.temperature),
                "top_p": float(args.top_p),
                "top_k": int(args.top_k),
                "repetition_penalty": float(args.repetition_penalty),
                "max_new_tokens": int(args.max_new_tokens),
            },
            "seed_base": int(args.seed_base),
            "deltas": deltas,
            "bytekgw_scheme": BYTEKGW_SCHEME,
            "bytekgw_head_max_byte_pos": BYTEKGW_HEAD_MAX_BYTE_POS,
            "use_prefix_bytes_in_prf": USE_PREFIX_BYTES_IN_PRF,
            "cap_cont_tokens_for_eval": CAP_CONT_TOKENS_FOR_EVAL,
            "prompts": [asdict(p) for p in prompts],
        }, f, ensure_ascii=False, indent=2)

    # multiprocessing
    mp.set_start_method("spawn", force=True)
    q: mp.Queue = mp.Queue()
    procs: List[mp.Process] = []

    args_dict = {
        "model": args.model,
        "dtype": args.dtype,
        "bytekgw_config": args.bytekgw_config,
        "kgw_config": args.kgw_config,
        "add_special_tokens": bool(args.add_special_tokens),
        "max_new_tokens": int(args.max_new_tokens),
        "do_sample": bool(args.do_sample),
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "top_k": int(args.top_k),
        "repetition_penalty": float(args.repetition_penalty),
        "seed_base": int(args.seed_base),
        "deltas": deltas,
        "out_dir": args.out_dir,
    }

    print("=" * 120)
    print(f"[model] {args.model}")
    print(f"[devices] {devices}")
    print(f"[c4_paths] {c4_paths}")
    print(f"[prompts] loaded={len(prompts)}  prompt_words={args.prompt_words}")
    print(f"[gen] do_sample={args.do_sample} temp={args.temperature} top_p={args.top_p} top_k={args.top_k} "
          f"rep_penalty={args.repetition_penalty} max_new_tokens={args.max_new_tokens} add_special_tokens={args.add_special_tokens}")
    print(f"[deltas] {deltas}")
    print(f"[ByteKGWv5] scheme={BYTEKGW_SCHEME} head_max_byte_pos={BYTEKGW_HEAD_MAX_BYTE_POS} use_prefix_bytes_in_prf={USE_PREFIX_BYTES_IN_PRF}")
    print(f"[out_dir] {args.out_dir}")
    print("=" * 120)

    for r, dev in enumerate(devices):
        p = mp.Process(target=worker_main, args=(r, dev, buckets[r], args_dict, q))
        p.start()
        procs.append(p)

    merged: Dict[str, List[Dict[str, Any]]] = {}
    for _ in procs:
        part = q.get()
        for k, rows in part.items():
            merged.setdefault(k, []).extend(rows)

    for p in procs:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"Worker exited with code {p.exitcode}")

    # write CSVs
    # 1) baseline
    write_csv(os.path.join(args.out_dir, "hf_generate.csv"), sorted(merged.get("hf_generate", []), key=lambda r: r["prompt_id"]))
    # 2) per-delta KGW and ByteKGW(head)
    for d in deltas:
        dk = int(d)
        write_csv(
            os.path.join(args.out_dir, f"kgw_delta{dk}.csv"),
            sorted(merged.get(f"kgw_delta{dk}", []), key=lambda r: r["prompt_id"])
        )
        write_csv(
            os.path.join(args.out_dir, f"bytekgw_head_delta{dk}.csv"),
            sorted(merged.get(f"bytekgw_head_delta{dk}", []), key=lambda r: r["prompt_id"])
        )

    print("\n[DONE] wrote files:")
    print(f"  - {os.path.join(args.out_dir, 'hf_generate.csv')}")
    for d in deltas:
        dk = int(d)
        print(f"  - {os.path.join(args.out_dir, f'kgw_delta{dk}.csv')}")
        print(f"  - {os.path.join(args.out_dir, f'bytekgw_head_delta{dk}.csv')}")
    print(f"  - {meta_path}")


if __name__ == "__main__":
    main()
