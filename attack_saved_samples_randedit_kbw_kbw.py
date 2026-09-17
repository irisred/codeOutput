#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import math
import os
import random
import tempfile
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


# ---------------------------
# Utilities
# ---------------------------

def parse_csv_list(s: str, cast=float) -> List[Any]:
    if s is None or s == "":
        return []
    out = []
    for x in s.split(","):
        x = x.strip()
        if not x:
            continue
        out.append(cast(x))
    return out

def parse_ops(s: str) -> List[str]:
    if not s:
        return ["replace", "delete", "insert"]
    ops = []
    for x in s.split(","):
        x = x.strip().lower()
        if x:
            ops.append(x)
    return ops

def safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return default

def empirical_fpr(zs: List[float], thr: float) -> float:
    if not zs:
        return 0.0
    return sum(1 for z in zs if z > thr) / float(len(zs))

def choose_threshold_at_most(zs: List[float], target_fpr: float) -> Tuple[float, float]:
    """
    Rule: predict WM if stat > thr
    Choose thr so that achieved FPR on NEG is as close as possible to target_fpr, but NOT exceeding it.
    """
    if not zs:
        return (float("inf"), 0.0)
    uniq = sorted(set(zs))
    cands = [uniq[-1] + 1e-9] + uniq  # first => FPR=0
    best_thr = cands[0]
    best_fpr = empirical_fpr(zs, best_thr)
    for thr in cands:
        fpr = empirical_fpr(zs, thr)
        if fpr <= target_fpr:
            if (fpr > best_fpr + 1e-12) or (abs(fpr - best_fpr) <= 1e-12 and thr < best_thr):
                best_thr, best_fpr = thr, fpr
    return best_thr, best_fpr

def choose_threshold_at_least(zs: List[float], target_fpr: float) -> Tuple[float, float]:
    """
    Rule: predict WM if stat > thr
    Choose thr so that achieved FPR on NEG is as close as possible to target_fpr, but AT LEAST it.
    """
    if not zs:
        return (float("-inf"), 1.0)
    uniq = sorted(set(zs))
    cands = [uniq[0] - 1e-9] + uniq  # first => FPR=1
    best_thr = cands[0]
    best_fpr = empirical_fpr(zs, best_thr)
    for thr in cands:
        fpr = empirical_fpr(zs, thr)
        if fpr >= target_fpr:
            if (fpr < best_fpr - 1e-12) or (abs(fpr - best_fpr) <= 1e-12 and thr > best_thr):
                best_thr, best_fpr = thr, fpr
    return best_thr, best_fpr

def choose_threshold(zs: List[float], target_fpr: float, thr_mode: str) -> Tuple[float, float]:
    thr_mode = (thr_mode or "at_most").lower()
    if thr_mode == "at_least":
        return choose_threshold_at_least(zs, target_fpr)
    return choose_threshold_at_most(zs, target_fpr)

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def materialize_cfg(cfg: Dict[str, Any]) -> str:
    fd, tmp = tempfile.mkstemp(prefix="wm_cfg_", suffix=".json")
    os.close(fd)
    write_json(tmp, cfg)
    return tmp

def read_csv_rows(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows

def pick_text_fields(row: Dict[str, Any]) -> Tuple[str, str, str]:
    """
    Return (prompt, completion, full_text)
    """
    prompt = row.get("prompt") or row.get("c4_prompt") or row.get("input") or ""
    completion = row.get("completion") or row.get("completion_text") or row.get("gen") or row.get("generated") or ""
    full = row.get("full_text") or row.get("text") or row.get("orig_text") or row.get("output") or ""
    if not full:
        if prompt and completion:
            full = prompt + completion
        elif completion:
            full = completion
        else:
            full = prompt
    if not completion and prompt and full.startswith(prompt):
        completion = full[len(prompt):]
    return prompt, completion, full

def get_ppl(row: Dict[str, Any]) -> float:
    for k in ["ppl", "perplexity", "ppl_mean"]:
        if k in row and row[k] != "":
            return safe_float(row[k], float("nan"))
    return float("nan")

def stable_seed(base: int, *xs: Any) -> int:
    h = base
    for x in xs:
        h = (h * 1315423911 + hash(x)) & 0x7fffffff
    return h


# ---------------------------
# Random edit attack (token-id level)
# ---------------------------

@dataclass
class AttackResult:
    adv_text: str
    edit_ops: int
    orig_len: int

def build_random_token_sampler(tokenizer) -> List[int]:
    special = set(getattr(tokenizer, "all_special_ids", []) or [])
    V = len(tokenizer)
    pool = [i for i in range(V) if i not in special]
    return pool if pool else list(range(V))

def randedit_token_ids(
    ids: List[int],
    *,
    rng: random.Random,
    vocab_pool: List[int],
    edit_rate: float,
    ops: List[str],
) -> Tuple[List[int], int]:
    if not ids:
        return ids, 0
    L0 = len(ids)
    k = int(round(edit_rate * L0))
    if k <= 0:
        return ids, 0
    out = list(ids)
    edits = 0
    for _ in range(k):
        op = rng.choice(ops)
        if op == "delete":
            if len(out) > 1:
                i = rng.randrange(0, len(out))
                out.pop(i)
                edits += 1
        elif op == "insert":
            i = rng.randrange(0, len(out) + 1)
            out.insert(i, rng.choice(vocab_pool))
            edits += 1
        else:  # replace
            i = rng.randrange(0, len(out))
            out[i] = rng.choice(vocab_pool)
            edits += 1
    return out, edits

def attack_text(
    *,
    tokenizer,
    prompt: str,
    completion: str,
    full_text: str,
    attack_generated_only: bool,
    edit_rate: float,
    ops: List[str],
    seed: int,
) -> AttackResult:
    rng = random.Random(seed)
    vocab_pool = build_random_token_sampler(tokenizer)

    if attack_generated_only:
        ids = tokenizer.encode(completion, add_special_tokens=False)
        adv_ids, edits = randedit_token_ids(ids, rng=rng, vocab_pool=vocab_pool, edit_rate=edit_rate, ops=ops)
        adv_comp = tokenizer.decode(adv_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        adv_text = (prompt or "") + adv_comp
        return AttackResult(adv_text=adv_text, edit_ops=edits, orig_len=len(ids))
    else:
        ids = tokenizer.encode(full_text, add_special_tokens=False)
        adv_ids, edits = randedit_token_ids(ids, rng=rng, vocab_pool=vocab_pool, edit_rate=edit_rate, ops=ops)
        adv_text = tokenizer.decode(adv_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        return AttackResult(adv_text=adv_text, edit_ops=edits, orig_len=len(ids))


# ---------------------------
# Build aligned MarkLLM configs / detectors
# ---------------------------

def build_transformers_config(model, tokenizer, device: torch.device, dtype: torch.dtype, gen_kwargs: Dict[str, Any]):
    """
    Robustly instantiate MarkLLM.utils.transformers_config.TransformersConfig
    """
    from MarkLLM.utils.transformers_config import TransformersConfig
    import inspect

    cand = {
        "model": model,
        "tokenizer": tokenizer,
        "generation_tokenizer": tokenizer,
        "detection_tokenizer": tokenizer,
        "device": device,
        "dtype": dtype,
        "torch_dtype": dtype,
        "gen_kwargs": gen_kwargs,
        "generation_kwargs": gen_kwargs,
    }
    sig = inspect.signature(TransformersConfig)
    kwargs = {k: v for k, v in cand.items() if k in sig.parameters}
    obj = TransformersConfig(**kwargs)

    for k, v in cand.items():
        if not hasattr(obj, k):
            try:
                setattr(obj, k, v)
            except Exception:
                pass
    return obj

def build_bytekgw(bytekgw_cfg_path: str, tf_cfg, *, force_max_byte_pos: Optional[int]):
    cfg = load_json(bytekgw_cfg_path)
    if force_max_byte_pos is not None:
        cfg["max_byte_pos"] = int(force_max_byte_pos)
        if "detector_mode" in cfg:
            cfg["detector_mode"] = "first_byte" if force_max_byte_pos == 1 else cfg.get("detector_mode", "all_bytes")
    tmp = materialize_cfg(cfg)

    from MarkLLM.watermark.bytekgwV5 import ByteKGWv5
    return ByteKGWv5(tmp, tf_cfg)

def build_kgw(kgw_cfg_path: str, tf_cfg):
    try:
        from MarkLLM.watermark.kgw import KGW
    except Exception:
        from MarkLLM.watermark.kgw.kgw import KGW
    return KGW(kgw_cfg_path, tf_cfg)

@torch.inference_mode()
def score_bytekgw(wm, text: str) -> float:
    raw = wm.detect_watermark(text, return_dict=True, add_special_tokens=False)
    if isinstance(raw, dict) and "z" in raw:
        return float(raw["z"])
    for k, v in raw.items():
        if isinstance(v, (int, float)):
            return float(v)
    raise RuntimeError(f"ByteKGWv5 detect returned no numeric field: keys={list(raw.keys())}")

@torch.inference_mode()
def score_kgw(kgw, text: str) -> float:
    """
    KGW 默认 detector：有 min_prefix_len=4 的硬约束。
    攻击后可能导致有效 token 太少 -> KGW 会抛 ValueError。
    我们把这种情况视为“检测分数极低”（-inf），让流程继续且不误判为 WM。
    """
    # 尽量用 KGW 自己的 detection tokenizer，确保对齐
    tok = getattr(getattr(kgw, "config", None), "detection_tokenizer", None)
    if tok is None:
        tok = getattr(getattr(kgw, "config", None), "generation_tokenizer", None)
    if tok is None:
        # 最后兜底：用 kgw.config 里可能存在的 tokenizer 字段
        tok = getattr(getattr(kgw, "config", None), "tokenizer", None)

    # min_prefix_len：优先从 utils 取（KGW 代码里常见）
    min_prefix_len = getattr(getattr(kgw, "utils", None), "min_prefix_len", 4)

    try:
        if tok is not None:
            ids = tok.encode(text, add_special_tokens=False)
            if len(ids) <= int(min_prefix_len):
                return float("-inf")

        raw = kgw.detect_watermark(text, return_dict=True, add_special_tokens=False)

        # MarkLLM KGW 常见返回：{'is_watermarked': bool, 'score': float}
        if isinstance(raw, dict):
            if "score" in raw:
                return float(raw["score"])
            if "z" in raw:
                return float(raw["z"])
            # 兜底：找第一个数值字段
            for _, v in raw.items():
                if isinstance(v, (int, float)):
                    return float(v)

        raise RuntimeError(f"KGW detect returned unexpected type/keys: {type(raw)} {getattr(raw, 'keys', lambda: [])()}")

    except ValueError as e:
        # 典型：token 不足导致 KGW 抛错
        return float("-inf")

# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--model", type=str, required=True, help="HF model path/name (will LOAD weights: required by ByteKGWv5 init)")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])

    ap.add_argument("--bytekgw_config", type=str, required=True)
    ap.add_argument("--kgw_config", type=str, required=True)

    ap.add_argument("--deltas", type=str, default="1,2,3,4,5")
    ap.add_argument("--fprs", type=str, default="0.01,0.05,0.10,0.20")
    ap.add_argument("--thr_mode", type=str, default="at_most", choices=["at_most", "at_least"])

    ap.add_argument("--attack_edit_rate", type=float, default=0.05)
    ap.add_argument("--attack_ops", type=str, default="replace,delete,insert")
    ap.add_argument("--attack_generated_only", action="store_true")
    ap.add_argument("--seed", type=int, default=1234)

    args = ap.parse_args()

    run_dir = args.run_dir
    deltas = [int(x) for x in parse_csv_list(args.deltas, int)]
    fprs = [float(x) for x in parse_csv_list(args.fprs, float)]
    ops = parse_ops(args.attack_ops)

    device = torch.device(args.device)
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]

    print("=" * 120)
    print(f"[RUN] run_dir={run_dir}")
    print(f"[device]={device}  (KGW/ByteKGW 对齐要求：生成在哪个 device，就用哪个 device 检测)")
    print(f"[attack] edit_rate={args.attack_edit_rate} ops={ops} generated_only={args.attack_generated_only} seed={args.seed}")
    print(f"[eval] deltas={deltas} fprs={fprs} thr_mode={args.thr_mode}")
    print("=" * 120)

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # ✅ FIX: load model weights (required by ByteKGWv5 __init__ which builds engine/stepper)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.eval()
    model.to(device)

    # tf_cfg
    tf_cfg = build_transformers_config(model, tokenizer, device=device, dtype=dtype, gen_kwargs={})

    # determine bytekgw head/all from run_dir
    force_max_byte_pos = None
    head_path = os.path.join(run_dir, "bytekgw_head_delta1.csv")
    all_path = os.path.join(run_dir, "bytekgw_all_delta1.csv")
    if os.path.exists(head_path) and not os.path.exists(all_path):
        force_max_byte_pos = 1
        bytekgw_alg_name = "bytekgw_head"
    elif os.path.exists(all_path):
        force_max_byte_pos = None
        bytekgw_alg_name = "bytekgw_all"
    else:
        force_max_byte_pos = 1
        bytekgw_alg_name = "bytekgw_head"

    bytekgw = build_bytekgw(args.bytekgw_config, tf_cfg, force_max_byte_pos=force_max_byte_pos)
    kgw = build_kgw(args.kgw_config, tf_cfg)

    # NEG calibration set
    neg_csv = os.path.join(run_dir, "hf_generate.csv")
    if not os.path.exists(neg_csv):
        raise FileNotFoundError(f"missing NEG file: {neg_csv}")

    neg_rows = read_csv_rows(neg_csv)
    print(f"[NEG] loaded {len(neg_rows)} from {neg_csv}")

    # score NEG clean
    z_neg_byte: List[float] = []
    z_neg_kgw: List[float] = []
    neg_ppls: List[float] = []

    print("[NEG] scoring (clean) ...")
    for row in tqdm(neg_rows, desc="NEG scoring", ncols=100):
        prompt, completion, full = pick_text_fields(row)
        z_neg_byte.append(score_bytekgw(bytekgw, full))
        z_neg_kgw.append(score_kgw(kgw, full))
        ppl = get_ppl(row)
        if not math.isnan(ppl):
            neg_ppls.append(ppl)

    thresholds: Dict[str, Dict[float, Tuple[float, float]]] = {bytekgw_alg_name: {}, "kgw": {}}
    for fpr in fprs:
        thr_b, ach_b = choose_threshold(z_neg_byte, fpr, args.thr_mode)
        thr_k, ach_k = choose_threshold(z_neg_kgw, fpr, args.thr_mode)
        thresholds[bytekgw_alg_name][fpr] = (thr_b, ach_b)
        thresholds["kgw"][fpr] = (thr_k, ach_k)

    out_dir = os.path.join(run_dir, "attack_randedit")
    ensure_dir(out_dir)

    neg_thr_path = os.path.join(out_dir, "neg_thresholds.csv")
    with open(neg_thr_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["alg", "target_fpr", "thr_mode", "threshold", "neg_fpr_emp"])
        for fpr in fprs:
            thr_b, ach_b = thresholds[bytekgw_alg_name][fpr]
            thr_k, ach_k = thresholds["kgw"][fpr]
            w.writerow([bytekgw_alg_name, fpr, args.thr_mode, thr_b, ach_b])
            w.writerow(["kgw", fpr, args.thr_mode, thr_k, ach_k])
    print(f"[WROTE] {neg_thr_path}")

    def process_pos(alg: str, pos_csv: str, scorer, out_csv: str) -> Dict[float, Dict[str, float]]:
        rows = read_csv_rows(pos_csv)
        z_before = []
        z_after = []
        pos_ppls = []

        with open(out_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "delta", "idx", "seed",
                "prompt", "completion",
                "orig_text", "adv_text",
                "stat_before", "stat_after",
                "ppl",
                "attack_edit_rate", "attack_ops", "attack_generated_only",
                "edit_ops", "orig_token_len",
                "text_sim_ratio"
            ])

            for idx, row in enumerate(tqdm(rows, desc=f"ATTACK {alg}", ncols=100)):
                prompt, completion, full = pick_text_fields(row)
                ppl = get_ppl(row)
                if not math.isnan(ppl):
                    pos_ppls.append(ppl)

                s0 = scorer(full)
                s = stable_seed(args.seed, alg, os.path.basename(pos_csv), idx)
                atk = attack_text(
                    tokenizer=tokenizer,
                    prompt=prompt,
                    completion=completion,
                    full_text=full,
                    attack_generated_only=args.attack_generated_only,
                    edit_rate=args.attack_edit_rate,
                    ops=ops,
                    seed=s,
                )
                s1 = scorer(atk.adv_text)

                z_before.append(s0)
                z_after.append(s1)

                sim = SequenceMatcher(None, full, atk.adv_text).ratio()

                w.writerow([
                    row.get("delta", ""), idx, s,
                    prompt, completion,
                    full, atk.adv_text,
                    s0, s1,
                    ppl,
                    args.attack_edit_rate, ",".join(ops), bool(args.attack_generated_only),
                    atk.edit_ops, atk.orig_len,
                    sim
                ])

        summary: Dict[float, Dict[str, float]] = {}
        for fpr in fprs:
            thr, ach = thresholds[alg][fpr]
            hit0 = [1 if x > thr else 0 for x in z_before]
            hit1 = [1 if x > thr else 0 for x in z_after]
            tpr0 = sum(hit0) / max(1, len(hit0))
            tpr1 = sum(hit1) / max(1, len(hit1))
            num0 = sum(hit0)
            num1 = sum(hit1)
            asr = (num0 - num1) / num0 if num0 > 0 else 0.0
            mean_drop = sum((a - b) for a, b in zip(z_before, z_after)) / max(1, len(z_before))

            summary[fpr] = {
                "threshold": thr,
                "neg_fpr_emp": ach,
                "tpr_before": tpr0,
                "tpr_after": tpr1,
                "attack_success_rate": asr,
                "pos_z_mean_before": float(sum(z_before) / max(1, len(z_before))),
                "pos_z_mean_after": float(sum(z_after) / max(1, len(z_after))),
                "mean_z_drop": mean_drop,
                "pos_ppl_mean": float(sum(pos_ppls) / max(1, len(pos_ppls))) if pos_ppls else float("nan"),
            }
        return summary

    attack_summary_rows = []
    neg_ppl_mean = float(sum(neg_ppls) / max(1, len(neg_ppls))) if neg_ppls else float("nan")

    for d in deltas:
        # ByteKGW
        pos_csv_b = os.path.join(run_dir, f"{bytekgw_alg_name}_delta{d}.csv")
        if not os.path.exists(pos_csv_b):
            raise FileNotFoundError(f"missing POS file for {bytekgw_alg_name}: {pos_csv_b}")
        out_b = os.path.join(out_dir, f"attack_{bytekgw_alg_name}_delta{d}.csv")
        sum_b = process_pos(bytekgw_alg_name, pos_csv_b, lambda t: score_bytekgw(bytekgw, t), out_b)
        print(f"[WROTE] {out_b}")

        # KGW
        pos_csv_k = os.path.join(run_dir, f"kgw_delta{d}.csv")
        if not os.path.exists(pos_csv_k):
            raise FileNotFoundError(f"missing POS file for kgw: {pos_csv_k}")
        out_k = os.path.join(out_dir, f"attack_kgw_delta{d}.csv")
        sum_k = process_pos("kgw", pos_csv_k, lambda t: score_kgw(kgw, t), out_k)
        print(f"[WROTE] {out_k}")

        for fpr in fprs:
            sb = sum_b[fpr]
            sk = sum_k[fpr]
            attack_summary_rows.append({
                "alg": bytekgw_alg_name,
                "delta": d,
                "target_fpr": fpr,
                "thr_mode": args.thr_mode,
                "threshold": sb["threshold"],
                "neg_fpr_emp": sb["neg_fpr_emp"],
                "tpr_before": sb["tpr_before"],
                "tpr_after": sb["tpr_after"],
                "attack_success_rate": sb["attack_success_rate"],
                "mean_z_drop": sb["mean_z_drop"],
                "neg_ppl_mean": neg_ppl_mean,
                "pos_ppl_mean": sb["pos_ppl_mean"],
                "pos_z_mean_before": sb["pos_z_mean_before"],
                "pos_z_mean_after": sb["pos_z_mean_after"],
            })
            attack_summary_rows.append({
                "alg": "kgw",
                "delta": d,
                "target_fpr": fpr,
                "thr_mode": args.thr_mode,
                "threshold": sk["threshold"],
                "neg_fpr_emp": sk["neg_fpr_emp"],
                "tpr_before": sk["tpr_before"],
                "tpr_after": sk["tpr_after"],
                "attack_success_rate": sk["attack_success_rate"],
                "mean_z_drop": sk["mean_z_drop"],
                "neg_ppl_mean": neg_ppl_mean,
                "pos_ppl_mean": sk["pos_ppl_mean"],
                "pos_z_mean_before": sk["pos_z_mean_before"],
                "pos_z_mean_after": sk["pos_z_mean_after"],
            })

    summ_path = os.path.join(out_dir, "attack_summary.csv")
    keys = [
        "alg", "delta", "target_fpr", "thr_mode",
        "threshold", "neg_fpr_emp",
        "tpr_before", "tpr_after", "attack_success_rate",
        "mean_z_drop",
        "neg_ppl_mean", "pos_ppl_mean",
        "pos_z_mean_before", "pos_z_mean_after",
    ]
    with open(summ_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in attack_summary_rows:
            w.writerow(r)

    print("=" * 120)
    print(f"[DONE] wrote: {summ_path}")
    print(f"[DONE] wrote: {neg_thr_path}")
    print("=" * 120)


if __name__ == "__main__":
    main()
