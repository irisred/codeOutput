#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import inspect
import json
import math
import os
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoTokenizer

# ---- robust KGW import (MarkLLM 有时是单文件，有时是包) ----
def import_kgw():
    try:
        from MarkLLM.watermark.kgw import KGW
        return KGW
    except Exception:
        from MarkLLM.watermark.kgw.kgw import KGW
        return KGW


def read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def norm_prompt(s: str) -> str:
    return " ".join((s or "").strip().split())


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_transformers_config(model, tokenizer, device: torch.device, dtype: torch.dtype, gen_kwargs: Dict[str, Any]):
    """
    兼容不同版本 MarkLLM TransformersConfig 构造函数签名
    （你 pasted.txt 里就是这么干的）
    """
    from MarkLLM.utils.transformers_config import TransformersConfig

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
        "vocab_size": int(len(tokenizer)),
    }
    sig = inspect.signature(TransformersConfig)
    kwargs = {k: v for k, v in cand.items() if k in sig.parameters}
    obj = TransformersConfig(**kwargs)

    # 兜底：把缺的属性手动塞进去
    for k, v in cand.items():
        if not hasattr(obj, k):
            try:
                setattr(obj, k, v)
            except Exception:
                pass
    return obj


@torch.inference_mode()
def kgw_score(kgw, text: str) -> float:
    out = kgw.detect_watermark(text, return_dict=True)  # KGW 内部固定 add_special_tokens=False :contentReference[oaicite:1]{index=1}
    if isinstance(out, dict):
        if "score" in out:
            return float(out["score"])
        if "z" in out:
            return float(out["z"])
        # 兜底：找第一个数值字段
        for v in out.values():
            if isinstance(v, (int, float)):
                return float(v)
    raise RuntimeError(f"Unexpected detect output: {out}")


def parse_stat_value(row: Dict[str, str]) -> float:
    sv = row.get("stat_value", "")
    if sv == "":
        sv = row.get("z", "") or row.get("score", "")
    return float(sv)


def compare_on_pos_csv(kgw, csv_path: str, limit: int = 0) -> Tuple[int, float, float, List[Tuple[float, str]]]:
    rows = read_csv(csv_path)
    if limit and limit > 0:
        rows = rows[:limit]

    diffs = []
    for r in rows:
        full = r.get("full_text", "")
        z_csv = parse_stat_value(r)
        z_new = kgw_score(kgw, full)
        diffs.append((abs(z_new - z_csv), r.get("prompt_text", "")[:120]))

    if not diffs:
        return 0, float("nan"), float("nan"), []

    diffs_sorted = sorted(diffs, key=lambda x: -x[0])
    max_abs = diffs_sorted[0][0]
    mean_abs = sum(d for d, _ in diffs) / len(diffs)
    return len(diffs), max_abs, mean_abs, diffs_sorted[:10]


def compare_on_neg_stats(kgw, hf_csv: str, neg_stats_csv: str, limit: int = 0) -> Tuple[int, float, float, List[Tuple[float, str]]]:
    hf = read_csv(hf_csv)
    neg = read_csv(neg_stats_csv)

    hf_map = {norm_prompt(r.get("prompt_text", "")): r for r in hf if r.get("prompt_text")}
    neg_map = {norm_prompt(r.get("prompt_text", "")): r for r in neg if r.get("prompt_text")}

    keys = sorted(set(hf_map.keys()) & set(neg_map.keys()))
    if limit and limit > 0:
        keys = keys[:limit]

    diffs = []
    for k in keys:
        full = hf_map[k].get("full_text", "")
        z_csv = float(neg_map[k].get("stat_value", "nan"))
        z_new = kgw_score(kgw, full)
        diffs.append((abs(z_new - z_csv), hf_map[k].get("prompt_text", "")[:120]))

    if not diffs:
        return 0, float("nan"), float("nan"), []

    diffs_sorted = sorted(diffs, key=lambda x: -x[0])
    max_abs = diffs_sorted[0][0]
    mean_abs = sum(d for d, _ in diffs) / len(diffs)
    return len(diffs), max_abs, mean_abs, diffs_sorted[:10]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head_dir", required=True, help="outputs/c4_samples_head_200")
    ap.add_argument("--kgw_config", required=True, help="path to KGW config json (e.g., config/KGW.json)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--deltas", default="1,2,3,4,5")
    ap.add_argument("--limit", type=int, default=0, help="limit rows per csv (0 means all)")
    args = ap.parse_args()

    dt = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    device = torch.device(args.device)

    meta = load_json(os.path.join(args.head_dir, "run_metadata.json"))
    model_path = meta["model"]

    # tokenizer（KGW detection 只需要 tokenizer/vocab_size/device；不需要真的跑 generate）
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token

    # ⚠️ KGWConfig 里会读 transformers_config.model，但我们这里只做 detect，不用 model.generate
    dummy_model = object()

    tf_cfg = build_transformers_config(
        model=dummy_model,
        tokenizer=tok,
        device=device,
        dtype=dt,
        gen_kwargs={},
    )

    KGW = import_kgw()
    kgw = KGW(args.kgw_config, tf_cfg)

    # 1) 验证 NEG：hf_generate.csv vs neg_stats_kgw.csv
    hf_csv = os.path.join(args.head_dir, "hf_generate.csv")
    neg_stats = os.path.join(args.head_dir, "neg_stats_kgw.csv")
    if os.path.exists(hf_csv) and os.path.exists(neg_stats):
        n, mx, ma, top = compare_on_neg_stats(kgw, hf_csv, neg_stats, limit=args.limit)
        print("\n[NEG] compare hf_generate.csv  VS  neg_stats_kgw.csv")
        print(f"  compared={n}  max_abs_diff={mx:.6g}  mean_abs_diff={ma:.6g}")
        for d, p in top:
            print(f"   - diff={d:.6g}  prompt={p}")
    else:
        print("\n[NEG] skip (missing hf_generate.csv or neg_stats_kgw.csv)")

    # 2) 验证 POS：kgw_delta*.csv 的 stat_value
    deltas = [int(x.strip()) for x in args.deltas.split(",") if x.strip()]
    for d in deltas:
        path = os.path.join(args.head_dir, f"kgw_delta{d}.csv")
        if not os.path.exists(path):
            print(f"\n[POS] skip delta={d} (missing {path})")
            continue
        n, mx, ma, top = compare_on_pos_csv(kgw, path, limit=args.limit)
        print(f"\n[POS] compare {os.path.basename(path)}")
        print(f"  compared={n}  max_abs_diff={mx:.6g}  mean_abs_diff={ma:.6g}")
        for diff, p in top:
            print(f"   - diff={diff:.6g}  prompt={p}")

    print("\n[OK] done.")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
