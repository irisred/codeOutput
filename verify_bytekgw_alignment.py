#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import inspect
import json
import math
import os
import random
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ----------------------------
# CSV utils
# ----------------------------
def read_csv(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def pick_stat(row: Dict[str, Any]) -> float:
    for k in ["stat_value", "stat", "z", "score"]:
        if k in row and row[k] not in (None, ""):
            try:
                return float(row[k])
            except Exception:
                pass
    return float("nan")


def pick_prompt(row: Dict[str, Any]) -> str:
    for k in ["prompt_text", "prompt", "input", "instruction"]:
        if k in row and row[k]:
            return str(row[k])
    return ""


def pick_full_text(row: Dict[str, Any]) -> str:
    for k in ["full_text", "text"]:
        if k in row and row[k]:
            return str(row[k])

    p = ""
    c = ""
    for k in ["prompt_text", "prompt"]:
        if k in row and row[k]:
            p = str(row[k])
            break
    for k in ["completion_text", "completion", "output"]:
        if k in row and row[k]:
            c = str(row[k])
            break
    if p or c:
        return p + c
    return ""


def key_prompt(s: str) -> str:
    return " ".join((s or "").strip().split())


def finite(x: float) -> bool:
    return (x is not None) and (not math.isnan(x)) and (not math.isinf(x))


# ----------------------------
# MarkLLM glue
# ----------------------------
def build_transformers_config(model, tokenizer, device: str, dtype, gen_kwargs: Dict[str, Any]):
    """
    兼容不同版本 MarkLLM TransformersConfig 构造签名
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

    # 兜底：把缺的字段手动塞进去
    for k, v in cand.items():
        if not hasattr(obj, k):
            try:
                setattr(obj, k, v)
            except Exception:
                pass
    return obj


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def materialize_cfg(cfg: Dict[str, Any]) -> str:
    fd, tmp = tempfile.mkstemp(prefix="tmp_bytekgw_cfg_", suffix=".json")
    os.close(fd)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    return tmp


def build_bytekgw(
    bytekgw_cfg_path: str,
    tf_cfg,
    *,
    force_max_byte_pos: Optional[int],
    use_prefix_bytes_in_prf: Optional[bool],
    add_special_tokens_cfg: Optional[bool],
):
    """
    用 ByteKGWv5.detect_watermark() 复算 z，和 CSV stat_value 对齐。
    """
    cfg = load_json(bytekgw_cfg_path)

    cfg["algorithm_name"] = "ByteKGWv5"
    cfg["scheme"] = cfg.get("scheme", "byte_tree")
    cfg["delta"] = float(cfg.get("delta", 1.0))

    if force_max_byte_pos is not None:
        cfg["max_byte_pos"] = int(force_max_byte_pos)
        # detector_mode 对 head/all 可能有帮助
        if int(force_max_byte_pos) == 1:
            cfg["detector_mode"] = "first_byte"
        else:
            cfg.setdefault("detector_mode", "all_bytes")

    if use_prefix_bytes_in_prf is not None:
        cfg["use_prefix_bytes_in_prf"] = bool(use_prefix_bytes_in_prf)
    if add_special_tokens_cfg is not None:
        cfg["add_special_tokens"] = bool(add_special_tokens_cfg)

    tmp = materialize_cfg(cfg)
    from MarkLLM.watermark.bytekgwV5 import ByteKGWv5
    return ByteKGWv5(tmp, tf_cfg), cfg, tmp


def _as_float(x: Any) -> float:
    if x is None:
        return float("nan")
    if torch.is_tensor(x):
        return float(x.detach().cpu().view(-1)[0].item())
    if isinstance(x, (list, tuple)) and len(x) > 0:
        return float(x[0])
    try:
        import numpy as np  # optional
        if isinstance(x, np.ndarray):
            return float(x.reshape(-1)[0])
    except Exception:
        pass
    return float(x)


@torch.inference_mode()
def detect_z(bytekgw, text: str, add_special_tokens_call: bool, which: str = "z") -> float:
    raw = bytekgw.detect_watermark(text, return_dict=True, add_special_tokens=add_special_tokens_call)
    if not isinstance(raw, dict):
        raise RuntimeError(f"Unexpected detect_watermark return type: {type(raw)}")
    if which in raw:
        return _as_float(raw[which])
    # fallback
    for k in ["z", "z_unweighted", "score"]:
        if k in raw:
            return _as_float(raw[k])
    raise RuntimeError(f"detect_watermark dict missing z fields: keys={list(raw.keys())}")


def eval_alignment(
    *,
    name: str,
    csv_path: str,
    bytekgw,
    add_special_tokens_call: bool,
    which: str,
    limit: int,
    seed: int,
) -> Dict[str, Any]:
    rows = read_csv(csv_path)

    items = []
    for r in rows:
        p = key_prompt(pick_prompt(r))
        t = pick_full_text(r)
        z = pick_stat(r)
        if not p or not t or not finite(z):
            continue
        items.append((p, t, z))

    if not items:
        raise RuntimeError(f"[{name}] No valid rows with prompt+text+stat in {csv_path}")

    rng = random.Random(seed)
    if limit > 0 and len(items) > limit:
        items = rng.sample(items, limit)

    diffs = []
    worst = None  # (absdiff, prompt, z_csv, z_new)

    for i, (p, t, z_csv) in enumerate(items, 1):
        z_new = detect_z(bytekgw, t, add_special_tokens_call=add_special_tokens_call, which=which)
        ad = abs(z_new - z_csv)
        diffs.append(ad)
        if (worst is None) or (ad > worst[0]):
            worst = (ad, p, z_csv, z_new)
        if i % 25 == 0 or i == len(items):
            print(f"[{name}] which={which} addsp={add_special_tokens_call}  scored {i}/{len(items)}", flush=True)

    diffs_sorted = sorted(diffs)

    def pct(pctv: float) -> float:
        if not diffs_sorted:
            return float("nan")
        k = int(round((pctv / 100.0) * (len(diffs_sorted) - 1)))
        k = min(max(k, 0), len(diffs_sorted) - 1)
        return float(diffs_sorted[k])

    out = {
        "name": name,
        "csv": csv_path,
        "n": len(diffs),
        "which": which,
        "add_special_tokens_call": add_special_tokens_call,
        "mean_abs_diff": float(sum(diffs) / len(diffs)),
        "max_abs_diff": float(max(diffs)),
        "p50_abs_diff": pct(50),
        "p90_abs_diff": pct(90),
        "p99_abs_diff": pct(99),
        "worst_prompt": (worst[1] if worst else ""),
        "worst_csv": (worst[2] if worst else float("nan")),
        "worst_new": (worst[3] if worst else float("nan")),
    }
    return out


def run_one(name: str, csv_path: str, wm, limit: int, seed: int) -> Dict[str, Any]:
    """
    自动比较四种组合：
      add_special_tokens_call ∈ {True, False}
      which ∈ {z, z_unweighted}
    选 max_abs_diff 最小的一组作为“对齐模式”
    """
    candidates = []
    for addsp in [True, False]:
        for which in ["z", "z_unweighted"]:
            r = eval_alignment(
                name=name,
                csv_path=csv_path,
                bytekgw=wm,
                add_special_tokens_call=addsp,
                which=which,
                limit=limit,
                seed=seed,
            )
            candidates.append(r)

    best = min(candidates, key=lambda x: (x["max_abs_diff"], x["mean_abs_diff"]))
    print("\n" + "-" * 110)
    print(f"[RESULT] {name}  csv={os.path.basename(csv_path)}")
    for r in sorted(candidates, key=lambda x: (x["max_abs_diff"], x["mean_abs_diff"])):
        print(f"  which={r['which']:<12} addsp={str(r['add_special_tokens_call']):<5} "
              f"max={r['max_abs_diff']:.6g} mean={r['mean_abs_diff']:.6g} p99={r['p99_abs_diff']:.6g}")
    print(f"  ==> BEST: which={best['which']} addsp={best['add_special_tokens_call']}  "
          f"max_abs_diff={best['max_abs_diff']:.6g} mean_abs_diff={best['mean_abs_diff']:.6g}")
    print(f"  worst prompt: {best['worst_prompt'][:120]}")
    print(f"  worst: csv={best['worst_csv']:.6f} new={best['worst_new']:.6f}")
    return best


# ----------------------------
# main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head_dir", required=True)
    ap.add_argument("--all_dir", required=True)
    ap.add_argument("--bytekgw_config", default="config/ByteKGWv5.json")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--deltas", default="1,2,3,4,5")
    ap.add_argument("--limit", type=int, default=50, help="sample size per csv (0=all)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--all_csv_suffix", default="", help="e.g. _reweighted")
    args = ap.parse_args()

    deltas = [int(x.strip()) for x in args.deltas.split(",") if x.strip()]
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    device = args.device

    head_meta = load_json(os.path.join(args.head_dir, "run_metadata.json"))
    all_meta = load_json(os.path.join(args.all_dir, "run_metadata.json"))
    model_path = head_meta["model"]

    head_addsp_cfg = bool(head_meta.get("add_special_tokens", True))
    all_addsp_cfg = bool(all_meta.get("add_special_tokens", True))

    head_use_prefix = bool(head_meta.get("use_prefix_bytes_in_prf", True))
    all_use_prefix = bool(all_meta.get("use_prefix_bytes_in_prf", True))

    all_max_byte_pos = int(all_meta.get("bytekgw_all_max_byte_pos", all_meta.get("max_byte_pos", 64)))

    print("=" * 110)
    print(f"[CFG] model={model_path}")
    print(f"[CFG] device={device} dtype={args.dtype}")
    print(f"[HEAD meta] add_special_tokens={head_addsp_cfg} use_prefix_bytes_in_prf={head_use_prefix} force_max_byte_pos=1")
    print(f"[ALL  meta] add_special_tokens={all_addsp_cfg}  use_prefix_bytes_in_prf={all_use_prefix}  force_max_byte_pos={all_max_byte_pos}")
    print("=" * 110)

    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.eval()
    model.to(device)

    tf_cfg = build_transformers_config(model, tok, device=device, dtype=dtype, gen_kwargs={})

    byte_head, _, _ = build_bytekgw(
        args.bytekgw_config,
        tf_cfg,
        force_max_byte_pos=1,
        use_prefix_bytes_in_prf=head_use_prefix,
        add_special_tokens_cfg=head_addsp_cfg,
    )
    byte_all, _, _ = build_bytekgw(
        args.bytekgw_config,
        tf_cfg,
        force_max_byte_pos=all_max_byte_pos,
        use_prefix_bytes_in_prf=all_use_prefix,
        add_special_tokens_cfg=all_addsp_cfg,
    )

    head_paths = [os.path.join(args.head_dir, f"bytekgw_head_delta{d}.csv") for d in deltas]
    all_paths = [os.path.join(args.all_dir, f"bytekgw_all_delta{d}{args.all_csv_suffix}.csv") for d in deltas]
    for p in head_paths + all_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    best_summaries = []
    for d, hp, apath in zip(deltas, head_paths, all_paths):
        best_summaries.append(run_one(f"bytekgw_head[d={d}]", hp, byte_head, limit=args.limit, seed=args.seed))
        best_summaries.append(run_one(f"bytekgw_all[d={d}]", apath, byte_all, limit=args.limit, seed=args.seed))

    print("\n" + "=" * 110)
    print("SUMMARY (best mode per file):")
    for s in best_summaries:
        print(f"- {s['name']}: BEST which={s['which']:<12} addsp={s['add_special_tokens_call']}  "
              f"max_abs_diff={s['max_abs_diff']:.6g} mean_abs_diff={s['mean_abs_diff']:.6g} n={s['n']}")
    print("=" * 110)

    print("\n判定建议：")
    print("- 若 BEST max_abs_diff ~ 1e-6 到 1e-4：基本确认 detector 与 CSV stat_value 对齐。")
    print("- 若 BEST max_abs_diff >= 1e-2：说明参数（add_special_tokens/use_prefix_bytes_in_prf/max_byte_pos/prefix_length）或 CSV 字段不一致。")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
