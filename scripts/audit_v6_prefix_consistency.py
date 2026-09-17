#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audit ByteKGWv6 prefix consistency:
Compare fingerprint/is_green computed from
  (A) tokenizer.decode(prefix_ids)   [detector-style]
vs
  (B) raw text slice text[:char_start]  [offset-style]

It reports:
- tail_equal rate
- fp_equal rate + fp_hd distribution
- is_green mismatch rate (decode vs slice)
- dumps top mismatching examples to TSV

Usage:
TOKENIZERS_PARALLELISM=false /home/star/jf/python/stega/bin/python \
  scripts/audit_v6_prefix_consistency.py \
  --run_meta outputs/c4_samples_head_200/run_metadata.json \
  --v6_config config/ByteKGWv6.json \
  --input_csv outputs/v6_vs_kgw_gen/bytekgw_v6_delta2.0.csv \
  --indices 0,1,2 \
  --max_tokens 200 \
  --stride 1 \
  --device cuda:0 \
  --out_dir audits_v6_prefix

Optional (also audit attacked text):
  --attack_ratio 0.02 --attack_seed 0
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import inspect
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
from transformers import AutoTokenizer


# ----------------------------
# robust imports for your repo
# ----------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _try_import_v6() -> Tuple[Any, Any]:
    """
    Return (TokenByteVocabV6, RobustPartitioner).
    Try common roots to avoid the MarkLLM import error you saw.
    """
    candidates = [
        ("MarkLLM.watermark.bytekgwV6.token_bytes", "TokenByteVocabV6",
         "MarkLLM.watermark.bytekgwV6.prf", "RobustPartitioner"),
        ("watermark.bytekgwV6.token_bytes", "TokenByteVocabV6",
         "watermark.bytekgwV6.prf", "RobustPartitioner"),
    ]
    last_err = None
    for m1, c1, m2, c2 in candidates:
        try:
            mod1 = importlib.import_module(m1)
            mod2 = importlib.import_module(m2)
            return getattr(mod1, c1), getattr(mod2, c2)
        except Exception as e:
            last_err = e
            continue
    raise ModuleNotFoundError(
        f"Cannot import TokenByteVocabV6/RobustPartitioner. Last error: {last_err}. "
        "Fix: ensure repo root is on PYTHONPATH or adjust import paths."
    )


TokenByteVocabV6, RobustPartitioner = _try_import_v6()


# ----------------------------
# helpers
# ----------------------------
def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _to_bytes(key) -> bytes:
    """
    RobustPartitioner expects a bytes master_key. Accept int/str/bytes.
    """
    if isinstance(key, bytes):
        return key
    if isinstance(key, str):
        s = key.strip()
        if s.startswith("0x"):
            try:
                return int(s, 16).to_bytes(16, "little", signed=False)
            except Exception:
                pass
        return s.encode("utf-8")
    try:
        return int(key).to_bytes(16, "little", signed=False)
    except Exception:
        return b"default-key"


def xor_hd_bits(a: bytes, b: bytes) -> int:
    # a,b same length
    return sum((x ^ y).bit_count() for x, y in zip(a, b))


def tail_str(s: str, n: int) -> str:
    if n <= 0:
        return ""
    return s[-n:]


def apply_char_attack_replace_X(full_text: str, prompt: str, ratio: float, seed: int) -> str:
    if ratio <= 0:
        return full_text
    rng = random.Random(seed)

    start = len(prompt) if full_text.startswith(prompt) else 0
    cont = full_text[start:]
    n = len(cont)
    if n <= 0:
        return full_text

    k = max(1, int(round(n * ratio)))
    k = min(k, n)
    idxs = rng.sample(range(n), k)
    cont_list = list(cont)
    for i in idxs:
        cont_list[i] = "X"
    return full_text[:start] + "".join(cont_list)


def tokenize_with_offsets(tok, text: str, add_special_tokens: bool) -> Tuple[List[int], List[Tuple[int, int]]]:
    enc = tok(
        text,
        add_special_tokens=add_special_tokens,
        return_offsets_mapping=True,
    )
    if "offset_mapping" not in enc or enc["offset_mapping"] is None:
        raise RuntimeError("offset_mapping unavailable; please use a fast tokenizer.")
    ids = enc["input_ids"]
    offsets = [(int(a), int(b)) for (a, b) in enc["offset_mapping"]]
    return ids, offsets


def build_partitioner(v6_cfg: Dict[str, Any]) -> Any:
    """
    Try a few constructor patterns to match your repo's RobustPartitioner.
    """
    # common fields
    hash_key = _to_bytes(v6_cfg.get("hash_key", 15485863))
    m_bits = int(v6_cfg.get("m_bits", 256))
    target_anchors = int(v6_cfg.get("target_anchors", 96))
    k_choices = v6_cfg.get("k_choices", [4, 5, 6])
    normalize_whitespace = bool(v6_cfg.get("normalize_whitespace", True))

    # if classmethod exists
    if hasattr(RobustPartitioner, "from_config") and callable(getattr(RobustPartitioner, "from_config")):
        try:
            return RobustPartitioner.from_config(v6_cfg)
        except Exception:
            pass

    # try kwargs (most likely)
    tries = [
        lambda: RobustPartitioner(
            master_key=hash_key,
            m_bits=m_bits,
            target_anchors=target_anchors,
            k_choices=tuple(k_choices),
            normalize_whitespace=normalize_whitespace,
        ),
        lambda: RobustPartitioner(
            hash_key=hash_key,
            m_bits=m_bits,
            target_anchors=target_anchors,
            k_choices=tuple(k_choices),
            normalize_whitespace=normalize_whitespace,
        ),
        lambda: RobustPartitioner(hash_key, m_bits, target_anchors, tuple(k_choices), normalize_whitespace),
        lambda: RobustPartitioner(hash_key, m_bits=m_bits, target_anchors=target_anchors, k_choices=tuple(k_choices)),
    ]
    last = None
    for fn in tries:
        try:
            return fn()
        except Exception as e:
            last = e
    raise RuntimeError(f"Cannot construct RobustPartitioner with given config. Last error: {last}")


# ----------------------------
# audit core
# ----------------------------
def audit_one_text(
    *,
    name: str,
    text: str,
    prompt_text: str,
    tokenizer,
    firstn_ids: torch.Tensor,
    partitioner,
    seed_window_chars: int,
    add_special_tokens: bool,
    max_tokens: int,
    stride: int,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    ids, offsets = tokenize_with_offsets(tokenizer, text, add_special_tokens=add_special_tokens)

    prompt_chars = len(prompt_text) if text.startswith(prompt_text) else 0

    rows: List[Dict[str, Any]] = []
    total = 0
    tail_eq = 0
    fp_eq = 0
    green_eq = 0
    fp_hd_sum = 0
    fp_hd_nonzero = 0
    green_flip = 0

    for idx in range(len(ids)):
        if total >= max_tokens:
            break
        if (idx % max(1, stride)) != 0:
            continue

        char_start = offsets[idx][0]
        if char_start < prompt_chars:
            continue

        tok_id = int(ids[idx])
        uid = int(firstn_ids[tok_id].item())

        prefix_ids = ids[:idx]
        prefix_decode = tokenizer.decode(
            prefix_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        prefix_slice = text[:char_start]

        tail_d = tail_str(prefix_decode, seed_window_chars)
        tail_s = tail_str(prefix_slice, seed_window_chars)
        teq = int(tail_d == tail_s)

        fp_d = partitioner.fingerprint(prefix_decode, seed_window_chars)
        fp_s = partitioner.fingerprint(prefix_slice, seed_window_chars)
        fheq = int(fp_d == fp_s)
        hd = xor_hd_bits(fp_d, fp_s)

        g_d = int(partitioner.is_green(uid, fp_d))
        g_s = int(partitioner.is_green(uid, fp_s))
        geq = int(g_d == g_s)

        total += 1
        tail_eq += teq
        fp_eq += fheq
        green_eq += geq
        fp_hd_sum += hd
        if hd != 0:
            fp_hd_nonzero += 1
        if not geq:
            green_flip += 1

        rows.append(
            {
                "which": name,
                "pos": idx,
                "char_start": char_start,
                "token_id": tok_id,
                "uid": uid,
                "token_str": tokenizer.convert_ids_to_tokens([tok_id])[0],
                "tail_decode": tail_d.replace("\n", "\\n").replace("\t", "\\t"),
                "tail_slice": tail_s.replace("\n", "\\n").replace("\t", "\\t"),
                "tail_equal": teq,
                "fp_hd_bits": hd,
                "fp_equal": fheq,
                "is_green_decode": g_d,
                "is_green_slice": g_s,
                "green_equal": geq,
            }
        )

    df = pd.DataFrame(rows)
    summary = {
        "which": name,
        "positions_audited": int(total),
        "tail_equal_rate": float(tail_eq / max(total, 1)),
        "fp_equal_rate": float(fp_eq / max(total, 1)),
        "fp_hd_bits_mean": float(fp_hd_sum / max(total, 1)),
        "fp_hd_nonzero_rate": float(fp_hd_nonzero / max(total, 1)),
        "green_equal_rate": float(green_eq / max(total, 1)),
        "green_flip_rate": float(green_flip / max(total, 1)),
    }
    return summary, df


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_meta", required=True)
    ap.add_argument("--v6_config", required=True)
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--indices", default="0,1,2")
    ap.add_argument("--max_tokens", type=int, default=200)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--attack_ratio", type=float, default=0.0)
    ap.add_argument("--attack_seed", type=int, default=0)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_meta = load_json(args.run_meta)
    v6_cfg = load_json(args.v6_config)

    model_name = run_meta["model"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # config params
    add_special_tokens = bool(v6_cfg.get("add_special_tokens", True))
    n_bytes = int(v6_cfg.get("n_bytes", 3))
    seed_window_chars = int(v6_cfg.get("seed_window_chars", 18))

    # partitioner + vocab
    partitioner = build_partitioner(v6_cfg)

    vocab = TokenByteVocabV6.from_tokenizer(tokenizer, skip_markers=True).to(args.device)
    firstn_ids = vocab.first_n_id(torch.device(args.device), n_bytes).to("cpu")

    df_in = pd.read_csv(args.input_csv)
    indices = [int(x.strip()) for x in args.indices.split(",") if x.strip()]

    all_summaries: List[Dict[str, Any]] = []

    for ridx in indices:
        row = df_in.iloc[ridx]
        full = str(row["full_text"])
        prompt = str(row.get("prompt_text", ""))

        # clean audit
        s_clean, df_clean = audit_one_text(
            name=f"row{ridx}:clean",
            text=full,
            prompt_text=prompt,
            tokenizer=tokenizer,
            firstn_ids=firstn_ids,
            partitioner=partitioner,
            seed_window_chars=seed_window_chars,
            add_special_tokens=add_special_tokens,
            max_tokens=args.max_tokens,
            stride=args.stride,
        )
        all_summaries.append(s_clean)
        df_clean.to_csv(out_dir / f"row_{ridx}_clean_prefix_audit.tsv", sep="\t", index=False)

        print("\n[CLEAN]", json.dumps(s_clean, ensure_ascii=False, indent=2))

        # attacked audit (optional)
        if args.attack_ratio > 0:
            attacked = apply_char_attack_replace_X(full, prompt, args.attack_ratio, args.attack_seed + ridx)
            s_att, df_att = audit_one_text(
                name=f"row{ridx}:attack(r={args.attack_ratio})",
                text=attacked,
                prompt_text=prompt,
                tokenizer=tokenizer,
                firstn_ids=firstn_ids,
                partitioner=partitioner,
                seed_window_chars=seed_window_chars,
                add_special_tokens=add_special_tokens,
                max_tokens=args.max_tokens,
                stride=args.stride,
            )
            all_summaries.append(s_att)
            df_att.to_csv(out_dir / f"row_{ridx}_attack_prefix_audit.tsv", sep="\t", index=False)

            print("\n[ATTACK]", json.dumps(s_att, ensure_ascii=False, indent=2))

            # dump top mismatches for quick eyeballing
            mism = df_att[df_att["green_equal"] == 0].sort_values("fp_hd_bits", ascending=False).head(40)
            mism.to_csv(out_dir / f"row_{ridx}_attack_prefix_mismatch_top.tsv", sep="\t", index=False)

    # overall summary
    df_sum = pd.DataFrame(all_summaries)
    df_sum.to_csv(out_dir / "summary_all.tsv", sep="\t", index=False)
    (out_dir / "summary_all.json").write_text(json.dumps(all_summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote summary to: {out_dir}/summary_all.tsv")


if __name__ == "__main__":
    main()
