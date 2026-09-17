#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute trace_v6_attack_process-style summary stats for multiple attack ratios
over many rows, without writing per-token TSVs.

Outputs:
  - per-row CSV with tail_match_rate, green_flip_rate, extra_token_count, traced tokens, etc.
  - per-ratio mean CSV (same metrics averaged over rows).

Example:
TOKENIZERS_PARALLELISM=false /home/star/jf/python/stega/bin/python \
  scripts/trace_v6_attack_grid.py \
  --run_meta outputs/c4_samples_head_200/run_metadata.json \
  --v6_config config/ByteKGWv6.json \
  --input_csv outputs/v6_vs_kgw_gen/bytekgw_v6_delta2.0.csv \
  --attack_ratios 0,0.01,0.02,0.05,0.10 \
  --attack_seed 0 \
  --indices all \
  --max_tokens 200 --stride 1 \
  --device cuda:0 \
  --out_csv outputs/trace_v6_attack_grid.csv
"""

from __future__ import annotations

import argparse
import json
import concurrent.futures
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer
from tqdm import tqdm

# ------------------------------------
# imports from repo (robust to path)
# ------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from MarkLLM.watermark.bytekgwV6.token_bytes import TokenByteVocabV6  # type: ignore
    from MarkLLM.watermark.bytekgwV6.prf import RobustPartitioner  # type: ignore
except ModuleNotFoundError:
    from watermark.bytekgwV6.token_bytes import TokenByteVocabV6  # type: ignore
    from watermark.bytekgwV6.prf import RobustPartitioner  # type: ignore


# ------------------------------------
# helpers
# ------------------------------------
def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _to_bytes(key) -> bytes:
    if isinstance(key, bytes):
        return key
    if isinstance(key, int):
        return int(key).to_bytes(8, "little", signed=False)
    if isinstance(key, str):
        s = key.strip()
        try:
            if s.startswith("0x"):
                return bytes.fromhex(s[2:])
            return bytes.fromhex(s)
        except Exception:
            return s.encode("utf-8")
    raise TypeError(f"Unsupported key type: {type(key)}")


def popcount_bytes(x: bytes) -> int:
    return sum(int(b).bit_count() for b in x)


def tail_str(s: str, n: int) -> str:
    return s[-n:] if n > 0 else ""


def tokenize_with_offsets(tok, text: str, add_special_tokens: bool):
    enc = tok(
        text,
        return_tensors="pt",
        add_special_tokens=add_special_tokens,
        return_offsets_mapping=True,
    )
    ids = enc["input_ids"][0]
    offsets = None
    if "offset_mapping" in enc:
        try:
            offsets = [tuple(map(int, x)) for x in enc["offset_mapping"][0].tolist()]
        except Exception:
            offsets = None
    toks = tok.convert_ids_to_tokens(ids.tolist())
    return ids, offsets, toks


def apply_char_attack_replace_X(full_text: str, prompt: str, attack_ratio: float, seed: int) -> Tuple[str, List[int]]:
    if attack_ratio <= 0:
        return full_text, []
    rng = np.random.RandomState(seed)
    start = len(prompt)
    cont = full_text[start:]
    n = len(cont)
    if n <= 0:
        return full_text, []
    k = int(round(n * attack_ratio))
    k = max(1, k)
    k = min(k, n)
    idxs = rng.choice(n, size=k, replace=False)
    idxs = sorted(int(i) for i in idxs)
    cont_list = list(cont)
    for i in idxs:
        cont_list[i] = "X"
    attacked = full_text[:start] + "".join(cont_list)
    abs_idxs = [start + i for i in idxs]
    return attacked, abs_idxs


def build_v6_partitioner(v6_cfg: Dict[str, Any]) -> RobustPartitioner:
    return RobustPartitioner(
        master_key=_to_bytes(v6_cfg.get("hash_key", 15485863)),
        m_bits=int(v6_cfg.get("m_bits", 256)),
        target_anchors=int(v6_cfg.get("target_anchors", 96)),
        k_choices=tuple(v6_cfg.get("k_choices", [4, 5, 6])),
        normalize_whitespace=bool(v6_cfg.get("normalize_whitespace", True)),
    )


# ------------------------------------
# core
# ------------------------------------
def compute_row_summary(
    row: pd.Series,
    tok,
    vocab: TokenByteVocabV6,
    partitioner: RobustPartitioner,
    device: torch.device,
    seed_window_chars: int,
    n_bytes: int,
    attack_ratio: float,
    attack_seed: int,
    max_tokens: int,
    stride: int,
) -> Dict[str, Any]:
    full: str = row["full_text"]
    prompt: str = row.get("prompt_text", "")
    attacked, abs_attacked_chars = apply_char_attack_replace_X(full, prompt, attack_ratio, attack_seed)

    ids_c, offs_c, toks_c = tokenize_with_offsets(tok, full, add_special_tokens=True)
    ids_a, offs_a, toks_a = tokenize_with_offsets(tok, attacked, add_special_tokens=True)
    if offs_c is None or offs_a is None:
        raise RuntimeError("Tokenizer offsets unavailable; use a fast tokenizer.")

    firstn_ids = vocab.first_n_id(device, n_bytes)  # [V]

    # map start offset -> index for clean
    clean_start2i: Dict[int, int] = {}
    for i, (s, _e) in enumerate(offs_c):
        clean_start2i[s] = i

    prompt_len_chars = len(prompt)

    tail_eq_cnt = 0
    tail_den = 0
    flip_cnt = 0
    flip_den = 0
    extra_tok_cnt = 0
    traced = 0

    for pos_a, (s_a, e_a) in enumerate(offs_a):
        if traced >= max_tokens:
            break
        if s_a < prompt_len_chars:
            continue
        if (pos_a % max(1, stride)) != 0:
            continue

        traced += 1
        tok_id_a = int(ids_a[pos_a].item())
        uid_a = int(firstn_ids[tok_id_a].item())

        prefix_end = s_a
        prefix_a = attacked[:prefix_end]
        prefix_c = full[:min(prefix_end, len(full))]

        fp_a = partitioner.fingerprint(prefix_a, seed_window_chars)
        fp_c = partitioner.fingerprint(prefix_c, seed_window_chars)
        tail_a = tail_str(prefix_a, seed_window_chars)
        tail_c = tail_str(prefix_c, seed_window_chars)

        is_green_a = int(partitioner.is_green(uid_a, fp_a))
        is_green_uid_under_clean_fp = int(partitioner.is_green(uid_a, fp_c))
        green_flip_cleanfp_vs_attackfp = int(is_green_a != is_green_uid_under_clean_fp)

        tail_equal = int(tail_a == tail_c)
        tail_den += 1
        tail_eq_cnt += tail_equal

        flip_den += 1
        flip_cnt += green_flip_cleanfp_vs_attackfp

        pos_c = clean_start2i.get(s_a, -1)
        if pos_c == -1:
            extra_tok_cnt += 1

    tail_match_rate = float(tail_eq_cnt / max(1, tail_den))
    green_flip_rate = float(flip_cnt / max(1, flip_den))
    return {
        "row_idx": int(row.name),
        "attack_ratio": attack_ratio,
        "seed_window_chars": seed_window_chars,
        "n_bytes": n_bytes,
        "clean_tokens": int(ids_c.numel()),
        "attacked_tokens": int(ids_a.numel()),
        "extra_token_count_in_traced": int(extra_tok_cnt),
        "traced_tokens": int(traced),
        "tail_match_rate_over_traced": tail_match_rate,
        "green_flip_rate_uid_cleanfp_vs_attackfp": green_flip_rate,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_meta", required=True)
    ap.add_argument("--v6_config", required=True)
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--attack_ratios", default="0.0,0.02,0.05,0.10")
    ap.add_argument("--attack_seed", type=int, default=0)
    ap.add_argument("--indices", default="all", help="comma-separated row indices or 'all'")
    ap.add_argument("--max_tokens", type=int, default=200)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--num_workers", type=int, default=1, help=">1 for multiprocessing (CPU only recommended)")
    ap.add_argument("--out_csv", default="trace_v6_attack_grid.csv")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    use_gpu = args.device.startswith("cuda") and torch.cuda.is_available()
    if use_gpu and args.num_workers > 1:
        raise ValueError("Multi-worker is CPU-only. Set --device cpu or --num_workers 1 for GPU.")
    device = torch.device(args.device if use_gpu else "cpu")

    run_meta = load_json(args.run_meta)
    v6_cfg = load_json(args.v6_config)

    model_path = run_meta["model"]
    tok = AutoTokenizer.from_pretrained(model_path)

    add_special_tokens = bool(v6_cfg.get("add_special_tokens", True))
    seed_window_chars = int(v6_cfg.get("seed_window_chars", 18))
    n_bytes = int(v6_cfg.get("n_bytes", 3))

    partitioner = build_v6_partitioner(v6_cfg)
    vocab = TokenByteVocabV6.from_tokenizer(tok, skip_markers=True).to(device)

    df = pd.read_csv(args.input_csv)
    if args.indices.strip().lower() == "all":
        wanted = list(range(len(df)))
    else:
        wanted = [int(x) for x in args.indices.split(",") if x.strip()]

    ratios = [float(x) for x in args.attack_ratios.split(",") if x.strip()]

    rows_out: List[Dict[str, Any]] = []
    if args.num_workers == 1:
        for r in ratios:
            print(f"[attack_ratio={r}] processing {len(wanted)} rows...")
            for row_idx in tqdm(wanted, desc=f"r={r}", leave=False):
                row = df.iloc[row_idx]
                summary = compute_row_summary(
                    row=row,
                    tok=tok,
                    vocab=vocab,
                    partitioner=partitioner,
                    device=device,
                    seed_window_chars=seed_window_chars,
                    n_bytes=n_bytes,
                    attack_ratio=r,
                    attack_seed=args.attack_seed + row_idx,
                    max_tokens=args.max_tokens,
                    stride=args.stride,
                )
                rows_out.append(summary)
    else:
        # multiprocessing (CPU only)
        def process_chunk(chunk_indices: List[int]) -> List[Dict[str, Any]]:
            run_meta_local = load_json(args.run_meta)
            v6_cfg_local = load_json(args.v6_config)
            tok_local = AutoTokenizer.from_pretrained(run_meta_local["model"])
            partitioner_local = build_v6_partitioner(v6_cfg_local)
            vocab_local = TokenByteVocabV6.from_tokenizer(tok_local, skip_markers=True).to(device)
            seed_window_chars_local = int(v6_cfg_local.get("seed_window_chars", 18))
            n_bytes_local = int(v6_cfg_local.get("n_bytes", 3))
            df_local = pd.read_csv(args.input_csv)
            out_local: List[Dict[str, Any]] = []
            for r in ratios:
                for row_idx in chunk_indices:
                    row = df_local.iloc[row_idx]
                    summary = compute_row_summary(
                        row=row,
                        tok=tok_local,
                        vocab=vocab_local,
                        partitioner=partitioner_local,
                        device=device,
                        seed_window_chars=seed_window_chars_local,
                        n_bytes=n_bytes_local,
                        attack_ratio=r,
                        attack_seed=args.attack_seed + row_idx,
                        max_tokens=args.max_tokens,
                        stride=args.stride,
                    )
                    out_local.append(summary)
            return out_local

        chunks = np.array_split(wanted, args.num_workers)
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as ex:
            futures = [ex.submit(process_chunk, chunk.tolist()) for chunk in chunks]
            for fut in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="chunks"):
                rows_out.extend(fut.result())

    out_df = pd.DataFrame(rows_out)
    out_path = Path(args.out_csv)
    out_df.to_csv(out_path, index=False)

    mean_df = (
        out_df.groupby("attack_ratio")
        .agg(
            mean_tail_match=("tail_match_rate_over_traced", "mean"),
            mean_green_flip=("green_flip_rate_uid_cleanfp_vs_attackfp", "mean"),
            mean_extra_tokens=("extra_token_count_in_traced", "mean"),
            mean_traced=("traced_tokens", "mean"),
        )
        .reset_index()
    )
    mean_path = out_path.with_name(out_path.stem + "_mean.csv")
    mean_df.to_csv(mean_path, index=False)

    print(f"wrote per-row to {out_path} (rows={len(out_df)})")
    print(f"wrote per-ratio mean to {mean_path}")


if __name__ == "__main__":
    main()
