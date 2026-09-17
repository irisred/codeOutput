#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validate detection logic against existing CSV stats.

Example:
  python validate_detection.py --csv outputs/c4_samples_head_200/bytekgw_head_delta1.csv \
    --algo byte --model_path /path/to/model --byte_cfg config/ByteKGWv5.json --byte_maxpos 1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer


# ---------- helpers ----------
def read_json(p: Path) -> Dict[str, Any]:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def pick_text_col(df: pd.DataFrame) -> str:
    candidates = [
        "text",
        "output_text",
        "generated_text",
        "gen_text",
        "completion",
        "continuation",
        "watermarked_text",
        "response",
        "output",
        "sample",
        "full_text",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    for c in df.columns:
        if df[c].dtype == object:
            return c
    raise ValueError(f"Cannot find a text column. columns={list(df.columns)}")


def pick_z_col(df: pd.DataFrame) -> str:
    for c in ["stat_value", "z", "score", "z_score"]:
        if c in df.columns:
            return c
    raise ValueError(f"Cannot find z/stat column in CSV.")


# ---------- lightweight detectors (same logic as attack_char_3alg_mp) ----------
class _LiteKGWDetector:
    """Lightweight KGW detector that only needs tokenizer and config (no model)."""

    def __init__(self, *, cfg_path: str, tokenizer, device: str) -> None:
        cfg = read_json(Path(cfg_path))
        self.gamma = float(cfg.get("gamma", 0.5))
        self.hash_key = int(cfg.get("hash_key", 15485863))
        self.prefix_length = int(cfg.get("prefix_length", 4))
        self.z_threshold = float(cfg.get("z_threshold", 4.0))
        self.f_scheme = str(cfg.get("f_scheme", "time"))
        self.window_scheme = str(cfg.get("window_scheme", "left"))
        self.vocab_size = len(tokenizer)
        self.device = torch.device(device)
        self.tokenizer = tokenizer

        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(self.hash_key)
        self.prf = torch.randperm(self.vocab_size, device=self.device, generator=self.rng)

    def _f(self, input_ids: torch.LongTensor) -> int:
        ids = input_ids
        if self.f_scheme == "time":
            time_result = 1
            for i in range(self.prefix_length):
                time_result *= ids[-1 - i].item()
            return int(self.prf[time_result % self.vocab_size].item())
        if self.f_scheme == "additive":
            additive_result = 0
            for i in range(self.prefix_length):
                additive_result += ids[-1 - i].item()
            return int(self.prf[additive_result % self.vocab_size].item())
        if self.f_scheme == "skip":
            return int(self.prf[ids[-self.prefix_length].item()].item())
        if self.f_scheme == "min":
            return min(int(self.prf[ids[-1 - i].item()].item()) for i in range(self.prefix_length))
        raise ValueError(f"Unknown f_scheme {self.f_scheme}")

    def _get_greenlist_ids_left(self, input_ids: torch.LongTensor) -> List[int]:
        self.rng.manual_seed((self.hash_key * self._f(input_ids)) % self.vocab_size)
        greenlist_size = int(self.vocab_size * self.gamma)
        vocab_permutation = torch.randperm(self.vocab_size, device=self.device, generator=self.rng)
        return vocab_permutation[:greenlist_size].tolist()

    def _get_greenlist_ids_self(self, input_ids: torch.LongTensor) -> List[int]:
        greenlist_size = int(self.vocab_size * self.gamma)
        greenlist_ids: List[int] = []
        f_x = self._f(input_ids)
        for k in range(self.vocab_size):
            h_k = f_x * int(self.prf[k].item())
            self.rng.manual_seed(h_k % self.vocab_size)
            vocab_permutation = torch.randperm(self.vocab_size, device=self.device, generator=self.rng)
            temp_greenlist_ids = vocab_permutation[:greenlist_size]
            if k in temp_greenlist_ids:
                greenlist_ids.append(int(k))
        return greenlist_ids

    def _compute_z(self, observed: int, total: int) -> float:
        expected = self.gamma
        numer = observed - expected * total
        denom = (total * expected * (1.0 - expected)) ** 0.5
        denom = denom if denom > 1e-12 else 1e-12
        return float(numer / denom)

    def score(self, text: str) -> float:
        ids = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(self.device)
        if len(ids) - self.prefix_length < 1:
            return 0.0
        green_count = 0
        for idx in range(self.prefix_length, len(ids)):
            cur_ids = ids[:idx]
            if self.window_scheme == "left":
                greenlist = self._get_greenlist_ids_left(cur_ids)
            else:
                greenlist = self._get_greenlist_ids_self(cur_ids)
            if int(ids[idx].item()) in greenlist:
                green_count += 1
        return self._compute_z(green_count, len(ids) - self.prefix_length)


class _LiteByteDetector:
    """Lightweight ByteKGWv5 detector that only needs tokenizer and config (no model)."""

    def __init__(
        self,
        *,
        cfg_path: str,
        tokenizer,
        device: str,
        byte_maxpos: Optional[int],
        pos_weights: Optional[np.ndarray],
    ) -> None:
        from MarkLLM.watermark.bytekgwV5.detector import ByteKGWv5Detector, ByteTreeDetectorConfig
        from MarkLLM.watermark.bytekgwV5.prf import BytePRF, PRFConfig
        from MarkLLM.watermark.bytekgwV5.token_bytes import TokenByteVocab

        cfg = read_json(Path(cfg_path))
        gamma = float(cfg.get("gamma", 0.5))
        hash_key = int(cfg.get("hash_key", 15485863))
        prefix_length = int(cfg.get("prefix_length", 4))
        z_threshold = float(cfg.get("z_threshold", 4.0))
        max_byte_pos_cfg = int(cfg.get("max_byte_pos", 64))
        max_byte_pos_use = int(byte_maxpos) if byte_maxpos is not None else max_byte_pos_cfg
        use_prefix_bytes_in_prf = bool(cfg.get("use_prefix_bytes_in_prf", False))
        add_special_tokens = bool(cfg.get("add_special_tokens", True))

        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.add_special_tokens = add_special_tokens

        prf = BytePRF(PRFConfig(hash_key=hash_key, gamma=gamma), device=self.device)
        vocab = TokenByteVocab.from_tokenizer(self.tokenizer, skip_markers=True).to(self.device)

        det_cfg = ByteTreeDetectorConfig(
            prefix_length=prefix_length,
            gamma=gamma,
            z_threshold=z_threshold,
            max_byte_pos=max_byte_pos_use,
            use_prefix_bytes_in_prf=use_prefix_bytes_in_prf,
            pos_weights=None if pos_weights is None else [float(x) for x in pos_weights.tolist()],
        )
        self.detector = ByteKGWv5Detector(prf=prf, vocab=vocab, cfg=det_cfg, device=self.device)

    def score(self, text: str) -> float:
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=self.add_special_tokens,
        )
        input_ids = enc["input_ids"].to(self.device)
        ret = self.detector.detect(input_ids)
        z = ret["z"]
        if isinstance(z, torch.Tensor):
            return float(z[0].item() if z.dim() else z.item())
        return float(z)


def load_weights(path: Optional[str]) -> Optional[np.ndarray]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Weight file not found: {p}")
    wj = read_json(p)
    arr = wj["weights"] if isinstance(wj, dict) and "weights" in wj else wj
    w = np.array(arr, dtype=np.float32)
    return (w / (np.linalg.norm(w) + 1e-12)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True, help="Path to existing detection CSV.")
    ap.add_argument("--algo", type=str, required=True, choices=["kgw", "byte"], help="Which detector to run.")
    ap.add_argument("--model_path", type=str, required=True, help="HF model/tokenizer path.")
    ap.add_argument("--kgw_cfg", type=str, default="config/KGW.json")
    ap.add_argument("--byte_cfg", type=str, default="config/ByteKGWv5.json")
    ap.add_argument("--byte_maxpos", type=int, default=None, help="Override max_byte_pos for byte detector.")
    ap.add_argument("--byte_weights", type=str, default="", help="Optional JSON with pos weights for byte detector.")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--limit", type=int, default=0, help="Optional row limit for quick check.")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    if args.limit and args.limit > 0:
        df = df.head(args.limit).copy()
    text_col = pick_text_col(df)
    z_col = pick_z_col(df)

    tok = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token

    if args.algo == "kgw":
        detector = _LiteKGWDetector(cfg_path=args.kgw_cfg, tokenizer=tok, device=args.device)
    else:
        weights = load_weights(args.byte_weights)
        detector = _LiteByteDetector(
            cfg_path=args.byte_cfg,
            tokenizer=tok,
            device=args.device,
            byte_maxpos=args.byte_maxpos,
            pos_weights=weights,
        )

    texts = df[text_col].astype(str).tolist()
    z_csv = df[z_col].to_numpy(dtype=np.float64)

    zs: List[float] = []
    for t in tqdm(texts, desc="detect"):
        zs.append(detector.score(t))
    z_new = np.array(zs, dtype=np.float64)

    diff = z_new - z_csv
    abs_diff = np.abs(diff)

    print(f"[INFO] rows={len(df)} algo={args.algo} csv={csv_path}")
    print(
        f"[STATS] mean_abs={abs_diff.mean():.6f}  max_abs={abs_diff.max():.6f}  "
        f"corr={np.corrcoef(z_new, z_csv)[0,1]:.6f}"
    )
    for thr in [1e-3, 1e-2, 1e-1]:
        frac = float((abs_diff > thr).mean())
        print(f"[STATS] frac(|diff|>{thr})={frac:.4f}")

    topk = min(5, len(df))
    if topk > 0:
        worst_idx = np.argsort(-abs_diff)[:topk]
        print("\n[TOP DIFFS]")
        for i in worst_idx:
            print(
                f"idx={i} csv={z_csv[i]:.4f} new={z_new[i]:.4f} diff={diff[i]:+.4f} "
                f"text[:80]={texts[i][:80]!r}"
            )


if __name__ == "__main__":
    main()
