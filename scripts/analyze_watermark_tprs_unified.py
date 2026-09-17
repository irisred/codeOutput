#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute TPR at given FPRs for multiple watermark algos (KGW, ByteKGWv6, DiP, Unbiased)
across all matched CSVs, using a shared clean set for thresholding.

Assumptions:
- Each CSV has columns: prompt_text/full_text (hf_generate-like).
- Algo is inferred from filename:
    * bytekgw_v6_deltaX.csv   -> algo=bytekgw_v6, strength="deltaX"
    * kgw_deltaX.csv          -> algo=kgw, strength="deltaX"
    * dip_*.csv               -> algo=dip, strength=filename stem after "dip_"
    * unbiased.csv            -> algo=unbiased
- Clean CSV provided (hf_generate.csv) for threshold calibration.
- If CSV has a 'ppl' column and --ppl_bins is provided (e.g., "0,10,20,50"),
  results are bucketed by ppl bin; otherwise a single bin "all".

Example:
TOKENIZERS_PARALLELISM=false /home/star/jf/python/stega/bin/python scripts/analyze_watermark_tprs_unified.py \
  --run_meta outputs/c4_samples_head_200/run_metadata.json \
  --v6_config config/ByteKGWv6.json \
  --kgw_config config/KGW.json \
  --dip_config config/DIP.json \
  --unbiased_config config/Unbiased.json \
  --clean_csv outputs/v6_vs_kgw_gen/hf_generate.csv \
  --input_glob "outputs/**/*.csv" \
  --fprs 0.01,0.05,0.1,0.2 \
  --device cuda:0 \
  --out_csv outputs/wm_tpr_summary.csv
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from MarkLLM.watermark.bytekgwV6.token_bytes import TokenByteVocabV6  # type: ignore
from MarkLLM.watermark.bytekgwV6.prf import RobustPartitioner  # type: ignore
from MarkLLM.watermark.bytekgwV6.detector import ByteKGWv6Detector  # type: ignore
from MarkLLM.watermark.kgw.kgw import KGWUtils  # type: ignore
from MarkLLM.watermark.dip.dip import DIPConfig, DIPUtils  # type: ignore
from MarkLLM.watermark.unbiased.unbiased import UnbiasedConfig, UnbiasedUtils  # type: ignore
from MarkLLM.utils.transformers_config import TransformersConfig  # type: ignore


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_meta", required=True)
    ap.add_argument("--v6_config", required=True)
    ap.add_argument("--kgw_config", required=True)
    ap.add_argument("--dip_config", required=True)
    ap.add_argument("--unbiased_config", required=True)
    ap.add_argument("--clean_csv", required=True)
    ap.add_argument("--input_glob", required=True, help='e.g., "outputs/**/*.csv"')
    ap.add_argument("--fprs", required=True, help="comma-separated FPRs, e.g., 0.01,0.05,0.1")
    ap.add_argument("--ppl_bins", default=None, help="comma-separated bin edges, e.g., 0,10,20,50; else single bin 'all'")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out_csv", required=True)
    return ap.parse_args()


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def conservative_threshold(z: np.ndarray, fpr: float) -> float:
    if len(z) == 0:
        return float("nan")
    q = 1 - fpr
    return float(np.quantile(z, q, method="linear"))


def build_v6_detector(v6_cfg: Dict[str, Any], tokenizer, device: str) -> ByteKGWv6Detector:
    vocab = TokenByteVocabV6.from_tokenizer(tokenizer, skip_markers=True).to(device)
    partitioner = RobustPartitioner(
        master_key=int(v6_cfg.get("hash_key", 15485863)).to_bytes(16, "little", signed=False),
        m_bits=int(v6_cfg.get("m_bits", 256)),
        target_anchors=int(v6_cfg.get("target_anchors", 96)),
        k_choices=tuple(v6_cfg.get("k_choices", [4, 5, 6])),
        normalize_whitespace=bool(v6_cfg.get("normalize_whitespace", True)),
        k_weight_mode=str(v6_cfg.get("k_weight_mode", "linear")) if "k_weight_mode" in v6_cfg else "linear",
        decision_margin_bits=int(v6_cfg.get("decision_margin_bits", 12)) if "decision_margin_bits" in v6_cfg else 12,
    )
    det = ByteKGWv6Detector(
        tokenizer=tokenizer,
        vocab=vocab,
        partitioner=partitioner,
        n_bytes=int(v6_cfg.get("n_bytes", 3)),
        seed_window_chars=int(v6_cfg.get("seed_window_chars", 18)),
        z_threshold=float(v6_cfg.get("z_threshold", 4.0)),
        device=device,
        add_special_tokens=bool(v6_cfg.get("add_special_tokens", True)),
    )
    return det


class SimpleKGWConfig:
    def __init__(self, cfg: Dict[str, Any], vocab_size: int, device: str) -> None:
        self.gamma = float(cfg.get("gamma", 0.5))
        self.delta = float(cfg.get("delta", 2.0))
        self.hash_key = int(cfg.get("hash_key", 15485863))
        self.z_threshold = float(cfg.get("z_threshold", 4.0))
        self.prefix_length = int(cfg.get("prefix_length", 4))
        self.f_scheme = cfg.get("f_scheme", "additive")
        self.window_scheme = cfg.get("window_scheme", "left")
        self.vocab_size = int(vocab_size)
        self.device = device
        self.gen_kwargs = {}
        self.generation_model = None
        self.generation_tokenizer = None


def build_kgw_utils(kgw_cfg: Dict[str, Any], tokenizer, device: str) -> KGWUtils:
    cfg_obj = SimpleKGWConfig(kgw_cfg, vocab_size=len(tokenizer), device=device)
    return KGWUtils(cfg_obj)


def build_dip_utils(dip_cfg_path: str, tokenizer, device: str) -> DIPUtils:
    tcfg = TransformersConfig(model=None, tokenizer=tokenizer, device=device)
    cfg_obj = DIPConfig(dip_cfg_path, tcfg)
    return DIPUtils(cfg_obj)


def build_unbiased_utils(ub_cfg_path: str, tokenizer, device: str) -> UnbiasedUtils:
    tcfg = TransformersConfig(model=None, tokenizer=tokenizer, device=device)
    cfg_obj = UnbiasedConfig(ub_cfg_path, tcfg)
    return UnbiasedUtils(cfg_obj)


def score_algo(algo: str, texts: List[str], detectors: Dict[str, Any]) -> np.ndarray:
    z_list: List[float] = []
    if algo == "bytekgw_v6":
        det = detectors["v6"]
        for t in texts:
            res = det.detect(t, return_dict=True)
            z_list.append(float(res["z"]))
    elif algo == "kgw":
        kgw_utils = detectors["kgw"]
        tok = detectors["tok"]
        add_special = detectors["kgw_add_special"]
        prefix_len = detectors["kgw_prefix_len"]
        for t in texts:
            ids = tok(t, return_tensors="pt", add_special_tokens=add_special)["input_ids"][0].to(detectors["device"])
            if ids.numel() <= prefix_len:
                z_list.append(float("-inf"))
                continue
            z, _ = kgw_utils.score_sequence(ids)
            z_list.append(float(z))
    elif algo == "dip":
        dip_utils = detectors["dip"]
        tok = detectors["tok"]
        for t in texts:
            ids = tok(t, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(detectors["device"])
            z, _ = dip_utils.score_sequence(ids)
            z_list.append(float(z))
    elif algo == "unbiased":
        ub_utils = detectors["unbiased"]
        tok = detectors["tok"]
        for t in texts:
            ids = tok(t, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(detectors["device"])
            z, _ = ub_utils.score_sequence(ids)
            z_list.append(float(z))
    else:
        raise ValueError(f"Unknown algo: {algo}")
    return np.array(z_list, dtype=float)


def infer_algo_strength(path: Path) -> Tuple[str, str]:
    name = path.name.lower()
    if "bytekgw_v6_delta" in name:
        m = re.search(r"bytekgw_v6_delta([0-9.]+)", name)
        return "bytekgw_v6", f"delta{m.group(1) if m else '?'}"
    if "kgw_delta" in name:
        m = re.search(r"kgw_delta([0-9.]+)", name)
        return "kgw", f"delta{m.group(1) if m else '?'}"
    if name.startswith("dip_") or "dip_" in name:
        stem = path.stem
        m = re.search(r"dip_(.+)", stem)
        return "dip", (m.group(1) if m else stem)
    if "unbiased" in name:
        return "unbiased", "default"
    raise ValueError(f"Cannot infer algo from filename: {path}")


def bin_ppl(series: pd.Series, edges: Optional[List[float]]) -> pd.Series:
    if edges is None or "ppl" not in series.index:
        return pd.Series(["all"] * len(series))
    bins = edges
    labels = []
    for i in range(len(bins) - 1):
        labels.append(f"[{bins[i]},{bins[i+1]})")
    labels.append(f">={bins[-1]}")
    s = series.astype(float)
    cats = pd.cut(s, bins=bins + [math.inf], labels=labels, right=False)
    return cats.astype(str)


def main() -> None:
    args = parse_args()
    device = args.device
    fprs = [float(x) for x in args.fprs.split(",") if x.strip()]
    ppl_edges = [float(x) for x in args.ppl_bins.split(",")] if args.ppl_bins else None

    run_meta = load_json(args.run_meta)
    tok = AutoTokenizer.from_pretrained(run_meta["model"])

    # detectors
    det_v6 = build_v6_detector(load_json(args.v6_config), tok, device)
    kgw_cfg = load_json(args.kgw_config)
    kgw_utils = build_kgw_utils(kgw_cfg, tok, device)
    dip_utils = build_dip_utils(args.dip_config, tok, device)
    ub_utils = build_unbiased_utils(args.unbiased_config, tok, device)

    detectors = {
        "v6": det_v6,
        "kgw": kgw_utils,
        "dip": dip_utils,
        "unbiased": ub_utils,
        "tok": tok,
        "device": device,
        "kgw_add_special": bool(kgw_cfg.get("add_special_tokens", True)),
        "kgw_prefix_len": int(kgw_cfg.get("prefix_length", 4)),
    }

    # clean thresholds per algo
    df_clean = pd.read_csv(args.clean_csv)
    texts_clean = df_clean["full_text"].tolist()
    thresholds: Dict[str, Dict[float, float]] = {}
    for algo in ["bytekgw_v6", "kgw", "dip", "unbiased"]:
        z_clean = score_algo(algo if algo != "bytekgw_v6" else "bytekgw_v6", texts_clean, detectors)
        thresholds[algo] = {fpr: conservative_threshold(z_clean, fpr) for fpr in fprs}

    # gather files
    paths = [Path(p) for p in glob.glob(args.input_glob, recursive=True)]
    rows_out: List[Dict[str, Any]] = []

    for path in tqdm(paths, desc="files"):
        try:
            algo, strength = infer_algo_strength(path)
        except Exception:
            continue
        df = pd.read_csv(path)
        if "full_text" not in df.columns:
            continue
        texts = df["full_text"].tolist()
        z = score_algo("bytekgw_v6" if algo == "bytekgw_v6" else algo, texts, detectors)
        if ppl_edges and "ppl" in df.columns:
            bins = pd.cut(df["ppl"].astype(float), bins=ppl_edges + [math.inf], right=False)
            bin_labels = bins.astype(str)
        else:
            bin_labels = pd.Series(["all"] * len(df))
        for fpr in fprs:
            thr = thresholds[algo][fpr]
            for bin_name in bin_labels.unique():
                mask = bin_labels == bin_name
                if mask.sum() == 0:
                    continue
                tpr = float((z[mask.to_numpy()] >= thr).mean())
                rows_out.append(
                    {
                        "algo": algo,
                        "strength": strength,
                        "file": str(path),
                        "ppl_bin": bin_name,
                        "fpr": fpr,
                        "threshold": thr,
                        "tpr": tpr,
                        "n": int(mask.sum()),
                    }
                )

    pd.DataFrame(rows_out).to_csv(args.out_csv, index=False)
    print(f"wrote summary to {args.out_csv} (rows={len(rows_out)})")


if __name__ == "__main__":
    main()
