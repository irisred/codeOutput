#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute TPR per PPL bucket before/after char-attack for multiple algos (ByteKGWv6, KGW, DiP, Unbiased).
Thresholds are calibrated on the clean CSV (no attack). Buckets are taken from ppl_*.csv under bucket_root/<algo>/.

Example:
TOKENIZERS_PARALLELISM=false /home/star/jf/python/stega/bin/python scripts/analyze_attack_tpr_by_bucket.py \
  --run_meta outputs/c4_samples_head_200/run_metadata.json \
  --v6_config config/ByteKGWv6.json \
  --kgw_config config/KGW.json \
  --dip_config config/DIP.json \
  --unbiased_config config/Unbiased.json \
  --clean_csv outputs/wm_eval/clean/hf_generate.csv \
  --bucket_root outputs/wm_eval \
  --fprs 0.01,0.05,0.1,0.2 \
  --attack_ratio 0.02 --attack_seed 0 \
  --device cuda:0 \
  --out_csv outputs/wm_eval/tpr_attack_by_bucket.csv
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer
from tqdm import tqdm

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
from random_attack import RandomAttack  # type: ignore


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_meta", required=True)
    ap.add_argument("--v6_config", required=True)
    ap.add_argument("--kgw_config", required=True)
    ap.add_argument("--dip_config", required=True)
    ap.add_argument("--unbiased_config", required=True)
    ap.add_argument("--clean_csv", required=True)
    ap.add_argument("--bucket_root", required=True, help="root dir containing algo subdirs with ppl_*.csv")
    ap.add_argument("--fprs", default="0.01,0.05,0.1,0.2")
    ap.add_argument("--attack_ratio", type=float, default=0.0)
    ap.add_argument("--attack_seed", type=int, default=0)
    ap.add_argument("--atk_styles", default="char", help="comma-separated attack styles from random_attack.py (e.g., char,sand_char,mix_char)")
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


def apply_char_attack(text: str, prompt: str, ratio: float, seed: int) -> str:
    if ratio <= 0.0:
        return text
    rng = random.Random(seed)
    start = len(prompt) if text.startswith(prompt) else 0
    cont = text[start:]
    n = len(cont)
    if n == 0:
        return text
    k = max(1, int(round(n * ratio)))
    k = min(k, n)
    idxs = rng.sample(range(n), k)
    cont_list = list(cont)
    for i in idxs:
        cont_list[i] = "X"
    return text[:start] + "".join(cont_list)


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
        device = detectors["device"]
        for t in texts:
            ids = tok(t, return_tensors="pt", add_special_tokens=add_special)["input_ids"][0].to(device)
            if ids.numel() <= prefix_len:
                z_list.append(float("-inf"))
                continue
            z, _ = kgw_utils.score_sequence(ids)
            z_list.append(float(z))
    elif algo == "dip":
        dip_utils = detectors["dip"]
        tok = detectors["tok"]
        device = detectors["device"]
        for t in texts:
            ids = tok(t, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(device)
            z, _ = dip_utils.score_sequence(ids)
            z_list.append(float(z))
    elif algo == "unbiased":
        ub_utils = detectors["unbiased"]
        tok = detectors["tok"]
        device = detectors["device"]
        for t in texts:
            ids = tok(t, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(device)
            z, _ = ub_utils.score_sequence(ids)
            z_list.append(float(z))
    else:
        raise ValueError(f"Unknown algo: {algo}")
    return np.array(z_list, dtype=float)


def main() -> None:
    args = parse_args()
    device = args.device
    fprs = [float(x) for x in args.fprs.split(",") if x.strip()]

    run_meta = load_json(args.run_meta)
    tok = AutoTokenizer.from_pretrained(run_meta["model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # detectors
    v6_cfg = load_json(args.v6_config)
    kgw_cfg = load_json(args.kgw_config)
    det_v6 = build_v6_detector(v6_cfg, tok, device)
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
    atk_styles = [s.strip() for s in args.atk_styles.split(",") if s.strip()]
    attacker = RandomAttack(tokenizer=tok, device=device)

    # thresholds from clean (no attack)
    df_clean = pd.read_csv(args.clean_csv)
    texts_clean = df_clean["full_text"].tolist()
    thresholds: Dict[str, Dict[float, float]] = {}
    for algo in ["bytekgw_v6", "kgw", "dip", "unbiased"]:
        z_clean = score_algo(algo if algo != "bytekgw_v6" else "bytekgw_v6", texts_clean, detectors)
        thresholds[algo] = {fpr: conservative_threshold(z_clean, fpr) for fpr in fprs}

    rows_out: List[Dict[str, Any]] = []
    algo_dirs = {
        "bytekgw_v6": Path(args.bucket_root) / "bytekgw_v6",
        "kgw": Path(args.bucket_root) / "kgw",
        "dip": Path(args.bucket_root) / "dip",
        "unbiased": Path(args.bucket_root) / "unbiased",
    }

    for algo, dir_path in algo_dirs.items():
        files = sorted(dir_path.glob("ppl_*.csv"))
        for p in tqdm(files, desc=f"{algo} buckets"):
            m = re.match(r"ppl_(.+)\\.csv", p.name)
            bucket = m.group(1) if m else p.stem
            df = pd.read_csv(p)
            if "full_text" not in df.columns:
                continue
            texts_clean_bucket = df["full_text"].tolist()
            # attack
            if "prompt_text" in df.columns:
                prompts = df["prompt_text"].fillna("").tolist()
            else:
                prompts = [""] * len(df)
            z_clean_bucket = score_algo(algo, texts_clean_bucket, detectors)
            for atk_style in atk_styles:
                texts_att = []
                for i, (t, prm) in enumerate(zip(texts_clean_bucket, prompts)):
                    try:
                        adv = attacker.get_adv(
                            sentence=t,
                            atk_style=atk_style,
                            max_edit_rate=args.attack_ratio,
                            atk_times=1,
                        )
                        texts_att.append(adv["sentence"])
                    except Exception:
                        raise RuntimeError()
                        texts_att.append(apply_char_attack(t, prm, args.attack_ratio, args.attack_seed + i))
                z_att_bucket = score_algo(algo, texts_att, detectors)
                for fpr in fprs:
                    thr = thresholds[algo][fpr]
                    tpr_clean = float((z_clean_bucket >= thr).mean()) if len(z_clean_bucket) else float("nan")
                    tpr_att = float((z_att_bucket >= thr).mean()) if len(z_att_bucket) else float("nan")
                    rows_out.append(
                        {
                            "algo": algo,
                            "bucket": bucket,
                            "atk_style": atk_style,
                            "fpr": fpr,
                            "threshold": thr,
                            "tpr_clean": tpr_clean,
                            "tpr_attack": tpr_att,
                            "n": len(z_clean_bucket),
                            "file": str(p),
                            "attack_ratio": args.attack_ratio,
                            "attack_seed": args.attack_seed,
                        }
                    )

    pd.DataFrame(rows_out).to_csv(args.out_csv, index=False)
    print(f"wrote summary to {args.out_csv} (rows={len(rows_out)})")


if __name__ == "__main__":
    main()
