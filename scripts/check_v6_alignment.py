"""
Quick alignment check for ByteKGWv6:

Given a text (default: first row of hf_generate.csv), verify that
  - detector green mask == logits processor green mask
for the set of first-n byte IDs present in the text.

Usage:
  TOKENIZERS_PARALLELISM=false python scripts/check_v6_alignment.py \
    --run_meta outputs/c4_samples_head_200/run_metadata.json \
    --hf_csv outputs/c4_samples_head_200/hf_generate.csv \
    --v6_config config/ByteKGWv6.json \
    --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from MarkLLM.watermark.bytekgwV6.token_bytes import TokenByteVocabV6  # type: ignore
from MarkLLM.watermark.bytekgwV6.prf import RobustPartitioner  # type: ignore
from MarkLLM.watermark.bytekgwV6.logits_processor import ByteKGWv6LogitsProcessor  # type: ignore
from MarkLLM.watermark.bytekgwV6.detector import ByteKGWv6Detector  # type: ignore


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_meta", required=True)
    ap.add_argument("--hf_csv", required=True)
    ap.add_argument("--v6_config", required=True)
    ap.add_argument("--row", type=int, default=0, help="which row from hf_csv to use")
    ap.add_argument("--device", default="cpu")
    return ap.parse_args()


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _to_bytes(key) -> bytes:
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


def main() -> None:
    args = parse_args()
    run_meta = load_json(args.run_meta)
    cfg = load_json(args.v6_config)
    dev = args.device

    model_path = run_meta.get("model")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    n_bytes = int(cfg.get("n_bytes", 3))
    seed_window_chars = int(cfg.get("seed_window_chars", 10))
    add_special_tokens = bool(cfg.get("add_special_tokens", True))

    vocab = TokenByteVocabV6.from_tokenizer(tokenizer, skip_markers=True).to(dev)
    partitioner = RobustPartitioner(
        master_key=_to_bytes(cfg.get("hash_key", 15485863)),
        m_bits=int(cfg.get("m_bits", 256)),
        target_anchors=int(cfg.get("target_anchors", 96)),
        k_choices=tuple(cfg.get("k_choices", [4, 5, 6])),
        normalize_whitespace=bool(cfg.get("normalize_whitespace", True)),
    )

    detector = ByteKGWv6Detector(
        tokenizer=tokenizer,
        vocab=vocab,
        partitioner=partitioner,
        n_bytes=n_bytes,
        seed_window_chars=seed_window_chars,
        z_threshold=float(cfg.get("z_threshold", 4.0)),
        min_tokens=int(cfg.get("min_tokens", 0)),
        device=dev,
        add_special_tokens=add_special_tokens,
    )

    processor = ByteKGWv6LogitsProcessor(
        tokenizer=tokenizer,
        vocab=vocab,
        partitioner=partitioner,
        delta=0.0,  # no bias needed for mask
        n_bytes=n_bytes,
        seed_window_chars=seed_window_chars,
        device=dev,
    )

    df = pd.read_csv(args.hf_csv)
    row = df.iloc[args.row]
    text = row["full_text"]

    enc = tokenizer(text, return_tensors="pt", add_special_tokens=add_special_tokens).to(dev)
    ids = enc["input_ids"][0]
    firstn_seq = vocab.first_n_id(dev, n_bytes).index_select(0, ids)
    uniq_present, inv_present = torch.unique(firstn_seq, return_inverse=True)

    fp = partitioner.fingerprint(text, seed_window_chars)
    green_det = detector._green_mask_present(fp, uniq_present)  # [M]

    # Recompute processor mask over present IDs using its internal tables
    # Build id->idx mapping for processor unique IDs
    id_to_idx = {int(uid): i for i, uid in enumerate(processor._uniq_ids.tolist())}
    idxs = torch.tensor([id_to_idx[int(x)] for x in uniq_present.tolist()], device=dev, dtype=torch.long)
    green_proc = processor._greens_for_fingerprint(fp, device=dev).index_select(0, idxs)

    mismatches = (green_det != green_proc).sum().item()
    print(f"add_special_tokens={add_special_tokens}, n_bytes={n_bytes}, seed_window_chars={seed_window_chars}")
    print(f"uniq_present={len(uniq_present)}, mismatches={mismatches}")
    if mismatches:
        diff_idx = torch.nonzero(green_det != green_proc).view(-1)[:10]
        print("First few mismatched IDs:", uniq_present[diff_idx].tolist())
    else:
        print("Detector and processor green masks match on present IDs.")


if __name__ == "__main__":
    main()
