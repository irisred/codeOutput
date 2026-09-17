"""
Generate a short watermarked sequence and trace red/green decisions per step
on both generator (logits processor) and detector paths to verify alignment.

Usage:
  TOKENIZERS_PARALLELISM=false python scripts/trace_v6_step_alignment.py \
    --run_meta outputs/c4_samples_head_200/run_metadata.json \
    --v6_config config/ByteKGWv6.json \
    --prompt_row 0 \
    --max_new_tokens 5 \
    --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

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
    ap.add_argument("--v6_config", required=True)
    ap.add_argument("--prompt_row", type=int, default=0, help="which prompt from run_meta.prompts to use")
    ap.add_argument("--prompt_text", default=None, help="override prompt text")
    ap.add_argument("--max_new_tokens", type=int, default=5)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--do_sample", action="store_true", help="sample using run_meta gen params (top_p/top_k/temperature)")
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


def sample_from_logits(logits: torch.Tensor, *, do_sample: bool, top_p: float, top_k: int, temperature: float) -> int:
    if not do_sample:
        return int(torch.argmax(logits).item())

    logits = logits / max(temperature, 1e-6)
    # top_k
    if top_k > 0 and top_k < logits.numel():
        thresh, _ = torch.topk(logits, top_k)
        cutoff = thresh[-1]
        logits = torch.where(logits >= cutoff, logits, torch.full_like(logits, float("-inf")))
    # top_p
    if top_p < 1.0:
        probs = torch.softmax(logits, dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cum = torch.cumsum(sorted_probs, dim=-1)
        keep = cum <= top_p
        if keep.numel() > 0:
            keep[0] = True  # always keep max
        mask = torch.full_like(probs, False, dtype=torch.bool)
        mask[sorted_idx[keep]] = True
        logits = torch.where(mask, logits, torch.full_like(logits, float("-inf")))

    probs = torch.softmax(logits, dim=-1)
    next_id = torch.multinomial(probs, 1).item()
    return int(next_id)


def main() -> None:
    args = parse_args()
    run_meta = load_json(args.run_meta)
    cfg = load_json(args.v6_config)
    dev = args.device

    model_path = run_meta.get("model")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if dev.startswith("cuda") else None,
    ).to(dev)
    model.eval()

    prompt = args.prompt_text or run_meta["prompts"][args.prompt_row]["prompt_text"]
    gen_cfg = run_meta.get("gen", {}) or {}
    do_sample = bool(args.do_sample or gen_cfg.get("do_sample", False))
    top_p = float(gen_cfg.get("top_p", 1.0))
    top_k = int(gen_cfg.get("top_k", 0))
    temperature = float(gen_cfg.get("temperature", 1.0))

    n_bytes = int(cfg.get("n_bytes", 3))
    seed_window_chars = int(cfg.get("seed_window_chars", 10))
    add_special_tokens = bool(cfg.get("add_special_tokens", True))

    vocab = TokenByteVocabV6.from_tokenizer(tokenizer, skip_markers=True).to(dev)
    firstn_ids_full = vocab.first_n_id(torch.device(dev), n_bytes)
    partitioner = RobustPartitioner(
        master_key=_to_bytes(cfg.get("hash_key", 15485863)),
        m_bits=int(cfg.get("m_bits", 256)),
        target_anchors=int(cfg.get("target_anchors", 96)),
        k_choices=tuple(cfg.get("k_choices", [4, 5, 6])),
        normalize_whitespace=bool(cfg.get("normalize_whitespace", True)),
    )

    processor = ByteKGWv6LogitsProcessor(
        tokenizer=tokenizer,
        vocab=vocab,
        partitioner=partitioner,
        delta=float(cfg.get("delta", 2.0)),
        n_bytes=n_bytes,
        seed_window_chars=seed_window_chars,
        device=dev,
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

    # encode prompt
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=add_special_tokens).to(dev)
    input_ids = encoded["input_ids"][0]  # [L]

    logs: List[Dict[str, Any]] = []

    for step in range(args.max_new_tokens):
        # fingerprint on current prefix
        text_prefix = tokenizer.decode(input_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        fp = partitioner.fingerprint(text_prefix, seed_window_chars)

        # model logits
        outputs = model(input_ids=input_ids.unsqueeze(0))
        scores = outputs.logits[:, -1, :]  # [1,V]

        # bias scores
        biased = processor(input_ids.unsqueeze(0), scores).squeeze(0)

        # sample (greedy or sampling)
        next_id = sample_from_logits(
            biased,
            do_sample=do_sample,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
        )
        token_str = tokenizer.decode([next_id])

        # green decision generator path for this token
        firstn_id = int(firstn_ids_full[next_id].item())
        green_mask_unique = processor._greens_for_fingerprint(fp, device=dev)
        green_tokens = green_mask_unique[processor._inv]
        is_green_gen = bool(green_tokens[next_id].item())

        logs.append(
            {
                "step": step,
                "fp_hex": fp.hex(),
                "token_id": next_id,
                "token_str": token_str,
                "firstn_id": firstn_id,
                "green_gen": is_green_gen,
            }
        )

        # append token to input_ids
        input_ids = torch.cat([input_ids, torch.tensor([next_id], device=dev, dtype=torch.long)], dim=0)

    # Detection-side check on the same prefixes
    for log in logs:
        fid = log["firstn_id"]
        fp_bytes = bytes.fromhex(log["fp_hex"])
        green_det = detector._green_mask_present(fp_bytes, torch.tensor([fid], device=dev, dtype=torch.long))
        log["green_det"] = bool(green_det[0].item())
        log["match"] = log["green_det"] == log["green_gen"]

    # simple hit stats using green_gen path
    hits = sum(1 for log in logs if log["green_gen"])
    total = len(logs)
    hit_rate = hits / total if total > 0 else 0.0

    print(f"Prompt: {prompt!r}")
    print(f"do_sample={do_sample} top_p={top_p} top_k={top_k} temperature={temperature}")
    print(f"Hit rate (green_gen): {hits}/{total} = {hit_rate:.3f}")
    for log in logs:
        print(
            f"step={log['step']} tok={log['token_id']}({log['token_str']!r}) "
            f"firstn_id={log['firstn_id']} green_gen={log['green_gen']} green_det={log['green_det']} match={log['match']}"
        )


if __name__ == "__main__":
    main()
