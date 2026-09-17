"""
Quick TPR probe for ByteKGWv6 over a small set of prompts.

Workflow:
  1) Use hf_generate.csv (clean, delta=0) as negative set -> compute z distribution.
  2) For each delta in --deltas, generate N prompts with v6 logits processor, detect z.
  3) For each fpr in --fprs, compute threshold from clean z and report TPR.

Assumptions:
  - A v6 config JSON exists (hash_key, n_bytes, seed_window_chars, etc.).
  - Prompts are taken from run_metadata.json["prompts"][i]["prompt_text"].
  - Seed per row reuses hf_generate.csv["seed"] to keep generation deterministic.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from MarkLLM.watermark.bytekgwV6.token_bytes import TokenByteVocabV6  # type: ignore
from MarkLLM.watermark.bytekgwV6.prf import RobustPartitioner  # type: ignore
from MarkLLM.watermark.bytekgwV6.logits_processor import ByteKGWv6LogitsProcessor  # type: ignore
from MarkLLM.watermark.bytekgwV6.detector import ByteKGWv6Detector  # type: ignore
from MarkLLM.watermark.kgw.kgw import KGWLogitsProcessor, KGWUtils  # type: ignore


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_meta", required=True, help="run_metadata.json")
    ap.add_argument("--hf_csv", required=True, help="hf_generate.csv (clean texts)")
    ap.add_argument("--v6_config", required=True, help="ByteKGWv6 config JSON")
    ap.add_argument("--n_gen", type=int, default=10, help="number of prompts to watermarked-generate")
    ap.add_argument(
        "--n_clean",
        type=int,
        default=None,
        help="number of clean rows for threshold (default: all rows in hf_csv)",
    )
    ap.add_argument("--deltas", default="2,3,4,5", help="comma-separated deltas to test")
    ap.add_argument("--fprs", default="0.01,0.05,0.1,0.2", help="comma-separated FPRs for thresholds")
    ap.add_argument("--device", default=None, help="device, default use run_meta.devices[0] or cpu")
    ap.add_argument("--kgw_config", default=None, help="KGW config JSON to compare")
    ap.add_argument("--attack_ratio", type=float, default=0.0, help="char-level replace ratio on continuation (0 = no attack)")
    return ap.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_model(model_path: str, device: str):
    tok = AutoTokenizer.from_pretrained(model_path)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device.startswith("cuda") else None,
    ).to(device)
    mdl.eval()
    return tok, mdl


def build_components(cfg: Dict[str, Any], tokenizer, device: str):
    vocab = TokenByteVocabV6.from_tokenizer(tokenizer, skip_markers=True).to(device)
    partitioner = RobustPartitioner(
        master_key=_to_bytes(cfg.get("hash_key", 15485863)),
        m_bits=int(cfg.get("m_bits", 256)),
        target_anchors=int(cfg.get("target_anchors", 96)),
        k_choices=tuple(cfg.get("k_choices", [4, 5, 6])),
        normalize_whitespace=bool(cfg.get("normalize_whitespace", True)),
    )
    return vocab, partitioner


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


def apply_char_attack(text: str, prompt_text: str, ratio: float, *, seed: int = 0) -> str:
    """
    Simple char-level attack: replace ratio of continuation chars with 'X'.
    """
    if ratio <= 0.0:
        return text
    import random

    random.seed(seed)
    if not text.startswith(prompt_text):
        cont = text
        prefix = ""
    else:
        prefix = prompt_text
        cont = text[len(prompt_text) :]
    chars = list(cont)
    L = len(chars)
    k = max(1, int(L * ratio)) if L > 0 else 0
    idxs = random.sample(range(L), k) if L > 0 else []
    for i in idxs:
        chars[i] = "X"
    attacked = prefix + "".join(chars)
    return attacked


def conservative_threshold(z: np.ndarray, fpr: float) -> float:
    """Simple (1-fpr) quantile."""
    q = 1.0 - float(fpr)
    return float(np.quantile(z, q, interpolation="linear"))


@torch.no_grad()
def perplexity(model, tokenizer, text: str, device: str) -> float:
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)
    input_ids = encoded["input_ids"]
    attn = encoded.get("attention_mask", None)
    labels = input_ids.clone()
    outputs = model(input_ids=input_ids, attention_mask=attn, labels=labels)
    loss = outputs.loss
    return float(torch.exp(loss).item())


def generate_watermarked(
    tokenizer,
    model,
    processor: ByteKGWv6LogitsProcessor,
    prompt: str,
    gen_params: Dict[str, Any],
    seed: int,
    device: str,
) -> str:
    torch.manual_seed(int(seed))
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(int(seed))
    add_special_tokens = gen_params.pop("add_special_tokens", True)
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=add_special_tokens).to(device)
    output_ids = model.generate(
        **encoded,
        logits_processor=LogitsProcessorList([processor]),
        **gen_params,
    )
    return tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]


@torch.no_grad()
def compute_hits_z(
    tokenizer,
    detector: ByteKGWv6Detector,
    text: str,
    prompt_text: str,
) -> Dict[str, float]:
    """
    Compute hits/total/z on continuation tokens (skip prompt part) using detector.detect_ids.
    """
    enc_full = tokenizer(text, return_tensors="pt", add_special_tokens=detector.add_special_tokens).to(detector.device)
    ids_full = enc_full["input_ids"][0]
    enc_prompt = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=detector.add_special_tokens).to(detector.device)
    prompt_len = enc_prompt["input_ids"][0].numel()

    res = detector.detect_ids(ids_full, prompt_len=prompt_len, return_dict=True)
    return {
        "hits": res["hits"],
        "total": res["total"],
        "z": res["z"],
        "green_frac": res["hits"] / res["total"] if res["total"] else 0.0,
    }


def main():
    args = parse_args()
    run_meta = load_config(args.run_meta)
    cfg = load_config(args.v6_config)
    kgw_cfg = load_config(args.kgw_config) if args.kgw_config else None

    model_path = run_meta.get("model")
    device = args.device or (run_meta.get("devices") or ["cpu"])[0]
    deltas = [float(x) for x in args.deltas.split(",") if x.strip()]
    fprs = [float(x) for x in args.fprs.split(",") if x.strip()]

    tokenizer, model = load_model(model_path, device)
    vocab, partitioner = build_components(cfg, tokenizer, device)
    add_special_tokens = bool(cfg.get("add_special_tokens", True))

    detector = ByteKGWv6Detector(
        tokenizer=tokenizer,
        vocab=vocab,
        partitioner=partitioner,
        n_bytes=int(cfg.get("n_bytes", 3)),
        seed_window_chars=int(cfg.get("seed_window_chars", 10)),
        z_threshold=float(cfg.get("z_threshold", 4.0)),
        min_tokens=int(cfg.get("min_tokens", 0)),
        device=device,
        add_special_tokens=add_special_tokens,
    )

    # clean z from hf_generate.csv
    df_clean = pd.read_csv(args.hf_csv)
    z_list_clean: List[float] = []
    df_slice = df_clean if args.n_clean is None else df_clean.head(args.n_clean)
    for _, row in df_slice.iterrows():
        score = compute_hits_z(
            tokenizer,
            detector,
            text=row["full_text"],
            prompt_text=row["prompt_text"],
        )
        z_list_clean.append(score["z"])
    z_clean = np.array(z_list_clean, dtype=float)
    thresholds = {fpr: conservative_threshold(z_clean, fpr) for fpr in fprs}

    # prompts + seeds
    prompts = [p["prompt_text"] for p in run_meta["prompts"]][: args.n_gen]
    seeds = df_clean.head(args.n_gen)["seed"].astype(int).tolist()
    gen_params_base = json.loads(df_clean.iloc[0]["gen_params_json"])
    for k in ("eos_token_id", "pad_token_id"):
        if k in gen_params_base and pd.notna(gen_params_base[k]):
            gen_params_base[k] = int(gen_params_base[k])

    print(f"Using device={device}, model={model_path}")
    print(f"Clean z mean={z_clean.mean():.3f} std={z_clean.std():.3f}")

    for delta in deltas:
        processor = ByteKGWv6LogitsProcessor(
            tokenizer=tokenizer,
            vocab=vocab,
            partitioner=partitioner,
            delta=float(delta),
            n_bytes=int(cfg.get("n_bytes", 3)),
            seed_window_chars=int(cfg.get("seed_window_chars", 10)),
            device=device,
        )
        z_list = []
        ppl_list = []
        green_fracs = []
        z_list_att = []
        green_fracs_att = []
        for prompt, seed in zip(prompts, seeds):
            text = generate_watermarked(
                tokenizer,
                model,
                processor,
                prompt,
                gen_params_base.copy(),
                seed=seed,
                device=device,
            )
            score = compute_hits_z(
                tokenizer,
                detector,
                text=text,
                prompt_text=prompt,
            )
            z_list.append(score["z"])
            ppl_list.append(perplexity(model, tokenizer, text, device))
            green_fracs.append(score["green_frac"])
            # attacked
            attacked_text = apply_char_attack(text, prompt, args.attack_ratio, seed=seed)
            score_att = compute_hits_z(
                tokenizer,
                detector,
                text=attacked_text,
                prompt_text=prompt,
            )
            z_list_att.append(score_att["z"])
            green_fracs_att.append(score_att["green_frac"])
        z_arr = np.array(z_list, dtype=float)
        ppl_arr = np.array(ppl_list, dtype=float)
        z_arr_att = np.array(z_list_att, dtype=float)
        print(f"\nDelta={delta}: z mean={z_arr.mean():.3f} std={z_arr.std():.3f}")
        print(f"  ppl_mean={ppl_arr.mean():.3f} ppl_std={ppl_arr.std():.3f}")
        print("  per-sample hit_rate:", ", ".join(f"{g:.3f}" for g in green_fracs))
        for fpr in fprs:
            thr = thresholds[fpr]
            tpr = float((z_arr >= thr).mean())
            print(f"  fpr={fpr:.2f} thr={thr:.3f} -> tpr={tpr:.3f}")
        # highlight fpr=0.10 if present
        if 0.1 in fprs:
            thr = thresholds[0.1]
            tpr = float((z_arr >= thr).mean())
            print(f"  [summary] fpr=0.10 thr={thr:.3f} tpr={tpr:.3f}")
        # attacked summary
        print("  [attack] per-sample hit_rate:", ", ".join(f"{g:.3f}" for g in green_fracs_att))
        for fpr in fprs:
            thr = thresholds[fpr]
            tpr_att = float((z_arr_att >= thr).mean())
            print(f"  [attack] fpr={fpr:.2f} thr={thr:.3f} -> tpr={tpr_att:.3f}")
        if 0.1 in fprs:
            thr = thresholds[0.1]
            tpr_att = float((z_arr_att >= thr).mean())
            print(f"  [attack summary] fpr=0.10 thr={thr:.3f} tpr={tpr_att:.3f}")

    # KGW comparison (optional)
    if kgw_cfg is not None:
        kgw_config_obj = SimpleKGWConfig(kgw_cfg, vocab_size=len(tokenizer), device=device)
        kgw_utils = KGWUtils(kgw_config_obj)
        kgw_proc = KGWLogitsProcessor(kgw_config_obj, kgw_utils)

        # clean z for KGW
        z_list_clean_kgw: List[float] = []
        for _, row in df_slice.iterrows():
            enc_full = tokenizer(row["full_text"], return_tensors="pt", add_special_tokens=add_special_tokens).to(device)
            ids_full = enc_full["input_ids"][0]
            enc_prompt = tokenizer(row["prompt_text"], return_tensors="pt", add_special_tokens=add_special_tokens).to(device)
            prompt_len = enc_prompt["input_ids"][0].numel()
            cont_ids = ids_full[prompt_len:]
            if cont_ids.numel() <= kgw_config_obj.prefix_length:
                z_list_clean_kgw.append(float("-inf"))
                continue
            z_score, _ = kgw_utils.score_sequence(cont_ids)
            z_list_clean_kgw.append(z_score)
        z_clean_kgw = np.array(z_list_clean_kgw, dtype=float)
        thresholds_kgw = {fpr: conservative_threshold(z_clean_kgw, fpr) for fpr in fprs}
        print(f"\n[KGW] Clean z mean={z_clean_kgw.mean():.3f} std={z_clean_kgw.std():.3f}")

        for delta in deltas:
            kgw_config_obj.delta = float(delta)
            z_list = []
            hit_rates = []
            z_list_att = []
            hit_rates_att = []
            for prompt, seed in zip(prompts, seeds):
                torch.manual_seed(int(seed))
                if device.startswith("cuda"):
                    torch.cuda.manual_seed_all(int(seed))
                encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=add_special_tokens).to(device)
                gen_kwargs = gen_params_base.copy()
                gen_kwargs.pop("add_special_tokens", None)
                output_ids = model.generate(
                    **encoded,
                    logits_processor=LogitsProcessorList([kgw_proc]),
                    **gen_kwargs,
                )[0]
                prompt_len = encoded["input_ids"][0].numel()
                cont_ids = output_ids[prompt_len:]
                if cont_ids.numel() <= kgw_config_obj.prefix_length:
                    z_list.append(float("-inf"))
                    hit_rates.append(0.0)
                    z_list_att.append(float("-inf"))
                    hit_rates_att.append(0.0)
                    continue
                z_score, flags = kgw_utils.score_sequence(cont_ids)
                z_list.append(z_score)
                if len(flags) > kgw_config_obj.prefix_length:
                    hits = sum(1 for f in flags[kgw_config_obj.prefix_length:] if f == 1)
                    total = len(flags) - kgw_config_obj.prefix_length
                    hit_rates.append(hits / total if total > 0 else 0.0)
                else:
                    hit_rates.append(0.0)

                # attack on continuation text
                decoded = tokenizer.decode(output_ids, skip_special_tokens=True)
                attacked_text = apply_char_attack(decoded, prompt, args.attack_ratio, seed=seed)
                enc_full_att = tokenizer(attacked_text, return_tensors="pt", add_special_tokens=add_special_tokens).to(device)
                cont_ids_att = enc_full_att["input_ids"][0][prompt_len:]
                if cont_ids_att.numel() <= kgw_config_obj.prefix_length:
                    z_list_att.append(float("-inf"))
                    hit_rates_att.append(0.0)
                else:
                    z_att, flags_att = kgw_utils.score_sequence(cont_ids_att)
                    z_list_att.append(z_att)
                    if len(flags_att) > kgw_config_obj.prefix_length:
                        hits_att = sum(1 for f in flags_att[kgw_config_obj.prefix_length:] if f == 1)
                        total_att = len(flags_att) - kgw_config_obj.prefix_length
                        hit_rates_att.append(hits_att / total_att if total_att > 0 else 0.0)
                    else:
                        hit_rates_att.append(0.0)

            z_arr = np.array(z_list, dtype=float)
            z_arr_att = np.array(z_list_att, dtype=float)
            print(f"\n[KGW] Delta={delta}: z mean={z_arr.mean():.3f} std={z_arr.std():.3f}")
            print("  per-sample hit_rate:", ", ".join(f"{g:.3f}" for g in hit_rates))
            for fpr in fprs:
                thr = thresholds_kgw[fpr]
                tpr = float((z_arr >= thr).mean())
                print(f"  fpr={fpr:.2f} thr={thr:.3f} -> tpr={tpr:.3f}")
            if 0.1 in fprs:
                thr = thresholds_kgw[0.1]
                tpr = float((z_arr >= thr).mean())
                print(f"  [summary] fpr=0.10 thr={thr:.3f} tpr={tpr:.3f}")
            print("  [attack] per-sample hit_rate:", ", ".join(f"{g:.3f}" for g in hit_rates_att))
            for fpr in fprs:
                thr = thresholds_kgw[fpr]
                tpr_att = float((z_arr_att >= thr).mean())
                print(f"  [attack] fpr={fpr:.2f} thr={thr:.3f} -> tpr={tpr_att:.3f}")
            if 0.1 in fprs:
                thr = thresholds_kgw[0.1]
                tpr_att = float((z_arr_att >= thr).mean())
                print(f"  [attack summary] fpr=0.10 thr={thr:.3f} tpr={tpr_att:.3f}")


if __name__ == "__main__":
    main()
