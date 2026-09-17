#!/usr/bin/env python3
"""
Debug helper: run Charm generation and print per-step byte/token candidates.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from MarkLLM.charm_v2.charm_kgw import CharmKGW
from MarkLLM.charm_v2.generator import CharmByteGenerator
from MarkLLM.utils.transformers_config import TransformersConfig


def load_prompt(prompt_arg: str | None, prompt_file: str | None, prompt_id: int, trim: int) -> str:
    if prompt_arg:
        return prompt_arg
    if not prompt_file:
        raise SystemExit("Either --prompt or --prompt-file must be provided.")
    lines: List[str] = []
    with Path(prompt_file).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            lines.append(line[:trim] if trim > 0 else line)
    if not lines:
        raise SystemExit(f"No prompts in {prompt_file}")
    idx = min(max(prompt_id, 0), len(lines) - 1)
    return lines[idx]


def patch_first_byte_sampler(top_tokens: int, top_bytes: int) -> None:
    def _debug_sampler(
        self: CharmByteGenerator,
        logits: torch.Tensor,
        *,
        logits_processor=None,
        byte_window: bytes | None = None,
        prefix_length: int = 0,
    ) -> int:
        device = logits.device
        probs = torch.softmax(logits.to(torch.float64), dim=-1)
        eps = 1e-12
        mask = probs > eps
        if mask.sum().item() == 0:
            return int(torch.argmax(logits, dim=-1).item())
        cand_ids = torch.nonzero(mask, as_tuple=False).view(-1).to(torch.long)
        cand_probs = probs.index_select(0, cand_ids)

        tokenizer = getattr(self, "tokenizer", None)
        step = int(getattr(self, "_debug_step", 0))
        print(f"\n===== Sampling step {step} =====")
        print(f"Total candidates with prob>eps: {cand_ids.numel()}")
        window_bytes = bytes(byte_window or b"")
        print(f"Generator PRF window bytes (filtered): {window_bytes!r}")
        print(f"Detector-style PRF window bytes (filtered): {window_bytes!r}")

        first_vals_raw = self.byte_vocab.first_bytes.to(device=device).index_select(0, cand_ids).to(torch.long)
        end_mask = self.byte_vocab.end_token_mask.to(device=device).index_select(0, cand_ids)
        valid_byte_mask = (~end_mask) & (first_vals_raw >= 0)
        mask_empty = (~end_mask) & (first_vals_raw < 0)
        mask_end = end_mask
        mask_byte = valid_byte_mask

        topk = min(top_tokens, cand_probs.numel())
        if topk > 0 and tokenizer is not None:
            top_vals, top_indices = torch.topk(cand_probs, topk)
            tokens = tokenizer.convert_ids_to_tokens(cand_ids[top_indices].tolist(), skip_special_tokens=False)
            for rank in range(topk):
                idx = top_indices[rank].item()
                token_id = int(cand_ids[idx].item())
                prob = float(top_vals[rank].item())
                fb_val = int(first_vals_raw[idx].item())
                fb_str = fb_val if fb_val >= 0 else "EMPTY"
                bucket = "END" if bool(mask_end[idx].item()) else ("BYTE" if fb_val >= 0 else "EMPTY")
                token_str = tokens[rank]
                print(f"[token {rank}] id={token_id} prob={prob:.6f} first_byte={fb_str} bucket={bucket} piece={token_str!r}")

        group_masks = torch.stack([mask_empty, mask_end, mask_byte], dim=0)
        group_names = ("empty", "end", "byte")
        cand_probs_exp = cand_probs.unsqueeze(0)
        group_masses = (group_masks.to(cand_probs_exp.dtype) * cand_probs_exp).sum(dim=1)
        for idx, name in enumerate(group_names):
            print(f"Mass[{name}] = {float(group_masses[idx].item()):.6f}")

        available_mask = group_masses > 0.0
        if not available_mask.any():
            picked = int(torch.argmax(logits, dim=-1).item())
            print("No available group, fallback argmax:", picked)
            return picked
        available_idx = torch.nonzero(available_mask, as_tuple=False).view(-1)
        weights = group_masses.index_select(0, available_idx)
        weights = (weights / weights.sum()).to(torch.float32)
        choice_local = int(torch.multinomial(weights, 1).item())
        choice = int(available_idx[choice_local].item())
        choice_name = group_names[choice]
        print("Chosen group:", choice_name)

        def _sample_from_mask(mask_tensor: torch.Tensor) -> int | None:
            idx = torch.nonzero(mask_tensor, as_tuple=False).view(-1)
            if idx.numel() == 0:
                return None
            sub_probs = cand_probs.index_select(0, idx)
            total = float(sub_probs.sum().item())
            if total <= 0.0:
                sub_probs = torch.ones_like(sub_probs, dtype=torch.float32)
            else:
                sub_probs = (sub_probs / total).to(torch.float32)
            picked_idx = int(torch.multinomial(sub_probs, 1).item())
            real_token = int(cand_ids[idx[picked_idx]].item())
            return real_token

        if choice_name in ("empty", "end"):
            target_mask = mask_empty if choice_name == "empty" else mask_end
            picked_token = _sample_from_mask(target_mask)
            if picked_token is not None:
                print(f"Picked {choice_name} token:", picked_token)
                self._debug_step = step + 1
                return picked_token
            fallback_mask = mask & (~target_mask)
            picked_token = _sample_from_mask(fallback_mask)
            if picked_token is not None:
                print("Fallback token:", picked_token)
                self._debug_step = step + 1
                return picked_token
            picked = int(torch.argmax(logits, dim=-1).item())
            print("Fallback argmax:", picked)
            self._debug_step = step + 1
            return picked

        byte_idx = torch.nonzero(mask_byte, as_tuple=False).view(-1)
        if byte_idx.numel() == 0:
            picked = _sample_from_mask(mask)
            picked = picked if picked is not None else int(torch.argmax(logits, dim=-1).item())
            print("No byte candidates, fallback:", picked)
            self._debug_step = step + 1
            return picked

        byte_vals = first_vals_raw.index_select(0, byte_idx)
        byte_probs = cand_probs.index_select(0, byte_idx)
        byte_mass = torch.zeros(256, dtype=torch.float64, device=device)
        byte_mass.index_add_(0, byte_vals, byte_probs)

        if top_bytes > 0:
            nonzero_idx = torch.nonzero(byte_mass > 0, as_tuple=False).view(-1)
            if nonzero_idx.numel() > 0:
                take = min(top_bytes, nonzero_idx.numel())
                top_vals, top_idx = torch.topk(byte_mass.index_select(0, nonzero_idx), take)
                print("Top byte masses:")
                for rank in range(take):
                    b = int(nonzero_idx[top_idx[rank]].item())
                    mass = float(top_vals[rank].item())
                    print(f"  byte {b}: mass={mass:.6f}")

        base_logits = torch.log(byte_mass.clamp_min(eps)).unsqueeze(0)
        if logits_processor is not None:
            biased = logits_processor(base_logits, bytes(byte_window or b""))
        else:
            biased = base_logits
        byte_dist = torch.softmax(biased.squeeze(0), dim=-1)
        valid_byte_mask_vec = torch.zeros_like(byte_dist)
        valid_byte_mask_vec.scatter_(0, byte_vals.to(byte_dist.device), 1.0)
        masked = byte_dist * valid_byte_mask_vec
        total = float(masked.sum().item())
        if total <= 0.0:
            masked = valid_byte_mask_vec
            total = float(masked.sum().item())
        if total <= 0.0:
            picked = _sample_from_mask(mask_byte)
            picked = picked if picked is not None else int(torch.argmax(logits, dim=-1).item())
            print("Zero byte mass after bias, fallback token:", picked)
            self._debug_step = step + 1
            return picked
        final_dist = (masked / total).to(torch.float32)
        picked_byte = int(torch.multinomial(final_dist, 1).item())
        print("Picked byte:", picked_byte)

        final_mask = mask_byte & (first_vals_raw == picked_byte)
        picked_token = _sample_from_mask(final_mask)
        if picked_token is not None:
            print("Picked token:", picked_token)
            self._debug_step = step + 1
            return picked_token
        picked_token = _sample_from_mask(mask_byte)
        if picked_token is not None:
            print("Fallback byte token:", picked_token)
            self._debug_step = step + 1
            return picked_token
        picked = int(torch.argmax(logits, dim=-1).item())
        print("Argmax fallback:", picked)
        self._debug_step = step + 1
        return picked

    CharmByteGenerator._sample_token_via_first_byte = _debug_sampler


def build_runtime(model_path: str, device: str, max_new_tokens: int, temperature: float, top_p: float):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code="qwen" in model_path.lower())
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    torch_dtype = torch.float16 if device.startswith("cuda") else None
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        trust_remote_code="qwen" in model_path.lower(),
    ).to(device)
    model.eval()
    tf_cfg = TransformersConfig(
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
    )
    return tf_cfg


def main() -> None:
    ap = argparse.ArgumentParser(description="Debug Charm first-byte sampling.")
    ap.add_argument("--config", default="MarkLLM/config/CharmKGW.json")
    ap.add_argument("--model", default="../Meta-Llama-3-8B")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--prompt", default=None, help="Direct prompt text.")
    ap.add_argument("--prompt-file", default="data/prompts_c4.txt")
    ap.add_argument("--prompt-id", type=int, default=0)
    ap.add_argument("--prompt-trim", type=int, default=64)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--topk-tokens", type=int, default=10)
    ap.add_argument("--topk-bytes", type=int, default=10)
    args = ap.parse_args()

    prompt = load_prompt(args.prompt, args.prompt_file, args.prompt_id, args.prompt_trim)
    tf_cfg = build_runtime(args.model, args.device, args.max_new_tokens, args.temperature, args.top_p)
    patch_first_byte_sampler(args.topk_tokens, args.topk_bytes)
    charm = CharmKGW(args.config, tf_cfg)
    charm.runtime.byte_generator._debug_step = 0

    print("=== Prompt ===")
    print(prompt)
    print("====================")
    text = charm.generate_watermarked_text(prompt, max_new_tokens=args.max_new_tokens)
    print("\n=== Generated Text ===")
    print(text)
    det = charm.detector.detect(text, return_dict=True)
    bucket = det.get("bucket_stats", {})
    seg = bucket.get("segmented_start", {}).get("z")
    other = bucket.get("nonsegmented_start", {}).get("z")
    k_stats = det.get("k_stats", {})
    stat121 = k_stats.get(121) or k_stats.get("121")
    z_k121 = ""
    if stat121:
        hits = float(stat121.get("hits", 0.0))
        count = float(stat121.get("count", 0.0))
        gamma = charm.detector.gamma
        import math
        denom = math.sqrt(max(count * gamma * (1.0 - gamma), 1e-12))
        z_k121 = (hits - count * gamma) / denom if count > 0 else ""
    print("\n=== Detection Stats ===")
    print(f"z_first_byte: {seg}")
    print(f"z_other_byte: {other}")
    print(f"z_k121: {z_k121}")


if __name__ == "__main__":
    main()
