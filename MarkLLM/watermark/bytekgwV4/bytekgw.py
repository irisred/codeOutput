# MarkLLM/watermark/bytekgwV4/bytekgw.py
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union, Iterable

import torch
from transformers import PreTrainedTokenizerBase, PreTrainedModel

from .aligned_generation import AlignedGenerationModel
from .cudaBytePRF import BytePRFConfig, CudaBytePRF
from .token_bytes import TokenByteVocab
from .FastFirstByteIndex import FastFirstByteIndex
from .ByteWatermarkLogitsProcessor import ByteWatermarkLogitsProcessor
from .detector import ByteWMDetectorConfig, CudaByteWatermarkDetector


# -----------------------------
# Config
# -----------------------------
@dataclass
class ByteKGWConfig:
    algorithm_name: str = "ByteKGWv4"
    gamma: float = 0.5
    delta: float = 1.0
    hash_key: int = 15485863
    prefix_length: int = 4
    z_threshold: float = 4.0
    ignore_invalid_firstbyte: bool = True
    max_scan_iters: int = 32
    device: str = "cuda:0"

    @classmethod
    def from_json(cls, path_or_obj: Union[str, Dict[str, Any]]) -> "ByteKGWConfig":
        if isinstance(path_or_obj, str):
            with open(path_or_obj, "r", encoding="utf-8") as f:
                obj = json.load(f)
        else:
            obj = dict(path_or_obj)

        cfg = cls()
        for k, v in obj.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg


def _clean_gen_kwargs(d: Dict[str, Any]) -> Dict[str, Any]:
    """Remove None values to avoid accidentally overriding model defaults with None."""
    return {k: v for k, v in d.items() if v is not None}


def _extract_gen_kwargs(tf_cfg, tokenizer: PreTrainedTokenizerBase) -> Dict[str, Any]:
    """
    IMPORTANT: align with KGW behavior:
      - do NOT guess params from tf_cfg fields (they may not exist / mismatch).
      - use tf_cfg.gen_kwargs as the single source of truth, just like KGW does.
    """
    if hasattr(tf_cfg, "gen_kwargs") and isinstance(tf_cfg.gen_kwargs, dict):
        gen_kwargs = dict(tf_cfg.gen_kwargs)  # copy
    else:
        # fallback (best-effort only)
        gen_kwargs = {}
        for k in ["do_sample", "temperature", "top_p", "top_k", "repetition_penalty", "typical_p", "max_new_tokens", "max_length"]:
            if hasattr(tf_cfg, k):
                gen_kwargs[k] = getattr(tf_cfg, k)

    gen_kwargs = _clean_gen_kwargs(gen_kwargs)

    # Ensure pad_token_id is explicit to avoid HF warning & behavior drift
    eos_id = getattr(tokenizer, "eos_token_id", None)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if "pad_token_id" not in gen_kwargs:
        if pad_id is not None:
            gen_kwargs["pad_token_id"] = int(pad_id)
        elif eos_id is not None:
            gen_kwargs["pad_token_id"] = int(eos_id)

    # Ensure eos_token_id is explicit if tokenizer provides it
    if "eos_token_id" not in gen_kwargs and eos_id is not None:
        gen_kwargs["eos_token_id"] = int(eos_id)

    return gen_kwargs


def _as_eos_set(eos_token_id: Any) -> set[int]:
    if eos_token_id is None:
        return set()
    if isinstance(eos_token_id, (list, tuple, set)):
        out = set()
        for x in eos_token_id:
            try:
                out.add(int(x))
            except Exception:
                pass
        return out
    try:
        return {int(eos_token_id)}
    except Exception:
        return set()


def _resolve_max_steps(gen_kwargs: Dict[str, Any], prompt_len: int) -> int:
    """
    Prefer:
      - max_new_tokens
      - else max_length - prompt_len
      - else fallback 128
    """
    if "max_new_tokens" in gen_kwargs and gen_kwargs["max_new_tokens"] is not None:
        return int(gen_kwargs["max_new_tokens"])
    if "max_length" in gen_kwargs and gen_kwargs["max_length"] is not None:
        mx = int(gen_kwargs["max_length"])
        return max(0, mx - int(prompt_len))
    return 128


# -----------------------------
# Main class
# -----------------------------
class ByteKGW:
    """
    ByteKGWv4

    Alignment goal:
      - All generation knobs (do_sample/top_p/top_k/temperature/repetition_penalty/...) are taken
        from tf_cfg.gen_kwargs (single source of truth, like KGW).
      - AlignedGenerationModel is constructed using these exact gen_kwargs.
      - Watermark bias is applied in logits space; sampling happens once per step.

    Public API (consistent with KGW style):
      - generate_unwatermarked_text(prompt, max_new_tokens=...)
      - generate_watermarked_text(prompt, max_new_tokens=...)
      - detect_watermark(text, return_dict=False)
    """

    def __init__(self, config_path_or_dict: Union[str, Dict[str, Any], ByteKGWConfig], tf_cfg) -> None:
        self.config = config_path_or_dict if isinstance(config_path_or_dict, ByteKGWConfig) else ByteKGWConfig.from_json(config_path_or_dict)

        self.tf_cfg = tf_cfg
        self.model: PreTrainedModel = tf_cfg.model
        self.tokenizer: PreTrainedTokenizerBase = tf_cfg.tokenizer

        self.device = torch.device(getattr(tf_cfg, "device", self.config.device))
        if self.device.type != "cuda":
            raise ValueError("ByteKGWv4 (this implementation) is CUDA-only.")

        # Make tokenizer pad token explicit (optional; still also set pad_token_id in gen_kwargs)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 1) Extract gen kwargs EXACTLY like KGW (single source of truth)
        self.gen_kwargs: Dict[str, Any] = _extract_gen_kwargs(tf_cfg, self.tokenizer)

        # 2) Build aligned facade using the same gen_kwargs
        self.aligned = AlignedGenerationModel(self.model, gen_params=dict(self.gen_kwargs))

        # 3) Build TokenByteVocab / firstbyte table (shared by watermark + detector)
        self.token_byte_vocab = TokenByteVocab.from_tokenizer(self.tokenizer)
        self.byte_index = FastFirstByteIndex(
            self.token_byte_vocab,
            self.device,
            max_scan_iters=int(self.config.max_scan_iters),
        )

        # 4) PRF
        prf_cfg = BytePRFConfig(hash_key=int(self.config.hash_key), gamma=float(self.config.gamma))
        self.prf = CudaBytePRF(prf_cfg, device=self.device)

        # 5) Watermark biaser
        self.biaser = ByteWatermarkLogitsProcessor(
            prf=self.prf,
            token_byte_vocab=self.token_byte_vocab,
            delta=float(self.config.delta),
            prefix_length=int(self.config.prefix_length),
            device=self.device,
            max_scan_iters=int(self.config.max_scan_iters),
        )

        # 6) Detector
        det_cfg = ByteWMDetectorConfig(
            prefix_length=int(self.config.prefix_length),
            gamma=float(self.config.gamma),
            z_threshold=float(self.config.z_threshold),
            ignore_invalid_firstbyte=bool(self.config.ignore_invalid_firstbyte),
        )
        self.detector = CudaByteWatermarkDetector(
            prf=self.prf,
            firstbyte_vocab_tensor=self.byte_index.firstbyte_vocab_tensor,
            config=det_cfg,
            device=self.device,
        )

    def _encode(self, prompt: str) -> Dict[str, torch.Tensor]:
        enc = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=True,
        )
        return {k: v.to(self.device) for k, v in enc.items()}

    @torch.no_grad()
    def generate_unwatermarked_text(self, prompt: str, *, max_new_tokens: Optional[int] = None) -> str:
        saved = float(self.biaser.delta)
        self.biaser.delta = 0.0
        try:
            return self.generate_watermarked_text(prompt, max_new_tokens=max_new_tokens)
        finally:
            self.biaser.delta = saved

    @torch.no_grad()
    def generate_watermarked_text(self, prompt: str, *, max_new_tokens: Optional[int] = None) -> str:
        enc = self._encode(prompt)
        input_ids = enc["input_ids"]  # [1,T]
        attn = enc.get("attention_mask", torch.ones_like(input_ids, device=self.device))

        # Use EXACT gen_kwargs; only override max_new_tokens if user passes it
        gen_kwargs = dict(self.gen_kwargs)
        if max_new_tokens is not None:
            gen_kwargs["max_new_tokens"] = int(max_new_tokens)

        self.aligned.reset()

        do_sample = bool(gen_kwargs.get("do_sample", getattr(self.aligned.generation_config, "do_sample", True)))
        eos_set = _as_eos_set(gen_kwargs.get("eos_token_id", getattr(self.tokenizer, "eos_token_id", None)))

        max_steps = _resolve_max_steps(gen_kwargs, prompt_len=int(input_ids.size(1)))
        max_steps = max(0, int(max_steps))

        for _ in range(max_steps):
            # (A) HF logits_processor stage (pre-warp)
            scores = self.aligned.compute_logits_pre_warp(input_ids)  # [V]

            # (B) Apply byte watermark bias BEFORE warpers (delta=0 => identity)
            scores2d = self.biaser(input_ids, scores.unsqueeze(0)).squeeze(0)  # [V]

            # (C) HF warpers stage (temperature/top_p/top_k/...) if do_sample
            # NOTE: aligned_generation uses its internal generation_config built from gen_kwargs
            scores_final = self.aligned.apply_warpers(input_ids, scores2d)  # [V]

            # (D) Sample/greedy exactly once per step (RNG consumption matches HF-style)
            if do_sample:
                probs = torch.softmax(scores_final.to(torch.float32), dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)  # [1]
            else:
                next_id = torch.argmax(scores_final, dim=-1, keepdim=False).view(1)

            next_id_2d = next_id.view(1, 1)
            input_ids = torch.cat([input_ids, next_id_2d], dim=1)
            attn = torch.cat([attn, torch.ones_like(next_id_2d, device=self.device)], dim=1)

            if eos_set and int(next_id.item()) in eos_set:
                break

        return self.tokenizer.decode(input_ids[0], skip_special_tokens=True)

    @torch.no_grad()
    def detect_watermark(self, text: str, *, return_dict: bool = False):
        enc = self.tokenizer(text, return_tensors="pt", add_special_tokens=True)
        ids = enc["input_ids"].to(self.device)
        out = self.detector.detect(ids)
        if return_dict:
            return out
        # default: return bool per batch or scalar bool
        is_wm = out["is_watermarked"]
        if torch.is_tensor(is_wm):
            if is_wm.numel() == 1:
                return bool(is_wm.item())
            return is_wm.detach().cpu().tolist()
        return bool(is_wm)
