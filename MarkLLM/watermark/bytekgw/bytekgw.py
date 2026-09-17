# ============================================
# bytekgw.py
# Description: ByteKGW (byte-domain KGW-like watermark) main entry
# ============================================

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, Iterable, Optional, Set

import torch
from transformers import LogitsProcessorList

from ..base import BaseWatermark
from MarkLLM.utils.utils import load_config_file
from MarkLLM.utils.transformers_config import TransformersConfig
from MarkLLM.exceptions.exceptions import AlgorithmNameMismatchError
from MarkLLM.visualize.data_for_visualization import DataForVisualization

from .token_bytes import TokenByteVocab
from .cudaBytePRF import CudaBytePRF, CudaBytePRFConfig
from .ByteWatermarkLogitsProcessor import ByteWatermarkLogitsProcessor
from .detector import CudaByteWatermarkDetector, ByteWMDetectorConfig


def _collect_end_token_ids(tokenizer) -> Set[int]:
    """
    Collect eos/sep/pad token ids (supports scalar or list/tuple/set).
    """
    end_ids: Set[int] = set()
    for attr in ("eos_token_id", "sep_token_id", "pad_token_id"):
        tid = getattr(tokenizer, attr, None)
        if tid is None:
            continue
        if isinstance(tid, (list, tuple, set)):
            for item in tid:
                try:
                    end_ids.add(int(item))
                except Exception:
                    continue
        else:
            try:
                end_ids.add(int(tid))
            except Exception:
                continue
    return end_ids


class ByteKGWConfig:
    """
    Config class for ByteKGW algorithm.
    """

    def __init__(self, algorithm_config: Optional[str], transformers_config: TransformersConfig, *args, **kwargs) -> None:
        if algorithm_config is None:
            config_dict = load_config_file("config/ByteKGW.json")
        else:
            config_dict = load_config_file(algorithm_config)

        algo_name = str(config_dict.get("algorithm_name", ""))
        if algo_name not in ("ByteKGW", "BYTEKGW", "bytekgw"):
            raise AlgorithmNameMismatchError("ByteKGW", algo_name)

        # watermark params
        self.gamma = float(config_dict.get("gamma", 0.5))           # byte green ratio, default 128/256
        self.delta = float(config_dict["delta"])
        self.hash_key = int(config_dict["hash_key"])
        self.z_threshold = float(config_dict.get("z_threshold", 4.0))
        self.prefix_length = int(config_dict["prefix_length"])

        # impl knobs
        self.max_scan_iters = int(config_dict.get("max_scan_iters", 32))
        self.vectorized = bool(config_dict.get("vectorized", False))
        self.ignore_invalid_firstbyte = bool(config_dict.get("ignore_invalid_firstbyte", True))

        # model / runtime
        self.generation_model = transformers_config.model
        self.generation_tokenizer = transformers_config.tokenizer
        self.vocab_size = transformers_config.vocab_size
        self.device = transformers_config.device
        self.gen_kwargs = transformers_config.gen_kwargs

        # sanity: we expect exact 128/256
        green_size = int(round(256 * self.gamma))
        if green_size != 128:
            raise ValueError(f"ByteKGW expects gamma=0.5 (green_size=128). Got gamma={self.gamma} -> {green_size}/256.")


class ByteKGW(BaseWatermark):
    """
    Top-level ByteKGW algorithm.

    - PRF partitions bytes {0..255} into 128 green / 128 red each step
    - LogitsProcessor biases token logits by +delta iff token's firstbyte is green
      (fb=-1 tokens are not biased)
    - Detector recomputes the same green mask and reports z-score
      (fb=-1 tokens optionally ignored from scoring)
    """

    def __init__(self, algorithm_config: Optional[str], transformers_config: TransformersConfig, *args, **kwargs) -> None:
        self.config = ByteKGWConfig(algorithm_config, transformers_config)

        if torch.device(self.config.device).type != "cuda":
            raise ValueError("ByteKGW is CUDA-only: transformers_config.device must be a CUDA device.")

        # 1) Build unified TokenByteVocab (single source of truth for firstbyte rules)
        end_ids = _collect_end_token_ids(self.config.generation_tokenizer)
        self.token_byte_vocab = TokenByteVocab.from_tokenizer(
            self.config.generation_tokenizer,
            end_token_ids=end_ids,
        )

        # 2) PRF
        prf_cfg = CudaBytePRFConfig(
            prefix_length=self.config.prefix_length,
            hash_key=self.config.hash_key,
            green_size=128,
        )
        self.prf = CudaBytePRF(prf_cfg, device=self.config.device)

        # 3) Logits processor
        self.logits_processor = ByteWatermarkLogitsProcessor(
            prf=self.prf,
            token_byte_vocab=self.token_byte_vocab,
            delta=self.config.delta,
            prefix_length=self.config.prefix_length,
            device=self.config.device,
            vectorized=self.config.vectorized,
            max_scan_iters=self.config.max_scan_iters,
        )

        # 4) Detector
        det_cfg = ByteWMDetectorConfig(
            prefix_length=self.config.prefix_length,
            gamma=self.config.gamma,
            z_threshold=self.config.z_threshold,
            ignore_invalid_firstbyte=self.config.ignore_invalid_firstbyte,
            return_per_token=False,
            max_scan_iters=self.config.max_scan_iters,
        )
        self.detector = CudaByteWatermarkDetector(
            prf=self.prf,
            token_byte_vocab=self.token_byte_vocab,
            config=det_cfg,
            device=self.config.device,
        )

    def generate_watermarked_text(self, prompt: str, *args, **kwargs) -> str:
        generate_with_watermark = partial(
            self.config.generation_model.generate,
            logits_processor=LogitsProcessorList([self.logits_processor]),
            **self.config.gen_kwargs,
        )

        encoded_prompt = self.config.generation_tokenizer(
            prompt, return_tensors="pt", add_special_tokens=True
        ).to(self.config.device)

        encoded_out = generate_with_watermark(**encoded_prompt)
        return self.config.generation_tokenizer.batch_decode(encoded_out, skip_special_tokens=True)[0]

    @torch.no_grad()
    def detect_watermark(self, text: str, return_dict: bool = True, *args, **kwargs):
        ids = self.config.generation_tokenizer(
            text, return_tensors="pt", add_special_tokens=False
        )["input_ids"][0].to(self.config.device)

        out = self.detector.detect(ids)
        z = out["z"]
        # z can be a 0-d tensor
        z_float = float(z.item()) if torch.is_tensor(z) else float(z)

        if return_dict:
            return {
                "is_watermarked": bool(out["is_watermarked"].item() if torch.is_tensor(out["is_watermarked"]) else out["is_watermarked"]),
                "score": z_float,
                "z": z_float,
                "num_scored": int(out["num_scored"].item() if torch.is_tensor(out["num_scored"]) else out["num_scored"]),
                "num_green": int(out["num_green"].item() if torch.is_tensor(out["num_green"]) else out["num_green"]),
            }
        return (bool(out["is_watermarked"]), z_float)

    @torch.no_grad()
    def get_data_for_visualization(self, text: str, *args, **kwargs) -> DataForVisualization:
        ids = self.config.generation_tokenizer(
            text, return_tensors="pt", add_special_tokens=False
        )["input_ids"][0].to(self.config.device)

        # Make a temporary detector config that returns per-token hits
        viz_cfg = ByteWMDetectorConfig(
            prefix_length=self.config.prefix_length,
            gamma=self.config.gamma,
            z_threshold=self.config.z_threshold,
            ignore_invalid_firstbyte=self.config.ignore_invalid_firstbyte,
            return_per_token=True,
            max_scan_iters=self.config.max_scan_iters,
        )
        viz_detector = CudaByteWatermarkDetector(
            prf=self.prf,
            token_byte_vocab=self.token_byte_vocab,
            config=viz_cfg,
            device=self.config.device,
        )
        s = viz_detector.score(ids)

        L = int(self.config.prefix_length)
        T = int(ids.numel())
        highlight: list[int] = [-1] * min(L, T)

        if T > L and "per_token_hits" in s:
            hits = s["per_token_hits"]  # [T-L] (since ids is 1D)
            if hits.dim() != 1:
                hits = hits.view(-1)
            highlight.extend(hits.to(torch.int32).detach().cpu().tolist())

        decoded_tokens = [self.config.generation_tokenizer.decode(int(t.item())) for t in ids]
        return DataForVisualization(decoded_tokens, highlight)


__all__ = ["ByteKGW", "ByteKGWConfig"]
