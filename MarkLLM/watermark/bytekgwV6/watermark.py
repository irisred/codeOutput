from __future__ import annotations

from typing import Any, Dict, Union

import torch
from transformers import LogitsProcessorList, PreTrainedModel, PreTrainedTokenizerBase

from MarkLLM.watermark.base import BaseWatermark
from MarkLLM.utils.transformers_config import TransformersConfig

from .config import ByteKGWv6Config
from .detector import ByteKGWv6Detector
from .logits_processor import ByteKGWv6LogitsProcessor
from .prf import RobustPartitioner
from .token_bytes import TokenByteVocabV6


class ByteKGWv6(BaseWatermark):
    """
    Minimal wrapper for ByteKGWv6:
      - 生成：HF generate + logits processor (偏置 GREEN tokens by delta)
      - 检测：按前 n 字节聚类 token，指纹判定 GREEN，计算 z
    """

    def __init__(
        self,
        algorithm_config: str,
        transformers_config: TransformersConfig,
        *args,
        **kwargs,
    ) -> None:
        self.config = ByteKGWv6Config(algorithm_config, transformers_config)

        self.model: PreTrainedModel = self.config.generation_model
        self.tokenizer: PreTrainedTokenizerBase = self.config.generation_tokenizer
        self.device = torch.device(self.config.device)

        # vocab (first-n visible bytes)
        self.vocab = TokenByteVocabV6.from_tokenizer(self.tokenizer, skip_markers=True).to(self.device)

        # PRF partitioner
        self.partitioner = RobustPartitioner(
            master_key=self.config.hash_key_bytes,
            m_bits=int(self.config.m_bits),
            target_anchors=int(self.config.target_anchors),
            k_choices=tuple(self.config.k_choices),
            normalize_whitespace=bool(self.config.normalize_whitespace),
        )

        # logits processor for generation
        self.logits_processor = ByteKGWv6LogitsProcessor(
            tokenizer=self.tokenizer,
            vocab=self.vocab,
            partitioner=self.partitioner,
            delta=float(self.config.delta),
            n_bytes=int(self.config.n_bytes),
            seed_window_chars=int(self.config.seed_window_chars),
            device=self.device,
        )

        # detector
        self.detector = ByteKGWv6Detector(
            tokenizer=self.tokenizer,
            vocab=self.vocab,
            partitioner=self.partitioner,
            n_bytes=int(self.config.n_bytes),
            seed_window_chars=int(self.config.seed_window_chars),
            z_threshold=float(self.config.z_threshold),
            min_tokens=int(self.config.min_tokens),
            device=self.device,
            add_special_tokens=bool(self.config.add_special_tokens),
        )

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate_watermarked_text(self, prompt: str, *args, **kwargs) -> str:
        gen_kwargs: Dict[str, Any] = self.config.gen_cfg_dict()
        # allow caller overrides
        gen_kwargs.update(kwargs)

        add_special_tokens = gen_kwargs.pop("add_special_tokens", bool(self.config.add_special_tokens))
        encoded = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=add_special_tokens).to(self.device)

        output_ids = self.model.generate(
            **encoded,
            logits_processor=LogitsProcessorList([self.logits_processor]),
            **gen_kwargs,
        )
        return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

    def generate_unwatermarked_text(self, prompt: str, *args, **kwargs) -> str:
        # use base generation without logits processor
        gen_kwargs: Dict[str, Any] = self.config.gen_cfg_dict()
        gen_kwargs.update(kwargs)
        add_special_tokens = gen_kwargs.pop("add_special_tokens", bool(self.config.add_special_tokens))
        encoded = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=add_special_tokens).to(self.device)
        output_ids = self.model.generate(**encoded, **gen_kwargs)
        return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------
    def detect_watermark(self, text: str, return_dict: bool = True):
        return self.detector.detect(text, return_dict=return_dict)


__all__ = ["ByteKGWv6"]
