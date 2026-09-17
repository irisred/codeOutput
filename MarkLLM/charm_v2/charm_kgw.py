from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Union

import torch
from MarkLLM.watermark.base import BaseWatermark, BaseConfig
from MarkLLM.utils.transformers_config import TransformersConfig
from .model import CharmGenerationModel
from .vocab import build_byte_vocab
from .generator import CharmByteGenerator
from .logits_processor import CharmByteLogitsProcessor
from .detector import CharmDetectorV2     # <- 新增这一行，根据你实际路径改


class CharmKGWConfig(BaseConfig):
    """Minimal config wrapper for the new Charm v2 pipeline."""

    def initialize_parameters(self) -> None:
        cfg = self.config_dict
        self.gamma = float(cfg["gamma"])
        self.delta = float(cfg["delta"])
        self.hash_key = int(cfg["hash_key"])
        self.prefix_length = int(cfg["prefix_length"])
        self.charm_cfg = cfg.get("charm_cfg", {})
        gen_overrides = cfg.get("generation_kwargs", {})
        self.gen_kwargs.update(gen_overrides)
        self.first_byte_only_bias = bool(cfg.get("first_byte_only_bias", False))

        # ====== 检测相关参数（新增） ======
        self.z_threshold = float(cfg.get("z_threshold", 3.0))

        self.weight_first = float(cfg.get("first_byte_weight", 1.0))
        self.weight_other = float(cfg.get("other_byte_weight", 0.0))
        
    @property
    def algorithm_name(self) -> str:
        return "CharmKGW"


@dataclass
class CharmRuntime:
    model: CharmGenerationModel
    byte_generator: CharmByteGenerator
    logits_processor: CharmByteLogitsProcessor


class CharmKGW(BaseWatermark):
    """Thin BaseWatermark façade for the refactored Charm pipeline."""

    def __init__(self, algorithm_config: str, transformers_config: TransformersConfig, *args, **kwargs) -> None:
        self.config = CharmKGWConfig(algorithm_config, transformers_config)
        gen_model = CharmGenerationModel(
            transformers_config.model,
            gen_params=self.config.gen_kwargs,
        )
        tok = self.config.generation_tokenizer
        byte_vocab = build_byte_vocab(tok, device=torch.device(self.config.device))
        byte_generator = CharmByteGenerator(gen_model, byte_vocab, tok)
        # 对齐生成端与检测端的滑窗长度：Charm 专用的前缀长度优先使用 charm_cfg.prefix_length
        charm_window_len = self.config.prefix_length
        logits_processor = CharmByteLogitsProcessor(
            hash_key=self.config.hash_key,
            token_prefix_length=charm_window_len,
            gamma=self.config.gamma,
            delta=self.config.delta,
        )
        self.runtime = CharmRuntime(
            model=gen_model,
            byte_generator=byte_generator,
            logits_processor=logits_processor,
        )
        
        # ====== 新增：构建检测器，使其与生成端参数对齐 ======
        self.detector = CharmDetectorV2(
            tokenizer=tok,
            hash_key=self.config.hash_key,
            gamma=self.config.gamma,
            prefix_length=charm_window_len,           # 与生成端一致
            z_threshold=self.config.z_threshold,
            weight_first=self.config.weight_first,
            weight_other=self.config.weight_other,
        )


    def generate_watermarked_text(self, prompt: str, *args, **kwargs) -> str:
        prompt_inputs = self.config.generation_tokenizer(
            prompt,
            return_tensors="pt",
            # PRF 对齐：与检测端保持一致，不再额外插入 special tokens
            add_special_tokens=False,
        ).to(self.config.device)
        gen_kwargs: Dict[str, Any] = dict(self.config.gen_kwargs)
        gen_kwargs.update(kwargs)

        out_ids = self.runtime.byte_generator.generate_token(
            prompt_inputs=prompt_inputs,
            logits_processor=self.runtime.logits_processor,
            gen_kwargs=gen_kwargs,
            charm_cfg=self.config,
        )
        text = self.config.generation_tokenizer.batch_decode(out_ids, skip_special_tokens=True)[0]
        return text

    def generate_unwatermarked_text(self, prompt: str, *args, **kwargs) -> str:
        """Generate unwatermarked text."""
        
        # Encode prompt
        encoded_prompt = self.config.generation_tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(self.config.device)
        gen_kwargs: Dict[str, Any] = dict(self.config.gen_kwargs)
        gen_kwargs.update(kwargs)
        # Generate unwatermarked text
        encoded_unwatermarked_text = self.config.generation_model.generate(**encoded_prompt, **gen_kwargs)
        # Decode
        unwatermarked_text = self.config.generation_tokenizer.batch_decode(encoded_unwatermarked_text, skip_special_tokens=True)[0]
        return unwatermarked_text

    def detect_watermark(
        self,
        text: str,
        return_dict: bool = True,
        *args,
        **kwargs,
    ) -> Union[tuple, dict]:
        """
        BaseWatermark 兼容接口：
        - return_dict=True 时，返回一个 dict（包含 is_watermarked / score / bucket_stats 等）
        - return_dict=False 时，返回 (is_watermarked: bool, score: float)
        """
        verify: bool = bool(kwargs.pop("verify", False))

        result = self.detector.detect(
            text=text,
            return_dict=return_dict,
            verify=verify,
        )
        return result
