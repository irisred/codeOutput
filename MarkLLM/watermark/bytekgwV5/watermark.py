# MarkLLM/watermark/bytekgwV5/watermark.py
from __future__ import annotations

from typing import Union

from transformers import PreTrainedModel, PreTrainedTokenizerBase

from MarkLLM.watermark.base import BaseWatermark
from MarkLLM.utils.transformers_config import TransformersConfig

from .config import ByteKGWv5Config
from .generator import ByteKGWv5 as ByteKGWv5Engine
from .detector import ByteKGWv5Detector, ByteTreeDetectorConfig
from .prf import BytePRF, PRFConfig
from .token_bytes import TokenByteVocab


class ByteKGWv5(BaseWatermark):
    """
    MarkLLM wrapper for ByteKGWv5.

    IMPORTANT (single knob alignment):
      - max_byte_pos controls BOTH generation bias depth and detector scoring depth.
        * max_byte_pos=1   -> first-byte bias + first-byte detection
        * max_byte_pos>1   -> multi-byte bias + multi-byte detection
      - use_prefix_bytes_in_prf MUST match between gen/det.
    """

    def __init__(
        self,
        algorithm_config: str,
        transformers_config: TransformersConfig,
        *args,
        **kwargs,
    ) -> None:
        self.config = ByteKGWv5Config(algorithm_config, transformers_config)

        self.model: PreTrainedModel = self.config.generation_model
        self.tokenizer: PreTrainedTokenizerBase = self.config.generation_tokenizer
        self.device = self.config.device

        # shared vocab
        self.vocab = TokenByteVocab.from_tokenizer(self.tokenizer, skip_markers=True).to(self.device)

        # PRF
        prf_cfg = PRFConfig(hash_key=int(self.config.hash_key), gamma=float(self.config.gamma))
        self.prf = BytePRF(prf_cfg, device=self.device)

        # Generator engine
        self.engine = ByteKGWv5Engine(
            model=self.model,
            tokenizer=self.tokenizer,
            wm_cfg=self.config,
            gen_cfg=self.config.gen_cfg_dict(),
            prf=self.prf,
            vocab=self.vocab,
            device=self.device,
            use_cache=bool(self.config.use_cache),
            scheme=str(self.config.scheme),
            use_torch_generator=bool(self.config.use_torch_generator),
        )

        # Detector (aligned by max_byte_pos)
        det_cfg = ByteTreeDetectorConfig(
            prefix_length=int(self.config.prefix_length),
            gamma=float(self.config.gamma),
            z_threshold=float(self.config.z_threshold),
            max_byte_pos=int(self.config.max_byte_pos),
            use_prefix_bytes_in_prf=bool(self.config.use_prefix_bytes_in_prf),

            # NEW: pass trained weights (may be None -> detector uses default)
            pos_weights=getattr(self.config, "pos_weights", None),
        )
        self.detector = ByteKGWv5Detector(prf=self.prf, vocab=self.vocab, cfg=det_cfg, device=self.device)

    # ---------------------------------------------------------------------
    # Generation
    # ---------------------------------------------------------------------
    def generate_watermarked_text(self, prompt: str, *args, **kwargs) -> str:
        seed = kwargs.pop("seed", None)
        if seed is not None:
            self.engine.set_seed(int(seed))

        add_special_tokens = kwargs.pop("add_special_tokens", self.config.add_special_tokens)
        max_new_tokens = kwargs.pop("max_new_tokens", None)
        skip_special_tokens = kwargs.pop("skip_special_tokens", True)

        return self.engine.generate_text(
            prompt,
            add_special_tokens=bool(add_special_tokens),
            max_new_tokens=max_new_tokens,
            skip_special_tokens=bool(skip_special_tokens),
            *args,
            **kwargs,
        )

    # ---------------------------------------------------------------------
    # Detection
    # ---------------------------------------------------------------------
    def detect_watermark(self, text: str, *args, **kwargs):
        add_special_tokens = kwargs.pop("add_special_tokens", self.config.add_special_tokens)
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=bool(add_special_tokens),
        )
        input_ids = enc["input_ids"].to(self.device)
        return self.detector.detect(input_ids)
