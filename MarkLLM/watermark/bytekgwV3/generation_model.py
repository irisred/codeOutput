# MarkLLM/watermark/bytekgwV2/generation_model.py
from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

try:
    from transformers.generation.logits_process import (
        LogitsProcessorList,
        TemperatureLogitsWarper,
        TopPLogitsWarper,
        TopKLogitsWarper,
        TypicalLogitsWarper,
    )
except Exception:
    # older transformers
    from transformers.generation.logits_process import LogitsProcessorList  # type: ignore
    TemperatureLogitsWarper = None
    TopPLogitsWarper = None
    TopKLogitsWarper = None
    TypicalLogitsWarper = None


class HFAlignedGenerationModel:
    """
    Step-by-step generation that reuses HF's logits_processor (and warpers if available).
    Works for decoder-only causal LM.

    - Uses past_key_values cache (use_cache=True).
    - Builds processors/warpers from model + generation_config.
    """

    def __init__(self, model, tokenizer, gen_kwargs: Dict[str, Any], device: torch.device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        # build generation_config aligned with HF defaults
        if hasattr(self.model, "_prepare_generation_config"):
            self.generation_config, self._model_kwargs_template = self.model._prepare_generation_config(
                None,
                use_model_defaults=True,
                **(gen_kwargs or {}),
            )
        else:
            self.generation_config = getattr(self.model, "generation_config", None)
            self._model_kwargs_template = {}

        # ensure pad_token_id to avoid warnings / different behaviors
        pad = tokenizer.pad_token_id
        eos = tokenizer.eos_token_id
        if pad is None and eos is not None:
            pad = eos
        if getattr(self.generation_config, "pad_token_id", None) is None and pad is not None:
            self.generation_config.pad_token_id = int(pad)
        if getattr(self.model.config, "pad_token_id", None) is None and pad is not None:
            self.model.config.pad_token_id = int(pad)

        self.reset()

    def reset(self):
        self.past_key_values = None

    def _build_logits_processor(self, input_ids: torch.Tensor) -> "LogitsProcessorList":
        # transformers signature drift handler
        if not hasattr(self.model, "_get_logits_processor"):
            return LogitsProcessorList()

        seq_len = int(input_ids.shape[-1])
        kwargs_common = dict(
            generation_config=self.generation_config,
            input_ids_seq_length=seq_len,
            encoder_input_ids=None,
            prefix_allowed_tokens_fn=None,
            logits_processor=None,
            device=self.device,
        )

        # newer signature includes model_kwargs and negative prompts
        try:
            return self.model._get_logits_processor(
                **kwargs_common,
                model_kwargs=dict(self._model_kwargs_template),
                negative_prompt_ids=None,
                negative_prompt_attention_mask=None,
            )
        except TypeError:
            # older signature
            try:
                return self.model._get_logits_processor(**kwargs_common)
            except TypeError:
                return LogitsProcessorList()

    def _build_logits_warper(self) -> "LogitsProcessorList":
        # if model provides _get_logits_warper, use it
        if hasattr(self.model, "_get_logits_warper"):
            try:
                return self.model._get_logits_warper(self.generation_config)
            except Exception:
                pass

        # fallback manual warpers from generation_config
        warpers = LogitsProcessorList()

        temp = getattr(self.generation_config, "temperature", None)
        top_p = getattr(self.generation_config, "top_p", None)
        top_k = getattr(self.generation_config, "top_k", None)
        typical_p = getattr(self.generation_config, "typical_p", None)

        if TemperatureLogitsWarper is not None and temp is not None and float(temp) != 1.0:
            warpers.append(TemperatureLogitsWarper(float(temp)))
        if TopKLogitsWarper is not None and top_k is not None and int(top_k) > 0:
            warpers.append(TopKLogitsWarper(int(top_k)))
        if TopPLogitsWarper is not None and top_p is not None and float(top_p) < 1.0:
            warpers.append(TopPLogitsWarper(float(top_p)))
        if TypicalLogitsWarper is not None and typical_p is not None and float(typical_p) < 1.0:
            warpers.append(TypicalLogitsWarper(float(typical_p)))

        return warpers

    @torch.no_grad()
    def step(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        input_ids: [B,T] on CUDA
        returns processed scores: [B,V] (after logits_processor + logits_warper)
        """
        if self.past_key_values is None:
            inp = input_ids
        else:
            inp = input_ids[:, -1:]  # only last token if cached

        out = self.model(
            input_ids=inp,
            attention_mask=attention_mask,
            past_key_values=self.past_key_values,
            use_cache=True,
        )
        self.past_key_values = out.past_key_values
        logits = out.logits[:, -1, :]  # [B,V]

        processors = self._build_logits_processor(input_ids)
        if processors is not None and len(processors) > 0:
            logits = processors(input_ids, logits)

        warpers = self._build_logits_warper()
        if warpers is not None and len(warpers) > 0:
            logits = warpers(input_ids, logits)

        return logits
