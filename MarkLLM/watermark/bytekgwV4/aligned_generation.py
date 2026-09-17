from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor
from transformers import GenerationConfig, PreTrainedModel

from MarkLLM.charm_v2.kv_cache import _DualKVCache


class AlignedGenerationModel:
    """
    HF-aligned step facade:
      - owns KV cache
      - produces next-token logits with HF official:
          scores = logits_processor(input_ids, scores)
          if do_sample: scores = logits_warper(input_ids, scores)
    """

    def __init__(
        self,
        model: PreTrainedModel,
        *,
        gen_params: Optional[Dict[str, Any]] = None,
    ):
        self.model = model
        self.device = next(model.parameters()).device
        self.cache = _DualKVCache()
        self._model_kwargs_template: Dict[str, Any] = {}

        # Prepare EXACT GenerationConfig as model.generate(**gen_params) would use
        self.generation_config, self._model_kwargs_template = self.model._prepare_generation_config(
            None,
            use_model_defaults=True,
            **(gen_params or {}),
        )
        self._ensure_pad_token()

    def _ensure_pad_token(self) -> None:
        pad_id = getattr(self.model.config, "pad_token_id", None)
        eos_id = getattr(self.model.config, "eos_token_id", None)
        if pad_id is None:
            pad_id = eos_id
        if pad_id is None:
            pad_id = getattr(self.generation_config, "eos_token_id", None)
        if self.generation_config.pad_token_id is None and pad_id is not None:
            self.generation_config.pad_token_id = int(pad_id)
        if getattr(self.model.config, "pad_token_id", None) is None and pad_id is not None:
            self.model.config.pad_token_id = int(pad_id)

    def reset(self) -> None:
        self.cache.reset()

    def _clone_generation_config(self) -> GenerationConfig:
        cfg = self.generation_config
        if hasattr(cfg, "clone"):
            return cfg.clone()
        return GenerationConfig.from_dict(cfg.to_dict())

    def _get_logits_processor(self, input_ids: Tensor):
        return self.model._get_logits_processor(
            generation_config=self.generation_config,
            input_ids_seq_length=int(input_ids.shape[-1]),
            encoder_input_ids=None,
            prefix_allowed_tokens_fn=None,
            logits_processor=None,
            device=self.device,
            model_kwargs=dict(self._model_kwargs_template),
            negative_prompt_ids=None,
            negative_prompt_attention_mask=None,
        )

    def _get_logits_warper(self, input_ids: Tensor):
        # Newer HF: model._get_logits_warper exists
        if hasattr(self.model, "_get_logits_warper"):
            try:
                return self.model._get_logits_warper(
                    generation_config=self.generation_config,
                    device=self.device,
                )
            except TypeError:
                # Signature variations across versions
                try:
                    return self.model._get_logits_warper(self.generation_config)
                except Exception:
                    pass

        # Fallback: compose warpers manually from generation_config
        from transformers.generation.logits_process import (
            TemperatureLogitsWarper,
            TopKLogitsWarper,
            TopPLogitsWarper,
        )

        warpers = []
        cfg = self.generation_config

        temp = getattr(cfg, "temperature", None)
        if temp is not None and float(temp) != 1.0:
            warpers.append(TemperatureLogitsWarper(float(temp)))

        top_k = getattr(cfg, "top_k", None)
        if top_k is not None and int(top_k) > 0:
            warpers.append(TopKLogitsWarper(int(top_k)))

        top_p = getattr(cfg, "top_p", None)
        if top_p is not None and float(top_p) < 1.0:
            warpers.append(TopPLogitsWarper(float(top_p)))

        # Warper list is callable like LogitsProcessorList; we can just return a list and apply sequentially.
        return warpers

    @torch.inference_mode()
    def compute_logits_pre_warp(self, input_ids: Tensor) -> Tensor:
        """
        Return processed logits after HF logits_processor (repetition penalty, etc.),
        but BEFORE sampling warpers (temperature/top_p/top_k).
        """
        if input_ids.device != self.device:
            input_ids = input_ids.to(self.device)

        raw_logits = self.cache.forward_for_sequence(self.model, input_ids)  # [1,T,V] or [V]
        if raw_logits.dim() == 1:
            scores = raw_logits.unsqueeze(0)  # [1,V]
        else:
            scores = raw_logits[:, -1, :]     # [1,V]

        lp = self._get_logits_processor(input_ids)
        if lp:
            scores = lp(input_ids, scores)

        return scores.squeeze(0)  # [V]

    @torch.inference_mode()
    def apply_warpers(self, input_ids: Tensor, scores: Tensor) -> Tensor:
        """
        Apply sampling warpers (temperature/top_p/top_k) if do_sample=True in generation_config.
        scores: [V] or [1,V]
        """
        if input_ids.device != self.device:
            input_ids = input_ids.to(self.device)
        if scores.dim() == 1:
            scores_2d = scores.unsqueeze(0)
        else:
            scores_2d = scores

        do_sample = bool(getattr(self.generation_config, "do_sample", False))
        if not do_sample:
            return scores_2d.squeeze(0)

        warper = self._get_logits_warper(input_ids)

        if warper is None:
            return scores_2d.squeeze(0)

        if isinstance(warper, (list, tuple)):
            x = scores_2d
            for w in warper:
                x = w(input_ids, x)
            return x.squeeze(0)

        # HF LogitsProcessorList-like
        return warper(input_ids, scores_2d).squeeze(0)

    @torch.inference_mode()
    def compute_logits_full(self, input_ids: Tensor) -> Tensor:
        """
        Convenience: logits_processor + (if do_sample) logits_warper.
        """
        scores = self.compute_logits_pre_warp(input_ids)
        scores = self.apply_warpers(input_ids, scores)
        return scores
