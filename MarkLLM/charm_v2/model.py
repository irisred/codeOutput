from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import Tensor
from transformers import GenerationConfig, PreTrainedModel

from MarkLLM.charm_v2.kv_cache import _DualKVCache


class CharmGenerationModel:
    """
    Minimal façade over a HuggingFace causal LM that exposes a single method:
    given input_ids, return the processed next-token logits. It owns the
    KV cache so repeated calls with growing prefixes stay efficient.
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
        self._model_kwargs_template = {}
        # Let HF prepare the exact GenerationConfig that `model.generate(**gen_params)`
        # would use, so that all logits processors / warpers match official behavior.
        self.generation_config, self._model_kwargs_template = self.model._prepare_generation_config(
            None,
            use_model_defaults=True,
            **(gen_params or {}),
        )
        # Ensure pad_token_id is set consistently (HF will still treat eos as pad at
        # generate-time if needed, but we also patch the model config for safety).
        self._ensure_pad_token()

    def _ensure_pad_token(self) -> None:
        pad_id = getattr(self.model.config, "pad_token_id", None)
        eos_id = getattr(self.model.config, "eos_token_id", None)
        if pad_id is None:
            pad_id = eos_id
        if pad_id is None:
            pad_id = getattr(self.generation_config, "eos_token_id", None)
        if self.generation_config.pad_token_id is None and pad_id is not None:
            self.generation_config.pad_token_id = pad_id
        if self.model.config.pad_token_id is None and pad_id is not None:
            self.model.config.pad_token_id = pad_id

    def _clone_generation_config(self) -> GenerationConfig:
        cfg = self.generation_config
        if hasattr(cfg, "clone"):
            return cfg.clone()
        return GenerationConfig.from_dict(cfg.to_dict())

    def reset(self) -> None:
        self.cache.reset()

    def dump_cache_snapshot(self, path: str, *, extra: Optional[Dict[str, Any]] = None) -> None:
        payload: Dict[str, Any] = {
            "cache": self.cache.snapshot(),
            "generation_config": self.generation_config.to_dict(),
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)

    def load_cache_snapshot(self, path: str, device: Optional[torch.device] = None) -> Dict[str, Any]:
        data = torch.load(path, map_location="cpu")
        target_device = device if device is not None else self.device
        cache_state = data.get("cache")
        if cache_state is not None:
            self.cache.restore(cache_state, target_device)
        return data

    def _compute_processed_scores(self, input_ids: Tensor) -> Tensor:
        # Debug: inspect raw and processed logits for the specific prompt#2 + continuation case.
        ids_list = input_ids[0].tolist()
      
        if input_ids.device != self.device:
            input_ids = input_ids.to(self.device)

        raw_logits = self.cache.forward_for_sequence(self.model, input_ids)

        if raw_logits.dim() == 1:
            scores = raw_logits.unsqueeze(0)
        else:
            scores = raw_logits[:, -1, :]

        logits_processor = self.model._get_logits_processor(
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



        if logits_processor:
            scores = logits_processor(input_ids, scores)


        return scores.squeeze(0)

    @torch.inference_mode()
    def compute_logits(self, input_ids: Tensor) -> Tensor:
        """
        input_ids: [1, T] tensor on any device.
        returns: processed logits vector [V] on model.device.
        """
        if input_ids.device != self.device:
            input_ids = input_ids.to(self.device)

        scores = self._compute_processed_scores(input_ids)
        return scores

    @torch.inference_mode()
    def compute_logits_via_generate(self, input_ids: Tensor) -> Tensor:
        """
        Debug helper aligned with HuggingFace GenerationMixin: for the provided sequence,
        obtain next-token logits after applying the exact same logits processors / warpers
        that `model.generate()` would use (temperature, top-p, repetition penalty, etc.).
        """
        if input_ids.device != self.device:
            input_ids = input_ids.to(self.device)

        scores = self._compute_processed_scores(input_ids)
        return scores
