# MarkLLM/watermark/bytekgwV5/aligned.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Union, Dict

import torch
from torch import Tensor

from transformers import PreTrainedModel, PreTrainedTokenizerBase

# Logits processors / warpers live here across most recent Transformers versions.
from transformers.generation.logits_process import LogitsProcessorList

try:
    from transformers.generation.logits_process import (
        RepetitionPenaltyLogitsProcessor,
        NoRepeatNGramLogitsProcessor,
        MinLengthLogitsProcessor,
        TemperatureLogitsWarper,
        TopKLogitsWarper,
        TopPLogitsWarper,
    )
except Exception:  # pragma: no cover
    # In case some names move in older versions, keep imports minimal.
    RepetitionPenaltyLogitsProcessor = None
    NoRepeatNGramLogitsProcessor = None
    MinLengthLogitsProcessor = None
    TemperatureLogitsWarper = None
    TopKLogitsWarper = None
    TopPLogitsWarper = None


def _cfg_get(cfg: Any, name: str, default: Any = None) -> Any:
    """Read generation knobs from dict-like or attribute-like configs."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def _make_position_ids(attention_mask: Tensor, input_ids_chunk_len: int) -> Tensor:
    """
    Build position_ids consistent with common decoder-only LMs when using KV cache:
      position_ids = cumsum(attn_mask) - 1
    and slice to the last `input_ids_chunk_len` positions.
    """
    # [B,T]
    pos = attention_mask.long().cumsum(dim=-1) - 1
    pos = pos.clamp_min(0)
    if input_ids_chunk_len < pos.size(1):
        pos = pos[:, -input_ids_chunk_len:]
    return pos


@dataclass
class StepState:
    """Mutable generation state for an aligned, step-by-step forward."""
    input_ids: Tensor                 # [B,T] long
    attention_mask: Tensor            # [B,T] long/bool
    past_key_values: Any = None       # model-specific cache object/tuple
    finished: Optional[Tensor] = None # [B] bool (optional)


class HFAlignedStepper:
    """
    A lightweight stepper that replicates the *per-step* logits processing path of HF generate:
      model forward -> next_token_logits -> logits_processors -> logits_warpers

    IMPORTANT DESIGN CHOICE:
    - We DO NOT call model.prepare_inputs_for_generation() to avoid `cache_position=None` issues
      seen in some Transformers versions. Instead, we:
        - first step: forward full prompt (past=None)
        - next steps: forward only last token with past_key_values, and compute position_ids
          from attention_mask (works for Llama-style decoder-only models).
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        gen_cfg: Any,
        *,
        device: Union[str, torch.device] = "cuda:0",
        use_cache: bool = True,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.use_cache = bool(use_cache)

        # snapshot knobs
        self.do_sample = bool(_cfg_get(gen_cfg, "do_sample", False))
        self.temperature = float(_cfg_get(gen_cfg, "temperature", 1.0) or 1.0)
        self.top_k = int(_cfg_get(gen_cfg, "top_k", 0) or 0)
        self.top_p = float(_cfg_get(gen_cfg, "top_p", 1.0) or 1.0)
        self.repetition_penalty = float(_cfg_get(gen_cfg, "repetition_penalty", 1.0) or 1.0)
        self.no_repeat_ngram_size = int(_cfg_get(gen_cfg, "no_repeat_ngram_size", 0) or 0)
        self.min_length = int(_cfg_get(gen_cfg, "min_length", 0) or 0)

        # eos/pad (for processors like MinLength)
        self.eos_token_id = _cfg_get(gen_cfg, "eos_token_id", getattr(tokenizer, "eos_token_id", None))
        self.pad_token_id = _cfg_get(gen_cfg, "pad_token_id", getattr(tokenizer, "pad_token_id", None))

        self.logits_processor = self._build_logits_processor()
        self.logits_warper = self._build_logits_warper()

        self.model.to(self.device)
        self.model.eval()

    def _build_logits_processor(self) -> LogitsProcessorList:
        procs = LogitsProcessorList()

        if self.repetition_penalty != 1.0 and RepetitionPenaltyLogitsProcessor is not None:
            procs.append(RepetitionPenaltyLogitsProcessor(penalty=self.repetition_penalty))

        if self.no_repeat_ngram_size and self.no_repeat_ngram_size > 0 and NoRepeatNGramLogitsProcessor is not None:
            procs.append(NoRepeatNGramLogitsProcessor(self.no_repeat_ngram_size))

        if self.min_length and self.min_length > 0 and MinLengthLogitsProcessor is not None:
            # HF uses eos_token_id for min_length constraint
            if self.eos_token_id is not None:
                procs.append(MinLengthLogitsProcessor(self.min_length, eos_token_id=int(self.eos_token_id)))

        return procs

    def _build_logits_warper(self) -> LogitsProcessorList:
        warpers = LogitsProcessorList()

        # HF only uses warpers when do_sample=True
        if not self.do_sample:
            return warpers

        if self.temperature is not None and self.temperature != 1.0 and TemperatureLogitsWarper is not None:
            warpers.append(TemperatureLogitsWarper(self.temperature))

        # top_k: HF treats <=0 as disabled
        if self.top_k is not None and int(self.top_k) > 0 and TopKLogitsWarper is not None:
            warpers.append(TopKLogitsWarper(int(self.top_k)))

        # top_p: HF treats >=1 as disabled
        if self.top_p is not None and float(self.top_p) < 1.0 and TopPLogitsWarper is not None:
            warpers.append(TopPLogitsWarper(float(self.top_p)))

        return warpers

    @torch.no_grad()
    def init_state(self, input_ids: Tensor, attention_mask: Optional[Tensor] = None) -> StepState:
        if input_ids.device != self.device:
            input_ids = input_ids.to(self.device)
        if input_ids.dtype != torch.long:
            input_ids = input_ids.to(torch.long)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=self.device)
        else:
            if attention_mask.device != self.device:
                attention_mask = attention_mask.to(self.device)
            if attention_mask.dtype not in (torch.long, torch.int64, torch.bool):
                attention_mask = attention_mask.to(torch.long)

        finished = torch.zeros((input_ids.size(0),), device=self.device, dtype=torch.bool)
        return StepState(input_ids=input_ids, attention_mask=attention_mask, past_key_values=None, finished=finished)

    @torch.no_grad()
    def append(self, state: StepState, next_token_ids: Tensor) -> StepState:
        """
        Append sampled token(s) to the state. next_token_ids should be [B] or [B,1].
        """
        if next_token_ids.dim() == 1:
            next_token_ids = next_token_ids.view(-1, 1)
        if next_token_ids.device != self.device:
            next_token_ids = next_token_ids.to(self.device)
        if next_token_ids.dtype != torch.long:
            next_token_ids = next_token_ids.to(torch.long)

        state.input_ids = torch.cat([state.input_ids, next_token_ids], dim=1)
        add_mask = torch.ones((state.attention_mask.size(0), 1), device=self.device, dtype=state.attention_mask.dtype)
        state.attention_mask = torch.cat([state.attention_mask, add_mask], dim=1)
        return state

    @torch.no_grad()
    def step(self, state: StepState) -> tuple[Tensor, StepState]:
        """
        Compute post-processor/warper logits for the *next token*.
        Returns:
          scores_post: [B,V]
          updated_state: StepState with updated past_key_values
        """
        input_ids = state.input_ids
        attn = state.attention_mask

        if input_ids.device != self.device:
            input_ids = input_ids.to(self.device)
        if attn.device != self.device:
            attn = attn.to(self.device)

        # Prepare model inputs.
        # First step: feed full prompt.
        # Subsequent steps: feed last token only, with KV cache.
        model_inputs: Dict[str, Any] = {"use_cache": self.use_cache}

        if state.past_key_values is None:
            model_inputs["input_ids"] = input_ids
            model_inputs["attention_mask"] = attn
        else:
            last_ids = input_ids[:, -1:].contiguous()
            model_inputs["input_ids"] = last_ids
            model_inputs["attention_mask"] = attn
            model_inputs["past_key_values"] = state.past_key_values

            # Many decoder-only models (e.g., Llama) accept position_ids.
            pos_ids = _make_position_ids(attn, input_ids_chunk_len=last_ids.size(1))
            model_inputs["position_ids"] = pos_ids

        # Forward with a safe fallback if model doesn't accept position_ids.
        try:
            out = self.model(**model_inputs, return_dict=True)
        except TypeError:
            model_inputs.pop("position_ids", None)
            out = self.model(**model_inputs, return_dict=True)

        logits = out.logits  # [B, T_chunk, V]
        if logits.dim() != 3:
            raise RuntimeError(f"Unexpected logits shape: {tuple(logits.shape)}")

        next_token_logits = logits[:, -1, :]  # [B,V]

        # Apply processors then warpers (same order as HF generate)
        scores = next_token_logits
        if len(self.logits_processor) > 0:
            scores = self.logits_processor(input_ids, scores)
        if len(self.logits_warper) > 0:
            scores = self.logits_warper(input_ids, scores)

        # Update cache
        pkv = getattr(out, "past_key_values", None)
        state.past_key_values = pkv if self.use_cache else None

        return scores, state
