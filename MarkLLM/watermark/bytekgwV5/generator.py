# MarkLLM/watermark/bytekgwV5/generator.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .aligned import HFAlignedStepper, StepState
from .samplers import TokenSampler, ByteTreeSampler
from .token_bytes import TokenByteVocab


def _cfg_get(cfg: Any, name: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


@dataclass
class GenerateResult:
    input_ids: Tensor           # [B, T_total]
    prompt_len: int
    scores_post: Optional[List[Tensor]] = None  # list of [B,V] per step (optional)


class ByteKGWv5:
    """
    V5 generation engine:
      - Uses HFAlignedStepper to compute HF-consistent post-warp scores per step.
      - Supports two sampling schemes:
          scheme="token"     -> TokenSampler (delta=0 can perfectly align HF.generate)
          scheme="byte_tree" -> ByteTreeSampler (delta=0 distributionally equivalent; not per-run identical)
      - Watermark bias:
          token scheme: (optional) you can later add token-level biaser;
                        for now delta is handled inside ByteTreeSampler branches.
          byte_tree scheme: branch bias via PRF green bytes (ByteTreeSampler).

    This class is intentionally "engine-level" (no MarkLLM BaseWatermark glue).
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        *,
        wm_cfg: Any,
        gen_cfg: Any,
        prf: Any,
        vocab: Optional[TokenByteVocab] = None,
        device: Union[str, torch.device] = "cuda:0",
        use_cache: bool = True,
        scheme: str = "token",
        use_torch_generator: bool = False,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.use_cache = bool(use_cache)

        self.wm_cfg = wm_cfg
        self.gen_cfg = gen_cfg
        self.scheme = str(scheme)
        if self.scheme not in ("token", "byte_tree"):
            raise ValueError("scheme must be 'token' or 'byte_tree'")

        self.delta = float(_cfg_get(wm_cfg, "delta", 0.0))
        self.prefix_length = int(_cfg_get(wm_cfg, "prefix_length", 4))

        self.max_new_tokens = int(_cfg_get(gen_cfg, "max_new_tokens", 128))
        self.do_sample = bool(_cfg_get(gen_cfg, "do_sample", False))

        # eos/pad
        self.eos_token_id = _cfg_get(gen_cfg, "eos_token_id", getattr(tokenizer, "eos_token_id", None))
        self.pad_token_id = _cfg_get(gen_cfg, "pad_token_id", getattr(tokenizer, "pad_token_id", None))

        # stepper (post-warp logits)
        self.stepper = HFAlignedStepper(
            model=model,
            tokenizer=tokenizer,
            gen_cfg=gen_cfg,
            device=self.device,
            use_cache=self.use_cache,
        )

        # vocab / samplers
        if vocab is None:
            vocab = TokenByteVocab.from_tokenizer(tokenizer, skip_markers=True).to(self.device)
        else:
            vocab = vocab.to(self.device)
        self.vocab = vocab

        self.prf = prf  # BytePRF instance (already on device ideally)
        try:
            self.prf.to(self.device)
        except Exception:
            pass

        self.token_sampler = TokenSampler(gen_cfg, self.device)
        self.byte_sampler = ByteTreeSampler(vocab=self.vocab, prf=self.prf, wm_cfg=wm_cfg, gen_cfg=gen_cfg, device=self.device)

        # RNG handling: either rely on global torch RNG (HF default),
        # or use a dedicated torch.Generator to isolate consumption.
        self.use_torch_generator = bool(use_torch_generator)
        self._generator: Optional[torch.Generator] = None
        if self.use_torch_generator:
            self._generator = torch.Generator(device=str(self.device))

    def set_seed(self, seed: int) -> None:
        """
        Set deterministic seed for this engine.
        Note: If use_torch_generator=False, this only seeds global RNG.
        """
        seed = int(seed)
        torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        if self._generator is not None:
            self._generator.manual_seed(seed)

    def _encode_prompt(self, prompt: str, *, add_special_tokens: bool) -> Tuple[Tensor, Tensor]:
        enc = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=bool(add_special_tokens),
        )
        input_ids = enc["input_ids"].to(self.device)
        attn = enc.get("attention_mask", None)
        if attn is None:
            attn = torch.ones_like(input_ids, dtype=torch.long, device=self.device)
        else:
            attn = attn.to(self.device)
        return input_ids, attn

    def _should_stop(self, next_ids: Tensor) -> Tensor:
        """
        next_ids: [B]
        returns stop mask [B] (True means stop)
        """
        if self.eos_token_id is None:
            return torch.zeros((next_ids.size(0),), device=next_ids.device, dtype=torch.bool)
        return next_ids.eq(int(self.eos_token_id))

    @torch.no_grad()
    def generate_ids(
        self,
        prompt: str,
        *,
        add_special_tokens: bool = True,
        max_new_tokens: Optional[int] = None,
        return_scores: bool = False,
    ) -> GenerateResult:
        """
        Generate token ids using the configured scheme.

        return_scores:
          if True, returns list of per-step post-warp scores (before sampling).
        """
        max_new = int(self.max_new_tokens if max_new_tokens is None else max_new_tokens)

        input_ids, attn = self._encode_prompt(prompt, add_special_tokens=add_special_tokens)
        B = input_ids.size(0)
        prompt_len = int(input_ids.size(1))

        state = self.stepper.init_state(input_ids, attention_mask=attn)

        scores_hist: Optional[List[Tensor]] = [] if return_scores else None

        finished = torch.zeros((B,), device=self.device, dtype=torch.bool)

        for _ in range(max_new):
            scores_post, state = self.stepper.step(state)  # [B,V] HF post-warp

            if return_scores:
                # store a detached copy to keep memory stable
                scores_hist.append(scores_post.detach().clone())

            # sample
            if self.scheme == "token":
                next_ids, _dbg = self.token_sampler.sample(scores_post, generator=self._generator)
            else:
                # byte-tree needs context ids (use full state.input_ids; PRF can internally slice)
                next_ids, _dbg = self.byte_sampler.sample(scores_post, ctx_ids=state.input_ids, generator=self._generator)

            # append
            state = self.stepper.append(state, next_ids)

            # stop on eos
            finished = finished | self._should_stop(next_ids)
            if bool(finished.all().item()):
                break

        return GenerateResult(input_ids=state.input_ids, prompt_len=prompt_len, scores_post=scores_hist)

    @torch.no_grad()
    def decode(self, ids: Tensor, *, skip_special_tokens: bool = True) -> str:
        if ids.dim() == 2:
            ids = ids[0]
        return self.tokenizer.decode(ids.tolist(), skip_special_tokens=skip_special_tokens)

    @torch.no_grad()
    def generate_text(
        self,
        prompt: str,
        *,
        add_special_tokens: bool = True,
        max_new_tokens: Optional[int] = None,
        skip_special_tokens: bool = True,
    ) -> str:
        out = self.generate_ids(prompt, add_special_tokens=add_special_tokens, max_new_tokens=max_new_tokens, return_scores=False)
        return self.decode(out.input_ids, skip_special_tokens=skip_special_tokens)
