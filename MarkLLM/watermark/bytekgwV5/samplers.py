# MarkLLM/watermark/bytekgwV5/samplers.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

import torch
from torch import Tensor

from .token_bytes import END_BYTE, TokenByteVocab


@dataclass
class _SamplerDebug:
    scheme: str
    selected_bytes: Optional[List[int]] = None
    final_candidate_count: Optional[int] = None


class TokenSampler:
    """
    Simple token-level sampler on HF post-warp scores.
    This sampler is used when scheme == "token".
    """

    def __init__(self, gen_cfg: Dict[str, Any], device: torch.device):
        self.device = device
        self.do_sample: bool = bool(gen_cfg.get("do_sample", False))
        self.sampling_dtype: str = str(gen_cfg.get("sampling_dtype", "model"))  # "model" or "float32"

    @torch.no_grad()
    def sample(
        self,
        scores_post: Tensor,                 # [B, V] (HF post-warp)
        *,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        if scores_post.dim() != 2:
            raise ValueError(f"TokenSampler expects [B,V], got {tuple(scores_post.shape)}")

        if not self.do_sample:
            next_ids = torch.argmax(scores_post, dim=-1, keepdim=True)
            return next_ids.to(torch.long), {"scheme": "token", "mode": "greedy"}

        # do_sample
        if self.sampling_dtype == "float32":
            probs = torch.softmax(scores_post.to(torch.float32), dim=-1)
        else:
            # still do softmax in float32 for numerical stability
            probs = torch.softmax(scores_post.to(torch.float32), dim=-1)

        next_ids = torch.multinomial(probs, num_samples=1, generator=generator)
        return next_ids.to(torch.long), {"scheme": "token", "mode": "multinomial"}


class ByteTreeSampler:
    """
    Byte-tree sampler:
      - Build 257-way branch masses at each byte position (0..255 + END_BYTE=256)
      - Apply watermark bias on GREEN bytes: mass *= exp(delta) for green branches (bytes only, not END)
      - Sample a branch byte, restrict candidate tokens, continue
      - Finally sample/choose a token from remaining candidates

    IMPORTANT (unified convention):
      - PRF context is ALWAYS the last L=prefix_length tokens (same as detector).
    """

    def __init__(
        self,
        *,
        vocab: TokenByteVocab,
        prf: Any,                  # BytePRF-like: green_mask(ctx_ids, byte_pos, prefix_bytes?)
        wm_cfg: Any,               # expects gamma, delta, prefix_length, max_byte_pos, use_prefix_bytes_in_prf
        gen_cfg: Dict[str, Any],
        device: torch.device,
    ):
        self.vocab = vocab
        self.prf = prf
        self.device = device

        # watermark cfg
        self.cfg = wm_cfg
        self.gamma: float = float(getattr(wm_cfg, "gamma", 0.5))
        self.delta: float = float(getattr(wm_cfg, "delta", 0.0))
        self.prefix_length: int = int(getattr(wm_cfg, "prefix_length", 4))
        self.max_byte_pos: int = int(getattr(wm_cfg, "max_byte_pos", 64))
        self.use_prefix_bytes_in_prf: bool = bool(getattr(wm_cfg, "use_prefix_bytes_in_prf", False))

        self._bias_factor: float = float(math.exp(self.delta)) if self.delta != 0.0 else 1.0

        # generation cfg
        self.do_sample: bool = bool(gen_cfg.get("do_sample", False))
        self.sampling_dtype: str = str(gen_cfg.get("sampling_dtype", "model"))  # "model" or "float32"

        # cache bytepos tables per pos on device
        self._bytepos_cache: Dict[int, Tensor] = {}

    def _bytepos_table(self, pos: int, V: int, device: torch.device) -> Tensor:
        """
        Return [V] int64 in [0..256] for byte position `pos`.
        Cached by pos, then sliced to V if needed.
        """
        pos = int(pos)
        if pos not in self._bytepos_cache:
            t = self.vocab.bytepos_tensor(device, pos).to(torch.long)  # [vocab_size]
            self._bytepos_cache[pos] = t
        t = self._bytepos_cache[pos]
        if t.numel() < V:
            raise ValueError(f"bytepos table too short: {t.numel()} < logits V={V}")
        if t.numel() > V:
            return t[:V]
        return t

    def _green_mask_256(
        self,
        ctx_wm: Tensor,                      # [1, L] (already sliced)
        byte_pos: int,
        prefix_bytes: Optional[Tensor],      # [1, p] uint8 or None
    ) -> Tensor:
        """
        Return [256] bool.
        Compatible with PRF signatures with/without prefix_bytes kwarg.
        """
        # PRF API variants: green_mask(ctx_ids, byte_pos=..., prefix_bytes=...) or green_mask(ctx_ids, byte_pos=...)
        try:
            m = self.prf.green_mask(ctx_wm, byte_pos=int(byte_pos), prefix_bytes=prefix_bytes)
        except TypeError:
            m = self.prf.green_mask(ctx_wm, byte_pos=int(byte_pos))
        if m.dim() == 2:
            m = m[0]
        return m.to(torch.bool)

    @torch.no_grad()
    def _apply_branch_bias(
        self,
        mass257: Tensor,                     # [257] float32
        *,
        ctx_wm: Tensor,                      # [1, L] long (already sliced)
        byte_pos: int,
        prefix_bytes: Optional[Tensor],      # [1, p] uint8 or None
    ) -> Tensor:
        """
        Apply exp(delta) bias to green byte branches (0..255). END branch (256) is NOT biased.
        Returns biased mass257 (float32).
        """
        if self.delta == 0.0:
            return mass257

        # gating: only apply watermark if we have at least L tokens in ctx
        if ctx_wm.size(1) < int(self.prefix_length):
            return mass257

        green = self._green_mask_256(ctx_wm, byte_pos=byte_pos, prefix_bytes=prefix_bytes)  # [256] bool

        # IMPORTANT: do NOT use chained index "*=" (can silently not write back in some patterns)
        w = float(self._bias_factor)
        mass256 = mass257[:256]
        mass257[:256] = torch.where(green, mass256 * w, mass256)
        return mass257

    @torch.no_grad()
    def sample(
        self,
        scores_post: Tensor,                 # [B, V]
        *,
        ctx_ids: Tensor,                     # [B, T]
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        if scores_post.dim() != 2:
            raise ValueError(f"ByteTreeSampler expects scores_post [B,V], got {tuple(scores_post.shape)}")
        if ctx_ids.dim() == 1:
            ctx_ids = ctx_ids.unsqueeze(0)
        if ctx_ids.dim() != 2:
            raise ValueError(f"ByteTreeSampler expects ctx_ids [B,T], got {tuple(ctx_ids.shape)}")

        B, V = int(scores_post.size(0)), int(scores_post.size(1))
        ctx_ids = ctx_ids.to(device=scores_post.device, dtype=torch.long)

        # probs in float32 for stability
        probs = torch.softmax(scores_post.to(torch.float32), dim=-1)  # [B, V]

        next_ids = torch.zeros((B, 1), device=scores_post.device, dtype=torch.long)
        dbg = {"scheme": "byte_tree", "delta": float(self.delta), "prefix_length": int(self.prefix_length)}

        # --- process each batch independently (B usually 1 in your use-case) ---
        for b in range(B):
            # ====== UNIFIED PRF CONTEXT CONVENTION ======
            # Always slice to last L tokens for PRF, consistent with detector: ctx = x[:, i-L:i]
            L = int(self.prefix_length)
            if L > 0 and ctx_ids.size(1) > L:
                ctx_wm = ctx_ids[b : b + 1, -L:]     # [1, L]
            else:
                ctx_wm = ctx_ids[b : b + 1, :]       # [1, T] (T<=L)

            cand_mask = torch.ones((V,), device=scores_post.device, dtype=torch.bool)
            selected_bytes: List[int] = []

            prefix_bytes_list: List[int] = []
            prefix_bytes_t: Optional[Tensor] = None  # [1,p] uint8

            # iterate byte positions
            for pos in range(int(self.max_byte_pos)):
                bytepos = self._bytepos_table(pos, V, scores_post.device)  # [V] long in 0..256

                probs_b = probs[b] * cand_mask.to(torch.float32)          # masked probs
                total_mass = float(probs_b.sum().item())
                if total_mass <= 0.0:
                    break

                # branch mass [257]
                mass257 = torch.zeros((257,), device=scores_post.device, dtype=torch.float32)
                mass257.scatter_add_(0, bytepos, probs_b)

                # prefix_bytes for PRF (optional)
                if self.use_prefix_bytes_in_prf and len(prefix_bytes_list) > 0:
                    prefix_bytes_t = torch.tensor(prefix_bytes_list, device=scores_post.device, dtype=torch.uint8).view(1, -1)
                else:
                    prefix_bytes_t = None

                # watermark bias
                mass257 = self._apply_branch_bias(
                    mass257,
                    ctx_wm=ctx_wm,
                    byte_pos=pos,
                    prefix_bytes=prefix_bytes_t,
                )

                # sample branch
                if self.do_sample:
                    denom = mass257.sum()
                    if float(denom.item()) <= 0.0:
                        break
                    p_branch = mass257 / denom
                    chosen = torch.multinomial(p_branch, num_samples=1, generator=generator)  # [1]
                    chosen_byte = int(chosen.item())
                else:
                    chosen_byte = int(torch.argmax(mass257).item())

                selected_bytes.append(chosen_byte)

                # restrict candidates
                cand_mask = cand_mask & (bytepos == chosen_byte)

                # update prefix bytes list (exclude END)
                if self.use_prefix_bytes_in_prf and chosen_byte != END_BYTE and 0 <= chosen_byte <= 255:
                    prefix_bytes_list.append(chosen_byte)

                # early stop if 0/1 candidate left
                cand_count = int(cand_mask.sum().item())
                if cand_count <= 1:
                    break

            # finalize token choice among remaining candidates
            cand_idx = cand_mask.nonzero(as_tuple=False).view(-1)  # [K]
            if cand_idx.numel() == 0:
                # fallback: sample/greedy from full distribution
                if self.do_sample:
                    next_ids[b, 0] = int(torch.multinomial(probs[b], 1, generator=generator).item())
                else:
                    next_ids[b, 0] = int(torch.argmax(probs[b]).item())
                final_k = 0
            elif cand_idx.numel() == 1:
                next_ids[b, 0] = int(cand_idx[0].item())
                final_k = 1
            else:
                # sample/greedy within candidate set
                p_cand = probs[b].index_select(0, cand_idx)
                s = float(p_cand.sum().item())
                if s <= 0.0:
                    # fallback to argmax among candidates by scores_post
                    sc = scores_post[b].index_select(0, cand_idx)
                    next_ids[b, 0] = int(cand_idx[int(torch.argmax(sc).item())].item())
                else:
                    if self.do_sample:
                        p_cand = p_cand / p_cand.sum()
                        j = int(torch.multinomial(p_cand, 1, generator=generator).item())
                        next_ids[b, 0] = int(cand_idx[j].item())
                    else:
                        j = int(torch.argmax(p_cand).item())
                        next_ids[b, 0] = int(cand_idx[j].item())
                final_k = int(cand_idx.numel())

            # attach per-batch debug
            dbg.setdefault("selected_bytes", [])
            dbg["selected_bytes"].append(selected_bytes)
            dbg.setdefault("final_candidate_count", [])
            dbg["final_candidate_count"].append(final_k)

        return next_ids, dbg


__all__ = ["TokenSampler", "ByteTreeSampler"]
