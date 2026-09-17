from __future__ import annotations

from __future__ import annotations

from typing import Iterable, Optional, List, Dict, Sequence

import torch

from .prf import ByteWindowPRF, TokenPrefixBytePRF
from .prf_trace import PRFTracer

class CharmByteLogitsProcessor:
    def __init__(
        self,
        *,
        hash_key: int,
        token_prefix_length: int,   # 这里是“前 h 个 token ids”
        gamma: float,
        delta: float,
        f_scheme: str = "additive",
        device: str | torch.device = "cpu",
    ) -> None:
        self.prf = TokenPrefixBytePRF(
            hash_key=hash_key,
            token_prefix_length=token_prefix_length,
            gamma=gamma,
            f_scheme=f_scheme,
            device=device,
        )
        self.gamma = float(gamma)
        self.delta = float(delta)
        self.entropy_threshold: float = 0.0
        self._trace: List[Dict[str, object]] = []
        self._debug_tracer: Optional[PRFTracer] = None
        self._debug_phase: str = "gen"
        self._debug_byte_pos_filter: Optional[int] = None

    def set_debug_tracer(
        self,
        tracer: Optional[PRFTracer],
        *,
        phase: str = "gen",
        byte_pos_filter: Optional[int] = None,
    ) -> None:
        self._debug_tracer = tracer
        self._debug_phase = phase
        self._debug_byte_pos_filter = byte_pos_filter

    def __call__(
        self,
        scores: torch.FloatTensor,
        *,
        token_ids: Sequence[int],     # 上下文 token ids（通常取前 h 个）
        prefix_bytes: bytes,          # 当前 token 已确定 bytes 前缀（不含本次要采的 byte）
        byte_pos: int,                # 当前在 token 内的 byte 位置
    ) -> torch.FloatTensor:

        # trace（可选）
        self._trace.append(
            {
                "step": len(self._trace),
                "byte_pos": int(byte_pos),
                "prefix_len": len(prefix_bytes),
                "token_ctx": list(token_ids),
            }
        )

        if scores.numel() == 0:
            return scores
        if scores.shape[-1] != 256:
            raise ValueError("CharmByteLogitsProcessor expects a [*, 256] logits vector.")

        # 熵门控（你原来就有）
        if self.entropy_threshold > 0.0:
            base_for_ent = scores if scores.dim() == 2 else scores.unsqueeze(0)
            probs = torch.softmax(base_for_ent, dim=-1)
            log_probs = torch.log(probs.clamp_min(1e-12))
            ent = -(probs * log_probs).sum(dim=-1)
            if float(ent.mean().item()) < self.entropy_threshold:
                return scores

        if self.delta == 0.0:
            return scores

        if (
            self._debug_tracer is not None
            and len(token_ids) >= self.prf.h
            and (self._debug_byte_pos_filter is None or self._debug_byte_pos_filter == byte_pos)
        ):
            self._debug_tracer.log(
                self._debug_phase,
                token_index=len(token_ids),
                token_ids=token_ids,
                prefix_bytes=bytes(prefix_bytes),
                byte_pos=int(byte_pos),
                greenlist=list(self.prf.greenlist(token_ids, prefix_bytes, byte_pos)),
            )

        greenlist: Iterable[int] = self.prf.greenlist(token_ids, prefix_bytes, byte_pos)
        if not greenlist:
            return scores

        base = scores if scores.dim() == 2 else scores.unsqueeze(0)
        idx = torch.tensor(list(greenlist), dtype=torch.long, device=base.device)
        if idx.numel() > 0:
            base[..., idx] = base[..., idx] + self.delta
        return base.squeeze(0)
    
    
__all__ = ["CharmByteLogitsProcessor"]
