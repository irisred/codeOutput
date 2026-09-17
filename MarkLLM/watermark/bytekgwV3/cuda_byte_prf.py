# MarkLLM/watermark/bytekgwV2/cuda_byte_prf.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class BytePRFConfig:
    gamma: float = 0.5
    hash_key: int = 15485863


class CudaBytePRF:
    """
    CUDA-only PRF:
      seed = f(ctx_tokens, prefix_bytes, byte_pos, hash_key)  (all int64, 63-bit non-negative)
      green bytes = topK of splitmix-like hash(seed ^ byte)
    """

    def __init__(self, cfg: BytePRFConfig, device: str | torch.device = "cuda:0"):
        self.cfg = cfg
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("CudaBytePRF is CUDA-only.")
        self.K = int(round(256 * float(cfg.gamma)))
        self.K = max(1, min(255, self.K))

        self.mask63 = torch.tensor(0x7FFFFFFFFFFFFFFF, device=self.device, dtype=torch.int64)
        self.key63 = torch.tensor(int(cfg.hash_key) & 0x7FFFFFFFFFFFFFFF, device=self.device, dtype=torch.int64)
        self.bytes = torch.arange(256, device=self.device, dtype=torch.int64)  # int64 OK on CUDA

    @torch.no_grad()
    def _mix63(self, x: torch.Tensor) -> torch.Tensor:
        # simple 63-bit mix (LCG + xorshift), keep non-negative
        x = (x * 6364136223846793005 + 1442695040888963407) & self.mask63
        x ^= (x >> 23)
        x = (x * 2862933555777941757) & self.mask63
        x ^= (x >> 29)
        return x & self.mask63

    @torch.no_grad()
    def _seed(self, ctx: torch.Tensor, prefix_bytes: Optional[torch.Tensor], byte_pos: int) -> torch.Tensor:
        """
        ctx: [B,L] int64/long on CUDA
        prefix_bytes: [B,P] uint8 on CUDA or None
        return: [B] int64 non-negative
        """
        x = self.key63.clone().expand(ctx.size(0))
        # fold ctx tokens
        t = ctx.to(torch.int64) & self.mask63
        x = (x ^ self._mix63(t.sum(dim=1))) & self.mask63

        # fold byte_pos
        bp = torch.tensor(int(byte_pos) & 0x7FFFFFFFFFFFFFFF, device=self.device, dtype=torch.int64)
        x = (x ^ self._mix63(bp.expand_as(x))) & self.mask63

        # fold prefix bytes if provided
        if prefix_bytes is not None and prefix_bytes.numel() > 0:
            pb = prefix_bytes.to(torch.int64)  # [B,P]
            x = (x ^ self._mix63(pb.sum(dim=1))) & self.mask63

        return x & self.mask63

    @torch.no_grad()
    def green_bytes_mask(
        self,
        ctx: torch.Tensor,                        # [B,L] long on CUDA
        prefix_bytes: Optional[torch.Tensor] = None,  # [B,P] uint8 on CUDA
        byte_pos: int = 0,
    ) -> torch.Tensor:
        """
        return: [B,256] bool
        """
        if ctx.device != self.device:
            raise ValueError("ctx must be on PRF CUDA device.")
        B = ctx.size(0)

        seed = self._seed(ctx, prefix_bytes, int(byte_pos))  # [B]
        # hash per byte: [B,256]
        h = self._mix63(seed[:, None] ^ self.bytes[None, :])  # int64 non-negative

        # choose topK as green (largest h)
        top = torch.topk(h, k=self.K, dim=1, largest=True).indices  # [B,K]
        mask = torch.zeros((B, 256), device=self.device, dtype=torch.bool)
        mask.scatter_(1, top, True)
        return mask
