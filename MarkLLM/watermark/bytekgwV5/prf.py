# MarkLLM/watermark/bytekgwV5/prf.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import torch
from torch import Tensor


@dataclass
class PRFConfig:
    """
    Deterministic PRF config for byte-tree watermarking.
    - hash_key: global secret (int)
    - gamma: fraction of green bytes in [0,1] (K = round(gamma*256), clamped to [1,256])
    """
    hash_key: int = 15485863
    gamma: float = 0.5


class BytePRF:
    """
    Deterministic PRF producing a per-step "green bytes" set.

    API:
      green_mask(ctx_ids, byte_pos, prefix_bytes=None) -> [B,256] bool

    Design notes:
    - Works on CUDA and CPU (all int64 ops), but intended for CUDA use.
    - Avoids uint64 and avoids constants >= 2^63 (CUDA signed int64 safety).
    - Seed depends on:
        (hash_key, ctx token ids, byte_pos, optional prefix_bytes)
    - Green bytes = K smallest hashes over bytes 0..255 (K=round(gamma*256))
    """

    def __init__(self, cfg: PRFConfig, device: Union[str, torch.device] = "cuda:0") -> None:
        self.cfg = cfg
        self.device = torch.device(device)

        K = int(round(float(cfg.gamma) * 256.0))
        self.K = max(1, min(256, K))

        # constants (< 2^63)
        self._A = 6364136223846793005  # LCG mult
        self._B = 1442695040888963407  # LCG inc
        self._C = 2862933555777941757  # extra mix mult
        self._MASK63 = (1 << 63) - 1

        self._bytes = torch.arange(256, device=self.device, dtype=torch.int64)  # [256]

    def to(self, device: Union[str, torch.device]) -> "BytePRF":
        self.device = torch.device(device)
        self._bytes = torch.arange(256, device=self.device, dtype=torch.int64)
        return self

    @torch.no_grad()
    def _mix63(self, x: Tensor) -> Tensor:
        """
        63-bit mix (LCG + xorshift), keep non-negative.
        x: int64 tensor
        """
        x = x.to(self.device, dtype=torch.int64) & self._MASK63
        x = (x * self._A + self._B) & self._MASK63
        x ^= (x >> 23)
        x = (x * self._C) & self._MASK63
        x ^= (x >> 29)
        return x & self._MASK63

    @torch.no_grad()
    def _seed63(self, ctx_ids: Tensor, *, byte_pos: int, prefix_bytes: Optional[Tensor]) -> Tensor:
        """
        ctx_ids: [B,L] long
        prefix_bytes: [B,P] uint8 or None
        returns: [B] int64 in [0..2^63-1]
        """
        if ctx_ids.dim() == 1:
            ctx_ids = ctx_ids.unsqueeze(0)
        x = ctx_ids.to(self.device, dtype=torch.int64) & self._MASK63
        B = x.size(0)

        # start from key
        s = torch.full(
            (B,),
            int(self.cfg.hash_key) & self._MASK63,
            device=self.device,
            dtype=torch.int64,
        )

        # fold ctx tokens (order-sensitive, but cheap)
        # s = mix( s ^ mix(tok_j + j + 1 + byte_pos) )
        inc = int(byte_pos) + 1
        for j in range(x.size(1)):
            tok = (x[:, j] + (j + 1) + inc) & self._MASK63
            s = (s ^ self._mix63(tok)) & self._MASK63
            s = self._mix63(s)

        # fold byte_pos itself
        bp = torch.full((B,), int(byte_pos) & self._MASK63, device=self.device, dtype=torch.int64)
        s = (s ^ self._mix63(bp)) & self._MASK63
        s = self._mix63(s)

        # fold prefix bytes if provided (sum is enough; this is just to decorrelate steps)
        if prefix_bytes is not None and prefix_bytes.numel() > 0:
            pb = prefix_bytes.to(self.device, dtype=torch.int64)  # [B,P]
            pb_sum = (pb.sum(dim=1) & self._MASK63)  # [B]
            s = (s ^ self._mix63(pb_sum)) & self._MASK63
            s = self._mix63(s)

        return s & self._MASK63

    @torch.no_grad()
    def _hash32(self, seed63: Tensor) -> Tensor:
        """
        Hash bytes 0..255 with a 32-bit mix keyed by seed63.

        seed63: [B] int64
        returns: [B,256] int64 in [0..2^32-1]
        """
        seed63 = seed63.to(self.device, dtype=torch.int64) & self._MASK63
        # derive seed32
        seed32 = (seed63 ^ (seed63 >> 31)) & 0xFFFFFFFF
        seed32 = seed32.unsqueeze(1)  # [B,1]

        b = self._bytes.unsqueeze(0)  # [1,256]
        h = (seed32 ^ b) & 0xFFFFFFFF

        # 32-bit mix in int64 lanes
        h ^= (h >> 16)
        h = (h * 0x7FEB352D) & 0xFFFFFFFF
        h ^= (h >> 15)
        h = (h * 0x846CA68B) & 0xFFFFFFFF
        h ^= (h >> 16)

        return h  # [B,256]

    @torch.no_grad()
    def green_ids(
        self,
        ctx_ids: Tensor,
        *,
        byte_pos: int = 0,
        prefix_bytes: Optional[Tensor] = None,
    ) -> Tensor:
        """
        returns: [B,K] int64 in [0..255]
        """
        if ctx_ids.device != self.device:
            ctx_ids = ctx_ids.to(self.device)
        if ctx_ids.dim() == 1:
            ctx_ids = ctx_ids.unsqueeze(0)

        seed = self._seed63(ctx_ids, byte_pos=int(byte_pos), prefix_bytes=prefix_bytes)
        h = self._hash32(seed)  # [B,256]

        # pick K smallest hashes as green bytes
        _, idx = torch.topk(h, k=self.K, dim=1, largest=False, sorted=False)  # [B,K]
        return idx.to(torch.int64)

    @torch.no_grad()
    def green_mask(
        self,
        ctx_ids: Tensor,
        *,
        byte_pos: int = 0,
        prefix_bytes: Optional[Tensor] = None,
    ) -> Tensor:
        """
        returns: [B,256] bool
        """
        if ctx_ids.device != self.device:
            ctx_ids = ctx_ids.to(self.device)
        if ctx_ids.dim() == 1:
            ctx_ids = ctx_ids.unsqueeze(0)

        idx = self.green_ids(ctx_ids, byte_pos=int(byte_pos), prefix_bytes=prefix_bytes)  # [B,K]
        B = idx.size(0)
        mask = torch.zeros((B, 256), device=self.device, dtype=torch.bool)
        mask.scatter_(1, idx, True)
        return mask
