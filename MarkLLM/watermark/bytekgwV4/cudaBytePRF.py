from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass
class BytePRFConfig:
    hash_key: int = 15485863
    gamma: float = 0.5  # e.g. 128/256


class CudaBytePRF:
    """
    Stateless PRF on CUDA:
      ctx_ids [B,L] -> green bytes set (size K=round(gamma*256))

    We avoid uint64 on CUDA and avoid >2^63 constants. We:
      1) derive a per-batch 63-bit seed via a safe LCG mix over ctx token ids
      2) derive seed32, then hash each byte b in [0..255] using a 32-bit mix
      3) pick K smallest hashes => green bytes
    """

    def __init__(self, cfg: BytePRFConfig, device: str | torch.device = "cuda:0"):
        self.cfg = cfg
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("CudaBytePRF is CUDA-only.")
        self.K = int(round(float(cfg.gamma) * 256.0))
        self.K = max(1, min(256, self.K))

        self.bytes = torch.arange(256, device=self.device, dtype=torch.int64)  # [256]

        # Safe signed int64 constants (< 2^63)
        self._A = 6364136223846793005  # LCG multiplier (<2^63)
        self._B = 1442695040888963407  # LCG increment  (<2^63)
        self._MASK63 = (1 << 63) - 1

    @torch.no_grad()
    def _seed63(self, ctx_ids: torch.Tensor, byte_pos: int = 0) -> torch.Tensor:
        """
        ctx_ids: [B,L] long
        return: [B] int64 in [0..2^63-1]
        """
        x = ctx_ids.to(self.device, dtype=torch.int64)
        B = x.size(0)
        s = torch.full((B,), int(self.cfg.hash_key) & self._MASK63, device=self.device, dtype=torch.int64)

        # Mix tokens (LCG-like) safely under MASK63
        # s = (s*A + (tok+B)) mod 2^63
        inc = int(byte_pos) + 1
        for j in range(x.size(1)):
            tok = (x[:, j] + self._B + inc) & self._MASK63
            s = (s * self._A + tok) & self._MASK63
            s = (s ^ (s >> 29)) & self._MASK63

        return s

    @torch.no_grad()
    def _hash32_bytes(self, seed63: torch.Tensor) -> torch.Tensor:
        """
        seed63: [B] int64
        returns: h [B,256] int64, each element in [0..2^32-1]
        """
        # seed32 in [0..2^32-1]
        seed32 = (seed63 ^ (seed63 >> 31)) & 0xFFFFFFFF
        seed32 = seed32.to(torch.int64).unsqueeze(1)  # [B,1]

        b = self.bytes.unsqueeze(0)  # [1,256]
        h = (seed32 ^ b) & 0xFFFFFFFF

        # 32-bit mix (done in int64 with mask)
        h ^= (h >> 16)
        h = (h * 0x7FEB352D) & 0xFFFFFFFF
        h ^= (h >> 15)
        h = (h * 0x846CA68B) & 0xFFFFFFFF
        h ^= (h >> 16)
        return h  # [B,256]

    @torch.no_grad()
    def green_bytes_ids(self, ctx_ids: torch.Tensor, *, byte_pos: int = 0) -> torch.Tensor:
        """
        returns: [B,K] int64 in [0..255]
        """
        if ctx_ids.device != self.device:
            ctx_ids = ctx_ids.to(self.device)
        if ctx_ids.dim() == 1:
            ctx_ids = ctx_ids.unsqueeze(0)

        seed63 = self._seed63(ctx_ids, byte_pos=int(byte_pos))
        h = self._hash32_bytes(seed63)  # [B,256]

        # pick K smallest hashes as green bytes
        _, idx = torch.topk(h, k=self.K, dim=1, largest=False, sorted=False)  # [B,K]
        return idx.to(torch.int64)

    @torch.no_grad()
    def green_bytes_mask(self, ctx_ids: torch.Tensor, *, byte_pos: int = 0) -> torch.Tensor:
        """
        returns: [B,256] bool
        """
        if ctx_ids.device != self.device:
            ctx_ids = ctx_ids.to(self.device)
        if ctx_ids.dim() == 1:
            ctx_ids = ctx_ids.unsqueeze(0)

        green = self.green_bytes_ids(ctx_ids, byte_pos=int(byte_pos))  # [B,K]
        B = green.size(0)
        mask = torch.zeros((B, 256), device=self.device, dtype=torch.bool)
        mask.scatter_(1, green, True)
        return mask
