from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import torch


@dataclass(frozen=True)
class CudaBytePRFConfig:
    hash_key: int
    prefix_length: int = 1
    green_size: int = 128  # 256 * gamma (gamma=0.5 -> 128)


class CudaBytePRF:
    """
    CUDA-only PRF over byte domain [0..255].

    Public API (keyword-only for prefix_bytes/byte_pos):
      - green_bytes_mask(input_ids, *, prefix_bytes=None, byte_pos=0) -> [B,256] bool
      - green_bytes_ids (same args) -> [B,K] uint8
      - seed63(...) -> [B] int64
    """

    MASK_T = (1 << 63) - 1  # keep in signed int64 safe range

    # splitmix-like constants (stored as Python ints; will be converted to signed int64 tensors per device)
    A = 0xBF58476D1CE4E5B9
    B = 0x94D049BB133111EB

    C_POS = 0x9E3779B97F4A7C15
    C_PFX = 0xD6E8FEB86659FD93
    C_BP  = 0xA5A35625D3C2A3C1

    KB1 = 0xD6E8FEB86659FD93
    KB2 = 0xA5A35625D3C2A3C1

    def __init__(self, cfg: Union[CudaBytePRFConfig, dict], device: Union[str, torch.device] = "cuda") -> None:
        if isinstance(cfg, dict):
            cfg = CudaBytePRFConfig(
                hash_key=int(cfg["hash_key"]),
                prefix_length=int(cfg.get("prefix_length", 1)),
                green_size=int(cfg.get("green_size", 128)),
            )

        self.hash_key = int(cfg.hash_key)
        self.prefix_length = max(0, int(cfg.prefix_length))
        self.green_size = int(cfg.green_size)
        self.green_size = max(0, min(256, self.green_size))

        self._init_device = torch.device(device)
        self._const_cache: Dict[torch.device, Dict[str, torch.Tensor]] = {}

    # -----------------------------
    # public API
    # -----------------------------
    @torch.no_grad()
    def green_bytes_mask(
        self,
        input_ids: torch.LongTensor,
        *,
        prefix_bytes: Optional[Union[bytes, bytearray, torch.Tensor]] = None,
        byte_pos: Union[int, torch.Tensor] = 0,
    ) -> torch.BoolTensor:
        x, squeezed = self._ensure_2d(input_ids)
        self._check_cuda(x)
        dev = x.device

        green = self.green_bytes_ids(x, prefix_bytes=prefix_bytes, byte_pos=byte_pos)

        if green.dim() == 1:
            m = torch.zeros((256,), device=dev, dtype=torch.bool)
            if green.numel() > 0:
                m.scatter_(0, green.to(torch.int64), True)
            return m

        B = green.shape[0]
        m = torch.zeros((B, 256), device=dev, dtype=torch.bool)
        if green.numel() > 0:
            m.scatter_(1, green.to(torch.int64), True)
        return m[0] if squeezed else m

    @torch.no_grad()
    def green_bytes_ids(
        self,
        input_ids: torch.LongTensor,
        *,
        prefix_bytes: Optional[Union[bytes, bytearray, torch.Tensor]] = None,
        byte_pos: Union[int, torch.Tensor] = 0,
    ) -> torch.Tensor:
        x, squeezed = self._ensure_2d(input_ids)
        self._check_cuda(x)
        dev = x.device
        c = self._get_consts(dev)

        seed = self.seed63(x, prefix_bytes=prefix_bytes, byte_pos=byte_pos)  # [B] int64

        k = int(self.green_size)
        if k <= 0:
            out = torch.empty((x.shape[0], 0), device=dev, dtype=torch.uint8)
            return out[0] if squeezed else out

        # r(seed, byte) for all bytes -> [B,256]
        r = self._mix63((seed[:, None] ^ c["BYTE_TERM"][None, :]) & c["MASK"], c)  # [B,256]

        _, idx = torch.topk(r, k=k, dim=1, largest=False, sorted=False)
        out = idx.to(torch.uint8)
        return out[0] if squeezed else out

    @torch.no_grad()
    def seed63(
        self,
        input_ids: torch.LongTensor,
        *,
        prefix_bytes: Optional[Union[bytes, bytearray, torch.Tensor]] = None,
        byte_pos: Union[int, torch.Tensor] = 0,
    ) -> torch.Tensor:
        x, squeezed = self._ensure_2d(input_ids)
        self._check_cuda(x)
        dev = x.device
        c = self._get_consts(dev)

        B, T = x.shape

        # ctx window
        if self.prefix_length <= 0:
            w = x[:, :0]
        else:
            start = max(0, T - self.prefix_length)
            w = x[:, start:]  # [B,<=L]

        # ctx mix
        if w.numel() == 0:
            ctx_mix = torch.zeros((B,), device=dev, dtype=torch.int64)
        else:
            w_i64 = w.to(torch.int64) & c["MASK"]
            L = w_i64.shape[1]
            pos = c["POS_CACHE"][:L]
            z0 = (w_i64 + (pos + 1) * c["C_POS"]) & c["MASK"]
            ctx_mix = self._mix63(z0, c).sum(dim=1, dtype=torch.int64) & c["MASK"]

        # prefix bytes mix
        pfx_mix = torch.zeros((B,), device=dev, dtype=torch.int64)
        if prefix_bytes is not None:
            pb = self._normalize_prefix_bytes(prefix_bytes, dev)  # [P] or [B,P]
            if pb.dim() == 1:
                pb = pb.unsqueeze(0)
            if pb.shape[0] == 1 and B > 1:
                pb = pb.expand(B, -1)
            if pb.shape[0] != B:
                raise ValueError(f"prefix_bytes batch mismatch: got {pb.shape[0]} vs B={B}")

            pb = pb.to(torch.int64)
            pb = torch.where(pb < 0, torch.zeros_like(pb), pb) & 0xFF

            P = pb.shape[1]
            ppos = c["POS_CACHE"][:P]
            y0 = (pb + (ppos + 1) * c["C_PFX"]) & c["MASK"]
            pfx_mix = self._mix63(y0, c).sum(dim=1, dtype=torch.int64) & c["MASK"]

        # byte_pos mix
        bp = self._normalize_byte_pos(byte_pos, B=B, dev=dev) & c["MASK"]
        pos_mix = ((bp + 1) * c["C_BP"]) & c["MASK"]

        seed = self._mix63((ctx_mix ^ pfx_mix ^ pos_mix ^ c["HASH_KEY"]) & c["MASK"], c)
        return seed[0] if squeezed else seed

    # -----------------------------
    # internals
    # -----------------------------
    @staticmethod
    def _u64_to_i64_scalar(x: int) -> int:
        """Map 0..2^64-1 to signed int64 range, preserving low 64-bit pattern."""
        x &= 0xFFFFFFFFFFFFFFFF
        if x >= 0x8000000000000000:
            x -= 0x10000000000000000
        return int(x)

    def _t_i64(self, x: int, dev: torch.device) -> torch.Tensor:
        """Create int64 tensor from possibly >2^63-1 Python int safely."""
        return torch.tensor(self._u64_to_i64_scalar(x), device=dev, dtype=torch.int64)

    def _get_consts(self, dev: torch.device) -> Dict[str, torch.Tensor]:
        cached = self._const_cache.get(dev)
        if cached is not None:
            return cached

        mask = torch.tensor(self.MASK_T, device=dev, dtype=torch.int64)

        A = self._t_i64(self.A, dev)
        B = self._t_i64(self.B, dev)

        C_POS = self._t_i64(self.C_POS, dev)
        C_PFX = self._t_i64(self.C_PFX, dev)
        C_BP  = self._t_i64(self.C_BP, dev)

        KB1 = self._t_i64(self.KB1, dev)
        KB2 = self._t_i64(self.KB2, dev)

        bytes_i64 = torch.arange(256, device=dev, dtype=torch.int64)
        byte_term = (bytes_i64 * KB1 + KB2) & mask  # [256] cached

        POS_CACHE_LEN = 1024
        pos_cache = torch.arange(POS_CACHE_LEN, device=dev, dtype=torch.int64)

        t = {
            "MASK": mask,
            "A": A,
            "B": B,
            "C_POS": C_POS,
            "C_PFX": C_PFX,
            "C_BP": C_BP,
            "KB1": KB1,
            "KB2": KB2,
            "HASH_KEY": (torch.tensor(int(self.hash_key), device=dev, dtype=torch.int64) & mask),
            "BYTE_TERM": byte_term,
            "POS_CACHE": pos_cache,
        }
        self._const_cache[dev] = t
        return t

    @staticmethod
    def _ensure_2d(input_ids: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        if input_ids.dim() == 1:
            return input_ids.unsqueeze(0), True
        if input_ids.dim() != 2:
            raise ValueError("input_ids must have shape [T] or [B,T].")
        return input_ids, False

    @staticmethod
    def _check_cuda(x: torch.Tensor) -> None:
        if x.device.type != "cuda":
            raise ValueError("input_ids must be on CUDA.")

    @staticmethod
    def _normalize_prefix_bytes(prefix_bytes: Union[bytes, bytearray, torch.Tensor], dev: torch.device) -> torch.Tensor:
        if isinstance(prefix_bytes, (bytes, bytearray)):
            if len(prefix_bytes) == 0:
                return torch.empty((0,), device=dev, dtype=torch.int64)
            return torch.tensor(list(prefix_bytes), device=dev, dtype=torch.int64)
        if torch.is_tensor(prefix_bytes):
            return prefix_bytes.to(device=dev, dtype=torch.int64)
        raise TypeError("prefix_bytes must be bytes/bytearray/torch.Tensor.")

    @staticmethod
    def _normalize_byte_pos(byte_pos: Union[int, torch.Tensor], *, B: int, dev: torch.device) -> torch.Tensor:
        if torch.is_tensor(byte_pos):
            bp = byte_pos.to(device=dev, dtype=torch.int64)
            if bp.dim() == 0:
                return bp.expand(B)
            if bp.dim() == 1:
                if bp.numel() == 1:
                    return bp.expand(B)
                if bp.numel() == B:
                    return bp
                raise ValueError(f"byte_pos must be scalar, [1], or [B]; got shape {tuple(bp.shape)} with B={B}")
            raise ValueError("byte_pos tensor must be scalar or 1D [B].")
        return torch.full((B,), int(byte_pos), device=dev, dtype=torch.int64)

    @torch.no_grad()
    def _mix63(self, x: torch.Tensor, c: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = x.to(torch.int64) & c["MASK"]
        x = (x ^ (x >> 30)) & c["MASK"]
        x = (x * c["A"]) & c["MASK"]
        x = (x ^ (x >> 27)) & c["MASK"]
        x = (x * c["B"]) & c["MASK"]
        x = (x ^ (x >> 31)) & c["MASK"]
        return x


__all__ = ["CudaBytePRF", "CudaBytePRFConfig"]
