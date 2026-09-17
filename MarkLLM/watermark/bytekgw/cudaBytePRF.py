# watermark/bytekgw/cudaBytePRF.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Union

import torch


@dataclass
class CudaBytePRFConfig:
    prefix_length: int
    hash_key: int
    green_size: int = 128


class CudaBytePRF:
    """
    CUDA-only PRF that partitions bytes {0..255} into exactly 128 green + 128 red each step.

    This implementation avoids CUDA uint64 entirely (many builds don't fully support it).
    We use int64 with a 63-bit mask so all values remain non-negative and right-shifts behave
    like logical shifts.

    Seed is derived from token_ids (input_ids), NOT from firstbyte.
    Green bytes are the 128 smallest PRF values over bytes 0..255.
    """

    def __init__(self, config: CudaBytePRFConfig, device: Union[str, torch.device]) -> None:
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("CudaBytePRF is CUDA-only: device must be CUDA (e.g. cuda:0).")

        self.config = config
        if config.prefix_length <= 0:
            raise ValueError("prefix_length must be > 0.")
        if int(config.green_size) != 128:
            raise ValueError("This implementation is fixed to green_size=128 (128/128 split).")

        self.prefix_length = int(config.prefix_length)
        self.green_size = int(config.green_size)

        # Keep everything in [0, 2^63-1] to avoid negative int64 values.
        self.MASK: int = (1 << 63) - 1
        self.MASK_T = torch.tensor(self.MASK, device=self.device, dtype=torch.int64)

        # Keyed hashing
        hk = int(config.hash_key) & self.MASK
        self.hash_key = torch.tensor(hk, device=self.device, dtype=torch.int64)

        # Bytes 0..255 (int64 for math, and uint8 view for output)
        self.bytes_i64 = torch.arange(256, device=self.device, dtype=torch.int64)  # [256]
        self.bytes_u8 = self.bytes_i64.to(torch.uint8)

        # 63-bit-safe constants (< 2^63). Chosen to mix well; feel free to keep as-is.
        self.C_POS = torch.tensor(0x1BD11BDAA9FC1A22, device=self.device, dtype=torch.int64)  # positional perturb
        self.A = torch.tensor(0x165667B19E3779F9, device=self.device, dtype=torch.int64)
        self.B = torch.tensor(0x27D4EB2F165667C5, device=self.device, dtype=torch.int64)

        self.KB1 = torch.tensor(0x2545F4914F6CDD1D, device=self.device, dtype=torch.int64)
        self.KB2 = torch.tensor(0x369DEA0F31A53F85, device=self.device, dtype=torch.int64)

    # -----------------------------
    # Public API
    # -----------------------------

    @torch.no_grad()
    def seed64_from_input_ids(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """
        input_ids: [T] or [B,T] on CUDA
        returns:   [B] int64 (non-negative, < 2^63) or scalar if input was 1D
        """
        x, squeezed = self._ensure_2d(input_ids)
        self._check_cuda(x)

        B, T = x.shape
        start = max(0, T - self.prefix_length)
        w = x[:, start:]  # [B, <=L]
        if w.numel() == 0:
            seed = torch.zeros((B,), device=self.device, dtype=torch.int64)
            return seed[0] if squeezed else seed

        # Mask token ids to 63-bit non-negative
        w_i64 = w.to(torch.int64) & self.MASK_T

        # Positional perturbation: token + C_POS*(pos+1)
        pos = torch.arange(w_i64.shape[1], device=self.device, dtype=torch.int64)  # [<=L]
        x0 = (w_i64 + (pos + 1) * self.C_POS) & self.MASK_T

        # Mix each, then sum
        mixed = self._mix63(x0)  # [B,<=L]
        s = mixed.sum(dim=1, dtype=torch.int64) & self.MASK_T  # [B]

        seed = self._mix63((s ^ self.hash_key) & self.MASK_T)  # [B]
        return seed[0] if squeezed else seed

    @torch.no_grad()
    def green_bytes_ids(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """
        returns: [B,128] uint8 (or [128] if input was 1D)
        """
        x, squeezed = self._ensure_2d(input_ids)
        self._check_cuda(x)

        seed = self.seed64_from_input_ids(x)  # [B] int64 in [0..2^63-1]

        # r(seed, byte) for all bytes: [B,256]
        # keep everything masked to 63-bit
        byte_term = (self.bytes_i64[None, :] * self.KB1 + self.KB2) & self.MASK_T
        r = self._mix63((seed[:, None] ^ byte_term) & self.MASK_T)  # [B,256]

        # pick 128 smallest
        _, idx = torch.topk(r, k=self.green_size, dim=1, largest=False, sorted=False)  # [B,128]
        green_ids = idx.to(torch.uint8)

        return green_ids[0] if squeezed else green_ids

    @torch.no_grad()
    def green_bytes_mask(self, input_ids: torch.LongTensor) -> torch.BoolTensor:
        """
        returns: [B,256] bool (or [256] if input was 1D)
        """
        x, squeezed = self._ensure_2d(input_ids)
        self._check_cuda(x)

        green_ids = self.green_bytes_ids(x)  # [B,128] uint8
        if green_ids.dim() == 1:
            mask = torch.zeros((256,), device=self.device, dtype=torch.bool)
            mask.scatter_(0, green_ids.to(torch.int64), True)
            return mask

        B = green_ids.shape[0]
        mask = torch.zeros((B, 256), device=self.device, dtype=torch.bool)
        mask.scatter_(1, green_ids.to(torch.int64), True)
        return mask[0] if squeezed else mask

    # -----------------------------
    # Internal helpers
    # -----------------------------

    def _ensure_2d(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        if input_ids.dim() == 1:
            return input_ids.unsqueeze(0), True
        if input_ids.dim() != 2:
            raise ValueError("input_ids must have shape [T] or [B,T].")
        return input_ids, False

    def _check_cuda(self, x: torch.Tensor) -> None:
        if not torch.is_tensor(x):
            raise TypeError("input_ids must be a torch.Tensor.")
        if x.device.type != "cuda":
            raise ValueError("input_ids must be on CUDA.")
        if x.device != self.device:
            raise ValueError(f"input_ids must be on the same device as PRF ({self.device}).")

    @torch.no_grad()
    def _mix63(self, x: torch.Tensor) -> torch.Tensor:
        """
        63-bit mixing function (int64 only, non-negative domain).
        Ensures all results are in [0, 2^63-1].
        """
        x = x.to(torch.int64) & self.MASK_T

        x = (x ^ (x >> 30)) & self.MASK_T
        x = (x * self.A) & self.MASK_T
        x = (x ^ (x >> 27)) & self.MASK_T
        x = (x * self.B) & self.MASK_T
        x = (x ^ (x >> 31)) & self.MASK_T
        return x
