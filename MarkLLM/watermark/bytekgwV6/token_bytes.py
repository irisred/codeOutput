from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import Tensor

END_BYTE = 256  # sentinel for “no byte” (token shorter than requested pos)


def _visible_bytes(b: bytes, skip_markers: bool = True) -> bytes:
    """
    Extract visible bytes from a decoded token string.
    For now we treat all bytes from .encode as visible and optionally drop common markers.
    """
    if not b:
        return b
    if skip_markers:
        # drop common marker bytes (e.g., space/Ġ encoded) at the beginning
        while b and b[0] in (ord(" "), 0xC4, 0xC4 | 0x80):  # heuristic; keep simple
            b = b[1:]
    return b


class TokenByteVocabV6:
    """
    Minimal byte vocab helper for ByteKGWv6.

    Stores visible bytes for each token and provides:
      - bytepos_tensor(device, pos): [V] int16/END_BYTE at given position.
      - bytes_len_tensor(device): [V] lengths.
      - first_n_id(device, n): [V] int64, base-256 encoding of first n bytes (pad with END_BYTE).
    """

    def __init__(self, *, vocab_size: int, byte_data: Tensor, offsets: Tensor) -> None:
        self.vocab_size = int(vocab_size)
        self.byte_data = byte_data
        self.offsets = offsets
        self._pos_cache: Dict[Tuple[str, int], Tensor] = {}
        self._firstn_cache: Dict[Tuple[str, int], Tensor] = {}

    @classmethod
    def from_tokenizer(cls, tokenizer, *, skip_markers: bool = True) -> "TokenByteVocabV6":
        V = len(tokenizer)
        offsets = [0]
        chunks = []
        for tid in range(V):
            try:
                s = tokenizer.decode([tid], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            except Exception:
                s = ""
            vb = _visible_bytes(s.encode("utf-8", errors="ignore"), skip_markers=skip_markers)
            offsets.append(offsets[-1] + len(vb))
            if vb:
                chunks.append(torch.tensor(list(vb), dtype=torch.uint8))
        byte_data = torch.cat(chunks, dim=0) if chunks else torch.empty((0,), dtype=torch.uint8)
        offsets_t = torch.tensor(offsets, dtype=torch.int32)
        return cls(vocab_size=V, byte_data=byte_data, offsets=offsets_t)

    def to(self, device: torch.device) -> "TokenByteVocabV6":
        self.byte_data = self.byte_data.to(device, non_blocking=True)
        self.offsets = self.offsets.to(device, non_blocking=True)
        self._pos_cache.clear()
        self._firstn_cache.clear()
        return self

    def bytepos_tensor(self, device: torch.device, pos: int) -> Tensor:
        key = (str(device), int(pos))
        if key in self._pos_cache:
            return self._pos_cache[key]
        if self.byte_data.device != device or self.offsets.device != device:
            raise ValueError("Move vocab to target device first via .to(device).")

        V = self.vocab_size
        pos = int(pos)
        starts = self.offsets[:-1].to(torch.int64)
        ends = self.offsets[1:].to(torch.int64)
        lens = ends - starts

        out = torch.full((V,), END_BYTE, device=device, dtype=torch.int16)
        valid = lens > pos
        if valid.any():
            idx = (starts[valid] + pos).to(torch.int64)
            out[valid] = self.byte_data.index_select(0, idx).to(torch.int16)

        self._pos_cache[key] = out
        return out

    def bytes_len_tensor(self, device: torch.device) -> Tensor:
        if self.offsets.device != device:
            raise ValueError("Move vocab to target device first via .to(device).")
        return (self.offsets[1:] - self.offsets[:-1]).to(torch.int32)

    def first_n_id(self, device: torch.device, n: int) -> Tensor:
        """
        Base-256 encode first n visible bytes per token into int64.
        Pads missing bytes with END_BYTE (256).
        """
        key = (str(device), int(n))
        if key in self._firstn_cache:
            return self._firstn_cache[key]
        if self.byte_data.device != device or self.offsets.device != device:
            raise ValueError("Move vocab to target device first via .to(device).")

        n = max(1, int(n))
        V = self.vocab_size
        starts = self.offsets[:-1].to(torch.int64)
        ends = self.offsets[1:].to(torch.int64)
        lens = ends - starts

        out = torch.zeros((V,), device=device, dtype=torch.int64)
        base = 1
        for i in range(n):
            bpos = self.bytepos_tensor(device, i).to(torch.int64)  # [V]
            out += bpos * base
            base *= (END_BYTE + 1)

        self._firstn_cache[key] = out
        return out
