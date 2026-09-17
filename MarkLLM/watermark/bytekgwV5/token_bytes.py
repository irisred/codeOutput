# MarkLLM/watermark/bytekgwV5/token_bytes.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase

# 257-way byte tree: 0..255 are real bytes, 256 means "END / no byte at this pos"
END_BYTE: int = 256


def _skip_prefix_markers(payload: bytes) -> int:
    """
    Skip leading marker bytes:
      - ASCII spaces 0x20
      - UTF-8 for 'Ġ' (C4 A0)
      - UTF-8 for '▁' (E2 96 81)
    Repeat until stable, also skipping spaces between markers.
    """
    if not payload:
        return 0
    i = 0
    n = len(payload)
    while i < n and payload[i] == 0x20:
        i += 1

    marker_gpt2 = "Ġ".encode("utf-8")  # b"\xC4\xA0"
    marker_spm = "▁".encode("utf-8")   # b"\xE2\x96\x81"

    changed = True
    while changed:
        changed = False
        if i + len(marker_gpt2) <= n and payload[i:i + len(marker_gpt2)] == marker_gpt2:
            i += len(marker_gpt2)
            changed = True
        if i + len(marker_spm) <= n and payload[i:i + len(marker_spm)] == marker_spm:
            i += len(marker_spm)
            changed = True
        while i < n and payload[i] == 0x20:
            i += 1
    return i


def _visible_bytes(payload: bytes, *, skip_markers: bool) -> bytes:
    if not payload:
        return b""
    i = _skip_prefix_markers(payload) if skip_markers else 0
    if i >= len(payload):
        return b""
    return payload[i:]


@dataclass
class TokenByteVocab:
    """
    Compact token_id -> visible UTF-8 bytes table + GPU-friendly views.

    Key points for v5:
      - V = len(tokenizer) (NOT tokenizer.vocab_size) to avoid 128000 vs 128256 mismatch.
      - Visible bytes are derived from tokenizer.decode([tid]) then utf-8 encode.
      - 257-way byte tree uses END_BYTE=256 sentinel for "no byte at this pos".
    """
    vocab_size: int
    byte_data: Tensor          # uint8 flat
    offsets: Tensor            # int32 len=vocab_size+1
    _pos_cache: Dict[Tuple[str, int], Tensor]

    @classmethod
    def from_tokenizer(
        cls,
        tokenizer: PreTrainedTokenizerBase,
        *,
        skip_markers: bool = True,
    ) -> "TokenByteVocab":
        # IMPORTANT: use len(tokenizer) as the true V
        V = int(len(tokenizer))

        offsets = [0]
        chunks = []

        for tid in range(V):
            try:
                s = tokenizer.decode([tid], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            except Exception:
                s = ""
            payload = s.encode("utf-8", errors="ignore") if s else b""
            vb = _visible_bytes(payload, skip_markers=skip_markers)

            offsets.append(offsets[-1] + len(vb))
            if vb:
                chunks.append(torch.tensor(list(vb), dtype=torch.uint8))

        byte_data = (
            torch.cat(chunks, dim=0) if len(chunks) > 1
            else (chunks[0] if chunks else torch.empty((0,), dtype=torch.uint8))
        )
        offsets_t = torch.tensor(offsets, dtype=torch.int32)

        return cls(vocab_size=V, byte_data=byte_data, offsets=offsets_t, _pos_cache={})

    def to(self, device: torch.device) -> "TokenByteVocab":
        self.byte_data = self.byte_data.to(device, non_blocking=True)
        self.offsets = self.offsets.to(device, non_blocking=True)
        self._pos_cache = {}
        return self

    def bytepos_tensor(self, device: torch.device, pos: int) -> Tensor:
        """
        Return [V] int16 in [0..256], where:
          - 0..255 : real byte at index `pos`
          - 256    : END (no byte at this pos / token has shorter visible bytes)
        Cached per (device,pos).
        """
        key = (str(device), int(pos))
        if key in self._pos_cache:
            return self._pos_cache[key]

        if self.byte_data.device != device or self.offsets.device != device:
            raise ValueError("TokenByteVocab tensors must be moved to target device via .to(device).")

        V = self.vocab_size
        pos = int(pos)

        starts = self.offsets[:-1].to(torch.int64)  # [V]
        ends = self.offsets[1:].to(torch.int64)     # [V]
        lens = ends - starts                        # [V]

        out = torch.full((V,), END_BYTE, device=device, dtype=torch.int16)
        valid = lens > pos
        if valid.any():
            idx = (starts[valid] + pos).to(torch.int64)
            out[valid] = self.byte_data.index_select(0, idx).to(torch.int16)

        self._pos_cache[key] = out
        return out

    def bytes_len_tensor(self, device: torch.device) -> Tensor:
        """Return [V] int32 lengths of visible bytes."""
        if self.offsets.device != device:
            raise ValueError("Move vocab to device first.")
        return (self.offsets[1:] - self.offsets[:-1]).to(torch.int32)
