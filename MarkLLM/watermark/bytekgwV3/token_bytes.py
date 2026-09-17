# MarkLLM/watermark/bytekgwV2/token_bytes.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase


def _first_visible_index(payload: bytes) -> int:
    """
    Skip leading spaces (0x20), then skip any UTF-8 continuation bytes at start (0x80..0xBF).
    If nothing left, return -1.
    """
    if not payload:
        return -1
    i = 0
    n = len(payload)
    while i < n and payload[i] == 0x20:
        i += 1
    while i < n and (payload[i] & 0xC0) == 0x80:
        i += 1
    return -1 if i >= n else i


@dataclass
class TokenByteVocab:
    """
    Stores 'visible bytes' per token_id in a compact (byte_data, offsets) representation.
    visible bytes are defined as: convert_tokens_to_string([token]) -> utf8 bytes,
    then drop leading spaces and drop leading utf8 continuation bytes.
    If no valid start remains, the token's visible bytes are empty (invalid for watermark/detect).
    """
    vocab_size: int
    byte_data: Tensor        # uint8 [N]
    offsets: Tensor          # int32 [V+1], offsets[i]: start index in byte_data
    # cache for per-pos lookup tables on a device
    _pos_cache: Dict[Tuple[str, int], Tensor]

    @staticmethod
    def from_tokenizer(tokenizer: PreTrainedTokenizerBase, device: Optional[torch.device] = None) -> "TokenByteVocab":
        V = int(tokenizer.vocab_size)

        # get pieces (tokens)
        try:
            pieces: List[str] = tokenizer.convert_ids_to_tokens(list(range(V)), skip_special_tokens=False)
        except Exception:
            pieces = [str(i) for i in range(V)]

        offsets: List[int] = [0]
        chunks: List[Tensor] = []

        for tid in range(V):
            tok = pieces[tid]

            # IMPORTANT: use tokenizer's own conversion to surface string (handles Ġ/▁, byte-level, etc.)
            try:
                s = tokenizer.convert_tokens_to_string([tok])
            except Exception:
                s = tok

            payload = s.encode("utf-8", errors="ignore")

            j = _first_visible_index(payload)
            if j < 0:
                visible = b""
            else:
                visible = payload[j:]

            offsets.append(offsets[-1] + len(visible))
            if visible:
                chunks.append(torch.tensor(list(visible), dtype=torch.uint8))

        byte_data = (
            torch.cat(chunks, dim=0) if len(chunks) > 1
            else (chunks[0] if chunks else torch.empty((0,), dtype=torch.uint8))
        )
        offsets_t = torch.tensor(offsets, dtype=torch.int32)

        if device is not None:
            byte_data = byte_data.to(device)
            offsets_t = offsets_t.to(device)

        return TokenByteVocab(
            vocab_size=V,
            byte_data=byte_data,
            offsets=offsets_t,
            _pos_cache={},
        )

    def to(self, device: torch.device) -> "TokenByteVocab":
        self.byte_data = self.byte_data.to(device)
        self.offsets = self.offsets.to(device)
        # clear cache because device changes
        self._pos_cache = {}
        return self

    def bytepos_tensor(self, device: torch.device, pos: int) -> Tensor:
        """
        Return a [V] int16 tensor where out[tid] is the visible byte at index `pos`,
        or -1 if pos out of range or token invalid/empty.
        Cached per (device,pos).
        """
        key = (str(device), int(pos))
        if key in self._pos_cache:
            return self._pos_cache[key]

        if self.byte_data.device != device:
            raise ValueError("TokenByteVocab.byte_data must be moved to the target device via .to(device).")

        V = self.vocab_size
        pos = int(pos)

        starts = self.offsets[:-1].to(torch.int64)         # [V]
        ends = self.offsets[1:].to(torch.int64)            # [V]
        lens = ends - starts                               # [V]

        out = torch.full((V,), -1, device=device, dtype=torch.int16)
        valid = lens > pos
        if valid.any():
            idx = (starts[valid] + pos).to(torch.int64)    # indices in byte_data
            out[valid] = self.byte_data.index_select(0, idx).to(torch.int16)

        self._pos_cache[key] = out
        return out

    def byte_at(self, token_ids: Tensor, pos: int) -> Tensor:
        """
        token_ids: [N] on same device as byte_data
        returns: int16 [N], byte value 0..255 or -1
        """
        table = self.bytepos_tensor(self.byte_data.device, int(pos))
        return table[token_ids]
