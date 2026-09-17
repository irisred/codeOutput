from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase


def _skip_prefix_markers(payload: bytes) -> int:
    """
    Skip leading marker bytes:
      - ASCII spaces 0x20
      - UTF-8 for 'Ġ' (C4 A0)
      - UTF-8 for '▁' (E2 96 81)
    Return new index.
    """
    if not payload:
        return 0
    i = 0
    # skip spaces
    while i < len(payload) and payload[i] == 0x20:
        i += 1

    # skip repeated marker bytes if present
    marker_gpt2 = "Ġ".encode("utf-8")  # b"\xC4\xA0"
    marker_spm = "▁".encode("utf-8")   # b"\xE2\x96\x81"

    changed = True
    while changed:
        changed = False
        if i + len(marker_gpt2) <= len(payload) and payload[i : i + len(marker_gpt2)] == marker_gpt2:
            i += len(marker_gpt2)
            changed = True
        if i + len(marker_spm) <= len(payload) and payload[i : i + len(marker_spm)] == marker_spm:
            i += len(marker_spm)
            changed = True
        # skip spaces again
        while i < len(payload) and payload[i] == 0x20:
            i += 1
    return i


def first_visible_byte(payload: bytes) -> int:
    """
    Unified rule:
      - skip spaces / Ġ / ▁
      - if first remaining byte is UTF-8 continuation (10xxxxxx), return -1
      - else return that byte (0..255), or -1 if empty
    """
    if not payload:
        return -1
    i = _skip_prefix_markers(payload)
    if i >= len(payload):
        return -1
    b = payload[i]
    # UTF-8 continuation byte => ignore token for watermark/detection
    if (b & 0xC0) == 0x80:
        return -1
    return int(b)


@dataclass
class TokenByteVocab:
    """
    Compact token_id -> bytes table + GPU-friendly views.
    bytes are derived from tokenizer.decode([tid]) then utf-8 encode, so it matches output stream.
    """
    vocab_size: int
    _byte_data: Tensor          # uint8 flat
    _offsets: Tensor            # int32 len=vocab_size+1
    _firstbyte_cache: dict      # device->tensor cache

    @classmethod
    def from_tokenizer(cls, tokenizer: PreTrainedTokenizerBase) -> "TokenByteVocab":
        vocab_size = int(getattr(tokenizer, "vocab_size", None) or len(tokenizer))
        offsets = [0]
        chunks = []
        for tid in range(vocab_size):
            try:
                s = tokenizer.decode([tid], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            except Exception:
                s = ""
            payload = s.encode("utf-8") if s else b""
            offsets.append(offsets[-1] + len(payload))
            if payload:
                chunks.append(torch.tensor(list(payload), dtype=torch.uint8))

        byte_data = (
            torch.cat(chunks)
            if len(chunks) > 1
            else (chunks[0] if chunks else torch.empty(0, dtype=torch.uint8))
        )
        offsets_t = torch.tensor(offsets, dtype=torch.int32)
        return cls(
            vocab_size=vocab_size,
            _byte_data=byte_data,
            _offsets=offsets_t,
            _firstbyte_cache={},
        )

    def _ensure_views(self):
        # Kept for compatibility with your older code style.
        return

    def firstbyte_tensor(self, device: torch.device, *, max_scan_iters: int = 32) -> Tensor:
        """
        Returns [V] int16 tensor on device, values 0..255 or -1.
        max_scan_iters is kept for API compatibility; we do CPU precompute then move to GPU.
        """
        dev = torch.device(device)
        key = str(dev)
        if key in self._firstbyte_cache:
            t = self._firstbyte_cache[key]
            if t.device == dev:
                return t

        # Precompute on CPU once (O(V) decode was already done); now just scan bytes table.
        byte_data = self._byte_data.cpu()
        offsets = self._offsets.cpu()
        fb = torch.full((self.vocab_size,), -1, dtype=torch.int16)

        for tid in range(self.vocab_size):
            s = int(offsets[tid].item())
            e = int(offsets[tid + 1].item())
            if e <= s:
                continue
            payload = byte_data[s:e].numpy().tobytes()
            fb[tid] = int(first_visible_byte(payload))

        fb = fb.to(dev, non_blocking=True)
        self._firstbyte_cache[key] = fb
        return fb

    def byte_views(self, device: torch.device) -> tuple[Tensor, Tensor]:
        """
        Returns (_byte_data, _offsets) moved to device.
        """
        dev = torch.device(device)
        return (
            self._byte_data.to(dev, non_blocking=True),
            self._offsets.to(dev, non_blocking=True),
        )
