from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Set

import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase

from .token_bytes import TokenByteVocab


@dataclass
class ByteVocab:
    """
    Compact mapping of token_id -> byte sequence.
    All data is precomputed once so lookups are O(1) slices.
    """

    byte_data: Tensor
    offsets: Tensor  # len = vocab_size + 1
    first_bytes: Tensor  # len = vocab_size, stores first visible byte or -1
    end_token_mask: Tensor  # bool tensor marking EOS-like tokens

    def bytes_of(self, token_id: int) -> bytes:
        tid = int(token_id)
        if tid < 0 or tid >= self.size:
            return b""
        start = int(self.offsets[tid].item())
        end = int(self.offsets[tid + 1].item())
        if end <= start:
            return b""
        return self.byte_data[start:end].cpu().numpy().tobytes()

    def first_visible_byte(self, token_id: int) -> Tuple[str, Optional[int]]:
        tid = int(token_id)
        if tid < 0 or tid >= self.size:
            return ("EMPTY", None)
        if bool(self.end_token_mask[tid].item()):
            return ("EOS", None)
        start = int(self.offsets[tid].item())
        end = int(self.offsets[tid + 1].item())
        if end <= start:
            return ("EMPTY", None)
        value = int(self.first_bytes[tid].item())
        if value < 0:
            return ("EMPTY", None)
        return ("BYTE", value)

    @property
    def size(self) -> int:
        return int(self.offsets.shape[0] - 1)


def first_visible_byte_info(payload: bytes, piece: Optional[str]) -> Tuple[int, int]:
    if not payload:
        return -1, -1
    idx = 0
    prefix_mode = ""
    if piece:
        if piece.startswith("Ġ"):
            prefix_mode = "gpt2"
        elif piece.startswith("▁"):
            prefix_mode = "spm"
    if prefix_mode == "gpt2":
        while idx < len(payload) and payload[idx] == 0x20:
            idx += 1
    elif prefix_mode == "spm":
        marker = "▁".encode("utf-8")
        while idx + len(marker) <= len(payload) and payload[idx : idx + len(marker)] == marker:
            idx += len(marker)
    else:
        while idx < len(payload) and payload[idx] == 0x20:
            idx += 1
    while idx < len(payload) and (payload[idx] & 0xC0) == 0x80:
        idx += 1
    if idx >= len(payload):
        return -1, -1
    return int(payload[idx]), idx


def build_byte_vocab(
    tokenizer: PreTrainedTokenizerBase,
    *,
    device: Optional[torch.device] = None,
) -> ByteVocab:
    """
    Precompute the token->bytes table once using TokenByteVocab logic.
    This guarantees consistency across tokenizer types without calling
    to_visible_bytes repeatedly.
    """
    token_vocab = TokenByteVocab.from_tokenizer(tokenizer)
    vocab_size = token_vocab.vocab_size

    end_token_ids: Set[int] = set()
    for attr in ("eos_token_id", "sep_token_id", "pad_token_id"):
        tid = getattr(tokenizer, attr, None)
        if tid is None:
            continue
        if isinstance(tid, (list, tuple, set)):
            for item in tid:
                try:
                    end_token_ids.add(int(item))
                except Exception:
                    continue
        else:
            try:
                end_token_ids.add(int(tid))
            except Exception:
                continue

    pieces: List[Optional[str]]
    try:
        pieces = tokenizer.convert_ids_to_tokens(list(range(vocab_size)), skip_special_tokens=False)
    except Exception:
        pieces = [None] * vocab_size

    offsets = [0]
    chunks = []
    first_bytes = torch.full((vocab_size,), -1, dtype=torch.int16)
    end_mask = torch.zeros(vocab_size, dtype=torch.bool)
    for tid in range(vocab_size):
        payload = token_vocab.mapping.get(tid, b"")
        payload = payload if isinstance(payload, (bytes, bytearray)) else bytes(payload)
        # Replace special ByteLevel marker for leading space (Ġ) with actual ASCII space.
        if payload:
            payload = payload.replace(b"\xC4\xA0", b" ")
        piece = pieces[tid] if tid < len(pieces) else None
        first_val, _ = first_visible_byte_info(payload, piece)
        first_bytes[tid] = first_val
        if tid in end_token_ids:
            end_mask[tid] = True
        offsets.append(offsets[-1] + len(payload))
        if payload:
            chunks.append(torch.tensor(list(payload), dtype=torch.uint8))
    byte_data = (
        torch.cat(chunks)
        if len(chunks) > 1
        else (chunks[0] if chunks else torch.empty(0, dtype=torch.uint8))
    )
    offsets_tensor = torch.tensor(offsets, dtype=torch.int32)
    if device is not None:
        byte_data = byte_data.to(device)
        offsets_tensor = offsets_tensor.to(device)
        first_bytes = first_bytes.to(device)
        end_mask = end_mask.to(device)
    return ByteVocab(
        byte_data=byte_data,
        offsets=offsets_tensor,
        first_bytes=first_bytes,
        end_token_mask=end_mask,
    )


__all__ = ["ByteVocab", "build_byte_vocab", "first_visible_byte_info"]
