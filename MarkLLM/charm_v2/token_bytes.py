from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

import torch


def _build_byte_level_decoder() -> Dict[str, int]:
    """
    Reproduce HF byte-level BPE encoder/decoder tables so we can translate
    pieces like 'Ġ' or 'Â' back to their original byte values.
    Reference: https://github.com/openai/gpt-2/blob/master/src/encoder.py
    """
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    cs = [chr(c) for c in cs]
    byte_encoder = dict(zip(bs, cs))
    return {v: k for k, v in byte_encoder.items()}


_BYTE_LEVEL_DECODER = _build_byte_level_decoder()


def _piece_to_bytes(piece: str) -> bytes:
    """
    Convert a tokenizer piece into bytes, preferring byte-level reconstruction.
    Falls back to UTF-8 encode if the piece contains unknown symbols.
    """
    if not piece:
        return b""
    try:
        return bytes(_BYTE_LEVEL_DECODER[ch] for ch in piece)
    except KeyError:
        return piece.encode("utf-8", errors="ignore")


@dataclass
class TokenByteVocab:
    """
    Bidirectional mapping between token IDs and byte sequences.
    Precomputes contiguous storage so lookups avoid runtime decode/encode.
    """

    mapping: Dict[int, bytes]
    vocab_size: int
    _byte_data: torch.Tensor = field(init=False, repr=False)
    _offsets: torch.Tensor = field(init=False, repr=False)
    _dirty: bool = field(init=False, default=True, repr=False)

    def __post_init__(self) -> None:
        self._byte_data = torch.empty(0, dtype=torch.uint8)
        self._offsets = torch.zeros(self.vocab_size + 1, dtype=torch.int32)
        self._ensure_views()

    @classmethod
    def from_tokenizer(
        cls,
        tokenizer,
        *,
        end_token_ids: Optional[Iterable[int]] = None,
        override_token_bytes: Optional[Dict[int, bytes]] = None,
    ) -> "TokenByteVocab":
        vocab_map = tokenizer.get_vocab()
        if not vocab_map:
            raise ValueError("tokenizer.get_vocab() returned an empty mapping.")
        mapping: Dict[int, bytes] = {}
        max_id = -1
        end_ids: set[int] = set()
        if end_token_ids is not None:
            for tid in end_token_ids:
                try:
                    end_ids.add(int(tid))
                except Exception:
                    continue
        backend = getattr(tokenizer, "backend_tokenizer", None)
        is_bytelevel = _is_bytelevel_tokenizer(tokenizer, backend)
        for tok, tid in vocab_map.items():
            tid = int(tid)
            if tid > max_id:
                max_id = tid
            if tid in end_ids:
                mapping[tid] = b""
            else:
                mapping[tid] = cls._decode_token_bytes(
                    tokenizer, tid, backend=backend, is_bytelevel=is_bytelevel
                )
        if override_token_bytes:
            for tid, payload in override_token_bytes.items():
                mapping[int(tid)] = bytes(payload)
        vocab_size = max(0, max_id + 1)
        return cls(mapping=mapping, vocab_size=vocab_size)

    # --------------- public API ---------------
    def token_bytes(self, token_id: int) -> bytes:
        self._ensure_views()
        tid = int(token_id)
        if tid < 0 or tid >= self.vocab_size:
            return b""
        start = int(self._offsets[tid].item())
        end = int(self._offsets[tid + 1].item())
        if end <= start:
            return b""
        return self._byte_data[start:end].cpu().numpy().tobytes()

    def tokens_to_bytes(
        self,
        tokens: Sequence[int] | torch.Tensor,
        *,
        start: int = 0,
    ) -> bytes:
        self._ensure_views()
        if isinstance(tokens, torch.Tensor):
            token_ids = tokens.detach()
            if token_ids.device.type != "cpu":
                token_ids = token_ids.to("cpu")
            token_ids = token_ids.to(torch.long)
        else:
            token_ids = torch.as_tensor(list(tokens), dtype=torch.long)

        if start > 0:
            token_ids = token_ids[start:]
        if token_ids.numel() == 0:
            return b""

        starts = self._offsets.index_select(0, token_ids.clamp(0, self.vocab_size - 1))
        ends = self._offsets.index_select(0, (token_ids + 1).clamp(0, self.vocab_size))
        lengths = torch.clamp(ends - starts, min=0)
        total = int(lengths.sum().item())
        if total <= 0:
            return b""

        out = torch.empty(total, dtype=torch.uint8)
        dest = 0
        starts_list = starts.tolist()
        lengths_list = lengths.tolist()
        for s, length in zip(starts_list, lengths_list):
            if length <= 0:
                continue
            e = s + length
            out[dest : dest + length] = self._byte_data[s:e]
            dest += length
        return out.numpy().tobytes()

    def bytes_to_text(self, data: bytes, *, errors: str = "ignore") -> str:
        return data.decode("utf-8", errors=errors)

    def _ensure_views(self) -> None:
        if not self._dirty:
            return
        chunks: List[torch.Tensor] = []
        offsets = [0]
        for tid in range(self.vocab_size):
            payload = self.mapping.get(tid, b"")
            payload = payload if isinstance(payload, (bytes, bytearray)) else bytes(payload)
            length = len(payload)
            if length:
                chunks.append(torch.tensor(list(payload), dtype=torch.uint8))
            offsets.append(offsets[-1] + length)
        if chunks:
            self._byte_data = torch.cat(chunks) if len(chunks) > 1 else chunks[0]
        else:
            self._byte_data = torch.empty(0, dtype=torch.uint8)
        self._offsets = torch.tensor(offsets, dtype=torch.int32)
        self._dirty = False

    # --------------- internal helpers ---------------
    @staticmethod
    def _decode_token_bytes(
        tokenizer,
        token_id: int,
        *,
        backend,
        is_bytelevel: bool,
    ) -> bytes:
        piece = _piece_from_tokenizer(tokenizer, token_id)
        if piece:
            data = _piece_to_bytes(piece)
            if data:
                return data

        raw = None
        if backend is not None:
            try:
                raw = backend.id_to_token(int(token_id))
            except Exception:
                raw = None
        if isinstance(raw, bytes):
            return bytes(raw)
        if isinstance(raw, str):
            return raw.encode("utf-8", errors="backslashreplace")

        piece = _piece_from_tokenizer(tokenizer, token_id)
        if piece:
            return piece.encode("utf-8", errors="backslashreplace")

        try:
            decoded = tokenizer.decode([int(token_id)], skip_special_tokens=False)
            return decoded.encode("utf-8", errors="backslashreplace")
        except Exception:
            return f"<UNK_{token_id}>".encode("utf-8")


def _piece_from_tokenizer(tokenizer, token_id: int) -> str:
    try:
        pieces = tokenizer.convert_ids_to_tokens([int(token_id)], skip_special_tokens=False)
        if pieces:
            return pieces[0]
    except Exception:
        return ""
    return ""


def _is_bytelevel_tokenizer(tokenizer, backend) -> bool:
    name = getattr(tokenizer, "__class__", type(tokenizer)).__name__.lower()
    if any(sub in name for sub in ["gpt2", "bytelevel", "bpe"]):
        return True
    if backend is not None and "bytelevel" in str(type(backend)):
        return True
    return False
