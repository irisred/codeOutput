# MarkLLM/watermark/bytekgw/token_bytes.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple, Union

import torch


def _bytes_to_unicode() -> Dict[int, str]:
    """
    GPT-2 byte-level BPE bytes_to_unicode mapping.
    Returns dict: byte -> unicode char.
    """
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    cs = [chr(c) for c in cs]
    return dict(zip(bs, cs))


def _build_byte_level_decoder() -> Dict[str, int]:
    """
    Invert GPT-2 bytes_to_unicode so we can decode byte-level token strings.
    Returns dict: unicode char -> byte.
    """
    enc = _bytes_to_unicode()
    return {v: k for k, v in enc.items()}


_BYTE_LEVEL_DECODER = _build_byte_level_decoder()


def _decode_bytelevel_piece(piece: str) -> Optional[bytes]:
    """
    Try to decode a piece as GPT2/ByteLevel-BPE "bytes_to_unicode" text.
    Returns bytes if successful, else None.
    """
    try:
        return bytes(_BYTE_LEVEL_DECODER[ch] for ch in piece)
    except Exception:
        return None


def _piece_to_bytes(piece: str) -> bytes:
    """
    Convert tokenizer piece -> bytes.
    Heuristics:
      - Leading 'Ġ' (RoBERTa byte-level) treated as leading space.
      - Leading '▁' (SentencePiece) treated as leading space.
      - Try byte-level decode; fallback to UTF-8.
    """
    if not piece:
        return b""

    # RoBERTa / ByteLevelBPE: leading space marker
    if piece.startswith("Ġ"):
        rest = piece[1:]
        decoded = _decode_bytelevel_piece(rest)
        if decoded is not None:
            return b" " + decoded
        return b" " + rest.encode("utf-8", errors="backslashreplace")

    # SentencePiece: underline marker for word boundary
    if piece.startswith("▁"):
        rest = piece[1:]
        # SentencePiece tokens are usually plain unicode text
        return b" " + rest.encode("utf-8", errors="backslashreplace")

    decoded = _decode_bytelevel_piece(piece)
    if decoded is not None:
        return decoded

    return piece.encode("utf-8", errors="backslashreplace")


def _piece_from_tokenizer(tokenizer, token_id: int) -> Optional[str]:
    """
    Best-effort: get the raw token string for token_id.
    """
    try:
        return tokenizer.convert_ids_to_tokens([int(token_id)], skip_special_tokens=False)[0]
    except Exception:
        try:
            return tokenizer.convert_ids_to_tokens(int(token_id), skip_special_tokens=False)
        except Exception:
            return None


@dataclass
class TokenByteVocab:
    """
    Mapping token_id -> byte sequence (CPU mapping), and GPU-cached views.

    Visible-byte policy (UPDATED):
      - Skip leading tokenization markers / separators:
          * ASCII space 0x20
          * UTF-8 bytes for 'Ġ' : C4 A0
          * UTF-8 bytes for '▁' : E2 96 81
      - After skipping, visible[pos] = pos-th byte in the remaining byte string.
      - -1 means empty/out-of-range (end tokens are also forced to -1)

    IMPORTANT CHANGE:
      - UTF-8 continuation bytes (0x80..0xBF) are NOT treated as invalid anymore.
        If the first visible byte is a continuation, we still return it and allow watermark/detect.
    """

    mapping: Dict[int, bytes]
    vocab_size: int
    end_token_ids: set[int] = field(default_factory=set)

    # CPU canonical packed storage
    _byte_data_cpu: torch.Tensor = field(init=False, repr=False)
    _offsets_cpu: torch.Tensor = field(init=False, repr=False)  # int32, len=V+1
    _end_mask_cpu: torch.Tensor = field(init=False, repr=False)  # bool, len=V

    # device caches
    _views_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = field(default_factory=dict, init=False, repr=False)
    _firstbyte_cache: Dict[Tuple[str, int], torch.Tensor] = field(default_factory=dict, init=False, repr=False)
    _bytetable_cache: Dict[Tuple[str, int, int], torch.Tensor] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self._ensure_views_cpu()

    @classmethod
    def from_tokenizer(
        cls,
        tokenizer,
        *,
        end_token_ids: Optional[Sequence[int]] = None,
    ) -> "TokenByteVocab":
        vocab = tokenizer.get_vocab()
        if not vocab:
            raise ValueError("tokenizer.get_vocab() returned an empty mapping.")

        max_id = max(int(i) for i in vocab.values())
        vocab_size = max_id + 1

        end_ids: set[int] = set()
        if end_token_ids is not None:
            for tid in end_token_ids:
                try:
                    end_ids.add(int(tid))
                except Exception:
                    continue

        # Try fast path: convert_ids_to_tokens over full range
        pieces: list[Optional[str]]
        try:
            pieces = tokenizer.convert_ids_to_tokens(list(range(vocab_size)), skip_special_tokens=False)
        except Exception:
            # fallback: invert vocab map
            id_to_piece: Dict[int, str] = {}
            for tok, tid in vocab.items():
                try:
                    id_to_piece[int(tid)] = str(tok)
                except Exception:
                    continue
            pieces = [id_to_piece.get(i) for i in range(vocab_size)]

        mapping: Dict[int, bytes] = {}
        for tid in range(vocab_size):
            piece = pieces[tid] if tid < len(pieces) else None
            if piece is None:
                piece = _piece_from_tokenizer(tokenizer, tid)

            if piece is not None:
                payload = _piece_to_bytes(piece)
            else:
                # last-resort: decode id -> text
                try:
                    decoded = tokenizer.decode([int(tid)], skip_special_tokens=False)
                    payload = decoded.encode("utf-8", errors="backslashreplace")
                except Exception:
                    payload = b""

            mapping[tid] = payload

        return cls(mapping=mapping, vocab_size=vocab_size, end_token_ids=end_ids)

    # -------------------------
    # CPU packing + device views
    # -------------------------

    def _ensure_views_cpu(self) -> None:
        offsets = [0]
        chunks = []
        end_mask = torch.zeros((self.vocab_size,), dtype=torch.bool)
        for tid in range(self.vocab_size):
            if tid in self.end_token_ids:
                end_mask[tid] = True
            payload = self.mapping.get(tid, b"")
            if payload:
                chunks.append(torch.tensor(list(payload), dtype=torch.uint8))
                offsets.append(offsets[-1] + len(payload))
            else:
                offsets.append(offsets[-1])

        self._offsets_cpu = torch.tensor(offsets, dtype=torch.int32)
        self._end_mask_cpu = end_mask
        if chunks:
            self._byte_data_cpu = torch.cat(chunks) if len(chunks) > 1 else chunks[0]
        else:
            self._byte_data_cpu = torch.empty((0,), dtype=torch.uint8)

    def _views(self, device: Union[str, torch.device]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dev = torch.device(device)
        if dev.type == "cuda" and dev.index is None:
            dev = torch.device("cuda", torch.cuda.current_device())
        key = str(dev)

        if key in self._views_cache:
            return self._views_cache[key]

        byte_data = self._byte_data_cpu.to(dev, non_blocking=True)
        offsets = self._offsets_cpu.to(dev, non_blocking=True)
        end_mask = self._end_mask_cpu.to(dev, non_blocking=True)

        self._views_cache[key] = (byte_data, offsets, end_mask)
        return byte_data, offsets, end_mask

    # -------------------------
    # Visible-byte policy helpers
    # -------------------------

    @staticmethod
    def _skip_visible_prefix(
        byte_data: torch.Tensor,     # [N] uint8
        starts: torch.Tensor,        # [V] int64
        lengths: torch.Tensor,       # [V] int64
        max_scan_iters: int,
    ) -> torch.Tensor:
        """
        Compute per-token skip length (bytes to skip at the beginning) according to:
          - skip leading spaces 0x20
          - skip leading UTF-8 'Ġ' marker bytes: C4 A0
          - skip leading UTF-8 '▁' marker bytes: E2 96 81
        """
        V = starts.numel()
        if V == 0:
            return torch.zeros((0,), device=starts.device, dtype=torch.int64)

        N = byte_data.numel()
        if N == 0:
            return torch.zeros((V,), device=starts.device, dtype=torch.int64)

        p = torch.zeros((V,), device=starts.device, dtype=torch.int64)

        for _ in range(int(max_scan_iters)):
            has0 = p < lengths
            if not bool(has0.any().item()):
                break

            idx0 = starts + p
            idx0c = torch.clamp(idx0, 0, max(int(N - 1), 0))
            b0 = byte_data.index_select(0, idx0c)

            # space
            is_space = has0 & (b0 == 0x20)

            # 'Ġ' UTF-8 bytes: C4 A0
            has1 = (p + 1) < lengths
            idx1 = idx0 + 1
            idx1c = torch.clamp(idx1, 0, max(int(N - 1), 0))
            b1 = byte_data.index_select(0, idx1c)
            is_g_marker = has0 & has1 & (b0 == 0xC4) & (b1 == 0xA0)

            # '▁' UTF-8 bytes: E2 96 81
            has2 = (p + 2) < lengths
            idx2 = idx0 + 2
            idx2c = torch.clamp(idx2, 0, max(int(N - 1), 0))
            b2 = byte_data.index_select(0, idx2c)
            is_u_marker = has0 & has1 & has2 & (b0 == 0xE2) & (b1 == 0x96) & (b2 == 0x81)

            inc = torch.zeros_like(p)
            inc = torch.where(is_u_marker, torch.full_like(inc, 3), inc)
            inc = torch.where(~is_u_marker & is_g_marker, torch.full_like(inc, 2), inc)
            inc = torch.where((~is_u_marker) & (~is_g_marker) & is_space, torch.full_like(inc, 1), inc)

            cont = inc > 0
            if not bool(cont.any().item()):
                break
            p = p + inc

        return p

    # -------------------------
    # Public GPU tables
    # -------------------------

    def firstbyte_tensor(
        self,
        device: Union[str, torch.device],
        *,
        max_scan_iters: int = 32,
    ) -> torch.Tensor:
        """
        Returns: [V] int16, value in 0..255 or -1
        """
        dev = torch.device(device)
        if dev.type == "cuda" and dev.index is None:
            dev = torch.device("cuda", torch.cuda.current_device())
        key = (str(dev), int(max_scan_iters))
        if key in self._firstbyte_cache:
            return self._firstbyte_cache[key]

        byte_data, offsets, end_mask = self._views(dev)

        V = int(self.vocab_size)
        lengths = (offsets[1:] - offsets[:-1]).to(torch.int64)            # [V]
        starts = offsets[:-1].to(torch.int64)                             # [V]

        skip = self._skip_visible_prefix(byte_data, starts, lengths, max_scan_iters=int(max_scan_iters))  # [V]
        pos = skip
        valid = pos < lengths

        out = torch.full((V,), -1, device=dev, dtype=torch.int16)

        if byte_data.numel() > 0 and bool(valid.any().item()):
            idx = starts + pos
            idxc = torch.clamp(idx, 0, int(byte_data.numel() - 1))
            b = byte_data.index_select(0, idxc).to(torch.int16)
            out[valid] = b[valid]

        # end tokens forced invalid
        out = torch.where(end_mask, torch.full_like(out, -1), out)

        self._firstbyte_cache[key] = out
        return out

    def byte_table_tensor(
        self,
        device: Union[str, torch.device],
        *,
        max_depth: int = 8,
        max_scan_iters: int = 32,
    ) -> torch.Tensor:
        """
        Returns: [V, D] int16, values 0..255 or -1
        visible[pos] = (byte after skipping leading markers/spaces) at position pos.
        """
        dev = torch.device(device)
        if dev.type == "cuda" and dev.index is None:
            dev = torch.device("cuda", torch.cuda.current_device())
        D = int(max_depth)
        key = (str(dev), D, int(max_scan_iters))
        if key in self._bytetable_cache:
            return self._bytetable_cache[key]

        byte_data, offsets, end_mask = self._views(dev)

        V = int(self.vocab_size)
        lengths = (offsets[1:] - offsets[:-1]).to(torch.int64)     # [V]
        starts = offsets[:-1].to(torch.int64)                      # [V]
        skip = self._skip_visible_prefix(byte_data, starts, lengths, max_scan_iters=int(max_scan_iters))  # [V]

        table = torch.full((V, D), -1, device=dev, dtype=torch.int16)

        if D == 0:
            self._bytetable_cache[key] = table
            return table

        if byte_data.numel() > 0:
            pos = torch.arange(D, device=dev, dtype=torch.int64).view(1, D)      # [1,D]
            rel = skip.view(V, 1) + pos                                           # [V,D]
            valid = rel < lengths.view(V, 1)

            idx = starts.view(V, 1) + rel
            idxc = torch.clamp(idx, 0, int(byte_data.numel() - 1))
            gathered = byte_data.index_select(0, idxc.view(-1)).view(V, D).to(torch.int16)

            table[valid] = gathered[valid]

        # end tokens forced invalid across all positions
        table = torch.where(end_mask.view(V, 1), torch.full_like(table, -1), table)

        self._bytetable_cache[key] = table
        return table

    # -------------------------
    # Convenience / debug helpers
    # -------------------------

    def bytes_of(self, token_id: int) -> bytes:
        tid = int(token_id)
        if tid < 0 or tid >= self.vocab_size:
            return b""
        return self.mapping.get(tid, b"")


__all__ = ["TokenByteVocab"]
