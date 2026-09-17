# MarkLLM/watermark/bytekgwV5/token_bytes.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase

END_BYTE = 256  # 257th branch, unambiguous "token ended / no byte at this pos"


# -----------------------------
# Legacy byte-level decoder (GPT-2 encoder.py)
# -----------------------------
def _build_byte_level_decoder() -> Dict[str, int]:
    """
    Reproduce HF/OpenAI byte-level BPE encoder/decoder table so we can translate
    pieces (unicode chars) back to original byte values.

    This matches the legacy MarkLLM behavior:
      bytes(piece) = bytes(_BYTE_LEVEL_DECODER[ch] for ch in piece)
      fallback: piece.encode('utf-8', errors='ignore')
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


def _piece_to_bytes_legacy(piece: str) -> bytes:
    """
    Legacy conversion:
      - Prefer byte-level reconstruction via _BYTE_LEVEL_DECODER
      - If the piece contains unknown symbols, fallback to UTF-8 encode(ignore)
    """
    if not piece:
        return b""
    try:
        return bytes(_BYTE_LEVEL_DECODER[ch] for ch in piece)
    except KeyError:
        return piece.encode("utf-8", errors="ignore")


# -----------------------------
# Visible bytes rule (skip markers)
# -----------------------------
_GPT2_MARKER = "Ġ".encode("utf-8")  # b"\xC4\xA0"
_SPM_MARKER = "▁".encode("utf-8")   # b"\xE2\x96\x81"


def _skip_prefix_markers(payload: bytes) -> int:
    """
    Skip leading marker bytes:
      - ASCII spaces 0x20
      - UTF-8 for 'Ġ' (C4 A0)  [in case fallback path produced literal Ġ bytes]
      - UTF-8 for '▁' (E2 96 81)
    Then skip spaces again, and repeat until stable.
    """
    if not payload:
        return 0
    i = 0
    n = len(payload)

    def skip_spaces(ii: int) -> int:
        while ii < n and payload[ii] == 0x20:
            ii += 1
        return ii

    i = skip_spaces(i)

    changed = True
    while changed:
        changed = False
        if i + len(_GPT2_MARKER) <= n and payload[i : i + len(_GPT2_MARKER)] == _GPT2_MARKER:
            i += len(_GPT2_MARKER)
            changed = True
        if i + len(_SPM_MARKER) <= n and payload[i : i + len(_SPM_MARKER)] == _SPM_MARKER:
            i += len(_SPM_MARKER)
            changed = True
        i2 = skip_spaces(i)
        if i2 != i:
            i = i2
            changed = True
    return i


def _visible_bytes(payload: bytes) -> bytes:
    """
    Visible bytes = payload after skipping prefix markers.
    IMPORTANT: We do NOT apply "UTF-8 continuation => invalid" rule anymore,
    per your new spec (avoid -1 ambiguity).
    """
    if not payload:
        return b""
    j = _skip_prefix_markers(payload)
    if j >= len(payload):
        return b""
    return payload[j:]


# -----------------------------
# Piece acquisition (avoid decode)
# -----------------------------
def _get_piece_fast(tokenizer: PreTrainedTokenizerBase, token_id: int) -> str:
    """
    Prefer backend_tokenizer.id_to_token if available (fast tokenizers),
    else fall back to convert_ids_to_tokens([id]).
    """
    tid = int(token_id)

    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is not None:
        try:
            s = backend.id_to_token(tid)
            if isinstance(s, str):
                return s
        except Exception:
            pass

    try:
        out = tokenizer.convert_ids_to_tokens([tid], skip_special_tokens=False)
        if out:
            return out[0] or ""
    except Exception:
        pass
    return ""


@dataclass
class TokenByteVocab:
    """
    Token -> visible-bytes table (legacy compatible), stored as:
      - byte_data: uint8 [N] concat of visible bytes
      - offsets: int32 [V+1], token i occupies byte_data[offsets[i]:offsets[i+1]]

    Core APIs:
      - bytepos_tensor(device, pos): int16 [V], value in [0..255] or END_BYTE(256)
      - precompute_bytepos(device, max_pos): int16 [max_pos+1, V]
    """
    vocab_size: int
    byte_data_cpu: Tensor                 # uint8 [N] on CPU
    offsets_cpu: Tensor                   # int32 [V+1] on CPU

    # device views cache: device -> (byte_data_dev, offsets_dev)
    _views: Dict[str, Tuple[Tensor, Tensor]] = field(default_factory=dict)

    # per (device,pos) cache
    _bytepos_cache: Dict[Tuple[str, int], Tensor] = field(default_factory=dict)

    @staticmethod
    def from_tokenizer(
        tokenizer: PreTrainedTokenizerBase,
        *,
        vocab_size: Optional[int] = None,
        allow_decode_fallback: bool = False,
    ) -> "TokenByteVocab":
        """
        Build mapping WITHOUT using tokenizer.decode by default.

        vocab_size MUST match model logits V. Prefer passing model.config.vocab_size.
        """
        V = int(vocab_size if vocab_size is not None else getattr(tokenizer, "vocab_size", len(tokenizer)))

        offsets: List[int] = [0]
        chunks: List[Tensor] = []

        for tid in range(V):
            piece = _get_piece_fast(tokenizer, tid)
            payload = _piece_to_bytes_legacy(piece)

            # Optional last-resort fallback (off by default)
            if (not payload) and allow_decode_fallback:
                try:
                    s = tokenizer.decode([tid], skip_special_tokens=False, clean_up_tokenization_spaces=False)
                    payload = s.encode("utf-8", errors="ignore") if s else b""
                except Exception:
                    payload = b""

            vis = _visible_bytes(payload)
            offsets.append(offsets[-1] + len(vis))
            if vis:
                chunks.append(torch.tensor(list(vis), dtype=torch.uint8))

        byte_data = (
            torch.cat(chunks, dim=0)
            if len(chunks) > 1
            else (chunks[0] if chunks else torch.empty((0,), dtype=torch.uint8))
        )
        offsets_t = torch.tensor(offsets, dtype=torch.int32)
        return TokenByteVocab(vocab_size=V, byte_data_cpu=byte_data, offsets_cpu=offsets_t)

    def _views_on(self, device: Union[str, torch.device]) -> Tuple[Tensor, Tensor]:
        dev = torch.device(device)
        key = str(dev)
        cached = self._views.get(key, None)
        if cached is not None:
            return cached
        bd = self.byte_data_cpu.to(dev, non_blocking=True)
        off = self.offsets_cpu.to(dev, non_blocking=True)
        self._views[key] = (bd, off)
        return bd, off

    @torch.no_grad()
    def bytepos_tensor(self, device: Union[str, torch.device], pos: int) -> Tensor:
        """
        Return int16 [V] table: token_id -> visible_byte_at_pos or END_BYTE(256).

        This is the fast primitive you need for byte-tree grouping.
        """
        dev = torch.device(device)
        key = (str(dev), int(pos))
        cached = self._bytepos_cache.get(key, None)
        if cached is not None:
            return cached

        bd, off = self._views_on(dev)
        V = self.vocab_size
        pos = int(pos)

        starts = off[:-1].to(torch.int64)       # [V]
        ends = off[1:].to(torch.int64)          # [V]
        lens = ends - starts                     # [V]

        out = torch.full((V,), END_BYTE, device=dev, dtype=torch.int16)  # default END
        valid = lens > pos
        if valid.any():
            idx = (starts[valid] + pos).to(torch.int64)  # indices into bd
            out[valid] = bd.index_select(0, idx).to(torch.int16)

        self._bytepos_cache[key] = out
        return out

    @torch.no_grad()
    def precompute_bytepos(self, device: Union[str, torch.device], max_pos: int) -> Tensor:
        """
        Convenience: precompute tables for pos=0..max_pos.
        Returns int16 [max_pos+1, V].
        """
        dev = torch.device(device)
        tables = [self.bytepos_tensor(dev, p) for p in range(int(max_pos) + 1)]
        return torch.stack(tables, dim=0)  # [P+1, V]

    def token_visible_bytes(self, token_id: int) -> bytes:
        """Debug helper (CPU): visible bytes for a token."""
        tid = int(token_id)
        if tid < 0 or tid >= self.vocab_size:
            return b""
        s = int(self.offsets_cpu[tid].item())
        e = int(self.offsets_cpu[tid + 1].item())
        if e <= s:
            return b""
        return self.byte_data_cpu[s:e].cpu().numpy().tobytes()
