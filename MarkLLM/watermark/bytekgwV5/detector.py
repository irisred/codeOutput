# MarkLLM/watermark/bytekgwV5/detector.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union, List

import torch
from torch import Tensor

from .token_bytes import TokenByteVocab, END_BYTE


# --------------------------------------------------------------------------------------
# Fallback hard-coded position weights (P=64, use_prefix_bytes_in_prf=True)
# (Only used if cfg.pos_weights is None)
# --------------------------------------------------------------------------------------
_DEFAULT_POS_WEIGHTS_P64_PREFIX1: List[float] = [
    0.981351374494843,
    0.1815092181638641,
    0.04171760259741589,
    0.03917083647715886,
    0.0184150020121611,
    0.0,
    0.012565318622533506,
    0.010371240953985067,
    0.0,
    0.0,
    6.300652345582473e-05,
    0.009510680950180855,
    0.0,
    0.0056423413804956215,
    0.0,
    0.0014762411718245718,
    0.0,
    0.00036906029295614295,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
]


def _l2_normalize(w: Tensor) -> Tensor:
    denom = torch.linalg.vector_norm(w).clamp_min(1e-12)
    return w / denom


def _prepare_pos_weights(
    *,
    device: torch.device,
    max_pos: int,
    user_weights: Optional[List[float]],
) -> Tensor:
    """
    Build [max_pos] float32 weights on device and L2-normalize.
    If user_weights is shorter than max_pos -> pad zeros.
    If longer -> slice.
    """
    if user_weights is None:
        base = _DEFAULT_POS_WEIGHTS_P64_PREFIX1
    else:
        base = user_weights

    w = torch.tensor(base, dtype=torch.float32, device=device)

    if w.numel() < max_pos:
        pad = torch.zeros((max_pos - w.numel(),), dtype=torch.float32, device=device)
        w = torch.cat([w, pad], dim=0)
    else:
        w = w[:max_pos].contiguous()

    return _l2_normalize(w)


@dataclass
class ByteTreeDetectorConfig:
    """
    Detector config for ByteKGWv5 (aligned with generation).

    Single-knob alignment rule:
      - max_byte_pos controls BOTH generation bias depth and detector scoring depth.
        * max_byte_pos=1   -> only score byte_pos=0  (first-byte mode)
        * max_byte_pos>1   -> score byte_pos in [0 .. max_byte_pos-1]  (multi-byte mode)

    If use_prefix_bytes_in_prf=True:
      - for byte_pos>0, PRF additionally conditions on already-observed bytes within the SAME token.
      - MUST match generation side, otherwise pos>0 will not align.

    NEW:
      - pos_weights: optional list[float] controlling weighted multi-byte detection.
        If None -> fallback to _DEFAULT_POS_WEIGHTS_P64_PREFIX1.
    """
    prefix_length: int
    gamma: float = 0.5
    z_threshold: float = 4.0
    max_byte_pos: int = 64
    use_prefix_bytes_in_prf: bool = False
    pos_weights: Optional[List[float]] = None


class ByteKGWv5Detector:
    """
    Engine-level detector for ByteKGWv5.

    Scoring rule:
      - Per (token, byte_pos) event, success if observed byte is in PRF green set.
      - For max_byte_pos==1: z is the classic single-z (same as old first-byte).
      - For max_byte_pos>1 : z is a weighted sum of per-pos z-like statistics:
            x_p = (G_p - gamma*N_p)/sqrt(N_p*gamma*(1-gamma))
            z_weighted = sum_p w_p * x_p
        We also output z_unweighted for debugging (the classic pooled z over all events).
    """

    def __init__(
        self,
        *,
        prf: Any,
        vocab: TokenByteVocab,
        cfg: ByteTreeDetectorConfig,
        device: Union[str, torch.device] = "cuda:0",
    ) -> None:
        self.prf = prf
        self.vocab = vocab
        self.cfg = cfg
        self.device = torch.device(device)

        # move PRF + vocab to device if possible
        try:
            self.prf.to(self.device)
        except Exception:
            pass
        self.vocab.to(self.device)

        if int(self.cfg.max_byte_pos) < 1:
            raise ValueError("cfg.max_byte_pos must be >= 1")

        # cache bytepos tables on device
        self._bytepos_cache: Dict[int, Tensor] = {}

        # cache per-token visible byte lengths (optional speed)
        self._lens_cache: Optional[Tensor] = None

        # weights (from cfg.pos_weights if provided)
        max_pos = int(self.cfg.max_byte_pos)
        self._pos_weights = _prepare_pos_weights(
            device=self.device,
            max_pos=max_pos,
            user_weights=self.cfg.pos_weights,
        )

    def _ensure_2d(self, x: Tensor) -> tuple[Tensor, bool]:
        if x.dim() == 1:
            return x.unsqueeze(0), True
        if x.dim() == 2:
            return x, False
        raise ValueError("input_ids must be [T] or [B,T].")

    def _bytepos_table(self, pos: int) -> Tensor:
        pos = int(pos)
        if pos in self._bytepos_cache:
            return self._bytepos_cache[pos]
        t = self.vocab.bytepos_tensor(self.device, pos).to(torch.int64)  # [V] in 0..256
        self._bytepos_cache[pos] = t
        return t

    def _lens_table(self) -> Tensor:
        """Return [V] visible-byte lengths."""
        if self._lens_cache is not None:
            return self._lens_cache
        if hasattr(self.vocab, "bytes_len_tensor"):
            lens = self.vocab.bytes_len_tensor(self.device).to(torch.int64)
        else:
            starts = self.vocab.offsets[:-1].to(torch.int64)
            ends = self.vocab.offsets[1:].to(torch.int64)
            lens = (ends - starts).to(torch.int64)
        self._lens_cache = lens
        return lens

    @torch.no_grad()
    def score(self, input_ids: Tensor) -> Dict[str, Tensor]:
        """
        input_ids: [T] or [B,T]
        Returns:
          - num_scored: [B] int64 (total over all pos)
          - num_green : [B] int64 (total over all pos)
          - z         : [B] float32  (weighted z if max_byte_pos>1, else classic first-byte z)
          - z_unweighted: [B] float32 (classic pooled z over all events, for debugging)
        """
        x, squeezed = self._ensure_2d(input_ids)
        if x.device != self.device:
            x = x.to(self.device)
        if x.dtype != torch.long:
            x = x.to(torch.long)

        B, T = x.shape
        L = int(self.cfg.prefix_length)
        max_pos = int(self.cfg.max_byte_pos)
        gamma = float(self.cfg.gamma)

        if T <= L:
            out = {
                "num_scored": torch.zeros((B,), device=self.device, dtype=torch.int64),
                "num_green": torch.zeros((B,), device=self.device, dtype=torch.int64),
                "z": torch.zeros((B,), device=self.device, dtype=torch.float32),
                "z_unweighted": torch.zeros((B,), device=self.device, dtype=torch.float32),
            }
            return out if not squeezed else {k: v[0] for k, v in out.items()}

        lens_table = self._lens_table()  # [V]

        # Fast path: max_pos==1 is exactly "first-byte"
        if max_pos == 1:
            num_scored = torch.zeros((B,), device=self.device, dtype=torch.int64)
            num_green = torch.zeros((B,), device=self.device, dtype=torch.int64)
            b0_table = self._bytepos_table(0)  # [V]

            for i in range(L, T):
                ctx = x[:, i - L : i]
                tok = x[:, i]
                b0 = b0_table.index_select(0, tok)  # [B]
                valid = b0.ne(END_BYTE)
                if valid.any():
                    green = self.prf.green_mask(ctx, byte_pos=0)  # [B,256]
                    b0c = torch.clamp(b0, 0, 255).view(B, 1)
                    hit = green.gather(1, b0c).view(B) & valid
                    num_scored += valid.to(torch.int64)
                    num_green += hit.to(torch.int64)

            N = num_scored.to(torch.float32)
            G = num_green.to(torch.float32)
            denom = torch.sqrt(torch.clamp(N * gamma * (1.0 - gamma), min=1e-12))
            z = (G - gamma * N) / denom

            out = {"num_scored": num_scored, "num_green": num_green, "z": z, "z_unweighted": z}
            return out if not squeezed else {k: v[0] for k, v in out.items()}

        # General path: per-pos counts
        num_scored_pos = torch.zeros((B, max_pos), device=self.device, dtype=torch.int64)
        num_green_pos = torch.zeros((B, max_pos), device=self.device, dtype=torch.int64)

        byte_tables = [self._bytepos_table(p) for p in range(max_pos)]

        for i in range(L, T):
            ctx = x[:, i - L : i]  # [B,L]
            tok = x[:, i]          # [B]
            tok_len = lens_table.index_select(0, tok)  # [B]

            prefix_bytes_lists = [[] for _ in range(B)] if self.cfg.use_prefix_bytes_in_prf else None

            for pos in range(max_pos):
                valid = tok_len.gt(pos)
                if not valid.any():
                    break

                b = byte_tables[pos].index_select(0, tok)  # [B]
                valid = valid & b.ne(END_BYTE)
                if not valid.any():
                    continue

                prefix_bytes_t: Optional[Tensor] = None
                if self.cfg.use_prefix_bytes_in_prf and pos > 0:
                    pb = torch.zeros((B, pos), device=self.device, dtype=torch.uint8)
                    for bi in range(B):
                        bl = prefix_bytes_lists[bi]
                        if bl:
                            pb[bi, : len(bl)] = torch.tensor(bl, device=self.device, dtype=torch.uint8)
                    prefix_bytes_t = pb

                green = self.prf.green_mask(ctx, byte_pos=pos, prefix_bytes=prefix_bytes_t)  # [B,256]
                bc = torch.clamp(b, 0, 255).view(B, 1)
                hit = green.gather(1, bc).view(B) & valid

                num_scored_pos[:, pos] += valid.to(torch.int64)
                num_green_pos[:, pos] += hit.to(torch.int64)

                if self.cfg.use_prefix_bytes_in_prf:
                    b_cpu = b.detach().to("cpu")
                    for bi in range(B):
                        if bool(valid[bi].item()):
                            prefix_bytes_lists[bi].append(int(b_cpu[bi].item()) & 0xFF)

        # per-pos statistic x_p
        Np = num_scored_pos.to(torch.float32)   # [B,P]
        Gp = num_green_pos.to(torch.float32)    # [B,P]
        denom_p = torch.sqrt(torch.clamp(Np * gamma * (1.0 - gamma), min=1e-12))
        x_pos = (Gp - gamma * Np) / denom_p     # [B,P]

        # weighted z
        w = self._pos_weights.view(1, -1)       # [1,P]
        z_weighted = (x_pos * w).sum(dim=1).to(torch.float32)

        # classic pooled z for debugging
        num_scored = num_scored_pos.sum(dim=1)  # [B]
        num_green = num_green_pos.sum(dim=1)    # [B]
        N = num_scored.to(torch.float32)
        G = num_green.to(torch.float32)
        denom = torch.sqrt(torch.clamp(N * gamma * (1.0 - gamma), min=1e-12))
        z_unweighted = ((G - gamma * N) / denom).to(torch.float32)

        out = {
            "num_scored": num_scored,
            "num_green": num_green,
            "z": z_weighted,
            "z_unweighted": z_unweighted,
        }
        return out if not squeezed else {k: v[0] for k, v in out.items()}

    @torch.no_grad()
    def detect(self, input_ids: Tensor) -> Dict[str, Tensor]:
        s = self.score(input_ids)
        z = s["z"]
        s["is_watermarked"] = z > float(self.cfg.z_threshold)
        return s
