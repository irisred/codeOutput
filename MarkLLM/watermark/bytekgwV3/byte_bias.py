# MarkLLM/watermark/bytekgwV2/byte_bias.py
from __future__ import annotations

import torch


class ByteLogitBiaser:
    """
    Apply byte-level bias on token logits.
    Bias rule: if token's visible byte at byte_pos is green => +delta else +0.
    Invalid token byte (-1) => no bias.
    """

    def __init__(
        self,
        prf,
        token_byte_vocab,
        *,
        delta: float,
        prefix_length: int,
        byte_pos: int = 0,
        device: torch.device,
        vectorized: bool = True,
    ):
        self.prf = prf
        self.vocab = token_byte_vocab
        self.delta = float(delta)
        self.prefix_length = int(prefix_length)
        self.byte_pos = int(byte_pos)
        self.device = device
        self.vectorized = bool(vectorized)

        # [V] int16 0..255 or -1
        b = self.vocab.bytepos_tensor(self.device, self.byte_pos)
        b_i64 = b.to(torch.int64)
        # map -1 -> sentinel 256
        self.byte_idx = torch.where(b_i64 >= 0, b_i64, torch.full_like(b_i64, 256))
        self.byte_idx = torch.clamp(self.byte_idx, 0, 256)  # [V] long

    @torch.no_grad()
    def apply(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """
        input_ids: [B,T] long CUDA
        scores: [B,V] float CUDA (already processed by HF)
        returns biased scores (in-place add)
        """
        if self.delta == 0.0:
            return scores  # STRICT no-op

        B = scores.size(0)
        L = self.prefix_length
        if input_ids.size(1) < L:
            return scores

        ctx = input_ids[:, -L:]  # [B,L]
        green_mask = self.prf.green_bytes_mask(ctx, prefix_bytes=None, byte_pos=self.byte_pos)  # [B,256] bool
        byte_bias = green_mask.to(dtype=scores.dtype) * self.delta  # [B,256]

        # append sentinel column for invalid byte_idx=256 (bias=0)
        zero_col = torch.zeros((B, 1), device=self.device, dtype=scores.dtype)
        byte_bias_aug = torch.cat([byte_bias, zero_col], dim=1)  # [B,257]

        if self.vectorized:
            scores.add_(byte_bias_aug[:, self.byte_idx])  # [B,V] gather
        else:
            for b in range(B):
                scores[b].add_(byte_bias_aug[b][self.byte_idx])

        return scores
