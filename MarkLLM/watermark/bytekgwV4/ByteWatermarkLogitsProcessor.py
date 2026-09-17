from __future__ import annotations

import torch
from transformers import LogitsProcessor


class ByteWatermarkLogitsProcessor(LogitsProcessor):
    """
    Byte watermark biaser (CUDA-only).

    - Uses TokenByteVocab.firstbyte_tensor(device) as single source of truth.
    - Bias rule: if token's firstbyte is green => +delta, else +0.
    - firstbyte == -1 => no bias (ignored).
    """

    def __init__(
        self,
        prf,
        token_byte_vocab,
        *,
        delta: float,
        prefix_length: int,
        device: str | torch.device = "cuda:0",
        max_scan_iters: int = 32,
    ) -> None:
        super().__init__()
        self.prf = prf
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("ByteWatermarkLogitsProcessor is CUDA-only.")

        self.delta = float(delta)
        self.prefix_length = int(prefix_length)

        # unified firstbyte on GPU
        firstbyte = token_byte_vocab.firstbyte_tensor(self.device, max_scan_iters=int(max_scan_iters))  # [V] int16
        fb_i64 = firstbyte.to(torch.int64)

        # map invalid (-1) => sentinel 256
        sentinel = torch.full_like(fb_i64, 256)
        fb_idx = torch.where((fb_i64 >= 0) & (fb_i64 < 256), fb_i64, sentinel)
        self.firstbyte_idx = fb_idx.to(torch.int64)  # [V] in [0..256]

    @torch.no_grad()
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # move to device if needed (do NOT hard fail; keep alignment stable)
        if input_ids.device != self.device:
            input_ids = input_ids.to(self.device)
        if scores.device != self.device:
            scores = scores.to(self.device)

        if input_ids.dtype != torch.long:
            input_ids = input_ids.to(torch.long)

        if input_ids.size(-1) < self.prefix_length or self.delta == 0.0:
            return scores

        B = scores.size(0)
        L = self.prefix_length
        ctx = input_ids[:, -L:]  # [B,L]

        green_mask = self.prf.green_bytes_mask(ctx)  # [B,256] bool
        if green_mask.dim() == 1:
            green_mask = green_mask.unsqueeze(0)

        # byte_bias [B,256]: green->delta
        byte_bias = green_mask.to(dtype=scores.dtype) * self.delta

        # append sentinel column => 0 for invalid firstbyte
        zero_col = torch.zeros((B, 1), device=self.device, dtype=scores.dtype)
        byte_bias_aug = torch.cat([byte_bias, zero_col], dim=1)  # [B,257]

        # add to logits: scores[b,v] += byte_bias_aug[b, firstbyte_idx[v]]
        scores.add_(byte_bias_aug[:, self.firstbyte_idx])  # creates [B,V] view-indexed tensor
        return scores
