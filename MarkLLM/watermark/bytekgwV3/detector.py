# MarkLLM/watermark/bytekgwV2/detector.py
from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass
class ByteWMDetectorConfig:
    prefix_length: int
    gamma: float = 0.5
    z_threshold: float = 2.61
    byte_pos: int = 0
    ignore_invalid_byte: bool = True


class CudaByteWatermarkDetector:
    """
    CUDA-only detector. Uses same PRF + TokenByteVocab bytepos_tensor rules.
    Score: z = (G - gamma*N) / sqrt(N*gamma*(1-gamma))
    """

    def __init__(self, prf, token_byte_vocab, cfg: ByteWMDetectorConfig, device: str | torch.device = "cuda:0"):
        self.prf = prf
        self.cfg = cfg
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("Detector is CUDA-only.")

        # [V] int16 0..255 or -1
        self.byte_table = token_byte_vocab.bytepos_tensor(self.device, int(cfg.byte_pos))
        self.byte_table_i64 = self.byte_table.to(torch.int64)

    def _ensure_2d(self, x: torch.Tensor):
        if x.dim() == 1:
            return x.unsqueeze(0), True
        if x.dim() == 2:
            return x, False
        raise ValueError("input_ids must be [T] or [B,T].")

    @torch.no_grad()
    def score(self, input_ids: torch.LongTensor):
        x, squeezed = self._ensure_2d(input_ids)
        if x.device != self.device:
            raise ValueError("input_ids must be on detector CUDA device.")
        if x.dtype != torch.long:
            x = x.to(torch.long)

        B, T = x.shape
        L = int(self.cfg.prefix_length)
        if T <= L:
            out = {
                "num_scored": torch.zeros((B,), device=self.device, dtype=torch.int64),
                "num_green": torch.zeros((B,), device=self.device, dtype=torch.int64),
                "z": torch.zeros((B,), device=self.device, dtype=torch.float32),
            }
            return out if not squeezed else {k: v[0] for k, v in out.items()}

        num_green = torch.zeros((B,), device=self.device, dtype=torch.int64)
        num_scored = torch.zeros((B,), device=self.device, dtype=torch.int64)

        gamma = float(self.cfg.gamma)

        for i in range(L, T):
            ctx = x[:, i - L : i]  # [B,L]
            green_mask = self.prf.green_bytes_mask(ctx, prefix_bytes=None, byte_pos=int(self.cfg.byte_pos))  # [B,256]

            tok = x[:, i]  # [B]
            bval = self.byte_table_i64.index_select(0, tok)  # [B]
            valid = bval.ge(0) & bval.lt(256)

            if self.cfg.ignore_invalid_byte:
                num_scored += valid.to(torch.int64)
            else:
                num_scored += torch.ones_like(valid, dtype=torch.int64)

            bclamp = torch.clamp(bval, 0, 255)
            hit = green_mask.gather(1, bclamp.view(B, 1)).view(B) & valid
            num_green += hit.to(torch.int64)

        N = num_scored.to(torch.float32)
        G = num_green.to(torch.float32)
        denom = torch.sqrt(torch.clamp(N * gamma * (1.0 - gamma), min=1e-12))
        z = (G - gamma * N) / denom

        out = {"num_scored": num_scored, "num_green": num_green, "z": z}
        return out if not squeezed else {k: v[0] for k, v in out.items()}

    @torch.no_grad()
    def detect(self, input_ids: torch.LongTensor):
        s = self.score(input_ids)
        s["is_watermarked"] = s["z"] > float(self.cfg.z_threshold)
        return s
