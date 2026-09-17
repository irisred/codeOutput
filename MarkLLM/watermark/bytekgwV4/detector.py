from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass
class ByteWMDetectorConfig:
    prefix_length: int
    gamma: float = 0.5
    z_threshold: float = 4.0
    ignore_invalid_firstbyte: bool = True


class CudaByteWatermarkDetector:
    """
    CUDA-only detector.
    input_ids: [T] or [B,T] on CUDA
    firstbyte_vocab_tensor: [V] int16 (0..255 or -1) on CUDA
    """

    def __init__(
        self,
        prf,
        firstbyte_vocab_tensor: torch.Tensor,
        config: ByteWMDetectorConfig,
        device: str | torch.device = "cuda:0",
    ):
        self.prf = prf
        self.cfg = config
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("Detector is CUDA-only.")

        if firstbyte_vocab_tensor.device != self.device:
            firstbyte_vocab_tensor = firstbyte_vocab_tensor.to(self.device)
        self.firstbyte = firstbyte_vocab_tensor

    def _ensure_2d(self, x: torch.Tensor):
        if x.dim() == 1:
            return x.unsqueeze(0), True
        if x.dim() != 2:
            raise ValueError("input_ids must be [T] or [B,T].")
        return x, False

    @torch.no_grad()
    def score(self, input_ids: torch.LongTensor):
        x, squeezed = self._ensure_2d(input_ids)
        if x.device != self.device:
            x = x.to(self.device)
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

        for i in range(L, T):
            ctx = x[:, i - L : i]  # [B,L]
            green_mask = self.prf.green_bytes_mask(ctx)  # [B,256] bool

            tok = x[:, i]  # [B]
            fb = self.firstbyte.index_select(0, tok).to(torch.int64)  # [B]

            valid = (fb >= 0) & (fb < 256)
            if self.cfg.ignore_invalid_firstbyte:
                num_scored += valid.to(torch.int64)
            else:
                num_scored += torch.ones_like(valid, dtype=torch.int64)

            fb_clamped = torch.clamp(fb, 0, 255)
            hit = green_mask.gather(1, fb_clamped.view(B, 1)).view(B) & valid
            num_green += hit.to(torch.int64)

        gamma = float(self.cfg.gamma)
        N = num_scored.to(torch.float32)
        G = num_green.to(torch.float32)
        denom = torch.sqrt(torch.clamp(N * gamma * (1.0 - gamma), min=1e-12))
        z = (G - gamma * N) / denom

        out = {
            "num_scored": num_scored,
            "num_green": num_green,
            "z": z,
        }
        return out if not squeezed else {k: v[0] for k, v in out.items()}

    @torch.no_grad()
    def detect(self, input_ids: torch.LongTensor):
        s = self.score(input_ids)
        z = s["z"]
        is_wm = z > float(self.cfg.z_threshold)
        s["is_watermarked"] = is_wm
        return s
