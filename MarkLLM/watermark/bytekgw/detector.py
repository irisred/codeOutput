from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Union

import torch


@dataclass
class ByteWMDetectorConfig:
    prefix_length: int
    gamma: float = 0.5          # 128/256
    z_threshold: float = 4.0
    ignore_invalid_firstbyte: bool = True  # fb=-1 不计入 N，也不计入 G
    return_per_token: bool = False
    max_scan_iters: int = 32


class CudaByteWatermarkDetector:
    """
    CUDA-only detector.

    Unified semantics:
      - TokenByteVocab.firstbyte_tensor(device) returns fb in 0..255, or -1 for EXCLUDED/INVALID.
      - If ignore_invalid_firstbyte=True:
          * fb=-1 tokens are skipped (do not affect num_scored or num_green).
      - Uses input_ids.device as runtime device; lazily ensures firstbyte-table and PRF on that device.
    """

    def __init__(
        self,
        prf,
        token_byte_vocab,
        config: ByteWMDetectorConfig,
        device: Union[str, torch.device] = "cuda",
    ) -> None:
        self.prf = prf
        self.tokbytes = token_byte_vocab
        self.cfg = config

        init_dev = torch.device(device)
        if init_dev.type != "cuda":
            raise ValueError("CudaByteWatermarkDetector is CUDA-only.")

        # canonicalize "cuda" -> "cuda:<current>"
        if init_dev.type == "cuda" and init_dev.index is None:
            init_dev = torch.device("cuda", torch.cuda.current_device())
        self.init_device = init_dev

        # build once on init_device; runtime may rebuild on other device
        self.firstbyte = self.tokbytes.firstbyte_tensor(
            self.init_device, max_scan_iters=int(self.cfg.max_scan_iters)
        )

    def _ensure_2d(self, input_ids: torch.Tensor):
        if input_ids.dim() == 1:
            return input_ids.unsqueeze(0), True
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be [T] or [B,T].")
        return input_ids, False

    def _ensure_on(self, device: torch.device) -> None:
        # firstbyte table
        if self.firstbyte.device != device:
            self.firstbyte = self.tokbytes.firstbyte_tensor(
                device, max_scan_iters=int(self.cfg.max_scan_iters)
            )

        # PRF
        if getattr(self.prf, "device", device) != device:
            cfg = getattr(self.prf, "config", None)
            if cfg is None:
                raise ValueError("PRF device mismatch and cannot rebuild PRF (missing .config).")
            self.prf = type(self.prf)(cfg, device=device)

    @torch.no_grad()
    def score(self, input_ids: torch.LongTensor) -> Dict[str, Any]:
        x, squeezed = self._ensure_2d(input_ids)

        if x.device.type != "cuda":
            raise ValueError("Detector requires CUDA input_ids.")

        device = x.device
        self._ensure_on(device)

        if x.dtype != torch.long:
            x = x.to(torch.long)

        B, T = x.shape
        L = int(self.cfg.prefix_length)

        if T <= L:
            out = {
                "num_scored": torch.zeros((B,), device=device, dtype=torch.int64),
                "num_green": torch.zeros((B,), device=device, dtype=torch.int64),
                "z": torch.zeros((B,), device=device, dtype=torch.float32),
            }
            return out if not squeezed else {k: v[0] for k, v in out.items()}

        num_green = torch.zeros((B,), device=device, dtype=torch.int64)
        num_scored = torch.zeros((B,), device=device, dtype=torch.int64)

        per_token_hits = [] if self.cfg.return_per_token else None

        for i in range(L, T):
            start = max(0, i - L)
            ctx = x[:, start:i]  # [B,<=L]

            green_mask = self.prf.green_bytes_mask(ctx)  # [B,256] bool
            if green_mask.dim() == 1:
                green_mask = green_mask.unsqueeze(0)
            if green_mask.size(0) != B:
                raise ValueError(f"Batch mismatch: green_mask B={green_mask.size(0)} vs input_ids B={B}")

            tok = x[:, i]  # [B]
            fb = self.firstbyte.index_select(0, tok).to(torch.int64)  # [B], fb=-1 => excluded/invalid

            valid = (fb >= 0) & (fb < 256)
            if self.cfg.ignore_invalid_firstbyte:
                num_scored += valid.to(torch.int64)
            else:
                # invalid 也计入 N，但永远不算 green hit（等价当 red）
                num_scored += torch.ones_like(valid, dtype=torch.int64, device=device)

            fb_clamped = torch.clamp(fb, 0, 255)
            hit = green_mask.gather(1, fb_clamped.view(B, 1)).view(B) & valid
            num_green += hit.to(torch.int64)

            if per_token_hits is not None:
                per_token_hits.append(hit)

        gamma = float(self.cfg.gamma)
        N = num_scored.to(torch.float32)
        G = num_green.to(torch.float32)
        denom = torch.sqrt(torch.clamp(N * gamma * (1.0 - gamma), min=1e-12))
        z = (G - gamma * N) / denom

        out: Dict[str, Any] = {"num_scored": num_scored, "num_green": num_green, "z": z}
        if per_token_hits is not None:
            out["per_token_hits"] = torch.stack(per_token_hits, dim=1)  # [B, T-L]

        if squeezed:
            return {k: (v[0] if torch.is_tensor(v) and v.dim() > 0 else v) for k, v in out.items()}
        return out

    @torch.no_grad()
    def detect(self, input_ids: torch.LongTensor) -> Dict[str, Any]:
        s = self.score(input_ids)
        z = s["z"]
        s["is_watermarked"] = z > float(self.cfg.z_threshold)
        return s
