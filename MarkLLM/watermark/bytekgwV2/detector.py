from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Union, Optional

import torch


@dataclass
class ByteWMDetectorConfig:
    prefix_length: int
    gamma: float = 0.5              # green_size / 256
    z_threshold: float = 4.0
    max_byte_pos: int = 16          # multi-byte detection depth
    ignore_invalid_byte: bool = True  # if True: only count bytes in [0..255]
    return_per_byte: bool = False     # if True: return per-byte hits [B, N_trials]


class CudaByteWatermarkDetector:
    """
    CUDA-only multi-byte watermark detector.

    For each token position i >= prefix_length:
      ctx = tokens [i-prefix_length : i]
      For each byte_pos in [0 .. max_byte_pos-1]:
         b = visible_byte(token_i, byte_pos)
         if b < 0: break (token ended)
         prefix_bytes = visible bytes of token_i at [0..byte_pos-1]
         green_mask = prf.green_bytes_mask(ctx, prefix_bytes=prefix_bytes, byte_pos=byte_pos)  # [B,256]
         hit if green_mask[b] is True

    Each checked byte is one Bernoulli trial with success prob gamma.
    """

    def __init__(
        self,
        prf,
        byte_index,  # FastVisibleByteIndex
        config: ByteWMDetectorConfig,
        device: Union[str, torch.device] = "cuda:0",
    ):
        self.prf = prf
        self.byte_index = byte_index
        self.cfg = config

        dev = torch.device(device)
        if dev.type != "cuda":
            raise ValueError("CudaByteWatermarkDetector is CUDA-only.")
        if dev.index is None:
            dev = torch.device("cuda", torch.cuda.current_device())
        self.device = dev

        # Sanity: byte_index must live on same CUDA index
        idx_dev = self.byte_index.byte_table().device
        if idx_dev.type != "cuda" or idx_dev.index != self.device.index:
            raise ValueError(f"byte_index is on {idx_dev}, but detector is on {self.device}.")

    def _ensure_2d(self, input_ids: torch.Tensor):
        if input_ids.dim() == 1:
            return input_ids.unsqueeze(0), True
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be [T] or [B,T].")
        return input_ids, False

    @torch.no_grad()
    def score(self, input_ids: torch.LongTensor) -> Dict[str, torch.Tensor]:
        x, squeezed = self._ensure_2d(input_ids)

        if x.device.type != "cuda":
            raise ValueError("input_ids must be CUDA.")
        if x.device.index != self.device.index:
            # 更稳：自动搬到 detector device（避免 cuda vs cuda:0 / 多卡不一致炸）
            x = x.to(self.device)
        if x.dtype != torch.long:
            x = x.to(torch.long)

        B, T = x.shape
        L = int(self.cfg.prefix_length)
        max_pos = int(self.cfg.max_byte_pos)

        if T <= L or max_pos <= 0:
            out = {
                "num_scored": torch.zeros((B,), device=self.device, dtype=torch.int64),
                "num_green": torch.zeros((B,), device=self.device, dtype=torch.int64),
                "z": torch.zeros((B,), device=self.device, dtype=torch.float32),
            }
            return out if not squeezed else {k: v[0] for k, v in out.items()}

        num_green = torch.zeros((B,), device=self.device, dtype=torch.int64)
        num_scored = torch.zeros((B,), device=self.device, dtype=torch.int64)

        per_byte_hits = [] if self.cfg.return_per_byte else None

        # 逐 token 位置统计（每个 token 贡献 0..max_pos 个 trial）
        for i in range(L, T):
            # ctx 用“生成时的窗口”：tokens [i-L, i)
            start = max(0, i - L)
            ctx = x[:, start:i]  # [B, <=L]

            tok = x[:, i]  # [B]

            # 我们一次取出该 token 的可见 byte 表行，避免每个 pos 重复查表
            # rows: [B, D] int16  (D = byte_index.max_depth)
            rows = self.byte_index.byte_table()[tok]  # advanced indexing

            # 对每个 byte_pos
            prefix_buf = []  # 用于构造 prefix_bytes；这里用 tensor 拼更快：我们直接 slice rows
            for pos in range(max_pos):
                bvals = rows[:, pos].to(torch.int64)  # [B]
                # token 在该 pos 没有 byte => 结束这个 token 的 byte 扫描
                ended = bvals.lt(0)
                if bool(ended.all().item()):
                    break

                valid = bvals.ge(0) & bvals.lt(256)

                if self.cfg.ignore_invalid_byte:
                    num_scored += valid.to(torch.int64)
                else:
                    # 不忽略：把 invalid 当 red，但仍计入 N（这里 ended 的不计，因为已经 break 掉）
                    num_scored += (~ended).to(torch.int64)

                # prefix_bytes = rows[:, :pos]
                if pos == 0:
                    prefix_bytes = rows[:, :0]  # [B,0]
                else:
                    prefix_bytes = rows[:, :pos]  # [B,pos]
                # 将 ended 的行置空（避免把 -1 带进 PRF）
                # 注意：PRF 允许 -1 padding 也行，但这里更干净
                if prefix_bytes.numel() > 0:
                    prefix_bytes = torch.where(prefix_bytes < 0, torch.zeros_like(prefix_bytes), prefix_bytes)
                prefix_bytes = prefix_bytes.to(torch.int64)

                # green mask: [B,256]
                green_mask = self.prf.green_bytes_mask(ctx, prefix_bytes=prefix_bytes, byte_pos=pos)
                if green_mask.dim() == 1:
                    green_mask = green_mask.unsqueeze(0)

                # hit：valid 且 green_mask[b, byte] == True
                b_clamped = torch.clamp(bvals, 0, 255)
                hit = green_mask.gather(1, b_clamped.view(B, 1)).view(B) & valid

                num_green += hit.to(torch.int64)
                if per_byte_hits is not None:
                    # 只记录被计入的 trial（valid 或非 ended，取决于配置）
                    if self.cfg.ignore_invalid_byte:
                        per_byte_hits.append(hit)
                    else:
                        per_byte_hits.append(hit & (~ended))

                # 如果有些 batch 行 ended，有些没 ended，我们不能 break（因为其他还没结束）
                # 继续 pos+1，ended 的行后续 bvals 仍为 -1，不会再计入

        # z-score (Bernoulli trials)
        gamma = float(self.cfg.gamma)
        N = num_scored.to(torch.float32)
        G = num_green.to(torch.float32)
        denom = torch.sqrt(torch.clamp(N * gamma * (1.0 - gamma), min=1e-12))
        z = (G - gamma * N) / denom

        out: Dict[str, torch.Tensor] = {
            "num_scored": num_scored,
            "num_green": num_green,
            "z": z,
        }
        if per_byte_hits is not None and len(per_byte_hits) > 0:
            out["per_byte_hits"] = torch.stack(per_byte_hits, dim=1)  # [B, N_trials]

        if squeezed:
            return {k: (v[0] if torch.is_tensor(v) and v.dim() > 0 else v) for k, v in out.items()}
        return out

    @torch.no_grad()
    def detect(self, input_ids: torch.LongTensor) -> Dict[str, torch.Tensor]:
        s = self.score(input_ids)
        z = s["z"]
        is_wm = z > float(self.cfg.z_threshold)
        s["is_watermarked"] = is_wm
        return s


__all__ = ["ByteWMDetectorConfig", "CudaByteWatermarkDetector"]
