from __future__ import annotations

import math
from typing import Dict
import hmac
import hashlib

import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase

from .prf import RobustPartitioner
from .token_bytes import TokenByteVocabV6


class ByteKGWv6Detector:
    """
    Lightweight detector for ByteKGWv6.

    - 只看前 n_bytes 可见字节划分红/绿（与生成端一致）。
    - seed_window_chars 决定取最近多少字符做指纹（需与生成端一致）。
    - 设计为无状态、可在多进程复用：预计算全词表 first-n ID 的唯一集合及其 PRF 向量。
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        vocab: TokenByteVocabV6,
        partitioner: RobustPartitioner,
        *,
        n_bytes: int,
        seed_window_chars: int,
        z_threshold: float = 4.0,
        min_tokens: int = 0,
        device: str | torch.device = "cpu",
        add_special_tokens: bool = False,
    ) -> None:
        self.tokenizer = tokenizer
        self.partitioner = partitioner
        self.n_bytes = max(1, int(n_bytes))
        self.seed_window_chars = max(1, int(seed_window_chars))
        self.z_threshold = float(z_threshold)
        self.min_tokens = int(min_tokens)
        self.device = torch.device(device)
        self.add_special_tokens = bool(add_special_tokens)

        # 全词表 first-n IDs
        self.firstn_ids = vocab.first_n_id(self.device, self.n_bytes)  # [V] int64

        # 唯一 ID 及映射
        uniq, inv = torch.unique(self.firstn_ids.cpu(), return_inverse=True)
        self._uniq_ids = uniq  # CPU int64
        self._inv_vocab = inv.to(self.device)  # [V] on device

        # 预计算 PRF token 向量与 tie bit（按 uniq 顺序）
        w_list = []
        tie_list = []
        for uid in uniq.tolist():
            w_bytes = partitioner.token_vector(uid)
            tie_bit = self._tie_bit(uid)
            w_list.append(torch.tensor(list(w_bytes), dtype=torch.uint8))
            tie_list.append(torch.tensor(tie_bit, dtype=torch.uint8))
        self._w = torch.stack(w_list, dim=0) if w_list else torch.empty((0, partitioner.m_bytes), dtype=torch.uint8)
        self._tie = torch.stack(tie_list, dim=0) if tie_list else torch.empty((0,), dtype=torch.uint8)

        # popcount LUT
        self._lut = torch.tensor([bin(i).count("1") for i in range(256)], dtype=torch.uint8)

        # id -> index 映射（Python dict，用于快速子集取索引）
        self._id_to_idx: Dict[int, int] = {int(v): i for i, v in enumerate(uniq.tolist())}

        self.half_bits = partitioner.m_bits // 2
        self.to(self.device)

    def _tie_bit(self, uid: int) -> int:
        return hmac.new(
            self.partitioner.k_tok,
            b"tie" + uid.to_bytes(8, "little", signed=False),
            hashlib.sha256,
        ).digest()[0] & 1

    def to(self, device: str | torch.device) -> "ByteKGWv6Detector":
        dev = torch.device(device)
        self.device = dev
        self.firstn_ids = self.firstn_ids.to(dev)
        self._inv_vocab = self._inv_vocab.to(dev)
        self._w = self._w.to(dev)
        self._tie = self._tie.to(dev)
        self._lut = self._lut.to(dev)
        return self

    @torch.no_grad()
    def _green_mask_present(self, fp: bytes, present_ids: Tensor) -> Tensor:
        """
        fp: fingerprint bytes
        present_ids: [M] int64 unique first-n ids present in the text
        returns: [M] bool
        """
        if present_ids.numel() == 0:
            return torch.empty((0,), device=self.device, dtype=torch.bool)

        idxs = torch.tensor([self._id_to_idx[int(x)] for x in present_ids.tolist()], device=self.device, dtype=torch.long)
        w = self._w.index_select(0, idxs)  # [M, m_bytes]
        tie_bits = self._tie.index_select(0, idxs)  # [M] uint8

        fp_t = torch.tensor(list(fp), device=self.device, dtype=torch.uint8)  # [m_bytes]
        xor = torch.bitwise_xor(w, fp_t.unsqueeze(0))  # [M, m_bytes]
        # use torch.take to avoid advanced-index quirk across devices
        hd = torch.take(self._lut, xor.view(-1).to(torch.long)).view(xor.shape).sum(dim=1)  # [M]

        green = hd < self.half_bits
        tie_mask = hd == self.half_bits
        if tie_mask.any():
            green[tie_mask] = (tie_bits[tie_mask] & 1) == 0
        return green

    @torch.no_grad()
    def detect(self, text: str, return_dict: bool = True):
        """
        Sliding-window detection over the whole text (counts all tokens).
        """
        enc = self.tokenizer(text, return_tensors="pt", add_special_tokens=self.add_special_tokens)
        ids = enc["input_ids"][0].to(self.device)
        return self.detect_ids(ids, prompt_len=0, return_dict=return_dict)

    @torch.no_grad()
    def detect_ids(self, ids: Tensor, prompt_len: int = 0, return_dict: bool = True):
        """
        Sliding-window detection on already-tokenized ids.
        Counts hits only for positions >= prompt_len (to optionally skip prompts).
        """
        ids = ids.to(self.device)
        N = ids.numel()
        if N == 0 or N < self.min_tokens:
            res = {
                "is_watermarked": False,
                "z": float("-inf"),
                "hits": 0,
                "total": int(N),
                "green_frac": 0.0,
                "threshold": self.z_threshold,
            }
            return res if return_dict else (False, float("-inf"))

        hits = 0
        total = 0
        for idx in range(max(prompt_len, 0), N):
            # prefix excludes current token (generation uses current prefix to pick next token)
            prefix_ids = ids[:idx]
            prefix_text = self.tokenizer.decode(
                prefix_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            fp = self.partitioner.fingerprint(prefix_text, self.seed_window_chars)

            firstn = self.firstn_ids[ids[idx]]
            green = self._green_mask_present(fp, torch.tensor([firstn], device=self.device, dtype=torch.int64))
            is_green = bool(green[0].item())
            hits += int(is_green)
            total += 1

        g = 0.5  # green ratio
        eps = 1e-12
        denom = math.sqrt(max(total * g * (1 - g), eps))
        z = (hits - total * g) / denom

        res = {
            "is_watermarked": bool(z >= self.z_threshold),
            "z": float(z),
            "hits": hits,
            "total": total,
            "green_frac": float(hits / total),
            "threshold": float(self.z_threshold),
        }
        return res if return_dict else (res["is_watermarked"], res["z"])


__all__ = ["ByteKGWv6Detector"]
