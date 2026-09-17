from __future__ import annotations

import hmac
import hashlib
from typing import List

import torch
import hmac, hashlib
import torch
from typing import List, Sequence


class ByteGreenlistPRF:
    """
    Fixed-256 byte-domain PRF with a single unbiased implementation.

    当前版本基于“histogram + HMAC”的方案：对可见字节窗口做直方图摘要，
    然后对每个候选字节 b 计算 HMAC(key, digest || b)，将前 32bit
    映射到 [0, 1)，再与 gamma 比较决定是否进入 greenlist。
    """

    def __init__(self, hash_key: int, prefix_length: int = 0) -> None:
        self.hash_key = int(hash_key)
        self.prefix_length = max(0, int(prefix_length))
        self._key = self._int_to_bytes(self.hash_key)
        self._cnt_clip = 8
        if self.prefix_length <= 4:
            self._q_bits = 0
        elif self.prefix_length <= 8:
            self._q_bits = 1
        else:
            self._q_bits = 2

    @staticmethod
    def _int_to_bytes(value: int, length: int = 32) -> bytes:
        return int(value).to_bytes(length, "big", signed=False)

    def _digest_window(self, byte_window: bytes) -> bytes:
        clip = self._cnt_clip
        q = self._q_bits
        counts = [0] * 256
        if byte_window:
            for b in byte_window:
                bb = int(b) & 0xFF
                if counts[bb] < clip:
                    counts[bb] += 1
        if q:
            counts = [(c >> q) << q for c in counts]
        return bytes(counts)

    def greenlist(self, byte_window: bytes, gamma: float) -> List[int]:
        byte_window = byte_window or b""
        gamma = float(gamma)
        if not (0.0 < gamma < 1.0):
            gamma = min(max(gamma, 1e-6), 1.0 - 1e-6)
        digest = self._digest_window(byte_window)
        threshold = int(gamma * (1 << 32))
        result: List[int] = []
        for b in range(256):
            msg = b"HIST|" + digest + b"|B" + bytes([b])
            hv = hmac.new(self._key, msg, hashlib.sha256).digest()
            val = int.from_bytes(hv[:4], "big")
            if val < threshold:
                result.append(b)
        return result


class ByteKGWPRF:
    """
    KGW-style byte-domain PRF (leftHash 风格)：

    - vocab 固定为 256 个字节值 [0..255]
    - 先用 hash_key 初始化一个固定的随机排列 prf[0..255]
    - 对于窗口 window（可见字节序列），用“time”风格的 f()：
        * 取最近 prefix_length 个字节，做乘积 time_result
        * f(window) = prf[ time_result % 256 ]
    - 再用 (hash_key * f(window)) % 256 作为随机数种子，
      生成 vocab 的一个随机排列，并取前 gamma * 256 个字节作为 greenlist。
    """

    def __init__(
        self,
        hash_key: int,
        prefix_length: int,
        *,
        vocab_size: int = 256,
        device: str | torch.device = "cpu",
    ) -> None:
        self.hash_key = int(hash_key)
        self.prefix_length = max(1, int(prefix_length))
        self.vocab_size = int(vocab_size)
        self.device = torch.device(device)

        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(self.hash_key)
        self.prf = torch.randperm(self.vocab_size, generator=self.rng)

    def _f_time(self, byte_window: bytes) -> int:
        """Time-style f: multiply the last prefix_length bytes (order-sensitive)."""
        if not byte_window:
            return int(self.prf[0].item())
        time_result = 1
        n = min(self.prefix_length, len(byte_window))
        for i in range(n):
            b = int(byte_window[-1 - i]) & 0xFF
            time_result *= b if b > 0 else 1
        idx = time_result % self.vocab_size
        return int(self.prf[idx].item())

    def _seed_for_window(self, byte_window: bytes) -> int:
        fx = self._f_time(byte_window)
        return int((self.hash_key * fx) % self.vocab_size)

    def greenlist(self, byte_window: bytes, gamma: float) -> List[int]:
        window = byte_window or b""
        gamma = float(gamma)
        if not (0.0 < gamma < 1.0):
            gamma = min(max(gamma, 1e-6), 1.0 - 1e-6)

        green_size = int(self.vocab_size * gamma)
        if green_size <= 0:
            return []

        seed = self._seed_for_window(window)
        self.rng.manual_seed(seed)
        perm = torch.randperm(self.vocab_size, generator=self.rng)
        gl = perm[:green_size].tolist()
        return [int(x) for x in gl]


class ByteWindowPRF:
    """
    窗口顺序敏感的字节 PRF：

    - 只使用最近 prefix_length 个字节（默认 4）构成一个固定长度的窗口；
    - 将窗口内容按顺序直接喂给 HMAC-SHA256（key 由 hash_key 派生）；
    - 对每个候选字节 b 追加一个标记，然后取前 32bit 作为伪随机值；
    - 与 gamma 比较决定该字节是否进入 greenlist。

    相比 histogram 版本，这个实现：
    - 不做计数裁剪 / 量化，保留窗口顺序信息；
    - 每个不同的长度<=prefix_length 的窗口几乎都会映射到完全不同的 PRF 轨迹；
    - 从 PRF 角度更加“接近理想随机”。
    """

    def __init__(self, hash_key: int, prefix_length: int = 4) -> None:
        self.hash_key = int(hash_key)
        self.prefix_length = max(1, int(prefix_length))
        self._key = self._int_to_bytes(self.hash_key)

    @staticmethod
    def _int_to_bytes(value: int, length: int = 32) -> bytes:
        return int(value).to_bytes(length, "big", signed=False)

    def _norm_window(self, byte_window: bytes) -> bytes:
        """取最近 prefix_length 个字节，不足左侧补 0。"""
        w = byte_window or b""
        if len(w) >= self.prefix_length:
            w = w[-self.prefix_length :]
        else:
            pad = b"\x00" * (self.prefix_length - len(w))
            w = pad + w
        return w

    def greenlist(self, byte_window: bytes, gamma: float) -> List[int]:
        window = self._norm_window(byte_window or b"")
        gamma = float(gamma)
        if not (0.0 < gamma < 1.0):
            gamma = min(max(gamma, 1e-6), 1.0 - 1e-6)
        threshold = int(gamma * (1 << 32))

        result: List[int] = []
        for b in range(256):
            # 用窗口顺序 + byte 值构造消息
            msg = b"W|" + window + b"|B" + bytes([b])
            hv = hmac.new(self._key, msg, hashlib.sha256).digest()
            val = int.from_bytes(hv[:4], "big")
            if val < threshold:
                result.append(b)
        return result
    
import hmac, hashlib
import torch
from typing import Sequence, List

class TokenPrefixBytePRF:
    def __init__(
        self,
        *,
        hash_key: int,
        token_prefix_length: int,
        gamma: float,
        f_scheme: str = "additive",  # "time"/"additive"/"skip"/"min"
        device: str | torch.device = "cpu",
    ) -> None:
        self.hash_key = int(hash_key)
        self.h = int(token_prefix_length)
        self.gamma = float(gamma)
        self.f_scheme = str(f_scheme)
        self.device = torch.device(device)
        self.rng = torch.Generator(device=self.device)
        self._key = self.hash_key.to_bytes(32, "big", signed=False)

    def _f(self, token_ids: Sequence[int]) -> int:
        ids = list(token_ids)
        if self.h <= 0:
            return 0
        if len(ids) < self.h:
            ids = [0] * (self.h - len(ids)) + ids
        tail = ids[-self.h:]

        if self.f_scheme == "time":
            v = 1
            for x in tail:
                xi = int(x)
                v *= (xi if xi != 0 else 1)
            return v
        if self.f_scheme == "additive":
            return sum(int(x) for x in tail)
        if self.f_scheme == "skip":
            return int(tail[0])
        if self.f_scheme == "min":
            return min(int(x) for x in tail)
        raise ValueError(f"Unknown f_scheme={self.f_scheme}")

    def _seed64(self, token_ids: Sequence[int], prefix_bytes: bytes, byte_pos: int) -> int:
        fx = self._f(token_ids)

        # 关键：编码要无歧义（加 tag/pos）
        msg = (
            b"FX|" + int(fx).to_bytes(8, "big", signed=False)
            + b"|POS|" + int(byte_pos).to_bytes(2, "big", signed=False)
            + b"|PFX|" + bytes(prefix_bytes)
        )
        digest = hmac.new(self._key, msg, hashlib.sha256).digest()
        return int.from_bytes(digest[:8], "big", signed=False)
    
    def greenlist(self, token_ids: Sequence[int], prefix_bytes: bytes, byte_pos: int) -> List[int]:
        # ===== DEBUG BEGIN =====
        TAIL_N = 3
        token_ids_tail = list(token_ids[-TAIL_N:])

        # print(
        #     "[PRF greenlist DEBUG]\n"
        #     f"  token_ids_tail(len={len(token_ids_tail)}) = {token_ids_tail}\n"
        #     f"  prefix_bytes = {prefix_bytes.hex()}\n"
        #     f"  byte_pos     = {byte_pos}\n"
        # )
        # ===== DEBUG END =====

        g = min(max(self.gamma, 1e-6), 1.0 - 1e-6)
        k = int(256 * g)
        if k <= 0:
            return []

        seed = self._seed64(token_ids, prefix_bytes, byte_pos)
        self.rng.manual_seed(seed)
        perm = torch.randperm(256, generator=self.rng, device=self.device)
        return [int(x) for x in perm[:k].tolist()]


__all__ = ["ByteGreenlistPRF", "ByteKGWPRF", "ByteWindowPRF","TokenPrefixBytePRF"]
