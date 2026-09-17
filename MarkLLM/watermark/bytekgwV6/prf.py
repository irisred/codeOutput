# prf.py
from __future__ import annotations

import hashlib
import hmac
import math
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Set, Tuple, Union

# -------------------------
# low-level helpers
# -------------------------

def _u64(b: bytes) -> int:
    # interpret first 8 bytes as little-endian unsigned
    return int.from_bytes(b[:8], "little", signed=False)

def xor_bytes(a: bytes, b: bytes) -> bytes:
    return bytes(x ^ y for x, y in zip(a, b))

def popcount_bytes(x: bytes) -> int:
    # Python 3.8+: int.bit_count exists on int, not bytes
    # Use per-byte bit_count for clarity
    return sum(byte.bit_count() for byte in x)

def _collapse_ws(s: str) -> str:
    # keep it conservative: collapse any whitespace run into a single space
    return re.sub(r"\s+", " ", s)

def _to_master_key(master_key: Union[bytes, int, str]) -> bytes:
    if isinstance(master_key, bytes):
        return master_key
    if isinstance(master_key, int):
        # stable across machines
        return master_key.to_bytes(8, "little", signed=False)
    if isinstance(master_key, str):
        return master_key.encode("utf-8", errors="ignore")
    raise TypeError(f"Unsupported master_key type: {type(master_key)}")


# -------------------------
# RobustPartitioner
# -------------------------

@dataclass(frozen=True)
class RobustPartitionerConfig:
    hash_key: int
    m_bits: int = 256
    target_anchors: int = 96
    k_choices: Tuple[int, ...] = (1, 2, 3, 4, 5, 6)
    normalize_whitespace: bool = True
    k_weight_mode: str = "linear"          # "none" | "linear"
    decision_margin_bits: int = 12         # 0 disables margin band


class RobustPartitioner:
    """
    Robust PRF partitioner for ByteKGWv6-style schemes.

    Main knobs:
      - k_choices: include smaller k to increase feature count under fixed window length
      - k_weight_mode:
          * "none": all k grams weight=1
          * "linear": weight = (k_max + 1 - k), so smaller k gets larger weight (more robust)
      - decision_margin_bits:
          * 0 -> legacy: green iff hd < m/2, tie at hd==m/2
          * >0 -> green iff hd <= m/2 - tau, red iff hd >= m/2 + tau, band uses stable tie-bit
    """

    def __init__(
        self,
        master_key: Union[bytes, int, str],
        *,
        m_bits: int = 256,
        target_anchors: int = 96,
        k_choices: Tuple[int, ...] = (1, 2, 3, 4, 5, 6),
        normalize_whitespace: bool = True,
        k_weight_mode: str = "linear",
        decision_margin_bits: int = 12,
    ) -> None:
        mk = _to_master_key(master_key)

        self.m_bits = int(m_bits)
        if self.m_bits <= 0 or self.m_bits % 8 != 0:
            raise ValueError("m_bits must be positive and a multiple of 8 (e.g., 256).")
        self.m_bytes = self.m_bits // 8

        self.target_anchors = int(target_anchors)
        self.k_choices = tuple(int(k) for k in k_choices if int(k) > 0)
        if len(self.k_choices) == 0:
            raise ValueError("k_choices must contain at least one positive integer.")
        self.k_max = max(self.k_choices)

        self.normalize_whitespace = bool(normalize_whitespace)
        self.k_weight_mode = str(k_weight_mode)
        self.decision_margin_bits = max(0, int(decision_margin_bits))

        # subkeys
        self.k_tok = hmac.new(mk, b"tok", hashlib.sha256).digest()     # token vectors / tie bits
        self.k_gram = hmac.new(mk, b"gram", hashlib.sha256).digest()   # blake2b keyed hash
        self.k_sim = hmac.new(mk, b"sim", hashlib.sha256).digest()     # expand per-anchor bits

    @classmethod
    def from_config(cls, cfg: Dict) -> "RobustPartitioner":
        # accept your existing json shape
        hash_key = cfg.get("hash_key", cfg.get("master_key", 0))
        if isinstance(hash_key, (bytes, str)):
            # allow but prefer int
            mk = hash_key
        else:
            mk = int(hash_key)

        return cls(
            mk,
            m_bits=int(cfg.get("m_bits", 256)),
            target_anchors=int(cfg.get("target_anchors", 96)),
            k_choices=tuple(cfg.get("k_choices", [1, 2, 3, 4, 5, 6])),
            normalize_whitespace=bool(cfg.get("normalize_whitespace", True)),
            k_weight_mode=str(cfg.get("k_weight_mode", "linear")),
            decision_margin_bits=int(cfg.get("decision_margin_bits", 12)),
        )

    # -------------------------
    # token-side PRF primitives
    # -------------------------

    def token_vector(self, token_id: int) -> bytes:
        """
        Deterministic pseudo-random bit-vector W(token_id) of length m_bits, as bytes of length m_bytes.
        """
        tid = int(token_id)
        tid_b = tid.to_bytes(8, "little", signed=False)
        out = bytearray()
        ctr = 0
        while len(out) < self.m_bytes:
            msg = b"tv" + tid_b + ctr.to_bytes(2, "little", signed=False)
            out.extend(hmac.new(self.k_tok, msg, hashlib.sha256).digest())
            ctr += 1
        return bytes(out[: self.m_bytes])

    def _stable_bit(self, label: bytes, token_id: int) -> int:
        tid = int(token_id).to_bytes(8, "little", signed=False)
        return hmac.new(self.k_tok, label + tid, hashlib.sha256).digest()[0] & 1

    # -------------------------
    # fingerprint (SimHash)
    # -------------------------

    def _k_weight(self, k: int) -> int:
        if self.k_weight_mode == "linear":
            # smaller k => larger weight
            return max(1, (self.k_max + 1 - int(k)))
        return 1

    def _sample_anchors(self, seed: str) -> Set[Tuple[int, bytes]]:
        """
        Return a set of (k, h8) anchors where h8 is an 8-byte keyed hash of the k-gram.
        Mask-sampling keeps expected anchor count around target_anchors.
        """
        anchors: Set[Tuple[int, bytes]] = set()
        if not seed:
            return anchors

        total_grams = 0
        for k in self.k_choices:
            if len(seed) >= k:
                total_grams += (len(seed) - k + 1)
        total_grams = max(total_grams, 1)

        ratio = total_grams / max(self.target_anchors, 1)
        r = 0 if ratio <= 1 else int(math.ceil(math.log2(ratio)))
        mask = (1 << r) - 1

        for k in self.k_choices:
            if len(seed) < k:
                continue
            for i in range(0, len(seed) - k + 1):
                gram = seed[i : i + k].encode("utf-8", errors="ignore")
                h = hashlib.blake2b(gram, key=self.k_gram, digest_size=8).digest()
                if (_u64(h) & mask) == 0:
                    anchors.add((k, h))

        # fallback: ensure not too few anchors
        if len(anchors) < 8:
            whole = seed.encode("utf-8", errors="ignore")
            h = hashlib.blake2b(whole, key=self.k_gram, digest_size=8).digest()
            anchors.add((self.k_max, h))

        return anchors

    def fingerprint(self, prefix_text: str, seed_window_chars: int) -> bytes:
        """
        SimHash-like fingerprint of the last `seed_window_chars` characters of prefix_text.
        """
        L = max(1, int(seed_window_chars))
        tail = prefix_text[-L:] if prefix_text else ""

        if self.normalize_whitespace:
            tail = _collapse_ws(tail)

        anchors = self._sample_anchors(tail)

        # counters over m_bits
        counters = [0] * self.m_bits

        # For each anchor, expand to m_bytes pseudorandom bytes; update counters with weight
        for k, a in anchors:
            w = self._k_weight(k)

            buf = bytearray()
            ctr = 0
            while len(buf) < self.m_bytes:
                buf.extend(hmac.new(self.k_sim, a + ctr.to_bytes(2, "little", signed=False), hashlib.sha256).digest())
                ctr += 1
            buf = buf[: self.m_bytes]

            bit_index = 0
            for byte in buf:
                # LSB-first to match earlier implementations
                for b in range(8):
                    bit = (byte >> b) & 1
                    counters[bit_index] += w if bit else -w
                    bit_index += 1
                    if bit_index >= self.m_bits:
                        break
                if bit_index >= self.m_bits:
                    break

        # pack bits: bit=1 if counter>=0 (ties to 1)
        out = bytearray(self.m_bytes)
        for i in range(self.m_bits):
            if counters[i] >= 0:
                out[i // 8] |= (1 << (i % 8))
        return bytes(out)

    # -------------------------
    # partition decision
    # -------------------------

    def is_green(self, token_id: int, fingerprint: bytes) -> bool:
        """
        Decide green/red for this token_id under given fingerprint.
        """
        w = self.token_vector(int(token_id))
        x = xor_bytes(w, fingerprint)
        hd = popcount_bytes(x)

        half = self.m_bits // 2
        tau = self.decision_margin_bits

        if tau > 0:
            # hard regions
            if hd <= (half - tau):
                return True
            if hd >= (half + tau):
                return False
            # band region: stable per-token bit (independent of fingerprint)
            return self._stable_bit(b"band", int(token_id)) == 0

        # legacy
        if hd != half:
            return hd < half
        return self._stable_bit(b"tie", int(token_id)) == 0


__all__ = ["RobustPartitioner", "RobustPartitionerConfig"]
