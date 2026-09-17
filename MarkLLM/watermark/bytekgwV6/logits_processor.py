from __future__ import annotations

import hmac
import hashlib
import torch
from torch import Tensor
from transformers import LogitsProcessor

from .prf import RobustPartitioner
from .token_bytes import TokenByteVocabV6


class ByteKGWv6LogitsProcessor(LogitsProcessor):
    """
    Simple logits biaser for ByteKGWv6.

    - Bias rule: if token's first-n visible bytes are GREEN => logits + delta.
    - GREEN is decided by RobustPartitioner(fingerprint(last_n_chars_of_text)).
    - Tokens that share the same first-n bytes share the same color.

    Notes:
      * Works on CPU or CUDA (all torch ops); delta applied in-place to scores.
      * We precompute per-unique first-n-byte id:
          - PRF token vectors (uint8) for fast Hamming-distance scoring
          - tie-break bits for exact-half cases
      * Per-call cost: one fingerprint decode + XOR+popcount over unique ids.
    """

    def __init__(
        self,
        *,
        tokenizer,
        vocab: TokenByteVocabV6,
        partitioner: RobustPartitioner,
        delta: float,
        n_bytes: int = 1,
        seed_window_chars: int = 512,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.partitioner = partitioner
        self.delta = float(delta)
        self.n_bytes = max(1, int(n_bytes))
        self.seed_window_chars = max(1, int(seed_window_chars))
        self.device = torch.device(device)

        # map tokens -> first-n-byte id (base-257 with END_BYTE sentinel)
        self.firstn_ids = vocab.first_n_id(self.device, self.n_bytes)  # [V] int64

        # unique ids to reduce PRF work
        uniq, inv = torch.unique(self.firstn_ids.cpu(), return_inverse=True)
        self._uniq_ids = uniq  # CPU int64
        self._inv = inv.to(self.device)  # [V] on target device

        # precompute token vectors and tie bits per unique id
        w_list = []
        tie_list = []
        for uid in uniq.tolist():
            w_bytes = partitioner.token_vector(uid)
            tie = hmac.new(partitioner.k_tok, b"tie" + uid.to_bytes(8, "little", signed=False), hashlib.sha256).digest()[0] & 1
            tie_list.append(tie)
            w_list.append(torch.tensor(list(w_bytes), dtype=torch.uint8))
        self._w = torch.stack(w_list, dim=0) if w_list else torch.empty((0, partitioner.m_bytes), dtype=torch.uint8)
        self._tie = torch.tensor(tie_list, dtype=torch.uint8)

        # popcount LUT on device
        self._lut = torch.tensor([bin(i).count("1") for i in range(256)], dtype=torch.uint8)

        self.half_bits = partitioner.m_bits // 2
        self.to(self.device)

    def to(self, device: str | torch.device) -> "ByteKGWv6LogitsProcessor":
        dev = torch.device(device)
        self.device = dev
        self.firstn_ids = self.firstn_ids.to(dev)
        self._inv = self._inv.to(dev)
        self._w = self._w.to(dev)
        self._tie = self._tie.to(dev)
        self._lut = self._lut.to(dev)
        return self

    def _greens_for_fingerprint(self, fp: bytes, *, device: torch.device) -> Tensor:
        """
        Compute green mask over unique first-n ids for a given fingerprint.
        returns: [U] bool on device
        """
        if not fp:
            fp = bytes([0] * self.partitioner.m_bytes)
        fp_t = torch.tensor(list(fp), device=device, dtype=torch.uint8)  # [m_bytes]
        xor = torch.bitwise_xor(self._w, fp_t.unsqueeze(0))  # [U,m_bytes]
        hd = torch.take(self._lut, xor.view(-1).to(torch.long)).view(xor.shape).sum(dim=1)  # [U]

        green = hd < self.half_bits
        tie_mask = hd == self.half_bits
        if tie_mask.any():
            tie_bits = self._tie[tie_mask] & 1
            green[tie_mask] = tie_bits == 0
        return green

    @torch.no_grad()
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.delta == 0.0:
            return scores

        dev = scores.device
        if self.device != dev:
            self.to(dev)
        if input_ids.device != dev:
            input_ids = input_ids.to(dev)

        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        B = input_ids.size(0)

        for b in range(B):
            ids = input_ids[b].tolist()
            text = self.tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            fp = self.partitioner.fingerprint(text, self.seed_window_chars)
            green_unique = self._greens_for_fingerprint(fp, device=scores.device)  # [U] bool

            # map back to vocab
            green_tokens = green_unique[self._inv]  # [V] bool
            bias = green_tokens.to(dtype=scores.dtype) * self.delta
            scores[b].add_(bias)

        return scores


__all__ = ["ByteKGWv6LogitsProcessor"]
