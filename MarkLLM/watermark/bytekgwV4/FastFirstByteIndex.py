from __future__ import annotations

from typing import Union
import torch


class FastFirstByteIndex:
    """
    GPU wrapper exposing:
      - firstbyte_vocab_tensor: [V] int16 on CUDA, 0..255 or -1
      - firstbyte(token_ids): fast gather lookup
    """

    def __init__(self, token_byte_vocab, device: Union[str, torch.device], *, max_scan_iters: int = 32):
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("FastFirstByteIndex is CUDA-only.")

        self.token_byte_vocab = token_byte_vocab
        self.vocab_size = int(token_byte_vocab.vocab_size)

        token_byte_vocab._ensure_views()
        self.firstbyte_vocab_tensor = token_byte_vocab.firstbyte_tensor(
            self.device, max_scan_iters=int(max_scan_iters)
        )  # [V] int16, 0..255 or -1

    def firstbyte(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.device != self.device:
            token_ids = token_ids.to(self.device)
        return self.firstbyte_vocab_tensor[token_ids]

    def byte_at(self, token_ids: torch.Tensor, byte_pos: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("byte_at is reserved for future multi-byte positions.")
