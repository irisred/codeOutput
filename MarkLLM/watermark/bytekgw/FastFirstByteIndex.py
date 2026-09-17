import torch
from typing import Union


class FastFirstByteIndex:
    """
    Thin GPU wrapper that exposes a vocab-sized firstbyte tensor and fast lookup.

    IMPORTANT:
      - Does NOT re-implement firstbyte logic.
      - Uses TokenByteVocab.firstbyte_tensor(device) as the single source of truth.
      - Ensures watermark + detector share identical firstbyte behavior.
    """

    def __init__(
        self,
        token_byte_vocab,
        device: Union[str, torch.device],
        *,
        max_scan_iters: int = 32,
    ):
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("FastFirstByteIndex is CUDA-only: device must be CUDA.")

        self.token_byte_vocab = token_byte_vocab
        self.vocab_size = int(token_byte_vocab.vocab_size)

        # Make sure TokenByteVocab has built views (optional but safe)
        token_byte_vocab._ensure_views()

        # Unified, cached on TokenByteVocab side; stays on GPU
        self.firstbyte_vocab_tensor = token_byte_vocab.firstbyte_tensor(
            self.device,
            max_scan_iters=int(max_scan_iters),
        )  # int16 [V], values 0..255 or -1

    def firstbyte(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        token_ids: tensor on CUDA (same device).
        returns: firstbyte tensor with same shape as token_ids (int16).
        """
        if token_ids.device != self.device:
            raise ValueError("token_ids must be on the same CUDA device as FastFirstByteIndex.")
        return self.firstbyte_vocab_tensor[token_ids]

    def byte_at(self, token_ids: torch.Tensor, byte_pos: torch.Tensor) -> torch.Tensor:
        """
        Placeholder for future: return the byte at position byte_pos for each token_id.
        """
        raise NotImplementedError
