from __future__ import annotations

import torch
from typing import Union


class FastVisibleByteIndex:
    """
    CUDA-only fast lookup for visible bytes.

    Single source of truth:
      - TokenByteVocab.byte_table_tensor(device, max_depth, max_scan_iters, [dtype])
      - TokenByteVocab.firstbyte_tensor(device, max_scan_iters, [dtype])

    Visible-byte policy is defined ONLY by TokenByteVocab.
    (You said: skip Ġ/▁/space prefixes; continuation bytes are still valid.)
    """

    def __init__(
        self,
        token_byte_vocab,
        device: Union[str, torch.device] = "cuda:0",
        *,
        max_depth: int = 8,
        max_scan_iters: int = 64,
        table_dtype: torch.dtype = torch.int16,
    ) -> None:
        dev = torch.device(device)
        if dev.type != "cuda":
            raise ValueError("FastVisibleByteIndex is CUDA-only: device must be CUDA.")
        # canonicalize "cuda" -> "cuda:<current>"
        if dev.index is None:
            dev = torch.device("cuda", torch.cuda.current_device())
        self.device = dev

        self.token_byte_vocab = token_byte_vocab
        self.vocab_size = int(token_byte_vocab.vocab_size)

        self.max_depth = int(max_depth)
        self.max_scan_iters = int(max_scan_iters)
        if self.max_depth <= 0:
            raise ValueError("max_depth must be > 0.")

        # Optional: keep compatibility with old TokenByteVocab implementations
        if hasattr(token_byte_vocab, "_ensure_views"):
            token_byte_vocab._ensure_views()

        # Build/cache on GPU.
        # Support both signatures:
        #   - new: byte_table_tensor(device, max_depth=..., max_scan_iters=...)
        #   - old: byte_table_tensor(device, max_depth=..., max_scan_iters=..., dtype=...)
        self._byte_table = self._call_byte_table_tensor(table_dtype)
        self._firstbyte = self._call_firstbyte_tensor(table_dtype)

    def _call_byte_table_tensor(self, dtype: torch.dtype) -> torch.Tensor:
        fn = self.token_byte_vocab.byte_table_tensor
        try:
            t = fn(
                self.device,
                max_depth=self.max_depth,
                max_scan_iters=self.max_scan_iters,
                dtype=dtype,
            )
        except TypeError:
            t = fn(
                self.device,
                max_depth=self.max_depth,
                max_scan_iters=self.max_scan_iters,
            )
            if t.dtype != dtype:
                t = t.to(dtype)
        return t

    def _call_firstbyte_tensor(self, dtype: torch.dtype) -> torch.Tensor:
        fn = self.token_byte_vocab.firstbyte_tensor
        try:
            t = fn(
                self.device,
                max_scan_iters=self.max_scan_iters,
                dtype=dtype,
            )
        except TypeError:
            t = fn(
                self.device,
                max_scan_iters=self.max_scan_iters,
            )
            if t.dtype != dtype:
                t = t.to(dtype)
        return t

    # -------------------------
    # basic accessors
    # -------------------------
    def byte_table(self) -> torch.Tensor:
        """Return cached [V,D] visible-byte table (CUDA int16)."""
        return self._byte_table

    def firstbyte_vocab_tensor(self) -> torch.Tensor:
        """Return cached [V] firstbyte tensor (CUDA int16)."""
        return self._firstbyte

    # -------------------------
    # internal: device check
    # -------------------------
    def _check_cuda_device(self, x: torch.Tensor, name: str) -> None:
        if x.device.type != "cuda":
            raise ValueError(f"{name} must be a CUDA tensor.")
        if x.device.index != self.device.index:
            raise ValueError(
                f"{name} must be on {self.device}, but got {x.device}. "
                f"(Fix by initializing FastVisibleByteIndex with device={x.device}.)"
            )

    # -------------------------
    # lookup ops
    # -------------------------
    @torch.no_grad()
    def firstbyte(self, token_ids: torch.Tensor) -> torch.Tensor:
        self._check_cuda_device(token_ids, "token_ids")
        if token_ids.dtype != torch.long:
            token_ids = token_ids.to(torch.long)
        return self._firstbyte[token_ids]

    @torch.no_grad()
    def byte_at(self, token_ids: torch.Tensor, pos: Union[int, torch.Tensor]) -> torch.Tensor:
        """
        token_ids: CUDA tensor [...]
        pos:
          - int scalar
          - or CUDA tensor [...] same shape as token_ids
        returns: CUDA int16 tensor [...] in 0..255 or -1
        """
        self._check_cuda_device(token_ids, "token_ids")
        if token_ids.dtype != torch.long:
            token_ids = token_ids.to(torch.long)

        D = self.max_depth
        table = self._byte_table  # [V,D]

        # scalar pos path
        if isinstance(pos, int):
            p = int(pos)
            if p < 0 or p >= D:
                return torch.full_like(token_ids, -1, dtype=table.dtype, device=self.device)
            return table[token_ids, p]

        # tensor pos path
        if not torch.is_tensor(pos):
            raise TypeError("pos must be int or torch.Tensor.")
        self._check_cuda_device(pos, "pos")
        if pos.shape != token_ids.shape:
            raise ValueError("pos must have the same shape as token_ids.")

        pos_i64 = pos.to(torch.int64)
        in_range = (pos_i64 >= 0) & (pos_i64 < D)
        pos_safe = torch.clamp(pos_i64, 0, D - 1)

        out = table[token_ids, pos_safe]  # elementwise indexing
        out = torch.where(in_range, out, torch.full_like(out, -1, dtype=table.dtype))
        return out

    @torch.no_grad()
    def visible_len(self, token_ids: torch.Tensor) -> torch.Tensor:
        """visible_len = count(byte_table[token] >= 0) (<=max_depth)"""
        self._check_cuda_device(token_ids, "token_ids")
        if token_ids.dtype != torch.long:
            token_ids = token_ids.to(torch.long)
        rows = self._byte_table[token_ids]  # [..., D]
        return (rows >= 0).sum(dim=-1, dtype=torch.int64)

    # -------------------------
    # PRF helper: prefix bytes
    # -------------------------
    @torch.no_grad()
    def prefix_bytes_tensor(
        self,
        token_ids: torch.Tensor,
        byte_pos: Union[int, torch.Tensor],
        *,
        pad_value: int = -1,
    ) -> torch.Tensor:
        """
        If byte_pos is int p:
          returns shape token_ids.shape + (p,) containing bytes [0..p-1] (visible bytes).
        If byte_pos is tensor same shape as token_ids:
          returns [N, D] (flattened) padded with pad_value, mask col < byte_pos.
        """
        self._check_cuda_device(token_ids, "token_ids")
        if token_ids.dtype != torch.long:
            token_ids = token_ids.to(torch.long)

        table = self._byte_table
        D = self.max_depth
        dtype = table.dtype

        if isinstance(byte_pos, int):
            p = int(byte_pos)
            if p <= 0:
                shape = list(token_ids.shape) + [0]
                return torch.empty(shape, device=self.device, dtype=dtype)
            p = min(p, D)
            return table[token_ids, :p]

        if not torch.is_tensor(byte_pos):
            raise TypeError("byte_pos must be int or torch.Tensor.")
        self._check_cuda_device(byte_pos, "byte_pos")
        if byte_pos.shape != token_ids.shape:
            raise ValueError("byte_pos must have the same shape as token_ids.")

        tok_flat = token_ids.reshape(-1)
        pos_flat = byte_pos.to(torch.int64).reshape(-1)  # [N]
        rows = table[tok_flat]  # [N,D]

        col = torch.arange(D, device=self.device, dtype=torch.int64)[None, :]  # [1,D]
        mask = col < pos_flat[:, None]  # [N,D]

        pad = torch.full((tok_flat.numel(), D), int(pad_value), device=self.device, dtype=dtype)
        return torch.where(mask, rows, pad)
