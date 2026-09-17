from __future__ import annotations

import math
import torch
from transformers.generation.logits_process import LogitsProcessor


class ByteFactorizedLogitsWarper(LogitsProcessor):
    """
    Strict trie factorized sampler (CUDA-only).

    Compatibility:
      - Accepts legacy kwargs: max_byte_pos, eps (even if not used for truncation).
      - Keeps ByteKGW(bytekgw.py) constructor unchanged.

    Correctness:
      - When delta == 0, the resulting token distribution is EXACTLY equal to
        sampling once from softmax(scores) over the finite candidate set.

    Requirements:
      - byte_index.byte_at(token_ids, pos) -> int tensor, values 0..255 or <0 meaning END/invalid/no more visible bytes.
      - prf.green_bytes_mask(ctx, prefix_bytes=..., byte_pos=pos) -> [B,256] bool
    """

    def __init__(
        self,
        prf,
        byte_index,
        *,
        delta: float,
        prefix_length: int,
        max_byte_pos: int | None = None,  # legacy/compat (NOT used to truncate, to preserve exactness)
        eps: float = 1e-20,
        max_prefix_bytes: int = 2048,
    ):
        super().__init__()
        self.prf = prf
        self.byte_index = byte_index
        self.delta = float(delta)
        self.prefix_length = int(prefix_length)

        # compat: keep but don't truncate when strictness is required
        self.max_byte_pos = None if max_byte_pos is None else int(max_byte_pos)

        self.eps = float(eps)
        self.max_prefix_bytes = int(max_prefix_bytes)

    @torch.no_grad()
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if input_ids.device.type != "cuda" or scores.device.type != "cuda":
            raise ValueError("ByteFactorizedLogitsWarper is CUDA-only.")
        if input_ids.dtype != torch.long:
            input_ids = input_ids.to(torch.long)
        if input_ids.device != scores.device:
            input_ids = input_ids.to(scores.device)

        B, V = scores.shape
        dev = scores.device
        neg_inf = -float("inf")

        # Context window (aligned with detector)
        L = self.prefix_length
        ctx = input_ids[:, -L:] if L > 0 else input_ids[:, :0]  # [B,L] or [B,0]

        # exp(delta) only when delta != 0
        exp_delta = None
        if self.delta != 0.0:
            exp_delta = torch.tensor(math.exp(self.delta), device=dev, dtype=torch.float64)

        for b in range(B):
            row = scores[b]  # [V]
            ctx_b = ctx[b].unsqueeze(0)  # [1,L]

            # candidate set after HF warpers (top_p/top_k may set -inf)
            cand_ids = torch.nonzero(torch.isfinite(row), as_tuple=False).view(-1).to(torch.long)
            if cand_ids.numel() == 0:
                picked = torch.argmax(row).view(())
                keep_val = row[picked].clone()
                row.fill_(neg_inf)
                row[picked] = keep_val
                continue

            # token probs over candidates (float64 for exact mass accounting)
            cand_logits = row.index_select(0, cand_ids).to(torch.float64)
            cand_p = torch.softmax(cand_logits, dim=-1)  # [M], sum=1

            active_ids = cand_ids
            active_p = cand_p

            # prefix bytes buffer
            prefix_buf = torch.empty((self.max_prefix_bytes,), device=dev, dtype=torch.uint8)
            prefix_len = 0
            pos = 0

            while True:
                if active_ids.numel() == 1:
                    picked_token = active_ids[0]
                    break

                # If someone insists on truncation, note: truncation breaks strict equivalence.
                # We ignore max_byte_pos for strict correctness.
                # (kept only for compat / config files)
                bvals = self.byte_index.byte_at(active_ids, pos).to(torch.int16)  # [M]
                exact = bvals.lt(0)     # END at this node
                extend = ~exact

                end_mass = active_p[exact].sum()  # float64
                if extend.any():
                    ext_p = active_p[extend]
                    ext_b = bvals[extend].to(torch.long)  # 0..255

                    byte_mass = torch.zeros((256,), device=dev, dtype=torch.float64)
                    byte_mass.index_add_(0, ext_b, ext_p)
                else:
                    byte_mass = torch.zeros((256,), device=dev, dtype=torch.float64)

                total_mass = end_mass + byte_mass.sum()
                if (not torch.isfinite(total_mass)) or float(total_mass.item()) <= 0.0:
                    # fallback to direct sampling on remaining active set
                    idx = torch.multinomial(active_p, 1).view(())
                    picked_token = active_ids[idx]
                    break

                # 257-way weights: 0..255 bytes + 256 END
                weights = torch.empty((257,), device=dev, dtype=torch.float64)
                weights[:256] = byte_mass
                weights[256] = end_mass

                # Apply watermark bias only to bytes (END never biased)
                if exp_delta is not None:
                    if prefix_len > 0:
                        pfx = prefix_buf[:prefix_len].unsqueeze(0)  # [1,prefix_len]
                    else:
                        pfx = prefix_buf[:0].unsqueeze(0)  # [1,0]

                    green_mask = self.prf.green_bytes_mask(ctx_b, prefix_bytes=pfx, byte_pos=pos)  # [1,256] bool
                    if green_mask.dim() == 1:
                        green_mask = green_mask.unsqueeze(0)

                    scale = torch.where(
                        green_mask[0],
                        exp_delta,
                        torch.ones((1,), device=dev, dtype=torch.float64),
                    )
                    weights[:256] = weights[:256] * scale

                # sample one branch
                probs = weights / weights.sum()
                branch = int(torch.multinomial(probs, 1).item())

                if branch == 256:
                    # END: sample among exact tokens proportionally to their original probs
                    if exact.any():
                        p_end = active_p[exact]
                        p_end = p_end / p_end.sum()
                        idx = torch.multinomial(p_end, 1).view(())
                        picked_token = active_ids[exact][idx]
                    else:
                        # numerical edge; fallback
                        idx = torch.multinomial(active_p, 1).view(())
                        picked_token = active_ids[idx]
                    break

                # choose byte branch
                chosen_byte = branch
                if prefix_len >= self.max_prefix_bytes:
                    idx = torch.multinomial(active_p, 1).view(())
                    picked_token = active_ids[idx]
                    break

                prefix_buf[prefix_len] = chosen_byte
                prefix_len += 1

                keep = extend & (bvals.to(torch.long) == chosen_byte)
                if not keep.any():
                    idx = torch.multinomial(active_p, 1).view(())
                    picked_token = active_ids[idx]
                    break

                active_ids = active_ids[keep]
                active_p = active_p[keep]
                active_p = active_p / active_p.sum()

                pos += 1

            # collapse logits to the chosen token
            keep_val = row[picked_token].clone()
            row.fill_(neg_inf)
            row[picked_token] = keep_val

        return scores


__all__ = ["ByteFactorizedLogitsWarper"]
