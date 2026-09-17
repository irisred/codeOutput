from __future__ import annotations

import math
import torch
from transformers.generation.logits_process import LogitsProcessor


class ByteFactorizedLogitsWarper(LogitsProcessor):
    """
    Strict trie factorized sampler.

    Guarantees:
      - When delta == 0, the sampled token distribution is exactly the same as
        sampling once from softmax(scores) over the finite candidate set.
      - Works with multi-byte token byte strings.
      - END is treated as a real branch at every depth (critical for correctness).

    Requirements on byte_index:
      - byte_index.byte_at(token_ids, pos) -> int16 tensor, value in [0..255] or <0 for END/invalid.
        (<0 is treated as END / no further visible bytes)
    Requirements on prf:
      - prf.green_bytes_mask(ctx, prefix_bytes=..., byte_pos=pos) -> [B,256] bool
    """

    def __init__(
        self,
        prf,
        byte_index,
        *,
        delta: float,
        prefix_length: int,
        eps: float = 1e-20,
        max_prefix_bytes: int = 2048,  # safety buffer for prefix storage
    ):
        super().__init__()
        self.prf = prf
        self.byte_index = byte_index
        self.delta = float(delta)
        self.prefix_length = int(prefix_length)
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
        neg_inf = -float("inf")
        dev = scores.device

        # context: last L tokens (same as detector)
        L = self.prefix_length
        ctx = input_ids[:, -L:] if L > 0 else input_ids[:, :0]  # [B,L] or [B,0]

        # precompute exp(delta) on device (float64 for stable mass scaling)
        if self.delta == 0.0:
            exp_delta = None
        else:
            exp_delta = torch.tensor(math.exp(self.delta), device=dev, dtype=torch.float64)

        for b in range(B):
            row = scores[b]  # [V]
            ctx_b = ctx[b].unsqueeze(0)  # [1,L]

            # finite candidates after HF warpers (top_p/top_k may have masked -inf)
            cand_ids = torch.nonzero(torch.isfinite(row), as_tuple=False).view(-1).to(torch.long)
            if cand_ids.numel() == 0:
                # fallback: keep argmax
                picked = torch.argmax(row).view(())
                keep_val = row[picked].clone()
                row.fill_(neg_inf)
                row[picked] = keep_val
                continue

            # token probabilities over candidates (use float64 for correctness)
            cand_logits = row.index_select(0, cand_ids).to(torch.float64)
            cand_p = torch.softmax(cand_logits, dim=-1)  # [M] sum=1

            # active set initially is all candidates, with their original probs
            active_ids = cand_ids
            active_p = cand_p  # float64

            # prefix bytes chosen so far
            prefix_buf = torch.empty((self.max_prefix_bytes,), device=dev, dtype=torch.uint8)
            prefix_len = 0
            pos = 0

            # Main loop: sample from trie until END chosen or only one token remains
            while True:
                if active_ids.numel() == 1:
                    picked_token = active_ids[0]
                    break

                # compute byte_at for all active tokens at this pos
                bvals = self.byte_index.byte_at(active_ids, pos).to(torch.int16)  # [M], 0..255 or <0
                exact = bvals.lt(0)   # END tokens at this node
                extend = ~exact

                end_mass = active_p[exact].sum()  # float64
                if extend.any():
                    ext_ids = active_ids[extend]
                    ext_p = active_p[extend]
                    ext_b = bvals[extend].to(torch.long)  # [Me], 0..255

                    # mass per byte child
                    byte_mass = torch.zeros((256,), device=dev, dtype=torch.float64)
                    byte_mass.index_add_(0, ext_b, ext_p)
                else:
                    byte_mass = torch.zeros((256,), device=dev, dtype=torch.float64)

                total_mass = end_mass + byte_mass.sum()

                # If numerical weirdness: fallback to direct token sampling (still correct)
                if (not torch.isfinite(total_mass)) or float(total_mass.item()) <= 0.0:
                    idx = torch.multinomial(active_p, 1).view(())
                    picked_token = active_ids[idx]
                    break

                # Build 257-way branch weights: [0..255]=bytes, [256]=END
                weights = torch.empty((257,), device=dev, dtype=torch.float64)
                weights[:256] = byte_mass
                weights[256] = end_mass

                # Apply watermark bias ONLY to bytes (0..255), never to END.
                if exp_delta is not None:
                    if prefix_len > 0:
                        pfx = prefix_buf[:prefix_len].unsqueeze(0)  # [1,prefix_len]
                    else:
                        pfx = prefix_buf[:0].unsqueeze(0)  # [1,0]
                    green_mask = self.prf.green_bytes_mask(ctx_b, prefix_bytes=pfx, byte_pos=pos)  # [1,256] bool
                    if green_mask.dim() == 1:
                        green_mask = green_mask.unsqueeze(0)

                    # scale green bytes by exp(delta): w = mass * (green?exp(delta):1)
                    scale = torch.where(green_mask[0], exp_delta, torch.ones_like(exp_delta)).to(torch.float64)
                    weights[:256] = weights[:256] * scale

                # Normalize to probs and sample one branch
                probs = weights / weights.sum()
                branch = torch.multinomial(probs, 1).item()

                # END branch
                if branch == 256:
                    if exact.any():
                        p_end = active_p[exact]
                        p_end = p_end / p_end.sum()
                        idx = torch.multinomial(p_end, 1).view(())
                        picked_token = active_ids[exact][idx]
                    else:
                        # Should be rare (end_mass=0 but picked END due to numerical), fallback
                        idx = torch.multinomial(active_p, 1).view(())
                        picked_token = active_ids[idx]
                    break

                # Byte child branch
                chosen_byte = int(branch)
                if prefix_len >= self.max_prefix_bytes:
                    # Safety fallback: sample token directly (keeps distribution on remaining active set)
                    idx = torch.multinomial(active_p, 1).view(())
                    picked_token = active_ids[idx]
                    break
                prefix_buf[prefix_len] = chosen_byte
                prefix_len += 1

                # filter to tokens that extend and match this byte at current pos
                keep = extend & (bvals.to(torch.long) == chosen_byte)
                if not keep.any():
                    # numerical corner: fallback direct sampling
                    idx = torch.multinomial(active_p, 1).view(())
                    picked_token = active_ids[idx]
                    break

                active_ids = active_ids[keep]
                active_p = active_p[keep]
                active_p = active_p / active_p.sum()

                pos += 1

            # Collapse distribution to chosen token (keep its original logit value)
            keep_val = row[picked_token].clone()
            row.fill_(neg_inf)
            row[picked_token] = keep_val

        return scores


__all__ = ["ByteFactorizedLogitsWarper"]
