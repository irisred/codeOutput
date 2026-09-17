from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Sequence, Tuple, Optional

import torch
from torch import Tensor

from MarkLLM.charm_v2.model import CharmGenerationModel
from MarkLLM.charm_v2.vocab import ByteVocab

# group ID 定义，每个候选 token 在展开为 bytes 时，某个 byte 被抽象为一个 “组（group）”。
GROUP_EOS = 256       # EOS 组（当 token 本身即是 eos_token）
NUM_GROUPS = 257      # 0-255 为普通 byte，256 EOS


@dataclass
class ByteCandidate:
    """
    表示一个候选 token 序列（tokens），并跟踪其：
    - 当前 prefix 的概率（prob）
    - 当前 token 的 bytes（last_bytes）
    - offset 用于指示我们已经消耗了多少 bytes
    """

    tokens: Tuple[int, ...]  # Token 序列（从初始 prompt 之后的 continuation）
    prob: float              # 当前候选路径的概率（未归一化或归一化后）
    last_token: int          # 当前最后一个 token ID
    last_bytes: bytes        # 最后一个 token 转换成的 byte 序列
    offset: int = 0          # 用于指示当前 byte 已消费到的位置

    def next_byte(self) -> Optional[int]:
        """返回下一个未消费的 byte，如果没有则返回 None"""
        if self.offset < len(self.last_bytes):
            return int(self.last_bytes[self.offset])
        return None

    def at_boundary(self) -> bool:
        """判断 last_bytes 是否已消耗完"""
        return self.offset >= len(self.last_bytes)


class CharmByteGenerator:
    """
    基于 CharmGenerationModel 的 byte-level 生成器。

    它的核心思想是：
    - pool 里存储多个 ByteCandidate（token path + 当前未完成的 bytes）
    - 每一轮按 byte 进行采样（不是直接 token 采样）
    - 当某个 candidate 的 byte 消耗完时，会“扩展边界”，就是让模型再预测下一 token
    - 保持 top-k pool，类似 beam search + byte 级展开组合的变体
    """

    def __init__(
        self,
        charm_model: CharmGenerationModel,
        byte_vocab: ByteVocab,
        tokenizer,
    ):
        self.charm_model = charm_model
        self.byte_vocab = byte_vocab
        self.tokenizer = tokenizer

    def generate_token(
        self,
        prompt_inputs: Dict[str, Tensor],
        gen_kwargs: Dict[str, Any],
        charm_cfg,
        *,
        force_max_tokens: bool = False,
        logits_processor=None,
    ) -> Tensor:
        """
        Token-level reference implementation that replays HF generate() step-by-step.

        Updated for TokenPrefixBytePRF:
        - Do NOT compute/maintain byte_window.
        - Do NOT pass prefix_length (PRF already has token_prefix_length in __init__).
        - Pass full current token_ids to sampler; PRF/processor decides how many to use.
        """
        device = next(self.charm_model.model.parameters()).device

        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if force_max_tokens:
            eos_token_id = None

        max_new_tokens = int(gen_kwargs.get("max_new_tokens", 128))
        do_sample = bool(getattr(self.charm_model.generation_config, "do_sample", True))

        prompt_ids = prompt_inputs["input_ids"].to(device)

        for _ in range(max_new_tokens):
            logits = self.charm_model.compute_logits_via_generate(prompt_ids)
            logits = logits.to(torch.float32)

            if force_max_tokens and getattr(self.tokenizer, "eos_token_id", None) is not None:
                logits[self.tokenizer.eos_token_id] = float("-inf")

            if do_sample:
                sampler = (
                    self._sample_token_via_first_byte
                    if getattr(charm_cfg, "first_byte_only_bias", False)
                    else self._sample_token_via_prefix_factorization
                )

                if logits_processor is not None:
                    # ✅ 传当前“全部 token ids”（prompt + 已生成）
                    full_token_ids = prompt_ids[0].tolist()

                    next_token = sampler(
                        logits,
                        logits_processor=logits_processor,
                        token_ids=full_token_ids,  # ✅ 不截断
                    )
                else:
                    next_token = sampler(logits)
            else:
                next_token = int(torch.argmax(logits, dim=-1).item())

            next_token_tensor = torch.tensor([[next_token]], dtype=prompt_ids.dtype, device=device)
            prompt_ids = torch.cat([prompt_ids, next_token_tensor], dim=1)

            if eos_token_id is not None and next_token == eos_token_id:
                break

        return prompt_ids



    def _sample_token_via_prefix_factorization(
        self,
        logits: Tensor,
        *,
        logits_processor=None,
        token_ids: Optional[Sequence[int]] = None,   # ✅ 新增：全量 token ids
    ) -> int:
        """
        Prefix-factorization sampler.

        Updated for TokenPrefixBytePRF:
        - token_ids: pass full current token sequence (prompt + generated so far).
        - At each byte decision, call logits_processor with:
            (token_ids, prefix_bytes, byte_pos)
        - No byte_window / prefix_length arguments anymore.
        """
        device = logits.device
        probs = torch.softmax(logits.to(torch.float64), dim=-1)
        eps = 1e-12
        byte_data = self.byte_vocab.byte_data.to(device)
        offsets = self.byte_vocab.offsets.to(device)

        full_token_ids = list(token_ids or [])  # ✅ 原样传下去；截断在 PRF 内部做

        # 1) 收集候选 token
        mask = probs > eps
        candidate_idx = torch.nonzero(mask, as_tuple=False).view(-1)
        if candidate_idx.numel() == 0:
            return int(torch.argmax(logits, dim=-1).item())

        cand_ids = candidate_idx.to(torch.long)
        cand_probs = probs.index_select(0, cand_ids)
        token_offsets = offsets.index_select(0, cand_ids)
        token_lens = offsets.index_select(0, cand_ids + 1) - token_offsets

        def _sample_from_mask(mask_tensor: torch.Tensor) -> int:
            idx = torch.nonzero(mask_tensor, as_tuple=False).view(-1)
            if idx.numel() == 0:
                raise RuntimeError("Cannot sample from empty mask.")
            sub_probs = cand_probs.index_select(0, idx)
            total = float(sub_probs.sum().item())
            if total <= 0.0:
                sub_probs = torch.ones_like(sub_probs, dtype=torch.float32)
            else:
                sub_probs = (sub_probs / total).to(torch.float32)
            picked = int(torch.multinomial(sub_probs, 1).item())
            return int(cand_ids[idx[picked]].item())

        nonempty_mask = token_lens > 0
        empty_mask = token_lens == 0

        if nonempty_mask.sum().item() == 0:
            return _sample_from_mask(empty_mask)

        # 2) empty vs nonempty（不加偏置）
        if empty_mask.any():
            mass_empty = float(
                cand_probs.index_select(0, torch.nonzero(empty_mask, as_tuple=False).view(-1)).sum().item()
            )
            mass_nonempty = float(
                cand_probs.index_select(0, torch.nonzero(nonempty_mask, as_tuple=False).view(-1)).sum().item()
            )
            if mass_empty > 0.0 and mass_nonempty > 0.0:
                pair = torch.tensor([mass_empty, mass_nonempty], dtype=torch.float64, device=device)
                pair = (pair / pair.sum()).to(torch.float32)
                choice = int(torch.multinomial(pair, 1).item())
                if choice == 0:
                    return _sample_from_mask(empty_mask)
            elif mass_empty > 0.0:
                return _sample_from_mask(empty_mask)

        active_mask = nonempty_mask.clone()
        prefix = b""  # ✅ 当前 token 已确定的 prefix bytes（PRF 输入之一）

        while True:
            pos = len(prefix)  # ✅ byte_pos（PRF 输入之一）
            valid_mask = active_mask & (token_lens > pos)
            valid_idx = torch.nonzero(valid_mask, as_tuple=False).view(-1)
            if valid_idx.numel() == 0:
                return _sample_from_mask(active_mask)

            # 3) 在 pos 位置上做 byte 分组，累积质量
            byte_vals = torch.full((cand_ids.size(0),), -1, dtype=torch.long, device=device)
            byte_positions = token_offsets.index_select(0, valid_idx) + pos
            byte_vals[valid_idx] = byte_data.index_select(0, byte_positions).to(torch.long)

            byte_mass = torch.zeros(256, dtype=torch.float64, device=device)
            byte_mass.index_add_(0, byte_vals[valid_idx], cand_probs.index_select(0, valid_idx))

            base_logits = torch.log(byte_mass.clamp_min(eps)).unsqueeze(0)  # [1,256]

            # ✅ 核心改动：只传 token_ids/prefix_bytes/byte_pos
            if logits_processor is not None:
                biased = logits_processor(
                    base_logits,
                    token_ids=full_token_ids,
                    prefix_bytes=prefix,
                    byte_pos=pos,
                )
            else:
                biased = base_logits

            byte_probs = torch.softmax(biased.squeeze(0), dim=-1)

            valid_bytes = byte_vals[valid_idx]
            m = torch.zeros_like(byte_probs)
            m[valid_bytes] = 1.0
            masked = byte_probs * m
            total = float(masked.sum().item())
            if total <= 0.0:
                byte_dist = torch.ones_like(masked) / masked.numel()
            else:
                byte_dist = (masked / total).to(torch.float32)

            picked_byte = int(torch.multinomial(byte_dist, 1).item())

            # 4) 更新 prefix（供下一轮 PRF）
            prefix = prefix + bytes([picked_byte])

            # 5) 收缩候选池并做 exact/extend 决策
            active_mask = valid_mask & (byte_vals == picked_byte)
            pos_len = len(prefix)
            exact_mask = active_mask & (token_lens == pos_len)
            extend_mask = active_mask & (token_lens > pos_len)

            mass_exact = (
                float(cand_probs.index_select(0, torch.nonzero(exact_mask, as_tuple=False).view(-1)).sum().item())
                if exact_mask.any() else 0.0
            )
            mass_extend = (
                float(cand_probs.index_select(0, torch.nonzero(extend_mask, as_tuple=False).view(-1)).sum().item())
                if extend_mask.any() else 0.0
            )

            if mass_exact <= 0.0 and extend_mask.any():
                active_mask = extend_mask
                continue
            if not extend_mask.any():
                return _sample_from_mask(exact_mask) if exact_mask.any() else _sample_from_mask(active_mask)

            denom = mass_exact + mass_extend
            if denom <= 0.0:
                return _sample_from_mask(active_mask)

            pair = torch.tensor([mass_exact, mass_extend], dtype=torch.float64, device=device)
            pair = (pair / denom).to(torch.float32)
            choice = int(torch.multinomial(pair, 1).item())
            if choice == 0 and exact_mask.any():
                return _sample_from_mask(exact_mask)
            active_mask = extend_mask

            
    def _sample_token_via_first_byte(
        self,
        logits: Tensor,
        *,
        logits_processor=None,
        token_ids: Optional[Sequence[int]] = None,
    ) -> int:
        import collections
        device = logits.device
        probs = torch.softmax(logits.to(torch.float64), dim=-1)
        eps = 1e-12

        mask = probs > eps
        if mask.sum().item() == 0:
            return int(torch.argmax(logits, dim=-1).item())

        cand_ids = torch.nonzero(mask, as_tuple=False).view(-1).to(torch.long)
        if cand_ids.numel() == 0:
            return int(torch.argmax(logits, dim=-1).item())

        cand_probs = probs.index_select(0, cand_ids)

        full_token_ids = list(token_ids or [])

        # ---------------- debug knobs (no signature change) ----------------
        debug = False
        debug_topk_tokens = 100   # 打印多少个候选 token
        debug_topk_per_group = 1000 # 每组最多展示多少 token
        debug_topk_bytes = 10000   # 打印多少个 byte 组
        debug_every = 1             # 每隔多少步打印一次
        debug_max_steps = 100000  # 最多打印到第几步
        step = len(full_token_ids)
        do_debug = debug and (step % max(debug_every, 1) == 0) and (step <= debug_max_steps)

        def _byte_label(b: int) -> str:
            b = int(b) & 0xFF
            if b == 0x20:
                return "<sp>"
            if 32 <= b <= 126:
                return chr(b)
            return f"0x{b:02x}"

        def _tok_piece(tid: int) -> str:
            # 用 convert_ids_to_tokens 更接近你看到的 BPE piece（如 Ġxxx / ▁xxx）
            try:
                s = self.tokenizer.convert_ids_to_tokens([int(tid)], skip_special_tokens=False)[0]
            except Exception:
                s = str(int(tid))
            # 简单清理换行
            return s.replace("\n", "\\n")

        def _sample_from_mask(mask_tensor: torch.Tensor) -> Optional[int]:
            idx = torch.nonzero(mask_tensor, as_tuple=False).view(-1)
            if idx.numel() == 0:
                return None
            sub_probs = cand_probs.index_select(0, idx)
            total = float(sub_probs.sum().item())
            if total <= 0.0:
                sub_probs = torch.ones_like(sub_probs, dtype=torch.float32)
            else:
                sub_probs = (sub_probs / total).to(torch.float32)
            picked = int(torch.multinomial(sub_probs, 1).item())
            return int(cand_ids[idx[picked]].item())

        first_vals_raw = self.byte_vocab.first_bytes.to(device=device).index_select(0, cand_ids).to(torch.long)
        end_mask = self.byte_vocab.end_token_mask.to(device=device).index_select(0, cand_ids)
        valid_byte_mask = (~end_mask) & (first_vals_raw >= 0)

        mask_empty = (~end_mask) & (first_vals_raw < 0)
        mask_end = end_mask
        mask_byte = valid_byte_mask

        group_masks = torch.stack([mask_empty, mask_end, mask_byte], dim=0)
        group_names = ("empty", "end", "byte")

        cand_probs_exp = cand_probs.unsqueeze(0)
        group_masses = (group_masks.to(cand_probs_exp.dtype) * cand_probs_exp).sum(dim=1)

        available_mask = group_masses > 0.0
        if not available_mask.any():
            fallback = _sample_from_mask(mask)
            if fallback is not None:
                return fallback
            return int(torch.argmax(logits, dim=-1).item())

        available_idx = torch.nonzero(available_mask, as_tuple=False).view(-1)
        weights = group_masses.index_select(0, available_idx)
        weights = (weights / weights.sum()).to(torch.float32)
        choice_local = int(torch.multinomial(weights, 1).item())
        choice = group_names[int(available_idx[choice_local].item())]

        # empty/end 不加偏置
        if choice in ("empty", "end"):
            target_mask = mask_empty if choice == "empty" else mask_end
            picked = _sample_from_mask(target_mask)
            if picked is not None:
                return picked
            other_mask = mask & (~target_mask)
            picked = _sample_from_mask(other_mask)
            if picked is not None:
                return picked
            return int(torch.argmax(logits, dim=-1).item())

        # ---------------- byte group ----------------
        byte_idx = torch.nonzero(valid_byte_mask, as_tuple=False).view(-1)
        if byte_idx.numel() == 0:
            picked = _sample_from_mask(mask)
            if picked is not None:
                return picked
            return int(torch.argmax(logits, dim=-1).item())

        byte_vals = first_vals_raw.index_select(0, byte_idx)
        byte_probs = cand_probs.index_select(0, byte_idx)

        byte_mass = torch.zeros(256, dtype=torch.float64, device=device)
        byte_mass.index_add_(0, byte_vals, byte_probs)

        # base byte logits
        base_logits = torch.log(byte_mass.clamp_min(eps)).unsqueeze(0)  # [1,256]

        # watermark bias (TokenPrefixBytePRF 对齐：prefix_bytes=b"", byte_pos=0)
        if logits_processor is not None:
            biased = logits_processor(
                base_logits,
                token_ids=full_token_ids,
                prefix_bytes=b"",
                byte_pos=0,
            )
        else:
            biased = base_logits

        byte_dist = torch.softmax(biased.squeeze(0), dim=-1)  # [256]

        # 只允许候选中出现过的首字节
        valid_byte_mask_vec = torch.zeros_like(byte_dist)
        valid_byte_mask_vec.scatter_(0, byte_vals.to(byte_dist.device), 1.0)

        masked = byte_dist * valid_byte_mask_vec
        total = float(masked.sum().item())
        if total <= 0.0:
            masked = valid_byte_mask_vec
            total = float(masked.sum().item())
        if total <= 0.0:
            picked = _sample_from_mask(valid_byte_mask)
            if picked is not None:
                return picked
            return int(torch.argmax(logits, dim=-1).item())

        final_dist = (masked / total).to(torch.float32)

        # ---------------- DEBUG PRINTS ----------------
        if do_debug:
            # 1) TopK 候选 token（按 token 概率）
            k = min(debug_topk_tokens, cand_ids.numel())
            topv, topi = torch.topk(cand_probs.to(torch.float32), k=k, largest=True)
            topi = topi.to(torch.long)

            # 2) 按首字节分组：{label: [tok(p), ...]}
            grouped = collections.defaultdict(list)
            for rank in range(k):
                j = int(topi[rank].item())
                tid = int(cand_ids[j].item())
                p = float(cand_probs[j].item())
                fb = int(first_vals_raw[j].item())
                if fb < 0:
                    key = "<empty>"
                else:
                    key = _byte_label(fb)
                tok = _tok_piece(tid)
                grouped[key].append((tok, p))

            # 每组内部按 p 降序，并截断展示
            grouped_str = {}
            for key, lst in grouped.items():
                lst_sorted = sorted(lst, key=lambda x: x[1], reverse=True)[:debug_topk_per_group]
                grouped_str[key] = [f"{t}({p:.4f})" for (t, p) in lst_sorted]

            # 3) 打印首字节“组概率”：base_p 来自 byte_mass，final_p 来自 final_dist
            base_mass_sum = float(byte_mass.sum().item())
            # 候选首字节集合
            present_bytes = torch.unique(byte_vals).to(torch.long)
            base_p = {}
            final_p = {}
            for b in present_bytes.tolist():
                b = int(b)
                lbl = _byte_label(b)
                bm = float(byte_mass[b].item())
                base_p[lbl] = (bm / base_mass_sum) if base_mass_sum > 0 else 0.0
                final_p[lbl] = float(final_dist[b].item())

            # 按 final_p 排序，挑前 debug_topk_bytes 展示
            top_bytes = sorted(final_p.items(), key=lambda kv: kv[1], reverse=True)[:debug_topk_bytes]

            print("\n" + "=" * 90)
            print(f"[DEBUG first-byte] step={step}  #cands={cand_ids.numel()}  topK={k}")
            print(f"[DEBUG] group_masses(empty/end/byte)={group_masses.detach().cpu().tolist()}")
            print("[DEBUG] token groups by first-byte (top tokens per group):")
            # 你要的 {t:[...], n:[...]} 风格
            print(grouped_str)

            print("[DEBUG] first-byte group probs (base_p vs final_p), top by final_p:")
            for lbl, fp in top_bytes:
                bp = base_p.get(lbl, 0.0)
                print(f"  {lbl}: base_p={bp:.6f}  final_p={fp:.6f}")

        # ---------------- sample a byte then sample a token in that byte group ----------------
        picked_byte = int(torch.multinomial(final_dist, 1).item())

        final_mask = valid_byte_mask & (first_vals_raw == picked_byte)
        picked = _sample_from_mask(final_mask)
        if picked is not None:
            return picked

        picked = _sample_from_mask(valid_byte_mask)
        if picked is not None:
            return picked

        return int(torch.argmax(logits, dim=-1).item())

    
    
    def _gather_group_mass_and_expand(
        self,
        pool: List[ByteCandidate],
        prompt_ids: Tensor,
        eos_token_id: Optional[int],
        device: torch.device,
    ) -> Tuple[torch.Tensor, List[ByteCandidate]]:
        """
        累加每个候选在下一个 byte 组的概率，同时对“字节耗尽但尚未结束”的候选即时扩展。
        """
        group = torch.zeros(NUM_GROUPS, dtype=torch.float64)
        prompt_ids = prompt_ids.to(device)
        dtype = prompt_ids.dtype

        ready: List[ByteCandidate] = []
        pending: List[ByteCandidate] = []
        eos_ready: List[ByteCandidate] = []

        for cand in pool:
            nb = cand.next_byte()

            if nb is not None:
                group[nb] += cand.prob
                ready.append(cand)
                continue
            if eos_token_id is not None and cand.last_token == eos_token_id:
                group[GROUP_EOS] += cand.prob
                eos_ready.append(cand)
                continue
            pending.append(cand)

        if pending:
            expanded = self._expand_pending_candidates(pending, prompt_ids, dtype, device)
            for cand in expanded:
                nb = cand.next_byte()
                if nb is not None:
                    group[nb] += cand.prob
                    ready.append(cand)
                elif eos_token_id is not None and cand.last_token == eos_token_id:
                    group[GROUP_EOS] += cand.prob
                    eos_ready.append(cand)
                else:
                    raise RuntimeError("Expanded candidate still lacks visible bytes.")

        self._renorm(ready, label="gather")
        return group, ready, eos_ready

    def _expand_pending_candidates(
        self,
        pending: List[ByteCandidate],
        prompt_ids: Tensor,
        dtype: torch.dtype,
        device: torch.device,
    ) -> List[ByteCandidate]:
        """
        对“已无可见字节但尚未到 EOS” 的候选集合:
        1. 计算其总概率
        2. 按概率挑选一个代表做一次模型前向
        3. 用 softmax 结果生成新的 ByteCandidate（概率 = 总概率 * 子概率）
        """
        total_prob = sum(c.prob for c in pending)
        if total_prob <= 0:
            return []

        weights = torch.tensor([c.prob for c in pending], dtype=torch.float64)
        probs = (weights / weights.sum()).to(torch.float32)
        rep_idx = int(torch.multinomial(probs, 1).item())
        rep = pending[rep_idx]

        continuation = torch.tensor([rep.tokens], dtype=dtype, device=device)
        full_ids = torch.cat([prompt_ids, continuation], dim=1)
        logits = self.charm_model.compute_logits_via_generate(full_ids)
    
        return self._candidates_from_logits(rep.tokens, total_prob, logits)

    def _candidates_from_logits(
        self,
        prefix_tokens: Tuple[int, ...],
        prefix_prob: float,
        logits: Tensor,
    ) -> List[ByteCandidate]:
        probs = torch.softmax(logits, dim=-1)
        out: List[ByteCandidate] = []
        for token_id, prob in enumerate(probs.tolist()):
            if prob <= 0:
                continue
            payload = self.byte_vocab.bytes_of(token_id)
            out.append(
                ByteCandidate(
                    tokens=prefix_tokens + (token_id,),
                    prob=float(prefix_prob * prob),
                    last_token=int(token_id),
                    last_bytes=payload,
                    offset=0,
                )
            )
        return out

    def _advance_pool(
        self,
        pool: List[ByteCandidate],
        pick: int,
        pick_mass: float,
    ) -> List[ByteCandidate]:
        """
        根据采样到的 byte，保留并推进能输出该 byte 的候选，然后重新归一化。
        """
        selected: List[ByteCandidate] = []

        for cand in pool:
            if cand.next_byte() == pick:
                selected.append(
                    ByteCandidate(
                        tokens=cand.tokens,
                        prob=cand.prob/pick_mass,
                        last_token=cand.last_token,
                        last_bytes=cand.last_bytes,
                        offset=cand.offset + 1,
                    )
                )

        return selected

    @staticmethod
    def _renorm(pool: List[ByteCandidate], *, label: str = "") -> None:
        total = sum(c.prob for c in pool)
        if not pool:
            return
        if not (0.999 <= total <= 1.001):
            raise RuntimeError(f"Pool probability drift detected ({label}): sum={total}")

    @staticmethod
    def _visible_window_bytes(committed: bytearray, prefix_length: int) -> bytes:
        if prefix_length <= 0:
            return bytes(committed)
        if not committed:
            return b""
        visible: List[int] = []
        i = len(committed)
        while i > 0 and len(visible) < prefix_length:
            bval = committed[i - 1]
            if bval == 0x20:
                i -= 1
                continue
            visible.append(bval)
            i -= 1
        return bytes(reversed(visible))
