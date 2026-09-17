from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
from torch import Tensor
from transformers import PreTrainedModel


class _SimpleKVCache:
    def __init__(self):
        self.tokens: Optional[Tuple[int, ...]] = None
        self.past: Any = None
        self.logits: Optional[Tensor] = None  # [1, V]
        self.device: Optional[torch.device] = None
        self.dtype_ids: Optional[torch.dtype] = None

    def reset(self):
        self.tokens = None
        self.past = None
        self.logits = None
        self.device = None
        self.dtype_ids = None

    def snapshot(self):
        return {
            "tokens": list(self.tokens) if self.tokens is not None else None,
            "logits": self.logits.detach().cpu() if self.logits is not None else None,
            "logits_dtype": str(self.logits.dtype) if self.logits is not None else None,
            "past": self._deep_to_cpu(self.past),
            "dtype_ids": str(self.dtype_ids) if self.dtype_ids is not None else None,
        }

    def restore(self, snapshot, device):
        tokens = snapshot.get("tokens")
        self.tokens = tuple(tokens) if tokens is not None else None
        logits = snapshot.get("logits")
        self.logits = logits.to(device) if logits is not None else None
        dtype_name = snapshot.get("dtype_ids")
        self.dtype_ids = getattr(torch, dtype_name.split(".")[-1]) if dtype_name else None
        past = snapshot.get("past")
        self.past = self._deep_to_device(past, device)
        self.device = device

    def _as_tuple(self, ids: Tensor) -> Tuple[int, ...]:
        return tuple(int(x) for x in ids[0].tolist())

    def _am_ones(self, total_len: int, device: torch.device) -> Tensor:
        return torch.ones((1, int(total_len)), device=device, dtype=torch.long)

    def _past_seq_len(self) -> int:
        """
        尽量从 past_key_values 结构里推断出已经缓存的序列长度。
        支持几种常见结构：tuple, list, dict, 带 .key 的对象, 直接 tensor。
        出问题就回退到 len(self.tokens)。
        """
        if self.past is None:
            return len(self.tokens) if self.tokens is not None else 0

        obj = self.past
        key_tensor = None

        try:
            # 常见: tuple[layers] -> (k, v)
            if isinstance(obj, (tuple, list)) and len(obj) > 0:
                first = obj[0]
                if isinstance(first, (tuple, list)) and len(first) > 0:
                    key_tensor = first[0]
                elif isinstance(first, dict):
                    key_tensor = first.get("key") or first.get("k")
                elif torch.is_tensor(first):
                    key_tensor = first
                elif hasattr(first, "key"):
                    key_tensor = getattr(first, "key")
                else:
                    key_tensor = first
            elif isinstance(obj, dict):
                key_tensor = obj.get("key") or obj.get("k")
            elif torch.is_tensor(obj):
                key_tensor = obj
            elif hasattr(obj, "key"):
                key_tensor = getattr(obj, "key")
        except Exception:
            key_tensor = None

        if key_tensor is None or not torch.is_tensor(key_tensor) or key_tensor.ndim < 2:
            return len(self.tokens) if self.tokens is not None else 0

        try:
            return int(key_tensor.shape[-2])
        except Exception:
            return len(self.tokens) if self.tokens is not None else 0

    def classify(self, req_tokens: Tuple[int, ...]) -> int:
        """
        返回：
        0 = miss
        1 = 前缀命中（req_tokens 以 cached tokens 开头，且更长）
        2 = 完全命中（长度也一样）
        要求 tokens + past + logits 都存在才算命中。
        """
        if self.tokens is None or self.past is None or self.logits is None:
            return 0

        cached = self.tokens
        if len(req_tokens) < len(cached):
            return 0
        if req_tokens[: len(cached)] != cached:
            return 0
        if len(req_tokens) == len(cached):
            return 2
        return 1

    # 保留接口，但不再在 DualKVCache 中使用浅拷贝 past（避免别名问题）
    def adopt_from(self, other: "_SimpleKVCache"):
        self.tokens = other.tokens
        self.past = other.past
        self.logits = other.logits
        self.device = other.device
        self.dtype_ids = other.dtype_ids

    def _deep_to_cpu(self, obj):
        if obj is None:
            return None
        if torch.is_tensor(obj):
            return obj.detach().cpu()
        if isinstance(obj, tuple):
            return tuple(self._deep_to_cpu(x) for x in obj)
        if isinstance(obj, list):
            return [self._deep_to_cpu(x) for x in obj]
        if isinstance(obj, dict):
            return {k: self._deep_to_cpu(v) for k, v in obj.items()}
        # DynamicCache 之类：如果有 .to 接口，试一下
        if hasattr(obj, "to"):
            try:
                return obj.to("cpu")
            except Exception:
                return obj
        return obj

    def _deep_to_device(self, obj, device):
        if obj is None:
            return None
        if torch.is_tensor(obj):
            return obj.to(device)
        if isinstance(obj, tuple):
            return tuple(self._deep_to_device(x, device) for x in obj)
        if isinstance(obj, list):
            return [self._deep_to_device(x, device) for x in obj]
        if isinstance(obj, dict):
            return {k: self._deep_to_device(v, device) for k, v in obj.items()}
        if hasattr(obj, "to"):
            try:
                return obj.to(device)
            except Exception:
                return obj
        return obj

    @torch.inference_mode()
    def forward_prefix_or_full(
        self,
        model: PreTrainedModel,
        req_tokens: Tuple[int, ...],
        full_input_ids: Tensor,
    ) -> Tensor:
        dev = full_input_ids.device
        dt_ids = full_input_ids.dtype

        # 设备不一致时，迁移缓存到当前 device
        if self.device is not None and self.device != dev:
            if self.past is not None:
                self.past = self._deep_to_device(self.past, dev)
            if self.logits is not None:
                self.logits = self.logits.to(dev)
            self.device = dev

        hit = self.classify(req_tokens)

        # 前缀命中时，检查 past_seq_len 与 cached_len 是否一致，不一致说明 cache 状态有问题
        if hit == 1:
            cached_len = len(self.tokens) if self.tokens is not None else 0
            past_seq_len = self._past_seq_len()
            if past_seq_len != cached_len:
                print(
                    f"[cache] inconsistent past_len={past_seq_len} tokens_len={cached_len}, "
                    "fallback to full + reset"
                )
                self.reset()
                hit = 0

        if hit == 2:
            return self.logits.squeeze(0)

        if hit == 1:
            cached_len = len(self.tokens)
            suffix = req_tokens[cached_len:]
            # print(f"[cache] hit=1 cached_len={cached_len} suffix_len={len(suffix)} tail={suffix[-5:]}")

            if not suffix:
                # 理论上不会发生，保险起见
                return self.logits.squeeze(0)

            tok = torch.tensor([list(suffix)], device=dev, dtype=dt_ids)
            suffix_len = len(suffix)
            past_seq_len = self._past_seq_len()
            if past_seq_len <= 0:
                past_seq_len = cached_len
            total_len = past_seq_len + suffix_len
            attn_mask = self._am_ones(total_len=total_len, device=dev)
            position_ids = torch.arange(
                past_seq_len,
                past_seq_len + suffix_len,
                device=dev,
                dtype=torch.long,
            ).unsqueeze(0)

            out = model(
                input_ids=tok,
                attention_mask=attn_mask,
                position_ids=position_ids,
                use_cache=True,
                past_key_values=self.past,
            )
            self.tokens = req_tokens
            self.past = getattr(out, "past_key_values", None)
            self.logits = out.logits[:, -1, :]
            self.device = dev
            self.dtype_ids = dt_ids
            return self.logits.squeeze(0)

        # miss 或上面强制回退的情况：完整 forward
        am_full = self._am_ones(total_len=len(req_tokens), device=dev)
        # print("[cache] miss -> full forward len", len(req_tokens))
        out = model(input_ids=full_input_ids, attention_mask=am_full, use_cache=True)
        self.tokens = req_tokens
        self.past = getattr(out, "past_key_values", None)
        self.logits = out.logits[:, -1, :]
        self.device = dev
        self.dtype_ids = dt_ids
        return self.logits.squeeze(0)


class _DualKVCache:
    """
    双缓存版本：
    - slot0 / slot1 完全独立，不再通过 adopt_from 共享同一个 past 对象。
    - 命中策略：
        * 两个都命中：用 tokens 更长的那个（缓存更“深”）
        * 只命中一个：用它
        * 都 miss：选择 tokens 更短的那个 slot 做 full forward（类似牺牲者）
    """

    def __init__(self):
        self.slot0 = _SimpleKVCache()
        self.slot1 = _SimpleKVCache()

    def reset(self):
        self.slot0.reset()
        self.slot1.reset()

    def snapshot(self):
        return {
            "slot0": self.slot0.snapshot(),
            "slot1": self.slot1.snapshot(),
        }

    def restore(self, snapshot, device):
        slot0_state = snapshot.get("slot0")
        slot1_state = snapshot.get("slot1")
        if slot0_state is not None:
            self.slot0.restore(slot0_state, device)
        if slot1_state is not None:
            self.slot1.restore(slot1_state, device)

    def _as_tuple(self, ids: Tensor) -> Tuple[int, ...]:
        return tuple(int(x) for x in ids[0].tolist())

    def _len_cached(self, slot: _SimpleKVCache) -> int:
        return 0 if slot.tokens is None else len(slot.tokens)

    @torch.inference_mode()
    def forward_for_sequence(
        self,
        model: PreTrainedModel,
        full_input_ids: Tensor,
    ) -> Tensor:
        req = self._as_tuple(full_input_ids)

        h0 = self.slot0.classify(req)
        h1 = self.slot1.classify(req)

        # 两个都命中：用缓存更长的那个
        if h0 > 0 and h1 > 0:
            len0, len1 = self._len_cached(self.slot0), self._len_cached(self.slot1)
            target = self.slot0 if len0 >= len1 else self.slot1
            return target.forward_prefix_or_full(model, req, full_input_ids)

        # 只有一个命中：直接用它（不再做 adopt_from）
        if h0 > 0:
            return self.slot0.forward_prefix_or_full(model, req, full_input_ids)
        if h1 > 0:
            return self.slot1.forward_prefix_or_full(model, req, full_input_ids)

        # 都 miss：踢掉缓存更短的那个 slot
        len0, len1 = self._len_cached(self.slot0), self._len_cached(self.slot1)
        target = self.slot0 if len0 <= len1 else self.slot1
        # 这里可以选择 reset 或不 reset；full forward 会覆盖掉状态，其实 reset 不必
        # target.reset()
        return target.forward_prefix_or_full(model, req, full_input_ids)
