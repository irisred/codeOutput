# charm/utils.py
from __future__ import annotations
from typing import Optional, Tuple
import torch
from torch import Tensor


def apply_temperature(logits: Tensor, temperature: float) -> Tensor:
    """
    温度缩放：logits / T（T==1 返回原值；T<=0 会报错）
    """
    if temperature is None or float(temperature) == 1.0:
        return logits
    T = float(temperature)
    if T <= 0:
        raise ValueError(f"temperature must be > 0, got {T}")
    return logits / T


def _topk_indices(logits: Tensor, top_k: int) -> Tensor:
    """
    返回 top_k 的索引（按 logit 从大到小排序）；当 top_k>=V 时返回所有索引。
    """
    V = logits.shape[-1]
    if top_k is None or top_k <= 0 or top_k >= V:
        return torch.arange(V, device=logits.device, dtype=torch.long)
    _, idx = torch.topk(logits, k=top_k, dim=-1, largest=True, sorted=True)
    return idx


def _topp_indices_from_logits(logits: Tensor, top_p: float) -> Tensor:
    """
    Nucleus (top-p) 过滤：对 logits 做 softmax 得到概率，按概率从大到小累加，
    直到累积 >= p，返回被保留的索引（降序）。
    - 若 top_p >= 1.0：返回所有索引
    - 若没有元素（极小 p）也至少保留一个（最大者）
    """
    p = float(top_p)
    V = logits.shape[-1]
    all_idx = torch.arange(V, device=logits.device, dtype=torch.long)
    if p >= 1.0:
        return all_idx

    # 按 logit 从大到小排序
    sorted_vals, sorted_idx = torch.sort(logits, dim=-1, descending=True)
    probs = torch.softmax(sorted_vals, dim=-1)
    cumsum = torch.cumsum(probs, dim=-1)
    # 找到第一个使得累积 >= p 的位置
    cutoff = torch.searchsorted(cumsum, torch.tensor(p, device=logits.device, dtype=probs.dtype))
    cutoff = int(cutoff.item())
    cutoff = max(0, cutoff)  # 至少保留一个
    kept = sorted_idx[: cutoff + 1]
    return kept


def _intersection_preserve_order(primary: Tensor, maskset: Tensor, V: int) -> Tensor:
    """
    交集：保留 primary 的顺序。maskset 是一个包含若干索引的 LongTensor。
    返回 primary 中同时存在于 maskset 的那些索引，顺序与 primary 相同。
    """
    device = primary.device
    is_in = torch.zeros(V, dtype=torch.bool, device=device)
    is_in[maskset] = True
    return primary[is_in[primary]]


def apply_topk_topp(
    logits_t: Tensor,
    top_k: Optional[int],
    top_p: float,
) -> Tuple[Tensor, Tensor]:
    """
    兼容版：返回 *filtered logits* 和 kept 索引（历史实现）。
    - filtered_logits: 在未保留位置为 -inf，方便后续再 softmax
    - kept:            LongTensor[K]
    【保留用于旧代码；新代码建议用 apply_topk_topp_masked_probs】
    """
    V = logits_t.shape[-1]
    device = logits_t.device

    K = _topk_indices(logits_t, int(top_k) if top_k is not None else None)
    P = _topp_indices_from_logits(logits_t, float(top_p))
    kept = _intersection_preserve_order(P, K, V)

    # 兜底：至少保留一个
    if kept.numel() == 0:
        kept = torch.argmax(logits_t, dim=-1, keepdim=True)
        if kept.dim() == 0:
            kept = kept.unsqueeze(0)

    # 生成 filtered logits：其他位置置为 -inf
    filtered = logits_t.new_full((V,), float("-inf"))
    filtered.scatter_(0, kept, logits_t.index_select(0, kept))
    return filtered, kept


def apply_topk_topp_masked_probs(
    logits_t: Tensor,
    top_k: Optional[int],
    top_p: float,
) -> Tuple[Tensor, Tensor]:
    """
    新实现（推荐）：对“完整词表”先 softmax 得到 probs_full，然后将未保留项概率置 0。
    返回：
      - probs_masked: Tensor[V]，未保留项为 0，保留项为原始 softmax 概率（以全词表为归一化域）
      - kept:         LongTensor[K]
    说明：
      * 与“裁剪 logits 再 softmax”在保留集合内的相对比例完全一致；
        但这里保留了“全词表归一”的含义，利于检测侧在固定 256 字节域或全词表域重放。
      * 下游若要采样，可直接对 probs_masked 归一化（sum>0 则 /sum，否则退回均匀），
        结果与传统裁剪后再 softmax 等价。
    """
    V = logits_t.shape[-1]
    device = logits_t.device

    # 选集合（以 P 降序为主序，交集保序）
    K = _topk_indices(logits_t, int(top_k) if top_k is not None else None)
    P = _topp_indices_from_logits(logits_t, float(top_p))
    kept = _intersection_preserve_order(P, K, V)

    # 兜底：至少保留一个
    if kept.numel() == 0:
        kept = torch.argmax(logits_t, dim=-1, keepdim=True)
        if kept.dim() == 0:
            kept = kept.unsqueeze(0)

    # 全词表 softmax -> 屏蔽未保留项为 0
    probs_full = torch.softmax(logits_t, dim=-1)  # sum==1
    probs_masked = torch.zeros_like(probs_full, dtype=torch.float, device=device)
    probs_masked.scatter_(0, kept, probs_full.index_select(0, kept))
    return probs_masked, kept


def sample_softmax(logits_or_probs_1d: Tensor) -> int:
    """
    从一维向量采样一个 index：
      - 若输入是 logits（含 -inf），会先做 softmax
      - 若输入是概率（可能未归一化、含 0），会按 sum>0 归一化后采样
    """
    x = logits_or_probs_1d
    # 判断是否包含 -inf（视为 logits）
    if torch.isinf(x).any():
        if torch.isinf(x).all():
            probs = torch.ones_like(x, dtype=torch.float) / x.numel()
        else:
            probs = torch.softmax(x, dim=-1)
    else:
        p = x.clamp_min(0)
        s = float(p.sum().item())
        probs = (p / s) if s > 0.0 else (torch.ones_like(p, dtype=torch.float) / p.numel())
    idx = torch.multinomial(probs, num_samples=1)
    return int(idx.item())


def logsumexp(x: Tensor) -> Tensor:
    """
    torch.logsumexp 的一维便捷封装，返回标量 Tensor。
    """
    if x.numel() == 0:
        return x.new_tensor(float("-inf"))
    return torch.logsumexp(x, dim=-1)
