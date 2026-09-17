from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass

import builtins
import math
import torch
from torch import Tensor
from transformers import PreTrainedModel, PreTrainedTokenizerBase, LogitsProcessorList
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.generation.logits_process import (
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

from MarkLLM.charm_v2.kv_cache import _DualKVCache
from .visible import to_visible_bytes, text_to_visible_bytes

EPS = 1e-40
NUM_BYTE_VALUES = 256
GROUP_EPSILON = NUM_BYTE_VALUES  # 256
GROUP_EOS = NUM_BYTE_VALUES + 1  # 257
NUM_GROUPS = NUM_BYTE_VALUES + 2  # 0..255 bytes + epsilon + EOS


# ========== 以“模型真实 vocab size”构建 token→bytes 映射 ==========
def _model_vocab_size(model: PreTrainedModel) -> int:
    # 优先从输出嵌入矩阵
    emb = getattr(model, "get_output_embeddings", None)
    if callable(emb):
        out = model.get_output_embeddings()
        if out is not None and hasattr(out, "weight"):
            return int(out.weight.size(0))
    # 备选：config.vocab_size
    if hasattr(model, "config") and hasattr(model.config, "vocab_size"):
        return int(model.config.vocab_size)
    raise RuntimeError("Cannot determine model vocab size.")


def _build_token_byte_tables(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    device: torch.device,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Returns:
      vis_buf:  uint8 [S]       — 所有 token 的可见字节串拼接
      tok_st:   int64 [M]       — 每个 token_id 在 vis_buf 的起始下标（M = 模型 vocab size）
      tok_len:  int32 [M]       — 每个 token_id 的可见字节长度
    仅对 tokenizer.get_vocab() 出现的 id 写入真实起点与长度；其余 id 置 len=0（自然落入 ε 组）。
    """
    M = _model_vocab_size(model)

    tok_st  = torch.zeros(M, dtype=torch.long)
    tok_len = torch.zeros(M, dtype=torch.int32)

    vocab = tokenizer.get_vocab()  # dict[str] -> id
    ids = [int(v) for v in vocab.values() if 0 <= int(v) < M]

    bytelist: List[int] = []
    for tid in ids:
        vb = to_visible_bytes(tokenizer, tid)  # —— 只使用 visible.py
        tok_st[tid]  = len(bytelist)
        tok_len[tid] = len(vb)
        if vb:
            bytelist.extend(vb)

    vis_buf = torch.tensor(bytelist, dtype=torch.uint8)
    return vis_buf.to(device), tok_st.to(device), tok_len.to(device)

# ========== UTF-8 规范化 + 尾段多 token 精确分段（trie） ==========
_TOKTRIE_CACHE: Dict[int, Tuple[List[Dict[int, int]], List[List[int]]]] = {}

def _build_token_byte_trie(
    tokenizer: PreTrainedTokenizerBase,
) -> Tuple[List[Dict[int, int]], List[List[int]]]:
    key = id(tokenizer)
    trie = _TOKTRIE_CACHE.get(key)
    if trie is not None:
        return trie

    next_: List[Dict[int, int]] = [dict()]  # root 0
    term:  List[List[int]]      = [[]]

    def new_node() -> int:
        next_.append({})
        term.append([])
        return len(next_) - 1

    vocab = tokenizer.get_vocab()
    for tid in vocab.values():
        tid = int(tid)
        vb = to_visible_bytes(tokenizer, tid)
        if not vb:
            continue
        node = 0
        for b in vb:
            if b not in next_[node]:
                next_[node][b] = new_node()
            node = next_[node][b]
        term[node].append(tid)

    _TOKTRIE_CACHE[key] = (next_, term)
    return next_, term


def _segment_tail_to_tokids(
    tail: bytes,
    tokenizer: PreTrainedTokenizerBase,
) -> Optional[List[int]]:
    """
    用 trie 对 bytes tail 做精确分段。DP + 回溯（长匹配优先）。失败返回 None。
    """
    if not tail:
        return []
    next_, term = _build_token_byte_trie(tokenizer)
    n = len(tail)
    memo: Dict[int, Optional[List[int]]] = {}

    def dfs(i: int) -> Optional[List[int]]:
        if i in memo:
            return memo[i]
        if i == n:
            memo[i] = []
            return []
        node = 0
        j = i
        ends: List[Tuple[int, List[int]]] = []
        while j < n and (tail[j] in next_[node]):
            node = next_[node][tail[j]]
            j += 1
            if term[node]:
                ends.append((j, term[node]))
        for end_pos, tids in reversed(ends):  # 长匹配优先
            for tid in sorted(tids):
                rest = dfs(end_pos)
                if rest is not None:
                    memo[i] = [tid] + rest
                    return memo[i]
        memo[i] = None
        return None

    return dfs(0)


# ========= CHARM 适配器（258 组：0..255 字节 + ε + EOS） ==========
@dataclass
class _GenerationRuntime:
    config: GenerationConfig
    eos_tuple: Tuple[int, ...]
    stopping_criteria: StoppingCriteriaList
    logits_processor: LogitsProcessorList | None
    logits_warper: LogitsProcessorList | None
    max_new_tokens: int


class CharmAdapter:
    _DCACHE = _DualKVCache()  # 类级双缓存

    @staticmethod
    def _reset_cache():
        CharmAdapter._DCACHE.reset()

    @staticmethod
    def _build_generation_config(model: PreTrainedModel, gen_kwargs: Dict[str, Any]) -> GenerationConfig:
        if hasattr(model, "generation_config") and model.generation_config is not None:
            generation_config = model.generation_config
        elif hasattr(model, "config") and model.config is not None:
            generation_config = GenerationConfig.from_model_config(model.config)
        else:
            generation_config = GenerationConfig()
        for key, value in gen_kwargs.items():
            if hasattr(generation_config, key):
                setattr(generation_config, key, value)
        return generation_config

    @staticmethod
    def _build_logits_warper(generation_config: GenerationConfig) -> LogitsProcessorList:
        warpers = LogitsProcessorList()
        temperature = getattr(generation_config, "temperature", 1.0)
        if temperature is not None and temperature != 1.0:
            warpers.append(TemperatureLogitsWarper(temperature))
        top_k = getattr(generation_config, "top_k", 0)
        if top_k is not None and top_k > 0:
            warpers.append(TopKLogitsWarper(top_k))
        top_p = getattr(generation_config, "top_p", 1.0)
        if top_p is not None and 0.0 < top_p < 1.0:
            warpers.append(TopPLogitsWarper(top_p, min_tokens_to_keep=1))
        return warpers

    @staticmethod
    def _prepare_logits_components(
        model: PreTrainedModel,
        generation_config: GenerationConfig,
        input_ids: Tensor,
    ) -> tuple[LogitsProcessorList | None, LogitsProcessorList | None]:
        logits_processor = model._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids.shape[-1],
            encoder_input_ids=None,
            prefix_allowed_tokens_fn=None,
            logits_processor=LogitsProcessorList(),
        )
        get_warper = getattr(model, "_get_logits_warper", None)
        if callable(get_warper):
            logits_warper = get_warper(generation_config)
        else:
            logits_warper = CharmAdapter._build_logits_warper(generation_config)
        return logits_processor, logits_warper

    @staticmethod
    def _apply_official_processors(
        logits_processor: LogitsProcessorList | None,
        logits_warper: LogitsProcessorList | None,
        input_ids: Tensor,
        logits: Tensor,
    ) -> Tensor:
        if logits_processor:
            logits = logits_processor(input_ids, logits)
        if logits_warper:
            logits = logits_warper(input_ids, logits)
        return logits

    @staticmethod
    def _compute_next_logits(
        model: PreTrainedModel,
        input_ids: Tensor,
        runtime: _GenerationRuntime,
    ) -> Tensor:
        logits = CharmAdapter._DCACHE.forward_for_sequence(model, input_ids)
        if runtime.logits_processor or runtime.logits_warper:
            logits = logits.unsqueeze(0)
            logits = CharmAdapter._apply_official_processors(
                runtime.logits_processor, runtime.logits_warper, input_ids, logits
            )
            logits = logits.squeeze(0)
        return logits

    @staticmethod
    def _prepare_generation_runtime(
        model: PreTrainedModel,
        gen_kwargs: Dict[str, Any],
        input_ids: Tensor,
        device: torch.device,
    ) -> _GenerationRuntime:
        generation_config = CharmAdapter._build_generation_config(model, gen_kwargs)
        generation_config, _ = model._prepare_generation_config(generation_config, **{})
        model._prepare_special_tokens(generation_config, kwargs_has_attention_mask=None, device=device)
        input_length = input_ids.shape[-1]
        eos_ids = generation_config.eos_token_id
        if eos_ids is None:
            eos_tuple: Tuple[int, ...] = tuple()
        elif isinstance(eos_ids, (list, tuple, set)):
            eos_tuple = tuple(int(x) for x in eos_ids if x is not None)
        else:
            eos_tuple = (int(eos_ids),)
        max_new = int(getattr(generation_config, "max_new_tokens", gen_kwargs.get("max_new_tokens", 128)))
        generation_config.max_new_tokens = max_new
        generation_config.max_length = input_length + max_new
        stopping_criteria: StoppingCriteriaList = model._get_stopping_criteria(
            generation_config, stopping_criteria=StoppingCriteriaList()
        )
        logits_processor, logits_warper = CharmAdapter._prepare_logits_components(model, generation_config, input_ids)
        return _GenerationRuntime(
            config=generation_config,
            eos_tuple=eos_tuple,
            stopping_criteria=stopping_criteria,
            logits_processor=logits_processor,
            logits_warper=logits_warper,
            max_new_tokens=max_new,
        )
        
    @staticmethod
    def _segment_bytes_to_tokids(
        committed: bytes,
        tokenizer: PreTrainedTokenizerBase,
    ) -> list[int]:
        """
        将任意 committed bytes 分为：
        utf8_head 可解码前缀  -> encode 为 token ids
        tail 剩余非法字节部分 -> 用 token-bytes trie 精确分段
        """
        n = len(committed)
        head_text = ""
        tail = b""
        for i in range(n, -1, -1):
            head = committed[:i]
            try:
                head_text = head.decode("utf-8", errors="strict")
                tail = committed[i:]
                break
            except UnicodeDecodeError:
                continue

        head_tokids = tokenizer.encode(head_text, add_special_tokens=False, return_tensors=None)

        if len(tail) > 0:
            next_, term = _build_token_byte_trie(tokenizer)
            node = 0
            seg = []
            for b in tail:
                if b not in next_[node]:
                    raise ValueError(f"非法字节 {b:#x} 无法在词表中找到匹配。")
                node = next_[node][b]
                if term[node]:
                    seg.append(term[node][0])  # 简化：取第一个匹配
                    node = 0
            if node != 0:
                raise ValueError("尾部存在未完全匹配的残余字节。")
            head_tokids.extend(seg)

        return list(map(int, head_tokids))
    
    @staticmethod
    @torch.inference_mode()
    def _forward_from_committed(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        input_ids0: Tensor,        # [1, T0]
        runtime: _GenerationRuntime,
        device: torch.device,
    ) -> Tuple[Tensor, Tensor, int]:
        canon_ids = input_ids0

        logits = CharmAdapter._compute_next_logits(model, canon_ids, runtime)
        logp_all = torch.log_softmax(logits, dim=-1)
        finite_mask = torch.isfinite(logits)
        if finite_mask.any():
            cand_ids = finite_mask.nonzero(as_tuple=False).squeeze(1)
        else:
            cand_ids = torch.arange(logits.shape[-1], device=logits.device, dtype=torch.long)
        logp_sel = logp_all.index_select(0, cand_ids).to(dtype=torch.float32)

        return cand_ids, logp_sel, int(canon_ids.shape[1])
    
    
    @staticmethod
    @torch.inference_mode()
    def _forward_from_tokenseq(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        input_ids0: Tensor,          # [1, T0] 原始 prompt
        tokseq: list[int] | Tensor,  # 要追加的 token 序列
        runtime: _GenerationRuntime,
        device: torch.device,
    ) -> Tuple[Tensor, Tensor, int]:

        tail_t = torch.tensor([tokseq], dtype=input_ids0.dtype, device=device) if isinstance(tokseq, list) \
                 else tokseq.to(device=device, dtype=input_ids0.dtype).unsqueeze(0)
        canon_ids = torch.cat([input_ids0, tail_t], dim=1)

        logits = CharmAdapter._compute_next_logits(model, canon_ids, runtime)
        logp_all = torch.log_softmax(logits, dim=-1)
        finite_mask = torch.isfinite(logits)
        if finite_mask.any():
            cand_ids = finite_mask.nonzero(as_tuple=False).squeeze(1)
        else:
            cand_ids = torch.arange(logits.shape[-1], device=logits.device, dtype=torch.long)
        logp_sel = logp_all.index_select(0, cand_ids).to(dtype=torch.float32)

        return cand_ids, logp_sel, int(canon_ids.shape[1])
    
    @staticmethod
    def _sample_group(groups):
        group_logmass = torch.full((NUM_GROUPS,), -float("inf"), dtype=torch.float32)
        for i, g in enumerate(groups):
            if not g:
                continue
            logps = torch.tensor([x[1] for x in g], dtype=torch.float32)
            group_logmass[i] = torch.logsumexp(logps, dim=0)

        u = torch.rand_like(group_logmass)
        gumbel = -torch.log(-torch.log(u.clamp_min(1e-40)))
        pick = int(torch.argmax(group_logmass + gumbel).item())

        return pick, group_logmass

    # ---------- 主流程（字节级逐步选择；ε 单独为一组；组内到末尾才前向） ----------
    @staticmethod
    @torch.no_grad()
    def generate(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        prompt_inputs: Dict[str, Tensor],
        logits_processors: LogitsProcessorList,
        gen_kwargs: Dict[str, Any],
        charm_cfg,
    ) -> Tensor:
        """
        CHARM 生成（组内归一化语义版 + 安全防护）：
        - 先按 258 组（0..255 字节 + ε + EOS）做组概率；
        - 采样一组；
        - 组内归一化成条件概率，非边界保留，边界子集求 mass 并只扩“1 条边界路径”；
        - 新孩子概率 = cand_prob_new * mass_boundary；
        - 下一池 = 该组选中后留下的“非边界候选(组内概率)” + “新孩子(乘边界质量)”，总和恒为 1；
        - 只有 pick < 256 时才提交字节并让 off + 1。
        """
        if not getattr(charm_cfg, "enabled", True):
            return model.generate(
                logits_processor=logits_processors,
                **gen_kwargs,
                **prompt_inputs,
            )

        DEBUG_FULL_256 = bool(getattr(charm_cfg, "debug_full_byte_probs", False))
        DEBUG_TOPK     = int(getattr(charm_cfg, "debug_topk", 8))
        DEBUG_LISTS    = bool(getattr(charm_cfg, "debug_list_ops", False))
        TRACE_LOGS     = bool(
            getattr(charm_cfg, "trace_steps", False)
            or getattr(charm_cfg, "debug_trace", False)
            or getattr(charm_cfg, "debug_logs", False)
            or DEBUG_LISTS
        )
        anomaly_log_bytes_raw = getattr(charm_cfg, "anomaly_log_bytes", None)
        if anomaly_log_bytes_raw is None:
            ANOMALY_LOG_BYTES = set()
        elif isinstance(anomaly_log_bytes_raw, (list, tuple, set)):
            ANOMALY_LOG_BYTES = {int(x) for x in anomaly_log_bytes_raw}
        else:
            ANOMALY_LOG_BYTES = {int(anomaly_log_bytes_raw)}
        ANOMALY_LOG_PREV = int(getattr(charm_cfg, "anomaly_log_prev", 0))
        recent_byte_logs: List[Tuple[int, List[Tuple[int, float]], float, float]] = []

        def _info_print(*args, **kwargs):
            if TRACE_LOGS:
                builtins.print(*args, **kwargs)

        def _warn_print(*args, **kwargs):
            builtins.print(*args, **kwargs)

        CharmAdapter._reset_cache()
        device = next(model.parameters()).device
        input_ids0: Tensor = prompt_inputs["input_ids"].to(device)
        canon_len_init = int(input_ids0.shape[1])
        assert input_ids0.shape[0] == 1, "只支持 batch_size=1"

        runtime = CharmAdapter._prepare_generation_runtime(model, gen_kwargs, input_ids0, device)
        max_new_tokens = runtime.max_new_tokens
        temperature    = float(gen_kwargs.get("temperature", 1.0))
        top_k          = int(gen_kwargs.get("top_k", 0)) or None
        top_p          = float(gen_kwargs.get("top_p", 1.0))
        byte_window_len = int(getattr(charm_cfg, "prefix_length", 0))
        eos_id_val = gen_kwargs.get("eos_token_id", getattr(tokenizer, "eos_token_id", None))
        if isinstance(eos_id_val, (list, tuple)):
            eos_id_val = eos_id_val[0] if eos_id_val else None
        try:
            eos_id = int(eos_id_val) if eos_id_val is not None else None
        except Exception:
            eos_id = None

        # 以模型 vocab 大小构建 token→bytes 线性表
        vis_buf, tok_starts, tok_lens = _build_token_byte_tables(model, tokenizer, device)
        V = int(tok_lens.numel())
        _info_print(f"[init] V={V}, vis_buf={int(vis_buf.numel())}, "
            f"T0={canon_len_init}, max_new_tokens={max_new_tokens}, temp={temperature}, top_k={top_k}, top_p={top_p}")

        # 初始化：从空 committed 前向
        committed: bytearray = bytearray()
        def _current_byte_window() -> bytes:
            if byte_window_len <= 0 or len(committed) <= byte_window_len:
                return bytes(committed)
            return bytes(committed[-byte_window_len:])

        def _decode_tail(tokseq: Tuple[int, ...], limit_tokens: int = 8) -> str:
            if not tokseq:
                return ""
            tail = tokseq[-limit_tokens:]
            try:
                return tokenizer.decode(tail, skip_special_tokens=False)
            except Exception:
                return repr(tail)

        # 仅在 UTF-8 边界时对可见字符施加偏置
        visible_bias_only = bool(getattr(charm_cfg, "visible_bias_only", False))

        def _at_utf8_boundary(data: bytearray) -> bool:
            if not data:
                return True
            idx = len(data) - 1
            cont = 0
            while idx >= 0 and (data[idx] & 0xC0) == 0x80:
                cont += 1
                idx -= 1
            if idx < 0:
                return False
            lead = data[idx]
            if (lead & 0x80) == 0x00:
                return cont == 0
            if (lead & 0xE0) == 0xC0:
                need = 1
            elif (lead & 0xF0) == 0xE0:
                need = 2
            elif (lead & 0xF8) == 0xF0:
                need = 3
            else:
                return False
            return cont == need

        def _apply_byte_logits_processors(byte_probs: torch.Tensor, utf8_boundary: bool) -> tuple[torch.Tensor, bool]:
            if not logits_processors:
                return byte_probs, False
            if visible_bias_only and not utf8_boundary:
                return byte_probs, False
            mass = float(byte_probs.sum().item())
            if (not math.isfinite(mass)) or mass <= 0:
                return byte_probs, False
            byte_logits = torch.log(byte_probs.clamp_min(EPS)).unsqueeze(0)
            window_bytes = _current_byte_window()
            if isinstance(logits_processors, (list, tuple)):
                proc_list = list(logits_processors)
            else:
                proc_list = list(logits_processors) if hasattr(logits_processors, "__iter__") else [logits_processors]
            adjusted_logits = byte_logits
            for proc in proc_list:
                try:
                    adjusted_logits = proc(scores=adjusted_logits, byte_window=window_bytes)
                except TypeError:
                    adjusted_logits = proc(None, adjusted_logits)
            adjusted_probs = torch.softmax(adjusted_logits.squeeze(0), dim=-1)
            return adjusted_probs * mass, True
        
        cand_ids, cand_logp, canon_len = CharmAdapter._forward_from_committed(
            model, tokenizer, input_ids0, runtime, device,
        )
        if isinstance(cand_ids, list):
            cand_ids = torch.tensor(cand_ids, device=device, dtype=torch.long)
        else:
            cand_ids = cand_ids.to(device=device, dtype=torch.long)

        # 用 float64 存概率，降低下溢
        cand_prob = torch.exp(cand_logp.to(device=device, dtype=torch.float64))
        _info_print(f"[step0] init_candidates={int(cand_ids.numel())}, canon_len={canon_len}")

        # 父指针（Python list）
        node_parent: List[int] = [-1]
        node_token:  List[int] = [-1]
        base = 1
        K0 = int(cand_ids.numel())
        node_parent.extend([0] * K0)
        node_token.extend(cand_ids.detach().cpu().tolist())

        # 候选池（全部在 GPU）
        pool_node_ix  = torch.arange(base, base + K0, device=device, dtype=torch.long)  # 每个节点在 parent/token 表中的索引
        pool_last_tok = cand_ids.clone()                                                # 每个候选路径的“最后一个 token id”
        pool_off      = torch.zeros(K0, device=device, dtype=torch.long)                # 下一个要读的“可见字节位置”
        pool_prob     = cand_prob.clone().to(torch.float64)                             # 当前池中各路径的概率（总和=1）

        step = 0
        byte_n = 0

        def _unroll_tokseq(node_ix: int) -> List[int]:
            out: List[int] = []
            while node_ix > 0:
                out.append(int(node_token[node_ix]))
                node_ix = int(node_parent[node_ix])
            out.reverse()
            return out

        while canon_len - canon_len_init <= max_new_tokens:
            step += 1
            K = int(pool_node_ix.numel())
            if K == 0:
                _info_print(f"[stop] empty candidate pool at step{step}")
                break
            last_bias_applied = False

            # ------------- 安全获取“第 n 个可见字节” -------------
            last_tok = pool_last_tok
            # 基本 id 范围检查，避免对 tok_lens/tok_starts 的越界 index_select
            Vt = int(tok_lens.numel())
            bad_id = (last_tok < 0) | (last_tok >= Vt)
            if bad_id.any():
                if DEBUG_LISTS:
                    bad_ids = last_tok[bad_id][:8].detach().cpu().tolist()
                    _warn_print(f"[fatal] last_tok out-of-range: {bad_ids} (V={Vt}) -> clamp")
                else:
                    _warn_print(f"[fatal] last_tok out-of-range: count={int(bad_id.sum().item())} (V={Vt}) -> clamp")
                last_tok = last_tok.clamp_(0, Vt - 1)

            lens = tok_lens.index_select(0, last_tok).to(torch.long)        # [K]
            has_byte = pool_off < lens                                       # [K] bool
            nth_byte = torch.full((K,), GROUP_EPSILON, device=device, dtype=torch.long)

            if has_byte.any():
                idx_h   = has_byte.nonzero(as_tuple=False).squeeze(1)       # 相对池内下标
                start_h = tok_starts.index_select(0, last_tok.index_select(0, idx_h)).to(torch.long)
                idx_n_raw = start_h + pool_off.index_select(0, idx_h)       # vis_buf 里的目标下标

                Nvis = int(vis_buf.numel())
                valid = (idx_n_raw >= 0) & (idx_n_raw < Nvis)
                if (~valid).any():
                    if DEBUG_LISTS:
                        bp = (~valid).nonzero(as_tuple=False).squeeze(1)
                        s = min(8, int(bp.numel()))
                        s_idx = bp[:s]
                        bad_abs = idx_h.index_select(0, s_idx)
                        dbg_tid = last_tok.index_select(0, bad_abs).detach().cpu().tolist()
                        dbg_off = pool_off.index_select(0, bad_abs).detach().cpu().tolist()
                        dbg_sta = start_h.index_select(0, s_idx).detach().cpu().tolist()
                        dbg_idx = idx_n_raw.index_select(0, s_idx).detach().cpu().tolist()
                        _warn_print(f"[fatal] idx_n out-of-range (vis={Nvis}). "
                              f"examples tid={dbg_tid}, off={dbg_off}, start={dbg_sta}, idx_n={dbg_idx} -> mark epsilon")
                    else:
                        count = int((~valid).sum().item())
                        _warn_print(f"[fatal] idx_n out-of-range (vis={Nvis}). invalid_entries={count} -> mark epsilon")
                if valid.any():
                    vals = vis_buf.index_select(0, idx_n_raw.index_select(0, valid.nonzero(as_tuple=False).squeeze(1)))
                    nth_byte.index_copy_(
                        0,
                        idx_h.index_select(0, valid.nonzero(as_tuple=False).squeeze(1)),
                        vals.to(torch.long)
                    )

            # eos 单独映射到第 257 组
            if eos_id is not None and pool_last_tok.numel() > 0:
                eos_mask = (pool_last_tok == eos_id)
                if eos_mask.any():
                    nth_byte = nth_byte.clone()
                    nth_byte[eos_mask] = GROUP_EOS

            # ------------- 组质量→概率（严格合法） -------------
            pool_prob = pool_prob.nan_to_num(0.0).clamp_min_(0.0)
            group_mass = torch.zeros(NUM_GROUPS, device=device, dtype=torch.float64)
            group_mass.scatter_add_(0, nth_byte, pool_prob)

            group_probs = group_mass.nan_to_num(0.0).clamp_min(0.0)
            gp_sum = group_probs.sum()
            if (not torch.isfinite(gp_sum)) or (gp_sum <= 0):
                group_probs.fill_(1.0 / float(NUM_GROUPS))
                _warn_print(f"[warn] group_mass sum invalid ({float(gp_sum):.3e}), fallback to uniform over {NUM_GROUPS}")
            else:
                group_probs.div_(gp_sum)

            byte_mass_total = float(group_probs[:NUM_BYTE_VALUES].sum().item())
            utf8_boundary = _at_utf8_boundary(committed)
            if byte_mass_total > 0 and len(logits_processors) > 0:
                adjusted, applied_now = _apply_byte_logits_processors(group_probs[:NUM_BYTE_VALUES].clone(), utf8_boundary)
                last_bias_applied = applied_now
                group_probs = group_probs.clone()
                group_probs[:NUM_BYTE_VALUES] = adjusted
                gp_sum = group_probs.sum()
                if (not torch.isfinite(gp_sum)) or (gp_sum <= 0):
                    group_probs.fill_(1.0 / float(NUM_GROUPS))
                else:
                    group_probs.div_(gp_sum)

            probs256 = group_probs[:NUM_BYTE_VALUES]
            eps_prob = float(group_probs[GROUP_EPSILON].item())
            eos_prob = float(group_probs[GROUP_EOS].item())
            need_top = DEBUG_LISTS or TRACE_LOGS or ANOMALY_LOG_BYTES or ANOMALY_LOG_PREV > 0
            top_list: List[Tuple[int, float]] = []
            if need_top:
                keep_k = min(max(DEBUG_TOPK, 32) if DEBUG_LISTS else 32, NUM_BYTE_VALUES)
                topv, topi = torch.topk(probs256.to(torch.float32), k=keep_k)
                top_list = [(int(i), float(v)) for i, v in zip(topi.tolist(), topv.tolist())]
            log_this_byte = bool(ANOMALY_LOG_BYTES and (byte_n in ANOMALY_LOG_BYTES))
            if DEBUG_LISTS and DEBUG_FULL_256:
                _info_print(f"[byte@{byte_n}] probs256={probs256.detach().cpu().to(torch.float32).tolist()}, "
                      f"eps_prob={eps_prob:.6f}, eos_prob={eos_prob:.6f}")
            elif DEBUG_LISTS and top_list:
                _info_print(f"[byte@{byte_n}] top{len(top_list)}={top_list[:DEBUG_TOPK]}, "
                    f"eps_prob={eps_prob:.6f}, eos_prob={eos_prob:.6f}")
            if ANOMALY_LOG_BYTES or ANOMALY_LOG_PREV > 0:
                recent_byte_logs.append((byte_n, top_list[:32], eps_prob, eos_prob))
                if len(recent_byte_logs) > max(ANOMALY_LOG_PREV, 32):
                    recent_byte_logs.pop(0)

            # ------------- 采样一组 -------------
            pick = int(torch.multinomial(group_probs, 1).item())
            selected_mask = (nth_byte == pick)
            sel_idx_abs = selected_mask.nonzero(as_tuple=False).squeeze(1)  # 绝对索引（相对于池）
            S = int(sel_idx_abs.numel())
            group_label = "byte"
            if pick == GROUP_EPSILON:
                group_label = "epsilon"
            elif pick == GROUP_EOS:
                group_label = "eos"
            _info_print(f"[step{step}] byte_n={byte_n}, K={K}, picked_group={pick} "
                f"({group_label}), selected_size={S}, "
                f"has_n={(has_byte.sum().item())}, eps={(K - has_byte.sum().item())}")

            if S == 0:
                _warn_print(f"[warn] picked empty group {pick}, resample uniformly")
                # 退化防护：均匀从非空组中取一组
                nonempty = (group_mass > 0).nonzero(as_tuple=False).squeeze(1)
                if nonempty.numel() == 0:
                    break
                pick = int(nonempty[torch.randint(0, nonempty.numel(), (1,), device=device)].item())
                selected_mask = (nth_byte == pick)
                sel_idx_abs = selected_mask.nonzero(as_tuple=False).squeeze(1)
                S = int(sel_idx_abs.numel())
                if S == 0:
                    break

            if log_this_byte:
                prefix_bytes = bytes(committed[-128:]) if committed else b""
                try:
                    prefix_text = prefix_bytes.decode("utf-8", errors="ignore")
                except Exception:
                    prefix_text = ""
                top_for_log = list(top_list[:5])
                if pick < NUM_BYTE_VALUES and not any(c == pick for c, _ in top_for_log):
                    top_for_log.append((pick, float(probs256[pick].item())))
                readable = [(c, repr(chr(c))[1:-1], prob) for c, prob in top_for_log]
                chosen_char = repr(chr(pick))[1:-1] if pick < NUM_BYTE_VALUES else f"group{pick}"
                builtins.print(
                    f"[anomaly_log byte={byte_n}] prefix='{prefix_text[-40:]}' "
                    f"pick={pick}('{chosen_char}') "
                    f"top={readable}, eps_prob={eps_prob:.6f}, eos_prob={eos_prob:.6f}"
                )
                if ANOMALY_LOG_PREV > 0:
                    prev_entries = recent_byte_logs[:-1][-ANOMALY_LOG_PREV:]
                    for b_n, t_list, e_prob, eo_prob in prev_entries:
                        readable_prev = [(c, repr(chr(c))[1:-1], prob) for c, prob in t_list[:5]]
                        builtins.print(
                            f"[anomaly_prev byte={b_n}] top5={readable_prev}, "
                            f"eps_prob={e_prob:.6f}, eos_prob={eo_prob:.6f}"
                        )

            # 组内概率（条件概率）——关键点：这里必须“相对索引”
            sel_probs = pool_prob.index_select(0, sel_idx_abs).nan_to_num(0.0).clamp_min(0.0)  # [S]
            sel_sum = sel_probs.sum()
            if (not torch.isfinite(sel_sum)) or (sel_sum <= 0):
                sel_probs = torch.ones_like(sel_probs, dtype=torch.float64) / float(sel_probs.numel())
            else:
                sel_probs = sel_probs / sel_sum

            if eos_id is not None and pick == GROUP_EOS:
                eos_choice_rel = int(torch.multinomial(sel_probs, 1).item())
                eos_abs = int(sel_idx_abs[eos_choice_rel].item())
                eos_node_ix = int(pool_node_ix[eos_abs].item())
                eos_tokseq = _unroll_tokseq(eos_node_ix)
                continuation = torch.tensor([eos_tokseq], dtype=input_ids0.dtype, device=device)
                _info_print(f"[step{step}] EOS group selected, returning path_len={len(eos_tokseq)}")
                return torch.cat([input_ids0, continuation], dim=1)

            # 边界检测：也用“相对选中”的张量
            lens_sel = lens.index_select(0, sel_idx_abs)                    # [S]
            off_sel  = pool_off.index_select(0, sel_idx_abs)                # [S]
            boundary_mask_sel = (off_sel + 1) >= lens_sel                   # [S] True 表示到达该 token 可见字节末尾
            B = int(boundary_mask_sel.sum().item())
            _info_print(f"⚑ 发现 {B} 个边界 token (within selected group)")

            # 拆成“非边界相对索引”和“边界相对索引”
            keep_rel = (~boundary_mask_sel).nonzero(as_tuple=False).squeeze(1)   # 相对索引
            b_rel    = boundary_mask_sel.nonzero(as_tuple=False).squeeze(1)      # 相对索引

            # 对应的绝对索引（用于搬运结构数组）
            if keep_rel.numel() > 0:
                keep_abs = sel_idx_abs.index_select(0, keep_rel)
            else:
                keep_abs = torch.empty(0, dtype=torch.long, device=device)
            if b_rel.numel() > 0:
                b_abs = sel_idx_abs.index_select(0, b_rel)
            else:
                b_abs = torch.empty(0, dtype=torch.long, device=device)

            # === 边界扩展（只扩一条边界路径） ===
            new_nodes_ix = new_last_tok = new_off = new_prob = None
            if B > 0:
                # 使用“组内概率”来挑选 boundary 路径；并计算“边界质量”
                b_probs = sel_probs.index_select(0, b_rel)                  # 已组内归一
                b_probs = b_probs.nan_to_num(0.0).clamp_min(0.0)
                b_sum = b_probs.sum()
                if (not torch.isfinite(b_sum)) or (b_sum <= 0):
                    # 退化均匀（避免 multinomial device assert）
                    b_pick_rel = int(torch.multinomial(torch.ones(B, device=device, dtype=torch.float64) / B, 1).item())
                    mass_boundary = 0.0  # 理论上 b_sum=0，质量设为 0
                    _warn_print(f"[warn] boundary probs all-zero/NaN -> uniform pick within boundary set")
                else:
                    b_pick_rel = int(torch.multinomial((b_probs / b_sum), 1).item())
                    mass_boundary = float(b_probs.sum().item())            # (= b_sum.item()，∈[0,1])

                picked_abs = int(b_abs[b_pick_rel].item())
                picked_node_ix = int(pool_node_ix[picked_abs].item())
                picked_tokseq = _unroll_tokseq(picked_node_ix)

                if log_this_byte:
                    prefix_bytes2 = bytes(committed[-256:]) if committed else b""
                    try:
                        prefix_text2 = prefix_bytes2.decode("utf-8", errors="ignore")
                    except Exception:
                        prefix_text2 = ""
                    picked_text = _decode_tail(tuple(picked_tokseq), limit_tokens=16)
                    builtins.print(
                        f"[anomaly_boundary byte={byte_n}] prefix='{prefix_text2[-80:]}' "
                        f"picked_tok_tail='{picked_text[-80:]}' "
                        f"boundary_mass={mass_boundary:.6e} remaining_nonboundary={int(keep_rel.numel())}"
                    )

                _info_print(f"[step{step}] boundary_mass={mass_boundary:.6e}, "
                    f"boundary_max={float(b_probs.max().item()) if B>0 else 0:.6e}, "
                    f"boundary_min={float(b_probs.min().item()) if B>0 else 0:.6e}, "
                    f"remaining_nonboundary={int(keep_rel.numel())}")

                # 前向：对“完整 token 序列”做一次扩展
                cand_ids_new, cand_logp_new, canon_len = CharmAdapter._forward_from_tokenseq(
                    model=model, tokenizer=tokenizer, input_ids0=input_ids0, tokseq=picked_tokseq,
                    runtime=runtime, device=device,
                )
                if isinstance(cand_ids_new, list):
                    cand_ids_new = torch.tensor(cand_ids_new, device=device, dtype=torch.long)
                else:
                    cand_ids_new = cand_ids_new.to(device=device, dtype=torch.long)

                # 新孩子概率 = 子分布（已 softmax 到候选集） * 边界质量
                cand_prob_new = torch.exp(cand_logp_new.to(torch.float64))
                cand_prob_new = cand_prob_new.nan_to_num(0.0).clamp_min(0.0)
                if mass_boundary > 0:
                    new_prob = cand_prob_new * mass_boundary
                else:
                    new_prob = torch.zeros_like(cand_prob_new, dtype=torch.float64, device=device)

                if log_this_byte:
                    top_new = min(8, int(new_prob.numel()))
                    if top_new > 0:
                        top_vals, top_idx = torch.topk(new_prob, k=top_new)
                        cand_ids_top = cand_ids_new.index_select(0, top_idx)
                        child_entries: List[Tuple[int, str, float]] = []
                        for tok_id, prob in zip(cand_ids_top.tolist(), top_vals.tolist()):
                            try:
                                tok_text = tokenizer.decode([tok_id], skip_special_tokens=False)
                            except Exception:
                                tok_text = repr(tok_id)
                            child_entries.append((int(tok_id), tok_text, float(prob)))
                        builtins.print(
                            f"[anomaly_newtok byte={byte_n}] child_top={child_entries}"
                        )

                # 追加到 parent/token 表
                base2 = len(node_parent)
                Vn = int(cand_ids_new.numel())
                node_parent.extend([picked_node_ix] * Vn)
                node_token.extend(cand_ids_new.detach().cpu().tolist())
                new_nodes_ix = torch.arange(base2, base2 + Vn, device=device, dtype=torch.long)
                new_last_tok = cand_ids_new.clone()
                new_off      = torch.zeros(Vn, device=device, dtype=torch.long)

                pv = min(5, Vn)
                for j in range(pv):
                    _info_print(f"[step{step}]   child#{j}: id={int(new_last_tok[j].item())}, "
                        f"prob={float(new_prob[j].item()):.6e}")

            # === 组内非边界（条件概率）+ 新孩子（乘边界质量） => 下一池 ===
            if keep_rel.numel() > 0 and new_nodes_ix is not None:
                # 注意：这里 sel_probs / off_sel 都要用“相对索引 keep_rel”
                next_node_ix  = torch.cat([pool_node_ix.index_select(0, keep_abs), new_nodes_ix], dim=0)
                next_last_tok = torch.cat([pool_last_tok.index_select(0, keep_abs), new_last_tok], dim=0)
                next_off      = torch.cat([off_sel.index_select(0, keep_rel),       new_off], dim=0)
                next_prob     = torch.cat([sel_probs.index_select(0, keep_rel),     new_prob], dim=0).to(torch.float64)
            elif keep_rel.numel() > 0:
                next_node_ix  = pool_node_ix.index_select(0, keep_abs)
                next_last_tok = pool_last_tok.index_select(0, keep_abs)
                next_off      = off_sel.index_select(0, keep_rel)
                next_prob     = sel_probs.index_select(0, keep_rel).to(torch.float64)
            elif new_nodes_ix is not None:
                next_node_ix, next_last_tok, next_off, next_prob = new_nodes_ix, new_last_tok, new_off, new_prob.to(torch.float64)
            else:
                next_node_ix  = torch.empty(0, dtype=torch.long, device=device)
                next_last_tok = torch.empty(0, dtype=torch.long, device=device)
                next_off      = torch.empty(0, dtype=torch.long, device=device)
                next_prob     = torch.empty(0, dtype=torch.float64, device=device)

            # 提交真实字节（仅 pick < NUM_BYTE_VALUES 时），推进 byte_n 和 offset
            if pick < NUM_BYTE_VALUES:
                committed.append(pick)
                byte_n += 1
                if next_off.numel() > 0:
                    if keep_rel.numel() > 0:
                        prefix = keep_rel.numel()
                        next_off[:prefix] = next_off[:prefix] + 1
                allow_trace = (not visible_bias_only) or last_bias_applied
                if allow_trace:
                    for proc in logits_processors:
                        observer = getattr(proc, "trace_observation", None)
                        if callable(observer):
                            observer(int(pick))
                if byte_n >= 1 and committed[byte_n-1] == 32 and committed[byte_n-2] == 32:
                    builtins.print(f"[double_space] byte_n={byte_n-2}->{byte_n-1} prefix='{bytes(committed[max(0, byte_n-40):byte_n]).decode('utf-8', errors='ignore')}'")
                _info_print(f"[step{step}] committed_byte={pick}, byte_n -> {byte_n}")
            else:
                _info_print(f"[step{step}] epsilon picked, byte_n stays {byte_n}")

            # 写回池
            prev_K = K
            pool_node_ix  = next_node_ix
            pool_last_tok = next_last_tok
            pool_off      = next_off
            pool_prob     = next_prob  # 已是“该组选中后的条件概率”，总和≈1


            if pool_node_ix.numel() > 0:
                lens_next = tok_lens.index_select(0, pool_last_tok).to(torch.long)
                has_next = (pool_off < lens_next)
                cnt_has_next = int(has_next.sum().item())
            else:
                cnt_has_next = 0
            _info_print(f"[step{step}] pool_update: K {prev_K} -> {int(pool_node_ix.numel())}, "
                f"next_has_n={cnt_has_next}, next_eps={int(pool_node_ix.numel()) - cnt_has_next}")

            if (canon_len - canon_len_init) > max_new_tokens:
                _info_print(f"[stop] reached max_new_tokens: canon_len={canon_len}, init={canon_len_init}")
                break

        # 结束：从最终池取概率最大的路径
        if pool_prob.numel() == 0:
            return input_ids0

        best_i = int(torch.argmax(pool_prob).item())
        best_node_ix = int(pool_node_ix[best_i].item())
        best_tokseq = _unroll_tokseq(best_node_ix)
        _info_print(f"[done] best_path_len={len(best_tokseq)}")

        continuation = torch.tensor([best_tokseq], dtype=input_ids0.dtype, device=device)
        return torch.cat([input_ids0, continuation], dim=1)



    # ---------- 纯 token 域（回退/对照） ----------
    @staticmethod
    @torch.inference_mode()
    def generate_plain(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        prompt_inputs: Dict[str, Tensor],
        gen_kwargs: Dict[str, Any],
    ) -> Tensor:
        CharmAdapter._reset_cache()
        device = next(model.parameters()).device
        input_ids = prompt_inputs["input_ids"].to(device)
        assert input_ids.shape[0] == 1, "Plain adapter supports batch_size=1 only."

        runtime = CharmAdapter._prepare_generation_runtime(model, gen_kwargs, input_ids, device)
        generation_config = runtime.config
        eos_tuple = runtime.eos_tuple
        new_tokens = 0
        while new_tokens < runtime.max_new_tokens:
            logits = CharmAdapter._compute_next_logits(model, input_ids, runtime)
            if generation_config.do_sample:
                probs = torch.softmax(logits, dim=-1)
                next_token = int(torch.multinomial(probs, 1).item())
            else:
                next_token = int(torch.argmax(logits, dim=-1).item())

            add = torch.tensor([[next_token]], device=input_ids.device, dtype=input_ids.dtype)
            input_ids = torch.cat([input_ids, add], dim=1)
            new_tokens += 1

            if eos_tuple and next_token in eos_tuple:
                break
            if runtime.stopping_criteria(input_ids, None):
                break

        return input_ids
