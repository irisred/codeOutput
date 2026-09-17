# watermark/charm/detector.py
from __future__ import annotations
import math
from typing import Optional, List, Dict, Any
import torch
from transformers import PreTrainedTokenizerBase

from .visible import text_to_visible_bytes, to_visible_bytes
from .adapter import _build_token_byte_trie


class CharmDetector:
    """
    统一检测包装：
      - 若 charm_cfg.enabled=True：
          用“可见字节固定 256 域 + 滑窗(prefix_length 字节)”做检测（字符级）。
      - 否则（未启用 CHARM）：
          回落到“原始 token 域”的 KGW 检测，支持 prompt 对齐与 verify 逐步输出。
    """
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        algo_utils,              # KGWUtils（需提供：get_greenlist_ids / get_greenlist_ids_bytes / _compute_z_score）
        charm_cfg,
        device: str,
        gamma: float,
        prefix_length: int,
        z_threshold: float,
        normalize_form: Optional[str] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.utils = algo_utils
        self.charm_cfg = charm_cfg
        self.device = device
        self.gamma = float(gamma)
        self.prefix_length = int(prefix_length)
        self.z_threshold = float(z_threshold)
        self.normalize_form = normalize_form

    # ------------------ 工具 ------------------
    def _visible_bytes(self, text: str) -> bytes:
        return text_to_visible_bytes(text)


    def _utf8_start_mask(self, vb: bytes) -> List[bool]:
        mask: List[bool] = [False] * len(vb)
        expected = 0  # remaining continuation bytes
        for idx, b in enumerate(vb):
            if expected > 0:
                if (b & 0xC0) == 0x80:
                    mask[idx] = False
                    expected -= 1
                    continue
                else:
                    expected = 0  # invalid continuation, treat current as new head

            # treat current byte as a new codepoint start
            mask[idx] = True
            if (b & 0x80) == 0x00:
                expected = 0
            elif (b & 0xE0) == 0xC0:
                expected = 1
            elif (b & 0xF0) == 0xE0:
                expected = 2
            elif (b & 0xF8) == 0xF0:
                expected = 3
            else:
                expected = 0
        return mask

    def _z_from_hits(self, hits: int, trials: int) -> float:
        if trials <= 0:
            return float("-inf")

        # 1) gamma 做强制转换与裁剪，确保在 (0,1)
        eps = 1e-12
        g = float(self.gamma)
        if not (0.0 < g < 1.0):  # 配置异常时自动夹到开区间
            g = min(max(g, eps), 1.0 - eps)

        # 2) 稳定计算
        numer = float(hits) - float(trials) * g
        denom_sq = float(trials) * g * (1.0 - g)
        denom = math.sqrt(max(denom_sq, eps))  # 防止负/零 -> 复数/除零

        return numer / denom

    def _strip_prompt_bytes(self, text: str, prompt: Optional[str]) -> bytes:
        vb_full = self._visible_bytes(text)
        if not prompt:
            return vb_full
        vb_prompt = self._visible_bytes(prompt)
        # 若生成端是 “输出=仅新 tokens 解码”，则 text 里不含 prompt；这里兼容“text = prompt+new”场景
        if vb_full.startswith(vb_prompt):
            return vb_full[len(vb_prompt):]
        return vb_full

    def _strip_prompt_tokens(self, text: str, prompt: Optional[str]) -> torch.Tensor:
        """
        返回去掉 prompt 部分后的 token 序列（LongTensor）。
        与生成侧保持一致：生成用 add_special_tokens=True 编码 prompt，再只 decode 新 tokens。
        检测为了对齐 PRF 窗口，这里把 prompt 的 token 数量砍掉。
        """
        enc_all = self.tokenizer(text, return_tensors="pt", add_special_tokens=True)["input_ids"][0]
        if not prompt:
            return enc_all
        enc_prompt = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)["input_ids"][0]
        n = enc_prompt.numel()
        if enc_all.numel() >= n and torch.equal(enc_all[:n], enc_prompt):
            return enc_all[n:]
        # 兜底：若无法严格匹配，就不裁剪，避免误伤
        return enc_all
    
    def _strip_prompt_text(self, text: str, prompt: Optional[str]) -> tuple[str, bool]:
        if not prompt:
            return text, False
        enc_all = self.tokenizer(text, return_tensors="pt", add_special_tokens=True)["input_ids"][0]
        enc_prompt = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)["input_ids"][0]
        n = enc_prompt.numel()
        if enc_all.numel() >= n and torch.equal(enc_all[:n], enc_prompt):
            cont_ids = enc_all[n:]
            decoded = self.tokenizer.decode(cont_ids, skip_special_tokens=True)
            return decoded, True
        return text, False
    
    # ------------------ 候选容量 K 计算（按“重新分词 + 前缀延伸”机制） ------------------
    def _compute_k_per_byte(self, text: str) -> List[int]:
        """
        返回每个可见字节位置的候选容量 K_i：
          - token 首字节: K_i = |StartBytes|（所有 token 首字节集合大小）
          - token 内部:   K_i = 当前 token 前缀节点的子边数量（去重按字节）
        说明：
          - 使用与生成端一致的可见字节映射（to_visible_bytes）
          - 使用 adapter 的 token→bytes trie（_build_token_byte_trie），仅依赖 vocab 结构
        """
        # 构建/取缓存的 trie
        next_, term = _build_token_byte_trie(self.tokenizer)
        # StartBytes = root 的所有子边字节
        start_bytes = list(next_[0].keys())
        K_start = len(start_bytes)

        # 重新分词（当前文本自洽）
        ids = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        K_list: List[int] = []

        for tid in ids.tolist():
            bseq = to_visible_bytes(self.tokenizer, int(tid))
            if not bseq:
                continue
            # 预先走一遍前缀节点：node_after_r[r] 表示消费 r 个字节后的节点（r>=0）
            node_after_r: List[int] = [0]
            node = 0
            for bj in bseq:
                bj = int(bj) & 0xFF
                nxt = next_[node].get(bj)
                if nxt is None:
                    # 理论上不应发生；容错：中断后续并以 0 候选填充
                    node = None  # type: ignore
                    break
                node = nxt
                node_after_r.append(node)
            # 计算每个偏移的 K
            # r=0: token 首字节
            K_list.append(K_start)
            # r>=1: 取对应前缀节点的子边数
            for r in range(1, len(bseq)):
                if r < len(node_after_r) and node_after_r[r] is not None:  # type: ignore
                    node_r = node_after_r[r]
                    K_list.append(len(next_[node_r]))
                else:
                    # 容错：路径丢失时给 0
                    K_list.append(0)
        return K_list

    # ------------------ 主入口 ------------------
    def detect(
        self,
        text: str,
        return_dict: bool = True,
        prompt: Optional[str] = None,
        verify: bool = False,
    ):
        enabled = bool(getattr(self.charm_cfg, "enabled", False))

        # ---- 情况 A：启用 CHARM -> 固定 256 域字节滑窗检测 ----
        if enabled:
            vb = self._strip_prompt_bytes(text, prompt)
            L = len(vb)
            visible_bias_only = bool(getattr(self.charm_cfg, "visible_bias_only", False))
            bucket_cfg = getattr(self.charm_cfg, "bucket_detector", None)
            use_bucket_detector = bool(bucket_cfg and bucket_cfg.get("enabled", False))
            bucket_medium = (2, 64)
            bucket_high = (65, 10 ** 9)
            weight_medium = 1.0
            weight_high = 0.0
            if use_bucket_detector:
                ranges = bucket_cfg.get("ranges", {})
                bucket_medium = tuple(ranges.get("medium", [2, 64]))
                bucket_high = tuple(ranges.get("high", [65, 10 ** 9]))
                weights_cfg = bucket_cfg.get("weights", {})
                weight_medium = float(weights_cfg.get("medium", 1.0))
                weight_high = float(weights_cfg.get("high", 0.0))

            utf8_mask = self._utf8_start_mask(vb) if L > 0 else []

            positions: List[int] = []
            for t in range(self.prefix_length, L):
                if visible_bias_only and (not utf8_mask or not utf8_mask[t]):
                    continue
                positions.append(t)

            T = len(positions)
            if T <= 0:
                res = {"is_watermarked": False, "score": float('-inf')}
                if return_dict:
                    return res
                else:
                    return (False, float('-inf'))

            # 计算每个字节位置的候选容量 K（与 tokenizer/vocab 一致）
            stripped_text, matched = self._strip_prompt_text(text, prompt)
            if matched:
                K_all = self._compute_k_per_byte(stripped_text)
                start_offset = 0
            else:
                K_all = self._compute_k_per_byte(text)
                start_offset = 0
                if prompt:
                    vb_full = self._visible_bytes(text)
                    vb_prompt = self._visible_bytes(prompt)
                    if vb_full.startswith(vb_prompt):
                        start_offset = len(vb_prompt)

            green_hits = 0
            trace: List[Dict[str, Any]] = [] if verify else None
            # 统计：不同 K 的命中概率（经验估计）
            k_totals: Dict[int, int] = {}
            k_hits: Dict[int, int] = {}
            m_hits = m_count = h_hits = h_count = 0
            for step_idx, t in enumerate(positions):
                window = vb[t - self.prefix_length: t]
                # 位置盐：默认关闭，如需启用需在配置中打开 use_pos_salt
                if bool(getattr(self.charm_cfg, "use_pos_salt", False)):
                    try:
                        j = start_offset + t
                        pos_bin = ((int(j) // 8) % 16) & 0xFF
                        self.utils.set_position_bin(pos_bin)
                    except Exception:
                        self.utils.set_position_bin(0)
                green = self.utils.get_greenlist_ids_bytes(window, vocab_size=256)
                b = int(vb[t])
                hit = (b in green)
                if hit:
                    green_hits += 1
                # 记录 K_i
                j = start_offset + t
                Ki = int(K_all[j]) if 0 <= j < len(K_all) else 0
                k_totals[Ki] = k_totals.get(Ki, 0) + 1
                if hit:
                    k_hits[Ki] = k_hits.get(Ki, 0) + 1
                if use_bucket_detector and Ki > 0:
                    if bucket_medium[0] <= Ki <= bucket_medium[1]:
                        m_count += 1
                        if hit:
                            m_hits += 1
                    elif bucket_high[0] <= Ki <= bucket_high[1]:
                        h_count += 1
                        if hit:
                            h_hits += 1
                if verify:
                    rel = t - self.prefix_length
                    trace.append({
                        "step": step_idx,
                        "byte_index": t,
                        "byte_offset": rel,
                        "window_hex": window.hex(),
                        "window_len": len(window),
                        "byte": b,
                        "in_green": hit,
                        "green_size": len(green),
                        "K": Ki,
                        "utf8_start": bool(utf8_mask[t]) if utf8_mask else None,
                    })

            z_overall = self._z_from_hits(green_hits, T)
            bucket_stats: Optional[Dict[str, Any]] = None
            if use_bucket_detector:
                def bucket_z(hits: int, cnt: int) -> float:
                    return self._z_from_hits(hits, cnt) if cnt > 0 else 0.0
                z_m = bucket_z(m_hits, m_count)
                z_h = bucket_z(h_hits, h_count)
                bucket_stats = {
                    "medium": {"hits": m_hits, "count": m_count, "z": z_m, "weight": weight_medium},
                    "high": {"hits": h_hits, "count": h_count, "z": z_h, "weight": weight_high},
                    "combined": weight_medium * z_m + weight_high * z_h,
                }
                z = bucket_stats["combined"]
            else:
                z = z_overall
            # 组装 K 命中统计
            k_stats: Dict[str, Any] = {}
            if k_totals:
                by_k: Dict[int, Dict[str, float]] = {}
                for k, tot in sorted(k_totals.items()):
                    hits = k_hits.get(k, 0)
                    rate = float(hits) / float(tot) if tot > 0 else 0.0
                    by_k[k] = {"hits": int(hits), "count": int(tot), "rate": rate}
                k_stats = {"by_k": by_k}

            result = {"is_watermarked": bool(z > self.z_threshold), "score": z, "k_stats": k_stats}
            if bucket_stats is not None:
                result["bucket_stats"] = bucket_stats
                result["raw_score_all"] = z_overall
            if verify:
                result["trace"] = trace
            return result if return_dict else (result["is_watermarked"], z)

        # ---- 情况 B：未启用 CHARM 或未开 fixed_256 -> 回落到 token 域检测 ----
        # 这里严格对齐 prompt 的 token 级裁剪；verify 会给出逐步命中情况
        input_ids = self._strip_prompt_tokens(text, prompt).to(self.device)
        num_tokens_scored = int(input_ids.numel()) - self.prefix_length
        if num_tokens_scored <= 0:
            res = {"is_watermarked": False, "score": float('-inf')}
            if return_dict:
                return res
            else:
                return (False, float('-inf'))

        green_hits = 0
        trace_tok: List[Dict[str, Any]] = [] if verify else None

        for idx in range(self.prefix_length, input_ids.numel()):
            prefix = input_ids[:idx]
            token = int(input_ids[idx].item())
            gl = self.utils.get_greenlist_ids(prefix, vocab_size=getattr(self.utils.config, "vocab_size", None) or self.tokenizer.vocab_size)
            hit = (token in gl)
            if hit:
                green_hits += 1

            if verify:
                # 尽量压缩 trace 体积：给 token id / token 文本 / greenlist 规模，避免吐出全量 green_ids
                try:
                    tok_text = self.tokenizer.decode([token], skip_special_tokens=False)
                except Exception:
                    tok_text = ""
                trace_tok.append({
                    "step": idx - self.prefix_length,
                    "token_id": token,
                    "token_text": tok_text,
                    "in_green": hit,
                    "green_size": len(gl),
                    # "green_ids": gl,  # 如需要，可打开；体积可能很大
                })

        # 与官方相同的 z 统计公式
        z = self.utils._compute_z_score(green_hits, num_tokens_scored)
        result = {"is_watermarked": bool(z > self.z_threshold), "score": float(z)}
        if verify:
            result["trace"] = trace_tok
        return result if return_dict else (result["is_watermarked"], z)
