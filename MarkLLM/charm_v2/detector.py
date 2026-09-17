from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from transformers import PreTrainedTokenizerBase

from MarkLLM.charm.adapter import _build_token_byte_trie
from MarkLLM.charm.visible import to_visible_bytes  # 仅用于 K 计算（沿用你原来的方式）
from MarkLLM.charm_v2.prf import TokenPrefixBytePRF
from MarkLLM.charm_v2.prf_trace import PRFTracer
from MarkLLM.charm_v2.vocab import build_byte_vocab, first_visible_byte_info


class CharmDetectorV2:
    """
    TokenPrefixBytePRF 对齐版检测器：

    - PRF 输入：token_ids_prefix（前 N 个 token） + 当前 token 已确定 prefix_bytes + byte_pos
    - prefix_length 语义：token_prefix_length（burn-in：前 prefix_length 个 token 不计入统计）
    - 仍保留：
        * overall z
        * segmented_start / nonsegmented_start 的 first_visible_byte 桶
        * K=1..256 的统计（基于 trie 估计）
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        *,
        hash_key: int,
        gamma: float,
        prefix_length: int,
        z_threshold: float,
        weight_first: float = 1.0,
        weight_other: float = 0.0,
        f_scheme: str = "additive",
        prf_device: str = "cpu",
    ) -> None:
        self.tokenizer = tokenizer
        self.gamma = float(gamma)

        # ✅ 注意：prefix_length 现在表示 token_prefix_length
        self.prefix_length = int(prefix_length)

        self.z_threshold = float(z_threshold)
        self.weight_first = float(weight_first)
        self.weight_other = float(weight_other)

        # ✅ 使用 TokenPrefixBytePRF
        self.prf = TokenPrefixBytePRF(
            hash_key=int(hash_key),
            token_prefix_length=self.prefix_length,
            gamma=self.gamma,
            f_scheme=f_scheme,
            device=prf_device,
        )

        # trie / byte_vocab（用于 K 统计 & first_visible_byte 定位）
        self._trie_next, self._trie_term = _build_token_byte_trie(self.tokenizer)
        self.byte_vocab = build_byte_vocab(self.tokenizer)
        self._debug_tracer: Optional[PRFTracer] = None
        self._debug_byte_pos_filter: Optional[int] = None

    def set_debug_tracer(self, tracer: Optional[PRFTracer], *, byte_pos_filter: Optional[int] = None) -> None:
        self._debug_tracer = tracer
        self._debug_byte_pos_filter = byte_pos_filter

    def _z_from_hits(self, hits: int, trials: int) -> float:
        if trials <= 0:
            return float("-inf")
        eps = 1e-12
        g = float(self.gamma)
        if not (0.0 < g < 1.0):
            g = min(max(g, eps), 1.0 - eps)
        numer = float(hits) - float(trials) * g
        denom_sq = float(trials) * g * (1.0 - g)
        denom = math.sqrt(max(denom_sq, eps))
        return numer / denom

    def _token_K_list(self, token_id: int) -> List[int]:
        """
        为单个 token 的每个 byte 位置产生 K 值：
          - r=0: K_start = |StartBytes|
          - r>=1: trie 节点出边数
        这里沿用你之前 _compute_k_per_byte 的逻辑，但按 token 逐个算，避免全局对齐问题。
        """
        next_ = self._trie_next
        start_bytes = list(next_[0].keys())
        K_start = len(start_bytes)

        bseq = to_visible_bytes(self.tokenizer, int(token_id))
        if not bseq:
            return []

        # 走 trie 得到每个 r 后的节点
        node_after_r: List[Optional[int]] = [0]
        node: Optional[int] = 0
        for bj in bseq:
            bj = int(bj) & 0xFF
            nxt = next_[node].get(bj) if node is not None else None
            if nxt is None:
                node = None
                node_after_r.append(None)
                break
            node = nxt
            node_after_r.append(node)

        K_list: List[int] = []
        # r=0
        K_list.append(K_start)
        # r>=1
        for r in range(1, len(bseq)):
            if r < len(node_after_r) and node_after_r[r] is not None:
                node_r = int(node_after_r[r])
                K_list.append(len(next_[node_r]))
            else:
                K_list.append(0)
        return K_list

    def detect(
        self,
        text: str,
        return_dict: bool = True,
        verify: bool = False,
    ):
        # 1) tokenize
        enc = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        ids = enc["input_ids"][0].tolist()
        pieces = self.tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=False)

        if len(ids) <= self.prefix_length:
            res = {"is_watermarked": False, "score": float("-inf"), "k_stats": {}}
            return res if return_dict else (False, float("-inf"))

        # 2) 统计变量
        total_hits = 0
        total_trials = 0

        seg_fb_hits = seg_fb_count = 0
        nonseg_fb_hits = nonseg_fb_count = 0

        K_MAX = 256
        k_hits = [0] * (K_MAX + 1)
        k_count = [0] * (K_MAX + 1)

        trace: Optional[List[Dict[str, Any]]] = [] if verify else None

        # 3) 逐 token、逐 byte 扫描（PRF 完全对齐生成端：token_prefix + prefix_bytes + byte_pos）
        global_byte_index = 0  # 仅用于 trace / debug 展示

        for tok_idx, (tid, piece) in enumerate(zip(ids, pieces)):
            payload = self.byte_vocab.bytes_of(int(tid))
            if not payload:
                continue

            # 当前 token 的 first_visible_byte 位置（用于分桶）
            _, local_offset = first_visible_byte_info(payload, piece)

            # 当前 token 的 K 列表（用于 k_stats）
            K_list = self._token_K_list(int(tid))
            # 若 K_list 长度与 payload 不一致（有些 tokenizer 映射会不一致），就用 0 fallback
            # （不影响 watermark 判定，只影响 k_stats 的解释）
            # byte_pos -> Ki:
            #   if byte_pos < len(K_list): Ki = K_list[byte_pos] else 0

            # burn-in：前 prefix_length 个 token 不计入统计（但仍推进 global index）
            score_this_token = (tok_idx >= self.prefix_length)

            for byte_pos in range(len(payload)):
                b = int(payload[byte_pos]) & 0xFF
                token_prefix = ids[:tok_idx]  # 不包含当前 token
                prefix_bytes = payload[:byte_pos]  # 当前 token 已确定的 bytes

                # 始终记录 tracer，便于对齐调试（即便未计分也写入，但 greenlist 为空）
                if score_this_token:
                    green = self.prf.greenlist(
                        token_prefix,
                        prefix_bytes,
                        byte_pos,
                    )
                    hit = (b in green)
                else:
                    green = []
                    hit = False

                if score_this_token:
                    total_trials += 1
                    if hit:
                        total_hits += 1

                    Ki = int(K_list[byte_pos]) if byte_pos < len(K_list) else 0
                    if 1 <= Ki <= K_MAX:
                        k_count[Ki] += 1
                        if hit:
                            k_hits[Ki] += 1

                    # first_visible_byte 分桶（沿用你原来的 segmented_start / nonsegmented_start）
                    if local_offset >= 0 and byte_pos == local_offset:
                        is_start = piece.startswith("Ġ") or piece.startswith("▁")
                        if is_start:
                            seg_fb_count += 1
                            if hit:
                                seg_fb_hits += 1
                        else:
                            nonseg_fb_count += 1
                            if hit:
                                nonseg_fb_hits += 1

                if self._debug_tracer is not None:
                    if self._debug_byte_pos_filter is None or byte_pos == self._debug_byte_pos_filter:
                        self._debug_tracer.log(
                            "det",
                            token_index=tok_idx,
                            token_ids=token_prefix,
                            prefix_bytes=bytes(prefix_bytes),
                        byte_pos=byte_pos,
                        greenlist=green,
                        extra={"byte": b, "hit": bool(hit)},
                    )

                if verify and trace is not None:
                    trace.append(
                        {
                            "token_index": tok_idx,
                            "token_id": int(tid),
                            "piece": piece,
                            "global_byte_index": global_byte_index,
                            "byte_pos": byte_pos,
                            "byte": b,
                            "in_green": bool(hit),
                            "green_size": len(green),
                            "Ki": int(K_list[byte_pos]) if byte_pos < len(K_list) else 0,
                            "prefix_bytes_hex": prefix_bytes.hex(),
                            "ctx_tail": token_prefix[-self.prefix_length :],
                        }
                    )

                global_byte_index += 1

        # 4) 计算 z + 加权 score
        z_overall = self._z_from_hits(total_hits, total_trials)
        z_seg = self._z_from_hits(seg_fb_hits, seg_fb_count) if seg_fb_count > 0 else float("-inf")
        z_nonseg = self._z_from_hits(nonseg_fb_hits, nonseg_fb_count) if nonseg_fb_count > 0 else float("-inf")

        # 这里保持你原来的组合逻辑（如果某个桶 count=0 会是 -inf，你可以按需要改成 0）
        score = self.weight_first * z_seg + self.weight_other * z_nonseg

        # 5) K stats
        k_stats: Dict[int, Dict[str, float]] = {}
        for k in range(1, K_MAX + 1):
            c = k_count[k]
            if c <= 0:
                continue
            h = k_hits[k]
            k_stats[k] = {
                "hits": float(h),
                "count": float(c),
                "ratio": float(h / float(c)),
            }

        result: Dict[str, Any] = {
            "is_watermarked": bool(score > self.z_threshold),
            "score": float(score),
            "raw_score_all": float(z_overall),
            "bucket_stats": {
                "segmented_start": {
                    "hits": seg_fb_hits,
                    "count": seg_fb_count,
                    "z": float(z_seg) if seg_fb_count > 0 else float("-inf"),
                    "weight": float(self.weight_first),
                },
                "nonsegmented_start": {
                    "hits": nonseg_fb_hits,
                    "count": nonseg_fb_count,
                    "z": float(z_nonseg) if nonseg_fb_count > 0 else float("-inf"),
                    "weight": float(self.weight_other),
                },
                "combined": float(score),
            },
            "k_stats": k_stats,
            "first_byte_stats": {
                "segmented_start": {
                    "hits": seg_fb_hits,
                    "count": seg_fb_count,
                    "hit_rate": (seg_fb_hits / seg_fb_count) if seg_fb_count > 0 else 0.0,
                },
                "nonsegmented_start": {
                    "hits": nonseg_fb_hits,
                    "count": nonseg_fb_count,
                    "hit_rate": (nonseg_fb_hits / nonseg_fb_count) if nonseg_fb_count > 0 else 0.0,
                },
            },
            "meta": {
                "token_prefix_length": int(self.prefix_length),
                "total_trials": int(total_trials),
            },
        }
        if verify and trace is not None:
            result["trace"] = trace

        return result if return_dict else (result["is_watermarked"], score)
