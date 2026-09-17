# watermark/algorithms/kgw.py
# Copyright 2024 THU-BPM MarkLLM.
# Licensed under the Apache License, Version 2.0

from __future__ import annotations
import torch
import math
from math import sqrt
from typing import Optional, List
import hashlib
import hmac
import random

from MarkLLM.charm.adapter import CharmAdapter, EPS, NUM_BYTE_VALUES
from MarkLLM.charm.config import CharmConfig
from MarkLLM.charm.detector import CharmDetector
from MarkLLM.watermark.base import BaseWatermark, BaseConfig
from MarkLLM.utils.transformers_config import TransformersConfig
from transformers import LogitsProcessor, LogitsProcessorList
from MarkLLM.visualize.data_for_visualization import DataForVisualization


class CharmKGWConfig(BaseConfig):
    """Config class for KGW algorithm"""
    def initialize_parameters(self) -> None:
        self.gamma = self.config_dict['gamma']
        self.delta = self.config_dict['delta']
        self.hash_key = self.config_dict['hash_key']
        self.z_threshold = self.config_dict['z_threshold']
        self.prefix_length = self.config_dict['prefix_length']
        self.f_scheme = self.config_dict['f_scheme']
        self.window_scheme = self.config_dict['window_scheme']
        # CHARM 适配器配置（已默认固定 256 字节域：enabled / normalize_form / prefix_length）
        charm_cfg_dict = self.config_dict.get('charm_cfg', {})
        self.charm_cfg = CharmConfig(
            enabled=charm_cfg_dict.get('enabled', True),
            normalize_form=charm_cfg_dict.get('normalize_form', "NONE"),
            use_unbiased_prf=charm_cfg_dict.get('use_unbiased_prf', True),
            use_pos_salt=charm_cfg_dict.get('use_pos_salt', False),
            prefix_length=charm_cfg_dict.get('prefix_length', self.prefix_length),
            debug_token_dump=charm_cfg_dict.get('debug_token_dump', 0),
            debug_full_byte_probs=charm_cfg_dict.get('debug_full_byte_probs', False),
            debug_topk=charm_cfg_dict.get('debug_topk', 8),
            debug_list_ops=charm_cfg_dict.get('debug_list_ops', False),
            visible_bias_only=charm_cfg_dict.get('visible_bias_only', False),
        )
        # 熵 gating 可在 config 或 charm_cfg 中配置
        self.h_low = float(self.config_dict.get("h_low", charm_cfg_dict.get("h_low", 0.0)))
        # extra detector options
        for key, val in charm_cfg_dict.items():
            if not hasattr(self.charm_cfg, key):
                setattr(self.charm_cfg, key, val)

    @property
    def algorithm_name(self) -> str:
        return 'CharmKGW'


class CharmKGWUtils:
    """
    - 保留原 token 域 PRF/greenlist（向后兼容）
    - 新增：基于“可见字节窗口”的 PRF/greenlist（fixed-256 专用，跨设备可复现）
    """
    def __init__(self, config: CharmKGWConfig, *args, **kwargs) -> None:
        self.config = config
        # 统一在 CPU 做 PRF，跨设备可复现
        self._cpu = torch.device('cpu')
        # 位置盐（由外部注入）：默认 0；由生成/检测流程设置
        self._pos_bin: int = 0
        self.f_scheme_map = {
            "time": self._f_time,
            "additive": self._f_additive,
            "skip": self._f_skip,
            "min": self._f_min,
        }

    # -------- 原 token 域（保留兼容） --------
    def _prf(self, vocab_size: int, device) -> torch.Tensor:
        g = torch.Generator(device=device)
        g.manual_seed(int(self.config.hash_key))
        return torch.randperm(vocab_size, device=device, generator=g)

    def _f(self, input_ids: torch.LongTensor, vocab_size: int) -> int:
        return int(self.f_scheme_map[self.config.f_scheme](input_ids, vocab_size))

    def _f_time(self, input_ids: torch.LongTensor, vocab_size: int) -> int:
        prf = self._prf(vocab_size, input_ids.device)
        acc = 1
        for i in range(0, self.config.prefix_length):
            acc *= input_ids[-1 - i].item()
        return prf[acc % vocab_size].item()

    def _f_additive(self, input_ids: torch.LongTensor, vocab_size: int) -> int:
        prf = self._prf(vocab_size, input_ids.device)
        acc = 0
        for i in range(0, self.config.prefix_length):
            acc += input_ids[-1 - i].item()
        return prf[acc % vocab_size].item()

    def _f_skip(self, input_ids: torch.LongTensor, vocab_size: int) -> int:
        prf = self._prf(vocab_size, input_ids.device)
        return prf[input_ids[-self.config.prefix_length].item() % vocab_size].item()

    def _f_min(self, input_ids: torch.LongTensor, vocab_size: int) -> int:
        prf = self._prf(vocab_size, input_ids.device)
        return min(
            prf[input_ids[-1 - i].item() % vocab_size].item()
            for i in range(0, self.config.prefix_length)
        )

    def _get_greenlist_ids_left(self, input_ids: torch.LongTensor, vocab_size: int) -> List[int]:
        seed = (int(self.config.hash_key) * self._f(input_ids, vocab_size)) % vocab_size
        g = torch.Generator(device=input_ids.device)
        g.manual_seed(int(seed))
        greenlist_size = int(vocab_size * self.config.gamma)
        if greenlist_size <= 0:
            return []
        vocab_perm = torch.randperm(vocab_size, device=input_ids.device, generator=g)
        return vocab_perm[:greenlist_size].tolist()

    def _get_greenlist_ids_self(self, input_ids: torch.LongTensor, vocab_size: int) -> List[int]:
        greenlist_size = int(vocab_size * self.config.gamma)
        if greenlist_size <= 0:
            return []
        prf = self._prf(vocab_size, input_ids.device)
        res: List[int] = []
        f_x = self._f(input_ids, vocab_size)
        for k in range(vocab_size):
            h_k = f_x * int(prf[k])
            g = torch.Generator(device=input_ids.device)
            g.manual_seed(int(h_k % vocab_size))
            vocab_perm = torch.randperm(vocab_size, device=input_ids.device, generator=g)
            if k in vocab_perm[:greenlist_size]:
                res.append(k)
        return res

    def get_greenlist_ids(self, input_ids: torch.LongTensor, vocab_size: int | None = None) -> List[int]:
        v = int(vocab_size or self.config.vocab_size)
        if self.config.window_scheme == "left":
            return self._get_greenlist_ids_left(input_ids, v)
        else:
            return self._get_greenlist_ids_self(input_ids, v)

    # -------- 新增：字节域（fixed-256 专用） --------
    def _prf_cpu(self, vocab_size: int) -> torch.Tensor:
        g = torch.Generator(device=self._cpu)
        g.manual_seed(int(self.config.hash_key) ^ int(vocab_size))  # 混入 vocab_size 使不同域独立
        return torch.randperm(vocab_size, device=self._cpu, generator=g)

    # ===== 新版：直方图（bag-of-bytes）PRF，用于 fixed-256 字节域 =====
    def _key_to_bytes(self, k: int, length: int = 32) -> bytes:
        # 将整型 hash_key 规范化为定长字节串
        return int(k).to_bytes(length, "big", signed=False)

    def _derive_subkeys(self, master_key: bytes, M: int) -> List[bytes]:
        subs: List[bytes] = []
        for j in range(M):
            h = hashlib.sha256()
            h.update(master_key)
            h.update(b"subkey")
            h.update(j.to_bytes(4, "big"))
            subs.append(h.digest())  # 32 bytes
        return subs

    def _perm_from_key(self, key: bytes) -> List[int]:
        # 用 key 派生确定性 seed，Fisher–Yates 得到 0..255 的置换
        seed = int.from_bytes(hashlib.sha256(key).digest(), "big") % (2**32)
        rng = random.Random(seed)
        perm = list(range(256))
        for i in range(255, 0, -1):
            j = rng.randint(0, i)
            perm[i], perm[j] = perm[j], perm[i]
        return perm

    def _weights_from_key(self, key: bytes) -> List[int]:
        # 为直方图各字节生成权重 r_j[b] ∈ {1,3,5,...,255}（奇数），保证 mod-256 的可逆性与良好分布
        seed = int.from_bytes(hashlib.sha256(key + b"/weights").digest(), "big") % (2**32)
        rng = random.Random(seed)
        weights = []
        for _ in range(256):
            x = rng.randrange(0, 128)  # 0..127
            weights.append((x << 1) | 1)  # 奇数 1..255
        return weights

    def _hmac_sha256_int(self, key: bytes, data: bytes) -> int:
        return int.from_bytes(hmac.new(key, data, hashlib.sha256).digest(), "big")

    def _init_hist_prf_once(self):
        # 一次性初始化：M、子密钥、置换、权重、量化位数与计数裁剪
        if getattr(self, "_hist_prf_ready", False):
            return
        self._M = 4
        # 自适应量化：窗口小则降低量化步长，避免摘要退化为全零
        w = int(getattr(self.config, "prefix_length", 1))
        if w <= 4:
            self._q_bits = 0
        elif w <= 8:
            self._q_bits = 1
        else:
            self._q_bits = 2
        self._cnt_clip = 8
        master = self._key_to_bytes(getattr(self.config, "hash_key", 0))
        self._subkeys = self._derive_subkeys(master, self._M)
        self._perms = [self._perm_from_key(sk) for sk in self._subkeys]
        self._weights = [self._weights_from_key(sk) for sk in self._subkeys]
        # pi_0 的顺序用于平票打破
        self._order_map = {b: i for i, b in enumerate(self._perms[0])} if self._perms else {b: b for b in range(256)}
        self._hist_prf_ready = True
    
    def set_position_bin(self, pos_bin: int) -> None:
        """由调用方注入位置盐（可见字节索引的粗分桶），用于无偏 PRF 去相关。"""
        try:
            self._pos_bin = int(pos_bin) & 0xFF
        except Exception:
            self._pos_bin = 0
    
    def _digest_window_hist(self, byte_window: bytes) -> bytes:
        """
        稳定摘要：256维计数直方图（clip 到 _cnt_clip），每个计数按 2^_q_bits 量化，再编码成 bytes。
        """
        clip = int(self._cnt_clip)
        q = int(self._q_bits)
        cnt = [0] * 256
        if byte_window:
            for b in byte_window:
                bb = int(b) & 0xFF
                if cnt[bb] < clip:
                    cnt[bb] += 1
        # 量化（右移 q 后再左移 q，相当于以 2^q 为步长）
        quant = [(c >> q) << q for c in cnt]
        # 简单编码为 256字节（每项0..255）
        return bytes(quant)

    def get_greenlist_ids_bytes(self, byte_window: bytes, vocab_size: int | None = None) -> List[int]:
        """
        无偏 PRF（默认）：
          green(window, b) = 1{ HMAC_k(digest(window) || b) < gamma }
        保留鲁棒性：digest(window) 使用“量化的直方图”作为稳定摘要。
        兼容旧版：如配置 use_unbiased_prf=False，则退回“直方图 PRF（shift→perm→top-K）”。
        """
        self._init_hist_prf_once()
        v = int(vocab_size or 256)
        if v != 256 or v <= 0:
            return []
        if bool(getattr(self.config, "charm_cfg", None)) and bool(getattr(self.config.charm_cfg, "use_unbiased_prf", True)):
            # 无偏：逐字节抛硬币
            gamma = float(self.config.gamma)
            if not (0.0 < gamma < 1.0):
                gamma = min(max(gamma, 1e-6), 1.0 - 1e-6)
            digest = self._digest_window_hist(byte_window)
            key = self._subkeys[0] if self._subkeys else self._key_to_bytes(getattr(self.config, "hash_key", 0))
            threshold = int(gamma * (1 << 32))
            use_pos = bool(getattr(self.config.charm_cfg, "use_pos_salt", False))
            pos_tag = bytes([getattr(self, "_pos_bin", 0) & 0xFF]) if use_pos else b""
            green: List[int] = []
            for b in range(256):
                if use_pos:
                    msg = b"HIST|" + digest + b"|B" + bytes([b]) + b"|P" + pos_tag
                else:
                    msg = b"HIST|" + digest + b"|B" + bytes([b])
                hv = hmac.new(key, msg, hashlib.sha256).digest()
                val = int.from_bytes(hv[:4], "big")
                if val < threshold:
                    green.append(b)
            return green
        else:
            # 回退：旧直方图 PRF（shift→perm→top-K）
            v = int(vocab_size or 256)
            if v != 256 or v <= 0:
                return []
            K = int(round(float(self.config.gamma) * 256.0))
            if K <= 0:
                return []
            if K > 256:
                K = 256
            # 统计窗口直方图（可选裁剪，抑制重复字符的极端影响）
            cnt = [0] * 256
            if byte_window:
                clip = int(self._cnt_clip)
                for b in byte_window:
                    bb = int(b) & 0xFF
                    if cnt[bb] < clip:
                        cnt[bb] += 1
            # 多通道投票
            votes = [0] * 256
            q = int(self._q_bits)
            mask_q = ~((1 << q) - 1) & 0xFF  # 量化到 2^q 的步长
            for j in range(self._M):
                c_j = self._hmac_sha256_int(self._subkeys[j], b"offset") & 0xFF
                wj = self._weights[j]
                acc = c_j
                for b in range(256):
                    c = cnt[b]
                    if c:
                        acc = (acc + (wj[b] * c)) & 0xFF
                shift = acc & mask_q
                pi = self._perms[j]
                for r in range(K):
                    bb = pi[(shift + r) & 0xFF]
                    votes[bb] += 1
            idxs = list(range(256))
            order_map = self._order_map
            idxs.sort(key=lambda b: (-votes[b], order_map.get(b, b)))
            return idxs[:K]

    def score_sequence(self, input_ids: torch.Tensor) -> tuple[float, list[int]]:
        num_tokens_scored = len(input_ids) - self.config.prefix_length
        if num_tokens_scored < 1:
            raise ValueError(
                f"Must have at least 1 token to score after the first min_prefix_len={self.config.prefix_length}."
            )
        green_token_count = 0
        green_token_flags = [-1 for _ in range(self.config.prefix_length)]
        for idx in range(self.config.prefix_length, len(input_ids)):
            curr_token = int(input_ids[idx].item())
            greenlist_ids = self.get_greenlist_ids(input_ids[:idx])  # 回落到模型词表
            if curr_token in greenlist_ids:
                green_token_count += 1
                green_token_flags.append(1)
            else:
                green_token_flags.append(0)
        z_score = self._compute_z_score(green_token_count, num_tokens_scored)
        return z_score, green_token_flags


class CharmKGWLogitsProcessor(torch.nn.Module):
    """LogitsProcessor for KGW algorithm（支持 fixed-256 + trace）"""

    def __init__(self, config: CharmKGWConfig, utils: CharmKGWUtils, *args, **kwargs) -> None:
        super().__init__()
        self.config = config
        self.utils = utils
        self.byte_window: Optional[bytes] = None
        self._pos_bin: int = 0
        # trace
        self._record_trace: bool = False
        self._trace: List[dict] = []

    # 遵循 transformers LogitsProcessor 接口
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # 不加偏置
        if float(self.config.delta) == 0.0:
            return scores

        batch, local_vocab = scores.shape[0], scores.shape[-1]
        # 固定为 256 字节域模式
        use_fixed_256 = True
        prefix_len = int(getattr(self.config, "prefix_length", 0))

        # --- 固定 256 字节域 ---
        if use_fixed_256:
            if local_vocab != 256:
                return scores
            if prefix_len > 0 and (self.byte_window is None or len(self.byte_window) < prefix_len):
                return scores
            # 可选位置盐：默认关闭
            if bool(getattr(self.config.charm_cfg, "use_pos_salt", False)):
                try:
                    self.utils.set_position_bin(int(self._pos_bin))
                except Exception:
                    self.utils.set_position_bin(0)
            green_ids = self.utils.get_greenlist_ids_bytes(self.byte_window or b"", vocab_size=local_vocab)

            # 记录 trace（窗口 & greenlist）
            if self._record_trace:
                self._trace.append({
                    "window_hex": (self.byte_window or b"").hex(),
                    "window_len": len(self.byte_window or b""),
                    "green_ids": list(green_ids),
                })

            if not green_ids:
                return scores

            # -------- 按当前 byte 分布的熵做“容量 gating” --------
            # 低于 h_low => 直接屏蔽（scale=0），否则保持原强度（scale=1）。
            h_low = float(getattr(self.config, "h_low", getattr(self.config.charm_cfg, "h_low", 0.0)))
            scale = 1.0
            if h_low > 0.0 and scores.numel() > 0:
                probs = torch.softmax(scores, dim=-1)[0]  # [256]
                H = -(probs * torch.log(probs.clamp_min(EPS))).sum()
                H_max = math.log(NUM_BYTE_VALUES)
                h_norm = float((H / H_max).item()) if H_max > 0 else 0.0
                if h_norm < h_low:
                    scale = 0.0

            mask = torch.zeros_like(scores, dtype=torch.bool)
            mask[0, torch.tensor(green_ids, device=scores.device)] = True
            delta_eff = float(self.config.delta) * scale
            if delta_eff != 0.0:
                scores[mask] = scores[mask] + delta_eff
            return scores

        return scores

    # adapter 注入当前窗口
    def set_byte_window(self, b: Optional[bytes]):
        self.byte_window = b
    
    def set_byte_index(self, idx: int, *, stride: int = 8, modulo: int = 16):
        """
        注入当前位置的可见字节索引（或其粗分桶），用于 PRF 的位置盐。
        选择较大的 stride/modulo 可在“去相关”与“鲁棒性”间平衡。
        """
        if bool(getattr(self.config.charm_cfg, "use_pos_salt", False)):
            try:
                i = int(idx)
                bin_id = ((i // int(stride)) % int(modulo)) & 0xFF
            except Exception:
                bin_id = 0
            self._pos_bin = bin_id

    # trace 控制
    def enable_trace(self, on: bool = True):
        self._record_trace = bool(on)
        self._trace = []

    def get_trace(self, clear: bool = True) -> List[dict]:
        t = self._trace
        if clear:
            self._trace = []
        return t

    def trace_observation(self, byte_value: int):
        if self._record_trace and self._trace and "byte" not in self._trace[-1]:
            self._trace[-1]["byte"] = int(byte_value)
            gi = self._trace[-1].get("green_ids", [])
            self._trace[-1]["in_green"] = (int(byte_value) in gi)


class CharmKGW(BaseWatermark):
    """Top-level class for KGW algorithm."""

    def __init__(self, algorithm_config: str | CharmKGWConfig, transformers_config: TransformersConfig | None = None, *args, **kwargs) -> None:
        if isinstance(algorithm_config, str):
            self.config = CharmKGWConfig(algorithm_config, transformers_config)
        elif isinstance(algorithm_config, CharmKGWConfig):
            self.config = algorithm_config
        else:
            raise TypeError("algorithm_config must be either a path string or a KGWConfig instance")
    
        self.utils = CharmKGWUtils(self.config)
        self.logits_processor = CharmKGWLogitsProcessor(self.config, self.utils)

    def _prepare_prompt_inputs(self, prompt: str):
        tokenizer = self.config.generation_tokenizer
        device = self.config.device
        chat_template = getattr(tokenizer, "chat_template", None)
        if chat_template and hasattr(tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            encoded_prompt = tokenizer(
                prompt_text,
                return_tensors="pt",
                add_special_tokens=False,
            )
        else:
            encoded_prompt = tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=True,
            )
        return encoded_prompt.to(device)

    # 生成（可选返回 trace）
    def generate_watermarked_text(self, prompt: str, return_trace: bool = False, *args, **kwargs):
        # 开/关 trace
        self.logits_processor.enable_trace(return_trace)

        encoded_prompt = self._prepare_prompt_inputs(prompt)

        logits_processors = LogitsProcessorList([self.logits_processor])

        out_ids = CharmAdapter.generate(
            model=self.config.generation_model,
            tokenizer=self.config.generation_tokenizer,
            prompt_inputs=encoded_prompt,
            logits_processors=logits_processors,
            gen_kwargs=self.config.gen_kwargs,
            charm_cfg=self.config.charm_cfg,   # enabled / normalize_form / prefix_length
        )

        text = self.config.generation_tokenizer.batch_decode(out_ids, skip_special_tokens=True)[0]

        if return_trace:
            gen_trace = self.logits_processor.get_trace(clear=True)
            return text, gen_trace
        return text

    def generate_unwatermarked_text(self, prompt: str, *args, **kwargs) -> str:
        encoded_prompt = self._prepare_prompt_inputs(prompt)

        plain_kwargs = dict(self.config.gen_kwargs)
        plain_kwargs["debug_token_dump"] = int(getattr(self.config.charm_cfg, "debug_token_dump", 0))
        out_ids = CharmAdapter.generate_plain(
            model=self.config.generation_model,
            tokenizer=self.config.generation_tokenizer,
            prompt_inputs=encoded_prompt,
            gen_kwargs=plain_kwargs,
        )

        text = self.config.generation_tokenizer.batch_decode(out_ids, skip_special_tokens=True)[0]
        return text
    
    # 检测（支持 prompt 对齐 + verify）
    def detect_watermark(self, text: str, return_dict: bool = True, prompt: Optional[str] = None, verify: bool = False, *args, **kwargs):
        detector = CharmDetector(
            tokenizer=self.config.generation_tokenizer,
            algo_utils=self.utils,
            charm_cfg=self.config.charm_cfg,
            device=self.config.device,
            gamma=self.config.gamma,
            prefix_length=self.config.prefix_length,
            z_threshold=self.config.z_threshold,
            normalize_form=getattr(self.config.charm_cfg, "normalize_form", None),
        )
        return detector.detect(text, return_dict=return_dict, prompt=prompt, verify=verify)
        
