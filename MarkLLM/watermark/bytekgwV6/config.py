from __future__ import annotations

import json
from typing import Any, Dict, List, Union

from MarkLLM.watermark.base import BaseConfig


def _get(d: Dict[str, Any], k: str, default: Any) -> Any:
    return d.get(k, default)


def _to_bytes(key: Union[int, str, bytes]) -> bytes:
    """
    Normalize hash_key into bytes.
    - int: little-endian 16 bytes
    - hex str: parsed then treated as int
    - bytes: returned as-is
    """
    if isinstance(key, bytes):
        return key
    if isinstance(key, str):
        s = key.strip()
        if s.startswith("0x") or s.startswith("0X"):
            try:
                key_int = int(s, 16)
                return key_int.to_bytes(16, "little", signed=False)
            except Exception:
                pass
        # fallback: encode raw string
        return s.encode("utf-8")
    try:
        key_int = int(key)
        return key_int.to_bytes(16, "little", signed=False)
    except Exception:
        return b"default-key"


class ByteKGWv6Config(BaseConfig):
    """
    Minimal config binder for ByteKGWv6.

    Keep it KGW-style简洁：只暴露核心水印参数，其余走代码默认。
    """

    @property
    def algorithm_name(self) -> str:
        return "ByteKGWv6"

    def initialize_parameters(self) -> None:
        d = self.config_dict

        # 核心水印参数
        self.delta: float = float(_get(d, "delta", 2.0))
        self.n_bytes: int = int(_get(d, "n_bytes", 3))
        self.seed_window_chars: int = int(_get(d, "seed_window_chars", 10))

        # PRF 主密钥（存 bytes 方便直接传给 RobustPartitioner）
        self.hash_key_raw: Union[int, str, bytes] = _get(d, "hash_key", 15485863)
        self.hash_key_bytes: bytes = _to_bytes(self.hash_key_raw)

        # 轻量默认项（可不在 JSON 里出现）
        self.m_bits: int = int(_get(d, "m_bits", 256))
        self.target_anchors: int = int(_get(d, "target_anchors", 96))
        self.k_choices: List[int] = list(_get(d, "k_choices", [4, 5, 6]))
        self.normalize_whitespace: bool = bool(_get(d, "normalize_whitespace", True))
        self.add_special_tokens: bool = bool(_get(d, "add_special_tokens", True))

        # 可选检测阈值（如需 z 判定）
        self.z_threshold: float = float(_get(d, "z_threshold", 4.0))
        self.min_tokens: int = int(_get(d, "min_tokens", 0))

        # 生成参数覆盖（可空）
        self._gen_overrides: Dict[str, Any] = dict(_get(d, "gen", {}) or {})

    def gen_cfg_dict(self) -> Dict[str, Any]:
        """
        合并 TransformersConfig.gen_kwargs 与用户的 gen 覆盖。
        """
        cfg = dict(self.gen_kwargs)
        for k, v in self._gen_overrides.items():
            cfg[k] = v
        # 确保常用值存在
        cfg.setdefault("do_sample", False)
        cfg.setdefault("temperature", 1.0)
        cfg.setdefault("top_p", 1.0)
        cfg.setdefault("top_k", 0)
        cfg.setdefault("repetition_penalty", 1.0)
        cfg.setdefault("max_new_tokens", cfg.get("max_new_tokens", 128))

        if getattr(self.generation_tokenizer, "eos_token_id", None) is not None:
            cfg.setdefault("eos_token_id", int(self.generation_tokenizer.eos_token_id))
        if getattr(self.generation_tokenizer, "pad_token_id", None) is not None:
            cfg.setdefault("pad_token_id", int(self.generation_tokenizer.pad_token_id))
        else:
            if cfg.get("eos_token_id", None) is not None:
                cfg.setdefault("pad_token_id", int(cfg["eos_token_id"]))

        cfg.setdefault("add_special_tokens", bool(self.add_special_tokens))
        return cfg


__all__ = ["ByteKGWv6Config"]
