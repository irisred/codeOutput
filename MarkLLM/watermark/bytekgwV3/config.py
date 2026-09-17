# MarkLLM/watermark/bytekgwV2/config.py
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ByteKGWConfig:
    algorithm_name: str = "ByteKGW"
    gamma: float = 0.5
    delta: float = 1.0
    hash_key: int = 15485863
    prefix_length: int = 1
    z_threshold: float = 2.61

    # byte-level bias position (0 = first visible byte)
    byte_pos: int = 0

    # skip invalid bytes (empty/utf8-continuation-start)
    ignore_invalid_byte: bool = True

    # prompt encoding behavior
    add_special_tokens: bool = True

    # generation defaults (can be overridden by TransformersConfig / caller kwargs)
    gen: Optional[Dict[str, Any]] = None

    @staticmethod
    def from_json(path: str) -> "ByteKGWConfig":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        cfg = ByteKGWConfig()
        for k, v in d.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg
