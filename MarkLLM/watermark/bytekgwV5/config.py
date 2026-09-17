# MarkLLM/watermark/bytekgwV5/config.py
from __future__ import annotations

import json
from typing import Any, Dict, Optional, List

from MarkLLM.watermark.base import BaseConfig


def _get(d: Dict[str, Any], k: str, default: Any) -> Any:
    return d.get(k, default)


class ByteKGWv5Config(BaseConfig):
    """
    Config binder for ByteKGWv5.

    Reads algorithm_config JSON and binds:
      - watermark knobs: gamma/delta/hash_key/prefix_length/z_threshold
      - generation knobs: do_sample/temperature/top_p/top_k/repetition_penalty/max_new_tokens...
      - scheme: "token" or "byte_tree"
      - prompt encoding: add_special_tokens
      - detector knobs: detector_mode / max_byte_pos / use_prefix_bytes_in_prf
      - runtime knobs: use_cache / use_torch_generator
      - (NEW) pos_weights / pos_weights_path for all-byte weighted detection
    """

    @property
    def algorithm_name(self) -> str:
        return "ByteKGWv5"

    def _load_pos_weights_from_path(self, path: str) -> Optional[List[float]]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
        except Exception:
            return None

        # support:
        #  - {"weights":[...], ...}
        #  - [...]
        if isinstance(obj, dict) and "weights" in obj:
            arr = obj["weights"]
        else:
            arr = obj

        if not isinstance(arr, list):
            return None
        try:
            return [float(x) for x in arr]
        except Exception:
            return None

    def initialize_parameters(self) -> None:
        d = self.config_dict

        # watermark core
        self.gamma: float = float(_get(d, "gamma", 0.5))
        self.delta: float = float(_get(d, "delta", 1.0))
        self.hash_key: int = int(_get(d, "hash_key", 15485863))
        self.prefix_length: int = int(_get(d, "prefix_length", 4))
        self.z_threshold: float = float(_get(d, "z_threshold", 4.0))

        # scheme
        self.scheme: str = str(_get(d, "scheme", "token"))
        if self.scheme not in ("token", "byte_tree"):
            raise ValueError("ByteKGWv5Config.scheme must be 'token' or 'byte_tree'")

        # prompt encoding behavior
        self.add_special_tokens: bool = bool(_get(d, "add_special_tokens", True))

        # detector / byte-tree details
        self.max_byte_pos: int = int(_get(d, "max_byte_pos", 64))
        self.detector_mode: str = str(_get(d, "detector_mode", "first_byte"))
        if self.detector_mode not in ("first_byte", "all_bytes"):
            raise ValueError("detector_mode must be 'first_byte' or 'all_bytes'")
        self.use_prefix_bytes_in_prf: bool = bool(_get(d, "use_prefix_bytes_in_prf", False))

        # runtime
        self.use_cache: bool = bool(_get(d, "use_cache", True))
        self.use_torch_generator: bool = bool(_get(d, "use_torch_generator", False))

        # generation overrides (optional)
        self._gen_overrides: Dict[str, Any] = dict(_get(d, "gen", {}) or {})

        # -----------------------------
        # NEW: load detector pos-weights
        # -----------------------------
        self.pos_weights: Optional[List[float]] = None

        if isinstance(d.get("pos_weights", None), list):
            try:
                self.pos_weights = [float(x) for x in d["pos_weights"]]
            except Exception:
                self.pos_weights = None

        # file path wins if provided (allows big lists without bloating ByteKGWv5.json)
        pos_weights_path = d.get("pos_weights_path", None)
        if isinstance(pos_weights_path, str) and pos_weights_path.strip():
            loaded = self._load_pos_weights_from_path(pos_weights_path.strip())
            if loaded:
                self.pos_weights = loaded

    def gen_cfg_dict(self) -> Dict[str, Any]:
        """
        Generation knobs dict used by:
          - HFAlignedStepper (to build logits_processor/warper path)
          - samplers (temp/top_p/top_k/do_sample/rep_penalty)
          - engine (max_new_tokens/eos/pad)
        """
        cfg = dict(self.gen_kwargs)  # from TransformersConfig + BaseConfig defaults

        # overlay algorithm-config overrides (if provided)
        for k, v in self._gen_overrides.items():
            cfg[k] = v

        # make sure some common knobs exist (reasonable defaults)
        cfg.setdefault("do_sample", False)
        cfg.setdefault("temperature", 1.0)
        cfg.setdefault("top_p", 1.0)
        cfg.setdefault("top_k", 0)
        cfg.setdefault("repetition_penalty", 1.0)
        cfg.setdefault("max_new_tokens", cfg.get("max_new_tokens", 128))

        # ensure eos/pad IDs (align with BaseConfig._ensure_model_pad_token)
        if getattr(self.generation_tokenizer, "eos_token_id", None) is not None:
            cfg.setdefault("eos_token_id", int(self.generation_tokenizer.eos_token_id))
        if getattr(self.generation_tokenizer, "pad_token_id", None) is not None:
            cfg.setdefault("pad_token_id", int(self.generation_tokenizer.pad_token_id))
        else:
            # fall back to eos if tokenizer has no pad
            if cfg.get("eos_token_id", None) is not None:
                cfg.setdefault("pad_token_id", int(cfg["eos_token_id"]))

        return cfg
