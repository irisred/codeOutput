from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import torch

try:
    from transformers.generation.logits_process import LogitsProcessorList
except Exception:
    LogitsProcessorList = None  # type: ignore

from .token_bytes import TokenByteVocab
from .FastVisibleByteIndex import FastVisibleByteIndex
from .cudaBytePRF import CudaBytePRF
from .ByteFactorizedLogitsWarper import ByteFactorizedLogitsWarper
from .detector import CudaByteWatermarkDetector, ByteWMDetectorConfig


def _canonical_cuda_device(device: Union[str, torch.device]) -> torch.device:
    dev = torch.device(device)
    if dev.type != "cuda":
        raise ValueError("ByteKGW is CUDA-only.")
    if dev.index is None:
        dev = torch.device("cuda", torch.cuda.current_device())
    return dev


def _append_listlike(obj, item):
    """Append item to LogitsProcessorList-like or list-like object safely."""
    try:
        obj.append(item)
        return obj
    except Exception:
        if LogitsProcessorList is None:
            return list(obj) + [item]
        return LogitsProcessorList(list(obj) + [item])


@dataclass
class ByteKGWConfig:
    algorithm_name: str = "ByteKGW"
    gamma: float = 0.5
    delta: float = 1.0
    hash_key: int = 15485863
    prefix_length: int = 1
    z_threshold: float = 4.0

    max_byte_pos: int = 16
    max_scan_iters: int = 64
    eps: float = 1e-12

    ignore_invalid_byte: bool = True
    return_per_byte: bool = False

    device: str = "cuda:0"
    config_dict: Dict[str, Any] = None  # type: ignore

    @classmethod
    def from_json(cls, path: str) -> "ByteKGWConfig":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        cfg = cls(
            algorithm_name=str(d.get("algorithm_name", "ByteKGW")),
            gamma=float(d.get("gamma", 0.5)),
            delta=float(d.get("delta", 1.0)),
            hash_key=int(d.get("hash_key", 15485863)),
            prefix_length=int(d.get("prefix_length", 1)),
            z_threshold=float(d.get("z_threshold", 4.0)),
            max_byte_pos=int(d.get("max_byte_pos", d.get("max_depth", 16))),
            max_scan_iters=int(d.get("max_scan_iters", 64)),
            eps=float(d.get("eps", 1e-12)),
            ignore_invalid_byte=bool(d.get("ignore_invalid_byte", True)),
            return_per_byte=bool(d.get("return_per_byte", False)),
            device=str(d.get("device", "cuda:0")),
        )
        cfg.config_dict = d
        return cfg


@dataclass
class ByteKGWRuntime:
    token_byte_vocab: TokenByteVocab
    byte_index: FastVisibleByteIndex
    prf: CudaBytePRF
    warper: ByteFactorizedLogitsWarper
    detector: CudaByteWatermarkDetector


class ByteKGW:
    """
    ByteKGW (factorized byte sampling) that runs AFTER HF built-in warpers.

    In transformers versions where _get_logits_warper does not exist,
    temperature/top_p/top_k/typical are included inside the logits_processor chain.
    We patch _get_logits_processor and append our factorized module at the end.
    """

    def __init__(self, config_path: str, transformers_config: Any) -> None:
        self.config = ByteKGWConfig.from_json(config_path)
        self.tf_cfg = transformers_config

        self.model = getattr(transformers_config, "model")
        self.tokenizer = getattr(transformers_config, "tokenizer")

        dev = getattr(transformers_config, "device", None)
        if dev is None:
            try:
                dev = self.model.device
            except Exception:
                dev = self.config.device
        self.device = _canonical_cuda_device(dev)

        # move model
        if getattr(self.model, "device", None) is None or self.model.device != self.device:
            self.model = self.model.to(self.device)
        self.model.eval()

        # pad token to suppress warning
        if getattr(self.tokenizer, "pad_token_id", None) is None and getattr(self.tokenizer, "eos_token_id", None) is not None:
            try:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            except Exception:
                pass
        try:
            if getattr(self.model, "config", None) is not None and getattr(self.tokenizer, "pad_token_id", None) is not None:
                self.model.config.pad_token_id = int(self.tokenizer.pad_token_id)
        except Exception:
            pass

        # build byte vocab + index
        token_byte_vocab = TokenByteVocab.from_tokenizer(self.tokenizer)
        byte_index = FastVisibleByteIndex(
            token_byte_vocab,
            device=self.device,
            max_depth=int(self.config.max_byte_pos),
            max_scan_iters=int(self.config.max_scan_iters),
        )

        green_size = int(round(256.0 * float(self.config.gamma)))
        green_size = max(0, min(256, green_size))
        prf_cfg = {
            "hash_key": int(self.config.hash_key),
            "prefix_length": int(self.config.prefix_length),
            "green_size": green_size,
        }
        prf = CudaBytePRF(prf_cfg, device=self.device)

        warper = ByteFactorizedLogitsWarper(
            prf=prf,
            byte_index=byte_index,
            delta=float(self.config.delta),
            prefix_length=int(self.config.prefix_length),
            max_byte_pos=int(self.config.max_byte_pos),
            eps=float(self.config.eps),
        )

        det_cfg = ByteWMDetectorConfig(
            prefix_length=int(self.config.prefix_length),
            gamma=float(self.config.gamma),
            z_threshold=float(self.config.z_threshold),
            max_byte_pos=int(self.config.max_byte_pos),
            ignore_invalid_byte=bool(self.config.ignore_invalid_byte),
            return_per_byte=bool(self.config.return_per_byte),
        )
        detector = CudaByteWatermarkDetector(
            prf=prf,
            byte_index=byte_index,
            config=det_cfg,
            device=self.device,
        )

        self.runtime = ByteKGWRuntime(
            token_byte_vocab=token_byte_vocab,
            byte_index=byte_index,
            prf=prf,
            warper=warper,
            detector=detector,
        )

    def _encode_prompt(self, prompt: str) -> Dict[str, torch.Tensor]:
        enc = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        for k, v in list(enc.items()):
            if torch.is_tensor(v):
                enc[k] = v.to(self.device)
        return enc

    def _generation_kwargs(self, *, max_new_tokens: Optional[int] = None) -> Dict[str, Any]:
        kw: Dict[str, Any] = {}
        for name in [
            "max_new_tokens",
            "min_new_tokens",
            "do_sample",
            "temperature",
            "top_p",
            "top_k",
            "typical_p",
            "repetition_penalty",
            "num_beams",
            "num_return_sequences",
        ]:
            val = getattr(self.tf_cfg, name, None)
            if val is not None:
                kw[name] = val
        if max_new_tokens is not None:
            kw["max_new_tokens"] = int(max_new_tokens)

        if getattr(self.tokenizer, "pad_token_id", None) is not None:
            kw["pad_token_id"] = int(self.tokenizer.pad_token_id)
        if getattr(self.tokenizer, "eos_token_id", None) is not None:
            kw["eos_token_id"] = int(self.tokenizer.eos_token_id)
        return kw

    @torch.no_grad()
    def generate_unwatermarked_text(self, prompt: str, *, max_new_tokens: Optional[int] = None) -> str:
        enc = self._encode_prompt(prompt)
        gen_kw = self._generation_kwargs(max_new_tokens=max_new_tokens)
        out = self.model.generate(**enc, **gen_kw)
        return self.tokenizer.decode(out[0], skip_special_tokens=True)

    @torch.no_grad()
    def generate_watermarked_text(self, prompt: str, *, max_new_tokens: Optional[int] = None) -> str:
        enc = self._encode_prompt(prompt)
        gen_kw = self._generation_kwargs(max_new_tokens=max_new_tokens)

        # factorized sampling needs do_sample=True
        if "do_sample" in gen_kw and not bool(gen_kw["do_sample"]):
            raise ValueError("ByteKGW factorized sampling requires do_sample=True.")

        # Patch strategy:
        # - If model has _get_logits_warper: append to warper chain.
        # - Else (your case): append to _get_logits_processor chain (which already includes top_p/top_k/temperature/...).
        orig_get_logits_warper = getattr(self.model, "_get_logits_warper", None)
        orig_get_logits_processor = getattr(self.model, "_get_logits_processor", None)

        if orig_get_logits_warper is not None:
            def patched_get_logits_warper(*args, **kwargs):
                warpers = orig_get_logits_warper(*args, **kwargs)
                return _append_listlike(warpers, self.runtime.warper)

            self.model._get_logits_warper = patched_get_logits_warper
            try:
                out = self.model.generate(**enc, **gen_kw)
            finally:
                self.model._get_logits_warper = orig_get_logits_warper
            return self.tokenizer.decode(out[0], skip_special_tokens=True)

        if orig_get_logits_processor is None:
            raise RuntimeError(
                "transformers model has neither _get_logits_warper nor _get_logits_processor; "
                "cannot append custom factorized module in a generate-aligned way."
            )

        def patched_get_logits_processor(*args, **kwargs):
            procs = orig_get_logits_processor(*args, **kwargs)
            # IMPORTANT: append last, so it runs after HF temp/top_p/top_k/typical
            return _append_listlike(procs, self.runtime.warper)

        self.model._get_logits_processor = patched_get_logits_processor
        try:
            out = self.model.generate(**enc, **gen_kw)
        finally:
            self.model._get_logits_processor = orig_get_logits_processor

        return self.tokenizer.decode(out[0], skip_special_tokens=True)

    @torch.no_grad()
    def detect_watermark(self, text: str) -> Dict[str, Any]:
        enc = self.tokenizer(text, return_tensors="pt", add_special_tokens=True)
        ids = enc["input_ids"].to(self.device)
        out = self.runtime.detector.detect(ids)
        res: Dict[str, Any] = {}
        for k, v in out.items():
            res[k] = v.detach() if torch.is_tensor(v) else v
        return res


__all__ = ["ByteKGW", "ByteKGWConfig"]
