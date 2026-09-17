# MarkLLM/watermark/bytekgwV2/bytekgw.py
from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from .config import ByteKGWConfig
from .token_bytes import TokenByteVocab
from .cuda_byte_prf import BytePRFConfig, CudaBytePRF
from .generation_model import HFAlignedGenerationModel
from .byte_bias import ByteLogitBiaser
from .detector import ByteWMDetectorConfig, CudaByteWatermarkDetector


class ByteKGW:
    """
    ByteKGW V2:
      - step-by-step generate aligned with HF (processors/warpers)
      - apply byte-level logits bias AFTER HF processing
      - delta==0 => strict no-op (sampling == HF)
    """

    def __init__(self, config_path: str, tf_cfg, device: Optional[str | torch.device] = None):
        self.config = ByteKGWConfig.from_json(config_path)

        self.model = tf_cfg.model
        self.tokenizer = tf_cfg.tokenizer

        self.device = torch.device(device if device is not None else tf_cfg.device)
        if self.device.type != "cuda":
            raise ValueError("ByteKGW V2 is CUDA-only in this setup.")
        self.model.to(self.device)
        self.model.eval()

        # build gen kwargs aligned with tf_cfg (can be overridden by config.gen)
        gen_kwargs: Dict[str, Any] = dict(
            do_sample=bool(getattr(tf_cfg, "do_sample", True)),
            temperature=float(getattr(tf_cfg, "temperature", 1.0)),
            top_p=float(getattr(tf_cfg, "top_p", 1.0)),
            top_k=int(getattr(tf_cfg, "top_k", 0)),
            repetition_penalty=float(getattr(tf_cfg, "repetition_penalty", 1.0)),
            typical_p=float(getattr(tf_cfg, "typical_p", 1.0)) if hasattr(tf_cfg, "typical_p") else 1.0,
            max_new_tokens=int(getattr(tf_cfg, "max_new_tokens", 128)),
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        if self.config.gen:
            gen_kwargs.update(self.config.gen)

        self.gen_kwargs = gen_kwargs

        # TokenByteVocab (visible bytes) on GPU
        self.token_byte_vocab = TokenByteVocab.from_tokenizer(self.tokenizer, device=self.device)

        # PRF
        prf_cfg = BytePRFConfig(gamma=float(self.config.gamma), hash_key=int(self.config.hash_key))
        self.prf = CudaBytePRF(prf_cfg, device=self.device)

        # HF-aligned generation core
        self.aligned = HFAlignedGenerationModel(self.model, self.tokenizer, gen_kwargs=self.gen_kwargs, device=self.device)

        # byte biaser (STRICT no-op when delta==0)
        self.biaser = ByteLogitBiaser(
            prf=self.prf,
            token_byte_vocab=self.token_byte_vocab,
            delta=float(self.config.delta),
            prefix_length=int(self.config.prefix_length),
            byte_pos=int(self.config.byte_pos),
            device=self.device,
            vectorized=True,
        )

        # detector
        det_cfg = ByteWMDetectorConfig(
            prefix_length=int(self.config.prefix_length),
            gamma=float(self.config.gamma),
            z_threshold=float(self.config.z_threshold),
            byte_pos=int(self.config.byte_pos),
            ignore_invalid_byte=bool(self.config.ignore_invalid_byte),
        )
        self.detector = CudaByteWatermarkDetector(self.prf, self.token_byte_vocab, det_cfg, device=self.device)

    def _encode(self, text: str):
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=bool(self.config.add_special_tokens),
        )
        return {k: v.to(self.device) for k, v in enc.items()}

    @torch.no_grad()
    def generate_unwatermarked_text(self, prompt: str, max_new_tokens: Optional[int] = None) -> str:
        enc = self._encode(prompt)
        kw = dict(self.gen_kwargs)
        if max_new_tokens is not None:
            kw["max_new_tokens"] = int(max_new_tokens)

        out = self.model.generate(**enc, **kw)
        return self.tokenizer.decode(out[0], skip_special_tokens=True)

    @torch.no_grad()
    def generate_watermarked_text(self, prompt: str, max_new_tokens: Optional[int] = None) -> str:
        enc = self._encode(prompt)
        input_ids = enc["input_ids"]
        attn = enc.get("attention_mask", torch.ones_like(input_ids, device=self.device))

        max_new = int(max_new_tokens if max_new_tokens is not None else self.gen_kwargs.get("max_new_tokens", 128))
        do_sample = bool(self.gen_kwargs.get("do_sample", True))
        eos_id = self.tokenizer.eos_token_id

        self.aligned.reset()

        for _ in range(max_new):
            scores = self.aligned.step(input_ids, attn)  # [B,V] after HF processors/warpers

            # apply byte bias (delta==0 => strict no-op)
            scores = self.biaser.apply(input_ids, scores)

            if do_sample:
                probs = F.softmax(scores, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1)  # [B,1]
            else:
                next_ids = torch.argmax(scores, dim=-1, keepdim=True)

            input_ids = torch.cat([input_ids, next_ids], dim=1)
            attn = torch.cat([attn, torch.ones_like(next_ids, device=self.device)], dim=1)

            if eos_id is not None:
                if bool((next_ids.view(-1) == int(eos_id)).all().item()):
                    break

        return self.tokenizer.decode(input_ids[0], skip_special_tokens=True)

    @torch.no_grad()
    def detect_watermark(self, text: str) -> Dict[str, Any]:
        enc = self._encode(text)
        ids = enc["input_ids"]  # [1,T]
        out = self.detector.detect(ids)
        # return python-friendly scalars for convenience
        return {
            "z": float(out["z"][0].item()),
            "num_scored": int(out["num_scored"][0].item()),
            "num_green": int(out["num_green"][0].item()),
            "is_watermarked": bool(out["is_watermarked"][0].item()),
        }

    def firstbyte_vocab_tensor(self) -> torch.Tensor:
        # byte_pos=0 table (int16 [V], 0..255 or -1)
        return self.token_byte_vocab.bytepos_tensor(self.device, 0)
