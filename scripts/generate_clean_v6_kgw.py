"""
Generate clean, ByteKGWv6, and KGW texts on the same prompts/seeds and
write CSVs in hf_generate-like format.

Supports multiple devices by round-robin assignment (loads one model per device).

NEW in this patched version:
  - Pass PRF params from v6_config into RobustPartitioner:
      k_weight_mode, decision_margin_bits
  - Optional shard support for true multi-GPU parallel runs:
      --indices_file : only generate prompts with these (0-based) indices
      --out_tag      : suffix appended to output CSV filenames to avoid overwrite

Example (single process, round-robin; NOT truly parallel):
  TOKENIZERS_PARALLELISM=false /home/star/jf/python/stega/bin/python scripts/generate_clean_v6_kgw.py \
    --run_meta outputs/c4_samples_head_200/run_metadata.json \
    --v6_config config/ByteKGWv6.json \
    --kgw_config config/KGW.json \
    --output_dir outputs/v6_vs_kgw_gen_prfnew \
    --deltas 1,2,3,4,5 \
    --n_prompts 200 \
    --devices cuda:0,cuda:1

Example (true 2-GPU parallel via two processes):
  # shard0 even indices on GPU0
  CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false /home/star/jf/python/stega/bin/python scripts/generate_clean_v6_kgw.py \
    --run_meta outputs/c4_samples_head_200/run_metadata.json \
    --v6_config config/ByteKGWv6.json \
    --kgw_config config/KGW.json \
    --output_dir outputs/v6_vs_kgw_gen_prfnew_shard0 \
    --deltas 1,2,3,4,5 \
    --n_prompts 200 \
    --devices cuda:0 \
    --indices_file /tmp/prompt_idx_even.txt

  # shard1 odd indices on GPU1
  CUDA_VISIBLE_DEVICES=1 TOKENIZERS_PARALLELISM=false /home/star/jf/python/stega/bin/python scripts/generate_clean_v6_kgw.py \
    --run_meta outputs/c4_samples_head_200/run_metadata.json \
    --v6_config config/ByteKGWv6.json \
    --kgw_config config/KGW.json \
    --output_dir outputs/v6_vs_kgw_gen_prfnew_shard1 \
    --deltas 1,2,3,4,5 \
    --n_prompts 200 \
    --devices cuda:0 \
    --indices_file /tmp/prompt_idx_odd.txt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from MarkLLM.watermark.bytekgwV6.token_bytes import TokenByteVocabV6  # type: ignore
from MarkLLM.watermark.bytekgwV6.prf import RobustPartitioner  # type: ignore
from MarkLLM.watermark.bytekgwV6.logits_processor import ByteKGWv6LogitsProcessor  # type: ignore
from MarkLLM.watermark.kgw.kgw import KGWLogitsProcessor, KGWUtils  # type: ignore


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_meta", required=True)
    ap.add_argument("--v6_config", required=True)
    ap.add_argument("--kgw_config", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--deltas", default="1,2,3,4,5")
    ap.add_argument("--n_prompts", type=int, default=None, help="number of prompts to generate (default: all)")
    ap.add_argument("--devices", default="cuda:0", help="comma-separated devices, e.g., cuda:0,cuda:1 or cpu")

    # NEW: sharding helpers (for true multi-GPU parallel)
    ap.add_argument(
        "--indices_file",
        default=None,
        help="Optional text file containing 0-based prompt indices to generate (one per line).",
    )
    ap.add_argument(
        "--out_tag",
        default="",
        help="Optional suffix appended to output CSV filenames (e.g., '_shard0'). "
             "If empty, writes default filenames.",
    )
    return ap.parse_args()


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _to_bytes(key) -> bytes:
    if isinstance(key, bytes):
        return key
    if isinstance(key, str):
        s = key.strip()
        if s.startswith("0x"):
            try:
                return int(s, 16).to_bytes(16, "little", signed=False)
            except Exception:
                pass
        return s.encode("utf-8")
    try:
        return int(key).to_bytes(16, "little", signed=False)
    except Exception:
        return b"default-key"


class SimpleKGWConfig:
    def __init__(self, cfg: Dict[str, Any], vocab_size: int, device: str) -> None:
        self.gamma = float(cfg.get("gamma", 0.5))
        self.delta = float(cfg.get("delta", 2.0))
        self.hash_key = int(cfg.get("hash_key", 15485863))
        self.z_threshold = float(cfg.get("z_threshold", 4.0))
        self.prefix_length = int(cfg.get("prefix_length", 4))
        self.f_scheme = cfg.get("f_scheme", "additive")
        self.window_scheme = cfg.get("window_scheme", "left")
        self.vocab_size = int(vocab_size)
        self.device = device
        self.gen_kwargs = {}
        self.generation_model = None
        self.generation_tokenizer = None


def prepare_prompts(run_meta: Dict[str, Any], n_prompts: int | None) -> List[Dict[str, Any]]:
    prompts = run_meta["prompts"]
    if n_prompts is not None:
        prompts = prompts[:n_prompts]
    return prompts


def load_indices(indices_file: Optional[str]) -> Optional[List[int]]:
    if not indices_file:
        return None
    p = Path(indices_file)
    if not p.exists():
        raise FileNotFoundError(f"--indices_file not found: {indices_file}")
    idxs: List[int] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        idxs.append(int(s))
    return idxs


def prompt_seed(seed_base: int, prompt_id: int) -> int:
    return int(seed_base) + int(prompt_id)


def split_prompt_cont(tokenizer, full_ids: torch.Tensor, prompt_text: str, add_special_tokens: bool) -> Tuple[str, str]:
    prompt_enc = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=add_special_tokens)
    prompt_len = prompt_enc["input_ids"][0].numel()
    prompt_dec = tokenizer.decode(full_ids[:prompt_len], skip_special_tokens=True)
    full_dec = tokenizer.decode(full_ids, skip_special_tokens=True)
    cont = full_dec[len(prompt_dec):]
    return full_dec, cont


def main() -> None:
    args = parse_args()
    run_meta = load_json(args.run_meta)
    v6_cfg = load_json(args.v6_config)
    kgw_cfg = load_json(args.kgw_config)
    deltas = [float(x) for x in args.deltas.split(",") if x.strip()]
    devices = [d.strip() for d in args.devices.split(",") if d.strip()]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = run_meta["model"]
    add_special_tokens = bool(run_meta.get("add_special_tokens", True))
    gen_params_base = run_meta.get("gen", {}) or {}

    prompts_all = prepare_prompts(run_meta, args.n_prompts)

    idxs = load_indices(args.indices_file)
    if idxs is None:
        prompts = list(enumerate(prompts_all))
    else:
        # keep stable order
        idx_set = set(idxs)
        prompts = [(i, prompts_all[i]) for i in range(len(prompts_all)) if i in idx_set]

    seed_base = int(run_meta.get("seed_base", 1234))

    # cache per-device resources
    cache: Dict[str, Any] = {}

    def get_resources(dev: str):
        if dev in cache:
            return cache[dev]
        tok = AutoTokenizer.from_pretrained(model_path)
        mdl = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if dev.startswith("cuda") else None,
        ).to(dev)
        mdl.eval()

        vocab_v6 = TokenByteVocabV6.from_tokenizer(tok, skip_markers=True).to(dev)

        # ✅ IMPORTANT: pass new PRF params so your modified prf.py + json actually take effect
        partitioner = RobustPartitioner(
            master_key=_to_bytes(v6_cfg.get("hash_key", 15485863)),
            m_bits=int(v6_cfg.get("m_bits", 256)),
            target_anchors=int(v6_cfg.get("target_anchors", 96)),
            k_choices=tuple(v6_cfg.get("k_choices", [4, 5, 6])),
            normalize_whitespace=bool(v6_cfg.get("normalize_whitespace", True)),
            k_weight_mode=str(v6_cfg.get("k_weight_mode", "linear")),
            decision_margin_bits=int(v6_cfg.get("decision_margin_bits", 12)),
        )

        kgw_config_obj = SimpleKGWConfig(kgw_cfg, vocab_size=len(tok), device=dev)
        kgw_utils = KGWUtils(kgw_config_obj)

        cache[dev] = (tok, mdl, vocab_v6, partitioner, kgw_config_obj, kgw_utils)
        return cache[dev]

    rows_clean: List[Dict[str, Any]] = []
    rows_v6: Dict[float, List[Dict[str, Any]]] = {delta: [] for delta in deltas}
    rows_kgw: Dict[float, List[Dict[str, Any]]] = {delta: [] for delta in deltas}

    print(f"Starting generation: {len(prompts)} prompts (filtered), deltas={deltas}, devices={devices}")
    for local_i, (global_idx, prompt_obj) in enumerate(tqdm(prompts, desc="Prompts", unit="prompt")):
        dev = devices[local_i % len(devices)]
        tok, mdl, vocab_v6, partitioner, kgw_config_obj, kgw_utils = get_resources(dev)

        prompt_id = int(prompt_obj.get("prompt_id", global_idx))
        prompt_text = prompt_obj["prompt_text"]
        c4_path = prompt_obj.get("source_path", "")
        c4_line = prompt_obj.get("source_line", "")
        c4_head = prompt_obj.get("source_head", "")
        seed = prompt_seed(seed_base, prompt_id)

        # common gen kwargs
        gen_kwargs = dict(gen_params_base)
        gen_kwargs.setdefault("max_new_tokens", 128)

        # tokenize once (same enc reused for all variants)
        enc = tok(prompt_text, return_tensors="pt", add_special_tokens=add_special_tokens).to(dev)

        # Clean (hf.generate)
        torch.manual_seed(seed)
        if dev.startswith("cuda"):
            torch.cuda.manual_seed_all(seed)
        out_ids = mdl.generate(**enc, **gen_kwargs)[0]
        full_text, cont_text = split_prompt_cont(tok, out_ids, prompt_text, add_special_tokens)
        rows_clean.append(
            {
                "algo": "hf.generate",
                "delta": 0,
                "device": dev,
                "model_path": model_path,
                "v6_config": args.v6_config,
                "kgw_config": args.kgw_config,
                "seed": seed,
                "prompt_index": global_idx,
                "prompt_id": prompt_id,
                "c4_source_path": c4_path,
                "c4_source_line": c4_line,
                "prompt_text": prompt_text,
                "c4_source_head": c4_head,
                "full_text": full_text,
                "continuation_text": cont_text,
                "gen_params_json": json.dumps(gen_kwargs),
            }
        )

        # ByteKGWv6 for each delta
        for delta in deltas:
            proc_v6 = ByteKGWv6LogitsProcessor(
                tokenizer=tok,
                vocab=vocab_v6,
                partitioner=partitioner,
                delta=float(delta),
                n_bytes=int(v6_cfg.get("n_bytes", 3)),
                seed_window_chars=int(v6_cfg.get("seed_window_chars", 18)),
                device=dev,
            )
            torch.manual_seed(seed)
            if dev.startswith("cuda"):
                torch.cuda.manual_seed_all(seed)
            out_ids = mdl.generate(
                **enc,
                logits_processor=LogitsProcessorList([proc_v6]),
                **gen_kwargs,
            )[0]
            full_text, cont_text = split_prompt_cont(tok, out_ids, prompt_text, add_special_tokens)
            rows_v6[delta].append(
                {
                    "algo": "bytekgwV6",
                    "delta": delta,
                    "device": dev,
                    "model_path": model_path,
                    "v6_config": args.v6_config,
                    "kgw_config": args.kgw_config,
                    "seed": seed,
                    "prompt_index": global_idx,
                    "prompt_id": prompt_id,
                    "c4_source_path": c4_path,
                    "c4_source_line": c4_line,
                    "prompt_text": prompt_text,
                    "c4_source_head": c4_head,
                    "full_text": full_text,
                    "continuation_text": cont_text,
                    "gen_params_json": json.dumps(gen_kwargs),
                }
            )

        # KGW for each delta
        for delta in deltas:
            kgw_config_obj.delta = float(delta)
            kgw_proc = KGWLogitsProcessor(kgw_config_obj, kgw_utils)
            torch.manual_seed(seed)
            if dev.startswith("cuda"):
                torch.cuda.manual_seed_all(seed)
            out_ids = mdl.generate(
                **enc,
                logits_processor=LogitsProcessorList([kgw_proc]),
                **gen_kwargs,
            )[0]
            full_text, cont_text = split_prompt_cont(tok, out_ids, prompt_text, add_special_tokens)
            rows_kgw[delta].append(
                {
                    "algo": "kgw",
                    "delta": delta,
                    "device": dev,
                    "model_path": model_path,
                    "v6_config": args.v6_config,
                    "kgw_config": args.kgw_config,
                    "seed": seed,
                    "prompt_index": global_idx,
                    "prompt_id": prompt_id,
                    "c4_source_path": c4_path,
                    "c4_source_line": c4_line,
                    "prompt_text": prompt_text,
                    "c4_source_head": c4_head,
                    "full_text": full_text,
                    "continuation_text": cont_text,
                    "gen_params_json": json.dumps(gen_kwargs),
                }
            )

    tag = args.out_tag.strip()
    if tag and not tag.startswith("_"):
        tag = "_" + tag

    # write CSVs
    pd.DataFrame(rows_clean).to_csv(out_dir / f"hf_generate{tag}.csv", index=False)
    for delta, rows in rows_v6.items():
        pd.DataFrame(rows).to_csv(out_dir / f"bytekgw_v6_delta{delta}{tag}.csv", index=False)
    for delta, rows in rows_kgw.items():
        pd.DataFrame(rows).to_csv(out_dir / f"kgw_delta{delta}{tag}.csv", index=False)

    print("Wrote outputs to:", out_dir)


if __name__ == "__main__":
    main()
