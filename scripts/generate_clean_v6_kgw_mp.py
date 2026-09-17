#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-command, 2-GPU parallel generation for clean + v6 + kgw.

- No manual indices files.
- Spawns 2 worker processes, each uses one GPU and generates its shard.
- Automatically merges shard CSVs into a single output_dir.

IMPORTANT:
- This script assumes you have the following imports in your repo:
    MarkLLM.watermark.bytekgwV6.token_bytes.TokenByteVocabV6
    MarkLLM.watermark.bytekgwV6.prf.RobustPartitioner
    MarkLLM.watermark.bytekgwV6.logits_processor.ByteKGWv6LogitsProcessor
    MarkLLM.watermark.kgw.kgw.KGWLogitsProcessor, KGWUtils
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import torch
import torch.multiprocessing as mp
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


def load_json(p: str) -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _to_bytes(key) -> bytes:
    if isinstance(key, bytes):
        return key
    if isinstance(key, str):
        return key.encode("utf-8", errors="ignore")
    return int(key).to_bytes(16, "little", signed=False)


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


def prompt_seed(seed_base: int, prompt_id: int) -> int:
    return int(seed_base) + int(prompt_id)


def split_prompt_cont(tokenizer, full_ids: torch.Tensor, prompt_text: str, add_special_tokens: bool) -> Tuple[str, str]:
    prompt_enc = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=add_special_tokens)
    prompt_len = prompt_enc["input_ids"][0].numel()
    prompt_dec = tokenizer.decode(full_ids[:prompt_len], skip_special_tokens=True)
    full_dec = tokenizer.decode(full_ids, skip_special_tokens=True)
    cont = full_dec[len(prompt_dec):]
    return full_dec, cont


def worker(rank: int, world_size: int, args_dict: Dict[str, Any]) -> None:
    # bind this worker to one GPU
    gpu_ids = args_dict["gpu_ids"]
    gpu = gpu_ids[rank]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    run_meta = load_json(args_dict["run_meta"])
    v6_cfg = load_json(args_dict["v6_config"])
    kgw_cfg = load_json(args_dict["kgw_config"])

    deltas = [float(x) for x in args_dict["deltas"].split(",") if x.strip()]
    out_dir = Path(args_dict["output_dir"])
    shard_dir = out_dir / f"_shard{rank}"
    shard_dir.mkdir(parents=True, exist_ok=True)

    model_path = run_meta["model"]
    add_special_tokens = bool(v6_cfg.get("add_special_tokens", True))  # keep aligned with your v6 cfg
    gen_params_base = run_meta.get("gen", {}) or {}
    n_prompts = args_dict["n_prompts"]

    prompts_all = run_meta["prompts"]
    if n_prompts is not None:
        prompts_all = prompts_all[: int(n_prompts)]

    # shard split: even/odd by index
    shard_prompts = [(i, p) for i, p in enumerate(prompts_all) if (i % world_size) == rank]

    tok = AutoTokenizer.from_pretrained(model_path)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device.startswith("cuda") else None,
    ).to(device)
    mdl.eval()

    vocab_v6 = TokenByteVocabV6.from_tokenizer(tok, skip_markers=True).to(device)

    # IMPORTANT: pass new PRF params
    partitioner = RobustPartitioner(
        master_key=_to_bytes(v6_cfg.get("hash_key", 15485863)),
        m_bits=int(v6_cfg.get("m_bits", 256)),
        target_anchors=int(v6_cfg.get("target_anchors", 96)),
        k_choices=tuple(v6_cfg.get("k_choices", [4, 5, 6])),
        normalize_whitespace=bool(v6_cfg.get("normalize_whitespace", True)),
        k_weight_mode=str(v6_cfg.get("k_weight_mode", "linear")),
        decision_margin_bits=int(v6_cfg.get("decision_margin_bits", 12)),
    )

    kgw_config_obj = SimpleKGWConfig(kgw_cfg, vocab_size=len(tok), device=device)
    kgw_utils = KGWUtils(kgw_config_obj)

    seed_base = int(run_meta.get("seed_base", 1234))

    rows_clean: List[Dict[str, Any]] = []
    rows_v6: Dict[float, List[Dict[str, Any]]] = {d: [] for d in deltas}
    rows_kgw: Dict[float, List[Dict[str, Any]]] = {d: [] for d in deltas}

    for global_idx, prompt_obj in tqdm(shard_prompts, desc=f"GPU{gpu}", unit="prompt"):
        prompt_id = int(prompt_obj.get("prompt_id", global_idx))
        prompt_text = prompt_obj["prompt_text"]
        seed = prompt_seed(seed_base, prompt_id)

        gen_kwargs = dict(gen_params_base)
        gen_kwargs.setdefault("max_new_tokens", int(args_dict["max_new_tokens"]))
        gen_kwargs.setdefault("temperature", float(args_dict["temperature"]))
        gen_kwargs.setdefault("top_p", float(args_dict["top_p"]))

        enc = tok(prompt_text, return_tensors="pt", add_special_tokens=add_special_tokens).to(device)

        torch.manual_seed(seed)
        if device.startswith("cuda"):
            torch.cuda.manual_seed_all(seed)
        out_ids = mdl.generate(**enc, **gen_kwargs)[0]
        full_text, cont_text = split_prompt_cont(tok, out_ids, prompt_text, add_special_tokens)
        rows_clean.append(
            dict(
                algo="hf.generate",
                delta=0,
                device=str(gpu),
                seed=seed,
                prompt_index=global_idx,
                prompt_id=prompt_id,
                prompt_text=prompt_text,
                full_text=full_text,
                continuation_text=cont_text,
                gen_params_json=json.dumps(gen_kwargs),
            )
        )

        for delta in deltas:
            proc_v6 = ByteKGWv6LogitsProcessor(
                tokenizer=tok,
                vocab=vocab_v6,
                partitioner=partitioner,
                delta=float(delta),
                n_bytes=int(v6_cfg.get("n_bytes", 3)),
                seed_window_chars=int(v6_cfg.get("seed_window_chars", 18)),
                device=device,
            )
            torch.manual_seed(seed)
            if device.startswith("cuda"):
                torch.cuda.manual_seed_all(seed)
            out_ids = mdl.generate(**enc, logits_processor=LogitsProcessorList([proc_v6]), **gen_kwargs)[0]
            full_text, cont_text = split_prompt_cont(tok, out_ids, prompt_text, add_special_tokens)
            rows_v6[delta].append(
                dict(
                    algo="bytekgwV6",
                    delta=delta,
                    device=str(gpu),
                    seed=seed,
                    prompt_index=global_idx,
                    prompt_id=prompt_id,
                    prompt_text=prompt_text,
                    full_text=full_text,
                    continuation_text=cont_text,
                    gen_params_json=json.dumps(gen_kwargs),
                )
            )

        for delta in deltas:
            kgw_config_obj.delta = float(delta)
            kgw_proc = KGWLogitsProcessor(kgw_config_obj, kgw_utils)
            torch.manual_seed(seed)
            if device.startswith("cuda"):
                torch.cuda.manual_seed_all(seed)
            out_ids = mdl.generate(**enc, logits_processor=LogitsProcessorList([kgw_proc]), **gen_kwargs)[0]
            full_text, cont_text = split_prompt_cont(tok, out_ids, prompt_text, add_special_tokens)
            rows_kgw[delta].append(
                dict(
                    algo="kgw",
                    delta=delta,
                    device=str(gpu),
                    seed=seed,
                    prompt_index=global_idx,
                    prompt_id=prompt_id,
                    prompt_text=prompt_text,
                    full_text=full_text,
                    continuation_text=cont_text,
                    gen_params_json=json.dumps(gen_kwargs),
                )
            )

    # write shard outputs
    pd.DataFrame(rows_clean).to_csv(shard_dir / "hf_generate.csv", index=False)
    for d, rows in rows_v6.items():
        pd.DataFrame(rows).to_csv(shard_dir / f"bytekgw_v6_delta{d}.csv", index=False)
    for d, rows in rows_kgw.items():
        pd.DataFrame(rows).to_csv(shard_dir / f"kgw_delta{d}.csv", index=False)


def merge_shards(output_dir: str, deltas: List[float]) -> None:
    out = Path(output_dir)
    shard0 = out / "_shard0"
    shard1 = out / "_shard1"

    def merge_csv(name: str) -> None:
        df0 = pd.read_csv(shard0 / name)
        df1 = pd.read_csv(shard1 / name)
        df = pd.concat([df0, df1], ignore_index=True)
        df.to_csv(out / name, index=False)
        print("merged", name, "rows=", len(df))

    merge_csv("hf_generate.csv")
    for d in deltas:
        merge_csv(f"bytekgw_v6_delta{d}.csv")
        merge_csv(f"kgw_delta{d}.csv")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_meta", required=True)
    ap.add_argument("--v6_config", required=True)
    ap.add_argument("--kgw_config", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--deltas", default="1,2,3,4,5")
    ap.add_argument("--n_prompts", type=int, default=200)
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)  # reserved
    ap.add_argument("--gpus", default="0,1", help="physical GPU ids, e.g. '0,1'")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gpu_ids = [int(x.strip()) for x in args.gpus.split(",") if x.strip()]
    if len(gpu_ids) != 2:
        raise ValueError("This script expects exactly 2 GPUs, e.g. --gpus 0,1")

    world_size = 2
    args_dict = dict(
        run_meta=args.run_meta,
        v6_config=args.v6_config,
        kgw_config=args.kgw_config,
        output_dir=args.output_dir,
        deltas=args.deltas,
        n_prompts=args.n_prompts,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        gpu_ids=gpu_ids,
    )

    mp.set_start_method("spawn", force=True)
    mp.spawn(worker, args=(world_size, args_dict), nprocs=world_size, join=True)

    deltas = [float(x) for x in args.deltas.split(",") if x.strip()]
    merge_shards(args.output_dir, deltas)
    print("Done. Final merged CSVs are in:", args.output_dir)


if __name__ == "__main__":
    main()
