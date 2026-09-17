#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multiprocess Unbiased generator: splits prompts across workers/devices.

Example:
TOKENIZERS_PARALLELISM=false /home/star/jf/python/stega/bin/python scripts/generate_clean_unbiased_mp.py \
  --run_meta outputs/c4_samples_head_200/run_metadata.json \
  --unbiased_config config/Unbiased.json \
  --output_dir outputs/unbiased_gen_mp \
  --n_prompts 200 \
  --devices cuda:0,cuda:1 \
  --num_workers 2 \
  --skip_clean
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from MarkLLM.watermark.unbiased.unbiased import UnbiasedConfig, UnbiasedUtils, UnbiasedLogitsProcessor  # type: ignore
from MarkLLM.utils.transformers_config import TransformersConfig  # type: ignore


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_meta", required=True)
    ap.add_argument("--unbiased_config", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--n_prompts", type=int, default=None)
    ap.add_argument("--devices", default="cuda:0")
    ap.add_argument("--num_workers", type=int, default=1)
    ap.add_argument("--skip_clean", action="store_true")
    ap.add_argument("--progress", action="store_true", help="show per-worker prompt progress")
    return ap.parse_args()


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def prepare_prompts(run_meta: Dict[str, Any], n_prompts: int | None) -> List[Dict[str, Any]]:
    prompts = run_meta["prompts"]
    if n_prompts is not None:
        prompts = prompts[:n_prompts]
    return prompts


def prompt_seed(seed_base: int, prompt_id: int) -> int:
    return int(seed_base) + int(prompt_id)


def split_prompt_cont(tokenizer, full_ids: torch.Tensor, prompt_text: str, add_special_tokens: bool) -> Tuple[str, str]:
    prompt_enc = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=add_special_tokens)
    prompt_len = prompt_enc["input_ids"][0].numel()
    prompt_dec = tokenizer.decode(full_ids[:prompt_len], skip_special_tokens=True)
    full_dec = tokenizer.decode(full_ids, skip_special_tokens=True)
    cont = full_dec[len(prompt_dec) :]
    return full_dec, cont


def worker(task: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    devs = task["devices"]
    dev = devs[task["worker_idx"] % len(devs)]
    prompts = task["prompts"]
    add_special_tokens = task["add_special_tokens"]
    gen_params = task["gen_params"]
    model_path = task["model_path"]
    ub_cfg_path = task["ub_cfg_path"]
    seed_base = task["seed_base"]
    skip_clean = task["skip_clean"]
    show_prog = task.get("progress", False)

    tok = AutoTokenizer.from_pretrained(model_path)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if dev.startswith("cuda") else None,
    ).to(dev)
    mdl.eval()

    tcfg = TransformersConfig(model=mdl, tokenizer=tok, device=dev)
    ub_cfg_obj = UnbiasedConfig(ub_cfg_path, tcfg)
    ub_utils = UnbiasedUtils(ub_cfg_obj)
    ub_proc = UnbiasedLogitsProcessor(ub_cfg_obj, ub_utils)

    rows_clean: List[Dict[str, Any]] = []
    rows_ub: List[Dict[str, Any]] = []

    iterable = tqdm(prompts, desc=f"worker{task['worker_idx']}", leave=False) if show_prog else prompts
    for prompt_id, prompt_row in iterable:
        if isinstance(prompt_row, dict):
            prompt_text = prompt_row.get("prompt") or prompt_row.get("prompt_text") or prompt_row.get("text")
            c4_path = prompt_row.get("c4_source_path", "")
            c4_line = prompt_row.get("c4_source_line", -1)
            c4_head = prompt_row.get("c4_source_head", "")
        else:
            prompt_text = str(prompt_row)
            c4_path = ""
            c4_line = -1
            c4_head = ""
        if prompt_text is None:
            raise KeyError(f"prompt missing for prompt_id={prompt_id}")

        seed = prompt_seed(seed_base, prompt_id)
        torch.manual_seed(seed)
        if dev.startswith("cuda"):
            torch.cuda.manual_seed_all(seed)

        enc = tok(prompt_text, return_tensors="pt", add_special_tokens=add_special_tokens).to(dev)

        if not skip_clean:
            out_clean = mdl.generate(**enc, **gen_params)[0]
            full_clean, cont_clean = split_prompt_cont(tok, out_clean, prompt_text, add_special_tokens)
            rows_clean.append(
                {
                    "algo": "clean",
                    "device": dev,
                    "model_path": model_path,
                    "unbiased_config": ub_cfg_path,
                    "seed": seed,
                    "prompt_id": prompt_id,
                    "c4_source_path": c4_path,
                    "c4_source_line": c4_line,
                    "prompt_text": prompt_text,
                    "c4_source_head": c4_head,
                    "full_text": full_clean,
                    "continuation_text": cont_clean,
                    "gen_params_json": json.dumps(gen_params),
                }
            )

        out_ub = mdl.generate(**enc, logits_processor=LogitsProcessorList([ub_proc]), **gen_params)[0]
        full_ub, cont_ub = split_prompt_cont(tok, out_ub, prompt_text, add_special_tokens)
        rows_ub.append(
            {
                "algo": "unbiased",
                "device": dev,
                "model_path": model_path,
                "unbiased_config": ub_cfg_path,
                "seed": seed,
                "prompt_id": prompt_id,
                "c4_source_path": c4_path,
                "c4_source_line": c4_line,
                "prompt_text": prompt_text,
                "c4_source_head": c4_head,
                "full_text": full_ub,
                "continuation_text": cont_ub,
                "gen_params_json": json.dumps(gen_params),
            }
        )
    return rows_clean, rows_ub


def main() -> None:
    args = parse_args()
    run_meta = load_json(args.run_meta)
    ub_cfg_path = args.unbiased_config
    devices = [d.strip() for d in args.devices.split(",") if d.strip()]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = run_meta["model"]
    add_special_tokens = bool(run_meta.get("add_special_tokens", True))
    gen_params_base = run_meta.get("gen", {}) or {}

    prompts = prepare_prompts(run_meta, args.n_prompts)
    seed_base = int(run_meta.get("seed_base", 1234))

    n_workers = max(1, args.num_workers)
    chunks: List[List[Tuple[int, Dict[str, Any]]]] = [[] for _ in range(n_workers)]
    for i, p in enumerate(prompts):
        chunks[i % n_workers].append((i, p))

    rows_clean_all: List[Dict[str, Any]] = []
    rows_ub_all: List[Dict[str, Any]] = []

    if n_workers == 1:
        rc, ru = worker(
            {
                "devices": devices,
                "worker_idx": 0,
                "prompts": chunks[0],
                "add_special_tokens": add_special_tokens,
                "gen_params": gen_params_base,
                "model_path": model_path,
                "unbiased_config": ub_cfg_path,
                "seed_base": seed_base,
                "skip_clean": args.skip_clean,
                "ub_cfg_path": ub_cfg_path,
            }
        )
        rows_clean_all.extend(rc)
        rows_ub_all.extend(ru)
    else:
        tasks = []
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            for widx in range(n_workers):
                if not chunks[widx]:
                    continue
                tasks.append(
                    ex.submit(
                        worker,
                        {
                            "devices": devices,
                            "worker_idx": widx,
                            "prompts": chunks[widx],
                            "add_special_tokens": add_special_tokens,
                            "gen_params": gen_params_base,
                            "model_path": model_path,
                            "unbiased_config": ub_cfg_path,
                            "ub_cfg_path": ub_cfg_path,
                            "seed_base": seed_base,
                            "skip_clean": args.skip_clean,
                            "progress": args.progress,
                        },
                    )
                )
            for fut in tqdm(as_completed(tasks), total=len(tasks), desc="workers"):
                rc, ru = fut.result()
                rows_clean_all.extend(rc)
                rows_ub_all.extend(ru)

    if not args.skip_clean:
        pd.DataFrame(rows_clean_all).to_csv(out_dir / "hf_generate.csv", index=False)
    pd.DataFrame(rows_ub_all).to_csv(out_dir / "unbiased.csv", index=False)
    print(
        f"wrote clean={0 if args.skip_clean else len(rows_clean_all)} and {len(rows_ub_all)} unbiased rows to {out_dir}"
    )


if __name__ == "__main__":
    main()
