#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate clean and DiP-marked texts on the same prompts/seeds, write CSVs.

Example:
  TOKENIZERS_PARALLELISM=false /home/star/jf/python/stega/bin/python scripts/generate_clean_dip.py \
    --run_meta outputs/c4_samples_head_200/run_metadata.json \
    --dip_config config/DIP.json \
    --output_dir outputs/dip_gen \
    --n_prompts 200 \
    --devices cuda:0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from MarkLLM.watermark.dip.dip import DIPConfig, DIPUtils, DIPLogitsProcessor  # type: ignore
from MarkLLM.utils.transformers_config import TransformersConfig  # type: ignore


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_meta", required=True)
    ap.add_argument("--dip_config", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--n_prompts", type=int, default=None, help="number of prompts to generate (default: all)")
    ap.add_argument("--devices", default="cuda:0", help="comma-separated devices, e.g., cuda:0,cuda:1 or cpu")
    ap.add_argument("--skip_clean", action="store_true", help="do not write hf_generate.csv (use existing clean set)")
    ap.add_argument("--alphas", default=None, help="comma-separated override(s) for alpha; generate one CSV per alpha")
    ap.add_argument("--gammas", default=None, help="comma-separated override(s) for gamma; paired with alphas (cartesian)")
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


def main() -> None:
    args = parse_args()
    run_meta = load_json(args.run_meta)
    dip_cfg_path = args.dip_config
    devices = [d.strip() for d in args.devices.split(",") if d.strip()]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = run_meta["model"]
    add_special_tokens = bool(run_meta.get("add_special_tokens", True))
    gen_params_base = run_meta.get("gen", {}) or {}

    prompts = prepare_prompts(run_meta, args.n_prompts)
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

        tcfg = TransformersConfig(model=mdl, tokenizer=tok, device=dev)
        dip_cfg_obj = DIPConfig(dip_cfg_path, tcfg)
        dip_utils = DIPUtils(dip_cfg_obj)
        dip_proc = DIPLogitsProcessor(dip_cfg_obj, dip_utils)

        cache[dev] = (tok, mdl, dip_cfg_obj, dip_utils, dip_proc)
        return cache[dev]

    rows_clean: List[Dict[str, Any]] = []
    outputs: Dict[str, List[Dict[str, Any]]] = {}

    # prepare strength sweeps
    alpha_list = [float(x) for x in args.alphas.split(",")] if args.alphas else [None]
    gamma_list = [float(x) for x in args.gammas.split(",")] if args.gammas else [None]

    for idx, prompt_row in tqdm(list(enumerate(prompts)), desc="prompts"):
        prompt_id = prompt_row.get("id", idx)
        prompt_text = prompt_row["prompt"]
        c4_path = prompt_row.get("c4_source_path", "")
        c4_line = prompt_row.get("c4_source_line", -1)
        c4_head = prompt_row.get("c4_source_head", "")

        dev = devices[idx % len(devices)]
        tok, mdl, dip_cfg_obj, dip_utils, dip_proc = get_resources(dev)

        seed = prompt_seed(seed_base, prompt_id)
        torch.manual_seed(seed)
        if dev.startswith("cuda"):
            torch.cuda.manual_seed_all(seed)

        gen_kwargs = dict(gen_params_base)

        enc = tok(prompt_text, return_tensors="pt", add_special_tokens=add_special_tokens).to(dev)

        # clean (optional)
        if not args.skip_clean:
            out_clean = mdl.generate(**enc, **gen_kwargs)[0]
            full_clean, cont_clean = split_prompt_cont(tok, out_clean, prompt_text, add_special_tokens)
            rows_clean.append(
                {
                    "algo": "clean",
                    "device": dev,
                    "model_path": model_path,
                    "dip_config": dip_cfg_path,
                    "seed": seed,
                    "prompt_id": prompt_id,
                    "c4_source_path": c4_path,
                    "c4_source_line": c4_line,
                    "prompt_text": prompt_text,
                    "c4_source_head": c4_head,
                    "full_text": full_clean,
                    "continuation_text": cont_clean,
                    "gen_params_json": json.dumps(gen_kwargs),
                }
            )

        # DiP sweeps
        for alpha in alpha_list:
            for gamma in gamma_list:
                # override strengths on the shared config object (thread-safe here, single-process)
                if alpha is not None:
                    dip_cfg_obj.alpha = float(alpha)
                if gamma is not None:
                    dip_cfg_obj.gamma = float(gamma)
                out_dip = mdl.generate(**enc, logits_processor=LogitsProcessorList([dip_proc]), **gen_kwargs)[0]
                full_dip, cont_dip = split_prompt_cont(tok, out_dip, prompt_text, add_special_tokens)
                rows_key = f"a{alpha if alpha is not None else dip_cfg_obj.alpha}_g{gamma if gamma is not None else dip_cfg_obj.gamma}"
                rows = outputs.setdefault(rows_key, [])
                rows.append(
                    {
                        "algo": f"dip",
                        "alpha": float(alpha if alpha is not None else dip_cfg_obj.alpha),
                        "gamma": float(gamma if gamma is not None else dip_cfg_obj.gamma),
                        "device": dev,
                        "model_path": model_path,
                        "dip_config": dip_cfg_path,
                        "seed": seed,
                        "prompt_id": prompt_id,
                        "c4_source_path": c4_path,
                        "c4_source_line": c4_line,
                        "prompt_text": prompt_text,
                        "c4_source_head": c4_head,
                        "full_text": full_dip,
                        "continuation_text": cont_dip,
                        "gen_params_json": json.dumps(gen_kwargs),
                    }
                )

    # write CSVs
    if not args.skip_clean:
        pd.DataFrame(rows_clean).to_csv(out_dir / "hf_generate.csv", index=False)
    # dip variants
    for key, rows in outputs.items():
        pd.DataFrame(rows).to_csv(out_dir / f"dip_{key}.csv", index=False)
    print(
        f"wrote clean={0 if args.skip_clean else len(rows_clean)} and {sum(len(v) for v in outputs.values())} dip rows "
        f"to {out_dir}"
    )


if __name__ == "__main__":
    main()
