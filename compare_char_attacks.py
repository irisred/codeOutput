"""Run char-level random attacks against multiple watermarks and summarize metrics.

This script reuses the existing `test_rand_attack` pipeline to keep all attack
logic identical to the rest of the repo. For each watermark we:
  1. Load reference detector settings from the specified attack config JSON.
  2. Invoke `test_rand_attack` with the provided parameters (char attack only).
  3. Read the generated records from `saved_attk_data` and compute summary stats.

Example:

python compare_char_attacks.py \
    --config_path attk_config/opt_rand_config.json \
    --llm_name facebook/opt-1.3b \
    --wm_names CharmKGW KGW \
    --max_edit_rate 0.1 \
    --max_token_num 100 \
    --atk_times 10 \
    --data_aug 9 \
    --device 0
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Any

import numpy as np

from test_random_attack import test_rand_attack


def _load_attack_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _format_saved_file_path(
    *,
    wm_name: str,
    llm_name: str,
    ref_model: str,
    max_edit_rate: float,
    max_token_num: int,
    atk_style: str,
    atk_times: int,
    ori_flag: bool,
    def_stl: str,
    attk_name: str,
) -> Path:
    ref_model_tag = ref_model.replace("saved_model/", "") if ref_model else "no_ref_model"
    wm_tag = wm_name.replace("/", "_")
    llm_tag = llm_name.replace("/", "_")
    def_tag = def_stl if def_stl else "no_def"
    filename = "_".join(
        [
            attk_name,
            wm_tag,
            llm_tag,
            str(max_edit_rate),
            str(max_token_num),
            atk_style,
            str(atk_times),
            str(ori_flag),
            def_tag,
            ref_model_tag,
        ]
    ) + ".json"
    return Path("saved_attk_data") / filename


def _summarize_records(records: List[Dict[str, Any]]) -> Dict[str, float]:
    if not records:
        return {}

    def _mean(values):
        return float(np.mean(values)) if values else 0.0

    asr = _mean([not bool(rec["adv_detect"]["is_watermarked"]) for rec in records])
    wdr = _mean([rec["wm_score_drop"] for rec in records])
    token_budget = _mean(
        [rec["t_edit_dist"] / max(rec["token_num"], 1) for rec in records]
    )
    char_budget = _mean(
        [rec["c_edit_dist"] / max(rec["char_num"], 1) for rec in records]
    )
    bleu = _mean([rec["belu"] for rec in records])
    rouge = _mean([rec["rouge-f1"] for rec in records])
    ppl_rate = _mean([rec["ppl_rate"] for rec in records])
    adv_ppl = _mean([rec["adv_ppl"] for rec in records])

    return {
        "count": len(records),
        "ASR": asr,
        "WDR": wdr,
        "token_budget": token_budget,
        "char_budget": char_budget,
        "BLEU": bleu,
        "ROUGE": rouge,
        "ppl_rate": ppl_rate,
        "adv_ppl": adv_ppl,
    }


def run_attack_for_watermark(
    wm_name: str,
    config: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, float]:
    if wm_name not in config:
        raise ValueError(f"Watermark {wm_name} not found in config {args.config_path}")

    wm_cfg = config[wm_name]
    ref_tokenizer = wm_cfg["ref_tokenizer"]
    ref_model = wm_cfg["ref_model"][str(args.data_aug)]

    os.makedirs("saved_attk_data", exist_ok=True)

    test_rand_attack(
        llm_name=args.llm_name,
        wm_name=wm_name,
        max_edit_rate=args.max_edit_rate,
        max_token_num=args.max_token_num,
        atk_style="char",
        ref_tokenizer=ref_tokenizer,
        ref_model=ref_model,
        atk_times=args.atk_times,
        ori_flag=args.ori_flag,
        def_stl=args.def_stl,
        device=args.device,
        char_op=args.char_op,
    )

    data_path = _format_saved_file_path(
        wm_name=wm_name,
        llm_name=args.llm_name,
        ref_model=ref_model or "",
        max_edit_rate=args.max_edit_rate,
        max_token_num=args.max_token_num,
        atk_style="char",
        atk_times=args.atk_times,
        ori_flag=args.ori_flag,
        def_stl=args.def_stl,
        attk_name="Rand" if args.char_op == 2 else f"RandChar_{args.char_op}",
    )
    if not data_path.exists():
        raise FileNotFoundError(f"Expected attack records at {data_path}")

    with open(data_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    return _summarize_records(records)


def main():
    parser = argparse.ArgumentParser(description="Compare char-level attacks across watermarks.")
    parser.add_argument("--config_path", type=str, default="attk_config/opt_rand_config.json")
    parser.add_argument("--llm_name", type=str, default="facebook/opt-1.3b")
    parser.add_argument("--wm_names", nargs="+", default=["CharmKGW", "KGW"])
    parser.add_argument("--max_edit_rate", type=float, default=0.1)
    parser.add_argument("--max_token_num", type=int, default=100)
    parser.add_argument("--atk_times", type=int, default=10)
    parser.add_argument("--data_aug", type=int, default=9)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--ori_flag", action="store_true")
    parser.add_argument("--def_stl", type=str, default="")
    parser.add_argument("--char_op", type=int, default=2)
    args = parser.parse_args()

    config = _load_attack_config(args.config_path)

    summaries = {}
    for wm in args.wm_names:
        print(f"\n=== Running char attack for watermark: {wm} ===")
        summaries[wm] = run_attack_for_watermark(wm, config, args)

    print("\n=== Summary Metrics ===")
    header = ["Watermark", "Samples", "ASR", "WDR", "TokBudget", "CharBudget", "BLEU", "ROUGE", "PPL_rate", "Adv_PPL"]
    print("\t".join(header))
    for wm, stats in summaries.items():
        if not stats:
            print(f"{wm}\tNo data")
            continue
        row = [
            wm,
            str(stats["count"]),
            f"{stats['ASR']:.3f}",
            f"{stats['WDR']:.3f}",
            f"{stats['token_budget']:.3f}",
            f"{stats['char_budget']:.3f}",
            f"{stats['BLEU']:.3f}",
            f"{stats['ROUGE']:.3f}",
            f"{stats['ppl_rate']:.3f}",
            f"{stats['adv_ppl']:.3f}",
        ]
        print("\t".join(row))


if __name__ == "__main__":
    main()
