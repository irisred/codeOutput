#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate v6 & kgw samples with 2 GPUs into a new output directory.

- Splits prompts into two shards and runs one process per GPU.
- Ensures new PRF params are passed to RobustPartitioner:
    k_weight_mode, decision_margin_bits

Expected CSV outputs under output_dir:
  - bytekgw_v6_delta{d}.csv
  - kgw_delta{d}.csv
(or whatever your original generator uses; adapt filenames if needed.)

If your repo already has a generator utility, you can plug in there.
This script is "minimal glue": it loads configs, builds watermark processors,
and calls the existing generation function in your repo.

You MUST edit the two import lines below to match your repo layout if needed.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


def load_json(p: str) -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def split_indices(n: int, shard: int, num_shards: int) -> List[int]:
    return [i for i in range(n) if (i % num_shards) == shard]


def write_indices(path: Path, idxs: List[int]) -> None:
    path.write_text("\n".join(map(str, idxs)) + "\n", encoding="utf-8")


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
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--gpus", default="0,1")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # We re-use your existing single-GPU generator to avoid re-implementing generation.
    # The only requirement: it must accept an --indices_file (list of prompt indices)
    # or something equivalent. If your current generator doesn't, see note below.
    single_gpu_script = Path("scripts/generate_clean_v6_kgw.py")
    if not single_gpu_script.exists():
        raise FileNotFoundError(
            f"Expected {single_gpu_script} to exist. "
            "If your generator script has a different name, update single_gpu_script."
        )

    gpus = [x.strip() for x in args.gpus.split(",") if x.strip()]
    if len(gpus) != 2:
        raise ValueError("--gpus must contain exactly 2 GPU ids like '0,1'.")

    deltas = [x.strip() for x in args.deltas.split(",") if x.strip()]
    run_meta = load_json(args.run_meta)

    # Prepare shard index files
    idx0 = split_indices(args.n_prompts, shard=0, num_shards=2)
    idx1 = split_indices(args.n_prompts, shard=1, num_shards=2)
    shard0 = out / "prompt_indices_shard0.txt"
    shard1 = out / "prompt_indices_shard1.txt"
    write_indices(shard0, idx0)
    write_indices(shard1, idx1)

    # Launch two processes
    env0 = os.environ.copy()
    env0["CUDA_VISIBLE_DEVICES"] = gpus[0]
    env0["TOKENIZERS_PARALLELISM"] = "false"

    env1 = os.environ.copy()
    env1["CUDA_VISIBLE_DEVICES"] = gpus[1]
    env1["TOKENIZERS_PARALLELISM"] = "false"

    # NOTE: This assumes your existing generator supports --indices_file.
    # If it doesn't, easiest patch is to add this option to that script:
    # load only prompts with given indices, or just skip others.
    cmd_common = [
        args.python, str(single_gpu_script),
        "--run_meta", args.run_meta,
        "--v6_config", args.v6_config,
        "--kgw_config", args.kgw_config,
        "--output_dir", str(out),
        "--deltas", ",".join(deltas),
        "--max_new_tokens", str(args.max_new_tokens),
        "--temperature", str(args.temperature),
        "--top_p", str(args.top_p),
        "--seed", str(args.seed),
        "--append",  # assume your generator can append/merge; if not, remove and ensure unique files per shard
    ]

    cmd0 = cmd_common + ["--indices_file", str(shard0), "--shard_tag", "gpu0"]
    cmd1 = cmd_common + ["--indices_file", str(shard1), "--shard_tag", "gpu1"]

    print("Launching GPU0:", " ".join(cmd0))
    p0 = subprocess.Popen(cmd0, env=env0)

    print("Launching GPU1:", " ".join(cmd1))
    p1 = subprocess.Popen(cmd1, env=env1)

    r0 = p0.wait()
    r1 = p1.wait()
    if r0 != 0 or r1 != 0:
        raise SystemExit(f"Generation failed: gpu0={r0}, gpu1={r1}")

    print("\nDone. Outputs in:", out)
    print("Now run analyze_v6_kgw_attack.py on this new output_dir.")


if __name__ == "__main__":
    main()
