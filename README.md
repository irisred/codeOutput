# Prefix-Carrier Watermark Repro Package

This is a cleaned, runnable subset of the watermark project under `/home/star/jf/watermark`.

It focuses on the watermark method used in Chapter 4 of the thesis:

- prefix-carrier watermarking;
- visible-byte prefix classes;
- fingerprint-driven PRF partitioning;
- robust detection under token-level and character-level edits.

The closest current implementation is `ByteKGWv6`.

## Important Paths

```text
MarkLLM/watermark/bytekgwV6/      # current method implementation
MarkLLM/watermark/kgw/            # KGW baseline
MarkLLM/watermark/dip/            # DiP baseline
MarkLLM/watermark/unbiased/       # Unbiased baseline
config/ByteKGWv6.json             # current method config
scripts/                          # generation/evaluation/attack/diagnostic scripts
outputs/                          # selected generated data and summaries
plotting/                         # located plotting scripts and their CSV inputs
thesis/                           # Chapter 4 tex and final thesis figure assets
WATERMARK_CODE_MAP.md             # detailed source/figure map
REPRODUCIBILITY.md                # runnable commands and workflow
DATA_PROVENANCE.md                # what data is included and how it maps to scripts
EXPERIMENT_AUDIT.md               # one-by-one map from thesis experiments to code/data/status
FIGURE_REPRO_STATUS.md            # per-figure reproduction status
SMOKE_TEST_RESULTS.md             # commands actually run and observed small-sample results
ATTACK_PAPER_ORIGINAL_README.md   # historical README from the character-attack source tree
```

## Environment

Current local environment:

```bash
cd /home/star/watermark-repro-clean
source .venv/bin/activate
export PYTHONPATH=.
```

This `.venv` has been created under the repository directory and smoke-tested with the local CUDA PyTorch/Transformers stack. See `docs_zh/LOCAL_RUN_STATUS_zh.md` for the exact local setup and commands that have already passed.

Local model weights are not copied into git. In the current local setup, `models/` contains symlinks to the model directories under `/home/star/jf/`:

```text
models/Meta-Llama-3-8B-Instruct -> /home/star/jf/Meta-Llama-3-8B-Instruct
models/Qwen2.5-3B -> /home/star/jf/Qwen2.5-3B
```

The main metadata file currently uses `models/Meta-Llama-3-8B-Instruct`. Adjust `outputs/c4_samples_head_200/run_metadata.json` or pass another metadata file if your model path differs.

## What Is Included

Confirmed current-method data:

```text
outputs/v6_vs_kgw_gen_prfnew_mp/bytekgw_v6_delta*.csv
outputs/wm_eval/bytekgw_v6/
```

Baseline data:

```text
outputs/v6_vs_kgw_gen_prfnew_mp/kgw_delta*.csv
outputs/wm_eval/kgw/
outputs/wm_eval/dip/
outputs/wm_eval/unbiased/
```

The included completed comparison-result rows are for KGW, DiP, and Unbiased. Other MarkLLM methods are included mainly as source/config baselines, but their full experiment result tables are not part of this clean package.

`ATTACK_PAPER_ORIGINAL_README.md` is kept only as historical context for the character-perturbation attack code. It is not the primary reproduction guide for this package.

Diagnostics:

```text
outputs/trace_v6_attack_grid_bytekgw*.csv
sweep_v6_prf_r002/
traces_v6_process_r002/
```

Plotting inputs and scripts:

```text
plotting/p.py
plotting/pic.py
plotting/pp.py
plotting/tpr_attack_by_bucket_*.csv
```

## Quick Checks

From this directory:

```bash
cd /home/star/watermark-repro-clean
source .venv/bin/activate
export PYTHONPATH=.

python scripts/smoke_imports.py
python -m py_compile \
  MarkLLM/watermark/bytekgwV6/watermark.py \
  scripts/generate_clean_v6_kgw.py \
  scripts/analyze_v6_kgw_tpr.py \
  scripts/trace_v6_attack_grid.py
```

## First Things To Read

1. `DATA_PROVENANCE.md`
2. `REPRODUCIBILITY.md`
3. `WATERMARK_CODE_MAP.md`
4. `EXPERIMENT_AUDIT.md`
5. `FIGURE_REPRO_STATUS.md`
6. `SMOKE_TEST_RESULTS.md`
