# Reproducibility Guide

All commands assume:

```bash
cd /home/star/jf/watermark-repro-clean
export PYTHONPATH=.
```

Use the original local environment if available:

```bash
source /home/star/jf/python/stega/bin/activate
```

## 1. Import Smoke Test

```bash
python scripts/smoke_imports.py
```

Expected: it prints `OK` lines for ByteKGWv6, KGW, DiP, Unbiased, and Charm modules.

For a record of small commands that were actually executed on this cleaned repository, see:

```text
SMOKE_TEST_RESULTS.md
```

## 2. Regenerate ByteKGWv6 / KGW Samples

This is the closest command to the included `outputs/v6_vs_kgw_gen_prfnew_mp/` data.

```bash
TOKENIZERS_PARALLELISM=false python scripts/generate_clean_v6_kgw.py \
  --run_meta outputs/c4_samples_head_200/run_metadata.json \
  --v6_config config/ByteKGWv6.json \
  --kgw_config config/KGW.json \
  --output_dir outputs/repro_v6_vs_kgw \
  --deltas 1,2,3,4,5 \
  --n_prompts 200 \
  --devices cuda:0
```

Notes:

- The model path is read from `outputs/c4_samples_head_200/run_metadata.json`.
- The original metadata uses `../Meta-Llama-3-8B-Instruct`.
- If your model is elsewhere, edit `model` in the metadata file or create a copied metadata file.
- Exact generated text may vary across hardware/library versions.

For a fast smoke run:

```bash
TOKENIZERS_PARALLELISM=false python scripts/generate_clean_v6_kgw.py \
  --run_meta outputs/c4_samples_head_200/run_metadata.json \
  --v6_config config/ByteKGWv6.json \
  --kgw_config config/KGW.json \
  --output_dir outputs/smoke_v6_vs_kgw \
  --deltas 1 \
  --n_prompts 2 \
  --devices cuda:0
```

## 3. Evaluate Clean TPR/FPR/PPL

Use the included generated data:

```bash
TOKENIZERS_PARALLELISM=false python scripts/analyze_v6_kgw_tpr.py \
  --run_meta outputs/c4_samples_head_200/run_metadata.json \
  --v6_config config/ByteKGWv6.json \
  --kgw_config config/KGW.json \
  --input_dir outputs/v6_vs_kgw_gen_prfnew_mp \
  --deltas 1,2,3,4,5 \
  --fprs 0.01,0.05,0.1,0.2 \
  --device cuda:0
```

For a faster structural check without PPL:

```bash
TOKENIZERS_PARALLELISM=false python scripts/analyze_v6_kgw_tpr.py \
  --run_meta outputs/c4_samples_head_200/run_metadata.json \
  --v6_config config/ByteKGWv6.json \
  --kgw_config config/KGW.json \
  --input_dir outputs/v6_vs_kgw_gen_prfnew_mp \
  --deltas 3 \
  --fprs 0.01,0.05 \
  --n_clean 20 \
  --n_samples 20 \
  --skip_ppl \
  --device cuda:0
```

## 4. Evaluate Character-Attack Robustness

```bash
TOKENIZERS_PARALLELISM=false python scripts/analyze_v6_kgw_attack.py \
  --run_meta outputs/c4_samples_head_200/run_metadata.json \
  --v6_config config/ByteKGWv6.json \
  --kgw_config config/KGW.json \
  --input_dir outputs/v6_vs_kgw_gen_prfnew_mp \
  --deltas 1,2,3,4,5 \
  --fprs 0.01,0.05,0.1,0.2 \
  --attack_ratio 0.02 \
  --n_clean 200 \
  --n_samples 200 \
  --device cuda:0 \
  --skip_ppl
```

## 5. Mechanism Diagnostics

### Prefix/PRF Trace Grid

```bash
TOKENIZERS_PARALLELISM=false python scripts/trace_v6_attack_grid.py \
  --run_meta outputs/c4_samples_head_200/run_metadata.json \
  --v6_config config/ByteKGWv6.json \
  --input_csv outputs/v6_vs_kgw_gen_prfnew_mp/bytekgw_v6_delta2.0.csv \
  --attack_ratios 0,0.01,0.02,0.05,0.10 \
  --attack_seed 0 \
  --indices all \
  --max_tokens 200 \
  --stride 1 \
  --device cuda:0 \
  --out_csv outputs/repro_trace_v6_attack_grid_bytekgw.csv
```

### Mixed Character-Attack Trace

```bash
TOKENIZERS_PARALLELISM=false python scripts/trace_v6_attack_grid_mixchar.py \
  --run_meta outputs/c4_samples_head_200/run_metadata.json \
  --v6_config config/ByteKGWv6.json \
  --input_csv outputs/v6_vs_kgw_gen_prfnew_mp/bytekgw_v6_delta2.0.csv \
  --attack_ratios 0,0.01,0.02,0.05,0.10 \
  --attack_seed 0 \
  --indices all \
  --max_tokens 200 \
  --stride 1 \
  --device cuda:0 \
  --out_csv outputs/repro_trace_v6_attack_grid_bytekgw_mixchar.csv
```

## 6. Entropy / Prefix-Retention Diagnostic

```bash
TOKENIZERS_PARALLELISM=false python scripts/entropy_prefix_sweep.py \
  --run_meta outputs/c4_samples_head_200/run_metadata.json \
  --prompt_csv outputs/c4_samples_head_200/hf_generate.csv \
  --prompt_row 0 \
  --device cuda:0 \
  --output_dir outputs/repro_entropy_prefix_sweep
```

## 7. Plotting

The located plotting scripts are under `plotting/`.

```bash
cd plotting
python p.py
python pic.py
python pp.py
```

Outputs:

```text
robustness_vs_editrate_retention.png/pdf
retention_vs_relppl.png/pdf
entropy_retention_modern.png/pdf
```

Important: thesis final figure names differ slightly:

```text
thesis/figures/robustness_vs_edit_rate.png
thesis/figures/tpr_retention_relppl.png
thesis/figures/prefix_entropy_retention.png
```

The plotting scripts are the closest located reproducible sources for these styles and data, but not guaranteed to reproduce the thesis PNGs byte-for-byte.

For a per-figure reproduction status table, see:

```text
FIGURE_REPRO_STATUS.md
```

## 8. Thesis Material

Watermark chapter:

```text
thesis/chapters/ch04_watermark.tex
```

Final figure assets:

```text
thesis/figures/
```

## Known Caveats

- Model weights are not included.
- Some scripts depend on exact local model paths inside `run_metadata.json`.
- The two-GPU helper `scripts/generate_v6_vs_kgw_prfnew_2gpus.py` was preserved, but the single-process `scripts/generate_clean_v6_kgw.py` is the safer command path to start with.
- Some final thesis table PNGs and framework diagrams are included as final assets; their exact editable/plot source was not found.
- Bit-for-bit generation reproducibility may vary across PyTorch/Transformers/CUDA versions.
