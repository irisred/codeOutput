# Smoke Test Results

This file records the small-sample checks run on 2026-06-02 to verify that the clean repository is executable and that the included code/data can reproduce the main experimental directions at a lightweight scale.

All commands were run from:

```bash
cd /home/star/jf/watermark-repro-clean
export PYTHONPATH=.
```

The local Python used was:

```bash
/home/star/jf/python/stega/bin/python
```

## 1. Import And Syntax Checks

Command:

```bash
/home/star/jf/python/stega/bin/python scripts/smoke_imports.py
```

Result: passed. The script successfully imported:

- `MarkLLM.watermark.bytekgwV6.*`
- `MarkLLM.watermark.kgw.kgw`
- `MarkLLM.watermark.dip.dip`
- `MarkLLM.watermark.unbiased.unbiased`
- `MarkLLM.charm_v2.*`

The key generation, detection, trace, entropy, and plotting scripts were also compiled with `py_compile`.

## 2. ByteKGWv6 / KGW Generation Smoke Test

Command:

```bash
TOKENIZERS_PARALLELISM=false /home/star/jf/python/stega/bin/python \
  scripts/generate_clean_v6_kgw.py \
  --run_meta outputs/c4_samples_head_200/run_metadata.json \
  --v6_config config/ByteKGWv6.json \
  --kgw_config config/KGW.json \
  --output_dir outputs/smoke_v6_vs_kgw \
  --deltas 1 \
  --n_prompts 2 \
  --devices cuda:0
```

Result: passed. The command loaded the local Llama3 model and wrote:

```text
outputs/smoke_v6_vs_kgw/hf_generate.csv
outputs/smoke_v6_vs_kgw/bytekgw_v6_delta1.0.csv
outputs/smoke_v6_vs_kgw/kgw_delta1.0.csv
```

This verifies that the current method and KGW comparison generation entrypoint are runnable.

## 3. TPR/FPR Detection Smoke Test

Command:

```bash
TOKENIZERS_PARALLELISM=false /home/star/jf/python/stega/bin/python \
  scripts/analyze_v6_kgw_tpr.py \
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

Observed result:

```text
ByteKGWv6 clean z mean = -0.050, std = 0.682
ByteKGWv6 delta=3.0 z mean = 5.371, std = 1.108
ByteKGWv6 TPR at FPR=0.01 = 1.000
ByteKGWv6 TPR at FPR=0.05 = 1.000

KGW clean z mean = -0.420, std = 1.043
KGW delta=3.0 z mean = 5.972, std = 1.062
KGW TPR at FPR=0.01 = 1.000
KGW TPR at FPR=0.05 = 1.000
```

This is not a full-table reproduction, but it confirms that the clean data, configs, tokenizer, and detector code produce the expected separation between clean and watermarked text.

## 4. Attack Trace Smoke Test

Command:

```bash
TOKENIZERS_PARALLELISM=false /home/star/jf/python/stega/bin/python \
  scripts/trace_v6_attack_grid.py \
  --run_meta outputs/c4_samples_head_200/run_metadata.json \
  --v6_config config/ByteKGWv6.json \
  --input_csv outputs/v6_vs_kgw_gen_prfnew_mp/bytekgw_v6_delta2.0.csv \
  --attack_ratios 0,0.02 \
  --attack_seed 0 \
  --indices 0,1,2 \
  --max_tokens 50 \
  --stride 1 \
  --device cuda:0 \
  --out_csv outputs/smoke_trace_v6_attack_grid_bytekgw.csv
```

Observed per-ratio mean:

```text
attack_ratio,mean_tail_match,mean_green_flip,mean_extra_tokens,mean_traced
0.0,1.0,0.0,0.0,50.0
0.02,0.7000000000000001,0.02666666666666667,5.0,50.0
```

This verifies that the robustness/trace diagnostic path runs and produces interpretable process statistics.

## 5. Entropy Prefix-Retention Smoke Test

Command:

```bash
TOKENIZERS_PARALLELISM=false /home/star/jf/python/stega/bin/python \
  scripts/entropy_prefix_sweep.py \
  --run_meta outputs/c4_samples_head_200/run_metadata.json \
  --prompt_csv outputs/c4_samples_head_200/hf_generate.csv \
  --prompt_row 0 \
  --device cuda:0 \
  --samples 1 \
  --max_new_tokens 4 \
  --output_dir outputs/smoke_entropy_prefix_sweep
```

Observed summary:

```text
n,mean_entropy_prefix,mean_entropy_full,mean_ratio
1,0.640774,1.105886,0.579422
2,0.862808,1.105886,0.780196
3,1.086462,1.105886,0.982435
4,1.100254,1.105886,0.994907
5,1.100254,1.105886,0.994907
6,1.100254,1.105886,0.994907
7,1.100254,1.105886,0.994907
```

This matches the expected qualitative behavior: a few prefix bytes retain most of the full token-byte entropy.

## 6. Plotting Smoke Test

Command:

```bash
cd plotting
MPLBACKEND=Agg /home/star/jf/python/stega/bin/python p.py
MPLBACKEND=Agg /home/star/jf/python/stega/bin/python pic.py
MPLBACKEND=Agg /home/star/jf/python/stega/bin/python pp.py
```

Result: passed. The plotting scripts regenerated:

```text
plotting/robustness_vs_editrate_retention.pdf
plotting/robustness_vs_editrate_retention.png
plotting/retention_vs_relppl.pdf
plotting/retention_vs_relppl.png
plotting/entropy_retention_modern.pdf
plotting/entropy_retention_modern.png
```

Only pandas future-version warnings were observed; the figure files were produced successfully.

## Interpretation

These smoke tests verify that the clean repository can:

- run the current ByteKGWv6 method and KGW baseline generation;
- evaluate clean-vs-watermarked detection behavior from included data;
- run the robustness trace diagnostic;
- run the entropy-prefix diagnostic;
- regenerate the included plotting outputs.

They do not replace the full paper-scale experiments. For full reproduction, use the larger commands in `REPRODUCIBILITY.md`.
