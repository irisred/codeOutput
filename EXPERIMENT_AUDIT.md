# Chapter 4 Experiment Audit

This document maps the experiments described in `thesis/chapters/ch04_watermark.tex` to the code, data, figures, and smoke checks included in this clean repository.

It is intentionally conservative: if an exact final-table rendering script was not found, the status says so even when the underlying result CSVs are included.

## 1. Global Experimental Setup

Thesis location:

```text
thesis/chapters/ch04_watermark.tex
```

Relevant section:

```text
Section 4.3.1: 实验设置
```

Paper setting:

```text
Dataset: C4
Prompt count: 200
Model: Llama3-8B-Chat / Llama3-8B-Instruct local checkpoint
Baselines: KGW, DiPmark, Unbiased Watermark
Main metrics: TPR at calibrated FPR, attacked TPR, retention, RelPPL buckets
Attack types: token-level edit and character-level edit
```

Included evidence:

```text
outputs/c4_samples_head_200/run_metadata.json
outputs/wm_eval/clean/hf_generate.csv
dataset/c4/realnewslike/c4-train.00000-of-00512.json.gz
```

Included sample counts:

```text
outputs/wm_eval/clean/hf_generate.csv: 200
outputs/wm_eval/bytekgw_v6/bytekgw_v6_delta*.csv: 5 files x 200
outputs/wm_eval/kgw/kgw_delta*.csv: 5 files x 200
outputs/wm_eval/dip/dip_a*.csv: 3 files x 200
outputs/wm_eval/unbiased/unbiased.csv: 200
```

Status:

```text
Covered. The repository includes the data scale and method set used by the Chapter 4 comparison.
```

## 2. Method Implementation

Thesis content:

```text
Section 4.2: prefix-carrier watermarking, adaptive prefix retention, fingerprint-driven PRF, z-score detection
```

Current method source:

```text
MarkLLM/watermark/bytekgwV6/watermark.py
MarkLLM/watermark/bytekgwV6/logits_processor.py
MarkLLM/watermark/bytekgwV6/detector.py
MarkLLM/watermark/bytekgwV6/prf.py
MarkLLM/watermark/bytekgwV6/token_bytes.py
MarkLLM/watermark/bytekgwV6/config.py
config/ByteKGWv6.json
```

Smoke status:

```text
Passed import and py_compile checks.
Generation smoke test also passed for ByteKGWv6.
```

See:

```text
SMOKE_TEST_RESULTS.md
```

## 3. Baseline Methods

Thesis content:

```text
Section 4.3.1: KGW, DiPmark, Unbiased Watermark
```

Included source/config:

```text
MarkLLM/watermark/kgw/
MarkLLM/watermark/dip/
MarkLLM/watermark/unbiased/
config/KGW.json
config/DIP.json
config/Unbiased.json
```

Included result data:

```text
outputs/wm_eval/kgw/
outputs/wm_eval/dip/
outputs/wm_eval/unbiased/
outputs/dip_gen_mp/
outputs/unbiased_gen_mp/
```

Smoke status:

```text
KGW generation and detection smoke tested.
DiP and Unbiased import paths smoke tested.
The included comparison CSVs contain completed result rows for KGW, DiP, and Unbiased.
```

Important boundary:

```text
Additional MarkLLM methods are included as source/config where lightweight, but completed Chapter 4 comparison-result rows are included for KGW, DiP, and Unbiased only.
```

## 4. Prefix Entropy-Retention Diagnostic

Thesis asset:

```text
thesis/figures/prefix_entropy_retention.png
```

Purpose:

```text
Show that short visible-byte prefixes retain most of the token distribution entropy after a few bytes.
```

Included code/data:

```text
scripts/entropy_prefix_sweep.py
outputs/entropy_prefix_sweep/entropy_per_step.csv
outputs/entropy_prefix_sweep/entropy_summary.csv
plotting/pp.py
```

Smoke status:

```text
Executed successfully with samples=1 and max_new_tokens=4.
Observed entropy ratio rose from about 0.58 at 1 byte to about 0.98 at 3 bytes.
```

Status:

```text
Experiment logic covered and smoke-tested.
Final thesis PNG is included.
The regenerated plot is not guaranteed to be byte-for-byte identical to the thesis PNG.
```

## 5. Clean Detection / TPR At Calibrated FPR

Thesis content:

```text
Section 4.3.2 uses calibrated FPR thresholds and reports TPR_clean.
```

Included code/data:

```text
scripts/analyze_tpr_by_bucket.py
scripts/analyze_watermark_tprs_unified.py
outputs/wm_eval/tpr_by_bucket.csv
outputs/wm_eval/*/ppl_*.csv
```

Quick aggregate check over PPL buckets:

```text
Clean TPR, FPR=0.01:
kgw        0.882
bytekgw_v6 0.848
dip        0.745
unbiased   0.736
```

Status:

```text
Covered. The clean TPR result is available and consistent with a mixed clean-detection picture: KGW can be slightly higher than ByteKGWv6 before attacks at strict FPR.
```

## 6. Token-Level Attack Table

Thesis asset:

```text
thesis/figures/tab_token_attack.png
```

Purpose:

```text
Compare TPR_clean and TPR_attack under token-level perturbation at fixed edit budget.
```

Included code/data:

```text
scripts/analyze_attack_tpr_by_bucket.py
scripts/run_attack_grid.py
outputs/wm_eval/tpr_attack_token_by_bucket.csv
outputs/wm_eval/*/ppl_*.csv
```

Quick aggregate check over PPL buckets:

```text
Token attack, FPR=0.01:
bytekgw_v6 retention = 0.957
kgw        retention = 0.902
unbiased   retention = 0.777
dip        retention = 0.760
```

Status:

```text
Underlying experiment data and analysis code are covered.
The final rendered table PNG is included.
Exact script that renders this CSV into the final table PNG was not found.
```

## 7. Character-Level Attack Table

Thesis asset:

```text
thesis/figures/tab_char_attack.png
```

Purpose:

```text
Compare TPR_clean and TPR_attack under character-level perturbation at fixed edit budget.
```

Included code/data:

```text
scripts/analyze_attack_tpr_by_bucket.py
scripts/run_attack_grid.py
outputs/wm_eval/tpr_attack_by_bucket.csv
outputs/wm_eval/*/ppl_*.csv
```

Quick aggregate check over PPL buckets:

```text
Character attack, FPR=0.01:
bytekgw_v6 retention = 0.953
kgw        retention = 0.907
dip        retention = 0.794
unbiased   retention = 0.719
```

Status:

```text
Underlying experiment data and analysis code are covered.
The final rendered table PNG is included.
Exact script that renders this CSV into the final table PNG was not found.
```

## 8. Retention By RelPPL Figure

Thesis asset:

```text
thesis/figures/tpr_retention_relppl.png
```

Purpose:

```text
Show attack retention grouped by RelPPL at fixed edit budget.
```

Included code/data:

```text
plotting/pic.py
plotting/tpr_attack_by_bucket_0.01.csv
plotting/tpr_attack_by_bucket_0.02.csv
plotting/tpr_attack_by_bucket_0.05.csv
plotting/tpr_attack_by_bucket_0.10.csv
outputs/wm_eval/tpr_attack_by_bucket.csv
outputs/wm_eval/tpr_attack_token_by_bucket.csv
```

Smoke status:

```text
Plotting script executed successfully and regenerated:
plotting/retention_vs_relppl.png
plotting/retention_vs_relppl.pdf
```

Status:

```text
Covered and smoke-tested.
Regenerated plot dimensions differ from the final thesis PNG, so this is style/effect reproduction rather than byte-for-byte reproduction.
```

## 9. Edit-Rate Sensitivity Figure

Thesis asset:

```text
thesis/figures/robustness_vs_edit_rate.png
```

Purpose:

```text
Sweep edit rate from 0 to 10% and compare retention degradation for token/character attacks.
```

Included code/data:

```text
scripts/run_attack_grid.py
scripts/analyze_attack_tpr_by_bucket.py
plotting/p.py
plotting/tpr_attack_by_bucket_0.01.csv
plotting/tpr_attack_by_bucket_0.02.csv
plotting/tpr_attack_by_bucket_0.05.csv
plotting/tpr_attack_by_bucket_0.10.csv
```

Smoke status:

```text
Plotting script executed successfully and regenerated:
plotting/robustness_vs_editrate_retention.png
plotting/robustness_vs_editrate_retention.pdf
```

Status:

```text
Covered and smoke-tested.
The full edit-rate grid data used by plotting are included as CSVs.
The plot is not guaranteed to be byte-for-byte identical to the thesis PNG.
```

## 10. Robustness Analysis Table

Thesis asset:

```text
thesis/figures/tab_robustness_stats.png
```

Purpose:

```text
Explain robustness degradation using:
1. green decision flip rate;
2. prefix UID / carrier-category match rate;
3. extra token count caused by tokenization drift.
```

Included code/data:

```text
scripts/trace_v6_attack_grid.py
scripts/trace_v6_attack_grid_mixchar.py
scripts/analyze_v6_components.py
outputs/trace_v6_attack_grid_bytekgw.csv
outputs/trace_v6_attack_grid_bytekgw_mean.csv
outputs/trace_v6_attack_grid_bytekgw_mixchar.csv
outputs/trace_v6_attack_grid_bytekgw_mixchar_mean.csv
outputs/trace_v6_attack_grid_bytekgw_mixchar_mean_by_style.csv
sweep_v6_prf_r002/v6_prf_sweep_results.tsv
traces_v6_process_r002/
```

Smoke status:

```text
scripts/trace_v6_attack_grid.py was executed on 3 rows and 2 attack ratios.
Observed:
attack_ratio=0.0  -> tail_match=1.0, green_flip=0.0, extra_tokens=0.0
attack_ratio=0.02 -> tail_match=0.70, green_flip=0.0267, extra_tokens=5.0
```

Status:

```text
Analysis logic and diagnostic CSVs are covered and smoke-tested.
The final rendered table PNG is included.
Exact script that renders this analysis CSV into the final table PNG was not found.
```

## 11. RelPPL / PPL Bucket Construction

Thesis content:

```text
Lines 196-203 define RelPPL and bucketed reporting.
```

Included code/data:

```text
scripts/add_ppl_inplace.py
scripts/compute_conditional_ppl_csv.py
scripts/summary_conditional_ppl.py
scripts/analyze_tpr_by_bucket.py
outputs/wm_eval/*/ppl_0_3.csv
outputs/wm_eval/*/ppl_3_5.csv
outputs/wm_eval/*/ppl_5_8.csv
outputs/wm_eval/*/ppl_8_12.csv
outputs/wm_eval/*/ppl_12_20.csv
outputs/wm_eval/*/ppl_ge_20.csv
```

Status:

```text
Covered. Bucketed result files are included for ByteKGWv6, KGW, DiP, and Unbiased.
```

Important note:

```text
The robust summary tables use existing bucketed CSVs. Full PPL recomputation requires loading the Llama3 model and is slower; this was not rerun fully in the smoke test.
```

## 12. Conceptual Figures

Thesis assets:

```text
thesis/figures/watermark_framework.png
thesis/figures/prf_partition.png
thesis/figures/tokenization_perturbation_attack.png
```

Status:

```text
Final PNG assets are included.
These are conceptual diagrams rather than experiment plots.
Editable diagram sources were not found in the located repository.
```

## Overall Conclusion

The clean repository covers the main Chapter 4 experimental claims:

```text
1. 200-prompt C4/Llama3 setup.
2. ByteKGWv6 current method implementation.
3. KGW, DiP, and Unbiased baseline result data.
4. Clean TPR and calibrated FPR evaluation.
5. Token-level and character-level attack evaluation.
6. RelPPL/PPL bucket grouping.
7. Edit-rate sensitivity plotting.
8. Prefix entropy-retention diagnostic.
9. Robustness mechanism analysis via green flip / UID match / extra token statistics.
```

The main limitations are:

```text
1. Some final thesis PNGs are rendered assets without exact source scripts.
2. The exact table-rendering scripts for tab_token_attack.png, tab_char_attack.png, and tab_robustness_stats.png were not found.
3. Full paper-scale regeneration was not rerun; only small smoke tests were executed.
4. Bit-for-bit reproduction is not guaranteed because the original command lines, code commit hash, and library versions are not embedded in all CSVs.
```

The strongest accurate statement is:

```text
The package can reproduce the main experiment logic and qualitative effects, and includes the result CSVs behind the main comparisons. It is suitable for future verification and scaled reruns, but not every final thesis image is byte-for-byte reproducible from a located plotting script.
```
