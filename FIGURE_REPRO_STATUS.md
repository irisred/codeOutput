# Figure Reproduction Status

This file records which Chapter 4 watermark figures have reproducible experiment/plotting material in this clean package.

## Status Summary

| Thesis asset | Role | Included source/data status | Smoke-tested? | Notes |
|---|---|---:|---:|---|
| `thesis/figures/prefix_entropy_retention.png` | Prefix entropy-retention diagnostic | `scripts/entropy_prefix_sweep.py`; `outputs/entropy_prefix_sweep/`; `plotting/pp.py` | Yes | Qualitative trend reproduced: prefix entropy ratio rises quickly and reaches about 0.98 by 3 bytes in the smoke run. |
| `thesis/figures/tpr_retention_relppl.png` | TPR retention vs RelPPL buckets | `plotting/pic.py`; `plotting/tpr_attack_by_bucket_*.csv` | Yes | Plot regenerates as `plotting/retention_vs_relppl.png/pdf`. Dimensions differ from the final thesis PNG, so this is not a byte-for-byte recreation. |
| `thesis/figures/robustness_vs_edit_rate.png` | TPR retention vs edit rate | `plotting/p.py`; `plotting/tpr_attack_by_bucket_*.csv` | Yes | Plot regenerates as `plotting/robustness_vs_editrate_retention.png/pdf`. Dimensions differ from the final thesis PNG, so this is not a byte-for-byte recreation. |
| `thesis/figures/tab_token_attack.png` | Token-attack table figure | Relevant result data under `outputs/wm_eval/` and related CSVs | Partially | Final table PNG is included. Exact table-rendering script was not found in the located code. |
| `thesis/figures/tab_char_attack.png` | Character-attack table figure | Relevant result data under `outputs/wm_eval/` and related CSVs | Partially | Final table PNG is included. Exact table-rendering script was not found in the located code. |
| `thesis/figures/tab_robustness_stats.png` | Robustness process-statistics table | `scripts/trace_v6_attack_grid.py`; `outputs/trace_v6_attack_grid_bytekgw*.csv` | Yes for CSV diagnostic | Trace/statistics code was smoke-tested. Exact PNG table-rendering script was not found. |
| `thesis/figures/watermark_framework.png` | Method framework diagram | Final PNG only | No | Conceptual diagram, not an experiment plot. Editable source was not found. |
| `thesis/figures/prf_partition.png` | PRF partition diagram | Final PNG only | No | Conceptual diagram, not an experiment plot. Editable source was not found. |
| `thesis/figures/tokenization_perturbation_attack.png` | Tokenization perturbation diagram | Final PNG only | No | Conceptual diagram, not an experiment plot. Editable source was not found. |

## What Was Actually Reproduced

The following plotting commands were executed successfully:

```bash
cd plotting
MPLBACKEND=Agg /home/star/jf/python/stega/bin/python p.py
MPLBACKEND=Agg /home/star/jf/python/stega/bin/python pic.py
MPLBACKEND=Agg /home/star/jf/python/stega/bin/python pp.py
```

They regenerated:

```text
plotting/robustness_vs_editrate_retention.png/pdf
plotting/retention_vs_relppl.png/pdf
plotting/entropy_retention_modern.png/pdf
```

The regenerated outputs are consistent with the experiment direction and visual style, but are not guaranteed to match the thesis PNGs byte-for-byte.

## Effect Check

Small-sample checks confirm the expected qualitative effects:

- Clean text z-scores stay near zero.
- Watermarked text at `delta=3` separates strongly from clean text.
- Attack diagnostics show no drift at edit rate 0 and increasing token/prefix disruption under character edits.
- Prefix entropy-retention rises quickly as prefix length increases, supporting the thesis motivation for short visible-byte prefix classes.

## Comparison-Result Check

The included comparison summaries cover these methods:

```text
bytekgw_v6
kgw
dip
unbiased
```

They are stored in:

```text
outputs/wm_eval/tpr_by_bucket.csv
outputs/wm_eval/tpr_attack_by_bucket.csv
outputs/wm_eval/tpr_attack_token_by_bucket.csv
```

When averaged over the included PPL buckets, the qualitative comparison is consistent with the thesis claim that the proposed method is more stable under edit attacks. For example:

```text
Character attack, FPR=0.01:
bytekgw_v6 retention = 0.953
kgw        retention = 0.907
dip        retention = 0.794
unbiased   retention = 0.719

Token attack, FPR=0.01:
bytekgw_v6 retention = 0.957
kgw        retention = 0.902
unbiased   retention = 0.777
dip        retention = 0.760
```

The clean, non-attacked TPR comparison is more mixed. KGW can be slightly higher than `bytekgw_v6` at low FPR before attacks, while `bytekgw_v6` becomes strongest or near-strongest after edit attacks. Therefore, the most accurate interpretation is:

```text
The reproduced/included comparison data support the robustness-retention claim, not a blanket claim that ByteKGWv6 has the highest clean TPR in every setting.
```

For exact smoke commands and observed values, see `SMOKE_TEST_RESULTS.md`.
