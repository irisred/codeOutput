# Clean Repository Manifest

Created from local sources under `/home/star/jf` on 2026-06-02.

## Included From `/home/star/jf/watermark`

Core code:

```text
MarkLLM/
config/
scripts/
selected top-level experiment/evaluation Python files
```

Selected data:

```text
dataset/c4/realnewslike/c4-train.00000-of-00512.json.gz
dataset/c4_zh/
data/
outputs/v6_vs_kgw_gen_prfnew_mp/
outputs/wm_eval/
outputs/entropy_prefix_sweep/
outputs/dip_gen_mp/
outputs/unbiased_gen_mp/
outputs/c4_samples_head_200/
outputs/c4_samples_bytekgw_all_200/
outputs/trace_v6_attack_grid_bytekgw*.csv
sweep_v6_prf_r002/
traces_v6_process_r002/
```

## Included From `/home/star/jf/hduthesis-extracted/hduthesis`

```text
thesis/chapters/ch04_watermark.tex
thesis/figures/*.png
```

Only Chapter 4 watermark-related figures were copied.

## Included From `/home/star/jf/stega_expand`

```text
plotting/p.py
plotting/pic.py
plotting/pp.py
plotting/tpr_attack_by_bucket_*.csv
plotting/*retention*.png/pdf
plotting/entropy_retention_modern.png/pdf
```

These are the closest located plotting scripts for the retention and entropy figures.

## Excluded

Large or unrelated material was excluded:

```text
.git/
__pycache__/
*.pyc
metrics/
textattack/
his/
large log files
c.zip
cache_person.pt
MarkLLM/dataset/
MarkLLM/watermark/xsir/dictionary/
MarkLLM/watermark/xsir/mapping/
MarkLLM/watermark/sir/mapping/
outputs/wm_eval_attacked/
```

The excluded mapping/dictionary folders are not needed for the ByteKGWv6/KGW/DiP/Unbiased path documented here.

