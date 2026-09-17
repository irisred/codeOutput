# 干净复现仓库清单

本复现包于 2026-06-02 从 `/home/star/jf` 下的本地材料整理生成。

## 来自 `/home/star/jf/watermark` 的内容

核心代码：

```text
MarkLLM/
config/
scripts/
若干顶层实验/评估 Python 文件
```

选择保留的数据：

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

## 来自 `/home/star/jf/hduthesis-extracted/hduthesis` 的内容

```text
thesis/chapters/ch04_watermark.tex
thesis/figures/*.png
```

只复制了第 4 章水印相关的图。

## 来自 `/home/star/jf/stega_expand` 的内容

```text
plotting/p.py
plotting/pic.py
plotting/pp.py
plotting/tpr_attack_by_bucket_*.csv
plotting/*retention*.png/pdf
plotting/entropy_retention_modern.png/pdf
```

这些是目前定位到的、最接近 retention 和 entropy 相关图的画图脚本。

## 排除内容

大型或无关材料被排除：

```text
.git/
__pycache__/
*.pyc
metrics/
textattack/
his/
大型日志文件
c.zip
cache_person.pt
MarkLLM/dataset/
MarkLLM/watermark/xsir/dictionary/
MarkLLM/watermark/xsir/mapping/
MarkLLM/watermark/sir/mapping/
outputs/wm_eval_attacked/
```

被排除的 mapping/dictionary 目录不是当前文档所覆盖的 `ByteKGWv6/KGW/DiP/Unbiased` 复现路径所必需的内容。
