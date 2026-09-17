# 水印代码与图表映射

本文档记录 `/home/star/jf` 下与水印相关的论文材料、方法源码、实验脚本、结果文件和画图脚本的位置。

## 论文材料

论文归档来源：

```text
/home/star/jf/hduthesis.7z
```

解压目录：

```text
/home/star/jf/hduthesis-extracted/hduthesis/
```

主要论文 PDF：

```text
/home/star/jf/hduthesis-extracted/hduthesis/thesis.pdf
/home/star/jf/hduthesis-extracted/hduthesis/231270014_蒋锋_面向大语言模型的信息隐藏技术研究.pdf
```

水印章节：

```text
/home/star/jf/hduthesis-extracted/hduthesis/chapters/ch04_watermark.tex
```

第 4 章核心思想：

- 前缀载体水印：使用短可见字节前缀类别，而不是依赖精确 token 身份。
- 自适应前缀粒度：选择较短前缀长度，同时保留足够熵。
- 指纹驱动 PRF 划分：使用局部可见文本指纹和 prefix UID 决定 green/red 划分。
- 检测：在接收文本上重建 prefix UID 和 fingerprint，并计算 z-score。
- 鲁棒性目标：抵抗 token 级和字符级编辑引起的分词漂移。

## 论文图表

论文包中包含的水印相关图：

```text
/home/star/jf/hduthesis-extracted/hduthesis/figures/watermark_framework.png
/home/star/jf/hduthesis-extracted/hduthesis/figures/prefix_entropy_retention.png
/home/star/jf/hduthesis-extracted/hduthesis/figures/prf_partition.png
/home/star/jf/hduthesis-extracted/hduthesis/figures/tokenization_perturbation_attack.png
/home/star/jf/hduthesis-extracted/hduthesis/figures/tab_token_attack.png
/home/star/jf/hduthesis-extracted/hduthesis/figures/tab_char_attack.png
/home/star/jf/hduthesis-extracted/hduthesis/figures/tpr_retention_relppl.png
/home/star/jf/hduthesis-extracted/hduthesis/figures/robustness_vs_edit_rate.png
/home/star/jf/hduthesis-extracted/hduthesis/figures/tab_robustness_stats.png
```

这些图在 LaTeX 中的引用位置：

```text
/home/star/jf/hduthesis-extracted/hduthesis/chapters/ch04_watermark.tex
```

重要说明：论文目录包含最终 PNG 图，但不是每张最终 PNG 都在 `watermark/` 仓库中有同名精确画图脚本。有些图可能是手动组装，或由当前 `watermark/` 树之外的临时脚本生成。

## 主要方法源码

### 当前论文方法：ByteKGWv6

第 4 章前缀载体/指纹方法最接近的实现位于：

```text
/home/star/jf/watermark/MarkLLM/watermark/bytekgwV6/
```

关键文件：

```text
/home/star/jf/watermark/MarkLLM/watermark/bytekgwV6/watermark.py
/home/star/jf/watermark/MarkLLM/watermark/bytekgwV6/logits_processor.py
/home/star/jf/watermark/MarkLLM/watermark/bytekgwV6/detector.py
/home/star/jf/watermark/MarkLLM/watermark/bytekgwV6/prf.py
/home/star/jf/watermark/MarkLLM/watermark/bytekgwV6/token_bytes.py
/home/star/jf/watermark/MarkLLM/watermark/bytekgwV6/config.py
```

职责：

- `watermark.py`：生成和检测的封装入口。
- `logits_processor.py`：生成阶段对 green tokens 加 `delta` 偏置。
- `detector.py`：重建前缀类别和 PRF 判定，然后计算 z-score。
- `prf.py`：鲁棒局部上下文指纹和 token-prefix 向量划分。
- `token_bytes.py`：将 token ID 映射到可见字节序列和 prefix ID。
- `config.py`：解析 ByteKGWv6 参数。

主配置：

```text
/home/star/jf/watermark/config/ByteKGWv6.json
```

已观察到的重要设置：

```text
delta = 3.0
n_bytes = 3
seed_window_chars = 18
m_bits = 256
target_anchors = 96
k_choices = [1, 2, 3, 4, 5, 6]
k_weight_mode = linear
decision_margin_bits = 12
normalize_whitespace = true
add_special_tokens = true
z_threshold = 4.0
```

### 早期 ByteKGW 版本

旧版本保留在：

```text
/home/star/jf/watermark/MarkLLM/watermark/bytekgw/
/home/star/jf/watermark/MarkLLM/watermark/bytekgwV2/
/home/star/jf/watermark/MarkLLM/watermark/bytekgwV3/
/home/star/jf/watermark/MarkLLM/watermark/bytekgwV4/
/home/star/jf/watermark/MarkLLM/watermark/bytekgwV5/
```

`ByteKGWv5` 被较早的 CSV 实验大量使用：

```text
/home/star/jf/watermark/config/ByteKGWv5.json
```

### Charm/CharmKGW 代码

Charm 相关实现有两套：

```text
/home/star/jf/watermark/MarkLLM/charm/
/home/star/jf/watermark/MarkLLM/charm_v2/
```

v2 关键文件：

```text
/home/star/jf/watermark/MarkLLM/charm_v2/charm_kgw.py
/home/star/jf/watermark/MarkLLM/charm_v2/detector.py
/home/star/jf/watermark/MarkLLM/charm_v2/generator.py
/home/star/jf/watermark/MarkLLM/charm_v2/logits_processor.py
/home/star/jf/watermark/MarkLLM/charm_v2/prf.py
/home/star/jf/watermark/MarkLLM/charm_v2/token_bytes.py
/home/star/jf/watermark/MarkLLM/charm_v2/vocab.py
```

配置：

```text
/home/star/jf/watermark/MarkLLM/config/CharmKGW.json
```

### 基线水印方法

MarkLLM 基线方法位于：

```text
/home/star/jf/watermark/MarkLLM/watermark/kgw/
/home/star/jf/watermark/MarkLLM/watermark/dip/
/home/star/jf/watermark/MarkLLM/watermark/unbiased/
/home/star/jf/watermark/MarkLLM/watermark/synthid/
/home/star/jf/watermark/MarkLLM/watermark/unigram/
```

配置：

```text
/home/star/jf/watermark/config/KGW.json
/home/star/jf/watermark/config/DIP.json
/home/star/jf/watermark/config/Unbiased.json
/home/star/jf/watermark/config/SynthID.json
/home/star/jf/watermark/config/Unigram.json
```

## 生成与评估脚本

### ByteKGWv6 vs KGW 生成

```text
/home/star/jf/watermark/scripts/generate_clean_v6_kgw.py
/home/star/jf/watermark/scripts/generate_v6_vs_kgw_prfnew_2gpus.py
```

典型输出：

```text
/home/star/jf/watermark/outputs/v6_vs_kgw_gen_prfnew_mp/hf_generate.csv
/home/star/jf/watermark/outputs/v6_vs_kgw_gen_prfnew_mp/bytekgw_v6_delta*.csv
/home/star/jf/watermark/outputs/v6_vs_kgw_gen_prfnew_mp/kgw_delta*.csv
```

运行 metadata：

```text
/home/star/jf/watermark/outputs/c4_samples_head_200/run_metadata.json
/home/star/jf/watermark/outputs/c4_samples_bytekgw_all_200/run_metadata.json
```

metadata 中观察到的模型：

```text
../Meta-Llama-3-8B-Instruct
```

### TPR/FPR/PPL 评估

通用 TPR/PPL 脚本：

```text
/home/star/jf/watermark/scripts/analyze_v6_kgw_tpr.py
/home/star/jf/watermark/scripts/analyze_v6_kgw_attack.py
/home/star/jf/watermark/scripts/analyze_watermark_tprs_unified.py
/home/star/jf/watermark/eval_3alg.py
/home/star/jf/watermark/eval_fpr10_tpr_ppl_compare.py
/home/star/jf/watermark/eval_multi_fpr_tpr_ppl_compare.py
```

已有 summary CSV：

```text
/home/star/jf/watermark/outputs/eval_3alg_from_existing.csv
```

该文件包含类似以下列：

```text
target_fpr, algo, delta, thr, achieved_fpr, tpr, ppl_mean, ppl_gap_vs_neg, n_prompts
```

### 攻击实验

字符攻击：

```text
/home/star/jf/watermark/attack_char_3alg_mp.py
/home/star/jf/watermark/attack_char_edit_run_dir.py
/home/star/jf/watermark/attack_charedit_3alg_aligned.py
```

Token 编辑攻击：

```text
/home/star/jf/watermark/scripts/attack_token_edit_run_dir.py
```

混合/随机攻击工具：

```text
/home/star/jf/watermark/random_attack.py
/home/star/jf/watermark/attack_saved_samples_randedit_kbw_kbw.py
/home/star/jf/watermark/scripts/run_attack_grid.py
```

攻击输出 summary：

```text
/home/star/jf/watermark/outputs/attack_char_3alg_er002/attack_char_3alg_summary.csv
/home/star/jf/watermark/outputs/attack_char_3alg_er002_mixed/summary.csv
/home/star/jf/watermark/outputs/attack_char_3alg_er004_mix/summary.csv
/home/star/jf/watermark/outputs/attack_char_3alg_er010_mix/summary.csv
```

### 鲁棒性诊断 / 机制分析

Trace 和机制诊断脚本：

```text
/home/star/jf/watermark/scripts/trace_v6_attack_process.py
/home/star/jf/watermark/scripts/trace_v6_attack_grid.py
/home/star/jf/watermark/scripts/trace_v6_attack_grid_mixchar.py
/home/star/jf/watermark/scripts/analyze_v6_components.py
/home/star/jf/watermark/scripts/sweep_v6_prf_robustness.py
/home/star/jf/watermark/scripts/audit_v6_prefix_consistency.py
/home/star/jf/watermark/scripts/analyze_v6_prf_overlap.py
```

已有诊断输出：

```text
/home/star/jf/watermark/outputs/trace_v6_attack_grid_bytekgw.csv
/home/star/jf/watermark/outputs/trace_v6_attack_grid_bytekgw_mean.csv
/home/star/jf/watermark/outputs/trace_v6_attack_grid_bytekgw_mixchar.csv
/home/star/jf/watermark/outputs/trace_v6_attack_grid_bytekgw_mixchar_mean.csv
/home/star/jf/watermark/outputs/trace_v6_attack_grid_bytekgw_mixchar_mean_by_style.csv
/home/star/jf/watermark/sweep_v6_prf_r002/v6_prf_sweep_results.tsv
/home/star/jf/watermark/traces_v6_process_r002/
```

这些内容最相关的论文讨论包括：

- UID / prefix class 稳定性；
- green decision flip rate；
- 攻击后的 extra token count；
- 字符编辑导致的 tokenization disruption。

### 熵 / 前缀保留诊断

脚本：

```text
/home/star/jf/watermark/scripts/entropy_prefix_sweep.py
```

输出：

```text
/home/star/jf/watermark/outputs/entropy_prefix_sweep/entropy_per_step.csv
/home/star/jf/watermark/outputs/entropy_prefix_sweep/entropy_summary.csv
```

这很可能是 `prefix_entropy_retention.png` 的数据生成侧。

## 画图代码

### 已确认的 Python 画图脚本

以下脚本位于 `watermark/` 目录之外，但与第 4 章水印图直接相关：

```text
/home/star/jf/stega_expand/p.py
/home/star/jf/stega_expand/pic.py
/home/star/jf/stega_expand/pp.py
```

它们使用：

```text
matplotlib
pandas
numpy
```

并使用较统一的现代风格：

```text
font: Inter / Segoe UI / Helvetica / Arial / DejaVu Sans
palette:
  Ours     #3B82F6
  KGW      #10B981
  DIP      #F59E0B
  UniBased #8B5CF6
horizontal grid alpha around 0.15
top/right spines hidden
dpi=300 for PNG
```

具体输出：

```text
/home/star/jf/stega_expand/p.py
  -> robustness_vs_editrate_retention.pdf
  -> robustness_vs_editrate_retention.png

/home/star/jf/stega_expand/pic.py
  -> retention_vs_relppl.pdf
  -> retention_vs_relppl.png

/home/star/jf/stega_expand/pp.py
  -> entropy_retention_modern.pdf
  -> entropy_retention_modern.png
```

`p.py` 和 `pic.py` 的输入 CSV：

```text
/home/star/jf/stega_expand/tpr_attack_by_bucket_0.01.csv
/home/star/jf/stega_expand/tpr_attack_by_bucket_0.02.csv
/home/star/jf/stega_expand/tpr_attack_by_bucket_0.05.csv
/home/star/jf/stega_expand/tpr_attack_by_bucket_0.10.csv
```

这些 CSV 包含：

```text
algo,bucket,atk_style,fpr,threshold,tpr_clean,tpr_attack,n,file,attack_ratio,attack_seed
```

当前生成图：

```text
/home/star/jf/stega_expand/robustness_vs_editrate_retention.png
/home/star/jf/stega_expand/retention_vs_relppl.png
/home/star/jf/stega_expand/entropy_retention_modern.png
```

论文使用的文件名相近，但不完全相同：

```text
/home/star/jf/hduthesis-extracted/hduthesis/figures/robustness_vs_edit_rate.png
/home/star/jf/hduthesis-extracted/hduthesis/figures/tpr_retention_relppl.png
/home/star/jf/hduthesis-extracted/hduthesis/figures/prefix_entropy_retention.png
```

因此这些脚本应视作“目前定位到的最接近画图来源”，而不是论文最终 PNG 的字节级复现脚本。

### 其他画图/可视化代码

MarkLLM 通用可视化器：

```text
/home/star/jf/watermark/MarkLLM/visualize/
```

参考检测器训练图：

```text
/home/star/jf/watermark/train_ref_detector.py
```

该脚本调用 `plt.savefig(fig_path)` 保存训练曲线。

已有图：

```text
/home/star/jf/watermark/plots/asr_comparison.png
```

## 重要结果/数据文件

生成样本和分数：

```text
/home/star/jf/watermark/outputs/v6_vs_kgw_gen_prfnew_mp/
/home/star/jf/watermark/outputs/c4_samples_head_200/
/home/star/jf/watermark/outputs/c4_samples_bytekgw_all_200/
/home/star/jf/watermark/outputs/dip_gen_mp/
/home/star/jf/watermark/outputs/unbiased_gen_mp/
```

水印评估分桶汇总：

```text
/home/star/jf/watermark/outputs/wm_eval/tpr_by_bucket.csv
/home/star/jf/watermark/outputs/wm_eval/tpr_attack_by_bucket.csv
/home/star/jf/watermark/outputs/wm_eval/tpr_attack_token_by_bucket.csv
```

旧备份/权衡 CSV：

```text
/home/star/jf/watermark/backup/tradeoff_charm.csv
/home/star/jf/watermark/backup/tradeoff_charm_ppl.csv
/home/star/jf/watermark/backup/tradeoff_charm_scores.csv
/home/star/jf/watermark/backup/tradeoff_kgw.csv
/home/star/jf/watermark/backup/tradeoff_plain.csv
/home/star/jf/watermark/backup/tradeoff_token.csv
```

## 可能的复现流程

第 4 章实验较可能的复现顺序：

1. 生成 plain、ByteKGWv6 和 KGW 样本：

```text
scripts/generate_clean_v6_kgw.py
```

或使用双 GPU wrapper：

```text
scripts/generate_v6_vs_kgw_prfnew_2gpus.py
```

2. 生成或加载 DiP/Unbiased 样本：

```text
scripts/generate_clean_dip.py
scripts/generate_clean_dip_mp.py
scripts/generate_clean_unbiased.py
scripts/generate_clean_unbiased_mp.py
```

3. 计算 clean TPR/FPR/PPL：

```text
scripts/analyze_v6_kgw_tpr.py
scripts/analyze_watermark_tprs_unified.py
eval_3alg.py
```

4. 运行 token/字符编辑攻击：

```text
scripts/attack_token_edit_run_dir.py
attack_char_3alg_mp.py
scripts/run_attack_grid.py
```

5. 计算机制诊断：

```text
scripts/trace_v6_attack_grid.py
scripts/trace_v6_attack_grid_mixchar.py
scripts/analyze_v6_components.py
scripts/sweep_v6_prf_robustness.py
```

6. 绘制 retention 和 prefix entropy：

```text
/home/star/jf/stega_expand/p.py
/home/star/jf/stega_expand/pic.py
/home/star/jf/stega_expand/pp.py
```

## 尚未找到的内容 / 缺口

- 没有在 `watermark/` 中找到与 `hduthesis/figures/` 下最终 PNG 同名的精确画图脚本。
- `tab_token_attack.png`、`tab_char_attack.png`、`tab_robustness_stats.png` 等表格 PNG 看起来是最终渲染资产，未定位到精确表格生成脚本。
- `watermark_framework.png` 和 `prf_partition.png` 等框架图是最终图像资产，本轮未找到可编辑源文件。
- `stega_expand/p.py`、`pic.py` 和 `pp.py` 是线图/柱图风格的强候选来源，但输出名与最终论文图略有不同。
