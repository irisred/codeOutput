# 第 4 章实验审计

本文档把 `thesis/chapters/ch04_watermark.tex` 中的实验，映射到 clean 仓库中已经包含的代码、数据、图表和冒烟测试。

该审计采用保守口径：如果没有找到精确的最终表格渲染脚本，即使底层 CSV 数据已包含，也会明确说明。

## 1. 全局实验设置

论文位置：

```text
thesis/chapters/ch04_watermark.tex
```

相关章节：

```text
Section 4.3.1: 实验设置
```

论文实验设置：

```text
Dataset: C4
Prompt count: 200
Model: Llama3-8B-Chat / Llama3-8B-Instruct local checkpoint
Baselines: KGW, DiPmark, Unbiased Watermark
Main metrics: TPR at calibrated FPR, attacked TPR, retention, RelPPL buckets
Attack types: token-level edit and character-level edit
```

已包含证据：

```text
outputs/c4_samples_head_200/run_metadata.json
outputs/wm_eval/clean/hf_generate.csv
dataset/c4/realnewslike/c4-train.00000-of-00512.json.gz
```

已包含样本规模：

```text
outputs/wm_eval/clean/hf_generate.csv: 200
outputs/wm_eval/bytekgw_v6/bytekgw_v6_delta*.csv: 5 files x 200
outputs/wm_eval/kgw/kgw_delta*.csv: 5 files x 200
outputs/wm_eval/dip/dip_a*.csv: 3 files x 200
outputs/wm_eval/unbiased/unbiased.csv: 200
```

状态：

```text
已覆盖。仓库包含第 4 章比较实验所用的数据规模和方法集合。
```

## 2. 方法实现

论文内容：

```text
Section 4.2: prefix-carrier watermarking, adaptive prefix retention, fingerprint-driven PRF, z-score detection
```

当前方法源码：

```text
MarkLLM/watermark/bytekgwV6/watermark.py
MarkLLM/watermark/bytekgwV6/logits_processor.py
MarkLLM/watermark/bytekgwV6/detector.py
MarkLLM/watermark/bytekgwV6/prf.py
MarkLLM/watermark/bytekgwV6/token_bytes.py
MarkLLM/watermark/bytekgwV6/config.py
config/ByteKGWv6.json
```

冒烟状态：

```text
通过 import 和 py_compile 检查。
ByteKGWv6 生成冒烟测试也已通过。
```

详情见：

```text
SMOKE_TEST_RESULTS.md
```

## 3. 基线方法

论文内容：

```text
Section 4.3.1: KGW, DiPmark, Unbiased Watermark
```

已包含源码/配置：

```text
MarkLLM/watermark/kgw/
MarkLLM/watermark/dip/
MarkLLM/watermark/unbiased/
config/KGW.json
config/DIP.json
config/Unbiased.json
```

已包含结果数据：

```text
outputs/wm_eval/kgw/
outputs/wm_eval/dip/
outputs/wm_eval/unbiased/
outputs/dip_gen_mp/
outputs/unbiased_gen_mp/
```

冒烟状态：

```text
KGW 生成和检测已冒烟测试。
DiP 和 Unbiased 的导入路径已冒烟测试。
已包含的比较 CSV 有 KGW、DiP 和 Unbiased 的完整结果行。
```

重要边界：

```text
其他 MarkLLM 方法在体积允许时作为源码/配置保留，但第 4 章完整比较结果行只包含 KGW、DiP 和 Unbiased。
```

## 4. 前缀熵保留诊断

论文图：

```text
thesis/figures/prefix_entropy_retention.png
```

目的：

```text
展示短可见字节前缀在若干字节后可以保留 token 分布中的大部分熵。
```

已包含代码/数据：

```text
scripts/entropy_prefix_sweep.py
outputs/entropy_prefix_sweep/entropy_per_step.csv
outputs/entropy_prefix_sweep/entropy_summary.csv
plotting/pp.py
```

冒烟状态：

```text
以 samples=1 和 max_new_tokens=4 成功执行。
观测到熵比例从 1 字节约 0.58 上升到 3 字节约 0.98。
```

状态：

```text
实验逻辑已覆盖并通过冒烟测试。
最终论文 PNG 已包含。
重新生成的图不保证与论文 PNG 字节级完全一致。
```

## 5. Clean detection / calibrated FPR 下的 TPR

论文内容：

```text
Section 4.3.2 使用 calibrated FPR 阈值并报告 TPR_clean。
```

已包含代码/数据：

```text
scripts/analyze_tpr_by_bucket.py
scripts/analyze_watermark_tprs_unified.py
outputs/wm_eval/tpr_by_bucket.csv
outputs/wm_eval/*/ppl_*.csv
```

按 PPL 桶聚合的快速检查：

```text
Clean TPR, FPR=0.01:
kgw        0.882
bytekgw_v6 0.848
dip        0.745
unbiased   0.736
```

状态：

```text
已覆盖。clean TPR 结果可用；结果显示 clean 检测并非单边优势：严格 FPR 下，攻击前 KGW 可能略高于 ByteKGWv6。
```

## 6. Token 级攻击表

论文图：

```text
thesis/figures/tab_token_attack.png
```

目的：

```text
在固定编辑预算下比较 token 级扰动后的 TPR_clean 与 TPR_attack。
```

已包含代码/数据：

```text
scripts/analyze_attack_tpr_by_bucket.py
scripts/run_attack_grid.py
outputs/wm_eval/tpr_attack_token_by_bucket.csv
outputs/wm_eval/*/ppl_*.csv
```

按 PPL 桶聚合的快速检查：

```text
Token attack, FPR=0.01:
bytekgw_v6 retention = 0.957
kgw        retention = 0.902
unbiased   retention = 0.777
dip        retention = 0.760
```

状态：

```text
底层实验数据和分析代码已覆盖。
最终渲染出的表格 PNG 已包含。
未找到把该 CSV 渲染为最终表格 PNG 的精确脚本。
```

## 7. 字符级攻击表

论文图：

```text
thesis/figures/tab_char_attack.png
```

目的：

```text
在固定编辑预算下比较字符级扰动后的 TPR_clean 与 TPR_attack。
```

已包含代码/数据：

```text
scripts/analyze_attack_tpr_by_bucket.py
scripts/run_attack_grid.py
outputs/wm_eval/tpr_attack_by_bucket.csv
outputs/wm_eval/*/ppl_*.csv
```

按 PPL 桶聚合的快速检查：

```text
Character attack, FPR=0.01:
bytekgw_v6 retention = 0.953
kgw        retention = 0.907
dip        retention = 0.794
unbiased   retention = 0.719
```

状态：

```text
底层实验数据和分析代码已覆盖。
最终渲染出的表格 PNG 已包含。
未找到把该 CSV 渲染为最终表格 PNG 的精确脚本。
```

## 8. RelPPL 下的 retention 图

论文图：

```text
thesis/figures/tpr_retention_relppl.png
```

目的：

```text
在固定编辑预算下，按 RelPPL 分桶展示攻击后的 retention。
```

已包含代码/数据：

```text
plotting/pic.py
plotting/tpr_attack_by_bucket_0.01.csv
plotting/tpr_attack_by_bucket_0.02.csv
plotting/tpr_attack_by_bucket_0.05.csv
plotting/tpr_attack_by_bucket_0.10.csv
outputs/wm_eval/tpr_attack_by_bucket.csv
outputs/wm_eval/tpr_attack_token_by_bucket.csv
```

冒烟状态：

```text
画图脚本成功执行，并重新生成：
plotting/retention_vs_relppl.png
plotting/retention_vs_relppl.pdf
```

状态：

```text
已覆盖并通过冒烟测试。
重新生成的图尺寸与最终论文 PNG 不同，因此属于风格/效果复现，不是字节级复现。
```

## 9. 编辑率敏感性图

论文图：

```text
thesis/figures/robustness_vs_edit_rate.png
```

目的：

```text
编辑率从 0 到 10% 扫描，比较 token/字符攻击下 retention 的退化。
```

已包含代码/数据：

```text
scripts/run_attack_grid.py
scripts/analyze_attack_tpr_by_bucket.py
plotting/p.py
plotting/tpr_attack_by_bucket_0.01.csv
plotting/tpr_attack_by_bucket_0.02.csv
plotting/tpr_attack_by_bucket_0.05.csv
plotting/tpr_attack_by_bucket_0.10.csv
```

冒烟状态：

```text
画图脚本成功执行，并重新生成：
plotting/robustness_vs_editrate_retention.png
plotting/robustness_vs_editrate_retention.pdf
```

状态：

```text
已覆盖并通过冒烟测试。
画图所用完整编辑率网格数据已作为 CSV 包含。
图不保证与论文 PNG 字节级完全一致。
```

## 10. 鲁棒性分析表

论文图：

```text
thesis/figures/tab_robustness_stats.png
```

目的：

```text
用以下过程统计解释鲁棒性退化：
1. green decision flip rate；
2. prefix UID / carrier-category match rate；
3. tokenization drift 导致的 extra token count。
```

已包含代码/数据：

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

冒烟状态：

```text
scripts/trace_v6_attack_grid.py 在 3 行数据和 2 个攻击比例上运行成功。
观测：
attack_ratio=0.0  -> tail_match=1.0, green_flip=0.0, extra_tokens=0.0
attack_ratio=0.02 -> tail_match=0.70, green_flip=0.0267, extra_tokens=5.0
```

状态：

```text
分析逻辑和诊断 CSV 已覆盖并通过冒烟测试。
最终渲染表格 PNG 已包含。
未找到将分析 CSV 渲染为最终 PNG 的精确脚本。
```

## 11. RelPPL / PPL 桶构造

论文内容：

```text
Lines 196-203 定义 RelPPL 和分桶报告。
```

已包含代码/数据：

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

状态：

```text
已覆盖。ByteKGWv6、KGW、DiP 和 Unbiased 都包含分桶结果文件。
```

重要说明：

```text
鲁棒性 summary 表使用已有分桶 CSV。完整 PPL 重新计算需要加载 Llama3 模型，耗时更长；冒烟测试没有完整重跑。
```

## 12. 概念图

论文图：

```text
thesis/figures/watermark_framework.png
thesis/figures/prf_partition.png
thesis/figures/tokenization_perturbation_attack.png
```

状态：

```text
最终 PNG 资产已包含。
这些是概念图，不是实验图。
未在已定位仓库中找到可编辑图源。
```

## 总体结论

clean 仓库覆盖第 4 章主要实验主张：

```text
1. 200-prompt C4/Llama3 设置。
2. ByteKGWv6 当前方法实现。
3. KGW、DiP 和 Unbiased 基线结果数据。
4. clean TPR 与 calibrated FPR 评估。
5. token 级和字符级攻击评估。
6. RelPPL/PPL 分桶。
7. 编辑率敏感性画图。
8. 前缀熵保留诊断。
9. 基于 green flip / UID match / extra token 统计的鲁棒性机制分析。
```

主要限制：

```text
1. 部分最终论文 PNG 只有渲染资产，没有找到精确源码脚本。
2. 未找到 tab_token_attack.png、tab_char_attack.png、tab_robustness_stats.png 的精确表格渲染脚本。
3. 没有完整重跑论文规模实验，只执行了小规模冒烟测试。
4. 因为并非所有 CSV 都包含原始命令、代码 commit hash 和库版本，所以不保证 bit-for-bit 复现。
```

最准确的表述是：

```text
本包可以复现主要实验逻辑和定性效果，并包含主要比较背后的结果 CSV。它适合未来验证和扩大规模重跑，但并非每一张最终论文图都能从已定位的画图脚本中做到字节级复现。
```
