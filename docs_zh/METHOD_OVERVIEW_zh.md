# 论文方法与仓库使用总览

本文档按论文第 4 章“基于前缀承载的水印算法”的思路，概括方法动机、核心设计、实验验证，以及本仓库与论文之间的关系。它适合作为进入本仓库前的第一篇导读。

## 1. 研究问题

推理期文本水印的目标是在不明显破坏生成质量的前提下，让模型输出文本携带可检测的统计信号。典型方法会在生成时根据密钥和上下文构造绿色集合，对绿色 token 施加轻微偏置；检测时再用相同规则重建绿色集合，统计绿色命中次数并计算 z-score。

问题在于，真实传播环境中的文本常会经历复制、轻度润色、改写、token 级替换或字符级编辑。尤其是字符级扰动，哪怕只改动少量字符，也可能触发分词边界变化，导致检测端看到的 token 序列与生成端不再对齐。传统依赖“具体 token 身份”和“精确 token 上下文”的水印方法，在这种情况下容易出现两个失稳点：

- 承载对象失稳：原来承载水印的 token 被拆分、合并或替换，检测端无法在相同位置看到相同 token。
- 分区结构失稳：绿色集合由精确上下文驱动，局部编辑后检测端难以复现生成端的伪随机分区。

论文第 4 章要解决的核心问题就是：如何让推理期文本水印在分词扰动条件下仍然保持较好的可检测性。

## 2. 方法主线

论文提出的方法可以概括为一句话：

```text
把水印承载对象从完整 token 身份提升到更稳定的可见字节前缀类别，并用局部可见文本指纹驱动 PRF 分区，从而降低检测对精确分词边界的依赖。
```

它不是要恢复某段显式消息，而是判断一段文本是否呈现出带水印生成过程诱导出的统计偏移。

整体流程分为生成端和检测端：

- 生成端：模型给出下一 token 分布后，方法为每个候选 token 提取前缀类别，结合当前可见上下文指纹和密钥判断它是否属于绿色集合；对绿色集合加 logit 偏置后采样。
- 检测端：对待检测文本逐位置重建前缀类别和上下文指纹，用相同密钥复现绿色判定，统计绿色命中比例并计算 z-score。

对应代码中，当前最接近论文方法的是：

```text
MarkLLM/watermark/bytekgwV6/
```

## 3. 关键设计一：前缀承载

传统 KGW 类方法通常把完整 token 身份作为水印承载对象。这样做在无扰动场景下简单有效，但在字符级编辑引发分词变化时很脆弱。

论文的方法不直接使用完整 token 身份，而是取 token 可见字节表示的短前缀，把具有相同短前缀的 token 归入同一类。这个类别就是水印承载对象。

这样设计的直觉是：

- 完整 token 身份粒度太细，分词稍变就可能完全对不上。
- 短前缀类别粒度更粗，即使 token 被替换或分词边界发生变化，也更可能保留相近的可见前缀。
- 因此，水印信号绑定在前缀类别上，比绑定在具体 token 上更稳定。

但前缀不能无限短。过短会把大量候选压到少数类别中，损失原分布的不确定性，水印承载空间变小；过长又接近完整 token，鲁棒性优势减弱。

因此论文引入前缀熵保留思想：比较前缀类别分布熵与原 token 分布熵，选择能保留足够信息量的较短前缀。论文中用信息保留率描述这个权衡，实验中展示了短前缀在若干字节后能快速保留大部分分布熵。

代码映射：

```text
MarkLLM/watermark/bytekgwV6/token_bytes.py
MarkLLM/watermark/bytekgwV6/logits_processor.py
scripts/entropy_prefix_sweep.py
outputs/entropy_prefix_sweep/
```

## 4. 关键设计二：指纹驱动 PRF 分区

仅把承载对象改成前缀类别还不够。如果绿色集合仍直接依赖精确 token 上下文，那么轻微编辑后，检测端仍可能无法复现生成端的分区。

论文进一步引入局部可见文本指纹。生成端和检测端都从当前位置之前的可见文本中提取一段局部上下文，并映射成固定长度二进制指纹。实现上采用类似 SimHash 的思想，让相近上下文具有相对稳定的二进制表示。

同时，每个前缀类别会通过密钥控制的 PRF 映射成一个同维度的二进制向量。然后比较上下文指纹与前缀类别向量之间的汉明距离，决定当前候选是否属于绿色集合。

这一步的意义是：

- 分区输入不再是精确 token 上下文，而是可见文本指纹。
- 候选身份不再是完整 token，而是前缀类别。
- 密钥仍控制伪随机性，外部不知道密钥时难以预测绿色集合。
- 检测端在轻度编辑后更有机会重建相近或一致的判定结构。

代码映射：

```text
MarkLLM/watermark/bytekgwV6/prf.py
MarkLLM/watermark/bytekgwV6/detector.py
MarkLLM/watermark/bytekgwV6/logits_processor.py
config/ByteKGWv6.json
```

常见相关参数：

```text
m_bits
target_anchors
k_choices
k_weight_mode
decision_margin_bits
seed_window_chars
normalize_whitespace
```

## 5. 水印嵌入与检测

生成阶段，语言模型先输出原始 logits。方法根据前缀类别和上下文指纹构造绿色集合，对绿色候选加上偏置强度 `delta`，再从偏置后的分布采样。

简化理解：

```text
原始 logits -> 判定绿色集合 -> green token 加 delta -> 采样生成
```

检测阶段，算法对待检测文本逐位置重建同样的绿色判定，统计实际 token 是否命中绿色集合。若绿色命中数显著高于无水印文本的基准水平，则 z-score 升高，超过阈值时判为带水印文本。

检测并不需要访问模型 logits。它依赖的是：

- 相同 tokenizer；
- 相同密钥；
- 相同前缀类别构造规则；
- 相同指纹与 PRF 判定规则；
- 待检测文本本身。

代码映射：

```text
MarkLLM/watermark/bytekgwV6/watermark.py
MarkLLM/watermark/bytekgwV6/logits_processor.py
MarkLLM/watermark/bytekgwV6/detector.py
scripts/generate_clean_v6_kgw.py
scripts/analyze_v6_kgw_tpr.py
scripts/analyze_v6_kgw_attack.py
```

## 6. 实验设计

论文实验主要围绕三个问题展开：

1. 无扰动或轻度扰动下，方法是否能有效检测水印？
2. token 级和字符级编辑后，方法相较 KGW、DiP、Unbiased 是否保留更高检测能力？
3. 编辑率升高时，方法的检测保持率如何退化，退化原因是什么？

实验设置大致为：

```text
数据集：C4
prompt 数量：200
模型：Llama3-8B-Chat / Llama3-8B-Instruct 本地 checkpoint
基线：KGW、DiPmark、Unbiased Watermark
攻击：token 级编辑、字符级编辑
指标：TPR_clean、TPR_attack、Retention、RelPPL 分桶
阈值：在无水印文本上按目标 FPR 校准
```

本仓库中保留的主要结果数据位于：

```text
outputs/v6_vs_kgw_gen_prfnew_mp/
outputs/wm_eval/
outputs/entropy_prefix_sweep/
outputs/trace_v6_attack_grid_bytekgw*.csv
sweep_v6_prf_r002/
traces_v6_process_r002/
```

## 7. 实验结论的准确表述

从 clean 包中保留的数据和冒烟测试看，最稳妥的结论是：

```text
ByteKGWv6 支持论文关于编辑攻击下鲁棒性保持的主要主张；它在 token 级和字符级编辑后通常具有更高或更稳定的 retention。
```

需要注意的是，论文和仓库数据并不支持一个过强说法：即 ByteKGWv6 在所有无攻击 clean TPR 设置下都最高。实际上，在严格 FPR 和无攻击条件下，KGW 有时可能略高于 ByteKGWv6。ByteKGWv6 的优势主要体现在编辑扰动之后的保持率和退化速度上。

这个边界在以下文档中也有说明：

```text
docs_zh/FIGURE_REPRO_STATUS_zh.md
docs_zh/EXPERIMENT_AUDIT_zh.md
```

## 8. 本仓库与论文的关系

`watermark-repro-clean` 不是原始完整开发目录，而是从 `/home/star/jf/watermark` 清理出的论文第 4 章复现包。

它和论文的关系可以这样理解：

- `thesis/chapters/ch04_watermark.tex` 是论文第 4 章正文。
- `thesis/figures/` 包含论文最终使用的图表 PNG。
- `MarkLLM/watermark/bytekgwV6/` 是当前方法的核心实现。
- `MarkLLM/watermark/kgw/`、`dip/`、`unbiased/` 是主要基线实现。
- `scripts/` 中保存生成、检测、攻击、机制诊断等脚本。
- `outputs/` 中保存筛选后的关键实验结果和统计文件。
- `plotting/` 中保存目前定位到的最接近论文部分图的画图脚本和输入 CSV。
- `docs_zh/` 中保存中文导读和复现说明。

也就是说，本仓库的目标是让人能理解并复查第 4 章方法与主要实验逻辑，而不是完整保留所有开发历史、中间日志、缓存和全部旧实验。

完整原始开发现场是：

```text
/home/star/jf/watermark
```

clean 复现包是：

```text
/home/star/jf/watermark-repro-clean
```

## 9. 如何使用本仓库

### 9.1 只想理解方法

建议按这个顺序读：

```text
docs_zh/METHOD_OVERVIEW_zh.md
thesis/chapters/ch04_watermark.tex
docs_zh/WATERMARK_CODE_MAP_zh.md
MarkLLM/watermark/bytekgwV6/
```

重点文件：

```text
MarkLLM/watermark/bytekgwV6/token_bytes.py
MarkLLM/watermark/bytekgwV6/prf.py
MarkLLM/watermark/bytekgwV6/logits_processor.py
MarkLLM/watermark/bytekgwV6/detector.py
```

### 9.2 想确认数据是否对应这套代码

先看：

```text
docs_zh/DATA_PROVENANCE_zh.md
outputs/c4_samples_head_200/run_metadata.json
outputs/v6_vs_kgw_gen_prfnew_mp/
outputs/wm_eval/
```

其中主模型路径来自：

```text
outputs/c4_samples_head_200/run_metadata.json
```

脚本通过 `--run_meta` 读取该文件，不是在 ByteKGWv6 核心代码里写死模型路径。

### 9.3 想运行快速检查

从仓库根目录：

```bash
cd /home/star/jf/watermark-repro-clean
export PYTHONPATH=.
python scripts/smoke_imports.py
```

更多已经实际运行过的检查见：

```text
docs_zh/SMOKE_TEST_RESULTS_zh.md
```

### 9.4 想复现实验

先准备环境和模型权重，然后按：

```text
docs_zh/REPRODUCIBILITY_zh.md
```

中的命令运行。主流程通常是：

1. 读取 `run_metadata.json` 中的 prompt、模型路径和生成参数。
2. 用 `scripts/generate_clean_v6_kgw.py` 生成 clean、ByteKGWv6、KGW 样本。
3. 用 `scripts/analyze_v6_kgw_tpr.py` 评估 clean TPR/FPR/PPL。
4. 用 `scripts/analyze_v6_kgw_attack.py` 或相关攻击脚本评估编辑扰动。
5. 用 `plotting/` 中脚本重画部分图。

### 9.5 想检查论文图表是否能复现

看：

```text
docs_zh/FIGURE_REPRO_STATUS_zh.md
docs_zh/EXPERIMENT_AUDIT_zh.md
```

重要边界：

- 部分图可以通过已定位脚本重新生成相近风格和趋势。
- 部分最终表格 PNG 只保留了最终图，未找到精确渲染脚本。
- 概念图如 `watermark_framework.png`、`prf_partition.png` 是最终资产，未找到可编辑源文件。

## 10. 复现边界与注意事项

本仓库适合：

- 理解论文第 4 章方法；
- 查看当前方法 ByteKGWv6 的实现；
- 复查主要实验数据来源；
- 运行小规模检查；
- 在具备模型权重和环境时重跑主要实验流程。

本仓库不保证：

- 完整保存所有历史实验与日志；
- 所有论文最终 PNG 都可由已定位脚本字节级重现；
- 在不同 CUDA、PyTorch、Transformers、模型 revision 下生成文本完全一致；
- 所有 MarkLLM baseline 都有完整第 4 章结果表。

最准确的定位是：

```text
watermark-repro-clean 是第 4 章水印方法的清理复现包。它保留了核心实现、关键数据、主要结果和复现说明，适合验证论文方法逻辑与主要实验结论；若要追溯全部历史开发材料，应回到原始 watermark 目录。
```

