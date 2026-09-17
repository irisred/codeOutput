# 数据来源说明

本文档回答一个实际问题：

> 这个复现包里是否包含当前方法的数据？能否判断这些数据是否由本包代码生成？

## 简短结论

包含当前方法数据。

当前方法是 `ByteKGWv6`，其生成样本为：

```text
outputs/v6_vs_kgw_gen_prfnew_mp/bytekgw_v6_delta1.0.csv
outputs/v6_vs_kgw_gen_prfnew_mp/bytekgw_v6_delta2.0.csv
outputs/v6_vs_kgw_gen_prfnew_mp/bytekgw_v6_delta3.0.csv
outputs/v6_vs_kgw_gen_prfnew_mp/bytekgw_v6_delta4.0.csv
outputs/v6_vs_kgw_gen_prfnew_mp/bytekgw_v6_delta5.0.csv
```

同一次生成中的 clean 文本和 KGW 基线文件为：

```text
outputs/v6_vs_kgw_gen_prfnew_mp/hf_generate.csv
outputs/v6_vs_kgw_gen_prfnew_mp/kgw_delta*.csv
```

已经评估并按桶划分的当前方法文件为：

```text
outputs/wm_eval/bytekgw_v6/
```

## 文件与代码匹配的证据

### 1. CSV 中的方法标签

`outputs/v6_vs_kgw_gen_prfnew_mp/bytekgw_v6_delta*.csv` 文件包含：

```text
algo = bytekgwV6
```

并包含以下列：

```text
algo,delta,device,seed,prompt_index,prompt_id,prompt_text,full_text,continuation_text,gen_params_json
```

这与 `scripts/generate_clean_v6_kgw.py` 产生的输出 schema 匹配。

### 2. 脚本输出命名与现有文件匹配

`scripts/generate_clean_v6_kgw.py` 会写出：

```text
hf_generate.csv
bytekgw_v6_delta{delta}.csv
kgw_delta{delta}.csv
```

本包中的目录正好使用这一命名模式：

```text
outputs/v6_vs_kgw_gen_prfnew_mp/
```

### 3. prompt metadata 与生成 CSV 匹配

生成 metadata 文件：

```text
outputs/c4_samples_head_200/run_metadata.json
```

包含：

```text
model = ../Meta-Llama-3-8B-Instruct
n_prompts = 200
seed_base = 1234
deltas = [1.0, 2.0, 3.0, 4.0, 5.0]
generation: temperature=1.0, top_p=0.95, top_k=50, max_new_tokens=128
```

`outputs/v6_vs_kgw_gen_prfnew_mp/` 中的生成 CSV 使用相同的 prompts、seeds、deltas 和 generation kwargs。

### 4. 配置与论文方法匹配

当前方法配置文件为：

```text
config/ByteKGWv6.json
```

重要参数：

```text
delta = 3.0 default, swept as 1.0..5.0 in generation
n_bytes = 3
seed_window_chars = 18
m_bits = 256
target_anchors = 96
k_choices = [1, 2, 3, 4, 5, 6]
k_weight_mode = linear
decision_margin_bits = 12
normalize_whitespace = true
```

这些参数对应第 4 章中的方法描述：前缀类别加指纹驱动的 PRF 划分。

## 仅从文件无法证明的内容

CSV 文件没有嵌入精确 shell 命令、git commit 或源码文件 hash。

因此最准确的表述是：

```text
本包包含相关数据，并且这些数据与本包中的 scripts/generate_clean_v6_kgw.py 和 ByteKGWv6 实现、metadata、config 高度一致。原始精确命令行没有写入 CSV 文件。
```

这也是 `REPRODUCIBILITY.md` 提供“可生成同类文件的命令”，而不承诺 bit-for-bit 完全一致的原因。完全一致可能受 CUDA、Transformers、PyTorch、模型版本和采样确定性影响。

## 包含的数据类别

### 当前方法

```text
outputs/v6_vs_kgw_gen_prfnew_mp/bytekgw_v6_delta*.csv
outputs/wm_eval/bytekgw_v6/
```

### clean cover texts

```text
outputs/v6_vs_kgw_gen_prfnew_mp/hf_generate.csv
outputs/wm_eval/clean/hf_generate.csv
```

### 基线方法

```text
outputs/v6_vs_kgw_gen_prfnew_mp/kgw_delta*.csv
outputs/wm_eval/kgw/
outputs/wm_eval/dip/
outputs/wm_eval/unbiased/
outputs/dip_gen_mp/
outputs/unbiased_gen_mp/
```

本包中由完整实验数据实际支撑的比较方法是：

```text
KGW
DiP
Unbiased
```

其他 MarkLLM 方法，如 SynthID、Unigram、SIR、SWEET、TS、UPV、XSIR、EXP 和 EXP-Gumbel，在体积允许时保留了源码/配置模块，但 clean 包不包含这些方法的完整第 4 章实验结果表。它们保留为可运行或可检查的基线，而不是完整结果行。

### 机制诊断

```text
outputs/trace_v6_attack_grid_bytekgw*.csv
sweep_v6_prf_r002/
traces_v6_process_r002/
outputs/entropy_prefix_sweep/
```

### 画图输入

```text
plotting/tpr_attack_by_bucket_0.01.csv
plotting/tpr_attack_by_bucket_0.02.csv
plotting/tpr_attack_by_bucket_0.05.csv
plotting/tpr_attack_by_bucket_0.10.csv
```

## 排除的数据

大型或相关性较低的历史目录没有纳入，例如：

```text
outputs/wm_eval_attacked/
metrics/
textattack/
his/
large cache/log/archive files
```

这些内容不是当前 `ByteKGWv6` 复现路径所必需的。
