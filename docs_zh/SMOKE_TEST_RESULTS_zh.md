# 冒烟测试结果

本文档记录 2026-06-02 在 clean 仓库上运行的小规模检查，用于验证仓库可执行，并确认已包含的代码/数据能够在轻量规模上复现主要实验方向。

所有命令均在以下目录运行：

```bash
cd /home/star/jf/watermark-repro-clean
export PYTHONPATH=.
```

使用的本地 Python：

```bash
/home/star/jf/python/stega/bin/python
```

## 1. 导入与语法检查

命令：

```bash
/home/star/jf/python/stega/bin/python scripts/smoke_imports.py
```

结果：通过。脚本成功导入：

- `MarkLLM.watermark.bytekgwV6.*`
- `MarkLLM.watermark.kgw.kgw`
- `MarkLLM.watermark.dip.dip`
- `MarkLLM.watermark.unbiased.unbiased`
- `MarkLLM.charm_v2.*`

关键生成、检测、trace、熵诊断和画图脚本也通过了 `py_compile` 编译检查。

## 2. ByteKGWv6 / KGW 生成冒烟测试

命令：

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

结果：通过。命令加载了本地 Llama3 模型，并写出：

```text
outputs/smoke_v6_vs_kgw/hf_generate.csv
outputs/smoke_v6_vs_kgw/bytekgw_v6_delta1.0.csv
outputs/smoke_v6_vs_kgw/kgw_delta1.0.csv
```

这验证了当前方法和 KGW 比较生成入口可以运行。

## 3. TPR/FPR 检测冒烟测试

命令：

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

观测结果：

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

这不是完整表格复现，但确认 clean 数据、配置、tokenizer 和检测代码可以产生 clean 与水印文本之间的预期分离。

## 4. 攻击 trace 冒烟测试

命令：

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

按 attack ratio 求均值的观测结果：

```text
attack_ratio,mean_tail_match,mean_green_flip,mean_extra_tokens,mean_traced
0.0,1.0,0.0,0.0,50.0
0.02,0.7000000000000001,0.02666666666666667,5.0,50.0
```

这验证鲁棒性/trace 诊断路径可以运行，并产生可解释的过程统计。

## 5. 前缀熵保留冒烟测试

命令：

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

观测 summary：

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

这符合预期定性行为：少量前缀字节即可保留大部分完整 token 字节熵。

## 6. 画图冒烟测试

命令：

```bash
cd plotting
MPLBACKEND=Agg /home/star/jf/python/stega/bin/python p.py
MPLBACKEND=Agg /home/star/jf/python/stega/bin/python pic.py
MPLBACKEND=Agg /home/star/jf/python/stega/bin/python pp.py
```

结果：通过。画图脚本重新生成了：

```text
plotting/robustness_vs_editrate_retention.pdf
plotting/robustness_vs_editrate_retention.png
plotting/retention_vs_relppl.pdf
plotting/retention_vs_relppl.png
plotting/entropy_retention_modern.pdf
plotting/entropy_retention_modern.png
```

只观察到 pandas 未来版本警告；图文件成功生成。

## 解释

这些冒烟测试验证 clean 仓库能够：

- 运行当前 ByteKGWv6 方法和 KGW 基线生成；
- 基于已包含数据评估 clean 与水印文本的检测行为；
- 运行鲁棒性 trace 诊断；
- 运行前缀熵诊断；
- 重新生成已包含的画图输出。

这些检查不能替代完整论文规模实验。完整复现应使用 `REPRODUCIBILITY.md` 中的大规模命令。
