# 本地自包含环境运行记录

本文档记录 `/home/star/watermark-repro-clean` 当前工作副本的本地环境、模型路径和已通过的轻量验证。记录时间：2026-06-16。

## 当前状态

已完成：

- 在项目目录内创建虚拟环境：`.venv/`
- 按 `requirements.txt` 安装依赖
- 在项目目录内创建本地模型软链接：`models/`
- 将主流程 metadata 的模型路径改为项目内路径
- 跑通论文主路径的轻量 smoke 测试

轻量备份仍保留在：

```text
/home/star/jf/watermark-repro-clean
```

当前工作副本位于：

```text
/home/star/watermark-repro-clean
```

## 虚拟环境

虚拟环境路径：

```text
/home/star/watermark-repro-clean/.venv
```

激活方式：

```bash
cd /home/star/watermark-repro-clean
source .venv/bin/activate
export PYTHONPATH=.
```

已验证版本：

```text
python = 3.12.3
torch = 2.5.1+cu124
torch cuda = 12.4
cuda_available = True
device_count = 2
transformers = 4.49.0
```

`.venv` 大小约 6.1G。由于模型使用软链接，`models/` 本身只占很小空间。

## 模型路径

模型没有从 Hugging Face 下载，而是通过软链接复用 `/home/star/jf` 下已有本地模型：

```text
models/Meta-Llama-3-8B-Instruct -> /home/star/jf/Meta-Llama-3-8B-Instruct
models/Qwen2.5-3B -> /home/star/jf/Qwen2.5-3B
```

已验证 tokenizer 可本地加载：

```text
models/Meta-Llama-3-8B-Instruct -> PreTrainedTokenizerFast, vocab size 128256
models/Qwen2.5-3B -> Qwen2TokenizerFast, vocab size 151665
```

主流程 metadata 已改为：

```json
"model": "models/Meta-Llama-3-8B-Instruct"
```

位置：

```text
outputs/c4_samples_head_200/run_metadata.json
```

## 已通过的检查

### 1. 基础导入

命令：

```bash
PYTHONPATH=. .venv/bin/python scripts/smoke_imports.py
```

结果：通过。以下模块导入成功：

```text
MarkLLM.watermark.bytekgwV6.watermark
MarkLLM.watermark.bytekgwV6.logits_processor
MarkLLM.watermark.bytekgwV6.detector
MarkLLM.watermark.bytekgwV6.prf
MarkLLM.watermark.bytekgwV6.token_bytes
MarkLLM.watermark.kgw.kgw
MarkLLM.watermark.dip.dip
MarkLLM.watermark.unbiased.unbiased
MarkLLM.charm_v2.charm_kgw
MarkLLM.charm_v2.detector
```

### 2. 关键脚本编译

命令：

```bash
PYTHONPATH=. .venv/bin/python -m py_compile \
  MarkLLM/watermark/bytekgwV6/watermark.py \
  scripts/generate_clean_v6_kgw.py \
  scripts/analyze_v6_kgw_tpr.py \
  scripts/trace_v6_attack_grid.py \
  scripts/entropy_prefix_sweep.py
```

结果：通过。

### 3. 生成 smoke

命令：

```bash
TOKENIZERS_PARALLELISM=false PYTHONPATH=. .venv/bin/python \
  scripts/generate_clean_v6_kgw.py \
  --run_meta outputs/c4_samples_head_200/run_metadata.json \
  --v6_config config/ByteKGWv6.json \
  --kgw_config config/KGW.json \
  --output_dir outputs/smoke_v6_vs_kgw_localvenv \
  --deltas 1 \
  --n_prompts 2 \
  --devices cuda:0
```

结果：通过，输出：

```text
outputs/smoke_v6_vs_kgw_localvenv/hf_generate.csv
outputs/smoke_v6_vs_kgw_localvenv/bytekgw_v6_delta1.0.csv
outputs/smoke_v6_vs_kgw_localvenv/kgw_delta1.0.csv
```

### 4. 检测 smoke

命令：

```bash
TOKENIZERS_PARALLELISM=false PYTHONPATH=. .venv/bin/python \
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

结果：通过。关键观测：

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

### 5. Trace 诊断 smoke

命令：

```bash
TOKENIZERS_PARALLELISM=false PYTHONPATH=. .venv/bin/python \
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
  --out_csv outputs/smoke_trace_v6_attack_grid_bytekgw_localvenv.csv
```

结果：通过，输出：

```text
outputs/smoke_trace_v6_attack_grid_bytekgw_localvenv.csv
outputs/smoke_trace_v6_attack_grid_bytekgw_localvenv_mean.csv
```

### 6. 前缀熵诊断 smoke

命令：

```bash
TOKENIZERS_PARALLELISM=false PYTHONPATH=. .venv/bin/python \
  scripts/entropy_prefix_sweep.py \
  --run_meta outputs/c4_samples_head_200/run_metadata.json \
  --prompt_csv outputs/c4_samples_head_200/hf_generate.csv \
  --prompt_row 0 \
  --device cuda:0 \
  --samples 1 \
  --max_new_tokens 4 \
  --output_dir outputs/smoke_entropy_prefix_sweep_localvenv
```

结果：通过，输出：

```text
outputs/smoke_entropy_prefix_sweep_localvenv/entropy_per_step.csv
outputs/smoke_entropy_prefix_sweep_localvenv/entropy_summary.csv
```

观测 summary：

```text
n=1 mean_ratio=0.555202
n=2 mean_ratio=0.642852
n=3 mean_ratio=0.955156
n=4 mean_ratio=0.985779
n=5 mean_ratio=0.993904
n=6 mean_ratio=0.996595
n=7 mean_ratio=0.996595
```

### 7. 画图 smoke

命令：

```bash
cd plotting
MPLBACKEND=Agg PYTHONPATH=.. ../.venv/bin/python p.py
MPLBACKEND=Agg PYTHONPATH=.. ../.venv/bin/python pic.py
MPLBACKEND=Agg PYTHONPATH=.. ../.venv/bin/python pp.py
```

结果：通过。输出：

```text
plotting/robustness_vs_editrate_retention.png/pdf
plotting/retention_vs_relppl.png/pdf
plotting/entropy_retention_modern.png/pdf
```

运行时仅出现 pandas `FutureWarning`，不影响图生成。

## 后续建议

如果要继续扩大规模复现，建议优先使用以下顺序：

1. 用当前 `.venv` 跑 `REPRODUCIBILITY_zh.md` 中的完整 ByteKGWv6/KGW 生成命令。
2. 再跑 clean TPR/FPR/PPL 评估。
3. 再跑字符攻击和 token 攻击评估。
4. 最后再做图表复现与结果汇总。

若要使用 Qwen2.5-3B 做轻量调试，建议另建一份 metadata，不要覆盖当前 Llama3 主流程 metadata，以免混淆论文复现路径。

