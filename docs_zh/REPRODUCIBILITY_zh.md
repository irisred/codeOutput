# 复现指南

所有命令默认在以下环境中执行：

```bash
cd /home/star/watermark-repro-clean
source .venv/bin/activate
export PYTHONPATH=.
```

当前目录下的 `.venv` 已经安装依赖并通过 smoke 检查。模型通过 `models/` 软链接指向 `/home/star/jf/` 下已有权重，不需要重新从 Hugging Face 下载。

## 1. 导入冒烟测试

```bash
python scripts/smoke_imports.py
```

预期结果：脚本会为 ByteKGWv6、KGW、DiP、Unbiased 和 Charm 模块打印 `OK` 行。

已经在本 clean 仓库中实际执行过的小命令记录见：

```text
SMOKE_TEST_RESULTS.md
```

## 2. 重新生成 ByteKGWv6 / KGW 样本

下面命令最接近包内 `outputs/v6_vs_kgw_gen_prfnew_mp/` 数据：

```bash
TOKENIZERS_PARALLELISM=false python scripts/generate_clean_v6_kgw.py \
  --run_meta outputs/c4_samples_head_200/run_metadata.json \
  --v6_config config/ByteKGWv6.json \
  --kgw_config config/KGW.json \
  --output_dir outputs/repro_v6_vs_kgw \
  --deltas 1,2,3,4,5 \
  --n_prompts 200 \
  --devices cuda:0
```

注意：

- 模型路径从 `outputs/c4_samples_head_200/run_metadata.json` 读取。
- 当前本地 metadata 中的模型路径是 `models/Meta-Llama-3-8B-Instruct`。
- 如果模型在其他位置，可以修改 metadata 中的 `model` 字段，或复制一份 metadata 再修改。
- 生成文本可能随硬件和库版本变化。

快速冒烟运行：

```bash
TOKENIZERS_PARALLELISM=false python scripts/generate_clean_v6_kgw.py \
  --run_meta outputs/c4_samples_head_200/run_metadata.json \
  --v6_config config/ByteKGWv6.json \
  --kgw_config config/KGW.json \
  --output_dir outputs/smoke_v6_vs_kgw \
  --deltas 1 \
  --n_prompts 2 \
  --devices cuda:0
```

## 3. 评估 clean TPR/FPR/PPL

使用包内已生成数据：

```bash
TOKENIZERS_PARALLELISM=false python scripts/analyze_v6_kgw_tpr.py \
  --run_meta outputs/c4_samples_head_200/run_metadata.json \
  --v6_config config/ByteKGWv6.json \
  --kgw_config config/KGW.json \
  --input_dir outputs/v6_vs_kgw_gen_prfnew_mp \
  --deltas 1,2,3,4,5 \
  --fprs 0.01,0.05,0.1,0.2 \
  --device cuda:0
```

不计算 PPL 的快速结构检查：

```bash
TOKENIZERS_PARALLELISM=false python scripts/analyze_v6_kgw_tpr.py \
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

## 4. 评估字符攻击鲁棒性

```bash
TOKENIZERS_PARALLELISM=false python scripts/analyze_v6_kgw_attack.py \
  --run_meta outputs/c4_samples_head_200/run_metadata.json \
  --v6_config config/ByteKGWv6.json \
  --kgw_config config/KGW.json \
  --input_dir outputs/v6_vs_kgw_gen_prfnew_mp \
  --deltas 1,2,3,4,5 \
  --fprs 0.01,0.05,0.1,0.2 \
  --attack_ratio 0.02 \
  --n_clean 200 \
  --n_samples 200 \
  --device cuda:0 \
  --skip_ppl
```

## 5. 机制诊断

### Prefix/PRF trace grid

```bash
TOKENIZERS_PARALLELISM=false python scripts/trace_v6_attack_grid.py \
  --run_meta outputs/c4_samples_head_200/run_metadata.json \
  --v6_config config/ByteKGWv6.json \
  --input_csv outputs/v6_vs_kgw_gen_prfnew_mp/bytekgw_v6_delta2.0.csv \
  --attack_ratios 0,0.01,0.02,0.05,0.10 \
  --attack_seed 0 \
  --indices all \
  --max_tokens 200 \
  --stride 1 \
  --device cuda:0 \
  --out_csv outputs/repro_trace_v6_attack_grid_bytekgw.csv
```

### 混合字符攻击 trace

```bash
TOKENIZERS_PARALLELISM=false python scripts/trace_v6_attack_grid_mixchar.py \
  --run_meta outputs/c4_samples_head_200/run_metadata.json \
  --v6_config config/ByteKGWv6.json \
  --input_csv outputs/v6_vs_kgw_gen_prfnew_mp/bytekgw_v6_delta2.0.csv \
  --attack_ratios 0,0.01,0.02,0.05,0.10 \
  --attack_seed 0 \
  --indices all \
  --max_tokens 200 \
  --stride 1 \
  --device cuda:0 \
  --out_csv outputs/repro_trace_v6_attack_grid_bytekgw_mixchar.csv
```

## 6. 熵 / 前缀保留率诊断

```bash
TOKENIZERS_PARALLELISM=false python scripts/entropy_prefix_sweep.py \
  --run_meta outputs/c4_samples_head_200/run_metadata.json \
  --prompt_csv outputs/c4_samples_head_200/hf_generate.csv \
  --prompt_row 0 \
  --device cuda:0 \
  --output_dir outputs/repro_entropy_prefix_sweep
```

## 7. 画图

已定位的画图脚本位于 `plotting/`。

```bash
cd plotting
python p.py
python pic.py
python pp.py
```

输出：

```text
robustness_vs_editrate_retention.png/pdf
retention_vs_relppl.png/pdf
entropy_retention_modern.png/pdf
```

注意：论文最终图名略有不同：

```text
thesis/figures/robustness_vs_edit_rate.png
thesis/figures/tpr_retention_relppl.png
thesis/figures/prefix_entropy_retention.png
```

这些画图脚本是目前定位到的最接近的可复现来源，但不保证与论文 PNG 字节级完全一致。

逐图复现状态见：

```text
FIGURE_REPRO_STATUS.md
```

## 8. 论文材料

水印章节：

```text
thesis/chapters/ch04_watermark.tex
```

最终图资源：

```text
thesis/figures/
```

## 已知注意事项

- 不包含模型权重。
- 部分脚本依赖 `run_metadata.json` 中的本地模型路径。
- 双 GPU 辅助脚本 `scripts/generate_v6_vs_kgw_prfnew_2gpus.py` 被保留，但更推荐先使用单进程的 `scripts/generate_clean_v6_kgw.py`。
- 部分最终论文表格 PNG 和框架图只作为最终资产保留，未找到精确可编辑/画图源码。
- bit-for-bit 生成复现可能受 PyTorch、Transformers 和 CUDA 版本影响。
