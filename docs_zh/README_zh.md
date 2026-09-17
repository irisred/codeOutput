# 前缀载体水印复现包

这是从 `/home/star/jf/watermark` 清理出来的、可运行的水印项目子集。

本包主要面向论文第 4 章使用的文本水印方法，重点包括：

- 前缀载体水印；
- 可见字节前缀类别；
- 由指纹驱动的 PRF 划分；
- 在 token 级和字符级编辑扰动下的鲁棒检测。

当前最接近论文方法的实现是 `ByteKGWv6`。

## 重要路径

```text
MarkLLM/watermark/bytekgwV6/      # 当前方法实现
MarkLLM/watermark/kgw/            # KGW 基线
MarkLLM/watermark/dip/            # DiP 基线
MarkLLM/watermark/unbiased/       # Unbiased 基线
config/ByteKGWv6.json             # 当前方法配置
scripts/                          # 生成、评估、攻击和诊断脚本
outputs/                          # 已选择保留的生成数据和统计结果
plotting/                         # 已定位的画图脚本及 CSV 输入
thesis/                           # 第 4 章 tex 与最终论文图
WATERMARK_CODE_MAP.md             # 代码、实验、图表映射
REPRODUCIBILITY.md                # 可运行命令和复现流程
DATA_PROVENANCE.md                # 数据来源与代码对应关系
EXPERIMENT_AUDIT.md               # 论文实验到代码/数据/状态的逐项审计
FIGURE_REPRO_STATUS.md            # 图表复现状态
SMOKE_TEST_RESULTS.md             # 已实际运行的小规模检查结果
ATTACK_PAPER_ORIGINAL_README.md   # 字符扰动攻击代码的历史 README
```

## 环境

当前本地环境：

```bash
cd /home/star/watermark-repro-clean
source .venv/bin/activate
export PYTHONPATH=.
```

这个 `.venv` 已经安装在仓库目录下，并通过了导入、生成、检测、trace、entropy 和 plotting 的 smoke 检查。实际运行记录见 `docs_zh/LOCAL_RUN_STATUS_zh.md`。

本复现包不把模型权重纳入 git。当前本地设置是在 `models/` 下放软链接，指向 `/home/star/jf/` 中已有模型：

```text
models/Meta-Llama-3-8B-Instruct -> /home/star/jf/Meta-Llama-3-8B-Instruct
models/Qwen2.5-3B -> /home/star/jf/Qwen2.5-3B
```

当前 `outputs/c4_samples_head_200/run_metadata.json` 使用的是 `models/Meta-Llama-3-8B-Instruct`。如果模型路径不同，可以修改 metadata 中的 `model` 字段，或复制一份 metadata 后改其中的模型路径。

## 包含内容

当前方法数据：

```text
outputs/v6_vs_kgw_gen_prfnew_mp/bytekgw_v6_delta*.csv
outputs/wm_eval/bytekgw_v6/
```

基线数据：

```text
outputs/v6_vs_kgw_gen_prfnew_mp/kgw_delta*.csv
outputs/wm_eval/kgw/
outputs/wm_eval/dip/
outputs/wm_eval/unbiased/
```

本包中包含完整比较结果行的基线主要是 `KGW`、`DiP` 和 `Unbiased`。其他 MarkLLM 方法主要作为源码和配置基线保留，但不包含完整的第 4 章实验结果表。

`ATTACK_PAPER_ORIGINAL_README.md` 只作为字符扰动攻击代码的历史背景保留，不是本包的主要复现指南。

诊断数据：

```text
outputs/trace_v6_attack_grid_bytekgw*.csv
sweep_v6_prf_r002/
traces_v6_process_r002/
```

画图输入和脚本：

```text
plotting/p.py
plotting/pic.py
plotting/pp.py
plotting/tpr_attack_by_bucket_*.csv
```

## 快速检查

在本目录下运行：

```bash
cd /home/star/watermark-repro-clean
source .venv/bin/activate
export PYTHONPATH=.

python scripts/smoke_imports.py
python -m py_compile \
  MarkLLM/watermark/bytekgwV6/watermark.py \
  scripts/generate_clean_v6_kgw.py \
  scripts/analyze_v6_kgw_tpr.py \
  scripts/trace_v6_attack_grid.py
```

## 建议阅读顺序

1. `DATA_PROVENANCE.md`
2. `REPRODUCIBILITY.md`
3. `WATERMARK_CODE_MAP.md`
4. `EXPERIMENT_AUDIT.md`
5. `FIGURE_REPRO_STATUS.md`
6. `SMOKE_TEST_RESULTS.md`
