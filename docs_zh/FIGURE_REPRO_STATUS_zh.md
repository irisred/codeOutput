# 图表复现状态

本文档记录论文第 4 章水印相关图表，在 clean 复现包中是否具有对应的实验数据、画图脚本或最终资产。

## 状态总览

| 论文图表 | 作用 | 已包含的源码/数据状态 | 是否冒烟测试 | 备注 |
|---|---|---:|---:|---|
| `thesis/figures/prefix_entropy_retention.png` | 前缀熵保留诊断 | `scripts/entropy_prefix_sweep.py`、`outputs/entropy_prefix_sweep/`、`plotting/pp.py` | 是 | 小规模测试复现了趋势：前缀熵比例快速上升，3 字节附近达到约 0.98。 |
| `thesis/figures/tpr_retention_relppl.png` | TPR retention 与 RelPPL 桶关系 | `plotting/pic.py`、`plotting/tpr_attack_by_bucket_*.csv` | 是 | 可重新生成 `plotting/retention_vs_relppl.png/pdf`，尺寸与论文最终 PNG 不完全一致。 |
| `thesis/figures/robustness_vs_edit_rate.png` | TPR retention 与编辑率关系 | `plotting/p.py`、`plotting/tpr_attack_by_bucket_*.csv` | 是 | 可重新生成 `plotting/robustness_vs_editrate_retention.png/pdf`，尺寸与论文最终 PNG 不完全一致。 |
| `thesis/figures/tab_token_attack.png` | token 攻击表格图 | 相关结果数据在 `outputs/wm_eval/` 及相关 CSV 中 | 部分 | 最终表格 PNG 已包含，但未找到精确表格渲染脚本。 |
| `thesis/figures/tab_char_attack.png` | 字符攻击表格图 | 相关结果数据在 `outputs/wm_eval/` 及相关 CSV 中 | 部分 | 最终表格 PNG 已包含，但未找到精确表格渲染脚本。 |
| `thesis/figures/tab_robustness_stats.png` | 鲁棒性过程统计表 | `scripts/trace_v6_attack_grid.py`、`outputs/trace_v6_attack_grid_bytekgw*.csv` | CSV 诊断已测 | trace/统计代码经过冒烟测试，但未找到精确 PNG 表格渲染脚本。 |
| `thesis/figures/watermark_framework.png` | 方法框架图 | 只有最终 PNG | 否 | 概念图，不是实验图，未找到可编辑源文件。 |
| `thesis/figures/prf_partition.png` | PRF 划分示意图 | 只有最终 PNG | 否 | 概念图，不是实验图，未找到可编辑源文件。 |
| `thesis/figures/tokenization_perturbation_attack.png` | 分词扰动攻击示意图 | 只有最终 PNG | 否 | 概念图，不是实验图，未找到可编辑源文件。 |

## 已实际复现的内容

以下画图命令已成功执行：

```bash
cd plotting
MPLBACKEND=Agg /home/star/jf/python/stega/bin/python p.py
MPLBACKEND=Agg /home/star/jf/python/stega/bin/python pic.py
MPLBACKEND=Agg /home/star/jf/python/stega/bin/python pp.py
```

它们重新生成了：

```text
plotting/robustness_vs_editrate_retention.png/pdf
plotting/retention_vs_relppl.png/pdf
plotting/entropy_retention_modern.png/pdf
```

这些输出与实验方向和视觉风格一致，但不保证与论文 PNG 字节级完全相同。

## 效果检查

小规模检查确认了预期的定性现象：

- clean 文本的 z-score 接近 0。
- `delta=3` 的水印文本与 clean 文本明显分离。
- 攻击诊断显示：编辑率为 0 时无漂移；字符编辑增加时，token/prefix 扰动增强。
- 前缀熵保留率随前缀长度快速上升，支持论文中使用短可见字节前缀类别的动机。

## 比较结果检查

已包含的比较汇总覆盖以下方法：

```text
bytekgw_v6
kgw
dip
unbiased
```

汇总文件位于：

```text
outputs/wm_eval/tpr_by_bucket.csv
outputs/wm_eval/tpr_attack_by_bucket.csv
outputs/wm_eval/tpr_attack_token_by_bucket.csv
```

对已包含的 PPL 桶求平均后，定性比较支持论文中关于“所提方法在编辑攻击下 retention 更稳定”的主张。例如：

```text
Character attack, FPR=0.01:
bytekgw_v6 retention = 0.953
kgw        retention = 0.907
dip        retention = 0.794
unbiased   retention = 0.719

Token attack, FPR=0.01:
bytekgw_v6 retention = 0.957
kgw        retention = 0.902
unbiased   retention = 0.777
dip        retention = 0.760
```

clean、未攻击情况下的 TPR 对比更复杂。低 FPR 下，KGW 在攻击前可能略高于 `bytekgw_v6`；而在编辑攻击后，`bytekgw_v6` 变成最强或接近最强。因此最准确的解读是：

```text
已复现/已包含的比较数据支持鲁棒性 retention 主张，而不是“ByteKGWv6 在所有设置下 clean TPR 都最高”的绝对说法。
```

具体冒烟命令和观测值见 `SMOKE_TEST_RESULTS.md`。
