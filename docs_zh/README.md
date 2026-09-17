# 中文文档索引

这里集中放置 `watermark-repro-clean` 的中文说明文档。英文原文仍保留在项目根目录，方便对照。

## 最快运行方式

当前本地可运行目录是 `/home/star/watermark-repro-clean`，虚拟环境也已经放在该目录下：

```bash
cd /home/star/watermark-repro-clean
source .venv/bin/activate
export PYTHONPATH=.
python scripts/smoke_imports.py
```

模型不需要从 Hugging Face 重新下载；当前目录下的 `models/` 使用软链接指向 `/home/star/jf/` 里的本地模型。已经实际跑通的 smoke 命令和结果见 [LOCAL_RUN_STATUS_zh.md](LOCAL_RUN_STATUS_zh.md)。

建议阅读顺序：

1. [METHOD_OVERVIEW_zh.md](METHOD_OVERVIEW_zh.md)：按论文思路介绍方法、实验、本仓库与论文的关系，以及如何使用。
2. [README_zh.md](README_zh.md)：复现包总览。
3. [DATA_PROVENANCE_zh.md](DATA_PROVENANCE_zh.md)：数据来源与代码对应关系。
4. [REPRODUCIBILITY_zh.md](REPRODUCIBILITY_zh.md)：复现命令和流程。
5. [WATERMARK_CODE_MAP_zh.md](WATERMARK_CODE_MAP_zh.md)：水印代码、实验脚本、结果和图表位置。
6. [EXPERIMENT_AUDIT_zh.md](EXPERIMENT_AUDIT_zh.md)：第 4 章实验到代码/数据/状态的逐项映射。
7. [FIGURE_REPRO_STATUS_zh.md](FIGURE_REPRO_STATUS_zh.md)：论文图表复现状态。
8. [SMOKE_TEST_RESULTS_zh.md](SMOKE_TEST_RESULTS_zh.md)：原 clean 包已经实际运行过的小规模检查。
9. [LOCAL_RUN_STATUS_zh.md](LOCAL_RUN_STATUS_zh.md)：当前 `/home/star/watermark-repro-clean` 本地 `.venv`、模型软链接和 smoke 运行记录。
10. [MANIFEST_zh.md](MANIFEST_zh.md)：clean 包包含/排除内容清单。
11. [ATTACK_PAPER_ORIGINAL_README_zh.md](ATTACK_PAPER_ORIGINAL_README_zh.md)：字符扰动攻击历史 README 的中文说明。
