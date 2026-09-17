# 字符级扰动破坏 LLM 水印

该历史 README 提供了缩小版实验，用于验证论文中的两个主要结论：

1. 在相同编辑率下，字符级扰动比 token 级扰动具有更强的水印移除效果；
2. 基于遗传算法的优化可以利用参考检测器的指导，进一步提升移除攻击性能。

注意：在本 clean 复现包中，它主要作为字符扰动攻击代码的历史背景保留；主复现流程请优先看 `README.md` 和 `REPRODUCIBILITY.md`。

## 硬件依赖

建议使用至少 8 个 CPU 核心、16GB RAM 的普通桌面机器。GPU，尤其是支持 CUDA 的 NVIDIA GPU，强烈推荐用于加速批量评估和参考检测器训练。

## 软件依赖

Python 3.9。其他依赖见 `requirements.txt`。

```bash
pip install -r requirements.txt
```

## 数据集与模型

使用 C4 数据集作为 prompts 来源，用于查询目标 LLM 并生成带水印文本。C4 数据集可在 Hugging Face 获取：

```text
https://huggingface.co/datasets/allenai/c4
```

建议通过 git 下载：

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/allenai/c4
cd c4
git lfs pull --include "realnewslike/*"
```

参考检测器从 BERT 微调而来。句子级攻击使用 DIPPER 作为改写模型。相关模型权重均可从 Hugging Face 模型仓库获取。

## 实验流程

- 基线移除评估：运行随机攻击脚本，验证字符级扰动在相同编辑率下比 token 级扰动更能移除水印。对应论文表 II。
- 有指导移除评估：先在带水印数据上训练参考检测器，再运行 Best-of-N 和遗传算法攻击，展示其更强攻击性能。

## 基线移除评估

### 准备

实验需要 C4 数据集。建议存放在：

```text
../../dataset/c4/realnewslike
```

带水印文本由 OPT-1.3B 生成。模型权重不需要单独下载，脚本运行时会自动获取。

生成数据示例：

```bash
python collect_wm_text.py --wm_name "KGW" \
--dataset_name "../../dataset/c4/realnewslike" \
--model_name "facebook/opt-1.3b" --device 0 \
--file_num 50 --file_data_num 100
```

该命令会在 GPU 0 上使用 `facebook/opt-1.3b` 与 KGW 水印生成 `file_num * file_data_num = 5000` 条样本。若要生成其他水印方案的数据，修改 `--wm_name` 即可。建议至少生成 5000 条样本以支持参考检测器训练。

### 执行

```bash
python test_rand_sh.py \
--llm_name "facebook/opt-1.3b" \
--wm_name_list "['KGW']" \
--atk_style_list "['token','char']" \
--max_edit_rate_list "[0.1, 0.5]" \
--do_flag "True" --atk_times_list "[1]" \
--max_token_num_list "[100]"
```

该示例中，编辑率设为 0.1 和 0.5，带水印文本长度固定为 100 tokens，与论文设置一致。因为这里是随机策略，`atk_times_list` 设为 1。若要评估其他水印方案，将其名称加入 `wm_name_list`。

结果会保存为：

```text
attack_log/Rand
saved_attk_data
```

如果将 `do_flag` 设为 `False`，则结果会直接打印到终端。输出应展示：在相同编辑率下，字符级扰动通常具有更高的水印分数下降率 WDR 和攻击成功率 ASR。

## 有指导移除评估

### 准备

训练参考检测器：

```bash
python train_ref_detector.py  --device 0 \
--wm_name "KGW" --num_epochs 15 \
--rand_char_rate 0.15 --rand_times 9 \
--llm_name "facebook/opt-1.3b" --ths 4
```

该示例使用 KGW 水印。每个样本增强 9 次，编辑率为 0.15。`ths` 是用于评估参考检测器的检测阈值。训练 epoch 数为 15。由于数据增强和训练都较耗时，原 README 提到在 `saved_data` 和 `saved_model` 中提供了预处理数据和预训练参考检测器，便于 artifact evaluation。

### 执行

Best-of-N：

```bash
python test_rand_sh.py \
--llm_name "facebook/opt-1.3b" \
--wm_name_list "['KGW']" \
--atk_style_list "['token','char']" \
--data_aug_list "[9]" --max_token_num_list "[100]" \
--max_edit_rate_list "[0.1]" \
--do_flag "True" --atk_times_list "[10]"
```

`atk_times_list` 指定 Best-of-N 中的 `N`，即每个输入采样多少个扰动候选。`data_aug_list` 用于选择参考检测器，因为训练参考检测器时 `rand_times` 设为 9，所以这里也设为 9。

Sand：

```bash
python test_rand_sh.py \
    --llm_name "facebook/opt-1.3b" \
    --wm_name_list "['KGW']" \
    --atk_style_list "['sand_token','sand_char']" \
    --data_aug_list "[9]" --do_flag "True" \
    --max_edit_rate_list "[0.1]" \
    --atk_times_list "[1]" \
    --max_token_num_list "[100]"
```

GA：

```bash
python test_ga_sh.py \
--llm_name "facebook/opt-1.3b" \
--wm_name "KGW" --atk_style "char" \
--data_aug 9 --do_flag "True" \
--num_generations 15 --max_edit_rate 0.13 \
--max_token_num_list "[100]"
```

三类方法的结果默认保存为：

```text
attack_log/Rand
attack_log/GA
saved_attk_data
```

将 `do_flag` 设为 `False` 可直接在终端打印结果。

## 使用到的项目

```text
MarkLLM: https://github.com/THU-BPM/MarkLLM
TextAttack: https://github.com/QData/TextAttack
```
