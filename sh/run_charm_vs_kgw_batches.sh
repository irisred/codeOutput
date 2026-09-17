#!/usr/bin/env bash

set -euo pipefail

# 切到仓库根目录（假设本脚本位于 sh/ 子目录中）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="/home/star/jf/python/stega/bin/python"

# 确保 Python 能找到本仓库里的模块
export PYTHONPATH="${REPO_ROOT}"

MODEL="../Meta-Llama-3-8B"
PROMPTS="data/prompts_c4.txt"
N_PROMPTS=10
SAMPLES=1000
MAX_NEW=128
DEVICES="cuda:0,cuda:1"

echo "[1/3] plain + KGW (entropy_threshold=0.0, ${SAMPLES} samples/combo) ..."
"${PYTHON_BIN}" scripts/charm_vs_kgw_ppl_sweep.py \
  --model "${MODEL}" \
  --prompts "${PROMPTS}" \
  --n-prompts "${N_PROMPTS}" \
  --samples-per-combo "${SAMPLES}" \
  --max-new-tokens "${MAX_NEW}" \
  --devices "${DEVICES}" \
  --method-types plain,kgw \
  --charm-entropy-threshold 0.0 \
  --local-files-only \
  --output charm_vs_kgw_ppl_plain_kgw_s${SAMPLES}.csv

echo "[2/3] Charm only (entropy_threshold=0.5, ${SAMPLES} samples/combo) ..."
"${PYTHON_BIN}" scripts/charm_vs_kgw_ppl_sweep.py \
  --model "${MODEL}" \
  --prompts "${PROMPTS}" \
  --n-prompts "${N_PROMPTS}" \
  --samples-per-combo "${SAMPLES}" \
  --max-new-tokens "${MAX_NEW}" \
  --devices "${DEVICES}" \
  --method-types charm \
  --charm-entropy-threshold 0.5 \
  --local-files-only \
  --output charm_vs_kgw_ppl_charm_ent05_s${SAMPLES}.csv

echo "[3/3] Charm only (entropy_threshold=0.7, ${SAMPLES} samples/combo) ..."
"${PYTHON_BIN}" scripts/charm_vs_kgw_ppl_sweep.py \
  --model "${MODEL}" \
  --prompts "${PROMPTS}" \
  --n-prompts "${N_PROMPTS}" \
  --samples-per-combo "${SAMPLES}" \
  --max-new-tokens "${MAX_NEW}" \
  --devices "${DEVICES}" \
  --method-types charm \
  --charm-entropy-threshold 0.7 \
  --local-files-only \
  --output charm_vs_kgw_ppl_charm_ent07_s${SAMPLES}.csv

echo "All batches done."
