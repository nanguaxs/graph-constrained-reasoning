#!/usr/bin/env bash
set -euo pipefail

# 设置 PYTHONPATH，确保 Python 能够找到 src 模块
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}"

# Windows 系统上的 Docker 容器访问宿主机 Clash 代理的正确设置
# export HTTP_PROXY="http://host.docker.internal:7897"
# export HTTPS_PROXY="http://host.docker.internal:7897"

# 设置 HF_HOME，将 Hugging Face 数据集下载到当前目录
# 建议开启离线模式，防止代码偷偷联网检查更新
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
# HF_HOME="."
# export HF_HOME

DATA_PATH="./offline_assets/datasets"
# DATA_LIST="RoG-webqsp RoG-cwq"
DATA_LIST="COKG_QA"
SPLIT="test"
INDEX_LEN=2
ATTN_IMP="flash_attention_2"

MODEL_PATH="./offline_assets/models/Qwen3.5-0.8B"
# GCR-Qwen2-0.5B-Instruct
# Qwen3.5-0.8B
MODEL_NAME="$(basename "${MODEL_PATH}")"

K="8" # 3 5 10 20
for DATA in ${DATA_LIST}; do
  for k in ${K}; do
    python workflow/predict_paths_and_answers.py \
      --data_path "${DATA_PATH}" \
      --d "${DATA}" \
      --split "${SPLIT}" \
      --index_path_length "${INDEX_LEN}" \
      --model_name "${MODEL_NAME}" \
      --model_path "${MODEL_PATH}" \
      --k "${k}" \
      --prompt_mode zero-shot \
      --generation_mode beam \
      --attn_implementation "${ATTN_IMP}"
  done
done
