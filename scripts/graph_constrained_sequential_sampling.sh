#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}"

export HF_HUB_OFFLINE=1

DATA_PATH="./offline_assets/datasets"
DATA_LIST="onedata"
SPLIT="test"
INDEX_LEN=2
ATTN_IMP="flash_attention_2"

MODEL_PATH="./offline_assets/models/Qwen3.5-0.8B"
MODEL_NAME="$(basename "${MODEL_PATH}")"

K="8"
for DATA in ${DATA_LIST}; do
  for k in ${K}; do
    python workflow/predict_paths_and_answers_sequential_sampling.py \
      --data_path "${DATA_PATH}" \
      --d "${DATA}" \
      --split "${SPLIT}" \
      --index_path_length "${INDEX_LEN}" \
      --model_name "${MODEL_NAME}" \
      --model_path "${MODEL_PATH}" \
      --k "${k}" \
      --prompt_mode zero-shot \
      --generation_mode sampling \
      --attn_implementation "${ATTN_IMP}"
  done
done
