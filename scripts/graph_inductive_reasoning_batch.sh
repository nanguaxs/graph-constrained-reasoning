#!/bin/bash

# 设置PYTHONPATH，确保Python能够找到src模块
PYTHONPATH="/workspace/graph-constrained-reasoning"
export PYTHONPATH

# 设置HF离线模式
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

DATA_PATH="./offline_assets/datasets"
DATA_LIST="kg_qa_dataset"
SPLIT="train"
INDEX_LEN=2

# 定义模型路径（与graph_constrained_decoding_multi.sh保持一致）
MODEL_PATHS=(
    "./offline_assets/models/GCR-Qwen2-0.5B-Instruct"
    #"./offline_assets/models/另一个模型路径"
    # "./offline_assets/models/第三个模型路径"
)

# 定义k值（与graph_constrained_decoding_multi.sh保持一致）
K_VALUES=(3 5 10 )

# 定义推理模型配置
MODEL_NAME=gpt-4o
N_THREAD=10

# MODEL_NAME=gpt-4o-mini
# N_THREAD=10

echo "=========================================="
echo "Batch Inductive Reasoning Script"
echo "=========================================="

# 遍历所有模型
for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    MODEL_NAME_BASE=$(basename "$MODEL_PATH")
    echo "=========================================="
    echo "Processing model: $MODEL_NAME_BASE"
    echo "=========================================="

    # 遍历所有数据集
    for DATA in ${DATA_LIST}; do
        echo "Dataset: $DATA"

        # 遍历所有k值
        for k in "${K_VALUES[@]}"; do
            echo "Running inductive reasoning with k=$k..."

            # 构建推理路径（与graph_constrained_decoding_multi.sh的输出路径对应）
            REASONING_PATH="results/GenPaths/${DATA}/${MODEL_NAME_BASE}/${SPLIT}/zero-shot-group-beam-k${k}-index_len${INDEX_LEN}/predictions.jsonl"

            # 检查推理路径是否存在
            if [ ! -f "$REASONING_PATH" ]; then
                echo "Warning: Reasoning path not found: $REASONING_PATH"
                echo "Skipping k=$k..."
                echo "------------------------------------------"
                continue
            fi

            echo "Using reasoning path: $REASONING_PATH"

            # 运行最终答案预测
            python workflow/predict_final_answer.py \
                --data_path ${DATA_PATH} \
                --d ${DATA} \
                --split ${SPLIT} \
                --model_name ${MODEL_NAME} \
                --reasoning_path ${REASONING_PATH} \
                --add_path True \
                -n ${N_THREAD}

            echo "Completed k=$k"
            echo "------------------------------------------"
        done
    done
done

echo "=========================================="
echo "All inductive reasoning tasks completed!"
echo "=========================================="
