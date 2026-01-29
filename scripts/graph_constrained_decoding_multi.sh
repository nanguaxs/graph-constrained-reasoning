#!/bin/bash

# 指定使用 GPU 0
export CUDA_VISIBLE_DEVICES=0

# 设置PYTHONPATH，确保Python能够找到src模块
PYTHONPATH="/workspace/graph-constrained-reasoning"
export PYTHONPATH

# Windows系统上的Docker容器访问宿主机Clash代理的正确设置
# export HTTP_PROXY="http://host.docker.internal:7897"
# export HTTPS_PROXY="http://host.docker.internal:7897"

# 设置HF_HOME，将Hugging Face数据集下载到当前目录
# 建议开启离线模式，防止代码偷偷联网检查更新
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
# HF_HOME="."
# export HF_HOME

DATA_PATH="./offline_assets/datasets"
# DATA_LIST="RoG-webqsp RoG-cwq"
DATA_LIST="kg_qa_dataset"
SPLIT="train"
INDEX_LEN=2
ATTN_IMP=flash_attention_2

# 定义多个模型路径（可以根据需要添加更多模型）
MODEL_PATHS=(
    "./offline_assets/models/GCR-Qwen2-0.5B-Instruct"
    #"./offline_assets/models/另一个模型路径"
    # "./offline_assets/models/第三个模型路径"
)

# 定义多个 k 值
K_VALUES=(3 5 10 )

# 遍历所有模型
for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    MODEL_NAME=$(basename "$MODEL_PATH")
    echo "=========================================="
    echo "Testing model: $MODEL_NAME"
    echo "=========================================="

    # 遍历所有数据集
    for DATA in ${DATA_LIST}; do
        echo "Dataset: $DATA"

        # 遍历所有 k 值
        for k in "${K_VALUES[@]}"; do
            echo "Running with k=$k..."
            python workflow/predict_paths_and_answers.py \
                --data_path ${DATA_PATH} \
                --d ${DATA} \
                --split ${SPLIT} \
                --index_path_length ${INDEX_LEN} \
                --model_name ${MODEL_NAME} \
                --model_path ${MODEL_PATH} \
                --k ${k} \
                --prompt_mode zero-shot \
                --generation_mode group-beam \
                --attn_implementation ${ATTN_IMP}

            echo "Completed k=$k"
            echo "------------------------------------------"
        done
    done
done

echo "=========================================="
echo "All tests completed!"
echo "=========================================="
