#!/bin/bash

# 测试API服务器限流情况的脚本

# 设置Python路径
PYTHONPATH="/workspace/graph-constrained-reasoning"
export PYTHONPATH

# 加载环境变量
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

# 测试配置
NUM_REQUESTS=20      # 每个模型发送的请求数
NUM_THREADS=5        # 并发线程数
DELAY=0              # 请求间隔（秒）
MODEL_DELAY=2        # 不同模型之间的延迟（秒）

# 输出文件（自动生成时间戳）
OUTPUT_FILE="results/rate_limit_test_$(date +%Y%m%d_%H%M%S).json"

echo "=========================================="
echo "API限流测试"
echo "=========================================="
echo "自动获取模型列表并测试"
echo "每个模型请求数: ${NUM_REQUESTS}"
echo "并发线程数: ${NUM_THREADS}"
echo "请求间隔: ${DELAY}秒"
echo "模型间隔: ${MODEL_DELAY}秒"
echo "输出文件: ${OUTPUT_FILE}"
echo "=========================================="
echo ""

# 运行测试（不指定--models参数，自动获取所有模型）
python src/test_rate_limit.py \
    --num_requests ${NUM_REQUESTS} \
    --num_threads ${NUM_THREADS} \
    --delay ${DELAY} \
    --model_delay ${MODEL_DELAY} \
    --output ${OUTPUT_FILE}

echo ""
echo "测试完成！"
