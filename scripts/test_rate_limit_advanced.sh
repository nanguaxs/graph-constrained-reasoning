#!/bin/bash

# 高级限流测试脚本 - 测试不同并发数下的限流情况

# 设置Python路径
PYTHONPATH="/workspace/graph-constrained-reasoning"
export PYTHONPATH

# 加载环境变量
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

# 测试配置
NUM_REQUESTS=30      # 每个模型发送的请求数
MODEL_DELAY=3        # 不同模型之间的延迟（秒）

# 要测试的并发数列表
THREAD_CONFIGS=(1 3 5 10 15 20)

# 输出目录
OUTPUT_DIR="results/rate_limit_tests_$(date +%Y%m%d_%H%M%S)"
mkdir -p ${OUTPUT_DIR}

echo "=========================================="
echo "高级API限流测试"
echo "=========================================="
echo "测试不同并发数下的限流情况"
echo "并发数配置: ${THREAD_CONFIGS[@]}"
echo "每个模型请求数: ${NUM_REQUESTS}"
echo "输出目录: ${OUTPUT_DIR}"
echo "=========================================="
echo ""

# 测试每个并发数配置
for NUM_THREADS in "${THREAD_CONFIGS[@]}"; do
    echo ""
    echo "=========================================="
    echo "测试并发数: ${NUM_THREADS}"
    echo "=========================================="

    OUTPUT_FILE="${OUTPUT_DIR}/threads_${NUM_THREADS}.json"

    python src/test_rate_limit.py \
        --num_requests ${NUM_REQUESTS} \
        --num_threads ${NUM_THREADS} \
        --delay 0 \
        --model_delay ${MODEL_DELAY} \
        --output ${OUTPUT_FILE}

    echo ""
    echo "并发数 ${NUM_THREADS} 测试完成，结果保存到: ${OUTPUT_FILE}"
    echo ""

    # 不同并发数测试之间等待一段时间
    if [ ${NUM_THREADS} != ${THREAD_CONFIGS[-1]} ]; then
        echo "等待 10 秒后进行下一组测试..."
        sleep 10
    fi
done

echo ""
echo "=========================================="
echo "所有测试完成！"
echo "=========================================="
echo "结果保存在目录: ${OUTPUT_DIR}"
echo ""
echo "查看结果:"
echo "  ls -lh ${OUTPUT_DIR}"
echo ""
