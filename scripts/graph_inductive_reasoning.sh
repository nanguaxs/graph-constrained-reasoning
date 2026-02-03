PYTHONPATH="/workspace/graph-constrained-reasoning"
export PYTHONPATH
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
DATA_PATH="./offline_assets/datasets"
# DATA_LIST="RoG-webqsp RoG-cwq"
DATA_LIST="industry-kg"
SPLIT="train"

MODEL_NAME=gpt-4o
N_THREAD=3  # 降低并发数以避免限流（原值：10）
REQUEST_DELAY=0.5  # 每次请求前的延迟（秒）

# MODEL_NAME=gpt-4o-mini
# N_THREAD=10

for DATA in ${DATA_LIST}; do
  REASONING_PATH="results/GenPaths/${DATA}/GCR-Qwen2-0.5B-Instruct/train/zero-shot-group-beam-k10-index_len2/predictions.jsonl"

  python workflow/predict_final_answer.py --data_path ${DATA_PATH} --d ${DATA} --split ${SPLIT} --model_name ${MODEL_NAME} --reasoning_path ${REASONING_PATH} --add_path True -n ${N_THREAD} --request_delay ${REQUEST_DELAY}
done
