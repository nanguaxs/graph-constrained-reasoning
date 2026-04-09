# vLLM GCR

这个目录提供一套独立于现有 Hugging Face 推理链路的 `vLLM` 图约束解码实现。

## 包含内容

- `logits_processor.py`
  把仓库现有的 trie 图约束迁移为 vLLM custom logits processor。
- `model.py`
  封装离线 `vllm.LLM(...)` 推理，支持按请求携带不同的图约束。
- `predict_paths_and_answers.py`
  对应 `workflow/predict_paths_and_answers.py` 的 vLLM 版本。

## 前提

- `vLLM` 当前要求 Linux 环境；Windows 建议用 WSL。
- 模型 tokenizer 必须已经把 `<PATH>` 和 `</PATH>` 训练成单独 token。
  `workflow/finetune_kg_specialized_llm.py` 就是按这个方式做的。

## 安装建议

建议在新的 Linux / WSL Python 环境中安装：

```bash
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu129
```

如果你使用 `uv`，也可以直接按 vLLM 官方文档安装。

## 离线批量推理

在仓库根目录执行：

```bash
python -m vllm_gcr.predict_paths_and_answers \
  --model_path your-model-or-local-path \
  --model_name qwen-gcr-vllm \
  --generation_mode greedy \
  --batch_size 8 \
  --split "test[:100]"
```

如果你想采样多个候选路径：

```bash
python -m vllm_gcr.predict_paths_and_answers \
  --model_path your-model-or-local-path \
  --generation_mode sampling \
  --k 5 \
  --temperature 0.8 \
  --top_p 0.95
```

## 用 `vllm serve` 部署

`GraphConstraintLogitsProcessor` 既能离线用，也能作为 `vllm serve` 的自定义 logits processor 加载：

请在仓库根目录启动，或者先把仓库根目录加入 `PYTHONPATH`，这样 `vLLM` 才能导入 `vllm_gcr.logits_processor`。

```bash
vllm serve your-model-or-local-path \
  --trust-remote-code \
  --logits_processors vllm_gcr.logits_processor:GraphConstraintLogitsProcessor
```

在线请求时，把图约束通过 `vllm_xargs` 传入。`vllm_xargs` 会映射到 `SamplingParams.extra_args`：

```json
{
  "gcr_trie_sequences": [[32000, 101, 102], [32000, 205, 206]],
  "gcr_start_token_id": 32000,
  "gcr_end_token_id": 32001,
  "gcr_eos_token_id": 151643,
  "gcr_enable_constrained_by_default": false
}
```

其中：

- `gcr_trie_sequences` 是可行路径对应的 token id 序列列表。
- `gcr_start_token_id` / `gcr_end_token_id` 对应 `<PATH>` 和 `</PATH>`。
- `gcr_eos_token_id` 用于在检测到 `</PATH>` 后强制结束生成。

## 设计说明

- 这版实现优先保证和现有仓库的 trie 约束逻辑一致。
- 当前脚本只支持 `greedy` 和 `sampling`，没有接 beam/group-beam。
- `vLLM` 自身已经会做动态批处理，所以这里没有继续沿用 `multiprocessing.Pool`。
