# finetune 训练全流程详解

本文档基于当前仓库代码整理，重点解释 `finetune/` 目录下的图约束路径生成微调流程，并补充 `workflow/finetune_kg_specialized_llm.py` 这条标准 TRL SFT 训练线。目标是把一次大语言模型微调从数据、prompt、tokenizer、模型加载、LoRA、生成、奖励、loss、反向传播、保存、合并到推理使用的每个环节讲清楚。

## 1. 代码入口总览

当前仓库里和微调直接相关的代码有两组。

第一组是 `finetune/` 下的新实验代码，包含两种训练方式：

| 文件 | 作用 |
| --- | --- |
| `finetune/train.py` | GRPO 图约束强化学习微调入口 |
| `finetune/grpo_trainer.py` | 自定义 GRPO 训练循环、rollout、loss |
| `finetune/reward.py` | 路径奖励函数，包括答案命中、路径匹配、绕路惩罚、语义相似度 |
| `finetune/sft_train.py` | 自定义 SFT 监督微调入口 |
| `finetune/sft_trainer.py` | 自定义 SFT 训练循环 |
| `finetune/dataset.py` | GRPO 和 SFT 两套数据集封装 |
| `finetune/config.py` | GRPOConfig 和 SFTConfig |
| `finetune/merge_adapter.py` | LoRA adapter 合并到基座模型 |
| `finetune/logging_utils.py` | finetune 模块日志配置 |

第二组是 `workflow/finetune_kg_specialized_llm.py`，配合 `scripts/train_kg_specialized_llm.sh` 和 `accelerate_configs/` 使用。这是一条更标准的 TRL SFT 训练线，支持 `accelerate`、DeepSpeed ZeRO-3、全量微调或 PEFT/LoRA。

推荐理解顺序：

1. 先读 `finetune/config.py`，了解模型、数据、LoRA、生成、奖励、训练超参数。
2. 再读 `finetune/dataset.py`，了解样本怎样变成 prompt、ground paths 和 trie。
3. 对 GRPO，读 `finetune/train.py`、`finetune/grpo_trainer.py`、`finetune/reward.py`。
4. 对 SFT，读 `finetune/sft_train.py`、`finetune/sft_trainer.py`。
5. 最后读 `finetune/merge_adapter.py` 和推理代码，了解训练产物怎么使用。

## 2. 训练目标

这个项目不是普通聊天指令微调，而是针对知识图谱问答中的“推理路径生成”做微调。模型输入一个问题和问题中的主题实体，需要生成知识图谱中的合法路径，例如：

```text
<PATH>实体A -> 关系1 -> 实体B -> 关系2 -> 实体C</PATH>
```

其中：

| 概念 | 含义 |
| --- | --- |
| `q_entity` | 问题中的起点实体，也叫 topic entities |
| `a_entity` | 答案实体 |
| `graph` | 当前问题关联的子图，格式通常是三元组列表 |
| `ground_paths` | 从 `q_entity` 到 `a_entity` 的真值路径 |
| `trie` | 当前问题可生成路径的 token 前缀树，用于图约束解码 |
| `<PATH>` / `</PATH>` | 路径边界 special tokens |

训练的直接目标是让模型更倾向生成：

1. 终点能命中答案实体的路径。
2. 与真值路径结构一致或接近的路径。
3. hop 数不过长、不绕圈、不重复访问实体的路径。
4. 在语义上与真值路径相近的路径。
5. 在图约束解码下可被知识图谱候选路径集合接受的路径。

## 3. 数据格式与数据来源

### 3.1 `finetune/` 数据字段

`finetune/dataset.py` 期望数据样本至少包含以下字段：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `id` | string | 样本 ID |
| `question` | string | 问题文本 |
| `q_entity` | list[string] | 起点实体列表 |
| `a_entity` | list[string] 或 string | 答案实体列表 |
| `graph` | list[list] | 子图三元组列表，每个元素是 `[head, relation, tail]` |
| `paths` | 可选 | 预先构建好的候选路径集合 |
| `choices` | 可选 | 选择题选项 |

如果样本没有 `paths`，代码会从 `graph` 出发，通过 DFS 动态生成候选路径。

### 3.2 本地数据注意点

`GRPOConfig` 和 `SFTConfig` 默认：

```python
data_path = "offline_assets/datasets/COKG_QA"
train_split = "train"
```

`PathGenerationDataset` 的加载逻辑是：

```python
if data_path.endswith(".json"):
    load_dataset("json", data_files=data_path, split=split)
else:
    load_dataset(data_path, split=split)
```

因此训练前需要确认 `data_path` 能被 HuggingFace `datasets.load_dataset` 正确识别，并且存在 `train` split。当前工作区里 `offline_assets/datasets/COKG_QA` 下看到的是 `test.jsonl`，如果没有 `train` split，直接用默认配置训练会失败，需要补充训练集或把 `train_split` 改成实际存在的 split。

### 3.3 `workflow/` 数据字段

`workflow/finetune_kg_specialized_llm.py` 使用的是 `datasets.load_from_disk(data_path)`，入口参数叫 `data_path_list`，例如脚本中：

```bash
DATASET_LIST="data/shortest_path_index/RoG-webqsp/train data/shortest_path_index/RoG-cwq/train"
```

它期望字段包括：

| 字段 | 说明 |
| --- | --- |
| `question` | 问题 |
| `q_entity` | 主题实体 |
| `a_entity` | 答案实体 |
| `ground_truth_paths` | 真值路径，路径本身是三元组列表 |

这一条 workflow 训练线不使用每个样本的 trie 做 rollout，而是把每条 `ground_truth_paths` 展开成标准 SFT 文本样本。

## 4. 图和路径预处理

### 4.1 图构建

图构建函数在 `src/utils/graph_utils.py`：

```python
def build_graph(graph: list, undirected=False):
    if undirected:
        G = nx.Graph()
    else:
        G = nx.DiGraph()
    for triplet in graph:
        h, r, t = triplet
        G.add_edge(h.strip(), t.strip(), relation=r.strip())
    return G
```

默认 `undirected=False`，也就是训练时使用有向图。每条三元组 `[h, r, t]` 会变成 NetworkX 图中的一条边，边属性 `relation=r`。

### 4.2 候选路径生成

候选路径由 `dfs(graph, start_node_list, max_length)` 生成，`max_length` 对应配置中的 `index_path_length`。

逻辑是：

1. 从每个 `q_entity` 出发。
2. 沿图的出边 DFS。
3. 收集长度不超过 `index_path_length` 的所有路径。
4. 每条路径由三元组序列组成。

例如一条 2-hop 路径内部格式是：

```python
[
    ("实体A", "关系1", "实体B"),
    ("实体B", "关系2", "实体C"),
]
```

会被 `src/utils/utils.py` 中的 `path_to_string` 转成：

```text
实体A -> 关系1 -> 实体B -> 关系2 -> 实体C
```

### 4.3 真值路径生成

真值路径来自 `get_truth_paths(q_entity, a_entity, graph)`：

1. 遍历每个起点实体 `h`。
2. 遍历每个答案实体 `t`。
3. 在图中找 `h` 到 `t` 的所有最短路径。
4. 把节点路径还原成三元组路径。

因此 `ground_paths` 是“从问题实体到答案实体的最短路径集合”。GRPO 奖励函数和 SFT 监督标签都依赖它。

### 4.4 Trie 构建

`ChinesePathGenerationWithAnswerPromptBuilder` 继承自 `JointReasoningPromptBuilder`。它的 `get_graph_index` 会把候选路径转成带标签的文本：

```text
<PATH>实体A -> 关系1 -> 实体B</PATH>
```

然后用 tokenizer 分词：

```python
tokenized_paths = tokenizer(paths_list_str, padding=False, add_special_tokens=False).input_ids
trie = MarisaTrie(tokenized_paths, max_token_id=len(tokenizer) + 1)
```

这个 trie 的作用是：生成时只允许模型输出“某条候选路径的合法下一个 token”。这就是图约束解码的核心。

如果候选路径为空，`get_graph_index` 返回 `None`。GRPO 数据集会跳过这个样本：

```python
if trie is None:
    return None
```

## 5. Prompt 构造

### 5.1 原始 prompt

`finetune/dataset.py` 使用：

```python
ChinesePathGenerationWithAnswerPromptBuilder(
    tokenizer,
    "zero-shot",
    undirected=undirected,
    index_path_length=index_path_length,
    add_rule=False,
)
```

这个 prompt builder 的 zero-shot prompt 语义是：

```text
推理路径是知识图谱中连接问题主题实体与答案实体的三元组序列。
给定一个问题，请从主题实体出发，在知识图谱中生成若干推理路径。
实体和关系之间用 -> 连接。

# 问题:
{question}
# 主题实体:
{entities}
<PATH>
```

注意：prompt 末尾直接带 `<PATH>`。这样模型生成的第一个内容就是路径正文，而不是先自由输出一段解释。

### 5.2 Chat template 适配

Qwen、Llama 这类 chat model 通常需要特殊对话模板。GRPO 和 SFT 都有类似逻辑：

```python
if tokenizer.chat_template:
    if query.endswith("<PATH>"):
        user_content = query[:-len("<PATH>")]
        chat_query = [{"role": "user", "content": user_content}]
        return tokenizer.apply_chat_template(
            chat_query,
            tokenize=False,
            add_generation_prompt=True,
        ) + "<PATH>"
```

这样处理有两个目的：

1. 把问题放在 `user` 轮次里。
2. 把 `<PATH>` 放在 assistant 生成起点之后。

如果不这样做，`<PATH>` 可能落在 user prompt 中，模型继续生成时 trie 前缀会错位，图约束解码可能失效。

## 6. Tokenizer 与特殊 token

训练入口都会添加路径边界 token：

```python
special_tokens = {"additional_special_tokens": ["<PATH>", "</PATH>"]}
tokenizer.add_special_tokens(special_tokens)
model.resize_token_embeddings(len(tokenizer))
```

这一步非常关键：

1. tokenizer 需要把 `<PATH>` 和 `</PATH>` 当作独立 token。
2. 模型 embedding 矩阵需要扩展，否则新 token 没有对应向量。
3. LoRA 保存时需要保留扩展后的 embedding 层，否则后续加载 adapter 可能因为词表大小不一致报错。

因此保存时使用：

```python
model.save_pretrained(config.output_dir, save_embedding_layers=True)
tokenizer.save_pretrained(config.output_dir)
```

SFT 入口还会处理 pad token：

```python
if tokenizer.pad_token is None and tokenizer.eos_token is not None:
    tokenizer.pad_token = tokenizer.eos_token
```

## 7. 模型加载

`finetune/train.py` 和 `finetune/sft_train.py` 的加载逻辑基本相同：

```python
tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    config.model_name,
    dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto" or {"": gpu_id},
)
```

关键点：

| 配置 | 作用 |
| --- | --- |
| `dtype=torch.bfloat16` | 用 bf16 加载模型，降低显存占用 |
| `trust_remote_code=True` | 允许加载模型仓库自定义代码 |
| `device_map="auto"` | 由 Transformers 自动放置模型到可用设备 |
| `device_map={"": gpu_id}` | 强制整模型放到指定 GPU |

SFT 入口还会设置：

```python
model.config.use_cache = False
```

训练时通常关闭 KV cache，因为训练需要完整反向传播，cache 主要用于推理加速。

当前配置里模型路径是：

```python
model_name = "offline_assets/models/Qwen3.5-0.8B"
```

工作区中实际模型目录看起来是 `offline_assets/models/Qwen_Qwen3.5-0.8B`。如果训练时报模型路径不存在，应先修正 `config.py` 中的 `model_name`。

## 8. LoRA 微调配置

两条 `finetune/` 训练线都使用 PEFT LoRA：

```python
lora_config = LoraConfig(
    r=config.lora_r,
    lora_alpha=config.lora_alpha,
    target_modules=config.target_modules,
    lora_dropout=config.lora_dropout,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
```

默认 LoRA 参数：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `lora_r` | 64 | 低秩矩阵的 rank，越大可训练容量越强 |
| `lora_alpha` | 128 | LoRA 缩放系数 |
| `lora_dropout` | 0.05 | LoRA 分支 dropout |
| `bias` | none | 不训练 bias |
| `task_type` | CAUSAL_LM | 自回归语言模型任务 |

默认 target modules：

```python
[
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
```

这意味着不仅注意力投影层会被适配，MLP 的 gate/up/down 投影也会被适配。相比只训练 `q_proj`、`v_proj`，容量更强，显存和训练开销也更高。

## 9. GRPO 训练全流程

### 9.1 启动入口

GRPO 训练从 `finetune/train.py` 开始：

```python
def main():
    config = GRPOConfig()
    configure_logging(config.log_level)
    tokenizer = AutoTokenizer.from_pretrained(...)
    model = AutoModelForCausalLM.from_pretrained(...)
    tokenizer.add_special_tokens(...)
    model.resize_token_embeddings(...)
    model = get_peft_model(model, lora_config)
    train_dataset = PathGenerationDataset(...)
    train_loader = DataLoader(...)
    reward_calculator = PathRewardCalculator(...)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    trainer = GRPOTrainer(...)
    trainer.train(train_loader, config.num_epochs)
    model.save_pretrained(...)
```

整体顺序是：

1. 读取配置。
2. 配置日志。
3. 加载 tokenizer。
4. 加载基座模型。
5. 添加 `<PATH>` 和 `</PATH>`。
6. 扩展 embedding。
7. 注入 LoRA。
8. 加载训练数据。
9. 初始化奖励函数。
10. 初始化 AdamW。
11. 进入 GRPO 训练循环。
12. 保存 LoRA adapter 和 tokenizer。

### 9.2 GRPOConfig 关键参数

```python
generation_mode = "sampling"
num_generations = 12
max_new_tokens = 256
max_resample_rounds = 4
num_beams = 12
temperature = 1.0
top_p = 0.9
top_k = 0
num_beam_groups = 4
diversity_penalty = 0.4
kl_penalty_beta = 0.04
learning_rate = 2e-6
num_epochs = 2
batch_size = 8
gradient_accumulation_steps = 8
index_path_length = 3
use_semantic_reward = True
```

注意：`GRPOConfig` 中虽然有 `gradient_accumulation_steps`，但当前 `GRPOTrainer` 没有真正使用这个参数。GRPO 的 optimizer step 是每个 DataLoader batch 调一次，而不是累积多个 batch 后再 step。

### 9.3 DataLoader 输出

GRPO 使用 `PathGenerationDataset`。每个有效样本返回：

```python
{
    "id": data["id"],
    "question": data["question"],
    "q_entity": data["q_entity"],
    "a_entity": data["a_entity"],
    "graph": data["graph"],
    "input_query": input_query,
    "ground_paths": ground_paths,
    "trie": trie,
}
```

collate 函数不做 padding，而是过滤掉 `None` 样本后返回 list：

```python
batch = [sample for sample in batch if sample is not None]
return batch
```

原因是 GRPO 每个问题要单独做图约束生成，不能像普通 SFT 那样直接把所有样本 pad 成一个大 tensor。

### 9.4 每个 epoch 的外层循环

`GRPOTrainer.train`：

```python
for epoch in range(num_epochs):
    for batch in train_loader:
        if batch is None:
            continue
        loss = self.train_step(batch)
```

每个 batch 是多个问题样本。`train_step` 会逐个 sample 做 rollout、奖励计算和反向传播，最后统一 `optimizer.step()`。

### 9.5 单个样本的 rollout

`train_step` 对每个样本调用：

```python
generated_texts, sequences, sequence_attention_mask = self.generate_group_paths(
    sample["input_query"],
    sample["trie"],
    sample["ground_paths"],
    self.config.num_generations,
)
```

目标是为同一个问题生成 `num_generations` 条候选路径。默认是 12 条。

### 9.6 GT 路径注入

`generate_group_paths` 的第一步不是直接让模型采样，而是先注入一部分 ground truth path：

```python
for ground_path in self._select_ground_paths_for_injection(...):
    clean_path_text, full_sequence, sequence_length, path_token_ids = ...
    unique_texts.append(clean_path_text)
    unique_sequences.append(full_sequence)
    blocked_path_tokens.add(path_token_ids)
```

注入策略：

1. 如果只有 1 条 ground path，最多注入 1 条。
2. 如果有多条 ground paths，最多注入 2 条。
3. 注入数量不会超过 `num_generations`。

这样做的目的：

1. 保证每组候选里至少有较高奖励样本，避免全是垃圾路径导致 advantage 信号弱。
2. 给 GRPO 的组内相对比较提供正样本。
3. 让模型稳定看到标准路径格式。

注入后的 GT 路径会加入 `blocked_path_tokens`，后续从 trie 里过滤，防止模型又重复生成同一条路径。

### 9.7 补采样和去重

注入后如果还没有达到 `num_generations`，代码会继续生成：

```python
for round_index in range(max_rounds):
    remaining = num_generations - len(unique_texts)
    filtered_trie = self._build_filtered_trie(trie, blocked_path_tokens)
    round_texts, round_sequences, round_lengths = self.generate_paths_once(...)
```

去重使用：

```python
canonical_path = " ".join(str(text).split()).strip()
```

补采样停止条件：

1. 已获得 `num_generations` 条唯一路径。
2. trie 候选路径被耗尽。
3. 连续两轮没有新增唯一路径。
4. 达到 `max_resample_rounds`。
5. 如果是 `beam_search`，最多只生成一轮。

如果最终不足 `num_generations`，代码不会报错，而是用已有路径继续训练。

### 9.8 图约束解码

生成时使用 HuggingFace `model.generate`，但传入了：

```python
prefix_allowed_tokens_fn = gcr.allowed_tokens_fn
stopping_criteria = StoppingCriteriaList([PathEndStoppingCriteria(...)])
```

`GraphConstrainedDecoding.allowed_tokens_fn` 每生成一个 token 都会被调用，用来决定“下一步允许哪些 token”。

核心逻辑：

1. 如果当前序列里已经出现最近的 `<PATH>`，且之后还没出现 `</PATH>`，进入约束状态。
2. 在约束状态下，取当前路径前缀，调用 `trie.get(prefix)` 得到合法下一个 token 集合。
3. 如果当前 beam 已经生成 `</PATH>`，只允许生成 EOS。
4. 如果 trie 查不到当前前缀，退回允许全词表。

伪代码：

```python
if has_start_token and not has_end_token_after_last_start:
    allowed = trie.get(tokens_after_last_start)
elif has_end_token_after_last_start:
    allowed = [eos_token_id]
else:
    allowed = all_tokens
```

这就是“图约束”：模型不能随便拼路径，正常情况下只能沿 trie 中存在的候选路径往下生成。

注意：当前实现里如果 trie 前缀匹配失败，会 fallback 到全词表。这能避免生成过程崩溃，但也意味着某些异常前缀下约束会变弱。

### 9.9 停止条件

`PathEndStoppingCriteria` 会检查每个 beam：

1. 找到最后一个 `<PATH>`。
2. 如果该 `<PATH>` 后还没出现 `</PATH>`，说明该 beam 没结束。
3. 当所有 beam 都已经生成 `</PATH>` 后，整个 generate 停止。

同时 `allowed_tokens_fn` 在某条 beam 已经闭合 `</PATH>` 后只允许 EOS，避免模型继续输出无关文本。

### 9.10 生成模式

`GRPOConfig.generation_mode` 支持：

| 模式 | 行为 |
| --- | --- |
| `sampling` | 普通随机采样，使用 temperature、top_p、top_k |
| `beam_search` | 确定性 beam search |
| `beam_sample` | beam search + 采样 |
| `group_beam_search` | 分组 beam search，加 diversity penalty |

`sampling` 默认参数：

```python
do_sample=True
temperature=1.0
top_p=0.9
top_k=0
```

`top_k=0` 表示不启用 top-k 截断。

`beam_search` 会确保：

```python
num_beams >= num_generations
```

`group_beam_search` 会确保：

```python
num_beams >= num_generations
num_beams >= num_beam_groups
num_beams % num_beam_groups == 0
```

### 9.11 奖励函数

生成一组路径后，调用：

```python
rewards, advantages = reward_calculator.calculate_group_rewards(
    generated_texts,
    sample["question"],
    sample["a_entity"],
    sample["ground_paths"],
)
```

单条路径奖励由以下部分组成：

```text
reward = answer_reward
       + path_match_reward
       + detour_reward
       + loop_penalty
       + semantic_reward
```

#### 9.11.1 路径解析

`build_path_info` 会先清理 `<PATH>`、`</PATH>` 和答案段，再按 `->` 切分：

```text
实体A -> 关系1 -> 实体B -> 关系2 -> 实体C
```

会解析成：

```python
segments = ["实体A", "关系1", "实体B", "关系2", "实体C"]
triples = [
    ("实体A", "关系1", "实体B"),
    ("实体B", "关系2", "实体C"),
]
relations = ["关系1", "关系2"]
hops = 2
final_entity = "实体C"
```

如果路径为空、分段数量为偶数、长度小于 3，认为无效，直接奖励 `-1.0`。

#### 9.11.2 答案命中奖励

```python
answer_reward = 5.0 if final_entity in answer_entities else 0.0
```

只检查路径终点实体是否精确命中 `a_entity`。这里是字符串精确匹配，不做别名、大小写、模糊匹配。

#### 9.11.3 路径结构匹配奖励

生成路径会和每条 ground path 比较，选择 match score 最高的 ground path 作为参考。

结构匹配分：

```python
match_score = 2.5 * exact + 1.75 * prefix_ratio + 1.0 * rel_sim
```

其中：

| 项 | 含义 |
| --- | --- |
| `exact` | 三元组序列完全一致则为 1，否则 0 |
| `prefix_ratio` | 生成路径和真值路径共享的三元组前缀长度 / 真值 hop 数 |
| `rel_sim` | 关系序列 LCS 长度 / 真值关系数 |

这使奖励既鼓励完全匹配，也鼓励关系顺序和路径前缀接近。

#### 9.11.4 绕路惩罚

如果生成路径比参考 ground path 更长：

```python
extra_hops = max(0, generated_hops - reference_hops)
detour_reward = -1.2 * extra_hops
```

这会惩罚不必要的长路径。

#### 9.11.5 环路惩罚

代码检查路径中的实体序列：

```python
entities = generated_info["segments"][::2]
```

如果同一个实体重复出现，说明路径绕圈：

```python
loop_penalty = max(-4.0, -2.0 * revisit_count)
```

最多扣 4 分。

#### 9.11.6 语义辅助奖励

如果 `use_semantic_reward=True`，会调用 embedding API：

```python
generated_emb = get_embeddings([generated_path_text])
ground_embs = get_embeddings(ground_texts)
similarities = cosine_similarity(generated_emb, ground_embs)
semantic_reward = 0.5 * best_similarity
```

作用是：即使结构没有完全匹配，只要生成路径在语义上接近真值路径，也给一点弱奖励。

实现里有 embedding cache，同样的文本不会重复请求 API。

工程注意点：

1. 当前配置文件包含 `embedding_api_url`、`embedding_model_name`、`embedding_api_key` 字段。
2. API key 不应硬编码在仓库代码里，建议改成环境变量读取。
3. 如果 API 请求失败，语义奖励会退化为 0，不会中断训练。

### 9.12 Advantage 计算

一组路径的 rewards 会被标准化成 advantages：

```python
advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
```

GRPO 的思想是做组内相对优化：同一个问题下，高于组均值的路径 advantage 为正，低于组均值的路径 advantage 为负。

如果一组路径奖励完全相同，advantages 约等于 0，该样本对策略更新贡献很小。

### 9.13 log probability 计算

训练器先计算旧 log prob：

```python
with torch.no_grad():
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    old_log_probs = sum_token_log_probs(...)
```

再计算当前 log prob：

```python
outputs = model(input_ids=full_sequences, attention_mask=sequence_attention_mask)
new_log_probs = sum_token_log_probs(...)
```

token 级 log prob 的计算方式：

1. 模型输出 logits。
2. 去掉最后一个位置的 logits，因为每个位置预测下一个 token。
3. 对 logits 做 `log_softmax`。
4. 用真实的 `input_ids[:, 1:]` gather 出每个目标 token 的 log prob。
5. 用 `attention_mask[:, 1:]` 去掉 padding。
6. 沿序列求和，得到整条路径序列 log prob。

### 9.14 GRPO loss

当前实现的 loss：

```python
kl_div = new_log_probs - old_log_probs
loss = -(advantages * new_log_probs).mean() + beta * kl_div.mean()
```

含义：

1. 如果 advantage 为正，增大该路径 log prob 会降低 loss。
2. 如果 advantage 为负，降低该路径 log prob 会降低 loss。
3. `kl_penalty_beta` 控制 KL 惩罚项强度。

重要实现细节：

当前 `old_log_probs` 是同一个模型在本 step 内用 `torch.no_grad()` 重新算出来的，并不是一个冻结 reference model 或 rollout 前长期保存的 old policy。因此这个 KL 项不是标准 PPO/GRPO 中严格意义上的“相对旧策略 KL”。在当前代码里，它主要起到形式上的正则项，初始通常非常接近 0。

### 9.15 反向传播和优化器 step

`train_step` 对 batch 内每个 sample 都执行：

```python
loss.backward()
```

然后在整个 batch 结束后：

```python
optimizer.step()
optimizer.zero_grad()
```

因此当前 GRPO 的有效更新粒度是：

```text
1 个 optimizer step = 1 个 DataLoader batch 中所有有效 question 样本的累计梯度
```

默认 `batch_size=8`，每个 question 默认生成最多 12 条路径，所以一次 step 理论上会涉及最多 `8 * 12 = 96` 条路径序列的 log prob。

### 9.16 GRPO 保存产物

训练结束后：

```python
model.save_pretrained(config.output_dir, save_embedding_layers=True)
tokenizer.save_pretrained(config.output_dir)
```

`output_dir` 默认自动生成，规则大致是：

```text
finetune/output/{model_name}__{data_name}__{split}__{generation_mode}__pathlen_{index_path_length}__semantic_on
```

例如：

```text
finetune/output/Qwen3.5-0.8B__COKG_QA__train__sampling__pathlen_3__semantic_on
```

保存的是 LoRA adapter 加 tokenizer，不一定是完整合并后的模型。

## 10. SFT 训练全流程

### 10.1 启动入口

SFT 从 `finetune/sft_train.py` 开始：

```python
config = SFTConfig()
tokenizer = AutoTokenizer.from_pretrained(...)
model = AutoModelForCausalLM.from_pretrained(...)
tokenizer.add_special_tokens(...)
model.resize_token_embeddings(...)
model.config.use_cache = False
model = get_peft_model(model, lora_config)
train_dataset = SFTPathDataset(...)
train_loader = DataLoader(...)
optimizer = torch.optim.AdamW(...)
trainer = SFTTrainer(...)
trainer.train(...)
model.save_pretrained(...)
```

与 GRPO 不同，SFT 不做 rollout，不调用 reward，也不使用 trie 约束生成。它直接把真值路径作为监督答案，让模型做 teacher forcing。

### 10.2 SFTConfig 关键参数

```python
learning_rate = 2e-6
num_epochs = 2
batch_size = 8
gradient_accumulation_steps = 8
index_path_length = 3
max_ground_paths_per_sample = None
```

`max_ground_paths_per_sample=None` 表示每个问题的全部 ground paths 都展开成训练样本。如果设为 2，就更接近 GRPO 中“每题最多注入 2 条 GT path”的设置。

### 10.3 SFT 样本展开

`SFTPathDataset._build_samples` 对每条原始数据：

1. 调用 prompt builder 得到 `input_query` 和 `ground_paths`。
2. 不构建 trie，因为 SFT 不需要图约束 rollout。
3. 对 ground paths 去标签、去重、截断。
4. 每条 ground path 生成一个 SFT 样本。

样本结构：

```python
{
    "id": f"{source_id}__path_{path_index}",
    "source_id": source_id,
    "prompt_text": prompt_text,
    "target_text": target_text,
    "ground_path": clean_path,
}
```

如果 prompt 已经以 `<PATH>` 结尾，target 只包含：

```text
实体A -> 关系1 -> 实体B</PATH>
```

否则 target 包含完整路径：

```text
<PATH>实体A -> 关系1 -> 实体B</PATH>
```

### 10.4 SFT collate 与 labels

`SFTPathDataset.collate_fn` 会分别 tokenize prompt 和 target：

```python
prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
target_ids = tokenizer(target_text, add_special_tokens=False).input_ids
full_input_ids = prompt_ids + target_ids + [eos_token_id]
full_labels = [-100] * len(prompt_ids) + target_ids + [eos_token_id]
```

关键点：

1. prompt 部分 label 是 `-100`，不参与 loss。
2. target 和 EOS 参与 loss。
3. input 用 `eos_token_id` padding。
4. labels 用 `-100` padding。
5. attention mask 标记真实 token。

这是标准 causal LM SFT 的做法：模型看见 prompt，学习预测答案路径。

### 10.5 SFT loss

`SFTTrainer.train` 中直接调用：

```python
outputs = model(
    input_ids=batch["input_ids"],
    attention_mask=batch["attention_mask"],
    labels=batch["labels"],
)
loss = outputs.loss
```

Transformers 的 CausalLM 会自动做 shift：

1. 第 t 个位置的 hidden state 预测第 t+1 个 token。
2. label 为 `-100` 的位置忽略。
3. 对 target token 做交叉熵。

### 10.6 SFT 梯度累积

SFT 实现了梯度累积：

```python
(loss / accumulation_steps).backward()
if accumulated_batches >= accumulation_steps:
    optimizer.step()
    optimizer.zero_grad()
```

因此 SFT 的有效 batch size 是：

```text
batch_size * gradient_accumulation_steps
```

默认是：

```text
8 * 8 = 64 条 SFT path 样本
```

如果最后不足一个 accumulation window，也会在 epoch 末尾执行一次 `optimizer.step()`。

### 10.7 SFT 保存产物

保存方式和 GRPO 一样：

```python
model.save_pretrained(config.output_dir, save_embedding_layers=True)
tokenizer.save_pretrained(config.output_dir)
```

默认目录包含：

```text
finetune/output/{model_name}__{data_name}__{split}__sft__pathlen_{index_path_length}__{path_tag}
```

其中 `path_tag` 是：

1. `all_paths`：使用全部 ground paths。
2. `top_{N}_paths`：每题最多使用 N 条 ground paths。

## 11. GRPO 与 SFT 的区别

| 对比项 | GRPO | SFT |
| --- | --- | --- |
| 训练信号 | 奖励函数 + 组内 advantage | ground path 交叉熵 |
| 是否生成 rollout | 是 | 否 |
| 是否使用 trie | 是，用于图约束解码 | 否 |
| 是否需要 reward | 是 | 否 |
| 是否调用 embedding API | 可选 | 否 |
| 是否支持负反馈 | 是，低奖励路径 advantage 为负 | 否，只模仿 GT |
| 稳定性 | 更复杂，依赖 reward 质量 | 更稳定 |
| 探索能力 | 更强，可以优化非完全匹配路径 | 较弱，只学习给定路径 |
| 当前梯度累积 | 配置存在但未实现 | 已实现 |

实践上常见策略是：

1. 先 SFT，让模型学会路径格式和基本图谱关系表达。
2. 再 GRPO，让模型在图约束候选中偏向更能命中答案、更短、更合理的路径。

## 12. 标准 TRL SFT workflow

### 12.1 启动方式

脚本是 `scripts/train_kg_specialized_llm.sh`：

```bash
accelerate launch --config_file ${CONFIG} workflow/finetune_kg_specialized_llm.py \
    --data_path_list ${DATASET_LIST} \
    --model_name_or_path ${MODEL_PATH} \
    --output_dir ${SAVE_PATH} \
    --use_peft ${USE_PEFT} \
    --bf16 True \
    --num_train_epochs ${EPOCH} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --learning_rate 2e-5 \
    --lr_scheduler_type "cosine" \
    --gradient_checkpointing ${GRADIENT_CHECKPOINTING} \
    --attn_implementation ${ATTN_IMP} \
    --response_template "${RESPONSE_TEMPLATE}"
```

这条训练线使用：

1. HuggingFace `TrainingArguments`。
2. TRL `SFTTrainer`。
3. TRL `DataCollatorForCompletionOnlyLM`。
4. 可选 PEFT LoRA。
5. accelerate 多 GPU 或 DeepSpeed ZeRO-3。

### 12.2 全量微调与 LoRA

脚本里有两组配置。

LoRA 微调示例：

```bash
BATCH_SIZE=50
USE_PEFT=True
EPOCH=20
GRADIENT_CHECKPOINTING=False
GRADIENT_ACCUMULATION_STEPS=1
CONFIG="accelerate_configs/multi_gpu.yaml"
```

全量微调示例：

```bash
BATCH_SIZE=4
USE_PEFT=False
EPOCH=3
GRADIENT_CHECKPOINTING=True
GRADIENT_ACCUMULATION_STEPS=16
CONFIG="accelerate_configs/deepspeed_zero3.yaml"
```

区别：

| 方式 | 特点 |
| --- | --- |
| LoRA | 只训练 adapter，显存低，保存小，适合快速实验 |
| 全量微调 | 更新全部模型参数，能力上限更高，显存和算力开销大 |

### 12.3 workflow 模型加载

`workflow/finetune_kg_specialized_llm.py` 加载模型：

```python
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    trust_remote_code=True,
    token=HF_TOKEN,
    torch_dtype=torch.bfloat16,
    attn_implementation=attn_implementation,
    load_in_4bit=load_in_4bit,
    load_in_8bit=load_in_8bit,
)
model.config.use_cache = False
```

支持：

1. bf16。
2. flash attention。
3. 4bit / 8bit 加载。
4. HuggingFace token。

如果 `use_peft=True`，LoRA 只作用于：

```python
target_modules=["q_proj", "v_proj"]
```

这和 `finetune/` 新代码的 target modules 不同。

### 12.4 workflow 数据格式化

`input_formatter` 对每条 ground truth path 构造一个 chat 样本：

```python
raw_input = ZERO_SHOT_PROMPT.format(
    question=question,
    entities=",".join(start_node),
)
ground_path_string = f"<PATH>{path_to_string(path)}</PATH>"
path_answer = path[-1][-1].strip()
response = ANS_TEMPLATE.format(
    reasoning_path=ground_path_string,
    answer=path_answer,
)
chat = [
    {"role": "user", "content": raw_input},
    {"role": "assistant", "content": response},
]
final_input = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
```

assistant 的输出格式是：

```text
# Reasoning Path:
<PATH>实体A -> 关系1 -> 实体B</PATH>
# Answer:
实体B
```

这条 workflow 不只是训练路径，还训练模型在路径后给出答案。

### 12.5 Completion-only loss

workflow 使用：

```python
DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer, mlm=False)
```

它会把 assistant 回复之前的 token label mask 掉，只对 assistant response 计算 loss。不同模型的 assistant 起始标记不同，所以脚本需要配置 `RESPONSE_TEMPLATE`：

| 模型 | RESPONSE_TEMPLATE |
| --- | --- |
| Qwen2 Instruct | `<|im_start|>assistant` |
| Llama 2 chat | `[/INST]` |
| Llama 3.1 Instruct | `<|start_header_id|>assistant<|end_header_id|>` |

如果这个模板配置错，collator 可能找不到 assistant 起点，导致 loss mask 错误。

### 12.6 accelerate 配置

`accelerate_configs/multi_gpu.yaml`：

```yaml
distributed_type: MULTI_GPU
mixed_precision: bf16
num_processes: 2
```

`accelerate_configs/deepspeed_zero3.yaml`：

```yaml
distributed_type: DEEPSPEED
mixed_precision: bf16
zero_stage: 3
zero3_save_16bit_model: true
num_processes: 2
```

ZeRO-3 会把参数、梯度、优化器状态切分到多张 GPU 上，适合全量微调 7B/8B 模型。

## 13. LoRA adapter 合并

训练完成后，如果只保存了 LoRA adapter，推理有两种方式：

1. 推理时加载 base model + adapter。
2. 先把 adapter 合并进 base model，再像普通模型一样加载。

`finetune/merge_adapter.py` 实现第二种。

流程：

1. 读取 `MergeConfig`。
2. 优先从 adapter 目录加载 tokenizer。
3. 如果 adapter 目录没有 tokenizer，回退到 base model tokenizer。
4. 加载 base model。
5. 如果 tokenizer 词表大小和 base embedding 不一致，执行 `resize_token_embeddings`。
6. 用 `PeftModel.from_pretrained(base_model, adapter_path)` 加载 adapter。
7. 执行 `merge_and_unload()`。
8. 保存完整模型、tokenizer、generation_config。

核心代码：

```python
base_model = AutoModelForCausalLM.from_pretrained(...)
if len(tokenizer) != vocab_size:
    base_model.resize_token_embeddings(len(tokenizer))
peft_model = PeftModel.from_pretrained(base_model, adapter_path)
merged_model = peft_model.merge_and_unload()
merged_model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)
```

合并后的目录默认是：

```text
{adapter_path}__merged
```

## 14. 一次完整训练的数据流

以 GRPO 为例，一条样本从原始数据到参数更新的完整路径如下：

```text
原始 JSON/HF Dataset 样本
  |
  |-- question / q_entity / a_entity / graph
  v
build_graph(graph)
  |
  |-- DFS 得到候选路径 paths
  |-- shortest path 得到 ground_paths
  v
path_to_string + <PATH> 标签
  |
  |-- 候选路径 tokenize -> MarisaTrie
  |-- ground_paths 保留文本形式
  v
构造 input_query prompt
  |
  v
DataLoader batch
  |
  v
prepare_model_prompt + chat_template
  |
  v
注入少量 GT path
  |
  v
model.generate + prefix_allowed_tokens_fn + trie
  |
  v
得到 K 条路径文本和 token sequence
  |
  v
reward.py 计算每条路径 reward
  |
  v
组内标准化得到 advantage
  |
  v
模型 forward 计算 sequence log_prob
  |
  v
loss = -advantage * log_prob + KL penalty
  |
  v
loss.backward()
  |
  v
AdamW optimizer.step()
  |
  v
保存 LoRA adapter + tokenizer
```

以 SFT 为例：

```text
原始样本
  |
  v
build_graph + shortest path 得到 ground_paths
  |
  v
每条 ground path 展开成一个监督样本
  |
  v
prompt_ids + target_ids + eos
  |
  v
labels = -100(prompt) + target + eos
  |
  v
model(input_ids, labels)
  |
  v
cross entropy loss
  |
  v
loss / gradient_accumulation_steps backward
  |
  v
AdamW step
  |
  v
保存 LoRA adapter + tokenizer
```

## 15. 训练涉及的大语言模型环节

### 15.1 数据清洗与切分

训练前必须保证：

1. `question` 文本可读。
2. `q_entity` 在 `graph` 里能找到。
3. `a_entity` 在 `graph` 里能找到。
4. `graph` 三元组方向和 `undirected` 设置一致。
5. 训练 split、验证 split、测试 split 不混用。
6. ground truth path 能通过 `get_truth_paths` 找到，否则 SFT 会跳过，GRPO 奖励也会变弱。

### 15.2 Prompt 工程

本项目 prompt 的关键不是写得复杂，而是保证：

1. 明确要求输出知识图谱路径。
2. 路径格式统一为 `实体 -> 关系 -> 实体`。
3. 使用 `<PATH>` 和 `</PATH>` 标记边界。
4. 对 chat model 使用正确 chat template。
5. `<PATH>` 出现在 assistant 生成区间，而不是 user 区间。

### 15.3 Tokenizer 适配

大模型训练中 tokenizer 改动会影响模型结构：

1. 添加 special tokens 会改变词表大小。
2. 必须 resize embedding。
3. 保存 adapter 时要保存新增 embedding。
4. 推理、合并、训练必须使用同一份 tokenizer。

### 15.4 模型初始化

模型从预训练或指令模型开始，而不是从零训练：

1. 基座模型提供语言能力。
2. LoRA adapter 学习图路径生成偏好。
3. bf16 降低显存。
4. `device_map` 控制模型放置。

### 15.5 参数高效微调

LoRA 不直接改原始权重，而是在目标线性层旁边加低秩增量：

```text
W' = W + scale * B @ A
```

训练时只更新 A 和 B。好处是：

1. 显存低。
2. 保存体积小。
3. 可以快速切换不同任务 adapter。
4. 合并后推理不需要 PEFT 结构。

### 15.6 前向传播

SFT 前向传播输入是完整的 prompt + target。GRPO 前向传播输入是 rollout 生成出来的完整序列。

共同点：

1. 使用 causal attention。
2. 每个 token 预测下一个 token。
3. padding 位置通过 attention mask 屏蔽。

### 15.7 Loss 设计

SFT loss：

```text
交叉熵(prompt 被 mask，只学习 target)
```

GRPO loss：

```text
组内相对优势加权的 sequence log probability
```

两者本质区别：

1. SFT 学“标准答案是什么”。
2. GRPO 学“同一个问题下哪些候选更值得提高概率”。

### 15.8 反向传播

反向传播只更新 LoRA 参数和必要的 embedding 保存层。基座模型主干参数被冻结。

### 15.9 优化器

两条 `finetune/` 训练线都用：

```python
torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
```

当前没有显式 scheduler、warmup、weight decay 配置。workflow 训练线通过 HuggingFace TrainingArguments 使用：

```bash
--lr_scheduler_type "cosine"
--warmup_ratio 0.03
--weight_decay 0.
```

### 15.10 混合精度

`finetune/` 默认 bf16 加载模型，但没有使用 `accelerate` 或 AMP scaler。workflow 通过 accelerate 配置 `mixed_precision: bf16`。

### 15.11 Checkpoint 与恢复

`finetune/` 配置里有 `save_steps`，但当前 GRPO/SFT 自定义 trainer 没有按 step 保存 checkpoint，也没有断点恢复逻辑。它们只在训练结束保存最终 adapter。

workflow 训练线使用 HuggingFace Trainer，包含 checkpoint 检测逻辑：

```python
last_checkpoint = get_last_checkpoint(training_args.output_dir)
trainer.train(resume_from_checkpoint=checkpoint)
```

但脚本中默认：

```bash
--save_strategy "no"
```

这意味着如果不改脚本，也不会定期保存中间 checkpoint。

### 15.12 评估

`finetune/` 当前训练代码没有内置 validation loop。训练日志主要包括：

1. batch 是否跳过。
2. 每个样本的 reward max / mean。
3. 每个 epoch 的 avg loss。

真正评估需要训练后用项目推理脚本生成路径和答案，再看 `eval_result.txt` 或详细预测文件。

## 16. 关键工程风险

### 16.1 默认路径可能不匹配

配置里的模型路径是：

```text
offline_assets/models/Qwen3.5-0.8B
```

当前本地模型目录看起来是：

```text
offline_assets/models/Qwen_Qwen3.5-0.8B
```

训练前要确认路径真实存在。

### 16.2 默认数据 split 可能不存在

配置默认 `train_split="train"`，但当前本地 `offline_assets/datasets/COKG_QA` 只看到 `test.jsonl`。如果没有训练 split，需要补数据或改配置。

### 16.3 GRPO 的 `gradient_accumulation_steps` 没生效

`GRPOConfig` 有这个字段，但 `GRPOTrainer` 没使用。不要按这个字段估算 GRPO 等效 batch size。

### 16.4 GRPO KL 不是标准冻结参考模型 KL

当前 `old_log_probs` 不是由单独冻结 reference model 计算的。若要更接近标准 GRPO/PPO，应引入 reference model 或保存 rollout 时的旧策略 log prob。

### 16.5 embedding API key 不应硬编码

`config.py` 中有 API key 配置字段。实际项目应使用环境变量或密钥管理，不应把真实 key 写入仓库。

### 16.6 图约束有 fallback

`GraphConstrainedDecoding` 在 trie 无匹配时会返回全词表，这会让路径合法性约束变弱。通常原因可能是：

1. `<PATH>` 所在位置不对。
2. tokenizer 对 special token 处理不一致。
3. trie 路径 token 和生成前缀不一致。
4. prompt 中多了意外 token。

### 16.7 SFT 日志/注释存在编码异常

`sft_train.py`、`sft_trainer.py`、`merge_adapter.py` 的部分中文注释和日志在当前环境显示为 mojibake，但核心 Python 逻辑仍可读。建议后续统一保存为 UTF-8。

## 17. 推荐训练流程

如果要稳定训练一个图谱路径模型，推荐流程：

1. 准备训练集，确保每条样本有 `question`、`q_entity`、`a_entity`、`graph`。
2. 先用小样本跑 `PathGenerationDataset`，确认 `trie is not None` 的比例足够高。
3. 跑 SFT，让模型先学会路径格式和基础图谱表达。
4. 检查 SFT 输出是否稳定包含 `<PATH>...</PATH>`。
5. 用 SFT adapter 或合并模型作为 GRPO 起点。
6. 开启 GRPO，先关闭语义奖励或使用稳定 embedding 服务，确认 reward 分布正常。
7. 观察每个样本的 `reward_max`、`reward_mean`、`unique/num_generations`。
8. 训练结束保存 adapter。
9. 用 `merge_adapter.py` 合并模型。
10. 用图约束推理脚本做测试集评估。

## 18. 常用命令

### 18.1 GRPO 训练

```bash
cd finetune
python train.py
```

训练前通常需要改 `finetune/config.py`：

```python
model_name = "真实存在的模型目录"
data_path = "真实存在的数据集目录或 json 文件"
train_split = "train"
generation_mode = "sampling"
num_generations = 12
index_path_length = 3
use_semantic_reward = True
```

### 18.2 SFT 训练

```bash
cd finetune
python sft_train.py
```

可考虑：

```python
max_ground_paths_per_sample = 2
```

用于限制每个问题展开的路径数量。

### 18.3 workflow SFT 训练

```bash
bash scripts/train_kg_specialized_llm.sh
```

适用于多 GPU、DeepSpeed 或标准 HuggingFace Trainer 流程。

### 18.4 合并 adapter

```bash
cd finetune
python merge_adapter.py
```

运行前修改：

```python
base_model_path = "基座模型目录"
adapter_path = "训练输出 adapter 目录"
output_path = "合并后完整模型目录"
```

## 19. 调参建议

### 19.1 `index_path_length`

控制 DFS 候选路径最大 hop 数。

| 值 | 影响 |
| --- | --- |
| 太小 | 可能找不到答案路径，trie 候选少 |
| 太大 | 候选路径爆炸，生成慢，噪声多 |

当前默认是 3，适合先跑通。

### 19.2 `num_generations`

GRPO 每题保留多少条路径。

| 值 | 影响 |
| --- | --- |
| 小 | 组内比较弱，advantage 不稳定 |
| 大 | rollout 成本高，显存和时间增加 |

默认 12。

### 19.3 `generation_mode`

推荐顺序：

1. 先用 `sampling`，探索更多路径。
2. 如果重复太多，调低 temperature 或使用 `group_beam_search`。
3. 如果想稳定复现实验，用 `beam_search`，但多样性会弱。

### 19.4 `use_semantic_reward`

如果 embedding API 不稳定，先设为 `False` 跑通结构奖励。确认训练稳定后再打开。

### 19.5 LoRA rank

默认 `r=64` 容量较强。显存紧张时可以降到 16 或 32。

## 20. 最小可运行前检查清单

训练前至少检查：

1. `config.model_name` 路径存在。
2. `config.data_path` 能被 `load_dataset` 加载。
3. `config.train_split` 存在。
4. tokenizer 有 `eos_token_id`。
5. 添加 `<PATH>`、`</PATH>` 后模型执行了 `resize_token_embeddings`。
6. 至少一部分样本能构建非空 trie。
7. `ground_paths` 非空比例不能太低。
8. 如果启用语义奖励，embedding API 可访问。
9. GPU 支持 bf16，或需要把 dtype 改成 fp16/fp32。
10. 输出目录不会覆盖重要实验结果。

## 21. 总结

`finetune/` 下的新训练代码围绕“图约束路径生成”设计。SFT 分支负责让模型模仿真值路径，GRPO 分支负责在图约束候选路径中通过奖励函数做偏好优化。两者都使用 LoRA，只保存 adapter 和 tokenizer，必要时通过 `merge_adapter.py` 合并成完整模型。

从大语言模型训练角度看，这套流程覆盖了数据构造、prompt 模板、special token 扩词表、模型加载、LoRA 参数注入、batch 构造、前向传播、loss 计算、反向传播、优化器更新、模型保存和部署合并。当前代码的主要缺口是 GRPO 梯度累积未实现、KL 不是冻结 reference KL、缺少中间 checkpoint 和验证循环，后续如果要做稳定实验，这几个点应优先补齐。
