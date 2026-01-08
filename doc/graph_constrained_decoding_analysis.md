# Graph-Constrained Decoding 详细分析

## 概述

Graph-Constrained Decoding 是 GCR 框架的核心技术，通过在 LLM 解码过程中引入知识图谱约束，确保生成的推理路径在 KG 中真实存在。

---

## 整体流程图

```
输入: Question + KG
    ↓
[步骤1] 构建 KG 索引 (Trie)
    ↓
[步骤2] 构造 Prompt
    ↓
[步骤3] LLM 图约束解码 (核心)
    ↓
[步骤4] 解析输出
    ↓
输出: Multiple KG paths + Candidate answers
```

---

## 输入输出总览

### 📥 总输入
```json
{
    "question": "Where was Barack Obama born?",
    "id": "WebQSP_001",
    "q_entity": ["Barack Obama"],
    "a_entity": ["Honolulu"],
    "answer": ["Honolulu"],
    "graph": [
        ["Barack Obama", "people.person.place_of_birth", "Honolulu"],
        ["Barack Obama", "people.person.profession", "Politician"],
        ["Honolulu", "location.location.containedby", "Hawaii"],
        ["Hawaii", "location.location.contains", "Honolulu"],
        ...
    ]
}
```

### 📤 总输出
```json
{
    "id": "WebQSP_001",
    "question": "Where was Barack Obama born?",
    "prediction": [
        "# Reasoning Path:\nBarack Obama -> people.person.place_of_birth -> Honolulu\n# Answer:\nHonolulu",
        "# Reasoning Path:\nBarack Obama -> people.person.place_of_birth -> Honolulu -> location.location.containedby -> Hawaii\n# Answer:\nHawaii",
        ...
    ],
    "ground_truth": ["Honolulu"],
    "input": "..."
}
```

---

## 详细步骤分析

---

## 步骤1: 构建 KG 索引 (Trie)

### 代码位置
- `src/qa_prompt_builder.py` → `PathGenerationWithAnswerPromptBuilder.get_graph_index()`
- `workflow/build_graph_index.py`

### 1.1 子步骤: DFS 遍历 KG

**输入:**
```python
{
    "q_entity": ["Barack Obama"],  # 起始实体
    "graph": [                      # KG 三元组列表
        ["Barack Obama", "place_of_birth", "Honolulu"],
        ["Barack Obama", "profession", "Politician"],
        ["Honolulu", "containedby", "Hawaii"],
        ...
    ]
}
```

**处理:**
```python
# src/utils/graph_utils.py → dfs()
def dfs(graph, start_node_list, max_length=2):
    """
    从起始节点出发，深度优先搜索所有长度 <= max_length 的路径
    """
    path_lists = set()
    for start_node in start_node_list:
        dfs_visit(start_node, [])
    return list(path_lists)

# 示例执行:
# 起始: "Barack Obama"
# max_length = 2

# 找到的路径:
# 1跳: Barack Obama -> place_of_birth -> Honolulu
# 1跳: Barack Obama -> profession -> Politician
# 2跳: Barack Obama -> place_of_birth -> Honolulu -> containedby -> Hawaii
# ...
```

**输出: 路径列表**
```python
paths_list = [
    [("Barack Obama", "place_of_birth", "Honolulu")],
    [("Barack Obama", "profession", "Politician")],
    [
        ("Barack Obama", "place_of_birth", "Honolulu"),
        ("Honolulu", "containedby", "Hawaii")
    ],
    ...
]
# 可能有 1万 ~ 10万+ 条路径
```

---

### 1.2 子步骤: 路径转字符串

**输入:** 路径三元组列表

**处理:**
```python
# src/utils/utils.py → path_to_string()
def path_to_string(path: list) -> str:
    result = ""
    for i, (h, r, t) in enumerate(path):
        if i == 0:
            result += f"{h} -> {r} -> {t}"
        else:
            result += f" -> {r} -> {t}"  # 后续跳省略头实体
    return result

# 示例:
path = [
    ("Barack Obama", "place_of_birth", "Honolulu"),
    ("Honolulu", "containedby", "Hawaii")
]
path_to_string(path)
# 输出: "Barack Obama -> place_of_birth -> Honolulu -> containedby -> Hawaii"
```

**输出: 字符串列表**
```python
paths_list_str = [
    "Barack Obama -> place_of_birth -> Honolulu",
    "Barack Obama -> profession -> Politician",
    "Barack Obama -> place_of_birth -> Honolulu -> containedby -> Hawaii",
    ...
]
```

---

### 1.3 子步骤: Tokenize 路径

**输入:** 路径字符串列表

**处理:**
```python
# 使用模型的 tokenizer 将字符串转为 token ID
tokenized_paths = tokenizer(paths_list_str, 
                            padding=False, 
                            add_special_tokens=False).input_ids

# 示例:
# "Barack Obama -> place_of_birth -> Honolulu" 
# → [2045, 8084, 4287, 98623, 6254, 24671]
```

**输出: Token 序列列表**
```python
tokenized_path_list = [
    [2045, 8084, 4287, 98623, 6254, 24671],
    [2045, 8084, 4287, 67890, 12121, 45678],
    [2045, 8084, 4287, 98623, 6254, 24671, 5555, 6666],
    ...
]
# 每条路径后加 EOS token
tokenized_path_list = [
    ids + [tokenizer.eos_token_id] for ids in tokenized_paths
]
```

---

### 1.4 子步骤: 构建 MarisaTrie

**输入:** Token 序列列表

**处理:**
```python
# src/trie.py → MarisaTrie
trie = MarisaTrie(tokenized_path_list, 
                  max_token_id=len(tokenizer) + 1)

# 内部结构 (概念图):
# {
#   2045: {  # "Barack"
#     8084: {  # "Obama"
#       4287: {  # "->"
#         98623: {  # "place_of_birth"
#           6254: {  # "->"
#             24671: {EOS}  # "Honolulu" + EOS
#           }
#         },
#         67890: {...}  # "profession" 分支
#       }
#     }
#   }
# }
```

**输出: MarisaTrie 对象**
```python
trie.get([])  
# → [2045]  # 第一个 token 只能是 2045 (Barack)

trie.get([2045, 8084, 4287])  
# → [98623, 67890, ...]  # "Barack Obama ->" 后可接的关系
```

---

## 步骤2: 构造 Prompt

### 代码位置
- `src/qa_prompt_builder.py` → `PathGenerationWithAnswerPromptBuilder.process_input()`

### 输入
```python
{
    "question": "Where was Barack Obama born?",
    "q_entity": ["Barack Obama"]
}
```

### 处理
```python
# Zero-shot prompt template
ZERO_SHOT_PROMPT = """Reasoning path is a sequence of triples in the KG that connects the topic entities in the question to answer entities. Given a question, please generate some reasoning paths in the KG starting from the topic entities to answer the question.

# Question: 
{question}
# Topic entities: 
{entities}
"""

# 填充
input_query = ZERO_SHOT_PROMPT.format(
    question="Where was Barack Obama born?",
    entities="Barack Obama"
)
```

### 输出
```python
input_query = """Reasoning path is a sequence of triples in the KG that connects the topic entities in the question to answer entities. Given a question, please generate some reasoning paths in the KG starting from the topic entities to answer the question.

# Question: 
Where was Barack Obama born?
# Topic entities: 
Barack Obama
"""
```

---

## 步骤3: LLM 图约束解码 (核心!)

### 代码位置
- `src/graph_constrained_decoding.py` → `GraphConstrainedDecoding.allowed_tokens_fn()`
- `src/llms/graph_constrained_decoding_model.py` → `generate_sentence()`

### 3.1 子步骤: 准备模型输入

**输入:** 
```python
llm_input = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Reasoning path is a sequence of triples in the KG...
# Question: 
Where was Barack Obama born?
# Topic entities: 
Barack Obama
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
# (Chat template 格式化后)
```

**处理:**
```python
# Tokenize
inputs = tokenizer(llm_input, return_tensors="pt")
input_ids = inputs.input_ids  # Shape: [1, L]
# 例: [1, 128000, 271, 49, ..., 78191]  (L=50 tokens)
```

---

### 3.2 子步骤: 创建约束函数

**处理:**
```python
# src/graph_constrained_decoding.py
gcr = GraphConstrainedDecoding(
    tokenizer=tokenizer,
    trie=trie,  # 步骤1构建的 Trie
    start_token_ids=None,  # 不使用 <PATH> 标记
    end_token_ids=None,
    enable_constrained_by_default=False  # 默认不约束
)

# 关键函数
def allowed_tokens_fn(batch_id: int, sent: torch.Tensor):
    """
    每生成一个 token 都会调用此函数
    
    Args:
        batch_id: beam 的索引
        sent: 已生成的完整 token 序列 [input_ids + generated_ids]
    
    Returns:
        allow_tokens: 允许生成的 token ID 列表
    """
    
    # 提取已生成的部分 (去掉 input)
    L_input = 50  # 输入长度
    generated = sent[L_input:]  # 已生成的 token
    
    # 查询 Trie
    allow_tokens = trie.get(generated.tolist())
    
    if len(allow_tokens) == 0:
        # Trie 中没有匹配的路径 → 允许所有 token (退出约束)
        return list(range(len(tokenizer)))
    
    return allow_tokens
```

---

### 3.3 子步骤: Beam Search with Constraint

**处理:**
```python
# HuggingFace generate 方法
res = model.generate(
    input_ids=input_ids,  # [1, 50]
    generation_config=GenerationConfig(
        num_beams=10,           # 10 个 beam
        num_return_sequences=10,
        num_beam_groups=10,      # Group beam search
        diversity_penalty=1.0,   # 鼓励多样性
        max_new_tokens=1024
    ),
    prefix_allowed_tokens_fn=gcr.allowed_tokens_fn,  # 约束函数!
    return_dict_in_generate=True
)
```

**逐步解码示例:**

```
初始状态:
input_ids = [1, 128000, ..., 78191]  (50 tokens, 结尾是 "assistant\n")

Beam 1:
Step 1: 生成第 1 个 token
  - 当前: [input_ids]
  - 调用 allowed_tokens_fn(0, [input_ids])
  - generated = []
  - trie.get([]) → [2045]  (只能是 "Barack")
  - 模型在允许列表 [2045] 中选择 → 生成 2045
  - 现在: [input_ids, 2045]

Step 2: 生成第 2 个 token
  - 当前: [input_ids, 2045]
  - 调用 allowed_tokens_fn(0, [input_ids, 2045])
  - generated = [2045]
  - trie.get([2045]) → [8084]  (只能是 "Obama")
  - 生成 8084
  - 现在: [input_ids, 2045, 8084]

Step 3: 生成第 3 个 token
  - 当前: [input_ids, 2045, 8084]
  - generated = [2045, 8084]
  - trie.get([2045, 8084]) → [4287]  (只能是 "->")
  - 生成 4287
  - 现在: [input_ids, 2045, 8084, 4287]

Step 4: 生成第 4 个 token
  - 当前: [input_ids, 2045, 8084, 4287]
  - generated = [2045, 8084, 4287]
  - trie.get([2045, 8084, 4287]) → [98623, 67890, 12345, ...]
  - 这里有多个选择! (place_of_birth, profession, ...)
  - 模型根据概率选择 98623 (place_of_birth)
  - 现在: [input_ids, 2045, 8084, 4287, 98623]

...继续直到生成完整路径和答案...

最终 Beam 1: 
"Barack Obama -> place_of_birth -> Honolulu\n# Answer:\nHonolulu"
```

**Group Beam Search 产生多样化结果:**
```
Beam 1: Barack Obama -> place_of_birth -> Honolulu
Beam 2: Barack Obama -> place_of_birth -> Honolulu -> containedby -> Hawaii
Beam 3: Barack Obama -> place_of_birth -> Honolulu -> containedby -> Hawaii -> type -> US_State
Beam 4: ... (其他路径)
...
Beam 10: ...
```

---

### 3.4 输出解码

**输入:** 生成的 token 序列

**处理:**
```python
# res.sequences: [10, L_total]  # 10 个 beam 的结果
# 去掉输入部分，只保留生成的
generated_ids = res.sequences[:, input_ids.shape[1]:]

# Decode
responses = []
for seq in generated_ids:
    text = tokenizer.decode(seq, skip_special_tokens=True)
    responses.append(text)
```

**输出:**
```python
responses = [
    "# Reasoning Path:\nBarack Obama -> place_of_birth -> Honolulu\n# Answer:\nHonolulu",
    "# Reasoning Path:\nBarack Obama -> place_of_birth -> Honolulu -> containedby -> Hawaii\n# Answer:\nHawaii",
    ...
]
```

---

## 步骤4: 解析输出

### 代码位置
- `workflow/predict_paths_and_answers.py` → `prediction()`

### 输入
```python
prediction = [
    "# Reasoning Path:\nBarack Obama -> place_of_birth -> Honolulu\n# Answer:\nHonolulu",
    ...
]
```

### 处理
```python
# 构建结果字典
result = {
    "id": "WebQSP_001",
    "question": "Where was Barack Obama born?",
    "prediction": prediction,  # 保持原格式
    "ground_truth": ["Honolulu"],
    "ground_truth_paths": [
        "Barack Obama -> place_of_birth -> Honolulu"
    ],
    "input": llm_input
}

# 保存到 JSONL
with open('predictions.jsonl', 'a') as f:
    f.write(json.dumps(result) + '\n')
```

### 输出文件
```jsonl
{"id": "WebQSP_001", "question": "Where was Barack Obama born?", "prediction": ["# Reasoning Path:\nBarack Obama -> place_of_birth -> Honolulu\n# Answer:\nHonolulu", ...], "ground_truth": ["Honolulu"], ...}
{"id": "WebQSP_002", ...}
...
```

---

## 核心机制详解

### 🔑 约束如何工作？

```python
# 关键在于 prefix_allowed_tokens_fn

# 每个解码 step:
def decode_step(current_tokens):
    # 1. 模型计算所有 token 的概率
    logits = model(current_tokens)  # Shape: [vocab_size]
    probs = softmax(logits)         # [0.001, 0.002, ..., 0.05, ...]
    
    # 2. 调用约束函数
    allowed = allowed_tokens_fn(current_tokens)
    # 例: [2045, 8084, 4287]  (只允许这3个 token)
    
    # 3. 屏蔽不允许的 token
    for i in range(vocab_size):
        if i not in allowed:
            probs[i] = -inf  # 概率为0
    
    # 4. 在允许的 token 中选择概率最高的
    next_token = argmax(probs)  # 只能是 2045, 8084, 或 4287 中的一个
    
    return next_token
```

**结果:**
- 模型**物理上不可能**生成 KG 中不存在的路径
- "零推理幻觉"的实现机制

---

## 数据流总结

```
输入数据:
├─ question: "Where was Barack Obama born?"
├─ q_entity: ["Barack Obama"]
├─ graph: [[h, r, t], ...]  (1000+ triples)
└─ a_entity: ["Honolulu"]

↓ [步骤1: 构建索引]

中间数据:
├─ paths_list: [[(h,r,t), ...], ...]  (10,000+ paths)
├─ tokenized_paths: [[token_ids], ...]
└─ trie: MarisaTrie 对象

↓ [步骤2: 构造 Prompt]

中间数据:
└─ input_query: "Reasoning path is... # Question: ..."

↓ [步骤3: 解码]

生成过程:
每个 step 调用 allowed_tokens_fn(sent)
├─ 提取 generated tokens
├─ trie.get(generated) → allowed list
└─ 在 allowed list 中生成下一个 token

↓ [步骤4: 输出]

最终输出:
├─ prediction: [path1+answer1, path2+answer2, ...]  (10 candidates)
├─ ground_truth: ["Honolulu"]
└─ 保存到 predictions.jsonl
```

---

## 实际例子: 完整流程

### 输入
```json
{
    "id": "test_001",
    "question": "Who is the president of USA?",
    "q_entity": ["USA"],
    "a_entity": ["Joe Biden"],
    "graph": [
        ["USA", "government.country.president", "Joe Biden"],
        ["USA", "location.country.capital", "Washington DC"],
        ["Joe Biden", "people.person.profession", "Politician"],
        ...
    ]
}
```

### 执行步骤1: 构建 Trie
```
DFS 从 "USA" 出发:
Path 1: USA -> government.country.president -> Joe Biden
Path 2: USA -> location.country.capital -> Washington DC
Path 3: USA -> government.country.president -> Joe Biden -> profession -> Politician
...
(总共 15,342 条路径)

构建 Trie:
包含 15,342 条 tokenized 路径
```

### 执行步骤2: 构造 Prompt
```
# Question:
Who is the president of USA?
# Topic entities:
USA
```

### 执行步骤3: 约束解码
```
Beam 1 生成过程:
Step 1: [] → trie.get([]) → [5467] → 生成 5467 ("USA")
Step 2: [5467] → trie.get([5467]) → [4287] → 生成 4287 ("->")
Step 3: [5467, 4287] → trie.get([5467, 4287]) → [12234, 8899, ...] → 生成 12234 ("government.country.president")
Step 4: [5467, 4287, 12234] → ... → 生成后续 token

最终生成:
"# Reasoning Path:\nUSA -> government.country.president -> Joe Biden\n# Answer:\nJoe Biden"

10 个 beams 生成 10 条不同路径
```

### 输出
```json
{
    "id": "test_001",
    "question": "Who is the president of USA?",
    "prediction": [
        "# Reasoning Path:\nUSA -> government.country.president -> Joe Biden\n# Answer:\nJoe Biden",
        "# Reasoning Path:\nUSA -> government.country.president -> Joe Biden -> profession -> Politician\n# Answer:\nPolitician",
        ...
    ],
    "ground_truth": ["Joe Biden"]
}
```

---

## 关键配置参数

### 模型配置
```python
model_path = "rmanluo/GCR-Meta-Llama-3.1-8B-Instruct"  # 微调后的模型
max_new_tokens = 1024  # 最大生成长度
attn_implementation = "flash_attention_2"  # Flash Attention 加速
```

### 解码配置
```python
generation_mode = "group-beam"  # Group Beam Search
k = 10  # 生成 10 条候选路径
num_beam_groups = 10  # 10 个独立的 beam 组
diversity_penalty = 1.0  # 多样性惩罚
```

### 索引配置
```python
index_path_length = 2  # DFS 最大深度 (2跳路径)
undirected = False  # 有向图
```

---

## 性能考量

### 时间复杂度
```
步骤1 (构建 Trie):
- DFS: O(N * K^D)  # N=起始节点数, K=平均出度, D=深度
- 构建 Trie: O(M * L)  # M=路径数, L=平均长度
- 总计: 通常 1-10 秒

步骤3 (解码):
- 每个 step: O(M) Trie 查询
- 总 steps: O(T)  # T=生成的 token 数
- 总计: O(T * M)
- Beam Search: 乘以 beam 数量
- 实际: 每个问题 5-30 秒
```

### 内存占用
```
Trie: 通常 10-100 MB (取决于路径数量)
Model: 8B 模型约 16 GB (bf16)
总计: ~20 GB GPU 内存
```

---

## 总结

### Graph-Constrained Decoding 的核心价值

1. **硬约束**: Trie 物理阻止不存在的路径
2. **零幻觉**: 生成的路径 100% 在 KG 中
3. **可解释**: 每个答案都有明确的推理路径
4. **高效**: O(M) 查询 vs O(N*M) 遍历

### 技术创新点

1. **Trie 索引**: 将 KG 路径编码为 token 序列的前缀树
2. **约束解码**: 使用 `prefix_allowed_tokens_fn` 在生成时约束
3. **Beam Search**: Group Beam 产生多样化的候选路径
4. **两阶段推理**: 先生成路径，再归纳答案 (下一步)

### 与传统方法对比

| 方法       | 路径生成     | 是否会幻觉 | 可解释性 |
| ---------- | ------------ | ---------- | -------- |
| 直接生成   | LLM 自由生成 | ✅ 会       | ❌ 低     |
| 后处理过滤 | 生成后验证   | ✅ 会(浪费) | ⚠️ 中     |
| **GCR**    | 约束生成     | ❌ 不会     | ✅ 高     |

这就是 Graph-Constrained Decoding 的完整工作机制！