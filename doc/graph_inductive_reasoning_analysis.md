# Graph Inductive Reasoning (图归纳推理) 详细讲解

## 目录
1. [概述](#概述)
2. [在GCR框架中的位置](#在gcr框架中的位置)
3. [输入输出格式](#输入输出格式)
4. [核心代码分析](#核心代码分析)
5. [具体执行步骤](#具体执行步骤)
6. [Prompt构建详解](#prompt构建详解)
7. [完整数据流示例](#完整数据流示例)
8. [与Stage 1的协作关系](#与stage-1的协作关系)
9. [实际应用场景](#实际应用场景)

---

## 概述

### 什么是 Graph Inductive Reasoning？

**Graph Inductive Reasoning（图归纳推理）** 是 GCR 框架的第二个阶段，也是最终推理阶段。它的核心思想是：

> **使用通用大型语言模型（如 GPT-3.5-turbo、GPT-4）对 Stage 1 生成的多条候选推理路径和假设答案进行归纳推理，产生最终答案。**

### 为什么需要这个阶段？

| 问题             | Stage 1的局限                                        | Stage 2的解决方案               |
| ---------------- | ---------------------------------------------------- | ------------------------------- |
| **候选答案多样** | KG-specialized LLM生成多个候选答案，但不知道哪个正确 | 通用LLM对多个候选进行投票和归纳 |
| **路径理解**     | 路径是图结构化的，缺乏自然语言解释                   | 通用LLM理解路径语义，进行推理   |
| **知识整合**     | KG-specialized LLM专注于路径生成                     | 通用LLM具有更强的常识推理能力   |
| **答案聚合**     | 可能有多条路径指向不同答案                           | 综合分析所有路径得出最可靠答案  |

### 核心特点

✅ **无需额外训练**：使用现成的通用LLM（如ChatGPT）  
✅ **推理能力强**：利用大模型的常识和归纳能力  
✅ **结果可靠**：基于多条路径的证据进行推理  
✅ **可解释性高**：LLM可以解释为什么选择某个答案

---

## 在GCR框架中的位置

### 完整的两阶段流程

```
┌─────────────────────────────────────────────────────────────────┐
│                    GCR 完整推理流程                              │
└─────────────────────────────────────────────────────────────────┘

Stage 1: Graph-Constrained Decoding (图约束解码)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
输入：问题 + 知识图谱
      ↓
加载预构建的图索引
      ↓
将路径Token化 → 构建Trie
      ↓
KG-specialized LLM + 约束解码 (Beam Search)
      ↓
输出：候选推理路径 + 假设答案
      [路径1 → 答案A, 路径2 → 答案A, 路径3 → 答案B, ...]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Stage 2: Graph Inductive Reasoning (图归纳推理) ← 本文档重点
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
输入：问题 + 候选路径 + 候选答案
      ↓
构建推理Prompt（包含所有候选路径）
      ↓
通用LLM (GPT-3.5/GPT-4) 进行归纳推理
      ↓
输出：最终答案
      答案: Hawaii
```

### 为什么要分两阶段？

| 维度     | Stage 1 (KG-specialized LLM)  | Stage 2 (通用LLM)      |
| -------- | ----------------------------- | ---------------------- |
| **专长** | 图结构遍历、路径生成          | 自然语言理解、归纳推理 |
| **约束** | 严格受Trie约束，保证忠实性    | 无约束，自由推理       |
| **成本** | 小模型（0.5B-8B），可本地部署 | 大模型（API调用）      |
| **速度** | 快速生成多条候选              | 单次推理，相对较慢     |
| **输出** | 多个候选（发散）              | 单个最终答案（收敛）   |

**分工明确**：
- Stage 1 负责"找"（在KG中找到所有可能的推理路径）
- Stage 2 负责"判"（判断哪条路径最合理，答案是什么）

---

## 输入输出格式

### Stage 2 的输入

来自三个来源：

#### 1. 原始问题数据（从数据集加载）
```python
{
    'id': 'q_001',
    'question': 'Where was Barack Obama born?',
    'q_entity': ['Barack Obama'],      # 问题实体
    'a_entity': ['Hawaii'],            # 答案实体（仅用于评估）
    'answer': ['Hawaii'],              # 真实答案（仅用于评估）
    'graph': [...]                     # 知识图谱（可选）
}
```

#### 2. Stage 1 的输出（候选路径和答案）
从 `results/GenPaths/{dataset}/{model}/test/predictions.jsonl` 加载：

```python
{
    'id': 'q_001',
    'question': 'Where was Barack Obama born?',
    'prediction': [
        'Barack Obama -> born_in -> Hawaii',
        'Barack Obama -> born_in -> Hawaii',
        'Barack Obama -> lived_in -> Washington',
        'Barack Obama -> born_in -> Hawaii',
    ],
    'ground_truth': ['Hawaii'],
    'ground_truth_paths': [
        'Barack Obama -> born_in -> Hawaii'
    ]
}
```

#### 3. 合并后的数据（merge_path_result）
```python
{
    'id': 'q_001',
    'question': 'Where was Barack Obama born?',
    'q_entity': ['Barack Obama'],
    'a_entity': ['Hawaii'],
    'answer': ['Hawaii'],
    'predicted_paths': [          # 从Stage 1加载
        'Barack Obama -> born_in -> Hawaii',
        'Barack Obama -> born_in -> Hawaii',
        'Barack Obama -> lived_in -> Washington',
        'Barack Obama -> born_in -> Hawaii'
    ],
    'ground_paths': [             # 真实路径（评估用）
        'Barack Obama -> born_in -> Hawaii'
    ]
}
```

### Stage 2 的输出

保存到 `results/KGQA/{dataset}/{model}/test/predictions.jsonl`：

```python
{
    'id': 'q_001',
    'question': 'Where was Barack Obama born?',
    'prediction': 'Hawaii',      # LLM生成的最终答案（或多个答案）
    'ground_truth': ['Hawaii'],  # 真实答案
    'input': '...'               # 完整的输入prompt
}
```

### 文件路径示例

```
Stage 1 输出路径（输入到Stage 2）：
results/GenPaths/RoG-webqsp/GCR-Meta-Llama-3.1-8B-Instruct/test/zero-shot-group-beam-k10/predictions.jsonl

Stage 2 输出路径：
results/KGQA/RoG-webqsp/gpt-3.5-turbo/test/add_path_results_GenPaths_..../predictions.jsonl
```

---

## 核心代码分析

### 1. 主函数：`main(args, LLM)`

```python
def main(args, LLM):
    # 1. 加载数据集
    input_file = os.path.join(args.data_path, args.d)
    dataset = load_dataset(input_file, split=args.split)
    
    # 2. 合并Stage 1的路径结果
    if args.add_path:
        paths_datasets = []
        with open(args.reasoning_path, "r") as f:
            for line in f:
                paths_datasets.append(json.loads(line))
        dataset = merge_path_result(
            dataset, paths_datasets, 
            filter_empty=args.filter_empty,
            remove_dup_path=args.remove_dup_path
        )
    
    # 3. 初始化模型和Prompt构建器
    model = LLM(args)
    input_builder = PromptBuilder(
        add_path=args.add_path,
        maximun_token=model.maximun_token,
        tokenize=model.token_len
    )
    
    # 4. 准备推理
    model.prepare_for_inference()
    
    # 5. 对每个问题进行推理
    for data in tqdm(dataset):
        res = make_prediction(data, args, processed_list, input_builder, model)
        if res is not None:
            fout.write(json.dumps(res) + "\n")
    
    # 6. 评估结果
    eval_result(os.path.join(output_dir, f"predictions.jsonl"))
```

**关键步骤**：
1. **加载数据**：从HuggingFace加载问答数据集
2. **合并路径**：将Stage 1生成的路径合并到数据集中
3. **初始化模型**：加载通用LLM（如GPT-3.5）
4. **逐个推理**：对每个问题构建prompt并推理
5. **评估结果**：计算准确率、Hit率、F1等指标

### 2. 路径合并函数：`merge_path_result`

```python
def merge_path_result(qa_dataset, path_dataset, n_proc=1, 
                      filter_empty=False, remove_dup_path=False):
    # 构建问题ID到路径的映射
    question_to_path = dict()
    for data in path_dataset:
        qid = data["id"]
        predicted_paths = (
            list(set(data["prediction"])) if remove_dup_path 
            else data["prediction"]
        )
        ground_paths = data["ground_truth_paths"]
        question_to_path[qid] = {
            "predicted_paths": predicted_paths,
            "ground_paths": ground_paths,
        }

    # 将路径添加到每个样本中
    def find_path(sample):
        qid = sample["id"]
        sample["predicted_paths"] = []
        sample["ground_paths"] = []
        if qid in question_to_path:
            sample["predicted_paths"] = question_to_path[qid]["predicted_paths"]
            sample["ground_paths"] = question_to_path[qid]["ground_paths"]
        return sample

    qa_dataset = qa_dataset.map(find_path, num_proc=n_proc)
    
    # 可选：过滤掉没有路径的样本
    if filter_empty:
        qa_dataset = qa_dataset.filter(
            lambda x: len(x["ground_paths"]) > 0, num_proc=n_proc
        )
    
    return qa_dataset
```

**功能**：
- 从Stage 1的输出文件中加载路径
- 将路径按问题ID匹配到数据集
- 可选去重和过滤空路径

### 3. 推理函数：`make_prediction`

```python
def make_prediction(data, args, processed_list, input_builder, model):
    question = data["question"]
    answer = data["answer"]
    id = data["id"]
    
    # 跳过已处理的样本
    if id in processed_list:
        return None
    
    # 构建输入prompt
    input = input_builder.process_input(data)
    input = model.prepare_model_prompt(input)
    
    # LLM生成答案
    prediction = model.generate_sentence(input)
    
    if prediction is None:
        return None
    
    # 返回结果
    result = {
        "id": id,
        "question": question,
        "prediction": prediction,
        "ground_truth": answer,
        "input": input,
    }
    return result
```

**关键点**：
- 使用 `input_builder.process_input(data)` 构建prompt
- 调用 `model.generate_sentence(input)` 生成答案
- 返回包含问题、预测、真实答案的结果字典

---

## 具体执行步骤

### 步骤流程图

```
┌─────────────────────────────────────────────────────────────┐
│ Step 1: 数据准备                                              │
└─────────────────────────────────────────────────────────────┘
  - 加载问答数据集（question, entities, answer）
  - 加载Stage 1的路径预测结果（predictions.jsonl）
  - 合并数据：将路径添加到每个问题样本中
          ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 2: Prompt构建                                           │
└─────────────────────────────────────────────────────────────┘
  - 提取predicted_paths（候选推理路径）
  - 格式化路径为自然语言
  - 构建完整prompt：
      • 指令部分（Instruction）
      • 推理路径部分（Reasoning Paths）
      • 问题部分（Question）
          ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 3: 长度检查与截断                                         │
└─────────────────────────────────────────────────────────────┘
  - 检查prompt总长度（token数）
  - 如果超过模型最大长度：
      • 随机打乱路径顺序
      • 逐条添加路径直到达到token限制
      • 保证prompt不会超出模型上下文窗口
          ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 4: LLM推理                                              │
└─────────────────────────────────────────────────────────────┘
  - 将prompt输入到通用LLM（GPT-3.5/GPT-4）
  - LLM阅读所有推理路径
  - LLM进行归纳推理
  - 生成最终答案
          ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 5: 答案提取与保存                                         │
└─────────────────────────────────────────────────────────────┘
  - 提取LLM生成的答案文本
  - 保存到predictions.jsonl：
      • id, question, prediction, ground_truth
          ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 6: 评估                                                 │
└─────────────────────────────────────────────────────────────┘
  - 计算准确率（Accuracy）
  - 计算命中率（Hit Rate）
  - 计算F1分数、精确率、召回率
  - 生成评估报告
```

### 详细步骤说明

#### Step 1: 数据准备

**代码位置**：`main()` 函数开始部分

```python
# 加载问答数据集
dataset = load_dataset(input_file, split=args.split)
# 示例：RoG-webqsp 测试集

# 加载Stage 1的路径
paths_datasets = []
with open(args.reasoning_path, "r") as f:
    for line in f:
        paths_datasets.append(json.loads(line))

# 合并
dataset = merge_path_result(dataset, paths_datasets)
```

**输入示例**：
```python
# 原始数据集样本
{
    'id': 'WebQTest-1234',
    'question': 'Where was Barack Obama born?',
    'q_entity': ['Barack Obama'],
    'answer': ['Hawaii']
}

# Stage 1路径预测
{
    'id': 'WebQTest-1234',
    'prediction': [
        'Barack Obama -> born_in -> Hawaii',
        'Barack Obama -> born_in -> Hawaii',
        'Barack Obama -> spouse -> Michelle Obama',
        'Barack Obama -> born_in -> Hawaii'
    ]
}

# 合并后
{
    'id': 'WebQTest-1234',
    'question': 'Where was Barack Obama born?',
    'predicted_paths': [
        'Barack Obama -> born_in -> Hawaii',
        'Barack Obama -> spouse -> Michelle Obama'
    ],  # 注意：去重后只有2条
    'answer': ['Hawaii']
}
```

#### Step 2: Prompt构建

**代码位置**：`PromptBuilder.process_input()`

**Prompt模板**：
```python
SAQ_RULE_INSTRUCTION = """Based on the reasoning paths, please answer the given question. Please keep the answer as simple as possible and only return answers."""

GRAPH_CONTEXT = """Reasoning Paths:
{context}

"""

QUESTION = """Question:
{question}"""
```

**构建过程**：
```python
def process_input(self, question_dict):
    question = question_dict["question"]
    
    # 1. 提取候选路径
    lists_of_paths = question_dict['predicted_paths']
    
    # 2. 构建问题部分
    input = self.QUESTION.format(question=question)
    # 结果: "Question:\nWhere was Barack Obama born?"
    
    # 3. 构建指令部分
    instruction = self.SAQ_RULE_INSTRUCTION
    
    # 4. 检查长度并截断路径
    context = self.check_prompt_length(
        other_prompt, lists_of_paths, self.maximun_token
    )
    
    # 5. 组装完整prompt
    input = self.GRAPH_CONTEXT.format(context=context) + input
    input = PROMPT_TEMPLATE.format(instruction=instruction, input=input)
    
    return input
```

**生成的Prompt示例**：
```
Reasoning Paths:
Barack Obama -> born_in -> Hawaii
Barack Obama -> born_in -> Hawaii
Barack Obama -> spouse -> Michelle Obama

Question:
Where was Barack Obama born?

Based on the reasoning paths, please answer the given question. Please keep the answer as simple as possible and only return answers.
```

#### Step 3: 长度检查与截断

**代码位置**：`PromptBuilder.check_prompt_length()`

```python
def check_prompt_length(self, prompt, list_of_paths, maximun_token):
    all_paths = "\n".join(list_of_paths)
    all_tokens = prompt + all_paths
    
    # 如果长度合适，直接返回
    if self.tokenize(all_tokens) < maximun_token:
        return all_paths
    else:
        # 超长：随机打乱并逐条添加
        random.shuffle(list_of_paths)
        new_list_of_paths = []
        
        for p in list_of_paths:
            tmp_all_paths = "\n".join(new_list_of_paths + [p])
            tmp_all_tokens = prompt + tmp_all_paths
            if self.tokenize(tmp_all_tokens) > maximun_token:
                return "\n".join(new_list_of_paths)
            new_list_of_paths.append(p)
```

**功能**：
- 防止prompt超过模型的最大context长度
- 如果路径太多，优先保留部分路径
- 随机打乱保证公平性

**示例**：
```python
# 假设最大长度 = 4096 tokens
# 路径列表有100条，总长度 = 5000 tokens

# 执行后：
# - 随机选择约80条路径
# - 总长度 < 4096 tokens
# - 既保证多样性，又不超长
```

#### Step 4: LLM推理

**代码位置**：`make_prediction()` → `model.generate_sentence()`

```python
# 将prompt输入到LLM
prediction = model.generate_sentence(input)
```

**LLM内部处理**（以GPT-3.5为例）：
1. **理解问题**：识别出在问"出生地"
2. **分析路径**：
   - 路径1: `Barack Obama -> born_in -> Hawaii` ✓ 相关
   - 路径2: `Barack Obama -> born_in -> Hawaii` ✓ 相关（重复）
   - 路径3: `Barack Obama -> spouse -> Michelle Obama` ✗ 不相关
3. **归纳推理**：
   - 多条路径指向 "Hawaii"
   - "born_in" 关系直接回答了问题
   - "spouse" 关系与问题无关
4. **生成答案**：`Hawaii`

**输出示例**：
```python
# 简单答案
prediction = "Hawaii"

# 或带解释的答案
prediction = "Hawaii\n\nBased on the reasoning paths, Barack Obama was born in Hawaii."
```

#### Step 5: 答案提取与保存

**代码位置**：`make_prediction()` 返回部分

```python
result = {
    "id": "WebQTest-1234",
    "question": "Where was Barack Obama born?",
    "prediction": "Hawaii",
    "ground_truth": ["Hawaii"],
    "input": "Reasoning Paths:\n..."
}

# 写入文件
fout.write(json.dumps(result) + "\n")
```

**输出文件格式**（predictions.jsonl）：
```jsonl
{"id": "q_001", "question": "...", "prediction": "Hawaii", "ground_truth": ["Hawaii"], "input": "..."}
{"id": "q_002", "question": "...", "prediction": "USA", "ground_truth": ["United States"], "input": "..."}
{"id": "q_003", "question": "...", "prediction": "1961", "ground_truth": ["1961"], "input": "..."}
```

#### Step 6: 评估

**代码位置**：`eval_result()` 函数

```python
def eval_result(predict_file):
    acc_list = []
    hit_list = []
    f1_list = []
    
    with open(predict_file, "r") as f:
        for line in f:
            data = json.loads(line)
            prediction = data["prediction"]
            answer = data["ground_truth"]
            
            # 计算指标
            acc = eval_acc(prediction, answer)
            hit = eval_hit(prediction, answer)
            f1 = eval_f1(prediction, answer)
            
            acc_list.append(acc)
            hit_list.append(hit)
            f1_list.append(f1)
    
    # 输出平均结果
    print(f"Accuracy: {mean(acc_list) * 100}")
    print(f"Hit: {mean(hit_list) * 100}")
    print(f"F1: {mean(f1_list) * 100}")
```

**评估指标**：
- **Accuracy（准确率）**：预测答案与真实答案的完全匹配程度
- **Hit Rate（命中率）**：预测答案中是否包含任意一个真实答案
- **F1 Score**：精确率和召回率的调和平均
- **Precision（精确率）**：预测正确的答案占所有预测的比例
- **Recall（召回率）**：预测正确的答案占真实答案的比例

**评估示例**：
```python
# 问题1
prediction = "Hawaii"
ground_truth = ["Hawaii"]
→ Accuracy = 1.0, Hit = 1, F1 = 1.0

# 问题2
prediction = "USA"
ground_truth = ["United States", "USA", "US"]
→ Accuracy = 0.33, Hit = 1, F1 = 0.5

# 问题3
prediction = "Paris, London"
ground_truth = ["Paris"]
→ Accuracy = 0.5, Hit = 1, F1 = 0.67
```

---

## Prompt构建详解

### Prompt的三个核心部分

```
┌──────────────────────────────────────────┐
│ 1. Instruction (指令部分)                 │
│    告诉LLM要做什么                         │
└──────────────────────────────────────────┘
Based on the reasoning paths, please answer 
the given question. Please keep the answer 
as simple as possible and only return answers.

┌──────────────────────────────────────────┐
│ 2. Context (上下文/推理路径部分)           │
│    提供推理所需的知识和路径                 │
└──────────────────────────────────────────┘
Reasoning Paths:
Barack Obama -> born_in -> Hawaii
Barack Obama -> spouse -> Michelle Obama
Barack Obama -> profession -> Politician

┌──────────────────────────────────────────┐
│ 3. Question (问题部分)                    │
│    要回答的具体问题                        │
└──────────────────────────────────────────┘
Question:
Where was Barack Obama born?
```

### 不同的Prompt模式

#### 模式1: 基于路径推理（默认）

```python
# args.add_path = True
Reasoning Paths:
Barack Obama -> born_in -> Hawaii
Barack Obama -> born_in -> Hawaii
Barack Obama -> spouse -> Michelle Obama

Question:
Where was Barack Obama born?

Based on the reasoning paths, please answer the given question. 
Please keep the answer as simple as possible and only return answers.
```

#### 模式2: 无路径推理

```python
# args.add_path = False
Question:
Where was Barack Obama born?

Please answer the given question. Please keep the answer as 
simple as possible and only return answers.
```

#### 模式3: ROG风格Prompt

```python
# args.use_rog_prompt = True
Based on the reasoning paths, please answer the given question. 
Please keep the answer as simple as possible and return all the 
possible answers as a list.

Reasoning Paths:
Barack Obama -> born_in -> Hawaii
Barack Obama -> spouse -> Michelle Obama

Question:
Where was Barack Obama born?
```

### Prompt长度管理策略

**问题**：当候选路径很多时，prompt可能超过模型的最大context长度

**解决方案**：

```python
# 1. 计算可用空间
available_tokens = maximun_token - len(instruction) - len(question)

# 2. 随机打乱路径（保证公平性）
random.shuffle(list_of_paths)

# 3. 贪心添加路径
selected_paths = []
for path in list_of_paths:
    if token_count(selected_paths + [path]) <= available_tokens:
        selected_paths.append(path)
    else:
        break

# 4. 使用截断后的路径
context = "\n".join(selected_paths)
```

**优势**：
- ✅ 保证prompt不会超长导致错误
- ✅ 随机选择路径，不偏向某些路径
- ✅ 尽可能利用context空间

---

## 完整数据流示例

### 示例问题："Who is the prime minister of Ethiopia?"

#### 输入数据（Stage 1 + 原始数据）

```python
# 原始问题数据
{
    'id': 'WebQTest-5678',
    'question': 'who is the prime minister of ethiopia?',
    'q_entity': ['Ethiopia'],
    'a_entity': ['Hailemariam Desalegn'],
    'answer': ['Hailemariam Desalegn']
}

# Stage 1生成的候选路径
{
    'id': 'WebQTest-5678',
    'prediction': [
        'Ethiopia -> government.governmental_jurisdiction.governing_officials -> m.0l0j4x3 -> government.government_position_held.office_holder -> Hailemariam Desalegn',
        'Ethiopia -> government.governmental_jurisdiction.governing_officials -> m.0l0j4x3 -> government.government_position_held.office_holder -> Hailemariam Desalegn',
        'Ethiopia -> location.location.contains -> Addis Ababa',
        'Ethiopia -> government.governmental_jurisdiction.governing_officials -> m.0abc123 -> government.government_position_held.office_holder -> Meles Zenawi'
    ]
}
```

#### Prompt构建

```
Reasoning Paths:
Ethiopia -> government.governmental_jurisdiction.governing_officials -> m.0l0j4x3 -> government.government_position_held.office_holder -> Hailemariam Desalegn
Ethiopia -> location.location.contains -> Addis Ababa
Ethiopia -> government.governmental_jurisdiction.governing_officials -> m.0abc123 -> government.government_position_held.office_holder -> Meles Zenawi

Question:
who is the prime minister of ethiopia?

Based on the reasoning paths, please answer the given question. Please keep the answer as simple as possible and only return answers.
```

#### LLM推理过程

```
LLM内部思考：
1. 问题问的是"谁是埃塞俄比亚总理"
2. 分析路径：
   - 路径1: 指向 "Hailemariam Desalegn"，关系是政府官员
   - 路径2: 指向 "Addis Ababa"（首都），与问题无关
   - 路径3: 指向 "Meles Zenawi"，也是政府官员
3. 两个候选答案：Hailemariam Desalegn 和 Meles Zenawi
4. 路径1出现了2次（原始有重复），更可信
5. 最终答案：Hailemariam Desalegn
```

#### LLM输出

```python
prediction = "Hailemariam Desalegn"
```

#### 保存结果

```json
{
    "id": "WebQTest-5678",
    "question": "who is the prime minister of ethiopia?",
    "prediction": "Hailemariam Desalegn",
    "ground_truth": ["Hailemariam Desalegn"],
    "input": "Reasoning Paths:\nEthiopia -> ..."
}
```

#### 评估

```python
# 匹配成功
normalize("Hailemariam Desalegn") == normalize("Hailemariam Desalegn")
→ Accuracy = 1.0, Hit = 1, F1 = 1.0
```

---

## 与Stage 1的协作关系

### 完整的协作流程

```
┌─────────────────────────────────────────────────────────────────┐
│                        两阶段协作                                │
└─────────────────────────────────────────────────────────────────┘

输入: 问题 + 知识图谱
         │
         ▼
┌─────────────────────────┐
│   Stage 1: 图约束解码   │  ← 专注于"找"
│   KG-specialized LLM    │
└─────────────────────────┘
         │
         │ 输出: 候选路径列表
         │ [路径1, 路径2, ..., 路径N]
         │
         ▼
┌─────────────────────────┐
│   Stage 2: 图归纳推理   │  ← 专注于"判"
│   通用LLM (GPT)         │
└─────────────────────────┘
         │
         │ 输出: 最终答案
         ▼
      Answer
```

### 信息流动

| 阶段        | 输入             | 处理                    | 输出                            | 传递给下一阶段                  |
| ----------- | ---------------- | ----------------------- | ------------------------------- | ------------------------------- |
| **Stage 1** | 问题<br>KG       | Beam Search<br>约束解码 | 10-30条候选路径<br>多个假设答案 | `predicted_paths`<br>(路径列表) |
| **Stage 2** | 问题<br>候选路径 | LLM归纳推理             | 1个最终答案                     | 无（终点）                      |

### 两阶段的互补性

| 方面         | Stage 1的作用          | Stage 2的作用            | 互补效果           |
| ------------ | ---------------------- | ------------------------ | ------------------ |
| **路径生成** | 在KG中找到所有可能路径 | 不生成新路径             | 保证路径来自真实KG |
| **答案筛选** | 生成多个候选答案       | 从候选中选出最佳答案     | 结合覆盖度和准确性 |
| **推理能力** | 结构化推理（图遍历）   | 语义推理（自然语言理解） | 结合结构和语义     |
| **可解释性** | 提供推理路径（过程）   | 给出最终答案（结果）     | 完整的推理链条     |
| **幻觉控制** | 100%基于KG，零幻觉     | 可能产生幻觉             | Stage1约束限制幻觉 |

### 为什么不用一个阶段？

#### 方案A：只用KG-specialized LLM
```
问题: 谁是埃塞俄比亚总理?
      ↓
KG-specialized LLM (单独)
      ↓
输出: Ethiopia -> ... -> Hailemariam Desalegn
      Ethiopia -> ... -> Meles Zenawi
      Ethiopia -> ... -> Addis Ababa

❌ 问题: 生成了多个答案，不知道哪个是最终答案
```

#### 方案B：只用通用LLM
```
问题: 谁是埃塞俄比亚总理?
      ↓
通用LLM (GPT-4)
      ↓
输出: Hailemariam Desalegn

❌ 问题: 可能产生幻觉，答案不一定基于KG
❌ 问题: 没有推理路径，无法验证
❌ 问题: 知识可能过时
```

#### 方案C：两阶段结合（GCR）
```
问题: 谁是埃塞俄比亚总理?
      ↓
Stage 1: KG-specialized LLM
      ↓
候选路径（基于KG，零幻觉）:
  - Ethiopia -> ... -> Hailemariam Desalegn
  - Ethiopia -> ... -> Meles Zenawi
      ↓
Stage 2: 通用LLM
      ↓
最终答案: Hailemariam Desalegn

✅ 优势: 路径来自KG，答案由强LLM筛选
✅ 优势: 兼具忠实性和推理能力
✅ 优势: 有推理路径可追溯
```

---

## 实际应用场景

### 1. 知识图谱问答（KGQA）

**场景**：基于Freebase、Wikidata等大规模KG回答复杂问题

**示例**：
```
问题: "Who directed the movie that won Best Picture in 2010?"

Stage 1 生成路径:
- Best Picture (2010) -> awarded_to -> The Hurt Locker
- The Hurt Locker -> directed_by -> Kathryn Bigelow

Stage 2 归纳推理:
→ 答案: Kathryn Bigelow
```

**优势**：
- 路径来自真实KG，准确可靠
- LLM理解复杂的多跳推理
- 可以处理间接问题

### 2. 医疗问答系统

**场景**：基于医疗知识图谱回答疾病、药物相关问题

**示例**：
```
问题: "What medications can treat Type 2 Diabetes?"

Stage 1 生成路径:
- Type 2 Diabetes -> treated_by -> Metformin
- Type 2 Diabetes -> treated_by -> Insulin
- Type 2 Diabetes -> treated_by -> Glipizide

Stage 2 归纳推理:
→ 答案: Metformin, Insulin, Glipizide
```

**优势**：
- 零幻觉：所有药物来自医疗KG
- 可追溯：可以查看推理路径
- 安全性高：不会推荐KG外的药物

### 3. 金融风控

**场景**：基于企业关系图谱进行风险分析

**示例**：
```
问题: "Is Company A related to any sanctioned entities?"

Stage 1 生成路径:
- Company A -> subsidiary_of -> Company B
- Company B -> shareholder -> Person X
- Person X -> sanctioned_by -> US Treasury

Stage 2 归纳推理:
→ 答案: Yes, Company A is indirectly related to 
Person X who is on the US sanctions list.
```

**优势**：
- 发现隐藏的多跳关系
- 提供完整的关系链条
- 帮助合规决策

### 4. 学术研究检索

**场景**：基于学术引用图谱找到相关论文

**示例**：
```
问题: "What papers are related to Graph Neural Networks 
for drug discovery?"

Stage 1 生成路径:
- GNN paper A -> application -> Drug Discovery
- GNN paper A -> cited_by -> Paper B
- Paper B -> topic -> Molecular Property Prediction

Stage 2 归纳推理:
→ 答案: Paper A, Paper B (with reasoning about relevance)
```

---

## 性能优化与最佳实践

### 1. 路径去重（remove_dup_path）

**问题**：Stage 1可能生成重复路径（Beam Search的不同分支）

**解决**：
```python
dataset = merge_path_result(
    dataset, paths_datasets, 
    remove_dup_path=True  # 启用去重
)
```

**效果**：
```python
# 去重前
predicted_paths = [
    'A -> r1 -> B',
    'A -> r1 -> B',  # 重复
    'A -> r1 -> B',  # 重复
    'A -> r2 -> C'
]

# 去重后
predicted_paths = [
    'A -> r1 -> B',
    'A -> r2 -> C'
]
```

**优势**：
- 减少prompt长度
- 提高LLM推理效率
- 避免答案偏向重复路径

### 2. 空路径过滤（filter_empty）

**问题**：有些问题在KG中找不到路径

**解决**：
```python
dataset = merge_path_result(
    dataset, paths_datasets, 
    filter_empty=True  # 过滤空路径
)
```

**效果**：
```python
# 过滤前: 1000个问题
# 过滤后: 850个问题（150个没有路径被过滤）
```

**适用场景**：
- 只关注有路径的问题
- 评估Stage 1的路径生成能力
- 排除KG覆盖不足的问题

### 3. 并行处理（-n参数）

**问题**：逐个推理速度慢

**解决**：
```bash
python workflow/predict_final_answer.py \
  --model_name gpt-3.5-turbo \
  -n 10  # 10个并行线程
```

**效果**：
- 单线程：100个问题 = 10分钟
- 10线程：100个问题 = 1-2分钟

**注意事项**：
- API限流：注意OpenAI的rate limit
- 成本控制：并行会增加API调用速度
- 错误处理：需要处理并发错误

### 4. Prompt工程

**技巧1：明确指令**
```python
# 不好
"Answer the question."

# 好
"Based on the reasoning paths, please answer the given question. 
Please keep the answer as simple as possible and only return answers."
```

**技巧2：格式要求**
```python
# 要求每个答案一行
each_line = True
instruction += " Please return each answer in a new line."
```

**技巧3：使用示例（Few-shot）**
```python
# 在instruction中添加示例
instruction = """Based on the reasoning paths, answer the question.

Example:
Question: Where is Paris?
Reasoning Paths: Paris -> located_in -> France
Answer: France

Now answer the following:
"""
```

---

## 命令行使用

### 基本命令

```bash
python workflow/predict_final_answer.py \
  --data_path rmanluo \
  --d RoG-webqsp \
  --split test \
  --model_name gpt-3.5-turbo \
  --reasoning_path results/GenPaths/RoG-webqsp/.../predictions.jsonl \
  --add_path True \
  -n 10
```

### 参数说明

| 参数                | 说明            | 默认值       | 示例                           |
| ------------------- | --------------- | ------------ | ------------------------------ |
| `--data_path`       | 数据集路径      | `rmanluo`    | HuggingFace数据集ID            |
| `--d`               | 数据集名称      | `RoG-webqsp` | `RoG-webqsp`, `RoG-cwq`        |
| `--split`           | 数据集划分      | `test`       | `train`, `test`, `dev`         |
| `--model_name`      | LLM模型         | `gpt2`       | `gpt-3.5-turbo`, `gpt-4o-mini` |
| `--reasoning_path`  | Stage 1输出路径 | 必需         | `.../predictions.jsonl`        |
| `--add_path`        | 是否添加路径    | `False`      | `True` 启用Stage 2             |
| `-n`                | 并行线程数      | `1`          | `10`（适合API调用）            |
| `--remove_dup_path` | 去重路径        | `True`       | `True`/`False`                 |
| `--filter_empty`    | 过滤空路径      | `False`      | `True` 仅评估有路径的问题      |

### 完整示例脚本

```bash
#!/bin/bash

DATA_PATH="rmanluo"
DATA_LIST="RoG-webqsp RoG-cwq"
SPLIT="test"

MODEL_NAME=gpt-3.5-turbo
N_THREAD=10

for DATA in ${DATA_LIST}; do
  REASONING_PATH="results/GenPaths/${DATA}/GCR-Meta-Llama-3.1-8B-Instruct/test/zero-shot-group-beam-k10/predictions.jsonl"

  python workflow/predict_final_answer.py \
    --data_path ${DATA_PATH} \
    --d ${DATA} \
    --split ${SPLIT} \
    --model_name ${MODEL_NAME} \
    --reasoning_path ${REASONING_PATH} \
    --add_path True \
    -n ${N_THREAD}
done
```

---

## 总结

### Graph Inductive Reasoning 的核心价值

1. **归纳推理**：从多条候选路径中归纳出最可靠的答案
2. **语义理解**：利用通用LLM的强大自然语言理解能力
3. **答案聚合**：处理多样化的候选答案，输出单一最佳答案
4. **可扩展性**：支持任何通用LLM（GPT、Claude、Llama等）
5. **零训练成本**：无需微调，直接使用现成LLM

### 与Stage 1的完美配合

```
Stage 1 (图约束解码)        +    Stage 2 (图归纳推理)
─────────────────────────────────────────────────────
结构化推理                  +    语义推理
路径生成（发散）            +    答案筛选（收敛）
保证忠实性（零幻觉）        +    提升准确性
小模型（快速、便宜）        +    大模型（强大、贵）
KG专用                      +    通用能力
         ↓                             ↓
        ─────────────────────────────────
              完整的GCR推理系统
        ─────────────────────────────────
```

### 适用场景

✅ **最适合**：
- 知识图谱问答（KGQA）
- 需要可解释推理的场景
- 对幻觉零容忍的领域（医疗、金融、法律）
- 有预算调用API的应用

❌ **不适合**：
- 实时性要求极高的场景（API调用有延迟）
- 预算非常有限（API调用成本）
- 不需要推理路径的简单问答

---

## 附录：输入输出完整示例

### 完整输入文件示例（predictions.jsonl from Stage 1）

```jsonl
{"id": "WebQTest-1234", "question": "where was barack obama born?", "prediction": ["Barack Obama -> born_in -> Hawaii", "Barack Obama -> born_in -> Hawaii", "Barack Obama -> spouse -> Michelle Obama"], "ground_truth": ["Hawaii"], "ground_truth_paths": ["Barack Obama -> born_in -> Hawaii"]}
{"id": "WebQTest-5678", "question": "who is the prime minister of ethiopia?", "prediction": ["Ethiopia -> government.governmental_jurisdiction.governing_officials -> m.0l0j4x3 -> government.government_position_held.office_holder -> Hailemariam Desalegn"], "ground_truth": ["Hailemariam Desalegn"], "ground_truth_paths": ["Ethiopia -> government.governmental_jurisdiction.governing_officials -> m.0l0j4x3 -> government.government_position_held.office_holder -> Hailemariam Desalegn"]}
```

### 完整输出文件示例（predictions.jsonl from Stage 2）

```jsonl
{"id": "WebQTest-1234", "question": "where was barack obama born?", "prediction": "Hawaii", "ground_truth": ["Hawaii"], "input": "Reasoning Paths:\nBarack Obama -> born_in -> Hawaii\n\nQuestion:\nwhere was barack obama born?\n\nBased on the reasoning paths, please answer the given question. Please keep the answer as simple as possible and only return answers."}
{"id": "WebQTest-5678", "question": "who is the prime minister of ethiopia?", "prediction": "Hailemariam Desalegn", "ground_truth": ["Hailemariam Desalegn"], "input": "Reasoning Paths:\nEthiopia -> government.governmental_jurisdiction.governing_officials -> m.0l0j4x3 -> government.government_position_held.office_holder -> Hailemariam Desalegn\n\nQuestion:\nwho is the prime minister of ethiopia?\n\nBased on the reasoning paths, please answer the given question. Please keep the answer as simple as possible and only return answers."}
```

### 评估结果文件示例（eval_result.txt）

```
Accuracy: 67.8 Hit: 72.5 F1: 70.2 Precision: 68.9 Recall: 71.6
```

---

**文档版本**: v1.0  
**生成日期**: 2026-01-08  
**作者**: AI Analysis