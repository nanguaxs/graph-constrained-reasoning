# 20260206工作报告



## KG_QA_Dataset 测试结果报告

## 1. 测试概述

本报告总结了在自建知识图谱问答数据集(kg_qa_dataset)上的测试结果，对比了两个不同规模的语言模型在图约束推理任务中的表现。

**测试日期**: 2026年1月**数据集**: kg_qa_dataset (自建)
**评估模型**: gpt-4o，qwen3-235b

## 2. 测试模型

| 模型名称              | 参数规模 | 测试环境 |
| --------------------- | -------- | -------- |
| Qwen2-0.5B-Instruct   | 0.5B     | 本地机器 |
| Llama-3.1-8B-Instruct | 8B       | 服务器   |

## 3. 测试流程说明

本测试分为两个阶段：

1. **GenPaths 阶段**: 使用 图谱专用小模型 生成候选推理路径
2. **KGQA 阶段**: 使用 GPT-4o 或 和qwen3-235b 基于生成的路径进行答案推理

## 4. GenPaths 阶段测试结果

### 4.1 GCR-Qwen2-0.5B-Instruct 路径生成结果

| K值  | Accuracy (%) | Hit (%) | Answer F1 (%) | Answer Precision (%) | Answer Recall (%) | Path F1 (%) | Path Precision (%) | Path Recall (%) |
| ---- | ------------ | ------- | ------------- | -------------------- | ----------------- | ----------- | ------------------ | --------------- |
| k=3  | 31.0         | 31.0    | 1.42          | 1.17                 | 2.0               | 29.65       | 29.08              | 31.0            |
| k=5  | 35.0         | 35.0    | 3.30          | 2.14                 | 7.5               | 30.54       | 29.23              | 35.0            |
| k=10 | 37.0         | 37.0    | 5.69          | 3.25                 | 24.0              | 25.62       | 21.55              | 37.0            |



### 4.2 GCR-Llama-3.1-8B-Instruct 路径生成结果

| K值  | Accuracy (%) | Hit (%) | Answer F1 (%) | Answer Precision (%) | Answer Recall (%) | Path F1 (%) | Path Precision (%) | Path Recall (%) |
| ---- | ------------ | ------- | ------------- | -------------------- | ----------------- | ----------- | ------------------ | --------------- |
| k=3  | 35.0         | 35.0    | 2.58          | 2.03                 | 3.8               | 33.21       | 32.45              | 35.0            |
| k=5  | 39.0         | 39.0    | 4.72          | 3.21                 | 9.2               | 34.87       | 33.56              | 39.0            |
| k=10 | 42.0         | 42.0    | 8.15          | 5.12                 | 28.5              | 28.93       | 24.78              | 42.0            |



## 5. KGQA 阶段测试结果

### 5.1 Qwen2-0.5B-Instruct 测试结果

| K值  | Accuracy (%) | Hit (%) | F1 Score (%) | Precision (%) | Recall (%) |
| ---- | ------------ | ------- | ------------ | ------------- | ---------- |
| k=3  | 34.0         | 34.0    | 31.92        | 30.92         | 34.0       |
| k=5  | 34.5         | 34.5    | 33.03        | 32.38         | 34.5       |
| k=10 | 34.4         | 34.4    | 26.21        | 23.99         | 34.4       |

**关键发现**:

- 最佳准确率出现在 k=5，达到 34.5%
- k=10 时 F1 和 Precision 显著下降，表明路径数量增加导致噪声增多
- Recall 与 Accuracy 保持一致，说明模型能够覆盖正确答案

### 5.2 Llama-3.1-8B-Instruct 测试结果

| K值  | Accuracy (%) | Hit (%) | F1 Score (%) | Precision (%) | Recall (%) |
| ---- | ------------ | ------- | ------------ | ------------- | ---------- |
| k=3  | 32.0         | 32.0    | 30.08        | 29.12         | 32.0       |
| k=5  | 34.0         | 34.0    | 32.54        | 31.79         | 34.0       |
| k=10 | 37.0         | 37.0    | 28.49        | 25.97         | 37.0       |

**关键发现**:

- 准确率随 k 值增加而提升，k=10 时达到最高 37.0%
- 相比 Qwen2-0.5B，在 k=10 时表现更优，提升 2.6 个百分点
- F1 和 Precision 在 k=10 时同样下降，但整体准确率更高





---

## 更换数据集

### 1.1 原有数据集存在的问题

在项目初期使用的自建数据集（kg_qa_dataset）存在以下局限性：

- **数据质量不高**: 知识图谱的关系种类较少，覆盖范围有限
- **问答对质量较低**: 生成的问答对缺乏多样性和复杂性
- **缺乏标准评估**: 没有标准的测试集进行对比验证
- **领域覆盖不足**: 数据集规模和领域覆盖度不够

### 1.2 选择COKG_QA数据集的原因

为了解决上述问题，我们选择了COKG_QA数据集作为新的测试基准：

- **领域专业性**: 基于医疗领域的中文知识图谱，具有较高的专业性
- **自带问答对**: 数据集包含高质量的人工标注问答对
- **多跳推理支持**: 包含一跳、两跳、三跳的推理数据，可以测试不同复杂度的推理能力
- **标准化评估**: 作为公开数据集，便于与其他方法进行对比

---

## 在新数据集上的工作

### 2.1 数据格式转换脚本开发

#### 2.1.1 格式差异分析

COKG_QA数据集的原始格式与GCR框架要求的格式存在显著差异：

**COKG_QA原始格式**:
- 知识图谱: JSONL格式，每行为 `[头实体, 关系, 尾实体, [类型]]`
- 问答对: TXT格式，每行为 `问题\t答案\t类型`
- 答案分隔: 使用 `##` 分隔多个答案

**GCR要求格式**:
```json
{
  "id": "样本ID",
  "question": "问题文本",
  "answer": ["答案1", "答案2"],
  "q_entity": ["问题实体"],
  "a_entity": ["答案实体"],
  "graph": [["头", "关系", "尾"], ...]
}
```

#### 2.1.2 转换脚本实现

开发了 `src/gendata/convert_cokg_qa.py` 脚本，实现以下核心功能：

**1. 多答案实体支持**
- 自动为每个答案实体查找最短路径
- 确保所有答案实体都包含在子图中
- 不会因为找到一个答案就停止搜索

**2. 最短路径查找算法**
- 使用BFS（广度优先搜索）算法
- 从问题实体到每个答案实体查找最短路径
- 最大搜索深度: 5跳
- 支持多个问题实体和多个答案实体的组合

**3. 智能子图构建策略**
```
第1步: 添加问题→答案的最短路径（核心路径）
第2步: 添加问题实体的2跳邻居（上下文信息）
第3步: 添加答案实体的2跳邻居（答案相关信息）
```

**4. 性能优化措施**
- 邻居数限制: 每个实体最多200个邻居
- 预先构建实体索引，加快查找速度
- 流式写入JSONL格式，节省内存

### 2.2 模型泛化性测试

#### 2.2.1 测试模型配置

**测试模型1: GCR-Qwen2-0.5B-Instruct（作者微调版本）**
- 基础模型: Qwen2-0.5B-Instruct
- 微调数据: 作者原始数据集（kg_qa_dataset）
- 微调方法: LoRA或全量微调
- 特点: 在原始数据集上训练，学习了生成 路径和`<PATH>` 标签的能力

**测试模型2: Qwen2.5-0.5B-Instruct（原生未微调）**
- 基础模型: Qwen2.5-0.5B-Instruct
- 微调状态: 未微调
- 特点: 通用语言模型，未经过图推理任务训练

#### 2.2.2 测试结果与问题分析

**问题1: 微调模型泛化能力不佳**

在GCR-Qwen2-0.5B-Instruct模型上测试COKG_QA数据集时，发现以下问题：

1. **生成不相关的英文内容**
   - 现象: 模型在中文医疗数据集上生成英文路径或实体
   - 原因: 原始训练数据可能包含英文数据集（如WebQSP、CWQ）
   - 影响: 生成的路径与中文医疗知识图谱不匹配

2. **生成非预期路径**
   - 现象: 生成的路径与用户查询不相关
   - 原因: 模型过拟合到原始数据集的实体和关系分布
   - 影响: 无法正确理解新领域的问题意图

3. **路径命中率低**
   - 数据: 路径命中率仅约40%（Path Recall）
   - 对比: 在原始数据集上可达到更高的命中率
   - 结论: 模型在新领域数据上泛化能力不足

**问题2: 原生模型无法生成约束路径**

在Qwen2.5-0.5B-Instruct原生模型上测试时：

1. **无法生成 `<PATH>` 标签**
   - 现象: 模型不知道需要生成 `<PATH></PATH>` 标签
   - 原因: 未经过图推理任务的微调训练
   - 影响: 图约束解码机制无法启动

2. **完全无约束的文本生成**
   - 现象: 模型生成自由文本，而非结构化路径
   - 原因: 约束推理只在 `<PATH></PATH>` 标签内部生效
   - 影响: 无法利用知识图谱结构进行约束推理

**测试结论**:
- 微调模型在新领域数据上存在严重的泛化问题
- 原生模型缺乏图推理任务的基本能力
- 需要在COKG_QA数据集上重新微调模型

---

### 2.3 多答案实体路径生成问题

在医疗知识图谱问答中，许多问题的答案包含多个实体，例如：

**示例问题**: "糖尿病的并发症有哪些？"

**答案实体**:
- 糖尿病肾病
- 糖尿病视网膜病变
- 糖尿病足
- 糖尿病神经病变
- 心血管疾病

**推理要求**:
- 需要找到从"糖尿病"到每个并发症的推理路径
- 每条路径可能经过不同的关系和中间实体
- 模型需要生成多条路径才能覆盖所有答案

#### 技术挑战

1. **路径数量控制**
   - 如何确定生成多少条路径（k值）
   - 路径过少: 无法覆盖所有答案实体
   - 路径过多: 引入噪声，降低精确度

2. **路径质量保证**
   - 确保每条路径都指向不同的答案实体
   - 避免生成重复或相似的路径
   - 保持路径的多样性和相关性

3. **评估指标设计**
   - 答案实体覆盖率: 生成的路径覆盖了多少答案实体
   - 路径准确率: 生成的路径是否正确连接问题和答案
   - 答案完整性: 是否找到了所有正确答案

#### 2.3.3 当前解决方案

**数据转换层面**（已实现）:

- `convert_cokg_qa.py` 脚本已支持多答案实体
- 使用BFS算法为每个答案实体查找最短路径
- 统计答案实体覆盖率，确保数据质量

**模型生成层面**（进行中）:
- 使用Group Beam Search生成多条候选路径
- 整合不同答案，输出实体列表

**评估指标**:
- Path Recall: 生成的路径中有多少覆盖了真实答案实体
- Answer Coverage: 答案实体的覆盖比例
- Answer F1: 综合考虑答案的精确率和召回率

---

# 知识图谱专用LLM微调流程说明

## 微调方法

项目支持两种微调方式：

### 1. LoRA微调（参数高效）

**配置示例：**

```bash
BATCH_SIZE=50
USE_PEFT=True
EPOCH=20
GRADIENT_CHECKPOINTING=False
GRADIENT_ACCUMULATION_STEPS=1
auto_find_batch_size=True
CONFIG="accelerate_configs/multi_gpu.yaml"
```



### 2. 全量微调（Full Fine-tuning）

**配置示例：**

```bash
BATCH_SIZE=4
USE_PEFT=False
EPOCH=3
GRADIENT_CHECKPOINTING=True
GRADIENT_ACCUMULATION_STEPS=16
auto_find_batch_size=False
CONFIG="accelerate_configs/deepspeed_zero3.yaml"
```

## 支持的模型

项目已测试以下模型：

### Qwen系列

```bash
MODEL_PATH=Qwen/Qwen2-0.5B-Instruct
ATTN_IMP=flash_attention_2
RESPONSE_TEMPLATE="<|im_start|>assistant"
```

### Llama系列

```bash
# Llama 2
MODEL_PATH=meta-llama/Llama-2-7b-chat-hf
RESPONSE_TEMPLATE="[/INST]"

# Llama 3.1
MODEL_PATH=meta-llama/Meta-Llama-3.1-8B-Instruct
RESPONSE_TEMPLATE="<|start_header_id|>assistant<|end_header_id|>"
```

**注意：** 不同模型需要配置对应的`RESPONSE_TEMPLATE`，用于标识助手回复的起始位置。

## 数据处理流程

### 数据格式

训练数据使用HuggingFace Datasets格式，存储在`data/shortest_path_index/`目录下。

**数据集字段：**

- `question`: 问题文本
- `q_entity`: 问题中的主题实体列表
- `a_entity`: 答案实体列表
- `ground_truth_paths`: 真实推理路径列表（非原始数据集中的数据，是根据头尾实体在图谱中查找最短路径得到的）

**路径格式：** 每条路径是三元组列表 `[(head, relation, tail), ...]`

### 数据预处理

代码位置：[workflow/finetune_kg_specialized_llm.py:154-185](workflow/finetune_kg_specialized_llm.py#L154-L185)

**处理步骤：**

1. **加载数据集**

   ```python
   data_list = [
       datasets.load_from_disk(data_path)
       for data_path in script_args.data_path_list
   ]
   dataset = datasets.concatenate_datasets(data_list)
   ```

2. **格式化输入**

   使用提示模板：

   ```
   Reasoning path is a sequence of triples in the KG that connects
   the topic entities in the question to answer entities.
   Given a question, please generate some reasoning paths in the KG
   starting from the topic entities to answer the question.
   
   # Question:
   {question}
   # Topic entities:
   {entities}
   ```

3. **路径转换**

   使用`path_to_string`函数（[src/utils/utils.py:34-44](src/utils/utils.py#L34-L44)）将路径转换为字符串：

   ```python
   def path_to_string(path: list) -> str:
       result = ""
       for i, p in enumerate(path):
           if i == 0:
               h, r, t = p
               result += f"{h} -> {r} -> {t}"
           else:
               _, r, t = p
               result += f" -> {r} -> {t}"
       return result.strip()
   ```

   **示例输出：**

   ```
   <PATH>实体A -> 关系1 -> 实体B -> 关系2 -> 实体C</PATH>
   ```

4. **构建训练样本**

   每条路径生成一个训练样本：

   ```python
   chat = [
       {"role": "user", "content": raw_input},
       {"role": "assistant", "content": response},
   ]
   final_input = tokenizer.apply_chat_template(
       chat, tokenize=False, add_generation_prompt=False
   )
   ```

5. **特殊Token**

   添加路径标记符：

   - `<PATH>`: 路径开始标记
   - `</PATH>`: 路径结束标记
   - `<PAD>`: 填充标记（如果模型没有）

### 数据采样

- 参数`n_path_per_sample=10`控制每个问题采样的路径数量
- 每条路径独立作为一个训练样本
- 空路径会被过滤掉



## 后续工作计划

1. **在COKG_QA数据集上微调模型**
   - 使用转换后的训练集进行微调
   - 测试LoRA和全量微调的效果
   - 对比不同规模模型的性能
2. **优化多答案路径生成**
   - 实现答案实体覆盖率评估
   - 调整k值选择策略
   - 改进路径去重和排序算法

