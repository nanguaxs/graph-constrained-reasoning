# 图约束路径生成 GRPO 微调

基于 GRPO（群体相对策略优化）的图约束路径生成微调方案。

## 核心特性

- **GRPO 算法**：无需独立 Critic，降低显存占用
- **图约束生成**：Rollout 阶段使用 Trie 约束，保证路径合法
- **智能奖励**：终点命中 + 语义相似度 + 绕路惩罚
- **LoRA 训练**：高效微调，A800 80GB 显存安全

## 文件结构

```
finetune/
├── config.py          # 训练配置
├── dataset.py         # 数据集处理
├── reward.py          # 奖励函数
├── grpo_trainer.py    # GRPO 训练器
├── train.py           # 训练启动脚本
└── README.md          # 使用说明
```

## 快速开始

### 1. 安装依赖

```bash
pip install transformers peft sentence-transformers datasets torch
```

### 2. 修改配置

编辑 `config.py` 中的参数：
- `model_name`: 基座模型（建议 7B/8B Instruct）
- `data_path`: 数据集路径
- `train_split`: 训练集划分
- `num_generations`: 每个问题生成路径数（建议 4-10）

### 3. 启动训练

```bash
cd finetune
python train.py
```

## 核心参数说明

**LoRA 配置**：
- `lora_r=64`, `lora_alpha=128`
- 目标模块：q/k/v/o/gate/up/down_proj

**GRPO 配置**：
- `num_generations=4`: 每组生成 4 条路径
- `temperature=0.8`: 保持多样性
- `kl_penalty_beta=0.04`: KL 散度惩罚系数

**训练配置**：
- `learning_rate=2e-6`: 低学习率避免 KL 爆炸
- `batch_size=1`, `gradient_accumulation_steps=8`

## 奖励函数

1. **终点命中**：+10.0
2. **语义相似度**：cos_sim × 3.0
3. **绕路惩罚**：超出预期跳数每跳 -0.5
