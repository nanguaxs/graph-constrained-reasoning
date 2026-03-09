# 图约束路径生�?GRPO 微调

基于 GRPO（群体相对策略优化）的图约束路径生成微调方案�?
## 核心特�?
- **GRPO 算法**：无需独立 Critic，降低显存占�?- **图约束生�?*：Rollout 阶段使用 Trie 约束，保证路径合�?- **智能奖励**：终点命�?+ 语义相似�?+ 绕路惩罚
- **LoRA 训练**：高效微调，A800 80GB 显存安全

## 文件结构

```
finetune/
├── config.py          # 训练配置
├── dataset.py         # 数据集处�?├── reward.py          # 奖励函数
├── grpo_trainer.py    # GRPO 训练�?├── train.py           # 训练启动脚本
└── README.md          # 使用说明
```

## 快速开�?
### 1. 安装依赖

```bash
pip install "transformers>=4.51.0" peft sentence-transformers datasets torch
```

### 2. 修改配置

编辑 `config.py` 中的参数�?- `model_name`: 基座模型（建�?`Qwen/Qwen3.5-0.8B`�?- `data_path`: 数据集路�?- `train_split`: 训练集划�?- `num_generations`: 每个问题生成路径数（建议 4-10�?
### 3. 启动训练

```bash
cd finetune
python train.py
```

## 核心参数说明

**LoRA 配置**�?- `lora_r=64`, `lora_alpha=128`
- 目标模块：q/k/v/o/gate/up/down_proj

**GRPO 配置**�?- `num_generations=4`: 每组生成 4 条路�?- `temperature=0.8`: 保持多样�?- `kl_penalty_beta=0.04`: KL 散度惩罚系数

**训练配置**�?- `learning_rate=2e-6`: 低学习率避免 KL 爆炸
- `batch_size=1`, `gradient_accumulation_steps=8`

## 奖励函数

1. **终点命中**�?10.0
2. **语义相似�?*：cos_sim × 3.0
3. **绕路惩罚**：超出预期跳数每�?-0.5

