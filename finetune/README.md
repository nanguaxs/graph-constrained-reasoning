# 图约束路径生成 GRPO 微调

基于 GRPO（群体相对策略优化）的图约束路径生成微调方案。

## 核心特性

- **GRPO 算法**：无需独立 Critic，降低显存占用
- **图约束生成**：Rollout 阶段使用 Trie 约束，保证路径合法
- **智能奖励**：终点命中 + 语义相似度 + 绕路惩罚
- **LoRA 训练**：高效微调，A800 80GB 显存安全
- **日志分级**：可在配置中切换 `DEBUG` / `INFO` / `WARNING`
- **多种生成模式**：支持 `beam_search`、`sampling`、`beam_sample`

## 文件结构

```text
finetune/
├── config.py          # 训练配置
├── dataset.py         # 数据集处理
├── reward.py          # 奖励函数
├── grpo_trainer.py    # GRPO 训练器
├── train.py           # 训练启动脚本
├── sft_trainer.py     # SFT 训练器
├── sft_train.py       # SFT 启动脚本
├── logging_utils.py   # 日志工具
└── README.md          # 使用说明
```

## 快速开始

### 1. 安装依赖

```bash
pip install transformers peft sentence-transformers datasets torch requests
```

### 2. 修改配置

编辑 `config.py` 中的参数：
- `log_level`: 日志级别，默认 `INFO`
- `generation_mode`: 生成模式，可选 `beam_search`、`sampling`、`beam_sample`
- `num_generations`: 每个问题保留的路径数
- `num_beams`: 束搜索/束采样使用的 beam 数
- `temperature` / `top_p` / `top_k`: 采样相关参数

### 3. 启动训练

```bash
cd finetune
python train.py
```

### 4. 合并 LoRA adapter 与基座模型

编辑 `merge_adapter.py` 顶部的 `MergeConfig` 路径配置，然后运行：

```bash
cd finetune
python merge_adapter.py
```

默认会输出到 `adapter_path + "__merged"`，生成可直接用于推理脚本的完整模型目录。

## SFT 对比训练

如果想用标准监督微调直接学习 GT path，可编辑 `config.py` 中 `SFTConfig` 的参数，然后运行：

```bash
cd finetune
python sft_train.py
```

说明：
- `SFTPathDataset` 会把每条 GT path 展开成一个监督样本
- 默认 `max_ground_paths_per_sample=None`，即使用样本中的全部 GT path
- 若想和当前 GRPO 中“每题最多注入 2 条 GT path”的设置更接近，可将 `max_ground_paths_per_sample=2`
