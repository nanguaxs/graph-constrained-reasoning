"""GRPO 训练启动脚本"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
import os

from config import GRPOConfig
from dataset import PathGenerationDataset
from reward import PathRewardCalculator
from grpo_trainer import GRPOTrainer

def main():
    # 加载配置
    config = GRPOConfig()

    # 加载模型和分词器
    print(f"加载模型: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    # 添加特殊 token
    special_tokens = {"additional_special_tokens": ["<PATH>", "</PATH>"]}
    tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))

    # 配置 LoRA
    print("配置 LoRA")
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 加载数据集
    print(f"加载数据集: {config.data_path}")
    train_dataset = PathGenerationDataset(
        config.data_path,
        config.train_split,
        tokenizer,
        config.index_path_length,
        config.undirected
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn
    )

    # 初始化奖励计算器
    print(f"使用嵌入模型 API: {config.embedding_api_url}")
    reward_calculator = PathRewardCalculator(config.embedding_api_url)

    # 初始化优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    # 初始化训练器
    trainer = GRPOTrainer(model, tokenizer, reward_calculator, optimizer, config)

    # 开始训练
    print("开始训练")
    os.makedirs(config.output_dir, exist_ok=True)
    trainer.train(train_loader, config.num_epochs)

    # 保存模型
    print(f"保存模型到: {config.output_dir}")
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    print("训练完成！")

if __name__ == "__main__":
    main()
