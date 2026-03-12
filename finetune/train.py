"""GRPO 训练启动脚本。"""
import os

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import GRPOConfig
from dataset import PathGenerationDataset
from grpo_trainer import GRPOTrainer
from logging_utils import configure_logging, get_logger
from reward import PathRewardCalculator


logger = get_logger("train")


def log_config_summary(config):
    logger.info(
        "训练配置: model=%s gpu_id=%s data=%s split=%s mode=%s num_generations=%s num_beams=%s num_beam_groups=%s diversity_penalty=%.3f temperature=%.3f top_p=%.3f top_k=%s semantic=%s log_level=%s",
        config.model_name,
        config.gpu_id,
        config.data_path,
        config.train_split,
        config.generation_mode,
        config.num_generations,
        config.num_beams,
        config.num_beam_groups,
        config.diversity_penalty,
        config.temperature,
        config.top_p,
        config.top_k,
        config.use_semantic_reward,
        config.log_level,
    )


def main():
    config = GRPOConfig()
    configure_logging(config.log_level)
    log_config_summary(config)

    logger.info("加载模型: %s", config.model_name)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    model_load_kwargs = {
        "dtype": torch.bfloat16,
        "trust_remote_code": True,
    }

    if config.gpu_id is None:
        model_load_kwargs["device_map"] = "auto"
        logger.info("未指定 gpu_id，使用 device_map=auto")
    else:
        if not torch.cuda.is_available():
            raise RuntimeError(f"gpu_id={config.gpu_id} 已设置，但当前环境未检测到 CUDA")
        if config.gpu_id >= torch.cuda.device_count():
            raise RuntimeError(
                f"gpu_id={config.gpu_id} 超出可用 GPU 数量范围，当前仅检测到 {torch.cuda.device_count()} 张 GPU"
            )

        torch.cuda.set_device(config.gpu_id)
        model_load_kwargs["device_map"] = {"": config.gpu_id}
        logger.info("指定使用 GPU: cuda:%s", config.gpu_id)

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        **model_load_kwargs,
    )

    special_tokens = {"additional_special_tokens": ["<PATH>", "</PATH>"]}
    tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))

    logger.info("配置 LoRA")
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    logger.info("加载数据集: %s", config.data_path)
    train_dataset = PathGenerationDataset(
        config.data_path,
        config.train_split,
        tokenizer,
        config.index_path_length,
        config.undirected,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
    )

    logger.info("使用嵌入模型 API: %s", config.embedding_api_url)
    logger.info("嵌入模型名称: %s", config.embedding_model_name)
    logger.info("启用语义监督: %s", config.use_semantic_reward)
    reward_calculator = PathRewardCalculator(
        config.embedding_api_url,
        config.embedding_model_name,
        config.embedding_api_key,
        config.use_semantic_reward,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    trainer = GRPOTrainer(model, tokenizer, reward_calculator, optimizer, config)

    logger.info("开始训练")
    os.makedirs(config.output_dir, exist_ok=True)
    trainer.train(train_loader, config.num_epochs)

    logger.info("保存模型到: %s", config.output_dir)
    model.save_pretrained(config.output_dir, save_embedding_layers=True)
    tokenizer.save_pretrained(config.output_dir)
    logger.info("训练完成")


if __name__ == "__main__":
    main()
