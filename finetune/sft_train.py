"""SFT 训练启动脚本。"""
import os

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import SFTConfig
from dataset import SFTPathDataset
from logging_utils import configure_logging, get_logger
from sft_trainer import SFTTrainer


logger = get_logger("sft_train")


def log_config_summary(config):
    logger.info(
        "SFT 配置: model=%s gpu_id=%s data=%s split=%s batch_size=%s grad_acc=%s lr=%s pathlen=%s max_gt_paths=%s log_level=%s",
        config.model_name,
        config.gpu_id,
        config.data_path,
        config.train_split,
        config.batch_size,
        config.gradient_accumulation_steps,
        config.learning_rate,
        config.index_path_length,
        config.max_ground_paths_per_sample,
        config.log_level,
    )


def main():
    config = SFTConfig()
    configure_logging(config.log_level)
    log_config_summary(config)

    logger.info("加载模型: %s", config.model_name)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

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
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

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

    logger.info("加载 SFT 数据集: %s", config.data_path)
    train_dataset = SFTPathDataset(
        config.data_path,
        config.train_split,
        tokenizer,
        config.index_path_length,
        config.undirected,
        config.max_ground_paths_per_sample,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
    )
    logger.info("SFT 训练样本数: %s", len(train_dataset))

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    trainer = SFTTrainer(model, optimizer, config)

    logger.info("开始 SFT 训练")
    os.makedirs(config.output_dir, exist_ok=True)
    trainer.train(train_loader, config.num_epochs)

    logger.info("保存模型到: %s", config.output_dir)
    model.save_pretrained(config.output_dir, save_embedding_layers=True)
    tokenizer.save_pretrained(config.output_dir)
    logger.info("SFT 训练完成")


if __name__ == "__main__":
    main()
