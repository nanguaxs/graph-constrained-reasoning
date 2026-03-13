"""LoRA adapter 与基座模型合并脚本。"""
import os
from dataclasses import dataclass

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from logging_utils import configure_logging, get_logger


logger = get_logger("merge_adapter")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))


@dataclass
class MergeConfig:
    base_model_path: str = "offline_assets/models/Qwen3.5-0.8B"
    adapter_path: str = (
        "finetune/output/Qwen3.5-0.8B__COKG_QA__train__sampling__pathlen_3__semantic_on"
    )
    output_path: str | None = None
    dtype: str = "bf16"
    device_map: str | dict | None = "auto"
    trust_remote_code: bool = True
    safe_serialization: bool = True
    log_level: str = "INFO"

    def resolve_output_path(self) -> str:
        if self.output_path:
            return self.resolve_path(self.output_path)
        return self.resolve_path(f"{self.adapter_path}__merged")

    def resolve_path(self, path_value: str) -> str:
        if os.path.isabs(path_value):
            return path_value
        return os.path.join(REPO_ROOT, path_value)

    def resolve_dtype(self):
        dtype_map = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }
        normalized_dtype = str(self.dtype).strip().lower()
        if normalized_dtype not in dtype_map:
            raise ValueError(f"不支持的 dtype: {self.dtype}")
        return dtype_map[normalized_dtype]


def load_tokenizer(config: MergeConfig):
    tokenizer_source = config.resolve_path(config.adapter_path)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            trust_remote_code=config.trust_remote_code,
        )
        logger.info("从 adapter 目录加载 tokenizer: %s", tokenizer_source)
        return tokenizer
    except Exception:
        tokenizer_source = config.resolve_path(config.base_model_path)
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            trust_remote_code=config.trust_remote_code,
        )
        logger.warning("adapter 目录缺少 tokenizer，回退到基座模型目录: %s", tokenizer_source)
        return tokenizer


def main():
    config = MergeConfig()
    configure_logging(config.log_level)

    output_path = config.resolve_output_path()
    base_model_path = config.resolve_path(config.base_model_path)
    adapter_path = config.resolve_path(config.adapter_path)

    logger.info("基座模型路径: %s", base_model_path)
    logger.info("Adapter 路径: %s", adapter_path)
    logger.info("输出路径: %s", output_path)

    tokenizer = load_tokenizer(config)

    logger.info("加载基座模型")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        dtype=config.resolve_dtype(),
        device_map=config.device_map,
        trust_remote_code=config.trust_remote_code,
    )

    vocab_size = base_model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) != vocab_size:
        logger.info("调整词表大小: %s -> %s", vocab_size, len(tokenizer))
        base_model.resize_token_embeddings(len(tokenizer))

    logger.info("加载 adapter 并执行 merge")
    peft_model = PeftModel.from_pretrained(base_model, adapter_path)
    merged_model = peft_model.merge_and_unload()

    os.makedirs(output_path, exist_ok=True)
    logger.info("保存合并后的完整模型")
    merged_model.save_pretrained(output_path, safe_serialization=config.safe_serialization)
    tokenizer.save_pretrained(output_path)
    if getattr(merged_model, "generation_config", None) is not None:
        merged_model.generation_config.save_pretrained(output_path)

    logger.info("合并完成: %s", output_path)


if __name__ == "__main__":
    main()
