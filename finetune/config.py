"""训练配置。"""
from dataclasses import dataclass


@dataclass
class GRPOConfig:
    # 模型配置
    model_name: str = "offline_assets/models/Qwen3.5-0.8B"

    # LoRA 配置
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    target_modules: list = None

    # 生成配置
    generation_mode: str = "beam_sample"
    num_generations: int = 8
    num_beams: int = 8
    max_new_tokens: int = 128
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 0
    early_stopping: bool = True
    max_resample_rounds: int = 4

    # GRPO 配置
    kl_penalty_beta: float = 0.04

    # 日志配置
    log_level: str = "INFO"

    # 训练配置
    learning_rate: float = 2e-6
    num_epochs: int = 1
    batch_size: int = 1
    gradient_accumulation_steps: int = 8

    # 数据配置
    # data_path: str = "offline_assets/datasets/onedata/train_dataset.json"
    data_path: str = "COKG_QA/threehop"
    train_split: str = "train"
    index_path_length: int = 3
    undirected: bool = False

    # 奖励配置
    embedding_api_url: str = "https://yunwu.ai/v1/embeddings"
    embedding_model_name: str = "text-embedding-3-large"
    embedding_api_key: str = "sk-5j4c26Aw7VS26K8PWz7Ayqp5zsjWo1J2krY4vzTiMDBiXlJ2"

    # 保存配置
    output_dir: str = None
    save_steps: int = 500

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]

        self.generation_mode = str(self.generation_mode).strip().lower()
        self.log_level = str(self.log_level).strip().upper()

        valid_generation_modes = {"beam_search", "sampling", "beam_sample"}
        if self.generation_mode not in valid_generation_modes:
            raise ValueError(
                f"generation_mode 必须是 {sorted(valid_generation_modes)} 之一，当前为: {self.generation_mode}"
            )

        valid_log_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.log_level not in valid_log_levels:
            raise ValueError(f"log_level 必须是 {sorted(valid_log_levels)} 之一，当前为: {self.log_level}")

        if self.num_generations < 1:
            raise ValueError("num_generations 必须大于等于 1")
        if self.num_beams < 1:
            raise ValueError("num_beams 必须大于等于 1")
        if self.max_resample_rounds < 1:
            raise ValueError("max_resample_rounds 必须大于等于 1")
        if self.temperature <= 0:
            raise ValueError("temperature 必须大于 0")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p 必须在 (0, 1] 范围内")
        if self.top_k < 0:
            raise ValueError("top_k 不能为负数")

        if self.output_dir is None:
            model_name = self.model_name.split("/")[-1]
            self.output_dir = f"finetune/output/{model_name}"
