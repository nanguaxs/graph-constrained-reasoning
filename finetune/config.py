"""训练配置"""
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

    # GRPO 配置
    num_generations: int = 8
    max_new_tokens: int = 128
    temperature: float = 1.1
    kl_penalty_beta: float = 0.04
    max_resample_rounds: int = 4

    # 训练配置
    learning_rate: float = 2e-6
    num_epochs: int = 1
    batch_size: int = 1
    gradient_accumulation_steps: int = 8

    # 数据配置COKG_QA\threehop
    #data_path: str = "offline_assets/datasets/onedata/train_dataset.json"
    data_path: str = "COKG_QA/threehop"
    train_split: str = "train"
    index_path_length: int = 3
    undirected: bool = False

    # 奖励配置
    embedding_api_url: str = "https://yunwu.ai/v1/embeddings"
    embedding_model_name: str = "text-embedding-3-large"
    embedding_api_key: str = "sk-5j4c26Aw7VS26K8PWz7Ayqp5zsjWo1J2krY4vzTiMDBiXlJ2"
    # 保存配置
    output_dir: str = None  # 将在 __post_init__ 中根据模型名称设置
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

        # 根据模型名称设置输出目录
        if self.output_dir is None:
            model_name = self.model_name.split("/")[-1]
            self.output_dir = f"finetune/output/{model_name}"

