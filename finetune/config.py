"""训练配置"""
from dataclasses import dataclass

@dataclass
class GRPOConfig:
    # 模型配置
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"

    # LoRA 配置
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    target_modules: list = None

    # GRPO 配置
    num_generations: int = 4
    max_new_tokens: int = 128
    temperature: float = 0.8
    kl_penalty_beta: float = 0.04

    # 训练配置
    learning_rate: float = 2e-6
    num_epochs: int = 3
    batch_size: int = 1
    gradient_accumulation_steps: int = 8

    # 数据配置
    data_path: str = "rmanluo/RoG-webqsp"
    train_split: str = "train[:1000]"
    index_path_length: int = 2
    undirected: bool = False

    # 奖励配置
    embedding_api_url: str = "http://localhost:8000/embed"
    expected_hops: int = 2

    # 保存配置
    output_dir: str = "./output"
    save_steps: int = 500

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
