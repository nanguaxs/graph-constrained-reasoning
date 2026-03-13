"""训练配置。"""
import os
from dataclasses import dataclass


@dataclass
class GRPOConfig:
    # 模型配置
    model_name: str = "offline_assets/models/Qwen3.5-0.8B"
    gpu_id: int | None = None

    # LoRA 配置
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    target_modules: list = None

    # 生成配置（全部模式通用）
    generation_mode: str = "sampling"#group——beam不可用
    num_generations: int = 12
    max_new_tokens: int = 256
    early_stopping: bool = True
    max_resample_rounds: int = 4

    # 生成配置（beam_search / beam_sample / group_beam_search）
    num_beams: int = 12

    # 生成配置（sampling / beam_sample）
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 0

    # 生成配置（group_beam_search）
    num_beam_groups: int = 4
    diversity_penalty: float = 0.4

    # GRPO 配置
    kl_penalty_beta: float = 0.04

    # 日志配置
    log_level: str = "DEBUG"

    # 训练配置
    learning_rate: float = 2e-6
    num_epochs: int = 2
    batch_size: int = 8
    gradient_accumulation_steps: int = 8

    # 数据配置
    data_path: str = "offline_assets/datasets/COKG_QA"
    #data_path: str = "COKG_QA/threehop"
    train_split: str = "train"
    index_path_length: int = 3
    undirected: bool = False

    # 奖励配置
    use_semantic_reward: bool = True
    embedding_api_url: str = "https://yunwu.ai/v1/embeddings"
    embedding_model_name: str = "text-embedding-3-large"
    embedding_api_key: str = "sk-5j4c26Aw7VS26K8PWz7Ayqp5zsjWo1J2krY4vzTiMDBiXlJ2"

    # 保存配置
    output_dir: str = None
    save_steps: int = 500

    @staticmethod
    def _normalize_leaf_name(path_like):
        normalized_name = str(path_like).rstrip("/\\")
        return os.path.basename(normalized_name) or normalized_name

    def _build_default_output_name(self):
        model_name = self._normalize_leaf_name(self.model_name)
        data_name = self._normalize_leaf_name(self.data_path)
        semantic_tag = "semantic_on" if self.use_semantic_reward else "semantic_off"
        parts = [
            model_name,
            data_name,
            self.train_split,
            self.generation_mode,
            f"pathlen_{self.index_path_length}",
            semantic_tag,
        ]
        return "__".join(parts)

    def _build_default_output_dir(self):
        return os.path.join(
            "finetune",
            "output",
            self._build_default_output_name(),
        )

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

        generation_mode_aliases = {
            "groupbeam": "group_beam_search",
            "group_beam": "group_beam_search",
        }
        self.generation_mode = generation_mode_aliases.get(self.generation_mode, self.generation_mode)

        valid_generation_modes = {"beam_search", "sampling", "beam_sample", "group_beam_search"}
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
        if self.num_beam_groups < 1:
            raise ValueError("num_beam_groups 必须大于等于 1")
        if self.diversity_penalty < 0:
            raise ValueError("diversity_penalty 不能为负数")
        if self.gpu_id is not None and self.gpu_id < 0:
            raise ValueError("gpu_id 不能为负数")
        if self.generation_mode == "group_beam_search":
            if self.num_beam_groups < 2:
                raise ValueError("group_beam_search 模式下 num_beam_groups 必须大于等于 2")
            if self.diversity_penalty <= 0:
                raise ValueError("group_beam_search 模式下 diversity_penalty 必须大于 0")

        if self.output_dir is None:
            self.output_dir = self._build_default_output_dir()
