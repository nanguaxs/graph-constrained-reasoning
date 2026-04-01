from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from types import SimpleNamespace
from typing import Any


@dataclass
class InteractiveClientConfig:
    model_name: str = "Qwen_Qwen3.5-0.8B"
    model_path: str = "offline_assets/models/Qwen_Qwen3.5-0.8B"
    global_kg_file: str = "interactive_client/KG.json"
    dataset_file: str | None = "offline_assets/datasets/COKG_QA/test.jsonl"

    prompt_mode: str = "zero-shot"
    index_path_length: int = 2
    subgraph_hops: int = 2
    max_subgraph_triples: int | None = 1500
    max_paths_in_trie: int | None = 2048
    undirected: bool = False

    dtype: str = "fp16"
    quant: str = "none"
    attn_implementation: str = "sdpa"
    generation_mode: str = "beam"
    k: int = 3
    max_new_tokens: int = 128
    maximun_token: int = 4096
    chat_model: bool = True
    use_assistant_model: bool = False
    assistant_model_path: str | None = None

    entity_match_mode: str = "hybrid"
    max_entity_candidates: int = 5
    max_selected_entities: int = 1
    min_entity_similarity: float = 0.35
    interactive_entity_selection: bool = True

    encoding: str = "utf-8"
    seed: int = 42

    @classmethod
    def from_json(cls, config_path: str | Path | None) -> "InteractiveClientConfig":
        if config_path is None:
            return cls()

        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with path.open("r", encoding="utf-8") as handle:
            raw_config = json.load(handle)

        valid_fields = {field.name for field in fields(cls)}
        filtered_config = {
            key: value
            for key, value in raw_config.items()
            if key in valid_fields
        }
        return cls(**filtered_config)

    def update_from_mapping(self, overrides: dict[str, Any]) -> None:
        valid_fields = {field.name for field in fields(self)}
        for key, value in overrides.items():
            if key not in valid_fields or value is None:
                continue
            setattr(self, key, value)

    def resolve_path(self, value: str | None, repo_root: Path) -> Path | None:
        if value is None:
            return None
        path = Path(value)
        if path.is_absolute():
            return path
        return (repo_root / path).resolve()

    def resolved(self, repo_root: Path) -> "ResolvedInteractiveClientConfig":
        model_path = self.resolve_path(self.model_path, repo_root)
        global_kg_file = self.resolve_path(self.global_kg_file, repo_root)
        dataset_file = self.resolve_path(self.dataset_file, repo_root)
        assistant_model_path = self.resolve_path(self.assistant_model_path, repo_root)
        resolved_values = asdict(self)
        resolved_values["model_path"] = model_path
        resolved_values["global_kg_file"] = global_kg_file
        resolved_values["dataset_file"] = dataset_file
        resolved_values["assistant_model_path"] = assistant_model_path
        return ResolvedInteractiveClientConfig(**resolved_values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ResolvedInteractiveClientConfig(InteractiveClientConfig):
    model_path: Path
    global_kg_file: Path | None = None
    dataset_file: Path | None = None
    assistant_model_path: Path | None = None

    def to_model_args(self) -> SimpleNamespace:
        return SimpleNamespace(
            model_name=self.model_name,
            model_path=str(self.model_path),
            maximun_token=self.maximun_token,
            max_new_tokens=self.max_new_tokens,
            dtype=self.dtype,
            quant=self.quant,
            attn_implementation=self.attn_implementation,
            generation_mode=self.generation_mode,
            k=self.k,
            chat_model=self.chat_model,
            use_assistant_model=self.use_assistant_model,
            assistant_model_path=(
                None if self.assistant_model_path is None else str(self.assistant_model_path)
            ),
        )
