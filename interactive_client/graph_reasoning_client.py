from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import json
from pathlib import Path
import random
import sys
from typing import Any

from transformers import StoppingCriteriaList


if __package__ is None or __package__ == "":
    REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from interactive_client.config import InteractiveClientConfig
    from interactive_client.entity_extractor import EntityCandidate, EntityExtractor
    from interactive_client.graph_store import KnowledgeGraphStore
else:
    REPO_ROOT = Path(__file__).resolve().parents[1]
    from .config import InteractiveClientConfig
    from .entity_extractor import EntityCandidate, EntityExtractor
    from .graph_store import KnowledgeGraphStore

import src.utils as utils
from src.graph_constrained_decoding import GraphConstrainedDecoding, PathEndStoppingCriteria
from src.llms import get_registed_model
from src.qa_prompt_builder import ChinesePathGenerationWithAnswerPromptBuilder


class GraphReasoningClient:
    def __init__(self, config: InteractiveClientConfig) -> None:
        self.repo_root = REPO_ROOT
        self.config = config.resolved(self.repo_root)
        self._validate_config()
        self._seed_everything(self.config.seed)
        self.kg_source = self.config.global_kg_file or self.config.dataset_file

        print("[startup] loading knowledge graph...")
        self.graph_store = KnowledgeGraphStore.from_file(
            self.kg_source,
            encoding=self.config.encoding,
            undirected=self.config.undirected,
        )
        self.entity_extractor = EntityExtractor(self.graph_store.entities)
        print(
            f"[startup] graph ready: source={self.kg_source} triples={len(self.graph_store.triples)} entities={len(self.graph_store.entities)}"
        )

        print("[startup] loading model...")
        model_class = get_registed_model(self.config.model_name)
        self.model = model_class(self.config.to_model_args())
        self.model.prepare_for_inference()
        self.prompt_builder = ChinesePathGenerationWithAnswerPromptBuilder(
            self.model.tokenizer,
            self.config.prompt_mode,
            undirected=self.config.undirected,
            index_path_length=self.config.index_path_length,
        )
        print("[startup] model ready.")

    def _validate_config(self) -> None:
        if not self.config.model_path.exists():
            raise FileNotFoundError(f"Model path not found: {self.config.model_path}")
        if self.config.global_kg_file is not None and not self.config.global_kg_file.exists():
            raise FileNotFoundError(f"Global KG file not found: {self.config.global_kg_file}")
        if self.config.global_kg_file is None:
            if self.config.dataset_file is None or not self.config.dataset_file.exists():
                raise FileNotFoundError(
                    "Either global_kg_file or dataset_file must point to an existing file."
                )
        if self.config.k < 1:
            raise ValueError("k must be >= 1")
        if self.config.index_path_length < 1:
            raise ValueError("index_path_length must be >= 1")
        if self.config.subgraph_hops < 1:
            raise ValueError("subgraph_hops must be >= 1")
        if self.config.max_entity_candidates < 1:
            raise ValueError("max_entity_candidates must be >= 1")
        if self.config.max_selected_entities < 1:
            raise ValueError("max_selected_entities must be >= 1")

    @staticmethod
    def _seed_everything(seed: int) -> None:
        random.seed(seed)
        try:
            import torch

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception:
            pass

    def _truncate_paths(self, paths: list[list[tuple[str, str, str]]]) -> list[list[tuple[str, str, str]]]:
        sorted_paths = sorted(
            paths,
            key=lambda path: (len(path), utils.path_to_string(path)),
        )
        if self.config.max_paths_in_trie is None:
            return sorted_paths
        return sorted_paths[: self.config.max_paths_in_trie]

    def _build_prompt_and_trie(
        self,
        question: str,
        selected_entities: list[str],
    ) -> tuple[str, Any, dict[str, int]]:
        local_triples = self.graph_store.extract_k_hop_subgraph(
            selected_entities,
            hops=self.config.subgraph_hops,
            max_triples=self.config.max_subgraph_triples,
        )
        if not local_triples:
            raise ValueError("No local subgraph could be built from the selected entity.")

        local_graph = utils.build_graph(local_triples, self.config.undirected)
        candidate_paths = utils.dfs(
            local_graph,
            selected_entities,
            self.config.index_path_length,
        )
        candidate_paths = self._truncate_paths(candidate_paths)
        if not candidate_paths:
            raise ValueError("No candidate paths found for the selected entity.")

        question_payload = {
            "question": question,
            "q_entity": selected_entities,
            "a_entity": [],
            "graph": local_triples,
            "paths": candidate_paths,
        }

        prompt, _, trie = self.prompt_builder.process_input(question_payload)
        if trie is None:
            raise ValueError("Trie construction failed because no valid path prefixes were found.")

        return prompt, trie, {
            "subgraph_triples": len(local_triples),
            "candidate_paths": len(candidate_paths),
        }

    def _normalize_predictions(self, prediction: str | list[str]) -> list[str]:
        if prediction is None:
            return []
        if isinstance(prediction, str):
            values = [prediction]
        else:
            values = list(prediction)

        normalized_predictions: list[str] = []
        seen: set[str] = set()
        for value in values:
            text = " ".join(str(value).split()).strip()
            if not text or text in seen:
                continue
            normalized_predictions.append(text)
            seen.add(text)
        return normalized_predictions

    def _generate_constrained_paths(self, prompt: str, trie: Any) -> list[str]:
        model_input = self.model.prepare_model_prompt(prompt)
        inputs = self.model.tokenizer(
            model_input,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = inputs.input_ids.to(self.model.model.device)
        attention_mask = inputs.attention_mask.to(self.model.model.device)

        start_token_id = self.model.tokenizer.convert_tokens_to_ids(
            self.prompt_builder.PATH_START_TOKEN
        )
        end_token_id = self.model.tokenizer.convert_tokens_to_ids(
            self.prompt_builder.PATH_END_TOKEN
        )

        constrained_decoder = GraphConstrainedDecoding(
            self.model.tokenizer,
            trie,
            start_token_id,
            end_token_id,
            False,
        )
        stopping_criteria = StoppingCriteriaList(
            [PathEndStoppingCriteria(start_token_id, end_token_id)]
        )

        generation_config = copy.deepcopy(self.model.generation_cfg)
        generation_config.pad_token_id = self.model.tokenizer.eos_token_id
        generation_config.return_dict_in_generate = True

        outputs = self.model.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
            prefix_allowed_tokens_fn=constrained_decoder.allowed_tokens_fn,
            stopping_criteria=stopping_criteria,
        )

        prompt_length = input_ids.shape[1]
        decoded = [
            self.model.tokenizer.decode(
                sequence[prompt_length:],
                skip_special_tokens=True,
            )
            for sequence in outputs.sequences
        ]
        return self._normalize_predictions(decoded)

    def _resolve_manual_entities(self, manual_entities: list[str]) -> list[str]:
        resolved_entities: list[str] = []
        for entity in manual_entities:
            resolved = self.entity_extractor.resolve_entity(entity)
            if resolved is None:
                raise ValueError(f"Unable to resolve entity from input: {entity}")
            resolved_entities.append(resolved)

        deduplicated: list[str] = []
        seen: set[str] = set()
        for entity in resolved_entities:
            if entity in seen:
                continue
            deduplicated.append(entity)
            seen.add(entity)
        return deduplicated

    def _select_entities(
        self,
        question: str,
        manual_entities: list[str] | None = None,
    ) -> tuple[list[str], list[EntityCandidate]]:
        if manual_entities:
            resolved = self._resolve_manual_entities(manual_entities)
            candidates = [
                EntityCandidate(entity=entity, score=999.0, source="manual")
                for entity in resolved
            ]
            return resolved, candidates

        candidates = self.entity_extractor.extract(
            question,
            mode=self.config.entity_match_mode,
            top_k=self.config.max_entity_candidates,
            min_score=self.config.min_entity_similarity,
        )
        if not candidates:
            raise ValueError("No entity candidates could be extracted from the question.")

        if len(candidates) == 1 or not self.config.interactive_entity_selection:
            default_entities = [
                candidate.entity
                for candidate in candidates[: self.config.max_selected_entities]
            ]
            return default_entities, candidates

        print("[entity] candidates:")
        for index, candidate in enumerate(candidates, start=1):
            print(
                f"  {index}. {candidate.entity} "
                f"(score={candidate.score:.3f}, source={candidate.source})"
            )

        prompt = (
            f"[entity] select up to {self.config.max_selected_entities} candidate indices "
            "(comma-separated, Enter for default, or type a custom entity): "
        )
        response = input(prompt).strip()
        if not response:
            default_entities = [
                candidate.entity
                for candidate in candidates[: self.config.max_selected_entities]
            ]
            return default_entities, candidates

        if all(part.strip().isdigit() for part in response.split(",")):
            selected_entities: list[str] = []
            for part in response.split(","):
                index = int(part.strip()) - 1
                if 0 <= index < len(candidates):
                    selected_entities.append(candidates[index].entity)
                if len(selected_entities) >= self.config.max_selected_entities:
                    break
            if selected_entities:
                return selected_entities, candidates

        resolved = self._resolve_manual_entities([response])
        return resolved, candidates

    def run_once(
        self,
        question: str,
        manual_entities: list[str] | None = None,
    ) -> dict[str, Any]:
        selected_entities, entity_candidates = self._select_entities(
            question,
            manual_entities=manual_entities,
        )
        prompt, trie, stats = self._build_prompt_and_trie(question, selected_entities)
        generated_paths = self._generate_constrained_paths(prompt, trie)
        return {
            "question": question,
            "selected_entities": selected_entities,
            "entity_candidates": [asdict(candidate) for candidate in entity_candidates],
            "subgraph_triples": stats["subgraph_triples"],
            "candidate_paths": stats["candidate_paths"],
            "generated_paths": generated_paths,
        }

    def interactive_loop(self) -> None:
        print("Interactive graph-constrained client")
        print("Type /config to print config, /quit to exit.")
        while True:
            question = input("\nQuestion> ").strip()
            if not question:
                continue
            if question.lower() in {"/quit", "/exit"}:
                break
            if question.lower() == "/config":
                print(json.dumps(asdict(self.config), ensure_ascii=False, indent=2, default=str))
                continue

            try:
                result = self.run_once(question)
            except Exception as exc:
                print(f"[error] {exc}")
                continue

            print("[result] selected_entities:", ", ".join(result["selected_entities"]))
            print(
                "[result] subgraph_triples:",
                result["subgraph_triples"],
                "candidate_paths:",
                result["candidate_paths"],
            )
            if not result["generated_paths"]:
                print("[result] no constrained path was generated.")
                continue
            print("[result] generated_paths:")
            for index, path in enumerate(result["generated_paths"], start=1):
                print(f"  {index}. {path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive graph-constrained decoding client")
    parser.add_argument(
        "--config",
        type=str,
        default=str(REPO_ROOT / "interactive_client" / "default_config.json"),
        help="Path to a JSON config file.",
    )
    parser.add_argument("--question", type=str, default=None, help="Run a single question and exit.")
    parser.add_argument(
        "--entity",
        action="append",
        default=None,
        help="Manually specify a subject entity. Can be repeated.",
    )
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--global-kg-file", type=str, default=None)
    parser.add_argument("--dataset-file", type=str, default=None)
    parser.add_argument("--prompt-mode", type=str, default=None)
    parser.add_argument("--index-path-length", type=int, default=None)
    parser.add_argument("--subgraph-hops", type=int, default=None)
    parser.add_argument("--max-subgraph-triples", type=int, default=None)
    parser.add_argument("--max-paths-in-trie", type=int, default=None)
    parser.add_argument("--dtype", type=str, default=None)
    parser.add_argument("--quant", type=str, default=None)
    parser.add_argument("--attn-implementation", type=str, default=None)
    parser.add_argument("--generation-mode", type=str, default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--entity-match-mode", type=str, default=None)
    parser.add_argument("--max-entity-candidates", type=int, default=None)
    parser.add_argument("--max-selected-entities", type=int, default=None)
    parser.add_argument("--min-entity-similarity", type=float, default=None)
    parser.add_argument(
        "--no-interactive-entity-selection",
        action="store_true",
        help="Disable the entity-selection prompt and use the highest-scoring candidates directly.",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print the resolved configuration and exit.",
    )
    return parser


def build_config_from_args(args: argparse.Namespace) -> InteractiveClientConfig:
    config = InteractiveClientConfig.from_json(args.config)
    config.update_from_mapping(
        {
            "model_name": args.model_name,
            "model_path": args.model_path,
            "global_kg_file": args.global_kg_file,
            "dataset_file": args.dataset_file,
            "prompt_mode": args.prompt_mode,
            "index_path_length": args.index_path_length,
            "subgraph_hops": args.subgraph_hops,
            "max_subgraph_triples": args.max_subgraph_triples,
            "max_paths_in_trie": args.max_paths_in_trie,
            "dtype": args.dtype,
            "quant": args.quant,
            "attn_implementation": args.attn_implementation,
            "generation_mode": args.generation_mode,
            "k": args.k,
            "max_new_tokens": args.max_new_tokens,
            "entity_match_mode": args.entity_match_mode,
            "max_entity_candidates": args.max_entity_candidates,
            "max_selected_entities": args.max_selected_entities,
            "min_entity_similarity": args.min_entity_similarity,
        }
    )
    if args.no_interactive_entity_selection:
        config.interactive_entity_selection = False
    return config


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    config = build_config_from_args(args)

    if args.print_config:
        print(json.dumps(config.to_dict(), ensure_ascii=False, indent=2))
        return

    client = GraphReasoningClient(config)

    if args.question:
        result = client.run_once(args.question, manual_entities=args.entity)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    client.interactive_loop()


if __name__ == "__main__":
    main()
