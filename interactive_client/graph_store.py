from __future__ import annotations

from collections import deque
import json
from pathlib import Path

import src.utils as utils


class KnowledgeGraphStore:
    def __init__(self, triples: list[list[str]], undirected: bool = False) -> None:
        self.triples = triples
        self.undirected = undirected
        self.graph = utils.build_graph(triples, undirected=undirected)
        self.entities = sorted(str(node) for node in self.graph.nodes())

    @staticmethod
    def _clean_triple(triple: list[str] | tuple[str, str, str] | None) -> tuple[str, str, str] | None:
        if triple is None or len(triple) < 3:
            return None

        head, relation, tail = triple[:3]
        cleaned = (str(head).strip(), str(relation).strip(), str(tail).strip())
        if not all(cleaned):
            return None
        return cleaned

    @classmethod
    def _append_cleaned_triple(
        cls,
        triple: list[str] | tuple[str, ...] | None,
        unique_triples: set[tuple[str, str, str]],
        ordered_triples: list[list[str]],
    ) -> None:
        cleaned = cls._clean_triple(triple)
        if cleaned is None or cleaned in unique_triples:
            return
        unique_triples.add(cleaned)
        ordered_triples.append(list(cleaned))

    @classmethod
    def _load_from_dataset_records(
        cls,
        handle,
        unique_triples: set[tuple[str, str, str]],
        ordered_triples: list[list[str]],
    ) -> None:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Failed to parse JSON on line {line_number}"
                ) from exc

            if isinstance(record, dict):
                for triple in record.get("graph", []):
                    cls._append_cleaned_triple(triple, unique_triples, ordered_triples)
            elif isinstance(record, list):
                cls._append_cleaned_triple(record, unique_triples, ordered_triples)

    @classmethod
    def _load_from_json_payload(
        cls,
        payload,
        unique_triples: set[tuple[str, str, str]],
        ordered_triples: list[list[str]],
    ) -> None:
        if isinstance(payload, dict):
            for triple in payload.get("graph", []):
                cls._append_cleaned_triple(triple, unique_triples, ordered_triples)
            return

        if not isinstance(payload, list):
            return

        for item in payload:
            if isinstance(item, dict):
                for triple in item.get("graph", []):
                    cls._append_cleaned_triple(triple, unique_triples, ordered_triples)
            elif isinstance(item, list):
                cls._append_cleaned_triple(item, unique_triples, ordered_triples)

    @classmethod
    def from_file(
        cls,
        source_file: str | Path,
        encoding: str = "utf-8",
        undirected: bool = False,
    ) -> "KnowledgeGraphStore":
        source_path = Path(source_file)
        if not source_path.exists():
            raise FileNotFoundError(f"Knowledge graph file not found: {source_path}")

        unique_triples: set[tuple[str, str, str]] = set()
        ordered_triples: list[list[str]] = []

        try:
            with source_path.open("r", encoding=encoding) as handle:
                payload = json.load(handle)
            cls._load_from_json_payload(payload, unique_triples, ordered_triples)
        except json.JSONDecodeError:
            with source_path.open("r", encoding=encoding) as handle:
                cls._load_from_dataset_records(handle, unique_triples, ordered_triples)

        return cls(ordered_triples, undirected=undirected)

    def extract_k_hop_subgraph(
        self,
        start_entities: list[str],
        hops: int,
        max_triples: int | None = None,
    ) -> list[list[str]]:
        if hops < 1:
            return []

        queue: deque[tuple[str, int]] = deque()
        visited_nodes: set[str] = set()
        collected_triples: list[list[str]] = []
        seen_triples: set[tuple[str, str, str]] = set()

        for entity in start_entities:
            if entity in self.graph:
                queue.append((entity, 0))
                visited_nodes.add(entity)

        while queue:
            current_node, depth = queue.popleft()
            if depth >= hops or current_node not in self.graph:
                continue

            for neighbor in self.graph.neighbors(current_node):
                relation = str(self.graph[current_node][neighbor]["relation"]).strip()
                triple = (str(current_node).strip(), relation, str(neighbor).strip())
                if triple not in seen_triples:
                    seen_triples.add(triple)
                    collected_triples.append(list(triple))
                    if max_triples is not None and len(collected_triples) >= max_triples:
                        return collected_triples

                if neighbor not in visited_nodes:
                    visited_nodes.add(neighbor)
                    queue.append((neighbor, depth + 1))

        return collected_triples
