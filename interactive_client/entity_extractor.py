from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
import re
import unicodedata


BRACKET_PATTERNS = (
    re.compile(r"\[([^\[\]]+)\]"),
    re.compile(r"【([^【】]+)】"),
    re.compile(r"\(([^()]+)\)"),
    re.compile(r"（([^（）]+)）"),
)


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text))
    kept_chars = []
    for char in normalized.lower():
        category = unicodedata.category(char)
        if category.startswith(("P", "Z", "C")):
            continue
        kept_chars.append(char)
    return "".join(kept_chars).strip()


@dataclass(frozen=True)
class EntityCandidate:
    entity: str
    score: float
    source: str


class EntityExtractor:
    def __init__(self, entities: list[str]) -> None:
        unique_entities = sorted(
            {str(entity).strip() for entity in entities if str(entity).strip()},
            key=lambda item: (-len(item), item),
        )
        self.entities = unique_entities
        self.normalized_entities = [normalize_text(entity) for entity in unique_entities]

        self.normalized_to_entities: dict[str, list[str]] = defaultdict(list)
        self.char_to_indices: dict[str, set[int]] = defaultdict(set)
        for index, entity in enumerate(self.entities):
            normalized = self.normalized_entities[index]
            if not normalized:
                continue
            self.normalized_to_entities[normalized].append(entity)
            for char in set(normalized):
                self.char_to_indices[char].add(index)

    def _collect_bracket_terms(self, question: str) -> list[str]:
        terms: list[str] = []
        for pattern in BRACKET_PATTERNS:
            for match in pattern.findall(question):
                if match:
                    terms.append(match.strip())
        return terms

    def _update_candidate(
        self,
        bucket: dict[str, EntityCandidate],
        entity: str,
        score: float,
        source: str,
    ) -> None:
        current = bucket.get(entity)
        if current is None or score > current.score:
            bucket[entity] = EntityCandidate(entity=entity, score=score, source=source)

    def lookup_exact(self, text: str) -> list[str]:
        normalized = normalize_text(text)
        return list(self.normalized_to_entities.get(normalized, []))

    def resolve_entity(self, text: str) -> str | None:
        exact_matches = self.lookup_exact(text)
        if exact_matches:
            return exact_matches[0]

        candidates = self.extract(
            text,
            mode="hybrid",
            top_k=1,
            min_score=0.0,
        )
        if not candidates:
            return None
        return candidates[0].entity

    def extract(
        self,
        question: str,
        mode: str = "hybrid",
        top_k: int = 5,
        min_score: float = 0.35,
    ) -> list[EntityCandidate]:
        question_normalized = normalize_text(question)
        if not question_normalized:
            return []

        collected: dict[str, EntityCandidate] = {}

        for term in self._collect_bracket_terms(question):
            for entity in self.lookup_exact(term):
                self._update_candidate(
                    collected,
                    entity,
                    score=10.0 + len(entity),
                    source="bracket",
                )

        if mode in {"exact", "hybrid"}:
            for normalized_entity, entity_names in self.normalized_to_entities.items():
                if not normalized_entity or normalized_entity not in question_normalized:
                    continue
                base_score = 5.0 + len(normalized_entity)
                for entity in entity_names:
                    self._update_candidate(
                        collected,
                        entity,
                        score=base_score,
                        source="exact",
                    )

        if mode in {"fuzzy", "hybrid"}:
            candidate_indices: set[int] = set()
            for char in set(question_normalized):
                candidate_indices.update(self.char_to_indices.get(char, set()))

            question_char_set = set(question_normalized)
            for index in candidate_indices:
                normalized_entity = self.normalized_entities[index]
                if not normalized_entity:
                    continue

                entity_char_set = set(normalized_entity)
                overlap = len(question_char_set & entity_char_set) / max(1, len(entity_char_set))
                ratio = SequenceMatcher(None, question_normalized, normalized_entity).ratio()
                containment_bonus = 0.2 if normalized_entity in question_normalized else 0.0
                score = 0.6 * ratio + 0.4 * overlap + containment_bonus
                if score < min_score:
                    continue

                self._update_candidate(
                    collected,
                    self.entities[index],
                    score=score,
                    source="fuzzy",
                )

        return sorted(
            collected.values(),
            key=lambda candidate: (-candidate.score, -len(candidate.entity), candidate.entity),
        )[:top_k]

