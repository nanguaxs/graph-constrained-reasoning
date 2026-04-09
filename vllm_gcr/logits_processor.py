from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from src.trie import Trie

try:
    from vllm import SamplingParams
    from vllm.v1.sample.logits_processor import (
        AdapterLogitsProcessor,
        RequestLogitsProcessor,
    )
except ImportError as exc:  # pragma: no cover - exercised only in vLLM envs.
    raise ImportError(
        "vllm_gcr.logits_processor requires vLLM. "
        "Install vLLM in a Linux/WSL environment before importing this module."
    ) from exc


GCR_TRIE_SEQUENCES_KEY = "gcr_trie_sequences"
GCR_START_TOKEN_ID_KEY = "gcr_start_token_id"
GCR_END_TOKEN_ID_KEY = "gcr_end_token_id"
GCR_EOS_TOKEN_ID_KEY = "gcr_eos_token_id"
GCR_ENABLE_BY_DEFAULT_KEY = "gcr_enable_constrained_by_default"


def build_graph_constraint_extra_args(
    trie_sequences: Sequence[Sequence[int]],
    start_token_id: int | None,
    end_token_id: int | None,
    eos_token_id: int | None,
    enable_constrained_by_default: bool = False,
) -> dict[str, Any]:
    return {
        GCR_TRIE_SEQUENCES_KEY: [list(map(int, sequence)) for sequence in trie_sequences],
        GCR_START_TOKEN_ID_KEY: start_token_id,
        GCR_END_TOKEN_ID_KEY: end_token_id,
        GCR_EOS_TOKEN_ID_KEY: eos_token_id,
        GCR_ENABLE_BY_DEFAULT_KEY: enable_constrained_by_default,
    }


@dataclass(frozen=True)
class GraphConstraintConfig:
    trie_sequences: list[list[int]]
    start_token_id: int | None
    end_token_id: int | None
    eos_token_id: int | None
    enable_constrained_by_default: bool = False

    @classmethod
    def from_sampling_params(
        cls,
        params: SamplingParams,
    ) -> "GraphConstraintConfig | None":
        extra_args = params.extra_args or {}
        if GCR_TRIE_SEQUENCES_KEY not in extra_args:
            return None

        trie_sequences = extra_args[GCR_TRIE_SEQUENCES_KEY]
        start_token_id = extra_args.get(GCR_START_TOKEN_ID_KEY)
        end_token_id = extra_args.get(GCR_END_TOKEN_ID_KEY)
        eos_token_id = extra_args.get(GCR_EOS_TOKEN_ID_KEY)
        enable_by_default = extra_args.get(GCR_ENABLE_BY_DEFAULT_KEY, False)

        cls._validate_trie_sequences(trie_sequences)
        cls._validate_optional_int(GCR_START_TOKEN_ID_KEY, start_token_id)
        cls._validate_optional_int(GCR_END_TOKEN_ID_KEY, end_token_id)
        cls._validate_optional_int(GCR_EOS_TOKEN_ID_KEY, eos_token_id)
        if not isinstance(enable_by_default, bool):
            raise ValueError(
                f"{GCR_ENABLE_BY_DEFAULT_KEY} must be a bool, got "
                f"{type(enable_by_default).__name__}."
            )
        if start_token_id is not None and end_token_id is not None and eos_token_id is None:
            raise ValueError(
                f"{GCR_EOS_TOKEN_ID_KEY} must be provided when both "
                f"{GCR_START_TOKEN_ID_KEY} and {GCR_END_TOKEN_ID_KEY} are set."
            )

        normalized_sequences = [
            [int(token_id) for token_id in sequence]
            for sequence in trie_sequences
        ]
        return cls(
            trie_sequences=normalized_sequences,
            start_token_id=start_token_id,
            end_token_id=end_token_id,
            eos_token_id=eos_token_id,
            enable_constrained_by_default=enable_by_default,
        )

    @staticmethod
    def _validate_optional_int(name: str, value: Any) -> None:
        if value is not None and not isinstance(value, int):
            raise ValueError(f"{name} must be an int or None, got {type(value).__name__}.")

    @staticmethod
    def _validate_trie_sequences(trie_sequences: Any) -> None:
        if not isinstance(trie_sequences, Sequence):
            raise ValueError(
                f"{GCR_TRIE_SEQUENCES_KEY} must be a sequence of token-id sequences."
            )
        for sequence in trie_sequences:
            if not isinstance(sequence, Sequence):
                raise ValueError(
                    "Each graph-constrained path must be a sequence of token ids."
                )
            for token_id in sequence:
                if not isinstance(token_id, int):
                    raise ValueError("Trie token ids must all be integers.")


class GraphConstraintRequestProcessor:
    def __init__(self, config: GraphConstraintConfig):
        self.config = config
        self.trie = Trie(config.trie_sequences)

    def __call__(
        self,
        prompt_ids: list[int],
        output_ids: list[int],
        logits: torch.Tensor,
    ) -> torch.Tensor:
        allowed_token_ids = self._resolve_allowed_token_ids(prompt_ids, output_ids)
        if not allowed_token_ids:
            return logits
        return self._mask_logits(logits, allowed_token_ids)

    def _resolve_allowed_token_ids(
        self,
        prompt_ids: list[int],
        output_ids: list[int],
    ) -> list[int] | None:
        if (
            self.config.start_token_id is not None
            and self.config.end_token_id is not None
        ):
            full_sequence = list(prompt_ids) + list(output_ids)
            last_start_index = self._find_last_index(
                full_sequence,
                self.config.start_token_id,
            )
            if last_start_index is not None:
                prefix = full_sequence[last_start_index:]
                if self.config.end_token_id in prefix:
                    return (
                        [self.config.eos_token_id]
                        if self.config.eos_token_id is not None
                        else None
                    )
                return self._lookup_trie(prefix)
            if not self.config.enable_constrained_by_default:
                return None
            return self._lookup_trie(output_ids)

        if not self.config.enable_constrained_by_default:
            return None
        return self._lookup_trie(output_ids)

    def _lookup_trie(self, prefix: Sequence[int]) -> list[int] | None:
        allowed_token_ids = self.trie.get(list(prefix))
        if len(allowed_token_ids) == 0:
            return None
        return allowed_token_ids

    @staticmethod
    def _find_last_index(sequence: Sequence[int], token_id: int) -> int | None:
        for index in range(len(sequence) - 1, -1, -1):
            if sequence[index] == token_id:
                return index
        return None

    @staticmethod
    def _mask_logits(logits: torch.Tensor, allowed_token_ids: Sequence[int]) -> torch.Tensor:
        vocab_size = logits.shape[-1]
        valid_token_ids = sorted(
            {
                int(token_id)
                for token_id in allowed_token_ids
                if 0 <= int(token_id) < vocab_size
            }
        )
        if not valid_token_ids:
            return logits

        cols = torch.tensor(valid_token_ids, dtype=torch.long, device=logits.device)
        kept_values = logits[cols].clone()
        logits[:] = float("-inf")
        logits[cols] = kept_values
        return logits


class GraphConstraintLogitsProcessor(AdapterLogitsProcessor):
    """vLLM wrapper that adapts per-request graph constraints to batch decoding."""

    @classmethod
    def validate_params(cls, params: SamplingParams) -> None:
        GraphConstraintConfig.from_sampling_params(params)

    def is_argmax_invariant(self) -> bool:
        return False

    def new_req_logits_processor(
        self,
        params: SamplingParams,
    ) -> RequestLogitsProcessor | None:
        config = GraphConstraintConfig.from_sampling_params(params)
        if config is None or len(config.trie_sequences) == 0:
            return None
        return GraphConstraintRequestProcessor(config)


__all__ = [
    "GCR_ENABLE_BY_DEFAULT_KEY",
    "GCR_END_TOKEN_ID_KEY",
    "GCR_EOS_TOKEN_ID_KEY",
    "GCR_START_TOKEN_ID_KEY",
    "GCR_TRIE_SEQUENCES_KEY",
    "GraphConstraintLogitsProcessor",
    "build_graph_constraint_extra_args",
]
