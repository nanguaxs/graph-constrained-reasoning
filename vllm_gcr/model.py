from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Sequence

import dotenv
from transformers import AutoTokenizer

from .logits_processor import (
    GraphConstraintLogitsProcessor,
    build_graph_constraint_extra_args,
)

dotenv.load_dotenv()


HF_TOKEN = os.getenv("HF_TOKEN")


@dataclass
class GenerationRequest:
    prompt: str
    sampling_params: object


class VLLMGraphConstrainedModel:
    @staticmethod
    def add_args(parser) -> None:
        parser.add_argument(
            "--model_path",
            type=str,
            required=True,
            help="Hugging Face model id or local model path.",
        )
        parser.add_argument("--model_name", type=str, default="vllm-gcr")
        parser.add_argument("--max_new_tokens", type=int, default=1024)
        parser.add_argument(
            "--generation_mode",
            type=str,
            default="greedy",
            choices=["greedy", "sampling"],
        )
        parser.add_argument("--k", type=int, default=1, help="Number of returned paths.")
        parser.add_argument("--temperature", type=float, default=1.0)
        parser.add_argument("--top_p", type=float, default=1.0)
        parser.add_argument("--top_k", type=int, default=-1)
        parser.add_argument("--dtype", type=str, default="auto")
        parser.add_argument("--tensor_parallel_size", type=int, default=1)
        parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
        parser.add_argument("--max_model_len", type=int, default=None)
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument("--batch_size", type=int, default=8)
        parser.add_argument(
            "--chat_model",
            default="true",
            type=lambda x: str(x).lower() == "true",
        )
        parser.add_argument(
            "--trust_remote_code",
            default="true",
            type=lambda x: str(x).lower() == "true",
        )
        parser.add_argument("--enforce_eager", action="store_true")

    def __init__(self, args):
        self.args = args
        self.llm = None
        self.SamplingParams = None
        self.maximun_token = None

    def prepare_for_inference(self) -> None:
        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:  # pragma: no cover - runtime dependency
            raise ImportError(
                "vLLM is not installed. Please install vLLM in a Linux/WSL "
                "environment before running vllm_gcr."
            ) from exc

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args.model_path,
            token=HF_TOKEN,
            trust_remote_code=self.args.trust_remote_code,
        )
        self._validate_path_tokens()

        llm_kwargs = {
            "model": self.args.model_path,
            "tokenizer": self.args.model_path,
            "trust_remote_code": self.args.trust_remote_code,
            "dtype": self.args.dtype,
            "tensor_parallel_size": self.args.tensor_parallel_size,
            "gpu_memory_utilization": self.args.gpu_memory_utilization,
            "seed": self.args.seed,
            "logits_processors": [GraphConstraintLogitsProcessor],
            "enforce_eager": self.args.enforce_eager,
        }
        if self.args.max_model_len is not None:
            llm_kwargs["max_model_len"] = self.args.max_model_len

        self.llm = LLM(**llm_kwargs)
        self.SamplingParams = SamplingParams
        self.maximun_token = self.tokenizer.model_max_length

        if self.args.generation_mode == "greedy" and self.args.k != 1:
            raise ValueError("generation_mode=greedy requires k=1 for vllm_gcr.")

    def token_len(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False).input_ids)

    def prepare_model_prompt(self, query: str) -> str:
        if self.args.chat_model:
            path_start = "<PATH>"
            if query.endswith(path_start):
                user_content = query[: -len(path_start)]
                chat_query = [{"role": "user", "content": user_content}]
                return (
                    self.tokenizer.apply_chat_template(
                        chat_query,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    + path_start
                )
            chat_query = [{"role": "user", "content": query}]
            return self.tokenizer.apply_chat_template(
                chat_query,
                tokenize=False,
                add_generation_prompt=True,
            )
        return query

    def build_request(
        self,
        prompt: str,
        trie: Iterable[Sequence[int]],
        start_token_id: int | None,
        end_token_id: int | None,
        enable_constrained_by_default: bool = False,
    ) -> GenerationRequest:
        trie_sequences = [list(map(int, sequence)) for sequence in trie]
        sampling_params = self._build_sampling_params(
            trie_sequences=trie_sequences,
            start_token_id=start_token_id,
            end_token_id=end_token_id,
            enable_constrained_by_default=enable_constrained_by_default,
        )
        return GenerationRequest(prompt=prompt, sampling_params=sampling_params)

    def generate_sentence(
        self,
        llm_input: str,
        trie: Iterable[Sequence[int]],
        start_token_ids: int | None = None,
        end_token_ids: int | None = None,
        enable_constrained_by_default: bool = False,
    ):
        request = self.build_request(
            prompt=llm_input,
            trie=trie,
            start_token_id=start_token_ids,
            end_token_id=end_token_ids,
            enable_constrained_by_default=enable_constrained_by_default,
        )
        return self.generate_batch([request])[0]

    def generate_batch(self, requests: Sequence[GenerationRequest]) -> list[str | list[str]]:
        if self.llm is None or self.SamplingParams is None:
            raise RuntimeError("Call prepare_for_inference() before generate_batch().")
        if len(requests) == 0:
            return []

        prompts = [request.prompt for request in requests]
        sampling_params = [request.sampling_params for request in requests]
        outputs = self.llm.generate(prompts, sampling_params, use_tqdm=False)

        predictions: list[str | list[str]] = []
        for output in outputs:
            texts = [candidate.text for candidate in output.outputs]
            predictions.append(texts[0] if len(texts) == 1 else texts)
        return predictions

    def _build_sampling_params(
        self,
        trie_sequences: Sequence[Sequence[int]],
        start_token_id: int | None,
        end_token_id: int | None,
        enable_constrained_by_default: bool,
    ):
        if self.SamplingParams is None:
            raise RuntimeError("SamplingParams is unavailable before prepare_for_inference().")

        extra_args = build_graph_constraint_extra_args(
            trie_sequences=trie_sequences,
            start_token_id=start_token_id,
            end_token_id=end_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            enable_constrained_by_default=enable_constrained_by_default,
        )

        sampling_kwargs = {
            "max_tokens": self.args.max_new_tokens,
            "n": self.args.k if self.args.generation_mode == "sampling" else 1,
            "skip_special_tokens": True,
            "extra_args": extra_args,
        }
        if self.tokenizer.eos_token_id is not None:
            sampling_kwargs["stop_token_ids"] = [self.tokenizer.eos_token_id]

        if self.args.generation_mode == "greedy":
            sampling_kwargs["temperature"] = 0.0
        else:
            sampling_kwargs["temperature"] = self.args.temperature
            sampling_kwargs["top_p"] = self.args.top_p
            if self.args.top_k is not None and self.args.top_k > 0:
                sampling_kwargs["top_k"] = self.args.top_k

        return self.SamplingParams(**sampling_kwargs)

    def _validate_path_tokens(self) -> None:
        for token in ("<PATH>", "</PATH>"):
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            if token_id is None:
                raise ValueError(
                    f"Tokenizer for {self.args.model_path} does not define {token!r}."
                )
            if (
                self.tokenizer.unk_token_id is not None
                and token_id == self.tokenizer.unk_token_id
            ):
                raise ValueError(
                    f"Tokenizer for {self.args.model_path} maps {token!r} to <unk>. "
                    "Please use the tokenizer produced by finetuning with "
                    "workflow/finetune_kg_specialized_llm.py."
                )
            encoded = self.tokenizer(token, add_special_tokens=False).input_ids
            if encoded != [token_id]:
                raise ValueError(
                    f"{token!r} must map to exactly one token id for vLLM graph "
                    f"constraints, but got {encoded}."
                )


__all__ = ["GenerationRequest", "VLLMGraphConstrainedModel"]
