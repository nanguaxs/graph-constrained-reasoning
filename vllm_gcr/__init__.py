"""vLLM-based graph-constrained decoding helpers."""

from importlib import import_module


__all__ = [
    "GCR_ENABLE_BY_DEFAULT_KEY",
    "GCR_END_TOKEN_ID_KEY",
    "GCR_EOS_TOKEN_ID_KEY",
    "GCR_START_TOKEN_ID_KEY",
    "GCR_TRIE_SEQUENCES_KEY",
    "GraphConstraintLogitsProcessor",
    "VLLMGraphConstrainedModel",
    "build_graph_constraint_extra_args",
]


def __getattr__(name):
    if name in {
        "GCR_ENABLE_BY_DEFAULT_KEY",
        "GCR_END_TOKEN_ID_KEY",
        "GCR_EOS_TOKEN_ID_KEY",
        "GCR_START_TOKEN_ID_KEY",
        "GCR_TRIE_SEQUENCES_KEY",
        "GraphConstraintLogitsProcessor",
        "build_graph_constraint_extra_args",
    }:
        module = import_module("vllm_gcr.logits_processor")
        return getattr(module, name)
    if name == "VLLMGraphConstrainedModel":
        module = import_module("vllm_gcr.model")
        return getattr(module, name)
    raise AttributeError(f"module 'vllm_gcr' has no attribute {name!r}")
