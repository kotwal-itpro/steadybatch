"""Batch providers. Each one is imported lazily so you only need the SDKs you use."""

from __future__ import annotations

from .base import BatchProvider
from .fake import FakeProvider, demo_answer


def get_provider(name: str, model: str | None = None) -> BatchProvider:
    if name == "fake":
        # Misbehaves like a real API: a few lines dropped, errored or malformed.
        return FakeProvider(answer=demo_answer, drop_rate=0.01, error_rate=0.01, invalid_rate=0.01,
                            fail_first_attempt_only=True)
    if name == "openai":
        from .openai_batch import OpenAIBatch
        return OpenAIBatch()
    if name == "anthropic":
        from .anthropic_batch import AnthropicBatch
        return AnthropicBatch()
    if name == "vllm":
        from .vllm_offline import VLLMOffline
        if not model:
            raise ValueError("vllm needs a model name")
        return VLLMOffline(model)
    if name == "gemini":
        from .gemini_batch import GeminiBatch
        return GeminiBatch()
    if name == "bedrock":
        raise NotImplementedError(
            "the Bedrock adapter is next on the list; see docs/ROADMAP.md. Contributions welcome."
        )
    raise ValueError(f"unknown provider: {name}")


__all__ = ["BatchProvider", "FakeProvider", "get_provider"]
