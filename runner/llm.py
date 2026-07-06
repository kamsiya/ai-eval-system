"""LLM client boundary for evaluation runs.

The default implementation is deterministic so the project works without an API
key. Replace or extend ``BaseLLMClient`` to integrate a hosted model later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class BaseLLMClient(Protocol):
    """Minimal interface used by the benchmark runner."""

    def generate(self, prompt: str) -> str:
        """Return a model response for a prompt."""


@dataclass
class MockLLMClient:
    """Deterministic mock model for local evaluation and CI."""

    def generate(self, prompt: str) -> str:
        normalized_prompt = prompt.strip().lower()

        if "capital of france" in normalized_prompt:
            return "Paris"
        if "2 + 2" in normalized_prompt or "2+2" in normalized_prompt:
            return "4"
        if "machine ____" in normalized_prompt:
            return "learning"

        return "I do not know."


def create_llm_client(provider: str = "mock") -> BaseLLMClient:
    """Create an LLM client by provider name.

    This factory keeps model selection out of the runner. A future implementation
    can add providers such as ``openai`` or ``anthropic`` without changing the
    evaluation code.
    """

    if provider == "mock":
        return MockLLMClient()

    raise ValueError(
        f"Unsupported LLM provider '{provider}'. Available providers: mock"
    )
