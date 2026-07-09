"""LLM client boundary for evaluation runs.

The default implementation is deterministic so the project works without an API
key. The OpenAI implementation is imported lazily so local mock runs do not need
third-party packages installed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional, Protocol


DEFAULT_OPENAI_MODEL = "gpt-5.5"
DEFAULT_INSTRUCTIONS = (
    "You are answering an LLM evaluation case. Return only the final answer. "
    "Keep the response concise and avoid extra explanation unless asked."
)


class BaseLLMClient(Protocol):
    """Minimal interface used by the benchmark runner."""

    provider: str
    model: str

    def generate(self, prompt: str) -> str:
        """Return a model response for a prompt."""


@dataclass
class MockLLMClient:
    """Deterministic mock model for local evaluation and CI."""

    model: str = "mock-v1"
    provider: str = "mock"

    def generate(self, prompt: str) -> str:
        normalized_prompt = prompt.strip().lower()

        if "capital of france" in normalized_prompt:
            return "Paris"
        if "2 + 2" in normalized_prompt or "2+2" in normalized_prompt:
            return "4"
        if "machine ____" in normalized_prompt:
            return "learning"

        return "I do not know."


@dataclass
class OpenAIResponsesClient:
    """OpenAI Responses API client.

    Requires ``OPENAI_API_KEY`` in the environment and the ``openai`` package
    from ``requirements.txt``.
    """

    model: str = DEFAULT_OPENAI_MODEL
    api_key: Optional[str] = None
    timeout: float = 60.0
    max_retries: int = 2
    instructions: str = DEFAULT_INSTRUCTIONS
    provider: str = "openai"

    def __post_init__(self) -> None:
        api_key = self.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is required when using --provider openai."
            )

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The openai package is required for --provider openai. "
                "Install it with: pip install -r requirements.txt"
            ) from exc

        self.api_key = api_key
        self._client = OpenAI(
            api_key=api_key,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

    def generate(self, prompt: str) -> str:
        response = self._client.responses.create(
            model=self.model,
            instructions=self.instructions,
            input=prompt,
        )
        return _extract_response_text(response)


def create_llm_client(
    provider: str = "mock",
    model: Optional[str] = None,
    timeout: float = 60.0,
    max_retries: int = 2,
) -> BaseLLMClient:
    """Create an LLM client by provider name.

    This factory keeps model selection out of the runner. New providers can be
    added here without changing evaluation code.
    """

    normalized_provider = provider.strip().lower()

    if normalized_provider == "mock":
        return MockLLMClient(model=model or "mock-v1")

    if normalized_provider == "openai":
        resolved_model = model or os.getenv("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL
        return OpenAIResponsesClient(
            model=resolved_model,
            timeout=timeout,
            max_retries=max_retries,
        )

    raise ValueError(
        f"Unsupported LLM provider '{provider}'. Available providers: mock, openai"
    )


def _extract_response_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return str(output_text).strip()

    if hasattr(response, "model_dump"):
        response_data = response.model_dump()
    elif hasattr(response, "to_dict"):
        response_data = response.to_dict()
    else:
        response_data = response

    text = _find_text_value(response_data)
    if text:
        return text.strip()

    raise RuntimeError("OpenAI response did not contain output text.")


def _find_text_value(value: Any) -> Optional[str]:
    if isinstance(value, dict):
        text_value = value.get("text")
        if isinstance(text_value, str) and text_value.strip():
            return text_value

        for nested_value in value.values():
            found = _find_text_value(nested_value)
            if found:
                return found

    if isinstance(value, list):
        for item in value:
            found = _find_text_value(item)
            if found:
                return found

    return None
