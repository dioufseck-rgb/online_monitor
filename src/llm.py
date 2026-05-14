"""
Provider-agnostic LLM adapter.

Three providers supported, all behind one interface:
- gemini       (personal lane)
- azure_openai (NFCU lane — when WS6 lifts this in)
- anthropic    (default; useful for parity testing)

The shape we expose to callers is deliberately narrow:

    classify_structured(prompt, response_model) -> response_model instance

That's it. Every classifier in classify.py uses this. Adding a provider
means implementing one method; classifier code never changes.

Structured output is enforced by the Pydantic model. If the provider
returns malformed JSON or the model rejects it, we raise — silent
failure is the bug we're hunting.

Transient failures (rate limits, 503s, timeouts) are retried with
exponential backoff in the public `classify_structured` method. Subclasses
implement `_classify_structured_once` instead.
"""

from __future__ import annotations
import json
import os
import random
import time
from abc import ABC, abstractmethod
from typing import Type, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


# Substrings that indicate a transient failure worth retrying.
_RETRYABLE_MARKERS = (
    "503",
    "UNAVAILABLE",
    "RESOURCE_EXHAUSTED",
    "429",
    "rate limit",
    "high demand",
    "overloaded",
    "timeout",
    "timed out",
)


def _is_retryable(exc: Exception) -> bool:
    s = str(exc).lower()
    return any(m.lower() in s for m in _RETRYABLE_MARKERS)


class LLMAdapter(ABC):
    """Minimal interface every provider implements."""

    # Tunables — subclasses or callers can override.
    max_retries: int = 4
    base_backoff_seconds: float = 2.0

    def classify_structured(
        self,
        system: str,
        user: str,
        response_model: Type[T],
        temperature: float = 0.0,
    ) -> T:
        """Public entry point. Retries transient failures with exponential
        backoff + jitter. Permanent failures (schema validation, bad request)
        are raised immediately."""
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return self._classify_structured_once(system, user, response_model, temperature)
            except Exception as e:
                last_exc = e
                if attempt >= self.max_retries or not _is_retryable(e):
                    raise
                # Exponential backoff with jitter: 2s, 4s, 8s, 16s
                delay = self.base_backoff_seconds * (2 ** attempt)
                delay += random.uniform(0, delay * 0.25)
                time.sleep(delay)
        # Should be unreachable
        raise last_exc  # type: ignore[misc]

    @abstractmethod
    def _classify_structured_once(
        self,
        system: str,
        user: str,
        response_model: Type[T],
        temperature: float = 0.0,
    ) -> T:
        """Provider-specific single-call implementation."""
        ...


# -----------------------------------------------------------------------------
# Gemini (personal lane)
# -----------------------------------------------------------------------------

class GeminiAdapter(LLMAdapter):
    def __init__(self, model: str = "gemini-2.0-flash", api_key: str | None = None):
        try:
            from google import genai
        except ImportError as e:
            raise RuntimeError(
                "google-genai not installed. pip install google-genai"
            ) from e
        self._genai = genai
        self._client = genai.Client(api_key=api_key or os.environ["GEMINI_API_KEY"])
        self._model = model

    def _classify_structured_once(self, system, user, response_model, temperature=0.0):
        response = self._client.models.generate_content(
            model=self._model,
            contents=f"{system}\n\n{user}",
            config={
                "response_mime_type": "application/json",
                "response_schema": response_model,
                "temperature": temperature,
            },
        )
        # google-genai returns a parsed object directly when response_schema is set
        if hasattr(response, "parsed") and response.parsed is not None:
            return response.parsed
        # Fallback: parse the text
        return _parse_or_raise(response.text, response_model)


# -----------------------------------------------------------------------------
# Azure OpenAI (NFCU lane)
# -----------------------------------------------------------------------------

class AzureOpenAIAdapter(LLMAdapter):
    def __init__(
        self,
        deployment: str,
        endpoint: str | None = None,
        api_key: str | None = None,
        api_version: str = "2024-10-21",
    ):
        try:
            from openai import AzureOpenAI
        except ImportError as e:
            raise RuntimeError("openai not installed. pip install openai") from e
        self._client = AzureOpenAI(
            azure_endpoint=endpoint or os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=api_key or os.environ["AZURE_OPENAI_API_KEY"],
            api_version=api_version,
        )
        self._deployment = deployment

    def _classify_structured_once(self, system, user, response_model, temperature=0.0):
        # Azure OpenAI supports the OpenAI structured-outputs API on recent models
        completion = self._client.beta.chat.completions.parse(
            model=self._deployment,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format=response_model,
            temperature=temperature,
        )
        parsed = completion.choices[0].message.parsed
        if parsed is None:
            raise RuntimeError(
                f"Azure OpenAI returned null parsed result. "
                f"Refusal: {completion.choices[0].message.refusal}"
            )
        return parsed


# -----------------------------------------------------------------------------
# Anthropic (default, parity testing)
# -----------------------------------------------------------------------------

class AnthropicAdapter(LLMAdapter):
    def __init__(self, model: str = "claude-haiku-4-5-20251001", api_key: str | None = None):
        try:
            import anthropic
        except ImportError as e:
            raise RuntimeError("anthropic not installed. pip install anthropic") from e
        self._client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self._model = model

    def _classify_structured_once(self, system, user, response_model, temperature=0.0):
        # Anthropic doesn't have first-class structured output yet; we use
        # tool-use as the structured-output mechanism (forced single tool call).
        schema = response_model.model_json_schema()
        tool_name = response_model.__name__
        message = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            temperature=temperature,
            system=system,
            tools=[{
                "name": tool_name,
                "description": f"Return a {tool_name} result.",
                "input_schema": schema,
            }],
            tool_choice={"type": "tool", "name": tool_name},
            messages=[{"role": "user", "content": user}],
        )
        for block in message.content:
            if block.type == "tool_use" and block.name == tool_name:
                return response_model.model_validate(block.input)
        raise RuntimeError(f"Anthropic returned no {tool_name} tool_use block")


# -----------------------------------------------------------------------------
# Factory + helpers
# -----------------------------------------------------------------------------

def build_adapter(provider: str, **kwargs) -> LLMAdapter:
    """Single entry point used by the orchestrator."""
    provider = provider.lower()
    if provider == "gemini":
        return GeminiAdapter(**kwargs)
    if provider == "azure_openai":
        return AzureOpenAIAdapter(**kwargs)
    if provider == "anthropic":
        return AnthropicAdapter(**kwargs)
    raise ValueError(f"Unknown provider: {provider}")


def _parse_or_raise(text: str, response_model: Type[T]) -> T:
    """Strip code fences and validate. Used as fallback for raw-text responses."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"LLM returned non-JSON: {text[:200]}") from e
    try:
        return response_model.model_validate(data)
    except ValidationError as e:
        raise RuntimeError(f"LLM JSON failed schema validation: {e}") from e
