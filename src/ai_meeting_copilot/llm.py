"""Prompting and streaming answers from OpenAI or Anthropic."""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from typing import Protocol

from .config import openrouter_api_key, openrouter_base_url
from .rag import RetrievedChunk

SYSTEM_PROMPT = """\
You are a real-time meeting copilot. You hear what the OTHER person in a live \
conversation says (via speech recognition, so expect transcription errors and fix \
them silently) and you help the user reply. The user will read your answer off the \
screen while speaking, so it must be fast to scan.

Reply in exactly this format and nothing else:

ANSWER: <what the user can say, in the first person, in natural spoken English. \
Lead with the direct answer, then the key supporting points. 3-6 sentences. No \
markdown, no bullet points, no preamble like "Great question".>

FOLLOW-UP: <the single most likely follow-up question the other person will ask \
next, so the user can prepare.>

Rules:
- If CONTEXT from the user's documents is provided and relevant, ground the answer \
in it (their real experience, numbers, product facts). Ignore it when it is not relevant.
- Never invent personal facts about the user (employers, projects, metrics) that are \
not in the CONTEXT. Speak generally instead.
- If what they said is not a question (small talk, a statement), suggest a short, \
natural response.
- If the transcript is too garbled to understand, say what you think they asked in \
the ANSWER and give your best reply to that."""


class LLMClient(Protocol):
    name: str

    def stream(self, system: str, messages: list[dict]) -> Iterator[str]:
        """Yield the response text incrementally."""
        ...


def format_context(context: Sequence[RetrievedChunk]) -> str:
    return "\n\n".join(f"[{i}] ({r.chunk.source})\n{r.chunk.text}" for i, r in enumerate(context, start=1))


def build_user_turn(transcript: str, context: Sequence[RetrievedChunk] = ()) -> str:
    parts = []
    if context:
        parts.append(f"CONTEXT (excerpts from the user's documents):\n{format_context(context)}")
    parts.append(f"THEY SAID: {transcript}")
    return "\n\n".join(parts)


def build_messages(
    history: Sequence[tuple[str, str]], transcript: str, context: Sequence[RetrievedChunk] = ()
) -> list[dict]:
    """Earlier exchanges become alternating user/assistant turns, which both APIs require.

    Context is attached only to the latest turn: old excerpts would waste tokens.
    """
    messages: list[dict] = []
    for heard, reply in history:
        messages.append({"role": "user", "content": f"THEY SAID: {heard}"})
        messages.append({"role": "assistant", "content": reply})
    messages.append({"role": "user", "content": build_user_turn(transcript, context)})
    return messages


_FOLLOW_UP = re.compile(r"\n?\s*FOLLOW-UP:\s*", re.IGNORECASE)


def parse_response(text: str) -> tuple[str, str]:
    """Split a model reply into (answer, follow_up), tolerating a missing section."""
    parts = _FOLLOW_UP.split(text, maxsplit=1)
    answer = re.sub(r"^\s*ANSWER:\s*", "", parts[0], flags=re.IGNORECASE).strip()
    follow_up = parts[1].strip() if len(parts) > 1 else ""
    return answer, follow_up


class OpenAIChat:
    """OpenAI Chat Completions, or any API that copies it (OpenRouter) via `base_url`."""

    def __init__(
        self,
        model: str = "gpt-4o",
        max_tokens: int = 600,
        base_url: str | None = None,
        api_key: str | None = None,
        label: str = "OpenAI",
        fallback_models: Sequence[str] = (),
    ):
        from openai import OpenAI

        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.fallback_models = list(fallback_models)
        self.name = f"{label} {model}"
        self.last_model: str | None = None  # which model actually answered (differs after a fallback)

    def stream(self, system: str, messages: list[dict]) -> Iterator[str]:
        extra = {}
        if self.fallback_models:
            # OpenRouter-only: if a model is rate-limited or down, OpenRouter tries the next one.
            extra["extra_body"] = {"models": [self.model, *self.fallback_models]}
        self.last_model = None
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, *messages],
            max_completion_tokens=self.max_tokens,
            stream=True,
            **extra,
        )
        try:
            for chunk in response:
                if self.last_model is None and getattr(chunk, "model", None):
                    self.last_model = chunk.model
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        finally:
            response.close()  # stops the HTTP stream if the consumer quits early


class AnthropicChat:
    def __init__(self, model: str = "claude-sonnet-5-5", max_tokens: int = 600):
        from anthropic import Anthropic

        self._client = Anthropic()
        self.model = model
        self.max_tokens = max_tokens
        self.name = f"Anthropic {model}"

    def stream(self, system: str, messages: list[dict]) -> Iterator[str]:
        with self._client.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=messages,
        ) as stream:
            yield from stream.text_stream


# Free OpenRouter models to fall back on, best first. Free models come and go, so the
# list is checked against OpenRouter's live catalogue at startup (see choose_fallbacks).
FREE_FALLBACKS = (
    "google/gemma-4-31b-it:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "qwen/qwen3.8-27b:free",
    "google/gemma-4-26b-a4b-it:free",
)


def openrouter_models(timeout: float = 5.0) -> set[str] | None:
    """Ids of all models OpenRouter currently serves (public endpoint), or None if unreachable."""
    import httpx

    try:
        resp = httpx.get(f"{openrouter_base_url()}/models", timeout=timeout)
        resp.raise_for_status()
        return {m["id"] for m in resp.json()["data"]}
    except Exception:
        return None


def choose_fallbacks(
    model: str, configured: Sequence[str], available: set[str] | None, limit: int = 2
) -> list[str]:
    """Fallbacks for `model`: the user's list, or other free models when `model` is free."""
    candidates = list(configured) or (list(FREE_FALLBACKS) if model.endswith(":free") else [])
    chosen = [m for m in candidates if m != model and (available is None or m in available)]
    return chosen[:limit]


def describe_error(exc: Exception) -> str:
    """Turn API errors into one readable line instead of a JSON dump."""
    status = getattr(exc, "status_code", None)
    body = getattr(exc, "body", None)
    if isinstance(body, dict) and isinstance(body.get("error"), dict):
        body = body["error"]
    detail = ""
    if isinstance(body, dict):
        meta = body.get("metadata") if isinstance(body.get("metadata"), dict) else {}
        detail = str(meta.get("raw") or body.get("message") or "")
    detail = detail.split(". Please retry")[0].strip()
    if status == 429:
        return (
            f"Rate-limited ({detail or 'too many requests'}). Free models share a busy pool and allow "
            "50 requests/day under $10 of credits: try manual mode, --fallback-models, or a paid model."
        )
    if status == 401:
        return "API key rejected (401): check the key in .env."
    if status == 402:
        return "Out of credits (402): top up your account or switch to a :free model."
    if status is not None:
        return f"API error {status}: {detail or exc}"
    return f"{type(exc).__name__}: {exc}"


def create_llm(provider: str, model: str, max_tokens: int, fallback_models: Sequence[str] = ()) -> LLMClient:
    if provider == "openai":
        return OpenAIChat(model, max_tokens)
    if provider == "anthropic":
        return AnthropicChat(model, max_tokens)
    if provider == "openrouter":
        return OpenAIChat(
            model,
            max_tokens,
            base_url=openrouter_base_url(),
            api_key=openrouter_api_key(),
            label="OpenRouter",
            fallback_models=fallback_models,
        )
    raise ValueError(f"Unknown provider {provider!r} (expected 'openai', 'anthropic' or 'openrouter')")
