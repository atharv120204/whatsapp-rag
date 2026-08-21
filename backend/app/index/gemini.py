"""
Thin wrapper over google-genai.

Adds the things a bulk ingest needs and the raw SDK does not: a lazily built
client, retry with backoff on the transient failures you always hit when
pushing thousands of files, and model discovery so the app does not hard-fail
when Google renames a model.
"""

from __future__ import annotations

import random
import time
from typing import Any, Callable, TypeVar

from ..config import settings
from .ratelimit import limiter, retry_after

T = TypeVar("T")

_client = None

# Substrings of errors that are worth retrying. Anything else (a bad API key,
# a malformed request) fails fast rather than burning the retry budget.
_RETRYABLE = (
    "429", "500", "502", "503", "504",
    "resource_exhausted", "unavailable", "deadline", "internal error",
    "rate limit", "overloaded", "timeout",
)


class GeminiUnavailable(RuntimeError):
    """Raised when no API key is configured."""


def get_client():
    global _client
    if _client is None:
        if not settings.has_api_key:
            raise GeminiUnavailable(
                "GEMINI_API_KEY is not set. Copy .env.example to .env and add "
                "your key from https://aistudio.google.com/apikey"
            )
        from google import genai

        _client = genai.Client(api_key=settings.api_key)
    return _client


def reset_client() -> None:
    """
    Drop the cached client.

    Called when the API key changes from the Settings screen: a client built
    with the old key would keep being used until restart otherwise.
    """
    global _client
    _client = None


def is_retryable(exc: Exception) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return any(marker in text for marker in _RETRYABLE)


def with_retry(
    fn: Callable[[], T],
    *,
    attempts: int = 5,
    base_delay: float = 1.5,
    label: str = "gemini call",
    model: str | None = None,
) -> T:
    """
    Run `fn`, retrying transient failures with jittered exponential backoff.

    Jitter matters when several worker threads are throttled at once: without
    it they all retry on the same schedule and get throttled together again.
    """
    last: Exception | None = None
    for attempt in range(attempts):
        # Blocks to stay under the configured rate, and raises DailyQuotaReached
        # when the day's budget is spent rather than burning it on retries.
        limiter.acquire(model)
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last = exc
            if _is_throttle(exc):
                limiter.note_throttled(str(exc), model)
            if not is_retryable(exc) or attempt == attempts - 1:
                raise
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1.0)
            # When the API states its own wait, believe it: backoff guesses
            # low and spends the retry budget rediscovering the same window.
            stated = retry_after(str(exc))
            if stated:
                delay = max(delay, stated + random.uniform(0, 1.0))
            print(f"[gemini] {label} failed ({exc}); retrying in {delay:.1f}s "
                  f"({attempt + 1}/{attempts - 1})")
            time.sleep(delay)
    raise last  # pragma: no cover


def _is_throttle(exc: Exception) -> bool:
    text = f"{exc}".lower()
    return "429" in text or "resource_exhausted" in text or "rate limit" in text


def list_models() -> list[dict[str, Any]]:
    """What this API key can actually call. Used by the CLI to verify config."""
    client = get_client()
    out = []
    for m in client.models.list():
        actions = list(getattr(m, "supported_actions", None) or [])
        out.append({
            "name": getattr(m, "name", ""),
            "display_name": getattr(m, "display_name", ""),
            "supported_actions": actions,
            "input_token_limit": getattr(m, "input_token_limit", None),
            "output_token_limit": getattr(m, "output_token_limit", None),
        })
    return out


def check_config() -> dict[str, Any]:
    """
    Verify the configured model ids exist on this key.

    Model names drift; a typo or a retired id otherwise surfaces as a confusing
    404 in the middle of an hour-long ingest instead of at startup.
    """
    result: dict[str, Any] = {"ok": False, "available": [], "missing": [], "error": None}
    try:
        models = list_models()
    except Exception as exc:  # noqa: BLE001
        result["error"] = str(exc)
        return result

    names = {m["name"].removeprefix("models/") for m in models}
    configured = {
        "chat_model": settings.chat_model,
        "heavy_model": settings.heavy_model,
        "vision_model": settings.vision_model,
        "embed_model": settings.embed_model,
    }
    for role, model in configured.items():
        bucket = "available" if model.removeprefix("models/") in names else "missing"
        result[bucket].append({"role": role, "model": model})

    result["ok"] = not result["missing"]
    result["model_count"] = len(names)
    return result


def embed_texts(
    texts: list[str],
    *,
    task_type: str = "RETRIEVAL_DOCUMENT",
) -> list[list[float]]:
    """
    Embed a batch of strings.

    task_type matters for retrieval quality: documents and queries are embedded
    into the same space but with different instructions, and mixing them up
    measurably degrades results.
    """
    if not texts:
        return []

    from google.genai import types

    client = get_client()

    def _call():
        return client.models.embed_content(
            model=settings.embed_model,
            contents=texts,
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=settings.embed_dims,
            ),
        )

    response = with_retry(_call, label=f"embed {len(texts)} texts",
                          model=settings.embed_model)
    vectors = [list(e.values) for e in response.embeddings]

    # Truncated Matryoshka embeddings need re-normalising before cosine
    # similarity is meaningful; the API only normalises at full dimensionality.
    return [_normalize(v) for v in vectors]


def _normalize(vec: list[float]) -> list[float]:
    norm = sum(v * v for v in vec) ** 0.5
    if norm == 0:
        return vec
    return [v / norm for v in vec]
