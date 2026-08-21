"""
Chat model providers.

The agent needs one thing from a model: given a conversation and a set of tool
definitions, either call a tool or answer. Two wire protocols implement that --
Gemini's, and the OpenAI chat-completions format that almost everyone else
speaks (Groq, xAI, OpenRouter, Together, a local Ollama). This module hides the
difference behind one interface so the agent loop does not care.

The neutral format here is OpenAI's, because it is the de facto standard and
the majority of providers need no translation at all. Gemini is converted at
its own boundary.

Why this exists: free-tier allowances differ by orders of magnitude between
providers, and being locked to one means being locked to its quota. Embeddings
stay on Gemini regardless -- most OpenAI-compatible hosts do not serve an
embeddings endpoint at all, and vectors are cheap and cached forever.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from ..config import settings


@dataclass
class ToolCallRequest:
    """A tool the model wants run."""

    id: str
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    # Groq's gpt-oss models return their thinking separately, and sometimes put
    # the finished reply there with `text` empty. Kept apart rather than merged
    # into `text`: on a normal turn this is private monologue and must never
    # reach the user. Only the closing call, where the model has been told to
    # write the final answer and has no tools, reads it.
    reasoning: str = ""

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


class LLMError(RuntimeError):
    pass


# --- known endpoints ----------------------------------------------------------
# Base URLs only. Model names change constantly and are configured separately,
# so nothing here needs updating when a provider ships a new model.
PRESETS: dict[str, dict[str, str]] = {
    "gemini": {
        "label": "Google Gemini",
        "base_url": "",
        "keys_url": "https://aistudio.google.com/apikey",
        "default_model": "gemini-2.5-flash",
        "note": "Native SDK. Also used for embeddings and media understanding.",
    },
    "groq": {
        "label": "Groq",
        "base_url": "https://api.groq.com/openai/v1",
        "keys_url": "https://console.groq.com/keys",
        "default_model": "llama-3.3-70b-versatile",
        "note": "Generous free tier, very fast. No embeddings endpoint.",
    },
    "xai": {
        "label": "xAI (Grok)",
        "base_url": "https://api.x.ai/v1",
        "keys_url": "https://console.x.ai",
        "default_model": "grok-4-fast",
        "note": "Paid only, no free tier. No embeddings endpoint.",
    },
    "openrouter": {
        "label": "OpenRouter",
        "base_url": "https://openrouter.ai/api/v1",
        "keys_url": "https://openrouter.ai/keys",
        "default_model": "meta-llama/llama-3.3-70b-instruct",
        "note": "Routes to many providers, including some free models.",
    },
    "ollama": {
        "label": "Ollama (local)",
        "base_url": "http://localhost:11434/v1",
        "keys_url": "",
        "default_model": "llama3.1",
        "note": "Runs on this machine. No cost, no limits, nothing leaves it.",
    },
    "openai": {
        "label": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "keys_url": "https://platform.openai.com/api-keys",
        "default_model": "gpt-4.1-mini",
        "note": "Paid.",
    },
    "custom": {
        "label": "Other OpenAI-compatible",
        "base_url": "",
        "keys_url": "",
        "default_model": "",
        "note": "Any endpoint serving /chat/completions.",
    },
}


def openai_tool_schema(declarations: list[dict]) -> list[dict]:
    """Wrap the shared tool declarations in OpenAI's envelope."""
    return [
        {
            "type": "function",
            "function": {
                "name": d["name"],
                "description": d["description"],
                "parameters": d.get("parameters") or {
                    "type": "object", "properties": {}},
            },
        }
        for d in declarations
    ]


# "Please try again in 23.37s", "try again in 1m2s", "retry after 4 seconds".
_WAIT_HINT = re.compile(
    r"(?:try again|retry)(?:\s+\w+){0,3}?\s+(?:in|after)\s+"
    r"(?:(\d+)m)?\s*([\d.]+)\s*(m|s|ms|sec|secs|seconds|minutes?)?",
    re.IGNORECASE,
)


def is_too_large(message: str) -> bool:
    """
    Distinguish "you are going too fast" from "this request cannot fit".

    Both arrive as 429 on some providers, but they need opposite responses:
    waiting fixes the first and can never fix the second. Retrying an oversized
    request just burns the clock and the request budget.
    """
    text = (message or "").lower()
    return ("request too large" in text
            or "reduce your message size" in text
            or "context length" in text
            or "too many tokens" in text
            or "maximum context" in text)


# Beyond this, waiting inside a request is worse than saying so. A minute of a
# spinner is tolerable; eleven is not, and the request would very likely be
# cancelled long before it succeeded.
MAX_WAIT_SECONDS = 45.0


def is_daily_limit(message: str) -> bool:
    """
    A quota that resets tomorrow, not in a moment.

    Providers report these as ordinary 429s, but the response is entirely
    different: a per-minute limit clears by waiting a few seconds, a daily one
    does not clear at all today. Retrying it burns minutes of the user's time
    to arrive at the same refusal.
    """
    text = (message or "").lower()
    return ("per day" in text
            or "tpd" in text
            or "perday" in text.replace(" ", "")
            or "daily" in text
            or "requests_per_day" in text)


def _retry_delay(response, body: str, attempt: int) -> float:
    """
    How long to wait before retrying a rejected request.

    Prefers what the provider said over anything we would guess: a Retry-After
    header first, then a wait hint in the message body. Falls back to
    exponential backoff. Capped so a bad hint cannot hang the request for
    minutes.
    """
    header = ""
    try:
        header = response.headers.get("retry-after", "") or ""
    except Exception:  # noqa: BLE001 - stubbed responses may lack headers
        header = ""

    if header.strip():
        try:
            return min(max(float(header.strip()), 0.5), 90.0)
        except ValueError:
            pass

    found = _WAIT_HINT.search(body or "")
    if found:
        minutes = float(found.group(1) or 0)
        value = float(found.group(2))
        unit = (found.group(3) or "s").lower()
        seconds = minutes * 60 + (
            value / 1000 if unit == "ms"
            else value * 60 if unit.startswith("m") and unit != "ms"
            else value
        )
        # A shade over what was asked for: waiting exactly the stated time
        # tends to land on the same boundary again.
        return min(max(seconds + 0.75, 0.5), 90.0)

    return min(2.0 * (2 ** attempt), 30.0)


def _exhausted_message(provider: str, model: str, body: str,
                       wait: float) -> str:
    """A refusal the reader can act on, rather than a wall of provider JSON."""
    daily = is_daily_limit(body)
    when = (f"about {wait / 60:.0f} minutes" if wait >= 90
            else f"about {wait:.0f} seconds")

    if daily:
        head = (f"{provider} has no quota left today for {model}.")
        advice = (
            "Switch to a different model on the Settings tab -- quotas are per "
            "model, so another one on the same key usually still has room -- "
            "or wait for the daily reset."
        )
    else:
        head = f"{provider} is rate limited on {model} for {when}."
        advice = ("Try again shortly, or switch to a model with more headroom "
                  "on the Settings tab.")

    return f"{head} {advice}"


class OpenAICompatProvider:
    """
    Any endpoint speaking POST /chat/completions with tool calling.

    Implemented over httpx rather than the openai SDK: the request is a single
    JSON post, and avoiding the dependency keeps the install light and stops a
    provider-specific client dictating behaviour.
    """

    def __init__(self, base_url: str, api_key: str, model: str,
                 provider: str = "custom"):
        self.base_url = (base_url or "").rstrip("/")
        self.api_key = api_key or ""
        self.model = model
        self.provider = provider
        if not self.base_url:
            raise LLMError(
                f"No base URL configured for provider {provider!r}."
            )

    @property
    def name(self) -> str:
        return f"{self.provider}:{self.model}"

    def complete(self, messages: list[dict], tools: list[dict] | None,
                 temperature: float = 0.2) -> LLMResponse:
        import httpx

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        response = self._post_with_retry(httpx, payload, headers)

        try:
            body = response.json()
            choice = body["choices"][0]["message"]
        except (KeyError, IndexError, ValueError) as exc:
            raise LLMError(
                f"Unexpected response from {self.provider}: "
                f"{response.text[:300]}") from exc

        calls = []
        for call in choice.get("tool_calls") or []:
            function = call.get("function") or {}
            raw_args = function.get("arguments") or "{}"
            try:
                arguments = json.loads(raw_args) if isinstance(raw_args, str) \
                    else dict(raw_args)
            except json.JSONDecodeError:
                # A model that emits malformed JSON should not kill the turn;
                # the agent reports the failure back and it can try again.
                arguments = {"__malformed__": str(raw_args)[:500]}
            calls.append(ToolCallRequest(
                id=call.get("id") or uuid.uuid4().hex[:12],
                name=function.get("name", ""),
                arguments=arguments,
            ))

        return LLMResponse(
            text=(choice.get("content") or "").strip(),
            reasoning=(choice.get("reasoning") or "").strip(),
            tool_calls=calls,
        )


    def _post_with_retry(self, httpx, payload: dict, headers: dict,
                         attempts: int = 4):
        """
        Post, waiting out rate limits rather than surfacing them as failures.

        Providers rate-limit on tokens per minute as well as request count, and
        an agent conversation grows with every tool result, so the third or
        fourth call in a run is the one that trips it. They also say exactly how
        long to wait -- in a Retry-After header or in the message itself -- and
        giving up on a 429 that came with "try again in 23s" throws away an
        answer that was seconds from working.
        """
        last_error = ""
        for attempt in range(attempts):
            try:
                response = httpx.post(f"{self.base_url}/chat/completions",
                                      json=payload, headers=headers,
                                      timeout=180.0)
            except httpx.HTTPError as exc:
                raise LLMError(
                    f"Could not reach {self.base_url}: {exc}") from exc

            if response.status_code < 400:
                return response

            last_error = response.text[:600]
            retryable = response.status_code in (408, 409, 429, 500, 502, 503, 504)

            # A payload that does not fit will not fit a second later either.
            if response.status_code == 413 or is_too_large(last_error):
                raise LLMError(
                    f"{response.status_code} from {self.provider}: {last_error}")

            wait = _retry_delay(response, last_error, attempt)

            # A daily quota, or a wait longer than anyone will sit through, is
            # not a retry -- it is an answer. Say it once instead of blocking
            # for minutes and refusing anyway.
            if is_daily_limit(last_error) or wait > MAX_WAIT_SECONDS:
                raise LLMError(_exhausted_message(
                    self.provider, self.model, last_error, wait))

            if not retryable or attempt == attempts - 1:
                break

            print(f"[llm] {self.provider} returned {response.status_code}; "
                  f"waiting {wait:.1f}s before retry "
                  f"({attempt + 1}/{attempts - 1})")
            time.sleep(wait)

        raise LLMError(
            f"{response.status_code} from {self.provider} after "
            f"{attempts} attempts: {last_error}"
        )

    def list_models(self) -> list[str]:
        """
        Ask the endpoint what it can actually run.

        Model ids churn constantly and differ per account, so the UI offers
        what the key really has rather than a hardcoded list that goes stale.
        """
        import httpx

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            response = httpx.get(f"{self.base_url}/models", headers=headers,
                                 timeout=30.0)
            response.raise_for_status()
            data = response.json().get("data") or []
            return sorted(m.get("id", "") for m in data if m.get("id"))
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"Could not list models from {self.provider}: "
                           f"{str(exc)[:200]}") from exc


class GeminiProvider:
    """Google's native SDK, translated to and from the neutral format."""

    def __init__(self, model: str):
        self.model = model

    @property
    def name(self) -> str:
        return f"gemini:{self.model}"

    def complete(self, messages: list[dict], tools: list[dict] | None,
                 temperature: float = 0.2) -> LLMResponse:
        from google.genai import types

        from ..index.gemini import get_client, with_retry

        system_text, contents = self._to_gemini(messages, types)

        config_args: dict[str, Any] = {
            "system_instruction": system_text or None,
            "temperature": temperature,
        }
        if tools:
            config_args["tools"] = [types.Tool(function_declarations=[
                t["function"] for t in tools])]
            config_args["automatic_function_calling"] = \
                types.AutomaticFunctionCallingConfig(disable=True)

        client = get_client()
        response = with_retry(
            lambda: client.models.generate_content(
                model=self.model, contents=contents,
                config=types.GenerateContentConfig(**config_args)),
            label="agent turn",
            model=self.model,
        )

        candidate = (response.candidates or [None])[0]
        if candidate is None or not candidate.content:
            raise LLMError("Gemini returned no content.")

        calls = []
        for part in candidate.content.parts or []:
            fc = getattr(part, "function_call", None)
            if fc:
                calls.append(ToolCallRequest(
                    id=uuid.uuid4().hex[:12],
                    name=fc.name,
                    arguments=dict(fc.args or {}),
                ))

        return LLMResponse(text=(response.text or "").strip(), tool_calls=calls)

    @staticmethod
    def _to_gemini(messages: list[dict], types):
        """
        Convert OpenAI-shaped messages into Gemini contents.

        Gemini has no "tool" role: a tool result is a function_response part on
        a user turn, and consecutive results must be merged into one turn.
        """
        system_parts: list[str] = []
        contents = []
        pending_results: list[Any] = []

        def flush_results():
            if pending_results:
                contents.append(types.Content(role="user",
                                              parts=list(pending_results)))
                pending_results.clear()

        for message in messages:
            role = message.get("role")
            content = message.get("content") or ""

            if role == "system":
                system_parts.append(content)
                continue

            if role == "tool":
                try:
                    payload = json.loads(content)
                except (json.JSONDecodeError, TypeError):
                    payload = {"result": content}
                pending_results.append(types.Part.from_function_response(
                    name=message.get("name") or "tool",
                    response={"result": payload},
                ))
                continue

            flush_results()

            if role == "assistant" and message.get("tool_calls"):
                parts = []
                if content:
                    parts.append(types.Part.from_text(text=content))
                for call in message["tool_calls"]:
                    function = call["function"]
                    args = function["arguments"]
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {}
                    parts.append(types.Part.from_function_call(
                        name=function["name"], args=args))
                contents.append(types.Content(role="model", parts=parts))
                continue

            if not content:
                continue
            contents.append(types.Content(
                role="model" if role == "assistant" else "user",
                parts=[types.Part.from_text(text=content)],
            ))

        flush_results()
        return "\n\n".join(system_parts), contents


def build_provider():
    """Construct the configured chat provider."""
    provider = (settings.chat_provider or "gemini").strip().lower()

    if provider == "gemini":
        if not settings.has_api_key:
            raise LLMError(
                "No Gemini API key configured. Add one on the Settings tab, or "
                "switch the chat provider to Groq or a local Ollama."
            )
        return GeminiProvider(settings.chat_model)

    base_url = settings.chat_base_url or PRESETS.get(provider, {}).get("base_url", "")
    key = settings.chat_api_key

    # Ollama runs locally and needs no key; everything else does.
    if not key and provider != "ollama":
        raise LLMError(
            f"No API key configured for {PRESETS.get(provider, {}).get('label', provider)}. "
            "Add one on the Settings tab."
        )

    return OpenAICompatProvider(base_url=base_url, api_key=key,
                                model=settings.chat_model, provider=provider)


def describe_provider() -> dict:
    """What the UI shows about the current chat provider."""
    provider = (settings.chat_provider or "gemini").strip().lower()
    preset = PRESETS.get(provider, PRESETS["custom"])
    key = (settings.api_key if provider == "gemini" else settings.chat_api_key) or ""
    return {
        "provider": provider,
        "label": preset["label"],
        "model": settings.chat_model,
        "base_url": settings.chat_base_url or preset.get("base_url", ""),
        "keys_url": preset.get("keys_url", ""),
        "note": preset.get("note", ""),
        "default_model": preset.get("default_model", ""),
        "key_set": bool(key.strip()) or provider == "ollama",
        "needs_key": provider != "ollama",
    }
