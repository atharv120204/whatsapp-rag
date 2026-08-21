"""
Speech-to-text for voice notes.

Groq serves Whisper over the OpenAI audio-transcriptions API, on a free tier
that is far more generous than Gemini's. Since voice notes are usually the
largest group of attachments in a chat archive, moving them off Gemini leaves
its small daily allowance for the images, which nothing else here can read.

Falls back to Gemini when no speech provider is configured, so behaviour is
unchanged for anyone who has not set one up.
"""

from __future__ import annotations

import mimetypes
from dataclasses import dataclass
from pathlib import Path

from ..config import settings

# Endpoints that serve POST /audio/transcriptions.
SPEECH_PRESETS: dict[str, dict[str, str]] = {
    "groq": {
        "label": "Groq (Whisper)",
        "base_url": "https://api.groq.com/openai/v1",
        "default_model": "whisper-large-v3",
    },
    "openai": {
        "label": "OpenAI (Whisper)",
        "base_url": "https://api.openai.com/v1",
        "default_model": "whisper-1",
    },
    "custom": {
        "label": "Other OpenAI-compatible",
        "base_url": "",
        "default_model": "whisper-large-v3",
    },
}

# Whisper's own limit; WhatsApp voice notes are far below it, but a shared
# recording can be large.
MAX_AUDIO_BYTES = 24 * 1024 * 1024

_MIME = {
    ".opus": "audio/ogg",
    ".ogg": "audio/ogg",
    ".m4a": "audio/mp4",
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".aac": "audio/aac",
    ".flac": "audio/flac",
    ".amr": "audio/amr",
}


@dataclass
class Transcript:
    text: str = ""
    language: str = ""
    model: str = ""
    error: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.text.strip()) and not self.error


class SpeechUnavailable(RuntimeError):
    pass


def is_configured() -> bool:
    return bool((settings.speech_provider or "").strip()
                and (settings.speech_provider or "").lower() != "gemini")


def describe_config() -> dict:
    provider = (settings.speech_provider or "").strip().lower()
    preset = SPEECH_PRESETS.get(provider, {})
    key = _resolve_key(provider)
    return {
        "provider": provider or "gemini",
        "label": preset.get("label", "Google Gemini"),
        "model": settings.speech_model or preset.get("default_model", ""),
        "base_url": _resolve_base_url(provider),
        "key_set": bool(key),
        "language": settings.speech_language,
        "enabled": is_configured(),
    }


def _resolve_base_url(provider: str) -> str:
    if settings.speech_base_url:
        return settings.speech_base_url
    return SPEECH_PRESETS.get(provider, {}).get("base_url", "")


def _resolve_key(provider: str) -> str:
    """
    Reuse the chat key when the speech provider is the same service.

    Nobody wants to paste the same Groq key twice, and a mismatch between the
    two is a confusing way to fail.
    """
    if settings.speech_api_key:
        return settings.speech_api_key
    if provider and provider == (settings.chat_provider or "").lower():
        return settings.chat_api_key
    return ""


def transcribe_file(path: Path) -> Transcript:
    """Send one audio file for transcription."""
    provider = (settings.speech_provider or "").strip().lower()
    if not is_configured():
        raise SpeechUnavailable("No speech-to-text provider configured.")

    base_url = _resolve_base_url(provider)
    if not base_url:
        return Transcript(error=f"No base URL for speech provider {provider!r}.")

    key = _resolve_key(provider)
    if not key:
        return Transcript(
            error=f"No API key for the speech provider ({provider}).")

    size = path.stat().st_size
    if size > MAX_AUDIO_BYTES:
        return Transcript(
            error=f"Audio is {size / 1048576:.0f} MB, over the "
                  f"{MAX_AUDIO_BYTES // 1048576} MB transcription limit.")
    if size == 0:
        return Transcript(error="Empty audio file.")

    model = settings.speech_model or \
        SPEECH_PRESETS.get(provider, {}).get("default_model", "whisper-large-v3")

    mime = _MIME.get(path.suffix.lower()) or \
        mimetypes.guess_type(path.name)[0] or "audio/ogg"

    payload = {"model": model, "response_format": "verbose_json"}
    if settings.speech_language:
        payload["language"] = settings.speech_language

    import httpx

    from .ratelimit import credential_id, limiter

    def _call():
        with path.open("rb") as fh:
            return httpx.post(
                f"{base_url.rstrip('/')}/audio/transcriptions",
                headers={"Authorization": f"Bearer {key}"},
                files={"file": (path.name, fh, mime)},
                data=payload,
                timeout=180.0,
            )

    # Counted against this model's own budget. It used to share one global
    # counter with Gemini, so transcribing voice notes on Groq -- free, and
    # separately limited -- ate a Gemini allowance it never needed.
    try:
        limiter.acquire(settings.speech_model,
                        credential=credential_id(key))
    except Exception:  # noqa: BLE001 - quota errors propagate to the caller
        raise

    try:
        response = _call()
    except httpx.HTTPError as exc:
        return Transcript(error=f"Could not reach {base_url}: {exc}")

    if response.status_code >= 400:
        limiter.note_throttled(
            response.text, settings.speech_model,
            credential=credential_id(key))
        return Transcript(
            error=f"{response.status_code} from {provider}: "
                  f"{response.text[:300]}")

    try:
        body = response.json()
    except ValueError:
        return Transcript(error=f"Unexpected response: {response.text[:200]}")

    return Transcript(
        text=(body.get("text") or "").strip(),
        language=(body.get("language") or "").strip(),
        model=model,
    )


def check() -> dict:
    """Verify the speech provider answers, for the Settings screen."""
    if not is_configured():
        return {"ok": False,
                "error": "No speech-to-text provider configured; voice notes "
                         "will be transcribed by Gemini."}

    provider = (settings.speech_provider or "").lower()
    base_url = _resolve_base_url(provider)
    key = _resolve_key(provider)
    if not key:
        return {"ok": False, "error": f"No API key for {provider}."}

    import httpx

    try:
        response = httpx.get(f"{base_url.rstrip('/')}/models",
                             headers={"Authorization": f"Bearer {key}"},
                             timeout=30.0)
        response.raise_for_status()
        models = sorted(m.get("id", "") for m in response.json().get("data", []))
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)[:200]}

    wanted = settings.speech_model
    speech_models = [m for m in models if "whisper" in m.lower()]
    return {
        "ok": wanted in models,
        "provider": provider,
        "model": wanted,
        "model_available": wanted in models,
        "speech_models": speech_models,
    }
