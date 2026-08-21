"""
Per-device settings, editable from the UI.

Everyone runs their own copy of this app with their own Gemini key, so
requiring them to find and edit a .env file is a bad first five minutes. The
key can be set from the Settings screen instead and is written to
data/config.json, which is gitignored along with the rest of data/.

Precedence: this file wins over the environment. Something typed into the UI
should take effect without also having to clear a stale environment variable.

The key is stored in plain text. That is the same exposure as a .env file and
appropriate for a local single-user tool, but it is not a secret store: the API
never sends the key back to the browser, only whether one is set and its last
four characters.
"""

from __future__ import annotations

import json
import threading
from typing import Any

from .config import settings

# Reentrant on purpose: save() reads the current config while holding the
# lock, and a plain Lock deadlocks on that nested acquire -- which hung
# every attempt to save a setting, from both the CLI and the UI.
# What the environment (.env / real env vars) provided at startup. Removing a
# value from config.json falls back to this rather than leaving the previously
# applied value in place -- otherwise "clear my API key" appears to do nothing
# until the process restarts.
_ENV_BASELINE = {
    "gemini_api_key": settings.api_key,
    "chat_provider": settings.chat_provider,
    "chat_base_url": settings.chat_base_url,
    "chat_api_key": settings.chat_api_key,
    "speech_provider": settings.speech_provider,
    "speech_model": settings.speech_model,
    "speech_base_url": settings.speech_base_url,
    "speech_api_key": settings.speech_api_key,
    "speech_language": settings.speech_language,
    "chat_model": settings.chat_model,
    "vision_model": settings.vision_model,
    "embed_model": settings.embed_model,
    "session_gap_hours": settings.session_gap_hours,
    "describe_media": settings.describe_media,
    "transcribe_audio": settings.transcribe_audio,
    "media_concurrency": settings.media_concurrency,
    "max_requests_per_minute": settings.max_requests_per_minute,
    "max_requests_per_day": settings.max_requests_per_day,
}

_lock = threading.RLock()
_cache: dict[str, Any] | None = None
_cache_mtime: float | None = None
_applying = False        # guards the re-apply triggered from load()

# Only these may be written from the UI. An open-ended dict would let the
# browser set file paths and model ids that the server then acts on.
ALLOWED_KEYS = {
    "gemini_api_key",
    "chat_provider",
    "chat_base_url",
    "chat_api_key",
    "speech_provider",
    "speech_model",
    "speech_base_url",
    "speech_api_key",
    "speech_language",
    "chat_model",
    "vision_model",
    "embed_model",
    "session_gap_hours",
    "describe_media",
    "transcribe_audio",
    "media_concurrency",
    "max_requests_per_minute",
    "max_requests_per_day",
}


def _config_path():
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    return settings.data_dir / "config.json"


def load() -> dict[str, Any]:
    """
    Read the stored config, reloading if the file changed on disk.

    The mtime check matters because the CLI and the server are separate
    processes: setting a key with `app.cli setkey` while the server is running
    would otherwise have no effect until a restart.
    """
    global _cache, _cache_mtime, _applying
    with _lock:
        path = _config_path()
        mtime = path.stat().st_mtime if path.exists() else None

        if _cache is not None and mtime == _cache_mtime:
            return dict(_cache)

        if path.exists():
            try:
                _cache = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                _cache = {}
        else:
            _cache = {}
        _cache_mtime = mtime
        snapshot = dict(_cache)

    # The file changed underneath us, so the live Settings object is stale.
    # Re-applying here makes every read path self-healing: without it, a key
    # written by the CLI shows up in /api/settings (which reads the file) but
    # not to the Gemini client (which reads Settings), and the two disagree.
    # _cache_mtime is already current, so the nested load() below hits the
    # cache and cannot recurse; the flag guards re-entry regardless.
    if not _applying:
        _applying = True
        try:
            apply_to_settings()
        except Exception:  # noqa: BLE001 - a bad config must not break reads
            pass
        finally:
            _applying = False

    return snapshot


def save(updates: dict[str, Any]) -> dict[str, Any]:
    """Merge updates into the stored config and apply them to live settings."""
    global _cache, _cache_mtime
    rejected = set(updates) - ALLOWED_KEYS
    if rejected:
        raise ValueError(f"Not a settable option: {', '.join(sorted(rejected))}")

    # Switching provider without switching model leaves a Gemini model id
    # pointed at Groq, which fails with a confusing 404. Move the model too,
    # unless the same request already sets one explicitly.
    new_provider = updates.get("chat_provider")
    if new_provider and "chat_model" not in updates:
        from .agent.llm import PRESETS

        preset = PRESETS.get(str(new_provider).strip().lower())
        current_model = (load().get("chat_model") or settings.chat_model or "")
        belongs_elsewhere = not any(
            current_model == p.get("default_model")
            for p in [preset] if p
        )
        if preset and preset.get("default_model") and belongs_elsewhere:
            updates = dict(updates)
            updates["chat_model"] = preset["default_model"]

    with _lock:
        current = load()
        for key, value in updates.items():
            # An empty string means "unset", not "set to empty".
            if value is None or (isinstance(value, str) and not value.strip()):
                current.pop(key, None)
            else:
                current[key] = value
        path = _config_path()
        path.write_text(json.dumps(current, indent=2), encoding="utf-8")
        _cache = current
        _cache_mtime = path.stat().st_mtime

    apply_to_settings()
    return current


def apply_to_settings() -> None:
    """
    Push stored values onto the live Settings object.

    Anything absent from config.json reverts to what the environment supplied
    at startup, so unsetting a value takes effect immediately.
    """
    stored = load()

    def resolve(key):
        value = stored.get(key)
        if value is None or (isinstance(value, str) and not value.strip()):
            return _ENV_BASELINE[key]
        return value

    settings.api_key = str(resolve("gemini_api_key") or "").strip()
    settings.chat_provider = str(resolve("chat_provider") or "gemini").strip().lower()
    settings.chat_base_url = str(resolve("chat_base_url") or "").strip()
    settings.chat_api_key = str(resolve("chat_api_key") or "").strip()
    settings.speech_provider = str(resolve("speech_provider") or "").strip().lower()
    settings.speech_model = str(resolve("speech_model") or "").strip()
    settings.speech_base_url = str(resolve("speech_base_url") or "").strip()
    settings.speech_api_key = str(resolve("speech_api_key") or "").strip()
    settings.speech_language = str(resolve("speech_language") or "").strip()
    settings.chat_model = str(resolve("chat_model")).strip()
    settings.vision_model = str(resolve("vision_model")).strip()
    settings.embed_model = str(resolve("embed_model")).strip()

    try:
        settings.session_gap_hours = float(resolve("session_gap_hours"))
    except (TypeError, ValueError):
        settings.session_gap_hours = _ENV_BASELINE["session_gap_hours"]

    try:
        settings.media_concurrency = max(1, int(resolve("media_concurrency")))
    except (TypeError, ValueError):
        settings.media_concurrency = _ENV_BASELINE["media_concurrency"]

    for key in ("max_requests_per_minute", "max_requests_per_day"):
        try:
            setattr(settings, key, max(0, int(resolve(key))))
        except (TypeError, ValueError):
            setattr(settings, key, _ENV_BASELINE[key])

    settings.describe_media = bool(resolve("describe_media"))
    settings.transcribe_audio = bool(resolve("transcribe_audio"))

    # A changed key must not keep using a client built with the old one.
    from .index import gemini

    gemini.reset_client()


def public_view() -> dict[str, Any]:
    """
    Settings safe to send to the browser.

    The key itself never leaves the server; the last four characters are enough
    for someone to recognise which key is configured.
    """
    stored = load()
    key = str(stored.get("gemini_api_key") or settings.api_key or "")
    chat_key = str(stored.get("chat_api_key") or settings.chat_api_key or "")

    from .agent.llm import PRESETS, describe_provider
    from .index.transcribe import SPEECH_PRESETS, describe_config

    return {
        "chat": describe_provider(),
        "speech": describe_config(),
        "speech_providers": [{"id": pid, **preset}
                             for pid, preset in SPEECH_PRESETS.items()],
        "chat_api_key_set": bool(chat_key.strip()),
        "chat_api_key_hint": f"...{chat_key[-4:]}" if len(chat_key) >= 4 else "",
        "chat_provider": settings.chat_provider,
        "chat_base_url": settings.chat_base_url,
        "providers": [
            {"id": pid, **{k: v for k, v in preset.items()}}
            for pid, preset in PRESETS.items()
        ],
        "api_key_set": bool(key.strip()),
        "api_key_hint": f"...{key[-4:]}" if len(key) >= 4 else "",
        "api_key_source": "config" if stored.get("gemini_api_key") else
                          ("env" if settings.api_key else "none"),
        "chat_model": settings.chat_model,
        "vision_model": settings.vision_model,
        "embed_model": settings.embed_model,
        "embed_dims": settings.embed_dims,
        "session_gap_hours": settings.session_gap_hours,
        "describe_media": settings.describe_media,
        "transcribe_audio": settings.transcribe_audio,
        "media_concurrency": settings.media_concurrency,
        "max_requests_per_minute": settings.max_requests_per_minute,
        "max_requests_per_day": settings.max_requests_per_day,
    }
