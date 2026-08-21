"""
Turn attachments into searchable text with Gemini.

A photo of a train ticket, a two-minute voice note, a screenshot of a bill --
in a raw archive these are opaque, and every question about them fails. Here
each file is read by a multimodal model and reduced to description, transcript,
visible text and object tags, all of which land in the same searchable stream
as ordinary messages.

Three properties this needs, because real archives are large:

  cached      keyed on file content, so a second ingest costs nothing
  resumable   progress is committed per file, so an interrupted run continues
  concurrent  a thread pool, because these calls are latency-bound
"""

from __future__ import annotations

import json
import mimetypes
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Callable

from ..config import settings
from ..db import get_cache_connection
from ..parse.media import MediaFile
from .gemini import get_client, with_retry
from .ratelimit import DailyQuotaReached

# Per-kind instructions. Each asks for the same JSON shape so downstream code
# does not branch, but the emphasis differs: a voice note is mostly transcript,
# a screenshot is mostly OCR.
_PROMPTS: dict[str, str] = {
    "image": (
        "Describe this image from a WhatsApp chat. Return JSON with:\n"
        "  description: 2-3 sentences on what is shown, including setting, "
        "people (count and what they are doing, never guess identities), and "
        "the apparent purpose of sharing it.\n"
        "  ocr_text: every piece of text visible in the image, verbatim. Empty "
        "string if there is none.\n"
        "  objects: 3-10 short tags for the salient things present.\n"
        "  category: one of photo, screenshot, meme, document, receipt, "
        "poster, chart, selfie, group_photo, product, other.\n"
        "Be concrete and factual. Do not speculate about who people are."
    ),
    "sticker": (
        "This is a WhatsApp sticker. Return JSON with:\n"
        "  description: one sentence on what it depicts and the emotion or "
        "reaction it conveys.\n"
        "  ocr_text: any text on the sticker, verbatim.\n"
        "  objects: 2-5 tags.\n"
        "  category: sticker"
    ),
    "voice": (
        "This is a WhatsApp voice note. Return JSON with:\n"
        "  transcript: a verbatim transcript. Preserve the original language; "
        "if it is code-mixed (for example Hindi and English together), "
        "transcribe it as spoken rather than translating.\n"
        "  description: one sentence summarising what the speaker says.\n"
        "  objects: 2-5 topic tags.\n"
        "  category: voice_note\n"
        "If there is no intelligible speech, set transcript to an empty string "
        "and say so in description."
    ),
    "audio": (
        "This is an audio file from a chat. Return JSON with:\n"
        "  transcript: verbatim transcript of any speech, in the original "
        "language. Empty string if it is music or noise.\n"
        "  description: one or two sentences on what the audio is.\n"
        "  objects: 2-5 tags.\n"
        "  category: one of voice_note, music, recording, other"
    ),
    "video": (
        "This is a video from a WhatsApp chat. Return JSON with:\n"
        "  description: 2-4 sentences on what happens, including setting and "
        "visible actions.\n"
        "  transcript: verbatim transcript of any speech, original language. "
        "Empty string if silent.\n"
        "  ocr_text: any on-screen text, verbatim.\n"
        "  objects: 3-10 tags.\n"
        "  category: one of clip, screen_recording, meme, event, other"
    ),
    "document": (
        "This is a document shared in a chat. Return JSON with:\n"
        "  description: 2-4 sentences summarising what the document is and "
        "what it contains.\n"
        "  ocr_text: the key text content, up to roughly 2000 characters. "
        "Prioritise names, dates, amounts and identifiers.\n"
        "  objects: 3-8 tags.\n"
        "  category: one of invoice, ticket, report, form, notes, slides, "
        "spreadsheet, other"
    ),
}

_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "description": {"type": "STRING"},
        "transcript": {"type": "STRING"},
        "ocr_text": {"type": "STRING"},
        "objects": {"type": "ARRAY", "items": {"type": "STRING"}},
        "category": {"type": "STRING"},
    },
    "required": ["description"],
}


@dataclass
class Understanding:
    """What the model made of one file."""

    description: str = ""
    transcript: str = ""
    ocr_text: str = ""
    objects: list[str] = field(default_factory=list)
    category: str = ""
    model_used: str = ""
    status: str = "done"          # done | skipped | error
    error: str = ""

    def as_row(self) -> tuple:
        return (
            self.description or None,
            self.transcript or None,
            self.ocr_text or None,
            self.objects or None,
            self.status,
            self.error or None,
            self.model_used or None,
        )


@dataclass
class MediaProgress:
    total: int = 0
    done: int = 0
    cached: int = 0
    skipped: int = 0
    errors: int = 0
    quota_reached: bool = False
    quota_message: str = ""

    def as_dict(self) -> dict:
        return {
            "total": self.total,
            "done": self.done,
            "cached": self.cached,
            "skipped": self.skipped,
            "errors": self.errors,
            "quota_reached": self.quota_reached,
            "quota_message": self.quota_message,
            "pct": round(100 * self.done / self.total, 1) if self.total else 100.0,
        }


def _mime_for(mf: MediaFile) -> str:
    """
    Best-effort MIME type.

    WhatsApp voice notes are Ogg Opus but Python's mimetypes does not know
    .opus, and Gemini rejects an empty type, so the common chat formats are
    spelled out.
    """
    explicit = {
        ".opus": "audio/ogg",
        ".ogg": "audio/ogg",
        ".m4a": "audio/mp4",
        ".amr": "audio/amr",
        ".3gp": "video/3gpp",
        ".3gpp": "video/3gpp",
        ".webp": "image/webp",
        ".heic": "image/heic",
        ".heif": "image/heif",
        ".mkv": "video/x-matroska",
    }
    if mf.ext in explicit:
        return explicit[mf.ext]
    guessed, _ = mimetypes.guess_type(mf.path.name)
    return guessed or "application/octet-stream"


def _parse_vcard(path: Path) -> Understanding:
    """Contact cards are plain text; reading them locally costs nothing."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return Understanding(status="error", error=str(exc))

    names, phones = [], []
    for line in text.splitlines():
        if line.upper().startswith("FN:"):
            names.append(line[3:].strip())
        elif line.upper().startswith("TEL"):
            _, _, number = line.partition(":")
            phones.append(number.strip())

    who = ", ".join(names) or "an unnamed contact"
    return Understanding(
        description=f"Shared contact card for {who}."
        + (f" Phone: {', '.join(phones)}." if phones else ""),
        ocr_text=text[:2000],
        objects=["contact card"] + names[:3],
        category="contact",
        model_used="local",
    )


def _transcribe_with_speech_provider(mf: MediaFile) -> Understanding | None:
    """
    Try a dedicated speech-to-text provider for audio.

    Returns None when none is configured, so the Gemini path runs unchanged.
    """
    from . import transcribe as speech

    if not speech.is_configured():
        return None

    result = speech.transcribe_file(mf.path)
    if result.error:
        # Fall through to Gemini rather than losing the voice note entirely.
        print(f"[media] speech provider failed on {mf.filename}: "
              f"{result.error[:160]}")
        return None

    if not result.text:
        return Understanding(
            description="Voice note with no intelligible speech.",
            transcript="", category="voice_note",
            model_used=result.model or "whisper", status="done")

    summary = result.text.strip()
    if len(summary) > 200:
        summary = summary[:197].rstrip() + "..."

    return Understanding(
        description=f"Voice note: {summary}",
        transcript=result.text.strip(),
        objects=[],
        category="voice_note",
        model_used=result.model or "whisper",
        status="done",
    )


def describe_file(mf: MediaFile) -> Understanding:
    """Send one file to Gemini and parse the structured response."""
    if mf.kind == "contact":
        return _parse_vcard(mf.path)

    # Audio goes to a dedicated speech model when one is configured: Whisper is
    # better at this than a general multimodal model, and on Groq it is free.
    if mf.kind in ("voice", "audio") and settings.transcribe_audio:
        spoken = _transcribe_with_speech_provider(mf)
        if spoken is not None:
            return spoken

    if not mf.readable_by_gemini:
        return Understanding(
            status="skipped",
            error=f"{mf.ext or 'no extension'} is not a format Gemini can read",
            description="",
        )

    if mf.kind == "video" and mf.size_bytes > settings.max_video_mb * 1024 * 1024:
        return Understanding(
            status="skipped",
            error=f"video larger than MAX_VIDEO_MB ({settings.max_video_mb} MB)",
        )

    if mf.kind in ("voice", "audio") and not settings.transcribe_audio:
        return Understanding(status="skipped", error="audio transcription disabled")

    prompt = _PROMPTS.get(mf.kind, _PROMPTS["image"])
    client = get_client()
    from google.genai import types

    uploaded = None
    try:
        if mf.needs_upload:
            # Too big to inline: hand it to the Files API and reference it.
            uploaded = with_retry(
                lambda: client.files.upload(file=str(mf.path)),
                label=f"upload {mf.filename}",
                model=settings.vision_model,
            )
            uploaded = _await_file_active(client, uploaded)
            part = uploaded
        else:
            part = types.Part.from_bytes(
                data=mf.path.read_bytes(),
                mime_type=_mime_for(mf),
            )

        def _call():
            return client.models.generate_content(
                model=settings.vision_model,
                contents=[part, prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=_RESPONSE_SCHEMA,
                    temperature=0.1,
                ),
            )

        response = with_retry(_call, label=f"describe {mf.filename}",
                              model=settings.vision_model)
        payload = json.loads(response.text or "{}")

        return Understanding(
            description=(payload.get("description") or "").strip(),
            transcript=(payload.get("transcript") or "").strip(),
            ocr_text=(payload.get("ocr_text") or "").strip(),
            objects=[str(o) for o in (payload.get("objects") or [])][:12],
            category=(payload.get("category") or "").strip(),
            model_used=settings.vision_model,
            status="done",
        )

    except json.JSONDecodeError as exc:
        return Understanding(status="error", error=f"bad JSON from model: {exc}",
                             model_used=settings.vision_model)
    except DailyQuotaReached:
        raise                       # terminal for the run; handled by the caller
    except Exception as exc:  # noqa: BLE001 - one bad file must not stop ingest
        return Understanding(status="error", error=str(exc)[:500],
                             model_used=settings.vision_model)
    finally:
        if uploaded is not None:
            try:
                client.files.delete(name=uploaded.name)
            except Exception:  # noqa: BLE001 - cleanup is best effort
                pass


def _await_file_active(client, file_obj, timeout: float = 300.0):
    """
    Wait for an uploaded file to finish processing.

    Video uploads return immediately in a PROCESSING state; generating against
    them before they are ACTIVE fails with an opaque error.
    """
    import time

    deadline = time.time() + timeout
    while time.time() < deadline:
        state = str(getattr(file_obj.state, "name", file_obj.state) or "")
        if state == "ACTIVE":
            return file_obj
        if state == "FAILED":
            raise RuntimeError(f"upload failed for {file_obj.name}")
        time.sleep(2.0)
        file_obj = client.files.get(name=file_obj.name)
    raise TimeoutError(f"file {file_obj.name} still processing after {timeout}s")


def process_media(
    files: list[MediaFile],
    conn,
    *,
    on_progress: Callable[[MediaProgress], None] | None = None,
) -> MediaProgress:
    """
    Describe every file, using and filling the content-hash cache.

    Results are written per file inside a lock rather than batched at the end,
    so killing the process loses at most the in-flight calls.
    """
    progress = MediaProgress(total=len(files))
    if not files:
        return progress

    # The cache is shared by every archive on the device: the same forwarded
    # photo appearing in two chats is one API call, not two.
    cache_conn = get_cache_connection()
    cached_rows = cache_conn.execute(
        "SELECT content_hash, description, transcript, ocr_text, "
        "detected_objects, model_used FROM media_cache"
    ).fetchall()
    cache = {
        row[0]: Understanding(
            description=row[1] or "", transcript=row[2] or "",
            ocr_text=row[3] or "", objects=list(row[4] or []),
            model_used=row[5] or "", status="done",
        )
        for row in cached_rows
    }

    write_lock = Lock()

    def _persist(mf: MediaFile, u: Understanding, from_cache: bool) -> None:
        with write_lock:
            conn.execute(
                "UPDATE media SET description=?, transcript=?, ocr_text=?, "
                "detected_objects=?, status=?, error=?, model_used=?, "
                "processed_at=? WHERE content_hash=?",
                [*u.as_row(), datetime.now(), mf.content_hash],
            )
            if not from_cache and u.status == "done":
                cache_conn.execute(
                    "INSERT INTO media_cache (content_hash, kind, description, "
                    "transcript, ocr_text, detected_objects, model_used) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT (content_hash) "
                    "DO NOTHING",
                    [mf.content_hash, mf.kind, u.description or None,
                     u.transcript or None, u.ocr_text or None,
                     u.objects or None, u.model_used or None],
                )

            progress.done += 1
            if from_cache:
                progress.cached += 1
            elif u.status == "skipped":
                progress.skipped += 1
            elif u.status == "error":
                progress.errors += 1
        if on_progress:
            on_progress(progress)

    pending: list[MediaFile] = []
    for mf in files:
        hit = cache.get(mf.content_hash)
        if hit is not None:
            _persist(mf, hit, from_cache=True)
        else:
            pending.append(mf)

    if not pending:
        return progress

    # Deduplicate within this run too: the same photo forwarded twice is one
    # API call, not two.
    unique: dict[str, MediaFile] = {}
    duplicates: list[MediaFile] = []
    for mf in pending:
        if mf.content_hash in unique:
            duplicates.append(mf)
        else:
            unique[mf.content_hash] = mf

    with ThreadPoolExecutor(max_workers=max(1, settings.media_concurrency)) as pool:
        futures = {pool.submit(describe_file, mf): mf for mf in unique.values()}
        for fut in as_completed(futures):
            mf = futures[fut]
            try:
                understanding = fut.result()
            except DailyQuotaReached as exc:
                # Out of budget for today. Everything already described is
                # committed and cached, so tomorrow's run resumes from here.
                progress.quota_reached = True
                progress.quota_message = str(exc)
                for pending_future in futures:
                    pending_future.cancel()
                break
            except Exception as exc:  # noqa: BLE001
                understanding = Understanding(status="error", error=str(exc)[:500])
            cache[mf.content_hash] = understanding
            _persist(mf, understanding, from_cache=False)

    for mf in duplicates:
        _persist(mf, cache.get(mf.content_hash, Understanding(status="error")),
                 from_cache=True)

    return progress


def estimate_cost(files: list[MediaFile]) -> dict:
    """
    Rough sizing so a large ingest is a decision, not a surprise.

    Token counts are approximations from Google's published guidance
    (~258 tokens an image, ~32 tokens a second of audio, ~300 a second of
    video at 1 fps); actual billing will differ. Duration is inferred from
    file size, which is crude but enough to tell 50 files from 5000.
    """
    by_kind: dict[str, dict] = {}
    est_tokens = 0

    for mf in files:
        bucket = by_kind.setdefault(mf.kind, {"count": 0, "bytes": 0})
        bucket["count"] += 1
        bucket["bytes"] += mf.size_bytes

        if not mf.readable_by_gemini:
            continue
        if mf.kind in ("image", "sticker"):
            est_tokens += 300
        elif mf.kind in ("voice", "audio"):
            seconds = mf.size_bytes / 2000          # ~16 kbps opus
            est_tokens += int(seconds * 32)
        elif mf.kind == "video":
            seconds = mf.size_bytes / 250_000        # ~2 Mbps
            est_tokens += int(seconds * 300)
        elif mf.kind == "document":
            est_tokens += 3000

    return {
        "file_count": len(files),
        "readable_count": sum(1 for f in files if f.readable_by_gemini),
        "by_kind": by_kind,
        "estimated_input_tokens": est_tokens,
        "note": "Rough estimate from file sizes, not a quote. Check current "
                "pricing at ai.google.dev/pricing before a large run.",
    }
