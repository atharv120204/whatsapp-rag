"""
Work an archive still needs done, and what it will cost to do it.

All of this was already reachable from the CLI. The reason it is also an API is
that the person deciding whether to spend four hundred Gemini calls is sitting
in front of a browser, and a prompt nobody sees is a prompt nobody answers.

Cost is quoted in **API requests**, not money. On the free tier requests are the
scarce thing, and the daily allowance is what decides whether a job finishes
this afternoon or over three weeks. Quoting anything else would be dishonest
about which number actually constrains the work.

Two properties make it safe to offer these as one-click actions:

  * every unit of work is cached by content hash, so a job stopped by the daily
    cap resumes tomorrow and re-pays for nothing, and
  * both jobs only ever look for *missing* results, so running one twice is a
    no-op rather than a second bill.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..archives import Archive
from ..config import settings
from ..index.ratelimit import limiter
from ..parse.media import INLINE_LIMIT_BYTES, MediaFile

# One Gemini vision request per distinct file. A video above the inline limit
# costs a second request to upload it first.
VISION_KINDS = ("image", "sticker", "video", "document")

# Voice and audio go to the speech provider (Groq Whisper, free and separately
# limited); a contact card is parsed on this machine. None of them touch the
# Gemini daily budget, so quoting them as a cost would overstate it.
FREE_KINDS = ("voice", "audio", "contact")

_PENDING_WHERE = """
    readable
    AND coalesce(description, '') = ''
    AND coalesce(transcript, '') = ''
"""


# --- media ----------------------------------------------------------------------

def pending_media_by_kind(conn) -> list[dict[str, Any]]:
    """
    Undescribed attachments, grouped by kind and counted by distinct file.

    Distinct file, not row: the same photo forwarded into the chat four times is
    four rows and one API call. Counting rows would inflate the quoted cost by
    about a quarter on this archive, and the whole point of the number is that
    it can be trusted.
    """
    rows = conn.execute(f"""
        SELECT kind,
               count(DISTINCT content_hash)              AS files,
               count(*)                                  AS rows_,
               round(sum(size_bytes) / 1048576.0, 1)     AS mb,
               count(DISTINCT CASE WHEN size_bytes > {INLINE_LIMIT_BYTES}
                                   THEN content_hash END) AS large
        FROM media
        WHERE {_PENDING_WHERE}
        GROUP BY kind
        ORDER BY files DESC
    """).fetchall()

    out = []
    for kind, files, row_count, mb, large in rows:
        gemini = kind not in FREE_KINDS
        out.append({
            "kind": kind,
            "files": int(files),
            "rows": int(row_count),
            "mb": float(mb or 0),
            # The upload of an oversized file is a request of its own.
            "requests": (int(files) + int(large or 0)) if gemini else 0,
            "provider": "gemini" if gemini else "local-or-groq",
        })
    return out


def pending_media_files(conn, kinds: list[str] | None = None) -> list[MediaFile]:
    """
    Rebuild `MediaFile` objects for attachments that still need describing.

    Reconstructed from the media table rather than by re-reading the export,
    because the export may be long gone -- the files live in the archive's own
    media directory now. `needs_upload` is derived from the stored size for the
    same reason the catalogue derives it: it is a function of the file, not a
    fact about this run.
    """
    filters = [_PENDING_WHERE]
    params: list[Any] = []
    if kinds:
        placeholders = ", ".join("?" * len(kinds))
        filters.append(f"kind IN ({placeholders})")
        params.extend(kinds)

    rows = conn.execute(f"""
        SELECT any_value(filename), any_value(path), kind, any_value(ext),
               max(size_bytes), content_hash
        FROM media
        WHERE {" AND ".join(filters)}
        GROUP BY content_hash, kind
        ORDER BY kind, content_hash
    """, params).fetchall()

    files: list[MediaFile] = []
    for filename, path, kind, ext, size_bytes, content_hash in rows:
        location = Path(path)
        # A file listed in the table but gone from disk would otherwise fail one
        # request at a time. Skip it here so the quoted cost stays honest.
        if not location.exists():
            continue
        files.append(MediaFile(
            filename=filename,
            path=location,
            kind=kind,
            ext=ext or location.suffix.lower(),
            size_bytes=int(size_bytes or 0),
            content_hash=content_hash,
            readable_by_gemini=True,
            needs_upload=int(size_bytes or 0) > INLINE_LIMIT_BYTES,
        ))
    return files


# --- survey ---------------------------------------------------------------------

def _embed_task(conn) -> dict[str, Any] | None:
    total = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
    if not total:
        return None

    pending = conn.execute("""
        SELECT count(*) FROM chunks c
        LEFT JOIN chunk_vectors v USING (chunk_id)
        WHERE v.chunk_id IS NULL
    """).fetchone()[0]

    batch = max(1, settings.embed_batch_size)
    requests = -(-pending // batch)      # ceil

    return {
        "task": "embed",
        "title": "Build semantic search",
        "pending": pending,
        "total": total,
        "unit": "conversation windows",
        "requests": requests,
        "detail": [],
        "why": (
            "Without vectors, search matches words only: asking about \"the "
            "hotel thing\" will not find \"that place we booked\". Embedding "
            "adds meaning-based retrieval alongside the keyword index."
        ),
        "cost_note": (
            f"{pending:,} windows in batches of {batch} is about "
            f"{requests} Gemini request{'s' if requests != 1 else ''}."
        ),
    }


def _media_task(conn) -> dict[str, Any] | None:
    by_kind = pending_media_by_kind(conn)
    if not by_kind:
        return None

    files = sum(k["files"] for k in by_kind)
    requests = sum(k["requests"] for k in by_kind)
    described = conn.execute(
        "SELECT count(DISTINCT content_hash) FROM media "
        "WHERE coalesce(description, '') <> '' OR coalesce(transcript, '') <> ''"
    ).fetchone()[0]

    return {
        "task": "describe_media",
        "title": "Describe photos, video and voice notes",
        "pending": files,
        "total": files + described,
        "unit": "files",
        "requests": requests,
        "detail": by_kind,
        "why": (
            "An undescribed photo is invisible to search and reads as "
            "\"[image sent, not yet described]\" to the agent. Describing it "
            "makes its contents searchable and quotable."
        ),
        "cost_note": (
            f"{files:,} distinct files, {requests:,} Gemini "
            f"request{'s' if requests != 1 else ''} — duplicates are one call, "
            "and voice notes go to the speech provider instead."
        ),
    }


def survey(conn, archive: Archive) -> dict[str, Any]:
    """Everything the UI needs to ask "shall I?" without a second round trip."""
    usage = limiter.snapshot().as_dict()

    # Each task is gated by its own model's budget, not the summed figure.
    # Quoting the total was what let a spent vision quota advertise itself as
    # 180 embedding requests still available.
    budget_for = {
        "embed": limiter.snapshot(settings.embed_model).as_dict(),
        "describe_media": limiter.snapshot(settings.vision_model).as_dict(),
    }

    found = [t for t in (_embed_task(conn), _media_task(conn)) if t]

    # Finished work is reported as a one-line reassurance rather than as a task
    # with nothing to do, so the UI never has to render "0 of 541 pending".
    tasks = [t for t in found if t["pending"] > 0]
    complete = [
        f"{t['title']}: all {t['total']:,} {t['unit']} done"
        for t in found if t["pending"] == 0
    ]

    for task in tasks:
        warnings: list[str] = []
        requests = task["requests"]
        model_usage = budget_for[task["task"]]
        remaining = model_usage.get("remaining_today")
        task["model"] = model_usage.get("model")
        task["remaining_today"] = remaining

        if not settings.has_api_key:
            task["runnable"] = False
            task["blocked_reason"] = (
                "No Gemini API key on this device. Add one on the Settings tab."
            )
        elif remaining == 0 and requests > 0:
            # Offering a button that can only do nothing is worse than saying
            # so: the daily allowance resets, and nothing already done is lost.
            task["runnable"] = False
            task["blocked_reason"] = (
                "This API key's budget for today is used up, so the job would "
                "not get through a single file. It resumes where it left off "
                "tomorrow — everything finished so far is cached and never "
                "re-paid for. A different key on the Settings tab has its own "
                "quota and would start again immediately."
            )
        else:
            task["runnable"] = True
            task["blocked_reason"] = None

        # `remaining > 0` because a used-up budget is already stated, more
        # usefully, as the blocked reason above.
        if remaining is not None and remaining > 0 and requests > remaining:
            warnings.append(
                f"Your daily budget has {remaining} request"
                f"{'s' if remaining != 1 else ''} left, so this will get "
                f"through roughly {remaining} of {requests} today and then "
                "stop cleanly. Run it again tomorrow — finished work is cached "
                "and never re-paid for."
            )

        # The configured cap is a stop-loss, not the real quota. Gemini's own
        # free-tier allowance on the vision model is far lower, and saying so
        # is the difference between "this takes a while" and a user who thinks
        # the job is broken on day two.
        if task["task"] == "describe_media" and requests > 20:
            warnings.append(
                f"Gemini's free tier allows roughly 20 vision requests a day, "
                f"so {requests:,} files is on the order of "
                f"{max(1, round(requests / 20))} days of runs. Deselect the "
                "kinds you do not need first."
            )

        task["warnings"] = warnings

    return {
        "archive_id": archive.archive_id,
        "archive_name": archive.name,
        "api_key_configured": settings.has_api_key,
        "usage": usage,
        "tasks": tasks,
        "complete": complete,
        "pending_total": sum(t["pending"] for t in tasks),
    }
