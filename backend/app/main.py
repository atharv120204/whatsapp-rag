"""
FastAPI application.

Every data endpoint is scoped to an archive by an `archive` query parameter.
There is no server-side "current archive": with several chats loaded, hidden
state is how you end up answering a question about the wrong one, and two
browser tabs should be able to look at different archives at once.
"""

from __future__ import annotations

import json
import shutil
import threading
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from . import archives as archive_store
from . import settings_store
from .api import stats as stats_api
from .archives import Archive, ArchiveNotFound
from .config import settings
from .db import get_connection, get_cursor, get_meta, set_meta


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings.ensure_dirs()
    settings_store.apply_to_settings()
    yield
    from .db import close_all

    close_all()


app = FastAPI(title="Chat Archive", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# One ingest at a time per archive; the UI polls this for progress.
_ingest_state: dict[str, dict[str, Any]] = {}
_ingest_lock = threading.Lock()


def _resolve(archive_id: str | None) -> Archive:
    try:
        return archive_store.resolve(archive_id)
    except ArchiveNotFound as exc:
        raise HTTPException(404, f"Archive not found: {exc}") from exc


def _conn(archive_id: str | None):
    """A fresh cursor per request; see db.get_cursor for why."""
    return get_cursor(_resolve(archive_id))


# --- models -------------------------------------------------------------------

class ChatTurn(BaseModel):
    role: str
    text: str


class ChatRequest(BaseModel):
    question: str
    history: list[ChatTurn] = []
    archive: str | None = None


class CreateArchiveRequest(BaseModel):
    name: str


class RenameArchiveRequest(BaseModel):
    name: str


class RenameParticipantRequest(BaseModel):
    participant_id: str
    display_name: str


class SettingsRequest(BaseModel):
    gemini_api_key: str | None = None
    chat_provider: str | None = None
    chat_base_url: str | None = None
    chat_api_key: str | None = None
    speech_provider: str | None = None
    speech_model: str | None = None
    speech_base_url: str | None = None
    speech_api_key: str | None = None
    speech_language: str | None = None
    chat_model: str | None = None
    vision_model: str | None = None
    embed_model: str | None = None
    session_gap_hours: float | None = None
    describe_media: bool | None = None
    transcribe_audio: bool | None = None
    media_concurrency: int | None = None
    max_requests_per_minute: int | None = None
    max_requests_per_day: int | None = None


# --- health, settings ----------------------------------------------------------

@app.get("/api/health")
def health() -> dict:
    return {"ok": True}


@app.get("/api/settings")
def get_settings() -> dict:
    return settings_store.public_view()


@app.post("/api/settings")
def update_settings(request: SettingsRequest) -> dict:
    """
    Save per-device settings.

    Everyone runs their own copy with their own key, so this exists to avoid
    making a text editor a prerequisite for using the app.
    """
    updates = {k: v for k, v in request.model_dump().items() if v is not None}
    if not updates:
        return settings_store.public_view()
    try:
        settings_store.save(updates)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return settings_store.public_view()


@app.get("/api/usage")
def api_usage() -> dict:
    """How much of today's self-imposed API budget has been spent."""
    from .index.ratelimit import limiter

    return limiter.snapshot().as_dict()


@app.get("/api/config/check")
def config_check() -> dict:
    """Verify the configured chat provider and model actually work."""
    from .agent.llm import LLMError, build_provider

    result: dict = {"chat": {}, "gemini": {}}

    try:
        provider = build_provider()
        result["chat"] = {"ok": True, "provider": provider.name, "models": []}
        if hasattr(provider, "list_models"):
            try:
                models = provider.list_models()
                result["chat"]["models"] = models
                result["chat"]["model_available"] = settings.chat_model in models
            except LLMError as exc:
                result["chat"]["models_error"] = str(exc)
    except LLMError as exc:
        result["chat"] = {"ok": False, "error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        result["chat"] = {"ok": False, "error": str(exc)[:300]}

    # Gemini is checked separately: it powers embeddings and media even when
    # chat has been moved elsewhere.
    if settings.has_api_key:
        from .index.gemini import check_config

        result["gemini"] = check_config()
    else:
        result["gemini"] = {
            "ok": False,
            "error": "No Gemini API key: semantic search and media "
                     "understanding are unavailable.",
        }

    from .index.transcribe import check as check_speech

    result["speech"] = check_speech()
    result["ok"] = bool(result["chat"].get("ok"))
    return result


# --- archives ------------------------------------------------------------------

@app.get("/api/archives")
def list_archives() -> dict:
    return {"archives": [a.as_dict() for a in archive_store.list_archives()]}


@app.post("/api/archives")
def create_archive(request: CreateArchiveRequest) -> dict:
    archive = archive_store.create_archive(request.name)
    return archive.as_dict()


@app.get("/api/archives/{archive_id}")
def get_archive(archive_id: str) -> dict:
    archive = _resolve(archive_id)
    conn = get_connection(archive)
    try:
        counts = conn.execute("""
            SELECT
                (SELECT COUNT(*) FROM messages),
                (SELECT COUNT(*) FROM participants),
                (SELECT COUNT(*) FROM media),
                (SELECT COUNT(*) FROM chunks),
                (SELECT COUNT(*) FROM chunk_vectors)
        """).fetchone()
    except Exception:  # noqa: BLE001 - empty archive
        counts = (0, 0, 0, 0, 0)

    overview_raw = get_meta("overview", conn)
    payload = archive.as_dict()
    payload.update({
        "ingested": bool(counts[0]),
        "messages": counts[0],
        "participants": counts[1],
        "media": counts[2],
        "chunks": counts[3],
        "embeddings": counts[4],
        "semantic_search_ready": bool(counts[4]),
        "api_key_configured": settings.has_api_key,
        "overview": json.loads(overview_raw) if overview_raw else None,
    })
    return payload


@app.patch("/api/archives/{archive_id}")
def rename_archive(archive_id: str, request: RenameArchiveRequest) -> dict:
    try:
        return archive_store.rename_archive(archive_id, request.name).as_dict()
    except ArchiveNotFound as exc:
        raise HTTPException(404, str(exc)) from exc


@app.delete("/api/archives/{archive_id}")
def delete_archive(archive_id: str) -> dict:
    """Permanently delete an archive, its database and all its media."""
    with _ingest_lock:
        state = _ingest_state.get(archive_id)
        if state and state.get("running"):
            raise HTTPException(409, "An ingest is running on this archive.")

    try:
        archive_store.delete_archive(archive_id)
    except ArchiveNotFound as exc:
        raise HTTPException(404, str(exc)) from exc
    except OSError as exc:
        raise HTTPException(500, str(exc)) from exc

    with _ingest_lock:
        _ingest_state.pop(archive_id, None)
    return {"deleted": archive_id}


# --- ingest --------------------------------------------------------------------

def _run_ingest(path: Path, archive: Archive, mode: str,
                describe_media: bool, embed: bool) -> None:
    from .index.build import ingest

    def progress(stage: str, message: str, detail: dict) -> None:
        with _ingest_lock:
            state = _ingest_state.setdefault(archive.archive_id, {})
            state.update(stage=stage, message=message, detail=detail or {})

    try:
        result = ingest(path, archive, mode=mode, describe_media=describe_media,
                        embed=embed, progress=progress)
        with _ingest_lock:
            state = _ingest_state.setdefault(archive.archive_id, {})
            state["result"] = result.as_dict()
            state["error"] = None if result.ok else "; ".join(result.errors)
    except Exception as exc:  # noqa: BLE001 - surfaced to the UI
        # Keep the traceback, not just the message. A job that failed with a
        # bare "tuple index out of range" cost a whole debugging session and
        # was never diagnosed, because the only copy went to a console
        # nobody was watching and had scrolled away by the time it mattered.
        detail = traceback.format_exc()
        print(detail)
        with _ingest_lock:
            state = _ingest_state.setdefault(archive.archive_id, {})
            state["error"] = str(exc)
            state["traceback"] = detail
    finally:
        with _ingest_lock:
            state = _ingest_state.setdefault(archive.archive_id, {})
            state["running"] = False
            state["stage"] = "done"


@app.post("/api/ingest/upload")
async def ingest_upload(
    file: UploadFile = File(...),
    archive: str | None = Query(None),
    archive_name: str | None = Query(None),
    mode: str = Query("replace", pattern="^(replace|merge)$"),
    describe_media: bool = Query(True),
    embed: bool = Query(True),
) -> dict:
    """
    Accept an export and start ingesting it.

    Pass `archive` to load into an existing one, or `archive_name` to create a
    new one. `mode=merge` deduplicates against what is already there;
    `mode=replace` discards it first.

    The upload is streamed to disk rather than read into memory: a with-media
    export of a busy group runs to gigabytes.
    """
    if archive:
        target_archive = _resolve(archive)
    else:
        name = (archive_name or Path(file.filename or "Chat").stem).strip()
        target_archive = archive_store.create_archive(name)

    with _ingest_lock:
        state = _ingest_state.get(target_archive.archive_id)
        if state and state.get("running"):
            raise HTTPException(409, "An ingest is already running on this archive.")

    from .parse.media import safe_filename

    name = safe_filename(file.filename or "export.zip", fallback="export.zip")
    if Path(name).suffix.lower() not in (".zip", ".txt"):
        raise HTTPException(400, "Upload a .zip or .txt WhatsApp export.")

    target_archive.ensure_dirs()
    destination = target_archive.raw_dir / name
    with destination.open("wb") as out:
        shutil.copyfileobj(file.file, out, length=1024 * 1024)

    with _ingest_lock:
        _ingest_state[target_archive.archive_id] = {
            "running": True, "stage": "queued", "message": "Starting",
            "detail": {}, "result": None, "error": None,
            "archive_id": target_archive.archive_id, "mode": mode,
        }

    threading.Thread(
        target=_run_ingest,
        args=(destination, target_archive, mode, describe_media, embed),
        daemon=True,
    ).start()

    return {
        "started": True,
        "archive_id": target_archive.archive_id,
        "archive_name": target_archive.name,
        "mode": mode,
        "filename": name,
        "size_bytes": destination.stat().st_size,
    }


@app.post("/api/ingest/sample")
def ingest_sample(archive_name: str = Query("Sample group chat")) -> dict:
    """Generate and ingest a synthetic chat into a new archive."""
    from .sample_data import generate

    settings.ensure_dirs()
    target_archive = archive_store.create_archive(archive_name)
    target_archive.ensure_dirs()

    path, truth = generate(target_archive.raw_dir)

    with _ingest_lock:
        _ingest_state[target_archive.archive_id] = {
            "running": True, "stage": "queued", "message": "Starting",
            "detail": {}, "result": None, "error": None,
            "archive_id": target_archive.archive_id, "mode": "replace",
        }

    threading.Thread(
        target=_run_ingest,
        args=(path, target_archive, "replace", True, True),
        daemon=True,
    ).start()
    return {
        "started": True,
        "archive_id": target_archive.archive_id,
        "archive_name": target_archive.name,
        "ground_truth": truth,
    }


@app.get("/api/ingest/status")
def ingest_status(archive: str | None = Query(None)) -> dict:
    with _ingest_lock:
        if archive:
            return dict(_ingest_state.get(archive, {"running": False}))
        running = [dict(v) for v in _ingest_state.values() if v.get("running")]
        return {"running": bool(running), "jobs": running}


# --- maintenance ----------------------------------------------------------------
#
# Embedding and media description used to be reachable only from the CLI, which
# in practice meant they did not happen: the person who has to weigh 412 API
# calls is in the browser. These two endpoints put the decision where the user
# is, with the request cost attached.
#
# They deliberately share `_ingest_state` with ingest. One writer per archive is
# not a nicety -- DuckDB allows a single read-write process per file, and both
# jobs write. Sharing the slot means the 409 below is the only guard needed.

def _run_maintenance(archive: Archive, task: str, kinds: list[str] | None) -> None:
    from .api.maintenance import pending_media_files

    def progress(stage: str, message: str, detail: dict) -> None:
        with _ingest_lock:
            state = _ingest_state.setdefault(archive.archive_id, {})
            state.update(stage=stage, message=message, detail=detail or {})

    try:
        conn = get_connection(archive)
        if task == "embed":
            from .index.embed import embed_chunks

            progress("embed", "Embedding conversation windows", {})
            stats = embed_chunks(conn, on_progress=lambda done, total: progress(
                "embed", f"Embedding {done} of {total}",
                {"done": done, "total": total,
                 "pct": round(100 * done / total, 1) if total else 100.0},
            ))
            summary = dict(stats)
            covered, total = conn.execute(
                "SELECT (SELECT count(*) FROM chunk_vectors), "
                "(SELECT count(*) FROM chunks)").fetchone()
            summary["message"] = (
                f"Semantic search now covers {covered:,} of {total:,} windows."
            )
        else:
            from .index.media_understanding import process_media

            files = pending_media_files(conn, kinds)
            progress("media", f"Describing {len(files)} files", {})
            state = process_media(files, conn, on_progress=lambda p: progress(
                "media", f"Described {p.done} of {p.total}", p.as_dict()))
            summary = state.as_dict()

            # `done` counts files written, including the ones written as a
            # failure. Reporting those as described would be a lie the Media
            # tab immediately contradicts, so they are counted separately.
            described = state.done - state.cached - state.errors - state.skipped
            parts = [f"{described} newly described"]
            if state.cached:
                parts.append(f"{state.cached} already known")
            if state.errors:
                parts.append(f"{state.errors} could not be read")
            if state.skipped:
                parts.append(f"{state.skipped} skipped")
            summary["message"] = ", ".join(parts) + "."

        with _ingest_lock:
            job = _ingest_state.setdefault(archive.archive_id, {})
            job["result"] = {"ok": True, "task": task, "summary": summary}
            # Running out of budget is the designed ending, not a failure. It
            # goes to `notice` so the UI can say so calmly instead of in red.
            job["notice"] = summary.get("quota_message") or None
            job["error"] = None
    except Exception as exc:  # noqa: BLE001 - surfaced to the UI
        # Keep the traceback, not just the message. A job that failed with a
        # bare "tuple index out of range" cost a whole debugging session and
        # was never diagnosed, because the only copy went to a console
        # nobody was watching and had scrolled away by the time it mattered.
        detail = traceback.format_exc()
        print(detail)
        with _ingest_lock:
            state = _ingest_state.setdefault(archive.archive_id, {})
            state["error"] = str(exc)
            state["traceback"] = detail
    finally:
        with _ingest_lock:
            job = _ingest_state.setdefault(archive.archive_id, {})
            job["running"] = False
            job["stage"] = "done"


@app.get("/api/maintenance")
def maintenance_survey(archive: str | None = Query(None)) -> dict:
    """What still needs doing to this archive, and what it would cost."""
    from .api.maintenance import survey

    target = _resolve(archive)
    return survey(_conn(archive), target)


@app.post("/api/maintenance/run")
def maintenance_run(
    task: str = Query(..., pattern="^(embed|describe_media)$"),
    archive: str | None = Query(None),
    kinds: str | None = Query(
        None, description="Comma-separated media kinds; all pending if omitted."),
) -> dict:
    """
    Start one maintenance job. Progress is read from /api/ingest/status.

    Both jobs are resumable and idempotent: they look only for missing results,
    so a second run after the daily cap stops the first one picks up where it
    left off and re-pays for nothing.
    """
    target = _resolve(archive)

    if not settings.has_api_key:
        raise HTTPException(400, "No Gemini API key configured on this device.")

    with _ingest_lock:
        state = _ingest_state.get(target.archive_id)
        if state and state.get("running"):
            raise HTTPException(
                409, "Something is already running on this archive. "
                     "Wait for it to finish, then try again.")
        _ingest_state[target.archive_id] = {
            "running": True, "stage": "queued", "message": "Starting",
            "detail": {}, "result": None, "error": None,
            "archive_id": target.archive_id, "job": task,
        }

    selected = [k.strip() for k in (kinds or "").split(",") if k.strip()] or None

    threading.Thread(
        target=_run_maintenance,
        args=(target, task, selected),
        daemon=True,
    ).start()

    return {"started": True, "task": task, "archive_id": target.archive_id,
            "kinds": selected}


# --- chat -----------------------------------------------------------------------

@app.post("/api/chat")
def chat(request: ChatRequest) -> dict:
    from .agent.router import ask

    if not request.question.strip():
        raise HTTPException(400, "Question is empty.")

    answer = ask(request.question, [t.model_dump() for t in request.history],
                 archive=_resolve(request.archive))
    return answer.as_dict()


@app.post("/api/chat/stream")
def chat_stream(request: ChatRequest) -> StreamingResponse:
    """Server-sent events, so tool calls appear as the agent makes them."""
    from .agent.router import ask_stream

    target = _resolve(request.archive)

    def generate():
        try:
            for event in ask_stream(request.question,
                                    [t.model_dump() for t in request.history],
                                    archive=target):
                yield f"data: {json.dumps(event, default=str)}\n\n"
        except Exception as exc:  # noqa: BLE001
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# --- stats ----------------------------------------------------------------------

@app.get("/api/archives/{archive_id}/doctor")
def archive_doctor(archive_id: str) -> dict:
    """Audit an archive for inconsistencies that would make answers wrong."""
    from .doctor import check_archive

    archive = _resolve(archive_id)
    return check_archive(get_connection(archive), archive).as_dict()


@app.get("/api/insights")
def archive_insights(archive: str | None = Query(None)) -> dict:
    """Superlatives, rhythms and the most notable conversations."""
    from .api.insights import report

    return report(_conn(archive))


@app.get("/api/insights/moments")
def archive_moments(archive: str | None = Query(None),
                    kind: str = Query("funny"),
                    limit: int = Query(5, le=10)) -> dict:
    from .api.insights import find_moments

    return find_moments(_conn(archive), kind=kind, limit=limit)


@app.get("/api/stats/dashboard")
def stats_dashboard(archive: str | None = Query(None)) -> dict:
    return stats_api.dashboard(_conn(archive))


@app.get("/api/stats/leaderboard")
def stats_leaderboard(archive: str | None = Query(None)) -> list[dict]:
    return stats_api.leaderboard(_conn(archive))


@app.get("/api/stats/timeline")
def stats_timeline(archive: str | None = Query(None),
                   granularity: str = Query("month")) -> list[dict]:
    return stats_api.timeline(_conn(archive), granularity)


@app.get("/api/stats/initiation")
def stats_initiation(archive: str | None = Query(None),
                     gap_hours: float | None = Query(None)) -> dict:
    return stats_api.initiation_analysis(_conn(archive), gap_hours)


@app.get("/api/stats/heatmap")
def stats_heatmap(archive: str | None = Query(None)) -> list[dict]:
    return stats_api.activity_heatmap(_conn(archive))


@app.get("/api/stats/media")
def stats_media(archive: str | None = Query(None)) -> dict:
    return stats_api.media_breakdown(_conn(archive))


@app.get("/api/stats/streaks")
def stats_streaks(archive: str | None = Query(None)) -> dict:
    return stats_api.streaks(_conn(archive))


# --- browsing -------------------------------------------------------------------

@app.get("/api/messages")
def browse_messages(
    archive: str | None = Query(None),
    q: str | None = Query(None),
    sender: str | None = Query(None),
    msg_type: str | None = Query(None),
    after: str | None = Query(None),
    before: str | None = Query(None),
    limit: int = Query(100, le=500),
    offset: int = Query(0),
) -> dict:
    filters, params = ["1=1"], []
    if q:
        filters.append("content ILIKE ?")
        params.append(f"%{q}%")
    if sender:
        filters.append("sender ILIKE ?")
        params.append(f"%{sender}%")
    if msg_type:
        filters.append("msg_type = ?")
        params.append(msg_type)
    if after:
        filters.append("ts >= ?::TIMESTAMP")
        params.append(after)
    if before:
        filters.append("ts <= ?::TIMESTAMP")
        params.append(before)

    where = " AND ".join(filters)
    conn = _conn(archive)
    total = conn.execute(
        f"SELECT COUNT(*) FROM v_searchable WHERE {where}", params
    ).fetchone()[0]

    rows = conn.execute(f"""
        SELECT msg_id, ts, sender, msg_type, content
        FROM v_searchable WHERE {where}
        ORDER BY ts LIMIT ? OFFSET ?
    """, [*params, limit, offset]).fetchall()

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "messages": [
            {"msg_id": r[0], "ts": str(r[1]), "sender": r[2],
             "type": r[3], "text": r[4]}
            for r in rows
        ],
    }


@app.get("/api/media")
def list_media(archive: str | None = Query(None),
               kind: str | None = Query(None),
               q: str | None = Query(None),
               limit: int = Query(60, le=300)) -> dict:
    from .agent.tools import build_tools

    tools = build_tools(_conn(archive))
    return tools["find_media"](query=q, kind=kind, limit=limit)


@app.get("/api/media/{media_id}/file")
def media_file(media_id: int, archive: str | None = Query(None)):
    """
    Serve an attachment.

    The path comes from the database rather than the request, and must resolve
    inside this archive's own media directory. Serving a caller-supplied path
    would be an arbitrary file read, and allowing any archive's directory would
    let one chat's id fetch another's photos.
    """
    target = _resolve(archive)
    conn = get_connection(target)
    row = conn.execute(
        "SELECT path, filename FROM media WHERE media_id = ?", [media_id]
    ).fetchone()
    if not row:
        raise HTTPException(404, "No such media.")

    path = Path(row[0]).resolve()
    allowed_root = target.media_dir.resolve()
    try:
        path.relative_to(allowed_root)
    except ValueError as exc:
        raise HTTPException(403, "Media file is outside this archive.") from exc
    if not path.exists():
        raise HTTPException(404, "File is missing from disk.")

    return FileResponse(path, filename=row[1])


@app.get("/api/participants")
def list_participants(archive: str | None = Query(None)) -> list[dict]:
    rows = _conn(archive).execute("""
        SELECT participant_id, display_name, aliases, is_phone_only, message_count
        FROM participants ORDER BY message_count DESC
    """).fetchall()
    return [
        {"participant_id": r[0], "display_name": r[1], "aliases": list(r[2] or []),
         "is_phone_only": r[3], "message_count": r[4]}
        for r in rows
    ]


@app.post("/api/participants/rename")
def rename_participant(request: RenameParticipantRequest,
                       archive: str | None = Query(None)) -> dict:
    """Put a real name on a participant who only appears as a phone number."""
    from .index.build import compute_overview

    conn = _conn(archive)
    conn.execute("UPDATE participants SET display_name = ? WHERE participant_id = ?",
                 [request.display_name, request.participant_id])
    set_meta("overview", json.dumps(compute_overview(conn), default=str), conn)
    return {"ok": True}
