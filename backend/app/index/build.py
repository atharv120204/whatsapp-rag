"""
The ingest pipeline: export file in, queryable archive out.

  1. unpack     accept a .zip or a bare .txt
  2. parse      transcript -> messages
  3. combine    replace, or merge and deduplicate against what is there
  4. resolve    sender strings -> participants
  5. enrich     sessions, initiations, reply gaps, text features
  6. load       write messages and participants
  7. media      catalogue attachments, describe them with Gemini
  8. chunk      window into retrieval units (including media descriptions)
  9. embed      vectors for semantic search
 10. index      BM25 index and cached overview statistics

Merge mode rebuilds every derived table rather than appending to them. Session
boundaries, reply gaps and chunk windows are all properties of the complete
ordered stream, so inserting older messages in the middle changes them; a merge
that only appended would leave the archive quietly wrong. Rebuilding is
affordable because media descriptions and embeddings come from content-hash
caches, so almost nothing is recomputed.

Stages 7 and 9 need an API key; the rest do not. Without one the pipeline still
completes and every statistical question works -- only semantic search and media
understanding are missing.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

from ..archives import Archive, IngestRecord
from ..config import settings
from ..db import (bulk_insert, clear_archive_data, get_connection, init_schema,
                  rebuild_fts, set_meta)
from ..parse.media import MediaCatalog, extract_zip, find_chat_txt
from ..parse.normalize import build_alias_lookup, resolve_participants
from ..parse.sessions import enrich
from ..parse.whatsapp import parse_file
from .chunk import build_chunks, store_chunks
from .dedup import assign_keys, load_existing, merge

ProgressFn = Callable[[str, str, dict], None]


@dataclass
class IngestResult:
    ok: bool = False
    mode: str = "replace"
    archive_id: str = ""
    archive_name: str = ""
    stages: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "mode": self.mode,
            "archive_id": self.archive_id,
            "archive_name": self.archive_name,
            "stages": self.stages,
            "warnings": self.warnings,
            "errors": self.errors,
            "elapsed_seconds": round(self.elapsed_seconds, 1),
        }


def _noop(stage: str, message: str, data: dict) -> None:
    detail = f" {data}" if data else ""
    print(f"[ingest:{stage}] {message}{detail}")


def ingest(
    source: Path,
    archive: Archive,
    *,
    mode: str = "replace",
    describe_media: bool | None = None,
    embed: bool = True,
    progress: ProgressFn | None = None,
) -> IngestResult:
    """
    Load an export into an archive.

    mode="replace" discards the archive's current contents first.
    mode="merge"   deduplicates against them and keeps both.
    """
    started = time.time()
    report = progress or _noop
    result = IngestResult(mode=mode, archive_id=archive.archive_id,
                          archive_name=archive.name)
    settings.ensure_dirs()
    archive.ensure_dirs()

    conn = get_connection(archive)
    init_schema(conn)

    source = Path(source)
    if not source.exists():
        result.errors.append(f"No such file: {source}")
        return result

    # --- 1. unpack ------------------------------------------------------
    media_root: Path | None = None
    if source.suffix.lower() == ".zip":
        # Each export unpacks into its own folder, so two merged exports cannot
        # overwrite each other's identically named files. The folder name is
        # derived from a hash rather than the filename: chat exports are named
        # after the chat, which means emoji, trailing spaces (fatal on Windows,
        # which silently drops them from a directory name and then cannot find
        # the path again), and lengths that overrun MAX_PATH. Hashing is stable,
        # so re-uploading the same file reuses the same folder.
        digest = hashlib.sha1(source.name.encode("utf-8")).hexdigest()[:10]
        target = archive.media_dir / f"export-{digest}"
        report("unpack", f"Extracting {source.name}", {})
        extract_zip(source, target)

        extracted = [p for p in target.rglob("*") if p.is_file()]
        if not extracted:
            result.errors.append(
                f"Nothing could be extracted from {source.name}. The archive "
                "may be corrupt or still uploading."
            )
            return result
        report("unpack", f"{len(extracted)} files extracted", {})

        transcript = find_chat_txt(target)
        media_root = target
        if transcript is None:
            result.errors.append(
                f"No .txt transcript found among the {len(extracted)} extracted "
                "files. Is this a WhatsApp 'Export chat' archive?"
            )
            return result
    else:
        transcript = source
        if any(p.is_file() and p.suffix.lower() != ".txt"
               for p in source.parent.iterdir()):
            media_root = source.parent

    report("unpack", f"Transcript: {transcript.name}", {})

    # --- 2. parse -------------------------------------------------------
    parsed, parse_report = parse_file(str(transcript))
    result.stages["parse"] = parse_report.as_dict()
    result.warnings.extend(parse_report.warnings)
    report("parse", f"{parse_report.parsed_messages} messages",
           {"attachments": parse_report.attached_files})

    if not parsed:
        result.errors.append("Parsed zero messages; check the file format.")
        return result

    # System notices ("X joined using this group's invite link") are counted in
    # the parse report but never stored -- enrich() drops them so they cannot
    # inflate anyone's statistics. Dropping them here too keeps merge
    # accounting honest: otherwise they are absent from the archive, look new
    # on every merge, and each re-import reports phantom additions.
    parsed = [m for m in parsed if m.msg_type != "system"]
    if not parsed:
        result.errors.append(
            "This export contains only system notices, no actual messages."
        )
        return result

    # --- 3. combine -----------------------------------------------------
    if mode == "merge":
        existing, existing_keys = load_existing(conn)

        overlap = _participant_overlap(existing, parsed)
        if existing and overlap is not None and overlap < 0.2:
            # Almost certainly a different chat. Merging would blend two
            # people's conversations into one archive, and every per-person
            # statistic afterwards would be meaningless.
            result.errors.append(
                "This export looks like a different chat: it shares "
                f"{overlap:.0%} of its participants with "
                f"'{archive.name}'. Load it into a new archive instead, or use "
                "replace mode if you really mean to overwrite this one."
            )
            return result

        outcome = merge(existing, existing_keys, parsed)
        messages, dedup_keys = outcome.messages, outcome.keys
        result.stages["merge"] = outcome.as_dict()
        report("merge", f"{outcome.added} new, {outcome.skipped} already present",
               {"upgraded": outcome.upgraded})
        if outcome.added == 0:
            result.warnings.append(
                "Every message in this export was already in the archive. "
                "Nothing new was added."
            )
    else:
        messages, dedup_keys = parsed, assign_keys(parsed)
        result.stages["merge"] = {
            "total": len(messages), "added": len(messages),
            "skipped": 0, "upgraded": 0,
        }

    # A replace drops everything; a merge rebuilds from the combined set, so
    # both start from an empty set of derived tables.
    clear_archive_data(conn)

    # --- 4. resolve participants ----------------------------------------
    sender_counts = _count_senders(messages)
    participants, suggestions = resolve_participants(sender_counts)
    alias_lookup = build_alias_lookup(participants)
    result.stages["participants"] = {
        "count": len(participants),
        "people": [p.as_dict() for p in
                   sorted(participants.values(), key=lambda x: -x.message_count)],
        "merge_suggestions": suggestions,
    }
    result.warnings.extend(suggestions)
    report("participants", f"{len(participants)} people", {})

    # --- 5. enrich ------------------------------------------------------
    enriched = enrich(messages, alias_lookup, gap_hours=settings.session_gap_hours)
    key_by_identity = _key_lookup(messages, dedup_keys)
    attachment_by_msg, key_by_msg = _link_rows(messages, dedup_keys, enriched)
    report("enrich", f"{enriched[-1].session_id if enriched else 0} sessions", {})

    # --- 6. load --------------------------------------------------------
    bulk_insert(
        conn, "participants",
        ["participant_id", "display_name", "aliases", "is_phone_only",
         "message_count"],
        [(p.participant_id, p.display_name, sorted(p.aliases),
          p.is_phone_only, p.message_count) for p in participants.values()],
    )
    bulk_insert(
        conn, "messages",
        ["msg_id", "dedup_key", "ts", "participant_id", "sender_raw", "text",
         "msg_type", "attachment", "source_file", "char_count", "word_count",
         "emoji_count", "has_url", "is_question", "date", "hour", "weekday",
         "year_month", "session_id", "is_session_start", "gap_seconds",
         "prev_participant_id", "reply_gap_seconds"],
        [
            (e.msg_id, key_by_msg.get(e.msg_id), e.ts, e.participant_id,
             e.sender_raw, e.text, e.msg_type, attachment_by_msg.get(e.msg_id),
             source.name,
             e.char_count, e.word_count, e.emoji_count, e.has_url, e.is_question,
             e.ts.date(), e.hour, e.weekday, e.year_month,
             e.session_id, e.is_session_start, e.gap_seconds,
             e.prev_participant_id, e.reply_gap_seconds)
            for e in enriched
        ],
    )
    result.stages["load"] = {"messages": len(enriched)}
    report("load", f"{len(enriched)} rows written", {})

    _reconcile(conn, messages, enriched, attachment_by_msg, result)
    del key_by_identity

    # --- 7. media -------------------------------------------------------
    do_media = settings.describe_media if describe_media is None else describe_media
    media_stage = _load_media(conn, archive, attachment_by_msg, transcript, report)
    result.stages["media"] = media_stage
    result.warnings.extend(media_stage.get("warnings", []))

    files_to_process = media_stage.pop("_files", [])
    if do_media and files_to_process:
        if settings.has_api_key:
            from .media_understanding import process_media

            report("media", f"Describing {len(files_to_process)} attachments", {})
            state = process_media(
                files_to_process, conn,
                on_progress=lambda p: report("media", "describing", p.as_dict()),
            )
            result.stages["media"]["understanding"] = state.as_dict()
            if state.quota_reached:
                result.warnings.append(
                    f"{state.quota_message} "
                    f"{state.done} of {state.total} attachments were described; "
                    "the rest stay searchable by filename and caption until then."
                )
        else:
            result.warnings.append(
                "No Gemini API key configured: attachments were catalogued but "
                "not described, so their contents are not searchable."
            )

    # --- 8. chunk -------------------------------------------------------
    chunks = build_chunks(conn)
    store_chunks(conn, chunks)
    result.stages["chunks"] = {"count": len(chunks)}
    report("chunk", f"{len(chunks)} retrieval windows", {})

    # --- 9. embed -------------------------------------------------------
    if embed and settings.has_api_key and chunks:
        from .embed import embed_chunks

        stats = embed_chunks(conn, on_progress=lambda done, total: report(
            "embed", "embedding", {"done": done, "total": total}))
        result.stages["embeddings"] = stats
        report("embed", f"{stats['embedded']} new, {stats['cached']} cached", {})
        if stats.get("quota_reached"):
            result.warnings.append(
                f"{stats['quota_message']} "
                "Semantic search covers the chunks embedded so far; keyword "
                "search and every statistic are unaffected."
            )
    elif embed and not settings.has_api_key:
        result.warnings.append(
            "No Gemini API key configured: no embeddings built, so semantic "
            "search is unavailable. Keyword search and all statistics work."
        )

    # --- 10. index ------------------------------------------------------
    result.stages["fts"] = {"ok": rebuild_fts(conn)}

    overview = compute_overview(conn)
    set_meta("overview", json.dumps(overview, default=str), conn)
    set_meta("ingested_at", datetime.now().isoformat(), conn)
    result.stages["overview"] = overview

    archive.sources.append(IngestRecord(
        filename=source.name,
        ingested_at=datetime.now().isoformat(timespec="seconds"),
        mode=mode,
        messages_added=result.stages["merge"]["added"],
        messages_skipped=result.stages["merge"]["skipped"],
        media_added=media_stage.get("media_rows", 0),
    ))
    archive.stats = {
        "messages": overview.get("total_messages", 0),
        "participants": overview.get("participant_count", 0),
        "media": sum(m["count"] for m in overview.get("media", [])),
        "first_message": overview.get("first_message"),
        "last_message": overview.get("last_message"),
    }
    archive.save()

    result.ok = True
    result.elapsed_seconds = time.time() - started
    report("done", f"Ingest finished in {result.elapsed_seconds:.1f}s", {})
    return result


def _reconcile(conn, messages, enriched, attachment_by_msg, result) -> None:
    """
    Verify nothing was lost between parsing and storage.

    Tests only cover the variations someone thought to write down. This runs on
    every real import and checks the arithmetic directly, so a format nobody
    anticipated produces a visible complaint rather than an archive that is
    quietly missing 8% of its messages and looks completely normal.
    """
    checks: list[str] = []

    stored = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    if stored != len(enriched):
        checks.append(
            f"{len(enriched)} messages were prepared but {stored} reached the "
            "database."
        )
    if len(enriched) != len(messages):
        checks.append(
            f"{len(messages)} messages went into enrichment but {len(enriched)} "
            "came out."
        )

    # Every attachment named in the transcript must be attached to a row.
    named = sum(1 for m in messages if m.attachment)
    linked = conn.execute(
        "SELECT COUNT(*) FROM messages WHERE attachment IS NOT NULL"
    ).fetchone()[0]
    if linked != named:
        checks.append(
            f"{named} attachments were named in the transcript but {linked} "
            "were linked to a message."
        )
    if len(attachment_by_msg) != named:
        checks.append(
            f"{named} attachments were named but {len(attachment_by_msg)} could "
            "be matched back to their message."
        )

    # Nobody may be silently dropped: every non-system message needs a sender.
    orphaned = conn.execute(
        "SELECT COUNT(*) FROM messages WHERE participant_id IS NULL "
        "AND msg_type <> 'system'"
    ).fetchone()[0]
    if orphaned:
        checks.append(f"{orphaned} stored messages have no identified sender.")

    result.stages["reconciliation"] = {
        "ok": not checks,
        "stored": stored,
        "prepared": len(enriched),
        "attachments_named": named,
        "attachments_linked": linked,
        "problems": checks,
    }

    for problem in checks:
        result.warnings.append(
            f"Consistency check: {problem} This is a bug -- the numbers in this "
            "archive may be wrong. Please report it with the export's format."
        )


def _participant_overlap(existing, incoming) -> float | None:
    """
    Fraction of the incoming export's people who already appear in the archive.

    Two exports of the same chat share nearly everyone. Two different chats
    usually share nobody. Returns None when either side has no identifiable
    senders, in which case we have no evidence and do not block.
    """
    from ..parse.normalize import canonical_key

    old = {canonical_key(m.sender) for m in existing if m.sender}
    new = {canonical_key(m.sender) for m in incoming if m.sender}
    if not old or not new:
        return None
    return len(old & new) / len(new)


def _count_senders(messages) -> "Counter":
    from collections import Counter

    counts: Counter = Counter()
    for m in messages:
        if m.sender and m.msg_type != "system":
            counts[m.sender] += 1
    return counts


def _key_lookup(messages, keys) -> dict:
    return {id(m): k for m, k in zip(messages, keys)}


def _link_rows(messages, dedup_keys, enriched):
    """
    Attach filenames and dedup keys to the enriched rows.

    enrich() filters system notices and re-sorts, so positions shift. Both
    lists are in timestamp order, so walking them together and matching on
    (timestamp, sender) reattaches each row to the message it came from.
    """
    pending: dict[tuple, list[tuple[str | None, str]]] = {}
    for message, key in zip(messages, dedup_keys):
        if message.msg_type == "system":
            continue
        pending.setdefault((message.ts, message.sender), []).append(
            (message.attachment, key)
        )

    attachments: dict[int, str] = {}
    keys_by_msg: dict[int, str] = {}
    for e in enriched:
        bucket = pending.get((e.ts, e.sender_raw))
        if not bucket:
            continue
        attachment, key = bucket.pop(0)
        keys_by_msg[e.msg_id] = key
        if attachment:
            attachments[e.msg_id] = attachment

    return attachments, keys_by_msg


def _load_media(conn, archive: Archive, attachment_by_msg: dict[int, str],
                transcript: Path, report: ProgressFn) -> dict:
    """
    Catalogue attachment files and create one media row per reference.

    Scans the archive's whole media directory, not just this export's folder,
    so a merge can match messages from a text-only export against files that a
    previous with-media export already supplied.
    """
    referenced = set(attachment_by_msg.values())

    if not archive.media_dir.exists() or not any(archive.media_dir.rglob("*")):
        if referenced:
            return {
                "files": 0, "media_rows": 0,
                "warnings": [
                    f"{len(referenced)} messages reference attachments but no "
                    "media files are present. Export the chat again with "
                    "'Attach media' and merge it in to add photos, voice notes "
                    "and videos."
                ],
            }
        return {"files": 0, "media_rows": 0, "warnings": []}

    report("media", "Cataloguing attachments", {})
    catalog = MediaCatalog(archive.media_dir, transcript=transcript)
    media_report = catalog.finalize(referenced)

    rows, files_to_process = [], []
    media_id = 0
    seen_hashes: set[str] = set()

    for msg_id, filename in attachment_by_msg.items():
        mf = catalog.lookup(filename)
        if mf is None:
            continue
        rows.append((
            media_id, msg_id, mf.filename, str(mf.path), mf.kind, mf.ext,
            mf.size_bytes, mf.content_hash, mf.readable_by_gemini,
            None, None, None, None,
            "pending" if mf.readable_by_gemini else "skipped",
            None if mf.readable_by_gemini else f"unsupported format {mf.ext}",
            None, None,
        ))
        media_id += 1
        if mf.readable_by_gemini and mf.content_hash not in seen_hashes:
            seen_hashes.add(mf.content_hash)
            files_to_process.append(mf)

    if rows:
        bulk_insert(
            conn, "media",
            ["media_id", "msg_id", "filename", "path", "kind", "ext",
             "size_bytes", "content_hash", "readable", "description",
             "transcript", "ocr_text", "detected_objects", "status", "error",
             "model_used", "processed_at"],
            rows,
        )

    stage = media_report.as_dict()
    stage["files"] = len(files_to_process)
    stage["media_rows"] = len(rows)
    stage["_files"] = files_to_process
    return stage


def compute_overview(conn) -> dict:
    """
    Cached headline statistics.

    Injected into the agent's system prompt on every turn so it always knows
    the shape of the archive without spending a tool call to find out.
    """
    row = conn.execute("""
        SELECT COUNT(*), MIN(ts), MAX(ts),
               COUNT(DISTINCT participant_id), COUNT(DISTINCT date),
               COUNT(DISTINCT session_id)
        FROM messages WHERE msg_type <> 'system'
    """).fetchone()

    people = conn.execute("""
        SELECT COALESCE(p.display_name, m.sender_raw) AS sender,
               COUNT(*) AS messages,
               SUM(CASE WHEN m.is_session_start THEN 1 ELSE 0 END) AS initiations,
               SUM(CASE WHEN m.msg_type = 'media' THEN 1 ELSE 0 END) AS media,
               ROUND(AVG(m.word_count), 1) AS avg_words
        FROM messages m
        LEFT JOIN participants p USING (participant_id)
        WHERE m.msg_type <> 'system'
        GROUP BY 1 ORDER BY messages DESC
    """).fetchall()

    media_kinds = conn.execute(
        "SELECT kind, COUNT(*), SUM(CASE WHEN status='done' THEN 1 ELSE 0 END) "
        "FROM media GROUP BY 1 ORDER BY 2 DESC"
    ).fetchall()

    return {
        "total_messages": row[0],
        "first_message": str(row[1]) if row[1] else None,
        "last_message": str(row[2]) if row[2] else None,
        "participant_count": row[3],
        "active_days": row[4],
        "session_count": row[5],
        "session_gap_hours": settings.session_gap_hours,
        "participants": [
            {"name": p[0], "messages": p[1], "initiations": p[2],
             "media_sent": p[3], "avg_words": float(p[4] or 0)}
            for p in people
        ],
        "media": [
            {"kind": k, "count": c, "described": d} for k, c, d in media_kinds
        ],
    }
