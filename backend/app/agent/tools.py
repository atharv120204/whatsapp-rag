"""
The tools the agent chooses between.

The split is the whole point of the design:

  run_sql          exact aggregates -- counts, rankings, time patterns
  search_chat      meaning -- "what did we decide about the hotel"
  get_context      the messages around a hit, for quoting accurately
  find_media       photos, voice notes and videos by what is in them

"How many messages did Rohit send" must never be answered by retrieval. It is
a COUNT, and retrieval of twelve chunks out of ten thousand cannot produce one.
Conversely "why was everyone annoyed in March" is not a SQL question. Routing
between them is what makes the archive genuinely queryable.

Every tool is bound to one archive's connection by `build_tools`. Nothing here
reaches for a global handle: with several archives on the device, a tool that
resolved its own connection could answer a question about the wrong chat, and
that failure would be silent.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

from ..config import settings
from ..db import get_meta
from .sql_guard import validate

# --- schema documentation shown to the model ---------------------------------

SCHEMA_DOC = """
v_messages -- one row per message (use this, not `messages`: it has the name)
  msg_id, ts, date, hour, weekday (0=Mon), year_month, sender, participant_id,
  text, msg_type ('text'|'media'|'deleted'), attachment, source_file,
  char_count, word_count, emoji_count, has_url, is_question,
  session_id, is_session_start, gap_seconds, prev_participant_id,
  reply_gap_seconds

  is_session_start   first message after >{gap}h silence -> count per sender
                     to answer "who starts conversations"
  reply_gap_seconds  wait before this message when someone else spoke last;
                     NULL otherwise, so AVG() is already response time

v_searchable -- msg_id, ts, date, hour, weekday, sender, msg_type, content
  `content` folds in media descriptions, transcripts and OCR. Match text here
  so photos and voice notes are not invisible. An undescribed attachment reads
  as "[image sent, not yet described]".

v_media -- media_id, msg_id, filename, kind ('image'|'video'|'voice'|'audio'|
  'sticker'|'document'|'contact'), size_bytes, description, transcript,
  ocr_text, status, ts, date, sender, caption

participants -- participant_id, display_name, aliases, is_phone_only,
  message_count

sessions_at(gap_hours) -- re-segment conversations at another threshold:
  SELECT participant_id, COUNT(*) FROM sessions_at(8.0) WHERE is_start GROUP BY 1

NOTES
- System notices are not stored; every row is a real message.
- A media message's `text` is only its caption, usually empty. Never filter
  `text <> ''` to find a day's activity -- most of a busy day is attachments.
- DuckDB: date_trunc, list_contains, ILIKE, strftime(ts,'%Y-%m').
"""


_DATE_ONLY = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def normalise_range(after: str | None, before: str | None) -> tuple[str | None, str | None]:
    """
    Turn user-supplied bounds into an inclusive timestamp range.

    A bare date means the whole day. Casting "2026-02-09" straight to a
    timestamp gives midnight, so asking for after=2026-02-09 and
    before=2026-02-09 -- the obvious way to say "that day" -- matched only
    messages sent exactly at 00:00:00 and returned nothing.
    """
    if after and _DATE_ONLY.match(after.strip()):
        after = f"{after.strip()} 00:00:00"
    if before and _DATE_ONLY.match(before.strip()):
        before = f"{before.strip()} 23:59:59"
    return after, before


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)


def _reciprocal_rank_fusion(result_lists: list[list[dict]], k: int,
                            damping: int = 60) -> list[dict]:
    """
    Merge ranked lists by reciprocal rank.

    RRF combines rankings without needing the scores to be comparable, which
    matters because cosine similarity and BM25 are on entirely different
    scales and normalising them against each other is guesswork.
    """
    scores: dict[str, float] = {}
    payload: dict[str, dict] = {}

    for results in result_lists:
        for rank, item in enumerate(results):
            key = str(item.get("chunk_id", f"m{item.get('msg_id')}"))
            scores[key] = scores.get(key, 0.0) + 1.0 / (damping + rank + 1)
            if key not in payload or item.get("source") == "semantic":
                payload[key] = item

    ordered = sorted(scores.items(), key=lambda kv: -kv[1])[:k]
    out = []
    for key, score in ordered:
        item = dict(payload[key])
        item["fusion_score"] = round(score, 5)
        out.append(item)
    return out


def build_tools(conn) -> dict[str, Callable]:
    """Bind every tool to one archive's connection."""

    def run_sql(query: str) -> dict[str, Any]:
        guard = validate(query, max_rows=settings.max_sql_rows)
        if not guard.ok:
            return {"error": guard.reason,
                    "hint": "Rewrite as a single read-only SELECT."}
        try:
            cur = conn.execute(guard.sql)
            columns = [d[0] for d in cur.description]
            rows = cur.fetchall()
        except Exception as exc:  # noqa: BLE001 - surfaced so the model retries
            return {"error": f"SQL failed: {exc}",
                    "hint": "Check column names against the schema and retry."}

        return {
            "columns": columns,
            "rows": [[_jsonable(v) for v in row] for row in rows],
            "row_count": len(rows),
            "truncated": len(rows) >= settings.max_sql_rows,
            "sql": guard.sql,
        }

    def _keyword_search(query: str, k: int, participant: str | None,
                        after: str | None, before: str | None) -> list[dict]:
        filters, params = [], []
        if participant:
            filters.append("s.sender ILIKE ?")
            params.append(f"%{participant}%")
        if after:
            filters.append("s.ts >= ?::TIMESTAMP")
            params.append(after)
        if before:
            filters.append("s.ts <= ?::TIMESTAMP")
            params.append(before)
        where = (" AND " + " AND ".join(filters)) if filters else ""

        try:
            rows = conn.execute(f"""
                WITH scored AS (
                    SELECT msg_id, fts_main_fts_docs.match_bm25(msg_id, ?) AS score
                    FROM fts_docs
                )
                SELECT s.msg_id, s.ts, s.sender, s.content, scored.score
                FROM scored
                JOIN v_searchable s USING (msg_id)
                WHERE scored.score IS NOT NULL {where}
                ORDER BY scored.score DESC
                LIMIT ?
            """, [query, *params, k]).fetchall()
        except Exception:  # noqa: BLE001 - no FTS index built
            rows = conn.execute(f"""
                SELECT s.msg_id, s.ts, s.sender, s.content, 1.0 AS score
                FROM v_searchable s
                WHERE s.content ILIKE ? {where}
                ORDER BY s.ts DESC
                LIMIT ?
            """, [f"%{query}%", *params, k]).fetchall()

        return [
            {
                "msg_id": r[0],
                "start_ts": str(r[1]),
                "end_ts": str(r[1]),
                "participants": [r[2]] if r[2] else [],
                "text": f"{r[1]:%d %b %Y %H:%M} {r[2]}: {r[3]}"
                        if hasattr(r[1], "strftime") else f"{r[2]}: {r[3]}",
                "score": round(float(r[4] or 0), 4),
                "source": "keyword",
            }
            for r in rows
        ]

    def search_chat(query: str | None = None, k: int | None = None,
                    participant: str | None = None,
                    after: str | None = None,
                    before: str | None = None) -> dict[str, Any]:
        """
        Hybrid retrieval: semantic vectors fused with BM25 keyword matching.

        Neither alone is enough. Vectors miss exact strings -- a phone number, a
        booking reference, an unusual name. BM25 misses paraphrase, which is
        most of how people refer to past conversations.
        """
        k = k or settings.top_k
        after, before = normalise_range(after, before)

        # "What were we talking about on the 9th" has no search terms -- the
        # date is the whole question. Rather than returning nothing, read the
        # window out in order, which is what was actually being asked for.
        if not (query or "").strip():
            if not (after or before or participant):
                return {
                    "results": [], "result_count": 0,
                    "error": "Give either something to search for, or a date "
                             "range / participant to read through.",
                }
            return _read_window(after, before, participant, k)

        semantic: list[dict] = []
        if settings.has_api_key:
            try:
                from ..index.embed import vector_search
                semantic = vector_search(conn, query, k=k, participant=participant,
                                         after=after, before=before)
            except Exception as exc:  # noqa: BLE001 - keyword search still works
                print(f"[tools] vector search unavailable: {exc}")

        keyword = _keyword_search(query, k, participant, after, before)
        fused = _reciprocal_rank_fusion([semantic, keyword], k=k)

        if not fused and (after or before):
            # Nothing matched the words, but the window itself was the point.
            fallback = _read_window(after, before, participant, k)
            fallback["note"] = (
                f"No message matched {query!r}, so here is the conversation in "
                "that window instead."
            )
            return fallback

        return {
            "results": fused,
            "result_count": len(fused),
            "semantic_available": bool(semantic),
            "note": None if semantic else
                    "Semantic search is unavailable (no API key or no "
                    "embeddings); these are keyword matches only.",
        }

    def _read_window(after: str | None, before: str | None,
                     participant: str | None, k: int) -> dict[str, Any]:
        """
        Render a window of conversation as a compact transcript.

        Returns one string rather than a list of rows, for a reason worth
        stating: a busy day in a group chat is mostly attachments. Of 108
        messages on the archive's busiest day, 68 were images. Returned as rows
        and capped at the first N, the model saw a wall of
        "[image sent, not yet described]" and never reached the actual
        conversation, so it reported that no content was available -- while the
        day contained an argument about a shared bill and a birthday.

        So runs of consecutive attachments from one person collapse to a single
        line, and every word anyone actually typed is kept. That is far more
        signal for the same number of tokens.
        """
        filters, params = ["m.msg_type <> 'system'"], []
        if after:
            filters.append("s.ts >= ?::TIMESTAMP")
            params.append(after)
        if before:
            filters.append("s.ts <= ?::TIMESTAMP")
            params.append(before)
        if participant:
            filters.append("s.sender ILIKE ?")
            params.append(f"%{participant}%")
        where = " AND ".join(filters)

        rows = conn.execute(f"""
            SELECT s.ts, s.sender, s.content, s.msg_type
            FROM v_searchable s
            JOIN messages m USING (msg_id)
            WHERE {where}
            ORDER BY s.ts
        """, params).fetchall()

        if not rows:
            return {"transcript": "", "message_count": 0,
                    "note": "No messages in that window."}

        lines: list[str] = []
        spoken = attachments = 0
        pending_sender: str | None = None
        pending_count = 0
        pending_time = ""

        def flush() -> None:
            nonlocal pending_sender, pending_count
            if pending_sender and pending_count:
                what = "attachment" if pending_count == 1 else "attachments"
                lines.append(f"{pending_time} {pending_sender}: "
                             f"({pending_count} {what})")
            pending_sender, pending_count = None, 0

        empty = 0
        for ts, sender, content, msg_type in rows:
            text = (content or "").strip()
            stamp = ts.strftime("%H:%M") if hasattr(ts, "strftime") else str(ts)

            # Messages with no content at all say nothing and, worse, split a
            # run of attachments in two, so one person's burst of photos reads
            # as several separate bursts.
            if not text:
                empty += 1
                continue

            is_placeholder = text.startswith("[") and text.endswith("]")

            if is_placeholder:
                attachments += 1
                if sender == pending_sender:
                    pending_count += 1
                else:
                    flush()
                    pending_sender, pending_count, pending_time = sender, 1, stamp
                continue

            flush()
            spoken += 1
            lines.append(f"{stamp} {sender}: {text}")

        flush()

        header = ""
        if hasattr(rows[0][0], "strftime"):
            first, last = rows[0][0], rows[-1][0]
            header = (f"[{first:%A, %d %B %Y}]" if first.date() == last.date()
                      else f"[{first:%d %b %Y} to {last:%d %b %Y}]")

        transcript = "\n".join(([header] if header else []) + lines)

        # Keep the whole exchange if it fits; otherwise favour the end, which
        # is where a conversation usually resolves.
        limit = settings.tool_result_max_chars - 500
        truncated = False
        if len(transcript) > limit:
            transcript = ("[earlier messages omitted]\n"
                          + transcript[-limit:])
            truncated = True

        return {
            "transcript": transcript,
            "message_count": len(rows),
            "messages_with_text": spoken,
            "attachments": attachments,
            "empty_messages": empty,
            "truncated": truncated,
            "note": (f"{len(rows)} messages in this window: {spoken} with text, "
                     f"{attachments} attachments (runs of attachments are "
                     f"collapsed).") if attachments else None,
        }

    def get_context(msg_id: int, before: int = 10, after: int = 10) -> dict[str, Any]:
        rows = conn.execute("""
            SELECT msg_id, ts, sender, msg_type, content
            FROM v_searchable
            WHERE msg_id BETWEEN ? AND ?
            ORDER BY msg_id
        """, [msg_id - before, msg_id + after]).fetchall()

        return {
            "center_msg_id": msg_id,
            "messages": [
                {"msg_id": r[0], "ts": str(r[1]), "sender": r[2],
                 "type": r[3], "text": r[4]}
                for r in rows
            ],
        }

    def find_media(query: str | None = None, kind: str | None = None,
                   sender: str | None = None, limit: int = 20) -> dict[str, Any]:
        filters, params = ["1=1"], []
        if query:
            filters.append(
                "(description ILIKE ? OR transcript ILIKE ? OR ocr_text ILIKE ? "
                "OR caption ILIKE ? OR filename ILIKE ?)"
            )
            params.extend([f"%{query}%"] * 5)
        if kind:
            filters.append("kind = ?")
            params.append(kind)
        if sender:
            filters.append("sender ILIKE ?")
            params.append(f"%{sender}%")

        params.append(min(limit, 100))
        rows = conn.execute(f"""
            SELECT media_id, msg_id, filename, kind, ts, sender, caption,
                   description, transcript, ocr_text, status
            FROM v_media
            WHERE {' AND '.join(filters)}
            ORDER BY ts
            LIMIT ?
        """, params).fetchall()

        return {
            "results": [
                {"media_id": r[0], "msg_id": r[1], "filename": r[2], "kind": r[3],
                 "ts": str(r[4]), "sender": r[5], "caption": r[6],
                 "description": r[7], "transcript": r[8], "ocr_text": r[9],
                 "status": r[10]}
                for r in rows
            ],
            "result_count": len(rows),
        }

    def find_moments(kind: str = "funny", limit: int = 5) -> dict[str, Any]:
        from ..api.insights import find_moments as _find

        return _find(conn, kind=kind, limit=limit)

    def get_overview() -> dict[str, Any]:
        raw = get_meta("overview", conn)
        if not raw:
            return {"error": "No data ingested into this archive yet."}
        return json.loads(raw)

    return {
        "run_sql": run_sql,
        "search_chat": search_chat,
        "get_context": get_context,
        "find_media": find_media,
        "find_moments": find_moments,
        "get_overview": get_overview,
    }


# --- declarations handed to Gemini -------------------------------------------

TOOL_DECLARATIONS = [
    {
        "name": "run_sql",
        "description": (
            "Read-only SQL. Use for anything countable: totals, rankings, "
            "averages, per-person stats, time patterns, response times, "
            "who starts conversations. Never count with search."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A single DuckDB SELECT statement.",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_chat",
        "description": (
            "Find conversations by meaning or wording, or read a date range "
            "in order (leave query empty and set after/before). Returns "
            "message windows including text from photos and voice notes. "
            "Not for counting."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to look for. Leave empty to read a whole date range or one person's messages in order."},
                "k": {"type": "integer", "description": "How many windows (default 12)."},
                "participant": {"type": "string", "description": "Restrict to one person."},
                "after": {"type": "string", "description": "ISO date lower bound (inclusive; a bare date means from 00:00)."},
                "before": {"type": "string", "description": "ISO date upper bound (inclusive; a bare date covers the whole day)."},
            },
        },
    },
    {
        "name": "get_context",
        "description": (
            "Messages around a msg_id, to quote accurately or resolve who "
            "'he' or 'that' means."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "msg_id": {"type": "integer"},
                "before": {"type": "integer", "description": "Default 10."},
                "after": {"type": "integer", "description": "Default 10."},
            },
            "required": ["msg_id"],
        },
    },
    {
        "name": "find_media",
        "description": (
            "Search attachments by what is in them: what a photo shows, what "
            "a voice note says, text inside an image."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to look for inside the media."},
                "kind": {
                    "type": "string",
                    "description": "image, video, voice, audio, sticker, document or contact.",
                },
                "sender": {"type": "string"},
                "limit": {"type": "integer", "description": "Default 20."},
            },
        },
    },
    {
        "name": "find_moments",
        "description": (
            "Rank every conversation in the archive and return the notable "
            "ones with excerpts. kind: funny | argument (friction, "
            "venting) | deep | late_night | busiest. Use for 'funniest "
            "moments', 'did we ever fight', 'deepest conversations'. "
            "Quote from the excerpts."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "kind": {
                    "type": "string",
                    "description": "funny, argument, deep, late_night or busiest.",
                },
                "limit": {"type": "integer", "description": "Default 5, max 10."},
            },
        },
    },
    {
        "name": "get_overview",
        "description": "Headline stats: people, totals, date range, media counts.",
        "parameters": {"type": "object", "properties": {}},
    },
]
