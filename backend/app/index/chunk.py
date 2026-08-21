"""
Build the retrieval units.

Embedding individual messages does not work for chat. Half of them are "yeah",
"ok", "lol" -- strings with no standalone meaning, whose vectors are noise. The
meaning lives in the exchange, so the unit here is a *window of consecutive
messages* rendered as a small transcript, with speaker names and timestamps
kept in the text so the model can see who said what and when.

Windows never straddle a session boundary. Splicing the tail of Tuesday's
argument onto Wednesday's dinner plan produces a chunk that is about neither.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime

from ..config import settings
from ..db import bulk_insert


@dataclass
class Chunk:
    start_msg_id: int
    end_msg_id: int
    session_id: int
    start_ts: datetime
    end_ts: datetime
    participants: list[str]
    n_messages: int
    body: str


def _render(rows: list[tuple]) -> str:
    """
    Render rows as a readable transcript.

    Rows are (msg_id, ts, sender, msg_type, content). The date is stamped once
    per chunk and times per line, which keeps temporal questions answerable
    from the retrieved text alone without repeating the date 25 times.
    """
    if not rows:
        return ""

    lines = [f"[{rows[0][1]:%A, %d %B %Y}]"]
    for _, ts, sender, msg_type, content in rows:
        text = (content or "").strip()
        if not text:
            text = {
                "media": "(sent an attachment)",
                "deleted": "(deleted a message)",
                "system": "(system notice)",
            }.get(msg_type, "")
        if not text:
            continue
        lines.append(f"{ts:%H:%M} {sender}: {text}")
    return "\n".join(lines)


def build_chunks(conn) -> list[Chunk]:
    """
    Window the message stream into chunks.

    Reads from v_searchable, so whatever Gemini extracted from photos, voice
    notes and documents is embedded alongside the words people typed. Without
    that, asking about a shared photo retrieves nothing.
    """
    rows = conn.execute("""
        SELECT s.msg_id, s.ts, s.sender, s.msg_type, s.content, m.session_id
        FROM v_searchable s
        JOIN messages m USING (msg_id)
        WHERE m.msg_type <> 'system'
        ORDER BY s.ts, s.msg_id
    """).fetchall()

    size = max(2, settings.chunk_size)
    overlap = min(max(0, settings.chunk_overlap), size - 1)
    step = size - overlap

    # Group by session first so windows stay inside one conversation.
    sessions: dict[int, list[tuple]] = {}
    for msg_id, ts, sender, msg_type, content, session_id in rows:
        sessions.setdefault(session_id, []).append(
            (msg_id, ts, sender, msg_type, content)
        )

    chunks: list[Chunk] = []
    for session_id, msgs in sessions.items():
        for start in range(0, len(msgs), step):
            window = msgs[start:start + size]
            if not window:
                continue

            # Skip a trailing window already fully covered by the previous one.
            if start > 0 and len(window) <= overlap:
                break

            body = _render(window)
            if not body.strip():
                continue

            speakers = sorted({row[2] for row in window if row[2]})
            chunks.append(Chunk(
                start_msg_id=window[0][0],
                end_msg_id=window[-1][0],
                session_id=session_id,
                start_ts=window[0][1],
                end_ts=window[-1][1],
                participants=speakers,
                n_messages=len(window),
                body=body,
            ))

    return chunks


def store_chunks(conn, chunks: list[Chunk]) -> int:
    conn.execute("DELETE FROM chunk_vectors")
    conn.execute("DELETE FROM chunks")
    bulk_insert(
        conn, "chunks",
        ["chunk_id", "body_hash", "start_msg_id", "end_msg_id", "session_id",
         "start_ts", "end_ts", "participants", "n_messages", "body"],
        [
            (i, hashlib.sha256(c.body.encode("utf-8")).hexdigest(),
             c.start_msg_id, c.end_msg_id, c.session_id, c.start_ts,
             c.end_ts, c.participants, c.n_messages, c.body)
            for i, c in enumerate(chunks)
        ],
    )
    return len(chunks)
