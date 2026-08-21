"""
Conversation-level features derived at ingest time.

The questions people actually ask a chat archive -- "who starts conversations?",
"how fast does X reply?", "who gets ignored?" -- are not retrieval questions.
No amount of semantic search answers them. They are windowed aggregates over an
ordered message stream, so we compute the underlying columns once, here, and let
SQL answer exactly afterwards.

A *session* is a run of messages with no silence longer than `gap_hours`. The
first message of a session is an *initiation*. The threshold is a judgement
call, not a fact: at 1 hour you measure conversational bursts, at 8 hours you
measure who texts first each day. We store the raw gap alongside the default
labelling so either question can be answered later without re-ingesting.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .whatsapp import RawMessage, emoji_count, has_url

DEFAULT_GAP_HOURS = 4.0


@dataclass
class EnrichedMessage:
    """A message with everything the analytics layer needs, precomputed."""

    msg_id: int
    ts: datetime
    participant_id: str | None
    sender_raw: str | None
    text: str
    msg_type: str

    # Derived text features
    char_count: int
    word_count: int
    emoji_count: int
    has_url: bool
    is_question: bool

    # Derived temporal features
    date: str
    hour: int
    weekday: int          # 0 = Monday
    year_month: str

    # Derived conversation features
    session_id: int
    is_session_start: bool
    gap_seconds: float | None       # silence before this message
    prev_participant_id: str | None
    reply_gap_seconds: float | None  # gap only when the speaker changed


def _is_question(text: str) -> bool:
    stripped = text.strip()
    if "?" in stripped:
        return True
    opener = stripped.split(" ")[0].casefold() if stripped else ""
    return opener in {
        "who", "what", "when", "where", "why", "how", "which",
        "can", "could", "should", "would", "is", "are", "do", "does",
        "did", "will", "any", "anyone",
    }


def enrich(
    messages: list[RawMessage],
    alias_lookup: dict[str, str],
    gap_hours: float = DEFAULT_GAP_HOURS,
    include_system: bool = False,
) -> list[EnrichedMessage]:
    """
    Turn parsed messages into analysis-ready rows.

    System notices are excluded from session logic by default -- WhatsApp's
    "X joined using this group's invite link" is not someone starting a
    conversation, and counting it as one inflates whoever adds members most.
    """
    stream = [m for m in messages if include_system or m.msg_type != "system"]
    stream.sort(key=lambda m: m.ts)

    gap_seconds_threshold = gap_hours * 3600.0
    out: list[EnrichedMessage] = []

    session_id = 0
    prev_ts: datetime | None = None
    prev_pid: str | None = None

    for idx, m in enumerate(stream):
        pid = alias_lookup.get(m.sender) if m.sender else None

        gap = (m.ts - prev_ts).total_seconds() if prev_ts is not None else None
        is_start = gap is None or gap > gap_seconds_threshold
        if is_start:
            session_id += 1

        # A reply gap only means something when the other person spoke last.
        reply_gap = (
            gap if (gap is not None and not is_start and prev_pid != pid) else None
        )

        out.append(
            EnrichedMessage(
                msg_id=idx,
                ts=m.ts,
                participant_id=pid,
                sender_raw=m.sender,
                text=m.text,
                msg_type=m.msg_type,
                char_count=len(m.text),
                word_count=len(m.text.split()),
                emoji_count=emoji_count(m.text),
                has_url=has_url(m.text),
                is_question=_is_question(m.text) if m.msg_type == "text" else False,
                date=m.ts.strftime("%Y-%m-%d"),
                hour=m.ts.hour,
                weekday=m.ts.weekday(),
                year_month=m.ts.strftime("%Y-%m"),
                session_id=session_id,
                is_session_start=is_start,
                gap_seconds=gap,
                prev_participant_id=prev_pid,
                reply_gap_seconds=reply_gap,
            )
        )

        prev_ts = m.ts
        prev_pid = pid

    return out
