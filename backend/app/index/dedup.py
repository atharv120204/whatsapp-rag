"""
Message identity across overlapping exports.

Every WhatsApp export contains the whole history up to the moment it was taken,
so two exports of the same chat overlap almost entirely. Merging them naively
doubles the archive and every statistic in it. This module decides when two
rows from different files are the same message.

The key is (timestamp, sender, content) plus an occurrence counter, because
sending "ok" twice inside the same minute is a real thing people do and the two
are genuinely different messages. Matching them positionally, in order, keeps
both on the first import and matches both on the second.

The case that needs care
------------------------
The same photo looks completely different in the two export modes:

    with media      IMG-20230812-WA0001.jpg (file attached)   + caption
    without media   <Media omitted>

Hashing the text would give them different keys, so combining a with-media
export (which WhatsApp truncates to roughly the last 10,000 messages) with a
full-history text-only export -- the main reason to merge at all -- would
duplicate every single photo, voice note and video. So attachments are keyed on
being *an attachment at that moment from that person*, not on their filename or
placeholder text.

When both versions of a message are present we keep the richer one: a real
filename and caption beat "<Media omitted>".
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from datetime import datetime

from ..parse.whatsapp import RawMessage


def _text_fingerprint(text: str) -> str:
    normalized = " ".join((text or "").split()).casefold()
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]


def base_key(ts: datetime, sender: str | None, text: str,
             msg_type: str, attachment: str | None) -> str:
    """
    Identity of a message, ignoring which export it came from.

    Media is keyed without its content because the two export modes describe
    the same attachment in irreconcilably different ways.
    """
    who = (sender or "~system~").strip().casefold()
    stamp = ts.strftime("%Y-%m-%dT%H:%M")     # exports are minute-precision

    if msg_type == "media" or attachment:
        return f"{stamp}|{who}|media"
    if msg_type == "deleted":
        return f"{stamp}|{who}|deleted"
    return f"{stamp}|{who}|{_text_fingerprint(text)}"


def assign_keys(messages: list[RawMessage]) -> list[str]:
    """
    Give every message a key unique within its own export.

    The occurrence suffix distinguishes genuine repeats. Because both exports
    list them in the same order, the Nth "ok" in one file matches the Nth "ok"
    in the other.
    """
    seen: Counter[str] = Counter()
    keys: list[str] = []
    for m in messages:
        base = base_key(m.ts, m.sender, m.text, m.msg_type, m.attachment)
        keys.append(f"{base}#{seen[base]}")
        seen[base] += 1
    return keys


def _richness(m: RawMessage) -> tuple:
    """
    How much a version of a message tells us. Higher wins on a collision.

    An attachment filename is worth more than any amount of placeholder text,
    because it is the difference between having the file and not.
    """
    return (
        1 if m.attachment else 0,
        1 if m.msg_type != "media" or m.text else 0,
        len(m.text or ""),
    )


@dataclass
class MergeOutcome:
    messages: list[RawMessage]
    keys: list[str]
    added: int = 0
    skipped: int = 0
    upgraded: int = 0

    def as_dict(self) -> dict:
        return {
            "total": len(self.messages),
            "added": self.added,
            "skipped": self.skipped,
            "upgraded": self.upgraded,
        }


def merge(existing: list[RawMessage], existing_keys: list[str],
          incoming: list[RawMessage]) -> MergeOutcome:
    """
    Combine an archive's current messages with a newly parsed export.

    Returns the full deduplicated set in timestamp order, ready to re-enrich.
    Session ids, reply gaps and chunk boundaries all depend on the complete
    ordered stream, so a merge necessarily rebuilds everything downstream --
    the caches are what stop that costing anything.
    """
    combined: dict[str, RawMessage] = dict(zip(existing_keys, existing))
    incoming_keys = assign_keys(incoming)

    outcome = MergeOutcome(messages=[], keys=[])

    for key, message in zip(incoming_keys, incoming):
        current = combined.get(key)
        if current is None:
            combined[key] = message
            outcome.added += 1
        elif _richness(message) > _richness(current):
            # Same message, better version -- the with-media export naming the
            # file that the text-only export could only call "<Media omitted>".
            combined[key] = message
            outcome.upgraded += 1
            outcome.skipped += 1
        else:
            outcome.skipped += 1

    ordered = sorted(combined.items(), key=lambda kv: (kv[1].ts, kv[0]))
    outcome.keys = [key for key, _ in ordered]
    outcome.messages = [message for _, message in ordered]
    return outcome


def load_existing(conn) -> tuple[list[RawMessage], list[str]]:
    """
    Rebuild RawMessages from what is already stored.

    A merge re-derives every computed column from scratch, so the stored rows
    only need to carry the raw facts: when, who, what, and which file.
    """
    try:
        rows = conn.execute("""
            SELECT dedup_key, ts, sender_raw, text, msg_type, attachment
            FROM messages ORDER BY ts, msg_id
        """).fetchall()
    except Exception:  # noqa: BLE001 - table absent on a fresh archive
        return [], []

    messages, keys = [], []
    for i, (key, ts, sender, text, msg_type, attachment) in enumerate(rows):
        messages.append(RawMessage(
            ts=ts,
            sender=sender,
            text=text or "",
            msg_type=msg_type,
            line_no=i,
            attachment=attachment,
        ))
        keys.append(key)

    # Rows written before dedup keys existed have none. Recompute the whole
    # set rather than defaulting each to occurrence 0, which would collapse
    # genuine repeats ("ok" sent twice in a minute) into a single message.
    if any(k is None for k in keys):
        keys = assign_keys(messages)

    return messages, keys
