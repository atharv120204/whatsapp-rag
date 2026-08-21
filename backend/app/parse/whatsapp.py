"""
Parser for WhatsApp "Export chat" .txt files.

Handles the format variance you actually hit in the wild:

  Android 24h   12/08/2023, 21:14 - Karan: hey
  Android 12h   12/08/2023, 9:14 pm - Karan: hey
  iOS           [12/08/2023, 9:14:03 PM] Karan: hey
  ISO dates     2023-08-12, 21:14 - Karan: hey

plus the invisible characters WhatsApp injects (U+200E LRM, U+202F narrow
no-break space before AM/PM), multi-line message bodies, and system messages
that have no sender at all.

Day/month order is *inferred from the file*, not assumed. An export from an
Indian phone is d/m/Y; one from a US phone is m/d/Y; the same bytes parse to
different dates. Getting this wrong silently corrupts every temporal answer.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterator, Literal

MessageType = Literal["text", "media", "system", "deleted"]

# WhatsApp sprinkles these through exports; they break naive regexes.
_INVISIBLE = dict.fromkeys(map(ord, "‎‏⁦⁧⁨⁩﻿"))
_NARROW_SPACES = str.maketrans({" ": " ", " ": " "})

# A header is: [optional bracket] date, time [optional am/pm] [bracket close or " - "]
_HEADER = re.compile(
    r"""^
    \[?                                  # iOS wraps the stamp in brackets
    (?P<d1>\d{1,4})[/.\-](?P<d2>\d{1,2})[/.\-](?P<d3>\d{2,4})
    ,?\s+
    (?P<hh>\d{1,2}):(?P<mm>\d{2})(?::(?P<ss>\d{2}))?
    (?:\s*(?P<ampm>[APap]\.?[Mm]\.?))?
    \s*
    (?:\]\s*|\s-\s)                      # "] " (iOS) or " - " (Android)
    (?P<body>.*)$
    """,
    re.VERBOSE | re.MULTILINE,   # MULTILINE so scans over whole-file text work
)

# Sender is everything up to the first ": ". Guard against message text that
# merely contains a colon by capping the length and rejecting newlines.
_SENDER = re.compile(r"^(?P<sender>[^:\n]{1,60}?):\s(?P<text>.*)$", re.DOTALL)

# Substrings that mark a line as a WhatsApp system notice rather than a message.
_SYSTEM_PATTERNS = [
    r"messages and calls are end-to-end encrypted",
    r"messages to this (chat and call|group) are now secured",
    r"created (this )?group",
    r"created group",
    r"added [^\n]+$",
    r"\badded you\b",
    r"\bleft\b$",
    r"removed [^\n]+$",
    r"joined using this group.s invite link",
    r"joined from the community",
    r"changed the subject (from|to)",
    r"changed this group.s (icon|description|settings)",
    r"changed their phone number",
    r"changed to \+?\d",
    r"is now an admin",
    r"you.re now an admin",
    r"no longer an admin",
    r"turned on disappearing messages",
    r"turned off disappearing messages",
    r"pinned a message",
    r"unpinned a message",
    r"deleted this group",
    r"security code changed",
    r"missed (voice|video|group) call",
    r"blocked this contact",
    r"you were added",
    r"this group was upgraded",
    r"only admins can",
    r"waiting for this message",
    r"tap to learn more",
]
_SYSTEM_RE = re.compile("|".join(_SYSTEM_PATTERNS), re.IGNORECASE)

# Placeholder bodies in a *without-media* export: the file is gone, only a
# marker remains.
_MEDIA_PATTERNS = [
    r"^<media omitted>$",
    r"^(image|video|audio|sticker|document|gif|contact card|photo) omitted$",
    r"^(voice call|video call|missed voice call|missed video call)$",
    r"^live location shared$",
    r"^location: https?://",
    r"^null$",
]
_MEDIA_RE = re.compile("|".join(_MEDIA_PATTERNS), re.IGNORECASE)

# A *with-media* export names the file instead. Two dialects:
#   Android   IMG-20230812-WA0001.jpg (file attached)
#   iOS       <attached: 00000042-PHOTO-2023-08-12-21-15-10.jpg>
# Capturing the filename is what lets us look at the actual photo later.
_ATTACHMENT_RES = [
    re.compile(r"^<attached:\s*(?P<fn>[^>]+?)\s*>$", re.IGNORECASE),
    re.compile(r"^(?P<fn>\S[^\n]*?)\s*\((?:file attached|archivo adjunto)\)$",
               re.IGNORECASE),
]


def extract_attachment(text: str) -> tuple[str | None, str]:
    """
    Split a body into (attached filename, caption).

    A captioned photo puts the filename on the first line and the caption
    underneath, so this looks at line one only and returns the rest as text:

        IMG-20230812-WA0001.jpg (file attached)
        look at this view

    Returns (None, original_text) when nothing is attached.
    """
    stripped = text.strip()
    if not stripped:
        return None, text

    head, _, rest = stripped.partition("\n")
    for pattern in _ATTACHMENT_RES:
        m = pattern.match(head.strip())
        if m:
            return m.group("fn").strip(), rest.strip()
    return None, text

_DELETED_RE = re.compile(
    r"^(this message was deleted|you deleted this message)\.?$",
    re.IGNORECASE,
)

_URL_RE = re.compile(r"https?://\S+|www\.\S+")


@dataclass
class RawMessage:
    """One parsed line-group, before normalization or enrichment."""

    ts: datetime
    sender: str | None          # None for system notices
    text: str
    msg_type: MessageType
    line_no: int
    attachment: str | None = None   # filename, when the export includes media


@dataclass
class ParseReport:
    """What the parser saw. Surfaced in the UI so ingest failures are visible."""

    total_lines: int = 0
    parsed_messages: int = 0
    system_messages: int = 0
    media_messages: int = 0
    deleted_messages: int = 0
    attached_files: int = 0
    continuation_lines: int = 0
    unparsed_lines: int = 0
    date_order: str = "unknown"
    date_order_confidence: str = "unknown"
    senders: Counter = field(default_factory=Counter)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "total_lines": self.total_lines,
            "parsed_messages": self.parsed_messages,
            "system_messages": self.system_messages,
            "media_messages": self.media_messages,
            "deleted_messages": self.deleted_messages,
            "attached_files": self.attached_files,
            "continuation_lines": self.continuation_lines,
            "unparsed_lines": self.unparsed_lines,
            "date_order": self.date_order,
            "date_order_confidence": self.date_order_confidence,
            "distinct_senders": len(self.senders),
            "senders": dict(self.senders.most_common()),
            "warnings": self.warnings,
        }


def _clean(line: str) -> str:
    return line.translate(_INVISIBLE).translate(_NARROW_SPACES).rstrip("\r\n")


def detect_date_order(text: str) -> tuple[str, str]:
    """
    Infer whether the export uses D/M/Y or M/D/Y.

    Returns (order, confidence). A value >12 in the first position proves D/M/Y;
    >12 in the second proves M/D/Y. If neither appears (a chat spanning only the
    first twelve days of months), we fall back to D/M/Y -- WhatsApp's default
    outside the US -- and say so, rather than pretending we know.
    """
    first_gt12 = second_gt12 = 0
    for m in _HEADER.finditer(text):
        d1, d2 = m.group("d1"), m.group("d2")
        if len(d1) == 4:            # ISO 2023-08-12, unambiguous
            return "YMD", "certain"
        if int(d1) > 12:
            first_gt12 += 1
        if int(d2) > 12:
            second_gt12 += 1

    if first_gt12 and not second_gt12:
        return "DMY", "certain"
    if second_gt12 and not first_gt12:
        return "MDY", "certain"
    if first_gt12 and second_gt12:
        return "DMY", "conflicting"     # corrupt/concatenated export
    return "DMY", "assumed"


def _build_timestamp(m: re.Match, order: str) -> datetime | None:
    d1, d2, d3 = int(m.group("d1")), int(m.group("d2")), int(m.group("d3"))

    if order == "YMD" or len(m.group("d1")) == 4:
        year, month, day = d1, d2, d3
    elif order == "MDY":
        month, day, year = d1, d2, d3
    else:
        day, month, year = d1, d2, d3

    if year < 100:                       # two-digit year: 23 -> 2023
        year += 2000

    hh, mm = int(m.group("hh")), int(m.group("mm"))
    ss = int(m.group("ss") or 0)

    ampm = (m.group("ampm") or "").replace(".", "").lower()
    if ampm == "pm" and hh != 12:
        hh += 12
    elif ampm == "am" and hh == 12:
        hh = 0

    try:
        return datetime(year, month, day, hh, mm, ss)
    except ValueError:
        return None


def _classify(sender: str | None, text: str) -> MessageType:
    stripped = text.strip()
    if sender is None:
        return "system"
    if _DELETED_RE.match(stripped):
        return "deleted"

    # Match the placeholder on the first line only. A captioned attachment puts
    # the caption underneath, and testing the whole multi-line body against an
    # anchored pattern silently reclassified those as ordinary text -- so the
    # media count came out low and the caption was never linked to its file.
    first_line = stripped.split("\n", 1)[0].strip()
    if _MEDIA_RE.match(first_line) or extract_attachment(stripped)[0]:
        return "media"
    return "text"


def _split_sender(body: str, known_senders: set[str] | None) -> tuple[str | None, str]:
    """
    Split "Name: message" into its parts.

    Two-pass aware: once we know the real sender set, a line whose prefix isn't
    a known sender is treated as a system notice even if it contains a colon
    (e.g. "Karan changed the subject to: Trip 2024").
    """
    m = _SENDER.match(body)
    if not m:
        return None, body

    sender = m.group("sender").strip()
    text = m.group("text")

    if known_senders is not None:
        if sender not in known_senders:
            return None, body
        return sender, text

    # First pass: reject prefixes that look like system-notice prose.
    if _SYSTEM_RE.search(sender):
        return None, body
    return sender, text


def _iter_headers(lines: list[str], order: str) -> Iterator[tuple[int, datetime, str]]:
    for i, raw in enumerate(lines):
        m = _HEADER.match(raw)
        if not m:
            continue
        ts = _build_timestamp(m, order)
        if ts is not None:
            yield i, ts, m.group("body")


def _order_fits(lines: list[str], order: str) -> tuple[int, int]:
    """Count how many header-shaped lines yield a valid date under `order`."""
    ok = total = 0
    for raw in lines:
        m = _HEADER.match(raw)
        if not m:
            continue
        total += 1
        if _build_timestamp(m, order) is not None:
            ok += 1
    return ok, total


def _resolve_date_order(lines: list[str], order: str, report: ParseReport) -> str:
    """
    Verify the inferred order actually parses, and flip it if it doesn't.

    A wrong guess turns "08/13/2023" into month 13, which raises and makes the
    line look like message continuation -- the export appears half-empty rather
    than failing loudly. Cross-checking both orders catches that.
    """
    ok, total = _order_fits(lines, order)
    if total == 0 or ok == total:
        return order

    alternative = "MDY" if order == "DMY" else "DMY"
    alt_ok, _ = _order_fits(lines, alternative)
    if alt_ok > ok:
        report.warnings.append(
            f"Inferred {order} date order but {total - ok} of {total} timestamps "
            f"were invalid under it; using {alternative} instead."
        )
        report.date_order = alternative
        report.date_order_confidence = "corrected"
        return alternative

    report.warnings.append(
        f"{total - ok} of {total} timestamps could not be parsed under either "
        "date order; those lines were skipped."
    )
    return order


def parse_export(text: str) -> tuple[list[RawMessage], ParseReport]:
    """
    Parse a full export into messages plus a report of what happened.

    Runs two passes: the first learns the sender vocabulary, the second uses it
    to disambiguate system notices from messages whose text contains a colon.
    """
    report = ParseReport()
    lines = [_clean(line) for line in text.splitlines()]
    report.total_lines = len(lines)

    order, confidence = detect_date_order(text)
    report.date_order, report.date_order_confidence = order, confidence
    if confidence == "assumed":
        report.warnings.append(
            "No date above the 12th appears in this export, so day/month order "
            "is ambiguous. Assuming D/M/Y (WhatsApp's non-US default)."
        )
    elif confidence == "conflicting":
        report.warnings.append(
            "Dates conflict: both positions exceed 12 somewhere in the file. "
            "This export may be two files concatenated. Assuming D/M/Y."
        )

    order = _resolve_date_order(lines, order, report)

    # --- Pass 1: learn the sender vocabulary -------------------------------
    candidates: Counter[str] = Counter()
    for _, _, body in _iter_headers(lines, order):
        sender, _ = _split_sender(body, known_senders=None)
        if sender:
            candidates[sender] += 1

    # A candidate is a real sender unless it reads as system-notice prose --
    # "Karan changed the subject to" is a prefix, not a person.
    #
    # An earlier version also required singletons to be at most five words,
    # which quietly deleted real people: a participant with a long display name
    # who sent exactly one message failed the test, their message was
    # reclassified as a system notice, and system notices are never stored. They
    # vanished from every count with nothing reported. Word count is not
    # evidence of anything, so it is gone.
    known = {
        s for s, n in candidates.items()
        if n > 1 or not _SYSTEM_RE.search(s)
    }

    # --- Pass 2: build messages, attaching continuation lines ---------------
    messages: list[RawMessage] = []
    header_idx = {i: (ts, body) for i, ts, body in _iter_headers(lines, order)}
    current: RawMessage | None = None

    for i, raw in enumerate(lines):
        if i in header_idx:
            ts, body = header_idx[i]
            sender, text = _split_sender(body, known_senders=known)
            current = RawMessage(
                ts=ts,
                sender=sender,
                text=text,
                msg_type=_classify(sender, text),
                line_no=i + 1,
            )
            messages.append(current)
            continue

        if not raw.strip():
            if current is not None:
                current.text += "\n"
                report.continuation_lines += 1
            continue

        if current is not None:
            # Continuation of a multi-line message.
            current.text += "\n" + raw
            report.continuation_lines += 1
        else:
            report.unparsed_lines += 1

    # Re-classify after continuations are attached (a media placeholder can
    # only be judged on the complete body).
    for msg in messages:
        msg.text = msg.text.strip()
        msg.msg_type = _classify(msg.sender, msg.text)

        if msg.msg_type == "media":
            # Peel the filename off the body so `text` holds only the caption.
            filename, caption = extract_attachment(msg.text)
            if filename:
                msg.attachment = filename
                msg.text = caption
                report.attached_files += 1
            else:
                # No file, just a placeholder. Drop the placeholder line so the
                # caption is not polluted with "<Media omitted>" when searching.
                head, _, rest = msg.text.strip().partition("\n")
                if _MEDIA_RE.match(head.strip()):
                    msg.text = rest.strip()

        if msg.msg_type == "system":
            report.system_messages += 1
        elif msg.msg_type == "media":
            report.media_messages += 1
        elif msg.msg_type == "deleted":
            report.deleted_messages += 1
        if msg.sender:
            report.senders[msg.sender] += 1

    report.parsed_messages = len(messages)
    if report.parsed_messages == 0:
        report.warnings.append(
            "No messages parsed. Is this a WhatsApp 'Export chat' .txt file?"
        )

    return messages, report


def parse_file(path: str) -> tuple[list[RawMessage], ParseReport]:
    """Read an export from disk, tolerating the encodings WhatsApp emits."""
    for encoding in ("utf-8-sig", "utf-8", "utf-16", "latin-1"):
        try:
            with open(path, "r", encoding=encoding) as fh:
                text = fh.read()
        except (UnicodeDecodeError, UnicodeError):
            continue
        # A successful decode that yields no headers means we picked wrong.
        if _HEADER.search(text) or encoding == "latin-1":
            return parse_export(text)
    raise ValueError(f"Could not decode {path} as a WhatsApp export")


def has_url(text: str) -> bool:
    return bool(_URL_RE.search(text))


def emoji_count(text: str) -> int:
    """Count emoji-ish codepoints without pulling in a dependency."""
    n = 0
    for ch in text:
        cp = ord(ch)
        if (
            0x1F300 <= cp <= 0x1FAFF
            or 0x2600 <= cp <= 0x27BF
            or cp in (0x2764, 0x2B50, 0x2B55)
            or 0x1F000 <= cp <= 0x1F0FF
        ):
            n += 1
        elif unicodedata.category(ch) == "So":
            n += 1
    return n
