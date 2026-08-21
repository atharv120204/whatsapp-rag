"""
Participant identity resolution.

In a group export the same human shows up under several strings:

    +91 98765 43210        before you saved their number
    ~Rohit                 WhatsApp push name for an unsaved contact
    Rohit Sharma           after you saved them
    Rohit  Sharma          stray double space

Counting these as four people makes every per-person statistic wrong, so we
collapse them to a canonical participant before anything else runs. Automatic
merging is deliberately conservative -- it only merges what it can justify --
and anything it is unsure about is reported for the user to confirm.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

_PHONE_RE = re.compile(r"^\+?[\d\s\-()]{7,20}$")
_NON_DIGIT = re.compile(r"\D")
_WS = re.compile(r"\s+")

# "~Rohit" / "~ Rohit" push-name prefix.
_PUSH_PREFIX = re.compile(r"^~\s*")


@dataclass
class Participant:
    """One resolved human, with every string they appeared as."""

    participant_id: str          # stable key: canonical form
    display_name: str            # nicest label we have
    aliases: set[str] = field(default_factory=set)
    is_phone_only: bool = False
    message_count: int = 0

    def as_dict(self) -> dict:
        return {
            "participant_id": self.participant_id,
            "display_name": self.display_name,
            "aliases": sorted(self.aliases),
            "is_phone_only": self.is_phone_only,
            "message_count": self.message_count,
        }


def is_phone_number(name: str) -> bool:
    stripped = _PUSH_PREFIX.sub("", name).strip()
    if not _PHONE_RE.match(stripped):
        return False
    return len(_NON_DIGIT.sub("", stripped)) >= 7


def canonical_key(name: str) -> str:
    """
    Reduce a sender string to a matching key.

    Phone numbers collapse to their last 10 digits, so "+91 98765 43210",
    "+919876543210" and "98765 43210" all agree. Names collapse to
    casefolded, whitespace-normalised text.
    """
    stripped = _PUSH_PREFIX.sub("", name).strip()
    if is_phone_number(stripped):
        digits = _NON_DIGIT.sub("", stripped)
        return "phone:" + digits[-10:]
    return "name:" + _WS.sub(" ", stripped).casefold()


def _prettiest(aliases: set[str]) -> str:
    """
    Pick the label a human would recognise: a real name over a phone number,
    longer over shorter (fuller names win), push-name tildes stripped.
    """
    cleaned = {_PUSH_PREFIX.sub("", a).strip() for a in aliases}
    named = [a for a in cleaned if not is_phone_number(a)]
    pool = named or list(cleaned)
    return max(pool, key=lambda a: (len(a), a))


def resolve_participants(
    sender_counts: Counter[str],
    manual_aliases: dict[str, str] | None = None,
) -> tuple[dict[str, Participant], list[str]]:
    """
    Group raw sender strings into participants.

    `manual_aliases` maps a raw sender string to the participant_id it should
    join, letting the user fix links we cannot infer (a number saved under a
    nickname, say). Returns the participant table and a list of suggestions
    worth showing the user.
    """
    manual_aliases = manual_aliases or {}
    participants: dict[str, Participant] = {}

    for raw, count in sender_counts.items():
        key = manual_aliases.get(raw) or canonical_key(raw)
        p = participants.get(key)
        if p is None:
            p = Participant(participant_id=key, display_name=raw)
            participants[key] = p
        p.aliases.add(raw)
        p.message_count += count

    for p in participants.values():
        p.display_name = _prettiest(p.aliases)
        p.is_phone_only = all(is_phone_number(a) for a in p.aliases)

    return participants, _suggest_merges(participants)


def _suggest_merges(participants: dict[str, Participant]) -> list[str]:
    """
    Flag pairs that are probably the same person but cannot be merged safely.

    We suggest rather than act: silently merging two people corrupts the data
    in a way that is very hard to notice downstream.
    """
    suggestions: list[str] = []
    names = [(p, p.display_name.casefold()) for p in participants.values()
             if not p.is_phone_only]

    for i, (p_a, name_a) in enumerate(names):
        for p_b, name_b in names[i + 1:]:
            # One name is a prefix of the other: "Rohit" vs "Rohit Sharma".
            if name_a != name_b and (
                name_b.startswith(name_a + " ") or name_a.startswith(name_b + " ")
            ):
                suggestions.append(
                    f"{p_a.display_name!r} ({p_a.message_count} msgs) and "
                    f"{p_b.display_name!r} ({p_b.message_count} msgs) may be "
                    "the same person."
                )

    phone_only = [p for p in participants.values() if p.is_phone_only]
    if phone_only:
        suggestions.append(
            f"{len(phone_only)} participant(s) appear only as phone numbers. "
            "Map them to names to make per-person answers readable."
        )
    return suggestions


def build_alias_lookup(participants: dict[str, Participant]) -> dict[str, str]:
    """Flat raw-sender -> participant_id map, for tagging messages at ingest."""
    return {
        alias: p.participant_id
        for p in participants.values()
        for alias in p.aliases
    }
