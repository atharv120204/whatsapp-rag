"""
Tests for combining overlapping exports.

The scenario these protect: WhatsApp truncates a with-media export to roughly
the last 10,000 messages but exports the full history without media. Combining
the two is the main reason merge exists, and it is exactly the case where naive
deduplication fails, because the same photo appears as a filename in one file
and as "<Media omitted>" in the other.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.index.dedup import assign_keys, base_key, merge  # noqa: E402
from app.parse.whatsapp import RawMessage, parse_export  # noqa: E402


def _msg(minute: int, sender: str, text: str, msg_type: str = "text",
         attachment: str | None = None) -> RawMessage:
    return RawMessage(
        ts=datetime(2024, 1, 1, 9, 0) + timedelta(minutes=minute),
        sender=sender,
        text=text,
        msg_type=msg_type,
        line_no=minute,
        attachment=attachment,
    )


def test_reimporting_the_same_export_adds_nothing():
    messages = [_msg(0, "A", "hey"), _msg(1, "B", "hi"), _msg(2, "A", "ok")]
    keys = assign_keys(messages)

    outcome = merge(messages, keys, messages)
    assert outcome.added == 0
    assert outcome.skipped == 3
    assert len(outcome.messages) == 3


def test_repeated_identical_messages_are_kept_separately():
    # Sending "ok" twice in the same minute is two messages, not one, and
    # re-importing must match both rather than collapsing them.
    messages = [_msg(0, "A", "ok"), _msg(0, "A", "ok")]
    keys = assign_keys(messages)
    assert len(set(keys)) == 2

    outcome = merge(messages, keys, messages)
    assert outcome.added == 0
    assert len(outcome.messages) == 2


def test_new_messages_are_appended():
    old = [_msg(0, "A", "hey"), _msg(1, "B", "hi")]
    new = old + [_msg(5, "A", "still there?")]

    outcome = merge(old, assign_keys(old), new)
    assert outcome.added == 1
    assert len(outcome.messages) == 3
    assert outcome.messages[-1].text == "still there?"


def test_older_history_merges_in_correct_order():
    # A full-history export reaches further back than what is already stored.
    stored = [_msg(100, "A", "recent")]
    full_history = [_msg(0, "B", "ancient"), _msg(50, "A", "middle"),
                    _msg(100, "A", "recent")]

    outcome = merge(stored, assign_keys(stored), full_history)
    assert outcome.added == 2
    assert [m.text for m in outcome.messages] == ["ancient", "middle", "recent"]


def test_media_matches_across_export_modes():
    """The whole point: <Media omitted> and the real filename are one message."""
    without_media = [
        _msg(0, "A", "look at this"),
        _msg(1, "B", "", msg_type="media"),            # <Media omitted>
        _msg(2, "A", "nice"),
    ]
    with_media = [
        _msg(0, "A", "look at this"),
        _msg(1, "B", "at the beach", msg_type="media",
             attachment="IMG-20240101-WA0001.jpg"),
        _msg(2, "A", "nice"),
    ]

    outcome = merge(without_media, assign_keys(without_media), with_media)

    # Three messages, not four: the photo was not duplicated.
    assert len(outcome.messages) == 3
    assert outcome.added == 0

    # And the richer version won, so the file is now attached.
    photo = outcome.messages[1]
    assert photo.attachment == "IMG-20240101-WA0001.jpg"
    assert photo.text == "at the beach"
    assert outcome.upgraded == 1


def test_richer_version_does_not_lose_to_poorer_one():
    """Merging in the *other* order must not discard the attachment."""
    with_media = [_msg(1, "B", "at the beach", msg_type="media",
                       attachment="IMG-1.jpg")]
    without_media = [_msg(1, "B", "", msg_type="media")]

    outcome = merge(with_media, assign_keys(with_media), without_media)
    assert len(outcome.messages) == 1
    assert outcome.messages[0].attachment == "IMG-1.jpg"
    assert outcome.upgraded == 0


def test_two_photos_in_the_same_minute_stay_distinct():
    stored = [
        _msg(1, "B", "", msg_type="media", attachment="IMG-1.jpg"),
        _msg(1, "B", "", msg_type="media", attachment="IMG-2.jpg"),
    ]
    outcome = merge(stored, assign_keys(stored), stored)
    assert len(outcome.messages) == 2
    assert outcome.added == 0


def test_media_key_ignores_filename_and_caption():
    ts = datetime(2024, 1, 1, 9, 0)
    a = base_key(ts, "B", "", "media", None)
    b = base_key(ts, "B", "at the beach", "media", "IMG-1.jpg")
    assert a == b


def test_text_key_is_whitespace_and_case_insensitive():
    ts = datetime(2024, 1, 1, 9, 0)
    assert base_key(ts, "A", "Hey  there", "text", None) == \
           base_key(ts, "A", "hey there", "text", None)


def test_different_senders_never_collide():
    ts = datetime(2024, 1, 1, 9, 0)
    assert base_key(ts, "A", "ok", "text", None) != \
           base_key(ts, "B", "ok", "text", None)


def test_end_to_end_two_export_modes():
    """Parse both export dialects of the same chat and merge them."""
    without_media = """01/01/2024, 09:00 - Karan: look at this
01/01/2024, 09:01 - Priya: <Media omitted>
01/01/2024, 09:02 - Karan: nice
"""
    with_media = """01/01/2024, 09:00 - Karan: look at this
01/01/2024, 09:01 - Priya: IMG-20240101-WA0001.jpg (file attached)
at the beach
01/01/2024, 09:02 - Karan: nice
01/01/2024, 09:30 - Priya: heading back
"""
    first, _ = parse_export(without_media)
    second, _ = parse_export(with_media)

    outcome = merge(first, assign_keys(first), second)

    assert len(outcome.messages) == 4      # 3 shared + 1 genuinely new
    assert outcome.added == 1
    assert outcome.upgraded == 1
    assert outcome.messages[1].attachment == "IMG-20240101-WA0001.jpg"


def test_participant_overlap_detects_a_different_chat():
    """Merging an unrelated chat must be caught, not silently blended in."""
    from app.index.build import _participant_overlap

    goa = [_msg(0, "Karan", "hey"), _msg(1, "Priya", "hi")]
    same_chat_later = [_msg(0, "Karan", "hey"), _msg(9, "Priya", "still on?")]
    different_chat = [_msg(0, "Kabir", "lunch?"), _msg(1, "Meera", "sure")]

    assert _participant_overlap(goa, same_chat_later) == 1.0
    assert _participant_overlap(goa, different_chat) == 0.0
    assert _participant_overlap([], different_chat) is None


def test_partial_overlap_is_allowed():
    # A group where one person left and another joined still overlaps enough
    # to be the same chat, and must not be blocked.
    from app.index.build import _participant_overlap

    before = [_msg(0, "A", "x"), _msg(1, "B", "y"), _msg(2, "C", "z")]
    after = [_msg(3, "A", "x"), _msg(4, "B", "y"), _msg(5, "D", "new here")]
    assert _participant_overlap(before, after) > 0.2



if __name__ == "__main__":
    import traceback

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL  {t.__name__}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
