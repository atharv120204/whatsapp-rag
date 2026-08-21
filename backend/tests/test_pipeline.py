"""Tests for identity resolution, session logic and the SQL guard."""

import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agent.sql_guard import validate  # noqa: E402
from app.parse.normalize import (  # noqa: E402
    build_alias_lookup,
    canonical_key,
    is_phone_number,
    resolve_participants,
)
from app.parse.sessions import enrich  # noqa: E402
from app.parse.whatsapp import RawMessage  # noqa: E402


# --- identity resolution ------------------------------------------------------

def test_phone_detection():
    assert is_phone_number("+91 98765 43210")
    assert is_phone_number("+919876543210")
    assert is_phone_number("~ +91 98765 43210")
    assert not is_phone_number("Rohit Sharma")
    assert not is_phone_number("R2D2")


def test_phone_variants_collapse_to_one_person():
    # The same number written three ways must be one participant, or every
    # per-person statistic double counts.
    assert (
        canonical_key("+91 98765 43210")
        == canonical_key("+919876543210")
        == canonical_key("98765 43210")
    )


def test_push_name_prefix_is_stripped():
    assert canonical_key("~Rohit") == canonical_key("Rohit")
    assert canonical_key("~ Rohit") == canonical_key("rohit")


def test_resolve_prefers_readable_name():
    counts = Counter({"~Priya": 5, "Priya Sharma": 30, "+91 90000 00001": 12})
    participants, _ = resolve_participants(counts)
    assert len(participants) == 3

    lookup = build_alias_lookup(participants)
    phone_id = lookup["+91 90000 00001"]
    assert participants[phone_id].is_phone_only
    assert participants[phone_id].display_name == "+91 90000 00001"

    named = participants[lookup["Priya Sharma"]]
    assert named.display_name == "Priya Sharma"
    assert not named.is_phone_only


def test_merge_suggestion_for_prefix_names():
    counts = Counter({"Rohit": 10, "Rohit Sharma": 20})
    _, suggestions = resolve_participants(counts)
    assert any("Rohit" in s for s in suggestions)


# --- session segmentation -----------------------------------------------------

def _msg(offset_minutes: int, sender: str, base=datetime(2024, 1, 1, 9, 0)):
    return RawMessage(
        ts=base + timedelta(minutes=offset_minutes),
        sender=sender,
        text="hello",
        msg_type="text",
        line_no=offset_minutes,
    )


def test_sessions_split_on_silence():
    messages = [
        _msg(0, "A"), _msg(5, "B"), _msg(9, "A"),      # session 1
        _msg(9 + 5 * 60, "B"), _msg(9 + 5 * 60 + 3, "A"),  # session 2 (5h gap)
    ]
    lookup = {"A": "name:a", "B": "name:b"}
    rows = enrich(messages, lookup, gap_hours=4.0)

    assert [r.session_id for r in rows] == [1, 1, 1, 2, 2]
    assert [r.is_session_start for r in rows] == [True, False, False, True, False]
    # B started the second conversation.
    assert rows[3].participant_id == "name:b"


def test_reply_gap_only_counts_cross_speaker():
    messages = [_msg(0, "A"), _msg(4, "A"), _msg(10, "B")]
    rows = enrich(messages, {"A": "name:a", "B": "name:b"}, gap_hours=4.0)

    assert rows[0].reply_gap_seconds is None      # first message
    assert rows[1].reply_gap_seconds is None      # A following themselves
    assert rows[2].reply_gap_seconds == 6 * 60    # B replying to A


def test_system_messages_excluded_from_sessions():
    messages = [
        _msg(0, "A"),
        RawMessage(ts=datetime(2024, 1, 1, 9, 1), sender=None,
                   text="X joined", msg_type="system", line_no=2),
        _msg(3, "B"),
    ]
    rows = enrich(messages, {"A": "name:a", "B": "name:b"}, gap_hours=4.0)
    assert len(rows) == 2
    assert all(r.msg_type != "system" for r in rows)


def test_messages_are_sorted_by_time():
    # A malformed export can list messages out of order; session logic depends
    # on a monotonic stream.
    messages = [_msg(30, "A"), _msg(0, "B"), _msg(15, "A")]
    rows = enrich(messages, {"A": "name:a", "B": "name:b"}, gap_hours=4.0)
    assert [r.ts for r in rows] == sorted(r.ts for r in rows)


# --- SQL guard ----------------------------------------------------------------

def test_guard_allows_plain_select():
    result = validate("SELECT sender, COUNT(*) FROM v_messages GROUP BY 1")
    assert result.ok
    assert "LIMIT" in result.sql.upper()


def test_guard_allows_cte():
    assert validate("WITH x AS (SELECT 1 AS a) SELECT * FROM x").ok


def test_guard_blocks_writes():
    for query in (
        "DROP TABLE messages",
        "DELETE FROM messages",
        "UPDATE messages SET text = 'x'",
        "INSERT INTO messages VALUES (1)",
        "CREATE TABLE evil (a INT)",
        "ATTACH 'other.db' AS other",
    ):
        assert not validate(query).ok, query


def test_guard_blocks_stacked_statements():
    assert not validate("SELECT 1; DROP TABLE messages").ok


def test_guard_blocks_filesystem_functions():
    for query in (
        "SELECT * FROM read_csv('/etc/passwd')",
        "SELECT * FROM read_parquet('x.parquet')",
        "SELECT * FROM glob('*')",
    ):
        assert not validate(query).ok, query


def test_guard_blocks_catalog_snooping():
    assert not validate("SELECT * FROM information_schema.tables").ok
    assert not validate("SELECT * FROM duckdb_settings()").ok


def test_guard_ignores_keywords_inside_string_literals():
    # A message that merely mentions "drop table" must not block a valid query.
    result = validate("SELECT * FROM v_messages WHERE text = 'drop table now'")
    assert result.ok


def test_guard_respects_existing_limit():
    result = validate("SELECT * FROM v_messages LIMIT 5")
    assert result.ok
    assert result.sql.strip().upper().endswith("LIMIT 5")


# --- reading a window of conversation -------------------------------------------

def _window_archive():
    """An in-memory archive shaped like a real busy day: mostly attachments."""
    import duckdb

    from app.db import INDEXES, MACROS, SCHEMA, VIEWS

    conn = duckdb.connect()
    conn.execute(SCHEMA.format(dims=768))
    for stmt in INDEXES + MACROS + VIEWS:
        conn.execute(stmt)
    conn.execute("INSERT INTO participants VALUES "
                 "('p1','Rohit',['Rohit'],false,0),"
                 "('p2','Neha',['Neha'],false,0)")

    rows = []
    msg_id = 0
    def add(minute, pid, text, msg_type):
        nonlocal msg_id
        rows.append((msg_id, f"2026-02-09 10:{minute:02d}:00", pid,
                     "x", text, msg_type))
        msg_id += 1

    add(0, "p1", "Crazy", "text")
    for _ in range(40):                    # a burst of photos
        add(4, "p2", "", "media")
    add(8, "p1", "Ek toh party ke paise nahi diye", "text")
    add(9, "p1", "", "text")               # an empty message mid-burst
    for _ in range(10):
        add(10, "p2", "", "media")
    add(12, "p1", "Happy Birthday Rohit", "text")

    for r in rows:
        conn.execute(
            "INSERT INTO messages (msg_id, ts, participant_id, sender_raw, "
            "text, msg_type, date) VALUES (?, ?::TIMESTAMP, ?, ?, ?, ?, "
            "'2026-02-09')", list(r))
    return conn


def test_window_read_collapses_attachment_runs():
    """
    A busy day is mostly attachments; they must not crowd out the words.

    Returned as one row per message, 50 attachments filled the result and the
    model never reached the conversation -- it reported that no content was
    available for a day that contained an argument and a birthday.
    """
    from app.agent.tools import build_tools

    tools = build_tools(_window_archive())
    result = tools["search_chat"](after="2026-02-09", before="2026-02-09")

    assert result["message_count"] == 54
    assert result["messages_with_text"] == 3
    assert result["attachments"] == 50
    assert result["empty_messages"] == 1

    transcript = result["transcript"]
    # Every word anyone typed survives.
    assert "Ek toh party ke paise nahi diye" in transcript
    assert "Happy Birthday Rohit" in transcript
    assert "Crazy" in transcript
    # The bursts collapse rather than repeating fifty times.
    assert "(40 attachments)" in transcript
    assert "(10 attachments)" in transcript
    assert transcript.count("attachment") == 2


def test_window_read_is_compact_enough_to_send():
    """The whole day must fit inside a single tool result."""
    from app.config import settings
    from app.agent.tools import build_tools

    tools = build_tools(_window_archive())
    result = tools["search_chat"](after="2026-02-09", before="2026-02-09")
    assert len(result["transcript"]) < settings.tool_result_max_chars


def test_empty_message_does_not_split_an_attachment_run():
    """An empty message between photos would otherwise fragment the run."""
    from app.agent.tools import build_tools

    tools = build_tools(_window_archive())
    transcript = tools["search_chat"](after="2026-02-09",
                                      before="2026-02-09")["transcript"]
    assert "(40 attachments)" in transcript      # not 39 + 1


def test_window_read_needs_a_filter():
    """Reading the entire archive by accident is not useful."""
    from app.agent.tools import build_tools

    tools = build_tools(_window_archive())
    result = tools["search_chat"]()
    assert result.get("error")



# --- concurrency ----------------------------------------------------------------

def test_concurrent_reads_do_not_corrupt_each_other():
    """
    FastAPI serves every endpoint from a threadpool, so reads overlap.

    Sharing one DuckDB handle let two queries interleave and tear each other's
    result sets apart -- which surfaced as a column of sender names being read
    as integers, not as an obvious failure. Cursors keep them separate.
    """
    import tempfile
    import threading
    from pathlib import Path as P

    from app import archives, db
    from app.api import insights, stats
    from app.config import settings

    with tempfile.TemporaryDirectory() as tmp:
        settings.data_dir = P(tmp)
        db.close_all()
        archive = archives.create_archive("Concurrency")
        conn = db.get_connection(archive)
        conn.execute("INSERT INTO participants VALUES "
                     "('p1','Alice',['Alice'],false,0),"
                     "('p2','Bob',['Bob'],false,0)")
        for i in range(120):
            conn.execute(
                "INSERT INTO messages (msg_id, ts, participant_id, sender_raw, "
                "text, msg_type, date, hour, weekday, year_month, session_id, "
                "is_session_start, word_count, emoji_count, is_question) "
                "VALUES (?, ?::TIMESTAMP, ?, 'x', ?, 'text', '2026-02-09', 10, "
                "0, '2026-02', ?, ?, 3, 0, false)",
                [i, f"2026-02-09 10:{i % 60:02d}:00",
                 "p1" if i % 2 else "p2", f"haha message {i}", i // 10,
                 i % 10 == 0])

        errors: list[str] = []

        def hammer(fn):
            for _ in range(5):
                try:
                    fn(db.get_cursor(archive))
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"{type(exc).__name__}: {exc}")

        jobs = [
            lambda c: stats.leaderboard(c),
            lambda c: insights.superlatives(c),
            lambda c: insights.find_moments(c, "funny", 3),
            lambda c: stats.activity_heatmap(c),
        ]
        threads = [threading.Thread(target=hammer, args=(j,))
                   for j in jobs for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(120)
            assert not t.is_alive(), "a concurrent read hung"

        db.close_all()
        assert not errors, f"{len(errors)} concurrent read failures: {errors[:3]}"


def test_moment_finders_survive_an_empty_archive():
    """Every finder must return cleanly when there is nothing to find."""
    import tempfile
    from pathlib import Path as P

    from app import archives, db
    from app.api.insights import MOMENT_KINDS, find_moments
    from app.config import settings

    with tempfile.TemporaryDirectory() as tmp:
        settings.data_dir = P(tmp)
        db.close_all()
        archive = archives.create_archive("Empty")
        conn = db.get_cursor(archive)

        for kind in MOMENT_KINDS:
            result = find_moments(conn, kind, 3)
            assert result["count"] == 0, kind
            assert result["empty_reason"], kind
        db.close_all()



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
