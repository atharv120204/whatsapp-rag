"""
Character analysis of a group chat: funny moments, arguments, late-night talks,
and the superlatives people actually want to know.

The approach matters. Asking a model to "find the funny bits" in tens of thousands of messages
means either sending it everything (impossible on any sane budget) or sending a
sample (which finds whatever happened to be sampled). So candidates are found
here, in SQL, over every message at once and for free -- then only the handful
that score highest are handed to a model to describe.

The scoring is heuristic and says so. A conversation dense with 😂 is probably
funny; a fast two-person exchange with long messages, no laughter and an
apology afterwards is probably an argument. These are signals, not verdicts,
and the excerpt is always returned alongside so the reader can judge.

Markers cover Hinglish and Devanagari as well as English, because a chat that
is half "bhai kya kar raha hai" scores nothing on English-only patterns.
"""

from __future__ import annotations

from typing import Any

# --- signal vocabularies ------------------------------------------------------

LAUGH = [
    "😂", "🤣", "😹", "😆", "😅", "🥲", "lmao", "lmfao", "rofl",
    "haha", "hahaha", "hehe", "hehehe", "lol", "lel", "xd",
    "hasi", "mazak", "😭",          # 😭 is used for laughing-crying in Hinglish
]

ARGUMENT = [
    "wtf", "seriously", "shut up", "stop it", "enough", "whatever",
    "bakwas", "bakwaas", "pagal", "chup", "galat", "jhagda", "ladai",
    "gussa", "bezzati", "tameez", "attitude", "blame", "fault",
    "बकवास", "पागल", "गलत", "झगड़ा", "गुस्सा",
]

APOLOGY = [
    "sorry", "sry", "maaf", "galti", "my bad", "apologies", "मुझे माफ",
    "माफ", "गलती", "chill", "shant", "shaant", "peace", "sorted",
]

DEEP = [
    "life", "future", "career", "scared", "afraid", "honestly", "truth",
    "feel", "feeling", "lonely", "depress", "anxiety", "family", "parents",
    "love", "trust", "regret", "dream", "believe", "think about",
    "zindagi", "sach", "dar", "akela", "tension", "soch", "pyaar",
    "जिंदगी", "सच", "डर", "अकेला", "सोच",
]

PLAN = [
    "plan", "meet", "tomorrow", "tonight", "book", "ticket", "trip",
    "party", "kal", "aaj", "chalo", "chalte", "milte", "jaana", "jayenge",
    "reach", "leaving", "time kya", "kitne baje",
]


def _contains(column: str, words: list[str]) -> str:
    """SQL that is true when the column mentions any of these words."""
    parts = [f"LOWER({column}) LIKE '%{w.lower()}%'" for w in words]
    return "(" + " OR ".join(parts) + ")"


def _rows(conn, sql: str, params: list | None = None) -> list[dict]:
    cur = conn.execute(sql, params or [])
    columns = [d[0] for d in cur.description]
    return [
        {col: _clean(val) for col, val in zip(columns, row)}
        for row in cur.fetchall()
    ]


def _clean(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return str(value)


# --- per-session scoring ------------------------------------------------------

_SESSION_STATS = f"""
WITH stats AS (
    SELECT
        session_id,
        COUNT(*)                                   AS messages,
        COUNT(DISTINCT participant_id)             AS people,
        MIN(ts)                                    AS started,
        MAX(ts)                                    AS ended,
        date_diff('minute', MIN(ts), MAX(ts))      AS minutes,
        ROUND(AVG(word_count), 1)                  AS avg_words,
        MAX(word_count)                            AS longest_message,
        SUM(emoji_count)                           AS emojis,
        SUM(CASE WHEN msg_type = 'text' THEN 1 ELSE 0 END)      AS text_messages,
        SUM(CASE WHEN is_question THEN 1 ELSE 0 END)            AS questions,
        SUM(CASE WHEN {_contains('text', LAUGH)} THEN 1 ELSE 0 END)    AS laughs,
        SUM(CASE WHEN {_contains('text', ARGUMENT)} THEN 1 ELSE 0 END) AS heat,
        SUM(CASE WHEN {_contains('text', APOLOGY)} THEN 1 ELSE 0 END)  AS apologies,
        SUM(CASE WHEN {_contains('text', DEEP)} THEN 1 ELSE 0 END)     AS reflective,
        SUM(CASE WHEN hour >= 0 AND hour < 5 THEN 1 ELSE 0 END) AS night_messages,
        ROUND(MEDIAN(reply_gap_seconds), 0)        AS median_reply_seconds
    FROM v_messages
    WHERE msg_type <> 'system'
    GROUP BY session_id
)
"""


def _excerpt(conn, session_id: int, limit: int = 24) -> str:
    """Render a session as a compact transcript, collapsing attachment runs."""
    rows = conn.execute("""
        SELECT s.ts, s.sender, s.content
        FROM v_searchable s
        JOIN messages m USING (msg_id)
        WHERE m.session_id = ? AND m.msg_type <> 'system'
        ORDER BY s.ts
    """, [session_id]).fetchall()

    lines: list[str] = []
    pending_sender, pending_count, pending_time = None, 0, ""

    def flush():
        nonlocal pending_sender, pending_count
        if pending_sender and pending_count:
            noun = "attachment" if pending_count == 1 else "attachments"
            lines.append(f"{pending_time} {pending_sender}: "
                         f"({pending_count} {noun})")
        pending_sender, pending_count = None, 0

    for ts, sender, content in rows:
        text = (content or "").strip()
        if not text:
            continue
        stamp = ts.strftime("%H:%M") if hasattr(ts, "strftime") else str(ts)
        if text.startswith("[") and text.endswith("]"):
            if sender == pending_sender:
                pending_count += 1
            else:
                flush()
                pending_sender, pending_count, pending_time = sender, 1, stamp
            continue
        flush()
        lines.append(f"{stamp} {sender}: {text}")

    flush()

    if len(lines) > limit:
        half = limit // 2
        lines = lines[:half] + [f"... {len(lines) - limit} more messages ..."] \
            + lines[-half:]
    return "\n".join(lines)


def _with_excerpts(conn, rows: list[dict], limit: int = 24) -> list[dict]:
    for row in rows:
        row["excerpt"] = _excerpt(conn, row["session_id"], limit)
    return rows


# --- moment finders -----------------------------------------------------------

def baseline(conn) -> dict:
    """
    What is normal for *this* chat.

    Absolute thresholds do not transfer between archives. A group whose average
    message is four words is not less thoughtful than one averaging fifteen --
    it just types differently. Scoring against the archive's own habits is what
    stops "deep conversation" from meaning "long by some other group's
    standard", and it is why an early version surfaced a session of late-night
    jokes as the most profound thing in the chat.
    """
    row = conn.execute(f"""
        SELECT
            ROUND(AVG(word_count), 2),
            ROUND(AVG(CASE WHEN {_contains('text', LAUGH)} THEN 1.0 ELSE 0.0 END), 3),
            ROUND(AVG(emoji_count), 2),
            COUNT(*)
        FROM v_messages
        WHERE msg_type = 'text'
    """).fetchone()

    return {
        "avg_words": float(row[0] or 1.0),
        "laugh_rate": max(float(row[1] or 0.01), 0.01),
        "emoji_rate": float(row[2] or 0.0),
        "text_messages": int(row[3] or 0),
    }


def funny_moments(conn, limit: int = 5) -> list[dict]:
    """Conversations far denser in laughter than this chat's norm."""
    base = baseline(conn)
    rows = _rows(conn, _SESSION_STATS + """
        SELECT session_id, messages, people, started, minutes, laughs, emojis,
               ROUND(1.0 * laughs / messages, 2) AS laugh_rate,
               ROUND(1.0 * laughs / messages / ?, 1) AS times_normal
        FROM stats
        WHERE messages >= 5
          AND laughs >= 3
          AND 1.0 * laughs / messages > ? * 1.5
        ORDER BY times_normal DESC, laughs DESC
        LIMIT ?
    """, [base["laugh_rate"], base["laugh_rate"], limit])
    return _with_excerpts(conn, rows)


def arguments(conn, limit: int = 5) -> list[dict]:
    """
    Conversations that look like a disagreement.

    Several signals together, because no single one is reliable: heated words,
    few people going back and forth quickly, messages longer than this chat's
    norm, markedly less laughter than usual, and an apology or an unusually
    long silence afterwards. "pagal" is an insult or a term of affection
    depending entirely on who is speaking, so one keyword decides nothing.
    """
    base = baseline(conn)
    rows = _rows(conn, _SESSION_STATS + """
        , scored AS (
            SELECT *,
                   1.0 * laughs / messages          AS laugh_rate,
                   1.0 * heat / messages            AS heat_rate,
                   avg_words / ?                    AS wordiness
            FROM stats
        )
        SELECT session_id, messages, people, started, minutes,
               heat, apologies, laughs, avg_words, longest_message,
               ROUND(heat_rate, 2)  AS heat_rate,
               ROUND(wordiness, 2)  AS wordiness_vs_normal,
               ROUND(heat_rate * 10 + apologies * 1.5 + wordiness
                     - laugh_rate * 12, 2) AS tension
        FROM scored
        WHERE messages >= 5
          AND people BETWEEN 2 AND 5
          AND (heat >= 2 OR apologies >= 2)
          AND laugh_rate < ?
        ORDER BY tension DESC
        LIMIT ?
    """, [base["avg_words"], base["laugh_rate"], limit])
    return _with_excerpts(conn, rows)


def deep_conversations(conn, limit: int = 5) -> list[dict]:
    """
    Long, wordy, unusually serious exchanges.

    Requires laughter well below this chat's norm. Without that the ranking
    fills with the longest sessions, which in most group chats are the ones
    everyone was joking in.
    """
    base = baseline(conn)
    rows = _rows(conn, _SESSION_STATS + """
        , scored AS (
            SELECT *,
                   1.0 * laughs / messages AS laugh_rate,
                   avg_words / ?           AS wordiness
            FROM stats
        )
        SELECT session_id, messages, people, started, minutes,
               avg_words, longest_message, reflective, night_messages,
               ROUND(wordiness, 2) AS wordiness_vs_normal,
               ROUND(laugh_rate, 2) AS laugh_rate,
               ROUND(wordiness * 3 + reflective * 1.5
                     + longest_message / 20.0 - laugh_rate * 15, 2) AS depth
        FROM scored
        WHERE messages >= 8
          AND people <= 5
          AND avg_words > ? * 1.4
          AND laugh_rate < ? * 0.7
        ORDER BY depth DESC
        LIMIT ?
    """, [base["avg_words"], base["avg_words"], base["laugh_rate"], limit])
    return _with_excerpts(conn, rows)


def late_night(conn, limit: int = 5) -> list[dict]:
    """Conversations that happened when everyone should have been asleep."""
    rows = _rows(conn, _SESSION_STATS + """
        SELECT session_id, messages, people, started, minutes,
               night_messages, avg_words, laughs
        FROM stats
        WHERE night_messages >= 5
        ORDER BY night_messages DESC, messages DESC
        LIMIT ?
    """, [limit])
    return _with_excerpts(conn, rows)


def busiest_conversations(conn, limit: int = 5) -> list[dict]:
    """The longest single bursts of talking."""
    rows = _rows(conn, _SESSION_STATS + """
        SELECT session_id, messages, people, started, minutes, avg_words,
               laughs, questions
        FROM stats
        ORDER BY messages DESC
        LIMIT ?
    """, [limit])
    return _with_excerpts(conn, rows)


MOMENT_KINDS = {
    "funny": funny_moments,
    "argument": arguments,
    "deep": deep_conversations,
    "late_night": late_night,
    "busiest": busiest_conversations,
}


def find_moments(conn, kind: str = "funny", limit: int = 5) -> dict:
    finder = MOMENT_KINDS.get(kind)
    if finder is None:
        return {"error": f"Unknown kind {kind!r}. "
                         f"Choose from: {', '.join(sorted(MOMENT_KINDS))}."}
    found = finder(conn, min(limit, 10))
    empty_note = {
        "funny": "No conversation stands out as much funnier than this chat's "
                 "normal level of joking.",
        "argument": "No friction found. Either this group does not fight in "
                    "the chat, or it does so without the words this looks for.",
        "deep": "No conversation is markedly longer and more serious than "
                "this chat's norm.",
        "late_night": "No sustained after-midnight conversations.",
        "busiest": "No conversations found.",
    }
    return {
        "kind": kind,
        "count": len(found),
        "moments": found,
        "empty_reason": None if found else empty_note.get(kind),
        "note": "Ranked by heuristic signals over every conversation in the "
                "archive, not by a model's reading. The excerpt is the "
                "evidence -- judge it yourself.",
    }


# --- superlatives -------------------------------------------------------------

def superlatives(conn) -> dict:
    """The per-person awards people actually want to see."""
    out: dict[str, Any] = {}

    out["night_owl"] = _rows(conn, """
        SELECT sender, COUNT(*) AS messages_after_midnight
        FROM v_messages
        WHERE msg_type <> 'system' AND hour >= 0 AND hour < 5
        GROUP BY 1 ORDER BY 2 DESC LIMIT 3
    """)

    out["early_bird"] = _rows(conn, """
        SELECT sender, COUNT(*) AS messages_before_8am
        FROM v_messages
        WHERE msg_type <> 'system' AND hour >= 5 AND hour < 8
        GROUP BY 1 ORDER BY 2 DESC LIMIT 3
    """)

    out["fastest_replier"] = _rows(conn, """
        SELECT sender, ROUND(MEDIAN(reply_gap_seconds) / 60.0, 1) AS median_minutes,
               COUNT(*) AS replies
        FROM v_messages
        WHERE reply_gap_seconds IS NOT NULL
        GROUP BY 1 HAVING COUNT(*) >= 20
        ORDER BY median_minutes ASC LIMIT 3
    """)

    out["slowest_replier"] = _rows(conn, """
        SELECT sender, ROUND(MEDIAN(reply_gap_seconds) / 60.0, 1) AS median_minutes,
               COUNT(*) AS replies
        FROM v_messages
        WHERE reply_gap_seconds IS NOT NULL
        GROUP BY 1 HAVING COUNT(*) >= 20
        ORDER BY median_minutes DESC LIMIT 3
    """)

    out["biggest_texter"] = _rows(conn, """
        SELECT sender, ROUND(AVG(word_count), 1) AS avg_words,
               MAX(word_count) AS longest
        FROM v_messages
        WHERE msg_type = 'text'
        GROUP BY 1 HAVING COUNT(*) >= 20
        ORDER BY avg_words DESC LIMIT 3
    """)

    out["emoji_lover"] = _rows(conn, """
        SELECT sender, SUM(emoji_count) AS emojis,
               ROUND(1.0 * SUM(emoji_count) / COUNT(*), 2) AS per_message
        FROM v_messages WHERE msg_type <> 'system'
        GROUP BY 1 HAVING COUNT(*) >= 20
        ORDER BY per_message DESC LIMIT 3
    """)

    out["media_sharer"] = _rows(conn, """
        SELECT sender, COUNT(*) AS attachments
        FROM v_messages WHERE msg_type = 'media'
        GROUP BY 1 ORDER BY 2 DESC LIMIT 3
    """)

    out["question_asker"] = _rows(conn, """
        SELECT sender, SUM(CASE WHEN is_question THEN 1 ELSE 0 END) AS questions,
               ROUND(100.0 * SUM(CASE WHEN is_question THEN 1 ELSE 0 END)
                     / COUNT(*), 1) AS pct_of_their_messages
        FROM v_messages WHERE msg_type <> 'system'
        GROUP BY 1 HAVING COUNT(*) >= 20
        ORDER BY pct_of_their_messages DESC LIMIT 3
    """)

    out["conversation_starter"] = _rows(conn, """
        SELECT sender, COUNT(*) AS conversations_started
        FROM v_messages
        WHERE is_session_start AND msg_type <> 'system'
        GROUP BY 1 ORDER BY 2 DESC LIMIT 3
    """)

    # A monologue: consecutive messages by one person with nobody interrupting.
    out["longest_monologue"] = _rows(conn, """
        WITH runs AS (
            SELECT sender, session_id, ts,
                   ROW_NUMBER() OVER (ORDER BY ts, msg_id)
                   - ROW_NUMBER() OVER (PARTITION BY sender ORDER BY ts, msg_id)
                       AS run_id
            FROM v_messages WHERE msg_type <> 'system'
        )
        SELECT sender, COUNT(*) AS messages_in_a_row,
               MIN(ts) AS started
        FROM runs
        GROUP BY sender, run_id
        ORDER BY messages_in_a_row DESC
        LIMIT 3
    """)

    out["link_sharer"] = _rows(conn, """
        SELECT sender, COUNT(*) AS links
        FROM v_messages WHERE has_url
        GROUP BY 1 ORDER BY 2 DESC LIMIT 3
    """)

    out["ghosted_most"] = _rows(conn, """
        SELECT COALESCE(p.display_name, m.prev_participant_id) AS sender,
               ROUND(MEDIAN(m.reply_gap_seconds) / 60.0, 1) AS median_wait_minutes,
               COUNT(*) AS times
        FROM messages m
        LEFT JOIN participants p ON p.participant_id = m.prev_participant_id
        WHERE m.reply_gap_seconds IS NOT NULL
        GROUP BY 1 HAVING COUNT(*) >= 20
        ORDER BY median_wait_minutes DESC LIMIT 3
    """)

    return out


def rhythms(conn) -> dict:
    """When this group is alive, and how that changed."""
    return {
        "by_weekday": _rows(conn, """
            SELECT weekday, COUNT(*) AS messages
            FROM v_messages WHERE msg_type <> 'system'
            GROUP BY 1 ORDER BY 1
        """),
        "quietest_days": _rows(conn, """
            SELECT date, COUNT(*) AS messages
            FROM v_messages WHERE msg_type <> 'system'
            GROUP BY 1 HAVING COUNT(*) > 0
            ORDER BY messages ASC LIMIT 5
        """),
        "longest_silences": _rows(conn, """
            SELECT ts AS broken_at, sender,
                   ROUND(gap_seconds / 86400.0, 1) AS days_of_silence
            FROM v_messages
            WHERE gap_seconds IS NOT NULL AND msg_type <> 'system'
            ORDER BY gap_seconds DESC LIMIT 5
        """),
        "busiest_months": _rows(conn, """
            SELECT year_month, COUNT(*) AS messages,
                   COUNT(DISTINCT participant_id) AS active_people
            FROM v_messages WHERE msg_type <> 'system'
            GROUP BY 1 ORDER BY messages DESC LIMIT 5
        """),
    }


def report(conn) -> dict:
    """Everything the Insights screen needs, in one round trip."""
    return {
        "baseline": baseline(conn),
        "superlatives": superlatives(conn),
        "rhythms": rhythms(conn),
        "moments": {
            kind: finder(conn, 3) for kind, finder in MOMENT_KINDS.items()
        },
    }
