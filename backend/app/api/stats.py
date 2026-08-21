"""
Precomputed analytics for the dashboard.

These are the questions people ask often enough that they should be one click
rather than one conversation. Each is a plain SQL query so the numbers are
exact and reproducible.
"""

from __future__ import annotations

from typing import Any

_NON_SYSTEM = "msg_type <> 'system'"


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


def leaderboard(conn) -> list[dict]:
    """Per-person totals: the answer to 'who sent how many'."""
    return _rows(conn, f"""
        SELECT
            sender,
            COUNT(*)                                              AS messages,
            ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1)    AS pct,
            SUM(word_count)                                       AS words,
            ROUND(AVG(word_count), 1)                             AS avg_words,
            SUM(CASE WHEN is_session_start THEN 1 ELSE 0 END)     AS initiations,
            SUM(CASE WHEN msg_type = 'media' THEN 1 ELSE 0 END)   AS media_sent,
            SUM(CASE WHEN is_question THEN 1 ELSE 0 END)          AS questions,
            SUM(emoji_count)                                      AS emojis,
            SUM(CASE WHEN has_url THEN 1 ELSE 0 END)              AS links,
            ROUND(AVG(reply_gap_seconds) / 60.0, 1)               AS avg_reply_min,
            ROUND(MEDIAN(reply_gap_seconds) / 60.0, 1)            AS median_reply_min,
            MIN(ts)                                               AS first_seen,
            MAX(ts)                                               AS last_seen
        FROM v_messages
        WHERE {_NON_SYSTEM}
        GROUP BY sender
        ORDER BY messages DESC
    """)


def activity_heatmap(conn) -> list[dict]:
    """Messages by weekday and hour, for a 7x24 grid."""
    return _rows(conn, f"""
        SELECT weekday, hour, COUNT(*) AS messages
        FROM v_messages
        WHERE {_NON_SYSTEM}
        GROUP BY weekday, hour
        ORDER BY weekday, hour
    """)


def timeline(conn, granularity: str = "month") -> list[dict]:
    """Message volume over time, per person."""
    unit = {"day": "day", "week": "week", "month": "month"}.get(granularity, "month")
    return _rows(conn, f"""
        SELECT
            date_trunc('{unit}', ts)::DATE AS period,
            sender,
            COUNT(*) AS messages
        FROM v_messages
        WHERE {_NON_SYSTEM}
        GROUP BY 1, 2
        ORDER BY 1, 2
    """)


def hourly_distribution(conn) -> list[dict]:
    return _rows(conn, f"""
        SELECT hour, sender, COUNT(*) AS messages
        FROM v_messages
        WHERE {_NON_SYSTEM}
        GROUP BY 1, 2 ORDER BY 1, 2
    """)


def initiation_analysis(conn, gap_hours: float | None = None) -> dict:
    """
    Who starts conversations, and who they are answering.

    Also reports who *ends* them -- the person whose message is followed by
    the silence -- which is a different and often more interesting question.
    """
    if gap_hours is None:
        starters = _rows(conn, f"""
            SELECT sender, COUNT(*) AS initiations
            FROM v_messages
            WHERE is_session_start AND {_NON_SYSTEM}
            GROUP BY 1 ORDER BY 2 DESC
        """)
    else:
        starters = _rows(conn, """
            SELECT COALESCE(p.display_name, s.participant_id) AS sender,
                   COUNT(*) AS initiations
            FROM sessions_at(?) s
            LEFT JOIN participants p USING (participant_id)
            WHERE s.is_start
            GROUP BY 1 ORDER BY 2 DESC
        """, [gap_hours])

    enders = _rows(conn, f"""
        WITH ranked AS (
            SELECT sender, session_id,
                   ROW_NUMBER() OVER (PARTITION BY session_id
                                      ORDER BY ts DESC, msg_id DESC) AS rn
            FROM v_messages WHERE {_NON_SYSTEM}
        )
        SELECT sender, COUNT(*) AS conversations_ended
        FROM ranked WHERE rn = 1
        GROUP BY 1 ORDER BY 2 DESC
    """)

    sizes = _rows(conn, f"""
        SELECT session_id, COUNT(*) AS n, MIN(ts) AS started,
               date_diff('minute', MIN(ts), MAX(ts)) AS duration_min
        FROM v_messages WHERE {_NON_SYSTEM}
        GROUP BY 1 ORDER BY n DESC LIMIT 10
    """)

    return {
        "initiators": starters,
        "enders": enders,
        "longest_conversations": sizes,
        "gap_hours": gap_hours,
    }


def response_matrix(conn) -> list[dict]:
    """
    Who replies to whom, and how fast.

    Reveals the sub-groups inside a big chat: pairs who talk mostly to each
    other rather than to the group.
    """
    return _rows(conn, f"""
        SELECT
            COALESCE(prev.display_name, m.prev_participant_id) AS responding_to,
            COALESCE(cur.display_name, m.participant_id)       AS responder,
            COUNT(*)                                           AS replies,
            ROUND(MEDIAN(m.reply_gap_seconds) / 60.0, 1)       AS median_min
        FROM messages m
        LEFT JOIN participants cur  ON cur.participant_id  = m.participant_id
        LEFT JOIN participants prev ON prev.participant_id = m.prev_participant_id
        WHERE m.reply_gap_seconds IS NOT NULL AND m.{_NON_SYSTEM}
        GROUP BY 1, 2
        HAVING COUNT(*) >= 3
        ORDER BY replies DESC
        LIMIT 100
    """)


def media_breakdown(conn) -> dict:
    by_kind = _rows(conn, """
        SELECT kind, COUNT(*) AS count,
               SUM(CASE WHEN status = 'done' THEN 1 ELSE 0 END) AS described,
               SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) AS failed,
               SUM(CASE WHEN status = 'skipped' THEN 1 ELSE 0 END) AS skipped,
               ROUND(SUM(size_bytes) / 1048576.0, 1) AS total_mb
        FROM media GROUP BY 1 ORDER BY count DESC
    """)
    by_sender = _rows(conn, """
        SELECT sender, kind, COUNT(*) AS count
        FROM v_media WHERE sender IS NOT NULL
        GROUP BY 1, 2 ORDER BY count DESC
    """)
    return {"by_kind": by_kind, "by_sender": by_sender}


def word_stats(conn) -> dict:
    top_words = _rows(conn, f"""
        WITH words AS (
            SELECT sender, LOWER(UNNEST(regexp_split_to_array(text, '\\s+'))) AS w
            FROM v_messages
            WHERE {_NON_SYSTEM} AND msg_type = 'text'
        ),
        cleaned AS (
            SELECT sender, regexp_replace(w, '[^a-z0-9'']', '', 'g') AS w
            FROM words
        )
        SELECT w AS word, COUNT(*) AS uses, COUNT(DISTINCT sender) AS used_by
        FROM cleaned
        WHERE LENGTH(w) >= 4
        GROUP BY 1 HAVING COUNT(*) >= 5
        ORDER BY uses DESC LIMIT 60
    """)

    longest = _rows(conn, f"""
        SELECT sender, ts, word_count, LEFT(text, 300) AS preview
        FROM v_messages
        WHERE {_NON_SYSTEM} AND msg_type = 'text'
        ORDER BY word_count DESC LIMIT 10
    """)

    return {"top_words": top_words, "longest_messages": longest}


def streaks(conn) -> dict:
    """Busiest days, and the longest run of consecutive active days."""
    busiest = _rows(conn, f"""
        SELECT date, COUNT(*) AS messages
        FROM v_messages WHERE {_NON_SYSTEM}
        GROUP BY 1 ORDER BY messages DESC LIMIT 10
    """)

    # Gaps-and-islands: consecutive dates share (date - row_number).
    longest = _rows(conn, f"""
        WITH days AS (
            SELECT DISTINCT date FROM v_messages WHERE {_NON_SYSTEM}
        ),
        grouped AS (
            SELECT date,
                   date - (ROW_NUMBER() OVER (ORDER BY date))::INTEGER AS grp
            FROM days
        )
        SELECT MIN(date) AS start_date, MAX(date) AS end_date,
               COUNT(*) AS days
        FROM grouped GROUP BY grp ORDER BY days DESC LIMIT 5
    """)

    quiet = _rows(conn, f"""
        SELECT ts AS resumed_at, sender,
               ROUND(gap_seconds / 86400.0, 1) AS silent_days
        FROM v_messages
        WHERE gap_seconds IS NOT NULL AND {_NON_SYSTEM}
        ORDER BY gap_seconds DESC LIMIT 10
    """)

    return {"busiest_days": busiest, "longest_streaks": longest,
            "longest_silences": quiet}


def emoji_stats(conn) -> list[dict]:
    return _rows(conn, f"""
        SELECT sender, SUM(emoji_count) AS emojis, COUNT(*) AS messages,
               ROUND(1.0 * SUM(emoji_count) / COUNT(*), 3) AS per_message
        FROM v_messages WHERE {_NON_SYSTEM}
        GROUP BY 1 ORDER BY emojis DESC
    """)


def dashboard(conn) -> dict:
    """Everything the dashboard needs, in one round trip."""
    return {
        "leaderboard": leaderboard(conn),
        "heatmap": activity_heatmap(conn),
        "timeline": timeline(conn, "month"),
        "hourly": hourly_distribution(conn),
        "initiation": initiation_analysis(conn),
        "responses": response_matrix(conn),
        "media": media_breakdown(conn),
        "words": word_stats(conn),
        "streaks": streaks(conn),
        "emoji": emoji_stats(conn),
    }
