"""
Audit an ingested archive for internal inconsistency.

Reconciliation at import time catches problems as they happen. This catches
them afterwards, on data that is already sitting there, which matters for two
reasons: an archive may have been built by an older version of the code, and a
merge rebuilds everything so a fault can appear without any new import.

Every check answers "could this make an answer wrong?". Nothing here is style;
each finding is a reason to distrust a number the chatbot would report.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Finding:
    level: str          # error | warning | note
    check: str
    detail: str

    def as_dict(self) -> dict:
        return {"level": self.level, "check": self.check, "detail": self.detail}


@dataclass
class Report:
    findings: list[Finding] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.level == "error"]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.level == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def add(self, level: str, check: str, detail: str) -> None:
        self.findings.append(Finding(level, check, detail))

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "stats": self.stats,
            "findings": [f.as_dict() for f in self.findings],
        }


def _scalar(conn, sql: str, params: list | None = None):
    return conn.execute(sql, params or []).fetchone()[0]


def check_archive(conn, archive=None) -> Report:
    """Run every consistency check against one archive's database."""
    report = Report()

    total = _scalar(conn, "SELECT COUNT(*) FROM messages")
    report.stats["messages"] = total
    if total == 0:
        report.add("note", "empty", "This archive has no messages.")
        return report

    # --- identity ---------------------------------------------------------
    orphaned = _scalar(
        conn,
        "SELECT COUNT(*) FROM messages WHERE participant_id IS NULL "
        "AND msg_type <> 'system'",
    )
    if orphaned:
        report.add("error", "senders",
                   f"{orphaned} messages have no sender, so they are missing "
                   "from every per-person statistic.")

    unknown = _scalar(conn, """
        SELECT COUNT(*) FROM messages m
        LEFT JOIN participants p USING (participant_id)
        WHERE m.participant_id IS NOT NULL AND p.participant_id IS NULL
    """)
    if unknown:
        report.add("error", "senders",
                   f"{unknown} messages reference a participant that does not "
                   "exist in the participants table.")

    stored_counts_wrong = _scalar(conn, """
        SELECT COUNT(*) FROM (
            SELECT p.participant_id, p.message_count AS claimed,
                   COUNT(m.msg_id) AS actual
            FROM participants p
            LEFT JOIN messages m ON m.participant_id = p.participant_id
                                 AND m.msg_type <> 'system'
            GROUP BY 1, 2
            HAVING p.message_count <> COUNT(m.msg_id)
        )
    """)
    if stored_counts_wrong:
        report.add("warning", "counts",
                   f"{stored_counts_wrong} participants have a cached message "
                   "count that disagrees with the messages table.")

    # --- ordering and sessions -------------------------------------------
    out_of_order = _scalar(conn, """
        SELECT COUNT(*) FROM (
            SELECT ts, LAG(ts) OVER (ORDER BY msg_id) AS previous
            FROM messages WHERE msg_type <> 'system'
        ) WHERE previous IS NOT NULL AND ts < previous
    """)
    if out_of_order:
        report.add("error", "ordering",
                   f"{out_of_order} messages are stored out of time order. "
                   "Session boundaries and reply times derived from this are "
                   "unreliable.")

    bad_sessions = _scalar(conn, """
        SELECT COUNT(*) FROM (
            SELECT session_id, MIN(is_session_start::INT) AS has_start
            FROM messages WHERE msg_type <> 'system'
            GROUP BY session_id
            HAVING SUM(is_session_start::INT) <> 1
        )
    """)
    if bad_sessions:
        report.add("error", "sessions",
                   f"{bad_sessions} conversations do not have exactly one "
                   "starting message, so 'who initiates' would be wrong.")

    negative_gaps = _scalar(
        conn, "SELECT COUNT(*) FROM messages WHERE gap_seconds < 0")
    if negative_gaps:
        report.add("error", "timing",
                   f"{negative_gaps} messages have a negative gap to the "
                   "previous message.")

    self_replies = _scalar(conn, """
        SELECT COUNT(*) FROM messages
        WHERE reply_gap_seconds IS NOT NULL
          AND prev_participant_id = participant_id
    """)
    if self_replies:
        report.add("error", "timing",
                   f"{self_replies} messages count as a reply to their own "
                   "sender, which inflates response-time averages.")

    # --- media ------------------------------------------------------------
    named = _scalar(conn,
                    "SELECT COUNT(*) FROM messages WHERE attachment IS NOT NULL")
    media_rows = _scalar(conn, "SELECT COUNT(*) FROM media")
    report.stats["attachments_named"] = named
    report.stats["media_rows"] = media_rows

    if media_rows and named != media_rows:
        report.add("warning", "media",
                   f"{named} messages name an attachment but there are "
                   f"{media_rows} media rows.")

    dangling = _scalar(conn, """
        SELECT COUNT(*) FROM media md
        LEFT JOIN messages m USING (msg_id)
        WHERE m.msg_id IS NULL
    """)
    if dangling:
        report.add("error", "media",
                   f"{dangling} media rows point at a message that no longer "
                   "exists.")

    if archive is not None:
        missing_files = 0
        for (path,) in conn.execute("SELECT path FROM media").fetchall():
            from pathlib import Path

            if not Path(path).exists():
                missing_files += 1
        if missing_files:
            report.add("warning", "media",
                       f"{missing_files} attachments are recorded but the file "
                       "is no longer on disk.")

    # --- retrieval --------------------------------------------------------
    chunks = _scalar(conn, "SELECT COUNT(*) FROM chunks")
    vectors = _scalar(conn, "SELECT COUNT(*) FROM chunk_vectors")
    report.stats["chunks"] = chunks
    report.stats["embeddings"] = vectors

    if chunks == 0:
        report.add("warning", "retrieval",
                   "No retrieval chunks exist, so search will find nothing.")
    elif vectors == 0:
        report.add("note", "retrieval",
                   "No embeddings, so search is keyword-only. Re-ingest with "
                   "embeddings enabled to add semantic search.")
    elif vectors < chunks:
        report.add("note", "retrieval",
                   f"{chunks - vectors} of {chunks} chunks have no embedding, "
                   "probably because a daily API budget was reached. Re-run to "
                   "fill them in.")

    uncovered = _scalar(conn, """
        SELECT COUNT(*) FROM messages m
        WHERE m.msg_type <> 'system'
          AND NOT EXISTS (
            SELECT 1 FROM chunks c
            WHERE m.msg_id BETWEEN c.start_msg_id AND c.end_msg_id
          )
    """)
    if uncovered:
        report.add("warning", "retrieval",
                   f"{uncovered} messages are not inside any chunk, so search "
                   "can never return them.")

    # --- duplicates -------------------------------------------------------
    dupe_keys = _scalar(conn, """
        SELECT COUNT(*) FROM (
            SELECT dedup_key FROM messages
            WHERE dedup_key IS NOT NULL
            GROUP BY dedup_key HAVING COUNT(*) > 1
        )
    """)
    if dupe_keys:
        report.add("error", "duplicates",
                   f"{dupe_keys} deduplication keys appear more than once, "
                   "which means a merge double-counted messages.")

    # --- plausibility -----------------------------------------------------
    from datetime import datetime

    future = _scalar(conn, "SELECT COUNT(*) FROM messages WHERE ts > ?",
                     [datetime.now()])
    if future:
        report.add("warning", "dates",
                   f"{future} messages are dated in the future. The day/month "
                   "order may have been misread.")

    ancient = _scalar(conn,
                      "SELECT COUNT(*) FROM messages WHERE ts < '2009-01-01'")
    if ancient:
        report.add("warning", "dates",
                   f"{ancient} messages predate WhatsApp itself, which suggests "
                   "a date parsing problem.")

    return report
