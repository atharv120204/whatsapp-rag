"""
DuckDB schema and connections.

One database file per archive: messages, media, chunks and embedding vectors
for a single chat, with nothing shared. Vectors are a native FLOAT[n] column
queried with array_cosine_similarity, keyword search is DuckDB's BM25 index,
and aggregates are ordinary SQL over the same rows -- so hybrid retrieval and
exact counting happen in one process with no vector service to keep in sync.

Two caches sit *outside* the archives, keyed by content hash:

    media_cache   what Gemini saw or heard in a file
    embed_cache   the vector for a chunk of text

Both are shared across every archive on the device. Describing a photo and
embedding a conversation window are the only expensive operations here, and
both are pure functions of their input, so paying twice for identical bytes is
waste. This is what makes merging a second export affordable: a merge
re-derives every downstream table, but almost nothing has to be recomputed.
"""

from __future__ import annotations

import threading
from pathlib import Path

import duckdb

from .config import settings

_lock = threading.Lock()
_connections: dict[str, duckdb.DuckDBPyConnection] = {}
_cache_conn: duckdb.DuckDBPyConnection | None = None


SCHEMA = """
-- One row per resolved human.
CREATE TABLE IF NOT EXISTS participants (
    participant_id   VARCHAR PRIMARY KEY,
    display_name     VARCHAR NOT NULL,
    aliases          VARCHAR[],
    is_phone_only    BOOLEAN,
    message_count    BIGINT
);

-- One row per message. Derived columns are computed at ingest so that
-- counting questions are plain SQL rather than guesswork.
CREATE TABLE IF NOT EXISTS messages (
    msg_id               BIGINT PRIMARY KEY,
    dedup_key            VARCHAR,        -- identity across overlapping exports
    ts                   TIMESTAMP NOT NULL,
    participant_id       VARCHAR,
    sender_raw           VARCHAR,
    text                 VARCHAR,
    msg_type             VARCHAR,        -- text | media | system | deleted
    attachment           VARCHAR,        -- filename, when media was exported
    source_file          VARCHAR,        -- which export this came from

    char_count           INTEGER,
    word_count           INTEGER,
    emoji_count          INTEGER,
    has_url              BOOLEAN,
    is_question          BOOLEAN,

    date                 DATE,
    hour                 SMALLINT,
    weekday              SMALLINT,       -- 0 = Monday
    year_month           VARCHAR,

    session_id           BIGINT,
    is_session_start     BOOLEAN,
    gap_seconds          DOUBLE,
    prev_participant_id  VARCHAR,
    reply_gap_seconds    DOUBLE
);

-- One row per attachment, including what Gemini saw or heard in it.
CREATE TABLE IF NOT EXISTS media (
    media_id         BIGINT PRIMARY KEY,
    msg_id           BIGINT,
    filename         VARCHAR,
    path             VARCHAR,
    kind             VARCHAR,
    ext              VARCHAR,
    size_bytes       BIGINT,
    content_hash     VARCHAR,
    readable         BOOLEAN,

    description      VARCHAR,
    transcript       VARCHAR,
    ocr_text         VARCHAR,
    detected_objects VARCHAR[],
    status           VARCHAR,           -- pending | done | skipped | error
    error            VARCHAR,
    model_used       VARCHAR,
    processed_at     TIMESTAMP
);

-- Retrieval unit: a window of consecutive messages, not a single message.
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id       BIGINT PRIMARY KEY,
    body_hash      VARCHAR,
    start_msg_id   BIGINT,
    end_msg_id     BIGINT,
    session_id     BIGINT,
    start_ts       TIMESTAMP,
    end_ts         TIMESTAMP,
    participants   VARCHAR[],
    n_messages     INTEGER,
    body           VARCHAR
);

CREATE TABLE IF NOT EXISTS chunk_vectors (
    chunk_id   BIGINT PRIMARY KEY,
    embedding  FLOAT[{dims}]
);

CREATE TABLE IF NOT EXISTS meta (
    key    VARCHAR PRIMARY KEY,
    value  VARCHAR
);
"""

CACHE_SCHEMA = """
CREATE TABLE IF NOT EXISTS media_cache (
    content_hash     VARCHAR PRIMARY KEY,
    kind             VARCHAR,
    description      VARCHAR,
    transcript       VARCHAR,
    ocr_text         VARCHAR,
    detected_objects VARCHAR[],
    model_used       VARCHAR,
    created_at       TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS embed_cache (
    body_hash   VARCHAR,
    model       VARCHAR,
    dims        INTEGER,
    embedding   FLOAT[],
    created_at  TIMESTAMP DEFAULT now(),
    PRIMARY KEY (body_hash, model, dims)
);
"""

INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_messages_ts ON messages(ts)",
    "CREATE INDEX IF NOT EXISTS idx_messages_participant ON messages(participant_id)",
    "CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id)",
    "CREATE INDEX IF NOT EXISTS idx_messages_type ON messages(msg_type)",
    "CREATE INDEX IF NOT EXISTS idx_messages_dedup ON messages(dedup_key)",
    "CREATE INDEX IF NOT EXISTS idx_media_msg ON media(msg_id)",
    "CREATE INDEX IF NOT EXISTS idx_media_hash ON media(content_hash)",
    "CREATE INDEX IF NOT EXISTS idx_chunks_hash ON chunks(body_hash)",
]

# Lets conversation sessions be recomputed at query time with a different
# silence threshold, without re-ingesting.
MACROS = [
    """
    CREATE OR REPLACE MACRO sessions_at(gap_hours) AS TABLE
    SELECT
        msg_id, ts, participant_id, msg_type,
        SUM(CASE WHEN gap_seconds IS NULL
                 OR gap_seconds > gap_hours * 3600 THEN 1 ELSE 0 END)
            OVER (ORDER BY ts, msg_id
                  ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS session_no,
        (gap_seconds IS NULL OR gap_seconds > gap_hours * 3600) AS is_start
    FROM messages
    WHERE msg_type <> 'system'
    """,
]

VIEWS = [
    """
    CREATE OR REPLACE VIEW v_messages AS
    SELECT
        m.*,
        COALESCE(p.display_name, m.sender_raw, '(system)') AS sender
    FROM messages m
    LEFT JOIN participants p USING (participant_id)
    """,
    """
    CREATE OR REPLACE VIEW v_media AS
    SELECT
        md.*,
        m.ts,
        m.date,
        m.participant_id,
        COALESCE(p.display_name, m.sender_raw) AS sender,
        m.text AS caption
    FROM media md
    LEFT JOIN messages m USING (msg_id)
    LEFT JOIN participants p ON p.participant_id = m.participant_id
    """,
    # Everything searchable as one stream: message text plus what was inside
    # each attachment. Without this, media is invisible to search.
    """
    CREATE OR REPLACE VIEW v_searchable AS
    SELECT
        m.msg_id,
        m.ts,
        -- Exposed because "what happened on this date" is the most natural
        -- query to write against this view, and its absence cost real tool
        -- calls to discover.
        m.date,
        m.hour,
        m.weekday,
        COALESCE(p.display_name, m.sender_raw, '(system)') AS sender,
        m.msg_type,
        CASE
            WHEN md.media_id IS NULL THEN
                -- A media message with no file (an export made without media)
                -- still needs to read as something, or a day of shared photos
                -- comes back as a column of blank lines.
                CASE WHEN m.msg_type = 'media' AND COALESCE(m.text, '') = ''
                     THEN '[attachment, not included in the export]'
                     WHEN m.msg_type = 'deleted' THEN '[message deleted]'
                     ELSE m.text END
            ELSE TRIM(
                COALESCE(NULLIF(m.text, ''), '') ||
                CASE WHEN md.description IS NOT NULL AND md.description <> ''
                     THEN ' [' || md.kind || ': ' || md.description || ']'
                     -- Not described yet: name the kind so the reader at least
                     -- knows a photo was sent rather than seeing nothing.
                     ELSE ' [' || md.kind || ' sent, not yet described]' END ||
                CASE WHEN md.transcript IS NOT NULL AND md.transcript <> ''
                     THEN ' [said: ' || md.transcript || ']' ELSE '' END ||
                CASE WHEN md.ocr_text IS NOT NULL AND md.ocr_text <> ''
                     THEN ' [text in image: ' || md.ocr_text || ']' ELSE '' END
            )
        END AS content
    FROM messages m
    LEFT JOIN participants p USING (participant_id)
    LEFT JOIN media md USING (msg_id)
    """,
]


# --- connections ---------------------------------------------------------------

def get_connection(archive=None, archive_id: str | None = None):
    """
    Connection for one archive, created and cached on first use.

    DuckDB permits a single writer per file, so handles are pooled by archive
    id rather than opened per request.
    """
    from .archives import resolve

    if archive is None:
        archive = resolve(archive_id)

    # The whole create-and-initialise runs under the lock. Releasing it after
    # the cache check let two threads each open a handle and each run
    # init_schema, whose CREATE OR REPLACE VIEW statements then collided with a
    # catalog write-write conflict -- and left one handle orphaned.
    with _lock:
        existing = _connections.get(archive.archive_id)
        if existing is not None:
            return existing

        return _open_locked(archive)


def _open_locked(archive):
    """Open and initialise one archive. Caller must hold the lock."""
    archive.ensure_dirs()
    try:
        conn = duckdb.connect(str(archive.db_path))
    except duckdb.IOException as exc:
        # DuckDB allows one read-write process per file. Hitting this almost
        # always means the API server is running and the CLI was used at the
        # same time, which is worth saying plainly.
        if "another process" in str(exc) or "being used" in str(exc):
            raise RuntimeError(
                f"Archive '{archive.name}' is open in another process. The API "
                "server and the CLI cannot use the same archive at once -- stop "
                "one and retry."
            ) from exc
        raise

    _install_extensions(conn)
    init_schema(conn)

    # No lock here: the caller already holds it. _lock is not reentrant, so
    # taking it again would deadlock on the first connection ever opened.
    _connections[archive.archive_id] = conn
    return conn


def get_cursor(archive=None, archive_id: str | None = None):
    """
    A per-operation handle onto an archive.

    DuckDB connections are not safe to use from several threads at once, and
    FastAPI serves every endpoint from a threadpool. Sharing one handle meant
    two concurrent reads interleaved and tore each other's result sets apart --
    which surfaced as a column of names being read as integers, not as an
    obvious error. cursor() is DuckDB's supported answer: a separate handle
    over the same database, cheap to create.

    Writers (ingest) keep using get_connection so their work stays on one
    transaction.
    """
    return get_connection(archive, archive_id).cursor()


def close_connection(archive_id: str) -> None:
    """
    Close and forget an archive's handle, so its files can be deleted.

    Checkpoints first: an un-merged write-ahead log leaves a .wal file beside
    the database that Windows may still hold open, which is what makes a
    delete immediately after a write fail.
    """
    with _lock:
        conn = _connections.pop(archive_id, None)
    if conn is None:
        return

    try:
        conn.execute("CHECKPOINT")
    except Exception:  # noqa: BLE001 - nothing to checkpoint is fine
        pass
    try:
        conn.close()
    except Exception:  # noqa: BLE001 - already closed is fine
        pass


def close_all() -> None:
    for archive_id in list(_connections):
        close_connection(archive_id)


def get_cache_connection():
    """Shared, cross-archive cache of media descriptions and embeddings."""
    global _cache_conn
    with _lock:
        if _cache_conn is not None:
            return _cache_conn

    settings.data_dir.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(settings.data_dir / "cache.duckdb"))
    conn.execute(CACHE_SCHEMA)

    with _lock:
        _cache_conn = conn
    return conn


def _install_extensions(conn) -> None:
    try:
        conn.execute("INSTALL fts; LOAD fts;")
    except Exception as exc:  # noqa: BLE001 - offline installs are non-fatal
        print(f"[db] full-text search unavailable ({exc}); keyword search will "
              "fall back to LIKE matching.")


def init_schema(conn) -> None:
    """Create tables, indexes, macros and views. Safe to call repeatedly."""
    conn.execute(SCHEMA.format(dims=settings.embed_dims))
    for stmt in INDEXES:
        conn.execute(stmt)
    for stmt in MACROS + VIEWS:
        conn.execute(stmt)


def bulk_insert(conn, table: str, columns: list[str],
                rows: list[tuple], chunk_size: int = 50_000) -> int:
    """
    Insert many rows via Arrow instead of executemany.

    duckdb's executemany binds parameters one row at a time and, on a build
    without numpy importable, retries a failed optional import per value --
    Python does not cache failed imports, so every attempt rescans sys.path.
    Measured here that was upwards of two minutes for 5,000 rows against
    roughly one second for 20,000 through Arrow. Anything bulk goes this way.
    """
    if not rows:
        return 0

    import pyarrow as pa

    total = 0
    for start in range(0, len(rows), chunk_size):
        window = rows[start:start + chunk_size]
        table_data = {
            name: [row[i] for row in window]
            for i, name in enumerate(columns)
        }
        arrow_table = pa.table(table_data)
        conn.register("_bulk_staging", arrow_table)
        try:
            column_list = ", ".join(f'"{c}"' for c in columns)
            conn.execute(
                f'INSERT INTO {table} ({column_list}) '
                f'SELECT {column_list} FROM _bulk_staging'
            )
        finally:
            conn.unregister("_bulk_staging")
        total += len(window)

    return total


def rebuild_fts(conn) -> bool:
    """
    (Re)build the BM25 index over searchable content.

    Must run after every ingest: DuckDB's FTS index is a snapshot, not a live
    index, so rows added afterwards are invisible to match_bm25 until rebuilt.
    """
    try:
        conn.execute("LOAD fts;")
        conn.execute("DROP TABLE IF EXISTS fts_docs")
        conn.execute("""
            CREATE TABLE fts_docs AS
            SELECT msg_id, content FROM v_searchable
            WHERE content IS NOT NULL AND content <> ''
        """)
        conn.execute(
            "PRAGMA create_fts_index('fts_docs', 'msg_id', 'content', overwrite=1)"
        )
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[db] could not build FTS index: {exc}")
        return False


def set_meta(key: str, value: str, conn) -> None:
    conn.execute(
        "INSERT INTO meta VALUES (?, ?) "
        "ON CONFLICT (key) DO UPDATE SET value = excluded.value",
        [key, value],
    )


def get_meta(key: str, conn, default: str | None = None) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", [key]).fetchone()
    return row[0] if row else default


def clear_archive_data(conn) -> None:
    """
    Empty an archive's tables, keeping the file and the shared caches.

    Used by a replace-mode ingest. The caches live in a separate database, so
    media descriptions and embeddings survive and are reused immediately.
    """
    for table in ("chunk_vectors", "chunks", "media", "messages",
                  "participants", "fts_docs", "meta"):
        conn.execute(f"DROP TABLE IF EXISTS {table}")
    init_schema(conn)
