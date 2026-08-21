"""
Embedding and vector search.

Embeddings are cached by a hash of the chunk text, in a database shared by
every archive on the device. That matters most on a merge: rebuilding chunks
after adding a second export regenerates every window, but only the windows
whose text actually changed are new, so the API is called for a handful of
chunks instead of all of them.
"""

from __future__ import annotations

import hashlib
from typing import Callable

from ..config import settings
from ..db import get_cache_connection
from .gemini import embed_texts
from .ratelimit import DailyQuotaReached


def body_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_cached(hashes: list[str]) -> dict[str, list[float]]:
    if not hashes:
        return {}
    cache = get_cache_connection()
    placeholders = ", ".join("?" * len(hashes))
    rows = cache.execute(
        f"SELECT body_hash, embedding FROM embed_cache "
        f"WHERE model = ? AND dims = ? AND body_hash IN ({placeholders})",
        [settings.embed_model, settings.embed_dims, *hashes],
    ).fetchall()
    return {row[0]: list(row[1]) for row in rows}


def _store_cached(pairs: list[tuple[str, list[float]]]) -> None:
    if not pairs:
        return
    cache = get_cache_connection()
    cache.executemany(
        "INSERT INTO embed_cache (body_hash, model, dims, embedding) "
        "VALUES (?, ?, ?, ?) ON CONFLICT DO NOTHING",
        [(h, settings.embed_model, settings.embed_dims, vec) for h, vec in pairs],
    )


def embed_chunks(conn, on_progress: Callable[[int, int], None] | None = None) -> dict:
    """
    Give every chunk a vector, reusing cached ones.

    Resumable: an interrupted run leaves committed vectors in place and the
    next call fills only the gaps.
    """
    rows = conn.execute("""
        SELECT c.chunk_id, c.body_hash, c.body
        FROM chunks c
        LEFT JOIN chunk_vectors v USING (chunk_id)
        WHERE v.chunk_id IS NULL
        ORDER BY c.chunk_id
    """).fetchall()

    stats = {"total": len(rows), "cached": 0, "embedded": 0}
    if not rows:
        return stats

    cached = _load_cached([r[1] for r in rows])
    to_embed = [(cid, h, body) for cid, h, body in rows if h not in cached]

    # Write the cache hits straight through -- no API call needed.
    hits = [(cid, cached[h]) for cid, h, _ in rows if h in cached]
    if hits:
        conn.executemany(
            "INSERT INTO chunk_vectors VALUES (?, ?) "
            "ON CONFLICT (chunk_id) DO UPDATE SET embedding = excluded.embedding",
            hits,
        )
        stats["cached"] = len(hits)
        if on_progress:
            on_progress(len(hits), len(rows))

    done = len(hits)
    batch = max(1, settings.embed_batch_size)

    for start in range(0, len(to_embed), batch):
        window = to_embed[start:start + batch]
        try:
            vectors = embed_texts([body for _, _, body in window],
                                  task_type="RETRIEVAL_DOCUMENT")
        except DailyQuotaReached as exc:
            # Partial embeddings are still useful, and the next run fills the
            # gaps: embed_chunks only looks at chunks without a vector.
            stats["quota_reached"] = True
            stats["quota_message"] = str(exc)
            break

        conn.executemany(
            "INSERT INTO chunk_vectors VALUES (?, ?) "
            "ON CONFLICT (chunk_id) DO UPDATE SET embedding = excluded.embedding",
            [(cid, vec) for (cid, _, _), vec in zip(window, vectors)],
        )
        _store_cached([(h, vec) for (_, h, _), vec in zip(window, vectors)])

        done += len(window)
        stats["embedded"] += len(window)
        if on_progress:
            on_progress(done, len(rows))

    return stats


def vector_search(conn, query: str, k: int | None = None,
                  participant: str | None = None,
                  after: str | None = None, before: str | None = None) -> list[dict]:
    """
    Semantic search over conversation windows.

    The query is embedded with RETRIEVAL_QUERY rather than RETRIEVAL_DOCUMENT:
    the two task types place text in the same space but from different angles,
    and using the document type for queries measurably hurts recall.
    """
    k = k or settings.top_k

    if not conn.execute("SELECT COUNT(*) FROM chunk_vectors").fetchone()[0]:
        return []

    query_vec = embed_texts([query], task_type="RETRIEVAL_QUERY")[0]

    filters, params = [], [query_vec]
    if participant:
        filters.append("list_contains(c.participants, ?)")
        params.append(participant)
    if after:
        filters.append("c.start_ts >= ?::TIMESTAMP")
        params.append(after)
    if before:
        filters.append("c.end_ts <= ?::TIMESTAMP")
        params.append(before)

    where = ("WHERE " + " AND ".join(filters)) if filters else ""
    params.append(k)

    rows = conn.execute(f"""
        SELECT c.chunk_id, c.start_ts, c.end_ts, c.participants, c.n_messages,
               c.body,
               array_cosine_similarity(
                   v.embedding, ?::FLOAT[{settings.embed_dims}]) AS score
        FROM chunks c
        JOIN chunk_vectors v USING (chunk_id)
        {where}
        ORDER BY score DESC
        LIMIT ?
    """, params).fetchall()

    return [
        {
            "chunk_id": r[0],
            "start_ts": str(r[1]),
            "end_ts": str(r[2]),
            "participants": list(r[3] or []),
            "n_messages": r[4],
            "text": r[5],
            "score": round(float(r[6]), 4),
            "source": "semantic",
        }
        for r in rows
    ]
