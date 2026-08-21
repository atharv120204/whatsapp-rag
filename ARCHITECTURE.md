# Architecture: the RAG pipeline

This document maps the system onto standard retrieval-augmented generation
terminology, states the parameters actually used, and — more usefully for a
write-up — explains the three places where it deliberately departs from a
textbook pipeline, and what breaks if it does not.

In the literature this design is **agentic RAG with query routing over a hybrid
index**. The routing is the part worth defending.

---

## 1. The problem with textbook RAG here

The canonical pipeline is:

```
documents → chunk → embed → vector store
query → embed → top-k similarity → stuff into prompt → generate
```

Applied to a chat archive it answers a large class of questions *confidently
and wrongly*. Consider "how many messages did Rohit send?" against tens of thousands of
messages:

- retrieval returns the 12 chunks most similar to the query
- the model counts what it can see and reports a number
- the true answer, 1,239, is not derivable from 12 chunks

Nothing about the output signals the error. This is the well-known failure of
similarity retrieval on **aggregation queries** — questions whose answer is a
property of the whole corpus rather than of any passage in it. Local questions
("what did we decide about the trip?") retrieve fine; global ones do not.

The archive's most-asked questions are global: message counts, who initiates
conversations, response times, activity by hour. So the pipeline routes.

---

## 2. Indexing pipeline (offline, once per import)

### 2.1 Loading and parsing — `parse/whatsapp.py`

A WhatsApp export is semi-structured text, not a document. The parser handles
the format variance that actually occurs: Android vs iOS line shapes, 12h/24h
times, four date orders, two-digit years, CRLF, and the invisible LRM and
narrow-no-break-space characters WhatsApp injects.

Day/month order is **inferred from the corpus**, not assumed: a value above 12
in the first position proves D/M/Y, in the second proves M/D/Y. If the inferred
order produces invalid dates, the alternative is tried. Getting this wrong
silently corrupts every temporal answer.

### 2.2 Preprocessing and enrichment

Two steps that a generic RAG pipeline has no equivalent for:

**Entity resolution** (`parse/normalize.py`). One person appears as
`+91 98765 43210`, `~Rohit`, and `Rohit Sharma`. Treating these as three people
corrupts every per-person statistic. Phone numbers canonicalise to their last
ten digits; push-name prefixes are stripped; ambiguous merges are *reported*
rather than performed.

**Conversation segmentation** (`parse/sessions.py`). A silence longer than
`SESSION_GAP_HOURS` (default 4) ends a conversation. This yields:

| column | meaning |
| --- | --- |
| `session_id` | which conversation a message belongs to |
| `is_session_start` | this message opened a conversation |
| `gap_seconds` | silence before this message |
| `reply_gap_seconds` | silence before it *when someone else spoke last* — actual response time |

These are computed once at ingest because they are the answer to the most
common questions, and because computing them at query time from retrieved
passages is impossible.

### 2.3 Multimodal extraction — `index/media_understanding.py`, `index/transcribe.py`

Attachments are converted to text *before* indexing, so they enter the same
retrieval stream as messages:

| type | processor | produces |
| --- | --- | --- |
| images, stickers | Gemini vision | description, OCR text, object tags |
| voice notes, audio | Whisper (Groq) | verbatim transcript |
| video | Gemini | description + speech transcript |
| documents | Gemini | summary + extracted text |
| contact cards | local parse | name, phone |

Results are cached by content hash, so re-importing costs nothing and a photo
forwarded five times is one API call.

### 2.4 Chunking — `index/chunk.py`

**Unit: a window of 25 consecutive messages, 5 overlapping, never crossing a
session boundary.**

Two decisions worth defending:

*Why not per-message?* Half of chat is `"yeah"`, `"ok"`, `"lol"` — strings with
no standalone meaning whose embeddings are noise. Meaning lives in the
exchange, not the utterance.

*Why session-bounded?* Splicing the end of Tuesday's argument onto Wednesday's
dinner plan produces a chunk about neither, which then matches queries about
both.

Chunks are rendered as a transcript with speaker names and timestamps inline,
so the retrieved text is self-describing:

```
[Monday, 09 February 2026]
10:08 Rohit Sharma: Ek toh party ke paise nahi diye
10:22 Priya Nair: Happy Birthday Rohit 🎉
```

12,000 messages → **~1,400 chunks**.

### 2.5 Embedding — `index/embed.py`

- Model `gemini-embedding-001`, **768 dimensions**
- Documents embedded with task type `RETRIEVAL_DOCUMENT`, queries with
  `RETRIEVAL_QUERY`. These place text in the same space from different angles;
  using the document type for queries measurably degrades recall.
- Truncated Matryoshka vectors are **re-normalised to unit length**, because the
  API only normalises at full dimensionality and cosine similarity assumes it.
- Cached by `sha256(chunk_text)`, shared across archives.

### 2.6 The index — `db.py`

A single DuckDB file per archive holds all three retrieval structures:

| structure | implementation | serves |
| --- | --- | --- |
| relational | `messages` + derived columns | aggregation queries |
| sparse | DuckDB FTS, BM25 | keyword / exact-string retrieval |
| dense | `FLOAT[768]` column, `array_cosine_similarity` | semantic retrieval |

No separate vector database. For a few thousand chunks a brute-force scan is
sub-millisecond,
and keeping vectors beside the rows removes an entire class of
index-out-of-sync bug. This would need revisiting past roughly 10⁵–10⁶ chunks,
where an HNSW index earns its complexity.

The searchable text is a view, `v_searchable`, that folds media descriptions,
transcripts and OCR into the message stream — so a photo is retrievable by what
is *in* it.

---

## 3. Query pipeline (online, per question)

### 3.1 Routing

The router is **LLM function calling**, not a trained classifier: the model is
given four tools with descriptions and picks. The system prompt states the
governing rule explicitly — anything countable goes to SQL, never to retrieval.

| query type | route |
| --- | --- |
| aggregation, ranking, time patterns | `run_sql` |
| local semantic ("what did we decide") | `search_chat` |
| corpus-level qualitative ("funniest moments") | `find_moments` |
| attachment content | `find_media` |

### 3.2 Route A — text-to-SQL (`run_sql`)

The model writes DuckDB SQL against a documented schema. Because every message
is untrusted text reaching a component that emits SQL, the query is validated
structurally, not by trusting the prompt (`agent/sql_guard.py`):

1. exactly one statement — nothing smuggled after a semicolon
2. must be `SELECT` or `WITH`
3. no write verbs, no filesystem functions (`read_csv`, `glob`), no catalog access
4. an implicit `LIMIT 500`

### 3.3 Route B — hybrid retrieval (`search_chat`)

Dense and sparse retrieval run **in parallel**, then fuse:

```
dense:  cosine(query_embedding, chunk_embedding)  → ranked list
sparse: BM25(query, fts_docs)                     → ranked list
                    ↓
        Reciprocal Rank Fusion
        score(d) = Σ  1 / (60 + rank_i(d))
```

**Why RRF rather than weighted score blending?** Cosine similarity is bounded
in [-1,1]; BM25 is unbounded and corpus-dependent. Normalising them against
each other requires a constant with no principled value. RRF uses only rank, so
the scales never have to be reconciled. The damping constant 60 is the standard
value from Cormack et al. (2009).

**Why both?** They fail differently. Dense retrieval misses exact strings — a
phone number, a booking reference, an unusual name. Sparse misses paraphrase,
which is how people actually refer to past conversations.

Filters (`participant`, `after`, `before`) are applied as SQL predicates before
ranking. A bare date expands to cover the whole day.

### 3.4 Route C — corpus-level scoring (`find_moments`)

For "what were the funniest moments", neither retrieval nor SQL alone works:
the model must survey every conversation, which is too much to send.

So SQL scores **every conversation** on measured signals — laughter density,
conflict markers, wordiness, night activity — and only the top few are passed
to the model with excerpts. Thresholds are **relative to the archive's own
baseline** — its own mean message length and its own laughter rate — because
absolute thresholds do not transfer between groups.

This is a map-reduce: cheap deterministic scoring over everything, expensive
model attention on a shortlist.

### 3.5 Generation

An agent loop, maximum 8 steps. Each iteration: model emits a tool call, the
tool runs, the result is appended, repeat until it answers.

Context is managed between steps: results are trimmed to a character budget,
oldest and largest first, while recent and small results are kept whole. If the
provider still refuses the request as too large, the budget halves and retries.

Every tool call is returned with the answer and displayed, so any stated number
can be traced to the SQL that produced it.

---

## 4. Deviations from textbook RAG, and their justification

| # | Deviation | Why | Cost if removed |
| --- | --- | --- | --- |
| 1 | Query routing to text-to-SQL | Aggregation queries are unanswerable by top-k retrieval | Counts, rankings and time patterns become confident guesses |
| 2 | Conversation-window chunking, session-bounded | Individual chat messages carry no standalone meaning | Embeddings of "ok" and "lol" dominate the index |
| 3 | Precomputed conversational features | Sessions and reply gaps are stream properties, not passage properties | "Who initiates conversations" becomes unanswerable |
| 4 | Multimodal → text before indexing | Otherwise a third of the archive is invisible | Photos and voice notes cannot be retrieved at all |
| 5 | Hybrid + RRF rather than dense-only | Dense retrieval misses exact strings | Names, numbers and references become unfindable |
| 6 | Corpus-level scoring for qualitative questions | Surveying everything exceeds any context window | "Funniest moments" answers from an arbitrary sample |

---

## 5. Honest limitations

Worth stating explicitly in a write-up; examiners ask.

- **No cross-encoder reranker.** RRF fuses two rankings but nothing re-scores
  the fused set. A cross-encoder over the top ~50 would likely improve
  precision, at real latency cost.
- **No query expansion or HyDE.** Queries are embedded verbatim; a vague query
  retrieves vaguely.
- **No evaluation harness.** There is no labelled question set and therefore no
  measured recall@k, MRR or answer accuracy. Correctness is currently
  established by ground-truth comparison on synthetic data and by fuzz-testing
  the ingest for silent data loss — not by retrieval metrics. **This is the
  largest gap.**
- **Brute-force vector scan.** Fine at a few thousand chunks, wrong at 10⁶.
- **Fixed chunk size.** No semantic or adaptive chunking.
- **Routing is prompt-governed.** A model that ignores the instruction can
  still answer a counting question from retrieval. The instruction is
  emphatic, but it is not an architectural guarantee.
- **Timestamps have no timezone.** Local time as exported from the phone.
- **Media descriptions are model output**, not ground truth.

## 6. If you extend it

In rough order of value for a project:

1. **Build an evaluation set** — 30–50 questions with known answers, and report
   retrieval and answer accuracy. This is what turns the project from a
   demonstration into a result.
2. **Add a reranker** and measure the delta against the eval set.
3. **Compare against a baseline** — implement naive dense-only RAG and show,
   with numbers, where it fails on aggregation queries. This makes the routing
   argument empirical rather than asserted, and is the strongest thing you can
   put in a report.
