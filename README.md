# Chat Archive

A self-hosted chatbot for exported WhatsApp chats. Load any number of chats,
switch between them, and ask questions about any of them — including the
photos, voice notes, videos and documents inside.

Each person runs their own copy on their own machine with their own API key.
Nothing is uploaded anywhere.

```
> Who starts conversations most often?
Rohit starts the most, 21 of the 80 conversations in this archive (26%),
just ahead of Karan (20) and Priya (19). The unsaved number +91 98765 43210
starts only 7 despite sending nearly as many messages overall — they reply
much more than they open.

  [run_sql] SELECT sender, COUNT(*) ... WHERE is_session_start ...  (7ms)
```

## Why this is not a plain RAG app

The obvious build — embed every message, retrieve the top 20, ask an LLM — gets
most of these questions wrong. "How many messages did Rohit send?" is a `COUNT`
over the whole table; retrieval of 20 chunks out of 50,000 messages cannot
produce it, and the model will confidently invent a number instead.

So there are two retrieval paths behind one agent:

| Question | Path |
| --- | --- |
| How many messages did each person send? | SQL |
| Who initiates conversations? | SQL |
| When is this group most active? | SQL |
| How fast does everyone reply? | SQL |
| What did we decide about the hotel? | semantic + keyword search |
| What was in that photo of the bill? | media descriptions + OCR |

Gemini function-calling picks the path. Every tool call is shown in the UI, so
when the bot states a number you can read the exact SQL that produced it.

Three things are computed at ingest, not query time, because they are what make
the interesting questions answerable at all:

- **Sessions.** Silence longer than a threshold ends a conversation. The first
  message of the next one is an initiation. Stored as a column, and
  recomputable at any other threshold via a SQL macro.
- **Reply gaps.** Time since the previous message *by someone else* — actual
  response time, not the gap to your own follow-up.
- **Identity resolution.** `+91 98765 43210`, `~Rohit` and `Rohit Sharma` are
  one person, not three. Getting this wrong corrupts every per-person figure.

## Archives

Every chat is a separate archive: its own DuckDB file, its own media, its own
directory under `data/archives/`. Nothing is shared between them, so deleting
one is deleting a directory and there is no way for one chat's messages to
appear in another's answers.

Two content-addressed caches sit *outside* the archives and are shared:
descriptions of media files, and embeddings of text. Both are pure functions of
their input, so paying twice for identical bytes is waste — and it is what
makes merging affordable.

## Adding a second export to a chat

When you load an export into an archive that already has data, you choose:

- **Merge** — deduplicates against what is there. Re-uploading the same export
  changes nothing.
- **Replace** — discards the archive's contents first.

Merge exists because of a specific WhatsApp behaviour: a **with-media** export
is capped at roughly the last 10,000 messages, while a **text-only** export
covers the full history. To get both, you need both files.

That is also the case where naive deduplication fails, because the same photo
appears in the two files completely differently:

```
with media      IMG-20230812-WA0001.jpg (file attached)  + caption
without media   <Media omitted>
```

Hashing the text would give these different keys and duplicate every photo,
voice note and video in the overlap. So attachments are keyed on *being an
attachment at that moment from that person*, not on filename or placeholder
text — and when both versions are present, the one naming a real file wins, so
merging fills in media rather than duplicating it.

Repeated messages are handled too: sending "ok" twice in the same minute is two
messages, and both are matched positionally on re-import rather than collapsed
into one.

Because session boundaries, reply gaps and chunk windows are properties of the
complete ordered stream, a merge rebuilds every derived table rather than
appending. The caches mean almost nothing is recomputed.

Merging an export that shares under 20% of its participants with the archive is
refused: it is almost certainly a different chat, and blending two people's
conversations would quietly ruin every statistic.

## Setup

> **New to this, or setting it up on a fresh machine?** [SETUP.md](SETUP.md) is a
> click-by-click walkthrough that assumes no programming knowledge — installing
> Python and Node, getting the two free API keys, exporting your chat from the
> phone, and what to do when something goes wrong. The short version follows.

```bash
cd whatsapp-rag/backend
py -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt
```

```bash
cd whatsapp-rag/frontend && npm install
```

Run both:

```bash
./.venv/Scripts/python.exe -m app.cli serve
```

```bash
npm run dev --prefix frontend
```

Open http://localhost:5173, then set up a model on the **Settings** tab.
You need a Gemini key for semantic search and media understanding
([get one](https://aistudio.google.com/apikey)); the chat agent can use
Gemini, Groq, xAI, OpenRouter, OpenAI or a local Ollama — see below.

## Choosing a chat provider

Two different jobs need a model, and they have different requirements:

| job | what it needs | provider |
| --- | --- | --- |
| Chat agent — answering, writing SQL | tool calling | **any** of Gemini, Groq, xAI, OpenRouter, OpenAI, local Ollama |
| Semantic search | an embeddings endpoint | **Gemini** |
| Reading photos, voice notes, video | multimodal input | **Gemini** |

The chat agent talks to whatever you pick on the **Settings** tab. Everything
except Gemini goes over the OpenAI chat-completions protocol, so anything
serving `/chat/completions` with tool calling works, including a local Ollama.

Embeddings and media stay on Gemini deliberately: most OpenAI-compatible hosts
serve no embeddings endpoint at all (xAI and Groq included). That is not a
limitation worth fighting — vectors are cheap, and every one is cached by
content hash forever, so they cost almost nothing after the first ingest.

**Why you would move chat off Gemini:** free-tier allowances differ by orders of
magnitude. Gemini's free tier allows 20 requests per day for `gemini-2.5-flash`,
and a single question costs several. Groq's free tier is far more generous.
Moving chat to Groq and leaving embeddings on Gemini means your Gemini quota is
spent only on vectors and media, which is exactly what it is good for.

Pick the provider, paste its key, then press **Test** — it lists the models
your key can actually call, so you are choosing from reality rather than from a
list that went stale. Nothing is hardcoded: base URLs are presets you can
override, and the model is just a string.

A local Ollama needs no key and no network at all. Small models write weaker
SQL, so counting questions may take a retry, but nothing leaves the machine.

### Rate limits

Free tiers cap two different things, and both bite:

- **requests per day** — Gemini's free tier allows 20/day on `gemini-2.5-flash`
- **requests per minute** — 100/minute on `gemini-embedding-001`, counted per
  text, so a batch of 64 chunks nearly trips it on its own
- **tokens per minute** — Groq's free tier allows 8,000 TPM on some models

Those numbers differ by two orders of magnitude between models, so the app's own
`max_requests_per_day` (default 200, editable on Settings) is a stop-loss against
a runaway agent loop rather than a guess at any one model's quota. Whichever real
per-day limit is reached first is read out of the 429 and adopted for the rest of
the day. Set it to 0 to disable the check.

The second is the awkward one. An agent conversation grows with every tool
result, so it is the third or fourth call in a run that trips it, right when
the model is closest to an answer. Three things keep that from becoming a
failure:

**Retries honour what the provider said.** A 429 usually arrives with a
`Retry-After` header or a "please try again in 23.37s" in the message. That is
better information than any backoff we would invent, so it is parsed and used;
blind exponential backoff is only the fallback. Client errors like a bad key
are not retried, because they will not fix themselves.

**The conversation is trimmed to a budget.** Roughly 2,900 tokens are fixed
overhead per request (system prompt, schema, tool declarations). Older tool
results are reduced to their shape — row counts and the SQL that produced them
— while recent ones stay whole, because those are what the model is reasoning
about. The system prompt is never touched; it carries the schema needed to
write correct SQL. Tune with `CONTEXT_BUDGET_CHARS` and
`TOOL_RESULT_MAX_CHARS`.

**Repeated calls are answered from the first result.** A model that gets an
unhelpful result often reruns the identical query several times, spending the
step budget on something that cannot start returning rows. The second identical
call returns the first result with a note saying so, which leaves the budget
for a different approach.

Gemini's daily cap is also learned rather than guessed: the limiter reads the
real number out of the 429 body and adopts it, so a wrong configured value
costs one request instead of a day's worth.

## Getting your chat out of WhatsApp

Open the chat → menu → **More** → **Export chat** → **Attach media**.

Media matters: without it the transcript has `<Media omitted>` where photos and
voice notes were, and a large part of the archive is invisible to search. For a
long chat, export both ways and merge them (see above).

Drop the `.zip` on the **Add chat** tab, or:

```bash
./.venv/Scripts/python.exe -m app.cli ingest ~/Downloads/chat.zip --name "Family"
```

## Try it without your own data

```bash
./.venv/Scripts/python.exe -m app.cli sample
```

Generates a synthetic 60-day group chat with media, ingests it into a new
archive, and prints ground-truth message and initiation counts so you can check
the answers rather than trust them.

## CLI

```bash
python -m app.cli archives                          # list archives
python -m app.cli sample --name "Test"              # synthetic chat
python -m app.cli ingest <export.zip> --name "Work" # into a new archive
python -m app.cli ingest <export.zip> --into <id> --merge
python -m app.cli embed --archive <id>              # backfill missing vectors
python -m app.cli estimate <export.zip>             # media cost before paying it
python -m app.cli ask "who talks most?" --archive <id>
python -m app.cli chat -v --archive <id>
python -m app.cli stats --archive <id>              # no API key needed
python -m app.cli models --all                      # what your key can call
python -m app.cli setkey <key>
python -m app.cli delete --archive <id>
python -m app.cli serve
```

`--archive` can be omitted when there is exactly one archive. With several it
is required: guessing would mean answering about the wrong chat.

**The CLI and the server cannot use the same archive at once.** DuckDB allows
one read-write process per file, so stop `serve` before running CLI commands
against an archive it has open.

## Cost control

Describing media is the only expensive part. It is:

- **cached** by file content hash, across every archive on the device
- **deduplicated**, so a photo forwarded five times is one API call
- **resumable**, so an interrupted run continues where it stopped
- **skippable** — untick both boxes on upload (or `--no-media --no-embed`) for a
  free ingest in seconds, with every statistic still working

Check the size first:

```bash
python -m app.cli estimate ~/Downloads/chat.zip
```

### Deciding in the app, not the terminal

Skipping the expensive parts on upload is the right default, but it leaves work
undone — and work that can only be resumed from a terminal tends to stay undone.
So the **Dashboard**, **Media** and **Insights** tabs carry a *Pending work*
card whenever an archive is missing vectors or descriptions. It states the cost
in API requests before you agree to it, priced in distinct files rather than
media rows, and checked against how much of today's budget is actually left:

> 400 files pending · Stickers (200) · Photos (160) · Video (40)
> **403 API requests · 199 left in today's budget**
> Gemini's free tier allows roughly 20 vision requests a day, so 403 files is on
> the order of 20 days of runs. Untick what you do not need.

Deselect stickers and the estimate falls to 203 requests and the warning to 10
days, because the numbers move with the checkboxes. Confirming runs the job in
the background with a progress bar; running out of daily budget is reported as
the expected ending rather than a failure, and the next run resumes from the
cache. Nothing is charged twice.

## Without an API key

Ingest, all statistics, the dashboard, keyword search, the media gallery and
the message browser work with no key at all. Only the chat agent, semantic
search and media descriptions need one. This is deliberate: the archive should
be useful before you have configured anything.

## How this is tested

Hand-written examples cannot tell you a parser is robust, because the examples
come from the same assumptions as the code. The original test data here was
generated by the same mind that wrote the parser, so it only ever exercised the
happy path -- and every bug found in real use came from a property that
generator never produced: a two-digit US date, emoji in the filename, fifteen
participants, a bot account, an extensionless attachment, "<Media omitted>"
interleaved with real files.

So robustness rests on three things instead, in order of how much they are
worth:

**1. Randomised exports, checked for silent loss.** `tests/adversarial.py`
holds a catalogue of every variation WhatsApp is known to emit -- five date
formats, five time formats, two line layouts, both line endings, invisible
direction marks, phone-number senders, push names, RTL and Devanagari names,
emoji display names, empty bodies, 3000-character bodies, every media
placeholder dialect, thirteen system-notice phrasings. `test_robustness.py`
combines them randomly and asserts, on every single run, that nothing is lost:

    parsed messages == messages written
    no unparsed lines
    every kind classified consistently
    enrichment returns exactly what it was given
    sessions ordered, contiguous, numbered from 1

The second property matters more than "it does not crash". A crash is loud and
gets fixed. A parser that quietly drops 8% of messages produces statistics that
are simply wrong, and nothing about the output looks wrong.

Failures print the seed, so any failing combination replays exactly.

    ./.venv/Scripts/python.exe tests/test_robustness.py 2000

This found two real bugs on its first run, both silent-loss:

- a captioned media placeholder was classified as ordinary text, because the
  anchored pattern was tested against the whole multi-line body
- a participant with a display name longer than five words who sent exactly one
  message was reclassified as a system notice and dropped from the database
  entirely -- they vanished from every count, with nothing reported

The first was affecting a real multi-thousand-message import: 31 messages misfiled.

**2. Reconciliation on every import.** Tests only cover what someone thought to
write down, so `_reconcile()` checks the arithmetic on real data every time an
export is loaded: messages prepared vs messages stored, attachments named vs
attachments linked, and no stored message without a sender. A mismatch is
surfaced as a loud warning rather than a plausible-looking archive.

**3. An auditor you can run afterwards.** `python -m app.cli doctor` re-checks
an existing archive: orphaned senders, cached counts that disagree with the
messages table, messages out of time order, conversations without exactly one
starting message, replies attributed to their own sender, dangling media rows,
missing files, messages outside every chunk, duplicate dedup keys, and dates in
the future or before WhatsApp existed. Each check answers "could this make an
answer wrong?".

None of this makes the parser correct for every export in the world. It does
mean that a format it cannot handle produces a visible complaint rather than a
quietly wrong number -- and that the failure gets encoded as a new axis in the
generator, so it cannot come back.

## Notes and limitations

- **Timestamps have no timezone.** WhatsApp exports local time as it was on the
  exporting phone. Cross-timezone comparisons are not reliable.
- **Day/month order is inferred** from the file and cross-checked by attempting
  both orders. If a chat spans only the first twelve days of each month the
  order is genuinely ambiguous; the parser says so in its warnings.
- **Media descriptions are model output**, not ground truth. Good evidence,
  not testimony.
- **Deleted messages** exist as rows with no content — countable, not readable.
- **System notices** ("X joined using this group's invite link") have no sender.
  They are excluded from storage entirely, so they cannot inflate anyone's
  statistics.
- **Your API key is stored in plain text** in `data/config.json` — the same
  exposure as a `.env` file, appropriate for a local single-user tool. It is
  never sent to the browser.
- **There is no authentication.** Anyone who can reach the port can read every
  archive. Keep it on localhost; do not expose it to a network.
- **The agent writes its own SQL**, guarded to a single read-only `SELECT` with
  filesystem and catalog functions blocked. Worth knowing, because every
  message in an archive is untrusted text reaching a component that emits SQL.

## Layout

```
backend/app/
  archives.py              archive registry: create, list, delete, paths
  settings_store.py        per-device settings (API key) editable from the UI
  db.py                    per-archive schema, shared content caches
  parse/whatsapp.py        export formats, dates, multiline, attachments
  parse/normalize.py       phone/name/push-name identity resolution
  parse/sessions.py        sessions, initiations, reply gaps
  parse/media.py           zip extraction, media catalogue, matching
  index/dedup.py           message identity across overlapping exports
  index/build.py           the ingest pipeline
  index/media_understanding.py  Gemini vision/audio, cached and resumable
  index/chunk.py           conversation-window chunking
  index/embed.py           embeddings, cached by content hash
  agent/tools.py           run_sql, search_chat, get_context, find_media
  agent/sql_guard.py       read-only SQL validation
  agent/router.py          the function-calling loop
  api/stats.py             dashboard analytics
  main.py                  FastAPI
  cli.py                   command line
frontend/src/components/
  Chat.tsx                 chat with live tool traces
  Dashboard.tsx            charts and tables
  Media.tsx                searchable media gallery
  Browse.tsx               paged message browser
  Archives.tsx             manage, rename, delete archives
  Ingest.tsx               upload, target archive, merge/replace
  Settings.tsx             API key, models, thresholds
```

## Tests

```bash
for f in tests/test_*.py; do ./.venv/Scripts/python.exe "$f"; done
```

98 tests plus randomised fuzzing:

| file | covers |
| --- | --- |
| `test_parser.py` | export format variants, dates, multiline, attachments |
| `test_pipeline.py` | identity resolution, sessions, reply gaps, SQL guard |
| `test_merge.py` | deduplication across overlapping exports |
| `test_settings.py` | settings store, lock deadlocks, key handling |
| `test_zip_safety.py` | path traversal, awkward filenames |
| `test_llm_providers.py` | provider abstraction, message translation |
| `test_robustness.py` | randomised exports, silent-loss invariants |

Raise the fuzz count for a longer run: `tests/test_robustness.py 2000`.
