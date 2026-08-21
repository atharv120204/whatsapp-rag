"""
Command line interface.

    python -m app.cli archives                      list archives on this device
    python -m app.cli sample                        synthetic chat in a new archive
    python -m app.cli ingest <export.zip> --into X  load an export
    python -m app.cli ask "who talks most?"         one question
    python -m app.cli chat                          interactive session
    python -m app.cli stats                         headline numbers
    python -m app.cli doctor                        audit an archive for problems
    python -m app.cli models                        what your API key can call
    python -m app.cli serve                         run the API server

Commands that read an archive take `--archive <id>`. With exactly one archive
on the device it can be omitted; with several it is required, because guessing
would mean answering about the wrong chat.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import archives as archive_store
from . import settings_store
from .archives import ArchiveNotFound
from .config import settings


def _print_table(rows: list[dict], columns: list[str] | None = None) -> None:
    if not rows:
        print("  (no rows)")
        return
    columns = columns or list(rows[0].keys())
    widths = {
        c: max(len(str(c)), max(len(str(r.get(c, ""))) for r in rows))
        for c in columns
    }
    print("  " + "  ".join(str(c).ljust(widths[c]) for c in columns))
    print("  " + "  ".join("-" * widths[c] for c in columns))
    for r in rows:
        print("  " + "  ".join(str(r.get(c, "")).ljust(widths[c]) for c in columns))


def _resolve_archive(archive_id: str | None):
    try:
        return archive_store.resolve(archive_id)
    except ArchiveNotFound as exc:
        print(f"Error: {exc}", file=sys.stderr)
        existing = archive_store.list_archives()
        if existing:
            print("\nAvailable archives:", file=sys.stderr)
            for a in existing:
                print(f"  {a.archive_id}  {a.name}", file=sys.stderr)
            print("\nPass --archive <id>.", file=sys.stderr)
        else:
            print("Create one with: python -m app.cli sample", file=sys.stderr)
        sys.exit(1)


def _conn(archive_id: str | None):
    from .db import get_connection

    return get_connection(_resolve_archive(archive_id))


# --- archive management --------------------------------------------------------

def cmd_archives(args) -> int:
    existing = archive_store.list_archives()
    if not existing:
        print("No archives yet. Create one with `python -m app.cli sample` or "
              "`python -m app.cli ingest <export.zip>`.")
        return 0

    _print_table(
        [
            {
                "id": a.archive_id,
                "name": a.name,
                "messages": a.stats.get("messages", 0),
                "media": a.stats.get("media", 0),
                "exports": len(a.sources),
                "size_mb": round(a.as_dict()["size_bytes"] / 1048576, 1),
                "updated": a.updated_at[:16].replace("T", " "),
            }
            for a in existing
        ],
        ["id", "name", "messages", "media", "exports", "size_mb", "updated"],
    )
    return 0


def cmd_delete(args) -> int:
    archive = _resolve_archive(args.archive)
    if not args.yes:
        print(f"This permanently deletes '{archive.name}' "
              f"({archive.stats.get('messages', 0)} messages) and all its media.")
        if input("Type the archive name to confirm: ").strip() != archive.name:
            print("Not deleted.")
            return 1
    archive_store.delete_archive(archive.archive_id)
    print(f"Deleted {archive.archive_id}.")
    return 0


# --- ingest ---------------------------------------------------------------------

def cmd_sample(args) -> int:
    from .index.build import ingest
    from .sample_data import generate

    settings.ensure_dirs()
    archive = archive_store.create_archive(args.name)
    archive.ensure_dirs()

    path, truth = generate(archive.raw_dir, days=args.days)
    print(f"Generated {path}")
    print(f"Archive: {archive.archive_id} ({archive.name})\n")

    result = ingest(path, archive, mode="replace",
                    describe_media=args.media, embed=args.embed)
    _report(result)

    print("\nGround truth to check answers against:")
    _print_table([
        {"sender": name, "messages": n,
         "initiations": truth["initiations_by_sender"].get(name, 0)}
        for name, n in sorted(truth["messages_by_sender"].items(),
                              key=lambda kv: -kv[1])
    ])
    return 0 if result.ok else 1


def cmd_ingest(args) -> int:
    from .index.build import ingest

    path = Path(args.path)
    if not path.exists():
        print(f"No such file: {path}", file=sys.stderr)
        return 1

    if args.into:
        archive = _resolve_archive(args.into)
    else:
        name = args.name or path.stem
        archive = archive_store.create_archive(name)
        print(f"Created archive {archive.archive_id} ({archive.name})")

    mode = "merge" if args.merge else "replace"
    if mode == "replace" and archive.stats.get("messages") and not args.yes:
        print(f"'{archive.name}' already holds "
              f"{archive.stats['messages']} messages, which replace mode will "
              f"discard.\nUse --merge to combine instead, or --yes to confirm.")
        return 1

    result = ingest(path, archive, mode=mode,
                    describe_media=args.media, embed=args.embed)
    _report(result)
    return 0 if result.ok else 1


def cmd_embed(args) -> int:
    """
    Backfill vectors for chunks that lack them.

    Separate from ingest because an archive can outlive the key it was ingested
    without: the chunks are already in place, so semantic search only needs the
    vectors filling in. Resumable, and every vector is cached by content hash,
    so a run stopped by the daily cap resumes tomorrow and re-pays for nothing.
    """
    from .index.embed import embed_chunks

    if not settings.has_api_key:
        print("No Gemini API key on this device. Set one with: "
              "python -m app.cli setkey <key>", file=sys.stderr)
        return 1

    conn = _conn(args.archive)
    total = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
    pending = conn.execute("""
        SELECT count(*) FROM chunks c
        LEFT JOIN chunk_vectors v USING (chunk_id)
        WHERE v.chunk_id IS NULL
    """).fetchone()[0]

    if not pending:
        print(f"All {total} chunks already have vectors.")
        return 0

    print(f"{pending} of {total} chunks need vectors "
          f"(batches of {settings.embed_batch_size}).")

    def progress(done: int, of: int) -> None:
        print(f"  {done}/{of}", end="\r", flush=True)

    stats = embed_chunks(conn, on_progress=progress)
    print(f"  {' ' * 20}", end="\r")

    print(f"{stats['embedded']} embedded, {stats['cached']} from cache.")
    if stats.get("quota_reached"):
        print(f"\nStopped early: {stats['quota_message']}")

    covered = conn.execute("SELECT count(*) FROM chunk_vectors").fetchone()[0]
    print(f"Semantic search now covers {covered}/{total} chunks.")
    return 0 if covered else 1


def _report(result) -> None:
    print()
    if result.errors:
        print("ERRORS")
        for e in result.errors:
            print(f"  ! {e}")
    if result.warnings:
        print("WARNINGS")
        for w in result.warnings:
            print(f"  - {w}")

    merge = result.stages.get("merge") or {}
    if result.mode == "merge":
        print(f"\nMerge: {merge.get('added', 0)} new, "
              f"{merge.get('skipped', 0)} already present, "
              f"{merge.get('upgraded', 0)} upgraded with media")

    overview = result.stages.get("overview") or {}
    if overview:
        print(f"\n{overview.get('total_messages', 0)} messages from "
              f"{overview.get('participant_count', 0)} people, "
              f"{str(overview.get('first_message'))[:10]} to "
              f"{str(overview.get('last_message'))[:10]} "
              f"({overview.get('active_days', 0)} active days, "
              f"{overview.get('session_count', 0)} conversations)\n")
        _print_table(overview.get("participants", []),
                     ["name", "messages", "initiations", "media_sent", "avg_words"])
        if overview.get("media"):
            print()
            _print_table(overview["media"], ["kind", "count", "described"])
    print(f"\nFinished in {result.elapsed_seconds:.1f}s")


def cmd_estimate(args) -> int:
    """Size up a media ingest before paying for it."""
    from .index.media_understanding import estimate_cost
    from .parse.media import MediaCatalog, extract_zip, find_chat_txt

    path = Path(args.path)
    settings.ensure_dirs()

    if path.suffix.lower() == ".zip":
        target = settings.data_dir / "_estimate" / path.stem
        print(f"Extracting {path.name} ...")
        extract_zip(path, target)
        root, transcript = target, find_chat_txt(target)
    else:
        root, transcript = path.parent, path

    catalog = MediaCatalog(root, transcript=transcript)
    est = estimate_cost(catalog.all_files())

    print(f"\n{est['file_count']} files, {est['readable_count']} readable by Gemini")
    _print_table(
        [{"kind": k, "count": v["count"], "MB": round(v["bytes"] / 1048576, 1)}
         for k, v in sorted(est["by_kind"].items(), key=lambda kv: -kv[1]["count"])]
    )
    print(f"\nEstimated input tokens: {est['estimated_input_tokens']:,}")
    print(est["note"])
    return 0


# --- querying --------------------------------------------------------------------

def cmd_ask(args) -> int:
    from .agent.router import ask

    answer = ask(args.question, archive=_resolve_archive(args.archive))
    if answer.error:
        print(f"Error: {answer.error}", file=sys.stderr)
        return 1

    for call in answer.tool_calls:
        preview = json.dumps(call.arguments, default=str)
        print(f"  [{call.name}] {preview[:160]} ({call.duration_ms}ms)")
    print()
    print(answer.text)
    return 0


def cmd_chat(args) -> int:
    from .agent.router import ask

    archive = _resolve_archive(args.archive)
    print(f"Asking about '{archive.name}'. Ctrl-C or 'exit' to quit.\n")
    history: list[dict] = []
    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not question or question.lower() in ("exit", "quit"):
            return 0

        answer = ask(question, history, archive=archive)
        if answer.error:
            print(f"  error: {answer.error}\n")
            continue

        if args.verbose:
            for call in answer.tool_calls:
                print(f"  [{call.name}] "
                      f"{json.dumps(call.arguments, default=str)[:140]}")
        print(f"\n{answer.text}\n")
        history.append({"role": "user", "text": question})
        history.append({"role": "model", "text": answer.text})


def cmd_stats(args) -> int:
    from .api import stats

    conn = _conn(args.archive)

    print("\nPER PERSON")
    _print_table(stats.leaderboard(conn),
                 ["sender", "messages", "pct", "initiations", "media_sent",
                  "questions", "avg_words", "median_reply_min"])

    print("\nWHO ENDS CONVERSATIONS")
    _print_table(stats.initiation_analysis(conn)["enders"])

    st = stats.streaks(conn)
    print("\nBUSIEST DAYS")
    _print_table(st["busiest_days"][:5])
    print("\nLONGEST ACTIVE STREAKS")
    _print_table(st["longest_streaks"][:3])

    media = stats.media_breakdown(conn)
    if media["by_kind"]:
        print("\nMEDIA")
        _print_table(media["by_kind"])
    return 0


# --- configuration ----------------------------------------------------------------

def cmd_doctor(args) -> int:
    """Audit an archive for inconsistencies that would make answers wrong."""
    from .doctor import check_archive

    archive = _resolve_archive(args.archive)
    report = check_archive(_conn(args.archive), archive)

    print()
    print(f"Archive: {archive.name} ({archive.archive_id})")
    for key, value in report.stats.items():
        print(f"  {key:20} {value}")
    print()

    if not report.findings:
        print("  No problems found.")
        return 0

    for level in ("error", "warning", "note"):
        group = [f for f in report.findings if f.level == level]
        if not group:
            continue
        print(f"{level.upper()}S")
        for f in group:
            print(f"  [{f.check}] {f.detail}")
        print()

    return 1 if report.errors else 0


def cmd_models(args) -> int:
    from .index.gemini import check_config, list_models

    settings_store.apply_to_settings()
    check = check_config()
    if check.get("error"):
        print(f"Error: {check['error']}", file=sys.stderr)
        return 1

    print(f"{check['model_count']} models visible to this key.\n")
    print("Configured:")
    for entry in check["available"]:
        print(f"  ok       {entry['role']:14} {entry['model']}")
    for entry in check["missing"]:
        print(f"  MISSING  {entry['role']:14} {entry['model']}")

    if args.all:
        print("\nAll models:")
        for m in list_models():
            print(f"  {m['name'].removeprefix('models/')}")
    if check["missing"]:
        print("\nFix the missing ids in .env or the Settings screen.")
        return 1
    return 0


def cmd_setkey(args) -> int:
    settings_store.save({"gemini_api_key": args.key})
    view = settings_store.public_view()
    print(f"Key saved to {settings.data_dir / 'config.json'} ({view['api_key_hint']}).")
    return 0


def cmd_serve(args) -> int:
    import uvicorn

    uvicorn.run("app.main:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def main(argv: list[str] | None = None) -> int:
    settings_store.apply_to_settings()

    parser = argparse.ArgumentParser(prog="app.cli", description="Chat archive")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("archives", help="list archives on this device")
    p.set_defaults(func=cmd_archives)

    p = sub.add_parser("delete", help="permanently delete an archive")
    p.add_argument("--archive", help="archive id")
    p.add_argument("--yes", action="store_true", help="skip confirmation")
    p.set_defaults(func=cmd_delete)

    p = sub.add_parser("sample", help="synthetic chat in a new archive")
    p.add_argument("--name", default="Sample group chat")
    p.add_argument("--days", type=int, default=60)
    p.add_argument("--no-media", dest="media", action="store_false")
    p.add_argument("--no-embed", dest="embed", action="store_false")
    p.set_defaults(func=cmd_sample, media=True, embed=True)

    p = sub.add_parser("ingest", help="load a WhatsApp export")
    p.add_argument("path")
    p.add_argument("--into", help="existing archive id (default: create one)")
    p.add_argument("--name", help="name for a newly created archive")
    p.add_argument("--merge", action="store_true",
                   help="combine with what is already there, skipping duplicates")
    p.add_argument("--yes", action="store_true",
                   help="confirm replacing a non-empty archive")
    p.add_argument("--no-media", dest="media", action="store_false",
                   help="skip describing photos, voice notes and video")
    p.add_argument("--no-embed", dest="embed", action="store_false",
                   help="skip embeddings (keyword search and stats still work)")
    p.set_defaults(func=cmd_ingest, media=True, embed=True)

    p = sub.add_parser("embed", help="build missing semantic search vectors")
    p.add_argument("--archive", help="archive id (default: the only one)")
    p.set_defaults(func=cmd_embed)

    p = sub.add_parser("estimate", help="size up media processing first")
    p.add_argument("path")
    p.set_defaults(func=cmd_estimate)

    p = sub.add_parser("ask", help="ask one question")
    p.add_argument("question")
    p.add_argument("--archive")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("chat", help="interactive session")
    p.add_argument("--archive")
    p.add_argument("-v", "--verbose", action="store_true", help="show tool calls")
    p.set_defaults(func=cmd_chat)

    p = sub.add_parser("stats", help="print headline statistics")
    p.add_argument("--archive")
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("doctor", help="audit an archive for inconsistencies")
    p.add_argument("--archive")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("models", help="check Gemini model configuration")
    p.add_argument("--all", action="store_true")
    p.set_defaults(func=cmd_models)

    p = sub.add_parser("setkey", help="save your Gemini API key on this device")
    p.add_argument("key")
    p.set_defaults(func=cmd_setkey)

    p = sub.add_parser("serve", help="run the API server")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true")
    p.set_defaults(func=cmd_serve)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
