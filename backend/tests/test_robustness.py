"""
Robustness tests: many randomised exports, checked for silent data loss.

The question these answer is "how do I know someone else's export will not
break the way mine did". The honest answer is that hand-written examples cannot
give that assurance, because the examples come from the same assumptions as the
code. So instead of more examples, this runs hundreds of randomised exports
drawn from a catalogue of every variation WhatsApp is known to emit, and
asserts two things every time:

    1. it never raises
    2. nothing is silently lost

The second matters more. A crash is loud and gets fixed. A parser that quietly
drops 8% of messages produces statistics that are simply wrong, and nothing
about the output looks wrong.

Failures print the seed, so any failing combination can be replayed exactly.
"""

import random
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adversarial import (DATE_FORMATS, LINE_SHAPES, TIME_FORMATS,  # noqa: E402
                         describe, generate)
from app.parse.normalize import build_alias_lookup, resolve_participants  # noqa: E402
from app.parse.sessions import enrich  # noqa: E402
from app.parse.whatsapp import parse_export  # noqa: E402

RUNS = int(sys.argv[1]) if len(sys.argv) > 1 else 300


def _check_one(seed: int, force: dict | None = None) -> list[str]:
    """Parse one generated export and return a list of invariant violations."""
    rng = random.Random(seed)
    text, expected = generate(rng, n_messages=rng.randint(5, 80), force=force)
    problems: list[str] = []

    messages, report = parse_export(text)
    context = f"seed={seed} {describe(expected)}"

    # 1. Nothing is lost. Every line that was written must come back.
    if report.parsed_messages != expected.total_messages:
        problems.append(
            f"{context}: parsed {report.parsed_messages} of "
            f"{expected.total_messages} messages "
            f"({expected.total_messages - report.parsed_messages} lost)"
        )

    # 2. No line is left unaccounted for.
    if report.unparsed_lines:
        problems.append(f"{context}: {report.unparsed_lines} unparsed lines")

    # 3. Message kinds are classified consistently.
    if report.system_messages != expected.system_messages:
        problems.append(
            f"{context}: system {report.system_messages} != "
            f"{expected.system_messages}")
    if report.deleted_messages != expected.deleted_messages:
        problems.append(
            f"{context}: deleted {report.deleted_messages} != "
            f"{expected.deleted_messages}")
    if report.media_messages != expected.media_messages:
        problems.append(
            f"{context}: media {report.media_messages} != "
            f"{expected.media_messages}")
    if report.attached_files != expected.attachments:
        problems.append(
            f"{context}: attachments {report.attached_files} != "
            f"{expected.attachments}")

    # 4. Timestamps must be usable: every message needs one.
    if any(m.ts is None for m in messages):
        problems.append(f"{context}: message with no timestamp")

    # 5. The rest of the pipeline must survive whatever the parser produced.
    try:
        real = [m for m in messages if m.msg_type != "system"]
        counts = {}
        for m in real:
            if m.sender:
                counts[m.sender] = counts.get(m.sender, 0) + 1
        participants, _ = resolve_participants(counts)
        enriched = enrich(real, build_alias_lookup(participants), gap_hours=4.0)
    except Exception as exc:  # noqa: BLE001
        problems.append(f"{context}: pipeline raised {type(exc).__name__}: {exc}")
        return problems

    # 6. Enrichment must not drop messages either.
    if len(enriched) != len(real):
        problems.append(
            f"{context}: enrich returned {len(enriched)} of {len(real)}")

    # 7. Sessions must be ordered and contiguous.
    if enriched:
        stamps = [e.ts for e in enriched]
        if stamps != sorted(stamps):
            problems.append(f"{context}: enriched output not time-ordered")
        ids = [e.session_id for e in enriched]
        if ids != sorted(ids):
            problems.append(f"{context}: session ids not monotonic")
        if any(e.is_session_start for e in enriched) and ids[0] != 1:
            problems.append(f"{context}: session ids do not start at 1")

    return problems


def test_fuzz_no_data_loss():
    """Hundreds of randomised exports, none of which may lose a message."""
    failures: list[str] = []
    for seed in range(RUNS):
        try:
            failures.extend(_check_one(seed))
        except Exception as exc:  # noqa: BLE001
            failures.append(f"seed={seed}: raised {type(exc).__name__}: {exc}")

    if failures:
        shown = "\n  ".join(failures[:15])
        raise AssertionError(
            f"{len(failures)} invariant violation(s) across {RUNS} exports:\n"
            f"  {shown}"
        )


def test_every_format_combination_parses():
    """
    Exhaustive sweep of date x time x layout.

    Randomised runs sample this space; this covers all of it, so a format that
    is merely rare cannot slip through unnoticed.
    """
    failures: list[str] = []
    seed = 9000
    for date_fmt, _ in DATE_FORMATS:
        for time_fmt, _ in TIME_FORMATS:
            for shape in LINE_SHAPES:
                seed += 1
                force = {"date_format": date_fmt, "time_format": time_fmt,
                         "shape": shape}
                try:
                    failures.extend(_check_one(seed, force))
                except Exception as exc:  # noqa: BLE001
                    failures.append(
                        f"{date_fmt}/{time_fmt}/{shape}: "
                        f"raised {type(exc).__name__}: {exc}")

    if failures:
        shown = "\n  ".join(failures[:15])
        raise AssertionError(
            f"{len(failures)} format combination(s) failed:\n  {shown}")


def test_degenerate_inputs_do_not_crash():
    """Empty, truncated and malformed files must fail gracefully."""
    for name, text in [
        ("empty", ""),
        ("whitespace", "   \n\n  \n"),
        ("no timestamps", "just some text\nand more text\n"),
        ("header only", "12/08/2023, 21:14 - "),
        ("truncated mid-line", "12/08/2023, 21:1"),
        ("only system", "12/08/2023, 21:14 - Messages and calls are "
                        "end-to-end encrypted. Tap to learn more.\n"),
        ("null bytes", "12/08/2023, 21:14 - A: hi\x00there\n"),
        ("lone surrogate-ish", "12/08/2023, 21:14 - A: ��\n"),
        ("very long line", "12/08/2023, 21:14 - A: " + "x" * 200_000),
        ("crlf only", "\r\n\r\n"),
        ("bom", "﻿12/08/2023, 21:14 - A: hi\n"),
        ("impossible date", "31/02/2023, 21:14 - A: hi\n"),
        ("hour 25", "12/08/2023, 25:14 - A: hi\n"),
    ]:
        try:
            messages, report = parse_export(text)
        except Exception as exc:  # noqa: BLE001
            raise AssertionError(
                f"{name!r} raised {type(exc).__name__}: {exc}") from exc

        # Whatever it decides, it must be self-consistent.
        assert report.parsed_messages == len(messages), name


def test_message_containing_a_fake_header_is_reported_not_hidden():
    """
    A message whose text looks like a header is genuinely ambiguous.

    WhatsApp escapes nothing, so "12/08/2023, 21:14 - Fake: hi" typed inside a
    message is byte-identical to a real one. Nothing can resolve that. What
    matters is that the count stays self-consistent rather than the extra line
    vanishing.
    """
    text = (
        "12/08/2023, 21:14 - Karan: look what someone sent\n"
        "12/08/2023, 21:15 - Impostor: this line was typed by a user\n"
        "12/08/2023, 21:16 - Priya: ok\n"
    )
    messages, report = parse_export(text)
    assert report.parsed_messages == len(messages) == 3


def test_extraction_survives_awkward_archives():
    """Zips whose entry names carry the awkward properties real exports have."""
    from app.parse.media import extract_zip, find_chat_txt

    names = [
        "WhatsApp Chat with \U0001F346\U0001F351J.txt",   # emoji
        "WhatsApp Chat with Mum .txt",                     # trailing space
        "DOC-20251030-WA0005.",                            # no extension
        "sub dir/IMG-1.jpg",                               # nested
        "गीत.opus",                         # non-latin
        "x" * 120 + ".jpg",                                # long name
    ]

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        zip_path = tmp_path / "awkward.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            for n in names:
                zf.writestr(n, b"12/08/2023, 21:14 - A: hi\n")

        dest = tmp_path / "out"
        extract_zip(zip_path, dest)

        written = [p for p in dest.rglob("*") if p.is_file()]
        assert len(written) == len(names), (
            f"extracted {len(written)} of {len(names)} entries")
        assert find_chat_txt(dest) is not None


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
    print(f"\n{len(tests) - failed}/{len(tests)} passed  ({RUNS} fuzz runs)")
    sys.exit(1 if failed else 0)
