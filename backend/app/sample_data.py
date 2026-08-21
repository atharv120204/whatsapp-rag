"""
Generate a synthetic with-media export for testing and demos.

Deliberately includes the awkward cases that break naive parsers: a participant
who appears first as a phone number and later as a saved name, captioned
photos, multi-line messages, system notices, deleted messages, and a stretch of
silence long enough to split sessions.

The ground truth is printed alongside, so answers from the chatbot can be
checked rather than trusted.
"""

from __future__ import annotations

import random
import shutil
import struct
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

PEOPLE = ["Karan", "Rohit Sharma", "Priya", "Sneha", "+91 98765 43210"]

OPENERS = [
    "guys anyone up?", "morning", "yo", "quick question", "so what's the plan",
    "anyone free this weekend?", "update on the trip?", "hey", "need help",
    "did everyone see the mail?",
]
REPLIES = [
    "yeah", "on it", "haha true", "lol", "give me 5 min", "same here",
    "not sure tbh", "sounds good", "count me in", "can't make it sorry",
    "let me check and revert", "okay done", "why though", "that works",
    "I'll book the tickets then", "sending the screenshot now",
    "we should leave by 6", "budget is around 12k per person",
    "the hotel near the beach looks decent", "exams get over on the 20th",
]
QUESTIONS = [
    "what time?", "who's coming?", "how much?", "can we shift it to sunday?",
    "did you book?", "is the venue confirmed?", "anyone got the notes?",
]

MEDIA_PLAN = [
    ("IMG-20240115-WA0001.jpg", "image", "check out this view"),
    ("IMG-20240116-WA0002.jpg", "image", ""),
    ("PTT-20240116-WA0003.opus", "voice", ""),
    ("VID-20240118-WA0004.mp4", "video", "watch till the end"),
    ("STK-20240119-WA0005.webp", "sticker", ""),
    ("IMG-20240120-WA0006.jpg", "image", "the bill"),
    ("DOC-20240121-WA0007.pdf", "document", "itinerary"),
    ("PTT-20240122-WA0008.opus", "voice", ""),
]


def _tiny_jpeg() -> bytes:
    """A minimal valid JPEG (1x1 grey). Enough for format detection."""
    return bytes.fromhex(
        "ffd8ffe000104a46494600010100000100010000ffdb004300ffffffffffffffff"
        "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
        "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
        "ffffffffffffffffffffffffffffffc00011080001000101011100ffc4001f0000"
        "010501010101010100000000000000000102030405060708090a0bffc400b51000"
        "02010303020403050504040000017d01020300041105122131410613516107"
        "227114328191a1082342b1c11552d1f02433627282090a161718191a2526272829"
        "2a3435363738393a434445464748494a535455565758595a636465666768696a73"
        "7475767778797a838485868788898a92939495969798999aa2a3a4a5a6a7a8a9aa"
        "b2b3b4b5b6b7b8b9bac2c3c4c5c6c7c8c9cad2d3d4d5d6d7d8d9dae1e2e3e4e5e6"
        "e7e8e9eaf1f2f3f4f5f6f7f8f9faffda0008010100003f00bf8000ffd9"
    )


def _tiny_webp() -> bytes:
    body = b"VP8 " + struct.pack("<I", 10) + b"\x00" * 10
    return b"RIFF" + struct.pack("<I", len(body) + 4) + b"WEBP" + body


def _tiny_ogg() -> bytes:
    """An Ogg page header with an OpusHead packet."""
    return (b"OggS\x00\x02" + b"\x00" * 20 + b"\x01\x13"
            + b"OpusHead\x01\x02\x38\x01\x80\xbb\x00\x00\x00\x00\x00")


def _tiny_mp4() -> bytes:
    return (b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom"
            + b"\x00\x00\x00\x08free")


def _tiny_pdf() -> bytes:
    return (b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
            b"2 0 obj<</Type/Pages/Kids[]/Count 0>>endobj\n"
            b"trailer<</Root 1 0 R>>\n%%EOF\n")


_BUILDERS = {
    "image": _tiny_jpeg,
    "sticker": _tiny_webp,
    "voice": _tiny_ogg,
    "video": _tiny_mp4,
    "document": _tiny_pdf,
}


def generate(out_dir: Path, days: int = 60, seed: int = 7) -> tuple[Path, dict]:
    """Write a sample export zip. Returns (zip_path, ground_truth)."""
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / "_build"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    start = datetime(2024, 1, 15, 9, 0)
    lines: list[str] = []
    truth: dict = {"messages_by_sender": {}, "initiations_by_sender": {},
                   "media_by_sender": {}}

    def stamp(dt: datetime) -> str:
        return dt.strftime("%d/%m/%Y, %H:%M")

    lines.append(f"{stamp(start)} - Messages and calls are end-to-end "
                 "encrypted. No one outside of this chat, not even WhatsApp, "
                 "can read or listen to them. Tap to learn more.")
    lines.append(f'{stamp(start)} - Karan created group "Goa Trip 2024"')
    lines.append(f"{stamp(start)} - Karan added Rohit Sharma")

    media_queue = list(MEDIA_PLAN)
    now = start + timedelta(minutes=2)
    prev_ts = None

    for day in range(days):
        day_start = start + timedelta(days=day,
                                      hours=rng.randint(0, 3),
                                      minutes=rng.randint(0, 59))
        if rng.random() < 0.15:      # some days are silent
            continue

        bursts = rng.randint(1, 3)
        # Time must only ever move forward: a real export is ordered, and a
        # burst that starts before the previous one ended would make session
        # boundaries -- and therefore the ground truth -- meaningless.
        now = max(now, day_start)
        for _ in range(bursts):
            now = now + timedelta(hours=rng.randint(1, 6),
                                  minutes=rng.randint(0, 59))
            # Sneha and the unsaved number start conversations less often --
            # a real, checkable skew for the initiation question.
            initiator = rng.choices(
                PEOPLE, weights=[35, 25, 20, 12, 8], k=1
            )[0]
            burst_len = rng.randint(2, 12)

            for i in range(burst_len):
                sender = initiator if i == 0 else rng.choice(PEOPLE)
                now += timedelta(minutes=rng.randint(1, 9))

                is_start = prev_ts is None or (now - prev_ts) > timedelta(hours=4)
                if is_start:
                    truth["initiations_by_sender"][sender] = (
                        truth["initiations_by_sender"].get(sender, 0) + 1)
                prev_ts = now

                if media_queue and rng.random() < 0.05:
                    filename, kind, caption = media_queue.pop(0)
                    blob = _BUILDERS[kind]()
                    (work / filename).write_bytes(blob)
                    body = f"{filename} (file attached)"
                    if caption:
                        body += f"\n{caption}"
                    truth["media_by_sender"][sender] = (
                        truth["media_by_sender"].get(sender, 0) + 1)
                elif rng.random() < 0.08:
                    body = "\n".join([
                        "plan for the day:",
                        "1. breakfast at 8",
                        "2. leave by 9",
                        "3. beach till lunch",
                    ])
                elif i == 0:
                    body = rng.choice(OPENERS)
                elif rng.random() < 0.2:
                    body = rng.choice(QUESTIONS)
                else:
                    body = rng.choice(REPLIES)

                lines.append(f"{stamp(now)} - {sender}: {body}")
                truth["messages_by_sender"][sender] = (
                    truth["messages_by_sender"].get(sender, 0) + 1)

        if day == 20:
            now += timedelta(minutes=5)
            lines.append(f"{stamp(now)} - Sneha: This message was deleted")
            truth["messages_by_sender"]["Sneha"] = (
                truth["messages_by_sender"].get("Sneha", 0) + 1)
        if day == 30:
            now += timedelta(minutes=5)
            lines.append(f"{stamp(now)} - +91 98765 43210 left")

    truth["total_messages"] = sum(truth["messages_by_sender"].values())
    truth["media_files"] = len(MEDIA_PLAN) - len(media_queue)

    chat_txt = work / "_chat.txt"
    chat_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    zip_path = out_dir / "sample_group_chat.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(work.iterdir()):
            zf.write(p, p.name)

    return zip_path, truth


if __name__ == "__main__":
    from .config import settings

    path, truth = generate(settings.data_dir / "sample")
    print(f"Wrote {path}")
    print(f"Ground truth: total={truth['total_messages']} "
          f"media={truth['media_files']}")
    for name, n in sorted(truth["messages_by_sender"].items(),
                          key=lambda kv: -kv[1]):
        print(f"  {name:22} {n:5} messages, "
              f"{truth['initiations_by_sender'].get(name, 0):4} initiations")
