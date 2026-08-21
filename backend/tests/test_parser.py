"""Fixture-driven tests for the WhatsApp export parser."""

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.parse.whatsapp import (  # noqa: E402
    detect_date_order,
    emoji_count,
    parse_export,
)

ANDROID_24H = """12/08/2023, 21:14 - Messages and calls are end-to-end encrypted. No one outside of this chat, not even WhatsApp, can read or listen to them. Tap to learn more.
12/08/2023, 21:14 - Karan created group "Goa Trip"
12/08/2023, 21:15 - Karan: hey guys, dates locked?
12/08/2023, 21:16 - Rohit Sharma: yeah 14th to 18th
12/08/2023, 21:16 - Priya: <Media omitted>
13/08/2023, 09:02 - Karan: morning
13/08/2023, 09:05 - Priya: This message was deleted
"""

IOS_12H = """[12/08/2023, 9:14:03 PM] Karan: hey guys
[12/08/2023, 9:15:10 PM] Rohit Sharma: image omitted
[13/08/2023, 12:01:00 AM] Priya: midnight
[13/08/2023, 12:01:00 PM] Priya: noon
"""

MULTILINE = """12/08/2023, 21:15 - Karan: here is the plan
1. reach airport 6am
2. checkin

3. board
12/08/2023, 21:20 - Priya: got it
"""

COLON_IN_TEXT = """12/08/2023, 21:15 - Karan: reminder: bring sunscreen
12/08/2023, 21:16 - Karan changed the subject to: Goa Trip 2023
12/08/2023, 21:17 - Priya: ok
"""

US_ORDER = """08/13/2023, 21:15 - Karan: hey
08/14/2023, 21:16 - Priya: yo
12/25/2023, 10:00 - Karan: merry christmas
"""


def test_android_24h_basic():
    msgs, rep = parse_export(ANDROID_24H)
    assert rep.parsed_messages == 7
    assert rep.system_messages == 2          # encryption notice + group created
    assert rep.media_messages == 1
    assert rep.deleted_messages == 1
    assert rep.senders["Karan"] == 2        # only real messages, not the system line
    assert msgs[2].sender == "Karan"
    assert msgs[2].text == "hey guys, dates locked?"
    assert msgs[2].ts == datetime(2023, 8, 12, 21, 15)


def test_ios_12h_ampm():
    msgs, rep = parse_export(IOS_12H)
    assert rep.parsed_messages == 4
    assert msgs[0].ts == datetime(2023, 8, 12, 21, 14, 3)
    assert msgs[1].msg_type == "media"
    assert msgs[2].ts == datetime(2023, 8, 13, 0, 1)     # 12 AM -> 00:00
    assert msgs[3].ts == datetime(2023, 8, 13, 12, 1)    # 12 PM -> 12:00


def test_multiline_body_is_joined():
    msgs, rep = parse_export(MULTILINE)
    assert rep.parsed_messages == 2
    assert "1. reach airport 6am" in msgs[0].text
    assert "3. board" in msgs[0].text
    assert msgs[1].text == "got it"


def test_colon_in_message_text_is_not_a_sender():
    msgs, rep = parse_export(COLON_IN_TEXT)
    assert msgs[0].sender == "Karan"
    assert msgs[0].text == "reminder: bring sunscreen"
    # The subject-change line must be a system notice, not a sender named
    # "Karan changed the subject to".
    assert msgs[1].sender is None
    assert msgs[1].msg_type == "system"
    assert set(rep.senders) == {"Karan", "Priya"}


def test_date_order_detection():
    assert detect_date_order(ANDROID_24H) == ("DMY", "certain")
    assert detect_date_order(US_ORDER) == ("MDY", "certain")
    assert detect_date_order("05/06/2023, 21:15 - A: x\n") == ("DMY", "assumed")


def test_us_order_parses_as_mdy():
    msgs, _ = parse_export(US_ORDER)
    assert msgs[0].ts == datetime(2023, 8, 13, 21, 15)
    assert msgs[2].ts == datetime(2023, 12, 25, 10, 0)


def test_emoji_count():
    assert emoji_count("hey") == 0
    assert emoji_count("nice \U0001F600 work ❤") == 2


WITH_MEDIA = """12/08/2023, 21:16 - Priya: IMG-20230812-WA0001.jpg (file attached)
look at this view
12/08/2023, 21:17 - Rohit Sharma: PTT-20230812-WA0002.opus (file attached)
12/08/2023, 21:18 - Karan: <attached: 00000042-PHOTO-2023-08-12-21-15-10.jpg>
12/08/2023, 21:19 - Karan: plain text here
12/08/2023, 21:20 - Priya: <Media omitted>
"""


def test_attachments_and_captions():
    msgs, rep = parse_export(WITH_MEDIA)
    assert rep.attached_files == 3
    assert rep.media_messages == 4          # 3 attached + 1 omitted

    photo = msgs[0]
    assert photo.msg_type == "media"
    assert photo.attachment == "IMG-20230812-WA0001.jpg"
    assert photo.text == "look at this view"   # caption survives, filename does not

    voice = msgs[1]
    assert voice.attachment == "PTT-20230812-WA0002.opus"
    assert voice.text == ""

    ios = msgs[2]
    assert ios.attachment == "00000042-PHOTO-2023-08-12-21-15-10.jpg"

    assert msgs[3].msg_type == "text"
    assert msgs[3].attachment is None

    omitted = msgs[4]
    assert omitted.msg_type == "media"
    assert omitted.attachment is None       # without-media export, nothing to load


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
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
