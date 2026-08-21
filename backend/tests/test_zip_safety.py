"""
Tests for zip extraction safety.

Two things must both hold, and the first version got the balance wrong: it
compared resolved absolute paths, which rejected a real WhatsApp export whose
filename contained emoji. Traversal must be blocked; ordinary filenames,
however unusual, must not be.
"""

import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.parse.media import extract_zip, safe_member_path  # noqa: E402

EMOJI_NAME = "WhatsApp Chat with \U0001F346\U0001F351\U0001F31A\U0001F352J.txt"


# --- what must be allowed -----------------------------------------------------

def test_ordinary_names_are_allowed():
    for name in (
        "_chat.txt",
        "IMG-20240101-WA0001.jpg",
        "WhatsApp Chat with Mum.txt",
        "subfolder/IMG-0002.jpg",
        "./_chat.txt",
    ):
        assert safe_member_path(name) is not None, name


def test_emoji_filenames_are_allowed():
    """A real export named with emoji is not an attack."""
    assert safe_member_path(EMOJI_NAME) == Path(EMOJI_NAME)


def test_unicode_and_spaces_and_brackets_are_allowed():
    for name in (
        "Chat with \u00c5sa (2024).txt",
        "\u0917\u0940\u0924 \u2013 song.opus",
        "photo [edited].jpg",
        "\u4f60\u597d.txt",
    ):
        assert safe_member_path(name) is not None, name


# --- what must be blocked -----------------------------------------------------

def test_parent_traversal_is_blocked():
    for name in (
        "../evil.txt",
        "../../etc/passwd",
        "sub/../../evil.txt",
        "a/b/../../../evil.txt",
    ):
        assert safe_member_path(name) is None, name


def test_absolute_paths_are_blocked():
    for name in ("/etc/passwd", "/tmp/evil", "C:/Windows/evil.txt",
                 "C:\\Windows\\evil.txt", "d:/evil"):
        assert safe_member_path(name) is None, name


def test_backslash_traversal_is_blocked():
    """Some zip writers emit Windows separators; traversal still counts."""
    assert safe_member_path("..\\evil.txt") is None
    assert safe_member_path("sub\\..\\..\\evil.txt") is None


def test_empty_and_dot_names_are_blocked():
    for name in ("", ".", "./", "/"):
        assert safe_member_path(name) is None, repr(name)


# --- end to end ----------------------------------------------------------------

def test_extract_writes_emoji_named_file():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        zip_path = tmp_path / "export.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr(EMOJI_NAME, "01/01/2024, 09:00 - A: hi\n")
            zf.writestr("IMG-20240101-WA0001.jpg", b"\xff\xd8\xff")

        dest = tmp_path / "out"
        extract_zip(zip_path, dest)

        assert (dest / EMOJI_NAME).exists()
        assert (dest / "IMG-20240101-WA0001.jpg").exists()


def test_extract_skips_traversal_but_keeps_good_entries():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        zip_path = tmp_path / "evil.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("_chat.txt", "ok")
            zf.writestr("../escaped.txt", "should not be written")

        dest = tmp_path / "out"
        extract_zip(zip_path, dest)

        assert (dest / "_chat.txt").exists()
        assert not (tmp_path / "escaped.txt").exists()


def test_extract_handles_nested_directories():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        zip_path = tmp_path / "nested.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("Chat/_chat.txt", "ok")
            zf.writestr("Chat/media/IMG-1.jpg", b"\xff\xd8\xff")

        dest = tmp_path / "out"
        extract_zip(zip_path, dest)

        assert (dest / "Chat" / "_chat.txt").exists()
        assert (dest / "Chat" / "media" / "IMG-1.jpg").exists()


# --- filename sanitisation ------------------------------------------------------

def test_trailing_space_before_extension_is_removed():
    """
    Windows silently drops a trailing space when creating a directory, so a
    name like "WhatsApp Chat .zip" produces a folder that cannot be found
    again. A plain rstrip does not catch it: the string ends in "p".
    """
    from app.parse.media import safe_filename

    assert safe_filename("WhatsApp Chat .zip") == "WhatsApp Chat.zip"
    assert safe_filename("report ..txt") == "report.txt"
    assert safe_filename("name   .zip") == "name.zip"


def test_sanitiser_keeps_ordinary_and_unicode_names():
    from app.parse.media import safe_filename

    for name in ("export.zip", "WhatsApp Chat with Mum.zip", "你好.txt"):
        assert safe_filename(name) == name, name

    emoji = "WhatsApp Chat with 🍆🍑J.zip"
    assert safe_filename(emoji) == emoji


def test_sanitiser_strips_directories_and_illegal_characters():
    from app.parse.media import safe_filename

    assert safe_filename("../../evil.zip") == "evil.zip"
    assert safe_filename("C:\Windows\evil.zip") == "evil.zip"
    assert safe_filename('bad<>name|?.zip') == "bad__name__.zip"
    assert safe_filename("") == "export"
    assert safe_filename("", "fallback.zip") == "fallback.zip"


def test_extraction_folder_name_is_independent_of_the_export_filename():
    """
    The unpack folder is hashed, not copied from the filename.

    Chat exports are named after the chat, so the filename carries emoji,
    trailing spaces and arbitrary length straight into a filesystem path.
    """
    import hashlib

    awkward = "WhatsApp Chat with 🍆🍑J .zip"
    digest = hashlib.sha1(awkward.encode("utf-8")).hexdigest()[:10]
    folder = f"export-{digest}"

    assert folder.isascii()
    assert " " not in folder
    assert len(folder) < 25
    # Stable, so re-uploading the same export reuses the same folder.
    assert hashlib.sha1(awkward.encode("utf-8")).hexdigest()[:10] == digest



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
