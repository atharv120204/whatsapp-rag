"""
Media catalogue for a with-media WhatsApp export.

An export-with-media is a .zip holding one _chat.txt plus every attachment,
named like IMG-20230812-WA0001.jpg or PTT-20230812-WA0002.opus. This module
unpacks it, indexes the files, classifies each one, and matches them back to
the messages that reference them.

Matching is looser than exact-path equality on purpose: iOS nests attachments
in subfolders, Android sometimes differs in case, and re-zipping by hand can
add a top-level directory. Missing a match means silently losing a photo from
the archive, so we fall back through several strategies and report whatever is
still unmatched instead of dropping it quietly.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

MediaKind = str  # "image" | "video" | "audio" | "voice" | "sticker" | "document" | "other"

_EXT_KIND: dict[str, MediaKind] = {
    # images
    ".jpg": "image", ".jpeg": "image", ".png": "image", ".heic": "image",
    ".heif": "image", ".bmp": "image", ".gif": "image",
    # stickers / webp (WhatsApp stickers are webp; refined by prefix below)
    ".webp": "image",
    # video
    ".mp4": "video", ".mov": "video", ".3gp": "video", ".3gpp": "video",
    ".mkv": "video", ".avi": "video", ".webm": "video", ".wmv": "video",
    # audio
    ".opus": "audio", ".ogg": "audio", ".m4a": "audio", ".mp3": "audio",
    ".aac": "audio", ".wav": "audio", ".amr": "audio", ".flac": "audio",
    # documents
    ".pdf": "document", ".doc": "document", ".docx": "document",
    ".xls": "document", ".xlsx": "document", ".ppt": "document",
    ".pptx": "document", ".txt": "document", ".csv": "document",
    ".rtf": "document",
    # contact cards
    ".vcf": "contact",
}

# WhatsApp encodes intent in the filename prefix, which is more reliable than
# the extension: a .webp is a sticker if it is STK-, and .opus is a voice note
# if it is PTT- (push to talk) rather than a shared music file.
_PREFIX_KIND: dict[str, MediaKind] = {
    "IMG-": "image",
    "VID-": "video",
    "AUD-": "audio",
    "PTT-": "voice",
    "STK-": "sticker",
    "DOC-": "document",
    "GIF-": "video",
}

# iOS names files by role too: 00000042-PHOTO-..., -VIDEO-, -AUDIO-, -STICKER-
_IOS_ROLE_KIND: dict[str, MediaKind] = {
    "PHOTO": "image",
    "VIDEO": "video",
    "AUDIO": "voice",
    "STICKER": "sticker",
    "DOCUMENT": "document",
}

# Formats Gemini can read directly. Anything else is catalogued but not
# described, and reported as such rather than pretended over.
GEMINI_READABLE = {
    "image": {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif", ".gif"},
    "sticker": {".webp", ".png"},
    "video": {".mp4", ".mov", ".3gp", ".3gpp", ".webm", ".avi", ".wmv", ".mkv"},
    "audio": {".opus", ".ogg", ".m4a", ".mp3", ".aac", ".wav", ".flac"},
    "voice": {".opus", ".ogg", ".m4a", ".mp3", ".aac", ".wav", ".flac"},
    "document": {".pdf", ".txt", ".csv"},
    "contact": {".vcf"},
    "other": set(),
}

# Inline request bodies are capped; bigger files go through the Files API.
INLINE_LIMIT_BYTES = 18 * 1024 * 1024


@dataclass
class MediaFile:
    """One attachment on disk."""

    filename: str
    path: Path
    kind: MediaKind
    ext: str
    size_bytes: int
    content_hash: str
    readable_by_gemini: bool
    needs_upload: bool          # too large to inline

    def as_dict(self) -> dict:
        return {
            "filename": self.filename,
            "path": str(self.path),
            "kind": self.kind,
            "ext": self.ext,
            "size_bytes": self.size_bytes,
            "content_hash": self.content_hash,
            "readable_by_gemini": self.readable_by_gemini,
            "needs_upload": self.needs_upload,
        }


@dataclass
class MediaReport:
    files_found: int = 0
    matched: int = 0
    unmatched_references: list[str] = field(default_factory=list)
    orphan_files: list[str] = field(default_factory=list)
    by_kind: dict[str, int] = field(default_factory=dict)
    unreadable: dict[str, int] = field(default_factory=dict)
    total_bytes: int = 0
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "files_found": self.files_found,
            "matched": self.matched,
            "unmatched_references": self.unmatched_references[:50],
            "unmatched_reference_count": len(self.unmatched_references),
            "orphan_files": self.orphan_files[:50],
            "orphan_file_count": len(self.orphan_files),
            "by_kind": self.by_kind,
            "unreadable": self.unreadable,
            "total_bytes": self.total_bytes,
            "warnings": self.warnings,
        }


def classify(filename: str) -> MediaKind:
    """Work out what an attachment is, preferring WhatsApp's naming convention."""
    name = Path(filename).name
    upper = name.upper()

    for prefix, kind in _PREFIX_KIND.items():
        if upper.startswith(prefix):
            # A .webp named IMG- is still a sticker if WhatsApp says so, but a
            # plain photo shared as webp is an image. Trust the prefix.
            if kind == "image" and Path(name).suffix.lower() == ".webp":
                return "sticker"
            return kind

    for role, kind in _IOS_ROLE_KIND.items():
        if f"-{role}-" in upper:
            return kind

    return _EXT_KIND.get(Path(name).suffix.lower(), "other")


def content_hash(path: Path) -> str:
    """
    Fingerprint a file for caching.

    Hashes size plus the head and tail rather than every byte: a full hash of a
    multi-gigabyte media folder costs minutes on every ingest, and for real
    photos and videos a size + 2 MiB collision does not happen in practice.
    """
    h = hashlib.sha256()
    size = path.stat().st_size
    h.update(str(size).encode())
    window = 1024 * 1024
    with path.open("rb") as fh:
        h.update(fh.read(window))
        if size > window * 2:
            fh.seek(-window, 2)
            h.update(fh.read(window))
    return h.hexdigest()


_DRIVE_PREFIX = re.compile(r"^[A-Za-z]:")


def safe_member_path(name: str) -> Path | None:
    """
    Turn a zip entry name into a path safely under the destination.

    Returns None if the entry tries to escape. The check is purely lexical, on
    the archive-relative name, which is the property that actually matters: an
    entry may not be absolute and may not climb out with "..".

    This deliberately does not resolve against the filesystem. Comparing
    resolved absolute paths sounds stricter but is fragile -- a non-existent
    target, a OneDrive reparse point, or a long-path prefix can each make two
    equivalent paths compare unequal, which rejects perfectly ordinary files.
    An export whose name contains emoji is not an attack.
    """
    # The zip spec mandates forward slashes, but some writers emit backslashes.
    cleaned = name.replace("\\", "/")

    if cleaned.startswith("/") or _DRIVE_PREFIX.match(cleaned):
        return None

    parts = [p for p in cleaned.split("/") if p not in ("", ".")]
    if not parts or any(p == ".." for p in parts):
        return None

    return Path(*parts)


# Characters Windows forbids in a filename, plus control codes.
_UNSAFE_FILENAME = re.compile(r'[<>:"|?*]|[\x00-\x1f]')


def safe_filename(name: str, fallback: str = "export") -> str:
    """
    Make a browser-supplied filename safe to use as a path component.

    Multipart encodes filenames inconsistently, so this can arrive with lone
    surrogates from a bad decode, characters Windows rejects outright, or
    trailing dots and spaces that Windows silently strips. Any of those turn a
    later path operation into a confusing failure a long way from the cause.
    Unicode is preserved -- emoji in a chat name are perfectly legitimate.
    """
    base = Path(name.replace("\\", "/")).name

    # Drop anything that survived a broken decode.
    base = base.encode("utf-8", "replace").decode("utf-8", "replace")
    base = _UNSAFE_FILENAME.sub("_", base)
    base = base.strip()

    # Strip trailing dots and spaces from the *stem*, not just the whole name.
    # "WhatsApp Chat .zip" keeps its trailing space under a plain rstrip because
    # the string ends in "p" -- and Windows then silently drops that space when
    # creating a directory, so every subsequent write targets a path that does
    # not exist. That failure surfaces far from its cause, as a bare Errno 2.
    stem, dot, suffix = base.rpartition(".")
    if dot and stem:
        stem = stem.rstrip(". ")
        base = f"{stem}.{suffix}" if stem else suffix
    else:
        base = base.rstrip(". ")

    if not base or base in (".", ".."):
        return fallback
    return base[:150]


def extract_zip(zip_path: Path, dest: Path) -> Path:
    """
    Unpack an export zip, guarding against path traversal.

    A zip is untrusted input even when it came from your own phone; entries
    named ../../x would otherwise write outside the destination.
    """
    dest.mkdir(parents=True, exist_ok=True)
    skipped: list[str] = []

    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue

            relative = safe_member_path(member.filename)
            if relative is None:
                skipped.append(member.filename)
                continue

            target = dest / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, target.open("wb") as out:
                shutil.copyfileobj(src, out)

    if skipped:
        # Skip rather than abort: one malformed entry should not make an
        # otherwise good export unusable, but it must not pass silently.
        print(f"[media] skipped {len(skipped)} unsafe archive entr"
              f"{'y' if len(skipped) == 1 else 'ies'}: {skipped[:3]}")
    return dest


def find_chat_txt(root: Path) -> Path | None:
    """
    Locate the transcript inside an extracted export.

    iOS calls it _chat.txt; Android names it after the group, which may collide
    with a .txt someone shared in the chat. Prefer the conventional names, then
    fall back to the largest .txt, which is overwhelmingly the transcript.
    """
    candidates = sorted(root.rglob("*.txt"))
    if not candidates:
        return None

    for c in candidates:
        if c.name.lower() in ("_chat.txt", "chat.txt"):
            return c
    for c in candidates:
        if "whatsapp chat" in c.name.lower():
            return c
    return max(candidates, key=lambda p: p.stat().st_size)


class MediaCatalog:
    """Indexes attachment files and resolves message references to them."""

    def __init__(self, root: Path, transcript: Path | None = None):
        self.root = root
        self._by_name: dict[str, MediaFile] = {}
        self._by_lower: dict[str, MediaFile] = {}
        self.report = MediaReport()
        self._scan(transcript)

    def _scan(self, transcript: Path | None) -> None:
        for path in sorted(self.root.rglob("*")):
            if not path.is_file():
                continue
            if transcript and path.resolve() == transcript.resolve():
                continue
            if path.name.startswith("."):
                continue

            ext = path.suffix.lower()
            kind = classify(path.name)
            size = path.stat().st_size
            readable = ext in GEMINI_READABLE.get(kind, set())

            mf = MediaFile(
                filename=path.name,
                path=path,
                kind=kind,
                ext=ext,
                size_bytes=size,
                content_hash=content_hash(path),
                readable_by_gemini=readable,
                needs_upload=size > INLINE_LIMIT_BYTES,
            )
            self._by_name[path.name] = mf
            self._by_lower[path.name.lower()] = mf

            self.report.files_found += 1
            self.report.total_bytes += size
            self.report.by_kind[kind] = self.report.by_kind.get(kind, 0) + 1
            if not readable:
                self.report.unreadable[ext or "(none)"] = (
                    self.report.unreadable.get(ext or "(none)", 0) + 1
                )

    def lookup(self, reference: str) -> MediaFile | None:
        """Resolve a filename from the transcript to a file on disk."""
        if reference in self._by_name:
            return self._by_name[reference]

        base = Path(reference).name
        if base in self._by_name:
            return self._by_name[base]
        if base.lower() in self._by_lower:
            return self._by_lower[base.lower()]

        # Last resort: ignore the extension. Some exports rewrite .opus to .ogg.
        stem = Path(base).stem.lower()
        for name_lower, mf in self._by_lower.items():
            if Path(name_lower).stem == stem:
                return mf
        return None

    def finalize(self, referenced: set[str]) -> MediaReport:
        """Record which files were never referenced and which refs never resolved."""
        resolved_paths: set[str] = set()
        for ref in referenced:
            mf = self.lookup(ref)
            if mf is None:
                self.report.unmatched_references.append(ref)
            else:
                self.report.matched += 1
                resolved_paths.add(str(mf.path))

        self.report.orphan_files = [
            mf.filename for mf in self._by_name.values()
            if str(mf.path) not in resolved_paths
        ]

        if self.report.unmatched_references:
            self.report.warnings.append(
                f"{len(self.report.unmatched_references)} message(s) reference an "
                "attachment that is not in the export. Re-export the chat with "
                "'Attach media' to include them."
            )
        if self.report.orphan_files:
            self.report.warnings.append(
                f"{len(self.report.orphan_files)} file(s) in the export are not "
                "referenced by any message; they will still be described and "
                "indexed."
            )
        return self.report

    def all_files(self) -> list[MediaFile]:
        return list(self._by_name.values())
