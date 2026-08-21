"""
Archive registry.

An archive is one chat: its own DuckDB file, its own extracted media, its own
uploaded exports, in its own directory. Nothing is shared between archives
except the content-addressed caches, so deleting an archive is deleting a
directory and there is no way for one chat's messages to leak into another's
answers.

    data/archives/<archive_id>/
        chat.duckdb      messages, media, chunks, vectors
        meta.json        name, timestamps, ingest history
        media/           extracted attachments
        raw/             the export files as uploaded

The id is derived from the name but is not the name: renaming must not move
files, and two archives may legitimately be called "Family".
"""

from __future__ import annotations

import json
import re
import shutil
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .config import settings

_SLUG_STRIP = re.compile(r"[^a-z0-9]+")


@dataclass
class IngestRecord:
    """One export that was loaded into this archive."""

    filename: str
    ingested_at: str
    mode: str                  # replace | merge
    messages_added: int = 0
    messages_skipped: int = 0
    media_added: int = 0

    def as_dict(self) -> dict:
        return {
            "filename": self.filename,
            "ingested_at": self.ingested_at,
            "mode": self.mode,
            "messages_added": self.messages_added,
            "messages_skipped": self.messages_skipped,
            "media_added": self.media_added,
        }


@dataclass
class Archive:
    archive_id: str
    name: str
    created_at: str
    updated_at: str
    root: Path
    sources: list[IngestRecord] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    # --- paths ---------------------------------------------------------
    @property
    def db_path(self) -> Path:
        return self.root / "chat.duckdb"

    @property
    def media_dir(self) -> Path:
        return self.root / "media"

    @property
    def raw_dir(self) -> Path:
        return self.root / "raw"

    @property
    def meta_path(self) -> Path:
        return self.root / "meta.json"

    def ensure_dirs(self) -> None:
        for d in (self.root, self.media_dir, self.raw_dir):
            d.mkdir(parents=True, exist_ok=True)

    # --- persistence ---------------------------------------------------
    def save(self) -> None:
        self.ensure_dirs()
        self.updated_at = datetime.now().isoformat(timespec="seconds")
        self.meta_path.write_text(
            json.dumps(
                {
                    "archive_id": self.archive_id,
                    "name": self.name,
                    "created_at": self.created_at,
                    "updated_at": self.updated_at,
                    "sources": [s.as_dict() for s in self.sources],
                    "stats": self.stats,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def as_dict(self) -> dict:
        return {
            "archive_id": self.archive_id,
            "name": self.name,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "sources": [s.as_dict() for s in self.sources],
            "stats": self.stats,
            "has_data": self.db_path.exists(),
            "size_bytes": _directory_size(self.root),
        }


class ArchiveNotFound(KeyError):
    pass


def _slugify(name: str) -> str:
    slug = _SLUG_STRIP.sub("-", name.strip().lower()).strip("-")
    return slug[:40] or "chat"


def _directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def archives_root() -> Path:
    root = settings.data_dir / "archives"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _load(root: Path) -> Archive | None:
    meta_path = root / "meta.json"
    if not meta_path.exists():
        return None
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    return Archive(
        archive_id=data.get("archive_id", root.name),
        name=data.get("name", root.name),
        created_at=data.get("created_at", ""),
        updated_at=data.get("updated_at", ""),
        root=root,
        sources=[IngestRecord(**s) for s in data.get("sources", [])],
        stats=data.get("stats", {}),
    )


def list_archives() -> list[Archive]:
    """Every archive on this device, newest activity first."""
    found = []
    for child in archives_root().iterdir():
        if not child.is_dir():
            continue
        archive = _load(child)
        if archive is not None:
            found.append(archive)
    return sorted(found, key=lambda a: a.updated_at or a.created_at, reverse=True)


def get_archive(archive_id: str) -> Archive:
    root = archives_root() / archive_id
    archive = _load(root)
    if archive is None:
        raise ArchiveNotFound(archive_id)
    return archive


def resolve(archive_id: str | None) -> Archive:
    """
    Find the archive a request is about.

    With no id given, fall back to the only archive if there is exactly one --
    the common single-chat case should not require the caller to know an id.
    Beyond that, refuse to guess: silently picking one would answer questions
    about the wrong person's chat.
    """
    if archive_id:
        return get_archive(archive_id)

    existing = list_archives()
    if len(existing) == 1:
        return existing[0]
    if not existing:
        raise ArchiveNotFound("no archives exist yet")
    raise ArchiveNotFound(
        f"{len(existing)} archives exist; specify which one"
    )


def create_archive(name: str) -> Archive:
    """Make a new, empty archive. Names may repeat; ids never do."""
    clean_name = name.strip() or "Untitled chat"
    archive_id = f"{_slugify(clean_name)}-{uuid.uuid4().hex[:6]}"
    now = datetime.now().isoformat(timespec="seconds")

    archive = Archive(
        archive_id=archive_id,
        name=clean_name,
        created_at=now,
        updated_at=now,
        root=archives_root() / archive_id,
    )
    archive.save()
    return archive


def rename_archive(archive_id: str, name: str) -> Archive:
    archive = get_archive(archive_id)
    archive.name = name.strip() or archive.name
    archive.save()
    return archive


def delete_archive(archive_id: str) -> None:
    """
    Remove an archive and everything in it.

    Closes the database handle first, then retries the removal. On Windows a
    file handle is not always released the instant the connection closes, so a
    single rmtree straight after a write fails intermittently -- which showed
    up as a delete that silently did nothing and worked on the second click.
    A few short retries make it deterministic.
    """
    from .db import close_connection

    archive = get_archive(archive_id)
    close_connection(archive.archive_id)

    last_error: Exception | None = None
    for attempt in range(5):
        shutil.rmtree(archive.root, ignore_errors=True)
        if not archive.root.exists():
            return
        last_error = OSError(f"{archive.root} still present")
        time.sleep(0.2 * (attempt + 1))

    raise OSError(
        f"Could not delete {archive.root}. Something still holds a file open "
        f"-- close any program using it and try again. ({last_error})"
    )
