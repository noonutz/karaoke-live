"""Setlist: the persistent pre-render queue (stdlib + numpy only).

A `Setlist` is a small JSON file listing sources waiting for offline vocal
separation, plus the ones already rendered. One `threading.Lock` guards every
mutation, writes go through a temp file + `os.replace` so a crash never leaves
a half-written file, and reads hand out copies so callers cannot mutate the
stored state by accident.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import tempfile
import threading
import time
import urllib.parse
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path

from . import audio_io

#: Allowed `Item.status` values.
STATUSES = ("pending", "rendering", "ready", "failed")

_URL_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")


def default_state_dir() -> Path:
    """Per-OS directory for karaokelive state (queue file, stems, ...)."""
    if sys.platform == "win32":
        base = os.environ.get("APPDATA") or str(Path.home())
        return Path(base) / "karaokelive"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "karaokelive"
    base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / "karaokelive"


def _is_url(source: str) -> bool:
    return bool(_URL_RE.match(source))


def _normalize_source(source: str) -> str:
    """Local paths become absolute; URLs are stored exactly as given."""
    if not source:
        raise ValueError("empty source")
    if _is_url(source):
        return source
    return os.path.abspath(os.path.expanduser(source))


def _default_title(source: str) -> str:
    if _is_url(source):
        name = Path(urllib.parse.urlsplit(source).path).stem
    else:
        name = Path(source).stem
    return name or source


def _stable_id(source: str) -> str:
    """8 hex chars derived from the source, so the id survives re-adds."""
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:8]


@dataclass
class Item:
    """One queued source."""
    id: str  # 8-char hex, stable for a given source
    title: str
    source: str  # absolute path, or a URL exactly as given
    status: str = "pending"  # one of STATUSES
    instrumental: str | None = None  # rendered output path, once ready
    error: str | None = None  # failure reason, once failed
    added_at: float = field(default_factory=time.time)
    duration_s: float | None = None  # probed at add time, None when unknown


class Setlist:
    """A JSON-backed queue of `Item`s; see the module docstring."""

    def __init__(self, path: str | Path | None = None):
        self._lock = threading.Lock()
        self.path = (Path(path) if path is not None
                     else default_state_dir() / "setlist.json")
        self._items: list[Item] = []
        self.load()

    def add(self, source: str, title: str | None = None) -> Item:
        """Queue `source`; a source already queued raises `ValueError`.

        The duration is probed when ffprobe can read the file and is `None`
        otherwise -- probing never raises out of `add`.
        """
        norm = _normalize_source(source)
        with self._lock:
            if any(item.source == norm for item in self._items):
                raise ValueError(f"already queued: {source}")
            item = Item(
                id=_stable_id(norm),
                title=title or _default_title(norm),
                source=norm,
                added_at=time.time(),
                duration_s=self._probe_duration(norm),
            )
            self._items.append(item)
            self._write_locked()
            return replace(item)

    @staticmethod
    def _probe_duration(source: str) -> float | None:
        if _is_url(source) or not os.path.exists(source):
            return None
        try:
            return float(audio_io.probe(source).duration_s)
        except Exception:
            return None

    def remove(self, id: str) -> bool:
        """Drop one item; `False` when the id is not queued."""
        with self._lock:
            for index, item in enumerate(self._items):
                if item.id == id:
                    del self._items[index]
                    self._write_locked()
                    return True
            return False

    def items(self) -> list[Item]:
        """All items, oldest first; copies, so mutating them is harmless."""
        with self._lock:
            return [replace(item) for item in self._items]

    def get(self, id: str) -> Item | None:
        """One item by id, or `None`; a copy."""
        with self._lock:
            for item in self._items:
                if item.id == id:
                    return replace(item)
            return None

    def next_pending(self) -> Item | None:
        """Oldest `pending` item (newest last), or `None` when the queue is dry."""
        with self._lock:
            pending = [item for item in self._items if item.status == "pending"]
            if not pending:
                return None
            return replace(min(pending, key=lambda item: item.added_at))

    def set_status(self, id: str, status: str,
                   instrumental: str | None = None,
                   error: str | None = None) -> Item:
        """Move an item to `status`; unknown ids raise `KeyError`."""
        if status not in STATUSES:
            raise ValueError(f"bad status {status!r} "
                             f"(choose from {', '.join(STATUSES)})")
        with self._lock:
            for index, item in enumerate(self._items):
                if item.id == id:
                    updated = replace(item, status=status,
                                      instrumental=instrumental, error=error)
                    self._items[index] = updated
                    self._write_locked()
                    return replace(updated)
            raise KeyError(f"unknown queue id: {id}")

    def clear(self, status: str | None = None) -> int:
        """Drop every item (or only the `status` ones); returns how many went."""
        if status is not None and status not in STATUSES:
            raise ValueError(f"bad status {status!r} "
                             f"(choose from {', '.join(STATUSES)})")
        with self._lock:
            before = len(self._items)
            if status is None:
                self._items = []
            else:
                self._items = [item for item in self._items
                               if item.status != status]
            self._write_locked()
            return before - len(self._items)

    def save(self) -> None:
        """Write the queue file now (mutations already save on their own)."""
        with self._lock:
            self._write_locked()

    def _write_locked(self) -> None:
        """Caller must hold `self._lock`; writes atomically via tmp + replace."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps([asdict(item) for item in self._items],
                             indent=2, ensure_ascii=False)
        fd, tmp = tempfile.mkstemp(dir=str(self.path.parent),
                                   prefix=self.path.name + ".", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
            os.replace(tmp, self.path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def load(self) -> None:
        """Re-read the queue file; a missing or unreadable file means empty."""
        with self._lock:
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                self._items = []
                return
            except (OSError, ValueError):
                self._items = []
                return
            known = {entry.name for entry in fields(Item)}
            items = []
            for entry in raw if isinstance(raw, list) else []:
                if not isinstance(entry, dict):
                    continue
                if not all(key in entry for key in ("id", "title", "source")):
                    continue
                items.append(Item(**{k: v for k, v in entry.items()
                                     if k in known}))
            self._items = items


__all__ = ["Item", "Setlist", "STATUSES", "default_state_dir"]
