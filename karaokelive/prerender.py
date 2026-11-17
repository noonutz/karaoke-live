"""Offline pre-rendering: separate vocals with the demucs CLI in a subprocess.

Torch is never imported here -- everything demucs-related runs in
`sys.executable -m demucs.separate`, so the calling process stays light and a
crash in the model cannot take the queue down with it.
"""
from __future__ import annotations

import subprocess
import sys
import threading
from pathlib import Path

from .setlist import Setlist

_ERROR_TAIL_LINES = 20
_ERROR_MAX_CHARS = 300


def separate_file(source: str, out_dir: str, model: str = "htdemucs") -> str:
    """Separate `source` and return the written `no_vocals.*` path.

    Raises `RuntimeError` (carrying the last 20 lines of stderr) when demucs
    is missing or the separation fails.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    before = set(out.rglob("no_vocals.*"))
    cmd = [sys.executable, "-m", "demucs.separate",
           "--two-stems", "vocals", "-n", model,
           "-o", str(out), str(source)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    except OSError as exc:
        raise RuntimeError(f"demucs failed to start: {exc}") from exc
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-_ERROR_TAIL_LINES:]
        detail = "\n".join(tail) if tail else f"exit code {proc.returncode}"
        raise RuntimeError(f"demucs failed (exit {proc.returncode}):\n{detail}")
    fresh = [p for p in out.rglob("no_vocals.*") if p not in before]
    if not fresh:
        fresh = list(out.rglob("no_vocals.*"))
    if not fresh:
        raise RuntimeError(f"demucs finished but wrote no no_vocals file in {out}")
    return str(max(fresh, key=lambda p: p.stat().st_mtime))


class PrerenderWorker(threading.Thread):
    """Drain a `Setlist` in the background, one item at a time, newest last.

    `separator` is `(source, out_dir) -> instrumental path` and defaults to
    `separate_file`; pass a stub in tests. Failed items stay failed (no
    retries) until the user re-adds them.
    """

    def __init__(self, setlist: Setlist, out_dir: str | Path,
                 separator=None, poll_s: float = 2.0):
        super().__init__(name="prerender", daemon=True)
        self.setlist = setlist
        self.out_dir = str(out_dir)
        self.separator = separator or separate_file
        self.poll_s = poll_s
        self._stop_event = threading.Event()

    def run(self) -> None:
        # Items stuck in "rendering" (e.g. from an interrupted run) go back
        # to pending so they are picked up again.
        for item in self.setlist.items():
            if item.status == "rendering":
                self._mark(item.id, "pending")
        while not self._stop_event.is_set():
            item = self.setlist.next_pending()
            if item is None:
                self._stop_event.wait(self.poll_s)
                continue
            try:
                self.setlist.set_status(item.id, "rendering")
            except KeyError:
                continue  # removed while we were looking
            try:
                instrumental = self.separator(item.source, self.out_dir)
            except Exception as exc:  # every failure becomes a failed item
                message = str(exc) or type(exc).__name__
                self._mark(item.id, "failed", error=message[:_ERROR_MAX_CHARS])
                continue
            self._mark(item.id, "ready", instrumental=str(instrumental))

    def _mark(self, id: str, status: str, **kwargs) -> None:
        try:
            self.setlist.set_status(id, status, **kwargs)
        except KeyError:
            pass  # removed mid-render; nothing left to mark

    def stop(self, timeout: float | None = None) -> None:
        """Signal the loop to exit and wait for the thread to finish."""
        self._stop_event.set()
        self.join(timeout=timeout)


__all__ = ["PrerenderWorker", "separate_file"]
