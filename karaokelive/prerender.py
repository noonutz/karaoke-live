"""Offline pre-rendering: separate stems with the demucs CLI in a subprocess.

Torch is never imported here -- everything demucs-related runs in
`sys.executable -m demucs.separate`, so the calling process stays light and a
crash in the model cannot take the queue down with it.
"""
from __future__ import annotations

import subprocess
import sys
import threading
from pathlib import Path

from . import audio_io
from . import tempo
from .setlist import Setlist

_ERROR_TAIL_LINES = 20
_ERROR_MAX_CHARS = 300

#: The four stems demucs writes per track.
STEMS = ("vocals", "drums", "bass", "other")


def separate_file(source: str, out_dir: str, model: str = "htdemucs",
                  timeout: float = 3600.0) -> dict[str, str]:
    """Separate `source` into four stems and return `{stem: path}`.

    Runs demucs **without** `--two-stems`, so it writes `vocals.wav`,
    `drums.wav`, `bass.wav` and `other.wav` under
    `<out_dir>/<model>/<track name>/`. Each stem is then transcoded to a
    sibling `<stem>-pcm.wav` (16-bit, 48 kHz, stereo plain PCM via
    `audio_io.to_pcm_wav`) and a dict of those four PCM absolute paths is
    returned -- demucs writes IEEE-float WAVs that stdlib `wave` (and
    therefore `StemPlayer`) cannot open, so the player must see the PCM
    copies. The demucs originals stay on disk untouched.

    Raises `RuntimeError` (carrying the last 20 lines of stderr) when demucs
    is missing or the separation fails, a Thai `RuntimeError` naming the
    model and the file when demucs takes longer than `timeout` seconds, a
    Thai `RuntimeError` when any of the four stems is missing afterwards,
    and a `RuntimeError` naming the stem when its PCM transcode fails.
    No extra mix files are written -- the player mixes at playback time.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "demucs.separate",
           "-n", model,
           "-o", str(out), str(source)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"demucs ใช้เวลานานเกินกำหนด (timeout {timeout:g} วินาที) "
            f"โมเดล {model} ไฟล์ {source}") from exc
    except OSError as exc:
        raise RuntimeError(f"demucs failed to start: {exc}") from exc
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-_ERROR_TAIL_LINES:]
        detail = "\n".join(tail) if tail else f"exit code {proc.returncode}"
        raise RuntimeError(f"demucs failed (exit {proc.returncode}):\n{detail}")
    track = Path(source).stem or Path(source).name
    roots = [out / model / track]
    if not roots[0].is_dir():
        roots = [p for p in out.rglob(track) if p.is_dir()]
    stems: dict[str, str] = {}
    for stem in STEMS:
        found = next((p for root in roots
                      for p in root.glob(f"{stem}.*")), None)
        if found is None or not found.is_file():
            raise RuntimeError(
                f"แยกเสียงไม่สมบูรณ์ — ไม่พบสเต็ม {stem} ของโมเดล {model} "
                f"ไฟล์ {source} ใน {out}")
        pcm = found.with_name(f"{found.stem}-pcm.wav")
        try:
            stems[stem] = audio_io.to_pcm_wav(
                found, pcm, sample_rate=48000, bits=16)
        except RuntimeError as exc:
            raise RuntimeError(
                f"แปลงสเต็ม {stem} ({found}) เป็น PCM WAV ไม่สำเร็จ: {exc}"
            ) from exc
    return stems


def detect_bpm(source: str) -> float | None:
    """Probe `source` for its BPM; `None` when probing fails.

    Decodes at 22050 Hz mono via `audio_io.load` to keep
    `tempo.detect_tempo` fast, and rounds to one decimal. Never raises.
    """
    try:
        audio = audio_io.load(source, sample_rate=22050, channels=1)
        if audio.size == 0:
            return None
        return round(float(tempo.detect_tempo(audio, 22050).bpm), 1)
    except Exception:
        return None


class PrerenderWorker(threading.Thread):
    """Drain a `Setlist` in the background, one item at a time, newest last.

    `separator` is `(source, out_dir) -> stems dict or instrumental path`
    and defaults to `separate_file`; pass a stub in tests. A dict result is
    stored as 4-stem `stems` (plus a probed `bpm`, never failing the render);
    a plain string result is stored as the legacy 2-stem `instrumental`
    path. Failed items stay failed (no retries) until the user re-adds them.
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
                result = self.separator(item.source, self.out_dir)
            except Exception as exc:  # every failure becomes a failed item
                message = str(exc) or type(exc).__name__
                self._mark(item.id, "failed", error=message[:_ERROR_MAX_CHARS])
                continue
            if isinstance(result, dict):
                self._mark(item.id, "ready",
                           stems=result, bpm=detect_bpm(item.source))
            else:  # legacy 2-stem separator returning a plain path
                self._mark(item.id, "ready", instrumental=str(result))

    def _mark(self, id: str, status: str, **kwargs) -> None:
        try:
            self.setlist.set_status(id, status, **kwargs)
        except KeyError:
            pass  # removed mid-render; nothing left to mark

    def stop(self, timeout: float | None = 5.0) -> None:
        """Signal the loop to exit and wait for the thread to finish."""
        self._stop_event.set()
        self.join(timeout=timeout)


__all__ = ["STEMS", "PrerenderWorker", "detect_bpm", "separate_file"]
