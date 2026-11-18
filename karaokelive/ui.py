"""Local web control panel for karaoke-live (stdlib HTTP only).

One `ControlPanel` owns a single processor pipeline and an HTTP server:

* `process_loop()` runs in a worker thread. When `running` it opens the
  audio stream, copies blocks through `engine.process`, measures input and
  output RMS in dB (clamped at -90), feeds a `RollingKeyDetector`, and stores
  everything in a lock-protected state dict. Any exception lands in
  `state["error"]` and stops the run. Device lookup (`find_device`) and the
  high-latency ring buffers (`Ring`) are reused from `realtime.py`; the
  zero-latency path mirrors `LiveRouter`'s inline callback.
* `demo=True` needs no audio device at all: a thread synthesises plausible
  meters (a slow random walk between -45 and -6 dB) and rotates the detected
  key every ~8 s, so the whole UI can be exercised on a headless machine.

Routes (all JSON is `application/json; charset=utf-8`):

* `GET /`             -> the single-file UI in `karaokelive/web/index.html`
* `GET /api/state`    -> the state dict (includes `demo`: bool,
  `queue`: list, `now_playing`: dict | None)
* `POST /api/config`  -> partial config update; 409 while running
* `POST /api/start` / `POST /api/stop` -> new state, or 400 + device list
* `GET /api/devices`  -> `{"input": [...], "output": [...]}`
* `GET /api/queue`    -> `{"items": [...]}` (the pre-render queue)
* `POST /api/queue/add {"source", "title"?}` -> 200 `{"item": ...}`;
  400 (Thai) on duplicate source or missing file
* `POST /api/queue/remove {"id"}` -> 200 `{"items": [...]}`
* `POST /api/queue/run` -> starts/reuses the PrerenderWorker, 200
* `POST /api/queue/play {"id"}` -> playback of a `ready` item: four stems
  via `StemPlayer`, else the `FilePlayer` mix; 409 (Thai) when not ready,
  400 when something else is playing
* `POST /api/queue/stop` -> stops file playback (`now_playing` -> null)
* `POST /api/playback {"drums", "vocals", "loop_bars"}` -> live stem gains
  + 4-bar loop; 409 (Thai) without stems playing; returns the `stems` block
* `POST /api/panic` -> silence everything (playback + live run off, error
  cleared); 200, the panel restarts immediately afterwards
* `GET /api/events`   -> Server-Sent Events, one frame every 500 ms
  (payload also carries `now_playing`, `queue_count` and `stems`)

Anything else is a 404. Local only, so no CORS handling.
"""
from __future__ import annotations

import collections
import json
import math
import os
import random
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np

from . import audio_io, realtime
from .engines import ENGINES, create_engine
from .keydetect import RollingKeyDetector, estimate_chord
from .prerender import PrerenderWorker
from .range import Profile, estimate_range, fit_song, midi_to_name
from .setlist import Setlist, default_state_dir
from .stem_player import STEMS as STEM_NAMES
from .stem_player import StemPlayer
from .tempo import RollingTempoDetector

SAMPLE_RATE = 48000
BLOCKSIZE = 512
SSE_INTERVAL_S = 0.5
MIN_DB = -90.0

#: Largest accepted JSON request body (1 MiB); larger declared bodies are
#: rejected without being read.
MAX_JSON_BODY_BYTES = 1024 * 1024

WEB_DIR = Path(__file__).resolve().parent / "web"

#: Keys cycled through in demo mode: (tonic, mode, camelot).
_DEMO_KEYS = (
    ("A", "minor", "8A"),
    ("C", "major", "8B"),
    ("G", "major", "9B"),
    ("D", "minor", "7A"),
    ("F", "major", "7B"),
    ("E", "minor", "9A"),
)

#: Chords cycled through in demo mode: A minor's diatonic set.
_DEMO_CHORDS = ("Am", "F", "C", "G")


def _bar_seconds(bpm: float) -> float:
    """Seconds per 4/4 bar at `bpm` (four beats)."""
    return 60.0 / float(bpm) * 4.0


def _snap_loop(position: float, bpm: float, bars: int) -> tuple[float, float]:
    """`(start_s, end_s)` of a `bars`-bar loop starting at/before `position`.

    The start snaps DOWN to the 2-bar phrase grid (bar-aligned, so a 4-bar
    loop always begins on an even bar — e.g. 7.3 s at 120 BPM gives
    (4.0, 12.0)) and runs `bars` bars forward from there.
    """
    bar = _bar_seconds(bpm)
    grid = 2.0 * bar
    start = math.floor(float(position) / grid) * grid
    start = max(0.0, start)
    end = start + int(bars) * bar
    return (float(start), float(end))


def _empty_stems() -> dict:
    """Fresh `state["stems"]` block: nothing available, unity gains, no loop."""
    return {"available": False, "drums": 1.0, "vocals": 1.0, "loop": None}


def _db(block: np.ndarray) -> float:
    """RMS level of a float block in dBFS, floored at -90."""
    x = np.asarray(block, dtype=np.float32)
    if x.size == 0:
        return MIN_DB
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    if rms <= 1e-9:
        return MIN_DB
    return max(MIN_DB, 20.0 * math.log10(rms))


def _empty_key() -> dict:
    return {"key": None, "mode": None, "confidence": 0.0,
            "camelot": "", "updated_at": None}


def _profile_payload(profile: Profile | None = None) -> dict | None:
    """Panel-friendly profile view, or `None` when unset."""
    if profile is None:
        try:
            profile = Profile.load()
        except Exception:
            return None
    if profile is None:
        return None
    return {"low": float(profile.low), "high": float(profile.high),
            "notes": (f"{midi_to_name(profile.low)}-"
                      f"{midi_to_name(profile.high)}")}


def _demo_profile() -> Profile:
    """Fixed demo profile (A2-C5) so the UI can be exercised headless."""
    return Profile(low=45.0, high=72.0, updated_at=0.0)


def _demo_range_fit() -> dict:
    """Plausible demo pair: song A3-E5 against profile A2-C5 (too high)."""
    profile = _demo_profile()
    fit = fit_song(57.0, 76.0, profile)
    return {"song_low": 57.0, "song_high": 76.0, "song_notes": "A3-E5",
            "shift": int(fit.shift), "verdict": fit.verdict,
            "message": fit.message, "approximate": False}


def _range_fit_payload(song_low: float, song_high: float,
                       profile: Profile | None = None) -> dict | None:
    """Fit dict for the panel state, or `None` when no profile is set."""
    if profile is None:
        try:
            profile = Profile.load()
        except Exception:
            return None
    if profile is None:
        return None
    fit = fit_song(song_low, song_high, profile)
    return {"song_low": float(song_low), "song_high": float(song_high),
            "shift": int(fit.shift), "verdict": fit.verdict,
            "message": fit.message}


def estimate_song_range(item_id: str | None = None,
                        setlist: Setlist | None = None,
                        approximate: bool = False) -> dict | None:
    """Estimate a song's range for the panel (vocal stem when available).

    When `item_id` points at a queued item whose instrumental file exists,
    the ``vocals.*`` sibling next to it is estimated; otherwise the
    instrumental/mix at that path is used and the result is marked
    approximate.
    """
    path: str | None = None
    if item_id is not None and setlist is not None:
        try:
            item = setlist.get(item_id)
        except Exception:
            item = None
        if item is not None and item.instrumental:
            inst = Path(item.instrumental)
            vocals: Path | None = None
            for candidate in inst.parent.glob("vocals.*"):
                vocals = candidate
                break
            if vocals is not None and vocals.is_file():
                path = str(vocals)
                approximate = False
            elif inst.is_file():
                path = str(inst)
                approximate = True
    if path is None:
        return None
    try:
        data = audio_io.load(path, SAMPLE_RATE)
        result = estimate_range(data, SAMPLE_RATE)
    except ValueError:
        return None
    except Exception:
        return None
    payload = _range_fit_payload(result.low_midi, result.high_midi)
    if payload is None:
        return None
    payload["song_notes"] = result.notes
    payload["approximate"] = bool(approximate)
    return payload


_QUEUE_STATUS_TH = {
    "pending": "รอคิว",
    "rendering": "กำลังแยก",
    "ready": "พร้อม",
    "failed": "ผิดพลาด",
}


def _item_payload(item) -> dict:
    """One queue row for `/api/queue` and `/api/state`."""
    return {
        "id": item.id,
        "title": item.title,
        "status": item.status,
        "duration_s": item.duration_s,
        "instrumental": item.instrumental,
        "error": item.error,
    }


class FilePlayer:
    """Plays one rendered instrumental file.

    Real mode preloads the whole file with `audio_io.load` and serves an
    `sd.OutputStream` callback from that array in a loop, so the stream can
    never under-run. Demo mode never touches an audio device: a timer thread
    just advances `position_s` and stops at the end. Device selection comes
    from the panel's current `output`. Any failure is returned as an error
    string so the panel can put it in `state["error"]` and stop playback.
    """

    def __init__(self, panel: "ControlPanel", item_id: str, title: str,
                 path: str, duration_s: float, demo: bool, output: str):
        self._panel = panel
        self.item_id = item_id
        self.title = title
        self.path = path
        self.duration_s = float(duration_s)
        self._demo = demo
        self._output = output
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._stream = None
        self._data: np.ndarray | None = None
        self._pos = 0
        # Panel parity with `StemPlayer`: stored (no audio effect on a mix),
        # so `/api/playback` can treat both players alike.
        self._gains = {"drums": 1.0, "vocals": 1.0}
        self._loop: tuple[float | None, float | None] | None = None

    def set_gain(self, stem: str, value: float) -> None:
        """Remember a stem gain (a mix has no stems: no audio effect)."""
        self._gains[str(stem)] = float(min(1.0, max(0.0, value)))

    def set_loop(self, start_s: float | None,
                 end_s: float | None) -> None:
        """Remember a loop range (a mix loops whole: no audio effect)."""
        if start_s is None and end_s is None:
            self._loop = None
            return
        self._loop = (start_s, end_s)

    @property
    def position_s(self) -> float:
        """Current playback position in seconds (from the panel state)."""
        return self._panel._now_playing_position(self.item_id)

    def start(self) -> str | None:
        """Start playback; returns an error message, or `None` on success."""
        if self._demo:
            self._thread = threading.Thread(
                target=self._demo_run, name="file-player", daemon=True)
            self._thread.start()
            return None
        try:
            data = audio_io.load(self.path, SAMPLE_RATE)
        except Exception as exc:
            return f"เปิดไฟล์เสียงไม่ได้ ({exc})"
        if data.size == 0 or data.ndim != 2:
            return "เปิดไฟล์เสียงไม่ได้ (ไฟล์เสียงไม่ถูกต้อง)"
        try:
            import sounddevice as sd
        except Exception:
            return "ไม่พบ sounddevice: ติดตั้งด้วย pip install sounddevice"
        try:
            dev = None
            if self._output:
                try:
                    dev = realtime.find_device(self._output, want_input=False)
                except realtime.DeviceNotFound as exc:
                    return str(exc)
                except Exception as exc:
                    return f"เริ่มเล่นไม่สำเร็จ: {type(exc).__name__}"
            self._data = np.ascontiguousarray(data, dtype=np.float32)
            self._stream = sd.OutputStream(
                samplerate=SAMPLE_RATE, channels=2, dtype="float32",
                device=dev, blocksize=BLOCKSIZE, latency="low",
                callback=self._callback)
            self._stream.start()
        except Exception as exc:
            self._stream = None
            return f"เล่นเสียงไม่ได้ ({exc})"
        return None

    def stop(self) -> None:
        """Stop playback and release the device/thread (safe when idle)."""
        self._stop.set()
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass
        thread, self._thread = self._thread, None
        if (thread is not None and thread.is_alive()
                and thread is not threading.current_thread()):
            thread.join(timeout=5)

    # -- internals -----------------------------------------------------
    def _callback(self, outdata, frames, _time_info, _status) -> None:
        if self._stop.is_set() or self._data is None:
            outdata.fill(0)
            return
        data = self._data
        total = len(data)
        idx = (self._pos + np.arange(frames)) % total
        outdata[:] = data[idx]
        self._pos = (self._pos + frames) % total
        try:
            self._panel._advance_playback(
                frames / SAMPLE_RATE, loop=True,
                item_id=self.item_id, block=np.asarray(outdata))
        except Exception:
            pass

    def _demo_run(self) -> None:
        step = 0.1
        while not self._stop.is_set():
            time.sleep(step)
            if self._stop.is_set():
                break
            try:
                ended = self._panel._advance_playback(
                    step, loop=False, item_id=self.item_id)
            except Exception:
                break
            if ended:
                break


class StemPlayback(StemPlayer):
    """A `StemPlayer` the panel can start/stop like a `FilePlayer`.

    Real mode drives `read()` from an `sd.OutputStream` callback; demo mode
    advances through `read()` on a timer thread (so loops wrap exactly as on
    device) and mirrors the position into the panel state. Gains and the loop
    are the inherited `StemPlayer` ones, so `/api/playback` applies live.
    """

    def __init__(self, panel: "ControlPanel", item_id: str, title: str,
                 paths: dict[str, str], demo: bool, output: str):
        super().__init__(paths, sample_rate=SAMPLE_RATE)
        self._panel = panel
        self.item_id = item_id
        self.title = title
        self.duration_s = float(self.duration_s)
        self._demo = demo
        self._output = output
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._stream = None
        self._finished = False

    def start(self) -> str | None:
        """Start playback; returns an error message, or `None` on success."""
        if self._demo:
            self._thread = threading.Thread(
                target=self._demo_run, name="stem-player", daemon=True)
            self._thread.start()
            return None
        try:
            import sounddevice as sd
        except Exception:
            return "ไม่พบ sounddevice: ติดตั้งด้วย pip install sounddevice"
        try:
            dev = None
            if self._output:
                try:
                    dev = realtime.find_device(self._output, want_input=False)
                except realtime.DeviceNotFound as exc:
                    return str(exc)
                except Exception as exc:
                    return f"เริ่มเล่นไม่สำเร็จ: {type(exc).__name__}"
            self._stream = sd.OutputStream(
                samplerate=SAMPLE_RATE, channels=2, dtype="float32",
                device=dev, blocksize=BLOCKSIZE, latency="low",
                callback=self._callback)
            self._stream.start()
        except Exception as exc:
            self._stream = None
            return f"เล่นเสียงไม่ได้ ({exc})"
        return None

    def stop(self) -> None:
        """Stop playback and release the device/thread (safe when idle)."""
        self._stop.set()
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass
        thread, self._thread = self._thread, None
        if (thread is not None and thread.is_alive()
                and thread is not threading.current_thread()):
            thread.join(timeout=5)

    @property
    def _at_end(self) -> bool:
        """True when the song is over (a loop wraps forever, never ending)."""
        return self._loop is None and self._pos >= self._frames

    # -- internals -----------------------------------------------------
    def _callback(self, outdata, frames, _time_info, _status) -> None:
        if self._stop.is_set():
            outdata.fill(0)
            return
        try:
            data = self.read(frames)
        except Exception:
            outdata.fill(0)
            return
        got = data.shape[0]
        if got >= frames:
            outdata[:] = data[:frames]
        else:
            if got:
                outdata[:got] = data
            outdata[got:] = 0
        try:
            self._panel._sync_stem_position(
                self, self.item_id, finished=self._at_end,
                block=np.asarray(outdata))
        except Exception:
            pass
        if self._at_end and not self._finished:
            # End of song on the audio thread: go silent now, release the
            # device from a helper thread (never block the callback).
            self._finished = True
            helper = threading.Thread(target=self._delayed_stop,
                                      name="stem-player-stop", daemon=True)
            helper.start()

    def _delayed_stop(self) -> None:
        time.sleep(0.3)
        try:
            self.stop()
        except Exception:
            pass

    def _demo_run(self) -> None:
        step = 0.1
        while not self._stop.is_set():
            time.sleep(step)
            if self._stop.is_set():
                break
            try:
                self.read(max(int(step * self.sample_rate), 1))
                ended = self._at_end
                self._panel._sync_stem_position(
                    self, self.item_id, finished=ended)
            except Exception:
                break
            if ended:
                break


class ControlPanel:
    """Owns one processor pipeline and the HTTP server in front of it."""

    def __init__(self, port: int = 8765, host: str = "127.0.0.1",
                 demo: bool = False, setlist: Setlist | None = None,
                 prerender: PrerenderWorker | None = None):
        self.host = host
        self.port = int(port)
        self.demo = bool(demo)
        self._lock = threading.Lock()
        self._state: dict = {
            "capture": "",
            "output": "",
            "engine": "center",
            "vocal_level": 0.85,
            "bass_hz": 140.0,
            "mono_guard": True,
            "running": False,
            "latency_ms": 0.0,
            "error": None,
            "demo": self.demo,
            "meters": {"in_db": MIN_DB, "out_db": MIN_DB},
            "key": _empty_key(),
            "tempo": None,
            "chord": None,
            "devices": [],
            "queue": [],
            "now_playing": None,
            "stems": _empty_stems(),
            "panicked_at": None,
            "range_fit": _demo_range_fit() if self.demo else None,
            "profile": (_profile_payload(_demo_profile())
                        if self.demo else _profile_payload()),
        }
        self.setlist = (setlist if setlist is not None
                        else Setlist(path=default_state_dir() / "setlist.json"))
        self.prerender = prerender
        self._player: FilePlayer | StemPlayer | None = None
        self._sync_queue()
        try:
            self._state["devices"] = realtime.list_devices()
        except Exception:
            self._state["devices"] = []
        self._index_html: bytes | None = None
        self._server: ThreadingHTTPServer | None = None
        self._server_thread: threading.Thread | None = None
        self._worker: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._engine_failures = 0
        self._in_dev: int | None = None
        self._out_dev: int | None = None

    # -- state ---------------------------------------------------------
    def snapshot(self) -> dict:
        """A JSON-serialisable copy of the state dict."""
        with self._lock:
            return self.snapshot_locked()

    def _sync_queue(self) -> None:
        """Refresh `state["queue"]` from the setlist (caller need not hold)."""
        try:
            items = [_item_payload(item) for item in self.setlist.items()]
        except Exception:
            items = []
        with self._lock:
            self._state["queue"] = items

    def _set(self, **fields) -> None:
        with self._lock:
            self._state.update(fields)

    def _set_meters(self, in_db: float, out_db: float) -> None:
        with self._lock:
            self._state["meters"] = {"in_db": float(in_db),
                                     "out_db": float(out_db)}

    def _set_key_result(self, result) -> None:
        with self._lock:
            self._state["key"] = {
                "key": result.key,
                "mode": result.mode,
                "confidence": float(result.confidence),
                "camelot": result.camelot,
                "chroma": [float(v) for v in result.chroma],
                "updated_at": time.time(),
            }

    def _set_tempo_result(self, result) -> None:
        with self._lock:
            self._state["tempo"] = {
                "bpm": float(result.bpm),
                "confidence": float(result.confidence),
            }

    def _set_chord(self, chord: str) -> None:
        with self._lock:
            self._state["chord"] = chord

    def _is_running(self) -> bool:
        with self._lock:
            return bool(self._state["running"])

    def _fail(self, message: str) -> None:
        with self._lock:
            self._state["error"] = message
            self._state["running"] = False

    def _engine_failure(self, exc: BaseException) -> None:
        """Record one failed `engine.process` call.

        Fires `self._fail(...)` (and requests a stream stop) only on the 5th
        consecutive failure, so a transient glitch does not kill the run while
        a persistent torch/demucs error cannot stay silent forever.
        """
        self._engine_failures += 1
        if self._engine_failures >= 5:
            reason = str(exc) or repr(exc)
            self._fail(f"แยกเสียงไม่สำเร็จ: {reason} — หยุดเล่นแล้ว")
            self._stop_event.set()

    def _engine_ok(self) -> None:
        self._engine_failures = 0

    def device_lists(self) -> dict:
        """Split the device table into capture/playback candidates."""
        try:
            rows = realtime.list_devices()
        except Exception:
            rows = []
        with self._lock:
            self._state["devices"] = rows
        return {
            "input": [r for r in rows if r.get("inputs", 0) > 0],
            "output": [r for r in rows if r.get("outputs", 0) > 0],
        }

    def index_html(self) -> bytes:
        if self._index_html is None:
            self._index_html = (WEB_DIR / "index.html").read_bytes()
        return self._index_html

    def overlay_html(self) -> bytes:
        if not hasattr(self, "_overlay_html") or self._overlay_html is None:
            self._overlay_html = (WEB_DIR / "overlay.html").read_bytes()  # type: ignore[attr-defined]
        return self._overlay_html  # type: ignore[attr-defined]

    # -- API -----------------------------------------------------------
    def api_config(self, body: dict) -> tuple[int, dict]:
        """Partial config update; rejected with 409 while running."""
        updates: dict = {}
        if "capture" in body:
            updates["capture"] = str(body["capture"])
        if "output" in body:
            updates["output"] = str(body["output"])
        if "engine" in body:
            kind = str(body["engine"]).lower()
            if kind not in ENGINES:
                return 400, {"error": (
                    f"เอนจิน {body['engine']!r} ไม่ถูกต้อง "
                    f"(เลือกจาก: {', '.join(ENGINES)})")}
            updates["engine"] = kind
        if "vocal_level" in body:
            try:
                level = float(body["vocal_level"])
            except (TypeError, ValueError):
                return 400, {"error": "ระดับเสียงร้องต้องเป็นตัวเลข 0-100"}
            if level > 1.0:  # accept percent as well as 0..1
                level /= 100.0
            updates["vocal_level"] = float(np.clip(level, 0.0, 1.0))
        if "bass_hz" in body:
            try:
                bass = float(body["bass_hz"])
            except (TypeError, ValueError):
                return 400, {"error": "ความถี่เบสต้องเป็นตัวเลข"}
            if bass <= 0:
                return 400, {"error": "ความถี่เบสต้องมากกว่า 0 Hz"}
            updates["bass_hz"] = bass
        if "mono_guard" in body:
            updates["mono_guard"] = bool(body["mono_guard"])
        # The running check and the write share one lock hold, so a start
        # that lands in between cannot mutate config under a live engine.
        with self._lock:
            if self._state["running"]:
                return 409, {
                    "error": ("กำลังเล่นอยู่ กรุณากดหยุดก่อนเปลี่ยน"
                              "การตั้งค่า"),
                    "state": self.snapshot_locked(),
                }
            self._state.update(updates)
            return 200, self.snapshot_locked()

    def _stems_snapshot_locked(self) -> dict:
        """Copy of `state["stems"]`; caller must hold the lock."""
        stems = self._state.get("stems") or {}
        loop = stems.get("loop")
        return {"available": bool(stems.get("available", False)),
                "drums": float(stems.get("drums", 1.0)),
                "vocals": float(stems.get("vocals", 1.0)),
                "loop": dict(loop) if isinstance(loop, dict) else None}

    def snapshot_locked(self) -> dict:
        """Copy of state; caller must hold the lock (see `api_config`)."""
        st = self._state
        tempo = st["tempo"]
        now = st.get("now_playing")
        if not st.get("demo"):
            # Pick up profiles saved by `karaokelive range` while running.
            st["profile"] = _profile_payload()
        range_fit = st.get("range_fit")
        profile = st.get("profile")
        return {**st, "meters": dict(st["meters"]),
                "key": dict(st["key"]),
                "tempo": dict(tempo) if tempo is not None else None,
                "devices": list(st["devices"]),
                "queue": [dict(item) for item in st.get("queue", [])],
                "now_playing": dict(now) if now is not None else None,
                "stems": self._stems_snapshot_locked(),
                "range_fit": (dict(range_fit)
                              if isinstance(range_fit, dict) else None),
                "profile": (dict(profile)
                            if isinstance(profile, dict) else None)}

    def api_start(self) -> tuple[int, dict]:
        """Start processing; 400 (Thai) when no device can be resolved."""
        if self._is_running():
            return 200, self.snapshot()
        self.stop_audio()
        with self._lock:
            capture = self._state["capture"]
            output = self._state["output"]
        if not self.demo:
            if not capture or not output:
                msg = "กรุณาเลือกอุปกรณ์รับเสียงและอุปกรณ์เล่นเสียงก่อนกดเริ่ม"
                self._fail(msg)
                return 400, {"error": msg, "devices": self.device_lists()}
            try:
                self._in_dev = realtime.find_device(capture, want_input=True)
                self._out_dev = realtime.find_device(output, want_input=False)
            except realtime.DeviceNotFound as exc:
                msg = str(exc)
                self._fail(msg)
                return 400, {"error": msg, "devices": self.device_lists()}
            except Exception as exc:
                msg = f"เริ่มเล่นไม่สำเร็จ: {type(exc).__name__}"
                self._fail(msg)
                return 400, {"error": msg, "devices": self.device_lists()}
        self._engine_failures = 0  # per-run counter (see _engine_failure)
        with self._lock:
            self._state.update({
                "running": True,
                "error": None,
                "meters": {"in_db": MIN_DB, "out_db": MIN_DB},
                "key": _empty_key(),
                "tempo": None,
                "chord": None,
                "latency_ms": BLOCKSIZE / SAMPLE_RATE * 1000.0,
            })
        self._stop_event.clear()
        worker = threading.Thread(target=self.process_loop,
                                  name="panel-audio", daemon=True)
        with self._lock:
            self._worker = worker
        worker.start()
        return 200, self.snapshot()

    def api_stop(self) -> tuple[int, dict]:
        self.stop_audio()
        return 200, self.snapshot()

    # -- queue ---------------------------------------------------------
    def api_queue_add(self, body: dict) -> tuple[int, dict]:
        """Queue `body["source"]`; 400 (Thai) on duplicates/missing files."""
        source = str(body.get("source") or "").strip()
        title = body.get("title")
        title = str(title).strip() if title is not None else None
        if not source:
            return 400, {"error": "กรุณาระบุไฟล์เพลง (source)"}
        try:
            item = self.setlist.add(source, title=title or None)
        except ValueError:
            return 400, {"error": "เพลงนี้อยู่ในคิวแล้ว"}
        self._sync_queue()
        return 200, {"item": _item_payload(item)}

    def api_queue_remove(self, body: dict) -> tuple[int, dict]:
        """Drop one queue item; 400 (Thai) when the id is missing."""
        item_id = str(body.get("id") or "").strip()
        if not item_id:
            return 400, {"error": "กรุณาระบุรหัสเพลง (id)"}
        if item_id == self._now_playing_id():
            self.api_queue_stop()
        self.setlist.remove(item_id)
        self._sync_queue()
        return 200, {"items": [_item_payload(item)
                               for item in self.setlist.items()]}

    def api_queue_run(self) -> tuple[int, dict]:
        """Start the pre-render worker (reused while it is alive)."""
        worker = self.prerender
        if worker is None or not worker.is_alive():
            worker = PrerenderWorker(
                self.setlist, str(default_state_dir() / "stems"), poll_s=1.0)
            worker.start()
            self.prerender = worker
        return 200, {"started": True}

    def _stem_paths(self, item) -> dict[str, str] | None:
        """The four stem paths when `item` carries all of them, else `None`."""
        stems = getattr(item, "stems", None)
        if not isinstance(stems, dict):
            return None
        if not all(stems.get(name) for name in STEM_NAMES):
            return None
        return {name: str(stems[name]) for name in STEM_NAMES}

    def api_queue_play(self, body: dict) -> tuple[int, dict]:
        """Play a `ready` item; live engine is bypassed.

        Items with four stems play through a `StemPlayback` (per-stem gains
        and a loop, driven by `/api/playback`); anything else falls back to
        the `FilePlayer` mix. Returns 404 when the id is unknown, 409 (Thai)
        when the item is not ready yet or has failed, and 400 when something
        else is playing or the stream cannot be opened.
        """
        item_id = str(body.get("id") or "").strip()
        if not item_id:
            return 400, {"error": "กรุณาระบุรหัสเพลง (id)"}
        item = self.setlist.get(item_id)
        if item is None:
            return 404, {"error": "ไม่พบเพลงนี้ในคิว"}
        if item.status == "failed":
            return 409, {"error": f"แยกเสียงไม่สำเร็จ ({item.error or 'ดูรายละเอียดในคิว'})"}
        stem_paths = self._stem_paths(item) if item.status == "ready" else None
        if item.status != "ready" or (stem_paths is None
                                      and not item.instrumental):
            label = _QUEUE_STATUS_TH.get(item.status, item.status)
            return 409, {"error": f"เพลงนี้ยังไม่พร้อม (สถานะ: {label})"}
        with self._lock:
            if self._state["now_playing"] is not None:
                return 400, {"error": "กำลังเล่นเพลงอื่นอยู่ กรุณากดหยุดก่อน"}
        with self._lock:
            output = self._state["output"]
        if stem_paths is not None:
            player: FilePlayer | StemPlayer = StemPlayback(
                self, item.id, item.title, stem_paths, self.demo, output)
            try:
                duration_s = float(item.duration_s) if item.duration_s else 0.0
            except Exception:
                duration_s = 0.0
            if not duration_s or duration_s <= 0:
                duration_s = player.duration_s
            if not duration_s or duration_s <= 0:
                try:
                    player.stop()
                except Exception:
                    pass
                if isinstance(player, StemPlayer):
                    try:
                        player.close()
                    except Exception:
                        pass
                return 400, {"error": "อ่านความยาวเพลงไม่ได้ กรุณาลองใหม่อีกครั้ง"}
        else:
            assert item.instrumental is not None
            if not os.path.exists(item.instrumental):
                return 400, {"error": "ไม่พบไฟล์เสียงที่แยกไว้ กรุณารันแยกเสียงใหม่"}
            try:
                duration_s = float(item.duration_s) \
                    if item.duration_s else self._probe_duration(item.instrumental)
            except Exception:
                duration_s = 0.0
            if not duration_s or duration_s <= 0:
                return 400, {"error": "อ่านความยาวเพลงไม่ได้ กรุณาลองใหม่อีกครั้ง"}
            player = FilePlayer(self, item.id, item.title, item.instrumental,
                                duration_s, self.demo, output)
        # Switch to file playback first (the live engine is bypassed, so the
        # vocal level drops to 0): the player thread only advances a
        # `now_playing` row that already exists, so the state must be in
        # place before `start()` lets the thread tick.
        self.stop_audio()
        if not self.demo:
            # Prefer the vocal stem; fall back to the mix (approximate) --
            # never let range estimation break queue playback.
            try:
                range_fit = estimate_song_range(item.id, self.setlist)
            except Exception:
                range_fit = None
        else:
            range_fit = _demo_range_fit()
        with self._lock:
            self._state.update({
                "running": False,
                "vocal_level": 0.0,
                "error": None,
                "now_playing": {"id": item.id, "title": item.title,
                                "position_s": 0.0, "duration_s": duration_s},
                "stems": {"available": stem_paths is not None,
                          "drums": 1.0, "vocals": 1.0, "loop": None},
                "range_fit": range_fit,
            })
            self._player = player
        error = player.start()
        if error is not None:
            try:
                player.stop()
            except Exception:
                pass
            with self._lock:
                if self._player is player:
                    self._player = None
                self._state["now_playing"] = None
                self._state["stems"] = _empty_stems()
                self._state["error"] = error
            return 400, {"error": error}
        return 200, self.snapshot()

    def api_playback(self, body: dict) -> tuple[int, dict]:
        """Live stem mixing: gains plus a 4-bar loop; 409 without stems.

        `body` carries the panel's full values (`drums`/`vocals` in 0..1 and
        `loop_bars` 4 or 0) so a stale toggle can never desync the server.
        `loop_bars == 4` loops four bars snapped DOWN to the bar grid from the
        item's `bpm`; `0` clears the loop. Returns the new `stems` block.
        """
        if not isinstance(body, dict):
            return 400, {"error": "ข้อมูล playback ต้องเป็น JSON object"}
        with self._lock:
            player = self._player
            now = self._state.get("now_playing")
            stems = self._stems_snapshot_locked()
        if (player is None or now is None or not stems["available"]
                or not isinstance(player, StemPlayer)):
            return 409, {"error": "ยังไม่มีเพลงแบบแยกสเต็มกำลังเล่น"}
        try:
            drums = float(body.get("drums", stems["drums"]))
            vocals = float(body.get("vocals", stems["vocals"]))
        except (TypeError, ValueError):
            return 400, {"error": "ระดับเสียงกลอง/เสียงร้องต้องเป็นตัวเลข 0-1"}
        if not 0.0 <= drums <= 1.0 or not 0.0 <= vocals <= 1.0:
            return 400, {"error": "ระดับเสียงกลอง/เสียงร้องต้องอยู่ระหว่าง 0 ถึง 1"}
        current_bars = 0
        current_loop = stems["loop"]
        if isinstance(current_loop, dict) and current_loop.get("active"):
            try:
                current_bars = int(current_loop.get("bars", 4))
            except (TypeError, ValueError):
                current_bars = 4
        try:
            loop_bars = int(body.get("loop_bars", current_bars))
        except (TypeError, ValueError):
            return 400, {"error": "วนลูปได้เฉพาะ 0 หรือ 4 ห้อง"}
        if loop_bars not in (0, 4):
            return 400, {"error": "วนลูปได้เฉพาะ 0 หรือ 4 ห้อง"}
        loop: dict | None = None
        if loop_bars == 4:
            try:
                item = self.setlist.get(str(now.get("id") or ""))
            except Exception:
                item = None
            bpm = getattr(item, "bpm", None) if item is not None else None
            try:
                bpm_f = float(bpm) if bpm is not None else 0.0
            except (TypeError, ValueError):
                bpm_f = 0.0
            if bpm_f <= 0:
                return 400, {"error": "เพลงนี้ไม่มีข้อมูลจังหวะ (bpm) วนลูปไม่ได้"}
            try:
                position = float(player.position_s)
            except Exception:
                position = 0.0
            start_s, end_s = _snap_loop(position, bpm_f, loop_bars)
            try:
                player.set_loop(start_s, end_s)
            except Exception as exc:
                return 400, {"error": f"ตั้งวนลูปไม่ได้ ({exc})"}
            loop = {"active": True, "start_s": float(start_s),
                    "end_s": float(end_s), "bars": loop_bars}
        else:
            try:
                player.set_loop(None, None)
            except Exception as exc:
                return 400, {"error": f"ยกเลิกวนลูปไม่ได้ ({exc})"}
        try:
            player.set_gain("drums", drums)
            player.set_gain("vocals", vocals)
        except Exception as exc:
            return 400, {"error": f"ปรับเสียงสเต็มไม่ได้ ({exc})"}
        with self._lock:
            self._state["stems"] = {"available": True, "drums": float(drums),
                                    "vocals": float(vocals), "loop": loop}
            return 200, self._stems_snapshot_locked()

    def api_panic(self) -> tuple[int, dict]:
        """Silence everything: stop playback and the live run, clear errors.

        Records `panicked_at` and returns 200; start/queue-play work again
        immediately afterwards.
        """
        self.stop_audio()
        self.api_queue_stop()
        with self._lock:
            self._state["error"] = None
            self._state["panicked_at"] = time.time()
        return 200, self.snapshot()

    def api_queue_stop(self) -> tuple[int, dict]:
        """Stop file playback; `now_playing` becomes null."""
        with self._lock:
            player, self._player = self._player, None
        if player is not None:
            try:
                player.stop()
            except Exception:
                pass
            if isinstance(player, StemPlayer):
                try:
                    player.close()
                except Exception:
                    pass
        with self._lock:
            self._state["now_playing"] = None
            self._state["stems"] = _empty_stems()
            self._state["meters"] = {"in_db": MIN_DB, "out_db": MIN_DB}
        return 200, self.snapshot()

    @staticmethod
    def _probe_duration(path: str) -> float:
        try:
            return float(audio_io.probe(path).duration_s)
        except Exception:
            return 0.0

    def _now_playing_id(self) -> str | None:
        with self._lock:
            now = self._state.get("now_playing")
            return now.get("id") if isinstance(now, dict) else None

    def _now_playing_position(self, item_id: str) -> float:
        """Current `now_playing.position_s` for `item_id` (0 when not live)."""
        with self._lock:
            now = self._state.get("now_playing")
            if isinstance(now, dict) and now.get("id") == item_id:
                try:
                    return max(0.0, float(now.get("position_s") or 0.0))
                except (TypeError, ValueError):
                    return 0.0
            return 0.0

    def _sync_stem_position(self, player: StemPlayer, item_id: str,
                            finished: bool, block=None) -> bool:
        """Mirror a stem player's position into `now_playing`.

        `finished` comes from the driver (end of song with no loop); a loop
        wraps inside the player so it never finishes. Returns True when the
        driver should exit (finished or superseded).
        """
        with self._lock:
            now = self._state.get("now_playing")
            if now is None or now.get("id") != item_id:
                return True  # superseded or stopped; let the driver exit
            if finished:
                self._state["now_playing"] = None
                if self._player is player:
                    self._player = None
            else:
                try:
                    position = max(0.0, float(player.position_s))
                except Exception:
                    position = float(now.get("position_s") or 0.0)
                self._state["now_playing"] = {**now,
                                              "position_s": position}
                if block is not None:
                    self._state["meters"] = {
                        "in_db": MIN_DB,
                        "out_db": _db(np.asarray(block))}
            return finished

    def _advance_playback(self, delta_s: float, loop: bool,
                          item_id: str, block=None) -> bool:
        """Advance `now_playing.position_s`; returns True when finished."""
        with self._lock:
            now = self._state.get("now_playing")
            if now is None or now.get("id") != item_id:
                return True  # superseded or stopped; let the player exit
            duration = float(now.get("duration_s") or 0.0)
            position = float(now.get("position_s") or 0.0) + float(delta_s)
            if loop and duration > 0:
                position %= duration
            finished = duration > 0 and position >= duration
            if finished:
                position = duration
            self._state["now_playing"] = {**now, "position_s": position}
            if block is not None:
                self._state["meters"] = {"in_db": MIN_DB,
                                         "out_db": _db(np.asarray(block))}
            ended = finished and not loop
            if ended:
                self._state["now_playing"] = None
        if ended:
            with self._lock:
                player = self._player
                if player is None or player.item_id != item_id:
                    # Superseded: another song started after this block was
                    # queued; leave the new player alone.
                    return ended
                self._player = None
            if player is not None:
                try:
                    player.stop()
                except Exception:
                    pass
        return ended

    def stop_audio(self) -> None:
        """Signal the worker to stop and wait for it (safe when idle)."""
        self._stop_event.set()
        with self._lock:
            self._state["running"] = False
            worker = self._worker
            self._worker = None
        if worker and worker.is_alive() \
                and worker is not threading.current_thread():
            worker.join(timeout=5)

    # -- audio worker --------------------------------------------------
    def process_loop(self) -> None:
        """Worker thread: open the stream and process until stopped."""
        if self.demo:
            self._demo_loop()
            return
        try:
            self._run_audio()
        except realtime.DeviceNotFound as exc:
            self._fail(str(exc))
        except Exception as exc:
            self._fail(f"เริ่มเล่นไม่สำเร็จ: {type(exc).__name__}")
        finally:
            with self._lock:
                self._state["running"] = False

    def _build_engine(self):
        with self._lock:
            kind = self._state["engine"]
            level = self._state["vocal_level"]
            bass = self._state["bass_hz"]
            guard = self._state["mono_guard"]
        if kind == "center":
            return create_engine("center", SAMPLE_RATE, level,
                                 bass_hz=bass, mono_guard=guard)
        return create_engine(kind, SAMPLE_RATE, level)

    def _run_audio(self) -> None:
        try:
            import sounddevice as sd
        except Exception as exc:
            raise RuntimeError(
                "ไม่พบ sounddevice: ติดตั้งด้วย pip install sounddevice"
            ) from exc
        engine = self._build_engine()
        # Reset engine state so a second run never starts with a biquad tail
        # or a half-filled demucs buffer (a no-op for a fresh engine).
        engine.reset()
        assert self._in_dev is not None and self._out_dev is not None
        self._set(latency_ms=(engine.latency_frames + BLOCKSIZE)
                  / SAMPLE_RATE * 1000.0)
        detector = RollingKeyDetector(SAMPLE_RATE)
        tempo_detector = RollingTempoDetector(SAMPLE_RATE)
        pending: collections.deque = collections.deque()  # input blocks
        if engine.latency_frames == 0:
            def callback(indata, outdata, frames, _time_info, _status):
                block = np.asarray(indata, dtype=np.float32)
                try:
                    out = np.asarray(engine.process(block), dtype=np.float32)
                    self._engine_ok()
                except Exception as exc:
                    self._engine_failure(exc)
                    out = np.zeros_like(block)
                if out.shape[0] < frames:
                    pad = np.zeros((frames - out.shape[0], out.shape[1]),
                                   dtype=np.float32)
                    out = np.concatenate([out, pad], axis=0)
                outdata[:] = out[:frames]
                self._set_meters(_db(block), _db(outdata))
                pending.append(block.copy())

            with sd.Stream(samplerate=SAMPLE_RATE, blocksize=BLOCKSIZE,
                           device=(self._in_dev, self._out_dev), channels=2,
                           dtype="float32", latency="low", callback=callback):
                self._drain_loop(detector, tempo_detector, pending)
            return

        # Engines with lookahead (demucs): rings + engine worker, mirroring
        # LiveRouter's threaded path; the audio callback only copies.
        ring_in = realtime.Ring.for_engine(engine, BLOCKSIZE)
        ring_out = realtime.Ring.for_engine(engine, BLOCKSIZE)
        # No prefill: the delayed engine already returns latency_frames of
        # silence at the head of its own output, so prefilling would add the
        # delay twice.
        stop = self._stop_event

        def threaded_callback(indata, outdata, frames, _time_info, _status):
            block = np.asarray(indata, dtype=np.float32)
            ring_in.write(block)
            pending.append(block.copy())
            data, _short = ring_out.read(frames)
            self._set_meters(_db(block), _db(data))
            outdata[:] = data

        def engine_worker() -> None:
            while not stop.is_set():
                if ring_in.count >= BLOCKSIZE:
                    blk, _ = ring_in.read(BLOCKSIZE)
                    try:
                        out = engine.process(blk)
                        self._engine_ok()
                    except Exception as exc:
                        self._engine_failure(exc)
                        out = np.zeros_like(blk)
                    ring_out.write(np.asarray(out, dtype=np.float32))
                else:
                    time.sleep(0.002)

        worker = threading.Thread(target=engine_worker, name="panel-engine",
                                  daemon=True)
        worker.start()
        try:
            with sd.Stream(samplerate=SAMPLE_RATE, blocksize=BLOCKSIZE,
                           device=(self._in_dev, self._out_dev), channels=2,
                           dtype="float32", latency="high",
                           callback=threaded_callback):
                self._drain_loop(detector, tempo_detector, pending)
        finally:
            stop.set()
            worker.join(timeout=5)

    def _set_range_fit(self, payload: dict | None) -> None:
        with self._lock:
            self._state["range_fit"] = payload

    def _drain_loop(self, detector: RollingKeyDetector,
                    tempo_detector: RollingTempoDetector,
                    pending: collections.deque) -> None:
        """Feed queued input blocks to the key/tempo detectors until stopped."""
        mix: collections.deque = collections.deque()
        mixed = 0
        since_mix = 0
        need = 8 * SAMPLE_RATE
        every = 4 * SAMPLE_RATE
        window = 8 * SAMPLE_RATE
        while self._is_running() and not self._stop_event.is_set():
            try:
                while True:
                    block = pending.popleft()
                    result = detector.feed(block)
                    if result is not None:
                        self._set_key_result(result)
                    tempo = tempo_detector.feed(block)
                    if tempo is not None:
                        self._set_tempo_result(tempo)
                    try:
                        self._set_chord(estimate_chord(block, SAMPLE_RATE))
                    except Exception:
                        pass
                    try:
                        mono = np.asarray(block, dtype=np.float32)
                        if mono.ndim == 2:
                            mono = mono.mean(axis=1)
                        mono = mono.reshape(-1)
                    except Exception:
                        mono = None
                    if mono is not None and mono.size:
                        mix.append(mono.copy())
                        mixed += len(mono)
                        since_mix += len(mono)
                        while mix and mixed - len(mix[0]) >= window:
                            mixed -= len(mix.popleft())
            except IndexError:
                pass
            if mixed >= need and since_mix >= every:
                since_mix = 0
                try:
                    audio = np.concatenate(list(mix))[-window:]
                    song = estimate_range(audio, SAMPLE_RATE)
                    payload = _range_fit_payload(song.low_midi,
                                                 song.high_midi)
                    if payload is not None:
                        payload["song_notes"] = song.notes
                        payload["approximate"] = True
                        self._set_range_fit(payload)
                except ValueError:
                    pass
                except Exception:
                    pass
            time.sleep(0.05)

    def _demo_loop(self) -> None:
        """Synthesise meters + rotating keys; no audio device needed."""
        index = 0
        chord_index = 0
        bpm = 120.0
        self._set_demo_key(index, 0.82)
        self._set_demo_tempo_chord(bpm, _DEMO_CHORDS[chord_index])
        next_switch = time.time() + 8.0
        next_chord = time.time() + 2.0
        in_db, out_db = -24.0, -26.0
        while self._is_running() and not self._stop_event.is_set():
            now = time.time()
            if now >= next_switch:
                index = (index + 1) % len(_DEMO_KEYS)
                self._set_demo_key(index, 0.6 + 0.35 * random.random())
                next_switch = now + 8.0
            if now >= next_chord:
                chord_index = (chord_index + 1) % len(_DEMO_CHORDS)
                bpm = min(128.0, max(118.0,
                                     bpm + random.uniform(-2.0, 2.0)))
                self._set_demo_tempo_chord(bpm, _DEMO_CHORDS[chord_index])
                next_chord = now + 2.0
            in_db = min(-6.0, max(-45.0, in_db + random.uniform(-2.5, 2.5)))
            out_db = min(-6.0, max(-45.0, out_db + random.uniform(-2.5, 2.5)))
            self._set_meters(in_db, out_db)
            time.sleep(0.1)

    def _set_demo_tempo_chord(self, bpm: float, chord: str) -> None:
        with self._lock:
            self._state["tempo"] = {
                "bpm": float(bpm),
                "confidence": 0.8,
            }
            self._state["chord"] = chord

    def _set_demo_key(self, index: int, confidence: float) -> None:
        from .keydetect import NOTE_NAMES

        key, mode, camelot = _DEMO_KEYS[index]
        # Believable triad chroma: tonic/third/fifth high, the rest low noise.
        tonic = NOTE_NAMES.index(key)
        third = tonic + (3 if mode == "minor" else 4)
        triad = {tonic % 12, third % 12, (tonic + 7) % 12}
        chroma = [0.02 + 0.01 * random.random() for _ in range(12)]
        for pitch in triad:
            chroma[pitch] = 0.25 + 0.08 * random.random()
        total = sum(chroma)
        chroma = [v / total for v in chroma]
        with self._lock:
            self._state["key"] = {
                "key": key,
                "mode": mode,
                "confidence": float(confidence),
                "camelot": camelot,
                "chroma": chroma,
                "updated_at": time.time(),
            }

    # -- server ----------------------------------------------------------
    @property
    def bound_port(self) -> int:
        if self._server is not None:
            return self._server.server_address[1]
        return self.port

    def _create_server(self) -> ThreadingHTTPServer:
        server = ThreadingHTTPServer((self.host, self.port), _PanelHandler)
        server.daemon_threads = True
        server.panel = self  # type: ignore[attr-defined]
        return server

    def start_in_background(self) -> "ControlPanel":
        """Bind and serve in a daemon thread (used by tests)."""
        if self._server is None:
            self._server = self._create_server()
            self._server_thread = threading.Thread(
                target=self._server.serve_forever,
                kwargs={"poll_interval": 0.2},
                name="panel-http", daemon=True)
            self._server_thread.start()
        return self

    def serve_forever(self) -> None:
        """Bind and serve in the current thread (used by the CLI)."""
        if self._server is None:
            self._server = self._create_server()
        try:
            self._server.serve_forever(poll_interval=0.2)
        except KeyboardInterrupt:
            pass

    def shutdown(self) -> None:
        self.stop_audio()
        try:
            self.api_queue_stop()
        except Exception:
            pass
        worker = self.prerender
        self.prerender = None
        if worker is not None and worker.is_alive():
            try:
                worker.stop(timeout=2)
            except Exception:
                pass
        server, self._server = self._server, None
        if server is not None:
            try:
                server.shutdown()
            except Exception:
                pass
            try:
                server.server_close()
            except Exception:
                pass
        thread, self._server_thread = self._server_thread, None
        if thread and thread.is_alive() \
                and thread is not threading.current_thread():
            thread.join(timeout=5)


class _PanelHandler(BaseHTTPRequestHandler):
    server_version = "karaoke-live-ui/0.1"

    def log_message(self, *args) -> None:  # keep test output clean
        pass

    # -- helpers ------------------------------------------------------
    def _json(self, code: int, obj: dict) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        try:
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            # the browser closed the tab or the SSE reader went away -- normal
            # for a local control panel, and not worth a traceback in the log
            pass

    def _read_json_body(self) -> tuple[int | None, dict]:
        raw_value = self.headers.get("Content-Length")
        try:
            length = int(raw_value) if raw_value not in (None, "") else 0
        except (TypeError, ValueError):
            return 400, {"error": "ความยาวข้อมูลไม่ถูกต้อง (Content-Length)"}
        if length < 0:
            return 400, {"error": "ความยาวข้อมูลไม่ถูกต้อง (Content-Length)"}
        if length > MAX_JSON_BODY_BYTES:
            return 413, {"error": "ข้อมูลใหญ่เกินไป (สูงสุด 1 MB)"}
        raw = self.rfile.read(length) if length > 0 else b""
        if not raw.strip():
            return None, {}
        try:
            body = json.loads(raw.decode("utf-8"))
        except Exception:
            return 400, {"error": "อ่านข้อมูล JSON ไม่ได้ (invalid JSON)"}
        if not isinstance(body, dict):
            return 400, {"error": "ข้อมูล config ต้องเป็น JSON object"}
        return None, body

    # -- routes ---------------------------------------------------------
    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        panel: ControlPanel = self.server.panel  # type: ignore[attr-defined]
        if path == "/":
            try:
                page = panel.index_html()
            except FileNotFoundError:
                self._json(500, {"error": "ไม่พบไฟล์หน้าเว็บ (web/index.html)"})
                return
            try:
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(page)))
                self.end_headers()
                self.wfile.write(page)
            except (BrokenPipeError, ConnectionResetError):
                pass  # tab closed while the page was in flight
        elif path == "/overlay":
            try:
                page = panel.overlay_html()
            except FileNotFoundError:
                self._json(500, {"error": "ไม่พบไฟล์โอเวอร์เลย์ (web/overlay.html)"})
                return
            try:
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(page)))
                self.end_headers()
                self.wfile.write(page)
            except (BrokenPipeError, ConnectionResetError):
                pass  # tab closed while the page was in flight
        elif path == "/api/state":
            self._json(200, panel.snapshot())
        elif path == "/api/devices":
            self._json(200, panel.device_lists())
        elif path == "/api/queue":
            panel._sync_queue()
            self._json(200, {"items": panel.snapshot()["queue"]})
        elif path == "/api/events":
            self._events(panel)
        else:
            self._json(404, {"error": "ไม่พบเส้นทางนี้ (404)"})

    def do_POST(self) -> None:
        path = urlsplit(self.path).path
        panel: ControlPanel = self.server.panel  # type: ignore[attr-defined]
        if path not in ("/api/config", "/api/start", "/api/stop",
                        "/api/queue/add", "/api/queue/remove",
                        "/api/queue/run", "/api/queue/play", "/api/queue/stop",
                        "/api/playback", "/api/panic"):
            self._json(404, {"error": "ไม่พบเส้นทางนี้ (404)"})
            return
        fail, body = self._read_json_body()
        if fail is not None:
            self._json(fail, body)
            return
        if path == "/api/config":
            code, obj = panel.api_config(body)
        elif path == "/api/start":
            code, obj = panel.api_start()
        elif path == "/api/stop":
            code, obj = panel.api_stop()
        elif path == "/api/queue/add":
            code, obj = panel.api_queue_add(body)
        elif path == "/api/queue/remove":
            code, obj = panel.api_queue_remove(body)
        elif path == "/api/queue/run":
            code, obj = panel.api_queue_run()
        elif path == "/api/queue/play":
            code, obj = panel.api_queue_play(body)
        elif path == "/api/playback":
            code, obj = panel.api_playback(body)
        elif path == "/api/panic":
            code, obj = panel.api_panic()
        else:
            code, obj = panel.api_queue_stop()
        self._json(code, obj)

    def _events(self, panel: ControlPanel) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        try:
            while True:
                st = panel.snapshot()
                payload = {
                    "meters": st["meters"],
                    "key": st["key"],
                    "tempo": st["tempo"],
                    "chord": st["chord"],
                    "running": st["running"],
                    "error": st["error"],
                    "latency_ms": st["latency_ms"],
                    "now_playing": st.get("now_playing"),
                    "queue_count": len(st.get("queue", [])),
                    "stems": st.get("stems"),
                    "range_fit": st.get("range_fit"),
                    "profile": st.get("profile"),
                }
                frame = ("data: " + json.dumps(payload, ensure_ascii=False)
                         + "\n\n").encode("utf-8")
                self.wfile.write(frame)
                self.wfile.flush()
                end = time.time() + SSE_INTERVAL_S
                while time.time() < end:
                    time.sleep(0.05)
        except (BrokenPipeError, ConnectionResetError, ValueError, OSError):
            pass  # client disconnected; end the stream cleanly
