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
* `GET /api/state`    -> the state dict (includes `demo`: bool)
* `POST /api/config`  -> partial config update; 409 while running
* `POST /api/start` / `POST /api/stop` -> new state, or 400 + device list
* `GET /api/devices`  -> `{"input": [...], "output": [...]}`
* `GET /api/events`   -> Server-Sent Events, one frame every 500 ms

Anything else is a 404. Local only, so no CORS handling.
"""
from __future__ import annotations

import collections
import json
import math
import random
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np

from . import realtime
from .engines import ENGINES, create_engine
from .keydetect import RollingKeyDetector, estimate_chord
from .tempo import RollingTempoDetector

SAMPLE_RATE = 48000
BLOCKSIZE = 512
SSE_INTERVAL_S = 0.5
MIN_DB = -90.0

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


class ControlPanel:
    """Owns one processor pipeline and the HTTP server in front of it."""

    def __init__(self, port: int = 8765, host: str = "127.0.0.1",
                 demo: bool = False):
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
        }
        try:
            self._state["devices"] = realtime.list_devices()
        except Exception:
            self._state["devices"] = []
        self._index_html: bytes | None = None
        self._server: ThreadingHTTPServer | None = None
        self._server_thread: threading.Thread | None = None
        self._worker: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._in_dev: int | None = None
        self._out_dev: int | None = None

    # -- state ---------------------------------------------------------
    def snapshot(self) -> dict:
        """A JSON-serialisable copy of the state dict."""
        with self._lock:
            st = self._state
            tempo = st["tempo"]
            return {
                **st,
                "meters": dict(st["meters"]),
                "key": dict(st["key"]),
                "tempo": dict(tempo) if tempo is not None else None,
                "devices": list(st["devices"]),
            }

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
        with self._lock:
            if self._state["running"]:
                return 409, {
                    "error": ("กำลังเล่นอยู่ กรุณากดหยุดก่อนเปลี่ยน"
                              "การตั้งค่า"),
                    "state": self.snapshot_locked(),
                }
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
        self._set(**updates)
        return 200, self.snapshot()

    def snapshot_locked(self) -> dict:
        """Copy of state; caller must hold the lock (see `api_config`)."""
        st = self._state
        tempo = st["tempo"]
        return {**st, "meters": dict(st["meters"]),
                "key": dict(st["key"]),
                "tempo": dict(tempo) if tempo is not None else None,
                "devices": list(st["devices"])}

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
            except SystemExit as exc:
                msg = ("ไม่พบอุปกรณ์เสียงที่ระบุ กรุณาเลือกจากรายการ "
                       f"({exc})")
                self._fail(msg)
                return 400, {"error": msg, "devices": self.device_lists()}
            except RuntimeError as exc:
                msg = f"เปิดเสียงสดไม่ได้ ({exc})"
                self._fail(msg)
                return 400, {"error": msg, "devices": self.device_lists()}
            except Exception as exc:  # pragma: no cover - defensive
                msg = f"เปิดเสียงสดไม่ได้ ({exc})"
                self._fail(msg)
                return 400, {"error": msg, "devices": self.device_lists()}
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
        except SystemExit as exc:
            self._fail(str(exc) or "เลือกอุปกรณ์เสียงไม่ถูกต้อง")
        except Exception as exc:
            self._fail(str(exc) or repr(exc))
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
                except Exception:
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
        ring_in = realtime.Ring(BLOCKSIZE * 32)
        ring_out = realtime.Ring(BLOCKSIZE * 32)
        ring_out.write(np.zeros((engine.latency_frames, 2), np.float32))
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
                    except Exception:
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

    def _drain_loop(self, detector: RollingKeyDetector,
                    tempo_detector: RollingTempoDetector,
                    pending: collections.deque) -> None:
        """Feed queued input blocks to the key/tempo detectors until stopped."""
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
            except IndexError:
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
        length = int(self.headers.get("Content-Length") or 0)
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
        elif path == "/api/events":
            self._events(panel)
        else:
            self._json(404, {"error": "ไม่พบเส้นทางนี้ (404)"})

    def do_POST(self) -> None:
        path = urlsplit(self.path).path
        panel: ControlPanel = self.server.panel  # type: ignore[attr-defined]
        if path not in ("/api/config", "/api/start", "/api/stop"):
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
        else:
            code, obj = panel.api_stop()
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
