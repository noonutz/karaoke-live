"""Live routing with sounddevice/PortAudio.

Two paths:

* `latency_frames == 0` engines (center, channel) run inside the audio
  callback. Nothing to queue, nothing to drift.
* engines that need context (demucs) run in a worker thread behind a ring
  buffer sized with `Ring.for_engine`; the callback only copies. The engine's
  own output already starts with `latency_frames` of silence, so no prefill is
  written, and a starved callback emits silence (counted as an underrun)
  rather than stale audio.

Device selection is by case-insensitive name substring, which is what virtual
cables need: "BlackHole 2ch" on macOS, "CABLE Input" on Windows.
"""
from __future__ import annotations

import threading
import time
from typing import Iterable

import numpy as np

from .engines.base import Engine


def _sd():
    try:
        import sounddevice as sd
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "live mode needs sounddevice: pip install sounddevice "
            "(macOS: brew install portaudio)"
        ) from exc
    return sd


def list_devices() -> list[dict]:
    sd = _sd()
    rows = []
    for index, dev in enumerate(sd.query_devices()):
        hostapi = sd.query_hostapis(dev["hostapi"])["name"]
        rows.append({
            "index": index,
            "name": dev["name"],
            "inputs": dev["max_input_channels"],
            "outputs": dev["max_output_channels"],
            "hostapi": hostapi,
        })
    return rows


def format_devices(rows: Iterable[dict]) -> str:
    lines = [f"{'idx':>3}  {'in':>3} {'out':>3}  {'hostapi':<12} name"]
    for r in rows:
        lines.append(f"{r['index']:>3}  {r['inputs']:>3} {r['outputs']:>3}  "
                     f"{r['hostapi']:<12} {r['name']}")
    return "\n".join(lines)


class DeviceNotFound(Exception):
    """No audio device matches a name query.

    `str(exc)` is a short Thai message for the UI; the full device table
    rides along in `device_table` for the CLI only (`LiveRouter.run`
    prints it).
    """

    def __init__(self, query: str, device_table: str = ""):
        super().__init__(f'ไม่พบอุปกรณ์ชื่อ "{query}"')
        self.query = query
        self.device_table = device_table


def find_device(query: str, want_input: bool) -> int:
    """Resolve a name substring to a device index."""
    sd = _sd()
    original = query.strip()
    query = original.lower()
    rows = list_devices()
    matches = [
        r for r in rows
        if query in r["name"].lower()
        and (r["inputs"] > 0 if want_input else r["outputs"] > 0)
    ]
    if not matches:
        raise DeviceNotFound(original, device_table=format_devices(rows))
    if len(matches) > 1:
        # prefer the default host API entry when several share a name
        try:
            default = sd.query_devices(kind="input" if want_input else "output")["name"]
        except Exception:
            default = ""
        if default:
            matches.sort(key=lambda r: r["name"] != default)
    return matches[0]["index"]


class Ring:
    """Single-producer/single-consumer frame ring."""

    @classmethod
    def for_engine(cls, engine: Engine, blocksize: int,
                   slack_blocks: int = 32) -> "Ring":
        """Capacity for a delayed engine's ring: latency + `slack_blocks`."""
        return cls(engine.latency_frames + blocksize * slack_blocks)

    def __init__(self, capacity_frames: int, channels: int = 2):
        self.capacity = int(capacity_frames)
        self.buf = np.zeros((self.capacity, channels), dtype=np.float32)
        self.write_pos = 0
        self.read_pos = 0
        self.count = 0
        self.lock = threading.Lock()

    def write(self, data: np.ndarray) -> int:
        n = len(data)
        with self.lock:
            space = self.capacity - self.count
            n = min(n, space)
            if n <= 0:
                return 0
            first = min(n, self.capacity - self.write_pos)
            self.buf[self.write_pos:self.write_pos + first] = data[:first]
            rest = n - first
            if rest:
                self.buf[:rest] = data[first:first + rest]
            self.write_pos = (self.write_pos + n) % self.capacity
            self.count += n
            return n

    def read(self, frames: int) -> tuple[np.ndarray, bool]:
        """Return up to `frames` frames; second value is True when short."""
        with self.lock:
            n = min(frames, self.count)
            out = np.zeros((frames, self.buf.shape[1]), dtype=np.float32)
            if n:
                first = min(n, self.capacity - self.read_pos)
                out[:first] = self.buf[self.read_pos:self.read_pos + first]
                rest = n - first
                if rest:
                    out[first:first + rest] = self.buf[:rest]
                self.read_pos = (self.read_pos + n) % self.capacity
                self.count -= n
            return out, n < frames


class LiveRouter:
    def __init__(self, engine: Engine, capture: str, playback: str,
                 sample_rate: int = 48000, blocksize: int = 512,
                 verbose: bool = True):
        self.engine = engine
        self.capture_query = capture
        self.playback_query = playback
        self.sample_rate = sample_rate
        self.blocksize = blocksize
        self.verbose = verbose
        self.xruns = 0
        self.underruns = 0
        self.started = 0.0
        self._stop = threading.Event()

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    def run(self, seconds: float | None = None) -> None:
        # Reset engine state so a reused engine never starts a run with a
        # biquad tail or a half-filled demucs buffer from the previous run.
        self.engine.reset()
        sd = _sd()
        try:
            in_dev = find_device(self.capture_query, want_input=True)
            out_dev = find_device(self.playback_query, want_input=False)
        except DeviceNotFound as exc:
            # CLI only: show the full device table, then exit with the
            # short Thai message (the panel uses `str(exc)` instead).
            if exc.device_table:
                print(exc.device_table)
            raise SystemExit(str(exc)) from exc
        latency = (self.engine.latency_frames + self.blocksize) / self.sample_rate
        self._log(f"capture  : [{in_dev}] {sd.query_devices(in_dev)['name']}")
        self._log(f"playback : [{out_dev}] {sd.query_devices(out_dev)['name']}")
        self._log(f"engine   : {self.engine.name}  vocal level "
                  f"{self.engine.vocal_level * 100:.0f}%  latency ~{latency * 1000:.0f} ms")

        if self.engine.latency_frames == 0:
            stream = sd.Stream(
                samplerate=self.sample_rate, blocksize=self.blocksize,
                device=(in_dev, out_dev), channels=2, dtype="float32",
                latency="low", callback=self._inline_callback,
            )
            self._run_stream(stream, seconds)
            return

        self._ring_in = Ring.for_engine(self.engine, self.blocksize)
        self._ring_out = Ring.for_engine(self.engine, self.blocksize)
        # No prefill: the delayed engine already returns latency_frames of
        # silence at the head of its own output, so prefilling would add the
        # delay twice.
        worker = threading.Thread(target=self._worker, name="separator", daemon=True)
        worker.start()
        stream = sd.Stream(
            samplerate=self.sample_rate, blocksize=self.blocksize,
            device=(in_dev, out_dev), channels=2, dtype="float32",
            latency="high", callback=self._threaded_callback,
        )
        try:
            self._run_stream(stream, seconds)
        finally:
            self._stop.set()
            worker.join(timeout=5)

    def _run_stream(self, stream, seconds: float | None) -> None:
        self.started = time.time()
        with stream:
            try:
                if seconds is None:
                    while True:
                        time.sleep(0.5)
                        self._report()
                else:
                    end = time.time() + seconds
                    while time.time() < end:
                        time.sleep(min(0.5, max(0.0, end - time.time())))
                        self._report()
            except KeyboardInterrupt:
                self._log("stopping")
        self._report(final=True)

    def _report(self, final: bool = False) -> None:
        if not self.verbose:
            return
        elapsed = time.time() - self.started
        if final or elapsed % 5 < 0.6:
            extra = ""
            if self.engine.latency_frames:
                extra = f"  underruns {self.underruns}"
            self._log(f"[{elapsed:6.1f}s] playing, xruns {self.xruns}{extra}")

    # -- callbacks --------------------------------------------------------
    def _inline_callback(self, indata, outdata, frames, _time_info, status):
        if status:
            self.xruns += 1
        outdata[:] = self.engine.process(np.asarray(indata))

    def _threaded_callback(self, indata, outdata, frames, _time_info, status):
        if status:
            self.xruns += 1
        self._ring_in.write(np.asarray(indata))
        data, short = self._ring_out.read(frames)
        if short:
            self.underruns += 1
        outdata[:] = data

    def _worker(self) -> None:
        while not self._stop.is_set():
            if self._ring_in.count >= self.blocksize:
                block, _ = self._ring_in.read(self.blocksize)
                out = self.engine.process(block)
                self._ring_out.write(out)
            else:
                time.sleep(0.002)
