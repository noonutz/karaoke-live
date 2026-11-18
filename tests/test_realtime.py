"""Checks for the live-audio ring sizing, failure reporting, and resets.

Run: .venv/bin/python -m unittest discover -s tests -t . -v
"""
from __future__ import annotations

import tempfile
import unittest
from unittest import mock

import numpy as np

from karaokelive import realtime
from karaokelive.setlist import Setlist
from karaokelive.ui import ControlPanel


class _DelayedStub:
    """Demucs-shaped stub: delayed output, no model needed."""

    name = "delayed-stub"
    latency_frames = 121920  # default demucs chunk at 48 kHz
    vocal_level = 0.85

    def __init__(self):
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1

    def process(self, block: np.ndarray) -> np.ndarray:
        return np.asarray(block, dtype=np.float32)


class _ZeroStub(_DelayedStub):
    name = "zero-stub"
    latency_frames = 0


class RingTests(unittest.TestCase):
    def test_for_engine_fits_the_latency_prefill(self):
        engine = _DelayedStub()
        blocksize = 512
        ring = realtime.Ring.for_engine(engine, blocksize)
        self.assertGreaterEqual(
            ring.capacity, engine.latency_frames + blocksize * 32)
        written = ring.write(
            np.zeros((engine.latency_frames, 2), dtype=np.float32))
        self.assertEqual(written, engine.latency_frames,
                         "latency prefill must not be truncated")

    def test_plain_ring_still_truncates_oversize_writes(self):
        ring = realtime.Ring(100)
        written = ring.write(np.zeros((300, 2), dtype=np.float32))
        self.assertEqual(written, 100)


class EngineFailureTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.panel = ControlPanel(
            port=0, demo=True,
            setlist=Setlist(path=f"{self._tmp.name}/setlist.json"))
        self.addCleanup(self.panel.shutdown)

    def test_five_consecutive_failures_trip(self):
        self.panel._set(running=True)
        for _ in range(4):
            self.panel._engine_failure(RuntimeError("boom"))
            self.assertIsNone(self.panel.snapshot()["error"])
        self.panel._engine_failure(RuntimeError("boom"))
        state = self.panel.snapshot()
        self.assertIsInstance(state["error"], str)
        self.assertIn("แยกเสียงไม่สำเร็จ", state["error"])
        self.assertFalse(state["running"])
        self.assertTrue(self.panel._stop_event.is_set())

    def test_success_resets_the_counter(self):
        self.panel._set(running=True)
        for _ in range(4):
            self.panel._engine_failure(RuntimeError("boom"))
        self.panel._engine_ok()  # one successful block
        for _ in range(4):
            self.panel._engine_failure(RuntimeError("boom"))
        state = self.panel.snapshot()
        self.assertIsNone(state["error"])
        self.assertTrue(state["running"])
        self.panel._engine_failure(RuntimeError("boom"))  # 5th in a row
        self.assertIn("แยกเสียงไม่สำเร็จ",
                      self.panel.snapshot()["error"])


class DummyStream:
    def __init__(self, events: list):
        self._events = events

    def __enter__(self):
        self._events.append("enter")
        return self

    def __exit__(self, *args):
        return False


class FakeSD:
    """Stand-in for the sounddevice module (no audio device needed)."""

    def __init__(self, events: list):
        self._events = events

    def query_devices(self, kind=None):
        table = [
            {"name": "Fake In", "max_input_channels": 2,
             "max_output_channels": 0, "hostapi": 0},
            {"name": "Fake Out", "max_input_channels": 0,
             "max_output_channels": 2, "hostapi": 0},
        ]
        if kind == "input":
            return {"name": "Fake In"}
        if kind == "output":
            return {"name": "Fake Out"}
        if isinstance(kind, int):
            return table[kind]
        return table

    def query_hostapis(self, index):
        return {"name": "FakeAPI"}

    def Stream(self, **kwargs):
        return DummyStream(self._events)


class LiveRouterResetTests(unittest.TestCase):
    def test_run_resets_engine_before_starting_audio(self):
        engine = _ZeroStub()
        events: list = []
        original_reset = engine.reset

        def _reset_and_log():
            events.append("reset")
            original_reset()

        router = realtime.LiveRouter(
            engine, capture="in", playback="out", verbose=False)
        with mock.patch.object(engine, "reset", _reset_and_log):
            with mock.patch.object(realtime, "_sd",
                                   return_value=FakeSD(events)):
                router.run(seconds=0)
        self.assertEqual(engine.resets, 1)
        self.assertEqual(events, ["reset", "enter"])


if __name__ == "__main__":
    unittest.main()
