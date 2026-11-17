"""Checks for the setlist queue and the pre-render worker.

Run: .venv/bin/python -m unittest discover -s tests -t . -v
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
import tempfile
import time
import unittest
from pathlib import Path

from karaokelive.prerender import PrerenderWorker, separate_file
from karaokelive.setlist import Item, Setlist, default_state_dir

HEX8 = re.compile(r"^[0-9a-f]{8}$")


def _wait_for(sl: Setlist, id: str, status: str, timeout: float = 10.0) -> Item:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        item = sl.get(id)
        if item is not None and item.status == status:
            return item
        time.sleep(0.02)
    current = sl.get(id)
    raise AssertionError(f"{id} never reached {status!r} "
                         f"(still {current.status if current else None!r})")


class SetlistTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.path = self.dir / "setlist.json"

    def make(self) -> Setlist:
        return Setlist(path=self.path)

    def test_add_list_remove_roundtrip(self):
        sl = self.make()
        one = sl.add("/music/one.mp3")
        two = sl.add("/music/two.mp3", title="Two")
        self.assertTrue(HEX8.match(one.id), f"bad id {one.id!r}")
        self.assertEqual(one.title, "one")  # file stem by default
        self.assertEqual(two.title, "Two")
        self.assertEqual(one.status, "pending")
        self.assertEqual([item.id for item in sl.items()], [one.id, two.id])
        self.assertTrue(sl.remove(one.id))
        self.assertFalse(sl.remove(one.id))
        self.assertIsNone(sl.get(one.id))
        self.assertEqual([item.id for item in sl.items()], [two.id])

    def test_duplicate_source_rejected(self):
        sl = self.make()
        sl.add("/music/same.mp3")
        with self.assertRaises(ValueError):
            sl.add("/music/same.mp3")

    def test_url_source_stored_as_given(self):
        sl = self.make()
        url = "https://example.com/gig/mix.mp3?token=abc"
        item = sl.add(url)
        self.assertEqual(item.source, url)
        self.assertEqual(item.title, "mix")
        self.assertIsNone(item.duration_s)
        with self.assertRaises(ValueError):
            sl.add(url)

    def test_add_never_raises_on_unreadable_source(self):
        sl = self.make()
        item = sl.add("/definitely/not/here.mp3")
        self.assertIsNone(item.duration_s)

    def test_persistence_across_instances(self):
        sl1 = self.make()
        item = sl1.add("/music/a.mp3", title="เพลง ก")
        sl1.set_status(item.id, "ready", instrumental="/out/no_vocals.wav")
        sl2 = Setlist(path=self.path)
        back = sl2.get(item.id)
        self.assertIsNotNone(back)
        self.assertEqual(back.id, item.id)  # stable across load
        self.assertEqual(back.title, "เพลง ก")
        self.assertEqual(back.status, "ready")
        self.assertEqual(back.instrumental, "/out/no_vocals.wav")

    def test_atomic_write_leaves_no_tmp(self):
        sl = self.make()
        for i in range(3):
            sl.add(f"/music/s{i}.mp3")
        sl.remove(sl.items()[0].id)
        self.assertEqual(list(self.dir.glob("*.tmp")), [])
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(len(data), 2)

    def test_next_pending_skips_ready_and_failed(self):
        sl = self.make()
        a = sl.add("/a.mp3")
        b = sl.add("/b.mp3")
        c = sl.add("/c.mp3")
        self.assertEqual(sl.next_pending().id, a.id)  # oldest first
        sl.set_status(a.id, "ready", instrumental="/x.wav")
        sl.set_status(b.id, "failed", error="boom")
        self.assertEqual(sl.next_pending().id, c.id)
        sl.set_status(c.id, "rendering")
        self.assertIsNone(sl.next_pending())

    def test_reads_return_copies(self):
        sl = self.make()
        item = sl.add("/music/x.mp3")
        sl.items()[0].title = "mutated"
        self.assertEqual(sl.get(item.id).title, "x")

    def test_clear(self):
        sl = self.make()
        a = sl.add("/a.mp3")
        sl.add("/b.mp3")
        sl.set_status(a.id, "ready", instrumental="/x.wav")
        self.assertEqual(sl.clear(status="ready"), 1)
        self.assertEqual([item.id for item in sl.items()], [sl.items()[0].id])
        self.assertEqual(sl.clear(), 1)
        self.assertEqual(sl.items(), [])

    def test_bad_status_rejected(self):
        sl = self.make()
        item = sl.add("/a.mp3")
        with self.assertRaises(ValueError):
            sl.set_status(item.id, "nope")
        with self.assertRaises(KeyError):
            sl.set_status("deadbeef", "ready")


class DefaultStateDirTests(unittest.TestCase):
    def test_respects_xdg_data_home(self):
        if not sys.platform.startswith("linux"):
            self.skipTest("XDG only applies on Linux")
        old = os.environ.get("XDG_DATA_HOME")
        try:
            with tempfile.TemporaryDirectory() as tmp:
                os.environ["XDG_DATA_HOME"] = tmp
                self.assertEqual(default_state_dir(), Path(tmp) / "karaokelive")
        finally:
            if old is None:
                os.environ.pop("XDG_DATA_HOME", None)
            else:
                os.environ["XDG_DATA_HOME"] = old

    def test_falls_back_without_xdg(self):
        if not sys.platform.startswith("linux"):
            self.skipTest("XDG only applies on Linux")
        old = os.environ.pop("XDG_DATA_HOME", None)
        try:
            self.assertEqual(default_state_dir(),
                             Path.home() / ".local" / "share" / "karaokelive")
        finally:
            if old is not None:
                os.environ["XDG_DATA_HOME"] = old


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def make(self) -> Setlist:
        return Setlist(path=self.dir / "setlist.json")

    def test_marks_ready_and_stores_path(self):
        sl = self.make()
        item = sl.add("/music/song.mp3")
        target = str(self.dir / "stems" / "no_vocals.wav")
        worker = PrerenderWorker(sl, str(self.dir / "stems"),
                                 separator=lambda _src, _out: target,
                                 poll_s=0.05)
        worker.start()
        try:
            done = _wait_for(sl, item.id, "ready")
        finally:
            worker.stop()
        self.assertFalse(worker.is_alive())
        self.assertEqual(done.instrumental, target)
        self.assertIsNone(done.error)

    def test_failure_marks_failed_and_does_not_retry(self):
        sl = self.make()
        item = sl.add("/music/song.mp3")
        calls: list = []

        def stub(_src, _out):
            calls.append(1)
            raise RuntimeError("demucs exploded")

        worker = PrerenderWorker(sl, str(self.dir), separator=stub, poll_s=0.05)
        worker.start()
        try:
            done = _wait_for(sl, item.id, "failed")
            time.sleep(0.2)  # let the loop spin; a retry would call again
        finally:
            worker.stop()
        self.assertIn("demucs exploded", done.error)
        self.assertIsNone(done.instrumental)
        self.assertEqual(len(calls), 1)

    def test_long_error_message_is_truncated(self):
        sl = self.make()
        item = sl.add("/music/song.mp3")

        def stub(_src, _out):
            raise ValueError("x" * 1000)

        worker = PrerenderWorker(sl, str(self.dir), separator=stub, poll_s=0.05)
        worker.start()
        try:
            done = _wait_for(sl, item.id, "failed")
        finally:
            worker.stop()
        self.assertLessEqual(len(done.error), 300)

    def test_stop_is_prompt(self):
        sl = self.make()  # empty queue: the worker just idles
        worker = PrerenderWorker(sl, str(self.dir),
                                 separator=lambda _src, _out: _out,
                                 poll_s=5.0)
        worker.start()
        time.sleep(0.05)
        started = time.monotonic()
        worker.stop()
        self.assertFalse(worker.is_alive())
        self.assertLess(time.monotonic() - started, 1.0)


@unittest.skipIf(importlib.util.find_spec("demucs") is not None,
                 "demucs is installed")
class SeparateFileTests(unittest.TestCase):
    def test_missing_demucs_raises_runtime_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                separate_file("/nonexistent/song.mp3", tmp)


if __name__ == "__main__":
    unittest.main(verbosity=2)
