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

    def test_stems_and_bpm_roundtrip(self):
        sl1 = self.make()
        item = sl1.add("/music/b.mp3")
        stems = {stem: f"/out/{stem}.wav"
                 for stem in ("vocals", "drums", "bass", "other")}
        sl1.set_status(item.id, "ready", stems=stems, bpm=128.0)
        sl2 = Setlist(path=self.path)
        back = sl2.get(item.id)
        self.assertEqual(back.status, "ready")
        self.assertEqual(back.stems, stems)
        self.assertEqual(back.bpm, 128.0)
        self.assertIsNone(back.instrumental)
        # The handed-out dict is a copy: mutating it is harmless.
        back.stems["vocals"] = "mutated"
        self.assertEqual(sl2.get(item.id).stems["vocals"], "/out/vocals.wav")
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(data[0]["stems"], stems)
        self.assertEqual(data[0]["bpm"], 128.0)

    def test_old_json_without_stems_still_loads(self):
        sl1 = self.make()
        sl1.add("/music/c.mp3")
        data = json.loads(self.path.read_text(encoding="utf-8"))
        for entry in data:
            entry.pop("stems", None)
            entry.pop("bpm", None)
        self.path.write_text(json.dumps(data), encoding="utf-8")
        back = Setlist(path=self.path).get(sl1.items()[0].id)
        self.assertIsNotNone(back)
        self.assertIsNone(back.stems)
        self.assertIsNone(back.bpm)
        self.assertIsNone(back.instrumental)

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


class SeparateFileTimeoutTests(unittest.TestCase):
    def test_timeout_expired_raises_thai_error(self):
        import subprocess
        from unittest import mock

        def hung(cmd, **kwargs):
            time.sleep(0.3)  # a hung demucs, slower than the timeout below
            raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout"))

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch("karaokelive.prerender.subprocess.run",
                            side_effect=hung) as fake:
                with self.assertRaises(RuntimeError) as ctx:
                    separate_file("/music/song.mp3", tmp,
                                  model="htdemucs", timeout=0.2)
        message = str(ctx.exception)
        self.assertRegex(message, r"[ก-๙]")  # Thai message
        self.assertIn("htdemucs", message)  # names the model
        self.assertIn("/music/song.mp3", message)  # names the file
        _, kwargs = fake.call_args
        self.assertEqual(kwargs.get("timeout"), 0.2)

    def test_slow_separator_failure_marks_failed_with_thai(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "setlist.json"
            sl = Setlist(path=path)
            item = sl.add("/music/song.mp3")

            def stub(_src, _out):
                time.sleep(0.3)  # simulate a hung demucs
                raise RuntimeError(
                    "demucs ใช้เวลานานเกินกำหนด (timeout 0.2 วินาที) "
                    "โมเดล htdemucs ไฟล์ /music/song.mp3")

            worker = PrerenderWorker(sl, tmp, separator=stub, poll_s=0.05)
            worker.start()
            try:
                done = _wait_for(sl, item.id, "failed")
            finally:
                worker.stop(timeout=5)
        self.assertRegex(done.error, r"[ก-๙]")
        self.assertIsNone(done.instrumental)

    def test_stop_defaults_to_bounded_join(self):
        import inspect

        self.assertEqual(
            inspect.signature(PrerenderWorker.stop).parameters["timeout"]
            .default, 5.0)


class CorruptQueueTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.path = self.dir / "setlist.json"

    def test_corrupt_file_is_backed_up_not_wiped(self):
        self.path.write_text("{not valid json!!!", encoding="utf-8")
        sl = Setlist(path=self.path)
        backups = list(self.dir.glob("setlist.json.corrupt-*"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(sl.items(), [])
        self.assertIsNotNone(sl.load_error)
        self.assertRegex(sl.load_error, r"[ก-๙]")
        self.assertIn(backups[0].name, sl.load_error)
        # The corrupt bytes survive in the backup ...
        self.assertIn("not valid", backups[0].read_text(encoding="utf-8"))
        # ... and the next write starts a fresh queue instead of
        # overwriting the backup.
        sl.add("/music/fresh.mp3")
        self.assertTrue(backups[0].is_file())
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(len(data), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
