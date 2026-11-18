"""Checks for the local web control panel (demo mode, no audio device).

Run: .venv/bin/python -m unittest discover -s tests -t . -v
"""
from __future__ import annotations

import json
import re
import tempfile
import unittest
import urllib.error
import urllib.request

from karaokelive.setlist import Setlist
from karaokelive.ui import ControlPanel


def _get(base: str, path: str):
    with urllib.request.urlopen(base + path, timeout=10) as res:
        return res.status, res.headers.get_content_type(), res.read()


def _post(base: str, path: str, body: dict):
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        base + path, data=data,
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as res:
            payload = json.loads(res.read().decode("utf-8"))
            return res.status, payload
    except urllib.error.HTTPError as exc:
        try:
            payload = json.loads(exc.read().decode("utf-8"))
        finally:
            exc.close()
        return exc.code, payload


class PanelTests(unittest.TestCase):
    def setUp(self):
        # Ephemeral port; demo mode needs no audio device.
        # Each test gets its own throwaway setlist file.
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.setlist = Setlist(
            path=f"{self._tmp.name}/setlist.json")
        self.panel = ControlPanel(port=0, demo=True,
                                  setlist=self.setlist).start_in_background()
        self.base = f"http://127.0.0.1:{self.panel.bound_port}"

    def tearDown(self):
        self.panel.shutdown()

    def test_state(self):
        status, ctype, raw = _get(self.base, "/api/state")
        self.assertEqual(status, 200)
        self.assertIn("application/json", ctype)
        state = json.loads(raw.decode("utf-8"))
        for field in ("capture", "output", "engine", "vocal_level",
                      "bass_hz", "mono_guard", "running", "latency_ms",
                      "error", "meters", "key", "devices"):
            self.assertIn(field, state)
        self.assertIn("in_db", state["meters"])
        self.assertIn("camelot", state["key"])
        self.assertTrue(state["demo"])  # demo panel exposes demo in /api/state

    def test_devices(self):
        status, _ctype, raw = _get(self.base, "/api/devices")
        self.assertEqual(status, 200)
        body = json.loads(raw.decode("utf-8"))
        self.assertIn("input", body)
        self.assertIn("output", body)

    def test_config_round_trip(self):
        status, body = _post(self.base, "/api/config",
                             {"engine": "channel", "vocal_level": 0.5,
                              "capture": "demo-in", "output": "demo-out",
                              "bass_hz": 120.0, "mono_guard": False})
        self.assertEqual(status, 200)
        _status, _ctype, raw = _get(self.base, "/api/state")
        state = json.loads(raw.decode("utf-8"))
        self.assertEqual(state["engine"], "channel")
        self.assertAlmostEqual(state["vocal_level"], 0.5)
        self.assertEqual(state["capture"], "demo-in")
        self.assertEqual(state["output"], "demo-out")
        self.assertAlmostEqual(state["bass_hz"], 120.0)
        self.assertFalse(state["mono_guard"])

    def test_config_rejected_while_running(self):
        status, _body = _post(self.base, "/api/start", {})
        self.assertEqual(status, 200)
        try:
            status, body = _post(self.base, "/api/config",
                                 {"vocal_level": 0.1})
            self.assertEqual(status, 409)
            self.assertIn("error", body)
        finally:
            _post(self.base, "/api/stop", {})

    def test_start_stop_demo(self):
        status, body = _post(self.base, "/api/start", {})
        self.assertEqual(status, 200)
        self.assertTrue(body["running"])
        status, body = _post(self.base, "/api/stop", {})
        self.assertEqual(status, 200)
        self.assertFalse(body["running"])

    def test_events_stream(self):
        # The SSE endpoint must emit at least one frame, then end on close.
        req = urllib.request.Request(self.base + "/api/events")
        res = urllib.request.urlopen(req, timeout=10)
        try:
            self.assertIn("text/event-stream",
                          res.headers.get_content_type())
            frame = b""
            for _ in range(200):  # up to ~10 s of 50 ms server ticks
                line = res.readline()
                if not line:
                    break
                frame += line
                if line.startswith(b"data:"):
                    break
            self.assertTrue(frame.startswith(b"data:"),
                            f"no SSE frame received: {frame[:80]!r}")
            payload = json.loads(frame[len(b"data:"):].decode("utf-8"))
            for field in ("meters", "key", "running", "error", "latency_ms"):
                self.assertIn(field, payload)
        finally:
            res.close()

    def test_index_icons_and_buttons(self):
        # Redesigned panel: inline Phosphor SVGs (no emoji), all buttons named.
        _status, ctype, raw = _get(self.base, "/")
        self.assertIn("text/html", ctype)
        html = raw.decode("utf-8")
        svgs = re.findall(r"<svg\b", html)
        self.assertGreaterEqual(len(svgs), 8,
                                f"expected >= 8 inline SVGs, found {len(svgs)}")
        self.assertNotRegex(html, "[\U0001F000-\U0001FAFF☀-➿⬀-⯿]")
        buttons = re.findall(r"<button\b([^>]*)>(.*?)</button>",
                             html, re.DOTALL | re.IGNORECASE)
        self.assertTrue(buttons, "expected at least one <button>")
        for attrs, inner in buttons:
            text = re.sub(r"<[^>]+>", "", inner).strip()
            labelled = bool(text) or "aria-label" in attrs.lower()
            self.assertTrue(labelled, f"button without label: {attrs!r}")

    def _state(self):
        status, _ctype, raw = _get(self.base, "/api/state")
        self.assertEqual(status, 200)
        return json.loads(raw.decode("utf-8"))

    def test_queue_add_list_remove_roundtrip(self):
        status, _ctype, raw = _get(self.base, "/api/queue")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(raw.decode("utf-8")), {"items": []})

        status, body = _post(self.base, "/api/queue/add",
                             {"source": "/music/one.mp3", "title": "หนึ่ง"})
        self.assertEqual(status, 200)
        item = body["item"]
        self.assertEqual(item["title"], "หนึ่ง")
        self.assertEqual(item["status"], "pending")

        state = self._state()
        self.assertEqual(len(state["queue"]), 1)
        self.assertEqual(state["queue"][0]["id"], item["id"])

        # Duplicate source is a 400 with a Thai message.
        status, body = _post(self.base, "/api/queue/add",
                             {"source": "/music/one.mp3"})
        self.assertEqual(status, 400)
        self.assertIn("error", body)
        self.assertRegex(body["error"], r"[ก-๙]")

        # Removing an unknown id still returns the remaining items.
        status, body = _post(self.base, "/api/queue/remove",
                             {"id": "deadbeef"})
        self.assertEqual(status, 200)
        self.assertEqual(len(body["items"]), 1)
        status, body = _post(self.base, "/api/queue/remove",
                             {"id": item["id"]})
        self.assertEqual(status, 200)
        self.assertEqual(body["items"], [])

        state = self._state()
        self.assertEqual(state["queue"], [])

    def test_queue_play_guards(self):
        status, body = _post(self.base, "/api/queue/add",
                             {"source": "/music/pending.mp3"})
        self.assertEqual(status, 200)
        pending_id = body["item"]["id"]

        # Pending items cannot play: 409 + Thai message.
        status, body = _post(self.base, "/api/queue/play",
                             {"id": pending_id})
        self.assertEqual(status, 409)
        self.assertIn("error", body)
        self.assertRegex(body["error"], r"[ก-๙]")

        self.assertIsNone(self._state()["now_playing"])

        # Stop is idempotent even when nothing plays.
        status, body = _post(self.base, "/api/queue/stop", {})
        self.assertEqual(status, 200)
        self.assertIsNone(self._state()["now_playing"])

    def test_queue_drawer_markup(self):
        _status, ctype, raw = _get(self.base, "/")
        self.assertIn("text/html", ctype)
        html = raw.decode("utf-8")
        self.assertIn('id="btnQueue"', html)
        self.assertIn("คิวเพลง", html)
        self.assertIn('id="queueDrawer"', html)
        self.assertIn('role="dialog"', html)
        self.assertIn('aria-modal="true"', html)
        self.assertIn('id="queueScrim"', html)
        self.assertIn("วาง path ไฟล์เสียง", html)
        self.assertIn("แยกเสียงเพลงที่รอ", html)
        self.assertIn("คิวเพลง (", html)

    def test_unknown_paths_404(self):
        try:
            urllib.request.urlopen(self.base + "/nope", timeout=10)
        except urllib.error.HTTPError as exc:
            self.assertEqual(exc.code, 404)
            exc.close()
        else:
            self.fail("expected 404")
        _status, _body = _post(self.base, "/api/nope", {})
        self.assertEqual(_status, 404)

    def _raw_post(self, path: str, payload: bytes, headers: dict):
        import http.client
        from urllib.parse import urlsplit

        parts = urlsplit(self.base)
        conn = http.client.HTTPConnection(parts.hostname, parts.port,
                                          timeout=10)
        try:
            conn.request("POST", path, body=payload, headers=headers)
            res = conn.getresponse()
            return res.status, json.loads(res.read().decode("utf-8"))
        finally:
            conn.close()

    def test_garbage_content_length_is_400(self):
        status, body = self._raw_post(
            "/api/config", b"{}",
            {"Content-Type": "application/json", "Content-Length": "abc"})
        self.assertEqual(status, 400)
        self.assertRegex(body["error"], r"[ก-๙]")

    def test_oversize_declared_body_is_413_without_reading(self):
        status, body = self._raw_post(
            "/api/config", b"{}",
            {"Content-Type": "application/json",
             "Content-Length": str(2 * 1024 * 1024)})
        self.assertEqual(status, 413)
        self.assertIn("1 MB", body["error"])
        self.assertRegex(body["error"], r"[ก-๙]")

    def test_config_raced_with_start_still_409(self):
        # Call the panel methods directly (no device needed): a start that
        # lands before the config write must yield a Thai 409.
        panel = self.panel
        panel._set(running=True)  # a start landed first
        try:
            code, body = panel.api_config({"vocal_level": 0.1})
        finally:
            panel._set(running=False)  # leave the shared panel idle
        self.assertEqual(code, 409)
        self.assertRegex(body["error"], r"[ก-๙]")

    def test_config_check_and_write_share_one_lock_hold(self):
        # Atomicity: the running check and the write happen under a single
        # lock hold, so a start cannot slip between them. Deterministic:
        # exactly one acquire/release pair by this thread during the call.
        import threading as _threading

        panel = self.panel
        panel._set(running=False)

        class RecordingLock:
            def __init__(self):
                self._inner = _threading.Lock()
                self.events: list = []  # (thread id, "acquire"/"release")

            def __enter__(self):
                self._inner.acquire()
                self.events.append((_threading.get_ident(), "acquire"))
                return self

            def __exit__(self, *exc):
                self.events.append((_threading.get_ident(), "release"))
                self._inner.release()
                return False

        real_lock = panel._lock
        rec = RecordingLock()
        panel._lock = rec
        stop = _threading.Event()

        def racer():
            while not stop.is_set():
                panel._set(running=True)
                panel._set(running=False)

        other = _threading.Thread(target=racer, daemon=True)
        other.start()
        try:
            me = _threading.get_ident()
            before = len(rec.events)
            code, body = panel.api_config({"vocal_level": 0.5})
            mine = [e for e in rec.events[before:] if e[0] == me]
        finally:
            stop.set()
            other.join(timeout=10)
            panel._lock = real_lock
            panel._set(running=False)
        self.assertEqual(mine, [(me, "acquire"), (me, "release")])
        self.assertIn(code, (200, 409))
        if code == 409:
            self.assertRegex(body["error"], r"[ก-๙]")


if __name__ == "__main__":
    unittest.main(verbosity=2)
