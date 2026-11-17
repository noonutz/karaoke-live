"""Checks for the local web control panel (demo mode, no audio device).

Run: .venv/bin/python -m unittest discover -s tests -t . -v
"""
from __future__ import annotations

import json
import unittest
import urllib.error
import urllib.request

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
        self.panel = ControlPanel(port=0, demo=True).start_in_background()
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
