"""Checks the OBS overlay route and the tempo/chord demo payload.

Run: .venv/bin/python -m unittest discover -s tests -t . -v
"""
from __future__ import annotations

import json
import time
import unittest
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
    with urllib.request.urlopen(req, timeout=10) as res:
        return res.status, json.loads(res.read().decode("utf-8"))


class OverlayTests(unittest.TestCase):
    def setUp(self):
        # Ephemeral port; demo mode needs no audio device.
        self.panel = ControlPanel(port=0, demo=True).start_in_background()
        self.base = f"http://127.0.0.1:{self.panel.bound_port}"

    def tearDown(self):
        self.panel.shutdown()

    def test_overlay_page(self):
        status, ctype, raw = _get(self.base, "/overlay")
        self.assertEqual(status, 200)
        self.assertIn("text/html", ctype)
        html = raw.decode("utf-8")
        self.assertIn("background:transparent", html)
        self.assertIn("EventSource(", html)

    def test_demo_state_exposes_tempo_and_chord(self):
        status, body = _post(self.base, "/api/start", {})
        self.assertEqual(status, 200)
        try:
            time.sleep(0.3)  # let the demo loop publish a frame
            _status, _ctype, raw = _get(self.base, "/api/state")
            state = json.loads(raw.decode("utf-8"))
            self.assertIn("tempo", state)
            self.assertIn("chord", state)
            self.assertIsNotNone(state["tempo"])
            self.assertIn("bpm", state["tempo"])
            self.assertIn("confidence", state["tempo"])
            self.assertIsNotNone(state["chord"])
        finally:
            _post(self.base, "/api/stop", {})

    def test_events_include_tempo_and_chord(self):
        _status, _body = _post(self.base, "/api/start", {})
        req = urllib.request.Request(self.base + "/api/events")
        try:
            res = urllib.request.urlopen(req, timeout=10)
            try:
                frame = b""
                for _ in range(200):
                    line = res.readline()
                    if not line:
                        break
                    frame += line
                    if line.startswith(b"data:"):
                        break
                self.assertTrue(frame.startswith(b"data:"),
                                f"no SSE frame received: {frame[:80]!r}")
                payload = json.loads(frame[len(b"data:"):].decode("utf-8"))
                self.assertIn("tempo", payload)
                self.assertIn("chord", payload)
            finally:
                res.close()
        finally:
            _post(self.base, "/api/stop", {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
