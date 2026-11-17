"""Checks that tempo detection finds the right BPM from click tracks.

Run: .venv/bin/python -m unittest discover -s tests -t . -v
"""
from __future__ import annotations

import unittest

import numpy as np

from karaokelive.tempo import RollingTempoDetector, detect_tempo

SR = 44100


def click_track(bpm: float, seconds: float = 8.0) -> np.ndarray:
    """Short noise bursts on every beat; stereo with identical channels."""
    n = int(seconds * SR)
    audio = np.zeros(n, dtype=np.float32)
    rng = np.random.default_rng(0)
    burst = (rng.standard_normal(int(0.02 * SR)) * 0.9).astype(np.float32)
    burst *= np.hanning(len(burst))
    step = 60.0 / bpm * SR
    t = 0.0
    while int(t) + len(burst) < n:
        audio[int(t):int(t) + len(burst)] += burst
        t += step
    return np.stack([audio, audio], axis=1)


class DetectTempoTests(unittest.TestCase):
    def test_click_tracks(self):
        for bpm in (90, 120, 128, 174):
            with self.subTest(bpm=bpm):
                result = detect_tempo(click_track(bpm), SR)
                self.assertAlmostEqual(result.bpm, bpm, delta=2.0)

    def test_result_shape(self):
        result = detect_tempo(click_track(120), SR)
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)
        self.assertEqual(result.beat_frames.size, 0)

    def test_mono_input(self):
        result = detect_tempo(click_track(120)[:, 0], SR)
        self.assertAlmostEqual(result.bpm, 120, delta=2.0)

    def test_noise_low_confidence(self):
        rng = np.random.default_rng(1)
        noise = (rng.standard_normal(int(8.0 * SR)) * 0.5).astype(np.float32)
        stereo = np.stack([noise, noise], axis=1)
        result = detect_tempo(stereo, SR)
        self.assertLess(result.confidence, 0.3)


class RollingTempoDetectorTests(unittest.TestCase):
    def test_returns_none_then_a_result(self):
        detector = RollingTempoDetector(SR)
        audio = click_track(120)
        first = detector.feed(audio[:SR // 4])  # 0.25 s: not enough yet
        self.assertIsNone(first)
        result = None
        for start in range(SR // 4, len(audio), 1024):
            result = detector.feed(audio[start:start + 1024])
        self.assertIsNotNone(result)
        assert result is not None
        self.assertAlmostEqual(result.bpm, 120, delta=2.0)

    def test_reset(self):
        detector = RollingTempoDetector(SR)
        audio = click_track(120)
        detector.feed(audio)
        detector.reset()
        self.assertIsNone(detector.feed(audio[:SR // 4]))

    def test_recompute_at_most_once_per_second(self):
        detector = RollingTempoDetector(SR)
        audio = click_track(120)
        results = []
        for start in range(0, len(audio), SR // 8):  # 0.125 s blocks
            results.append(detector.feed(audio[start:start + SR // 8]))
        decided = [r for r in results if r is not None]
        self.assertTrue(decided)
        for r in decided:
            self.assertAlmostEqual(r.bpm, 120, delta=2.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
