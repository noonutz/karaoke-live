"""Checks that key detection finds the right key from synthesised chords.

Run: .venv/bin/python -m unittest discover -s tests -t . -v
"""
from __future__ import annotations

import re
import unittest

import numpy as np

from karaokelive.keydetect import RollingKeyDetector, detect_key

SR = 44100


def midi_freq(midi: int) -> float:
    return 440.0 * 2.0 ** ((midi - 69) / 12)


def chord(root_midi: int, seconds: float, minor: bool = False) -> np.ndarray:
    """One triad: root + third + fifth, equal sine tones."""
    third = root_midi + (3 if minor else 4)
    notes = (root_midi, third, root_midi + 7)
    t = np.arange(int(seconds * SR)) / SR
    mix = sum(0.2 * np.sin(2 * np.pi * midi_freq(m) * t) for m in notes)
    return mix.astype(np.float32)


def progression(roots: list[int], minor: bool = False) -> np.ndarray:
    """One second per triad, concatenated; stereo with identical channels."""
    audio = np.concatenate([chord(root, 1.0, minor=minor) for root in roots])
    return np.stack([audio, audio], axis=1).astype(np.float32)


class DetectKeyTests(unittest.TestCase):
    def test_c_major_progression(self):
        # C F G C triads: C-E-G, F-A-C, G-B-D, C-E-G
        audio = progression([60, 65, 67, 60])
        result = detect_key(audio, SR)
        self.assertEqual((result.key, result.mode), ("C", "major"))

    def test_a_minor_progression(self):
        # Am Dm Em Am triads: A-C-E, D-F-A, E-G-B, A-C-E
        audio = progression([57, 62, 64, 57], minor=True)
        result = detect_key(audio, SR)
        self.assertEqual((result.key, result.mode), ("A", "minor"))

    def test_e_major_progression(self):
        # E A B E triads: guards against octave errors on the low E
        audio = progression([64, 69, 71, 64])
        result = detect_key(audio, SR)
        self.assertEqual((result.key, result.mode), ("E", "major"))

    def test_result_shape(self):
        audio = progression([60, 65, 67, 60])
        result = detect_key(audio, SR)
        self.assertAlmostEqual(float(result.chroma.sum()), 1.0, delta=1e-6)
        self.assertRegex(result.camelot, r"^([1-9]|1[0-2])[AB]$")
        self.assertEqual(len(result.alternatives), 2)

    def test_mono_input(self):
        audio = progression([60, 65, 67, 60])[:, 0]
        result = detect_key(audio, SR)
        self.assertEqual((result.key, result.mode), ("C", "major"))


class RollingKeyDetectorTests(unittest.TestCase):
    def test_returns_none_then_a_result(self):
        detector = RollingKeyDetector(SR)
        audio = progression([60, 65, 67, 60])
        first = detector.feed(audio[:SR // 4])
        self.assertIsNone(first)
        result = None
        for start in range(SR // 4, len(audio), 1024):
            result = detector.feed(audio[start:start + 1024])
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual((result.key, result.mode), ("C", "major"))

    def test_reset(self):
        detector = RollingKeyDetector(SR)
        audio = progression([60, 65, 67, 60])
        detector.feed(audio)
        detector.reset()
        self.assertIsNone(detector.feed(audio[:SR // 4]))

    def test_camelot_format(self):
        self.assertTrue(re.match(r"^([1-9]|1[0-2])[AB]$",
                                 detect_key(progression([57, 62, 64, 57],
                                                        minor=True), SR).camelot))


if __name__ == "__main__":
    unittest.main(verbosity=2)
