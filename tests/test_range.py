"""Checks for vocal-range estimation and song-fit advice.

Run: .venv/bin/python -m unittest discover -s tests -t . -v
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from karaokelive.range import (
    Profile,
    estimate_range,
    fit_song,
    midi_to_name,
    name_to_midi,
)

SR = 44100


def midi_freq(midi: float) -> float:
    return 440.0 * 2.0 ** ((midi - 69) / 12)


def note(midi: int, seconds: float = 0.5) -> np.ndarray:
    """One tone with a second harmonic: 0.2*(sin + 0.3*sin2x)."""
    t = np.arange(int(seconds * SR)) / SR
    freq = midi_freq(midi)
    wave = np.sin(2.0 * np.pi * freq * t) + 0.3 * np.sin(4.0 * np.pi * freq * t)
    return (0.2 * wave).astype(np.float32)


def melody(midis: list[int]) -> np.ndarray:
    """Concatenated 0.5 s tones, mono."""
    return np.concatenate([note(m, 0.5) for m in midis]).astype(np.float32)


class EstimateRangeTests(unittest.TestCase):
    def test_a3_to_e4(self):
        audio = melody([57, 59, 60, 62, 64])  # A3 B3 C4 D4 E4
        result = estimate_range(audio, SR)
        self.assertLessEqual(abs(result.low_midi - 57), 1.0)
        self.assertLessEqual(abs(result.high_midi - 64), 1.0)
        self.assertGreater(result.voiced_ratio, 0.5)

    def test_c4_to_c5(self):
        audio = melody([60, 62, 64, 65, 67, 69, 71, 72])
        result = estimate_range(audio, SR)
        self.assertLessEqual(abs(result.low_midi - 60), 1.0)
        self.assertLessEqual(abs(result.high_midi - 72), 1.0)

    def test_wide_a2_to_a4(self):
        audio = melody([45, 48, 50, 52, 55, 57, 60, 62, 64, 67, 69])
        result = estimate_range(audio, SR)
        self.assertLessEqual(abs(result.low_midi - 45), 1.0)
        self.assertLessEqual(abs(result.high_midi - 69), 1.0)
        self.assertAlmostEqual(result.semitones(),
                               result.high_midi - result.low_midi)

    def test_stereo_input(self):
        mono = melody([57, 59, 60, 62, 64])
        stereo = np.stack([mono, mono], axis=1)
        result = estimate_range(stereo, SR)
        self.assertLessEqual(abs(result.low_midi - 57), 1.0)
        self.assertLessEqual(abs(result.high_midi - 64), 1.0)

    def test_white_noise_raises(self):
        rng = np.random.default_rng(0)
        noise = rng.standard_normal(SR * 2).astype(np.float32)
        with self.assertRaises(ValueError):
            estimate_range(noise, SR)


class NoteNameTests(unittest.TestCase):
    def test_round_trip(self):
        for name, midi in (("C4", 60), ("F#3", 54), ("Bb4", 70), ("A0", 21)):
            self.assertAlmostEqual(name_to_midi(name), float(midi))
            # Flats come back as the enharmonic sharp (Bb4 -> A#4).
            self.assertAlmostEqual(
                name_to_midi(midi_to_name(float(midi))), float(midi))
        self.assertEqual(midi_to_name(60), "C4")
        self.assertEqual(midi_to_name(54), "F#3")
        self.assertEqual(midi_to_name(21), "A0")

    def test_bad_name_raises(self):
        with self.assertRaises(ValueError):
            name_to_midi("H2")


class ProfileTests(unittest.TestCase):
    def _patched(self, tmp: str):
        return mock.patch("karaokelive.range.default_state_dir",
                          return_value=Path(tmp))

    def test_save_load_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self._patched(tmp):
                saved = Profile(low=45.0, high=72.0).save()
                self.assertTrue((Path(tmp) / "profile.json").is_file())
                loaded = Profile.load()
                assert loaded is not None
                self.assertAlmostEqual(loaded.low, 45.0)
                self.assertAlmostEqual(loaded.high, 72.0)
                self.assertEqual(saved.as_dict().keys(),
                                 {"low", "high", "updated_at"})

    def test_load_missing_is_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self._patched(tmp):
                self.assertIsNone(Profile.load())

    def test_semitones_for(self):
        profile = Profile(low=45.0, high=72.0)  # A2-C5
        self.assertEqual(profile.semitones_for(57.0, 64.0), 0)
        self.assertTrue(profile.fits(57.0, 64.0))
        # Too high: smallest fitting shift is negative.
        shift = profile.semitones_for(57.0, 76.0)  # A3-E5
        self.assertEqual(shift, -4)
        self.assertTrue(profile.fits(57.0 + shift, 76.0 + shift))
        self.assertTrue(profile.fits(57.0 + shift + 1, 76.0 + shift + 1)
                        or shift == -4)
        # Too low: smallest fitting shift is positive.
        shift_low = profile.semitones_for(36.0, 55.0)
        self.assertGreater(shift_low, 0)
        self.assertTrue(profile.fits(36.0 + shift_low, 55.0 + shift_low))

    def test_fit_song_verdict_matches_shift(self):
        profile = Profile(low=45.0, high=72.0)
        ok = fit_song(57.0, 64.0, profile)
        self.assertEqual(ok.shift, 0)
        self.assertEqual(ok.verdict, "ok")
        self.assertEqual(ok.message, "ร้องได้สบาย")
        high = fit_song(57.0, 76.0, profile)
        self.assertLess(high.shift, 0)
        self.assertEqual(high.verdict, "too_high")
        self.assertIn("สูงเกิน", high.message)
        self.assertIn("ลด", high.message)
        low = fit_song(36.0, 55.0, profile)
        self.assertGreater(low.shift, 0)
        self.assertEqual(low.verdict, "too_low")
        self.assertIn("ต่ำเกิน", low.message)
        self.assertIn("เพิ่ม", low.message)
        # Verdict is ok exactly when no shift is needed.
        for song in ((57.0, 64.0), (57.0, 76.0), (36.0, 55.0)):
            fit = fit_song(*song, profile)
            self.assertEqual(fit.verdict == "ok", fit.shift == 0)


class PanelDemoTests(unittest.TestCase):
    def test_demo_state_exposes_range_fit(self):
        from karaokelive.setlist import Setlist
        from karaokelive.ui import ControlPanel

        with tempfile.TemporaryDirectory() as tmp:
            panel = ControlPanel(
                port=0, demo=True,
                setlist=Setlist(path=f"{tmp}/setlist.json"))
            try:
                state = panel.snapshot()
            finally:
                panel.shutdown()
        fit = state.get("range_fit")
        self.assertIsNotNone(fit)
        assert fit is not None
        for key in ("song_low", "song_high", "shift", "verdict", "message"):
            self.assertIn(key, fit)
        self.assertLess(fit["shift"], 0)
        self.assertEqual(fit["verdict"], "too_high")


if __name__ == "__main__":
    unittest.main(verbosity=2)
