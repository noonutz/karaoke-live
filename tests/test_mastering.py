"""Checks for the output safety limiter (karaokelive.mastering.Limiter).

Run: .venv/bin/python -m unittest discover -s tests -t . -v
"""
from __future__ import annotations

import unittest

import numpy as np

from karaokelive.mastering import Limiter

SR = 48000
THRESHOLD = 10.0 ** (-1.0 / 20.0)  # default -1 dBFS threshold, linear


def const(value: float, frames: int = 512) -> np.ndarray:
    """One constant stereo block."""
    return np.full((frames, 2), value, dtype=np.float32)


class LimiterTests(unittest.TestCase):
    def test_quiet_block_passes_unchanged(self):
        amp = 10.0 ** (-12.0 / 20.0)  # -12 dBFS, well under the threshold
        block = const(amp)
        limiter = Limiter(SR)
        out = limiter.process(block)
        self.assertEqual(limiter.gain_reduction_db, 0.0)
        self.assertLess(float(np.max(np.abs(out - block))), 1e-6,
                        "a -12 dBFS block must pass through untouched")

    def test_hot_step_never_exceeds_full_scale(self):
        amp = THRESHOLD * 10.0 ** (12.0 / 20.0)  # +12 dB over the threshold
        limiter = Limiter(SR)
        out = limiter.process(const(amp))
        self.assertLessEqual(float(np.max(np.abs(out))), 1.0,
                             "the true peak must never exceed 1.0")
        self.assertLess(limiter.gain_reduction_db, 0.0,
                        "a hot block must report a negative reduction")

    def test_release_recovers_after_quiet(self):
        amp = THRESHOLD * 10.0 ** (12.0 / 20.0)
        limiter = Limiter(SR)
        limiter.process(const(amp))
        self.assertLess(limiter.gain_reduction_db, -0.1)
        quiet_frames = int(0.2 * SR)  # 200 ms of silence, in small blocks
        step = 512
        for start in range(0, quiet_frames, step):
            limiter.process(const(0.0, frames=min(step, quiet_frames - start)))
        self.assertGreater(limiter.gain_reduction_db, -0.1,
                            "200 ms of quiet must release the limiter")

    def test_block_splitting_matches_one_shot(self):
        rng = np.random.default_rng(7)
        signal = np.zeros((5120, 2), dtype=np.float32)
        signal[:2048] = 0.1
        signal[2048:3072] = (rng.standard_normal((1024, 2)) * 2.0)
        signal[3072:] = 0.05
        one = Limiter(SR).process(signal)
        many = Limiter(SR)
        pieces = [many.process(signal[i:i + 512])
                  for i in range(0, len(signal), 512)]
        stitched = np.concatenate(pieces, axis=0)
        self.assertLess(float(np.max(np.abs(one - stitched))), 1e-5,
                        "smoothing must be samplerate-driven, not block-driven")

    def test_stereo_link_applies_one_gain(self):
        block = np.zeros((4800, 2), dtype=np.float32)
        block[:, 0] = 1.5  # hot left: drives the shared gain down
        block[:, 1] = 0.5  # quiet right: untouched on its own
        out = Limiter(SR).process(block)
        settled = slice(-1024, None)  # past the attack, gain is steady
        left_factor = out[settled, 0] / block[settled, 0]
        right_factor = out[settled, 1] / block[settled, 1]
        np.testing.assert_allclose(
            left_factor, right_factor, atol=1e-6,
            err_msg="both channels must see the same gain (no image shift)")
        self.assertLess(float(np.max(np.abs(out[settled, 1]))), 0.5,
                        "the link must pull the quiet channel down too")


if __name__ == "__main__":
    unittest.main(verbosity=2)
