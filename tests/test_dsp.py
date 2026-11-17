"""Checks that the DSP does what the comments claim.

Run: .venv/bin/python -m unittest discover -s tests -v
"""
from __future__ import annotations

import unittest

import numpy as np

from karaokelive.engines.center import CenterCancel
from karaokelive.engines.channel import ChannelPick
from karaokelive.engines.demucs_ml import DemucsEngine

SR = 48000


def tone(freq: float, seconds: float, amp: float = 0.2) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def band_level(signal: np.ndarray, freq: float) -> float:
    """Magnitude of one frequency bin (Hann-windowed FFT, single channel)."""
    window = np.hanning(len(signal))
    spectrum = np.abs(np.fft.rfft(signal * window))
    index = int(round(freq * len(signal) / SR))
    return float(spectrum[max(0, index - 2):index + 3].max())


def db(a: float, b: float) -> float:
    return 20.0 * np.log10(max(a, 1e-12) / max(b, 1e-12))


def stereo_mix() -> np.ndarray:
    """Centred vocal + a wide backing: each instrument only on one side."""
    vocal = tone(440.0, 2.0)
    left_inst = tone(220.0, 2.0)
    right_inst = tone(660.0, 2.0)
    return np.stack([vocal + left_inst, vocal + right_inst], axis=1)


class CenterCancelTests(unittest.TestCase):
    def test_kills_the_centred_vocal(self):
        mix = stereo_mix()
        engine = CenterCancel(SR, vocal_level=1.0, mono_guard=False)
        out = engine.process(mix)
        before = band_level(mix[:, 0], 440.0)
        after = band_level(out[:, 0], 440.0)
        self.assertLess(db(after, before), -30.0,
                        f"centred 440 Hz only dropped {db(after, before):.1f} dB")

    def test_keeps_the_side_content(self):
        mix = stereo_mix()
        engine = CenterCancel(SR, vocal_level=1.0, mono_guard=False)
        out = engine.process(mix)
        for freq, channel in ((220.0, 0), (660.0, 1)):
            before = band_level(mix[:, channel], freq)
            after = band_level(out[:, channel], freq)
            self.assertLess(abs(db(after, before)), 1.0,
                            f"{freq} Hz moved {db(after, before):.1f} dB")

    def test_bass_is_restored(self):
        bass = tone(60.0, 2.0, amp=0.3)
        vocal = tone(440.0, 2.0)
        mix = np.stack([bass + vocal, bass + vocal], axis=1)
        engine = CenterCancel(SR, vocal_level=1.0, mono_guard=False, bass_hz=140)
        out = engine.process(mix)
        keep = db(band_level(out[:, 0], 60.0), band_level(mix[:, 0], 60.0))
        self.assertGreater(keep, -3.0, f"60 Hz lost {keep:.1f} dB")

    def test_half_blend_keeps_most_of_the_vocal(self):
        # level-compensated knob: residual = (1-k)/(1-k/2) = -3.5 dB at k=0.5
        mix = stereo_mix()
        engine = CenterCancel(SR, vocal_level=0.5, mono_guard=False)
        out = engine.process(mix)
        residual = db(band_level(out[:, 0], 440.0), band_level(mix[:, 0], 440.0))
        self.assertAlmostEqual(residual, -3.5, delta=0.7)

    def test_mono_guard_caps_the_level(self):
        engine = CenterCancel(SR, vocal_level=1.0, mono_guard=True)
        self.assertAlmostEqual(engine.effective_level, 0.85)

    def test_block_chaining_matches_one_shot(self):
        mix = stereo_mix()
        one = CenterCancel(SR, vocal_level=0.9)
        many = CenterCancel(SR, vocal_level=0.9)
        whole = one.process(mix)
        pieces = [many.process(mix[i:i + 512]) for i in range(0, len(mix), 512)]
        stitched = np.concatenate(pieces, axis=0)
        self.assertLess(float(np.max(np.abs(whole - stitched))), 1e-6)

    def test_zero_level_is_transparent(self):
        mix = stereo_mix()
        engine = CenterCancel(SR, vocal_level=0.0)
        out = engine.process(mix)
        self.assertLess(float(np.max(np.abs(out - mix))), 1e-6)


class ChannelPickTests(unittest.TestCase):
    def test_takes_the_instrumental_side(self):
        mix = stereo_mix()
        out = ChannelPick(SR, side="right").process(mix)
        np.testing.assert_allclose(out[:, 0], mix[:, 1])
        np.testing.assert_allclose(out[:, 1], mix[:, 1])


class _IdentityEngine(DemucsEngine):
    """No torch: pretend the model found no vocals, so output == input."""

    def _separate(self, chunk: np.ndarray) -> np.ndarray:
        return np.zeros_like(chunk)


class DemucsStreamingTests(unittest.TestCase):
    def test_length_is_preserved_and_audio_is_delayed_not_lost(self):
        engine = _IdentityEngine(SR, vocal_level=1.0, chunk_s=0.5, overlap_s=0.1)
        rng = np.random.default_rng(0)
        signal = (rng.standard_normal((SR, 2)) * 0.1).astype(np.float32)
        pieces = []
        for start in range(0, len(signal), 1024):
            out = engine.process(signal[start:start + 1024])
            self.assertEqual(len(out), len(signal[start:start + 1024]))
            pieces.append(out)
        pieces.append(engine.flush())
        rendered = np.concatenate(pieces, axis=0)
        self.assertGreaterEqual(len(rendered), len(signal))
        aligned = rendered[engine.latency_frames:engine.latency_frames + len(signal)]
        self.assertEqual(len(aligned), len(signal))
        self.assertLess(float(np.max(np.abs(aligned - signal))), 1e-6)

    def test_hop_size_and_latency(self):
        engine = DemucsEngine(SR, chunk_s=2.5, overlap_s=0.5)
        self.assertEqual(engine.hop_frames, int(2.0 * SR))
        # a commit needs hop + crossfade + lookahead of input before it can go
        self.assertEqual(engine.latency_frames,
                         engine.hop_frames + engine.xfade + engine.overlap_frames)
        self.assertAlmostEqual(engine.latency_frames / SR, 2.54, delta=0.06)


class AudioIoTests(unittest.TestCase):
    def test_wav_roundtrip(self):
        import tempfile
        from pathlib import Path

        from karaokelive import audio_io

        data = stereo_mix()[:SR]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "probe.wav"
            audio_io.save(path, data, SR)
            info = audio_io.probe(path)
            self.assertEqual(info.sample_rate, SR)
            self.assertEqual(info.channels, 2)
            back = audio_io.load(path, SR)
            self.assertEqual(back.shape[1], 2)
            self.assertLess(float(np.max(np.abs(back - data))), 2e-3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
