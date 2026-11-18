"""Checks for the block-streaming 4-stem player.

Run: .venv/bin/python -m unittest discover -s tests -t . -v
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from karaokelive import audio_io
from karaokelive.stem_player import STEMS, StemPlayer

SR = 48000
DUR = 2.0  # seconds
N = int(SR * DUR)


def _build_stems(tmp: Path) -> dict[str, np.ndarray]:
    """Four 48 kHz stereo signals: drums = click train, rest = pure tones."""
    t = np.arange(N, dtype=np.float64) / SR
    drums = np.zeros(N)
    drums[::SR // 2] = 0.8  # click every 0.5 s
    signals = {
        "vocals": 0.4 * np.sin(2 * np.pi * 440.0 * t),
        "drums": drums,
        "bass": 0.4 * np.sin(2 * np.pi * 110.0 * t),
        "other": 0.4 * np.sin(2 * np.pi * 330.0 * t),
    }
    return {k: np.stack([v, v], axis=1).astype(np.float32)
            for k, v in signals.items()}


def _band_energy(block: np.ndarray, freq: float) -> float:
    """Energy of `block` (mono) within ~±25 Hz of `freq` (Goertzel-ish)."""
    mono = block.mean(axis=1).astype(np.float64)
    n = len(mono)
    if n == 0:
        return 0.0
    k = freq * n / SR
    w = 2.0 * np.pi * np.arange(n) / n * k
    return float((mono * np.cos(w)).sum() ** 2
                 + (mono * np.sin(w)).sum() ** 2) / n


def _click_energy(block: np.ndarray) -> float:
    """Mean square in ±8 samples around each 0.5 s click.

    Highpasses above 2 kHz first: the 110/330/440 Hz tones vanish, while
    the single-sample clicks (broadband impulses) keep their energy.
    """
    mono = block.mean(axis=1).astype(np.float64)
    n = len(mono)
    if n == 0:
        return 0.0
    spectrum = np.fft.rfft(mono)
    freqs = np.fft.rfftfreq(n, 1.0 / SR)
    spectrum[freqs < 2000.0] = 0.0
    hp = np.fft.irfft(spectrum, n=n)
    energy = 0.0
    count = 0
    for click in range(0, N, SR // 2):
        lo = max(click - 8, 0)
        hi = min(click + 8, n)
        if lo < hi:
            energy += float((hp[lo:hi] ** 2).mean())
            count += 1
    return energy / max(count, 1)


class StemPlayerTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.signals = _build_stems(self.dir)
        self.paths = {}
        for stem in STEMS:
            path = str(self.dir / f"{stem}.wav")
            audio_io.save(path, self.signals[stem], SR, bits=16)
            self.paths[stem] = path
        self.players: list[StemPlayer] = []
        self.addCleanup(self._close_players)

    def _close_players(self):
        for player in self.players:
            try:
                player.close()
            except Exception:
                pass

    def _player(self, **kwargs) -> StemPlayer:
        player = StemPlayer(dict(self.paths), **kwargs)
        self.players.append(player)
        return player

    def _read_all(self, player: StemPlayer,
                  blocksize: int = 4096) -> np.ndarray:
        chunks = []
        while True:
            block = player.read(blocksize)
            if len(block) == 0:
                break
            chunks.append(block)
        return (np.concatenate(chunks, axis=0) if chunks
                else np.zeros((0, 2), dtype=np.float32))

    def test_unit_gains_sum_to_signals(self):
        player = self._player()
        mixed = self._read_all(player)
        expected = sum(self.signals[stem] for stem in STEMS)
        expected = np.clip(expected, -1.0, 1.0)
        self.assertEqual(mixed.shape, (N, 2))
        self.assertLess(float(np.abs(mixed - expected).max()), 1e-4)

    def test_mute_drums_kills_clicks_keeps_tones(self):
        player = self._player()
        player.set_gain("drums", 0)
        mixed = self._read_all(player)
        expected_no_drums = np.clip(
            sum(self.signals[s] for s in ("vocals", "bass", "other")),
            -1.0, 1.0)
        # Click moments: energy drops > 30 dB vs the all-stems mix.
        full = np.clip(sum(self.signals[s] for s in STEMS), -1.0, 1.0)
        before = _click_energy(full)
        after = _click_energy(mixed)
        self.assertGreater(10 * np.log10(before / max(after, 1e-30)), 30.0)
        # Tones stay within 0.5 dB of the drums-free expectation.
        for freq in (440.0, 110.0, 330.0):
            got = _band_energy(mixed, freq)
            want = _band_energy(expected_no_drums, freq)
            self.assertLess(abs(10 * np.log10(got / max(want, 1e-30))),
                            0.5, f"{freq} Hz off")

    def test_seek_matches_sequential_read(self):
        player = self._player()
        player.seek(1.0)
        from_seek = player.read(N)
        player2 = self._player()
        player2.read(SR)  # skip the first second
        sequential = player2.read(N)
        length = min(len(from_seek), len(sequential))
        self.assertGreater(length, 0)
        self.assertLess(
            float(np.abs(from_seek[:length] - sequential[:length]).max()),
            1e-5)

    def test_loop_wraps_exactly(self):
        # Reference: one block read straight from the loop start.
        ref = self._player(blocksize=4096)
        ref.seek(0.5)
        expected_start = ref.read(256)
        self.assertEqual(len(expected_start), 256)
        # Play 2 s with a 0.5-1.5 s loop: content repeats every 1.0 s, so
        # sample k >= 0.5 s must equal the loop-start block at (k - 0.5) % 1.0.
        player = self._player(blocksize=256)
        player.set_loop(0.5, 1.5)
        total = player.read(int(2.0 * SR))
        self.assertEqual(len(total), int(2.0 * SR))
        # ... so the position wraps exact: 0.5 + ((2.0-0.5) % 1.0) == 1.0.
        self.assertAlmostEqual(player.position_s, 1.0, places=6)
        # Content at the wrap matches the loop start, sample for sample:
        # samples straddling the 1.5 s wrap equal the loop body (read
        # straight from 0.5 s with no loop) at phase (k - 0.5) % 1.0 s.
        at_wrap = total[SR - 128: SR + 128]  # straddles the 1.5 s wrap
        self.assertEqual(len(at_wrap), 256)
        loop_body = self._read_all_at(0.5, SR)  # the 1.0 s loop body
        self.assertEqual(len(loop_body), SR)
        base = SR - 128 - SR // 2  # phase of total[SR-128] inside the body
        want = np.concatenate(
            [loop_body[(base + i) % (SR // 2):][:1] for i in range(256)],
            axis=0)
        self.assertLess(float(np.abs(at_wrap - want).max()), 1e-6)
        # And the loop start itself matches a direct seek+read exactly.
        self.assertLess(
            float(np.abs(total[SR // 2: SR // 2 + 256]
                         - expected_start).max()), 1e-6)

    def _read_all_at(self, start_s: float, frames: int) -> np.ndarray:
        """Read `frames` samples sequentially from `start_s` (no loop)."""
        player = self._player(blocksize=4096)
        player.seek(start_s)
        chunks = []
        while sum(len(c) for c in chunks) < frames:
            block = player.read(min(4096, frames - sum(len(c)
                                                       for c in chunks)))
            if len(block) == 0:
                break
            chunks.append(block)
        return (np.concatenate(chunks, axis=0) if chunks
                else np.zeros((0, 2), dtype=np.float32))

    def test_clear_loop_stops_wrapping(self):
        player = self._player(blocksize=256)
        player.set_loop(0.5, 1.5)
        player.read(SR)  # reach the loop
        player.set_loop(None, None)
        self.assertIsNone(player.loop_range)
        tail = self._read_all(player)
        self.assertAlmostEqual(player.position_s, DUR, places=6)
        self.assertEqual(len(tail), N - SR)

    def test_missing_stem_is_silence(self):
        paths = dict(self.paths)
        del paths["bass"]
        player = StemPlayer(paths)
        self.players.append(player)
        mixed = self._read_all(player)
        expected = np.clip(
            sum(self.signals[s] for s in ("vocals", "drums", "other")),
            -1.0, 1.0)
        self.assertLess(float(np.abs(mixed - expected).max()), 1e-4)

    def test_wrong_sample_rate_raises_naming_file(self):
        (self.dir / "bass.wav").write_bytes(b"")  # replaced below
        audio_io.save(str(self.dir / "bass.wav"), self.signals["bass"], 44100, bits=16)
        player = self._player()
        with self.assertRaises(ValueError) as ctx:
            player.read(64)
        self.assertIn("bass.wav", str(ctx.exception))

    def test_read_never_over_returns_and_close_raises(self):
        player = self._player(blocksize=512)
        tail = player.read(10 * N)  # past EOF without a loop
        self.assertLessEqual(len(tail), N)
        self.assertEqual(len(player.read(64)), 0)  # EOF: empty, not an error
        player.close()
        with self.assertRaises(ValueError):
            player.read(64)
        with self.assertRaises(ValueError):
            player.set_gain("drums", 0.0)

    def test_gains_returns_a_copy(self):
        player = self._player()
        player.set_gain("vocals", 0.25)
        self.assertEqual(player.gains["vocals"], 0.25)
        player.gains["vocals"] = 99.0
        self.assertEqual(player.gains["vocals"], 0.25)
        with self.assertRaises(KeyError):
            player.set_gain("kazoo", 1.0)

    def _write_float_wav(self, path: Path, stem: str = "vocals") -> None:
        """Write a float32 WAV the way demucs does (`-c:a pcm_f32le`)."""
        data = np.ascontiguousarray(self.signals[stem], dtype="<f4")
        subprocess.run(
            [shutil.which("ffmpeg"), "-v", "error", "-y",
             "-f", "f32le", "-ar", str(SR), "-ac", "2", "-i", "-",
             "-c:a", "pcm_f32le", str(path)],
            input=data.tobytes(), check=True,
        )

    def test_float_wav_raises_thai_pcm_error(self):
        float_path = self.dir / "vocals-float.wav"
        self._write_float_wav(float_path)
        paths = dict(self.paths)
        paths["vocals"] = str(float_path)
        with self.assertRaises(ValueError) as ctx:
            StemPlayer(paths)
        self.assertIn("ต้องเป็นไฟล์ WAV แบบ PCM", str(ctx.exception))

    def test_to_pcm_wav_makes_float_playable(self):
        import wave

        float_path = self.dir / "vocals-float.wav"
        self._write_float_wav(float_path)
        pcm_path = str(self.dir / "vocals-pcm.wav")
        out = audio_io.to_pcm_wav(str(float_path), pcm_path)
        self.assertEqual(out, pcm_path)
        with wave.open(pcm_path, "rb") as wav:  # plain PCM: wave reads it
            self.assertEqual(wav.getcomptype(), "NONE")
            self.assertEqual(wav.getsampwidth(), 2)
            self.assertEqual(wav.getframerate(), SR)
            self.assertEqual(wav.getnchannels(), 2)
        paths = dict(self.paths)
        paths["vocals"] = pcm_path
        player = StemPlayer(paths)
        self.players.append(player)
        mixed = self._read_all(player)
        expected = np.clip(
            sum(self.signals[s] for s in STEMS), -1.0, 1.0)
        self.assertLess(float(np.abs(mixed - expected).max()), 1e-4)
        self.assertGreater(float(np.abs(mixed).max()), 0.1)  # non-silent

    def test_short_stem_is_zero_padded(self):
        short = self.signals["other"][: N // 2]
        audio_io.save(str(self.dir / "other.wav"), short, SR, bits=16)
        player = self._player()
        mixed = self._read_all(player)
        self.assertEqual(len(mixed), N)
        padded = np.zeros_like(self.signals["other"])
        padded[: N // 2] = short
        expected = np.clip(
            self.signals["vocals"] + self.signals["drums"]
            + self.signals["bass"] + padded, -1.0, 1.0)
        self.assertLess(float(np.abs(mixed - expected).max()), 1e-4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
