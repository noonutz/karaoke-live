"""Block-streaming 4-stem player (numpy + stdlib `wave` only).

Each stem is streamed from its WAV file in blocks -- a whole song is never
loaded into RAM (a review finding against the old `FilePlayer` preload).
`read(n)` mixes the current block of every stem (`gain[stem] * stem_block`,
summed and clipped to [-1, 1]) and returns float32 stereo.

Missing stem files read as silence so the other stems keep playing;
a stem at the wrong sample rate raises `ValueError` naming the file.
Positions are integer frame counts, so a loop wraps sample-exactly and
never drifts. Threading: one `StemPlayer` is driven by a single audio
callback thread; `set_gain`/`seek`/`set_loop` from another thread take
effect on the next `read`.
"""
from __future__ import annotations

import wave
from collections.abc import Mapping

import numpy as np

#: Stem names (`separate_file` writes these four WAVs per track).
STEMS = ("vocals", "drums", "bass", "other")


class StemPlayer:
    """Stream four stem WAVs, mixed on read, with per-stem gains and a loop."""

    def __init__(self, paths: Mapping[str, str],
                 sample_rate: int = 48000, blocksize: int = 512):
        self.sample_rate = int(sample_rate)
        self.blocksize = int(blocksize)
        if self.sample_rate <= 0:
            raise ValueError(f"bad sample_rate {sample_rate!r}")
        if self.blocksize <= 0:
            raise ValueError(f"bad blocksize {blocksize!r}")
        streams: dict[str, wave.Wave_read | None] = {}
        self._paths: dict[str, str | None] = {}
        frames = 0
        for stem in STEMS:
            path = paths.get(stem) if paths is not None else None
            stream = self._open(path) if path else None
            streams[stem] = stream
            self._paths[stem] = str(path) if path else None
            if stream is not None:
                frames = max(frames, stream.getnframes())
        self._streams = streams
        self._frames = frames
        self.duration_s = frames / float(self.sample_rate) if frames else 0.0
        self._pos = 0  # absolute frame index of the next frame to read
        self._gains = {stem: 1.0 for stem in STEMS}
        self._loop: tuple[int, int | None] | None = None
        self._closed = False

    @staticmethod
    def _open(path: str) -> wave.Wave_read | None:
        """Open one stem WAV, or `None` when the file is missing (silence).

        A non-PCM WAV (e.g. a raw demucs float32 stem, or a 24-bit WAV with
        a `WAVE_FORMAT_EXTENSIBLE` header that this Python's `wave` refuses)
        raises `ValueError` naming the file and the required format, instead
        of leaking `wave.Error`/`EOFError`.
        """
        try:
            return wave.open(str(path), "rb")
        except FileNotFoundError:
            return None
        except OSError:
            return None
        except (wave.Error, EOFError) as exc:
            raise ValueError(
                f"{path}: ต้องเป็นไฟล์ WAV แบบ PCM (16/24-bit) — "
                "ไฟล์ float32 หรือ 24-bit แบบ EXTENSIBLE "
                "อ่านด้วย wave ไม่ได้") from exc

    # -- properties -----------------------------------------------------

    @property
    def gains(self) -> dict[str, float]:
        """A copy of the current per-stem gains."""
        return dict(self._gains)

    @property
    def position_s(self) -> float:
        """Current playback position in seconds."""
        return self._pos / float(self.sample_rate) if self.sample_rate else 0.0

    @property
    def loop_range(self) -> tuple[float, float | None] | None:
        """`(start_s, end_s)` of the active loop, or `None` when cleared."""
        if self._loop is None:
            return None
        start, end = self._loop
        return (start / float(self.sample_rate),
                end / float(self.sample_rate) if end is not None else None)

    # -- control --------------------------------------------------------

    def set_gain(self, stem: str, value: float) -> None:
        """Set one stem's gain (linear multiplier, applied on `read`)."""
        if stem not in self._gains:
            raise KeyError(f"unknown stem {stem!r} "
                           f"(choose from {', '.join(STEMS)})")
        self._ensure_open()
        self._gains[stem] = float(value)

    def seek(self, seconds: float) -> None:
        """Jump to `seconds` (clamped to `[0, duration_s]`)."""
        self._ensure_open()
        frame = int(round(float(seconds) * self.sample_rate))
        self._pos = min(max(frame, 0), self._frames)

    def set_loop(self, start_s: float | None,
                 end_s: float | None) -> None:
        """Loop `[start_s, end_s)`; `set_loop(None, None)` clears the loop.

        `end_s=None` loops from `start_s` to the end of the song.
        Bounds are clamped to `[0, duration_s]` and a degenerate range
        (`end <= start`) clears the loop.
        """
        self._ensure_open()
        if start_s is None and end_s is None:
            self._loop = None
            return
        start = int(round(float(start_s or 0.0) * self.sample_rate))
        start = min(max(start, 0), self._frames)
        end = (int(round(float(end_s) * self.sample_rate))
               if end_s is not None else None)
        if end is not None:
            end = min(max(end, 0), self._frames)
            if end <= start:
                self._loop = None
                return
        self._loop = (start, end)

    def close(self) -> None:
        """Close every stem file; a second `close()` is harmless."""
        for stream in self._streams.values():
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass
        self._streams = {stem: None for stem in STEMS}
        self._closed = True

    # -- playback -------------------------------------------------------

    def read(self, frames: int) -> np.ndarray:
        """Mix up to `frames` of audio; `(frames, 2)` float32 in [-1, 1].

        Returns fewer than `frames` rows only at the end of the song with no
        loop covering it (possibly zero rows). Raises `ValueError` after
        `close()`.
        """
        self._ensure_open()
        want = max(int(frames), 0)
        if want == 0:
            return np.zeros((0, 2), dtype=np.float32)
        out = np.zeros((want, 2), dtype=np.float32)
        got = 0
        while got < want:
            if self._loop is not None:
                self._pos = self._apply_loop(self._pos)
            if self._pos >= self._frames:
                break
            stop = self._loop_stop()
            chunk = min(self.blocksize, want - got, stop - self._pos)
            if chunk <= 0:
                break  # defensive; loop math above should prevent this
            out[got: got + chunk] = self._read_chunk(chunk)
            got += chunk
            self._pos += chunk
        return out[:got]

    # -- internals ------------------------------------------------------

    def _ensure_open(self) -> None:
        if self._closed:
            raise ValueError("StemPlayer is closed")

    def _loop_stop(self) -> int:
        """Exclusive end frame of the current read region (loop end or EOF)."""
        if self._loop is None:
            return self._frames
        _, end = self._loop
        return self._frames if end is None else end

    def _apply_loop(self, pos: int) -> int:
        """Wrap `pos` into the loop range (integer frames: never drifts)."""
        assert self._loop is not None
        start, end = self._loop
        stop = self._frames if end is None else end
        if stop <= start:
            return pos
        if pos < start:
            return pos  # pre-roll plays straight through into the loop
        if pos >= stop:
            pos = start + (pos - start) % (stop - start)
        return pos

    def _read_chunk(self, chunk: int) -> np.ndarray:
        """Mix exactly `chunk` frames at `self._pos`; float32 `(chunk, 2)`."""
        mixed = np.zeros((chunk, 2), dtype=np.float32)
        for stem in STEMS:
            stream = self._streams[stem]
            gain = self._gains[stem]
            if stream is None or gain == 0.0:
                continue  # missing stem, or muted: pure silence
            block = self._read_block(stem, stream, chunk)
            if block.shape[0]:
                mixed[: block.shape[0]] += block * gain
        np.clip(mixed, -1.0, 1.0, out=mixed)
        return mixed

    def _read_block(self, stem: str, stream: wave.Wave_read,
                    chunk: int) -> np.ndarray:
        """Read up to `chunk` frames at `self._pos` as float32 stereo.

        A stem shorter than the song reads as silence past its own end
        (short `wave` files raise on `setpos` past EOF, so check first).
        """
        if stream.getframerate() != self.sample_rate:
            raise ValueError(
                f"สเต็ม {stem} ({self._paths.get(stem)}) มี sample rate "
                f"{stream.getframerate()} แต่ต้องการ {self.sample_rate}")
        if self._pos >= stream.getnframes():
            return np.zeros((0, 2), dtype=np.float32)
        stream.setpos(self._pos)
        raw = stream.readframes(chunk)
        if not raw:
            return np.zeros((0, 2), dtype=np.float32)
        width = stream.getsampwidth()
        channels = max(stream.getnchannels(), 1)
        data = np.frombuffer(raw, dtype=np.uint8)
        frames = len(data) // (width * channels)
        data = data[: frames * width * channels]
        if width == 1:  # unsigned 8-bit PCM centres on 128
            samples = (data.astype(np.int16) - 128).astype(np.float32) / 128.0
        elif width == 2:
            samples = (data.view(np.int16).astype(np.float32)
                       / 32768.0)
        elif width == 3:  # 24-bit little-endian PCM (audio_io.save writes this)
            raw3 = data.reshape(-1, 3).astype(np.int32)
            signed = (raw3[:, 0] | (raw3[:, 1] << 8) | (raw3[:, 2] << 16))
            signed = np.where(signed >= 0x800000,
                              signed - 0x1000000, signed)
            samples = signed.astype(np.float32) / 8388608.0
        elif width == 4:
            samples = data.view(np.int32).astype(np.float32) / 2147483648.0
        else:
            raise ValueError(f"สเต็ม {stem} ({self._paths.get(stem)}) "
                             f"ใช้ความลึกเสียง {width} ไบต์ ซึ่งไม่รองรับ")
        samples = samples.reshape(frames, channels)
        if channels == 1:
            samples = np.repeat(samples, 2, axis=1)
        else:
            samples = samples[:, :2]
        return samples.astype(np.float32, copy=False)

    # -- context manager -------------------------------------------------

    def __enter__(self) -> StemPlayer:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


__all__ = ["STEMS", "StemPlayer"]
