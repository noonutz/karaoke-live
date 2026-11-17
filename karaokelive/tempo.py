"""Tempo (BPM) estimation from audio (numpy only, no other dependencies).

Method: the input (mono or stereo) is downmixed to mono, sliced into
Hann-windowed frames (1024 samples, hop 256), and an onset-strength
envelope is built as positive spectral flux -- for each frame the
magnitude spectrum's positive bin-wise differences from the previous
frame are summed, then the envelope is normalised by its maximum to
0..1.

The envelope's biased autocorrelation is computed for lags
corresponding to 60-200 BPM (lag = envelope_rate * 60 / bpm, where
envelope_rate = sample_rate / 256). The winning lag is the smallest
lag whose score reaches 95% of the global maximum -- this breaks the
near-tie between a period and its double (e.g. at 174 BPM the
two-beat lag scores almost as high as the true one-beat lag) in
favour of the faster, true tempo. The resulting BPM is then folded
into 70-180 BPM by repeated doubling (too slow) or halving (too
fast), which corrects octave errors (e.g. 60 -> 120, 210 -> 105).

Confidence is (peak - mean) / mean of the lag scores, clipped to
0..1: a strongly periodic envelope scores near 1 (peak far above the
mean), while a flat/noisy envelope scores near 0 (peak barely above
the mean).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional

import numpy as np

FRAME_SIZE = 1024
HOP_SIZE = 256
BPM_MIN_SEARCH = 60.0
BPM_MAX_SEARCH = 200.0
BPM_FOLD_LO = 70.0
BPM_FOLD_HI = 180.0


@dataclass
class TempoResult:
    """One tempo estimate."""

    bpm: float  # beats per minute, folded into 70..180
    confidence: float  # 0..1, (peak-mean)/mean of the lag scores
    beat_frames: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )  # optional beat positions (samples); may be empty


def _to_mono(samples: np.ndarray) -> np.ndarray:
    """Downmix (frames, 2) or (frames,) float audio to mono float32."""
    audio = np.asarray(samples, dtype=np.float32)
    if audio.ndim == 2:
        return audio.mean(axis=1).astype(np.float32, copy=False)
    return audio.reshape(-1).astype(np.float32, copy=False)


def _onset_envelope(mono: np.ndarray) -> np.ndarray:
    """Positive spectral flux over a 1024/256 STFT, normalised to 0..1."""
    if mono.size < FRAME_SIZE + HOP_SIZE:
        return np.zeros(0, dtype=np.float64)
    n_frames = 1 + (len(mono) - FRAME_SIZE) // HOP_SIZE
    if n_frames < 2:
        return np.zeros(0, dtype=np.float64)
    window = np.hanning(FRAME_SIZE)
    prev: Optional[np.ndarray] = None
    flux = np.zeros(n_frames - 1, dtype=np.float64)
    for i in range(n_frames):
        frame = mono[i * HOP_SIZE: i * HOP_SIZE + FRAME_SIZE] * window
        mag = np.abs(np.fft.rfft(frame))
        if prev is not None:
            diff = mag - prev
            diff[diff < 0.0] = 0.0
            flux[i - 1] = float(diff.sum())
        prev = mag
    peak = float(flux.max()) if flux.size else 0.0
    if peak <= 0.0:
        return np.zeros_like(flux)
    return flux / peak


def detect_tempo(samples: np.ndarray, sample_rate: int) -> TempoResult:
    """Estimate the tempo of `samples` (mono or stereo float audio).

    Confidence is ``(peak - mean) / mean`` of the autocorrelation lag
    scores, clipped to 0..1: ~0 for a flat/noisy onset envelope
    (peak barely above the mean) and ~1 for a strongly periodic one.
    """
    mono = _to_mono(samples)
    sr = int(sample_rate)
    if sr <= 0 or mono.size == 0:
        return TempoResult(bpm=120.0, confidence=0.0,
                           beat_frames=np.array([], dtype=np.float64))
    env = _onset_envelope(mono)
    if env.size < 4:
        return TempoResult(bpm=120.0, confidence=0.0,
                           beat_frames=np.array([], dtype=np.float64))
    env_rate = sr / float(HOP_SIZE)
    lag_min = max(1, int(round(env_rate * 60.0 / BPM_MAX_SEARCH)))
    lag_max = int(round(env_rate * 60.0 / BPM_MIN_SEARCH))
    lag_max = min(lag_max, len(env) - 1)
    if lag_max <= lag_min:
        return TempoResult(bpm=120.0, confidence=0.0,
                           beat_frames=np.array([], dtype=np.float64))
    lags = range(lag_min, lag_max + 1)
    scores = np.empty(len(lags), dtype=np.float64)
    for idx, lag in enumerate(lags):
        scores[idx] = float(np.dot(env[lag:], env[: len(env) - lag]))
    mean = float(scores.mean()) if scores.size else 0.0
    if mean <= 1e-12:
        return TempoResult(bpm=120.0, confidence=0.0,
                           beat_frames=np.array([], dtype=np.float64))
    peak = float(scores.max())
    confidence = float(np.clip((peak - mean) / mean, 0.0, 1.0))
    # Prefer the fastest tempo among near-tied peaks: the smallest lag
    # within 95% of the maximum. Autocorrelation scores a period and its
    # multiples almost equally, and argmax alone lands on the slower
    # octave (e.g. 87 instead of 174 BPM).
    near = np.nonzero(scores >= 0.95 * peak)[0]
    best_lag = lag_min + (int(near[0]) if near.size else int(np.argmax(scores)))
    bpm = 60.0 * env_rate / float(best_lag) if best_lag > 0 else 120.0
    while bpm < BPM_FOLD_LO:
        bpm *= 2.0
    while bpm > BPM_FOLD_HI:
        bpm /= 2.0
    return TempoResult(bpm=float(bpm), confidence=confidence,
                       beat_frames=np.array([], dtype=np.float64))


class RollingTempoDetector:
    """Live tempo estimate over a sliding window of recent audio.

    Keeps the last `window_s` seconds of mono audio in a deque-backed
    buffer. `feed` appends one block and recomputes the tempo at most
    every 1.0 s of new audio; calls in between return the previous
    result. Returns None until enough audio (~4 s) has arrived for a
    first estimate.
    """

    def __init__(self, sample_rate: int, window_s: float = 12.0):
        self.sample_rate = sample_rate
        self.window_s = window_s
        self._capacity = max(1, int(window_s * sample_rate))
        self._need = min(self._capacity, 4 * sample_rate)  # ~4 s first
        self._recompute_every = max(1, int(1.0 * sample_rate))
        self._buffer: Deque[np.ndarray] = deque()
        self._buffered = 0
        self._since_recompute = 0
        self._last: Optional[TempoResult] = None

    def feed(self, block: np.ndarray) -> Optional[TempoResult]:
        """Append `block` (mono or stereo) and return the tempo estimate."""
        mono = _to_mono(block)
        if mono.size:
            self._buffer.append(mono)
            self._buffered += len(mono)
            self._since_recompute += len(mono)
            while self._buffered - len(self._buffer[0]) >= self._capacity:
                self._buffered -= len(self._buffer.popleft())
        if self._buffered < self._need:
            return self._last
        if self._last is None or self._since_recompute >= self._recompute_every:
            audio = np.concatenate(list(self._buffer))[-self._capacity:]
            self._last = detect_tempo(audio, self.sample_rate)
            self._since_recompute = 0
        return self._last

    def reset(self) -> None:
        """Drop all buffered audio and the last result."""
        self._buffer.clear()
        self._buffered = 0
        self._since_recompute = 0
        self._last = None
