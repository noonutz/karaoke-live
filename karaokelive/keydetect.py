"""Musical key detection from audio (numpy only, no other dependencies).

Method: the input (mono or stereo) is downmixed to mono, sliced into
Hann-windowed frames (4096 samples, hop 2048), and each frame's magnitude
spectrum is folded into a 12-bin chroma vector. Every FFT bin between 55 Hz
and 4000 Hz is mapped to a pitch class with ``midi = 69 + 12*log2(f/440)``
(A4 = MIDI 69); the bin's magnitude is added to that class. Frame chromas are
accumulated over the whole signal and normalised to sum to 1.

The chroma is compared against the Krumhansl-Schmuckler key profiles (one
major, one minor, each rotated over all 12 tonics -> 24 candidates) using
Pearson correlation; the highest correlation wins. A Camelot wheel code
(``8A`` .. ``1B``) is attached for DJ/key-mixing use.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple

import numpy as np

FRAME_SIZE = 4096
HOP_SIZE = 2048
FMIN_HZ = 55.0
FMAX_HZ = 4000.0

#: Pitch-class names, index 0 = C.
NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F",
              "F#", "G", "G#", "A", "A#", "B")

#: Krumhansl-Schmuckler probe-tone profiles, relative to the tonic.
KS_MAJOR = (6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
            2.52, 5.19, 2.39, 3.66, 2.29, 2.88)
KS_MINOR = (6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
            2.54, 4.75, 3.98, 2.69, 3.34, 3.17)

_MAJOR_CAMELOT = {
    "C": "8B", "G": "9B", "D": "10B", "A": "11B", "E": "12B", "B": "1B",
    "F#": "2B", "C#": "3B", "G#": "4B", "D#": "5B", "A#": "6B", "F": "7B",
}
_MINOR_CAMELOT = {
    "A": "8A", "E": "9A", "B": "10A", "F#": "11A", "C#": "12A", "G#": "1A",
    "D#": "2A", "A#": "3A", "F": "4A", "C": "5A", "G": "6A", "D": "7A",
}

#: Correlation gap that counts as a decisive win (see `detect_key`).
_MARGIN_FULL = 0.15


@dataclass
class KeyResult:
    """One key estimate."""
    key: str  # tonic, e.g. "A"
    mode: str  # "major" or "minor"
    confidence: float  # 0..1, blends fit quality with the runner-up margin
    chroma: np.ndarray  # 12 floats, sums to 1.0, index 0 = C
    camelot: str  # e.g. "8A"
    # 2nd/3rd best guesses as (label, raw Pearson correlation with the
    # key profile), e.g. [("C major", 0.72)]. These are correlation scores,
    # not confidence: do not compare them directly against `confidence`.
    alternatives: List[Tuple[str, float]]


def _to_mono(samples: np.ndarray) -> np.ndarray:
    """Downmix (frames, 2) or (frames,) float audio to mono float32."""
    audio = np.asarray(samples, dtype=np.float32)
    if audio.ndim == 2:
        return audio.mean(axis=1).astype(np.float32, copy=False)
    return audio.reshape(-1).astype(np.float32, copy=False)


def _chroma(mono: np.ndarray, sample_rate: int) -> np.ndarray:
    """Fold the magnitude spectrum of `mono` into a normalised 12-bin chroma."""
    chroma = np.zeros(12, dtype=np.float64)
    if mono.size == 0:
        return np.full(12, 1.0 / 12.0)
    if len(mono) < FRAME_SIZE:
        mono = np.pad(mono, (0, FRAME_SIZE - len(mono)))
    freqs = np.fft.rfftfreq(FRAME_SIZE, d=1.0 / sample_rate)
    in_range = (freqs >= FMIN_HZ) & (freqs <= FMAX_HZ)
    # Pitch class of each kept FFT bin: nearest MIDI note mod 12.
    with np.errstate(divide="ignore"):
        midi = 69.0 + 12.0 * np.log2(freqs[in_range] / 440.0)
    pitch_class = np.mod(np.floor(midi + 0.5).astype(int), 12)
    window = np.hanning(FRAME_SIZE)
    for start in range(0, len(mono) - FRAME_SIZE + 1, HOP_SIZE):
        magnitude = np.abs(np.fft.rfft(mono[start:start + FRAME_SIZE] * window))
        chroma += np.bincount(pitch_class, weights=magnitude[in_range],
                              minlength=12)
    total = chroma.sum()
    if total <= 0.0:
        return np.full(12, 1.0 / 12.0)
    return chroma / total


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation coefficient of two equal-length vectors."""
    a = np.asarray(a, dtype=np.float64) - np.mean(a)
    b = np.asarray(b, dtype=np.float64) - np.mean(b)
    denom = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
    if denom <= 0.0:
        return 0.0
    return float(np.sum(a * b) / denom)


def detect_key(samples: np.ndarray, sample_rate: int) -> KeyResult:
    """Estimate the musical key of `samples` (mono or stereo float audio).

    Confidence blends the best profile correlation with its margin over the
    runner-up::

        confidence = max(0, r_best) * (0.5 + 0.5 * min(1, gap / 0.15))

    where ``gap = r_best - r_second``, clipped to 0..1. It is monotone
    non-decreasing in both the best correlation (better fit -> surer) and the
    gap (clearer win -> surer): a perfect lone match scores ~1, a tie for
    first scores half of the best correlation, and anything uncorrelated
    scores 0.
    """
    chroma = _chroma(_to_mono(samples), sample_rate)
    scored: List[Tuple[str, str, str, float]] = []  # label, key, mode, r
    for tonic in range(12):
        name = NOTE_NAMES[tonic]
        major = np.roll(np.array(KS_MAJOR), tonic)
        minor = np.roll(np.array(KS_MINOR), tonic)
        scored.append((f"{name} major", name, "major", _pearson(chroma, major)))
        scored.append((f"{name} minor", name, "minor", _pearson(chroma, minor)))
    scored.sort(key=lambda item: item[3], reverse=True)
    (label, key, mode, r_best) = scored[0]
    r_second = scored[1][3]
    gap = max(0.0, r_best - r_second)
    confidence = float(np.clip(max(0.0, r_best) * (0.5 + 0.5 * gap / _MARGIN_FULL),
                               0.0, 1.0))
    camelot = (_MAJOR_CAMELOT if mode == "major" else _MINOR_CAMELOT)[key]
    return KeyResult(
        key=key,
        mode=mode,
        confidence=confidence,
        chroma=chroma.astype(np.float32),
        camelot=camelot,
        alternatives=[(scored[1][0], scored[1][3]), (scored[2][0], scored[2][3])],
    )


def estimate_chord(samples: np.ndarray, sample_rate: int) -> str:
    """Estimate the current triad from `samples` (mono or stereo float audio).

    Each of the 24 fixed templates (12 major, 12 minor triads) is
    scored by the summed chroma energy on its root/third/fifth; the
    highest score wins. Major triads are labelled ``C``, ``F`` ...
    and minor triads ``Am``, ``C#m`` ... Returns ``"?"`` when the
    chroma is too flat to call, i.e. the margin between the best and
    second-best scores is below 0.02.
    """
    chroma = _chroma(_to_mono(samples), sample_rate)
    scored: list[tuple[float, str]] = []
    for tonic in range(12):
        name = NOTE_NAMES[tonic]
        for minor in (False, True):
            third = (tonic + (3 if minor else 4)) % 12
            triad = (tonic, third, (tonic + 7) % 12)
            score = float(chroma[list(triad)].sum())
            label = name + ("m" if minor else "")
            scored.append((score, label))
    scored.sort(key=lambda item: item[0], reverse=True)
    if scored[0][0] - scored[1][0] < 0.02:
        return "?"
    return scored[0][1]


class RollingKeyDetector:
    """Live key estimate over a sliding window of recent audio.

    Keeps the last `window_s` seconds of mono audio in a deque-backed buffer.
    `feed` appends one block and recomputes the key at most every 0.5 s of new
    audio (so a live callback never pays for an FFT on every block); calls in
    between return the previous result. Returns None until enough audio
    (~1 s) has arrived for a first estimate.
    """

    def __init__(self, sample_rate: int, window_s: float = 10.0):
        self.sample_rate = sample_rate
        self.window_s = window_s
        self._capacity = max(1, int(window_s * sample_rate))
        self._need = min(self._capacity, sample_rate)  # ~1 s for a first result
        self._recompute_every = max(1, int(0.5 * sample_rate))
        self._buffer: Deque[np.ndarray] = deque()
        self._buffered = 0
        self._since_recompute = 0
        self._last: Optional[KeyResult] = None

    def feed(self, block: np.ndarray) -> Optional[KeyResult]:
        """Append `block` (mono or stereo) and return the current key estimate."""
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
            self._last = detect_key(audio, self.sample_rate)
            self._since_recompute = 0
        return self._last

    def reset(self) -> None:
        """Drop all buffered audio and the last result."""
        self._buffer.clear()
        self._buffered = 0
        self._since_recompute = 0
        self._last = None
