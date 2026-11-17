"""Vocal range estimation and song-fit advice (numpy only, no new dependencies).

Method (`pitch_track`): the input (mono or stereo) is downmixed to mono,
sliced into rectangular frames (2048 samples, hop 512), and each frame's
fundamental is estimated with normalised autocorrelation::

    r[lag] = sum_n x[n] * x[n+lag] / r[0]

computed via FFT (power spectrum -> inverse FFT). The winning lag in the
``sample_rate / fmax .. sample_rate / fmin`` range is refined with a
parabolic interpolation over its two neighbours, and ``f0 = sr / lag``.
Frames with near-silence (RMS below 0.01) or a weak periodic peak
(``r < 0.35``) are reported as 0.0 (unvoiced).

Limits: autocorrelation assumes one dominant periodic source, so it
reports the strongest voice in a mix (backing instruments leak in),
halves/doubles on strong harmonics or sub-harmonics (octave errors --
`estimate_range` trims these with the 5th/95th percentiles), needs
~43 ms of audio per frame (2048 samples at 48 kHz), and degrades on
breathy/unvoiced consonants, heavy reverb, and polyphonic material.
It is a guide for karaoke key choice, not a laboratory pitch tracker.
"""
from __future__ import annotations

import json
import math
import os
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .setlist import default_state_dir

WINDOW = 2048
HOP = 512

#: Pitch-class names, index 0 = C (matches keydetect.NOTE_NAMES).
NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F",
              "F#", "G", "G#", "A", "A#", "B")

_BASE_SEMITONE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_NAME_RE = re.compile(r"^([A-Ga-g])([#b]?)(-?\d+)$")

_RMS_FLOOR = 0.01
_PEAK_FLOOR = 0.35


def _to_mono(samples: np.ndarray) -> np.ndarray:
    """Downmix (frames, 2) or (frames,) float audio to mono float64."""
    audio = np.asarray(samples, dtype=np.float64)
    if audio.ndim == 2:
        return audio.mean(axis=1)
    return audio.reshape(-1)


def pitch_track(samples: np.ndarray, sample_rate: int,
                fmin: float = 70.0, fmax: float = 1200.0) -> np.ndarray:
    """Estimate f0 per frame; see the module docstring for method/limits."""
    mono = _to_mono(samples)
    sr = int(sample_rate)
    if sr <= 0 or mono.size == 0:
        return np.zeros(0, dtype=np.float64)
    lag_min = max(1, int(round(sr / fmax)))
    lag_max = min(WINDOW - 2, int(round(sr / fmin)))
    if lag_max <= lag_min:
        return np.zeros(0, dtype=np.float64)
    nfft = 1
    while nfft < 2 * WINDOW:
        nfft *= 2
    if len(mono) < WINDOW:
        mono = np.pad(mono, (0, WINDOW - len(mono)))
    f0: list[float] = []
    for start in range(0, len(mono) - WINDOW + 1, HOP):
        frame = mono[start:start + WINDOW]
        x = frame - float(frame.mean())
        rms = float(np.sqrt(np.mean(x * x)))
        if rms < _RMS_FLOOR:
            f0.append(0.0)
            continue
        spec = np.fft.rfft(x, n=nfft)
        corr = np.fft.irfft(spec * np.conj(spec), n=nfft)[:WINDOW]
        r0 = float(corr[0].real)
        if r0 <= 1e-12:
            f0.append(0.0)
            continue
        r = np.asarray(corr.real, dtype=np.float64) / r0
        segment = r[lag_min:lag_max + 1]
        best = int(np.argmax(segment))
        peak = float(segment[best])
        if peak < _PEAK_FLOOR:
            f0.append(0.0)
            continue
        lag = lag_min + best
        if 0 < lag < WINDOW - 1:
            y0, y1, y2 = r[lag - 1], r[lag], r[lag + 1]
            denom = float(y0 - 2.0 * y1 + y2)
            if denom < 0.0:
                shift = 0.5 * float(y0 - y2) / denom
                lag = lag + max(-1.0, min(1.0, shift))
        f0.append(float(sr) / float(lag) if lag > 0 else 0.0)
    if not f0:
        f0.append(0.0)
    return np.array(f0, dtype=np.float64)


def midi_to_name(midi: float) -> str:
    """Nearest MIDI note to scientific pitch name, e.g. 60 -> ``C4``."""
    number = int(round(float(midi)))
    return f"{NOTE_NAMES[number % 12]}{number // 12 - 1}"


def name_to_midi(name: str) -> float:
    """Scientific pitch name (``A2``, ``C#5``, ``Bb4``) to fractional MIDI."""
    match = _NAME_RE.match(str(name).strip())
    if not match:
        raise ValueError(f"bad note name: {name!r} (want e.g. A2, C#5, Bb4)")
    letter, accidental, octave = match.groups()
    semitone = _BASE_SEMITONE[letter.upper()]
    if accidental == "#":
        semitone += 1
    elif accidental == "b":
        semitone -= 1
    return float((int(octave) + 1) * 12 + semitone)


def _hz_to_midi(freq_hz: float) -> float:
    return 69.0 + 12.0 * math.log2(freq_hz / 440.0)


@dataclass
class RangeResult:
    """A robust vocal-range estimate (all pitch fields in MIDI)."""

    low_midi: float
    high_midi: float
    p05: float  # 5th percentile of the voiced track, in MIDI
    p95: float  # 95th percentile of the voiced track, in MIDI
    voiced_ratio: float  # fraction of frames that were voiced
    notes: str  # e.g. "A2-C5"

    def semitones(self) -> float:
        """Width of the estimated range in semitones."""
        return float(self.high_midi - self.low_midi)


def estimate_range(samples: np.ndarray, sample_rate: int) -> RangeResult:
    """Estimate the sung range from `samples` (mono or stereo float audio).

    The 5th/95th percentiles of the voiced f0 track are converted to MIDI;
    the percentiles (rather than min/max) make a couple of octave-error
    frames harmless. Raises `ValueError` when fewer than 5% of frames are
    voiced (silence, noise, or purely instrumental material).
    """
    f0 = pitch_track(samples, int(sample_rate))
    voiced = f0[f0 > 0.0]
    ratio = float(len(voiced) / len(f0)) if len(f0) else 0.0
    if len(voiced) == 0 or ratio < 0.05:
        raise ValueError(
            f"ประเมินช่วงเสียงไม่ได้: เสียงร้องน้อยเกินไป "
            f"(voiced {ratio * 100:.0f}%)")
    p05_hz = float(np.percentile(voiced, 5))
    p95_hz = float(np.percentile(voiced, 95))
    low = _hz_to_midi(p05_hz)
    high = _hz_to_midi(p95_hz)
    return RangeResult(
        low_midi=low,
        high_midi=high,
        p05=low,
        p95=high,
        voiced_ratio=ratio,
        notes=f"{midi_to_name(low)}-{midi_to_name(high)}",
    )


@dataclass
class Profile:
    """The singer's comfortable range, persisted as JSON on disk."""

    low: float  # MIDI, bottom of the comfortable range
    high: float  # MIDI, top of the comfortable range
    updated_at: float = 0.0

    @staticmethod
    def path() -> Path:
        """JSON location: ``default_state_dir()/profile.json``."""
        return default_state_dir() / "profile.json"

    @classmethod
    def load(cls) -> "Profile | None":
        """Read the saved profile; `None` when missing or unreadable."""
        try:
            raw = json.loads(cls.path().read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, ValueError):
            return None
        if not isinstance(raw, dict):
            return None
        try:
            low = float(raw["low"])
            high = float(raw["high"])
        except (KeyError, TypeError, ValueError):
            return None
        try:
            updated_at = float(raw.get("updated_at", 0.0))
        except (TypeError, ValueError):
            updated_at = 0.0
        return cls(low=min(low, high), high=max(low, high),
                   updated_at=updated_at)

    def save(self) -> "Profile":
        """Write the profile atomically (tmp file + os.replace)."""
        path = self.path()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.updated_at = time.time()
        payload = json.dumps(self.as_dict(), indent=2, ensure_ascii=False)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent),
                                   prefix=path.name + ".", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        return self

    def as_dict(self) -> dict:
        """JSON-serialisable view (exactly the on-disk fields)."""
        return {"low": float(self.low), "high": float(self.high),
                "updated_at": float(self.updated_at)}

    def fits(self, low: float, high: float) -> bool:
        """`True` when the song range sits inside the profile."""
        lo, hi = (min(low, high), max(low, high))
        return bool(lo >= self.low and hi <= self.high)

    def semitones_for(self, low: float, high: float) -> int:
        """Integer transposition that fits the song inside the profile.

        0 when it already fits; otherwise the smallest shift that makes
        it fit, searching outward from 0 and preferring down (negative)
        on ties. When the song is wider than the profile no shift fits,
        so the centred shift is returned as a best effort.
        """
        lo, hi = (min(low, high), max(low, high))
        if self.fits(lo, hi):
            return 0
        for step in range(1, 49):
            if self.fits(lo - step, hi - step):
                return -step
            if self.fits(lo + step, hi + step):
                return step
        return int(round((self.low + self.high - lo - hi) / 2.0))


@dataclass
class FitResult:
    """How a song range fits a singer's profile."""

    shift: int  # semitones to transpose (negative = down)
    verdict: str  # one of "ok", "too_high", "too_low"
    message: str  # Thai advice for the panel/CLI


def fit_song(song_low: float, song_high: float,
             profile: Profile) -> FitResult:
    """Compare a song range against `profile` and advise a transposition."""
    lo, hi = (min(song_low, song_high), max(song_low, song_high))
    shift = profile.semitones_for(lo, hi)
    if shift == 0 and profile.fits(lo, hi):
        return FitResult(shift=0, verdict="ok", message="ร้องได้สบาย")
    if shift == 0:  # wider than the profile: name the worse side
        over_top = hi - profile.high
        over_bottom = profile.low - lo
        if over_top >= over_bottom:
            shift = -int(math.ceil(over_top))
            return FitResult(
                shift=shift, verdict="too_high",
                message=(f"สูงเกินช่วงเสียง {-shift} semitone "
                         f"— แนะนำลดคีย์ {-shift}"))
        shift = int(math.ceil(over_bottom))
        return FitResult(
            shift=shift, verdict="too_low",
            message=(f"ต่ำเกินช่วงเสียง {shift} semitone "
                     f"— แนะนำเพิ่มคีย์ {shift}"))
    if shift < 0:
        return FitResult(
            shift=shift, verdict="too_high",
            message=(f"สูงเกินช่วงเสียง {-shift} semitone "
                     f"— แนะนำลดคีย์ {-shift}"))
    return FitResult(
        shift=shift, verdict="too_low",
        message=(f"ต่ำเกินช่วงเสียง {shift} semitone "
                 f"— แนะนำเพิ่มคีย์ {shift}"))


__all__ = [
    "WINDOW", "HOP", "NOTE_NAMES",
    "RangeResult", "Profile", "FitResult",
    "pitch_track", "estimate_range",
    "midi_to_name", "name_to_midi", "fit_song",
]
