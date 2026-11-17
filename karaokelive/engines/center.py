"""Center-cancel engine: the classic karaoke trick, done carefully.

`out = gain * (mix - k * highpassed(mid))`

- Everything panned to the centre (lead vocal, kick, bass, snare) is what gets
  removed. `k` widens that to 0-100%, the same control consumer speakers expose
  as 25/50/100.
- The subtraction uses a **4th-order Butterworth** lowpass, not a one-pole: a
  one-pole at 140 Hz only attenuates a 440 Hz vocal by ~10 dB, which leaks a
  clearly audible ghost back in. Four poles put it ~40 dB down.
- `output_gain` compensates the level loss the subtraction causes on
  panned content (a side-only instrument loses `1 - k/2`), so turning the knob
  up does not make the backing quieter.
- No buffering, so it runs inside the audio callback with zero added latency.
"""
from __future__ import annotations

import math

import numpy as np

from .base import Engine


class Biquad:
    """Direct-form I biquad with explicit state (RBJ lowpass coefficients)."""

    def __init__(self, cutoff_hz: float, sample_rate: int, q: float = 0.70710678):
        self.sample_rate = sample_rate
        self.set(cutoff_hz, q)
        self.x1 = self.x2 = self.y1 = self.y2 = 0.0

    def set(self, cutoff_hz: float, q: float = 0.70710678) -> None:
        cutoff_hz = max(10.0, min(float(cutoff_hz), self.sample_rate * 0.45))
        w0 = 2.0 * math.pi * cutoff_hz / self.sample_rate
        cos_w0, sin_w0 = math.cos(w0), math.sin(w0)
        alpha = sin_w0 / (2.0 * q)
        b0 = (1.0 - cos_w0) / 2.0
        b1 = 1.0 - cos_w0
        b2 = b0
        a0 = 1.0 + alpha
        a1 = -2.0 * cos_w0
        a2 = 1.0 - alpha
        self.b0, self.b1, self.b2 = b0 / a0, b1 / a0, b2 / a0
        self.a1, self.a2 = a1 / a0, a2 / a0
        self.cutoff_hz = cutoff_hz

    def process(self, x: np.ndarray) -> np.ndarray:
        b0, b1, b2, a1, a2 = self.b0, self.b1, self.b2, self.a1, self.a2
        x1, x2, y1, y2 = self.x1, self.x2, self.y1, self.y2
        out = np.empty(len(x), dtype=np.float32)
        for i in range(len(x)):
            xi = float(x[i])
            yi = b0 * xi + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
            x2, x1 = x1, xi
            y2, y1 = y1, yi
            out[i] = yi
        self.x1, self.x2, self.y1, self.y2 = x1, x2, y1, y2
        return out

    def reset(self) -> None:
        self.x1 = self.x2 = self.y1 = self.y2 = 0.0


class ButterworthLowpass:
    """4th-order (two cascaded biquads) lowpass, flat in the passband."""

    QS = (0.54119610, 1.30656296)

    def __init__(self, cutoff_hz: float, sample_rate: int):
        self.stages = [Biquad(cutoff_hz, sample_rate, q) for q in self.QS]
        self.cutoff_hz = cutoff_hz

    def process(self, x: np.ndarray) -> np.ndarray:
        for stage in self.stages:
            x = stage.process(x)
        return x

    def reset(self) -> None:
        for stage in self.stages:
            stage.reset()


class CenterCancel(Engine):
    name = "center"
    latency_frames = 0

    def __init__(self, sample_rate: int, vocal_level: float = 1.0,
                 bass_hz: float = 140.0, mono_guard: bool = True):
        super().__init__(sample_rate, vocal_level)
        self.bass_hz = float(bass_hz)
        self.mono_guard = bool(mono_guard)
        self._low = ButterworthLowpass(self.bass_hz, sample_rate)
        self.rendered = 0

    @property
    def effective_level(self) -> float:
        # at k=1 the output is anti-phase and a mono downmix cancels it entirely
        return min(self.vocal_level, 0.85) if self.mono_guard else self.vocal_level

    @property
    def output_gain(self) -> float:
        # a side-only component comes out at (1 - k/2); restore it exactly
        return 1.0 / (1.0 - 0.5 * self.effective_level)

    def process(self, block: np.ndarray) -> np.ndarray:
        if block.ndim == 1:
            block = np.stack([block, block], axis=1)
        if block.shape[1] == 1:
            block = np.repeat(block, 2, axis=1)
        left = block[:, 0]
        right = block[:, 1]
        k = self.effective_level
        self.rendered += len(block)
        if k <= 0.0:
            return block.astype(np.float32, copy=True)
        mid = 0.5 * (left + right)
        low = self._low.process(mid)
        sub = (k * (mid - low)).astype(np.float32)
        gain = self.output_gain
        out = np.empty_like(block)
        out[:, 0] = (left - sub) * gain
        out[:, 1] = (right - sub) * gain
        return np.clip(out, -1.0, 1.0)

    def reset(self) -> None:
        self._low.reset()
        self.rendered = 0

    @property
    def used_frames(self) -> int:
        return self.rendered
