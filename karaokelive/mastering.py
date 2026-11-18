"""Stereo-linked peak safety limiter (numpy only, zero latency).

Every block that leaves toward a device runs through one of these:
centre-cancel can boost panned content by up to +6 dB and stems sum, so
without a limiter OBS can see more than 0 dBFS and the stream distorts.

Trade: no lookahead means it is a safety limiter, not a mastering-grade
one, and it is chosen so the live path adds zero latency. Attack cannot
catch the very first hot sample (that sample is hard-clipped to [-1, 1]),
and release is a plain slew — transparent enough to stop clipping, not
transparent enough to master with.
"""
from __future__ import annotations

import math

import numpy as np


class Limiter:
    """Stereo-linked peak limiter with no lookahead (zero added latency).

    No lookahead means it is a safety limiter, not a mastering-grade one,
    and it is chosen so the live path adds zero latency: attack cannot
    catch the very first hot sample (that sample is hard-clipped to
    [-1, 1]) instead of being anticipated.

    Per sample: ``peak = max(|L|, |R|)``, ``target = min(1, thr / peak)``,
    one shared gain slews toward the target (fast attack going down, slow
    release going up) and is applied equally to both channels so the stereo
    image does not shift. Anything that still overshoots (the attack cannot
    be instant) is hard-clipped to [-1, 1], so the true peak never exceeds
    1.0. Smoothing is per-sample and samplerate-driven, so splitting audio
    into different block sizes gives the same result.
    """

    def __init__(self, sample_rate: int, threshold_db: float = -1.0,
                 attack_ms: float = 1.0, release_ms: float = 120.0):
        if sample_rate <= 0:
            raise ValueError(f"bad sample_rate {sample_rate!r}")
        self.sample_rate = int(sample_rate)
        self.threshold_db = float(threshold_db)
        self.attack_ms = float(attack_ms)
        self.release_ms = float(release_ms)
        self._threshold = 10.0 ** (self.threshold_db / 20.0)
        attack_samples = max(1, int(round(self.sample_rate * self.attack_ms
                                          / 1000.0))) if self.attack_ms > 0 else 1
        release_samples = max(1, int(round(self.sample_rate * self.release_ms
                                           / 1000.0))) if self.release_ms > 0 else 1
        # Full-scale slew per sample: gain can fall from 1 to 0 in
        # attack_ms and climb back in release_ms, so any depth of reduction
        # fully recovers within release_ms of quiet.
        self._attack_step = 1.0 / float(attack_samples)
        self._release_step = 1.0 / float(release_samples)
        self._gain = 1.0

    def reset(self) -> None:
        """Restore unity gain (no reduction)."""
        self._gain = 1.0

    @property
    def gain_reduction_db(self) -> float:
        """Last applied reduction in dB (<= 0.0; 0.0 means untouched)."""
        if self._gain >= 1.0:
            return 0.0
        if self._gain <= 1e-9:
            return -180.0
        return float(20.0 * math.log10(self._gain))

    def process(self, block: np.ndarray) -> np.ndarray:
        """Limit one ``(frames, channels)`` block; stateful across calls."""
        x = np.asarray(block, dtype=np.float32)
        if x.size == 0:
            return np.zeros_like(x, dtype=np.float32)
        mono = x.ndim == 1
        if mono:
            x = x[:, None]
        n, channels = x.shape[0], x.shape[1]
        peaks = np.max(np.abs(x.astype(np.float64)), axis=1)
        thr = self._threshold
        gain = float(self._gain)
        attack_step = self._attack_step
        release_step = self._release_step
        gains = np.empty(n, dtype=np.float64)
        for i in range(n):
            peak = float(peaks[i])
            target = 1.0 if peak <= thr else thr / peak
            if target < gain:
                gain = target if gain - attack_step < target else gain - attack_step
            elif target > gain:
                gain = target if gain + release_step > target else gain + release_step
            gains[i] = gain
        self._gain = gain
        y = x.astype(np.float64) * gains[:, None]
        np.clip(y, -1.0, 1.0, out=y)
        out = np.ascontiguousarray(y, dtype=np.float32)
        if mono:
            return np.ascontiguousarray(out[:, 0], dtype=np.float32)
        return out
