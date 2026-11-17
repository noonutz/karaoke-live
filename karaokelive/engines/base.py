"""Engine interface: one stereo block in, one stereo block out.

`latency_frames` is how far the engine's output lags its input. The router
uses it to decide whether processing can run inside the audio callback
(latency 0, no ring buffers) or needs a worker thread and a delay line.
"""
from __future__ import annotations

import numpy as np


class Engine:
    name: str = "base"

    #: output lags input by this many frames at `sample_rate`
    latency_frames: int = 0

    def __init__(self, sample_rate: int, vocal_level: float = 1.0):
        self.sample_rate = sample_rate
        self.vocal_level = float(np.clip(vocal_level, 0.0, 1.0))

    def set_vocal_level(self, level: float) -> None:
        self.vocal_level = float(np.clip(level, 0.0, 1.0))

    def process(self, block: np.ndarray) -> np.ndarray:
        """block: float32 (frames, 2) -> float32 (frames, 2)"""
        raise NotImplementedError

    def flush(self) -> np.ndarray:
        """Drain any buffered audio (engines with latency only)."""
        return np.zeros((0, 2), dtype=np.float32)

    def reset(self) -> None:
        pass

    @property
    def used_frames(self) -> int:
        """Frames actually rendered so far (for progress reporting)."""
        return 0
