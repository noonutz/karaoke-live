"""Multiplex source picker.

Some karaoke sources (old VCD/tape rips, a few Thai karaoke uploads) put the
guide vocal hard on one channel and the backing track on the other. For those,
no phase trick is needed and none works better: just take the instrumental
channel and duplicate it.
"""
from __future__ import annotations

import numpy as np

from .base import Engine


class ChannelPick(Engine):
    name = "channel"
    latency_frames = 0

    def __init__(self, sample_rate: int, vocal_level: float = 0.0,
                 side: str = "right"):
        super().__init__(sample_rate, vocal_level)
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")
        self.side = side
        self.rendered = 0

    def process(self, block: np.ndarray) -> np.ndarray:
        if block.ndim == 1:
            block = np.stack([block, block], axis=1)
        index = 0 if self.side == "left" else 1
        if block.shape[1] == 1:
            index = 0
        mono = block[:, index]
        self.rendered += len(block)
        return np.stack([mono, mono], axis=1)

    def reset(self) -> None:
        self.rendered = 0

    @property
    def used_frames(self) -> int:
        return self.rendered
