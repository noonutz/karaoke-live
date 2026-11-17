"""Engine registry."""
from __future__ import annotations

from .base import Engine
from .center import CenterCancel
from .channel import ChannelPick

ENGINES = ("center", "channel", "demucs")


def create_engine(kind: str = "center", sample_rate: int = 48000,
                  vocal_level: float = 1.0, **options) -> Engine:
    kind = (kind or "center").lower()
    if kind == "center":
        return CenterCancel(sample_rate, vocal_level, **options)
    if kind == "channel":
        return ChannelPick(sample_rate, vocal_level, **options)
    if kind == "demucs":
        from .demucs_ml import DemucsEngine
        return DemucsEngine(sample_rate, vocal_level, **options)
    raise ValueError(f"unknown engine {kind!r} (choose from {', '.join(ENGINES)})")


__all__ = ["Engine", "CenterCancel", "ChannelPick", "create_engine", "ENGINES"]
