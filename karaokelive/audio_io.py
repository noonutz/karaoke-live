"""Audio decode/encode through ffmpeg.

Everything is float32, interleaved stereo, at the requested sample rate.
ffmpeg is the only hard external dependency: it decodes mp3/m4a/opus/wav and
encodes the result, so the Python side never grows a codec zoo.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class FFmpegMissing(RuntimeError):
    pass


def _ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise FFmpegMissing("ffmpeg not found in PATH (brew install ffmpeg)")
    return exe


def _ffprobe() -> str:
    exe = shutil.which("ffprobe")
    if not exe:
        raise FFmpegMissing("ffprobe not found in PATH (brew install ffmpeg)")
    return exe


@dataclass
class MediaInfo:
    path: str
    sample_rate: int
    channels: int
    duration_s: float


def probe(path: str | Path) -> MediaInfo:
    out = subprocess.run(
        [_ffprobe(), "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=sample_rate,channels:format=duration",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    data = json.loads(out)
    stream = (data.get("streams") or [{}])[0]
    return MediaInfo(
        path=str(path),
        sample_rate=int(stream.get("sample_rate") or 0),
        channels=int(stream.get("channels") or 0),
        duration_s=float((data.get("format") or {}).get("duration") or 0.0),
    )


def load(path: str | Path, sample_rate: int, channels: int = 2) -> np.ndarray:
    """Decode any media file to float32 (frames, channels) at `sample_rate`."""
    proc = subprocess.run(
        [_ffmpeg(), "-v", "error", "-i", str(path),
         "-f", "f32le", "-acodec", "pcm_f32le",
         "-ac", str(channels), "-ar", str(sample_rate), "-"],
        capture_output=True, check=True,
    )
    pcm = np.frombuffer(proc.stdout, dtype="<f4")
    if channels == 1:
        return pcm.astype(np.float32)
    return pcm.reshape(-1, channels).astype(np.float32, copy=False)


def save(path: str | Path, data: np.ndarray, sample_rate: int) -> None:
    """Encode float32 (frames, channels) to `path`; format comes from the suffix."""
    data = np.ascontiguousarray(np.clip(data, -1.0, 1.0), dtype="<f4")
    if data.ndim == 1:
        data = data[:, None]
    channels = data.shape[1]
    codec = []
    suffix = Path(path).suffix.lower()
    if suffix == ".wav":
        codec = ["-c:a", "pcm_s24le"]
    elif suffix == ".mp3":
        codec = ["-c:a", "libmp3lame", "-b:a", "192k"]
    elif suffix in (".m4a", ".aac"):
        codec = ["-c:a", "aac", "-b:a", "192k"]
    elif suffix == ".flac":
        codec = ["-c:a", "flac"]
    subprocess.run(
        [_ffmpeg(), "-v", "error", "-y", "-f", "f32le", "-ar", str(sample_rate),
         "-ac", str(channels), "-i", "-", *codec, str(path)],
        input=data.tobytes(), check=True,
    )
