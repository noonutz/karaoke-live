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


def to_pcm_wav(src: str | Path, dst: str | Path,
               sample_rate: int = 48000, bits: int = 16) -> str:
    """Transcode `src` to a plain-PCM WAV at `dst`; returns `str(dst)`.

    This conversion exists because stdlib `wave` — and therefore
    `StemPlayer` — can only read plain PCM WAVs (format tag `0x0001`).
    Demucs writes IEEE-float WAVs (tag `0x0003`) that `wave.open` rejects
    on every Python, and ffmpeg's 24-bit WAVs carry a
    `WAVE_FORMAT_EXTENSIBLE` header that older Pythons (e.g. 3.11) refuse,
    so separated stems must be normalized to plain PCM before playback.

    Decodes `src` and re-encodes stereo `bits`-bit PCM (`pcm_s16le` for
    `bits=16`, `pcm_s24le` for `bits=24`) at `sample_rate` Hz. 16-bit is
    the default because it is the widest-readable choice. Anything else
    for `bits` raises `ValueError`; a failed transcode raises
    `RuntimeError` carrying the ffmpeg stderr tail.
    """
    codec = {16: "pcm_s16le", 24: "pcm_s24le"}.get(int(bits))
    if codec is None:
        raise ValueError(f"bits must be 16 or 24, got {bits!r}")
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [_ffmpeg(), "-v", "error", "-y", "-i", str(src),
         "-c:a", codec, "-ar", str(sample_rate), "-ac", "2", str(dst)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-20:]
        detail = "\n".join(tail) if tail else f"exit code {proc.returncode}"
        raise RuntimeError(
            f"ffmpeg failed to transcode {src} to PCM WAV "
            f"(exit {proc.returncode}):\n{detail}")
    return str(dst)
