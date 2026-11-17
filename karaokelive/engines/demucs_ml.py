"""Optional ML engine: real Demucs separation, chunked for live use.

Contract: `process(block)` always returns exactly `len(block)` frames, and the
stream is delayed by `latency_frames` (the caller drops that many frames from
the front to align).

Why the latency is a whole chunk: a non-causal model needs the future as well
as the past, so nothing can be emitted until the samples after it have arrived.
Each chunk therefore covers `history + emit + lookahead` and only the middle is
taken, which makes the stream latency `chunk_s` (plus the crossfade and
lookahead). That is the honest trade: shorter chunks = less latency = less
context for the model.

Seams: consecutive chunks overlap by `xfade_ms` and are summed with
complementary linear windows (they add to exactly 1), so there is neither a
click nor a level dip at a boundary.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from .base import Engine

DEFAULT_MODEL = "htdemucs"


class DemucsEngine(Engine):
    name = "demucs"

    def __init__(self, sample_rate: int, vocal_level: float = 1.0,
                 model_name: str = DEFAULT_MODEL, device: str | None = None,
                 chunk_s: float = 2.5, overlap_s: float = 0.5,
                 xfade_ms: float = 40.0):
        super().__init__(sample_rate, vocal_level)
        self.model_name = model_name
        self.device = device
        self.hop_frames = max(1024, int((chunk_s - overlap_s) * sample_rate))
        self.overlap_frames = max(0, int(overlap_s * sample_rate))
        self.xfade = max(2, int(xfade_ms * sample_rate / 1000))
        self.emit_frames = self.hop_frames + self.xfade
        self.latency_frames = self.hop_frames + self.xfade + self.overlap_frames
        self._separator = None
        self._model_sr = 44100
        # `_buf` always starts `overlap` frames before the next chunk to emit;
        # the head of the stream is pre-padded with that silent history.
        self._buf = np.zeros((self.overlap_frames, 2), dtype=np.float32)
        self._acc = np.zeros((0, 2), dtype=np.float32)
        self._pos = 0
        self._out: deque[np.ndarray] = deque(
            [np.zeros((self.latency_frames, 2), np.float32)])
        self._out_frames = self.latency_frames
        self._input_frames = 0
        self._drained = 0
        self._eof = False
        self.rendered = 0

    # -- model ------------------------------------------------------------
    def _load(self):
        if self._separator is not None:
            return self._separator
        try:
            import torch  # noqa: F401
            from demucs.api import Separator
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "the demucs engine needs PyTorch and demucs: "
                "pip install 'demucs>=4.0'  (the center engine needs neither)"
            ) from exc
        device = self.device
        if device is None:
            import torch
            mps = getattr(torch.backends, "mps", None)
            device = "mps" if mps is not None and mps.is_available() else "cpu"
        self._separator = Separator(model=self.model_name, device=device,
                                    progress=False)
        self._model_sr = int(getattr(self._separator, "samplerate", 44100))
        return self._separator

    def _separate(self, chunk: np.ndarray) -> np.ndarray:
        """chunk (frames, 2) float32 -> vocals (frames, 2) float32."""
        import torch
        import torchaudio.functional as AF

        sep = self._load()
        tensor = torch.from_numpy(np.ascontiguousarray(chunk.T))
        if self._model_sr != self.sample_rate:
            tensor = AF.resample(tensor, self.sample_rate, self._model_sr)
        with torch.inference_mode():
            _origin, separated = sep.separate_tensor(tensor, sr=self._model_sr)
        vocals = separated["vocals"]
        if self._model_sr != self.sample_rate:
            vocals = AF.resample(vocals, self._model_sr, self.sample_rate)
        vocals = vocals.detach().cpu().numpy().T.astype(np.float32)
        if vocals.shape[0] < chunk.shape[0]:
            vocals = np.pad(vocals, ((0, chunk.shape[0] - vocals.shape[0]),
                                     (0, 0)))
        return vocals[:chunk.shape[0]]

    # -- streaming --------------------------------------------------------
    @property
    def _chunk_frames(self) -> int:
        return self.overlap_frames + self.emit_frames + self.overlap_frames

    def _renderable(self) -> bool:
        return len(self._buf) >= self._chunk_frames

    def _render(self, final: bool = False) -> None:
        chunk = self._buf[:self._chunk_frames]
        if len(chunk) < self._chunk_frames:
            if not final:
                return
            missing = self._chunk_frames - len(chunk)
            chunk = np.concatenate([chunk, np.zeros((missing, 2), np.float32)])
        vocals = self._separate(chunk)
        start = self.overlap_frames
        mix_part = chunk[start:start + self.emit_frames]
        voc_part = vocals[start:start + self.emit_frames]
        seg = (mix_part - self.vocal_level * voc_part).astype(np.float32)

        # complementary windows: fade in at the head, out at the tail, and let
        # the neighbouring chunk supply the other half of the ramp
        window = np.ones(self.emit_frames, dtype=np.float32)
        if self._pos > 0:
            window[:self.xfade] = np.linspace(0.0, 1.0, self.xfade,
                                              dtype=np.float32)
        if not final:
            window[-self.xfade:] = np.linspace(1.0, 0.0, self.xfade,
                                               dtype=np.float32)
        if len(self._acc) < self.emit_frames:
            self._acc = np.zeros((self.emit_frames, 2), dtype=np.float32)
        self._acc[:self.emit_frames] += seg * window[:, None]

        # the first `hop` frames are complete: nothing else will touch them
        committed = np.clip(self._acc[:self.hop_frames], -1.0, 1.0)
        self._out.append(committed.astype(np.float32))
        self._out_frames += len(committed)
        self.rendered += len(committed)
        self._acc = np.concatenate(
            [self._acc[self.hop_frames:],
             np.zeros((self.hop_frames, 2), dtype=np.float32)], axis=0)
        self._pos += self.hop_frames
        self._buf = self._buf[self.hop_frames:]

    def process(self, block: np.ndarray) -> np.ndarray:
        if self._eof:
            raise RuntimeError("engine already flushed")
        if block.ndim == 1:
            block = np.stack([block, block], axis=1)
        block = block.astype(np.float32, copy=False)
        self._buf = np.concatenate([self._buf, block], axis=0)
        self._input_frames += len(block)
        while self._renderable():
            self._render()
        return self._drain(len(block))

    def _drain(self, frames: int) -> np.ndarray:
        out = np.zeros((frames, 2), dtype=np.float32)
        filled = 0
        while filled < frames and self._out:
            head = self._out[0]
            take = min(frames - filled, len(head))
            out[filled:filled + take] = head[:take]
            filled += take
            if take == len(head):
                self._out.popleft()
            else:
                self._out[0] = head[take:]
            self._out_frames -= take
        self._drained += filled
        return out

    def flush(self) -> np.ndarray:
        """Render the tail and return exactly the frames still owed."""
        self._eof = True
        while self._renderable():
            self._render()
        if len(self._buf) > self.overlap_frames:
            self._render(final=True)
        owed = self._input_frames + self.latency_frames - self._drained
        if owed <= 0:
            return np.zeros((0, 2), dtype=np.float32)
        return self._drain(owed)

    def reset(self) -> None:
        self._buf = np.zeros((self.overlap_frames, 2), dtype=np.float32)
        self._acc = np.zeros((0, 2), dtype=np.float32)
        self._pos = 0
        self._out.clear()
        self._out.append(np.zeros((self.latency_frames, 2), np.float32))
        self._out_frames = self.latency_frames
        self._input_frames = 0
        self._drained = 0
        self._eof = False
        self.rendered = 0

    @property
    def used_frames(self) -> int:
        return self.rendered
