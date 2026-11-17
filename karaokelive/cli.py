"""Command line entry point."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

from . import audio_io
from .engines import ENGINES, create_engine
from .keydetect import NOTE_NAMES, detect_key
from .realtime import LiveRouter, format_devices, list_devices

DEFAULT_SR = 48000


def _level(value: str) -> float:
    level = float(value)
    if level > 1.0:
        level /= 100.0
    if not 0.0 <= level <= 1.0:
        raise argparse.ArgumentTypeError("vocal level is 0-100 (%) or 0.0-1.0")
    return level


def _engine(args, sample_rate: int):
    extra: dict = {}
    if args.engine == "center":
        extra = {"bass_hz": args.bass_hz, "mono_guard": not args.no_mono_guard}
    elif args.engine == "channel":
        extra = {"side": args.side}
    elif args.engine == "demucs":
        extra = {"chunk_s": args.chunk_s, "overlap_s": args.overlap_s}
    return create_engine(args.engine, sample_rate, args.vocal_level, **extra)


def cmd_devices(_args) -> int:
    print(format_devices(list_devices()))
    return 0


def cmd_file(args) -> int:
    info = audio_io.probe(args.input)
    if info.channels == 1 and args.engine == "center":
        print("warning: the source is mono -- center removal needs stereo; "
              "use --engine channel or a stereo source", file=sys.stderr)
    sample_rate = args.samplerate or max(DEFAULT_SR, info.sample_rate or DEFAULT_SR)
    print(f"loading {args.input} ({info.duration_s:.1f}s, {info.sample_rate} Hz, "
          f"{info.channels} ch) -> {sample_rate} Hz stereo")
    data = audio_io.load(args.input, sample_rate)
    engine = _engine(args, sample_rate)
    block = 8192
    chunks = []
    total = len(data)
    last_tenth = -1
    for start in range(0, total, block):
        chunks.append(engine.process(data[start:start + block]))
        done = min(start + block, total)
        tenth = int(done / max(1, total) * 10)
        if tenth != last_tenth:
            last_tenth = tenth
            print(f"\r{done / sample_rate:6.1f}s / {total / sample_rate:.1f}s",
                  end="", file=sys.stderr, flush=True)
    tail = engine.flush()
    print(file=sys.stderr)
    out = np.concatenate(chunks + ([tail] if len(tail) else []), axis=0)
    if engine.latency_frames:
        out = out[engine.latency_frames:engine.latency_frames + total]
    out = np.ascontiguousarray(out[:total] if len(out) >= total
                               else np.pad(out, ((0, total - len(out)), (0, 0))))
    audio_io.save(args.output, out, sample_rate)
    peak = float(np.max(np.abs(out))) if len(out) else 0.0
    print(f"wrote {args.output}  ({total / sample_rate:.1f}s, peak {peak:.3f})")
    return 0


def cmd_live(args) -> int:
    sample_rate = args.samplerate or DEFAULT_SR
    engine = _engine(args, sample_rate)
    router = LiveRouter(engine, args.capture, args.output, sample_rate,
                        args.blocksize)
    print("Ctrl-C to stop")
    router.run(seconds=args.seconds)
    return 0


def _chroma_bar(chroma: np.ndarray) -> str:
    """One-line chroma readout: 'C 12% ▏ D 3% ...' with a proportional block."""
    blocks = (" ", "▏", "▎", "▍", "▌", "▋", "▊", "▉", "█")
    parts = []
    peak = float(np.max(chroma)) if len(chroma) else 0.0
    for name, value in zip(NOTE_NAMES, chroma):
        level = int(round(8 * float(value) / peak)) if peak > 0 else 0
        parts.append(f"{name:>2} {float(value) * 100:3.0f}% {blocks[level]}")
    return "chroma: " + " ".join(parts)


def cmd_key(args) -> int:
    from .keydetect import KeyResult

    info = audio_io.probe(args.input)
    sample_rate = info.sample_rate or DEFAULT_SR
    data = audio_io.load(args.input, sample_rate)
    result: KeyResult = detect_key(data, sample_rate)
    if args.json:
        payload = asdict(result)
        payload["chroma"] = [float(v) for v in result.chroma]
        print(json.dumps(payload, indent=2))
        return 0
    print(f"เพลงนี้อยู่ key: {result.key} {result.mode}   "
          f"(Camelot {result.camelot})")
    print(f"ความมั่นใจ: {result.confidence * 100:.0f}%")
    alts = ", ".join(f"{label} (คะแนน {score:.2f})"
                     for label, score in result.alternatives)
    print(f"รองลงมา: {alts}")
    if result.confidence < 0.5:
        print("หมายเหตุ: ข้อมูลเสียงยังน้อย major/minor อาจสลับกันได้")
    print(_chroma_bar(result.chroma))
    return 0


def cmd_prerender(args) -> int:
    """Offline separation with the demucs CLI; the best-quality option."""
    import shutil
    import subprocess

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    cmd = [python, "-m", "demucs.separate", "--two-stems", "vocals",
           "-n", args.model, "-o", str(outdir), str(args.input)]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    stems = sorted(outdir.rglob("no_vocals.*"))
    if stems:
        print(f"\ninstrumental: {stems[0]}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="karaokelive",
        description="Reduce lead vocals from any audio, live or offline.",
    )
    parser.add_argument("--version", action="version", version="karaokelive 0.1.0")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("devices", help="list audio devices").set_defaults(func=cmd_devices)

    def add_engine_options(p):
        p.add_argument("--engine", choices=ENGINES, default="center",
                       help="center = phase cancel (no deps, no latency); "
                            "channel = take one channel (multiplex sources); "
                            "demucs = ML separation (needs torch)")
        p.add_argument("--vocal-level", type=_level, default=0.85,
                       help="how much vocal to remove: 0-100 (default 85)")
        p.add_argument("--samplerate", type=int, default=None)
        p.add_argument("--bass-hz", type=float, default=140.0,
                       help="centre content below this stays (center engine)")
        p.add_argument("--no-mono-guard", action="store_true",
                       help="allow 100%% removal, which cancels in mono downmix")
        p.add_argument("--side", choices=("left", "right"), default="right",
                       help="instrumental channel for --engine channel")
        p.add_argument("--chunk-s", type=float, default=2.5,
                       help="ML chunk length (demucs engine)")
        p.add_argument("--overlap-s", type=float, default=0.5,
                       help="ML latency / context (demucs engine)")

    p_file = sub.add_parser("file", help="process a file offline")
    p_file.add_argument("input")
    p_file.add_argument("output")
    add_engine_options(p_file)
    p_file.set_defaults(func=cmd_file)

    p_live = sub.add_parser("live", help="process system audio in real time")
    p_live.add_argument("--capture", required=True,
                        help='capture device name, e.g. "BlackHole 2ch"')
    p_live.add_argument("--output", required=True,
                        help='playback device name, e.g. "MacBook Pro Speakers"')
    p_live.add_argument("--blocksize", type=int, default=512)
    p_live.add_argument("--seconds", type=float, default=None,
                        help="stop after N seconds (default: run until Ctrl-C)")
    add_engine_options(p_live)
    p_live.set_defaults(func=cmd_live)

    p_key = sub.add_parser("key", help="detect the musical key of a file")
    p_key.add_argument("input")
    p_key.add_argument("--json", action="store_true",
                       help="print the full result as JSON")
    p_key.set_defaults(func=cmd_key)

    p_pre = sub.add_parser("prerender",
                           help="offline 2-stem separation with the demucs CLI")
    p_pre.add_argument("input")
    p_pre.add_argument("outdir", nargs="?", default="stems")
    p_pre.add_argument("--model", default="htdemucs")
    p_pre.set_defaults(func=cmd_prerender)

    p_ui = sub.add_parser("ui", help="serve the local web control panel")
    p_ui.add_argument("--port", type=int, default=8765)
    p_ui.add_argument("--host", default="127.0.0.1")
    p_ui.add_argument("--demo", action="store_true",
                      help="synthesize meters/key with no audio device")
    p_ui.set_defaults(func=cmd_ui)
    return parser


def cmd_ui(args) -> int:
    from .ui import ControlPanel

    panel = ControlPanel(port=args.port, host=args.host, demo=args.demo)
    print(f"เปิดแผงควบคุมที่ http://{args.host}:{args.port} "
          f"กด Ctrl-C เพื่อหยุด")
    try:
        panel.serve_forever()
    except KeyboardInterrupt:
        print("\nหยุดแผงควบคุมแล้ว")
    finally:
        panel.shutdown()
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
