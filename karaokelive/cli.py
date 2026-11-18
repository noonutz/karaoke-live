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
from .range import Profile, estimate_range, fit_song, midi_to_name, name_to_midi
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
    profile = Profile.load()
    if profile is not None:
        try:
            song = estimate_range(data, sample_rate)
        except ValueError:
            print("ประเมินช่วงเสียงเพลงไม่ได้ (เสียงร้องน้อยเกินไป)")
        else:
            fit = fit_song(song.low_midi, song.high_midi, profile)
            print(f"ช่วงเสียงเพลง: {song.notes} "
                  f"(กว้าง {song.semitones():.1f} semitone) — {fit.message}")
    return 0


def _parse_range_arg(text: str) -> tuple[float, float]:
    """Parse ``LOW:HIGH`` note names (``A2:C5``) into MIDI, low first."""
    parts = str(text).split(":")
    if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
        raise argparse.ArgumentTypeError(
            'รูปแบบช่วงเสียงต้องเป็น "LOW:HIGH" เช่น A2:C5')
    try:
        low = name_to_midi(parts[0].strip())
        high = name_to_midi(parts[1].strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return (min(low, high), max(low, high))


def cmd_range(args) -> int:
    """Manage the singer's vocal-range profile."""
    if args.set is not None:
        low, high = _parse_range_arg(args.set)
        Profile(low=low, high=high).save()
        print(f"บันทึกช่วงเสียง: {midi_to_name(low)}-{midi_to_name(high)} "
              f"(กว้าง {high - low:.0f} semitone)")
        return 0
    if args.show:
        profile = Profile.load()
        if profile is None:
            print("ยังไม่ได้ตั้งช่วงเสียง (karaokelive range --set \"A2:C5\")")
            return 0
        print(f"ช่วงเสียงของคุณ: {midi_to_name(profile.low)}-"
              f"{midi_to_name(profile.high)} "
              f"(กว้าง {profile.high - profile.low:.0f} semitone)")
        return 0
    if args.from_file is not None:
        info = audio_io.probe(args.from_file)
        sample_rate = info.sample_rate or DEFAULT_SR
        data = audio_io.load(args.from_file, sample_rate)
        try:
            song = estimate_range(data, sample_rate)
        except ValueError:
            print("ประเมินช่วงเสียงไม่ได้: เสียงร้องน้อยเกินไป "
                  "(voiced ต่ำกว่า 5%) — ลองไฟล์ที่มีเสียงร้องชัดกว่านี้",
                  file=sys.stderr)
            return 1
        Profile(low=song.low_midi, high=song.high_midi).save()
        print(f"บันทึกช่วงเสียง: {song.notes} "
              f"(กว้าง {song.semitones():.1f} semitone, "
              f"voiced {song.voiced_ratio * 100:.0f}%)")
        return 0
    if args.record is not None:
        seconds = float(args.record)
        if seconds <= 0:
            print("error: ระยะเวลาอัดเสียงต้องมากกว่า 0 วินาที",
                  file=sys.stderr)
            return 2
        try:
            import sounddevice as sd
        except Exception:
            print("อัดเสียงไม่ได้: ไม่พบ sounddevice — "
                  "ใช้ karaokelive range --from-file <audio> แทน",
                  file=sys.stderr)
            return 1
        try:
            frames = int(seconds * DEFAULT_SR)
            print(f"กำลังอัดเสียง {seconds:.0f} วินาที... ร้องได้เลย")
            rec = sd.rec(frames, samplerate=DEFAULT_SR, channels=1,
                         dtype="float32", blocking=True)
        except Exception:
            print("อัดเสียงไม่ได้: ไม่มีอุปกรณ์รับเสียง — "
                  "ใช้ karaokelive range --from-file <audio> แทน",
                  file=sys.stderr)
            return 1
        try:
            song = estimate_range(np.asarray(rec).reshape(-1), DEFAULT_SR)
        except ValueError:
            print("ประเมินช่วงเสียงไม่ได้: เสียงร้องน้อยเกินไป "
                  "(voiced ต่ำกว่า 5%) — ลองไฟล์ที่มีเสียงร้องชัดกว่านี้",
                  file=sys.stderr)
            return 1
        Profile(low=song.low_midi, high=song.high_midi).save()
        print(f"บันทึกช่วงเสียง: {song.notes} "
              f"(กว้าง {song.semitones():.1f} semitone, "
              f"voiced {song.voiced_ratio * 100:.0f}%)")
        return 0
    print('กรุณาระบุ --from-file <audio>, --set "A2:C5", --show '
          'หรือ --record <วินาที>')
    return 2


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

    p_range = sub.add_parser("range", help="manage your vocal-range profile")
    p_range.add_argument("--from-file", default=None, metavar="AUDIO",
                         help="estimate and save the profile from an audio file")
    p_range.add_argument("--set", default=None, metavar='"LOW:HIGH"',
                         help='save a manual profile, e.g. --set "A2:C5"')
    p_range.add_argument("--show", action="store_true",
                         help="print the saved profile")
    p_range.add_argument("--record", default=None, metavar="SECONDS", type=float,
                         help="record N seconds from the default input device")
    p_range.set_defaults(func=cmd_range)

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

    p_queue = sub.add_parser("queue", help="pre-render queue of songs to separate")
    qsub = p_queue.add_subparsers(dest="queue_command", required=True)

    p_add = qsub.add_parser("add", help="add songs to the queue")
    p_add.add_argument("source", nargs="+")
    p_add.add_argument("--title", default=None)
    p_add.set_defaults(func=cmd_queue_add)

    qsub.add_parser("list", help="list queued songs").set_defaults(
        func=cmd_queue_list)

    p_remove = qsub.add_parser("remove", help="remove a song from the queue")
    p_remove.add_argument("id")
    p_remove.set_defaults(func=cmd_queue_remove)

    p_clear = qsub.add_parser("clear", help="drop queued songs")
    p_clear.add_argument("--status",
                         choices=("pending", "rendering", "ready", "failed"),
                         default=None,
                         help="clear only this status (default: everything)")
    p_clear.set_defaults(func=cmd_queue_clear)

    p_run = qsub.add_parser("run", help="separate everything queued")
    p_run.add_argument("--outdir", default=None)
    p_run.add_argument("--model", default="htdemucs")
    p_run.set_defaults(func=cmd_queue_run)
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


QUEUE_STATUS_TH = {
    "pending": "รอคิว",
    "rendering": "กำลังแยก",
    "ready": "พร้อม",
    "failed": "ผิดพลาด",
}


def _fmt_duration(seconds) -> str:
    if seconds is None:
        return "-"
    try:
        total = int(float(seconds))
    except (TypeError, ValueError):
        return "-"
    if total < 0:
        return "-"
    return f"{total // 60}:{total % 60:02d}"


def cmd_queue_add(args) -> int:
    from .setlist import Setlist

    if args.title is not None and len(args.source) > 1:
        print("error: --title ใช้ได้เมื่อเพิ่มทีละไฟล์เท่านั้น",
              file=sys.stderr)
        return 2
    sl = Setlist()
    failed = 0
    for source in args.source:
        title = args.title if len(args.source) == 1 else None
        try:
            item = sl.add(source, title=title)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            failed = 1
            continue
        print(f"เพิ่มเข้าคิว: {item.title} ({item.id})")
    return failed


def cmd_queue_list(_args) -> int:
    from .setlist import Setlist

    sl = Setlist()
    if sl.load_error is not None:
        print(sl.load_error)
    items = sl.items()
    if not items:
        print("คิวว่าง")
        return 0
    for item in items:
        status = QUEUE_STATUS_TH.get(item.status, item.status)
        duration = _fmt_duration(item.duration_s)
        print(f"[{status}] {item.title} ({item.id}) {duration}")
    return 0


def cmd_queue_remove(args) -> int:
    from .setlist import Setlist

    if Setlist().remove(args.id):
        print(f"ลบออกจากคิว: {args.id}")
        return 0
    print(f"error: ไม่พบ {args.id} ในคิว", file=sys.stderr)
    return 1


def cmd_queue_clear(args) -> int:
    from .setlist import Setlist

    count = Setlist().clear(status=args.status)
    print(f"ล้างคิวแล้ว ({count} รายการ)")
    return 0


def cmd_queue_run(args) -> int:
    import time

    from .prerender import PrerenderWorker, separate_file
    from .setlist import Setlist, default_state_dir

    sl = Setlist()
    waiting = [item for item in sl.items()
               if item.status in ("pending", "rendering")]
    if not waiting:
        print("คิวว่าง ไม่มีอะไรต้องแยก")
        return 0
    outdir = args.outdir or str(default_state_dir() / "stems")
    model = args.model
    worker = PrerenderWorker(
        sl, outdir,
        separator=lambda src, out: separate_file(src, out, model=model),
        poll_s=1.0,
    )
    print(f"เริ่มแยกเสียง {len(waiting)} รายการ กด Ctrl-C เพื่อหยุด")
    known = {item.id: item.status for item in sl.items()}
    worker.start()
    try:
        while worker.is_alive():
            time.sleep(0.5)
            for item in sl.items():
                if known.get(item.id) == item.status:
                    continue
                known[item.id] = item.status
                if item.status == "rendering":
                    print(f"กำลังแยก: {item.title} ({item.id})")
                elif item.status == "ready":
                    print(f"พร้อม: {item.title} -> {item.instrumental}")
                elif item.status == "failed":
                    print(f"ผิดพลาด: {item.title} ({item.error})")
            if not any(item.status in ("pending", "rendering")
                       for item in sl.items()):
                print("เสร็จแล้ว")
                break
    except KeyboardInterrupt:
        # The interrupt lands on this thread only: stop the worker promptly
        # and leave the current item's status untouched (a "rendering" item
        # is picked up again on the next run, never marked failed by this).
        worker.stop(timeout=5)
        print("\nหยุดแล้ว")
    finally:
        worker.stop(timeout=5)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
