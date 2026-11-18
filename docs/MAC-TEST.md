# Testing on a Mac

Written before the machine arrived: these are the checks that cannot be done on the
headless Linux box the project was built on. Run them in order — each one fails in a
different place, so a pass at step 2 tells you step 1 is fine.

Setup once:

```bash
git clone https://github.com/noonutz/karaoke-live && cd karaoke-live
python3 -m venv .venv && .venv/bin/pip install -e '.[live,ml]'
brew install ffmpeg portaudio blackhole-2ch
```

## 1. Devices are visible

```bash
karaokelive devices
```

Pass: your interface, the speakers and `BlackHole 2ch` are listed with the right channel
counts. Fail: empty list or only "default" — portaudio did not build against CoreAudio.

## 2. The zero-latency path makes sound

Set the system output to **BlackHole 2ch**, then:

```bash
karaokelive live --capture "BlackHole 2ch" --output "<your interface>" --seconds 10
```

Pass: you hear the music, the vocal is thinner, meters move, no traceback.
Fail: silence — check the system output actually moved to BlackHole (the browser must go
quiet), and that `--output` is the device you are listening to, not the cable.

## 3. Latency is what it claims

Play a track with a clear beat, sing along, and listen for the slap-back. Then compare
with `karaokelive ui`'s latency read-out. Anything around 10-20 ms is inaudible; 100 ms+
means the buffer is too small and the callback is missing deadlines.

## 4. The UI and the overlay

```bash
karaokelive ui                    # http://127.0.0.1:8765
karaokelive ui --demo             # no audio device needed
```

Then add a Browser Source in OBS at `http://127.0.0.1:8765/overlay`, size 1000x200.
Pass: the page controls the stream, the overlay shows key/BPM/chord over the camera with
a transparent background.

## 5. The microphone range profile

```bash
karaokelive range --record 15     # sing your lowest and highest comfortable notes
karaokelive range --show
karaokelive key some_song.mp3     # should end with a fit verdict
```

Pass: a plausible range (not one octave, not the full keyboard) and a verdict line.
Fail: `--record` falls back to a Thai hint — the input device is not resolvable by name.

## 6. Four stems end to end

```bash
karaokelive queue add song.mp3
karaokelive queue run             # first run downloads the demucs weights
karaokelive ui                    # open the queue drawer, press play
```

Pass: the item reaches พร้อม, playback works, and the three play-along controls enable
(ตัดกลอง / วน 4 ห้อง / เสียงร้อง). The loop should land on a bar, not mid-word.

## 7. The things that can only break on hardware

- Unplug the interface mid-song: the panel must show a Thai error, not a traceback, and a
  restart must work.
- Switch the system output while playing: same.
- Let it run 10 minutes with `demucs` selected: watch for growing CPU or a drift between
  the vocal and the backing.

Send back: the command, what you expected, what happened, and the last ~30 lines of
output. That is enough to fix almost anything quickly.
