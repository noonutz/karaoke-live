# karaoke-live

**English** · [ไทย](README.th.md)

![The control panel](docs/control-panel.png)

Remove the lead vocal from whatever your computer is playing, live, and sing
over it. Built for the case the existing tools miss: not "hear the voices in a
movie", but "keep the music, drop the singer".

Three engines, same interface:

| engine | what it does | latency | deps |
|---|---|---|---|
| `center` (default) | phase-cancel the centre image, bass restored, 0-100% blend | **0** | numpy + ffmpeg |
| `channel` | take one channel (multiplex sources: vocal hard left, music right) | **0** | numpy + ffmpeg |
| `demucs` | real ML 2-stem separation, chunked | **~chunk_s** (default 2.5 s) | + torch, demucs |

`center` is the one that always works: no model download, no GPU, no added
latency, and it is the same trick karaoke amplifiers have used since the 90s --
just with a 4th-order crossover so the low end survives and a level-compensated
blend so turning the knob up does not make the backing quieter.

`demucs` sounds better (keeps stereo, keeps drums, no anti-phase ghost) at the
cost of one chunk of latency. It is the honest trade: a non-causal model cannot
output anything until it has heard the samples after it.

## Install

```bash
git clone https://github.com/<you>/karaoke-live && cd karaoke-live
python3 -m venv .venv && .venv/bin/pip install -e .        # center/channel only
.venv/bin/pip install -e '.[live,ml]'                     # + realtime audio, + demucs
```

macOS also needs `brew install ffmpeg portaudio`; Windows needs ffmpeg on PATH.

## Use

```bash
# what devices exist?
karaokelive devices

# system audio -> your headphones, 70% vocal removal
karaokelive live --capture "BlackHole 2ch" --output "MacBook Pro Speakers" --vocal-level 70

# better quality, ~2.5 s behind
karaokelive live --capture "BlackHole 2ch" --output "MacBook Pro Speakers" \
    --engine demucs --vocal-level 100

# offline: render an instrumental file
karaokelive file song.mp3 song_karaoke.wav --vocal-level 100 --no-mono-guard

# offline, best quality: real stems via the demucs CLI
karaokelive prerender song.mp3 stems/

# what key is this song in?
karaokelive key song.mp3 --json

# local control panel (phones and tablets on the same machine only)
karaokelive ui            # http://127.0.0.1:8765
karaokelive ui --demo     # exercise the UI with no audio device at all

# a setlist: render instrumentals in the background while you sing
karaokelive queue add song1.mp3 song2.flac
karaokelive queue list
karaokelive queue run     # demucs, one item at a time

# your comfortable singing range, then a fit verdict for every song
karaokelive range --from-file my_voice.wav
karaokelive range --set "A2:C5"
```

Stop with Ctrl-C.

## Key detection

`karaokelive key` reports the tonic, the mode, a confidence, the Camelot code and
the 12-bin chroma. It is a chromagram (Hann STFT 4096/2048, 55 Hz - 4 kHz, mapped
to pitch classes with `midi = 69 + 12*log2(f/440)`) correlated against the
Krumhansl-Schmuckler major and minor profiles, rotated over all 24 keys.

- Confidence blends the winning correlation with its margin over the runner-up, so
  a genuinely ambiguous track (a bare fifth with no third, a single held chord) reads
  low instead of guessing loudly. Below 50% the CLI says so in words.
- The runners-up line shows **correlation scores**, not confidence -- a different
  scale, labelled as such so 0.70 next to 48% cannot be misread as a winner.
- Verified on synthesised progressions in all 12 major and all 12 minor keys:
  24/24. Real recordings are harder than a synthetic triad (drums, bass, distorted
  guitars, unresolved motion), so treat the confidence, not the key, as the signal.
- Live mode feeds the same detector from a 10 s rolling window, recomputed at most
  every 0.5 s so the audio callback never pays for it.

## Overlay for OBS

`http://127.0.0.1:8765/overlay` is a transparent page for an OBS Browser Source
(1000x200 is a good size). It shows the key, the Camelot code, the BPM, the current
chord and how old the estimate is, on a translucent pill that stays readable over a
bright camera image. `?scale=1.6` enlarges it, `?simple=1` keeps only the key,
`?label=` prefixes a caption. Nothing is drawn outside the pill, so the OBS source
composites cleanly over video.

## Setlist and pre-render

Live ML separation costs a chunk of latency because the model needs the future. A
setlist sidesteps that: queue the songs before the stream, let `queue run` (or the
drawer's *แยกเสียงเพลงที่รอ*) separate them in the background while you sing the
current one, then play the ready instrumental with zero separation latency. Queue
state lives in JSON under the platform data dir, writes are atomic, and a failure
(missing demucs, an unreadable file) is recorded on the item with the real message
instead of disappearing.

Playback of a ready item replaces the capture source, so the panel is playing the file
rather than your system audio; stop it to go back.

## Playing with the song

A queued song that was rendered to four stems (vocals / drums / bass / other) gets three
controls, because that is what is actually useful while singing rather than DJing:

- **ตัดกลอง** mutes the drums (drop them for the build, bring them back on the chorus).
- **วน 4 ห้อง** loops four bars. The loop start snaps down to the bar grid computed from the
  BPM measured at render time, so the loop is musical instead of wherever you clicked.
- **เสียงร้อง** is the vocal-stem fader: 0 is a normal karaoke track, 50 is a backing vocal to
  sing harmony with, 100 is the original.

`Esc` (or the red **เงียบ** button) is the panic key: it stops playback and the live run in one
press, and the panel can start again immediately.

Stems stream from disk block by block — nothing loads a whole song into RAM — and a loop
carries the sub-block remainder so it cannot drift.

## Output limiter

Every block on its way to the device goes through a stereo-linked peak limiter at -1 dBFS.
Centre-cancel can boost panned content by up to +6 dB and stems sum, so without it a hot track
clips in OBS. It has no lookahead on purpose: zero added latency, with a hard clip as the last
resort for the single sample the attack cannot catch. The panel shows `GR -3.2 dB` while it is
working.

## Vocal range fit

`karaokelive range --from-file` estimates your comfortable range with an
autocorrelation pitch track (5th-95th percentile of voiced frames, so a couple of
wrong octaves cannot move it) and saves it as a profile. After that, `karaokelive key`
and the panel's key card report the song's range and whether it fits, with the
smallest transposition that would make it fit — the shift is computed on the song's
measured range, not guessed from the key.

## Control panel

`karaokelive ui` serves a single-page control panel on 127.0.0.1: device pickers,
the vocal-level knob with a "how much is left" readout, in/out level meters with
peak hold, the detected key with its relative key, start/stop and the latency
readout. It is bound to localhost and has no authentication, by design: it is a
remote control for your own machine, not a service.

`--demo` runs the whole UI against a simulated signal (meters and a rotating key),
which is how it is developed and tested on a headless box.

## Routing (the part everyone gets wrong)

macOS: install [BlackHole](https://github.com/ExistentialAudio/BlackHole) 2ch
(`brew install blackhole-2ch`), set the system output to it, then point
`--capture` at it. Your browser goes silent because everything now flows
through karaoke-live. To feed OBS at the same time, create a **Multi-Output
Device** in Audio MIDI Setup (BlackHole + your interface) or let OBS capture the
interface.

Windows: install [VB-CABLE](https://vb-audio.com/Cable/), set *CABLE Input* as
the default playback device, and use `--capture "CABLE Output"`. Add VB-CABLE
A+B (`--capture "CABLE-A Output"`) if you need several independent cables.

OBS: add an *Audio Output Capture* on the virtual cable you play into, or use
*Audio Input Capture* if you route the processed audio back through an
interface. Because the whole music path is delayed identically, the singer and
the backing stay in sync -- what drifts is **video**, which is never told about
the delay. Cover the video, or delay it in OBS by the same amount.

## Things worth knowing before you trust it

- **Mono downmix is the trap.** With `center` at 100%, left and right are
  anti-phase and a mono sum cancels the music almost completely. Listeners on a
  phone speaker or a single Bluetooth speaker would hear almost nothing. The
  default caps removal at 85% (`--no-mono-guard` to override); on speakers or
  headphones it is fine.
- **Centre content goes too.** Kick, snare and bass sit in the middle like the
  vocal does. That is why the crossover exists (`--bass-hz`, default 140 Hz) --
  everything below it is added back.
- **Multiplex sources are free wins.** If the vocal is only on one channel, use
  `--engine channel` and skip the DSP entirely.
- **The singer hears more ghost than the audience.** On headphones, the
  residual anti-phase vocal is obvious to you and invisible to anyone listening
  on speakers. If it bothers you, that is what `demucs` is for.
- **Licence of the weights is not the licence of the code.** Demucs code is
  MIT; its pretrained weights are not covered by that licence and part of its
  training data is undocumented. Fine for personal use, a question mark for
  redistribution. This repo never ships weights -- `demucs` downloads them.

## Testing on a Mac

`docs/MAC-TEST.md` is the checklist for the parts that cannot be verified on a headless
Linux box: devices, latency, the overlay in OBS, the mic range profile, four-stem playback,
and the two hardware failure modes worth knowing about.

## Status

- `limiter`: -12 dBFS passes untouched (GR 0.0, peak preserved), +12 dB over full scale comes
  out at exactly 1.0 with GR -12.8 dB, recovery after 200 ms of silence, and ten 512-frame
  blocks produce bit-identical output to one 5120-frame block.
- `stems`: a 333-frame blocksize lands 96,237 frames of playback at 0.5049 s, 0.00 ms off the
  expected position, and the block after a loop wrap matches the loop start sample for sample.
- `range`: detected A3-E4, C4-C5, A2-A3, E3-E4, G3-G4 and D4-D5 exactly (0 misses,
  voiced 100%) on synthesised melodies.
- `tempo`: 85-174 BPM within 1.5 BPM on click tracks, 0.05 confidence on white noise,
  no octave doubling on 8th-note hats.
- `queue`: backend and API covered, including a missing-demucs failure that marks the
  item failed with the real stderr instead of hanging.
- `keydetect`: 24/24 on synthetic major and minor progressions across all 12 tonics.
- `ui`: HTTP tests in demo mode plus real 1280x800 browser runs that verified the
  meters, the key card, the 409-while-running path, the queue drawer, the OBS overlay
  composited over a bright background, and the range row.
- `center` and `channel`: covered by tests, verified end to end on real files
  (a centred 440 Hz tone drops 33.8 dB while panned content moves < 0.1 dB).
- `demucs`: the chunking, crossfade and latency bookkeeping are tested against a
  stubbed separator (bit-exact pass-through, alignment verified by lag search);
  the model call itself has not been run on this machine -- no torch here. Treat
  it as the experimental path.
- Live mode needs a real audio device; it is developed on a headless box, so the
  routing path is the least-exercised part. `devices` first, then a 10 s
  `--seconds 10` run, is the sane first test.

## Attribution

Standing on [Poise Voice Isolator](https://github.com/chabandou/poise-voice-isolator)
(MIT) for the shape of the problem -- it plugs a model into the system audio
graph and solves the opposite direction (keep speech, drop everything else).
[Demucs](https://github.com/facebookresearch/demucs) does the heavy lifting for
the ML engine, [BlackHole](https://github.com/ExistentialAudio/BlackHole) and
[VB-CABLE](https://vb-audio.com/Cable/) do the routing.

MIT licensed.
