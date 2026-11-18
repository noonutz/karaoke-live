# The long version

Everything that is true but too long for the front page.

- [Routing](#routing-the-part-everyone-gets-wrong)
- [Things worth knowing before you trust it](#things-worth-knowing-before-you-trust-it)
- [Key detection, in detail](#key-detection-in-detail)
- [Vocal range fit](#vocal-range-fit)
- [Setlist and playback](#setlist-and-playback)
- [Engines, in detail](#engines-in-detail)
- [Verification log](#verification-log)
- [Testing on hardware](MAC-TEST.md)

## Routing (the part everyone gets wrong)

**macOS.** Install [BlackHole](https://github.com/ExistentialAudio/BlackHole) 2ch
(`brew install blackhole-2ch`), set the system output to it, then point `--capture` at it.
Your browser goes silent because everything now flows through karaoke-live. To feed OBS at
the same time, create a **Multi-Output Device** in Audio MIDI Setup (BlackHole + your
interface), or let OBS capture the interface.

**Windows.** Install [VB-CABLE](https://vb-audio.com/Cable/), set *CABLE Input* as the default
playback device, and use `--capture "CABLE Output"`. Add VB-CABLE A+B
(`--capture "CABLE-A Output"`) if you need independent cables.

**Linux.** PipeWire or PulseAudio; `karaokelive devices` lists the sinks.

**OBS.** Add an *Audio Output Capture* on the cable you play into, or *Audio Input Capture*
if you route the processed audio back through an interface. Because the whole music path is
delayed identically, the singer and the backing stay in sync — what drifts is **video**, which
is never told about the delay. Cover the video, or delay it in OBS by the same amount.

## Things worth knowing before you trust it

- **Mono downmix is the trap.** With `center` at 100%, left and right are anti-phase and a mono
  sum cancels the music almost completely. Listeners on a phone speaker or a single Bluetooth
  speaker would hear almost nothing. The default caps removal at 85%
  (`--no-mono-guard` to override); on speakers or headphones it is fine.
- **Centre content goes too.** Kick, snare and bass sit in the middle like the vocal does. That
  is why the crossover exists (`--bass-hz`, default 140 Hz) — everything below it is added back.
- **Multiplex sources are free wins.** If the vocal is only on one channel, use
  `--engine channel` and skip the DSP entirely.
- **The singer hears more ghost than the audience.** On headphones the residual anti-phase
  vocal is obvious to you and invisible to anyone listening on speakers. If it bothers you,
  that is what `demucs` is for.
- **A one-pole crossover is not enough.** A first-order lowpass at 140 Hz only rejects a 440 Hz
  vocal by ~10 dB, which is plainly audible as a ghost. The engine cascades two RBJ biquads
  (4th-order Butterworth) to put it ~40 dB down.
- **Licence of the weights is not the licence of the code.** Demucs' code is MIT; its
  pretrained weights are not covered by that licence and part of its training data is
  undocumented. Fine for personal use, a question mark for redistribution. This repo never
  ships weights — `demucs` downloads them.

## Key detection, in detail

`karaokelive key` reports the tonic, the mode, a confidence, the Camelot code and the 12-bin
chroma. Method: chromagram (Hann STFT 4096/2048, 55 Hz–4 kHz, mapped to pitch classes with
`midi = 69 + 12*log2(f/440)`) correlated against the Krumhansl-Schmuckler major and minor
profiles, rotated over all 24 keys.

- Confidence blends the winning correlation with its margin over the runner-up, so a genuinely
  ambiguous track (a bare fifth with no third, a single held chord) reads low instead of
  guessing loudly. Below 50% the CLI says so in words.
- The runners-up line shows **correlation scores**, not confidence — a different scale,
  labelled as such so `0.70` next to `48%` cannot be misread as a winner.
- Verified 24/24 on synthesised progressions in all twelve major and all twelve minor keys.
  Real recordings are harder than a synthetic triad (drums, bass, distorted guitars,
  unresolved motion), so treat the confidence, not the key, as the signal.
- Live mode feeds the same detector from a 10 s rolling window, recomputed at most every 0.5 s
  so the audio callback never pays for it.

Tempo uses the same shape: positive spectral flux over a 1024/256 STFT, autocorrelated over
lags for 60–200 BPM, folded into 70–180 so an 8th-note hi-hat cannot double the answer.

## Vocal range fit

`karaokelive range --from-file` estimates your comfortable range with an autocorrelation pitch
track (2048/512 frames, 70–1200 Hz, parabolic peak refinement) and takes the 5th–95th
percentile of voiced frames, so a couple of wrong octaves cannot move it.

```bash
karaokelive range --record 15          # sing your lowest and highest comfortable notes
karaokelive range --set "A2:C5"        # or type it
karaokelive range --show
```

After that, `karaokelive key` and the panel's key card report the song's measured range and
whether it fits, plus the smallest transposition that would make it fit. The shift is computed
on the song's range, not guessed from the key.

## Setlist and playback

Live ML separation costs a chunk of latency because the model needs the future. A setlist
sidesteps that: queue the songs before the stream, let `queue run` (or the drawer's
*แยกเสียงเพลงที่รอ*) separate them in the background while you sing the current one, then play
the ready stems with zero separation latency.

- Queue state is JSON under the platform data dir (`$XDG_DATA_HOME`, `~/Library/Application
  Support`, `%APPDATA%`), written atomically, one lock around mutations.
- A failure (missing demucs, an unreadable file, a timeout) is recorded on the item with the
  real message instead of disappearing. A corrupt queue file is renamed to
  `<name>.corrupt-<timestamp>` rather than silently overwritten.
- Playback of a ready item replaces the capture source, so the panel is playing the file
  rather than your system audio; stop it to go back.
- Stems stream from disk block by block with the stdlib `wave` module — nothing loads a whole
  song into RAM — and a loop carries the sub-block remainder so it cannot drift.

## Engines, in detail

**`center`** — `out = gain * (mix − k * highpassed(mid))`. Subtracting only the high-passed
mid keeps the low end, and `gain = 1/(1 − k/2)` compensates the level the subtraction removes
from panned content, so turning the knob up does not make the backing quieter. No state beyond
one biquad, so it runs inside the audio callback at zero latency.

**`channel`** — take one channel and duplicate it. No phase tricks; for multiplex sources.

**`demucs`** — each chunk is `history + hop + lookahead` and only the middle hop is emitted;
consecutive hops are summed with complementary linear windows that add to exactly 1. A
non-causal model cannot emit anything before a whole chunk has arrived, so the stream latency
equals the chunk length — that is the number the performer feels, and why shorter chunks trade
quality for latency.

## Verification log

| what | how | result |
|:--|:--|:--|
| Centre removal | real file through the CLI | centred tone −33.8 dB, panned content < 0.1 dB |
| Block chaining | 512-frame blocks vs one shot | < 1e-6 |
| Key | 24 synthesised progressions | 24/24 |
| Tempo | 9 click tracks, 85–174 BPM | within 1.5 BPM; noise 0.05 confidence |
| Range | 6 melodies | exact, 0 misses |
| Stem loop | 333-frame blocksize | 0.00 ms drift over 96,237 frames |
| Limiter | +12 dB over full scale | exactly 1.0; 10×512 == 1×5120 bit-identical |
| Chunked ML plumbing | stub separator, lag search | best lag == `latency_frames`, error 3e-8 |

**Not verified yet**, because the project was built on a headless Linux box: real audio
devices, the demucs weights, and the microphone. [MAC-TEST.md](MAC-TEST.md) is the checklist
for the machine that will prove those.
