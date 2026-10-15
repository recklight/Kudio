# Rooms and channels

A recording reaches a file through three kinds of damage, and each is a
different kind of arithmetic. Noise is additive: something else reaches the
microphone alongside the speech, $y + n$. A room is convolutive: it adds
nothing, but smears every sound across the next few hundred milliseconds, for
as long as the room rings, $y \ast h$. The third is the link the recording
travelled over, which kudio's functions call the channel. It is neither: a
codec quantises, a converter throws bits away, a network drops packets, and
none of that can be written as a sum or a convolution.

This page is about rooms and links; mixing in noise has its own page,
[Building a noisy dataset](synthesis.md). The examples share one setup, three
seconds of a stand-in for speech with pauses around two phrases. For a
recording of your own, load it with [`file_load`][kudio.core.io.file_load]
instead, which mixes down to mono unless you pass `mono=False`. Everything here
takes mono audio and refuses a `(frames, channels)` array with
[`FeatureError`][kudio.exceptions.FeatureError]; read as one interleaved
signal, a two-channel copy of a 0.6 s response would measure T30 1.20 s with a
fit of 1.000. Put each channel through on its own.

```python
import numpy as np
import kudio

sr = 16000
t = np.arange(3 * sr) / sr
# a 140 Hz voice with a 4 Hz syllable rhythm, silent between the phrases
voice = sum(np.sin(2 * np.pi * 140 * k * t) / k for k in range(1, 12))
y = 0.1 * voice * (0.5 + 0.5 * np.sin(2 * np.pi * 4 * t))
y[(t < 0.5) | ((t > 1.4) & (t < 1.9)) | (t > 2.6)] = 0.0
y += 0.003 * np.random.default_rng(0).standard_normal(t.size)   # a faint floor
y = y.astype(np.float32)
```

## Why the kind matters

A tool that undoes one kind leaves the others alone.
[`spectral_enhance`][kudio.enhance.spectral.spectral_enhance] estimates a noise
spectrum and takes it out, and a room adds no spectrum to take out. The same
denoiser on the same clip, scored with [`si_sdr`][kudio.core.evaluator.si_sdr],
once with white noise at 5 dB SNR and once in a 0.6 s room:

```python
noise = np.random.default_rng(1).standard_normal(len(y)).astype(np.float32)
noisy = kudio.add_noise_snr(y, noise, snr_db=5, seed=0)
wet = kudio.apply_rir(y, kudio.rir(sr, rt60=0.6, drr_db=6.0, seed=0))

for name, degraded in (("noise", noisy), ("room", wet)):
    before = kudio.si_sdr(y, degraded)
    after = kudio.si_sdr(y, kudio.spectral_enhance(degraded, sr))
    print(f"{name:5} {before:4.1f} dB -> {after:4.1f} dB SI-SDR")
```

```text
noise  4.9 dB -> 13.1 dB SI-SDR
room   6.5 dB ->  6.5 dB SI-SDR
```

Taking reverberation out is a different problem, and kudio does not attempt
it; the room functions make reverberation you control and measure it when you
have an impulse response. A link is not undone either: a lost packet is gone,
and all `dropouts` decides is what fills the hole. What can be taken out is on
[Denoising](enhancement.md).

The three always happen in the same order:

```text
talker -> room -> + noise -> link -> file
```

The room acts on the talker, the noise reaches the microphone beside what the
room left, and the link carries whatever the microphone picked up, noise
included. [Building a noisy dataset](synthesis.md) applies the chain to a
corpus and says what its SNR is measured against.

## Rooms

### Making one

[`rir`][kudio.core.room.rir] builds an impulse response out of exponentially
decaying noise, the standard model of a diffuse reverberant tail. `rt60` is
the time its envelope takes to fall 60 dB. `drr_db` is how far the direct path
sits above the reverberant energy, which is distance in a readable unit: +10 dB
or more is a close microphone, around 0 is a metre or two away, and the far end
of a hall is negative. The length defaults to 1.5 × `rt60`, enough for the
35 dB of decay T30 needs, and the same `seed` builds the same room.

```python
ir = kudio.rir(sr, rt60=0.6, drr_db=6.0, seed=0)
print(len(ir) / sr)
print(kudio.rt60(ir, sr))
```

```text
0.9
T30 0.60 s (EDT 0.55 s)  ·  C50 +11.7 dB  ·  DRR +6.0 dB  ·  fit 1.000
```

`drr_db` is solved for with the 2.5 ms window the measurement uses, so in rooms
of 0.3 s and longer every value from −6 to +20 dB comes back to the hundredth
of a decibel; a 0.2 s room can miss the lowest values by half a decibel, and
shorter rooms by more. `rt60` sets the envelope's slope, and with `seed=0` and
the default `drr_db` a 0.2 s room measures 0.204 s and a 2.0 s room 1.994 s.
A `drr_db` below what the tail already puts inside the window is refused with
`FeatureError` rather than built differently. The shorter the room, the higher
that floor: −9.2 dB at 0.2 s, −13.3 dB at 0.5 s and −19.4 dB at 2.0 s, with
`seed=0`.

The room is a statistical one, with no early reflections and no walls, so it
will not stand in for a measured response when the shape of the early energy
is what you are studying. [`reverb`][kudio.effects.augment.reverb] is cruder
still, the direct sound and seven echoes, 50 ms apart by default, with no
reverberation time to ask for ([Editing and effects](editing.md) has it). When
a number has to go on the dataset, use `rir`.

### Putting a clip in it

[`apply_rir`][kudio.core.room.apply_rir] convolves, and by default does two
more things. It cuts the result back to the input's length (`trim=True`),
since a clip that grows each time it is processed stops lining up with
everything measured against it. And it puts the peak back where the input's
was (`normalise=True`), since a room with a loud direct path has a gain, and
an augmentation that also changes the level changes two things at once.

A synthetic room's direct path is its first sample. A recorded response starts
with the time the sound took to reach the microphone, which `apply_rir` keeps
as a delay, and `apply_rir` takes no sample rate. So load a recorded response
at the clip's rate and cut it at the direct sound:

```python
measured, _ = kudio.file_load("measured_ir.wav", sr=sr)   # at the clip's rate
measured = measured[np.argmax(np.abs(measured)):]         # no time of flight
wet = kudio.apply_rir(y, measured)
```

### Measuring one

[`rt60`][kudio.core.room.rt60] returns a
[`Reverberation`][kudio.core.room.Reverberation]. T20, T30 and EDT are read
off one backward-integrated decay curve,
[`schroeder_curve`][kudio.core.room.schroeder_curve]: the time it takes to fall
from −5 to −25 dB, from −5 to −35 dB and from 0 to −10 dB, carried on to 60 dB.
Those are the evaluation ranges of ISO 3382-1, and T30 is the figure usually
quoted as the RT60. C50, C80 and DRR are energy ratios taken straight from the
response: the first 50 or 80 ms after the peak against everything later, and
±2.5 ms around it against the rest. `fit` is the R² of the line fitted for T30.

![Schroeder decay of a synthetic room with its T30 fit, and the same room over a noise floor](../images/degradation-schroeder-light.webp#only-light)
![Schroeder decay of a synthetic room with its T30 fit, and the same room over a noise floor](../images/degradation-schroeder-dark.webp#only-dark)

On the left, a 0.6 s room: from −5 to −35 dB the decay is a straight line, and
the time that line takes to fall 60 dB is T30. The right panel belongs to the
first warning below.

Read `fit` before the rest. Backward integration makes any signal decay, so
`rt60` returns figures for noisy speech or a truncated response as readily as
for a room:

```python
print(kudio.rt60(noisy, sr))
cut = kudio.rir(sr, rt60=1.0, seed=0)[:int(0.05 * sr)]   # 1 s room, 50 ms of it
room = kudio.rt60(cut, sr)
print(f"T30 {room.t30:.2f} s, fit {room.fit:.3f}, reliable: {room.reliable()}")
```

```text
T30 2.62 s (EDT 18.47 s)  ·  C50 -11.2 dB  ·  DRR -20.3 dB  ·  fit 0.924  — but EDT and T30 disagree, so this may not be an impulse response
T30 0.19 s, fit 0.802, reliable: False
```

[`reliable()`][kudio.core.room.Reverberation.reliable] asks two questions. The
first is whether the decay was straight, a `fit` of at least 0.9; the cut room
fails it. The noisy speech passes it, which is why there is a second: whether
EDT and T30 agree. Away from the source the first 10 dB and the −5 to −35 dB
span describe the same tail, so EDT should be within a factor of
[`MAX_EDT_RATIO`][kudio.core.room.MAX_EDT_RATIO] (2) of T30. That bound is a
sanity check, not a figure from the standard. Over forty seeds of the 0.6 s,
+6 dB room EDT/T30 stays between 0.87 and 1.16; the noisy speech puts it at 7.

A recorded response does not need cutting before it is measured. `rt60`
starts at the onset, the first sample within 20 dB of the peak, so the time of
flight is not counted as part of the decay: with 5 ms of silence in front of
it, the 0.6 s room still measures EDT 0.55 s and passes `reliable()`.

`rt60` also measures the whole band it is given, while room-acoustics reports
quote these figures per octave. For a band figure, filter with
`zero_phase=False` and cut at the peak again, since the filter delays the
direct sound: `kudio.bandpass(ir, sr, low=355, high=710, zero_phase=False)`,
the 500 Hz octave, gives T30 0.60 s and EDT 0.49 s for the room above, where
the default zero-phase filter left EDT at 0.01 s. Read only the decay times
off the result; this +6 dB room measures a DRR of +0.1 dB in the band.

!!! warning "A noise floor is not taken out, and `reliable()` cannot see one"

    The right panel is the same 0.6 s room over white noise 50 dB below the
    direct sound. Its tail starts only about 22 dB above the floor, and the
    raw response (the grey trace) stops falling at about −45 dB, so the decay
    curve falls too slowly and then drops at the cut. Cut at 0.45 s, T30 comes
    back as 1.23 s with a fit of 0.962. No cut fixes it: from 0.15 to 0.45 s
    the answer runs from 0.39 to 1.23 s, 0.54 s where the tail meets the floor,
    and `reliable()` says True every time. A decay curve always falls past
    −35 dB at a cut, so check the response itself: T30 means something only
    when the raw response is still clear of the noise where the decay curve
    crosses −35 dB. If it is not, the recording holds no T30 to quote, and the
    answer is to measure again with more level.

!!! warning "`reliable()` cannot judge a microphone just under +10 dB DRR"

    Close to the source the direct sound fills the first 10 dB by itself, and
    EDT drops: `drr_db=9` gives 0.35 s against a T30 of 0.60 s, and `drr_db=9.6`
    gives 0.04 s, which fails the EDT test on a decay that is perfectly
    straight. From about +10 dB the decay curve falls past −10 dB in its first
    sample, so there is no EDT at all. It comes back NaN, the summary says
    "no EDT: the direct sound alone falls 10 dB", and `reliable()` judges on the
    fit alone: `rir(sr, rt60=0.6, drr_db=12, seed=0)` passes with T30 0.60 s and
    a fit of 1.000. Just under +10 dB, read `fit` and `t30` and set the verdict
    aside.

## Channels

### A phone line

[`telephone`][kudio.effects.channel.telephone] does what a narrowband line
does, in the order it does it: band-limit to 300–3400 Hz
([`TELEPHONE_BAND`][kudio.effects.channel.TELEPHONE_BAND]) at the original
rate, resample to 8 kHz, compand through G.711 mu-law, and come back. The
band-pass comes first because the resampler's anti-alias filter is not the
line's passband. The result keeps the rate and the length it went in with, so
it sits beside its source in a dataset.

```python
phone = kudio.telephone(y, sr)
print(len(phone) == len(y))
for problem in kudio.audio_report(phone, sr).problems():
    print(problem)
```

```text
True
content stops at 3766 Hz although the file claims 16000 Hz — it was very likely upsampled from 8000 Hz, and the extra rate is empty
```

[`audio_report`][kudio.core.report.audio_report] sees through it: the content
stops below 4 kHz, so it names 8 kHz as the rate the file really had.
[Inspecting a recording](inspection.md) explains how it decides.

### Companding and bit depth

[`mu_law`][kudio.effects.channel.mu_law] and
[`a_law`][kudio.effects.channel.a_law] push the signal through a G.711
companding curve, quantise it to 8-bit code words and expand it back. The
steps are logarithmic, so the quantisation noise follows the signal down
instead of sitting at a fixed floor. [`bit_depth`][kudio.effects.channel.bit_depth]
spends the same 8 bits linearly, and [`snr`][kudio.core.evaluator.snr] shows
what companding buys:

```python
full = y / np.max(np.abs(y))
for level in (0, -6, -12, -18, -24):
    x = (full * 10 ** (level / 20)).astype(np.float32)
    print(f"{level:3} dBFS   mu-law {kudio.snr(x, kudio.mu_law(x)):.1f}   "
          f"A-law {kudio.snr(x, kudio.a_law(x)):.1f}   "
          f"8-bit linear {kudio.snr(x, kudio.bit_depth(x, 8)):.1f}")
```

```text
  0 dBFS   mu-law 37.9   A-law 38.1   8-bit linear 40.2
 -6 dBFS   mu-law 37.9   A-law 38.1   8-bit linear 34.3
-12 dBFS   mu-law 37.7   A-law 38.0   8-bit linear 28.2
-18 dBFS   mu-law 37.4   A-law 37.7   8-bit linear 22.9
-24 dBFS   mu-law 36.6   A-law 36.6   8-bit linear 18.5
```

Over 24 dB of level, mu-law gives up 1.3 dB of SNR and linear 8-bit gives up
21.7 dB, close to the 24 dB that 6 dB per halving predicts.

These are the companding curves themselves, not the standard's
piecewise-linear segments. Against Python's `audioop`, which is G.711 itself
(and was removed in Python 3.13, so it cannot be a dependency), kudio's
round-trip SNR comes out 0.25 to 0.83 dB higher over the levels in the table,
and single samples differ by a few hundredths of full scale at most: close
enough for augmentation, and not a bit-exact decoder.

A full-scale sine through `bit_depth` comes back at $6.02\,b + 1.76$ dB for
$b$ bits: 49.9 dB at 8 bits and 98.1 dB at 16, on a 997 Hz tone.
`dither=True` adds triangular (TPDF) dither two steps wide, peak to peak,
before rounding, and costs 4.7 dB on that tone at 8 bits: the dither's own
power is twice the rounding error's, so the noise triples, which is 4.8 dB.
What that buys is an error that no longer depends on the signal. A sine a
quarter of a step high comes back at its full level on average (1.00 over 20
seeds), where without dither nothing of it comes back. Pass `seed` with it, or
every call dithers differently.

### Packet loss

[`dropouts`][kudio.effects.channel.dropouts] cuts the clip into packets of
`packet_ms` (default 20 ms, [`PACKET_MS`][kudio.effects.channel.PACKET_MS], the
packet G.711 uses over RTP) and loses some of them. The count is fixed at
`round(loss * packets)` rather than drawn packet by packet, and it is rounded
per file: at 5%, a 1 s clip (50 packets) loses 2, which is 4%, and a clip of
0.2 s (10 packets) or less loses nothing. `return_spans=True` also returns
where each lost packet was, in seconds:

```python
lossy, gaps = kudio.dropouts(y, sr, loss=0.05, seed=0, return_spans=True)
print(len(gaps), [f"{a:.2f}-{b:.2f} s" for a, b in gaps[:3]])
```

```text
8 ['0.04-0.06 s', '0.12-0.14 s', '0.22-0.24 s']
```

Three seconds is 150 packets, and 5% of 150 rounds to 8, or 5.3%.
`conceal='silence'`, the default, leaves the holes empty. `conceal='hold'`
repeats the packet before each one, the crudest concealment a decoder does.
Across a run of lost packets it repeats the same 20 ms over and over, and a
packet lost at the very start stays silent because nothing came before it.

!!! warning "A waveform metric cannot choose a concealment"

    Whether `hold` scores better than `silence` depends on the talker's pitch
    against the packet length. When a whole number of pitch periods fits in a
    packet, the repeat lands in phase and is exact. A tenth of a period off,
    it scores about the same as the hole; a third to a half of a period off,
    it lands out of phase and scores 4 to 5 dB worse. `kudio.snr` with 10% of
    packets lost from four seconds of a tone with nine harmonics (seed 2):

    | Pitch | Cycles per 20 ms | `silence` | `hold` |
    |---|---|---|---|
    | 100 Hz | 2.00 | 10.0 dB | exact |
    | 105 Hz | 2.10 | 10.0 dB | 9.8 dB |
    | 133 Hz | 2.66 | 10.0 dB | 5.6 dB |
    | 150 Hz | 3.00 | 10.0 dB | exact |
    | 175 Hz | 3.50 | 10.0 dB | 5.1 dB |

    Choosing between the two on SNR or SI-SDR measures that accident, not the
    concealment. [Measuring quality](metrics.md) has what each metric does
    see.

#### Losses in runs

`burst=None`, the default, loses packets independently. Real links lose them
in runs (a fade, a queue overflowing, a handover), and a run is harder to
conceal than the same number of scattered packets: after its first packet
there is no recent speech left to repeat, so `hold` repeats one packet across
the whole run. `burst=` sets the mean run length in packets:

![Thirty seconds of 5% packet loss, independent and in runs of 6](../images/degradation-loss-light.webp#only-light)
![Thirty seconds of 5% packet loss, independent and in runs of 6](../images/degradation-loss-dark.webp#only-dark)

The run lengths are geometric, as in the bad state of Gilbert's two-state
model, and runs stay at least one surviving packet apart, so two never join
into one and the runs you count in the output are the runs that were drawn.

`burst=1` is not the default under another name. Independent draws land side
by side now and then, so the default already averages `1 / (1 - loss)` packets
a run, which [`independent_burst`][kudio.effects.channel.independent_burst]
returns: 1.05 at 5% loss, 1.25 at 20%. `burst=1` asks for runs of exactly one,
which is *less* clustered than leaving it alone. The setup clip tiled to four
minutes (`np.tile(y, 80)`), 12000 packets with 600 of them lost, `seed=0`:

| `burst` | Runs | Mean run | Longest run |
|---|---|---|---|
| `None` | 567 | 1.06 | 3 |
| `1` | 600 | 1.00 | 1 |
| `4` | 138 | 4.35 | 22 |
| `6` | 92 | 6.52 | 34 |

Each row is one draw, so the mean run lands near `burst`, not on it; over 50
seeds it comes to 5.9 for `burst=6`.

### The whole link at once

[`apply_channel`][kudio.effects.channel.apply_channel] runs a link in the
order a link does it: band and rate, codec, converter, network. Packets are
lost from the encoded signal, so the loss goes last. The link is a spec: a
preset name, a dict of any of the
[`CHANNEL_DEFAULTS`][kudio.effects.channel.CHANNEL_DEFAULTS] keys, or a preset
with overrides under a `'preset'` key.
[`CHANNEL_PRESETS`][kudio.effects.channel.CHANNEL_PRESETS] holds `telephone`
and `telephone_a` (the band and the 8 kHz rate, with mu-law or A-law), `mu_law`
and `a_law` (the codec alone, at full band), `voip` (mu-law at full band, 5%
loss in runs of 4, `hold`) and `none`.
[`channel_spec`][kudio.effects.channel.channel_spec] turns any of these forms
into one dict, and [`channel_tag`][kudio.effects.channel.channel_tag] into the
label a manifest carries: `mu_law+loss10%x4-hold` for
`{"preset": "voip", "loss": 0.1}`.

The order shows up in the samples. Lose the packets first and the band-pass
filter and the two resamplings inside `telephone` spread the speech on either
side into every hole:

```python
right = kudio.apply_channel(y, sr, {"preset": "telephone", "loss": 0.1}, seed=0)
wrong = kudio.telephone(kudio.dropouts(y, sr, loss=0.1, seed=0), sr)
print(np.count_nonzero(right == 0.0), np.count_nonzero(wrong == 0.0))
```

```text
4800 0
```

In the right order the 15 lost packets are 4800 samples of exact silence, 320
to a packet; in the wrong order not one sample is zero.

`channel_spec` refuses an unknown setting, codec or concealment, a loss
outside 0 to 1, fewer than 2 bits, a `burst` under 1 and a `packet_ms` of zero
or less. `apply_channel` and `Synthesizer` call it before any audio goes
through, so a `Synthesizer` given a bad link fails when it is built, not on its
first file. A spec's `band` is on or off; for other edges, call
`telephone(..., band=)` yourself.

The whole chain for one clip, room, noise and link in that order:

```python
wet = kudio.apply_rir(y, kudio.rir(sr, rt60=0.5, drr_db=3.0, seed=0))
mix = kudio.add_noise_snr(wet, noise, snr_db=10, seed=0)
out = kudio.apply_channel(mix, sr, "voip", seed=0)
```

A [`Synthesizer`][kudio.core.synth.Synthesizer] takes the same spec as
`channel=` and applies it to every finished mixture; [Building a noisy
dataset](synthesis.md) covers the per-file seeding, the manifest tag and why
the link is not an axis.

## From the command line

`kudio room` measures a response and exits 1 when `reliable()` is False.
`--make` builds a room, and `--apply` puts a file through a synthetic one
built from `--rt60`, `--drr` and `--seed`; a recorded response goes through
`apply_rir` in Python. `kudio channel` degrades a file and reports what it
did, including the runs it lost.

```bash
kudio room ir.wav
kudio room --make room.wav --rt60 0.6 --drr 6 --seed 0
kudio room speech.wav --apply wet.wav --rt60 0.6 --seed 0
kudio channel in.wav out.wav --telephone
kudio channel in.wav out.wav --loss 0.05 --burst 6 --conceal hold --seed 0
```

The flags are on [Command line](../cli.md), and every parameter of every
function here is in the [reference](../reference/degradation.md).
