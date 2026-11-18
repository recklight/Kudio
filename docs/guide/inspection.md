# Inspecting a recording

Every score on [Measuring quality](metrics.md) compares a processed clip with a
clean reference. A synthetic dataset has both; a recording from the field has
only itself, and that is usually the one you need to judge. Score a denoised
field recording against the noisy original and you measure how much the
denoiser changed it, not whether it got better. So first, the questions a file
can answer on its own:

| Question | Reach for | Needs |
|---|---|---|
| What is this file? | [`audio_info`][kudio.core.io.audio_info] | the core install |
| Is it technically sound? | [`audio_report`][kudio.core.report.audio_report] | the core install |
| How loud does it sound, and where did the level move? | [`loudness`][kudio.core.loudness.loudness] and the functions beside it | the core install |
| How reverberant is it? | [`srmr`][kudio.core.srmr.srmr] | the core install |
| How far does the speech sit above its noise? | [`wada_snr`][kudio.core.wada.wada_snr] | the core install |
| What would a listener say about it? | [`dnsmos`][kudio.core.dnsmos.dnsmos] | `kudio[dnsmos]` and model weights kudio does not ship |

The examples share one synthetic take, so they run anywhere: a 140 Hz voice
with harmonics up to 7.6 kHz, switched on and off to leave pauses, over a hiss
at -54 dBFS. The last line asks the file what it is.

```python
import numpy as np
import kudio

sr = 16000
rng = np.random.default_rng(0)
t = np.arange(20 * sr) / sr
voice = sum(np.cos(2 * np.pi * 140 * k * t) / k for k in range(1, 55))
words = np.sin(2 * np.pi * 0.7 * t) > -0.3          # talk, pause, talk, ...
take = 0.05 * voice * words + 0.002 * rng.standard_normal(t.size)
kudio.save_wave("take.wav", take, sr)
print(kudio.audio_info("take.wav"))
```

```text
take.wav: 20.000s, 16000 Hz, 1ch, PCM_16, peak 0.2343
```

[`audio_info`][kudio.core.io.audio_info] reads the rate, channels, length and
encoding from the header and scans the samples for the peak, a block at a
time, without loading the file. [Audio files](audio-files.md) has the rest of
it, including how to skip the scan on a large corpus.

## A health check with no reference

[`audio_report`][kudio.core.report.audio_report] measures what can be measured
without a clean twin and returns an
[`AudioReport`][kudio.core.report.AudioReport], whose
[`problems()`][kudio.core.report.AudioReport.problems] puts the findings into
plain words. An empty list is good news.

```python
y, sr = kudio.file_load("take.wav")
report = kudio.audio_report(y, sr)
print(report)
print(report.problems())
```

```text
20.00s @ 16000 Hz · peak -12.6 dBFS · -29.2 LUFS · SNR~28 dB · bandwidth 8000 Hz
[]
```

It takes mono audio and raises a [`FeatureError`][kudio.exceptions.FeatureError]
on a multi-channel array. The default `file_load` averages the channels, which
can hide a fault in one of them, so check a stereo file a channel at a time:
load it with `mono=False` and report `y[:, 0]` and `y[:, 1]`. Frames are 30 ms
at a 15 ms hop, and the thresholds are fixed, not parameters:

| Field | Measures | Flagged when |
|---|---|---|
| `peak_dbfs` | the loudest sample, in dBFS | over 0, over -0.5 or under -30 |
| `clipped_samples`, `clipped_ratio` | samples in a flat run at full scale | any |
| `dc_offset` | the mean sample value | size over 0.01 |
| `silence_ratio` | share of frames more than 40 dB below the 95th-percentile frame | over 0.6 |
| `noise_floor_dbfs` | level of the 10th-percentile frame | never |
| `estimated_snr_db` | 95th-percentile frame level minus the 10th | under 10 dB |
| `bandwidth_hz`, `band_limited` | highest frequency still carrying content | under 0.9 of Nyquist |
| `lufs`, `rms_dbfs` | integrated loudness (`nan` under 0.4 s) and plain RMS | never |
| `non_finite_samples` | samples that are NaN or inf | any |

The three peak limits mean past full scale, one gain stage from clipping, and
too little of the bit depth in use. A peak of `-inf` (digital silence) and any
non-finite sample are each reported on their own, with nothing else beside them.

### Clipping

A loud sample is not a clipped one. Float audio amplified past 1.0 still traces
its waveform; the damage happens when it is written to an integer format.
Clipping is information already destroyed, and it looks like a plateau, so the
report counts a sample as clipped only inside a run of at least three samples
at 0.999 of full scale or more that vary by no more than 1e-4. A sample or two
touching 1.0 is not counted; it shows up as a peak right against the ceiling.

```python
hot = 8 * take                              # float: louder, but intact
for clip in (hot, np.clip(hot, -1.0, 1.0)):
    for problem in kudio.audio_report(clip, sr).problems():
        print(problem)
```

```text
peaks at +5.5 dBFS, past full scale — it will clip the moment it is written to an integer format
3321 samples clipped (1.04% of the clip) — the recording was already too hot; no processing gets that back
```

The run rule has two blind spots. A peaky voice pushed past full scale clips a
sample or two per pitch period: the take normalised to -14 LUFS and written as
`PCM_16` loses 2618 samples that way, and the report on the file says only
that it peaks at 0.0 dBFS, right against the ceiling. A clipped recording turned down afterwards, even by
1 dB, has its plateaus below 0.999 and is not reported at all. Run the report
before writing, while the samples past 1.0 are still there to see; the second
case needs a look at the waveform.

### Upsampled and band-limited files

The fault that does the quietest damage to a dataset is a 16 kHz file holding
8 kHz audio: resampled up at some point, nothing above 4 kHz, and treated as
wideband by everything downstream. A recording made at its stated rate usually
has a noise floor spread over the whole band, so its spectrum reaches Nyquist.
Upsampled audio has nothing above the old Nyquist at all.

So the report averages the spectrum over 1024-sample Hann frames and takes the
highest frequency still within 50 dB of the strongest one. Below 0.9 of Nyquist
the clip counts as band-limited. When twice that frequency is within 15 % of a
common rate (8, 11.025, 16, 22.05, 32, 44.1 or 48 kHz), the finding names it
as the rate the file was upsampled from; otherwise it names the frequency
itself, "band-limited to about 3219 Hz".

```python
narrow = kudio.resample(kudio.resample(take, sr, 8000), 8000, sr)
for problem in kudio.audio_report(narrow, sr).problems():
    print(problem)
```

```text
content stops at 3797 Hz although the file claims 16000 Hz — it was very likely upsampled from 8000 Hz, and the extra rate is empty
```

![Average spectrum of the take and of its upsampled copy, with the 50 dB line where the report reads the bandwidth](../images/inspection-bandwidth-light.webp#only-light)
![Average spectrum of the take and of its upsampled copy, with the 50 dB line where the report reads the bandwidth](../images/inspection-bandwidth-dark.webp#only-dark)

[`telephone`][kudio.effects.channel.telephone] on
[Rooms and channels](degradation.md) makes such a file on purpose, and the
report reads it the same way. The check makes two assumptions. The content has
to be broadband, as speech is: a pure 220 Hz tone is reported as band-limited
and as noisy, which is correct arithmetic and no use to anybody. And the noise
floor has to sit within 50 dB of the strongest part of the spectrum. A clean
recording with a quieter hiss and a quiet top octave stops short of 0.9 of
Nyquist with nothing resampled: the take's voice with its harmonics falling as
`1 / k**1.5` instead of `1 / k`, over hiss at -66 dBFS, is reported as
stopping at 6578 Hz. On a quiet, clean file, take the finding as a reason to
look at the spectrum, not as proof.

### When one fault changes another's reading

The measurements all come from the same samples, so one fault could move a
number that belongs to another. A DC offset does not: the frame levels are
measured with the clip's mean taken out, so an offset is reported as an offset
and the floor is read from what is left. Clipping does. It spreads energy up
the spectrum, past the 50 dB line, so a clipped file looks wideband whatever
its history:

```python
for label, clip in [("as is", narrow),
                    ("+0.03 DC", narrow + 0.03),
                    ("clipped", np.clip(8 * narrow, -1.0, 1.0))]:
    r = kudio.audio_report(clip, sr)
    print(f"{label:10}  floor {r.noise_floor_dbfs:6.1f} dBFS  "
          f"SNR~{r.estimated_snr_db:3.0f} dB  "
          f"bandwidth {r.bandwidth_hz:4.0f} Hz  band-limited {r.band_limited}")
```

```text
as is       floor  -57.5 dBFS  SNR~ 31 dB  bandwidth 3797 Hz  band-limited True
+0.03 DC    floor  -57.5 dBFS  SNR~ 31 dB  bandwidth 3797 Hz  band-limited True
clipped     floor  -37.9 dBFS  SNR~ 28 dB  bandwidth 7984 Hz  band-limited False
```

With the offset in, `problems()` adds the DC finding and nothing else; take the
offset out with [`remove_dc`][kudio.effects.edit.remove_dc] anyway, since it
wastes headroom. The clipped copy loses the finding that it was upsampled from
8 kHz, so read nothing into the bandwidth of a clipped file.

### Corrupt samples

A clip with NaN or inf in it is measured, not refused: checking a corpus is
exactly when you want to hear about every broken file, not stop at the first.

```python
broken = take.copy()
broken[5000:5010] = np.nan
for problem in kudio.audio_report(broken, sr).problems():
    print(problem)
```

```text
10 samples are not finite (NaN or inf) — the clip is corrupt, not quiet, and every measurement here is meaningless. Repair it or drop it from the set
```

It comes first and alone, because the other checks cannot be relied on to
notice: a comparison against NaN is false whichever way round it is written.
Read `problems()`, not the fields; beside the count, this report still gives
an estimated SNR of 28 dB. The loudness functions check for themselves, so a
corrupt clip never measures `-inf`, the reading for silence:
[`loudness`][kudio.core.loudness.loudness] and
[`true_peak`][kudio.core.loudness.true_peak] return `nan` with a warning, a
curve from [`loudness_over_time`][kudio.core.loudness.loudness_over_time]
reads "171 of 171 window(s) unmeasurable", and `normalize_lufs` and
`match_loudness` raise `FeatureError` rather than scale by a level that is not
there. Loading, `audio_info` and `save_wave` each handle such a file their own
way ([Audio files](audio-files.md)).

### What the numbers cannot tell you

The SNR estimate needs pauses with noise in them: it reads the floor off the
10th-percentile frame, which works only when a tenth of the frames hold
nothing but noise. While the voice is on, the take's hiss sits 27 dB under it,
and the report says 28. Averaged over the whole file, pauses included, the way
a mix SNR is counted, the take is at 24.8 dB, which is why the report reads
several dB above the SNR a file was mixed at
([Getting started](../getting-started.md) has an example). Without the right
kind of pause, the quietest frames are speech:

```python
steady = 0.05 * voice + 0.002 * rng.standard_normal(t.size)   # never pauses
clean = 0.05 * voice * words                    # pauses of exact zeros, no hiss
for name, clip in [("no pauses", steady), ("no hiss", clean)]:
    print(f"{name:10} SNR~{kudio.audio_report(clip, sr).estimated_snr_db:.1f} dB")
```

```text
no pauses  SNR~0.7 dB
no hiss    SNR~inf dB
```

The first is reported as "estimated SNR around 1 dB", and the finding is wrong:
the quietest tenth of its frames are speech. Treat a low estimate on a clip
without noisy pauses as unknown rather than bad. The second clip is clean and
is read as clean: when at least a tenth of the frames are exact zeros, the
floor is digital silence, `-inf` dBFS, and nothing is listed. [Speech activity and pitch](speech.md) finds where the speech is, and
[Denoising](enhancement.md) takes a real hiss out. An empty list is only a
technical verdict, too: a clip can pass everything here and still sound poor,
which is what [DNSMOS](#dnsmos-what-a-listener-would-say) is for.

### A folder at a time

`file_load` raises [`AudioIOError`][kudio.exceptions.AudioIOError] on a file
soundfile cannot open, and `audio_report` a `FeatureError` on an empty one.
Catch both, or the sweep stops at the first broken file:

```python
from pathlib import Path

for path in sorted(Path("corpus").rglob("*.wav")):
    try:
        y, sr = kudio.file_load(path)
        problems = kudio.audio_report(y, sr).problems()
    except (kudio.AudioIOError, kudio.FeatureError) as e:
        problems = [f"not measured: {e}"]
    for problem in problems:
        print(f"{path.as_posix()}: {problem}")
```

## How loud it sounds

The peak says little about how loud a clip sounds, since one stray click sets
it ([Editing and effects](editing.md) shows the damage).
[`loudness`][kudio.core.loudness.loudness] measures what a listener hears, to
ITU-R BS.1770-4,[^bs1770] as gated integrated loudness in LUFS:

1. K-weight the signal: a high shelf worth +4 dB from about 5 kHz up, for the
   effect of the head, and a high pass about 3 dB down at 60 Hz, for rumble.
2. Take the mean square over 400 ms blocks, a new block every 100 ms.
3. Drop the blocks below -70 LUFS, then those more than 10 LU below the level
   of what is left. These are the absolute and relative gates.
4. Average the rest in the power domain.

The gates are what make the number usable. Plain RMS falls every time
somebody stops talking and the room tone takes over; gated loudness does not:

```python
padded = np.concatenate([take, 0.002 * rng.standard_normal(10 * sr)])  # room tone
for clip in (take, padded):
    rms = 10 * np.log10(np.mean(clip ** 2))
    print(f"RMS {rms:.1f} dBFS, loudness {kudio.loudness(clip, sr):.1f} LUFS")
```

```text
RMS -29.2 dBFS, loudness -29.2 LUFS
RMS -30.9 dBFS, loudness -29.2 LUFS
```

The room tone measures -51.3 LUFS on its own, far above the -70 LUFS absolute
gate, so it is the relative gate that leaves it out.

The standard publishes its filters at 48 kHz only, and the textbook equations
do not reproduce them, so kudio carries the published ones to other rates
through a prewarped bilinear transform, at most 0.07 dB off the 48 kHz response
at 16 kHz and 0.29 dB at 8 kHz. A full-scale 1 kHz sine, -3.01 LUFS in the
standard, reads -3.004 at 48 kHz, -3.05 at 16 kHz and -3.21 at 8 kHz. With
pyloudnorm installed (the `[check]` extra), kudio's tests hold it to that
independent meter within 0.1 LU at 8, 16, 44.1 and 48 kHz.

Multi-channel audio goes in channels last, `(samples, channels)`: up to five
channels, weighted 1.0, 1.0, 1.0, 1.41 and 1.41 for left, right, centre and the
two surrounds. There is no slot for an LFE channel, so a six-channel file is
refused. Load a stereo file with `mono=False` before measuring it: the default
load averages the channels, which reads 3.01 LU low when the two are the same
and 6.02 LU low when only one carries the signal. A clip shorter than one
400 ms block raises a `FeatureError` that points at `block_size=`; a smaller
block gives a number, but not a BS.1770 one.

### Normalising to a target

[`normalize_lufs`][kudio.core.loudness.normalize_lufs] applies the one gain
that brings a clip to a target, -23 LUFS by default (EBU R 128[^ebu]), and returns
silence unchanged. It does nothing else. Samples the gain pushes past full
scale come back past 1.0, not limited, because limiting them would change the
loudness you just asked for:

```python
loud = kudio.normalize_lufs(take, sr, lufs=-14.0)
print(f"{kudio.loudness(loud, sr):.1f} LUFS, peak {np.max(np.abs(loud)):.2f}")
```

```text
-14.0 LUFS, peak 1.34
```

It raises a `UserWarning` when that happens, which Python shows without any
setup.
Save such a clip with `subtype="FLOAT"` or aim lower, since as `PCM_16` it
clips in the way the report barely notices ([Clipping](#clipping)). The
warning is about samples only; the [true peak](#true-peak) between them can be
higher still. [`convert_folder`][kudio.core.io.convert_folder] with
`lufs=-23.0` normalises a whole folder ([Audio files](audio-files.md)).

### Matching two versions before an A/B

A listener comparing two versions tends to prefer the louder one, whatever its
quality. A denoiser takes some level away along with the noise, so an
unmatched comparison partly measures the level change.
[`match_loudness`][kudio.core.loudness.match_loudness] scales one clip to the
integrated loudness of another:

```python
noisy = take + 0.02 * rng.standard_normal(t.size)
cleaned = kudio.spectral_enhance(noisy, sr)
matched = kudio.match_loudness(cleaned, noisy, sr)
for name, clip in [("noisy", noisy), ("cleaned", cleaned),
                   ("matched", matched)]:
    print(f"{name:8} {kudio.loudness(clip, sr):6.1f} LUFS")
```

```text
noisy     -27.5 LUFS
cleaned   -30.0 LUFS
matched   -27.5 LUFS
```

Unmatched, the noisy version would start any listening test 2.5 LU ahead.
Clips at different rates can be matched too, with `reference_sr=`, since
loudness belongs to the sound, not the sampling.

## Loudness over time

One number for a whole recording is what you normalise against, and no help
in finding the moment somebody turned away from the microphone.
[`loudness_over_time`][kudio.core.loudness.loudness_over_time] measures over a
sliding window and returns a
[`LoudnessCurve`][kudio.core.loudness.LoudnessCurve]:

```python
drift = take.copy()
drift[8 * sr:13 * sr] *= 10 ** (-12 / 20)          # five seconds facing away

curve = kudio.loudness_over_time(drift, sr)
print(curve)
print(curve.spans_below(curve.max - 6, min_seconds=1.0))
```

```text
-29.7 LUFS at 16.30s, -42.1 LUFS at 11.00s (12.4 LU apart)
[(9.1, 11.9)]
```

![Momentary and short-term loudness of the take with its level dropped by 12 dB from 8 to 13 s](../images/inspection-loudness-light.webp#only-light)
![Momentary and short-term loudness of the take with its level dropped by 12 dB from 8 to 13 s](../images/inspection-loudness-dark.webp#only-dark)

The window defaults to `SHORT_TERM` (3 s), which follows how loud a passage
is; [`MOMENTARY`][kudio.core.loudness.MOMENTARY] (400 ms) follows syllables.
Those are the two EBU Tech 3341 meters, and any other length in seconds works
too; each steps by 100 ms ([`CURVE_HOP`][kudio.core.loudness.CURVE_HOP]), as
the meters do, unless `hop=` says otherwise.
The short-term curve rides over the pauses but blurs the edges, placing a drop
that ran from 8 to 13 s at 9.1 to 11.9 s. The momentary curve dives into every
pause: on it, `spans_below(max - 6)` finds the drop and ten dips of 0.3 to
0.4 s, and `min_seconds=0.5`, longer than the dips, leaves only the drop, at
8.1 to 12.9 s. For a first look at a long take, and for
[`loudest()`][kudio.core.loudness.LoudnessCurve.loudest] and
[`quietest()`][kudio.core.loudness.LoudnessCurve.quietest], the short-term
curve is the steadier one.

Two choices matter when you plot or search a curve. `curve.times` holds the
centre of each window, not its end as on a meter, so the first value of a 3 s
curve sits at 1.5 s, lined up with the audio that produced it. And the curve
is not gated, since holes where the gate fired would describe a recording that
stops existing between words. A window of digital silence reads `-inf`, which
`quietest()` skips and
[`spans_below()`][kudio.core.loudness.LoudnessCurve.spans_below] counts as
below.

### Loudness range

[`loudness_range`][kudio.core.loudness.loudness_range] reduces the short-term
curve to one figure in LU, to EBU Tech 3342: its relative gate sits 20 LU below
the mean, not the 10 LU of the integrated figure, and it takes the spread from
the 10th to the 95th percentile, so a single door slam does not set it. 20 ms
at full scale in 20 s of a steady tone peaking at 0.3 moves it by 0.03 LU. A short-term curve
you already have gives
the same number as [`range_lu`][kudio.core.loudness.LoudnessCurve.range_lu],
without measuring again:

```python
print(f"{kudio.loudness_range(take, sr):.1f} LU")
print(f"{kudio.loudness_range(drift, sr):.1f} LU, {curve.range_lu:.1f} LU")
```

```text
0.4 LU
12.0 LU, 12.0 LU
```

## True peak

The largest sample is not the largest value the signal reaches. Samples are
points on a curve that passes between them, and a converter, a lossy encoder or
a resampler rebuilds that curve and clips where it overshoots.
[`true_peak`][kudio.core.loudness.true_peak] measures the overshoot in dBTP, to
BS.1770-4 Annex 2, by oversampling. The textbook case is a full-scale sine at a
quarter of the sample rate, phased so that every sample lands on ±0.707:

```python
n = np.arange(sr)
tone = np.sin(2 * np.pi * (sr / 4) * n / sr + np.pi / 4)
tricky = kudio.fade(tone, sr, 0.01, 0.01)
peak = 20 * np.log10(np.max(np.abs(tricky)))
print(f"samples {peak:.2f} dBFS, true peak {kudio.true_peak(tricky, sr):+.2f} dBTP")
```

```text
samples -3.01 dBFS, true peak +0.02 dBTP
```

The sine reaches 1.0 between the samples, 0 dBTP, and the meter lands within
0.02 dB of it. Without the 10 ms fades it reads +0.12 dBTP: a clip that starts
and stops dead at 0.707 rings at both ends, and a converter playing it would
ring the same way. The meter oversamples to at least 192 kHz, as the standard
asks, which is 4x at 48 kHz and 12x at 16 kHz.

EBU R 128 allows a true peak of at most -1 dBTP. `normalize_lufs` warns only
about samples past full scale, so measure the true peak after normalising:

```python
for target in (-23.0, -17.0):
    out = kudio.normalize_lufs(take, sr, lufs=target)
    sample = 20 * np.log10(np.max(np.abs(out)))
    print(f"{target:.0f} LUFS: samples {sample:+.2f} dBFS, "
          f"true peak {kudio.true_peak(out, sr):+.2f} dBTP")
```

```text
-23 LUFS: samples -6.43 dBFS, true peak -6.14 dBTP
-17 LUFS: samples -0.43 dBFS, true peak -0.14 dBTP
```

At -17 LUFS every sample is below full scale, no warning is raised, and the
take is still over the ceiling.

## How reverberant: SRMR

A room keeps every frequency ringing after the sound that excited it has
stopped, and that fills the dips between syllables.
[`srmr`][kudio.core.srmr.srmr] measures how much of that has happened with no
clean recording to compare against: the speech-to-reverberation modulation
energy ratio of Falk, Zheng and Chan.[^srmr] It follows the envelope of 23
auditory bands and puts the energy of their slow modulations, 4 to 18 Hz
where syllables are, over the energy of the faster ones a room adds. Higher
means drier. kudio follows the authors' MATLAB toolbox step by step, and on
the toolbox's own test recording gives the four ratios the toolbox stores to
within 1e-12 of each, relative
([How the metrics are checked](../validation.md#srmr)). It needs nothing
beyond numpy and scipy, and runs at 8 and 16 kHz only.

The take above has no syllables, only a tone switched on and off every
1.4 s, and SRMR finds nothing in it to measure: 1.29 dry and 1.29 to 1.34 in
rooms of 0.3 to 1 s. Give it four syllables a second and the rooms come
apart:

```python
syllables = 0.5 + 0.5 * np.sin(2 * np.pi * 4 * t)      # four a second
hiss = 0.002 * np.random.default_rng(1).standard_normal(t.size)
talk = 0.05 * voice * words * syllables + hiss
print(f"dry         SRMR {kudio.srmr(talk, sr):.2f}")
for rt60 in (0.3, 0.6, 1.0):
    wet = kudio.apply_rir(talk, kudio.rir(sr, rt60=rt60, seed=0))
    print(f"RT60 {rt60:.1f} s  SRMR {kudio.srmr(wet, sr):.2f}")
```

```text
dry         SRMR 9.80
RT60 0.3 s  SRMR 10.09
RT60 0.6 s  SRMR 9.42
RT60 1.0 s  SRMR 8.12
```

From 0.3 s on it falls as the room grows, but the dry take scores below the
0.3 s room: on this voice, SRMR cannot tell dry from a short room. It is a
ratio rather than an opinion score and has no fixed range, so I compare a
take only with the same take somewhere else, in another room or before and
after dereverberation, never one talker with another. Noise lowers it as
well, since a noise floor fluctuates fast; `norm=True` is the later variant
meant for noise and reverberation together. How the silence cut works, the
fast mode, and why SRMRpy gives other numbers for the same file are in the
[reference](../reference/inspection.md).

## How noisy: WADA-SNR

[`wada_snr`][kudio.core.wada.wada_snr] estimates how far the speech sits above
its noise from the recording alone, by how peaked its sample values are. Kim
and Stern[^wada] model clean speech, pauses included, as gamma-distributed
and the noise as Gaussian. Added noise makes the samples less peaked, by an
amount that depends only on the SNR, and kudio reads the SNR off that curve,
computed from the model rather than from the Monte Carlo table that
circulates with the method. It needs nothing beyond numpy and scipy, and the
rate is there only to measure the clip in seconds.

```python
speech = 0.05 * voice * words * syllables
noise = np.random.default_rng(2).standard_normal(t.size)
for snr_db in (0, 5, 10, 20, 30):
    noisy = kudio.add_noise_snr(speech, noise, snr_db=snr_db, seed=0)
    print(f"{snr_db:2d} dB  WADA-SNR {kudio.wada_snr(noisy, sr):6.2f}")
```

```text
 0 dB  WADA-SNR   0.55
 5 dB  WADA-SNR   5.14
10 dB  WADA-SNR  10.57
20 dB  WADA-SNR  25.48
30 dB  WADA-SNR  73.34
```

Up to 10 dB it is within 0.6 dB here. Above about 20 dB the shape of the
speech decides more than the noise does, and this voice is not the model's
speech: 20 dB reads 25.5 and 30 dB reads 73. So I use it to sort and filter
recordings, and read anything above 30 dB as "clean" rather than as a figure.
The noise has to be close to Gaussian as well; a second talker in the
background is read as more speech.

The estimate comes from the pauses, where the noise is on its own, so a clip
whose pauses are digital silence has nothing to read. `speech` on its own
gets NaN and a warning that 40% of it is digital silence, and so does a
noise-gated or edited recording. A clip shorter than 1 s, or one that falls
outside the -20 to 100 dB the curve covers, also gets NaN with a warning,
never the number at the end of the range.

## DNSMOS: what a listener would say

Whether a recording sounds good is a judgement, and with no reference and no
listening panel the only source of one is a model trained on a panel's
opinions. Microsoft's
[DNS-Challenge](https://github.com/microsoft/DNS-Challenge) distributes
three, and kudio runs each as the challenge's `dnsmos_local.py` does.

- [`dnsmos`][kudio.core.dnsmos.dnsmos] runs DNSMOS P.835,[^dnsmos] which
  predicts the three ITU-T P.835 scores, each on its 1 to 5 scale, as a
  [`DnsmosScore`][kudio.core.dnsmos.DnsmosScore]: SIG for the speech, BAK
  for how unobtrusive the background is, OVRL overall. Under denoising SIG
  and BAK move apart, since aggressive suppression eats the speech along
  with the noise, and one number would hide that.
- `dnsmos(..., personalized=True)` runs the personalised P.835 model, which
  also marks down a second talker. It was released with the ICASSP 2023
  challenge;[^dns2023] its mapping onto the MOS scale is in
  `dnsmos_local.py`, and I know of no paper on how it was trained or
  tested.
- [`dnsmos_p808`][kudio.core.dnsmos.dnsmos_p808] runs the P.808
  model,[^p808] which gives one overall score and reads a log mel
  spectrogram rather than the waveform.

The weights belong to Microsoft and are distributed with DNS-Challenge, and
I would rather not bundle someone else's binary in a library that is
otherwise source only. So this needs two things the rest of the page does
not: `pip install "kudio[dnsmos]"` for onnxruntime, and a checkout of that
repository.

```python
score = kudio.dnsmos(take, sr, model_dir="DNS-Challenge/DNSMOS")
print(score.sig, score.bak, score.ovrl)
print(score.summary())
print(kudio.dnsmos_p808(take, sr, model_dir="DNS-Challenge/DNSMOS"))
```

[`find_dnsmos_model`][kudio.core.dnsmos.find_dnsmos_model] looks in
`model_dir`, then in `KUDIO_DNSMOS_DIR`; either can name the file or a
folder to search, subfolders included. The two P.835 models share a file
name, so a search takes the one in `pDNSMOS/` last for `dnsmos` and first
for `personalized=True`. A model from the other folder is run with a
warning, since the polynomials applied to it were fitted to the other one.
Among files of one name the first in sorted path order wins, so point at
the file when it matters which one runs. Then:

- The audio is resampled to 16 kHz, the rate the models were trained at.
  kudio resamples with soxr, as a current librosa does, and the script's
  librosa 0.8.1 with resampy, so the agreement on
  [How the metrics are checked](../validation.md#dnsmos) holds for 16 kHz
  audio.
- It is cut into 9.01 s windows at a 1 s hop, the length the model expects,
  and the windows' scores are averaged; `score.segments` says over how many.
  The windows are laid out as in DNS-Challenge's `dnsmos_local.py`, so a
  score here matches one published from that script: they start on whole
  seconds and end inside the clip's last whole second, so a 10.5 s clip gets
  one and a 20 s clip 11, with its last 0.99 s left out. The script itself
  scores only 7 of those 11: it works out each window's end in floating
  point, which rounds the windows starting at 7 to 23 s one sample short, and
  skips them. I keep them, so the two agree on any clip that stays under
  17 s once doubled, and kudio scores more of a longer one.
- A clip shorter than one window is doubled until it holds one, as that
  script does, rather than padded with zeros, and every window of the result
  is scored: a 2 s clip becomes 16 s and seven windows. Silence scores as
  ruined speech, and seven seconds of it after a two-second clip would report
  a problem the clip does not have.
- Each window's raw P.835 outputs go onto the MOS scale through
  [`POLYFIT`][kudio.core.dnsmos.POLYFIT], or
  [`POLYFIT_PERSONALIZED`][kudio.core.dnsmos.POLYFIT_PERSONALIZED] for the
  personalised model, the polynomials in `dnsmos_local.py`, before the
  average is taken, the order the script uses. A `polyfit.json` beside the
  model replaces them, which any other model needs, and `polyfit=False`
  keeps the raw outputs.
- The P.808 model gets each window's log mel spectrogram as the script
  computes it with librosa, and the script's requirements.txt pins librosa
  0.8.1. librosa 0.9 changed that spectrogram twice: it pads the ends of the
  signal with zeros where 0.8.1 reflected them, and for the script's
  321-point FFT it puts the bins at their true frequencies, 49.84 Hz apart,
  where 0.8.1 put them 50 Hz apart and left the lowest of the 120 mel bands
  empty. The script under a current librosa, and torchmetrics, compute the
  newer one, and on the clips of the check the score moves by up to 0.19
  between the two. `dnsmos_p808` follows the pin; `librosa="0.9"` gives the
  other, and with it torchmetrics 1.9.0's score.

Input is held to the metrics' rule: malformed input raises `ValueError`,
and an empty clip scores nan with a warning (see
[Input a metric cannot score](metrics.md#input-a-metric-cannot-score)).

[`summary()`][kudio.core.dnsmos.DnsmosScore.summary] calls OVRL from 4.0 up
clean, from 3.0 usable, from 2.0 poor and below that bad. When SIG and BAK are
more than 0.7 apart it says which way: SIG 2.6, BAK 4.1 and OVRL 2.9 come out
as "poor (OVRL 2.90) — the background is cleaner than the speech, which is
what over-aggressive denoising looks like".

For a folder, open the model once and hand the session to every call;
`dnsmos_p808` takes one the same way. The model directory is then never
consulted, so a `polyfit.json` there is not read either and the polynomials
of `dnsmos_local.py` apply:

```python
from pathlib import Path

import onnxruntime as ort

model = kudio.find_dnsmos_model("DNS-Challenge/DNSMOS")
session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
for path in sorted(Path("corpus").rglob("*.wav")):
    try:
        y, sr = kudio.file_load(path)
        print(path.as_posix(), kudio.dnsmos(y, sr, session=session).summary())
    except (kudio.AudioIOError, kudio.FeatureError, ValueError) as e:
        print(path.as_posix(), f"not measured: {e}")
```

For a test set's mean SRMR, WADA-SNR or DNSMOS with a confidence interval,
[`score_dataset`][kudio.core.dataset_score.score_dataset] runs them over a
folder with no reference at all, and `kudio score` does the same from a shell
([Command line](../cli.md#score)).

## From the command line

```bash
kudio info take.wav                        # header, encoding and peak
kudio report take.wav || echo "needs another take"
kudio loudness take.wav --target -23       # exit 1 when over 1 LU off
```

`kudio report` prints the report and its findings and exits 1 when there are
any. `kudio loudness` prints the integrated loudness, the range, the true peak
and the loudest and quietest moments; `--over-time` adds the curve, one line
per 100 ms. `kudio report` reads the file as mono, averaging the channels;
`kudio loudness` keeps them, since BS.1770 weights and sums the channels and
the mono mix of a stereo file reads 3 LU low. The flags are on [Command line](../cli.md), and every parameter
of the functions here is in the [API reference](../reference/inspection.md).

[^srmr]: T. H. Falk, C. Zheng and W.-Y. Chan, "A non-intrusive quality and intelligibility measure of reverberant and dereverberated speech", IEEE Trans. Audio, Speech, Lang. Process. 18(7), pp. 1766-1774, 2010, doi:10.1109/TASL.2010.2052247. The normalised variant: J. F. Santos, M. Senoussaoui and T. H. Falk, "An improved non-intrusive intelligibility metric for noisy and reverberant speech", Proc. IWAENC 2014, pp. 55-59, doi:10.1109/IWAENC.2014.6953337.
[^bs1770]: Recommendation ITU-R BS.1770-4 (10/2015), "Algorithms to measure audio programme loudness and true-peak audio level". Annex 1 is the loudness measurement and Annex 2 the true-peak meter.
[^ebu]: EBU R 128 (2023), "Loudness normalisation and permitted maximum level of audio signals"; EBU Tech 3341 (November 2023), "Loudness metering: 'EBU Mode' metering to supplement EBU R 128 loudness normalization"; EBU Tech 3342 (November 2023), "Loudness range: a measure to supplement EBU R 128 loudness normalization".
[^dnsmos]: C. K. A. Reddy, V. Gopal and R. Cutler, "DNSMOS P.835: A non-intrusive perceptual objective speech quality metric to evaluate noise suppressors", Proc. ICASSP 2022, pp. 886-890, doi:10.1109/ICASSP43922.2022.9746108. It was trained on ratings collected to ITU-T Recommendation P.835 (11/2003), "Subjective test methodology for evaluating speech communication systems that include noise suppression algorithm".
[^wada]: C. Kim and R. M. Stern, "Robust signal-to-noise ratio estimation based on waveform amplitude distribution analysis", Proc. Interspeech 2008, pp. 2598-2601, doi:10.21437/Interspeech.2008-644.
[^p808]: C. K. A. Reddy, V. Gopal and R. Cutler, "DNSMOS: A non-intrusive perceptual objective speech quality metric to evaluate noise suppressors", Proc. ICASSP 2021, pp. 6493-6497, doi:10.1109/ICASSP39728.2021.9414878. It was trained on ratings collected with the crowdsourcing method of ITU-T Recommendation P.808 (06/2018), "Subjective evaluation of speech quality with a crowdsourcing approach".
[^dns2023]: H. Dubey, A. Aazami, V. Gopal, B. Naderi, S. Braun, R. Cutler, A. Ju, M. Zohourian, M. Tang, M. Golestaneh and R. Aichner, "ICASSP 2023 Deep Noise Suppression Challenge", IEEE Open Journal of Signal Processing 5, pp. 725-737, 2024, doi:10.1109/OJSP.2024.3378602; first posted as arXiv:2303.11510, 2023.
