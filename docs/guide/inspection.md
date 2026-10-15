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
that it peaks at -0.0 dBFS. A clipped recording turned down afterwards, even by
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
as the rate the file was upsampled from; otherwise it says "band-limited to
about" twice the bandwidth, 6438 Hz for content that stops at 3219 Hz.

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

A file soundfile cannot open raises
[`AudioIOError`][kudio.exceptions.AudioIOError], and an empty one a
`FeatureError`. Catch both, or the sweep stops at the first broken file:

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
ITU-R BS.1770-4, as gated integrated loudness in LUFS:

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
that brings a clip to a target, -23 LUFS by default (EBU R 128), and returns
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

It logs a warning when that happens, and kudio's logging is silent until your
program configures it ([Getting started](../getting-started.md) shows how).
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
too; all of them step by 100 ms ([`CURVE_HOP`][kudio.core.loudness.CURVE_HOP]).
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
at full scale in 20 s of steady tone moves it by 0.03 LU. A short-term curve
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

At -17 LUFS every sample is below full scale, nothing is logged, and the take
is still over the ceiling.

## DNSMOS: what a listener would say

Whether a recording sounds good is a judgement, and with no reference and no
listening panel the only source of one is a model trained on a panel's
opinions. [`dnsmos`][kudio.core.dnsmos.dnsmos] runs DNSMOS, which predicts the
three ITU-T P.835 scores on the 1 to 5 MOS scale as a
[`DnsmosScore`][kudio.core.dnsmos.DnsmosScore]: SIG for the speech, BAK for
how unobtrusive the background is, OVRL overall. Under denoising SIG and BAK
move apart, since aggressive suppression eats the speech along with the
noise, and one number would hide that.

The weights belong to Microsoft's
[DNS-Challenge](https://github.com/microsoft/DNS-Challenge) and are distributed
there, and I would rather not bundle someone else's binary in a library that is
otherwise source only. So this needs two things the rest of the page does not:
`pip install kudio[dnsmos]` for onnxruntime, and a checkout of that repository.

```python
score = kudio.dnsmos(take, sr, model_dir="DNS-Challenge/DNSMOS")
print(score.sig, score.bak, score.ovrl)
print(score.summary())
```

[`find_dnsmos_model`][kudio.core.dnsmos.find_dnsmos_model] looks in `model_dir`,
then in `KUDIO_DNSMOS_DIR`; either can name the file or a directory to search,
subfolders included. When a checkout holds more than one `sig_bak_ovr.onnx`,
the first in sorted path order wins, so point at the file when it matters
which one runs. Then:

- The audio is resampled to 16 kHz, the rate the model was trained at.
- It is cut into 9.01 s windows at a 1 s hop, the length the model expects,
  and the raw outputs of the windows are averaged; `score.segments` says over
  how many. What comes after the last full window is not scored: a 20 s clip
  gets 11 windows, and its last 0.99 s is left out.
- A clip shorter than one window is repeated to fill it, not padded with
  zeros. Silence scores as ruined speech, and seven seconds of it after a
  two-second clip would report a problem the clip does not have.
- The average goes onto the MOS scale through
  [`POLYFIT`][kudio.core.dnsmos.POLYFIT], the polynomials published for the
  non-personalised model. A `polyfit.json` beside the model replaces them,
  which any other model needs, and `polyfit=False` keeps the raw outputs.

[`summary()`][kudio.core.dnsmos.DnsmosScore.summary] calls OVRL from 4.0 up
clean, from 3.0 usable, from 2.0 poor and below that bad. When SIG and BAK are
more than 0.7 apart it says which way: SIG 2.6, BAK 4.1 and OVRL 2.9 come out
as "poor (OVRL 2.90) — the background is cleaner than the speech, which is
what over-aggressive denoising looks like".

For a folder, open the model once and hand the session to every call. The
model directory is then never consulted, so a `polyfit.json` there is not read
either and the published coefficients apply:

```python
from pathlib import Path

import onnxruntime as ort

model = kudio.find_dnsmos_model("DNS-Challenge/DNSMOS")
session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
for path in sorted(Path("corpus").rglob("*.wav")):
    try:
        y, sr = kudio.file_load(path)
        print(path.as_posix(), kudio.dnsmos(y, sr, session=session).summary())
    except (kudio.AudioIOError, kudio.FeatureError) as e:
        print(path.as_posix(), f"not measured: {e}")
```

## From the command line

```bash
kudio info take.wav                        # header, encoding and peak
kudio report take.wav || echo "needs another take"
kudio loudness take.wav --target -23       # exit 1 when over 1 LU off
```

`kudio report` prints the report and its findings and exits 1 when there are
any. `kudio loudness` prints the integrated loudness, the range, the true peak
and the loudest and quietest moments; `--over-time` adds the curve, one line
per 100 ms. Both read the file as mono, averaging the channels. On a stereo
file that puts `kudio loudness` 3.01 LU or more under the BS.1770 figure, and
`--target` judges that number, so measure a stereo master in Python with
`mono=False`. The flags are on [Command line](../cli.md), and every parameter
of the functions here is in the [API reference](../reference/inspection.md).
