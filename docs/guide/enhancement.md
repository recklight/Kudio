# Denoising

`kudio.enhance` is single-channel statistical denoising: no model, no training
data, nothing to download. It estimates the noise, works out how much speech
each time-frequency bin holds, and turns each bin down accordingly. Seven gain
rules and four noise estimators share one STFT loop, so any rule runs over any
estimator, and four of the rules also run on a live stream.

| You have | Reach for |
|---|---|
| one noisy clip | [`spectral_enhance`][kudio.enhance.spectral.spectral_enhance] |
| a clip, and the question of which method suits it | [`compare_enhancers`][kudio.enhance.spectral.compare_enhancers] |
| audio that is still arriving | [`StreamEnhancer`][kudio.enhance.streaming.StreamEnhancer] |
| a folder of recordings | [`enhance_folder`][kudio.enhance.spectral.enhance_folder] |

??? example "The clip the examples run on"

    Five seconds at 16 kHz: two harmonic "voices" with pauses before, between
    and after them, and white noise mixed in at 5 dB SNR with
    [`add_noise_snr`][kudio.effects.augment.add_noise_snr]. That SNR is over
    the whole clip, pauses included; while the voices are speaking it is
    7.0 dB. It is seeded, so the blocks below, run after this one, print what
    the page shows.

    ```python
    import numpy as np

    import kudio

    sr = 16000
    t = np.arange(sr * 5) / sr
    rng = np.random.default_rng(0)

    def voice(start, stop, f0):
        """Thirteen harmonics of f0, rising and falling 3.5 times a second."""
        span = (t >= start) & (t < stop)
        tone = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 14))
        swell = 0.5 + 0.5 * np.sin(2 * np.pi * 3.5 * t)
        return np.where(span, tone * swell / 7, 0.0)

    clean = (voice(0.5, 2.0, 140) + voice(2.8, 4.5, 170)).astype(np.float32)
    noisy = kudio.add_noise_snr(clean, rng.standard_normal(len(clean)),
                                snr_db=5, seed=0)
    ```

## One call

```python
enhanced = kudio.spectral_enhance(noisy, sr)   # log-MMSE, MCRA noise estimate
before, after = kudio.si_sdr(clean, noisy), kudio.si_sdr(clean, enhanced)
print(f"SI-SDR {before:.2f} dB -> {after:.2f} dB")
```

```text
SI-SDR 4.98 dB -> 12.21 dB
```

![Spectrograms of the noisy clip and the log-MMSE output from 0 to 4 kHz. After the first two seconds the noise in the output is about 19 dB lower and the harmonics are intact; before that, horizontal streaks of residual noise remain.](../images/enhancement-before-after-light.webp#only-light)
![Spectrograms of the noisy clip and the log-MMSE output from 0 to 4 kHz. After the first two seconds the noise in the output is about 19 dB lower and the harmonics are intact; before that, horizontal streaks of residual noise remain.](../images/enhancement-before-after-dark.webp#only-dark)

The streaks in the first two seconds are the MCRA noise estimate settling;
[Four noise estimators](#four-noise-estimators) says why.

The output is float32, as long as the input and in step with it sample for
sample, so it goes straight into a metric or an A/B. The audio has to be mono,
and a clip shorter than one frame (512 samples unless you pass `n_fft=`) is
refused rather than padded. The method's own parameters are keywords, and one
it does not take is an error:

```python
kudio.spectral_enhance(noisy, sr, "wiener", q=0.4)
```

```text
kudio.exceptions.FeatureError: 'wiener' does not take q. It takes: alpha, gain_floor_db
```

## Noise estimate, then gain

For each STFT frame $l$ and bin $k$, every method estimates the noise power
$\lambda_d(k,l)$ with one of four estimators (`noise=`) and turns it into a
gain between a floor and 1, which multiplies the complex spectrum and leaves
the noisy phase alone. `wiener`, `mmse_stsa`, `logmmse` and `omlsa` take one
more step in between: the decision-directed estimate of the a priori SNR
$\xi(k,l)$, how much speech the bin holds, from Ephraim and Malah[^em84]. They
differ only in the gain function after it:

$$
\xi(k,l) = \alpha\,\frac{|\hat S(k,l-1)|^2}{\lambda_d(k,l)}
         + (1-\alpha)\,\max\bigl(\gamma(k,l)-1,\;0\bigr),
\qquad
\gamma(k,l) = \frac{|Y(k,l)|^2}{\lambda_d(k,l)}
$$

$\hat S(k,l-1)$ is the previous frame's speech estimate. At $\alpha = 0.98$ the
estimate leans almost entirely on it, which keeps musical noise down and makes
these four slow to open up at a speech onset. $\gamma$ is clamped at 40 dB,
since one loud frame would otherwise drag the estimate somewhere it takes
seconds to leave, and $\xi$ is floored at −25 dB. `specsub`, `multiband` and
`spectral_gate` go from the noise estimate straight to a gain.

I keep the estimator and the gain rule apart so one can be held still while
the other varies. On this clip the seven rules over one estimator span 5.7 dB
of SI-SDR on average, and the four estimators under one rule span 7.5 dB.
Which matters more depends on the recording, which is why
[`compare_enhancers`][kudio.enhance.spectral.compare_enhancers] can sweep both.

## Seven gain rules

| Method | What it does | Why you would pick it | Its own parameters |
|---|---|---|---|
| `specsub` | Subtracts a multiple of the noise power: Boll's method[^boll] with Berouti's over-subtraction and floor[^berouti] | Cheapest; the origin of "musical noise" | `over_subtraction` 2.0, `spectral_floor` 0.002 |
| `multiband` | Subtraction with a factor per band: 4.75 below −5 dB band SNR, sliding to 1.0 above 20 dB[^kamath] | Colored noise is not equally loud everywhere | `bands` 6, `spectral_floor` 0.002 |
| `wiener` | Gain $\xi/(1+\xi)$ | Smooth, so little musical noise; muffles at low SNR | `alpha` 0.98 |
| `mmse_stsa` | MMSE estimate of the amplitude[^em84] | Keeps more speech than Wiener | `alpha` 0.98 |
| `logmmse` | MMSE in the log-amplitude domain[^em85] | The default; with MCRA it came within 0.05 dB of the best rule on this clip | `alpha` 0.98 |
| `omlsa` | log-MMSE weighted by a speech-presence probability[^omlsa] | Takes bins it judges speech-free down to the gain floor, so the pauses here come out about 3 dB quieter than log-MMSE's; `q` sets how readily a bin counts as empty | `alpha` 0.98, `q` 0.30 |
| `spectral_gate` | Keeps bins more than `n_std_thresh` times the noise estimate's spread over time (in dB) above it, through a smoothed mask | A plain threshold, as in noisereduce. On this page's white noise it came last under every estimator and turned the speech down 5 dB along with the noise | `n_std_thresh` 1.5, `prop_decrease` 1.0, `freq_smooth_hz` 500, `time_smooth_ms` 50 |

A noise estimate that holds still has next to no spread over time, `quantile`
none at all and `initial` only over the opening frames it averages, so with
those `spectral_gate`'s threshold sits on the estimate or just above it. All
the others also take `gain_floor_db`, −25 dB by default: how far
any bin may be turned down. A hard zero is the wrong goal, because total
suppression is what makes the residue warble. For the same $\xi$ and $\gamma$,
Wiener turns a bin down furthest, MMSE-STSA least and log-MMSE in between, and
at high SNR MMSE-STSA comes out the same as Wiener. The test suite checks both.

[`METHODS`][kudio.enhance.spectral.METHODS] holds all of this as data. Each
entry is a [`Method`][kudio.enhance.spectral.Method] whose
[`Param`][kudio.enhance.spectral.Param]s carry a default, range, step, unit and
a sentence of help, enough for a GUI or a command line to build its controls
without restating them:

```python
q = {p.name: p for p in kudio.METHODS["omlsa"].params}["q"]
print(q.default, q.low, q.high, q.step)
print(q.help)
```

```text
0.3 0.01 0.95 0.05
A priori probability that a bin holds no speech. Higher suppresses more and risks clipping onsets.
```

## Four noise estimators

| Estimator | What it assumes | `noise_kwargs` and defaults | Streams |
|---|---|---|---|
| `mcra` | The noise is a running average of each bin's power, held still where the bin sits more than 5 times above its running minimum, since it probably holds speech there[^mcra] | `window=100` frames, `delta=5` (also `ass`, `ad`, `ap`) | yes |
| `quantile` | The noise is a low quantile of each bin over the whole file, scaled up to the mean of steady noise | `quantile=0.15` | no |
| `initial` | The opening frames are noise alone; their mean is held from then on | `n_frames=6` | yes |
| `minimum` | The noise is a sliding-window minimum of the smoothed spectrum, times a fixed bias | `window=60`, `bias=1.5`, `smoothing=0.85` | yes |

`mcra` is the default. [`NOISE_ESTIMATORS`][kudio.enhance.spectral.NOISE_ESTIMATORS]
holds a sentence on each, and on this clip the choice moves the result more
than the gain rule does:

```python
print(kudio.NOISE_ESTIMATORS["initial"])
for noise in kudio.NOISE_ESTIMATORS:
    out = kudio.spectral_enhance(noisy, sr, "logmmse", noise=noise)
    print(f"{noise:9} {kudio.si_sdr(clean, out):.2f} dB")
```

```text
Average of the first few frames, assumed to be room tone. Exact when that is true and badly wrong when it is not.
mcra      12.21 dB
quantile  14.16 dB
initial   14.86 dB
minimum   6.86 dB
```

`initial` wins here because the clip opens with half a second of noise and
nothing else, which is its assumption exactly. Its six frames reach 56 ms into
the clip at 16 kHz with the default framing. The frame is 512 samples whatever
the rate, so at 48 kHz the same six frames reach 19 ms; raise `n_frames` when
the room tone lasts longer.

`mcra` explains the streaks in the figure above. Its estimate and its running
minimum both start from the first frame, and the offline STFT is centered, so
half of that frame is zero padding: here a third of its bins start more than
7 dB under the noise. Such a bin looks like speech, so its estimate holds
still, and it stays too low until the minimum search has run through two
100-frame windows, 1.6 s at 16 kHz. The opening half second keeps about 10 dB
more residual noise than the pause at 2.0–2.8 s (−38.5 against −48.1 dBFS). In
a recording that opens with a few seconds of room tone, that is over before
anyone speaks.

`quantile` takes the 15th percentile of each bin over the whole file and
scales it up. Steady noise power in a bin is exponentially distributed, so its
q-th percentile sits a factor of $-\ln(1-q)$ under its mean, 7.9 dB at the
default, and the estimate is divided by that. Without assuming anything about
how the file opens, it lands within 0.7 dB of `initial` here. The quantile has
to stay below the share of the file in which a bin holds no speech: at
`noise_kwargs={"quantile": 0.5}` speech gets into the statistic and this clip
drops to 12.87 dB. `noise_kwargs` goes to the estimator, and a key it does not
take is refused with a `FeatureError` that lists the ones it does.

For a loop of your own,
[`estimate_noise_psd`][kudio.enhance.spectral.estimate_noise_psd] runs any
estimator over a `(frames, bins)` power spectrogram, and
[`make_noise_tracker`][kudio.enhance.spectral.make_noise_tracker] gives every
one but `quantile` as state plus a one-frame update.
[`DecisionDirected`][kudio.enhance.spectral.DecisionDirected] does the same for
the a priori SNR, and [`make_gain_rule`][kudio.enhance.spectral.make_gain_rule]
returns the gain function of one of the four decision-directed rules, without
the floor: clip its output to `[10 ** (gain_floor_db / 20), 1]` as
`spectral_enhance` does. The offline and the streaming path are both built
from these objects, so they share every update rule.

## Choosing a method

[`compare_enhancers`][kudio.enhance.spectral.compare_enhancers] runs several
methods over one clip and measures each. A run is a (method, estimator) pair:
`methods=` picks the rules (all seven when omitted), `noises=` sweeps the
estimators, and the two together give the full grid.

```python
results = kudio.compare_enhancers(noisy, sr, reference=clean,
                                  noises=list(kudio.NOISE_ESTIMATORS),
                                  metrics=False)
ranked = sorted((r for r in results if not r.error),
                key=lambda r: r.si_sdr_db, reverse=True)
print(len(ranked))
for r in ranked[:4]:
    print(f"{r.label:20} SI-SDR {r.si_sdr_db:+.2f} dB",
          f"  dSNR {r.delta_snr_db:+.2f} dB")
```

```text
28
wiener + initial     SI-SDR +15.37 dB   dSNR +10.48 dB
omlsa + initial      SI-SDR +15.26 dB   dSNR +10.36 dB
logmmse + initial    SI-SDR +14.86 dB   dSNR +10.00 dB
multiband + quantile SI-SDR +14.52 dB   dSNR +9.65 dB
```

Each row is an [`EnhanceResult`][kudio.enhance.spectral.EnhanceResult] with
the time, noise floor and loudness. With `reference=` it also has SNR, SI-SDR
and `delta_snr_db`, and with `metrics=True`, the default, PESQ and STOI from
`kudio[pesq]` and `kudio[eval]` where installed (PESQ only at 8 and 16 kHz).
They are most of the time a comparison takes: on my machine the 28 runs above
took 2.0 s with STOI and 1.0 s without. A run that raises comes back as
a row with `error` set, its floor and loudness NaN and its scores `None`, so
filter those out before sorting, as above. The clock starts after a warm-up run, so the first method
is not charged for librosa's JIT compilation.

`delta_snr_db` is the output's SNR minus the untouched input's, both against
the reference: the number that answers "did this help". It is plain SNR, so a
method that also turns the speech down pays for it, while SI-SDR ignores level.
The two can tell different stories. In the same 28 runs, `spectral_gate + mcra`
gains 3.15 dB of SI-SDR (4.98 to 8.13) but only 1.16 dB of SNR, because it
turns the speech down 5 dB along with the noise. `pesq_mode=` and
`pesq_scale=` go to PESQ; [Measuring quality](metrics.md) explains which scale
a paper means.

Other keywords go to each run that accepts them and are dropped for the rest,
the opposite of [`spectral_enhance`][kudio.enhance.spectral.spectral_enhance]:
`q=0.9` reaches `omlsa` alone, so it can share a call with six methods that
have no `q`, while `n_fft=` and `noise_kwargs=` reach every run. A keyword that
no method in the call takes is a typo, and is refused before anything runs; a
`noise_kwargs` key that only one swept estimator takes fails the others' runs.
`noise="initial"` fixes the estimator instead of sweeping it; together with
`noises=` it is refused.

Without a reference, the normal case for a real recording, you get the noise
floor (the 10th percentile of 30 ms frame levels, as
[`audio_report`][kudio.core.report.audio_report] measures it on
[Inspecting a recording](inspection.md)) and the loudness:

```python
for r in kudio.compare_enhancers(noisy, sr, ["wiener", "logmmse", "omlsa"]):
    print(f"{r.label:16} floor {r.noise_floor_db:.1f} dBFS   {r.lufs:.1f} LUFS")
```

```text
wiener + mcra    floor -53.6 dBFS   -24.4 LUFS
logmmse + mcra   floor -48.6 dBFS   -24.3 LUFS
omlsa + mcra     floor -51.2 dBFS   -24.5 LUFS
```

The lowest floor is not the best result: `wiener` leaves the least noise here
and still scores below `logmmse` against the reference, 11.84 against
12.21 dB. Neither column says what happened to the speech, so listen before
trusting the smallest number.

### Ranking your own denoiser

A trained model, a wrapper around another library or a one-off experiment can
sit in the same table: pass a function of `(y, sr)` that returns the enhanced
waveform, bare or as `(name, function)`.

```python
def lowpassed(y, sr):
    return kudio.lowpass(kudio.spectral_enhance(y, sr), sr, 4000)

for r in kudio.compare_enhancers(noisy, sr, ["logmmse", kudio.trad_enhance,
                                             ("lowpassed", lowpassed)],
                                 reference=clean, metrics=False):
    print(f"{r.label:20} SI-SDR {r.si_sdr_db:+.2f} dB")
```

```text
logmmse + mcra       SI-SDR +12.21 dB
trad_enhance + own   SI-SDR +14.09 dB
lowpassed + own      SI-SDR +13.08 dB
```

A bare function is named after itself, and the estimator column reads `own`
([`CUSTOM_NOISE`][kudio.enhance.spectral.CUSTOM_NOISE]): it brings its own
noise handling, so it runs once even when `noises=` sweeps the built-in ones.
[`CustomEnhancer`][kudio.enhance.spectral.CustomEnhancer] is the same entry as
an object. A model trained with
[KudioEnhance](https://github.com/recklight/KudioEnhance) goes in the same way,
as `("model", Enhancer.load(run_dir).enhance)` with `Enhancer` from
`kudioenhance.inference`. The reference metrics compare the samples both
signals have, so an output that comes back a little short is fine, and a
delayed one is not: line it up first, as [Measuring quality](metrics.md) shows.

## Where it does not help

!!! warning "Two limits, measured on the voices above"

    These methods are for noisy audio. Each one trades speech distortion for
    noise removal, and the trade gets worse as the input gets cleaner. Here is
    `delta_snr_db` in dB over MCRA, with white noise mixed into the clean
    voices at each input SNR:

    | Input SNR | Best | Worst of the other six | `spectral_gate` |
    |---|---|---|---|
    | −5 dB | +10.5 | +6.9 | +6.5 |
    | 0 dB | +9.0 | +6.5 | +3.9 |
    | 5 dB | +7.6 | +6.0 | +0.4 |
    | 10 dB | +5.9 | +4.3 | −4.6 |
    | 15 dB | +3.1 | +0.5 | −9.9 |
    | 20 dB | −0.1 | −4.0 | −15.0 |

    `spectral_gate` drops below zero from 10 dB. The other six keep gaining up
    to 15 dB, and at 20 dB all seven leave the clip worse than they found it.
    That is why every `EnhanceResult` carries `delta_snr_db`.

    They need to hear the noise on its own. MCRA learns the noise in the gaps
    between words. The same voices back to back, without the pauses, give it
    almost nothing to learn from, and `initial` then averages speech and calls
    it noise. A second of room tone in front is what `initial` is for:

    | `delta_snr_db` of `logmmse`, 5 dB input SNR | `mcra` | `initial` |
    |---|---|---|
    | With pauses | +7.6 | +9.4 |
    | No pauses | +1.3 | −2.4 |
    | No pauses, one second of room tone first | +3.7 | +8.7 |

??? note "How the two tables were measured"

    ```python
    noise = np.random.default_rng(1).standard_normal(len(clean))
    for snr_in in (-5, 0, 5, 10, 15, 20):
        mixed = kudio.add_noise_snr(clean, noise, snr_db=snr_in, seed=0)
        runs = kudio.compare_enhancers(mixed, sr, reference=clean,
                                       metrics=False)
        delta = {r.method: r.delta_snr_db for r in runs}
        gate = delta.pop("spectral_gate")
        print(f"{snr_in:+3d} dB  best {max(delta.values()):+5.1f}",
              f" worst of six {min(delta.values()):+5.1f}  gate {gate:+5.1f}")

    steady = (voice(0.0, 2.5, 140) + voice(2.5, 5.0, 170)).astype(np.float32)
    lead_in = np.concatenate([np.zeros(sr, np.float32), steady])
    cases = {"pauses": clean, "no pauses": steady, "room tone": lead_in}
    for name, signal in cases.items():
        hiss = np.random.default_rng(1).standard_normal(len(signal))
        mixed = kudio.add_noise_snr(signal, hiss, snr_db=5, seed=0)
        deltas = [kudio.compare_enhancers(mixed, sr, ["logmmse"], noise=est,
                                          reference=signal,
                                          metrics=False)[0].delta_snr_db
                  for est in ("mcra", "initial")]
        print(f"{name:10} mcra {deltas[0]:+.1f}  initial {deltas[1]:+.1f}")
    ```

    ```text
     -5 dB  best +10.5  worst of six  +6.9  gate  +6.5
     +0 dB  best  +9.0  worst of six  +6.5  gate  +3.9
     +5 dB  best  +7.6  worst of six  +6.0  gate  +0.4
    +10 dB  best  +5.9  worst of six  +4.3  gate  -4.6
    +15 dB  best  +3.1  worst of six  +0.5  gate  -9.9
    +20 dB  best  -0.1  worst of six  -4.0  gate -15.0
    pauses     mcra +7.6  initial +9.4
    no pauses  mcra +1.3  initial -2.4
    room tone  mcra +3.7  initial +8.7
    ```

## Audio that is still arriving

[`StreamEnhancer`][kudio.enhance.streaming.StreamEnhancer] runs the same
update a block at a time, so a take can be heard through the denoiser while it
is being recorded. Blocks can be any size:
[`process`][kudio.enhance.streaming.StreamEnhancer.process] returns whatever
audio is finished, and [`flush`][kudio.enhance.streaming.StreamEnhancer.flush]
the rest once nothing more is coming.

```python
enhancer = kudio.StreamEnhancer(sr, "logmmse")
print(f"latency {enhancer.latency_seconds * 1000:.0f} ms")

pieces = [enhancer.process(noisy[i:i + 1024])       # one 64 ms block at a time
          for i in range(0, len(noisy), 1024)]
pieces.append(enhancer.flush())
streamed = np.concatenate(pieces)
print(len(noisy), len(streamed))
print(f"streamed {kudio.si_sdr(clean, streamed):.2f} dB,",
      f"offline {kudio.si_sdr(clean, enhanced):.2f} dB")
```

```text
latency 32 ms
80000 80000
streamed 13.45 dB, offline 12.21 dB
```

The latency is one analysis frame, `n_fft / sr`: nothing comes out until a
whole frame has arrived, and then the output follows in hop-sized pieces. That
is a wait, not a shift, so sample *i* of everything concatenated lines up with
sample *i* of the input. `flush()` zero-pads the last partial frame so the tail
is processed rather than dropped, and stops where the input stopped, so the
whole output is as long as the input. The test suite feeds the same audio in blocks of
1, 7, 128, 512, 1024 and 9999 samples and gets the same output to within
10⁻⁶. [`reset`][kudio.enhance.streaming.StreamEnhancer.reset] clears every
buffer and estimate for the next take.

Streamed and offline output are close but not identical. A stream's first
frame is a whole one, while the offline STFT's is half zero padding, and with
`mcra` that start is worth something to the stream: over ten noise draws of
this clip it came out 0.8 to 1.6 dB of SI-SDR ahead every time. With `initial`
and `minimum` the two stay within 0.7 dB.

Four methods stream, the ones in
[`STREAMABLE_METHODS`][kudio.enhance.streaming.STREAMABLE_METHODS]: `wiener`,
`mmse_stsa`, `logmmse` and `omlsa`, over `mcra`, `initial` or `minimum`.
`spectral_gate` cannot: it sets its threshold from the spread of the noise
estimate over the whole file and smooths its mask forwards and backwards in
time. Nor can `quantile`, a statistic of the whole file. A streaming version of
either would be a different algorithm under the same name. `specsub` and
`multiband` only need the current frame, but `StreamEnhancer` runs the
decision-directed loop and nothing else, so it does not take them. Asking for
any of the three is a `FeatureError` that lists the four that stream and gives
the `spectral_enhance` call that runs the method offline.

From a microphone, feed it
[`StreamRecorder.drain()`][kudio.core.stream.StreamRecorder.drain] and never
[`tail()`][kudio.core.stream.StreamRecorder.tail]: `tail()` hands out
overlapping windows, and samples seen twice corrupt every recursive estimate
the enhancer holds. Keep the recorder mono: `process()` refuses a
two-channel block with a `FeatureError`.
[Recording and live views](live-audio.md) has the loop.

## A whole folder

[`enhance_folder`][kudio.enhance.spectral.enhance_folder] denoises every `.wav`
file under one folder into another, mirroring the subfolders the way
[`convert_folder`][kudio.core.io.convert_folder] does, so two takes with the
same name in different folders do not collide:

```python
hiss = np.random.default_rng(1).standard_normal(len(clean))
for name, snr_db in [("day1/take1.wav", 0), ("day1/take2.wav", 5),
                     ("day2/take1.wav", 10)]:
    kudio.save_wave(f"noisy/{name}",
                    kudio.add_noise_snr(clean, hiss, snr_db=snr_db, seed=0), sr)

result = kudio.enhance_folder("noisy", "enhanced", "logmmse", report=True)
print(result)
for path, before, after in result.floors:
    print(f"{path.as_posix()}: floor {before:.1f} -> {after:.1f} dBFS")
print(f"{result.mean_reduction_db():.1f} dB down on average")
```

```text
3/3 written, 0 failed
enhanced/day1/take1.wav: floor -24.4 -> -43.5 dBFS
enhanced/day1/take2.wav: floor -29.4 -> -48.4 dBFS
enhanced/day2/take1.wav: floor -34.4 -> -53.3 dBFS
19.0 dB down on average
```

The result is an [`EnhanceFolderResult`][kudio.enhance.spectral.EnhanceFolderResult]:
a [`ConvertResult`][kudio.core.io.ConvertResult], false in an `if` when
anything failed, plus the `floors` that `report=True` measures. Parameters are
checked once, before the first file, so a typo is one error and not one
failure per file. `noise=`, `n_fft=`, `hop_length=`, `window=` and
`noise_kwargs=` go to [`spectral_enhance`][kudio.enhance.spectral.spectral_enhance]
as they would in a single call, and the method can also be anything
`compare_enhancers` accepts, a trained model included. A list or a `.txt` list
of files is mirrored from the deepest folder they share, the way
`convert_folder` does it. Each file is loaded as mono and written
as 16-bit PCM, and the [reference][kudio.enhance.spectral.enhance_folder]
covers `sr=`, `subtype=`, `on_error=`, `progress=` and `overwrite=`. A second
run into the same folder keeps what is there and writes the new
`day1/take1.wav` as `day1/take12.wav` unless `overwrite=True`.

## Two fixed recipes

[`trad_enhance`][kudio.enhance.traditional.trad_enhance] is one fixed recipe
from Loizou's *Speech Enhancement: Theory and Practice*: 512-sample Hamming
frames at 50 % overlap, an MCRA noise estimate, power spectral subtraction
with an over-subtraction factor set from each frame's SNR, then a Wiener-style
gain. There is nothing to tune or sweep, and it does well on this clip:

```python
out = kudio.trad_enhance(noisy, sr)
print(len(noisy), len(out), out.dtype)
print(f"{kudio.si_sdr(clean, out):.2f} dB")
```

```text
80000 80000 float32
14.09 dB
```

The output is as long as the input. Its frames step by 256 samples, so a tail
shorter than that has no frame of its own and comes back silent: the last 128
samples here, 8 ms. A clip shorter than one 512-sample frame is refused with
`FeatureError`, since it would come back as silence.

[`wavelet_low_pass_filter`][kudio.enhance.traditional.wavelet_low_pass_filter]
soft-thresholds the detail coefficients of a full `db2` decomposition with one
threshold for every level, `thres` times the clip's peak, `max |y|`. The
fine levels go first, so it acts as a low-pass, and at the default `thres=0.8`
a heavy one: on this clip the voices themselves lost 7.6 dB below 1 kHz, and
SI-SDR fell from 4.98 to 4.00 dB (11.20 dB at `thres=0.1`). The output has
the input's length, and a silent clip comes back as it went in. For a plain
cutoff frequency, the
Butterworth filters in [Editing and effects](editing.md) are the tool.

## From the command line

`kudio enhance` runs
[`spectral_enhance`][kudio.enhance.spectral.spectral_enhance] on a file, and
`kudio compare` runs
[`compare_enhancers`][kudio.enhance.spectral.compare_enhancers], best first
when there is a reference:

```bash
kudio enhance in.wav out.wav --method omlsa --noise mcra --set q=0.4
kudio compare in.wav --reference clean.wav
kudio compare in.wav --reference clean.wav --noises all   # the full grid
```

[Command line](../cli.md) lists every option.

[^boll]: S. F. Boll, "Suppression of acoustic noise in speech using spectral subtraction", IEEE Trans. Acoust., Speech, Signal Process. 27(2), 1979.
[^berouti]: M. Berouti, R. Schwartz and J. Makhoul, "Enhancement of speech corrupted by acoustic noise", ICASSP 1979.
[^kamath]: S. Kamath and P. Loizou, "A multi-band spectral subtraction method for enhancing speech corrupted by colored noise", ICASSP 2002.
[^em84]: Y. Ephraim and D. Malah, "Speech enhancement using a minimum mean-square error short-time spectral amplitude estimator", IEEE Trans. Acoust., Speech, Signal Process. 32(6), 1984.
[^em85]: Y. Ephraim and D. Malah, "Speech enhancement using a minimum mean-square error log-spectral amplitude estimator", IEEE Trans. Acoust., Speech, Signal Process. 33(2), 1985.
[^omlsa]: I. Cohen and B. Berdugo, "Speech enhancement for non-stationary noise environments", Signal Processing 81(11), 2001.
[^mcra]: I. Cohen and B. Berdugo, "Noise estimation by minima controlled recursive averaging for robust speech enhancement", IEEE Signal Process. Lett. 9(1), 2002.
