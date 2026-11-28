# Denoising

`kudio.enhance` is single-channel statistical denoising: no model, nothing to
download. It estimates the noise, works out how much speech each time-frequency
bin holds, and turns each bin down accordingly. Seven gain rules run over any of
four noise estimators, and four of the rules also run on a live stream.

| You have | Use |
|---|---|
| one noisy clip | [`spectral_enhance`][kudio.enhance.spectral.spectral_enhance] |
| a clip, and the question of which method suits it | [`compare_enhancers`][kudio.enhance.spectral.compare_enhancers] |
| audio that is still arriving | [`StreamEnhancer`][kudio.enhance.streaming.StreamEnhancer] |
| a folder of recordings | [`enhance_folder`][kudio.enhance.spectral.enhance_folder] |

??? example "The clip the examples run on"

    Five seconds at 16 kHz: two harmonic "voices" with pauses before, between
    and after them, under white noise at 5 dB SNR over the whole clip.

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
print(f"SI-SDR {kudio.si_sdr(clean, noisy):.2f} dB -> {kudio.si_sdr(clean, enhanced):.2f} dB")
```

```text
SI-SDR 4.98 dB -> 12.21 dB
```

![Spectrograms of the noisy clip and the log-MMSE output from 0 to 4 kHz. After the first two seconds the noise in the output is about 19 dB lower and the harmonics are intact; before that, horizontal streaks of residual noise remain.](../images/enhancement-before-after-light.webp#only-light)
![Spectrograms of the noisy clip and the log-MMSE output from 0 to 4 kHz. After the first two seconds the noise in the output is about 19 dB lower and the harmonics are intact; before that, horizontal streaks of residual noise remain.](../images/enhancement-before-after-dark.webp#only-dark)

The output is float32 and as long as the input, in step with it sample for
sample. The audio has to be mono, and a clip shorter than one frame (`n_fft`,
512) is refused. The method's own parameters are keywords; one it does not take
raises a [`FeatureError`][kudio.exceptions.FeatureError] that lists the ones it
does. `hop_length=` is a quarter of `n_fft` by default; a hop with too little
window overlap is refused.

## Gain rules and noise estimators

Every rule turns a noise estimate $\lambda_d(k,l)$ into a gain between a floor
and 1 on the complex spectrum, leaving the noisy phase. `wiener`, `mmse_stsa`,
`logmmse` and `omlsa` go through the decision-directed a priori SNR $\xi$ of
Ephraim and Malah[^em84], with $\alpha = 0.98$, $\gamma$ clamped at 40 dB and
$\xi$ floored at −25 dB.

| Method | What it does | Its own parameters |
|---|---|---|
| `specsub` | power spectral subtraction[^boll] with a fixed over-subtraction[^berouti] and a floor | `over_subtraction` 2.0, `spectral_floor` 0.002 |
| `multiband` | subtraction with a factor per band, from that band's SNR[^kamath] | `bands` 6, `spectral_floor` 0.002 |
| `wiener` | gain $\xi/(1+\xi)$ | `alpha` 0.98 |
| `mmse_stsa` | MMSE estimate of the amplitude[^em84] | `alpha` 0.98 |
| `logmmse` | MMSE in the log-amplitude domain[^em85]; the default | `alpha` 0.98 |
| `omlsa` | log-MMSE weighted by a speech-presence probability[^omlsa] | `alpha` 0.98, `q` 0.30 |
| `spectral_gate` | a smoothed threshold mask above the noise estimate's spread | `n_std_thresh` 1.5, `prop_decrease` 1.0, `freq_smooth_hz` 500, `time_smooth_ms` 50 |

All but `spectral_gate` take `gain_floor_db`, −25 dB: how far a bin may be
turned down. [`METHODS`][kudio.enhance.spectral.METHODS] holds every parameter's
default, range, step, unit and a sentence of help, for a GUI to build controls
from.

| Estimator (`noise=`) | Assumes | `noise_kwargs` | Streams |
|---|---|---|---|
| `mcra` (default) | a running average of each bin, held where the bin stands 5 times above its running minimum[^mcra] | `window=100`, `delta=5` | yes |
| `quantile` | a low quantile of each bin over the file, scaled up to the mean of steady noise | `quantile=0.15` | no |
| `initial` | the opening frames are noise alone | `n_frames=6` | yes |
| `minimum` | a sliding minimum of the smoothed spectrum, times a bias | `window=60`, `bias=1.5`, `smoothing=0.85` | yes |

```python
for noise in kudio.NOISE_ESTIMATORS:
    out = kudio.spectral_enhance(noisy, sr, "logmmse", noise=noise)
    print(f"{noise:9} {kudio.si_sdr(clean, out):.2f} dB")
```

```text
mcra      12.21 dB
quantile  14.16 dB
initial   14.86 dB
minimum   6.86 dB
```

`initial` wins here because the clip opens on half a second of noise, its
assumption exactly; [`NOISE_ESTIMATORS`][kudio.enhance.spectral.NOISE_ESTIMATORS]
has a sentence on each. `mcra` starts low and needs about 1.6 s to settle, the
streaks in the figure. [`estimate_noise_psd`][kudio.enhance.spectral.estimate_noise_psd],
[`make_noise_tracker`][kudio.enhance.spectral.make_noise_tracker],
[`DecisionDirected`][kudio.enhance.spectral.DecisionDirected] and
[`make_gain_rule`][kudio.enhance.spectral.make_gain_rule] are the pieces, for a
loop of your own.

## Choosing a method

```python
results = kudio.compare_enhancers(noisy, sr, reference=clean,
                                  noises=list(kudio.NOISE_ESTIMATORS), metrics=False)
ranked = sorted((r for r in results if not r.error), key=lambda r: r.si_sdr_db, reverse=True)
print(len(ranked))
for r in ranked[:3]:
    print(f"{r.label:20} SI-SDR {r.si_sdr_db:+.2f} dB  dSNR {r.delta_snr_db:+.2f} dB")
```

```text
28
wiener + initial     SI-SDR +15.37 dB  dSNR +10.48 dB
omlsa + initial      SI-SDR +15.26 dB  dSNR +10.36 dB
logmmse + initial    SI-SDR +14.86 dB  dSNR +10.00 dB
```

[`compare_enhancers`][kudio.enhance.spectral.compare_enhancers] runs every
(method, estimator) pair asked for and returns an
[`EnhanceResult`][kudio.enhance.spectral.EnhanceResult] each: time, noise floor
and loudness, and with `reference=` SNR, SI-SDR and `delta_snr_db`, the SNR
gained over the input. `metrics=True`, the default, adds STOI and, with
`kudio[pesq]`, PESQ. A run that fails comes back with `error` set. Other
keywords go to the runs that take them; one that no run takes is refused before
anything runs. Without a reference you get the floor and loudness only, and the
lowest floor is not the best result.

A function of `(y, sr)`, bare or as `(name, function)`, sits in the same table
with the estimator column `own`
([`CustomEnhancer`][kudio.enhance.spectral.CustomEnhancer]), so a model trained
with [KudioEnhance](https://github.com/recklight/KudioEnhance) is ranked
against these.

## Where it does not help

`delta_snr_db` over MCRA, with white noise mixed into the clean voices at each
input SNR:

| Input SNR | Best rule | Worst of the other six | `spectral_gate` |
|---|---|---|---|
| −5 dB | +10.5 | +6.9 | +6.5 |
| 5 dB | +7.6 | +6.0 | +0.4 |
| 15 dB | +3.1 | +0.5 | −9.9 |
| 20 dB | −0.1 | −4.0 | −15.0 |

These methods are for noisy audio: at 20 dB every rule leaves the clip worse
than it found it. They also need to hear the noise on its own. Without the
pauses `logmmse` over `mcra` gains +1.3 dB instead of +7.6; a second of room tone
in front, with `noise="initial"`, brings it to +8.7.

## Audio that is still arriving

```python
enhancer = kudio.StreamEnhancer(sr, "logmmse")
pieces = [enhancer.process(noisy[i:i + 1024])       # one 64 ms block at a time
          for i in range(0, len(noisy), 1024)]
pieces.append(enhancer.flush())
streamed = np.concatenate(pieces)
print(f"latency {enhancer.latency_seconds * 1000:.0f} ms, {len(noisy)} in, {len(streamed)} out,",
      f"SI-SDR {kudio.si_sdr(clean, streamed):.2f} dB")
```

```text
latency 32 ms, 80000 in, 80000 out, SI-SDR 13.45 dB
```

[`StreamEnhancer`][kudio.enhance.streaming.StreamEnhancer] runs the same update
a block of any size at a time. The latency is one frame, `n_fft / sr`, a wait
rather than a shift, so sample *i* of the output lines up with sample *i* of the
input; [`flush`][kudio.enhance.streaming.StreamEnhancer.flush] processes the
last partial frame, and [`reset`][kudio.enhance.streaming.StreamEnhancer.reset]
clears everything for the next take. The four rules in
[`STREAMABLE_METHODS`][kudio.enhance.streaming.STREAMABLE_METHODS] stream over
`mcra`, `initial` or `minimum`; the others raise a `FeatureError` that gives the
offline call. Feed it [`StreamRecorder.drain()`][kudio.core.stream.StreamRecorder.drain],
never `tail()`, and keep it mono ([Recording and live views](live-audio.md)).

## A whole folder

```python
hiss = np.random.default_rng(1).standard_normal(len(clean))
for name, snr_db in [("day1/take1.wav", 0), ("day1/take2.wav", 5), ("day2/take1.wav", 10)]:
    kudio.save_wave(f"noisy/{name}", kudio.add_noise_snr(clean, hiss, snr_db=snr_db, seed=0), sr)
result = kudio.enhance_folder("noisy", "enhanced", "logmmse", report=True)
print(result)
for path, before, after in result.floors:
    print(f"{path.as_posix()}: floor {before:.1f} -> {after:.1f} dBFS")
```

```text
3/3 written, 0 failed
enhanced/day1/take1.wav: floor -24.4 -> -43.5 dBFS
enhanced/day1/take2.wav: floor -29.4 -> -48.4 dBFS
enhanced/day2/take1.wav: floor -34.4 -> -53.3 dBFS
```

[`enhance_folder`][kudio.enhance.spectral.enhance_folder] mirrors the subfolders
as [`convert_folder`][kudio.core.io.convert_folder] does and returns an
[`EnhanceFolderResult`][kudio.enhance.spectral.EnhanceFolderResult], false when
anything failed, with the floors `report=True` measures. The method and its
parameters are checked once, before the first file. Files are loaded mono and
written as 16-bit PCM; a taken output name gets a free one unless
`overwrite=True`.

## Two fixed recipes

[`trad_enhance`][kudio.enhance.traditional.trad_enhance] is one recipe from
Loizou's *Speech Enhancement: Theory and Practice*: 512-sample Hamming frames at
50 % overlap, an MCRA noise estimate, power spectral subtraction with an
SNR-dependent factor, then a Wiener-style gain, with nothing to tune; it scores
14.09 dB SI-SDR on this clip. A clip shorter than one frame is refused.
[`wavelet_low_pass_filter`][kudio.enhance.traditional.wavelet_low_pass_filter]
soft-thresholds the detail coefficients of a `db2` decomposition at `thres`
times the peak, which acts as a heavy low-pass at the default 0.8.

`kudio enhance in.wav out.wav --method omlsa --noise mcra --set q=0.4` and
`kudio compare in.wav --reference clean.wav --noises all` do the same from a
shell ([Command line](../cli.md)).

[^boll]: S. F. Boll, "Suppression of acoustic noise in speech using spectral subtraction", IEEE Trans. Acoust., Speech, Signal Process. 27(2), pp. 113-120, 1979, doi:10.1109/TASSP.1979.1163209.
[^berouti]: M. Berouti, R. Schwartz and J. Makhoul, "Enhancement of speech corrupted by acoustic noise", Proc. ICASSP 1979, vol. 4, pp. 208-211, doi:10.1109/ICASSP.1979.1170788.
[^kamath]: S. Kamath and P. Loizou, "A multi-band spectral subtraction method for enhancing speech corrupted by colored noise", Proc. ICASSP 2002, p. IV-4164, doi:10.1109/ICASSP.2002.5745591. The proceedings print the abstract only.
[^em84]: Y. Ephraim and D. Malah, "Speech enhancement using a minimum mean-square error short-time spectral amplitude estimator", IEEE Trans. Acoust., Speech, Signal Process. 32(6), pp. 1109-1121, 1984, doi:10.1109/TASSP.1984.1164453.
[^em85]: Y. Ephraim and D. Malah, "Speech enhancement using a minimum mean-square error log-spectral amplitude estimator", IEEE Trans. Acoust., Speech, Signal Process. 33(2), pp. 443-445, 1985, doi:10.1109/TASSP.1985.1164550.
[^omlsa]: I. Cohen and B. Berdugo, "Speech enhancement for non-stationary noise environments", Signal Processing 81(11), pp. 2403-2418, 2001, doi:10.1016/S0165-1684(01)00128-1.
[^mcra]: I. Cohen and B. Berdugo, "Noise estimation by minima controlled recursive averaging for robust speech enhancement", IEEE Signal Process. Lett. 9(1), pp. 12-15, 2002, doi:10.1109/97.988717.
