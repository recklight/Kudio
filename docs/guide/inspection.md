# Inspecting a recording

A recording from the field has no clean copy to compare with. These functions
judge it on its own:

| Question | Use | Needs |
|---|---|---|
| Is it technically sound? | [`audio_report`][kudio.core.report.audio_report] | the core install |
| Does a hum or a whine stand out? | [`tonal_lines`][kudio.core.lines.tonal_lines] | the core install |
| How loud does it sound? | [`loudness`][kudio.core.loudness.loudness] and the functions beside it | the core install |
| How reverberant is it? | [`srmr`][kudio.core.srmr.srmr] | the core install |
| How far is the speech above its noise? | [`wada_snr`][kudio.core.wada.wada_snr] | the core install |
| What would a listener say? | [`dnsmos`][kudio.core.dnsmos.dnsmos] | `kudio[dnsmos]` and the DNS-Challenge weights |

The examples share one take: a 140 Hz voice switched on and off over hiss.

```python
import numpy as np
import kudio

sr = 16000
rng = np.random.default_rng(0)
t = np.arange(20 * sr) / sr
voice = sum(np.cos(2 * np.pi * 140 * k * t) / k for k in range(1, 55))
words = np.sin(2 * np.pi * 0.7 * t) > -0.3          # talk, pause, talk, ...
take = 0.05 * voice * words + 0.002 * rng.standard_normal(t.size)
```

## A health check with no reference

```python
report = kudio.audio_report(take, sr)
print(report)
print(report.problems())
```

```text
20.00s @ 16000 Hz · peak -12.6 dBFS · -29.2 LUFS · SNR~28 dB · bandwidth 8000 Hz
[]
```

[`audio_report`][kudio.core.report.audio_report] takes mono audio, in 30 ms
frames at a 15 ms hop, and [`problems()`][kudio.core.report.AudioReport.problems]
puts what is worth acting on into sentences; an empty list is good news. Check a
stereo file a channel at a time.

| Field | Measures | Flagged when |
|---|---|---|
| `peak_dbfs` | the loudest sample, in dBFS | over 0, over -0.5 or under -30 |
| `clipped_samples`, `clipped_ratio` | samples in a flat run at full scale, `clip_run` (3) long or more | any |
| `full_scale_samples` | samples at 0.999 of full scale or more, in a run or not | never |
| `dc_offset` | the mean sample value | size over 0.01 |
| `silence_ratio` | share of frames more than 40 dB below the 95th-percentile frame | over 0.6 |
| `noise_floor_dbfs` | level of the 10th-percentile frame | never |
| `estimated_snr_db` | 95th-percentile frame level minus the 10th | under 10 dB |
| `bandwidth_hz` | highest frequency within 50 dB of the strongest | never |
| `band_limited`, `source_rates`, `mirrored`, `band_edge_hz` | an edge with an empty band above it, or the mirror image linear interpolation leaves | when found |
| `sample_step`, `grid_bits` | the grid every sample sits on | coarser than 16 bits |
| `lufs`, `rms_dbfs` | integrated loudness (`nan` under 0.4 s) and plain RMS | never |
| `non_finite_samples` | samples that are NaN or inf | any, reported alone |

The SNR estimate reads the floor off the quietest tenth of the frames, so it
needs pauses that hold noise: without them it reads about 1 dB, which means
unknown, not bad. A clip with NaN or inf is measured, and the count is its only
finding. Passing everything is a technical verdict only;
[DNSMOS](#dnsmos-what-a-listener-would-say) judges how it sounds.

### Clipping and band limits

```python
hot = np.clip(8 * take, -1.0, 1.0)                         # written too hot
narrow = kudio.resample(kudio.resample(take, sr, 8000), 8000, sr)   # 8 kHz audio
for clip in (hot, narrow):
    for problem in kudio.audio_report(clip, sr).problems():
        print(problem)
```

```text
3321 samples clipped (1.04% of the clip) — the recording was already too hot; no processing gets that back
content stops at 3816 Hz, although a 16000 Hz file holds up to 8000 Hz — it was very likely upsampled from 8000 Hz (a codec that cuts there looks the same), and the band above is empty
```

A sample counts as clipped inside a run of `clip_run` (3) or more at 0.999 of
full scale; float audio past 1.0 is still intact and is reported as a peak.

![Average spectrum of the take and of its upsampled copy, with the two bands the report compares at 4 kHz](../images/inspection-bandwidth-light.webp#only-light)
![Average spectrum of the take and of its upsampled copy, with the two bands the report compares at 4 kHz](../images/inspection-bandwidth-dark.webp#only-dark)

An upsampled file has an edge with an empty band above it, 20 dB under the
band below the edge and 40 dB under it all the way to Nyquist; `source_rates`
names the rates it fits. Content less than 40 dB over the floor and an edge
above 0.9 of Nyquist are missed ([the
cases](../validation.md#upsampling-and-band-limits-in-audio_report)). Samples on
a grid coarser than 16 bits give `grid_bits` and a finding.

### Hum and other lines

[`tonal_lines`][kudio.core.lines.tonal_lines] finds the peaks of the averaged
spectrum that stand 20 dB or more above the running median around them, and
groups those on whole multiples of one frequency into a comb, as mains hum is:
on hiss with a 50 Hz hum and a 1 kHz whine it reports `6 lines, the
strongest at 49.9 Hz, 35 dB above its neighbourhood; 50.00 Hz comb, 5 lines
(harmonics 1-5)`. It says which frequencies stand out, not why; a held vowel is
a comb too.

## How loud it sounds

[`loudness`][kudio.core.loudness.loudness] gives gated integrated loudness in
LUFS, to ITU-R BS.1770-4:[^bs1770] K-weighting, the mean square over 400 ms
blocks every 100 ms, the -70 LUFS and relative -10 LU gates, then the average.
The gates keep pauses out: ten seconds of room tone after the take lower its RMS
from -29.2 to -30.9 dBFS and leave its loudness at -29.2 LUFS. Several channels
go in channels last, up to five, so load a stereo file with `mono=False`. A clip
shorter than one block raises a `FeatureError` that points at `block_size=`.

### Normalising and matching

```python
loud = kudio.normalize_lufs(take, sr, lufs=-14.0)
noisy = take + 0.02 * rng.standard_normal(t.size)
matched = kudio.match_loudness(kudio.spectral_enhance(noisy, sr), noisy, sr)
print(f"{kudio.loudness(loud, sr):.1f} LUFS, peak {np.max(np.abs(loud)):.2f}")
print(f"{kudio.loudness(noisy, sr):.1f} and {kudio.loudness(matched, sr):.1f} LUFS")
```

```text
-14.0 LUFS, peak 1.34
-27.5 and -27.5 LUFS
```

[`normalize_lufs`][kudio.core.loudness.normalize_lufs] applies one gain, to -23
LUFS by default (EBU R 128[^ebu]), and does not limit: samples past full scale
come back past 1.0 with a `UserWarning`. [`match_loudness`][kudio.core.loudness.match_loudness]
scales one clip to the loudness of another before an A/B comparison.

## Loudness over time

```python
drift = take.copy()
drift[8 * sr:13 * sr] *= 10 ** (-12 / 20)          # five seconds facing away
curve = kudio.loudness_over_time(drift, sr)
print(curve)
print(curve.spans_below(curve.max - 6, min_seconds=1.0))
print(f"{kudio.loudness_range(drift, sr):.1f} LU")
```

```text
-29.7 LUFS at 16.30s, -42.1 LUFS at 11.00s (12.4 LU apart)
[(9.1, 11.9)]
12.0 LU
```

![Momentary and short-term loudness of the take with its level dropped by 12 dB from 8 to 13 s](../images/inspection-loudness-light.webp#only-light)
![Momentary and short-term loudness of the take with its level dropped by 12 dB from 8 to 13 s](../images/inspection-loudness-dark.webp#only-dark)

[`loudness_over_time`][kudio.core.loudness.loudness_over_time] measures a 3 s
window, or 400 ms with [`MOMENTARY`][kudio.core.loudness.MOMENTARY], every 100 ms;
`times` are window centres and silence reads `-inf`.
[`loudness_range`][kudio.core.loudness.loudness_range] is EBU Tech 3342's figure.

## True peak

```python
tone = np.sin(2 * np.pi * (sr / 4) * np.arange(sr) / sr + np.pi / 4)  # samples at ±0.707
tricky = kudio.fade(tone, sr, 0.01, 0.01)
peak = 20 * np.log10(np.max(np.abs(tricky)))
print(f"samples {peak:.2f} dBFS, true peak {kudio.true_peak(tricky, sr):+.2f} dBTP")
```

```text
samples -3.01 dBFS, true peak +0.02 dBTP
```

[`true_peak`][kudio.core.loudness.true_peak] oversamples to at least 192 kHz
(BS.1770-4 Annex 2). EBU R 128 allows -1 dBTP, and `normalize_lufs` warns only
about samples, so measure the true peak after normalising.

## How reverberant: SRMR

```python
syllables = 0.5 + 0.5 * np.sin(2 * np.pi * 4 * t)      # four a second
talk = 0.05 * voice * words * syllables + 0.002 * rng.standard_normal(t.size)
for rt60 in (0.3, 0.6, 1.0):
    wet = kudio.apply_rir(talk, kudio.rir(sr, rt60=rt60, seed=0))
    print(f"RT60 {rt60:.1f} s  SRMR {kudio.srmr(wet, sr):.2f}")
```

```text
RT60 0.3 s  SRMR 10.08
RT60 0.6 s  SRMR 9.40
RT60 1.0 s  SRMR 8.11
```

[`srmr`][kudio.core.srmr.srmr] is the speech-to-reverberation modulation energy
ratio[^srmr], higher for drier speech, at 8 and 16 kHz. It has no fixed range,
so compare a take only with itself elsewhere; `norm=True` is the variant for
noise and reverberation together.

## How noisy: WADA-SNR

```python
speech = 0.05 * voice * words * syllables
noise = np.random.default_rng(2).standard_normal(t.size)
for snr_db in (0, 10, 20):
    noisy = kudio.add_noise_snr(speech, noise, snr_db=snr_db, seed=0)
    print(f"{snr_db:2d} dB  WADA-SNR {kudio.wada_snr(noisy, sr):6.2f}")
```

```text
 0 dB  WADA-SNR   0.55
10 dB  WADA-SNR  10.57
20 dB  WADA-SNR  25.48
```

[`wada_snr`][kudio.core.wada.wada_snr] reads the SNR off how peaked the samples
are.[^wada] Above 20 dB the speech decides more than the noise, so read high
values as "clean". A clip under 1 s, or with digitally silent pauses, gets nan.

## DNSMOS: what a listener would say

[`dnsmos`][kudio.core.dnsmos.dnsmos] runs Microsoft's DNSMOS P.835 model[^dnsmos]
as DNS-Challenge's `dnsmos_local.py` does, and returns SIG, BAK and OVRL on the
1 to 5 scale; `personalized=True` runs the personalised model,[^dns2023] which
also marks down a second talker, and [`dnsmos_p808`][kudio.core.dnsmos.dnsmos_p808]
the P.808 model[^p808], one overall score. Install `kudio[dnsmos]` and clone
[DNS-Challenge](https://github.com/microsoft/DNS-Challenge) for the weights:

```python
score = kudio.dnsmos(take, sr, model_dir="DNS-Challenge/DNSMOS")
print(score.summary())
```

The audio is scored at 16 kHz in 9.01 s windows at a 1 s hop, a shorter clip
repeated until it fills one; `session=` reuses an opened model over a folder.
`dnsmos_p808` follows the script's pinned librosa 0.8.1, `librosa="0.9"` the
later spectrogram ([How the metrics are checked](../validation.md#dnsmos)).

## From the command line

`kudio report take.wav` exits 1 when it finds a problem, `kudio loudness
take.wav --target -23` when the file misses the target; `kudio score` gives a
test set's mean SRMR, WADA-SNR or DNSMOS ([Command line](../cli.md#score)).

[^srmr]: T. H. Falk, C. Zheng and W.-Y. Chan, "A non-intrusive quality and intelligibility measure of reverberant and dereverberated speech", IEEE Trans. Audio, Speech, Lang. Process. 18(7), pp. 1766-1774, 2010, doi:10.1109/TASL.2010.2052247. The normalised variant: J. F. Santos, M. Senoussaoui and T. H. Falk, "An improved non-intrusive intelligibility metric for noisy and reverberant speech", Proc. IWAENC 2014, pp. 55-59, doi:10.1109/IWAENC.2014.6953337.
[^bs1770]: Recommendation ITU-R BS.1770-4 (10/2015), "Algorithms to measure audio programme loudness and true-peak audio level". Annex 1 is the loudness measurement and Annex 2 the true-peak meter.
[^ebu]: EBU R 128 (2023), "Loudness normalisation and permitted maximum level of audio signals"; EBU Tech 3341 (November 2023), "Loudness metering: 'EBU Mode' metering to supplement EBU R 128 loudness normalization"; EBU Tech 3342 (November 2023), "Loudness range: a measure to supplement EBU R 128 loudness normalization".
[^dnsmos]: C. K. A. Reddy, V. Gopal and R. Cutler, "DNSMOS P.835: A non-intrusive perceptual objective speech quality metric to evaluate noise suppressors", Proc. ICASSP 2022, pp. 886-890, doi:10.1109/ICASSP43922.2022.9746108. It was trained on ratings collected to ITU-T Recommendation P.835 (11/2003), "Subjective test methodology for evaluating speech communication systems that include noise suppression algorithm".
[^wada]: C. Kim and R. M. Stern, "Robust signal-to-noise ratio estimation based on waveform amplitude distribution analysis", Proc. Interspeech 2008, pp. 2598-2601, doi:10.21437/Interspeech.2008-644.
[^p808]: C. K. A. Reddy, V. Gopal and R. Cutler, "DNSMOS: A non-intrusive perceptual objective speech quality metric to evaluate noise suppressors", Proc. ICASSP 2021, pp. 6493-6497, doi:10.1109/ICASSP39728.2021.9414878. It was trained on ratings collected with the crowdsourcing method of ITU-T Recommendation P.808 (06/2018), "Subjective evaluation of speech quality with a crowdsourcing approach".
[^dns2023]: H. Dubey, A. Aazami, V. Gopal, B. Naderi, S. Braun, R. Cutler, A. Ju, M. Zohourian, M. Tang, M. Golestaneh and R. Aichner, "ICASSP 2023 Deep Noise Suppression Challenge", IEEE Open Journal of Signal Processing 5, pp. 725-737, 2024, doi:10.1109/OJSP.2024.3378602; first posted as arXiv:2303.11510, 2023.
