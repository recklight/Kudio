# Measuring quality

Every metric here scores a processed recording against the clean recording it
came from; without one, use [Inspecting a recording](inspection.md). SNR, SI-SDR,
segmental SNR and SDI compare sample *i* with sample *i*, so one sample of delay
takes a perfect copy down to about +12 dB of SI-SDR, and nothing in the score
says a delay is why. Line the pair up before choosing a metric.

## A first score

The examples share one clip: two phrases of a harmonic tone over room tone,
mixed with white noise at 5 dB SNR and denoised with log-MMSE.

```python
import numpy as np
import kudio

sr = 16000
rng = np.random.default_rng(0)

def burst(seconds, f0):
    """A harmonic tone that swells four times a second, like syllables."""
    t = np.arange(int(seconds * sr)) / sr
    tone = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 12))
    return tone * (0.5 + 0.5 * np.sin(2 * np.pi * 4 * t)) / 6

clean = 0.01 * rng.standard_normal(4 * sr)       # room tone
clean[8000:24000] += burst(1.0, 120)             # two phrases with
clean[32000:52800] += burst(1.3, 150)            # a pause between
clean = clean.astype(np.float32)

noise = rng.standard_normal(4 * sr).astype(np.float32)
noisy = kudio.add_noise_snr(clean, noise, snr_db=5, seed=0)
enhanced = kudio.spectral_enhance(noisy, sr)     # log-MMSE

for name, value in kudio.score(clean, enhanced, sr).items():
    print(f"{name:7s} {value:7.3f}")
```

```text
snr      12.002
si_sdr   11.725
segsnr    5.120
sdi       0.063
stoi      0.601
estoi     0.276
sdr      13.325
```

[`score`][kudio.core.evaluator.score] takes the reference first, the signal
being judged second, and returns a dict. With no list it runs `snr`, `si_sdr`,
`segsnr`, `sdi`, `stoi`, `estoi` and `sdr`, and `pesq` where `kudio[pesq]` is
installed and the rate is 8 or 16 kHz; the measures further down run when they
are named. Each is also a function of its own with the same argument order. An
unknown name raises `ValueError`, a missing package a
[`DependencyError`][kudio.exceptions.DependencyError] naming the extra.
[`check_metrics_install`][kudio.core.evaluator.check_metrics_install] returns
`(pesq_ok, stoi_ok, sdr_ok)`; [`eval_metrics`][kudio.core.evaluator.eval_metrics]
takes the rate first and always returns `(pesq, stoi, sdr)`, `None` where one
cannot run.

## Line the pair up first

![SI-SDR against delay: a perfect copy one sample late scores +11.9 dB, and from four samples on both the copy and the log-MMSE output score below the unprocessed noisy input](../images/metrics-delay-light.webp#only-light)
![SI-SDR against delay: a perfect copy one sample late scores +11.9 dB, and from four samples on both the copy and the log-MMSE output score below the unprocessed noisy input](../images/metrics-delay-dark.webp#only-dark)

Four samples late, the copy and the denoised take both score below the
unprocessed noisy input. Filters, resamplers, models that pad their frames and
re-recordings all add delay, and a file trimmed to its input's length gives no
warning.

### Finding the delay

```python
processed = np.concatenate([np.zeros(128, np.float32), enhanced])  # 8 ms late
found = kudio.find_delay(clean, processed, sr)
print(found)
a, b = kudio.align(clean, processed, sr)
print(len(a), len(b), round(kudio.si_sdr(a, b), 2))
print(kudio.find_delay(clean, noise, sr))
```

```text
lags by 128 samples (8.0 ms), correlation 0.968
64000 64000 11.72
leads by 5873 samples (367.1 ms), correlation 0.016, polarity inverted
```

[`find_delay`][kudio.core.align.find_delay] measures the delay to the nearest
sample by cross-correlation, normalised over the overlap; a positive `delay`
means the second signal starts later. [`align`][kudio.core.align.align] returns
the pair trimmed to what they share, and
[`Alignment.apply`][kudio.core.align.Alignment.apply] trims a later pair at a
delay already found. Cross-correlation always has a maximum, so two unrelated
signals get a delay too: `align` refuses below
[`MIN_CORRELATION`][kudio.core.align.MIN_CORRELATION] (0.5) with a
`FeatureError`. Bound the search with `max_seconds=` when the delay is known to
be small. An inverted copy is reported as `polarity inverted`, logged by `align`
and left as it is. Both take mono audio at the one `sr` you pass. `kudio align
clean.wav processed.wav --metrics` does this for a pair of files.

## When the lengths differ

```python
import warnings

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    kudio.snr(clean, clean[:60000])
print(caught[0].message)
```

```text
snr(): ref is 64000 samples and deg 60000: only the first 60000 samples are scored and the rest is ignored. Use kudio.align if the short one is delayed rather than just short, or trim the pair yourself to say which part you meant.
```

`snr`, `si_sdr`, `segmental_snr`, `sdi`, and `score` for everything but PESQ,
cut both signals to the length they share and warn. Once more than half of the
longer one would go unmeasured,
[`MAX_LENGTH_MISMATCH`][kudio.core.evaluator.MAX_LENGTH_MISMATCH] (0.5), the pair
is refused with a `ValueError`; set `kudio.core.evaluator.MAX_LENGTH_MISMATCH`
to score a lopsided pair anyway. PESQ gets both signals whole, the way the ITU
program reads two files, since it finds the delay itself.
[`AudioEvaluate`][kudio.core.evaluator.AudioEvaluate] checks every pair the same
way.

## What each metric sees

```python
def late(y, n):
    return np.concatenate([np.zeros(n, np.float32), y])[:len(y)]

cases = {"noise, 5 dB SNR": noisy, "half the gain": 0.5 * clean,
         "polarity flipped": -clean, "low-pass, 2 kHz": kudio.lowpass(clean, sr, cutoff=2000),
         "1 sample late": late(clean, 1), "1 ms late": late(clean, 16)}
names = ["snr", "si_sdr", "segsnr", "sdi", "stoi", "sdr"]
print(f"{'':17}" + "".join(f"{n:>8}" for n in names))
for case, deg in cases.items():
    s = kudio.score(clean, deg, sr, names)
    print(f"{case:17}" + "".join(f"{s[n]:8.2f}" for n in names))
```

```text
                      snr  si_sdr  segsnr     sdi    stoi     sdr
noise, 5 dB SNR      5.00    4.97   -2.11    0.32    0.61    5.01
half the gain        6.02  138.88    6.02    0.25    1.00  283.00
polarity flipped    -6.02  144.90   -6.02    4.00    1.00  283.00
low-pass, 2 kHz     18.24   18.17   10.31    0.02    0.93   24.95
1 sample late       12.09   11.89    4.58    0.06    1.00   91.30
1 ms late           -1.45   -9.97   -2.17    1.40    0.99   53.57
```

Above 100 dB means the metric found nothing wrong; the figure comes from a
1e-12 guard or round-off. [`snr`][kudio.core.evaluator.snr] charges for
everything, level included. [`si_sdr`][kudio.core.evaluator.si_sdr] removes the
mean and the best-fitting gain,[^sisdr] so level and polarity cost nothing while
filtering and delay do: the first number to read for a denoiser.

- `segsnr`, [`segmental_snr`][kudio.core.evaluator.segmental_snr]: SNR per frame,
  held to [-10, 35] dB and averaged; 512-sample frames at a hop of 256 through
  `score`. Pauses count as much as speech, and a silent reference frame scores
  -10 dB with a warning.
- [`sdi`][kudio.core.evaluator.sdi]: `sum((ref - deg)**2) / sum(ref**2)`, 0 for a
  perfect copy, on the whole output, so not the speech-distortion index of the
  literature,[^sdi] which leaves the noise out.
- `stoi`, `estoi`: short-time objective intelligibility[^stoi] and its extended
  version, means of correlations from -1 to 1, a little under 0 where nothing of the speech is left;
  gain, polarity and 1 ms of delay barely move them. Under about 0.4 s of speech: nan, with a warning.
- `sdr`: BSS Eval's SDR,[^bss] which forgives the best 512-tap causal filter from
  reference to output; compare it only with SDR computed the same way.

Metrics disagree about one denoiser: log-MMSE gains 6.8 dB of SI-SDR here while
STOI stays put and ESTOI drops.

## PESQ: say which one

PESQ comes in two standards and two scales.[^p862] P.862 scores narrowband
speech, raw on -0.5 to 4.5 or mapped to MOS-LQO through P.862.1; P.862.2, the
wideband extension, gives a MOS-LQO and runs at 16 kHz.
[`pesq`][kudio.core.evaluator.pesq] takes the bandwidth as `mode=` and the scale
as `scale=`, and `score`, `eval_metrics` and `AudioEvaluate` pass them on as
`pesq_mode=` and `pesq_scale=`. It needs `kudio[pesq]`:

| `mode` | `scale` | at 8 kHz | at 16 kHz |
|---|---|---|---|
| `'auto'` (default) | `'lqo'` (default) | P.862, MOS-LQO | P.862.2, MOS-LQO |
| `'nb'` | `'lqo'` | P.862, MOS-LQO | P.862, MOS-LQO |
| `'wb'` | `'lqo'` | refused | P.862.2, MOS-LQO |
| `'auto'` or `'nb'` | `'raw'` | P.862, raw | P.862, raw |
| `'wb'` | `'raw'` | refused | refused |

```python
kudio.pesq(clean, enhanced, 16000)                          # P.862.2 MOS-LQO
kudio.pesq(clean, enhanced, 16000, mode="nb", scale="raw")  # raw P.862
```

Any other rate raises `ValueError`. Raw 2.0 is a narrowband MOS-LQO of 1.63, so
pass both arguments and name the scale in a table. The `pesq` package's wideband
code is the 2005 reference; ITU's 2018 correction, in Torcoli et al.'s
`pesqc2`,[^torcoli] scores wideband 0.73 higher on average.

## Hu and Loizou's measures

```python
names = ["llr", "is", "cep", "wss", "fwsegsnr"]
for label, deg in (("noisy", noisy), ("enhanced", enhanced)):
    s = kudio.score(clean, deg, sr, names)
    print(f"{label:8s}", "  ".join(f"{n} {s[n]:.3f}" for n in names))
```

```text
noisy    llr 0.388  is 1.648  cep 3.228  wss 23.960  fwsegsnr 9.678
enhanced llr 0.283  is 2.324  cep 3.293  wss 35.328  fwsegsnr 11.118
```

The log-likelihood ratio, Itakura-Saito distance, cepstral distance, weighted
spectral slope and frequency-weighted segmental SNR,[^hl08] as Loizou's MATLAB
code computes them: [`llr`][kudio.core.quality.llr],
[`itakura_saito`][kudio.core.quality.itakura_saito],
[`cepstral_distance`][kudio.core.quality.cepstral_distance],
[`wss`][kudio.core.quality.wss] and
[`fw_segmental_snr`][kudio.core.quality.fw_segmental_snr]. They run at 8 and
16 kHz on 30 ms frames, so line the pair up first. Lower is better for the first
four, higher for fwSegSNR. LLR, IS, CEP and WSS average the best 95% of frames;
WSS's 25 bands stop at 3.8 kHz at both rates and read every band under -100 dB
as -100 dB, so bring a quiet pair to a normal level first (`wss` warns).

### CSIG, CBAK and COVL

```python
c = kudio.composite(clean, enhanced, sr, pesq=1.88)   # a PESQ score you have
print(f"CSIG {c.csig:.3f}  CBAK {c.cbak:.3f}  COVL {c.covl:.3f}  LLR {c.llr:.3f}  WSS {c.wss:.3f}")
```

```text
CSIG 3.617  CBAK 2.595  COVL 2.715  LLR 0.283  WSS 35.328
```

[`composite`][kudio.core.quality.composite] predicts the three P.835[^p835]
ratings on the 1 to 5 scale from PESQ, LLR, WSS and a segmental SNR. It computes
PESQ with `kudio[pesq]`, or takes your own as `pesq=`; through `score` they are
`csig`, `cbak` and `covl`. The fit was made on raw narrowband PESQ, while most
published 16 kHz figures have wideband MOS-LQO inside, which `composite` uses by
default at 16 kHz; `mode="nb", scale="raw"` gives the other, and the two differ
by about 0.8 on COVL, so name the PESQ variant in a table. The
[`CompositeScore`][kudio.core.quality.CompositeScore] carries its four inputs.

## Log-spectral and mel-cepstral distance

```python
low = kudio.lowpass(clean, sr, cutoff=2000)
print(f"LSD {kudio.lsd(clean, low, sr):.2f} ln, {kudio.lsd(clean, low, sr, scale='log10'):.2f} log10")
for label, deg in (("noisy", noisy), ("enhanced", enhanced)):
    print(f"{label:8s} MCD {kudio.mcd(clean, deg, sr):.2f} dB")
```

```text
LSD 18.00 ln, 7.82 log10
noisy    MCD 6.85 dB
enhanced MCD 5.40 dB
```

[`lsd`][kudio.core.evaluator.lsd] is the RMS over frequency of the log power
ratio, averaged over frames, 0 for the same spectrum; it sees a missing band the
SNR family averages away. The defaults give the URGENT 2025 challenge's
number:[^urgent] natural log, 32 ms frames every 16 ms, the output scaled by its
least-squares gain; `scale="log10", n_fft=2048, hop_length=441,
match_gain=False, eps=1e-12` at 44.1 kHz gives NVSR's `ssr_eval`.[^nvsr] The
scales `ln`, `log10` and `db` differ by a constant factor, so find out which a
published number is on.

[`mcd`][kudio.core.melcepstrum.mcd] is the mel-cepstral distortion in dB after
dynamic time warping pairs the frames, as URGENT 2025's script computes it:
SPTK's analysis of the order its table gives for the rate, c0 included, FastDTW
with radius 1, gain matched. The warping lets it score pairs that differ in
timing; pass `match_gain=False` there. It has settings for 8, 16, 22.05, 24, 32,
44.1 and 48 kHz; at any other rate pass `order` and `alpha`. Implementations
differ, so compare MCDs from one code only.

## Intelligibility: NCM, CSII and SIIB

```python
for label, deg in (("noisy", noisy), ("enhanced", enhanced)):
    high, mid, low = kudio.csii(clean, deg, sr)
    print(f"{label:8s} NCM {kudio.ncm(clean, deg, sr):.3f}  "
          f"CSII high {high:.3f}  mid {mid:.3f}  low {low:.3f}")
```

```text
noisy    NCM 0.574  CSII high 0.661  mid 0.473  low 0.141
enhanced NCM 0.490  CSII high 0.660  mid 0.436  low 0.116
```

[`ncm`][kudio.core.intelligibility.ncm] and [`csii`][kudio.core.intelligibility.csii]
estimate, 0 to 1, how much of the speech a listener could follow, as Loizou's
NCM.m and CSII.m compute them.[^mhl09] NCM correlates the envelopes of the two
signals in 20 bands, at 8 and 16 kHz only. CSII, the coherence SII,[^ka05]
measures the coherent share in critical bands up to 3.7 kHz over the reference's
loud, middle and quiet frames (`csii_high`, `csii_mid`, `csii_low` in `score`).
Neither hears level, both need the pair lined up, short clips read high, and a
DC offset on the reference credits a silent output.

[`siib`][kudio.core.siib.siib] and [`siib_gauss`][kudio.core.siib.siib_gauss]
score the information clean speech and output share, in bits per
second,[^vkh18a] with the Kraskov estimator[^ksg04] or, 25 times faster, a
Gaussian channel,[^vkh18b] as SIIB.m and SIIB_Gauss.m do. A copy scores the
ceiling, 1335.76 bit/s. They need 5.45 s of speech at the least (20 s by the
authors' advice), hear level, need the pair lined up, and take other rates to
16 kHz first, so give the rate with the score.

## Several sources: SI-SDRi, PIT and BSS Eval

```python
other = 0.01 * rng.standard_normal(4 * sr)       # a second talker, higher
other[16000:48000] += burst(2.0, 210)
sources = np.stack([clean, other])
mixture = sources.sum(axis=0)
hiss = 0.005 * rng.standard_normal((2, 4 * sr))
separated = np.stack([other + 0.3 * clean,       # in the wrong order, each
                      clean + 0.1 * other]) + hiss   # with some of the other
s = kudio.pit_si_sdr(sources, separated, mixture)
b = kudio.bss_eval(sources, separated, mixture, permutation=s.permutation)
print("matched:", s.permutation)
for label, values in (("SI-SDRi", s.si_sdri), ("SDR", b.sdr), ("SIR", b.sir), ("SAR", b.sar)):
    print(f"{label:8s}", "  ".join(f"{v:6.2f}" for v in values))
print(f"{kudio.si_sdr_improvement(clean, enhanced, noisy):.2f} dB SI-SDRi for the denoiser")
```

```text
matched: (1, 0)
SI-SDRi   17.83   10.13
SDR       18.63    9.68
SIR       20.67    9.92
SAR       22.93   22.76
6.76 dB SI-SDRi for the denoiser
```

[`si_sdr_improvement`][kudio.core.separation.si_sdr_improvement] and
[`sdr_improvement`][kudio.core.separation.sdr_improvement] give what an output
gained over its input.[^convtasnet] [`pit_si_sdr`][kudio.core.separation.pit_si_sdr]
matches outputs to sources by the highest mean SI-SDR, the criterion of
utterance-level permutation invariant training,[^upit] and
[`bss_eval`][kudio.core.separation.bss_eval] gives SDR, SIR and SAR;[^bss] by
itself it matches by the highest mean SIR. Every figure is in the order of the
sources; papers report the mean over them. SI-SDR takes both signals to zero
mean first, as Conv-TasNet's SI-SNR does.

## Input a metric cannot score

Every metric follows one rule. Malformed input raises `ValueError` with a message
that starts with the function's name; well-formed input that leaves nothing to
score gives nan and one `UserWarning`, `<function>(): <reason>; scored nan`.
Signals are one channel of real numbers, as a vector, a column `(n, 1)` or a row
`(1, n)`, integers read on their type's full scale; int64 and uint64, complex,
boolean and text arrays and masked samples are refused. A rate has to be one
positive number.

| Metric | `ValueError` for | nan and a warning for |
|---|---|---|
| `snr`, `si_sdr`, `segmental_snr`, `sdi` | NaN or inf samples, two channels; bad frame settings | an empty pair, a silent reference (for `si_sdr`, any constant one), a pair too loud to square; for `segmental_snr`, a pair shorter than one frame |
| `pesq` | NaN or inf, two channels, a rate other than 8 or 16 kHz, a mode and scale PESQ cannot give | an empty side, a silent reference, a side under 0.25 s, no utterance found, a silent output |
| `lsd` | NaN or inf, two channels, a bad rate | an empty pair, a pair shorter than one frame, a silent reference |
| `llr`, `itakura_saito`, `cepstral_distance`, `wss`, `fw_segmental_snr`, `composite` | NaN or inf, two channels, a rate other than 8 or 16 kHz | a pair under 37.5 ms, a silent reference |
| `ncm`, `csii` | NaN or inf, two channels, a rate `ncm` or `csii` does not take | an empty or too short pair, a silent reference |
| `siib`, `siib_gauss` | NaN or inf, two channels, a rate under 4000 Hz or not whole | under 436 frames of speech, a silent or constant reference |
| `stoi`, `estoi`, `sdr` | two channels | an empty pair, a silent side, too little speech (STOI) |
| [`wer`, `cer`, `eer`, the F0 measures](task-metrics.md) | sides that do not pair up, bad labels or values | nothing to score |

An output that is digital silence is scored, not refused, since dropping out is
a defect the score should show.
[How the metrics are checked](../validation.md#the-input-rule-and-the-import-without-optional-packages)
has how the rule is tested.

## A whole test set

[`score_dataset`][kudio.core.dataset_score.score_dataset], or
[`kudio score`](../cli.md#score) from a shell, pairs the files by id, runs any of
`score`'s metrics and the ones that need no reference, and gives each mean with
its confidence interval, the number of utterances, the
[signature](#signing-a-result) and what to cite:

```python
import re

result = kudio.score_dataset("testset/enhanced", reference="testset/clean",
                             baseline="testset/noisy", metrics=["pesq", "stoi", "si_sdr"],
                             groups=re.compile(r"^(p\d+)_"))
print(result.to_markdown())          # or to_latex(), to_json()
result.save_rows("scores.csv")       # one row per utterance
```

`groups` takes a compiled pattern whose first group is the speaker, a dict, a
function, or the name of a Kaldi `utt2spk` or CSV file. [Command
line](../cli.md#score) shows the table it prints.

[`AudioEvaluate`][kudio.core.evaluator.AudioEvaluate] scores lists of files
paired by position, the clean references, the noisy inputs and one list per
method, with PESQ (where installed), STOI and SDR. `ev.eval(name, files)` adds a
method, [`get_df`][kudio.core.evaluator.AudioEvaluate.get_df] gives the means
with the noisy files as the Baseline row (`kudio[data]`), and
[`per_file`][kudio.core.evaluator.AudioEvaluate.per_file] one row per method and
file, with the `clean`, `noisy` and `file` paths, to join with a
[`Synthesizer`](synthesis.md) manifest on `noisy`. A file a metric cannot score
gets NaN there and is left out of the means on both sides, with a warning.

## Confidence intervals and tests

```python
rng = np.random.default_rng(23)
speakers = np.repeat([f"spk{k:02d}" for k in range(20)], 30)   # 20 speakers, 30 each
voice = np.repeat(rng.normal(0.0, 0.35, 20), 30)    # how each voice scores
gain = np.repeat(rng.normal(0.12, 0.10, 20), 30)    # how much each voice gains
pesq_base = 2.6 + voice + rng.normal(0.0, 0.3, 600)
pesq_a = pesq_base + gain + rng.normal(0.0, 0.1, 600)
ci = kudio.paired_ci(pesq_a, pesq_base, groups=speakers)
test = kudio.permutation_test(pesq_a, pesq_base, groups=speakers)
print(f"{ci.estimate:+.3f} PESQ, 95% CI [{ci.low:+.3f}, {ci.high:+.3f}], "
      f"{ci.n_groups} speakers, p = {test.pvalue:.4f}")
alone = kudio.paired_ci(pesq_a, pesq_base)
print(f"by utterance: [{alone.low:+.3f}, {alone.high:+.3f}]")
```

```text
+0.106 PESQ, 95% CI [+0.057, +0.168], 20 speakers, p = 0.0007
by utterance: [+0.094, +0.119]
```

[`paired_ci`][kudio.core.stats.paired_ci] gives a bootstrap interval for the mean
difference between two systems on the same utterances, and
[`permutation_test`][kudio.core.stats.permutation_test] the sign-flip p-value.
With `groups`, whole speakers are resampled and flipped together; by utterance
the interval is far too narrow, because 600 utterances of 20 voices are not 600
independent draws. [`bootstrap_ci`][kudio.core.stats.bootstrap_ci] gives one
system's mean, [`holm`][kudio.core.stats.holm] adjusts the p-values of several
systems against one baseline, and [`rate_ci`][kudio.core.stats.rate_ci] and
[`paired_rate_ci`][kudio.core.stats.paired_rate_ci] take error counts and
reference lengths per utterance, so a WER is pooled in every resample.

The defaults are BCa intervals from 10,000 resamples with seed 0. With 13 groups
or fewer the test runs through every sign pattern and its p-value is exact. Under
10 groups the intervals warn that they mean little, and with 5 or fewer no
two-sided p-value can fall under 0.05. NaN scores are refused: drop those
utterances from every list, `groups` included. Report the number of speakers
with every interval.

## Reporting and citing a metric

```python
info = kudio.metric_info("csii_high")
print(info.function, info.field, info.better, info.range, info.validation)
text = kudio.signature(["stoi", "si_sdr", "lsd"], sr)
print(text)
print(kudio.signature({"eer": {}, "min_dcf": {"p_target": 0.01}}, n=4874))
print(kudio.parse_signature(text)["lsd"])
```

```text
kudio.csii high higher (0.0, 1.0) validation.md#ncm-and-csii
kudio:5.0.0|sr:16000|pair:aligned=no,cut=shorter,limit=0.5|si_sdr|stoi|lsd:n_fft=512,hop_length=256,scale=ln,match_gain=yes,eps=1e-08
kudio:5.0.0|eer:method=linear|min_dcf:p_target=0.01,c_miss=1.0,c_fa=1.0|n:4874
{'n_fft': '512', 'hop_length': '256', 'scale': 'ln', 'match_gain': 'yes', 'eps': '1e-08'}
```

[`metric_info`][kudio.core.registry.metric_info] gives each metric's direction,
range, the settings that change its number (`variants`) and the works it comes
from (`refs`); [`cite`][kudio.core.registry.cite] writes their BibTeX, kudio's
entry first.

### Signing a result

[`signature`][kudio.core.signature.signature] writes one line, after sacreBLEU's,[^post]
that names every setting that changes a number as it was resolved (PESQ's
`'auto'` is written as `mode=wb` at 16 kHz), how the pair was treated
(`aligned`, `cut`, `limit`), and the version of any package that computed one,
`pesq` among them. The metrics come in the registry's order, so one set of metrics and
settings gives one line. A metric called on its own takes its keyword arguments
as a dict; `n=` and `ci=` record how many items were scored and how the interval
was computed, and DNSMOS adds 8 hex digits of its model's SHA-256. Write the
signature in the run that computed the numbers;
[`parse_signature`][kudio.core.signature.parse_signature] reads one back. The
parameters are under [Metrics and alignment](../reference/metrics.md).

[^sisdr]: J. Le Roux, S. Wisdom, H. Erdogan and J. R. Hershey, "SDR – half-baked or well done?", Proc. ICASSP 2019, pp. 626-630, doi:10.1109/ICASSP.2019.8683855.
[^sdi]: J. Chen, J. Benesty, Y. Huang and S. Doclo, "New insights into the noise reduction Wiener filter", IEEE Trans. Audio, Speech, Lang. Process. 14(4), pp. 1218-1234, 2006, doi:10.1109/TSA.2005.860851. The index is their Eq. 34.
[^stoi]: C. H. Taal, R. C. Hendriks, R. Heusdens and J. Jensen, "An algorithm for intelligibility prediction of time-frequency weighted noisy speech", IEEE Trans. Audio, Speech, Lang. Process. 19(7), pp. 2125-2136, 2011, doi:10.1109/TASL.2011.2114881. ESTOI: J. Jensen and C. H. Taal, "An algorithm for predicting the intelligibility of speech masked by modulated noise maskers", IEEE/ACM Trans. Audio, Speech, Lang. Process. 24(11), pp. 2009-2022, 2016, doi:10.1109/TASLP.2016.2585878.
[^p862]: ITU-T Recommendation P.862 (02/2001), "Perceptual evaluation of speech quality (PESQ): An objective method for end-to-end speech quality assessment of narrow-band telephone networks and speech codecs"; P.862.1 (11/2003), "Mapping function for transforming P.862 raw result scores to MOS-LQO"; P.862.2 (11/2007), "Wideband extension to Recommendation P.862 for the assessment of wideband telephone networks and speech codecs". P.862 Amendment 2 (11/2005), "Revised Annex A – Reference implementations and conformance testing for ITU-T Recs P.862, P.862.1 and P.862.2", whose A.2 sets the sampling rates and whose C code, version 2.0, is the reference program; P.862 Corrigendum 2 (03/2018). ITU withdrew P.862, P.862.1, P.862.2 and P.862.3 on 5 January 2024 and points to P.863 instead.
[^torcoli]: M. Torcoli, M. M. Halimeh and E. A. P. Habets, "Navigating PESQ: Up-to-date versions and open implementations", ITG Conference on Speech Communication 2025, arXiv:2505.19760.
[^p835]: ITU-T Recommendation P.835 (11/2003), "Subjective test methodology for evaluating speech communication systems that include noise suppression algorithm".
[^hl08]: Y. Hu and P. C. Loizou, "Evaluation of objective quality measures for speech enhancement", IEEE Trans. Audio, Speech, Lang. Process. 16(1), pp. 229-238, 2008, doi:10.1109/TASL.2007.911054. The CSIG, CBAK and COVL equations: Y. Hu and P. C. Loizou, "Evaluation of objective measures for speech enhancement", Proc. Interspeech 2006, doi:10.21437/Interspeech.2006-84.
[^urgent]: K. Saijo, W. Zhang, S. Cornell, R. Scheibler, C. Li, Z. Ni, A. Kumar, M. Sach, Y. Fu, W. Wang, T. Fingscheidt and S. Watanabe, "Interspeech 2025 URGENT speech enhancement challenge", Proc. Interspeech 2025, pp. 858-862, doi:10.21437/Interspeech.2025-1363. The scoring script is `lsd_metric` in `evaluation_metrics/calculate_intrusive_se_metrics.py`, <https://github.com/urgent-challenge/urgent2025_challenge>.
[^nvsr]: H. Liu, W. Choi, X. Liu, Q. Kong, Q. Tian and D. Wang, "Neural vocoder is all you need for speech super-resolution", Proc. Interspeech 2022, pp. 4227-4231, doi:10.21437/Interspeech.2022-11017, Eq. 7; scored with `ssr_eval`, <https://github.com/haoheliu/ssr_eval>.
[^mhl09]: J. Ma, Y. Hu and P. C. Loizou, "Objective measures for predicting speech intelligibility in noisy conditions based on new band-importance functions", J. Acoust. Soc. Am. 125(5), pp. 3387-3405, 2009, doi:10.1121/1.3097493.
[^ka05]: J. M. Kates and K. H. Arehart, "Coherence and the speech intelligibility index", J. Acoust. Soc. Am. 117(4), pp. 2224-2237, 2005, doi:10.1121/1.1862575.
[^bss]: E. Vincent, R. Gribonval and C. Févotte, "Performance measurement in blind audio source separation", IEEE Trans. Audio, Speech, Lang. Process. 14(4), pp. 1462-1469, 2006, doi:10.1109/TSA.2005.858005.
[^upit]: M. Kolbæk, D. Yu, Z.-H. Tan and J. Jensen, "Multitalker speech separation with utterance-level permutation invariant training of deep recurrent neural networks", IEEE/ACM Trans. Audio, Speech, Lang. Process. 25(10), pp. 1901-1913, 2017, doi:10.1109/TASLP.2017.2726762.
[^convtasnet]: Y. Luo and N. Mesgarani, "Conv-TasNet: surpassing ideal time-frequency magnitude masking for speech separation", IEEE/ACM Trans. Audio, Speech, Lang. Process. 27(8), pp. 1256-1266, 2019, doi:10.1109/TASLP.2019.2915167. SI-SNR is their eq. 15, both signals taken to zero mean; SI-SNRi and SDRi are "the additive values over the original mixture".
[^post]: M. Post, "A call for clarity in reporting BLEU scores", Proc. Third Conference on Machine Translation: Research Papers, Brussels, October 2018, pp. 186-191, doi:10.18653/v1/W18-6319.
[^vkh18a]: S. Van Kuyk, W. B. Kleijn and R. C. Hendriks, "An instrumental intelligibility metric based on information theory", IEEE Signal Process. Lett. 25(1), pp. 115-119, 2018, doi:10.1109/LSP.2017.2774250.
[^vkh18b]: S. Van Kuyk, W. B. Kleijn and R. C. Hendriks, "An evaluation of intrusive instrumental intelligibility metrics", IEEE/ACM Trans. Audio, Speech, Lang. Process. 26(11), pp. 2153-2166, 2018, doi:10.1109/TASLP.2018.2856374.
[^ksg04]: A. Kraskov, H. Stögbauer and P. Grassberger, "Estimating mutual information", Phys. Rev. E 69(6), 066138, 2004, doi:10.1103/PhysRevE.69.066138.
