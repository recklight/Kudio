# Measuring quality

Every metric on this page scores a processed recording against the clean
recording it came from. When there is no clean recording, the checks on
[Inspecting a recording](inspection.md) still work, DNSMOS among them;
everything here needs the reference.

Four of them, SNR, SI-SDR, segmental SNR and SDI, also compare sample *i* of
one signal with sample *i* of the other. One sample of delay takes a perfect
copy down to +11.9 dB of SI-SDR, a little above what the denoiser on this page
scores with no delay at all (+11.7 dB), and nothing in the score says that a
delay is what happened. SDR forgives a late start, STOI about a millisecond,
and PESQ lines the pair up itself, but none of them reports the delay either.
So lining the pair up comes before choosing a metric.

## A first score

The examples on this page share one synthetic clip: two phrases of a harmonic
tone over room tone, with a pause between them, mixed with white noise at
5 dB SNR by [`add_noise_snr`][kudio.effects.augment.add_noise_snr] and
denoised by [`spectral_enhance`][kudio.enhance.spectral.spectral_enhance]
(see [Denoising](enhancement.md)). Everything is seeded, so apart from the
figures above 100 dB, which are round-off, the numbers below should come out
the same on your machine.

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
```

[`score`][kudio.core.evaluator.score] takes the reference first and the
signal being judged second, runs the metrics by name and returns a dict. With
no list it runs `snr`, `si_sdr`, `segsnr`, `sdi` and whichever of `pesq`,
`stoi`, `estoi` and `sdr` can run on this machine at this rate, which here is
all of them but PESQ. The measures further down the page, from
[Hu and Loizou's](#hu-and-loizous-measures) on, run when they are named.

```python
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

`snr`, `si_sdr`, `segsnr` and `sdi` are plain numpy and need nothing beyond
kudio. Each is also a function of its own with the same argument order:
[`snr`][kudio.core.evaluator.snr], [`si_sdr`][kudio.core.evaluator.si_sdr],
[`segmental_snr`][kudio.core.evaluator.segmental_snr] and
[`sdi`][kudio.core.evaluator.sdi]. `stoi`, `estoi` and `sdr` need
`kudio[eval]`; `pesq` needs `kudio[pesq]` and a rate of 8 or 16 kHz. Name a
metric that cannot run and `score` raises instead of leaving it out: an
unknown name is a `ValueError`, a missing package a
[`DependencyError`][kudio.exceptions.DependencyError] that names the extra to
install.

[Getting started](../getting-started.md) has why mir_eval is pinned below 0.9.
[`check_metrics_install`][kudio.core.evaluator.check_metrics_install] returns
`(pesq_ok, stoi_ok, sdr_ok)`, which back-ends this machine has, and logs an
install hint for each missing one once per process.
`AudioEvaluate` and `eval_metrics` call it too, so configure logging before
the first of them, or that one hint goes nowhere.

[`eval_metrics`][kudio.core.evaluator.eval_metrics] is the fixed-shape
version, for code that unpacks three values: always `(pesq, stoi, sdr)`, with
`None` for any of them that is not installed, cannot run on the pair or fails
on it. Unlike everything else on this page, it takes the rate **first**:
`kudio.eval_metrics(sr, clean, noisy)` returns
`(None, 0.6065905900244255, 5.010295956402773)` here.

## Line the pair up first

Here is the clip scored against itself, and the log-MMSE output scored
against the clip, each starting a few samples late. Both keep their length,
so there is nothing for a length check to notice:

![SI-SDR against delay: a perfect copy one sample late scores +11.9 dB, and from four samples on both the copy and the log-MMSE output score below the unprocessed noisy input](../images/metrics-delay-light.webp#only-light)
![SI-SDR against delay: a perfect copy one sample late scores +11.9 dB, and from four samples on both the copy and the log-MMSE output score below the unprocessed noisy input](../images/metrics-delay-dark.webp#only-dark)

Four samples late, a quarter of a millisecond at 16 kHz, the copy scores
+3.9 dB and the denoised take +4.1 dB, both below the noisy input that was
never processed at all (+5.0 dB). From three samples on, the two curves stay
within 0.4 dB of each other: the delay decides the score and the denoising
hardly shows in it. Read without knowing about the delay, the number says the
denoiser does worse than nothing.

Delays come from ordinary places: a causal filter, a resampler, a model that
pads its input frames, a take played through a speaker and recorded again.
When the processed file comes back longer than the clean one, the
[length check](#when-the-lengths-differ) warns. When something trimmed it back
to the input's length, nothing does.

### Finding the delay

[`find_delay`][kudio.core.align.find_delay] measures the delay to the nearest
sample by cross-correlation, and [`align`][kudio.core.align.align] returns the
pair trimmed to the part they share:

```python
processed = np.concatenate([np.zeros(128, np.float32), enhanced])  # 8 ms late

found = kudio.find_delay(clean, processed, sr)
print(found)

a, b = kudio.align(clean, processed, sr)
print(len(a), len(b), round(kudio.si_sdr(a, b), 2))
```

```text
lags by 128 samples (8.0 ms), correlation 0.968
64000 64000 11.72
```

Scored as it sits, the same pair gets -7.97 dB, plus a warning that the
lengths differ.

In the [`Alignment`][kudio.core.align.Alignment] that `find_delay` returns, a
positive `delay` means the second signal starts later. The correlation is
normalised over the overlap only, so a real match that shares one second of a
long file still scores high. A chain with a fixed latency needs measuring only
once, on a loud passage, and
[`Alignment.apply`][kudio.core.align.Alignment.apply] trims any later pair at
the delay already found:

```python
other = kudio.add_noise_snr(clean, noise, snr_db=10, seed=0)
other_out = np.concatenate([np.zeros(128, np.float32),
                            kudio.spectral_enhance(other, sr)])

a, b = found.apply(clean, other_out)     # the 128 samples measured above
print(round(kudio.si_sdr(a, b), 2))
```

```text
14.74
```

### The wrong file still gets a delay

Cross-correlation always has a maximum, so `find_delay` returns a delay for any
two signals, including two that have nothing in common. The correlation is
what tells the cases apart, and `align` refuses below
[`MIN_CORRELATION`][kudio.core.align.MIN_CORRELATION], which is 0.5:

```python
print(kudio.find_delay(clean, noise, sr))
kudio.align(clean, noise, sr)
```

```text
leads by 5873 samples (367.1 ms), correlation 0.016, polarity inverted
kudio.exceptions.FeatureError: these do not look like the same recording: best correlation 0.016 at -5873 samples, below 0.5. Aligning them would put a number on two unrelated signals; pass min_correlation=0 to do it anyway.
```

At a correlation of 0.016 the polarity note means nothing either.

I set the threshold low on purpose. The denoised take above sits at 0.968; the
threshold is there to catch somebody picking the wrong file, not to grade the
processing.

Clearing it does not prove the lag is the one you meant, either. Speech
repeats itself, and an unbounded search over a long file can find a better
match half a minute away. When you know the delay is small, bound the search
with `max_seconds=`, set above the largest delay you expect. What comes back
is the best lag inside the bound, and that can still clear 0.5 when the real
lag lies outside it.

### Polarity

An inverted copy correlates at -1. It is the same take, and something in the
chain flipped it. I treat that as a finding rather than something to correct
on the way past: `find_delay` reports `polarity inverted`, and `align` logs a
warning, seen once logging is configured, and returns the pair as it is.
SI-SDR does not see the flip and SNR does: lined up with `-processed`, the
pair above still scores 11.72 dB of SI-SDR, and its SNR drops to -5.77 dB.

### Mono, at one rate

`find_delay` and `align` take mono audio. A stereo array is refused rather
than mixed down, because which channel led is a question of its own. The
metrics refuse one too, with a `ValueError`, rather than score both channels
as one signal; score a channel at a time, or mix down first. One channel can
come as a vector, a column `(n, 1)` or a row `(1, n)`. Both signals also
have to be at the one `sr` you pass. `find_delay` cannot tell when they are
not, so bring them together with [`resample`][kudio.core.io.resample] first.

For a single pair of files, `kudio align clean.wav processed.wav --metrics`
prints the delay and the SI-SDR before and after lining them up, and exits 1
when the two do not look like the same recording (see
[Command line](../cli.md)).

## When the lengths differ

[`snr`][kudio.core.evaluator.snr], `si_sdr`, `segmental_snr` and `sdi` cut
both signals to the length they share, and so do `score` and `eval_metrics`
for every metric they run but PESQ. The tail of the longer one is never compared, so a
difference in length is warned about on every call:

```python
kudio.snr(clean, clean[:60000])
```

```text
UserWarning: snr(): ref is 64000 samples and deg 60000: only the first 60000 samples are scored and the rest is ignored. Use kudio.align if the short one is delayed rather than just short, or trim the pair yourself to say which part you meant.
```

Python shows an identical warning from one line only once, so in a loop over
pairs all cut by the same amount only the first warns;
`warnings.simplefilter("always")` shows each.

Once more than half of the longer signal would go unmeasured, the pair is
refused. Without that limit, the first second of the clip would score as a
perfect copy of all four seconds, because the three it lacks are never looked
at:

```python
kudio.snr(clean, clean[:16000])
```

```text
ValueError: snr(): ref is 64000 samples and deg 16000: comparing them leaves 48000 of the longer one's 64000 samples unmeasured (75.0%), past the 50% limit, so the score would be about the overlap rather than about the pair. Check that these two are the same utterance, line them up with kudio.align, or raise kudio.core.evaluator.MAX_LENGTH_MISMATCH to score them as they sit.
```

The limit is
[`MAX_LENGTH_MISMATCH`][kudio.core.evaluator.MAX_LENGTH_MISMATCH], 0.5 of the
longer signal. A real pipeline loses a window or two at the tail, or a sample
to a resampler's rounding, and half the file is far beyond that. The constant
is read on every call, so a pair that really is that lopsided can be scored
after `kudio.core.evaluator.MAX_LENGTH_MISMATCH = 0.8`. It has to be set on
that module: there is no `kudio.MAX_LENGTH_MISMATCH`, and assigning one
changes nothing.

[`AudioEvaluate`][kudio.core.evaluator.AudioEvaluate] checks every pair of
files it reads the same way, with both file names in the message. The refusal
is a plain `ValueError`, so an `except` for
[`KudioError`][kudio.exceptions.KudioError] around a batch will not catch it.

Of what `score`, `eval_metrics` and `AudioEvaluate` run, PESQ is the one
metric that gets the two whole: [`pesq`][kudio.core.evaluator.pesq], and the
PESQ those three and `composite` compute, take both signals as they are, the
way the ITU program reads two files. PESQ finds the delay between the two
itself and works over both as they are, so a cut changes its score even
where it drops only silence: on the 40 pairs of ITU's conformance test 2(b),
whose tails past the shorter file are silence or low noise, cutting moved the
score by up to 0.56 ([How the metrics are checked](../validation.md#pesq)).
Where PESQ runs beside metrics that cut, the length warning says that PESQ
got both whole; `score` asked for PESQ alone, and `AudioEvaluate.pesq_score`,
do not check the lengths at all.

## What each metric sees

The same clip, damaged six ways, scored by six metrics:

| Damage | `snr` | `si_sdr` | `segsnr` | `sdi` | `stoi` | `sdr` |
|---|---:|---:|---:|---:|---:|---:|
| noise, 5 dB SNR | 5.0 | 5.0 | -2.1 | 0.32 | 0.61 | 5.0 |
| half the gain | 6.0 | 138.9 | 6.0 | 0.25 | 1.00 | 283.0 |
| polarity flipped | -6.0 | 144.9 | -6.0 | 4.00 | 1.00 | 283.0 |
| low-pass, 2 kHz | 18.2 | 18.2 | 10.3 | 0.02 | 0.93 | 24.9 |
| 1 sample late | 12.1 | 11.9 | 4.6 | 0.06 | 1.00 | 91.3 |
| 1 ms late | -1.4 | -10.0 | -2.2 | 1.40 | 0.99 | 53.6 |

??? note "The script behind the table"

    ```python
    def late(y, n):
        return np.concatenate([np.zeros(n, np.float32), y])[:len(y)]

    cases = {
        "noise, 5 dB SNR": noisy,
        "half the gain": 0.5 * clean,
        "polarity flipped": -clean,
        "low-pass, 2 kHz": kudio.lowpass(clean, sr, cutoff=2000),
        "1 sample late": late(clean, 1),
        "1 ms late": late(clean, 16),
    }
    names = ["snr", "si_sdr", "segsnr", "sdi", "stoi", "sdr"]
    for case, deg in cases.items():
        s = kudio.score(clean, deg, sr, names)
        print(f"| {case} | {s['snr']:.1f} | {s['si_sdr']:.1f} | {s['segsnr']:.1f} "
              f"| {s['sdi']:.2f} | {s['stoi']:.2f} | {s['sdr']:.1f} |")
    ```

Anything above 100 dB means the metric found nothing wrong. In `snr` and
`si_sdr` the figure comes from a 1e-12 guard against dividing by zero, so it
moves with the clip's energy, and 138.9 and 144.9 are the same verdict. The
283.0 of `sdr` is round-off left in mir_eval's residual.

[`snr`][kudio.core.evaluator.snr]
:   Charges for everything: noise, gain, polarity, filtering, delay. A perfect
    copy at half the level scores 6.0 dB. Use it when the level is part of
    what you are judging.

[`si_sdr`][kudio.core.evaluator.si_sdr]
:   Removes the mean of both signals, then the best-fitting gain,[^sisdr]
    so a level change or a flipped polarity costs nothing while filtering
    and delay still do. It is the number I look at first for a denoiser,
    which should not lose points for coming back quieter.

`segsnr`, [`segmental_snr`][kudio.core.evaluator.segmental_snr]
:   SNR per frame, each frame held to [-10, 35] dB, then averaged. Through
    `score` the frames are always 512 samples with a hop of 256, which is
    32 ms at 16 kHz and 10.7 ms at 48 kHz; `segmental_snr` takes
    `frame_length=` and `hop_length=` to change them. Pauses count as much as
    speech: in the 5 dB mix, 116 of the 249 frames sit at the -10 dB floor,
    most of them in the room tone around the phrases, and that pulls the
    average to -2.1 dB. A perfect copy scores exactly 35.0 unless some of
    its frames are digital silence: such a frame scores -10 dB, or 0 dB where
    the output is silent there too, and a warning counts them. A clip
    shorter than one frame scores nan, with a warning.

[`sdi`][kudio.core.evaluator.sdi]
:   `sum((ref - deg)**2) / sum(ref**2)`, which is SNR on a linear scale,
    `10 ** (-snr / 10)`: 0 is a perfect copy, 1 is as much error as there is
    speech, and lower is better. It is taken on the whole output, so leftover
    noise counts as error along with distortion, and the unprocessed 5 dB mix
    scores 0.32. The speech-distortion index of the noise-reduction
    literature[^sdi] passes the clean speech through the filter on its own,
    which leaves the noise out, and an untouched input scores 0 there.
    Compare `sdi` only with figures computed the way it is.

`stoi`, `estoi`
:   Short-time objective intelligibility,[^stoi] through `pystoi`. It
    compares short-time envelopes in one-third-octave bands after resampling
    both signals to 10 kHz and dropping silent frames, so gain, polarity and
    a millisecond of delay barely move it, while noise does. It forgives
    little more than that: 5 ms late scores 0.85 and 10 ms 0.63, about what
    5 dB of noise costs (0.61). `estoi`, the extended version, gives 0.31 on
    the 5 dB mix where STOI gives 0.61. A perfect copy scores 1 on both, and
    higher is better. pystoi needs about 0.4 s of audio left once the silent
    frames are gone. Below that it hands back 0.00001 even for a perfect copy,
    a placeholder rather than a score, so kudio scores such a clip NaN with a
    warning, and `AudioEvaluate` leaves it out of the mean.

`sdr`
:   The SDR of BSS Eval,[^bss] through `mir_eval`. It fits the best 512-tap
    filter from the reference to the output (512 is mir_eval's length; the
    paper leaves it open) and counts as distortion only
    what that filter cannot produce. Gain, polarity, a low-pass and a late
    start of up to 511 samples are partly or fully forgiven, so one sample
    late scores 91.3 dB. The filter is causal, and an output that starts
    early gets none of that: one sample early scores 15.2 dB. Compare it with
    SDR figures computed the same way, never with SI-SDR.

Different metrics also disagree about the same denoiser. On this clip,
log-MMSE gains 6.8 dB of SI-SDR (4.97 to 11.72) while STOI stays where it was
(0.607 to 0.601), and ESTOI drops, from 0.305 to 0.276.

## PESQ: say which one

PESQ comes in two standards and two scales, and they give different numbers
for the same file.[^p862] ITU-T P.862 scores narrowband speech, and kudio
runs it at 8 or 16 kHz; P.862.2 is its wideband extension. P.862.2 gives input
filters for 8 kHz as well as 16, but P.862's Annex A requires a wideband
implementation to run at 16 kHz, and the reference program, the `pesq` package
and kudio all do. P.862 can be reported raw, on -0.5 to 4.5, or mapped
to MOS-LQO through P.862.1. P.862.2 reports only a MOS-LQO: its raw model
output always goes through a mapping of its own. [`pesq`][kudio.core.evaluator.pesq] takes the bandwidth as `mode=`
and the scale as `scale=` rather than picking one without telling you, and
`score`, `eval_metrics` and `AudioEvaluate` pass them on as `pesq_mode=` and
`pesq_scale=`:

| `mode` | `scale` | at 8 kHz | at 16 kHz |
|---|---|---|---|
| `'auto'` (default) | `'lqo'` (default) | P.862, MOS-LQO | P.862.2, MOS-LQO |
| `'nb'` | `'lqo'` | P.862, MOS-LQO | P.862, MOS-LQO |
| `'wb'` | `'lqo'` | refused | P.862.2, MOS-LQO |
| `'auto'` or `'nb'` | `'raw'` | P.862, raw | P.862, raw |
| `'wb'` | `'raw'` | refused | refused |

Any other rate raises a `ValueError` that says to resample first.

A paper that just says "PESQ" may mean the raw P.862 score or a MOS-LQO,
narrowband or wideband, so find out which before setting your numbers beside
it. When a number goes into a table, I pass both arguments and name the scale
next to it. Both lines need the `pesq` package, which compiles from source
([Getting started](../getting-started.md) has what that takes):

```python
kudio.pesq(clean, enhanced, 16000)                          # P.862.2 MOS-LQO
kudio.pesq(clean, enhanced, 16000, mode="nb", scale="raw")  # raw P.862
```

The `pesq` package returns MOS-LQO, and kudio takes the raw score back
through the P.862.1 mapping; the formula is in the reference for `pesq`. The
two scales are not a constant apart. A raw 1.0 is a narrowband MOS-LQO of
1.16, 2.0 is 1.63, 3.0 is 2.82 and 4.0 is 4.15. They cross at raw 1.22 and
3.37, and between the two the raw score is the higher, so a narrowband
MOS-LQO set against a paper's raw PESQ makes a method look worse than it is.
P.862.2 maps with coefficients of its own, so none of this converts a
wideband score.

P.862.2 itself does not recommend its wideband extension for systems with
noise suppression between where the signal goes in and where it is captured
(its Sec. 1), and an enhanced output has been through exactly that. Wideband
PESQ in enhancement papers is a convention, not a use the Recommendation
vouches for.

The wideband PESQ of the `pesq` package, and so kudio's, is the 2005
reference code as it was published. ITU corrected it in 2018, in P.862's
Corrigendum 2: the code puts wideband signals into its loudness model at the
wrong level, and so predicts listeners 0.8 MOS low on average. The correction
scales the wideband input filter by 0.251188, to bring the signals to 76 dB
SPL. In ITU's own wideband table for the 1,736 pairs of Supplement 23, the
corrected scores are 0.73 higher on average, from 0.08 lower to 1.40 higher.
Torcoli et al. take the uncorrected version to be the one most used, and put
out a corrected package, `pesqc2`.[^torcoli] A wideband score from it and
one from `pesq` cannot be set side by side.

## Hu and Loizou's measures

Enhancement papers often report a second family beside PESQ: the
log-likelihood ratio (LLR), the Itakura-Saito distance (IS), the cepstral
distance (CEP), the weighted spectral slope (WSS), the frequency-weighted
segmental SNR (fwSegSNR), and the composite CSIG, CBAK and COVL built from
LLR, WSS, a segmental SNR and PESQ.[^hl08] kudio computes them as the MATLAB
code Loizou published with that work does, the code the published composite
figures come from. The module docstring lists where the code and the paper
differ, and [How the metrics are checked](../validation.md) how closely kudio
agrees with the code's Python port. They need nothing beyond numpy and are
defined only at 8 and 16 kHz. Each compares the pair 30 ms frame by 30 ms frame, so line
the pair up first, as for SNR.

`score` runs them when they are named, as `llr`, `is` (Itakura-Saito),
`cep`, `wss` and `fwsegsnr`, and each is a function of its own:
[`llr`][kudio.core.quality.llr],
[`itakura_saito`][kudio.core.quality.itakura_saito],
[`cepstral_distance`][kudio.core.quality.cepstral_distance],
[`wss`][kudio.core.quality.wss] and
[`fw_segmental_snr`][kudio.core.quality.fw_segmental_snr].

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

Lower is better for the first four and higher for fwSegSNR, and they do not
agree about log-MMSE: LLR and fwSegSNR say it helped, IS and WSS say it hurt,
and CEP barely moves. LLR, IS and CEP read the LPC envelope of each frame,
WSS the slopes between 25 bands, and fwSegSNR how the spectrum is shared
among those bands, so each sees a different part of what the denoiser did.
Three things to keep in mind when reading them:

- LLR, IS, CEP and WSS average only the best 95% of frames, so damage
  confined to less than 5% of a file can leave them untouched.
- The 25 bands stop at 3.8 kHz at both rates, so WSS reads nothing above
  that, even in a 16 kHz file. fwSegSNR's bands stop there too, but it first
  scales both spectra to unit area over every bin up to half the rate, so
  noise or loss above 4 kHz still lowers it.
- WSS depends on the level of a quiet pair. Loizou's code reads every band
  energy under -100 dB of full scale as -100 dB, so from a clean peak of
  about -50 dBFS down the quieter bands sit on that floor: the 8 kHz test
  pair that scores 35.98 at any peak from -6 to -46 dBFS scores 27.95 at -66.
  `wss` and `composite` warn when it happens; bring both signals to a normal
  level with one gain first.

### CSIG, CBAK and COVL

[`composite`][kudio.core.quality.composite] predicts the three ratings of
ITU-T P.835[^p835] on the 1 to 5 scale: CSIG for signal distortion, CBAK for how
intrusive the background is, COVL for overall quality. They are linear fits
of PESQ, LLR, WSS and a segmental SNR to listener ratings, so they need the
`pesq` package, or a PESQ score of your own passed as `pesq=`. Through
`score` they are `csig`, `cbak` and `covl`, all three from one call.

Which PESQ goes in moves all three. The fit was made on the raw narrowband
P.862 score of 8 kHz speech, but nearly every published 16 kHz figure,
VoiceBank-DEMAND's included, has wideband MOS-LQO inside, as Loizou's own
2012 code and the original pysepm compute it. `composite` does the same by
default: wideband MOS-LQO at 16 kHz and raw narrowband at 8 kHz.
`mode="nb", scale="raw"` gives the variant the fit was made on:

```python
for kwargs in ({}, {"mode": "nb", "scale": "raw"}):
    c = kudio.composite(clean, enhanced, sr, **kwargs)
    print(f"PESQ {c.pesq:.3f}  CSIG {c.csig:.3f}  CBAK {c.cbak:.3f}  COVL {c.covl:.3f}")
```

```text
PESQ 1.880  CSIG 3.617  CBAK 2.595  COVL 2.715
PESQ 2.875  CSIG 4.217  CBAK 3.071  COVL 3.516
```

The same output scores 0.8 apart on COVL, so name the PESQ variant beside
the ratings in a table. `score` always uses the default, whatever
`pesq_scale` says. The
[`CompositeScore`][kudio.core.quality.CompositeScore] carries the four
inputs as well, so a rating that moved can be traced to the input that moved
it. Its `segsnr` is the composite's own, on windowed 30 ms frames, and not
the `segsnr` of `score`.

## Log-spectral distance

[`lsd`][kudio.core.evaluator.lsd], `lsd` in `score`, compares the power
spectra of the pair frame by frame: the root mean square over frequency of
the log power ratio, averaged over frames. 0 is the same spectrum, and lower
is better. It sees what the SNR family averages away, such as a missing
band: the clip low-passed at 2 kHz scores 18.0, where the 5 dB mix scores
2.76 and the log-MMSE output 2.42.

The defaults give the number of the URGENT 2025 challenge's scoring
script:[^urgent] the natural log, 32 ms frames every 16 ms, and the output
first scaled by its least-squares gain against the reference, so a level
change costs nothing. Bandwidth-extension papers take base 10 and their own
STFT; at 44.1 kHz, `scale="log10", n_fft=2048, hop_length=441,
match_gain=False, eps=1e-12` gives the number of NVSR's `ssr_eval`.[^nvsr]
The three scales are one distance, so they differ by a constant factor:

```python
low = kudio.lowpass(clean, sr, cutoff=2000)
for scale in ("ln", "log10", "db"):
    print(f"{scale:5s} {kudio.lsd(clean, low, sr, scale=scale):6.2f}")
```

```text
ln     18.00
log10   7.82
db     78.17
```

Find out which scale a published number is on before setting yours beside
it; a table captioned "dB" is not proof. The gain match is a projection onto
the reference, so it needs the pair lined up: the clip 1 ms late scores 2.42
with it and 0.38 with `match_gain=False`. Frames where the reference is
digital silence score 18.4 each on the default scale, whatever the output
holds there, and a warning says how many of them there were.

## Mel-cepstral distortion

[`mcd`][kudio.core.melcepstrum.mcd] is the distance between the
mel-cepstra of the pair, frame by frame, after dynamic time warping has
paired the frames, in dB; lower is better. Implementations disagree on the
analysis, on the order, on whether the energy term c0 counts and on the
warping, so MCDs from different code cannot be compared. This one is the
URGENT 2025 challenge's,[^urgent] every setting as its script has it: SPTK's
mel-cepstral analysis of the order its table gives for the rate, c0
included, FastDTW with a radius of 1, and the output first scaled by its
least-squares gain against the reference. Because of the warping it also
scores a pair that differs in timing, as speech synthesis and conversion
produce; pass `match_gain=False` there, since the gain match needs the pair
lined up.

```python
for label, deg in (("noisy", noisy), ("enhanced", enhanced)):
    print(f"{label:8s} MCD {kudio.mcd(clean, deg, sr):.2f} dB")
slower = kudio.time_stretch(clean, 0.9)          # 10% longer
print(f"stretched MCD {kudio.mcd(clean, slower, sr, match_gain=False):.2f} dB")
```

```text
noisy    MCD 6.85 dB
enhanced MCD 5.40 dB
stretched MCD 3.23 dB
```

The rates with settings are 8, 16, 22.05, 24, 32, 44.1 and 48 kHz; at any
other, pass `order` and `alpha` yourself. It is slower than the other
measures here: ten minutes at 16 kHz take about 16 s.

## Intelligibility: NCM and CSII

[`ncm`][kudio.core.intelligibility.ncm] and
[`csii`][kudio.core.intelligibility.csii] estimate, like STOI, how much of
the speech a listener could still follow, from 0 to 1, higher better. Ma, Hu
and Loizou set both against listeners' scores for noise-suppressed speech in
72 conditions of car, babble, train and street noise.[^mhl09] NCM, the
normalized covariance metric, correlates the envelopes of the two signals in
20 bands. CSII, Kates and Arehart's coherence speech intelligibility
index,[^ka05] measures how much of the output stays coherent with the
reference in the SII's critical bands (CSII.m keeps the 16 centred up to
3.4 kHz), separately over the reference's loud, middle and quiet frames.
Kates and Arehart combine the three into one prediction; CSII.m stops at the
three, and so does kudio, so it returns three numbers. Both follow Loizou's MATLAB
code, NCM.m and CSII.m, which pysepm also ports, and need only numpy and
scipy.

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

`high` is over the frames at or above the reference's overall RMS, `mid`
over those up to 10 dB below it, and `low` over everything further down,
pauses included. Kates and Arehart stop the low level 30 dB down; CSII.m
sets no lower limit, and I keep the code's. Like ESTOI, both score the log-MMSE output below the noisy
input it came from. Through `score` the names are `ncm`, `csii_high`,
`csii_mid` and `csii_low`; the three CSII levels come from one call.

What to watch for:

- NCM is defined at 8 and 16 kHz only, and its bands move with the rate:
  300 to 3400 Hz at 8 kHz, 300 to 7400 Hz at 16 kHz. The same pair scores
  differently at the two, so give the rate with the number. CSII takes any
  rate from 8 kHz up, and its bands stop at 3.7 kHz at every rate, so it
  sees little of what happens above 4 kHz:

    ```python
    from scipy.signal import resample_poly

    clean8, noisy8 = resample_poly(clean, 1, 2), resample_poly(noisy, 1, 2)
    print(f"NCM {kudio.ncm(clean, noisy, sr):.3f} at 16 kHz, "
          f"{kudio.ncm(clean8, noisy8, 8000):.3f} at 8 kHz")
    ```

    ```text
    NCM 0.574 at 16 kHz, 0.663 at 8 kHz
    ```

- Neither hears level or has a hearing threshold. The clip low-passed at
  2 kHz still scores CSII 1.000 at every level and NCM 0.810: they say how
  much of the speech's structure the output keeps, not whether a listener
  would hear it.
- Both need the pair lined up, CSII most. The clean clip 5 ms late scores
  0.757 on high-level CSII and 0.948 on NCM, and 10 ms late 0.575 and 0.833.
- Short clips read high, since the covariance and the coherence are
  estimated over the clip. An unrelated stretch of synthetic speech scores
  NCM 0.36 on average against 1 s clips, 0.10 against 3 s and 0.02 against
  10 s. Score whole sentences, and compare clips of like length.
- Zeros are not neutral. Zero padding on both signals counts as agreement
  and raises NCM, and it lands in CSII's low level and lowers it.
- CSII reads down to 0 Hz, so a DC offset on the reference gives digital
  silence in the output credit. Score `np.zeros_like(clean)` once to see
  whether a reference does this; if it does, high-pass the reference at
  20 Hz, forwards and backwards, and take the output's mean out before
  scoring:

    ```python
    from scipy.signal import butter, sosfiltfilt

    muted = np.zeros_like(clean)
    offset = clean + 0.001 * np.abs(clean).max()        # DC at 0.1% of the peak
    sos = butter(2, 20, "highpass", fs=sr, output="sos")
    for ref in (clean, offset, sosfiltfilt(sos, offset)):
        print(f"{kudio.csii(ref, muted, sr).low:.3f}")
    ```

    ```text
    0.000
    0.255
    0.000
    ```

    Filter only the reference: a filter rings into the output's digital
    silence and moves its score.

## Intelligibility in bits: SIIB

[`siib`][kudio.core.siib.siib] and [`siib_gauss`][kudio.core.siib.siib_gauss]
come at intelligibility from information theory. Van Kuyk, Kleijn and
Hendriks treat speaking and listening as a chain, from a message to the
talker's speech to what reaches the listener, and score how much information
the clean speech and the output share, in bits per second.[^vkh18a] Both
signals are read as an auditory log-spectrum, 28 gammatone bands from 100 to
6500 Hz with forward masking; 15 frames (187.5 ms) are taken at a time and
turned into 420 channels that are close to independent, and the information
is summed over the channels. SIIB estimates each channel's information with
the nearest-neighbour estimator of Kraskov, Stögbauer and
Grassberger.[^ksg04] SIIB^Gauss, from the authors' later evaluation of
intrusive measures,[^vkh18b] reads each channel as a Gaussian channel and
needs only its correlation, so no random numbers enter it and it runs about
25 times faster on the pair below. Where the papers and the authors' MATLAB
code, SIIB.m and SIIB_Gauss.m, differ, kudio follows the code; the module
notes in the [API reference][kudio.core.siib] list each difference and how
much it moves the score. Both need only numpy and scipy.

The 4 s clip of this page is too short for either: SIIB needs 436 frames of
speech, 5.45 s once the reference's silence is dropped, and scores the clip
nan with a warning that says so. Here are 30 s of phrases like the clip's,
under white noise:

```python
gen = np.random.default_rng(1)
talk = 0.01 * gen.standard_normal(30 * sr)        # 30 s of room tone
at = sr // 4
while at < 28 * sr:                               # phrases like the clip's
    phrase = burst(gen.uniform(0.6, 1.4), gen.uniform(100, 220))
    talk[at:at + len(phrase)] += phrase
    at += len(phrase) + int(gen.uniform(0.2, 0.5) * sr)
hiss = gen.standard_normal(len(talk))

for snr in (-5, 0, 5, 10):
    heard = kudio.add_noise_snr(talk, hiss, snr_db=snr, seed=0)
    print(f"{snr:3d} dB    SIIB {kudio.siib(talk, heard, sr):6.1f}  "
          f"SIIB^Gauss {kudio.siib_gauss(talk, heard, sr):6.1f}")

heard = kudio.add_noise_snr(talk, hiss, snr_db=0, seed=0)
denoised = kudio.spectral_enhance(heard, sr)      # log-MMSE, from 0 dB
print(f"denoised  SIIB {kudio.siib(talk, denoised, sr):6.1f}  "
      f"SIIB^Gauss {kudio.siib_gauss(talk, denoised, sr):6.1f}")
print(f"copy      SIIB {kudio.siib(talk, talk, sr):.2f}")
```

```text
 -5 dB    SIIB   34.1  SIIB^Gauss   17.7
  0 dB    SIIB   46.7  SIIB^Gauss   25.4
  5 dB    SIIB   66.1  SIIB^Gauss   40.5
 10 dB    SIIB  123.0  SIIB^Gauss   75.2
denoised  SIIB   50.1  SIIB^Gauss   21.3
copy      SIIB 1335.76
```

The scale is not 0 to 1. No two readings of a sentence are the same, which
the authors model as a correlation of 0.75 between the message and the
speech in every channel; through it a channel carries 0.596 bits at most,
and a copy of the reference scores the ceiling of both measures,
1335.76 bit/s. On the authors' data, speech that listeners understood
completely scored about 150 bit/s. A score says how intelligible speech is
only through a mapping fitted to a listening test (their Eq. 13), and each
measure needs its own. Nor do the two move together: here SIIB^Gauss reads a
little over half of SIIB, and they disagree about the log-MMSE output, which
SIIB scores above the 0 dB input it came from and SIIB^Gauss below it.
Through `score` the names are `siib` and `siib_gauss`; neither runs unless
it is named.

What to watch for:

- It needs speech, and plenty of it. A pair with less than 5.45 s of speech,
  436 frames, once the frames more than 40 dB under the reference's loud end
  are dropped, scores nan with a warning. The covariance of 420 dimensions
  needs 421 stacks to have full rank; with fewer, SIIB.m still gives a
  number, but part of it comes from eigenvectors the eigen-solver picks at
  will, so it depends on the LAPACK it ran on. That is the least, not
  enough: SIIB.m's notes advise 20 s of speech or more, concatenating short
  utterances, as the evaluation paper did for each condition.[^vkh18b]
  Length enters the score as well. kudio's tests use 36 s of synthetic
  syllables with pauses the 40 dB threshold drops; under white noise at
  0 dB they score SIIB 102.2 over the first 12 s and 124.1 over all 36.
  Score each condition over stretches of like length.
- Line the pair up. The frames of the two are compared one against one: the
  talk above 5 ms late scores SIIB 700.1 and SIIB^Gauss 371.0, 10 ms late
  246.1 and 134.5, and 20 ms late 100.5 and 53.7, less than the 10 dB pair.
  Use [`kudio.align`][kudio.core.align.align] first.
- It hears level. The masking floor comes from the reference and holds the
  output's spectrum as well, so an output at another level is no longer a
  copy: twice as loud, the talk still scores SIIB's ceiling, 1335.76, but
  SIIB^Gauss 1112.1, and half as loud SIIB 1184.8 and SIIB^Gauss 700.9. In
  noise it matters less: the 0 dB pair at twice the level scores 46.2 and
  25.3 instead of 46.7 and 25.4. NCM and CSII hear no level at all.
- Only the frames where the reference speaks count. They are chosen on the
  reference, and what the output holds in the others is never looked at, so
  noise an enhancer leaves in the pauses costs it nothing: on the test
  syllables, white noise 7.5 dB louder than the speech, added to a copy in
  the pauses alone, leaves SIIB at its ceiling.
- Nothing takes DC out. An offset passes the window into the seven lowest
  bands and adds to every frame's level, which can lift the reference's
  pauses over the 40 dB threshold and into the score. On the test syllables
  at 0 dB, an offset of 0.005 on both signals, 18 dB under the speech's RMS,
  keeps all 2878 frames where 1645 were kept, and takes SIIB from 124.1 to
  72.8 and SIIB^Gauss from 64.0 to 36.8. Take the mean out of a recording
  that has an offset before scoring it.
- Other rates are taken to 16 kHz first, with the filter MATLAB's `resample`
  designs, as in SIIB.m. The 0 dB pair taken to 8 kHz first scores SIIB 50.3
  and SIIB^Gauss 25.6, and taken to 44.1 kHz 48.5 and 25.4, against 46.7 and
  25.4 at 16 kHz. SIIB depends on that filter. Through a sharper one, the
  test syllables made at 8 and 20 kHz by a near-ideal filter, under white
  noise at -10 to 20 dB, moved by up to 5.1 bit/s on SIIB and 1.6 on
  SIIB^Gauss, and made at 8 kHz by `resample_poly`'s own filter, by up to 1.9
  and 0.9; `tools/validation/siib.py` runs that sweep. An implementation
  that resamples otherwise can differ from kudio by a few bit/s at any rate
  but 16 kHz. pySIIB, the Python port, is one: over the sweep it gives the
  sharper filter's scores to 1e-3 bit/s, while at 16 kHz it gives kudio's
  SIIB^Gauss to under 1e-12 of itself and its SIIB to 1.4e-5 at most. Give
  the rate with the score.
- SIIB's neighbour count grows with the speech, so its time grows faster
  than the speech: 2.6 s for the 30 s above, and half a minute for two
  minutes. SIIB^Gauss took 0.1 s here, and its time grows with the speech.

## Several sources: SI-SDRi, PIT and BSS Eval

Enhancement and separation papers report what a system gained over its
input as well as the score itself: SI-SDRi, the SI-SDR of the output minus
that of the mixture it came from, and SDRi, the same with BSS Eval's
SDR.[^convtasnet]
[`si_sdr_improvement`][kudio.core.separation.si_sdr_improvement] and
[`sdr_improvement`][kudio.core.separation.sdr_improvement] take the
reference, the output and the input; the second needs mir_eval
(`kudio[eval]`), as `sdr` in `score` does.

```python
print(f"{kudio.si_sdr_improvement(clean, enhanced, noisy):.2f}")
print(f"{kudio.sdr_improvement(clean, enhanced, noisy):.2f}")
```

```text
6.76
8.32
```

A separator returns its outputs in no particular order, so each one has to
be matched with its source before anything is scored.
[`pit_si_sdr`][kudio.core.separation.pit_si_sdr] takes the sources and the
outputs as `(sources, samples)` arrays, scores every pairing and keeps the
assignment with the highest mean SI-SDR, the criterion of utterance-level
permutation invariant training;[^upit] given the mixture, it gives the
improvement too. [`bss_eval`][kudio.core.separation.bss_eval] gives BSS
Eval's SDR, SIR and SAR,[^bss] which are mir_eval's `bss_eval_sources`
numbers unchanged.

```python
other = 0.01 * rng.standard_normal(4 * sr)       # a second talker, higher
other[16000:48000] += burst(2.0, 210)
sources = np.stack([clean, other])
mixture = sources.sum(axis=0)
hiss = 0.005 * rng.standard_normal((2, 4 * sr))
separated = np.stack([other + 0.3 * clean,       # in the wrong order, each
                      clean + 0.1 * other]) + hiss   # with some of the other

s = kudio.pit_si_sdr(sources, separated, mixture)
print("matched:", s.permutation)
for name, values in (("SI-SDR", s.si_sdr), ("SI-SDRi", s.si_sdri)):
    print(f"{name:8s}", "  ".join(f"{v:6.2f}" for v in values))

b = kudio.bss_eval(sources, separated, mixture, permutation=s.permutation)
for name, label in (("sdr", "SDR"), ("sir", "SIR"), ("sar", "SAR"), ("sdri", "SDRi")):
    print(f"{label:8s}", "  ".join(f"{v:6.2f}" for v in getattr(b, name)))
```

```text
matched: (1, 0)
SI-SDR    18.58    9.60
SI-SDRi   17.83   10.13
SDR       18.63    9.68
SIR       20.67    9.92
SAR       22.93   22.76
SDRi      17.76   10.06
```

`matched` says which output went with each source: here the second output
is the first talker. Every figure is in the order of the sources, and papers
report the mean over them, `np.mean(s.si_sdri)`, averaged over the test set.
Left to itself, `bss_eval` matches by the highest mean SIR, as mir_eval
does; `permutation=s.permutation` scores the whole table with one
assignment. SI-SDR takes both signals to zero mean first, as Conv-TasNet's
SI-SNR does; torchmetrics' SI-SDR does that only with `zero_mean=True`, so
an output with a DC offset scores differently there.

## Input a metric cannot score

Every metric on this page follows one rule. Input that is malformed raises
`ValueError`, with a message that starts with the function's name,
`llr(): ...`, and says what to do. Input that is well formed but leaves
nothing to score gives nan and one `UserWarning`,
`<function>(): <reason>; scored nan`, attributed to the line that called
kudio, also when the call went through `score`. `AudioEvaluate` leaves such
a nan out of the mean.

Every metric takes the same signals: one channel of real numbers, as a
vector, a column `(n, 1)` or a row `(1, n)`, with integer samples read on
their type's full scale, int16 over 32768 and int32 over 2^31, as soundfile
reads a file. int64 and uint64 are refused. No audio file holds them, numpy
gives them for a list of ints or for `astype(int)`, and on their full scale
16-bit samples sit 288 dB down, where the 1e-12 guards decide the score.
Complex, boolean and text arrays and masked samples are refused too. A rate
has to be one positive number; `np.array(16000)` and `16000.0` are 16000,
and `score` passes it on as it is given. A signal that peaks under one step
of 16-bit audio, -90.3 dBFS, is scored with one warning: int16 samples held
in an int32 array land there, and that far down the fixed floors of `snr`,
`si_sdr`, `sdi`, `segmental_snr`, `lsd`, `itakura_saito` and `wss` change
their scores.

| Metric | `ValueError` for | nan and a warning for |
|---|---|---|
| `snr`, `si_sdr`, `segmental_snr`, `sdi` | NaN or inf samples, two channels; for `segmental_snr`, a `frame_length` under 1 or a `hop_length` outside 1 to `frame_length` | an empty pair, a reference that is digital silence throughout (for `si_sdr`, any constant reference: silence, a DC offset alone, one sample), a pair too loud to square in float64; for `segmental_snr`, a pair shorter than one frame |
| `pesq` | NaN or inf, two channels, a rate other than 8 or 16 kHz, a mode and scale PESQ cannot give together (wideband at 8 kHz, a raw wideband score) | an empty side, a silent reference, a side under 0.25 s, no utterance found in the reference, an output at or near digital silence |
| `lsd` | NaN or inf samples, two channels, a rate that is not a positive number of Hz | an empty pair, a pair shorter than one frame (32 ms by default), a reference that is digital silence throughout, a pair too loud to square |
| `llr`, `itakura_saito`, `cepstral_distance`, `wss`, `fw_segmental_snr` | NaN or inf, two channels, a rate other than 8 or 16 kHz | a pair under 37.5 ms (300 samples at 8 kHz, 600 at 16 kHz), a silent reference |
| `composite` | as above, a `pesq=` outside its scale, `mode="wb", scale="raw"` | as above, in every field; CSIG, CBAK, COVL and PESQ alone where the `pesq` package cannot score the pair (under 0.25 s, no utterance found, a silent output) |
| `ncm` | NaN or inf, two channels, a rate other than 8 or 16 kHz | an empty pair, a pair under 62.6 ms (501 samples at 8 kHz, 1001 at 16 kHz), a silent reference |
| `csii` | NaN or inf, two channels, a rate under 8 kHz | every level: an empty pair, a pair under 45 ms (360 samples at 8 kHz, 720 at 16 kHz), a silent reference; one level: a level with fewer than two frames |
| `siib`, `siib_gauss` | NaN or inf, two channels, a rate that is not a whole number of Hz or is under 4000 Hz (a rate in kHz) | an empty pair, a pair under 5.46 s (87,401 samples at 16 kHz), under 436 frames of speech once the reference's silence is dropped, a reference that is digital silence or one value throughout, an output too loud to square |
| `stoi`, `estoi`, `sdr` in `score` | two channels | an empty pair; either signal digital silence throughout; a pair too loud to square; for STOI and ESTOI, under 25.6 ms, or under about 0.4 s of speech once pystoi drops the silent frames; for SDR, a pair `bss_eval_sources` scores as inf |
| `score` | NaN or inf, two channels, a length mismatch past the limit unless PESQ is the only metric asked for, a name it does not know | an empty pair, with one warning for all the metrics asked for |
| [`srmr`][kudio.core.srmr.srmr] | NaN or inf, more than one channel, a rate other than 8 or 16 kHz | an empty or silent clip, a clip shorter than one 256 ms window once its silence is cut |
| [`wada_snr`][kudio.core.wada.wada_snr] | NaN or inf, more than one channel, a rate that is not positive | an empty clip, a clip under 1 s, more than 1% digital silence in runs of 10 ms or more, an SNR outside the -20 to 100 dB its curve covers |
| [`dnsmos`][kudio.core.dnsmos.dnsmos], [`dnsmos_p808`][kudio.core.dnsmos.dnsmos_p808] | NaN or inf, more than one channel, a rate that is not positive | an empty clip |
| [`wer`, `cer`](task-metrics.md#word-and-character-error-rates) | a side that is not text, sides that do not pair up | a reference side with no words, or for `cer` no characters |
| [`eer`, `min_dcf`](task-metrics.md#speaker-verification-eer-minimum-dcf-and-the-det-curve) | NaN or inf scores, labels other than booleans or 0/1, masked trials, a `p_target` outside (0, 1) | an empty list, a list without a target or without a non-target trial |
| [the F0 measures](task-metrics.md#pitch-trackers-gpe-vde-ffe-fpe-and-the-rmse-in-cents) | tracks of different lengths, negative or infinite values, masked frames | empty tracks; GPE, FPE and the RMSE when no frame is voiced in both |

Where only the output is digital silence, wholly or in part, the metrics
score it rather than refuse it, since an output that drops out has a defect
the score should show. `si_sdr` gives an output that is constant throughout
0 dB, which is what its 1e-12 guards make of 0 / 0, and warns that it does.
PESQ, also inside `composite`, STOI, ESTOI and SDR are the exceptions the
table lists, since their packages cannot score a silent output; in
`AudioEvaluate` such a file then drops out of the means on both sides rather
than pulling them down. Every metric that takes a reference warns and
refuses as described under [When the lengths differ](#when-the-lengths-differ).
How the rule is tested is on
[How the metrics are checked](../validation.md#the-input-rule-and-the-import-without-optional-packages).

## A whole test set

For the table under a paper,
[`score_dataset`][kudio.core.dataset_score.score_dataset], or
[`kudio score`](../cli.md#score) from a shell, does the whole job in one call:
it pairs the files by id, runs any of `score`'s metrics and the ones that need
no reference, and gives each mean with its confidence interval, the number of
utterances, the [signature](#signing-a-result) and what to cite. From Python,
the run of [Command line](../cli.md#score) is the following; its `testset/` comes
from that page's [example files](../cli.md#the-example-files).

```python
import re

result = kudio.score_dataset("testset/enhanced", reference="testset/clean",
                             baseline="testset/noisy", metrics=["pesq", "stoi", "si_sdr"],
                             groups=re.compile(r"^(p\d+)_"))
print(result.to_markdown())          # or to_latex(), to_json()
result.save_rows("scores.csv")       # one row per utterance
```

`groups` takes a compiled pattern whose first group is the speaker, a dict
or a function from id to speaker; a plain string is read as the name of a
file of speakers, a Kaldi `utt2spk` or a CSV with `id` and `speaker` columns.
The table it prints, and what each column means, are on that page.

[`AudioEvaluate`][kudio.core.evaluator.AudioEvaluate] scores lists of files:
the clean references, the noisy inputs, and one list per method, paired by
position. `clean_files[i]`, `noisy_files[i]` and each method's `i`-th file
have to be the same utterance. It reports PESQ, STOI and SDR, whichever are
installed, and needs pandas (`kudio[data]`) for its tables.

```python
files = {"clean": [], "noisy": [], "logmmse": [], "wiener": []}
snr_of = {}
for snr_db in (0, 5, 10):
    mix = kudio.add_noise_snr(clean, noise, snr_db=snr_db, seed=0)
    takes = {"clean": clean, "noisy": mix,
             "logmmse": kudio.spectral_enhance(mix, sr, "logmmse"),
             "wiener": kudio.spectral_enhance(mix, sr, "wiener")}
    for name, wave in takes.items():
        path = f"eval/{name}_{snr_db}dB.wav"
        kudio.save_wave(path, wave, sr)          # creates eval/
        files[name].append(path)
    snr_of[files["noisy"][-1]] = snr_db

ev = kudio.AudioEvaluate(files["clean"], files["noisy"],
                         pesq_mode="nb", pesq_scale="raw")
ev.eval("logmmse", files["logmmse"])
ev.eval("wiener", files["wiener"])
print(ev.get_df())
```

```text
  Enh_method    STOI      SDR
0    logmmse  0.6002  13.1913
1     wiener  0.5990  13.3809
2   Baseline  0.6060   5.0117
```

The first `eval` also scores the noisy files once, against the same
references, and [`get_df`][kudio.core.evaluator.AudioEvaluate.get_df] puts
them last as the Baseline row. Each value is a mean over the files, rounded
to four decimals. A PESQ column joins the table when the `pesq` package is
installed; this run had none. Each pass also draws a tqdm bar on stderr,
which [Getting started](../getting-started.md) shows how to turn off.

When a metric has no score for one file, that file gets NaN in its column
and is left out of the means on both sides rather than taking the whole
column down; after each pass one warning per column names those files and
why. A file that cannot be scored at all, unreadable, at a different rate
from its reference, past the length limit or with NaN or inf in it, gets NaN
in every column and a warning naming it, and the rest of the list is still
scored. The Baseline row is averaged over
the files the last method scored, so for a method with NaN files, compare it
with Baseline in `per_file` rather than in this table.

[`per_file`][kudio.core.evaluator.AudioEvaluate.per_file] has one row per
method and file, Baseline included, with the `clean`, `noisy` and scored
`file` paths beside the scores, so the results can be broken down by anything
you know about each file:

```python
table = ev.per_file()
table["snr_db"] = table["noisy"].map(snr_of)
print(table.pivot(index="snr_db", columns="method", values="SDR").round(2))
```

```text
method  Baseline  logmmse  wiener
snr_db
0           0.01     9.21    9.67
5           5.01    13.33   13.51
10         10.02    17.04   16.96
```

Wiener leads at 0 and 5 dB, and log-MMSE at 10 dB. For a dataset built with
the [`Synthesizer`][kudio.core.synth.Synthesizer], join on the `noisy` column
with its manifest, which records the noise, SNR, room and link of every
mixture ([Building a noisy dataset](synthesis.md)). The join compares path
strings, so build the file lists from the manifest itself, with
`[p.noisy for p in pairs]` after
[`load_manifest`][kudio.core.dataset.load_manifest], rather than from a fresh
folder listing. Without pandas, `ev.rows` holds the method rows as plain
dicts. There is no SI-SDR column; for that, loop over the pairs with `score`.

To try kudio's own denoising methods side by side on one recording, with
SI-SDR and the same PESQ options, see
[`compare_enhancers`][kudio.enhance.spectral.compare_enhancers] on
[Denoising](enhancement.md). Every parameter of the functions on this page is
in the API reference under [Metrics and alignment](../reference/metrics.md).

## Confidence intervals and tests

A mean over a test set is one draw of utterances, and a difference between
two systems has to be set against how far that draw alone could move it.
[`paired_ci`][kudio.core.stats.paired_ci] gives a bootstrap interval for the
mean difference between two systems scored on the same utterances, and
[`permutation_test`][kudio.core.stats.permutation_test] its p-value. Pass the
speaker of each utterance as `groups`, and whole speakers are resampled, and
flipped, together. Below, `pesq_base` and `pesq_a` stand for per-utterance
PESQ of a baseline and a system, the `PESQ` column of
[`per_file`][kudio.core.evaluator.AudioEvaluate.per_file] for two methods;
they are drawn at random so that the example runs on its own.

```python
import numpy as np
import kudio

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

In a paper: system A scores 0.106 above the baseline in PESQ (95% CI 0.057
to 0.168, bootstrap over 20 speakers; sign-flip test by speaker,
p = 0.0007). Resampled by utterance, the interval is under a quarter as
wide, because it takes the 600 utterances for 600 independent draws when
they are 20 voices, and a voice the system suits it suits in all 30 of its
utterances. In
kudio's simulation of speakers with an intra-class correlation of 0.3,
utterance-level 95% intervals held the true mean about half the time, and
speaker-level ones 89% at 10 speakers and 92% at 30; see
[How the metrics are checked](../validation.md#confidence-intervals-and-the-sign-flip-test).
Under 10 groups `bootstrap_ci` and `paired_ci` warn that the interval means
little, and `permutation_test` warns once the groups are too few for any
p-value under 0.05, which for a two-sided test is 5 or fewer.
VoiceBank-DEMAND's test set has two speakers, and an interval over them can
only run between the two speakers' own means; report the number of speakers
with every interval.

The number in the table itself, one system's mean, gets its interval from
[`bootstrap_ci`][kudio.core.stats.bootstrap_ci], resampled the same way:

```python
for name, scores in (("baseline", pesq_base), ("system A", pesq_a)):
    m = kudio.bootstrap_ci(scores, groups=speakers)
    print(f"{name:8s} {m.estimate:.3f} [{m.low:.3f}, {m.high:.3f}]")
```

```text
baseline 2.578 [2.412, 2.712]
system A 2.684 [2.519, 2.853]
```

The two intervals share most of their width, and the gain is still clear.
Each of them carries how much the 20 voices differ, and the paired interval
leaves that out, since both systems were scored on the same voices. Whether
two systems differ is read from the interval of the difference, not from
whether their own intervals overlap.

The defaults are BCa intervals from 10,000 resamples with seed 0, so a rerun
gives the same numbers. With 13 groups or fewer, or 13 utterances when there
are no groups, the test runs through every sign pattern (8,192 at 13) and
its p-value is exact; with more it draws 10,000 at random, and no p-value
comes out under 1/10,001.

Several systems against one baseline are several tests, and
[`holm`][kudio.core.stats.holm] adjusts their p-values so that the chance of
any false finding among them stays at the level:

```python
pesq_b = pesq_base + np.repeat(rng.normal(0.03, 0.10, 20), 30) + rng.normal(0.0, 0.1, 600)
p = {name: kudio.permutation_test(scores, pesq_base, groups=speakers).pvalue
     for name, scores in (("A", pesq_a), ("B", pesq_b))}
print({name: round(value, 4) for name, value in p.items()})
print({name: round(value, 4) for name, value in kudio.holm(p).items()})
```

```text
{'A': 0.0007, 'B': 0.2932}
{'A': 0.0014, 'B': 0.2932}
```

An error rate is total errors over total reference words, not a mean of
per-utterance rates, so [`rate_ci`][kudio.core.stats.rate_ci] and
[`paired_rate_ci`][kudio.core.stats.paired_rate_ci] take the two counts per
utterance and add both up again in every resample. With transcripts the
counts come from [`word_errors`][kudio.core.wer.word_errors] run on each
pair: `c.errors` and `c.reference_length`. Given `totals`,
`permutation_test` reports the difference of the pooled rates, and
`alternative="less"` asks whether the system makes fewer errors:

```python
words = 5 + rng.poisson(10, 600)                   # reference words per utterance
hard = np.repeat(rng.beta(2, 14, 20), 30)          # each speaker's own error rate
errors_base = rng.binomial(words, hard)
errors_a = rng.binomial(words, 0.85 * hard)
wer = kudio.rate_ci(errors_base, words, groups=speakers)
gap = kudio.paired_rate_ci(errors_a, errors_base, words, groups=speakers)
test = kudio.permutation_test(errors_a, errors_base, totals=words,
                              groups=speakers, alternative="less")
print(f"baseline WER {wer.estimate:.2%} [{wer.low:.2%}, {wer.high:.2%}]")
print(f"A minus baseline {gap.estimate:+.2%} [{gap.low:+.2%}, {gap.high:+.2%}], "
      f"p = {test.pvalue:.4f}")
```

```text
baseline WER 13.20% [9.75%, 20.64%]
A minus baseline -2.11% [-3.68%, -1.22%], p = 0.0004
```

Each result carries what it was computed from: `n`, `n_groups`,
`n_resamples` and `method`, and for the test whether every sign pattern was
run through (`exact`). A NaN among the scores is refused: drop the
utterances a metric could not score from every list, `groups` included, and
say how many. An interval with no width, every resample giving the
estimate, comes with a warning: a WER of 0 on 500 utterances says nothing
about how far the WER could move on the next 500. Why BCa, how the grouped
test's null is set and how the p-value counts the observed pattern are in
the module's docstring, under [`kudio.core.stats`][kudio.core.stats] in the
API reference.

## Reporting and citing a metric

Every metric kudio computes has an entry in `kudio.METRIC_INFO`, under the
name `score` takes and under its function's name.
[`metric_info`][kudio.core.registry.metric_info] gives one:

```python
info = kudio.metric_info("csii_high")
print(info.function, info.field, info.better, info.range)
print(info.validation)
```

```text
kudio.csii high higher (0.0, 1.0)
validation.md#ncm-and-csii
```

`variants` names the settings that change the number, which is what a table
needs beside it, and `refs` the works it comes from.
[`cite`][kudio.core.registry.cite] turns a list of names into BibTeX: kudio's
own entry first, then each work once, in the order the metrics name them.
Names `score` takes and function names both work.

```python
print(kudio.cite(["pesq", "stoi"]))
```

The entries are the references these pages give in their footnotes, with
accents written as LaTeX spells them, so they work in BibTeX and biblatex
alike.

### Signing a result

Two tables that both say "PESQ 2.45" may not have computed the same thing. A
raw P.862 2.0 is a narrowband MOS-LQO of 1.63; the output in
[CSIG, CBAK and COVL](#csig-cbak-and-covl) scores 0.80 apart on COVL with one
PESQ or the other inside it, and VoiceBank-DEMAND's noisy test set 0.85 on
average; and a min DCF means nothing without its prior. Post found the same
of BLEU and wrote sacreBLEU, which prints a short signature with every
score.[^post] [`signature`][kudio.core.signature.signature] does that for
kudio: given the metrics and the settings they ran with, it writes one line
that names every setting that changes a number, as it was resolved, and the
version of every package that computed one. On the clip of
[A first score](#a-first-score):

```python
scores = kudio.score(clean, enhanced, sr, ["pesq", "stoi", "si_sdr"])
text = kudio.signature(scores.keys(), sr)
print(text)
```

```text
kudio:4.0.0|sr:16000|pair:aligned=no,cut=shorter,limit=0.5,whole=pesq|si_sdr|pesq:mode=wb,scale=lqo|stoi|pkg:pesq=0.0.4,pystoi=0.4.1
```

That is with pesq 0.0.4 and pystoi 0.4.1 installed. `mode=wb` is what
`pesq_mode='auto'` became at 16 kHz; a default is written out as what it came
to, so the line stays readable after a default changes. `pair` says the two
signals were not lined up with `kudio.align` first (pass `aligned=True` if
they were), that the metrics cut the pair to the shorter of the two and
refused a pair that would have left more than half of the longer one
unmeasured (`limit`, `MAX_LENGTH_MISMATCH`), and that PESQ got both whole.
The metrics come in the registry's order whatever order they were given in,
so one set of metrics and settings always gives one line.

A metric called on its own takes its keyword arguments in a dict, the same
dict the call took:

```python
print(kudio.signature({"eer": {}, "min_dcf": {"p_target": 0.01}}, n=4874))
```

```text
kudio:4.0.0|eer:method=linear|min_dcf:p_target=0.01,c_miss=1.0,c_fa=1.0|n:4874
```

`n` is how many items were scored, and `ci` which confidence interval was
computed over them and how: the function as `kind`, the method its result
reports, which is `'percentile'` where BCa had no interval to give, and the
level, resamples, seed and groups it was given. The interval of
[Confidence intervals and tests](#confidence-intervals-and-tests) signs like
this:

```python
print(kudio.signature(["pesq"], sr, n=ci.n,
                      ci=dict(kind="paired_ci", method=ci.method, confidence=0.95,
                              n_resamples=10000, seed=0, groups=speakers)))
```

```text
kudio:4.0.0|sr:16000|pair:aligned=no,whole=pesq|pesq:mode=wb,scale=lqo|pkg:pesq=0.0.4|n:600|ci:kind=paired_ci,method=bca,confidence=0.95,n_resamples=10000,seed=0,groups=20
```

For DNSMOS the line also names the model file and the polynomials that map
its outputs, each by 8 hex digits of a SHA-256, so runs on two copies of the
weights can be told apart. The versions and the model are read from the
machine it runs on, so I write the signature in the run that computed the
numbers. [`parse_signature`][kudio.core.signature.parse_signature] reads one
back into a dict, for comparing two field by field:

```python
print(kudio.parse_signature(text)["pesq"])
```

```text
{'mode': 'wb', 'scale': 'lqo'}
```

The signature goes under the table or in a footnote, and the references come
from `kudio.cite`.

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
