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
no list it runs every metric that can run on this machine, which here is all
of them but PESQ:

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

[Getting started](../getting-started.md) has why mir_eval is pinned below 0.9,
and shows
[`check_metrics_install`][kudio.core.evaluator.check_metrics_install], which
logs an install hint for each missing back-end once per process.
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
as one signal; score a channel at a time, or mix down first. Both signals also
have to be at the one `sr` you pass. `find_delay` cannot tell when they are
not, so bring them together with [`resample`][kudio.core.io.resample] first.

For a single pair of files, `kudio align clean.wav processed.wav --metrics`
prints the delay and the SI-SDR before and after lining them up, and exits 1
when the two do not look like the same recording (see
[Command line](../cli.md)).

## When the lengths differ

[`snr`][kudio.core.evaluator.snr], `si_sdr`, `segmental_snr` and `sdi` cut
both signals to the length they share, and so do `score` and `eval_metrics`
for every metric they run. The tail of the longer one is never compared, so a
difference in length is warned about on every call:

```python
kudio.snr(clean, clean[:60000])
```

```text
UserWarning: ref is 64000 samples and deg 60000: only the first 60000 samples are scored and the rest is ignored. Use kudio.align if the short one is delayed rather than just short, or trim the pair yourself to say which part you meant.
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
ValueError: ref is 64000 samples and deg 16000: comparing them leaves 48000 of the longer one's 64000 samples unmeasured (75.0%), past the 50% limit, so the score would be about the overlap rather than about the pair. Check that these two are the same utterance, line them up with kudio.align, or raise kudio.core.evaluator.MAX_LENGTH_MISMATCH to score them as they sit.
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
files it reads the same way, with both file names in the message, and so
does [`pesq`][kudio.core.evaluator.pesq] called on its own. The refusal is a
plain `ValueError`, so an `except` for
[`KudioError`][kudio.exceptions.KudioError] around a batch will not catch it.

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
:   Removes the mean of both signals and the best-fitting gain before
    comparing, so a level change or a flipped polarity costs nothing while
    filtering and delay still do.[^sisdr] It is the number I look at first for
    a denoiser, which should not lose points for coming back quieter.

`segsnr`, [`segmental_snr`][kudio.core.evaluator.segmental_snr]
:   SNR per frame, each frame held to [-10, 35] dB, then averaged. Through
    `score` the frames are always 512 samples with a hop of 256, which is
    32 ms at 16 kHz and 10.7 ms at 48 kHz; `segmental_snr` takes
    `frame_length=` and `hop_length=` to change them. Pauses count as much as
    speech: in the 5 dB mix, 116 of the 249 frames sit at the -10 dB floor,
    most of them in the room tone around the phrases, and that pulls the
    average to -2.1 dB. A perfect copy scores exactly 35.0, and a clip
    shorter than one frame scores NaN.

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
    filter from the reference to the output and counts as distortion only
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
for the same file. ITU-T P.862 scores narrowband speech at 8 or 16 kHz;
P.862.2 scores wideband speech at 16 kHz only. P.862 can be reported raw, on
-0.5 to 4.5, or mapped to MOS-LQO through P.862.1. P.862.2 defines only the
MOS-LQO. [`pesq`][kudio.core.evaluator.pesq] takes the bandwidth as `mode=`
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

## A whole test set

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

When a metric fails on one file, that file gets NaN with a logged warning and
is left out of the mean rather than taking the whole column down. A file that
cannot be scored at all, unreadable, at a different rate from its reference
or past the length limit, gets NaN in every column and a warning naming it,
and the rest of the list is still scored. The Baseline row is averaged over
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

[^sisdr]: J. Le Roux, S. Wisdom, H. Erdogan and J. R. Hershey, "SDR – half-baked or well done?", ICASSP 2019.
[^sdi]: J. Chen, J. Benesty, Y. Huang and S. Doclo, "New insights into the noise reduction Wiener filter", IEEE Trans. Audio, Speech, Lang. Process. 14(4), 2006.
[^stoi]: C. H. Taal, R. C. Hendriks, R. Heusdens and J. Jensen, "An algorithm for intelligibility prediction of time-frequency weighted noisy speech", IEEE Trans. Audio, Speech, Lang. Process., 2011. ESTOI: J. Jensen and C. H. Taal, "An algorithm for predicting the intelligibility of speech masked by modulated noise maskers", IEEE/ACM Trans. Audio, Speech, Lang. Process., 2016.
[^bss]: E. Vincent, R. Gribonval and C. Févotte, "Performance measurement in blind audio source separation", IEEE Trans. Audio, Speech, Lang. Process. 14(4), 2006.
