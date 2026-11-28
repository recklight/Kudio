# Command line

Installing kudio installs a `kudio` command for the jobs that need no script.
Each subcommand calls the library and prints what it returns; the function in
the last column has every parameter.

```bash
kudio --help                        # every subcommand, one line each
kudio report --help                 # the options of one of them
python -m kudio.cli info take.wav   # when the kudio script is not on PATH
```

| | Command | What it answers | Built on |
|---|---|---|---|
| Inspecting a recording | [`info`](#info) | What is this file? | [`audio_info`][kudio.core.io.audio_info] |
| | [`report`](#report) | Is it technically sound? | [`audio_report`][kudio.core.report.audio_report] |
| | [`loudness`](#loudness) | How loud is it, and where did the level move? | [`loudness`][kudio.core.loudness.loudness], [`loudness_over_time`][kudio.core.loudness.loudness_over_time], [`true_peak`][kudio.core.loudness.true_peak] |
| | [`vad`](#vad) | Where is the speech? | [`vad`][kudio.core.vad.vad] |
| | [`pitch`](#pitch) | What pitch is the voice on? | [`f0`][kudio.core.pitch.f0] |
| Converting and editing | [`convert`](#convert) | Resample or re-encode a file or a folder | [`convert_folder`][kudio.core.io.convert_folder] |
| | [`normalize`](#normalize) | Bring a file to a loudness or a peak | [`normalize_lufs`][kudio.core.loudness.normalize_lufs] |
| | [`trim`](#trim) | Cut the silence off both ends | [`trim_silence`][kudio.effects.silence.trim_silence] |
| Rooms and channels | [`room`](#room) | Measure a room, build one, or put a file in one | [`rt60`][kudio.core.room.rt60], [`rir`][kudio.core.room.rir] |
| | [`channel`](#channel) | What a phone line or a lossy link does to a file | [`telephone`][kudio.effects.channel.telephone], [`dropouts`][kudio.effects.channel.dropouts] |
| Building a noisy dataset | [`synth`](#synth) | A noisy corpus from a clean folder and a noise folder | [`Synthesizer`][kudio.core.synth.Synthesizer] |
| Denoising | [`enhance`](#enhance) | Denoise a file or a folder | [`spectral_enhance`][kudio.enhance.spectral.spectral_enhance] |
| | [`compare`](#compare) | Which denoiser suits this file? | [`compare_enhancers`][kudio.enhance.spectral.compare_enhancers] |
| Lining up two recordings | [`align`](#align) | How far apart are two versions? | [`find_delay`][kudio.core.align.find_delay] |
| Scoring a test set | [`score`](#score) | How does a system do over a test set? | [`score_dataset`][kudio.core.dataset_score.score_dataset] |
| Audio devices | [`devices`](#devices) | Which audio devices are there? | [`list_devices`][kudio.util.check.list_devices] |

## What every command has in common

- Audio is read at the file's own rate, and as mono by most commands; `info`
  reports the channels as they are, and `loudness` and `normalize` keep them.
- A single input can be anything libsndfile reads, and with the `compressed`
  extra [m4a, aac and webm](guide/audio-files.md#formats-soundfile-does-not-read).
  The folder modes and `synth` collect `.wav` files in every subfolder.
- Output is 16-bit PCM unless the command has a `--subtype`. A single output
  takes its format from its name and is replaced without asking; `convert
  --recursive` gives an existing output a new name; `synth` empties its output
  folder first.
- Results go to standard output; errors, log lines, warnings and progress bars
  to standard error. Test the exit code, not the output.
- `-v` goes before the subcommand (`kudio -v report take.wav`): DEBUG logging
  and a traceback instead of a one-line error. `kudio --version` prints the
  installed version.

## The example files

Every output on this page is what the command printed on these synthetic files.
`take.wav` is ten seconds at 16 kHz, four voiced phrases with the third 12 dB
quieter; `hot.wav` is it five times louder and clipped, `excerpt.wav` four
seconds of it, `silence.wav` three seconds of zeros, `noisy.wav` it under white
noise at 5 dB SNR, and `processed.wav` that denoised and 128 samples late.
`raw/` holds three 44.1 kHz stereo takes, `corpus/` three talkers and two
noises, `testset/` 24 sentences of 12 talkers, clean and noisy.

??? example "Make the example files"

    Run it in an empty folder, then the commands in page order.

    ```python
    import numpy as np
    import kudio

    sr = 16000


    def phrase(seconds, f_start, f_end, level=1.0):
        """Voiced: 11 harmonics on a gliding pitch, four syllables a second."""
        t = np.arange(int(seconds * sr)) / sr
        phase = 2 * np.pi * np.cumsum(np.linspace(f_start, f_end, t.size)) / sr
        voice = sum(np.sin(k * phase) / k for k in range(1, 12))
        syllables = 0.5 - 0.5 * np.cos(2 * np.pi * 4 * t)
        edges = np.minimum(1.0, np.minimum(t, t[::-1]) / 0.03)    # no clicks
        return level * voice * syllables * edges / 6


    def take(seed=0):
        """Ten seconds: four phrases, the third 12 dB down, over room tone."""
        y = 0.003 * np.random.default_rng(seed).standard_normal(10 * sr)
        phrases = [(0.6, 1.8, 110, 140, 1.0), (3.0, 2.2, 150, 120, 1.0),
                   (6.0, 1.4, 115, 170, 0.25), (8.2, 1.3, 130, 105, 1.0)]
        for start, seconds, f_start, f_end, level in phrases:
            p = phrase(seconds, f_start, f_end, level)
            y[int(start * sr):int(start * sr) + p.size] += p
        return y.astype(np.float32)


    y = take()
    kudio.save_wave("take.wav", y, sr)
    kudio.save_wave("hot.wav", np.clip(5 * y, -1, 1), sr)       # recorded too hot
    kudio.save_wave("excerpt.wav", y[int(4.2 * sr):int(8.2 * sr)], sr)
    kudio.save_wave("silence.wav", np.zeros(3 * sr, np.float32), sr)

    hiss = np.random.default_rng(1).standard_normal(y.size).astype(np.float32)
    noisy = kudio.add_noise_snr(y, hiss, snr_db=5, seed=0)
    kudio.save_wave("noisy.wav", noisy, sr)
    denoised = kudio.spectral_enhance(noisy, sr)
    late = np.concatenate([np.zeros(128, np.float32), denoised])  # 8 ms late
    kudio.save_wave("processed.wav", late, sr)

    for i, name in enumerate(["day1/mic_a", "day1/mic_b", "day2/mic_a"]):
        mono = kudio.resample(take(seed=10 + i), sr, 44100)
        stereo = np.stack([mono, 0.8 * mono], axis=1)
        kudio.save_wave(f"raw/{name}.wav", stereo, 44100, subtype="PCM_24")

    for i in range(3):
        talker = 0.5 * take(seed=20 + i)
        kudio.save_wave(f"corpus/clean/spk{i + 1}.wav", talker, sr)
    t = np.arange(4 * sr) / sr
    fan = 0.1 * np.random.default_rng(5).standard_normal(t.size)
    hum = 0.1 * sum(np.sin(2 * np.pi * 50 * k * t) / k for k in (1, 2, 3))
    kudio.save_wave("corpus/noise/fan.wav", fan, sr)
    kudio.save_wave("corpus/noise/hum.wav", hum, sr)

    for k in range(12):                    # testset/: 12 talkers, two sentences each
        for n in (1, 2):
            rng = np.random.default_rng(100 * k + n)
            low = 95 + 9 * k
            voice = np.concatenate([phrase(1.2, low, low + 20 * n), np.zeros(sr // 4),
                                    phrase(1.0, low + 25, low)])
            clean = 0.5 * voice + 0.003 * rng.standard_normal(voice.size)
            noisy = kudio.add_noise_snr(clean, rng.standard_normal(clean.size),
                                        snr_db=5, seed=0)
            kudio.save_wave(f"testset/clean/p{k + 1:02d}_{n:03d}.wav", clean, sr)
            kudio.save_wave(f"testset/noisy/p{k + 1:02d}_{n:03d}.wav", noisy, sr)
    ```

## Inspecting a recording

### info

```bash
kudio info raw/day1/mic_a.wav
```

```text
file      : raw/day1/mic_a.wav
samplerate: 44100 Hz
channels  : 2
duration  : 10.000 s (441000 samples)
encoding  : WAV / PCM_24
peak      : 0.2922
```

The header, and the peak read a block at a time ([`audio_info`][kudio.core.io.audio_info]).

### report

```bash
kudio report hot.wav
```

```text
10.00s @ 16000 Hz · peak 0.0 dBFS · -9.7 LUFS · SNR~33 dB · bandwidth 8000 Hz

· 3390 samples clipped (2.12% of the clip) — the recording was already too hot; no processing gets that back

(This is a technical check. A perceptual score with no reference needs a trained model — see kudio.dnsmos.)
```

The [`audio_report`][kudio.core.report.audio_report] summary and its findings;
it exits 1 when there are any. The file is read as mono.

### loudness

```bash
kudio loudness take.wav
kudio loudness excerpt.wav --over-time
```

```text
integrated    -23.09 LUFS
range           6.68 LU
short-term    -22.82 LUFS at    3.40s  (loudest)
              -36.90 LUFS at    6.70s  (quietest)
true peak     -10.60 dBTP
integrated    -25.98 LUFS
range           5.98 LU
short-term    -26.47 LUFS at    1.50s  (loudest)
              -36.90 LUFS at    2.50s  (quietest)
true peak     -10.69 dBTP

     1.50s    -26.47  #################
     1.60s    -26.50  #################
     1.70s    -27.27  ################
     1.80s    -27.60  ################
     1.90s    -27.94  ################
     2.00s    -29.15  ###############
     2.10s    -29.19  ###############
     2.20s    -30.88  ##############
     2.30s    -31.63  ##############
     2.40s    -32.46  ##############
     2.50s    -36.90  ###########
```

Integrated loudness, loudness range and true peak, with the loudest and quietest
moments; `--over-time` adds the short-term curve, one line per 100 ms
(`--momentary` for 400 ms windows). `--target -23` exits 1 when the file is more
than 1 LU off. The channels are kept.

### vad

```bash
kudio vad take.wav
```

```text
    1     0.550 -    2.460  (1.910s)
    2     2.950 -    5.270  (2.320s)
    3     5.980 -    7.460  (1.480s)
    4     8.150 -    9.560  (1.410s)
4 span(s), 7.12s of 10.00s (71%)
```

The speech spans in seconds; `--split-to DIR` writes each to its own file,
`--labels FILE` an Audacity label track. Exits 1 when there is no speech.

### pitch

```bash
kudio pitch take.wav --segments --labels voiced.txt
```

```text
median 128.5 Hz  ·  111.2-158.9 Hz (6.2 semitones)  ·  71% voiced
    1     0.560 -    2.460  (1.900s)
    2     2.960 -    5.250  (2.290s)
    3     5.970 -    7.450  (1.480s)
    4     8.160 -    9.550  (1.390s)
wrote 4 label(s) to voiced.txt
```

The [`f0`][kudio.core.pitch.f0] summary; `--segments` adds the voiced spans,
`--labels` writes them, `--fmin` and `--fmax` set the search range. Exits 1 when
nothing is voiced.

## Converting and editing

### convert

```bash
kudio convert raw/day1/mic_a.wav mic_a.wav --rate 16000
kudio convert raw/ 16k/ --rate 16000 --recursive --lufs -23
```

```text
wrote mic_a.wav (16000 Hz, PCM_16)
3/3 file(s) -> 16k/
```

One file, or with `--recursive` a whole tree through
[`convert_folder`][kudio.core.io.convert_folder], mirrored below the output
folder: `--rate`, `--subtype`, `--lufs` or `--peak`, `--trim-db`. Exits 1 when a
file failed.

### normalize

```bash
kudio normalize hot.wav hot_r128.wav --lufs -23
kudio normalize take.wav loud.wav --lufs -6
```

```text
wrote hot_r128.wav normalised to -23.0 LUFS
wrote loud.wav normalised to -6.0 LUFS
warning: normalize_lufs: -6.0 LUFS puts the peak at 2.09 (6.4 dBFS over); the signal will clip when written to an integer format
```

`--lufs` (BS.1770 loudness) or `--peak`; the channels are kept. A gain that
pushes samples past full scale is reported, and the 16-bit file is clipped, so
write `--subtype FLOAT` or aim lower.

### trim

```bash
kudio trim take.wav take_trim.wav --top-db 30
```

```text
trimmed [9216:153088] -> take_trim.wav (143872 samples)
```

Cuts the silence off both ends with [`trim_silence`][kudio.effects.silence.trim_silence]
at `--top-db` (30) under the loudest frame.

## Rooms and channels

### room

```bash
kudio room --make room.wav --rt60 0.6 --drr 3 --seed 0
kudio room room.wav
kudio room take.wav --apply take_room.wav --rt60 0.6 --drr 3 --seed 0
```

```text
wrote room.wav (0.90s at 16000 Hz)
T30 0.60 s (EDT 0.58 s)  ·  C50 +9.2 dB  ·  DRR +3.0 dB  ·  fit 1.000
T30 0.60 s (EDT 0.58 s)  ·  C50 +9.2 dB  ·  DRR +3.0 dB  ·  fit 1.000
T20            0.603 s
T30            0.598 s
EDT            0.583 s
C50            +9.24 dB
C80           +12.11 dB
DRR            +3.00 dB
wrote take_room.wav — 0.6s room, DRR +3 dB, at 16000 Hz
T30 0.60 s (EDT 0.58 s)  ·  C50 +9.2 dB  ·  DRR +3.0 dB  ·  fit 1.000
```

`kudio room FILE` measures an impulse response with [`rt60`][kudio.core.room.rt60]
and exits 1 when the decay does not look like one; `--make` builds a room with
[`rir`][kudio.core.room.rir]; `--apply` puts a file through a synthetic room
built from `--rt60`, `--drr` and `--seed`.

### channel

```bash
kudio channel take.wav phone.wav --telephone
kudio channel take.wav bursty.wav --loss 0.05 --burst 6 --seed 0
```

```text
wrote phone.wav
  · telephone (mu_law)
bandwidth     8000 Hz ->    3797 Hz
si_sdr       -4.54 dB against the original
wrote bursty.wav
  · 25 packet(s) lost in 6 run(s) of 4.2, 0.50s, silence
bandwidth     8000 Hz ->    8000 Hz
si_sdr      +23.84 dB against the original
```

`--telephone`, `--codec`, `--bits`, `--loss`, `--burst`, `--conceal` and
`--packet-ms` build one link, applied in the order a link applies them, and the
command prints what it did, including the packets it lost.

## Building a noisy dataset

### synth

```bash
kudio synth --clean corpus/clean --noise corpus/noise --out corpus/mixed --snr -5 0 5 --seed 17
```

```text
synthesized 18 files -> corpus/mixed (seed 17)
INFO kudio.core.synth: ==== Audio synthesizing ====
INFO kudio.core.synth: Clean: 3, Noise: 2, SNR: [-5.0, 0.0, 5.0]
INFO kudio.core.synth: 18 noisy files from 3 SNRs, 2 noises and 1 room(s)
```

A [`Synthesizer`][kudio.core.synth.Synthesizer] run: `--mode` (`inc` by
default), `--snr`, `--rt60`, `--drr`, `--targets`, `--channel`, `--subtype`,
`--rate` and `--manifest FILE`. It empties `--out` under the guards on
[Building a noisy dataset](guide/synthesis.md) and prints the seed it used.

## Denoising

### enhance

```bash
kudio enhance noisy.wav denoised.wav
kudio enhance noisy.wav denoised_omlsa.wav --method omlsa --noise quantile --set q=0.4
```

```text
log-MMSE + mcra noise estimate -> denoised.wav
  noise floor -28.8 -> -48.1 dBFS
OM-LSA + quantile noise estimate -> denoised_omlsa.wav
  noise floor -28.8 -> -51.5 dBFS
```

[`spectral_enhance`][kudio.enhance.spectral.spectral_enhance] on one file, or with
`--recursive` on a folder through [`enhance_folder`][kudio.enhance.spectral.enhance_folder].
`--set NAME=VALUE` passes a method parameter. The floor before and after is
printed. Exits 1 when a file of a folder failed.

### compare

```bash
kudio compare noisy.wav --reference take.wav --methods logmmse wiener --noises all
```

```text
method + noise                   ms  floor dBFS    SI-SDR    dSNR   PESQ   STOI
-------------------------------------------------------------------------------
wiener + initial                 23       -45.7     15.50   10.62      —   0.53
logmmse + quantile               46       -48.6     14.95   10.01      —   0.53
logmmse + initial                58       -42.7     14.75    9.89      —   0.53
wiener + quantile                26       -53.2     14.75    9.79      —   0.52
logmmse + mcra                   57       -48.1     11.97    7.03      —   0.52
wiener + mcra                    35       -53.0     11.45    6.49      —   0.52
logmmse + minimum                83       -46.2      6.58    1.80      —   0.50
wiener + minimum                 54       -51.3      5.23    0.75      —   0.50
WARNING kudio.core.evaluator: PESQ unavailable: pip install "kudio[pesq]" (builds from source: needs a C compiler -- on Windows, the MSVC Build Tools)
```

[`compare_enhancers`][kudio.enhance.spectral.compare_enhancers]: every method,
or `--methods`, over the default estimator, `--noise`, or `--noises all`, best
first when there is a reference. `--write-to DIR` writes each output.

## Lining up two recordings

### align

```bash
kudio align take.wav processed.wav --metrics --write-to aligned/
kudio align take.wav corpus/noise/fan.wav
```

```text
lags by 128 samples (8.0 ms), correlation 0.970
overlap     10.000s of 10.000s / 10.008s

si_sdr         -2.16 dB  as the files sit
               11.97 dB  aligned

wrote 2 file(s) to aligned
leads by 11727 samples (732.9 ms), correlation 0.016
overlap     4.000s of 10.000s / 4.000s

these do not look like the same recording — the delay above does not mean anything
```

[`find_delay`][kudio.core.align.find_delay] and [`align`][kudio.core.align.align]:
the delay, the correlation and, with `--metrics`, SI-SDR before and after;
`--write-to DIR` writes the aligned pair. Exits 1 when the two do not look like
the same recording.

## Scoring a test set

### score

```bash
kudio enhance testset/noisy testset/enhanced --recursive
kudio score testset/enhanced --reference testset/clean --baseline testset/noisy --metrics stoi si_sdr --groups "^(p\d+)_" --per-item scores.csv
```

```text
24/24 file(s) -> testset/enhanced
  noise floor down 9.7 dB on average
| Metric | Better | Baseline [95% CI] | Mean [95% CI] | Difference [95% CI] | p | p (Holm) | n | Not scored |
|---|---|---|---|---|---|---|---|---|
| STOI | higher | 0.720 [0.694, 0.740] | 0.732 [0.700, 0.758] | +0.012 [+0.005, +0.022] | 0.0181 | 0.0181 | 24 | 0 |
| SI-SDR (dB) | higher | 5.00 [4.99, 5.01] | 12.00 [11.90, 12.10] | +7.00 [+6.90, +7.10] | 0.0005 | 0.00098 | 24 | 0 |

Signature for the means and the baselines: kudio:5.0.0|sr:16000|pair:aligned=no,cut=shorter,limit=0.5|si_sdr|stoi|n:24|ci:kind=bootstrap_ci,method=bca,confidence=0.95,n_resamples=10000,seed=0,groups=12
Signature for the differences: kudio:5.0.0|sr:16000|pair:aligned=no,cut=shorter,limit=0.5|si_sdr|stoi|n:24|ci:kind=paired_ci,method=bca,confidence=0.95,n_resamples=10000,seed=0,groups=12
References: kudio, taal2011stoi, leroux2019sdr, luo2019convtasnet
wrote 24 row(s) to scores.csv
```

[`score_dataset`][kudio.core.dataset_score.score_dataset] pairs the files by id,
from folders, `id path` lists or CSV files, and prints the table a paper needs:
each mean with its 95% BCa interval over speakers (`--groups`, a pattern, a
Kaldi `utt2spk` or a CSV), the difference from `--baseline` with its interval and
sign-flip p-value, Holm-adjusted across metrics, the
[signature](guide/metrics.md#signing-a-result) and the BibTeX keys; `--bib FILE`
writes the entries, `--per-item FILE` one row per utterance, `--format` gives
`latex` or `json`. Without `--reference`, only the metrics that need none run.
Exits 1 when some utterance has no score in some column.

## Audio devices

### devices

```bash
kudio devices
```

Every audio device with its index, its host API, and its input and output
channels; the index is what `device=` takes ([Recording and live views](guide/live-audio.md)).

## Exit codes and shell tests

| Command | Exits 1 when |
|---|---|
| `report` | it has any finding |
| `loudness --target` | the loudness is more than 1 LU off, or the file never rises above the -70 LUFS gate |
| `vad` | no speech is found |
| `pitch` | no frame is voiced |
| `align` | the correlation is below 0.5 |
| `room FILE` | the decay does not look like an impulse response |
| `convert --recursive`, `enhance --recursive` | any file failed |
| `score` | some utterance has no score in some column |

Exit 2 is an error, one line on standard error: a usage error, a file that cannot
be read, an argument the library refuses. Otherwise the command exits 0. A check
is one line in a shell script; `||` fires on exit 2 as well, so test for 1 and 2
separately when it matters:

```bash
kudio report hot.wav > /dev/null || echo "needs another take"
```
