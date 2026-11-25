# Command line

Installing kudio also installs a `kudio` command for the jobs that do not need
a script: is this take usable, how loud is it, where is the speech, what would
a phone line do to it, and the folder jobs of converting and building a noisy
dataset. Each subcommand calls the library and prints what it returns, and each
section below links the function that does the work, where every parameter is
described.

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
| | [`pitch`](#pitch) | What pitch is the voice on, and where is it voiced? | [`f0`][kudio.core.pitch.f0] |
| Converting and editing | [`convert`](#convert) | Resample or re-encode a file or a folder | [`convert_folder`][kudio.core.io.convert_folder] |
| | [`normalize`](#normalize) | Bring a file to a loudness or a peak | [`normalize_lufs`][kudio.core.loudness.normalize_lufs] |
| | [`trim`](#trim) | Cut the silence off both ends | [`trim_silence`][kudio.effects.silence.trim_silence] |
| Rooms and channels | [`room`](#room) | Measure a room, build one, or put a file in one | [`rt60`][kudio.core.room.rt60], [`rir`][kudio.core.room.rir] |
| | [`channel`](#channel) | What a phone line or a lossy link does to a file | [`telephone`][kudio.effects.channel.telephone], [`dropouts`][kudio.effects.channel.dropouts] |
| Building a noisy dataset | [`synth`](#synth) | A noisy corpus from a clean folder and a noise folder | [`Synthesizer`][kudio.core.synth.Synthesizer] |
| Denoising | [`enhance`](#enhance) | Denoise a file | [`spectral_enhance`][kudio.enhance.spectral.spectral_enhance] |
| | [`compare`](#compare) | Which denoiser suits this file? | [`compare_enhancers`][kudio.enhance.spectral.compare_enhancers] |
| Lining up two recordings | [`align`](#align) | How far apart are two versions, and what did that cost? | [`find_delay`][kudio.core.align.find_delay] |
| Scoring a test set | [`score`](#score) | How does a system do over a whole test set, and does it beat its baseline? | [`score_dataset`][kudio.core.dataset_score.score_dataset] |
| Audio devices | [`devices`](#devices) | Which audio devices are there? | [`list_devices`][kudio.util.check.list_devices] |

## What every command has in common

- Audio is read at the file's own rate, and as mono by most commands: a
  stereo file is averaged to one channel before anything is measured or
  written, so `convert` on a stereo file writes a mono one. `info` reports
  the channels as they are, and `loudness` and `normalize` keep them, since
  BS.1770 weights and sums the channels and a mono mix of a stereo file reads
  3 LU low.
- A single input can be anything libsndfile reads, FLAC as well as WAV, and
  with audioread and ffmpeg installed
  [m4a, aac, webm and the like](guide/audio-files.md#formats-soundfile-does-not-read).
  The folder modes and `synth` collect files ending in `.wav`, in every subfolder,
  whatever the case of the suffix, so a recorder's `ZOOM0001.WAV` is found on
  every system.
- Output is 16-bit PCM unless the command has a `--subtype`, and fixed-point
  output is clipped at full scale. A single output takes its format from its
  name, so `out.flac` is a FLAC file and cannot hold `--subtype FLOAT`; the
  folder modes always write `.wav`.
- A single output file is replaced without asking. `convert --recursive` never
  replaces anything: an output that exists already gets a new name instead
  (`mic_a2.wav`). `synth` goes the other way and empties its output folder
  first.
- Results go to standard output, together with the summary line under them and
  the one-line message a command prints when it finds nothing
  (`no speech detected`). Errors and refusals, log lines, warnings and progress
  bars go to standard error. So `kudio vad take.wav > spans.txt` keeps the
  spans and their summary, and a script should test the exit code rather than
  the file.
- `-v` turns kudio's logging up to DEBUG, shows a Python traceback for an
  error instead of the one-line message, and goes before the subcommand:
  `kudio -v report take.wav`. Placed after it, the argument parser rejects it.
  Other libraries' logging stays where it was.
- `kudio --version` prints the installed version, the line a bug report or a
  methods section needs.

## The example files

Every output on this page is what the command printed, run on a few synthetic
files. `take.wav` is ten seconds at 16 kHz: four voiced phrases on a gliding
pitch between 105 and 170 Hz, the third 12 dB quieter, with pauses and room
tone. `hot.wav` is the take five times louder and clipped, `excerpt.wav` the
four seconds from 4.2 s, and `silence.wav` three seconds of zeros. `noisy.wav`
adds white noise at 5 dB SNR, and `processed.wav` is `noisy.wav` denoised and
128 samples late. `raw/` holds three 44.1 kHz stereo 24-bit takes in two
subfolders, and `corpus/` three clean talkers and two noises. `testset/` holds
24 sentences of 12 talkers, `p01_001.wav` to `p12_002.wav`, clean and with
white noise at 5 dB SNR.

??? example "Make the example files"

    Run it in an empty folder. A few commands read what an earlier one wrote
    (`phone.wav` from `channel`, `corpus/mixed` from `synth`), so run them in
    page order.

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

These five read a file and leave it as it is. The ones that can find something
wrong exit 1 when they do, which is what makes them usable in a script; the
[last section](#exit-codes-and-shell-tests) has the full list.
[Inspecting a recording](guide/inspection.md) and
[Speech activity and pitch](guide/speech.md) explain what the numbers mean.

### info

What the header says, plus the largest absolute sample. This is the one
command that leaves the channels alone.

```bash
kudio info raw/day1/mic_a.wav
```

```text title="Output"
file      : raw/day1/mic_a.wav
samplerate: 44100 Hz
channels  : 2
duration  : 10.000 s (441000 samples)
encoding  : WAV / PCM_24
peak      : 0.2922
```

Of an m4a the encoding is `not read by soundfile, decoded with audioread`.

### report

A health check that needs no clean reference: level, clipping, DC offset,
silence, noise floor, an estimated SNR and the bandwidth the content fills. It
prints the one-line summary of an
[`AudioReport`][kudio.core.report.AudioReport], then the findings of
[`problems()`][kudio.core.report.AudioReport.problems], and exits 1 when there
are any.

```bash
kudio report hot.wav
```

```text title="Output"
10.00s @ 16000 Hz · peak 0.0 dBFS · -9.7 LUFS · SNR~33 dB · bandwidth 8000 Hz

· 3390 samples clipped (2.12% of the clip) — the recording was already too hot; no processing gets that back

(This is a technical check. A perceptual score with no reference needs a trained model — see kudio.dnsmos.)
```

On `take.wav` the middle line reads "Nothing wrong that a measurement can see."
and the exit code is 0. The last line is printed either way, because a clean
report says nothing about how the recording sounds; that needs
[`dnsmos`][kudio.core.dnsmos.dnsmos], which has no command of its own.

### loudness

Integrated loudness to BS.1770, the loudness range to EBU Tech 3342, the
loudest and quietest 3 s windows with the time they are centred on, and the
true peak, oversampled to at least 192 kHz before it is read. The quietest
window skips digital silence, since a pause is not a finding.

```bash
kudio loudness take.wav
```

```text title="Output"
integrated    -23.09 LUFS
range           6.68 LU
short-term    -22.82 LUFS at    3.40s  (loudest)
              -36.90 LUFS at    6.70s  (quietest)
true peak     -10.60 dBTP
```

| Option | Default | What it does |
|---|---|---|
| `--over-time` | off | Print the curve: one line per 100 ms, one `#` per 2 LU above -60 LUFS |
| `--momentary` | off | Use the 400 ms window instead of the 3 s one, for the extremes and the curve |
| `--target LUFS` | none | Exit 1 when the integrated loudness is more than 1 LU off it, or the file is silent |

`--over-time` is for finding where a take drifted without opening an editor.
On `excerpt.wav` the level sinks by 10 LU as the window moves off the second
phrase and onto the quiet third one:

```bash
kudio loudness excerpt.wav --over-time
```

```text title="Output"
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

Each line is stamped with the centre of its window, so with a 3 s window the
first one is at 1.50 s. A clip shorter than 3 s still gets its integrated
loudness and true peak, and the range line says it could not fill the window;
with `--momentary` it also gets its 400 ms extremes, and the range line says
it needs 3 s. Below 400 ms, one BS.1770 block, there is no integrated loudness
either, and the command stops with an error. A file of digital silence prints
`-inf` and says so. So does one that has something in it and never rises above
the -70 LUFS gate, and that line gives its peak. A stereo file is measured as
stereo.

`--target` checks a delivery level:

```bash
kudio loudness hot.wav --target -23
```

```text title="Output"
integrated     -9.70 LUFS
range           6.63 LU
short-term     -9.22 LUFS at    3.20s  (loudest)
              -22.92 LUFS at    6.70s  (quietest)
true peak       0.34 dBTP

13.30 LU over the -23 LUFS target
```

It exits 1. Its true peak is above 0 dBTP although no sample in the file is
above full scale: the clipped waveform overshoots between the samples.

### vad

The speech spans [`vad`][kudio.core.vad.vad] finds, in seconds. The threshold
is measured up from the file's own noise floor, so the same value works on a
quiet file and a hissy one. `--threshold-db` (default 8) is how far above that
floor a frame has to be: lower it when a quiet talker is missed, raise it when
bursts of room noise come through as speech. `--split-to DIR` also writes each
span as its own file, named `<stem>_001.wav` and on.

```bash
kudio vad take.wav
```

```text title="Output"
    1     0.550 -    2.460  (1.910s)
    2     2.950 -    5.270  (2.320s)
    3     5.980 -    7.460  (1.480s)
    4     8.150 -    9.560  (1.410s)
4 span(s), 7.12s of 10.00s (71%)
```

On `noisy.wav` it finds three spans, not four. The third phrase is 12 dB
under the rest, and at 5 dB SNR that puts it below the noise. A file with no
speech prints `no speech detected` and exits 1.

### pitch

A summary of the [`f0`][kudio.core.pitch.f0] contour: the median, the 5th to
95th percentile range in Hz and in semitones, and how much of the file is
voiced. Percentiles rather than the extremes, because one octave error would
set an extreme on its own.

```bash
kudio pitch take.wav --segments --labels voiced.txt
```

```text title="Output"
median 128.5 Hz  ·  111.2-158.9 Hz (6.2 semitones)  ·  71% voiced
    1     0.560 -    2.460  (1.900s)
    2     2.960 -    5.250  (2.290s)
    3     5.970 -    7.450  (1.480s)
    4     8.160 -    9.550  (1.390s)
wrote 4 label(s) to voiced.txt
```

`--labels` writes the voiced spans with
[`save_labels`][kudio.core.dataset.save_labels]. `voiced.txt` is an Audacity
label track (start, end, text, tab-separated), so the spans open on top of the
waveform in an editor; a `.txt` or `.tsv` name gives that format and any other
suffix gives JSON. With nothing voiced the
command exits 1: on the 3 s `silence.wav` it prints
`no voiced frames in 301 (65-400 Hz)`, 301 being the number of frames.

`--fmin` and `--fmax` (65 and 400 Hz) set the search range. `--hop` is in
samples, not milliseconds; without it the hop is 10 ms, 160 samples at 16 kHz.
`--frame-length` has to hold more than two periods of `--fmin`, or the command
stops with an error. The check is there because the alternative is quiet: the
pitches whose periods do not fit are lost and the rest look fine, which reads
as a property of the recording rather than a wrong setting. Left out, it is
2048 samples, or 4096 at 88.2 and 96 kHz, where two periods of 65 Hz need 2714
and 2954.

## Converting and editing

[Audio files](guide/audio-files.md) covers reading, writing and converting from
Python, and [Editing and effects](guide/editing.md) trimming and level.

### convert

Resample and re-encode. Without `--rate` the file keeps its rate, and
`--subtype` takes any soundfile subtype, `PCM_16` by default.

```bash
kudio convert raw/day1/mic_a.wav mic_a.wav --rate 16000
```

```text title="Output"
wrote mic_a.wav (16000 Hz, PCM_16)
```

`mic_a.wav` has one channel: the 44.1 kHz stereo original was averaged down on
the way in. With `--recursive` the two arguments are folders and the work goes
to [`convert_folder`][kudio.core.io.convert_folder], which mirrors the layout
below the source folder. `--lufs` loudness-normalises and `--trim-db` trims
silence, both after the resample, on a single file as well as a folder. A file
too quiet to have a loudness is written without the gain, and standard error
says so. A single file with no samples in it is an error, and nothing is
written.

```bash
kudio convert raw/ 16k/ --rate 16000 --recursive --lufs -23
```

```text title="Output"
3/3 file(s) -> 16k/
```

It exits 1 if any file failed and lists the first ten with the reason. Run it
a second time and nothing is replaced: `16k/day1/mic_a.wav` stays and the new
output is written as `16k/day1/mic_a2.wav`, which the log on standard error
reports. Empty the folder first when the point is to redo it.

### normalize

Scale a file to a loudness, `--lufs` (-23 when nothing is given), or to a
peak, `--peak`, but not both. The peak is a linear amplitude above 0 and at
most 1, such as 0.99, not dBFS; anything else stops with an error, and so does
a file of digital silence, which has no level to scale. With a loudness target
a file that never rises above the -70 LUFS gate is an error as well, since it
has no loudness; `--peak` still scales it. The channels are kept.
Loudness is the one to use before comparing two files by ear, since the louder
of two versions tends to win whatever else is different about them.

```bash
kudio normalize hot.wav hot_r128.wav --lufs -23
kudio loudness hot_r128.wav --target -23 > /dev/null && echo "on target"
```

```text title="Output"
wrote hot_r128.wav normalised to -23.0 LUFS
on target
```

Normalising applies a gain and nothing else. When a loudness target pushes the
peaks past full scale, [`normalize_lufs`][kudio.core.loudness.normalize_lufs]
says so on standard error and the 16-bit file is clipped; the command still
exits 0.

```bash
kudio normalize take.wav loud.wav --lufs -6
```

```text title="stderr"
warning: normalize_lufs: -6.0 LUFS puts the peak at 2.09 (6.4 dBFS over); the signal will clip when written to an integer format
```

`kudio report loud.wav` afterwards finds 12352 clipped samples, 7.72% of the
file. I keep the gain exact rather than limiting it, because a limiter would
change the loudness that was asked for.

### trim

Cut leading and trailing silence with
[`trim_silence`][kudio.effects.silence.trim_silence]. `--top-db` (default 30)
is measured down from the loudest frame, not from full scale. It has to be a
finite number above 0: at 0 the loudest frame is silence too. A file with no
samples in it stops with an error, and nothing is written.

```bash
kudio trim take.wav take_trim.wav --top-db 30
```

```text title="Output"
trimmed [9216:153088] -> take_trim.wav (143872 samples)
```

The numbers in brackets are the sample range kept. At `--top-db 40` the same
take comes back whole, `[0:160000]`: its room tone sits 32 to 35 dB under the
loudest frame, inside the 40 dB, so it counts as content. A fixed threshold
like this suits a quiet studio take. On a file with a real noise floor, trim
from Python with [`vad_trim`][kudio.core.vad.vad_trim], which sets its
threshold from the floor and keeps everything from the first speech span to
the last. [`kudio vad`](#vad) shows where that cut would fall, but no command
writes it.

## Rooms and channels

Two kinds of degradation that are not additive noise. A room is a
convolution; a phone line or a lossy link quantises and drops packets.
[Rooms and channels](guide/degradation.md) covers both.

### room

With a file and no other option, `room` measures it as an impulse response,
over the whole band, with ISO 3382-1's evaluation ranges. `--make` builds a synthetic room instead and prints the
measurement of what it built, the first two lines below; measuring that file
prints the same summary again with the six figures behind it. `--apply` puts a
file through a room. `--make` with a file as well is refused, since it reads
nothing.

```bash
kudio room --make room.wav --rt60 0.6 --drr 3 --seed 0
kudio room room.wav
```

```text title="Output"
wrote room.wav (0.90s at 16000 Hz)
T30 0.60 s (EDT 0.58 s)  ·  C50 +9.2 dB  ·  DRR +3.0 dB  ·  fit 1.000
T30 0.60 s (EDT 0.58 s)  ·  C50 +9.2 dB  ·  DRR +3.0 dB  ·  fit 1.000
T20            0.603 s
T30            0.598 s
EDT            0.583 s
C50            +9.24 dB
C80           +12.11 dB
DRR            +3.00 dB
```

[`rir`][kudio.core.room.rir] solves for the direct sound that gives the DRR
asked for, so measuring the room back with [`rt60`][kudio.core.room.rt60]
returns the numbers that built it. The file is 1.5 times the RT60 long. With
`--apply` the room is built at the input file's rate and the output keeps the
input's length and peak:

```bash
kudio room take.wav --apply take_room.wav --rt60 0.6 --drr 3 --seed 0
```

```text title="Output"
wrote take_room.wav — 0.6s room, DRR +3 dB, at 16000 Hz
T30 0.60 s (EDT 0.58 s)  ·  C50 +9.2 dB  ·  DRR +3.0 dB  ·  fit 1.000
```

`--rt60` (default 0.5 s) and `--drr` (default 0 dB) describe the room to
build. A DRR of +10 dB and up is a close microphone, 0 dB a metre or two away.
`--make` builds at `--rate` (16000 by default), and the same `--seed` builds
the same room.

Measuring something that is not an impulse response still produces numbers,
because backward integration makes any signal decay. The command checks the
decay before it trusts it and exits 1 when the straight-line fit is below
R² 0.9, or when EDT and T30 differ by more than a factor of two. A close
response whose direct sound alone falls 10 dB has no EDT, and is judged on its
fit:

```bash
kudio room take.wav
```

```text title="Output"
T30 41.05 s (EDT 60.73 s)  ·  C50 -15.5 dB  ·  DRR -24.5 dB  ·  fit 0.361  — but the decay is not straight, so read it with care
T20           48.208 s
T30           41.047 s
EDT           60.733 s
C50           -15.49 dB
C80           -14.74 dB
DRR           -24.50 dB
```

### channel

Degrade a file the way a transmission does, in the order a link does it:
band-limiting and companding, then requantising, then packet loss. The output
has the input's rate and length, and the summary compares the two.

```bash
kudio channel take.wav phone.wav --telephone
```

```text title="Output"
wrote phone.wav
  · telephone (mu_law)
bandwidth     8000 Hz ->    3797 Hz
si_sdr       -4.54 dB against the original
```

`--telephone` is [`telephone`][kudio.effects.channel.telephone]: 300 to
3400 Hz, down to 8 kHz, through G.711 mu-law and back up. Add `--codec a_law`
for the European companding curve. `kudio report phone.wav` then exits 1 with
one finding, "content stops at 3768 Hz, although a 16000 Hz file holds up to
8000 Hz", which is what gives a phone recording away.

Packet loss is [`dropouts`][kudio.effects.channel.dropouts]. `--loss` is the
fraction of 20 ms packets to drop, and that many are dropped exactly, not drawn
at that rate: 5% of 500 packets is 25. Dropped independently, most packets go
one at a time; `--burst N` asks for runs of N packets on average instead, the
way a real link loses them.

```bash
kudio channel take.wav lossy.wav --loss 0.05 --seed 0
kudio channel take.wav bursty.wav --loss 0.05 --burst 6 --seed 0
```

```text title="Output"
wrote lossy.wav
  · 25 packet(s) lost in 23 run(s) of 1.1, 0.50s, silence
bandwidth     8000 Hz ->    8000 Hz
si_sdr      +14.03 dB against the original
wrote bursty.wav
  · 25 packet(s) lost in 6 run(s) of 4.2, 0.50s, silence
bandwidth     8000 Hz ->    8000 Hz
si_sdr      +23.84 dB against the original
```

The line reports the runs that were made. Here they are 4, 6, 1, 1, 4 and 9
packets long, and a mean of six random lengths comes out at 4.2 rather than 6.
The almost 10 dB between the two SI-SDR lines is where the packets fell, not
what bursts do: the six runs took 0.4% of the take's energy and the 23 single
losses 3.8%, and over 30 seeds the bursty version lands anywhere from 9.8 to
26.3 dB. For the same reason, do not choose a `--conceal` mode by the SI-SDR
line. Whether repeating the last packet (`hold`) measures better or worse than
leaving a hole (`silence`) depends mostly on whether the voice's pitch period
happens to divide the packet length.

| Option | Default | What it does |
|---|---|---|
| `--telephone` | off | Band, rate and companding together |
| `--codec` | `mu_law` with `--telephone` | Which G.711 curve. With `--telephone` it picks the telephone's; alone it compands without the band-limiting |
| `--bits` | none | Requantise linearly to this many bits |
| `--loss`, `--packet-ms` | none, 20 ms | Fraction of packets lost, and their length |
| `--burst` | independent | Mean run length in packets, at least 1 |
| `--conceal` | `silence` | `silence` leaves a hole, `hold` repeats the last packet |
| `--seed` | none | The same seed drops the same packets |

They stack: `--codec a_law --bits 8 --loss 0.02` applies all three. With none
of them the command writes nothing and stops with exit 2.

## Building a noisy dataset

### synth

Mix every clean file with noise at the SNRs asked for, through
[`Synthesizer`][kudio.core.synth.Synthesizer].
[Building a noisy dataset](guide/synthesis.md) explains the class; this is the
same thing from a shell.

```bash
kudio synth --clean corpus/clean --noise corpus/noise --out corpus/mixed \
    --snr -5 0 5 --seed 17
```

```text title="Output"
synthesized 18 files -> corpus/mixed (seed 17)
```

`--mode inc`, the default, writes every clean file with every noise at every
SNR: three talkers, two noises and three SNRs make 18 files, named
`spk1_fan_n5.wav`, `spk1_fan_0dB.wav`, `spk1_fan_5dB.wav` and so on. `--mode reg`
writes one file per clean file and takes the noises and SNRs in turn.

| Option | Default | What it does |
|---|---|---|
| `--clean`, `--noise` | required | A folder, a `.wav`, or a `.txt` file with one path per line |
| `--out` | required | The output folder, emptied first |
| `--snr` | -5 0 5 | SNRs in dB, fractions such as 2.5 included |
| `--seed` | drawn and printed | The same seed rebuilds the same files |
| `--rt60` | none | Reverberation times in seconds, one room per value |
| `--drr` | 0 dB | Direct-to-reverberant ratio of those rooms, as in [`room`](#room) |
| `--targets` | off | Also write the reverberant speech without noise; needs `--rt60` |
| `--channel` | none | A link from [`CHANNEL_PRESETS`][kudio.effects.channel.CHANNEL_PRESETS], applied to every mixture |
| `--channel-loss` | the preset's | Replace that link's loss fraction; refused without `--channel` |
| `--channel-burst` | the preset's | Replace its mean run length; refused without `--channel` |
| `--manifest FILE` | none | One JSON record per output file |

!!! warning "`--out` is emptied before every run"
    Whatever is in the output folder is deleted first, so a rerun cannot leave
    files from an older run mixed in with the new ones. The command refuses
    when the folder is a link or a drive root, or holds one of the input files,
    the working directory, the home directory or a file that cannot be deleted,
    but anything else in it goes. It tries
    [what can be tried](guide/synthesis.md) without the samples first, and
    leaves the folder alone if that fails. Give every corpus a folder of its
    own.

Every file draws its noise crop, its room and its packet loss from the seed
and its own path, so the same command rebuilds the same files, byte for byte.
Without `--seed` a seed is drawn at random and printed at the end of the first
line; write it down, or pass `--seed` and keep it in the command.

Rooms and a link go in the order a recording meets them: the room acts on the
talker, the noise joins at the microphone, and the link carries the mixture.

```bash
kudio synth --clean corpus/clean --noise corpus/noise --out corpus/rooms \
    --snr 0 5 --mode reg --rt60 0.3 0.6 --targets \
    --manifest corpus/rooms.json --seed 17
```

```text title="Output"
synthesized 6 files -> corpus/rooms (seed 17)
rooms: 0.3s, 0.6s at DRR +0 dB — the SNR is against the reverberant speech, and the clean file is still the target
targets: 6 reverberant-clean file(s) -> corpus/rooms/TARGETS
manifest -> corpus/rooms.json
```

Each `--rt60` value multiplies the output count, in `reg` mode as well as
`inc`: three clean files became six. The room goes in the name
(`spk1_fan_0dB_rt300ms.wav`). `--targets` writes the reverberant speech to
`TARGETS/` under the same names, for a model that should remove the noise and
leave the room; without `--rt60` there is nothing to write and a warning says
so. [`load_manifest`][kudio.core.dataset.load_manifest] reads the manifest
back, one record per file with the paths, noise, SNR, RT60, link and target.

`--channel` takes `telephone`, `telephone_a`, `mu_law`, `a_law`, `voip`
(mu-law with 5% loss in runs of 4, concealed by holding) or `none`.
`--channel voip --channel-loss 0.1 --channel-burst 6` changes the loss and
keeps the rest, and the summary names that link `mu_law+loss10%x6-hold`. The
link is fixed for the whole run, so two links make two corpora;
[Rooms and channels](guide/degradation.md) says why.

The files are written as 16-bit PCM, and a mixture that lands past full scale
is clipped, with a warning on standard error that counts the files. From
Python, `Synthesizer` also takes `subtype="FLOAT"`.

## Denoising

[Denoising](guide/enhancement.md) explains what each gain rule and noise
estimator assumes and how to choose between them; `compare` is how to choose
on your own file.

### enhance

Denoise one file with
[`spectral_enhance`][kudio.enhance.spectral.spectral_enhance]. The defaults,
log-MMSE over an MCRA noise estimate, are the ones I would start from without a
reason to pick anything else.

```bash
kudio enhance noisy.wav denoised.wav
kudio enhance noisy.wav denoised_omlsa.wav \
    --method omlsa --noise quantile --set q=0.4
```

```text title="Output"
log-MMSE + mcra noise estimate -> denoised.wav
  noise floor -28.8 -> -48.1 dBFS
OM-LSA + quantile noise estimate -> denoised_omlsa.wav
  noise floor -28.8 -> -51.5 dBFS
```

`--method` is one of the seven gain rules in
[`METHODS`][kudio.enhance.spectral.METHODS] (`specsub`, `multiband`, `wiener`,
`mmse_stsa`, `logmmse`, `omlsa`, `spectral_gate`) and `--noise` one of the four
estimators in [`NOISE_ESTIMATORS`][kudio.enhance.spectral.NOISE_ESTIMATORS]
(`mcra`, `quantile`, `initial`, `minimum`). `--n-fft` (default 512) is the
frame length, with a hop of a quarter of it.

`--set NAME=VALUE` sets a parameter of the method and can be repeated; a name
the method does not have stops with an error that lists the ones it has, and
so does a value outside the range `METHODS` gives it (`--set alpha=1.5`:
"logmmse: alpha=1.5 is outside 0.5 to 0.999"). `q=0.4` above belongs to
OM-LSA, its prior probability that a bin holds no speech (default 0.30), and
not to the quantile estimator. The estimators' own settings, such as which
quantile of each bin's power `quantile` takes (0.15), cannot be reached from
the command line, only through `noise_kwargs` in `spectral_enhance`.

The output has the same length as the input, so it lines up with it sample for
sample. The second line is the noise floor before and after; a lower floor is
less residual noise, which is not the same as better speech. To find out
whether denoising helped at all, use [`compare`](#compare).

With `--recursive` the two arguments are folders, and
[`enhance_folder`][kudio.enhance.spectral.enhance_folder] denoises every
`.wav` below the first into the same layout below the second, with the same
`--method`, `--noise`, `--n-fft` and `--set`:

```bash
kudio enhance corpus/mixed corpus/denoised --recursive
```

```text title="Output"
18/18 file(s) -> corpus/denoised
  noise floor down 21.4 dB on average
```

It exits 1 if any file failed, and lists the first ten with the reason.

### compare

Run every method over one file and put the results in one table. Without a
reference the table has only the `ms` and floor columns, and a note under it
says they are residual-noise readings, not quality scores. With the clean
signal as `--reference`, the table is ranked by SI-SDR, and `dSNR` is the SNR
gained over the untouched input. A negative `dSNR` means the method made the
file worse than doing nothing, and if every row is negative the command says
so.

```bash
kudio compare noisy.wav --reference take.wav
```

```text title="Output"
method + noise                   ms  floor dBFS    SI-SDR    dSNR   PESQ   STOI
-------------------------------------------------------------------------------
multiband + mcra                 29       -47.7     12.59    7.75      —   0.54
mmse_stsa + mcra                108       -43.7     12.22    7.41      —   0.53
logmmse + mcra                   60       -48.1     11.97    7.03      —   0.52
wiener + mcra                    39       -53.0     11.45    6.49      —   0.52
omlsa + mcra                     76       -50.7     11.22    6.26      —   0.52
specsub + mcra                   31       -39.0     11.15    6.47      —   0.53
spectral_gate + mcra             44       -41.4      6.14    0.28      —   0.54
```

The input is synthetic, so read this for the shape of the table and not as a
ranking of the methods on speech. The lowest floor, `wiener`, is fourth here.
This machine had `kudio[eval]` and not `kudio[pesq]`, so the STOI column is
filled in and the PESQ one is not.

The noise estimator matters as much as the gain rule, so `--noises` sweeps it
too; `all` expands to all four. `--noise` holds it at one, `mcra` by default,
and passing both is refused with exit 2.

```bash
kudio compare noisy.wav --reference take.wav \
    --methods logmmse wiener --noises all
```

```text title="Output"
method + noise                   ms  floor dBFS    SI-SDR    dSNR   PESQ   STOI
-------------------------------------------------------------------------------
wiener + initial                 27       -45.7     15.50   10.62      —   0.53
logmmse + quantile               47       -48.6     14.95   10.01      —   0.53
logmmse + initial                54       -42.7     14.75    9.89      —   0.53
wiener + quantile                28       -53.2     14.75    9.79      —   0.52
logmmse + mcra                   61       -48.1     11.97    7.03      —   0.52
wiener + mcra                    38       -53.0     11.45    6.49      —   0.52
logmmse + minimum                86       -46.2      6.58    1.80      —   0.50
wiener + minimum                 64       -51.3      5.23    0.75      —   0.50
```

`initial` takes the noise from the first frames of the file, and `noisy.wav`
opens with 0.6 s of noise before anybody speaks, which is the case it is made
for. Cut the same pair to start at 1.0 s, inside the first phrase, and the two
`initial` rows drop to the bottom of the table at -2.89 and -3.44 dB `dSNR`,
worse than leaving the file alone.

| Option | Default | What it does |
|---|---|---|
| `--reference FILE` | none | The clean signal, resampled to the input's rate. It has to line up and cover the same span: run [`align`](#align) first |
| `--methods` | all seven | Gain rules to run |
| `--write-to DIR` | none | Write each result as `<stem>_<method>.wav`, or `<stem>_<method>_<noise>.wav` when several estimators run |
| `--no-metrics` | off | Skip PESQ and STOI even when installed |

`compare` checks the pair's lengths the way
[`si_sdr`][kudio.core.evaluator.si_sdr] does: with a small difference, SNR,
SI-SDR and STOI are scored over the shorter of the two with a warning and
PESQ gets the two whole, and a 3 s reference against the 10 s `noisy.wav`,
70% of it unmeasured, is refused. `align --write-to` writes
a pair of equal length.

`ms` is wall-clock time per run, taken after a warm-up so the first method is
not charged librosa's one-time compilation. STOI needs `kudio[eval]` and PESQ
`kudio[pesq]` ([Getting started](getting-started.md) says what each extra
installs); without them the column shows `—` and a warning on standard error
says what to install. The PESQ column is on kudio's default scale, MOS-LQO:
wideband at 16 kHz, narrowband at 8 kHz, and empty at any other rate. Papers
differ on which PESQ they quote, so check before setting this column beside
one; [Measuring quality](guide/metrics.md) has
[`pesq`][kudio.core.evaluator.pesq] with both scales.

## Lining up two recordings

### align

Measure how far one recording sits after another with
[`find_delay`][kudio.core.align.find_delay]. Every reference metric compares
sample *i* with sample *i*, and none of them can tell when the two files are
offset; they only report a bad score.

```bash
kudio align take.wav processed.wav --metrics --write-to aligned/
```

```text title="Output"
lags by 128 samples (8.0 ms), correlation 0.970
overlap     10.000s of 10.000s / 10.008s

si_sdr         -2.16 dB  as the files sit
               11.97 dB  aligned

wrote 2 file(s) to aligned
```

Eight milliseconds of offset cost 14 dB of SI-SDR here. `--write-to` writes
the two trimmed to the span they share, as `take_aligned.wav` and
`processed_aligned.wav`, both the same length. `lags` means the second file
starts after the first, `leads` before it, and `already aligned` that the best
lag is zero; a negative correlation is reported as polarity inverted rather
than corrected. When the two files have different rates the second is
resampled to the first's before anything is measured.

Cross-correlation always has a maximum, so a delay comes out of any two files.
The correlation decides whether to believe it: below 0.5
([`MIN_CORRELATION`][kudio.core.align.MIN_CORRELATION]) the command says the
files are unrelated and exits 1.

```bash
kudio align take.wav corpus/noise/fan.wav
```

```text title="Output"
leads by 11727 samples (732.9 ms), correlation 0.016
overlap     4.000s of 10.000s / 4.000s

these do not look like the same recording — the delay above does not mean anything
```

`--max-seconds` bounds the search either way, which keeps a long and repetitive
file from matching a passage half a minute away. Set it above the largest delay
you expect. Bounded below the real one, the search returns the best lag inside
the bound, and that can still pass the 0.5 check:
`kudio align take.wav processed.wav --max-seconds 0.005` (80 samples, under the
real 128) reports a lag of 2 samples at correlation 0.621 and exits 0.

## Scoring a test set

### score

Score every utterance of a test set and print the table a paper needs, with
[`score_dataset`][kudio.core.dataset_score.score_dataset]: each metric's mean
over the utterances with a 95% confidence interval, how many utterances it is
over, the [signature](guide/metrics.md#signing-a-result) that says how the
numbers were computed, and what to cite. Here the noisy files of `testset/` are
denoised first, then scored against the clean ones, with the noisy ones as the
baseline:

```bash
kudio enhance testset/noisy testset/enhanced --recursive
kudio score testset/enhanced --reference testset/clean --baseline testset/noisy \
    --metrics pesq stoi si_sdr --groups "^(p\d+)_" --per-item scores.csv --bib refs.bib
```

```text title="Output"
24/24 file(s) -> testset/enhanced
  noise floor down 9.7 dB on average
| Metric | Better | Baseline [95% CI] | Mean [95% CI] | Difference [95% CI] | p | p (Holm) | n | Not scored |
|---|---|---|---|---|---|---|---|---|
| PESQ (WB MOS-LQO) | higher | 1.26 [1.24, 1.27] | 1.66 [1.60, 1.72] | +0.40 [+0.34, +0.47] | 0.0005 | 0.0015 | 24 | 0 |
| STOI | higher | 0.720 [0.694, 0.740] | 0.732 [0.700, 0.758] | +0.012 [+0.005, +0.022] | 0.0181 | 0.0181 | 24 | 0 |
| SI-SDR (dB) | higher | 5.00 [4.99, 5.01] | 12.00 [11.90, 12.10] | +7.00 [+6.90, +7.10] | 0.0005 | 0.0015 | 24 | 0 |

Signature for the means and the baselines: kudio:4.0.0|sr:16000|pair:aligned=no,cut=shorter,limit=0.5,whole=pesq|si_sdr|pesq:mode=wb,scale=lqo|stoi|pkg:pesq=0.0.4,pystoi=0.4.1|n:24|ci:kind=bootstrap_ci,method=bca,confidence=0.95,n_resamples=10000,seed=0,groups=12
Signature for the differences: kudio:4.0.0|sr:16000|pair:aligned=no,cut=shorter,limit=0.5,whole=pesq|si_sdr|pesq:mode=wb,scale=lqo|stoi|pkg:pesq=0.0.4,pystoi=0.4.1|n:24|ci:kind=paired_ci,method=bca,confidence=0.95,n_resamples=10000,seed=0,groups=12
References: kudio, itut2001p862, itut2003p8621, itut2007p8622, itut2005p862amd2, taal2011stoi, leroux2019sdr, luo2019convtasnet
```

This run had pesq 0.0.4 and pystoi 0.4.1 installed, and the signature names
both; the `compare` tables above had pystoi alone, and the other examples
neither. The input is synthetic, so read the table for its shape.

The sides pair up by id. In a folder the id of a file is its path below the
folder without the extension, so `testset/clean/p01_001.wav` and
`testset/enhanced/p01_001.wav` are both `p01_001`, and a subfolder is part of
the id (`p232/001`). Every file in a format soundfile reads is taken, by its
suffix (`.wav`, `.flac`, `.ogg`, `.mp3`, `.aiff` and the rest of
`soundfile.available_formats()`, raw PCM aside), in every subfolder. A warning
names any other file whose name is no id of the folder, so that no utterance
leaves the set unsaid; a transcript `p01_001.txt` beside `p01_001.wav` is not
named. A file with two or more channels is averaged to one, as the other
commands read audio, and its row says so in `note`.
Instead of a folder, a side can be a list: one `id path` per line, as a Kaldi
`wav.scp` has it, or a `.csv` file with `id` and `path` columns. A relative
path in a list is read from the working directory. A list can name an m4a, which
audioread and ffmpeg decode; a folder is not searched for one. An id missing from one
side, an id given twice, two files that come to one id (`001.wav` and
`001.flac`) and a listed file that does not exist stop the command with exit 2
before anything is scored, with the first five ids named and the rest counted.
Scoring whatever happened to pair up would put a different test set under the
table's name.

`Mean` is over the utterances scored, `Baseline` is the same metric on the
baseline files, and `Difference` is the mean minus the baseline. With a
baseline all three are over the utterances both sides have a score for, so the
difference is exactly the one mean minus the other. Every interval is BCa from
10,000 resamples with seed 0, as
[Confidence intervals and tests](guide/metrics.md#confidence-intervals-and-tests)
explains. `--groups` gives the speaker of each utterance, here taken from the
id by the first group of the regular expression, and whole speakers are
resampled together. `p` is the two-sided sign-flip test of the difference,
with each speaker's utterances flipped together, and `p (Holm)` is that
p-value adjusted over the rows of the table. I adjust over the rows because
the table is read as one claim, that the system and its baseline differ, and
the claim stands if any row says so: what has to stay at 5% is the chance that
some row shows a difference that is not there. Whether a difference is for
the better is read from its sign and the `Better` column. The rows are scored
on the same utterances and move together, and Holm's procedure holds however
they depend on each other. A row with no p-value, a metric that scored no
utterance on both sides, is left out of the family, and a line under the table
says which. Several systems against one baseline, one run each, make a larger
family; adjust their p-values together with [`holm`][kudio.core.stats.holm].
A p-value is printed to four places, and to more where four would round it
onto 0.05, 0.01 or 0.001 without its being that: 4/4096, the Holm-adjusted
p-value of the run without a reference further down, prints as 0.00098.

Under the table come the signatures, each line saying what it covers: here
one for the means and the baselines (`kind=bootstrap_ci`) and one for the
differences (`kind=paired_ci`). A metric that has no score for some
utterances gets a line of its own, since its `n` is not the others', and so
does an interval where BCa gave way to the percentile one; an interval that
does not exist, over a single utterance, is signed without `ci`. Every line
says how the run treated its pairs, so a line for PESQ alone still has
`cut=shorter,limit=0.5` when other metrics cut the pairs. `References` lists
the BibTeX keys of kudio and of the metrics, and `--bib` writes the entries,
as [`cite`][kudio.core.registry.cite] gives them.

`--per-item` writes one row per utterance, CSV for `.csv` and JSON lines for
`.jsonl`, with the baseline's scores in the `baseline_` columns:

```bash
head -3 scores.csv
```

```text title="Output"
id,group,pesq,stoi,si_sdr,baseline_pesq,baseline_stoi,baseline_si_sdr,reason,note,baseline_reason,baseline_note
p01_001,p01,1.4695559740066528,0.6275048324562252,11.680727575582601,1.310567021369934,0.640674455569799,4.971942205323741,,,,
p01_002,p01,1.6462552547454834,0.6418690459730046,12.199555794752698,1.2882053852081299,0.6593455739888636,5.004082450104905,,,,
```

An utterance a metric cannot score, a clip under 0.25 s for PESQ or a silent
reference, has an empty cell there (null in JSON lines) and in `reason` the
warning [`score`][kudio.core.evaluator.score] gives for the pair, word for
word, with the baseline called `baseline` on its side. A pair that cannot be
scored at all,
a file that cannot be read or one far shorter than its reference, has the
error there instead. `note` keeps the warnings given for an utterance that was
still scored, such as a length difference cut away. Such an utterance is left
out of the means of the columns it has no score in, the table counts it under
`Not scored`, a warning on standard error names the first five, and the
command exits 1.

The set is scored at one rate, that of the references, read from their headers
before anything is scored. References at two rates are refused, and so is a
metric that is not defined at the rate, PESQ at 44.1 kHz say, both with exit 2.
A file scored at another rate than its reference is not resampled behind your
back: that utterance gets no score, and the reason says why. `--sr` resamples
every file to one rate as it is read, nothing is then refused for its rate,
and the signature names the resampler (`resample:soxr_hq`, with librosa and
soxr under `pkg`).

Without `--reference` only the metrics that need no reference can run:
dnsmos, dnsmos_p808, srmr and wada_snr. When `--metrics` is not given, that is
srmr and wada_snr at 8 and 16 kHz, and wada_snr alone at any other rate, since
SRMR is defined at those two.

```bash
kudio score testset/enhanced --baseline testset/noisy --groups "^(p\d+)_"
```

```text title="Output"
| Metric | Better | Baseline [95% CI] | Mean [95% CI] | Difference [95% CI] | p | p (Holm) | n | Not scored |
|---|---|---|---|---|---|---|---|---|
| SRMR | higher | 5.91 [5.52, 6.27] | 31.91 [29.59, 34.15] | +25.99 [+23.69, +28.07] | 0.0005 | 0.00098 | 24 | 0 |
| WADA-SNR (dB) | higher | 1.63 [1.51, 1.75] | 12.45 [12.13, 12.84] | +10.82 [+10.44, +11.21] | 0.0005 | 0.00098 | 24 | 0 |

Signature for the means and the baselines: kudio:4.0.0|sr:16000|srmr:fast=no,norm=no,min_cf=4.0,max_cf=128.0,drop_silence=yes|wada_snr|n:24|ci:kind=bootstrap_ci,method=bca,confidence=0.95,n_resamples=10000,seed=0,groups=12
Signature for the differences: kudio:4.0.0|sr:16000|srmr:fast=no,norm=no,min_cf=4.0,max_cf=128.0,drop_silence=yes|wada_snr|n:24|ci:kind=paired_ci,method=bca,confidence=0.95,n_resamples=10000,seed=0,groups=12
References: kudio, falk2010srmr, santos2014srmrnorm, kim2008wada
```

On these synthetic files SRMR comes out far above the 6.12 that SRMRToolbox's
own test recording gives; the example is there for the form of the command.
The two kinds mix:
`--metrics pesq srmr` scores PESQ on the pair and SRMR on the scored file
alone, whole, with `--align` or without, so a reference that cannot be read
costs PESQ its score and not SRMR. dnsmos gives three columns, SIG, BAK and
OVRL, and needs DNS-Challenge's models, found through `--dnsmos-dir` or
`$KUDIO_DNSMOS_DIR` as [`find_dnsmos_model`][kudio.core.dnsmos.find_dnsmos_model]
finds them. Each model is opened once before anything is scored, so one that
takes the other model's input, `sig_bak_ovr.onnx` given for dnsmos_p808 say,
stops the command with exit 2, and the personalised model in `pDNSMOS/`, which
dnsmos maps with the other model's polynomials, gets the warning
[`dnsmos`][kudio.core.dnsmos.dnsmos] gives for it.

`--align` lines each pair up with [`align`][kudio.core.align.align] first, and
the signature says `aligned=yes`; a pair it takes for two different recordings
(correlation under 0.5) gets no score, with the reason. A pair with NaN or inf
in it is refused as such before it is lined up. `--jobs N` scores N utterances
at a time, each in a process of its own, and gives the same rows in the same
order as one process. Each process gets an equal share of the CPU's threads
for BLAS, unless `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS` or
`MKL_NUM_THREADS` is set. Without that share, four processes on a 20-thread
CPU each start 20: SDR of the 24 files of `testset/` then took 95 s, against
2.6 s in one process and 1.9 s with four processes of 5 threads. For a given
number of BLAS threads the numbers are the same to the last bit; across
thread counts SI-SDR and SDR can differ by about 1e-14.

`--format latex` prints the table as a LaTeX `tabular` with booktabs rules
(`\usepackage{booktabs}`), and the signatures and a `\cite` of the keys under
it as comments. The first `score` command with `--format latex` and without
`--baseline` prints:

```text title="Output"
\begin{tabular}{lrr}
\toprule
Metric & Mean [95\% CI] & $n$ \\
\midrule
PESQ (WB MOS-LQO) $\uparrow$ & 1.66 [1.60, 1.72] & 24 \\
STOI $\uparrow$ & 0.732 [0.700, 0.758] & 24 \\
SI-SDR (dB) $\uparrow$ & 12.00 [11.90, 12.10] & 24 \\
\bottomrule
\end{tabular}
% signature: kudio:4.0.0|sr:16000|pair:aligned=no,cut=shorter,limit=0.5,whole=pesq|si_sdr|pesq:mode=wb,scale=lqo|stoi|pkg:pesq=0.0.4,pystoi=0.4.1|n:24|ci:kind=bootstrap_ci,method=bca,confidence=0.95,n_resamples=10000,seed=0,groups=12
% \cite{kudio,itut2001p862,itut2003p8621,itut2007p8622,itut2005p862amd2,taal2011stoi,leroux2019sdr,luo2019convtasnet}
```

STOI is printed to three places, as are the other measures on a scale that
ends at 1, and SDI; everything else to two. An interval that does not exist
prints as `[n/a]` in both formats. `--format json` gives the same summary
unrounded, with every interval's settings, the columns Holm's adjustment was
over and null for a number that does not exist. A warning about the intervals,
too few speakers for one say, names the columns and sides it is about.
Standard output holds the table alone, in whichever format, so
`> table.tex` keeps nothing else; the `wrote` lines go to standard error with
the progress bar and the warnings.

| Option | Default | What it does |
|---|---|---|
| `--reference REF` | none | The clean references: a folder, an `id path` list or a CSV |
| `--baseline BASE` | none | A third side scored against the same references; adds the difference and its test |
| `--metrics` | `score`'s set, or srmr and wada_snr | Names as [`score`][kudio.core.evaluator.score] takes them, and dnsmos, dnsmos_p808, srmr and wada_snr |
| `--pesq-mode`, `--pesq-scale` | `auto`, `lqo` | As [`pesq`][kudio.core.evaluator.pesq] takes them |
| `--sr` | the files' rate | Resample every file to this rate as it is read |
| `--align` | off | Line each pair up first |
| `--groups` | none | The speakers: a CSV with `id` and `speaker` columns, `id speaker` lines as in a Kaldi `utt2spk`, or a regular expression whose first group is taken from the id |
| `--jobs N` | 1 | Utterances scored at once, one process each |
| `--per-item PATH` | none | One row per utterance, `.csv` or `.jsonl` |
| `--format` | `markdown` | `markdown`, `latex` or `json` |
| `--bib PATH` | none | Write the BibTeX entries |
| `--confidence`, `--resamples`, `--seed` | 0.95, 10000, 0 | The intervals' level, resamples and seed; the test draws as many sign patterns where it cannot run through them all |
| `--dnsmos-dir DIR` | `$KUDIO_DNSMOS_DIR` | Where the DNSMOS models are |

## Audio devices

### devices

Lists the input and output devices, needs `kudio[audio]`, and always exits 0.
The two audio back-ends number devices independently, so it prints both lists
with a label saying which functions take which index: sounddevice indices go
to [`record`][kudio.core.stream.record], PyAudio indices to the streaming
classes. An index from one list means nothing in the other. Without the extra
it says what is missing:

```bash
kudio devices
```

```text title="Output"
sounddevice indices — use with kudio.record(device=...)
  unavailable: 'sounddevice' is required for this feature. Install it with: pip install "kudio[audio]"

PyAudio indices — use with kudio.Recorder / stream readers
  unavailable: 'PyAudio' is required for this feature. Install it with: pip install "kudio[audio]"
```

With the extra installed each device is one line: its index, input and output
channel counts, default rate and name, and in the PyAudio list its host API.
[Recording and live views](guide/live-audio.md) covers the rest.

## Exit codes and shell tests

The commands that can find something wrong exit 1 when they do:

| Command | Exits 1 when |
|---|---|
| `report` | It has any finding: clipping, a peak within 0.5 dB of full scale or past it, a peak under -30 dBFS, a DC offset above 0.01, content stopping below 90% of Nyquist, more than 60% silence, an estimated SNR under 10 dB, digital silence, or NaN and inf samples |
| `loudness --target` | The integrated loudness is more than 1 LU from the target, or the file is silent or never above the -70 LUFS gate |
| `vad` | No speech is found |
| `pitch` | No frame is voiced |
| `align` | The correlation is below 0.5: not the same recording |
| `room FILE` | The decay does not look like an impulse response |
| `convert --recursive`, `enhance --recursive` | Any file failed |
| `score` | Some utterance has no score in some column, on either side |

Exit 2 is an error, reported as one line on standard error. That covers usage
errors (an unknown option, a value of the wrong type, a missing argument,
`enhance --set` without an `=`, `compare` given both `--noise` and
`--noises`, `channel` with nothing to do, `score` given sides that do not pair
up), a file that cannot be read, and an argument the library refuses. For
`score`, an audio file that cannot be read is one utterance without a score,
which exits 1 as the table above says; a list or a groups file that cannot be
read exits 2. `-v` shows the traceback instead. Otherwise the
command exits 0, even when `compare` finds that no method beat the input or
`normalize` writes a clipped file.

That makes each check one line in a shell script. `phone.wav` is the file the
[`channel`](#channel) section wrote.

=== "bash"

    ```bash
    kudio report hot.wav > /dev/null || echo "needs another take"

    for f in take.wav hot.wav phone.wav silence.wav; do
        kudio report "$f" > /dev/null || echo "$f"
    done
    ```

=== "PowerShell"

    ```powershell
    kudio report hot.wav | Out-Null
    if ($LASTEXITCODE -ne 0) { "needs another take" }

    foreach ($f in "take.wav", "hot.wav", "phone.wav", "silence.wav") {
        kudio report $f | Out-Null
        if ($LASTEXITCODE -ne 0) { $f }
    }
    ```

```text title="Output"
needs another take
hot.wav
phone.wav
silence.wav
```

Three files, three different findings: clipping, a phone-line bandwidth, and
nothing in it at all. `||` fires on exit 2 as well, so
`kudio report missing.wav || echo "needs another take"` prints the message for
a file that does not exist; a script that has to tell a finding from an error
tests for 1 and 2 separately.
