# Getting started

Installing kudio, one recording taken from a noisy file to a denoised one, and
what kudio raises, warns and logs when something is wrong.

## Installing

```bash
pip install kudio
```

The core install is numpy, scipy, soundfile, soxr, PyYAML and tqdm: files,
features, effects, loudness, denoising, dataset synthesis and every metric
except PESQ and DNSMOS. kudio needs Python 3.9 or newer; CI tests 3.9, 3.11
and 3.13 on Linux and Windows. `import kudio` imports no optional package, so
a missing extra shows up when you call the function that needs it.

### Extras

Quote the brackets, or zsh reads them as a file pattern:

```bash
pip install "kudio[audio,viz]"
```

| Extra | Installs | For |
|---|---|---|
| `audio` | sounddevice | [`record`][kudio.core.stream.record], [`StreamRecorder`][kudio.core.stream.StreamRecorder], [`play_audio`][kudio.core.stream.play_audio], [`list_devices`][kudio.util.check.list_devices]; on Linux it needs the system's PortAudio, `libportaudio2` on Debian and Ubuntu |
| `pesq` | pesq | PESQ (ITU-T P.862 and P.862.2), also inside [`composite`][kudio.core.quality.composite]; PyPI has it as source only, so it needs a C compiler, the MSVC Build Tools on Windows |
| `dnsmos` | onnxruntime | [`dnsmos`][kudio.core.dnsmos.dnsmos] and [`dnsmos_p808`][kudio.core.dnsmos.dnsmos_p808]; the weights are Microsoft's, so clone the DNS-Challenge repository and pass its DNSMOS folder as `model_dir=`, or set `KUDIO_DNSMOS_DIR` |
| `viz` | matplotlib | the plotting helpers in [`kudio.util.visual`][kudio.util.visual] |
| `data` | pandas, openpyxl | [`AudioEvaluate.get_df`][kudio.core.evaluator.AudioEvaluate.get_df], [`per_file`][kudio.core.evaluator.AudioEvaluate.per_file], [`data2xlsx`][kudio.util.conv.data2xlsx] |
| `compressed` | audioread | m4a, aac, webm and whatever else ffmpeg decodes ([Audio files](guide/audio-files.md#formats-soundfile-does-not-read)); ffmpeg has to be on the PATH |
| `eval` | nothing | an empty extra: STOI, ESTOI and SDR need nothing beyond the core |
| `all` | `audio`, `eval`, `dnsmos`, `viz`, `data`, `compressed` | every runtime extra except `pesq`, which needs a compiler |

`[dev]`, `[check]` and `[docs]` are for working on kudio itself.

## A first session

One noisy recording, checked, denoised, scored and saved. Run this block in
an empty folder first: it writes `clean.wav`, two voiced stretches with pauses
around them, and `noisy.wav`, the same under white noise at 2 dB SNR.

??? example "Making `clean.wav` and `noisy.wav`"

    ```python
    import numpy as np

    import kudio

    sr = 16000
    t = np.arange(int(4.5 * sr)) / sr


    def syllables(start, stop, f0):
        """Harmonics of f0 under a 4 Hz syllable envelope, between two times."""
        inside = (t >= start) & (t < stop)
        tone = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 12))
        envelope = 0.5 - 0.5 * np.cos(2 * np.pi * 4 * (t - start))
        return np.where(inside, tone * envelope / 5, 0.0)


    clean = (syllables(0.5, 1.75, 120) + syllables(2.5, 4.0, 150)).astype(np.float32)
    noise = np.random.default_rng(0).standard_normal(len(t)).astype(np.float32)
    noisy = kudio.add_noise_snr(clean, noise, snr_db=2, seed=0)

    kudio.save_wave("clean.wav", clean, sr)
    kudio.save_wave("noisy.wav", noisy, sr)
    ```

### Load and check

```python
import kudio

y, sr = kudio.file_load("noisy.wav")
print(kudio.audio_info("noisy.wav"))
report = kudio.audio_report(y, sr)
print(report)
for problem in report.problems():
    print(problem)
```

```text
noisy.wav: 4.500s, 16000 Hz, 1ch, PCM_16, peak 0.5430
4.50s @ 16000 Hz · peak -5.3 dBFS · -18.4 LUFS · SNR~9 dB · bandwidth 8000 Hz
estimated SNR around 9 dB — noise close to the content gives that, and so does a steady sound such as hum or a running machine, whose loud and quiet frames hardly differ; the estimate cannot tell the two apart
```

[`file_load`][kudio.core.io.file_load] returns float32 at the file's own rate,
or at `sr=`, mixed to mono unless `mono=False` asks for `(frames, channels)`.
[`audio_report`][kudio.core.report.audio_report] needs no clean reference. Its
SNR compares loud 30 ms frames with quiet ones, so a file mixed at 2 dB reads
about 9: it spots a noisy file, it does not measure the mix.

### Denoise

```python
denoised = kudio.spectral_enhance(y, sr)
```

[`spectral_enhance`][kudio.enhance.spectral.spectral_enhance] defaults to a
log-MMSE gain over an MCRA noise estimate and returns as many samples as it
was given.

![Frame levels of noisy.wav and denoised.wav over 4.5 seconds, with the 95th and 10th percentiles of the noisy levels marked 8.8 dB apart](images/start-levels-light.webp#only-light)
![Frame levels of noisy.wav and denoised.wav over 4.5 seconds, with the 95th and 10th percentiles of the noisy levels marked 8.8 dB apart](images/start-levels-dark.webp#only-dark)

The noise floor drops from -23.6 to -42.8 dBFS, the syllable peaks by about
1 dB. [Denoising](guide/enhancement.md) has the other gain rules and noise
estimators.

### Score against the clean signal

```python
ref, _ = kudio.file_load("clean.wav")
before = kudio.score(ref, y, sr, ["snr", "si_sdr", "segsnr"])
after = kudio.score(ref, denoised, sr, ["snr", "si_sdr", "segsnr"])
for name in before:
    print(f"{name:7s} {before[name]:6.2f} -> {after[name]:6.2f} dB")
```

```text
snr       2.00 ->  11.37 dB
si_sdr    1.99 ->  11.04 dB
segsnr   -3.56 ->   1.01 dB
```

Each call also warns that 105 of the reference's 280 frames are digital
silence, which segmental SNR scores at its -10 dB floor.
[`score`][kudio.core.evaluator.score] with no list runs `snr`, `si_sdr`,
`segsnr`, `sdi`, `stoi`, `estoi` and `sdr`, and `pesq` where `[pesq]` is
installed and the rate is 8 or 16 kHz. It compares sample *i* with sample *i*,
so [`align`][kudio.core.align.align] a pair that may be shifted:
[Measuring quality](guide/metrics.md).

### Save

```python
kudio.save_wave("denoised.wav", denoised, sr)
print(kudio.audio_info("denoised.wav"))
```

```text
denoised.wav: 4.500s, 16000 Hz, 1ch, PCM_16, peak 0.3796
```

[`save_wave`][kudio.core.io.save_wave] writes 16-bit PCM unless `subtype=` says
otherwise, clips past ±1.0 in the integer formats and refuses NaN or inf. From
a shell the session is `kudio report noisy.wav` and `kudio enhance noisy.wav
denoised.wav` ([Command line](cli.md)).

## Errors

kudio's own exceptions derive from [`KudioError`][kudio.exceptions.KudioError]:

| Class | Raised when |
|---|---|
| [`AudioIOError`][kudio.exceptions.AudioIOError] | a file cannot be read or written; [`UnsupportedFormatError`][kudio.exceptions.UnsupportedFormatError] is a subclass kudio does not raise itself |
| [`FeatureError`][kudio.exceptions.FeatureError] | a transform or measurement refuses its input: wrong shape, too short, an unknown method or parameter; also a `ValueError` |
| [`DeviceError`][kudio.exceptions.DeviceError] | no input device records, or sounddevice finds no PortAudio library |
| [`SynthesisError`][kudio.exceptions.SynthesisError] | [`Synthesizer`][kudio.core.synth.Synthesizer] will not empty a folder that holds its inputs, the working directory, your home directory or a filesystem root, or meets a silent file it cannot set an SNR with |
| [`DependencyError`][kudio.exceptions.DependencyError] | an optional package is missing; also an `ImportError`, and its message names the extra |

Outside `KudioError`:

- Argument checks raise `ValueError` or `TypeError`: an unknown metric, a
  stereo pair for a metric, a rate of 0. A missing input is `FileNotFoundError`.
- Every metric refuses NaN, inf and int64 samples with a `ValueError` that
  starts with its name, and scores nan with one `UserWarning` where there is
  nothing to score, such as an empty pair or a silent reference.
- A buffer that waits past its `timeout=` raises `TimeoutError`.
- [`play_audio`][kudio.core.stream.play_audio] returns a message instead of
  raising when it cannot play something, and `None` when it played.

## Warnings and logging

A `UserWarning` means something is wrong with the audio or the result: NaN in
a file, a clip pushed past full scale, two signals cut to a common length. It
shows with logging off. The log holds bookkeeping, on loggers below `kudio`,
and prints nothing until your program configures `logging`; after
`logging.basicConfig(level=logging.WARNING)`, asking
[`check_file`][kudio.core.io.check_file] for a name that is taken logs
`WARNING:kudio.core.io:Existing file will be overwritten: noisy.wav`.

The `kudio` command logs at `INFO`, `DEBUG` with `-v`. `TQDM_DISABLE=1` turns
off the progress bars of the batch helpers (tqdm 4.66 or newer).

Silent on purpose: the integer formats clip past ±1.0, so write
`subtype="FLOAT"` when a mix may run hot; [`pesq`][kudio.core.evaluator.pesq]
gives wideband MOS-LQO at 16 kHz unless `mode=` and `scale=` say otherwise;
everything random takes `seed=`, `None` by default, a fresh draw on each run.

The [User guide](guide/index.md) picks a page by the job, and the
[API reference](reference/index.md) has every parameter.
