# Kudio

[![PyPI](https://img.shields.io/pypi/v/kudio.svg)](https://pypi.org/project/kudio/)
[![Python](https://img.shields.io/pypi/pyversions/kudio.svg)](https://pypi.org/project/kudio/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE.txt)

**Lightweight, composable audio toolkit for acoustic inspection.**

Kudio gives you audio I/O, streaming/recording, feature extraction, noisy-data
synthesis, effects & augmentation, enhancement, and quality metrics — as small
composable functions with a clean, typed API.

The core stays light (numpy + soundfile + librosa); heavy or niche features
(plotting, dataframes, PESQ/STOI, live audio) are opt-in extras.

## Install

```bash
pip install kudio            # core: I/O, features, effects, synthesis, metrics
pip install kudio[audio]     # + live capture / playback (PyAudio, sounddevice)
pip install kudio[eval]      # + PESQ / STOI / SDR back-ends
pip install kudio[viz]       # + plotting helpers (matplotlib)
pip install kudio[data]      # + Excel / dataframe export (pandas, openpyxl)
pip install kudio[all]       # everything
```

Requires **Python 3.9+** (tested on CPython 3.9–3.13).

> The Python badge above reflects the version currently published on PyPI; it
> updates automatically once v3.0 is released.

## Quick start

```python
import kudio

# --- I/O (soundfile-backed; float32) --------------------------------------
y, sr = kudio.file_load("clip.wav")             # native rate, no resample
kudio.save_wave("out.wav", y, sr, subtype="PCM_24")   # 16/24-bit or FLOAT

# --- features -------------------------------------------------------------
spec = kudio.waveform_to_spectrogram(y, norm=True)    # log-power spectrogram
feat = kudio.mfcc(y, sr, n_mfcc=40)                    # (1, frames, 40)
mel  = kudio.melspectrogram("clip.wav", n_mels=64)

# --- effects & augmentation ----------------------------------------------
clip, (a, b) = kudio.trim_silence(y, top_db=30)       # isolate the event
segments      = kudio.split_on_silence(y)
aug = kudio.pitch_shift(y, sr, n_steps=2)
aug = kudio.add_noise_snr(y, noise, snr_db=5, seed=0) # reproducible
spec = kudio.spec_augment(spec, freq_mask_width=8, time_mask_width=16, seed=0)

# --- noisy-data synthesis -------------------------------------------------
syx = kudio.Synthesizer("data/clean", "data/noise",
                        out_path="data/mixed", snr_ratio=(-5, 0, 5))
syx.syn(mode="inc", seed=17)                          # reproducible

# --- metrics (pure numpy, no extra deps) ----------------------------------
print(kudio.si_sdr(ref, est), kudio.snr(ref, est), kudio.segmental_snr(ref, est))

# --- recording (needs kudio[audio]) --------------------------------------
wave = kudio.record(seconds=3, sr=16000)
```

## Command line

```bash
kudio devices                                  # list audio devices
kudio info clip.wav                            # sr / channels / duration / peak
kudio convert in.wav out.wav --rate 16000 --subtype PCM_16
kudio synth --clean C --noise N --out O --snr -5 0 5 --seed 17
kudio trim in.wav out.wav --top-db 30
```

## Modules

| Module | What's inside |
|---|---|
| `kudio.core.io` | `file_load`, `save_wave`, `check_input`, `load_waves`, `copy_waves` |
| `kudio.core.feature` | `waveform_to_spectrogram`, `spectrogram_to_waveform`, `mfcc`, `melspectrogram`, ... |
| `kudio.effects` | `trim_silence`, `split_on_silence`, `time_stretch`, `pitch_shift`, `add_noise_snr`, `reverb`, `spec_augment` |
| `kudio.core.synth` | `Synthesizer` (SNR mixing, seedable) |
| `kudio.core.evaluator` | `si_sdr`, `snr`, `segmental_snr` (dep-free); `AudioEvaluate` (PESQ/STOI/SDR) |
| `kudio.core.stream` | `record`, `play_audio`, `Recorder`, `LocalStreamReader`, `RemoteStreamReader` |
| `kudio.util` | `CheckDevice`, `map_waves`, colored console helpers, timers |

Everything commonly used is importable straight from the top level (`kudio.…`).

## Errors

All library errors derive from `kudio.KudioError` (`AudioIOError`,
`FeatureError`, `DeviceError`, `SynthesisError`, `DependencyError`), so you can
catch them in one place. Missing an optional extra raises a `DependencyError`
that tells you exactly what to install.

## Logging

kudio uses the standard `logging` module (loggers named `kudio.*`) and prints
nothing by default:

```python
import logging; logging.basicConfig(level=logging.INFO)
```

## Migrating from v2

v3 is a cleanup release. The short cryptic names still work but now emit a
`DeprecationWarning` — switch to the canonical names:

| v2 (deprecated) | v3 canonical |
|---|---|
| `w2s`, `wavform2spec` | `waveform_to_spectrogram` |
| `spec2wavform` | `spectrogram_to_waveform` |
| `f2s`, `wav2spec` | `file_to_spectrogram` |
| `s2w`, `spec2wav` | `save_spectrogram_as_wave` |
| `w2mfcc`, `wav2mfcc` | `mfcc` |
| `concat_mfcc_` | `mfcc_from_files` |
| `concat_logspec_`, `contextual_LogSpectrogram` | `logspec_from_files` |
| `wav2mel` | `melspectrogram` |

Also: `matplotlib` / `pandas` / `openpyxl` are no longer installed by default
(use the `[viz]` / `[data]` extras), and the deprecated `_config` / `_enh` /
`version` compatibility shims were removed.

## License

MIT — see [LICENSE.txt](LICENSE.txt).
