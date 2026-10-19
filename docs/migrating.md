# Migrating from v2

kudio 3.0 was a cleanup release. It gave the feature functions descriptive
names, moved plotting, dataframes and live audio out of the default install,
and removed the compatibility modules v2 still carried. Most v2 code runs on
3.x as it is, with warnings. Four things need attention: a new environment,
because 2.0 installs only on Python 3.8; imports of the removed modules and of
names that left the `kudio` namespace; calls that need a package the core
install no longer brings; and PESQ, which comes from a different package now.

The later 3.x releases changed a few results for the same call, most of them
fixes. They are listed at the end, because a v2 project upgrading today gets
all of them at once.

## Upgrading

kudio 2.0 came as wheels for CPython 3.8 only, and every 3.x release needs
Python 3.9 or newer, so an environment that has 2.0 cannot take 3.x.
`pip install -U kudio` there keeps 2.0 and says nothing. Make a new
environment on 3.9 or newer and install into that:

```bash
python -m venv .venv                # this python must be 3.9 or newer
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install kudio                   # plus the extras you use, see below
python -c "import kudio; print(kudio.__version__)"
```

Your own code moves with it. 2.0 pinned numpy at 1.20 or older, scipy at
1.5.4 or older and librosa at 0.7.2 or older; the current release needs numpy
1.24 or newer and below 2.6, scipy 1.10 or newer and librosa 0.10 or newer,
and reads files through soundfile.

## Finding the old names

Every call to an old name works and emits a `DeprecationWarning` that names
the replacement:

```text
DeprecationWarning: kudio.w2s() is deprecated and will be removed in a future release; use kudio.waveform_to_spectrogram() instead.
```

Python prints a `DeprecationWarning` only when the call sits in the script you
ran. The same call inside a module of your own package is silent by default,
and that is where most v2 calls live. pytest shows them in its warnings
summary, so running your test suite is the quickest way to list them. To turn
each one into an error, so the test that reaches it fails:

```bash
python -m pytest -W "error:kudio.:DeprecationWarning"
python -W "error:kudio.:DeprecationWarning" your_script.py
```

The filter matches warnings whose message starts with `kudio.`, so
deprecations from other libraries stay warnings.

## Renamed functions

| v2 name | 3.x name |
|---|---|
| `w2s`, `wavform2spec` | [`waveform_to_spectrogram`][kudio.core.feature.waveform_to_spectrogram] |
| `spec2wavform` | [`spectrogram_to_waveform`][kudio.core.feature.spectrogram_to_waveform] |
| `f2s`, `wav2spec` | [`file_to_spectrogram`][kudio.core.feature.file_to_spectrogram] |
| `s2w`, `spec2wav` | [`save_spectrogram_as_wave`][kudio.core.feature.save_spectrogram_as_wave] |
| `w2mfcc`, `wav2mfcc` | [`mfcc`][kudio.core.feature.mfcc] |
| `concat_mfcc`, `concat_mfcc_` | [`mfcc_from_files`][kudio.core.feature.mfcc_from_files] |
| `concat_logspec`, `concat_logspec_`, `contextual_LogSpectrogram` | [`logspec_from_files`][kudio.core.feature.logspec_from_files] |
| `wav2mel` | [`melspectrogram`][kudio.core.feature.melspectrogram] |

Each old name is now the new function behind a warning, so it takes the new
function's arguments and gives its result. All fifteen are still importable
from `kudio` and from `kudio.core.feature`, and will be removed in a future
release. Two things can still catch you out. `wav2mel(file_name=...)` fails,
because `melspectrogram` calls that argument `source`: pass the path
positionally or as `source=`. And the STFT functions changed their default
`win_length` in 3.2, which the last part of this page covers.

The console colour helpers warn from the top level too: `kudio.red`,
`kudio.warn`, `kudio.error`, `kudio.info`, `kudio.h1` and the rest of
`kudio.util.colors`, along with the misspelt `upercase`. They stay in
`kudio.util`, where `kudio.util.red(...)` is silent, and `upercase` is
`uppercase`.

## Removed modules

3.0 removed four compatibility modules. Importing one now fails with
`ModuleNotFoundError`:

| Removed | Use instead |
|---|---|
| `kudio._config` | [`KudioConfig`][kudio.config.KudioConfig], also as `kudio.KC` |
| `kudio._enh` | [`trad_enhance`][kudio.enhance.traditional.trad_enhance], [`wavelet_low_pass_filter`][kudio.enhance.traditional.wavelet_low_pass_filter] |
| `kudio.version` | `kudio.__version__`, `kudio.__author__`; there is no `__date__` |
| `kudio.util.others` | [`find_duplicate`][kudio.util.tools.find_duplicate] |

Every replacement is importable straight from `kudio`.

[`KudioConfig.default_config`][kudio.config.KudioConfig.default_config] holds
audio settings only, under `waveform`, `device` and `f_mfe`; the training keys
are gone, and so is `load_config`'s `type=` argument. A project that read
model or training settings from kudio's defaults should keep them in a YAML
file of its own and load it with
[`KudioConfig.load_config`][kudio.config.KudioConfig.load_config]. Check that
the file exists first. On a path it cannot find, `load_config` returns the
audio defaults with a `UserWarning`, and a key that is not in the config reads
as an empty config rather than an error, so `cfg.train.lr` after a mistyped
path is `{}`. Test for a section with `"train" in cfg`, or call `cfg.freeze()`
to make a missing key raise `KeyError`.

## Packages that are no longer installed

kudio 2.0 installed matplotlib, pandas and PyAudio along with itself. 3.x
installs none of them, and `import kudio` imports none of them either. Code
that used them through kudio needs the matching extra:

```bash
pip install "kudio[audio,viz,data]"
```

| Needs | Extra | Without it |
|---|---|---|
| `Recorder`, `LocalStreamReader`, `RemoteStreamReader` (from `initialize_params()` on), `play_audio`, `CheckDevice` ([Recording and live views](guide/live-audio.md)) | `[audio]` | `DependencyError`, naming `kudio[audio]`; it is an `ImportError` too |
| [`kudio.util.visual`][kudio.util.visual]: `WaveVisualizer`, `history_plot`, `Common`, the `plot_*` helpers | `[viz]` | importing the module raises `DependencyError` |
| [`AudioEvaluate.get_df`][kudio.core.evaluator.AudioEvaluate.get_df] and [`per_file`][kudio.core.evaluator.AudioEvaluate.per_file] | `[data]` | the call raises `DependencyError` |
| [`data2xlsx`][kudio.util.conv.data2xlsx] | `[data]` | the call raises `DependencyError` |

The `DependencyError` names the extra:

```text
kudio.exceptions.DependencyError: 'matplotlib' is required for this feature. Install it with: pip install kudio[viz]
```

For four names the extra is not the whole fix. 2.0 imported `WaveVisualizer`,
`history_plot`, `Common` and `data2xlsx` along with `kudio`, and 3.x keeps
none of them in the `kudio` namespace, so `kudio.WaveVisualizer(...)` fails
with `AttributeError` even with `[viz]` installed. Import them from their
modules:

```python
from kudio.util.conv import data2xlsx
from kudio.util.visual import Common, WaveVisualizer, history_plot
```

`[audio]` also installs sounddevice, which the newer
[`record`][kudio.core.stream.record] and
[`list_devices`][kudio.util.check.list_devices] run on, and `[data]` brings
openpyxl, which writing `.xlsx` files needs. `AudioEvaluate.eval` runs
without pandas, and the evaluator's
[`rows`][kudio.core.evaluator.AudioEvaluate.rows] attribute holds one plain
dict per enhanced file, for code that does not want a dataframe. The noisy
baseline's per-file scores only come out of `per_file()`; without pandas you
get their means, as `base_stoi` and the like.

## PESQ

PESQ runs on the `pesq` package from PyPI, installed by its own extra:

```bash
pip install "kudio[eval,pesq]"           # [pesq] compiles: it needs a C compiler
```

`[pesq]` is not part of `[eval]` or `[all]`: it builds from source and would
stop those two from installing on a machine without a compiler. 3.x does not
use pysepm, and never installs anything at run time.

Without `[pesq]` nothing fails.
[`AudioEvaluate`][kudio.core.evaluator.AudioEvaluate] leaves the PESQ column
out of `get_df()` and `per_file()`, and
[`eval_metrics`][kudio.core.evaluator.eval_metrics] returns `None` for it. The
only notice is a log line, which stays hidden until logging is configured
([Getting started](getting-started.md) shows how). Check with
[`check_metrics_install`][kudio.core.evaluator.check_metrics_install], or ask
for PESQ by name with `kudio.score(..., ["pesq"])`, which raises.

By default PESQ is MOS-LQO, wideband for 16 kHz audio. Score the v2 outputs
again with 3.x rather than set new numbers beside an old table, because the
two need not be on the same scale. The raw narrowband P.862 score most
enhancement papers quote is two arguments away:

```python
ev = kudio.AudioEvaluate(clean_files, noisy_files,
                         pesq_mode="nb", pesq_scale="raw")
```

[`pesq`][kudio.core.evaluator.pesq] takes the same two choices as `mode=` and
`scale=`, and [`score`][kudio.core.evaluator.score] and `eval_metrics` as
`pesq_mode=` and `pesq_scale=`. The ITU `pesq.exe` binary prints that raw
score too, and
[`pesq_score_exe`][kudio.core.evaluator.AudioEvaluate.pesq_score_exe] still
runs it, on Windows and with `[data]`. [Measuring quality](guide/metrics.md)
covers which scale to report.

## Results that change for the same call

These are not renames: the call is the same and the output differs. Where a
heading carries a version, that [changelog](changelog.md) section describes
the change.

### `win_length` follows `n_fft` (3.2)

`win_length` used to default to 512 whatever `n_fft` was. It now follows
`n_fft`, as librosa's does, in `waveform_to_spectrogram`,
`spectrogram_to_waveform`, `file_to_spectrogram`, `save_spectrogram_as_wave`
and `logspec_from_files`. Calls that leave `n_fft` at its default of 512 are
unaffected. `n_fft=1024` analyses with a 1024-sample window where it used to
zero-pad a 512-sample one, so to reproduce the features a model was trained
on, pass the old value:

```python
spec = kudio.waveform_to_spectrogram(y, n_fft=1024, win_length=512)
```

The same change makes an `n_fft` below 512 work at all; `n_fft=256` used to
raise from inside librosa.

### `Synthesizer` keeps the clean files' rate (3.1)

[`Synthesizer`][kudio.core.synth.Synthesizer] used to resample every mixture
to 16 kHz, so an 8 kHz corpus came out as 16 kHz files. Each output now has
the rate of its clean file, and the noise is resampled to match. Nothing
changes for a corpus that was already at 16 kHz. To keep getting 16 kHz:

```python
syx.syn(mode="inc", target_sr=16000)
```

### `save_spectrogram_as_wave` keeps the noisy file's rate

It reads the noisy file at its own rate where v2 loaded it at 16 kHz, and
writes the result at that rate. Compute the spectrogram from audio at the
file's own rate, or resample the file first.

### Past full scale clips instead of wrapping around (3.7)

A mixture louder than full scale used to wrap around in a 16-bit file: 1.2
came out as -0.8. `Synthesizer`, [`save_wave`][kudio.core.io.save_wave] and
`save_spectrogram_as_wave` now clip at full scale and round 16-bit samples to
the nearest step, and `syn()` warns with the number of files it clipped.
`Synthesizer(subtype="FLOAT")` keeps everything past 1.0.

### `file_load(..., mono=False)` is channels last (3.0, 3.3)

In v2, a multi-channel file loaded with `mono=False` came back
`(channels, frames)`, the way librosa returns it. 3.x returns
`(frames, channels)` with or without `sr=`: 3.0 changed the native-rate path
and 3.3 the resampling one. Code that took the first channel as `y[0]` needs
`y[:, 0]` now; `y[0]` is the first frame, and nothing fails to say so. Mono
loading, the default, is unaffected. [Audio files](guide/audio-files.md) has
the details.

### A file that cannot be read raises (3.0)

[`file_load`][kudio.core.io.file_load] raises
[`AudioIOError`][kudio.exceptions.AudioIOError] where v2 printed a message and
returned `None`. Code that checked the result for `None` should catch the
exception instead.

### `reg` mode pairs noise and SNR differently (3.7)

In the default mode, [`Synthesizer.syn`][kudio.core.synth.Synthesizer.syn]
cycled the noise list and the SNR list side by side. When the two counts share
a factor, some pairs never came up: 14 noises against 21 SNRs paired each
noise with only 3 SNRs. The SNR now steps on by one at the end of every such
cycle, so any run of noises × SNRs consecutive outputs holds every pair
exactly once. When the two counts share no factor, nothing changes.

### `is_silence=True` adds a share of noise-only files (3.7)

The flag was handed to every file, so every output was written as noise alone,
and `syn_extra_mode`, which always sets it, never produced a mixture with
speech in it. Now only a `p_silence` share is written again as noise alone,
and the mixtures keep their speech. A dataset built with the flag before 3.7
needs building again.

## Coming from 3.0 to 3.6

Two more changes concern only code written for 3.0 to 3.6; 2.0 had neither
the keyword nor the seed.

- `Synthesizer.syn(desired_sample=16000)` is now
  `Synthesizer.syn(target_sr=16000)`. `desired_sample` read like a number of
  samples and meant a sample rate. It still works and warns, with a message
  that starts `Synthesizer.syn`, so add
  `-W "error:Synthesizer.syn:DeprecationWarning"` to the filters above to
  catch it.
- A corpus built from a seed before 3.7 cannot be rebuilt from that seed. Each
  output file now draws its noise crop, room and packet loss from the seed
  and its own path, so a seeded run gives the same files with or without the
  process pool. Keep [`Synthesizer.seed`][kudio.core.synth.Synthesizer.seed]
  after a run and the same corpus can be built again;
  [Building a noisy dataset](guide/synthesis.md) shows how.
