# Changelog

All notable changes to **Kudio** are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.2.0] - 2026-08-06

Building two packages on top of kudio surfaced a set of helpers that each of
them had written for itself. Every one of them is a place where two components
have to agree — on STFT geometry, on a sample rate, on normalisation
statistics — and where, until now, they agreed only by convention.

### Added
- **`STFT`** — the STFT settings as an immutable value, with `forward()` /
  `inverse()` / `n_bins` / `n_frames()` / `frame_times()` / `to_dict()`. A
  spectrogram and the waveform reconstructed from it must share `n_fft`,
  `hop_length`, `win_length` and `window`; passing those four to two separate
  functions and keeping them aligned by hand is how a producer and a consumer
  drift apart. `to_dict()` is meant to be stored next to a checkpoint, so a
  model and the geometry it was trained on travel together.
- **`Standardizer`** — per-column mean/std with `fit`/`transform`/`inverse`
  and `save`/`load`. Normalisation statistics are as much a part of a trained
  model as its weights; a model reloaded without them produces confident
  nonsense.
- **`stack_context(frames, context, pad=)`** — context-frame stacking for *any*
  feature matrix. The logic already existed but was fused into
  `waveform_to_spectrogram(forward_backward=)`, out of reach for MFCC or mel
  features. That path now delegates here, so there is one implementation.
- **`frame_windows(frames, n_frames, pad=)`** — cut a feature matrix into
  fixed-length windows, tail padded rather than dropped.
- **`normalize(y, peak=)` / `normalize_db(y, dbfs=)`** — peak normalisation
  with the divide-by-zero guard built in.
- **`resample(y, orig_sr, target_sr)`** — the in-memory counterpart of
  `file_load(..., sr=)`, which could only resample on the way in from disk.
  A no-op when the rates already match, so it is safe to call unconditionally.

### Fixed
- **`win_length` no longer defaults to a hard-coded 512.** It follows `n_fft`,
  the librosa convention. `waveform_to_spectrogram(y, n_fft=256)` previously
  raised `ParameterError: Target size (256) must be at least input size (512)`
  — a non-default FFT size simply did not work without also passing
  `win_length`. Applies to `waveform_to_spectrogram`,
  `spectrogram_to_waveform`, `file_to_spectrogram`, `logspec_from_files` and
  `save_spectrogram_as_wave`.

### Changed
- Following from that fix, `n_fft` **above** 512 with no explicit `win_length`
  now uses a window of `n_fft` rather than a 512-sample window zero-padded to
  `n_fft`. Results change for those calls. Pass `win_length=512` to keep the
  old behaviour.

## [3.1.0] - 2026-08-06

> Never released to PyPI on its own — these changes ship as part of 3.2.0.
> Kept as a separate section because they are a distinct set of fixes.

### Fixed
- **`Synthesizer` no longer resamples every mixture to 16 kHz.** The output
  rate came from a hard-coded fallback, so an 8 kHz corpus silently produced
  16 kHz mixtures — the dataset on disk disagreed with its source, and any
  model trained at the source rate then refused the files it was meant to
  consume. The default now keeps each clean file's own rate.
- **Noise is resampled to match its clean partner.** Previously both sides were
  loaded at the same fixed rate, which hid the problem; with a rate-preserving
  default, a noise file recorded at a different rate would otherwise have been
  mixed in as-is.

### Added
- `list_devices(kind=None)` — devices as **sounddevice** sees them, which is the
  index space `record(device=…)` accepts. `CheckDevice` enumerates through
  PyAudio for the streaming classes, and the two bindings number devices
  independently: an index from one was never valid in the other, and nothing
  said so. `kudio devices` now prints both spaces, labelled.
- `melspectrogram()` accepts a **waveform array** as well as a path (arrays
  need `sr=`), matching `mfcc(y, sr)`. Previously it was file-only, which
  forced a round-trip through disk for in-memory audio.
- `check_metrics_install()` is exported from the top level; it reports which of
  PESQ / STOI / SDR are installed and was already the documented way to degrade
  gracefully, but callers had to reach into `kudio.core.evaluator` for it.

### Changed
- `Synthesizer.syn(desired_sample=…)` is renamed **`target_sr`**. The old name
  read like a sample *count* — `kudio.core.feature` uses `desired_samples` for
  exactly that — while it actually meant a sample *rate*. `desired_sample` still
  works and emits a `DeprecationWarning`.

### Migration
Nothing to do if your audio was already 16 kHz. To keep the old behaviour on a
corpus at another rate, ask for it explicitly:

```python
syx.syn(mode="inc", target_sr=16000)
```

## [3.0.1] - 2026-07-11

### Fixed
- Packaging: removed `pysepm` from the `[eval]` extra. `pysepm` is not
  published on PyPI, so `pip install kudio[eval]` / `kudio[all]` failed to
  resolve. PESQ is now documented as an optional manual install
  (`pip install https://github.com/schmiph2/pysepm/archive/master.zip`);
  `[eval]` keeps `mir_eval` + `pystoi` (STOI/SDR), which the code already
  degrades around when PESQ is absent.

## [3.0.0] - 2026-07-11

Major cleanup release: focus kudio into a lightweight, composable audio core.

### Added
- **Effects & augmentation** (`kudio.effects`): `trim_silence`,
  `split_on_silence`, `time_stretch`, `pitch_shift`, `gain`, `random_gain`,
  `add_noise_snr`, `reverb`, `spec_augment` (all seedable where random).
- **Dependency-free metrics**: `si_sdr`, `snr`, `segmental_snr`.
- **Exception hierarchy** rooted at `KudioError` (`AudioIOError`,
  `FeatureError`, `DeviceError`, `SynthesisError`, `DependencyError`).
- **`kudio` CLI**: `devices`, `info`, `convert`, `synth`, `trim`.
- `record()` (sounddevice backend), `map_waves()` parallel helper.
- `save_wave(..., subtype=...)` — 16/24-bit and float output via soundfile.
- `Synthesizer.syn(..., seed=...)` for reproducible synthesis.
- Docs scaffold (mkdocs + mkdocstrings) and GitHub Actions CI/publish workflow.

### Changed
- **Dependencies slimmed**: `matplotlib`, `pandas`, `openpyxl` moved to the
  `[viz]` / `[data]` extras. `import kudio` no longer pulls them.
- **Canonical API names** with deprecation-warning aliases (see README table);
  e.g. `w2s` → `waveform_to_spectrogram`.
- `file_load` reads via soundfile on the native-rate path (no librosa/numba).
- `KudioConfig` defaults trimmed to audio-only (training keys moved to chptrain).

### Removed
- Deprecated compatibility shims `_config.py`, `_enh/`, `version.py`,
  `util/others.py`.

## [1.0.0] - 2026-07-11

First packaged release: extracted from the ckAudiux application into a
standalone, installable, MIT-licensed package.

### Added
- Core audio API: file I/O (`file_load`, `save_wave`, `check_input`),
  streaming/recording (`Recorder`, `LocalStreamReader`, `RemoteStreamReader`),
  thread-safe buffering (`AudioBuffer`, `FetchBuffer`).
- Feature extraction and spectrogram <-> waveform conversion (`w2s`,
  `spec2wavform`, `w2mfcc`, `wav2mel`, ...).
- Noisy-data synthesis (`Synthesizer`) and enhancement (`trad_enhance`,
  `wavelet_low_pass_filter`).
- Speech-quality evaluation (`AudioEvaluate`, `eval_metrics`: PESQ/STOI/SDR).
- Typed configuration (`KudioConfig`), `py.typed` marker.
- Optional extras: `audio` (PyAudio/sounddevice), `eval` (mir_eval/pystoi),
  `dev` (pytest). 57-test suite.

### Changed
- Modernized to librosa >= 0.10, numpy 2.x, pandas 2.x; replaced removed APIs
  (`np.fromstring`/`tostring`, `DataFrame.append`, `scipy.signal.hanning`,
  `librosa.waveplot`, ...).
- Rewrote `AudioBuffer` on `threading.Condition`/`deque` (was manipulating
  private `Semaphore._value`).
- Replaced `print` with the standard `logging` module; heavy dependencies
  (PyAudio) are now imported lazily.

### Removed
- Hidden AES license check that ran at import time (moved out of the package).
- Runtime `pip install` calls in the evaluation module.

### Fixed
- `check_path` sibling-increment logic, `Recorder.save` byte handling, and
  several latent shape/argument bugs.
