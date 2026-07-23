# Changelog

All notable changes to **Kudio** are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
