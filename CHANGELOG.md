# Changelog

All notable changes to **Kudio** are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.5.0] - 2026-09-10

Three things the toolkit could not do: say which part of a denoiser was doing
the work, draw a spectrogram of audio that has not finished arriving, and say
what note somebody is on.

`trad_enhance` was the whole of kudio's statistical enhancement: one function,
no parameters, one noise estimator welded to one gain rule. There was no way to
ask which part was doing the work, and no way to try a different one.

### Added — enhancement as interchangeable parts

- **`spectral_enhance(y, sr, method=…, noise=…)`** with **seven gain rules** —
  `specsub`, `multiband`, `wiener`, `mmse_stsa`, `logmmse`, `omlsa`,
  `spectral_gate` — over **four noise estimators**: `mcra`, `quantile`,
  `initial`, `minimum`.

  Every method is the same three decisions: estimate the noise, estimate the a
  priori SNR, turn it into a gain. Separating them is what makes them
  comparable — hold the noise estimator still and the difference you hear is
  the gain rule.

  The `wiener` / `mmse_stsa` / `logmmse` / `omlsa` family shares the
  decision-directed a priori SNR of Ephraim & Malah, so what distinguishes them
  really is only the gain function. Their known ordering —
  `Wiener ≤ log-MMSE ≤ MMSE-STSA`, all three converging on Wiener at high SNR —
  is asserted in the test suite, which is a stronger check on the formulas than
  any citation.

  **The noise estimator is as large a lever as the gain rule.** Across the
  full 7 x 4 grid on the suite's fixture: 5.8 dB of mean SI-SDR spread when the
  gain rule varies, 6.4 dB when the estimator does. Neither dominates, so a
  comparison that varies only the method is looking at half the problem.

- **`compare_enhancers(y, sr, reference=None)` → `[EnhanceResult]`** — run
  several over identical audio and measure each: wall time, residual noise
  floor, loudness, and (given a reference) SNR / SI-SDR / PESQ / STOI plus
  **`delta_snr_db`, the improvement over doing nothing**, which is the number
  that answers "did this help". A method that raises is reported with `error`
  set rather than taking the comparison down with it.

  **Both axes sweep.** `noises=[...]` compares estimators, and giving both
  compares the full grid — a run is a *(method, estimator)* pair, not a method,
  because the two are separate choices of comparable size.

  **Anything callable can join the table.** `('my model', fn)` or a bare
  function ranks a trained model, a wrapper around another library or a one-off
  experiment beside the built-ins. "Is the model actually better than log-MMSE"
  is the question people have, and two separate printouts is how it goes
  unanswered. A custom entry brings its own noise handling, so it runs once
  rather than being swept, and its `noise` column says `own`.

- **`StreamEnhancer`** — the same four decision-directed methods, frame by
  frame, for audio that has not finished arriving. `spectral_enhance` needs the
  whole clip, which is useless for monitoring a microphone. Everything needed
  was already recursive, so the noise trackers and the a priori SNR estimator
  are now *state plus an update* (`make_noise_tracker`, `DecisionDirected`)
  shared by both paths rather than written twice.

  One frame of latency (32 ms at the default), block-size independent, and
  refuses `quantile` / `specsub` / `multiband` / `spectral_gate` rather than
  quietly meaning something different from what those names mean offline.

- **`enhance_folder(src, dst, method, report=True)`** — the denoising
  counterpart of `convert_folder`, mirroring the input's directory structure
  the same way. `report=True` measures each file's noise floor before and
  after; `mean_reduction_db()` summarises it.

- **`crossfade` / `splice`** — join clips through a short overlap instead of
  butting them together, since a step in the waveform is a click. The result is
  *shorter* by the overlap, which is documented rather than papered over:
  anything tracking positions has to account for it.

  The two default to different shapes, measured rather than assumed. A splice
  joins two moments of the same recording, which are usually alike, and
  equal-power over-shoots on alike material — on a sine cut at a period
  boundary it raised the peak from 0.500 to 0.567 while linear left it at
  0.500. So `splice` is linear and `crossfade`, which joins genuinely different
  material, is equal-power.

- **`Label` / `save_labels` / `load_labels`** — named time spans, as JSON or as
  an **Audacity label track**. Three columns of plain text that half the audio
  world can already read is worth more than a private format only kudio
  understands. `kudio.vad` output drops straight in.

- **CLI**: `kudio enhance --recursive`, and `kudio compare --noises all` for the
  full grid.

  Timing is warmed up first: librosa's STFT is JIT-compiled and scipy's Bessel
  functions import lazily, so whichever method ran first was being charged ~3 s
  of one-time cost — in a table whose entire purpose is comparing methods.

- **`METHODS` / `NOISE_ESTIMATORS`** describe every method and every parameter:
  range, default, step, unit and a sentence of help. KudioStudio builds its
  controls from this rather than restating it, so a parameter added here
  appears there without anyone editing the GUI.

- **`STFT.analyse()` / `STFT.synthesise()`** — the complex transform.
  `forward()` returns log-power and throws the phase away, which is right for a
  feature and useless for anything that has to reconstruct the signal.

- **CLI**: `kudio enhance in.wav out.wav --method logmmse --set q=0.4` and
  `kudio compare in.wav --reference clean.wav --write-to results/`.

### Added — audio that is still arriving

- **`SpectrogramStream(sr, …)`** — a rolling log-magnitude spectrogram computed
  one column at a time. `push(block)` returns the columns that block completed;
  `columns()` is the window as an image, newest on the right.

  The point is that each column is computed **once**, when its samples arrive.
  Re-running an offline STFT over the last few seconds several times a second
  — the usual workaround — recomputes about 240 columns to gain three.

  Blocks may be any size: pushing the same audio in blocks of 1, 97, 512 or
  4096 gives byte-identical columns, so the picture never depends on what the
  device chose to hand over. Feed it `StreamRecorder.drain()`, never `tail()`.

  Columns follow the **`center=False`** convention — a stream cannot pad audio
  it has not heard — and the suite checks them against
  `librosa.stft(…, center=False)` rather than asserting the claim. Values are
  dBFS normalised by the window, so a full-scale sine reads 0 dB at any
  `n_fft`. `n_mels=` switches to a mel axis; 2-D input is refused rather than
  flattened, since flattening interleaved channels analyses a signal nobody
  recorded.

### Added — fundamental frequency

- **`f0(y, sr)` → `PitchTrack`** — pYIN (Mauch & Dixon 2014), with the
  voiced/unvoiced decision beside the contour. `f0` is `NaN` wherever the frame
  is unvoiced, because an estimator asked "what is the pitch here" always
  answers — for silence, for a slamming door, for the `s` in *this* — and a
  contour drawn through those answers is a picture of noise with a line
  through it.

  `PitchTrack` carries `median_hz`, `range_hz()`, `semitone_range` (the unit
  pitch is actually heard in), `voiced_ratio`, `voiced_segments()` and
  `to_labels()`, which turns the voiced runs into `kudio.Label`s that
  `save_labels` writes as an Audacity label track.

  Three settings are **refused rather than silently wrong**: a `frame_length`
  too short to hold two periods of `fmin` (which otherwise returns "unvoiced
  everywhere", reading as a property of the recording), an `fmax` above
  Nyquist, and `fmin >= fmax`.

- **`kudio pitch file.wav`** — median, range in Hz and semitones, voiced
  percentage; `--segments` lists the voiced spans and `--labels` writes them as
  a label track. Exits non-zero when nothing is voiced, so it drops into a
  shell test the way `report` and `vad` do.

### Note on `trad_enhance`

Unchanged, and still exported. It is roughly `specsub` with MCRA and fixed
parameters, but it returns **128 samples fewer than its input**; the new
methods return exactly as many, so their output lines up sample-for-sample with
what it came from.

## [3.4.0] - 2026-08-15

Two gaps closed. kudio could analyse audio and augment it, but it could not
**edit** it — no fade, no filter, no reversal — and it could not tell you
**where in a recording somebody is talking** without a fixed silence threshold
that real rooms defeat. Plus the folder-level helpers three downstream projects
had each written for themselves.

### Added — editing

- **`fade_in` / `fade_out` / `fade`** (`linear`, `cosine`, `exponential`).
  The 10 ms default is the anti-click amount: long enough to remove the step at
  a cut, short enough not to be heard. Overlapping fades raise rather than
  silently multiplying the middle.
- **`reverse`**, **`remove_dc`**. The DC mean is accumulated in float64 — a
  float32 mean over a long clip carries enough rounding error to leave behind
  the offset it was asked to remove.
- **`highpass` / `lowpass` / `bandpass` / `bandstop` / `band_filter`** —
  Butterworth, **zero-phase by default** (`sosfiltfilt`). A filtered copy that
  is shifted in time no longer lines up with the original, which breaks an A/B,
  a spectrogram overlay and a training target all at once.
  `wavelet_low_pass_filter` in `kudio.enhance` is a denoising algorithm that
  happens to be low-pass; these are the plain filters, and the two are not
  substitutes.

### Added — voice activity

- **`vad(y, sr)` → speech spans in seconds**, plus `vad_split`, `vad_trim` and
  `speech_ratio`. `split_on_silence` cuts on level alone, which is right for a
  studio take and wrong for anything with a noise floor: above some noise level
  no fixed `top_db` both keeps the quiet consonants and drops the room.

  This decides per frame on level **relative to the recording's own noise
  floor**, on **spectral flatness** (speech is harmonic and peaky; steady noise
  is flat), and on zero-crossing rate for the unvoiced fricatives — but the
  ZCR test only rescues frames **adjacent to** something voiced. On its own, an
  /s/ and a hiss are the same measurement, and a detector that fires on the air
  conditioning is not a detector. The cost of that rule, stated plainly in the
  docstring: a whisper has no voiced frames and is not detected.

### Added — reference-free perception

- **`dnsmos(y, sr, model_dir=)` → `DnsmosScore(sig, bak, ovrl)`**, ITU-T P.835.
  `audio_report` says whether a recording is technically sound; this predicts
  what a listener would say about it. SIG and BAK move in opposite directions
  under denoising — aggressive suppression raises BAK and lowers SIG because it
  eats the speech with the noise — so `summary()` names that trade rather than
  averaging it away.

  **The weights are not bundled.** They belong to Microsoft's DNS-Challenge;
  point `model_dir` (or `$KUDIO_DNSMOS_DIR`) at a checkout. `pip install
  kudio[dnsmos]` adds onnxruntime. The published P.835 mapping coefficients are
  included and attributed, overridable by a `polyfit.json` beside the model, and
  `polyfit=False` returns the raw outputs.

### Added — files and folders

- **`audio_info(source)` → `AudioInfo`** (sr, channels, frames, duration, peak,
  subtype, format). The `kudio info` CLI computed this and printed it,
  KudioStudio's transport bar recomputed it and KudioEnhance checked rates —
  three implementations of one question. `peak=False` reads the header only.
- **`convert_folder(src, dst, ...)` → `ConvertResult`** — resample, re-encode,
  trim and normalise a whole tree. Output **mirrors the input's directory
  structure**, so two files with the same name in different subfolders do not
  collide. `peak=` and `lufs=` are mutually exclusive; normalising twice would
  undo the first one.
- **`Pair` / `save_manifest` / `load_manifest` / `split_pairs`** — dataset
  manifests, moved out of KudioEnhance because nothing in them is specific to
  denoising. Plain JSON, so another tool can read it without importing kudio.
  `Pair` is frozen and therefore hashable: split overlap is checked with sets,
  which silently catches nothing if the pairs are not.
- **`StreamRecorder`** — record for as long as you like while watching the
  level. `record(seconds=…)` is one blocking call: you commit to a duration up
  front and nothing is observable until it returns, which is fine for a script
  and useless for a person at a microphone. `start()` / `level()` / `tail()` /
  `elapsed()` / `stop()`, safe to poll from a UI timer, and **nothing is
  dropped** — unlike `AudioBuffer`, which is a monitor and discards when full.
- `LocalStreamReader` gained an `on_frame=` callback, which is what makes the
  above possible without duplicating its device negotiation.

### Added — CLI

`kudio report` (reference-free health check, exits 1 when something is wrong,
so it works in a shell test), `kudio loudness`, `kudio normalize --lufs/--peak`,
`kudio vad [--split-to DIR]`, and `kudio convert --recursive` for folders.
`kudio info` now goes through `audio_info` and prints the encoding.

### Internal

Framing moved to one private home (`kudio.core._framing.frame_view`); the
report and the VAD had begun to need identical framing, and written twice they
would eventually have disagreed about what frame 40 covers.

## [3.3.0] - 2026-08-15

Two things a recording can be asked about without anything to compare it to:
**how loud does this sound**, and **is this usable at all**. Neither had an
answer in kudio before — `normalize` scales by the peak sample, and every
quality metric needs the clean signal beside the degraded one.

### Added
- **`loudness(y, sr)`** — gated integrated loudness to **ITU-R BS.1770-4**, in
  LUFS. `normalize` and `normalize_db` scale by the *peak* sample, which says
  nothing about how loud something sounds: a clip with one stray transient
  peak-normalises to a whisper. Nothing in kudio could answer "how loud is this
  to a listener" until now.
- **`normalize_lufs(y, sr, lufs=-23.0)`** — normalise to a loudness target
  (-23 LUFS is EBU R 128). Logs a warning rather than silently limiting when
  the gain would push the signal past full scale.
- **`match_loudness(y, reference, sr, reference_sr=None)`** — scale one clip to
  another's loudness. **An A/B comparison is not valid until this has been
  done**: of two versions of a clip, the louder is reliably preferred whatever
  its quality, so an unmatched comparison partly measures level. Handles the
  two sides being at different sample rates, since loudness is a property of
  the sound rather than of the sampling.

The standard publishes its K-weighting coefficients at 48 kHz only, and the
textbook shelf/high-pass equations do not reproduce them. The filters are
instead carried to the s-plane and re-discretised at the requested rate through
a prewarped bilinear transform: exact at 48 kHz by construction, within
0.06 dB of the reference curve at 16 kHz and 0.21 dB at 8 kHz. Cross-checked
against `pyloudnorm` — worst disagreement 0.043 LU across 8/16/44.1/48 kHz —
by a test that runs whenever that package happens to be installed. It is **not**
a dependency; this is numpy and scipy only.

- **`audio_report(y, sr)` → `AudioReport`** — the measurements you can make
  with **no clean reference**. Every existing metric (SNR, SI-SDR, PESQ, STOI)
  needs the clean signal next to the degraded one; synthetic datasets have
  that, real recordings never do. Reports level, true clipping, DC offset,
  silence ratio, noise floor, an SNR estimate and the bandwidth the content
  really occupies, plus `.problems()` — the findings in plain words.

  Two details worth knowing. **Clipping is a flat top, not a big number**:
  float audio amplified past 1.0 still traces its waveform and has lost
  nothing, so it gets a separate, weaker warning than a run of samples pinned
  at the rail. And **band-limit detection catches upsampled files** — a clip
  labelled 16 kHz whose content stops at 3.8 kHz is 8 kHz audio wearing a
  bigger number, which quietly wastes half a dataset. Audio recorded at its
  stated rate has a noise floor spanning the whole band and reaches Nyquist;
  upsampled audio has nothing above the old one.

### Fixed
- **`file_load(mono=False)` returned two different channel layouts.** The
  `sr=None` branch reads through soundfile and gave `(frames, channels)`; the
  resampling branch goes through librosa and gave `(channels, frames)`. The
  layout therefore depended on whether you happened to ask for a resample, and
  nothing said so. Both now return **channels last**, matching soundfile and
  what `loudness()` expects.

  Mono loading — the default, and almost all use — is unaffected. If you were
  loading multi-channel audio *with* an explicit `sr`, you were getting
  `(channels, frames)` and will now get its transpose.

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
