# Changelog

All notable changes to **Kudio** are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Documentation and type annotations; nothing in the library behaves
differently.

### Changed

- Every public function and method the API reference shows has a type on
  each parameter and on its return: 60 of them were missing some, so their
  signatures on the site were incomplete. `lowercase` and `uppercase` have
  docstrings. All of it is annotation only; every module already had
  `from __future__ import annotations`, so nothing is evaluated at run time.

### Documentation

- The Traditional Chinese site covers every guide page, the migration notes
  and the API overview, where it had the home page, Getting started and the
  command line. The API pages stay in English, since they are the
  docstrings.
- On the Chinese pages, a line break between Chinese and Latin text, a number
  or inline code lost its space when the build joined the lines, so the pages
  printed "Python裡" and "400 ms的": 56 places on the three pages already
  published.
- Guide pages that still described 3.7: `STFT.inverse` and
  `spectrogram_to_waveform` rebuild every sample and refuse a wrong shape
  with `FeatureError`; `RemoteStreamReader` puts a chunk that arrives in
  pieces back together (40 of 40 through a server cutting them into
  1460-byte pieces, where 3.7.3 got one); the mel rows of
  `SpectrogramStream.freqs` are the band centres; `kudio loudness` measures a
  stereo file as stereo; and `normalize_lufs` pushing a peak past full scale
  raises a `UserWarning` rather than logging.
- Claims the code never matched: the band-limited finding names the
  bandwidth, not twice it; `spectral_gate` keeps its threshold on a
  `quantile` or `initial` estimate rather than taking the noisy signal's
  spread; `save_wave(d_sample=)` is not mono only; `trad_enhance`'s output is
  as long as its input; and `syn_extra_mode` still reads each noise at the
  rate of the file it is mixed into. The home page described KudioEnhance as
  Keras on TensorFlow, which it has not been since its 0.2.0.

## [3.9.0] - 2026-10-19

A minor version rather than a patch, because two results move: DNSMOS lays
out its windows as DNS-Challenge's script does, and an integer array is read
on its type's full scale wherever it goes in, which changes every number
measured on one.

### Changed — results that move

- **DNSMOS cuts its windows as DNS-Challenge's `dnsmos_local.py` does.** A
  clip shorter than one 9.01 s window is doubled until it holds one, and
  every window of the result is scored: seven for a 2 s clip, where kudio
  filled a single window. Windows end inside the clip's last whole second, so
  12.01 s gets three where it got four. The script works out each window's
  end in floating point, which leaves the windows starting at 7 to 23 s a
  sample short, and skips them; kudio keeps them, so the two agree on any
  clip that stays under 17 s once doubled.

### Changed — interfaces

- **`StreamEnhancer.flush()` stops where the input stopped**, so everything
  concatenated is as long as the input. 16000 samples in came back as 16383,
  the tail of the zero padding, and the guide said to trim it.
- **`CheckDevice.input_device` raises `DeviceError`** rather than a plain
  `RuntimeError` that `except kudio.KudioError` let through. `DeviceError` is a
  `RuntimeError` as well, so code catching that still does.
- **`KudioConfig.load_config` warns when it falls back to the defaults**: on a
  path that is not there, a folder it was given with no `cfg.yaml` in it, or a
  search that finds more than one. It said so at `INFO`, which nobody hears,
  so a misspelt file name ran on the defaults.

### Deprecated

- The console colour helpers at the top level, `kudio.red`, `kudio.warn`,
  `kudio.error`, `kudio.info`, `kudio.h1` and the rest of `kudio.util.colors`,
  and the misspelt `upercase`. `error`, `info` and `warn` read as part of
  kudio's error handling, and a star import put them over a caller's own
  names. They still work, with a `DeprecationWarning`; use `kudio.util.red`
  and so on, and `uppercase`.

### Fixed — integer audio

- **Everything that measures or processes audio read an integer array as
  plain numbers.** An int16 clip measured 90.3 dB louder than the file it came
  from in `loudness` and `true_peak`, `audio_report` called it past full
  scale, `snr`, `segmental_snr` and `sdi` scored an int16 reference against a
  float clip 32768 times apart, and the channel effects, `spectral_enhance`,
  `trad_enhance`, `apply_rir`, `StreamEnhancer`, `SpectrogramStream`,
  `melspectrogram`, `STFT.analyse` and `dnsmos` worked on a signal 32768 times
  too loud; `waveform_to_spectrogram` and `mfcc` refused it. Every one reads
  integers on their type's full scale now, the way soundfile reads an integer
  file and `gain` already did: int16 over 32768, int32 over 2**31, uint8
  around 128, where an 8-bit file's silence sits, so `trim_silence` used to
  find none in one. What returns float32 still does; `normalize_lufs` and
  `match_loudness` return the input's type, rounded and clipped, as
  `normalize` does.
- **Filters, fades, crossfades and `remove_dc` cast integer audio straight
  back.** A filter overshoots at an edge and an equal-power seam adds 3 dB, so
  a near-full-scale int16 square wave through a 2 kHz low-pass had 1194
  samples wrap round to the other sign, and two int16 clips at 30000
  crossfaded into a seam that was negative for 280 of its 320 samples, down
  to -32732. They round and clip within the type now, as `gain` and
  `normalize` already did. `time_stretch` and `pitch_shift` refused integer
  audio outright and take it the same way.
- **`add_noise_snr` squared int16 audio in int16**, which wraps round, so a
  quarter-scale tone mixed at 10 dB came out at 55.9 dB, and cast the mixture
  back by truncating. It measures and mixes in float and returns the input's
  type, rounded and clipped. `reverb` took the peak of int16 audio with
  `abs`, which leaves -32768 negative, so a clip with one sample pinned there
  came back at a peak of 20000 rather than 32768, and
  `AudioDataManager.play_audio(volume=2)` played an int16 20000 as -25536.

### Fixed — corrupt, silent and short input

- **A clip with NaN in it was still misread in places.** `trim_silence` cut
  it to nothing under librosa 0.11, `vad` left the frames holding the bad
  samples out and judged the rest, `compare_enhancers` returned a table in
  which every run had failed, and `f0` and `spectral_enhance` stopped on
  librosa's own error, which names neither the function nor the fix. Each
  refuses it with `FeatureError` now, and the command line reports it on one
  line with exit 2, where `kudio pitch` printed a traceback and `loudness`,
  `vad` and `compare` exited 0. `audio_report` stops warning about the
  samples it already names as its first finding.
- **`trad_enhance` returned NaN** for a clip that opens with digital silence
  (768 samples of it here) or is silent throughout, and silence for a clip
  shorter than one 512-sample frame. The power has a floor under it, a short
  clip is refused, and stereo is refused by name rather than by numpy.
  `wavelet_low_pass_filter` returned NaN for silence and one sample too many
  for an odd length.
- **`Synthesizer` failed without saying which file.** An empty noise file
  divided by zero, and a file with NaN in it stopped on librosa's error; both
  are a `SynthesisError` naming the file. `target_sr=0` is refused before the
  output folder is emptied, not after.
- `mfcc` and `STFT.analyse` read a stereo array as thousands of two-sample
  signals and came back the wrong shape, and `melspectrogram` on one failed
  with a numpy broadcast error; all three refuse it by name. `denoise_wav2spec`
  divided by the peak of a silent clip and stopped on librosa's error.

### Fixed — elsewhere

- `check_path` globbed for numbered siblings, so `runs/exp[1]` missed its own
  `runs/exp[1]2` and handed that folder back as free.
- `vad_frames(center=True)` counted `1 + n // hop` frames, one more than the
  spectrogram has for an odd `n_fft`.
- TPDF dither in `bit_depth` used `2 / 2**bits` as its step, half the grid's
  step at 2 bits; it is two of the grid's own steps wide at every depth.
- `AudioDataManager.save_all` split a mono clip into `<name>_ch0.wav` beside
  its source when the manager was set to two channels; the clip's own channel
  count decides now.
- `AudioBuffer.get_fixed_size_data` let a second capture take over the first
  one's state, and the second then failed with a `TypeError`; it raises
  `RuntimeError` saying a capture is running.
  `LocalStreamReader.stream_start()` and `play_audio()` without a device raise
  `DeviceError` rather than an `AttributeError`, and `StreamRecorder.start()`
  refuses to continue a take when the device reopens at another rate or
  channel count.
- `load_manifest` names the row that lacks `noisy` or `clean` (a
  `ValueError`, where the dataclass raised a `TypeError`).
- `LoadAudio` collects files the way `check_input` does, `.WAV` included;
  `data2xlsx(fit_column=True)` sized the columns only when the workbook
  already existed; `pesq_score_exe` runs the binary with an argument list
  rather than a shell line, so a path with a space in it runs, and a binary
  that is not there is named in a `FileNotFoundError` rather than surfacing
  as an `IndexError`; the `*_score` methods say to call `eval()` first;
  `compare_enhancers` scores STOI even when PESQ finds no utterance.

### Added

- `kudio.PA_FLOAT32` and `kudio.PA_INT16`, the `wav_format` codes
  `LocalStreamReader` takes, which only `kudio.core.stream` had.
- `window=` on `file_to_spectrogram` and `logspec_from_files`, which were
  Hamming only. It comes after their other arguments, so a call that passes
  those by position is unchanged.

### Documentation

- The package description, the README's opening line and the site's English
  description say what the site's home page says: a toolkit for audio, speech
  most of all.
- `syn_extra_mode` puts `dB` on both signs of an SNR (`n5dB`), where `syn()`
  writes `n5`. Both stay, since corpora on disk use them; its docstring and
  the synthesis guide say so and point at the manifest's `snr_db`.
- The inspection guide still said DNSMOS averages the windows' raw outputs
  and maps the average onto the MOS scale. Each window has been mapped first
  since 3.8.0, as the code and the 3.8.0 entry below say.

## [3.8.0] - 2026-10-15

A documentation site, and the fixes that writing it turned up. Every example
on the site was run against the library, and where the output and the
docstring disagreed, the library was usually the one at fault. A minor version
rather than a patch, because several of the fixes change numbers that existing
code may have stored: the `quantile` estimator, dither, DNSMOS, the report's
noise floor and RT60 on a recorded response all read differently now, and
`spec_augment` reads its input the other way round. Those are listed first.

### Changed — results that move

- **`spec_augment` reads `(frames, bins)`**, the layout of every spectrogram
  kudio makes. It read `(freq, time)`, so on kudio's own spectrograms the
  "frequency" masks blanked time and the "time" masks blanked frequency. The
  default `mask_value=None` fills with the spectrogram's mean, as SpecAugment
  does: the old fill of 0 is a power of 1 in log10 power, louder than most of
  the cells it was meant to hide.
- **The `quantile` noise estimate sits at the noise, not 7.9 dB under it.**
  Steady noise power in a bin is exponentially distributed, so its 15th
  percentile is 0.16 of its mean. The estimate is divided by `-ln(1 - q)` now,
  which takes `quantile` from the weakest of the four estimators on the test
  clips (2.8 dB of SI-SDR gained, averaged over the gain rules) to the best
  (7.8 to 8.4). On the CLI page's example, `logmmse` over `quantile` goes from
  8.69 to 14.95 dB.
- **`bit_depth(dither=True)` adds TPDF dither two steps wide**, peak to peak.
  One step loosened the tie between the error and the signal without
  breaking it: a sine a quarter of a step high came back at 0.42 of its level
  on average, and comes back whole now. It costs 4.7 dB of SNR at 8 bits
  where one step cost 1.8, close to the 4.8 dB that tripling the noise power
  predicts.
- **`audio_report` takes the DC offset out before reading the noise floor.**
  An offset of 0.03 read as a floor of -30.5 dBFS and an SNR of 6 dB on a
  take whose real floor was -57.5 and SNR 31. When a tenth of the frames are
  exact zeros the floor is digital silence, `-inf`, with an SNR of `inf`:
  pauses of zeros were left out of the percentiles, so a perfectly clean clip
  was reported at 0.7 dB.
- **`rt60` measures from the onset**, the first sample within 20 dB of the
  peak. A recorded response starts with the time the sound took to arrive,
  and 5 ms of it in front of a 0.6 s room dropped EDT from 0.55 to 0.25 s and
  failed `reliable()`. A response whose direct sound alone falls 10 dB has no
  EDT; `reliable()` judges it on its fit instead of failing it, and the summary
  says why EDT is missing.
- **DNSMOS maps each window through the polynomial before averaging**, the
  order DNS-Challenge's own script uses; with a quadratic the two orders
  differ. Only `sig_bak_ovr.onnx` is looked for (`model_v8.onnx` is the P.808
  model, one score rather than three), and the personalised model in
  `pDNSMOS/` comes last in a search and warns when used.
- **`STFT.n_frames` is right for an odd `n_fft`.** `n // hop + 1` promised 126
  frames for 32000 samples at `n_fft=511` and `forward` gave 125.
- **`lowpass` at or above Nyquist hands the clip back** as a copy rather than
  raising, since there is nothing to remove. That includes the 8 kHz default
  at 16 kHz and below.

### Changed — interfaces

- **The command line reports an error on one line and exits 2.** A file that
  cannot be read or an argument the library refuses printed a traceback and
  exited 1, the same code as a finding, so `kudio report missing.wav || echo
  ...` could not tell the two apart. `-v` shows the traceback. Refusals and
  warnings go to standard error, warnings as one line each, and `-v` raises
  kudio's own logging only, not every library's.
- **Problems with the audio are warnings, not log lines.** NaN or inf in a
  loaded file, `normalize_lufs` pushing a peak past full scale, unknown keys in
  a manifest and `syn(overwrite=False)` finding the folder full were logged, and
  the `kudio` logger has a `NullHandler`, so nobody who had not set up logging
  heard about them. Bookkeeping (a name taken, a folder emptied) stays in the
  log.
- **`AudioInfo.peak` is `None` when it was not measured** (`peak=False`); it
  was 0.0, the peak of silence.
- **A missing PyAudio raises `DependencyError`**, like every other extra,
  rather than a plain `ImportError` that `except kudio.KudioError` let
  through. It is still an `ImportError`. sounddevice installed on a system
  without the PortAudio library raises `DeviceError`, which names the system
  package, since pip cannot fix it.
- **`STFT.from_dict` wants every key.** A stored geometry without `hop_length`
  analysed at 256 without a word.
- **`f0`'s `frame_length` defaults to `None`**: 2048 samples, or the next power
  of two that holds more than two periods of `fmin`, 4096 at 88.2 and 96 kHz,
  where the old default was refused. The check now asks for what librosa
  needs, `2 * floor(sr / fmin) + 2` samples; at exactly two periods the lowest
  pitches were lost and the rest looked fine.
- **`file_load(sr=0)` raises `ValueError`**, a mistake in the call, rather
  than an `AudioIOError` about the file. A `Synthesizer` input that does not
  exist raises `FileNotFoundError` rather than `NotADirectoryError`.
- **`melspectrogram` returns float32**, and with a path and `sr=` resamples the
  file; `sr` was ignored for a path.
- **`Standardizer.save` returns the path it wrote**, `.npz` included, and
  `load("stats")` finds `stats.npz`.
- **`copy_waves` returns where each file went, and renames a collision.** Two
  talkers' `a.wav` copied into one folder left one file.
- **`AudioEvaluate.get_df()` is an empty DataFrame** before anything is
  scored, not an empty list.
- **Manifests are written with forward slashes**, so one made on Windows names
  real files on Linux and macOS, and `manifest()` returns them that way. The
  noise-only files of `is_silence=True` are left out of it: the clean file
  they were drawn for is not their reference. They are still in what `syn()`
  returns.

### Fixed — channels last

kudio's multi-channel layout is `(frames, channels)`, and numpy, scipy and
librosa read the last axis as time. Every place that handed such an array on
without saying which axis was time worked along the channels.

- `resample`, and `save_wave(d_sample=)` with it: a 1 s stereo clip came back
  `(16000, 1)`, or as a 2 s mono file.
- `highpass`, `lowpass`, `bandpass` and `bandstop` filtered across the
  channels with `zero_phase=False`, and with the default failed on scipy's
  padding of a two-sample "signal"; `time_stretch` and `pitch_shift` worked
  across them too, and a
  stereo second came back from `time_stretch` as one channel; `trim_silence`
  cut nothing, so `convert_folder(mono=False, trim_db=...)` trimmed no stereo
  file; and `reverb` failed on a stereo array.
- `rt60`, `schroeder_curve` and `apply_rir` read a stereo array as one
  interleaved signal: a two-column 0.6 s response measured 1.20 s with a fit
  of 1.000. `f0` read a stereo clip at half its pitch, 65 Hz for a voice at
  126, and `StreamEnhancer.process` denoised two channels as one. These take
  mono and say so now, as do the metrics, which scored both channels as one
  signal or failed with a numpy shape error depending on the metric, and the
  spectrogram functions, which took each frame for a two-sample signal.
- `play_audio` through PyAudio opened a stereo array as one channel and
  played it at half speed; a path is now read with soundfile, so FLAC and
  FLOAT WAV play too, and a 32-bit integer WAV no longer plays as float noise.
- `Recorder.save(resample=)` resampled the interleaved channels as one signal
  and stretched the take to `record_times`, which it never quite is, moving
  every pitch.
- The command line's `loudness` and `normalize` mixed a stereo file to mono
  first. BS.1770 weights and sums the channels, so the mono mix of a file
  with matching channels read 3 LU low, and `normalize` wrote mono.

### Fixed — numbers that looked fine and were not

- **A corrupt clip measured as silence.** One NaN runs through the K-weighting
  filter into every block after it, the gate dropped them all, and the clip
  read `-inf` LUFS. `loudness`, `true_peak` and `loudness_range` return `nan`
  with a warning, a curve says how many windows it could not measure, and
  `normalize_lufs` and `match_loudness` refuse the clip instead of handing it
  back untouched.
- **STOI's placeholder was averaged in as a score.** pystoi returns 1e-5 for a
  clip with under about 0.4 s of speech, even for a perfect copy. kudio scores
  that `nan` with a warning, `AudioEvaluate` leaves it out of the mean, and
  `eval_metrics` returns `None` for it, as for any metric that cannot run.
- **`pesq` called on its own skipped the length check** every other metric
  has; and one file `AudioEvaluate` could not read or align ended the whole
  run, throwing away every score before it. That file gets `nan` and a
  warning now, and the list goes on.
- **`compare_enhancers` scored any overlap of the reference**, and took a
  keyword no method has (`alpah=0.5`) by running every method on its
  defaults. It applies the length check, and refuses the keyword.
- **A gain-rule parameter outside its range ran anyway.** `gain_floor_db=10`
  clipped every gain to 1 and returned the input as its own denoised version.
  The ranges in `METHODS` are enforced by `spectral_enhance`,
  `enhance_folder` and `StreamEnhancer`. An unknown `noise_kwargs` key is
  named by the estimator's public name, not a private class's.
- **`trad_enhance` dropped the last overlap**, leaving its final 256 samples at
  zero, and returned a different length from its input; it returns float32
  of the input's length. `wavelet_low_pass_filter` took its threshold from the
  signed maximum, so a clip below zero throughout got a negative threshold,
  which soft thresholding turns into a boost: its detail came back amplified
  rather than removed.
- **`STFT.inverse` and `save_spectrogram_as_wave` repeated one value** over the
  last `len(y) % hop` samples instead of rebuilding them, and accepted a
  spectrogram of the wrong shape.
- **Integer audio wrapped round.** `gain` turned an int16 20000 at +6 dB into
  -25631, `normalize` returned int16 as zeros, and `resample` truncated
  integers toward zero. They round and clip within the type now. `save_wave`
  divided every integer type by 32768, writing half-scale int32 clipped and
  uint8 as near-silence; each type is scaled by its own full scale, and
  `DOUBLE` keeps float64.
- **`PCM_24` and `PCM_U8` came out a third and half a step low** on average,
  because only `PCM_16` was rounded before libsndfile's flooring conversion.
  Every PCM width is rounded now.
- **`add_noise_snr` with constant noise returned the clean signal** at any
  SNR, since the mean is removed before scaling. It raises, as `Synthesizer`
  does, and takes mono only.
- **A silent clean file made a silent mixture.** The noise is scaled against
  the speech, so `Synthesizer` wrote a file of zeros labelled with an SNR. It
  raises `SynthesisError` naming the file.
- **The `SpectrogramStream` mel axis put its bands at the wrong
  frequencies**: `n_mels` points spread from `fmin` to `fmax`, where the
  filters are centred on the inner points of `n_mels + 2`. Its docstring also
  promised dBFS, where a mel axis gives band power.

### Fixed — files, folders and names

- **`check_input` matched `.wav` case-sensitively** on Linux and macOS, so a
  recorder's `ZOOM0001.WAV` was found on Windows only, and named directly it
  was refused everywhere. `.txt` lists are read as UTF-8, with or without a
  BOM, rather than in the system code page, where a list of Chinese names
  raised `UnicodeDecodeError`.
- **`convert_folder` and `enhance_folder` dropped an unusable entry without
  a word**, so a list with a typo in it reported "1/1 written". It is a
  failure in the result now. `enhance_folder` mirrors a list from the folder
  its files share, as `convert_folder` does, where it wrote them flat and
  renamed collisions. `convert_folder` leaves a `dst` inside `src` out of the
  input, rather than converting the last run's output again one folder deeper.
- **`kudio enhance --recursive` failed before reading a file** in 3.7.3:
  `enhance_folder` did not take `n_fft`. It takes `n_fft`, `hop_length`,
  `window` and `noise_kwargs` like `spectral_enhance`.
- **`check_file("newdir/")` returned `newdir.wav`.** A trailing separator
  means a folder, whether or not it exists yet.
- **`Synthesizer` copied two talkers' clean files to one name** under
  `RAW_DATA/`; the copies mirror the layout like the mixtures.
  `syn_extra_mode()` resampled its output to the rate of the last `syn()` on
  the same object, takes `is_pool=`, and warns when clean files share a
  name. A requested `is_silence` share rounds to at least one file, and
  `rdn_choice_num` above the corpus size warns.
- **`load_labels` read Audacity's frequency rows as labels**, the `\` lines
  that follow a label with a frequency range.
- **`channel_spec` checked `burst` and `packet_ms` only on the first clip**,
  after `Synthesizer` had emptied its output folder. They are checked when
  the link is described. `channel_tag` named the A-law telephone band
  `banda_law`; it is `telephone_a`, the preset's name.

### Fixed — live audio

- **A timed-out `FetchBuffer` capture kept waiting** on the buffer and took
  the next capture's frames. `get_fixed_size_data(timeout=)` ends the capture
  with the timeout, and a frame count of 2.5 or `'3'` is refused rather than
  read as 2 or 3.
- **`StreamRecorder.start()` after `stop()` failed** on the closed PortAudio
  handle; it opens the stream again and carries on the take.
- **`RemoteStreamReader` dropped chunks that arrived split**, because each
  further read asked for a whole chunk and ran into the next; a chunk of
  silence, all zero bytes, was taken for a text reply; `start()` before
  connecting spun a core forever; and a retry left the previous PortAudio
  instance running.
- **`CheckDevice` raised while describing a device it could not describe**,
  from inside the message meant to explain the failure.

### Fixed — the command line

- `kudio synth` prints the seed it used, so a run without `--seed` can be
  rebuilt, and `--snr` takes fractions. `--channel-loss` and
  `--channel-burst` without `--channel` are refused rather than ignored.
- `kudio convert` applied `--lufs` and `--trim-db` to folders only, and
  ignored them on a single file without a message.
- `kudio enhance --rate` resampled in folder mode only.
- `kudio compare --write-to --noises` named each file after the method, so
  every estimator overwrote the last; the estimator is in the name when
  several run.
- `kudio pitch` takes the same automatic frame length as `f0`.
- `kudio align --metrics` on two files too different in length to score
  stopped with a traceback after printing the delay; it says so and still
  prints the aligned score.
- `kudio room --make FILE` with a file as well is refused rather than ignoring
  the file, and `kudio normalize` refuses digital silence and a `--peak`
  outside (0, 1].

### Added

- **A documentation site**, at https://recklight.github.io/Kudio/: getting
  started, ten guide pages from reading files to recording, the command line,
  migrating from 2.x, and the API reference generated from the docstrings.
  Every figure is drawn by a script in `tools/docs/`, and CI builds the site
  with `--strict` on every push and publishes it from `master`.
- The `[docs]` extra installs what the site needs.

### Documentation

- README: the error and logging sections say what is raised, warned and
  logged; the enhancement caveats and the spreads between gain rules and
  estimators are re-measured on a named clip; the PESQ note no longer claims
  one scale for "most papers"; the migration table lists the helpers that
  moved out of the top level.

## [3.7.3] - 2026-10-14

Documentation and packaging metadata only; nothing in the library behaves
differently.

- The README no longer says the Python badge is waiting for v3.0. 3.0.0 has
  been on PyPI since July.

## [3.7.2] - 2026-10-11

### Fixed — files destroyed in the modes meant to protect them

- **`convert_folder(..., overwrite=False)` could destroy a file it was never
  asked to touch.** `check_file` chose its anti-collision name with
  `Path.with_suffix`, which cannot tell an extension from a dotted name, so
  `take.v2.wav` was renamed to `take.wav` -- over the top of a different
  recording -- and the log said `Renamed to: take.wav` as though a collision
  had been avoided. This is the defect 3.7.0 fixed inside `Synthesizer`; three
  callers were still on the old pattern (`convert_folder`, `enhance_folder`,
  `Recorder.save`). A suffix is now replaced only when it is a real audio
  container, and every candidate name is checked before it is handed back.
- **`AudioDataManager.save_all()` wrote two clips to one file and returned
  `True`.** Clips named `take` and `take.v2` both landed on `take.wav`, and in
  the multi-channel branch so did `take.v2_ch0` and `take.v2_ch1`. Dotted
  names keep their own files now, and two clips that would still share one
  are refused before anything is written, naming both.

### Fixed — numbers that looked fine and were not

- **`split_pairs` put the same example in more than one split.** A manifest
  with a repeated row -- which `snr_ratio=[0, 0.0]` produces -- put 5 of 12
  examples into both train and validation, so part of every score was
  measured on training data, and nothing said so. Exact repeats are now
  dropped with a warning that counts them; rows that list one mixture with
  conflicting fields are refused, because nothing can tell which is true.
- **The reference metrics trimmed any length mismatch without bound.** 100
  samples against 16000 scored `snr 119.70 dB` -- a bit-perfect copy -- with
  99.4 % of the reference never compared. Any mismatch now warns with both
  counts, and past `MAX_LENGTH_MISMATCH` (0.5) it is refused, the way a
  sample-rate mismatch already was. 20 % short is still scored: a frame or two
  of framing slop is not a different recording.
- **A clip with NaN in it was reported as digital silence.** `audio_report`
  read ten NaN samples in an audible tone as a muted microphone, and the CLI
  exited 0. `problems()` now names non-finite samples first and on their own,
  `AudioReport.non_finite_samples` carries the count, `save_wave` refuses to
  write such a waveform in any subtype, and `file_load` logs a warning naming
  the file. Loading is still allowed, because repairing a corpus means reading
  it -- and FLOAT corpora written by 3.7.0 and earlier can hold these.
- **A manifest could name the wrong clean file.** When two different clean
  files collapsed onto one output name, the manifest gave both as the
  reference for the single mixture on disk, so training read a mislabeled
  pair. The clean file that lost is now left out of the run and out of
  `manifest()`, with a warning naming it. A repeated SNR is unaffected: there
  both rows describe the same request, and the file is right for both.

### Fixed — a hang, and a seed lost

- **`AudioBuffer` blocked forever on a frame count of zero or less**, which
  `int(seconds * sr / chunk)` produces for any request shorter than a chunk.
  No traceback, nothing to interrupt. It is refused now. `FetchBuffer` on a
  timeout raises `TimeoutError` rather than returning `None`, which a caller
  could not tell from an empty capture.
- **An aborted `syn()` or `syn_extra_mode()` threw away the seed of the corpus
  on disk.** The seed was drawn before the check that refuses to clear the
  output folder, so a refused run still overwrote `.seed`, and the one value
  that rebuilt the existing corpus was gone. It is drawn after the check now.

### Fixed — packaging and documentation

- **The setuptools build floor was below what `pyproject.toml` needs.** The
  licence is declared in the PEP 639 form, which setuptools reads only from
  77.0.0; 68.2.2 and 76.1.0 both stop with `configuration error:
  project.license must be valid exactly by one definition`. Now `>=77`.
- **`numpy<2.5` is now `numpy<2.6`.** The cap follows numba, and numba 0.67.0
  moved to `<2.6` on 2026-08-11. numpy 2.5 needs Python 3.12, so the old cap
  bit only there -- as a resolution failure, or a quiet downgrade to 2.4.
- **The Changelog link on PyPI was a 404.** It pointed at `main`; the branch
  is `master`.
- **`kudio.util.visual` raised a bare `ModuleNotFoundError`** instead of the
  `DependencyError` that names `kudio[viz]`.
- **`mkdocs build` failed outright**, with or without `--strict`. A docstring
  line in `kudio.core.vad` read as a cross-reference. It arrived with 3.7.1's
  expansion of the API Reference to 23 modules.
- **The sdist carried 28 of the 29 test files** and left out `conftest.py`,
  which six of the others import, so the shipped suite could not be collected.
  `MANIFEST.in` grafts `tests/` whole.

### Added

- `kudio[check]` (pyloudnorm), so the BS.1770 loudness can be checked against
  a reference implementation -- a test no environment could run before.
- `kudio[docs]`, the toolchain `mkdocs.yml` asks for.
- CI: a leg that installs the optional metric back-ends, so the `mir_eval`
  and `pystoi` call sites run at all; and a release job that checks the built
  files with `twine check --strict`, imports every module of the installed
  wheel from outside the source tree, and refuses a tag that does not match
  `kudio/_version.py`.

## [3.7.1] - 2026-10-07

### Fixed — three ways 3.7.0 could still write a corpus nobody could use

Two of these are the same shape as the bugs 3.7.0 fixed: a mixture written
with no speech in it, and a name that silently overwrites another.

- **A noise file with no variation produced an all-zero mixture.** Scaling the
  noise to the requested SNR divides by its standard deviation, so digital
  silence or a constant level made every sample NaN, and int16 wrote NaN as
  zero. The mixture came out with no speech in it and nothing said — exactly
  the damage `is_silence` was doing before 3.7.0, reached a different way. One
  truncated file in a noise corpus was enough. Now a `SynthesisError` that
  names the file and says what is wrong with it, because there is no gain that
  puts a constant at any SNR; a mixture that comes out non-finite is refused
  the same way.

- **Noise-only file names joined two stems with nothing between them.**
  `f"{clean.stem}{noise.stem}_n00.wav"` made `ab` + `c` and `a` + `bc` one
  file, and the de-duplication then kept whichever the RNG drew first. It was
  also the only output in the class that did not follow the `clean_noise_snr`
  shape a reader can parse. The separator is back, and two *different* pairs
  landing on one name now warn instead of one of them vanishing — the
  collapse of one pair's two SNRs, which is intended, is unchanged.

- **`syn()` still returned numpy scalars for numpy input.** 3.7.0's changelog
  said it did not, and the manifest path was indeed fixed, but the four-tuples
  themselves carried `np.int64` whenever `snr_ratio` came from `np.arange`, so
  `json.dumps` on them still raised. Normalised at the source, so every input
  form — list, `arange`, `linspace`, fractional — comes back as a plain `int`
  or `float`.

### Fixed — the PESQ scale could not be reached from anything that reports it

3.7.0 added `mode=`/`scale=` to `kudio.pesq` because enhancement papers mostly
quote the raw narrowband P.862 score and kudio reported wideband MOS-LQO. But
`eval_metrics` did not pass them on and `compare_enhancers` went through
`eval_metrics`, so every table a user actually reads was stuck on the other
scale with no way to ask. `_compute` already took both arguments; nothing
carried them.

- **`eval_metrics(sr, ref, deg, pesq_mode=, pesq_scale=)`** and
  **`compare_enhancers(..., pesq_mode=, pesq_scale=)`**, keyword-only with the
  existing defaults, so no caller's numbers move.

### Fixed — packaging and documentation

- **`[eval]` pins `mir_eval<0.9`.** `sdr` calls
  `mir_eval.separation.bss_eval_sources`, deprecated in 0.8 and removed in 0.9,
  while the floor allowed anything from 0.7 up. The bound is load-bearing, not
  caution.
- **`mkdocs.yml` declared `docstring_style: google`** and not one docstring in
  the library is Google-style — they all use `:param x:`. Every parameter block
  rendered in the API Reference as prose instead of a table. Now `sphinx`.
- **The API Reference was missing most of the library.** It listed nine modules;
  `room`, `align`, `loudness`, `pitch`, `stft`, `spectrogram`, `vad`, `channel`,
  `edit`, `filters`, `dataset`, `report`, `dnsmos` and `enhance` were absent, so
  `vad_frames` and everything else added since 3.3 was undocumented. 23 modules
  now.
- **`Synthesizer.clipped` and `.seed`** were listed as new public API in 3.7.0
  and appeared in no document; both are in the README's module table now.
- `docs/V3_CHANGES.md` was the last place still describing `[eval]` as
  containing pysepm.

### Fixed — the test suite was carrying the bug it had just fixed

- **`tests/conftest.py`'s `write_wav` still used
  `(wave * iinfo(int16).max).astype(int16)`** — the exact wrap-around this
  release fixed in `Synthesizer`. The shared noise fixture peaks at 1.207, so
  twelve of its samples came back with the sign flipped and every test mixing
  against it was mixing against clicks it never asked for. It writes through
  `save_wave` now.
- **That fixture's clean files are 12 dB quieter.** A mixture at its lowest SNR
  peaked at 2.68, so twelve unrelated tests emitted the new clipping warning. A
  warning that fires on a dozen of your own tests is one nobody reads; the
  deliberate case has its own `loud_corpus` fixture and still warns. Down from
  14 warnings to 1.

## [3.7.0] - 2026-10-06

A speech-enhancement project ran kudio's synthesis and evaluation end to end on
TIMIT and wrote down everything that went wrong. Nearly all of it was silent:
files that wrapped around instead of clipping, a `reg` corpus that met a
seventh of its noise x SNR combinations, a seed the process pool ignored, and a
PESQ that was on the wrong scale where it ran and NaN where it did not. This
release fixes those, and two worse problems found on the way: `is_silence=True`
wrote a dataset without a word of speech in it, and `overwrite=True` could
delete the corpus it was reading from.

### Fixed — synthesis

- **`is_silence=True` wrote a dataset with no speech in it.** The flag was meant
  to add a 2% share of noise-only files, but it was handed to *every* file, so
  every mixture was written as noise alone. `syn_extra_mode`, which always sets
  it, has never produced a mixture with speech in it. Only the share is
  noise-only now, and it is drawn without replacement.

- **A mixture past full scale wrapped around.** `(y * 32767).astype(int16)`
  turns 1.2 into -0.8 without a word, so a loud talker at a low SNR came out
  with clicks of the opposite sign. Files are now written through `save_wave`,
  which clips at full scale and rounds to the nearest step; `syn` warns with
  the count and the worst peak, and **`Synthesizer.clipped`** lists the files.
  **`Synthesizer(subtype=...)`** writes `'PCM_24'` or `'FLOAT'` instead, and
  `'FLOAT'` keeps what is past 1.0.

- **`reg` mode met only `lcm(noises, SNRs)` of the combinations.** The noise and
  SNR lists were cycled side by side, so 14 noises against 21 SNRs paired each
  noise with just 3 SNRs. The SNR now steps on by one at the end of every such
  cycle, so any `noises x SNRs` consecutive outputs hold every combination
  exactly once. **When the two counts share no factor nothing changes**: the
  assignment is exactly the cycle it was.

- **`seed=` did not survive the process pool.** The noise crop came from each
  worker's own global RNG, so a seeded run with `is_pool=True` differed from
  itself (77 of 210 files on a test corpus) and from a serial run. Each file now
  draws its crop, its room and its packet loss from the seed and its own path
  below the output folder -- the path rather than the name, because a mirrored
  tree repeats names and every TIMIT talker reads `sa1`. `syn()` also no longer
  calls `np.random.seed()`, which reset the caller's global RNG.

- **`overwrite=True` could delete the corpus.** `syn()` emptied the output
  folder with `rmtree`, as `syn_extra_mode()` always did, so an `out_path=` one
  level too high took the clean and noise files with it. Both now refuse, with
  `SynthesisError`, to empty a folder that holds an input file, the working
  directory or the home directory. An unknown `mode=` is refused before
  anything is deleted rather than after.

- **`Synthesizer` takes what its docstring always said it took.** The
  constructor demanded two folders, though the docstring promised anything
  `check_input` accepts, so a subset -- TIMIT's SI and SX without SA -- had to be
  copied out first. A list or a `.txt` manifest now works, mirrored below the
  folder its files share or below **`root=`**; an entry that is not an
  existing `.wav` is reported instead of skipped in silence.

- **Fractional SNRs and dotted names.** `with_suffix` read `.5dB` as a file
  suffix, so 2.5 dB and 2.7 dB both became `..._2.wav` and one overwrote the
  other, and every output of `take.v2.wav` was called `take.wav`. A decimal
  point is written `p` (`2p5dB`, `n2p5`) and names are joined, not suffixed.
  Integer SNRs keep their names exactly. The manifest records 2.5 rather than
  `int(2.5)`, and output names that still collide -- a tree flattened with
  `mkdir_parents=False` -- get a warning.

- `rdn_choice_num` samples without replacement; a repeat was written twice
  under one name. `syn_extra_mode` keeps its scratch files in a temporary
  directory rather than in `./tmp_mixed`, which it deleted on the way in.

### Fixed — evaluation

- **PESQ depended on `pysepm`, which is not on PyPI, and at 16 kHz failed on
  NumPy 2.** pysepm returns `(np.NaN, MOS-LQO)` at 16 kHz, and `np.NaN` is gone
  from NumPy 2, so every 16 kHz score raised; `AudioEvaluate` logged each one
  and averaged them into NaN. PESQ now runs on the **`pesq`** package from PyPI,
  installed by the new extra **`kudio[pesq]`**. PyPI carries it as source only,
  so it needs a C compiler -- the MSVC Build Tools on Windows -- and it is left
  out of `[eval]` and `[all]` so that those still install anywhere.

- **And it was not the PESQ most papers report.** kudio reported P.862.2
  wideband MOS-LQO at 16 kHz. Enhancement papers that say "PESQ" mostly mean
  the raw narrowband P.862 score, a different scale: 2.03 raw is about 1.65
  MOS-LQO. **`kudio.pesq(ref, deg, sr, mode=, scale=)`** takes the bandwidth
  (`'nb'`, `'wb'`, `'auto'`) and the scale (`'lqo'`, or `'raw'` by inverting
  P.862.1). The default is still what kudio reported, and
  `AudioEvaluate(pesq_mode=, pesq_scale=)` passes both through.

- **`AudioEvaluate` read files with `scipy.io.wavfile`**: int16 values, WAV
  only, and TIMIT's own NIST SPHERE refused. It reads with `file_load` now --
  float, mono, anything soundfile opens. Each file is read once per method
  instead of once per metric, the noisy baseline is scored once instead of once
  per method, lists that do not pair up are refused, and `get_df()` no longer
  adds another 'Baseline' row each time it is called.

- **`eval_metrics` logged its install hints on every call**, which per-file
  scoring turned into a wall of warnings. A missing back-end is reported once
  per process. A metric that cannot run on a pair -- PESQ off 8 and 16 kHz, or
  on silence -- comes back `None` and leaves the others standing, where one
  failure used to take every score with it.

### Fixed — features and files

- **Spectrograms of float32 audio came back float64 under NumPy 2.** The
  epsilon added to the spectrum was a numpy float64 scalar, and NumPy 2's
  promotion rules let it widen the whole array: twice the memory for every
  feature matrix. Back to float32. The values are unchanged, including the
  -31.3 a digitally silent bin is floored at, and `norm=True` still computes its
  statistics in float64.
- **`save_spectrogram_as_wave` wrapped around** past full scale, the same way
  the Synthesizer did. It writes through `save_wave` now.
- **`save_wave` clips and rounds fixed-point output itself.** libsndfile's
  conversion floors, leaving every 16-bit sample half a step low, and wraps
  around unless clipping is switched on. int16 input is scaled by 1/32768, the
  step soundfile reads it back with, so a 16-bit array is written bit for bit
  (it was 1/32767).
- **`convert_folder` flattened a list.** Only a folder input was mirrored, so
  two talkers' `sx133.wav` collided and one was renamed `sx1332.wav`, which
  reads like another sentence. A list or manifest is mirrored below the folder
  its files share, or below the new **`root=`**.

### Added

- **`score(ref, deg, sr, metrics=None)`** returns `{name: value}` for any of
  `snr`, `si_sdr`, `segsnr`, `sdi`, `pesq`, `stoi`, `estoi`, `sdr`. With
  `metrics=None` it runs everything that can run here; a metric asked for by
  name that cannot -- not installed, or the wrong rate -- raises instead of
  quietly going missing.
- **`AudioEvaluate.per_file()`** -- every file's scores, one row per method and
  file, with the clean, noisy and scored paths (`rows` holds the same as plain
  dicts). Means cannot be broken down by noise, SNR or talker; this can, by
  joining on `noisy` with a `save_manifest` manifest.
- **`sdi(ref, deg)`** -- the speech distortion index, `sum((x - x_hat)**2) /
  sum(x**2)`: `snr` on a linear scale, under the name the enhancement
  literature reports it by.
- **`ContextFrames(utterances, context=5)`** -- `stack_context` over a whole
  corpus, a batch at a time. Stacking ±5 frames up front makes a training set
  eleven times the size of its features (about 8 GB in the project that asked);
  this keeps the features once, padded at each utterance's own edges, and
  `bank[batch]` is row for row `np.vstack([stack_context(...)])[batch]`.
- **`vad_frames(y, sr, hop_length=256)`** -- `vad` as one boolean per STFT
  frame, lined up with the rows of `waveform_to_spectrogram` and `STFT`.
  Padding defaults to 0 here: it keeps a cut from clipping a consonant, but on
  a frame label it only marks the frames next to the speech as speech too.
- `Synthesizer(root=, subtype=)`, `Synthesizer.clipped`, `Synthesizer.seed` (the
  base the last run drew from, so a run given no seed can be rebuilt),
  `syn_extra_mode(seed=)`, `convert_folder(root=)`.

### Changed

- **A dataset rebuilt from the same seed is not bit-identical to 3.6.0's.**
  Crops are drawn per file, the clean-file sample and the silence share come
  from `default_rng(seed)`, 16-bit samples are rounded instead of truncated,
  and `reg` pairings move where the counts share a factor. Rooms and packet
  loss of files directly in the output folder keep their 3.6.0 seeds; in a
  mirrored tree they are now seeded from the path, not the name.
- **`seed=None` is now random for everything.** In 3.6.0 it fixed the rooms and
  packet loss (as if `seed=0`) while the crops varied. Now one base is drawn
  from numpy's global RNG -- so `np.random.seed()` still reproduces a run -- and
  kept in `Synthesizer.seed`.
- `syn()` returns each SNR as it was given rather than as a numpy scalar, which
  `json` could not serialise; `Pair.snr_db` is annotated `Optional[float]`.
- The manifest names a link only on files that went down it, so the output of
  `syn_extra_mode`, which applies none, no longer claims one.
- In `AudioEvaluate` a metric failing on one file gives NaN for that file with
  a warning; only PESQ did that before, and a failing STOI or SDR stopped the
  evaluation.

### Not changed

- **`n5` beside `5dB`** in output names: the asymmetry is old, but scripts and
  folders already depend on those names, and the manifest records each SNR
  exactly. Only the names that were broken -- fractional SNRs, dotted stems --
  changed.
- **The -31.3 floor of a digitally silent bin.** Every model trained on kudio's
  log spectrogram has seen it, so changing it would move their inputs; clamp
  it where it matters, e.g. `np.maximum(spec, -10)`.

### Migration

```bash
pip install -U "kudio[eval,pesq]"      # [pesq] needs a C compiler
```

```python
# the PESQ enhancement papers quote: raw narrowband P.862
ev = kudio.AudioEvaluate(clean, noisy, pesq_mode="nb", pesq_scale="raw")

# a subset without copying it out, written without clipping
syx = kudio.Synthesizer(si_sx_files, "noise/", out_path="mixed",
                        root="TIMIT/TRAIN", subtype="FLOAT")
```

## [3.6.0] - 2026-09-18

The chain between a talker and a file, end to end. 3.5 could describe a room
and 3.5's late work could describe a link; neither could put a corpus through
both, and packet loss was still scattered evenly in a way no real link loses
it.

### Added — burst loss

`dropouts` lost packets independently. Real links lose them in runs — a fade,
a queue overflowing, a handover — and a run is far harder to conceal than the
same number of packets scattered about, because there is no recent speech
left to repeat.

- **`dropouts(y, sr, loss=0.05, burst=6)`** — one parameter, the mean run
  length in packets. The run lengths are geometric, which is exactly what
  Gilbert's two-state chain gives; what is *not* taken from Gilbert is its
  marginal, because that chain fixes the loss *rate* and then delivers 4.1%
  or 6.3% on any given file. **The count stays exact**, as it has been since
  `dropouts` existed, because the number on a corpus label should be true of
  the corpus. Runs are kept at least one surviving packet apart, without
  which two runs could abut and the mean would drift above the one asked for.

- **`independent_burst(loss)`** — because the obvious guess about the default
  is wrong. Independent draws land side by side sometimes, so they already
  average `1 / (1 - loss)`: 1.05 packets at 5%, 1.25 at 20%. `burst=1` is
  therefore a request for something **less** clustered than leaving it alone,
  not the same thing, which is why independent loss is spelled `burst=None`.

- A regime that cannot be built is **refused rather than fudged**: 80% of
  packets lost in runs of one needs four survivors for every loss and there
  are not four. Silently merging the runs would mean the burst length asked
  for was not the burst length delivered.

- **`kudio channel --burst 6`**, and the CLI now reports the runs it made:
  `20 packet(s) lost in 5 run(s) of 4.0`.

### Added — the link as one thing

Five functions in the order a link applies them, so a corpus builder, the
command line and a GUI describe a link the same way instead of each
inventing an argument order.

- **`apply_channel(y, sr, spec, seed=)`** — band and rate, then the codec,
  then the converter, then the wire. The loss goes last because packets are
  lost from the encoded signal; running it first would let the resampler
  smear every hole into its neighbours.

- **`channel_spec(spec)`** normalises a preset name, a dict, or a preset with
  overrides (`{'preset': 'voip', 'loss': 0.1}`) into one dictionary, and
  returns `{}` — falsy — for "no link", so one check downstream means "is
  there a channel at all" instead of five that have to agree.
  **`CHANNEL_PRESETS`**: `telephone`, `telephone_a`, `voip`, `mu_law`,
  `a_law`, `none`.

- **`channel_tag(spec)`** writes it short and stable for a manifest or a
  label: `telephone`, `mu_law+loss5%x4`, `8bit`.

### Added — the whole chain in `Synthesizer`

    talker → room (rt60=) → + noise (snr_ratio=) → link (channel=) → file

- **`Synthesizer(..., channel='telephone')`** runs the finished mixture down
  a link. The **mixture**, not the speech: a wire carries whatever reached
  the microphone, noise included, which is why it comes last. Packet loss is
  seeded per output file, so no two files lose the same packets and a rerun
  loses the same ones again.

  **The link is deliberately not an axis.** `snr_ratio` and `rt60` are swept
  and multiply the output count; `channel` is constant for the run and does
  not appear in the file names. A corpus is recorded over a phone line or it
  is not — that is a property of the corpus rather than a dimension to cross
  with the others, and four axes already multiply enough. To compare two
  links, run the Synthesizer twice into two directories.

- **`Synthesizer(..., write_targets=True)`** writes the reverberant-but-clean
  signal to `OUT/TARGETS/`, named to match its mixture. The clean file is
  still the dry target, so the default asks a model to undo the room as well
  as the noise; this is the same speech with the room left on, for training
  that should remove the noise and **leave** the room. The link does not
  reach it: asking a model to reproduce G.711 is not denoising.

  Passing it without `rt60=` warns instead of writing, since with no room the
  clean file already *is* that signal.

- **`Pair` gained `channel` and `target`**, both optional and last, so an
  older manifest still loads and a newer one still loads in an older kudio.
  `channel` is the tag **string** rather than the spec dict it came from:
  `Pair` is frozen so that `split_pairs` can check for overlap with a set,
  and a dict field would quietly have made it unhashable again.

- **`kudio synth --channel telephone --channel-loss 0.05 --channel-burst 6
  --targets`**.

### Added — the channel: what the transmission did to it

Speech arrives degraded in three ways and kudio could describe two. Noise is
**additive**, a room is **convolutive**, and the channel is neither: a codec
quantises, a link drops packets, a converter throws bits away. None of that
can be written as `y + n` or `y * h`, which is why none of it was here.

- **`mu_law(y)` / `a_law(y)`** — the G.711 companding curves, quantised to
  8 bits and back. These are the curves, not the standard's piecewise-linear
  segment layout: measured against Python's own `audioop` (the reference, and
  removed in 3.13, which is why it cannot be a dependency) the round-trip SNR
  agrees to 0.5 dB and the waveforms to 0.013 full scale.

  **The property that makes companding companding is asserted, not assumed.**
  Logarithmic steps follow the signal down, so on this repo's fixture mu-law
  holds 37 dB of SNR across a 24 dB range of levels while linear 8-bit loses
  the textbook 6 dB per halving — 38.1 dB down to 15.1 dB. A wrong
  implementation would not have that.

- **`bit_depth(y, bits=8, dither=False)`** — linear requantisation. A
  full-scale sine comes back at `6.02 * bits + 1.76` dB, matched to within a
  decibel from 4 bits to 16, which is the figure with no fudge in it.

- **`dropouts(y, sr, loss=, packet_ms=, conceal=)`** — packet loss at 20 ms,
  G.711 over RTP's packet. The loss rate is **achieved exactly** rather than
  drawn per packet, so a corpus labelled 5% carries 5%; `return_spans=True`
  says where the holes are; `conceal='hold'` repeats the last packet that
  arrived.

  Worth knowing before choosing a concealment strategy: **whether holding
  measures better or worse than a hole is decided by an accident**, not by
  merit. When the talker's pitch period divides the packet the repeat lands
  in phase and is numerically exact; when it does not, it is a larger error
  signal than the silence it replaced. On a 20 ms packet a 150 Hz voice
  (3.00 cycles) scores +275 dB and a 133 Hz voice (2.66) scores worse than
  the hole. Anything picking a concealment strategy on a waveform metric is
  measuring that accident, and there are tests from both sides.

- **`telephone(y, sr)`** — the lot, in the order a line does it: band-limit
  to 300-3400 Hz at the original rate, resample to 8 kHz, compand, come back.
  Same length and same rate as it went in, and `audio_report` afterwards says
  band-limited at about 3.4 kHz.

- **`kudio channel in.wav out.wav --telephone / --codec / --bits / --loss`**.

### Added — rooms in `Synthesizer`

`Synthesizer` mixed clean × noise × SNR. It now takes `rt60=` as a fourth
axis, putting the speech through a room **before** the noise is mixed in --
the order a microphone meets them, which is also the only order in which the
SNR you asked for is the SNR you get, since it is then measured against the
reverberant speech.

- **`Synthesizer(..., rt60=[0.3, 0.8], drr_db=3.0)`**. Each output file gets
  its own room, seeded from the base seed and the output name, so the same
  request rebuilds the same dataset and two files sharing an RT60 still get
  different rooms. The room goes in the filename (`..._rt300ms.wav`).

- **`Synthesizer.manifest()` → `List[Pair]`**, and `Pair` gained an optional
  `rt60`. The room travels there rather than in `syn()`'s return value, which
  is still the four-tuple it has always been: a return shape that changes
  when a keyword is passed is the bug this codebase has met most often, and
  `file_load`'s two channel layouts are in the notes as exactly that.

- **`kudio synth --rt60 0.3 0.6 --manifest out.json`**.

The clean file stays the training target, so a model trained on this data is
asked to undo the room as well as the noise. That is the standard setup and
worth choosing on purpose; keeping the room and removing only the noise needs
the reverberant signal as the target, which this class does not write.

### Added — reverberation, as a degradation and a measurement

kudio could degrade a recording with *additive* noise and take it back out
again. Reverberation is the other kind and it is **convolutive**: the room adds
nothing, it smears what was already there across the following half-second.
Nothing here could describe that or measure it.

- **`rir(sr, rt60=, drr_db=, seed=)`** — a synthetic room impulse response:
  exponentially decaying noise, the textbook model of a diffuse tail. Both
  knobs are solved for against the measurement's own definitions, so what you
  ask for is what comes back: RT60 within 3%, DRR within 0.01 dB. Seeded, so a
  dataset built from it can be rebuilt.

- **`apply_rir(y, ir)`** — put a clip in a room, keeping its length and its
  level by default. A clip that grows every time it is processed stops lining
  up with everything measured against it, and an augmentation that also
  changes the level is measuring two things at once.

- **`rt60(ir, sr)` → `Reverberation`** — T20, T30, EDT, C50, C80 and DRR to
  ISO 3382-1, all off one backward-integrated decay curve
  (`schroeder_curve`) so they cannot disagree about what the tail did.

- **`Reverberation.reliable()` asks two questions**, because the R² of the
  fitted decay only answers one of them. Backward integration makes *any*
  signal decay monotonically, and over a long clip that decay can fit a
  straight line well — this repo's speech fixture reaches R² 0.94 and comes
  back claiming a 3.9 second room. What gives it away is EDT and T30
  disagreeing by a factor of eleven, where a real room has them within a few
  per cent. Truncated impulse responses are caught by the R² alone: cutting a
  1 s room to 50 ms still plunges 35 dB, and reports 0.19 s.

- **`kudio room`** measures an impulse response, `--make` synthesises one and
  `--apply` puts a file through one. Exits non-zero when the decay does not
  look like an impulse response.

### Added — the alignment every reference metric assumes

`si_sdr`, `snr`, `segmental_snr`, PESQ and STOI all compare sample *i* of one
signal with sample *i* of the other. They trim to the shorter of the two and
assume the rest lines up, which is right for a process that returns what it
was given and silently wrong for anything with latency — a filter, a
resampler, a frame-based denoiser that pads.

Measured on a clip scored against **itself**:

| shift | SI-SDR |
|---|---|
| 0 | +145 dB |
| 1 sample | +10.6 dB |
| 1 ms | -10.1 dB |
| 10 ms | -11.1 dB |

One sample costs 134 dB. A millisecond scores worse than not processing at
all, and nothing in the number says "these are misaligned" rather than "your
denoiser destroyed the signal".

- **`find_delay(ref, deg, sr)` → `Alignment`** — the offset to the nearest
  sample, by normalised cross-correlation. `delay` is signed (positive means
  *deg* starts later), `apply()` trims a pair to what they share, and a delay
  measured once on one passage can be applied to every file that came through
  the same chain.

- **`align(ref, deg, sr)`** — the pair, lined up and ready for a metric.

- **The confidence is half of it.** Cross-correlation always has a maximum, so
  it always returns a delay — including for two recordings that have nothing
  to do with each other. `Alignment.correlation` separates "the same take, 8 ms
  apart" from "two unrelated files", and `align` refuses below
  `MIN_CORRELATION` rather than putting a confident-looking number on nothing.
  Pass `min_correlation=0` to insist.

- **Polarity is reported, not silently fixed.** An inverted copy correlates at
  -1: it is the same take, and something in the chain flipped it. That is a
  finding, so `Alignment.inverted` names it instead of correcting it on the
  way past.

- **`kudio align ref.wav deg.wav`**, with `--metrics` to show the score before
  and after, `--write-to` for the aligned pair, and `--max-seconds` to bound
  the search. Exits non-zero when the two do not look like the same recording.
  Resamples first when the rates differ, because a sample offset is not a
  quantity yet if the two are not on the same clock.

### Added — loudness over time

`loudness()` answers "how loud is this" with one number, which is what you
normalise against and useless for finding the moment somebody turned away from
the microphone.

- **`loudness_over_time(y, sr, window=...)` → `LoudnessCurve`** — the two
  EBU Tech 3341 meters, `MOMENTARY` (400 ms) and `SHORT_TERM` (3 s), stepped
  by 100 ms. The curve carries `loudest()`, `quietest()`, `spans_above()`,
  `spans_below()` and a `summary()`; times are window **centres**, so a curve
  drawn beside a waveform lines up with the audio that produced it.

  **These are ungated.** Gating belongs to the integrated figure, where its job
  is to stop pauses dragging the average down. A curve with holes punched in it
  where the gate fired would describe a recording that stops existing between
  words.

- **`loudness_range(y, sr)` → LU**, to EBU Tech 3342: the 10th-to-95th
  percentile spread of the short-term loudness, after an absolute gate at
  -70 LUFS and a relative one **20 LU** below the power-mean of the survivors
  — not the integrated measurement's 10. The percentiles are what stop a
  single door slam from setting the answer.

- **`true_peak(y, sr)` → dBTP**, to BS.1770-4 Annex 2. The largest sample is
  not the largest value the signal reaches: samples are points on a curve that
  passes between them, and the curve overshoots. A full-scale sine at a quarter
  of the sample rate, phased so every sample lands at 0.707, reads -3.01 dBFS
  and +0.1 dBTP — and the converter, encoder or resampler that reconstructs it
  meets the second number. Oversampled to at least 192 kHz as the standard
  asks, in chunks, so a ten-minute file is not held four times over.

  That case is in the test suite, which is the point of having it.

- **`kudio loudness`** now prints range, true peak and the loudest and quietest
  moments with their timestamps. `--over-time` draws the curve, `--momentary`
  switches windows, and `--target -23` exits non-zero when a file misses its
  delivery level by more than 1 LU.

### Internal

K-weighted block power moved to one private helper (`_block_power`), which the
integrated measurement and both curves now share — they had begun to need the
same thing and written twice they would eventually have disagreed about what a
block contains.

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
- `KudioConfig` defaults trimmed to audio-only (training keys removed).

### Removed
- Deprecated compatibility shims `_config.py`, `_enh/`, `version.py`,
  `util/others.py`.

## [1.0.0] - 2026-07-11

First packaged release: a standalone, installable, MIT-licensed package.

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
