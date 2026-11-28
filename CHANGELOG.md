# Changelog

All notable changes to **Kudio** are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

kudio no longer needs librosa. Resampling, the short-time transform and its
inverse, mel bands and MFCC, cutting at silence, pYIN, the time stretch and the
plots are kudio's own, on numpy, scipy and soxr, and give librosa's numbers to
the bit wherever the arithmetic is the same. Where a result moves, it is listed
first. A clip, a setting and a flag are read the same way in every function, and
work shared out among processes no longer waits for ever on one that died.

### Changed — results that move

- `f0` on Python 3.12 and later. librosa 1.0, which pip installs there, leaves
  out of pYIN's search every move less likely than one in ten thousand, and `f0`
  had that search. kudio searches all of them, as librosa 0.11 did on Python
  3.11 and earlier, so every Python now gives what 3.11 gave. At the default hop
  of 10 ms 4 of 528 hard signals are voiced otherwise, a sweep at 22.05 and at
  48 kHz in float32 and in float64, in 1 and 3 frames. With a longer hop it
  happens more often. Of a minute of a voice with pauses in it, 16 kHz and 65 to
  400 Hz over white noise 60 or 40 dB under the voice, 1.0's search voiced no
  frame otherwise up to a hop of 30 ms, 1 of 1501 at 40 ms, 5 of 1334 at 45, 267
  of 1201 at 50, 63 of 1001 at 60 and 21 or 20 of 858 at 70
  (tools/validation/against_librosa.py).
- From a rate of the 11025 Hz family (11.025 to 176.4 kHz) to 12, 24, 48, 96 or
  192 kHz a clip has one sample less for most of the lengths that come out
  whole: a second of 44.1 kHz is 48000 samples at 48 kHz, and was 48001, the
  last one zero, or with `res_type='fft'` the whole second drawn out by it. The
  length is `ceil(n * target_sr / orig_sr)` worked out in whole numbers; in
  floating point the product comes out a hair over a whole number for those 25
  pairs. The other 157 pairs of the 14 usual rates are as they were, and so is a
  rate that is not whole.
- `time_stretch` and `pitch_shift` give other samples: two seconds of a voice
  stretched six ways, by up to 0.47 % of the peak against librosa 0.11 and 4.3
  parts in a million against 1.0, and at the end by up to 0.8 % and 1.1 %. The
  algorithm is librosa's, a phase vocoder in frames of 2048, and two things
  about it are not:
  - The phases are kept in float64, so a long clip keeps its pitch to the end.
    librosa 0.11, which Python 3.11 and earlier got, kept those of float32 audio
    in float32 and rounded the frequency of every bin, more coarsely as the clip
    went on. 3001.7 Hz at 48 kHz stretched by 1.25 came out 0.03 cents sharp
    after ten seconds and 1.5 cents flat after five minutes, 187.3 Hz 1.3 cents
    sharp and 20640 Hz 0.9; of the 3001.7 Hz tone 64 dB then lay between it and
    what was left, and 4 dB in the seconds in which its frequency stepped. It is
    117 dB throughout now, and no cent off. float64 audio was kept in float64
    before too.
  - Past the last frame the magnitudes are held and the phases turn on as they
    last did. librosa 0.11 faded the end of a clip that was speeded up (a voice
    1.1 dB down over its last 50 ms at 1.25, noise 0.9 dB), and 1.0 drew the
    magnitudes on and let the phases stand.

  The length is what it was, `round(n / rate)`.
- `f0` calls a frame unvoiced where fewer than two periods of `fmin` are in the
  clip, 493 samples at 16 kHz, and warns, giving the lowest pitch that can be
  told. Ten samples of anything were 65 Hz, voiced.
- `res_type` of `resample` takes `soxr_vhq`, `soxr_hq`, `soxr_mq`, `soxr_lq`,
  `soxr_qq`, `fft`, `scipy` and `polyphase`; any other name is a `ValueError`
  that lists them, looked at whether or not there is anything to resample. The
  other names librosa knows took resampy or samplerate, which kudio does not
  install, and ended in a `ModuleNotFoundError`; so did a misspelt one.
- `audio_report` of a clip with NaN or inf in it says how many in a warning and
  measures it as before; code that runs with warnings as errors gets one there.
  `kudio report` leaves the warning out, since the report lists them first.
- `loudness_over_time` and `loudness_range` refuse a clip with NaN or inf in it,
  as the other measurements of a clip do. They gave a curve with a `max` of -inf
  and a `range_lu` of 0, the readings of silence. `rt60` refuses a response that
  is all zeros, which gave C50, C80 and DRR of +inf.
- The plots of `kudio.util.visual` raise `FeatureError`, a `ValueError`, where
  they raised librosa's `ParameterError`, which is not one; see Changed for what
  they draw.
- On Python 3.11 and earlier librosa 0.11 brought audioread, which reads m4a,
  aac and webm with ffmpeg. It takes `pip install "kudio[compressed]"` there too
  now, as it took audioread on 3.12 and later. Until then such a file is an
  `AudioIOError` that says what to install.
- `signature()` has no librosa under `pkg`, so a signature made with this
  release is not the one 4.3.0 made, which carries another kudio version anyway.

### Changed

- kudio installs without librosa and what came with it: numba, llvmlite,
  scikit-learn, joblib, pooch and others. The core is numpy, scipy, soundfile,
  soxr, PyYAML, tqdm and PyWavelets, 11 packages and 156 MB on Python 3.13 where
  it was 30 and 317 MB, and 10 and 198 MB on 3.9 where it was 29 and 323. numpy
  has no ceiling; the one there was is numba's. `import kudio` takes as long as
  it did, 0.25 s on Python 3.9 and 0.33 on 3.13; the first `f0` takes 0.19 s
  where it took 1.35, and the first `resample` 8 ms where it took 1.18 s,
  numba's cache warm (0.23 s against 1.29 and 9 ms against 0.54 s on 3.13).
- `wavelet_low_pass_filter` computes its own discrete wavelet transform, with
  PyWavelets' filters for its 106 discrete wavelets, and gives PyWavelets'
  results to the bit. PyWavelets is not a dependency; `[check]` brings it for
  `tests/test_against_pywavelets.py`. A name that is none of the 106 is refused
  before anything else is looked at, a value that is not a name is a
  `TypeError`, and a clip too short for one level comes back as a copy.
- Live audio runs on sounddevice alone, and `[audio]` is sounddevice; on Linux
  it needs the system's `libportaudio2`. Every class and function numbers the
  devices one way, `list_devices` names each one's host API, and
  `kudio devices` prints one list. `CheckDevice` tries a device with a
  callback stream and `Recorder` records with one, which Windows' WDM-KS needs;
  a device that opens and will not start is a `DeviceError` that names it.
  Given no device, every one of them takes the system's default input, or with
  none the last input that opens and starts; a device that does not record is
  a `DeviceError` from `record` too.
- `sdr`, `bss_eval` and `sdr_improvement` compute BSS Eval themselves, mir_eval
  0.8.2's numbers to the bit, and `[eval]` is pystoi alone; `[check]` brings
  mir_eval for `tests/test_against_mir_eval.py`. References that make the
  normal equations singular are scored by least squares.
- `stoi` and `estoi` compute STOI and ESTOI themselves, pystoi 0.4.1's steps
  without the noise its ESTOI draws from numpy's global random state, and
  `[eval]` is empty; `[check]` brings pystoi for `tests/test_against_pystoi.py`.
- `resample`, and with it `file_load(sr=...)`, `save_wave(d_sample=...)`,
  `convert_folder(sr=...)`, `Synthesizer`, `score_dataset(sr=...)`, `telephone`
  and DNSMOS's input, resample with soxr itself. The samples are the same to the
  bit: 1097 calls at 13 pairs of rates, eight resamplers, twelve sample types
  and eight kinds of file, against 4.3.0 on librosa 0.11 and on 1.0.1.
  `signature()` names soxr under `pkg` where a file or DNSMOS's input was
  resampled.
- soxr that is there and will not import, one built for another numpy say, is a
  `DependencyError` that says what it failed on, before `Synthesizer` empties a
  folder.
- `resample` refuses samples that are not finite, and an array of no or three
  dimensions, with a `ValueError` that says how many or which shape. Both were
  librosa's `ParameterError`, which is neither a `ValueError` nor a
  `KudioError`.
- A measure asked for at a rate it is not defined at says to use
  `kudio.resample(y, sr, 16000)`; it named scipy's `resample_poly` beside it,
  for machines without librosa.
- The short-time transform and its inverse are kudio's own:
  `waveform_to_spectrogram`, `spectrogram_to_waveform` and the functions on
  files, `STFT`, `denoise_wav2spec` and `denoise_spec2wav`, `wave_slicing`,
  `spectral_enhance`, `compare_enhancers` and `enhance_folder`. They are
  librosa's to the bit, 1062 transforms and 1644 inverses over 16 and 15
  geometries against librosa 0.11 and 1.0.1, and of the 1224 calls 4.3.0
  answered 1202 are answered the same; the other 22 are among the fixes below.
  The first call no longer loads numba: 0.41 s instead of 1.2 on Python 3.9,
  0.64 instead of 1.1 on 3.13. Ten minutes at 48 kHz are turned back from their
  spectrogram in 964 MB at the peak instead of 1718, and enhanced in 3091
  instead of 3595.
- Samples that are not finite, and sizes that are not a whole number of 1 or
  more, are refused by these functions by name, with a `FeatureError`. A NaN was
  librosa's `ParameterError`, and to `denoise_wav2spec` a "silent clip"; a bad
  size was whatever numpy, scipy or librosa made of it, and `win_length=0` a
  spectrogram of silence. 512.0 is 512.
- A clip shorter than a frame is transformed with a warning that names the
  function and points at the line that called it. It was librosa's warning.
- `StreamEnhancer.process` refuses a block with NaN or inf in it and stays as it
  was. One NaN got into the noise estimate, and every block after it was NaN
  until `reset()`. `NoiseTracker.update` and `DecisionDirected.update` take
  their frame as it comes: checking it there made a frame of `StreamEnhancer`
  and of `spectral_enhance` 8 % slower, and the two refuse what they are not to
  be given before it reaches them.
- Mel bands, MFCC and silence are kudio's own too: `melspectrogram`, `mfcc`,
  `mfcc_from_files`, `SpectrogramStream(n_mels=...)`, `trim_silence`,
  `split_on_silence`, `convert_folder(trim_db=...)` and `kudio trim`. They are
  librosa's to the bit: 5880 sets of filters, 912 spectrograms and cepstra and
  2100 clips cut two ways, against librosa 0.11 and 1.0.1. Ten minutes at 48 kHz
  are trimmed in 0.11 s and 258 MB at the peak, 147 of them there before, where
  it took 1.9 s and 851 MB.
- A mel band with no FFT bin in it is counted in a warning that names the
  function and points at the line that called it, with what to do about it, and
  says so of an `fmax` above half the rate. It was librosa's warning, which did
  not count, and left out a lowest band that starts at 0 Hz.
- `SpectrogramStream` with a hop longer than its frame gives the same columns
  whatever blocks the audio comes in: the samples between two columns were
  skipped only as far as they had arrived. A block with NaN or inf in it is
  refused, and the window stays as it was.
- Pitch is kudio's own: `f0` and `kudio pitch`. The numbers are those of
  `librosa.pyin` to the bit, 0.11's and 1.0's when 1.0 is asked for its whole
  search (`transition_min_prob=None`): 528 hard signals, and 1243 more over
  hops, frames, ranges, rates and lengths. Five minutes at 48 kHz take 3.5 s and
  186 MB at the peak, 92 of them there before, where they took 39 s and 1430 MB
  with librosa 0.11 on Python 3.9 (3.1 s and 196 MB against 8.2 s and 1422 MB
  with 1.0.1 on 3.13). A minute at 16 kHz is 0.82 s, and the first call has no
  compiling to wait for.
- `f0` warns about a hop over 24 ms: from there the longer the hop the more of
  the noise in a pause comes back voiced, since pYIN tells voicing by how little
  a pitch moves from a frame to the next. Of the noise in a pause, at 65 to 400
  Hz, 3.2 frames in 100 come back voiced at 24 ms, 5.6 at 25 and more than 40 at
  50.
- `f0` takes a rate that is not whole as it is; 16000.9 was read as 16000. A hop
  in which a pitch could cross the whole range, and `center=False` with a clip
  shorter than a frame, are a `FeatureError` that says so. They were
  `librosa.ParameterError`, "Target size (315) must be at least input size
  (321)". The track of an empty clip is float64 like every other; its arrays
  were float32.
- `time_stretch` and `pitch_shift` are kudio's own; what they give is under
  Changed — results that move. A clip under 512 samples can be stretched and
  shifted: with librosa 1.0 it came back all NaN from `time_stretch` and was
  refused by `pitch_shift` as "Audio buffer is not finite everywhere". Under
  2048 there is a warning, since what is worked on is mostly the zeros around
  it. float16 and big-endian floats are taken, and an empty clip comes back
  empty; `time_stretch` made 512 samples of one at 0.5. A rate of 1 and 0
  semitones are a copy. Five minutes at 48 kHz are stretched by 1.25 in 1.7 s
  and 211 MB at the peak, 92 of them there before, where it took 3.6 s and 874
  MB (librosa 0.11, Python 3.9; 1.7 s and 222 MB against 3.0 s and 1550 MB with
  1.0.1 on 3.13).
- `reverb` and `add_noise_snr` of an empty clip are an empty clip. They were
  numpy's "zero-size array to reduction operation".
- The plots of `kudio.util.visual` need matplotlib and nothing else, and show
  what they did, with these differences:
  - A waveform of more than 10000 samples is drawn as strokes from the lowest to
    the highest of every few samples, where it was a band as high above zero as
    below: an offset or a lopsided wave is seen for what it is. Zooming in comes
    to the samples, as before.
  - Every axis of time goes from 0 to the end of the clip, so the panels of a
    figure line up.
  - The curves of `plot_3_features`, `plot_4_features`, `basic_analyze`,
    `plot_zcr`, `plot_spectral_centroid` and `plot_spectral_rolloff` stand at
    the times of their frames. They were spread over the clip's whole seconds:
    of 1.3 s, over 1 s, and the axis ended there too. A clip under a second was
    shown from -0.05 to 0.05 s, with every curve on 0.
  - A mel band stands at its own frequency. `librosa.display` put the lowest of
    40 bands at 0 Hz and the highest at half the rate, where neither has
    anything.
  - Ticks of time are right to the place they are written to: 1.05 s was written
    1.1, and ten seconds before the start 0:50.
  - What is no sound is refused before a figure is made, and so leaves none
    open: NaN, inf and a file without a sample are a `FeatureError` naming the
    plot.
  - `plot_spectral_centroid`, `plot_spectral_rolloff` and `plot_zcr` take one
    channel as a list, a row, a column or integers, read on their type's full
    scale. Integers and a list were refused, `plot_zcr` apart, which took
    integers for plain numbers; a row of 8000 samples was drawn as 16 lines of
    one point, a column as one point. Two channels are refused.
- Work shared out among processes, `map_waves(workers=n)`,
  `load_waves(use_pool=True)`, `score_dataset(jobs=n)` and
  `Synthesizer.syn(is_pool=True)`, goes through a pool of kudio's own. A worker
  that dies, killed for want of memory say, is an error that names the items it
  had and its exit code, in about a second; `multiprocessing.Pool` started
  another in its place and waited for the dead one's answer for ever. What comes
  back, its order, the BLAS threads of each worker and the progress bar are as
  they were. `concurrent.futures` hung 6 times in 6 on Python 3.9 and 3.10 with
  items of 1 MB or more, so it is not that. On Windows before Python 3.13 the
  pool holds 31 processes at most, since waiting on more than 63 handles is a
  `ValueError` there and each process is two, so `map_waves(workers=70)` runs
  in 31. `multiprocessing.Pool` had the same limit: with 70 workers on Python
  3.9 a thread of its own raised that `ValueError`, and `map_waves` worked
  through its items and did not come back: 200 items took a second, and the
  call was still waiting 150 seconds later.
- A clip is read the same way everywhere: masked samples, complex numbers,
  booleans, text and rows of different lengths are refused by name, by the
  effects and edits too, which read the numbers under a mask and the real part
  of a complex one. A clip is `(frames,)` or `(frames, channels)`: a number that
  is in no array, three axes or more and frames of no channel are refused.
  `(0,)` and `(0, channels)` are a clip of none, as they were. 16-bit samples in
  an int32 or uint32 array, which read 96 dB down, are said to be with a warning
  that names the level and int16.
- A setting is a number in its range, or refused by the function it was given
  to, naming it: the thresholds and spans of `vad`, `tonal_lines`,
  `voiced_segments` and `LoudnessCurve.spans_*`, the room of `rir`, `align`'s
  bounds, the fades and joins, the filters' order and cutoffs, `random_gain`'s
  range, `spec_augment`'s masks, the companders' bits, the noise estimators'
  `noise_kwargs`, the seeds, `lsd`'s frame, `frame_windows`, `stack_context` and
  `ContextFrames`. The seed of `Synthesizer.syn` is kept to its last digit, and
  3.0 is 3, as it was. `None` is the default, and 0 is not, where the
  documentation gives 0 no meaning: `f0(hop_length=0)` and `kudio pitch --hop
  0`, `rir(seconds=0)`, `desired_length=0`, `desired_samples=0` and
  `SpectrogramStream(n_mels=False)` are refused and say "None for the default".
  The 0 of the hop of `spectral_enhance`, `compare_enhancers` and
  `StreamEnhancer`, `SpectrogramStream(n_mels=0)` and `pitch_shift(n_steps=0)`
  is kept.
- A flag is True or False, numpy's too, or a whole number that is 0 or 1;
  anything else is refused by the function it was given to, naming it. `'no'`, 2
  and NaN were True. `dnsmos`'s `polyfit` and `personalized`, `signature`'s
  `aligned`, `cut` and `resampled`, `score_dataset(align=)`, `cer` and
  `char_errors`'s `spaces`, `c_primary(minimum=)`, `der` and
  `diarization_errors`'s `skip_overlap` and `mcd(match_gain=)` take True and
  False only, as they did.
- `convert_folder` checks its `subtype`, `trim_db`, `peak`, `lufs` and flags,
  and `enhance_folder` its `subtype`, frame, hop and flags, before the first
  file: each failed once a file, and the output folder was made for them.
- `copy_waves` raises when the folder to copy to is a file or ends in `.wav`,
  the two arguments the other way round; a folder with a dot in its name is a
  folder.
- A rate that is not whole is kept: `rt60`, `find_delay`, `audio_report`,
  `loudness_over_time`, `SpectrogramStream` and `StreamEnhancer` gave 16000.9
  back as 16000.
- `wave_separate` gives the columns of a `(frames, channels)` array, and refuses
  another number of channels than it holds, or interleaved samples that are no
  whole number of frames. It gave every second frame, and
  `AudioDataManager.add_new_file(ch=1)` kept a stereo array as one channel.
- `features2matrix` refuses anything but matrices of one width: two vectors of 5
  became a matrix of 2 rows with 10 labels. `wave_slicing` takes one channel,
  `(n, 1)` among them, and refuses two, a clip shorter than a frame and a bad
  size by name.
- `mfcc_from_files` refuses a negative length, which cut samples off the end of
  each file, files that give different numbers of frames, by name, and no files
  at all. A negative `desired_length` of `waveform_to_spectrogram` is refused;
  it cut samples off the end. `STFT.synthesise(length=0)` gives no samples,
  where 0 was read as no length given, and a negative or fractional length and a
  spectrum with NaN in it are refused.

### Changed — input that is refused

What these took and what it came to. Those that raised before and only say it in
kudio's own words now are not listed.

- `hop_length=None` of `waveform_to_spectrogram`, `spectrogram_to_waveform` and
  the functions on files was a quarter of the window, 128 of 512, where the
  default is 256; of `melspectrogram` a quarter of the frame, where the default
  is 512. `win_length=0` was a window of nothing and a spectrogram of silence,
  -31.3 throughout.
- `n_mels=0` and `n_frames=0` of `melspectrogram` gave an empty array; `power`
  of NaN or below 0 and `fmax` of NaN, 0 or below 0 gave numbers of no meaning;
  `SpectrogramStream` took '64' for 64, 2.5 for 2 and NaN for a range.
- Booleans and complex samples went into most of these as numbers, the imaginary
  part dropped.
- `time_stretch(rate=True)` was 1, and inf no samples at all;
  `pitch_shift(n_steps='2')` was 2 and True 1. `n_steps` goes to 120, ten
  octaves, either way.
- NaN, inf, True and text for the settings of `gain`, `normalize`,
  `normalize_db`, `add_noise_snr` and `reverb`. NaN made an int16 clip all zeros
  and a float one all NaN. A `delay_ms` of 0 or less was an echo a sample later.
- One NaN or inf in a clip to `normalize`, `normalize_db`, `reverb` or
  `add_noise_snr` spread over all of it: with an inf in it the rest of a clip
  came back normalised to 0. `gain` keeps it to its sample, and refuses to make
  an integer of it.
- `f0`: a `hop_length` of 2.5 was 2, of True 1 and of '160' 160; a
  `frame_length` of 2048.5 was 2048; `fmin=True` was 1 Hz; `hop_length=0` was 10
  ms.
- `vad`: `threshold_db`, `flatness_max` or `min_speech` of NaN (no speech, a
  `speech_ratio` of 0.0, and "no speech detected" from `kudio vad --threshold-db
  nan`), `min_silence='1'` (taken), `pad=-0.1` (0), `frame=-1` (a frame of 16
  samples) and `hop=-1` (a hop of one sample); `vad_frames(hop_length=2.5)`
  (frames that line up with nothing).
- `tonal_lines`: `min_prominence_db` of NaN, a `resolution_hz` the rate cannot
  give, 1000 at 16 kHz, and so too few samples for a line (no lines, as of a
  clip that has none), and `max_lines=-1` (all but one). A clip of 60 samples
  still answers with no lines, and says it is too short.
- `PitchTrack.voiced_segments`, `LoudnessCurve.spans_below` and `spans_above`
  with a NaN (`[]`).
- `rir(drr_db=nan)` (all NaN) and `rir(seconds=0)` (the default length).
- One NaN or inf: to `apply_rir`, in the clip or the room (all NaN); to `rt60`
  (a DRR of +inf); to `schroeder_curve` (NaN up to it); to `find_delay` ("leads
  by 15999 samples", correlation 0); to `remove_dc`, the filters and
  `wavelet_low_pass_filter` (all NaN); to `trad_enhance` (NaN from it on); to
  `mu_law` (inf was 1.0), `a_law` (NaN) and `bit_depth` (-inf was -1.0); to
  `spectrogram_to_waveform`'s spectrogram (NaN at 16640 of 32000 samples) and
  `denoise_spec2wav` (all NaN). A log power of -inf is no power and is taken.
- Complex numbers, whose real part was used: `rt60`, `find_delay`, `loudness`,
  `true_peak`, `mu_law`, `spectral_enhance`, `save_wave` (written),
  `estimate_noise_psd` (taken for power), `spectrogram_to_waveform`,
  `stack_context`, `frame_windows`, `Standardizer.fit` and `spec_augment`
  (masked with the real part of their mean). `find_delay(None, y)` was an
  answer.
- Masked arrays, whose samples under the mask were used: `loudness`,
  `audio_report` (the peak under the mask), and the effects and edits,
  `highpass`, `gain`, `trim_silence` and the rest.
- A number that is in no array: `gain`, `random_gain`, `normalize` and
  `normalize_db` (a number back), `mu_law`, `a_law`, `bit_depth`, `dropouts` and
  `apply_channel` (one sample), `vad` and `vad_split` (`[]`), `vad_frames`,
  `find_delay` (no delay, correlation 0), `tonal_lines`, `audio_report` and
  `SpectrogramStream.push` (no columns).
- Three axes: `gain`, `random_gain`, `normalize` and `normalize_db` (the array,
  scaled), `reverse` and `remove_dc` (along the first axis). Frames of no
  channel: `gain`, `random_gain`, `normalize`, `normalize_db`, the fades, the
  filters, `reverse`, `remove_dc` and `resample` (the empty array back),
  `audio_info` (0 channels), `speech_ratio` (0.0) and `true_peak` (-inf).
- `true_peak(oversample=2.9)` (2), `lsd(n_fft=512.5)` (512),
  `highpass(cutoff='80')` (80) and `normalize_lufs(lufs=nan)` (all NaN).
- `channel_spec`: `band='no'` and `band=(300, 3400)` (both the telephone band);
  `band` is True, False, None, 0 or 1.
- `spec_augment(n_freq_masks=-1)` (no masks); `random_gain(db_range=(1.0,))` (+1
  dB every time).
- `noise_kwargs`: `bias=-1` of `'minimum'` (the clip back as it was), `ad=5` and
  `ass=nan` of `'mcra'`, `window=-1` and `n_frames=-3` (answers).
- `desired_length=0` (all of it) and `SpectrogramStream(n_mels=False)` (a linear
  axis).
- `stack_context(context=True)` and
  `waveform_to_spectrogram(forward_backward=True)` (a frame each side);
  `forward_backward=False` is no context, as it was.
- Objects, text and bytes to `stack_context`, `ContextFrames`, `frame_windows`
  and `Standardizer` (the numbers they hold, or what the text read as).
- `estimate_noise_psd` of power below 0 (every bin 1e-12), a spectrogram in dB
  among them (the noise estimate of its numbers).
- `audio_info` of a 3-D array (4 channels of 4 frames).
- `Synthesizer.syn`: `seed=1.5` and `seed='1'` (1), `p_silence` of 2 or NaN (no
  output was noise alone) and `rdn_choice_num=-3` (as not given).
- A flag that is neither True nor False, nor 0 or 1: `'no'`, `'False'`, 2, -1,
  NaN, 1.0, None and `[True]` were each True, or False for None, wherever one
  was taken: `highpass(zero_phase='no')`, `f0(center=2)`, `apply_rir(trim=nan)`,
  `audio_info(peak='no')`, `lsd(match_gain='no')`, `srmr(fast='no')` and 45
  more.
- `convert_folder` with a `subtype` a WAV file does not hold, a `trim_db` or
  `peak` that is no number above 0, or a `lufs` that is no number, and
  `enhance_folder` with such a `subtype` (one failure a file, and the folder
  made).

### Added

- The extra `compressed`, in `all` too: audioread, for what soundfile does not
  read. ffmpeg is installed by itself.
- `mel_band_hz(n_mels, sr, *, fmin=0.0, fmax=None)`: the frequency each mel band
  of `melspectrogram`, `mfcc` or `SpectrogramStream` is centred on, to label an
  axis with. The feet of the lowest and the highest band are not among them;
  `librosa.mel_frequencies` gives those too, and is asked for `n_mels + 2`.
- tools/validation/against_librosa.py, which compares kudio with librosa
  wherever kudio follows it, and prints a table.

### Fixed

- `dropouts`, `apply_channel` and `Synthesizer(channel=...)` with a `burst` of
  1e19 packets or more never came back on numpy 1.24, which draws the lowest
  integer there is for a number no integer holds. Later numpy draws the highest.
- `Standardizer.transform` and `inverse` of frames of another width are refused
  (one column was spread over 257, a number came back as 257); `fit` of no rows,
  of NaN or inf and of complex numbers is refused (fitted, with means of NaN or
  of the real part).
- `wave_encode` of float audio at full scale 1 (all zeros), of NaN (0), of
  floats past 2 and of 64-bit integers past 16 bits (wrapped round). 16-bit
  numbers in a 32- or 64-bit integer array are their own bytes, and what is
  between 1 and 2 is held at full scale with a warning.
- `Synthesizer.syn(is_silence=True, p_silence=1.0)` writes every output as noise
  alone (none); 0.99 wrote all of them.
- `KudioConfig.load_config` of an empty file is the defaults, with a warning
  (`{}`).
- `denoise_spec2wav` of a clip of one frame ((257,) against (257, 1), 257 frames
  of sound) and of a spectrogram and phase that do not go together (numpy's
  broadcast error).
- `kudio room --rt60 inf` is one line and exit 2 (a traceback and exit 1);
  `kudio align --max-seconds inf` is no bound (a traceback); `kudio compare
  --reference` of a file of one sample is the length rule's one line and exit 2
  (a `TypeError` and its traceback).
- `trim_silence` of samples whose square float32 does not hold, 1e30, says so
  (all of it silence, an empty clip).
- `compare_enhancers` of one sample (a `TypeError`) and of `noises=[]` (an
  `IndexError`).
- `stack_context` of no frames is `(0, dim * (2 * context + 1))`, as
  `ContextFrames` gives (numpy's error).
- `audio_info` takes a list of samples (a `TypeError` of pathlib's).
- `plot_PSD` and `plot_power_spectral_density` of NaN or of no samples are a
  `FeatureError` (empty axes); `WaveVisualizer(subplots=None)` makes as many
  rows as there are files (no axes, an `AttributeError`), and more files than
  rows are refused.
- `AudioDataManager.set_params` takes 1 or 2 channels (6 was 1, and a 6-channel
  file was saved as one channel six times as long).

## [4.3.0] - 2026-11-25

### Changed

- `FeatureError` is a `ValueError` as well as a `KudioError`. `vad` refuses a
  bad rate with a `FeatureError` and `lsd` with a plain `ValueError`, and one
  `except ValueError` catches both. `except FeatureError` and
  `except KudioError` catch what they did.
- A sample rate is one positive number wherever a function takes one, and a
  file's is a whole number. 16000.0, `np.int32(16000)` and a 0-d array are the
  int they equal; `True`, text, NaN, inf, 0 and negative numbers are refused
  with a `ValueError` that names the function or the argument. An array can be
  at 22254.5 Hz, so `resample` and whatever takes a rate beside an array take
  that too. `save_wave` raised `AudioIOError` for a bad `sr`, and so did
  `file_load` for NaN and inf, as if the file were bad, and `TypeError` for
  text. `convert_folder` and `enhance_folder` made *dst* and went through the
  files one by one; they stop before the first.
- A 2-D array is `(frames, channels)`. With two rows or more and more columns
  than rows it is channels first, and a `ValueError` says to transpose it:
  `remove_dc` gave half the difference of the channels, `reverse` swapped them,
  `trim_silence` and `resample` cut across them, and `audio_info` counted 16000
  channels. `(1, n)` is one channel everywhere, where `remove_dc` gave zeros,
  `speech_ratio` 1.0 and `save_wave` a file of n channels, and so is `(n, 1)`
  in a function that takes one channel, some of which refused it. A file says
  how many channels it has, and `convert_folder` and the command line keep them
  however few frames it holds.
- 64-bit integers are refused by whatever measures, changes or writes samples,
  as the metrics refuse them. A list of ints or `astype(int)` gives them, and
  16-bit samples among them came back as zeros from `mu_law`, as -inf from
  `loudness` and as no speech from `vad`, and `save_wave` wrote a silent file.
- `save_wave` refuses an array that is neither `(frames,)` nor `(frames,
  channels)` with 1 to 1024 channels with a `ValueError`; it was an
  `AudioIOError`.
- `save_wave` writes beside the file and replaces it with `os.replace` when the
  new one is whole. Windows does not replace a file that another program has
  open; after 0.3 s of trying, the finished one is copied over it if it opens
  for writing, and is an `AudioIOError` if it does not. Only then can a disk
  error damage the file. A link stays one, the permissions of the file
  are kept, and a file that may not be written is not replaced. MPC2K, SD2 and
  SVX, which libsndfile writes the file's name into or as two files, are written
  in place.
- Reading does not go through `librosa.load`. A file at the rate asked for,
  and `Synthesizer` without resampling, need no librosa. Of 8883 calls on 249
  files soundfile reads, each gives the same bits as with librosa 0.11. With
  librosa 1.0 ten differ, in the sign of a zero: its mono mix turned -0.0 into
  0.0, which the samples keep.
- A file soundfile recognises and cannot read is not handed to ffmpeg. With
  librosa 0.11, `file_load(sr=...)` and `Synthesizer` tried audioread after
  any error of soundfile's, and a FLAC file cut to half came back as 8192 of
  its 17647 frames without a word. It is an `AudioIOError` with soundfile's
  reason, as it is with `sr=None`. What ffmpeg does read is a format soundfile
  does not recognise, or an encoding it does not support, like ALAC in a CAF
  file.
- `crossfade` and `splice` give a result of the first part's type and read each
  part on its own scale. A float clip after an int16 one came out as zeros, and
  an int16 one after float 32768 times too loud.
- `trim_silence` and `split_on_silence` refuse a `top_db` that is not a finite
  number above 0. At 0, below it and at NaN the loudest frame is silence too,
  and they returned an empty array and an empty list.
- `vad_trim` and `vad_split` return copies, as `trim_silence` and
  `split_on_silence` do. Writing into a cut wrote into the clip it came from.
- `normalize_lufs` and `match_loudness` warn when a clip that is not silent
  measures -inf, which is any clip with no block above the -70 LUFS gate, and
  hand it back as it was. `kudio normalize` of such a file is an error, as it
  is for digital silence: it wrote the file unchanged and said "normalised to
  -23.0 LUFS". `kudio loudness` gives its peak where it said "the file is
  digital silence".
- `kudio trim` and `kudio convert` exit 2 when nothing is left to write, as
  `convert_folder` counts it a failure. `kudio trim take.wav take.wav --top-db
  0` put an empty file over the recording and exited 0, and a file with no
  samples was written out again.
- `mfcc` and `mfcc_from_files` refuse an `n_mfcc` above the number of mel bands.
  `mfcc(y, sr, n_mels=32)` with the default 40 reshaped 25 frames of 32 into 20
  of 40, and any number of frames not a multiple of 5 was a reshape error.
- `spectral_enhance` and `StreamEnhancer` refuse a `hop_length` at which the
  overlapping windows fall under 9% of their peak: 18.1 came out of 0.404 at
  512. `STFT.synthesise` refuses one that leaves samples under no window, and
  anything but a complex `(frames, bins)`: a librosa-shaped spectrum came out
  as a burst of noise and a magnitude as a waveform with every phase zero.
- `STFT` refuses an `n_fft`, `hop_length` or `win_length` that is not a whole
  number of 1 or more and keeps numpy's as Python ints. `matches(16000.9)` cut
  its argument to 16000 and was True for it.
- `channel_spec`, `dropouts`, `mu_law`, `a_law` and `bit_depth` refuse bits that
  are not a whole number from 2 to 64, and a loss, `burst` or `packet_ms` that
  is not one finite number. 8.9 bits were 8, 8.5 were 361 levels, a `burst` of
  NaN lost packets one at a time and a loss of `True` lost them all.
- `Synthesizer` takes `rt60` and `drr_db` that are finite, and `rt60` that is
  not `True`, and raises `ValueError` when it is built with anything else.
  `background_snr` of `syn_extra_mode` has to be finite as well. It and
  `snr_ratio` lie between -300 and 3000 dB: at 3083 the gain was an
  `OverflowError`, and at -400 the noise overflowed in a loud file and not in a
  quiet one, where it was 10^20 times the speech.
- `load_labels` names the row of a JSON label file that is not an object, has
  no `start` or `end`, or has one that is not a number, with a `ValueError`.
  The first two were an `AttributeError` and a `TypeError` about `__init__`; a
  time held as text or null was read as it stood. A time that is not finite
  and an end before its start are refused by it and by `save_labels`, and by
  `load_rttm` with a negative duration. What is not JSON or not UTF-8 is named
  with its file, in `load_manifest` too.
- `AudioDataManager.add_new_file` raises `ValueError` for a `wave` that is not
  a list of one waveform, where it logged a warning behind kudio's
  `NullHandler` and returned.

### Added

- m4a, mp4, aac, webm, 3gp, amr, wma and whatever else ffmpeg decodes, wherever
  kudio reads a file and with or without `sr`: `file_load`, `audio_info`,
  `play_audio`, `Synthesizer`, `save_spectrogram_as_wave`, `plot_stft_w_2mfe`,
  the command line and the paths of a `score_dataset` list. It takes audioread
  (`pip install audioread`) and ffmpeg on the PATH; neither comes with kudio.
  `file_load(sr=...)` and `Synthesizer` read these through librosa 0.11, and
  nothing else did. librosa 1.0, which a new install gets on Python 3.12 and
  later, has no audioread and reads none of them, ffmpeg or not. Where
  librosa 0.11 read a file, what comes back is the same to the bit: 475 calls
  on 25 files with ten extensions, whole and cut short.
- Two warnings about a file ffmpeg read. One that gives less than its header
  promises: ffmpeg decodes as far as a file goes without an error, and half of
  an m4a made for streaming, or of a webm, came back as half the clip.
  Containers that store their length (MP4, Matroska) are held to it; a bare AAC
  or AMR stream has none. And one whose name ends in a format soundfile reads
  and which soundfile does not recognise, a FLAC that lost its beginning or an
  m4a called `.mp3`. Of the last libsndfile says that it does not exist.
- A warning about a WAV that ends before its header says, with the two lengths:
  half of a 6 s take was read as a take of 3 s. `file_load`, `audio_info`,
  `peak=False` included, and whatever else reads a file give it, and return
  what they returned. It is judged by the header and the size of the file. A
  length of 0 or of 0x7FFFF000 and up, which a program that streams its output
  leaves there, is not held to, and neither is RF64, W64, AIFF or any other
  container. 876 whole files in every subtype of WAV are not warned about.

### Fixed

- `save_wave` emptied the file it could not write: libsndfile opens it first,
  which truncates it, and refuses then. Nine channels of FLAC, three of MP3,
  FLAC at 1 MHz and an array of three dimensions each left 0 or 44 bytes of the
  recording that was there, as a full disk or Ctrl-C left half of the new one.
- `Synthesizer.syn` emptied the output folder and then found that a setting
  cannot work: `rt60=nan`, `drr_db=-99`, a telephone band under a rate of
  500 Hz, a negative seed, a noise to resample without librosa. The corpus was
  gone and nothing was written. Every room at the rate of its file, the link,
  the rates, the seed and librosa, if anything has to be resampled, are tried
  first, and a run that is refused leaves the folder and `seed`, `target_sr`,
  `rooms` and `manifest()` as they were. A run that is not refused writes the
  same files. `syn_extra_mode` looks at `background_snr`, the seed and librosa
  first.
- `Synthesizer` emptied its output folder for a run that the headers of its
  inputs, the names of its outputs or the folder itself show cannot be made. Of
  4000 random calls, most of them with a setting or a file that is odd or
  wrong, 1247 raised with the folder gone; 41 do, each for what only the
  samples show.
- A link that loses more than half of its packets in runs is tried for every
  output of a `Synthesizer`, with the output's own length and seed.
  `{'loss': 0.9, 'burst': 1}` failed on the first file, with the folder empty.
  The `FeatureError` names the output.
- `Synthesizer` reads the header of every input before it empties the folder.
  One that cannot be read, that has no frames or a single one at the rate it is
  mixed at is the `SynthesisError` it was. `syn` and `syn_extra_mode` also look
  again at the SNRs, the subtype and the lists of files, which can be changed
  after the `Synthesizer` was built, and at a room that does not fit in memory,
  and start the process pool. A warning about a header that the caller's
  filters make an error of is raised by then.
- `Synthesizer` writes a sample at every rate, and one under the longest names,
  in a folder beside the output one before it empties that. A name of more than
  255 letters, a subtype or a rate that libsndfile does not write, and a name
  that would be a file and a folder were found by the first file they stopped.
- On Windows a read-only file in the output folder of a `Synthesizer`, or one
  that another program had open, stopped the emptying half-way. It is looked
  through first. The guard against deleting the inputs follows links and
  junctions, through which they could be in the folder unseen, and a link
  itself is not emptied.
- `syn_extra_mode` mixes the noises with the backgrounds before it empties the
  folder, so whatever is wrong with one of those leaves it as it is.
- A file with NaN or inf in it could be read with `sr=None` alone; any `sr`,
  the file's own included, raised "Audio buffer is not finite everywhere". At
  its own rate it is read, with the warning. At another it is an
  `AudioIOError` that counts the samples and says to read it with `sr=None`,
  since a resampler spreads each one over its neighbours. `Synthesizer` and
  `save_spectrogram_as_wave` count them too.
- `True` was a rate of 1 Hz: `rt60` read 6424 s off a room of 0.4 s,
  `file_load(sr=True)` gave 2 samples for 2 s of audio, and `save_wave(sr=True)`
  wrote a file at 1 Hz. `file_load(sr=16000.5)` resampled to that and
  reported 16000, and `Synthesizer.syn(target_sr=8000.5)` wrote files that say
  8000. A rate of 0 was "above Nyquist" to `lowpass`, which handed the clip
  back, a duration of 0 to `audio_info` and "not given" to
  `match_loudness(reference_sr=0)`; `audio_info(y, sr=-1)` on 32000 samples gave
  -32000 s. `resample` with a source rate of inf did not return.
  `save_wave(sr=16000.0)` failed with "an integer is required".
- `file_load`, `audio_info` and the other readers say when a path does not
  exist, when it is a folder and when the file is empty, and for a file that
  will not open what the operating system says. libsndfile says
  `System error.` for all of them but the empty one, and with librosa 0.11
  two warnings about audioread came first.
- With librosa 0.11, an m4a read with `sr=...` ended in `()` when there was no
  ffmpeg, when ffmpeg could not decode the file, and when the file was empty.
  A missing audioread, a missing ffmpeg and a file that cannot be decoded are
  told apart, and any other error without text is named by its type.
- `play_audio` said it plays what `file_load` reads and played what soundfile
  reads. It reads the file the same way.
- `vad`, `vad_split`, `vad_trim`, `vad_frames` and `speech_ratio` judge a clip
  of any length. From 0.04 to 0.43 s, at every rate, they raised "operands
  could not be broadcast together", since `np.convolve` in its `same` mode
  answers with the longer of the clip and the 41 frames a fricative reaches.
  With `band=`, a clip under 30 ms raised an `IndexError`, as `vad_frames` did
  for one sample.
- `spectral_enhance` with `spectral_gate` takes every clip of one frame or
  more and the whole range of its smoothing. By default 512 to 639 samples
  raised the same error, and so did `time_smooth_ms=500` on 0.48 s and
  `freq_smooth_hz=4000` at 6 kHz.
- With `hop_length` near half of `n_fft`, `spectral_enhance` divided the last
  samples of a clip by next to nothing: with a Hann window and a hop of 256, a
  clip that ends 255 samples past the last frame's centre came out at 109 times
  its own peak. The samples under 3% of the last frame's window are faded. With
  the default window there are none at a quarter or a third of `n_fft`, and
  every other sample is the same to the bit.
- The fades, `crossfade`, `splice`, `remove_dc`, the filters, `resample` and the
  reference of `spectrogram_to_waveform` read uint8 and uint16 around their
  midpoint, as everything else did. A fade took uint8 silence, 128, to 0, which
  is full scale, and resampled it swung by up to a quarter of that at both ends.
  Signed integers and floats come out the same to the bit.
- `resample` holds an int32 to its range. The largest one is not a float32, and
  a peak clipped there came back as full scale of the other sign.
- `wave_encode` refuses an array that is channels first, which it wrote out one
  channel after the other, and `wave_separate` splits a row `(1, n)`, whose
  second channel came back empty.
- `save_manifest`, `save_labels` and `KudioConfig.write_config` finish the text
  before they open the file. A value JSON or YAML has no form for left a
  manifest cut off partway and a config of 0 bytes where a good one was. It is
  a `ValueError`, which for a manifest or labels names the row, and the file
  stays as it was.
- The same three take numpy numbers and `Path`s: an `np.int64` SNR, an
  `np.float32` time, a path out of `glob`. A path is written with forward
  slashes.
- `KudioConfig.to_dict` reaches dicts inside lists inside lists, which
  `write_config` could not write.
- `load_rttm`, `load_labels` and `load_manifest` read a file that starts with
  a byte-order mark. `load_rttm` took the first SPEAKER line for one of
  another type and skipped it, so a perfect system scored a DER of 1.0 on a
  reference of two turns; the other two refused the file.
- A carriage return in a label's text ends up a space in an Audacity track, as
  a newline does. Written as it was, it split the row, and `load_labels` could
  not read the file back.
- `AudioDataManager.add_new_file(override=True)` builds the new clip before it
  gives up the old one. A bare array passed for `wave` removed the clip it was
  to replace.
- `AudioDataManager.add_local_file` reads the file before it adds the clip. One
  that could not be read left an empty clip under its name.

## [4.2.0] - 2026-11-20

### Changed

- `audio_report` decides `band_limited` by an edge, not by where the spectrum
  falls 50 dB under its peak. A recording made at its own rate keeps its
  converter's floor going above the content however low the content stops,
  and is not band-limited: a 192 kHz measurement with content to 50 kHz was
  told its extra rate was empty, and so was a pure tone or a clean take with
  a quiet top octave. An edge is a band at least 20 dB under the one just
  below it and 40 dB under it to Nyquist, looked for at the Nyquist
  frequencies of ten common rates from 8 to 96 kHz and anywhere from 0.1 to
  0.9 of Nyquist. Linear interpolation, which leaves no edge, is found by the
  mirror image it leaves above the old Nyquist frequency.
  tools/validation/band_limit.py has the cases it was tried on: eight
  resamplers and 720 upsampled files, of which 468 of the 480 at 0 and
  -20 dB are flagged (the 12 left are linear interpolation by a ratio that
  is not a whole number) and none is named after a rate it was not made
  from, and 470 recordings at their own rate, none flagged. `bandwidth_hz`
  is the same number as before.
- The band-limit finding compares the content with half the rate ("a 16000 Hz
  file holds up to 8000 Hz"), where it said the file claims the rate; gives
  where the edge test finds the content ending, `band_edge_hz`, where it
  quoted `bandwidth_hz`, which a resampler's stopband carried to 27750 Hz on
  white noise taken from 48 to 192 kHz and called 48 kHz; names
  one source rate, two neighbours 9% apart when the edge fits both, or none
  when it sits where no resampler would leave it; and says a codec cutting
  there looks the same. `telephone` output is band-limited without a rate:
  its 3400 Hz filter ends the content at 0.94 of 4 kHz.
- An SNR estimate under 10 dB says that a steady sound, hum or a running
  machine, reads the same as noise close to the content, since its loud and
  quiet frames hardly differ; calling it noise was a guess.

### Added

- `tonal_lines`: the peaks of the averaged spectrum that stand 20 dB or more
  above the running median around them, each with its frequency, level and
  prominence, and the combs they make on whole multiples of one frequency,
  found from the spacing as well, so hum without its fundamental is still a
  50 Hz comb. It is not one of the report's findings, because a voice or a
  note held at one pitch is a comb too.
- `AudioReport.source_rates`, `mirrored`, `band_edge_hz`,
  `full_scale_samples`, `sample_step` and `grid_bits`, after the other
  fields and with defaults, so an `AudioReport` built by keyword keeps
  working.
  `full_scale_samples` counts every sample at 0.999 of full scale or more, in
  a run or not. `sample_step` and `grid_bits` are the grid every sample
  sits on, read at 24 bits; a grid coarser than 16 bits is a finding.
  `grid_bits` is not ENOB, which is read from a sine's SINAD.
- `audio_report(..., clip_run=3)`, the shortest flat run that counts as
  clipping, for comparing with tools that count two samples in a row, or one.
- `vad(..., band=None)`: `(low, high)` judges the frames band-passed to it,
  the flatness over that band alone. In a 192 kHz file a floor above 20 kHz
  can outweigh a quiet voice on the whole band; `band=(100, 4000)` finds it.
- `mfcc(..., n_fft=None, hop_length=None, n_mels=None, fmax=None)`, passed to
  librosa's mel spectrogram and left to librosa's own defaults when not
  given. At 192 kHz those leave 3 of 128 bands empty; `n_fft=8192` leaves
  none.

### Fixed

- A clipped plateau with a sample just under full scale at its edge, 0.9993
  before five samples at 1.0, was judged together with that sample, found
  not flat, and not counted at all. The plateau is counted on its own, when
  it holds at least twice the share of its run above 0.999 that a smooth peak
  of the same height keeps within 1e-4 of its top, so a sine normalised to
  full scale stays unclipped at any frequency. The take on
  docs/guide/inspection.md keeps its 3321; `hot.wav` on docs/cli.md counts
  3390 where it counted 3342.

## [4.1.0] - 2026-11-19

### Fixed

- `llr`, and with it CSIG, CBAK and COVL, scored nan when the reference has
  no noise in it at all, a harmonic source alone for instance. Order-16 LPC
  predicts some of its frames to round-off, the clean frame's own prediction
  error came out below zero (-7.1e-10 against a frame energy of 0.0277 on the
  signal in the tests), and its log is NaN; the warning blamed a frame too
  loud to square. Both quadratic forms are floored at eps times the frame's
  energy, so such a frame scores the cap, as it does in pysepm. On 22 pairs
  of the test suite and 4 others that scored before, `llr` and the
  composite's LLR are unchanged to the bit.
- `kudio vad` and `kudio pitch --segments` ended with exit 2 when their
  output went to a file or a pipe under a code page without "→", cp1252 on
  an English Windows. Spans are listed as `0.550 -    2.460`, and any
  character the code page lacks comes out as `?` instead of ending the
  command. The same goes for `--help`, whose em dashes cp932 lacks.
- `CheckDevice.system_devices()` listed host API 0 alone. On a Windows 11
  machine with no default recording device that held no input at all, while
  `input_device()` found the microphones on WDM-KS. It lists every host API,
  naming each in a new `api` key, and logs each device at DEBUG rather than
  INFO. `kudio devices` prints the host API and keeps each device on one
  line; Windows names a Bluetooth hands-free endpoint with a line break in
  it.
- `save_dir` of the plots in `kudio.util.visual`, and `fig_dir` of
  `hist_plt`, were used as a file name: `save_dir="figs"` wrote `figs.png`,
  and the next plot replaced it. It is a folder, made if missing, that gets
  `<stem>_<kind>.png`; a name ending in a format matplotlib writes is still
  the file. `Common(fig_dir=..., is_save=True)` made its folder and then
  saved a bare name from `save()` in the working folder; it saves it in
  `fig_dir`.
- `score_dataset` given a path that does not exist says so, and for a
  relative one where it looked.
- `save_wave` to a `.aif` path raised `AudioIOError`: soundfile reads the
  container from `.aiff` and not from `.aif`. Both write AIFF.
- `PitchTrack.voiced_segments` and `to_labels`, and with them
  `kudio pitch --segments` and `--labels`, ended a run voiced to the last
  frame a hop past the end of the signal: 3.010 s on a 3 s sweep at a 10 ms
  hop. They end it with the signal.
- `save_labels` wrote JSON times as they came out of the arithmetic,
  `3.1399999999999997` for 4.02 - 0.88. They are written to the microsecond,
  as the Audacity format already was.

### Added

- `kudio --version`.
- `cite()` gives Zenodo's concept DOI, 10.5281/zenodo.24219274.
- `PitchTrack.length`, the tracked signal's length in samples; `None` for a
  track built without it.

### Changed

- Every `pip install` of an extra, in the README, the docs and the messages
  naming one, quotes it: `pip install "kudio[audio]"`. Unquoted, zsh reads
  the brackets as a file pattern.

## [4.0.0] - 2026-11-14

Metrics for enhancement, intelligibility, separation, recognition, speaker
verification, diarization and pitch tracking, each compared with the
implementation its published numbers come from or with its definition worked
out a second way; confidence intervals and tests over a test set, a signature
that says how a result was computed, BibTeX for every metric, and
`kudio score`. The metrics 3.9.1 had give the same numbers on the pairs they
could score, except PESQ on a pair of two lengths, and `score` with no list
runs the same eight.

### Added

- `llr`, `itakura_saito`, `cepstral_distance`, `wss`, `fw_segmental_snr` and
  `composite` (CSIG, CBAK, COVL as a `CompositeScore`): Hu and Loizou's
  measures as Loizou's MATLAB code computes them, at 8 and 16 kHz. The
  composite's PESQ is wideband MOS-LQO at 16 kHz and raw narrowband P.862 at
  8 kHz by default; `mode="nb", scale="raw"` is the variant the regression
  was fitted on.
- `lsd`, the log-spectral distance, with URGENT 2025's settings by default
  and NVSR's `ssr_eval` settings as arguments.
- `mcd`, the mel-cepstral distortion as URGENT 2025 computes it, with SPTK's
  mel-cepstral analysis and FastDTW restated in numpy, so it needs no
  compiler or extra package.
- `ncm` and `csii` (three levels in a `CsiiScore`): the normalized covariance
  metric and the coherence SII, as Loizou's NCM.m and CSII.m compute them.
- `siib` and `siib_gauss`: speech intelligibility in bits per second, as the
  authors' SIIB.m and SIIB_Gauss.m compute it, in numpy and scipy. A pair
  with less than 5.45 s of speech scores nan, and the noise that breaks ties
  in SIIB's estimator has a fixed seed, so a pair scores the same every time.
  At 16 kHz they give pySIIB's numbers, SIIB^Gauss to under 1e-12 of itself
  and SIIB to 1.4e-5 at most.
- `srmr` and `wada_snr`: reverberation and SNR estimated from the recording
  alone.
- `si_sdr_improvement` and `sdr_improvement`, SI-SDRi and SDRi over the
  mixture. `pit_si_sdr` (a `PitScore`) matches several estimates to their
  sources by the highest mean SI-SDR and gives each source's SI-SDR and
  SI-SDRi; `bss_eval` (a `BssEval`) gives mir_eval's SDR, SIR, SAR and SDRi,
  with its own permutation or the one `pit_si_sdr` found.
- `wer`, `cer`, `word_errors` and `char_errors` (an `ErrorCounts`): error
  rates pooled over a corpus, with the hits, substitutions, deletions and
  insertions behind them. `cer` leaves whitespace out unless `spaces=True`.
- `eer`, `min_dcf` and `det_curve` (a `DetCurve`) for speaker verification.
  The EER interpolates between operating points as the VoxSRC scorer does,
  or takes BOSARIS's convex hull with `method="rocch"`; `min_dcf` has no
  default `p_target`.
- `cllr`, `min_cllr`, `act_dcf` and `c_primary` for scores that are
  log-likelihood ratios: Brümmer and du Preez's C_llr and its minimum after
  the best calibration, the actual DCF at the Bayes threshold, and NIST SRE
  2016's primary cost. min C_llr keeps tied scores together, as PYLLR does.
- `der`, `diarization_errors` (a `DiarizationErrors`), `jer`,
  `detection_error_rate` and `load_rttm`: DER as md-eval.pl scores it, the
  collar per side and the speakers mapped before the collars and the overlap
  are taken out, and JER with dscore's pairing of the speakers.
- `dnsmos_p808`, the score of DNS-Challenge's P.808 model. Its log mel
  spectrogram is by default the one `dnsmos_local.py` computes under the
  librosa its requirements.txt pins, 0.8.1; `librosa="0.9"` gives the one
  librosa 0.9 and later compute, as the script under a current librosa and
  torchmetrics do. The two scores are up to 0.19 apart.
  `dnsmos(personalized=True)` runs the personalised P.835 model with its own
  polynomials, `POLYFIT_PERSONALIZED`, and `find_dnsmos_model(model=)` finds
  any of the three models.
- `gross_pitch_error`, `voicing_decision_error`, `f0_frame_error`,
  `fine_pitch_error`, `f0_rmse` and `f0_errors` (an `F0Errors`), for two F0
  tracks in Hz or two `PitchTrack`s.
- `score` takes `llr`, `is`, `cep`, `wss`, `fwsegsnr`, `lsd`, `csig`,
  `cbak`, `covl`, `ncm`, `csii_high`, `csii_mid`, `csii_low`, `siib` and
  `siib_gauss` by name.
- `tools/validation/`: one script per group of metrics that runs the
  comparison with the reference implementations again.
- `cite` and `metric_info`, with `METRIC_INFO` and `BIBTEX` behind them:
  every metric kudio computes, under `score`'s name and its function's, with
  which way is better, its range and unit, the settings that change its
  number, its section on the validation page and the works it comes from.
  `kudio.cite(["pesq", "stoi"])` gives the BibTeX for kudio and for those
  works, each once. `CITATION.cff` for the repository, and `.zenodo.json`
  with the same description for Zenodo's archive of each release.
- `bootstrap_ci`, `paired_ci`, `rate_ci` and `paired_rate_ci` (a
  `ConfidenceInterval`): bootstrap intervals, BCa by default, for the mean of
  a per-utterance score, the mean difference between two systems on the same
  utterances, a WER, CER or DER pooled over a corpus, and the difference of
  two such rates. `permutation_test` (a `PermutationResult`) is the paired
  sign-flip test of the same differences, exact when every sign pattern can
  be run, and `holm` adjusts several p-values together. With `groups`, whole
  speakers are resampled and flipped together: on simulated speakers, 95%
  intervals then held the true mean 92% of the time at 30 speakers, against
  about half when utterances are resampled.
- `signature` and `parse_signature`: one line, after sacreBLEU's, that says
  how a set of numbers was computed: kudio's version, the rate and any
  resampling, how the pairs were lined up and cut, each metric's settings as
  resolved (PESQ's `auto` as `wb` or `nb`, defaults written out), the
  version of each package that computes one, the DNSMOS model and
  polynomials by a short hash, and, when given, the number of items and the
  confidence interval. `parse_signature` reads it back into a dict.
- `score_dataset` (a `DatasetScore`) and `kudio score` score a whole test
  set. The sides pair up by id, from folders laid out alike, Kaldi-style
  `id path` lists or CSV files, and an id missing from a side or given twice
  is refused before anything is scored. Each pair runs through the same code
  as `score`; `dnsmos`, `dnsmos_p808`, `srmr` and `wada_snr` score the file
  alone. The result is one row per utterance, as CSV or JSON lines with
  `score`'s warning where a metric could not score it, and a table of means
  with BCa intervals (resampled by speaker when the speakers are given),
  counts, signatures and BibTeX keys, as Markdown, LaTeX or JSON. With a
  baseline the table adds the difference, its sign-flip test and p-values
  adjusted by Holm over its rows. `jobs` scores in several processes and
  gives the same rows in the same order.

### Changed

- `kudio.enhance.traditional`'s `INIT` and `EST` hold only the MCRA estimate
  `trad_enhance` runs. The weighted spectral average, continuous minimum
  tracking and MCRA-2, which nothing in kudio called and no test covered,
  are gone, and so is `INIT`'s sample rate, which only MCRA-2 read.
- One rule for input a metric cannot score, in every metric: malformed
  input, NaN or inf samples included, raises `ValueError` starting
  `<function>(): `, and input with nothing to score gives nan and one
  `UserWarning` ending `; scored nan`, pointed at the line that called kudio.
- `snr`, `si_sdr`, `segmental_snr`, `sdi` and `pesq` follow it too. An empty
  pair or a silent reference, which gave a number made by the 1e-12 guards
  (`sdi` of an empty pair was 0.0) or raised the `pesq` package's error,
  scores nan with a warning, and so, for PESQ, do an empty side, a side under
  0.25 s, a pair in which the package finds no utterance and a silent
  output. `si_sdr` of a constant output is still 0 dB, now with a warning,
  and `segmental_snr` warns how many of the reference's frames are digital
  silence.
- Every metric takes the same signals: one channel as a vector, a column
  `(n, 1)` or a row `(1, n)`, integer samples read on their type's full
  scale. int64 and uint64 are refused, since on their full scale 16-bit
  samples sit 288 dB down, where the 1e-12 guards decide the score (`snr` of
  int16 samples held as int64 was 0 dB); so are complex, boolean and text
  arrays, masked samples, and a rate that is not one positive number. A
  signal that peaks under one step of 16-bit audio (-90.3 dBFS), where int16
  samples held as int32 land, is scored with one warning.
- `dnsmos` follows the rule as well: malformed input raises `ValueError`
  where it raised `FeatureError`, NaN and inf samples included, which it
  passed to the model, and an empty clip scores nan with a warning. A session
  whose model takes a mel spectrogram, as the P.808 one does, is refused by
  name rather than by onnxruntime's error. A model from the other P.835
  folder is warned about only when it is mapped with the polynomials it was
  not fitted to. `MODEL_NAMES` is now `MODELS`, which maps each model to its
  folder and file.
- STOI, ESTOI and SDR through `score`, `eval_metrics` and `AudioEvaluate`
  score nan with a warning for an empty pair, a pair under 25.6 ms for STOI,
  and a signal that is digital silence throughout, where they raised numpy's
  `AxisError` or mir_eval's errors, or ESTOI returned a number that changed
  from call to call.
- PESQ gets the two signals whole, whatever their lengths, as the ITU
  program reads two files: in `pesq`, `score`, `eval_metrics`,
  `AudioEvaluate`, `compare_enhancers` and `composite`. PESQ finds the delay
  itself and works over both as they are, so cutting the longer one, as
  kudio did, moved the score even where the cut dropped only silence: by up
  to 0.56 on the 40 pairs of ITU's conformance test 2(b), where kudio now
  gives ITU's scores to the third decimal. Pairs of one length score as
  before. `pesq`, `score` asked for
  PESQ alone and `AudioEvaluate.pesq_score` no longer warn about two lengths
  or refuse a pair past `MAX_LENGTH_MISMATCH`; beside metrics that cut, the
  length warning says PESQ got both whole.
- `AudioEvaluate` names, after each pass, the files a metric had no score
  for and why, in one warning per column.
- `import kudio` loads librosa, scipy.signal, PyWavelets and tqdm only where
  a function needs them, which takes it from about 500 ms to 140 ms (Python
  3.9, numpy 2.0, timed in a fresh interpreter). Every metric above runs
  without librosa. Without it, a feature built on it raises `DependencyError`
  naming it, and `convert_folder` and `enhance_folder` stop at the first
  file. librosa is still installed with kudio.

### Fixed

- `DependencyError` keeps its message when pickled, so one raised in a
  worker process reaches the parent as it was raised.
- `score`, `pesq` and `eval_metrics` take a one-sample signal, which they
  squeezed to a 0-d array and then refused as a shape they could not score.
- `score`, `pesq` and `eval_metrics` pass the rate on as it is given. They
  took `int(sr)` first, so 16000.5 scored there and was refused by `llr`
  called on its own.
- A pair too loud to square in float64 scores nan with one warning that
  says so. `snr`, `si_sdr`, `segmental_snr`, `sdi` and `lsd` gave nan with
  numpy's warnings alone, 790 of them from `segmental_snr` on 3 s; STOI and
  ESTOI called the pair too short, and SDR gave mir_eval's nan with no
  reason.
- `eer`, `min_dcf` and `det_curve` refuse a masked array with trials
  masked; the hidden values were counted.
- `AudioEvaluate.pesq_score_exe` scored nan for every pair with version 2.0
  of the ITU program, which prints the MOS-LQO after the raw score on the
  same line. It reads the raw score, the first number after the `=`, names
  in one warning the runs that gave no score, and refuses a rate other than
  8000 or 16000, at which the program prints its usage text instead.
- A numba that will not import with the installed numpy raises
  `DependencyError` with numba's own message at the first feature built on
  librosa, and `kudio pitch` reports it on one line. It got past
  `import librosa` and failed inside librosa as a plain `ImportError`.
- `compare_enhancers` reports a run whose output the length check refuses
  as a row with `error` set, the check's `ValueError` in it, and scores the
  other runs. A custom enhancer that returned a third of the clip raised
  out of the whole comparison.
- `enhance_folder` checks `noise=` and `noise_kwargs` before the first
  file, as it checks the method's parameters, and refuses a misspelt
  estimator or a key it does not take with one `FeatureError`. Each came
  back as the same failure on every file, "0/2 written, 2 failed" for two.

### Documentation

- A guide page for recognizers, verifiers and pitch trackers; sections on
  Hu and Loizou's measures, the log-spectral distance, the mel-cepstral
  distortion, NCM and CSII, SIIB, several sources, SRMR, WADA-SNR, input a
  metric cannot score, a whole test set, confidence intervals and tests, and
  reporting and citing a metric; `kudio score` on the command-line page; and
  a page on how each metric was checked, with what, and how closely it
  agrees.
- Every work the guide cites has its full reference: authors, title, venue,
  year, pages and DOI where there is one.
- `tools/validation/pesq_conformance.py` checks `pesq` against ITU's
  conformance test 2(b), from the files ITU gives out. The PESQ section of
  the guide says what P.862's Annex A requires, what the ITU program prints,
  and that ITU corrected wideband PESQ in 2018, which the `pesq` package does
  not include. The migration page no longer says most enhancement papers
  quote the raw score.
- `tools/validation/sii_tables.py` checks the ANSI S3.5-1997 numbers in
  `ncm` and `csii` against two free copies of the standard's tables, and the
  docstrings name the column of Table B.1 the importance values come from:
  short passages of easy reading material.

## [3.9.1] - 2026-10-24

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
