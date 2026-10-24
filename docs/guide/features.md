# STFT and features

Most of what a model gets from kudio is a matrix with one row per frame,
whether a log-power spectrogram, MFCCs or stacked log-mel vectors. Shapes are
where these go wrong, so every section below says what comes back.

| To | Use |
|---|---|
| get a log-power spectrogram and keep its settings with the model | [`STFT`][kudio.core.stft.STFT], or [`waveform_to_spectrogram`][kudio.core.feature.waveform_to_spectrogram] for a one-off |
| get compact features for a classifier | [`melspectrogram`][kudio.core.feature.melspectrogram], [`mfcc`][kudio.core.feature.mfcc] |
| give every row its neighbours, or cut fixed-length windows | [`stack_context`][kudio.core.feature.stack_context], [`ContextFrames`][kudio.core.feature.ContextFrames], [`frame_windows`][kudio.core.feature.frame_windows] |
| scale features with the training set's statistics | [`Standardizer`][kudio.core.feature.Standardizer] |
| turn a predicted spectrogram back into audio | [`STFT.inverse`][kudio.core.stft.STFT.inverse], [`save_spectrogram_as_wave`][kudio.core.feature.save_spectrogram_as_wave] |
| apply a complex mask or gain | [`STFT.analyse`][kudio.core.stft.STFT.analyse] and [`STFT.synthesise`][kudio.core.stft.STFT.synthesise] |
| compute log spectrograms or MFCCs for a list of files | [`logspec_from_files`][kudio.core.feature.logspec_from_files], [`mfcc_from_files`][kudio.core.feature.mfcc_from_files] |
| mark which frames hold speech | [`vad_frames`][kudio.core.vad.vad_frames], on [Speech activity and pitch](speech.md) |

All of these work on a whole clip. To draw a spectrogram while the audio is
still arriving, use [`SpectrogramStream`][kudio.core.spectrogram.SpectrogramStream]
([Recording and live views](live-audio.md)). Its columns are framed
differently (a Hann window, a hop of `n_fft // 4`, not centred) and hold dB
rather than bels, so a model trained on the rows here should not be fed them.

The examples run in order on one two-second clip at 16 kHz, and later blocks
use names made in earlier ones (`stft`, `spec`, `windows`, `std`). Load a
recording of your own with [`file_load`][kudio.core.io.file_load], or follow
along with noise:

```python
import numpy as np
import kudio

sr = 16000
y = (0.1 * np.random.default_rng(0).standard_normal(2 * sr)).astype(np.float32)
```

## Rows are frames

librosa returns `(bins, frames)`. Every function on this page returns
`(frames, bins)`, time first: a row is one moment of the signal, which is what
a frame classifier takes as one sample and what a sequence model steps along.
Transpose only for code that puts frequency first:
`plt.imshow(spec.T, origin="lower", aspect="auto")` to draw frequency upwards,
`librosa.display.specshow(spec.T)`, and kudio's own
[`spec_augment`][kudio.effects.augment.spec_augment], whose call and mask
value are on [Editing and effects](editing.md).

For the clip above, with every other argument at its default (`spec` is the
first row's result):

| Call | Shape | This clip | Values |
|---|---|---|---|
| `kudio.waveform_to_spectrogram(y)` | `(frames, n_fft // 2 + 1)` | `(126, 257)` | log10 power, float32 |
| `kudio.STFT().analyse(y)` | `(frames, bins)` | `(126, 257)` | complex64 |
| `kudio.mfcc(y, sr)` | `(1, mel_frames, n_mfcc)` | `(1, 63, 40)` | float32 |
| `kudio.melspectrogram(y, sr=sr)` | `(mel_frames - n_frames + 1, n_mels * n_frames)` | `(59, 320)` | dB, float32 |
| `kudio.melspectrogram(y, sr=sr, n_frames=1)` | `(mel_frames, n_mels)` | `(63, 64)` | dB, float32 |
| `kudio.stack_context(spec, 2)` | `(frames, bins * 5)` | `(126, 1285)` | edge-padded, float32 |
| `kudio.frame_windows(spec, 64)` | `(ceil(frames / 64), 64, bins)` | `(2, 64, 257)` | tail padded, float32 |
| `kudio.spectrogram_to_waveform(y, spec)` | `(len(y),)` | `(32000,)` | float32 |

Here a frame is one position of the STFT window, not a row of the
`(frames, channels)` arrays on [Audio files](audio-files.md). `frames` above is
`len(y) // 256 + 1`, 126 for this clip; `mel_frames` is the same count at the
512-sample hop of [`mfcc`][kudio.core.feature.mfcc] and
[`melspectrogram`][kudio.core.feature.melspectrogram], 63 here.
`melspectrogram` takes `hop_length=`; `mfcc` does not.

## One object for the STFT settings

A spectrogram and the waveform rebuilt from it have to agree on `n_fft`,
`hop_length`, `win_length` and `window`. Passed by hand to two functions, the
four drift apart the first time someone changes one of them, so
[`STFT`][kudio.core.stft.STFT] keeps them together as one value, the geometry
of the transform, along with the sample rate they were chosen for. Its
[`forward`][kudio.core.stft.STFT.forward] is
[`waveform_to_spectrogram`][kudio.core.feature.waveform_to_spectrogram] with
those settings:

```python
stft = kudio.STFT(sr=16000, n_fft=512, hop_length=256)
print(stft.win_length, stft.window)             # 512 hamming
print(stft.n_bins, stft.n_frames(len(y)))       # 257 126
print(stft.frame_times(3))                      # [0.    0.016 0.032]
spec = stft.forward(y)                          # (126, 257) float32
```

- The window is Hamming unless you choose another, and `win_length` defaults
  to `n_fft`. librosa's defaults are a Hann window, `n_fft=2048` and a
  512-sample hop, so a spectrogram made with them differs from one made here
  in shape as well as values.
- The transform is centred: the signal gets `n_fft // 2` zeros at each end and
  frame *t* is centred on sample `t * hop_length`.
  [`frame_times`][kudio.core.stft.STFT.frame_times] returns those centres in
  seconds.
- The transforms take one channel. A `(frames, channels)` array from
  [`file_load`][kudio.core.io.file_load] with `mono=False` is refused with
  [`FeatureError`][kudio.exceptions.FeatureError] rather than read as one
  short frame per sample. Mix down with `y.mean(axis=1)` or analyse each
  channel on its own.
- The object is frozen, so a geometry handed to another component cannot
  change under it.
- `forward(context=n)` is `waveform_to_spectrogram(forward_backward=n)`: it
  stacks *n* zero-padded neighbours on each side of every frame
  ([Context and windows](#context-and-windows)).

## Log power, and what `norm=True` does

[`waveform_to_spectrogram`][kudio.core.feature.waveform_to_spectrogram] returns
`log10(|X|**2)`, the power in bels; multiply by 10 for decibels. A bin with no
energy has no logarithm, so `np.finfo(float).eps` (2.2e-16) is added to the
complex spectrum first, and a digitally silent bin comes out at -31.3. Only
digital zeros get there: the rounding noise of a 16-bit file sits around -8. I
keep that floor because every model trained on these features has seen -31.3
as silence, and moving it would shift their inputs. If it gets in the way,
clamp it yourself, for example `np.maximum(spec, -10)` (-100 dB).

`norm=True` standardises each frequency bin over the clip's own frames, so
every bin comes out with mean 0 and standard deviation 1. Two things follow:

- The level of the recording is gone, and so is its overall spectral shape.
  Scaling the noise clip by 0.01 (-40 dB) moves the result by less than 2e-5.
- The statistics are not returned, so a `norm=True` spectrogram cannot be
  turned back into the audio it came from. For a model whose output has to
  become sound again, use a [`Standardizer`][kudio.core.feature.Standardizer]
  ([below](#standardising-with-the-training-sets-statistics)).

## Mel spectrograms and MFCC

[`melspectrogram`][kudio.core.feature.melspectrogram] returns stacked log-mel
*vectors*, not a spectrogram. Row *i* holds frames *i* to *i* + `n_frames` - 1
of a log-mel spectrogram, earliest first, so the defaults (`n_mels=64`,
`n_frames=5`) give rows of 320 values: one vector per position of a five-frame
sliding window, the input a frame-level autoencoder trains on.

```python
feats = kudio.melspectrogram(y, sr=sr)               # (59, 320)
mel = kudio.melspectrogram(y, sr=sr, n_frames=1)     # (63, 64), plain log-mel
aligned = kudio.melspectrogram(y, sr=sr, n_frames=1,
                               n_fft=512, hop_length=256)   # (126, 64)
```

![The same clip as a linear log-power spectrogram and as a 64-band mel spectrogram](../images/features-linear-mel-light.webp#only-light)
![The same clip as a linear log-power spectrogram and as a 64-band mel spectrogram](../images/features-linear-mel-dark.webp#only-dark)

Two synthetic voiced syllables with a hiss between them, both pictures made
with `n_fft=512` and `hop_length=256` and each relative to its own peak. Below
the dashed line at 1 kHz lies an eighth of the linear picture and a third of
the mel one (21 of 64 bands). Above it the mel bands widen until harmonics
that stay separate on the left merge into one band.

- With the default `n_frames=5`, row *i* is centred on frame *i* + 2, and
  there are four fewer rows than frames, 59 against 63 here. Per-frame labels
  on the same 512-sample grid line up as `labels[2:2 + len(feats)]`. For one
  row per frame, take `n_frames=1` and add the neighbours with
  [`stack_context`][kudio.core.feature.stack_context]:
  `kudio.stack_context(mel, 2)` is `(63, 320)`, the layout of `feats` with
  padded rows at both edges.
- `source` is a file path or a NumPy array. A path is loaded mono, at `sr=`
  when it is given and at the file's own rate when it is not, so a two-second
  44.1 kHz file gives 63 rows with `sr=16000, n_frames=1` and 173 without.
- An array needs `sr=` as a keyword (the second positional argument is
  `n_mels`); without it the call raises
  [`FeatureError`][kudio.exceptions.FeatureError].
- The defaults are `n_fft=1024` and `hop_length=512`, not the spectrogram's
  512 and 256. Pass both, as in the last line above, to get one row per row of
  the [`STFT`][kudio.core.stft.STFT]. The window stays Hann, librosa's
  default; `melspectrogram` has no window argument.
- A clip shorter than `n_frames` frames gives an empty `(0, n_mels * n_frames)`
  array, not an error, and [`frame_windows`][kudio.core.feature.frame_windows]
  refuses it with `FeatureError`. Drop such clips before they reach a training
  set.
- The result is float32, like the spectrograms, so the two stack without a
  cast.

[`mfcc`][kudio.core.feature.mfcc] takes an array and its rate, no path, and
returns `(1, mel_frames, n_mfcc)`; `[0]` drops the batch axis. Everything
except `n_mfcc` is librosa's default: Hann window, `n_fft=2048`, a 512-sample
hop and 128 mel bands. For MFCCs on the spectrogram's 256-sample grid, call
librosa directly and transpose:
`librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=512, hop_length=256).T` is
`(126, 13)`.

## Context and windows

Three helpers turn a `(frames, dim)` feature matrix, a spectrogram or
`mfcc(...)[0]` alike, into what a model trains on, as float32.
[`stack_context`][kudio.core.feature.stack_context] and
[`frame_windows`][kudio.core.feature.frame_windows] take one 2-D matrix;
[`ContextFrames`][kudio.core.feature.ContextFrames] takes a list of them, and
reads a 3-D array as one matrix per utterance.

### `stack_context`

[`stack_context`][kudio.core.feature.stack_context] joins each frame with the
`context` frames before and after it: `(frames, dim)` becomes
`(frames, dim * (2 * context + 1))`, oldest frame first and the frame itself in
the middle. The edges are padded rather than dropped, so there is one output
row per input row and a prediction made from row *t* belongs to frame *t*.

```python
frames = np.arange(12, dtype=np.float32).reshape(4, 3)
print(kudio.stack_context(frames, context=1))
print(kudio.stack_context(frames, context=1, pad="zero")[0])
```

```text
[[ 0.  1.  2.  0.  1.  2.  3.  4.  5.]
 [ 0.  1.  2.  3.  4.  5.  6.  7.  8.]
 [ 3.  4.  5.  6.  7.  8.  9. 10. 11.]
 [ 6.  7.  8.  9. 10. 11.  9. 10. 11.]]
[0. 0. 0. 0. 1. 2. 3. 4. 5.]
```

`pad="edge"`, the default, repeats the first and last frames; `pad="zero"`
pads with zeros. I made edge the default because a zero row is not silence in
any of these features: in log10 power, 0 is a power of 1 (silence is -31.3),
and after a [`Standardizer`][kudio.core.feature.Standardizer] it is an average
frame. `forward_backward=` and `STFT.forward(context=)` always pad with zeros.
For edge padding, compute the spectrogram without context and call
`stack_context` on it.

### `ContextFrames`

Stacking ±5 frames makes every row eleven frames wide, and a training set
stacked in advance takes eleven times the memory of its features.
[`ContextFrames`][kudio.core.feature.ContextFrames] keeps the features once and
stacks only the rows you ask for:

```python
rng = np.random.default_rng(0)
utterances = [rng.standard_normal((n, 257)).astype(np.float32) for n in (300, 120, 410)]

bank = kudio.ContextFrames(utterances, context=5)
print(bank.shape)                                            # (830, 2827)
batch = bank[[0, 5, 829]]                                    # (3, 2827)

full = np.vstack([kudio.stack_context(u, 5) for u in utterances])
kept = sum(u.nbytes for u in utterances) / 2**20
print(f"{kept:.2f} MiB of features, {full.nbytes / 2**20:.2f} MiB stacked")
# 0.81 MiB of features, 8.95 MiB stacked
print(np.array_equal(batch, full[[0, 5, 829]]))              # True
```

Each utterance is padded at its own edges, so no window reaches into the next
one. Row *i* of the bank is row *i* of that `np.vstack`, so targets built with
`np.vstack(targets)` line up index for index. `np.asarray(bank)` builds the
whole stack, which on a real corpus is the memory you meant to save.

### `frame_windows`

[`frame_windows`][kudio.core.feature.frame_windows] cuts a matrix into
non-overlapping windows of `n_frames`: `(frames, dim)` becomes
`(ceil(frames / n_frames), n_frames, dim)`, the shape a sequence model trains
on. The tail is padded, with copies of the last frame by default or zeros with
`pad="zero"`, so a clip shorter than one window still gives one window.

```python
windows = kudio.frame_windows(spec, n_frames=64)             # (2, 64, 257)
print(np.array_equal(windows.reshape(-1, 257)[:len(spec)], spec))   # True
```

The 126 frames fill two windows of 64 with two frames of padding, which a
model takes for real frames. Keep `len(spec)`: the loss should skip those
frames, and the way back to audio needs them cut off again
([Back to a waveform](#back-to-a-waveform)).

## Standardising with the training set's statistics

A [`Standardizer`][kudio.core.feature.Standardizer] holds a mean and a
standard deviation for each feature dimension (each bin of a spectrogram),
fitted on the training features. Those two arrays are as much a part of the
trained model as its weights: a model given inputs on a different scale, or not
standardised at all, still answers, only wrongly. The block below reads a
training set of WAV files under `data/train/`; to follow along without one,
use `training_clips = [y]`.

```python
from pathlib import Path

training_clips = [kudio.file_load(p, sr=stft.sr)[0]
                  for p in sorted(Path("data/train").glob("*.wav"))]
train = np.vstack([stft.forward(c) for c in training_clips])    # (frames, 257)
std = kudio.Standardizer().fit(train)

x = std.transform(windows)            # (2, 64, 257) float32, the model's input
print(np.allclose(std.inverse(x), windows, atol=1e-4))          # True
```

`fit` takes a 2-D matrix only. Fit on the frames before they are cut into
windows, as above, so the tail padding stays out of the statistics;
`transform` and [`inverse`][kudio.core.feature.Standardizer.inverse]
broadcast over any leading axes. Unlike `norm=True`, the statistics come from
the training set and are kept, so the recording level survives, relative to
the training data, and `inverse` undoes the scaling.

The standard deviation gets 1e-8 added, so a dimension that never changes in
the training set does not divide by zero, but a later difference of 0.1 in it
comes out as 1e7. Check `(std.std < 1e-6).any()` after fitting.

### Saving the geometry and the statistics together

[`to_dict`][kudio.core.stft.STFT.to_dict] gives the STFT settings as a plain
dict, ready for JSON, and `Standardizer.save` writes the two arrays to a
`.npz`, creating the folder if needed. Store both next to the checkpoint and
load both with it:

```python
import json

run = Path("runs/exp1")
run.mkdir(parents=True, exist_ok=True)
(run / "stft.json").write_text(json.dumps(stft.to_dict()))
std.save(run / "stats.npz")

# wherever the model is loaded again
stft = kudio.STFT.from_dict(json.loads((run / "stft.json").read_text()))
std = kudio.Standardizer.load(run / "stats.npz")

clip, _ = kudio.file_load("clip.wav", sr=stft.sr)   # any recording, at the model's rate
model_input = std.transform(stft.forward(clip))
```

`save` returns the path it wrote. numpy adds `.npz` to a name without it, so
`std.save(run / "stats")` writes and returns `stats.npz`, and
`Standardizer.load(run / "stats")` finds it. `from_dict` refuses a key it does
not know (`ValueError: unknown STFT key(s): n_ftt`, for a typo) and a missing
one, rather than fill it with a default the model was not trained on, so
write the file with `to_dict()`, not by hand.

The transforms never resample. For audio that arrives as an array with its own
rate, such as a recording, [`matches`][kudio.core.stft.STFT.matches] says
whether it needs [`resample`][kudio.core.io.resample] first.

## Back to a waveform

A model trained on standardised windows predicts standardised windows.
[`Standardizer.inverse`][kudio.core.feature.Standardizer.inverse], with the
statistics its targets were scaled with, turns a prediction back into log10
power, and [`STFT.inverse`][kudio.core.stft.STFT.inverse] or
[`spectrogram_to_waveform`][kudio.core.feature.spectrogram_to_waveform] turns
that into audio. A log-power spectrogram has no phase, so both take it from a
reference waveform, normally the noisy input the model has just cleaned up
(pairs like that come from [Building a noisy dataset](synthesis.md)):

```python
# x stands in for the model's prediction: standardised, in windows
cleaned = std.inverse(x).reshape(-1, stft.n_bins)[:len(spec)]   # (126, 257)
cleaned[:, stft.n_bins // 2:] -= 2.0  # bins from 4 kHz up, 20 dB down
out = stft.inverse(y, cleaned)        # magnitude from cleaned, phase from y
print(out.shape, out.dtype)           # (32000,) float32

change = stft.forward(out) - stft.forward(y)
print(round(float(change[:, stft.n_bins // 2:].mean()), 3))   # -1.987, in bels
```

- The rows have to be plain log10 power from the same geometry, one per frame
  of the reference. A `norm=True` spectrogram runs without an error and comes
  back at the wrong level with a flattened spectrum. Context-stacked rows, or
  windows with their tail padding still on, raise a
  [`FeatureError`][kudio.exceptions.FeatureError] that gives both shapes,
  which is why the block above flattens the windows and cuts them back to
  `len(spec)` rows.
- The output has exactly `len(y)` samples, and every one of them is rebuilt
  from the spectrogram, a last partial hop included. Unmodified, the round
  trip is off by at most 6e-8, on this clip and on its first second alone,
  which ends half-way through a hop.

[`save_spectrogram_as_wave`][kudio.core.feature.save_spectrogram_as_wave] does
the same from a file:
`kudio.save_spectrogram_as_wave("enhanced.wav", "noisy.wav", cleaned)` takes
the phase from `noisy.wav` at that file's own rate and writes 16-bit PCM,
clipped at full scale rather than wrapped around. It takes the geometry as
loose arguments with the default STFT's values; for any other, call
`stft.inverse` and [`save_wave`][kudio.core.io.save_wave].

When you need the phase itself, for a complex mask or a gain on the complex
spectrum, use [`analyse`][kudio.core.stft.STFT.analyse] and
[`synthesise`][kudio.core.stft.STFT.synthesise]. The spectral methods on
[Denoising](enhancement.md) run on this pair, with their own framing: a Hann
window and a hop of `n_fft // 4`.

```python
X = stft.analyse(y)                   # (126, 257) complex64
gain = np.ones(stft.n_bins, np.float32)
gain[stft.n_bins // 2:] = 0.1         # the same 20 dB, on the complex spectrum
out = stft.synthesise(X * gain, length=len(y))

# a log-power spectrogram with the phase of y, every sample rebuilt
out = stft.synthesise(np.sqrt(10.0 ** cleaned) * np.exp(1j * np.angle(X)),
                      length=len(y))
```

Pass `length=len(y)`. Without it `synthesise` stops at the last whole hop,
15872 samples into a one-second clip.

## Many files at once

[`file_to_spectrogram`][kudio.core.feature.file_to_spectrogram] loads a file
mono and returns what
[`waveform_to_spectrogram`][kudio.core.feature.waveform_to_spectrogram] would,
with the Hamming window unless `window=` names another. Two more helpers run
over a list of files and stack the results:

```python
files = sorted(Path("data/clean").glob("*.wav"))   # three 16 kHz files: 2.0, 1.5 and 1.0 s

rows = kudio.logspec_from_files(files, sequence=False, forward_backward=0,
                                norm=False)                          # (283, 257)
seqs = kudio.logspec_from_files(files, sequence=True, forward_backward=0,
                                norm=False, desired_samples=sr)      # (3, 63, 257)
coeffs = kudio.mfcc_from_files(files, n_mfcc=13, desired_samples=sr)    # (3, 32, 13)
```

- A 3-D result, from `sequence=True` in
  [`logspec_from_files`][kudio.core.feature.logspec_from_files] or from
  [`mfcc_from_files`][kudio.core.feature.mfcc_from_files], needs every file to
  give the same number of frames, or numpy's `vstack` raises `ValueError`.
  `desired_samples` cuts each file to that many samples, which evens them out
  only if none is shorter.
- In `mfcc_from_files`, `desired_length` is a whole number of **seconds**; in
  `waveform_to_spectrogram` and `file_to_spectrogram` it is samples.
- Each file is read at its own rate, so in a folder that mixes 8 and 16 kHz a
  frame and a bin mean different things from file to file. Convert the folder
  first ([Audio files](audio-files.md)).

## Where the parameters are

The STFT and feature functions and classes, with their signatures and
defaults, are in the API reference under
[STFT and features](../reference/features.md). Code that still calls `w2s` or
`wav2mel` is covered on [Migrating from v2](../migrating.md).
