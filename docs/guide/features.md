# STFT and features

What a model gets from kudio is a matrix with one row per frame: a log-power
spectrogram, MFCCs or stacked log-mel vectors. The examples run in order on one
two-second clip at 16 kHz:

```python
import numpy as np
import kudio

sr = 16000
y = (0.1 * np.random.default_rng(0).standard_normal(2 * sr)).astype(np.float32)
```

For audio still arriving there is
[`SpectrogramStream`][kudio.core.spectrogram.SpectrogramStream]
([Recording and live views](live-audio.md)), framed differently and in dB.

## Rows are frames

Every function here returns `(frames, bins)`, time first; transpose for code
that puts frequency first. For the clip above, at the defaults:

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

`frames` is `len(y) // 256 + 1`; `mel_frames` is the same count at the
512-sample hop of [`mfcc`][kudio.core.feature.mfcc] and
[`melspectrogram`][kudio.core.feature.melspectrogram]. Both take `hop_length=`.

## One object for the STFT settings

[`STFT`][kudio.core.stft.STFT] keeps `n_fft`, `hop_length`, `win_length`,
`window` and the rate together, so the spectrogram and the waveform rebuilt from
it cannot drift apart:

```python
stft = kudio.STFT(sr=16000, n_fft=512, hop_length=256)
spec = stft.forward(y)
print(stft.win_length, stft.window, stft.n_bins, stft.n_frames(len(y)))
print(stft.frame_times(3), spec.shape, spec.dtype)
```

```text
512 hamming 257 126
[0.    0.016 0.032] (126, 257) float32
```

The window is Hamming and `win_length` is `n_fft` unless you choose
otherwise. The transform is centred: frame *t* is centred on sample
`t * hop_length`. It takes one channel; a `(frames, channels)` array is refused
with [`FeatureError`][kudio.exceptions.FeatureError]. The object is frozen.

## Log power, and what `norm=True` does

[`waveform_to_spectrogram`][kudio.core.feature.waveform_to_spectrogram] returns
`log10(|X|**2)`, the power in bels; multiply by 10 for decibels. 2.2e-16 is
added to the complex spectrum first, so a digitally silent bin reads -31.3; the
rounding noise of a 16-bit file sits around -8. Clamp with
`np.maximum(spec, -10)` if the floor gets in the way.

`norm=True` standardises each bin over the clip's own frames, to mean 0 and
standard deviation 1. That removes the level and the overall spectral shape,
and the statistics are not returned, so such a spectrogram cannot be turned back
into audio. A model whose output becomes sound again needs a
[`Standardizer`][kudio.core.feature.Standardizer]
([below](#standardising-with-the-training-sets-statistics)).

## Mel spectrograms and MFCC

[`melspectrogram`][kudio.core.feature.melspectrogram] returns stacked log-mel
vectors: row *i* holds frames *i* to *i* + `n_frames` - 1, so the defaults
(`n_mels=64`, `n_frames=5`) give rows of 320 values, centred on frame *i* + 2.

```python
feats = kudio.melspectrogram(y, sr=sr)
mel = kudio.melspectrogram(y, sr=sr, n_frames=1)
aligned = kudio.melspectrogram(y, sr=sr, n_frames=1, n_fft=512, hop_length=256)
coeffs = kudio.mfcc(y, sr, n_mfcc=13, n_fft=512, hop_length=256)[0]
print(feats.shape, mel.shape, aligned.shape, coeffs.shape)
```

```text
(59, 320) (63, 64) (126, 64) (126, 13)
```

![The same clip as a linear log-power spectrogram and as a 64-band mel spectrogram](../images/features-linear-mel-light.webp#only-light)
![The same clip as a linear log-power spectrogram and as a 64-band mel spectrogram](../images/features-linear-mel-dark.webp#only-dark)

Below 1 kHz (dashed) lies an eighth of the linear picture and a third of the
mel one. `source` is a path, loaded mono, or an array, which needs `sr=` as a
keyword. The defaults are `n_fft=1024`, `hop_length=512` and a Hann window. A
clip shorter than `n_frames` frames gives an empty `(0, n_mels * n_frames)` array.

[`mfcc`][kudio.core.feature.mfcc] takes an array and its rate and returns
`(1, mel_frames, n_mfcc)`; `[0]` drops the batch axis. Left out, `n_fft` is
2048, `hop_length` 512, `n_mels` 128 and the window Hann. `n_mfcc`, 40 by
default, may not exceed `n_mels`.

## Context and windows

### `stack_context`

[`stack_context`][kudio.core.feature.stack_context] joins each frame with the
`context` frames before and after it, oldest first. The edges are padded, so
row *t* of the output belongs to frame *t*:

```python
frames = np.arange(12, dtype=np.float32).reshape(4, 3)
print(kudio.stack_context(frames, context=1))
```

```text
[[ 0.  1.  2.  0.  1.  2.  3.  4.  5.]
 [ 0.  1.  2.  3.  4.  5.  6.  7.  8.]
 [ 3.  4.  5.  6.  7.  8.  9. 10. 11.]
 [ 6.  7.  8.  9. 10. 11.  9. 10. 11.]]
```

`pad="edge"`, the default, repeats the first and last frames, since a zero row
is not silence in log power; `pad="zero"`, `forward_backward=` and
`STFT.forward(context=)` pad with zeros.

### `ContextFrames`

[`ContextFrames`][kudio.core.feature.ContextFrames] keeps a list of feature
matrices once and stacks only the rows you ask for, each utterance padded at its own edges,
row *i* being row *i* of the `np.vstack` below:

```python
rng = np.random.default_rng(0)
utterances = [rng.standard_normal((n, 257)).astype(np.float32) for n in (300, 120, 410)]
bank = kudio.ContextFrames(utterances, context=5)
full = np.vstack([kudio.stack_context(u, 5) for u in utterances])
print(bank.shape, np.array_equal(bank[[0, 5, 829]], full[[0, 5, 829]]))
```

```text
(830, 2827) True
```

### `frame_windows`

[`frame_windows`][kudio.core.feature.frame_windows] cuts a matrix into
non-overlapping windows of `n_frames`, padding the tail with copies of the last
frame, or zeros with `pad="zero"`:

```python
windows = kudio.frame_windows(spec, n_frames=64)
print(windows.shape, np.array_equal(windows.reshape(-1, 257)[:len(spec)], spec))
```

```text
(2, 64, 257) True
```

Keep `len(spec)`: the loss should skip the padded frames, and the way back to
audio cuts them off again. A matrix with no frames raises `FeatureError`.

## Standardising with the training set's statistics

A [`Standardizer`][kudio.core.feature.Standardizer] holds a mean and a standard
deviation per feature dimension, fitted on the training frames. Those two arrays
belong to the trained model as much as its weights.

```python
std = kudio.Standardizer().fit(spec)     # fit on the training set's frames
x = std.transform(windows)               # the model's input
print(x.shape, x.dtype, np.allclose(std.inverse(x), windows, atol=1e-4))
```

```text
(2, 64, 257) float32 True
```

`fit` takes a 2-D matrix, so the window padding stays out of the statistics;
[`inverse`][kudio.core.feature.Standardizer.inverse] undoes `transform`. 1e-8 is
added to the standard deviation: check `(std.std < 1e-6).any()`.

### Saving the geometry and the statistics together

[`to_dict`][kudio.core.stft.STFT.to_dict] gives the STFT settings as plain JSON
for `STFT.from_dict`, and `std.save("stats.npz")` writes the statistics for
`Standardizer.load`; store both beside the checkpoint. `from_dict` refuses an
unknown or missing key rather than fill in a default the model was not trained
on. The transforms never resample; [`matches`][kudio.core.stft.STFT.matches]
says whether audio needs [`resample`][kudio.core.io.resample] first.

## Back to a waveform

A log-power spectrogram has no phase, so [`STFT.inverse`][kudio.core.stft.STFT.inverse]
and [`spectrogram_to_waveform`][kudio.core.feature.spectrogram_to_waveform] take
it from a reference waveform, normally the noisy input the model cleaned:

```python
cleaned = std.inverse(x).reshape(-1, stft.n_bins)[:len(spec)]   # the prediction
cleaned[:, stft.n_bins // 2:] -= 2.0        # bins from 4 kHz up, 20 dB down
out = stft.inverse(y, cleaned)              # magnitude from cleaned, phase from y
change = stft.forward(out) - stft.forward(y)
print(out.shape, out.dtype, round(float(change[:, stft.n_bins // 2:].mean()), 3))
```

```text
(32000,) float32 -1.987
```

The rows have to be plain log10 power from the same geometry, one per frame of
the reference; context-stacked rows or padded windows raise a `FeatureError`
that gives both shapes. The output has exactly `len(y)` samples, and the round
trip of an unmodified spectrogram is off by at most 6e-8.
[`save_spectrogram_as_wave`][kudio.core.feature.save_spectrogram_as_wave] does
the same from a file and writes 16-bit PCM.

For the phase itself, [`analyse`][kudio.core.stft.STFT.analyse] gives the
complex spectrum and [`synthesise`][kudio.core.stft.STFT.synthesise] takes it
back; pass `length=len(y)`, or it stops at the last whole hop.

## Many files at once

```python
from pathlib import Path

for name, seconds in [("a", 2.0), ("b", 1.5), ("c", 1.0)]:
    kudio.save_wave(f"data/clean/{name}.wav", y[:int(seconds * sr)], sr)
files = sorted(Path("data/clean").glob("*.wav"))
rows = kudio.logspec_from_files(files, sequence=False, forward_backward=0, norm=False)
seqs = kudio.logspec_from_files(files, sequence=True, forward_backward=0,
                                norm=False, desired_samples=sr)
coeffs = kudio.mfcc_from_files(files, n_mfcc=13, desired_samples=sr)
print(rows.shape, seqs.shape, coeffs.shape)
```

```text
(283, 257) (3, 63, 257) (3, 32, 13)
```

[`logspec_from_files`][kudio.core.feature.logspec_from_files] and
[`mfcc_from_files`][kudio.core.feature.mfcc_from_files] stack per-file results;
a 3-D result needs the same number of frames from every file, which
`desired_samples` gives when no file is shorter. Each file is read at its own
rate. The signatures are under [STFT and features](../reference/features.md).
