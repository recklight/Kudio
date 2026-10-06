# -*- coding: utf-8 -*-
"""Feature extraction and spectrogram <-> waveform conversions.

v3 canonical names
------------------
=================================  =============================================
canonical                          deprecated aliases
=================================  =============================================
``waveform_to_spectrogram``        ``w2s``, ``wavform2spec``
``spectrogram_to_waveform``        ``spec2wavform``
``file_to_spectrogram``            ``f2s``, ``wav2spec``
``save_spectrogram_as_wave``       ``s2w``, ``spec2wav``
``mfcc``                           ``w2mfcc``, ``wav2mfcc``
``mfcc_from_files``                ``concat_mfcc``, ``concat_mfcc_``
``logspec_from_files``             ``concat_logspec``, ``concat_logspec_``,
                                   ``contextual_LogSpectrogram``
``melspectrogram``                 ``wav2mel``
=================================  =============================================

The aliases still work but emit a ``DeprecationWarning``.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional, Sequence, Tuple

import librosa
import numpy as np
from tqdm import tqdm

from kudio._deprecation import alias
from kudio.core.io import file_load, save_wave
from kudio.exceptions import FeatureError

__all__ = [
    # canonical
    'wave_separate', 'wave_slicing', 'features2matrix',
    'waveform_to_spectrogram', 'spectrogram_to_waveform',
    'file_to_spectrogram', 'save_spectrogram_as_wave',
    'mfcc', 'mfcc_from_files', 'logspec_from_files', 'melspectrogram',
    'denoise_wav2spec', 'denoise_spec2wav',
    # deprecated aliases (kept for compatibility)
    'w2s', 'wavform2spec', 'spec2wavform', 'f2s', 'wav2spec', 's2w', 'spec2wav',
    'w2mfcc', 'wav2mfcc', 'concat_mfcc', 'concat_mfcc_', 'concat_logspec',
    'concat_logspec_', 'contextual_LogSpectrogram', 'wav2mel',
]

log = logging.getLogger(__name__)

#: added to the complex spectrum so a silent bin has a logarithm. A Python
#: float, not ``np.finfo(float).eps``: under NumPy 2 a numpy float64 scalar
#: promotes a complex64 spectrum to complex128, doubling every spectrogram
#: computed from float32 audio. The value -- and the -31.3 a digitally silent
#: bin comes out at -- is unchanged, since trained models have seen it.
_EPS = float(np.finfo(float).eps)


def wave_separate(wave: np.ndarray, channels: int) -> list:
    """Split an interleaved multi-channel waveform into per-channel arrays."""
    if channels < 1:
        raise FeatureError(f"channels must be >= 1, got {channels}")
    return [wave[c::channels] for c in range(channels)]


def waveform_to_spectrogram(y: np.ndarray,
                            sequence: bool = False,
                            forward_backward: int = 0,
                            norm: bool = False,
                            hop_length: int = 256,
                            n_fft: int = 512,
                            win_length: Optional[int] = None,
                            window: str = 'hamming',
                            desired_length: Optional[int] = None) -> np.ndarray:
    """Waveform -> log-power spectrogram.

    Parameters
    ----------
    y : waveform, shape ``(n,)``
    sequence : return a 3-D time-series array ``(1, frames, dim)``
    forward_backward : if > 0, stack that many context frames on each side
        (zero-padded; see :func:`stack_context` for the standalone version)
    norm : per-frequency-bin mean/std normalization
    win_length : defaults to *n_fft*
    desired_length : truncate the waveform to this many samples first
    """
    if desired_length and desired_length < len(y):
        y = y[:desired_length]
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                     win_length=win_length, window=window)
    D = D + _EPS
    Sxx = np.log10(np.abs(D) ** 2)

    if norm:
        # statistics in float64 -- a steady bin has a tiny std, and dividing
        # by it in float32 leaves the normalised mean visibly off zero --
        # then stored at the precision the audio came in at
        S64 = Sxx.astype(np.float64)
        Sxx_mean = np.mean(S64, axis=1, keepdims=True)
        Sxx_std = np.std(S64, axis=1, keepdims=True) + 1e-12
        Sxx_r = ((S64 - Sxx_mean) / Sxx_std).astype(Sxx.dtype)
    else:
        Sxx_r = np.array(Sxx)

    if forward_backward:
        # same zero-padded stacking as before, now sharing one implementation
        # with the standalone stack_context()
        stacked = stack_context(Sxx_r.T, forward_backward, pad='zero')
        if sequence:
            return stacked.reshape(1, *stacked.shape)
        return stacked

    Sxx_r = np.array(Sxx_r).T
    if sequence:
        return Sxx_r.reshape(1, *Sxx_r.shape)
    return Sxx_r


def spectrogram_to_waveform(y: np.ndarray, enhanced_spec: np.ndarray,
                            sequence: bool = False, hop_length: int = 256,
                            n_fft: int = 512,
                            win_length: Optional[int] = None,
                            window: str = 'hamming') -> np.ndarray:
    """Log-power spectrogram -> waveform, reusing the phase of reference *y*."""
    if sequence:
        enhanced_spec = enhanced_spec.squeeze()
    D = librosa.stft(y.astype('float32'), hop_length=hop_length,
                     n_fft=n_fft, win_length=win_length, window=window)
    D = D + _EPS
    phase = np.exp(1j * np.angle(D))
    magnitude = np.sqrt(10 ** np.array(enhanced_spec)).T
    reverse = np.multiply(magnitude, phase)
    result = librosa.istft(reverse, hop_length=hop_length,
                           win_length=win_length, window=window)
    return librosa.util.fix_length(result, size=len(y), mode='edge')


def file_to_spectrogram(file,
                        sequence: bool = False,
                        forward_backward: int = 0,
                        norm: bool = False,
                        hop_length: int = 256,
                        n_fft: int = 512,
                        win_length: Optional[int] = None,
                        desired_length: Optional[int] = None,
                        resample_rate: Optional[int] = None) -> np.ndarray:
    """File -> log-power spectrogram (see :func:`waveform_to_spectrogram`)."""
    y, _ = file_load(file, sr=resample_rate, mono=True)
    return waveform_to_spectrogram(y, sequence, forward_backward, norm,
                                   hop_length, n_fft, win_length,
                                   desired_length=desired_length)


def save_spectrogram_as_wave(wave_out_dir, noisy_file, enhanced_spec,
                             squeeze: bool = False, hop_length: int = 256,
                             n_fft: int = 512,
                             win_length: Optional[int] = None,
                             window: str = 'hamming') -> None:
    """Reconstruct a waveform from *enhanced_spec* (phase from *noisy_file*)
    and write it to *wave_out_dir* as 16-bit PCM, clipped at full scale."""
    y, rate = librosa.load(str(noisy_file), sr=None)
    y_out = spectrogram_to_waveform(y, enhanced_spec, squeeze, hop_length,
                                    n_fft, win_length, window)
    save_wave(wave_out_dir, y_out, rate)


def mfcc(y: np.ndarray, sr: int, n_mfcc: int = 40) -> np.ndarray:
    """Waveform -> MFCC, shaped ``(1, n_frames, n_mfcc)``."""
    return librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc).T.reshape(1, -1, n_mfcc)


def mfcc_from_files(wav_list: Sequence, n_mfcc: int = 40,
                    desired_samples: Optional[int] = None,
                    desired_length: Optional[int] = None) -> np.ndarray:
    """Load each wave file and stack their MFCC matrices ``(n_files, frames, n_mfcc)``."""
    mfcc_list = []
    for wav_dir in tqdm(wav_list, desc="[kudio] MFCC"):
        y, sr = file_load(wav_dir)
        if desired_length:
            y = y[:sr * desired_length]
        if desired_samples and desired_samples < len(y):
            y = y[:desired_samples]
        m = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc).T
        mfcc_list.append(m.reshape(1, m.shape[0], -1))
    return np.vstack(mfcc_list)


def logspec_from_files(wav_list: Sequence,
                       sequence: bool,
                       forward_backward: int,
                       norm: bool,
                       hop_length: int = 256,
                       n_fft: int = 512,
                       win_length: Optional[int] = None,
                       desired_samples: Optional[int] = None) -> np.ndarray:
    """Convert each file to a log spectrogram and stack them."""
    return np.vstack([
        file_to_spectrogram(file=file, sequence=sequence,
                            forward_backward=forward_backward, norm=norm,
                            hop_length=hop_length, n_fft=n_fft,
                            win_length=win_length, desired_length=desired_samples)
        for file in tqdm(wav_list, desc='[kudio] log-spectrograms')
    ])


def melspectrogram(source, n_mels: int = 64, n_frames: int = 5,
                   n_fft: int = 1024, hop_length: int = 512,
                   power: float = 2.0, *,
                   sr: Optional[int] = None) -> np.ndarray:
    """Audio -> stacked log-mel feature vectors ``(n_vectors, n_mels * n_frames)``.

    *source* is a file path **or** a waveform array; arrays require *sr*, the
    same way :func:`mfcc` does.

    >>> feats = kudio.melspectrogram('clip.wav')            # doctest: +SKIP
    >>> feats = kudio.melspectrogram(y, sr=16000)           # doctest: +SKIP
    """
    dims = n_mels * n_frames
    if isinstance(source, np.ndarray):
        if sr is None:
            raise FeatureError("melspectrogram() needs sr= when given a waveform")
        y = source
    else:
        y, sr = file_load(source, mono=True)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft,
                                         hop_length=hop_length, n_mels=n_mels,
                                         power=power)
    log_mel = 20.0 / power * np.log10(np.maximum(mel, sys.float_info.epsilon))
    n_vectors = log_mel.shape[1] - n_frames + 1
    if n_vectors < 1:
        return np.empty((0, dims))
    vectors = np.zeros((n_vectors, dims))
    for t in range(n_frames):
        vectors[:, n_mels * t: n_mels * (t + 1)] = log_mel[:, t: t + n_vectors].T
    return vectors


def denoise_wav2spec(waveform: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Peak-normalized magnitude spectrogram + phase, for denoising models."""
    D = librosa.stft(waveform / np.max(np.abs(waveform)),
                     n_fft=512, hop_length=256, win_length=512, window='hann')
    D += np.finfo(float).eps
    Sxx = np.abs(D)
    phase = np.angle(D)
    Sxx_mean = np.mean(Sxx, axis=1, keepdims=True)
    Sxx_std = np.std(Sxx, axis=1, keepdims=True) + 1e-12
    Sxx_r = (Sxx - Sxx_mean) / Sxx_std
    return np.reshape(Sxx_r.T, (1, Sxx_r.shape[1], Sxx_r.shape[0])), phase


def denoise_spec2wav(spec: np.ndarray, phase: np.ndarray) -> np.ndarray:
    """Inverse of :func:`denoise_wav2spec`; output peak-normalized."""
    spec = np.squeeze(spec).T
    pred_wav = librosa.istft(np.multiply(spec, np.exp(1j * phase)),
                             hop_length=256, win_length=512, window='hann')
    return pred_wav / np.max(np.abs(pred_wav))


def stack_context(frames: np.ndarray, context: int,
                  pad: str = 'edge') -> np.ndarray:
    """Stack ``±context`` neighbouring frames onto each frame.

    ``(frames, dim)`` -> ``(frames, dim * (2 * context + 1))``, one output row
    per input row: the edges are padded rather than dropped, so a prediction
    made from the result lines up 1:1 with the input.

    Works on *any* feature matrix — MFCC, mel, log-spectrogram — unlike the
    ``forward_backward`` argument of :func:`waveform_to_spectrogram`, which
    only reaches the spectrogram it computes itself (and now delegates here).

    :param pad: ``'edge'`` repeats the first/last frame, ``'zero'`` pads with
        silence. ``'zero'`` is what ``forward_backward`` has always done;
        ``'edge'`` avoids inventing a spectral discontinuity at the boundary.

    >>> stack_context(np.zeros((10, 40)), context=2).shape
    (10, 200)
    """
    frames = np.asarray(frames, dtype=np.float32)
    if frames.ndim != 2:
        raise FeatureError(f"frames must be 2-D, got shape {frames.shape}")
    if context < 0:
        raise FeatureError(f"context must be >= 0, got {context}")
    if context == 0:
        return frames
    if pad not in ('edge', 'zero'):
        raise FeatureError(f"pad must be 'edge' or 'zero', got {pad!r}")

    mode = 'edge' if pad == 'edge' else 'constant'
    padded = np.pad(frames, ((context, context), (0, 0)), mode=mode)
    width = 2 * context + 1
    n, dim = frames.shape
    out = np.empty((n, dim * width), dtype=np.float32)
    for i in range(width):
        out[:, i * dim:(i + 1) * dim] = padded[i:i + n]
    return out


class ContextFrames:
    """:func:`stack_context` over a whole corpus, stacked a batch at a time.

    Stacking ±5 frames makes every row eleven rows wide, so a training set
    stacked up front takes eleven times the memory of its features. This
    keeps the features once -- each utterance padded at its own edges, so no
    window reaches into the next utterance -- and stacks only the rows asked
    for::

        >>> bank = kudio.ContextFrames(features, context=5)   # (frames, dim) each
        >>> bank.shape
        (707112, 2827)
        >>> x = bank[batch]                # indices, a slice or a mask

    Row *i* is row *i* of ``np.vstack([stack_context(f, context, pad) for f in
    features])``, so a target built with ``np.vstack(targets)`` lines up with
    it index for index.
    """

    def __init__(self, utterances, context: int, pad: str = 'edge'):
        if context < 0:
            raise FeatureError(f"context must be >= 0, got {context}")
        if pad not in ('edge', 'zero'):
            raise FeatureError(f"pad must be 'edge' or 'zero', got {pad!r}")
        if isinstance(utterances, np.ndarray) and utterances.ndim == 2:
            utterances = [utterances]
        mats = [np.asarray(u, dtype=np.float32) for u in utterances]
        if not mats:
            raise FeatureError("ContextFrames needs at least one utterance")
        if any(m.ndim != 2 for m in mats) or len({m.shape[1] for m in mats}) != 1:
            raise FeatureError("every utterance must be (frames, dim) with one "
                               f"dim, got {sorted({m.shape for m in mats})}")
        self.context = int(context)
        self.dim = int(mats[0].shape[1])
        mode = 'edge' if pad == 'edge' else 'constant'
        padded, centres, offset = [], [], 0
        for m in mats:
            if not len(m):
                continue
            padded.append(np.pad(m, ((context, context), (0, 0)), mode=mode))
            centres.append(offset + context + np.arange(len(m)))
            offset += len(m) + 2 * context
        self._frames = (np.concatenate(padded) if padded
                        else np.empty((0, self.dim), np.float32))
        self._centres = (np.concatenate(centres) if centres
                         else np.empty(0, dtype=np.intp))
        self._window = np.arange(-context, context + 1)

    @property
    def shape(self) -> Tuple[int, int]:
        return len(self), self.dim * len(self._window)

    def __len__(self) -> int:
        return len(self._centres)

    def __getitem__(self, index) -> np.ndarray:
        centres = self._centres[index]
        rows = np.atleast_1d(centres)[:, None] + self._window
        out = self._frames[rows].reshape(len(rows), -1)
        return out[0] if np.ndim(centres) == 0 else out

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        out = self[:]
        return out if dtype is None else out.astype(dtype)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"<ContextFrames {self.shape[0]} x {self.shape[1]} "
                f"(context ±{self.context}, {self._frames.nbytes / 2**20:.0f} MiB)>")


def frame_windows(frames: np.ndarray, n_frames: int,
                  pad: str = 'edge') -> np.ndarray:
    """Cut a feature matrix into fixed-length windows.

    ``(frames, dim)`` -> ``(n_windows, n_frames, dim)``. The tail is padded
    rather than dropped, so a clip shorter than one window still yields one —
    the shape sequence models are trained on.

    >>> frame_windows(np.zeros((60, 40)), n_frames=16).shape
    (4, 16, 40)
    """
    frames = np.asarray(frames, dtype=np.float32)
    if frames.ndim != 2:
        raise FeatureError(f"frames must be 2-D, got shape {frames.shape}")
    if n_frames < 1:
        raise FeatureError(f"n_frames must be >= 1, got {n_frames}")
    if pad not in ('edge', 'zero'):
        raise FeatureError(f"pad must be 'edge' or 'zero', got {pad!r}")

    n, dim = frames.shape
    n_windows = max(1, int(np.ceil(n / n_frames)))
    missing = n_windows * n_frames - n
    if missing:
        mode = 'edge' if pad == 'edge' else 'constant'
        frames = np.pad(frames, ((0, missing), (0, 0)), mode=mode)
    return frames.reshape(n_windows, n_frames, dim)


class Standardizer:
    """Per-column mean/std normalisation, with the statistics saved alongside.

    Fit on the training features, then reused verbatim at inference — the
    statistics are as much a part of a trained model as its weights, and a
    model loaded without them produces confident nonsense.

    >>> std = kudio.Standardizer().fit(train_frames)
    >>> x = std.transform(frames)
    >>> frames_again = std.inverse(x)
    >>> std.save('runs/exp1/stats.npz')
    """

    def __init__(self, mean: Optional[np.ndarray] = None,
                 std: Optional[np.ndarray] = None):
        self.mean = mean
        self.std = std

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        if not self.fitted:
            return "<Standardizer unfitted>"
        return f"<Standardizer dim={len(self.mean)}>"

    @property
    def fitted(self) -> bool:
        return self.mean is not None and self.std is not None

    def fit(self, frames: np.ndarray) -> "Standardizer":
        frames = np.asarray(frames, dtype=np.float32)
        if frames.ndim != 2:
            raise FeatureError(f"frames must be 2-D, got shape {frames.shape}")
        self.mean = frames.mean(axis=0)
        self.std = frames.std(axis=0) + 1e-8
        return self

    def transform(self, frames: np.ndarray) -> np.ndarray:
        self._check()
        return ((np.asarray(frames, dtype=np.float32) - self.mean)
                / self.std).astype(np.float32)

    def inverse(self, frames: np.ndarray) -> np.ndarray:
        """Undo :meth:`transform` — used to map a prediction back."""
        self._check()
        return (np.asarray(frames, dtype=np.float32) * self.std
                + self.mean).astype(np.float32)

    def fit_transform(self, frames: np.ndarray) -> np.ndarray:
        return self.fit(frames).transform(frames)

    def _check(self) -> None:
        if not self.fitted:
            raise FeatureError("Standardizer has not been fitted")

    def save(self, path) -> Path:
        self._check()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(path), mean=self.mean, std=self.std)
        return path

    @classmethod
    def load(cls, path) -> "Standardizer":
        with np.load(str(path)) as data:
            return cls(mean=data['mean'], std=data['std'])


def wave_slicing(y: np.ndarray, frame_length: int = 2048,
                 hop_length: int = 512, is_reshape: bool = True) -> np.ndarray:
    """Slice a waveform into overlapping frames (flattened if *is_reshape*)."""
    frame_stack = librosa.util.frame(y, frame_length=frame_length,
                                     hop_length=hop_length)
    if is_reshape:
        frame_stack = frame_stack.T.reshape(-1)
    return frame_stack


def features2matrix(features: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Concatenate feature matrices; label rows by list index."""
    feature_matrix = np.vstack(features)
    labels = np.concatenate([i * np.ones(len(f)) for i, f in enumerate(features)])
    return feature_matrix, labels


# -- deprecated aliases (v2 names) ----------------------------------------------
w2s = alias(waveform_to_spectrogram, 'w2s')
wavform2spec = alias(waveform_to_spectrogram, 'wavform2spec')
spec2wavform = alias(spectrogram_to_waveform, 'spec2wavform')
f2s = alias(file_to_spectrogram, 'f2s')
wav2spec = alias(file_to_spectrogram, 'wav2spec')
s2w = alias(save_spectrogram_as_wave, 's2w')
spec2wav = alias(save_spectrogram_as_wave, 'spec2wav')
w2mfcc = alias(mfcc, 'w2mfcc')
wav2mfcc = alias(mfcc, 'wav2mfcc')
concat_mfcc = alias(mfcc_from_files, 'concat_mfcc')
concat_mfcc_ = alias(mfcc_from_files, 'concat_mfcc_')
concat_logspec = alias(logspec_from_files, 'concat_logspec')
concat_logspec_ = alias(logspec_from_files, 'concat_logspec_')
contextual_LogSpectrogram = alias(logspec_from_files, 'contextual_LogSpectrogram')
wav2mel = alias(melspectrogram, 'wav2mel')
