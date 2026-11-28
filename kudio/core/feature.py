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
import os
import sys
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np

from kudio._deprecation import alias
from kudio._numbers import count, flag, real, whole
from kudio._progress import tqdm
from kudio._samples import as_float, channels_last, finite, one_channel, real_samples
from kudio.core import _mel, _spectral
from kudio.core._policy import sample_rate
from kudio.core.io import PathLike, _read_mono, _reason, file_load, save_wave
from kudio.exceptions import AudioIOError, FeatureError

__all__ = [
    # canonical
    'wave_separate', 'wave_slicing', 'features2matrix',
    'waveform_to_spectrogram', 'spectrogram_to_waveform',
    'file_to_spectrogram', 'save_spectrogram_as_wave',
    'mfcc', 'mfcc_from_files', 'logspec_from_files', 'melspectrogram',
    'mel_band_hz',
    'denoise_wav2spec', 'denoise_spec2wav',
    'stack_context', 'ContextFrames', 'frame_windows', 'Standardizer',
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


def _mono(y, where: str) -> np.ndarray:
    """*y* as a 1-D float array, integers on their own full scale, or a
    FeatureError."""
    return _one_axis(one_channel(y, where, FeatureError), where)


def _one_axis(y: np.ndarray, where: str) -> np.ndarray:
    if y.ndim != 1:
        raise FeatureError(f"{where}() takes a mono waveform (n,), got shape {y.shape}; "
                           f"pick a channel or mix down first")
    return y


def _clip(y, where: str) -> np.ndarray:
    """:func:`_mono`, of finite numbers."""
    return finite(_mono(y, where), where, FeatureError)


def _samples(value, name: str, where: str) -> int:
    """*value* as a number of samples, 1 or more."""
    return count(value, name, where, FeatureError, ' of samples')


def _keep(value, name: str, where: str) -> Optional[int]:
    """How many samples to cut a clip to, None for all of it."""
    if value is None:
        return None
    keep = whole(value)
    if keep is None or keep < 1:
        raise FeatureError(f"{where}(): {name} is a whole number of samples to keep, 1 or more, "
                           f"or None for all of them, got {value!r}")
    return keep


def _geometry(n_fft, hop_length, win_length, where: str) -> Tuple[int, int, int]:
    """The three sizes of a transform, *win_length* defaulting to *n_fft*."""
    n_fft = _samples(n_fft, 'n_fft', where)
    win_length = n_fft if win_length is None else _samples(win_length, 'win_length', where)
    if win_length > n_fft:
        raise FeatureError(f"{where}(): win_length ({win_length}) cannot exceed n_fft ({n_fft})")
    return n_fft, _samples(hop_length, 'hop_length', where), win_length


def wave_separate(wave: np.ndarray, channels: int) -> list:
    """Split a multi-channel waveform into per-channel arrays, each a view of
    *wave*: an interleaved one, ``(n,)``, by taking every *channels*-th sample,
    and ``(frames, channels)`` by its columns. A row or a column is the
    interleaved waveform it holds.

    :raises FeatureError: for an array that is channels first, one with another
        number of columns than *channels*, an interleaved one that is no whole
        number of frames, and a *channels* that is not a whole number of 1 or more.
    """
    channels = count(channels, 'channels', 'wave_separate', FeatureError)
    wave = channels_last(wave, 'wave_separate', FeatureError)
    if wave.ndim == 2 and wave.shape[1] == 1:
        wave = wave[:, 0]
    if wave.ndim not in (1, 2):
        raise FeatureError(f"wave_separate() takes (frames,) or (frames, channels), got shape "
                           f"{wave.shape}")
    if wave.ndim == 1:
        if len(wave) % channels:
            raise FeatureError(f"wave_separate(): {len(wave)} interleaved samples are no whole "
                               f"number of frames of {channels} channels")
        return [wave[c::channels] for c in range(channels)]
    if wave.shape[1] != channels:
        raise FeatureError(f"wave_separate(): the array holds {wave.shape[1]} channels, and "
                           f"{channels} {'was' if channels == 1 else 'were'} asked for")
    return [wave[:, c] for c in range(channels)]


def waveform_to_spectrogram(y: np.ndarray,
                            sequence: bool = False,
                            forward_backward: int = 0,
                            norm: bool = False,
                            hop_length: int = 256,
                            n_fft: int = 512,
                            win_length: Optional[int] = None,
                            window: str = 'hamming',
                            desired_length: Optional[int] = None) -> np.ndarray:
    """Waveform -> log-power spectrogram ``(frames, bins)``, ``log10(|X|^2)``.

    :param y: a mono waveform, shape ``(n,)``.
    :param sequence: return ``(1, frames, dim)`` instead, for a sequence model.
    :param forward_backward: stack this many context frames on each side,
        zero-padded (:func:`stack_context` does the same for any feature).
    :param norm: standardise each frequency bin over this clip.
    :param hop_length: samples between frames.
    :param n_fft: FFT size; there are ``n_fft // 2 + 1`` bins.
    :param win_length: window length, defaulting to *n_fft*.
    :param window: a window :func:`scipy.signal.get_window` takes, by name or as
        ``(name, parameter)``, or its values.
    :param desired_length: cut the waveform to this many samples first.
    :raises FeatureError: for anything but a 1-D waveform of finite samples, for
        a size that is not a whole number of 1 or more, and for a window there
        is not. Of a ``(frames, channels)`` array, pick a channel or mix down
        first.

    A clip shorter than a frame is transformed with the zeros around it, and a
    warning says so.
    """
    sequence = flag(sequence, 'sequence', 'waveform_to_spectrogram', FeatureError)
    norm = flag(norm, 'norm', 'waveform_to_spectrogram', FeatureError)
    where = "waveform_to_spectrogram"
    n_fft, hop_length, win_length = _geometry(n_fft, hop_length, win_length, where)
    y = _clip(y, where)[:_keep(desired_length, 'desired_length', where)]
    D = _spectral.stft(y, n_fft, hop_length, win_length, window, where)
    D += _EPS
    Sxx = np.abs(D)
    del D
    Sxx **= 2
    np.log10(Sxx, out=Sxx)

    if norm:
        # statistics in float64 -- a steady bin has a tiny std, and dividing
        # by it in float32 leaves the normalised mean visibly off zero --
        # then stored at the precision the audio came in at
        S64 = Sxx.astype(np.float64)
        Sxx_mean = np.mean(S64, axis=1, keepdims=True)
        Sxx_std = np.std(S64, axis=1, keepdims=True) + 1e-12
        Sxx = ((S64 - Sxx_mean) / Sxx_std).astype(Sxx.dtype)

    spec = Sxx.T                                  # (frames, bins)
    if forward_backward:
        spec = stack_context(spec, forward_backward, pad='zero')
    return spec.reshape(1, *spec.shape) if sequence else spec


def spectrogram_to_waveform(y: np.ndarray, enhanced_spec: np.ndarray,
                            sequence: bool = False, hop_length: int = 256,
                            n_fft: int = 512,
                            win_length: Optional[int] = None,
                            window: str = 'hamming') -> np.ndarray:
    """Log-power spectrogram -> waveform, reusing the phase of reference *y*.

    *enhanced_spec* is ``(frames, bins)`` with one row per frame of *y*, the
    shape :func:`waveform_to_spectrogram` gives for it. The result has exactly
    ``len(y)`` samples.
    """
    sequence = flag(sequence, 'sequence', 'spectrogram_to_waveform', FeatureError)
    where = "spectrogram_to_waveform"
    n_fft, hop_length, win_length = _geometry(n_fft, hop_length, win_length, where)
    if sequence:
        enhanced_spec = np.squeeze(enhanced_spec)
    y = channels_last(real_samples(y, where, FeatureError), where, FeatureError)
    y = _one_axis(y[:, 0] if y.ndim == 2 and y.shape[1] == 1 else y, where)
    if np.issubdtype(y.dtype, np.unsignedinteger):
        # only the phase is taken from *y*, which no scale changes; the
        # midpoint unsigned audio sits on is an offset, which does
        y = as_float(y)
    finite(y, where, FeatureError)
    D = _spectral.stft(np.asarray(y, dtype=np.float32), n_fft, hop_length, win_length,
                       window, where)
    D += _EPS
    enhanced_spec = np.asarray(enhanced_spec)
    if enhanced_spec.shape != D.T.shape:
        raise FeatureError(
            f"the spectrogram is {enhanced_spec.shape} but the reference gives "
            f"{D.T.shape} (frames, bins): it needs one row per frame of the "
            f"reference, with no context stacked and no window padding")
    enhanced_spec = _numbers(enhanced_spec, where)
    if enhanced_spec.dtype.kind in 'iu':
        enhanced_spec = enhanced_spec.astype(np.float64)
    bad = int(np.count_nonzero(np.isnan(enhanced_spec) | np.isposinf(enhanced_spec)))
    if bad:                                             # -inf is no power at all
        raise FeatureError(f"{where}(): {bad} number(s) of the spectrogram are NaN or inf")
    phase = np.exp(1j * np.angle(D))
    del D
    magnitude = np.sqrt(10 ** enhanced_spec).T
    reverse = np.multiply(magnitude, phase)
    del magnitude, phase
    # with its length the inverse rebuilds the last partial hop; trimming and
    # padding afterwards repeated one sample over the last len(y) % hop_length
    return _spectral.istft(reverse, n_fft, hop_length, win_length, window, len(y), where)


def file_to_spectrogram(file: PathLike,
                        sequence: bool = False,
                        forward_backward: int = 0,
                        norm: bool = False,
                        hop_length: int = 256,
                        n_fft: int = 512,
                        win_length: Optional[int] = None,
                        desired_length: Optional[int] = None,
                        resample_rate: Optional[int] = None,
                        window: str = 'hamming') -> np.ndarray:
    """File -> log-power spectrogram (see :func:`waveform_to_spectrogram`).

    *window* comes last, after the arguments callers already pass by
    position.
    """
    sequence = flag(sequence, 'sequence', 'file_to_spectrogram', FeatureError)
    norm = flag(norm, 'norm', 'file_to_spectrogram', FeatureError)
    y, _ = file_load(file, sr=resample_rate, mono=True)
    return waveform_to_spectrogram(y, sequence, forward_backward, norm,
                                   hop_length, n_fft, win_length, window,
                                   desired_length=desired_length)


def save_spectrogram_as_wave(wave_out_dir: PathLike, noisy_file: PathLike,
                             enhanced_spec: np.ndarray,
                             squeeze: bool = False, hop_length: int = 256,
                             n_fft: int = 512,
                             win_length: Optional[int] = None,
                             window: str = 'hamming') -> None:
    """Reconstruct a waveform from *enhanced_spec* (phase from *noisy_file*)
    and write it to *wave_out_dir* as 16-bit PCM, clipped at full scale."""
    squeeze = flag(squeeze, 'squeeze', 'save_spectrogram_as_wave', FeatureError)
    try:
        y, rate = _read_mono(noisy_file)
    except Exception as e:
        raise AudioIOError(f"Could not load audio file: {noisy_file} "
                           f"({_reason(e, noisy_file)})") from e
    y_out = spectrogram_to_waveform(y, enhanced_spec, squeeze, hop_length,
                                    n_fft, win_length, window)
    save_wave(wave_out_dir, y_out, rate)


def mfcc(y: np.ndarray, sr: int, n_mfcc: int = 40, *,
         n_fft: Optional[int] = None, hop_length: Optional[int] = None,
         n_mels: Optional[int] = None, fmax: Optional[float] = None) -> np.ndarray:
    """Waveform -> MFCC, shaped ``(1, n_frames, n_mfcc)``. *y* is mono.

    The orthonormal DCT of the mel bands in dB, nothing further than 80 dB under
    the largest of the clip. The frames are 2048 samples long under a Hann
    window, 512 apart, and there are 128 bands from 0 Hz to half the rate;
    *n_fft*, *hop_length*, *n_mels* and *fmax* are for others. These are librosa's
    defaults, and the coefficients are its coefficients.

    At 192 kHz the defaults leave 3 of the 128 bands without a single FFT bin, and
    a warning says so; an *n_fft* of 8192, or 64 bands, leaves none empty.

    :raises FeatureError: for an *n_mfcc* above the number of mel bands, which is
        all the coefficients a DCT of them has; for samples that are not finite;
        and for a size that is not a whole number of 1 or more.
    """
    sr = sample_rate(sr, "mfcc", FeatureError)
    y = _clip(y, "mfcc")
    return _mfcc(y, sr, n_mfcc, "mfcc", n_fft, hop_length, n_mels, fmax)[np.newaxis]


def _mfcc(y: np.ndarray, sr: float, n_mfcc, where: str, n_fft=None, hop_length=None,
          n_mels=None, fmax=None) -> np.ndarray:
    """``(frames, n_mfcc)``."""
    n_mfcc = count(n_mfcc, 'n_mfcc', where, FeatureError)
    n_mels = 128 if n_mels is None else count(n_mels, 'n_mels', where, FeatureError)
    n_fft = 2048 if n_fft is None else _samples(n_fft, 'n_fft', where)
    hop_length = 512 if hop_length is None else _samples(hop_length, 'hop_length', where)
    if fmax is not None:
        top = real(fmax)
        if top is None or top <= 0:
            raise FeatureError(f"{where}(): fmax is a frequency in hertz above 0, got {fmax!r}")
        fmax = top
    if n_mfcc > n_mels:
        raise FeatureError(f"{where}(): n_mfcc={n_mfcc} asks for more coefficients than "
                           f"{n_mels} mel bands have; raise n_mels or lower n_mfcc")
    bands = _mel.power(y, sr, n_fft, hop_length, n_mels, fmax=fmax, who=where)
    return _mel.cepstrum(_mel.decibels(bands, range_db=80.0), n_mfcc).T


def mel_band_hz(n_mels: int, sr: float, *, fmin: float = 0.0,
                fmax: Optional[float] = None) -> np.ndarray:
    """The frequency each mel band is centred on, in hertz: ``n_mels`` of them,
    lowest first, what to label the axis of :func:`melspectrogram` with, or of
    :class:`SpectrogramStream` in mel.

    The bands are evenly spaced on Slaney's mel scale, which is linear up to 1 kHz
    and logarithmic above. Only the centres are given: the two feet, *fmin* under
    the lowest band and *fmax* over the highest, are not among them.

    >>> kudio.mel_band_hz(4, 16000).round(1)
    array([ 603.3, 1237.4, 2305.2, 4294.3])

    :param fmin: where the lowest band starts. :func:`mfcc` and
        :func:`melspectrogram` start at 0.
    :param fmax: where the highest one ends, half of *sr* when omitted.
    :raises FeatureError: for an *n_mels* that is not a whole number of 1 or more,
        and for limits that are not ``0 <= fmin < fmax``.
    """
    sr = sample_rate(sr, "mel_band_hz", FeatureError)
    n_mels = count(n_mels, 'n_mels', "mel_band_hz", FeatureError)
    return _mel.centres(n_mels, *_limits(fmin, fmax, sr, "mel_band_hz"))


def _limits(fmin, fmax, sr: float, where: str) -> Tuple[float, float]:
    """The range of mel bands, *fmax* defaulting to half the rate."""
    low, high = real(fmin), sr / 2.0 if fmax is None else real(fmax)
    if low is None or high is None or not 0 <= low < high:
        raise FeatureError(f"{where}(): need 0 <= fmin < fmax, in hertz, got fmin={fmin!r} and "
                           f"fmax={high if fmax is None else fmax!r}")
    return low, high


def mfcc_from_files(wav_list: Sequence, n_mfcc: int = 40,
                    desired_samples: Optional[int] = None,
                    desired_length: Optional[int] = None) -> np.ndarray:
    """Load each wave file and stack their MFCC matrices ``(n_files, frames, n_mfcc)``.

    :param desired_samples: cut each file to this many samples.
    :param desired_length: cut each file to this many **seconds** (the
        spectrogram functions' ``desired_length`` counts samples).
    :raises FeatureError: for an empty list, files that do not give the same
        number of frames, samples that are not finite, and a length that is
        not a number above 0.
    """
    where = "mfcc_from_files"
    keep = _keep(desired_samples, 'desired_samples', where)
    seconds = None if desired_length is None else real(desired_length)
    if desired_length is not None and (seconds is None or seconds <= 0):
        raise FeatureError(f"{where}(): desired_length is a number of seconds to keep, above 0, "
                           f"or None for all of them, got {desired_length!r}")
    matrices = []
    for wav_dir in tqdm(wav_list, desc="[kudio] MFCC"):
        y, sr = file_load(wav_dir)
        if seconds is not None:
            y = y[:int(round(sr * seconds))]
        matrices.append(_mfcc(finite(y[:keep], where, FeatureError), sr, n_mfcc, where))
        _one_shape(matrices, wav_dir, where)
    if not matrices:
        raise FeatureError(f"{where}() got no files")
    return np.stack(matrices)


def _one_shape(matrices: list, file, where: str) -> None:
    """Say which *file* does not fit the ones before it, as soon as it does not."""
    first, last = matrices[0], matrices[-1]
    if last.shape != first.shape:
        raise FeatureError(f"{where}(): {file} gives {len(last)} frames and the first file "
                           f"{len(first)}, and they are stacked; cut every file to one length")


def logspec_from_files(wav_list: Sequence,
                       sequence: bool,
                       forward_backward: int,
                       norm: bool,
                       hop_length: int = 256,
                       n_fft: int = 512,
                       win_length: Optional[int] = None,
                       desired_samples: Optional[int] = None,
                       window: str = 'hamming') -> np.ndarray:
    """Convert each file to a log spectrogram and stack them.

    :raises FeatureError: for an empty list, and what
        :func:`file_to_spectrogram` raises.
    """
    sequence = flag(sequence, 'sequence', 'logspec_from_files', FeatureError)
    norm = flag(norm, 'norm', 'logspec_from_files', FeatureError)
    spectrograms = [
        file_to_spectrogram(file=file, sequence=sequence,
                            forward_backward=forward_backward, norm=norm,
                            hop_length=hop_length, n_fft=n_fft,
                            win_length=win_length, desired_length=desired_samples,
                            window=window)
        for file in tqdm(wav_list, desc='[kudio] log-spectrograms')
    ]
    if not spectrograms:
        raise FeatureError("logspec_from_files() got no files")
    return np.vstack(spectrograms)


def melspectrogram(source: Union[PathLike, np.ndarray], n_mels: int = 64,
                   n_frames: int = 5,
                   n_fft: int = 1024, hop_length: int = 512,
                   power: float = 2.0, *,
                   sr: Optional[int] = None) -> np.ndarray:
    """Audio -> stacked log-mel feature vectors ``(n_vectors, n_mels * n_frames)``.

    *source* is a file path **or** a waveform; a waveform requires *sr*, the
    same way :func:`mfcc` does. With a path, *sr* resamples the file to that
    rate, and leaving it out keeps the file's own.

    The window is Hann, not the Hamming of :class:`STFT` and
    :func:`waveform_to_spectrogram`. :func:`mel_band_hz` gives the frequency
    each band is centred on.

    >>> feats = kudio.melspectrogram('clip.wav')            # doctest: +SKIP
    >>> feats = kudio.melspectrogram(y, sr=16000)           # doctest: +SKIP

    :param power: the exponent of the magnitude spectrogram the bands are taken
        of: 2 for power, 1 for magnitude. The features are in dB either way.
    :raises FeatureError: for anything but one channel of finite samples, for a
        size that is not a whole number of 1 or more, and for a *power* that is
        not above 0.
    """
    where = "melspectrogram"
    n_mels = count(n_mels, 'n_mels', where, FeatureError)
    n_frames = count(n_frames, 'n_frames', where, FeatureError)
    n_fft = _samples(n_fft, 'n_fft', where)
    hop_length = _samples(hop_length, 'hop_length', where)
    exponent = real(power)
    if exponent is None or exponent <= 0:
        raise FeatureError(f"{where}(): power is an exponent above 0, got {power!r}")
    dims = n_mels * n_frames
    if isinstance(source, (bytes, bytearray)):
        raise FeatureError(f"{where}() takes a path as text, or a waveform, and got bytes")
    if isinstance(source, (str, os.PathLike)):
        # sr used to be overwritten by the file's own rate without a word
        y, sr = file_load(source, sr=sr, mono=True)
        y = finite(y, where, FeatureError)
    else:
        if sr is None:
            raise FeatureError("melspectrogram() needs sr= when given a waveform")
        sr = sample_rate(sr, where, FeatureError)
        y = np.asarray(_clip(source, where), dtype=np.float32)
    mel = _mel.power(y, sr, n_fft, hop_length, n_mels, exponent, who=where)
    log_mel = 20.0 / exponent * np.log10(np.maximum(mel, sys.float_info.epsilon))
    n_vectors = log_mel.shape[1] - n_frames + 1
    if n_vectors < 1:
        return np.empty((0, dims), dtype=np.float32)
    vectors = np.zeros((n_vectors, dims), dtype=np.float32)
    for t in range(n_frames):
        vectors[:, n_mels * t: n_mels * (t + 1)] = log_mel[:, t: t + n_vectors].T
    return vectors


def denoise_wav2spec(waveform: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Peak-normalized magnitude spectrogram + phase, for denoising models.

    The two come back in different layouts: the spectrogram as
    ``(1, frames, bins)`` and the phase as ``(bins, frames)``, the way
    :func:`denoise_spec2wav` takes them back.

    :raises FeatureError: for a silent clip, which has no peak to normalise by.
    """
    waveform = _clip(waveform, "denoise_wav2spec")
    peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
    if not peak > 0:
        raise FeatureError("denoise_wav2spec() got a silent clip; there is no "
                           "peak to normalise by")
    D = _spectral.stft(waveform / peak, 512, 256, 512, 'hann', "denoise_wav2spec")
    D += np.finfo(float).eps
    Sxx = np.abs(D)
    phase = np.angle(D)
    Sxx_mean = np.mean(Sxx, axis=1, keepdims=True)
    Sxx_std = np.std(Sxx, axis=1, keepdims=True) + 1e-12
    Sxx_r = (Sxx - Sxx_mean) / Sxx_std
    return np.reshape(Sxx_r.T, (1, Sxx_r.shape[1], Sxx_r.shape[0])), phase


def denoise_spec2wav(spec: np.ndarray, phase: np.ndarray) -> np.ndarray:
    """Inverse of :func:`denoise_wav2spec`; output peak-normalized.

    :param spec: ``(frames, bins)``, with or without the axis of 1 in front.
    :param phase: ``(bins, frames)``.
    :raises FeatureError: for two that do not go together, or anything but finite
        real numbers.
    """
    where = "denoise_spec2wav"
    spec, phase = _numbers(spec, where), _numbers(phase, where)
    if phase.ndim != 2 or spec.shape not in (phase.shape[::-1], (1,) + phase.shape[::-1]):
        raise FeatureError(f"{where}() takes a spectrogram (1, frames, bins) or (frames, bins) and "
                           f"its phase (bins, frames), got shapes {spec.shape} and {phase.shape}")
    spec = finite(spec, where, FeatureError).reshape(phase.shape[::-1]).T
    pred_wav = _spectral.istft(np.multiply(spec, np.exp(1j * finite(phase, where, FeatureError))),
                               512, 256, 512, 'hann', who=where)
    peak = float(np.max(np.abs(pred_wav))) if pred_wav.size else 0.0
    return pred_wav / peak if peak > 0 else pred_wav


def _numbers(values, where: str) -> np.ndarray:
    """*values* as an array of real numbers."""
    values = np.asarray(values)
    if values.dtype.kind not in 'fiu':
        raise FeatureError(f"{where}() takes real numbers, got {values.dtype}")
    return values


def _matrix(frames, where: str) -> np.ndarray:
    """*frames* as float32: booleans, integers and floats. Complex numbers would lose
    their imaginary part, and objects and text are no numbers until something says
    which."""
    try:
        frames = np.asarray(frames)
    except (TypeError, ValueError) as e:              # rows of different lengths
        raise FeatureError(f"{where}() takes a matrix of numbers: {e}") from None
    if frames.dtype.kind not in 'biuf':
        raise FeatureError(f"{where}() takes real numbers, got {frames.dtype}")
    return np.asarray(frames, dtype=np.float32)


def _context(context) -> int:
    number = whole(context)
    if number is None or number < 0:
        raise FeatureError(f"context must be >= 0, a whole number, got {context!r}")
    return number


def stack_context(frames: np.ndarray, context: int,
                  pad: str = 'edge') -> np.ndarray:
    """Stack ``±context`` neighbouring frames onto each frame.

    ``(frames, dim)`` -> ``(frames, dim * (2 * context + 1))``, one output row
    per input row: the edges are padded rather than dropped, so a prediction
    made from the result lines up 1:1 with the input.

    Works on *any* feature matrix — MFCC, mel, log-spectrogram — unlike the
    ``forward_backward`` argument of :func:`waveform_to_spectrogram`, which
    only reaches the spectrogram it computes itself.

    :param pad: ``'edge'`` repeats the first/last frame; ``'zero'`` pads with
        zeros, which is what ``forward_backward`` uses. Zero is not silence: in
        log10 power it is a power of 1, and after a :class:`Standardizer` it is
        the mean frame. ``'edge'`` avoids inventing a step at the boundary.

    >>> stack_context(np.zeros((10, 40)), context=2).shape
    (10, 200)

    No frames are ``(0, dim * (2 * context + 1))``, as :class:`ContextFrames`
    gives them.

    :raises FeatureError: for anything but a matrix of real numbers, and a
        *context* that is not a whole number of 0 or more.
    """
    frames = _matrix(frames, 'stack_context')
    if frames.ndim != 2:
        raise FeatureError(f"frames must be 2-D, got shape {frames.shape}")
    context = _context(context)
    if context == 0:
        return frames.copy()                   # never the caller's own array
    if pad not in ('edge', 'zero'):
        raise FeatureError(f"pad must be 'edge' or 'zero', got {pad!r}")

    width = 2 * context + 1
    n, dim = frames.shape
    if not n:                                           # no edge to repeat
        return np.empty((0, dim * width), dtype=np.float32)
    mode = 'edge' if pad == 'edge' else 'constant'
    padded = np.pad(frames, ((context, context), (0, 0)), mode=mode)
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
    it index for index. A single ``(frames, dim)`` matrix is one utterance; a
    3-D array is read as a stack of them.
    """

    def __init__(self, utterances: Union[np.ndarray, Sequence[np.ndarray]],
                 context: int, pad: str = 'edge'):
        context = _context(context)
        if pad not in ('edge', 'zero'):
            raise FeatureError(f"pad must be 'edge' or 'zero', got {pad!r}")
        if isinstance(utterances, np.ndarray) and utterances.ndim == 2:
            utterances = [utterances]
        mats = [_matrix(u, 'ContextFrames') for u in utterances]
        if not mats:
            raise FeatureError("ContextFrames needs at least one utterance")
        if any(m.ndim != 2 for m in mats) or len({m.shape[1] for m in mats}) != 1:
            raise FeatureError("every utterance must be (frames, dim) with one "
                               f"dim, got {sorted({m.shape for m in mats})}")
        self.context = context
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
        """``(rows, dim * (2 * context + 1))``, as ``np.vstack`` would give."""
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
    frames = _matrix(frames, 'frame_windows')
    if frames.ndim != 2:
        raise FeatureError(f"frames must be 2-D, got shape {frames.shape}")
    given, n_frames = n_frames, whole(n_frames)
    if n_frames is None or n_frames < 1:
        raise FeatureError(f"n_frames must be >= 1, a whole number, got {given!r}")
    if pad not in ('edge', 'zero'):
        raise FeatureError(f"pad must be 'edge' or 'zero', got {pad!r}")

    n, dim = frames.shape
    if n == 0:
        raise FeatureError("there are no frames to cut into windows; a clip "
                           "too short for its features gives an empty matrix")
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
        """Learn each column's mean and standard deviation from *frames*
        ``(rows, dim)`` -- the training set, never the data it is applied to.

        :raises FeatureError: for anything but a matrix of finite numbers with a
            row or more. The statistics are as they were.
        """
        frames = self._numbers(frames, 'fit')
        if frames.ndim != 2:
            raise FeatureError(f"frames must be 2-D, got shape {frames.shape}")
        if not frames.size:
            raise FeatureError(f"Standardizer.fit() got no frames to learn from, shape "
                               f"{frames.shape}")
        bad = int(np.count_nonzero(~np.isfinite(frames)))
        if bad:
            raise FeatureError(f"Standardizer.fit() got {bad} number(s) that are not finite (NaN "
                               f"or inf), and one is the mean of its whole column")
        self.mean = frames.mean(axis=0)
        self.std = frames.std(axis=0) + 1e-8
        return self

    def transform(self, frames: np.ndarray) -> np.ndarray:
        """``(frames - mean) / std``, column by column, as float32. *frames* are
        ``(rows, dim)``, or anything else whose last axis is *dim* long: one
        frame, or a batch of sequences.

        :raises FeatureError: for frames of another width.
        """
        frames = self._columns(frames, 'transform')
        return ((frames - self.mean) / self.std).astype(np.float32)

    def inverse(self, frames: np.ndarray) -> np.ndarray:
        """Undo :meth:`transform` — used to map a prediction back."""
        frames = self._columns(frames, 'inverse')
        return (frames * self.std + self.mean).astype(np.float32)

    @staticmethod
    def _numbers(frames, where: str) -> np.ndarray:
        return _matrix(frames, f'Standardizer.{where}')

    def _columns(self, frames, where: str) -> np.ndarray:
        # one of another width would be spread over the columns, or they over it
        self._check()
        frames = self._numbers(frames, where)
        fitted = np.shape(self.mean)
        if fitted and (not frames.ndim or frames.shape[-1] != fitted[-1]):
            raise FeatureError(f"Standardizer.{where}() was fitted on frames {fitted[-1]} wide, "
                               f"and got shape {frames.shape}")
        return frames

    def fit_transform(self, frames: np.ndarray) -> np.ndarray:
        """:meth:`fit` on *frames*, then :meth:`transform` them."""
        return self.fit(frames).transform(frames)

    def _check(self) -> None:
        if not self.fitted:
            raise FeatureError("Standardizer has not been fitted")

    def save(self, path: PathLike) -> Path:
        """Write the statistics to an ``.npz`` file and return its path.

        numpy adds ``.npz`` to a name without it, so the path returned is the
        file actually written: ``save('stats')`` returns ``stats.npz``.
        """
        self._check()
        path = Path(path)
        if path.suffix != '.npz':
            path = path.parent / (path.name + '.npz')
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(path), mean=self.mean, std=self.std)
        return path

    @classmethod
    def load(cls, path: PathLike) -> "Standardizer":
        """Read statistics :meth:`save` wrote. ``'stats'`` finds ``stats.npz``."""
        path = Path(path)
        if not path.exists() and path.suffix != '.npz':
            path = path.parent / (path.name + '.npz')
        with np.load(str(path)) as data:
            return cls(mean=data['mean'], std=data['std'])


def wave_slicing(y: np.ndarray, frame_length: int = 2048,
                 hop_length: int = 512, is_reshape: bool = True) -> np.ndarray:
    """Slice a mono waveform into overlapping frames: one after the other in one
    vector, or ``(frame_length, frames)`` without *is_reshape*, a read-only view
    of *y*. Samples past the last whole frame are left out.

    :raises FeatureError: for a clip shorter than a frame, and for a size that is
        not a whole number of 1 or more.
    """
    is_reshape = flag(is_reshape, 'is_reshape', 'wave_slicing', FeatureError)
    frame_length = _samples(frame_length, 'frame_length', "wave_slicing")
    hop_length = _samples(hop_length, 'hop_length', "wave_slicing")
    y = channels_last(y, "wave_slicing", FeatureError)
    y = _one_axis(y[:, 0] if y.ndim == 2 and y.shape[1] == 1 else y, "wave_slicing")
    if len(y) < frame_length:
        raise FeatureError(f"wave_slicing(): the clip has {len(y)} sample(s), fewer than one "
                           f"frame of {frame_length}")
    frames = np.lib.stride_tricks.sliding_window_view(y, frame_length)[::hop_length]
    return frames.reshape(-1) if is_reshape else frames.T


def features2matrix(features: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Concatenate feature matrices ``(rows, dim)``, and label each row with the
    place of its matrix in the list.

    :raises FeatureError: for an empty list, and for anything in it that is not a
        matrix as wide as the first.
    """
    matrices = [np.asarray(f) for f in features]
    if not matrices:
        raise FeatureError("features2matrix() joins matrices, and there are none")
    for i, m in enumerate(matrices):
        if m.ndim != 2:
            raise FeatureError(f"features2matrix() joins matrices (rows, dim), and item {i} has "
                               f"shape {m.shape}")
        if m.shape[1] != matrices[0].shape[1]:
            raise FeatureError(f"features2matrix(): item {i} has {m.shape[1]} columns and item 0 "
                               f"has {matrices[0].shape[1]}")
    labels = np.concatenate([i * np.ones(len(m)) for i, m in enumerate(matrices)])
    return np.vstack(matrices), labels


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
