# -*- coding: utf-8 -*-
"""The short-time Fourier transform and its inverse, on numpy and scipy.

Private: :class:`kudio.STFT` and :func:`kudio.waveform_to_spectrogram` are what
to call. Both work on ``(bins, frames)``, the layout a transform is written in;
the callers turn it over.

Frame *k* is the ``n_fft`` samples centred on sample ``k * hop`` of the clip,
zeros past its ends, under the window, which is centred in the frame where it is
shorter. The inverse is Griffin and Lim's [GL84]_ least-squares estimate, eq. 6:
the windowed inverse transforms added up, over the squared windows added up.

Neither is new, and how they round is not either. The spectrum of float32 audio
is transformed in float64 and kept in complex64, the way back goes through a
float32 inverse transform and is added up in float32, frame after frame. librosa
does it so, and the spectrograms models were trained on came out of it; in
``tests/test_against_librosa.py`` the two give the same bits.

References
----------
.. [GL84] D. W. Griffin and J. S. Lim, "Signal estimation from modified
   short-time Fourier transform", IEEE Trans. Acoustics, Speech, and Signal
   Processing, vol. 32, no. 2, pp. 236-243, 1984,
   doi:10.1109/TASSP.1984.1164317.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from kudio.core._policy import warn
from kudio.exceptions import FeatureError

__all__ = ['window', 'stft', 'istft']

#: about how many bytes of float64 frames are transformed at a time. A minute of
#: 48 kHz at a 128-sample hop is 92 MB of them. Of 32 KB to 1 MB this was among the
#: fastest wherever it was called from; half a megabyte was as fast in one place and
#: took twice as long in another, and 1 MB twice as long in all of them.
_BLOCK = 1 << 18


def window(spec: Any, size: int, who: str) -> np.ndarray:
    """The *size* values of the window *spec*: anything
    :func:`scipy.signal.get_window` takes -- a name, a name with its parameter in
    a tuple, a Kaiser beta -- or the values themselves, or a function that gives
    them for a size. A named one is periodic.

    :param who: the function a refusal starts with.
    """
    if callable(spec):
        values = np.asarray(spec(size))
    elif isinstance(spec, (list, np.ndarray)):
        values = np.asarray(spec)
    else:
        from scipy.signal import get_window
        try:
            values = get_window(spec, size, fftbins=True)
        except (ValueError, TypeError, AttributeError) as e:
            raise FeatureError(f"{who}(): {spec!r} is not a window scipy knows: {e}") from e
    if values.ndim != 1:
        raise FeatureError(f"{who}(): the window has shape {values.shape}, not ({size},)")
    if len(values) != size:
        raise FeatureError(f"{who}(): the window has {len(values)} values, and {size} are needed")
    return values


def _centred(values: np.ndarray, n_fft: int) -> np.ndarray:
    """A window in the middle of a frame of *n_fft*, zeros around it."""
    before = (n_fft - len(values)) // 2
    return np.pad(values, (before, n_fft - len(values) - before))


def stft(y: np.ndarray, n_fft: int, hop: int, win_length: int, spec: Any,
         who: str = 'stft') -> np.ndarray:
    """*y*, 1-D and floating point, as ``(n_fft // 2 + 1, frames)``: complex128
    of float64 audio and complex64 of any other."""
    from scipy.fft import rfft
    shape = _centred(window(spec, win_length, who), n_fft)
    if len(y) < n_fft:
        if not len(y) and n_fft % 2:
            raise FeatureError(f"{who}(): the clip has no samples, and an odd frame of {n_fft} "
                               f"does not fit the {n_fft - 1} zeros around it")
        warn(f"{who}(): the clip has {len(y)} sample(s), fewer than one frame of {n_fft}; what "
             f"is transformed is mostly the zeros around it")
    padded = np.pad(y, n_fft // 2)
    frames = np.lib.stride_tricks.sliding_window_view(padded, n_fft)[::hop]
    # a column at a time in memory: a frame is written and read as one piece
    out = np.empty((n_fft // 2 + 1, len(frames)), order='F',
                   dtype=np.complex128 if y.dtype == np.float64 else np.complex64)
    step = max(1, _BLOCK // (8 * n_fft))
    for start in range(0, len(frames), step):
        part = slice(start, start + step)
        out[:, part] = rfft(shape * frames[part], axis=-1).T
    return out


def istft(spectrum: np.ndarray, n_fft: int, hop: int, win_length: int, spec: Any,
          length: Optional[int] = None, who: str = 'istft') -> np.ndarray:
    """``(n_fft // 2 + 1, frames)`` as the clip it is nearest to, *length* samples
    of it or as many as the frames cover: float64 of complex128, float32 of
    complex64."""
    from scipy.fft import irfft
    real = np.float64 if spectrum.dtype == np.complex128 else np.float32
    shape = _centred(window(spec, win_length, who), n_fft)
    before = n_fft // 2
    n_frames = spectrum.shape[1]
    if length is None:
        length = n_fft + hop * (n_frames - 1) - 2 * before
    else:
        n_frames = min(n_frames, -(-(length + 2 * before) // hop))

    # A frame is cut into pieces a hop long, and the clip is built up as rows of
    # a hop: piece k of every frame lands k rows below the frame's own, so the
    # frames are added a piece at a time instead of one by one. The last piece
    # first, which is a row's oldest frame: the sums are in float32, and the
    # order they are made in is part of the answer.
    pieces = -(-n_fft // hop)
    rows = np.zeros((n_frames + pieces - 1, hop), real)
    energy = np.zeros_like(rows)
    squared = np.zeros(pieces * hop)
    squared[:n_fft] = shape ** 2
    squared = squared.reshape(pieces, hop)
    step = max(1, _BLOCK // (8 * pieces * hop))
    for start in range(0, n_frames, step):
        stop = min(n_frames, start + step)
        frames = np.zeros((stop - start, pieces * hop))
        frames[:, :n_fft] = shape * irfft(spectrum[:, start:stop].T, n=n_fft, axis=-1)
        frames = frames.reshape(stop - start, pieces, hop)
        for k in range(pieces - 1, -1, -1):
            rows[start + k:stop + k] += frames[:, k]
            energy[start + k:stop + k] += squared[k]

    out, energy = (a.reshape(-1)[before:before + length] for a in (rows, energy))
    np.divide(out, energy, out=out, where=energy > np.finfo(real).tiny)
    return np.pad(out, (0, length - len(out))) if len(out) < length else out
