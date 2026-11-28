# -*- coding: utf-8 -*-
"""Mel bands, decibels and the cepstrum of them.

Private: :func:`kudio.melspectrogram`, :func:`kudio.mfcc` and
:func:`kudio.mel_band_hz` are what to call.

The scale is Slaney's [S98]_: 200/3 Hz a mel up to 1 kHz, 15 mel, and 27 mel to a
factor of 6.4 above it. A band is a triangle over the FFT bins from the centre of
the band below to the centre of the one above, as high as makes its area 1, so a
band's value is a mean over its width and not a sum that grows with it. librosa's
defaults are these and its filters come out the same to the bit, rounded to
float32 once as triangles and once more when they are scaled
(``tests/test_against_librosa.py``). HTK's scale and the other scalings librosa
offers are not here: nothing in kudio asks for them.

:mod:`kudio.core.dnsmos` keeps filters of its own, on the grid of the librosa its
model was made with.

References
----------
.. [S98] M. Slaney, "Auditory Toolbox, version 2", Technical Report 1998-010,
   Interval Research Corporation, 1998.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from kudio.core import _spectral
from kudio.core._policy import warn

__all__ = ['centres', 'filters', 'power', 'decibels', 'cepstrum']

#: hertz a mel below the break, the break, and the natural logarithm of the
#: factor a mel is above it
_LINEAR = 200.0 / 3
_BREAK_HZ = 1000.0
_BREAK_MEL = _BREAK_HZ / _LINEAR
_STEP = np.log(6.4) / 27.0


def _to_mel(hz: float) -> float:
    return hz / _LINEAR if hz < _BREAK_HZ else _BREAK_MEL + np.log(hz / _BREAK_HZ) / _STEP


def _to_hz(mel: np.ndarray) -> np.ndarray:
    hz = _LINEAR * mel
    above = mel >= _BREAK_MEL
    hz[above] = _BREAK_HZ * np.exp(_STEP * (mel[above] - _BREAK_MEL))
    return hz


def _edges(n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    """``n_mels + 2`` points evenly spaced in mel: the foot of the lowest band,
    the peak of each, the foot of the highest."""
    return _to_hz(np.linspace(_to_mel(fmin), _to_mel(fmax), n_mels + 2))


def centres(n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    """Where each band peaks, in hertz."""
    return _edges(n_mels, fmin, fmax)[1:-1]


def filters(sr: float, n_fft: int, n_mels: int, fmin: float = 0.0,
            fmax: Optional[float] = None, who: str = 'filters') -> np.ndarray:
    """``(n_mels, n_fft // 2 + 1)`` in float32. A band narrower than the bins are
    apart can have none of them in it, and a warning counts those; so does one
    above half the rate."""
    top = float(sr) / 2
    fmax = top if fmax is None else fmax
    edges = _edges(n_mels, fmin, fmax)
    bins = np.fft.rfftfreq(n=n_fft, d=1.0 / sr)
    out = np.zeros((n_mels, len(bins)), dtype=np.float32)
    for band in range(n_mels):
        low, peak, high = edges[band:band + 3]
        # a band of no width, where fmin and fmax are a hair apart, is empty
        with np.errstate(divide='ignore', invalid='ignore'):
            rising, falling = (bins - low) / (peak - low), (high - bins) / (high - peak)
        out[band] = np.maximum(0, np.minimum(rising, falling))
    out *= (2.0 / (edges[2:] - edges[:-2]))[:, np.newaxis]
    empty = int(np.count_nonzero(~(out.max(axis=1) > 0)))
    if empty:
        way_out = "raise n_fft or lower n_mels"
        if fmax > top:
            way_out += f", or hold fmax={fmax:g} Hz to half the rate, {top:g} Hz"
        warn(f"{who}(): {empty} of {n_mels} mel bands hold no FFT bin and stay empty; {way_out}")
    return out


def power(y: np.ndarray, sr: float, n_fft: int, hop: int, n_mels: int,
          exponent: float = 2.0, fmin: float = 0.0, fmax: Optional[float] = None,
          who: str = 'power') -> np.ndarray:
    """``(n_mels, frames)``: the magnitude spectrogram under a Hann window, to the
    *exponent*, in mel bands."""
    spectrum = np.abs(_spectral.stft(y, n_fft, hop, n_fft, 'hann', who))
    spectrum **= exponent
    # the product of two matrices, which einsum hands to BLAS the way round it
    # has always been done in: another would be rounded another way
    return np.einsum('ft,mf->mt', spectrum, filters(sr, n_fft, n_mels, fmin, fmax, who),
                     optimize=True)


def decibels(values: np.ndarray, ref: float = 1.0, floor: float = 1e-10,
             range_db: Optional[float] = None) -> np.ndarray:
    """Power in dB over *ref*, nothing under *floor* of it counted as less, and
    with *range_db* nothing further down than that from the largest."""
    out = 10.0 * np.log10(np.maximum(floor, values))
    # in the type of the values, and an array: numpy below 2 works out a float32
    # number beside a Python one in float64, and then the largest is not 0 dB
    ref = np.atleast_1d(np.asarray(ref, dtype=values.dtype))
    out -= 10.0 * np.log10(np.maximum(floor, ref))
    if range_db is not None:
        np.maximum(out, out.max(keepdims=True) - range_db, out=out)
    return out


def cepstrum(bands_db: np.ndarray, n: int) -> np.ndarray:
    """The first *n* coefficients of the orthonormal DCT-II of each column."""
    from scipy.fft import dct
    return dct(bands_db, axis=-2, type=2, norm='ortho')[:n]
