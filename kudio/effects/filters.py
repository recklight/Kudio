# -*- coding: utf-8 -*-
"""Butterworth band filters.

The everyday shaping moves — drop the rumble below 80 Hz, throw away everything
above the band you actually care about, isolate one. `wavelet_low_pass_filter`
in :mod:`kudio.enhance` is a denoising algorithm that happens to be low-pass;
this is the plain filter, and the two are not substitutes.

Filtering is **zero-phase by default** (`scipy.signal.sosfiltfilt`): the signal
is run forwards and backwards, so nothing is shifted in time. That matters when
a filtered copy has to line up sample-for-sample with the original — an A/B, a
spectrogram overlay, a training target. Pass ``zero_phase=False`` for the
causal, real-time-shaped version.
"""
from __future__ import annotations

from typing import Sequence, Tuple, Union

import numpy as np
from scipy.signal import butter, sosfilt, sosfiltfilt

from kudio.exceptions import FeatureError

__all__ = ['highpass', 'lowpass', 'bandpass', 'bandstop', 'band_filter']

_BTYPES = ('highpass', 'lowpass', 'bandpass', 'bandstop')


def _check_cutoff(cutoff: Union[float, Sequence[float]], sr: int,
                  btype: str) -> Union[float, Tuple[float, float]]:
    nyquist = sr / 2.0
    if btype in ('bandpass', 'bandstop'):
        try:
            low, high = (float(c) for c in cutoff)  # type: ignore[misc]
        except (TypeError, ValueError):
            raise FeatureError(
                f"{btype} needs two cutoffs, got {cutoff!r}") from None
        if not 0.0 < low < high < nyquist:
            raise FeatureError(
                f"{btype} needs 0 < low < high < Nyquist ({nyquist:.1f} Hz), "
                f"got {low} and {high}")
        return low, high

    value = float(cutoff)  # type: ignore[arg-type]
    if not 0.0 < value < nyquist:
        raise FeatureError(
            f"cutoff must be between 0 and Nyquist ({nyquist:.1f} Hz) for "
            f"sr={sr}, got {value}")
    return value


def band_filter(y: np.ndarray, sr: int, cutoff, btype: str = 'highpass',
                order: int = 4, zero_phase: bool = True) -> np.ndarray:
    """Butterworth filter of any of the four kinds.

    :param cutoff: one frequency in Hz, or ``(low, high)`` for band types.
    :param order: filter order. Zero-phase filtering applies it twice, so the
        effective roll-off is ``2 * order``.
    :param zero_phase: forwards-and-backwards (no delay) when True.
    """
    if btype not in _BTYPES:
        raise FeatureError(f"btype must be one of {_BTYPES}, got {btype!r}")
    if order < 1:
        raise FeatureError(f"order must be >= 1, got {order}")

    wn = _check_cutoff(cutoff, sr, btype)
    y = np.asarray(y)
    if y.size == 0:
        return y.copy()

    sos = butter(order, wn, btype=btype, fs=sr, output='sos')
    if not zero_phase:
        return np.asarray(sosfilt(sos, y), dtype=y.dtype)

    # sosfiltfilt reflects the signal at both ends by 3 * n_sections * 2
    # samples; a clip shorter than that cannot be padded and scipy raises.
    # Shrinking the pad is better than refusing to filter a short selection.
    padlen = min(3 * (2 * len(sos) + 1), y.shape[0] - 1)
    if padlen < 1:
        raise FeatureError(
            f"clip is too short to filter ({y.shape[0]} samples)")
    return np.asarray(sosfiltfilt(sos, y, padlen=padlen), dtype=y.dtype)


def highpass(y: np.ndarray, sr: int, cutoff: float = 80.0, order: int = 4,
             zero_phase: bool = True) -> np.ndarray:
    """Remove everything below *cutoff* Hz — rumble, handling noise, DC drift.

    >>> clean = kudio.highpass(y, sr, cutoff=80)
    """
    return band_filter(y, sr, cutoff, 'highpass', order, zero_phase)


def lowpass(y: np.ndarray, sr: int, cutoff: float = 8000.0, order: int = 4,
            zero_phase: bool = True) -> np.ndarray:
    """Remove everything above *cutoff* Hz — hiss, and anything past the band
    a downstream model was trained on."""
    return band_filter(y, sr, cutoff, 'lowpass', order, zero_phase)


def bandpass(y: np.ndarray, sr: int, low: float = 300.0, high: float = 3400.0,
             order: int = 4, zero_phase: bool = True) -> np.ndarray:
    """Keep only ``low..high`` Hz. The default is the telephone band."""
    return band_filter(y, sr, (low, high), 'bandpass', order, zero_phase)


def bandstop(y: np.ndarray, sr: int, low: float = 45.0, high: float = 55.0,
             order: int = 4, zero_phase: bool = True) -> np.ndarray:
    """Remove ``low..high`` Hz. The default straddles 50 Hz mains hum; use
    ``(55, 65)`` where the mains run at 60."""
    return band_filter(y, sr, (low, high), 'bandstop', order, zero_phase)
