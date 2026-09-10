# -*- coding: utf-8 -*-
"""Everyday editing moves: fades, reversal, DC removal.

None of these are clever. They are here because every audio editor has them and
because doing them by hand is where the off-by-one lives — a fade that reaches
zero one sample early leaves a click exactly where the fade was meant to
prevent one.
"""
from __future__ import annotations

import numpy as np

from kudio.exceptions import FeatureError

__all__ = ['fade_in', 'fade_out', 'fade', 'reverse', 'remove_dc',
           'crossfade', 'splice']

SHAPES = ('linear', 'cosine', 'exponential')
_EXP_FLOOR_DB = -60.0
#: Enough to remove the step at a cut, short enough that nothing is heard to
#: happen. Below ~2 ms the click starts coming back.
DEFAULT_CROSSFADE = 0.005


def _ramp(n: int, shape: str) -> np.ndarray:
    """A rising 0 → 1 curve of *n* samples, inclusive at both ends."""
    if shape not in SHAPES:
        raise FeatureError(f"shape must be one of {SHAPES}, got {shape!r}")
    if n <= 1:
        return np.ones(max(n, 0), dtype=np.float64)

    t = np.linspace(0.0, 1.0, n)
    if shape == 'linear':
        return t
    if shape == 'cosine':
        # raised cosine: flat at both ends, so the fade neither starts nor
        # stops abruptly -- the shape a crossfade wants
        return 0.5 - 0.5 * np.cos(np.pi * t)
    # exponential: linear in decibels, which is what "fade out" sounds like.
    # Rescaled so it still reaches exactly 0 and 1 rather than the -60 dB floor.
    curve = 10.0 ** (_EXP_FLOOR_DB * (1.0 - t) / 20.0)
    return (curve - curve[0]) / (1.0 - curve[0])


def _fade_samples(n_samples: int, duration: float, sr: int, what: str) -> int:
    if duration < 0:
        raise FeatureError(f"{what} duration must be >= 0, got {duration}")
    n = int(round(duration * sr))
    if n > n_samples:
        raise FeatureError(
            f"{what} of {duration}s is longer than the clip "
            f"({n_samples / sr:.3f}s)")
    return n


def fade_in(y: np.ndarray, sr: int, duration: float = 0.01,
            shape: str = 'linear') -> np.ndarray:
    """Ramp the first *duration* seconds up from silence.

    >>> y = kudio.fade_in(y, sr, duration=0.05, shape='cosine')

    A 10 ms default is the usual anti-click amount: long enough to remove the
    step at the start of a cut, short enough not to be heard as a fade.
    """
    y = np.asarray(y)
    n = _fade_samples(y.shape[0], duration, sr, "fade_in")
    if n == 0 or y.size == 0:
        return y.copy()
    out = y.astype(np.float64, copy=True)
    curve = _ramp(n, shape)
    out[:n] *= curve if y.ndim == 1 else curve[:, None]
    return out.astype(y.dtype, copy=False)


def fade_out(y: np.ndarray, sr: int, duration: float = 0.01,
             shape: str = 'linear') -> np.ndarray:
    """Ramp the last *duration* seconds down to silence."""
    y = np.asarray(y)
    n = _fade_samples(y.shape[0], duration, sr, "fade_out")
    if n == 0 or y.size == 0:
        return y.copy()
    out = y.astype(np.float64, copy=True)
    curve = _ramp(n, shape)[::-1]
    out[-n:] *= curve if y.ndim == 1 else curve[:, None]
    return out.astype(y.dtype, copy=False)


def fade(y: np.ndarray, sr: int, fade_in_s: float = 0.01,
         fade_out_s: float = 0.01, shape: str = 'linear') -> np.ndarray:
    """Fade both ends in one call.

    The two ramps are checked against the clip length **together**, so a fade
    in and a fade out that overlap is an error rather than a silently
    multiplied middle.
    """
    y = np.asarray(y)
    n_in = _fade_samples(y.shape[0], fade_in_s, sr, "fade_in")
    n_out = _fade_samples(y.shape[0], fade_out_s, sr, "fade_out")
    if n_in + n_out > y.shape[0]:
        raise FeatureError(
            f"fades overlap: {fade_in_s}s + {fade_out_s}s exceeds the clip "
            f"({y.shape[0] / sr:.3f}s)")
    return fade_out(fade_in(y, sr, fade_in_s, shape), sr, fade_out_s, shape)


def crossfade(first: np.ndarray, second: np.ndarray, sr: int,
              duration: float = DEFAULT_CROSSFADE,
              shape: str = 'equal_power') -> np.ndarray:
    """Join two clips through a short overlap, instead of butting them together.

    A hard splice puts a step in the waveform, and a step is a click. Blending
    the last *duration* of *first* into the first *duration* of *second*
    removes it.

    >>> joined = kudio.crossfade(before, after, sr)      # 5 ms

    **The result is shorter than the two parts** by *duration* — the overlap is
    shared, not inserted. That is not a rounding error to be papered over: a
    caller tracking positions in the audio has to account for it, and hiding it
    would make those positions silently wrong.

    :param shape: ``'equal_power'`` (sine/cosine) holds the perceived level
        steady when the two sides are **unrelated** — two different takes, two
        different sources. ``'linear'`` holds the *sum* steady, which is right
        when both sides are the same signal continuing; equal-power on alike
        material bumps the level by up to 3 dB. :func:`splice` therefore
        defaults to linear and this defaults to equal-power.
    """
    first = np.asarray(first)
    second = np.asarray(second)
    if first.size == 0:
        return second.copy()
    if second.size == 0:
        return first.copy()
    if duration < 0:
        raise FeatureError(f"crossfade duration must be >= 0, got {duration}")

    n = int(round(duration * sr))
    n = min(n, first.shape[0], second.shape[0])
    if n <= 0:
        return np.concatenate([first, second])
    if shape not in ('equal_power', 'linear'):
        raise FeatureError(
            f"shape must be 'equal_power' or 'linear', got {shape!r}")

    t = np.linspace(0.0, 1.0, n)
    if shape == 'equal_power':
        down, up = np.cos(t * np.pi / 2.0), np.sin(t * np.pi / 2.0)
    else:
        down, up = 1.0 - t, t
    if first.ndim > 1:
        down, up = down[:, None], up[:, None]

    blended = first[-n:] * down + second[:n] * up
    return np.concatenate([first[:-n], blended, second[n:]]).astype(
        first.dtype, copy=False)


def splice(parts, sr: int, duration: float = DEFAULT_CROSSFADE,
           shape: str = 'linear') -> np.ndarray:
    """Join several clips end to end, crossfading every seam.

    >>> merged = kudio.splice([before, replacement, after], sr)

    Empty parts are skipped rather than producing a seam with nothing on one
    side. The result is shorter than the sum of the parts by one *duration* per
    seam; :func:`crossfade` explains why that matters.

    **The default shape differs from `crossfade`'s, deliberately.** A splice
    joins two moments of the *same* recording, which are usually alike, and
    equal-power over-shoots on alike material: measured on a sine cut at a
    period boundary, equal-power raised the peak from 0.500 to 0.567 while
    linear left it at 0.500. A crossfade between genuinely different material
    is the other case, and defaults the other way.
    """
    usable = [np.asarray(p) for p in parts if np.asarray(p).size]
    if not usable:
        return np.zeros(0, dtype=np.float32)
    merged = usable[0]
    for part in usable[1:]:
        merged = crossfade(merged, part, sr, duration, shape)
    return merged


def reverse(y: np.ndarray) -> np.ndarray:
    """Play the waveform backwards. Multi-channel audio keeps its channels."""
    y = np.asarray(y)
    return np.ascontiguousarray(y[::-1])


def remove_dc(y: np.ndarray) -> np.ndarray:
    """Subtract the mean, per channel.

    A constant offset costs headroom and shows up as an audible thump at every
    cut. This removes a *constant* one; an offset that drifts over a long
    recording is a very low frequency, and :func:`kudio.highpass` is the tool
    for that.
    """
    y = np.asarray(y)
    if y.size == 0:
        return y.copy()
    # accumulate in float64: a float32 mean over a long clip carries enough
    # rounding error to leave behind the offset it was asked to remove
    offset = np.mean(y, axis=0, dtype=np.float64, keepdims=True)
    return (y - offset).astype(y.dtype, copy=False)
