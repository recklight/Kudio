# -*- coding: utf-8 -*-
"""How kudio reads an array as audio: what its numbers mean, and which way up
it is.

An integer array is on its type's own full scale, the reading soundfile gives
an integer file -- int16 over 32768, int32 over 2**31, uint8 around its
midpoint of 128 -- so a level measured on an int16 array is the level of the
file it came from. Read as plain numbers instead, a quarter-scale int16 tone
measured +74.6 LUFS.

A 2-D array is ``(frames, channels)``. A row is one channel, and an array with
more columns than rows is channels first, librosa's layout, where every
function here would take a channel for an instant. Samples out of a file are
not guessed about (:func:`layout_known`): two frames of 5.1 have more columns
than rows too.
"""
from __future__ import annotations

import contextlib
import contextvars
from typing import Callable, Iterator, Optional, Tuple, Type

import numpy as np

#: Set while kudio works on what it read from a file itself.
_KNOWN = contextvars.ContextVar('kudio_layout_known', default=False)


@contextlib.contextmanager
def layout_known() -> Iterator[None]:
    """Inside, a 2-D array is ``(frames, channels)`` whatever its shape. A file
    says how many channels it has, however few frames."""
    token = _KNOWN.set(True)
    try:
        yield
    finally:
        _KNOWN.reset(token)


def _scale(dtype) -> Tuple[float, float]:
    """``(full scale, offset)`` of an integer type."""
    limits = np.iinfo(dtype)
    if limits.min < 0:
        return -float(limits.min), 0.0
    full = (int(limits.max) + 1) / 2.0                 # unsigned: silence is mid-range
    return full, full


def refuse_64_bit(y, name: Optional[str] = None,
                  error: Type[Exception] = ValueError) -> np.ndarray:
    """*y* as an array, unless it holds 64-bit integers.

    No audio file does. numpy gives them for a list of ints and for
    ``astype(int)``, and 16-bit samples read on their full scale sit 288 dB
    down: they would measure as silence and be written out as zeros.
    """
    y = np.asarray(y)
    if y.dtype.kind in 'iu' and y.dtype.itemsize > 4:
        raise error(f"{f'{name}(): ' if name else ''}the samples are {y.dtype}, and no "
                    f"audio file holds 64-bit integers: read on that type's full scale, "
                    f"16-bit samples sit 288 dB down. A list of ints and astype(int) "
                    f"usually give int64; pass the samples as the type they were read "
                    f"as (int16 for 16-bit PCM), or as float")
    return y


def as_float(y, name: Optional[str] = None,
             error: Type[Exception] = ValueError) -> np.ndarray:
    """*y* as float audio: an integer array on its type's full scale, in
    float64, and anything else as it came. 64-bit integers are refused, the
    message starting with *name*, the function the array was passed to."""
    y = refuse_64_bit(y, name, error)
    if not np.issubdtype(y.dtype, np.integer):
        return y
    full, offset = _scale(y.dtype)
    return (y.astype(np.float64) - offset) / full


def to_float(y, name: Optional[str] = None, error: Type[Exception] = ValueError
             ) -> Tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    """*y* as float audio, and what puts a result back into *y*'s type.

    An integer result is rounded and clipped into range: cast straight back,
    a float 1.2 wraps round to -0.8.
    """
    y = np.asarray(y)
    if not np.issubdtype(y.dtype, np.integer):
        return y, lambda out: out.astype(y.dtype)
    x = as_float(y, name, error)
    limits = np.iinfo(y.dtype)
    full, offset = _scale(y.dtype)

    def back(out):
        return np.clip(np.round(out * full + offset), limits.min, limits.max).astype(y.dtype)
    return x, back


def channels_last(y, name: str, error: Type[Exception] = ValueError) -> np.ndarray:
    """*y* as ``(frames,)`` or ``(frames, channels)``.

    A row ``(1, n)`` is one channel and comes back ``(n,)``. With two rows or
    more and more columns than rows, an array is channels first; nothing else
    tells the two layouts apart, and half a second of 16-channel audio has
    8000 rows.
    """
    y = np.asarray(y)
    if y.ndim != 2 or _KNOWN.get():
        return y
    rows, columns = y.shape
    if rows == 1:
        return y.reshape(-1)
    if columns > rows >= 2:
        raise error(f"{name}(): an array of shape {y.shape} is channels first, and kudio "
                    f"takes (frames, channels); transpose it with y.T")
    return y


def one_channel(y, name: str, error: Type[Exception] = ValueError) -> np.ndarray:
    """:func:`as_float` and :func:`channels_last` for a function that takes one
    channel: ``(n,)``, a column ``(n, 1)`` and a row ``(1, n)`` come back
    ``(n,)``. More than one is returned as it is, for the function to refuse in
    its own words."""
    y = channels_last(as_float(y, name, error), name, error)
    return y[:, 0] if y.ndim == 2 and y.shape[1] == 1 else y
