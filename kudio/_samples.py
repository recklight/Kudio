# -*- coding: utf-8 -*-
"""How kudio reads an integer array: as audio on its type's own full scale.

That is the reading soundfile gives an integer file -- int16 over 32768, int32
over 2**31, uint8 around its midpoint of 128 -- so a level measured on an int16
array is the level of the file it came from. Read as plain numbers instead, a
quarter-scale int16 tone measured +74.6 LUFS.
"""
from __future__ import annotations

from typing import Callable, Tuple

import numpy as np


def _scale(dtype) -> Tuple[float, float]:
    """``(full scale, offset)`` of an integer type."""
    limits = np.iinfo(dtype)
    if limits.min < 0:
        return -float(limits.min), 0.0
    full = (int(limits.max) + 1) / 2.0                 # unsigned: silence is mid-range
    return full, full


def as_float(y) -> np.ndarray:
    """*y* as float audio: an integer array on its type's full scale, in
    float64, and anything else as it came."""
    y = np.asarray(y)
    if not np.issubdtype(y.dtype, np.integer):
        return y
    full, offset = _scale(y.dtype)
    return (y.astype(np.float64) - offset) / full


def to_float(y) -> Tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    """*y* as float audio, and what puts a result back into *y*'s type.

    An integer result is rounded and clipped into range: cast straight back,
    a float 1.2 wraps round to -0.8.
    """
    y = np.asarray(y)
    if not np.issubdtype(y.dtype, np.integer):
        return y, lambda out: out.astype(y.dtype)
    limits = np.iinfo(y.dtype)
    full, offset = _scale(y.dtype)

    def back(out):
        return np.clip(np.round(out * full + offset), limits.min, limits.max).astype(y.dtype)
    return as_float(y), back
