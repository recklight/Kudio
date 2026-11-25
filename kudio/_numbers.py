# -*- coding: utf-8 -*-
"""A setting that has to be one number, and what is not one.

``True`` is 1 to Python, ``float('0.05')`` works, and NaN is neither above a
limit nor below it, so ``loss <= 1`` and ``burst < 1`` on their own let all
three through. Each module puts the range and the message to it.
"""
from __future__ import annotations

import math
import numbers
from typing import Any, Optional

import numpy as np


def real(value: Any) -> Optional[float]:
    """*value* as a float when it is one finite number, of Python's types or
    numpy's, or a 0-d array of one; None when it is anything else."""
    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = value.item()
    if not isinstance(value, numbers.Real) or isinstance(value, (bool, np.bool_)):
        return None
    try:
        value = float(value)
    except (OverflowError, ValueError):              # an int past float64
        return None
    return value if math.isfinite(value) else None


def whole(value: Any) -> Optional[int]:
    """*value* as an int when it is one whole number, 8.0 included, else None."""
    number = real(value)
    return int(number) if number is not None and number.is_integer() else None
