# -*- coding: utf-8 -*-
"""A setting that has to be one number, and what is not one.

``True`` is 1 to Python, ``float('0.05')`` works, and NaN is neither above a
limit nor below it, so ``loss <= 1`` and ``burst < 1`` on their own let all
three through. Each module puts the range and the message to it.
"""
from __future__ import annotations

import math
import numbers
from typing import Any, Optional, Type

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


def within(value: Any, name: str, where: str, error: Type[Exception] = ValueError, *,
           low: Optional[float] = None, above: Optional[float] = None,
           high: Optional[float] = None, of: str = '', inf: bool = False) -> float:
    """*value* as a float when it is one number in range, or an *error* that names
    the function and the argument.

    :param low, high: the ends of the range, both in it.
    :param above: the lower end of it, and not in it.
    :param of: what it counts: ``' of seconds'``.
    :param inf: let infinity be a number, for a limit that may be none at all.
    """
    number = real(value)
    if number is None and inf and isinstance(value, (float, np.floating)) and math.isinf(value):
        number = float(value)
    if (number is not None and (low is None or number >= low)
            and (above is None or number > above) and (high is None or number <= high)):
        return number
    if low is not None and high is not None:
        span = f" from {low:g} to {high:g}"
    elif above is not None:
        span = f" above {above:g}" + ("" if high is None else f" and up to {high:g}")
    else:
        span = "" if low is None else f" of {low:g} or more"
    raise error(f"{where}(): {name} has to be a {'' if inf else 'finite '}number{of}{span}, "
                f"got {value!r}")


def flag(value: Any, name: str, where: str, error: Type[Exception] = ValueError,
         none: bool = False) -> Optional[bool]:
    """*value* as a bool when it is True, False, numpy's, or a whole number that is 0 or
    1; None when *none* lets it be. Text says yes and no to nobody: ``'no'`` is True to
    Python, and so are 2 and NaN."""
    if value is None and none:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, numbers.Integral) and value in (0, 1):
        return bool(value)
    raise error(f"{where}(): {name} is True or False{', or None' if none else ''}, "
                f"got {value!r}")


def generator(seed: Any, where: str, error: Type[Exception] = ValueError) -> np.random.Generator:
    """numpy's generator of *seed*, whatever it takes for one, or an *error* that names
    the function."""
    try:
        return np.random.default_rng(seed)
    except (TypeError, ValueError) as e:
        raise error(f"{where}(): seed has to be a whole number of 0 or more, or None, got "
                    f"{seed!r}") from e


def count(value: Any, name: str, where: str, error: Type[Exception] = ValueError,
          of: str = '') -> int:
    """*value* as a whole number of 1 or more, or an *error* that names the
    function and the argument. *of* is what is counted, where it needs saying:
    ``' of samples'``."""
    number = whole(value)
    if number is None or number < 1:
        raise error(f"{where}(): {name} must be >= 1, a whole number{of}, got {value!r}")
    return number
