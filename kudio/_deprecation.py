# -*- coding: utf-8 -*-
"""Helpers for keeping backwards-compatible aliases while nudging callers
towards the v3 canonical names."""
from __future__ import annotations

import functools
import warnings
from typing import Callable, Optional


def alias(new: Callable, name: Optional[str] = None) -> Callable:
    """Return a wrapper around *new* that emits a ``DeprecationWarning``.

    >>> w2s = alias(waveform_to_spectrogram, 'w2s')
    """
    old_name = name or new.__name__

    @functools.wraps(new)
    def wrapper(*args, **kwargs):
        warnings.warn(
            f"kudio.{old_name}() is deprecated and will be removed in a future "
            f"release; use kudio.{new.__name__}() instead.",
            DeprecationWarning, stacklevel=2)
        return new(*args, **kwargs)

    wrapper.__name__ = old_name
    wrapper.__qualname__ = old_name
    wrapper.__doc__ = f"Deprecated alias for :func:`{new.__name__}`."
    return wrapper
