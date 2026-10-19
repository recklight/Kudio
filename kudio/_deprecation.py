# -*- coding: utf-8 -*-
"""Helpers for keeping backwards-compatible aliases while nudging callers
towards the v3 canonical names."""
from __future__ import annotations

import functools
import warnings
from typing import Callable, Optional


def alias(new: Callable, name: Optional[str] = None, *,
          old_prefix: str = 'kudio', new_prefix: Optional[str] = None) -> Callable:
    """Return a wrapper around *new* that emits a ``DeprecationWarning``.

    >>> w2s = alias(waveform_to_spectrogram, 'w2s')
    >>> red = alias(colors.red, new_prefix='kudio.util')   # same name, moved

    *old_prefix* and *new_prefix* are where the warning says the old name is
    and where the new one lives; the new one defaults to the old.
    """
    old_name = name or new.__name__
    new_path = f"{new_prefix or old_prefix}.{new.__name__}"

    @functools.wraps(new)
    def wrapper(*args, **kwargs):
        warnings.warn(
            f"{old_prefix}.{old_name}() is deprecated and will be removed in a "
            f"future release; use {new_path}() instead.",
            DeprecationWarning, stacklevel=2)
        return new(*args, **kwargs)

    wrapper.__name__ = old_name
    wrapper.__qualname__ = old_name
    shown = new.__name__ if new_prefix is None else new_path
    wrapper.__doc__ = f"Deprecated alias for :func:`{shown}`."
    return wrapper
