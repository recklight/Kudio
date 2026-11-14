# -*- coding: utf-8 -*-
"""Progress bars, with tqdm imported the first time a bar is drawn.

Only the batch helpers draw one, and importing tqdm costs more than most of
kudio's own modules, so ``import kudio`` leaves it alone. The bars are tqdm's
own, so ``TQDM_DISABLE=1`` turns them off.
"""
from __future__ import annotations


def tqdm(*args, **kwargs):
    """``tqdm.tqdm(*args, **kwargs)``."""
    from tqdm import tqdm as bar
    return bar(*args, **kwargs)


def trange(*args, **kwargs):
    """``tqdm.trange(*args, **kwargs)``."""
    from tqdm import trange as bar_range
    return bar_range(*args, **kwargs)
