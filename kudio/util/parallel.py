# -*- coding: utf-8 -*-
"""Parallel batch processing helpers."""
from __future__ import annotations

from functools import partial
from multiprocessing import Pool
from typing import Callable, Iterable, List, Optional

from tqdm import tqdm

__all__ = ['map_waves']


def map_waves(fn: Callable, items: Iterable, workers: int = 1,
              desc: Optional[str] = None, chunksize: int = 1) -> List:
    """Apply *fn* to every item, optionally across ``workers`` processes.

    With ``workers <= 1`` runs sequentially (easier to debug); otherwise uses a
    multiprocessing pool. *fn* must be top-level (picklable) for ``workers > 1``.
    """
    items = list(items)
    if workers and workers > 1 and len(items) > 1:
        with Pool(workers) as pool:
            return list(tqdm(pool.imap(fn, items, chunksize=chunksize),
                             total=len(items), desc=desc or "map_waves"))
    return [fn(x) for x in tqdm(items, desc=desc or "map_waves")]
