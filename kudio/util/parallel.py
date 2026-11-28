# -*- coding: utf-8 -*-
"""Parallel batch processing helpers."""
from __future__ import annotations

from typing import Callable, Iterable, List, Optional

from kudio._progress import tqdm
from kudio._workers import Workers

__all__ = ['map_waves']


def map_waves(fn: Callable, items: Iterable, workers: int = 1,
              desc: Optional[str] = None, chunksize: int = 1) -> List:
    """Apply *fn* to every item, optionally across ``workers`` processes.

    With ``workers <= 1`` runs sequentially (easier to debug); otherwise in as
    many processes, and on Windows before Python 3.13 in 31 at most, which is as
    many as the wait for them can hold. *fn* must be top-level (picklable) for
    ``workers > 1``.

    :raises KudioError: when one of the processes dies.
    """
    items = list(items)
    if workers and workers > 1 and len(items) > 1:
        with Workers(min(workers, len(items)), 'map_waves()') as pool:
            return list(tqdm(pool.imap(fn, items, chunksize),
                             total=len(items), desc=desc or "map_waves"))
    return [fn(x) for x in tqdm(items, desc=desc or "map_waves")]
