# -*- coding: utf-8 -*-
"""Small generic helpers: timing decorators and list utilities."""
from __future__ import annotations

import logging
import threading
import time
from collections import Counter
from functools import wraps

__all__ = [
    'timer',
    'Timer',
    'find_duplicate',
]

log = logging.getLogger(__name__)


def timer(func):
    """Decorator: log the wall-clock run time of *func*."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        before = time.time()
        result = func(*args, **kwargs)
        log.info("%s(): %.7f s", func.__name__, time.time() - before)
        return result

    return wrapper


class Timer:
    """Decorator: run *func* in a daemon thread, optionally joining for at most
    ``time_delay`` seconds.

    >>> @Timer(time_delay=3)
    ... def job(): ...
    >>> thread = job()   # returns the started thread
    """

    def __init__(self, time_delay: float):
        self.time_delay = time_delay

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            th = threading.Thread(target=func, args=args, kwargs=kwargs, daemon=True)
            log.debug("Timer: thread start (%s)", func.__name__)
            th.start()
            if self.time_delay > 0:
                th.join(self.time_delay)
                log.debug("Timer: join finished (%s)", func.__name__)
            return th

        return wrapper


def find_duplicate(file_list):
    """Return the ``Counter`` dict of *file_list*, logging duplicated entries."""
    counts = dict(Counter(file_list))
    duplicates = {key: value for key, value in counts.items() if value > 1}
    if duplicates:
        log.info("Duplicates: %s", duplicates)
    return counts
