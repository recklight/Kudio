# -*- coding: utf-8 -*-
"""Writing a manifest, a label file or a config in one piece.

Opening a file for writing empties it. A value JSON or YAML has no form for,
met with the first rows already written, leaves half a manifest where a whole
one was, so the text is finished first and the file opened last.
"""
from __future__ import annotations

from pathlib import Path, PurePath
from typing import Any, Optional, Union

import numpy as np


def plain(value: Any) -> Any:
    """A numpy number as Python's own and a path as its text, with forward
    slashes; anything else as it came.

    A time is often measured and a path globbed, and JSON and YAML know
    neither ``np.float32`` nor ``Path``.
    """
    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = value[()]
    # not .item(), which hands a np.longdouble back as it is
    for numpys, pythons in ((np.bool_, bool), (np.integer, int),
                            (np.floating, float), (np.str_, str)):
        if isinstance(value, numpys):
            return pythons(value)
    if isinstance(value, PurePath):
        return value.as_posix()
    return value


def write(path: Union[str, PurePath], text: str, newline: Optional[str] = None,
          parents: bool = False) -> None:
    """*text* to *path* in UTF-8, with its folder made first if *parents*.
    Text UTF-8 cannot hold, a lone surrogate out of a file name say, is refused
    before either is touched."""
    text.encode("utf-8")
    if parents:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline=newline) as fh:
        fh.write(text)
