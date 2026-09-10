# -*- coding: utf-8 -*-
"""One home for "cut this waveform into overlapping frames".

Private. Two analysis modules need identical framing and, written twice, they
would eventually disagree about what frame 40 covers — the failure this
codebase keeps meeting. Not exported: callers who want framing as a feature
operation want :func:`kudio.wave_slicing` or :class:`kudio.STFT`.
"""
from __future__ import annotations

import numpy as np

__all__ = ['frame_view']


def frame_view(y: np.ndarray, size: int, hop: int) -> np.ndarray:
    """Non-copying ``(n_frames, size)`` view over *y*.

    A clip shorter than one frame comes back as a single frame holding all of
    it, rather than as an empty result — the caller asked a question about the
    audio it has.
    """
    if size <= 0 or len(y) < size:
        return y[np.newaxis, :]
    n = 1 + (len(y) - size) // hop
    return np.lib.stride_tricks.as_strided(
        y, shape=(n, size), strides=(y.strides[0] * hop, y.strides[0]),
        writeable=False)
