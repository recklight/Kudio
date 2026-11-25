# -*- coding: utf-8 -*-
"""One home for "cut this waveform into overlapping frames", and for smoothing
what was measured on them.

Private. Two analysis modules need identical framing and, written twice, they
would eventually disagree about what frame 40 covers — the failure this
codebase keeps meeting. Not exported: callers who want framing as a feature
operation want :func:`kudio.wave_slicing` or :class:`kudio.STFT`.
"""
from __future__ import annotations

import numpy as np

__all__ = ['frame_view', 'convolve_same', 'overlapped']


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


def convolve_same(values: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """``np.convolve(values, kernel, mode='same')`` with one answer per value,
    however few values there are.

    numpy's answer is as long as the longer of the two. A kernel 41 frames
    wide over a clip of 16 gives 41 numbers, centred on the kernel, and none
    of them belongs to a frame any more. This is the same sum cut to the
    values: what lies past either end of them counts as zero, as it does near
    the ends of a long clip.
    """
    if len(kernel) <= len(values):
        return np.convolve(values, kernel, mode='same')
    start = (len(kernel) - 1) // 2
    return np.convolve(values, kernel)[start:start + len(values)]


def overlapped(energy: np.ndarray, hop: int) -> np.ndarray:
    """What a frame's *energy* adds up to over a sample in the middle of a
    clip, frames starting every *hop* samples: one value for each of the *hop*
    places the sample can have under them."""
    total = np.zeros(hop)
    for start in range(0, len(energy), hop):
        part = energy[start:start + hop]
        total[:len(part)] += part
    return total
