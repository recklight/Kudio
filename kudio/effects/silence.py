# -*- coding: utf-8 -*-
"""Silence trimming and splitting.

Useful for isolating the informative part of an inspection recording (e.g. the
impact / chipping transient) from surrounding silence.
"""
from __future__ import annotations

from typing import List, Tuple

import librosa
import numpy as np

__all__ = ['trim_silence', 'split_on_silence']


def _time_last(y: np.ndarray) -> np.ndarray:
    """librosa reads its last axis as time; kudio's multi-channel layout is
    ``(frames, channels)``, so a 2-D array is handed over transposed. librosa
    then counts a frame as sound when any channel is."""
    y = np.asarray(y)
    if y.ndim > 2:
        raise ValueError(f"expected (frames,) or (frames, channels), got shape {y.shape}")
    return y.T if y.ndim == 2 else y


def trim_silence(y: np.ndarray, top_db: float = 30.0,
                 frame_length: int = 2048, hop_length: int = 512
                 ) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Trim leading/trailing silence below ``top_db`` dB under peak.

    Returns ``(trimmed_waveform, (start_sample, end_sample))``. The waveform is
    a copy, and a ``(frames, channels)`` array is trimmed along its frames.
    """
    y = np.asarray(y)
    _, index = librosa.effects.trim(
        _time_last(y), top_db=top_db, frame_length=frame_length,
        hop_length=hop_length)
    start, end = int(index[0]), int(index[1])
    return y[start:end].copy(), (start, end)


def split_on_silence(y: np.ndarray, top_db: float = 30.0,
                     frame_length: int = 2048, hop_length: int = 512
                     ) -> List[np.ndarray]:
    """Split *y* into non-silent segments (each below-``top_db`` gap is a cut).

    Each segment is a copy; a ``(frames, channels)`` array is split along its
    frames.
    """
    y = np.asarray(y)
    intervals = librosa.effects.split(
        _time_last(y), top_db=top_db, frame_length=frame_length,
        hop_length=hop_length)
    return [y[start:end].copy() for start, end in intervals]
