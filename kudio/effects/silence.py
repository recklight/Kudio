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


def trim_silence(y: np.ndarray, top_db: float = 30.0,
                 frame_length: int = 2048, hop_length: int = 512
                 ) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Trim leading/trailing silence below ``top_db`` dB under peak.

    Returns ``(trimmed_waveform, (start_sample, end_sample))``.
    """
    trimmed, index = librosa.effects.trim(
        y, top_db=top_db, frame_length=frame_length, hop_length=hop_length)
    return trimmed, (int(index[0]), int(index[1]))


def split_on_silence(y: np.ndarray, top_db: float = 30.0,
                     frame_length: int = 2048, hop_length: int = 512
                     ) -> List[np.ndarray]:
    """Split *y* into non-silent segments (each below-``top_db`` gap is a cut)."""
    intervals = librosa.effects.split(
        y, top_db=top_db, frame_length=frame_length, hop_length=hop_length)
    return [y[start:end] for start, end in intervals]
