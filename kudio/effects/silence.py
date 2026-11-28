# -*- coding: utf-8 -*-
"""Silence trimming and splitting.

Measured down from the loudest frame, so it suits a sound against a quiet
background: a studio take, a knock in a still room. Speech over a room that is
never quiet is :func:`kudio.vad`'s job.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from kudio._numbers import count, real
from kudio._samples import as_float, channels_last, finite
from kudio.core._mel import decibels
from kudio.exceptions import FeatureError

__all__ = ['trim_silence', 'split_on_silence']

#: about how many bytes of float32 frames are measured at a time
_BLOCK = 1 << 18


def _sounding(y: np.ndarray, top_db: float, frame_length, hop_length, name: str
              ) -> Tuple[np.ndarray, int]:
    """Which frames of *y*, ``(frames,)`` or ``(frames, channels)``, are within
    *top_db* of the loudest, and the hop between them.

    Frame *k* is centred on sample ``k * hop``, zeros past the ends of the clip,
    and is as loud as its RMS. The loudest is that of any channel, and a frame is
    sounding when any of its channels is. Silence throughout has no loudest
    frame to lie under, and all of it counts as sounding.
    """
    frame = count(frame_length, 'frame_length', name, FeatureError, ' of samples')
    hop = count(hop_length, 'hop_length', name, FeatureError, ' of samples')
    if y.ndim not in (1, 2):
        raise FeatureError(f"{name}() takes (frames,) or (frames, channels), got shape {y.shape}")
    if y.ndim == 2 and not y.shape[1]:
        raise FeatureError(f"{name}() got an array of shape {y.shape}, with no channel in it")
    # found on the float reading, where uint8 silence is 0 rather than 128
    x = finite(as_float(y, name, FeatureError), name, FeatureError).T.reshape(-1, len(y)) \
        if y.size else np.zeros((1, len(y)), np.float32)
    padded = np.pad(x, [(0, 0), (frame // 2, frame // 2)])
    if padded.shape[1] < frame:                     # no samples, and an odd frame
        return np.ones(1, bool), hop
    frames = np.lib.stride_tricks.sliding_window_view(padded, frame, axis=1)[:, ::hop]
    level = np.empty(frames.shape[:2], np.float32)
    step = max(1, _BLOCK // (4 * frame * len(x)))
    with np.errstate(over='ignore'):                    # said below, in kudio's words
        for start in range(0, level.shape[1], step):
            part = slice(start, start + step)
            level[:, part] = np.sqrt(np.mean(np.square(frames[:, part], dtype=np.float32),
                                             axis=-1))
    if not np.isfinite(level).all():
        raise FeatureError(f"{name}(): the samples go as high as {np.max(np.abs(x)):g}, and the "
                           f"square of that is past what float32 holds")
    db = decibels(np.square(level), ref=np.square(level.max(keepdims=True)), floor=1e-5 ** 2)
    return db.max(axis=0) > -top_db, hop


def _check_top_db(top_db, name: str) -> float:
    """*top_db* as a number, refused when it is not a distance down from the
    loudest frame.

    At 0 or below, and at NaN, the loudest frame itself is not above the line,
    so the whole clip is silence: what comes back is empty, and what is written
    from it is a file with nothing in it.
    """
    value = real(top_db)
    if value is None or value <= 0:
        raise FeatureError(f"{name}(): top_db has to be a finite number of dB above "
                           f"0, got {top_db!r}")
    return value


def trim_silence(y: np.ndarray, top_db: float = 30.0,
                 frame_length: int = 2048, hop_length: int = 512
                 ) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Trim leading/trailing silence below ``top_db`` dB under peak.

    The level is measured in frames of *frame_length* samples, one every
    *hop_length*, and the cut is made at the frames: the clip starts where the
    first one within *top_db* of the loudest is centred, and ends a hop after the
    last. A clip that is silent throughout has no peak to be under, and is kept
    whole.

    Returns ``(trimmed_waveform, (start_sample, end_sample))``. The waveform is
    a copy, and a ``(frames, channels)`` array is trimmed along its frames.

    :raises FeatureError: on a *top_db* that is not a finite number above 0, on
        samples that are not finite, and on a size that is not a whole number of
        1 or more.
    """
    top_db = _check_top_db(top_db, 'trim_silence')
    y = channels_last(y, 'trim_silence', FeatureError, clip=False)
    sounding, hop = _sounding(y, top_db, frame_length, hop_length, 'trim_silence')
    first, last = np.flatnonzero(sounding)[[0, -1]]
    start, end = int(first) * hop, min(len(y), int(last + 1) * hop)
    # cut from y itself, so the clip comes back in its own type
    return y[start:end].copy(), (start, end)


def split_on_silence(y: np.ndarray, top_db: float = 30.0,
                     frame_length: int = 2048, hop_length: int = 512
                     ) -> List[np.ndarray]:
    """Split *y* into non-silent segments (each below-``top_db`` gap is a cut),
    which are measured and cut as :func:`trim_silence` measures and cuts.

    Each segment is a copy; a ``(frames, channels)`` array is split along its
    frames.

    :raises FeatureError: as :func:`trim_silence` does.
    """
    top_db = _check_top_db(top_db, 'split_on_silence')
    y = channels_last(y, 'split_on_silence', FeatureError, clip=False)
    sounding, hop = _sounding(y, top_db, frame_length, hop_length, 'split_on_silence')
    # where a run of sounding frames starts, and where the next silence does
    turns = np.flatnonzero(np.diff(np.concatenate([[0], sounding, [0]]).astype(int)))
    cuts = np.minimum(turns * hop, len(y)).reshape(-1, 2)
    return [y[start:end].copy() for start, end in cuts]
