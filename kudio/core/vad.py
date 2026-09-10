# -*- coding: utf-8 -*-
"""Voice activity detection: where in this recording is somebody talking.

:func:`kudio.split_on_silence` cuts wherever the level drops, which is the
right answer for a clean studio take and the wrong one for anything with a
noise floor — a fan, a road outside, a hiss. Above a certain noise level a
fixed ``top_db`` either keeps the whole file or throws away the quiet
consonants, and there is no value in between that works.

This decides per frame instead, on three cheap signals:

* **level against the recording's own noise floor**, rather than against a
  fixed threshold, so it adapts to how noisy the file happens to be;
* **spectral flatness** — speech is harmonic and therefore peaky, steady noise
  is flat. This is what separates "a loud fan" from "a quiet voice";
* **zero-crossing rate**, which keeps the unvoiced fricatives (*s*, *f*, *sh*)
  that carry almost no energy and would otherwise be cut off the ends of words
  — but only where they sit **next to** something voiced, because on its own a
  fricative and a hiss are the same measurement.

The consequence of that last rule is worth knowing: a whisper, which has no
voiced frames at all, is not detected. Admitting it would mean admitting every
noise burst in every file, and a detector that fires on the air conditioning is
not a detector.

Then the frame decisions are smoothed: short gaps inside speech are bridged,
segments too short to be a word are dropped, and what survives is padded so the
cut does not land on the first consonant. Pure numpy — no webrtcvad, no torch.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

from kudio.core._framing import frame_view
from kudio.exceptions import FeatureError

__all__ = ['vad', 'vad_split', 'vad_trim', 'speech_ratio']

log = logging.getLogger(__name__)

#: Analysis frame / hop in seconds. 30 ms sits inside a single phoneme.
FRAME = 0.030
HOP = 0.010

#: How far above the estimated noise floor a frame has to be, in dB.
THRESHOLD_DB = 8.0
#: Spectral flatness above this is noise-like, not voice. 1.0 is white noise.
FLATNESS_MAX = 0.40
#: A frame this far under the loudest one is silence whatever else says.
FLOOR_REL_DB = 45.0
#: Zero-crossing rate above this rescues unvoiced fricatives.
ZCR_MIN = 0.25
#: ...but only within this many seconds of something actually voiced. A
#: fricative on its own is indistinguishable from broadband noise; a fricative
#: at the edge of a vowel is a consonant.
FRICATIVE_REACH = 0.20

_NOISE_PERCENTILE = 15.0


def _flatness(frames: np.ndarray) -> np.ndarray:
    """Wiener entropy per frame: geometric over arithmetic mean of the power."""
    window = np.hanning(frames.shape[-1])
    power = np.abs(np.fft.rfft(frames * window, axis=-1)) ** 2
    power = np.maximum(power[:, 1:], 1e-20)         # drop DC, keep the log sane
    geometric = np.exp(np.mean(np.log(power), axis=-1))
    arithmetic = np.mean(power, axis=-1)
    return geometric / np.maximum(arithmetic, 1e-20)


def _zcr(frames: np.ndarray) -> np.ndarray:
    return np.mean(np.diff(np.signbit(frames), axis=-1), axis=-1)


def _merge(spans: List[Tuple[float, float]], gap: float
           ) -> List[Tuple[float, float]]:
    """Join spans separated by less than *gap* seconds."""
    if not spans:
        return []
    merged = [list(spans[0])]
    for start, end in spans[1:]:
        if start - merged[-1][1] <= gap:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    return [(a, b) for a, b in merged]


def vad(y: np.ndarray, sr: int, *,
        threshold_db: float = THRESHOLD_DB,
        flatness_max: float = FLATNESS_MAX,
        min_speech: float = 0.10,
        min_silence: float = 0.15,
        pad: float = 0.05,
        frame: float = FRAME,
        hop: float = HOP) -> List[Tuple[float, float]]:
    """Find the spans of *y* that contain speech, in **seconds**.

    >>> spans = kudio.vad(y, sr)
    >>> spans[:2]
    [(0.31, 1.84), (2.42, 3.97)]

    :param threshold_db: how far above the estimated noise floor a frame must
        sit. Lower it for a very quiet talker, raise it for a noisy room.
    :param flatness_max: spectral flatness above which a loud frame is judged
        to be noise rather than voice. Raise it towards 1.0 to disable that
        test — useful for non-speech audio, where "harmonic" means nothing.
    :param min_speech: segments shorter than this are dropped (a click, a door).
    :param min_silence: gaps shorter than this do not split a segment; a stop
        consonant is a real silence in the middle of a word.
    :param pad: seconds added to each end, so the cut misses the consonant.
    :returns: non-overlapping ``(start, end)`` pairs, in order. Empty when
        nothing in the clip looks like speech.

    :raises FeatureError: on non-mono audio or a non-positive sample rate.
    """
    if sr <= 0:
        raise FeatureError(f"vad() needs a positive sample rate, got {sr}")
    y = np.ascontiguousarray(np.asarray(y, dtype=np.float64).squeeze())
    if y.ndim != 1:
        raise FeatureError(f"vad() takes mono audio, got shape {y.shape}")
    if y.size == 0:
        return []

    size = max(16, int(round(frame * sr)))
    step = max(1, int(round(hop * sr)))
    frames = frame_view(y, size, step)
    n_frames = frames.shape[0]
    duration = len(y) / sr

    rms = np.sqrt(np.mean(frames ** 2, axis=-1))
    with np.errstate(divide='ignore'):
        db = 20.0 * np.log10(np.maximum(rms, 0.0))
    finite = db[np.isfinite(db)]
    if finite.size == 0:                       # digital silence
        return []

    noise_floor = float(np.percentile(finite, _NOISE_PERCENTILE))
    loudest = float(finite.max())
    # two gates: above the room, and not absurdly far under the loudest moment.
    # The second stops a dead-quiet file from having its own hiss promoted to
    # speech simply because the hiss is 8 dB above the quietest hiss.
    active = (db > noise_floor + threshold_db) & (db > loudest - FLOOR_REL_DB)

    # what is loud *and* harmonic is the only thing trusted on its own
    core = active & (_flatness(frames) < flatness_max)
    # a hissy frame joins only if it is next to one of those. Zero-crossing
    # rate cannot tell an /s/ from a burst of white noise -- position can.
    reach = max(1, int(round(FRICATIVE_REACH / hop)))
    near_core = np.convolve(core.astype(np.float64),
                            np.ones(2 * reach + 1), mode='same') > 0
    speech = core | (active & (_zcr(frames) > ZCR_MIN) & near_core)

    if not speech.any():
        return []

    # frame index -> the seconds it covers
    edges = np.diff(np.concatenate(([0], speech.view(np.int8), [0])))
    starts = np.nonzero(edges == 1)[0]
    stops = np.nonzero(edges == -1)[0]
    spans = [(float(a * step) / sr,
              float(min((b - 1) * step + size, len(y))) / sr)
             for a, b in zip(starts, stops)]
    if n_frames == 1:                          # clip shorter than one frame
        spans = [(0.0, duration)] if speech[0] else []

    spans = _merge(spans, min_silence)
    spans = [s for s in spans if s[1] - s[0] >= min_speech]
    if pad > 0:
        spans = [(max(0.0, a - pad), min(duration, b + pad)) for a, b in spans]
        spans = _merge(spans, 0.0)
    return spans


def vad_split(y: np.ndarray, sr: int, **kwargs) -> List[np.ndarray]:
    """The speech segments themselves, as waveforms.

    The noise-adaptive counterpart of :func:`kudio.split_on_silence`; takes the
    same keyword arguments as :func:`vad`.
    """
    return [y[int(round(a * sr)):int(round(b * sr))]
            for a, b in vad(y, sr, **kwargs)]


def vad_trim(y: np.ndarray, sr: int, **kwargs
             ) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Trim to the first and last speech, keeping whatever is between them.

    Returns ``(trimmed, (start_sample, end_sample))``, matching
    :func:`kudio.trim_silence`. A clip with no detected speech comes back
    untouched with its full span, rather than as an empty array — deleting the
    audio is never the safer reading of "nothing found".
    """
    spans = vad(y, sr, **kwargs)
    if not spans:
        log.info("vad_trim: no speech detected, returning the clip unchanged")
        return np.asarray(y), (0, int(np.asarray(y).shape[0]))
    start = int(round(spans[0][0] * sr))
    end = min(int(round(spans[-1][1] * sr)), np.asarray(y).shape[0])
    return np.asarray(y)[start:end], (start, end)


def speech_ratio(y: np.ndarray, sr: int, **kwargs) -> float:
    """Fraction of the clip that :func:`vad` calls speech, in ``[0, 1]``.

    Useful as a dataset filter: a "speech" file at 0.02 is a mislabelled
    recording of a room.
    """
    y = np.asarray(y)
    if y.size == 0:
        return 0.0
    total = sum(b - a for a, b in vad(y, sr, **kwargs))
    return float(min(1.0, total / (y.shape[0] / sr)))
