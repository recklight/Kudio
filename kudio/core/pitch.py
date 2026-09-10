# -*- coding: utf-8 -*-
"""Fundamental frequency, and the voiced/unvoiced decision that goes with it.

A spectrogram shows *where* the energy is; it does not say what note the
speaker is on, and reading harmonic spacing off an image is guesswork the
moment the fundamental is weak -- which it is over a phone, through a
high-pass, or on any male voice whose f0 sits below the first bin anyone
bothers to draw.

>>> track = kudio.f0(y, sr)
>>> track.median_hz, track.voiced_ratio
(118.3, 0.61)
>>> track.f0[track.voiced].min()                          # doctest: +SKIP
94.2

**Half of this is the voicing decision.** An estimator asked "what is the
pitch of this frame" always answers, including for silence, for a door
slamming and for the ``s`` in *this* -- and a contour drawn through those
answers is a picture of noise with a line through it. :attr:`PitchTrack.f0`
is ``NaN`` wherever the frame is unvoiced, so plotting it leaves gaps where
there genuinely is no pitch, and :attr:`~PitchTrack.voiced_ratio` says how
much of the clip had one at all.

The estimator is **pYIN** (Mauch & Dixon 2014), librosa's implementation: YIN
with a probabilistic voicing decision resolved by a hidden Markov model over
the whole clip, rather than a per-frame threshold. That last part is why it
costs what it costs -- roughly a second of compute per ten seconds of audio at
16 kHz -- and why it does not flicker between voiced and unvoiced twice per
syllable the way a thresholded estimator does.

**Defaults are speech.** ``fmin=65`` to ``fmax=400`` Hz covers a low male
voice through a high female one. Music, children and singing all want a wider
range said out loud -- an octave outside the range is not detected, it is
reported as the nearest wrong answer inside it.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from kudio.exceptions import FeatureError

__all__ = ['PitchTrack', 'f0']

log = logging.getLogger(__name__)

#: Default search range, in Hz: a low male voice to a high female one.
SPEECH_FMIN = 65.0
SPEECH_FMAX = 400.0


@dataclass(frozen=True)
class PitchTrack:
    """A pitch contour plus the voicing decision that makes it readable.

    :param f0: one value per frame, in Hz, ``NaN`` where unvoiced.
    :param voiced: the voicing decision, aligned with *f0*.
    :param voiced_prob: pYIN's confidence, aligned with *f0*.
    :param times: centre time of each frame, in seconds.
    """

    f0: np.ndarray
    voiced: np.ndarray
    voiced_prob: np.ndarray
    times: np.ndarray
    sr: int
    fmin: float
    fmax: float
    hop_length: int

    def __len__(self) -> int:
        return int(len(self.f0))

    # ---------------------------------------------------------------- summary

    @property
    def voiced_ratio(self) -> float:
        """Fraction of frames with a pitch. Not the same as speech: an ``f``
        or an ``s`` is speech and has no pitch."""
        return float(np.mean(self.voiced)) if len(self) else 0.0

    @property
    def median_hz(self) -> float:
        """Median f0 over voiced frames; ``nan`` when nothing was voiced.

        The median and not the mean: an octave error is a factor of two, and
        one of those moves a mean far enough to matter.
        """
        values = self.f0[self.voiced]
        return float(np.median(values)) if values.size else float('nan')

    def range_hz(self, low: float = 5.0, high: float = 95.0) -> Tuple[float, float]:
        """Percentile range of the voiced f0, in Hz.

        Percentiles rather than min/max, because a single misjudged frame at
        the octave sets both extremes.
        """
        values = self.f0[self.voiced]
        if not values.size:
            return (float('nan'), float('nan'))
        lo, hi = np.percentile(values, (low, high))
        return (float(lo), float(hi))

    @property
    def semitone_range(self) -> float:
        """The same span in semitones -- the unit pitch is actually heard in.

        60 Hz of range means something different at the bottom of a male voice
        than at the top of a female one; twelve semitones is an octave either
        way.
        """
        lo, hi = self.range_hz()
        if not (np.isfinite(lo) and np.isfinite(hi)) or lo <= 0:
            return float('nan')
        return float(12.0 * np.log2(hi / lo))

    def summary(self) -> dict:
        """Everything above as plain numbers, for a report or a CSV row."""
        lo, hi = self.range_hz()
        return {
            'frames': len(self),
            'voiced_ratio': self.voiced_ratio,
            'median_hz': self.median_hz,
            'low_hz': lo,
            'high_hz': hi,
            'semitone_range': self.semitone_range,
        }

    def __str__(self) -> str:
        if not self.voiced_ratio:
            return (f"no voiced frames in {len(self)} "
                    f"({self.fmin:g}-{self.fmax:g} Hz)")
        lo, hi = self.range_hz()
        return (f"median {self.median_hz:.1f} Hz  ·  "
                f"{lo:.1f}-{hi:.1f} Hz ({self.semitone_range:.1f} semitones)  ·  "
                f"{self.voiced_ratio:.0%} voiced")

    # --------------------------------------------------------------- segments

    def voiced_segments(self, min_seconds: float = 0.05
                        ) -> List[Tuple[float, float]]:
        """Contiguous voiced runs as ``(start, end)`` in seconds.

        Runs shorter than *min_seconds* are dropped: at a 10 ms hop a single
        voiced frame in the middle of a fricative is an estimator hiccup, not
        a syllable.
        """
        if not len(self):
            return []
        hop = self.hop_length / self.sr
        segments: List[Tuple[float, float]] = []
        start: Optional[int] = None
        for i, on in enumerate(np.append(self.voiced, False)):
            if on and start is None:
                start = i
            elif not on and start is not None:
                a, b = self.times[start], self.times[i - 1] + hop
                if b - a >= min_seconds:
                    segments.append((float(a), float(b)))
                start = None
        return segments

    def to_labels(self, min_seconds: float = 0.05) -> List:
        """The voiced runs as :class:`kudio.Label` objects.

        Saveable with :func:`kudio.save_labels`, so a pitch track can be
        opened as a label track in an editor rather than only read as numbers.
        """
        from kudio.core.dataset import Label
        return [Label(start=a, end=b, text=f"voiced {i + 1}")
                for i, (a, b) in enumerate(self.voiced_segments(min_seconds))]


def f0(y: np.ndarray, sr: int, *, fmin: float = SPEECH_FMIN,
       fmax: float = SPEECH_FMAX, frame_length: int = 2048,
       hop_length: Optional[int] = None,
       center: bool = True) -> PitchTrack:
    """Track the fundamental frequency of *y* with pYIN.

    :param fmin: bottom of the search range, in Hz. Nothing below it is
        found; a voice below it is reported as something wrong above it.
    :param fmax: top of the search range.
    :param frame_length: analysis window. Must hold at least two periods of
        *fmin*, so a low *fmin* needs a long window -- checked, not assumed.
    :param hop_length: samples between frames; 10 ms when omitted, which is
        the usual resolution for a pitch contour.
    :param center: frames centred on their time stamp, so :attr:`times` lines
        up with the waveform. Turn it off to match an uncentred transform.
    :returns: a :class:`PitchTrack`.

    >>> track = kudio.f0(y, 16000, fmin=80, fmax=300)      # doctest: +SKIP
    >>> track.median_hz                                    # doctest: +SKIP
    124.7
    """
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    sr = int(sr)
    if sr <= 0:
        raise FeatureError(f"f0 needs a positive rate, got {sr}")
    if not (0 < fmin < fmax):
        raise FeatureError(
            f"need 0 < fmin < fmax, got fmin={fmin}, fmax={fmax}")
    if fmax > sr / 2:
        raise FeatureError(
            f"fmax={fmax} Hz is above Nyquist for sr={sr}; nothing up there "
            f"survived sampling.")
    hop = int(hop_length) if hop_length else max(1, sr // 100)
    # pYIN correlates a frame against itself one period later, so the frame
    # has to be at least two periods of the lowest pitch it is asked for.
    # Getting this wrong silently returns "unvoiced everywhere", which reads
    # as "this recording has no pitch" rather than as a settings mistake.
    needed = int(np.ceil(2 * sr / fmin))
    if frame_length < needed:
        raise FeatureError(
            f"frame_length={frame_length} is too short for fmin={fmin:g} Hz at "
            f"sr={sr}: two periods need {needed} samples. Raise frame_length "
            f"or raise fmin.")

    if y.size == 0:
        empty_f = np.zeros(0, dtype=np.float32)
        return PitchTrack(empty_f, np.zeros(0, dtype=bool), empty_f, empty_f,
                          sr, float(fmin), float(fmax), hop)

    import librosa
    values, voiced, prob = librosa.pyin(
        y, fmin=float(fmin), fmax=float(fmax), sr=sr,
        frame_length=int(frame_length), hop_length=hop, center=center)
    values = np.asarray(values, dtype=np.float64)
    voiced = np.asarray(voiced, dtype=bool)
    # pYIN can hand back a number on a frame it also calls unvoiced; keeping
    # both would let a caller plot the contour and get a line through silence.
    values = np.where(voiced, values, np.nan)
    # n_fft is the *uncentred* correction: it shifts a frame index to the
    # middle of the window it covers. A centred frame is already there, so
    # passing it in that case would move every time stamp half a frame late.
    times = librosa.times_like(values, sr=sr, hop_length=hop,
                               n_fft=None if center else int(frame_length))
    return PitchTrack(values, voiced, np.asarray(prob, dtype=np.float64),
                      np.asarray(times, dtype=np.float64), sr,
                      float(fmin), float(fmax), hop)
