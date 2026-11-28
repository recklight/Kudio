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

The estimator is **pYIN** (Mauch & Dixon 2014): YIN with a probabilistic voicing
decision resolved by a hidden Markov model over the whole clip, rather than a
per-frame threshold. That last part is why it costs more than plain YIN -- about
an eighth of a second per ten seconds of 16 kHz audio on a laptop -- and why it does not flicker
between voiced and unvoiced twice per syllable the way a thresholded estimator
does.

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

from kudio._numbers import count, flag, real, within
from kudio._samples import finite, one_channel
from kudio.core import _yin
from kudio.core._policy import sample_rate, warn
from kudio.exceptions import FeatureError

__all__ = ['PitchTrack', 'f0']

log = logging.getLogger(__name__)

#: Default search range, in Hz: a low male voice to a high female one.
SPEECH_FMIN = 65.0
#: Top of the default search range, in Hz: a high female voice.
SPEECH_FMAX = 400.0

#: Semitones the moves of a hop may span before there is a warning: 24 ms. Of the
#: noise in a pause 3.2 frames in 100 are voiced then at 65 to 400 Hz, 4.8 at 100 to
#: 300 and 7.2 at 120 to 500; at 25 ms 5.6, 8.7 and 10.6, and at 50 ms more than 40.
#: It is the span that does it and not its share of the range, which at 25 ms is
#: 0.35, 0.58 and 0.45.
_TOO_FAR = 10


@dataclass(frozen=True)
class PitchTrack:
    """A pitch contour plus the voicing decision that makes it readable.

    :param f0: one value per frame, in Hz, ``NaN`` where unvoiced.
    :param voiced: the voicing decision, aligned with *f0*.
    :param voiced_prob: pYIN's confidence, aligned with *f0*.
    :param times: centre time of each frame, in seconds.
    :param length: the tracked signal's length in samples, which ends a run
        voiced to the last frame; ``None`` for a track built without it.
    """

    f0: np.ndarray
    voiced: np.ndarray
    voiced_prob: np.ndarray
    times: np.ndarray
    sr: int
    fmin: float
    fmax: float
    hop_length: int
    length: Optional[int] = None

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

    def voiced_segments(self, min_seconds: float = 0.05) -> List[Tuple[float, float]]:
        """Contiguous voiced runs as ``(start, end)`` in seconds.

        Runs shorter than *min_seconds* are dropped: at a 10 ms hop a single
        voiced frame in the middle of a fricative is an estimator hiccup, not
        a syllable.

        A run ends a hop after its last frame's centre, and one voiced to the
        last frame ends with the signal, :attr:`length` samples in; a hop
        after the last centre would be 3.010 s on a 3 s file.
        """
        min_seconds = within(min_seconds, 'min_seconds', 'PitchTrack.voiced_segments',
                             FeatureError, low=0, of=' of seconds')
        if not len(self):
            return []
        hop = self.hop_length / self.sr
        end = None if self.length is None else self.length / self.sr
        segments: List[Tuple[float, float]] = []
        start: Optional[int] = None
        for i, on in enumerate(np.append(self.voiced, False)):
            if on and start is None:
                start = i
            elif not on and start is not None:
                a, b = self.times[start], self.times[i - 1] + hop
                if end is not None:
                    b = min(b, end)
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
       fmax: float = SPEECH_FMAX, frame_length: Optional[int] = None,
       hop_length: Optional[int] = None,
       center: bool = True) -> PitchTrack:
    """Track the fundamental frequency of *y* with pYIN.

    Not all of a pause is unvoiced. Of the frames of noise between words, at the
    default hop, so many are voiced, and the wider the range the more:

    ===========  ======
    range, Hz    voiced
    ===========  ======
    100 to 300   0 %
    65 to 400    2 %
    60 to 600    6 %
    50 to 1000   12 %
    ===========  ======

    :param fmin: bottom of the search range, in Hz. Nothing below it is
        found; a voice below it is reported as something wrong above it.
    :param fmax: top of the search range. The period of a pitch within half a
        sample of *fmax*'s is at the end of what is searched, and the pitch
        comes back an octave low: of 400 Hz, from 390 up at 8 kHz, from 395 at
        16 kHz and from 398 at 48 kHz. Leave it some room.
    :param frame_length: analysis window. Must hold more than two periods of
        *fmin*, so a low *fmin* needs a long window -- checked, not assumed.
        Omitted, it is 2048 samples, or the next power of two that is long
        enough when 2048 is not: 4096 at 88.2 and 96 kHz.
    :param hop_length: samples between frames; 10 ms when omitted, which is
        the usual resolution for a pitch contour. A longer one is not the same
        answer at fewer places: voicing is told by how little a pitch moves
        from a frame to the next, and from about 24 ms up, which a warning
        says, the more of a pause comes back voiced the longer the hop.
    :param center: frames centred on their time stamp, so :attr:`times` lines
        up with the waveform. Turn it off to match an uncentred transform.
    :returns: a :class:`PitchTrack`. A pitch has to come round twice to be told
        from a lower one, so in a clip shorter than two periods of *fmin*, 493
        samples at 16 kHz, a frame whose pitch has fewer than two in it is
        unvoiced, and a warning gives the lowest that can be told.
    :raises FeatureError: for a clip of more than one channel or with NaN in it,
        for a range, hop or frame that is not one, for a frame that does not
        hold two periods of *fmin*, for a hop in which a pitch could cross the
        whole range, 73 ms of 65 to 400 Hz, and with ``center=False`` for a
        clip shorter than a frame.

    >>> track = kudio.f0(y, 16000, fmin=80, fmax=300)      # doctest: +SKIP
    >>> track.median_hz                                    # doctest: +SKIP
    124.7
    """
    center = flag(center, 'center', 'f0', FeatureError)
    sr = sample_rate(sr, 'f0', FeatureError)
    y = np.asarray(one_channel(y, 'f0', FeatureError))
    if y.ndim != 1:
        # reshape(-1) used to read a stereo clip as one interleaved signal at
        # half its pitch: fan.wav came back at 65 Hz instead of 126
        raise FeatureError(f"f0 takes mono audio, got shape {y.shape}; pass "
                           f"one channel or mix down first")
    with np.errstate(over='ignore'):        # a float64 past float32 is inf in it
        y = finite(y, 'f0', FeatureError).astype(np.float32, copy=False)
    finite(y, 'f0', FeatureError)
    low, high = real(fmin), real(fmax)
    if low is None or high is None or not 0 < low < high:
        raise FeatureError(
            f"need 0 < fmin < fmax, got fmin={fmin!r}, fmax={fmax!r}")
    fmin, fmax = low, high
    if fmax > sr / 2:
        raise FeatureError(
            f"fmax={fmax} Hz is above Nyquist for sr={sr}; nothing up there "
            f"survived sampling.")
    if hop_length is None:
        hop = max(1, int(sr // 100))
    else:
        hop = count(hop_length, 'hop_length', 'f0', FeatureError,
                    ' of samples, or None for 10 ms')
    # pYIN correlates a frame against itself one period later, so the frame
    # has to hold more than two periods of the lowest pitch it is asked for.
    # Short of that, the pitches whose periods do not fit are lost and the
    # rest look fine, which reads as "nothing down there" rather than as a
    # settings mistake.
    needed = 2 * int(np.floor(sr / fmin)) + 2
    if frame_length is None:
        frame_length = max(2048, 1 << int(np.ceil(np.log2(needed))))
    else:
        frame_length = count(frame_length, 'frame_length', 'f0', FeatureError, ' of samples')
    if frame_length < needed:
        raise FeatureError(
            f"frame_length={frame_length} is too short for fmin={fmin:g} Hz at "
            f"sr={sr}: two periods need {needed} samples. Raise frame_length "
            f"or raise fmin.")

    n_bins, reach = _yin.states(sr, fmin, fmax, hop)
    if 2 * reach + 1 > n_bins:
        # voicing is told by how little a pitch moves. At a hop of 100 ms, over
        # 65 to 400 Hz, 86 frames of white noise in 100 would be voiced.
        raise FeatureError(f"f0(): what a pitch can move by in a hop of {hop} samples spans "
                           f"{(2 * reach + 1) / _yin.FINE:g} semitones, and {fmin:g}-{fmax:g} "
                           f"Hz only {n_bins / _yin.FINE:g}; shorten the hop or widen the range")
    if len(y) and not center and len(y) < frame_length:
        raise FeatureError(f"f0(): the clip has {len(y)} samples, fewer than the one frame of "
                           f"{frame_length} that center=False needs")

    if len(y):
        if 2 * reach > _TOO_FAR * _yin.FINE:
            limit = (_TOO_FAR + 0.5) / (12 * _yin.SPEED)
            warn(f"f0(): the hop is {hop} samples, {1000 * hop / sr:.3g} ms, and voicing is told "
                 f"by how little a pitch moves in one: from about {1000 * limit:.0f} ms up, the "
                 f"longer the hop the more of a pause comes back voiced")
        values, voiced, chance = _yin.track(y, sr, fmin, fmax, frame_length, hop, bool(center))
        least = 2 * sr / len(y)
        if least > fmin:
            # ten samples of anything came back as fmin, voiced
            untold = voiced & (values < least)
            values[untold], voiced[untold] = np.nan, False
            warn(f"f0(): the clip has {len(y)} sample(s) and two periods of fmin={fmin:g} Hz "
                 f"are {int(np.ceil(2 * sr / fmin))}, so no pitch " +
                 (f"under {least:.4g} Hz " if least < fmax else '') + "is told in it")
    else:
        values, voiced, chance = np.zeros(0), np.zeros(0, dtype=bool), np.zeros(0)
    # a frame that is not centred on its time stamp is stamped in its middle
    middle = 0 if center else frame_length // 2
    times = (np.arange(len(values)) * hop + middle) / float(sr)
    return PitchTrack(values, voiced, chance, times, sr, fmin, fmax, hop, len(y))
