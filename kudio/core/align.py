# -*- coding: utf-8 -*-
"""Line two recordings up before measuring one against the other.

Every reference metric in kudio -- :func:`kudio.si_sdr`, :func:`kudio.snr`,
:func:`kudio.segmental_snr`, PESQ, STOI -- compares sample *i* of one signal
with sample *i* of the other. They trim to the shorter of the two and assume
the rest lines up, which is right for a process that returns what it was given
and silently wrong for anything that introduces a delay.

How wrong is worth seeing. A clip scored against **itself**, shifted:

=========  ==========
shift      SI-SDR
=========  ==========
0          +145 dB
1 sample   +10.6 dB
1 ms       -10.1 dB
10 ms      -11.1 dB
=========  ==========

One sample costs 134 dB. A millisecond scores worse than not processing at
all, and nothing in the number says "these are misaligned" rather than "your
denoiser destroyed the signal" -- which is the wrong conclusion to hand
somebody about work that may be fine.

>>> shift = kudio.find_delay(clean, processed, sr)
>>> print(shift)                                          # doctest: +SKIP
processed lags by 128 samples (8.0 ms), correlation 0.998
>>> a, b = kudio.align(clean, processed, sr)
>>> kudio.si_sdr(a, b)                                    # doctest: +SKIP
18.4

**The confidence is half of it.** Cross-correlation always has a maximum, so
it always returns a delay -- including for two recordings that have nothing to
do with each other. :attr:`Alignment.correlation` is what separates "these are
the same take, 8 ms apart" from "these are different recordings and the number
means nothing", and :func:`align` refuses rather than lining up two unrelated
signals and letting a metric put a number on the result.

**Polarity is reported, not silently fixed.** An inverted copy correlates at
-1: it is the same take, and something in the chain flipped it. That is a
finding, so it is named rather than corrected on the way past.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from kudio.exceptions import FeatureError

__all__ = ['Alignment', 'find_delay', 'align', 'MIN_CORRELATION']

log = logging.getLogger(__name__)

#: Below this, two signals are not the same recording and the delay between
#: them does not mean anything. Chosen low: real pairs -- a denoised take
#: against its source, a resampled copy -- sit well above 0.8 even when the
#: processing is aggressive, and the point of the threshold is to catch
#: "somebody picked the wrong file", not to grade the processing.
MIN_CORRELATION = 0.5


@dataclass(frozen=True)
class Alignment:
    """Where two recordings sit relative to each other.

    :param delay: samples by which *deg* sits **after** *ref*. Negative means
        it starts first. Zero means they already line up.
    :param correlation: normalised cross-correlation at that delay, in
        ``[-1, 1]``. Negative means one of them is polarity-inverted.
    :param overlap: samples the two share once the delay is taken out.
    """

    delay: int
    correlation: float
    sr: int
    overlap: int

    @property
    def seconds(self) -> float:
        return self.delay / self.sr if self.sr else 0.0

    @property
    def milliseconds(self) -> float:
        return self.seconds * 1000.0

    @property
    def inverted(self) -> bool:
        """Is one of the two polarity-flipped relative to the other?"""
        return self.correlation < 0

    @property
    def strength(self) -> float:
        """How alike they are, ignoring polarity: ``abs(correlation)``."""
        return abs(self.correlation)

    def confident(self, minimum: float = MIN_CORRELATION) -> bool:
        """Are these plausibly the same recording?"""
        return self.strength >= minimum

    def apply(self, ref: np.ndarray,
              deg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Trim both signals to the region they share at this delay.

        Separate from :func:`find_delay` so a delay measured once -- on a loud
        passage, say -- can be applied to a whole set of files that came
        through the same chain.
        """
        ref = np.asarray(ref)
        deg = np.asarray(deg)
        if self.delay > 0:
            deg = deg[self.delay:]
        elif self.delay < 0:
            ref = ref[-self.delay:]
        n = min(len(ref), len(deg))
        return ref[:n], deg[:n]

    def __str__(self) -> str:
        if self.delay == 0:
            where = "already aligned"
        else:
            side = "lags" if self.delay > 0 else "leads"
            n = abs(self.delay)
            where = (f"{side} by {n} sample{'' if n == 1 else 's'} "
                     f"({abs(self.milliseconds):.1f} ms)")
        note = ", polarity inverted" if self.inverted else ""
        return f"{where}, correlation {self.strength:.3f}{note}"


def _mono(name: str, y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim > 1:
        raise FeatureError(
            f"{name} takes mono audio, got shape {y.shape}. Mix down first "
            f"(y.mean(axis=1)) -- which channel led is a question of its own.")
    return y.astype(np.float64, copy=False).reshape(-1)


def find_delay(ref: np.ndarray, deg: np.ndarray, sr: int, *,
               max_seconds: Optional[float] = None) -> Alignment:
    """Measure how far *deg* sits after *ref*, to the nearest sample.

    :param max_seconds: bound the search to this much delay either way.
        Worth setting when you know the answer is small: an unbounded search
        over a long file can find a better-correlating match half a minute
        away in something as repetitive as speech, and be perfectly right
        about a passage nobody meant.
    :returns: an :class:`Alignment`. It always returns one -- ask it whether
        to believe it.

    Both signals must already be at *sr*; loudness and pitch survive a
    resample and a sample offset does not, so aligning two rates would be
    measuring a delay in units that do not exist yet.
    """
    a = _mono("find_delay()", ref)
    b = _mono("find_delay()", deg)
    if sr <= 0:
        raise FeatureError(f"find_delay() needs a positive rate, got {sr}")
    if a.size == 0 or b.size == 0:
        raise FeatureError("find_delay() needs audio on both sides, got "
                           f"{a.size} and {b.size} samples")

    # A DC offset correlates with everything, including a step of silence, and
    # would drag the peak toward whichever lag overlaps most.
    a = a - a.mean()
    b = b - b.mean()

    from scipy.signal import correlate, correlation_lags
    corr = correlate(b, a, mode="full", method="fft")
    lags = correlation_lags(len(b), len(a), mode="full")

    if max_seconds is not None:
        limit = int(round(max_seconds * sr))
        if limit < 0:
            raise FeatureError(f"max_seconds must be >= 0, got {max_seconds}")
        keep = np.abs(lags) <= limit
        if not keep.any():                     # limit smaller than one sample
            keep = lags == 0
        corr, lags = corr[keep], lags[keep]

    delay = int(lags[int(np.argmax(np.abs(corr)))])

    # Normalise over the region the two actually share at that delay, rather
    # than over their whole lengths: a genuine match that only overlaps for a
    # second would otherwise score as badly as a mismatch.
    left, right = (a[-delay:], b) if delay < 0 else (a, b[delay:])
    n = min(len(left), len(right))
    left, right = left[:n], right[:n]
    energy = float(np.sqrt(np.dot(left, left) * np.dot(right, right)))
    correlation = float(np.dot(left, right) / energy) if energy > 0 else 0.0
    return Alignment(delay=delay, correlation=correlation, sr=int(sr),
                     overlap=int(n))


def align(ref: np.ndarray, deg: np.ndarray, sr: int, *,
          max_seconds: Optional[float] = None,
          min_correlation: float = MIN_CORRELATION
          ) -> Tuple[np.ndarray, np.ndarray]:
    """Return *ref* and *deg* trimmed to the region they share, lined up.

    The pair comes back ready for :func:`kudio.si_sdr` and friends.

    :param min_correlation: refuse below this, because two unrelated
        recordings still have a best lag and lining them up at it produces a
        confident-looking number about nothing. Pass ``0`` to align whatever
        you have anyway.
    :raises FeatureError: when the two do not look like the same recording.

    >>> a, b = kudio.align(clean, processed, sr)           # doctest: +SKIP
    >>> kudio.si_sdr(a, b)                                 # doctest: +SKIP
    18.4
    """
    found = find_delay(ref, deg, sr, max_seconds=max_seconds)
    if not found.confident(min_correlation):
        raise FeatureError(
            f"these do not look like the same recording: best correlation "
            f"{found.strength:.3f} at {found.delay} samples, below "
            f"{min_correlation:g}. Aligning them would put a number on two "
            f"unrelated signals; pass min_correlation=0 to do it anyway.")
    if found.inverted:
        log.warning("align: the two are polarity-inverted relative to each "
                    "other (correlation %.3f); lining them up as they are",
                    found.correlation)
    return found.apply(ref, deg)
