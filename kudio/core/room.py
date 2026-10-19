# -*- coding: utf-8 -*-
"""Rooms: describe one, apply it, measure what it did.

Noise is *additive*: :class:`kudio.Synthesizer` mixes it at a chosen SNR, and
:func:`kudio.spectral_enhance` takes it back out. Reverberation is the other
kind, and it is **convolutive**: the room does not add anything, it smears what
was already there across time. This module describes a room, puts a clip in
it, and measures one from its impulse response.

>>> ir = kudio.rir(16000, rt60=0.6, drr_db=6.0, seed=0)
>>> wet = kudio.apply_rir(speech, ir)
>>> print(kudio.rt60(ir, 16000))                 # doctest: +SKIP
T30 0.60 s (EDT 0.55 s)  ·  C50 +11.7 dB  ·  DRR +6.0 dB  ·  fit 1.000

**Removing it is a different problem and is not here.** Spectral subtraction
has nothing to subtract, and the single-channel predictive methods did not
survive measurement. What this module is for is *making* controlled
reverberation, for augmentation, and *measuring* it when you have an impulse
response.

Everything here is mono: a ``(frames, channels)`` array is refused rather than
read as one interleaved signal twice as long.

**The room here is a statistical one, not a real one.** Exponentially decaying
noise is the textbook model of a diffuse tail and it is what an augmentation
pipeline wants: one knob, reproducible, and the answer known in advance. It has
no early-reflection pattern, so it will not stand in for a measured impulse
response when the *shape* of the early energy is what matters. Feed
:func:`apply_rir` a real one when you have it.

**Every measurement says how much to believe it.** A decay curve that is not
straight -- too short a tail, noise reaching the floor, a recording that was
never impulsive -- produces a T30 anyway. :attr:`Reverberation.fit` is the R²
of the line that was fitted, and it is the difference between "this room rings
for 600 ms" and "this arithmetic ran".
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from kudio._samples import as_float
from kudio.exceptions import FeatureError

__all__ = ['Reverberation', 'rir', 'apply_rir', 'rt60', 'schroeder_curve',
           'MAX_EDT_RATIO']

log = logging.getLogger(__name__)

#: ISO 3382-1 evaluates the decay from -5 dB, to skip the direct sound, down to
#: -25 (T20) or -35 (T30); each is then extrapolated to a full 60 dB.
_T20_RANGE = (-5.0, -25.0)
_T30_RANGE = (-5.0, -35.0)
#: Early decay time uses the first 10 dB, which is what a listener hears as
#: "liveness" and often disagrees with T30 in a room that is not diffuse.
_EDT_RANGE = (0.0, -10.0)

#: Clarity splits early from late energy at these times, in milliseconds.
#: 50 ms is the speech figure, 80 ms the music one.
_C50_MS = 50.0
_C80_MS = 80.0

#: Half-width of the window counted as the direct sound, in milliseconds.
_DIRECT_MS = 2.5

#: How far EDT and T30 may disagree before the measurement stops counting as
#: one. A sanity bound, not a figure from the standard -- see
#: :meth:`Reverberation.reliable`.
MAX_EDT_RATIO = 2.0

_EPS = 1e-20


@dataclass(frozen=True)
class Reverberation:
    """What a room does, measured from its impulse response.

    :param t20: reverberation time from the -5 to -25 dB decay, in seconds.
    :param t30: the same from -5 to -35 dB. The usual figure quoted as "the
        RT60", and the one :attr:`seconds` returns.
    :param edt: early decay time -- the first 10 dB, extrapolated. Closer to
        what a listener calls liveness, and it disagrees with T30 in a room
        that is not diffuse.
    :param c50: clarity for speech, in dB: energy in the first 50 ms against
        everything after it. Positive means the direct sound wins.
    :param c80: the same split at 80 ms, the figure used for music.
    :param drr: direct-to-reverberant ratio in dB, taking +/-2.5 ms around the
        peak as the direct sound.
    :param fit: R² of the straight line fitted to the T30 decay. **Read this
        before quoting the rest.**
    """

    t20: float
    t30: float
    edt: float
    c50: float
    c80: float
    drr: float
    fit: float
    sr: int

    @property
    def seconds(self) -> float:
        """The reverberation time, in seconds -- T30 by convention."""
        return self.t30

    def reliable(self, minimum: float = 0.9,
                 edt_ratio: float = MAX_EDT_RATIO) -> bool:
        """Was this an impulse response, and was its decay straight?

        Two questions, because the R² alone answers only the second. Backward
        integration makes *any* signal decay monotonically, and over a long
        clip that decay can be quite straight -- this repo's speech fixture
        fits at R² 0.94 and comes back with a 3.9 second "reverberation time".

        What gives it away is that its EDT and T30 disagree by a factor of
        eleven. In a real room the first 10 dB and the 5-to-35 dB span describe
        the same tail and land within a few per cent of each other, so a large
        disagreement means the two spans were measuring different things and
        neither of them was a room. *edt_ratio* is a sanity bound and not a
        figure from the standard: over 40 seeds of ``rir(16000, 0.6)`` the
        ratio runs 0.87 to 1.16 at ``drr_db=6`` and 0.95 to 1.09 at 0, and
        published halls stay well inside 0.8 to 1.3.

        A close microphone has no EDT to compare: the direct sound alone
        carries the first 10 dB of the decay, so the EDT is ``nan``. That says
        nothing against the tail, and the decay is then judged on its fit
        alone.
        """
        if not (np.isfinite(self.fit) and self.fit >= minimum):
            return False
        if not np.isfinite(self.t30) or self.t30 <= 0:
            return False
        if not np.isfinite(self.edt):
            return True
        ratio = self.edt / self.t30
        return 1.0 / edt_ratio <= ratio <= edt_ratio

    def summary(self) -> dict:
        return {'t20': self.t20, 't30': self.t30, 'edt': self.edt,
                'c50': self.c50, 'c80': self.c80, 'drr': self.drr,
                'fit': self.fit}

    def __str__(self) -> str:
        if not np.isfinite(self.t30):
            return (f"no usable decay (fit {self.fit:.3f}) — the tail never "
                    f"falls 35 dB, so there is nothing to extrapolate")
        # No symbol: this string gets printed, and a legacy console cannot
        # encode a warning sign. The middle dot and the em dash it already
        # uses survive cp950; U+26A0 raises UnicodeEncodeError on the way out.
        note = ""
        if not self.reliable():
            straight = np.isfinite(self.fit) and self.fit >= 0.9
            note = ("  — but EDT and T30 disagree, so this may not be an "
                    "impulse response" if straight
                    else "  — but the decay is not straight, so read it with "
                         "care")
        edt = (f"EDT {self.edt:.2f} s" if np.isfinite(self.edt)
               else "no EDT: the direct sound alone falls 10 dB")
        return (f"T30 {self.t30:.2f} s ({edt})  ·  "
                f"C50 {self.c50:+.1f} dB  ·  DRR {self.drr:+.1f} dB  ·  "
                f"fit {self.fit:.3f}{note}")


def rir(sr: int, rt60: float = 0.5, *, drr_db: float = 0.0,
        seconds: Optional[float] = None,
        seed: Optional[int] = None) -> np.ndarray:
    """A synthetic room impulse response: exponentially decaying noise.

    :param rt60: the tail this room is asked for, in seconds -- the time the
        envelope takes to fall 60 dB.
    :param drr_db: how far the direct path sits above the reverberant energy,
        in dB. This is distance in a readable unit: a close microphone is
        +10 dB or more, a metre or two away is around 0, and the far end of a
        hall is negative. Solved for exactly, so :func:`rt60` measures back
        what was asked for.
    :param seconds: how long to make it. Defaults to 1.5x *rt60*, which holds
        the decay well past the 35 dB that T30 needs.
    :param seed: makes the room reproducible, the way
        :class:`kudio.Synthesizer` makes a mixture reproducible.

    >>> ir = kudio.rir(16000, rt60=0.6, drr_db=6.0, seed=0)
    >>> room = kudio.rt60(ir, 16000)
    >>> round(room.t30, 2), round(room.drr, 1)               # doctest: +SKIP
    (0.6, 6.0)
    """
    if sr <= 0:
        raise FeatureError(f"rir() needs a positive rate, got {sr}")
    if rt60 <= 0:
        raise FeatureError(f"rt60 must be positive, got {rt60}")

    length = float(seconds) if seconds else rt60 * 1.5
    n = int(round(length * sr))
    if n < 2:
        raise FeatureError(
            f"an impulse response of {length:g}s at {sr} Hz is {n} samples; "
            f"raise rt60 or pass a longer seconds=")

    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64) / sr
    # -60 dB of amplitude at t = rt60, which is what the name means
    envelope = 10.0 ** (-3.0 * t / rt60)
    tail = rng.standard_normal(n) * envelope
    tail[0] = 0.0                       # sample 0 belongs to the direct path

    # Solve for the direct sample that produces the DRR asked for, using the
    # same +/-2.5 ms split the measurement uses. Anything else would give a
    # knob whose number does not come back out.
    half = int(round(_DIRECT_MS / 1000.0 * sr))
    inside = float(np.sum(tail[1:half + 1] ** 2))
    outside = float(np.sum(tail[half + 1:] ** 2))
    wanted = outside * 10.0 ** (drr_db / 10.0)
    if wanted <= inside:
        # The early reflections inside the direct window already exceed the
        # ratio asked for; a room this diffuse cannot be built this way.
        raise FeatureError(
            f"drr_db={drr_db:g} is below what this tail already carries in its "
            f"first {_DIRECT_MS:g} ms "
            f"({10 * np.log10(max(inside, _EPS) / max(outside, _EPS)):.1f} dB). "
            f"Ask for a higher drr_db, or a longer rt60, whose tail puts less "
            f"of its energy inside that window.")
    tail[0] = float(np.sqrt(wanted - inside))
    return (tail / (np.max(np.abs(tail)) + _EPS)).astype(np.float32)


def apply_rir(y: np.ndarray, ir: np.ndarray, *,
              trim: bool = True, normalise: bool = True) -> np.ndarray:
    """Put *y* in the room described by *ir*.

    :param trim: return the same number of samples that went in. The
        convolution is longer by the length of the tail, and a clip that grows
        every time it is processed stops lining up with everything measured
        against it.
    :param normalise: keep the output's peak where the input's was. A room
        with a loud direct path has a gain, and an augmentation pipeline that
        also changes the level is measuring two things at once.
    """
    y = _mono("apply_rir()", y, "audio")
    ir = _mono("apply_rir()", ir, "impulse response")
    if y.size == 0 or ir.size == 0:
        raise FeatureError("apply_rir() needs audio and an impulse response, "
                           f"got {y.size} and {ir.size} samples")

    from scipy.signal import fftconvolve
    wet = fftconvolve(y, ir)
    if trim:
        wet = wet[:len(y)]
    if normalise:
        peak_in = float(np.max(np.abs(y)))
        peak_out = float(np.max(np.abs(wet)))
        if peak_out > 0 and peak_in > 0:
            wet = wet * (peak_in / peak_out)
    return wet.astype(np.float32)


def schroeder_curve(ir: np.ndarray) -> np.ndarray:
    """Backward-integrated energy decay of *ir*, in dB, starting at 0.

    Schroeder's method: the decay a room would show if you averaged infinitely
    many noise bursts, obtained from one impulse by integrating its energy from
    the end backwards. It is what makes a single measurement readable.
    """
    ir = _mono("schroeder_curve()", ir, "impulse response")
    if ir.size == 0:
        raise FeatureError("schroeder_curve() needs an impulse response")
    energy = np.cumsum((ir ** 2)[::-1])[::-1]
    total = energy[0]
    if total <= 0:
        return np.full(len(ir), -np.inf)
    with np.errstate(divide='ignore'):
        return 10.0 * np.log10(np.maximum(energy / total, _EPS))


def _mono(where: str, x, what: str) -> np.ndarray:
    """*x* as a 1-D float64 array, refusing a ``(frames, channels)`` one.

    Flattened, it would be one interleaved signal: a stereo clip would come
    out of apply_rir twice as long, and a two-column copy of a 0.6 s response
    would measure 1.2 s with a perfect fit.
    """
    x = np.asarray(as_float(x), dtype=np.float64)
    if x.ndim > 1 and x.shape[1] == 1:
        x = x[:, 0]
    if x.ndim != 1:
        raise FeatureError(f"{where} takes mono {what}, got shape {x.shape}; "
                           f"pass one channel at a time")
    return x


def _onset(ir: np.ndarray) -> int:
    """Where the response starts: the first sample within 20 dB of its peak,
    ISO 3382-1's rule for the arrival of the direct sound."""
    level = np.abs(ir)
    return int(np.argmax(level >= 0.1 * level.max())) if level.max() > 0 else 0


def _decay_time(curve: np.ndarray, sr: int,
                span: Tuple[float, float]) -> Tuple[float, float]:
    """Seconds to fall 60 dB, extrapolated from the *span* of the curve.

    Returns ``(seconds, r_squared)``. The R² is the point: a decay that is not
    a straight line still has a best-fit slope, and the slope of a curve that
    was never straight is a number about nothing.
    """
    top, bottom = span
    start = int(np.argmax(curve <= top)) if np.any(curve <= top) else 0
    if not np.any(curve <= bottom):
        return float('nan'), float('nan')
    stop = int(np.argmax(curve <= bottom))
    if stop <= start + 1:
        return float('nan'), float('nan')

    x = np.arange(start, stop, dtype=np.float64) / sr
    segment = curve[start:stop]
    slope, intercept = np.polyfit(x, segment, 1)
    if slope >= 0:
        return float('nan'), float('nan')

    predicted = slope * x + intercept
    spread = float(np.sum((segment - segment.mean()) ** 2))
    r2 = 1.0 - float(np.sum((segment - predicted) ** 2)) / spread \
        if spread > 0 else float('nan')
    return float(-60.0 / slope), r2


def _clarity(ir: np.ndarray, sr: int, split_ms: float) -> float:
    """Early energy against late, in dB, split *split_ms* after the peak."""
    peak = int(np.argmax(np.abs(ir)))
    edge = peak + int(round(split_ms / 1000.0 * sr)) + 1
    early = float(np.sum(ir[peak:edge] ** 2))
    late = float(np.sum(ir[edge:] ** 2))
    if late <= 0:
        return float('inf')
    return 10.0 * np.log10(max(early, _EPS) / late)


def rt60(ir: np.ndarray, sr: int) -> Reverberation:
    """Measure a room from its impulse response, to ISO 3382-1.

    T20, T30 and EDT are read off one backward-integrated decay curve
    (:func:`schroeder_curve`); C50, C80 and DRR are energy ratios taken from
    the response around its peak. The response is measured from its onset,
    the first sample within 20 dB of the peak, so leading silence -- time of
    flight, converter latency -- is not counted as decay.

    It does not find or remove a noise floor. A recorded response whose tail
    reaches the floor less than 35 dB down has no honest T30, and
    :meth:`Reverberation.reliable` cannot tell: check the raw response first.

    >>> room = kudio.rt60(ir, 16000)
    >>> room.seconds, room.reliable()                        # doctest: +SKIP
    (0.6, True)
    """
    if sr <= 0:
        raise FeatureError(f"rt60() needs a positive rate, got {sr}")
    ir = _mono("rt60()", ir, "impulse response")
    if ir.size < 2:
        raise FeatureError(f"rt60() needs an impulse response, got {ir.size} "
                           f"sample(s)")
    ir = ir[_onset(ir):]

    curve = schroeder_curve(ir)
    t20, _ = _decay_time(curve, sr, _T20_RANGE)
    t30, fit = _decay_time(curve, sr, _T30_RANGE)
    edt, _ = _decay_time(curve, sr, _EDT_RANGE)

    peak = int(np.argmax(np.abs(ir)))
    half = int(round(_DIRECT_MS / 1000.0 * sr))
    lo, hi = max(0, peak - half), min(len(ir), peak + half + 1)
    direct = float(np.sum(ir[lo:hi] ** 2))
    reverberant = float(np.sum(ir ** 2) - direct)
    drr = (10.0 * np.log10(max(direct, _EPS) / reverberant)
           if reverberant > 0 else float('inf'))

    return Reverberation(t20=t20, t30=t30, edt=edt,
                         c50=_clarity(ir, sr, _C50_MS),
                         c80=_clarity(ir, sr, _C80_MS),
                         drr=drr, fit=fit, sr=int(sr))
