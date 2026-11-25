# -*- coding: utf-8 -*-
"""Narrowband lines in a recording: hum, a converter's clock, a whine.

:func:`kudio.audio_report` reads a clip broadband -- level, noise floor, where
the content stops -- and a line 40 dB above its neighbourhood moves none of
those numbers much. It is still the defect a machine or a switching supply
leaves behind. :func:`tonal_lines` finds such lines, and groups those that sit
on whole multiples of one frequency into a comb, the way mains hum and most
electrical interference look.

A steady pitch is a comb as well: a held note, a machine's tone, a synthetic
voice at one pitch. Speech moves its pitch, which smears its harmonics in a
long average, so it seldom shows up; a sustained tone always does. The result
says which frequencies stand out, not why.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from kudio._samples import one_channel
from kudio.core._policy import sample_rate
from kudio.exceptions import FeatureError

__all__ = ['Line', 'Comb', 'TonalLines', 'tonal_lines']

#: The running median a line is measured against spans this many bins either
#: side: 50 at the default 5 Hz resolution is 250 Hz, which holds ten lines of
#: a 50 Hz hum and still has more background than line in it.
_BACKGROUND_BINS = 50
#: A comb needs this many members...
_COMB_MIN = 3


@dataclass(frozen=True)
class Line:
    """One line of :class:`TonalLines`."""

    #: frequency, refined between bins by a parabola through the peak
    hz: float
    #: the amplitude of a sine at that frequency, in dB under full scale:
    #: 0 for a full-scale sine. The parabola takes back most of the 1.42 dB a
    #: Hann window loses between two bins; docs/validation.md has the error
    level_dbfs: float
    #: how far the line stands above the running median of the spectrum
    #: around it; 3 dB more for each halving of ``resolution_hz`` when the
    #: background is noise
    prominence_db: float


@dataclass(frozen=True)
class Comb:
    """Lines on whole multiples of one frequency."""

    #: fitted to all the members, so it is finer than one bin
    fundamental_hz: float
    #: the multiple each member sits on, 1 for the fundamental itself
    harmonics: Tuple[int, ...]
    lines: Tuple[Line, ...]

    def __str__(self) -> str:
        return (f"{self.fundamental_hz:.2f} Hz comb, {len(self.lines)} lines "
                f"(harmonics {_spans(self.harmonics)})")


@dataclass(frozen=True)
class TonalLines:
    """What :func:`tonal_lines` found. Empty tuples when nothing stands out."""

    #: every line, by frequency
    lines: Tuple[Line, ...]
    #: the combs, the one with the most prominent member first
    combs: Tuple[Comb, ...]
    #: the bin width the spectrum was read at, in Hz
    resolution_hz: float
    min_prominence_db: float

    @property
    def strongest(self) -> Optional[Line]:
        """The most prominent line, or None."""
        return max(self.lines, key=lambda line: line.prominence_db, default=None)

    def __str__(self) -> str:
        if not self.lines:
            return (f"no line stands {self.min_prominence_db:g} dB above the "
                    f"spectrum around it")
        top = self.strongest
        text = (f"{len(self.lines)} line{'s' if len(self.lines) != 1 else ''}, the "
                f"strongest at {top.hz:.1f} Hz, {top.prominence_db:.0f} dB above "
                f"its neighbourhood")
        if self.combs:
            text += "; " + "; ".join(str(comb) for comb in self.combs)
        return text


def _spans(numbers) -> str:
    """1, 2, 3, 5 -> '1-3, 5'."""
    out, run = [], [numbers[0]]
    for n in numbers[1:]:
        if n == run[-1] + 1:
            run.append(n)
        else:
            out.append(run)
            run = [n]
    out.append(run)
    return ", ".join(f"{r[0]}-{r[-1]}" if len(r) > 2 else ", ".join(map(str, r))
                     for r in out)


def _combs(lines: List[Line], tolerance) -> Tuple[Comb, ...]:
    """Group *lines* into combs, greedily, the best-supported first.

    The candidates are the lines themselves and the gaps between neighbours,
    so a comb whose fundamental is missing (hum through a speaker that cannot
    reproduce 50 Hz) is still found at 50 Hz. A candidate scores one for each
    line it explains and half against itself for each multiple from 1 to its
    highest member that has no line: 500 Hz explains 1000, 2000, 2500 and
    3000 Hz, but misses 500 and 1500, and ties with 1000 Hz, which explains
    three and misses none. Ties go to the higher fundamental, since every
    comb is also a comb at half its spacing.
    """
    left = sorted(lines, key=lambda line: line.hz)
    found = []
    while len(left) >= _COMB_MIN:
        freqs = np.array([line.hz for line in left])
        candidates = set(np.round(freqs, 6))
        candidates.update(np.round(np.diff(freqs), 6))
        best = None
        for f0 in sorted(candidates):
            if f0 <= 0:
                continue
            members, multiples = _members(freqs, f0, tolerance)
            if len(members) < _COMB_MIN:
                continue
            # refit on what matched, then match again
            f0 = float(np.sum(multiples * freqs[members]) / np.sum(multiples ** 2))
            members, multiples = _members(freqs, f0, tolerance)
            if len(members) < _COMB_MIN:
                continue
            missing = int(multiples.max()) - len(set(multiples.tolist()))
            score = len(members) - 0.5 * missing
            if best is None or (score, f0) > (best[0], best[1]):
                best = (score, f0, members, multiples)
        if best is None or best[0] < 2:
            break
        _, f0, members, multiples = best
        found.append(Comb(fundamental_hz=f0,
                          harmonics=tuple(int(k) for k in multiples),
                          lines=tuple(left[i] for i in members)))
        taken = set(members.tolist())
        left = [line for i, line in enumerate(left) if i not in taken]
    return tuple(sorted(found, key=lambda comb: -max(m.prominence_db for m in comb.lines)))


def _members(freqs: np.ndarray, f0: float, tolerance):
    """Indices of the lines within *tolerance* of a multiple of *f0*, and the
    multiples; one line per multiple, the closest.

    Only one unbroken run of multiples counts, broken where three in a row
    have no line, and the run with the most lines is kept. Otherwise a 50 Hz
    hum takes in a 1000 Hz comb as its 20th, 40th and 60th harmonics, loses
    half a point for each of the 46 multiples between, and a comb that does
    not exist wins instead.
    """
    multiples = np.round(freqs / f0)
    close = (multiples >= 1) & (np.abs(freqs - multiples * f0) <= tolerance(freqs))
    chosen = {}
    for i in np.nonzero(close)[0]:
        k = int(multiples[i])
        error = abs(freqs[i] - k * f0)
        if k not in chosen or error < chosen[k][1]:
            chosen[k] = (i, error)
    if not chosen:
        return np.array([], dtype=int), np.array([])
    ks = sorted(chosen)
    runs, run = [], [ks[0]]
    for k in ks[1:]:
        if k - run[-1] > 3:
            runs.append(run)
            run = []
        run.append(k)
    runs.append(run)
    best = max(runs, key=len)
    index = np.array(sorted(chosen[k][0] for k in best), dtype=int)
    return index, multiples[index]


def tonal_lines(y: np.ndarray, sr: int, *, min_prominence_db: float = 20.0,
                resolution_hz: float = 5.0, max_lines: int = 100) -> TonalLines:
    """The narrowband lines in *y*, and the combs they form.

    >>> found = kudio.tonal_lines(y, sr)                   # doctest: +SKIP
    >>> print(found)                                       # doctest: +SKIP
    10 lines, the strongest at 50.0 Hz, 52 dB above its neighbourhood; 50.00 Hz comb, 10 lines (harmonics 1-10)

    The spectrum is Welch's, Hann segments half overlapped, as long as the
    clip allows up to the power of two that gives *resolution_hz* or finer. A
    line is a local peak standing *min_prominence_db* above the running
    median of the 101 bins around it. White noise does not reach 20 dB above
    that median: read off a single segment, where a bin's power is spread
    widest, the chance for one bin is 2**-100.

    :param min_prominence_db: how far a peak has to stand above the spectrum
        around it to count.
    :param resolution_hz: the bin width to aim for. Finer splits lines that
        sit close together and lifts a line further above a noise floor, but
        needs a longer clip; a clip shorter than one segment is read at the
        resolution its length gives.
    :param max_lines: the most prominent this many are kept.
    :raises FeatureError: on non-mono audio, samples that are not finite, a
        rate that is not a positive number, a non-positive resolution.
    """
    sr = sample_rate(sr, 'tonal_lines', FeatureError)
    if not resolution_hz > 0:
        raise FeatureError(f"tonal_lines() needs a positive resolution_hz, got "
                           f"{resolution_hz}")
    y = one_channel(y, 'tonal_lines', FeatureError)
    y = np.ascontiguousarray(np.asarray(y, dtype=np.float64).squeeze())
    if y.ndim != 1:
        raise FeatureError(f"tonal_lines() takes mono audio, got shape {y.shape}")
    if not np.all(np.isfinite(y)):
        raise FeatureError(f"tonal_lines() got {int(np.count_nonzero(~np.isfinite(y)))} "
                           f"sample(s) that are not finite (NaN or inf); repair the "
                           f"clip first")

    wanted = 2 ** int(np.ceil(np.log2(sr / resolution_hz)))
    size = min(len(y), wanted)
    if size < 64 or not np.any(y):
        return TonalLines((), (), float(sr) / max(size, 1), float(min_prominence_db))

    from scipy.ndimage import median_filter
    from scipy.signal import find_peaks, welch
    freqs, power = welch(y, sr, nperseg=size, noverlap=size // 2, scaling='spectrum')
    db = 10.0 * np.log10(np.maximum(power, 1e-30))
    step = float(freqs[1])
    background = median_filter(db, 2 * _BACKGROUND_BINS + 1, mode='nearest')
    stand = db - background
    # DC and the bins against Nyquist hold no line worth naming
    stand[:3] = -np.inf
    stand[freqs > 0.99 * sr / 2.0] = -np.inf
    peaks, _ = find_peaks(stand, height=min_prominence_db, distance=3)

    lines = []
    for p in peaks:
        offset = 0.0
        if 0 < p < len(db) - 1:
            a, b, c = db[p - 1], db[p], db[p + 1]
            curve = a - 2 * b + c
            if curve < 0:
                offset = 0.5 * (a - c) / curve
        top = db[p] - 0.25 * (db[p - 1] - db[p + 1]) * offset if 0 < p < len(db) - 1 else db[p]
        lines.append(Line(hz=float(freqs[p] + offset * step),
                          # 'spectrum' scaling gives a sine of amplitude A as A**2 / 2
                          level_dbfs=float(top + 10.0 * np.log10(2.0)),
                          prominence_db=float(stand[p] + (top - db[p]))))
    if len(lines) > max_lines:
        lines = sorted(lines, key=lambda line: -line.prominence_db)[:max_lines]
    lines.sort(key=lambda line: line.hz)

    def tolerance(f):
        return np.maximum(step, 0.001 * f)

    return TonalLines(lines=tuple(lines), combs=_combs(lines, tolerance),
                      resolution_hz=step, min_prominence_db=float(min_prominence_db))
