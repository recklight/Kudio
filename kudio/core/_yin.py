# -*- coding: utf-8 -*-
"""pYIN: how likely each pitch is in each frame, and the likeliest course through
the frames.

Private: :func:`kudio.f0` is what to call.

A frame's pitch is looked for in YIN's difference function [YIN]_: the energy of
the frame less itself moved along by a lag, the samples past its end being zeros,
over its mean up to that lag (eqs. 6 to 8). Its troughs are the periods the frame
repeats at, the deeper the better. YIN takes the first one under a threshold;
pYIN [PYIN]_ takes 100 thresholds, beta(2, 18) distributed, and gives each trough
the probability of the thresholds it would be taken at. A hidden Markov model over
the frames chooses among them: 2 states for each tenth of a semitone, voiced and
unvoiced, moves in pitch as likely as a triangle is high that is 35.92 octaves a
second wide, and one chance in a hundred a frame of voicing to begin or end.

The pitches, the thresholds and the chances are the paper's. How wide the triangle
is comes from librosa, and so does this: a threshold shares its probability out
among the troughs under it, each exp(-2) of the one before, where the paper gives it
all to the first. ``librosa.pyin`` comes out the same to the bit in version 0.11, and
in version 1.0 when it is asked for all of the search (``transition_min_prob=None``);
left alone, 1.0 skips moves less likely than one in ten thousand, and calls a frame
here and there the other way (``tests/test_against_librosa.py``).

References
----------
.. [YIN] A. de Cheveigné and H. Kawahara, "YIN, a fundamental frequency estimator
   for speech and music", J. Acoust. Soc. Am., vol. 111, no. 4, pp. 1917-1930,
   2002, doi:10.1121/1.1458024.
.. [PYIN] M. Mauch and S. Dixon, "pYIN: A fundamental frequency estimator using
   probabilistic threshold distributions", Proc. ICASSP, pp. 659-663, 2014,
   doi:10.1109/ICASSP.2014.6853678.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Tuple

import numpy as np

__all__ = ['FINE', 'SPEED', 'states', 'track']

#: states in a semitone
FINE = 10
#: octaves a second, how wide the triangle of the moves is
SPEED = 35.92
#: the chance, each frame, that voicing begins or ends
_SWITCH = 0.01
#: what the deepest trough has of the thresholds no trough is under
_LEFT_OVER = 0.01
#: how much less likely each trough under a threshold is than the one before: e**-2
_FALL = 2
#: how many thresholds, evenly spaced and 1 the last
_THRESHOLDS = 100
#: added to a probability before its logarithm is taken
_TINY = np.finfo(np.float64).tiny
#: about how many bytes of float32 frames are worked on at a time
_BLOCK = 1 << 18


def _difference(frames: np.ndarray, shortest: int, longest: int) -> np.ndarray:
    """YIN's cumulative mean normalised difference of each row, at the lags from
    *shortest* to *longest*."""
    from scipy.fft import irfft, next_fast_len, rfft
    size = next_fast_len(2 * frames.shape[1] - 1, real=True)
    spectrum = rfft(frames, n=size, axis=1)
    alike = irfft(spectrum.real ** 2 + spectrum.imag ** 2, n=size, axis=1)
    # Eq. 7: the energy of the frame, and of it moved along, which has lost that of
    # its first samples, as many as the lag, less twice what the two have in common.
    # At a lag of 1 that one sample is left in, which librosa does.
    lost = np.cumsum(np.square(frames[:, :longest]), axis=1)
    lost[:, 0] = 0
    apart = 2 * (alike[:, :1] - alike[:, 1:longest + 1]) - lost
    mean = np.cumsum(apart, axis=1) / np.arange(1, longest + 1, dtype=np.float64)
    return apart[:, shortest - 1:] / (mean[:, shortest - 1:] + _TINY)


def _troughs(d: np.ndarray) -> np.ndarray:
    """Where each row of *d* is lower than before and no higher than after; its
    ends count when they are lower than their one neighbour."""
    down = d[:, 1:] < d[:, :-1]
    out = np.empty(d.shape, dtype=bool)
    out[:, 1:] = down
    out[:, 1:-1] &= d[:, 1:-1] <= d[:, 2:]
    out[:, 0] = d[:, 0] < d[:, 1]
    return out


@lru_cache(maxsize=None)
def _weights() -> Tuple[np.ndarray, np.ndarray]:
    """How likely each threshold is, beta(2, 18) distributed with 0.1 the mean, and
    how likely the first none, one, two of them and so on are together."""
    from scipy.special import betainc
    x = np.linspace(0, 1, _THRESHOLDS + 1)
    below = betainc(2.0, 18.0, x)
    # 1 from where nothing is nearer, which scipy before 1.12 comes to at the end alone:
    # the last threshold had a weight of 1e-16 there, and every trough a chance
    below[(1 - x) ** 18 * (1 + 18 * x) < 2.0 ** -54] = 1
    weights = np.diff(below)
    return weights, np.array([weights[:n].sum() for n in range(_THRESHOLDS + 1)])


def _chances(depth: np.ndarray, frame: np.ndarray) -> np.ndarray:
    """How likely each trough is, *depth* deep and in frame *frame*."""
    weights, together = _weights()
    first = np.flatnonzero(np.r_[True, frame[1:] != frame[:-1]])
    last = np.r_[first[1:], len(frame)]
    under = depth[:, np.newaxis] < np.linspace(0, 1, _THRESHOLDS + 1)[1:]
    # which of the troughs of its frame under a threshold each one is, and of how many
    place = np.cumsum(under, axis=0, dtype=np.int32)
    before = np.where(first[:, np.newaxis] > 0, place[first - 1], 0)
    many = place[last - 1] - before
    of = np.repeat(np.arange(len(first)), last - first)
    place -= before[of] + 1
    fall = np.exp(-_FALL * np.arange(int(many.max()) + 1, dtype=np.float64))
    with np.errstate(divide='ignore'):
        whole = (1 - np.exp(-_FALL)) / (1 - fall)
    share = whole[many[of]] * fall[np.maximum(place, 0)]
    share[~under] = 0
    # the sum over the thresholds a frame at a time: what BLAS adds up in which
    # order depends on the shape of what it is given
    out = np.empty(len(depth))
    over = _THRESHOLDS - np.count_nonzero(under, axis=1)
    for a, b in zip(first.tolist(), last.tolist()):
        out[a:b] = share[a:b].dot(weights)
        deepest = a + int(np.argmin(depth[a:b]))
        out[deepest] += _LEFT_OVER * together[over[deepest]]
    return out


def _vertex(d: np.ndarray, frame: np.ndarray, at: np.ndarray) -> np.ndarray:
    """How far from each trough the bottom of the parabola through it and its two
    neighbours is; 0 at the ends, and where the three are no trough."""
    out = np.zeros(len(at))
    inside = np.flatnonzero((at > 0) & (at < d.shape[1] - 1))
    frame, at = frame[inside], at[inside]
    low, here, high = d[frame, at - 1], d[frame, at], d[frame, at + 1]
    bend = high + low - 2 * here
    slope = (high - low) / 2
    sharp = np.abs(slope) < np.abs(bend)
    out[inside[sharp]] = -slope[sharp] / bend[sharp]
    return out


def _seen(frames: np.ndarray, sr: float, fmin: float, shortest: int, longest: int,
          n_bins: int) -> np.ndarray:
    """``(n_bins, frames)``: the probability of each pitch in each frame."""
    d = _difference(frames, shortest, longest)
    frame, at = np.nonzero(_troughs(d))
    # a row more than there are pitches, for what is above the range
    out = np.zeros((n_bins + 1, len(frames)))
    if len(at):
        chance = _chances(d[frame, at], frame)
        some = np.flatnonzero(chance)
        chance, frame, at = chance[some], frame[some], at[some]
        hertz = sr / (shortest + at + _vertex(d, frame, at))
        pitch = np.round(12 * FINE * np.log2(hertz / fmin))
        # of two periods in a tenth of a semitone the longer one counts
        out[np.clip(pitch, 0, n_bins).astype(int), frame] = chance
    return out[:n_bins]


class _Search:
    """The likeliest course through the states of the model, Viterbi's search over
    all of it. State *k* is pitch *k*, voiced, and state ``n_bins + k`` the same
    unvoiced.

    A move further than a pitch can go has no probability, and in logarithms one of
    ``log(_TINY)``, so it is there to be taken when all else is worse: over a hop
    of a millisecond no move is allowed at all, and the pitch changes nonetheless.
    Those all cost the same, so of all of them it is enough to look at the one from
    the best state. The rest are within a band around each pitch, and what is kept
    to find the way back by is where in the band a state was come to from.

    :param reach: how many pitches away the furthest move goes, up to
        ``(n_bins - 1) // 2``.
    """

    def __init__(self, n_bins: int, reach: int, n_frames: int):
        self.n, self.reach, self.width = n_bins, reach, 2 * reach + 1
        pitch = np.arange(n_bins)
        moves = np.maximum(0, reach + 1 - np.abs(pitch - pitch[:, np.newaxis])) / (reach + 1)
        moves /= moves.sum(axis=1, keepdims=True)
        to = pitch[:, np.newaxis]
        origin = to + np.arange(-reach, reach + 1)
        inside = (origin >= 0) & (origin < n_bins)
        near = np.where(inside, moves[np.clip(origin, 0, n_bins - 1), to], 0)
        stay = 1 - _SWITCH
        self.same, self.other = (np.log(p * near + _TINY) for p in (stay, 1 - stay))
        # the two halves of the states, each between edges no course comes from
        self.value = np.full((2, n_bins + 2 * reach), -np.inf)
        self.states = self.value[:, reach:reach + n_bins]
        self.around = np.lib.stride_tricks.sliding_window_view(self.value, self.width, axis=1)
        self.total = np.empty((2, n_bins, 2 * self.width))
        self.rows = np.arange(2 * n_bins)
        self.far = 2 * self.width
        self.back = np.empty((n_frames, 2 * n_bins),
                             dtype=np.uint8 if self.far < 256 else np.uint16)
        self.jump = np.empty(n_frames, dtype=np.int32)
        self.t = 0

    def _origin(self, row, code):
        """The state a state was come to from, *code* being where in its band."""
        return row % self.n + code % self.width - self.reach + code // self.width * self.n

    def push(self, voiced: np.ndarray, unvoiced: np.ndarray) -> None:
        """Go on by some frames: the logarithms of how likely each pitch is in
        each, and of how likely any of the unvoiced states."""
        w, total = self.width, self.total
        for seen, silent in zip(voiced, unvoiced):
            if not self.t:
                best = np.full(2 * self.n, np.log(1 / (2 * self.n) + _TINY))
            else:
                for half in (0, 1):
                    np.add(self.around[0], self.other if half else self.same,
                           out=total[half, :, :w])
                    np.add(self.around[1], self.same if half else self.other,
                           out=total[half, :, w:])
                flat = total.reshape(2 * self.n, 2 * w)
                code = flat.argmax(axis=1)
                best = flat[self.rows, code]
                top = int(self.states.argmax())
                floor = self.states[top // self.n, top % self.n] + np.log(_TINY)
                worse = np.flatnonzero(best <= floor)
                if len(worse):
                    # the first of two as good, as everywhere
                    worse = worse[(best[worse] < floor) |
                                  (top < self._origin(worse, code[worse]))]
                    code[worse], best[worse] = self.far, floor
                self.back[self.t], self.jump[self.t] = code, top
            self.states[0] = seen + best[:self.n]
            self.states[1] = silent + best[self.n:]
            self.t += 1

    def path(self) -> np.ndarray:
        """The states of the likeliest course, one for each frame."""
        out = np.empty(self.t, dtype=np.int64)
        state = int(self.states.argmax())
        for t in range(self.t - 1, 0, -1):
            out[t] = state
            code = int(self.back[t, state])
            state = int(self.jump[t]) if code == self.far else self._origin(state, code)
        out[0] = state
        return out


def states(sr: float, fmin: float, fmax: float, hop: int) -> Tuple[int, int]:
    """How many pitches the model has, and how many of them a hop can move it by."""
    return (int(np.floor(12 * FINE * np.log2(fmax / fmin))) + 1,
            round(SPEED * 12 * hop / sr) * FINE // 2)


def track(y: np.ndarray, sr: float, fmin: float, fmax: float, frame_length: int, hop: int,
          center: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(hertz, voiced, chance)`` of each frame of *y*, 1-D and float32: the pitch,
    NaN where it is unvoiced, whether it is voiced, and how likely that looked in
    the frame alone."""
    if center:
        y = np.pad(y, frame_length // 2)
    frames = np.lib.stride_tricks.sliding_window_view(y, frame_length)[::hop]
    n_bins, reach = states(sr, fmin, fmax, hop)
    shortest, longest = int(np.floor(sr / fmax)), min(int(np.ceil(sr / fmin)), frame_length - 1)
    search = _Search(n_bins, reach, len(frames))
    chance = np.empty(len(frames))

    step = max(2, _BLOCK // (4 * frame_length))
    starts = list(range(0, len(frames), step))
    if len(starts) > 1 and len(frames) - starts[-1] == 1:
        # numpy adds up a lone column in another order than it does two
        del starts[-1]
    for start, stop in zip(starts, starts[1:] + [len(frames)]):
        seen = _seen(frames[start:stop], sr, fmin, shortest, longest, n_bins)
        chance[start:stop] = np.clip(seen.sum(axis=0), 0, 1)
        search.push(np.ascontiguousarray(np.log(seen + _TINY).T),
                    np.log((1 - chance[start:stop]) / n_bins + _TINY))

    state = search.path()
    voiced = state < n_bins
    hertz = (fmin * 2 ** (np.arange(n_bins) / (12 * FINE)))[state % n_bins]
    hertz[~voiced] = np.nan
    return hertz, voiced, chance
