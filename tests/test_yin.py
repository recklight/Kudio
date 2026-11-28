# -*- coding: utf-8 -*-
"""pYIN, part by part, against its equations worked out the long way and against
signals whose period is known."""
from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

import kudio
from kudio.core import _yin

SR = 16000
RNG = np.random.default_rng(0)


def voice(hz, seconds=0.5, sr=SR):
    t = np.arange(int(seconds * sr)) / sr
    return (0.2 * sum(np.sin(2 * np.pi * hz * k * t) / k for k in range(1, 8))).astype(np.float32)


# -- the difference function ---------------------------------------------------

def _the_long_way(x, shortest, longest):
    """Eqs. 6 to 8 of the YIN paper, over the frame and the zeros past its end."""
    x = np.r_[np.asarray(x, np.float64), np.zeros(longest)]
    n = len(x) - longest
    apart = np.array([sum((x[j] - x[j + lag]) ** 2 for j in range(n))
                      for lag in range(1, longest + 1)])
    mean = np.cumsum(apart) / np.arange(1, longest + 1)
    return (apart / mean)[shortest - 1:]


@pytest.mark.parametrize("n", [64, 65, 200])
def test_the_difference_is_the_papers(n):
    """To what float32 holds. A clip that starts at 0 has nothing for the lag of 1,
    which keeps the frame's first sample, to differ by."""
    frames = RNG.standard_normal((5, n)).astype(np.float32)
    frames[:, 0] = 0
    got = _yin._difference(frames, 3, 30)
    assert got.shape == (5, 28) and got.dtype == np.float64
    for row, x in zip(got, frames):
        assert row == pytest.approx(_the_long_way(x, 3, 30), rel=2e-5)


def test_a_lag_of_one_keeps_the_first_sample():
    """What librosa does, and what its numbers come out of: the difference at a lag
    of 1 is too large by that sample squared, and the means after it by their share
    of it."""
    x = np.zeros((1, 64), np.float32)
    x[0, 0], x[0, 1:] = 1.0, 0.5 * RNG.standard_normal(63)
    frame = np.r_[x[0].astype(np.float64), np.zeros(20)]
    apart = np.array([np.sum((frame[:64] - frame[lag:64 + lag]) ** 2) for lag in range(1, 21)])
    apart[0] += 1.0
    want = apart / (np.cumsum(apart) / np.arange(1, 21))
    assert _yin._difference(x, 1, 20)[0] == pytest.approx(want, rel=2e-5)


def test_the_difference_has_its_troughs_at_the_period_and_its_multiples():
    """A wave of 100 samples a period: at 100 and 200 samples nothing is left of it
    but the samples that meet the zeros past the end, 100 and 200 of 2048."""
    wave = np.sin(2 * np.pi * np.arange(2048) / 100).astype(np.float32)
    d = _yin._difference(wave[np.newaxis], 60, 230)[0]
    at = np.flatnonzero(_yin._troughs(d[np.newaxis])[0]) + 60
    assert at.tolist() == [100, 200]
    # the energy of the last one or two periods over the mean, twice the frame's
    assert d[100 - 60] == pytest.approx((100 / 2048) / 2, abs=0.006)
    assert d[200 - 60] == pytest.approx((200 / 2048) / 2, abs=0.006)


def test_silence_has_no_difference_and_no_trough():
    d = _yin._difference(np.zeros((3, 512), np.float32), 40, 247)
    assert not d.any()
    assert not _yin._troughs(d).any()


# -- its troughs ----------------------------------------------------------------

def test_the_troughs_are_lower_than_before_and_no_higher_than_after():
    d = np.round(RNG.uniform(0, 4, (200, 12)))           # a good many alike
    for row, got in zip(d, _yin._troughs(d)):
        want = [False] * 12
        want[0] = row[0] < row[1]
        want[11] = row[11] < row[10]
        for i in range(1, 11):
            want[i] = row[i] < row[i - 1] and row[i] <= row[i + 1]
        assert got.tolist() == want, row
    assert _yin._troughs(np.array([[2.0, 1.0], [1.0, 2.0], [1.0, 1.0]])).tolist() == [
        [False, True], [True, False], [False, False]]


def test_a_vertex_is_where_the_parabola_has_it():
    """Three points of a parabola give its lowest one back; at the ends of the lags,
    and where the lowest point is not between the neighbours, there is none."""
    i = np.arange(8.0)
    bottoms = [3.0, 3.25, 2.6, 4.5 - 1e-9]
    d = np.array([0.7 * (i - b) ** 2 + 0.1 for b in bottoms])
    at = np.array([round(b) for b in bottoms])
    got = _yin._vertex(d, np.arange(4), at)
    assert at + got == pytest.approx(bottoms, abs=1e-9)
    assert _yin._vertex(d, np.array([0, 0]), np.array([0, 7])).tolist() == [0, 0]
    flat = np.array([[1.0, 1.0, 1.0, 5.0, 4.0, 3.0]])       # no bend, and a straight line
    assert _yin._vertex(flat, np.array([0, 0]), np.array([1, 4])).tolist() == [0, 0]


# -- how likely each is ----------------------------------------------------------

def _chances_the_long_way(depths, weights):
    """Of the troughs of one frame: each threshold gives its weight to the troughs
    under it, each e**-2 of the one before, and the thresholds under which there is
    none give a hundredth of theirs to the deepest."""
    out = [0.0] * len(depths)
    for threshold, weight in zip([(k + 1) / 100 for k in range(100)], weights):
        under = [k for k, depth in enumerate(depths) if depth < threshold]
        if under:
            parts = [math.exp(-2 * place) for place in range(len(under))]
            for k, part in zip(under, parts):
                out[k] += weight * part / sum(parts)
        else:
            out[depths.index(min(depths))] += 0.01 * weight
    return out


def test_the_thresholds_are_beta_distributed_around_a_tenth():
    """beta(2, 18), the distribution the paper settles on: a mean of 0.1, the
    density 18 * 19 * x * (1 - x)**17."""
    weights, together = _yin._weights()
    assert len(weights) == 100 and weights.sum() == pytest.approx(1, abs=1e-12)
    assert together == pytest.approx(np.r_[0, np.cumsum(weights)], abs=1e-15)
    at = (np.arange(100) + 0.5) / 100
    assert weights @ at == pytest.approx(0.1, abs=1e-4)
    assert weights == pytest.approx(342 * at * (1 - at) ** 17 / 100, abs=1e-3)


def test_a_threshold_nothing_is_likely_at_has_no_weight():
    """Of the beta distribution what is above 0.9 is under half a bit of 1. In fractions,
    the weights are 0 from there on and nowhere else."""
    from fractions import Fraction
    exact = np.array([float(1 - Fraction(100 - k, 100) ** 18 * (1 + Fraction(18 * k, 100)))
                      for k in range(101)])
    weights = _yin._weights()[0]
    assert np.flatnonzero(weights == 0).tolist() == list(range(90, 100))
    assert np.flatnonzero(np.diff(exact) == 0).tolist() == list(range(90, 100))
    assert np.max(np.abs(weights - np.diff(exact))) < 2e-15


@pytest.mark.parametrize("sr, fmin, fmax", [(16000, 50.0, 2000.0), (22050, 50.0, 2000.0),
                                            (22050, 65.0, 400.0), (48000, 100.0, 600.0)])
def test_the_last_bit_of_the_distribution_decides_nothing(monkeypatch, sr, fmin, fmax):
    """scipy before 1.12 has it stop one bit short of 1 until the very end. Troughs of no
    chance then had one of 1e-16, and one such that shares a pitch with the deepest trough
    of its frame counted instead of it."""
    import scipy.special
    y = 0.1 * np.random.default_rng(5).standard_normal(int(1.5 * sr))
    want = kudio.f0(y, sr, fmin=fmin, fmax=fmax)
    real = scipy.special.betainc
    _yin._weights.cache_clear()
    monkeypatch.setattr(scipy.special, 'betainc', lambda a, b, x: np.where(
        x < 1, np.minimum(real(a, b, x), 1 - 2.0 ** -53), real(a, b, x)))
    try:
        got = kudio.f0(y, sr, fmin=fmin, fmax=fmax)
    finally:
        _yin._weights.cache_clear()
    assert np.array_equal(got.f0, want.f0, equal_nan=True)
    assert np.array_equal(got.voiced, want.voiced)
    assert np.max(np.abs(got.voiced_prob - want.voiced_prob)) < 1e-14


def test_the_chances_add_up_over_the_thresholds():
    weights = _yin._weights()[0]
    frames = [[0.3], [0.05, 0.2, 0.04], [1.2], [1.4, 1.1], [0.5, 0.5, 0.5], [0.01, 0.99],
              list(RNG.uniform(0, 1.3, 40)), [0.02]]
    depth = np.array([d for f in frames for d in f])
    frame = np.repeat([0, 1, 4, 5, 6, 9, 10, 11], [len(f) for f in frames])
    want = [p for f in frames for p in _chances_the_long_way(f, weights)]
    assert _yin._chances(depth, frame) == pytest.approx(want, rel=1e-12, abs=1e-300)


def test_a_frame_by_itself_is_as_likely_as_among_others():
    depth, frame = RNG.uniform(0, 1.2, 500), np.sort(RNG.integers(0, 40, 500))
    together = _yin._chances(depth, frame)
    for k in np.unique(frame):
        alone = _yin._chances(depth[frame == k], frame[frame == k])
        assert np.array_equal(alone, together[frame == k])


def test_of_two_periods_in_one_state_the_longer_one_counts(monkeypatch):
    """1000 and 1002 samples are 3.5 cents apart, and the states 10."""
    d = np.ones((1, 11))
    d[0, 4], d[0, 6] = 0.02, 0.06
    monkeypatch.setattr(_yin, '_difference', lambda *a: d)
    seen = _yin._seen(np.zeros((1, 8), np.float32), 16000, 16000 / 1002, 996, 1006, 3)
    chance = _yin._chances(np.array([0.02, 0.06]), np.zeros(2, int))
    assert seen[:, 0].tolist() == [chance[1], 0, 0]
    assert chance[1] < chance[0]


@pytest.mark.parametrize("sr, fmin, fmax, want", [(16000, 65.0, 400.0, (40, 247)),
                                                  (16000, 100.0, 400.0, (40, 160)),
                                                  (22254.5, 65.0, 400.0, (55, 343)),
                                                  (8000, 65.0, 4000.0, (2, 124))])
def test_the_lags_are_the_periods_of_the_range_and_what_is_left_of_a_sample(
        monkeypatch, sr, fmin, fmax, want):
    asked = []
    real = _yin._difference
    monkeypatch.setattr(_yin, '_difference', lambda frames, shortest, longest: asked.append(
        (shortest, longest)) or real(frames, shortest, longest))
    _yin.track(np.zeros(4000, np.float32), sr, fmin, fmax, 2048, 160, True)
    assert set(asked) == {want}


# -- the search -------------------------------------------------------------------

def _dense(n_bins, reach):
    """Every move there is, in logarithms: ``[from, to]`` over voiced and unvoiced."""
    k = np.arange(n_bins)
    moves = np.maximum(0, reach + 1 - abs(k - k[:, np.newaxis])) / (reach + 1)
    moves /= moves.sum(axis=1, keepdims=True)
    return np.log(np.kron([[0.99, 1 - 0.99], [1 - 0.99, 0.99]], moves) + np.finfo(float).tiny)


def _scored(path, seen, moves):
    return (-math.log(len(moves)) + sum(frame[s] for frame, s in zip(seen, path))
            + sum(moves[a, b] for a, b in zip(path[:-1], path[1:])))


def _searched(seen, n_bins, reach, pieces=1):
    search = _yin._Search(n_bins, reach, len(seen))
    for part in np.array_split(seen, pieces):
        search.push(part[:, :n_bins], part[:, n_bins])
    return search.path()


def _observed(rng, n_frames, n_bins, nothing=0.0):
    """Made-up logarithms: the unvoiced states of a frame are equally likely."""
    p = rng.uniform(0, 1, (n_frames, n_bins)) ** 4 + 1e-12
    p[rng.uniform(size=p.shape) < nothing] = 0
    quiet = np.repeat(rng.uniform(0, 1, (n_frames, 1)), n_bins, axis=1)
    return np.log(np.c_[p, quiet] + np.finfo(float).tiny)


@pytest.mark.parametrize("n_bins, reach", [(1, 0), (2, 0), (3, 0), (3, 1), (4, 1), (5, 2)])
def test_no_course_is_likelier_than_the_one_found(n_bins, reach):
    """Of all there are, over four frames: up to 10000."""
    rng = np.random.default_rng(n_bins)
    moves = _dense(n_bins, reach)
    for _ in range(20):
        seen = _observed(rng, 4, n_bins, nothing=0.3)
        found = _scored(_searched(seen, n_bins, reach), seen, moves)
        best = max(_scored(p, seen, moves)
                   for p in itertools.product(range(2 * n_bins), repeat=4))
        assert found == pytest.approx(best, rel=1e-12)


def _viterbi(seen, moves):
    """The search written out, over every move. Of two as good, the first state."""
    value = seen[0] + np.log(1 / len(moves) + np.finfo(float).tiny)
    back = []
    for frame in seen[1:]:
        into = value[:, np.newaxis] + moves
        back.append(into.argmax(axis=0))
        value = frame + into.max(axis=0)
    path = [int(value.argmax())]
    for step in back[::-1]:
        path.append(int(step[path[-1]]))
    return path[::-1]


@pytest.mark.parametrize("n_bins, reach", [(1, 0), (2, 0), (3, 1), (30, 0), (30, 1), (60, 7),
                                           (120, 20), (127, 63), (128, 63), (130, 64)])
@pytest.mark.parametrize("nothing", [0.0, 0.6, 0.99])
def test_it_is_the_whole_search(n_bins, reach, nothing):
    """The course over every move, ones too far to have a probability included. From
    a reach of 64 on, where a state was come to from takes two bytes."""
    assert _yin._Search(n_bins, reach, 1).back.dtype == (np.uint8 if reach < 64 else np.uint16)
    rng = np.random.default_rng(1)
    moves = _dense(n_bins, reach)
    for n_frames in (1, 2, 40):
        seen = _observed(rng, n_frames, n_bins, nothing)
        assert _searched(seen, n_bins, reach).tolist() == _viterbi(seen, moves)


def test_a_pitch_that_jumps_is_followed_where_no_move_is_allowed():
    """With no reach, as over a hop of a millisecond, a move costs log(tiny) = -708,
    and is worth it once staying has cost more. Nothing but the one move from the
    best state can be it."""
    moves = _dense(50, 0)
    seen = np.log(np.full((40, 100), 1e-30))
    seen[:20, 5] = seen[20:, 40] = 0.0
    path = _searched(seen, 50, 0).tolist()
    assert path == [5] * 20 + [40] * 20 == _viterbi(seen, moves)


def test_of_two_courses_as_likely_the_first_is_taken():
    """All alike, everywhere: the search stays in the first state there is."""
    seen = np.zeros((12, 20))
    for reach in (0, 3):
        assert _searched(seen, 10, reach).tolist() == _viterbi(seen, _dense(10, reach))


@pytest.mark.parametrize("near, far", [(2, 9), (9, 2)])
def test_of_a_far_move_and_a_near_one_as_good_the_first_is_taken(near, far):
    search = _yin._Search(10, 0, 2)
    search.t = 1
    search.states[:] = -1e6
    # numbers for which the two come out the same to the bit
    for k in range(1000):
        here = 600 + k * 2.0 ** -43
        there = here + search.same[near, 0] - np.log(_yin._TINY)
        if there + np.log(_yin._TINY) == here + search.same[near, 0]:
            break
    else:
        pytest.fail("no such numbers")
    search.states[0, near], search.states[0, far] = here, there
    search.push(np.zeros((1, 10)), np.zeros(1))
    assert search.jump[1] == far
    assert (search.back[1, near] == search.far) == (far < near)
    assert search.states[0, near] == here + search.same[near, 0]


def test_the_moves_are_a_triangle():
    """35.92 octaves a second over 10 ms are 4 whole semitones, 20 states either way:
    at the far end 1/21 as likely as staying."""
    assert _yin.states(16000, 65.0, 400.0, 160) == (315, 20)
    assert _yin.states(16000, 65.0, 400.0, 18) == (315, 0)
    search = _yin._Search(315, 20, 1)
    middle = np.exp(search.same[150])
    assert len(middle) == 41 and middle.sum() == pytest.approx(0.99, abs=1e-12)
    assert middle[20] / middle[[0, 40]] == pytest.approx([21, 21], rel=1e-9)
    assert np.exp(search.other[150]).sum() == pytest.approx(0.01, abs=1e-12)
    assert np.exp(search.same[0, :20]).max() < 1e-300          # under the lowest pitch


# -- the whole, in pieces -----------------------------------------------------------

@pytest.mark.parametrize("size", [1, 3 * 4 * 2048, 10 ** 9])
def test_the_blocks_it_is_worked_on_in_do_not_show(monkeypatch, size):
    """Two frames at a time, three, or all 151 at once. The last block never has a
    frame to itself, which numpy would add up in another order."""
    y = np.r_[voice(140.0), 0.05 * RNG.standard_normal(8000), voice(210.0)].astype(np.float32)
    want = _yin.track(y, SR, 65.0, 400.0, 2048, 160, True)
    assert len(want[0]) == 151
    sizes = []
    real = _yin._seen
    monkeypatch.setattr(_yin, '_seen', lambda frames, *a: sizes.append(len(frames)) or
                        real(frames, *a))
    monkeypatch.setattr(_yin, '_BLOCK', size)
    got = _yin.track(y, SR, 65.0, 400.0, 2048, 160, True)
    assert all(np.array_equal(a, b, equal_nan=a.dtype.kind == 'f') for a, b in zip(got, want))
    assert min(sizes) > 1 and sum(sizes) == 151
    assert len(sizes) == {1: 75, 3 * 4 * 2048: 50, 10 ** 9: 1}[size]


def test_f0_needs_no_scipy_signal_or_stats():
    """They are slow to import, and pYIN is the FFT and one special function."""
    import subprocess
    import sys
    code = ("import sys, numpy as np, kudio\n"
            "kudio.f0(np.zeros(4000), 16000)\n"
            "assert 'scipy.signal' not in sys.modules and 'scipy.stats' not in sys.modules\n")
    done = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                          timeout=120)
    assert done.returncode == 0, done.stderr


def test_the_pitch_is_one_of_the_states():
    track = kudio.f0(voice(187.3), SR)
    steps = 120 * np.log2(track.f0[track.voiced] / 65.0)
    assert np.abs(steps - np.round(steps)).max() < 1e-9
