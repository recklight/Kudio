# -*- coding: utf-8 -*-
"""kudio's STOI and ESTOI held to pystoi's. With pystoi's noise put to zeros,
``np.random.standard_normal`` replaced, they agree to the bit, at every rate,
float32 and float64, short and long; against pystoi as it is, STOI is the same to
the bit and ESTOI differs by that noise alone, under 1e-15.

kudio does not need pystoi and a plain install does not bring it, so on most
machines every test here skips. ``KUDIO_REQUIRE_PYSTOI=1`` is how a run says that it
must not: see ``conftest.py``.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import kudio
from kudio.core import _stoi

pystoi = pytest.importorskip("pystoi")


def _pair(sr, seconds, seed):
    rng = np.random.default_rng(seed)
    t = np.arange(int(sr * seconds)) / sr
    x = (0.3 * np.sin(2 * np.pi * 220 * t) * np.clip(np.sin(2 * np.pi * 3 * t), 0, 1)
         + 0.01 * rng.standard_normal(t.size))
    return x, x + 0.1 * rng.standard_normal(t.size)


def _theirs(x, y, sr, extended, noise=False):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if noise:
            return pystoi.stoi(x, y, sr, extended=extended)
        real = np.random.standard_normal
        np.random.standard_normal = lambda size=None: np.zeros(size)
        try:
            return pystoi.stoi(x, y, sr, extended=extended)
        finally:
            np.random.standard_normal = real


@pytest.mark.parametrize("sr", [8000, 10000, 16000, 22050, 44100, 48000])
def test_stoi_and_estoi_are_pystois_without_its_noise(sr):
    for seconds in (0.5, 1.5, 4.0):
        x, y = _pair(sr, seconds, sr)
        for dtype in (np.float32, np.float64):
            for extended in (False, True):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    got = _stoi.stoi(x.astype(dtype), y.astype(dtype), sr, extended)
                want = _theirs(x.astype(dtype), y.astype(dtype), sr, extended)
                assert type(got) is type(want) and got == want, (seconds, dtype, extended)


def test_the_edges_are_pystois():
    """4200 samples at 10 kHz leave exactly 30 frames, the fewest it scores; pauses
    42 dB under the peak are between the 40 dB it drops and anything more."""
    rng = np.random.default_rng(3)
    noise = rng.standard_normal(4200)
    t = np.arange(30000) / 10000
    bursts = np.sin(2 * np.pi * 200 * t) * (np.sin(2 * np.pi * 1.5 * t) > 0)
    paused = 0.3 * bursts + 0.3 * 10 ** (-42 / 20) * rng.standard_normal(t.size)
    for x in (noise, paused):
        y = x + 0.2 * rng.standard_normal(x.size)
        for extended in (False, True):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                got = _stoi.stoi(x, y, 10000, extended)
            assert got == _theirs(x, y, 10000, extended) and got != _stoi.TOO_SHORT


def test_against_pystoi_as_it_is_estoi_moves_by_its_noise_alone():
    x, y = _pair(16000, 3.0, 1)
    assert _stoi.stoi(x, y, 16000) == _theirs(x, y, 16000, False, noise=True)
    assert abs(_stoi.stoi(x, y, 16000, True) - _theirs(x, y, 16000, True, noise=True)) < 1e-15


def test_score_gives_pystois_numbers():
    x, y = _pair(16000, 3.0, 2)
    got = kudio.score(x, y, 16000, ['stoi', 'estoi'])
    assert got['stoi'] == float(_theirs(x, y, 16000, False))
    assert got['estoi'] == float(_theirs(x, y, 16000, True))
