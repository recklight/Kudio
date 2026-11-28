# -*- coding: utf-8 -*-
"""kudio's wavelet transform held to PyWavelets', bit for bit: the four filters of every
discrete wavelet in float64 and float32, one level each way, and
wavelet_low_pass_filter against the same steps in PyWavelets, at lengths from one
sample up, odd and even, in float32 and float64.

kudio does not need PyWavelets and a plain install does not bring it, so on most
machines every test here skips. ``KUDIO_REQUIRE_PYWAVELETS=1`` is how a run says
that it must not: see ``conftest.py``.
"""
from __future__ import annotations

import numpy as np
import pytest

import kudio
from kudio.enhance import _dwt

pywt = pytest.importorskip("pywt")

NAMES = pywt.wavelist(kind='discrete')
#: every length up to 20, where the filters reach past both ends, then a few odd ones
LENGTHS = list(range(1, 21)) + [31, 101, 1001]
F32, F64 = np.dtype(np.float32), np.dtype(np.float64)


def _same(got, want, what):
    """The same shape, the same type and the same bits."""
    got, want = np.asarray(got), np.asarray(want)
    assert got.shape == want.shape, f"{what}: shape {got.shape}, PyWavelets {want.shape}"
    assert got.dtype == want.dtype, f"{what}: {got.dtype}, PyWavelets {want.dtype}"
    if not np.array_equal(got.view(np.uint8), want.view(np.uint8)):
        raise AssertionError(f"{what}: {np.count_nonzero(got != want)} of {got.size} differ, "
                             f"by up to {np.max(np.abs(got - want)):.3g}")


def _their_float32(w):
    """The float32 filters PyWavelets' C code computes with, read from what it makes of
    an impulse: a single sample of one gives each tap on its own."""
    taps = w.dec_len
    n = 4 * taps + 8
    dec = np.zeros((2, taps), np.float32)
    for p in (n // 2, n // 2 + 1):
        x = np.zeros(n, np.float32)
        x[p] = 1
        for row, c in enumerate(pywt.dwt(x, w, mode='periodization')):
            for o in range(len(c)):
                m = taps // 2 + 2 * o - p
                if 0 <= m < taps:
                    dec[row, m] = c[o]
    half, start = taps // 2, taps // 4
    shift = 1 - half % 2
    n = 2 * taps + 4
    rec = np.zeros((2, taps), np.float32)
    for row in (0, 1):
        for q in range(n):
            a, d = np.zeros(n, np.float32), np.zeros(n, np.float32)
            (a, d)[row][q] = 1
            r = pywt.idwt(a, d, w, mode='periodization')
            for k in range(n):
                m = (start + k - q) % n
                if m < half:
                    first = (shift + 2 * k) % (2 * n)
                    rec[row, 2 * m], rec[row, 2 * m + 1] = r[first], r[(first + 1) % (2 * n)]
    return dec[0], dec[1], rec[0], rec[1]


def test_kudio_has_every_discrete_wavelet_pywavelets_has():
    assert sorted(_dwt.names()) == sorted(NAMES)
    assert len(NAMES) == 106


@pytest.mark.parametrize("name", NAMES)
def test_the_filters_are_pywavelets(name):
    """Equal in value: a zero in a padded biorthogonal filter may carry the other sign,
    which no sum here can tell, since every sum starts at +0."""
    w = pywt.Wavelet(name)
    mine = _dwt.filter_bank(name, 'test')
    for got, want in zip(mine[F64], w.filter_bank):
        assert got.dtype == F64 and np.array_equal(got, np.asarray(want, F64)), name
    for got, want in zip(mine[F32], _their_float32(w)):
        assert got.dtype == F32 and np.array_equal(got, want), name


@pytest.mark.parametrize("name", NAMES)
def test_one_level_each_way_is_pywavelets(name):
    w = pywt.Wavelet(name)
    rng = np.random.default_rng(len(name))
    bank = _dwt.filter_bank(name, 'test')
    for n in LENGTHS:
        for dtype in (F32, F64):
            x = rng.standard_normal(n).astype(dtype)
            want = pywt.dwt(x, w, mode='periodization')
            got = _dwt.dwt(x, *bank[dtype][:2])
            _same(got[0], want[0], f"{name} dwt {n} {dtype} approximation")
            _same(got[1], want[1], f"{name} dwt {n} {dtype} detail")
            a, d = rng.standard_normal((2, n)).astype(dtype)
            _same(_dwt.idwt(a, d, *bank[dtype][2:]), pywt.idwt(a, d, w, mode='periodization'),
                  f"{name} idwt {n} {dtype}")


def _theirs(y, thres, wavelet):
    """wavelet_low_pass_filter in PyWavelets' steps."""
    level = thres * float(np.nanmax(np.abs(y)))
    coeff = pywt.wavedec(y, wavelet, mode='periodization')
    coeff[1:] = (pywt.threshold(c, value=level, mode='soft') for c in coeff[1:])
    return pywt.waverec(coeff, wavelet, mode='periodization')[:len(y)]


@pytest.mark.parametrize("name", NAMES)
def test_the_filter_is_pywavelets(name):
    """Lengths odd and even and shorter than the filter, two thresholds in turn, and
    one clip long enough for every wavelet to go down several levels."""
    rng = np.random.default_rng(100 + len(name))
    cases = [(n, dtype, (0.05, 0.8)[n % 2]) for n in (1, 2, 3, 4, 5, 7, 8, 13, 31, 64, 101,
                                                     255, 1001) for dtype in (F32, F64)]
    for n, dtype, thres in cases + [(4097, F64, 0.8)]:
        y = (0.3 * rng.standard_normal(n)).astype(dtype)
        _same(kudio.wavelet_low_pass_filter(y, thres, name), _theirs(y, thres, name),
              f"{name} {n} {dtype} {thres}")


@pytest.mark.parametrize("dtype", ['f2', '>f4', '>f8', 'i2', 'u1'])
def test_other_sample_types_come_out_as_pywavelets(dtype):
    """float16 works in float32, the rest in float64; integers are read on their full
    scale first, so PyWavelets is given that."""
    rng = np.random.default_rng(3)
    for n in (5, 1001):
        y = 0.3 * rng.standard_normal(n)
        z = y.astype(dtype) if dtype[-2] == 'f' else (
            np.round(y * 100 + (128 if dtype == 'u1' else 0)).astype(dtype))
        read = kudio.wavelet_low_pass_filter(z, 0)              # what the filter is given
        for name in ('db2', 'coif3', 'bior2.2', 'dmey'):
            _same(kudio.wavelet_low_pass_filter(z, 0.5, name), _theirs(read, 0.5, name),
                  f"{dtype} {n} {name}")


@pytest.mark.parametrize("name", ['db2', 'coif3', 'bior2.2', 'rbio3.1', 'dmey'])
def test_a_pywavelets_wavelet_computes_as_its_name(name):
    """A pywt.Wavelet, and one made from the same filters under another name, as
    PyWavelets computes them: the second in float32 from the float64 filters."""
    rng = np.random.default_rng(5)
    for dtype in (F32, F64):
        y = (0.3 * rng.standard_normal(3001)).astype(dtype)
        made = pywt.Wavelet('made', filter_bank=pywt.Wavelet(name).filter_bank)
        for w in (pywt.Wavelet(name), made):
            _same(kudio.wavelet_low_pass_filter(y, 0.5, w), _theirs(y, 0.5, w),
                  f"{w.name} {dtype}")


def test_no_continuous_wavelet_is_taken():
    for name in pywt.wavelist(kind='continuous'):
        with pytest.raises(ValueError, match=r"is not a discrete wavelet this filter knows"):
            kudio.wavelet_low_pass_filter(np.ones(64), 0.5, name)
    with pytest.raises(ValueError, match=r"wavelet is a ContinuousWavelet"):
        kudio.wavelet_low_pass_filter(np.ones(64), 0.5, pywt.ContinuousWavelet('morl'))
