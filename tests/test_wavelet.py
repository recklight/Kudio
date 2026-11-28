# -*- coding: utf-8 -*-
"""kudio's own discrete wavelet transform, on what is known about it without anything
to compare with: the Haar sums, perfect reconstruction, energy, the filters' sums,
and wavelet_low_pass_filter around it. tests/test_against_pywavelets.py holds it to
PyWavelets bit for bit."""
from __future__ import annotations

import numpy as np
import pytest

import kudio
from kudio.enhance import _dwt

RNG = np.random.default_rng(7)
NAMES = _dwt.names()
ORTHOGONAL = [n for n in NAMES if not n.startswith(('bior', 'rbio'))]
F64 = np.dtype(np.float64)


def _bank(name):
    return _dwt.filter_bank(name, 'test')


def test_there_are_the_106_discrete_wavelets():
    families = {}
    for name in NAMES:
        family = name.rstrip('0123456789.')
        families.setdefault(family, []).append(name)
    assert {k: len(v) for k, v in families.items()} == {
        'bior': 15, 'coif': 17, 'db': 38, 'dmey': 1, 'haar': 1, 'rbio': 15, 'sym': 19}
    assert {'db1', 'db38', 'sym2', 'sym20', 'coif1', 'coif17', 'bior1.1', 'bior6.8',
            'rbio1.1', 'rbio6.8'} <= set(NAMES)


def test_haar_is_sums_and_differences_over_root_two():
    lo, hi = _bank('haar')[F64][:2]
    a, d = _dwt.dwt(np.array([1.0, 2.0, 3.0, 4.0]), lo, hi)
    np.testing.assert_allclose(a, np.array([3.0, 7.0]) / np.sqrt(2), rtol=1e-15)
    np.testing.assert_allclose(d, np.array([-1.0, -1.0]) / np.sqrt(2), rtol=1e-15)


def test_an_odd_length_is_padded_with_its_last_sample():
    lo, hi = _bank('haar')[F64][:2]
    a, d = _dwt.dwt(np.array([1.0, 2.0, 5.0]), lo, hi)
    np.testing.assert_allclose(a, np.array([3.0, 10.0]) / np.sqrt(2), rtol=1e-15)
    np.testing.assert_allclose(d, np.array([-1.0, 0.0]) / np.sqrt(2), atol=1e-15)


@pytest.mark.parametrize("name", NAMES)
def test_every_wavelet_reconstructs_what_it_took_apart(name):
    """Down every level the length allows and back up, for even and odd lengths and
    lengths shorter than the filter. dmey, a finite approximation to the Meyer
    wavelet, is not quite orthogonal (the squares of its filter add up to 1.0022)
    and gives the clip back to 0.6 % of its peak, PyWavelets' dmey as well."""
    bank = _bank(name)
    taps = len(bank[F64][0])
    for n in (2 * taps, 2 * taps + 1, 1001, 4096):
        x = RNG.standard_normal(n)
        back = _dwt.waverec(_dwt.wavedec(x, bank), bank)[:n]
        peak = np.abs(x).max()
        np.testing.assert_allclose(back, x, atol=0.01 * peak if name == 'dmey' else 1e-10,
                                   err_msg=str(n))
        y = x.astype(np.float32)
        back = _dwt.waverec(_dwt.wavedec(y, bank), bank)[:n]
        assert back.dtype == np.float32
        np.testing.assert_allclose(back, y, atol=0.01 * peak if name == 'dmey' else 2e-5,
                                   err_msg=str(n))


@pytest.mark.parametrize("name", ORTHOGONAL)
def test_an_orthogonal_wavelet_keeps_the_energy(name):
    """To 1e-10: PyWavelets gives the symlets to about 1e-11 (1.4e-11 off at worst)."""
    lo, hi = _bank(name)[F64][:2]
    x = RNG.standard_normal(512)
    a, d = _dwt.dwt(x, lo, hi)
    assert np.sum(a ** 2) + np.sum(d ** 2) == pytest.approx(np.sum(x ** 2),
                                                          rel=1e-2 if name == 'dmey' else 1e-10)


@pytest.mark.parametrize("name", NAMES)
def test_the_filters_are_a_wavelet_s(name):
    """The low-pass filters add up to root two and the high-pass ones to nothing, to
    1e-11 (the symlets' high-pass filters are 3.3e-12 off at worst); dmey's high-pass
    to 1.1e-3, and the squares of its filter to 1.0022."""
    dec_lo, dec_hi, rec_lo, rec_hi = _bank(name)[F64]
    tol = 2e-3 if name == 'dmey' else 1e-11
    assert np.sum(dec_lo) == pytest.approx(np.sqrt(2), abs=1e-11)
    assert np.sum(rec_lo) == pytest.approx(np.sqrt(2), abs=1e-11)
    assert abs(np.sum(dec_hi)) < tol and abs(np.sum(rec_hi)) < tol
    if name in ORTHOGONAL:
        assert np.sum(dec_lo ** 2) == pytest.approx(1, abs=3e-3 if name == 'dmey' else 1e-10)
    for f32, f64 in zip(_bank(name)[np.dtype(np.float32)], _bank(name)[F64]):
        assert f32.dtype == np.float32
        np.testing.assert_allclose(f32, f64, rtol=1e-6, atol=1e-8)


def test_levels_go_as_far_as_the_filter_fits():
    assert [_dwt.max_level(n, 4) for n in (0, 2, 3, 5, 6, 11, 12, 48000)] == [0, 0, 0, 0, 1, 1, 2, 13]
    assert _dwt.max_level(1000, 2) == 9 and _dwt.max_level(1000, 1) == 0


def test_names_are_taken_in_any_case():
    y = 0.1 * RNG.standard_normal(3000)
    want = kudio.wavelet_low_pass_filter(y, 0.3, 'db2')
    for name in ('DB2', 'Db2', 'dB2'):
        assert np.array_equal(kudio.wavelet_low_pass_filter(y, 0.3, name), want)


@pytest.mark.parametrize("name", ['db39', 'sym1', 'coif18', 'db', ' db2', 'morl', 'mexh', 'xyz'])
def test_a_name_that_is_no_discrete_wavelet_is_refused_whatever_the_clip(name):
    """A wavelet that does not exist is a mistake in the call, silent clip or not."""
    for y, thres in ((0.1 * RNG.standard_normal(3000), 0.5), (np.zeros(3000), 0.5),
                     (0.1 * RNG.standard_normal(3000), 0)):
        with pytest.raises(ValueError, match=r"^wavelet_low_pass_filter\(\): wavelet .* is not a "
                                             r"discrete wavelet this filter knows; the names are haar"):
            kudio.wavelet_low_pass_filter(y, thres, name)


@pytest.mark.parametrize("value", [2, 2.0, True, ['db2'], ('db2',)])
def test_a_wavelet_that_is_not_a_name_is_a_type_error(value):
    with pytest.raises(TypeError, match=r"^wavelet_low_pass_filter\(\): wavelet has to be the name"):
        kudio.wavelet_low_pass_filter(0.1 * RNG.standard_normal(3000), 0.5, value)


@pytest.mark.parametrize("value", ['', None])
def test_no_wavelet_is_a_type_error(value):
    with pytest.raises(TypeError, match=r"^wavelet_low_pass_filter\(\): wavelet has to be the name"):
        kudio.wavelet_low_pass_filter(0.1 * RNG.standard_normal(3000), 0.5, value)


def test_bytes_are_no_name():
    with pytest.raises(ValueError, match=r"^wavelet_low_pass_filter\(\): wavelet b'db2' is bytes"):
        kudio.wavelet_low_pass_filter(0.1 * RNG.standard_normal(3000), 0.5, b'db2')


@pytest.mark.parametrize("n, name", [(1, 'db2'), (5, 'db2'), (1, 'haar'), (149, 'db38')])
def test_a_clip_too_short_for_one_level_comes_back_as_a_copy(n, name):
    """Changing the answer must not change the clip that was passed in."""
    y = 0.1 * RNG.standard_normal(n)
    kept = y.copy()
    out = kudio.wavelet_low_pass_filter(y, 0.3, name)
    assert np.array_equal(out, kept) and out.dtype == y.dtype
    out[0] = 99.0
    assert np.array_equal(y, kept)


@pytest.mark.parametrize("y, thres", [(np.zeros(3000), 0.5), (0.1 * RNG.standard_normal(3000), 0),
                                     (0.1 * RNG.standard_normal(3000), 0.0)],
                         ids=["silence", "thres 0", "thres 0.0"])
def test_with_a_threshold_of_nothing_the_clip_comes_back_as_a_copy(y, thres):
    kept = y.copy()
    out = kudio.wavelet_low_pass_filter(y, thres)
    assert out is not y and np.array_equal(out, kept)
    out[:] = 1.0
    assert np.array_equal(y, kept)


def test_float16_and_float32_come_back_float32_the_rest_float64():
    y = 0.1 * RNG.standard_normal(3000)
    for dtype, want in (('f2', 'f4'), ('f4', 'f4'), ('f8', 'f8'), ('>f8', 'f8'), ('i2', 'f8')):
        z = (y * (1000 if dtype == 'i2' else 1)).astype(dtype)
        assert kudio.wavelet_low_pass_filter(z, 0.3).dtype == np.dtype(want), dtype


def test_a_threshold_above_every_detail_leaves_the_approximation_alone():
    """Every detail goes, so what comes back is the coarsest approximation put back up
    with zeros beside it."""
    y = 0.1 * RNG.standard_normal(1024)
    bank = _bank('db4')
    coeffs = _dwt.wavedec(y, bank)
    want = _dwt.waverec([coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]], bank)
    got = kudio.wavelet_low_pass_filter(y, 1e6, 'db4')
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-15)


def test_a_steady_clip_comes_through_unchanged():
    """A constant has no detail at any level of an orthogonal wavelet."""
    y = np.full(1024, 0.25)
    for name in ('haar', 'db2', 'sym8', 'coif3'):
        np.testing.assert_allclose(kudio.wavelet_low_pass_filter(y, 0.5, name), y, atol=1e-12)


def test_the_filters_are_left_to_the_first_call():
    """The coefficients of 106 wavelets, which `import kudio` has no need of."""
    import subprocess
    import sys
    code = ("import kudio, sys; assert 'kudio.enhance._wavelet_filters' not in sys.modules; "
            "kudio.wavelet_low_pass_filter([0.1, -0.2] * 50); "
            "assert 'kudio.enhance._wavelet_filters' in sys.modules")
    subprocess.check_call([sys.executable, "-c", code])
