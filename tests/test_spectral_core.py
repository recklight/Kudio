# -*- coding: utf-8 -*-
"""kudio.core._spectral, the transform under the spectrograms, against its
definition written out."""
from __future__ import annotations

import tracemalloc

import numpy as np
import pytest

import kudio
from kudio.core import _spectral

#: most of the clips here are shorter than a frame somewhere
pytestmark = pytest.mark.filterwarnings("ignore:stft.*fewer than one frame:UserWarning")

RNG = np.random.default_rng(0)

#: (n_fft, hop, win_length), odd and even, hops that do and do not divide the
#: frame and one longer than it
GEOMETRIES = [(64, 16, 64), (64, 32, 64), (64, 24, 50), (65, 20, 65), (65, 21, 33), (7, 2, 7),
              (6, 5, 6), (2, 1, 2), (1, 1, 1), (32, 40, 32), (512, 128, 400)]


def _clip(n, dtype=np.float64):
    return (0.1 * RNG.standard_normal(n)).astype(dtype)


def _by_definition(y, n_fft, hop, win_length, window):
    """Frame k is the n_fft samples centred on sample k * hop of the clip, zeros
    past its ends, under the window, which is centred in the frame."""
    from scipy.signal import get_window
    w = np.zeros(n_fft)
    at = (n_fft - win_length) // 2
    w[at:at + win_length] = get_window(window, win_length)
    padded = np.concatenate([np.zeros(n_fft // 2), y, np.zeros(n_fft // 2)])
    frames = 1 + (len(padded) - n_fft) // hop
    k = np.arange(n_fft // 2 + 1)[:, None]
    basis = np.exp(-2j * np.pi * k * np.arange(n_fft) / n_fft)
    return np.stack([basis @ (w * padded[t * hop:t * hop + n_fft])
                     for t in range(frames)], axis=1)


@pytest.mark.parametrize("n_fft, hop, win_length", GEOMETRIES)
@pytest.mark.parametrize("window", ['hann', 'hamming', ('kaiser', 8.6)], ids=str)
def test_the_transform_is_its_definition(n_fft, hop, win_length, window):
    for n in (1, hop, n_fft, n_fft + 1, 3 * n_fft + 5):
        y = _clip(n)
        want = _by_definition(y, n_fft, hop, win_length, window)
        got = _spectral.stft(y, n_fft, hop, win_length, window)
        assert got.shape == want.shape == (n_fft // 2 + 1, 1 + (n + 2 * (n_fft // 2) - n_fft) // hop)
        assert got.dtype == np.complex128
        assert np.max(np.abs(got - want)) <= 1e-13 * (np.max(np.abs(want)) + 1)
        single = _spectral.stft(y.astype(np.float32), n_fft, hop, win_length, window)
        assert single.dtype == np.complex64
        assert np.max(np.abs(single - want)) <= 2e-7 * (np.max(np.abs(want)) + 1)


def _inverse_by_definition(spec, n_fft, hop, win_length, window, size):
    """Griffin & Lim 1984, eq. 6: the windowed frames added up, over the windows
    squared added up."""
    from scipy.signal import get_window
    w = np.zeros(n_fft)
    at = (n_fft - win_length) // 2
    w[at:at + win_length] = get_window(window, win_length)
    total = np.zeros(n_fft + hop * spec.shape[1] + size)
    energy = np.zeros_like(total)
    for t in range(spec.shape[1]):
        total[t * hop:t * hop + n_fft] += w * np.fft.irfft(spec[:, t], n=n_fft)
        energy[t * hop:t * hop + n_fft] += w ** 2
    covered = energy > np.finfo(np.float64).tiny
    total[covered] /= energy[covered]
    return total[n_fft // 2:n_fft // 2 + size]


@pytest.mark.parametrize("n_fft, hop, win_length", [g for g in GEOMETRIES if g[0] > 1])
@pytest.mark.parametrize("window", ['hann', 'boxcar'])
def test_the_inverse_is_the_least_squares_one(n_fft, hop, win_length, window):
    """On a spectrum no waveform has, where the frames disagree about the samples
    they share and the weighting matters."""
    bins = n_fft // 2 + 1
    for frames in (1, 2, 3, 17):
        spec = RNG.standard_normal((bins, frames)) + 1j * RNG.standard_normal((bins, frames))
        natural = n_fft + hop * (frames - 1) - 2 * (n_fft // 2)
        for length in (None, natural, max(0, natural - 3), natural + 5, 1):
            size = natural if length is None else length
            got = _spectral.istft(spec, n_fft, hop, win_length, window, length)
            want = _inverse_by_definition(spec, n_fft, hop, win_length, window, size)
            assert got.shape == (size,) and got.dtype == np.float64
            assert np.max(np.abs(got - want), initial=0) <= 1e-9 * np.max(np.abs(want), initial=1)


@pytest.mark.parametrize("n_fft, hop, win_length, window", [
    (512, 128, 512, 'hann'), (512, 256, 512, 'hamming'), (512, 256, 400, 'hann'),
    (511, 100, 511, 'blackman'), (400, 160, 400, 'hann'), (2048, 512, 2048, 'hann')])
@pytest.mark.parametrize("dtype, off", [(np.float32, 1e-6), (np.float64, 1e-13)])
def test_there_and_back_is_the_clip(n_fft, hop, win_length, window, dtype, off):
    """As far as a frame reaches: the last hop samples at most may be under the
    edge of one window alone, where what is divided by is next to nothing."""
    for n in (n_fft, 5001, 16000):
        y = _clip(n, dtype)
        back = _spectral.istft(_spectral.stft(y, n_fft, hop, win_length, window),
                               n_fft, hop, win_length, window, n)
        assert back.shape == y.shape and back.dtype == dtype
        assert np.max(np.abs(back - y)[:n - hop]) < off


def test_a_sine_on_a_bin_has_the_level_and_phase_it_has():
    """Half its amplitude times the sum of the window, and the phase at the
    middle of the frame turned by half a frame: the bin's own sign flip."""
    n_fft, k, phase = 512, 37, 0.7
    y = 0.4 * np.cos(2 * np.pi * k * np.arange(8 * n_fft) / n_fft + phase)
    spec = _spectral.stft(y, n_fft, 128, n_fft, 'hann')
    frame = 16                                     # centred on sample 2048 = 4 * n_fft
    assert abs(spec[k, frame]) == pytest.approx(0.4 / 2 * n_fft / 2, rel=1e-12)
    # the frame starts half a frame, k * pi, before its centre
    assert np.angle(spec[k, frame] * np.exp(-1j * (phase - np.pi * k))) == pytest.approx(0, abs=1e-12)
    assert np.max(np.abs(np.delete(spec[:, frame], [k - 1, k, k + 1]))) < 1e-10


def test_the_energy_of_a_frame_is_the_energy_of_its_spectrum():
    y = _clip(4096)
    spec = _spectral.stft(y, 512, 512, 512, 'boxcar')
    inside = spec[:, 2]                            # samples 768 to 1280
    power = np.abs(inside) ** 2
    assert (power[0] + power[-1] + 2 * power[1:-1].sum()) / 512 == pytest.approx(
        np.sum(y[768:1280] ** 2), rel=1e-12)


@pytest.mark.parametrize("block", [1, 4096, 100000, 1 << 24])
@pytest.mark.parametrize("n_fft, hop", [(512, 128), (400, 150), (65, 80)])
def test_how_much_is_worked_on_at_a_time_changes_nothing(monkeypatch, block, n_fft, hop):
    for dtype in (np.float32, np.float64):
        y = _clip(9001, dtype)
        spec = _spectral.stft(y, n_fft, hop, n_fft, 'hann')
        back = _spectral.istft(spec, n_fft, hop, n_fft, 'hann', len(y))
        with monkeypatch.context() as m:
            m.setattr(_spectral, '_BLOCK', block)
            assert np.array_equal(_spectral.stft(y, n_fft, hop, n_fft, 'hann'), spec)
            assert np.array_equal(_spectral.istft(spec, n_fft, hop, n_fft, 'hann', len(y)), back)


def test_a_long_clip_is_not_held_as_frames():
    """As float64 frames a minute of 48 kHz at 512 and 128 is 92 MB, and their
    spectra in complex128 as much again."""
    y = _clip(48000 * 60, np.float32)
    tracemalloc.start()
    try:
        before = tracemalloc.get_traced_memory()[0]
        spec = _spectral.stft(y, 512, 128, 512, 'hann')
        forward = tracemalloc.get_traced_memory()[1] - before - spec.nbytes
        tracemalloc.reset_peak()
        before = tracemalloc.get_traced_memory()[0]
        back = _spectral.istft(spec, 512, 128, 512, 'hann', len(y))
        inverse = tracemalloc.get_traced_memory()[1] - before - back.nbytes
    finally:
        tracemalloc.stop()
    # the padded clip, and for the inverse the window energy, with a few blocks
    assert forward < y.nbytes + 8 * 2 ** 20, f"{forward / 2 ** 20:.0f} MB"
    assert inverse < 2 * y.nbytes + 8 * 2 ** 20, f"{inverse / 2 ** 20:.0f} MB"


@pytest.mark.parametrize("dtype, spectrum", [(np.float16, np.complex64), (np.float32, np.complex64),
                                             (np.float64, np.complex128), ('>f4', np.complex64)])
def test_the_spectrum_is_as_precise_as_the_audio(dtype, spectrum):
    y = _clip(2000).astype(dtype)
    got = _spectral.stft(y, 512, 128, 512, 'hann')
    assert got.dtype == spectrum
    assert np.allclose(got, _spectral.stft(y.astype(np.float64), 512, 128, 512, 'hann'), atol=1e-6)


def test_the_clip_and_the_spectrum_are_left_alone():
    y = _clip(3000, np.float32)
    kept = y.copy()
    spec = _spectral.stft(y, 512, 128, 512, 'hann')
    held = spec.copy()
    back = _spectral.istft(spec, 512, 128, 512, 'hann', len(y))
    assert np.array_equal(y, kept) and np.array_equal(spec, held)
    assert not np.shares_memory(back, y) and not np.shares_memory(back, spec)


@pytest.mark.parametrize("window, same_as", [
    ('hann', lambda n: 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)),
    ('hamming', lambda n: 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(n) / n)),
    ('boxcar', np.ones), (np.bartlett, np.bartlett),
    ([0.0, 0.5, 1.0, 1.0, 0.5, 0.0, 0.25, 0.75], lambda n: [0.0, 0.5, 1.0, 1.0, 0.5, 0.0, 0.25, 0.75]),
    (np.arange(8.0), lambda n: np.arange(8.0))], ids=lambda w: None)
def test_a_window_is_a_name_the_values_or_a_function(window, same_as):
    """Periodic under a name: the next frame begins where the window would."""
    assert np.allclose(_spectral.window(window, 8, 'here'), same_as(8), atol=1e-15)


@pytest.mark.parametrize("window, says", [
    ('nonsense', r"here\(\): 'nonsense' is not a window scipy knows"),
    (('kaiser',), r"here\(\): \('kaiser',\) is not a window scipy knows"),
    (None, r"here\(\): None is not a window scipy knows"),
    ([1.0] * 7, r"here\(\): the window has 7 values, and 8 are needed"),
    (lambda n: np.ones((n, 2)), r"here\(\): the window has shape \(8, 2\)")], ids=repr)
def test_what_is_not_a_window_is_refused(window, says):
    with pytest.raises(kudio.FeatureError, match=says):
        _spectral.window(window, 8, 'here')
