# -*- coding: utf-8 -*-
"""Mel-cepstral distortion as URGENT 2025 computes it."""
from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import kudio
from kudio import mcd
from kudio.core import melcepstrum as module

from conftest import make_speech

#: URGENT 2025's mcd_metric, and its calculate() without the gain match, on
#: 3 s of the speech fixture under white noise, computed with pysptk 1.0.1
#: and fastdtw 0.3.4 by tools/validation/melcepstrum.py's restatement
URGENT = {8000: (6.890108249708833, 9.115361897984503),
          16000: (7.270625570513369, 9.60345148922789)}


def _pair(sr):
    clean = make_speech(sr, 6.0, seed=0).astype(np.float64)[sr:4 * sr]
    return clean, clean + 0.05 * np.random.default_rng(1).standard_normal(clean.size)


@pytest.mark.parametrize("sr", sorted(URGENT))
def test_the_numbers_of_urgent_2025(sr):
    clean, noisy = _pair(sr)
    matched, unmatched = URGENT[sr]
    assert mcd(clean, noisy, sr) == pytest.approx(matched, abs=1e-10)
    assert mcd(clean, noisy, sr, match_gain=False) == pytest.approx(unmatched, abs=1e-10)


def test_digital_silence_is_analysed_on_the_periodogram_floor():
    """1 s of zeros either side of both signals: SPTK's 1e-6 floor on the
    periodogram is what the silent frames are analysed on. URGENT's number,
    as above."""
    clean, noisy = _pair(8000)
    pad = np.zeros(8000)
    assert mcd(np.r_[pad, clean, pad], np.r_[pad, noisy, pad], 8000) == pytest.approx(
        3.2028193092000086, abs=1e-10)


def test_a_copy_scores_zero_and_a_scaled_copy_zero_after_the_gain_match():
    clean, _ = _pair(16000)
    assert mcd(clean, clean, 16000) == pytest.approx(0.0, abs=1e-9)
    assert mcd(clean, 0.5 * clean, 16000) == pytest.approx(0.0, abs=1e-6)
    assert mcd(clean, 0.5 * clean, 16000, match_gain=False) > 2.0


def test_fastdtw_pairs_frames_as_the_package_does():
    """The example of fastdtw's own docstring."""
    x = np.array([1, 2, 3, 4, 5], dtype=float)[:, None]
    y = np.array([2, 3, 4], dtype=float)[:, None]
    assert module._fastdtw(x, y) == [(0, 0), (1, 0), (2, 1), (3, 2), (4, 2)]


def test_fastdtw_breaks_ties_as_the_pure_python_package_does():
    """Frames that repeat exactly, as digital silence does, leave paths of
    equal cost; these are the paths fastdtw 0.3.4's pure-Python version
    picks, each way round."""
    x = np.array([[0, 0], [0, 0], [0, 0], [1, 0], [1, 0], [2, 1], [0, 0], [0, 0]], float)
    y = np.array([[0, 0], [0, 0], [1, 0], [1, 0], [1, 0], [2, 1], [2, 1], [0, 0], [0, 0],
                  [0, 0]], float)
    assert module._fastdtw(x, y) == [(0, 0), (0, 1), (1, 1), (2, 1), (3, 2), (3, 3), (3, 4),
                                     (4, 4), (5, 5), (5, 6), (6, 7), (6, 8), (6, 9), (7, 9)]
    assert module._fastdtw(y, x) == [(0, 0), (0, 1), (0, 2), (1, 2), (2, 3), (2, 4), (3, 4),
                                     (4, 4), (5, 5), (6, 5), (7, 6), (7, 7), (8, 7), (9, 7)]


def test_signals_of_different_lengths_are_warped_without_the_gain_match():
    clean, noisy = _pair(16000)
    stretched = np.interp(np.linspace(0, clean.size - 1, int(clean.size * 1.2)),
                          np.arange(clean.size), clean)
    value = mcd(clean, stretched, 16000, match_gain=False)
    assert 0.0 < value < mcd(clean, noisy, 16000, match_gain=False)
    with pytest.raises(ValueError, match=r"^mcd\(\): ref is 48000 samples and deg 144000"):
        mcd(clean, np.tile(clean, 3), 16000)


def test_a_rate_without_settings_needs_order_and_alpha():
    clean, noisy = _pair(16000)
    with pytest.raises(ValueError, match=r"^mcd\(\): URGENT's MCD has settings for"):
        mcd(clean, noisy, 11025)
    assert math.isfinite(mcd(clean, noisy, 11025, order=24, alpha=0.4))
    assert mcd(clean, noisy, 16000, order=23, alpha=0.42) == mcd(clean, noisy, 16000)
    with pytest.raises(ValueError, match=r"^mcd\(\): pass order and alpha together"):
        mcd(clean, noisy, 16000, order=23)


@pytest.mark.parametrize("kwargs, said", [
    ({'alpha': 1.0, 'order': 23}, "alpha has to be"),
    ({'alpha': 0.42, 'order': 0}, "order has to be"),
    ({'n_fft': 32}, "n_fft"),
    ({'hop_length': 0}, "hop_length"),
    ({'match_gain': 'yes'}, "match_gain"),
])
def test_settings_out_of_range_are_refused_by_name(kwargs, said):
    clean, noisy = _pair(16000)
    with pytest.raises(ValueError, match=rf"^mcd\(\): .*{said}"):
        mcd(clean, noisy, 16000, **kwargs)


def test_a_signal_shorter_than_a_frame_is_nan_with_one_warning():
    clean, _ = _pair(16000)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = mcd(clean[:1000], clean[:1000], 16000)
    said = [str(w.message) for w in caught]
    assert math.isnan(value)
    assert len(said) == 1 and said[0].startswith("mcd(): ref needs at least one frame")
    assert said[0].endswith("; scored nan")


def test_the_settings_by_rate_are_urgents():
    assert kudio.MCD_SETTINGS == {8000: (13, 0.31), 16000: (23, 0.42), 22050: (34, 0.45),
                                  24000: (34, 0.46), 32000: (36, 0.50), 44100: (39, 0.53),
                                  48000: (39, 0.55)}
