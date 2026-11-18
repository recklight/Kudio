# -*- coding: utf-8 -*-
"""The Hu and Loizou measures: what a perfect copy and a change of level score,
agreement with pysepm and with the definitions computed another way, and the
input each one refuses."""
from __future__ import annotations

import math
import re
import sys
import tracemalloc
import types
import warnings

import numpy as np
import pytest
from scipy.linalg import solve_toeplitz, toeplitz
from scipy.signal import butter, lfilter, sosfilt

import kudio
from kudio.core import quality
from kudio.core.quality import (
    CompositeScore,
    cepstral_distance,
    composite,
    fw_segmental_snr,
    itakura_saito,
    llr,
    wss,
)
from kudio.exceptions import DependencyError

#: rough F1, F2, F3 of five vowels, in Hz
_VOWELS = ((730, 1090, 2440), (270, 2290, 3010), (300, 870, 2240),
           (530, 1840, 2480), (570, 840, 2410))


def _speech(sr: int, seconds: float = 2.0, seed: int = 0) -> np.ndarray:
    """Syllables of a harmonic source through three formant resonators, the
    pitch gliding inside each and the formants moving from one vowel to the
    next, with pauses over a noise floor 60 dB below the peak.

    Spectral-envelope measures need an envelope that changes: a steady tone
    gives LPC one frame shape to fit, and every frame of it agrees with every
    other.
    """
    rng = np.random.default_rng(seed)
    n = int(sr * seconds)
    y = np.zeros(n)
    start, k = int(0.1 * sr), 0
    while start < n - int(0.35 * sr):
        length = int(rng.uniform(0.15, 0.3) * sr)
        f0 = np.linspace(*rng.uniform(100.0, 220.0, 2), length)
        phase = 2 * np.pi * np.cumsum(f0) / sr
        source = sum(np.sin(h * phase) / h
                     for h in range(1, int(0.45 * sr / f0.max())))
        syllable = np.zeros(length)
        for j, part in enumerate(np.array_split(np.arange(length), 4)):
            piece = source[part]
            for f_from, f_to, bw in zip(_VOWELS[k % 5], _VOWELS[(k + 1) % 5],
                                        (80.0, 100.0, 140.0)):
                f = f_from + (f_to - f_from) * j / 3
                r = math.exp(-math.pi * bw / sr)
                piece = lfilter([1 - r], [1, -2 * r * math.cos(2 * math.pi * f / sr),
                                          r * r], piece)
            syllable[part] = piece
        y[start:start + length] = syllable * np.hanning(length)
        start += length + int(rng.uniform(0.05, 0.2) * sr)
        k += 1
    y *= 0.5 / np.max(np.abs(y))
    return y + 5e-4 * rng.standard_normal(n)


def _noise(n: int, seed: int, colour: str = 'white') -> np.ndarray:
    w = np.random.default_rng(seed).standard_normal(n)
    if colour == 'pink':
        spectrum = np.fft.rfft(w)
        f = np.arange(len(spectrum), dtype=float)
        f[0] = 1.0
        w = np.fft.irfft(spectrum / np.sqrt(f), n)
    return w / np.std(w)


def _mix(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    scale = np.sqrt(np.mean(clean ** 2) / (np.mean(noise ** 2) * 10 ** (snr_db / 10)))
    return clean + scale * noise


def _pair(sr: int, snr_db: float = 10.0, colour: str = 'white', seconds: float = 2.0):
    clean = _speech(sr, seconds)
    return clean, _mix(clean, _noise(len(clean), 1, colour), snr_db)


def _padded(sr: int) -> np.ndarray:
    """2 s of speech with 0.5 s of digital silence either side: 396 frames at
    either rate, 125 of them zeros throughout."""
    return np.concatenate([np.zeros(sr // 2), _speech(sr, 2.0), np.zeros(sr // 2)])


MEASURES = [llr, itakura_saito, cepstral_distance, wss, fw_segmental_snr]


# -- what a perfect copy and a change of level score ---------------------------

@pytest.mark.parametrize("sr", [8000, 16000])
def test_a_perfect_copy_scores_no_distortion_and_the_snr_cap(sr):
    clean = _speech(sr)
    assert llr(clean, clean, sr) == pytest.approx(0.0, abs=1e-12)
    assert itakura_saito(clean, clean, sr) == pytest.approx(0.0, abs=1e-12)
    assert cepstral_distance(clean, clean, sr) == pytest.approx(0.0, abs=1e-12)
    assert wss(clean, clean, sr) == pytest.approx(0.0, abs=1e-12)
    assert fw_segmental_snr(clean, clean, sr) == 35.0


@pytest.mark.parametrize("sr", [8000, 16000])
def test_level_and_polarity_cost_llr_cep_wss_and_fwsegsnr_nothing(sr):
    """Each of the four reads the shape of the spectrum only: LPC filters do
    not see a gain, the cepstrum leaves out its zeroth coefficient, WSS
    compares slopes in dB and fwSegSNR scales both spectra to unit area."""
    clean = _speech(sr)
    for deg in (2.0 * clean, 0.5 * clean, -clean):
        assert llr(clean, deg, sr) == pytest.approx(0.0, abs=1e-9)
        assert cepstral_distance(clean, deg, sr) == pytest.approx(0.0, abs=1e-6)
        assert wss(clean, deg, sr) == pytest.approx(0.0, abs=1e-6)
        assert fw_segmental_snr(clean, deg, sr) == 35.0


@pytest.mark.parametrize("sr", [8000, 16000])
def test_itakura_saito_charges_a_quieter_copy_more_than_a_louder_one(sr):
    """At twice the amplitude every frame's LPC spectrum is 4x the clean one,
    which is 1/4 - ln(1/4) - 1; at half, 4 - ln 4 - 1."""
    clean = _speech(sr)
    louder = itakura_saito(clean, 2.0 * clean, sr)
    quieter = itakura_saito(clean, 0.5 * clean, sr)
    assert louder == pytest.approx(0.25 + math.log(4.0) - 1.0, abs=1e-9)
    assert quieter == pytest.approx(4.0 - math.log(4.0) - 1.0, abs=1e-9)


@pytest.mark.parametrize("sr", [8000, 16000])
def test_more_noise_scores_worse_on_every_measure(sr):
    clean = _speech(sr)
    noise = _noise(len(clean), 1, 'pink')
    by_snr = {snr: [m(clean, _mix(clean, noise, snr), sr) for m in MEASURES]
              for snr in (20, 10, 5, 0)}
    for better, worse in ((20, 10), (10, 5), (5, 0)):
        *distances, fwseg = by_snr[better]
        *more, fwseg_lower = by_snr[worse]
        assert all(d < m for d, m in zip(distances, more)), (better, worse)
        assert fwseg > fwseg_lower, (better, worse)


# -- the numbers of other implementations, and the definitions themselves ------

#: pysepm_evo 0.1.1 (Python 3.10, numpy 2.2.6, scipy 1.15.2) on
#: ``_pair(sr, 10, 'white')`` and ``_pair(sr, 5, 'pink')``: its llr, its llr
#: with used_for_composite=True, wss, cepstrum_distance, fwSNRseg and SNRseg.
#: Both pairs are 2 s, which is 262 frames at either rate, and 0.95 * 262 =
#: 248.9 does not end in .5, where pysepm's rounding and MATLAB's part (see
#: test_the_best_95_percent_rounds_halves_up_the_way_matlab_does).
PYSEPM_EVO_0_1_1 = {
    (8000, 10, 'white'): {
        'llr': 1.1718670598342014,
        'llr_composite': 1.3896920535039994,
        'wss': 35.978993786442956,
        'cep': 6.539503644858894,
        'fwsegsnr': 6.734822117006508,
        'segsnr_composite': 0.08950560791750141,
    },
    (8000, 5, 'pink'): {
        'llr': 1.0793230703713834,
        'llr_composite': 1.1276614778289964,
        'wss': 58.63723299992713,
        'cep': 6.704734120051547,
        'fwsegsnr': 5.857250647113347,
        'segsnr_composite': -2.318112449687568,
    },
    (16000, 10, 'white'): {
        'llr': 1.2152022856269244,
        'llr_composite': 1.9987371410391976,
        'wss': 30.898754137997887,
        'cep': 6.339864785704115,
        'fwsegsnr': 6.587486329701602,
        'segsnr_composite': -0.15799247542422712,
    },
    (16000, 5, 'pink'): {
        'llr': 1.206665294179598,
        'llr_composite': 1.6207468310939142,
        'wss': 56.73095337920859,
        'cep': 6.503401150803159,
        'fwsegsnr': 4.343724758391272,
        'segsnr_composite': -2.52801433955951,
    },
}


@pytest.mark.parametrize("sr, snr_db, colour", [
    (8000, 10, 'white'), (8000, 5, 'pink'), (16000, 10, 'white'), (16000, 5, 'pink')])
def test_the_measures_match_pysepm_evo(sr, snr_db, colour):
    want = PYSEPM_EVO_0_1_1[(sr, snr_db, colour)]
    clean, noisy = _pair(sr, snr_db, colour)
    parts = composite(clean, noisy, sr, pesq=2.0)
    got = {'llr': llr(clean, noisy, sr), 'llr_composite': parts.llr,
           'wss': wss(clean, noisy, sr), 'cep': cepstral_distance(clean, noisy, sr),
           'fwsegsnr': fw_segmental_snr(clean, noisy, sr),
           'segsnr_composite': parts.segsnr}
    for name, value in want.items():
        assert got[name] == pytest.approx(value, rel=1e-9, abs=1e-9), name


#: pysepm_evo 0.1.1's wss (same versions as above) on ``_padded(sr)`` under
#: white noise at 10 dB SNR.
PYSEPM_EVO_0_1_1_WSS_PADDED = {8000: 27.86688153953944, 16000: 25.68243750327955}


@pytest.mark.parametrize("sr", [8000, 16000])
def test_wss_matches_pysepm_evo_where_the_reference_is_digital_silence(sr):
    """The zeros put every clean band of 125 frames far below the -100 dB
    floor of comp_wss.m, where any floor flattens them alike. The frames at
    the edges of the speech, four at 8 kHz and two at 16 kHz, have bands
    between -100 and -60 dB, which a floor at -60 dB would flatten too."""
    padded = _padded(sr)
    noisy = _mix(padded, _noise(len(padded), 1), 10)
    with pytest.warns(UserWarning, match=r"^wss\(\): 125 of the reference's 396 frames"):
        got = wss(padded, noisy, sr)
    assert got == pytest.approx(PYSEPM_EVO_0_1_1_WSS_PADDED[sr], rel=1e-9)


@pytest.mark.parametrize("sr", [8000, 16000])
def test_wss_reads_a_pair_40_db_down_as_it_reads_it_at_full_level(sr):
    """Band energies go to dB with a floor at -100 dB (comp_wss.m). 40 dB
    down, the quietest clean band here is at -97.6 dB at 8 kHz and -94.4 dB
    at 16 kHz, still above it, so the score does not move. A floor at -60 dB
    would flatten a band in 252 of the 262 frames and take 35.98 to 16.88 at
    8 kHz."""
    clean, noisy = _pair(sr, 10, 'white')
    assert wss(0.01 * clean, 0.01 * noisy, sr) == pytest.approx(
        wss(clean, noisy, sr), rel=1e-9)


def _frames(y: np.ndarray, sr: int) -> np.ndarray:
    """Built here from the definition rather than taken from the module: 30 ms,
    a quarter-frame hop, the code's window, eps added, the last frame left out."""
    size = int(round(0.03 * sr))
    hop = size // 4
    window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(1, size + 1) / (size + 1)))
    y = y + np.finfo(float).eps
    return np.array([y[i * hop:i * hop + size] * window
                     for i in range((len(y) - size) // hop)])


def _lpc_spectra(frames: np.ndarray, order: int, points: int = 1 << 16):
    """``(|A|^2 on a grid of points round the unit circle, prediction-error
    power)`` per frame, LPC solved with scipy's Toeplitz solver."""
    shapes, powers = [], []
    for frame in frames:
        r = np.array([frame[:len(frame) - k] @ frame[k:] for k in range(order + 1)])
        alpha = solve_toeplitz(r[:order], r[1:])
        shapes.append(np.abs(np.fft.fft(np.concatenate([[1.0], -alpha]), points)) ** 2)
        powers.append(r[0] - alpha @ r[1:])
    return np.array(shapes), np.array(powers)


def _mean_of_best(values: np.ndarray) -> float:
    ordered = np.sort(values)
    return float(np.mean(ordered[:int(math.floor(0.95 * len(ordered) + 0.5))]))


@pytest.mark.parametrize("sr", [8000, 16000])
def test_itakura_saito_is_the_divergence_between_the_two_lpc_spectra(sr):
    """No other implementation to check it against, so the integral itself:
    the mean over 65536 points of P_c/P_p - ln(P_c/P_p) - 1, with P = s/|A|^2
    and s the prediction-error power, for each frame of each signal."""
    clean, noisy = _pair(sr, 5, 'pink', seconds=1.0)
    order = {8000: 10, 16000: 16}[sr]
    shape_c, power_c = _lpc_spectra(_frames(clean, sr), order)
    shape_p, power_p = _lpc_spectra(_frames(noisy, sr), order)
    ratio = (power_c[:, None] / shape_c) / (power_p[:, None] / shape_p)
    per_frame = np.mean(ratio - np.log(ratio) - 1.0, axis=1)
    want = _mean_of_best(np.clip(per_frame, 0.0, 100.0))
    assert itakura_saito(clean, noisy, sr) == pytest.approx(want, rel=1e-9)


@pytest.mark.parametrize("sr", [8000, 16000])
def test_itakura_saito_holds_digital_silence_at_0_whatever_the_gain(sr):
    """Where both frames are zeros, comp_is.m's eps floors make the frame -1
    and the [0, 100] of [HL08] makes it 0. The best 376 of the 396 frames then
    hold the 125 silent ones and 251 of the 271 with signal, each of those at
    1/4 + ln 4 - 1 for twice the amplitude and 4 - ln 4 - 1 for half. Without
    the bound the exact copy would score -125/376 = -0.33."""
    padded = _padded(sr)
    assert itakura_saito(padded, padded, sr) == pytest.approx(0.0, abs=1e-12)
    for gain, per_frame in ((2.0, 0.25 + math.log(4.0) - 1.0),
                            (0.5, 4.0 - math.log(4.0) - 1.0)):
        assert itakura_saito(padded, gain * padded, sr) == pytest.approx(
            251 * per_frame / 376, abs=1e-9), gain


def _comp_is(clean: np.ndarray, processed: np.ndarray, order: int) -> np.ndarray:
    """comp_is.m's value for each pair of frames, LPC by scipy's Toeplitz
    solver: both prediction-error powers and the clean frame's own quadratic
    form floored at eps, and no range applied yet."""
    eps = np.finfo(float).eps
    values = []
    for c, p in zip(clean, processed):
        r_c = np.array([c[:len(c) - k] @ c[k:] for k in range(order + 1)])
        r_p = np.array([p[:len(p) - k] @ p[k:] for k in range(order + 1)])
        a_c = np.concatenate([[1.0], -solve_toeplitz(r_c[:order], r_c[1:])])
        a_p = np.concatenate([[1.0], -solve_toeplitz(r_p[:order], r_p[1:])])
        matrix = toeplitz(r_c)
        gain_c, gain_p = max(r_c @ a_c, eps), max(r_p @ a_p, eps)
        ratio = (a_p @ matrix @ a_p) / max(a_c @ matrix @ a_c, eps)
        values.append(gain_c / gain_p * ratio + math.log(gain_p / gain_c) - 1.0)
    return np.array(values)


@pytest.mark.parametrize("sr", [8000, 16000])
def test_itakura_saito_floors_the_reference_power_at_eps_as_comp_is_m_does(sr):
    """Where the reference is zeros its prediction-error power is below eps,
    and comp_is.m's floor stands in for it: against the noise, each of the
    125 silent frames scores ln(s_p / eps) - 1, 31.1 to 31.7 at 8 kHz and
    31.9 to 32.4 at 16 kHz. Without the floor on the clean side the pair
    would score 26.47 at 8 kHz instead of 11.78."""
    padded = _padded(sr)
    noisy = _mix(padded, _noise(len(padded), 1), 10)
    order = {8000: 10, 16000: 16}[sr]
    per_frame = _comp_is(_frames(padded, sr), _frames(noisy, sr), order)
    spread = {8000: "31.1 to 31.7", 16000: "31.9 to 32.4"}[sr]
    with pytest.warns(UserWarning, match=f"125 of the reference's 396 frames .* "
                                         f"they score {spread}\\."):
        got = itakura_saito(padded, noisy, sr)
    assert got == pytest.approx(_mean_of_best(np.clip(per_frame, 0.0, 100.0)), rel=1e-9)


@pytest.mark.parametrize("sr", [8000, 16000])
def test_llr_and_the_cepstral_distance_agree_with_their_spectral_forms(sr):
    """LLR's ratio of quadratic forms is the mean of |A_p|^2 / |A_c|^2 round
    the unit circle, and the LPC cepstrum is the inverse transform of
    -ln |A|^2: neither goes through the recursions the module uses."""
    clean, noisy = _pair(sr, 10, 'white', seconds=1.0)
    order = {8000: 10, 16000: 16}[sr]
    shape_c, _ = _lpc_spectra(_frames(clean, sr), order)
    shape_p, _ = _lpc_spectra(_frames(noisy, sr), order)
    llr_frames = np.log(np.mean(shape_p / shape_c, axis=1))
    assert llr(clean, noisy, sr) == pytest.approx(
        _mean_of_best(np.clip(llr_frames, 0.0, 2.0)), rel=1e-9)
    cep_c = np.fft.ifft(-np.log(shape_c), axis=1).real[:, 1:order + 1]
    cep_p = np.fft.ifft(-np.log(shape_p), axis=1).real[:, 1:order + 1]
    cep_frames = 10 / math.log(10) * np.sqrt(2 * np.sum((cep_c - cep_p) ** 2, axis=1))
    assert cepstral_distance(clean, noisy, sr) == pytest.approx(
        _mean_of_best(np.clip(cep_frames, 0.0, 10.0)), rel=1e-9)


def _voiced_without_noise(sr: int, seconds: float = 1.0) -> np.ndarray:
    """29 harmonics of 120 Hz with a wandering pitch, shaped by three formant
    bumps and cut into 4 Hz syllables, and no noise at all. Order-16 LPC
    predicts some of its frames down to round-off."""
    t = np.arange(int(sr * seconds)) / sr
    offset = np.random.default_rng(0).uniform(0, 6.28)
    f0 = 120.0 * (1 + 0.08 * np.sin(2 * np.pi * 0.7 * t + offset)
                  + 0.03 * np.sin(2 * np.pi * 3.1 * t))
    phase = 2 * np.pi * np.cumsum(f0) / sr
    y = np.zeros(len(t))
    for k in range(1, 30):
        level = sum(np.exp(-((k * 120.0 - fc) / bw) ** 2)
                    for fc, bw in ((500, 300), (1500, 400), (2500, 500)))
        y += (level + 0.05) / k * np.sin(k * phase)
    y *= 0.5 * (1 - np.cos(2 * np.pi * 4.0 * t))
    return 0.3 * y / np.max(np.abs(y))


def test_llr_scores_a_frame_predicted_to_round_off_instead_of_losing_the_pair():
    """Frame 35 of this signal has a clean prediction error of -7.1e-10
    against a frame energy of 0.0277: round-off below zero, whose log is NaN,
    and that one frame took the whole pair and the three composite ratings
    with it."""
    clean = _voiced_without_noise(16000)
    noisy = _mix(clean, _noise(len(clean), 1), 20.0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        score = llr(clean, noisy, 16000)
        parts = composite(clean, noisy, 16000, pesq=2.0)
        copy = llr(clean, clean, 16000)
    assert not [w for w in caught if 'NaN' in str(w.message)]
    assert 0.0 < score <= 2.0
    assert np.isfinite(parts.llr) and np.isfinite(parts.covl)
    assert copy == 0.0


def test_the_best_95_percent_rounds_halves_up_the_way_matlab_does():
    """0.95 * 30 = 28.5: MATLAB's round keeps 29 frames and Python's would
    keep 28, which is where pysepm and the MATLAB code part."""
    assert quality._best(np.arange(30.0)) == pytest.approx(np.mean(np.arange(29.0)))
    assert quality._best(np.arange(20.0)) == pytest.approx(np.mean(np.arange(19.0)))
    assert quality._best(np.array([7.0])) == 7.0


def test_the_wss_peak_on_a_rising_flank_stops_one_band_short_as_the_code_does():
    """Levels 10, 20, 30, 25, 15: band 0 rises to the peak at band 2, and
    comp_wss.m reads the level of band 1 there. From the falling side, band 3
    finds the peak itself."""
    level = np.array([[10.0, 20.0, 30.0, 25.0, 15.0]])
    peaks = quality._nearest_peaks(level, np.diff(level, axis=1))
    assert peaks.tolist() == [[20.0, 20.0, 30.0, 30.0]]


def test_the_bands_of_wss_and_fwsegsnr_end_below_4_khz_at_both_rates():
    for sr in (8000, 16000):
        nfft, bands = quality._bands(sr)
        top = np.flatnonzero(bands.any(axis=0)).max()
        assert top * sr / nfft == 3812.5
        assert bands.shape == (25, nfft // 2)


def test_at_16_khz_fwsegsnr_feels_what_happens_above_4_khz_and_wss_does_not():
    """The bands stop at 3.8 kHz, but fwSegSNR first scales each magnitude
    spectrum to unit area over every bin up to 8 kHz. Noise kept above 5 kHz
    (an 8th-order Butterworth high-pass) at 20 dB SNR takes the copy's 35 dB
    to 8.58, and cutting everything above 4 kHz, 0.034% of this speech's
    energy, takes it to 17.88, while WSS reads 0.021 and 0."""
    sr = 16000
    clean = _speech(sr)
    noise = sosfilt(butter(8, 5000, 'highpass', fs=sr, output='sos'),
                    _noise(len(clean), 1))
    hiss = _mix(clean, noise, 20)
    assert fw_segmental_snr(clean, hiss, sr) == pytest.approx(8.58, abs=0.01)
    assert wss(clean, hiss, sr) < 0.03
    spectrum = np.fft.rfft(clean)
    spectrum[np.fft.rfftfreq(len(clean), 1.0 / sr) > 4000] = 0.0
    muffled = np.fft.irfft(spectrum, len(clean))
    assert fw_segmental_snr(clean, muffled, sr) == pytest.approx(17.88, abs=0.01)
    assert wss(clean, muffled, sr) == pytest.approx(0.0, abs=1e-6)


# -- the composite -------------------------------------------------------------

@pytest.mark.parametrize("sr", [8000, 16000])
def test_the_composite_is_the_published_regression_on_its_own_parts(sr):
    """Hu and Loizou, Interspeech 2006, Sec. 4.1, with a PESQ passed in, so it
    runs without the pesq package."""
    clean, noisy = _pair(sr, 10, 'white')
    got = composite(clean, noisy, sr, pesq=2.31)
    assert isinstance(got, CompositeScore) and got.pesq == 2.31
    assert got.csig == pytest.approx(3.093 - 1.029 * got.llr + 0.603 * 2.31
                                     - 0.009 * got.wss)
    assert got.cbak == pytest.approx(1.634 + 0.478 * 2.31 - 0.007 * got.wss
                                     + 0.063 * got.segsnr)
    assert got.covl == pytest.approx(1.594 + 0.805 * 2.31 - 0.512 * got.llr
                                     - 0.007 * got.wss)
    assert all(1.0 < v < 5.0 for v in got[:3]), got


def test_the_composite_ratings_stay_on_the_one_to_five_scale():
    """A perfect copy with the best raw PESQ there is would rate 5.81, 5.99
    and 5.22 by the equations; a listener's scale stops at 5."""
    clean = _speech(16000)
    raw = dict(mode='nb', scale='raw')
    assert composite(clean, clean, 16000, pesq=4.5, **raw)[:3] == (5.0, 5.0, 5.0)
    hopeless = composite(clean, _noise(len(clean), 2), 16000, pesq=-0.5, **raw)
    assert min(hopeless[:3]) == 1.0


def test_the_composite_takes_wss_as_it_is_and_llr_without_the_cap_at_2():
    clean, noisy = _pair(16000, 0, 'white')
    parts = composite(clean, noisy, 16000, pesq=1.5)
    assert parts.wss == wss(clean, noisy, 16000)
    capped = llr(clean, noisy, 16000)
    assert parts.llr > capped, "at 0 dB some frames pass 2, and only llr() caps them"


def test_the_composite_segmental_snr_is_not_kudio_segmental_snr():
    """Windowed 30 ms frames against 512 rectangular samples, which at 8 kHz
    are 64 ms. Under white noise at 20 dB the composite's reads 6.02 dB and
    kudio.segmental_snr 8.08: two dB that would go into CBAK at 0.063 each."""
    clean, noisy = _pair(8000, 20, 'white')
    own = composite(clean, noisy, 8000, pesq=2.0).segsnr
    assert own == pytest.approx(6.02, abs=0.01)
    assert kudio.segmental_snr(clean, noisy) - own > 2.0


def test_zero_padding_in_the_reference_pulls_the_composite_segsnr_down():
    """A frame of digital silence carries less than eps of energy, and the
    code's ``10 log10(S / (N + eps) + eps)`` puts it at the -10 dB floor even
    against an exact copy. Half a second of zeros either side of 2 s does that
    to 125 of the 396 frames, and the copy scores 20.8 dB instead of 35."""
    speech = _speech(16000, 2.0)
    padded = np.concatenate([np.zeros(8000), speech, np.zeros(8000)])
    assert composite(speech, speech, 16000, pesq=4.5).segsnr == 35.0
    with pytest.warns(UserWarning, match="segSNR scores them -10 dB each"):
        got = composite(padded, padded, 16000, pesq=4.5)
    assert got.segsnr == pytest.approx((271 * 35.0 - 125 * 10.0) / 396)


def test_the_composite_without_a_pesq_score_needs_the_pesq_package(monkeypatch):
    clean, noisy = _pair(16000)
    monkeypatch.setitem(sys.modules, 'pesq', None)
    with pytest.raises(DependencyError, match=r"kudio\[pesq\]"):
        composite(clean, noisy, 16000)
    assert np.isfinite(composite(clean, noisy, 16000, pesq=2.0).csig)


@pytest.mark.parametrize("kwargs, match", [
    (dict(pesq=4.6, mode='nb', scale='raw'), "mode='wb', scale='lqo'"),
    (dict(pesq=0.8), "scale='raw'"),
    (dict(pesq=0.8, scale='lqo'), "scale='raw'"),
    (dict(scale='mos'), "scale must be 'auto', 'lqo' or 'raw'"),
    (dict(pesq=-0.7), "not a PESQ score"),
    (dict(pesq=45.0, scale='lqo'), "not a PESQ score"),
    (dict(pesq=float('nan')), "has to be a score"),
    (dict(pesq=True), "has to be a number"),
    (dict(pesq='2.5'), "has to be a number"),
    (dict(mode='wb', scale='raw'), "no raw score"),
])
def test_the_composite_refuses_a_pesq_no_variant_can_produce(kwargs, match):
    clean, noisy = _pair(16000)
    with pytest.raises(ValueError, match=match):
        composite(clean, noisy, 16000, **kwargs)


def test_the_composite_refuses_wideband_pesq_at_8_khz():
    clean, noisy = _pair(8000)
    with pytest.raises(ValueError, match="16 kHz"):
        composite(clean, noisy, 8000, mode='wb', scale='lqo', pesq=2.0)


def test_the_composite_takes_the_pesq_published_figures_were_made_with():
    """Wideband MOS-LQO at 16 kHz, as pysepm's pesq() and Loizou's 2012
    composite.m feed it, and raw narrowband P.862 at 8 kHz, the score the regression was
    fitted on. Under white noise at 20 dB the two PESQ scores at 16 kHz are
    1.86 and 2.59, and no rating reaches the ends of the scale, so each one
    moves by its PESQ coefficient times the difference."""
    pytest.importorskip("pesq")
    clean, noisy = _pair(16000, 20, 'white')
    default = composite(clean, noisy, 16000)
    assert default.pesq == kudio.pesq(clean, noisy, 16000, mode='wb')
    fitted = composite(clean, noisy, 16000, mode='nb', scale='raw')
    assert fitted.pesq == kudio.pesq(clean, noisy, 16000, mode='nb', scale='raw')
    gap = fitted.pesq - default.pesq
    assert gap > 0.5
    for field, coefficient in (('csig', 0.603), ('cbak', 0.478), ('covl', 0.805)):
        assert getattr(fitted, field) - getattr(default, field) == pytest.approx(
            coefficient * gap, abs=1e-9), field
    clean8, noisy8 = _pair(8000, 20, 'white')
    assert composite(clean8, noisy8, 8000).pesq == kudio.pesq(
        clean8, noisy8, 8000, mode='nb', scale='raw')


@pytest.mark.parametrize("sr, kwargs, mode, raw", [
    (16000, {}, 'wb', False),
    (8000, {}, 'nb', True),
    (16000, dict(mode='nb'), 'nb', True),
    (16000, dict(mode='nb', scale='raw'), 'nb', True),
    (16000, dict(scale='lqo'), 'wb', False),
    (8000, dict(scale='lqo'), 'nb', False),
])
def test_which_pesq_each_setting_runs(monkeypatch, sr, kwargs, mode, raw):
    """Run against a stand-in for the pesq package, so CI checks the choice
    without the package: the mode it is called with, and whether its MOS-LQO
    is taken back to the raw score through P.862.1."""
    called = []
    module = types.ModuleType('pesq')
    module.BufferTooShortError = _BufferTooShortError
    module.NoUtterancesError = _NoUtterancesError

    def score(fs, ref, deg, given):
        called.append((fs, given))
        return 2.5

    module.pesq = score
    monkeypatch.setitem(sys.modules, 'pesq', module)
    clean, noisy = _pair(sr)
    got = composite(clean, noisy, sr, **kwargs)
    assert called == [(sr, mode)]
    inverted = (4.6607 - math.log(4.0 / (2.5 - 0.999) - 1.0)) / 1.4945
    assert got.pesq == pytest.approx(inverted if raw else 2.5, abs=1e-12)


@pytest.mark.parametrize("sr", [8000, 16000])
def test_a_pair_the_pesq_package_cannot_score_gets_nan_ratings_and_keeps_the_rest(sr):
    """pesq 0.0.4 raises on all three: an output of digital silence, whose
    score it gets as NaN; 0.25 s of speech, in which it finds no utterance;
    and 0.2 s, under its 0.25 s minimum. LLR, WSS and segSNR need no PESQ."""
    pytest.importorskip("pesq")
    clean = _speech(sr)
    for ref, deg, cause in ((clean, np.zeros_like(clean), "digital silence"),
                            (clean[:sr // 4], clean[:sr // 4], "no utterance"),
                            (clean[:sr // 5], clean[:sr // 5], "0.25 s")):
        with pytest.warns(UserWarning, match=cause):
            got = composite(ref, deg, sr)
        assert np.isnan(got[:4]).all(), (cause, got)
        assert got[4:] == composite(ref, deg, sr, pesq=2.0)[4:], cause


class _BufferTooShortError(RuntimeError):
    """pesq.BufferTooShortError, for the stand-in package below."""


class _NoUtterancesError(RuntimeError):
    """pesq.NoUtterancesError, for the stand-in package below."""


def _stand_in_pesq(monkeypatch, answer) -> None:
    """Put a module where the pesq package would be, with the two error
    classes pesq 0.0.4 exports, whose pesq() raises *answer* if it is an
    exception and returns it otherwise."""
    module = types.ModuleType('pesq')
    module.BufferTooShortError = _BufferTooShortError
    module.NoUtterancesError = _NoUtterancesError

    def score(fs, ref, deg, mode):
        if isinstance(answer, Exception):
            raise answer
        return answer

    module.pesq = score
    monkeypatch.setitem(sys.modules, 'pesq', module)


@pytest.mark.parametrize("cause", ["0.25 s", "no utterance", "digital silence",
                                   "came back nan"])
def test_composite_turns_each_pesq_failure_into_nan_ratings_without_the_package(
        monkeypatch, cause):
    """The three failures of the test above, raised by a stand-in for the
    pesq package so that this runs where pesq is not built (CI leaves it
    out), and a fourth: a NaN returned instead of raised."""
    clean = _speech(8000)
    ref, deg, answer = {
        "0.25 s": (clean[:1600], clean[:1600], _BufferTooShortError()),
        "no utterance": (clean[:2000], clean[:2000], _NoUtterancesError()),
        "digital silence": (clean, np.zeros_like(clean),
                            ValueError("cannot convert float NaN to integer")),
        "came back nan": (clean, clean, float('nan')),
    }[cause]
    _stand_in_pesq(monkeypatch, answer)
    with pytest.warns(UserWarning, match=cause):
        got = composite(ref, deg, 8000)
    assert np.isnan(got[:4]).all(), got
    assert got[4:] == composite(ref, deg, 8000, pesq=2.0)[4:]


def test_composite_lets_any_other_error_from_the_pesq_package_through(monkeypatch):
    """Only the NaN the package gets on a silent output is read as a pair it
    cannot score; a ValueError about anything else is a fault and stays one."""
    _stand_in_pesq(monkeypatch, ValueError("mode should be either 'nb' or 'wb'"))
    clean, noisy = _pair(8000)
    with pytest.raises(ValueError, match="mode should be"):
        composite(clean, noisy, 8000)


# -- what is refused, and what scores nan --------------------------------------

@pytest.mark.parametrize("measure", MEASURES + [composite])
@pytest.mark.parametrize("sr", [11025, 22050, 44100, 48000])
def test_rates_the_measures_are_not_defined_at_are_refused(measure, sr):
    y = _speech(8000, 1.0)
    with pytest.raises(ValueError, match="resample to 16 kHz"):
        measure(y, y, sr)


@pytest.mark.parametrize("measure", MEASURES)
def test_stereo_is_refused_like_every_other_metric(measure):
    pair = np.zeros((8000, 2)) + 0.1
    with pytest.raises(ValueError, match="mono"):
        measure(pair, pair, 8000)


@pytest.mark.parametrize("measure", MEASURES)
def test_samples_that_are_not_finite_are_refused(measure):
    clean, noisy = _pair(8000)
    noisy[100] = np.nan
    with pytest.raises(ValueError, match="not finite"):
        measure(clean, noisy, 8000)


@pytest.mark.parametrize("measure", MEASURES)
def test_a_pair_too_short_for_one_frame_scores_nan_with_a_warning(measure):
    """One frame needs 240 samples plus a hop of 60 at 8 kHz, 37.5 ms."""
    y = _speech(8000, 1.0)
    with pytest.warns(UserWarning, match="37.5 ms"):
        assert np.isnan(measure(y[:299], y[:299], 8000))
    assert np.isfinite(measure(y[:300], y[:300], 8000))


def test_a_composite_that_cannot_be_scored_keeps_the_pesq_it_was_given():
    y = _speech(8000, 1.0)[:200]
    with pytest.warns(UserWarning, match="37.5 ms"):
        got = composite(y, y, 8000, pesq=3.0)
    assert got.pesq == 3.0 and all(np.isnan(v) for i, v in enumerate(got) if i != 3)


@pytest.mark.parametrize("measure", MEASURES)
def test_a_silent_reference_scores_nan_with_a_warning(measure):
    silent = np.zeros(8000)
    with pytest.warns(UserWarning, match="digital silence"):
        assert np.isnan(measure(silent, _speech(8000, 1.0), 8000))


@pytest.mark.parametrize("measure", [llr, itakura_saito, cepstral_distance, wss])
def test_a_pair_too_loud_to_measure_scores_nan_with_one_warning(measure):
    """At 1e160 of full scale a frame's power overflows. The pair is scored
    nan with kudio's warning, and nothing else: sorted with the rest, those
    frames would land at the end, where the 95% rule drops them."""
    loud = _speech(8000) * 1e160
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert np.isnan(measure(loud, 0.9 * loud, 8000))
    assert [str(w.message) for w in caught] == [
        f"{measure.__name__}(): 262 of the 262 frames came out NaN, as a frame "
        f"does when it is too loud to square in float64; scored nan"]


@pytest.mark.parametrize("measure, level, lost", [
    (wss, 1e153, 19), (wss, 1e155, 163), (itakura_saito, 1e155, 142)])
def test_a_pair_partly_too_loud_gets_kudios_warning_and_none_of_numpys(
        measure, level, lost):
    """Between 1e153 and 1e155 of full scale some frames overflow and the
    rest do not, and numpy warned about every operation that met one: five
    times for WSS at 1e153, ahead of the one warning that matters."""
    loud = _speech(8000) * level
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert np.isnan(measure(loud, 0.9 * loud, 8000))
    assert [str(w.message) for w in caught] == [
        f"{measure.__name__}(): {lost} of the 262 frames came out NaN, as a frame "
        f"does when it is too loud to square in float64; scored nan"]


def test_a_composite_too_loud_to_measure_says_so_once():
    """LLR, WSS and segSNR all overflow at 1e160, and one warning says which
    fields that leaves nan."""
    loud = _speech(8000) * 1e160
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got = composite(loud, 0.9 * loud, 8000, pesq=2.0)
    assert [str(w.message) for w in caught] == [
        "composite(): of the 262 frames, 262 in LLR, 262 in WSS and 262 in segSNR "
        "came out NaN, as a frame does when it is too loud to square in float64; "
        "CSIG, CBAK, COVL, LLR, WSS and segSNR scored nan, PESQ kept"]
    assert got.pesq == 2.0 and np.isnan([v for i, v in enumerate(got) if i != 3]).all()


# -- frames where the reference is digital silence ------------------------------

def _filled(sr: int):
    """_padded(sr), and a copy of it holding white noise at -60 dBFS RMS
    where it is zero: in 125 of the 396 frames the copy holds something and
    the reference nothing."""
    padded = _padded(sr)
    quiet = padded == 0.0
    copy = padded.copy()
    copy[quiet] = 1e-3 * _noise(len(padded), 1)[quiet]
    return padded, copy


#: what those 125 frames score at 8 kHz, as the warning puts it, and what the
#: pair scores, against 0 (35 for fwSegSNR) for the copy without the noise
_AGAINST_EPS = {
    'llr': ("2 each", 0.5626),
    'itakura_saito': ("25.4 to 26", 7.2211),
    'cepstral_distance': ("10 dB each", 2.8418),
    'wss': ("7.2 to 40.1", 5.7313),
    'fw_segmental_snr': ("-10 dB each", 20.4062),
}


@pytest.mark.parametrize("measure", MEASURES)
def test_frames_where_only_the_reference_is_digital_silence_are_named(measure):
    """There the clean frame is eps under the window, and the measure reads
    the output against that rather than against speech. For the white noise
    here, LLR, CEP and fwSegSNR put each such frame at their worst, IS at
    ln(s_p / eps) - 1, WSS at the output's own slopes."""
    padded, copy = _filled(8000)
    each, score = _AGAINST_EPS[measure.__name__]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got = measure(padded, copy, 8000)
    assert [str(w.message) for w in caught] == [
        f"{measure.__name__}(): 125 of the reference's 396 frames are digital "
        f"silence where deg is not, so those frames compare deg with eps, the "
        f"floor added to both signals, rather than with speech; they score "
        f"{each}. Trim the silence off both to compare the rest alone."]
    assert caught[0].filename == __file__
    assert got == pytest.approx(score, abs=1e-4)


#: what deg holds where _padded(8000) is zero: a DC offset, a 50 Hz hum
#: peaking at -40 dBFS, and white noise at -60 dBFS RMS
_FILLS = {
    'dc': lambda n: np.full(n, 1e-3),
    'hum': lambda n: 1e-2 * np.sin(2 * np.pi * 50 * np.arange(n) / 8000),
    'noise': lambda n: 1e-3 * _noise(n, 1),
}

_NUMBER = r"(-?\d+(?:\.\d+)?(?:e[-+]?\d+)?)"


def _against_eps(measure, fill: str):
    """The lowest and highest score of the frames where only the reference
    is digital silence, as the warning reports them."""
    padded = _padded(8000)
    deg = np.where(padded == 0.0, _FILLS[fill](len(padded)), padded)
    with pytest.warns(UserWarning, match="frames are digital silence") as caught:
        measure(padded, deg, 8000)
    said = re.search(rf"they score {_NUMBER}(?: dB)? (?:each|to {_NUMBER})",
                     str(caught[0].message))
    return float(said.group(1)), float(said.group(2) or said.group(1))


@pytest.mark.parametrize("measure, match, worst", [
    (llr, 0.0, 2.0), (cepstral_distance, 0.0, 10.0), (fw_segmental_snr, 35.0, -10.0)])
def test_what_a_frame_against_eps_scores_depends_on_what_deg_holds_there(
        measure, match, worst):
    """eps is a constant, and so is a DC offset: the three measures blind to
    level score such a frame as a match. Noise scores the worst there, and a
    50 Hz hum lands in between."""
    assert _against_eps(measure, 'dc') == pytest.approx((match, match), abs=1e-3)
    assert _against_eps(measure, 'noise') == (worst, worst)
    low, high = sorted((match, worst))
    hum = _against_eps(measure, 'hum')
    assert low < min(hum) and max(hum) < high, hum


@pytest.mark.parametrize("measure", MEASURES)
@pytest.mark.parametrize("gain", [1.0, 2.0])
def test_frames_silent_in_both_signals_are_a_match_and_go_unmentioned(measure, gain):
    """pysepm and a port of the MATLAB code both score an exact copy with
    digitally silent pauses as perfect, as these do: the two frames are the
    same eps under the window, as alike as any pause copied exactly."""
    padded = _padded(8000)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        measure(padded, gain * padded, 8000)


def test_only_the_frames_where_deg_is_not_silent_are_counted():
    """The speech 0.3 s late: of the 125 frames where the reference is zero,
    85 are zero in deg too and 40 hold the late speech."""
    padded = _padded(8000)
    with pytest.warns(UserWarning, match=r"^llr\(\): 40 of the reference's 396 frames"):
        llr(padded, np.roll(padded, 2400), 8000)


def test_finding_the_silent_frames_takes_a_byte_a_sample():
    """Which frames are digital silence is read off a boolean view of the
    signal, one byte a sample and no copy of the frames, and it runs on both
    signals of every pair a measure scores."""
    sr = 16000
    y = _noise(60 * sr, 2)
    y[sr:3 * sr] = 0.0
    y[::7] = 0.0                    # scattered zeros leave every frame audible
    size, hop = quality._frame_size(sr)
    count = (len(y) - size) // hop
    tracemalloc.start()
    try:
        silent = quality._silence(y, size, hop, count)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 2 * len(y), f"{peak / len(y):.1f} bytes a sample"
    starts = np.arange(count) * hop
    assert np.array_equal(silent, [not y[s:s + size].any() for s in starts])
    assert np.count_nonzero(silent) == 263


def test_composite_names_the_frames_its_segsnr_puts_at_the_floor():
    """Even against an exact copy, which the other parts score as a match."""
    padded = _padded(16000)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        composite(padded, padded, 16000, pesq=4.5)
    assert [str(w.message) for w in caught] == [
        "composite(): 125 of the reference's 396 frames are digital silence, and "
        "segSNR scores them -10 dB each whatever deg holds there. Trim the silence "
        "off both to compare the rest alone."]


def test_composite_names_where_llr_and_wss_read_eps_and_what_that_costs():
    """LLR goes into CSIG and COVL uncapped, so 125 frames at 19 to 22 take
    both to the floor of the scale, from 4.30 and 3.20 with PESQ at 2.0."""
    padded, copy = _filled(8000)
    with pytest.warns(UserWarning, match="segSNR scores them -10 dB each"):
        assert composite(padded, padded, 8000, pesq=2.0)[::2] == pytest.approx(
            (4.299, 3.204, 0.0, 20.795), abs=1e-3)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got = composite(padded, copy, 8000, pesq=2.0)
    assert [str(w.message) for w in caught] == [
        "composite(): 125 of the reference's 396 frames are digital silence, and "
        "segSNR scores them -10 dB each whatever deg holds there; in 125 of them deg "
        "is not silent, and there LLR, taken uncapped, scores 19.4 to 22.4 and WSS "
        "7.2 to 40.1, comparing deg with eps rather than with speech. Trim the "
        "silence off both to compare the rest alone."]
    assert (got.csig, got.covl) == (1.0, 1.0)


def test_a_silent_output_is_scored_rather_than_refused():
    """An enhancer that returns nothing has given an answer, and a bad one:
    the cepstral distance and IS sit at their caps and LLR at 1.91 of 2.
    fwSegSNR is at -0.87 dB, only just below 0, because a band that is
    missing outright scores 0 dB there; it takes energy added where the clean
    band has little to drive a band towards -10."""
    clean = _speech(8000)
    nothing = np.zeros_like(clean)
    assert cepstral_distance(clean, nothing, 8000) == 10.0
    assert itakura_saito(clean, nothing, 8000) == 100.0
    assert llr(clean, nothing, 8000) > 1.9
    assert -1.0 < fw_segmental_snr(clean, nothing, 8000) < 0.0


def test_a_length_mismatch_warns_at_the_callers_line():
    clean, noisy = _pair(8000)
    with pytest.warns(UserWarning, match="only the first") as caught:
        value = llr(clean, noisy[:-100], 8000)
    assert np.isfinite(value)
    assert caught[0].filename == __file__


def test_integer_audio_is_read_on_its_own_full_scale():
    """An int16 reference against its float copy: read as plain numbers it
    would be 32768 times louder, and Itakura-Saito would see that."""
    clean = _speech(8000)
    as_int = np.round(clean * 32767).astype(np.int16)
    back = as_int / 32768.0
    assert itakura_saito(as_int, back, 8000) == pytest.approx(0.0, abs=1e-9)



def _at_peak(y, dbfs, peak):
    return y * (10.0 ** (dbfs / 20.0) / peak)


@pytest.mark.parametrize("sr, normal, quiet", [(8000, 35.979, 27.952),
                                               (16000, 30.899, 25.195)])
def test_wss_says_when_its_band_floor_makes_it_depend_on_the_level(sr, normal, quiet):
    """comp_wss.m floors band energies at -100 dB of full scale, not of the
    frame, so a quiet pair scores otherwise than the same pair at a normal
    level. On this pair WSS is the same from -6 to -46 dBFS and moves from
    -56 on, and the warning comes exactly where the floor starts to bite."""
    clean, noisy = _pair(sr)
    peak = float(np.max(np.abs(clean)))
    for dbfs in (-6.0, -26.0, -46.0):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            value = wss(_at_peak(clean, dbfs, peak), _at_peak(noisy, dbfs, peak), sr)
        assert value == pytest.approx(normal, abs=5e-4)
    for measure in (wss, lambda r, d, sr: composite(r, d, sr, pesq=2.0).wss):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            value = measure(_at_peak(clean, -66.0, peak), _at_peak(noisy, -66.0, peak), sr)
        said = [str(w.message) for w in caught]
        assert value == pytest.approx(quiet, abs=5e-4)
        assert len(said) == 1 and "-100 dB floor" in said[0], said
        assert "ref peaks at -66.0 dBFS" in said[0], said
        assert caught[0].filename == __file__


def test_a_pair_under_one_16_bit_step_is_called_faint_once_not_floored_too():
    clean, noisy = _pair(8000)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        wss(clean * 1e-5, noisy * 1e-5, 8000)
    said = [str(w.message) for w in caught]
    assert len(said) == 1 and "16-bit" in said[0], said
