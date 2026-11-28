# -*- coding: utf-8 -*-
"""SIIB and SIIB^Gauss: what a copy, a gain, an unrelated output and noise at
known SNRs score, the estimator against its definition and against Gaussian
pairs whose information is known, the representation against values that
follow from its definition, and the input either one refuses or cannot
score."""
from __future__ import annotations

import importlib
import math
import re
import warnings

import numpy as np
import pytest
from scipy.signal import butter, firwin, lfilter, resample_poly, sosfilt
from scipy.special import digamma

from conftest import make_speech
from kudio.core import evaluator
from kudio.core.siib import siib, siib_gauss

# the module, whatever kudio's flat API binds to the name
module = importlib.import_module('kudio.core.siib')

#: the most either measure gives: 80 / 15 stacks a second, times 420
#: eigenchannels, times -1/2 log2(1 - 0.75^2) bits
CEILING = 2240 * -0.5 * math.log2(1 - 0.75 ** 2)

#: rough F1, F2, F3 of five vowels, in Hz
_VOWELS = ((730, 1090, 2440), (270, 2290, 3010), (300, 870, 2240),
           (530, 1840, 2480), (570, 840, 2410))

SR = 16000


def _resonator(x: np.ndarray, f: float, bandwidth: float) -> np.ndarray:
    r = math.exp(-math.pi * bandwidth / SR)
    return lfilter([1 - r], [1, -2 * r * math.cos(2 * math.pi * f / SR), r * r], x)


def _speech(seconds: float = 36.0, seed: int = 1, floor: float = 1e-3,
            syllables: bool = False):
    """Syllables at 16 kHz: a harmonic source gliding in pitch, through three
    formants moving from one vowel to another, two in five of them led by a
    burst of hiss from 3 to 7 kHz, each syllable at its own level within
    8 dB. Pauses of 50 to 250 ms lie between them on white noise of RMS
    *floor*, 54 dB under the peak by default. With *syllables* it returns as
    well where the syllables were written, a boolean array: their windows end
    in samples that are 0 or a rounding of it, depending on the numpy build,
    so the samples cannot say where a pause is."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    y = np.zeros(n)
    spoken = np.zeros(n, dtype=bool)
    hiss = butter(4, [3000, 7000], 'bandpass', fs=SR, output='sos')
    start = int(0.1 * SR)
    while start < n - int(0.45 * SR):
        length = int(rng.uniform(0.15, 0.32) * SR)
        f0 = np.linspace(*rng.uniform(90.0, 230.0, 2), length)
        phase = 2 * np.pi * np.cumsum(f0) / SR
        voice = sum(np.sin(h * phase) / h for h in range(1, int(0.45 * SR / f0.max())))
        first, last = _VOWELS[rng.integers(5)], _VOWELS[rng.integers(5)]
        syllable = np.zeros(length)
        for j, part in enumerate(np.array_split(np.arange(length), 4)):
            piece = voice[part]
            for f_from, f_to, bandwidth in zip(first, last, (80.0, 100.0, 140.0)):
                piece = _resonator(piece, f_from + (f_to - f_from) * j / 3, bandwidth)
            syllable[part] = piece
        syllable *= np.hanning(length) / np.max(np.abs(syllable))
        if rng.random() < 0.4:
            m = int(rng.uniform(0.05, 0.12) * SR)
            burst = sosfilt(hiss, rng.standard_normal(m)) * np.hanning(m)
            syllable = np.concatenate([0.3 * burst / np.max(np.abs(burst)), syllable])
        syllable *= 10 ** (rng.uniform(-8.0, 0.0) / 20)
        stop = min(start + len(syllable), n)
        y[start:stop] = syllable[:stop - start]
        spoken[start:stop] = True
        start = stop + int(rng.uniform(0.05, 0.25) * SR)
    y *= 0.5 / np.max(np.abs(y))
    if floor:
        y = y + floor * rng.standard_normal(n)
    return (y, spoken) if syllables else y


def _white(n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(n)


def _mix(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    scale = np.sqrt(np.mean(clean ** 2) / (np.mean(noise ** 2) * 10 ** (snr_db / 10)))
    return clean + scale * noise


def _quiet(fn, *args):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return fn(*args)


SNRS = (-10, -5, 0, 5, 10, 15, 20)


@pytest.fixture(scope="module")
def talk() -> np.ndarray:
    return _speech()


@pytest.fixture(scope="module")
def sweep(talk):
    """``{snr: (deg, siib, siib_gauss)}`` for the test speech in white noise."""
    noise = _white(len(talk), 7)
    out = {}
    for snr in SNRS:
        deg = _mix(talk, noise, snr)
        out[snr] = (deg, siib(talk, deg, SR), siib_gauss(talk, deg, SR))
    return out


# -- the test speech ---------------------------------------------------------------

def test_the_test_speech_holds_20_s_of_speech(talk):
    """The authors advise 20 s of speech or more. The voice-activity
    threshold keeps 1645 of the 2878 frames of the 36 s, 20.6 s."""
    x = talk / np.std(talk, ddof=1)
    levels = module._band_levels(x, x)[0]
    assert len(levels) == 2878
    assert int(np.count_nonzero(module._active(levels))) == 1645


# -- copies, gains and unrelated output ----------------------------------------------

def test_a_copy_and_an_inverted_copy_score_the_ceiling(talk):
    """Every eigenchannel of a copy carries far more than the 0.596 bits the
    production noise lets through, so all 420 are held there."""
    for deg in (talk, -talk):
        assert _quiet(siib, talk, deg, SR) == pytest.approx(CEILING, rel=1e-15)
        assert _quiet(siib_gauss, talk, deg, SR) == pytest.approx(CEILING, rel=1e-15)
    assert siib(talk, talk, SR) == siib_gauss(talk, talk, SR)


def test_a_gain_on_both_signals_changes_nothing(talk, sweep):
    """Both are divided by the reference's deviation first. A power of two
    scales exactly, so the score stays the same to the bit; another gain
    moves SIIB^Gauss by rounding, and SIIB here not at all."""
    deg, value, gauss = sweep[0]
    assert siib(0.25 * talk, 0.25 * deg, SR) == value
    assert siib_gauss(0.25 * talk, 0.25 * deg, SR) == gauss
    assert siib(0.3 * talk, 0.3 * deg, SR) == pytest.approx(value, rel=1e-6)
    for gain in (0.3, 7.0, 1e-4):
        assert siib_gauss(gain * talk, gain * deg, SR) == pytest.approx(gauss, rel=1e-9)


def test_a_gain_on_the_output_alone_is_heard(talk, sweep):
    """The masking floor of each band comes from the reference and holds the
    output too, so a copy at another level is no longer a copy: twice as
    loud, SIIB^Gauss falls 24% and SIIB 0.2%; half as loud, 34% and 5%. On
    the test speech in white noise at 0 dB, a gain of 2 or 1/2 on the output
    moves either by under 2%."""
    assert siib_gauss(talk, 2.0 * talk, SR) == pytest.approx(1018.23, abs=0.01)
    assert siib_gauss(talk, 0.5 * talk, SR) == pytest.approx(880.32, abs=0.01)
    assert siib(talk, 2.0 * talk, SR) == pytest.approx(1333.41, abs=0.01)
    assert siib(talk, 0.5 * talk, SR) == pytest.approx(1267.31, abs=0.01)
    deg, value, gauss = sweep[0]
    quieter = siib(talk, 0.5 * deg, SR)
    assert quieter == pytest.approx(126.09, abs=0.01)
    assert quieter == pytest.approx(value, rel=0.02)
    for gain in (2.0, 0.5):
        assert siib_gauss(talk, gain * deg, SR) == pytest.approx(gauss, rel=0.005)


def test_an_unrelated_output_scores_near_0(talk):
    """Not 0: the estimator and the correlations find a little in 420
    channels of noise. White noise as the output scores SIIB 4.7 and
    SIIB^Gauss 1.2 bit/s against the test speech, another talker 7.4 and
    1.1, against a ceiling of 1336."""
    noise = _white(len(talk), 11)
    other = _speech(seed=2)
    assert siib(talk, noise, SR) == pytest.approx(4.71, abs=0.01)
    assert siib_gauss(talk, noise, SR) == pytest.approx(1.24, abs=0.01)
    assert siib(talk, other, SR) == pytest.approx(7.44, abs=0.01)
    assert siib_gauss(talk, other, SR) == pytest.approx(1.12, abs=0.01)


def test_the_score_rises_with_the_snr(sweep):
    """White noise from -10 to +20 dB SNR, and the values at -5, 0 and 10 dB
    the module notes quote."""
    values = [sweep[snr][1] for snr in SNRS]
    gauss = [sweep[snr][2] for snr in SNRS]
    assert all(a < b for a, b in zip(values, values[1:])), values
    assert all(a < b for a, b in zip(gauss, gauss[1:])), gauss
    assert [sweep[snr][1] for snr in (-5, 0, 10)] == pytest.approx([79.18, 124.08, 293.41],
                                                                  abs=0.01)
    assert [sweep[snr][2] for snr in (-5, 0, 10)] == pytest.approx([40.26, 63.95, 154.96],
                                                                  abs=0.01)


def test_a_longer_pair_scores_more(talk, sweep):
    """Over its first 12 s the 0 dB pair scores SIIB 102.2 and SIIB^Gauss
    51.8, where all 36 s score 124.1 and 64.0; another talker scores 14.6
    and 2.3 over 12 s, 7.4 and 1.1 over 36 s."""
    n = 12 * SR
    deg = sweep[0][0]
    other = _speech(seed=2)
    assert siib(talk[:n], deg[:n], SR) == pytest.approx(102.24, abs=0.01)
    assert siib_gauss(talk[:n], deg[:n], SR) == pytest.approx(51.84, abs=0.01)
    assert siib(talk[:n], other[:n], SR) == pytest.approx(14.56, abs=0.01)
    assert siib_gauss(talk[:n], other[:n], SR) == pytest.approx(2.30, abs=0.01)


def test_neither_score_passes_the_ceiling(talk, sweep):
    """SIIB holds each channel to the ceiling's share, and SIIB^Gauss cannot
    pass it, the squared correlation being held to 1; both sum their
    channels the same way, so a copy reaches the ceiling by either and
    nothing passes it."""
    top = siib_gauss(talk, talk, SR)
    assert top == pytest.approx(CEILING, rel=1e-15)
    for _, value, gauss in sweep.values():
        assert 0.0 <= value <= top and 0.0 <= gauss <= top
    near = talk + 1e-9 * _white(len(talk), 3)
    assert siib_gauss(talk, near, SR) <= top
    assert module._rate_of(np.minimum(module._I_MAX, np.full(420, 1.0))) == top


def test_a_delay_costs_a_lot(talk):
    """The pair is compared frame against frame: a copy 10 ms late scores
    between the test speech in white noise at 10 dB and at 15 dB."""
    late = np.concatenate([np.zeros(160), talk[:-160]])
    assert siib(talk, late, SR) == pytest.approx(382.07, abs=0.01)
    assert siib_gauss(talk, late, SR) == pytest.approx(221.19, abs=0.01)


@pytest.mark.parametrize("window", ["as numpy gives it", "ending a rounding off 0"])
def test_noise_in_the_pauses_of_the_reference_costs_siib_nothing(talk, window, monkeypatch):
    """The frames are chosen on the reference alone, and what the output holds
    in the others is never looked at. White noise 7.5 dB louder than the
    speech, added to a copy in the pauses alone, the 213,900 samples outside
    every syllable, leaves SIIB at the ceiling and takes SIIB^Gauss to 845,
    through the frames at the edges of the syllables. The last sample of a
    Hann window is 0.5 - 0.5 cos(2 pi), 0 or 1.1e-16 by the numpy build and
    the CPU, and neither the pauses nor the score may follow it."""
    if window == "as numpy gives it":
        pauses = ~_speech(syllables=True)[1]
    else:
        hanning = np.hanning

        def nudged(m):
            w = hanning(m)
            w[-1] = 0.5 - 0.5 * np.nextafter(1.0, 0.0)
            return w

        with monkeypatch.context() as patch:
            patch.setattr(np, "hanning", nudged)
            talk, spoken = _speech(syllables=True)
        pauses = ~spoken
    assert 20 * math.log10(0.1 / np.sqrt(np.mean(talk ** 2))) == pytest.approx(7.55,
                                                                                abs=0.01)
    deg = talk + 0.1 * _white(len(talk), 9) * pauses
    assert siib(talk, deg, SR) == pytest.approx(CEILING, rel=1e-15)
    assert int(pauses.sum()) == 213900
    assert siib_gauss(talk, deg, SR) == pytest.approx(845.07, abs=0.01)


def test_an_output_of_digital_silence_scores_0_or_a_little(talk):
    """Its masked spectrum climbs to the reference's floor over the first 15
    frames and stays there, which leaves its channels nearly, but not quite,
    constant. Over the first 18 s, SIIB's estimates for those channels sum to
    less than 0 and the score is held at 0; SIIB^Gauss cannot go under 0 in
    any channel. SIIB over the whole 36 s rests on channels that are nearly
    constant, so its last digits follow the numpy build: 0.47 with numpy
    2.0 and 2.5, 0.48 with 2.2. What holds everywhere is a little over 0."""
    silence = np.zeros_like(talk)
    assert 0.0 < _quiet(siib, talk, silence, SR) < 1.0
    assert _quiet(siib_gauss, talk, silence, SR) == pytest.approx(0.55, abs=0.01)
    n = 18 * SR
    assert _quiet(siib, talk[:n], silence[:n], SR) == 0.0
    assert _quiet(siib_gauss, talk[:n], silence[:n], SR) == pytest.approx(1.16, abs=0.01)
    channels = module._eigenchannels(talk[:n], silence[:n], SR, 'siib', floor_deviation=True)
    assert np.sum(np.minimum(module._I_MAX, module._channel_bits(*channels))) < 0.0


def test_an_offset_is_not_taken_out(talk, sweep):
    """An offset 18 dB under the speech's RMS lifts every pause of the
    reference over the 40 dB threshold, and the 0 dB pair loses 41% of its
    SIIB and 42% of its SIIB^Gauss. Ten times that on the output alone moves
    either by 0.1% to 0.2%."""
    deg = sweep[0][0]
    rms = np.sqrt(np.mean(talk ** 2))
    assert 20 * math.log10(0.005 / rms) == pytest.approx(-18.47, abs=0.01)
    x = (talk + 0.005) / np.std(talk + 0.005, ddof=1)
    assert int(np.count_nonzero(module._active(module._band_levels(x, x)[0]))) == 2878
    assert siib(talk + 0.005, deg + 0.005, SR) == pytest.approx(72.75, abs=0.01)
    assert siib_gauss(talk + 0.005, deg + 0.005, SR) == pytest.approx(36.83, abs=0.01)
    assert siib(talk, deg + 0.05, SR) == pytest.approx(123.91, abs=0.01)
    assert siib_gauss(talk, deg + 0.05, SR) == pytest.approx(63.80, abs=0.01)


def test_two_calls_agree_to_the_bit(talk, sweep):
    """The noise that breaks ties comes from a generator with a fixed seed of
    its own, so neither numpy's global state nor an earlier call moves the
    score, and scoring leaves the global state alone."""
    deg, value, gauss = sweep[0]
    np.random.seed(12345)
    state = np.random.get_state()
    assert siib(talk, deg, SR) == value
    assert siib_gauss(talk, deg, SR) == gauss
    after = np.random.get_state()
    assert np.array_equal(state[1], after[1]) and state[2:] == after[2:]


def test_the_noise_that_breaks_ties_moves_siib_by_parts_in_1e6(talk, sweep, monkeypatch):
    """Another seed, no noise at all, or the samples passed on in full move
    the 0 dB pair's score, but by a few parts in 1e6 (measured once: 5e-7,
    3.9e-6 and 1.8e-6)."""
    deg, value, _ = sweep[0]
    with monkeypatch.context() as m:
        m.setattr(module, '_SEED', 1)
        other = siib(talk, deg, SR)
    with monkeypatch.context() as m:
        m.setattr(module, '_DITHER_NORMAL', 0.0)
        m.setattr(module, '_DITHER_UNIFORM', 0.0)
        still = siib(talk, deg, SR)
    with monkeypatch.context() as m:
        m.setattr(module, '_as_text', lambda a: np.array(a, dtype=float))
        full = siib(talk, deg, SR)
    for score in (other, still, full):
        assert score != value
        assert score == pytest.approx(value, rel=1e-5)


def test_siib_gauss_of_the_0_db_pair_to_nine_digits(sweep):
    """SIIB^Gauss has no random numbers in it and is a smooth function of
    the features, so it can be held far tighter than SIIB. A band mean off
    by 0.01, the last frame left as the curves have it, or the correlation
    taken about the channels' means each move it by 7e-6 to 1.4e-5 of
    itself."""
    assert sweep[0][2] == pytest.approx(63.95020685493268, rel=1e-9)


#: (SIIB, SIIB^Gauss) pySIIB gives for the test speech in white noise, from
#: its SIIB() with window='hann' (commit 226c2f3, numpy 2.2.6, scipy 1.15.2).
#: pySIIB is GPL-3.0; only its numbers are here.
PYSIIB = {-10: (51.18501799578119, 24.461682713983866),
          0: (124.08321347526369, 63.95020685493104),
          10: (293.4140394190359, 154.956951349507),
          20: (575.254599528147, 333.9533786345062)}


@pytest.mark.parametrize("snr", sorted(PYSIIB))
def test_the_scores_pysiib_gives_for_the_test_speech(sweep, snr):
    """SIIB^Gauss to 1e-12 of itself; SIIB to 1e-4, since the noise kudio
    adds to break ties moves it by parts in 1e5."""
    _, value, gauss = sweep[snr]
    siib_said, gauss_said = PYSIIB[snr]
    assert gauss == pytest.approx(gauss_said, rel=1e-12)
    assert value == pytest.approx(siib_said, rel=1e-4)


# -- rates -------------------------------------------------------------------------

@pytest.mark.parametrize("sr, up, down, value, gauss", [
    (8000, 1, 2, 122.39, 63.85),
    (44100, 441, 160, 124.90, 63.94),
])
def test_other_rates_are_taken_to_16_khz(talk, sweep, sr, up, down, value, gauss):
    """The 0 dB pair scores 124.1 and 64.0 at 16 kHz. Taken to 8 kHz, its
    bands over 4 kHz hold only what the resampler lets through; taken to
    44.1 kHz, it comes back through two resamplers."""
    ref, deg = resample_poly(talk, up, down), resample_poly(sweep[0][0], up, down)
    assert siib(ref, deg, sr) == pytest.approx(value, abs=0.01)
    assert siib_gauss(ref, deg, sr) == pytest.approx(gauss, abs=0.01)
    assert siib_gauss(ref, ref, sr) == pytest.approx(CEILING, rel=1e-15)


def test_scores_at_other_rates_depend_on_the_resampler(talk):
    """The pair goes to 16 kHz through resample_poly's defaults, the filter
    MATLAB's resample designs, 101 taps from 20 kHz. Through a sharper one
    of 365 taps, the test speech at 20 kHz in white noise at 10 dB scores
    SIIB 2.8 bit/s more and SIIB^Gauss 0.02 less."""
    longer = firwin(601, 1 / 5, window=('kaiser', 10.0))
    ref = resample_poly(talk, 5, 4, window=longer)
    deg = _mix(ref, _white(len(ref), 7), 10)
    plain = (resample_poly(ref, 4, 5), resample_poly(deg, 4, 5))
    assert siib_gauss(ref, deg, 20000) == pytest.approx(siib_gauss(*plain, SR), rel=1e-10)
    sharper = firwin(365, 1 / 5, window=('kaiser', 5.65))
    other = (resample_poly(ref, 4, 5, window=sharper), resample_poly(deg, 4, 5, window=sharper))
    assert siib(ref, deg, 20000) == pytest.approx(321.26, abs=0.01)
    assert siib(*other, SR) == pytest.approx(324.03, abs=0.01)
    assert siib_gauss(ref, deg, 20000) == pytest.approx(169.40, abs=0.01)
    assert siib_gauss(*other, SR) == pytest.approx(169.37, abs=0.01)


def test_a_rate_given_as_a_float_or_a_numpy_integer_is_a_rate():
    ref = _white(module._min_samples(8000), 1)
    deg = ref + _white(len(ref), 2)
    want = siib_gauss(ref, deg, 8000)
    assert siib_gauss(ref, deg, 8000.0) == siib_gauss(ref, deg, np.int64(8000)) == want
    assert siib_gauss(ref, deg, np.array(8000)) == want


def test_the_ratio_to_16_khz_is_in_lowest_terms():
    assert module._ratio(16000) == (1, 1)
    assert module._ratio(8000) == (2, 1)
    assert module._ratio(44100) == (160, 441)
    assert module._ratio(48000) == (1, 3)


# -- the estimator ------------------------------------------------------------------

def _ksg_written_out(u: np.ndarray, v: np.ndarray, k: int) -> float:
    """The second estimator of Kraskov, Stoegbauer and Grassberger from its
    definition, every distance taken."""
    n = len(u)
    du = np.abs(u[None, :] - u[:, None])
    dv = np.abs(v[None, :] - v[:, None])
    joint = np.maximum(du, dv)
    np.fill_diagonal(joint, np.inf)
    nearest = np.argsort(joint, axis=1, kind='stable')[:, :k]
    eu = np.take_along_axis(du, nearest, axis=1).max(axis=1)
    ev = np.take_along_axis(dv, nearest, axis=1).max(axis=1)
    np.fill_diagonal(du, np.inf)
    np.fill_diagonal(dv, np.inf)
    nu = np.sum(du <= eu[:, None], axis=1)
    nv = np.sum(dv <= ev[:, None], axis=1)
    return float(digamma(n) + digamma(k) - 1 / k - np.mean(digamma(nu) + digamma(nv)))


@pytest.mark.parametrize("n, k, r", [(300, 2, 0.7), (500, 4, 0.0), (1000, 7, 0.95)])
def test_the_estimator_is_its_definition_written_out(n, k, r):
    rng = np.random.default_rng(n)
    a, b = rng.standard_normal((2, n))
    u, v = a, r * a + math.sqrt(1 - r * r) * b
    assert module._ksg(u, v, k) == pytest.approx(_ksg_written_out(u, v, k), abs=1e-12)


@pytest.mark.parametrize("n, k, bits", [(1583, 11, 7.103974), (300, 2, 6.895116),
                                        (5, 2, 0.841572)])
def test_a_sequence_against_itself(n, k, bits):
    """Each point's k nearest neighbours in the plane are its k nearest along
    either axis, so both counts are k and the estimate is psi(N) - psi(k) -
    1/k. With 1583 points that is 7.1 bits, 12 times what a channel may keep;
    with 5 it is 0.84, still over the 0.596 a channel is held to."""
    u = np.random.default_rng(n).standard_normal(n)
    estimate = module._ksg(u, u, k)
    assert estimate == pytest.approx(digamma(n) - digamma(k) - 1 / k, abs=1e-12)
    assert estimate / math.log(2) == pytest.approx(bits, abs=1e-6)


@pytest.mark.parametrize("r", [0.0, 0.3, 0.6, 0.9])
def test_the_estimator_finds_the_information_of_a_gaussian_pair(r):
    """-1/2 ln(1 - r^2) nats for unit Gaussians with correlation r (Kraskov
    et al., Eq. 11). The mean of four draws of 20,000 points with k = 4 is
    held to 0.018 nats of it. Over 15 such means at each r, measured once,
    they scattered by 0.0025 to 0.0036 nats (one standard deviation, the
    1/sqrt(N) their Sec. III.B finds for the statistical error) and ran high
    by at most 0.0034, at r = 0.9, the systematic error of a finite N;
    0.018 is that bias and four deviations more."""
    exact = -0.5 * math.log(1 - r * r)
    draws = []
    for seed in range(4):
        rng = np.random.default_rng(100 + seed)
        a, b = rng.standard_normal((2, 20000))
        draws.append(module._ksg(a, r * a + math.sqrt(1 - r * r) * b, 4))
    assert np.mean(draws) == pytest.approx(exact, abs=0.018)


def test_the_neighbour_count_is_a_150th_of_the_stacks():
    assert [module._neighbours(n) for n in (2, 285, 300, 301, 985, 1583, 4785)] == \
        [2, 2, 2, 3, 7, 11, 32]


def test_the_samples_reach_the_estimator_with_8_significant_digits():
    """As SIIB.m's ASCII save writes them for MIxnyn to read back; "%.7e"
    rounds the same way."""
    rng = np.random.default_rng(0)
    a = rng.standard_normal(10000) * 10.0 ** rng.integers(-6, 4, 10000)
    a[:4] = (0.0, 9.99999996, -9.99999994, 1e-3)
    want = np.array([float(f"{value:.7e}") for value in a])
    assert np.array_equal(module._as_text(a), want)


def test_the_marginal_counts_take_the_distance_as_the_search_takes_it():
    """0.26 + |2.61 - 0.26| rounds to 2.6099999999999994, and 1.13 - |-0.67 -
    1.13| to -0.6699999999999999, so a search on u +/- e alone would leave
    out the point that sets e. The counts take it in."""
    u = np.array([0.26, 2.61, 1.13, -0.67])
    radius = np.array([abs(2.61 - 0.26), 0.0, abs(-0.67 - 1.13), 0.0])
    assert 0.26 + radius[0] < 2.61 and 1.13 - radius[2] > -0.67
    assert list(module._within(u, radius)) == [3, 0, 3, 0]


# -- the representation, against values that follow from its definition ------------

def test_28_bands_from_100_to_6500_hz():
    centres, widths, weights = module._filterbank()
    assert weights.shape == (28, 201)
    assert centres[[0, 1, 13, 27]] == pytest.approx((100.0, 138.8975, 1177.7988, 6500.0),
                                                    abs=1e-4)
    assert widths[[0, 27]] == pytest.approx((36.1538, 739.8067), abs=1e-4)
    # 3.2 / pi of the ERB, the bandwidth of a 4th-order gammatone one ERB wide
    assert widths / (24.7 * (1 + 4.37 * centres / 1000)) == pytest.approx(3.2 / math.pi)


def test_a_bands_weights_are_its_squared_response_cut_at_a_thousandth():
    """Band 0 is centred on 100 Hz, halfway between the bins at 80 and
    120 Hz, and stays over 0.001 of its peak up to 320 Hz; band 27, centred
    on 6500 Hz, runs from 2440 Hz to the top bin."""
    weights = module._filterbank()[2]
    amplitude = np.sqrt(weights)
    assert amplitude[0, :9] == pytest.approx(
        (0.022794, 0.121023, 1, 1, 0.121023, 0.022794, 0.006667, 0.002565, 0.001179),
        abs=1e-6)
    assert not amplitude[0, 9:].any()
    top = np.flatnonzero(amplitude[27])
    assert (top[0], top[-1]) == (61, 200)
    assert amplitude[27, [61, 161, 162, 163, 200]] == pytest.approx(
        (0.001034, 0.988417, 1, 1, 0.038338), abs=1e-6)
    sums = weights.sum(axis=1)
    assert sums.argmin() == 3 and sums.argmax() == 27
    assert (sums.min(), sums.max()) == pytest.approx((1.415170, 18.201885), abs=1e-6)


def test_the_masking_curve_falls_to_the_floor_in_15_frames():
    weights = module._masking_weights()
    assert weights[[0, 1, 3, 7, 15]] == pytest.approx((0, 0.25, 0.5, 0.75, 1), abs=1e-15)
    floor = np.array([-5.0])
    levels = np.full((40, 1), -5.0)
    levels[10] = 3.0
    masked = module._masked(levels, floor)
    assert masked[10:26, 0] == pytest.approx(3.0 - weights * 8.0)
    assert masked[24, 0] > -5.0 and masked[25, 0] == -5.0
    assert np.all(masked[26:, 0] == -5.0) and np.all(masked[:10, 0] == -5.0)


def test_the_last_frame_is_left_at_the_floor():
    """However loud it is, as SIIB.m leaves it."""
    levels = np.linspace(0.0, 4.0, 30)[:, None]
    masked = module._masked(levels, np.array([-1.0]))
    assert masked[-1, 0] == -1.0
    assert masked[-2, 0] == levels[-2, 0]


def test_the_frames_stop_short_of_the_last_sample():
    assert [module._frame_count(n) for n in (400, 401, 600, 601, 160000, 320000)] == \
        [0, 1, 1, 2, 798, 1598]


def test_the_threshold_is_read_at_0_999_of_the_frames_rounded_half_up():
    """Up to 500 frames the level is the loudest one's, and it moves one
    place down each time the count passes 500, 1500, 2500 and so on: 1600
    frames read the 1598th from the quietest, the third loudest. 0.999 *
    1500 is 1498.5 in floating point, which rounds to 1499. A frame exactly
    40 dB under that level is dropped."""
    counts = (499, 500, 501, 1000, 1500, 1501, 1600, 2500, 2501)
    assert [module._vad_rank(t) for t in counts] == \
        [499, 500, 500, 999, 1499, 1499, 1598, 2498, 2498]
    levels = np.arange(1600.0)
    active = module._active(levels)
    assert np.count_nonzero(active) == 42 and not active[1557] and active[1558]


def test_the_stacks_leave_out_the_one_ending_on_the_last_frame():
    features = np.arange(20 * 28, dtype=float).reshape(20, 28)
    stacked = module._stacked(features)
    assert stacked.shape == (5, 420)
    assert np.array_equal(stacked[0], features[:15].reshape(-1))
    assert np.array_equal(stacked[-1], features[4:19].reshape(-1))


def test_siib_gauss_of_a_fixed_correlation():
    """With every squared correlation r^2, -1120 log2(1 - 0.5625 r^2) bit/s:
    244.877 at r^2 = 0.25 and 533.611 at 0.5."""
    for r2, want in ((0.25, 244.877), (0.5, 533.611)):
        channels = -0.5 * np.log2(1 - 0.5625 * np.full(420, r2))
        assert module._rate_of(channels) == pytest.approx(want, abs=1e-3)


# -- what is refused, and what scores nan -------------------------------------------

@pytest.mark.parametrize("measure", [siib, siib_gauss])
def test_a_pair_shorter_than_436_frames_scores_nan(measure):
    """White noise keeps every frame, so 87,401 samples at 16 kHz are 436
    frames, 421 stacks and a covariance of full rank; one sample fewer is
    scored nan."""
    need = module._min_samples(SR)
    assert need == 87401 and module._frame_count(need) == 436
    ref = _white(need, 1)
    deg = ref + _white(need, 2)
    assert math.isfinite(_quiet(measure, ref, deg, SR))
    with pytest.warns(UserWarning, match=rf"^{measure.__name__}\(\): the pair is 87400 "
                                         r"samples \(5\.46 s at 16000 Hz\), and SIIB "
                                         r"needs at least 87401.*; scored nan$") as caught:
        assert math.isnan(measure(ref[:-1], deg[:-1], SR))
    assert len(caught) == 1 and caught[0].filename == __file__


@pytest.mark.parametrize("sr, need", [(8000, 43701), (44100, 240897), (48000, 262201)])
def test_the_shortest_pair_at_other_rates(sr, need):
    assert module._min_samples(sr) == need
    up, down = module._ratio(sr)
    assert module._frame_count(math.ceil(need * up / down)) == 436
    assert module._frame_count(math.ceil((need - 1) * up / down)) == 435


@pytest.mark.parametrize("measure", [siib, siib_gauss])
def test_too_little_speech_once_the_silence_is_dropped_scores_nan(measure):
    """The shared test signal with everything under 0.05 set to 0, which
    leaves its bursts and digital silence between them. Two clips keep 306
    frames, 3.8 s; six keep 918, 11.5 s."""
    clip = make_speech().astype(float)
    clip[np.abs(clip) < 0.05] = 0.0
    short = np.tile(clip, 2)
    deg = short + 0.01 * _white(len(short), 4)
    with pytest.warns(UserWarning, match=rf"^{measure.__name__}\(\): 306 of the pair's "
                                         r"958 frames hold speech by the 40 dB threshold "
                                         r"on ref \(3\.83 s\).*; scored nan$") as caught:
        assert math.isnan(measure(short, deg, SR))
    assert len(caught) == 1 and caught[0].filename == __file__
    long = np.tile(clip, 6)
    assert math.isfinite(_quiet(measure, long, long + 0.01 * _white(len(long), 4), SR))


@pytest.mark.parametrize("measure", [siib, siib_gauss])
def test_a_reference_of_digital_silence_or_one_value_scores_nan(measure):
    deg = _white(90000, 1)
    with pytest.warns(UserWarning, match=rf"^{measure.__name__}\(\): ref is digital "
                                         r"silence throughout.*; scored nan$") as caught:
        assert math.isnan(measure(np.zeros_like(deg), deg, SR))
    assert len(caught) == 1 and caught[0].filename == __file__
    with pytest.warns(UserWarning, match=r"ref holds one value throughout"):
        assert math.isnan(measure(np.full_like(deg, 0.3), deg, SR))


@pytest.mark.parametrize("measure", [siib, siib_gauss])
def test_an_empty_pair_scores_nan_with_a_warning(measure):
    with pytest.warns(UserWarning, match=r"needs at least 87401 .*; scored nan"):
        assert math.isnan(measure(np.zeros(0), np.zeros(0), SR))


@pytest.mark.parametrize("measure", [siib, siib_gauss])
def test_an_output_too_loud_to_square_scores_nan(measure):
    ref = _white(90000, 1)
    with pytest.warns(UserWarning, match=rf"^{measure.__name__}\(\): deg is too loud "
                                         r"against ref to square"):
        assert math.isnan(measure(ref, 1e160 * ref, SR))
    # the reference is brought to unit deviation first, whatever its level
    assert _quiet(measure, 1e160 * ref, 1e160 * ref, SR) == pytest.approx(CEILING,
                                                                         rel=1e-15)


@pytest.mark.parametrize("measure", [siib, siib_gauss])
@pytest.mark.parametrize("sr", [16000.5, 44100.25])
def test_a_rate_that_is_not_a_whole_number_is_refused(measure, sr):
    y = _white(1000, 1)
    with pytest.raises(ValueError, match=rf"^{measure.__name__}\(\): SIIB takes the pair "
                                         r"to 16 kHz by a ratio of whole numbers"):
        measure(y, y, sr)


@pytest.mark.parametrize("measure", [siib, siib_gauss])
@pytest.mark.parametrize("sr, said", [(16, '16'), (1, '1'), (8.0, '8'), (44.1, '44.1'),
                                      (3999, '3999')])
def test_a_rate_in_khz_or_under_4000_hz_is_refused(measure, sr, said, monkeypatch):
    """Taken for a rate in kHz, and refused before anything is resampled:
    taken up 16000 times, a rate of 1 would fill the memory."""
    def no_resampling(*args, **kwargs):
        raise AssertionError("resampled a rate it should have refused")

    import scipy.signal
    monkeypatch.setattr(scipy.signal, 'resample_poly', no_resampling)
    y = _white(1000, 1)
    with pytest.raises(ValueError, match=rf"^{measure.__name__}\(\): sr is in hertz, got "
                                         rf"{re.escape(said)}; SIIB refuses a rate under "
                                         r"4000 Hz"):
        measure(y, y, sr)


def test_4000_hz_is_the_lowest_rate_taken():
    ref = _white(module._min_samples(4000), 1)
    deg = ref + _white(len(ref), 2)
    assert module._ratio(4000) == (4, 1)
    assert math.isfinite(_quiet(siib_gauss, ref, deg, 4000))


@pytest.mark.parametrize("measure", [siib, siib_gauss])
@pytest.mark.parametrize("sr", [0, -16000, float('nan'), float('inf'), True, '16000', None])
def test_a_rate_that_is_not_a_positive_number_is_refused(measure, sr):
    y = _white(1000, 1)
    with pytest.raises(ValueError, match=rf"^{measure.__name__}\(\): sr has to be a "
                                         r"positive sample rate"):
        measure(y, y, sr)


@pytest.mark.parametrize("measure", [siib, siib_gauss])
def test_malformed_signals_are_refused(measure):
    y = _white(1000, 1)
    with pytest.raises(ValueError, match=rf"^{measure.__name__}\(\): ref has shape "
                                         r".*one at a time, or mix down"):
        measure(np.zeros((1000, 2)), y, SR)
    with pytest.raises(ValueError, match=rf"^{measure.__name__}\(\): deg is complex"):
        measure(y, y + 0j, SR)
    with pytest.raises(ValueError, match=rf"^{measure.__name__}\(\): ref is a single number"):
        measure(np.float64(0.5), y, SR)
    bad = y.copy()
    bad[10] = np.nan
    with pytest.raises(ValueError, match=rf"^{measure.__name__}\(\): deg has 1 sample"):
        measure(y, bad, SR)


@pytest.mark.parametrize("measure", [siib, siib_gauss])
def test_a_length_mismatch_warns_and_a_large_one_is_refused(measure):
    ref = _white(90000, 1)
    deg = ref + _white(90000, 2)
    with pytest.warns(UserWarning, match=rf"^{measure.__name__}\(\): ref is 90000 samples "
                                         r"and deg 89900: only the first") as caught:
        value = measure(ref, deg[:-100], SR)
    assert caught[0].filename == __file__
    assert value == measure(ref[:-100], deg[:-100], SR)
    assert evaluator.MAX_LENGTH_MISMATCH == 0.5
    with pytest.raises(ValueError, match=rf"^{measure.__name__}\(\): .*unmeasured"):
        measure(ref, deg[:40000], SR)


def test_integer_audio_is_read_on_its_own_full_scale():
    ref = _white(90000, 1)
    as_int = np.round(ref / np.max(np.abs(ref)) * 16000).astype(np.int16)
    assert siib_gauss(as_int, as_int / 32768.0, SR) == pytest.approx(CEILING, rel=1e-15)


def test_the_scores_are_floats(sweep):
    _, value, gauss = sweep[0]
    assert type(value) is float and type(gauss) is float


def test_the_module_reads_as_python_3_9():
    import ast
    with open(module.__file__, encoding='utf-8') as f:
        ast.parse(f.read(), feature_version=(3, 9))
