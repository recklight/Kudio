# -*- coding: utf-8 -*-
"""WADA-SNR: the curve against what can be worked out in closed form or by a
second route, the estimate against signals whose SNR is known, and the clips
it will not score."""
from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from kudio.core.wada import _SNR_SPAN, _curve, _g, _snr_of, wada_snr

SR = 16000


def _gamma_speech(n: int, rng) -> np.ndarray:
    """The model's own speech: two-sided gamma samples of shape 0.4."""
    return rng.gamma(0.4, 1.0, n) * rng.choice([-1.0, 1.0], n)


def _at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    gain = math.sqrt(np.mean(clean ** 2) / (np.mean(noise ** 2) * 10 ** (snr_db / 10)))
    return clean + gain * noise


def _syllables(seconds: float, seed: int) -> np.ndarray:
    """Voiced syllables in words, with digital silence between the words.

    Syllables of 150 to 300 ms, each a harmonic series on an f0 of 90 to
    220 Hz with a slow wobble, under a sin^2 envelope; words of three to six
    syllables, 0.2 to 0.6 s apart.
    """
    rng = np.random.default_rng(seed)
    y = np.zeros(int(SR * seconds))
    t = 0.0
    while t < seconds:
        for _ in range(int(rng.integers(3, 7))):
            dur = rng.uniform(0.15, 0.30)
            n, start = int(dur * SR), int(t * SR)
            if start + n > len(y):
                break
            tt = np.arange(n) / SR
            f0 = rng.uniform(90, 220) * (1 + 0.05 * np.sin(2 * np.pi * rng.uniform(2, 5) * tt))
            phase = 2 * np.pi * np.cumsum(f0) / SR
            voiced = sum(np.sin(k * phase) / k for k in range(1, 30))
            y[start:start + n] += voiced * np.sin(np.pi * tt / dur) ** 2 * rng.uniform(0.3, 1.0)
            t += dur
        t += rng.uniform(0.2, 0.6)
    return y


def _left_out(y: np.ndarray) -> float:
    """What the estimate would be with the zeros simply left out."""
    z = y[y != 0.0]
    z = np.abs(z - z.mean())
    return _snr_of(math.log(np.mean(z)) - np.mean(np.log(z)))


def _quiet(fn, *args):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return fn(*args)


# -------------------------------------------------------------- the curve

def test_the_curve_runs_from_noise_alone_to_speech_alone():
    """Both ends have closed forms. Gaussian noise alone:
    ln E|v| - E ln|v| = ln sqrt(2/pi) + (euler_gamma + ln 2) / 2. Gamma speech
    alone: ln(shape) - digamma(shape). Neither appears in the code."""
    from scipy.special import digamma
    noise_alone = 0.5 * math.log(2 / math.pi) + 0.5 * (np.euler_gamma + math.log(2))
    speech_alone = math.log(0.4) - digamma(0.4)
    assert _g(-200.0) == pytest.approx(noise_alone, abs=1e-10)
    # at +300 dB the noise still reaches the samples nearest zero: 1.8e-6 short
    assert _g(300.0) == pytest.approx(speech_alone, abs=1e-5)
    assert noise_alone < _g(_SNR_SPAN[0]) < _g(_SNR_SPAN[1]) < speech_alone


def test_the_curve_rises_strictly_so_every_g_has_one_snr():
    snr, curve = _curve()
    assert snr[0] == _SNR_SPAN[0] and snr[-1] == _SNR_SPAN[1]
    assert np.all(np.diff(curve) > 0)


@pytest.mark.parametrize("snr_db", [-10.0, 0.0, 20.0, 60.0])
def test_the_curve_agrees_with_a_second_route_through_the_noisy_log(snr_db):
    """The same expectations as Eq. 17 of the paper, taken another way:
    E|a + v| is the folded normal's mean, E ln|a + v| the Poisson mixture of
    central chi-square logs, and the gamma expectation is done by quad."""
    from scipy import integrate, special, stats
    shape = 0.4
    s = math.sqrt(shape * (shape + 1) / 10 ** (snr_db / 10))

    def folded_mean(a):
        return (s * math.sqrt(2 / math.pi) * math.exp(-a * a / (2 * s * s))
                + a * math.erf(a / (s * math.sqrt(2))))

    def mean_log(a):
        lam = a * a / (2 * s * s)
        if lam > 200:
            return math.log(a) - s * s / (2 * a * a)
        j = np.arange(int(lam + 40 * math.sqrt(lam + 1) + 60))
        mix = float(np.sum(stats.poisson.pmf(j, lam) * special.digamma(0.5 + j)))
        return math.log(s) + 0.5 * (math.log(2) + mix)

    def expect(f):
        density = lambda a: a ** (shape - 1) * math.exp(-a) / math.gamma(shape)
        return (integrate.quad(lambda a: density(a) * f(a), 0, 1, limit=400)[0]
                + integrate.quad(lambda a: density(a) * f(a), 1, np.inf, limit=400)[0])

    assert _g(snr_db) == pytest.approx(math.log(expect(folded_mean)) - expect(mean_log),
                                       abs=1e-6)


# entries of Alpha0.400000.txt, the table wada_snr.m reads in LabROSA's
# snreval of 2014-07-01 (mirrored at github.com/ludlows/snreval) and that its
# comment says came from Kim's C++ code: (SNR in dB, G)
_PUBLISHED = [(-10, 0.412317178), (0, 0.462211529), (10, 0.667508177),
              (20, 0.95655449), (30, 1.197035028)]


@pytest.mark.parametrize("snr_db, g", _PUBLISHED)
def test_the_published_table_reads_within_a_fifth_of_a_db_on_this_curve(snr_db, g):
    """The table came from a Monte Carlo integration, and its own noise is
    what separates it from the curve: read on the curve, these five entries
    land 0.006 to 0.163 dB from where the table puts them."""
    assert _snr_of(g) == pytest.approx(snr_db, abs=0.2)


def test_the_published_tables_last_entry_is_past_the_top_of_this_curve():
    """The table's 100 dB entry, 1.632041, is above the model's G at 100 dB
    (1.6268), and G only reaches 1.6451 with no noise at all."""
    _, curve = _curve()
    assert curve[-1] < 1.632041021


def test_inverting_the_curve_gives_back_the_snr():
    for snr_db in (-19.5, -3.3, 0.0, 7.25, 33.3, 99.9):
        assert _snr_of(_g(snr_db)) == pytest.approx(snr_db, abs=1e-8)


def test_the_curve_is_read_from_minus_20_to_100_db_and_nan_just_past_either_end():
    """-20 to 100 dB is the span of the table published with the method. A G
    at either end of the curve, or 1e-9 past it, gets nan and a warning and
    never an exception; 1e-9 inside, it reads the end of the span."""
    _, curve = _curve()
    for g in (curve[0] - 1e-9, curve[0]):
        with pytest.warns(UserWarning, match="no more peaked than Gaussian"):
            assert math.isnan(_snr_of(g))
    for g in (curve[-1], curve[-1] + 1e-9):
        with pytest.warns(UserWarning, match="sparser than gamma-distributed speech"):
            assert math.isnan(_snr_of(g))
    assert _quiet(_snr_of, curve[0] + 1e-9) == pytest.approx(-20.0, abs=1e-3)
    assert _quiet(_snr_of, curve[-1] - 1e-9) == pytest.approx(100.0, abs=1e-3)


# ------------------------------------------------ signals whose SNR is known

@pytest.mark.parametrize("snr_db", [-5.0, 0.0, 10.0, 20.0, 30.0])
def test_the_model_signal_reads_back_its_own_snr(snr_db):
    """Gamma speech in Gaussian noise, the case the method is built on: five
    clips of 10 s each, whose mean estimate lands within 0.2 dB of the truth,
    and within 0.5 dB at -5 dB, where the curve is flatter and the same
    scatter in G spans more decibels."""
    estimates = []
    for seed in range(5):
        rng = np.random.default_rng(1000 + 10 * int(snr_db + 5) + seed)
        clean = _gamma_speech(10 * SR, rng)
        noisy = _at_snr(clean, rng.standard_normal(len(clean)), snr_db)
        estimates.append(_quiet(wada_snr, noisy, SR))
    assert np.mean(estimates) == pytest.approx(snr_db, abs=0.5 if snr_db < 0 else 0.2)


def test_a_harmonic_signal_with_pauses_reads_close_up_to_20_db_and_high_above():
    """Ten seconds of voiced syllables, pauses between the words, white noise
    on top. Its samples are not spread like gamma speech, and the estimate
    shows it the way the paper's Fig. 2 says it will: within 1.6 dB of the
    truth from 0 to 20 dB, and 6.6 dB high at 30 dB, where the speech decides
    more than the noise does."""
    clean = _syllables(10.0, seed=100)
    noise = np.random.default_rng(1).standard_normal(len(clean))
    got = {snr: _quiet(wada_snr, _at_snr(clean, noise, snr), SR)
           for snr in (0, 5, 10, 15, 20, 30)}
    for snr in (0, 5, 10, 15, 20):
        assert got[snr] == pytest.approx(snr, abs=2.0), got
    assert got[30] > 34.0, "past 20 dB the answer says 'clean', not how clean"


def test_gaussian_noise_alone_is_nan_or_reads_minus_20_db_or_above():
    """No speech at all puts G on the low end of the curve, scattered about
    the model's value at -20 dB. Of these 20 clips of 10 s, 12 fall at or
    below it and are nan with a warning, and the other 8 read -13.9 to
    -10.3 dB. None reads under -20 dB, and none raises."""
    nan = 0
    for seed in range(20):
        noise = np.random.default_rng(500 + seed).standard_normal(10 * SR)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            got = wada_snr(noise, SR)
        if math.isnan(got):
            nan += 1
            assert "no more peaked than Gaussian" in str(caught[0].message)
        else:
            assert got >= -20.0 and not caught
    assert nan == 12


def test_the_level_does_not_enter():
    rng = np.random.default_rng(3)
    noisy = _at_snr(_gamma_speech(5 * SR, rng), rng.standard_normal(5 * SR), 10.0)
    assert wada_snr(10.0 * noisy, SR) == pytest.approx(wada_snr(noisy, SR), abs=1e-9)
    assert wada_snr(1e-3 * noisy, SR) == pytest.approx(wada_snr(noisy, SR), abs=1e-9)


def test_int16_audio_reads_like_the_float_it_came_from():
    """Peak at -1 dBFS and 10 dB of SNR: the noise is 632 LSB RMS, and
    rounding to int16 moves the answer by 0.01 dB."""
    rng = np.random.default_rng(4)
    noisy = _at_snr(_gamma_speech(10 * SR, rng), rng.standard_normal(10 * SR), 10.0)
    noisy = noisy / np.max(np.abs(noisy)) * 10 ** (-1 / 20)
    pcm = np.round(noisy * 32768).astype(np.int16)
    assert _quiet(wada_snr, pcm, SR) == pytest.approx(wada_snr(noisy, SR), abs=0.02)


def test_zeros_that_rounding_leaves_in_quiet_int16_audio_are_not_digital_silence():
    """At 60 dB, peaked at -1 dBFS, the noise is 2.4 LSB RMS and 1.1% of the
    int16 samples round to exactly zero: more than the 1% that digital
    silence may fill, but scattered one or two at a time, so the clip is
    scored."""
    rng = np.random.default_rng(12)
    noisy = _at_snr(_gamma_speech(10 * SR, rng), rng.standard_normal(10 * SR), 60.0)
    pcm = np.round(noisy / np.max(np.abs(noisy)) * 10 ** (-1 / 20) * 32768).astype(np.int16)
    assert np.mean(pcm == 0) > 0.01
    assert math.isfinite(_quiet(wada_snr, pcm, SR))


def test_a_dc_offset_is_taken_out_before_g_is_measured():
    """Fig. 3 of the paper subtracts the mean first. On this 10 dB clip an
    offset of 0.3 times the RMS would otherwise read about 4 dB low."""
    rng = np.random.default_rng(5)
    noisy = _at_snr(_gamma_speech(10 * SR, rng), rng.standard_normal(10 * SR), 10.0)
    shifted = noisy + 0.3 * np.sqrt(np.mean(noisy ** 2))
    assert wada_snr(shifted, SR) == pytest.approx(wada_snr(noisy, SR), abs=1e-6)
    magnitude = np.abs(shifted)
    left_in = _snr_of(math.log(np.mean(magnitude)) - np.mean(np.log(magnitude)))
    assert left_in < wada_snr(noisy, SR) - 3.5


def test_zero_padding_is_left_out_rather_than_floored():
    """Ten per cent of zeros around a 0 dB clip. Left out, as the paper's
    first option has it, they change nothing and need no warning. Floored at
    1e-10 after peak normalisation, as the published ports do, they read as
    speech so sparse that G lands past the top of the curve, where the ports
    report 100 dB."""
    rng = np.random.default_rng(6)
    noisy = _at_snr(_gamma_speech(9 * SR, rng), rng.standard_normal(9 * SR), 0.0)
    padded = np.concatenate([np.zeros(SR // 2), noisy, np.zeros(SR // 2)])
    assert _quiet(wada_snr, padded, SR) == pytest.approx(wada_snr(noisy, SR), abs=1e-9)
    floored = np.abs(padded / np.max(np.abs(padded)))
    floored[floored < 1e-10] = 1e-10
    _, curve = _curve()
    assert math.log(np.mean(floored)) - np.mean(np.log(floored)) > curve[-1]


def test_a_dropout_under_1_percent_of_the_clip_is_left_out_and_one_over_it_is_not():
    """Ten seconds at 10 dB with a stretch zeroed in the middle: 50 ms, half
    a per cent, is left out and moves the estimate by 0.008 dB; 200 ms, two
    per cent, is more digital silence than the method scores through."""
    rng = np.random.default_rng(11)
    noisy = _at_snr(_gamma_speech(10 * SR, rng), rng.standard_normal(10 * SR), 10.0)
    short, long = noisy.copy(), noisy.copy()
    short[5 * SR:5 * SR + SR // 20] = 0.0
    long[5 * SR:5 * SR + SR // 5] = 0.0
    assert _quiet(wada_snr, short, SR) == pytest.approx(wada_snr(noisy, SR), abs=0.01)
    with pytest.warns(UserWarning, match="digital silence"):
        assert math.isnan(wada_snr(long, SR))


def test_a_run_of_zeros_is_digital_silence_from_10_ms_on():
    """Eleven runs of zeros in 10 s at 10 dB, 1.1% of it all told. At 160
    samples each, 10 ms at 16 kHz, they are digital silence and the clip is
    refused; one sample shorter, they are left out like any other zeros and
    move the estimate by 0.018 dB."""
    rng = np.random.default_rng(11)
    noisy = _at_snr(_gamma_speech(10 * SR, rng), rng.standard_normal(10 * SR), 10.0)
    holed = {}
    for length in (159, 160):
        holed[length] = noisy.copy()
        for k in range(1, 12):
            holed[length][k * 13000:k * 13000 + length] = 0.0
    assert _quiet(wada_snr, holed[159], SR) == pytest.approx(wada_snr(noisy, SR), abs=0.02)
    with pytest.warns(UserWarning, match="digital silence"):
        assert math.isnan(wada_snr(holed[160], SR))


# ------------------------------------------------- what cannot be measured

@pytest.mark.parametrize("clip, match", [
    (np.zeros(SR), "all 16000 samples are zero"),
    (np.full(SR, 0.25), "same value"),
    (np.zeros(0), "empty"),
])
def test_silence_and_nothing_at_all_are_nan_with_a_warning(clip, match):
    with pytest.warns(UserWarning, match=match):
        assert math.isnan(wada_snr(clip, SR))


def test_a_noise_free_clip_with_its_pauses_at_digital_silence_is_nan():
    """The syllables with nothing added: 2.27 s of the 9.98 s from the first
    sound to the last are zeros between the words. Left out, those zeros
    would take the pauses with them, and a clip with no noise in it would
    read 23.3 dB."""
    clean = _syllables(10.0, seed=100)
    with pytest.warns(UserWarning, match=r"2\.27 s of the 9\.98 s .* digital silence"):
        assert math.isnan(wada_snr(clean, SR))
    assert _left_out(clean) == pytest.approx(23.3, abs=0.05)


def test_speech_gated_to_zero_in_its_pauses_is_nan():
    """The syllables at 20 dB of white noise, muted wherever they are silent,
    as a noise gate mutes them. Left out, the zeros would take the estimate
    from the 20.7 dB of the ungated clip to 13.4 dB."""
    clean = _syllables(10.0, seed=100)
    noisy = _at_snr(clean, np.random.default_rng(1).standard_normal(len(clean)), 20.0)
    gated = np.where(clean == 0.0, 0.0, noisy)
    with pytest.warns(UserWarning, match="digital silence"):
        assert math.isnan(wada_snr(gated, SR))
    assert _quiet(wada_snr, noisy, SR) == pytest.approx(20.7, abs=0.05)
    assert _left_out(gated) == pytest.approx(13.4, abs=0.05)


def test_under_a_second_of_audio_is_nan_with_a_warning():
    """One second is the shortest clip scored, counted from the first
    nonzero sample to the last, so zeros that pad a short clip out do not
    make it long enough."""
    rng = np.random.default_rng(9)
    noisy = _at_snr(_gamma_speech(2 * SR, rng), rng.standard_normal(2 * SR), 10.0)
    assert math.isfinite(_quiet(wada_snr, noisy[:SR], SR))
    with pytest.warns(UserWarning, match="at least 1 s"):
        assert math.isnan(wada_snr(noisy[:SR - 1], SR))
    padded = np.concatenate([np.zeros(SR), noisy[:SR // 2], np.zeros(SR)])
    with pytest.warns(UserWarning, match="once the zeros at its ends are left out"):
        assert math.isnan(wada_snr(padded, SR))


def test_a_hum_is_steadier_than_any_noise_the_curve_covers():
    """A sine's G is ln(4 / pi) = 0.24, below Gaussian noise's 0.41: it is
    not speech in Gaussian noise at any SNR, and the ports would call it
    -20 dB. The phase is offset so that no sample lands on a zero crossing,
    where floating point leaves a remainder instead of 0 whose log stands
    far below every other sample's."""
    hum = np.sin(2 * np.pi * 50 * np.arange(5 * SR) / SR + 0.3)
    with pytest.warns(UserWarning, match="no more peaked than Gaussian"):
        assert math.isnan(wada_snr(hum, SR))


def test_clicks_are_sparser_than_any_speech_the_curve_covers():
    """A click every 1000 samples, alternating in sign so the mean stays at
    zero, over a floor 80 dB down: the ports would call it 100 dB, and it is
    not a measurement."""
    clicks = 1e-4 * np.random.default_rng(7).standard_normal(5 * SR)
    clicks[::2000] = 1.0
    clicks[1000::2000] = -1.0
    with pytest.warns(UserWarning, match="sparser than gamma-distributed speech"):
        assert math.isnan(wada_snr(clicks, SR))


def test_one_channel_held_as_a_column_or_a_row_is_read_as_mono():
    rng = np.random.default_rng(10)
    noisy = _at_snr(_gamma_speech(5 * SR, rng), rng.standard_normal(5 * SR), 10.0)
    assert wada_snr(noisy[:, None], SR) == wada_snr(noisy, SR)
    assert wada_snr(noisy[None, :], SR) == wada_snr(noisy, SR)


def test_channels_are_refused():
    with pytest.raises(ValueError, match="mono"):
        wada_snr(np.zeros((SR, 2)) + 0.1, SR)


@pytest.mark.parametrize("sr", [0, -16000, float('nan')])
def test_a_rate_that_is_not_positive_is_refused(sr):
    with pytest.raises(ValueError, match="positive sample rate"):
        wada_snr(np.ones(SR), sr)


def test_samples_that_are_not_finite_are_refused():
    noisy = np.random.default_rng(8).standard_normal(SR)
    noisy[100] = np.nan
    with pytest.raises(ValueError, match="not finite"):
        wada_snr(noisy, SR)
