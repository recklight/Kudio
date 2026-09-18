# -*- coding: utf-8 -*-
"""Alignment: the assumption every reference metric makes and none can check."""
from __future__ import annotations

import numpy as np
import pytest

import kudio
from kudio.exceptions import FeatureError

from conftest import SR, make_speech


def delayed(y: np.ndarray, samples: int) -> np.ndarray:
    """*y* starting *samples* later, padded at the front with silence."""
    return np.concatenate([np.zeros(samples, dtype=np.float32), y])


# ---------------------------------------------------------- the delay itself

@pytest.mark.parametrize("shift", [0, 1, 7, 16, 160, 480, 4000])
def test_a_known_delay_is_found_exactly(shift):
    clean = make_speech()
    found = kudio.find_delay(clean, delayed(clean, shift), SR)

    assert found.delay == shift
    assert found.strength == pytest.approx(1.0, abs=1e-6)
    assert found.seconds == pytest.approx(shift / SR)
    assert found.milliseconds == pytest.approx(shift / SR * 1000)


def test_a_signal_that_starts_first_reports_a_negative_delay():
    """The sign says which way to move it, so it has to be unambiguous."""
    clean = make_speech()
    found = kudio.find_delay(clean, clean[320:], SR)

    assert found.delay == -320
    assert "leads" in str(found)
    assert kudio.find_delay(clean, delayed(clean, 320), SR).delay == 320
    assert "lags" in str(kudio.find_delay(clean, delayed(clean, 320), SR))


def test_identical_signals_are_already_aligned():
    clean = make_speech()
    found = kudio.find_delay(clean, clean, SR)
    assert found.delay == 0
    assert "already aligned" in str(found)


def test_one_sample_reads_as_one_sample():
    clean = make_speech()
    assert "by 1 sample " in str(kudio.find_delay(clean, delayed(clean, 1), SR))
    assert "by 2 samples " in str(kudio.find_delay(clean, delayed(clean, 2), SR))


# ------------------------------------------------- what it is actually for

@pytest.mark.parametrize("shift", [1, 16, 160, 480])
def test_aligning_restores_a_metric_that_misalignment_destroyed(shift):
    """A clip against itself scores +145 dB. One sample of delay costs 134 of
    them, and a millisecond scores worse than not processing at all -- with
    nothing in the number to say which of the two happened."""
    clean = make_speech()
    moved = delayed(clean, shift)

    assert kudio.si_sdr(clean, moved[:len(clean)]) < 15.0    # the trap
    a, b = kudio.align(clean, moved, SR)
    assert kudio.si_sdr(a, b) > 100.0                        # the way out


def test_a_processing_delay_is_recovered_on_real_processing():
    """Not a synthetic shift: a filter with real latency."""
    clean = make_speech()
    filtered = np.concatenate([np.zeros(64, dtype=np.float32),
                               kudio.lowpass(clean, SR, cutoff=3000)])

    found = kudio.find_delay(clean, filtered, SR)
    assert found.delay == 64
    assert found.confident()

    a, b = kudio.align(clean, filtered, SR)
    assert len(a) == len(b)
    assert kudio.si_sdr(a, b) > kudio.si_sdr(clean, filtered[:len(clean)])


def test_the_aligned_pair_is_the_overlap_and_nothing_else():
    """Padding the front of a clip and then taking the delay back out leaves
    the whole clip on both sides -- nothing added, nothing lost."""
    clean = make_speech()
    a, b = kudio.align(clean, delayed(clean, 800), SR)

    assert len(a) == len(b) == len(clean)
    assert np.allclose(a, b, atol=1e-6)


def test_a_shorter_take_gives_a_shorter_overlap():
    """Half a clip against the whole one: what comes back is the half they
    share, not the whole padded out to match."""
    clean = make_speech()
    half = clean[:len(clean) // 2]
    a, b = kudio.align(clean, half, SR)

    assert len(a) == len(b) == len(half)
    assert np.allclose(a, b, atol=1e-6)


# --------------------------------------------------------------- confidence

def test_two_unrelated_recordings_are_refused():
    """Cross-correlation always has a maximum, so it always returns a delay --
    including for signals that have nothing to do with each other."""
    clean = make_speech()
    noise = (np.random.default_rng(0)
             .standard_normal(len(clean)).astype(np.float32))

    found = kudio.find_delay(clean, noise, SR)
    assert not found.confident()
    with pytest.raises(FeatureError, match="same recording"):
        kudio.align(clean, noise, SR)


def test_you_can_insist():
    clean = make_speech()
    noise = (np.random.default_rng(1)
             .standard_normal(len(clean)).astype(np.float32))
    a, b = kudio.align(clean, noise, SR, min_correlation=0.0)
    assert len(a) == len(b) > 0


def test_a_denoised_take_is_still_recognisably_the_same_recording():
    """The threshold has to pass real processing, or it is just in the way."""
    clean = make_speech()
    noisy = kudio.add_noise_snr(clean, np.random.default_rng(2).standard_normal(
        len(clean)).astype(np.float32), snr_db=0, seed=0)
    enhanced = kudio.spectral_enhance(noisy, SR, method="logmmse")

    found = kudio.find_delay(clean, enhanced, SR)
    assert found.confident(), f"only {found.strength:.3f}"


# ---------------------------------------------------------------- polarity

def test_an_inverted_copy_is_named_rather_than_quietly_fixed():
    """It is the same take and something flipped it -- a finding, not a
    detail to correct on the way past."""
    clean = make_speech()
    found = kudio.find_delay(clean, -clean, SR)

    assert found.delay == 0
    assert found.correlation == pytest.approx(-1.0, abs=1e-6)
    assert found.inverted
    assert found.strength == pytest.approx(1.0, abs=1e-6)
    assert "inverted" in str(found)


def test_an_inverted_copy_still_aligns(caplog):
    clean = make_speech()
    with caplog.at_level("WARNING"):
        a, b = kudio.align(clean, -delayed(clean, 200)[0:], SR)
    assert len(a) == len(b)
    assert any("inverted" in r.message for r in caplog.records)


# ------------------------------------------------------------ bounding it

def test_the_search_can_be_bounded():
    """A bound is for when you know the answer is small. It restricts the
    search, and what comes back is the best lag *inside* it -- which is not
    the right one if the right one is outside."""
    clean = make_speech()
    moved = delayed(clean, 8000)                    # half a second

    unbounded = kudio.find_delay(clean, moved, SR)
    assert unbounded.delay == 8000
    assert unbounded.strength == pytest.approx(1.0, abs=1e-6)

    near = kudio.find_delay(clean, moved, SR, max_seconds=0.1)
    assert abs(near.delay) <= int(0.1 * SR)
    assert near.strength < unbounded.strength


def test_the_confidence_says_same_recording_not_right_lag():
    """Worth being explicit about, because they are different questions.

    `make_speech` is two harmonic bursts, so a half-second shift still lines
    part of it up: the correlation at the wrong lag is 0.55, over the
    threshold. The threshold catches "somebody picked the wrong file"; it is
    not a check that the lag you were handed is the one you wanted.
    """
    clean = make_speech()
    wrong = kudio.find_delay(clean, delayed(clean, 8000), SR, max_seconds=0.1)

    assert wrong.delay != 8000                       # not the right answer
    assert wrong.confident()                         # and still plausible


def test_a_zero_bound_asks_whether_they_already_line_up():
    clean = make_speech()
    found = kudio.find_delay(clean, delayed(clean, 500), SR, max_seconds=0.0)
    assert found.delay == 0


def test_applying_a_delay_measured_once():
    """Measure on one passage, apply to everything that came through the same
    chain."""
    clean = make_speech()
    found = kudio.find_delay(clean, delayed(clean, 300), SR)

    other = make_speech(seed=5)
    a, b = found.apply(other, delayed(other, 300))
    assert np.allclose(a, b, atol=1e-6)


# ------------------------------------------------------------ refused input

def test_stereo_is_refused_rather_than_mixed_down():
    clean = make_speech()
    stereo = np.stack([clean, clean], axis=1)
    with pytest.raises(FeatureError, match="mono"):
        kudio.find_delay(clean, stereo, SR)


def test_empty_audio_is_refused():
    with pytest.raises(FeatureError, match="both sides"):
        kudio.find_delay(make_speech(), np.zeros(0, dtype=np.float32), SR)


def test_a_bad_rate_is_refused():
    clean = make_speech()
    with pytest.raises(FeatureError, match="positive rate"):
        kudio.find_delay(clean, clean, 0)


def test_a_negative_bound_is_refused():
    clean = make_speech()
    with pytest.raises(FeatureError, match="max_seconds"):
        kudio.find_delay(clean, clean, SR, max_seconds=-1.0)
