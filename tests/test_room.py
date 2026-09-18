# -*- coding: utf-8 -*-
"""Rooms: a synthetic one whose answer is known, and the measurement that has
to recover it."""
from __future__ import annotations

import numpy as np
import pytest

import kudio
from kudio.exceptions import FeatureError

from conftest import SR, make_speech


# -------------------------------------------------------------- round trips

@pytest.mark.parametrize("want", [0.2, 0.3, 0.5, 0.8, 1.2, 2.0])
def test_the_reverberation_time_asked_for_is_the_one_measured(want):
    """The point of a synthetic room: the answer is known before the
    measurement runs, so the measurement can be wrong and be caught."""
    room = kudio.rt60(kudio.rir(SR, rt60=want, seed=0), SR)

    assert room.t30 == pytest.approx(want, rel=0.05)
    assert room.t20 == pytest.approx(want, rel=0.06)
    assert room.seconds == room.t30
    assert room.reliable()
    assert room.fit > 0.99


@pytest.mark.parametrize("sr", [8000, 16000, 44100, 48000])
def test_the_rate_does_not_change_the_room(sr):
    """Reverberation is a property of the room, not of the sampling."""
    room = kudio.rt60(kudio.rir(sr, rt60=0.6, seed=1), sr)
    assert room.t30 == pytest.approx(0.6, rel=0.06)


@pytest.mark.parametrize("want", [-6.0, -3.0, 0.0, 3.0, 6.0, 12.0, 20.0])
def test_the_direct_to_reverberant_ratio_round_trips(want):
    """`drr_db` is solved for against the same window the measurement uses;
    a knob whose number does not come back out is not a knob."""
    room = kudio.rt60(kudio.rir(SR, rt60=0.5, drr_db=want, seed=0), SR)
    assert room.drr == pytest.approx(want, abs=0.1)


def test_a_closer_microphone_is_a_clearer_one():
    far = kudio.rt60(kudio.rir(SR, rt60=0.6, drr_db=-3.0, seed=2), SR)
    near = kudio.rt60(kudio.rir(SR, rt60=0.6, drr_db=15.0, seed=2), SR)

    assert near.c50 > far.c50
    assert near.c80 > far.c80
    assert near.drr > far.drr
    # the tail itself is a property of the room, not of where you stood
    assert near.t30 == pytest.approx(far.t30, rel=0.05)


def test_the_same_seed_is_the_same_room():
    a = kudio.rir(SR, rt60=0.5, seed=7)
    b = kudio.rir(SR, rt60=0.5, seed=7)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, kudio.rir(SR, rt60=0.5, seed=8))


# ------------------------------------------------------ the measurement side

def test_the_decay_curve_starts_at_zero_and_falls():
    curve = kudio.schroeder_curve(kudio.rir(SR, rt60=0.5, seed=0))
    assert curve[0] == pytest.approx(0.0, abs=1e-9)
    assert np.all(np.diff(curve) <= 1e-9), "backward integration cannot rise"
    assert curve[-1] < -40


def test_a_longer_tail_measures_longer():
    short = kudio.rt60(kudio.rir(SR, rt60=0.3, seed=0), SR)
    long = kudio.rt60(kudio.rir(SR, rt60=1.2, seed=0), SR)
    assert long.t30 > short.t30 * 3


def test_a_truncated_impulse_response_is_flagged_not_believed():
    """Backward integration normalises to the energy it was given, so a
    truncated tail still plunges 35 dB -- and reports a T30 that is whatever
    the truncation made it. Cutting a 1 s room to 50 ms reads 0.19 s."""
    ir = kudio.rir(SR, rt60=1.0, seed=0)[:int(0.05 * SR)]
    room = kudio.rt60(ir, SR)

    assert np.isfinite(room.t30)                  # a number comes back
    assert room.t30 < 0.5                         # and it is badly wrong
    assert not room.reliable()                    # which is what has to show


@pytest.mark.parametrize("label,signal", [
    ("speech", make_speech()),
    ("noise", (np.random.default_rng(0)
               .standard_normal(SR * 3)).astype(np.float32)),
])
def test_something_that_is_not_an_impulse_response_is_not_believed(label, signal):
    """Backward integration makes *any* signal decay, and over a long clip
    that decay can fit a straight line well: the speech fixture reaches
    R² 0.94 and claims a 3.9 second room. What gives it away is EDT and T30
    disagreeing by a factor of eleven."""
    room = kudio.rt60(signal, SR)
    assert not room.reliable(), f"{label}: fit {room.fit:.3f}, "                                 f"edt/t30 {room.edt / room.t30:.2f}"


def test_the_straightness_alone_is_not_enough():
    """Stated as its own test because it was a real miss: an R² threshold by
    itself passes the speech fixture."""
    room = kudio.rt60(make_speech(), SR)
    assert room.fit > 0.9, "the line really does fit"
    assert room.edt / room.t30 > 5, "and the two spans really do disagree"
    assert not room.reliable()


def test_the_summary_is_plain_numbers():
    room = kudio.rt60(kudio.rir(SR, rt60=0.5, seed=0), SR)
    summary = room.summary()
    assert set(summary) == {'t20', 't30', 'edt', 'c50', 'c80', 'drr', 'fit'}
    assert summary['t30'] == room.t30


def test_a_reliable_room_prints_without_a_warning():
    room = kudio.rt60(kudio.rir(SR, rt60=0.5, seed=0), SR)
    assert "T30" in str(room) and "C50" in str(room) and "DRR" in str(room)
    assert "⚠" not in str(room)


# ------------------------------------------------------------- applying one

def test_a_room_smears_the_signal_across_the_silence_after_it():
    """The whole difference from additive noise: nothing was added, and yet
    there is now sound where there was none."""
    dry = np.concatenate([make_speech(seconds=6.0),
                          np.zeros(SR, dtype=np.float32)])
    wet = kudio.apply_rir(dry, kudio.rir(SR, rt60=0.8, seed=0),
                          normalise=False)

    quiet = slice(int(6.05 * SR), int(6.5 * SR))
    assert np.max(np.abs(dry[quiet])) == 0.0
    assert np.max(np.abs(wet[quiet])) > 1e-3


def test_the_length_is_kept_by_default():
    dry = make_speech()
    assert len(kudio.apply_rir(dry, kudio.rir(SR, rt60=0.5, seed=0))) == len(dry)

    longer = kudio.apply_rir(dry, kudio.rir(SR, rt60=0.5, seed=0), trim=False)
    assert len(longer) > len(dry)


def test_the_level_is_kept_by_default():
    """A room has a gain, and an augmentation that also changes the level is
    measuring two things at once."""
    dry = make_speech()
    wet = kudio.apply_rir(dry, kudio.rir(SR, rt60=0.5, drr_db=12.0, seed=0))
    assert np.max(np.abs(wet)) == pytest.approx(np.max(np.abs(dry)), rel=1e-5)


def test_a_dry_room_leaves_the_signal_almost_alone():
    """An impulse response that is a single spike is not a room."""
    dry = make_speech()
    spike = np.zeros(128, dtype=np.float32)
    spike[0] = 1.0
    assert np.allclose(kudio.apply_rir(dry, spike), dry, atol=1e-5)


def test_reverberation_is_what_the_denoiser_cannot_touch():
    """Worth stating once: `spectral_enhance` estimates an additive noise
    spectrum, and a room does not add one."""
    dry = make_speech()
    wet = kudio.apply_rir(dry, kudio.rir(SR, rt60=0.8, seed=0))
    cleaned = kudio.spectral_enhance(wet, SR, method="logmmse")

    a, b = kudio.align(dry, wet, SR, min_correlation=0.0)
    before = kudio.si_sdr(a, b)
    a, b = kudio.align(dry, cleaned, SR, min_correlation=0.0)
    assert kudio.si_sdr(a, b) < before + 3.0, "denoising should not fix a room"


# ------------------------------------------------------------ refused input

@pytest.mark.parametrize("kwargs", [{"rt60": 0.0}, {"rt60": -1.0}])
def test_an_impossible_room_is_refused(kwargs):
    with pytest.raises(FeatureError, match="rt60"):
        kudio.rir(SR, **kwargs)


def test_a_room_too_short_to_hold_its_own_tail_is_refused():
    with pytest.raises(FeatureError, match="raise rt60"):
        kudio.rir(SR, rt60=0.5, seconds=0.00001)


def test_a_drr_the_tail_cannot_reach_is_refused_rather_than_approximated():
    with pytest.raises(FeatureError, match="drr_db"):
        kudio.rir(SR, rt60=0.5, drr_db=-60.0, seed=0)


def test_bad_rates_are_refused():
    with pytest.raises(FeatureError, match="positive rate"):
        kudio.rir(0, rt60=0.5)
    with pytest.raises(FeatureError, match="positive rate"):
        kudio.rt60(kudio.rir(SR, rt60=0.5, seed=0), 0)


def test_measuring_nothing_is_refused():
    with pytest.raises(FeatureError, match="impulse response"):
        kudio.rt60(np.zeros(1, dtype=np.float32), SR)
    with pytest.raises(FeatureError, match="impulse response"):
        kudio.apply_rir(make_speech(), np.zeros(0, dtype=np.float32))
