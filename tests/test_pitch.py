# -*- coding: utf-8 -*-
"""Pitch tracking: does it find the pitch that is actually there, and does it
say nothing where there is none?"""
from __future__ import annotations

import numpy as np
import pytest

import kudio
from kudio.exceptions import FeatureError

from conftest import SR, make_speech

pytest.importorskip("librosa")


def buzz(freq: float, seconds: float = 1.0, sr: int = SR) -> np.ndarray:
    """A harmonic tone -- something with a fundamental to find."""
    t = np.arange(int(sr * seconds)) / sr
    harmonics = sum(np.sin(2 * np.pi * freq * k * t) / k for k in range(1, 10))
    return (0.3 * harmonics).astype(np.float32)


# --------------------------------------------------------------- it is right

@pytest.mark.parametrize("hz", [90.0, 150.0, 220.0])
def test_a_known_fundamental_is_found(hz):
    track = kudio.f0(buzz(hz), SR)
    assert track.median_hz == pytest.approx(hz, rel=0.02)
    assert track.voiced_ratio > 0.9


def test_a_missing_fundamental_is_still_found():
    """The point of a periodicity estimator over reading the spectrogram: a
    telephone band removes the fundamental and the pitch is unchanged."""
    y = kudio.highpass(buzz(120.0), SR, cutoff=300.0)
    track = kudio.f0(y, SR)
    assert track.median_hz == pytest.approx(120.0, rel=0.03)


def test_silence_is_not_voiced():
    track = kudio.f0(np.zeros(SR, dtype=np.float32), SR)
    assert track.voiced_ratio == 0.0
    assert np.all(np.isnan(track.f0))
    assert np.isnan(track.median_hz)
    assert "no voiced frames" in str(track)


def test_noise_is_mostly_not_voiced():
    """An estimator with no voicing decision answers here too, and the answer
    is a contour drawn through nothing."""
    rng = np.random.default_rng(0)
    y = (0.1 * rng.standard_normal(SR)).astype(np.float32)
    assert kudio.f0(y, SR).voiced_ratio < 0.25


def test_unvoiced_frames_hold_nan_not_a_number():
    """Otherwise plotting the contour draws a line through the silence."""
    track = kudio.f0(make_speech(), SR)
    assert np.all(np.isnan(track.f0[~track.voiced]))
    assert np.all(np.isfinite(track.f0[track.voiced]))


# ------------------------------------------------------------------- shape

def test_frames_line_up_with_the_waveform():
    y = make_speech(seconds=6.0)
    track = kudio.f0(y, SR, hop_length=160)

    assert len(track) == len(track.times) == len(track.voiced)
    assert len(track.voiced_prob) == len(track)
    assert track.times[0] == pytest.approx(0.0, abs=1e-6)
    assert track.times[-1] == pytest.approx(len(y) / SR, abs=0.05)


def test_the_hop_defaults_to_ten_milliseconds():
    track = kudio.f0(buzz(150.0, seconds=1.0), SR)
    assert track.hop_length == SR // 100
    assert len(track) == pytest.approx(100, abs=2)


def test_an_empty_clip_gives_an_empty_track():
    track = kudio.f0(np.zeros(0, dtype=np.float32), SR)
    assert len(track) == 0
    assert track.voiced_ratio == 0.0
    assert track.voiced_segments() == []


# ------------------------------------------------------------------ summary

def test_the_summary_describes_the_two_bursts():
    """`make_speech` puts a 120 Hz burst at 1.0-2.0 s and a 150 Hz one at
    3.5-4.8 s over a noise floor -- so both the range and the segments are
    known in advance."""
    track = kudio.f0(make_speech(), SR)
    lo, hi = track.range_hz()

    assert lo == pytest.approx(120.0, rel=0.05)
    assert hi == pytest.approx(150.0, rel=0.05)
    assert track.semitone_range == pytest.approx(12 * np.log2(150 / 120), abs=0.6)

    summary = track.summary()
    assert set(summary) == {'frames', 'voiced_ratio', 'median_hz', 'low_hz',
                            'high_hz', 'semitone_range'}
    assert summary['median_hz'] == track.median_hz


def test_voiced_segments_land_on_the_bursts():
    segments = kudio.f0(make_speech(), SR).voiced_segments()
    assert len(segments) == 2
    assert segments[0][0] == pytest.approx(1.0, abs=0.1)
    assert segments[0][1] == pytest.approx(2.0, abs=0.1)
    assert segments[1][0] == pytest.approx(3.5, abs=0.1)
    assert segments[1][1] == pytest.approx(4.8, abs=0.1)


def test_short_runs_are_dropped():
    track = kudio.f0(make_speech(), SR)
    assert len(track.voiced_segments(min_seconds=10.0)) == 0


def test_labels_round_trip_through_an_audacity_track(tmp_path):
    """A pitch track that can be opened in an editor beats one that can only
    be printed."""
    track = kudio.f0(make_speech(), SR)
    path = kudio.save_labels(tmp_path / "voiced.txt", track.to_labels())
    reloaded = kudio.load_labels(path)

    assert len(reloaded) == 2
    assert reloaded[0].text == "voiced 1"
    assert reloaded[0].start == pytest.approx(track.to_labels()[0].start, abs=1e-3)


def test_a_semitone_range_needs_two_edges():
    empty = kudio.f0(np.zeros(SR, dtype=np.float32), SR)
    assert np.isnan(empty.semitone_range)


# ------------------------------------------------------------ refused input

def test_a_range_the_wrong_way_round_is_refused():
    with pytest.raises(FeatureError, match="fmin < fmax"):
        kudio.f0(buzz(150.0), SR, fmin=400.0, fmax=65.0)


def test_a_range_above_nyquist_is_refused():
    with pytest.raises(FeatureError, match="Nyquist"):
        kudio.f0(buzz(150.0), 8000, fmax=5000.0)


def test_a_frame_too_short_for_fmin_is_refused_rather_than_silently_empty():
    """Two periods have to fit, and a frame that cannot hold them returns
    "unvoiced everywhere" -- which reads as a property of the recording."""
    with pytest.raises(FeatureError, match="too short for fmin"):
        kudio.f0(buzz(150.0), SR, fmin=50.0, frame_length=256)


def test_a_bad_rate_is_refused():
    with pytest.raises(FeatureError, match="positive rate"):
        kudio.f0(buzz(150.0), 0)
