# -*- coding: utf-8 -*-
"""audio_report: what can be measured with no clean reference to compare to."""
import numpy as np
import pytest

from kudio import audio_report, resample
from kudio.exceptions import FeatureError

SR = 16000


def _speech(sr=SR, seconds=3.0, noise=0.02, seed=0):
    """Harmonic stack under a syllable envelope, with a little broadband bed."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(sr * seconds)) / sr
    y = sum(0.4 / k * np.sin(2 * np.pi * 180 * k * t) for k in range(1, 12))
    y *= 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)
    y = y / np.max(np.abs(y)) * 0.6 + noise * rng.standard_normal(len(t))
    return y.astype(np.float32)


def test_a_healthy_clip_reports_nothing_to_fix():
    report = audio_report(_speech(), SR)
    assert report.problems() == []
    assert report.sr == SR
    assert report.duration == pytest.approx(3.0)
    assert -10 < report.peak_dbfs < 0
    assert report.clipped_samples == 0
    assert abs(report.dc_offset) < 1e-3


def test_clipping_is_counted_only_as_runs():
    """One sample touching full scale is luck; a run of them is clipping."""
    y = _speech()
    y[100] = 1.0                                    # a lone spike
    assert audio_report(y, SR).clipped_samples == 0

    y[500:520] = 1.0                                # a real flat top
    report = audio_report(y, SR)
    assert report.clipped_samples == 20
    assert report.clipped_ratio == pytest.approx(20 / len(y))
    assert any("clipped" in p for p in report.problems())


def test_a_signal_past_full_scale_is_not_the_same_as_a_clipped_one():
    """Amplified float audio still traces the waveform -- nothing is lost yet.
    It is heading for trouble on the way to an integer format, which is a
    different, weaker warning than "you already destroyed this"."""
    report = audio_report(_speech() * 2.0, SR)

    assert report.peak_dbfs > 0
    assert report.clipped_samples == 0, "loud is not clipped"
    problems = report.problems()
    assert any("past full scale" in p for p in problems)
    assert not any("clipped" in p for p in problems)

    # ...whereas actually flattening it is
    flattened = audio_report(np.clip(_speech() * 2.0, -1.0, 1.0), SR)
    assert flattened.clipped_samples > 0
    assert any("clipped" in p for p in flattened.problems())


def test_a_quiet_clip_is_flagged_without_being_called_broken():
    report = audio_report(_speech() * 0.01, SR)
    assert report.peak_dbfs < -30
    assert any("very quiet" in p for p in report.problems())
    assert not any("clipped" in p for p in report.problems())


def test_dc_offset_is_measured_and_reported():
    report = audio_report(_speech() * 0.5 + 0.05, SR)
    assert report.dc_offset == pytest.approx(0.05, abs=0.005)
    assert any("DC offset" in p for p in report.problems())


@pytest.mark.parametrize("source_sr,file_sr", [(8000, 16000), (16000, 44100)])
def test_upsampled_audio_is_caught_and_its_real_rate_named(source_sr, file_sr):
    """The failure that quietly wastes half a dataset: a file labelled 16 kHz
    that is really 8 kHz content with nothing above 4 kHz."""
    native = _speech(source_sr)
    upsampled = resample(native, source_sr, file_sr)

    report = audio_report(upsampled, file_sr)
    assert report.band_limited is True
    assert report.bandwidth_hz == pytest.approx(source_sr / 2, rel=0.1)
    assert any(f"upsampled from {source_sr} Hz" in p for p in report.problems())


@pytest.mark.parametrize("sr", [8000, 16000, 44100])
def test_audio_recorded_at_its_stated_rate_is_not_flagged(sr):
    report = audio_report(_speech(sr), sr)
    assert report.band_limited is False
    assert report.bandwidth_hz > 0.9 * (sr / 2)


def test_the_snr_estimate_tracks_the_noise_it_is_given():
    clean = audio_report(_speech(noise=0.001), SR).estimated_snr_db
    noisy = audio_report(_speech(noise=0.2), SR).estimated_snr_db
    assert clean > noisy + 10
    assert any("estimated SNR" in p for p in audio_report(
        _speech(noise=0.4), SR).problems())


def test_silence_is_measured_as_a_fraction_of_the_clip():
    y = np.concatenate([_speech(seconds=1.0), np.zeros(4 * SR, dtype=np.float32)])
    report = audio_report(y, SR)
    assert report.silence_ratio == pytest.approx(0.8, abs=0.05)
    assert any("silence" in p for p in report.problems())


def test_digital_silence_says_so_and_stops_there():
    report = audio_report(np.zeros(SR, dtype=np.float32), SR)
    assert report.peak_dbfs == float("-inf")
    assert np.isnan(report.lufs) or report.lufs == float("-inf")
    assert report.problems() == ["the clip is digital silence"]


def test_a_clip_too_short_for_bs1770_still_reports_everything_else():
    report = audio_report(_speech(seconds=0.2), SR)
    assert np.isnan(report.lufs)
    assert np.isfinite(report.peak_dbfs)
    assert report.bandwidth_hz > 0


def test_bad_input_is_rejected_clearly():
    with pytest.raises(FeatureError, match="sample rate"):
        audio_report(_speech(), 0)
    with pytest.raises(FeatureError, match="mono"):
        audio_report(np.zeros((1000, 2)), SR)
    with pytest.raises(FeatureError, match="empty"):
        audio_report(np.zeros(0), SR)
