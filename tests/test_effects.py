# -*- coding: utf-8 -*-
import numpy as np
import pytest

from kudio import (
    add_noise_snr,
    gain,
    pitch_shift,
    random_gain,
    reverb,
    spec_augment,
    split_on_silence,
    time_stretch,
    trim_silence,
    waveform_to_spectrogram,
)


def _tone(freq=440, sr=16000, seconds=1.0):
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_trim_silence():
    sr = 16000
    y = np.concatenate([np.zeros(sr // 2, np.float32), _tone(seconds=0.5),
                        np.zeros(sr // 2, np.float32)])
    trimmed, (start, end) = trim_silence(y, top_db=20)
    assert start > 0 and end < len(y)
    assert len(trimmed) < len(y)


def test_split_on_silence():
    sr = 16000
    gap = np.zeros(sr // 2, np.float32)
    y = np.concatenate([_tone(440, seconds=0.3), gap, _tone(880, seconds=0.3)])
    segments = split_on_silence(y, top_db=20)
    assert len(segments) >= 2


def test_time_stretch_changes_length():
    y = _tone()
    faster = time_stretch(y, rate=2.0)
    assert len(faster) < len(y)


def test_pitch_shift_keeps_length():
    y = _tone()
    shifted = pitch_shift(y, sr=16000, n_steps=2)
    assert abs(len(shifted) - len(y)) <= 1


def test_gain_and_random_gain():
    y = _tone()
    louder = gain(y, db=6.0)
    assert np.max(np.abs(louder)) > np.max(np.abs(y))
    rg = random_gain(y, db_range=(-3, 3), seed=1)
    assert rg.shape == y.shape


def test_add_noise_snr_reproducible():
    y = _tone()
    noise = np.random.default_rng(0).standard_normal(8000).astype(np.float32)
    a = add_noise_snr(y, noise, snr_db=5, seed=42)
    b = add_noise_snr(y, noise, snr_db=5, seed=42)
    np.testing.assert_array_equal(a, b)
    assert a.shape == y.shape


def test_reverb():
    y = _tone(seconds=0.5)
    wet = reverb(y, sr=16000)
    assert wet.shape == y.shape and np.all(np.isfinite(wet))


def test_spec_augment_masks():
    y = _tone()
    spec = waveform_to_spectrogram(y)          # (freq, time)
    masked = spec_augment(spec, freq_mask_width=5, time_mask_width=5, seed=0)
    assert masked.shape == spec.shape
    # at least some values were zeroed
    assert np.count_nonzero(masked == 0.0) >= np.count_nonzero(spec == 0.0)


# -- normalisation --------------------------------------------------------------

def test_normalize_scales_the_peak():
    from kudio import normalize

    y = np.array([0.1, -0.2, 0.05], dtype=np.float32)
    out = normalize(y, peak=0.99)

    assert np.isclose(np.max(np.abs(out)), 0.99)
    # shape of the signal is untouched, only its scale
    assert np.allclose(out / np.max(np.abs(out)), y / np.max(np.abs(y)))


def test_normalize_leaves_silence_alone():
    from kudio import normalize

    silence = np.zeros(16, dtype=np.float32)
    out = normalize(silence)

    assert np.allclose(out, 0.0)
    assert out is not silence          # still a copy, like every other effect


def test_normalize_handles_an_empty_array():
    from kudio import normalize

    assert normalize(np.array([], dtype=np.float32)).size == 0


def test_normalize_rejects_a_bad_peak():
    from kudio import normalize

    with pytest.raises(ValueError, match="peak must be > 0"):
        normalize(np.ones(4, dtype=np.float32), peak=0.0)


def test_normalize_db_matches_normalize():
    from kudio import normalize, normalize_db

    y = np.array([0.3, -0.6, 0.1], dtype=np.float32)
    assert np.allclose(normalize_db(y, -1.0), normalize(y, 10 ** (-1 / 20)))
    assert np.isclose(np.max(np.abs(normalize_db(y, 0.0))), 1.0)


def test_normalize_db_rejects_positive_dbfs():
    from kudio import normalize_db

    with pytest.raises(ValueError, match="dbfs must be <= 0"):
        normalize_db(np.ones(4, dtype=np.float32), dbfs=3.0)
