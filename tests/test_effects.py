# -*- coding: utf-8 -*-
import numpy as np

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
