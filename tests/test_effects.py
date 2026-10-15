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
    spec = waveform_to_spectrogram(y)          # (frames, bins), like every kudio spectrogram
    masked = spec_augment(spec, n_freq_masks=0, time_mask_width=5, seed=0)
    assert masked.shape == spec.shape
    rows = np.flatnonzero(np.any(masked != spec, axis=1))
    assert rows.size and np.all(masked[rows] == masked[rows][0, 0]), "a time mask blanks whole frames"
    masked = spec_augment(spec, n_time_masks=0, freq_mask_width=5, seed=0)
    cols = np.flatnonzero(np.any(masked != spec, axis=0))
    assert cols.size and cols.max() - cols.min() < 5, "a frequency mask blanks a few bins"


def test_spec_augment_masks_with_the_mean_by_default():
    """0 is a power of 1 on log10 power: louder than most of what it hides."""
    spec = waveform_to_spectrogram(_tone())
    masked = spec_augment(spec, n_freq_masks=0, time_mask_width=5, seed=0)
    hidden = masked[np.any(masked != spec, axis=1)]
    assert np.allclose(hidden, np.mean(spec))


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


# -- channels last and integer audio -------------------------------------------

SR_ = 16000


def _stereo():
    t = np.arange(SR_) / SR_
    left = np.sin(2 * np.pi * 220 * t) * np.r_[np.zeros(4000), np.ones(8000), np.zeros(4000)]
    return np.stack([0.5 * left, 0.25 * left], axis=1).astype(np.float32)


def test_filters_work_along_the_frames_of_a_stereo_clip():
    import kudio
    st = _stereo()
    out = kudio.highpass(st, SR_, cutoff=1000)
    assert out.shape == st.shape
    assert np.allclose(out[:, 1], kudio.highpass(st[:, 1], SR_, cutoff=1000), atol=1e-6)
    causal = kudio.highpass(st, SR_, cutoff=1000, zero_phase=False)
    assert np.allclose(causal[:, 0], kudio.highpass(st[:, 0], SR_, cutoff=1000, zero_phase=False), atol=1e-6)


def test_trim_and_split_work_along_the_frames_and_copy():
    import kudio
    st = _stereo()
    clip, (a, b) = kudio.trim_silence(st, top_db=30)
    # librosa trims in 2048-sample frames, so the edges land within one of the tone
    assert clip.shape[1] == 2 and 2000 < a <= 4000 and 12000 <= b < 14100
    clip[:] = 0
    assert np.any(st != 0), "the trimmed clip is a copy"
    pieces = kudio.split_on_silence(st, top_db=30)
    assert len(pieces) == 1 and pieces[0].shape[1] == 2


def test_time_stretch_and_pitch_shift_keep_the_channels():
    import kudio
    st = _stereo()
    assert kudio.time_stretch(st, 2.0).shape == (8000, 2)
    assert kudio.pitch_shift(st, SR_, 2).shape == st.shape
    assert kudio.reverb(st, SR_).shape == st.shape


def test_integer_audio_is_rounded_and_clipped_not_wrapped():
    import kudio
    y = np.array([20000, -20000, 100], dtype=np.int16)
    assert kudio.gain(y, 6.0).tolist() == [32767, -32768, 200]
    loud = kudio.normalize(y, peak=0.5)
    assert loud.dtype == np.int16 and np.max(np.abs(loud)) == 16384


def test_noise_with_nothing_in_it_is_refused():
    import kudio
    from kudio.exceptions import FeatureError
    y = _tone()
    for noise in (np.zeros(1000), np.full(1000, 0.3, dtype=np.float32)):
        with pytest.raises(FeatureError, match="no variation"):
            kudio.add_noise_snr(y, noise, snr_db=5, seed=0)


def test_splice_of_one_part_is_a_copy():
    import kudio
    y = _tone()
    assert kudio.splice([y], 16000) is not y
