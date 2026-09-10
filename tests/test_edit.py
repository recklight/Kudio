# -*- coding: utf-8 -*-
"""Fades, reversal, DC removal and the Butterworth filters."""
import numpy as np
import pytest

import kudio
from kudio.exceptions import FeatureError

SR = 16000


# ------------------------------------------------------------------- fades

@pytest.mark.parametrize("shape", ["linear", "cosine", "exponential"])
def test_fade_in_starts_at_silence_and_recovers(sine, shape):
    faded = kudio.fade_in(sine, SR, duration=0.1, shape=shape)
    assert faded[0] == pytest.approx(0.0, abs=1e-6)
    # past the ramp the signal is untouched
    assert np.allclose(faded[int(0.1 * SR) + 1:], sine[int(0.1 * SR) + 1:])


@pytest.mark.parametrize("shape", ["linear", "cosine", "exponential"])
def test_fade_out_ends_at_silence(sine, shape):
    faded = kudio.fade_out(sine, SR, duration=0.1, shape=shape)
    assert faded[-1] == pytest.approx(0.0, abs=1e-6)
    assert np.allclose(faded[:int(0.9 * SR) - 1], sine[:int(0.9 * SR) - 1])


def test_fade_ramps_are_monotonic(sine):
    """A fade that is not monotonic is a tremolo."""
    n = int(0.2 * SR)
    for shape in ("linear", "cosine", "exponential"):
        ramp = kudio.fade_in(np.ones(n, dtype=np.float32), SR, 0.2, shape)
        assert np.all(np.diff(ramp) >= -1e-7), shape
        assert ramp[-1] == pytest.approx(1.0, abs=1e-6)


def test_fade_does_both_ends(sine):
    faded = kudio.fade(sine, SR, 0.05, 0.05)
    assert faded[0] == pytest.approx(0.0, abs=1e-6)
    assert faded[-1] == pytest.approx(0.0, abs=1e-6)


def test_overlapping_fades_are_an_error(sine):
    with pytest.raises(FeatureError, match="overlap"):
        kudio.fade(sine, SR, 0.7, 0.7)


def test_fade_longer_than_the_clip_is_an_error(sine):
    with pytest.raises(FeatureError, match="longer than"):
        kudio.fade_in(sine, SR, duration=5.0)


def test_zero_duration_fade_is_a_no_op(sine):
    assert np.array_equal(kudio.fade_in(sine, SR, 0.0), sine)
    assert np.array_equal(kudio.fade_out(sine, SR, 0.0), sine)


def test_unknown_shape_is_rejected(sine):
    with pytest.raises(FeatureError, match="shape"):
        kudio.fade_in(sine, SR, 0.1, shape="sigmoid")


def test_fade_preserves_dtype_and_shape(sine):
    faded = kudio.fade_in(sine, SR, 0.1)
    assert faded.dtype == sine.dtype
    assert faded.shape == sine.shape


def test_fade_handles_stereo(sine):
    stereo = np.stack([sine, sine * 0.5], axis=1)
    faded = kudio.fade_in(stereo, SR, 0.1)
    assert faded.shape == stereo.shape
    assert np.allclose(faded[0], 0.0, atol=1e-6)


# ------------------------------------------------------- reverse / remove_dc

def test_reverse_round_trips(sine):
    assert np.array_equal(kudio.reverse(kudio.reverse(sine)), sine)


def test_reverse_keeps_stereo_channels_together(sine):
    stereo = np.stack([sine, sine * 0.5], axis=1)
    back = kudio.reverse(stereo)
    assert back.shape == stereo.shape
    assert np.array_equal(back[0], stereo[-1])


def test_remove_dc_centres_the_waveform(sine):
    offset = sine + 0.25
    assert np.mean(offset) == pytest.approx(0.25, abs=1e-3)
    assert np.mean(kudio.remove_dc(offset)) == pytest.approx(0.0, abs=1e-6)


def test_remove_dc_is_per_channel(sine):
    stereo = np.stack([sine + 0.2, sine - 0.3], axis=1)
    fixed = kudio.remove_dc(stereo)
    assert np.allclose(np.mean(fixed, axis=0), 0.0, atol=1e-6)


def test_remove_dc_on_empty_audio():
    assert kudio.remove_dc(np.zeros(0, dtype=np.float32)).size == 0


# ------------------------------------------------------------------ filters

def _band_energy(y, sr, low, high):
    spectrum = np.abs(np.fft.rfft(y)) ** 2
    freqs = np.fft.rfftfreq(len(y), 1.0 / sr)
    return float(spectrum[(freqs >= low) & (freqs < high)].sum())


@pytest.fixture
def two_tones():
    t = np.arange(SR) / SR
    return (0.4 * np.sin(2 * np.pi * 200 * t)
            + 0.4 * np.sin(2 * np.pi * 4000 * t)).astype(np.float32)


def test_highpass_removes_the_low_tone(two_tones):
    out = kudio.highpass(two_tones, SR, cutoff=1000)
    assert _band_energy(out, SR, 100, 400) < 0.01 * _band_energy(two_tones, SR, 100, 400)
    assert _band_energy(out, SR, 3800, 4200) > 0.5 * _band_energy(two_tones, SR, 3800, 4200)


def test_lowpass_removes_the_high_tone(two_tones):
    out = kudio.lowpass(two_tones, SR, cutoff=1000)
    assert _band_energy(out, SR, 3800, 4200) < 0.01 * _band_energy(two_tones, SR, 3800, 4200)
    assert _band_energy(out, SR, 100, 400) > 0.5 * _band_energy(two_tones, SR, 100, 400)


def test_bandpass_keeps_only_the_band(two_tones):
    out = kudio.bandpass(two_tones, SR, low=100, high=400)
    assert _band_energy(out, SR, 3800, 4200) < 0.01 * _band_energy(two_tones, SR, 3800, 4200)


def test_bandstop_removes_only_the_band(two_tones):
    out = kudio.bandstop(two_tones, SR, low=3500, high=4500)
    assert _band_energy(out, SR, 3800, 4200) < 0.01 * _band_energy(two_tones, SR, 3800, 4200)
    assert _band_energy(out, SR, 100, 400) > 0.5 * _band_energy(two_tones, SR, 100, 400)


def test_zero_phase_does_not_shift_the_signal(two_tones):
    """The point of sosfiltfilt: a filtered copy still lines up sample by sample."""
    out = kudio.lowpass(two_tones, SR, cutoff=1000)
    reference = kudio.bandpass(two_tones, SR, low=100, high=400)
    lag = int(np.argmax(np.correlate(out, reference, mode="same")) - len(out) // 2)
    assert abs(lag) <= 1


def test_causal_filtering_is_available(two_tones):
    out = kudio.lowpass(two_tones, SR, cutoff=1000, zero_phase=False)
    assert out.shape == two_tones.shape
    assert not np.allclose(out, kudio.lowpass(two_tones, SR, cutoff=1000))


def test_cutoff_above_nyquist_is_rejected(two_tones):
    with pytest.raises(FeatureError, match="Nyquist"):
        kudio.lowpass(two_tones, SR, cutoff=SR)


def test_band_cutoffs_must_be_ordered(two_tones):
    with pytest.raises(FeatureError, match="low < high"):
        kudio.bandpass(two_tones, SR, low=2000, high=500)


def test_band_filter_needs_two_cutoffs(two_tones):
    with pytest.raises(FeatureError, match="two cutoffs"):
        kudio.band_filter(two_tones, SR, 1000, btype="bandpass")


def test_unknown_btype_is_rejected(two_tones):
    with pytest.raises(FeatureError, match="btype"):
        kudio.band_filter(two_tones, SR, 1000, btype="allpass")


def test_short_selection_still_filters():
    """A 30 ms selection is shorter than sosfiltfilt's default padding."""
    short = np.random.default_rng(0).standard_normal(480).astype(np.float32)
    out = kudio.highpass(short, SR, cutoff=200)
    assert out.shape == short.shape


def test_filtering_empty_audio_returns_empty():
    assert kudio.highpass(np.zeros(0, dtype=np.float32), SR).size == 0


# --------------------------------------------------------------- crossfade

def test_a_crossfade_removes_the_step_at_a_join():
    """A step in the waveform is a click. That is the whole point."""
    a = np.full(1000, 0.5, dtype=np.float32)
    b = np.full(1000, -0.5, dtype=np.float32)
    hard = float(np.max(np.abs(np.diff(np.concatenate([a, b])))))
    soft = float(np.max(np.abs(np.diff(kudio.crossfade(a, b, SR, 0.005)))))
    assert hard == pytest.approx(1.0)
    assert soft < hard / 10.0


def test_the_overlap_is_shared_not_inserted():
    """The result is shorter, and callers tracking positions need to know."""
    a = np.ones(1000, dtype=np.float32)
    b = np.ones(1000, dtype=np.float32)
    n = int(0.005 * SR)
    assert len(kudio.crossfade(a, b, SR, 0.005)) == 2000 - n
    assert len(kudio.splice([a, b, a], SR, 0.005)) == 3000 - 2 * n


def test_zero_duration_is_a_hard_join():
    a = np.ones(10, dtype=np.float32)
    b = np.zeros(10, dtype=np.float32)
    assert np.array_equal(kudio.crossfade(a, b, SR, 0.0),
                          np.concatenate([a, b]))


def test_an_empty_side_is_returned_untouched():
    a = np.ones(10, dtype=np.float32)
    empty = np.zeros(0, dtype=np.float32)
    assert np.array_equal(kudio.crossfade(a, empty, SR), a)
    assert np.array_equal(kudio.crossfade(empty, a, SR), a)
    assert np.array_equal(kudio.splice([empty, a, empty], SR), a)
    assert kudio.splice([], SR).size == 0


def test_a_crossfade_longer_than_a_side_is_clamped():
    a = np.ones(20, dtype=np.float32)
    b = np.zeros(1000, dtype=np.float32)
    out = kudio.crossfade(a, b, SR, 1.0)         # 16000 samples asked for
    assert len(out) == 1000                      # clamped to len(a)


def test_linear_holds_the_level_on_alike_material():
    """Equal-power over-shoots where both sides are the same signal: measured
    at +1.1 dB on a sine, which is why splice defaults to linear."""
    t = np.arange(4000) / SR
    signal = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    a, b = signal[:2000], signal[2000:]

    linear = kudio.crossfade(a, b, SR, 0.005, 'linear')
    equal = kudio.crossfade(a, b, SR, 0.005, 'equal_power')
    assert np.max(np.abs(linear)) <= 0.5 + 1e-6
    assert np.max(np.abs(equal)) > 0.5 + 1e-3


def test_equal_power_holds_the_level_on_unrelated_material():
    """...and the other way round, which is why crossfade defaults to it."""
    rng = np.random.default_rng(0)
    a = (0.3 * rng.standard_normal(4000)).astype(np.float32)
    b = (0.3 * rng.standard_normal(4000)).astype(np.float32)
    n = int(0.02 * SR)
    seam = slice(4000 - n, 4000)

    linear = kudio.crossfade(a, b, SR, 0.02, 'linear')[seam]
    equal = kudio.crossfade(a, b, SR, 0.02, 'equal_power')[seam]
    reference = float(np.sqrt(np.mean(a ** 2)))
    # linear dips in the middle of the seam; equal-power holds the power
    assert np.sqrt(np.mean(equal ** 2)) > np.sqrt(np.mean(linear ** 2))
    assert np.sqrt(np.mean(equal ** 2)) == pytest.approx(reference, rel=0.2)


def test_the_defaults_differ_between_the_two_functions():
    t = np.arange(4000) / SR
    signal = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    a, b = signal[:2000], signal[2000:]
    assert np.allclose(kudio.splice([a, b], SR, 0.005),
                       kudio.crossfade(a, b, SR, 0.005, 'linear'))
    assert np.allclose(kudio.crossfade(a, b, SR, 0.005),
                       kudio.crossfade(a, b, SR, 0.005, 'equal_power'))


def test_a_bad_shape_or_duration_is_rejected():
    a = np.ones(100, dtype=np.float32)
    with pytest.raises(FeatureError, match="shape"):
        kudio.crossfade(a, a, SR, 0.005, 'sigmoid')
    with pytest.raises(FeatureError, match=">= 0"):
        kudio.crossfade(a, a, SR, -0.1)


def test_crossfade_handles_stereo():
    a = np.ones((1000, 2), dtype=np.float32)
    b = np.zeros((1000, 2), dtype=np.float32)
    out = kudio.crossfade(a, b, SR, 0.005)
    assert out.shape == (2000 - int(0.005 * SR), 2)
