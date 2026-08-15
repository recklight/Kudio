# -*- coding: utf-8 -*-
"""BS.1770 loudness: calibration, gating, rate independence, normalisation."""
import numpy as np
import pytest

from kudio import loudness, match_loudness, normalize, normalize_lufs
from kudio.exceptions import FeatureError


def _sine(freq=1000.0, sr=48000, seconds=4.0, amp=1.0):
    t = np.arange(int(sr * seconds), dtype=np.float64) / sr
    return amp * np.sin(2 * np.pi * freq * t)


def _speech_like(sr=16000, seconds=4.0):
    """An amplitude-modulated tone: loud syllables with quiet gaps."""
    t = np.arange(int(sr * seconds), dtype=np.float64) / sr
    return 0.4 * np.sin(2 * np.pi * 220 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t))


def test_full_scale_sine_reads_the_calibration_value():
    """BS.1770's own calibration point: a 0 dBFS 1 kHz sine is -3.01 LUFS."""
    assert loudness(_sine(), 48000) == pytest.approx(-3.01, abs=0.05)


def test_halving_the_amplitude_costs_six_db():
    quiet = loudness(_sine(amp=0.5), 48000)
    loud = loudness(_sine(amp=1.0), 48000)
    assert loud - quiet == pytest.approx(6.02, abs=0.01)


@pytest.mark.parametrize("sr", [8000, 16000, 22050, 44100, 48000])
def test_the_same_signal_measures_the_same_at_any_rate(sr):
    """The K-weighting curve is defined in Hz, so the rate must not matter.

    The standard only publishes its filters at 48 kHz; if the adaptation to
    other rates were wrong, this is where it would show.
    """
    assert loudness(_sine(sr=sr), sr) == pytest.approx(-3.01, abs=0.3)


def test_silence_between_words_is_gated_away():
    """A pause must not make a clip measure quieter -- that is the whole point
    of the gate, and the reason plain RMS is not usable for this."""
    sr = 16000
    speech = _speech_like(sr, seconds=4.0)
    padded = np.concatenate([speech, np.zeros(10 * sr)])

    assert loudness(padded, sr) == pytest.approx(loudness(speech, sr), abs=0.3)

    # the same signal by plain RMS: 4 s of content in 14 s of file, so
    # 10*log10(14/4) = 5.4 dB of pure bookkeeping
    rms_before = 20 * np.log10(np.sqrt(np.mean(speech ** 2)))
    rms_after = 20 * np.log10(np.sqrt(np.mean(padded ** 2)))
    assert rms_before - rms_after == pytest.approx(5.44, abs=0.05)


def test_a_stray_transient_fools_peak_normalisation_but_not_loudness():
    """The case that motivates the whole module: one click sets the peak, so
    peak normalisation leaves the clip inaudible while loudness sees through it.
    """
    sr = 16000
    quiet = _speech_like(sr) * 0.1
    clicked = quiet.copy()
    clicked[sr] = 1.0                       # a single full-scale sample

    peak_normalised = normalize(clicked, peak=0.99)
    loudness_normalised = normalize_lufs(clicked, sr, lufs=-23.0)

    assert loudness(peak_normalised, sr) < -35.0
    assert loudness(loudness_normalised, sr) == pytest.approx(-23.0, abs=0.1)


def test_digital_silence_is_minus_infinity_and_passes_through_untouched():
    sr = 16000
    silence = np.zeros(2 * sr)
    assert loudness(silence, sr) == float("-inf")
    assert np.array_equal(normalize_lufs(silence, sr), silence)
    assert np.array_equal(match_loudness(silence, _speech_like(sr), sr), silence)
    assert np.array_equal(match_loudness(_speech_like(sr), silence, sr),
                          _speech_like(sr))


def test_normalize_lufs_hits_its_target():
    sr = 16000
    y = _speech_like(sr)
    for target in (-16.0, -23.0, -31.0):
        assert loudness(normalize_lufs(y, sr, target), sr) == pytest.approx(
            target, abs=0.05)


def test_normalize_lufs_warns_instead_of_silently_clipping(caplog):
    sr = 16000
    y = _speech_like(sr) * 0.02
    with caplog.at_level("WARNING"):
        out = normalize_lufs(y, sr, lufs=0.0)
    assert np.max(np.abs(out)) > 1.0, "the gain really does overshoot"
    assert "clip" in caplog.text


def test_match_loudness_equalises_two_versions_of_a_clip():
    """The A/B primitive: after matching, the level difference is gone but the
    signal is otherwise untouched."""
    sr = 16000
    reference = _speech_like(sr)
    louder = reference * 4.0

    matched = match_loudness(louder, reference, sr)
    assert loudness(matched, sr) == pytest.approx(loudness(reference, sr), abs=0.01)
    assert np.allclose(matched, reference, atol=1e-9)


def test_match_loudness_across_two_sample_rates():
    """Loudness is a property of the sound, so a resampled copy still matches."""
    import kudio
    reference = _speech_like(16000)
    resampled = kudio.resample(reference, 16000, 8000) * 3.0

    matched = match_loudness(resampled, reference, 8000, reference_sr=16000)
    assert loudness(matched, 8000) == pytest.approx(
        loudness(reference, 16000), abs=0.2)


def test_stereo_is_measured_channels_last():
    sr = 16000
    mono = _speech_like(sr)
    stereo = np.stack([mono, mono], axis=1)
    # two identical channels at weight 1.0 each is twice the power: +3.01 dB
    assert loudness(stereo, sr) - loudness(mono, sr) == pytest.approx(3.01, abs=0.01)


def test_a_clip_shorter_than_a_block_says_so_and_offers_the_way_out():
    sr = 16000
    short = _speech_like(sr, seconds=0.2)

    with pytest.raises(FeatureError) as excinfo:
        loudness(short, sr)
    assert "block_size" in str(excinfo.value)

    assert np.isfinite(loudness(short, sr, block_size=0.1))


def test_a_bad_rate_or_shape_is_rejected():
    with pytest.raises(FeatureError):
        loudness(_speech_like(), 0)
    with pytest.raises(FeatureError):
        loudness(np.zeros((4, 100, 2)), 16000)
    with pytest.raises(FeatureError):
        loudness(np.zeros((16000, 9)), 16000)      # more channels than weights


@pytest.mark.parametrize("sr", [8000, 16000, 44100, 48000])
def test_agrees_with_pyloudnorm_where_it_is_installed(sr):
    """Cross-check against an independent BS.1770 implementation.

    Not a dependency -- kudio implements this itself -- but when pyloudnorm
    happens to be around, the two must not disagree. A hand-rolled filter
    adaptation is exactly the kind of thing that is quietly wrong.
    """
    pyln = pytest.importorskip("pyloudnorm")
    y = _speech_like(sr) + 0.05 * np.sin(2 * np.pi * 3000 * np.arange(4 * sr) / sr)
    assert loudness(y, sr) == pytest.approx(
        pyln.Meter(sr).integrated_loudness(y), abs=0.1)
