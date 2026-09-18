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


# =========================================================== over time

from kudio import (                                            # noqa: E402
    MOMENTARY,
    SHORT_TERM,
    loudness_over_time,
    loudness_range,
    true_peak,
)


def _steps(sr=48000, seconds=12.0, drop_db=12.0):
    """Loud for the first half, `drop_db` quieter for the second."""
    y = _sine(sr=sr, seconds=seconds, amp=0.5)
    half = len(y) // 2
    y[half:] *= 10 ** (-drop_db / 20.0)
    return y


def test_a_steady_tone_measures_the_same_all_the_way_along():
    """A curve of a constant signal has to be constant, and has to agree with
    the integrated figure -- they are the same measurement over different
    spans."""
    curve = loudness_over_time(_sine(seconds=10.0), 48000)

    assert len(curve) > 1
    assert curve.max - curve.min < 0.05
    assert curve.max == pytest.approx(loudness(_sine(seconds=10.0), 48000),
                                      abs=0.1)


def test_a_level_drop_shows_up_where_it_happened():
    curve = loudness_over_time(_steps(), 48000, window=MOMENTARY)

    early = curve.lufs[curve.times < 5.5]
    late = curve.lufs[curve.times > 6.5]
    assert float(np.mean(early) - np.mean(late)) == pytest.approx(12.0, abs=0.3)

    quiet_at, _ = curve.quietest()
    loud_at, _ = curve.loudest()
    assert quiet_at > 6.0 and loud_at < 6.0


def test_times_are_window_centres_not_edges():
    """A curve drawn beside a waveform has to line up with the audio that
    produced it."""
    curve = loudness_over_time(_sine(seconds=10.0), 48000, window=SHORT_TERM)
    assert curve.times[0] == pytest.approx(SHORT_TERM / 2, abs=1e-6)
    assert curve.times[1] - curve.times[0] == pytest.approx(0.1, abs=1e-6)
    assert curve.times[-1] <= 10.0 - SHORT_TERM / 2 + 1e-6


def _alternating(sr=16000, seconds=12.0, dip_db=12.0, period=2.0):
    """One second loud, one second `dip_db` down, over and over.

    `_speech_like` modulates at 3 Hz, which a 400 ms window averages straight
    over -- the right fixture for gating and the wrong one for showing that
    the two windows resolve different things.
    """
    t = np.arange(int(sr * seconds), dtype=np.float64) / sr
    gate = np.where((t % period) < period / 2, 1.0, 10 ** (-dip_db / 20.0))
    return 0.4 * np.sin(2 * np.pi * 220 * t) * gate


def test_momentary_resolves_what_short_term_smooths():
    """400 ms follows syllables; 3 s follows passages. Given something that
    changes every second, the short window has to swing further -- or the two
    are not measuring what their names claim."""
    y = _alternating(seconds=12.0, dip_db=12.0)
    fast = loudness_over_time(y, 16000, window=MOMENTARY)
    slow = loudness_over_time(y, 16000, window=SHORT_TERM)

    assert (fast.max - fast.min) == pytest.approx(12.0, abs=1.0)
    assert (slow.max - slow.min) < 3.0


def test_the_hop_sets_how_many_measurements_come_back():
    y = _sine(seconds=10.0)
    dense = loudness_over_time(y, 48000, window=MOMENTARY, hop=0.1)
    sparse = loudness_over_time(y, 48000, window=MOMENTARY, hop=0.5)
    assert len(dense) == pytest.approx(len(sparse) * 5, rel=0.05)


def test_the_curve_is_not_gated():
    """Gating belongs to the integrated figure. A curve with holes punched in
    it where the gate fired would describe a recording that stops existing
    between words."""
    sr = 48000
    y = _sine(sr=sr, seconds=12.0, amp=0.5)
    y[len(y) // 3: 2 * len(y) // 3] = 0.0          # a long true silence

    curve = loudness_over_time(y, sr, window=MOMENTARY)
    assert np.isneginf(curve.lufs).any(), "the silence must still be reported"
    assert len(curve) == pytest.approx((12.0 - MOMENTARY) / 0.1, abs=2)


def test_silence_is_a_silent_curve_rather_than_an_error():
    curve = loudness_over_time(np.zeros(48000 * 5), 48000)
    assert len(curve) > 0
    assert curve.finite.size == 0
    assert curve.max == float("-inf")
    assert np.isnan(curve.loudest()[0])
    assert "silent" in str(curve)
    assert curve.summary()["windows"] == len(curve)


def test_spans_find_the_quiet_stretch():
    curve = loudness_over_time(_steps(drop_db=20.0), 48000, window=MOMENTARY)
    threshold = (curve.max + curve.min) / 2

    below = curve.spans_below(threshold, min_seconds=0.5)
    assert len(below) == 1
    assert below[0][0] == pytest.approx(6.0, abs=0.4)
    assert below[0][1] == pytest.approx(12.0, abs=0.4)

    above = curve.spans_above(threshold, min_seconds=0.5)
    assert len(above) == 1
    assert above[0][0] < 1.0


def test_a_window_longer_than_the_clip_says_how_to_fix_it():
    with pytest.raises(FeatureError, match="shorter window"):
        loudness_over_time(_sine(seconds=1.0), 48000, window=SHORT_TERM)


@pytest.mark.parametrize("kwargs", [{"window": 0.0}, {"hop": 0.0},
                                    {"window": -1.0}])
def test_impossible_curve_settings_are_refused(kwargs):
    with pytest.raises(FeatureError):
        loudness_over_time(_sine(seconds=10.0), 48000, **kwargs)


# =========================================================== loudness range

def test_a_steady_tone_has_no_range():
    assert loudness_range(_sine(seconds=10.0), 48000) == pytest.approx(0.0, abs=0.3)


def test_a_clip_that_changes_level_has_the_range_it_changed_by():
    """Half loud, half 12 dB down, held long enough for the 3 s window to
    settle on each."""
    assert loudness_range(_steps(seconds=30.0, drop_db=12.0), 48000) == \
        pytest.approx(12.0, abs=1.0)


def test_silence_has_no_range_rather_than_an_undefined_one():
    assert loudness_range(np.zeros(48000 * 5), 48000) == 0.0


def test_a_stray_bang_does_not_set_the_range():
    """The percentiles are the whole reason it is 10-to-95 and not min-to-max."""
    sr = 48000
    y = _sine(sr=sr, seconds=20.0, amp=0.3)
    plain = loudness_range(y, sr)
    y[int(9.5 * sr):int(9.52 * sr)] = 1.0          # one 20 ms transient
    assert loudness_range(y, sr) == pytest.approx(plain, abs=1.0)


# ================================================================ true peak

def test_a_full_scale_sine_reads_about_zero_dbtp():
    assert true_peak(_sine(), 48000) == pytest.approx(0.0, abs=0.2)


def test_the_peak_between_the_samples_is_found():
    """The demonstration case: a sine at a quarter of the sample rate, phased
    so every sample lands at 0.707. The samples say -3 dBFS; the waveform they
    describe reaches full scale, and that is what a converter meets."""
    sr = 48000
    n = np.arange(sr)
    y = np.sin(2 * np.pi * (sr / 4) * n / sr + np.pi / 4)

    assert 20 * np.log10(np.max(np.abs(y))) == pytest.approx(-3.01, abs=0.01)
    assert true_peak(y, sr) == pytest.approx(0.0, abs=0.3)


def test_oversampling_can_never_report_less_than_the_samples_do():
    for signal in (_sine(), _speech_like(), _steps()):
        sr = 48000 if len(signal) != 64000 else 16000
        assert true_peak(signal, sr) >= 20 * np.log10(np.max(np.abs(signal))) - 1e-9


def test_no_oversampling_is_the_plain_sample_peak():
    y = _speech_like()
    assert true_peak(y, 16000, oversample=1) == \
        pytest.approx(20 * np.log10(np.max(np.abs(y))), abs=1e-9)


def test_silence_has_no_peak():
    assert true_peak(np.zeros(4800), 48000) == float("-inf")
    assert true_peak(np.zeros(0), 48000) == float("-inf")


def test_chunking_does_not_change_the_answer(monkeypatch):
    """Long files are oversampled in pieces with real audio either side; the
    seams must not be visible in the result."""
    import sys

    # by name out of sys.modules: `kudio.core.loudness` the module is shadowed
    # by `kudio.core.loudness` the function, and getattr finds the function
    module = sys.modules["kudio.core.loudness"]

    y = _speech_like(seconds=8.0)
    whole = true_peak(y, 16000)
    monkeypatch.setattr(module, "_CHUNK", 997)       # deliberately awkward
    assert true_peak(y, 16000) == pytest.approx(whole, abs=1e-9)


def test_stereo_reports_the_louder_channel():
    quiet = _sine(amp=0.25)
    loud = _sine(amp=1.0)
    stereo = np.stack([quiet, loud], axis=1)
    assert true_peak(stereo, 48000) == pytest.approx(true_peak(loud, 48000),
                                                     abs=1e-9)


def test_a_bad_rate_is_refused():
    with pytest.raises(FeatureError, match="positive sample rate"):
        true_peak(_sine(), 0)


def test_the_range_is_readable_off_a_curve_you_already_have():
    """Measuring the curve is the expensive half; asking it for its range
    afterwards has to be the same answer and nearly free."""
    y = _steps(seconds=30.0, drop_db=12.0)
    curve = loudness_over_time(y, 48000, window=SHORT_TERM)
    assert curve.range_lu == pytest.approx(loudness_range(y, 48000), abs=1e-9)
    assert curve.summary()["range_lu"] == curve.range_lu


def test_a_momentary_curve_still_answers_but_narrower():
    """Same arithmetic, different question -- how much the syllables vary,
    not the loudness range."""
    y = _alternating(seconds=20.0, dip_db=12.0)
    fast = loudness_over_time(y, 16000, window=MOMENTARY).range_lu
    slow = loudness_over_time(y, 16000, window=SHORT_TERM).range_lu
    assert fast > slow
