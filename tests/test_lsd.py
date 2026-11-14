# -*- coding: utf-8 -*-
"""Log-spectral distance: URGENT 2025's numbers by default, the
bandwidth-extension setting on request, and what the formula does with
level, delay and digital silence."""
from __future__ import annotations

import math
import re
import warnings

import numpy as np
import pytest

from kudio.core import evaluator
from kudio.core.evaluator import lsd

from conftest import make_speech


def _talk(sr: int, seconds: float = 3.0, seed: int = 0) -> np.ndarray:
    """Harmonic syllables, two a second, on an f0 wandering between 90 and
    150 Hz, over a faint noise floor that keeps every frame from being
    digital silence."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(sr * seconds)) / sr
    f0 = 120.0 + 30.0 * np.sin(2 * np.pi * 0.7 * t)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    voiced = sum(np.sin(k * phase) / k for k in range(1, 40) if k * 150 < sr / 2)
    envelope = np.clip(np.sin(2 * np.pi * 2.0 * t), 0.0, None) ** 2
    return 0.3 * voiced * envelope + 0.003 * rng.standard_normal(len(t))


def _noisy(ref: np.ndarray, snr_db: float, seed: int) -> np.ndarray:
    noise = np.random.default_rng(seed).standard_normal(len(ref))
    return ref + noise * math.sqrt(np.mean(ref ** 2) / 10 ** (snr_db / 10))


def _lowpassed(ref: np.ndarray, sr: int, cutoff: float) -> np.ndarray:
    from scipy.signal import butter, sosfiltfilt
    return sosfiltfilt(butter(8, cutoff, fs=sr, output='sos'), ref)


def _quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return fn(*args, **kwargs)


def _stft_magnitude(librosa, y, n_fft, hop):
    return np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop)).T


def _urgent_lsd(librosa, ref, deg, sr, eps=1e-8):
    """URGENT 2025's lsd_metric, as its scoring script states it: a
    least-squares gain on deg, 32 ms frames every 16 ms on librosa's STFT,
    and the natural log of the power ratio."""
    deg = deg * np.sum(ref * deg) / (np.sum(deg ** 2) + eps)
    n_fft, hop = int(sr * 0.032), int(sr * 0.016)
    r = _stft_magnitude(librosa, ref, n_fft, hop)
    d = _stft_magnitude(librosa, deg, n_fft, hop)
    return float(np.mean(np.sqrt(np.mean(np.log(r ** 2 / (d + eps) ** 2 + eps) ** 2, axis=1))))


# ------------------------------------------------- against the references

# URGENT 2025's lsd_metric (evaluation_metrics/calculate_intrusive_se_metrics.py
# at 93e16aa), its body run as written on float64 copies of these fixtures with
# librosa 0.11.0, numpy 2.2.6 and scipy 1.15.2. 22050 Hz gives an odd frame,
# int(0.032 * 22050) = 705.
_URGENT_2025 = [
    (8000, 'noisy', 4.12792757488937),
    (16000, 'noisy', 4.472351638883378),
    (22050, 'noisy', 4.654353236598868),
    (44100, 'noisy', 4.919388486238227),
    (48000, 'noisy', 4.9373865419269425),
    (16000, 'lowpassed', 14.624145480159434),
    (48000, 'lowpassed', 16.486048293188183),
]


def _deg(kind: str, ref: np.ndarray, sr: int) -> np.ndarray:
    return _noisy(ref, 10.0, sr + 1) if kind == 'noisy' else _lowpassed(ref, sr, sr / 4)


@pytest.mark.parametrize("sr, kind, golden", _URGENT_2025)
def test_the_defaults_give_urgent_2025s_number(sr, kind, golden):
    ref = _talk(sr, seed=sr)
    assert _quiet(lsd, ref, _deg(kind, ref, sr), sr) == pytest.approx(golden, rel=1e-7)


@pytest.mark.parametrize("sr", [8000, 16000, 22050, 44100, 48000])
def test_the_defaults_match_urgents_formula_on_librosas_stft(sr):
    librosa = pytest.importorskip("librosa")
    ref = _talk(sr, seed=sr)
    for deg in (_noisy(ref, 10.0, sr + 1), _lowpassed(ref, sr, sr / 4),
                0.5 * ref + 0.01 * np.random.default_rng(sr).standard_normal(len(ref))):
        assert _quiet(lsd, ref, deg, sr) == pytest.approx(_urgent_lsd(librosa, ref, deg, sr),
                                                          rel=1e-9)


def test_a_pair_one_frame_long_is_framed_the_way_librosa_frames_it():
    """librosa pads n_fft // 2 zeros at each end, so at 16 kHz, where a frame
    is 512 samples, a pair of 512 to 767 samples makes three frames and one
    of 768 makes four."""
    librosa = pytest.importorskip("librosa")
    ref = _talk(16000, seed=3)
    deg = _noisy(ref, 10.0, 4)
    for n in (512, 513, 767, 768, 4800):
        want = _urgent_lsd(librosa, ref[:n], deg[:n], 16000)
        assert _quiet(lsd, ref[:n], deg[:n], 16000) == pytest.approx(want, rel=1e-9)


@pytest.mark.parametrize("n", [1, 10, 511])
def test_a_pair_shorter_than_one_frame_is_nan_with_a_warning(n):
    """At 16 kHz a frame is 512 samples, and a shorter pair has no full frame
    to measure."""
    ref = _talk(16000, seed=3)
    deg = _noisy(ref, 10.0, 4)
    with pytest.warns(UserWarning, match="at least one full frame, 512 samples"):
        assert math.isnan(lsd(ref[:n], deg[:n], 16000))


# ssr_eval's AudioMetrics.lsd (haoheliu/ssr_eval at 7129ee8) at 44.1 kHz, its
# torch formula restated in numpy on librosa 0.11.0's STFT: n_fft
# int(2048 / (44100 / 44100)) = 2048, hop int(44100 / 100) = 441, base-10 log,
# EPS 1e-12, and no gain on the estimate.
_SSR_EVAL_44K = [('noisy', 2.215475495172847), ('lowpassed', 11.339808821938234)]


@pytest.mark.parametrize("kind, golden", _SSR_EVAL_44K)
def test_the_bandwidth_extension_setting_gives_ssr_evals_number(kind, golden):
    ref = _talk(44100, seed=7)
    deg = _noisy(ref, 10.0, 8) if kind == 'noisy' else _lowpassed(ref, 44100, 4000)
    got = _quiet(lsd, ref, deg, 44100, n_fft=2048, hop_length=441, scale='log10',
                 match_gain=False, eps=1e-12)
    assert got == pytest.approx(golden, rel=1e-7)


def test_the_bandwidth_extension_setting_matches_ssr_evals_formula_on_librosas_stft():
    librosa = pytest.importorskip("librosa")
    ref = _talk(44100, seed=7)
    for deg in (_noisy(ref, 10.0, 8), _lowpassed(ref, 44100, 4000)):
        t = _stft_magnitude(librosa, ref, 2048, 441)
        e = _stft_magnitude(librosa, deg, 2048, 441)
        want = float(np.mean(np.sqrt(np.mean(np.log10(t ** 2 / (e + 1e-12) ** 2 + 1e-12) ** 2,
                                             axis=1))))
        got = lsd(ref, deg, 44100, n_fft=2048, hop_length=441, scale='log10',
                  match_gain=False, eps=1e-12)
        assert got == pytest.approx(want, rel=1e-9)


# ----------------------------------------------------- what it measures

def test_a_signal_is_no_distance_from_itself():
    """Only eps keeps it off zero: 1.2e-6 here."""
    ref = _talk(16000)
    assert _quiet(lsd, ref, ref, 16000) == pytest.approx(0.0, abs=2e-6)


def test_a_level_difference_counts_only_when_the_gain_is_not_matched():
    """Doubling deg is a power ratio of 4 in every bin: 2 ln 2 = 1.386 in
    natural log, 0.602 in base 10, and 20 log10 2 = 6.02 dB. With the gain
    matched it is the 1.2e-6 of a signal against itself."""
    ref = _talk(16000)
    assert _quiet(lsd, ref, 2.0 * ref, 16000) == pytest.approx(0.0, abs=2e-6)
    unmatched = dict(sr=16000, match_gain=False)
    assert lsd(ref, 2.0 * ref, **unmatched) == pytest.approx(2 * math.log(2), rel=1e-6)
    assert lsd(ref, 2.0 * ref, scale='log10', **unmatched) == pytest.approx(
        2 * math.log10(2), rel=1e-6)
    assert lsd(ref, 2.0 * ref, scale='db', **unmatched) == pytest.approx(
        20 * math.log10(2), rel=1e-6)


def test_the_three_scales_are_one_distance():
    ref = _talk(16000, seed=11)
    deg = _noisy(ref, 5.0, 12)
    ln, base10, db = (lsd(ref, deg, 16000, scale=s) for s in ('ln', 'log10', 'db'))
    assert ln / base10 == pytest.approx(math.log(10), rel=1e-12)
    assert db / ln == pytest.approx(10 / math.log(10), rel=1e-12)


def test_more_noise_is_further_away():
    ref = _talk(16000, seed=13)
    scores = [lsd(ref, _noisy(ref, snr, 14), 16000) for snr in (30, 20, 10, 0)]
    assert scores == sorted(scores)


def test_a_one_millisecond_delay_shows_the_gain_match_needs_the_pair_aligned():
    """The speech fixture, deg 16 samples late. The least-squares gain falls
    to 0.30 and the default distance rises to 2.45; compared at their own
    levels the same pair scores 0.38."""
    ref = make_speech(16000, 6.0, seed=0).astype(np.float64)
    late = np.concatenate([np.zeros(16), ref])[:len(ref)]
    assert np.dot(ref, late) / np.dot(late, late) == pytest.approx(0.30, abs=0.01)
    assert lsd(ref, late, 16000) == pytest.approx(2.45, abs=0.01)
    assert lsd(ref, late, 16000, match_gain=False) == pytest.approx(0.38, abs=0.01)


def test_eps_decides_the_score_where_deg_is_digital_zero():
    """Half a second of deg dropped out to zeros, in 3 s: moving eps from
    1e-8 to 1e-12 takes the score from 4.90 to 7.74, because those frames
    are compared against a magnitude of eps. On a deg with noise in every
    bin the same change moves the score by 0.001."""
    ref = _talk(16000, seed=15)
    dropout = ref.copy()
    dropout[16000:24000] = 0.0
    assert lsd(ref, dropout, 16000) == pytest.approx(4.90, abs=0.01)
    assert lsd(ref, dropout, 16000, eps=1e-12) == pytest.approx(7.74, abs=0.01)
    noisy = _noisy(ref, 10.0, 16)
    assert lsd(ref, noisy, 16000, eps=1e-12) - lsd(ref, noisy, 16000) < 0.002


def test_frames_where_ref_is_digital_silence_are_counted_and_named():
    """Half a second of zeros before 2.5 s of the speech fixture: 31 of the
    188 frames are silent, each scores -ln(1e-8) = 18.42 whatever deg holds,
    so the pair scores 31 * 18.42 / 188 = 3.04 against itself."""
    x = np.concatenate([np.zeros(8000), make_speech(16000, 6.0, seed=0)[:40000]])
    with pytest.warns(UserWarning, match="31 of ref's 188 frames are digital silence"):
        got = lsd(x, x, 16000)
    assert got == pytest.approx(31 * -math.log(1e-8) / 188, abs=1e-4)


@pytest.mark.parametrize("scale, each", [('ln', '18.4'), ('log10', '8.0'), ('db', '80.0')])
def test_the_warning_gives_a_silent_frames_score_on_the_scale_asked_for(scale, each):
    """-ln(1e-8) is 18.4 in natural log, 8.0 in base 10 and 80.0 in dB."""
    x = np.concatenate([np.zeros(8000), make_speech(16000, 6.0, seed=0)[:40000]])
    with pytest.warns(UserWarning, match=f"each of them {re.escape(each)} whatever"):
        lsd(x, x, 16000, scale=scale)


def test_ref_that_is_mostly_digital_silence_still_scores():
    """0.3 s of the speech fixture after 2.7 s of zeros: 168 of the 188
    frames are silent and score 18.4 each, so the pair scores 16.46 against
    itself. Only a ref with no sound in it at all is nan."""
    speech = make_speech(16000, 6.0, seed=0)
    x = np.concatenate([np.zeros(43200), speech[16000:20800]])
    with pytest.warns(UserWarning, match="168 of ref's 188 frames"):
        assert lsd(x, x, 16000) == pytest.approx(16.46, abs=0.01)


def test_a_faint_stretch_in_ref_is_not_digital_silence():
    """Half a second of noise at 1e-7 RMS where the zeros were: no bin of
    those frames reaches 1e-5, but none is zero either, so no frame scores
    at the floor and the pair scores 0.0064 against itself."""
    speech = make_speech(16000, 6.0, seed=0)
    x = np.concatenate([1e-7 * np.random.default_rng(1).standard_normal(8000), speech[:40000]])
    assert _quiet(lsd, x, x, 16000) == pytest.approx(0.0064, abs=1e-4)


def test_long_files_are_transformed_in_blocks_with_the_same_result(monkeypatch):
    ref = _talk(16000, seconds=4.0, seed=17)
    deg = _noisy(ref, 10.0, 18)
    whole = lsd(ref, deg, 16000)
    monkeypatch.setattr(evaluator, "_LSD_BLOCK", 7)
    assert lsd(ref, deg, 16000) == pytest.approx(whole, rel=1e-12)


def test_int16_pairs_score_like_their_float_copies():
    ref = _talk(16000, seed=19)
    deg = _noisy(ref, 10.0, 20)
    peak = max(np.max(np.abs(ref)), np.max(np.abs(deg)))
    ref16 = np.round(ref / peak * 16000).astype(np.int16)
    deg16 = np.round(deg / peak * 16000).astype(np.int16)
    assert lsd(ref16, deg16, 16000) == pytest.approx(
        lsd(ref16 / 32768.0, deg16 / 32768.0, 16000), rel=1e-12)


# ------------------------------------------- what it cannot score or refuses

def test_ref_that_is_silent_throughout_is_nan_with_a_warning():
    deg = _talk(16000)
    with pytest.warns(UserWarning, match="digital silence throughout"):
        assert math.isnan(lsd(np.zeros_like(deg), deg, 16000))


def test_deg_that_is_silent_throughout_is_scored_with_a_warning():
    """URGENT scores it, so kudio does too, but the number is ref's spectrum
    against eps and moves with eps alone."""
    ref = _talk(16000)
    with pytest.warns(UserWarning, match="deg is digital silence throughout"):
        at_default = lsd(ref, np.zeros_like(ref), 16000)
    with pytest.warns(UserWarning, match="deg is digital silence throughout"):
        at_smaller = lsd(ref, np.zeros_like(ref), 16000, eps=1e-12)
    assert math.isfinite(at_default) and at_smaller > at_default


def test_an_empty_pair_is_nan_with_a_warning():
    with pytest.warns(UserWarning, match=r"^lsd\(\): the pair is empty; scored nan$"):
        assert math.isnan(lsd(np.zeros(0), np.zeros(0), 16000))


def test_a_pair_silent_on_both_sides_gets_one_warning_for_its_nan():
    """deg's silence throughout is worth saying only about a score; with ref
    silent too there is none."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert math.isnan(lsd(np.zeros(16000), np.zeros(16000), 16000))
    assert [str(w.message) for w in caught] == [
        "lsd(): ref is digital silence throughout, so there is no spectrum to "
        "compare deg against; scored nan"]


@pytest.mark.parametrize("sr", [math.inf, math.nan, -16000])
def test_a_rate_that_is_not_a_positive_number_is_refused(sr):
    with pytest.raises(ValueError, match=r"^lsd\(\): sr has to be a positive"):
        lsd(np.ones(16000), np.ones(16000), sr)


@pytest.mark.parametrize("kwargs, match", [
    (dict(scale='dB'), "scale"),
    (dict(n_fft=1), "at least 2 samples"),
    (dict(hop_length=0), "hop_length"),
    (dict(n_fft=512, hop_length=513), "hop_length"),
    (dict(eps=0.0), "eps"),
])
def test_settings_that_cannot_be_used_are_refused(kwargs, match):
    ref = _talk(16000)
    with pytest.raises(ValueError, match=match):
        lsd(ref, ref, 16000, **kwargs)


def test_a_rate_too_low_for_the_default_frame_says_to_pass_one():
    """At 50 Hz, 32 ms is 1.6 samples."""
    with pytest.raises(ValueError, match="pass n_fft in samples"):
        lsd(np.ones(100), np.ones(100), 50)
    with pytest.raises(ValueError, match="positive sample rate"):
        lsd(np.ones(100), np.ones(100), 0)


def test_the_pair_is_checked_like_every_other_metric():
    ref = _talk(16000)
    with pytest.raises(ValueError, match="mono"):
        lsd(np.stack([ref, ref], axis=1), np.stack([ref, ref], axis=1), 16000)
    with pytest.warns(UserWarning, match="only the first"):
        lsd(ref, ref[:int(0.9 * len(ref))], 16000)
    with pytest.raises(ValueError, match="unmeasured"):
        lsd(ref, ref[:int(0.3 * len(ref))], 16000)


def test_samples_that_are_not_finite_are_refused():
    ref = _talk(16000)
    deg = ref.copy()
    deg[10] = np.inf
    with pytest.raises(ValueError, match="not finite"):
        lsd(ref, deg, 16000)
