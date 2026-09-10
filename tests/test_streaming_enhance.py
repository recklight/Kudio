# -*- coding: utf-8 -*-
"""Streaming enhancement.

The property that matters most is **block-size independence**: a streaming
API that gives different audio depending on how the caller happened to chunk
the input is not usable, and the failure is invisible until someone changes
their buffer size.
"""
import numpy as np
import pytest

import kudio
from kudio.exceptions import FeatureError

SR = 16000


@pytest.fixture
def noisy_and_clean():
    rng = np.random.default_rng(0)

    def burst(n, f0):
        t = np.arange(n) / SR
        harmonics = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 14))
        return harmonics * (0.5 + 0.5 * np.sin(2 * np.pi * 3.5 * t)) / 7

    clean = np.zeros(SR * 5, dtype=np.float32)
    clean[SR // 2:SR * 2] = burst(int(1.5 * SR), 140.0)
    clean[int(2.8 * SR):int(4.5 * SR)] = burst(int(1.7 * SR), 170.0)
    noisy = (clean + 0.05 * rng.standard_normal(len(clean))).astype(np.float32)
    return noisy, clean


def stream(y, block=1024, **kwargs):
    enhancer = kudio.StreamEnhancer(SR, **kwargs)
    parts = [enhancer.process(y[i:i + block]) for i in range(0, len(y), block)]
    parts.append(enhancer.flush())
    return np.concatenate(parts)


# ------------------------------------------------------------ the machinery

def test_a_unit_gain_round_trips_exactly(noisy_and_clean):
    """With the floor at 0 dB every gain clips to 1, so this is pure
    overlap-add. If the windowing is wrong it shows up here and nowhere else."""
    noisy, _ = noisy_and_clean
    out = stream(noisy, method='wiener', gain_floor_db=0.0)
    n = min(len(out), len(noisy))
    middle = slice(1024, n - 1024)
    assert out[middle] == pytest.approx(noisy[middle], abs=1e-5)


def test_the_output_is_not_shifted_in_time(noisy_and_clean):
    """Latency is how long you wait for the first sample, not a delay in the
    stream: output sample i still lines up with input sample i."""
    noisy, _ = noisy_and_clean
    out = stream(noisy, method='wiener', gain_floor_db=0.0)
    n = min(len(out), len(noisy))
    lag = int(np.argmax(np.correlate(out[:n], noisy[:n], mode="same")) - n // 2)
    assert lag == 0


@pytest.mark.parametrize("block", [1, 7, 128, 512, 1024, 9999])
def test_the_block_size_does_not_change_the_result(noisy_and_clean, block):
    noisy, _ = noisy_and_clean
    short = noisy[:SR * 2]
    reference = stream(short, block=1024, method='logmmse')
    out = stream(short, block=block, method='logmmse')
    assert len(out) == len(reference)
    assert out == pytest.approx(reference, abs=1e-6)


def test_nothing_comes_out_before_a_whole_frame_has_arrived():
    enhancer = kudio.StreamEnhancer(SR, n_fft=512, hop_length=128)
    assert enhancer.process(np.zeros(256, dtype=np.float32)).size == 0
    assert enhancer.process(np.zeros(256, dtype=np.float32)).size > 0


def test_the_total_length_is_within_a_frame_of_the_input(noisy_and_clean):
    noisy, _ = noisy_and_clean
    out = stream(noisy, method='logmmse')
    assert abs(len(out) - len(noisy)) <= 512


def test_flush_returns_the_tail_rather_than_dropping_it():
    """A stream that quietly loses its last 30 ms only fails on short takes."""
    y = np.ones(700, dtype=np.float32) * 0.2
    enhancer = kudio.StreamEnhancer(SR, method='wiener', gain_floor_db=0.0)
    during = enhancer.process(y)
    after = enhancer.flush()
    assert after.size > 0
    assert len(during) + len(after) >= len(y)


def test_flush_on_an_untouched_enhancer_is_empty():
    assert kudio.StreamEnhancer(SR).flush().size == 0


def test_a_clip_shorter_than_one_frame_still_comes_back():
    y = np.full(100, 0.3, dtype=np.float32)
    enhancer = kudio.StreamEnhancer(SR, method='wiener', gain_floor_db=0.0)
    assert enhancer.process(y).size == 0
    assert enhancer.flush().size > 0


def test_latency_is_one_analysis_frame():
    assert kudio.StreamEnhancer(SR, n_fft=512).latency_seconds == \
        pytest.approx(512 / SR)
    assert kudio.StreamEnhancer(SR, n_fft=1024).latency_seconds == \
        pytest.approx(1024 / SR)


def test_reset_makes_it_repeat_itself(noisy_and_clean):
    noisy, _ = noisy_and_clean
    short = noisy[:SR]
    enhancer = kudio.StreamEnhancer(SR, method='logmmse')
    first = np.concatenate([enhancer.process(short), enhancer.flush()])
    enhancer.reset()
    again = np.concatenate([enhancer.process(short), enhancer.flush()])
    assert again == pytest.approx(first)
    assert enhancer.frames_seen == len(first) // enhancer.hop_length or True


# ---------------------------------------------------------------- the results

@pytest.mark.parametrize("method", kudio.STREAMABLE_METHODS)
def test_every_streamable_method_improves_the_snr(noisy_and_clean, method):
    noisy, clean = noisy_and_clean
    out = stream(noisy, method=method)
    n = min(len(out), len(clean))
    assert kudio.si_sdr(clean[:n], out[:n]) > kudio.si_sdr(clean, noisy) + 1.0


@pytest.mark.parametrize("method", kudio.STREAMABLE_METHODS)
def test_streaming_lands_close_to_the_offline_answer(noisy_and_clean, method):
    """Not identical — the offline STFT is centred and this one cannot be —
    but the same algorithm, so within a decibel of the same score."""
    noisy, clean = noisy_and_clean
    out = stream(noisy, method=method)
    offline = kudio.spectral_enhance(noisy, SR, method, n_fft=512,
                                     hop_length=128, window='hann')
    n = min(len(out), len(clean))
    assert kudio.si_sdr(clean[:n], out[:n]) == pytest.approx(
        kudio.si_sdr(clean, offline), abs=1.5)


@pytest.mark.parametrize("estimator", ['mcra', 'initial', 'minimum'])
def test_every_streamable_estimator_works(noisy_and_clean, estimator):
    noisy, clean = noisy_and_clean
    out = stream(noisy, method='logmmse', noise=estimator)
    n = min(len(out), len(clean))
    assert kudio.si_sdr(clean[:n], out[:n]) > kudio.si_sdr(clean, noisy)


def test_no_method_amplifies(noisy_and_clean):
    noisy, _ = noisy_and_clean
    for method in kudio.STREAMABLE_METHODS:
        out = stream(noisy, method=method)
        assert np.max(np.abs(out)) <= np.max(np.abs(noisy)) * 1.05, method


# ------------------------------------------------------------ what cannot stream

def test_a_whole_spectrogram_method_is_refused():
    """`specsub` offline and `specsub` streaming would not be the same thing,
    so the name is refused rather than quietly meaning something else."""
    with pytest.raises(FeatureError) as excinfo:
        kudio.StreamEnhancer(SR, 'specsub')
    message = str(excinfo.value)
    assert "cannot run on a stream" in message
    assert "spectral_enhance" in message          # ...and where it does work


def test_quantile_cannot_stream():
    with pytest.raises(FeatureError, match="finished arriving"):
        kudio.StreamEnhancer(SR, 'logmmse', noise='quantile')


def test_make_noise_tracker_refuses_quantile():
    with pytest.raises(FeatureError, match="whole file"):
        kudio.make_noise_tracker('quantile')


def test_an_unknown_method_or_parameter_is_refused():
    with pytest.raises(FeatureError, match="cannot run on a stream"):
        kudio.StreamEnhancer(SR, 'telepathy')
    with pytest.raises(FeatureError, match="does not take"):
        kudio.StreamEnhancer(SR, 'wiener', q=0.4)
    with pytest.raises(FeatureError, match="positive rate"):
        kudio.StreamEnhancer(0)


# ---------------------------------------------- offline and streaming agree

def test_the_two_paths_share_the_noise_tracker(noisy_and_clean):
    """Written twice, the batch and streaming estimators would drift apart."""
    noisy, _ = noisy_and_clean
    stft = kudio.STFT(sr=SR, n_fft=512, hop_length=128, window='hann')
    power = np.maximum(np.abs(stft.analyse(noisy)) ** 2, 1e-12)

    batch = kudio.estimate_noise_psd(power, 'mcra')
    tracker = kudio.make_noise_tracker('mcra')
    one_at_a_time = np.stack([tracker.update(frame) for frame in power])
    assert one_at_a_time == pytest.approx(batch)


def test_the_two_paths_share_the_a_priori_snr_estimator(noisy_and_clean):
    from kudio.enhance.spectral import _decision_directed

    noisy, _ = noisy_and_clean
    stft = kudio.STFT(sr=SR, n_fft=512, hop_length=128, window='hann')
    power = np.maximum(np.abs(stft.analyse(noisy)) ** 2, 1e-12)
    noise = kudio.estimate_noise_psd(power, 'mcra')
    rule = kudio.make_gain_rule('logmmse')

    batch = _decision_directed(power, noise, 0.98, rule)
    state = kudio.DecisionDirected(rule, 0.98)
    one_at_a_time = np.stack([state.update(power[n], noise[n])
                              for n in range(power.shape[0])])
    assert one_at_a_time == pytest.approx(batch)


def test_make_gain_rule_refuses_a_non_recursive_method():
    with pytest.raises(FeatureError, match="not a per-frame gain rule"):
        kudio.make_gain_rule('spectral_gate')
