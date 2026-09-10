# -*- coding: utf-8 -*-
"""Voice activity detection."""
import numpy as np
import pytest

import kudio
from kudio.exceptions import FeatureError

SR = 16000


def test_finds_the_two_bursts(speech):
    spans = kudio.vad(speech, SR)
    assert len(spans) == 2
    (a0, a1), (b0, b1) = spans
    # the bursts sit at 1.0-2.0 and 3.5-4.8; padding widens each by 0.05
    assert a0 == pytest.approx(0.95, abs=0.12)
    assert a1 == pytest.approx(2.05, abs=0.12)
    assert b0 == pytest.approx(3.45, abs=0.12)
    assert b1 == pytest.approx(4.85, abs=0.12)


def test_spans_are_ordered_and_disjoint(speech):
    spans = kudio.vad(speech, SR)
    for (_, end), (start, _) in zip(spans, spans[1:]):
        assert end <= start


def test_noise_alone_is_not_speech():
    rng = np.random.default_rng(1)
    noise = (0.01 * rng.standard_normal(SR * 4)).astype(np.float32)
    assert kudio.vad(noise, SR) == []


def test_digital_silence_is_not_speech():
    assert kudio.vad(np.zeros(SR, dtype=np.float32), SR) == []


def test_empty_input_returns_nothing():
    assert kudio.vad(np.zeros(0, dtype=np.float32), SR) == []


def test_adapts_to_the_noise_floor(speech):
    """The same speech under ten times the noise is still found."""
    rng = np.random.default_rng(2)
    noisier = speech + (0.05 * rng.standard_normal(len(speech))).astype(np.float32)
    assert len(kudio.vad(noisier, SR)) >= 1


def test_min_speech_drops_short_blips():
    rng = np.random.default_rng(3)
    y = (0.005 * rng.standard_normal(SR * 3)).astype(np.float32)
    y[SR:SR + 320] += 0.5          # a 20 ms click
    assert kudio.vad(y, SR, min_speech=0.5) == []


def test_min_silence_bridges_a_short_gap(speech):
    """A pause shorter than min_silence must not split a segment."""
    y = speech.copy()
    y[int(1.4 * SR):int(1.5 * SR)] *= 0.0        # 100 ms hole inside burst one
    joined = kudio.vad(y, SR, min_silence=0.3)
    split = kudio.vad(y, SR, min_silence=0.02, pad=0.0)
    assert len(joined) < len(split)


def test_flatness_test_rejects_a_loud_noise_burst():
    """Loud broadband noise passes the level gate; it must fail on flatness."""
    rng = np.random.default_rng(4)
    y = (0.005 * rng.standard_normal(SR * 4)).astype(np.float32)
    y[SR:2 * SR] += (0.3 * rng.standard_normal(SR)).astype(np.float32)
    assert kudio.vad(y, SR) == []
    # ...and relaxing the flatness test finds it, so it *was* loud enough
    assert kudio.vad(y, SR, flatness_max=1.0) != []


def test_vad_split_returns_the_audio(speech):
    spans = kudio.vad(speech, SR)
    segments = kudio.vad_split(speech, SR)
    assert len(segments) == len(spans)
    for segment, (start, end) in zip(segments, spans):
        assert len(segment) == pytest.approx((end - start) * SR, abs=2)


def test_vad_trim_keeps_the_middle(speech):
    trimmed, (start, end) = kudio.vad_trim(speech, SR)
    assert len(trimmed) == end - start
    assert start > 0
    assert end < len(speech)


def test_vad_trim_returns_the_clip_when_nothing_is_found():
    """Deleting the audio is never the safer reading of 'no speech'."""
    noise = (0.01 * np.random.default_rng(5).standard_normal(SR)).astype(np.float32)
    trimmed, (start, end) = kudio.vad_trim(noise, SR)
    assert (start, end) == (0, len(noise))
    assert len(trimmed) == len(noise)


def test_speech_ratio_is_a_fraction(speech):
    ratio = kudio.speech_ratio(speech, SR)
    assert 0.0 < ratio < 1.0
    assert kudio.speech_ratio(np.zeros(SR, dtype=np.float32), SR) == 0.0
    assert kudio.speech_ratio(np.zeros(0, dtype=np.float32), SR) == 0.0


def test_rejects_stereo_and_bad_rates(speech):
    with pytest.raises(FeatureError):
        kudio.vad(np.stack([speech, speech], axis=1), SR)
    with pytest.raises(FeatureError):
        kudio.vad(speech, 0)


def test_clip_shorter_than_one_frame_does_not_crash():
    assert isinstance(kudio.vad(np.zeros(100, dtype=np.float32), SR), list)
