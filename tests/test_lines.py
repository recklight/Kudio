# -*- coding: utf-8 -*-
"""tonal_lines: narrowband lines, and the combs they make."""
import numpy as np
import pytest

import kudio
from kudio.exceptions import FeatureError


def _noise(n, level, seed=0):
    return level * np.random.default_rng(seed).standard_normal(n)


def _hum(sr, seconds, f0, harmonics, level=0.05):
    t = np.arange(int(sr * seconds)) / sr
    return sum(level / k * np.sin(2 * np.pi * f0 * k * t + k) for k in harmonics)


def test_a_tone_is_one_line_at_its_frequency_and_level():
    """A sine at -20 dBFS between two bins, over white noise 40 dB down."""
    sr = 16000
    t = np.arange(3 * sr) / sr
    found = kudio.tonal_lines(0.1 * np.sin(2 * np.pi * 1002.3 * t) + _noise(t.size, 0.01), sr)
    assert len(found.lines) == 1 and found.combs == ()
    line = found.lines[0]
    assert line.hz == pytest.approx(1002.3, abs=0.1)
    assert line.level_dbfs == pytest.approx(-20.0, abs=0.5)
    assert line.prominence_db > 40
    assert found.strongest == line


@pytest.mark.parametrize("sr,n", [(16000, 48000), (192000, 192000), (16000, 1000)],
                         ids=["3 s at 16 kHz", "1 s at 192 kHz", "a short clip"])
def test_white_noise_has_no_lines(sr, n):
    found = kudio.tonal_lines(_noise(n, 0.1), sr)
    assert found.lines == () and found.strongest is None
    assert str(found) == "no line stands 20 dB above the spectrum around it"


def test_mains_hum_is_one_comb():
    sr = 48000
    y = _hum(sr, 2.0, 50.0, range(1, 11)) + _noise(2 * sr, 0.003)
    found = kudio.tonal_lines(y, sr)
    assert len(found.combs) == 1
    comb = found.combs[0]
    assert comb.fundamental_hz == pytest.approx(50.0, abs=0.01)
    assert comb.harmonics == tuple(range(1, 11))
    assert str(comb) == "50.00 Hz comb, 10 lines (harmonics 1-10)"


def test_a_comb_without_its_fundamental_is_found_at_the_spacing():
    """Hum through something that cannot reproduce 60 Hz itself."""
    sr = 48000
    found = kudio.tonal_lines(_hum(sr, 2.0, 60.0, range(2, 9)) + _noise(2 * sr, 0.003), sr)
    assert [round(c.fundamental_hz, 2) for c in found.combs] == [60.0]
    assert found.combs[0].harmonics == tuple(range(2, 9))


def test_a_line_off_the_comb_is_left_out_of_it():
    """1, 2 and 3 kHz with 2.5 kHz beside them: 500 Hz explains all four but
    misses two of its own multiples, and 1000 Hz wins the tie."""
    sr = 48000
    t = np.arange(2 * sr) / sr
    y = sum(0.02 * np.sin(2 * np.pi * f * t) for f in (1000, 2000, 2500, 3000))
    found = kudio.tonal_lines(y + _noise(t.size, 0.003), sr)
    assert len(found.lines) == 4
    assert [round(c.fundamental_hz) for c in found.combs] == [1000]
    assert 2500 not in [round(line.hz) for line in found.combs[0].lines]


def test_two_combs_are_kept_apart():
    """A 50 Hz hum with a 1 kHz comb on top: taken as its 20th, 40th and 60th
    harmonics, the 1 kHz lines made a 500 Hz comb out of the hum's 10th."""
    sr = 48000
    t = np.arange(2 * sr) / sr
    tones = sum(0.02 * np.sin(2 * np.pi * f * t) for f in (1000, 2000, 2500, 3000))
    y = _hum(sr, 2.0, 50.0, range(1, 11)) + tones + _noise(t.size, 0.003)
    combs = {round(c.fundamental_hz): c.harmonics for c in kudio.tonal_lines(y, sr).combs}
    assert combs == {50: tuple(range(1, 11)), 1000: (1, 2, 3)}


def test_a_voice_held_at_one_pitch_is_a_comb_too():
    """What the function says is that frequencies stand out, not why: the
    test suite's speech holds 120 and 150 Hz for a second each."""
    from conftest import make_speech
    found = kudio.tonal_lines(make_speech(), 16000)
    assert {round(c.fundamental_hz) for c in found.combs} == {120, 150}


def test_the_threshold_and_the_cap_are_kept():
    sr = 48000
    y = _hum(sr, 2.0, 50.0, range(1, 11)) + _noise(2 * sr, 0.003)
    every = kudio.tonal_lines(y, sr)
    few = kudio.tonal_lines(y, sr, max_lines=3)
    assert len(few.lines) == 3
    assert sorted(line.prominence_db for line in few.lines) == sorted(
        line.prominence_db for line in every.lines)[-3:]
    high = kudio.tonal_lines(y, sr, min_prominence_db=45)
    assert all(line.prominence_db >= 45 for line in high.lines)
    assert len(high.lines) < len(every.lines)


def test_the_resolution_follows_the_request_and_the_length():
    sr = 16000
    y = _noise(3 * sr, 0.1)
    assert kudio.tonal_lines(y, sr).resolution_hz == pytest.approx(16000 / 4096)
    assert kudio.tonal_lines(y, sr, resolution_hz=1.0).resolution_hz == pytest.approx(
        16000 / 16384)
    assert kudio.tonal_lines(y[:1000], sr).resolution_hz == pytest.approx(16.0)


def test_silence_has_no_lines():
    assert kudio.tonal_lines(np.zeros(16000), 16000).lines == ()


@pytest.mark.parametrize("call,match", [
    (lambda: kudio.tonal_lines(np.zeros((100, 2)), 16000), "mono"),
    (lambda: kudio.tonal_lines(np.r_[np.zeros(100), np.nan], 16000), "not finite"),
    (lambda: kudio.tonal_lines(np.zeros(100), 0), "sample rate"),
    (lambda: kudio.tonal_lines(np.zeros(100), 16000, resolution_hz=0), "resolution"),
])
def test_bad_input_is_refused(call, match):
    with pytest.raises(FeatureError, match=match):
        call()
