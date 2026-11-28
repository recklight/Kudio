# -*- coding: utf-8 -*-
"""Time stretch and pitch shift against signals whose answer is known: a tone keeps
its frequency, two clicks come out so far apart, noise stays noise, and the end of a
clip is like the rest of it."""
from __future__ import annotations

import warnings
from fractions import Fraction

import numpy as np
import pytest
from scipy.signal import hilbert, welch

import kudio
from kudio.effects import _vocoder
from kudio.exceptions import FeatureError

SR = 16000
RATES = [0.5, 0.8, 1.25, 2.0]
SEMITONES = [1, -1, 4, -4, 12, -12, 0.5, -2.37]


def sine(hz, seconds, sr=SR, amplitude=0.2, phase=0.0):
    return (amplitude * np.sin(2 * np.pi * hz * np.arange(int(seconds * sr)) / sr + phase)
            ).astype(np.float32)


def off(x, hz, sr, seconds=0.5):
    """What a sine of *hz* leaves of *x*, worst half second, in dB under it; how far
    its frequency is off, in cents; and how far its level, from 0.2, in dB. The
    first and the last 150 ms are left out, or two frames of 2048."""
    edge = max(int(0.15 * sr), 4096)
    x = np.asarray(x, np.float64)[edge:-edge]
    n = int(seconds * sr)
    left, phase, level = [], [], []
    for start in range(0, len(x) - n + 1, n):
        t = np.arange(start, start + n) / sr
        basis = np.stack([np.cos(2 * np.pi * hz * t), np.sin(2 * np.pi * hz * t)], axis=1)
        c = np.linalg.lstsq(basis, x[start:start + n], rcond=None)[0]
        amplitude = np.hypot(*c)
        left.append(10 * np.log10(np.mean((x[start:start + n] - basis @ c) ** 2)
                                  / (amplitude ** 2 / 2) + 1e-30))
        phase.append(np.angle(c[0] - 1j * c[1]))
        level.append(20 * np.log10(amplitude / 0.2))
    turns = np.polyfit(np.arange(len(phase)) * seconds, np.unwrap(phase), 1)[0] / (2 * np.pi)
    return max(left), abs(1200 * np.log2(1 + turns / hz)), max(level, key=abs)


# -- tones -----------------------------------------------------------------------------

#: what is left of a tone once the best sine is out of it, at most, in dB: least where
#: every other output frame stands on a frame of the clip and the rest halfway between
#: two that the sine is alike in, and most where a frame is made of two
LEFT = {0.5: -31, 0.8: -38, 1.25: -90, 2.0: -44}


@pytest.mark.parametrize("rate", RATES)
@pytest.mark.parametrize("sr, hz", [(16000, 187.3), (16000, 3001.7), (48000, 187.3),
                                    (48000, 3001.7), (48000, 20640.0), (8000, 1003.1)])
def test_a_tone_keeps_its_frequency(sr, hz, rate):
    out = kudio.time_stretch(sine(hz, 6, sr), rate)
    left, cents, level = off(out, hz, sr)
    assert cents < 1e-3
    assert left < LEFT[rate]
    assert -1.3 < level < 0.01


@pytest.mark.parametrize("semitones", SEMITONES)
@pytest.mark.parametrize("sr, hz", [(16000, 187.3), (48000, 1003.1)])
def test_a_tone_is_shifted_by_the_semitones_asked_for(sr, hz, semitones):
    y = sine(hz, 4, sr)
    out = kudio.pitch_shift(y, sr, semitones)
    assert out.shape == y.shape and out.dtype == np.float32
    left, cents, level = off(out, hz * 2 ** (semitones / 12), sr)
    assert cents < 1e-3
    assert left < (-40 if semitones > 0 else -48 if semitones == -12 else -100)
    assert -1.3 < level < 0.01


def test_five_minutes_are_as_clean_as_the_first():
    """In float32 what a frame turns by is rounded to what the sum of all before it
    resolves, and with it the frequency: with librosa 0.11 this tone, speeded up, was
    1.5 cents flat at the end, and in the seconds in which it stepped 4 dB over what
    was left of it. It is where it should be, and 117 dB over."""
    y = sine(3001.7, 300, 48000)
    assert off(kudio.time_stretch(y, 1.25)[-10 * 48000:], 3001.7, 48000)[0] < -110
    assert off(kudio.pitch_shift(y, 48000, -4)[-10 * 48000:],
               3001.7 * 2 ** (-4 / 12), 48000)[0] < -120


@pytest.mark.parametrize("rate, left", [(0.5, 37), (0.8, 42), (1.25, 70), (2.0, 44)])
def test_harmonics_stay_in_step(rate, left):
    """Twelve of them: what is left besides is that many dB under, and no phase of the
    first eight is off its line by a tenth of a degree in 5 s."""
    hz, ks = 143.7, np.arange(1, 13)
    t = np.arange(4 * SR) / SR
    y = sum(0.1 / k * np.sin(2 * np.pi * hz * k * t + k) for k in ks).astype(np.float32)
    x = np.asarray(kudio.time_stretch(y, rate), np.float64)[2400:-2400]
    n, residue, whole, phases = SR // 10, 0.0, 0.0, []
    for start in range(0, len(x) - n + 1, n):
        at = np.arange(start, start + n) / SR
        basis = np.concatenate([[np.cos(2 * np.pi * hz * k * at), np.sin(2 * np.pi * hz * k * at)]
                                for k in ks]).T
        c = np.linalg.lstsq(basis, x[start:start + n], rcond=None)[0]
        residue += np.sum((x[start:start + n] - basis @ c) ** 2)
        whole += np.sum((basis @ c) ** 2)
        phases.append(np.angle(c[0::2] - 1j * c[1::2])[:8])
    assert 10 * np.log10(whole / residue) > left - 0.5
    phases = np.unwrap(phases, axis=0)
    assert np.degrees(np.abs(phases - phases.mean(axis=0))).max() < 0.1


# -- what moves ------------------------------------------------------------------------

@pytest.mark.parametrize("rate", RATES)
def test_a_sweep_goes_by_at_the_rate(rate):
    """100 to 3000 Hz in 8 s over a floor 66 dB under it: its frequency at every
    moment, within 1 % for half of them. Its level is down, by up to 5 dB: the bins it
    passes through keep no step with each other."""
    t = np.arange(8 * SR) / SR
    y = (0.2 * np.sin(2 * np.pi * (100 * t + 0.5 * 2900 / 8 * t ** 2))
         + 1e-4 * np.random.default_rng(0).standard_normal(len(t)))
    x = kudio.time_stretch(y, rate)
    hz = np.diff(np.unwrap(np.angle(hilbert(x)))) * SR / (2 * np.pi)
    hz = np.convolve(hz, np.full(160, 1 / 160), 'same')
    truth = 100 + 2900 / 8 * (np.arange(len(hz)) + 0.5) / SR * rate
    apart = np.abs(1200 * np.log2(np.maximum(hz, 1) / truth))[4800:-4800]
    assert np.median(apart) < 15
    assert -5 < 10 * np.log10(np.mean(x[8000:-8000] ** 2) / 0.02) < 0


@pytest.mark.parametrize("rate", RATES)
def test_two_clicks_come_out_so_far_apart(rate):
    """To half a frame, 64 ms: a frame has a click somewhere in it, and the output
    frame that has it stands where the frame is, not where the click is."""
    for at in (12000, 12117, 12256):
        y = np.zeros(2 * SR)
        y[[at, at + 8000]] = 0.8
        energy = kudio.time_stretch(y, rate) ** 2
        where = []
        for click in (at, at + 8000):
            near = np.arange(max(0, int(click / rate) - 1920), int(click / rate) + 1920)
            where.append(np.sum(near * energy[near]) / energy[near].sum())
        assert abs(where[1] - where[0] - 8000 / rate) < 0.064 * SR


@pytest.mark.parametrize("rate, db", [(0.5, -1.6), (0.8, -3.3), (1.25, -3.5), (2.0, -3.6)])
def test_noise_stays_noise(rate, db):
    """White still, to a dB in any octave, and some 3 dB down: the frames of it that
    overlap are no longer in step, and add up as powers."""
    y = 0.1 * np.random.default_rng(1).standard_normal(20 * SR)
    x = kudio.time_stretch(y, rate)[SR:-SR]
    level = 10 * np.log10(np.mean(x ** 2) / 0.01)
    assert level == pytest.approx(db, abs=0.2)
    hz, before = welch(y, SR, nperseg=1024)
    after = welch(x, SR, nperseg=1024)[1]
    for low in (62.5, 125, 250, 500, 1000, 2000, 3999):
        band = (hz >= low) & (hz < 2 * low)
        assert abs(10 * np.log10(after[band].mean() / before[band].mean()) - level) < 1.0


# -- the end ---------------------------------------------------------------------------

@pytest.mark.parametrize("rate, left, level", [(0.8, -10, -2.9), (0.9, -10, -3.4), (1.1, -25, -0.5),
                                               (1.25, -24, -0.4), (1.5, -27, -0.5)])
def test_a_tone_lasts_to_the_last_sample(rate, left, level):
    """Over the last 50 ms, the median of eight lengths: how far down what is left
    besides the harmonics is, and the level. With a frame of zeros for the one after
    the last, -17 dB and -1.1 at 1.25."""
    hz, ks, got = 143.7, np.arange(1, 13), []
    power = np.sum((0.1 / ks) ** 2) / 2
    for extra in (0, 60, 137, 200, 256, 300, 400, 511):
        t = np.arange(2 * SR + extra) / SR
        x = kudio.time_stretch(sum(0.1 / k * np.sin(2 * np.pi * hz * k * t + k) for k in ks), rate)
        tail, at = x[-800:], np.arange(len(x) - 800, len(x)) / SR
        basis = np.concatenate([[np.cos(2 * np.pi * hz * k * at), np.sin(2 * np.pi * hz * k * at)]
                                for k in ks]).T
        c = np.linalg.lstsq(basis, tail, rcond=None)[0]
        got.append((10 * np.log10(np.mean((tail - basis @ c) ** 2) / power),
                    10 * np.log10(np.mean(tail ** 2) / power)))
    residue, end = np.median(got, axis=0)
    assert residue < left
    assert level < end < 0.5


@pytest.mark.parametrize("rate", [0.8, 0.9, 1.1, 1.25, 1.5, 2.0])
def test_noise_lasts_to_the_last_sample(rate):
    """The last 50 ms against the 100 before, over 80 draws, every length of a frame
    in them: with a frame of zeros after the last, 0.9 to 1.8 dB down, but at 2,
    where there is none."""
    rng = np.random.default_rng(0)
    end = before = 0.0
    for draw in range(80):
        x = kudio.time_stretch(0.1 * rng.standard_normal(SR + 13 * draw % 512), rate)
        end += np.mean(x[-800:] ** 2)
        before += np.mean(x[-2400:-800] ** 2)
    assert abs(10 * np.log10(end / before)) < 0.6


def test_the_phases_turn_on_past_the_last_frame():
    """Read at an eighth of its speed, seven output frames stand past the last frame of
    the clip. A tone in the middle of bin 101 turns by a quarter a hop: frames that
    stood still would take each other out."""
    y = 0.2 * np.sin(2 * np.pi * 101 * np.arange(8192 + 511) / 2048)
    x = kudio.time_stretch(y, 0.125)
    past = x[16 * 8 * 512 + 512:][:2048]
    assert np.sqrt(np.mean(past ** 2)) > 0.35 * np.sqrt(np.mean(x[20000:40000] ** 2))


# -- lengths ---------------------------------------------------------------------------

def test_the_length_is_the_nearest_to_the_clips_over_the_rate():
    """Whatever the rate, as a fraction of whole numbers is one: never off by more than
    half a sample, and of two as near the even one."""
    for p in range(1, 13):
        for q in range(1, 13):
            for n in (2048, 2049, 2050, 2051, 3000, 4410):
                got = len(kudio.time_stretch(np.zeros(n, np.float32), p / q))
                want = Fraction(n * q, p)
                assert abs(got - want) <= Fraction(1, 2), (n, p, q)
                if (want * 2).denominator == 1 and want.denominator == 2 and q in (1, 2, 4, 8):
                    assert got % 2 == 0, (n, p, q)          # p / q is what it says in binary


@pytest.mark.parametrize("semitones", [0, 0.01, 1, -1, 7, -7, 12, -12, 19.5, 36, -36, 60])
def test_a_shifted_clip_is_as_long_as_it_was(semitones):
    for n in (1, 2, 511, 512, 2047, 2048, 2049, 5000):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")                          # short ones
            assert len(kudio.pitch_shift(np.ones(n, np.float32), SR, semitones)) == n


def test_a_shift_up_fills_the_clip_to_its_end():
    """The clip stretched is as long as it takes, up to a whole sample, so nothing is
    missing at the end of what is resampled."""
    y = sine(200, 1.0)
    for semitones in (24, 30.5, -24, -30.5):
        x = kudio.pitch_shift(y, SR, semitones)
        assert np.abs(x[-40:]).max() > 0.02, semitones


# -- short clips -----------------------------------------------------------------------

def test_a_clip_shorter_than_a_hop_is_finite_and_warned_about():
    """With librosa 1.0 a clip under 512 samples came back all NaN from time_stretch and
    was refused by pitch_shift as "not finite everywhere"."""
    for n in (1, 2, 10, 100, 256, 511, 512, 1000, 2047):
        for call, length in ((lambda y: kudio.time_stretch(y, 0.5), 2 * n),
                             (lambda y: kudio.time_stretch(y, 2.0), round(n / 2)),
                             (lambda y: kudio.pitch_shift(y, SR, 4), n),
                             (lambda y: kudio.pitch_shift(y, SR, -4), n)):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                x = call(sine(300, n / SR))
            assert len(x) == length and np.isfinite(x).all()
            assert np.abs(x).max(initial=0) < 0.35
            (said,) = caught
            assert said.filename == __file__
            assert str(said.message).endswith(
                f"(): the clip has {n} sample(s), fewer than one frame of 2048; what it works on "
                f"is mostly the zeros around it")


def test_a_frame_is_enough_and_nothing_is_nothing():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert len(kudio.time_stretch(sine(300, 2048 / SR), 0.5)) == 4096
        for shape in ((0,), (0, 2)):
            for x in (kudio.time_stretch(np.zeros(shape, np.float32), 0.5),
                      kudio.pitch_shift(np.zeros(shape, np.float32), SR, 2)):
                assert x.shape == shape and x.dtype == np.float32


def test_a_clip_of_one_hop_or_less_speeded_up_is_its_beginning():
    """It has one frame, and the frame is the output."""
    y = sine(300, 400 / SR, phase=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert kudio.time_stretch(y, 2.0) == pytest.approx(y[:200], abs=1e-7)


# -- what it is handed ------------------------------------------------------------------

def test_no_change_is_a_copy():
    y = np.random.default_rng(0).standard_normal(3000).astype(np.float32)
    for x in (kudio.time_stretch(y, 1), kudio.time_stretch(y, 1.0), kudio.pitch_shift(y, SR, 0)):
        assert np.array_equal(x, y) and not np.shares_memory(x, y)
    # and what the vocoder makes of it at that speed is it to 14 places
    got = _vocoder.stretch(y.astype(np.float64), 1.0, len(y))
    assert np.abs(got - y).max() < 1e-13


def test_a_channel_is_treated_as_if_alone():
    rng = np.random.default_rng(0)
    y = 0.1 * rng.standard_normal((6000, 3))
    for call in (lambda a: kudio.time_stretch(a, 0.7), lambda a: kudio.pitch_shift(a, SR, 2.5)):
        together = call(y)
        assert together.shape == (len(call(y[:, 0])), 3)
        for c in range(3):
            assert np.array_equal(together[:, c], call(y[:, c]))
        assert np.array_equal(call(y[:, 0][np.newaxis]), call(y[:, 0]))
        assert np.array_equal(call(y[:, :1])[:, 0], call(y[:, 0]))


@pytest.mark.parametrize("dtype", ['f2', 'f4', 'f8', '>f4', '>f8', 'i1', 'i2', '>i2', 'i4', 'u1',
                                   'u2'])
def test_the_type_is_kept(dtype):
    """float16 and big-endian floats were refused, as a numba error. Integers are the
    floats of their full scale, rounded."""
    dtype = np.dtype(dtype)
    wave = 0.5 * np.sin(2 * np.pi * 300 * np.arange(6000) / SR)
    if dtype.kind == 'f':
        y, scale, middle = wave.astype(dtype), 1, 0
    else:
        limits = np.iinfo(dtype)
        scale = (int(limits.max) + 1) // (2 if limits.min == 0 else 1)
        middle = scale if limits.min == 0 else 0
        y = np.round(wave * (scale - 1) + middle).astype(dtype)
    for call in (lambda a: kudio.time_stretch(a, 1.3), lambda a: kudio.pitch_shift(a, SR, -3)):
        got = call(y)
        assert got.dtype == dtype
        if dtype.kind != 'f':
            as_float = call((y.astype(np.float64) - middle) / scale)
            want = np.round(as_float * scale + middle)
            assert np.array_equal(got, want.astype(dtype))
        else:
            near = call(y.astype(np.float64)).astype(dtype)
            assert np.abs(got.astype(np.float64) - near.astype(np.float64)).max() < (
                2e-3 if dtype.itemsize == 2 else 3e-7)


@pytest.mark.parametrize("size", [1, 3, 64, 10 ** 6])
def test_the_blocks_it_is_worked_on_in_do_not_show(monkeypatch, size):
    y = np.random.default_rng(0).standard_normal(30000)
    for rate in (0.4, 0.9, 1.3, 2.5, 7.0):
        want = _vocoder.stretch(y, rate, round(len(y) / rate))
        monkeypatch.setattr(_vocoder, '_BLOCK', size)
        assert np.array_equal(_vocoder.stretch(y, rate, len(want)), want)
        monkeypatch.undo()


# -- refused ---------------------------------------------------------------------------

@pytest.mark.parametrize("rate", [np.nan, np.inf, -np.inf, 0, 0.0, -1, True, False, '1.5', None,
                                  [1.5], np.array([1.5]), 1 + 0j, 10 ** 400])
def test_a_rate_is_a_finite_number_above_0(rate):
    """NaN was numpy's "arange: cannot compute length", inf gave no samples, True was 1
    and '1.5' a TypeError."""
    with pytest.raises(FeatureError) as caught:
        kudio.time_stretch(np.zeros(4000, np.float32), rate)
    assert str(caught.value) == (f"time_stretch(): rate has to be a finite number above 0, got "
                                 f"{rate!r}")


@pytest.mark.parametrize("steps", [np.nan, np.inf, -np.inf, True, False, '2', None, [2],
                                   np.array([2]), 2 + 0j, 121, -120.5])
def test_semitones_are_a_number_up_to_ten_octaves(steps):
    """'2' was 2, True 1, and -inf a ZeroDivisionError."""
    with pytest.raises(FeatureError) as caught:
        kudio.pitch_shift(np.zeros(4000, np.float32), SR, steps)
    assert str(caught.value) == (f"pitch_shift(): n_steps has to be a number of semitones from "
                                 f"-120 to 120, got {steps!r}")


@pytest.mark.parametrize("value", [2, np.int64(2), np.float32(2), np.array(2.0)])
def test_a_setting_may_be_any_kind_of_number(value):
    y = sine(300, 0.3)
    assert np.array_equal(kudio.time_stretch(y, value), kudio.time_stretch(y, 2.0))
    assert np.array_equal(kudio.pitch_shift(y, SR, value), kudio.pitch_shift(y, SR, 2.0))


CALLS = {'time_stretch': lambda y: kudio.time_stretch(y, 1.25),
         'pitch_shift': lambda y: kudio.pitch_shift(y, SR, 2)}


@pytest.mark.parametrize("name", CALLS)
@pytest.mark.parametrize("y, said", [
    (np.r_[np.ones(3000), np.nan], "got 1 sample(s) that are not finite"),
    (np.r_[np.ones(3000), np.inf, -np.inf], "got 2 sample(s) that are not finite"),
    (np.ones(3000, bool), "the samples are bool, not real numbers"),
    (np.ones(3000, complex), "the samples are complex128, not real numbers"),
    (np.array(['a'] * 3000), "the samples are <U1, not real numbers"),
    (np.arange(3000, dtype=np.int64), "int64, and no audio file holds 64-bit integers"),
    (np.ones((2, 3000)), "channels first"),
    (np.ones((4, 4, 4)), "() takes (frames,) or (frames, channels), got shape (4, 4, 4)"),
    (np.ones((3000, 0)), "() takes (frames,) or (frames, channels), got shape (3000, 0)"),
    (np.float32(1), "() takes (frames,) or (frames, channels), got shape ()")])
def test_what_is_no_clip_is_refused(name, y, said):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(FeatureError) as caught:
            CALLS[name](y)
    assert said in str(caught.value)


def test_samples_too_high_to_come_back_are_refused():
    """1.2 times the largest float32 is no float32, and the resampler makes NaN of
    samples of 1e300."""
    loud = np.full(6000, 3.4e38, np.float32)
    loud[np.random.default_rng(0).random(6000) < 0.5] *= -1
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(FeatureError, match=r"^time_stretch\(\): the samples go as high as "
                                               r"3\.4e\+38, and what comes of them goes past what "
                                               r"float32 holds$"):
            kudio.time_stretch(loud, 0.8)
        with pytest.raises(FeatureError, match="^pitch_shift.*go as high as 2e\\+299"):
            kudio.pitch_shift(1e300 * sine(300, 1, amplitude=0.2).astype(np.float64), SR, 2)


def test_a_length_past_any_number_is_refused():
    with pytest.raises(FeatureError, match="4000 sample.s. at a rate of 4.94066e-324 are more"):
        kudio.time_stretch(np.zeros(4000, np.float32), 5e-324)
