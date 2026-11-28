# -*- coding: utf-8 -*-
"""Pitch tracking: does it find the pitch that is actually there, and does it
say nothing where there is none?"""
from __future__ import annotations

import dataclasses
import functools
import warnings

import numpy as np
import pytest

import kudio
from kudio.exceptions import FeatureError

from conftest import SR, make_speech


def cents(got, want):
    return 1200 * np.log2(np.asarray(got) / want)


def buzz(freq: float, seconds: float = 1.0, sr: int = SR) -> np.ndarray:
    """A harmonic tone -- something with a fundamental to find."""
    t = np.arange(int(sr * seconds)) / sr
    harmonics = sum(np.sin(2 * np.pi * freq * k * t) / k for k in range(1, 10))
    return (0.3 * harmonics).astype(np.float32)


# --------------------------------------------------------------- it is right

@pytest.mark.parametrize("hz", [90.0, 150.0, 220.0])
def test_a_known_fundamental_is_found(hz):
    track = kudio.f0(buzz(hz), SR)
    assert track.median_hz == pytest.approx(hz, rel=0.02)
    assert track.voiced_ratio > 0.9


def test_a_missing_fundamental_is_still_found():
    """The point of a periodicity estimator over reading the spectrogram: a
    telephone band removes the fundamental and the pitch is unchanged."""
    y = kudio.highpass(buzz(120.0), SR, cutoff=300.0)
    track = kudio.f0(y, SR)
    assert track.median_hz == pytest.approx(120.0, rel=0.03)


#: the pitch is one of a grid of 10 cents, so 5 is the nearest it can come; the
#: rest is the zeros past the end of a frame, which pull a trough towards a shorter
#: period, the more the fewer of them the frame holds. 66 Hz at 48 kHz has 2.8
#: periods in its 2048 samples and comes out 13.6 cents sharp.
NEAR = 14


@pytest.mark.parametrize("sr", [8000, 16000, 22050, 44100, 48000, 96000, 192000])
def test_a_tone_is_told_to_a_seventh_of_a_semitone(sr):
    """Seven harmonics, 66 to 385 Hz. The first and the last 100 ms are left out: a
    frame there is half zeros."""
    t = np.arange(int(0.5 * sr)) / sr
    for hz in np.geomspace(66, 385, 17):
        y = 0.1 * sum(np.sin(2 * np.pi * hz * k * t) / k for k in range(1, 8) if k * hz < sr / 2)
        track = kudio.f0(y, sr)
        inside = slice(10, -10)
        assert track.voiced[inside].all(), hz
        assert np.abs(cents(track.f0[inside], hz)).max() < NEAR, hz


@pytest.mark.parametrize("hz", [66.0, 75.5, 103.2, 187.3, 250.0, 385.0])
def test_a_tone_with_no_harmonics_is_told_as_well(hz):
    t = np.arange(SR) / SR
    track = kudio.f0(0.3 * np.sin(2 * np.pi * hz * t), SR)
    assert track.voiced[10:-10].all()
    assert np.abs(cents(track.f0[10:-10], hz)).max() < NEAR


@pytest.mark.parametrize("octaves_a_second, within", [(0.5, 14), (1, 14), (2, 20), (4, 60)])
def test_a_glide_is_followed_frame_by_frame(octaves_a_second, within):
    """100 Hz upwards: what the pitch is at the middle of each frame is known. The
    frame is 128 ms long, and at 4 octaves a second half an octave goes by in it."""
    t = np.arange(int(2 * SR / max(1, octaves_a_second))) / SR
    phase = 2 * np.pi * np.cumsum(100 * 2 ** (octaves_a_second * t)) / SR
    track = kudio.f0(0.1 * sum(np.sin(k * phase) / k for k in range(1, 8)), SR)
    truth = 100 * 2 ** (octaves_a_second * track.times)
    inside = (track.times > 0.08) & (track.times < t[-1] - 0.08) & (truth < 380)
    assert inside.sum() > 30
    assert track.voiced[inside].all()
    assert np.abs(cents(track.f0, truth)[inside]).max() < within


@pytest.mark.parametrize("harmonics", [(2, 3, 4, 5), (3, 4, 5, 6, 7), (5, 7, 9), (3, 5, 7)])
def test_the_pitch_of_harmonics_alone_is_the_one_they_share(harmonics):
    """With nothing at the fundamental, and in the last two nothing at twice it
    either. On the nearest step of the grid."""
    t = np.arange(SR // 2) / SR
    for hz in (80, 120, 150, 200, 250):
        track = kudio.f0(0.1 * sum(np.sin(2 * np.pi * hz * k * t) for k in harmonics), SR)
        assert track.voiced[10:-10].all()
        assert np.abs(cents(track.f0[10:-10], hz)).max() <= 5


def test_a_tone_fading_by_140_db_keeps_its_pitch():
    """The difference function is over its own mean, so a level is nothing to it:
    to 5e-8 of full scale."""
    t = np.arange(2 * SR) / SR
    y = 0.5 * 10 ** (-7 * t / 2) * sum(np.sin(2 * np.pi * 140 * k * t) / k for k in range(1, 8))
    track = kudio.f0(y, SR)
    assert track.voiced[10:-10].all()
    assert np.abs(cents(track.f0[10:-10], 140)).max() <= 5


@pytest.mark.parametrize("hz", [80.0, 120.0, 180.0])
def test_a_second_harmonic_over_the_fundamental_is_no_octave_error(hz):
    """By 10 dB. At 15 the wave is nearly one of twice the pitch, and is told as
    that."""
    t = np.arange(SR) / SR
    y = np.sin(2 * np.pi * hz * t) + 10 ** 0.5 * np.sin(4 * np.pi * hz * t + 0.7)
    track = kudio.f0(0.05 * y, SR)
    assert np.abs(cents(track.f0[10:-10], hz)).max() <= 5


@pytest.mark.parametrize("offset, within", [(0.1, 5), (0.3, 14), (1.0, 70)])
def test_an_offset_moves_the_pitch_little(offset, within):
    """Of a tone of 0.3. It is a step at the end of the frame, where the zeros
    begin: at three times the tone's peak, 62 cents."""
    t = np.arange(SR) / SR
    track = kudio.f0(offset + 0.3 * np.sin(2 * np.pi * 140 * t), SR)
    assert track.voiced[10:-10].all()
    assert np.abs(cents(track.f0[10:-10], 140)).max() < within


@pytest.mark.parametrize("y", [np.zeros(SR), np.full(SR, 0.2), np.full(SR, -1e-9)])
def test_what_has_no_period_has_no_pitch(y):
    track = kudio.f0(y, SR)
    assert len(track) == 101
    assert not track.voiced.any()


def test_silence_is_not_voiced():
    track = kudio.f0(np.zeros(SR, dtype=np.float32), SR)
    assert track.voiced_ratio == 0.0
    assert np.all(np.isnan(track.f0))
    assert np.isnan(track.median_hz)
    assert "no voiced frames" in str(track)


def test_noise_is_mostly_not_voiced():
    """An estimator with no voicing decision answers here too, and the answer
    is a contour drawn through nothing."""
    rng = np.random.default_rng(0)
    y = (0.1 * rng.standard_normal(SR)).astype(np.float32)
    assert kudio.f0(y, SR).voiced_ratio < 0.25


def test_unvoiced_frames_hold_nan_not_a_number():
    """Otherwise plotting the contour draws a line through the silence."""
    track = kudio.f0(make_speech(), SR)
    assert np.all(np.isnan(track.f0[~track.voiced]))
    assert np.all(np.isfinite(track.f0[track.voiced]))


# ------------------------------------------------------------------- shape

def test_frames_line_up_with_the_waveform():
    y = make_speech(seconds=6.0)
    track = kudio.f0(y, SR, hop_length=160)

    assert len(track) == len(track.times) == len(track.voiced)
    assert len(track.voiced_prob) == len(track)
    assert track.times[0] == pytest.approx(0.0, abs=1e-6)
    assert track.times[-1] == pytest.approx(len(y) / SR, abs=0.05)


def test_the_hop_defaults_to_ten_milliseconds():
    track = kudio.f0(buzz(150.0, seconds=1.0), SR)
    assert track.hop_length == SR // 100
    assert len(track) == pytest.approx(100, abs=2)


def test_an_empty_clip_gives_an_empty_track():
    track = kudio.f0(np.zeros(0, dtype=np.float32), SR)
    assert len(track) == 0
    assert track.voiced_ratio == 0.0
    assert track.voiced_segments() == []


# ------------------------------------------------------------------ summary

def test_the_summary_describes_the_two_bursts():
    """`make_speech` puts a 120 Hz burst at 1.0-2.0 s and a 150 Hz one at
    3.5-4.8 s over a noise floor -- so both the range and the segments are
    known in advance."""
    track = kudio.f0(make_speech(), SR)
    lo, hi = track.range_hz()

    assert lo == pytest.approx(120.0, rel=0.05)
    assert hi == pytest.approx(150.0, rel=0.05)
    assert track.semitone_range == pytest.approx(12 * np.log2(150 / 120), abs=0.6)

    summary = track.summary()
    assert set(summary) == {'frames', 'voiced_ratio', 'median_hz', 'low_hz',
                            'high_hz', 'semitone_range'}
    assert summary['median_hz'] == track.median_hz


def test_voiced_segments_land_on_the_bursts():
    segments = kudio.f0(make_speech(), SR).voiced_segments()
    assert len(segments) == 2
    assert segments[0][0] == pytest.approx(1.0, abs=0.1)
    assert segments[0][1] == pytest.approx(2.0, abs=0.1)
    assert segments[1][0] == pytest.approx(3.5, abs=0.1)
    assert segments[1][1] == pytest.approx(4.8, abs=0.1)


def test_a_run_voiced_to_the_last_frame_ends_with_the_signal():
    """A 3 s sweep voiced throughout: the last frame's centre is 3.000 s, and
    a hop after it is past the end of the file."""
    t = np.arange(3 * SR) / SR
    sweep = 0.3 * np.sin(2 * np.pi * (250 * t + 100 * t ** 2))
    track = kudio.f0(sweep, SR, fmin=200, fmax=1000)
    assert track.length == 3 * SR
    assert track.times[-1] + track.hop_length / SR > 3.0
    assert track.voiced_segments()[-1][1] == 3.0
    assert track.to_labels()[-1].end == 3.0
    # a track built without the length keeps the hop past the end
    loose = dataclasses.replace(track, length=None)
    assert loose.voiced_segments()[-1][1] > 3.0


def test_the_cli_ends_the_spans_and_labels_with_the_file(tmp_path, capsys):
    from kudio.cli import main
    t = np.arange(3 * SR) / SR
    path = tmp_path / "sweep.wav"
    kudio.save_wave(path, (0.3 * np.sin(2 * np.pi * (250 * t + 100 * t ** 2))
                           ).astype(np.float32), SR)
    labels = tmp_path / "voiced.txt"
    assert main(["pitch", str(path), "--fmin", "200", "--fmax", "1000", "--segments",
                 "--labels", str(labels)]) == 0
    span = capsys.readouterr().out.splitlines()[-2]      # the last span, above "wrote"
    assert span.split()[3] == "3.000", span
    assert labels.read_text(encoding="utf-8").split()[-3] == "3.000000"   # end, "voiced", n


def test_short_runs_are_dropped():
    track = kudio.f0(make_speech(), SR)
    assert len(track.voiced_segments(min_seconds=10.0)) == 0


def test_labels_round_trip_through_an_audacity_track(tmp_path):
    """A pitch track that can be opened in an editor beats one that can only
    be printed."""
    track = kudio.f0(make_speech(), SR)
    path = kudio.save_labels(tmp_path / "voiced.txt", track.to_labels())
    reloaded = kudio.load_labels(path)

    assert len(reloaded) == 2
    assert reloaded[0].text == "voiced 1"
    assert reloaded[0].start == pytest.approx(track.to_labels()[0].start, abs=1e-3)


def test_a_semitone_range_needs_two_edges():
    empty = kudio.f0(np.zeros(SR, dtype=np.float32), SR)
    assert np.isnan(empty.semitone_range)


# ------------------------------------------------------------ refused input

def test_a_range_the_wrong_way_round_is_refused():
    with pytest.raises(FeatureError, match="fmin < fmax"):
        kudio.f0(buzz(150.0), SR, fmin=400.0, fmax=65.0)


def test_a_range_above_nyquist_is_refused():
    with pytest.raises(FeatureError, match="Nyquist"):
        kudio.f0(buzz(150.0), 8000, fmax=5000.0)


def test_a_frame_too_short_for_fmin_is_refused_rather_than_silently_empty():
    """Two periods have to fit, and a frame that cannot hold them returns
    "unvoiced everywhere" -- which reads as a property of the recording."""
    with pytest.raises(FeatureError, match="too short for fmin"):
        kudio.f0(buzz(150.0), SR, fmin=50.0, frame_length=256)


def test_a_bad_rate_is_refused():
    with pytest.raises(FeatureError, match="positive sample rate"):
        kudio.f0(buzz(150.0), 0)


# -- channels and the frame length ---------------------------------------------

def test_stereo_is_refused_rather_than_read_at_half_pitch():
    import kudio
    from kudio.exceptions import FeatureError
    with pytest.raises(FeatureError, match="mono"):
        kudio.f0(np.zeros((16000, 2), dtype=np.float32), 16000)


def test_the_default_frame_grows_with_the_rate():
    """2048 samples cannot hold two periods of 65 Hz at 96 kHz; the default
    call used to be refused there."""
    import kudio
    sr = 96000
    t = np.arange(sr) / sr
    track = kudio.f0((0.3 * np.sin(2 * np.pi * 150 * t)).astype(np.float32), sr)
    assert abs(track.median_hz - 150) < 3


# -- corrupt clips -------------------------------------------------------------
@pytest.mark.parametrize("bad", [np.nan, np.inf, 1e300])
def test_f0_refuses_a_corrupt_clip(bad):
    """1e300 is a number, and none in the float32 the work is done in."""
    y = 0.1 * np.random.default_rng(0).standard_normal(16000)
    y[100] = bad
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(kudio.FeatureError, match="not finite"):
            kudio.f0(y, 16000)


@pytest.mark.parametrize("y, what", [(np.ones(4000, bool), "bool"),
                                     (np.ones(4000, complex), "complex128"),
                                     (np.array(['a'] * 600), "<U1")])
def test_f0_refuses_what_is_no_audio(y, what):
    """True went in as 1, and of complex samples the real part."""
    with pytest.raises(FeatureError, match=f"the samples are {what}, not real numbers"):
        kudio.f0(y, 16000)


def test_integer_samples_are_the_floats_of_the_file():
    pcm = np.round(32767 * buzz(150.0) / 2).astype(np.int16)
    for y in (pcm / 32768, pcm[:, np.newaxis], pcm[np.newaxis]):
        got, want = kudio.f0(pcm, SR), kudio.f0(y, SR)
        assert np.array_equal(got.f0, want.f0, equal_nan=True)
        assert np.array_equal(got.voiced_prob, want.voiced_prob)


# -- clips too short to tell -----------------------------------------------------

def _told(y, sr=SR, **kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        track = kudio.f0(y, sr, **kwargs)
    return track, [(str(w.message), w.filename) for w in caught]


@pytest.mark.parametrize("n", [1, 2, 10, 80])
def test_a_few_samples_have_no_pitch(n):
    """Ten samples of anything came back as fmin, voiced: 65 Hz, whose period is
    246. 80 samples hold two periods of 400 Hz, and of nothing in the range."""
    track, said = _told(buzz(120.0)[:n])
    assert len(track) == 1 and not track.voiced.any() and np.isnan(track.f0).all()
    assert said == [(f"f0(): the clip has {n} sample(s) and two periods of fmin=65 Hz are 493, "
                     f"so no pitch is told in it", __file__)]


@pytest.mark.parametrize("hz", [100.0, 150.0, 300.0])
def test_a_pitch_is_told_once_it_has_come_round_twice(hz):
    """In a clip shorter than two periods of the lowest pitch there could be. Under
    two of its own, none is."""
    for n in range(40, 490, 15):
        track, said = _told(buzz(hz)[:n])
        if n >= 2.05 * SR / hz:
            assert track.voiced.all(), n
            assert np.abs(cents(track.f0, hz)).max() < 60
        elif n < 1.95 * SR / hz:
            assert not track.voiced.any(), n
        (text, where), = said
        assert where == __file__
        least = 2 * SR / n
        assert text.endswith(f"so no pitch under {least:.4g} Hz is told in it" if least < 400
                             else "so no pitch is told in it")


@pytest.mark.parametrize("n, kwargs", [(0, {}), (493, {}), (2000, {}), (200, dict(fmin=200.0))])
def test_a_clip_long_enough_for_the_range_is_not_warned_about(n, kwargs):
    track, said = _told(buzz(250.0)[:n], **kwargs)
    assert said == []
    assert len(track) == (1 + n // 160 if n else 0) and track.voiced.all()
    assert track.f0.dtype == track.times.dtype == track.voiced_prob.dtype == np.float64


def test_the_command_line_says_a_clip_is_too_short(tmp_path, capsys):
    from kudio.cli import main
    path = tmp_path / "short.wav"
    kudio.save_wave(path, buzz(120.0)[:10], SR)
    assert main(["pitch", str(path)]) == 1
    out, err = capsys.readouterr()
    assert out.strip() == "no voiced frames in 1 (65-400 Hz)"
    assert "warning: f0(): the clip has 10 sample(s)" in err


def test_an_uncentred_frame_needs_a_frame_of_samples():
    with pytest.raises(FeatureError, match="fewer than the one frame of 2048 that center=False"):
        kudio.f0(buzz(150.0)[:2047], SR, center=False)
    track = kudio.f0(buzz(150.0)[:2048 + 159], SR, center=False)
    assert len(track) == 1 and track.times[0] == 1024 / SR
    assert len(kudio.f0(buzz(150.0)[:2048 + 160], SR, center=False)) == 2


# -- ranges, hops and rates --------------------------------------------------------

@pytest.mark.parametrize("hop, fmin, fmax, moves, span", [
    (1170, 65, 400, 32.1, 31.5), (16000, 65, 400, 431.1, 31.5), (160, 100, 120, 4.1, 3.2),
    (160, 100, 125.9, 4.1, 4), (160, 399, 400, 4.1, 0.1), (320, 120, 160, 9.1, 5)])
def test_a_hop_in_which_a_pitch_could_cross_the_range_is_refused(hop, fmin, fmax, moves, span):
    """It was in librosa's words, "Target size (315) must be at least input size
    (321)", and not as a kudio error."""
    with pytest.raises(FeatureError) as caught:
        kudio.f0(buzz(150.0), SR, hop_length=hop, fmin=fmin, fmax=fmax)
    assert str(caught.value) == (f"f0(): what a pitch can move by in a hop of {hop} samples "
                                 f"spans {moves} semitones, and {fmin}-{fmax} Hz only {span}; "
                                 f"shorten the hop or widen the range")


@pytest.mark.parametrize("seconds", [0.01, 0.02, 0.03])
def test_a_pause_has_no_pitch_and_a_voice_has_one(seconds):
    """1.2 s of voice and 0.8 s of floor, 40 dB under it, over and over. The frame
    is 128 ms, so what is within 50 ms of an edge is left out."""
    t = np.arange(6 * SR) / SR
    y = (buzz(140.0, 6.0) * (t % 2 < 1.2)
         + 0.003 * np.random.default_rng(0).standard_normal(len(t)))
    track = _told(y, hop_length=int(seconds * SR))[0]
    at = track.times % 2
    assert track.voiced[(at > 0.15) & (at < 1.05)].all()
    assert not track.voiced[(at > 1.35) & (at < 1.85)].any()


@pytest.mark.parametrize("hop, sr, ms", [(390, 16000, '24.4'), (800, 16000, '50'),
                                         (1169, 16000, '73.1'), (1075, 44100, '24.4')])
def test_a_hop_too_long_to_tell_voicing_by_is_warned_about(hop, sr, ms):
    """Once a call, and not a sample under it."""
    track, said = _told(buzz(150.0, 0.5, sr), sr, hop_length=hop)
    assert said == [(f"f0(): the hop is {hop} samples, {ms} ms, and voicing is told by how little "
                     f"a pitch moves in one: from about 24 ms up, the longer the hop the more of "
                     f"a pause comes back voiced", __file__)]
    if hop in (390, 1075):
        assert _told(buzz(150.0, 0.5, sr), sr, hop_length=hop - 1)[1] == []
    assert _told(np.zeros(0), sr, hop_length=hop)[1] == []


@functools.lru_cache(maxsize=None)
def _voice_with_pauses():
    """80 s of a voice that pauses for 0.8 s every 2, and the times of its samples; the
    six cases below take their own floor under it."""
    t = np.arange(80 * SR) / SR
    hz = 150 * 2 ** (0.3 * np.sin(2 * np.pi * 0.9 * t))
    phase = 2 * np.pi * np.cumsum(hz) / SR
    voice = 0.07 * sum(np.sin(k * phase) / k for k in range(1, 9)) * (t % 2 < 1.2)
    t.flags.writeable = voice.flags.writeable = False
    return t, voice


@pytest.mark.parametrize("fmin, fmax", [(65, 400), (100, 300)])
def test_which_is_where_more_of_a_pause_comes_back_voiced(fmin, fmax):
    """26 s of them, 0.66 s at a time, their floor 20 dB under the voice: of 100 frames
    3 at most are voiced at a hop of 20 ms, and 17 to 26 at 40."""
    t, voice = _voice_with_pauses()
    y = voice + 0.01 * np.random.default_rng(0).standard_normal(len(t))
    for ms, least, most in ((20, 0, 0.05), (24, 0, 0.06), (40, 0.1, 1)):
        track = _told(y, hop_length=ms * 16, fmin=fmin, fmax=fmax)[0]
        at = track.times % 2
        assert least <= track.voiced[(at > 1.27) & (at < 1.93)].mean() <= most, ms
        assert track.voiced[(at > 0.07) & (at < 1.13)].all()


@pytest.mark.parametrize("fmin, fmax, least, most", [(100, 300, 0, 0.005), (65, 400, 0.005, 0.04),
                                                     (60, 600, 0.03, 0.1),
                                                     (50, 1000, 0.06, 0.2)])
def test_the_wider_the_range_the_more_of_a_pause_is_voiced(fmin, fmax, least, most):
    """0, 2, 6 and 12 frames in 100, which is in the docstring of f0."""
    t, voice = _voice_with_pauses()
    y = voice + 0.001 * np.random.default_rng(0).standard_normal(len(t))
    track = kudio.f0(y, SR, fmin=fmin, fmax=fmax)
    at = track.times % 2
    assert least <= track.voiced[(at > 1.27) & (at < 1.93)].mean() <= most


@pytest.mark.parametrize("hop, fmin, fmax", [(1169, 65, 400), (1, 65, 400), (8, 65, 400),
                                             (18, 149, 151), (160, 120, 160), (2205, 20, 8000)])
def test_up_to_that_a_hop_is_as_long_or_short_as_it_likes(hop, fmin, fmax):
    y = buzz(150.0, seconds=0.5)
    track = _told(y, hop_length=hop, fmin=fmin, fmax=fmax)[0]
    assert len(track) == 1 + len(y) // hop
    inside = (track.times >= 0.1) & (track.times <= 0.4)
    assert inside.any() and np.abs(cents(track.f0[inside], 150)).max() < NEAR


def test_a_rate_that_is_not_whole_is_not_cut_short():
    """16000.9 was read as 16000."""
    sr = 22254.5
    t = np.arange(int(sr)) / sr
    track = kudio.f0(0.3 * np.sin(2 * np.pi * 150 * t), sr)
    assert track.sr == sr and track.hop_length == 222
    assert track.times[-1] == 222 * 100 / sr
    assert np.abs(cents(track.f0[10:-10], 150)).max() < NEAR
    assert kudio.f0(buzz(150.0), 16000.9).sr == 16000.9
    same = kudio.f0(buzz(150.0), 16000.0)
    assert same.sr == 16000 and isinstance(same.sr, int)


@pytest.mark.parametrize("name", ["hop_length", "frame_length"])
@pytest.mark.parametrize("value", [-1, 0, 2.5, np.nan, np.inf, True, '2048'])
def test_a_length_is_a_whole_number_of_samples(name, value):
    """0 was 10 ms for a hop, and None is."""
    with pytest.raises(FeatureError) as caught:
        kudio.f0(buzz(150.0), SR, **{name: value})
    default = ", or None for 10 ms" if name == "hop_length" else ""
    assert str(caught.value) == (f"f0(): {name} must be >= 1, a whole number of samples{default}, "
                                 f"got {value!r}")


def test_a_length_may_be_a_whole_float():
    got = kudio.f0(buzz(150.0), SR, hop_length=160.0, frame_length=np.float32(2048))
    want = kudio.f0(buzz(150.0), SR)
    assert got.hop_length == 160 and np.array_equal(got.f0, want.f0)


@pytest.mark.parametrize("fmin, fmax", [(np.nan, 400), (65, np.nan), ('65', 400), (65, None),
                                        (True, 400), (0, 400), (-65, 400), (65, 65),
                                        (65, np.inf)])
def test_a_range_is_two_numbers(fmin, fmax):
    with pytest.raises(FeatureError, match="^need 0 < fmin < fmax, got fmin=.*, fmax="):
        kudio.f0(buzz(150.0), SR, fmin=fmin, fmax=fmax)
