# -*- coding: utf-8 -*-
import warnings

import numpy as np
import pytest

import kudio
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


def test_add_noise_snr_holds_the_snr_on_int16():
    """int16 squared wraps round in int16: a quarter-scale tone mixed at
    10 dB came out at 55.9 dB."""
    y = np.round(0.25 * np.sin(2 * np.pi * 440 * np.arange(16000) / 16000)
                 * 32768).astype(np.int16)
    noise = (3000 * np.random.default_rng(0).standard_normal(16000)).astype(np.int16)
    out = add_noise_snr(y, noise, snr_db=10, seed=0)
    added = out.astype(np.float64) - y
    assert out.dtype == np.int16
    assert abs(10 * np.log10(np.mean(y.astype(np.float64) ** 2) / np.mean(added ** 2)) - 10) < 0.05


def test_reverb_keeps_the_peak_of_int16_pinned_at_full_scale():
    """abs(-32768) is -32768 in int16, and one sample there left the result
    at a peak of 20000."""
    y = (20000 * np.sin(2 * np.pi * 220 * np.arange(16000) / 16000)).astype(np.int16)
    y[5000] = -32768
    out = reverb(y, sr=16000)
    assert out.dtype == np.int16 and np.abs(out.astype(np.int64)).max() == 32768


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


# -- settings that are no numbers, and samples that are not ------------------------

INT = (3000 * np.sin(2 * np.pi * 220 * np.arange(4000) / 16000)).astype(np.int16)
NOISE = np.random.default_rng(0).standard_normal(4000).astype(np.float32)
#: what each takes one setting by
SET = {
    'gain': lambda y, v: kudio.gain(y, v),
    'normalize': lambda y, v: kudio.normalize(y, v),
    'normalize_db': lambda y, v: kudio.normalize_db(y, v),
    'add_noise_snr': lambda y, v: kudio.add_noise_snr(y, NOISE, v, seed=0),
    'reverb decay': lambda y, v: kudio.reverb(y, 16000, decay=v),
    'reverb delay_ms': lambda y, v: kudio.reverb(y, 16000, delay_ms=v),
}


@pytest.mark.parametrize("name", SET)
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, True, '3', None, [3.0], 3 + 0j])
def test_a_setting_that_is_no_finite_number_is_refused(name, value):
    """With NaN an int16 clip came back all zeros, with numpy's "invalid value
    encountered in cast" to say so, and a float one all NaN."""
    for y in (INT, _tone()):
        with pytest.raises(ValueError, match="finite number"):
            SET[name](y, value)


@pytest.mark.parametrize("name, value", [('gain', 1e5), ('normalize', 0),
                                         ('normalize', -1), ('normalize_db', 0.1),
                                         ('normalize_db', -1e5), ('add_noise_snr', 1e5),
                                         ('add_noise_snr', -1e5), ('reverb delay_ms', 0),
                                         ('reverb delay_ms', -5)])
def test_a_setting_out_of_range_is_refused(name, value):
    """A reverb with a delay of 0 or less was an echo a sample later."""
    for kind in (float, np.float64):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(ValueError):
                SET[name](_tone(), kind(value))


@pytest.mark.parametrize("call", [lambda y: kudio.normalize(y), lambda y: kudio.normalize_db(y),
                                  lambda y: kudio.reverb(y, 16000),
                                  lambda y: kudio.add_noise_snr(y, NOISE, 5),
                                  lambda y: kudio.add_noise_snr(_tone(), y, 5)])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_one_sample_that_is_not_finite_is_not_spread_over_the_clip(call, bad):
    """The peak of a clip with inf in it is inf, and normalised, the rest came back 0."""
    y = _tone(seconds=0.5)
    y[100] = bad
    with pytest.raises(kudio.FeatureError, match="1 sample.s. that are not finite"):
        call(y)


def test_a_gain_keeps_to_the_sample_what_is_not_finite():
    """It is one sample of the result too, and nothing else's. An integer is neither."""
    y = _tone(seconds=0.1)
    y[100] = np.nan
    out = kudio.gain(y, -6.0)
    assert np.isnan(out[100]) and np.isfinite(np.delete(out, 100)).all()


def test_what_comes_out_not_finite_is_no_integer():
    from kudio._samples import to_float
    x, back = to_float(INT, 'this', kudio.FeatureError)
    for bad in (np.nan, np.inf):
        with pytest.raises(kudio.FeatureError, match=r"^this\(\): 2 of the samples that come out "
                                                     r"are not finite .NaN or inf., which int16"):
            back(np.r_[x[:10], bad, bad])
    assert np.array_equal(back(x), INT)


def test_an_empty_clip_comes_back_empty():
    for y in (np.zeros(0, np.float32), np.zeros(0, np.int16)):
        for out in (kudio.reverb(y, 16000), kudio.add_noise_snr(y, NOISE, 5), kudio.gain(y, 3),
                    kudio.normalize(y)):
            assert out.shape == (0,) and out.dtype == y.dtype


@pytest.mark.parametrize("kind", [float, np.float32, np.float64, np.array])
def test_a_setting_of_any_kind_of_number_is_taken(kind):
    y = _tone()
    for name, value in (('gain', 3.0), ('normalize', 0.5), ('normalize_db', -3.0),
                        ('add_noise_snr', 7.0), ('reverb decay', 0.25), ('reverb delay_ms', 40.0)):
        assert np.allclose(SET[name](y, kind(value)), SET[name](y, value), atol=1e-6)


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


# -- stretching and shifting integer audio -------------------------------------
@pytest.mark.parametrize("call", [
    lambda y: kudio.time_stretch(y, 1.5),
    lambda y: kudio.pitch_shift(y, 16000, 2),
])
def test_stretch_and_shift_take_integer_audio(call):
    y = (3000 * np.sin(2 * np.pi * 220 * np.arange(16000) / 16000)).astype(np.int16)
    out = call(y)
    assert out.dtype == np.int16 and 2000 < np.abs(out).max() < 6000


# -- a top_db that leaves nothing ----------------------------------------------------
@pytest.mark.parametrize("top_db", [0, 0.0, -5, float('nan'), float('inf'), float('-inf'),
                                    np.float32('nan'), np.array(-1.0), True, np.True_, None,
                                    '30', [30], np.array([30.0]), 30 + 0j,
                                    pytest.param(10 ** 400, id='past float64')])
def test_a_top_db_that_is_not_a_distance_down_is_refused(top_db):
    """At 0 or below, and at NaN, no frame is above the line and the clip comes
    back empty, the loudest frame with it."""
    import kudio
    y = np.concatenate([np.zeros(8000), _tone(), np.zeros(8000)])
    for call in (kudio.trim_silence, kudio.split_on_silence):
        with pytest.raises(ValueError, match=rf"^{call.__name__}\(\): top_db has to be a finite "
                                             rf"number of dB above 0, got ") as refused:
            call(y, top_db)
        assert isinstance(refused.value, kudio.FeatureError)


@pytest.mark.parametrize("top_db", [30, 30.0, np.int64(30), np.float32(30), np.array(30.0)])
def test_a_top_db_of_any_number_type_is_the_same_cut(top_db):
    import kudio
    y = np.concatenate([np.zeros(8000), _tone(), np.zeros(8000)])
    assert kudio.trim_silence(y, top_db)[1] == kudio.trim_silence(y)[1]
    assert ([len(p) for p in kudio.split_on_silence(y, top_db)]
            == [len(p) for p in kudio.split_on_silence(y)])


# -- where silence ends ---------------------------------------------------------------

def _sounding_by_hand(y, top_db, frame, hop):
    """Frame k, centred on sample k * hop, zeros past the ends, in plain loops."""
    y = np.asarray(y, dtype=np.float64)
    channels = y[:, None] if y.ndim == 1 else y
    n, half = len(channels), frame // 2
    rms = []
    for k in range(1 + (n + 2 * half - frame) // hop):
        row = []
        for c in range(channels.shape[1]):
            total = 0.0
            for s in range(k * hop - half, k * hop - half + frame):
                if 0 <= s < n:
                    total += channels[s, c] ** 2
            row.append(np.sqrt(total / frame))
        rms.append(row)
    rms = np.maximum(np.array(rms), 1e-5)
    db = 20 * np.log10(rms / rms.max())
    return db.max(axis=1) > -top_db


def _cuts(sounding, hop, n):
    """(start, end) of every run, in samples, held to the clip."""
    runs, start = [], None
    for k, on in enumerate(list(sounding) + [False]):
        if on and start is None:
            start = k
        elif not on and start is not None:
            runs.append((min(start * hop, n), min(k * hop, n)))
            start = None
    return runs


#: what there is to cut: a name, the clip, and the frame and hop to cut it in
TAKES = {
    'a burst': (np.r_[np.zeros(700), 0.3 * np.sin(np.arange(900) / 3), np.zeros(500)], 256, 64),
    'two bursts': (np.r_[np.zeros(300), np.full(200, 0.2), np.zeros(900), np.full(250, -0.2),
                         np.zeros(300)], 128, 32),
    'sound to both ends': (0.2 * np.sin(np.arange(1500) / 5), 128, 50),
    'one sample': (np.array([0.4]), 64, 16),
    'shorter than a frame': (np.r_[np.zeros(20), np.full(10, 0.5), np.zeros(20)], 64, 16),
    'exactly a frame': (np.r_[np.zeros(40), np.full(24, 0.5)], 64, 16),
    'silence': (np.zeros(500), 64, 16),
    'a constant': (np.full(500, 0.3), 64, 16),
    'a quiet constant': (np.full(500, 1e-7), 64, 16),
    'a hop longer than the clip': (np.r_[np.zeros(60), np.full(40, 0.5)], 64, 1000),
    'a hop longer than the frame': (np.r_[np.zeros(400), np.full(300, 0.5), np.zeros(300)], 32, 100),
    'an odd frame': (np.r_[np.zeros(400), np.full(300, 0.5), np.zeros(300)], 65, 20),
    'a frame of one sample': (np.r_[np.zeros(40), np.full(30, 0.5), np.zeros(30)], 1, 1),
    'two channels': (np.stack([np.r_[np.zeros(300), np.full(200, 0.4), np.zeros(500)],
                               np.r_[np.zeros(700), np.full(100, 0.1), np.zeros(200)]], 1), 64, 16),
    'six channels, one sounding': (np.stack([np.zeros(600)] * 5 + [
        np.r_[np.zeros(200), np.full(100, 0.2), np.zeros(300)]], 1), 64, 16),
}


@pytest.mark.parametrize("name", TAKES)
@pytest.mark.parametrize("top_db", [10, 30, 60.0])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_silence_is_cut_where_the_frames_say(name, top_db, dtype):
    y, frame, hop = TAKES[name]
    y = y.astype(dtype)
    runs = _cuts(_sounding_by_hand(y, top_db, frame, hop), hop, len(y))
    trimmed, (start, end) = kudio.trim_silence(y, top_db, frame, hop)
    assert (start, end) == (runs[0][0], runs[-1][1])
    assert np.array_equal(trimmed, y[start:end]) and trimmed.dtype == dtype
    pieces = kudio.split_on_silence(y, top_db, frame, hop)
    assert len(pieces) == len(runs)
    for piece, (a, b) in zip(pieces, runs):
        assert np.array_equal(piece, y[a:b]) and not np.shares_memory(piece, y)


def test_a_pulse_is_cut_at_the_first_frame_that_reaches_it():
    """At 60 dB down a frame of 2048 with one sample of a pulse in it is enough:
    one of 2048 is 33 dB under a full frame. So the first frame is the first
    centred within 1024 samples of where the pulse starts."""
    for at in (3000, 4096, 5000, 5119, 5120, 5121):
        y = np.zeros(16000, np.float32)
        y[at:at + 4000] = 0.5
        first = next(k for k in range(100) if k * 512 + 1024 > at)
        last = max(k for k in range(100) if k * 512 - 1024 < at + 4000)
        assert kudio.trim_silence(y, 60.0)[1] == (first * 512, (last + 1) * 512)


def test_silence_throughout_is_kept_whole():
    """There is no loudest frame for the others to lie under, and in a file of
    nothing but zeros an empty file would be no better an answer."""
    y = np.zeros(5000, np.float32)
    assert kudio.trim_silence(y)[1] == (0, 5000)
    assert [len(p) for p in kudio.split_on_silence(y)] == [5000]


@pytest.mark.parametrize("y", [np.zeros(0, np.float32), np.zeros((0, 2)), np.zeros(0, np.int16)])
@pytest.mark.parametrize("frame", [2048, 65])
def test_nothing_is_cut_from_nothing(y, frame):
    trimmed, cut = kudio.trim_silence(y, frame_length=frame)
    assert cut == (0, 0) and trimmed.shape == y.shape and trimmed.dtype == y.dtype
    assert [p.shape for p in kudio.split_on_silence(y, frame_length=frame)] == [y.shape]


@pytest.mark.parametrize("call", [kudio.trim_silence, kudio.split_on_silence],
                         ids=lambda f: f.__name__)
@pytest.mark.parametrize("y, says", [
    (np.r_[np.zeros(3000), np.nan, np.inf], r" got 2 sample\(s\) that are not finite"),
    (np.ones(3000, bool), ": the samples are bool, not real numbers"),
    (np.ones(3000, complex), ": the samples are complex128, not real numbers"),
    (np.zeros((4, 4, 4)), r" takes \(frames,\) or \(frames, channels\), got shape \(4, 4, 4\)"),
    (np.zeros((500, 0)), r" got an array of shape \(500, 0\), with no channel in it"),
    (np.float32(1), r" takes \(frames,\) or \(frames, channels\), got shape \(\)")])
def test_what_has_no_silence_to_cut_is_refused_by_name(call, y, says):
    with pytest.raises(kudio.FeatureError, match=rf"^{call.__name__}\(\)" + says):
        call(y)


@pytest.mark.parametrize("call", [kudio.trim_silence, kudio.split_on_silence],
                         ids=lambda f: f.__name__)
@pytest.mark.parametrize("size", ['frame_length', 'hop_length'])
@pytest.mark.parametrize("value", [0, -1, 2.5, float('nan'), True, '512', None], ids=repr)
def test_a_frame_that_is_not_a_number_of_samples_is_refused(call, size, value):
    with pytest.raises(kudio.FeatureError) as caught:
        call(np.zeros(3000), **{size: value})
    assert str(caught.value) == (f"{call.__name__}(): {size} must be >= 1, a whole number of "
                                 f"samples, got {value!r}")
    assert kudio.trim_silence(_tone(), **{size: 512.0})[1] == kudio.trim_silence(
        _tone(), **{size: 512})[1]


def test_a_long_clip_is_not_held_as_frames():
    """At 2048 and 512 its frames are four times the clip."""
    import tracemalloc
    y = (0.1 * np.random.default_rng(0).standard_normal(48000 * 60)).astype(np.float32)
    tracemalloc.start()
    try:
        kudio.trim_silence(y)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 2.2 * y.nbytes + 8 * 2 ** 20, f"{peak / y.nbytes:.1f} times the clip"
