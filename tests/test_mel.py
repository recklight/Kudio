# -*- coding: utf-8 -*-
"""Mel bands, decibels and the cepstrum, against what they are defined to be."""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import kudio
from kudio.core import _mel

SR = 16000


# -- the scale --------------------------------------------------------------------

@pytest.mark.parametrize("hz, mel", [
    (0.0, 0.0), (200.0 / 3, 1.0), (500.0, 7.5), (1000.0, 15.0), (6400.0, 42.0),
    (1000.0 * 6.4 ** (1 / 27), 16.0), (40960.0, 69.0)])
def test_the_scale_is_slaneys(hz, mel):
    """200/3 Hz a mel up to 1 kHz, and 27 mel to a factor of 6.4 above."""
    assert _mel._to_mel(hz) == pytest.approx(mel, abs=1e-12)
    assert _mel._to_hz(np.array([mel])) == pytest.approx([hz], abs=1e-9 * max(hz, 1))


def test_the_bands_are_evenly_spaced_in_mel():
    at = kudio.mel_band_hz(64, SR)
    assert at.shape == (64,) and at.dtype == np.float64
    mels = np.array([_mel._to_mel(f) for f in at])
    step = _mel._to_mel(SR / 2) / 65
    assert mels == pytest.approx(step * np.arange(1, 65), abs=1e-12)
    # 10 of the mels below 15 are 200/3 Hz apart, and above it the ratio is constant
    low = at[at < 1000.0]
    assert np.diff(low) == pytest.approx(step * 200 / 3)
    high = at[at > 1000.0]
    assert high[1:] / high[:-1] == pytest.approx(6.4 ** (step / 27))


def test_the_frequencies_of_the_bands_by_hand():
    """Four bands to 8 kHz: 8 kHz is 15 + 27 ln 8 / ln 6.4 = 45.245 mel, so the
    bands are 9.049 mel apart and the first is 603.27 Hz."""
    top = 15 + 27 * np.log(8) / np.log(6.4)
    want = [200 / 3 * top / 5] + [1000 * 6.4 ** ((k * top / 5 - 15) / 27) for k in (2, 3, 4)]
    assert kudio.mel_band_hz(4, SR) == pytest.approx(want, rel=1e-12)
    assert kudio.mel_band_hz(4, SR).round(1).tolist() == [603.3, 1237.4, 2305.2, 4294.3]


def test_the_limits_are_the_feet_of_the_outer_bands():
    at = kudio.mel_band_hz(10, 48000, fmin=300.0, fmax=3400.0)
    assert 300 < at[0] and at[-1] < 3400
    same = kudio.SpectrogramStream(48000, n_fft=2048, n_mels=10, fmin=300.0, fmax=3400.0)
    assert np.array_equal(same.freqs, at)
    assert np.array_equal(kudio.mel_band_hz(10, 48000, fmax=24000), kudio.mel_band_hz(10, 48000))


@pytest.mark.parametrize("value", [0, -1, 2.5, float('nan'), True, '4', None], ids=repr)
def test_a_number_of_bands_is_a_whole_number(value):
    with pytest.raises(kudio.FeatureError) as caught:
        kudio.mel_band_hz(value, SR)
    assert str(caught.value) == f"mel_band_hz(): n_mels must be >= 1, a whole number, got {value!r}"
    assert len(kudio.mel_band_hz(4.0, SR)) == 4


@pytest.mark.parametrize("fmin, fmax", [(-1, 8000), (4000, 4000), (5000, 4000), (float('nan'), 8000),
                                        (0, float('inf')), ('0', 8000), (None, 8000), (True, 8000),
                                        (8000, None), (0, 0)], ids=repr)
def test_limits_that_are_not_a_range_are_refused(fmin, fmax):
    with pytest.raises(kudio.FeatureError, match=r"^mel_band_hz\(\): need 0 <= fmin < fmax"):
        kudio.mel_band_hz(4, SR, fmin=fmin, fmax=fmax)


# -- the filters --------------------------------------------------------------------

@pytest.mark.parametrize("sr, n_fft, n_mels, fmin, fmax", [
    (16000, 8192, 40, 0.0, 8000.0), (48000, 16384, 64, 20.0, 20000.0), (8000, 4096, 23, 0.0, 4000.0),
    (44100, 32768, 128, 0.0, 22050.0)])
def test_a_band_is_a_triangle_of_area_one(sr, n_fft, n_mels, fmin, fmax):
    """Over the bins, which are df apart, and df times the sum is the area under a
    triangle to within a bin at each of its corners."""
    f = _mel.filters(sr, n_fft, n_mels, fmin, fmax)
    assert f.shape == (n_mels, n_fft // 2 + 1) and f.dtype == np.float32 and (f >= 0).all()
    df = sr / n_fft
    edges = np.r_[fmin, kudio.mel_band_hz(n_mels, sr, fmin=fmin, fmax=fmax), fmax]
    widths = edges[2:] - edges[:-2]
    assert np.all(np.abs(f.sum(axis=1) * df - 1) < 2 * df / widths)
    bins = np.arange(n_fft // 2 + 1) * df
    for band, row in enumerate(f):
        low, peak, high = edges[band:band + 3]
        assert not row[(bins <= low - 1e-6) | (bins >= high + 1e-6)].any(), "nothing outside its feet"
        assert row.max() <= 2 / (high - low) * (1 + 1e-6), "no higher than its area allows"
        want = 2 / (high - low) * np.interp(bins, [low, peak, high], [0, 1, 0])
        assert np.max(np.abs(row - want)) < 1e-6 * 2 / (high - low) + 1e-9


def test_neighbouring_triangles_add_up_to_one():
    """Unscaled: where one falls the next one rises."""
    f = _mel.filters(SR, 8192, 40).astype(np.float64)
    edges = np.r_[0.0, kudio.mel_band_hz(40, SR), 8000.0]
    unscaled = f * ((edges[2:] - edges[:-2]) / 2)[:, None]
    bins = np.arange(4097) * SR / 8192
    inside = (bins >= edges[1]) & (bins <= edges[-2])
    assert unscaled.sum(axis=0)[inside] == pytest.approx(np.ones(inside.sum()), abs=1e-5)


@pytest.mark.parametrize("sr, n_fft, n_mels, fmax, empty, says", [
    (192000, 2048, 128, None, 3, ""), (16000, 64, 40, None, None, ""),
    (16000, 2048, 40, 24000.0, None, ", or hold fmax=24000 Hz to half the rate, 8000 Hz")])
def test_bands_with_no_bin_are_counted_at_the_callers_line(sr, n_fft, n_mels, fmax, empty, says):
    y = np.random.default_rng(0).standard_normal(max(sr, 4 * n_fft)).astype(np.float32)
    with pytest.warns(UserWarning) as caught:
        kudio.mfcc(y, sr, n_mfcc=13, n_fft=n_fft, n_mels=n_mels, fmax=fmax)
    said, = caught
    count = int(np.count_nonzero(_quiet_filters(sr, n_fft, n_mels, fmax).max(axis=1) == 0))
    assert empty in (None, count) and count > 0
    assert str(said.message) == (f"mfcc(): {count} of {n_mels} mel bands hold no FFT bin and stay "
                                 f"empty; raise n_fft or lower n_mels{says}")
    assert said.filename == __file__


def _quiet_filters(sr, n_fft, n_mels, fmax):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _mel.filters(sr, n_fft, n_mels, 0.0, fmax)


def test_the_bands_nothing_is_wrong_with_are_not_spoken_of():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for sr, n_fft, n_mels in ((16000, 2048, 128), (48000, 1024, 64), (8000, 256, 23)):
            _mel.filters(sr, n_fft, n_mels)
        kudio.mfcc(np.zeros(4096, np.float32), SR)
        kudio.melspectrogram(np.zeros(4096, np.float32), sr=SR)
        kudio.SpectrogramStream(SR, n_mels=40)


def test_a_flat_spectrum_is_the_same_in_every_band():
    """The bands are of area 1, so a spectrum of 1 everywhere has 1 / df in each
    of them, as far as the triangle is drawn by the bins."""
    f = _mel.filters(SR, 16384, 40)
    bands = np.einsum('ft,mf->mt', np.ones((8193, 3), np.float32), f)
    assert bands == pytest.approx(np.full((40, 3), 16384 / SR), rel=0.03)


def test_a_click_in_the_middle_of_a_frame_is_flat():
    """One sample under the peak of the window has the same power in every bin:
    the window's value there, 1, squared. Its band is the sum of the filter."""
    y = np.zeros(8 * 1024, np.float32)
    y[4 * 1024] = 0.5
    got = _mel.power(y, SR, 1024, 512, 40, 2.0)[:, 8]
    assert got == pytest.approx(0.25 * _mel.filters(SR, 1024, 40).sum(axis=1), rel=1e-5)


@pytest.mark.parametrize("band", [3, 12, 25, 36])
def test_a_tone_at_the_centre_of_a_band_is_loudest_there(band):
    """Counted as each band's share of what it lets through, or a wide band of
    area 1 loses to a narrow one beside it."""
    hz = kudio.mel_band_hz(40, SR)[band]
    y = (0.1 * np.sin(2 * np.pi * hz * np.arange(SR) / SR)).astype(np.float32)
    mean = kudio.melspectrogram(y, n_mels=40, n_frames=1, n_fft=4096, sr=SR).mean(axis=0)
    assert int(np.argmax(mean)) == band


# -- decibels -----------------------------------------------------------------------

def test_decibels_by_hand():
    x = np.array([[1.0, 10.0, 0.1], [1e-12, 0.0, 2.0]])
    assert _mel.decibels(x) == pytest.approx(
        np.array([[0, 10, -10], [-100, -100, 10 * np.log10(2)]]), abs=1e-12)
    assert _mel.decibels(x, ref=10.0)[0] == pytest.approx([-10, 0, -20])
    assert _mel.decibels(x, floor=1e-5)[1] == pytest.approx([-50, -50, 10 * np.log10(2)])
    assert _mel.decibels(x, range_db=15.0) == pytest.approx(
        np.array([[0, 10, -5], [-5, -5, 10 * np.log10(2)]]), abs=1e-12)
    assert _mel.decibels(x.astype(np.float32)).dtype == np.float32
    assert _mel.decibels(x.astype(np.float32), ref=np.float32(3)).dtype == np.float32


# -- the cepstrum ---------------------------------------------------------------------

def _dct_by_definition(column, n):
    """The orthonormal DCT-II written out."""
    size = len(column)
    k = np.arange(n)[:, None]
    basis = np.cos(np.pi * k * (np.arange(size) + 0.5) / size)
    scale = np.where(k == 0, np.sqrt(1 / size), np.sqrt(2 / size))
    return (scale * basis) @ column


def test_the_cepstrum_is_the_transform_written_out():
    bands = np.random.default_rng(0).uniform(-80, 0, (40, 7))
    got = _mel.cepstrum(bands, 13)
    assert got.shape == (13, 7)
    assert got == pytest.approx(_dct_by_definition(bands, 13), abs=1e-10)


def test_a_flat_column_is_all_in_the_first_coefficient():
    got = _mel.cepstrum(np.full((64, 1), -30.0), 64)[:, 0]
    assert got[0] == pytest.approx(-30.0 * np.sqrt(64))
    assert got[1:] == pytest.approx(np.zeros(63), abs=1e-12)


def test_an_mfcc_by_definition():
    """From the transform of one frame, the triangles, the logarithm and the
    cosines, each worked out here."""
    from scipy.signal import get_window
    y = (0.2 * np.random.default_rng(1).standard_normal(16000)).astype(np.float64)
    n_fft, hop, n_mels = 1024, 256, 40
    padded = np.r_[np.zeros(n_fft // 2), y, np.zeros(n_fft // 2)]
    frames = np.stack([padded[t * hop:t * hop + n_fft] for t in range(1 + len(y) // hop)])
    power = np.abs(np.fft.rfft(frames * get_window('hann', n_fft), axis=1)) ** 2
    edges = np.r_[0.0, kudio.mel_band_hz(n_mels, SR), 8000.0]
    bins = np.arange(n_fft // 2 + 1) * SR / n_fft
    triangles = np.stack([2 / (edges[m + 2] - edges[m]) * np.interp(bins, edges[m:m + 3], [0, 1, 0])
                          for m in range(n_mels)])
    db = 10 * np.log10(np.maximum(power @ triangles.T, 1e-10)).T
    db = np.maximum(db, db.max() - 80)
    want = _dct_by_definition(db, 20).T
    got = kudio.mfcc(y, SR, 20, n_fft=n_fft, hop_length=hop, n_mels=n_mels)[0]
    assert got == pytest.approx(want, abs=1e-4)


def test_silence_has_one_coefficient():
    """Every band at the floor, -100 dB."""
    got = kudio.mfcc(np.zeros(8000, np.float32), SR, 13)[0]
    assert got[:, 0] == pytest.approx(np.full(len(got), -100 * np.sqrt(128)), rel=1e-6)
    assert np.max(np.abs(got[:, 1:])) < 1e-4


# -- what they are given -----------------------------------------------------------------

#: every function that takes a waveform into mel bands, and the name it answers to
BANDS = {
    'mfcc': lambda y, **k: kudio.mfcc(y, SR, **k),
    'melspectrogram': lambda y, **k: kudio.melspectrogram(y, sr=SR, **k),
}


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("name", BANDS)
def test_samples_that_are_not_finite_are_refused_by_name(name, bad):
    y = np.full(8000, 0.1)
    y[7:9] = bad
    with pytest.raises(kudio.FeatureError) as caught:
        BANDS[name](y)
    assert str(caught.value) == (f"{name}() got 2 sample(s) that are not finite (NaN or inf); "
                                 f"repair the clip first")


@pytest.mark.parametrize("value", [0, -1, 2.5, float('nan'), float('inf'), True, '40', [40]], ids=repr)
@pytest.mark.parametrize("name, size, of", [
    ('mfcc', 'n_fft', ' of samples'), ('mfcc', 'hop_length', ' of samples'), ('mfcc', 'n_mels', ''),
    ('mfcc', 'n_mfcc', ''), ('melspectrogram', 'n_fft', ' of samples'),
    ('melspectrogram', 'hop_length', ' of samples'), ('melspectrogram', 'n_mels', ''),
    ('melspectrogram', 'n_frames', '')])
def test_a_size_that_is_not_a_whole_number_is_refused(name, size, of, value):
    with pytest.raises(kudio.FeatureError) as caught:
        BANDS[name](np.zeros(8000, np.float32), **{size: value})
    assert str(caught.value) == f"{name}(): {size} must be >= 1, a whole number{of}, got {value!r}"


@pytest.mark.parametrize("ref", [lambda x: x.max(), lambda x: x.max(keepdims=True),
                                 lambda x: float(x.max()), lambda x: np.float64(x.max())],
                         ids=["a float32", "an array", "a Python float", "a float64"])
def test_the_largest_is_0_db_whatever_kind_of_number_it_is_given_as(ref):
    """numpy 1.24 takes the logarithm of a float32 number in float64, and that of the
    float32 values in float32: 3e-6 dB between them."""
    values = np.random.default_rng(0).random((40, 100)).astype(np.float32) ** 2
    out = _mel.decibels(values, ref=ref(values), range_db=80.0)
    assert out.dtype == np.float32 and out.max() == 0


@pytest.mark.parametrize("value", [40.0, np.int64(40), np.float32(40), np.array(40)], ids=repr)
def test_a_whole_number_of_any_type_is_that_size(value):
    y = np.random.default_rng(0).standard_normal(8000).astype(np.float32)
    assert np.array_equal(kudio.mfcc(y, SR, 13, n_mels=value), kudio.mfcc(y, SR, 13, n_mels=40))
    assert np.array_equal(kudio.melspectrogram(y, value, sr=SR), kudio.melspectrogram(y, 40, sr=SR))


@pytest.mark.parametrize("value", [0, -1.0, float('nan'), float('inf'), True, '2', None], ids=repr)
def test_an_exponent_is_a_number_above_0(value):
    with pytest.raises(kudio.FeatureError) as caught:
        kudio.melspectrogram(np.zeros(8000), sr=SR, power=value)
    assert str(caught.value) == f"melspectrogram(): power is an exponent above 0, got {value!r}"


@pytest.mark.parametrize("value", [0, -1.0, float('nan'), float('inf'), True, '8000'], ids=repr)
def test_an_upper_limit_is_a_frequency(value):
    with pytest.raises(kudio.FeatureError) as caught:
        kudio.mfcc(np.zeros(8000), SR, fmax=value)
    assert str(caught.value) == f"mfcc(): fmax is a frequency in hertz above 0, got {value!r}"


def test_bytes_are_not_a_waveform_without_a_rate():
    with pytest.raises(kudio.FeatureError, match="takes a path as text, or a waveform, and got bytes"):
        kudio.melspectrogram(b'clip.wav')


@pytest.mark.parametrize("given, says", [
    (dict(desired_samples=-8000), "desired_samples is a whole number of samples to keep, 1 or "
                                  "more, or None for all of them, got -8000"),
    (dict(desired_samples=2.5), "desired_samples is a whole number of samples to keep, 1 or more, "
                                "or None for all of them, got 2.5"),
    (dict(desired_samples=0), "desired_samples is a whole number of samples to keep, 1 or more, "
                              "or None for all of them, got 0"),
    (dict(desired_length=-0.5), "desired_length is a number of seconds to keep, above 0, or None "
                                "for all of them, got -0.5"),
    (dict(desired_length=0), "desired_length is a number of seconds to keep, above 0, or None "
                             "for all of them, got 0"),
    (dict(desired_length=float('nan')), "desired_length is a number of seconds to keep, above 0, "
                                        "or None for all of them, got nan"),
    (dict(desired_length='1'), "desired_length is a number of seconds to keep, above 0, or None "
                               "for all of them, got '1'")])
def test_a_length_to_cut_files_to_is_not_negative(wav_file, given, says):
    """-8000 samples and -0.5 s cut the end off each file instead."""
    with pytest.raises(kudio.FeatureError) as caught:
        kudio.mfcc_from_files([wav_file], **given)
    assert str(caught.value) == f"mfcc_from_files(): {says}"


def test_the_cut_of_each_file(wav_file):
    whole = kudio.mfcc_from_files([wav_file])
    assert whole.shape == (1, 32, 40)
    for given in (dict(desired_samples=None), dict(desired_length=None),
                  dict(desired_samples=99999), dict(desired_length=9.5)):
        assert np.array_equal(kudio.mfcc_from_files([wav_file], **given), whole), given
    half = kudio.mfcc_from_files([wav_file], desired_samples=8000.0)
    assert half.shape == (1, 16, 40)
    assert np.array_equal(kudio.mfcc_from_files([wav_file], desired_length=0.5), half)


def test_no_files_and_files_of_two_lengths_are_said(tmp_path, wav_file, sine):
    for call in (lambda: kudio.mfcc_from_files([]),
                 lambda: kudio.logspec_from_files([], False, 0, False)):
        with pytest.raises(kudio.FeatureError, match=r"_from_files\(\) got no files$"):
            call()
    short = tmp_path / "short.wav"
    kudio.save_wave(short, sine[:8000], SR)
    with pytest.raises(kudio.FeatureError) as caught:
        kudio.mfcc_from_files([wav_file, short])
    assert str(caught.value) == (f"mfcc_from_files(): {short} gives 16 frames and the first file 32, "
                                 f"and they are stacked; cut every file to one length")


# -- the stream ------------------------------------------------------------------------------

@pytest.mark.parametrize("hop", [512, 513, 1000, 2048, 5000])
@pytest.mark.parametrize("n_mels", [0, 40])
@pytest.mark.parametrize("pieces", [1, 2, 7, 40, 1000])
def test_the_blocks_a_stream_comes_in_change_nothing(hop, n_mels, pieces):
    """With a hop longer than the frame the samples between two columns are
    skipped in full, whatever blocks they arrive in; skipped only as far as
    they have arrived, 7 columns at once would be 13 in 40 blocks."""
    y = 0.1 * np.random.default_rng(0).standard_normal(20000)
    stream = kudio.SpectrogramStream(SR, n_fft=512, hop_length=hop, n_mels=n_mels, seconds=9)
    got = np.concatenate([stream.push(b) for b in np.array_split(y, pieces)])
    by_hand = kudio.SpectrogramStream(SR, n_fft=512, hop_length=512, n_mels=n_mels)
    want = np.concatenate([by_hand.push(y[s:s + 512]) for s in range(0, len(y) - 511, hop)])
    assert got.shape == want.shape and np.array_equal(got, want)
    assert stream.frames_seen == len(want)


def test_a_stream_reset_between_two_columns_starts_at_the_next_block():
    stream = kudio.SpectrogramStream(SR, n_fft=512, hop_length=5000)
    y = 0.1 * np.random.default_rng(0).standard_normal(600)
    first = stream.push(y)
    assert len(first) == 1 and len(stream.push(y)) == 0
    stream.reset()
    assert np.array_equal(stream.push(y), first)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_a_block_that_is_not_finite_is_refused_and_forgotten(bad):
    y = 0.1 * np.random.default_rng(0).standard_normal(4000)
    plain, stream = kudio.SpectrogramStream(SR), kudio.SpectrogramStream(SR)
    want = np.concatenate([plain.push(y[:2000]), plain.push(y[2000:])])
    out = [stream.push(y[:2000])]
    block = y[2000:2300].copy()
    block[3] = bad
    with pytest.raises(kudio.FeatureError, match=r"^SpectrogramStream\.push\(\) got 1 sample\(s\)"):
        stream.push(block)
    out.append(stream.push(y[2000:]))
    assert np.array_equal(np.concatenate(out), want)


@pytest.mark.parametrize("given, says", [
    (dict(n_fft=2.5), "n_fft must be >= 1, a whole number of samples, got 2.5"),
    (dict(n_fft=1), "n_fft must be >= 2, got 1"),
    (dict(hop_length=0), "hop_length must be >= 1, a whole number of samples, got 0"),
    (dict(hop_length=True), "hop_length must be >= 1, a whole number of samples, got True"),
    (dict(n_mels=-3), "n_mels is a whole number of mel bands, or 0 for a linear axis, got -3"),
    (dict(n_mels=2.5), "n_mels is a whole number of mel bands, or 0 for a linear axis, got 2.5"),
    (dict(n_mels=False), "n_mels is a whole number of mel bands, or 0 for a linear axis, got "
                         "False"),
    (dict(seconds=float('nan')), "seconds must be positive, a finite number, got nan"),
    (dict(top_db=float('inf')), "top_db must be positive, a finite number, got inf"),
    (dict(ref='1'), "ref must be positive, a finite number, got '1'"),
    (dict(n_mels=40, fmin=-1), "need 0 <= fmin < fmax, in hertz, got fmin=-1 and fmax=8000.0"),
    (dict(n_mels=40, fmax=float('nan')), "need 0 <= fmin < fmax, in hertz, got fmin=0.0 and fmax=nan"),
    (dict(n_mels=40, fmin=5000, fmax=4000), "need 0 <= fmin < fmax, in hertz, got fmin=5000 and fmax=4000")])
def test_a_setting_that_is_not_a_number_is_refused(given, says):
    with pytest.raises(kudio.FeatureError) as caught:
        kudio.SpectrogramStream(SR, **given)
    assert str(caught.value) == f"SpectrogramStream(): {says}"
