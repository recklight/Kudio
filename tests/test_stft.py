# -*- coding: utf-8 -*-
"""STFT geometry object: shapes, round trip, and storage."""
import numpy as np
import pytest

from kudio import STFT, waveform_to_spectrogram
from conftest import SR, make_sine


def test_win_length_defaults_to_n_fft():
    assert STFT(n_fft=256).win_length == 256
    assert STFT(n_fft=512, win_length=400).win_length == 400


def test_n_bins_and_frames():
    stft = STFT(sr=16000, n_fft=512, hop_length=256)
    assert stft.n_bins == 257

    spec = stft.forward(make_sine(seconds=1.0))
    assert spec.shape == (stft.n_frames(SR), stft.n_bins)


def test_frame_times_are_hop_spaced():
    stft = STFT(sr=8000, n_fft=256, hop_length=128)
    times = stft.frame_times(4)
    assert np.allclose(times, [0.0, 0.016, 0.032, 0.048])


@pytest.mark.parametrize("bad, match", [
    (dict(n_fft=256, win_length=512), "win_length"),
    (dict(hop_length=0), "hop_length"),
])
def test_invalid_geometry_is_rejected(bad, match):
    with pytest.raises(ValueError, match=match):
        STFT(**bad)


def test_forward_matches_the_underlying_function(sine):
    stft = STFT(sr=SR, n_fft=512, hop_length=256)
    assert np.allclose(
        stft.forward(sine),
        waveform_to_spectrogram(sine, n_fft=512, hop_length=256,
                                win_length=512, window='hamming'))


def test_round_trip_reconstructs_the_waveform(sine):
    stft = STFT(sr=SR)
    back = stft.inverse(sine, stft.forward(sine))

    assert len(back) == len(sine)
    assert np.corrcoef(back, sine)[0, 1] > 0.99


def test_context_widens_the_frames(sine):
    stft = STFT(sr=SR, n_fft=256, hop_length=128)
    plain = stft.forward(sine)
    stacked = stft.forward(sine, context=2)

    assert stacked.shape == (len(plain), stft.n_bins * 5)


def test_to_dict_from_dict_round_trips():
    stft = STFT(sr=8000, n_fft=256, hop_length=64, window='hann')
    assert STFT.from_dict(stft.to_dict()) == stft


def test_from_dict_rejects_unknown_keys():
    with pytest.raises(ValueError, match="n_ftt"):
        STFT.from_dict({'sr': 16000, 'n_ftt': 512})


def test_is_frozen_so_it_can_be_stored_safely():
    stft = STFT()
    with pytest.raises(Exception):
        stft.n_fft = 1024


def test_matches_checks_the_rate():
    stft = STFT(sr=16000)
    assert stft.matches(16000) is True
    assert stft.matches(8000) is False


# -- frame counts, stored geometry and the tail --------------------------------

@pytest.mark.parametrize("n_fft, n", [(512, 32000), (511, 32000), (511, 16384), (400, 999)])
def test_n_frames_counts_what_forward_returns(n_fft, n):
    """n // hop + 1 is right for an even n_fft only: by it STFT(n_fft=511) would
    promise 126 frames for 32000 samples and give 125."""
    stft = STFT(sr=SR, n_fft=n_fft, hop_length=256)
    y = np.random.default_rng(0).standard_normal(n).astype(np.float32)
    assert stft.n_frames(n) == len(stft.forward(y))


def test_from_dict_wants_every_key():
    """A stored geometry missing hop_length would analyse at 256 without a word."""
    with pytest.raises(ValueError, match="missing: hop_length"):
        STFT.from_dict({'sr': 16000, 'n_fft': 512, 'win_length': 512, 'window': 'hann'})


def test_inverse_rebuilds_the_tail_too():
    """The last len(y) % hop samples are the signal, not one value repeated."""
    stft = STFT(sr=SR, n_fft=512, hop_length=256)
    y = make_sine(seconds=1.0)                    # 16000 % 256 = 128 samples of tail
    back = stft.inverse(y, stft.forward(y))
    assert len(back) == len(y)
    assert np.max(np.abs(back[-128:] - y[-128:])) < 0.01
    assert len(np.unique(np.round(back[-128:], 6))) > 10


def test_inverse_refuses_a_spectrogram_of_the_wrong_shape():
    from kudio.exceptions import FeatureError
    stft = STFT(sr=SR, n_fft=512, hop_length=256)
    y = make_sine(seconds=1.0)
    with pytest.raises(FeatureError, match="one row per frame"):
        stft.inverse(y, stft.forward(y, context=2))


# -- settings that are none, and spectra that cannot be inverted --------------------
@pytest.mark.parametrize("bad", [
    dict(n_fft=0), dict(n_fft=-512), dict(n_fft=512.5), dict(n_fft=True), dict(n_fft='512'),
    dict(n_fft=None), dict(n_fft=float('nan')), dict(hop_length=0), dict(hop_length=-1),
    dict(hop_length=128.5), dict(hop_length=None), dict(hop_length=float('inf')),
    dict(win_length=0), dict(win_length=-5), dict(win_length=100.5), dict(win_length='400')])
def test_a_size_that_is_not_a_whole_number_of_samples_is_refused(bad):
    name, = bad
    with pytest.raises(ValueError, match=rf"^STFT\(\): {name} must be >= 1, a whole number"):
        STFT(**bad)


def test_numpy_sizes_are_stored_as_the_ints_they_are():
    """The dict goes beside a checkpoint, as JSON or YAML."""
    import json
    stft = STFT(sr=np.float64(16000), n_fft=np.int64(512), hop_length=np.int32(128),
                win_length=400.0)
    assert stft == STFT(16000, 512, 128, 400)
    assert json.loads(json.dumps(stft.to_dict())) == STFT(16000, 512, 128, 400).to_dict()


def test_synthesise_rebuilds_what_analyse_gave(sine):
    for hop, window in ((128, 'hann'), (256, 'hamming'), (512, 'hamming'), (512, 'boxcar')):
        stft = STFT(sr=SR, n_fft=512, hop_length=hop, window=window)
        back = stft.synthesise(stft.analyse(sine[:512 * 30]), length=512 * 30)
        assert np.max(np.abs(back - sine[:512 * 30])) < 1e-3


@pytest.mark.parametrize("wrong, said", [
    (lambda s: s.T, r"takes \(frames, 257\), the layout analyse\(\) gives, and got shape"),
    (lambda s: s[:0], r"takes \(frames, 257\)"),
    (lambda s: s[:, :-1], r"got shape \(63, 256\)"),
    (lambda s: s[0], r"got shape \(257,\)"),
    (lambda s: s[None], r"got shape \(1, 63, 257\)"),
    (np.abs, "takes the complex spectrum .* got float32: a magnitude has no phase"),
    (lambda s: s.real, "takes the complex spectrum"),
])
def test_a_spectrum_of_the_wrong_kind_is_not_inverted(sine, wrong, said):
    """The librosa layout came out as a burst of noise, and a magnitude as a
    waveform with every phase zero."""
    from kudio.exceptions import FeatureError
    stft = STFT(sr=SR, n_fft=512, hop_length=256)
    spec = stft.analyse(sine)
    with pytest.raises(FeatureError, match=r"^STFT\.synthesise\(\) .*" + said):
        stft.synthesise(wrong(spec), length=len(sine))


@pytest.mark.parametrize("hop", [512, 513, 1000])
def test_samples_under_no_window_cannot_be_rebuilt(sine, hop):
    """A Hann window is zero at its first sample, and past its length there is
    no window at all."""
    from kudio.exceptions import FeatureError
    stft = STFT(sr=SR, n_fft=512, hop_length=hop, window='hann')
    with pytest.raises(FeatureError, match=f"with hop_length={hop} some samples are under no part"):
        stft.synthesise(stft.analyse(sine), length=len(sine))
    assert len(stft.forward(sine)) == stft.n_frames(len(sine)), "a feature need not be invertible"


@pytest.mark.parametrize("length", [-1, 2.5, float('nan'), True, '100'], ids=repr)
def test_a_length_to_rebuild_is_a_number_of_samples(sine, length):
    from kudio.exceptions import FeatureError
    stft = STFT(sr=SR, n_fft=512, hop_length=128, window='hann')
    with pytest.raises(FeatureError) as caught:
        stft.synthesise(stft.analyse(sine), length=length)
    assert str(caught.value) == (f"STFT.synthesise(): length is a whole number of samples, 0 or "
                                 f"more, got {length!r}")


def test_no_samples_are_no_samples(sine):
    """0 is a length, not a length that is not given: no samples come back."""
    stft = STFT(sr=SR, n_fft=512, hop_length=128, window='hann')
    spec = stft.analyse(sine)
    assert stft.synthesise(spec, length=0).shape == (0,)
    assert np.array_equal(stft.synthesise(spec, length=np.float64(100)),
                          stft.synthesise(spec, length=100))
    assert len(stft.synthesise(spec)) == 128 * (len(spec) - 1)


def test_a_spectrum_with_nan_in_it_is_not_inverted(sine):
    """One NaN is every sample of the frames it is in, and the next thing to
    hear of it is whoever writes the file."""
    from kudio.exceptions import FeatureError
    stft = STFT(sr=SR, n_fft=512, hop_length=128, window='hann')
    spec = stft.analyse(sine)
    spec[7, 3] = np.nan
    with pytest.raises(FeatureError) as caught:
        stft.synthesise(spec)
    assert str(caught.value) == ("STFT.synthesise() got 1 value(s) that are not finite (NaN or "
                                 "inf)")


def test_the_window_can_be_anything_the_transform_takes(sine):
    """By name with its parameter, or as a Kaiser beta: both are what
    scipy.signal.get_window takes, and to_dict writes them as they are."""
    for window in (('kaiser', 8.6), 8.6, 'blackmanharris'):
        stft = STFT(sr=SR, n_fft=512, hop_length=128, window=window)
        back = stft.synthesise(stft.analyse(sine), length=len(sine))
        assert np.max(np.abs(back - sine)[:-128]) < 1e-6
        assert STFT.from_dict(stft.to_dict()) == stft
