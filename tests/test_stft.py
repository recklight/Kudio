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
