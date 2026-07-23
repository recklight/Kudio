# -*- coding: utf-8 -*-
import numpy as np

import warnings

from kudio import (
    features2matrix,
    logspec_from_files,
    melspectrogram,
    mfcc as w2mfcc,
    mfcc_from_files,
    spectrogram_to_waveform as spec2wavform,
    waveform_to_spectrogram as w2s,
    wave_separate,
    wave_slicing,
)


def test_wave_separate_roundtrip():
    interleaved = np.arange(12)
    chans = wave_separate(interleaved, 2)
    assert len(chans) == 2
    np.testing.assert_array_equal(chans[0], [0, 2, 4, 6, 8, 10])
    np.testing.assert_array_equal(chans[1], [1, 3, 5, 7, 9, 11])


def test_w2s_shapes(sine):
    spec = w2s(sine)
    assert spec.ndim == 2
    assert spec.shape[1] == 257  # n_fft/2 + 1

    seq = w2s(sine, sequence=True)
    assert seq.shape == (1, *spec.shape)

    ctx = w2s(sine, forward_backward=2)
    assert ctx.shape == (spec.shape[0], 5 * 257)

    ctx_seq = w2s(sine, sequence=True, forward_backward=2)
    assert ctx_seq.shape == (1, spec.shape[0], 5 * 257)


def test_deprecated_alias_still_works(sine):
    import kudio
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        np.testing.assert_array_equal(kudio.wavform2spec(sine), w2s(sine))


def test_w2s_norm(sine):
    spec = w2s(sine, norm=True)
    np.testing.assert_allclose(spec.mean(axis=0), 0, atol=1e-6)


def test_spec2wavform_roundtrip(sine):
    spec = w2s(sine)  # un-normalized log power spectrogram
    y_rec = spec2wavform(sine, spec)
    assert len(y_rec) == len(sine)
    # reconstruction with original phase should correlate strongly
    assert np.corrcoef(y_rec, sine)[0, 1] > 0.99


def test_w2mfcc_shape(sine):
    mfcc = w2mfcc(sine, 16000, n_mfcc=40)
    assert mfcc.ndim == 3
    assert mfcc.shape[0] == 1 and mfcc.shape[2] == 40


def test_mfcc_from_files(wav_dir):
    files = sorted(wav_dir.glob("*.wav"))
    out = mfcc_from_files(files, n_mfcc=20)
    assert out.shape[0] == 3 and out.shape[2] == 20


def test_logspec_from_files(wav_dir):
    files = sorted(wav_dir.glob("*.wav"))
    out = logspec_from_files(files, False, 0, False)
    assert out.ndim == 2 and out.shape[1] == 257


def test_wave_slicing(sine):
    frames = wave_slicing(sine, frame_length=2048, hop_length=512, is_reshape=False)
    assert frames.shape[0] == 2048
    flat = wave_slicing(sine, frame_length=2048, hop_length=512)
    assert flat.ndim == 1


def test_features2matrix():
    a = np.ones((3, 4))
    b = np.zeros((2, 4))
    mat, labels = features2matrix([a, b])
    assert mat.shape == (5, 4)
    np.testing.assert_array_equal(labels, [0, 0, 0, 1, 1])


def test_melspectrogram(wav_file):
    vec = melspectrogram(wav_file, n_mels=32, n_frames=4)
    assert vec.shape[1] == 32 * 4
    assert vec.shape[0] > 0
