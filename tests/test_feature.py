# -*- coding: utf-8 -*-
import numpy as np
import pytest

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


def test_melspectrogram_accepts_a_waveform(sine):
    from kudio import melspectrogram
    from conftest import SR

    feats = melspectrogram(sine, sr=SR)
    assert feats.ndim == 2
    assert feats.shape[1] == 64 * 5


def test_melspectrogram_waveform_and_file_agree(wav_file):
    """Same samples in, same features out -- whichever way they arrive.

    Compared against the file's *own* decoded samples, not the pre-quantization
    float array: writing to int16 raises the noise floor, which is a swing of
    tens of dB in the near-silent mel bins.
    """
    from kudio import file_load, melspectrogram

    y, sr = file_load(wav_file)
    from_file = melspectrogram(wav_file)
    from_array = melspectrogram(y, sr=sr)

    assert from_file.shape == from_array.shape
    assert np.allclose(from_file, from_array)


def test_melspectrogram_waveform_requires_sr(sine):
    from kudio import melspectrogram
    from kudio.exceptions import FeatureError

    with pytest.raises(FeatureError, match="needs sr="):
        melspectrogram(sine)


# -- context stacking / windowing / normalisation -------------------------------

def _naive_stack(frames, context, zero_pad):
    """Independent reference implementation, written the obvious slow way."""
    n, dim = frames.shape
    out = np.empty((n, dim * (2 * context + 1)), dtype=np.float32)
    for i in range(n):
        blocks = []
        for off in range(-context, context + 1):
            j = i + off
            if 0 <= j < n:
                blocks.append(frames[j])
            elif zero_pad:
                blocks.append(np.zeros(dim, dtype=np.float32))
            else:
                blocks.append(frames[0] if j < 0 else frames[n - 1])
        out[i] = np.concatenate(blocks)
    return out


def test_stack_context_matches_a_naive_implementation():
    from kudio import stack_context

    rng = np.random.default_rng(0)
    frames = rng.normal(size=(9, 4)).astype(np.float32)

    for context in (1, 2, 3):
        assert np.allclose(stack_context(frames, context, pad='edge'),
                           _naive_stack(frames, context, zero_pad=False))
        assert np.allclose(stack_context(frames, context, pad='zero'),
                           _naive_stack(frames, context, zero_pad=True))


def test_stack_context_zero_is_a_passthrough():
    from kudio import stack_context

    frames = np.arange(12, dtype=np.float32).reshape(3, 4)
    assert np.allclose(stack_context(frames, 0), frames)


def test_forward_backward_still_zero_pads_after_the_refactor(sine):
    """waveform_to_spectrogram delegates to stack_context; behaviour is unchanged."""
    from kudio import stack_context, waveform_to_spectrogram

    plain = waveform_to_spectrogram(sine, n_fft=256, hop_length=128)
    stacked = waveform_to_spectrogram(sine, n_fft=256, hop_length=128,
                                      forward_backward=2)

    assert stacked.shape == (len(plain), plain.shape[1] * 5)
    assert np.allclose(stacked, stack_context(plain, 2, pad='zero'))
    # the first row's leading context is silence, not a repeat of frame 0
    assert np.allclose(stacked[0, :plain.shape[1]], 0.0)


def test_forward_backward_sequence_shape(sine):
    from kudio import waveform_to_spectrogram

    seq = waveform_to_spectrogram(sine, n_fft=256, hop_length=128,
                                  forward_backward=1, sequence=True)
    assert seq.ndim == 3
    assert seq.shape[0] == 1


@pytest.mark.parametrize("n, size, expected", [
    (64, 16, 4),
    (60, 16, 4),      # tail padded, not dropped
    (5, 16, 1),       # shorter than one window still yields one
])
def test_frame_windows_pads_the_tail(n, size, expected):
    from kudio import frame_windows

    frames = np.random.default_rng(0).normal(size=(n, 7)).astype(np.float32)
    assert frame_windows(frames, size).shape == (expected, size, 7)


def test_frame_windows_preserves_the_leading_data():
    from kudio import frame_windows

    frames = np.arange(40, dtype=np.float32).reshape(10, 4)
    windows = frame_windows(frames, 4)
    assert np.allclose(windows.reshape(-1, 4)[:10], frames)


def test_feature_helpers_reject_bad_shapes():
    from kudio import frame_windows, stack_context
    from kudio.exceptions import FeatureError

    flat = np.zeros(10, dtype=np.float32)
    with pytest.raises(FeatureError, match="2-D"):
        stack_context(flat, 1)
    with pytest.raises(FeatureError, match="2-D"):
        frame_windows(flat, 4)
    with pytest.raises(FeatureError, match="context"):
        stack_context(np.zeros((4, 4), dtype=np.float32), -1)


def test_standardizer_round_trips():
    from kudio import Standardizer

    rng = np.random.default_rng(1)
    frames = rng.normal(3.0, 2.0, size=(200, 9)).astype(np.float32)

    std = Standardizer().fit(frames)
    z = std.transform(frames)
    assert np.allclose(z.mean(axis=0), 0, atol=1e-4)
    assert np.allclose(z.std(axis=0), 1, atol=1e-3)
    assert np.allclose(std.inverse(z), frames, atol=1e-3)
    assert np.allclose(Standardizer().fit_transform(frames), z)


def test_standardizer_survives_a_save_load(tmp_path):
    from kudio import Standardizer

    frames = np.random.default_rng(2).normal(size=(50, 9)).astype(np.float32)
    std = Standardizer().fit(frames)
    again = Standardizer.load(std.save(tmp_path / "stats.npz"))

    assert np.allclose(again.mean, std.mean)
    assert np.allclose(again.transform(frames), std.transform(frames))


def test_unfitted_standardizer_refuses_to_transform():
    from kudio import Standardizer
    from kudio.exceptions import FeatureError

    with pytest.raises(FeatureError, match="not been fitted"):
        Standardizer().transform(np.zeros((3, 4), dtype=np.float32))


def test_win_length_follows_n_fft(sine):
    """A non-default n_fft used to crash: win_length was pinned at 512."""
    from kudio import waveform_to_spectrogram

    spec = waveform_to_spectrogram(sine, n_fft=256, hop_length=128)
    assert spec.shape[1] == 129

    spec = waveform_to_spectrogram(sine, n_fft=1024, hop_length=256)
    assert spec.shape[1] == 513


def test_stft_object_and_function_agree_on_window(sine):
    """The wrapper must not silently use a different window than the function."""
    from kudio import STFT, waveform_to_spectrogram

    for n_fft in (256, 512, 1024):
        stft = STFT(n_fft=n_fft, hop_length=128)
        assert np.allclose(
            stft.forward(sine),
            waveform_to_spectrogram(sine, n_fft=n_fft, hop_length=128))
