# -*- coding: utf-8 -*-
import numpy as np
import pytest

import warnings

import kudio
from kudio import (
    features2matrix,
    logspec_from_files,
    melspectrogram,
    mfcc,
    mfcc_from_files,
    spectrogram_to_waveform,
    waveform_to_spectrogram,
    wave_separate,
    wave_slicing,
)


def test_wave_separate_roundtrip():
    interleaved = np.arange(12)
    chans = wave_separate(interleaved, 2)
    assert len(chans) == 2
    np.testing.assert_array_equal(chans[0], [0, 2, 4, 6, 8, 10])
    np.testing.assert_array_equal(chans[1], [1, 3, 5, 7, 9, 11])


def test_spectrogram_shapes(sine):
    spec = waveform_to_spectrogram(sine)
    assert spec.ndim == 2
    assert spec.shape[1] == 257  # n_fft/2 + 1

    seq = waveform_to_spectrogram(sine, sequence=True)
    assert seq.shape == (1, *spec.shape)

    ctx = waveform_to_spectrogram(sine, forward_backward=2)
    assert ctx.shape == (spec.shape[0], 5 * 257)

    ctx_seq = waveform_to_spectrogram(sine, sequence=True, forward_backward=2)
    assert ctx_seq.shape == (1, spec.shape[0], 5 * 257)


def test_spectrogram_norm(sine):
    spec = waveform_to_spectrogram(sine, norm=True)
    np.testing.assert_allclose(spec.mean(axis=0), 0, atol=1e-6)


def test_spectrogram_to_waveform_roundtrip(sine):
    spec = waveform_to_spectrogram(sine)  # un-normalized log power spectrogram
    y_rec = spectrogram_to_waveform(sine, spec)
    assert len(y_rec) == len(sine)
    # reconstruction with original phase should correlate strongly
    assert np.corrcoef(y_rec, sine)[0, 1] > 0.99


def test_mfcc_shape(sine):
    coeffs = mfcc(sine, 16000, n_mfcc=40)
    assert coeffs.ndim == 3
    assert coeffs.shape[0] == 1 and coeffs.shape[2] == 40


def test_mfcc_left_to_its_defaults_is_2048_512_and_128_bands(sine):
    np.testing.assert_array_equal(
        mfcc(sine, 16000, n_mfcc=20),
        mfcc(sine, 16000, n_mfcc=20, n_fft=2048, hop_length=512, n_mels=128, fmax=8000))


def test_mfcc_takes_the_analysis_of_the_mel_spectrogram():
    """At 192 kHz the 2048-point default leaves 3 of 128 bands empty and says so;
    8192 points fill them. The hop sets the number of frames."""
    y = np.random.default_rng(0).standard_normal(192000).astype(np.float32)
    with pytest.warns(UserWarning, match="3 of 128 mel bands hold no FFT bin"):
        mfcc(y, 192000)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        wide = mfcc(y, 192000, n_fft=8192, hop_length=1920)
    assert wide.shape == (1, 1 + 192000 // 1920, 40)
    assert mfcc(y, 192000, n_mels=64, fmax=24000).shape == (1, 1 + 192000 // 512, 40)


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


def test_forward_backward_zero_pads_as_stack_context_does(sine):
    """waveform_to_spectrogram stacks its context as stack_context(pad='zero') does."""
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
    """With win_length pinned at 512, a non-default n_fft would crash."""
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


# -- dtype, and what a reconstruction past full scale is written as ---------------

def test_float32_audio_gives_a_float32_spectrogram(sine):
    """NumPy 2 promoted the spectrum to complex128 through the epsilon, a
    numpy float64 scalar, doubling every spectrogram computed from float32."""
    spec = waveform_to_spectrogram(sine)
    assert spec.dtype == np.float32
    assert spectrogram_to_waveform(sine, spec).dtype == np.float32
    assert waveform_to_spectrogram(sine.astype(np.float64)).dtype == np.float64


def test_the_silence_floor_did_not_move():
    """The value a digitally silent bin gets is part of what trained models
    have seen; the dtype fix must not change it."""
    floor = waveform_to_spectrogram(np.zeros(4096, np.float32)).min()
    assert floor == pytest.approx(-31.307, abs=1e-3)


def test_save_spectrogram_as_wave_clips_rather_than_wrapping(tmp_path, sine):
    import kudio
    noisy = tmp_path / "noisy.wav"
    kudio.save_wave(noisy, sine, 16000)
    y, _ = kudio.file_load(noisy)
    louder = waveform_to_spectrogram(y) + 2.0                 # +20 dB: peaks near 5x full scale
    exact = spectrogram_to_waveform(y, louder)
    assert np.abs(exact).max() > 2.0
    out = tmp_path / "out.wav"
    kudio.save_spectrogram_as_wave(out, noisy, louder)
    written, _ = kudio.file_load(out)
    assert np.all(written[exact > 1.0] > 0.999)
    assert np.all(written[exact < -1.0] <= -0.999)


# -- context windows without stacking the corpus ---------------------------------

def _utterances():
    rng = np.random.default_rng(0)
    return [rng.standard_normal((n, 6)).astype(np.float32) for n in (9, 1, 0, 14)]


@pytest.mark.parametrize("pad", ["edge", "zero"])
def test_context_frames_is_stack_context_a_batch_at_a_time(pad):
    from kudio import ContextFrames, stack_context
    feats = _utterances()
    stacked = np.vstack([stack_context(f, 3, pad) for f in feats if len(f)])
    bank = ContextFrames(feats, context=3, pad=pad)
    assert bank.shape == stacked.shape and len(bank) == 24
    np.testing.assert_array_equal(np.asarray(bank), stacked)
    idx = np.random.default_rng(1).integers(0, len(bank), 32)
    np.testing.assert_array_equal(bank[idx], stacked[idx])
    np.testing.assert_array_equal(bank[5:12], stacked[5:12])
    np.testing.assert_array_equal(bank[7], stacked[7])
    mask = np.arange(len(bank)) % 3 == 0
    np.testing.assert_array_equal(bank[mask], stacked[mask])


def test_context_frames_never_reaches_into_the_next_utterance():
    from kudio import ContextFrames
    a = np.zeros((4, 2), np.float32)
    b = np.ones((4, 2), np.float32)
    bank = ContextFrames([a, b], context=2, pad='edge')
    assert not bank[:4].any() and bank[4:].all()


def test_context_frames_keeps_the_features_not_the_stack():
    from kudio import ContextFrames
    feats = [np.zeros((1000, 257), np.float32)] * 4
    bank = ContextFrames(feats, context=5)
    stacked_bytes = len(bank) * bank.shape[1] * 4
    assert bank._frames.nbytes < stacked_bytes / 10


def test_context_frames_refuses_bad_input():
    from kudio import ContextFrames
    from kudio.exceptions import FeatureError
    with pytest.raises(FeatureError):
        ContextFrames([np.zeros((3, 4)), np.zeros((3, 5))], context=1)
    with pytest.raises(FeatureError):
        ContextFrames([np.zeros((3, 4))], context=-1)
    with pytest.raises(FeatureError):
        ContextFrames([], context=1)
    single = ContextFrames(np.ones((3, 4), np.float32), context=1)
    assert single.shape == (3, 12)


# -- shapes, files and saved statistics ----------------------------------------

def test_standardizer_save_returns_the_file_it_wrote(tmp_path):
    from kudio import Standardizer
    std = Standardizer().fit(np.random.default_rng(0).standard_normal((50, 4)))
    path = std.save(tmp_path / "stats")
    assert path.name == "stats.npz" and path.exists()
    assert np.allclose(Standardizer.load(path).mean, std.mean)
    assert np.allclose(Standardizer.load(tmp_path / "stats").std, std.std)


def test_a_stereo_waveform_is_not_read_as_a_spectrogram():
    import kudio
    from kudio.exceptions import FeatureError
    with pytest.raises(FeatureError, match="mono"):
        kudio.waveform_to_spectrogram(np.zeros((16000, 2), dtype=np.float32))


def test_melspectrogram_resamples_a_file_when_asked(tmp_path):
    """sr= is the rate asked for, not the file's own."""
    import kudio
    sr = 44100
    path = tmp_path / "44k.wav"
    kudio.save_wave(path, 0.3 * np.sin(2 * np.pi * 440 * np.arange(2 * sr) / sr), sr)
    native = kudio.melspectrogram(path, n_frames=1)
    at_16k = kudio.melspectrogram(path, sr=16000, n_frames=1)
    assert native.dtype == at_16k.dtype == np.float32
    assert len(at_16k) == 1 + 32000 // 512 and len(native) == 1 + 88200 // 512


def test_stack_context_zero_is_a_copy():
    from kudio import stack_context
    x = np.zeros((5, 3), dtype=np.float32)
    assert stack_context(x, 0) is not x


def test_frame_windows_says_when_there_is_nothing_to_cut():
    from kudio import frame_windows
    from kudio.exceptions import FeatureError
    with pytest.raises(FeatureError, match="no frames"):
        frame_windows(np.zeros((0, 64)), 16)


# -- one channel, and a peak to normalise by -----------------------------------
@pytest.mark.parametrize("call", [
    lambda y: kudio.mfcc(y, 16000),
    lambda y: kudio.melspectrogram(y, sr=16000),
    lambda y: kudio.STFT().analyse(y),
])
def test_stereo_is_refused_rather_than_framed_as_two_sample_signals(call):
    y = np.zeros((16000, 2), dtype=np.float32)
    with pytest.raises(kudio.FeatureError, match="mono"):
        call(y)


def test_denoise_wav2spec_refuses_a_silent_clip():
    with pytest.raises(kudio.FeatureError, match="silent"):
        kudio.denoise_wav2spec(np.zeros(16000))


# -- the window, from a file ----------------------------------------------------
def test_file_to_spectrogram_takes_the_window(tmp_path):
    y = (0.1 * np.random.default_rng(0).standard_normal(8000)).astype(np.float32)
    path = tmp_path / "clip.wav"
    kudio.save_wave(path, y, 16000, subtype="FLOAT")
    hann = kudio.file_to_spectrogram(path, window="hann")
    np.testing.assert_array_equal(hann, kudio.waveform_to_spectrogram(y, window="hann"))
    assert not np.array_equal(hann, kudio.file_to_spectrogram(path))
    stacked = kudio.logspec_from_files([path, path], False, 0, False, window="hann")
    np.testing.assert_array_equal(stacked, np.vstack([hann, hann]))


# -- more coefficients than mel bands -----------------------------------------------
@pytest.mark.parametrize("seconds", [0.768, 1.0])        # 25 frames, and 32
def test_mfcc_refuses_more_coefficients_than_mel_bands(seconds):
    """25 frames of 32 coefficients are 800 numbers, which reshape to 20 frames
    of 40: the default n_mfcc beside n_mels=32."""
    y = 0.1 * np.random.default_rng(0).standard_normal(int(seconds * 16000))
    with pytest.raises(kudio.FeatureError, match=r"^mfcc\(\): n_mfcc=40 asks for more coefficients "
                                                 r"than 32 mel bands have"):
        kudio.mfcc(y, 16000, n_mels=32)
    with pytest.raises(kudio.FeatureError, match="n_mfcc=129 .* than 128 mel bands"):
        kudio.mfcc(y, 16000, n_mfcc=129)
    for n_mfcc in (0, -3):
        with pytest.raises(kudio.FeatureError, match="n_mfcc must be >= 1"):
            kudio.mfcc(y, 16000, n_mfcc=n_mfcc)
    frames = 1 + len(y) // 512
    assert kudio.mfcc(y, 16000, n_mfcc=32, n_mels=32).shape == (1, frames, 32)
    assert kudio.mfcc(y, 16000, n_mfcc=128).shape == (1, frames, 128)
    assert kudio.mfcc(y, 16000).shape == (1, frames, 40)


def test_mfcc_from_files_refuses_them_too(wav_file):
    """Unchecked, it would come back 128 wide, whatever was asked for."""
    with pytest.raises(kudio.FeatureError, match=r"^mfcc_from_files\(\): n_mfcc=200"):
        kudio.mfcc_from_files([wav_file], n_mfcc=200)
    assert kudio.mfcc_from_files([wav_file, wav_file], n_mfcc=13).shape == (2, 32, 13)


def test_an_unsigned_reference_gives_the_phase_of_the_audio():
    """128 is an offset, and a big one in the first bins."""
    t = np.arange(16000) / 16000
    y = 0.4 * np.sin(2 * np.pi * 30 * t) + 0.1 * np.sin(2 * np.pi * 700 * t)
    as_u8 = np.round(y * 128 + 128).astype(np.uint8)
    spec = kudio.waveform_to_spectrogram(as_u8)
    back = kudio.spectrogram_to_waveform(as_u8, spec)
    assert np.max(np.abs(back - (as_u8 / 128.0 - 1.0))) < 1e-3


# -- what a transform is given ---------------------------------------------------

#: every function that transforms a waveform at a geometry of the caller's:
#: the call, given *y* and keywords
FRAMED = {
    'waveform_to_spectrogram': lambda y, **k: kudio.waveform_to_spectrogram(y, **k),
    'spectrogram_to_waveform': lambda y, **k: kudio.spectrogram_to_waveform(
        y, kudio.waveform_to_spectrogram(np.zeros(4000, np.float32)), **k),
}
#: and every one that takes a waveform at all, with what its name is in a message
TRANSFORMS = {
    **{name: (call, name) for name, call in FRAMED.items()},
    'STFT.forward': (lambda y: kudio.STFT().forward(y), 'waveform_to_spectrogram'),
    'STFT.analyse': (lambda y: kudio.STFT().analyse(y), 'STFT.analyse'),
    'STFT.inverse': (lambda y: kudio.STFT().inverse(
        y, kudio.waveform_to_spectrogram(np.zeros(4000, np.float32))), 'spectrogram_to_waveform'),
    'denoise_wav2spec': (lambda y: kudio.denoise_wav2spec(y), 'denoise_wav2spec'),
}


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("name", TRANSFORMS)
def test_samples_that_are_not_finite_are_refused_by_name(name, bad):
    """One NaN would be every bin of the frames around it, and denoise_wav2spec
    would take a clip with one for a silent clip: it has no peak."""
    call, who = TRANSFORMS[name]
    y = np.full(4000, 0.1, np.float32)
    y[100:103] = bad
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(kudio.FeatureError) as caught:
            call(y)
    assert str(caught.value) == (f"{who}() got 3 sample(s) that are not finite (NaN or inf); "
                                 f"repair the clip first")


@pytest.mark.parametrize("y, what", [(np.ones(4000, bool), "bool"),
                                     (np.ones(4000, complex), "complex128"),
                                     (np.array(['a'] * 4000), "<U1")], ids=["bool", "complex", "text"])
@pytest.mark.parametrize("name", TRANSFORMS)
def test_samples_that_are_not_numbers_are_refused_by_name(name, y, what):
    call, who = TRANSFORMS[name]
    with pytest.raises(kudio.FeatureError) as caught:
        call(y)
    assert str(caught.value) == f"{who}(): the samples are {what}, not real numbers"


@pytest.mark.parametrize("value", [0, -1, 2.5, float('nan'), float('inf'), True, '512', [512]],
                         ids=repr)
@pytest.mark.parametrize("name", FRAMED)
@pytest.mark.parametrize("size", ['n_fft', 'hop_length', 'win_length'])
def test_a_size_that_is_not_a_number_of_samples_is_refused(name, size, value):
    with pytest.raises(kudio.FeatureError) as caught:
        FRAMED[name](np.zeros(4000, np.float32), **{size: value})
    assert str(caught.value) == (f"{name}(): {size} must be >= 1, a whole number of samples, "
                                 f"got {value!r}")


@pytest.mark.parametrize("value", [512.0, np.int64(512), np.float32(512), np.array(512)], ids=repr)
def test_a_whole_number_of_any_type_is_that_size(sine, value):
    want = kudio.waveform_to_spectrogram(sine, n_fft=512, hop_length=512, win_length=512)
    assert np.array_equal(kudio.waveform_to_spectrogram(
        sine, n_fft=value, hop_length=value, win_length=value), want)


@pytest.mark.parametrize("name", FRAMED)
def test_a_window_longer_than_the_frame_is_refused(name):
    with pytest.raises(kudio.FeatureError) as caught:
        FRAMED[name](np.zeros(4000, np.float32), n_fft=512, win_length=513)
    assert str(caught.value) == f"{name}(): win_length (513) cannot exceed n_fft (512)"


@pytest.mark.parametrize("name", FRAMED)
def test_a_window_there_is_not_is_refused_by_name(name):
    with pytest.raises(kudio.FeatureError, match=rf"^{name}\(\): 'nonsense' is not a window"):
        FRAMED[name](np.zeros(4000, np.float32), window='nonsense')


@pytest.mark.parametrize("value", [-1, -8000, 0, 2.5, float('nan'), '100', False], ids=repr)
def test_a_length_to_cut_to_is_a_number_of_samples(sine, value):
    """-8000 cut 8000 samples off the end, and 0 none."""
    with pytest.raises(kudio.FeatureError) as caught:
        kudio.waveform_to_spectrogram(sine, desired_length=value)
    assert str(caught.value) == (f"waveform_to_spectrogram(): desired_length is a whole number of "
                                 f"samples to keep, 1 or more, or None for all of them, got "
                                 f"{value!r}")
    assert np.array_equal(kudio.waveform_to_spectrogram(sine, desired_length=8000.0),
                          kudio.waveform_to_spectrogram(sine[:8000]))


@pytest.mark.parametrize("n", [1, 100, 511])
def test_a_clip_shorter_than_a_frame_is_transformed_with_a_warning(n):
    """It is mostly the zeros around it. Said at the caller's line and under the
    name of the function that was called."""
    y = np.full(n, 0.1, np.float32)
    calls = [('waveform_to_spectrogram', lambda: kudio.waveform_to_spectrogram(y)),
             ('STFT.analyse', lambda: kudio.STFT().analyse(y)),
             ('denoise_wav2spec', lambda: kudio.denoise_wav2spec(y))]
    for name, call in calls:
        with pytest.warns(UserWarning) as caught:
            call()
        said, = (w for w in caught if issubclass(w.category, UserWarning))
        assert str(said.message) == (f"{name}(): the clip has {n} sample(s), fewer than one frame "
                                     f"of 512; what is transformed is mostly the zeros around it")
        assert said.filename == __file__
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        kudio.waveform_to_spectrogram(np.full(512, 0.1, np.float32))


def test_an_odd_frame_fits_no_empty_clip():
    """There are n_fft // 2 zeros on each side, one fewer than an odd frame."""
    with pytest.raises(kudio.FeatureError, match=r"^waveform_to_spectrogram\(\): the clip has no "
                                                 r"samples, and an odd frame of 65"):
        kudio.waveform_to_spectrogram(np.zeros(0, np.float32), n_fft=65)


def test_what_a_spectrogram_costs_is_its_size_not_the_frames():
    import tracemalloc
    y = (0.1 * np.random.default_rng(0).standard_normal(16000 * 60)).astype(np.float32)
    tracemalloc.start()
    try:
        spec = kudio.waveform_to_spectrogram(y, n_fft=2048, hop_length=128)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    # the complex64 spectrum and its magnitude, which becomes the result
    assert peak < 3.2 * spec.nbytes + 2 * y.nbytes, f"{peak / spec.nbytes:.1f} times the result"


# -- cutting and stacking ----------------------------------------------------------

@pytest.mark.parametrize("channels", [2, 2.0, np.int64(2)], ids=repr)
def test_a_clip_with_its_channels_side_by_side_is_separated_by_column(channels):
    """Every second frame of it is not a channel."""
    wave = np.arange(12, dtype=np.int16).reshape(6, 2)
    left, right = wave_separate(wave, channels)
    assert (left.tolist(), right.tolist()) == ([0, 2, 4, 6, 8, 10], [1, 3, 5, 7, 9, 11])
    assert np.shares_memory(left, wave)


def test_a_column_is_split_as_the_waveform_it_holds():
    left, right = wave_separate(np.arange(8)[:, None], 2)
    assert (left.tolist(), right.tolist()) == ([0, 2, 4, 6], [1, 3, 5, 7])


@pytest.mark.parametrize("wave, channels, says", [
    (np.zeros((100, 2)), 1, r"the array holds 2 channels, and 1 was asked for"),
    (np.zeros((100, 2)), 3, r"the array holds 2 channels, and 3 were asked for"),
    (np.zeros((4, 4, 4)), 2, r"takes \(frames,\) or \(frames, channels\), got shape \(4, 4, 4\)"),
    (np.float64(1.0), 2, r"takes \(frames,\) or \(frames, channels\), got shape \(\)"),
    (np.zeros(100), 0, r"channels must be >= 1, a whole number, got 0"),
    (np.zeros(100), 1.5, r"channels must be >= 1, a whole number, got 1.5"),
    (np.zeros(100), True, r"channels must be >= 1, a whole number, got True")])
def test_what_cannot_be_separated_is_refused(wave, channels, says):
    with pytest.raises(kudio.FeatureError, match=r"^wave_separate\(\):? ?" + says):
        wave_separate(wave, channels)


def test_a_stereo_array_is_not_kept_as_one_channel_by_the_manager():
    manager = kudio.AudioDataManager()
    with pytest.raises(ValueError, match="holds 2 channels, and 1 was asked for"):
        manager.add_new_file("take", wave=[np.zeros((100, 2))], sr=16000, ch=1)
    manager.add_new_file("take", wave=[np.zeros((100, 2))], sr=16000, ch=2)
    assert [w.shape for w in manager.audios_["take"]["wave"]] == [(100,), (100,)]


@pytest.mark.parametrize("features, says", [
    ([np.zeros(5), np.zeros(5)], r"item 0 has shape \(5,\)"),
    ([np.zeros((2, 5)), np.zeros(5)], r"item 1 has shape \(5,\)"),
    ([np.zeros((2, 5)), np.zeros((2, 4))], r"item 1 has 4 columns and item 0 has 5"),
    ([np.zeros((2, 5, 1))], r"item 0 has shape \(2, 5, 1\)"), ([], "there are none")])
def test_features2matrix_takes_matrices_of_one_width(features, says):
    """Two vectors of 5 became a matrix of 2 rows with 10 labels."""
    with pytest.raises(kudio.FeatureError, match=r"^features2matrix\(\).*" + says):
        features2matrix(features)


def test_features2matrix_takes_lists_and_a_matrix_with_no_rows():
    matrix, labels = features2matrix([[[1, 2], [3, 4]], np.zeros((0, 2)), np.ones((1, 2))])
    assert matrix.tolist() == [[1, 2], [3, 4], [1, 1]] and labels.tolist() == [0, 0, 2]


@pytest.mark.parametrize("n, frame, hop", [(100, 100, 1), (1000, 64, 16), (1000, 64, 100),
                                           (1001, 7, 3), (5, 1, 2)])
def test_slices_are_the_frames_cut_by_hand(n, frame, hop):
    for dtype in (np.int16, np.float32):
        y = np.arange(n, dtype=dtype)
        want = np.stack([y[s:s + frame] for s in range(0, n - frame + 1, hop)])
        got = wave_slicing(y, frame, hop, is_reshape=False)
        assert got.dtype == dtype and np.array_equal(got, want.T)
        assert np.array_equal(wave_slicing(y, frame, hop), want.reshape(-1))


@pytest.mark.parametrize("y, frame, hop, says", [
    (np.zeros(100), 101, 1, r"the clip has 100 sample\(s\), fewer than one frame of 101"),
    (np.zeros(0), 10, 1, r"the clip has 0 sample\(s\), fewer than one frame of 10"),
    (np.zeros(100), 0, 1, r"frame_length must be >= 1, a whole number of samples, got 0"),
    (np.zeros(100), 10, 0, r"hop_length must be >= 1, a whole number of samples, got 0"),
    (np.zeros(100), 10, 1.5, r"hop_length must be >= 1, a whole number of samples, got 1.5"),
    (np.zeros((100, 2)), 10, 5, r"takes a mono waveform \(n,\), got shape \(100, 2\)")])
def test_what_cannot_be_sliced_is_refused(y, frame, hop, says):
    with pytest.raises(kudio.FeatureError, match=r"^wave_slicing\(\):? ?" + says):
        wave_slicing(y, frame, hop)
