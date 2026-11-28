# -*- coding: utf-8 -*-
"""kudio held to librosa, whose conventions its spectrograms, resampling and
pitch follow, bit for bit where the two are the same arithmetic.

kudio does not need librosa and a plain install does not bring it, so on most
machines every test here skips. ``KUDIO_REQUIRE_LIBROSA=0.11`` (or ``1.0``) is
how a run says that it must not: see ``conftest.py``. Of the pitch, where librosa
takes minutes, one case in nine is compared, the same ones every time, and with
``KUDIO_LIBROSA_CASES=all`` every one.
"""
from __future__ import annotations

import inspect
import os
import warnings
import zlib

import numpy as np
import pytest
import scipy
import scipy.signal
import soundfile as sf

import kudio
from kudio.core import _mel, _spectral, _yin

pytest.importorskip("librosa")

RNG = np.random.default_rng(0)


def _length(n, orig_sr, target_sr):
    """The samples *n* come to: n * target_sr / orig_sr, up to a whole number.
    librosa rounds up the product in floating point, and from 44.1 kHz to 48 kHz
    makes 161 of 147 samples, the last one zero."""
    if isinstance(orig_sr, int) and isinstance(target_sr, int):
        return -(-n * target_sr // orig_sr)
    return int(np.ceil(n * (target_sr / orig_sr)))


def _same(got, want, what):
    """The same shape, the same type and the same bits, a NaN and its place
    included."""
    got, want = np.asarray(got), np.asarray(want)
    assert got.shape == want.shape, f"{what}: shape {got.shape}, librosa {want.shape}"
    assert got.dtype == want.dtype, f"{what}: {got.dtype}, librosa {want.dtype}"
    if not np.array_equal(got, want, equal_nan=got.dtype.kind in 'fc'):
        apart = np.abs(got.astype(np.complex128) - want.astype(np.complex128))
        raise AssertionError(f"{what}: {np.count_nonzero(got != want)} of {got.size} differ, "
                             f"by up to {np.nanmax(apart):.3g}")


# -- resampling ----------------------------------------------------------------

SOXR = ('soxr_vhq', 'soxr_hq', 'soxr_mq', 'soxr_lq', 'soxr_qq')
WHOLE = [(8000, 16000), (16000, 8000), (44100, 16000), (48000, 16000), (16000, 48000),
         (192000, 16000), (22050, 44100), (96000, 48000), (11025, 16000), (48000, 44100),
         (44100, 48000)]
#: an array can be at a rate no file can: the first Macintosh's sound chip
NOT_WHOLE = [(22254.5, 16000), (16000, 22254.5)]


@pytest.mark.parametrize("orig_sr, target_sr", WHOLE + NOT_WHOLE)
def test_resampling(librosa, orig_sr, target_sr):
    """Every resampler librosa runs without resampy or samplerate, on 1, 2 and 6
    channels and lengths that do and do not come out whole. Where librosa makes a
    sample too many it is left off, and the Fourier method, which makes the length
    part of the arithmetic, is scipy's at the right one."""
    too_many = 0
    for res_type in SOXR + ('fft', 'scipy') + ('polyphase',) * ((orig_sr, target_sr) in WHOLE):
        for n in (int(0.11 * orig_sr), 1323, 4410):
            for shape in ((n,), (n, 2), (n, 6)):
                y = (0.1 * RNG.standard_normal(shape)).astype(np.float32)
                want = librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr,
                                        res_type=res_type, axis=0)
                right = _length(n, orig_sr, target_sr)
                if len(want) != right:
                    assert len(want) == right + 1
                    too_many += 1
                    want = (want[:right] if res_type not in ('fft', 'scipy') else
                            scipy.signal.resample(y, right, axis=0))
                _same(kudio.resample(y, orig_sr, target_sr, res_type=res_type), want,
                      f"{res_type} {orig_sr} -> {target_sr}, {shape}")
    assert (too_many > 0) == ((orig_sr, target_sr) == (44100, 48000))


@pytest.mark.parametrize("orig_sr, target_sr", [(8000, 16000), (16000, 8000), (48000, 8000),
                                                (44100, 48000), (22254.5, 16000)])
def test_resampling_a_few_samples(librosa, orig_sr, target_sr):
    """soxr gives none or one of them back at some rates, and the length is the
    rates' all the same."""
    for n in list(range(1, 25)) + [147, 160, 441]:
        y = (0.1 * RNG.standard_normal(n)).astype(np.float32)
        for res_type in ('soxr_hq', 'soxr_qq', 'fft'):
            if res_type == 'fft' and n % 147 == 0 and (orig_sr, target_sr) == (44100, 48000):
                continue                        # test_resampling has those
            want = librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr,
                                    res_type=res_type)
            _same(kudio.resample(y, orig_sr, target_sr, res_type=res_type),
                  want[:_length(n, orig_sr, target_sr)],
                  f"{res_type} {orig_sr} -> {target_sr}, {n} samples")


@pytest.mark.parametrize("dtype", ['f2', 'f4', 'f8', '>f4', '>f8', 'i1', 'i2', 'i4', '>i2',
                                   'u1', 'u2', 'u4'])
def test_resampling_every_type_of_sample(librosa, dtype):
    """The work is done in float32, unsigned samples around their midpoint, and
    put back into the type it came in."""
    dtype = np.dtype(dtype)
    wave = 0.4 * np.sin(2 * np.pi * 440 * np.arange(1700) / 16000)
    if dtype.kind == 'f':
        y, middle = wave.astype(dtype), 0
    else:
        limits = np.iinfo(dtype)
        middle = (int(limits.max) + 1) // 2 if limits.min == 0 else 0
        y = np.round(wave * (middle or limits.max) + middle).astype(dtype)
    x = (y.astype(np.float64) - middle).astype(np.float32) if middle else y.astype(np.float32)
    want = librosa.resample(x, orig_sr=16000, target_sr=22050)
    if dtype.kind != 'f':
        want = np.clip(np.round(want.astype(np.float64) + middle), limits.min, limits.max)
    _same(kudio.resample(y, 16000, 22050), want.astype(dtype), str(dtype))


def test_resampling_any_layout_in_memory(librosa):
    plain = (0.1 * RNG.standard_normal((2000, 2))).astype(np.float32)
    want = librosa.resample(plain, orig_sr=16000, target_sr=11025, axis=0)
    read_only = plain.copy()
    read_only.flags.writeable = False
    every_other = np.repeat(plain, 2, axis=0)[::2]
    for name, y in (("Fortran order", np.asfortranarray(plain)), ("read-only", read_only),
                    ("every other row", every_other), ("reversed twice", plain[::-1][::-1]),
                    ("two of four columns", np.hstack([plain, plain])[:, 2:])):
        _same(kudio.resample(y, 16000, 11025), want, name)


@pytest.mark.parametrize("extension, subtype", [("wav", "PCM_16"), ("wav", "FLOAT"),
                                                ("flac", "PCM_24"), ("ogg", "VORBIS"),
                                                ("aiff", "PCM_16"), ("mp3", "MPEG_LAYER_III")])
def test_reading_a_file_at_another_rate(librosa, tmp_path, extension, subtype):
    for sr in (8000, 16000, 44100, 48000):
        for channels in (1, 2):
            path = tmp_path / f"{sr}_{channels}.{extension}"
            sf.write(str(path), (0.1 * RNG.standard_normal((sr // 2, channels))), sr,
                     subtype=subtype)
            for target in (8000, 16000, 44100):
                for mono in (True, False):
                    got, rate = kudio.file_load(path, sr=target, mono=mono)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")      # 0.11 and audioread
                        want, _ = librosa.load(str(path), sr=target, mono=mono)
                    assert rate == target
                    _same(got, want.T if want.ndim == 2 else want,
                          f"{path.name} at {target}, mono={mono}")


# -- the short-time Fourier transform ----------------------------------------------

#: frames odd and even, hops that divide them and do not, one longer than the
#: frame, windows shorter than it
FRAMES = [(64, 16, None), (64, 64, None), (65, 20, None), (65, 33, 40), (7, 2, None), (2, 1, None),
          (1, 1, None), (32, 40, None), (400, 160, None), (512, 128, None), (512, 256, 400),
          (512, 100, 333), (513, 171, None), (1024, 256, None), (2048, 512, 1200),
          (8192, 2048, None)]
WINDOWS = ['hann', 'hamming', 'blackman', 'boxcar', ('kaiser', 8.6), 8.6, ('tukey', 0.3),
           np.hanning]


def _lengths(n_fft, hop):
    return sorted({1, hop - 1, hop, n_fft // 2, n_fft - 1, n_fft, n_fft + 1, 2 * n_fft + hop,
                   5001} - {0})


@pytest.mark.parametrize("n_fft, hop, win_length", FRAMES)
def test_the_transform(librosa, n_fft, hop, win_length):
    """Of float32, float64 and float16 clips from a sample to 5001, under every
    kind of window."""
    for window in WINDOWS:
        for n in _lengths(n_fft, hop) if window == 'hann' else (n_fft + 1, 5001):
            for dtype in ('f4', 'f8', 'f2'):
                y = (0.1 * RNG.standard_normal(n)).astype(dtype)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")             # shorter than a frame
                    want = librosa.stft(y, n_fft=n_fft, hop_length=hop, win_length=win_length,
                                        window=window)
                    got = _spectral.stft(y, n_fft, hop, win_length or n_fft, window)
                _same(got, want, f"{n_fft}/{hop}/{win_length} {window} {dtype}, {n} samples")


@pytest.mark.parametrize("n_fft, hop, win_length", [f for f in FRAMES if f[0] > 1])
def test_the_inverse(librosa, n_fft, hop, win_length):
    """Of spectra that are no clip's, 1 to 40 frames, to lengths short of what
    the frames cover and past it."""
    for window in ('hann', 'hamming', ('kaiser', 8.6)):
        for frames in (1, 2, 3, 40):
            shape = (n_fft // 2 + 1, frames)
            z = RNG.standard_normal(shape) + 1j * RNG.standard_normal(shape)
            whole = n_fft + hop * (frames - 1) - 2 * (n_fft // 2)
            for dtype in (np.complex64, np.complex128):
                for length in {None, whole, max(1, whole - 3), max(1, whole // 2), whole + 9}:
                    if not whole and length is None and n_fft // 2 < hop * frames:
                        continue                        # nothing is left of one frame
                    want = librosa.istft(z.astype(dtype), hop_length=hop, win_length=win_length,
                                         window=window, n_fft=n_fft, length=length)
                    got = _spectral.istft(z.astype(dtype), n_fft, hop, win_length or n_fft, window,
                                          length)
                    _same(got, want, f"{n_fft}/{hop}/{win_length} {window} {frames} frames "
                                     f"{dtype.__name__} to {length}")


@pytest.mark.parametrize("n_fft, hop, win_length", [(512, 256, None), (512, 128, 400),
                                                    (400, 160, None), (1024, 300, None)])
def test_the_spectrograms(librosa, n_fft, hop, win_length):
    """The public functions on top, as arithmetic on librosa's transform."""
    for dtype in ('f4', 'f8', 'i2', 'u1'):
        wave = 0.3 * np.sin(2 * np.pi * 300 * np.arange(9001) / 16000) + 0.01 * RNG.standard_normal(9001)
        y = (wave.astype(dtype) if dtype[0] == 'f' else
             np.round(wave * 100 + (128 if dtype == 'u1' else 0)).astype(dtype))
        x = kudio._samples.as_float(y)
        for window in ('hamming', 'hann'):
            how = dict(n_fft=n_fft, hop_length=hop, win_length=win_length, window=window)
            D = librosa.stft(x, **how)
            spec = kudio.waveform_to_spectrogram(y, **how)
            _same(spec, np.log10(np.abs(D + float(np.finfo(float).eps)) ** 2).T, f"forward {dtype}")
            single = librosa.stft(np.asarray(x if dtype == 'u1' else y, dtype=np.float32), **how)
            single = single + float(np.finfo(float).eps)
            rebuilt = np.sqrt(10 ** spec).T * np.exp(1j * np.angle(single))
            _same(kudio.spectrogram_to_waveform(y, spec, **how),
                  librosa.istft(rebuilt, length=len(y), **how), f"inverse {dtype}")
            stft = kudio.STFT(sr=16000, **how)
            _same(stft.analyse(y), librosa.stft(x.astype(np.float32), **how).T, f"analyse {dtype}")
            z = stft.analyse(y) * np.float32(0.5)
            _same(stft.synthesise(z, length=len(y)), librosa.istft(z.T, length=len(y), **how),
                  f"synthesise {dtype}")
            assert np.array_equal(stft._window_energy(),
                                  librosa.util.pad_center(librosa.filters.get_window(
                                      window, win_length or n_fft) ** 2, size=n_fft))


def test_the_pair_for_denoising_models(librosa):
    y = (0.2 * RNG.standard_normal(16000)).astype(np.float32)
    spec, phase = kudio.denoise_wav2spec(y)
    D = librosa.stft(y / float(np.max(np.abs(y))), n_fft=512, hop_length=256, win_length=512,
                     window='hann')
    D += np.finfo(float).eps
    _same(phase, np.angle(D), "phase")
    want = librosa.istft(np.squeeze(spec).T * np.exp(1j * phase), hop_length=256, win_length=512,
                         window='hann')
    _same(kudio.denoise_spec2wav(spec, phase), want / float(np.max(np.abs(want))), "waveform")


@pytest.mark.parametrize("n, frame, hop", [(100, 100, 1), (1000, 64, 16), (1000, 64, 100),
                                           (1001, 7, 3), (5, 1, 2)])
def test_slices(librosa, n, frame, hop):
    for dtype in ('i2', 'f4', 'f8'):
        y = np.arange(n, dtype=dtype)
        want = librosa.util.frame(y, frame_length=frame, hop_length=hop)
        _same(kudio.wave_slicing(y, frame, hop, is_reshape=False), want, dtype)
        _same(kudio.wave_slicing(y, frame, hop), want.T.reshape(-1), dtype)


# -- mel bands and their cepstrum ---------------------------------------------------

@pytest.mark.parametrize("sr", [8000, 16000, 22050, 44100, 48000, 192000, 22254.5])
def test_the_filters(librosa, sr):
    """Frames odd and even, more bands than there are bins for, and limits under,
    at and above half the rate. The scale and the scaling are librosa's defaults,
    which are all kudio has."""
    for n_fft in (64, 257, 512, 1024, 2048, 8192):
        for n_mels in (1, 2, 23, 40, 64, 128, 256):
            for fmin in (0.0, 20.0, 125.5, 1000.0):
                for fmax in (None, 3800.0, sr / 2, 0.9 * sr / 2, 1.5 * sr / 2):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")                 # bands with no bin
                        want = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin,
                                                   fmax=fmax, htk=False, norm='slaney')
                        got = _mel.filters(sr, n_fft, n_mels, fmin, fmax)
                    _same(got, want, f"{sr} {n_fft} {n_mels} {fmin} {fmax}")


def test_a_band_librosa_warns_about_is_one_kudio_counts(librosa):
    """Wherever it warns. kudio also counts a lowest band that starts at 0 Hz,
    which librosa leaves out of its check: three bands to 12 Hz, on bins 7.8 Hz
    apart, and the first has none."""
    more = []
    for sr, n_fft in ((16000, 64), (16000, 512), (16000, 2048), (192000, 2048), (8000, 8192)):
        for n_mels in (3, 23, 64, 128, 256):
            for fmin, fmax in ((0.0, None), (300.0, None), (0.0, sr), (50.0, 60.0), (0.0, 12.0)):
                with warnings.catch_warnings(record=True) as theirs:
                    warnings.simplefilter("always")
                    f = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
                with warnings.catch_warnings(record=True) as ours:
                    warnings.simplefilter("always")
                    _mel.filters(sr, n_fft, n_mels, fmin, fmax, 'here')
                empty = int(np.count_nonzero(f.max(axis=1) == 0))
                assert len(ours) == (empty > 0)
                assert not theirs or ours, (sr, n_fft, n_mels, fmin, fmax)
                if empty:
                    assert str(ours[0].message).startswith(f"here(): {empty} of {n_mels} mel bands")
                if ours and not theirs:
                    more.append((sr, n_fft, n_mels, fmax))
    assert more == [(16000, 2048, 3, 12.0)]


def test_the_frequencies(librosa):
    for n_mels in (1, 2, 40, 64, 128):
        for fmin, fmax in ((0.0, 8000.0), (125.5, 3800.0), (1000.0, 11025.0)):
            want = librosa.mel_frequencies(n_mels=n_mels + 2, fmin=fmin, fmax=fmax)[1:-1]
            _same(kudio.mel_band_hz(n_mels, 2 * fmax, fmin=fmin), want, f"{n_mels} {fmin}")
            _same(kudio.mel_band_hz(n_mels, 48000, fmin=fmin, fmax=fmax), want, f"{n_mels}")


def _takes():
    t = np.arange(24000) / 16000
    speech = (0.3 * np.sin(2 * np.pi * 150 * t) * np.clip(np.sin(2 * np.pi * 3 * t), 0, 1)
              + 0.003 * RNG.standard_normal(len(t)))
    click = np.zeros(6000)
    click[3000] = 0.9
    return {'speech': speech, 'noise': 0.1 * RNG.standard_normal(16000), 'silence': np.zeros(5000),
            'click': click, 'constant': np.full(5000, 0.2), 'short': 0.1 * RNG.standard_normal(700),
            'loud': 1e4 * RNG.standard_normal(5000), 'quiet': 1e-7 * RNG.standard_normal(5000)}


@pytest.mark.parametrize("name", list(_takes()))
def test_mfcc_and_mel_spectrograms(librosa, name):
    wave = _takes()[name]
    for dtype in ('f4', 'f8'):
        y = wave.astype(dtype)
        for sr in (8000, 16000, 44100):
            for given in ({}, dict(n_fft=512, hop_length=160, n_mels=40), dict(n_mels=64, fmax=3800),
                          dict(n_fft=400, hop_length=100, n_mels=23, fmax=0.8 * sr / 2),
                          dict(n_fft=1024, n_mels=80, fmax=0.75 * sr)):
                for n_mfcc in (1, 13, 20):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")             # short, or with no bin
                        want = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, **given)
                        got = kudio.mfcc(y, sr, n_mfcc, **given)
                    _same(got[0], want.T, f"mfcc {name} {dtype} {sr} {given} {n_mfcc}")
            for n_mels, n_fft, hop, power in ((64, 1024, 512, 2.0), (40, 512, 128, 1.0),
                                              (128, 2048, 512, 2), (80, 400, 160, 1.5)):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    mel = librosa.feature.melspectrogram(
                        y=y.astype(np.float32), sr=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels,
                        power=power)
                    want = 20.0 / power * np.log10(np.maximum(mel, float(np.finfo(float).eps)))
                    got = kudio.melspectrogram(y, n_mels, 1, n_fft, hop, power, sr=sr)
                _same(got, want.T, f"mel {name} {dtype} {sr} {n_mels} {n_fft} {hop} {power}")


def test_integer_audio_in_mel_bands(librosa):
    wave = _takes()['speech']
    for dtype, full, middle in (('i2', 32768, 0), ('i4', 2 ** 31, 0), ('u1', 128, 128)):
        y = np.round(wave * full * 0.9 + middle).astype(dtype)
        x = kudio._samples.as_float(y)
        _same(kudio.mfcc(y, 16000, 20)[0], librosa.feature.mfcc(y=x, sr=16000, n_mfcc=20).T, dtype)


def test_the_stream_in_mel(librosa):
    y = 0.1 * RNG.standard_normal(16000)
    for sr, n_fft, hop, n_mels, fmin, fmax in ((16000, 512, 128, 40, 0.0, None),
                                               (44100, 2048, 512, 64, 50.0, 15000.0)):
        stream = kudio.SpectrogramStream(sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels, fmin=fmin,
                                         fmax=fmax, top_db=300)
        window = scipy.signal.get_window('hann', n_fft, fftbins=True)
        basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin,
                                    fmax=fmax or sr / 2.0).astype(np.float64)
        scale = np.full(n_fft // 2 + 1, 2.0 / window.sum())
        scale[[0, -1]] = 1.0 / window.sum()
        want = [np.maximum(10 * np.log10(np.maximum(
            basis @ (np.abs(np.fft.rfft(y[s:s + n_fft] * window)) * scale) ** 2, 1e-20)), -300)
            for s in range(0, len(y) - n_fft + 1, hop)]
        _same(stream.push(y), np.array(want, dtype=np.float32), f"{sr} {n_fft}")


# -- silence ------------------------------------------------------------------------

def _cut_takes():
    """Speech over floors from 40 to 80 dB down, and the cases at the edges: a
    frame of it or less, nothing but zeros, a constant, more channels than one."""
    out = {}
    for seed in range(6):
        rng = np.random.default_rng(seed)
        t = np.arange(16000) / 16000
        y = 0.3 * np.sin(2 * np.pi * 150 * t) * np.clip(np.sin(2 * np.pi * 2.1 * t + seed), 0, 1)
        out[f"speech {seed}"] = y + 10 ** (-rng.uniform(40, 80) / 20) * rng.standard_normal(len(y))
    for n in (1, 2, 100, 511, 512, 513, 1024, 2047, 2048, 2049):
        out[f"{n} samples"] = np.r_[np.zeros(n // 2), 0.1 * RNG.standard_normal(n - n // 2)]
    out['zeros'] = np.zeros(5000)
    for level in (1.0, 0.01, 1e-5, 1e-7):
        out[f"constant {level}"] = np.full(5000, level)
    for channels in (2, 3, 6):
        out[f"{channels} channels"] = np.stack(
            [np.r_[np.zeros(1000 * c), 10.0 ** -c * RNG.standard_normal(9000 - 1000 * c)]
             for c in range(channels)], axis=1)
    out['channels of zeros'] = np.zeros((4000, 2))
    return out


@pytest.mark.parametrize("name", list(_cut_takes()))
def test_cutting_at_silence(librosa, name):
    wave = _cut_takes()[name]
    for dtype in ('f4', 'f8'):
        y = wave.astype(dtype)
        for top_db in (10, 20, 30.0, 40, 60, 80):
            for frame, hop in ((2048, 512), (512, 128), (401, 100), (64, 200), (256, 50000), (1, 1),
                               (33, 1)):
                how = dict(top_db=top_db, frame_length=frame, hop_length=hop)
                got, cut = kudio.trim_silence(y, **how)
                want, index = librosa.effects.trim(y.T, **how)
                assert cut == tuple(int(i) for i in index), f"{name} {dtype} {how}"
                _same(got, want.T, f"trim {name} {dtype} {how}")
                pieces = kudio.split_on_silence(y, **how)
                cuts = librosa.effects.split(y.T, **how)
                assert len(pieces) == len(cuts), f"{name} {dtype} {how}"
                for piece, (a, b) in zip(pieces, cuts):
                    _same(piece, y[a:b], f"split {name} {dtype} {how}")


def test_cutting_integer_audio_at_silence(librosa):
    wave = _cut_takes()['speech 1']
    for dtype, full, middle in (('i2', 32768, 0), ('i4', 2 ** 31, 0), ('u1', 128, 128)):
        y = np.round(wave * full * 0.9 + middle).astype(dtype)
        index = librosa.effects.trim(kudio._samples.as_float(y), top_db=30)[1]
        assert kudio.trim_silence(y, 30)[1] == tuple(int(i) for i in index)


# -- pitch -------------------------------------------------------------------------

def _all_of_the_search(librosa):
    """librosa 1.0 leaves moves less likely than 1e-4 out of it unless it is told
    not to; 0.11 searches them all."""
    there = 'transition_min_prob' in inspect.signature(librosa.pyin).parameters
    return {'transition_min_prob': None} if there else {}


EVERY = 1 if os.environ.get("KUDIO_LIBROSA_CASES") == "all" else 9


def _some(cases, *key):
    """Of a test's *cases* the ones this run compares, one at the least. *key* says
    which of the nine to begin at, so that not every test has the same ones."""
    cases = list(cases)
    return cases[zlib.crc32(repr(key).encode()) % EVERY::EVERY] or cases[:1]


def _frame(sr, fmin):
    """What f0 takes for a frame when it is not told."""
    return max(2048, 1 << int(np.ceil(np.log2(2 * int(np.floor(sr / fmin)) + 2))))


def _same_track(got, want, what):
    """The same pitch and the same voicing, and the same probability of it. Below scipy
    1.12 the last is the same but for its last bits: there ``special.betainc``, which
    kudio takes the thresholds' weights from, and ``stats.beta``, which librosa does, are
    two pieces of code, and 19 of the 100 weights are one bit apart."""
    _same(got[0], want[0], f"f0 of {what}")
    _same(got[1], want[1], f"voiced of {what}")
    if tuple(int(n) for n in scipy.__version__.split('.')[:2]) >= (1, 12):
        _same(got[2], want[2], f"voiced_prob of {what}")
    else:
        assert got[2].shape == want[2].shape and got[2].dtype == want[2].dtype
        assert np.max(np.abs(got[2] - want[2])) < 2e-15, f"voiced_prob of {what}"


def _same_pitch(librosa, y, sr, what, fmin=65.0, fmax=400.0, frame=None, hop=None, center=True):
    """kudio.f0 against librosa.pyin of the same float32 samples: the pitch, the
    voicing, its probability and the times."""
    frame, hop = frame or _frame(sr, fmin), hop or int(sr // 100)
    want = librosa.pyin(np.asarray(y, np.float32), fmin=fmin, fmax=fmax, sr=sr,
                        frame_length=frame, hop_length=hop, center=center,
                        **_all_of_the_search(librosa))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                          # a hop over 24 ms
        got = kudio.f0(y, sr, fmin=fmin, fmax=fmax, frame_length=frame, hop_length=hop,
                       center=center)
    _same_track((got.f0, got.voiced, got.voiced_prob), want,
                f"{what}, {sr} Hz, {fmin}-{fmax}, {frame}/{hop}, center={center}")
    _same(got.times, librosa.times_like(want[0], sr=sr, hop_length=hop,
                                        n_fft=None if center else frame), f"times of {what}")


def _hard(kind, sr, n, rng):
    t = np.arange(n) / sr
    if kind == 'clicks':
        y = np.zeros(n)
        y[::sr // 7] = 0.8
        return y
    return {'noise': 0.1 * rng.standard_normal(n), 'silence': np.zeros(n),
            'offset': np.full(n, 0.2), 'tone': 0.2 * np.sin(2 * np.pi * 187.3 * t),
            'sweep': 0.2 * np.sin(2 * np.pi * (60 * t + 0.5 * 740 / (n / sr) * t ** 2)),
            'loud': 1e6 * rng.standard_normal(n),
            'quiet': 1e-9 * rng.standard_normal(n)}[kind]


HARD = ('noise', 'silence', 'offset', 'tone', 'clicks', 'sweep', 'loud', 'quiet')


@pytest.mark.parametrize("sr", [8000, 16000, 22050, 48000])
@pytest.mark.parametrize("kind", HARD)
def test_the_pitch_of_hard_signals(librosa, kind, sr):
    """528 of them, with the ones too short for a frame that is not centred left
    out: 50 ms to 1.5 s, in three ranges."""
    rng = np.random.default_rng(5)
    cases = []
    for seconds in (0.05, 0.5, 1.5):
        for dtype in ('f4', 'f8'):
            for fmin, fmax, center in ((65.0, 400.0, True), (50.0, 2000.0, True),
                                       (100.0, 600.0, False)):
                y = _hard(kind, sr, int(sr * seconds), rng).astype(dtype)
                if center or len(y) >= _frame(sr, fmin):
                    cases.append((y, f"{kind}, {seconds} s {dtype}", fmin, fmax, center))
    for y, what, fmin, fmax, center in _some(cases, kind, sr):
        _same_pitch(librosa, y, sr, what, fmin, fmax, center=center)


def _jumps(sr, seconds, rng, every=0.05):
    """A voice that is somewhere else every 50 ms, over a floor."""
    hz = rng.choice([80, 110, 160, 230, 320, 390, 700, 1200], size=int(seconds / every) + 1)
    phase = 2 * np.pi * np.cumsum(np.repeat(hz, int(every * sr))[:int(seconds * sr)]) / sr
    return (0.2 * sum(np.sin(k * phase) for k in range(1, 5))
            + 0.002 * rng.standard_normal(len(phase)))


@pytest.mark.parametrize("sr", [8000, 16000, 48000])
def test_pitch_over_hops_in_which_it_can_hardly_move(librosa, sr):
    """Under 1.16 ms no move is allowed, and up to 3.5 ms half a semitone either way.
    Where the voice jumps the search takes one that has no probability, at
    log(tiny)."""
    rng = np.random.default_rng(sr)
    cases = [(_jumps(sr, 0.4, rng)[:max(600 * hop, int(2 * sr / fmin) + 1)], fmin, fmax, hop)
             for hop in sorted({1, 4, 9, 18, 19, 37, int(0.00115 * sr), int(0.0012 * sr),
                                int(0.0034 * sr), int(0.0036 * sr)})
             for fmin, fmax in ((65.0, 400.0), (50.0, 2000.0))]
    for y, fmin, fmax, hop in _some(cases, "hops", sr):
        _same_pitch(librosa, y, sr, "jumps", fmin, fmax, hop=hop)


@pytest.mark.parametrize("sr", [8000, 16000, 48000])
def test_pitch_of_a_few_samples(librosa, sr):
    """Of the tracker alone: f0 tells no pitch that has not come round twice. One
    frame, with a hop longer than the clip, and more."""
    rng = np.random.default_rng(1)
    cases = [(y.astype(np.float32), hop)
             for n in list(range(1, 40)) + [100, 500, 1000, 2047, 2048, 3000]
             for y in (0.1 * rng.standard_normal(n), _jumps(sr, 1.0, rng)[:n], np.full(n, 0.3))
             for hop in (160, 72 * sr // 1000)]         # the longest that librosa takes
    for y, hop in _some(cases, "a few samples", sr):
        want = librosa.pyin(y, fmin=65.0, fmax=400.0, sr=sr, frame_length=2048,
                            hop_length=hop, **_all_of_the_search(librosa))
        _same_track(_yin.track(y, sr, 65.0, 400.0, 2048, hop, True), want,
                    f"{len(y)} samples, hop {hop}")


@pytest.mark.parametrize("frame", [494, 495, 511, 512, 600, 1023, 1500, 2047, 2048, 3001, 4096])
def test_pitch_in_frames_odd_and_even_and_hops_that_do_not_divide_them(librosa, frame):
    rng = np.random.default_rng(2)
    cases = [(_jumps(16000, 0.5, rng), "jumps", hop, center)
             for hop in (37, 160, 441, 1000) for center in (True, False)]
    cases += [(_jumps(16000, 1.0, rng)[:n], f"{n} samples", None, False)
              for n in (frame, frame + 1, frame + 159, frame + 160)]
    for y, what, hop, center in _some(cases, "frames", frame):
        _same_pitch(librosa, y, 16000, what, frame=frame, hop=hop, center=center)


@pytest.mark.parametrize("sr", [8000, 11025, 16000, 22254.5, 44100, 96000, 192000])
def test_pitch_in_ranges_wide_and_narrow(librosa, sr):
    """Up to half the rate, down to 30 Hz, and as narrow as librosa takes: what a
    pitch can cross in a hop. Of noise, jumps, and a tone on the edges of the range."""
    rng = np.random.default_rng(3)
    t = np.arange(int(0.25 * sr)) / sr
    cases = []
    for fmin, fmax in ((65.0, 400.0), (30.0, 250.0), (400.0, sr / 2), (100.0, 127.0),
                       (sr / 4, sr / 2), (200.0, 401.0), (65.0, 2000.0)):
        for name, y in (('jumps', _jumps(int(sr), 0.25, rng)),
                        ('noise', 0.1 * rng.standard_normal(len(t))),
                        ('fmin', 0.3 * np.sin(2 * np.pi * fmin * t)),
                        ('fmax', 0.3 * np.sin(2 * np.pi * 0.995 * fmax * t)),
                        ('both', 0.2 * np.sin(2 * np.pi * fmin * t)
                         + 0.2 * np.sin(2 * np.pi * 0.98 * fmax * t))):
            cases.append((y, name, fmin, fmax))
    for y, name, fmin, fmax in _some(cases, "ranges", sr):
        _same_pitch(librosa, y, sr, name, fmin, fmax)


def test_pitch_of_integer_audio_and_of_audio_off_its_centre(librosa):
    rng = np.random.default_rng(4)
    y = _jumps(16000, 1.0, rng)
    y /= 1.1 * np.abs(y).max()
    _same_pitch(librosa, y + 0.09, 16000, "an offset")
    _same_pitch(librosa, 1e-3 * y - 5, 16000, "all offset")
    pcm = np.round(32767 * y).astype(np.int16)
    want = librosa.pyin((pcm / 32768).astype(np.float32), fmin=65.0, fmax=400.0, sr=16000,
                        hop_length=160, **_all_of_the_search(librosa))
    _same(kudio.f0(pcm, 16000).f0, want[0], "int16")
    _same(kudio.f0(pcm[:, np.newaxis], 16000).f0, want[0], "a column of int16")


@pytest.mark.parametrize("sr, frames", [(8000, []), (16000, []), (22050, [111]),
                                        (48000, [116, 117, 121])])
def test_where_librosa_1_0_leaves_moves_out_of_the_search(librosa, sr, frames):
    """Left alone it calls these frames the other way, of a sweep tracked from 100 to
    600 Hz in frames that are not centred, and gives the same pitch in every frame
    both call voiced. Of the 528 signals above these two are all, each there twice:
    f0 works in float32, and a float64 clip is that one over again."""
    y = _hard('sweep', sr, int(1.5 * sr), RNG).astype(np.float32)
    hertz, voiced, _ = librosa.pyin(y, fmin=100.0, fmax=600.0, sr=sr, center=False,
                                    frame_length=2048, hop_length=sr // 100)
    track = kudio.f0(y, sr, fmin=100.0, fmax=600.0, center=False)
    apart = np.flatnonzero(voiced != track.voiced).tolist()
    assert apart == (frames if _all_of_the_search(librosa) else [])
    both = voiced & track.voiced
    assert np.array_equal(hertz[both], track.f0[both])


# -- what the plots are made of --------------------------------------------------------

@pytest.mark.parametrize("sr", [8000, 16000, 44100])
def test_the_curves_and_the_levels_that_are_drawn(librosa, sr):
    """Of clips of every length about a frame's, of noise, a voice and silence."""
    pytest.importorskip("matplotlib")
    from kudio.util import _draw

    def highest(values, axis=None, keepdims=True):
        # an array of one: of a number numpy below 2 makes a float64, and librosa with it
        return values.max(axis=axis, keepdims=True)

    rng = np.random.default_rng(6)
    for n in (1, 511, 512, 2047, 2048, 2049, 9000):
        t = np.arange(n) / sr
        for name, y in (('noise', 0.1 * rng.standard_normal(n)), ('zeros', np.zeros(n)),
                        ('voice', 0.1 * sum(np.sin(2 * np.pi * 150 * k * t) / k for k in range(1, 9))
                         * (t > 0.1))):
            y = y.astype(np.float32)
            what = f"{name}, {n} samples at {sr} Hz"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")                          # short ones
                want = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]     # 1.0 in float32
                _same(_draw.rolloff(y, sr).astype(want.dtype), want, f"the roll-off of {what}")
                want = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
                got = _draw.centroid(y, sr)
                near = 1e-12 if want.dtype == np.float64 else 1e-5                  # 1.0 again
                assert got.shape == want.shape and np.allclose(got, want, rtol=near), what
                _same(_draw.crossings(y + 0.0001),
                      librosa.feature.zero_crossing_rate(y + 0.0001)[0], f"crossings of {what}")
                _same(_draw.level(y, ref=1.0), librosa.amplitude_to_db(np.abs(librosa.stft(y))),
                      f"the level of {what}")
                magnitude = np.abs(librosa.stft(y, n_fft=512, hop_length=256, win_length=512,
                                                window='hamming'))
                _same(_draw.level(y, 512, 256, 'hamming'),
                      librosa.amplitude_to_db(magnitude, ref=highest), f"hamming, {what}")
                for fmax in (sr / 2, sr / 4):
                    bands = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40, fmax=fmax)
                    _same(_draw.mel_level(y, sr, 40, fmax, 'test'),
                          librosa.power_to_db(bands, ref=highest), f"mel to {fmax} of {what}")


# -- time stretch --------------------------------------------------------------------

@pytest.mark.parametrize("sr, hz", [(8000, 120), (16000, 150), (44100, 220)])
@pytest.mark.parametrize("rate", [0.5, 0.8, 0.9, 1.1, 1.25, 2.0])
def test_time_stretch_but_for_its_end_and_the_phases_kept_in_float32(librosa, sr, hz, rate):
    """Not the same to the bit, and not meant to be. Away from the last frames two
    seconds of a voice are within 0.7 % of its peak of 0.11, whose phases have drifted
    that far, and within 5 parts in a million of 1.0."""
    t = np.arange(2 * sr) / sr
    phase = 2 * np.pi * np.cumsum(hz * (1 + 0.04 * np.sin(2 * np.pi * 3 * t))) / sr
    y = (0.1 * sum(np.sin(k * phase) / k for k in range(1, 15))
         * np.clip(2 * np.sin(2 * np.pi * 1.5 * t), 0.02, 1)
         + 1e-3 * np.random.default_rng(1).standard_normal(len(t))).astype(np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                  # 1.0's own deprecated arguments
        want = librosa.effects.time_stretch(y, rate=rate)
    got = kudio.time_stretch(y, rate)
    assert got.shape == want.shape and got.dtype == want.dtype
    inside = slice(None, -int(2048 + 1024 / min(rate, 1)))
    apart = np.abs(got.astype(np.float64) - want)[inside].max() / np.abs(want).max()
    assert apart < (7e-3 if librosa.__version__.startswith('0.') else 5e-6)
