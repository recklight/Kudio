# -*- coding: utf-8 -*-
"""Compare kudio with librosa where kudio follows librosa's conventions: the
resampling, the short-time transform and its inverse, the mel filters, the mel
spectrogram and MFCC, cutting at silence, pYIN and the time stretch.

Run it from anywhere; it imports the kudio of the checkout it sits in::

    python tools/validation/against_librosa.py [--all] [--long]

Besides what kudio itself needs it needs librosa, 0.10 or later; kudio does not.
I ran it with librosa 0.11.0 on Python 3.9.13 (numpy 2.0.2, scipy 1.13.1) and
with librosa 1.0.1 on Python 3.13.9 (numpy 2.5.4, scipy 1.18.1).

The cases are those of tests/test_against_librosa.py, made here from the same
seeds, so the counts below are the tests' counts:

1. resample() against librosa.resample: 13 pairs of rates, two of them not
   whole, every resampler librosa runs without resampy or samplerate, 1, 2
   and 6 channels; a few samples; twelve sample types; five layouts in
   memory; and file_load(sr=...) against librosa.load on six kinds of file.
   Where librosa's length is one sample more than n * target / orig rounded
   up, its last sample is left off, and the Fourier methods, whose length is
   part of the arithmetic, are scipy's at the right length. Then the length
   alone: from each of 14 rates to each other, for the lengths that come out
   whole, how often librosa's has one more.
2. The transform and its inverse over 16 geometries, eight windows and float16,
   float32 and float64, against librosa.stft and librosa.istft; and the
   spectrogram functions on top of them.
3. 5880 sets of mel filters against librosa.filters.mel, the band centres
   against librosa.mel_frequencies, and mfcc() and melspectrogram() of eight
   kinds of clip against librosa.feature.
4. trim_silence() and split_on_silence() against librosa.effects.trim and
   librosa.effects.split, 25 clips, 6 thresholds and 7 frames.
5. f0() against librosa.pyin: 528 hard signals and 1243 more over hops, frames,
   ranges, rates and lengths. librosa 1.0 leaves out of its search the moves
   less likely than 1e-4 unless it is passed transition_min_prob=None, and
   kudio searches them all, as librosa 0.11 does; so against 1.0 the whole
   search is what is compared, and its default is measured beside it: on the
   528 signals, and on a minute of a voice with pauses over a floor 60 and 40
   dB under it, at hops of 5 to 70 ms.
   Of the pitch one case in nine is compared, the same ones the tests take,
   unless --all is given.
6. time_stretch() against librosa.effects.time_stretch on two seconds of a
   voice, away from the end and at it; then tones stretched by 1.25 and 0.5,
   whose frequency is known, second by second: the frequency of the best sine
   found with its frequency left free, in cents off the tone, and what it
   leaves (SINAD). 10 s at 48 kHz; with --long 60 s and 300 s as well.

Everything in 1 to 5 is the same arithmetic, so the limit there is 0, to the
bit. The time stretch is not meant to be the same: kudio keeps the phases in
float64 where librosa 0.11 kept those of float32 audio in float32, and goes on
past the last frame where both librosas do something else. Its limits are the
tests': 0.7 % of the peak against 0.11 and 5 parts in a million against 1.0,
away from the last frames; what I measured is 0.47 % and 4.3 in a million. The
tones are held to what they should come out as.

It prints a table and exits with 1 if a difference is past its limit. The run
takes about a minute with librosa 0.11 and a minute and a half with 1.0, whose
default search is measured as well; --all adds about 3.5 minutes with 0.11 and
1.5 with 1.0, and --long about 1.5.

References:

* B. McFee, C. Raffel, D. Liang, D. P. W. Ellis, M. McVicar, E. Battenberg and
  O. Nieto, "librosa: Audio and music signal analysis in Python", Proc. 14th
  Python in Science Conference (SciPy 2015), pp. 18-24,
  doi:10.25080/Majora-7b98e3ed-003.
* M. Mauch and S. Dixon, "pYIN: A fundamental frequency estimator using
  probabilistic threshold distributions", Proc. ICASSP 2014, pp. 659-663,
  doi:10.1109/ICASSP.2014.6853678.
"""
from __future__ import annotations

import argparse
import inspect
import math
import sys
import tempfile
import time
import warnings
import zlib
from pathlib import Path

import numpy as np
import scipy
import scipy.signal
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import kudio  # noqa: E402
from kudio import _samples  # noqa: E402
from kudio.core import _mel, _spectral, _yin  # noqa: E402

try:
    import librosa
except ImportError:
    librosa = None

#: (check, cases, cases apart, largest difference, limit)
RESULTS = []
RNG = np.random.default_rng(0)


def report(check, cases, apart, diff, limit=0.0):
    RESULTS.append((check, cases, apart, diff, limit))


class Tally:
    """Cases compared to the bit: how many, how many apart and by how much at most. A
    case is one call unless begin() says where one starts, for a case that is several
    arrays: a track's pitch, voicing and probability, a cut and what it cut."""

    def __init__(self):
        self.cases = self.apart = 0
        self.diff = 0.0
        self.first = None
        self._grouped = self._off = False

    def begin(self):
        self._grouped, self._off = True, False
        self.cases += 1

    def same(self, got, want, what):
        got, want = np.asarray(got), np.asarray(want)
        if not self._grouped:
            self.cases += 1
            self._off = False
        if got.shape != want.shape or got.dtype != want.dtype:
            self._apart(math.inf, f"{what}: {got.shape} {got.dtype}, librosa {want.shape} "
                                  f"{want.dtype}")
        elif not np.array_equal(got, want, equal_nan=got.dtype.kind in 'fc'):
            far = np.abs(got.astype(np.complex128) - want.astype(np.complex128))
            self._apart(float(np.nanmax(far)), what)

    def _apart(self, diff, what):
        self.apart += not self._off
        self._off = True
        self.diff = max(self.diff, diff)
        self.first = self.first or what

    def report(self, check):
        report(check, self.cases, self.apart, self.diff)
        if self.first:
            print(f"  {check}: first apart, {self.first}")


def quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


# -- 1. resampling -------------------------------------------------------------------

SOXR = ('soxr_vhq', 'soxr_hq', 'soxr_mq', 'soxr_lq', 'soxr_qq')
WHOLE = [(8000, 16000), (16000, 8000), (44100, 16000), (48000, 16000), (16000, 48000),
         (192000, 16000), (22050, 44100), (96000, 48000), (11025, 16000), (48000, 44100),
         (44100, 48000)]
NOT_WHOLE = [(22254.5, 16000), (16000, 22254.5)]
#: what files come at, and the rates of 48 kHz and 44.1 kHz doubled and halved
USUAL = (8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000, 64000, 88200, 96000,
         176400, 192000)


def length(n, orig_sr, target_sr):
    if isinstance(orig_sr, int) and isinstance(target_sr, int):
        return -(-n * target_sr // orig_sr)
    return int(np.ceil(n * (target_sr / orig_sr)))


def check_resampling():
    tally, too_many, calls, where = Tally(), 0, 0, set()
    for orig_sr, target_sr in WHOLE + NOT_WHOLE:
        for res_type in SOXR + ('fft', 'scipy') + ('polyphase',) * ((orig_sr, target_sr) in WHOLE):
            for n in (int(0.11 * orig_sr), 1323, 4410):
                for shape in ((n,), (n, 2), (n, 6)):
                    y = (0.1 * RNG.standard_normal(shape)).astype(np.float32)
                    want = librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr,
                                            res_type=res_type, axis=0)
                    calls += 1
                    right = length(n, orig_sr, target_sr)
                    if len(want) != right:
                        too_many += 1
                        where.add(f"{orig_sr} -> {target_sr}")
                        want = (want[:right] if res_type not in ('fft', 'scipy') else
                                scipy.signal.resample(y, right, axis=0))
                    tally.same(kudio.resample(y, orig_sr, target_sr, res_type=res_type), want,
                               f"{res_type} {orig_sr} -> {target_sr}, {shape}")
    for orig_sr, target_sr in [(8000, 16000), (16000, 8000), (48000, 8000), (44100, 48000),
                               (22254.5, 16000)]:
        for n in list(range(1, 25)) + [147, 160, 441]:
            y = (0.1 * RNG.standard_normal(n)).astype(np.float32)
            for res_type in ('soxr_hq', 'soxr_qq', 'fft'):
                if res_type == 'fft' and n % 147 == 0 and (orig_sr, target_sr) == (44100, 48000):
                    continue
                want = librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr, res_type=res_type)
                tally.same(kudio.resample(y, orig_sr, target_sr, res_type=res_type),
                           want[:length(n, orig_sr, target_sr)], f"{res_type}, {n} samples")
    for dtype in ('f2', 'f4', 'f8', '>f4', '>f8', 'i1', 'i2', 'i4', '>i2', 'u1', 'u2', 'u4'):
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
        tally.same(kudio.resample(y, 16000, 22050), want.astype(dtype), str(dtype))
    plain = (0.1 * RNG.standard_normal((2000, 2))).astype(np.float32)
    want = librosa.resample(plain, orig_sr=16000, target_sr=11025, axis=0)
    read_only = plain.copy()
    read_only.flags.writeable = False
    for y in (np.asfortranarray(plain), read_only, np.repeat(plain, 2, axis=0)[::2],
              plain[::-1][::-1], np.hstack([plain, plain])[:, 2:]):
        tally.same(kudio.resample(y, 16000, 11025), want, "a layout in memory")
    with tempfile.TemporaryDirectory() as folder:
        for extension, subtype in (("wav", "PCM_16"), ("wav", "FLOAT"), ("flac", "PCM_24"),
                                   ("ogg", "VORBIS"), ("aiff", "PCM_16"),
                                   ("mp3", "MPEG_LAYER_III")):
            if subtype not in sf.available_subtypes(extension.upper() if extension != 'aiff'
                                                    else 'AIFF'):
                continue
            for sr in (8000, 16000, 44100, 48000):
                for channels in (1, 2):
                    path = Path(folder) / f"{sr}_{channels}.{extension}"
                    sf.write(str(path), 0.1 * RNG.standard_normal((sr // 2, channels)), sr,
                             subtype=subtype)
                    for target in (8000, 16000, 44100):
                        for mono in (True, False):
                            got, _ = kudio.file_load(path, sr=target, mono=mono)
                            want, _ = quiet(librosa.load, str(path), sr=target, mono=mono)
                            tally.same(got, want.T if want.ndim == 2 else want,
                                       f"{path.name} at {target}, mono={mono}")
    tally.report("resample: samples")
    print(f"  of the {calls} calls at 13 pairs of rates librosa gave one sample too many in "
          f"{too_many}, at {', '.join(sorted(where)) or 'none'}")


def check_lengths():
    """From each usual rate to each other, the lengths that come out whole: k times
    orig / gcd samples for k from 1 to 12."""
    more, pairs, lengths, wrong = [], 0, 0, 0
    for orig_sr in USUAL:
        for target_sr in USUAL:
            if orig_sr == target_sr:
                continue
            pairs += 1
            step = orig_sr // math.gcd(orig_sr, target_sr)
            extra = 0
            for k in range(1, 13):
                n = k * step
                lengths += 1
                theirs = len(librosa.resample(np.zeros(n, np.float32), orig_sr=orig_sr,
                                              target_sr=target_sr, res_type='soxr_qq'))
                ours = len(kudio.resample(np.zeros(n, np.float32), orig_sr, target_sr,
                                          res_type='soxr_qq'))
                wrong += ours != n * target_sr // orig_sr
                extra += theirs == n * target_sr // orig_sr + 1
            if extra:
                more.append((orig_sr, target_sr, extra))
    report("resample: length, whole ones", lengths, wrong, float(wrong))
    family = {(a, b) for a in (11025, 22050, 44100, 88200, 176400)
              for b in (12000, 24000, 48000, 96000, 192000)}
    found = {(a, b) for a, b, _ in more}
    print(f"  librosa: one sample more for {len(more)} of the {pairs} pairs of rates, "
          f"{sum(e for *_, e in more)} of their {12 * len(more)} whole lengths; "
          f"{'exactly' if found == family else 'NOT'} the pairs from the 11025 Hz family to "
          f"the 12000 Hz one")


# -- 2. the short-time transform -------------------------------------------------------

FRAMES = [(64, 16, None), (64, 64, None), (65, 20, None), (65, 33, 40), (7, 2, None), (2, 1, None),
          (1, 1, None), (32, 40, None), (400, 160, None), (512, 128, None), (512, 256, 400),
          (512, 100, 333), (513, 171, None), (1024, 256, None), (2048, 512, 1200),
          (8192, 2048, None)]
WINDOWS = ['hann', 'hamming', 'blackman', 'boxcar', ('kaiser', 8.6), 8.6, ('tukey', 0.3),
           np.hanning]


def check_transform():
    forward, inverse, functions = Tally(), Tally(), Tally()
    for n_fft, hop, win_length in FRAMES:
        for window in WINDOWS:
            lengths = (sorted({1, hop - 1, hop, n_fft // 2, n_fft - 1, n_fft, n_fft + 1,
                               2 * n_fft + hop, 5001} - {0}) if window == 'hann'
                       else (n_fft + 1, 5001))
            for n in lengths:
                for dtype in ('f4', 'f8', 'f2'):
                    y = (0.1 * RNG.standard_normal(n)).astype(dtype)
                    want = quiet(librosa.stft, y, n_fft=n_fft, hop_length=hop,
                                 win_length=win_length, window=window)
                    got = quiet(_spectral.stft, y, n_fft, hop, win_length or n_fft, window)
                    forward.same(got, want, f"{n_fft}/{hop}/{win_length} {window} {dtype}, {n}")
        if n_fft == 1:
            continue
        for window in ('hann', 'hamming', ('kaiser', 8.6)):
            for frames in (1, 2, 3, 40):
                shape = (n_fft // 2 + 1, frames)
                z = RNG.standard_normal(shape) + 1j * RNG.standard_normal(shape)
                whole = n_fft + hop * (frames - 1) - 2 * (n_fft // 2)
                for dtype in (np.complex64, np.complex128):
                    for to in {None, whole, max(1, whole - 3), max(1, whole // 2), whole + 9}:
                        if not whole and to is None and n_fft // 2 < hop * frames:
                            continue
                        want = librosa.istft(z.astype(dtype), hop_length=hop, win_length=win_length,
                                             window=window, n_fft=n_fft, length=to)
                        got = _spectral.istft(z.astype(dtype), n_fft, hop, win_length or n_fft,
                                              window, to)
                        inverse.same(got, want, f"{n_fft}/{hop} {window} {frames} frames to {to}")
    eps = float(np.finfo(float).eps)
    for n_fft, hop, win_length in ((512, 256, None), (512, 128, 400), (400, 160, None),
                                   (1024, 300, None)):
        for dtype in ('f4', 'f8', 'i2', 'u1'):
            wave = (0.3 * np.sin(2 * np.pi * 300 * np.arange(9001) / 16000)
                    + 0.01 * RNG.standard_normal(9001))
            y = (wave.astype(dtype) if dtype[0] == 'f' else
                 np.round(wave * 100 + (128 if dtype == 'u1' else 0)).astype(dtype))
            x = _samples.as_float(y)
            for window in ('hamming', 'hann'):
                how = dict(n_fft=n_fft, hop_length=hop, win_length=win_length, window=window)
                spec = kudio.waveform_to_spectrogram(y, **how)
                functions.same(spec, np.log10(np.abs(librosa.stft(x, **how) + eps) ** 2).T,
                               "waveform_to_spectrogram")
                single = librosa.stft(np.asarray(x if dtype == 'u1' else y, np.float32),
                                      **how) + eps
                rebuilt = np.sqrt(10 ** spec).T * np.exp(1j * np.angle(single))
                functions.same(kudio.spectrogram_to_waveform(y, spec, **how),
                               librosa.istft(rebuilt, length=len(y), **how),
                               "spectrogram_to_waveform")
                stft = kudio.STFT(sr=16000, **how)
                functions.same(stft.analyse(y), librosa.stft(x.astype(np.float32), **how).T,
                               "STFT.analyse")
                z = stft.analyse(y) * np.float32(0.5)
                functions.same(stft.synthesise(z, length=len(y)),
                               librosa.istft(z.T, length=len(y), **how), "STFT.synthesise")
    forward.report("stft")
    inverse.report("istft")
    functions.report("the spectrogram functions")


# -- 3. mel bands and their cepstrum -----------------------------------------------------

def takes():
    t = np.arange(24000) / 16000
    speech = (0.3 * np.sin(2 * np.pi * 150 * t) * np.clip(np.sin(2 * np.pi * 3 * t), 0, 1)
              + 0.003 * RNG.standard_normal(len(t)))
    click = np.zeros(6000)
    click[3000] = 0.9
    return {'speech': speech, 'noise': 0.1 * RNG.standard_normal(16000), 'silence': np.zeros(5000),
            'click': click, 'constant': np.full(5000, 0.2), 'short': 0.1 * RNG.standard_normal(700),
            'loud': 1e4 * RNG.standard_normal(5000), 'quiet': 1e-7 * RNG.standard_normal(5000)}


def check_mel():
    filters, centres, cepstra, spectra = Tally(), Tally(), Tally(), Tally()
    for sr in (8000, 16000, 22050, 44100, 48000, 192000, 22254.5):
        for n_fft in (64, 257, 512, 1024, 2048, 8192):
            for n_mels in (1, 2, 23, 40, 64, 128, 256):
                for fmin in (0.0, 20.0, 125.5, 1000.0):
                    for fmax in (None, 3800.0, sr / 2, 0.9 * sr / 2, 1.5 * sr / 2):
                        want = quiet(librosa.filters.mel, sr=sr, n_fft=n_fft, n_mels=n_mels,
                                     fmin=fmin, fmax=fmax, htk=False, norm='slaney')
                        filters.same(quiet(_mel.filters, sr, n_fft, n_mels, fmin, fmax), want,
                                     f"{sr} {n_fft} {n_mels} {fmin} {fmax}")
    for n_mels in (1, 2, 40, 64, 128):
        for fmin, fmax in ((0.0, 8000.0), (125.5, 3800.0), (1000.0, 11025.0)):
            want = librosa.mel_frequencies(n_mels=n_mels + 2, fmin=fmin, fmax=fmax)[1:-1]
            centres.same(kudio.mel_band_hz(n_mels, 2 * fmax, fmin=fmin), want, f"{n_mels}")
            centres.same(kudio.mel_band_hz(n_mels, 48000, fmin=fmin, fmax=fmax), want, f"{n_mels}")
    eps = float(np.finfo(float).eps)
    for name, wave in takes().items():
        for dtype in ('f4', 'f8'):
            y = wave.astype(dtype)
            for sr in (8000, 16000, 44100):
                for given in ({}, dict(n_fft=512, hop_length=160, n_mels=40),
                              dict(n_mels=64, fmax=3800),
                              dict(n_fft=400, hop_length=100, n_mels=23, fmax=0.8 * sr / 2),
                              dict(n_fft=1024, n_mels=80, fmax=0.75 * sr)):
                    for n_mfcc in (1, 13, 20):
                        want = quiet(librosa.feature.mfcc, y=y, sr=sr, n_mfcc=n_mfcc, **given)
                        cepstra.same(quiet(kudio.mfcc, y, sr, n_mfcc, **given)[0], want.T,
                                     f"mfcc {name} {dtype} {sr} {given} {n_mfcc}")
                for n_mels, n_fft, hop, power in ((64, 1024, 512, 2.0), (40, 512, 128, 1.0),
                                                  (128, 2048, 512, 2), (80, 400, 160, 1.5)):
                    mel = quiet(librosa.feature.melspectrogram, y=y.astype(np.float32), sr=sr,
                                n_fft=n_fft, hop_length=hop, n_mels=n_mels, power=power)
                    want = 20.0 / power * np.log10(np.maximum(mel, eps))
                    got = quiet(kudio.melspectrogram, y, n_mels, 1, n_fft, hop, power, sr=sr)
                    spectra.same(got, want.T, f"mel {name} {dtype} {sr} {n_mels}")
    filters.report("mel filters")
    centres.report("mel band centres")
    cepstra.report("mfcc")
    spectra.report("melspectrogram")


# -- 4. silence ----------------------------------------------------------------------------

def cut_takes():
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


def check_silence():
    trimmed, split = Tally(), Tally()
    for name, wave in cut_takes().items():
        for dtype in ('f4', 'f8'):
            y = wave.astype(dtype)
            for top_db in (10, 20, 30.0, 40, 60, 80):
                for frame, hop in ((2048, 512), (512, 128), (401, 100), (64, 200), (256, 50000),
                                   (1, 1), (33, 1)):
                    how = dict(top_db=top_db, frame_length=frame, hop_length=hop)
                    got, cut = kudio.trim_silence(y, **how)
                    want, index = librosa.effects.trim(y.T, **how)
                    trimmed.begin()
                    trimmed.same(np.array(cut, np.int64), np.array([int(i) for i in index]),
                                 f"where trim cuts {name} {how}")
                    trimmed.same(got, want.T, f"trim {name} {dtype} {how}")
                    pieces = kudio.split_on_silence(y, **how)
                    cuts = librosa.effects.split(y.T, **how)
                    split.begin()
                    split.same(np.array([len(pieces)]), np.array([len(cuts)]),
                               f"split {name} {how}")
                    for piece, (a, b) in zip(pieces, cuts):
                        split.same(piece, y[a:b], f"split {name} {dtype} {how}")
    trimmed.report("trim_silence")
    split.report("split_on_silence")


# -- 5. pitch --------------------------------------------------------------------------------

def whole_search():
    """librosa 1.0's argument for searching every move, if it has one."""
    there = 'transition_min_prob' in inspect.signature(librosa.pyin).parameters
    return {'transition_min_prob': None} if there else {}


EVERY = 9


def some(cases, *key):
    cases = list(cases)
    return cases[zlib.crc32(repr(key).encode()) % EVERY::EVERY] or cases[:1]


def frame_for(sr, fmin):
    return max(2048, 1 << int(np.ceil(np.log2(2 * int(np.floor(sr / fmin)) + 2))))


def hard(kind, sr, n, rng):
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


def jumps(sr, seconds, rng, every=0.05):
    hz = rng.choice([80, 110, 160, 230, 320, 390, 700, 1200], size=int(seconds / every) + 1)
    phase = 2 * np.pi * np.cumsum(np.repeat(hz, int(every * sr))[:int(seconds * sr)]) / sr
    return (0.2 * sum(np.sin(k * phase) for k in range(1, 5))
            + 0.002 * rng.standard_normal(len(phase)))


class Pitch:
    """f0 against pyin, the whole search; and against 1.0's default, voicing apart."""

    def __init__(self):
        self.tally = Tally()
        self.default = []           # (what, frames apart, frames) where 1.0's default differs

    def compare(self, y, sr, what, fmin=65.0, fmax=400.0, frame=None, hop=None, center=True,
                hard_signal=False):
        frame, hop = frame or frame_for(sr, fmin), hop or int(sr // 100)
        how = dict(fmin=fmin, fmax=fmax, sr=sr, frame_length=frame, hop_length=hop, center=center)
        want = librosa.pyin(np.asarray(y, np.float32), **how, **whole_search())
        got = quiet(kudio.f0, y, sr, fmin=fmin, fmax=fmax, frame_length=frame, hop_length=hop,
                    center=center)
        self.tally.begin()
        self.tally.same(got.f0, want[0], f"f0 of {what}")
        self.tally.same(got.voiced, want[1], f"voiced of {what}")
        if tuple(int(n) for n in scipy.__version__.split('.')[:2]) >= (1, 12):
            self.tally.same(got.voiced_prob, want[2], f"voiced_prob of {what}")
        if hard_signal and whole_search():
            plain = librosa.pyin(np.asarray(y, np.float32), **how)
            frames = np.flatnonzero(plain[1] != got.voiced).tolist()
            if frames:
                self.default.append((f"{what}, {sr} Hz, {fmin:g}-{fmax:g} Hz, center={center}",
                                     frames, len(plain[1])))


def check_pitch():
    p = Pitch()
    hard_cases = 0
    for sr in (8000, 16000, 22050, 48000):
        for kind in ('noise', 'silence', 'offset', 'tone', 'clicks', 'sweep', 'loud', 'quiet'):
            rng = np.random.default_rng(5)
            cases = []
            for seconds in (0.05, 0.5, 1.5):
                for dtype in ('f4', 'f8'):
                    for fmin, fmax, center in ((65.0, 400.0, True), (50.0, 2000.0, True),
                                               (100.0, 600.0, False)):
                        y = hard(kind, sr, int(sr * seconds), rng).astype(dtype)
                        if center or len(y) >= frame_for(sr, fmin):
                            cases.append((y, f"{kind}, {seconds} s {dtype}", fmin, fmax, center))
            hard_cases += len(cases)
            for y, what, fmin, fmax, center in some(cases, kind, sr):
                p.compare(y, sr, what, fmin, fmax, center=center, hard_signal=True)
    shown = p.tally.cases
    for sr in (8000, 16000, 48000):
        rng = np.random.default_rng(sr)
        cases = [(jumps(sr, 0.4, rng)[:max(600 * hop, int(2 * sr / fmin) + 1)], fmin, fmax, hop)
                 for hop in sorted({1, 4, 9, 18, 19, 37, int(0.00115 * sr), int(0.0012 * sr),
                                    int(0.0034 * sr), int(0.0036 * sr)})
                 for fmin, fmax in ((65.0, 400.0), (50.0, 2000.0))]
        for y, fmin, fmax, hop in some(cases, "hops", sr):
            p.compare(y, sr, "jumps", fmin, fmax, hop=hop)
    for sr in (8000, 16000, 48000):
        rng = np.random.default_rng(1)
        cases = [(y.astype(np.float32), hop)
                 for n in list(range(1, 40)) + [100, 500, 1000, 2047, 2048, 3000]
                 for y in (0.1 * rng.standard_normal(n), jumps(sr, 1.0, rng)[:n], np.full(n, 0.3))
                 for hop in (160, 72 * sr // 1000)]
        for y, hop in some(cases, "a few samples", sr):
            want = librosa.pyin(y, fmin=65.0, fmax=400.0, sr=sr, frame_length=2048, hop_length=hop,
                                **whole_search())
            got = _yin.track(y, sr, 65.0, 400.0, 2048, hop, True)
            p.tally.begin()
            for a, b, part in zip(got, want, ('f0', 'voiced', 'voiced_prob')):
                p.tally.same(a, b, f"{part} of {len(y)} samples, hop {hop}")
    for frame in (494, 495, 511, 512, 600, 1023, 1500, 2047, 2048, 3001, 4096):
        rng = np.random.default_rng(2)
        cases = [(jumps(16000, 0.5, rng), "jumps", hop, center)
                 for hop in (37, 160, 441, 1000) for center in (True, False)]
        cases += [(jumps(16000, 1.0, rng)[:n], f"{n} samples", None, False)
                  for n in (frame, frame + 1, frame + 159, frame + 160)]
        for y, what, hop, center in some(cases, "frames", frame):
            p.compare(y, 16000, what, frame=frame, hop=hop, center=center)
    for sr in (8000, 11025, 16000, 22254.5, 44100, 96000, 192000):
        rng = np.random.default_rng(3)
        t = np.arange(int(0.25 * sr)) / sr
        cases = []
        for fmin, fmax in ((65.0, 400.0), (30.0, 250.0), (400.0, sr / 2), (100.0, 127.0),
                           (sr / 4, sr / 2), (200.0, 401.0), (65.0, 2000.0)):
            for name, y in (('jumps', jumps(int(sr), 0.25, rng)),
                            ('noise', 0.1 * rng.standard_normal(len(t))),
                            ('fmin', 0.3 * np.sin(2 * np.pi * fmin * t)),
                            ('fmax', 0.3 * np.sin(2 * np.pi * 0.995 * fmax * t)),
                            ('both', 0.2 * np.sin(2 * np.pi * fmin * t)
                             + 0.2 * np.sin(2 * np.pi * 0.98 * fmax * t))):
                cases.append((y, name, fmin, fmax))
        for y, name, fmin, fmax in some(cases, "ranges", sr):
            p.compare(y, sr, name, fmin, fmax)
    rng = np.random.default_rng(4)
    y = jumps(16000, 1.0, rng)
    y /= 1.1 * np.abs(y).max()
    p.compare(y + 0.09, 16000, "an offset")
    p.compare(1e-3 * y - 5, 16000, "all offset")
    pcm = np.round(32767 * y).astype(np.int16)
    want = librosa.pyin((pcm / 32768).astype(np.float32), fmin=65.0, fmax=400.0, sr=16000,
                        hop_length=160, **whole_search())[0]
    for samples, what in ((pcm, "int16"), (pcm[:, np.newaxis], "a column of int16")):
        p.tally.begin()
        p.tally.same(kudio.f0(samples, 16000).f0, want, what)
    p.tally.report(f"f0 against pyin{', the whole search' if whole_search() else ''}")
    print(f"  of {hard_cases} hard signals {shown} compared, and {p.tally.cases - shown} "
          f"more{'' if EVERY == 1 else '; --all compares every case'}")
    if whole_search():
        print(f"  librosa {librosa.__version__}'s default search, transition_min_prob=1e-4, calls "
              f"frames the other way in {len(p.default)} of the hard signals compared:")
        for what, frames, total in p.default:
            print(f"    {what}: frames {frames} of {total}")
    return p


def voice_with_pauses(floor_db, sr=16000, seconds=60):
    """A voice of 150 Hz wavering a third of an octave either way, on for 1.2 s and
    off for 0.8, 26 dB under full scale, over white noise *floor_db* under it."""
    t = np.arange(seconds * sr) / sr
    hz = 150 * 2 ** (0.3 * np.sin(2 * np.pi * 0.9 * t))
    phase = 2 * np.pi * np.cumsum(hz) / sr
    voice = sum(np.sin(k * phase) / k for k in range(1, 9))
    voice *= 0.05 / np.sqrt(np.mean(voice ** 2))
    floor = 0.05 * 10 ** (-floor_db / 20) * np.random.default_rng(0).standard_normal(len(t))
    return (voice * ((t % 2.0) < 1.2) + floor).astype(np.float32)


def check_pitch_hops():
    """At 1.0's default search, how many frames of a minute are voiced otherwise."""
    if not whole_search():
        print("  (librosa 0.11 has only the whole search; no default search to measure)")
        return
    sr, cases, apart, worst = 16000, 0, 0, 0
    print(f"  a minute of a voice with pauses, 16 kHz, 65-400 Hz: frames librosa "
          f"{librosa.__version__}'s default search voices otherwise than f0 does")
    for floor_db in (60, 40):
        y, rows = voice_with_pauses(floor_db), []
        for ms in (5, 10, 15, 20, 25, 30, 40, 45, 50, 60, 70):
            hop = sr * ms // 1000
            got = quiet(kudio.f0, y, sr, hop_length=hop)
            how = dict(fmin=65.0, fmax=400.0, sr=sr, frame_length=2048, hop_length=hop)
            plain = librosa.pyin(y, **how)[1]
            whole = librosa.pyin(y, **how, **whole_search())[1]
            cases += 1
            frames = int(np.count_nonzero(whole != got.voiced))
            apart, worst = apart + (frames > 0), max(worst, frames)
            rows.append(f"{ms} ms {int(np.count_nonzero(plain != got.voiced))}/{len(plain)}")
        print(f"    floor {floor_db} dB under the voice: " + ", ".join(rows))
    report("f0 hops of 5-70 ms, the whole search (frames)", cases, apart, float(worst))


# -- 6. time stretch ----------------------------------------------------------------------------

def check_stretch():
    limit = 7e-3 if librosa.__version__.startswith('0.') else 5e-6
    inside_worst, end_worst, cases = 0.0, 0.0, 0
    for sr, hz in ((8000, 120), (16000, 150), (44100, 220)):
        for rate in (0.5, 0.8, 0.9, 1.1, 1.25, 2.0):
            t = np.arange(2 * sr) / sr
            phase = 2 * np.pi * np.cumsum(hz * (1 + 0.04 * np.sin(2 * np.pi * 3 * t))) / sr
            y = (0.1 * sum(np.sin(k * phase) / k for k in range(1, 15))
                 * np.clip(2 * np.sin(2 * np.pi * 1.5 * t), 0.02, 1)
                 + 1e-3 * np.random.default_rng(1).standard_normal(len(t))).astype(np.float32)
            want = quiet(librosa.effects.time_stretch, y, rate=rate)
            got = kudio.time_stretch(y, rate)
            cases += 1
            if got.shape != want.shape or got.dtype != want.dtype:
                inside_worst = math.inf
                continue
            cut = int(2048 + 1024 / min(rate, 1))
            apart = np.abs(got.astype(np.float64) - want) / np.abs(want).max()
            inside_worst = max(inside_worst, float(apart[:-cut].max()))
            end_worst = max(end_worst, float(apart[-cut:].max()))
    report("time_stretch of a voice, but its end (of the peak)", cases, 0, inside_worst, limit)
    report("time_stretch of a voice, its end (of the peak)", cases, 0, end_worst, math.inf)


def best_sine(x, sr, near):
    """(cents off *near*, SINAD in dB) of the best sine in *x*, its frequency free."""
    n = len(x)
    t = (np.arange(n) - n / 2) / sr
    spec = np.abs(np.fft.rfft(x * np.hanning(n), 4 * n))
    f = np.argmax(spec) * sr / (4 * n)
    for _ in range(30):
        c, s = np.cos(2 * np.pi * f * t), np.sin(2 * np.pi * f * t)
        a, b = np.linalg.lstsq(np.stack([c, s], 1), x, rcond=None)[0]
        d = 2 * np.pi * t * (-a * s + b * c)
        step = np.linalg.lstsq(np.stack([c, s, d], 1), x, rcond=None)[0][2]
        f += step
        if abs(step) < 1e-10 * f:
            break
    basis = np.stack([np.cos(2 * np.pi * f * t), np.sin(2 * np.pi * f * t)], 1)
    q = np.linalg.lstsq(basis, x, rcond=None)[0]
    left = x - basis @ q
    sinad = 10 * np.log10(np.sum(q ** 2) / 2 / (np.mean(left ** 2) + 1e-30))
    return 1200 * np.log2(f / near), sinad


def second_by_second(x, sr, hz):
    x = np.asarray(x, np.float64)[int(0.2 * sr):-int(0.2 * sr)]
    return np.array([best_sine(x[k * sr:(k + 1) * sr], sr, hz) for k in range(len(x) // sr)])


def check_tones(long):
    sr = 48000
    worst = 0.0
    print("  tones at 48 kHz, float32: cents off at the end (least to most over the clip), "
          "SINAD at the end / worst second")
    for seconds in (10, 60, 300) if long else (10,):
        for hz in (187.3, 3001.7, 20640.0):
            y = np.empty(seconds * sr, np.float32)
            for a in range(0, len(y), sr):
                y[a:a + sr] = 0.2 * np.sin(2 * np.pi * hz * np.arange(a, a + sr) / sr)
            for rate in (1.25, 0.5):
                row = []
                for name, stretch in ((f"librosa {librosa.__version__}",
                                       lambda v: quiet(librosa.effects.time_stretch, v, rate=rate)),
                                      ("kudio", lambda v: kudio.time_stretch(v, rate))):
                    v = second_by_second(stretch(y), sr, hz)
                    end = np.median(v[-3:], axis=0)
                    row.append(f"{name} {end[0]:+.3f} c ({v[:, 0].min():+.3f} to "
                               f"{v[:, 0].max():+.3f}), {end[1]:.1f} / {v[:, 1].min():.1f} dB")
                    if name == "kudio":
                        worst = max(worst, float(np.abs(v[:, 0]).max()))
                print(f"    {seconds} s, {hz} Hz, rate {rate}: " + " | ".join(row))
    report("time_stretch of tones, kudio's cents off", 6 * (3 if long else 1), 0, worst, 1e-3)


# -- the table ----------------------------------------------------------------------------------

def print_table():
    width = max(len(r[0]) for r in RESULTS)
    print(f"\n{'check':<{width}} {'cases':>6} {'apart':>6} {'largest':>9} {'limit':>7}")
    failed = False
    for check, cases, apart, diff, limit in RESULTS:
        over = diff > limit
        failed |= over
        shown = '-' if math.isinf(limit) else ('0' if not limit else
                                               f"{limit:.0e}".replace('e-0', 'e-'))
        print(f"{check:<{width}} {cases:>6} {apart:>6} {diff:9.2e} {shown:>7}"
              f"{'  PAST' if over else ''}")
    return failed


def main():
    global EVERY
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--all', action='store_true', help='every pitch case, not one in nine')
    parser.add_argument('--long', action='store_true', help='tones of 60 s and 300 s as well')
    args = parser.parse_args()
    if librosa is None:
        print("librosa is not installed, and every check here needs it: pip install librosa")
        return 1
    EVERY = 1 if args.all else 9
    start = time.perf_counter()
    print(f"kudio {kudio.__version__} from {Path(kudio.__file__).parent}; librosa "
          f"{librosa.__version__}, numpy {np.__version__}, scipy {scipy.__version__}, "
          f"Python {sys.version.split()[0]}")
    for check in (check_resampling, check_lengths, check_transform, check_mel, check_silence,
                  check_pitch, check_pitch_hops, check_stretch):
        began = time.perf_counter()
        print(f"\n{check.__name__[6:].replace('_', ' ')}:", flush=True)
        check()
        print(f"  {time.perf_counter() - began:.0f} s", flush=True)
    print("\ntones:", flush=True)
    check_tones(args.long)
    failed = print_table()
    print(f"\n{time.perf_counter() - start:.0f} s; "
          + ("SOME DIFFERENCES ARE PAST THEIR LIMIT" if failed
             else "every difference within its limit"))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
