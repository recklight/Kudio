# -*- coding: utf-8 -*-
"""STOI [THHJ11] and ESTOI [JT16] as pystoi 0.4.1 computes them, step for step: the
clip taken to 10 kHz by its Kaiser-windowed polyphase resampler, frames 40 dB under
the reference's loudest dropped, a 512-point STFT of 256-sample Hann frames, 15
one-third-octave bands from 150 Hz, and 30-frame segments.

The one step left out: pystoi's ESTOI adds noise of machine-epsilon size, drawn from
numpy's global random state, before each of its two normalisations, so that a
segment of zeros does not divide by zero. Drawn from the global state, it moves a
seeded script's other random numbers and changes ESTOI in the last digits from one
call to the next; here there is none. With pystoi's ``np.random.standard_normal``
put to zeros the two agree to the bit.

[THHJ11] C. H. Taal, R. C. Hendriks, R. Heusdens and J. Jensen, "An algorithm for
intelligibility prediction of time-frequency weighted noisy speech", IEEE Trans.
Audio, Speech, Lang. Process. 19(7), pp. 2125-2136, 2011.
[JT16] J. Jensen and C. H. Taal, "An algorithm for predicting the intelligibility
of speech masked by modulated noise maskers", IEEE/ACM Trans. Audio, Speech, Lang.
Process. 24(11), pp. 2009-2022, 2016.

Written after pystoi, under its licence:

MIT License

Copyright (c) 2018 Pariente Manuel

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from __future__ import annotations

import functools
import warnings

import numpy as np

EPS = np.finfo("float").eps
FS = 10000            # the rate STOI is computed at
N_FRAME = 256         # frame length
NFFT = 512            # FFT size
NUMBAND = 15          # one-third-octave bands
MINFREQ = 150         # centre of the first band, Hz
N = 30                # frames in a segment
BETA = -15.           # lower SDR bound
DYN_RANGE = 40        # frames this far under the loudest are silence
#: what pystoi returns when fewer than N frames are left
TOO_SHORT = 1e-5


def _resample_window(p, q):
    """The Kaiser-windowed sinc of MATLAB's resample, as pystoi builds it."""
    gcd = np.gcd(p, q)
    if gcd > 1:
        p /= gcd
        q /= gcd
    log10_rejection = -3.0
    stopband_cutoff_f = 1.0 / (2 * max(p, q))
    roll_off_width = stopband_cutoff_f / 10
    rejection_dB = -20 * log10_rejection
    L = np.ceil((rejection_dB - 8) / (28.714 * roll_off_width))
    t = np.arange(-L, L + 1)
    ideal_filter = 2 * p * stopband_cutoff_f * np.sinc(2 * stopband_cutoff_f * t)
    if (rejection_dB >= 21) and (rejection_dB <= 50):
        beta = 0.5842 * (rejection_dB - 21) ** 0.4 + 0.07886 * (rejection_dB - 21)
    elif rejection_dB > 50:
        beta = 0.1102 * (rejection_dB - 8.7)
    else:
        beta = 0.0
    return np.kaiser(2 * L + 1, beta) * ideal_filter


def _resample(x, p, q):
    from scipy.signal import resample_poly
    h = _resample_window(p, q)
    return resample_poly(x, p, q, window=h / np.sum(h))


@functools.lru_cache(maxsize=None)
def _third_octaves(fs, nfft, num_bands, min_freq):
    f = np.linspace(0, fs, nfft + 1)
    f = f[: int(nfft / 2) + 1]
    k = np.array(range(num_bands)).astype(float)
    cf = np.power(2.0 ** (1.0 / 3), k) * min_freq
    freq_low = min_freq * np.power(2.0, (2 * k - 1) / 6)
    freq_high = min_freq * np.power(2.0, (2 * k + 1) / 6)
    obm = np.zeros((num_bands, len(f)))
    for i in range(len(cf)):
        f_bin = np.argmin(np.square(f - freq_low[i]))
        freq_low[i] = f[f_bin]
        fl_ii = f_bin
        f_bin = np.argmin(np.square(f - freq_high[i]))
        freq_high[i] = f[f_bin]
        fh_ii = f_bin
        obm[i, fl_ii:fh_ii] = 1
    return obm, cf


def _stft(x, win_size, fft_size, overlap=4):
    hop = int(win_size / overlap)
    w = np.hanning(win_size + 2)[1:-1]
    return np.array([np.fft.rfft(w * x[i:i + win_size], n=fft_size)
                     for i in range(0, len(x) - win_size, hop)])


def _overlap_and_add(x_frames, hop):
    num_frames, framelen = x_frames.shape
    segments = -(-framelen // hop)
    signal = np.pad(x_frames, ((0, segments), (0, segments * hop - framelen)))
    signal = signal.reshape((num_frames + segments, segments, hop))
    signal = np.transpose(signal, [1, 0, 2])
    signal = signal.reshape((-1, hop))
    signal = signal[:-segments]
    signal = signal.reshape((segments, num_frames + segments - 1, hop))
    signal = np.sum(signal, axis=0)
    end = (len(x_frames) - 1) * hop + framelen
    return signal.reshape(-1)[:end]


def _remove_silent_frames(x, y, dyn_range, framelen, hop):
    w = np.hanning(framelen + 2)[1:-1]
    x_frames = np.array([w * x[i:i + framelen] for i in range(0, len(x) - framelen, hop)])
    y_frames = np.array([w * y[i:i + framelen] for i in range(0, len(x) - framelen, hop)])
    x_energies = 20 * np.log10(np.linalg.norm(x_frames, axis=1) + EPS)
    mask = (np.max(x_energies) - dyn_range - x_energies) < 0
    return (_overlap_and_add(x_frames[mask], hop), _overlap_and_add(y_frames[mask], hop))


def _row_col_normalize(x):
    """Each segment's rows and then its columns to zero mean and unit norm,
    pystoi's steps less its noise."""
    # where pystoi adds its noise, + 0.0: it makes a -0.0 0.0, as the noise did
    x_normed = x + 0.0
    x_normed -= np.mean(x_normed, axis=-1, keepdims=True)
    x_inv = 1. / np.sqrt(np.sum(np.square(x_normed), axis=-1, keepdims=True))
    x_diags = np.array([np.diag(x_inv[i].reshape(-1)) for i in range(x_inv.shape[0])])
    x_normed = np.matmul(x_diags, x_normed)
    x_normed += 0.0
    x_normed -= np.mean(x_normed, axis=1, keepdims=True)
    x_inv = 1. / np.sqrt(np.sum(np.square(x_normed), axis=1, keepdims=True))
    x_diags = np.array([np.diag(x_inv[i].reshape(-1)) for i in range(x_inv.shape[0])])
    return np.matmul(x_normed, x_diags)


def stoi(x, y, fs_sig, extended=False):
    """STOI, or ESTOI with *extended*, of *y* against the clean *x*, both 1-D of one
    length at *fs_sig*. Fewer than 30 frames left after the silent ones are dropped
    gives 1e-5 with a RuntimeWarning, as pystoi does."""
    if x.shape != y.shape:
        raise ValueError(f"stoi(): x and y have to be one length, got {x.shape} and {y.shape}")
    if fs_sig != FS:
        x = _resample(x, FS, fs_sig)
        y = _resample(y, FS, fs_sig)
    x, y = _remove_silent_frames(x, y, DYN_RANGE, N_FRAME, int(N_FRAME / 2))
    x_spec = _stft(x, N_FRAME, NFFT, overlap=2).transpose()
    y_spec = _stft(y, N_FRAME, NFFT, overlap=2).transpose()
    if x_spec.shape[-1] < N:
        warnings.warn("stoi(): fewer than 30 frames are left once the silent ones are "
                      "dropped; 1e-5", RuntimeWarning)
        return TOO_SHORT
    obm, _ = _third_octaves(FS, NFFT, NUMBAND, MINFREQ)
    x_tob = np.sqrt(np.matmul(obm, np.square(np.abs(x_spec))))
    y_tob = np.sqrt(np.matmul(obm, np.square(np.abs(y_spec))))
    x_segments = np.array([x_tob[:, m - N:m] for m in range(N, x_tob.shape[1] + 1)])
    y_segments = np.array([y_tob[:, m - N:m] for m in range(N, x_tob.shape[1] + 1)])
    if extended:
        x_n = _row_col_normalize(x_segments)
        y_n = _row_col_normalize(y_segments)
        return np.sum(x_n * y_n / N) / x_n.shape[0]
    normalization_consts = (np.linalg.norm(x_segments, axis=2, keepdims=True)
                            / (np.linalg.norm(y_segments, axis=2, keepdims=True) + EPS))
    y_segments_normalized = y_segments * normalization_consts
    clip_value = 10 ** (-BETA / 20)
    y_primes = np.minimum(y_segments_normalized, x_segments * (1 + clip_value))
    y_primes = y_primes - np.mean(y_primes, axis=2, keepdims=True)
    x_segments = x_segments - np.mean(x_segments, axis=2, keepdims=True)
    y_primes /= (np.linalg.norm(y_primes, axis=2, keepdims=True) + EPS)
    x_segments /= (np.linalg.norm(x_segments, axis=2, keepdims=True) + EPS)
    correlations_components = y_primes * x_segments
    J = x_segments.shape[0]
    M = x_segments.shape[1]
    return np.sum(correlations_components) / (J * M)
