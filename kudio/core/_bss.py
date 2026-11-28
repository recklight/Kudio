# -*- coding: utf-8 -*-
"""BSS Eval's decomposition of separated sources [VGF06], as mir_eval 0.8.2's
``bss_eval_sources`` computes it, step for step and call for call, so that the
numbers are mir_eval's to the bit: the least-squares projection through
``scipy.fftpack``, the Toeplitz Gram matrix, ``np.linalg.solve`` and, where
that matrix is singular, ``lstsq``, then ``fftconvolve``. Filters of 512 taps,
as mir_eval's.

[VGF06] E. Vincent, R. Gribonval and C. Fevotte, "Performance measurement in
blind audio source separation", IEEE Trans. Audio, Speech, Lang. Process.
14(4), pp. 1462-1469, 2006.

Written after mir_eval's ``separation`` module, under its licence:

The MIT License (MIT)

Copyright (c) 2014 Colin Raffel

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""
from __future__ import annotations

import itertools
import warnings
from typing import Tuple

import numpy as np

#: the filter length of the decomposition, in taps
FLEN = 512
#: more sources than this are refused, as mir_eval refuses them: their
#: permutations alone are past counting
MAX_SOURCES = 100


def _silent(sources: np.ndarray) -> bool:
    return bool(np.any(np.all(np.sum(sources, axis=tuple(range(2, sources.ndim))) == 0, axis=1)))


def _validate(refs: np.ndarray, ests: np.ndarray) -> None:
    if refs.shape != ests.shape:
        raise ValueError(f"bss_eval_sources(): the estimates are {ests.shape} and the "
                         f"references {refs.shape}; they have to match")
    if refs.ndim > 3 or ests.ndim > 3:
        raise ValueError(f"bss_eval_sources(): {max(refs.ndim, ests.ndim)} axes, at most 3")
    for label, y in (('a reference', refs), ('an estimate', ests)):
        if y.size and _silent(y):
            raise ValueError(f"bss_eval_sources(): {label} is all zeros")
    if max(refs.shape[0], ests.shape[0]) > MAX_SOURCES:
        raise ValueError(f"bss_eval_sources(): {max(refs.shape[0], ests.shape[0])} sources, "
                         f"at most {MAX_SOURCES}")


def bss_eval_sources(refs: np.ndarray, ests: np.ndarray, compute_permutation: bool = True
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """SDR, SIR, SAR of each estimate and the permutation, ``(nsrc, nsampl)`` each
    way. With *compute_permutation*, every estimate is decomposed against every
    reference and the permutation with the highest mean SIR is kept; without,
    estimate *j* goes with reference *j*."""
    if ests.ndim == 1:
        ests = ests[np.newaxis, :]
    if refs.ndim == 1:
        refs = refs[np.newaxis, :]
    _validate(refs, ests)
    if refs.size == 0 or ests.size == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])
    nsrc = ests.shape[0]
    if compute_permutation:
        sdr = np.empty((nsrc, nsrc))
        sir = np.empty((nsrc, nsrc))
        sar = np.empty((nsrc, nsrc))
        for jest in range(nsrc):
            for jtrue in range(nsrc):
                parts = _decompose(refs, ests[jest], jtrue, FLEN)
                sdr[jest, jtrue], sir[jest, jtrue], sar[jest, jtrue] = _criteria(*parts)
        perms = list(itertools.permutations(list(range(nsrc))))
        mean_sir = np.empty(len(perms))
        dum = np.arange(nsrc)
        for i, perm in enumerate(perms):
            mean_sir[i] = np.mean(sir[perm, dum])
        popt = perms[np.argmax(mean_sir)]
        idx = (popt, dum)
        return sdr[idx], sir[idx], sar[idx], np.asarray(popt)
    sdr = np.empty(nsrc)
    sir = np.empty(nsrc)
    sar = np.empty(nsrc)
    for j in range(nsrc):
        sdr[j], sir[j], sar[j] = _criteria(*_decompose(refs, ests[j], j, FLEN))
    return sdr, sir, sar, np.arange(nsrc)


def _decompose(refs: np.ndarray, est: np.ndarray, j: int, flen: int):
    """The target (reference *j*), its filtering distortion, the interference of
    the other references, and the artifacts that are left."""
    nsampl = est.size
    s_true = np.hstack((refs[j], np.zeros(flen - 1)))
    e_spat = _project(refs[j, np.newaxis, :], est, flen) - s_true
    e_interf = _project(refs, est, flen) - s_true - e_spat
    e_artif = -s_true - e_spat - e_interf
    e_artif[:nsampl] += est
    return s_true, e_spat, e_interf, e_artif


def _project(refs: np.ndarray, est: np.ndarray, flen: int) -> np.ndarray:
    """*est* projected onto the span of *refs* delayed by 0 to flen - 1 samples:
    the normal equations from inner products computed by FFT."""
    import scipy.fftpack
    from scipy.linalg import toeplitz
    from scipy.signal import fftconvolve

    nsrc, nsampl = refs.shape
    refs = np.hstack((refs, np.zeros((nsrc, flen - 1))))
    est = np.hstack((est, np.zeros(flen - 1)))
    n_fft = int(2 ** np.ceil(np.log2(nsampl + flen - 1.0)))
    sf = scipy.fftpack.fft(refs, n=n_fft, axis=1)
    sef = scipy.fftpack.fft(est, n=n_fft)
    G = np.zeros((nsrc * flen, nsrc * flen))
    for i in range(nsrc):
        for j in range(nsrc):
            ssf = np.real(scipy.fftpack.ifft(sf[i] * np.conj(sf[j])))
            ss = toeplitz(np.hstack((ssf[0], ssf[-1:-flen:-1])), r=ssf[:flen])
            G[i * flen:(i + 1) * flen, j * flen:(j + 1) * flen] = ss
            G[j * flen:(j + 1) * flen, i * flen:(i + 1) * flen] = ss.T
    D = np.zeros(nsrc * flen)
    for i in range(nsrc):
        ssef = np.real(scipy.fftpack.ifft(sf[i] * np.conj(sef)))
        D[i * flen:(i + 1) * flen] = np.hstack((ssef[0], ssef[-1:-flen:-1]))
    try:
        C = np.linalg.solve(G, D).reshape(flen, nsrc, order="F")
    except np.linalg.LinAlgError:
        # as mir_eval calls it, rcond left to numpy, whose default is not the same
        # in numpy 1 (which warns of it) and 2
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            C = np.linalg.lstsq(G, D)[0].reshape(flen, nsrc, order="F")
    sproj = np.zeros(nsampl + flen - 1)
    for i in range(nsrc):
        sproj += fftconvolve(C[:, i], refs[i])[:nsampl + flen - 1]
    return sproj


def _criteria(s_true, e_spat, e_interf, e_artif):
    """SDR, SIR and SAR in dB, the paper's eqs. 13, 14 and 16."""
    s_filt = s_true + e_spat
    sdr = _db(np.sum(s_filt ** 2), np.sum((e_interf + e_artif) ** 2))
    sir = _db(np.sum(s_filt ** 2), np.sum(e_interf ** 2))
    sar = _db(np.sum((s_filt + e_interf) ** 2), np.sum(e_artif ** 2))
    return sdr, sir, sar


def _db(num, den):
    if den == 0:
        return np.inf
    return 10 * np.log10(num / den)
