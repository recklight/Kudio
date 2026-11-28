# -*- coding: utf-8 -*-
"""Traditional (non-DNN) speech enhancement.

``trad_enhance`` is a fixed recipe put together from methods in Loizou,
*Speech Enhancement: Theory and Practice*, 2nd ed., CRC Press, 2013: an MCRA
noise estimate (Sec. 9.4.3.2), power spectral subtraction with an
over-subtraction factor set from each frame's SNR (Sec. 5.4), then a
Wiener-style gain. ``wavelet_low_pass_filter`` is a soft-threshold wavelet
denoiser.
"""
from __future__ import annotations

import numpy as np

from kudio._numbers import within
from kudio._samples import finite, one_channel
from kudio.core._policy import sample_rate
from kudio.enhance import _dwt
from kudio.exceptions import FeatureError

__all__ = ['trad_enhance', 'wavelet_low_pass_filter']


def wavelet_low_pass_filter(y: np.ndarray, thres: float = 0.8,
                            wavelet: str = 'db2') -> np.ndarray:
    """Soft-threshold the detail coefficients of a wavelet decomposition.

    The clip is taken down with periodization as many levels as its length
    allows, every detail coefficient is moved towards zero by *thres* times the
    clip's peak, ``max |y|``, and the clip is put back together. The result has
    the input's length, in float32 for float16 and float32 audio and in float64
    otherwise. A silent clip, and one too short for a single level (fewer than
    2(L - 1) samples for a filter of L taps: 6 for db2), come back as they went
    in, as a copy.

    :param wavelet: a discrete wavelet by name, in any case: ``haar``,
        ``db1`` to ``db38``, ``sym2`` to ``sym20``, ``coif1`` to ``coif17``,
        ``bior1.1`` to ``bior6.8``, ``rbio1.1`` to ``rbio6.8`` or ``dmey``,
        106 in all, with PyWavelets' filters and its results to the bit; or a
        ``pywt.Wavelet``.
    :raises ValueError: for a name that is none of those.
    :raises TypeError: for no name, or a value that is neither a name nor a
        wavelet.
    """
    y = one_channel(y, 'wavelet_low_pass_filter', FeatureError)
    if y.ndim != 1:
        raise FeatureError(f"wavelet_low_pass_filter() takes mono audio, got shape {y.shape}")
    finite(y, 'wavelet_low_pass_filter', FeatureError)
    thres = within(thres, 'thres', 'wavelet_low_pass_filter', FeatureError, low=0)
    bank = _dwt.filter_bank(wavelet, 'wavelet_low_pass_filter')
    # the peak, not the signed maximum: a clip below zero throughout would get
    # a negative threshold and come back unfiltered
    level = thres * float(np.nanmax(np.abs(y))) if y.size else 0.0
    if not level > 0:
        # a threshold of zero divides zero by zero below: NaN everywhere
        return y.copy()
    coeff = _dwt.wavedec(y, bank)
    if len(coeff) == 1:
        return y.copy()
    for k in range(1, len(coeff)):
        magnitude = np.absolute(coeff[k])
        with np.errstate(divide='ignore'):
            # a coefficient of zero gives 1 - inf, which the clip makes 0
            keep = 1 - level / magnitude
            keep.clip(min=0, max=None, out=keep)
            coeff[k] = coeff[k] * keep
    # periodization rounds an odd length up by one sample
    return _dwt.waverec(coeff, bank)[:len(y)]


def trad_enhance(y: np.ndarray, sr: int) -> np.ndarray:
    """Enhance waveform *y* using spectral subtraction with MCRA noise
    estimation and a Wiener-style gain function.

    Returns float32 of the same length as *y*. The last partial frame, under
    256 samples, has no frame of its own and comes back silent.

    :raises FeatureError: for anything but mono audio, or a clip shorter than
        one 512-sample frame, which would come back as silence.
    """
    sample_rate(sr, 'trad_enhance', FeatureError)
    y = np.asarray(one_channel(y, 'trad_enhance', FeatureError), dtype=np.float64)
    if y.ndim != 1:
        raise FeatureError(f"trad_enhance() takes mono audio, got shape {y.shape}")
    finite(y, 'trad_enhance', FeatureError)
    if len(y) < 512:
        raise FeatureError(f"trad_enhance() needs at least one 512-sample frame, "
                           f"got {len(y)} samples")
    len_ = 512           # frame size in samples
    PERC = 50            # window overlap as percent of frame
    len1 = len_ * PERC // 100
    len2 = len_ - len1
    Expnt = 2.0          # 1.0: magnitude domain, 2.0: power domain
    beta = 0.002
    win = np.hamming(len_)
    winGain = len2 / sum(win)
    nFFT = 2 * 2 ** 8

    k = 1
    x_old = np.zeros(len1)
    Nframes = len(y) // len2 - 1
    xfinal = np.zeros((Nframes + 1) * len2)

    para = {}
    for n in range(Nframes):
        insign = win * y[k - 1: k + len_ - 1]
        spec = np.fft.fft(insign, nFFT)
        sig = abs(spec)
        # a floor under the power: a frame of digital silence otherwise starts
        # the noise estimate at zero and every gain after it at 0/0
        ns_ps = sig ** 2 + 1e-12  # noisy power spectrum
        theta = np.angle(spec)    # noisy phase

        # noise estimation (MCRA)
        para = INIT(ns_ps).mcra() if n == 0 else EST(ns_ps, para).mcra()
        noise_mu = np.sqrt(para['noise_ps'])

        # posterior SNR; -inf for a frame of digital silence, which the
        # thresholds below read as the lowest SNR, so numpy need not warn
        with np.errstate(divide='ignore'):
            SNRpos = 10 * np.log10(
                np.linalg.norm(sig, 2) ** 2 / np.linalg.norm(noise_mu, 2) ** 2)

        # over-subtraction factor by SNR
        if Expnt == 1.0:
            alpha = 4 if SNRpos < -5.0 else (1 if SNRpos > 20 else 3 - SNRpos * 2 / 20)
        else:
            alpha = 5 if SNRpos < -5.0 else (1 if SNRpos > 20 else 4 - SNRpos * 3 / 20)

        sub_speech = sig ** Expnt - alpha * noise_mu ** Expnt

        # floor negative components at beta * noise
        diffw = sub_speech - beta * noise_mu ** Expnt
        negative = diffw < 0
        sub_speech[negative] = beta * noise_mu[negative] ** Expnt

        # priori SNR
        SNRpri = 10 * np.log10(
            np.linalg.norm(sub_speech ** (1 / Expnt), 2) ** 2
            / np.linalg.norm(noise_mu, 2) ** 2)

        # Wiener gain smoothing parameter
        mel_max = 10
        mel_0 = (1 + 4 * mel_max) / 5
        s = 25 / (mel_max - 1)
        mel = mel_max if SNRpri < -5.0 else (1 if SNRpri > 20 else mel_0 - SNRpri / s)

        g_k = sub_speech / (sub_speech + mel * noise_mu ** Expnt)
        wf_speech = g_k * sig

        # back to time domain with the noisy phase
        x_phase = wf_speech * np.exp(1j * theta)
        xi = np.fft.ifft(x_phase).real

        # overlap-add
        xfinal[k - 1: k + len2 - 1] = x_old + xi[0: len1]
        x_old = xi[len1: len_]
        k = k + len2

    # the last frame's second half has no later frame to overlap with;
    # dropped, it would leave the final 256 samples at zero
    if Nframes > 0:
        xfinal[k - 1: k - 1 + len1] = x_old
    out = np.zeros(len(y), dtype=np.float32)
    n = min(len(y), len(xfinal))
    out[:n] = winGain * xfinal[:n]
    return out


class INIT:
    """First-frame initialization of the MCRA noise-estimation state."""

    def __init__(self, ns_ps: np.ndarray):
        self.ns_ps = ns_ps

    def mcra(self) -> dict:
        """MCRA algorithm."""
        len_val = len(self.ns_ps)
        return {'n': 2, 'ad': 0.95, 'ass': 0.8, 'L': 1000 * 2 // 20, 'delta': 5,
                'ap': 0.2, 'leng': len_val, 'P': self.ns_ps, 'Pmin': self.ns_ps,
                'Ptmp': self.ns_ps, 'pk': np.zeros(len_val), 'noise_ps': self.ns_ps}


class EST:
    """Per-frame update of the MCRA noise-estimation state."""

    def __init__(self, ns_ps: np.ndarray, para: dict):
        self.ns_ps = ns_ps
        self.para = para

    def mcra(self) -> dict:
        """MCRA (Sec. 9.4.3.2)."""
        ass = self.para['ass']
        ad = self.para['ad']
        ap = self.para['ap']
        pk = self.para['pk']
        delta = self.para['delta']
        L = self.para['L']
        n = self.para['n']
        noise_ps = self.para['noise_ps']
        Pmin = self.para['Pmin']
        Ptmp = self.para['Ptmp']

        P = ass * self.para['P'] + (1 - ass) * self.ns_ps          # smoothed noisy power
        if n % L == 0:                                             # windowed minimum, L frames
            Pmin = np.minimum(Ptmp, P)
            Ptmp = P
        else:
            Pmin = np.minimum(Pmin, P)
            Ptmp = np.minimum(Ptmp, P)
        Srk = P / Pmin
        Ikl = (Srk > delta).astype(float)                          # speech present
        pk = ap * pk + (1 - ap) * Ikl                              # its probability
        adk = ad + (1 - ad) * pk                                   # smoothing factor
        noise_ps = adk * noise_ps + (1 - adk) * self.ns_ps

        self.para.update(pk=pk, n=n + 1, noise_ps=noise_ps, P=P, Pmin=Pmin, Ptmp=Ptmp)
        return self.para
