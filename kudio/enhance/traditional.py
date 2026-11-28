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
from kudio.exceptions import FeatureError

__all__ = ['trad_enhance', 'wavelet_low_pass_filter']


def wavelet_low_pass_filter(y: np.ndarray, thres: float = 0.8,
                            wavelet: str = 'db2') -> np.ndarray:
    """Soft-threshold the detail coefficients of a wavelet decomposition.

    The threshold is *thres* times the clip's peak, ``max |y|``. The result
    has the input's length; a silent clip comes back as it went in.
    """
    # float32 stays float32, as pywt keeps it
    y = one_channel(y, 'wavelet_low_pass_filter', FeatureError)
    if y.ndim != 1:
        raise FeatureError(f"wavelet_low_pass_filter() takes mono audio, got shape {y.shape}")
    finite(y, 'wavelet_low_pass_filter', FeatureError)
    thres = within(thres, 'thres', 'wavelet_low_pass_filter', FeatureError, low=0)
    # the peak, not the signed maximum: a clip below zero throughout got a
    # negative threshold and came back unfiltered
    level = thres * float(np.nanmax(np.abs(y))) if y.size else 0.0
    if not level > 0:
        # a threshold of zero makes pywt divide by zero: NaN everywhere
        return y.copy()
    # pywt brings scipy.fft and scipy.special with it, 0.23 s of importing
    # that `import kudio` would otherwise spend on every script
    import pywt
    coeff = pywt.wavedec(y, wavelet, mode='periodization')
    coeff[1:] = (pywt.threshold(i, value=level, mode='soft')
                 for i in coeff[1:])
    # periodization rounds an odd length up by one sample
    return pywt.waverec(coeff, wavelet, mode='periodization')[:len(y)]


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

    # the last frame's second half has no later frame to overlap with; it
    # used to be dropped, leaving the final 256 samples at zero
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
