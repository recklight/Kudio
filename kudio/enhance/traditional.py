# -*- coding: utf-8 -*-
"""Traditional (non-DNN) speech enhancement.

``trad_enhance`` implements spectral subtraction / Wiener filtering with MCRA
noise estimation (equation numbers reference Loizou, *Speech Enhancement:
Theory and Practice*). ``wavelet_low_pass_filter`` is a soft-threshold wavelet
denoiser.
"""
from __future__ import annotations

import numpy as np
import pywt

__all__ = ['trad_enhance', 'wavelet_low_pass_filter']


def wavelet_low_pass_filter(y: np.ndarray, thres: float = 0.8,
                            wavelet: str = 'db2') -> np.ndarray:
    """Soft-threshold the detail coefficients of a wavelet decomposition."""
    coeff = pywt.wavedec(y, wavelet, mode='periodization')
    coeff[1:] = (pywt.threshold(i, value=thres * np.nanmax(y), mode='soft')
                 for i in coeff[1:])
    return pywt.waverec(coeff, wavelet, mode='periodization')


def trad_enhance(y: np.ndarray, sr: int) -> np.ndarray:
    """Enhance waveform *y* using spectral subtraction with MCRA noise
    estimation and a Wiener-style gain function."""
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
        ns_ps = sig ** 2          # noisy power spectrum
        theta = np.angle(spec)    # noisy phase

        # noise estimation (MCRA)
        para = INIT(ns_ps, sr).mcra() if n == 0 else EST(ns_ps, para).mcra()
        noise_mu = np.sqrt(para['noise_ps'])

        # posterior SNR
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

    return winGain * xfinal


class INIT:
    """First-frame initialization of the noise-estimation state."""

    def __init__(self, ns_ps: np.ndarray, fs: int):
        self.ns_ps = ns_ps
        self.fs = fs

    def weight(self) -> dict:
        """Weighted spectral average."""
        return {'ass': 0.85, 'beta': 1.5, 'noise_ps': self.ns_ps, 'P': self.ns_ps}

    def com_min_track(self) -> dict:
        """Continuous minimal tracking."""
        return {'n': 2, 'leng': len(self.ns_ps), 'alpha': 0.7, 'beta': 0.96,
                'gamma': 0.998, 'noise_ps': self.ns_ps, 'pxk_old': self.ns_ps,
                'pxk': self.ns_ps, 'pnk_old': self.ns_ps, 'pnk': self.ns_ps}

    def mcra(self) -> dict:
        """MCRA algorithm."""
        len_val = len(self.ns_ps)
        return {'n': 2, 'ad': 0.95, 'ass': 0.8, 'L': 1000 * 2 // 20, 'delta': 5,
                'ap': 0.2, 'leng': len_val, 'P': self.ns_ps, 'Pmin': self.ns_ps,
                'Ptmp': self.ns_ps, 'pk': np.zeros(len_val), 'noise_ps': self.ns_ps}

    def mcra2(self) -> dict:
        """MCRA-2 algorithm with a frequency-dependent delta [9.60]."""
        len_val = len(self.ns_ps)
        freq_res = self.fs / len_val
        k_1khz = int(1000 // freq_res)
        k_3khz = int(3000 // freq_res)

        low_1 = 2 * np.ones(k_1khz, dtype=int)
        low_2 = 2 * np.ones(k_3khz - k_1khz, dtype=int)
        high = 5 * np.ones(len_val // 2 - k_3khz, dtype=int)
        delta_val = np.concatenate((low_1, low_2, high, high, low_2, low_1))

        return {'n': 2, 'leng': len_val, 'ad': 0.95, 'ass': 0.8, 'ap': 0.2,
                'beta': 0.8, 'beta1': 0.98, 'gamma': 0.998, 'alpha': 0.7,
                'delta': delta_val, 'pk': np.zeros(len_val), 'noise_ps': self.ns_ps,
                'pxk_old': self.ns_ps, 'pxk': self.ns_ps, 'pnk_old': self.ns_ps,
                'pnk': self.ns_ps}


class EST:
    """Per-frame update of the noise-estimation state."""

    def __init__(self, ns_ps: np.ndarray, para: dict):
        self.ns_ps = ns_ps
        self.para = para

    def weight(self) -> dict:
        """Weighted spectral average [9.30]."""
        ass = self.para['ass']
        beta = self.para['beta']
        noise_ps = self.para['noise_ps']
        P = ass * self.para['P'] + (1 - ass) * self.ns_ps

        smaller = P < beta * noise_ps
        noise_ps[smaller] = ass * noise_ps[smaller] + (1 - ass) * P[smaller]

        self.para['P'] = P
        self.para['noise_ps'] = noise_ps
        return self.para

    def com_min_track(self) -> dict:
        """Continuous minimal tracking [9.24-9.25]."""
        alpha = self.para['alpha']
        beta = self.para['beta']
        gamma = self.para['gamma']
        pxk_old = self.para['pxk_old']
        pnk_old = self.para['pnk_old']
        pnk = self.para['pnk']

        pxk = alpha * pxk_old + (1 - alpha) * self.ns_ps
        for t in range(self.para['leng']):
            if pnk_old[t] <= pxk[t]:
                pnk[t] = (gamma * pnk_old[t]) + \
                    (((1 - gamma) / (1 - beta)) * (pxk[t] - beta * pxk_old[t]))
            else:
                pnk[t] = pxk[t]

        self.para['n'] += 1
        self.para['noise_ps'] = pnk
        self.para['pnk'] = pnk
        self.para['pnk_old'] = pnk
        self.para['pxk'] = pxk
        self.para['pxk_old'] = pxk
        return self.para

    def mcra(self) -> dict:
        """MCRA [9.53-9.59]."""
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

        P = ass * self.para['P'] + (1 - ass) * self.ns_ps          # [9.55]
        if n % L == 0:                                             # [9.23]
            Pmin = np.minimum(Ptmp, P)
            Ptmp = P
        else:
            Pmin = np.minimum(Pmin, P)
            Ptmp = np.minimum(Ptmp, P)
        Srk = P / Pmin                                             # [9.58]
        Ikl = (Srk > delta).astype(float)
        pk = ap * pk + (1 - ap) * Ikl                              # [9.59]
        adk = ad + (1 - ad) * pk                                   # [9.54]
        noise_ps = adk * noise_ps + (1 - adk) * self.ns_ps         # [9.53]

        self.para.update(pk=pk, n=n + 1, noise_ps=noise_ps, P=P, Pmin=Pmin, Ptmp=Ptmp)
        return self.para

    def mcra2(self) -> dict:
        """MCRA-2 [9.61, 9.25, 9.53-9.59]."""
        ad = self.para['ad']
        ap = self.para['ap']
        beta = self.para['beta']
        gamma = self.para['gamma']
        alpha = self.para['alpha']
        pk = self.para['pk']
        delta = self.para['delta']
        noise_ps = self.para['noise_ps']
        pnk = self.para['pnk']
        pxk_old = self.para['pxk_old']
        pnk_old = self.para['pnk_old']

        pxk = alpha * pxk_old + (1 - alpha) * self.ns_ps           # [9.61]
        for i in range(len(pnk)):                                  # [9.25]
            if pnk_old[i] < pxk[i]:
                pnk[i] = (gamma * pnk_old[i]) + \
                    (((1 - gamma) / (1 - beta)) * (pxk[i] - beta * pxk_old[i]))
        Srk = pxk / pnk                                            # [9.57]
        Ikl = (Srk > delta).astype(float)                          # [9.58]
        pk = ap * pk + (1 - ap) * Ikl                              # [9.59]
        adk = ad + (1 - ad) * pk                                   # [9.54]
        noise_ps = adk * noise_ps + (1 - adk) * pxk                # [9.53]

        self.para.update(n=self.para['n'] + 1, pk=pk, noise_ps=noise_ps,
                         pnk=pnk, pnk_old=pnk, pxk=pxk, pxk_old=pxk)
        return self.para
