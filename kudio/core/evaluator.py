# -*- coding: utf-8 -*-
"""Speech-enhancement quality metrics: PESQ, STOI, SDR.

The metric back-ends are optional dependencies; install them with::

    pip install kudio[eval]   # mir_eval + pystoi
    pip install https://github.com/schmiph2/pysepm/archive/master.zip  # PESQ
"""
from __future__ import annotations

import importlib.util
import logging
import os
import re
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
from scipy.io import wavfile
from tqdm import tqdm

from kudio.exceptions import DependencyError

__all__ = ['AudioEvaluate', 'eval_metrics', 'si_sdr', 'snr', 'segmental_snr']

log = logging.getLogger(__name__)


def _pandas():
    try:
        import pandas as pd
        return pd
    except ImportError as e:
        raise DependencyError('pandas', extra='data') from e


# --- lightweight, dependency-free metrics --------------------------------------
def _align(ref: np.ndarray, deg: np.ndarray):
    n = min(len(ref), len(deg))
    return ref[:n].astype(np.float64), deg[:n].astype(np.float64)


def snr(ref: np.ndarray, deg: np.ndarray) -> float:
    """Signal-to-noise ratio (dB) of *deg* relative to *ref* (pure numpy)."""
    ref, deg = _align(ref, deg)
    noise = ref - deg
    return float(10 * np.log10((np.sum(ref ** 2) + 1e-12) /
                               (np.sum(noise ** 2) + 1e-12)))


def si_sdr(ref: np.ndarray, deg: np.ndarray) -> float:
    """Scale-invariant SDR (dB) — immune to overall gain differences."""
    ref, deg = _align(ref, deg)
    ref = ref - ref.mean()
    deg = deg - deg.mean()
    proj = (np.dot(deg, ref) / (np.dot(ref, ref) + 1e-12)) * ref
    noise = deg - proj
    return float(10 * np.log10((np.sum(proj ** 2) + 1e-12) /
                               (np.sum(noise ** 2) + 1e-12)))


def segmental_snr(ref: np.ndarray, deg: np.ndarray, frame_length: int = 512,
                  hop_length: int = 256) -> float:
    """Frame-averaged (segmental) SNR in dB, clipped to [-10, 35] per frame."""
    ref, deg = _align(ref, deg)
    seg = []
    for start in range(0, len(ref) - frame_length + 1, hop_length):
        r = ref[start:start + frame_length]
        d = deg[start:start + frame_length]
        noise = r - d
        val = 10 * np.log10((np.sum(r ** 2) + 1e-12) / (np.sum(noise ** 2) + 1e-12))
        seg.append(np.clip(val, -10, 35))
    return float(np.mean(seg)) if seg else float('nan')


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def check_metrics_install() -> Tuple[bool, bool, bool]:
    """Report availability of the optional metric back-ends.

    Returns ``(pesq_ok, stoi_ok, sdr_ok)``. Missing back-ends are logged with
    installation hints; nothing is installed automatically.
    """
    pesq_ok = _has_module("pysepm")
    stoi_ok = _has_module("pystoi")
    sdr_ok = _has_module("mir_eval")
    if not pesq_ok:
        log.warning("PESQ unavailable: pip install "
                    "https://github.com/schmiph2/pysepm/archive/master.zip")
    if not stoi_ok:
        log.warning("STOI unavailable: pip install pystoi")
    if not sdr_ok:
        log.warning("SDR unavailable: pip install mir_eval")
    return pesq_ok, stoi_ok, sdr_ok


class AudioEvaluate:
    """Evaluate enhanced audio against clean references.

    Usage
    -----
    >>> audio_eval = AudioEvaluate(clean_list, noisy_list)
    >>> audio_eval.eval('ddae', enhanced_list)
    >>> df = audio_eval.get_df()   # pandas.DataFrame（含 Baseline 列）

    File lists must be aligned: ``clean_list[i]`` / ``noisy_list[i]`` /
    ``enhanced_list[i]`` refer to the same utterance.
    """

    def __init__(self, clean_list: Sequence, noisy_list: Sequence):
        self.enhanced_list: Optional[Sequence] = None
        self.clean_list = clean_list
        self.noisy_list = noisy_list

        self.pesq_status, self.stoi_status, self.sdr_status = check_metrics_install()

        self.score_dict = {"Enh_method": []}
        if self.pesq_status:
            self.score_dict["PESQ"] = []
        if self.stoi_status:
            self.score_dict["STOI"] = []
        if self.sdr_status:
            self.score_dict["SDR"] = []
        self.base_pesq = self.base_stoi = self.base_sdr = None

    def eval(self, enhance_method: str, enhanced_list: Sequence) -> None:
        """Score *enhanced_list* with every available metric."""
        self.enhanced_list = enhanced_list
        self.score_dict['Enh_method'].append(enhance_method)
        if self.pesq_status:
            self.base_pesq, pesq = self.pesq_score()
            self.score_dict['PESQ'].append(np.around(pesq, 4))
        if self.stoi_status:
            self.base_stoi, stoi = self.stoi_score()
            self.score_dict['STOI'].append(np.around(stoi, 4))
        if self.sdr_status:
            self.base_sdr, sdr = self.sdr_score()
            self.score_dict['SDR'].append(np.around(sdr, 4))

    def get_df(self):
        """Return the collected scores (plus a Baseline row) as a DataFrame."""
        if self.base_pesq is None and self.base_stoi is None and self.base_sdr is None:
            return []
        self.score_dict['Enh_method'].append('Baseline')
        if self.base_pesq is not None:
            self.score_dict['PESQ'].append(np.around(self.base_pesq, 4))
        if self.base_stoi is not None:
            self.score_dict['STOI'].append(np.around(self.base_stoi, 4))
        if self.base_sdr is not None:
            self.score_dict['SDR'].append(np.around(self.base_sdr, 4))
        return _pandas().DataFrame.from_dict(self.score_dict, orient='columns')

    def _iter_triples(self):
        yield from tqdm(zip(self.clean_list, self.noisy_list, self.enhanced_list),
                        total=len(self.clean_list))

    def sdr_score(self):
        import mir_eval
        log.info("==== Evaluate SDR ====")
        rows = []
        for cln, noy, enh in self._iter_triples():
            _, cln_wave = wavfile.read(cln)
            _, noy_wave = wavfile.read(noy)
            _, enh_wave = wavfile.read(enh)
            cln_wave = cln_wave.reshape(1, -1)
            sdr_base, *_ = mir_eval.separation.bss_eval_sources(
                cln_wave, noy_wave.reshape(1, -1))
            sdr_enh, *_ = mir_eval.separation.bss_eval_sources(
                cln_wave, enh_wave.reshape(1, -1))
            rows.append({'wave': os.path.basename(cln),
                         'Baseline-SDR': float(sdr_base),
                         'Enhance-SDR': float(sdr_enh)})
        mean = _pandas().DataFrame(rows).dropna().mean(numeric_only=True)
        log.info("sdr_mean:\n%s", mean)
        return mean['Baseline-SDR'], mean['Enhance-SDR']

    def stoi_score(self):
        from pystoi import stoi
        log.info("==== Evaluate STOI ====")
        rows = []
        for cln, noy, enh in self._iter_triples():
            sr, cln_wave = wavfile.read(cln)
            _, noy_wave = wavfile.read(noy)
            _, enh_wave = wavfile.read(enh)
            rows.append({'wave': os.path.basename(cln),
                         'Baseline-STOI': float(stoi(cln_wave, noy_wave, sr, extended=False)),
                         'Enhance-STOI': float(stoi(cln_wave, enh_wave, sr, extended=False))})
        mean = _pandas().DataFrame(rows).dropna().mean(numeric_only=True)
        log.info("stoi_mean:\n%s", mean)
        return mean['Baseline-STOI'], mean['Enhance-STOI']

    def pesq_score(self):
        import pysepm
        log.info("==== Evaluate PESQ ====")
        rows = []
        for cln, noy, enh in self._iter_triples():
            sr, cln_wave = wavfile.read(cln)
            _, noy_wave = wavfile.read(noy)
            _, enh_wave = wavfile.read(enh)
            try:
                rows.append({'wave': os.path.basename(cln),
                             'Baseline-PESQ': float(pysepm.pesq(cln_wave, noy_wave, sr)[1]),
                             'Enhance-PESQ': float(pysepm.pesq(cln_wave, enh_wave, sr)[1])})
            except Exception as e:
                log.warning("PESQ failed on %s: %s", cln, e)
                rows.append({'wave': os.path.basename(cln),
                             'Baseline-PESQ': np.nan, 'Enhance-PESQ': np.nan})
        mean = _pandas().DataFrame(rows).dropna().mean(numeric_only=True)
        log.info("pesq_mean:\n%s", mean)
        return mean['Baseline-PESQ'], mean['Enhance-PESQ']

    def pesq_score_exe(self, pesq_exe: str = 'pesq.exe', rate: int = 16000):
        """Legacy PESQ via the ITU ``pesq.exe`` binary on PATH (Windows)."""
        log.info("==== Evaluate PESQ (pesq.exe) ====")
        rows = []
        for cln, noy, enh in self._iter_triples():
            pesq_base = os.popen(f'{pesq_exe} +{rate} "{cln}" "{noy}"').readlines()[-1]
            pesq_enh = os.popen(f'{pesq_exe} +{rate} "{cln}" "{enh}"').readlines()[-1]
            try:
                rows.append({
                    'wave': os.path.basename(str(cln)),
                    'Baseline-PESQ': float(re.sub(r'^.*= ', '', pesq_base.strip())),
                    'Enhance-PESQ': float(re.sub(r'^.*= ', '', pesq_enh.strip()))})
            except ValueError:
                rows.append({'wave': os.path.basename(str(cln)),
                             'Baseline-PESQ': np.nan, 'Enhance-PESQ': np.nan})
        mean = _pandas().DataFrame(rows).dropna().mean(numeric_only=True)
        log.info("pesq_mean:\n%s", mean)
        return mean['Baseline-PESQ'], mean['Enhance-PESQ']


def eval_metrics(rate: int, ref: np.ndarray, deg: np.ndarray):
    """Score a single (reference, degraded) pair with every available metric.

    Returns ``(pesq, stoi, sdr)``; unavailable metrics come back as ``None``.
    """
    pesq_status, stoi_status, sdr_status = check_metrics_install()
    pesq_base, stoi_base, sdr_base = None, None, None

    # align lengths
    n = min(len(ref), len(deg))
    ref, deg = ref[:n], deg[:n]

    if pesq_status:
        import pysepm
        pesq_base = pysepm.pesq(ref, deg, rate)[1]

    if stoi_status:
        from pystoi import stoi
        stoi_base = stoi(ref, deg, rate, extended=False)

    if sdr_status:
        import mir_eval
        sdr, *_ = mir_eval.separation.bss_eval_sources(
            ref.reshape(1, -1), deg.reshape(1, -1))
        sdr_base = float(sdr[0])

    return pesq_base, stoi_base, sdr_base


AudioEvaluate.eval_metrics = staticmethod(eval_metrics)
