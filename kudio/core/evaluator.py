# -*- coding: utf-8 -*-
"""Speech-enhancement quality metrics: PESQ, STOI, SDR, and the SNR family.

The SNR family (:func:`snr`, :func:`si_sdr`, :func:`segmental_snr`,
:func:`sdi`) is pure numpy. The rest are optional dependencies::

    pip install kudio[eval]   # mir_eval + pystoi: SDR, STOI
    pip install kudio[pesq]   # pesq: PESQ -- built from source, needs a C compiler

**Which PESQ.** ITU-T P.862 scores narrowband speech and P.862.2 wideband, and
each can be reported raw or mapped to MOS-LQO. They are different numbers for
one file: a raw narrowband 2.0 is a MOS-LQO of about 1.6. Speech-enhancement
papers that just say "PESQ" usually mean the raw narrowband score, which is
what the ITU reference binary prints. :func:`pesq` takes the bandwidth and the
scale as arguments rather than picking one silently.
"""
from __future__ import annotations

import functools
import importlib.util
import logging
import math
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

from kudio.exceptions import DependencyError

__all__ = ['AudioEvaluate', 'eval_metrics', 'score', 'pesq', 'sdi', 'METRICS',
           'si_sdr', 'snr', 'segmental_snr']

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


def sdi(ref: np.ndarray, deg: np.ndarray) -> float:
    """Speech distortion index: ``sum((ref - deg)**2) / sum(ref**2)``.

    0 is a perfect copy and 1 is as much error as there is speech; lower is
    better. It is :func:`snr` on a linear scale, ``10 ** (-snr / 10)``, under
    the name the enhancement literature reports it by.
    """
    ref, deg = _align(ref, deg)
    return float(np.sum((ref - deg) ** 2) / (np.sum(ref ** 2) + 1e-12))


# --- PESQ ---------------------------------------------------------------------
PESQ_MODES = ('auto', 'nb', 'wb')
PESQ_SCALES = ('lqo', 'raw')


def _pesq_mode(sr: int, mode: str, scale: str) -> str:
    """The bandwidth :func:`pesq` will run at, or the reason it cannot."""
    if mode not in PESQ_MODES:
        raise ValueError(f"PESQ mode must be one of {PESQ_MODES}, got {mode!r}")
    if scale not in PESQ_SCALES:
        raise ValueError(f"PESQ scale must be one of {PESQ_SCALES}, got {scale!r}")
    if sr not in (8000, 16000):
        raise ValueError(f"PESQ is defined at 8 and 16 kHz, not {sr} Hz; "
                         f"resample first (kudio.resample)")
    if mode == 'auto':
        # P.862.2 has no raw score, so a raw score is a narrowband one
        mode = 'wb' if sr == 16000 and scale == 'lqo' else 'nb'
    if mode == 'wb' and sr != 16000:
        raise ValueError("wideband PESQ (P.862.2) needs 16 kHz audio")
    if mode == 'wb' and scale == 'raw':
        raise ValueError("P.862.2 defines no raw score: use scale='lqo', or "
                         "mode='nb' for the raw P.862 score")
    return mode


def pesq(ref: np.ndarray, deg: np.ndarray, sr: int, *, mode: str = 'auto',
         scale: str = 'lqo') -> float:
    """PESQ of *deg* against *ref* (ITU-T P.862 / P.862.2).

    >>> kudio.pesq(clean, enhanced, 16000)                          # WB MOS-LQO
    >>> kudio.pesq(clean, enhanced, 16000, mode='nb', scale='raw')  # papers

    :param mode: ``'nb'`` -- P.862 narrowband, at 8 or 16 kHz. ``'wb'`` --
        P.862.2 wideband, 16 kHz only. ``'auto'`` (default) -- wideband at
        16 kHz and narrowband at 8 kHz, which is what kudio has always
        reported; narrowband whenever *scale* is ``'raw'``.
    :param scale: ``'lqo'`` (default) -- MOS-LQO, the mapped score both
        standards define. ``'raw'`` -- the narrowband P.862 score before the
        P.862.1 mapping, -0.5 to 4.5: the number the ITU reference binary
        prints, and the one most enhancement papers mean by PESQ.
    :raises ValueError: off 8 / 16 kHz, or for a combination the standards
        do not define (wideband at 8 kHz, a raw wideband score).
    :raises DependencyError: without the ``pesq`` package (``kudio[pesq]``).

    Runs on the ``pesq`` package from PyPI, which returns MOS-LQO; the raw
    score is that inverted through P.862.1,
    ``(4.6607 - ln(4 / (lqo - 0.999) - 1)) / 1.4945``.
    """
    mode = _pesq_mode(int(sr), mode, scale)
    try:
        from pesq import pesq as _pesq
    except ImportError as e:
        raise DependencyError('pesq', extra='pesq') from e
    ref = np.asarray(ref, dtype=np.float64).squeeze()
    deg = np.asarray(deg, dtype=np.float64).squeeze()
    lqo = float(_pesq(int(sr), ref, deg, mode))
    if scale == 'lqo':
        return lqo
    if not 0.999 < lqo < 4.999:              # outside the mapping's range
        return float('nan')
    return (4.6607 - math.log(4.0 / (lqo - 0.999) - 1.0)) / 1.4945


# --- every metric by name -----------------------------------------------------
#: the names :func:`score` takes. The first four need nothing installed.
METRICS = ('snr', 'si_sdr', 'segsnr', 'sdi', 'pesq', 'stoi', 'estoi', 'sdr')

#: optional metric -> (module it needs, the extra that installs it)
_BACKEND = {'pesq': ('pesq', 'pesq'), 'stoi': ('pystoi', 'eval'),
            'estoi': ('pystoi', 'eval'), 'sdr': ('mir_eval', 'eval')}


def _compute(name: str, ref: np.ndarray, deg: np.ndarray, sr: int,
             pesq_mode: str = 'auto', pesq_scale: str = 'lqo') -> float:
    """One metric on two already-aligned signals."""
    if name == 'snr':
        return snr(ref, deg)
    if name == 'si_sdr':
        return si_sdr(ref, deg)
    if name == 'segsnr':
        return segmental_snr(ref, deg)
    if name == 'sdi':
        return sdi(ref, deg)
    if name == 'pesq':
        return pesq(ref, deg, sr, mode=pesq_mode, scale=pesq_scale)
    if name in ('stoi', 'estoi'):
        from pystoi import stoi
        return float(stoi(ref, deg, sr, extended=(name == 'estoi')))
    if name == 'sdr':
        import mir_eval
        value, *_ = mir_eval.separation.bss_eval_sources(
            ref.reshape(1, -1), deg.reshape(1, -1))
        return float(value[0])
    raise ValueError(f"unknown metric {name!r}; choose from {METRICS}")


def score(ref: np.ndarray, deg: np.ndarray, sr: int,
          metrics: Optional[Sequence[str]] = None, *,
          pesq_mode: str = 'auto', pesq_scale: str = 'lqo') -> Dict[str, float]:
    """Score *deg* against *ref*: ``{metric: value}``.

    >>> kudio.score(clean, enhanced, 16000, ['pesq', 'stoi', 'sdi'],
    ...             pesq_mode='nb', pesq_scale='raw')
    {'pesq': 2.41, 'stoi': 0.874, 'sdi': 0.105}

    :param metrics: names from :data:`METRICS`. ``None`` means every one that
        can run here: the four that need nothing, each optional one whose
        package is installed, and PESQ only at 8 or 16 kHz. A metric named
        explicitly that cannot run raises instead of going missing.
    :param pesq_mode, pesq_scale: as *mode* / *scale* of :func:`pesq`.

    The two signals are compared sample for sample over their common length,
    as the rest of kudio's reference metrics are -- see :func:`kudio.align`
    for a pair that may not line up.
    """
    if metrics is None:
        names = [m for m in METRICS
                 if (m not in _BACKEND or _has_module(_BACKEND[m][0]))
                 and (m != 'pesq' or int(sr) in (8000, 16000))]
    else:
        names = [metrics] if isinstance(metrics, str) else list(metrics)
        unknown = [m for m in names if m not in METRICS]
        if unknown:
            raise ValueError(f"unknown metric(s) {unknown}; choose from {METRICS}")
        for m in names:
            if m in _BACKEND and not _has_module(_BACKEND[m][0]):
                raise DependencyError(*_BACKEND[m])
    ref, deg = _align(np.asarray(ref).squeeze(), np.asarray(deg).squeeze())
    return {m: _compute(m, ref, deg, int(sr), pesq_mode, pesq_scale) for m in names}


# --- back-end availability ----------------------------------------------------
@functools.lru_cache(maxsize=None)
def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


#: back-ends already reported missing in this process
_REPORTED: set = set()


def check_metrics_install() -> Tuple[bool, bool, bool]:
    """Report availability of the optional metric back-ends.

    Returns ``(pesq_ok, stoi_ok, sdr_ok)``. A missing back-end is logged once
    per process, with how to install it; nothing is installed automatically.
    """
    pesq_ok = _has_module("pesq")
    stoi_ok = _has_module("pystoi")
    sdr_ok = _has_module("mir_eval")
    for ok, what, hint in (
            (pesq_ok, "PESQ", "pip install kudio[pesq] (builds from source: "
                              "needs a C compiler -- on Windows, the MSVC "
                              "Build Tools)"),
            (stoi_ok, "STOI", "pip install kudio[eval]"),
            (sdr_ok, "SDR", "pip install kudio[eval]")):
        if not ok and what not in _REPORTED:
            _REPORTED.add(what)
            log.warning("%s unavailable: %s", what, hint)
    return pesq_ok, stoi_ok, sdr_ok


# --- whole lists of files -----------------------------------------------------
#: AudioEvaluate's column -> score()'s metric name
_COLUMNS = {'PESQ': 'pesq', 'STOI': 'stoi', 'SDR': 'sdr'}


class AudioEvaluate:
    """Evaluate enhanced audio against clean references.

    Usage
    -----
    >>> audio_eval = AudioEvaluate(clean_list, noisy_list)
    >>> audio_eval.eval('ddae', enhanced_list)
    >>> df = audio_eval.get_df()       # pandas.DataFrame（含 Baseline 列）
    >>> per = audio_eval.per_file()    # one row per method x file

    File lists must be aligned: ``clean_list[i]`` / ``noisy_list[i]`` /
    ``enhanced_list[i]`` refer to the same utterance. Files are read with
    :func:`kudio.file_load` -- float, mono, and any format soundfile reads,
    NIST SPHERE included.

    :param pesq_mode, pesq_scale: as in :func:`kudio.pesq`. The defaults keep
        what this class has always reported, MOS-LQO and wideband at 16 kHz;
        ``pesq_mode='nb', pesq_scale='raw'`` is the raw P.862 score most
        enhancement papers quote.

    :meth:`get_df` has the means. :meth:`per_file` has every file, so a result
    can be broken down by noise, SNR or talker: join it to a
    :func:`kudio.save_manifest` manifest on the ``noisy`` column.
    """

    def __init__(self, clean_list: Sequence, noisy_list: Sequence, *,
                 pesq_mode: str = 'auto', pesq_scale: str = 'lqo'):
        if len(clean_list) != len(noisy_list):
            raise ValueError(f"{len(clean_list)} clean files but "
                             f"{len(noisy_list)} noisy ones: the lists pair "
                             f"up by position")
        # 16 kHz allows every combination the standards define, so this
        # refuses exactly the undefined ones before any file is read
        _pesq_mode(16000, pesq_mode, pesq_scale)
        self.enhanced_list: Optional[Sequence] = None
        self.clean_list = clean_list
        self.noisy_list = noisy_list
        self.pesq_mode = pesq_mode
        self.pesq_scale = pesq_scale

        self.pesq_status, self.stoi_status, self.sdr_status = check_metrics_install()

        self.score_dict = {"Enh_method": []}
        if self.pesq_status:
            self.score_dict["PESQ"] = []
        if self.stoi_status:
            self.score_dict["STOI"] = []
        if self.sdr_status:
            self.score_dict["SDR"] = []
        self.base_pesq = self.base_stoi = self.base_sdr = None
        #: one dict per method and file -- ``method``, ``clean``, ``noisy``,
        #: ``file`` (the one scored) and a value per metric, NaN where it
        #: failed. The noisy files themselves are scored once, as 'Baseline'.
        self.rows: List[dict] = []
        self._baseline: Optional[List[dict]] = None

    def _columns(self) -> List[str]:
        return [c for c in ('PESQ', 'STOI', 'SDR') if c in self.score_dict]

    def eval(self, enhance_method: str, enhanced_list: Sequence) -> None:
        """Score *enhanced_list* with every available metric."""
        self._check_length(enhanced_list)
        self.enhanced_list = enhanced_list
        columns = self._columns()
        if self._baseline is None:
            self._baseline = self._score_rows('Baseline', self.noisy_list, columns)
        rows = self._score_rows(enhance_method, enhanced_list, columns)
        self.rows.extend(rows)
        self.score_dict['Enh_method'].append(enhance_method)
        for column in columns:
            base, enhanced = _paired_means(self._baseline, rows, column)
            self.score_dict[column].append(np.around(enhanced, 4))
            setattr(self, f"base_{column.lower()}", base)

    def get_df(self):
        """Return the collected scores (plus a Baseline row) as a DataFrame."""
        if self.base_pesq is None and self.base_stoi is None and self.base_sdr is None:
            return []
        table = {key: list(values) for key, values in self.score_dict.items()}
        table['Enh_method'].append('Baseline')
        if self.base_pesq is not None:
            table['PESQ'].append(np.around(self.base_pesq, 4))
        if self.base_stoi is not None:
            table['STOI'].append(np.around(self.base_stoi, 4))
        if self.base_sdr is not None:
            table['SDR'].append(np.around(self.base_sdr, 4))
        return _pandas().DataFrame.from_dict(table, orient='columns')

    def per_file(self):
        """Every file's scores as a DataFrame, one row per method and file.

        The 'Baseline' rows score the noisy files; :attr:`rows` holds the
        method rows as plain dicts, for use without pandas.
        """
        return _pandas().DataFrame(list(self._baseline or []) + self.rows)

    def _check_length(self, files: Sequence) -> None:
        if len(files) != len(self.clean_list):
            raise ValueError(f"{len(files)} files to score against "
                             f"{len(self.clean_list)} clean references")

    def _score_rows(self, method: str, files: Sequence,
                    columns: Sequence[str]) -> List[dict]:
        from kudio.core.io import file_load
        rows = []
        for cln, noy, deg in tqdm(zip(self.clean_list, self.noisy_list, files),
                                  total=len(self.clean_list),
                                  desc=f"[kudio] Scoring {method}"):
            ref, sr = file_load(cln)
            out, sr_deg = file_load(deg)
            if sr_deg != sr:
                raise ValueError(f"{deg} is at {sr_deg} Hz and its reference "
                                 f"{cln} at {sr} Hz; resample one of them")
            ref, out = _align(ref, out)
            row = {'method': method, 'clean': str(cln), 'noisy': str(noy),
                   'file': str(deg)}
            for column in columns:
                try:
                    row[column] = _compute(_COLUMNS[column], ref, out, sr,
                                           self.pesq_mode, self.pesq_scale)
                except Exception as e:                      # noqa: BLE001
                    log.warning("%s failed on %s: %s", column, deg, e)
                    row[column] = float('nan')
            rows.append(row)
        return rows

    def _means(self, column: str):
        self._check_length(self.enhanced_list)
        base = self._score_rows('Baseline', self.noisy_list, [column])
        enhanced = self._score_rows('enhanced', self.enhanced_list, [column])
        return _paired_means(base, enhanced, column)

    def sdr_score(self):
        """``(baseline, enhanced)`` mean SDR over :attr:`enhanced_list`."""
        log.info("==== Evaluate SDR ====")
        return self._means('SDR')

    def stoi_score(self):
        """``(baseline, enhanced)`` mean STOI over :attr:`enhanced_list`."""
        log.info("==== Evaluate STOI ====")
        return self._means('STOI')

    def pesq_score(self):
        """``(baseline, enhanced)`` mean PESQ over :attr:`enhanced_list`."""
        log.info("==== Evaluate PESQ ====")
        return self._means('PESQ')

    def pesq_score_exe(self, pesq_exe: str = 'pesq.exe', rate: int = 16000):
        """Legacy PESQ via the ITU ``pesq.exe`` binary on PATH (Windows).

        The binary prints the raw narrowband P.862 score, the same scale as
        ``pesq_mode='nb', pesq_scale='raw'``.
        """
        log.info("==== Evaluate PESQ (pesq.exe) ====")
        rows = []
        for cln, noy, enh in tqdm(zip(self.clean_list, self.noisy_list,
                                      self.enhanced_list),
                                  total=len(self.clean_list)):
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


def _paired_means(base: List[dict], enhanced: List[dict], column: str):
    """Mean of *column* over the files where both sides have a value -- a file
    that failed on one side is left out of both, so the two stay comparable."""
    pairs = [(b[column], e[column]) for b, e in zip(base, enhanced)
             if np.isfinite(b[column]) and np.isfinite(e[column])]
    if not pairs:
        return float('nan'), float('nan')
    b, e = zip(*pairs)
    return float(np.mean(b)), float(np.mean(e))


def eval_metrics(rate: int, ref: np.ndarray, deg: np.ndarray, *,
                 pesq_mode: str = 'auto', pesq_scale: str = 'lqo'):
    """Score a single (reference, degraded) pair with every available metric.

    Returns ``(pesq, stoi, sdr)`` -- always three values, whatever is
    installed. A metric comes back as ``None`` when its back-end is not
    installed, when it cannot run on this pair -- PESQ off 8 and 16 kHz -- or
    when it fails on it, which is logged.

    :param pesq_mode: ``'nb'``, ``'wb'`` or ``'auto'`` (by sample rate).
    :param pesq_scale: ``'lqo'`` for MOS-LQO, or ``'raw'`` for the narrowband
        P.862 score most enhancement papers quote. The defaults are what kudio
        has always reported, so an existing caller's numbers do not move --
        but a caller that wants the scale the literature uses can now say so,
        which before 3.7.1 only :func:`pesq` and :func:`score` could.
    """
    pesq_status, stoi_status, sdr_status = check_metrics_install()
    ref, deg = _align(np.asarray(ref).squeeze(), np.asarray(deg).squeeze())
    wanted = [('pesq', pesq_status and int(rate) in (8000, 16000)),
              ('stoi', stoi_status), ('sdr', sdr_status)]
    values = []
    for name, ok in wanted:
        value = None
        if ok:
            try:
                value = _compute(name, ref, deg, int(rate),
                                 pesq_mode=pesq_mode, pesq_scale=pesq_scale)
            except Exception as e:                          # noqa: BLE001
                log.warning("%s could not be computed: %s", name.upper(), e)
        values.append(value)
    return tuple(values)


AudioEvaluate.eval_metrics = staticmethod(eval_metrics)
