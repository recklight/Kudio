# -*- coding: utf-8 -*-
"""Mel-cepstral distortion, as URGENT 2025's scoring script computes it.

MCD is the distance between the mel-cepstra of two signals, frame by frame,
after dynamic time warping has paired the frames up. The name covers several
different numbers: implementations disagree on the mel-cepstral analysis
(SPTK's, WORLD's, a mel filterbank's), on whether the energy term c0 counts,
on the order and the warping, and on how the frames are paired, and the
results cannot be set side by side. :func:`mcd` is one of them, pinned to the
code of the URGENT 2025 challenge (``evaluation_metrics/mcd_utils.py`` and
``mcd_metric`` in ``calculate_intrusive_se_metrics.py``,
github.com/urgent-challenge/urgent2025_challenge, last changed in commits
c726558 of 2024-11-02 and 93e16aa of 2024-11-19), which ports ESPnet's
``utils/mcd_calculate.py`` by Wen-Chin Huang and Tomoki Hayashi:

1. *deg* is scaled by its least-squares gain against *ref*,
   ``sum(ref deg) / (sum(deg^2) + 1e-8)``.
2. Each signal is cut into frames of 1024 samples every 256, without
   padding, each frame weighted by a Hamming window scaled to unit energy,
   and analysed into a mel-cepstrum of order M with all-pass constant alpha
   by SPTK's ``mcep``, the mel-cepstral analysis of Fukada et al. [FTKI92]
   solved by Newton-Raphson iteration: 1e-6 added to the periodogram, at
   most 30 iterations, stopping once the residual power moves by less than
   0.1%. M and alpha depend on the rate: 13 and 0.31 at 8 kHz, 23 and 0.42
   at 16, 34 and 0.45 at 22.05, 34 and 0.46 at 24, 36 and 0.50 at 32, 39
   and 0.53 at 44.1, 39 and 0.55 at 48, as mcd_utils.py sets them after
   SPTK's documentation.
3. The two sequences of mel-cepstra, c0 included, are paired by FastDTW
   (Salvador and Chan [SC07]) with a radius of 1 and the Euclidean distance,
   in the order the script passes them: the reference first.
4. MCD is the mean over the pairs of ``(10 / ln 10) sqrt(2 sum_d (c_d - c'_d)^2)``,
   in dB.

The mel-cepstral analysis is SPTK 3's ``mcep`` restated in numpy for every
frame at once, with the frequency warps as matrices and the
Toeplitz-plus-Hankel system solved directly; FastDTW follows the pure-Python
``fastdtw`` package (0.3.4). ``tools/validation/melcepstrum.py`` sets both
against pysptk and fastdtw themselves. Ten minutes at 16 kHz take about
16 s and 350 MB, most of it in the analysis.

    [FTKI92] T. Fukada, K. Tokuda, T. Kobayashi and S. Imai, "An adaptive
             algorithm for mel-cepstral analysis of speech", Proc. ICASSP
             1992, vol. 1, pp. 137-140, doi:10.1109/ICASSP.1992.225953.
    [SC07]   S. Salvador and P. Chan, "Toward accurate dynamic time warping
             in linear time and space", Intelligent Data Analysis 11(5),
             pp. 561-580, 2007, doi:10.3233/IDA-2007-11508.

What to watch for:

* The gain match needs the pair lined up sample for sample, as for
  :func:`kudio.lsd`; for speech synthesis or conversion, where the two
  differ in length, pass ``match_gain=False``.
* c0 is the frame's log energy, so a level difference the gain match does
  not remove counts in full. Other MCDs leave c0 out; this one keeps it,
  as URGENT does.
* FastDTW is an approximation of DTW. The compiled ``fastdtw`` breaks ties
  between equal accumulated costs another way than its pure-Python version,
  which this follows, and the two pair the frames differently where the
  frames repeat exactly, as in digital silence.
"""
from __future__ import annotations

import functools
import math
import numbers
from typing import List, Optional, Tuple

import numpy as np

from kudio.core import evaluator as _evaluator
from kudio.core._policy import faint, sample_rate, scored_nan, signal

__all__ = ['mcd', 'MCD_SETTINGS']

#: rate -> (order M, all-pass constant alpha), from mcd_utils.py's
#: _get_best_mcep_params, which takes them from SPTK's mgcep documentation
MCD_SETTINGS = {8000: (13, 0.31), 16000: (23, 0.42), 22050: (34, 0.45),
                24000: (34, 0.46), 32000: (36, 0.50), 44100: (39, 0.53),
                48000: (39, 0.55)}

#: SPTK mcep's defaults as mcd_utils.py calls it: periodogram floor (etype=1),
#: least and most iterations, relative change of the residual power to stop at
_PERIODOGRAM_EPS = 1e-6
_MIN_ITERATIONS = 2
_MAX_ITERATIONS = 30
_THRESHOLD = 0.001


# --- SPTK's mel-cepstral analysis ------------------------------------------------

@functools.lru_cache(maxsize=16)
def _warp(m1: int, m2: int, a: float, recursive: bool) -> np.ndarray:
    """The matrix of SPTK's ``freqt`` (or, with *recursive*, ``frqtr``), from
    a sequence of order *m1* to one of order *m2*: both are linear, and
    pushing every unit vector through the recursion at once gives it."""
    g = np.zeros((m2 + 1, m1 + 1))
    b = 1.0 - a * a
    unit = np.eye(m1 + 1)
    for i in range(-m1, 1):
        d = g.copy()
        g[0] = unit[-i]
        if not recursive:
            g[0] = g[0] + a * d[0]
            if m2 >= 1:
                g[1] = b * d[0] + a * d[1]
            start = 2
        else:
            start = 1
        for j in range(start, m2 + 1):
            g[j] = d[j - 1] + a * (d[j] - g[j - 1])
    g.setflags(write=False)
    return g


def _mel_cepstra(frames: np.ndarray, order: int, alpha: float) -> np.ndarray:
    """SPTK's ``mcep(frame, order, alpha, eps=1e-6, etype=1)`` of every row
    of *frames*, windowed already: ``(frames, order + 1)``."""
    count, length = frames.shape
    half = length // 2
    m2 = 2 * order
    # the periodogram and everything taken from it are even, so their half
    # spectra are all there is: rfft and irfft give SPTK's full-length fftr
    # and ifftr to the last digits, in half the time
    periodogram = np.abs(np.fft.rfft(frames, axis=1)) ** 2 + _PERIODOGRAM_EPS
    cepstrum = np.fft.irfft(np.log(periodogram), length, axis=1)[:, :half + 1]
    cepstrum[:, 0] /= 2.0
    cepstrum[:, half] /= 2.0
    to_mel = _warp(half, order, alpha, False)
    to_linear = _warp(order, half, -alpha, False)
    to_lags = _warp(half, m2, alpha, True)
    mc = cepstrum @ to_mel.T
    previous = cepstrum[:, 0].copy()
    powers = (-alpha) ** np.arange(order + 1)
    n = order + 1
    rows, cols = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
    going = np.arange(count)
    for iteration in range(1, _MAX_ITERATIONS + 1):
        linear = np.zeros((going.size, length))
        linear[:, :half + 1] = mc[going] @ to_linear.T
        log_gain = np.fft.rfft(linear, axis=1).real
        ratio = periodogram[going] / np.exp(log_gain + log_gain)
        lags = np.fft.irfft(ratio, length, axis=1)[:, :half + 1] @ to_lags.T
        power = lags[:, 0]
        if iteration >= _MIN_ITERATIONS:
            # as SPTK's test reads it, a NaN keeps the frame going
            moving = ~(np.abs((power - previous[going]) / power) < _THRESHOLD)
            previous[going] = power
            going, lags = going[moving], lags[moving]
            if going.size == 0:
                break
        rhs = lags[:, :n] - powers
        hankel = lags.copy()
        hankel[:, 0::2] -= lags[:, [0]]
        toeplitz = lags[:, :n].copy()
        toeplitz[:, 2::2] += lags[:, [0]]
        toeplitz[:, 0] += lags[:, 0]
        system = toeplitz[:, np.abs(rows - cols)] + hankel[:, rows + cols]
        mc[going] += np.linalg.solve(system, rhs[..., None])[..., 0]
    return mc


#: Frames analysed at a time: 2048 frames of 1024 samples and their spectra
#: are about 100 MB, where ten minutes at 16 kHz at once would be 1.8 GB.
_BLOCK = 2048


def _frames(y: np.ndarray, n_fft: int, hop: int, first: int = 0,
            count: Optional[int] = None) -> np.ndarray:
    """Frames *first* to *first + count* of *y*, windowed."""
    total = (len(y) - n_fft) // hop + 1
    count = total - first if count is None else min(count, total - first)
    # np.hamming is SPTK's hamming before its scaling to unit energy
    window = np.hamming(n_fft)
    window = window / math.sqrt(float(np.sum(window * window)))
    starts = hop * np.arange(first, first + count)
    return y[starts[:, None] + np.arange(n_fft)] * window


def _analysed(y: np.ndarray, n_fft: int, hop: int, order: int, alpha: float) -> np.ndarray:
    """The mel-cepstrum of every frame of *y*, :data:`_BLOCK` frames at a time."""
    total = (len(y) - n_fft) // hop + 1
    return np.concatenate([_mel_cepstra(_frames(y, n_fft, hop, start, _BLOCK), order, alpha)
                           for start in range(0, total, _BLOCK)])


# --- FastDTW, as the pure-Python fastdtw 0.3.4 computes it ------------------------

def _dtw(x: np.ndarray, y: np.ndarray, window: Optional[List[Tuple[int, int]]]
         ) -> List[Tuple[int, int]]:
    """The warping path of x against y over *window* (all cells if None).
    Of equal accumulated costs, a step from (i-1, j) wins over one from
    (i, j-1), and that over one from (i-1, j-1), as fastdtw's ``min`` picks."""
    if window is None:
        window = [(i, j) for i in range(len(x)) for j in range(len(y))]
    cells = np.array(window, dtype=np.int64).reshape(-1, 2)
    steps = np.linalg.norm(x[cells[:, 0]] - y[cells[:, 1]], axis=1).tolist()
    inf = math.inf
    cost = {(0, 0): 0.0}
    back = {}
    for (i, j), step in zip(window, steps):
        i, j = i + 1, j + 1
        best, came = cost.get((i - 1, j), inf), (i - 1, j)
        left = cost.get((i, j - 1), inf)
        if left < best:
            best, came = left, (i, j - 1)
        corner = cost.get((i - 1, j - 1), inf)
        if corner < best:
            best, came = corner, (i - 1, j - 1)
        cost[i, j] = best + step
        back[i, j] = came
    path = []
    i, j = len(x), len(y)
    while not (i == j == 0):
        path.append((i - 1, j - 1))
        i, j = back[i, j]
    path.reverse()
    return path


def _expand(path: List[Tuple[int, int]], len_x: int, len_y: int,
            radius: int) -> List[Tuple[int, int]]:
    near = set(path)
    for i, j in path:
        for a in range(-radius, radius + 1):
            for b in range(-radius, radius + 1):
                near.add((i + a, j + b))
    cells = set()
    for i, j in near:
        cells.update(((2 * i, 2 * j), (2 * i, 2 * j + 1),
                      (2 * i + 1, 2 * j), (2 * i + 1, 2 * j + 1)))
    window = []
    start = 0
    for i in range(len_x):
        first = None
        for j in range(start, len_y):
            if (i, j) in cells:
                window.append((i, j))
                if first is None:
                    first = j
            elif first is not None:
                break
        start = first
    return window


def _halved(x: np.ndarray) -> np.ndarray:
    even = len(x) - len(x) % 2
    return (x[0:even:2] + x[1:even:2]) / 2


def _fastdtw(x: np.ndarray, y: np.ndarray, radius: int = 1) -> List[Tuple[int, int]]:
    if len(x) < radius + 2 or len(y) < radius + 2:
        return _dtw(x, y, None)
    path = _fastdtw(_halved(x), _halved(y), radius)
    return _dtw(x, y, _expand(path, len(x), len(y), radius))


# --- MCD -------------------------------------------------------------------------

def mcd(ref: np.ndarray, deg: np.ndarray, sr: int, *, match_gain: bool = True,
        order: Optional[int] = None, alpha: Optional[float] = None,
        n_fft: int = 1024, hop_length: int = 256) -> float:
    """Mel-cepstral distortion of *deg* from *ref*, in dB: 0 for the same
    mel-cepstra, and larger the further apart; lower is better.

    >>> kudio.mcd(clean, enhanced, 16000)                    # URGENT 2025
    >>> kudio.mcd(natural, synthesized, 22050, match_gain=False)

    The URGENT 2025 challenge's MCD, every setting as its scoring script has
    it; the module notes give the steps.

    :param match_gain: first scale *deg* by its least-squares gain against
        *ref*, as URGENT does, which needs the pair lined up and of one
        length (a small difference warns and is cut, as for
        :func:`kudio.snr`). ``False`` compares the levels as they are and
        lets the two differ in length, as speech synthesis needs.
    :param order: the order M of the mel-cepstrum; with *alpha*, for a rate
        :data:`MCD_SETTINGS` has no row for. Default by the rate.
    :param alpha: the all-pass constant, between -1 and 1. Default by the
        rate.
    :param n_fft: frame length in samples, 1024 by default.
    :param hop_length: samples between frames, 256 by default.
    :raises ValueError: for input that is not one channel of samples, a rate
        without a row in :data:`MCD_SETTINGS` when *order* and *alpha* are
        not both given, settings out of range, and with *match_gain*, a
        length mismatch past :data:`kudio.core.evaluator.MAX_LENGTH_MISMATCH`.
    :returns: the MCD in dB. nan, with a warning, for a signal shorter than
        one frame.
    """
    sr = sample_rate(sr, 'mcd')
    if (order is None) != (alpha is None):
        raise ValueError("mcd(): pass order and alpha together, or neither for "
                         "the rate's own")
    if order is None:
        if sr not in MCD_SETTINGS:
            raise ValueError(f"mcd(): URGENT's MCD has settings for "
                             f"{', '.join(str(r) for r in MCD_SETTINGS)} Hz, not "
                             f"{sr} Hz; resample to one of them, or pass order "
                             f"and alpha")
        order, alpha = MCD_SETTINGS[sr]
    if isinstance(order, bool) or not isinstance(order, numbers.Integral) or order < 1:
        raise ValueError(f"mcd(): order has to be a whole number from 1, got {order!r}")
    if isinstance(alpha, bool) or not isinstance(alpha, numbers.Real) or not -1 < alpha < 1:
        raise ValueError(f"mcd(): alpha has to be a number between -1 and 1, got {alpha!r}")
    for value, name, low in ((n_fft, 'n_fft', 4), (hop_length, 'hop_length', 1)):
        if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < low:
            raise ValueError(f"mcd(): {name} has to be a whole number of samples "
                             f"from {low}, got {value!r}")
    if n_fft < 2 * order + 2:
        raise ValueError(f"mcd(): n_fft ({n_fft}) has to be at least twice the "
                         f"order plus 2 ({2 * order + 2})")
    if not isinstance(match_gain, (bool, np.bool_)):
        raise ValueError(f"mcd(): match_gain has to be True or False, got {match_gain!r}")
    if match_gain:
        ref, deg = _evaluator._align(ref, deg, caller='mcd')
    else:
        ref, deg = signal(ref, 'ref', 'mcd'), signal(deg, 'deg', 'mcd')
        faint('mcd', [('ref', ref), ('deg', deg)])
    for label, y in (('ref', ref), ('deg', deg)):
        if len(y) < n_fft:
            return scored_nan('mcd', f"{label} needs at least one frame of {n_fft} "
                                     f"samples and has {len(y)}")
    if match_gain:
        with np.errstate(over='ignore', invalid='ignore'):
            gain = float(np.sum(ref * deg)) / (float(np.sum(deg * deg)) + 1e-8)
        if not math.isfinite(gain):
            return _evaluator._too_loud('mcd', ref, deg)
        deg = deg * gain
    order, alpha = int(order), float(alpha)
    n_fft, hop_length = int(n_fft), int(hop_length)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        first = _analysed(ref, n_fft, hop_length, order, alpha)
        second = _analysed(deg, n_fft, hop_length, order, alpha)
    if not (np.all(np.isfinite(first)) and np.all(np.isfinite(second))):
        return _evaluator._too_loud('mcd', ref, deg)
    path = np.array(_fastdtw(first, second, radius=1))
    diff = first[path[:, 0]] - second[path[:, 1]]
    return float(np.mean(10.0 / np.log(10.0) * np.sqrt(2.0 * np.sum(diff * diff, axis=1))))
