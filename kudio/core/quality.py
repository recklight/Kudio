# -*- coding: utf-8 -*-
"""Objective quality measures for enhanced speech, after Hu and Loizou: LLR,
Itakura-Saito, cepstral distance, WSS, frequency-weighted segmental SNR, and
the composite CSIG / CBAK / COVL built from three of them and PESQ.

Each one compares the processed signal with the clean one, 30 ms frame by
30 ms frame, and needs nothing beyond numpy. Hu and Loizou set them against
listeners on NOIZEUS: 16 sentences under four noises at 5 and 10 dB SNR,
unprocessed and through 13 enhancement algorithms, 1792 files in all, each
rated for signal distortion, background intrusiveness and overall quality to
ITU-T P.835. The composite measures are linear fits to those ratings. The
sources::

    [HL06]  Y. Hu and P. C. Loizou, "Evaluation of objective measures for
            speech enhancement", Proc. Interspeech 2006,
            doi:10.21437/Interspeech.2006-84.
    [HL08]  Y. Hu and P. C. Loizou, "Evaluation of objective quality measures
            for speech enhancement", IEEE Trans. Audio, Speech, Lang. Process.
            16(1), pp. 229-238, 2008.
    [HP98]  J. Hansen and B. Pellom, "An effective quality evaluation protocol
            for speech enhancement algorithms", Proc. ICSLP 1998, vol. 7,
            pp. 2819-2822; ref. [20] of [HL08].
    [L07]   P. C. Loizou, "Speech Enhancement: Theory and Practice", CRC Press,
            2007; ref. [26] of [HL08].
    [code]  the MATLAB files Loizou published: the 2006 composite.m, alone in
            composite.zip on his page, and the comp_*.m files with the 2012
            composite.m in the code that comes with the 2nd edition of [L07]
            (CRC Press, 2013, doi:10.1201/b14529).

**Where the paper and the code disagree, I follow the code**, except for the
lower bound of IS. The composite's coefficients were published in
composite.m, which computes their inputs this way, and the CSIG, CBAK and
COVL in the literature come from it or from ports of it. The differences:

* Window. [HL08] Sec. III-C says Hamming; the code uses
  ``0.5 * (1 - cos(2 pi n / (N + 1)))`` for n = 1..N, a Hann window without
  its zero end points.
* Itakura-Saito, the log term. [HL08] eq. (5) prints it as
  ``log(sigma_c^2 / sigma_p^2)``; the code has the reciprocal. Only the code's
  form is the Itakura-Saito divergence between the two LPC spectra. The
  printed one goes negative for an output louder than the clean frame: at
  twice its power and the same envelope, ``0.5 + ln 0.5 - 1 = -1.19``.
* Itakura-Saito, the range. [HL08] Sec. III-B holds each frame to [0, 100],
  and so does the header of comp_is.m, but its code only caps at 100. They
  differ only where both frames are digital silence, as in zero padding:
  there comp_is.m's eps floors make the frame -1, and an exact copy of 2 s of
  speech with 0.5 s of zeros either side would score -0.33. I keep the
  paper's range here. A copy that scores below zero is measuring the floors,
  not the signal, and IS does not go into CSIG, CBAK or COVL, so the choice
  moves none of them.
* Segmental SNR. [HL08] Sec. III-C leaves frames outside -10 to 35 dB out of
  the average. composite.m's snr() holds each frame to that range instead, as
  comp_fwseg.m does for fwSegSNR, and so do I.
* Frames. The code takes ``floor((L - N) / hop)`` frames of N samples out of
  L, one fewer than fit, so the last frame of a file is never scored.
* The best 95%. The code keeps ``round(0.95 * M)`` of M frames, rounded the
  MATLAB way, halves away from zero.

What to watch for:

* The measures are defined at 8 kHz, the rate of NOIZEUS, and at 16 kHz, where
  the code moves from 10th- to 16th-order LPC. Other rates are refused;
  resample to 16 kHz first (:func:`kudio.resample`, or
  :func:`scipy.signal.resample_poly`).
* Frame m of one signal is compared with frame m of the other, so the pair has
  to line up, as for :func:`kudio.snr`. :func:`kudio.align` comes first.
* LLR, IS, CEP and WSS average only the best 95% of frames, so damage
  confined to less than 5% of a file can leave them untouched. fwSegSNR and
  the composite's segmental SNR average every frame.
* The 25 bands of WSS and fwSegSNR are fixed in hertz and stop at 3.8 kHz, at
  16 kHz as at 8 kHz. WSS reads nothing above that, so damage above 4 kHz in a
  wideband file never reaches it. fwSegSNR does feel it: before the bands,
  both magnitude spectra are scaled to unit area over every bin up to Nyquist,
  as comp_fwseg.m scales them, so whatever is added or removed above 4 kHz
  changes the share left to every band. At 16 kHz, noise kept above 5 kHz at
  20 dB SNR takes a copy from 35 dB to about 9, and cutting everything above
  4 kHz takes it to about 18; WSS stays near 0 in both.
* Both signals get float64 eps (2.2e-16) added before framing, as in the code.
  It changes no audible sample and is what lets a frame of digital silence
  still give a number. Where the reference and the output are both digital
  silence, the two frames are the same and score as a perfect match, as any
  pause copied exactly does. Where only the reference is, the frame compares
  the output with eps, a constant, rather than with speech. Noise or speech
  there scores the worst LLR, CEP and fwSegSNR give, 2, 10 dB and -10 dB a
  frame; IS scores ``ln(s_p / eps) - 1`` and WSS the output's own slopes. A
  DC offset there is a constant too, and the three measures blind to level
  score it as a match: LLR under 0.005, CEP under 0.3 dB and fwSegSNR 35 dB
  a frame, for offsets of 1e-3 and 0.1. A hum at 50 to 100 Hz peaking at
  -40 dBFS lands anywhere between: LLR 0.075 to 2, CEP 1.49 to 10 dB and
  fwSegSNR -0.69 to 12 dB a frame. On 2 s of speech with 0.5 s of zeros
  either side, 125 of its 396 frames, a copy holding white noise at -60 dBFS
  where the reference is zero scores 0.56 on LLR, 7.2 on IS, 2.84 dB on CEP,
  5.73 on WSS and 20.4 dB on fwSegSNR, against 0, 0, 0, 0 and 35 for the
  copy itself (8 kHz; 16 kHz is within 3%). Every measure warns with how
  many such frames there were and what they scored, and :func:`composite`
  has a floor of its own. Trim the silence off both signals first. A
  reference that is silent throughout scores nan, with a warning, and so
  does a pair too short for one frame (37.5 ms).
* WSS depends on the level of a quiet pair. comp_wss.m floors each band's
  energy at -100 dB of full scale, so from a clean peak of about -50 dBFS
  down the quieter bands sit on the floor: on the test pair of kudio's tests,
  27.95 at -66 dBFS where it is 35.98 at any level from -6 to -46 (8 kHz).
  :func:`wss` and :func:`composite` warn when it happens; bring both signals
  to a normal level with one gain first.
* A DC offset on either signal counts as distortion, and WSS feels it most.
  On 2 s of synthetic speech, an offset of 1% of the peak added to the output
  scores 30.8 at 8 kHz and 35.0 at 16 kHz, where white noise at 10 dB SNR
  scores 36.0 and 30.9. Neither the code nor this module takes it out, so
  remove DC from both signals first if the processing can add one.
* The composite needs a PESQ score, and which PESQ matters: see
  :func:`composite`.
"""
from __future__ import annotations

import functools
import math
import numbers
from typing import NamedTuple, Optional, Tuple

import numpy as np

from kudio.core._framing import frame_view
from kudio.core._policy import ONE_STEP, sample_rate, scored_nan, warn
from kudio.core.evaluator import _align, _cut, _pesq_mode, _pesq_scored, _sides
from kudio.exceptions import DependencyError

__all__ = ['llr', 'itakura_saito', 'cepstral_distance', 'wss',
           'fw_segmental_snr', 'composite', 'CompositeScore']

#: Frame length in seconds. [HL08] Sec. III-C: 30 ms frames overlapping by
#: 75%; the code rounds the length to whole samples and takes the hop as a
#: quarter of it, rounded down: 240 and 60 samples at 8 kHz, 480 and 120 at
#: 16 kHz.
_FRAME_SECONDS = 0.030

#: LPC order by rate. 10 at 8 kHz is [HL08] Sec. III-C ("a tenth order LPC
#: analysis"); 16 at 16 kHz is the code's (comp_llr.m, comp_is.m, comp_cep.m
#: and composite.m all switch to 16 at and above 10 kHz). These two rates are
#: the only ones the measures are defined at.
_LPC_ORDER = {8000: 10, 16000: 16}

#: Share of frames kept, the ones with the least distortion. [HL08] Sec. III-B
#: states it for LLR, after [HP98]; the code applies it to IS, CEP and WSS as
#: well, and not to fwSegSNR or segmental SNR.
_KEEP = 0.95

#: Per-frame caps. LLR [0, 2], IS [0, 100] and CEP [0, 10] are [HL08]
#: Sec. III-B; comp_is.m's code caps IS at 100 only (see the module notes),
#: and the composite takes its LLR uncapped, as composite.m does. The -10 to
#: 35 dB of segmental SNR is the range of [HL08] Sec. III-C, where frames
#: outside it are left out; holding each frame to it is what composite.m's
#: snr() does, and comp_fwseg.m does the same to each fwSegSNR frame.
_LLR_RANGE = (0.0, 2.0)
_IS_RANGE = (0.0, 100.0)
_CEP_RANGE = (0.0, 10.0)
_SNR_RANGE = (-10.0, 35.0)

_EPS = float(np.finfo(np.float64).eps)

#: Frames taken at a time. Windowed frames hold every sample four times over
#: and their spectra more again: ten minutes at 16 kHz is 79,996 frames, 614 MB
#: of windowed frames for the pair, and fwSegSNR's spectra would take another
#: 1.3 GB if all of them were computed at once.
_BLOCK = 2048

#: The 25 bands of WSS and fwSegSNR, centre and bandwidth in Hz. [HL08]
#: Sec. III-C and eq. (10) set K = 25 bands "spaced in proportion to the ear's
#: critical bands" and take the Gaussian-shaped windows from [L07], whose 2nd
#: edition describes them on p. 511. The numbers themselves I have found only
#: in the code, the same table in comp_wss.m, comp_fwseg.m and composite.m,
#: which credits the filter routines to Pellom and Hansen, the authors of
#: [HP98]. The first seven are 70 Hz wide, and each centre lies one bandwidth
#: above the one before.
_BAND_CENTRES = (
    50.0000, 120.000, 190.000, 260.000, 330.000, 400.000, 470.000, 540.000,
    617.372, 703.378, 798.717, 904.128, 1020.38, 1148.30, 1288.72, 1442.54,
    1610.70, 1794.16, 1993.93, 2211.08, 2446.71, 2701.97, 2978.04, 3276.17,
    3597.63)
_BAND_WIDTHS = (
    70.0000, 70.0000, 70.0000, 70.0000, 70.0000, 70.0000, 70.0000, 77.3724,
    86.0056, 95.3398, 105.411, 116.256, 127.914, 140.423, 153.823, 168.154,
    183.457, 199.776, 217.153, 235.631, 255.255, 276.072, 298.126, 321.465,
    346.136)

#: Where each band's Gaussian is cut to zero: ``exp(-30 / (2 * 2.303))``,
#: compared with the shape after its height is scaled down for bandwidth.
#: comp_wss.m calls it the "-30 dB point of filter"; as a power ratio it is
#: -28.3 dB. Like the table, I know it only from the code.
_BAND_FLOOR = math.exp(-30.0 / (2.0 * 2.303))

#: Klatt's weighting constants for WSS: K_max for the distance from the
#: frame's largest band, K_locmax for the distance from the nearest peak.
#: Klatt, "Prediction of perceived phonetic distance from critical-band
#: spectra: a first step", Proc. ICASSP 1982, vol. 7, pp. 1278-1281, gives
#: K_max = 20 and K_locmax = 1 on p. 1280 for spectra in units of dB times
#: 10, that is 2 dB and 0.1 dB. comp_wss.m applies the same two numbers to
#: levels in dB, and I keep them that way, since the composite's
#: coefficients were fitted on that code: these weights halve 20 dB below
#: the largest band and 1 dB below the nearest peak, where Klatt's halve at
#: 2 dB and 0.1 dB.
_K_MAX = 20.0
_K_LOCMAX = 1.0

#: Band energies below 1e-10 (-100 dB) are read as -100 dB in WSS (comp_wss.m).
_WSS_FLOOR = 1e-10

#: The exponent of fwSegSNR's weights, ``W = |X|^gamma``: [HL08] eq. (9), and
#: 0.2 is where they found the best correlation over 0.1 to 2.
_GAMMA = 0.2

#: The composite measures, [HL06] Sec. 4.1: intercept, then the coefficients
#: of PESQ, LLR, WSS and segmental SNR.
_CSIG = (3.093, 0.603, -1.029, -0.009, 0.0)
_CBAK = (1.634, 0.478, 0.0, -0.007, 0.063)
_COVL = (1.594, 0.805, -0.512, -0.007, 0.0)

#: The composite is a rating on the P.835 scale, held to it as the composite.m
#: that ships with the 2012 pesq.m does.
_MOS_RANGE = (1.0, 5.0)

#: What each PESQ scale can produce. The raw P.862 score runs -0.5 to 4.5.
#: MOS-LQO maps it through ``0.999 + 4 / (1 + exp(-a x + b))``, with a = 1.4945
#: and b = 4.6607 in P.862.1 and a = 1.3669 and b = 3.8224 in P.862.2, which
#: take -0.5..4.5 to 1.017..4.549 and to 1.043..4.644. The bounds here are
#: rounded outwards.
_PESQ_RANGE = {'raw': (-0.5, 4.5), 'lqo': (1.0, 4.65)}


class CompositeScore(NamedTuple):
    """CSIG, CBAK and COVL, and the four measures they were computed from.

    The ratings are on the 1-5 scale of ITU-T P.835, held to it. Keep the
    components with them: a CSIG that moved is explained by its PESQ, LLR and
    WSS, and they say which.

    ``wss`` is the number :func:`wss` returns. ``pesq`` is the PESQ the
    ratings were computed from: by default wideband MOS-LQO at 16 kHz, the
    number :func:`kudio.pesq` gives there too, and raw narrowband P.862 at
    8 kHz, where :func:`kudio.pesq` gives MOS-LQO unless asked for
    ``scale='raw'``. ``llr`` keeps the frames above 2 that :func:`llr` caps,
    and ``segsnr`` comes from the composite's windowed 30 ms frames, not the
    512-sample rectangular ones of :func:`kudio.segmental_snr`: 6.02 dB where
    segmental_snr gives 8.08 on 2 s of synthetic speech under white noise at
    20 dB SNR (8 kHz). A table that also holds those metrics needs other names
    for these fields.
    """

    #: predicted rating of signal distortion
    csig: float
    #: predicted rating of background intrusiveness
    cbak: float
    #: predicted rating of overall quality
    covl: float
    #: the PESQ score used, of the variant :func:`composite` was asked for
    pesq: float
    #: LLR over the best 95% of frames, each frame uncapped
    llr: float
    #: WSS, the same number :func:`wss` returns
    wss: float
    #: segmental SNR in dB, the composite's own (see :func:`composite`)
    segsnr: float


# --- shared steps -------------------------------------------------------------

def _lpc_order(sr, name: str) -> int:
    """The LPC order at *sr*, refusing the rates the measures are not defined
    at."""
    sr = sample_rate(sr, name)
    if sr not in _LPC_ORDER:
        raise ValueError(f"{name}(): defined at 8 and 16 kHz only, not {sr} Hz; "
                         f"resample to 16 kHz first, with kudio.resample or, "
                         f"without librosa, scipy.signal.resample_poly(y, 16000, {sr})")
    return _LPC_ORDER[sr]


def _frame_size(sr: int) -> Tuple[int, int]:
    """``(samples per frame, hop)`` at *sr*."""
    size = int(round(_FRAME_SECONDS * sr))
    return size, size // 4


@functools.lru_cache(maxsize=4)
def _window(size: int) -> np.ndarray:
    """``0.5 * (1 - cos(2 pi n / (size + 1)))`` for n = 1..size: the code's
    window, a Hann window whose zero end points fall just outside the frame."""
    n = np.arange(1, size + 1)
    window = 0.5 * (1.0 - np.cos(2.0 * np.pi * n / (size + 1)))
    window.flags.writeable = False
    return window


class _Frames(NamedTuple):
    """The pair as ``(frames, samples)`` views, not yet windowed, and which
    frames are digital silence in each."""

    clean: np.ndarray
    processed: np.ndarray
    window: np.ndarray
    #: frames in which the reference is exact zeros throughout
    silent: np.ndarray
    #: the same for the processed signal
    silent_deg: np.ndarray


def _silence(y: np.ndarray, size: int, hop: int, count: int) -> np.ndarray:
    """Which of the first *count* frames of *y* hold nothing but exact zeros,
    read off a framed view of ``y != 0``, one byte a sample. A running count
    of the nonzero samples would take 16 bytes a sample, and a copy of the
    frames would hold every sample four times over."""
    return ~frame_view(y != 0.0, size, hop)[:count].any(axis=1)


def _framed(ref: np.ndarray, deg: np.ndarray, sr: int, name: str) -> Optional[_Frames]:
    """The pair in frames, or None, after a warning, when it cannot be
    scored."""
    size, hop = _frame_size(sr)
    count = (len(ref) - size) // hop
    if count < 1:
        need = size + hop
        scored_nan(name, f"the pair needs at least {need} samples "
                         f"({1000.0 * need / sr:.1f} ms at {sr} Hz) to score one "
                         f"frame, and has {len(ref)}")
        return None
    if not np.any(ref):
        scored_nan(name, "the reference is digital silence throughout, so there "
                         "is no speech to compare with")
        return None
    # The code adds eps to both signals before anything else. A frame of
    # digital silence then has an autocorrelation and a spectrum, where it
    # would otherwise divide by zero; on anything audible it changes nothing.
    clean = frame_view(ref + _EPS, size, hop)[:count]
    processed = frame_view(deg + _EPS, size, hop)[:count]
    return _Frames(clean, processed, _window(size), _silence(ref, size, hop, count),
                   _silence(deg, size, hop, count))


def _each(measure, framed: _Frames, *args) -> np.ndarray:
    """*measure* on every pair of windowed frames, :data:`_BLOCK` at a time.

    numpy's floating-point warnings are kept quiet here: a frame too loud to
    square comes out NaN, and :func:`_pooled` says so once, where numpy
    would say it once for every operation that met it.
    """
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        return np.concatenate([measure(framed.clean[i:i + _BLOCK] * framed.window,
                                       framed.processed[i:i + _BLOCK] * framed.window,
                                       *args)
                               for i in range(0, len(framed.clean), _BLOCK)])


def _best(values: np.ndarray) -> float:
    """Mean of the 95% of frames with the least distortion.

    ``round(0.95 * M)`` frames are kept with MATLAB's rounding, halves away
    from zero: at M = 30 that keeps 29, where Python's round would keep 28.
    """
    ordered = np.sort(values)
    keep = max(1, int(math.floor(len(ordered) * _KEEP + 0.5)))
    return float(np.mean(ordered[:keep]))


#: why a frame comes out NaN, in the words of the warning that says so
_TOO_LOUD = "as a frame does when it is too loud to square in float64"


def _spread(values: np.ndarray, unit: str = '') -> str:
    """``'2 each'`` or ``'25.4 to 26'``: what a set of frames scored."""
    low, high = f"{np.min(values):.3g}", f"{np.max(values):.3g}"
    return f"{low}{unit} each" if low == high else f"{low} to {high}{unit}"


def _pooled(values: np.ndarray, name: str, framed: _Frames, best: bool,
            unit: str = '') -> float:
    """Frame values to one number, or nan with a warning when a frame could
    not be measured. Sorting would otherwise put such frames last, where the
    95% rule drops them without a word.

    A frame where the reference is digital silence and deg is not compares
    deg with the eps added to both signals, and a warning after the score
    says how many such frames there were and what they scored. Where deg is
    silent too, the two frames are the same and score as a perfect match, as
    any pause copied exactly does, so those are not counted.
    """
    lost = int(np.count_nonzero(np.isnan(values)))
    if lost:
        return scored_nan(name, f"{lost} of the {len(values)} frames came out "
                                f"NaN, {_TOO_LOUD}")
    score = _best(values) if best else float(np.mean(values))
    against_eps = framed.silent & ~framed.silent_deg
    count = int(np.count_nonzero(against_eps))
    if count:
        warn(f"{name}(): {count} of the reference's {len(values)} frames are "
             f"digital silence where deg is not, so those frames compare deg "
             f"with eps, the floor added to both signals, rather than with "
             f"speech; they score {_spread(values[against_eps], unit)}. Trim "
             f"the silence off both to compare the rest alone.")
    return score


# --- LPC ----------------------------------------------------------------------

def _autocorrelation(frames: np.ndarray, order: int) -> np.ndarray:
    """Lags 0..*order* of every frame, ``(frames, order + 1)``."""
    size = frames.shape[1]
    return np.stack([np.einsum('ij,ij->i', frames[:, :size - k], frames[:, k:])
                     for k in range(order + 1)], axis=1)


def _levinson(r: np.ndarray) -> np.ndarray:
    """Prediction-error filters ``[1, -a_1, ..., -a_p]`` from autocorrelations,
    one row per frame.

    Durbin's recursion for the autocorrelation method (J. Makhoul, "Linear
    prediction: a tutorial review", Proc. IEEE 63(4), 1975), run on every
    frame at once: the loop is over the order.
    """
    frames, order = r.shape[0], r.shape[1] - 1
    a = np.zeros((frames, order))
    error = r[:, 0].copy()
    with np.errstate(divide='ignore', invalid='ignore'):
        for i in range(order):
            k = (r[:, i + 1] - np.einsum('ij,ij->i', a[:, :i], r[:, i:0:-1])) / error
            previous = a[:, :i].copy()
            a[:, i] = k
            a[:, :i] = previous - k[:, None] * previous[:, ::-1]
            error = (1.0 - k * k) * error
    return np.concatenate([np.ones((frames, 1)), -a], axis=1)


def _quadratic(a: np.ndarray, r: np.ndarray) -> np.ndarray:
    """``a R a^T`` per frame, R the Toeplitz matrix of the lags in *r*: the
    prediction error that filter *a* leaves on the signal *r* came from."""
    lag = np.abs(np.subtract.outer(np.arange(r.shape[1]), np.arange(r.shape[1])))
    return np.einsum('mi,mij,mj->m', a, r[:, lag], a)


def _lpc_pair(clean: np.ndarray, processed: np.ndarray, order: int):
    r_c = _autocorrelation(clean, order)
    r_p = _autocorrelation(processed, order)
    return r_c, _levinson(r_c), r_p, _levinson(r_p)


def _frame_llr(clean: np.ndarray, processed: np.ndarray, order: int) -> np.ndarray:
    r_c, a_c, _, a_p = _lpc_pair(clean, processed, order)
    # Both forms are prediction errors and cannot be negative, but a voiced
    # frame with no noise in it is predicted so well that its own error is
    # round-off and can land below zero (-7.1e-10 against a frame energy of
    # 0.0277), and the log of that is NaN. Both are floored at eps times the
    # frame's energy, under anything but round-off: such a frame then scores
    # the cap and an exact copy of it 0, while a frame of digital silence,
    # whose own form is around 1e-39 against an energy around 1e-29, keeps
    # its score. np.maximum keeps the NaN of a frame too loud to square.
    floor = _EPS * r_c[:, 0]
    numerator = np.maximum(_quadratic(a_p, r_c), floor)
    denominator = np.maximum(_quadratic(a_c, r_c), floor)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.log(numerator / denominator)


def _frame_is(clean: np.ndarray, processed: np.ndarray, order: int) -> np.ndarray:
    r_c, a_c, r_p, a_p = _lpc_pair(clean, processed, order)
    # sigma^2 of each frame, its prediction error r . a, floored at eps as
    # comp_is.m floors it, and so is the clean frame's own quadratic form
    gain_c = np.maximum(np.einsum('ij,ij->i', r_c, a_c), _EPS)
    gain_p = np.maximum(np.einsum('ij,ij->i', r_p, a_p), _EPS)
    ratio = _quadratic(a_p, r_c) / np.maximum(_quadratic(a_c, r_c), _EPS)
    with np.errstate(divide='ignore', invalid='ignore'):
        return gain_c / gain_p * ratio + np.log(gain_p / gain_c) - 1.0


def _cepstrum(a: np.ndarray) -> np.ndarray:
    """Coefficients 1..p of the cepstrum of ``1 / A(z)``, by the recursion of
    [HL08] eq. (6) on the predictor coefficients ``-a[:, 1:]``."""
    alpha = -a[:, 1:]
    order = alpha.shape[1]
    c = np.zeros_like(alpha)
    for m in range(1, order + 1):
        k = np.arange(1, m)
        c[:, m - 1] = alpha[:, m - 1] + (c[:, k - 1] * alpha[:, m - k - 1] * k
                                         ).sum(axis=1) / m
    return c


def _frame_cep(clean: np.ndarray, processed: np.ndarray, order: int) -> np.ndarray:
    _, a_c, _, a_p = _lpc_pair(clean, processed, order)
    distance = np.sum((_cepstrum(a_c) - _cepstrum(a_p)) ** 2, axis=1)
    # [HL08] eq. (7): a log-spectral distance in dB
    return 10.0 / math.log(10.0) * np.sqrt(2.0 * distance)


# --- the band measures --------------------------------------------------------

@functools.lru_cache(maxsize=4)
def _bands(sr: int) -> Tuple[int, np.ndarray]:
    """``(FFT length, filters)`` for WSS and fwSegSNR at *sr*.

    The FFT is the next power of two at or above twice the frame, 512 points at
    8 kHz and 1024 at 16 kHz, so a bin is 15.625 Hz at both rates. Each band
    is a Gaussian over bins 0 .. nfft/2 - 1, ``exp(-11 ((j - f0) / bw)^2)``,
    centred on the bin below its centre frequency and scaled to a height of
    ``70 / bandwidth`` so the bands carry about equal area. [HL08] Sec. III-C
    describes them as "overlapping Gaussian-shaped windows" and refers to
    [L07] for them (the 2nd edition describes them on p. 511); the -11, the
    scaling and the cut-off are those of comp_wss.m, and like the table I
    know them only from the code.
    """
    size, _ = _frame_size(sr)
    nfft = 1 << int(math.ceil(math.log2(2 * size)))
    half = nfft // 2
    centres = np.floor(np.asarray(_BAND_CENTRES) / (sr / 2.0) * half)
    widths = np.asarray(_BAND_WIDTHS) / (sr / 2.0) * half
    height = math.log(_BAND_WIDTHS[0]) - np.log(_BAND_WIDTHS)
    j = np.arange(half)
    shape = np.exp(-11.0 * ((j - centres[:, None]) / widths[:, None]) ** 2
                   + height[:, None])
    shape[shape <= _BAND_FLOOR] = 0.0
    shape.flags.writeable = False
    return nfft, shape


def _spectrum(frames: np.ndarray, nfft: int) -> np.ndarray:
    """Magnitude spectrum of each frame over bins 0 .. nfft/2 - 1."""
    return np.abs(np.fft.rfft(frames, nfft, axis=1))[:, :nfft // 2]


def _nearest_peaks(level: np.ndarray, slope: np.ndarray) -> np.ndarray:
    """For each band below the top one, the level of the spectral peak Klatt's
    second weight is measured from.

    Downhill or flat (slope <= 0), that is the peak behind the band: the
    nearest band, at or left of it, where the spectrum stopped rising. Uphill,
    the search runs right to where the rise ends, and comp_wss.m then takes
    the band *before* that point, one short of the peak. I keep its rule:
    composite.m, which carries the coefficients, computes WSS this way, and a
    port that took the peak itself would not reproduce the published figures.
    """
    count = slope.shape[1]
    index = np.arange(count)
    # first band at or after i where the rise stops; count if it never does
    stops = np.where(slope <= 0, index, count)
    stop = np.minimum.accumulate(stops[:, ::-1], axis=1)[:, ::-1]
    # last band at or before i that still rises; -1 if none does
    rises = np.where(slope > 0, index, -1)
    rise = np.maximum.accumulate(rises, axis=1)
    where = np.where(slope > 0, stop - 1, rise + 1)
    return np.take_along_axis(level, where, axis=1)


def _klatt_weights(level: np.ndarray, slope: np.ndarray) -> np.ndarray:
    """Klatt's weight on each slope: 1 at the frame's largest band, lower at
    lesser peaks, and lower again in the valleys between them."""
    below = level[:, :-1]
    top = level.max(axis=1, keepdims=True)
    peak = _nearest_peaks(level, slope)
    return (_K_MAX / (_K_MAX + top - below)
            * _K_LOCMAX / (_K_LOCMAX + peak - below))


def _frame_wss(clean: np.ndarray, processed: np.ndarray, sr: int,
               floored: Optional[list] = None) -> np.ndarray:
    """WSS of each pair of frames; with *floored*, also append to it how
    many of each clean frame's 25 band energies are under the floor."""
    nfft, bands = _bands(sr)
    levels = []
    for frames in (clean, processed):
        # a frame too loud to square comes out nan, and _pooled says so
        with np.errstate(over='ignore', invalid='ignore'):
            energy = (_spectrum(frames, nfft) ** 2) @ bands.T
        if floored is not None and frames is clean:
            floored.append(np.count_nonzero(energy < _WSS_FLOOR, axis=1))
        levels.append(10.0 * np.log10(np.maximum(energy, _WSS_FLOOR)))
    slope_c, slope_p = (np.diff(level, axis=1) for level in levels)
    weight = 0.5 * (_klatt_weights(levels[0], slope_c)
                    + _klatt_weights(levels[1], slope_p))
    return np.sum(weight * (slope_c - slope_p) ** 2, axis=1) / np.sum(weight, axis=1)


def _frame_fwsnr(clean: np.ndarray, processed: np.ndarray, sr: int) -> np.ndarray:
    nfft, bands = _bands(sr)
    shares = []
    for frames in (clean, processed):
        magnitude = _spectrum(frames, nfft)
        # [HL08] Sec. III-C: both spectra scaled to an area of one first,
        # which they found "critically important"
        shares.append((magnitude / magnitude.sum(axis=1, keepdims=True)) @ bands.T)
    x_c, x_p = shares
    error = np.maximum((x_c - x_p) ** 2, _EPS)
    weight = x_c ** _GAMMA
    snr = 10.0 * np.log10(x_c ** 2 / error)
    return np.clip(np.sum(weight * snr, axis=1) / np.sum(weight, axis=1), *_SNR_RANGE)


def _frame_segsnr(clean: np.ndarray, processed: np.ndarray) -> np.ndarray:
    signal = np.sum(clean ** 2, axis=1)
    noise = np.sum((clean - processed) ** 2, axis=1)
    return np.clip(10.0 * np.log10(signal / (noise + _EPS) + _EPS), *_SNR_RANGE)


# --- the measures -------------------------------------------------------------

def llr(ref: np.ndarray, deg: np.ndarray, sr: int) -> float:
    """Log-likelihood ratio of *deg* against *ref*: how far each processed
    frame's spectral envelope is from the clean one's. 0 is a perfect copy,
    lower is better.

    Per frame it is ``log(a_p R_c a_p^T / a_c R_c a_c^T)``, [HL08] eq. (4):
    ``a_c`` and ``a_p`` are the LPC error filters of the clean and processed
    frames and ``R_c`` the clean frame's autocorrelation matrix, so the ratio
    is how much more of the clean frame the processed frame's predictor leaves
    unexplained than the clean frame's own does, which is never less. LPC is
    the autocorrelation method, of order 10 at 8 kHz and 16 at 16 kHz, on
    30 ms frames with 75% overlap (see the module notes for the window). Each
    frame is held to [0, 2] and the mean is over the best 95% of frames,
    both from [HL08] Sec. III-B.

    Only the shape of the envelope counts: a gain or a flipped polarity costs
    nothing, while a filter that tilts the spectrum does.

    A frame where the reference is digital silence and *deg* is not compares
    *deg* there with eps, a constant under the window, which the clean
    frame's own predictor leaves almost nothing of. A predictor fitted to
    noise or speech leaves far more, and the frame scores 2, the cap. One
    fitted to a DC offset is the clean frame's own, since both are a constant
    under the window, and the frame scores 0; a hum at 50 to 100 Hz peaking
    at -40 dBFS scores 0.075 to 2. On 2 s of speech with 0.5 s of zeros
    either side, white noise at -60 dBFS in the zeros takes a copy from 0 to
    0.56, and a warning says how many such frames there were and what they
    scored. Where *deg* is digital silence too, the frame scores 0.

    :param ref: the clean signal, mono.
    :param deg: the processed signal, mono and lined up with *ref*.
    :param sr: 8000 or 16000.
    :raises ValueError: at any other rate, for a stereo array, for samples
        that are not finite, or for a pair too different in length (see
        :data:`kudio.core.evaluator.MAX_LENGTH_MISMATCH`).
    :returns: the mean LLR, 0 to 2; nan, with a warning, for a pair shorter
        than 37.5 ms, a reference that is digital silence throughout, or a
        pair so loud a frame cannot be squared in float64.
    """
    order = _lpc_order(sr, 'llr')
    ref, deg = _align(ref, deg, caller='llr')
    framed = _framed(ref, deg, int(sr), 'llr')
    if framed is None:
        return float('nan')
    values = np.clip(_each(_frame_llr, framed, order), *_LLR_RANGE)
    return _pooled(values, 'llr', framed, best=True)


def itakura_saito(ref: np.ndarray, deg: np.ndarray, sr: int) -> float:
    """Itakura-Saito distance of *deg* from *ref*, between the LPC spectra of
    each pair of frames. 0 is a perfect copy, lower is better.

    Per frame it is ``(s_c / s_p) (a_p R_c a_p^T / a_c R_c a_c^T) +
    log(s_p / s_c) - 1``, with LLR's terms and ``s_c``, ``s_p`` the
    prediction-error powers (the squared LPC gains) of the clean and processed
    frames. That is the Itakura-Saito divergence between the two frames' LPC
    spectra ``P_c = s_c / |A_c|^2`` and ``P_p = s_p / |A_p|^2``: the mean over
    frequency of ``P_c / P_p - log(P_c / P_p) - 1``. [HL08] eq. (5) prints the
    log term the other way up, ``log(s_c / s_p)``, which would let it go
    negative; I use the form in comp_is.m, which is the divergence.

    LPC order, frames and window are LLR's. Each frame is held to [0, 100], the
    range of [HL08] Sec. III-B and of comp_is.m's header (its code caps at 100
    only; see the module notes), and the mean is over the best 95% of frames.

    Unlike LLR it charges for level, and unevenly: a copy at twice the
    amplitude scores ``1/4 + ln 4 - 1 = 0.636`` and one at half the amplitude
    ``4 - ln 4 - 1 = 1.614``. The divergence grows in proportion to the clean
    power over the processed and only with the log of its inverse, so an
    output that comes back too quiet costs more than one too loud. Those are
    the scores of frames that hold signal. A frame of digital silence stays
    silence at any gain and scores 0, so 2 s of speech with 0.5 s of zeros
    either side, 125 of its 396 frames silent, scores 0.425 at twice the
    amplitude and 1.077 at half.

    The prediction-error powers, and the clean frame's own quadratic form, are
    floored at float64 eps as comp_is.m floors them. Where the reference is
    digital silence, that floor stands in for its power, and the frame scores
    ``ln(s_p / eps) - 1`` against whatever the output holds there: 25 to 36
    for white noise from -60 to -20 dBFS. Those numbers describe the floor,
    not the reference. On the padded speech above, white noise at -60 dBFS
    in the zeros takes a copy from 0 to 7.2, and a warning says how many such
    frames there were and what they scored.

    :param ref: the clean signal, mono.
    :param deg: the processed signal, mono and lined up with *ref*.
    :param sr: 8000 or 16000.
    :raises ValueError: as :func:`llr`.
    :returns: the mean distance, 0 to 100; nan, with a warning, as :func:`llr`.
    """
    order = _lpc_order(sr, 'itakura_saito')
    ref, deg = _align(ref, deg, caller='itakura_saito')
    framed = _framed(ref, deg, int(sr), 'itakura_saito')
    if framed is None:
        return float('nan')
    values = np.clip(_each(_frame_is, framed, order), *_IS_RANGE)
    return _pooled(values, 'itakura_saito', framed, best=True)


def cepstral_distance(ref: np.ndarray, deg: np.ndarray, sr: int) -> float:
    """Cepstral distance of *deg* from *ref*, in dB: the log-spectral distance
    between the two frames' LPC envelopes. 0 is a perfect copy, lower is
    better.

    Each frame's LPC filter is turned into cepstral coefficients 1..p by the
    recursion of [HL08] eq. (6), p the LPC order, and the distance per frame is
    ``10 / ln(10) * sqrt(2 * sum((c_c - c_p) ** 2))``, eq. (7) (Kitawaki,
    Nagabuchi and Itoh, IEEE J. Sel. Areas Commun. 6(2), 1988). The zeroth
    coefficient, the gain, is left out, so like LLR it ignores level.

    LPC order, frames and window are LLR's. Each frame is held to [0, 10] and
    the mean is over the best 95% of frames ([HL08] Sec. III-B and
    comp_cep.m).

    A frame where the reference is digital silence and *deg* is not scores
    10 dB, the cap, for noise or speech there, under 0.3 dB for a DC offset,
    whose envelope is eps's, and 1.49 to 10 dB for a hum at 50 to 100 Hz
    peaking at -40 dBFS; one silent in both scores 0. On the padded speech of
    :func:`llr`, white noise at -60 dBFS in the zeros takes a copy from 0 to
    2.84 dB, and a warning says how many such frames there were and what
    they scored.

    :param ref: the clean signal, mono.
    :param deg: the processed signal, mono and lined up with *ref*.
    :param sr: 8000 or 16000.
    :raises ValueError: as :func:`llr`.
    :returns: the mean distance in dB, 0 to 10; nan, with a warning, as
        :func:`llr`.
    """
    order = _lpc_order(sr, 'cepstral_distance')
    ref, deg = _align(ref, deg, caller='cepstral_distance')
    framed = _framed(ref, deg, int(sr), 'cepstral_distance')
    if framed is None:
        return float('nan')
    values = np.clip(_each(_frame_cep, framed, order), *_CEP_RANGE)
    return _pooled(values, 'cepstral_distance', framed, best=True, unit=' dB')


def wss(ref: np.ndarray, deg: np.ndarray, sr: int) -> float:
    """Weighted spectral slope distance of *deg* from *ref* (Klatt, 1982). 0 is
    a perfect copy, lower is better; there is no upper bound.

    Each frame's power spectrum is summed through 25 Gaussian bands (centres
    50 Hz to 3598 Hz, widths 70 Hz to 346 Hz, the table of comp_wss.m) and
    taken to dB, floored at -100 dB. What is compared is the slope from each
    band to the next, so a change of level or a fixed tilt matters little next
    to a formant that moves or flattens. Each squared slope difference is weighted
    by Klatt's rule: ``K_max / (K_max + L_max - L)`` times
    ``K_locmax / (K_locmax + L_peak - L)``, L the band's level, ``L_max`` the
    frame's largest and ``L_peak`` the nearest peak's, with ``K_max = 20`` and
    ``K_locmax = 1`` applied to levels in dB, as comp_wss.m applies them.
    Klatt (Proc. ICASSP 1982, p. 1280) gives the same two numbers for spectra
    in units of dB times 10, that is 2 dB and 0.1 dB, so these weights fall
    off ten times more slowly than his. The clean and processed frames'
    weights are averaged. The frame's distance is
    the weighted mean, [HL08] eq. (10), and the result the mean over the best
    95% of frames, as in comp_wss.m.

    The top band's last bin is 3812.5 Hz at both rates, so at 16 kHz what
    happens above 4 kHz is not seen. The search for the nearest peak follows
    comp_wss.m, which on a rising flank stops one band short of the peak, as
    composite.m does too.

    Frames and window are LLR's: 30 ms, 75% overlap, a 512-point FFT at
    8 kHz and 1024 at 16 kHz.

    Where the reference is digital silence, every clean band sits on the
    -100 dB floor and its slopes are flat, so the frame scores *deg*'s own
    slopes there: 5.44 to 62.2 a frame for white or pink noise at -60 dBFS
    over ten noise draws, 99 to 104 for a DC offset, and 0 where *deg* is
    digital silence too. On the padded speech of :func:`llr`, white noise at
    -60 dBFS in the zeros takes a copy from 0 to 5.73, and a warning says how
    many such frames there were and what they scored.

    The floor is fixed in absolute terms, 1e-10 of full scale squared, not
    set relative to the frame, so a quiet pair loses its quieter bands to it
    and scores otherwise than the same pair at a normal level. On the 2 s
    test pair of kudio's tests under white noise at 10 dB, WSS is 35.98 at
    8 kHz with the clean peak anywhere from -6 to -46 dBFS, 35.29 at -56 and
    27.95 at -66 (at 16 kHz 30.90, 30.84 and 25.19). A warning says when band
    energies of the reference's sounding frames are under the floor; bring
    both signals to a normal level with one gain before scoring.

    :param ref: the clean signal, mono.
    :param deg: the processed signal, mono and lined up with *ref*.
    :param sr: 8000 or 16000.
    :raises ValueError: as :func:`llr`.
    :returns: the mean distance; nan, with a warning, as :func:`llr`.
    """
    _lpc_order(sr, 'wss')
    ref, deg = _align(ref, deg, caller='wss')
    framed = _framed(ref, deg, int(sr), 'wss')
    if framed is None:
        return float('nan')
    floored = []
    score = _pooled(_each(_frame_wss, framed, int(sr), floored), 'wss', framed, best=True)
    _wss_floor('wss', ref, framed, floored)
    return score


def _wss_floor(name: str, ref: np.ndarray, framed: _Frames, floored: list) -> None:
    """Warn when band energies of the reference's sounding frames are under
    the -100 dB floor, which is where WSS starts to depend on the level.

    comp_wss.m floors each band's energy at 1e-10 of full scale squared, not
    at a level relative to the frame, so a quiet pair loses its quieter bands
    to the floor and scores otherwise than the same pair at a normal level.
    Frames of digital silence sit on the floor whatever the level and have a
    warning of their own, so they are not counted. A pair under one 16-bit
    step has been called faint already, which says the same.
    """
    if not floored:
        return
    per_frame = np.concatenate(floored)[~framed.silent]
    count = int(per_frame.sum())
    peak = float(np.max(np.abs(ref)))
    if not count or peak < ONE_STEP:
        return
    warn(f"{name}(): {count} of the {per_frame.size * len(_BAND_CENTRES)} band "
         f"energies of ref's sounding frames are under the -100 dB floor "
         f"comp_wss.m holds them at, so WSS here depends on how loud the pair "
         f"is (ref peaks at {20.0 * math.log10(peak):.1f} dBFS). Bring both to "
         f"a normal level with one gain first")


def fw_segmental_snr(ref: np.ndarray, deg: np.ndarray, sr: int) -> float:
    """Frequency-weighted segmental SNR of *deg* against *ref*, in dB. Higher
    is better; a perfect copy scores the cap, 35 dB.

    Per frame, [HL08] eq. (8): both magnitude spectra are scaled to an area of
    one, summed through the 25 Gaussian bands of :func:`wss` into ``X`` and
    ``X^``, and each band's SNR ``10 log10(X^2 / (X - X^)^2)`` is averaged with
    the weight ``X ** 0.2``, the clean band's level to the power gamma = 0.2,
    eq. (9). Each frame is held to [-10, 35] dB as comp_fwseg.m does, and the
    mean is over every frame, not the best 95%.

    Scaling to unit area makes it blind to level, unlike :func:`kudio.snr` and
    :func:`kudio.segmental_snr`. The bands stop at 3.8 kHz, as in :func:`wss`,
    but the area is taken over every bin up to Nyquist, as comp_fwseg.m takes
    it, so in a 16 kHz file whatever is added or removed above 4 kHz still
    moves the score through the share it leaves the bands: noise kept above
    5 kHz at 20 dB SNR takes a copy from 35 dB to about 9, and cutting
    everything above 4 kHz to about 18.

    A band the output drops altogether scores 0 dB, since the error is then
    the clean band itself; only energy added where the clean band has little
    drives a band towards -10 dB, so an output of pure silence scores about
    -1 dB. Frames and window are LLR's.

    A frame where the reference is digital silence and *deg* is not scores
    -10 dB, the floor, for noise or speech there, 35 dB for a DC offset,
    whose spectrum scaled to unit area is eps's, and -0.69 to 12 dB for a hum
    at 50 to 100 Hz peaking at -40 dBFS; one silent in both scores 35. On
    the padded speech of :func:`llr`, white noise at -60 dBFS in the zeros
    takes a copy from 35 to 20.4 dB, and a warning says how many such frames
    there were and what they scored.

    :param ref: the clean signal, mono.
    :param deg: the processed signal, mono and lined up with *ref*.
    :param sr: 8000 or 16000.
    :raises ValueError: as :func:`llr`.
    :returns: the mean in dB, -10 to 35; nan, with a warning, as :func:`llr`.
    """
    _lpc_order(sr, 'fw_segmental_snr')
    ref, deg = _align(ref, deg, caller='fw_segmental_snr')
    framed = _framed(ref, deg, int(sr), 'fw_segmental_snr')
    if framed is None:
        return float('nan')
    return _pooled(_each(_frame_fwsnr, framed, int(sr)), 'fw_segmental_snr',
                   framed, best=False, unit=' dB')


def _rating(coefficients, pesq: float, llr_: float, wss_: float,
            segsnr: float) -> float:
    intercept, b_pesq, b_llr, b_wss, b_snr = coefficients
    value = intercept + b_pesq * pesq + b_llr * llr_ + b_wss * wss_ + b_snr * segsnr
    return float(np.clip(value, *_MOS_RANGE))


def _checked_pesq(value, scale: str) -> float:
    """A PESQ score passed in, refused when no PESQ of *scale* can be it."""
    # float() would take True as 1.0 and '2.5' as 2.5 without a word
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"composite(): pesq has to be a number, got "
                         f"{type(value).__name__} {value!r}")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"composite(): pesq has to be a score, got {value}; "
                         f"leave it out to have one computed")
    low, high = _PESQ_RANGE[scale]
    if low <= value <= high:
        return value
    lowest, highest = _PESQ_RANGE['raw'][0], _PESQ_RANGE['lqo'][1]
    if not lowest <= value <= highest:
        raise ValueError(f"composite(): pesq={value:g} is not a PESQ score: every "
                         f"variant stays within {lowest:g} to {highest:g}")
    if scale == 'raw':
        raise ValueError(f"composite(): pesq={value:g} cannot be a raw P.862 "
                         f"score, which stops at {high:g}: pass mode='wb', "
                         f"scale='lqo' if it is a wideband MOS-LQO")
    raise ValueError(f"composite(): pesq={value:g} cannot be a MOS-LQO, which "
                     f"starts at {low:g}: pass scale='raw' if it is a raw P.862 "
                     f"score")


def _computed_pesq(ref: np.ndarray, deg: np.ndarray, sr: int, mode: str,
                   scale: str) -> Tuple[float, Optional[str]]:
    """PESQ of the pair for :func:`composite`, and None; or nan and the
    reason, when the ``pesq`` package cannot score it. The reasons are the
    ones :func:`kudio.pesq` warns, so the two say the same thing."""
    return _pesq_scored(ref, deg, sr, _pesq_mode(sr, mode, scale), scale)


def _listed(names) -> str:
    """``'CSIG, CBAK, COVL and PESQ'``."""
    names = list(names)
    return names[0] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"


def _composite_scale(sr: int, mode: str, scale: str) -> str:
    """The PESQ scale *scale* stands for: 'auto' is raw for narrowband and
    MOS-LQO for wideband, the pairing pysepm's pesq() and the 2012 composite.m
    use."""
    if scale == 'auto':
        wide = mode == 'wb' or (mode == 'auto' and sr == 16000)
        return 'lqo' if wide else 'raw'
    if scale not in _PESQ_RANGE:
        raise ValueError(f"composite(): scale must be 'auto', 'lqo' or 'raw', "
                         f"got {scale!r}")
    return scale


def composite(ref: np.ndarray, deg: np.ndarray, sr: int, *, mode: str = 'auto',
              scale: str = 'auto', pesq: Optional[float] = None) -> CompositeScore:
    """CSIG, CBAK and COVL: predicted listener ratings of signal distortion,
    background intrusiveness and overall quality, 1 to 5, higher is better.

    >>> kudio.composite(clean, enhanced, 16000)                   # doctest: +SKIP
    >>> kudio.composite(clean, enhanced, 16000, pesq=2.31)        # doctest: +SKIP

    The equations are [HL06] Sec. 4.1, fitted to P.835 ratings of NOIZEUS::

        CSIG = 3.093 - 1.029 LLR + 0.603 PESQ - 0.009 WSS
        CBAK = 1.634 + 0.478 PESQ - 0.007 WSS + 0.063 segSNR
        COVL = 1.594 + 0.805 PESQ - 0.512 LLR - 0.007 WSS

    each held to [1, 5] as composite.m does. These are the CSIG, CBAK and COVL
    quoted in enhancement papers. [HL08] Table VII fits other composites, with
    IS and CEP in them, on the same ratings; they are not these.

    WSS is :func:`wss`. LLR is :func:`llr` without the cap at 2, as
    composite.m takes it. segSNR is the composite's own: 30 ms frames under
    the window of the module notes, 75% overlap, each held to [-10, 35] dB
    as composite.m holds it ([HL08] leaves such frames out instead) and all
    of them averaged. :func:`kudio.segmental_snr` takes 512-sample
    rectangular frames with a hop of 256 and is a different number on the same
    pair, so it cannot stand in for this one: composite.m feeds CBAK the
    windowed version. Its per-frame formula, ``10 log10(S / (N + eps) +
    eps)``, puts a frame of digital silence at the -10 dB floor even against
    an exact copy, so zeros padded onto a file pull segSNR and CBAK down;
    composite.m and pysepm do the same. Half a second of zeros either side
    of 2 s of speech, 125 of its 396 frames, takes an exact copy's segSNR
    from 35 to 20.8 dB and its CBAK down by 0.89. Where the output is not
    silent in those frames, LLR, taken uncapped, compares it there with eps:
    noise or speech scores 14.1 to 26.1 a frame, a hum at 50 to 100 Hz
    peaking at -40 dBFS 0.075 to 13.2, and a DC offset within 0.005 of 0.
    White noise at -60 dBFS in the zeros of that copy takes CSIG and COVL
    from 4.30 and 3.20 to 1 and 1, the floor of the scale, with PESQ held at
    2.0. A warning says how many such frames there were and what they scored.

    LLR, WSS and segSNR compare the pair over the length the two share, and
    PESQ takes the two whole, since it finds the delay between them itself.
    composite.m does the same: it cuts its two files to one length for the
    first three and hands pesq() the files.

    **Which PESQ.** The regression was fitted on the raw narrowband P.862
    score of 8 kHz speech: [HL06] eq. (2) defines PESQ as ``4.5 - 0.1 D_ind -
    0.0309 A_ind``, and NOIZEUS is 8 kHz. At 16 kHz nearly every published
    CSIG, CBAK and COVL, VoiceBank-DEMAND's included, has wideband MOS-LQO
    (P.862.2) in its place. Loizou's own composite.m with the 2012 pesq.m does
    that at 16 kHz, and so does pysepm (schmiph2/pysepm, whose pesq() picks
    the wideband mode at 16 kHz), and the noisy-input row those tables share
    (PESQ 1.97, CSIG 3.35, CBAK 2.44, COVL 2.63; S. Pascual, A. Bonafonte and
    J. Serra, "SEGAN: Speech enhancement generative adversarial network",
    Proc. Interspeech 2017, pp. 3642-3646, Table 1,
    doi:10.21437/Interspeech.2017-1428) is what that variant gives. The
    defaults follow them: ``mode='auto', scale='auto'`` is wideband MOS-LQO at
    16 kHz and raw narrowband P.862 at 8 kHz. ``mode='nb', scale='raw'`` at
    16 kHz is the variant the regression was fitted on, which Loizou's 2007 toolbox and his
    composite.zip also use at that rate; it gives higher figures, 3.02, 3.99,
    2.95 and 3.48 on that same noisy set. Every point of PESQ moves CSIG by
    0.603, CBAK by 0.478 and COVL by 0.805, so a table should say which went
    in.

    Computed here, PESQ comes from the ``pesq`` package, which cannot score
    every pair the other measures can: it needs 0.25 s, finds no utterance in
    some short or quiet references, and gets NaN for an output that is
    digital silence or close to it. Then the ratings and the PESQ field are
    nan, with a warning that names the cause, and LLR, WSS and segSNR are
    still returned.

    :param mode: ``'auto'`` (default), ``'nb'`` or ``'wb'``, as in
        :func:`kudio.pesq`.
    :param scale: ``'auto'`` (default): raw for narrowband and MOS-LQO for
        wideband, which with ``mode='auto'`` is wideband MOS-LQO at 16 kHz and
        raw P.862 at 8 kHz. ``'raw'`` or ``'lqo'`` as in :func:`kudio.pesq`.
    :param pesq: a PESQ score you already have, used as it is instead of
        computing one; *mode* and *scale* then say what it is. Nothing can
        check that it belongs to this pair, only that it is in the range its
        *scale* allows.
    :raises ValueError: at a rate other than 8 or 16 kHz, for a PESQ variant
        that cannot be given (wideband at 8 kHz, a raw wideband score), for a
        *pesq* outside its scale's range,
        or for input :func:`llr` refuses.
    :raises DependencyError: without *pesq* and without the ``pesq`` package
        (``kudio[pesq]``).
    :returns: a :class:`CompositeScore`. Every field but a *pesq* passed in is
        nan, with a warning, for a pair too short for one frame (37.5 ms) or a
        silent reference; csig, cbak, covl and pesq are nan for a pair the
        ``pesq`` package cannot score, and csig, cbak, covl and whichever of
        llr, wss and segsnr has a frame too loud to square in float64 are
        nan. One warning names everything that is nan and why.
    """
    order = _lpc_order(sr, 'composite')
    scale = _composite_scale(int(sr), mode, scale)
    try:
        _pesq_mode(int(sr), mode, scale)
    except ValueError as e:
        raise ValueError(f"composite(): {e}") from None
    if pesq is None:
        try:
            import pesq as _pesq_package  # noqa: F401
        except ImportError as e:
            raise DependencyError('pesq', extra='pesq') from e
    else:
        pesq = _checked_pesq(pesq, scale)
    # PESQ gets the pair whole, as composite.m hands pesq() the two files,
    # and the frame measures the length the two share
    whole = _sides(ref, deg, caller='composite')
    ref, deg = _cut(*whole, caller='composite',
                    whole_for='PESQ' if pesq is None else None)
    framed = _framed(ref, deg, int(sr), 'composite')
    if framed is None:
        nan = float('nan')
        return CompositeScore(nan, nan, nan, nan if pesq is None else pesq,
                              nan, nan, nan)
    floored = []
    frames = {'LLR': _each(_frame_llr, framed, order),
              'WSS': _each(_frame_wss, framed, int(sr), floored),
              'segSNR': _each(_frame_segsnr, framed)}
    lost = {part: int(np.count_nonzero(np.isnan(v))) for part, v in frames.items()}
    llr_ = float('nan') if lost['LLR'] else _best(frames['LLR'])
    wss_ = float('nan') if lost['WSS'] else _best(frames['WSS'])
    segsnr = float(np.mean(frames['segSNR']))
    # one warning for everything that went missing, as every measure gives
    why = []
    if any(lost.values()):
        counted = _listed(f"{n} in {part}" for part, n in lost.items() if n)
        why.append(f"of the {len(framed.clean)} frames, {counted} came out NaN, "
                   f"{_TOO_LOUD}")
    if pesq is None:
        pesq, failed = _computed_pesq(*whole, int(sr), mode, scale)
        if failed is not None:
            why.append(failed)
    rated = CompositeScore(csig=_rating(_CSIG, pesq, llr_, wss_, segsnr),
                           cbak=_rating(_CBAK, pesq, llr_, wss_, segsnr),
                           covl=_rating(_COVL, pesq, llr_, wss_, segsnr),
                           pesq=float(pesq), llr=llr_, wss=wss_, segsnr=segsnr)
    if why:
        parts = {'PESQ': rated.pesq, 'LLR': llr_, 'WSS': wss_, 'segSNR': segsnr}
        unscored = ['CSIG', 'CBAK', 'COVL'] + [k for k, v in parts.items() if math.isnan(v)]
        kept = [k for k, v in parts.items() if not math.isnan(v)]
        warn(f"composite(): {'; '.join(why)}; {_listed(unscored)} scored nan"
             + (f", {_listed(kept)} kept" if kept else ""))
    else:
        _composite_silence(framed, frames)
        _wss_floor('composite', ref, framed, floored)
    return rated


def _composite_silence(framed: _Frames, frames: dict) -> None:
    """The warning :func:`_pooled` gives, for the parts of the composite.
    segSNR puts every frame where the reference is digital silence at its
    floor, even against an exact copy, so all of those are counted."""
    silent = int(np.count_nonzero(framed.silent))
    if not silent:
        return
    text = (f"composite(): {silent} of the reference's {len(framed.silent)} frames "
            f"are digital silence, and segSNR scores them "
            f"{_spread(frames['segSNR'][framed.silent], ' dB')} whatever deg holds "
            f"there")
    against_eps = framed.silent & ~framed.silent_deg
    count = int(np.count_nonzero(against_eps))
    if count:
        text += (f"; in {count} of them deg is not silent, and there LLR, taken "
                 f"uncapped, scores {_spread(frames['LLR'][against_eps])} and WSS "
                 f"{_spread(frames['WSS'][against_eps])}, comparing deg with eps "
                 f"rather than with speech")
    warn(f"{text}. Trim the silence off both to compare the rest alone.")
