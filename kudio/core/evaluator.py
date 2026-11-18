# -*- coding: utf-8 -*-
"""Speech-enhancement quality metrics: PESQ, STOI, SDR, the SNR family, and
the log-spectral distance; :func:`score` also runs Hu and Loizou's measures
from :mod:`kudio.core.quality`.

The SNR family (:func:`snr`, :func:`si_sdr`, :func:`segmental_snr`,
:func:`sdi`) and :func:`lsd` are pure numpy. The rest are optional
dependencies::

    pip install "kudio[eval]"   # mir_eval + pystoi: SDR, STOI
    pip install "kudio[pesq]"   # pesq: PESQ -- built from source, needs a C compiler

**Which PESQ.** ITU-T P.862 scores narrowband speech and P.862.2 wideband.
P.862 can be reported raw or mapped to MOS-LQO through P.862.1; P.862.2
reports only a MOS-LQO of its own. They are different numbers for one file: a
raw narrowband 2.0 is a MOS-LQO of about 1.6. Papers that just say "PESQ" do
not agree on which they mean -- some quote the raw narrowband score, which the
ITU reference program prints ahead of its MOS-LQO, and VoiceBank-DEMAND
tables, SEGAN's noisy-input row (PESQ 1.97) among them, carry wideband
MOS-LQO -- so read how a paper scored before comparing with it. :func:`pesq`
takes the bandwidth and the scale as arguments rather than picking one
silently.
"""
from __future__ import annotations

import functools
import importlib.util
import logging
import math
import numbers
import os
import shutil
import subprocess
import warnings
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

from kudio._progress import tqdm
from kudio.core._framing import frame_view
from kudio.core._policy import (_CHECKED, checked, faint, sample_rate, scored_nan,
                                signal, warn)
from kudio.exceptions import DependencyError

__all__ = ['AudioEvaluate', 'eval_metrics', 'score', 'pesq', 'sdi', 'METRICS',
           'si_sdr', 'snr', 'segmental_snr', 'lsd', 'MAX_LENGTH_MISMATCH']

log = logging.getLogger(__name__)


def _pandas():
    try:
        import pandas as pd
        return pd
    except ImportError as e:
        raise DependencyError('pandas', extra='data') from e


# --- lightweight, dependency-free metrics --------------------------------------
#: How much of the longer signal a comparison may leave unmeasured, as a
#: fraction of it. Every metric here but PESQ lines the pair up at sample zero
#: and cuts both to the common length, so the tail of the longer one is never
#: looked at; past half of it the score describes the overlap and not the file whose
#: name it is reported under. Half is also well clear of the slop a real
#: pipeline produces -- a window or two lost at the tail, a resampler's
#: rounding -- which is the case truncation is there to absorb in the first
#: place. Raise it to compare a pair that really is that lopsided.
MAX_LENGTH_MISMATCH = 0.5


def _align(ref: np.ndarray, deg: np.ndarray, *,
           names: Tuple[str, str] = ('ref', 'deg'),
           tolerance: Optional[float] = None, caller: Optional[str] = None):
    """Cut both signals to the length they share, and say when that cuts.

    A length difference is not a detail: the pair is compared sample for
    sample, so whatever hangs off the end of the longer one simply does not
    enter the number, and nothing in the result says so. Two files that
    overlap in their first 1000 samples and differ everywhere after score as a
    bit-perfect copy. So say it every time, and refuse once the part going
    unmeasured is past *tolerance* (default :data:`MAX_LENGTH_MISMATCH`).

    :param names: what to call the two in the message -- file paths where the
        caller has them, otherwise the argument names.
    :param caller: the metric's name, which then starts every message as
        ``lsd(): ...``.
    """
    ref, deg = _sides(ref, deg, names=names, caller=caller)
    return _cut(ref, deg, names=names, tolerance=tolerance, caller=caller)


def _sides(ref: np.ndarray, deg: np.ndarray, *,
           names: Tuple[str, str] = ('ref', 'deg'),
           caller: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Each of the two checked as a signal, and the pair called faint if it
    is, at the lengths they came in: what PESQ is given, since it finds the
    delay between the two itself."""
    # integer samples on their type's full scale: an int16 pair scored as
    # plain numbers is mostly right, being a ratio, until one side is float
    # and the scales differ by 32768
    ref, deg = signal(ref, names[0], caller), signal(deg, names[1], caller)
    faint(caller, [(names[0], ref), (names[1], deg)])
    return ref, deg


def _cut(ref: np.ndarray, deg: np.ndarray, *,
         names: Tuple[str, str] = ('ref', 'deg'),
         tolerance: Optional[float] = None,
         caller: Optional[str] = None,
         whole_for: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """The length half of :func:`_align`, for a pair :func:`_sides` has
    checked. *whole_for* names what the caller runs on the pair uncut, for
    the warning to say so."""
    n = min(len(ref), len(deg))
    longest = max(len(ref), len(deg))
    # under checked() the caller has already warned about this pair's
    # lengths, or refused it, before passing it on whole to the composite
    if longest != n and not _CHECKED.get():
        limit = MAX_LENGTH_MISMATCH if tolerance is None else float(tolerance)
        unmeasured = (longest - n) / longest
        counts = f"{names[0]} is {len(ref)} samples and {names[1]} {len(deg)}"
        if caller is not None:
            counts = f"{caller}(): {counts}"
        if unmeasured > limit:
            # the sample count leads and the percentage follows it: rounding a
            # percentage can read as a contradiction right at the gate, and
            # 15999 of 16000 unmeasured is not "100%" when one was scored
            raise ValueError(
                f"{counts}: comparing them leaves {longest - n} of the longer "
                f"one's {longest} samples unmeasured ({unmeasured * 100:.1f}%), "
                f"past the {limit * 100:.0f}% "
                f"limit, so the score would be about the overlap rather than "
                f"about the pair. Check that these two are the same "
                f"utterance, line them up with kudio.align, or raise "
                f"kudio.core.evaluator.MAX_LENGTH_MISMATCH to score them as "
                f"they sit.")
        but = f", except by {whole_for}, which gets both whole" if whole_for else ""
        warn(f"{counts}: only the first {n} samples are scored and the rest is "
             f"ignored{but}. Use kudio.align if the short one is delayed rather "
             f"than just short, or trim the pair yourself to say which part "
             f"you meant.")
    return ref[:n], deg[:n]


def _checked_pair(ref, deg, caller: str) -> Tuple[np.ndarray, np.ndarray]:
    """The pair cut to one length and refused if it is malformed, the way
    every metric here takes it."""
    return _align(ref, deg, caller=caller)


def _overflows(ref: np.ndarray, deg: np.ndarray) -> bool:
    """Whether a back-end squaring the pair could overflow float64.

    pystoi and mir_eval square and sum inside, where kudio cannot catch it:
    on 3 s of speech, pystoi's STOI comes back nan at 1e153 of full scale and
    0.0, the worst score there is, from 1e154; mir_eval's SDR is nan from
    1e152. The pair's energy with 1024 times to spare refuses from about
    1e150 on such a clip, and normal audio is 150 orders of magnitude clear
    of it.
    """
    with np.errstate(over='ignore', invalid='ignore'):
        energy = 1024.0 * (float(np.dot(ref, ref)) + float(np.dot(deg, deg)))
    return not math.isfinite(energy)


def _too_loud(caller: str, ref: np.ndarray, deg: np.ndarray) -> float:
    """nan, with the warning for a pair whose squares overflow float64.

    The measures here square samples and sum the squares, which past about
    1e154 of full scale overflows; they run with numpy's overflow warnings
    off, one for every frame of segmental_snr, and a result that is not
    finite, from input that is, can only have come from that.
    """
    peak = max(float(np.max(np.abs(ref))), float(np.max(np.abs(deg))))
    return scored_nan(caller, f"the pair is too loud to square in float64 (it "
                              f"peaks at {peak:.3g}); scale both signals down "
                              f"by the same factor")


def snr(ref: np.ndarray, deg: np.ndarray) -> float:
    """Signal-to-noise ratio (dB) of *deg* relative to *ref* (pure numpy).

    nan, with a warning, for an empty pair or a *ref* that is digital silence
    throughout, which has no signal to set the noise against. An exact copy
    scores the finite figure the 1e-12 guards allow, 10 log10 of the copy's
    energy over 1e-12.
    """
    ref, deg = _checked_pair(ref, deg, 'snr')
    if ref.size == 0:
        return scored_nan('snr', "the pair is empty")
    if not np.any(ref):
        return scored_nan('snr', "ref is digital silence throughout, so there is no "
                                 "signal to set the noise against")
    with np.errstate(over='ignore', invalid='ignore'):
        value = float(10 * np.log10((np.sum(ref ** 2) + 1e-12) /
                                    (np.sum((ref - deg) ** 2) + 1e-12)))
    return value if math.isfinite(value) else _too_loud('snr', ref, deg)


def si_sdr(ref: np.ndarray, deg: np.ndarray) -> float:
    """Scale-invariant SDR (dB) — immune to overall gain differences.

    nan, with a warning, for an empty pair or a constant *ref* (digital
    silence, a DC offset alone, or a single sample): with its mean taken out
    there is nothing to project *deg* on. A constant *deg* scores 0 dB, which
    is what the 1e-12 guards make of 0 / 0, and warns that it does; it stays
    a number so that an output that collapses still pulls a mean down.
    """
    ref, deg = _checked_pair(ref, deg, 'si_sdr')
    if ref.size == 0:
        return scored_nan('si_sdr', "the pair is empty")
    if not np.any(ref):
        return scored_nan('si_sdr', "ref is digital silence throughout, so there is "
                                    "nothing to project deg on")
    if np.all(ref == ref[0]):
        return scored_nan('si_sdr', "ref is constant, so once its mean is taken out "
                                    "there is nothing to project deg on")
    if np.all(deg == deg[0]):
        warn("si_sdr(): deg is constant (digital silence or a DC offset alone), so "
             "once its mean is taken out it has no projection on ref and no "
             "residual; the 0 dB is what the 1e-12 guards make of 0 / 0")
    value = _si_sdr_value(ref, deg)
    return value if math.isfinite(value) else _too_loud('si_sdr', ref, deg)


def _si_sdr_value(ref: np.ndarray, deg: np.ndarray) -> float:
    """SI-SDR of a checked pair of one length, with nothing said: nan or inf
    where the pair is too loud to square, 0 dB for a constant *deg*.
    :func:`si_sdr` and :mod:`kudio.core.separation` both come here, so the two
    give the same number to the last bit.

    Both signals have their mean taken out first, as Luo and Mesgarani
    define SI-SNR, the same quantity, for Conv-TasNet (IEEE/ACM TASLP 27(8),
    2019, eq. 15, doi:10.1109/TASLP.2019.2915167), and as asteroid's SI-SDR
    does by default; torchmetrics' does it only with ``zero_mean=True``. A
    DC offset is where the two readings part.
    """
    with np.errstate(over='ignore', invalid='ignore'):
        centred_ref = ref - ref.mean()
        centred_deg = deg - deg.mean()
        proj = (np.dot(centred_deg, centred_ref)
                / (np.dot(centred_ref, centred_ref) + 1e-12)) * centred_ref
        noise = centred_deg - proj
        return float(10 * np.log10((np.sum(proj ** 2) + 1e-12) /
                                   (np.sum(noise ** 2) + 1e-12)))


def _samples(value, name: str, low: int, high: Optional[int] = None) -> int:
    """*value* as a whole number of samples from *low* to *high*, or the
    ValueError :func:`segmental_snr` gives for it."""
    whole = isinstance(value, numbers.Integral) and not isinstance(value, bool)
    if not whole or value < low or (high is not None and value > high):
        span = f"{low} or more" if high is None else f"from {low} to {high}"
        raise ValueError(f"segmental_snr(): {name} has to be a whole number of "
                         f"samples, {span}, got {value!r}")
    return int(value)


def segmental_snr(ref: np.ndarray, deg: np.ndarray, frame_length: int = 512,
                  hop_length: int = 256) -> float:
    """Frame-averaged (segmental) SNR in dB, clipped to [-10, 35] per frame.

    nan, with a warning, for a pair shorter than one frame or a *ref* that is
    digital silence in every frame scored. A frame that is digital silence in
    *ref* alone scores -10 dB, or 0 dB where *deg* is silent there too, and a
    warning counts them: a copy padded with zeros scores under 35.

    :param frame_length: samples per frame, 1 or more.
    :param hop_length: samples from one frame to the next, from 1 to
        *frame_length*; a longer hop would leave samples unmeasured.
    """
    frame_length = _samples(frame_length, 'frame_length', 1)
    hop_length = _samples(hop_length, 'hop_length', 1, frame_length)
    ref, deg = _checked_pair(ref, deg, 'segmental_snr')
    if ref.size < frame_length:
        if ref.size == 0:
            return scored_nan('segmental_snr', "the pair is empty")
        return scored_nan('segmental_snr', f"the pair needs at least {frame_length} "
                                           f"samples to score one frame, and has "
                                           f"{ref.size}")
    seg = []
    silent = 0
    with np.errstate(over='ignore', invalid='ignore'):
        for start in range(0, len(ref) - frame_length + 1, hop_length):
            r = ref[start:start + frame_length]
            d = deg[start:start + frame_length]
            silent += not np.any(r)
            noise = r - d
            val = 10 * np.log10((np.sum(r ** 2) + 1e-12) / (np.sum(noise ** 2) + 1e-12))
            seg.append(np.clip(val, -10, 35))
    if not np.all(np.isfinite(seg)):
        return _too_loud('segmental_snr', ref, deg)
    if silent == len(seg):
        where = "throughout" if not np.any(ref) else "in every frame that is scored"
        return scored_nan('segmental_snr', f"ref is digital silence {where}, so there "
                                           f"is no signal to set the noise against")
    if silent:
        warn(f"segmental_snr(): {silent} of ref's {len(seg)} frames are digital "
             f"silence; each scores -10 dB, or 0 dB where deg is silent there too")
    return float(np.mean(seg))


def sdi(ref: np.ndarray, deg: np.ndarray) -> float:
    """Speech distortion index: ``sum((ref - deg)**2) / sum(ref**2)``.

    0 is a perfect copy and 1 is as much error as there is speech; lower is
    better. It is :func:`snr` on a linear scale, ``10 ** (-snr / 10)``: the
    error energy of the whole output against the clean signal. The
    speech-distortion index of the noise-reduction literature (J. Chen,
    J. Benesty, Y. Huang and S. Doclo, "New insights into the noise reduction
    Wiener filter", IEEE Trans. Audio, Speech, Lang. Process. 14(4),
    pp. 1218-1234, 2006, doi:10.1109/TSA.2005.860851) is taken differently, on
    the clean speech passed through the filter alone, so the two are not the
    same number.

    nan, with a warning, for an empty pair or a *ref* that is digital silence
    throughout, which leaves no speech energy to divide by.
    """
    ref, deg = _checked_pair(ref, deg, 'sdi')
    if ref.size == 0:
        return scored_nan('sdi', "the pair is empty")
    if not np.any(ref):
        return scored_nan('sdi', "ref is digital silence throughout, so there is no "
                                 "speech energy to divide by")
    with np.errstate(over='ignore', invalid='ignore'):
        value = float(np.sum((ref - deg) ** 2) / (np.sum(ref ** 2) + 1e-12))
    return value if math.isfinite(value) else _too_loud('sdi', ref, deg)


#: The frame :func:`lsd` uses unless told otherwise, in seconds: URGENT 2025's
#: 32 ms, which its scoring script turns into samples as ``int(sr * 0.032)``
#: -- 512 at 16 kHz, 1411 at 44.1 kHz.
_LSD_FRAME = 0.032

#: The scales :func:`lsd` reports on, as what each multiplies the natural log
#: by: base 10 is ln / ln 10, and decibels are ten of those.
_LSD_SCALES = {'ln': 1.0, 'log10': 1.0 / math.log(10.0),
               'db': 10.0 / math.log(10.0)}

#: Frames transformed at a time. One block of 1024 frames of 1536 samples
#: (48 kHz) is 12.6 MB; the whole of a ten-minute 48 kHz file at once would be
#: 460 MB per signal.
_LSD_BLOCK = 1024


def lsd(ref: np.ndarray, deg: np.ndarray, sr: int, *,
        n_fft: Optional[int] = None, hop_length: Optional[int] = None,
        scale: str = 'ln', match_gain: bool = True, eps: float = 1e-8) -> float:
    """Log-spectral distance of *deg* from *ref*: 0 for the same power
    spectrum, and larger the further apart the two are; lower is better.

    >>> kudio.lsd(clean, enhanced, 16000)                    # URGENT 2025
    >>> kudio.lsd(clean, extended, 44100, n_fft=2048, hop_length=441,
    ...           scale='log10', match_gain=False, eps=1e-12)   # ssr_eval

    For every STFT frame, the root mean square over frequency of the log of
    the power ratio, and then the mean over frames: the RMS log-spectral
    distance, which Kuleshov et al. (ICLR workshop 2017, Eq. 3) cite to
    Gray and Markel (1976)::

        mean_t sqrt( mean_f log(|R|^2 / (|D| + eps)^2 + eps)^2 )

    The defaults are those of ``lsd_metric`` in URGENT 2025's scoring script
    (``evaluation_metrics/calculate_intrusive_se_metrics.py`` in
    github.com/urgent-challenge/urgent2025_challenge), so a number from here
    can sit next to the challenge's. That is the reason they are
    the defaults: kudio scores enhancement, URGENT 2025 reports this distance
    beside PESQ, ESTOI and SDR at every rate from 8 to 48 kHz, and its script
    pins down every choice the papers leave open. The bandwidth-extension
    papers use the same formula on another scale and STFT: base 10 (NVSR,
    Liu et al., Interspeech 2022, Eq. 7, scored with the ``ssr_eval``
    package), as in the second example.

    The challenge's script reads files as float32, and its STFT then runs in
    single precision, as ``ssr_eval``'s does when it is handed float32 audio;
    this one runs in double. On full-band output they agree to within about
    1e-6. Where *deg* has an empty band they do not, because that band sits
    near float32's rounding floor: 0.38 apart from URGENT's number on a
    48 kHz test signal low-passed at 12 kHz, and 0.42 apart from
    ``ssr_eval``'s on one low-passed at 4 kHz. Rounded to 16 bits first, as
    a 16-bit file is, the empty band holds the rounding noise instead, and
    that pair agrees with ``ssr_eval`` to 4e-8 again.

    :param n_fft: frame length in samples. Default ``int(0.032 * sr)``,
        URGENT's 32 ms. Frames are periodic Hann windows, centred, with
        ``n_fft // 2`` zeros padded at each end of the signal, which is what
        ``librosa.stft`` has done by default since librosa 0.9.
    :param hop_length: samples between frames. Default ``n_fft // 2``, which
        is URGENT's 16 ms at every rate.
    :param scale: ``'ln'`` (default) -- the natural log of the power ratio,
        as URGENT 2025 scores it. ``'log10'`` -- base 10, as NVSR's Eq. 7
        and ``ssr_eval`` take it. ``'db'`` -- decibels, ten times base 10.
        One distance on three scales: base 10 is ``ln / 2.3026`` and
        decibels ``4.3429 * ln``. A published number is not always on the
        scale its label says: Kuleshov et al. caption their table "in dB"
        and take the natural log in their code. Find out which scale a
        number is on before setting a score beside it.
    :param match_gain: first scale *deg* by
        ``sum(ref * deg) / (sum(deg ** 2) + eps)``, its least-squares gain
        against *ref*, as URGENT 2025 does, so a level difference is not
        counted. That gain is a projection onto *ref*, so it needs the pair
        lined up in time like every metric here, and more than a spectral
        distance would suggest: on the speech fixture in kudio's tests,
        delaying *deg* by 1 ms takes the distance from 0 to 2.45, because
        the projection shrinks *deg* to 0.30 of its level; with ``False`` the
        same delay scores 0.38. ``False`` compares the levels as they are, as
        ``ssr_eval`` does.
    :param eps: added to the magnitude of *deg*, to the ratio, and to the
        gain's denominator. 1e-8 is URGENT 2025's; ``ssr_eval`` uses 1e-12.
        Where *deg* is digital zero, a magnitude of *eps* is all there is to
        compare against, so there it sets the score: half a second of
        dropout in a 3 s test signal scores 4.90 at 1e-8 and 7.74 at 1e-12,
        where the same signal with noise in every bin moves by 0.001.
    :raises ValueError: for an unknown *scale*, a frame or hop that cannot
        be used, a rate that is not a positive, finite number of Hz, an *eps*
        that is not positive, samples that are not finite, a
        ``(samples, channels)`` pair, and a length mismatch past
        :data:`MAX_LENGTH_MISMATCH` (a smaller one warns).
    :returns: the distance on *scale*. nan, with a warning, for an empty pair,
        for a pair shorter than one frame of *n_fft* samples, which leaves no
        full frame to measure, and for *ref* that is digital silence
        throughout. Frames where *ref* is digital silence score ``-ln(eps)``
        each (18.4 at the default) whatever *deg* holds there; they are
        counted, as URGENT counts them, and a warning says how many there
        were. A *deg* that is digital silence throughout is scored, since
        URGENT scores it, with a warning that the number then measures *ref*
        against *eps*.
    """
    if scale not in _LSD_SCALES:
        raise ValueError(f"lsd(): scale must be one of {tuple(_LSD_SCALES)}, "
                         f"got {scale!r}")
    sr = sample_rate(sr, 'lsd')
    if not eps > 0:
        raise ValueError(f"lsd(): eps must be positive, got {eps}: it is what keeps "
                         f"a band deg leaves empty from dividing by zero")
    n_fft = int(sr * _LSD_FRAME) if n_fft is None else int(n_fft)
    if n_fft < 2:
        raise ValueError(f"lsd(): the frames need at least 2 samples, got "
                         f"n_fft={n_fft}; pass n_fft in samples (the default "
                         f"is int(0.032 * sr))")
    hop = n_fft // 2 if hop_length is None else int(hop_length)
    if not 1 <= hop <= n_fft:
        raise ValueError(f"lsd(): hop_length must be from 1 to n_fft ({n_fft}), "
                         f"got {hop}: a longer hop leaves samples between the "
                         f"frames unmeasured")
    ref, deg = _align(ref, deg, caller='lsd')
    if ref.size == 0:
        return scored_nan('lsd', "the pair is empty")
    if ref.size < n_fft:
        return scored_nan('lsd', f"the pair needs at least one full frame, {n_fft} "
                                 f"samples ({1000 * n_fft / sr:.0f} ms at {sr:g} Hz), "
                                 f"and has {ref.size}")
    deg_silent = not np.any(deg)
    loud = ref, deg
    if match_gain:
        with np.errstate(over='ignore', invalid='ignore'):
            deg = deg * (float(np.dot(ref, deg)) / (float(np.dot(deg, deg)) + eps))

    # the periodic Hann, librosa's 'hann': it divides by n_fft, not n_fft - 1
    window = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n_fft) / n_fft)
    pad = n_fft // 2
    ref_frames = frame_view(np.pad(ref, pad), n_fft, hop)
    deg_frames = frame_view(np.pad(deg, pad), n_fft, hop)
    per_frame = np.empty(len(ref_frames))
    silent = 0
    for start in range(0, len(per_frame), _LSD_BLOCK):
        stop = min(start + _LSD_BLOCK, len(per_frame))
        r = np.abs(np.fft.rfft(ref_frames[start:stop] * window, axis=1))
        d = np.abs(np.fft.rfft(deg_frames[start:stop] * window, axis=1))
        silent += int(np.count_nonzero(~np.any(r > 0.0, axis=1)))
        with np.errstate(over='ignore', invalid='ignore'):
            log_ratio = np.log(r ** 2 / (d + eps) ** 2 + eps)
            per_frame[start:stop] = np.sqrt(np.mean(log_ratio ** 2, axis=1))

    factor = _LSD_SCALES[scale]
    if not np.all(np.isfinite(per_frame)):
        return _too_loud('lsd', *loud)
    if silent == len(per_frame):
        return scored_nan('lsd', "ref is digital silence throughout, so there is no "
                                 "spectrum to compare deg against")
    if deg_silent:
        warn("lsd(): deg is digital silence throughout, so the score measures ref "
             "against eps, the floor added to deg")
    if silent:
        warn(f"lsd(): {silent} of ref's {len(per_frame)} frames are digital "
             f"silence, and the formula scores each of them "
             f"{-math.log(eps) * factor:.1f} whatever deg holds there. Trim the "
             f"silence off both to compare the rest alone.")
    return float(np.mean(per_frame)) * factor


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
        # P.862.2 reports no raw score, so a raw score is a narrowband one
        mode = 'wb' if sr == 16000 and scale == 'lqo' else 'nb'
    if mode == 'wb' and sr != 16000:
        raise ValueError("wideband PESQ (P.862.2) runs at 16 kHz only, as "
                         "P.862's Annex A requires")
    if mode == 'wb' and scale == 'raw':
        raise ValueError("P.862.2 reports no raw score: use scale='lqo', or "
                         "mode='nb' for the raw P.862 score")
    return mode


def pesq(ref: np.ndarray, deg: np.ndarray, sr: int, *, mode: str = 'auto',
         scale: str = 'lqo') -> float:
    """PESQ of *deg* against *ref* (ITU-T P.862 / P.862.2).

    >>> kudio.pesq(clean, enhanced, 16000)                          # WB MOS-LQO
    >>> kudio.pesq(clean, enhanced, 16000, mode='nb', scale='raw')  # raw P.862

    :param mode: ``'nb'`` -- P.862 narrowband, at 8 or 16 kHz. ``'wb'`` --
        P.862.2 wideband, 16 kHz only, as P.862's Annex A (Amendment 2, 2005)
        requires of a wideband implementation. ``'auto'`` (default) --
        wideband at 16 kHz and narrowband at 8 kHz, which is what kudio has
        always reported; narrowband whenever *scale* is ``'raw'``.
    :param scale: ``'lqo'`` (default) -- MOS-LQO, the mapped score both
        standards define. ``'raw'`` -- the narrowband P.862 score before the
        P.862.1 mapping, -0.5 to 4.5. In narrowband mode the ITU reference
        program prints it ahead of the MOS-LQO.
    :raises ValueError: off 8 / 16 kHz, or for a combination the standards
        rule out (wideband at 8 kHz) or P.862.2 does not report (a raw
        wideband score).
    :raises DependencyError: without the ``pesq`` package (``kudio[pesq]``).

    The two go to the ``pesq`` package whole, whatever their lengths, the
    way the ITU program reads two files. PESQ finds the delay between them
    itself, utterance by utterance, and works over both as they are, so
    cutting the longer one changes the score even where the cut drops only
    silence: on the 40 pairs of ITU's conformance test 2(b), 22 of them two
    lengths, it moved the score by up to 0.56. NaN or inf samples are
    refused.
    A pair the ``pesq`` package cannot score gives nan and one warning saying
    why: an empty side, a *ref* that is digital silence, a side under 0.25 s,
    no utterance found in *ref*, and a *deg* at or near digital silence, for
    which the package gets NaN.

    Runs on the ``pesq`` package from PyPI, version 2.0 of the ITU code,
    which returns MOS-LQO; the raw score is that inverted through P.862.1,
    ``(4.6607 - ln(4 / (lqo - 0.999) - 1)) / 1.4945``. Its wideband scores
    are the 2005 code's, without ITU's 2018 correction (P.862 Corrigendum 2),
    which in ITU's own table raises them by 0.73 on average.
    """
    sr = sample_rate(sr, 'pesq')
    try:
        mode = _pesq_mode(sr, mode, scale)
    except ValueError as e:
        raise ValueError(f"pesq(): {e}") from None
    try:
        import pesq as _package  # noqa: F401
    except ImportError as e:
        raise DependencyError('pesq', extra='pesq') from e
    ref, deg = _sides(ref, deg, caller='pesq')
    value, why = _pesq_scored(ref, deg, sr, mode, scale)
    return value if why is None else scored_nan('pesq', why)


def _pesq_scored(ref: np.ndarray, deg: np.ndarray, sr: int, mode: str,
                 scale: str) -> Tuple[float, Optional[str]]:
    """PESQ of a pair with no NaN in it, and None; or nan and why the
    ``pesq`` package could not score it, worded to end a warning.

    :func:`pesq` warns the reason, and :func:`kudio.composite` puts it in the
    one warning it gives, so the two say the same thing and neither says it
    twice. *mode* is ``'nb'`` or ``'wb'``, already chosen.
    """
    nan = float('nan')
    if ref.size == 0 or deg.size == 0:
        return nan, ("the pair is empty" if ref.size == deg.size
                     else f"{'ref' if ref.size == 0 else 'deg'} is empty")
    if not np.any(ref):
        # the package would find no utterance, after two RuntimeWarnings of
        # its own about dividing by zero
        return nan, "ref is digital silence throughout, so there is no utterance in it"
    import pesq as package
    # looked up rather than imported, so that a stand-in for the package
    # without them still scores
    too_short = getattr(package, 'BufferTooShortError', ())
    no_utterance = getattr(package, 'NoUtterancesError', ())
    try:
        lqo = float(package.pesq(sr, ref, deg, mode))
    except too_short:
        # in samples: 3999 of them at 16 kHz would print as 0.250 s
        if ref.size == deg.size:
            return nan, (f"PESQ needs at least 0.25 s, {sr // 4} samples, and the "
                         f"pair has {ref.size}")
        return nan, (f"PESQ needs at least 0.25 s, {sr // 4} samples, on each "
                     f"side, and ref has {ref.size} and deg {deg.size}")
    except no_utterance:
        return nan, "PESQ found no utterance in ref to score"
    except ValueError as e:
        # pesq 0.0.4 gets a NaN score for an output at or near digital
        # silence (1e-30 of full scale fails, 1e-20 still scores) and then
        # fails turning that NaN into one of its error codes
        if 'NaN' not in str(e):
            raise
        level = "digital silence throughout" if not np.any(deg) else "close to digital silence"
        return nan, f"deg is {level}, and the pesq package gets NaN for it"
    if not math.isfinite(lqo):
        return nan, "PESQ came back nan for this pair"
    if scale == 'lqo':
        return lqo, None
    if not 0.999 < lqo < 4.999:
        return nan, (f"its MOS-LQO, {lqo:g}, is outside what P.862.1 maps to, so it "
                     f"has no raw score")
    # the raw score is 4.5 less the two disturbances, so never above 4.5; the
    # float32 MOS-LQO the package returns can map back to a hair over it
    return min(4.5, (4.6607 - math.log(4.0 / (lqo - 0.999) - 1.0)) / 1.4945), None


# --- every metric by name -----------------------------------------------------
#: the names :func:`score` takes. snr, si_sdr, segsnr, sdi and lsd need
#: nothing installed, and neither do llr, is, cep, wss, fwsegsnr and ncm,
#: which are defined at 8 and 16 kHz only, csii_high, csii_mid and
#: csii_low, which need 8 kHz or more, or siib and siib_gauss, which take
#: any whole rate and need 5.45 s of speech; pesq, stoi, estoi and sdr need
#: an optional package, and csig, cbak and covl need pesq.
METRICS = ('snr', 'si_sdr', 'segsnr', 'sdi', 'pesq', 'stoi', 'estoi', 'sdr',
           'llr', 'is', 'cep', 'wss', 'fwsegsnr', 'lsd', 'csig', 'cbak', 'covl',
           'ncm', 'csii_high', 'csii_mid', 'csii_low', 'siib', 'siib_gauss')

#: what :func:`score` runs when it is not given a list, where each can run
_DEFAULT = ('snr', 'si_sdr', 'segsnr', 'sdi', 'pesq', 'stoi', 'estoi', 'sdr')

#: optional metric -> (module it needs, the extra that installs it)
_BACKEND = {'pesq': ('pesq', 'pesq'), 'stoi': ('pystoi', 'eval'),
            'estoi': ('pystoi', 'eval'), 'sdr': ('mir_eval', 'eval'),
            'csig': ('pesq', 'pesq'), 'cbak': ('pesq', 'pesq'),
            'covl': ('pesq', 'pesq')}

#: score()'s names for the functions of kudio.core.quality
_QUALITY = {'llr': 'llr', 'is': 'itakura_saito', 'cep': 'cepstral_distance',
            'wss': 'wss', 'fwsegsnr': 'fw_segmental_snr'}

#: the three ratings one call of kudio.composite gives
_COMPOSITE = ('csig', 'cbak', 'covl')

#: score()'s names for the three levels one call of kudio.csii gives
_CSII = {'csii_high': 'high', 'csii_mid': 'mid', 'csii_low': 'low'}


#: what pystoi returns instead of a score when a clip is too short for it
_STOI_TOO_SHORT = 1e-5 + 1e-12


def _compute(name: str, ref: np.ndarray, deg: np.ndarray, sr: int,
             pesq_mode: str = 'auto', pesq_scale: str = 'lqo',
             whole: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> float:
    """One metric on two already-aligned signals. PESQ, and the composite
    for the PESQ inside it, take *whole*, the pair before it was cut, where
    the caller has it."""
    whole = (ref, deg) if whole is None else whole
    if name == 'snr':
        return snr(ref, deg)
    if name == 'si_sdr':
        return si_sdr(ref, deg)
    if name == 'segsnr':
        return segmental_snr(ref, deg)
    if name == 'sdi':
        return sdi(ref, deg)
    if name == 'lsd':
        return lsd(ref, deg, sr)
    if name in _QUALITY or name in _COMPOSITE:
        # quality imports this module, so it cannot be imported at the top
        from kudio.core import quality
        if name in _QUALITY:
            return getattr(quality, _QUALITY[name])(ref, deg, sr)
        return float(getattr(quality.composite(*whole, sr), name))
    if name == 'ncm' or name in _CSII:
        # intelligibility imports this module too
        from kudio.core import intelligibility
        if name == 'ncm':
            return intelligibility.ncm(ref, deg, sr)
        return float(getattr(intelligibility.csii(ref, deg, sr), _CSII[name]))
    if name in ('siib', 'siib_gauss'):
        # siib imports this module too; kudio.core binds the name siib to
        # the function, so the module is reached by its full path
        from kudio.core.siib import siib, siib_gauss
        return siib(ref, deg, sr) if name == 'siib' else siib_gauss(ref, deg, sr)
    if name == 'pesq':
        return pesq(*whole, sr, mode=pesq_mode, scale=pesq_scale)
    if name in ('stoi', 'estoi'):
        too_short = ("the pair needs about 0.4 s of speech left once its silent "
                     "frames are removed, and has less")
        # under 25.6 ms, one 256-sample frame at pystoi's 10 kHz, pystoi
        # raises numpy's AxisError instead of answering
        if len(ref) * 10000 < 256 * sr:
            return scored_nan(name, too_short)
        if _overflows(ref, deg):
            return _too_loud(name, ref, deg)
        # pystoi adds noise from numpy's global generator before normalising,
        # so on digital silence its answer would depend on the draw
        for label, y in (('ref', ref), ('deg', deg)):
            if not np.any(y):
                return scored_nan(name, f"{label} is digital silence throughout, so "
                                        f"there is no envelope to correlate")
        from pystoi import stoi
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            value = float(stoi(ref, deg, sr, extended=(name == 'estoi')))
        if value <= _STOI_TOO_SHORT:
            # pystoi's answer when under 30 frames survive its silence removal
            # (about 0.4 s): a placeholder, not a score, and averaged into a
            # mean it reads as the worst intelligibility there is
            return scored_nan(name, too_short)
        return value
    if name == 'sdr':
        if len(ref) == 0:
            return scored_nan(name, "the pair is empty")
        for label, y in (('ref', ref), ('deg', deg)):
            if not np.any(y):
                return scored_nan(name, f"{label} is digital silence throughout, which "
                                        f"mir_eval's bss_eval_sources refuses to score")
        if _overflows(ref, deg):
            return _too_loud(name, ref, deg)
        import mir_eval
        with warnings.catch_warnings():
            # mir_eval 0.8 warns that bss_eval_sources is going; [eval] pins
            # mir_eval below 0.9, so there is nothing for a caller to change
            warnings.simplefilter("ignore", FutureWarning)
            value, *_ = mir_eval.separation.bss_eval_sources(
                ref.reshape(1, -1), deg.reshape(1, -1))
        value = float(value[0])
        if not math.isfinite(value):
            # a one-sample pair comes back inf
            return scored_nan(name, f"bss_eval_sources gave {value} for this pair of "
                                    f"{len(ref)} sample(s)")
        return value
    raise ValueError(f"score(): unknown metric {name!r}; choose from {METRICS}")


def score(ref: np.ndarray, deg: np.ndarray, sr: int,
          metrics: Optional[Sequence[str]] = None, *,
          pesq_mode: str = 'auto', pesq_scale: str = 'lqo') -> Dict[str, float]:
    """Score *deg* against *ref*: ``{metric: value}``.

    >>> kudio.score(clean, enhanced, 16000, ['pesq', 'stoi', 'sdi'],
    ...             pesq_mode='nb', pesq_scale='raw')
    {'pesq': 2.41, 'stoi': 0.874, 'sdi': 0.105}

    :param metrics: names from :data:`METRICS`. ``None`` means snr, si_sdr,
        segsnr and sdi, and each of pesq, stoi, estoi and sdr whose package is
        installed, PESQ only at 8 or 16 kHz. The others are run when they are
        named: lsd (:func:`lsd` with URGENT 2025's settings), llr, is
        (Itakura-Saito), cep, wss and fwsegsnr (the functions of
        :mod:`kudio.core.quality`, 8 and 16 kHz only), and csig, cbak and
        covl (:func:`kudio.composite` with its defaults, so the PESQ inside
        them is wideband MOS-LQO at 16 kHz and raw narrowband at 8 kHz
        whatever *pesq_scale* says; the three come from one call), ncm
        (:func:`kudio.ncm`, 8 and 16 kHz only), csii_high, csii_mid and
        csii_low (the three levels of :func:`kudio.csii`, from one call),
        and siib and siib_gauss (:func:`kudio.siib` and
        :func:`kudio.siib_gauss`, in bits per second, nan with a warning for
        a pair with less than 5.45 s of speech). A metric named explicitly
        that cannot run raises instead of going missing.
    :param pesq_mode, pesq_scale: as *mode* / *scale* of :func:`pesq`, for
        the pesq entry.

    The two signals are compared sample for sample over their common length,
    as the rest of kudio's reference metrics are -- see :func:`kudio.align`
    for a pair that may not line up. A length difference is warned about, and
    refused past :data:`MAX_LENGTH_MISMATCH` of the longer signal. PESQ, and
    the PESQ inside csig, cbak and covl, get the two whole, as :func:`pesq`
    does, so a call for ``['pesq']`` alone is not checked for length. A pair
    with NaN or inf in it is refused, and an empty one gives nan for every
    metric with one warning; past that, each metric gives nan and a warning
    of its own for what it cannot score.
    """
    sr = sample_rate(sr, 'score')
    if metrics is None:
        names = [m for m in _DEFAULT
                 if (m not in _BACKEND or _has_module(_BACKEND[m][0]))
                 and (m != 'pesq' or sr in (8000, 16000))]
    else:
        names = [metrics] if isinstance(metrics, str) else list(metrics)
        unknown = [m for m in names if m not in METRICS]
        if unknown:
            raise ValueError(f"score(): unknown metric(s) {unknown}; choose from "
                             f"{METRICS}")
        for m in names:
            if m in _BACKEND and not _has_module(_BACKEND[m][0]):
                raise DependencyError(*_BACKEND[m])
    return _score_pair(ref, deg, sr, names, pesq_mode, pesq_scale)


#: warnings about the libraries rather than the pair, which _recorded drops
_NOT_ABOUT_THE_PAIR = (DeprecationWarning, PendingDeprecationWarning, FutureWarning)


def _recorded(compute: Callable[[], Any]) -> Tuple[Any, List[str], Optional[str]]:
    """``compute()``, the warnings it gave as text, and the error that stopped
    it, or None: what :func:`kudio.score_dataset` keeps with an item, so that
    one item that cannot be scored does not stop the rest."""
    from kudio.exceptions import KudioError
    value, error = None, None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            value = compute()
        except Exception as e:                          # noqa: BLE001 -- kept as the reason
            error = str(e) if isinstance(e, (ValueError, KudioError, OSError)) \
                else f"{type(e).__name__}: {e}"
    said = [str(w.message) for w in caught
            if not issubclass(w.category, _NOT_ABOUT_THE_PAIR)]
    return value, said, error


def _score_pair(ref: np.ndarray, deg: np.ndarray, sr, names: Sequence[str],
                pesq_mode: str = 'auto', pesq_scale: str = 'lqo', *,
                caller: str = 'score', labels: Tuple[str, str] = ('ref', 'deg'),
                tolerance: Optional[float] = None, recorded: bool = False) -> Dict[str, Any]:
    """The metrics *names*, already resolved, of one pair: the way
    :func:`score` runs them, written down once, since
    :func:`kudio.score_dataset` runs them the same way.

    The pair is checked, then cut to the length the two share, unless PESQ
    alone is asked for, which gets the pair whole as :func:`pesq` does. An
    empty pair is said once for every metric, and the composite and CSII are
    computed once for their three names each. *caller* and *labels* go into
    the messages, *tolerance* replaces :data:`MAX_LENGTH_MISMATCH`.

    With *recorded*, each metric runs under :func:`_recorded` and its entry is
    ``(value, warnings, error)``; the composite's and CSII's go with each of
    their three names. A problem with the pair itself, NaN samples or a length
    past the limit, is raised either way, and its warnings are given.
    """
    whole = _sides(ref, deg, names=labels, caller=caller)
    if set(names) == {'pesq'}:
        ref, deg = whole
    else:
        ref, deg = _cut(*whole, names=labels, tolerance=tolerance, caller=caller,
                        whole_for='PESQ' if any(m == 'pesq' or m in _COMPOSITE
                                                for m in names) else None)
    if ref.size == 0 and deg.size == 0:
        # one warning for the pair, rather than one from every metric on it
        if recorded:
            _, said, _ = _recorded(lambda: scored_nan(caller, "the pair is empty"))
            return {m: (float('nan'), said, None) for m in names}
        scored_nan(caller, "the pair is empty")
        return {m: float('nan') for m in names}
    run = _recorded if recorded else (lambda compute: compute())
    scores: Dict[str, Any] = {}
    shared: Dict[str, Any] = {}
    # the pair has been called faint above if it is; not again by each metric
    with checked():
        for m in names:
            if m in _COMPOSITE or m in _CSII:
                if m in _COMPOSITE and 'composite' not in shared:
                    from kudio.core.quality import composite
                    shared['composite'] = run(lambda: composite(*whole, sr))
                elif m in _CSII and 'csii' not in shared:
                    from kudio.core.intelligibility import csii
                    shared['csii'] = run(lambda: csii(ref, deg, sr))
                field = m if m in _COMPOSITE else _CSII[m]
                result = shared['composite' if m in _COMPOSITE else 'csii']
                if recorded:
                    got, said, error = result
                    scores[m] = (None if error else float(getattr(got, field)), said, error)
                else:
                    scores[m] = float(getattr(result, field))
            else:
                scores[m] = run(lambda: _compute(m, ref, deg, sr, pesq_mode, pesq_scale,
                                                 whole))
    return scores


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
            (pesq_ok, "PESQ", 'pip install "kudio[pesq]" (builds from source: '
                              "needs a C compiler -- on Windows, the MSVC "
                              "Build Tools)"),
            (stoi_ok, "STOI", 'pip install "kudio[eval]"'),
            (sdr_ok, "SDR", 'pip install "kudio[eval]"')):
        if not ok and what not in _REPORTED:
            _REPORTED.add(what)
            log.warning("%s unavailable: %s", what, hint)
    return pesq_ok, stoi_ok, sdr_ok


# --- whole lists of files -----------------------------------------------------
#: AudioEvaluate's column -> score()'s metric name
_COLUMNS = {'PESQ': 'pesq', 'STOI': 'stoi', 'SDR': 'sdr'}


class AudioEvaluate:
    """Evaluate enhanced audio against clean references.

    >>> audio_eval = AudioEvaluate(clean_list, noisy_list)
    >>> audio_eval.eval('ddae', enhanced_list)
    >>> df = audio_eval.get_df()       # pandas.DataFrame, with a Baseline row
    >>> per = audio_eval.per_file()    # one row per method x file

    File lists must be aligned: ``clean_list[i]`` / ``noisy_list[i]`` /
    ``enhanced_list[i]`` refer to the same utterance. Files are read with
    :func:`kudio.file_load` -- float, mono, and any format soundfile reads,
    NIST SPHERE included.

    :param pesq_mode, pesq_scale: as in :func:`kudio.pesq`. The defaults give
        MOS-LQO, wideband at 16 kHz and narrowband at 8 kHz;
        ``pesq_mode='nb', pesq_scale='raw'`` is the raw P.862 score.

    A file that cannot be scored -- unreadable, at another rate, too far from
    its reference in length, with NaN or inf in it -- gets nan in every
    column and a warning naming it, and the rest of the list is still scored.
    PESQ gets each pair whole, as :func:`kudio.pesq` does, so
    :meth:`pesq_score` does not check the lengths at all.
    A file one metric has nothing to score in, such as a silent output for
    PESQ, gets nan in that column; after each pass one warning per column
    names those files and why, and the means leave them out on both sides.

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

    def get_df(self) -> pd.DataFrame:
        """Return the collected scores (plus a Baseline row) as a DataFrame.

        Before anything has been scored the DataFrame is empty, with the
        columns it will have, rather than a list.
        """
        if not self.score_dict['Enh_method']:
            return _pandas().DataFrame(columns=list(self.score_dict))
        table = {key: list(values) for key, values in self.score_dict.items()}
        table['Enh_method'].append('Baseline')
        if self.base_pesq is not None:
            table['PESQ'].append(np.around(self.base_pesq, 4))
        if self.base_stoi is not None:
            table['STOI'].append(np.around(self.base_stoi, 4))
        if self.base_sdr is not None:
            table['SDR'].append(np.around(self.base_sdr, 4))
        return _pandas().DataFrame.from_dict(table, orient='columns')

    def per_file(self) -> pd.DataFrame:
        """Every file's scores as a DataFrame, one row per method and file.

        The 'Baseline' rows score the noisy files; :attr:`rows` holds the
        method rows as plain dicts, for use without pandas.
        """
        return _pandas().DataFrame(list(self._baseline or []) + self.rows)

    def _check_length(self, files: Sequence) -> None:
        if len(files) != len(self.clean_list):
            raise ValueError(f"{len(files)} files to score against "
                             f"{len(self.clean_list)} clean references")

    def _score_rows(self, method: str, files: Sequence, columns: Sequence[str],
                    caller: str = 'eval') -> List[dict]:
        from kudio.core.io import file_load
        rows = []
        #: column -> (file, why) for every file that column has no score for
        missed: Dict[str, List[Tuple[str, str]]] = {column: [] for column in columns}
        for cln, noy, deg in tqdm(zip(self.clean_list, self.noisy_list, files),
                                  total=len(self.clean_list),
                                  desc=f"[kudio] Scoring {method}"):
            row = {'method': method, 'clean': str(cln), 'noisy': str(noy),
                   'file': str(deg)}
            try:
                ref, sr = file_load(cln)
                out, sr_deg = file_load(deg)
                if sr_deg != sr:
                    raise ValueError(f"{deg} is at {sr_deg} Hz and its "
                                     f"reference {cln} at {sr} Hz; resample "
                                     f"one of them")
                whole = _sides(ref, out, names=(str(cln), str(deg)))
                if caller == 'pesq_score':
                    ref, out = whole
                else:
                    ref, out = _cut(*whole, names=(str(cln), str(deg)),
                                    whole_for='PESQ' if 'PESQ' in columns else None)
            except Exception as e:                          # noqa: BLE001
                # one bad file used to abort a run after every file before it
                # had been scored
                warn(f"AudioEvaluate.{caller}(): {deg} was not scored: {e}")
                rows.append({**row, **{column: float('nan') for column in columns}})
                continue
            for column in columns:
                # held back and said once per column below: one warning per
                # file would not even say which file it was about
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    failed = None
                    try:
                        with checked():
                            row[column] = _compute(_COLUMNS[column], ref, out, sr,
                                                   self.pesq_mode, self.pesq_scale,
                                                   whole)
                    except Exception as e:                  # noqa: BLE001
                        row[column], failed = float('nan'), str(e)
                said = [str(w.message) for w in caught]
                if np.isfinite(row[column]):
                    for message in said:
                        warn(message)
                else:
                    missed[column].append((str(deg), failed or _reason(said)))
            rows.append(row)
        for column, files_missed in missed.items():
            if files_missed:
                warn(_missed(caller, method, column, files_missed, len(rows)))
        return rows

    def _means(self, column: str) -> Tuple[float, float]:
        if self.enhanced_list is None:
            raise ValueError("no enhanced files yet: call eval(method, files) first")
        self._check_length(self.enhanced_list)
        caller = f"{column.lower()}_score"
        base = self._score_rows('Baseline', self.noisy_list, [column], caller)
        enhanced = self._score_rows('enhanced', self.enhanced_list, [column], caller)
        return _paired_means(base, enhanced, column)

    def sdr_score(self) -> Tuple[float, float]:
        """``(baseline, enhanced)`` mean SDR over :attr:`enhanced_list`."""
        log.info("==== Evaluate SDR ====")
        return self._means('SDR')

    def stoi_score(self) -> Tuple[float, float]:
        """``(baseline, enhanced)`` mean STOI over :attr:`enhanced_list`."""
        log.info("==== Evaluate STOI ====")
        return self._means('STOI')

    def pesq_score(self) -> Tuple[float, float]:
        """``(baseline, enhanced)`` mean PESQ over :attr:`enhanced_list`."""
        log.info("==== Evaluate PESQ ====")
        return self._means('PESQ')

    def pesq_score_exe(self, pesq_exe: str = 'pesq.exe', rate: int = 16000
                       ) -> Tuple[float, float]:
        """PESQ from the ITU program, ``pesq.exe`` by default, on PATH or
        given as a path.

        The program runs in narrowband mode, and the score taken is the raw
        P.862 one, the same scale as ``pesq_mode='nb', pesq_scale='raw'``:
        the first number after the last ``=`` it prints. Version 2.0 of the
        ITU code (October 2005) prints the MOS-LQO after it on that line. A
        file whose baseline or output gets no score drops out of both means,
        and one warning names those runs with the last line each printed.
        The program also appends every score to ``pesq_results.txt`` in the
        working directory.

        :param rate: 8000 or 16000, the two rates the program runs at.
        """
        log.info("==== Evaluate PESQ (pesq.exe) ====")
        if self.enhanced_list is None:
            raise ValueError("no enhanced files yet: call eval(method, files) first")
        if isinstance(rate, bool) or rate not in (8000, 16000):
            # any other rate makes the program print its usage text, and
            # every file would score nan
            raise ValueError(f"AudioEvaluate.pesq_score_exe(): the ITU program "
                             f"runs at 8000 or 16000 Hz, not {rate!r}")
        if shutil.which(pesq_exe) is None:
            # a wrong path is a mistake in the call, not a file that scored nan
            raise FileNotFoundError(f"pesq binary not found: {pesq_exe!r}")

        def run(clean, other) -> Tuple[float, str]:
            # an argument list, not a shell line: a file name with a quote or
            # a space in it is passed as it is
            out = subprocess.run([pesq_exe, f'+{int(rate)}', str(clean), str(other)],
                                 capture_output=True, text=True).stdout
            lines = out.strip().splitlines()
            last = lines[-1].strip() if lines else ''
            # "... = <raw>\t<MOS-LQO>" from version 2.0; a last line with no
            # '=' is an error report or the usage text, not a score, even
            # where it starts with a number
            _, equals, after = last.rpartition('=')
            try:
                return (float(after.split()[0]) if equals else float('nan')), last
            except (IndexError, ValueError):     # no score printed for this pair
                return float('nan'), last

        rows, missed = [], []
        for cln, noy, enh in tqdm(zip(self.clean_list, self.noisy_list,
                                      self.enhanced_list),
                                  total=len(self.clean_list)):
            row = {'wave': os.path.basename(str(cln))}
            for column, other in (('Baseline-PESQ', noy), ('Enhance-PESQ', enh)):
                row[column], last = run(cln, other)
                if math.isnan(row[column]):
                    missed.append(f"{other} ({last or 'nothing printed'})")
            rows.append(row)
        if missed:
            named = "; ".join(missed[:_NAMED])
            if len(missed) > _NAMED:
                named += f"; and {len(missed) - _NAMED} more"
            warn(f"AudioEvaluate.pesq_score_exe(): {pesq_exe} gave no score in "
                 f"{len(missed)} of {2 * len(rows)} runs, and a file with one of "
                 f"them drops out of both means: {named}")
        mean = _pandas().DataFrame(rows).dropna().mean(numeric_only=True)
        log.info("pesq_mean:\n%s", mean)
        return mean['Baseline-PESQ'], mean['Enhance-PESQ']


#: how many files the warning of a pass names before it points at per_file()
_NAMED = 5


def _reason(said: List[str]) -> str:
    """Why a metric gave nan, from the warning it gave: the text between
    ``<function>(): `` and ``; scored nan``."""
    for message in said:
        if message.endswith("; scored nan"):
            text = message[:-len("; scored nan")]
            return text.split("(): ", 1)[1] if "(): " in text else text
    return "; ".join(said) or "no reason given"


def _missed(caller: str, method: str, column: str, files: List[Tuple[str, str]],
            total: int) -> str:
    """The one warning of a pass for the files *column* has no score for."""
    named = "; ".join(f"{name} ({why})" for name, why in files[:_NAMED])
    if len(files) > _NAMED:
        named += f"; and {len(files) - _NAMED} more, in per_file()"
    if method == 'Baseline':
        out_of = f"the {column} means of the baseline and of every method"
    else:
        out_of = f"{method!r}'s {column} mean and the baseline's beside it"
    return (f"AudioEvaluate.{caller}(): {column} has no score for {len(files)} of "
            f"{total} files, which drop out of {out_of}: {named}")


def _paired_means(base: List[dict], enhanced: List[dict], column: str
                  ) -> Tuple[float, float]:
    """Mean of *column* over the files where both sides have a value -- a file
    that failed on one side is left out of both, so the two stay comparable."""
    pairs = [(b[column], e[column]) for b, e in zip(base, enhanced)
             if np.isfinite(b[column]) and np.isfinite(e[column])]
    if not pairs:
        return float('nan'), float('nan')
    b, e = zip(*pairs)
    return float(np.mean(b)), float(np.mean(e))


def eval_metrics(rate: int, ref: np.ndarray, deg: np.ndarray, *,
                 pesq_mode: str = 'auto', pesq_scale: str = 'lqo'
                 ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Score a single (reference, degraded) pair with every available metric.

    Returns ``(pesq, stoi, sdr)`` -- always three values, whatever is
    installed. A metric comes back as ``None`` when its back-end is not
    installed, when PESQ is off 8 and 16 kHz, when the metric has nothing to
    score in the pair, which warns as :func:`score` does, or when its
    back-end fails on the pair, which is logged. A pair with NaN or inf in it
    is refused. STOI and SDR compare the pair over the length the two share,
    with a warning, and a pair past :data:`MAX_LENGTH_MISMATCH` is refused,
    PESQ and all; PESQ gets the two whole, as :func:`pesq` does.

    :param pesq_mode: ``'nb'``, ``'wb'`` or ``'auto'`` (by sample rate).
    :param pesq_scale: ``'lqo'`` for MOS-LQO, or ``'raw'`` for the narrowband
        P.862 score. The defaults give MOS-LQO, wideband at 16 kHz and
        narrowband at 8 kHz.
    """
    rate = sample_rate(rate, 'eval_metrics')
    pesq_status, stoi_status, sdr_status = check_metrics_install()
    wanted = [('pesq', pesq_status and rate in (8000, 16000)),
              ('stoi', stoi_status), ('sdr', sdr_status)]
    runs = [name for name, ok in wanted if ok]
    whole = _sides(ref, deg, caller='eval_metrics')
    ref, deg = _cut(*whole, caller='eval_metrics',
                    whole_for='PESQ' if 'pesq' in runs else None)
    values = []
    for name, ok in wanted:
        value = None
        if ok:
            try:
                with checked():
                    value = _compute(name, ref, deg, rate, pesq_mode=pesq_mode,
                                     pesq_scale=pesq_scale, whole=whole)
                if value is not None and np.isnan(value):
                    value = None            # nothing to score, and it has said why
            except Exception as e:                          # noqa: BLE001
                log.warning("%s could not be computed: %s", name.upper(), e)
        values.append(value)
    return tuple(values)


AudioEvaluate.eval_metrics = staticmethod(eval_metrics)
