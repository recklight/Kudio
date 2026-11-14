# -*- coding: utf-8 -*-
"""SRMR: how reverberant a recording of speech is, judged from the speech alone.

The speech-to-reverberation modulation energy ratio (Falk, Zheng and Chan,
IEEE Trans. Audio, Speech, Lang. Process. 18(7):1766-1774, 2010) needs no
clean reference. It splits the
speech into 23 auditory bands, follows the envelope of each, and measures how
fast those envelopes move. Syllables move them a few times a second. A room
keeps every band ringing after the syllable has stopped, which fills the dips
between syllables and adds fast fluctuation of its own, so reverberation adds
envelope energy in the fast modulation bands while the slow ones change little
(about 0.1 dB in Falk et al.'s 4 Hz band, their sec. III-B).
SRMR is the energy in the four slowest of eight modulation bands, centred from
4 to 18 Hz, over the energy in the bands above them, up to a limit the
speech's bandwidth sets (K* in :func:`srmr`). Higher means drier.

:func:`srmr` follows the authors' MATLAB toolbox, SRMRToolbox, step for step,
and takes its defaults from it. On the toolbox's own test recording it gives
the four ratios the toolbox stores for that file (6.1168, 5.9412, 3.5254 and
2.3004 for the full, fast, fast normalised and normalised measures) to within
1e-12 of each, relative (tools/validation/srmr.py --toolbox-test reruns it
on the toolbox's files). It parts from the toolbox in three places. A rate
other than 8 or 16 kHz is refused, where the toolbox warns and scores it
anyway. A clip shorter than one 256 ms window once its silence is cut scores
``nan`` with a warning, where the toolbox scores it from windows that are all
partly zero padding. And a single long silence inside a clip is cut, where
the toolbox's ``vad_s.m`` keeps it: its branch for one silence repeats
``idx2(1)`` where ``idx2(2)`` belongs.

What to watch for:

* It is a ratio, not an opinion score. It has no fixed range, and a number
  means something only next to another number for the same speech: a dry
  take against the same take in a room is a fair comparison, two talkers are
  not. The envelope also beats at the voice's pitch, and a low voice puts
  that beating into the two fastest bands.
* Noise lowers it as well. A noise floor fluctuates fast, so a noisy dry
  recording reads like a reverberant one. ``norm=True`` is the variant of
  Santos, Senoussaoui and Falk (IWAENC 2014, pp. 55-59) meant for noise and
  reverberation together: it holds the modulation energies within 30 dB of a
  peak and stops the bands below the pitch, at 30 Hz as SRMR.m does; the
  paper reports its numbers with the bands stopping at 40 Hz.
* The silence cut goes by level alone. It takes off everything before the
  first and after the last sample within 50 dB of the peak, and every pause
  longer than 50 ms with nothing in it that close. A room's tail fills the
  pauses, and so does a noise floor 40 dB down, so a reverberant or noisy
  take keeps what a clean, synthetic or gated take of the same speech loses
  at digital zero, and the two are scored over different stretches of it.
  At 16 kHz, on a 3 s synthetic voice with pauses of 40 to 200 ms at digital
  zero, the cut took 1.3 s out and lowered the score by 8%; in a 0.3 s room
  it took 0.1 s out, and the score moved by 0.4%. An enhancer that gates
  its output to zero between words, scored against its noisy input, runs
  into the same thing. ``drop_silence=False`` scores every take whole.
* A DC offset gets in the way of the cut. The cut looks for samples far below
  the peak, and an offset holds every sample near it: on the 3 s voice above,
  an offset of 1% of the peak keeps every pause, which undoes the 8% the cut
  took off. Take DC out first (a high-pass well under 125 Hz) if the
  recording chain can leave one.
* The rate is part of the measure. The 23 bands run from 125 Hz to just
  under half the sampling rate, so one recording scores differently at 8 and at 16 kHz, and
  those are the two rates the toolbox's authors validated it at. Other rates
  are refused rather than scored.
* SRMRpy, the Python port, starts its 256 ms windows at the first sample
  rather than 192 ms before it, its fast mode cuts the short-time spectrum
  differently, and with ``norm=True`` it keeps the last modulation band at
  128 Hz where SRMR.m moves it to 30 Hz, so it does not reproduce the
  toolbox. On the toolbox's test recording it gives 6.055 for the full
  measure where the toolbox stores 6.117, 6.737 for the fast one where the
  toolbox stores 5.941, and 3.680 for the normalised one where the toolbox
  stores 2.300. Say which implementation a published number came from
  before comparing with it.

Sources: Falk, Zheng and Chan (2010), doi:10.1109/TASL.2010.2052247; Santos,
Senoussaoui and Falk (2014), doi:10.1109/IWAENC.2014.6953337; Senoussaoui,
Santos and Falk (2015), arXiv:1510.04707; Slaney, "An efficient
implementation of the Patterson-Holdsworth auditory filter bank", Apple
Computer Technical Report #35 (1993); Glasberg and Moore, Hearing Research
47:103-138 (1990); Ellis, "Gammatone-like spectrograms", web resource (2009),
www.ee.columbia.edu/~dpwe/resources/matlab/gammatonegram/. The ``.m``
files named below are SRMRToolbox's (github.com/MuSAELab/SRMRToolbox).
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Optional, Tuple

import numpy as np

from kudio.core._policy import faint, sample_rate, scored_nan, signal

__all__ = ['srmr']

#: The two rates the toolbox's authors validated SRMR at (toolbox README).
_RATES = (8000, 16000)

# -- the auditory filterbank -------------------------------------------------
# SRMR.m's fixed parameters: 23 gammatone channels (23 in Senoussaoui et al.
# 2015, sec. 2, as well), the lowest centred on 125 Hz and the highest one
# step of their spacing under half the rate.
_CHANNELS = 23
_LOW_HZ = 125.0

# Glasberg and Moore (1990): ERB(f) = 24.7 * (4.37 f / 1000 + 1) Hz, written as
# f / EarQ + minBW the way Slaney's report has it (Slaney 1993, sec. 2.1), so
# EarQ = 1000 / (24.7 * 4.37).
_EAR_Q = 9.26449
_MIN_BW = 24.7

# A fourth-order gammatone is one ERB wide when its bandwidth parameter is
# 1.019 ERB (Slaney 1993, sec. 3.0, after Patterson).
_B_PER_ERB = 1.019

# -- the modulation filterbank -----------------------------------------------
# SRMR.m: eight band-pass filters with Q = 2, centred from minCF = 4 Hz to
# maxCF = 128 Hz a constant ratio apart (computeModulationCFs.m). The
# normalised variant stops at 30 Hz in SRMR.m; Santos et al. (2014, sec. 2.3)
# tried 30 to 100 Hz and report 40 Hz (sec. 4).
_BANDS = 8
_Q = 2.0
_MIN_CF = 4.0
_MAX_CF = 128.0
_MAX_CF_NORM = 30.0

# The ratio puts the first four bands over the fifth to the K*-th (SRMR.m;
# eq. 3 of Senoussaoui, Santos and Falk 2015 writes it with K* = 8).
_SLOW = 4

# SRMR.m frames each band's output in 256 ms Hamming windows every 64 ms.
# Falk et al. (2010, sec. III-A) shift theirs by 32 ms and take a DFT of each
# window, grouping the bins into the eight bands, and Senoussaoui et al.
# (2015, sec. 2) do the same; the toolbox's filters and 64 ms are what its
# stored ratios were computed with.
_WINDOW_S = 0.256
_HOP_S = 0.064

# K* is set by the channel below which 90% of the modulation energy lies
# (SRMR.m, K90perc).
_SHARE = 90.0

# The normalised variant holds every energy within 30 dB of the peak of the
# channel-averaged energies: the toolbox's peak_energy * 0.001 (SRMR.m), the
# "30dB of the peak average energy" of Senoussaoui, Santos and Falk (ACE
# Challenge workshop 2015, arXiv:1510.04707, sec. 2.4).
_NORM_RANGE = 1e-3

# The fast front end: gammatonegram(s, fs, 0.010, 0.0025, ...) in SRMR.m,
# 10 ms windows every 2.5 ms, which puts the envelopes at 400 Hz, the
# modulation rate SRMR.m sets for this mode.
_FAST_WINDOW_S = 0.010
_FAST_HOP_S = 0.0025
_FAST_RATE = 400.0

# The toolbox scores speech with its silence taken out (preprocess.m,
# vad_s.m): a sample counts as silent when its power is more than 50 dB under
# the peak sample's, and a silent stretch longer than 50 ms is cut.
_SILENCE_DB = 50.0
_SILENCE_GAP_S = 0.05


def srmr(y: np.ndarray, sr: int, *, fast: bool = False, norm: bool = False,
         min_cf: float = _MIN_CF, max_cf: Optional[float] = None,
         drop_silence: bool = True) -> float:
    """Speech-to-reverberation modulation energy ratio of *y*. Higher is drier.

    >>> y, sr = kudio.file_load("SRMRToolbox/test/test.wav")  # doctest: +SKIP
    >>> round(kudio.srmr(y, sr), 4)                           # doctest: +SKIP
    6.1168
    >>> round(kudio.srmr(y, sr, norm=True), 4)                # doctest: +SKIP
    2.3004

    How it is measured, as the toolbox measures it:

    1. The silence is cut out (*drop_silence*).
    2. 23 gammatone filters, Slaney's four-section design, split the speech.
       Their centres sit at equal steps of ``ln(f + 228.8)``, the ERB-rate
       scale of Glasberg and Moore, from 125 Hz up to one step under half the
       rate (Slaney 1993, sec. 2.2). Each has a bandwidth parameter of
       1.019 ERB, which makes it one ERB wide, and a gain of 1 at its centre.
    3. The envelope of each channel is the magnitude of its analytic signal
       (Hilbert transform over the whole clip).
    4. Eight second-order band-pass filters with Q = 2, centred from *min_cf*
       to *max_cf* a constant ratio apart (4, 6.6, 10.8, 17.7, 29.0, 47.6, 78
       and 128 Hz by default), split each envelope again.
    5. Each of those 184 outputs is cut into 256 ms Hamming windows every
       64 ms, laid out as MATLAB's ``buffer`` lays them out: the first window
       is the first 64 ms behind 192 ms of zeros, and the last is padded with
       zeros. The energy of each window is averaged over the clip.
    6. BW is the ERB of the channel at which the channels, counted up from
       125 Hz, first hold more than 90% of all the modulation energy. K* is
       the last band whose lower edge, its centre minus half its bandwidth
       ``cf / Q``, lies at or below BW, and the score is the energy in bands
       1 to 4 over the energy in bands 5 to K*. With the default bands K* is
       never below 6, because the 125 Hz channel's 38.2 Hz bandwidth already
       passes the sixth band's edge at 35.7 Hz, and it is 8 once the 90%
       point reaches the channel near 700 Hz.

    :param y: mono speech, float or integer; integer audio is read on its
        type's full scale, though the score does not depend on the level.
    :param sr: 8000 or 16000. Resample anything else first:
        :func:`kudio.resample`, or ``scipy.signal.resample_poly(y, 16000,
        sr)`` where librosa is not installed.
    :param fast: build the envelopes from a short-time spectrum instead of
        filtering, as the toolbox's fast mode does: each channel's envelope is
        its filter's magnitude response applied to the magnitude spectrum of
        10 ms Hann windows every 2.5 ms (Ellis's gammatonegram, 2009), so the
        envelopes come out at 400 Hz. About twenty times faster at 16 kHz,
        and a different number: 5.941 against 6.117 on the toolbox's test
        recording. The full measure takes the Hilbert transform of the
        whole clip, 0.4 to 0.8 s per 10 s at 16 kHz on a desktop CPU; it is
        slower when the clip's length after the cut has a large prime
        factor. The modulation filters then run at 400 Hz, where the
        bilinear transform narrows the 128 Hz band to 28 Hz between its -3 dB
        points instead of 64; the toolbox's fast scores are computed that way
        and so are these.
    :param norm: the variant of Santos, Senoussaoui and Falk (2014), as
        SRMR.m computes it: every window's band energy is held between the
        peak of the channel-averaged energies and 30 dB below it before
        averaging, and *max_cf* defaults to 30 Hz. The paper takes the peak
        over channels and bands of the energies averaged over windows (its
        eq. 1), and reports its results with the bands stopping at 40 Hz.
    :param min_cf: centre of the first modulation band, in Hz.
    :param max_cf: centre of the last, in Hz: 128 by default, 30 with *norm*.
    :param drop_silence: cut the silence out first, as the toolbox does: the
        stretch before the first and after the last sample within 50 dB of the
        peak sample, and every stretch in between where two such samples lie
        more than 50 ms apart. Kept, a second of digital silence either side
        of the toolbox's test recording moves its score by 1 to 4%,
        depending on the mode. Pauses that are digital silence close up as
        well, so speech whose pauses were muted to zero is scored as if it
        had been spoken without them. ``False`` scores the clip as it is,
        which is what SRMRpy does.
    :raises ValueError: for a rate other than 8 or 16 kHz, a
        ``(frames, channels)`` array, samples that are not finite, or bands
        the measure cannot use: *min_cf* not below *max_cf*, a band at or
        above half the rate the envelopes are filtered at (200 Hz in the fast
        mode), or a fifth band starting above the 38.2 Hz bandwidth of the
        lowest channel, where some recordings would have nothing left to
        divide by.
    :returns: the ratio. A silent clip scores ``nan`` with a warning. So does
        one shorter than one 256 ms window once its silence is cut, 4096
        samples at 16 kHz: no window then lies wholly inside it. The toolbox
        would still score such a clip, from windows that are all partly zero
        padding.
    """
    sr = sample_rate(sr, 'srmr')
    if sr not in _RATES:
        raise ValueError(f"srmr(): defined at 8 and 16 kHz only, not {sr} Hz: its "
                         f"23 bands run from 125 Hz to half the rate, so at "
                         f"another rate it measures something else. Resample "
                         f"first, with kudio.resample or, without librosa, "
                         f"scipy.signal.resample_poly(y, 16000, {sr})")
    y = signal(y, 'y', 'srmr')
    faint('srmr', [('y', y)])
    rate = _FAST_RATE if fast else float(sr)
    if max_cf is None:
        max_cf = _MAX_CF_NORM if norm else _MAX_CF
    centres, lower = _modulation_bands(float(min_cf), float(max_cf), rate, sr)

    if y.size == 0:
        return scored_nan('srmr', "the clip is empty")
    if not np.any(y):
        return scored_nan('srmr', "the clip is silent, so there is no modulation "
                                  "energy to divide")
    # The ratio does not depend on the level, which is why the toolbox's step
    # to -26 dBov (preprocess.m) is left out. Scaling to the peak costs
    # nothing for the same reason, and keeps a very quiet clip clear of
    # underflow.
    y = y / np.max(np.abs(y))
    if drop_silence:
        y = _drop_silence(y, sr)
    # SRMR.m goes on with any length, buffer() padding its windows with
    # zeros; below one window none of them lies wholly inside the clip, and
    # I would rather say so than return that number
    need = int(math.ceil(_WINDOW_S * sr))
    if y.size < need:
        cut = " once its silence is cut" if drop_silence else ""
        return scored_nan('srmr', f"the clip needs at least one 256 ms window, "
                                  f"{need} samples at {sr} Hz, and has {y.size}{cut}")

    energy = _modulation_energy(y, sr, centres, rate, fast)
    if norm:
        peak = float(np.max(energy.mean(axis=0)))
        energy = np.clip(energy, peak * _NORM_RANGE, peak)
    return _ratio(energy.mean(axis=2), lower, sr)


# -- the auditory filterbank -------------------------------------------------

def _erb(f):
    """Equivalent rectangular bandwidth at *f* Hz (Glasberg and Moore 1990)."""
    return f / _EAR_Q + _MIN_BW


def _channel_centres(sr: int) -> np.ndarray:
    """The 23 centre frequencies, highest first, as the toolbox orders them.

    Equal steps of ``ln(f + EarQ * minBW)`` from just under ``sr / 2`` down to
    exactly 125 Hz; with Slaney's order-1 ERB, the ERB-rate scale is a
    multiple of that logarithm plus a constant (Slaney 1993, sec. 2.2).
    """
    offset = _EAR_Q * _MIN_BW
    step = (math.log(_LOW_HZ + offset) - math.log(sr / 2.0 + offset)) / _CHANNELS
    return (sr / 2.0 + offset) * np.exp(step * np.arange(1, _CHANNELS + 1)) - offset


@lru_cache(maxsize=4)
def _gammatone(sr: int) -> np.ndarray:
    """Slaney's gammatone filterbank at *sr* as ``(channel, section, 6)``
    second-order sections, highest channel first.

    Each channel is four sections with the same pair of poles,
    ``exp((-2 pi b + 2 pi j cf) / sr)`` with ``b = 1.019 ERB(cf)``, and one
    real zero each, at ``exp(-2 pi b / sr) * (cos t + s sin t)`` for
    ``t = 2 pi cf / sr`` and ``s = +/-sqrt(3 + 2**1.5)``,
    ``+/-sqrt(3 - 2**1.5)`` (Slaney 1993, secs. 3.1, 3.3 and 4.1; the two
    roots are 1 + sqrt(2) and sqrt(2) - 1). The first section carries the
    gain that makes the channel's response 1 at its centre (sec. 4.2).

    Kept as four sections, not one eighth-order polynomial: the eight poles
    sit in four identical pairs close to the unit circle, and multiplied out
    they lose accuracy. ``scipy.signal.gammatone`` is the same filter
    multiplied out, and at 16 kHz its 125 Hz channel comes out with a gain
    of 0.9986 at the centre. With its filters in place of these, the score of
    the toolbox's test recording moves by 2e-4 to 4e-4 in either direction
    (6.1155 to 6.1179 against 6.1168), depending on the last bits of the
    centre frequencies.
    """
    cfs = _channel_centres(sr)
    period = 1.0 / sr
    radius = np.exp(-2.0 * np.pi * _B_PER_ERB * _erb(cfs) * period)
    angle = 2.0 * np.pi * cfs * period
    root_hi, root_lo = math.sqrt(3.0 + 2.0 ** 1.5), math.sqrt(3.0 - 2.0 ** 1.5)
    slopes = np.array([root_hi, -root_hi, root_lo, -root_lo])
    zeros = radius[:, None] * (np.cos(angle)[:, None]
                               + slopes[None, :] * np.sin(angle)[:, None])

    sos = np.zeros((_CHANNELS, 4, 6))
    sos[:, :, 0] = period
    sos[:, :, 1] = -period * zeros
    sos[:, :, 3] = 1.0
    sos[:, :, 4] = (-2.0 * radius * np.cos(angle))[:, None]
    sos[:, :, 5] = (radius * radius)[:, None]
    sos[:, 0, :3] /= np.abs(_response(sos, angle[:, None]))
    return sos


def _response(sos: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """Complex response of each channel's sections at *omega* rad/sample;
    *omega* is ``(channels, points)`` or broadcasts to it."""
    z = np.exp(-1j * omega)[:, None, :]
    num = sos[:, :, 0, None] + sos[:, :, 1, None] * z + sos[:, :, 2, None] * z * z
    den = sos[:, :, 3, None] + sos[:, :, 4, None] * z + sos[:, :, 5, None] * z * z
    return np.prod(num / den, axis=1)


# -- the modulation filterbank -----------------------------------------------

def _modulation_bands(min_cf: float, max_cf: float, rate: float,
                      sr: int) -> Tuple[np.ndarray, np.ndarray]:
    """Centre frequencies of the eight bands, and the lower edges the K* rule
    measures them by, after checking the measure can use them."""
    if not (math.isfinite(min_cf) and math.isfinite(max_cf)
            and 0.0 < min_cf < max_cf):
        raise ValueError(f"srmr(): the modulation bands need 0 < min_cf < max_cf, "
                         f"got min_cf={min_cf:g} and max_cf={max_cf:g}")
    if max_cf >= rate / 2.0:
        whose = "the fast mode's envelopes" if rate == _FAST_RATE else "the envelopes"
        raise ValueError(f"srmr(): max_cf={max_cf:g} Hz is not below half the "
                         f"{rate:g} Hz rate of {whose}; keep it under "
                         f"{rate / 2.0:g} Hz")
    centres = min_cf * (max_cf / min_cf) ** (np.arange(_BANDS) / (_BANDS - 1.0))
    # calc_cutoffs.m: the edge is the centre less half the prewarped
    # bandwidth tan(pi cf / sr) / Q, taken back to Hz. That is about 0.75 cf
    # for Q = 2, where the filter's own -3 dB point is nearer 0.78 cf; K* is
    # defined by the toolbox's figure, so that is the one used. It is taken
    # at the audio rate even in the fast mode, as the toolbox takes it.
    lower = centres - np.tan(np.pi * centres / sr) / _Q * sr / (2.0 * np.pi)
    narrowest = float(_erb(_LOW_HZ))
    if lower[_SLOW] > narrowest:
        raise ValueError(
            f"srmr(): min_cf={min_cf:g} and max_cf={max_cf:g} start the fifth "
            f"modulation band at {lower[_SLOW]:.1f} Hz, above the "
            f"{narrowest:.1f} Hz bandwidth of the lowest channel, so a "
            f"recording with its energy that low would leave no band to "
            f"divide by; lower min_cf or max_cf")
    return centres, lower


def _modulation_filter(cf: float, rate: float) -> Tuple[np.ndarray, np.ndarray]:
    """Second-order band-pass at *cf* Hz with Q = 2, for signals at *rate*.

    The resonator ``(W / Q) s / (s^2 + (W / Q) s + W^2)`` through the
    bilinear transform, with ``W = tan(pi cf / rate)`` prewarped so that the
    peak, gain 1, lands on *cf* exactly (makeModulationFilter.m).
    """
    warped = math.tan(math.pi * cf / rate)
    width = warped / _Q
    a0 = 1.0 + width + warped * warped
    b = np.array([width, 0.0, -width]) / a0
    a = np.array([1.0, (2.0 * warped * warped - 2.0) / a0,
                  (1.0 - width + warped * warped) / a0])
    return b, a


def _modulation_windows(rate: float) -> Tuple[int, int, np.ndarray]:
    """Length and hop, in samples at *rate*, of SRMR.m's 256 ms windows every
    64 ms, both rounded up as SRMR.m rounds them (103 and 26 at the fast
    mode's 400 Hz), and the weights :func:`_framed_energy` takes: MATLAB's
    ``hamming``, which is the symmetric window, squared."""
    length = int(math.ceil(_WINDOW_S * rate))
    hop = int(math.ceil(_HOP_S * rate))
    weights = np.hamming(length) ** 2
    return length, hop, weights


def _framed_energy(x: np.ndarray, length: int, hop: int,
                   weights: np.ndarray) -> np.ndarray:
    """Energy of *x* in each window, ``sum((w * frame) ** 2)``, with the
    windows laid out as MATLAB's ``buffer(x, length, length - hop)`` lays
    them out: ``ceil(len(x) / hop)`` of them, the first starting
    ``length - hop`` samples before *x* and the last running past its end,
    both padded with zeros. *weights* is the window squared.
    """
    from kudio.core._framing import frame_view

    lead = length - hop
    count = -(-len(x) // hop)
    padded = np.zeros(lead + count * hop)
    padded[lead:lead + len(x)] = x * x
    return frame_view(padded, length, hop) @ weights


def _modulation_energy(y: np.ndarray, sr: int, centres: np.ndarray,
                       rate: float, fast: bool) -> np.ndarray:
    """Window energies, ``(channel, band, window)``, highest channel first."""
    from scipy.signal import lfilter

    sos = _gammatone(sr)
    length, hop, weights = _modulation_windows(rate)
    filters = [_modulation_filter(cf, rate) for cf in centres]
    envelopes = _spectral_envelopes(y, sr, sos) if fast else None

    energy = []
    for channel in range(_CHANNELS):
        if fast:
            envelope = envelopes[channel]
        else:
            envelope = _hilbert_envelope(_filter_channel(sos[channel], y))
        energy.append([_framed_energy(lfilter(b, a, envelope), length, hop, weights)
                       for b, a in filters])
    return np.array(energy)


def _filter_channel(sections: np.ndarray, y: np.ndarray) -> np.ndarray:
    from scipy.signal import sosfilt
    return sosfilt(sections, y)


def _hilbert_envelope(x: np.ndarray) -> np.ndarray:
    """``abs(hilbert(x))`` over the whole of *x*, with an FFT as long as *x*
    and no longer, which is what MATLAB's ``hilbert`` does."""
    from scipy.fft import ifft, rfft

    n = len(x)
    spectrum = np.zeros(n, dtype=complex)
    spectrum[:n // 2 + 1] = rfft(x)
    spectrum[1:(n + 1) // 2] *= 2.0     # the bins strictly between 0 and n / 2
    return np.abs(ifft(spectrum, overwrite_x=True))


def _spectral_envelopes(y: np.ndarray, sr: int, sos: np.ndarray) -> np.ndarray:
    """The fast mode's envelopes, ``(channel, frame)`` at 400 Hz.

    Ellis's gammatonegram (2009) with the arguments SRMR.m gives it: windows
    of ``round(0.010 * sr)`` samples every ``round(0.0025 * sr)``, an FFT of
    the next power of two at or above twice the window, and each channel's
    envelope the filter's magnitude response times the magnitude spectrum,
    summed over the bins and divided by the FFT length. The frames are those
    of MATLAB's ``specgram`` given a window length: a Hann window that does
    not reach zero at its ends (MATLAB's ``hanning``), the first frame at
    sample 0, and only frames that fit inside the clip.
    """
    length = int(round(_FAST_WINDOW_S * sr))
    hop = int(round(_FAST_HOP_S * sr))
    nfft = 1 << int(math.ceil(math.log2(2 * _FAST_WINDOW_S * sr)))
    window = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(1, length + 1) / (length + 1))
    bins = 2.0 * np.pi * np.arange(nfft // 2 + 1) / nfft
    gains = np.abs(_response(sos, bins[None, :]))

    from kudio.core._framing import frame_view
    frames = frame_view(y, length, hop)
    out = np.empty((_CHANNELS, frames.shape[0]))
    # in blocks: a ten-minute clip is 240000 frames, and its whole spectrum
    # at once would be a gigabyte of complex numbers
    block = 8192
    for start in range(0, frames.shape[0], block):
        part = np.abs(np.fft.rfft(frames[start:start + block] * window, nfft, axis=1))
        out[:, start:start + block] = gains @ part.T
    return out / nfft


def _drop_silence(y: np.ndarray, sr: int) -> np.ndarray:
    """*y* without its silence, as the toolbox cuts it before scoring.

    The toolbox's ``vad_s.m`` keeps the samples from the first to the last
    one whose power is within 50 dB of the peak sample's, and drops every
    stretch in between where two such samples lie more than 50 ms apart.
    When there is exactly one such stretch, ``vad_s.m`` keeps it: its branch
    for that case joins ``x(idx(1):idx2(1))`` to ``x(idx2(1):idx(end))``,
    the whole clip again, where its branch for two or more drops every one.
    This drops it either way, so a clip with a single long silence inside it
    scores differently here than in the toolbox.
    """
    # vad_s.m compares powers; amplitudes against 10 ** (-50 / 20) of the
    # peak draw the same line without squaring a quiet clip into underflow
    level = np.abs(y)
    loud = np.flatnonzero(level > np.max(level) * 10.0 ** (-_SILENCE_DB / 20.0))
    breaks = np.flatnonzero(np.diff(loud) > _SILENCE_GAP_S * sr)
    starts = np.concatenate(([loud[0]], loud[breaks + 1]))
    stops = np.concatenate((loud[breaks], [loud[-1]])) + 1
    if starts.size == 1:
        return y[starts[0]:stops[0]]
    return np.concatenate([y[a:b] for a, b in zip(starts, stops)])


def _ratio(average: np.ndarray, lower: np.ndarray, sr: int) -> float:
    """Bands 1-4 over bands 5-K* of the clip-averaged energies,
    ``(channel, band)`` with the highest channel first."""
    # the toolbox's order of operations, percentages first, so that a clip
    # sitting on the 90% line falls on the same side of it
    share = average.sum(axis=1) * 100.0 / average.sum()
    rising = np.cumsum(share[::-1])
    bandwidth = _erb(_channel_centres(sr)[::-1][int(np.argmax(rising > _SHARE))])
    k_star = _SLOW + int(np.count_nonzero(lower[_SLOW:] <= bandwidth))
    return float(average[:, :_SLOW].sum() / average[:, _SLOW:k_star].sum())
