# -*- coding: utf-8 -*-
"""Two intelligibility measures that compare processed speech with its clean
reference: the normalized covariance metric (NCM) and the coherence speech
intelligibility index (CSII).

Both say, on a scale of 0 to 1, how much of the speech survives the
processing, and both weight their bands by the importance the speech
intelligibility index (SII) standard, ANSI S3.5-1997, gives each band for
short passages of easy reading material. CSII also takes its bands from the
SII's critical bands; NCM spaces its own along the cochlea (NCM's bands,
below). They read the signal differently. :func:`ncm`
follows the envelope of 20 bands and asks how closely the output's envelope
moves with the reference's: in each band the squared correlation ``r^2`` of
the two envelopes becomes an apparent SNR, ``10 log10(r^2 / (1 - r^2))``,
held to +/-15 dB and mapped onto 0..1, and the bands are averaged with the
weights. :func:`csii` keeps the waveform: in each of 16 critical bands up to
3.7 kHz it estimates a signal-to-distortion ratio from the magnitude-squared
coherence of the two signals, and it does so separately over the reference's
loud, middle and quiet frames, which is why it returns three numbers. Ma, Hu
and Loizou set both against the scores of normal-hearing listeners for
noise-suppressed speech in 72 conditions of car, babble, train and street
noise [MHL09].

The covariance method is older than its name. [MHL09] compute it with the
SII's weights, call it NCM and credit it to Holube and Kollmeier (1996);
Goldsworthy and Greenberg, who describe it among the speech-based STI
methods ([GG04] Sec. II.B.4), credit Koch (1992) as well. The coherence SII
and its three levels are Kates and Arehart's [KA05], who go on to combine the
three into one estimate of intelligibility; CSII.m stops at the three, and so
does :func:`csii`. The sources::

    [MHL09] J. Ma, Y. Hu and P. C. Loizou, "Objective measures for predicting
            speech intelligibility in noisy conditions based on new
            band-importance functions", J. Acoust. Soc. Am. 125(5),
            pp. 3387-3405, 2009. doi:10.1121/1.3097493
    [KA05]  J. M. Kates and K. H. Arehart, "Coherence and the speech
            intelligibility index", J. Acoust. Soc. Am. 117(4),
            pp. 2224-2237, 2005. doi:10.1121/1.1862575
    [GG04]  R. L. Goldsworthy and J. E. Greenberg, "Analysis of speech-based
            speech transmission index methods with implications for
            nonlinear operations", J. Acoust. Soc. Am. 116(6), pp. 3679-3689,
            2004. doi:10.1121/1.1804628
    [G90]   D. D. Greenwood, "A cochlear frequency-position function for
            several species--29 years later", J. Acoust. Soc. Am. 87(6),
            pp. 2592-2605, 1990. doi:10.1121/1.399052
    [code]  NCM.m (Fei Chen and Philipos C. Loizou, 2011) and CSII.m (the
            same authors, 2012), in objective_measures/intelligibility of the
            MATLAB code that comes with Loizou's "Speech Enhancement: Theory
            and Practice" (2nd ed., CRC Press, 2013, doi:10.1201/b14529).
            The folder's readme scores an example pair with both.

The numbers below come from the test speech of tests/test_intelligibility.py:
3 s of synthetic syllables, a harmonic source through moving formants, with
pauses on a noise floor 60 dB down. tools/validation/intelligibility.py
reproduces the ones no test pins, and every comparison with pysepm.

**Where the papers and the code differ, I follow the code**: it is what
Loizou distributed for computing these two measures, and pysepm ports it line
for line. The differences, and the few places where I part from the code:

* NCM's envelopes. [MHL09] take them down to 25 Hz, which keeps modulations
  up to 12.5 Hz; NCM.m takes them to 32 Hz, up to 16 Hz. On 3 s of the test
  speech in white noise at 5 dB SNR, 25 Hz would give 0.556 where 32 Hz
  gives 0.546 (16 kHz).
* NCM's bands. [MHL09] work at 8 kHz, with the 20 bands of their Table 2,
  centred from 325 to 3227 Hz. NCM.m places the band edges at equal steps
  of cochlear place from 300 to 3400 Hz, [G90]'s function with its own
  constants (165 Hz, a slope of 2.1 over the cochlea, k = 1), which gives
  those centres; at 16 kHz it keeps 20 bands and stretches them to 7400 Hz,
  600 Hz under half the rate. A 16 kHz score is therefore a measure of its
  own: the pair above, taken down to 8 kHz, scores 0.635.
* NCM's weights. NCM.m carries the critical-band importance values of
  ANSI S3.5-1997 Table B.1 for short passages of easy reading material
  (tools/validation/sii_tables.py checks all 21) and interpolates them at
  its band centres. At 8 kHz that yields, to within 1e-4, the weights
  [MHL09] Table 2 lists for sentences; for consonant tests they used
  another set, which NCM.m does not have.
* NCM's filters. NCM.m filters each band with a Butterworth band-pass of
  order 8 (``butter(4, ...)``), forwards, as one transfer function. At
  16 kHz the transfer function of the lowest band, 300 to 370 Hz, is
  ill-conditioned: on the test speech its output strays from the exact
  filter by 1.6e-4 of the band's peak (4.2e-7 at 8 kHz). I run the same
  filters as second-order sections, which stay within 3e-14 of the exact
  filter. How far pysepm, which keeps the transfer functions, then lands
  from :func:`ncm` depends on the signal. Over the validation signals it is
  2.2e-6 at most at 16 kHz and 8.3e-9 at 8 kHz, a measurement rather than a
  bound, and there pysepm's own NCM moves by as much as 5.5e-8 when its
  inputs are only scaled, which NCM itself cannot feel. With its transfer
  functions put in place of the sections, the two agree to 1e-15.
* NCM of a copy. An r^2 that rounds above 1, as it does for an exact copy,
  a scaled one or an inverted one, is held at 1 and the band scores 1;
  pysepm takes the log of a negative number there and returns nan for all
  three. An output of digital silence scores 0, where NCM.m and pysepm
  divide 0 by 0 and return nan.
* NCM of a short clip. Through two points any two envelopes correlate
  perfectly, so a pair with fewer than three envelope samples would score 1
  whatever the output holds; it scores nan here.
* CSII's window. [MHL09] give 30 ms Hamming windows with 75% overlap; CSII.m
  uses ``0.5 * (1 - cos(2 pi n / (N + 1)))`` for n = 1..N, a Hann window
  without its zero end points, and so do I.
* CSII's low level. [MHL09], after [KA05], take it as the frames 10 to 30 dB
  under the reference's overall RMS. CSII.m sets no lower limit, so pauses
  count as low-level speech: of the 396 frames of the test speech, 226 are
  low-level and 166 of those lie more than 30 dB down. In white noise at
  5 dB SNR the code's low-level CSII is 0.084, where the -30 dB limit would
  give 0.198. I keep the code's.
* CSII adds float64 eps (2.2e-16) to both signals, as CSII.m does; pysepm
  does not. It changes nothing audible and gives a frame of digital silence
  a spectrum to divide by, all of it at 0 Hz. An output of digital silence
  therefore scores here, where pysepm returns nan: 0 at every level against
  a reference without an offset, more against one with (see the notes on
  offsets below).
* CSII's frames. Like the Hu and Loizou code (see :mod:`kudio.core.quality`),
  CSII.m takes ``floor(L / hop - N / hop)`` frames of N samples out of L, one
  fewer than fit, and the bins from 0 Hz up to the one below half the rate.
* A level that holds no frame, or one frame, scores nan here, with a
  warning. CSII.m reports 0 for a level with no frames, the value its
  accumulator starts from, and over one frame the coherence is 1 in every
  bin whatever the output holds. pysepm returns nan for an empty level.
* The coherence is held to [0, 1]. A copy of the reference makes it round a
  little above 1, which sends pysepm to nan for an exact, scaled or inverted
  copy; here they score 1.
* CSII's threshold. [KA05] run the whole SII procedure on the
  coherence-based spectra, auditory threshold included (Secs. IV.A and V),
  so their CSII depends on the presentation level. CSII.m keeps only the
  band SDRs, clipped and weighted, and so do I: neither measure hears level.

On the example pair in the code's folder, S_03_01.wav against
S_03_01_babble_sn0_klt.wav (8 kHz), :func:`csii` gives the readme's 0.5556,
0.2480 and 0.0457 to all four digits. :func:`ncm` gives 0.4152 where the
readme prints 0.4162. pysepm gives 0.4152 too, and none of the envelope
rates, filter orders and band layouts I tried reproduces the readme's
figure, so where its last 0.001 comes from is open.

What to watch for:

* Neither measure hears level. Both are ratios, band by band, so a gain on
  the output changes nothing, and neither has a hearing threshold, so a band
  the output turns down keeps its credit as long as its shape survives. A
  copy low-passed at 2 kHz scores CSII 1.0 at every level, and NCM 0.94 at
  8 kHz and 0.78 at 16 kHz. They say how much of the speech the output still
  carries, not whether a listener would hear it.
* Both need the pair lined up, CSII most: a copy 5 ms late scores 0.63 on
  high-level CSII and 0.98 on NCM, and 10 ms late 0.36 and 0.80. Use
  :func:`kudio.align` first.
* Short clips read high. The covariance and the coherence are estimated over
  the clip, and over a short one they find agreement that is not there: an
  unrelated stretch of the same kind of speech scores NCM 0.36 on average
  against 1 s clips of the test speech, 0.10 against 3 s and 0.02 against
  10 s, and white noise scores CSII 0.15 and 0.10 at the high and mid levels
  against 1 s and under 0.01 against 3 s (eight draws each). Score whole
  sentences or more, and compare scores over clips of like length.
* Zeros are not neutral. NCM correlates whole envelopes, so stretches where
  both signals are quiet count as agreement: 0.5 s of zeros either side of
  both takes the 5 dB pair above from 0.546 to 0.651, and the same output
  gated to digital silence between the syllables scores 0.642. The padding
  lands in CSII's low level and pulls it down instead, from 0.084 to 0.045,
  while the gate moves CSII by 0.001, as long as the reference carries no
  offset (see the next two notes).
* An offset on the output costs CSII where the speech is quiet: a copy with
  an offset of 1% of its peak scores 0.937 at the low level, and 5% scores
  0.853. NCM's band-pass filters take DC out, and it does not move.
* An offset on the reference credits silence. A frame of the output that is
  digital silence, or any constant, has all its power at 0 Hz, and CSII
  scores it by the coherence there, which says nothing about speech and
  which an offset on the reference raises. At 16 kHz a muted output scores
  0 at every level against the test speech, 0.46 at the low level once the
  reference carries an offset of 0.01% of its peak, and 0.055, 0.542 and
  0.831 at 0.1%. With 0.3% on both signals of the 5 dB pair, its output
  gated between the syllables scores 0.136 at the low level instead of
  0.083. Scoring ``np.zeros_like(ref)`` shows whether a reference does
  this. Taking the reference's mean out is not always enough, as the offset
  can differ between speech and pauses: the test speech so treated still
  gives the muted output 0.52 at the low level. What worked in every case
  the validation script tries, offsets of 0.001% to 5% of the peak and pink
  floors at 8 and 16 kHz, is to high-pass the reference at 20 Hz (second
  order, forwards and backwards, with scipy.signal.sosfiltfilt) and take
  the output's mean out. The muted output then scores 0, at the low level
  the gated one scores 0.082 and the copy with a 1% offset 0.999, and no
  score of the validation pairs moves by more than 1.5e-4. Filter only the
  reference: a filter rings into the output's digital silence, and the
  gated output would score 0.109 at the low level.
* The rate is part of NCM (see above) and not of CSII, whose bands are fixed
  in hertz: one pair scores the same CSII, to within 1e-4, at 8, 12, 16, 24,
  32 and 48 kHz. At 11.025, 22.05 and 44.1 kHz the 30 ms frame or its
  quarter-frame hop has to be rounded to whole samples, and the scores move
  by up to 0.011.
* CSII stops at 3.7 kHz at every rate, so it sees little of what happens
  above 4 kHz: noise kept above 5 kHz at 10 dB SNR costs it 0.01 to 0.03,
  and costs NCM 0.12 at 16 kHz.
* A level the reference never reaches has nothing to average and scores nan,
  with a warning: steady noise as the reference has no low level at all.
"""
from __future__ import annotations

import functools
import math
from typing import List, NamedTuple, Tuple

import numpy as np

from kudio.core import evaluator as _evaluator
from kudio.core._framing import frame_view
from kudio.core._policy import sample_rate, warn

__all__ = ['ncm', 'csii', 'CsiiScore']

_EPS = float(np.finfo(np.float64).eps)

#: The critical-band importance values NCM.m and CSII.m carry: the band
#: centres of ANSI S3.5-1997 Table 1, in Hz, and the share of
#: intelligibility each band holds for short passages of easy reading
#: material, the standard's Table B.1. The 21 values sum to 1.
_ANSI_CENTRES = (150.0, 250.0, 350.0, 450.0, 570.0, 700.0, 840.0, 1000.0,
                 1170.0, 1370.0, 1600.0, 1850.0, 2150.0, 2500.0, 2900.0,
                 3400.0, 4000.0, 4800.0, 5800.0, 7000.0, 8500.0)
_ANSI_IMPORTANCE = (0.0192, 0.0312, 0.0926, 0.1031, 0.0735, 0.0611, 0.0495,
                    0.0440, 0.0440, 0.0490, 0.0486, 0.0493, 0.0490, 0.0547,
                    0.0555, 0.0493, 0.0359, 0.0387, 0.0256, 0.0219, 0.0043)

#: Both measures hold their apparent SNR to +/-15 dB and map it linearly
#: onto 0..1 ([MHL09] Eqs. 9, 10 and 17). For NCM that is the STI's clip and
#: map ([GG04] Eqs. 3 and 5, applied to the SNR of their Eq. 10, as their
#: Sec. II.B.4 says); for CSII it is the SII's, whose procedure [KA05] keep
#: with the SDR in place of the SNR (their Sec. IV.A).
_SNR_LIMIT = 15.0

# -- NCM -------------------------------------------------------------------------

#: The rates NCM.m accepts. Its bands depend on the rate, so another rate
#: would be another measure.
_NCM_RATES = (8000, 16000)

#: NCM.m's filterbank: 20 bands with edges at equal steps of cochlear place,
#: ``x = (L / a) log10(f / A + k)`` with A = 165 Hz, a = 2.1, k = 1 and
#: L = 35 mm ([G90]'s function), from 300 Hz to 600 Hz under half the rate.
_NCM_BANDS = 20
_GREENWOOD = (165.0, 2.1, 1.0, 35.0)
_NCM_LOW_HZ = 300.0
_NCM_TOP_GAP_HZ = 600.0

#: The Butterworth prototype order NCM.m asks for, ``butter(4, [lo hi])``,
#: which makes a band-pass of order 8.
_BUTTER_ORDER = 4

#: The rate the envelopes are taken down to, in Hz, NCM.m's F_ENVELOPE:
#: modulations up to 16 Hz.
_ENVELOPE_RATE = 32

#: The fewest envelope samples the covariance is taken over: through two
#: points, any two envelopes correlate perfectly.
_MIN_ENVELOPE = 3

# -- CSII ------------------------------------------------------------------------

#: Bandwidths in Hz of CSII.m's 16 bands, the lowest 16 of the 21 critical
#: bands of ANSI S3.5-1997 Table 1, which [KA05] take the centres and widths
#: from (their Sec. IV.A), centred from 150 to 3400 Hz. They set how sharp
#: each band's rounded-exponential filter is.
_CSII_WIDTHS = (100.0, 100.0, 100.0, 110.0, 120.0, 140.0, 150.0, 160.0,
                190.0, 210.0, 240.0, 280.0, 320.0, 380.0, 450.0, 550.0)
_CSII_BANDS = len(_CSII_WIDTHS)

#: The lowest rate CSII is scored at. Its top band reaches 3675 Hz, so half
#: the rate has to clear that, and 8 kHz is the rate [MHL09] worked at.
_CSII_MIN_RATE = 8000.0

#: CSII.m's frames, 30 ms with a quarter-frame hop.
_FRAME_MS = 30.0

#: The levels, in dB against the reference's RMS over the whole pair: high at
#: or above 0 dB, mid from -10 dB up to 0, low under -10 dB.
_MID_FLOOR_DB = -10.0
_LEVELS = ('high', 'mid', 'low')

#: Frames measured and transformed at a time: ten minutes at 16 kHz is 79,996
#: frames, and their 512-bin spectra at once would be 655 MB a signal.
_BLOCK = 2048


class CsiiScore(NamedTuple):
    """CSII at the three levels of the reference, each 0 to 1, higher better.

    The levels split the reference's 30 ms frames by their RMS against the
    RMS of the whole: ``high`` takes the frames at or above it, ``mid`` those
    0 to 10 dB under it and ``low`` everything further down, pauses and
    digital silence included. Each value is the mean over its own frames.
    """

    #: frames at or above the reference's overall RMS
    high: float
    #: frames 0 to 10 dB under it
    mid: float
    #: frames more than 10 dB under it
    low: float


# --- input ------------------------------------------------------------------------

def _paired(ref, deg, name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Both signals as one channel of float64 each, cut to the length they
    share, as every metric of :mod:`kudio.core.evaluator` takes them: a
    mismatch warns, and past ``kudio.core.evaluator.MAX_LENGTH_MISMATCH`` of
    the longer one it is refused."""
    return _evaluator._align(ref, deg, caller=name)


# --- NCM --------------------------------------------------------------------------

def _ncm_edges(sr: int) -> np.ndarray:
    """The 21 band edges in Hz at *sr*, as NCM.m's Get_Band places them."""
    big_a, slope, k, length = _GREENWOOD
    low = (length / slope) * math.log10(_NCM_LOW_HZ / big_a + k)
    high = (length / slope) * math.log10((sr / 2.0 - _NCM_TOP_GAP_HZ) / big_a + k)
    place = low + np.arange(_NCM_BANDS + 1) * ((high - low) / _NCM_BANDS)
    return big_a * (10.0 ** (slope * place / length) - k)


@functools.lru_cache(maxsize=2)
def _ncm_filterbank(sr: int):
    """``(edges, weights, sections)`` at *sr*: the band edges, each band's
    importance weight, and each band's filter as second-order sections."""
    from scipy.signal import butter

    edges = _ncm_edges(sr)
    # read off the importance table at the arithmetic centre of each band,
    # along straight lines between its points (interp1 in NCM.m)
    weights = np.interp(0.5 * (edges[:-1] + edges[1:]), _ANSI_CENTRES,
                        _ANSI_IMPORTANCE)
    # left writable: scipy's sosfilt will not take a read-only array
    sections = tuple(butter(_BUTTER_ORDER, edges[k:k + 2] * (2.0 / sr),
                            btype='bandpass', output='sos')
                     for k in range(_NCM_BANDS))
    edges.flags.writeable = False
    weights.flags.writeable = False
    return edges, weights, sections


def _band(y: np.ndarray, sos: np.ndarray) -> np.ndarray:
    """*y* through one band's filter, forwards only, as NCM.m's ``filter``."""
    from scipy.signal import sosfilt
    return sosfilt(sos, y)


def _hilbert_magnitude(x: np.ndarray) -> np.ndarray:
    """``abs(hilbert(x))`` over the whole of *x*, with an FFT exactly as long
    as *x*, which is what MATLAB's ``hilbert`` does."""
    from scipy.fft import ifft, rfft

    n = len(x)
    spectrum = np.zeros(n, dtype=complex)
    spectrum[:n // 2 + 1] = rfft(x)
    spectrum[1:(n + 1) // 2] *= 2.0     # the bins strictly between 0 and n / 2
    return np.abs(ifft(spectrum, overwrite_x=True))


def _envelope(y: np.ndarray, sos: np.ndarray, sr: int) -> np.ndarray:
    """One band's envelope at 32 Hz.

    ``resample_poly(x, 32, sr)`` does what MATLAB's ``resample(x, 32, sr)``
    does: the same Kaiser-windowed low-pass (beta 5, ten output samples
    either side), the same zeros beyond the ends and the same alignment.
    """
    from scipy.signal import resample_poly
    return resample_poly(_hilbert_magnitude(_band(y, sos)), _ENVELOPE_RATE, sr)


def _transmission_index(x: np.ndarray, y: np.ndarray) -> float:
    """One band's TI from its two envelopes, [MHL09] Eqs. 8 to 10."""
    x = x - x.mean()
    y = y - y.mean()
    sxx, syy, sxy = float(x @ x), float(y @ y), float(x @ y)
    # an output envelope that never moves shares no modulation with the
    # reference; NCM.m divides 0 by 0 there
    r2 = 0.0 if syy == 0.0 else min(sxy * sxy / (sxx * syy), 1.0)
    snr = 10.0 * math.log10((r2 + _EPS) / (1.0 - r2 + _EPS))
    snr = min(max(snr, -_SNR_LIMIT), _SNR_LIMIT)
    return (snr + _SNR_LIMIT) / (2.0 * _SNR_LIMIT)


def ncm(ref: np.ndarray, deg: np.ndarray, sr: int) -> float:
    """Normalized covariance metric of *deg* against *ref*: 0 to 1, higher is
    more intelligible.

    >>> kudio.ncm(clean, enhanced, 16000)                        # doctest: +SKIP

    How it is computed, as NCM.m computes it:

    1. Each signal goes through 20 Butterworth band-pass filters of order 8,
       forwards, their edges at equal steps of cochlear place from 300 Hz to
       600 Hz under half the rate: 300 to 3400 Hz at 8 kHz, 300 to 7400 Hz
       at 16 kHz.
    2. Each band's envelope is the magnitude of its analytic signal over the
       whole clip, taken down to 32 Hz as MATLAB's ``resample`` takes it.
    3. In each band, ``r^2`` is the squared correlation of the two envelopes
       over the clip, ``SNR = 10 log10(r^2 / (1 - r^2))`` held to +/-15 dB,
       and ``TI = (SNR + 15) / 30``.
    4. NCM is the mean of the 20 TI values, weighted by the band importance
       NCM.m carries, read at each band's centre: the column of
       ANSI S3.5-1997 Table B.1 for short passages of easy reading
       material, not the standard's own Table 1 function.

    The module notes list where this parts from [MHL09] and from NCM.m, and
    what to watch for.

    Step 2 takes an FFT exactly as long as the clip, as MATLAB's ``hilbert``
    does, so the time depends on how the length factors: 4.6 s for two
    minutes at 16 kHz (1,920,000 samples) and 19.9 s for 1,919,987, a prime,
    on one desktop CPU. Padding to a faster length would change the
    envelopes and move the score off NCM.m's, so it is left as it is.

    No step depends on level. Each band's correlation is the same whatever
    gain either signal has, and so a band the output turns down still scores
    as long as its envelope keeps its shape. Both signals are scaled to a peak
    of 1 first, which moves no score and keeps a very loud or very quiet pair
    clear of overflow.

    :param ref: the clean signal, mono. Integer audio is read on its type's
        full scale.
    :param deg: the processed signal, mono and lined up with *ref*.
    :param sr: 8000 or 16000.
    :raises ValueError: at any other rate, for a signal with more than one
        channel (one held as ``(n, 1)`` or ``(1, n)`` is read as mono), for a
        single number, for samples that are complex, not numbers or not
        finite, and for a pair too different in length (see
        :data:`kudio.core.evaluator.MAX_LENGTH_MISMATCH`; a smaller
        difference warns, and the end of the longer one is left out).
    :returns: the NCM. An output of digital silence scores 0. nan, with a
        warning, for a reference that is digital silence throughout and for
        a pair too short for three envelope samples: 501 samples at 8 kHz,
        1001 at 16 kHz (62.6 ms).
    """
    rate = sample_rate(sr, 'ncm')
    if rate not in _NCM_RATES:
        raise ValueError(
            f"ncm(): NCM is defined at 8 and 16 kHz, not {rate} Hz: its 20 "
            f"bands run from 300 Hz to 600 Hz under half the rate, so another "
            f"rate is another measure, and NCM.m refuses it. Resample first, "
            f"with kudio.resample or, without librosa, "
            f"scipy.signal.resample_poly(y, 16000, sr)")
    sr = int(rate)
    ref, deg = _paired(ref, deg, 'ncm')
    need = (_MIN_ENVELOPE - 1) * (sr // _ENVELOPE_RATE) + 1
    if len(ref) < need:
        warn(f"ncm(): the pair needs at least {need} samples "
             f"({1000.0 * need / sr:.1f} ms at {sr} Hz) for three "
             f"envelope samples at 32 Hz, and has {len(ref)}; scored nan")
        return float('nan')
    peak = float(np.max(np.abs(ref)))
    if peak == 0.0:
        warn("ncm(): ref is digital silence throughout, so there is no "
             "envelope to compare deg with; scored nan")
        return float('nan')
    ref = ref / peak
    peak = float(np.max(np.abs(deg)))
    if peak > 0.0:
        deg = deg / peak
    _, weights, sections = _ncm_filterbank(sr)
    ti = np.array([_transmission_index(_envelope(ref, sos, sr), _envelope(deg, sos, sr))
                   for sos in sections])
    # summed the same way above and below the line, so that a copy, every
    # band at 1, scores exactly 1
    return float(np.sum(weights * ti) / np.sum(weights))


# --- CSII -------------------------------------------------------------------------

def _csii_rate(sr) -> float:
    """*sr* as a float, refused below 8 kHz and when it is not a rate."""
    rate = sample_rate(sr, 'csii')
    if rate < _CSII_MIN_RATE:
        raise ValueError(
            f"csii(): CSII needs a rate of 8 kHz or more, got {rate}: its 16 "
            f"bands reach 3675 Hz, and half the rate has to clear that. "
            f"Resample first, with kudio.resample or, without librosa, "
            f"scipy.signal.resample_poly(y, 16000, sr)")
    return float(rate)


@functools.lru_cache(maxsize=8)
def _csii_setup(sr: float):
    """``(frame, hop, nfft, window, filters, weights)`` at *sr*.

    The frame is 30 ms rounded the way MATLAB rounds, halves away from zero;
    the hop a quarter of it, rounded down; the FFT the next power of two at
    or above twice the frame. The filters are CSII.m's rounded exponentials
    over bins 0 .. nfft/2 - 1, ``(1 + p g) exp(-p g)`` with
    ``g = |1 - f / fc|`` and ``p = 4 fc / bandwidth``, each 1 at its centre.
    """
    size = int(math.floor(_FRAME_MS * sr / 1000.0 + 0.5))
    hop = size // 4
    nfft = 1 << (2 * size - 1).bit_length()
    half = nfft // 2
    n = np.arange(1, size + 1)
    window = 0.5 * (1.0 - np.cos(2.0 * np.pi * n / (size + 1)))
    # in kHz and back, the way CSII.m writes it
    centre = np.asarray(_ANSI_CENTRES[:_CSII_BANDS]) / 1000.0
    sharpness = 4.0 * 1000.0 * centre / np.asarray(_CSII_WIDTHS)
    distance = np.abs(1.0 - np.arange(half)[None, :] * (sr / nfft)
                      / (centre[:, None] * 1000.0))
    pg = sharpness[:, None] * distance
    filters = (1.0 + pg) * np.exp(-pg)
    weights = np.asarray(_ANSI_IMPORTANCE[:_CSII_BANDS])
    for a in (window, filters, weights):
        a.flags.writeable = False
    return size, hop, nfft, window, filters, weights


def _frame_count(n: int, size: int, hop: int) -> int:
    """How many frames CSII.m scores in *n* samples.

    Its own expression, ``floor(n / hop - size / hop)``: one frame fewer than
    fit, and at 11.025, 22.05 and 44.1 kHz, where the hop does not divide the
    frame, now and then one fewer again, the two quotients rounding apart.
    Kept, so that the same frames are scored as in CSII.m and pysepm.
    """
    return int(math.floor(n / hop - size / hop))


def _spectra(frames: np.ndarray, window: np.ndarray, nfft: int) -> np.ndarray:
    """Spectra of the windowed frames over bins 0 .. nfft/2 - 1."""
    return np.fft.rfft(frames * window, nfft, axis=1)[:, :nfft // 2]


def csii(ref: np.ndarray, deg: np.ndarray, sr: float) -> CsiiScore:
    """Coherence speech intelligibility index of *deg* against *ref*, at the
    reference's high, mid and low levels: each 0 to 1, higher is more
    intelligible.

    >>> high, mid, low = kudio.csii(clean, enhanced, 16000)      # doctest: +SKIP

    How it is computed, as CSII.m computes it:

    1. float64 eps is added to both signals, and each is cut into 30 ms
       frames with a quarter-frame hop. A frame's RMS against the
       reference's RMS over the whole pair puts it at one of the three
       levels (:class:`CsiiScore`).
    2. For each level, the magnitude-squared coherence of the two signals is
       estimated over that level's frames: in each frequency bin, ``|sum X
       Y*|^2 / (sum |X|^2 sum |Y|^2)``, X and Y the spectra of the windowed
       frames ([MHL09] Eq. 23).
    3. In each frame and each of 16 critical bands, centred from 150 to
       3400 Hz, the coherence splits the output's power into the part that
       follows the reference and the part that does not; their ratio, held
       to +/-15 dB, is mapped onto 0..1 ([MHL09] Eq. 24).
    4. A frame's score is its band values averaged with the same Table B.1
       importance values, and a level's score the mean over its frames
       ([MHL09] Eq. 25).

    The module notes list where this parts from [MHL09] and from CSII.m, and
    what to watch for. Like NCM it hears no level and has no hearing
    threshold. Unlike NCM, it is next to blind to a filter on the output
    whose response is short against the 30 ms frame, a low-pass or an
    equalizer, since the copy stays coherent with the reference in every
    bin; and its bands are fixed in hertz, so the rate does not enter.

    :param ref: the clean signal, mono. Integer audio is read on its type's
        full scale.
    :param deg: the processed signal, mono and lined up with *ref*.
    :param sr: the sample rate, 8000 Hz or more.
    :raises ValueError: for a rate under 8 kHz or one that is not a number,
        for a signal with more than one channel (one held as ``(n, 1)`` or
        ``(1, n)`` is read as mono), for a single number, for samples that
        are complex, not numbers or not finite, and for a pair too different
        in length (see :data:`kudio.core.evaluator.MAX_LENGTH_MISMATCH`; a
        smaller difference warns, and the end of the longer one is left out).
    :returns: a :class:`CsiiScore`. An output of digital silence scores 0 at
        every level against a reference without an offset, and more against
        one with; the module notes give the numbers, and the cure. Every
        field is nan, with a warning, for a reference that is digital
        silence throughout or too loud to square, and for a pair too short
        for two frames (45 ms: 360 samples at 8 kHz, 720 at 16 kHz). One
        field is nan, with a warning, when its level holds no frame or a
        single frame (over one frame the coherence is 1, whatever the output
        holds), or a frame of the output too loud to square.
    """
    rate = _csii_rate(sr)
    ref, deg = _paired(ref, deg, 'csii')
    nan = float('nan')
    size, hop, nfft, window, filters, weights = _csii_setup(rate)
    n = len(ref)
    count = _frame_count(n, size, hop)
    if count < 2:
        need = size + 2 * hop
        while _frame_count(need, size, hop) < 2:
            need += 1
        warn(f"csii(): the pair needs at least {need} samples "
             f"({1000.0 * need / rate:.1f} ms at {rate:g} Hz) for two "
             f"frames, the fewest a coherence can be taken over, and "
             f"has {n}; scored nan")
        return CsiiScore(nan, nan, nan)
    if not np.any(ref):
        warn("csii(): ref is digital silence throughout, so there is "
             "no speech to compare deg with; scored nan")
        return CsiiScore(nan, nan, nan)

    clean = ref + _EPS
    processed = deg + _EPS
    frames_c = frame_view(clean, size, hop)[:count]
    frames_p = frame_view(processed, size, hop)[:count]
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        overall = np.linalg.norm(clean) / math.sqrt(n)
        # a block of frames at a time, as the spectra are taken: squared all
        # at once, the frames would take four times the memory of the signal,
        # each sample sitting in four of them
        norms = np.empty(count)
        for start in range(0, count, _BLOCK):
            norms[start:start + _BLOCK] = np.linalg.norm(frames_c[start:start + _BLOCK],
                                                         axis=1)
        level = 20.0 * np.log10(norms / math.sqrt(size) / overall)
    if not (math.isfinite(overall) and np.all(np.isfinite(level))):
        warn("csii(): ref is too loud to square, so its frames have no "
             "level to sort them by; scored nan")
        return CsiiScore(nan, nan, nan)
    # 0 high, 1 mid, 2 low
    group = np.where(level >= 0.0, 0, np.where(level >= _MID_FLOOR_DB, 1, 2))
    msc = _coherence(frames_c, frames_p, group, window, nfft)
    score = np.empty(count)
    for start in range(0, count, _BLOCK):
        stop = min(start + _BLOCK, count)
        share = msc[group[start:stop]]
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            power = np.abs(_spectra(frames_p[start:stop], window, nfft)) ** 2
            follows = (power * share) @ filters.T
            departs = (power * (1.0 - share)) @ filters.T
            sdr = 10.0 * np.log10(follows / departs)
        t = (np.clip(sdr, -_SNR_LIMIT, _SNR_LIMIT) + _SNR_LIMIT) / (2.0 * _SNR_LIMIT)
        # as in ncm(): summed alike above and below, so that a frame with
        # every band at 1 scores exactly 1
        score[start:stop] = np.sum(t * weights, axis=1) / np.sum(weights)
    values, why = _levels(score, group)
    if why:
        warn(f"csii(): {'; '.join(why)}; scored nan")
    return CsiiScore(*values)


def _coherence(frames_c: np.ndarray, frames_p: np.ndarray, group: np.ndarray,
               window: np.ndarray, nfft: int) -> np.ndarray:
    """``(3, nfft / 2)``: each level's magnitude-squared coherence over that
    level's frames, held to [0, 1]. A level with no frames is nan."""
    half = nfft // 2
    cross = np.zeros((3, half), dtype=complex)
    power_c = np.zeros((3, half))
    power_p = np.zeros((3, half))
    for start in range(0, len(group), _BLOCK):
        stop = min(start + _BLOCK, len(group))
        x = _spectra(frames_c[start:stop], window, nfft)
        y = _spectra(frames_p[start:stop], window, nfft)
        g = group[start:stop]
        with np.errstate(over='ignore', invalid='ignore'):
            for k in range(3):
                inside = g == k
                if np.any(inside):
                    cross[k] += np.sum(x[inside] * np.conj(y[inside]), axis=0)
                    power_c[k] += np.sum(np.abs(x[inside]) ** 2, axis=0)
                    power_p[k] += np.sum(np.abs(y[inside]) ** 2, axis=0)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        # it cannot pass 1 in exact arithmetic, but a copy of the reference
        # rounds a little over, and 1 - msc would then go negative
        return np.clip(np.abs(cross) ** 2 / (power_c * power_p), 0.0, 1.0)


def _levels(score: np.ndarray, group: np.ndarray) -> Tuple[List[float], List[str]]:
    """Each level's mean frame score, or nan and the reason."""
    values, why = [], []
    for k, name in enumerate(_LEVELS):
        frames = score[group == k]
        lost = int(np.count_nonzero(np.isnan(frames)))
        if frames.size == 0:
            why.append(f"no frame of ref is at the {name} level, so CSII {name} "
                       f"has nothing to average")
        elif frames.size == 1:
            why.append(f"only one frame of ref is at the {name} level, and over "
                       f"one frame the coherence is 1 whatever deg holds, so "
                       f"CSII {name} would say nothing about deg")
        elif lost:
            why.append(f"{lost} of the {frames.size} frames at the {name} level "
                       f"could not be measured, deg being too loud to square")
        else:
            values.append(float(np.mean(frames)))
            continue
        values.append(float('nan'))
    return values, why
