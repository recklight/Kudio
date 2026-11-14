# -*- coding: utf-8 -*-
"""WADA-SNR: the SNR of a speech recording, estimated from the recording alone.

Kim and Stern, "Robust signal-to-noise ratio estimation based on waveform
amplitude distribution analysis", Interspeech 2008, pp. 2598-2601. The samples
of clean speech, pauses included, are spread like a two-sided gamma
distribution of shape 0.4 (their Eq. 1 and Fig. 1). Add Gaussian noise and the
spread changes in a way that one statistic follows and the SNR alone sets::

    G = ln(mean |z|) - mean(ln |z|)                        (their Eq. 8)

G is 0.4094 for Gaussian noise on its own and 1.6451 for the gamma speech on
its own, and rises from one to the other as the SNR does (their Eq. 17 and
Fig. 2). Measure G on a clip, find the SNR at which the model gives that G,
and that is the estimate: no clean reference, no voice activity detection, no
frames, one pass over the samples.

**The curve is computed from the model, not looked up.** The table published
with the method (``Alpha0.400000.txt`` in LabROSA's snreval, from Kim's C++
code, and copied by the Python ports of snreval's ``wada_snr.m``) came out of
a Monte Carlo integration (Sec. 3 of the paper) and carries that noise: it
dips at -17 dB, and at 97 dB it sits 0.0053 above the model. Here both
expectations of Eq. 17 are taken through the characteristic functions of the
gamma and the Gaussian part, which are closed forms, summed on a fixed
log-frequency grid. A second route, through the Poisson-mixture form of
``E ln|a + noise|``, agrees with it to 2.3e-11 from -20 to 110 dB
(tools/validation/wada.py), and a Monte Carlo run of 2e8
samples agrees within its standard error. Read on this curve, the table's
entries move by at most 0.25 dB between -10 and 30 dB, by 0.47 dB at 50 dB
and by 7.3 dB at 100 dB, where the curve is nearly flat.

What to watch for:

* The answer is read off -20 to 100 dB, the table's span. A clip whose G
  falls outside it gets nan and a warning instead of the end of the span. The
  ports clamp there, and the 100 dB they then report for a file with digital
  silence in it reads as a clean recording when the file may be anything but.
* Above about 20 dB the speech decides, not the noise. The paper's curves
  for shapes 0.40, 0.44 and 0.54 part there (Fig. 2), and real speech is not
  one shape. On a harmonic test signal with about a quarter of it pauses,
  averaged over 20 clips of 10 s, the estimate is 1.2 dB low at 0 dB, within
  0.7 dB from 5 to 15 dB, 2.0 dB high at 20 dB and 11.7 dB high at 30 dB.
  Read 35 dB as "clean", not as 35.
* Below about -10 dB there is little left to read. The curve is 0.0028
  above noise alone at -10 dB and 0.0004 at -15 dB, while G itself scatters
  by 0.0016 (one SD) over 10 s of Gaussian noise. Of 20 clips of the
  harmonic test signal at -10 dB, 5 come out nan and the other 15 read
  2.4 dB low on average, scattering by 2.5 dB; at -15 dB, 13 of the 20 are
  nan. Gaussian noise with no speech in it at all comes out nan in 23 of 40
  clips of 10 s and reads -10 to -16 dB in the other 17.
* The noise is taken to be Gaussian. Music and other talkers move the
  answer. With a second talker as the noise, the paper's estimates stay
  around 15 to 20 dB while the true SNR goes from -10 to 10 dB (Fig. 4a):
  the other talker is read as more speech.
* It needs a second at least, and seconds to settle. The statistic is a
  mean over samples, but what it averages is the alternation of syllables
  and pauses, and a short stretch may hold no syllable at all: of 60
  half-second windows cut from the harmonic test signal at 10 dB, 2 fell
  inside a pause. A clip shorter than 1 s, once the zeros at its ends are
  left out, scores nan. On that signal at 10 dB the estimate scatters by
  1.5 dB (one SD) on 1 s clips and by 0.5 dB on 5 s clips. The paper scored
  whole sentences.
* Pauses muted to digital silence leave nothing to read. G falls as the
  noise fills the pauses, so the pauses are where the estimate comes from.
  In a clip that was noise-gated, edited or made with no noise at all, they
  are runs of exact zeros, and no way of handling zeros gives an SNR there.
  Left out, the zeros take the pauses with them, and the voiced parts on
  their own are less peaked than speech: the harmonic test signal with no
  noise and its pauses at zero reads 23 dB, and with 20 dB of white noise
  gated off in its pauses, 14 dB instead of 22. Floored as the ports floor
  them, both read 100 dB. So once runs of exact zeros of 10 ms or more fill
  more than 1% of the clip between its first and last nonzero sample, it
  scores nan with a warning. Under that share they are left out at little
  cost (1% of the 20 dB clip gated off takes 0.2 dB from the estimate), and
  zeros before the first nonzero sample and after the last are padding,
  left out at any length.

Two steps follow the paper rather than the ports. The mean is removed first,
as in its Fig. 3; without that, a DC offset of 0.3 times the RMS takes a
10 dB clip to 5.9 dB. And samples that are exactly zero are left out, the
first of the two options its Sec. 3 gives. The ports take the other, a floor
of 1e-10, which turns zero padding into the sparsest speech there is: on a
0 dB clip, 1% of zero padding reads 9.4 dB, and 10% reads 100 dB, the end of
their table. On int16 audio, leaving the zeros out costs a little once the
noise is down to tens of LSB: against the same clip in float, -0.3 dB at
30 dB with the noise at 70 LSB RMS, and -0.9 dB at 40 dB with it at 22.
Rounding leaves its zeros a few samples long at most, 5 in the pauses of the
harmonic test signal with the noise at 5 LSB RMS, well short of the 10 ms
that counts as digital silence.
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Tuple

import numpy as np

from kudio.core._policy import faint, sample_rate, scored_nan, signal

__all__ = ['wada_snr']

#: Shape of the gamma distribution clean speech is modelled with. The paper
#: cites 0.4 or 0.5 as the best fit for clean speech (its refs. [8] and [9],
#: Paez and Glisson 1972 and Rabiner and Schafer 1978) and fixes 0.4 (Kim and
#: Stern 2008, Sec. 2).
_SPEECH_SHAPE = 0.4

#: The SNRs, in dB, the curve is read over: -20 to 100, the span of the table
#: published with the method (snreval's ``Alpha0.400000.txt``). Past -20 the
#: curve is within 5e-5 of noise alone, and past 100 within 0.019 of speech
#: alone, so neither end has much left to tell apart.
_SNR_SPAN = (-20.0, 100.0)

#: The grid the two characteristic-function integrals are summed on, in
#: u = ln t: from -60, below which both integrands are under 1e-24, to 45,
#: past which the noise's Gaussian factor is under exp(-800) for any SNR up
#: to 350 dB, in steps of 0.01. Halving the step and widening the span
#: changes G by under 3e-12.
_U_LOW, _U_HIGH, _U_STEP = -60.0, 45.0, 0.01

#: Shortest clip scored, in seconds, counted from its first nonzero sample to
#: its last. Of 60 half-second windows cut from the harmonic test signal at
#: 10 dB, 2 held no speech at all; from 1 s on, none did.
_MIN_SECONDS = 1.0

#: A run of exact zeros this long or longer, in seconds, is digital silence
#: rather than rounding. Rounded to int16, the harmonic test signal's noise
#: leaves runs of 5 samples at most at 5 LSB RMS and 8 at 1.7 LSB: 0.3 and
#: 0.5 ms at 16 kHz.
_SILENCE_RUN = 0.01

#: How much of the clip, between its first and last nonzero sample, digital
#: silence may fill before the clip is refused. Gating 1% of a 20 dB clip of
#: the harmonic test signal to zero takes 0.2 dB from the estimate, and
#: gating all of its pauses, a quarter of it on average, takes 7.8 dB.
_SILENCE_SHARE = 0.01


@lru_cache(maxsize=1)
def _grid() -> Tuple[np.ndarray, ...]:
    """The parts of the integrands that do not depend on the SNR.

    With the gamma's rate set to 1 -- G does not change with the scale of the
    signal -- the speech's characteristic function is
    ``(1 + t^2)^(-shape/2) * cos(shape * atan t)``, kept here as its log
    magnitude and its angle.
    """
    u = np.arange(_U_LOW, _U_HIGH + _U_STEP / 2, _U_STEP)
    t = np.exp(u)
    angle = _SPEECH_SHAPE * np.arctan(t)
    log_magnitude = -0.5 * _SPEECH_SHAPE * np.log1p(t * t)
    # 1 - cos(angle) as 2 sin^2(angle / 2): it is about angle^2 / 2 at small t,
    # where 1 - cos would cancel down to nothing
    versine = 2.0 * np.sin(0.5 * angle) ** 2
    return t * t, np.cos(angle), versine, log_magnitude, np.exp(-t), np.exp(-u)


def _g(snr_db: float) -> float:
    """G of the model at *snr_db*: gamma speech in Gaussian noise (Eq. 17).

    Both expectations come from the characteristic function phi of the noisy
    signal, the product of the speech's and the noise's, through two
    identities for a real z that is not zero::

        ln|z| = integral_0^inf (exp(-t) - cos(z t)) / t dt
        |z|   = (2 / pi) integral_0^inf (1 - cos(z t)) / t^2 dt

    so ``E ln|z|`` and ``E|z|`` are the same integrals with phi(t) in place of
    cos(z t). Substituting t = e^u makes both integrands smooth and fall off
    exponentially at either end, where a plain sum on an even grid converges
    faster than any power of the step.
    """
    t2, cos_angle, versine, log_magnitude, decay, inv_t = _grid()
    # Eq. 6: SNR = shape (shape + 1) / (sigma * rate)^2, with the rate at 1
    noise_var = _SPEECH_SHAPE * (_SPEECH_SHAPE + 1.0) / 10.0 ** (snr_db / 10.0)
    log_phi = log_magnitude - 0.5 * noise_var * t2
    mean_log = _U_STEP * float(np.sum(decay - np.exp(log_phi) * cos_angle))
    # 1 - phi without the cancellation: (1 - cos) + cos * (1 - |phi|)
    one_minus_phi = versine - cos_angle * np.expm1(log_phi)
    # past the grid's end phi is 0 and 1 - phi is 1, which leaves the
    # integral of e^-u from there on: e^-45
    tail = math.exp(-_U_HIGH)
    mean_abs = (2.0 / math.pi) * (_U_STEP * float(np.sum(one_minus_phi * inv_t)) + tail)
    return math.log(mean_abs) - mean_log


@lru_cache(maxsize=1)
def _curve() -> Tuple[np.ndarray, np.ndarray]:
    """``(snr_db, G)`` at every whole dB over :data:`_SNR_SPAN`.

    About 20 ms to build, once per process. It brackets the search in
    :func:`_snr_of`; the answer itself comes from :func:`_g`, not from
    interpolating these points.
    """
    snr = np.arange(_SNR_SPAN[0], _SNR_SPAN[1] + 0.5)
    return snr, np.array([_g(float(d)) for d in snr])


def _snr_of(g: float) -> float:
    """The SNR at which the model's G equals *g*.

    The curve rises strictly, so inside it there is exactly one. A *g* at or
    past either end is not a value the model gives over :data:`_SNR_SPAN`:
    it gets nan and a warning saying which end, and never reaches the search,
    whose bracket exists only inside the curve.
    """
    from scipy.optimize import brentq

    snr, curve = _curve()
    if g <= curve[0]:
        return scored_nan('wada_snr', (
            f"G = {g:.4f} is at or below {curve[0]:.4f}, the model's value at "
            f"{_SNR_SPAN[0]:g} dB, so the clip is no more peaked than Gaussian "
            f"noise. Its speech, if it has any, is {-_SNR_SPAN[0]:g} dB or more "
            f"under the noise, or the noise is steadier than Gaussian (a hum, a "
            f"whistle, heavy limiting), which the model cannot read"))
    if g >= curve[-1]:
        return scored_nan('wada_snr', (
            f"G = {g:.4f} is at or above {curve[-1]:.4f}, the model's value at "
            f"{_SNR_SPAN[1]:g} dB, so the clip is sparser than gamma-distributed "
            f"speech in any noise the curve covers. It has next to no noise, or "
            f"its speech is sparser than the model's, or it has clicks or long "
            f"stretches of near silence, and the {_SNR_SPAN[1]:g} dB at the end "
            f"of the curve would pass it off as a clean recording"))
    k = int(np.searchsorted(curve, g))
    return float(brentq(lambda d: _g(d) - g, snr[k - 1], snr[k], xtol=1e-9))


def _digital_silence(y: np.ndarray, sr: float) -> int:
    """How many samples of *y* sit in runs of exact zeros at least
    :data:`_SILENCE_RUN` long."""
    shortest = max(1, int(math.ceil(_SILENCE_RUN * sr)))
    edge = np.zeros(1, dtype=np.int8)
    zero = (y == 0.0).view(np.int8)
    # +1 where a run of zeros starts and -1 just past where it ends
    bounds = np.flatnonzero(np.diff(np.concatenate((edge, zero, edge))))
    lengths = bounds[1::2] - bounds[::2]
    return int(np.sum(lengths[lengths >= shortest]))


def wada_snr(y: np.ndarray, sr: int) -> float:
    """Estimate the SNR of the speech in *y*, in dB, with no clean reference.

    >>> kudio.wada_snr(noisy, 16000)                         # doctest: +SKIP
    9.7

    The method reads how the sample values are spread, not the spectrum, so
    neither the rate nor the level enters the estimate: ``y`` and ``10 * y``
    give the same answer. *sr* is there to measure the clip, and its runs of
    zeros, in seconds. An integer array is read on its own full scale, as
    everywhere in kudio.

    The mean is taken out and exact zeros are left out before G is measured;
    the module docstring says why, and what the estimate can and cannot be
    trusted with.

    :param y: mono audio. A ``(samples, 1)`` or ``(1, samples)`` array is
        read as the one channel it holds.
    :param sr: sample rate in Hz.
    :raises ValueError: for a rate that is not positive, a
        ``(samples, channels)`` array, or samples that are not finite.
    :returns: the estimate in dB. nan, with a warning saying why, when there
        is no waveform to read (an empty clip, all zeros, one value
        throughout); when the clip is shorter than 1 s from its first nonzero
        sample to its last; when runs of exact zeros of 10 ms or more fill
        over 1% of that stretch, as they do in a noise-gated or noise-free
        recording; and when G falls outside the -20 to 100 dB the curve is
        read over.
    """
    sr = sample_rate(sr, 'wada_snr')
    y = signal(y, 'y', 'wada_snr')
    faint('wada_snr', [('y', y)])
    if y.size == 0:
        return scored_nan('wada_snr', "the clip is empty")

    sound = y != 0.0
    if not sound.any():
        return scored_nan('wada_snr', f"all {y.size} samples are zero, so there is "
                                      f"no waveform to read an SNR from")
    first, stop = int(np.argmax(sound)), y.size - int(np.argmax(sound[::-1]))
    padded = first > 0 or stop < y.size
    y = y[first:stop]                     # zeros before and after are padding

    silent = _digital_silence(y, sr)
    if silent > _SILENCE_SHARE * y.size:
        return scored_nan('wada_snr', (
            f"{silent / sr:.2f} s of the {y.size / sr:.2f} s between the clip's "
            f"first and last nonzero sample ({100.0 * silent / y.size:.0f}%) is "
            f"digital silence, in runs of exact zeros of {1000 * _SILENCE_RUN:g} "
            f"ms or more. The estimate comes from the noise in the pauses, and "
            f"pauses muted to zero have none, so a noise-gated, edited or "
            f"noise-free clip cannot be scored (if the zeros only join takes "
            f"that each carry their own noise, score the takes one at a time)"))
    need = int(math.ceil(_MIN_SECONDS * sr))
    if y.size < need:
        trimmed = " once the zeros at its ends are left out" if padded else ""
        return scored_nan('wada_snr', f"the clip needs at least {_MIN_SECONDS:g} s "
                                      f"of audio, {need} samples at {sr:g} Hz, and "
                                      f"has {y.size}{trimmed}")

    z = y[y != 0.0]
    z = z - z.mean()
    z = z[z != 0.0]          # a sample that sat exactly on the mean
    if z.size < 2:
        return scored_nan('wada_snr', "apart from its zeros, every sample of the "
                                      "clip has the same value, so there is no "
                                      "waveform to read an SNR from")

    magnitude = np.abs(z)
    g = math.log(float(np.mean(magnitude))) - float(np.mean(np.log(magnitude)))
    return _snr_of(g)
