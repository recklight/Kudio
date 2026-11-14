# -*- coding: utf-8 -*-
"""SIIB, speech intelligibility in bits: how much of what a talker says
reaches a listener, in bits per second, from the clean speech and from what
the listener hears.

Van Kuyk, Kleijn and Hendriks model speech communication as a chain, from a
message to the talker's speech to the degraded speech the listener receives,
and take intelligibility to follow the information the message and the
received speech share [VKH18a]. That is bounded twice: by what the degraded
speech shares with the clean speech, and by what the clean speech can carry
of the message at all, since no two talkers say a sentence the same way.
SIIB takes the smaller of the two. The speech is read as an auditory
log-spectrum, 28 gammatone bands from 100 to 6500 Hz with forward masking at
80 frames a second; 15 consecutive frames (187.5 ms) are stacked into one
vector of 420, and the Karhunen-Loeve transform of the reference's stacks
turns them, for the reference and the output alike, into 420 channels that
are close to independent. The information is estimated a channel at a time
and summed ([VKH18a] Eq. 11)::

    SIIB = (F / K) sum_j min(-1/2 log2(1 - r^2), I(X_j; Y_j))

with F = 80 frames a second, K = 15 frames to a stack, r = 0.75 the
production-noise correlation, and I a nearest-neighbour estimate of the
mutual information [KSG04]. Through that production noise a channel carries
0.596 bits at most, so neither measure here can pass 2240 times that,
1335.76 bit/s, which is what a copy of the reference scores. On the
authors' data SIIB came to about 150 bit/s at full intelligibility ([VKH18a]
Sec. V); a score says how intelligible speech is only through a mapping
fitted to a listening test (their Eq. 13).

:func:`siib_gauss` is SIIB^Gauss ([VKH18b] Sec. IV-C): the same channels
read as Gaussian channels, -1/2 log2(1 - r^2 rho_j^2) bits each, rho_j the
correlation of the j-th pair. It needs no estimator and no random numbers,
and did about as well as SIIB in their evaluation of 12 measures on 11
listening tests, at a hundredth of the time in MATLAB. Here it runs 15 times
faster than SIIB on 17 s of speech and 70 times faster on two minutes (see
the last note below).

The sources::

    [VKH18a] S. Van Kuyk, W. B. Kleijn and R. C. Hendriks, "An instrumental
             intelligibility metric based on information theory", IEEE
             Signal Process. Lett. 25(1), pp. 115-119, 2018.
             doi:10.1109/LSP.2017.2774250
    [VKH18b] S. Van Kuyk, W. B. Kleijn and R. C. Hendriks, "An evaluation of
             intrusive instrumental intelligibility metrics", IEEE/ACM Trans.
             Audio, Speech, Lang. Process. 26(11), pp. 2153-2166, 2018.
             doi:10.1109/TASLP.2018.2856374
    [KSG04]  A. Kraskov, H. Stögbauer and P. Grassberger, "Estimating mutual
             information", Phys. Rev. E 69(6), 066138, 2004.
             doi:10.1103/PhysRevE.69.066138
    [RVD06]  K. S. Rhebergen, N. J. Versfeld and W. A. Dreschler, "Extended
             speech intelligibility index for the prediction of the speech
             reception threshold in fluctuating noise", J. Acoust. Soc. Am.
             120(6), pp. 3988-3997, 2006. doi:10.1121/1.2358008
    [S93]    M. Slaney, "An efficient implementation of the Patterson-
             Holdsworth auditory filter bank", Apple Computer Technical
             Report 35, 1993.
    [GM90]   B. R. Glasberg and B. C. J. Moore, "Derivation of auditory filter
             shapes from notched-noise data", Hear. Res. 47(1-2),
             pp. 103-138, 1990. doi:10.1016/0378-5955(90)90170-T
    [code]   SIIB.m and SIIB_Gauss.m (Steven Van Kuyk), the MATLAB code both
             papers point to. SIIB.m leaves the estimate to MIxnyn, the C
             program in Kraskov's MILCA package for the information an X
             and a Y of any dimension share, which SIIB.m gives one column
             each.

The numbers below come from the test speech of tests/test_siib.py: 36 s of
synthetic syllables, a harmonic source through moving formants, two in five
led by a burst of hiss, with pauses on a noise floor 54 dB under the peak.
The voice-activity threshold keeps 20.6 s of it.

**Where the papers and the code differ, I follow the code**: it is what the
authors distributed with the papers and what their evaluations ran. The
differences:

* The voice activity. [VKH18a] gives an energy-based detector with a 40 dB
  threshold, applied before the STFT. SIIB.m frames the reference as the
  STFT does, measures each frame's level in dB, and keeps the frames less
  than 40 dB under the level 0.999 of the way up the sorted levels, which is
  not the loudest: with 1600 frames it is the third loudest. The frames are
  taken out of both spectrograms by the reference's decision, and the kept
  ones are joined end to end, so the masking and the stacks run across the
  gaps.
* The filterbank. [VKH18a] writes ``ln G |x_t|^2`` for Slaney's gammatone
  filterbank [S93]. SIIB.m weights the DFT bins with the closed-form
  magnitude of a 4th-order gammatone, 1 at the bin nearest each centre and
  0 where it falls under 0.001, and squares the weights for the power
  spectrum; it adds float64 eps (2.2e-16) inside the log. The centres stop
  at 6.5 kHz, and the top bands reach 8 kHz through their skirts, as
  [VKH18b] says SIIB does.
* The masking. [VKH18a] applies the forward masking of [RVD06], which falls
  to the absolute hearing threshold. SIIB.m lets each level fall to the
  quietest level the reference reaches in that band over the frames kept,
  uses that floor for the output too, and runs the fall linearly in
  log-time over 16 frames (200 ms) counted from the masker's own, so that a
  level reaches the floor 15 frames (187.5 ms) after its frame. Each frame
  takes the highest curve that reaches it.
* The last frame. SIIB.m's masking leaves the last frame kept at the floor
  in every band, whatever its level, in both signals: the curves running past
  the end are all written onto it, the floor last. That frame is in no
  stack, and it reaches the score only through the band means. On the test
  speech in white noise at -10 to 20 dB, SIIB does not feel it; SIIB^Gauss,
  whose moments are taken about zero, moves by up to 9.1e-5 of itself, at
  -10 dB, and by 1.4e-5 at 0 dB.
* The stacks. [VKH18a] Eq. 12 stacks every run of 15 frames, T - 14 of
  them out of T frames. SIIB.m forms T - 15 and leaves out the stack that
  ends on the last frame. With T - 14, SIIB on the test speech in white
  noise at -10 to 20 dB would move by 0.05% to 2.1%, SIIB^Gauss by up to
  0.13%.
* The transform. [VKH18a] Eqs. 8 and 9 take the mean of the stacks out
  before projecting them. SIIB.m takes each band's mean over the frames kept
  out before stacking, and projects the stacks as they are.
* The estimator. The papers cite [KSG04] and give no settings. SIIB.m
  standardises each channel, adds Gaussian noise of 1e-10 to both, and
  hands the pair to MIxnyn as a text file, which keeps 8 significant digits.
  MIxnyn adds uniform noise of up to 1e-8, standardises again and computes
  the second estimator of [KSG04], their Eq. 9, the one with rectangles,
  with k = N / 150 neighbours rounded up and 2 at the least, N the number of
  stacks. The answer comes back in nats to 8 decimals. Standardising and the
  noise are what [KSG04] Sec. III.A advise, the noise to break ties. SIIB.m
  also offers a MATLAB version of their first estimator; MIxnyn is its
  default, and the one its notes give for the papers' evaluations.
* The clip. A channel's estimate can be negative and stays in the sum; only
  the total is held at 0, by both measures.
* SIIB^Gauss's correlation. [VKH18b] calls rho_j the correlation
  coefficient. SIIB_Gauss.m takes it from moments about zero,
  ``(mean xy)^2 / (mean x^2 mean y^2)``, and the channels' means are not
  quite 0: with them taken out, the test speech in white noise at -10 to
  20 dB would score up to 7.5e-5 of itself more, at -10 dB.

Where I part from the code:

* The noise that breaks ties. SIIB.m draws its 1e-10 from MATLAB's global
  generator without setting it, so its score depends on what the session
  drew before (a fresh session repeats its first call), and MIxnyn draws its
  1e-8 from the C library's, seeded from the data; a score from SIIB.m
  moves a little from call to call and from platform to platform. I draw
  the same amounts from numpy's PCG64 with a fixed seed of its own, so a
  pair scores the same every time. On the test speech in white noise at
  0 dB, five other seeds moved SIIB by up to 2e-6 of itself and no noise at
  all by 4e-6, and the 8 digits of the text file move it by 2e-6. On five
  6 s clips of tests/conftest.py's make_speech, seeds 5 to 9, end to end
  under white noise at 0 dB from numpy's default_rng(101), which score
  32.9 bit/s, the seeds moved it by up to 5e-6 and no noise by 1.2e-5. So
  a difference of a few parts in 1e5, between two runs of SIIB.m or between
  it and this, is the noise; I read nothing into one under 1e-4 of the
  score. The digits and the 8 decimals of the answer are kept as MIxnyn has
  them.
* Rounding. A few things are computed in another form than SIIB.m and
  MIxnyn compute them: a masking curve as ``(1 - w) L + w floor``, which
  puts the last frame at the floor to the bit; the digamma function exactly,
  not from MIxnyn's running table; the uniform noise on [0, 1), not
  [0, 1]; no shift of each coordinate to a least value of 0 before the
  neighbour search; and an exact tie at the k-th neighbour, which only a
  degenerate pair has, settled in the order of the search tree. Each
  changes the numbers by rounding alone, about 1e-14 of the features.
* Lengths. SIIB.m refuses a pair of unequal lengths. Here the pair is cut
  to the shorter one with a warning, as every metric in kudio takes a pair,
  and refused past :data:`kudio.core.evaluator.MAX_LENGTH_MISMATCH`.
* Too little speech scores nan, with a warning. The 420 dimensions need 421
  stacks for their covariance to have full rank, which takes 436 frames of
  speech, 5.45 s. With fewer, SIIB.m still gives a number from a handful of
  stacks up, but part of it comes from eigenvectors the eigen-solver picks
  at will in the covariance's null space, so it depends on the LAPACK it
  ran on.
* A reference of digital silence, or of one value throughout, scores nan
  with a warning. SIIB.m floors the reference's deviation at eps, which
  multiplies the output by 4.5e15, and returns a small random number;
  SIIB_Gauss.m divides by zero and stops at the eigendecomposition.
* A pair of SIIB^Gauss's channels in which either has no power at all is
  taken to share nothing; SIIB_Gauss.m divides 0 by 0 there and, MATLAB's
  max passing over NaN, scores the whole pair 0. rho^2 is also held to 1,
  which an exact copy can round past.
* A rate that is not a whole number of hertz is refused. MATLAB's resample,
  which SIIB.m uses, takes the pair to 16 kHz by a ratio of whole numbers,
  and so does scipy's resample_poly, whose defaults design the same filter:
  a sinc under a Kaiser window of beta 5, ten samples of the slower rate
  either side, aligned the same way.
* A rate under 4000 Hz is refused too. It is far more likely a rate given
  in kHz than a recording, and the pair would be taken up 16000 / sr times,
  a thousand times for 16, and held in memory at that length.

What to watch for:

* It hears level. The masking floor comes from the reference and holds the
  output's spectrum as well, so an output at another level is no longer a
  copy: twice as loud, a copy of the test speech scores SIIB 1333.4 and
  SIIB^Gauss 1018.2; half as loud, 1267.3 and 880.3. In noise it matters
  less: a gain of 2 or 1/2 on the output of the test speech in white noise
  at 0 dB moves either by under 2%. A gain on both signals changes nothing.
  NCM and CSII (:mod:`kudio.core.intelligibility`) hear no level at all.
* Only where the reference speaks counts. The frames are chosen on the
  reference, and the output's frames in its pauses are dropped unseen.
  White noise added to a copy in the pauses alone, 7.5 dB louder than the
  speech, leaves SIIB at its ceiling and takes SIIB^Gauss to 811, through
  the frames at the edges of the syllables. An enhancer that fills the
  pauses with noise goes unpunished.
* It needs the pair lined up. A copy 5 ms late scores SIIB 836 and
  SIIB^Gauss 506, 10 ms late 382 and 221, 20 ms late 146 and 85, where the
  test speech in white noise at 10 dB scores 293 and 155. Use
  :func:`kudio.align` first.
* Length enters the score. The test speech in white noise at 0 dB scores
  SIIB 102.2 over its first 12 s, 110.1 over 18 s and 124.1 over all 36 s
  (SIIB^Gauss 51.8, 55.8 and 64.0). Unrelated output does not score 0:
  another talker scores 14.6, 6.5 and 7.4 over the same spans (SIIB^Gauss
  2.3, 2.2 and 1.1), and white noise 4.7 over 36 s (1.2). SIIB.m's notes
  advise 20 s of speech or more, concatenating short stimuli, as [VKH18b]
  did for each condition. Compare scores over stretches of like length.
* An output of digital silence scores 0 or a little over on 20 s of speech
  or more: its masked spectrum rises to the reference's floor over 15
  frames and stays there, which leaves its channels nearly, but not quite,
  constant. Against the test speech it scores SIIB 0.47 or 0.48, as the
  numpy build rounds those channels, and SIIB^Gauss 0.55. Over the test
  speech's first 18 s SIIB's channel estimates sum to
  less than 0 and it scores 0, where SIIB^Gauss, none of whose channels can
  go under 0, scores 1.16. A shorter pair gives more: over the first 12 s,
  SIIB 2.19 and SIIB^Gauss 1.77, and against the 12 s of the first pair of
  tools/validation/siib.py, SIIB 7.6 to 7.7 and SIIB^Gauss 1.72. Most of
  SIIB there is the noise that breaks ties on channels that are nearly
  constant, and it moves by 0.1 with the numpy it runs on.
* Nothing takes DC out. An offset passes the Hann window at 0 and 40 Hz,
  into the seven lowest bands, and adds to every frame's level, which can
  carry the reference's pauses over the 40 dB threshold. An offset of
  0.005 on both signals of the test speech in white noise at 0 dB, 18 dB
  under the speech's RMS, keeps all 2878 frames where 1645 were kept and
  takes SIIB from 124.1 to 72.8 and SIIB^Gauss from 64.0 to 36.8. On the
  output alone it reaches only the seven lowest bands, and ten times that
  moves them by 0.1% and 0.2%. Take DC out first if a recording has it.
* Other rates are taken to 16 kHz first. Taken to 8 kHz, the test speech in
  white noise at 0 dB scores SIIB 122.4 and SIIB^Gauss 63.8, and to 44.1 kHz
  124.9 and 63.9, against 124.1 and 64.0 at 16 kHz. At 8 kHz the bands over
  4 kHz hold only what the resampler lets through, in both signals alike.
  SIIB depends on that resampler by a few bit/s. I take the pair to 16 kHz
  with the filter SIIB.m's resample designs, 101 taps from 20 kHz and 41
  from 8 kHz. Through a sharper one, a sinc under a Kaiser window for 60 dB
  (beta 5.65) of 365 taps, the test speech at 20 kHz in white noise at
  10 dB scores SIIB 324.0 instead of 321.3 and SIIB^Gauss 169.37 instead of
  169.40. tools/validation/siib.py runs the sweep: the test speech taken to
  20 kHz and to 8 kHz by a near-ideal filter (601 taps, Kaiser beta 10),
  white noise added at that rate (default_rng(7)) at -10 to 20 dB in 5 dB
  steps, and each pair scored through SIIB.m's filter and through the
  sharper one, 147 taps from 8 kHz. SIIB moved by up to 5.1 bit/s, at 8 kHz
  and 20 dB, and SIIB^Gauss by up to 1.6; with the 8 kHz speech made by
  resample_poly's own filter instead, by up to 1.9 and 0.9. So the spread
  depends on the signal as well, and a score from an implementation that
  resamples otherwise can differ from this one by a few bit/s at any rate
  but 16 kHz. pySIIB, the Python port, is one: over that sweep its scores
  are those of the sharper filter to 1e-3 bit/s. At 16 kHz it gives this
  module's SIIB^Gauss to under 1e-12 of itself and its SIIB to 1.4e-5 at
  most, the size of the noise that breaks ties.
* Time. The neighbour count grows with the speech, so SIIB's time grows
  faster than the speech, SIIB^Gauss's about as the speech. On one desktop
  CPU, SIIB took 1.1 s for 17 s of speech, 2.8 s for 31 s and 29 s for two
  minutes; SIIB^Gauss took 0.08, 0.11 and 0.42 s, 15 to 70 times less.
"""
from __future__ import annotations

import functools
import math
from typing import Optional, Tuple

import numpy as np

from kudio.core import evaluator as _evaluator
from kudio.core._framing import frame_view
from kudio.core._policy import sample_rate, warn

__all__ = ['siib', 'siib_gauss']

_EPS = float(np.finfo(np.float64).eps)

# -- the auditory representation ---------------------------------------------------

#: Everything is scored at 16 kHz, in 25 ms Hann frames every 12.5 ms: 80
#: frames a second ([VKH18a] Sec. III).
_RATE = 16000
_FRAME = 400
_HOP = 200

#: The lowest rate taken: under it, more than 4 times up to 16 kHz, a rate is
#: far more likely one given in kHz than a recording.
_LOWEST_RATE = 4000
_FRAMES_PER_SECOND = _RATE / _HOP

#: The voice-activity threshold, in dB under the loud end of the reference's
#: frame levels, and where SIIB.m reads that loud end: 0.999 of the way up
#: the sorted levels.
_VAD_RANGE_DB = 40.0
_VAD_QUANTILE = 0.999

#: Gammatone bands centred on equal steps of [GM90]'s ERB-rate scale from
#: 100 to 6500 Hz, as many as the span holds whole ERBs: 28.
_LOW_HZ = 100.0
_HIGH_HZ = 6500.0
_ORDER = 4

#: Where a band's amplitude response is cut to 0, against its peak.
_WEIGHT_FLOOR = 1e-3

#: Forward masking after [RVD06], in frames: T0 = 1 and Tf = 16 (200 ms),
#: the masker's own frame counted as the first, so that a level reaches the
#: floor 15 frames (187.5 ms) after its frame.
_MASK_START = 1
_MASK_LENGTH = int(math.floor(0.2 * _FRAMES_PER_SECOND))

#: Frames to a stack: 187.5 ms, against 80 ms for the mean phoneme
#: ([VKH18a] Sec. III).
_STACK = 15

# -- the information ------------------------------------------------------------------

#: The production-noise correlation, the same for every channel, and the
#: information a channel carries through it at most, -1/2 log2(1 - r^2)
#: bits ([VKH18a] Eq. 10). Taken through numpy's log2, as siib_gauss() takes
#: its channels, so that a copy reaches the same ceiling by either measure.
_R_P = 0.75
_I_MAX = float(-0.5 * np.log2(1.0 - _R_P ** 2))

#: The estimator's neighbours: a 150th of the stacks, rounded up, and 2 at
#: the least, as SIIB.m has it. The 2 only counts under 301 stacks, and
#: nothing under 421 is scored here.
_NEIGHBOUR_SHARE = 150
_MIN_NEIGHBOURS = 2

#: The noise SIIB.m and MIxnyn add to break ties, against a channel's
#: standard deviation, and the seed kudio draws it with.
_DITHER_NORMAL = 1e-10
_DITHER_UNIFORM = 1e-8
_SEED = 0

#: Significant digits a sample keeps in the text file MIxnyn reads, and
#: decimals the estimate keeps on the way back.
_TEXT_DIGITS = 8
_ANSWER_DECIMALS = 8

#: The fewest frames of speech scored: the 420 dimensions of a stack need
#: 421 stacks for a covariance of full rank, and 421 stacks take 436 frames.
_DIMENSIONS = 28 * _STACK
_MIN_FRAMES = _DIMENSIONS + 1 + _STACK

#: Frames windowed and transformed at a time, so that the spectra of a long
#: pair are never all held at once.
_BLOCK = 4096


# --- input ------------------------------------------------------------------------

def _paired(ref, deg, name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Both signals as one channel of float64 each, cut to the length they
    share, as every metric of :mod:`kudio.core.evaluator` takes them: a
    mismatch warns, and past ``kudio.core.evaluator.MAX_LENGTH_MISMATCH`` of
    the longer one it is refused."""
    return _evaluator._align(ref, deg, caller=name)


def _whole_rate(sr, name: str) -> int:
    rate = sample_rate(sr, name)
    if rate < _LOWEST_RATE:
        raise ValueError(
            f"{name}(): sr is in hertz, got {rate}; SIIB refuses a rate under "
            f"{_LOWEST_RATE} Hz, which it would have to take {_RATE / rate:g} "
            f"times up to 16 kHz")
    if not isinstance(rate, int):
        raise ValueError(
            f"{name}(): SIIB takes the pair to 16 kHz by a ratio of whole "
            f"numbers, as SIIB.m does with MATLAB's resample, so the rate has "
            f"to be a whole number of hertz, got {rate}")
    return rate


def _ratio(rate: int) -> Tuple[int, int]:
    """``(up, down)`` from *rate* to 16 kHz, in lowest terms."""
    g = math.gcd(_RATE, rate)
    return _RATE // g, rate // g


def _min_samples(rate: int) -> int:
    """The shortest pair at *rate* that gives :data:`_MIN_FRAMES` frames at
    16 kHz, the resampled length being ``ceil(n * up / down)``."""
    up, down = _ratio(rate)
    at_16k = _FRAME + (_MIN_FRAMES - 1) * _HOP + 1
    return ((at_16k - 1) * down) // up + 1


# --- the auditory representation ----------------------------------------------------

def _erb_rate(f):
    """[GM90]'s ERB-rate scale, in ERBs, at *f* in Hz."""
    return 21.4 * np.log10(1.0 + 4.37 * np.asarray(f, dtype=float) / 1000.0)


@functools.lru_cache(maxsize=1)
def _filterbank() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(centres, widths, weights)``: the band centres and bandwidths in Hz,
    and the ``(28, 201)`` weights the power spectrum is summed with."""
    low, high = float(_erb_rate(_LOW_HZ)), float(_erb_rate(_HIGH_HZ))
    bands = int(math.floor(high - low + 0.5))
    centres = 1000.0 * (10.0 ** (np.linspace(low, high, bands) / 21.4) - 1.0) / 4.37
    # the bandwidth parameter of an order-n gammatone whose equivalent
    # rectangular bandwidth is one ERB is (n-1)!^2 / (pi (2n-2)! 2^-(2n-2))
    # ERB; for n = 4 that is 3!^2 / (pi 6! 2^-6) = 3.2 / pi, the 1.019 of
    # [S93] written exactly
    widths = 3.2 / math.pi * 24.7 * (1.0 + 4.37 * centres / 1000.0)
    bins = np.arange(_FRAME // 2 + 1) * (_RATE / _FRAME)
    # the magnitude of an order-n resonance, its positive-frequency term
    # alone, and 1 at the bin nearest the centre rather than at the centre
    response = (widths[:, None] ** 2
                + (bins[None, :] - centres[:, None]) ** 2) ** (-_ORDER / 2.0)
    amplitude = response / response.max(axis=1, keepdims=True)
    amplitude[amplitude < _WEIGHT_FLOOR] = 0.0
    weights = amplitude ** 2
    for a in (centres, widths, weights):
        a.flags.writeable = False
    return centres, widths, weights


@functools.lru_cache(maxsize=1)
def _window() -> np.ndarray:
    """The periodic Hann window of 400 points: 0 at the first, 1 at the
    201st."""
    w = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(_FRAME) / _FRAME)
    w.flags.writeable = False
    return w


def _masking_weights() -> np.ndarray:
    """How far a level has fallen towards the floor after each lag,
    ``ln((lag + 1) / T0) / ln(Tf / T0)``: 0 at lag 0, 1 at lag 15."""
    lag = np.arange(_MASK_LENGTH)
    return np.log((lag + 1) / _MASK_START) / math.log(_MASK_LENGTH / _MASK_START)


def _frame_count(n: int) -> int:
    """Frames SIIB.m takes from *n* samples at 16 kHz: one for each start
    ``200 t`` with ``200 t + 400 <= n - 1``, so the last sample is in none."""
    return 0 if n <= _FRAME else (n - _FRAME - 1) // _HOP + 1


def _vad_rank(count: int) -> int:
    """The place, counted from 1 and from the quietest, of the frame level the
    40 dB is measured from: 0.999 of *count*, halves rounded up as MATLAB's
    round has them."""
    return int(math.floor(_VAD_QUANTILE * count + 0.5))


def _band_levels(x: np.ndarray, y: np.ndarray):
    """``(levels, bx, by)``: each frame's level in dB, from *x*, and the band
    powers of both signals, ``(frames, 28)`` each."""
    weights = _filterbank()[2]
    window = _window()
    count = _frame_count(len(x))
    fx = frame_view(x[:len(x) - 1], _FRAME, _HOP)[:count]
    fy = frame_view(y[:len(y) - 1], _FRAME, _HOP)[:count]
    levels = np.empty(count)
    bx = np.empty((count, weights.shape[0]))
    by = np.empty((count, weights.shape[0]))
    # an output too loud to square overflows here; the caller checks
    with np.errstate(over='ignore', invalid='ignore'):
        for start in range(0, count, _BLOCK):
            stop = min(start + _BLOCK, count)
            wx = fx[start:stop] * window
            wy = fy[start:stop] * window
            levels[start:stop] = np.mean(wx * wx, axis=1)
            bx[start:stop] = (np.abs(np.fft.rfft(wx, axis=1)) ** 2) @ weights.T
            by[start:stop] = (np.abs(np.fft.rfft(wy, axis=1)) ** 2) @ weights.T
    return 10.0 * np.log10(levels + _EPS), bx, by


def _active(levels: np.ndarray) -> np.ndarray:
    """The frames that hold speech: those less than 40 dB under the level at
    :func:`_vad_rank`."""
    loud = np.sort(levels)[_vad_rank(len(levels)) - 1]
    return levels > loud - _VAD_RANGE_DB


def _masked(log_power: np.ndarray, floor: np.ndarray) -> np.ndarray:
    """Forward masking of ``(frames, 28)`` log band powers.

    Each frame casts a curve over itself and the 15 frames after it, falling
    from its own level to *floor* as ``(1 - w) L + w floor``, and each frame
    takes the highest curve that reaches it.
    """
    weights = _masking_weights()
    count = len(log_power)
    out = log_power.copy()                     # lag 0, the frame's own level
    for lag in range(1, min(_MASK_LENGTH, count)):
        w = weights[lag]
        np.maximum(out[lag:], (1.0 - w) * log_power[:count - lag] + w * floor,
                   out=out[lag:])
    # SIIB.m writes the curves running past the end onto the last frame, the
    # lag of 15, which is the floor itself, last
    out[-1] = floor
    return out


def _stacked(features: np.ndarray) -> np.ndarray:
    """``(frames - 15, 420)``: 15 consecutive frames to a row, oldest first
    and band fastest, without the stack that ends on the last frame."""
    rows = len(features) - _STACK
    return np.concatenate([features[m:m + rows] for m in range(_STACK)], axis=1)


def _eigenchannels(ref, deg, sr, name: str,
                   floor_deviation: bool) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Both signals' eigenchannels, ``(N, 420)`` each, or None once a warning
    has said why the pair cannot be scored.

    :param floor_deviation: hold the reference's deviation to eps or more
        before dividing by it, as SIIB.m does and SIIB_Gauss.m does not.
    """
    rate = _whole_rate(sr, name)
    ref, deg = _paired(ref, deg, name)
    need = _min_samples(rate)
    if len(ref) < need:
        warn(f"{name}(): the pair is {len(ref)} samples ({len(ref) / rate:.2f} s "
             f"at {rate} Hz), and SIIB needs at least {need} ({need / rate:.2f} s) "
             f"for {_MIN_FRAMES} frames of speech, the fewest over which the "
             f"covariance of {_STACK} stacked frames has full rank; concatenate "
             f"utterances, 20 s of speech or more as the authors advise; "
             f"scored nan")
        return None
    if not np.any(ref):
        warn(f"{name}(): ref is digital silence throughout, so there is no "
             f"speech to compare deg with; scored nan")
        return None
    if np.ptp(ref) == 0.0:
        warn(f"{name}(): ref holds one value throughout, so there is no "
             f"speech to compare deg with; scored nan")
        return None

    # a power of two scales exactly, so a reference too loud or too quiet to
    # square gets its deviation, and to the bit the one it would have had
    _, exponent = math.frexp(float(np.max(np.abs(ref))))
    x = np.ldexp(ref, -exponent)
    y = np.ldexp(deg, -exponent)
    deviation = float(np.std(x, ddof=1))
    if floor_deviation:
        deviation = max(deviation, math.ldexp(_EPS, -exponent))
    x = x / deviation
    y = y / deviation
    if rate != _RATE:
        from scipy.signal import resample_poly
        up, down = _ratio(rate)
        x = resample_poly(x, up, down)
        y = resample_poly(y, up, down)

    levels, bx, by = _band_levels(x, y)
    active = _active(levels)
    kept = int(np.count_nonzero(active))
    if kept < _MIN_FRAMES:
        warn(f"{name}(): {kept} of the pair's {len(levels)} frames hold speech by "
             f"the {_VAD_RANGE_DB:g} dB threshold on ref "
             f"({kept / _FRAMES_PER_SECOND:.2f} s), and SIIB needs at least "
             f"{_MIN_FRAMES} ({_MIN_FRAMES / _FRAMES_PER_SECOND:.2f} s), the "
             f"fewest over which the covariance of {_STACK} stacked frames has "
             f"full rank; concatenate utterances, 20 s of speech or more as the "
             f"authors advise; scored nan")
        return None
    bx, by = bx[active], by[active]
    if not np.all(np.isfinite(by)):
        warn(f"{name}(): deg is too loud against ref to square in float64; "
             f"scale it down; scored nan")
        return None
    lx = np.log(bx + _EPS)
    ly = np.log(by + _EPS)
    # one floor a band, the quietest the reference gets there, for both
    floor = lx.min(axis=0)
    mx = _masked(lx, floor)
    my = _masked(ly, floor)
    sx = _stacked(mx - mx.mean(axis=0))
    sy = _stacked(my - my.mean(axis=0))
    _, basis = np.linalg.eigh(np.cov(sx, rowvar=False))
    # projected one at a time, so that a copy's channels come out the same
    # to the bit as the reference's
    return sx @ basis, sy @ basis


# --- the information ------------------------------------------------------------------

def _rate_of(information: np.ndarray) -> float:
    """Bits per second from bits per channel: the sum, for stacks of 15
    frames at 80 frames a second, and 0 at the least."""
    return max(0.0, _FRAMES_PER_SECOND / _STACK * float(np.sum(information)))


def _neighbours(count: int) -> int:
    return max(_MIN_NEIGHBOURS, -(-count // _NEIGHBOUR_SHARE))


def _standardised(a: np.ndarray, guard: float = 0.0) -> np.ndarray:
    """Each column at zero mean and unit sample deviation, *guard* added to
    the deviation."""
    a = a - a.mean(axis=0)
    return a / (a.std(axis=0, ddof=1) + guard)


def _as_text(a: np.ndarray) -> np.ndarray:
    """*a* to 8 significant digits, as the text file holds it."""
    out = np.array(a, dtype=float)
    some = out != 0.0
    size = np.abs(out[some])
    exponent = np.floor(np.log10(size))
    # log10 can land a hair either side of a power of ten
    exponent = np.where(size >= 10.0 ** (exponent + 1), exponent + 1, exponent)
    exponent = np.where(size < 10.0 ** exponent, exponent - 1, exponent)
    step = 10.0 ** (_TEXT_DIGITS - 1 - exponent)
    # an integer over a power of ten, both exact, divides to the nearest
    # double, which is what reading the digits back gives
    out[some] = np.round(out[some] * step) / step
    return out


def _within(values: np.ndarray, radius: np.ndarray) -> np.ndarray:
    """For each i, how many j other than i have ``|values[j] - values[i]| <=
    radius[i]``, the distance taken as the neighbour search takes it."""
    s = np.sort(values)
    n = len(s)
    lo = np.searchsorted(s, values - radius, side='left')
    hi = np.searchsorted(s, values + radius, side='right')

    def inside(j):
        return np.abs(s[np.clip(j, 0, n - 1)] - values) <= radius

    # values +/- radius is rounded, which can put an end one place from where
    # the distance itself puts it; move each end until the two agree
    while True:
        grow_hi = (hi < n) & inside(hi)
        shrink_hi = ~grow_hi & (hi > 0) & ~inside(hi - 1)
        grow_lo = (lo > 0) & inside(lo - 1)
        shrink_lo = ~grow_lo & (lo < n) & ~inside(lo)
        if not (grow_hi.any() or shrink_hi.any() or grow_lo.any() or shrink_lo.any()):
            return hi - lo - 1
        hi = hi + grow_hi - shrink_hi
        lo = lo - grow_lo + shrink_lo


def _ksg(u: np.ndarray, v: np.ndarray, k: int) -> float:
    """The mutual information of two sequences in nats, by the second
    estimator of [KSG04], their Eq. 9.

    For each point, its k nearest others in the maximum norm span a rectangle
    of half-widths eu and ev; nu and nv count the other points within eu of
    it along u and within ev along v, ends included. Then
    ``I = psi(k) - 1/k - <psi(nu) + psi(nv)> + psi(N)``.
    """
    from scipy.spatial import cKDTree
    from scipy.special import digamma

    n = len(u)
    points = np.column_stack([u, v])
    _, found = cKDTree(points).query(points, k=k + 1, p=np.inf)
    # a point is the first one it finds unless another sits on top of it;
    # drop it wherever it is, and the farthest where it is missing
    own = found == np.arange(n)[:, None]
    if own[:, 0].all():
        others = found[:, 1:]
    else:
        order = np.argsort(own, axis=1, kind='stable')[:, :k]
        others = np.take_along_axis(found, order, axis=1)
    eu = np.max(np.abs(u[others] - u[:, None]), axis=1)
    ev = np.max(np.abs(v[others] - v[:, None]), axis=1)
    nu = _within(u, eu)
    nv = _within(v, ev)
    return float(digamma(n) + digamma(k) - 1.0 / k
                 - np.mean(digamma(nu) + digamma(nv)))


def _channel_bits(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Each pair of eigenchannels' mutual information in bits, taken the way
    SIIB.m takes it through MIxnyn."""
    n, d = x.shape
    k = _neighbours(n)
    rng = np.random.default_rng(_SEED)
    u = _standardised(x, _EPS) + _DITHER_NORMAL * rng.standard_normal((n, d))
    v = _standardised(y, _EPS) + _DITHER_NORMAL * rng.standard_normal((n, d))
    u = _as_text(u) + _DITHER_UNIFORM * rng.random((n, d))
    v = _as_text(v) + _DITHER_UNIFORM * rng.random((n, d))
    u = np.ascontiguousarray(_standardised(u).T)
    v = np.ascontiguousarray(_standardised(v).T)
    nats = np.array([_ksg(u[j], v[j], k) for j in range(d)])
    return np.round(nats, _ANSWER_DECIMALS) / math.log(2.0)


def siib(ref: np.ndarray, deg: np.ndarray, sr: int) -> float:
    """Speech intelligibility in bits of *deg* against *ref*: the information
    they share, in bits per second, from 0 to 1335.76; higher is more
    intelligible.

    >>> kudio.siib(clean, enhanced, 16000)                       # doctest: +SKIP

    How it is computed, as SIIB.m computes it:

    1. Both signals are divided by the reference's standard deviation and
       taken to 16 kHz.
    2. The reference is cut into 25 ms Hann frames every 12.5 ms, and the
       frames less than 40 dB under its loud end are kept, in both signals.
    3. Each kept frame's power spectrum is summed into 28 gammatone bands from
       100 to 6500 Hz and its log taken. Forward masking lets each level fall
       to the quietest the reference gets in that band, which it reaches 15
       frames (187.5 ms) later.
    4. Each band loses its mean, 15 consecutive frames are stacked into a
       vector of 420, and both signals' stacks are projected onto the
       eigenvectors of the reference's covariance, 420 channels each.
    5. The mutual information of each pair of channels is estimated with the
       second estimator of [KSG04] and held to 0.596 bits, the most the
       production noise lets through. The sum, times 80 frames a second over
       15 frames a stack, is the score.

    The module notes list where this parts from [VKH18a] and from SIIB.m,
    and what to watch for: it hears the output's level, sees the output only
    where the reference speaks, and gives more for longer pairs. 30 s of
    speech, counted once the silence is dropped, takes about 3 s, and two
    minutes take half a minute.

    :param ref: the clean speech, mono. 5.45 s of speech at the least once
        its silence is dropped, and 20 s or more as the authors advise.
        Integer audio is read on its type's full scale.
    :param deg: the processed or degraded speech, mono and lined up with
        *ref*.
    :param sr: the rate of both, a whole number of hertz from 4000 up; any
        rate but 16 kHz is resampled.
    :raises ValueError: for a rate that is not a whole number of hertz from
        4000 up (a rate under 4000 Hz is taken for one given in kHz), for a
        signal with more than one channel (one held as ``(n, 1)`` or
        ``(1, n)`` is read as mono), for a single number, for samples that
        are complex, not numbers or not finite, and for a pair too different
        in length (see :data:`kudio.core.evaluator.MAX_LENGTH_MISMATCH`; a
        smaller difference warns, and the end of the longer one is left
        out).
    :returns: bits per second. A copy of the reference scores 1335.76, the
        ceiling, and an output of digital silence 0 or a little over on 20 s
        of speech, more on a shorter pair: about 7.7 against the 12 s of the
        first pair of tools/validation/siib.py. nan, with a warning, for a
        reference that is digital silence or one value throughout, for a
        pair with fewer than 436 frames of speech (5.45 s) once the
        reference's silence is dropped, and for an output too loud against
        the reference to square.
    """
    channels = _eigenchannels(ref, deg, sr, 'siib', floor_deviation=True)
    if channels is None:
        return float('nan')
    return _rate_of(np.minimum(_I_MAX, _channel_bits(*channels)))


def siib_gauss(ref: np.ndarray, deg: np.ndarray, sr: int) -> float:
    """SIIB^Gauss of *deg* against *ref*: SIIB with each pair of
    eigenchannels read as a Gaussian channel, in bits per second, from 0 to
    1335.76; higher is more intelligible.

    >>> kudio.siib_gauss(clean, enhanced, 16000)                 # doctest: +SKIP

    Steps 1 to 4 are those of :func:`siib`. Then, as SIIB_Gauss.m computes
    it, each pair of channels x and y gives ``rho^2 = (mean xy)^2 / (mean x^2
    mean y^2)``, from moments about zero, and ``-1/2 log2(1 - 0.75^2
    rho^2)`` bits; the sum, times 80 frames a second over 15 frames a stack,
    is the score.

    No random numbers enter it, it is a smooth function of the signals, and
    it takes a fraction of SIIB's time. It reads less than SIIB for the same
    pair, about half on the test speech in white noise, and the two are not
    interchangeable: each has its own mapping to intelligibility. The module
    notes say what to watch for, which holds for both.

    :param ref: the clean speech, mono. 5.45 s of speech at the least once
        its silence is dropped, and 20 s or more as the authors advise.
        Integer audio is read on its type's full scale.
    :param deg: the processed or degraded speech, mono and lined up with
        *ref*.
    :param sr: the rate of both, a whole number of hertz from 4000 up; any
        rate but 16 kHz is resampled.
    :raises ValueError: as :func:`siib` raises it.
    :returns: bits per second. A copy of the reference scores 1335.76, the
        ceiling, and an output of digital silence a little over 0 on 20 s of
        speech, more on a shorter pair: 1.72 against the 12 s of the first
        pair of tools/validation/siib.py. nan, with a warning, in the cases
        :func:`siib` gives nan.
    """
    channels = _eigenchannels(ref, deg, sr, 'siib_gauss', floor_deviation=False)
    if channels is None:
        return float('nan')
    x, y = channels
    xy = np.mean(x * y, axis=0)
    xx = np.mean(x * x, axis=0)
    yy = np.mean(y * y, axis=0)
    # a channel with no power shares nothing; SIIB_Gauss.m divides 0 by 0
    with np.errstate(divide='ignore', invalid='ignore'):
        rho2 = np.where(xx * yy > 0.0, xy * xy / (xx * yy), 0.0)
    rho2 = np.clip(rho2, 0.0, 1.0)
    information = -0.5 * np.log2(1.0 - _R_P ** 2 * rho2)
    return _rate_of(np.minimum(_I_MAX, information))
