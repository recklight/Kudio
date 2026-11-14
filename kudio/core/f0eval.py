# -*- coding: utf-8 -*-
"""Pitch-tracking errors: how far an estimated F0 track is from a reference
track on the same frames.

These are the numbers pitch trackers are compared by on corpora recorded with
a laryngograph, such as Keele and PTDB-TUG, and the F0 RMSE that speech
synthesis papers report. Each function takes two tracks, one F0 value per
frame in Hz with 0 or nan where the frame is unvoiced, or two
:class:`kudio.PitchTrack` objects, and compares frame i of one with frame i
of the other. Every frame falls in one of four cases ([R76], p. 407):
unvoiced in both; voiced in the reference only, a V/U error; voiced in the
estimate only, a U/V error; or voiced in both, where the estimate is either
close or a gross error. With N frames, N_VV of them voiced in both and N_GE
gross errors among those::

    function                    what it takes                       over
    gross_pitch_error      GPE  N_GE: |est - ref| > 0.2 ref         N_VV
    voicing_decision_error VDE  N_VU + N_UV                         N
    f0_frame_error         FFE  N_VU + N_UV + N_GE                  N
    fine_pitch_error       FPE  SD of (est - ref) / ref             N_VV - N_GE
    f0_rmse                     RMS of 1200 log2(est / ref), cents  N_VV

GPE, VDE and FFE are those of Chu and Alwan [CA09], eqs. 1 to 4, and
FFE = VDE + (N_VV / N) GPE, their eq. 5. FPE is Drugman and Alwan's [DA11]:
the standard deviation of the relative error over the frames voiced in both
without a gross error. The first three come back as fractions, and FPE as a
fraction of the reference F0; times 100 they are the percentages the papers
print::

    >>> ref = [0, 100, 102, 104, 0, 0, 120, 121]      # Hz, 0 where unvoiced
    >>> est = [0, 101, 204, 0, 0, 98, 119, 0]
    >>> kudio.gross_pitch_error(ref, est)     # frame 2 an octave up, of 3
    0.3333333333333333
    >>> kudio.voicing_decision_error(ref, est)   # frames 3, 5 and 7, of 8
    0.375
    >>> kudio.f0_frame_error(ref, est)           # all four, of 8
    0.5

:func:`f0_errors` returns all five at once, with the counts behind them.

Sources, and what each one counts::

    [R76]   L. R. Rabiner, M. J. Cheng, A. E. Rosenberg and C. A. McGonegal,
            "A comparative performance study of several pitch detection
            algorithms", IEEE Trans. Acoust., Speech, Signal Process. 24(5),
            pp. 399-418, 1976, doi:10.1109/TASSP.1976.1162846. A gross error
            is a pitch-period error of 1 ms or more (10 samples at their
            10 kHz), counted per utterance: at 100 Hz an estimate 9.1% low or
            11.1% high, at 200 Hz one 16.7% low or 25% high. The fine errors,
            the rest, are measured by their mean, which came out near 0 for
            every detector and is not tabulated, and by their standard
            deviation, in samples; V/U errors as a share of the reference's
            voiced frames, and U/V errors in the same form.
    [NIZ03] T. Nakatani, T. Irino and P. Zolfaghari, "Dominance spectrum
            based V/UV classification and F0 estimation", Proc. Eurospeech
            2003, pp. 2313-2316, doi:10.21437/Eurospeech.2003-210. A gross
            error is an estimate more than 5% off, and the fine error is the
            RMS of the relative error over the other frames.
    [CA09]  W. Chu and A. Alwan, "Reducing F0 frame error of F0 tracking
            algorithms under noisy conditions with an unvoiced/voiced
            classification frontend", Proc. ICASSP 2009, pp. 3969-3972,
            doi:10.1109/ICASSP.2009.4960497. GPE over N_VV, its threshold
            usually 20%, and VDE and FFE over all N frames.
    [DA11]  T. Drugman and A. Alwan, "Joint robust voicing detection and
            pitch estimation based on residual harmonics", Proc. Interspeech
            2011, pp. 1973-1976, doi:10.21437/Interspeech.2011-519. GPE, VDE
            and FFE as in [CA09], and FPE as the standard deviation, in
            percent, of the relative error of the frames under 20%.
    [CA12]  W. Chu and A. Alwan, "SAFE: a statistical approach to F0
            estimation under clean and noisy conditions", IEEE Trans. Audio,
            Speech, Lang. Process. 20(3), pp. 933-944, 2012,
            doi:10.1109/TASL.2011.2168518. GPE as in [CA09], with every
            tracker's voicing replaced by the reference's: a frame the
            reference calls voiced and the tracker unvoiced gets an F0
            interpolated from the nearest frames both call voiced (their
            eq. 20), and a frame the tracker calls voiced against the
            reference is set to 0. The fine errors are a mean (MFPE, eq. 23)
            and a root mean square about zero (SDFPE, eq. 24, though they
            call it a standard deviation), in Hz.
    [CH08]  A. Camacho and J. G. Harris, "A sawtooth waveform inspired pitch
            estimator for speech and music", J. Acoust. Soc. Am. 124(3),
            pp. 1638-1652, 2008, doi:10.1121/1.2951592, sec. III.C. A gross
            error rate with the 20% rule, counted only at the times the
            reference and every compared algorithm that gave an estimate
            every millisecond called the sound pitched, and on the corpora
            whose reference moves, after shifting each estimate by up to
            100 ms to its best alignment with it.
    [P11]   G. Pirker, M. Wohlmayr, S. Petrik and F. Pernkopf, "A pitch
            tracking corpus with evaluation on multipitch tracking scenario",
            Proc. Interspeech 2011, pp. 1509-1512,
            doi:10.21437/Interspeech.2011-317. PTDB-TUG. Its benchmark, two
            talkers at once, counts gross errors (over 20%) as a share of all
            frames, and cuts each utterance 70 ms before its first and after
            its last reference pitch value.
    [PMA95] F. Plante, G. F. Meyer and W. A. Ainsworth, "A pitch extraction
            reference database", Proc. Eurospeech 1995, pp. 837-840,
            doi:10.21437/Eurospeech.1995-191. Keele: a reference value every
            10 ms from the laryngograph, 0 where unvoiced and negative where
            the frame is uncertain, and uncertain frames are to be left out
            of comparisons.
    [K18]   J. W. Kim, J. Salamon, P. Li and J. P. Bello, "CREPE: a
            convolutional representation for pitch estimation", Proc. ICASSP
            2018, pp. 161-165, doi:10.1109/ICASSP.2018.8461329. Raw pitch
            accuracy within 50 cents, and within 25 and 10, from mir_eval.
    [R14]   C. Raffel, B. McFee, E. J. Humphrey, J. Salamon, O. Nieto,
            D. Liang and D. P. W. Ellis, "mir_eval: a transparent
            implementation of common MIR metrics", Proc. ISMIR 2014,
            pp. 367-372, doi:10.5281/zenodo.1416528. The MIREX melody
            measures: raw pitch accuracy over the reference's voiced frames,
            within half a semitone; voicing recall and false alarm; and
            overall accuracy over all frames.

Where the papers leave a choice open, I made it this way. The 20% is
measured against the reference and is strict, as [CA09]'s eq. 3 writes it,
so an estimate exactly 20% off is not a gross error: 120 Hz against 100 is
not. Exactly means in the binary values the arrays hold. Most decimals have
no exact binary form, and the comparison is rounded like any other, so two
decimals 20% apart can land either side: 120.12 Hz against 100.1 is a gross
error, 120.24 against 100.2 is not. FPE divides by the number of fine
frames, as [R76]'s eq. 3 does for its standard deviation; [DA11] does not
say. The RMSE counts every frame voiced in both, gross errors included. On
tracks built so that the 20% rule and a cent tolerance draw the same line,
mir_eval 0.8.2 gives the same numbers: 1 - overall_accuracy is FFE,
1 - raw_pitch_accuracy over the frames voiced in both is GPE, and with an
unlimited cent tolerance (``cent_tolerance=np.inf``) 1 - overall_accuracy
is VDE.
``tools/validation/f0eval.py`` repeats that check on the tests' pair, 7
pairs at the edges of what can be scored and 400 random ones, and checks
every measure against a frame-by-frame implementation of its own as well.
With numpy 2.3.5 the rates agree to 1.1e-16 and the RMSE to 4.5e-13 cents.

What to watch for:

* The denominators are where papers part. GPE here is over N_VV. [CA12]
  replaced every tracker's voicing with the reference's, interpolating an F0
  where a tracker gave none, [CH08] counted only where the reference and
  every compared algorithm called the sound pitched,
  and [P11] counts gross errors over all N frames, which is (N_VV / N)
  times the GPE here. The same two tracks give a different number under
  each; say which one a table holds.
* GPE on its own rewards a tracker for giving up. Calling a hard frame
  unvoiced takes it out of N_VV: turning one gross error into a V/U error
  lowers GPE, adds 1/N to VDE and leaves FFE where it was, which is why
  [CA09] proposed FFE. Report GPE with VDE, or FFE.
* VDE and FFE count every frame, so frames that both tracks call unvoiced
  dilute them: 300 frames with 30 voicing errors read 0.10, and with 100
  frames of silence added at each end, 0.06. [P11] cut its utterances to
  70 ms around the first and last pitch value. Trim every system's tracks
  the same way.
* Pooling and averaging are different numbers. Concatenate the tracks of a
  test set and score them once to pool their frames; the mean of per-file
  scores weighs a short file like a long one. A file with 5 gross errors in
  10 frames voiced in both and one with 10 in 1000 average to a GPE of
  0.255 and pool to 0.0149.
* 20% is 315.6 cents above the reference and 386.3 below it. Raw pitch
  accuracy, as MIREX and mir_eval score it [R14], takes a symmetric
  tolerance, 50 cents by default (2.9% up, 2.8% down), counts over the
  reference's voiced frames and reads an estimate's pitch whether or not it
  is called voiced, so 1 - RPA is not GPE.
* Negative values are refused, because they mean two things. In Keele's
  reference [PMA95] they mark uncertain frames, which are to be dropped from
  both tracks (``keep = ref >= 0``). In MIREX files a negative estimate is an
  unvoiced frame with a pitch guess, which ``np.maximum(est, 0)`` reads as
  unvoiced. A masked array with masked frames is refused for the same
  reason: ``np.ma.filled(est, 0)`` reads them as unvoiced, and
  ``keep = ~np.ma.getmaskarray(est)`` drops them from both tracks.
* Nothing is shifted. A lag of one 10 ms frame on a voice gliding 2 octaves
  per second puts every frame 24 cents off, and [CH08] shifted the estimates
  by up to 100 ms to their best alignment before counting. Synthesized
  speech is seldom on the reference's frames at all and has to be lined up
  with it first, by DTW for instance; the alignment is then part of the
  number.
* :func:`f0_rmse` counts gross errors at full size: one octave error among
  100 frames voiced in both, the rest exact, is 120 cents. FPE leaves the
  gross errors out, and being a standard deviation it leaves a constant
  offset out too: an estimate 1% sharp on every frame has an FPE of 0 and an
  RMSE of 17.2 cents. An RMSE in Hz is another number, which weighs a high
  voice more (a semitone is 5.9 Hz at 100 Hz and 11.9 Hz at 200 Hz); an
  RMSE of ln F0 is the RMSE in cents divided by 1731.2.
* FPE is relative, and a standard deviation that divides by the number of
  fine frames; MATLAB's ``std`` divides by one fewer, which reads 0.5%
  higher at 100 frames. The fine errors of [CA12] are in Hz, and their
  SDFPE, like [NIZ03]'s RMS, takes a constant offset in; those of [R76] are
  in samples of pitch period; none of them is this number.
"""
from __future__ import annotations

import numbers
from typing import NamedTuple, Optional, Tuple

import numpy as np

from kudio.core._policy import warn
from kudio.core.pitch import PitchTrack

__all__ = ['gross_pitch_error', 'voicing_decision_error', 'f0_frame_error',
           'fine_pitch_error', 'f0_rmse', 'f0_errors', 'F0Errors']

#: Cents in an octave: an error of 1200 log2(est / ref) cents.
_CENTS_PER_OCTAVE = 1200.0

#: How far apart, in seconds, the frame times of two PitchTracks may lie and
#: still be one grid: far above the rounding of times worked out from the
#: same hop, and far below any hop. kudio.f0 at 16 and at 22.05 kHz, 160 and
#: 220 samples a hop, puts frame 100 at times 2.3 ms apart.
_TIME_TOLERANCE = 1e-6

#: The fewest fine frames FPE is taken over. One frame has a standard
#: deviation of 0 whatever its error.
_MIN_FINE = 2

#: What FPE needs, as the warnings put it.
_FINE_NEED = f"at least {_MIN_FINE} frames voiced in both without a gross error"


class F0Errors(NamedTuple):
    """Every measure of :mod:`kudio.core.f0eval` for one pair of tracks, and
    the frame counts behind them.

    The counts say how much each figure rests on: a GPE of 0.5 with
    ``voiced_both=2`` is one frame in two. To pool a test set, concatenate
    its tracks and score them once rather than averaging these.
    """

    #: gross pitch error, ``gross / voiced_both``
    gpe: float
    #: voicing decision error, ``(voiced_to_unvoiced + unvoiced_to_voiced) / frames``
    vde: float
    #: F0 frame error, ``(gross + voiced_to_unvoiced + unvoiced_to_voiced) / frames``
    ffe: float
    #: fine pitch error, the standard deviation of ``(est - ref) / ref`` over
    #: the ``voiced_both - gross`` frames without a gross error
    fpe: float
    #: RMS error over the frames voiced in both, in cents
    rmse_cents: float
    #: frames compared, N
    frames: int
    #: frames voiced in both tracks, N_VV
    voiced_both: int
    #: frames voiced in both whose estimate is more than the threshold off, N_GE
    gross: int
    #: frames voiced in the reference and not in the estimate, N_VU
    voiced_to_unvoiced: int
    #: frames voiced in the estimate and not in the reference, N_UV
    unvoiced_to_voiced: int


class _Frames(NamedTuple):
    """One pair of tracks sorted into the four cases."""

    frames: int
    voiced_to_unvoiced: int
    unvoiced_to_voiced: int
    #: reference and estimate on the frames voiced in both, in Hz
    ref_hz: np.ndarray
    est_hz: np.ndarray
    #: which of those frames are gross errors
    gross: np.ndarray
    #: why nothing is voiced in both, when nothing is
    why_none: str


# --- reading the tracks --------------------------------------------------------

def _track(x, name: str, fn: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """*x* as F0 in Hz with 0 on every unvoiced frame, and its frame times
    when it is a :class:`PitchTrack`.

    An integer array is read as Hz. The rule that reads integer audio on its
    type's full scale is for samples, and a track holds frequencies.
    """
    if isinstance(x, PitchTrack):
        values = np.asarray(x.f0, dtype=np.float64).reshape(-1)
        voiced = np.asarray(x.voiced, dtype=bool).reshape(-1)
        if values.shape != voiced.shape:
            raise ValueError(f"{fn}(): {name} is a PitchTrack whose f0 has "
                             f"{values.size} frames and voiced {voiced.size}")
        with np.errstate(invalid='ignore'):
            missing = voiced & ~(np.isfinite(values) & (values > 0.0))
        if missing.any():
            raise ValueError(
                f"{fn}(): {name} is a PitchTrack with "
                f"{int(np.count_nonzero(missing))} frame(s) marked voiced but "
                f"no F0 above 0 Hz, the first at frame "
                f"{int(np.argmax(missing))}; mark those frames unvoiced or give "
                f"them their F0")
        times = np.asarray(x.times, dtype=np.float64).reshape(-1)
        if times.size != values.size:
            raise ValueError(f"{fn}(): {name} is a PitchTrack whose times holds "
                             f"{times.size} value(s) for {values.size} frames; "
                             f"give it one time per frame, in seconds")
        unknown = ~np.isfinite(times)
        if unknown.any():
            raise ValueError(
                f"{fn}(): {name} is a PitchTrack with "
                f"{int(np.count_nonzero(unknown))} frame time(s) that are not "
                f"finite, the first at frame {int(np.argmax(unknown))}; give "
                f"every frame its time in seconds")
        return np.where(voiced, values, 0.0), times
    if isinstance(x, (str, bytes)):
        raise ValueError(f"{fn}(): {name} has to be F0 values in Hz, got "
                         f"{type(x).__name__} {x!r}")
    # np.asarray drops a mask, and a frame masked for low confidence would be
    # scored as voiced
    if isinstance(x, np.ma.MaskedArray) and np.ma.getmaskarray(x).any():
        raise ValueError(
            f"{fn}(): {name} is a masked array with "
            f"{int(np.count_nonzero(np.ma.getmaskarray(x)))} masked frame(s), "
            f"and a mask can stand for unvoiced frames or for frames to leave "
            f"out. np.ma.filled({name}, 0) reads masked frames as unvoiced; to "
            f"leave them out of the comparison, drop them from both tracks "
            f"(keep = ~np.ma.getmaskarray({name})).")
    try:
        values = np.asarray(x)
    except (TypeError, ValueError) as e:      # a ragged list, for one
        raise ValueError(f"{fn}(): {name} is not an array of F0 values: {e}") from e
    if values.dtype == bool:
        raise ValueError(f"{fn}(): {name} is a boolean array, a voicing "
                         f"decision with no pitch in it; pass the F0 in Hz, "
                         f"with 0 or nan where a frame is unvoiced")
    if not (np.issubdtype(values.dtype, np.integer)
            or np.issubdtype(values.dtype, np.floating)):
        raise ValueError(f"{fn}(): {name} has to hold F0 values in Hz, got an "
                         f"array of {values.dtype}")
    if values.ndim == 0:
        raise ValueError(f"{fn}(): {name} is a single number, not a track; "
                         f"pass one F0 value per frame")
    if values.ndim > 1 and sum(n != 1 for n in values.shape) <= 1:
        values = values.reshape(-1)          # (n, 1) or (1, n), n = 0 included
    if values.ndim != 1:
        raise ValueError(f"{fn}(): {name} has shape {values.shape}; score one "
                         f"track at a time")
    hz = values.astype(np.float64)
    infinite = int(np.count_nonzero(np.isinf(hz)))
    if infinite:
        raise ValueError(f"{fn}(): {name} has {infinite} infinite value(s); a "
                         f"track holds F0 in Hz, with 0 or nan where a frame "
                         f"is unvoiced")
    negative = int(np.count_nonzero(hz < 0.0))
    if negative:
        raise ValueError(
            f"{fn}(): {name} has {negative} negative value(s), which mean "
            f"different things in different files. In Keele's reference a "
            f"negative value is an uncertain frame, to be left out of the "
            f"comparison: drop those frames from both tracks (keep = ref >= "
            f"0). In MIREX and mir_eval files a negative estimate is an "
            f"unvoiced frame with a pitch guess: np.maximum(est, 0) reads it "
            f"as unvoiced.")
    hz[np.isnan(hz)] = 0.0
    return hz, None


def _pair(ref, est, fn: str) -> Tuple[np.ndarray, np.ndarray]:
    """Both tracks read, and refused unless they sit on one frame grid."""
    r, ref_times = _track(ref, 'ref', fn)
    e, est_times = _track(est, 'est', fn)
    if r.size != e.size:
        raise ValueError(
            f"{fn}(): ref has {r.size} frames and est {e.size}. Frame i of one "
            f"is compared with frame i of the other, so both tracks have to "
            f"be on one frame grid: track both with the same hop and centring "
            f"(hop_length and center in kudio.f0), or resample est onto ref's "
            f"frame times, for instance with "
            f"mir_eval.melody.resample_melody_series, which keeps unvoiced "
            f"frames at 0. Cutting the longer one is right only when both "
            f"grids start at the same time.")
    if ref_times is not None and est_times is not None and r.size:
        gap = float(np.max(np.abs(ref_times - est_times)))
        if gap > _TIME_TOLERANCE:
            raise ValueError(
                f"{fn}(): ref and est have {r.size} frames each, but their "
                f"frame times differ by up to {1000.0 * gap:.3g} ms, so frame "
                f"i of one is not frame i of the other. Track both with the "
                f"same hop_length and center, or resample est onto ref.times.")
    return r, e


def _checked_threshold(threshold, fn: str) -> float:
    # float() would take True as 1.0 and '0.2' as 0.2 without a word
    if isinstance(threshold, bool) or not isinstance(threshold, numbers.Real):
        raise ValueError(f"{fn}(): threshold has to be a number, got "
                         f"{type(threshold).__name__} {threshold!r}")
    value = float(threshold)
    if not 0.0 < value < 1.0:
        raise ValueError(f"{fn}(): threshold is a fraction of the reference "
                         f"F0, 0.2 for the usual 20%, and has to lie between 0 "
                         f"and 1; got {value:g}")
    return value


def _frames(r: np.ndarray, e: np.ndarray,
            threshold: Optional[float] = None) -> _Frames:
    """Sort the frames of a pair into the four cases. Without a *threshold*
    no frame is taken for a gross error."""
    ref_on, est_on = r > 0.0, e > 0.0
    both = ref_on & est_on
    ref_hz, est_hz = r[both], e[both]
    if threshold is None:
        gross = np.zeros(ref_hz.size, dtype=bool)
    else:
        # [CA09] eq. 3, |est / ref - 1| > threshold, with both sides times ref
        gross = np.abs(est_hz - ref_hz) > threshold * ref_hz
    if not ref_on.any():
        why = "ref has no voiced frame"
    elif not est_on.any():
        why = "est has no voiced frame"
    else:
        why = "ref and est are never voiced in the same frame"
    return _Frames(
        frames=int(r.size),
        voiced_to_unvoiced=int(np.count_nonzero(ref_on & ~est_on)),
        unvoiced_to_voiced=int(np.count_nonzero(~ref_on & est_on)),
        ref_hz=ref_hz, est_hz=est_hz, gross=gross, why_none=why)


def _gpe(f: _Frames) -> float:
    return int(np.count_nonzero(f.gross)) / f.ref_hz.size


def _vde(f: _Frames) -> float:
    return (f.voiced_to_unvoiced + f.unvoiced_to_voiced) / f.frames


def _ffe(f: _Frames) -> float:
    errors = f.voiced_to_unvoiced + f.unvoiced_to_voiced + int(np.count_nonzero(f.gross))
    return errors / f.frames


def _fine(f: _Frames) -> np.ndarray:
    """The relative error ``(est - ref) / ref`` of the frames voiced in both
    without a gross error."""
    keep = ~f.gross
    return (f.est_hz[keep] - f.ref_hz[keep]) / f.ref_hz[keep]


def _rmse_cents(f: _Frames) -> float:
    with np.errstate(over='ignore', under='ignore'):
        ratio = f.est_hz / f.ref_hz
    # A ratio past the range of a double, 1e300 Hz against 1e-10, goes
    # through the log of each side instead. Not every frame goes that way:
    # an exact octave has a ratio of exactly 2 and comes out at exactly 1200
    # cents, which the difference of two logs need not.
    lost = ~((ratio >= np.finfo(np.float64).tiny) & (ratio < np.inf))
    with np.errstate(divide='ignore'):        # the log of a ratio lost to 0
        cents = _CENTS_PER_OCTAVE * np.log2(ratio)
    if lost.any():
        cents[lost] = _CENTS_PER_OCTAVE * (np.log2(f.est_hz[lost])
                                           - np.log2(f.ref_hz[lost]))
    return float(np.sqrt(np.mean(cents * cents)))


def _scored_nan(fn: str, reason: str) -> float:
    """nan, after the warning every measure here gives for a pair it cannot
    score; *reason* is what goes before '; scored nan'."""
    warn(f"{fn}(): {reason}; scored nan")
    return float('nan')


# --- the measures --------------------------------------------------------------

def gross_pitch_error(ref, est, *, threshold: float = 0.2) -> float:
    """Gross pitch error: the share of the frames voiced in both tracks where
    *est* is more than *threshold* off *ref*.

    N_GE / N_VV, [CA09] eq. 2: a frame voiced in both counts when
    ``|est - ref| > threshold * ref``, which is the 20% rule with the
    default, measured against the reference: 125 Hz against 100 counts, 100
    against 125 does not. Frames voiced in only one track do not enter it,
    so a tracker lowers its GPE by calling the frames it gets wrong
    unvoiced; :func:`voicing_decision_error` counts those, and
    :func:`f0_frame_error` counts both.

    :param ref: the reference track, F0 in Hz frame by frame with 0 or nan
        where unvoiced, or a :class:`kudio.PitchTrack`.
    :param est: the estimate, on the same frames, in the same form.
    :param threshold: how far off, as a fraction of the reference F0, an
        estimate has to be to count: 0.2, the 20% the papers use.
    :raises ValueError: for tracks of different lengths, two PitchTracks
        whose frame times differ, a PitchTrack with a voiced frame that has
        no F0 or without one finite time per frame, negative or infinite
        values, masked frames, an input that is not one track of numbers, or
        a *threshold* outside 0 to 1.
    :returns: the fraction of the N_VV frames with a gross error, 0 to 1.
        nan, with a warning, when no frame is voiced in both.
    """
    fn = 'gross_pitch_error'
    threshold = _checked_threshold(threshold, fn)
    r, e = _pair(ref, est, fn)
    if not r.size:
        return _scored_nan(fn, "got two empty tracks")
    f = _frames(r, e, threshold)
    if not f.ref_hz.size:
        return _scored_nan(fn, f"{f.why_none}, and GPE needs at least 1 frame "
                               f"voiced in both")
    return _gpe(f)


def voicing_decision_error(ref, est) -> float:
    """Voicing decision error: the share of all frames where *est* and *ref*
    disagree on whether the frame is voiced.

    (N_VU + N_UV) / N, [CA09] eq. 1. Every frame counts in N, so frames both
    tracks call unvoiced, such as silence before and after the speech, lower
    it; trim the tracks of every system compared the same way.

    :param ref: the reference track, F0 in Hz frame by frame with 0 or nan
        where unvoiced, or a :class:`kudio.PitchTrack`.
    :param est: the estimate, on the same frames, in the same form.
    :raises ValueError: as :func:`gross_pitch_error`.
    :returns: the fraction of all N frames with a voicing error, 0 to 1.
        nan, with a warning, for two empty tracks.
    """
    fn = 'voicing_decision_error'
    r, e = _pair(ref, est, fn)
    if not r.size:
        return _scored_nan(fn, "got two empty tracks")
    ref_on, est_on = r > 0.0, e > 0.0
    return int(np.count_nonzero(ref_on != est_on)) / r.size


def f0_frame_error(ref, est, *, threshold: float = 0.2) -> float:
    """F0 frame error: the share of all frames with any error, a voicing
    error or a gross pitch error.

    (N_VU + N_UV + N_GE) / N, [CA09] eq. 4, which is
    ``VDE + (N_VV / N) * GPE``. Calling a frame unvoiced instead of getting
    its pitch wrong moves the error from GPE to VDE and leaves FFE where it
    was, which is why it is one number to compare trackers by.

    :param ref: the reference track, F0 in Hz frame by frame with 0 or nan
        where unvoiced, or a :class:`kudio.PitchTrack`.
    :param est: the estimate, on the same frames, in the same form.
    :param threshold: as in :func:`gross_pitch_error`.
    :raises ValueError: as :func:`gross_pitch_error`.
    :returns: the fraction of all N frames with an error, 0 to 1. nan, with
        a warning, for two empty tracks.
    """
    fn = 'f0_frame_error'
    threshold = _checked_threshold(threshold, fn)
    r, e = _pair(ref, est, fn)
    if not r.size:
        return _scored_nan(fn, "got two empty tracks")
    return _ffe(_frames(r, e, threshold))


def fine_pitch_error(ref, est, *, threshold: float = 0.2) -> float:
    """Fine pitch error: how much *est* scatters around *ref* on the frames
    it gets right, as a fraction of the reference F0.

    The standard deviation of ``(est - ref) / ref`` over the frames voiced in
    both whose estimate is within *threshold* of the reference, as [DA11]
    defines FPE; times 100 it is their percentage. It divides by the number
    of those frames, as [R76]'s eq. 3 does. Being a standard deviation, it
    does not see a constant offset: an estimate 1% sharp on every frame
    scores 0 here and 17.2 cents in :func:`f0_rmse`.

    :param ref: the reference track, F0 in Hz frame by frame with 0 or nan
        where unvoiced, or a :class:`kudio.PitchTrack`.
    :param est: the estimate, on the same frames, in the same form.
    :param threshold: the line between fine and gross, as in
        :func:`gross_pitch_error`.
    :raises ValueError: as :func:`gross_pitch_error`.
    :returns: the standard deviation, as a fraction. nan, with a warning,
        when fewer than 2 frames are voiced in both without a gross error.
    """
    fn = 'fine_pitch_error'
    threshold = _checked_threshold(threshold, fn)
    r, e = _pair(ref, est, fn)
    if not r.size:
        return _scored_nan(fn, "got two empty tracks")
    f = _frames(r, e, threshold)
    if not f.ref_hz.size:
        return _scored_nan(fn, f"{f.why_none}, and FPE needs {_FINE_NEED}")
    fine = _fine(f)
    if fine.size < _MIN_FINE:
        return _scored_nan(fn, f"FPE needs {_FINE_NEED} to take a standard "
                               f"deviation over, and the pair has {fine.size}")
    return float(np.std(fine))


def f0_rmse(ref, est) -> float:
    """Root mean square F0 error in cents over the frames voiced in both.

    ``sqrt(mean((1200 log2(est / ref)) ** 2))``, the F0 RMSE that speech
    synthesis papers report. A cent is a hundredth of a semitone, so the
    same error counts the same in a low voice and a high one, which an RMSE
    in Hz does not. Gross errors count at full size: one octave error among
    100 frames, the rest exact, is 120 cents. Synthesized speech has to be
    put on the reference's frames first, by DTW for instance; this compares
    the frames as given.

    :param ref: the reference track, F0 in Hz frame by frame with 0 or nan
        where unvoiced, or a :class:`kudio.PitchTrack`.
    :param est: the estimate, on the same frames, in the same form.
    :raises ValueError: as :func:`gross_pitch_error`, *threshold* aside.
    :returns: the RMSE in cents. nan, with a warning, when no frame is voiced
        in both.
    """
    fn = 'f0_rmse'
    r, e = _pair(ref, est, fn)
    if not r.size:
        return _scored_nan(fn, "got two empty tracks")
    f = _frames(r, e)
    if not f.ref_hz.size:
        return _scored_nan(fn, f"{f.why_none}, and the RMSE needs at least 1 "
                               f"frame voiced in both")
    return _rmse_cents(f)


def f0_errors(ref, est, *, threshold: float = 0.2) -> F0Errors:
    """GPE, VDE, FFE, FPE and the RMSE in cents of *est* against *ref* in one
    pass, with the frame counts they come from.

    >>> kudio.f0_errors([0, 100, 102, 104, 0, 0, 120, 121],
    ...                 [0, 101, 204, 0, 0, 98, 119, 0])  # doctest: +NORMALIZE_WHITESPACE
    F0Errors(gpe=0.3333333333333333, vde=0.375, ffe=0.5,
             fpe=0.009166666666666667, rmse_cents=692.9421890055355,
             frames=8, voiced_both=3, gross=1, voiced_to_unvoiced=2,
             unvoiced_to_voiced=1)

    The octave error at frame 2 is 1200 of the RMSE's cents and none of
    FPE's, which is left with the 1% and -0.83% of the other two frames.
    The numbers are those of :func:`gross_pitch_error`,
    :func:`voicing_decision_error`, :func:`f0_frame_error`,
    :func:`fine_pitch_error` and :func:`f0_rmse`, and so are the rules for
    what each one cannot score.

    :param ref: the reference track, F0 in Hz frame by frame with 0 or nan
        where unvoiced, or a :class:`kudio.PitchTrack`.
    :param est: the estimate, on the same frames, in the same form.
    :param threshold: as in :func:`gross_pitch_error`.
    :raises ValueError: as :func:`gross_pitch_error`.
    :returns: an :class:`F0Errors`. A field that cannot be scored is nan, and
        one warning names them all: every rate for two empty tracks; gpe,
        fpe and rmse_cents when no frame is voiced in both; fpe alone when
        fewer than 2 frames are voiced in both without a gross error.
    """
    fn = 'f0_errors'
    threshold = _checked_threshold(threshold, fn)
    r, e = _pair(ref, est, fn)
    nan = float('nan')
    if not r.size:
        warn(f"{fn}(): got two empty tracks; scored nan")
        return F0Errors(nan, nan, nan, nan, nan, 0, 0, 0, 0, 0)
    f = _frames(r, e, threshold)
    counts = dict(frames=f.frames, voiced_both=int(f.ref_hz.size),
                  gross=int(np.count_nonzero(f.gross)),
                  voiced_to_unvoiced=f.voiced_to_unvoiced,
                  unvoiced_to_voiced=f.unvoiced_to_voiced)
    if not f.ref_hz.size:
        warn(f"{fn}(): {f.why_none}, so gpe, fpe and rmse_cents have "
             f"no frame to measure; scored nan")
        return F0Errors(gpe=nan, vde=_vde(f), ffe=_ffe(f), fpe=nan,
                        rmse_cents=nan, **counts)
    fine = _fine(f)
    if fine.size < _MIN_FINE:
        warn(f"{fn}(): fpe needs {_FINE_NEED}, and the pair has "
             f"{fine.size}; scored nan")
        fpe = nan
    else:
        fpe = float(np.std(fine))
    return F0Errors(gpe=_gpe(f), vde=_vde(f), ffe=_ffe(f), fpe=fpe,
                    rmse_cents=_rmse_cents(f), **counts)
