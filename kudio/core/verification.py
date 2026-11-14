# -*- coding: utf-8 -*-
"""Detection measures for speaker verification, and for any task scored trial
by trial: the equal error rate and the minimum detection cost of NIST's
speaker recognition evaluations, and the calibration measures that read the
scores as likelihood ratios.

A verification system gives every trial, an enrolled speaker against a test
recording, a score that is higher the more it believes the two are one
talker, and a threshold turns the scores into decisions: a target trial
scored below it is a miss, a non-target trial at or above it a false alarm.
Each distinct score is a threshold of its own, so a list of trials is a set
of operating points (P_miss, P_fa), the points of its DET curve (Martin et
al. 1997), and both measures read one number off them:

* :func:`eer`, the error rate at which P_miss and P_fa are equal;
* :func:`min_dcf`, the lowest weighted sum of the two over every threshold,
  NIST's C_Det divided by the cost of deciding without the scores.

:func:`det_curve` returns the operating points themselves, for a plot or for
choosing a threshold.

Both judge the ranking alone. Where the scores are meant as natural-log
likelihood ratios, as NIST's plans since SRE 2016 ask for, the decisions
they make themselves can be costed as well:

* :func:`act_dcf`, the normalized cost at the threshold Bayes' rule sets for
  the application, and :func:`c_primary`, its mean at SRE 2016's two priors;
* :func:`cllr`, Brümmer and du Preez's C_llr, the cost over every
  application at once in bits, and :func:`min_cllr`, what the ranking would
  cost after the best calibration it allows.

**Which EER.** P_miss moves in steps of 1/T and P_fa in steps of 1/N, for T
target and N non-target trials, and as a rule no operating point has the two
equal. Every EER is therefore a rule for reading between the points, and the
rules in use part on small lists. The default joins successive operating
points with straight lines and takes the rate where the line crosses
P_miss = P_fa. I chose it for three reasons:

* It is the rule of the scorers the evaluations published. The VoxSRC
  2020-2023 scorer finds the crossing on scikit-learn's ROC curve with
  brentq, and NIST's SRE 2016 scoring script interpolates between the two
  operating points either side of it, which is the same crossing unless a
  target and a non-target share a score. With such ties that script's EER
  moves with the order of the list; the second point under "What to watch
  for" has by how much.
* A real threshold reaches it, unless the trials tied at the crossing include
  both a target and a non-target: one of the operating points there has one
  rate equal to the EER and the other at or below it.
* Across a run of tied scores the line is what accepting each tied trial with
  one and the same probability gives on average, so the order of the list
  cannot move the result.

``method='rocch'`` gives the ROCCH-EER of the BOSARIS toolkit instead
(Brümmer and de Villiers, Sec. 2.6.2): the crossing of the convex hull of the
operating points, which they propose as a well-defined version of the EER. It
is the largest, over every prior p, of the lowest p P_miss + (1 - p) P_fa any
threshold gives (their Eq. 26), and it is never above the default.

Eleven trials show how far apart the rules can be. Ranked from the lowest
score they are four non-targets, three targets, two non-targets and two
targets, and the scorers read::

    kudio's default, NIST's SRE 2016 script, the VoxSRC scorer    1/3 = 0.333
    kudio, method='rocch'                                         3/14 = 0.214
    Kaldi's compute-eer                                           0.4
    SpeechBrain's EER                                             0.367
    voxceleb_trainer's tuneThresholdfromScore                     0.6

Kaldi counts up through the target scores and reports the miss rate at the
first one where the false-alarm rate is no higher than the miss rate.
SpeechBrain takes the mean of the two rates where they are closest.
voxceleb_trainer takes the larger of the two where they are closest, among
the points scikit-learn's ``roc_curve`` keeps by default; that drops the
points inside straight runs, here the one at P_fa = 1/3, P_miss = 0.4. On a
list of 40,000 trials, half of them targets, with Gaussian scores, the
default reads 1.090% and the hull 1.081%, and on that list and five more
like it the three other scorers stay within 0.01 percentage points of the
default.

**Which cost.** NIST's plans write the cost of a threshold as::

    C_Det = c_miss p_target P_miss + c_fa (1 - p_target) P_fa

and divide it by C_Default = min(c_miss p_target, c_fa (1 - p_target)), the
cost of always rejecting or always accepting (SRE 2008 and 2010, Sec. 3; SRE
2016 and 2019, Sec. 3.1, Eqs. 2-4). :func:`min_dcf` is the lowest of these
over every threshold, which is the minimum NIST reports beside the actual
cost, "assuming perfect calibration" in the words of the SRE 2010 plan. The
evaluations do not agree on the parameters, which is why ``p_target`` has no
default::

    SRE 2008, and SRE 2010's historical cost   c_miss 10, c_fa 1, p_target 0.01
    SRE 2010, core condition                   c_miss 1,  c_fa 1, p_target 0.001
    SRE 2016 and the SRE 2019 CTS challenge    c_miss 1,  c_fa 1, p_target 0.01 and 0.005
    SRE 2019 audio-visual, VoxSRC 2020-2023    c_miss 1,  c_fa 1, p_target 0.05

The scorers' defaults differ as well: 0.01 in Kaldi's ``compute_min_dcf.py``
and SpeechBrain's ``minDCF``, 0.05 in the VoxSRC scorer and voxceleb_trainer.
SpeechBrain's ``minDCF`` returns C_Det before the division. Its VoxCeleb
recipes print that times 100, which is the normalized cost at its default
p_target = 0.01 with unit costs and nowhere else: at 0.05 it is five times
the normalized cost.

What to watch for:

* A higher score has to mean "more likely the same speaker". Distances go in
  as their negatives. A list ranked exactly the wrong way round reads 1.0 by
  default and 0.5 with ``method='rocch'``, whose hull always contains the line
  of guessing.
* Only distinct scores are thresholds. Kaldi's ``compute_min_dcf.py``, which
  the VoxSRC scorer and voxceleb_trainer copy, and NIST's SRE 2016 script
  rank the trials one at a time, and so also put operating points between
  two trials with the same score, which no threshold reaches and which move
  with the order of the list. On 22,000 trials whose scores were rounded to
  a tenth of their standard deviation, listed non-targets first, that rule
  gives a min DCF of 0.400 at p_target = 0.05, where no threshold does better
  than 0.436. NIST's script reads the EER of the same list as 0.05985 when
  its sort leaves the tied trials non-targets first and 0.07050 when it
  leaves them targets first; the default reads 0.06535 in any order.
* The rates move in steps of 1/T and 1/N, and at a small p_target the false
  alarms decide. With unit costs the normalized cost is
  P_miss + beta P_fa, beta = (1 - p_target) / p_target, so at 0.001 each
  false alarm among 10,000 non-target trials adds 0.0999. A min DCF at that
  prior read off fewer non-target trials is the count of a handful of errors.
* ``p_target`` is the prior of the application the cost stands for, not the
  share of target trials in the list; the SRE 2010 plan says as much of its
  0.001.
* One cost has more than one spelling. Costs c_miss and c_fa at prior p give
  the same normalized cost as unit costs at the effective prior
  c_miss p / (c_miss p + c_fa (1 - p)) (Brümmer and de Villiers, Eq. 10), so
  SRE 2008's (10, 1, 0.01) is p_target = 0.0917 with unit costs, not 0.01.
  The copy of NIST's SRE 2016 script that WeSpeaker distributes prints the
  cost at (1, 1, 0.01) under the name minDCF08. On the first list of 40,000
  trials above, the two read 0.059 and 0.143.
* The minimum is found with the labels in hand, so it is the cost of the best
  threshold for this very list. A threshold for new trials is chosen on a
  development list, with :func:`det_curve`. NIST's primary costs in SRE 2016
  and 2019 are actual costs, at the threshold log(beta) on calibrated
  log-likelihood ratios, and SRE 2016 and the 2019 CTS challenge average them
  over partitions of the list; :func:`min_dcf` pools the list as it comes and
  needs no calibration.

Sources: NIST, "The NIST Year 2008 Speaker Recognition Evaluation Plan"
(3 April 2008); "The NIST Year 2010 Speaker Recognition Evaluation Plan"
(version r6, 21 April 2010); "NIST 2016 Speaker Recognition Evaluation Plan"
(4 August 2016); "NIST 2019 Speaker Recognition Evaluation: CTS Challenge"
(22 July 2019); "NIST 2019 Speaker Recognition Evaluation Plan" (16 August
2019, the audio-visual evaluation). N. Brümmer and E. de Villiers, "The
BOSARIS Toolkit: Theory, Algorithms and Code for Surviving the New DCF",
NIST SRE'11 Analysis Workshop, December 2011, arXiv:1304.2865. A. Martin,
G. Doddington, T. Kamm, M. Ordowski and M. Przybocki, "The DET curve in
assessment of detection task performance", Proc. Eurospeech 1997,
pp. 1895-1898, doi:10.21437/Eurospeech.1997-504. [BD06] N. Brümmer and
J. du Preez, "Application-independent evaluation of speaker detection",
Computer Speech and Language 20(2-3), pp. 230-275, 2006,
doi:10.1016/j.csl.2005.08.001. The other scorers are the
code they published: Kaldi's ``src/ivectorbin/compute-eer.cc`` and
``egs/sre08/v1/sid/compute_min_dcf.py``, the VoxSRC 2020-2023 validation
toolkits (``compute_EER.py``, ``compute_min_dcf.py``),
clovaai/voxceleb_trainer's ``tuneThreshold.py``, SpeechBrain's
``speechbrain/utils/metric_stats.py``, and NIST's SRE 2016 scoring script
(version 4.1, as WeSpeaker distributes it in ``wespeaker/utils/score_metrics.py``,
last changed in commit 3556586 of 2022-03-25; its ``min_dcf_08`` is
``compute_c_norm(fnr, fpr, p_target=0.01, c_miss=1, c_fa=1)``).
Their numbers above come from those rules restated in
``tools/validation/verification.py``, which reruns every comparison here;
``tools/validation/calibration.py`` does the same for the calibration
measures, with PYLLR's rules (github.com/bsxfan/PYLLR, commit 442b10b).
Each call sorts the list once, and a million trials take about a tenth of a
second on a desktop CPU.
"""
from __future__ import annotations

import math
import numbers
import sys
import warnings
from fractions import Fraction
from typing import NamedTuple, Optional, Tuple

import numpy as np

from kudio.core._policy import warn

__all__ = ['eer', 'min_dcf', 'det_curve', 'DetCurve', 'cllr', 'min_cllr',
           'act_dcf', 'c_primary', 'C_PRIMARY_PRIORS']

#: The two ways :func:`eer` reads the crossing: straight lines between
#: successive operating points, or the convex hull of all of them.
_EER_METHODS = ('linear', 'rocch')


class DetCurve(NamedTuple):
    """The operating points of a trial list, one per threshold, in rising
    order of threshold.

    A trial is accepted when its score is at or above the threshold. The
    first point accepts every trial (P_miss 0, P_fa 1) and the last, at
    ``inf``, rejects every trial (P_miss 1, P_fa 0); in between there is one
    point per distinct score. P_miss never falls and P_fa never rises from
    one point to the next.
    """

    #: the thresholds, float64: every distinct score, then ``inf``
    thresholds: np.ndarray
    #: share of target trials scored below each threshold
    p_miss: np.ndarray
    #: share of non-target trials scored at or above each threshold
    p_fa: np.ndarray


class _Counts(NamedTuple):
    """Error counts at every operating point, as :class:`DetCurve` orders
    them, and the class sizes."""

    thresholds: np.ndarray
    misses: np.ndarray            # int64, target trials below the threshold
    false_alarms: np.ndarray      # int64, non-target trials at or above it
    targets: int
    nontargets: int


def _vector(values, what: str, name: str) -> np.ndarray:
    """*values* as a one-dimensional array, a ``(n, 1)`` or ``(1, n)`` array
    read as the n values it holds, n = 0 included."""
    if isinstance(values, np.ma.MaskedArray) and np.ma.getmaskarray(values).any():
        # np.asarray drops the mask, and the masked trials would be counted
        # at whatever value the mask hides
        masked = int(np.count_nonzero(np.ma.getmaskarray(values)))
        raise ValueError(f"{name}(): {what} are a masked array with {masked} masked "
                         f"trial(s), and every trial is counted; drop those trials "
                         f"from scores and labels alike first (keep = "
                         f"~np.ma.getmaskarray({what}))")
    try:
        values = np.asarray(values)
    except ValueError as e:               # rows of different lengths
        raise ValueError(f"{name}(): {what} have to be one per trial, in a "
                         f"one-dimensional sequence; {e}") from None
    if values.ndim > 1 and sum(size != 1 for size in values.shape) <= 1:
        values = values.reshape(-1)
    if values.ndim != 1:
        raise ValueError(f"{name}(): {what} have to be one per trial, in a "
                         f"one-dimensional sequence; got shape {values.shape}")
    return values


def _is_real(value) -> bool:
    """Whether *value* is a real number of any of Python's or numpy's kinds,
    Decimal included, or a boolean. Text that reads as a number is not:
    astype(float) would turn '0.5' into 0.5 without a word."""
    if isinstance(value, (numbers.Real, np.bool_)):
        return True
    return isinstance(value, numbers.Number) and not isinstance(value, numbers.Complex)


def _unreadable(value) -> bool:
    """Whether *value* is not a real number float64 can hold: text, None or
    a complex number, but also Decimal('sNaN') or an integer past 1.8e308,
    which float() refuses."""
    if not _is_real(value):
        return True
    try:
        float(value)
    except (ValueError, OverflowError):
        return True
    return False


def _shown(value) -> str:
    """repr(*value*), cut short for a message."""
    text = repr(value)
    return text if len(text) <= 40 else text[:36] + '...'


def _numbers(values: np.ndarray):
    """An array of Python objects as float64, and None; or None and the
    first of them that is :func:`_unreadable`."""
    items = values.tolist()
    for item in items:
        if _unreadable(item):
            return None, item
    return np.asarray(items, dtype=np.float64), None


def _scores(scores, name: str, what: str = 'scores') -> np.ndarray:
    scores = _vector(scores, what, name)
    if scores.dtype.kind == 'O':
        scores, bad = _numbers(scores)
        if scores is None:
            kind_of = "numbers that float64 can hold" if _is_real(bad) else "numbers"
            raise ValueError(f"{name}(): {what} have to be {kind_of}, got "
                             f"{_shown(bad)} among them")
    kind = scores.dtype.kind
    if kind not in 'biuf':
        raise ValueError(f"{name}(): {what} have to be real numbers, got an "
                         f"array of {scores.dtype}")
    # ranked as float64, which is how `scores >= threshold` compares integer
    # scores with the thresholds det_curve returns; integers past 2**53 that
    # float64 cannot tell apart are tied
    scores = scores.astype(np.float64, copy=False)
    if what == 'llrs':
        # a calibrated system may be certain, and an infinite LLR says so;
        # a NaN says nothing
        bad = int(np.count_nonzero(np.isnan(scores)))
        if bad:
            raise ValueError(
                f"{name}(): {bad} of the {scores.size} llrs "
                f"{'is' if bad == 1 else 'are'} NaN. A trial without a "
                f"likelihood ratio cannot be costed: drop those trials or "
                f"score them")
        return scores
    bad = int(np.count_nonzero(~np.isfinite(scores)))
    if bad:
        raise ValueError(
            f"{name}(): {bad} of the {scores.size} scores "
            f"{'is' if bad == 1 else 'are'} not finite (NaN or inf). A "
            f"trial without a score has no place in the ranking: drop "
            f"those trials or score them, and replace an infinite score "
            f"with a finite one beyond all the others if it means "
            f"'certain'")
    return scores


def _labels(labels, name: str) -> np.ndarray:
    """*labels* as booleans, True for a target trial, after checking that
    each one is a boolean, a 0 or a 1."""
    labels = _vector(labels, 'labels', name)
    text_hint = (" Turn text labels into booleans first, e.g. "
                 "np.asarray(labels) == 'target'.")
    if labels.dtype.kind == 'O':
        # a list mixing booleans and numbers, or holding None or text, or a
        # pandas column of text
        labels, bad = _numbers(labels)
        if labels is None:
            hint = text_hint if isinstance(bad, str) else ""
            raise ValueError(f"{name}(): labels have to be booleans or 0/1, "
                             f"with 1 for a target trial; got {_shown(bad)} "
                             f"among them.{hint}")
    kind = labels.dtype.kind
    if kind == 'b':
        return labels
    if kind in 'iuf':
        wrong = (labels != 0) & (labels != 1)
        if not np.any(wrong):
            return labels == 1
        example = labels[np.argmax(wrong)].item()
        if example == -1:
            hint = " For -1/+1 labels, pass np.asarray(labels) == 1."
        elif isinstance(example, float) and math.isnan(example):
            hint = " A trial without a label cannot be counted; drop it first."
        elif kind == 'f':
            hint = (" If these are scores, the arguments are the wrong way "
                    "round: scores come first, labels second.")
        else:
            hint = ""
        raise ValueError(f"{name}(): labels have to be booleans or 0/1, with "
                         f"1 for a target trial; got {example!r} among them."
                         f"{hint}")
    hint = text_hint if kind in 'US' else ""
    raise ValueError(f"{name}(): labels have to be booleans or 0/1, with 1 for "
                     f"a target trial; got an array of {labels.dtype}.{hint}")


def _counted(scores, labels, name: str) -> _Counts:
    """The operating points of the list, after the checks every function here
    shares."""
    scores = _scores(scores, name)
    labels = _labels(labels, name)
    if scores.size != labels.size:
        raise ValueError(f"{name}(): {scores.size} scores and {labels.size} "
                         f"labels; they pair up by position, one of each per "
                         f"trial")
    order = np.argsort(scores)
    ranked = scores[order]
    target = labels[order]
    # where a new distinct score starts; tied trials share one threshold, so
    # the order the sort leaves them in does not matter
    first = np.empty(ranked.size, dtype=bool)
    first[:1] = True
    np.not_equal(ranked[1:], ranked[:-1], out=first[1:])
    starts = np.flatnonzero(first)
    if starts.size:
        targets_at = np.add.reduceat(target.astype(np.int64), starts)
        trials_at = np.diff(np.append(starts, ranked.size))
    else:
        targets_at = trials_at = np.zeros(0, dtype=np.int64)
    targets = int(targets_at.sum())
    nontargets = int(ranked.size - targets)
    # the threshold at the k-th distinct score accepts it and everything
    # above, so it misses the targets scored below it
    misses = np.concatenate(([0], np.cumsum(targets_at)))
    false_alarms = nontargets - np.concatenate(([0], np.cumsum(trials_at - targets_at)))
    thresholds = np.append(ranked[starts], np.inf)
    return _Counts(thresholds, misses.astype(np.int64),
                   false_alarms.astype(np.int64), targets, nontargets)


def _unscorable(counts: _Counts, name: str) -> Optional[str]:
    """Why the list cannot be scored, or None when it can."""
    need = "it needs at least one target and one non-target trial"
    if counts.targets == 0 and counts.nontargets == 0:
        return f"{name}(): the list is empty; {need}"
    if counts.targets == 0:
        return (f"{name}(): the list has no target trials, so the miss rate "
                f"is undefined; {need}")
    if counts.nontargets == 0:
        return (f"{name}(): the list has no non-target trials, so the "
                f"false-alarm rate is undefined; {need}")
    return None


def det_curve(scores, labels) -> DetCurve:
    """Every operating point of a trial list: for each threshold, the miss
    and false-alarm rates of accepting the trials scored at or above it.

    >>> curve = kudio.det_curve(dev_scores, dev_labels)      # doctest: +SKIP
    >>> worst = np.maximum(curve.p_miss, curve.p_fa)         # doctest: +SKIP
    >>> curve.thresholds[np.argmin(worst)]                   # doctest: +SKIP
    >>> cost = 0.05 * curve.p_miss + 0.95 * curve.p_fa       # doctest: +SKIP
    >>> curve.thresholds[np.argmin(cost)]                    # doctest: +SKIP

    The thresholds are the distinct scores, so trials with the same score are
    accepted or rejected together; no point sits between them. The order is
    rising threshold, the reverse of scikit-learn's ``roc_curve``, and
    nothing is dropped: the points inside a straight run are kept, and so are
    the two ends, accepting everything and rejecting everything.
    scikit-learn's ``det_curve`` rises as well, but keeps one point where
    P_miss is 0 and one where P_fa is lowest and leaves out the points beyond
    them, so whether it has either end depends on the list and on the
    release: 1.6.1 never returns the one at ``inf``, 1.7.2 can. For a DET
    plot, put both rates through ``scipy.stats.norm.ppf``; the ends map to
    infinity, which matplotlib leaves out.

    The first threshold in the example is the one whose larger error rate is
    the smallest. That rate is the EER of :func:`eer`, unless the trials tied
    at the crossing include both a target and a non-target; then no single
    threshold reaches the EER, and this one comes closest. The second is the threshold of :func:`min_dcf` at
    p_target 0.05 with unit costs. Chosen on a development list, either can
    be used on new trials.

    :param scores: one score per trial, higher for "more likely a target".
        Integer scores are taken as the numbers they are, and every score is
        compared as float64: integers past 2**53 that float64 cannot tell
        apart count as tied, so that ``scores >= threshold`` accepts exactly
        the trials each point counts.
    :param labels: one per trial, True or 1 for a target trial and False or
        0 for a non-target one.
    :raises ValueError: for scores that are not finite numbers float64 can
        hold, labels other than booleans or 0/1, a two-dimensional array (a
        ``(n, 1)`` or ``(1, n)`` one is read as n trials), a masked array
        with trials masked, or scores and labels of different lengths.
    :returns: a :class:`DetCurve`. For a list with no target trials p_miss
        is nan throughout, and with no non-target trials p_fa is, with a
        warning; an empty list gives the single point at ``inf``, both rates
        nan.
    """
    counts = _counted(scores, labels, 'det_curve')
    why = _unscorable(counts, 'det_curve')
    if why is not None:
        nan = ("both rates" if not counts.targets and not counts.nontargets
               else "p_miss" if not counts.targets else "p_fa")
        warn(f"{why}; {nan} came out nan")
    p_miss = (counts.misses / counts.targets if counts.targets
              else np.full(counts.misses.size, np.nan))
    p_fa = (counts.false_alarms / counts.nontargets if counts.nontargets
            else np.full(counts.false_alarms.size, np.nan))
    return DetCurve(counts.thresholds, p_miss, p_fa)


def _crossing(counts: _Counts, low: int, high: int) -> Fraction:
    """Where the straight line from operating point *low* (P_miss below P_fa)
    to *high* (above it) crosses P_miss = P_fa, exactly.

    With D = misses N - false alarms T, which has the sign of P_miss - P_fa,
    the crossing is ``(m_low D_high - m_high D_low) / (T (D_high - D_low))``;
    Python integers keep it exact at any list size.
    """
    t, n = counts.targets, counts.nontargets
    m_low, m_high = int(counts.misses[low]), int(counts.misses[high])
    d_low = m_low * n - int(counts.false_alarms[low]) * t
    d_high = m_high * n - int(counts.false_alarms[high]) * t
    return Fraction(m_low * d_high - m_high * d_low, t * (d_high - d_low))


def _hull_crossing(counts: _Counts, gap: np.ndarray) -> Tuple[Fraction, int]:
    """The ROCCH-EER, where the lower convex hull of the operating points
    crosses P_miss = P_fa, and how many chords it took to find.

    The hull edge that crosses is found the way quickhull finds edges,
    following only the side the crossing is on. Start from the chord between
    accepting everything and rejecting everything; the operating point
    furthest beyond it, towards zero error, is a hull vertex, and replaces
    the end of the chord on its own side of the crossing. When no point lies
    beyond the chord, the chord is the edge. The tests are cross products of
    the integer counts, so they are exact; scaling the axes by 1/N and 1/T
    changes neither convexity nor the side a point is on.

    Each chord costs one pass over the points between its ends. A list of
    1,003,302 trials built to give its hull 8,830 vertices, one for every
    primitive step (a, b) with a + b <= 170, took 14 chords.
    """
    misses, false_alarms = counts.misses, counts.false_alarms
    low, high = 0, misses.size - 1        # gap < 0 at low, > 0 at high
    chords = 0
    while high - low > 1:
        chords += 1
        x0, y0 = int(false_alarms[low]), int(misses[low])
        dx, dy = int(false_alarms[high]) - x0, int(misses[high]) - y0
        inside = slice(low + 1, high)
        beyond = dx * (misses[inside] - y0) - dy * (false_alarms[inside] - x0)
        k = int(np.argmax(beyond))
        if beyond[k] <= 0:
            break
        k += low + 1
        if gap[k] == 0:
            return Fraction(int(misses[k]), counts.targets), chords
        if gap[k] < 0:
            low = k
        else:
            high = k
    return _crossing(counts, low, high), chords


def eer(scores, labels, *, method: str = 'linear') -> float:
    """Equal error rate of a trial list: the rate at which misses and false
    alarms are equally likely. 0 is perfect separation, 0.5 is guessing.

    >>> scores = list(range(11))
    >>> labels = [0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1]
    >>> kudio.eer(scores, labels)                           # doctest: +SKIP
    0.3333333333333333
    >>> kudio.eer(scores, labels, method='rocch')           # doctest: +SKIP
    0.21428571428571427

    The result is a proportion; Kaldi's ``compute-eer``, voxceleb_trainer and
    the VoxSRC scorer print percent. Only the order of the scores matters,
    so any strictly rising transformation of them leaves it unchanged.

    :param scores: one score per trial, higher for "more likely a target".
    :param labels: one per trial, True or 1 for a target trial and False or
        0 for a non-target one.
    :param method: ``'linear'`` (default): straight lines join successive
        operating points, and the EER is where they cross P_miss = P_fa, as
        the VoxSRC 2020-2023 scorer computes it, and NIST's SRE 2016 script
        on lists where no target and non-target share a score; with such
        ties that script's result moves with the order of the list. Unless the trials tied at the crossing include both a target and a
        non-target, some threshold has one error rate equal to it and the
        other at or below it. ``'rocch'``: the crossing of the convex hull of
        the operating points, BOSARIS's ROCCH-EER, never above the default.
        The module docstring has why the default is the default and how far
        apart the two are.
    :raises ValueError: for an unknown *method*, and for the trial lists
        :func:`det_curve` refuses.
    :returns: the EER, 0 to 1 (0 to 0.5 with ``'rocch'``). nan, with a
        warning, for a list without a target or without a non-target trial.
    """
    if method not in _EER_METHODS:
        raise ValueError(f"eer(): method has to be one of {_EER_METHODS}, got "
                         f"{method!r}")
    counts = _counted(scores, labels, 'eer')
    why = _unscorable(counts, 'eer')
    if why is not None:
        warn(f"{why}; scored nan")
        return float('nan')
    # D = misses N - false alarms T has the sign of P_miss - P_fa and rises
    # strictly from point to point. It and the hull's cross products are at
    # most 2 T N <= n**2 / 2, so int64 holds them exactly below 2**32 trials.
    gap = counts.misses * counts.nontargets - counts.false_alarms * counts.targets
    if method == 'rocch':
        return float(_hull_crossing(counts, gap)[0])
    k = int(np.searchsorted(gap, 0))        # the first point with D >= 0
    if gap[k] == 0:
        return float(Fraction(int(counts.misses[k]), counts.targets))
    return float(_crossing(counts, k - 1, k))


def _cost_parameter(value, what: str, upper: Optional[float] = None,
                    name: str = 'min_dcf') -> float:
    """A cost parameter as a float, refused when it is not a number in its
    range: (0, upper) when *upper* is given, (0, inf) otherwise."""
    # float() would take True as 1.0 without a word
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name}(): {what} has to be a number, got "
                         f"{type(value).__name__} {value!r}")
    value = float(value)
    if upper is None:
        if not 0.0 < value < math.inf:
            raise ValueError(f"{name}(): {what} has to be a positive cost, "
                             f"got {value!r}")
    elif not 0.0 < value < upper:
        raise ValueError(
            f"{name}(): p_target has to be a probability strictly between 0 "
            f"and 1, got {value!r}. It is the prior of a target trial in the "
            f"application the cost stands for, as a fraction: 0.05 in "
            f"VoxSRC, 0.01 and 0.005 in NIST's SRE 2016 and 2019 CTS")
    return value


def _weights(p_target, c_miss, c_fa, name: str) -> Tuple[float, float]:
    """``(c_miss p_target, c_fa (1 - p_target))``, the parameters checked."""
    p_target = _cost_parameter(p_target, 'p_target', upper=1.0, name=name)
    c_miss = _cost_parameter(c_miss, 'c_miss', name=name)
    c_fa = _cost_parameter(c_fa, 'c_fa', name=name)
    weight_miss = c_miss * p_target
    weight_fa = c_fa * (1.0 - p_target)
    if min(weight_miss, weight_fa) < sys.float_info.min:
        # 0 would be divided by, and a subnormal float has lost precision
        raise ValueError(
            f"{name}(): c_miss p_target ({weight_miss:.3g}) and c_fa "
            f"(1 - p_target) ({weight_fa:.3g}) have to be at least "
            f"{sys.float_info.min:.3g}, the smallest normal float64. The "
            f"normalized cost depends only on their ratio, so scaling c_miss "
            f"and c_fa up by one factor leaves it unchanged")
    return weight_miss, weight_fa


def min_dcf(scores, labels, *, p_target: float, c_miss: float = 1.0,
            c_fa: float = 1.0) -> float:
    """Minimum normalized detection cost of a trial list, as NIST's speaker
    recognition evaluations define it. 0 is perfect; 1 is no better than
    deciding without the scores.

    VoxSRC's cost, then SRE 2008's:

    >>> kudio.min_dcf(scores, labels, p_target=0.05)              # doctest: +SKIP
    >>> kudio.min_dcf(scores, labels, p_target=0.01, c_miss=10)   # doctest: +SKIP

    The lowest, over every threshold, of::

        (c_miss p_target P_miss + c_fa (1 - p_target) P_fa) / C_Default
        C_Default = min(c_miss p_target, c_fa (1 - p_target))

    The thresholds are the operating points of :func:`det_curve`, including
    accepting everything and rejecting everything, so the result never
    exceeds 1. Tied scores are one threshold, as there.

    :param scores: one score per trial, higher for "more likely a target".
    :param labels: one per trial, True or 1 for a target trial and False or
        0 for a non-target one.
    :param p_target: prior probability of a target trial in the application
        the cost stands for, strictly between 0 and 1. It has no default,
        because the evaluations disagree: 0.001 in NIST's SRE 2010, 0.01 and
        0.005 in SRE 2016 and the 2019 CTS challenge (whose primary cost
        averages the two), 0.05 in the SRE 2019 audio-visual evaluation and
        VoxSRC; the module docstring has the table.
    :param c_miss: cost of a miss: 10 in SRE 2008 and in SRE 2010's
        historical cost, 1 in the later plans.
    :param c_fa: cost of a false alarm, 1 in every plan cited here.
    :raises ValueError: for a *p_target* outside (0, 1), a cost that is not
        positive and finite, a parameter that is not a number, weights
        c_miss p_target or c_fa (1 - p_target) below 2.2e-308, the smallest
        normal float64, and for the trial lists :func:`det_curve` refuses.
    :returns: the normalized minimum cost, 0 to 1. nan, with a warning, for
        a list without a target or without a non-target trial.
    """
    weight_miss, weight_fa = _weights(p_target, c_miss, c_fa, 'min_dcf')
    counts = _counted(scores, labels, 'min_dcf')
    why = _unscorable(counts, 'min_dcf')
    if why is not None:
        warn(f"{why}; scored nan")
        return float('nan')
    # rates first: they are exactly 0 and 1 at the two ends, so the cheaper
    # end costs exactly C_Default and the result cannot round past 1
    cost = (weight_miss * (counts.misses / counts.targets)
            + weight_fa * (counts.false_alarms / counts.nontargets))
    return float(np.min(cost)) / min(weight_miss, weight_fa)


# --- calibration: the scores read as likelihood ratios ------------------------

def _llr_lists(llrs, labels, name: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """``(target llrs, non-target llrs)``, or None after the warning for a
    list without one of the two."""
    llrs = _scores(llrs, name, what='llrs')
    labels = _labels(labels, name)
    if llrs.size != labels.size:
        raise ValueError(f"{name}(): {llrs.size} llrs and {labels.size} labels; "
                         f"they pair up by position, one of each per trial")
    targets, nontargets = llrs[labels], llrs[~labels]
    why = _unscorable(_Counts(np.zeros(0), np.zeros(0), np.zeros(0),
                              targets.size, nontargets.size), name)
    if why is not None:
        warn(f"{why}; scored nan")
        return None
    return targets, nontargets


def cllr(llrs, labels) -> float:
    """Log-likelihood-ratio cost of a trial list, in bits: how much the
    scores, read as likelihood ratios, cost over every application at once.
    0 is perfect; 1 is what LLRs of 0 throughout cost, a system that always
    says it cannot tell.

    >>> kudio.cllr(llrs, labels)                                  # doctest: +SKIP

    Brümmer and du Preez's C_llr [BD06]::

        C_llr = ( mean_tar log2(1 + exp(-llr)) + mean_non log2(1 + exp(llr)) ) / 2

    It charges for discrimination and for calibration together: a list ranked
    perfectly still costs where its LLRs say less, or more, than the
    evidence holds. :func:`min_cllr` is what the same ranking costs once
    calibrated as well as it can be, and the difference is the cost of
    calibration.

    :param llrs: one natural-log likelihood ratio per trial, as BOSARIS and
        NIST's SRE plans take them; a base-10 LLR, the forensic custom, has
        to be multiplied by ln 10 = 2.3026 first. Raw scores, cosine
        similarities say, are not LLRs, and their C_llr says little:
        calibrate them first. +inf and -inf are LLRs of certainty, and a
        certainty that is wrong costs inf.
    :param labels: as :func:`det_curve`.
    :raises ValueError: for an llr that is NaN, and for the lists
        :func:`det_curve` refuses on other grounds.
    :returns: C_llr in bits. nan, with a warning, for a list without a
        target or without a non-target trial.
    """
    lists = _llr_lists(llrs, labels, 'cllr')
    if lists is None:
        return float('nan')
    targets, nontargets = lists
    # log(1 + exp(x)) without overflow: logaddexp(0, x)
    return float((np.mean(np.logaddexp(0.0, -targets))
                  + np.mean(np.logaddexp(0.0, nontargets))) / (2.0 * math.log(2.0)))


def _pav_blocks(llrs: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """``(targets, trials)`` of each block of the pool-adjacent-violators
    fit, in rising score order: the target share of each block is the
    calibrated posterior of every trial in it.

    Tied scores start out in one block, since no threshold can part them, so
    the fit does not depend on the order the list leaves them in; PYLLR's
    PAV, which ranks the targets of a tie below its non-targets, comes to the
    same blocks. The pooling compares the shares as fractions of integers,
    so it is exact.
    """
    order = np.argsort(llrs, kind='stable')
    ranked, target = llrs[order], labels[order]
    first = np.empty(ranked.size, dtype=bool)
    first[:1] = True
    np.not_equal(ranked[1:], ranked[:-1], out=first[1:])
    starts = np.flatnonzero(first)
    hits = np.add.reduceat(target.astype(np.int64), starts).tolist()
    sizes = np.diff(np.append(starts, ranked.size)).tolist()
    pooled_hits, pooled_sizes = [], []
    for t, n in zip(hits, sizes):
        # merge while the block below has the larger share: t0/n0 > t/n
        while pooled_hits and pooled_hits[-1] * n > t * pooled_sizes[-1]:
            t += pooled_hits.pop()
            n += pooled_sizes.pop()
        pooled_hits.append(t)
        pooled_sizes.append(n)
    return np.array(pooled_hits, dtype=np.int64), np.array(pooled_sizes, dtype=np.int64)


def min_cllr(llrs, labels) -> float:
    """C_llr of the best calibration the ranking allows, in bits: what
    :func:`cllr` would be if the scores were turned into LLRs by the one
    rising mapping that suits this list best.

    >>> kudio.min_cllr(scores, labels)                            # doctest: +SKIP

    The mapping is the pool-adjacent-violators fit of the target share
    against the scores, which gives each trial a posterior probability; less
    the log of the list's own prior odds, T/N, those are LLRs, and their
    C_llr is this [BD06]. The scores only have to rank the trials, so raw
    scores will do. Tied scores share one LLR, as they share one threshold
    in :func:`det_curve`.

    BOSARIS's ``min_cllr.m`` sorts the non-targets of a tie before its
    targets and so lets the fit split ties, which reads lower where targets
    and non-targets share scores: by up to 0.30 bits on the rounded lists of
    kudio's validation script. It also adds ``1e-6 i / N`` to the i-th LLR to
    keep them strictly rising, which moves the result by under 1e-7 bits
    there. PYLLR, by the same author, keeps ties together, as here.

    :param llrs: one score per trial, higher for "more likely a target";
        LLRs or not. +inf and -inf rank like any other score.
    :param labels: as :func:`det_curve`.
    :raises ValueError: as :func:`cllr`.
    :returns: min C_llr in bits, from 0 to 1, never above :func:`cllr` of
        the same list. nan, with a warning, for a list without a target or
        without a non-target trial.
    """
    lists = _llr_lists(llrs, labels, 'min_cllr')
    if lists is None:
        return float('nan')
    targets, nontargets = lists
    scores = np.concatenate([targets, nontargets])
    is_target = np.concatenate([np.ones(targets.size, dtype=bool),
                                np.zeros(nontargets.size, dtype=bool)])
    hits, sizes = _pav_blocks(scores, is_target)
    misses = sizes - hits
    T, N = targets.size, nontargets.size
    # a target in a block with share t/n costs log(1 + (n - t) T / (t N)),
    # a non-target log(1 + t N / ((n - t) T)); a block holds a target only
    # if t > 0 and a non-target only if n - t > 0, so neither divides by 0
    with_targets = hits > 0
    with_misses = misses > 0
    tar = np.sum(hits[with_targets] * np.log1p(
        (misses[with_targets] * T) / (hits[with_targets] * N)))
    non = np.sum(misses[with_misses] * np.log1p(
        (hits[with_misses] * N) / (misses[with_misses] * T)))
    return float((tar / T + non / N) / (2.0 * math.log(2.0)))


def act_dcf(llrs, labels, *, p_target: float, c_miss: float = 1.0,
            c_fa: float = 1.0) -> float:
    """Actual normalized detection cost: the cost of the decisions the LLRs
    themselves make, at the threshold Bayes' rule sets for the application.
    0 is perfect; 1 is no better than deciding without the scores.

    >>> kudio.act_dcf(llrs, labels, p_target=0.01)                # doctest: +SKIP

    A trial is accepted when its LLR is at or above
    ``log(beta)``, ``beta = c_fa (1 - p_target) / (c_miss p_target)``, the
    threshold of NIST's SRE 2016 and 2019 CTS plans (Sec. 3.1), and the cost
    is :func:`min_dcf`'s, normalized the same way, at that one threshold
    instead of the best. A trial exactly at the threshold is accepted, as
    PYLLR's ``fast_Bayes_error_rate`` accepts it. The gap to :func:`min_dcf`
    is what calibration costs at that operating point.

    :param llrs: natural-log likelihood ratios, as :func:`cllr`.
    :param p_target: as :func:`min_dcf`; 0.01 and 0.005 in SRE 2016.
    :param c_miss: as :func:`min_dcf`.
    :param c_fa: as :func:`min_dcf`.
    :raises ValueError: as :func:`min_dcf`, and for an llr that is NaN.
    :returns: the normalized actual cost, from 0 up; above 1 when the LLRs
        decide worse than the prior alone. nan, with a warning, for a list
        without a target or without a non-target trial.
    """
    weight_miss, weight_fa = _weights(p_target, c_miss, c_fa, 'act_dcf')
    lists = _llr_lists(llrs, labels, 'act_dcf')
    if lists is None:
        return float('nan')
    targets, nontargets = lists
    threshold = math.log(weight_fa / weight_miss)
    p_miss = np.count_nonzero(targets < threshold) / targets.size
    p_fa = np.count_nonzero(nontargets >= threshold) / nontargets.size
    return (weight_miss * p_miss + weight_fa * p_fa) / min(weight_miss, weight_fa)


#: The two priors NIST's SRE 2016 and the SRE 2019 CTS challenge average
#: their primary cost over, with unit costs (Sec. 3.1, Table 1 of the 2016
#: plan).
C_PRIMARY_PRIORS = (0.01, 0.005)


def c_primary(llrs, labels, *, minimum: bool = False) -> float:
    """NIST's primary cost of SRE 2016 and the SRE 2019 CTS challenge: the
    mean of :func:`act_dcf` at p_target 0.01 and 0.005, with unit costs.

    >>> kudio.c_primary(llrs, labels)                             # doctest: +SKIP
    >>> kudio.c_primary(scores, labels, minimum=True)             # doctest: +SKIP

    :param minimum: average :func:`min_dcf` instead, the "minCprimary" that
        NIST's scoring script prints beside it.
    :returns: the mean of the two normalized costs. nan, with a warning, as
        :func:`act_dcf` gives it.

    The evaluations average the cost over partitions of the trial list,
    language and gender among them, with each partition's trials weighted
    equally (SRE 2016 plan, Sec. 3.1); this pools the list as it comes. To
    match an official number, compute the cost per partition and average
    them as the plan says.
    """
    if not isinstance(minimum, (bool, np.bool_)):
        raise ValueError(f"c_primary(): minimum has to be True or False, got "
                         f"{minimum!r}")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        measure = min_dcf if minimum else act_dcf
        try:
            costs = [measure(llrs, labels, p_target=p) for p in C_PRIMARY_PRIORS]
        except ValueError as e:
            text = str(e)
            raise ValueError("c_primary(): " + text.split("(): ", 1)[-1]) from None
    if caught:
        # the two costs fail for one reason; say it once, by this name
        text = str(caught[0].message)
        warn("c_primary(): " + text.split("(): ", 1)[1] if "(): " in text else text)
        return float('nan')
    return (costs[0] + costs[1]) / 2.0
