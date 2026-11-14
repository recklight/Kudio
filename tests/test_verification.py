# -*- coding: utf-8 -*-
"""EER, min DCF and the DET curve: against the VoxSRC and NIST scoring rules
and scikit-learn's ROC curve, against exact sweeps and hulls, what ties, cost
parameters and small lists do to them, and the trial lists they refuse or
cannot score."""
from __future__ import annotations

import bisect
import math
import warnings
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest

from kudio.core import verification as module
from kudio.core.verification import DetCurve, det_curve, eer, min_dcf


def _gaussian(targets, nontargets, shift, seed, decimals=None):
    """Scores N(shift, 1) for targets and N(0, 1) for non-targets, shuffled,
    rounded to *decimals* when given, which makes ties."""
    rng = np.random.default_rng(seed)
    labels = np.r_[np.ones(targets, bool), np.zeros(nontargets, bool)]
    scores = rng.normal(size=labels.size) + shift * labels
    if decimals is not None:
        scores = np.round(scores, decimals)
    order = rng.permutation(labels.size)
    return scores[order], labels[order]


def _forty_thousand(seed=0):
    """The list the module docstring quotes: 40,000 trials, half targets."""
    rng = np.random.default_rng(seed)
    labels = np.r_[np.ones(20000, bool), np.zeros(20000, bool)]
    return rng.normal(0, 1, 40000) + 4.65 * labels, labels


def _twenty_two_thousand():
    """The module docstring's tied list: 2,000 targets N(3, 1) and 20,000
    non-targets N(0, 1), rounded to 0.1, targets listed first."""
    rng = np.random.default_rng(0)
    labels = np.r_[np.ones(2000, bool), np.zeros(20000, bool)]
    return np.round(rng.normal(0, 1, labels.size) + 3.0 * labels, 1), labels


#: the module docstring's eleven trials, ranked by score
ELEVEN = (list(range(11)), [0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1])


def _small_lists(count, seed):
    """Random lists of 2 to 30 trials, two in three of them full of ties."""
    rng = np.random.default_rng(seed)
    lists = []
    while len(lists) < count:
        n = int(rng.integers(2, 31))
        labels = rng.random(n) < rng.uniform(0.2, 0.8)
        if labels.all() or not labels.any():
            continue
        if len(lists) % 3 == 0:
            scores = rng.normal(size=n) + 1.5 * labels
        else:
            scores = (rng.integers(0, int(rng.integers(1, 7)), n)
                      + rng.integers(0, 3) * labels).astype(float)
        lists.append((scores, labels))
    return lists


def _points(scores, labels):
    """(P_miss, P_fa) in exact fractions at every distinct score and above
    the highest, counted with bisect on each class: no code shared with
    kudio."""
    scores, labels = np.asarray(scores, float), np.asarray(labels, bool)
    targets, nontargets = sorted(scores[labels]), sorted(scores[~labels])
    cuts = sorted(set(scores.tolist())) + [math.inf]
    return [(Fraction(bisect.bisect_left(targets, c), len(targets)),
             Fraction(len(nontargets) - bisect.bisect_left(nontargets, c), len(nontargets)))
            for c in cuts]


def _line_crossing(points):
    for (m0, f0), (m1, f1) in zip(points, points[1:]):
        if m0 == f0:
            return m0
        if m0 < f0 and m1 >= f1:
            d0, d1 = m0 - f0, m1 - f1
            return m0 + (-d0 / (d1 - d0)) * (m1 - m0)
    raise AssertionError("no crossing")


def _hull_crossing(points):
    """Andrew's monotone chain on (P_fa, P_miss), cut at (1, 0), and where
    the hull crosses P_miss = P_fa."""
    hull = []
    for p in sorted(set((f, m) for m, f in points)):
        while len(hull) >= 2 and ((hull[-1][0] - hull[-2][0]) * (p[1] - hull[-2][1])
                                  - (hull[-1][1] - hull[-2][1]) * (p[0] - hull[-2][0])) <= 0:
            hull.pop()
        hull.append(p)
    hull = hull[:hull.index((1, 0)) + 1]
    for (f0, m0), (f1, m1) in zip(hull, hull[1:]):
        if m0 == f0:
            return m0
        if m0 > f0 and m1 <= f1:
            d0, d1 = m0 - f0, m1 - f1
            return m0 + (d0 / (d0 - d1)) * (m1 - m0)
    raise AssertionError("no crossing")


def _lowest_cost(points, p_target, c_miss=1.0, c_fa=1.0):
    p, cm, cf = Fraction(p_target), Fraction(c_miss), Fraction(c_fa)
    return min(cm * p * m + cf * (1 - p) * f for m, f in points) / min(cm * p, cf * (1 - p))


def _kaldi_min_dcf(scores, labels, p_target):
    """The rule of Kaldi's compute_min_dcf.py, which the VoxSRC scorer and
    voxceleb_trainer carry: a stable sort, and an operating point after
    every trial, tied or not."""
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranked = np.array([int(labels[i]) for i in order])
    fnr = np.cumsum(ranked) / ranked.sum()
    fpr = 1 - np.cumsum(1 - ranked) / (ranked.size - ranked.sum())
    return np.min(fnr * p_target + fpr * (1 - p_target)) / min(p_target, 1 - p_target)


def _nist_sre16_eer(scores, labels):
    """compute_pmiss_pfa_rbst and compute_eer of NIST's SRE 2016 scoring
    script (version 4.1), with a stable sort in place of its argsort so that
    tied trials stay in the order they are listed: an operating point after
    every sorted trial, and the line between the two either side of the
    crossing."""
    ranked = np.asarray(labels)[np.argsort(scores, kind='stable')].astype(float)
    fnr = np.cumsum(ranked) / np.sum(ranked)
    fpr = 1 - np.cumsum(1 - ranked) / np.sum(1 - ranked)
    diff = fnr - fpr
    x1 = np.flatnonzero(diff >= 0)[0]
    x2 = np.flatnonzero(diff < 0)[-1]
    a = (fnr[x1] - fpr[x1]) / (fpr[x2] - fpr[x1] - (fnr[x2] - fnr[x1]))
    return fnr[x1] + a * (fnr[x2] - fnr[x1])


def _quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return fn(*args, **kwargs)


# ---------------------------------------------------- against the references

# The VoxSRC 2020-2023 scorer's EER: calculate_eer of its compute_EER.py,
# brentq on interp1d of roc_curve, run with scikit-learn 1.7.2 and scipy
# 1.15.2 on these lists; brentq stops within 2e-12 of the crossing.
_VOXSRC = [
    (dict(targets=300, nontargets=700, shift=1.5, seed=11), 0.22333333333333355),
    (dict(targets=300, nontargets=700, shift=1.5, seed=11, decimals=1), 0.22200000000000003),
    (dict(targets=40, nontargets=4000, shift=2.5, seed=12), 0.10150000000149247),
]


@pytest.mark.parametrize("spec, golden", _VOXSRC)
def test_the_default_is_the_crossing_the_voxsrc_scorer_finds(spec, golden):
    assert _quiet(eer, *_gaussian(**spec)) == pytest.approx(golden, abs=1e-11)


# compute_pmiss_pfa_rbst and compute_eer of NIST's SRE 2016 scoring script
# (version 4.1), restated in tools/validation/verification.py and run with
# numpy 2.2.6, on the lists above without ties
_NIST_SRE16 = [
    (dict(targets=300, nontargets=700, shift=1.5, seed=11), 0.22333333333333333),
    (dict(targets=40, nontargets=4000, shift=2.5, seed=12), 0.10150000000000003),
]


@pytest.mark.parametrize("spec, golden", _NIST_SRE16)
def test_the_default_is_the_crossing_nists_sre16_script_finds(spec, golden):
    assert eer(*_gaussian(**spec)) == pytest.approx(golden, rel=1e-14)


# ROCCH-EER from an independent hull: Andrew's monotone chain in exact
# fractions (tools/validation/verification.py), on the same lists
_ROCCH = [
    (dict(targets=300, nontargets=700, shift=1.5, seed=11), 0.21673758865248227),
    (dict(targets=300, nontargets=700, shift=1.5, seed=11, decimals=1), 0.2175591985428051),
    (dict(targets=40, nontargets=4000, shift=2.5, seed=12), 0.09152892561983471),
]


@pytest.mark.parametrize("spec, golden", _ROCCH)
def test_rocch_is_the_crossing_of_an_independently_built_hull(spec, golden):
    assert eer(*_gaussian(**spec), method='rocch') == golden


# The lowest normalized cost over the points of
# roc_curve(drop_intermediate=False), scikit-learn 1.7.2, and on the lists
# without ties at p_target 0.05 the rule of Kaldi's compute_min_dcf.py,
# restated in tools/validation/verification.py: (spec, p_target, roc_curve,
# Kaldi)
_MIN_DCF = [
    (dict(targets=300, nontargets=700, shift=1.5, seed=11), 0.05,
     0.9476190476190477, 0.9476190476190482),
    (dict(targets=300, nontargets=700, shift=1.5, seed=11), 0.01, 0.9933333333333334, None),
    (dict(targets=300, nontargets=700, shift=1.5, seed=11, decimals=1), 0.05,
     0.9476190476190477, None),
    (dict(targets=300, nontargets=700, shift=1.5, seed=11, decimals=1), 0.01,
     0.9966666666666667, None),
    (dict(targets=40, nontargets=4000, shift=2.5, seed=12), 0.05,
     0.5734999999999999, 0.573499999999999),
    (dict(targets=40, nontargets=4000, shift=2.5, seed=12), 0.01, 0.7242500000000001, None),
]


@pytest.mark.parametrize("spec, p_target, on_roc, kaldi", _MIN_DCF)
def test_min_dcf_is_the_lowest_cost_on_the_roc_curve(spec, p_target, on_roc, kaldi):
    got = _quiet(min_dcf, *_gaussian(**spec), p_target=p_target)
    assert got == pytest.approx(on_roc, rel=1e-13)
    if kaldi is not None:
        assert got == pytest.approx(kaldi, rel=1e-13)


def test_det_curve_is_scikit_learns_roc_curve_turned_round():
    """roc_curve with drop_intermediate=False has the same points, from the
    highest threshold down, with P_miss as 1 - tpr."""
    roc_curve = pytest.importorskip("sklearn.metrics").roc_curve
    for spec, _ in _VOXSRC:
        scores, labels = _gaussian(**spec)
        fpr, tpr, thresholds = roc_curve(labels, scores, drop_intermediate=False)
        curve = det_curve(scores, labels)
        assert np.array_equal(curve.thresholds[::-1], thresholds)
        np.testing.assert_allclose(curve.p_fa[::-1], fpr, rtol=0, atol=1e-15)
        np.testing.assert_allclose(curve.p_miss[::-1], 1 - tpr, rtol=0, atol=1e-15)


def test_random_lists_match_the_voxsrc_rule_run_here():
    """The VoxSRC scorer's rule run live on 200 small lists, most with ties,
    where its straight lines through tied scores matter."""
    roc_curve = pytest.importorskip("sklearn.metrics").roc_curve
    from scipy.interpolate import interp1d
    from scipy.optimize import brentq
    for scores, labels in _small_lists(200, seed=1):
        fpr, tpr, _ = roc_curve(labels, scores)
        voxsrc = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
        assert eer(scores, labels) == pytest.approx(voxsrc, abs=1e-11)


def test_random_lists_match_an_exact_sweep_and_an_exact_hull():
    for scores, labels in _small_lists(300, seed=2):
        points = _points(scores, labels)
        assert eer(scores, labels) == float(_line_crossing(points))
        assert eer(scores, labels, method='rocch') == float(_hull_crossing(points))
        for p in (0.5, 0.05, 0.001, 0.9):
            assert min_dcf(scores, labels, p_target=p) == pytest.approx(
                float(_lowest_cost(points, p)), rel=1e-14, abs=1e-15)
        assert min_dcf(scores, labels, p_target=0.01, c_miss=100) == pytest.approx(
            float(_lowest_cost(points, 0.01, c_miss=100)), rel=1e-14, abs=1e-15)
        curve = det_curve(scores, labels)
        assert curve.p_miss.tolist() == [float(m) for m, f in points]
        assert curve.p_fa.tolist() == [float(f) for m, f in points]


def test_the_hull_crossing_is_the_largest_minimum_cost_over_priors():
    """Eq. 26 of the BOSARIS paper: the lowest p P_miss + (1 - p) P_fa, as a
    function of p, is concave and piecewise linear, and its peak, at a prior
    where two operating points cost the same, is the ROCCH-EER."""
    for scores, labels in _small_lists(120, seed=3):
        points = _points(scores, labels)
        priors = {Fraction(0), Fraction(1)}
        for m0, f0 in points:
            for m1, f1 in points:
                if (m0 - f0) != (m1 - f1):
                    p = (f1 - f0) / ((m0 - f0) - (m1 - f1))
                    if 0 <= p <= 1:
                        priors.add(p)
        peak = max(min(p * m + (1 - p) * f for m, f in points) for p in priors)
        assert eer(scores, labels, method='rocch') == float(peak)
        # and through min_dcf, which normalizes by min(p, 1 - p)
        for p in (0.02, 0.2, 0.5, 0.7):
            assert min_dcf(scores, labels, p_target=p) * min(p, 1 - p) <= float(peak) + 1e-15


def test_a_hull_with_hundreds_of_vertices_is_found_in_a_few_chords():
    """One vertex for every primitive step (a, b) with a + b <= 40: 489 steps
    over 13,110 trials. The chord search lands on the same crossing as the
    monotone chain."""
    pairs = sorted(((a, b) for a in range(1, 40) for b in range(1, 40)
                    if a + b <= 40 and math.gcd(a, b) == 1),
                   key=lambda ab: ab[1] / ab[0], reverse=True)
    labels = []
    for a, b in pairs:
        labels += [False] * a + [True] * b
    labels = np.array(labels[::-1])
    scores = np.arange(labels.size, dtype=float)
    assert len(pairs) == 489 and labels.size == 13110
    counted = module._counted(scores, labels, 'eer')
    gap = counted.misses * counted.nontargets - counted.false_alarms * counted.targets
    crossing, chords = module._hull_crossing(counted, gap)
    assert crossing == _hull_crossing(_points(scores, labels))
    assert chords <= 20


# ------------------------------------------------------- what the numbers mean

def test_eleven_trials_read_one_third_by_default_and_three_fourteenths_on_the_hull():
    """The module docstring's example. The crossing lies on the run of three
    targets at P_fa = 1/3; the hull runs from (P_fa 1/3, P_miss 0) to
    (0, 3/5) and crosses at 3/14."""
    assert _quiet(eer, *ELEVEN) == pytest.approx(1 / 3, rel=1e-15)
    assert _quiet(eer, *ELEVEN, method='rocch') == pytest.approx(3 / 14, rel=1e-15)
    curve = det_curve(*ELEVEN)
    # the point voxceleb_trainer would have needed: P_fa 1/3, P_miss 0.4
    assert any(f == pytest.approx(1 / 3) and m == pytest.approx(0.4)
               for m, f in zip(curve.p_miss, curve.p_fa))


def test_the_cheaper_of_the_two_decisions_without_scores_normalizes_the_cost():
    """C_Default is c_fa (1 - p_target) where accepting everything costs less
    than rejecting everything. On the eleven trials the best threshold at
    either setting below is the fifth score: no target missed, 2 of the 6
    non-targets accepted, so a normalized cost of 1/3 (exact sweep in
    fractions); divided by c_miss p_target it would read 1/27 and 0.33."""
    assert min_dcf(*ELEVEN, p_target=0.9) == pytest.approx(1 / 3, rel=1e-15)
    assert min_dcf(*ELEVEN, p_target=0.01, c_miss=100) == pytest.approx(1 / 3, rel=1e-15)
    points = _points(*ELEVEN)
    assert _lowest_cost(points, 0.9) == Fraction(1, 3)
    assert _lowest_cost(points, 0.01, c_miss=100) == Fraction(1, 3)


def test_forty_thousand_trials_read_1_090_and_1_081_percent():
    scores, labels = _forty_thousand()
    assert eer(scores, labels) == pytest.approx(0.0109, rel=1e-15)
    assert eer(scores, labels, method='rocch') == pytest.approx(0.010809090909090909, rel=1e-15)


def test_a_threshold_reaches_the_default_unless_classes_tie_at_the_crossing():
    """Where the step across P_miss = P_fa moves only one of the two rates,
    the trials it passes are all targets or all non-targets, and one
    threshold has one rate equal to the EER and the other at or below it.
    Where it moves both, a target and a non-target are tied there, and no
    threshold gets as low."""
    mixed = 0
    for scores, labels in _small_lists(300, seed=4):
        curve = det_curve(scores, labels)
        best = np.min(np.maximum(curve.p_miss, curve.p_fa))
        default = eer(scores, labels)
        k = int(np.argmax(curve.p_miss >= curve.p_fa))
        if curve.p_miss[k] == curve.p_fa[k]:
            assert default == best
        elif curve.p_miss[k] != curve.p_miss[k - 1] and curve.p_fa[k] != curve.p_fa[k - 1]:
            mixed += 1
            assert default < best
        else:
            assert default == best
    assert mixed > 20
    # two targets and two non-targets, one of each tied in the middle
    scores, labels = [0, 1, 1, 2], [0, 0, 1, 1]
    curve = det_curve(scores, labels)
    assert eer(scores, labels) == pytest.approx(0.25)
    assert np.min(np.maximum(curve.p_miss, curve.p_fa)) == pytest.approx(0.5)


def test_the_hull_is_never_above_the_default():
    for scores, labels in _small_lists(300, seed=5):
        assert eer(scores, labels, method='rocch') <= eer(scores, labels) + 1e-15


def test_ties_are_one_threshold_whatever_the_order_of_the_list():
    scores, labels = _gaussian(200, 800, 1.0, seed=6, decimals=1)
    want = (eer(scores, labels), eer(scores, labels, method='rocch'),
            min_dcf(scores, labels, p_target=0.05), det_curve(scores, labels))
    rng = np.random.default_rng(7)
    for _ in range(5):
        order = rng.permutation(scores.size)
        s, y = scores[order], labels[order]
        assert eer(s, y) == want[0]
        assert eer(s, y, method='rocch') == want[1]
        assert min_dcf(s, y, p_target=0.05) == want[2]
        curve = det_curve(s, y)
        assert all(np.array_equal(a, b) for a, b in zip(curve, want[3]))


def test_splitting_ties_reads_lower_than_any_threshold_can():
    """The module docstring's 22,000 trials, scores rounded to a tenth of
    their standard deviation. Kaldi's compute_min_dcf.py rule puts a point
    after every trial; listed non-targets first, its points inside the ties
    reject non-targets before targets and give 0.39965 where no threshold
    does better than 0.43615. Listed targets first, it agrees."""
    scores, labels = _twenty_two_thousand()
    nontargets_first = np.r_[np.arange(2000, 22000), np.arange(2000)]
    assert min_dcf(scores, labels, p_target=0.05) == pytest.approx(0.43615, rel=1e-12)
    assert min_dcf(scores[nontargets_first], labels[nontargets_first],
                   p_target=0.05) == pytest.approx(0.43615, rel=1e-12)
    assert _kaldi_min_dcf(list(scores[nontargets_first]), list(labels[nontargets_first]),
                          0.05) == pytest.approx(0.39965, rel=1e-12)
    assert _kaldi_min_dcf(list(scores), list(labels), 0.05) == pytest.approx(0.43615, rel=1e-12)


def test_nists_script_reads_an_eer_that_moves_with_the_order_of_tied_trials():
    """The same 22,000 trials. NIST's SRE 2016 script puts an operating point
    after every sorted trial, so where tied trials fall in its sort moves its
    EER: 0.05985 with the ties non-targets first, 0.07050 with targets
    first. The default reads 0.06535 in any order."""
    scores, labels = _twenty_two_thousand()
    nontargets_first = np.r_[np.arange(2000, 22000), np.arange(2000)]
    assert _nist_sre16_eer(scores[nontargets_first], labels[nontargets_first]) == \
        pytest.approx(0.05985, rel=1e-12)
    assert _nist_sre16_eer(scores, labels) == pytest.approx(0.0705, rel=1e-12)
    assert eer(scores, labels) == eer(scores[nontargets_first], labels[nontargets_first])
    assert eer(scores, labels) == pytest.approx(0.06535, abs=5e-6)
    # without ties the restated rule gives the stored golden of the script
    spec, golden = _NIST_SRE16[0]
    assert _nist_sre16_eer(*_gaussian(**spec)) == pytest.approx(golden, rel=1e-14)


def test_perfect_reversed_and_uninformative_lists():
    assert eer([1, 2, 3, 4], [0, 0, 1, 1]) == 0.0
    assert min_dcf([1, 2, 3, 4], [0, 0, 1, 1], p_target=0.01) == 0.0
    # ranked the wrong way round: every threshold errs on one side fully
    assert eer([1, 2, 3, 4], [1, 1, 0, 0]) == 1.0
    assert eer([1, 2, 3, 4], [1, 1, 0, 0], method='rocch') == 0.5
    assert min_dcf([1, 2, 3, 4], [1, 1, 0, 0], p_target=0.01) == 1.0
    # one score for every trial: accept all or reject all
    assert eer([5, 5, 5, 5], [0, 1, 0, 1]) == 0.5
    assert eer([5, 5, 5, 5], [0, 1, 0, 1], method='rocch') == 0.5
    assert min_dcf([5, 5, 5, 5], [0, 1, 0, 1], p_target=0.3) == 1.0


def test_only_the_order_of_the_scores_matters():
    scores, labels = _gaussian(300, 700, 1.5, seed=11)
    ranks = np.argsort(np.argsort(scores))
    for transformed in (np.exp(scores), 7.0 * scores - 3.0, ranks, np.arctan(scores)):
        assert eer(transformed, labels) == eer(scores, labels)
        assert (eer(transformed, labels, method='rocch')
                == eer(scores, labels, method='rocch'))
        assert (min_dcf(transformed, labels, p_target=0.01)
                == min_dcf(scores, labels, p_target=0.01))


def test_the_normalized_cost_never_exceeds_one():
    """Accepting everything and rejecting everything are operating points,
    and the cheaper of the two is what the cost is divided by."""
    for scores, labels in _small_lists(200, seed=8):
        for p in (0.001, 0.3, 0.6, 0.999):
            assert 0.0 <= min_dcf(scores, labels, p_target=p, c_miss=3.0) <= 1.0


def test_costs_and_their_effective_prior_give_one_normalized_cost():
    """SRE 2008's c_miss 10 at p_target 0.01 is unit costs at the effective
    prior 0.1 / 1.09 = 0.0917, and not unit costs at 0.01: 0.059 against
    0.143 on the module docstring's 40,000 trials."""
    scores, labels = _forty_thousand()
    sre08 = min_dcf(scores, labels, p_target=0.01, c_miss=10)
    assert sre08 == pytest.approx(min_dcf(scores, labels, p_target=0.1 / 1.09), rel=1e-12)
    assert sre08 == pytest.approx(0.059, abs=5e-4)
    assert min_dcf(scores, labels, p_target=0.01) == pytest.approx(0.143, abs=5e-4)
    for scores, labels in _small_lists(50, seed=9):
        for p, c_miss, c_fa in ((0.01, 10.0, 1.0), (0.2, 1.0, 4.0), (0.6, 2.5, 0.5)):
            effective = c_miss * p / (c_miss * p + c_fa * (1 - p))
            assert min_dcf(scores, labels, p_target=p, c_miss=c_miss, c_fa=c_fa) == \
                pytest.approx(min_dcf(scores, labels, p_target=effective), rel=1e-12, abs=1e-15)


def test_one_false_alarm_among_ten_thousand_costs_0_0999_at_p_target_0_001():
    """With unit costs the normalized cost is P_miss + 999 P_fa at 0.001. A
    single non-target scored above every target is either accepted with all
    of them, 999 / 10,000, or every target is missed, 1."""
    labels = np.r_[np.ones(100, bool), np.zeros(10000, bool)]
    scores = np.r_[np.linspace(1, 2, 100), np.linspace(-2, 0, 9999), [3.0]]
    assert min_dcf(scores, labels, p_target=0.001) == pytest.approx(0.0999, rel=1e-12)


def test_det_curve_runs_from_accepting_everything_to_rejecting_everything():
    scores = [0.3, 0.1, 0.3, 0.9, 0.5, 0.5, 0.5]
    labels = [0, 0, 1, 1, 1, 0, 1]
    curve = _quiet(det_curve, scores, labels)
    assert isinstance(curve, DetCurve)
    assert curve.thresholds.tolist() == [0.1, 0.3, 0.5, 0.9, math.inf]
    assert curve.p_miss.tolist() == [0.0, 0.0, 0.25, 0.75, 1.0]
    assert curve.p_fa.tolist() == pytest.approx([1.0, 2 / 3, 1 / 3, 0.0, 0.0])
    assert curve.thresholds.dtype == np.float64


def test_det_curves_recipes_give_the_thresholds_eer_and_min_dcf_describe():
    scores, labels = _gaussian(300, 700, 1.5, seed=11)
    curve = det_curve(scores, labels)
    at_eer = curve.thresholds[np.argmin(np.maximum(curve.p_miss, curve.p_fa))]
    accepted = scores >= at_eer
    p_miss = np.mean(~accepted[labels])
    p_fa = np.mean(accepted[~labels])
    assert max(p_miss, p_fa) == pytest.approx(eer(scores, labels), rel=1e-15)
    at_cost = curve.thresholds[np.argmin(0.05 * curve.p_miss + 0.95 * curve.p_fa)]
    accepted = scores >= at_cost
    cost = (0.05 * np.mean(~accepted[labels]) + 0.95 * np.mean(accepted[~labels])) / 0.05
    assert cost == pytest.approx(min_dcf(scores, labels, p_target=0.05), rel=1e-12)


# ----------------------------------------------------- what is refused or nan

@pytest.mark.parametrize("labels", [
    [True, False, True, False],
    [1, 0, 1, 0],
    np.array([1, 0, 1, 0], dtype=np.uint8),
    [1.0, 0.0, 1.0, 0.0],
    [True, 0, 1, False],
    np.array([np.True_, 0, Decimal(1), False], dtype=object),
    np.array([[1], [0], [1], [0]]),
])
def test_labels_are_booleans_or_zeros_and_ones_in_any_form(labels):
    scores = [0.9, 0.2, 0.4, 0.5]
    assert eer(scores, labels) == eer(scores, [True, False, True, False])


@pytest.mark.parametrize("labels, match", [
    ([1, -1, 1, -1], r"^eer\(\): .*got -1 among them\. For -1/\+1 labels, pass "
                     r"np\.asarray\(labels\) == 1"),
    ([1, 2, 0, 0], r"got 2 among them"),
    ([0.9, 0.2, 0.4, 0.5], r"wrong way round: scores come first"),
    (['target', 'nontarget', 'target', 'nontarget'], r"np\.asarray\(labels\) == 'target'"),
    # what a pandas column of text becomes
    (np.array(['target', 'nontarget', 'target', 'nontarget'], dtype=object),
     r"got 'target' among them\. Turn text labels into booleans first, e\.g\. "
     r"np\.asarray\(labels\) == 'target'"),
    ([1, None, 0, 1], r"got None among them"),
    ([1.0, float('nan'), 0.0, 1.0], r"got nan among them\. A trial without a label"),
])
def test_labels_that_are_not_booleans_or_zeros_and_ones_are_refused(labels, match):
    with pytest.raises(ValueError, match=match):
        eer([0.9, 0.2, 0.4, 0.5], labels)


def test_the_hints_in_the_label_messages_work_on_plain_lists():
    """A list compared with 1 or 'target' is just False; the hints go
    through numpy so that they compare element by element."""
    scores, want = [0.9, 0.2, 0.4, 0.5], eer([0.9, 0.2, 0.4, 0.5], [1, 0, 1, 0])
    assert eer(scores, np.asarray([1, -1, 1, -1]) == 1) == want
    text = ['target', 'nontarget', 'target', 'nontarget']
    assert eer(scores, np.asarray(text) == 'target') == want


def test_arguments_the_wrong_way_round_are_refused():
    scores, labels = _gaussian(30, 70, 1.5, seed=11)
    with pytest.raises(ValueError, match="wrong way round"):
        min_dcf(labels, scores, p_target=0.05)


@pytest.mark.parametrize("scores, match", [
    ([0.1, float('nan'), 0.3], r"^det_curve\(\): 1 of the 3 scores is not finite"),
    ([0.1, float('inf'), -float('inf')], r"2 of the 3 scores are not finite"),
    (['0.1', '0.2', '0.3'], r"real numbers, got an array of <U3"),
    ([0.1, None, 0.3], r"numbers, got None among them"),
    ([0.1 + 1j, 0.2, 0.3], r"real numbers, got an array of complex128"),
])
def test_scores_that_are_not_finite_real_numbers_are_refused(scores, match):
    with pytest.raises(ValueError, match=match):
        det_curve(scores, [1, 0, 1])


def test_scores_held_as_python_objects_are_read_as_numbers():
    """Decimals and fractions, mixed with floats, make an object array;
    they are numbers all the same. A Decimal NaN is still not finite."""
    mixed = np.array([Decimal('0.9'), Fraction(1, 5), 0.4, np.float32(0.5)], dtype=object)
    assert eer(mixed, [1, 0, 1, 0]) == eer([0.9, 0.2, 0.4, 0.5], [1, 0, 1, 0])
    with pytest.raises(ValueError, match="1 of the 2 scores is not finite"):
        eer(np.array([Decimal('NaN'), 0.5], dtype=object), [1, 0])


@pytest.mark.parametrize("bad, shown", [
    (Decimal('sNaN'), r"Decimal\('sNaN'\)"),
    (10 ** 400, r"1{1}0{35}\.\.\. among them"),
    (Fraction(10 ** 400, 3), r"Fraction\(10{26}\.\.\. among them"),
], ids=["Decimal sNaN", "10**400", "10**400 / 3"])
def test_numbers_that_float64_cannot_hold_are_refused(bad, shown):
    with pytest.raises(ValueError, match=r"^eer\(\): scores have to be numbers that float64 "
                                         r"can hold, got " + shown):
        eer([0.1, bad, 0.3], [1, 0, 1])
    with pytest.raises(ValueError, match=r"^eer\(\): labels have to be booleans or 0/1"):
        eer([0.1, 0.2, 0.3], [1, bad, 0])

def test_a_column_or_a_row_is_read_as_a_list_and_a_table_is_refused():
    scores, labels = np.array([0.9, 0.2, 0.4, 0.5]), np.array([1, 0, 1, 0])
    want = eer(scores, labels)
    assert eer(scores[:, None], labels[None, :]) == want
    assert eer(scores[None, :], labels[:, None]) == want
    with pytest.raises(ValueError, match=r"^eer\(\): scores have to be one per trial"):
        eer(np.ones((2, 2)), [1, 0, 1, 0])
    with pytest.raises(ValueError, match=r"labels have to be one per trial"):
        eer([1, 2, 3, 4], np.ones((2, 2)))
    with pytest.raises(ValueError, match=r"got shape \(\)"):
        eer(0.5, 1)
    with pytest.raises(ValueError, match=r"^eer\(\): scores have to be one per trial"):
        eer([[0.1, 0.2], [0.3]], [1, 0])


def test_an_empty_column_is_an_empty_list():
    """A column of no trials is read like any other column: as a list, here
    an empty one, which scores nan. An empty table is still a table."""
    with pytest.warns(UserWarning, match=r"^eer\(\): the list is empty"):
        assert math.isnan(eer(np.empty((0, 1)), np.empty((0, 1), bool)))
    with pytest.warns(UserWarning, match=r"^min_dcf\(\): the list is empty"):
        assert math.isnan(min_dcf(np.empty((1, 0)), [], p_target=0.05))
    with pytest.raises(ValueError, match=r"^eer\(\): scores .* got shape \(0, 5\)"):
        eer(np.empty((0, 5)), [])


def test_scores_and_labels_have_to_pair_up():
    with pytest.raises(ValueError, match=r"^min_dcf\(\): 4 scores and 3 labels; they pair up"):
        min_dcf([1, 2, 3, 4], [1, 0, 1], p_target=0.05)


@pytest.mark.parametrize("fn, kwargs", [(eer, {}), (eer, {'method': 'rocch'}),
                                        (min_dcf, {'p_target': 0.05})])
@pytest.mark.parametrize("scores, labels, why", [
    ([], [], "the list is empty"),
    ([0.1, 0.5, 0.9], [0, 0, 0], "the list has no target trials"),
    ([0.1, 0.5, 0.9], [1, 1, 1], "the list has no non-target trials"),
])
def test_a_list_without_both_kinds_of_trial_is_nan_with_one_warning(fn, kwargs, scores,
                                                                    labels, why):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert math.isnan(fn(scores, labels, **kwargs))
    assert len(caught) == 1
    message = str(caught[0].message)
    assert caught[0].category is UserWarning
    assert message.startswith(f"{fn.__name__}(): {why}")
    assert "at least one target and one non-target trial" in message
    assert message.endswith("; scored nan")
    assert caught[0].filename == __file__        # the warning points at the caller


def test_det_curve_of_a_list_without_both_kinds_of_trial_has_nan_rates():
    with pytest.warns(UserWarning, match=r"^det_curve\(\): the list has no target "
                                         r"trials.*p_miss came out nan"):
        curve = det_curve([0.1, 0.5], [0, 0])
    assert np.isnan(curve.p_miss).all() and curve.p_fa.tolist() == [1.0, 0.5, 0.0]
    with pytest.warns(UserWarning, match="both rates came out nan"):
        curve = det_curve([], [])
    assert curve.thresholds.tolist() == [math.inf]
    assert np.isnan(curve.p_miss).all() and np.isnan(curve.p_fa).all()


@pytest.mark.parametrize("kwargs, match", [
    (dict(p_target=0.0), "strictly between 0 and 1"),
    (dict(p_target=1.0), "strictly between 0 and 1"),
    (dict(p_target=5.0), "as a fraction"),
    (dict(p_target=float('nan')), "strictly between 0 and 1"),
    (dict(p_target=True), "has to be a number, got bool"),
    (dict(p_target='0.05'), "has to be a number, got str"),
    (dict(p_target=0.05, c_miss=0.0), "c_miss has to be a positive cost"),
    (dict(p_target=0.05, c_fa=-1.0), "c_fa has to be a positive cost"),
    (dict(p_target=0.05, c_fa=float('inf')), "c_fa has to be a positive cost"),
    (dict(p_target=0.05, c_miss=float('nan')), "c_miss has to be a positive cost"),
])
def test_cost_parameters_outside_their_range_are_refused(kwargs, match):
    with pytest.raises(ValueError, match=r"^min_dcf\(\): .*" + match):
        min_dcf([0.9, 0.2, 0.4, 0.5], [1, 0, 1, 0], **kwargs)


@pytest.mark.parametrize("kwargs", [dict(p_target=0.1, c_miss=5e-324),
                                    dict(p_target=0.5, c_fa=1e-310),
                                    dict(p_target=1e-320)])
def test_weights_below_the_smallest_normal_float64_are_refused(kwargs):
    """c_miss p_target 5e-324 x 0.1 rounds to 0, which the cost would be
    divided by; 1e-310 x 0.5 and 1e-320 are subnormal floats, which carry
    fewer than float64's 53 bits."""
    with pytest.raises(ValueError, match=r"^min_dcf\(\): c_miss p_target .* and c_fa "
                                         r"\(1 - p_target\) .* have to be at least 2\.23e-308"):
        min_dcf([0.9, 0.2, 0.4, 0.5], [1, 0, 1, 0], **kwargs)


def test_scaling_both_costs_by_one_factor_leaves_the_cost_unchanged():
    """What the message above suggests instead."""
    scores, labels = _gaussian(30, 70, 1.0, seed=13)
    for p, c_miss, c_fa in ((0.05, 1.0, 1.0), (0.01, 10.0, 1.0), (0.7, 1.0, 3.0)):
        want = min_dcf(scores, labels, p_target=p, c_miss=c_miss, c_fa=c_fa)
        for factor in (1e-300, 1e300):
            assert min_dcf(scores, labels, p_target=p, c_miss=c_miss * factor,
                           c_fa=c_fa * factor) == pytest.approx(want, rel=1e-14)


def test_p_target_has_no_default():
    with pytest.raises(TypeError, match="p_target"):
        min_dcf([0.9, 0.2], [1, 0])


def test_an_unknown_method_is_refused():
    with pytest.raises(ValueError,
                       match=r"^eer\(\): method has to be one of \('linear', 'rocch'\)"):
        eer([0.9, 0.2], [1, 0], method='bosaris')


def test_integer_scores_are_taken_as_the_numbers_they_are():
    """Not as audio on a full scale: an int16 score of 1000 is a threshold
    of 1000, and the error rates are those of the same list in float."""
    scores = np.array([-1000, 250, 1000, 32000, 250], dtype=np.int16)
    labels = [0, 1, 0, 1, 0]
    curve = det_curve(scores, labels)
    assert curve.thresholds.tolist() == [-1000.0, 250.0, 1000.0, 32000.0, math.inf]
    floats = det_curve(scores.astype(float), labels)
    assert np.array_equal(curve.p_miss, floats.p_miss)
    assert np.array_equal(curve.p_fa, floats.p_fa)
    assert eer(scores, labels) == eer(scores.astype(float), labels)


def test_scores_are_compared_as_float64():
    """Integers beyond 2**53 that float64 cannot tell apart count as tied,
    so every threshold returned accepts exactly the trials its rates count,
    as ``scores >= threshold`` reads them."""
    scores = np.array([2**53, 2**53 + 1, 2**53 + 2, 2**53 + 3], dtype=np.int64)
    labels = np.array([False, True, False, True])
    curve = det_curve(scores, labels)
    for threshold, p_miss, p_fa in zip(*curve):
        accepted = scores >= threshold
        assert p_miss == np.mean(~accepted[labels])
        assert p_fa == np.mean(accepted[~labels])
    assert len(curve.thresholds) == 4          # three distinct float64 values, then inf
    assert eer(scores, labels) == eer(scores.astype(np.float64), labels)


def test_boolean_scores_are_hard_decisions():
    """True scores above False, so the list has one threshold between them:
    here it misses one target in four and accepts one non-target in four."""
    decisions = [True, True, True, False, True, False, False, False]
    labels = [1, 1, 1, 1, 0, 0, 0, 0]
    assert eer(decisions, labels) == pytest.approx(0.25)
    assert det_curve(decisions, labels).thresholds.tolist() == [0.0, 1.0, math.inf]


def test_the_docstring_examples_run():
    """The examples that need kudio's top-level names or data of the reader's
    own are marked to be skipped; the others run as written."""
    import doctest
    failed, attempted = doctest.testmod(module, report=False)
    assert failed == 0 and attempted >= 2


@pytest.mark.parametrize("fn", [eer, det_curve, lambda s, l: min_dcf(s, l, p_target=0.01)],
                         ids=['eer', 'det_curve', 'min_dcf'])
@pytest.mark.parametrize("side", ['scores', 'labels'])
def test_a_masked_array_with_masked_trials_is_refused(fn, side):
    """np.asarray drops the mask: a trial masked out of the scores used to be
    counted at the value under the mask: an EER of 0.5 here, where the
    list without it has 0."""
    scores = [0.9, 0.8, 0.1, 0.95]
    labels = [1, 1, 0, 0]
    mask = [False, False, False, True]
    if side == 'scores':
        scores = np.ma.masked_array(scores, mask=mask)
    else:
        labels = np.ma.masked_array(labels, mask=mask)
    with pytest.raises(ValueError, match=rf"\(\): {side} are a masked array with 1 "
                                         r"masked trial\(s\)"):
        fn(scores, labels)


def test_a_mask_with_nothing_masked_is_read_as_its_values():
    scores, labels = [0.9, 0.8, 0.1, 0.2], [1, 1, 0, 0]
    assert eer(np.ma.masked_array(scores), np.ma.masked_array(labels)) == eer(scores, labels)
