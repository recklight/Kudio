# -*- coding: utf-8 -*-
"""Validation of kudio's detection measures: :func:`kudio.core.verification.eer`,
:func:`~kudio.core.verification.min_dcf` and
:func:`~kudio.core.verification.det_curve`.

Every comparison is run from scratch, on seeded synthetic trial lists, against
references that share no code with kudio:

* scikit-learn's ``roc_curve``, for the operating points, and through scipy's
  ``brentq`` on ``interp1d`` for the EER, which is how the VoxSRC 2020-2023
  scorer (``compute_EER.py``) computes it;
* a brute-force sweep that counts the errors trial by trial at every
  distinct score, every midpoint between two of them, and both ends, with
  the crossing and the costs worked out in exact fractions;
* the ROC convex hull built with Andrew's monotone chain in integers, and
  the same ROCCH-EER a third way, as the largest minimum cost over every
  prior (Brümmer and de Villiers, arXiv:1304.2865, Eq. 26);
* the rules of other scorers, restated here from the code they published:
  Kaldi's ``compute-eer.cc`` and ``compute_min_dcf.py``, voxceleb_trainer's
  ``tuneThresholdfromScore``, SpeechBrain's ``EER`` and ``minDCF`` (in
  float32, as torch's ``.float()`` computes them) and NIST's SRE 2016 scoring
  script (version 4.1).

It needs numpy, scipy and scikit-learn, and runs on the kudio of the checkout
it sits in:

    python tools/validation/verification.py

It prints the largest differences found, then every number the module
docstring quotes, measured again, then how long a million trials take.
"""
from __future__ import annotations

import bisect
import math
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kudio.core import verification  # noqa: E402
from kudio.core.verification import det_curve, eer, min_dcf  # noqa: E402

INF = math.inf


# ---------------------------------------------------------------- trial lists

def small_lists(count: int = 600, seed: int = 0):
    """Lists of 2 to 40 trials: a third with continuous scores, a third with
    a handful of integer levels, a third rounded to one decimal."""
    rng = np.random.default_rng(seed)
    lists = []
    while len(lists) < count:
        n = int(rng.integers(2, 41))
        labels = rng.random(n) < rng.uniform(0.15, 0.85)
        if labels.all() or not labels.any():
            continue
        kind = len(lists) % 3
        if kind == 0:
            scores = rng.normal(size=n) + rng.uniform(0, 3) * labels
        elif kind == 1:
            scores = (rng.integers(0, int(rng.integers(1, 6)), n)
                      + rng.integers(0, 3) * labels).astype(float)
        else:
            scores = np.round(rng.normal(size=n) + labels, 1)
        lists.append((scores, labels))
    return lists


def medium_lists(seed: int = 1):
    """Lists of 1,000 to 10,000 trials, some with ties."""
    rng = np.random.default_rng(seed)
    lists = []
    for targets, nontargets, shift, decimals in [
            (300, 700, 1.5, None), (300, 700, 1.5, 1), (50, 5000, 3.0, None),
            (2000, 2000, 0.5, 2), (1000, 9000, 2.5, None), (1000, 9000, 2.5, 1)]:
        labels = np.r_[np.ones(targets, bool), np.zeros(nontargets, bool)]
        scores = rng.normal(size=labels.size) + shift * labels
        if decimals is not None:
            scores = np.round(scores, decimals)
        order = rng.permutation(labels.size)
        lists.append((scores[order], labels[order]))
    return lists


def gaussian_40k(seed: int):
    """40,000 trials, half of them targets, scores N(0, 1) and N(4.65, 1),
    which puts the EER near 1%."""
    rng = np.random.default_rng(seed)
    labels = np.r_[np.ones(20000, bool), np.zeros(20000, bool)]
    return rng.normal(0, 1, 40000) + 4.65 * labels, labels


def lattice_list(steps: int):
    """A list whose hull has a vertex for every primitive step (a, b),
    a + b <= *steps*: b targets and then a non-targets per step, in rising
    score, the steepest step at the top. A rising threshold first misses the
    b targets and then stops accepting the a non-targets, so the corner
    between lies on the far side of the hull."""
    pairs = [(a, b) for a in range(1, steps) for b in range(1, steps)
             if a + b <= steps and math.gcd(a, b) == 1]
    pairs.sort(key=lambda ab: ab[1] / ab[0], reverse=True)
    labels = []
    for a, b in pairs:                    # from the top score down
        labels += [False] * a + [True] * b
    labels = np.array(labels[::-1])
    return np.arange(labels.size, dtype=float), labels


ELEVEN = (np.arange(11, dtype=float),
          np.array([0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1], bool))


# --------------------------------------------- references: exact operating points
#
# A list of points is ``(points, T, N)``: (misses, false alarms) pairs in
# rising order of threshold, and the numbers of target and non-target trials.

def brute_counts(scores, labels):
    """The errors at every threshold that can matter, each distinct score,
    each midpoint and both ends, counted trial by trial; each distinct pair
    once."""
    values = sorted(set(float(s) for s in scores))
    cuts = ([-INF] + values + [(a + b) / 2 for a, b in zip(values, values[1:])]
            + [INF])
    seen = set()
    for cut in cuts:
        misses = sum(1 for s, y in zip(scores, labels) if y and s < cut)
        false_alarms = sum(1 for s, y in zip(scores, labels) if not y and s >= cut)
        seen.add((misses, false_alarms))
    targets = int(np.sum(labels))
    # a rising threshold adds misses and sheds false alarms
    return sorted(seen, key=lambda p: (p[0], -p[1])), targets, len(labels) - targets


def sorted_counts(scores, labels):
    """The same points for lists too long to sweep trial by trial: each class
    sorted on its own, and the errors at each distinct score counted with
    bisect."""
    targets = sorted(float(s) for s, y in zip(scores, labels) if y)
    nontargets = sorted(float(s) for s, y in zip(scores, labels) if not y)
    cuts = sorted(set(targets) | set(nontargets)) + [INF]
    points = [(bisect.bisect_left(targets, c),
               len(nontargets) - bisect.bisect_left(nontargets, c)) for c in cuts]
    return points, len(targets), len(nontargets)


def crossing_linear(points, t, n):
    """Where straight lines between successive points cross P_miss = P_fa."""
    rates = [(Fraction(m, t), Fraction(f, n)) for m, f in points]
    for (m0, f0), (m1, f1) in zip(rates, rates[1:]):
        d0, d1 = m0 - f0, m1 - f1
        if d0 == 0:
            return m0
        if d0 < 0 <= d1:
            return m0 + (-d0 / (d1 - d0)) * (m1 - m0)
    raise AssertionError("no crossing")


def crossing_hull(points, t, n):
    """Lower convex hull of (P_fa, P_miss) by Andrew's monotone chain, in
    integers in units of 1/(T N), and where it crosses P_miss = P_fa; with
    the number of hull vertices from its lowest point at P_fa = 0 to
    accepting everything."""
    xy = sorted(set((f * t, m * n) for m, f in points))
    hull = []
    for x, y in xy:
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = hull[-2], hull[-1]
            if (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1) <= 0:
                hull.pop()
            else:
                break
        hull.append((x, y))
    # the chain carries on up the right-hand edge past accepting everything,
    # P_fa = 1 and P_miss = 0, which no threshold sweep traces
    hull = hull[:hull.index((t * n, 0)) + 1]
    for (x0, y0), (x1, y1) in zip(hull, hull[1:]):
        d0, d1 = y0 - x0, y1 - x1
        if d0 == 0:
            return Fraction(y0, t * n), len(hull)
        if d0 > 0 >= d1:
            return (y0 + Fraction(d0, d0 - d1) * (y1 - y0)) / (t * n), len(hull)
    raise AssertionError("no crossing")


def crossing_minimax(points, t, n):
    """The largest, over every prior p, of the lowest p P_miss + (1 - p) P_fa:
    the lower envelope is piecewise linear in p, so its maximum sits where
    two operating points cost the same, or at an end."""
    rates = [(Fraction(m, t), Fraction(f, n)) for m, f in points]
    priors = {Fraction(0), Fraction(1)}
    for m0, f0 in rates:
        for m1, f1 in rates:
            slope = (m0 - f0) - (m1 - f1)
            if slope:
                p = (f1 - f0) / slope
                if 0 <= p <= 1:
                    priors.add(p)
    return max(min(p * m + (1 - p) * f for m, f in rates) for p in priors)


def lowest_cost(points, t, n, p_target, c_miss=1.0, c_fa=1.0):
    p, cm, cf = Fraction(p_target), Fraction(c_miss), Fraction(c_fa)
    best = min(cm * p * Fraction(m, t) + cf * (1 - p) * Fraction(f, n) for m, f in points)
    return best / min(cm * p, cf * (1 - p))


# ------------------------------------------- references: other scorers' rules

def roc(scores, labels, **kwargs):
    from sklearn.metrics import roc_curve
    return roc_curve(labels.astype(int), scores, pos_label=1, **kwargs)


def voxsrc_eer(scores, labels):
    """compute_EER.py of the VoxSRC 2020-2023 validation toolkits."""
    from scipy.interpolate import interp1d
    from scipy.optimize import brentq
    fpr, tpr, _ = roc(scores, labels)
    return brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)


def nist_sre16_eer(scores, labels, kind='quicksort'):
    """compute_pmiss_pfa_rbst and compute_eer of NIST's SRE 2016 script:
    every sorted trial an operating point, and the line between the two
    either side of the crossing. The script sorts with argsort's default,
    which leaves tied trials in no set order; ``kind='stable'`` keeps them
    in the order they are listed."""
    lab = labels[np.argsort(scores, kind=kind)].astype(float)
    fnr = np.cumsum(lab) / np.sum(lab)
    fpr = 1 - np.cumsum(1 - lab) / np.sum(1 - lab)
    diff = fnr - fpr
    x1 = np.flatnonzero(diff >= 0)[0]
    x2 = np.flatnonzero(diff < 0)[-1]
    a = (fnr[x1] - fpr[x1]) / (fpr[x2] - fpr[x1] - (fnr[x2] - fnr[x1]))
    return fnr[x1] + a * (fnr[x2] - fnr[x1])


def kaldi_eer(scores, labels):
    """ComputeEer of Kaldi's compute-eer.cc, on scores read as float32."""
    s = np.asarray(scores, dtype=np.float32)
    targets, nontargets = np.sort(s[labels]), np.sort(s[~labels])
    t, n = targets.size, nontargets.size
    position = 0
    while position + 1 < t:
        place = max(n - 1 - int(n * position * 1.0 / t), 0)
        if nontargets[place] < targets[position]:
            break
        position += 1
    return position / t


def voxceleb_trainer_eer(scores, labels):
    """tuneThresholdfromScore of clovaai/voxceleb_trainer: the larger rate at
    the point of roc_curve, with its default drop_intermediate, where the two
    rates are closest."""
    fpr, tpr, _ = roc(scores, labels)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return max(fpr[i], fnr[i])


def _speechbrain_rates(scores, labels):
    """SpeechBrain's thresholds, every distinct score and every midpoint, and
    its rates there in float32: a target at or below the threshold is
    rejected, a non-target above it accepted."""
    s = np.asarray(scores, dtype=np.float64)
    pos, neg = np.sort(s[labels]), np.sort(s[~labels])
    cuts = np.unique(s)
    cuts = np.sort(np.concatenate([cuts, (cuts[:-1] + cuts[1:]) / 2]))
    frr = (np.searchsorted(pos, cuts, side='right').astype(np.float32)
           / np.float32(pos.size))
    far = ((neg.size - np.searchsorted(neg, cuts, side='right')).astype(np.float32)
           / np.float32(neg.size))
    return frr, far


def speechbrain_eer(scores, labels):
    """speechbrain.utils.metric_stats.EER: the first threshold where the two
    rates are closer than at any before it, and their mean. As there, the
    newest gap is taken in float32 and the best one so far from the Python
    floats it kept."""
    frr, far = _speechbrain_rates(scores, labels)
    frr_best, far_best = 0.0, 0.0
    for i in range(frr.size):
        if float(abs(far[i] - frr[i])) < abs(far_best - frr_best) or i == 0:
            frr_best, far_best = float(frr[i]), float(far[i])
    return (far_best + frr_best) / 2


def speechbrain_min_dcf(scores, labels, p_target=0.01, c_miss=1.0, c_fa=1.0):
    """speechbrain.utils.metric_stats.minDCF: C_Det before normalization."""
    frr, far = _speechbrain_rates(scores, labels)
    cost = (np.float32(c_miss) * frr * np.float32(p_target)
            + np.float32(c_fa) * far * np.float32(1 - p_target))
    return float(cost.min())


def kaldi_min_dcf(scores, labels, p_target, c_miss=1.0, c_fa=1.0):
    """ComputeErrorRates and ComputeMinDcf of Kaldi's compute_min_dcf.py, as
    the VoxSRC scorer and voxceleb_trainer carry them: a stable sort, and
    every sorted position an operating point."""
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranked = [int(labels[i]) for i in order]
    misses = np.cumsum(ranked)
    kept = np.cumsum([1 - y for y in ranked])
    fnr = misses / float(sum(ranked))
    fpr = 1 - kept / float(len(ranked) - sum(ranked))
    best = np.min(c_miss * fnr * p_target + c_fa * fpr * (1 - p_target))
    return best / min(c_miss * p_target, c_fa * (1 - p_target))


# --------------------------------------------------------------- comparisons

class Table:
    def __init__(self):
        self.rows = []

    def add(self, what, cases, diffs, note=""):
        diffs = np.abs(np.asarray(diffs, dtype=float))
        self.rows.append((what, cases, float(diffs.max()) if diffs.size else 0.0, note))

    def show(self):
        width = max(len(r[0]) for r in self.rows)
        print(f"{'comparison':<{width}}  {'cases':>5}  {'largest |diff|':>14}  note")
        for what, cases, worst, note in self.rows:
            print(f"{what:<{width}}  {cases:>5}  {worst:>14.3g}  {note}")


def has_mixed_ties(scores, labels):
    """Whether some score is shared by a target and a non-target trial."""
    return bool(set(scores[labels].tolist()) & set(scores[~labels].tolist()))


def against_exact(table: Table, small, medium):
    d_lin, d_hull, d_minimax, d_dcf, d_points = [], [], [], [], []
    for scores, labels in small:
        points, t, n = brute_counts(scores, labels)
        d_lin.append(eer(scores, labels) - float(crossing_linear(points, t, n)))
        hull, _ = crossing_hull(points, t, n)
        d_hull.append(eer(scores, labels, method='rocch') - float(hull))
        d_minimax.append(float(hull - crossing_minimax(points, t, n)))
        for p in (0.5, 0.05, 0.01, 0.001, 0.9):
            d_dcf.append(min_dcf(scores, labels, p_target=p)
                         - float(lowest_cost(points, t, n, p)))
        # c_miss 10 at 0.01 is SRE 2008's; at 0.9, and at 0.01 with c_miss
        # 100, accepting everything is the cheaper decision without scores
        for p, c_miss, c_fa in ((0.01, 10, 1), (0.01, 100, 1), (0.3, 1, 3)):
            d_dcf.append(min_dcf(scores, labels, p_target=p, c_miss=c_miss, c_fa=c_fa)
                         - float(lowest_cost(points, t, n, p, c_miss=c_miss, c_fa=c_fa)))
        curve = det_curve(scores, labels)
        d_points.append(len(points) - len(curve.thresholds))
        d_points.extend(np.array([m / t for m, f in points]) - curve.p_miss)
        d_points.extend(np.array([f / n for m, f in points]) - curve.p_fa)
    d_hull_m = []
    for scores, labels in medium:
        hull, _ = crossing_hull(*sorted_counts(scores, labels))
        d_hull_m.append(eer(scores, labels, method='rocch') - float(hull))
    table.add("eer, default, against the brute-force sweep", len(small), d_lin,
              "exact fractions")
    table.add("eer, rocch, against the monotone-chain hull", len(small), d_hull, "exact")
    table.add("eer, rocch, against the hull (1k-10k trials)", len(medium), d_hull_m, "exact")
    table.add("monotone-chain hull against Eq. 26 (max over priors)", len(small), d_minimax,
              "both exact")
    table.add("min_dcf against the brute-force sweep", len(small), d_dcf,
              "8 cost settings each")
    table.add("det_curve against the brute-force sweep", len(small), d_points,
              "points and rates")


def against_scorers(table: Table, lists):
    from sklearn.metrics import det_curve as sklearn_det_curve
    d_vox, d_nist, d_sk, d_skdcf, d_kaldi, d_skdet = [], [], [], [], [], []
    nist_failed = 0
    ends = [0, 0]                         # sklearn's det_curve with each end
    for scores, labels in lists:
        d_vox.append(eer(scores, labels) - voxsrc_eer(scores, labels))
        if not has_mixed_ties(scores, labels):
            try:
                d_nist.append(eer(scores, labels) - nist_sre16_eer(scores, labels))
            except IndexError:
                # its points start after the lowest trial, so a list whose
                # only target scores lowest has none with P_miss < P_fa
                nist_failed += 1
            if len(set(scores.tolist())) == scores.size:
                d_kaldi.append(min_dcf(scores, labels, p_target=0.05)
                               - kaldi_min_dcf(list(scores), list(labels), 0.05))
        fpr, tpr, thresholds = roc(scores, labels, drop_intermediate=False)
        curve = det_curve(scores, labels)
        d_sk.append(0.0 if np.array_equal(curve.thresholds[::-1], thresholds) else 1.0)
        d_sk.extend(curve.p_fa[::-1] - fpr)
        d_sk.extend(curve.p_miss[::-1] - (1 - tpr))
        for p in (0.05, 0.01):
            d_skdcf.append(min_dcf(scores, labels, p_target=p)
                           - np.min(p * (1 - tpr) + (1 - p) * fpr) / min(p, 1 - p))
        # sklearn's det_curve: a run of the same points, in the same order
        fpr, fnr, thresholds = sklearn_det_curve(labels.astype(int), scores)
        start = int(np.searchsorted(curve.thresholds, thresholds[0]))
        run = slice(start, start + thresholds.size)
        if not np.array_equal(curve.thresholds[run], thresholds):
            d_skdet.append(1.0)
            continue
        d_skdet.extend(curve.p_fa[run] - fpr)
        d_skdet.extend(curve.p_miss[run] - fnr)
        ends[0] += start == 0
        ends[1] += start + thresholds.size == curve.thresholds.size
    table.add("eer, default, against VoxSRC's brentq on roc_curve", len(lists), d_vox,
              "brentq xtol 2e-12")
    table.add("eer, default, against NIST's SRE 2016 script", len(d_nist), d_nist,
              f"lists without mixed ties; the script fails on {nist_failed} more")
    table.add("det_curve against roc_curve(drop_intermediate=False)", len(lists), d_sk,
              "thresholds, P_fa, P_miss")
    table.add("min_dcf against the lowest cost on roc_curve", len(lists), d_skdcf,
              "p_target 0.05 and 0.01")
    table.add("min_dcf against Kaldi's compute_min_dcf.py rule", len(d_kaldi), d_kaldi,
              "lists without ties")
    table.add("det_curve against sklearn's det_curve", len(lists), d_skdet,
              f"a run of the same points; accept-all kept on {ends[0]}, "
              f"reject-all on {ends[1]}")


def descriptions(table: Table, lists):
    """The module docstring describes two of the other rules in a sentence
    each; the restated code is held to those sentences."""
    d_kaldi, d_sb, kaldi_cases = [], [], 0
    for scores, labels in lists:
        curve = det_curve(scores, labels)
        if len(set(scores[labels].tolist())) == int(labels.sum()):
            kaldi_cases += 1
            # the miss rate at the first target score, counting up, where
            # P_fa <= P_miss; the loop never looks at the highest one
            said = (labels.sum() - 1) / labels.sum()
            for score in np.sort(scores[labels])[:-1]:
                k = int(np.searchsorted(curve.thresholds, score))
                if curve.p_fa[k] <= curve.p_miss[k]:
                    said = curve.p_miss[k]
                    break
            d_kaldi.append(kaldi_eer(scores, labels) - said)
        # SpeechBrain has no accept-all point; where two points are equally
        # close, its float32 rounding picks one, so either counts
        gap = np.abs(curve.p_miss - curve.p_fa)[1:]
        closest = 1 + np.flatnonzero(np.isclose(gap, gap.min(), rtol=1e-12, atol=0))
        means = (curve.p_miss[closest] + curve.p_fa[closest]) / 2
        d_sb.append(np.min(np.abs(speechbrain_eer(scores, labels) - means)))
    table.add("Kaldi's compute-eer against its description", kaldi_cases, d_kaldi,
              "lists without tied target scores")
    table.add("SpeechBrain's EER against its description", len(lists), d_sb,
              "its rates are float32")


# ------------------------------------------------- numbers the module quotes

def quoted():
    print("\nNumbers quoted in kudio.core.verification, measured again:")
    scores, labels = ELEVEN
    print("  eleven trials: default %.4f, rocch %.4f, NIST SRE 2016 %.4f, VoxSRC %.4f,"
          % (eer(scores, labels), eer(scores, labels, method='rocch'),
             nist_sre16_eer(scores, labels), voxsrc_eer(scores, labels)))
    print("                 Kaldi %.4f, SpeechBrain %.4f, voxceleb_trainer %.4f"
          % (kaldi_eer(scores, labels), speechbrain_eer(scores, labels),
             voxceleb_trainer_eer(scores, labels)))
    fpr, tpr, _ = roc(scores, labels, drop_intermediate=False)
    closest = int(np.argmin(np.abs((1 - tpr) - fpr)))
    kept_fpr, kept_tpr, _ = roc(scores, labels)
    kept = any(f == fpr[closest] and t == tpr[closest] for f, t in zip(kept_fpr, kept_tpr))
    print("                 the point where the rates are closest, P_fa %.4f and P_miss %.4f, "
          "is %s by roc_curve's default"
          % (fpr[closest], 1 - tpr[closest], "kept" if kept else "dropped"))

    worst = 0.0
    for seed in range(6):
        s, y = gaussian_40k(seed)
        default = eer(s, y)
        if seed == 0:
            print("  40,000 trials: default %.3f%%, rocch %.3f%%"
                  % (100 * default, 100 * eer(s, y, method='rocch')))
            print("                 c_miss 10 at 0.01: %.3f, unit costs at 0.01: %.3f, "
                  "unit costs at the effective prior 0.1/1.09: %.3f"
                  % (min_dcf(s, y, p_target=0.01, c_miss=10), min_dcf(s, y, p_target=0.01),
                     min_dcf(s, y, p_target=0.1 / 1.09)))
        for rule in (kaldi_eer, speechbrain_eer, voxceleb_trainer_eer):
            worst = max(worst, abs(rule(s, y) - default))
    print("                 Kaldi, SpeechBrain, voxceleb_trainer: within %.4f points of the "
          "default on six such lists" % (100 * worst))

    rng = np.random.default_rng(0)
    y = np.r_[np.ones(2000, bool), np.zeros(20000, bool)]
    s = np.round(rng.normal(0, 1, y.size) + 3.0 * y, 1)
    first = np.r_[np.arange(2000, 22000), np.arange(2000)]
    print("  22,000 trials rounded to 0.1, min DCF at p_target 0.05: kudio %.5f; Kaldi's rule, "
          "non-targets first %.5f, targets first %.5f"
          % (min_dcf(s, y, p_target=0.05), kaldi_min_dcf(list(s[first]), list(y[first]), 0.05),
             kaldi_min_dcf(list(s), list(y), 0.05)))
    shuffled = []
    for seed in range(5):
        order = np.random.default_rng(seed).permutation(y.size)
        shuffled.append(nist_sre16_eer(s[order], y[order]))
    print("                 EER: kudio %.5f; NIST's script with the ties non-targets first "
          "%.5f, targets first %.5f, with its own argsort on five shuffles %.5f to %.5f"
          % (eer(s, y), nist_sre16_eer(s[first], y[first], kind='stable'),
             nist_sre16_eer(s, y, kind='stable'), min(shuffled), max(shuffled)))

    y = np.r_[np.ones(100, bool), np.zeros(10000, bool)]
    s = np.r_[np.linspace(1, 2, 100), np.linspace(-2, 0, 9999), [3.0]]
    print("  one false alarm among 10,000 non-targets at p_target 0.001: %.4f"
          % min_dcf(s, y, p_target=0.001))

    s, y = gaussian_40k(0)
    printed = 100 * speechbrain_min_dcf(s, y, p_target=0.05)
    print("  SpeechBrain's minDCF times 100 at p_target 0.05, over the normalized cost: %.4f"
          % (printed / min_dcf(s, y, p_target=0.05)))
    printed = 100 * speechbrain_min_dcf(s, y, p_target=0.01)
    print("  the same at its default 0.01: %.4f" % (printed / min_dcf(s, y, p_target=0.01)))

    s, y = lattice_list(170)
    counted = verification._counted(s, y, 'eer')
    gap = counted.misses * counted.nontargets - counted.false_alarms * counted.targets
    crossing, chords = verification._hull_crossing(counted, gap)
    hull, vertices = crossing_hull(*sorted_counts(s, y))
    print("  hull search: %d trials, %d hull vertices, %d chords, crossing %s the "
          "monotone chain's" % (y.size, vertices, chords,
                                "equal to" if crossing == hull else "NOT equal to"))


def timing():
    rng = np.random.default_rng(1)
    n = 1_000_000
    labels = rng.random(n) < 0.5
    scores = rng.normal(0, 1, n) + 2.0 * labels

    def best_of(fn, *args, **kwargs):
        times = []
        for _ in range(3):
            start = time.perf_counter()
            fn(*args, **kwargs)
            times.append(time.perf_counter() - start)
        return min(times)

    print("\nA million trials, best of three:")
    print("  eer                %.3f s" % best_of(eer, scores, labels))
    print("  eer, rocch         %.3f s" % best_of(eer, scores, labels, method='rocch'))
    print("  min_dcf            %.3f s" % best_of(min_dcf, scores, labels, p_target=0.05))
    print("  det_curve          %.3f s" % best_of(det_curve, scores, labels))
    print("  sklearn roc_curve  %.3f s" % best_of(roc, scores, labels))


def main():
    import scipy
    import sklearn
    print(f"numpy {np.__version__}, scipy {scipy.__version__}, "
          f"scikit-learn {sklearn.__version__}, Python {sys.version.split()[0]}\n")
    small, medium = small_lists(), medium_lists()
    table = Table()
    against_exact(table, small, medium)
    against_scorers(table, small + medium + [ELEVEN])
    descriptions(table, small + medium + [ELEVEN])
    table.show()
    quoted()
    timing()


if __name__ == '__main__':
    main()
