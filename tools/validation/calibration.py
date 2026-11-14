# -*- coding: utf-8 -*-
"""Check kudio's calibration measures (kudio/core/verification.py): cllr,
min_cllr, act_dcf and c_primary, against PYLLR's rules, scikit-learn's
isotonic regression and the definitions worked out in exact arithmetic.

Run it from anywhere; it imports the kudio of the checkout it sits in::

    python tools/validation/calibration.py

PYLLR (github.com/bsxfan/PYLLR, MIT, commit 442b10b of 2023-02-21) is Niko
Brümmer's Python port of the BOSARIS algorithms, and it is not on PyPI. Its
PAV, min_cllr and fast_Bayes_error_rate are restated here from that code,
each a few lines: the PAV through scikit-learn's own isotonic regression
routine, as PYLLR calls it. scikit-learn is needed for those rows and for
the isotonic regression; mpmath, if it is installed, for the 40-digit C_llr.

Every list is made here from fixed seeds:

* calibrated LLRs: targets drawn from N(m, 2m) and non-targets from
  N(-m, 2m), for which the score is the LLR, with m from 0.5 to 4;
* the same scores shifted and scaled, so that they are no longer
  calibrated;
* lists rounded to whole numbers or a tenth, with ties between targets and
  non-targets;
* lists with an LLR of +inf or -inf in them.

What it checks:

1. cllr against the sum in 40-digit arithmetic, with mpmath.
2. min_cllr against PYLLR's min_cllr(PAV(scores, labels)), whose PAV ranks
   the targets of a tie below its non-targets and so keeps ties together;
   against scikit-learn's IsotonicRegression, which averages tied scores
   before the fit; and against BOSARIS's min_cllr.m restated, which sorts
   the non-targets of a tie first, splits ties, and adds 1e-6 i / N to the
   i-th LLR: equal on lists without ties to that 1e-6, lower with them.
3. min_cllr is never above cllr nor above 1, and is the same after any
   rising transform of the scores and for any order of tied trials.
4. act_dcf against PYLLR's fast_Bayes_error_rate over default_error_rate at
   the prior log odds -log(beta), and against a count in exact fractions,
   at p_target 0.001, 0.005, 0.01, 0.05 and 0.5 and with c_miss 10.
5. c_primary against the mean of act_dcf, and of min_dcf, at 0.01 and 0.005.

It prints the largest differences and exits with 1 if one is past its
tolerance. The run takes a few seconds.
"""
from __future__ import annotations

import math
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import kudio  # noqa: E402

try:
    from sklearn.isotonic import IsotonicRegression
except ImportError:
    IsotonicRegression = None

try:
    import mpmath
except ImportError:
    mpmath = None

RESULTS = []
PRIORS = (0.001, 0.005, 0.01, 0.05, 0.5)


def report(check, cases, diff, tolerance):
    RESULTS.append((check, cases, diff, tolerance))


def skipped(what, why):
    report(f'{what}: skipped, {why}', 0, 0.0, math.inf)


def lists(seed=0):
    """``(kind, llrs, labels)`` for every list."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(240):
        n = int(rng.integers(4, 400))
        labels = rng.random(n) < rng.uniform(0.1, 0.9)
        if labels.all() or not labels.any():
            continue
        m = rng.uniform(0.5, 4.0)
        llrs = np.where(labels, m, -m) + math.sqrt(2 * m) * rng.standard_normal(n)
        kind = i % 4
        if kind == 1:
            llrs = rng.uniform(0.3, 3.0) * llrs + rng.uniform(-3.0, 3.0)
            name = 'uncalibrated'
        elif kind == 2:
            llrs = np.round(llrs, int(rng.integers(0, 2)))
            name = 'rounded'
        elif kind == 3:
            llrs[rng.integers(0, n)] = math.inf
            llrs[rng.integers(0, n)] = -math.inf
            name = 'with infinite LLRs'
        else:
            name = 'calibrated'
        out.append((name, llrs, labels))
    return out


def has_ties(llrs, labels):
    common = set(llrs[labels].tolist()) & set(llrs[~labels].tolist())
    return bool(common)


# --- the oracles ------------------------------------------------------------------

def exact_cllr(llrs, labels):
    """C_llr in 40-digit arithmetic, or inf."""
    mpmath.mp.dps = 40

    def cost(x):
        if math.isinf(x):
            return mpmath.mpf(0) if x < 0 else mpmath.inf
        return mpmath.log(1 + mpmath.exp(mpmath.mpf(x)))
    tar = mpmath.fsum(cost(-x) for x in llrs[labels]) / int(labels.sum())
    non = mpmath.fsum(cost(x) for x in llrs[~labels]) / int((~labels).sum())
    return float((tar + non) / (2 * mpmath.log(2)))


def isotonic_min_cllr(llrs, labels):
    """min C_llr from scikit-learn's isotonic regression of the labels on
    the scores, ties averaged first, as its fit does."""
    finite = np.where(np.isinf(llrs), np.sign(llrs) * 1e300, llrs)
    p = IsotonicRegression(y_min=0.0, y_max=1.0).fit_transform(finite, labels.astype(float))
    T, N = int(labels.sum()), int((~labels).sum())
    with np.errstate(divide='ignore'):
        llr = np.log(p) - np.log1p(-p) - math.log(T / N)
    tar = np.logaddexp(0.0, -llr[labels])
    non = np.logaddexp(0.0, llr[~labels])
    return float((tar.mean() + non.mean()) / (2 * math.log(2)))


def bosaris_min_cllr(llrs, labels):
    """BOSARIS's min_cllr.m restated: opt_loglr with the non-targets of a
    tie sorted first (MATLAB's sort is stable and [non; tar] goes in), the
    PAV fit, LLRs less the prior log odds plus 1e-6 i / N, and cllr.m."""
    non, tar = llrs[~labels], llrs[labels]
    scores = np.concatenate([non, tar])
    ideal = np.concatenate([np.zeros(non.size), np.ones(tar.size)])
    order = np.argsort(scores, kind='stable')
    fitted = pav(ideal[order])
    N, T = non.size, tar.size
    with np.errstate(divide='ignore'):
        out = np.log(fitted) - np.log1p(-fitted) - math.log(T / N)
    out = out + np.arange(1, N + T + 1) * 1e-6 / (N + T)
    back = np.empty(N + T)
    back[order] = out
    t, n = back[N:], back[:N]
    return float((np.mean(np.logaddexp(0.0, -t)) + np.mean(np.logaddexp(0.0, n)))
                 / (2 * math.log(2)))


def pav(y):
    """The plain pool-adjacent-violators fit of a sequence, unit weights."""
    values, weights, counts = [], [], []
    for v in y:
        values.append(float(v))
        weights.append(1.0)
        counts.append(1)
        while len(values) > 1 and values[-2] > values[-1]:
            w = weights[-2] + weights[-1]
            values[-2] = (values[-2] * weights[-2] + values[-1] * weights[-1]) / w
            weights[-2] = w
            counts[-2] += counts[-1]
            values.pop()
            weights.pop()
            counts.pop()
    return np.repeat(values, counts)


def pyllr_min_cllr(llrs, labels):
    """PYLLR's PAV(scores, labels) and min_cllr(pav), restated: the trials
    sorted with the targets of a tie first (np.lexsort((-labels, scores))),
    scikit-learn's in-place isotonic regression on the labels, the blocks
    read back with np.unique, and the cost of each block's LLR."""
    from sklearn._isotonic import _inplace_contiguous_isotonic_regression
    labels = labels.astype(int)
    T = labels.sum()
    N = labels.size - T
    order = np.lexsort((-labels, llrs))
    y = labels[order].astype(np.float64)
    _inplace_contiguous_isotonic_regression(y, np.ones_like(y))
    p, first, counts = np.unique(y, return_index=True, return_counts=True)
    targets = np.rint(counts * p).astype(int)
    nontars = counts - targets
    with np.errstate(divide='ignore'):
        llr = np.log(p) - np.log1p(-p) - np.log(T / N)
    tar_costs = np.logaddexp(0.0, -llr)
    non_costs = np.logaddexp(0.0, llr)
    t, n = targets > 0, nontars > 0
    return float(((tar_costs[t] @ targets[t]) / T + (non_costs[n] @ nontars[n]) / N)
                 / (2 * math.log(2)))


def pyllr_bayes_error_rate(llrs, labels, plo):
    """PYLLR's fast_Bayes_error_rate over default_error_rate, restated for
    one prior log odds: ordinal ranks of the threshold among the scores,
    the threshold ranked before any score equal to it."""
    from scipy.stats import rankdata
    threshold = -plo
    tar, non = llrs[labels], llrs[~labels]
    T, N = tar.size, non.size
    rank = rankdata(np.concatenate(([threshold], tar)), method='ordinal')[0]
    p_miss = (rank - 1) / T
    rank = rankdata(np.concatenate(([threshold], non)), method='ordinal')[0]
    p_fa = (N - rank + 1) / N
    p_tar = 1.0 / (1.0 + math.exp(-plo))
    p_non = 1.0 / (1.0 + math.exp(plo))
    return (p_tar * p_miss + p_non * p_fa) / min(p_tar, p_non)


def exact_act_dcf(llrs, labels, p_target, c_miss=1, c_fa=1):
    p = Fraction(p_target)
    w_miss, w_fa = c_miss * p, c_fa * (1 - p)
    threshold = math.log(float(w_fa / w_miss))
    T, N = int(labels.sum()), int((~labels).sum())
    misses = sum(1 for x in llrs[labels] if x < threshold)
    alarms = sum(1 for x in llrs[~labels] if x >= threshold)
    cost = w_miss * Fraction(misses, T) + w_fa * Fraction(alarms, N)
    return cost / min(w_miss, w_fa)


# --- the checks -------------------------------------------------------------------

def check_cllr():
    if mpmath is None:
        skipped('cllr vs 40 digits', 'no mpmath')
        return
    worst = 0.0
    for _, x, y in lists():
        got, want = kudio.cllr(x, y), exact_cllr(x, y)
        worst = max(worst, 0.0 if got == want else abs(got - want))
    report('cllr vs 40 digits (bits), infinite LLRs included', 240, worst, 1e-13)


def check_min_cllr():
    data = lists()
    if IsotonicRegression is None:
        skipped("min_cllr vs PYLLR's rule and scikit-learn", 'no scikit-learn')
    else:
        diff = max(abs(kudio.min_cllr(x, y) - pyllr_min_cllr(x, y)) for _, x, y in data)
        report("min_cllr vs PYLLR's rule (bits), ties and inf included", len(data), diff, 1e-12)
        diff = max(abs(kudio.min_cllr(x, y) - isotonic_min_cllr(x, y)) for _, x, y in data)
        report('min_cllr vs scikit-learn isotonic regression (bits)', len(data), diff, 1e-12)
    untied = [(x, y) for _, x, y in data if not has_ties(x, y)]
    tied = [(x, y) for _, x, y in data if has_ties(x, y)]
    diff = max(abs(kudio.min_cllr(x, y) - bosaris_min_cllr(x, y)) for x, y in untied)
    report("min_cllr vs BOSARIS's rule, lists without ties (bits)", len(untied), diff, 1e-5)
    lower = [kudio.min_cllr(x, y) - bosaris_min_cllr(x, y) for x, y in tied]
    report(f"  with ties BOSARIS reads lower in {sum(d > 1e-5 for d in lower)} of "
           f"{len(tied)}, by up to (bits)", len(tied), max(lower), math.inf)


def check_properties():
    rng = np.random.default_rng(5)
    above = moved = reordered = 0.0
    data = lists()
    for _, x, y in data:
        m, c = kudio.min_cllr(x, y), kudio.cllr(x, y)
        above = max(above, m - min(c, 1.0))
        finite = np.isfinite(x)
        if finite.all():
            moved = max(moved, abs(kudio.min_cllr(np.exp(x / 4) * 3 + 1, y) - m))
        order = rng.permutation(x.size)
        reordered = max(reordered, abs(kudio.min_cllr(x[order], y[order]) - m))
    report('min_cllr above cllr or above 1, by (bits)', len(data), max(above, 0.0), 1e-12)
    report('min_cllr after a rising transform of the scores (bits)', len(data), moved, 1e-12)
    report('min_cllr of the list in another order (bits)', len(data), reordered, 1e-12)


def check_act_dcf():
    data = lists()
    worst = 0.0
    for _, x, y in data:
        for p in PRIORS:
            for c_miss in (1, 10):
                got = kudio.act_dcf(x, y, p_target=p, c_miss=c_miss)
                worst = max(worst, abs(got - float(exact_act_dcf(x, y, p, c_miss))))
    report('act_dcf vs a count in exact fractions', len(data) * 10, worst, 1e-12)
    worst = 0.0
    for _, x, y in data:
        for p in PRIORS:
            for c_miss in (1, 10):
                plo = math.log(c_miss * p / (1 - p))
                theirs = pyllr_bayes_error_rate(x, y, plo)
                worst = max(worst, abs(kudio.act_dcf(x, y, p_target=p, c_miss=c_miss) - theirs))
    report("act_dcf vs PYLLR's Bayes error rate over the default", len(data) * 10, worst, 1e-12)


def check_c_primary():
    worst = 0.0
    data = lists()
    for _, x, y in data:
        want = (kudio.act_dcf(x, y, p_target=0.01) + kudio.act_dcf(x, y, p_target=0.005)) / 2
        worst = max(worst, abs(kudio.c_primary(x, y) - want))
        finite = np.isfinite(x)
        if finite.all():
            want = (kudio.min_dcf(x, y, p_target=0.01) + kudio.min_dcf(x, y, p_target=0.005)) / 2
            worst = max(worst, abs(kudio.c_primary(x, y, minimum=True) - want))
    report('c_primary vs the mean at 0.01 and 0.005', len(data), worst, 0.0)


def main():
    start = time.perf_counter()
    import sklearn
    print(f"kudio {kudio.__version__} from {Path(kudio.__file__).parent}; numpy "
          f"{np.__version__}, scikit-learn {sklearn.__version__}, mpmath "
          f"{getattr(mpmath, '__version__', 'none')}, Python {sys.version.split()[0]}")
    check_cllr()
    check_min_cllr()
    check_properties()
    check_act_dcf()
    check_c_primary()
    width = max(len(r[0]) for r in RESULTS)
    print(f"\n{'check':<{width}} {'cases':>5} {'max |diff|':>11} {'limit':>8}")
    failed = False
    for check, cases, diff, tolerance in RESULTS:
        over = diff > tolerance
        failed |= over
        limit = '-' if math.isinf(tolerance) else f"{tolerance:.0e}"
        print(f"{check:<{width}} {cases:>5} {diff:>11.2e} {limit:>8}{'  PAST' if over else ''}")
    print(f"\n{time.perf_counter() - start:.0f} s; "
          + ("SOME DIFFERENCES ARE PAST TOLERANCE" if failed
             else "every difference within its tolerance"))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
