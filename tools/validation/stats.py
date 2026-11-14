# -*- coding: utf-8 -*-
"""Check kudio's confidence intervals and paired tests (kudio/core/stats.py)
against scipy.stats and against their definitions restated, then measure by
simulation how often the intervals hold the true value and how often the
test rejects a null that is true.

Run it from anywhere; it imports the kudio of the checkout it sits in::

    python tools/validation/stats.py
    python tools/validation/stats.py --repetitions 500      # coarser, faster

Besides kudio it needs numpy and scipy. I ran it with numpy 2.0.2 and
scipy 1.13.1 on Python 3.9.13, with numpy 2.3.5 and scipy 1.16.3 on Python
3.13.9, which printed the same figures, and with numpy 2.5.4 and scipy
1.18.1 on Python 3.13.9, which differs only where scipy 1.18 reads its own
interval ends another way (checks 1 and 2).

What it checks:

1. bootstrap_ci against scipy.stats.bootstrap on the mean, percentile and
   BCa, at 5 to 1000 items of normal and lognormal data and five seeds.
   kudio draws its resamples with the call scipy makes, so one seed gives
   both the same resamples, and kudio's percentile ends equal np.percentile
   over scipy's bootstrap_distribution exactly. scipy up to 1.16 reads its
   own ends that way; 1.18 reads them with scipy.stats.quantile, which lands
   within about 1e-15 of the width. The BCa ends agree to rounding, since
   kudio computes the jackknife another way. Where
   some resamples hold each item once (at 5 and 8 items), kudio counts
   them as equal to the estimate and scipy does not; there kudio is checked
   against the loop of check 3 instead, which sums with math.fsum and so
   makes those resamples equal to the estimate to the last bit.
2. paired_ci against scipy.stats.bootstrap with paired=True, which
   resamples the pairs with the same call.
3. rate_ci, paired_rate_ci and grouped intervals against a slow
   restatement: the same resamples drawn in one call, then a loop over them
   that adds up the errors and totals item by item, and BCa's acceleration
   from the jackknife leaving out one group at a time, recomputed from the
   items. Also that groups of one item each give the ungrouped result to
   the last bit.
4. permutation_test against scipy.stats.permutation_test with
   permutation_type='samples', every sign pattern enumerated: 2 to 16 items,
   with and without ties, and the three alternatives; and with groups,
   against scipy on the groups' sums. With patterns drawn at random, 20
   items against the exact p-value of all 2**20 patterns, and 50 and 200
   items against scipy's own random p-value, each within four standard
   errors. Signed tenths, whose patterns tie the observed sum in exact
   arithmetic but not in floats, against an enumeration in exact fractions;
   scipy agrees with both except where the observed sum is 0, where its
   tolerance, relative to that sum, is 0 too.
5. holm against the step-down procedure run at every level at which it
   can change its decisions.
6. How far another seed moves the ends of an interval from 10000
   resamples, over 60 seeds, on 50 and 824 values of normal and lognormal
   data.
7. By simulation, with --repetitions data sets each (2000 by default):

   - how often a 95% interval holds the true mean of i.i.d. normal and
     lognormal data, n = 20, 50 and 200, percentile and BCa, and of normal
     data at n = 2, 5 and 10;
   - the same on clustered data, speakers of 10 to 30 utterances with an
     intra-class correlation of 0.3, resampled by utterance and by speaker,
     at 10 and 30 speakers, and by speaker at 2 and 5;
   - the same for a WER pooled over 30 speakers of 3 to 12 utterances, each
     speaker with an error probability of its own drawn from Beta(2, 18),
     3 + Poisson(12) words an utterance and Poisson(0.3) insertions, so
     that the true rate is 0.1 + 0.3 / 15 = 0.12;
   - how often the sign-flip test rejects at 0.05 when the two systems do
     not differ: on i.i.d. differences, and on clustered differences flipping
     utterances or whole speakers. The i.i.d. cell runs five times as many
     data sets as the others, since it is the one compared with 0.05 itself.

   Beside every share the table prints its standard error.

   The figures the module docstring and the warnings quote are checked
   against these, within three standard errors of the simulation and the
   rounding of the figure.

It prints every comparison and the coverage table, and exits with 1 if a
difference is past its tolerance. The default run takes about 85 seconds on
a desktop CPU, most of it in the simulation; --repetitions 500 takes about
30, and its shares are coarse enough that a figure checked at 2000 can miss
its tolerance.

References: B. Efron, "Better bootstrap confidence intervals", J. Amer.
Statist. Assoc. 82(397), pp. 171-185, 1987, doi:10.1080/01621459.1987.10478410;
B. Phipson and G. K. Smyth, "Permutation p-values should never be zero",
Stat. Appl. Genet. Mol. Biol. 9(1), 2010, doi:10.2202/1544-6115.1585; the
rest are in the module docstring of kudio/core/stats.py.
"""
from __future__ import annotations

import argparse
import itertools
import math
import sys
import time
import warnings
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import kudio  # noqa: E402
from kudio.core import stats as module  # noqa: E402

#: (check, cases, largest difference, tolerance)
RESULTS = []


def report(check, cases, diff, tolerance):
    RESULTS.append((check, cases, float(diff), tolerance))


def quietly(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return fn(*args, **kwargs)


def mean_of_difference(a, b, axis):
    return np.mean(a - b, axis=axis)


# --- 1 and 2: the bootstrap against scipy --------------------------------------

def permutations_drawn(n, n_resamples, seed):
    """How many of the resamples scipy and kudio draw with *seed* hold each
    of the n items once: their mean is the estimate exactly."""
    pick = np.sort(np.random.default_rng(seed).integers(0, n, (n_resamples, n)), axis=1)
    return int(np.count_nonzero(np.all(pick == np.arange(n), axis=1)))


def check_bootstrap():
    worst = {'percentile': 0.0, 'bca': 0.0}
    worst_read = 0.0
    tied, cases = [], 0
    for n in (5, 8, 12, 50, 200, 1000):
        for seed in range(5):
            for kind in ('normal', 'lognormal'):
                rng = np.random.default_rng([n, seed])
                x = rng.normal(size=n) if kind == 'normal' else rng.lognormal(size=n)
                ties = permutations_drawn(n, 4000, seed)
                for method in worst:
                    ours = quietly(module.bootstrap_ci, x, method=method, seed=seed,
                                   n_resamples=4000)
                    theirs = quietly(scipy_stats.bootstrap, (x,), np.mean, method=method,
                                     n_resamples=4000,
                                     random_state=np.random.default_rng(seed))
                    ci = theirs.confidence_interval
                    width = ci.high - ci.low
                    apart = max(abs(ours.low - ci.low), abs(ours.high - ci.high)) / width
                    if method == 'percentile':
                        alpha = (1 - 0.95) / 2
                        read = theirs.bootstrap_distribution
                        read_apart = max(abs(ours.low - np.percentile(read, alpha * 100)),
                                         abs(ours.high - np.percentile(read, (1 - alpha) * 100)))
                        worst_read = max(worst_read, read_apart / width)
                    if ties and method == 'bca':
                        low, high = restated(x.tolist(), [1.0] * n, list(range(n)), 0.95,
                                             4000, method, seed)
                        exact = max(abs(ours.low - low), abs(ours.high - high)) / width
                        tied.append((apart, n, ties, exact))
                    else:
                        worst[method] = max(worst[method], apart)
                        cases += 1
    report('percentile ends vs np.percentile over scipy\'s resamples (of the width)', 60,
           worst_read, 0.0)
    report('percentile ends vs scipy.stats.bootstrap\'s own ends (of the width)', 60,
           worst['percentile'], 1e-14)
    report('BCa ends vs scipy.stats.bootstrap (of the width)', cases - 60,
           worst['bca'], 1e-12)
    tied.sort(reverse=True)
    print(f"\nBCa where some of the 4000 resamples hold each item once ({len(tied)} cases, "
          f"{min(t[1] for t in tied)} to {max(t[1] for t in tied)} items): kudio counts "
          f"them as ties with the estimate, scipy as above or below it as rounding "
          f"falls. Largest difference {tied[0][0]:.2e} of the width, at {tied[0][1]} "
          f"items with {tied[0][2]} such resamples")
    report('the same where resamples tie the estimate: differs on purpose', len(tied),
           tied[0][0], math.inf)
    report('those cases vs the loop over resamples summed with fsum (of the width)',
           len(tied), max(t[3] for t in tied), 1e-12)

    worst, cases = 0.0, 0
    for n in (12, 40, 300):
        for seed in range(5):
            rng = np.random.default_rng([n, seed, 1])
            system = rng.normal(size=n)
            baseline = system - rng.lognormal(size=n) + 1.0
            if permutations_drawn(n, 4000, seed):
                continue
            for method in ('percentile', 'bca'):
                ours = quietly(module.paired_ci, system, baseline, method=method,
                               seed=seed, n_resamples=4000)
                theirs = quietly(scipy_stats.bootstrap, (system, baseline),
                                 mean_of_difference, paired=True, method=method,
                                 n_resamples=4000,
                                 random_state=np.random.default_rng(seed))
                ci = theirs.confidence_interval
                width = ci.high - ci.low
                worst = max(worst, abs(ours.low - ci.low) / width,
                            abs(ours.high - ci.high) / width)
                cases += 1
    report('paired_ci vs scipy.stats.bootstrap(paired=True) (of the width)', cases, worst,
           1e-12)


# --- 3: grouped intervals and rates, restated ----------------------------------

def restated(numerator, denominator, groups, confidence, n_resamples, method, seed):
    """The grouped ratio interval the slow way: the resamples drawn in one
    call, every resample added up item by item, and the jackknife recomputed
    from the items without each group."""
    from scipy.special import ndtr, ndtri

    labels = list(dict.fromkeys(groups))
    members = {label: [i for i, g in enumerate(groups) if g == label] for label in labels}
    units = len(labels)
    pick = np.random.default_rng(seed).integers(0, units, (n_resamples, units))
    theta_b = np.empty(n_resamples)
    for b in range(n_resamples):
        top, bottom = [], []
        for k in pick[b]:
            for i in members[labels[k]]:
                top.append(numerator[i])
                bottom.append(denominator[i])
        # fsum rounds once, so a resample holding each group once gives the
        # estimate to the last bit
        theta_b[b] = math.fsum(top) / math.fsum(bottom)
    theta = math.fsum(numerator) / math.fsum(denominator)
    alpha = (1 - confidence) / 2
    if method == 'percentile':
        levels = (alpha, 1 - alpha)
    else:
        left_out = []
        for label in labels:
            keep = [i for i, g in enumerate(groups) if g != label]
            left_out.append(math.fsum(numerator[i] for i in keep)
                            / math.fsum(denominator[i] for i in keep))
        left_out = np.array(left_out)
        u = (units - 1) * (left_out.mean() - left_out)      # Efron and Tibshirani 14.15
        a = np.sum(u ** 3) / (6.0 * np.sum(u ** 2) ** 1.5)
        share = (np.sum(theta_b < theta) + np.sum(theta_b <= theta)) / (2 * n_resamples)
        z0 = ndtri(share)
        z = ndtri(alpha)
        levels = tuple(ndtr(z0 + (z0 + e) / (1 - a * (z0 + e))) for e in (z, -z))
    return (np.percentile(theta_b, levels[0] * 100), np.percentile(theta_b, levels[1] * 100))


def clustered_words(rng, speakers):
    """Words and errors per utterance for *speakers* speakers of 3 to 12
    utterances, each speaker with an error probability of its own."""
    sizes = rng.integers(3, 13, speakers)
    groups = np.repeat([f's{k:02d}' for k in range(speakers)], sizes)
    p = rng.beta(2.0, 18.0, speakers)
    words = 3 + rng.poisson(12, groups.size)
    errors = rng.binomial(words, np.repeat(p, sizes)) + rng.poisson(0.3, groups.size)
    return errors.astype(float), words.astype(float), groups


def check_grouped():
    worst, cases = 0.0, 0
    for seed in range(4):
        rng = np.random.default_rng([7, seed])
        errors, words, groups = clustered_words(rng, 6 + 4 * seed)
        other = np.maximum(errors + rng.integers(-2, 3, errors.size), 0)
        scores = rng.normal(size=errors.size) + np.repeat(
            rng.normal(size=len(set(groups))), np.unique(groups, return_counts=True)[1])
        for method in ('percentile', 'bca'):
            pairs = (
                (quietly(module.rate_ci, errors, words, groups=groups, method=method,
                         seed=seed, n_resamples=1500), errors, words),
                (quietly(module.paired_rate_ci, errors, other, words, groups=groups,
                         method=method, seed=seed, n_resamples=1500), errors - other, words),
                (quietly(module.bootstrap_ci, scores, groups=groups, method=method,
                         seed=seed, n_resamples=1500), scores, np.ones(scores.size)),
            )
            for ours, numerator, denominator in pairs:
                low, high = restated(numerator.tolist(), denominator.tolist(),
                                     groups.tolist(), 0.95, 1500, method, seed)
                width = high - low
                worst = max(worst, abs(ours.low - low) / width, abs(ours.high - high) / width)
                cases += 1
    report('grouped intervals vs the loop over resamples (of the width)', cases, worst, 1e-12)

    apart, cases = 0.0, 0
    for seed in range(4):
        rng = np.random.default_rng([8, seed])
        x = rng.lognormal(size=30 + 20 * seed)
        errors, words = rng.poisson(3, x.size).astype(float), 1.0 + rng.poisson(10, x.size)
        each = np.arange(x.size)[::-1].astype(str)
        for method in ('percentile', 'bca'):
            for fn, args in ((module.bootstrap_ci, (x,)), (module.rate_ci, (errors, words))):
                alone = quietly(fn, *args, method=method, seed=seed, n_resamples=2000)
                grouped = quietly(fn, *args, groups=each, method=method, seed=seed,
                                  n_resamples=2000)
                apart = max(apart, abs(alone.low - grouped.low), abs(alone.high - grouped.high))
                cases += 1
    report('groups of one item each vs no groups', cases, apart, 0.0)


# --- 4: the sign-flip test -------------------------------------------------------

def check_permutation():
    worst, cases = 0.0, 0
    for n in range(2, 17):
        for seed in range(3):
            rng = np.random.default_rng([n, seed, 2])
            system, baseline = rng.normal(size=n), rng.normal(size=n) + 0.4
            if seed == 2:                      # ties, and differences of 0
                system, baseline = np.round(system, 0), np.round(baseline, 0)
            for alternative in ('two-sided', 'greater', 'less'):
                ours = quietly(module.permutation_test, system, baseline,
                               alternative=alternative, n_resamples=2 ** 16)
                theirs = scipy_stats.permutation_test(
                    (system, baseline), mean_of_difference, permutation_type='samples',
                    alternative=alternative, n_resamples=2 ** 16, vectorized=True)
                worst = max(worst, abs(ours.pvalue - theirs.pvalue),
                            abs(ours.statistic - theirs.statistic))
                cases += 1
    report('permutation_test vs scipy, every pattern, 2 to 16 items', cases, worst, 0.0)

    worst, cases = 0.0, 0
    for seed in range(6):
        rng = np.random.default_rng([9, seed])
        groups = np.repeat(np.arange(10), rng.integers(2, 9, 10))
        difference = rng.normal(size=groups.size) + 0.15 + np.repeat(
            rng.normal(0, 0.5, 10), np.bincount(groups))
        sums = np.bincount(groups, weights=difference)
        for alternative in ('two-sided', 'greater', 'less'):
            ours = quietly(module.permutation_test, difference, np.zeros(difference.size),
                           groups=groups, alternative=alternative)
            theirs = scipy_stats.permutation_test(
                (sums, np.zeros(sums.size)), mean_of_difference, permutation_type='samples',
                alternative=alternative, n_resamples=2 ** 10, vectorized=True)
            worst = max(worst, abs(ours.pvalue - theirs.pvalue))
            cases += 1
    report('permutation_test with groups vs scipy on the group sums', cases, worst, 0.0)

    apart, scipy_apart, zero_sums, cases = 0.0, 0.0, 0, 0
    rng = np.random.default_rng(12)
    for _ in range(300):
        x = rng.integers(-7, 8, 6) / 10
        zero = sum(round(v * 10) for v in x) == 0
        zero_sums += zero
        for alternative in ('two-sided', 'greater', 'less'):
            ours = quietly(module.permutation_test, x, np.zeros(6), alternative=alternative)
            exact = in_fractions(x, alternative)
            apart = max(apart, abs(ours.pvalue - exact))
            if not zero:
                theirs = scipy_stats.permutation_test(
                    (x, np.zeros(6)), mean_of_difference, permutation_type='samples',
                    alternative=alternative, n_resamples=64, vectorized=True)
                scipy_apart = max(scipy_apart, abs(theirs.pvalue - exact))
            cases += 1
    report('permutation_test on signed tenths vs exact fractions', cases, apart, 0.0)
    report(f'scipy on the same, the {300 - zero_sums} sets whose sum is not 0', cases
           - 3 * zero_sums, scipy_apart, 0.0)

    rng = np.random.default_rng(20)
    system, baseline = rng.normal(size=20), rng.normal(size=20) + 0.45
    exact = module.permutation_test(system, baseline, n_resamples=2 ** 20)
    z = 0.0
    for seed in range(10):
        drawn = module.permutation_test(system, baseline, n_resamples=10000, seed=seed)
        se = math.sqrt(exact.pvalue * (1 - exact.pvalue) / 10000)
        z = max(z, abs(drawn.pvalue - exact.pvalue) / se)
    print(f"\n20 items: exact p over 2**20 patterns {exact.pvalue:.5f}; ten random runs of "
          f"10000 patterns within {z:.2f} standard errors of it")
    report('random patterns vs every pattern, 20 items (standard errors)', 10, z, 4.0)

    z, cases = 0.0, 0
    for n in (50, 200):
        rng = np.random.default_rng([21, n])
        system = rng.normal(size=n)
        baseline = system - rng.normal(0.25 if n == 50 else 0.12, 1.0, n)
        for seed in range(5):
            ours = module.permutation_test(system, baseline, n_resamples=10000, seed=seed)
            theirs = scipy_stats.permutation_test(
                (system, baseline), mean_of_difference, permutation_type='samples',
                n_resamples=10000, vectorized=True, random_state=np.random.default_rng(seed))
            p = (ours.pvalue + theirs.pvalue) / 2
            se = math.sqrt(2 * p * (1 - p) / 10000)
            z = max(z, abs(ours.pvalue - theirs.pvalue) / se)
            cases += 1
        print(f"{n} items: kudio p {ours.pvalue:.4f}, scipy p {theirs.pvalue:.4f} (last seed)")
    report('permutation_test vs scipy, random patterns, 50 and 200 items (s.e.)', cases, z,
           4.0)


def in_fractions(differences, alternative):
    """The sign-flip p-value over every pattern of a few tenths, in exact
    fractions."""
    d = [Fraction(round(v * 10), 10) for v in differences]
    t = sum(d)
    null = [sum(s * v for s, v in zip(signs, d))
            for signs in itertools.product((1, -1), repeat=len(d))]
    if alternative == 'two-sided':
        hits = sum(abs(z) >= abs(t) for z in null)
    elif alternative == 'greater':
        hits = sum(z >= t for z in null)
    else:
        hits = sum(z <= t for z in null)
    return hits / len(null)


# --- 5: Holm ---------------------------------------------------------------------

def step_down(p):
    """The smallest level at which Holm's step-down procedure rejects each
    hypothesis, trying every level at which its decisions can change."""
    m = len(p)
    order = sorted(range(m), key=lambda i: p[i])
    levels = sorted({min(1.0, (m - k) * p[i]) for k, i in enumerate(order)} | {1.0})
    first = [1.0] * m
    for level in reversed(levels):
        for k, i in enumerate(order):
            if (m - k) * p[i] > level:
                break
            first[i] = level
    return first


def check_holm():
    rng = np.random.default_rng(5)
    worst, cases = 0.0, 0
    for m in range(1, 13):
        for _ in range(40):
            p = rng.uniform(0, 0.3, m) ** rng.uniform(0.5, 3)
            if m > 2:
                p[1] = p[0]                    # a tie
            worst = max(worst, float(np.max(np.abs(module.holm(p) - step_down(p.tolist())))))
            cases += 1
    report('holm vs the step-down procedure at every level', cases, worst, 0.0)


# --- 7: the simulation -----------------------------------------------------------

def clustered(rng, speakers, icc=0.3):
    """Utterance scores with mean 0 for *speakers* speakers of 10 to 30
    utterances: a speaker effect of variance *icc*, the rest 1 - icc."""
    sizes = rng.integers(10, 31, speakers)
    groups = np.repeat(np.arange(speakers), sizes)
    x = (rng.normal(0, math.sqrt(icc), speakers)[groups]
         + rng.normal(0, math.sqrt(1 - icc), groups.size))
    return x, groups


def covered(fn, truth, repetitions, setting, make, **kwargs):
    """Share of *repetitions* intervals that hold *truth*."""
    held = 0
    for r in range(repetitions):
        args, extra = make(np.random.default_rng([setting, r]))
        ci = quietly(fn, *args, n_resamples=2000, seed=r, **kwargs, **extra)
        held += ci.low <= truth <= ci.high
    return held / repetitions


def rejected(repetitions, setting, make, **kwargs):
    hits = 0
    for r in range(repetitions):
        args, extra = make(np.random.default_rng([setting, r]))
        hits += quietly(module.permutation_test, *args, n_resamples=1000, seed=r,
                        **kwargs, **extra).pvalue <= 0.05
    return hits / repetitions


#: what kudio/core/stats.py says the simulation gives: (row, column) -> share
SAID = {
    ('i.i.d. lognormal, n=20', 'percentile'): 0.86,
    ('i.i.d. lognormal, n=20', 'bca'): 0.88,
    ('i.i.d. lognormal, n=200', 'percentile'): 0.94,
    ('i.i.d. lognormal, n=200', 'bca'): 0.94,
    ('i.i.d. normal, n=2', 'bca'): 0.50,
    ('i.i.d. normal, n=5', 'bca'): 0.85,
    ('i.i.d. normal, n=10', 'bca'): 0.90,
    ('clustered, 2 speakers', 'by speaker'): 0.50,
    ('clustered, 5 speakers', 'by speaker'): 0.83,
    ('clustered, 10 speakers', 'by utterance'): 0.52,
    ('clustered, 10 speakers', 'by speaker'): 0.89,
    ('clustered, 30 speakers', 'by utterance'): 0.50,
    ('clustered, 30 speakers', 'by speaker'): 0.92,
    ('error rate, 30 speakers', 'by utterance'): 0.67,
    ('error rate, 30 speakers', 'by speaker'): 0.93,
    ('null, clustered, 10 speakers', 'by utterance'): 0.48,
    ('null, clustered, 30 speakers', 'by utterance'): 0.46,
    ('null, clustered, 10 speakers', 'by speaker'): 0.053,
    ('null, clustered, 30 speakers', 'by speaker'): 0.046,
}


def simulate(repetitions):
    rows = []
    t0 = time.perf_counter()
    mean_lognormal = math.exp(0.5)
    for k, n in enumerate((20, 50, 200)):
        for name, truth, draw in (('normal', 0.0, lambda g, n=n: g.normal(size=n)),
                                  ('lognormal', mean_lognormal,
                                   lambda g, n=n: g.lognormal(size=n))):
            row = f'i.i.d. {name}, n={n}'
            setting = 100 + 10 * k + (name == 'lognormal')
            for method in ('percentile', 'bca'):
                rows.append((row, method, covered(
                    module.bootstrap_ci, truth, repetitions, setting,
                    lambda g, draw=draw: ((draw(g),), {}), method=method)))
    for k, n in enumerate((2, 5, 10)):
        for method in ('percentile', 'bca'):
            rows.append((f'i.i.d. normal, n={n}', method, covered(
                module.bootstrap_ci, 0.0, repetitions, 150 + k,
                lambda g, n=n: ((g.normal(size=n),), {}), method=method)))
    print(f"\n  i.i.d. rows: {time.perf_counter() - t0:.0f} s")

    t0 = time.perf_counter()
    for speakers in (2, 5, 10, 30):
        row = f'clustered, {speakers} speakers'
        setting = 200 + speakers
        make = (lambda g, s=speakers: (lambda x, groups: ((x,), {'groups': groups}))(
            *clustered(g, s)))
        if speakers >= 10:
            alone = (lambda g, s=speakers: ((clustered(g, s)[0],), {}))
            rows.append((row, 'by utterance', covered(module.bootstrap_ci, 0.0,
                                                      repetitions, setting, alone)))
        rows.append((row, 'by speaker', covered(module.bootstrap_ci, 0.0, repetitions,
                                                setting, make)))

    def rates(g, grouped):
        errors, words, groups = clustered_words(g, 30)
        return (errors, words), ({'groups': groups} if grouped else {})

    for grouped, column in ((False, 'by utterance'), (True, 'by speaker')):
        rows.append(('error rate, 30 speakers', column,
                     covered(module.rate_ci, 0.12, repetitions, 300,
                             lambda g, grouped=grouped: rates(g, grouped))))
    print(f"  clustered rows: {time.perf_counter() - t0:.0f} s")

    t0 = time.perf_counter()

    def null_iid(g):
        d = g.normal(size=50)
        return (d, np.zeros(d.size)), {}

    rows.append(('null, i.i.d., n=50', 'by utterance',
                 rejected(5 * repetitions, 400, null_iid), 5 * repetitions))
    for speakers in (10, 30):
        def null(g, grouped, s=speakers):
            d, groups = clustered(g, s)
            return (d, np.zeros(d.size)), ({'groups': groups} if grouped else {})

        row = f'null, clustered, {speakers} speakers'
        rows.append((row, 'by utterance', rejected(repetitions, 410 + speakers,
                                                   lambda g, null=null: null(g, False))))
        rows.append((row, 'by speaker', rejected(repetitions, 410 + speakers,
                                                 lambda g, null=null: null(g, True),
                                                 )))
    print(f"  null rows: {time.perf_counter() - t0:.0f} s")
    return rows


# --- 6: another seed ------------------------------------------------------------

def check_seeds():
    """How far another seed moves the ends of an interval from 10000
    resamples, as a share of its width: one standard deviation over 60
    seeds."""
    print("\nThe ends over 60 seeds, 10000 resamples each, one standard deviation as a "
          "share of the width:")
    spread = {}
    for kind in ('normal', 'lognormal'):
        for n in (50, 824):
            rng = np.random.default_rng([30, n])
            x = rng.normal(size=n) if kind == 'normal' else rng.lognormal(size=n)
            ends = np.array([quietly(module.bootstrap_ci, x, seed=seed)[1:3]
                             for seed in range(60)])
            width = float(np.mean(ends[:, 1] - ends[:, 0]))
            low, high = np.std(ends, axis=0, ddof=1) / width
            spread[kind, n] = (low, high)
            print(f"  {kind:>9}, {n:3d} values: low end {low:.2%}, high end {high:.2%}")
    normal = max(max(spread['normal', n]) for n in (50, 824))
    skewed = max(max(spread['lognormal', n]) for n in (50, 824))
    report(f'seed to seed, normal data: said about 1% of the width, got {normal:.2%}', 4,
           abs(normal - 0.01), 0.002)
    report(f'seed to seed, the long end of lognormal data: said about 2%, got '
           f'{skewed:.2%}', 4, abs(skewed - 0.02), 0.004)


def print_simulation(rows, repetitions):
    se = math.sqrt(0.95 * 0.05 / repetitions)
    print(f"\nShare of 95% intervals holding the true value, and share of tests "
          f"rejecting at 0.05 under the null, each with its standard error; "
          f"{repetitions} data sets per cell, {5 * repetitions} for the i.i.d. null:")
    columns = ('percentile', 'bca', 'by utterance', 'by speaker')
    table, sizes = {}, {}
    for row, column, share, *count in rows:
        table.setdefault(row, {})[column] = share
        sizes[row, column] = count[0] if count else repetitions
    width = max(len(r) for r in table)
    print(f"  {'':<{width}} " + ' '.join(f'{c:>15}' for c in columns))
    for row, cells in table.items():
        shown = []
        for c in columns:
            if c in cells:
                error = math.sqrt(cells[c] * (1 - cells[c]) / sizes[row, c])
                shown.append(f"{cells[c]:8.3f} ({error:.3f})")
            else:
                shown.append(f"{'':>15}")
        print(f"  {row:<{width}} " + ' '.join(shown))

    # three standard errors of the share, and the rounding of the figure said
    for (row, column), said in SAID.items():
        got = table[row][column]
        allowed = 3 * math.sqrt(said * (1 - said) / repetitions) + 0.005
        report(f'{row}, {column}: said {said:g}, got {got:.3f}', repetitions,
               abs(got - said), allowed)
    for row in ('i.i.d. normal, n=200', 'i.i.d. normal, n=50'):
        for column in ('percentile', 'bca'):
            got = table[row][column]
            report(f'{row}, {column}: within 3 s.e. of 0.95', repetitions,
                   abs(got - 0.95), 3 * se)
    got = table['null, i.i.d., n=50']['by utterance']
    report('null, i.i.d.: rejects within 3 s.e. of 0.05', 5 * repetitions,
           abs(got - 0.05), 3 * math.sqrt(0.05 * 0.95 / (5 * repetitions)))
    got = table['null, clustered, 10 speakers']['by speaker']
    report('null, clustered, 10 speakers by speaker: within 3 s.e. of 0.05', repetitions,
           abs(got - 0.05), 3 * math.sqrt(0.05 * 0.95 / repetitions))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--repetitions', type=int, default=2000,
                        help="data sets per cell of the simulation (default 2000)")
    args = parser.parse_args()
    start = time.perf_counter()
    import scipy
    print(f"kudio {kudio.__version__} from {Path(kudio.__file__).parent}; numpy "
          f"{np.__version__}, scipy {scipy.__version__}, Python {sys.version.split()[0]}")
    check_bootstrap()
    check_grouped()
    check_permutation()
    check_holm()
    check_seeds()
    rows = simulate(args.repetitions)
    print_simulation(rows, args.repetitions)
    failed = print_table()
    print(f"\n{time.perf_counter() - start:.0f} s; "
          + ("SOME DIFFERENCES ARE PAST TOLERANCE" if failed
             else "every difference within its tolerance"))
    return 1 if failed else 0


def print_table():
    """The table of every check; True if one is past its limit."""
    width = max(len(r[0]) for r in RESULTS)
    print(f"\n{'check':<{width}} {'cases':>5} {'max |diff|':>11} {'limit':>7}")
    failed = False
    for check, cases, diff, tolerance in RESULTS:
        over = diff > tolerance
        failed |= over
        limit = '-' if math.isinf(tolerance) else f'{tolerance:.1e}'.replace('.0e', 'e')
        print(f"{check:<{width}} {cases:>5} {diff:11.2e} {limit:>7}{'  PAST' if over else ''}")
    return failed


if __name__ == '__main__':
    sys.exit(main())
