# -*- coding: utf-8 -*-
"""Confidence intervals and paired tests: against scipy.stats on the same
resamples and over every sign pattern, against answers worked out by hand,
what groups change and what they must leave alone, and the input they refuse
or cannot put an interval on."""
from __future__ import annotations

import itertools
import math
import warnings
from fractions import Fraction

import numpy as np
import pytest
from scipy import stats as scipy_stats
from scipy.special import ndtr, ndtri

from kudio.core import stats as module
from kudio.core.stats import (FEW_UNITS, ConfidenceInterval, PermutationResult,
                              bootstrap_ci, holm, paired_ci, paired_rate_ci,
                              permutation_test, rate_ci)


def _quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return fn(*args, **kwargs)


def _mean_difference(a, b, axis):
    return np.mean(a - b, axis=axis)


def _speakers(seed, speakers=12, icc=0.3):
    """Scores with mean 0 for *speakers* speakers of 5 to 14 utterances, a
    speaker effect of variance *icc* and the rest 1 - icc."""
    rng = np.random.default_rng(seed)
    sizes = rng.integers(5, 15, speakers)
    which = np.repeat(np.arange(speakers), sizes)
    x = rng.normal(0, math.sqrt(icc), speakers)[which] + rng.normal(0, math.sqrt(1 - icc),
                                                                      which.size)
    return x, np.array([f'spk{k:02d}' for k in which])


def _width(ci):
    return ci.high - ci.low


# --- against scipy --------------------------------------------------------------

@pytest.mark.parametrize('n', [12, 60, 300])
@pytest.mark.parametrize('kind', ['normal', 'lognormal'])
def test_percentile_ends_are_scipys_resamples_read_by_np_percentile(n, kind):
    # the resamples are scipy's to the last bit, and so are the ends read off
    # them as scipy up to 1.16 reads them; 1.18 reads its own ends with
    # scipy.stats.quantile, which lands within about 1e-15 of the width
    rng = np.random.default_rng(n)
    x = rng.normal(size=n) if kind == 'normal' else rng.lognormal(size=n)
    ours = bootstrap_ci(x, method='percentile', n_resamples=2000, seed=3)
    theirs = scipy_stats.bootstrap((x,), np.mean, method='percentile', n_resamples=2000,
                                   random_state=np.random.default_rng(3))
    alpha = (1 - 0.95) / 2
    resampled = theirs.bootstrap_distribution
    assert ours.low == np.percentile(resampled, alpha * 100)
    assert ours.high == np.percentile(resampled, (1 - alpha) * 100)
    assert ours.low == pytest.approx(theirs.confidence_interval.low,
                                     abs=1e-14 * _width(ours))
    assert ours.high == pytest.approx(theirs.confidence_interval.high,
                                      abs=1e-14 * _width(ours))
    assert ours.estimate == np.mean(x)


@pytest.mark.parametrize('n', [20, 60, 300])
@pytest.mark.parametrize('kind', ['normal', 'lognormal'])
def test_bca_ends_are_scipys_to_rounding(n, kind):
    # from 20 items up no resample holds each item once (20!/20**20 = 2e-8)
    rng = np.random.default_rng(n + 1)
    x = rng.normal(size=n) if kind == 'normal' else rng.lognormal(size=n)
    ours = bootstrap_ci(x, n_resamples=2000, seed=4)
    ci = scipy_stats.bootstrap((x,), np.mean, method='BCa', n_resamples=2000,
                               random_state=np.random.default_rng(4)).confidence_interval
    assert ours.method == 'bca'
    assert ours.low == pytest.approx(ci.low, abs=1e-12 * _width(ours))
    assert ours.high == pytest.approx(ci.high, abs=1e-12 * _width(ours))


@pytest.mark.parametrize('method', ['percentile', 'bca'])
def test_paired_interval_is_scipys_paired_bootstrap(method):
    rng = np.random.default_rng(11)
    system = rng.normal(3.0, 0.4, 80)
    baseline = system - rng.lognormal(-2.0, 0.8, 80)
    ours = paired_ci(system, baseline, method=method, n_resamples=2000, seed=5)
    ci = scipy_stats.bootstrap((system, baseline), _mean_difference, paired=True,
                               method=method, n_resamples=2000,
                               random_state=np.random.default_rng(5)).confidence_interval
    assert ours.estimate == pytest.approx(np.mean(system - baseline), rel=1e-15)
    assert ours.low == pytest.approx(ci.low, abs=1e-12 * _width(ours))
    assert ours.high == pytest.approx(ci.high, abs=1e-12 * _width(ours))


def test_bca_takes_resamples_that_hold_each_item_once_as_the_estimate():
    """At 5 items 3.8% of the resamples hold each item once and give the
    estimate exactly; summed in another order they can miss it in the last
    bit. kudio counts them as equal to it."""
    x = np.random.default_rng(2).lognormal(size=5)
    pick = np.random.default_rng(0).integers(0, 5, (4000, 5))
    once = np.all(np.sort(pick, axis=1) == np.arange(5), axis=1)
    assert 100 < np.count_nonzero(once) < 200
    theta = np.mean(x)
    theta_b = x[pick].mean(axis=1)
    theta_b[once] = theta
    share = (np.sum(theta_b < theta) + np.sum(theta_b <= theta)) / (2 * 4000)
    d = x - theta                     # the jackknife of a mean, times n - 1
    a = np.sum(d ** 3) / (6 * np.sum(d ** 2) ** 1.5)
    z0, z = ndtri(share), ndtri(0.025)
    levels = [ndtr(z0 + (z0 + e) / (1 - a * (z0 + e))) for e in (z, -z)]
    want = np.percentile(theta_b, [levels[0] * 100, levels[1] * 100])
    ours = _quiet(bootstrap_ci, x, n_resamples=4000, seed=0)
    assert ours.low == pytest.approx(want[0], abs=1e-12 * _width(ours))
    assert ours.high == pytest.approx(want[1], abs=1e-12 * _width(ours))


@pytest.mark.parametrize('n', [2, 3, 5, 8, 11])
@pytest.mark.parametrize('alternative', ['two-sided', 'greater', 'less'])
def test_permutation_test_over_every_pattern_is_scipys(n, alternative):
    rng = np.random.default_rng(n)
    for system, baseline in ((rng.normal(size=n), rng.normal(size=n) + 0.5),
                             (np.round(rng.normal(size=n)), np.round(rng.normal(size=n)))):
        ours = _quiet(permutation_test, system, baseline, alternative=alternative)
        theirs = scipy_stats.permutation_test(
            (system, baseline), _mean_difference, permutation_type='samples',
            alternative=alternative, n_resamples=10000, vectorized=True)
        assert ours.exact and ours.n_resamples == 2 ** n
        assert ours.pvalue == theirs.pvalue
        assert ours.statistic == theirs.statistic


def test_grouped_permutation_test_is_scipys_on_the_group_sums():
    x, groups = _speakers(1, speakers=9)
    x = x + 0.3
    sums = np.array([x[groups == g].sum() for g in dict.fromkeys(groups)])
    for alternative in ('two-sided', 'greater', 'less'):
        ours = permutation_test(x, np.zeros(x.size), groups=groups, alternative=alternative)
        theirs = scipy_stats.permutation_test(
            (sums, np.zeros(sums.size)), _mean_difference, permutation_type='samples',
            alternative=alternative, n_resamples=2 ** 9, vectorized=True)
        assert ours.exact and ours.n_groups == 9 and ours.n_resamples == 512
        assert ours.pvalue == theirs.pvalue


def test_random_patterns_land_within_monte_carlo_error():
    rng = np.random.default_rng(16)
    system, baseline = rng.normal(size=16), rng.normal(size=16) + 0.6
    exact = permutation_test(system, baseline, n_resamples=2 ** 16)
    assert exact.exact and 0.02 < exact.pvalue < 0.98
    se = math.sqrt(exact.pvalue * (1 - exact.pvalue) / 5000)
    for seed in range(5):
        drawn = permutation_test(system, baseline, n_resamples=5000, seed=seed)
        assert not drawn.exact and drawn.n_resamples == 5000
        assert abs(drawn.pvalue - exact.pvalue) < 4 * se

    rng = np.random.default_rng(60)
    system = rng.normal(size=60)
    baseline = system - rng.normal(0.3, 1.0, 60)
    ours = permutation_test(system, baseline, n_resamples=5000)
    theirs = scipy_stats.permutation_test(
        (system, baseline), _mean_difference, permutation_type='samples',
        n_resamples=5000, vectorized=True, random_state=np.random.default_rng(0))
    p = (ours.pvalue + theirs.pvalue) / 2
    assert abs(ours.pvalue - theirs.pvalue) < 4 * math.sqrt(2 * p * (1 - p) / 5000)


def _exact_p(differences, alternative):
    """The sign-flip p-value over every pattern, in exact fractions of tenths."""
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


def test_patterns_that_tie_the_observed_one_count_as_extreme():
    # in floats the patterns that tie the observed sum of tenths land a bit
    # either side of it; counted only when at or past it, two-sided gives
    # 0.78125 here
    x = [0.5, -0.4, -0.6, -0.3, -0.1, 0.5]
    for alternative, want in (('two-sided', 0.8125), ('less', 0.40625)):
        ours = permutation_test(x, np.zeros(6), alternative=alternative)
        theirs = scipy_stats.permutation_test(
            (np.array(x), np.zeros(6)), _mean_difference, permutation_type='samples',
            alternative=alternative, n_resamples=10000, vectorized=True)
        assert ours.pvalue == theirs.pvalue == want == _exact_p(x, alternative)


def test_signed_tenths_against_exact_fractions_and_scipy():
    rng = np.random.default_rng(12)
    for _ in range(60):
        x = rng.integers(-7, 8, 6) / 10
        zero_sum = sum(round(v * 10) for v in x) == 0
        for alternative in ('two-sided', 'greater', 'less'):
            ours = _quiet(permutation_test, x, np.zeros(6), alternative=alternative)
            assert ours.pvalue == _exact_p(x, alternative)
            if not zero_sum:
                # where the observed sum is 0 scipy's tolerance, relative to
                # it, is 0 too, and its one-sided p-value follows the rounding
                theirs = scipy_stats.permutation_test(
                    (x, np.zeros(6)), _mean_difference, permutation_type='samples',
                    alternative=alternative, n_resamples=10000, vectorized=True)
                assert ours.pvalue == theirs.pvalue


def test_the_random_p_value_counts_the_observed_pattern():
    # every drawn pattern less extreme than the observed one: (0 + 1) / (B + 1)
    result = permutation_test(np.full(40, 1.0), np.zeros(40), n_resamples=1000)
    assert result.pvalue == 1 / 1001
    exact = permutation_test(np.full(6, 1.0), np.zeros(6), alternative='greater')
    assert exact.pvalue == 1 / 64


# --- Holm ------------------------------------------------------------------------

def test_holm_worked_by_hand():
    # ranked 0.005, 0.01, 0.03, 0.04: times 4, 3, 2, 1 gives 0.02, 0.03, 0.06,
    # 0.04, and the last is raised to the 0.06 before it
    adjusted = holm([0.01, 0.04, 0.03, 0.005])
    np.testing.assert_array_equal(adjusted, [3 * 0.01, 2 * 0.03, 2 * 0.03, 4 * 0.005])


def test_holm_caps_at_one_keeps_ties_equal_and_takes_a_dict():
    np.testing.assert_array_equal(holm([0.5, 0.6]), [1.0, 1.0])
    np.testing.assert_array_equal(holm([0.6, 0.7]), [1.0, 1.0])
    np.testing.assert_array_equal(holm([0.5, 0.02, 0.02]), [0.5, 3 * 0.02, 3 * 0.02])
    assert holm({'A': 0.01, 'B': 0.04}) == {'A': 2 * 0.01, 'B': 0.04}
    assert holm([]).shape == (0,)
    assert holm([0.3]).tolist() == [0.3]


# --- rates -----------------------------------------------------------------------

def test_a_rate_is_a_ratio_of_sums():
    # a mean of the two utterances' rates would be (1 + 0.3) / 2 = 0.65
    ci = _quiet(rate_ci, [1, 30], [1, 100])
    assert ci.estimate == 31 / 101


@pytest.mark.parametrize('method', ['percentile', 'bca'])
def test_rate_interval_over_two_items_worked_out(method):
    # resamples of the two items give 2/20, 4/20 or 6/20 with chances 1/4,
    # 1/2, 1/4, so the 2.5% and 97.5% points are the ends of that range;
    # BCa's z0 is 0 to within the draw and its acceleration 0 by symmetry
    ci = _quiet(rate_ci, [1, 3], [10, 10], method=method)
    assert (ci.estimate, ci.low, ci.high) == (0.2, 0.1, 0.3)
    grouped = _quiet(rate_ci, [1, 0, 3, 0], [5, 5, 5, 5], groups=['a', 'a', 'b', 'b'],
                     method=method)
    assert (grouped.estimate, grouped.low, grouped.high, grouped.n, grouped.n_groups) == (
        0.2, 0.1, 0.3, 4, 2)


def test_rate_interval_against_a_loop_over_the_same_resamples():
    rng = np.random.default_rng(3)
    groups = np.repeat(np.arange(15), rng.integers(2, 9, 15))
    words = 3.0 + rng.poisson(12, groups.size)
    errors = rng.binomial(words.astype(int), rng.beta(2, 18, 15)[groups]).astype(float)
    ours = rate_ci(errors, words, groups=groups, method='percentile', n_resamples=2000,
                   seed=7)
    pick = np.random.default_rng(7).integers(0, 15, (2000, 15))
    theta_b = []
    for row in pick:
        top = math.fsum(e for k in row for e in errors[groups == k])
        bottom = math.fsum(w for k in row for w in words[groups == k])
        theta_b.append(top / bottom)
    want = np.percentile(theta_b, [2.5, 97.5])
    assert ours.estimate == errors.sum() / words.sum()
    assert ours.low == pytest.approx(want[0], abs=1e-12 * _width(ours))
    assert ours.high == pytest.approx(want[1], abs=1e-12 * _width(ours))


def test_paired_rate_is_the_difference_of_the_pooled_rates():
    rng = np.random.default_rng(4)
    words = 5 + rng.poisson(10, 200)
    base = rng.binomial(words, 0.15)
    system = rng.binomial(words, 0.12)
    ci = paired_rate_ci(system, base, words, n_resamples=2000)
    assert ci.estimate == pytest.approx(system.sum() / words.sum() - base.sum() / words.sum(),
                                        abs=1e-16)
    assert ci.low < ci.estimate < ci.high < 0
    test = permutation_test(system, base, totals=words, alternative='less')
    assert test.statistic == pytest.approx(ci.estimate, abs=1e-16)
    # the totals only scale the statistic; the p-value is the counts' own
    assert test.pvalue == permutation_test(system, base, alternative='less').pvalue


# --- groups ----------------------------------------------------------------------

def test_groups_of_one_item_each_give_the_ungrouped_result_to_the_last_bit():
    rng = np.random.default_rng(5)
    x, y = rng.lognormal(size=40), rng.lognormal(size=40)
    errors, words = rng.poisson(3, 40), 1 + rng.poisson(10, 40)
    each = [f'u{k}' for k in range(40)][::-1]     # sorted, they would come in another order
    for method in ('percentile', 'bca'):
        for fn, args in ((bootstrap_ci, (x,)), (paired_ci, (x, y)),
                         (rate_ci, (errors, words)), (paired_rate_ci, (errors, x > 1, words))):
            alone = fn(*args, method=method, n_resamples=2000)
            grouped = fn(*args, groups=each, method=method, n_resamples=2000)
            assert grouped.n_groups == 40 and alone.n_groups is None
            assert grouped._replace(n_groups=None) == alone
    for n_resamples in (5000, 2 ** 40):
        x8, y8 = (x[:8], y[:8]) if n_resamples > 2 ** 20 else (x, y)
        alone = _quiet(permutation_test, x8, y8, n_resamples=n_resamples)
        grouped = _quiet(permutation_test, x8, y8, groups=each[:x8.size],
                         n_resamples=n_resamples)
        assert grouped._replace(n_groups=None) == alone


def test_grouped_bca_leaves_out_one_group_at_a_time():
    x, groups = _speakers(6, speakers=15)
    ours = bootstrap_ci(x, groups=groups, n_resamples=2000, seed=2)
    labels = list(dict.fromkeys(groups))
    sums = np.array([x[groups == g].sum() for g in labels])
    sizes = np.array([np.sum(groups == g) for g in labels])
    pick = np.random.default_rng(2).integers(0, 15, (2000, 15))
    theta_b = sums[pick].sum(axis=1) / sizes[pick].sum(axis=1)
    theta = x.mean()
    without = np.array([x[groups != g].mean() for g in labels])
    u = 14 * (without.mean() - without)
    a = np.sum(u ** 3) / (6 * np.sum(u ** 2) ** 1.5)
    z0 = ndtri((np.sum(theta_b < theta) + np.sum(theta_b <= theta)) / 4000)
    z = ndtri(0.025)
    levels = [ndtr(z0 + (z0 + e) / (1 - a * (z0 + e))) for e in (z, -z)]
    want = np.percentile(theta_b, [levels[0] * 100, levels[1] * 100])
    assert ours.low == pytest.approx(want[0], abs=1e-12 * _width(ours))
    assert ours.high == pytest.approx(want[1], abs=1e-12 * _width(ours))


def test_speakers_widen_the_interval_and_raise_the_p_value():
    # about 10 utterances a speaker at an intra-class correlation of 0.3: a
    # design effect near 3.6, so an interval near 1.9 times as wide
    x, groups = _speakers(8, speakers=20)
    x = x + 0.25
    alone = bootstrap_ci(x, n_resamples=2000)
    grouped = bootstrap_ci(x, groups=groups, n_resamples=2000)
    assert grouped.estimate == alone.estimate
    assert _width(grouped) > 1.5 * _width(alone)
    assert permutation_test(x, np.zeros(x.size)).pvalue < 0.001
    assert 0.01 < permutation_test(x, np.zeros(x.size), groups=groups).pvalue < 0.05


def test_group_labels_are_taken_as_given():
    x = np.arange(40.0)
    mixed = _quiet(bootstrap_ci, x[:4], groups=[1, '1', 2, '2'])
    assert mixed.n_groups == 4
    tuples = bootstrap_ci(x, groups=[('s', k // 4) for k in range(40)], n_resamples=1000)
    assert tuples.n_groups == 10
    assert tuples == bootstrap_ci(x, groups=np.arange(40) // 4, n_resamples=1000)
    with pytest.raises(ValueError, match=r"item 1 has .*nan.* for its group"):
        bootstrap_ci(x[:3], groups=np.array(['a', np.float32('nan'), 'b'], dtype=object))
    pd = pytest.importorskip('pandas')
    with pytest.raises(ValueError, match=r"item 2 has <NA> for its group"):
        bootstrap_ci(x[:3], groups=pd.array(['a', 'b', pd.NA], dtype='string'))
    with pytest.raises(ValueError, match=r"groups has to be one label per item, got a "
                                         r"single str"):
        bootstrap_ci(x[:3], groups='abc')


def test_values_whose_sums_leave_float64_are_refused():
    with pytest.raises(ValueError, match=r"^bootstrap_ci\(\): values reach 1e\+308, and "
                                         r"summed over resamples of 20 items they would "
                                         r"pass the largest float64"):
        bootstrap_ci(np.r_[np.ones(19), 1e308])
    with pytest.raises(ValueError, match=r"^paired_ci\(\): the differences reach inf"):
        paired_ci(np.full(20, 1e308), np.full(20, -1e308))
    with pytest.raises(ValueError, match=r"^rate_ci\(\): totals reach"):
        rate_ci(np.ones(20), np.full(20, 1e307))
    with pytest.raises(ValueError, match=r"^permutation_test\(\): the differences reach"):
        permutation_test(np.full(20, 1e307), np.zeros(20))


def test_group_labels_are_numbered_in_order_of_first_appearance():
    x = np.arange(6.0)
    one = _quiet(bootstrap_ci, x, groups=['b', 'b', 'a', 'a', 'c', 'c'], n_resamples=2000)
    other = _quiet(bootstrap_ci, x, groups=[7, 7, 3, 3, 9, 9], n_resamples=2000)
    assert one == other


# --- cases with no interval, and the warnings -------------------------------------

def _caught(fn, *args, **kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = fn(*args, **kwargs)
    return result, [str(w.message) for w in caught]


def test_an_interval_with_no_width_comes_with_a_warning():
    no_width = r"every resample gives the estimate, so the interval has no width"
    with pytest.warns(UserWarning, match=r"^bootstrap_ci\(\): " + no_width):
        ci = bootstrap_ci(np.full(30, 0.1))
    assert ci.low == ci.high == ci.estimate == np.mean(np.full(30, 0.1))
    with pytest.warns(UserWarning, match=r"^rate_ci\(\): " + no_width):
        wer = rate_ci(np.zeros(500), np.full(500, 12))
    assert (wer.estimate, wer.low, wer.high) == (0.0, 0.0, 0.0)
    paired, said = _caught(paired_ci, np.full(25, 3.2), np.full(25, 3.2),
                           groups=np.arange(25) // 2)
    assert (paired.low, paired.high, paired.estimate) == (0.0, 0.0, 0.0)
    assert len(said) == 1 and 'no width' in said[0]
    # equal group means over groups of different sizes: equal to rounding
    x, groups = np.repeat([2.0, 2.0, 2.0], [3, 7, 11]), np.repeat([0, 1, 2], [3, 7, 11])
    flat, said = _caught(bootstrap_ci, x, groups=groups)
    assert flat.low == flat.high == 2.0 and len(said) == 1 and 'no width' in said[0]
    assert permutation_test(np.full(25, 3.2), np.full(25, 3.2)).pvalue == 1.0


def test_one_resample_that_moves_is_not_an_interval_with_no_width():
    x = np.random.default_rng(3).normal(size=40)
    ci, said = _caught(bootstrap_ci, x, n_resamples=1, method='percentile')
    assert ci.low == ci.high != ci.estimate
    assert len(said) == 1 and 'with 1 resample each end' in said[0]


def test_one_group_and_one_item_have_no_interval():
    with pytest.warns(UserWarning, match=r"^bootstrap_ci\(\): all 3 items are in one "
                                         r"group.*; the interval is nan$"):
        ci = bootstrap_ci([1.0, 2.0, 3.0], groups=['s1'] * 3)
    assert ci.estimate == 2.0 and math.isnan(ci.low) and math.isnan(ci.high)
    assert ci.n_groups == 1
    with pytest.warns(UserWarning, match=r"^paired_ci\(\): there is 1 item"):
        assert math.isnan(paired_ci([1.0], [0.5]).low)
    with pytest.warns(UserWarning, match=r"^rate_ci\(\): there are no items"):
        empty = rate_ci([], [])
    assert math.isnan(empty.estimate) and empty.n == 0
    with pytest.warns(UserWarning, match=r"^permutation_test\(\): there are no items"):
        assert math.isnan(permutation_test([], []).pvalue)


def test_two_speakers_give_the_range_of_their_means_and_a_warning():
    # equal sizes: resamples give speaker a's mean, the overall mean or b's
    x = np.r_[np.full(50, 2.0), np.full(50, 3.0)] + np.tile(np.linspace(-0.1, 0.1, 50), 2)
    groups = np.repeat(['a', 'b'], 50)
    with pytest.warns(UserWarning, match=r"^bootstrap_ci\(\): 2 groups are too few for "
                                         r"this interval to mean much"):
        ci = bootstrap_ci(x, groups=groups)
    assert ci.low == pytest.approx(2.0) and ci.high == pytest.approx(3.0)
    with pytest.warns(UserWarning, match=r"no two-sided p-value under 0\.5 is possible"):
        assert permutation_test(x, np.zeros(100), groups=groups).pvalue == 0.5


def test_the_few_groups_warning_starts_under_few_units():
    x, groups = _speakers(8, speakers=FEW_UNITS)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        bootstrap_ci(x, groups=groups, n_resamples=1000)
    labels = np.where(groups == groups[-1], groups[0], groups)      # one group fewer
    with pytest.warns(UserWarning, match=f"{FEW_UNITS - 1} groups are too few"):
        bootstrap_ci(x, groups=labels, n_resamples=1000)
    with pytest.warns(UserWarning, match="5 items are too few"):
        bootstrap_ci(np.arange(5.0), n_resamples=1000)


def test_a_test_that_cannot_reach_005_says_so():
    with pytest.warns(UserWarning, match=r"5 groups allow 32 sign patterns, so no "
                                         r"two-sided p-value under 0\.0625"):
        permutation_test(np.arange(10.0), np.zeros(10), groups=np.arange(10) // 2)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert permutation_test(np.arange(1.0, 7.0), np.zeros(6)).pvalue == 2 / 64
        permutation_test(np.arange(1.0, 6.0), np.zeros(5), alternative='greater')


def test_few_resamples_warn():
    with pytest.warns(UserWarning, match=r"^bootstrap_ci\(\): with 200 resamples each "
                                         r"end of the interval rests"):
        bootstrap_ci(np.arange(30.0), n_resamples=200)


def test_bca_without_an_interval_falls_back_to_the_percentile_one():
    # one large value among twenty: an acceleration of 0.15, and at this
    # level 1 - a (z0 + z) is no longer positive
    x = np.r_[np.zeros(19), 1000.0]
    with pytest.warns(UserWarning, match=r"^bootstrap_ci\(\): BCa has no interval here: "
                                         r"the skew correction runs past its range"):
        ci = bootstrap_ci(x, confidence=1 - 1e-12)
    assert ci.method == 'percentile'
    assert ci == bootstrap_ci(x, confidence=1 - 1e-12, method='percentile')


def test_totals_of_zero():
    with pytest.warns(UserWarning, match=r"^rate_ci\(\): the totals add up to 0, so "
                                         r"there is no rate; the interval is nan$"):
        assert math.isnan(rate_ci([1, 2], [0, 0]).estimate)
    with pytest.warns(UserWarning, match=r"resamples drew only items whose totals are 0"):
        assert math.isnan(rate_ci([1] * 10 + [3], [0] * 10 + [5]).low)
    # one nan, one warning, though 3 items are also too few
    ci, said = _caught(rate_ci, [1, 0, 3], [0, 0, 5])
    assert math.isnan(ci.low) and len(said) == 1 and 'totals are 0' in said[0]
    # an empty reference among others adds its insertions to the errors
    ci = _quiet(rate_ci, [2, 1, 3, 0, 1, 2, 4, 1, 0, 2, 3], [0] + [10] * 10)
    assert ci.estimate == 19 / 100
    with pytest.warns(UserWarning, match=r"^permutation_test\(\): the totals add up to 0"):
        permutation_test([1, 2], [0, 0], totals=[0, 0])


def test_warnings_point_at_the_callers_line():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        bootstrap_ci([1.0, 2.0, 3.0], groups=['a', 'a', 'a'])
    assert caught and all(w.filename == __file__ for w in caught)


# --- determinism ----------------------------------------------------------------------

def test_a_seed_fixes_the_result_and_another_moves_it():
    x = np.random.default_rng(9).lognormal(size=50)
    assert bootstrap_ci(x, n_resamples=2000) == bootstrap_ci(x, n_resamples=2000, seed=0)
    assert bootstrap_ci(x, n_resamples=2000, seed=1) != bootstrap_ci(x, n_resamples=2000)
    generated = bootstrap_ci(x, n_resamples=2000, seed=np.random.default_rng(0))
    assert generated == bootstrap_ci(x, n_resamples=2000)
    assert math.isfinite(bootstrap_ci(x, n_resamples=2000, seed=None).low)
    assert (permutation_test(x, x[::-1], seed=4) == permutation_test(x, x[::-1], seed=4))
    assert isinstance(bootstrap_ci(x, n_resamples=2000), ConfidenceInterval)
    assert isinstance(permutation_test(x, x[::-1]), PermutationResult)


def test_few_random_patterns_warn():
    x = np.random.default_rng(11).normal(size=100)
    with pytest.warns(UserWarning, match=r"^permutation_test\(\): with 10 sign patterns "
                                         r"drawn at random .* under 1/11 = 0\.0909, so the "
                                         r"test cannot reject at 0\.05 whatever the data"):
        assert permutation_test(x, np.zeros(100), n_resamples=10).pvalue >= 1 / 11
    with pytest.warns(UserWarning, match=r"with 500 sign patterns drawn at random the "
                                         r"p-value rests on the draw"):
        permutation_test(x, np.zeros(100), n_resamples=500)


def test_many_items_take_the_random_path_without_overflow():
    x = np.random.default_rng(10).normal(size=3000)
    result = permutation_test(x, np.zeros(3000), n_resamples=1000)
    assert not result.exact and result.n_resamples == 1000 and result.n == 3000
    ci = bootstrap_ci(x, n_resamples=1000)
    assert ci.low < ci.estimate < ci.high and ci.n == 3000


# --- input that is refused ------------------------------------------------------------

@pytest.mark.parametrize('call, message', [
    (lambda: bootstrap_ci([1.0, np.nan, 2.0, 3.0, 4.0]),
     r"^bootstrap_ci\(\): 1 of the 5 values is NaN or infinite\. A metric gives NaN"),
    (lambda: bootstrap_ci([1.0, np.inf, -np.inf]), r"2 of the 3 values are NaN or infinite"),
    (lambda: paired_ci([1.0, 2.0, 3.0], [1.0, 2.0]),
     r"^paired_ci\(\): 3 system scores and 2 baseline scores; they pair up by position"),
    (lambda: bootstrap_ci([1.0, 2.0, 3.0], groups=['a', 'b']),
     r"^bootstrap_ci\(\): 2 group labels for 3 items"),
    (lambda: bootstrap_ci([1.0, 2.0, 3.0], groups=['a', None, 'b']),
     r"item 1 has None for its group"),
    (lambda: bootstrap_ci([1.0, 2.0, 3.0], groups=[1.0, np.nan, 2.0]),
     r"item 1 has nan for its group"),
    (lambda: bootstrap_ci([1.0, 2.0], groups=[[1, 2], [3, 4]]),
     r"group labels have to be numbers, text or tuples of them, got \[1, 2\]"),
    (lambda: bootstrap_ci([1.0, 2.0], groups=np.ones((2, 2))), r"got shape \(2, 2\)"),
    (lambda: rate_ci([1, -1], [5, 5]), r"^rate_ci\(\): errors cannot be negative"),
    (lambda: paired_rate_ci([1, 2], [1, 1], [5, -5]), r"totals cannot be negative"),
    (lambda: paired_rate_ci([1, 2], [1], [5, 5]), r"2 system error counts and 1 baseline"),
    (lambda: rate_ci([1, 2], [5]), r"^rate_ci\(\): 2 error counts and 1 total;"),
    (lambda: bootstrap_ci(np.ones((3, 2))), r"one-dimensional sequence; got shape \(3, 2\)"),
    (lambda: bootstrap_ci(['0.5', '0.7']), r"has to hold real numbers, got an array of <U3"),
    (lambda: bootstrap_ci([0.5, None]), r"has to hold real numbers, got None among them"),
    (lambda: bootstrap_ci([1 + 2j, 3j]), r"got an array of complex128"),
    (lambda: bootstrap_ci(np.ma.array([1.0, 2.0], mask=[0, 1])),
     r"is a masked array with 1 masked value"),
    (lambda: bootstrap_ci([1.0, 2.0], confidence=95), r"confidence has to be a level "
                                                      r"strictly between 0 and 1"),
    (lambda: bootstrap_ci([1.0, 2.0], confidence=1.0), r"confidence has to be"),
    (lambda: bootstrap_ci([1.0, 2.0], confidence=True), r"confidence has to be"),
    (lambda: bootstrap_ci([1.0, 2.0], n_resamples=0), r"n_resamples has to be a positive"),
    (lambda: bootstrap_ci([1.0, 2.0], n_resamples=1e4), r"n_resamples has to be a positive"),
    (lambda: permutation_test([1.0], [2.0], n_resamples=True), r"n_resamples has to be"),
    (lambda: bootstrap_ci([1.0, 2.0], method='BCa'), r"method has to be one of"),
    (lambda: permutation_test([1.0], [2.0], alternative='two_sided'),
     r"^permutation_test\(\): alternative has to be one of"),
    (lambda: permutation_test([1.0, 2.0], [2.0, 1.0], totals=[1.0]),
     r"2 scores and 1 total;"),
    (lambda: bootstrap_ci([1.0, 2.0], seed='abc'), r"seed has to be None"),
    (lambda: holm([0.5, 1.2]), r"^holm\(\): p-values lie between 0 and 1, got 1\.2"),
    (lambda: holm([0.5, np.nan]), r"^holm\(\): 1 of the 2 p-values is NaN or infinite\. "
                                  r"A test without a p-value"),
    (lambda: holm({'A': 0.1, 'B': 'x'}), r"^holm\(\): p-values has to hold real numbers"),
])
def test_malformed_input_is_refused(call, message):
    with pytest.raises(ValueError, match=message):
        call()


def test_booleans_and_column_vectors_are_read():
    flags = np.array([True, False, True, True] * 10)
    assert bootstrap_ci(flags, n_resamples=1000).estimate == 0.75
    column = np.arange(20.0).reshape(-1, 1)
    assert bootstrap_ci(column, n_resamples=1000) == bootstrap_ci(np.arange(20.0),
                                                                  n_resamples=1000)
    assert module.FEW_RESAMPLES == 1000
