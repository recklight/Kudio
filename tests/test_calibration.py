# -*- coding: utf-8 -*-
"""Calibration of a verifier's scores: C_llr, its minimum, the actual DCF and
NIST's primary cost."""
from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import kudio
from kudio import act_dcf, c_primary, cllr, min_cllr, min_dcf

LN2 = math.log(2.0)


def _calibrated(n=4000, m=2.0, share=0.3, seed=0):
    """Targets from N(m, 2m) and non-targets from N(-m, 2m): for those the
    score is the natural-log likelihood ratio."""
    rng = np.random.default_rng(seed)
    labels = rng.random(n) < share
    llrs = np.where(labels, m, -m) + math.sqrt(2 * m) * rng.standard_normal(n)
    return llrs, labels


def _quiet(fn, *args, **kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = fn(*args, **kwargs)
    return value, [str(w.message) for w in caught]


def test_cllr_of_llrs_of_zero_is_one_bit():
    assert cllr(np.zeros(7), [1, 0, 1, 0, 0, 1, 0]) == 1.0


def test_cllr_is_the_formula_term_by_term():
    llrs = np.array([2.0, -1.0, 0.5, 3.0, -4.0])
    labels = np.array([1, 1, 0, 0, 0], dtype=bool)
    tar = np.mean(np.log2(1 + np.exp(-llrs[labels])))
    non = np.mean(np.log2(1 + np.exp(llrs[~labels])))
    assert cllr(llrs, labels) == pytest.approx((tar + non) / 2, rel=1e-15)


def test_a_wrong_certainty_costs_inf_and_a_right_one_nothing():
    labels = [1, 1, 0, 0]
    assert cllr([math.inf, 1.0, -1.0, -math.inf], labels) == pytest.approx(
        (math.log1p(math.exp(-1.0)) / 2 + math.log1p(math.exp(-1.0)) / 2) / (2 * LN2))
    assert cllr([-math.inf, 1.0, -1.0, -2.0], labels) == math.inf


def test_cllr_does_not_overflow_on_large_llrs():
    assert cllr([800.0, -800.0], [1, 0]) == 0.0
    assert cllr([-800.0, 800.0], [1, 0]) == pytest.approx(800.0 / LN2)


def test_min_cllr_of_a_perfect_ranking_is_zero_and_of_no_ranking_one():
    assert min_cllr([0.1, 0.2, 0.9, 1.5], [0, 0, 1, 1]) == 0.0
    assert min_cllr(np.full(6, 0.3), [1, 0, 0, 1, 0, 1]) == pytest.approx(1.0, abs=1e-15)


def test_min_cllr_does_not_depend_on_how_the_scores_are_scaled():
    llrs, labels = _calibrated()
    base = min_cllr(llrs, labels)
    assert min_cllr(np.exp(llrs / 3) * 7 - 2, labels) == pytest.approx(base, abs=1e-14)
    assert base <= cllr(llrs, labels)


def test_min_cllr_of_calibrated_llrs_is_close_to_their_cllr():
    """Drawn from the model they are LLRs of, the scores are calibrated, and
    the best recalibration cannot gain much."""
    llrs, labels = _calibrated(n=200000)
    assert 0 < cllr(llrs, labels) - min_cllr(llrs, labels) < 0.01


def test_min_cllr_is_the_isotonic_fit_with_ties_pooled():
    """The PAV fit pools tied scores, so the fit is the isotonic regression
    of the labels on the scores, which scikit-learn computes the same way."""
    isotonic = pytest.importorskip("sklearn.isotonic")
    rng = np.random.default_rng(3)
    for _ in range(60):
        n = int(rng.integers(5, 300))
        labels = rng.random(n) < rng.uniform(0.2, 0.8)
        if labels.all() or not labels.any():
            continue
        scores = np.round(rng.normal(size=n) + 1.5 * labels, int(rng.integers(0, 2)))
        p = isotonic.IsotonicRegression(y_min=0.0, y_max=1.0).fit_transform(
            scores, labels.astype(float))
        T, N = labels.sum(), (~labels).sum()
        with np.errstate(divide='ignore'):
            llr = np.log(p) - np.log1p(-p) - math.log(T / N)
        want = (np.mean(np.logaddexp(0, -llr[labels]))
                + np.mean(np.logaddexp(0, llr[~labels]))) / (2 * LN2)
        assert min_cllr(scores, labels) == pytest.approx(want, abs=1e-12)


def test_tied_trials_give_the_same_min_cllr_in_any_order():
    scores = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 2.0])
    labels = np.array([1, 0, 0, 1, 0, 1], dtype=bool)
    reordered = [3, 0, 5, 2, 4, 1]
    assert min_cllr(scores, labels) == min_cllr(scores[reordered], labels[reordered])


def test_act_dcf_decides_at_log_beta_and_accepts_the_threshold_itself():
    beta = 0.99 / 0.01                     # p_target 0.01, unit costs
    t = math.log(beta)
    llrs = np.array([t, t + 1, t - 1, t - 2, t, t + 3])
    labels = np.array([1, 1, 1, 0, 0, 0], dtype=bool)
    # one miss of three targets, two false alarms of three non-targets
    want = (0.01 * (1 / 3) + 0.99 * (2 / 3)) / 0.01
    assert act_dcf(llrs, labels, p_target=0.01) == pytest.approx(want, rel=1e-15)


def test_act_dcf_is_never_below_min_dcf():
    llrs, labels = _calibrated()
    for p in (0.001, 0.01, 0.05, 0.5):
        assert act_dcf(llrs, labels, p_target=p) >= min_dcf(llrs, labels, p_target=p) - 1e-15


def test_act_dcf_of_shifted_llrs_shows_the_miscalibration():
    llrs, labels = _calibrated()
    calibrated = act_dcf(llrs, labels, p_target=0.01)
    assert act_dcf(llrs + 3.0, labels, p_target=0.01) > calibrated + 0.2
    assert min_dcf(llrs + 3.0, labels, p_target=0.01) == min_dcf(llrs, labels, p_target=0.01)


def test_c_primary_is_the_mean_of_the_two_costs():
    llrs, labels = _calibrated()
    assert c_primary(llrs, labels) == (act_dcf(llrs, labels, p_target=0.01)
                                       + act_dcf(llrs, labels, p_target=0.005)) / 2
    assert c_primary(llrs, labels, minimum=True) == (
        min_dcf(llrs, labels, p_target=0.01) + min_dcf(llrs, labels, p_target=0.005)) / 2
    assert kudio.C_PRIMARY_PRIORS == (0.01, 0.005)


@pytest.mark.parametrize("fn", [cllr, min_cllr, lambda s, l: act_dcf(s, l, p_target=0.01),
                                c_primary], ids=['cllr', 'min_cllr', 'act_dcf', 'c_primary'])
def test_a_list_of_one_kind_of_trial_is_nan_with_one_warning(fn):
    value, said = _quiet(fn, [0.5, 1.0, 2.0], [1, 1, 1])
    assert math.isnan(value)
    assert len(said) == 1 and said[0].endswith("; scored nan"), said
    assert "no non-target trials" in said[0]


@pytest.mark.parametrize("name", ['cllr', 'min_cllr', 'act_dcf', 'c_primary'])
def test_malformed_lists_are_refused_by_name(name):
    fn = getattr(kudio, name)
    kwargs = {'p_target': 0.01} if name == 'act_dcf' else {}
    for llrs, labels in (([0.5, math.nan], [1, 0]), ([0.5, 1.0], [1, 2]),
                         ([0.5, 1.0, 2.0], [1, 0]),
                         (np.ma.masked_array([0.5, 1.0], mask=[0, 1]), [1, 0])):
        with pytest.raises(ValueError, match=rf"^{name}\(\): "):
            fn(llrs, labels, **kwargs)


def test_cost_parameters_are_refused_by_the_name_of_the_function_given_them():
    with pytest.raises(ValueError, match=r"^act_dcf\(\): p_target has to be"):
        act_dcf([0.5, 1.0], [1, 0], p_target=1.5)
    with pytest.raises(ValueError, match=r"^c_primary\(\): minimum has to be"):
        c_primary([0.5, 1.0], [1, 0], minimum='yes')
