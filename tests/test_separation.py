# -*- coding: utf-8 -*-
"""SI-SDR and BSS Eval over several sources, matched to their estimates, and
the improvement over the mixture."""
from __future__ import annotations

import itertools
import math
import warnings

import numpy as np
import pytest

import kudio
from kudio import bss_eval, pit_si_sdr, sdr_improvement, si_sdr_improvement

from conftest import make_speech

SR = 8000


def _sources(k, n=2 * SR, seed=0):
    """k talkers: harmonic bursts at different pitches, each with its own
    noise floor."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / SR
    out = []
    for i in range(k):
        f0 = 110.0 + 37.0 * i
        y = sum(np.sin(2 * np.pi * f0 * h * t + rng.uniform(0, 6)) / h for h in range(1, 9))
        y *= 0.5 + 0.5 * np.sin(2 * np.pi * (2.0 + 0.7 * i) * t + rng.uniform(0, 6))
        out.append(0.1 * y + 0.003 * rng.standard_normal(n))
    return np.array(out)


def _separated(sources, leak=0.15, noise=0.02, seed=1):
    """What a separator might return: each source with some of the others
    leaking in and noise added."""
    rng = np.random.default_rng(seed)
    k = len(sources)
    mix = np.eye(k) + leak * (np.ones((k, k)) - np.eye(k)) * rng.uniform(0.2, 1.0, (k, k))
    return mix @ sources + noise * rng.standard_normal(sources.shape)


def _quiet(fn, *args, **kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = fn(*args, **kwargs)
    return value, [str(w.message) for w in caught]


def test_si_sdr_improvement_is_the_difference_of_two_si_sdrs():
    refs = _sources(2)
    mixture = refs.sum(axis=0)
    deg = _separated(refs)[0]
    assert si_sdr_improvement(refs[0], deg, mixture) == (
        kudio.si_sdr(refs[0], deg) - kudio.si_sdr(refs[0], mixture))


@pytest.mark.parametrize("k", [2, 3, 4])
@pytest.mark.parametrize("seed", range(5))
def test_pit_finds_the_assignment_an_exhaustive_search_finds(k, seed):
    refs = _sources(k, seed=seed)
    ests = _separated(refs, leak=0.6, noise=0.05, seed=seed + 10)
    shuffle = np.random.default_rng(seed).permutation(k)
    ests = ests[shuffle]
    score = pit_si_sdr(refs, ests)
    table = np.array([[kudio.si_sdr(r, e) for e in ests] for r in refs])
    best = max(itertools.permutations(range(k)),
               key=lambda p: sum(table[i, p[i]] for i in range(k)))
    assert np.sum(score.si_sdr) == pytest.approx(sum(table[i, best[i]] for i in range(k)),
                                                 rel=1e-12)
    for i in range(k):
        assert score.si_sdr[i] == table[i, score.permutation[i]]
    assert sorted(score.permutation) == list(range(k))


def test_a_shuffled_separation_is_put_back_in_order():
    refs = _sources(3)
    ests = _separated(refs)
    order = (2, 0, 1)
    score = pit_si_sdr(refs, ests[list(order)])
    assert score.permutation == (1, 2, 0)
    straight = pit_si_sdr(refs, ests)
    assert straight.permutation == (0, 1, 2)
    assert score.si_sdr == straight.si_sdr


def test_the_improvement_subtracts_the_mixture_source_by_source():
    refs = _sources(2)
    mixture = refs.sum(axis=0)
    ests = _separated(refs)
    score = pit_si_sdr(refs, ests, mixture)
    for i in range(2):
        assert score.si_sdri[i] == score.si_sdr[i] - kudio.si_sdr(refs[i], mixture)
    assert pit_si_sdr(refs, ests).si_sdri is None


def test_one_source_as_a_vector_is_si_sdr_itself():
    refs = _sources(1)
    deg = _separated(refs)[0]
    score = pit_si_sdr(refs[0], deg)
    assert score == ((kudio.si_sdr(refs[0], deg),), None, (0,))


def test_samples_by_sources_is_refused_with_the_way_out():
    refs = _sources(2)
    with pytest.raises(ValueError, match=r"^pit_si_sdr\(\): refs has shape \(16000, 2\).*transposed"):
        pit_si_sdr(refs.T, refs)


def test_unequal_numbers_of_sources_and_estimates_are_refused():
    refs = _sources(3)
    with pytest.raises(ValueError, match=r"^pit_si_sdr\(\): 3 reference source\(s\) and 2"):
        pit_si_sdr(refs, refs[:2])


def test_ragged_rows_are_refused_by_name():
    with pytest.raises(ValueError, match=r"^pit_si_sdr\(\): refs is not an array"):
        pit_si_sdr([[0.1, 0.2], [0.1]], [[0.1, 0.2], [0.1, 0.3]])


def test_a_constant_reference_leaves_nothing_to_score():
    refs = _sources(2)
    refs[1] = 0.0
    score, said = _quiet(pit_si_sdr, refs, _separated(_sources(2)))
    assert all(math.isnan(v) for v in score.si_sdr)
    assert len(said) == 1 and said[0].startswith("pit_si_sdr(): refs[1] is digital silence")
    assert said[0].endswith("; scored nan")


def test_a_constant_estimate_scores_zero_with_one_warning():
    refs = _sources(2)
    ests = _separated(refs)
    ests[0] = 0.0
    score, said = _quiet(pit_si_sdr, refs, ests)
    assert 0.0 in score.si_sdr
    assert len(said) == 1 and "ests[0] is constant" in said[0], said


def test_lengths_are_cut_to_the_shortest_with_one_warning():
    refs = _sources(2)
    ests = _separated(refs)[:, :-100]
    score, said = _quiet(pit_si_sdr, refs, ests, refs.sum(axis=0))
    assert len(said) == 1 and "only the first 15900" in said[0], said
    assert score == pit_si_sdr(refs[:, :-100], ests, refs.sum(axis=0)[:-100])
    with pytest.raises(ValueError, match=r"^pit_si_sdr\(\): .*past the 50% limit"):
        pit_si_sdr(refs, ests[:, :4000])


def test_too_loud_to_square_is_nan_with_one_warning():
    refs = _sources(2) * 1e160
    score, said = _quiet(pit_si_sdr, refs, refs * 0.9)
    assert all(math.isnan(v) for v in score.si_sdr)
    assert len(said) == 1 and "too loud to square" in said[0], said


# --- BSS Eval, against mir_eval itself --------------------------------------------

@pytest.fixture
def mir_eval():
    module = pytest.importorskip("mir_eval")
    return module


def _bss(mir_eval, refs, ests, permute=True):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return mir_eval.separation.bss_eval_sources(refs, ests, compute_permutation=permute)


@pytest.mark.parametrize("k", [2, 3])
def test_bss_eval_gives_mir_evals_numbers(mir_eval, k):
    refs = _sources(k, n=SR)
    ests = _separated(refs, leak=0.3)[::-1]
    score = bss_eval(refs, ests)
    sdr, sir, sar, perm = _bss(mir_eval, refs, ests)
    assert score.sdr == tuple(sdr) and score.sir == tuple(sir) and score.sar == tuple(sar)
    assert score.permutation == tuple(perm)


def test_bss_eval_takes_the_permutation_pit_found(mir_eval):
    refs = _sources(2, n=SR)
    ests = _separated(refs, leak=0.3)[::-1]
    order = pit_si_sdr(refs, ests).permutation
    score = bss_eval(refs, ests, permutation=order)
    sdr, sir, sar, _ = _bss(mir_eval, refs, ests[list(order)], permute=False)
    assert score.sdr == tuple(sdr) and score.permutation == order
    with pytest.raises(ValueError, match=r"^bss_eval\(\): permutation has to hold"):
        bss_eval(refs, ests, permutation=(0, 0))


def test_sdri_subtracts_the_mixture_as_every_estimate(mir_eval):
    refs = _sources(2, n=SR)
    mixture = refs.sum(axis=0)
    ests = _separated(refs, leak=0.3)
    score = bss_eval(refs, ests, mixture)
    of_mixture, *_ = _bss(mir_eval, refs, np.tile(mixture, (2, 1)), permute=False)
    assert score.sdri == tuple(a - b for a, b in zip(score.sdr, of_mixture))


def test_sdr_improvement_is_the_difference_of_two_sdrs_from_score():
    ref = make_speech(SR, 6.0, seed=0).astype(np.float64)[SR:3 * SR]
    noise = np.random.default_rng(2).standard_normal(ref.size)
    mixture = ref + 0.3 * noise
    deg = ref + 0.05 * noise
    expected = (kudio.score(ref, deg, SR, ['sdr'])['sdr']
                - kudio.score(ref, mixture, SR, ['sdr'])['sdr'])
    assert sdr_improvement(ref, deg, mixture) == expected


def test_singular_normal_equations_are_solved_by_least_squares():
    """Two references of two samples, one twice the other, make the normal equations
    singular; BSS Eval goes on by least squares and both estimates score."""
    score = bss_eval(np.array([[1.0, 0.0], [2.0, 0.0]]), np.array([[0.9, 0.1], [2.2, -0.3]]))
    assert all(np.isfinite(v) for v in score.sdr)


def test_one_source_has_no_interference():
    refs = _sources(1, n=SR)
    score = bss_eval(refs, _separated(refs))
    assert score.sir == (math.inf,)


def test_a_silent_estimate_is_nan_with_one_warning():
    refs = _sources(2, n=SR)
    ests = _separated(refs)
    ests[1] = 0.0
    score, said = _quiet(bss_eval, refs, ests)
    assert all(math.isnan(v) for v in score.sdr)
    assert len(said) == 1 and said[0].startswith("bss_eval(): ests[1] is digital silence")
