# -*- coding: utf-8 -*-
"""kudio's BSS Eval held to mir_eval's bss_eval_sources, bit for bit: one source and
several, with and without the search for the permutation, clips shorter and longer
than the 512-tap filter, float32 and float64, references that make the normal
equations singular, and score(), bss_eval() and sdr_improvement() on top.

kudio does not need mir_eval and a plain install does not bring it, so on most
machines every test here skips. ``KUDIO_REQUIRE_MIR_EVAL=1`` is how a run says that
it must not: see ``conftest.py``.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import kudio
from kudio.core import _bss

mir_eval = pytest.importorskip("mir_eval")
import mir_eval.separation  # noqa: E402


@pytest.fixture(autouse=True)
def theirs_as_written(monkeypatch):
    """mir_eval 0.8.2 catches ``np.linalg.linalg.LinAlgError``, a name numpy 2.5 has
    not got, so where the normal equations are singular it raises AttributeError
    instead of going on by least squares. The name put back, it does what it says."""
    monkeypatch.setattr(np.linalg, "linalg", np.linalg, raising=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)     # 0.8 says 0.9 drops it
        yield


def _theirs(refs, ests, permute=True):
    return mir_eval.separation.bss_eval_sources(refs, ests, compute_permutation=permute)


def _same(got, want, what):
    for a, b, part in zip(got, want, ("sdr", "sir", "sar", "permutation")):
        a, b = np.asarray(a), np.asarray(b)
        assert a.dtype == b.dtype and a.shape == b.shape, f"{what} {part}: {a.dtype}{a.shape}"
        assert np.array_equal(a, b), f"{what} {part}: {a} against mir_eval's {b}"


@pytest.mark.parametrize("k, lengths", [(1, (1, 7, 511, 512, 513, 3000)), (2, (5, 300, 1500)),
                                        (3, (600,))])
def test_bss_eval_sources_is_mir_evals(k, lengths):
    rng = np.random.default_rng(k)
    for n in lengths:
        for dtype in (np.float32, np.float64):
            refs = rng.standard_normal((k, n)).astype(dtype)
            ests = (0.8 * refs[::-1] + 0.3 * rng.standard_normal((k, n))).astype(dtype)
            for permute in (True, False) if k > 1 else (True,):
                _same(_bss.bss_eval_sources(refs, ests, permute), _theirs(refs, ests, permute),
                      f"{k} source(s), {n} samples, {dtype.__name__}, permute={permute}")


def test_singular_normal_equations_go_on_by_least_squares(monkeypatch):
    """Two references of two samples, one twice the other: np.linalg.solve finds the
    Gram matrix of their delays singular, and both go on by least squares."""
    refs = np.array([[1.0, 0.0], [2.0, 0.0]])
    ests = np.array([[0.9, 0.1], [2.2, -0.3]])
    singular = []
    solve = np.linalg.solve

    def watched(G, D):
        try:
            return solve(G, D)
        except np.linalg.LinAlgError:
            singular.append(G.shape)
            raise
    monkeypatch.setattr(np.linalg, "solve", watched)
    _same(_bss.bss_eval_sources(refs, ests), _theirs(refs, ests), "singular")
    assert singular


def test_the_metrics_on_top_are_mir_evals_numbers():
    rng = np.random.default_rng(9)
    refs = rng.standard_normal((2, 4000))
    mixture = refs.sum(axis=0)
    ests = np.vstack([refs[0] + 0.3 * refs[1], refs[1] + 0.2 * rng.standard_normal(4000)])
    score = kudio.bss_eval(refs, ests, mixture)
    sdr, sir, sar, perm = _theirs(refs, ests)
    assert score.sdr == tuple(sdr) and score.sir == tuple(sir) and score.sar == tuple(sar)
    assert score.permutation == tuple(perm)
    of_mixture, *_ = _theirs(refs, np.tile(mixture, (2, 1)), False)
    assert score.sdri == tuple(a - b for a, b in zip(score.sdr, of_mixture))
    want, *_ = _theirs(refs[:1], ests[:1])
    assert kudio.score(refs[0], ests[0], 16000, ['sdr'])['sdr'] == float(want[0])
    mix, *_ = _theirs(refs[:1], mixture[np.newaxis])
    assert kudio.sdr_improvement(refs[0], ests[0], mixture) == float(want[0]) - float(mix[0])
