# -*- coding: utf-8 -*-
import warnings

import numpy as np
import pytest

import kudio
from kudio import segmental_snr, si_sdr, snr
from kudio.core import evaluator


def test_snr_perfect_is_high():
    ref = np.random.default_rng(0).standard_normal(16000)
    assert snr(ref, ref) > 100          # identical -> very high SNR


def test_snr_decreases_with_noise():
    rng = np.random.default_rng(0)
    ref = rng.standard_normal(16000)
    low = ref + 0.1 * rng.standard_normal(16000)
    high = ref + 1.0 * rng.standard_normal(16000)
    assert snr(ref, low) > snr(ref, high)


def test_si_sdr_perfect_is_high():
    ref = np.random.default_rng(0).standard_normal(16000)
    assert si_sdr(ref, ref) > 100


def test_si_sdr_scale_invariant():
    rng = np.random.default_rng(0)
    ref = rng.standard_normal(16000)
    deg = ref + 0.3 * rng.standard_normal(16000)   # non-degenerate estimate
    # scaling the estimate must not change SI-SDR
    assert abs(si_sdr(ref, deg) - si_sdr(ref, 5.0 * deg)) < 1e-4


def test_segmental_snr_bounds():
    rng = np.random.default_rng(0)
    ref = rng.standard_normal(16000)
    deg = ref + 0.2 * rng.standard_normal(16000)
    val = segmental_snr(ref, deg)
    assert -10 <= val <= 35


def test_metrics_align_lengths():
    a = np.ones(1000)
    b = np.ones(800)
    # should not raise despite length mismatch -- 20% short is framing slop,
    # not a different recording. It is still said out loud, so catch the
    # warning here rather than leaving it in the suite's output: one nobody
    # can act on is one nobody reads.
    with pytest.warns(UserWarning, match="only the first 800 samples"):
        assert np.isfinite(snr(a, b))
    with pytest.warns(UserWarning, match="only the first 800 samples"):
        assert np.isfinite(si_sdr(a, b))


def _pair(deg_length, total=16000):
    ref = np.random.default_rng(0).standard_normal(total)
    return ref, ref[:deg_length]


def test_a_truncated_pair_says_so():
    """100 samples short of 16000 is the framing slop a pipeline really
    produces, so it still scores -- but silently scoring it is how a tail
    nobody looked at ends up inside a published number."""
    ref, deg = _pair(15900)
    with pytest.warns(UserWarning) as caught:
        value = snr(ref, deg)
    assert np.isfinite(value)
    message = str(caught[0].message)
    assert "16000" in message and "15900" in message
    assert "15900 samples are scored" in message


def test_a_pair_that_barely_overlaps_is_refused():
    """The one that caught me out: 1000 samples against 16000 came back as a
    bit-perfect copy, because the 15000 that disagree were never compared."""
    ref, deg = _pair(1000)
    with pytest.raises(ValueError) as excinfo:
        snr(ref, deg)
    message = str(excinfo.value)
    assert "16000" in message and "1000" in message
    # the exact count leads, because a rounded percentage reads as a
    # contradiction right at the gate: 15999 of 16000 is not "100%" when one
    # sample was in fact scored
    assert "15000 of the longer one's 16000 samples" in message
    assert "93.8%" in message
    assert "kudio.align" in message


@pytest.mark.parametrize("call", [
    lambda ref, deg: snr(ref, deg),
    lambda ref, deg: si_sdr(ref, deg),
    lambda ref, deg: segmental_snr(ref, deg),
    lambda ref, deg: kudio.sdi(ref, deg),
    lambda ref, deg: kudio.score(ref, deg, 16000, ["snr"]),
    lambda ref, deg: evaluator.eval_metrics(16000, ref, deg),
])
def test_every_route_into_the_metrics_refuses_it(call):
    """snr is not the only way in; they all end up in the same truncation."""
    ref, deg = _pair(1000)
    with pytest.raises(ValueError, match="unmeasured"):
        call(ref, deg)


def test_the_limit_can_be_moved():
    """A caller who really does want the overlap scored can say so, rather
    than being told to re-implement the metric."""
    ref, deg = _pair(1000)
    with pytest.warns(UserWarning):
        cut_ref, cut_deg = evaluator._align(ref, deg, tolerance=0.99)
    assert len(cut_ref) == len(cut_deg) == 1000


def test_the_module_limit_is_what_the_metrics_read(monkeypatch):
    ref, deg = _pair(1000)
    monkeypatch.setattr(evaluator, "MAX_LENGTH_MISMATCH", 0.99)
    with pytest.warns(UserWarning):
        assert np.isfinite(snr(ref, deg))
    monkeypatch.setattr(evaluator, "MAX_LENGTH_MISMATCH", 0.001)
    with pytest.raises(ValueError):
        snr(ref, deg)


def test_matching_lengths_stay_quiet():
    """Whatever else happens, the ordinary case must not start warning."""
    ref = np.random.default_rng(0).standard_normal(16000)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert snr(ref, ref) > 100
        assert np.isfinite(si_sdr(ref, ref + 0.1))
