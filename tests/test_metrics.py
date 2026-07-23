# -*- coding: utf-8 -*-
import numpy as np

from kudio import segmental_snr, si_sdr, snr


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
    # should not raise despite length mismatch
    assert np.isfinite(snr(a, b))
    assert np.isfinite(si_sdr(a, b))
