# -*- coding: utf-8 -*-
import numpy as np

from kudio import trad_enhance, wavelet_low_pass_filter


def _si_snr(est: np.ndarray, ref: np.ndarray) -> float:
    """Scale-invariant SNR (dB) — immune to overall gain differences."""
    est = est - est.mean()
    ref = ref - ref.mean()
    proj = (np.dot(est, ref) / np.dot(ref, ref)) * ref
    noise = est - proj
    return 10 * np.log10(np.dot(proj, proj) / np.dot(noise, noise))


def _noisy_sine(sr=16000, seconds=1.0, lead_in=0.2, noise_scale=0.2, seed=0):
    """Sine with a noise-only lead-in so MCRA can lock onto the noise floor."""
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    clean = 0.5 * np.sin(2 * np.pi * 440 * t)
    clean[:int(sr * lead_in)] = 0.0
    rng = np.random.default_rng(seed)
    noisy = clean + noise_scale * rng.standard_normal(len(t))
    return clean, noisy


def test_trad_enhance_improves_si_snr():
    sr = 16000
    clean, noisy = _noisy_sine(sr)
    enhanced = trad_enhance(noisy, sr)
    assert np.all(np.isfinite(enhanced))

    n = len(enhanced)
    assert _si_snr(enhanced, clean[:n]) > _si_snr(noisy[:n], clean[:n])


def test_wavelet_filter_shape():
    _, noisy = _noisy_sine()
    filtered = wavelet_low_pass_filter(noisy)
    assert len(filtered) == len(noisy)
    assert np.all(np.isfinite(filtered))
