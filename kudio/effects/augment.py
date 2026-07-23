# -*- coding: utf-8 -*-
"""Waveform / spectrogram data augmentation.

Every function is pure (returns a new array) and, where randomness is involved,
accepts a ``seed`` for reproducibility.
"""
from __future__ import annotations

from typing import Optional, Tuple

import librosa
import numpy as np

__all__ = [
    'time_stretch', 'pitch_shift', 'gain', 'random_gain',
    'add_noise_snr', 'reverb', 'spec_augment',
]


def time_stretch(y: np.ndarray, rate: float) -> np.ndarray:
    """Speed up (``rate>1``) or slow down (``rate<1``) without changing pitch."""
    if rate <= 0:
        raise ValueError(f"rate must be > 0, got {rate}")
    return librosa.effects.time_stretch(y, rate=rate)


def pitch_shift(y: np.ndarray, sr: int, n_steps: float) -> np.ndarray:
    """Shift pitch by *n_steps* semitones without changing duration."""
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)


def gain(y: np.ndarray, db: float) -> np.ndarray:
    """Apply a fixed gain in decibels."""
    return (y * (10.0 ** (db / 20.0))).astype(y.dtype)


def random_gain(y: np.ndarray, db_range: Tuple[float, float] = (-6.0, 6.0),
                seed: Optional[int] = None) -> np.ndarray:
    """Apply a random gain sampled uniformly from *db_range* dB."""
    rng = np.random.default_rng(seed)
    return gain(y, rng.uniform(*db_range))


def add_noise_snr(y: np.ndarray, noise: np.ndarray, snr_db: float,
                  seed: Optional[int] = None) -> np.ndarray:
    """Mix *noise* into *y* at the target signal-to-noise ratio (dB).

    *noise* is tiled/cropped (with a random offset) to match ``len(y)``.
    """
    rng = np.random.default_rng(seed)
    if len(noise) < len(y):
        noise = np.tile(noise, len(y) // len(noise) + 1)[:len(y)]
    elif len(noise) > len(y):
        off = rng.integers(0, len(noise) - len(y) + 1)
        noise = noise[off:off + len(y)]
    noise = noise - np.mean(noise)
    sig_pwr = np.mean(y ** 2)
    noise_pwr = np.mean(noise ** 2) + 1e-12
    scale = np.sqrt(sig_pwr / (noise_pwr * 10 ** (snr_db / 10)))
    return (y + scale * noise).astype(y.dtype)


def reverb(y: np.ndarray, sr: int, decay: float = 0.3,
           delay_ms: float = 50.0) -> np.ndarray:
    """Cheap synthetic reverb via an exponentially-decaying comb filter."""
    delay = max(1, int(sr * delay_ms / 1000.0))
    ir_len = delay * 8
    impulse = np.zeros(ir_len, dtype=np.float32)
    t = 0
    amp = 1.0
    while t < ir_len:
        impulse[t] += amp
        amp *= decay
        t += delay
    wet = np.convolve(y, impulse)[:len(y)]
    peak = np.max(np.abs(wet)) + 1e-12
    return (wet / peak * np.max(np.abs(y))).astype(y.dtype)


def spec_augment(spec: np.ndarray, n_freq_masks: int = 1, n_time_masks: int = 1,
                 freq_mask_width: int = 8, time_mask_width: int = 16,
                 mask_value: float = 0.0, seed: Optional[int] = None) -> np.ndarray:
    """SpecAugment: zero out random frequency and time bands of a spectrogram.

    *spec* is ``(freq, time)`` (or squeezable to 2-D). Returns a masked copy.
    """
    rng = np.random.default_rng(seed)
    out = np.array(spec, copy=True)
    work = out if out.ndim == 2 else out.squeeze()
    if work.ndim != 2:
        raise ValueError(f"spec must be 2-D after squeeze, got shape {spec.shape}")
    n_freq, n_time = work.shape

    for _ in range(n_freq_masks):
        w = int(rng.integers(0, min(freq_mask_width, n_freq) + 1))
        if w:
            f0 = int(rng.integers(0, n_freq - w + 1))
            work[f0:f0 + w, :] = mask_value
    for _ in range(n_time_masks):
        w = int(rng.integers(0, min(time_mask_width, n_time) + 1))
        if w:
            t0 = int(rng.integers(0, n_time - w + 1))
            work[:, t0:t0 + w] = mask_value
    return work.reshape(spec.shape)
