# -*- coding: utf-8 -*-
"""Waveform / spectrogram data augmentation.

Every function is pure (returns a new array) and, where randomness is involved,
accepts a ``seed`` for reproducibility.
"""
from __future__ import annotations

import warnings
from typing import Callable, Optional, Tuple

import librosa
import numpy as np
from scipy.signal import fftconvolve

from kudio.exceptions import FeatureError

__all__ = [
    'time_stretch', 'pitch_shift', 'gain', 'random_gain',
    'normalize', 'normalize_db',
    'add_noise_snr', 'reverb', 'spec_augment',
]


def _each_channel(y: np.ndarray, fn: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """*fn* on a mono clip, or on every channel of a ``(frames, channels)`` one.

    librosa and numpy both read the last axis as time, which on kudio's
    channels-last layout is the channel axis: time_stretch on a stereo second
    came back as one 16000-sample channel, with no error.
    """
    y = np.asarray(y)
    if y.ndim == 1:
        return fn(y)
    if y.ndim != 2:
        raise FeatureError(f"expected (frames,) or (frames, channels), got shape {y.shape}")
    return np.stack([fn(y[:, c]) for c in range(y.shape[1])], axis=1)


def _quietly(fn: Callable, *args, **kwargs):
    """librosa 1.0's time_stretch and pitch_shift hand phase_vocoder its own
    deprecated arguments and warn about it twice a call. Nothing a caller does
    changes that, so it is not passed on."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning,
                                message=r".*deprecated as of 1\.0.*")
        return fn(*args, **kwargs)


def _to_float(y: np.ndarray):
    """Integer audio as float on its own full scale, and what undoes it."""
    y = np.asarray(y)
    if not np.issubdtype(y.dtype, np.integer):
        return y, lambda out: out.astype(y.dtype)
    limits = np.iinfo(y.dtype)
    full = -float(limits.min) if limits.min < 0 else (int(limits.max) + 1) / 2.0
    offset = 0.0 if limits.min < 0 else full

    def back(out):
        # round and clip into range: casting a float back wraps 1.2 round to -0.8
        return np.clip(np.round(out * full + offset), limits.min, limits.max).astype(y.dtype)
    return (y.astype(np.float64) - offset) / full, back


def time_stretch(y: np.ndarray, rate: float) -> np.ndarray:
    """Speed up (``rate>1``) or slow down (``rate<1``) without changing pitch.

    A ``(frames, channels)`` array is stretched channel by channel.
    """
    if rate <= 0:
        raise ValueError(f"rate must be > 0, got {rate}")
    return _each_channel(y, lambda c: _quietly(librosa.effects.time_stretch, c, rate=rate))


def pitch_shift(y: np.ndarray, sr: int, n_steps: float) -> np.ndarray:
    """Shift pitch by *n_steps* semitones without changing duration.

    A ``(frames, channels)`` array is shifted channel by channel.
    """
    return _each_channel(
        y, lambda c: _quietly(librosa.effects.pitch_shift, c, sr=sr, n_steps=n_steps))


def gain(y: np.ndarray, db: float) -> np.ndarray:
    """Apply a fixed gain in decibels.

    Integer audio stays in its type, rounded and clipped at full scale rather
    than wrapped round, which would turn an int16 20000 at +6 dB into -25631.
    """
    x, back = _to_float(y)
    return back(x * (10.0 ** (db / 20.0)))


def normalize(y: np.ndarray, peak: float = 0.99) -> np.ndarray:
    """Scale so the loudest sample sits at *peak*.

    Silence is returned unchanged rather than divided by zero — the guard
    everyone writes inline and someone eventually forgets. For integer audio
    *peak* is a fraction of the type's full scale, as it is for float, and the
    result keeps the input's type.
    """
    if peak <= 0:
        raise ValueError(f"peak must be > 0, got {peak}")
    y = np.asarray(y)
    x, back = _to_float(y)
    current = float(np.max(np.abs(x))) if x.size else 0.0
    if current == 0.0:
        return y.copy()
    return back(x * (peak / current))


def normalize_db(y: np.ndarray, dbfs: float = -1.0) -> np.ndarray:
    """Scale so the peak sits at *dbfs* decibels below full scale.

    ``normalize_db(y, -1.0)`` is ``normalize(y, 10 ** (-1 / 20))``; use
    whichever unit the rest of the pipeline speaks.
    """
    if dbfs > 0:
        raise ValueError(f"dbfs must be <= 0 (full scale), got {dbfs}")
    return normalize(y, peak=10.0 ** (dbfs / 20.0))


def random_gain(y: np.ndarray, db_range: Tuple[float, float] = (-6.0, 6.0),
                seed: Optional[int] = None) -> np.ndarray:
    """Apply a random gain sampled uniformly from *db_range* dB."""
    rng = np.random.default_rng(seed)
    return gain(y, rng.uniform(*db_range))


def add_noise_snr(y: np.ndarray, noise: np.ndarray, snr_db: float,
                  seed: Optional[int] = None) -> np.ndarray:
    """Mix *noise* into *y* at the target signal-to-noise ratio (dB).

    *noise* is tiled/cropped (with a random offset) to match ``len(y)``. Both
    are mono.

    :raises FeatureError: for noise with no variation in the stretch used --
        silence or a constant. Its mean is removed first, so nothing would be
        left to scale, and the result would be *y* back with no noise in it at
        any SNR. Synthesizer refuses the same file for the same reason.
    """
    y, noise = np.asarray(y), np.asarray(noise)
    if y.ndim != 1 or noise.ndim != 1:
        raise FeatureError(f"add_noise_snr mixes mono signals, got shapes "
                           f"{y.shape} and {noise.shape}")
    if not noise.size:
        raise FeatureError("noise is empty")
    rng = np.random.default_rng(seed)
    if len(noise) < len(y):
        noise = np.tile(noise, len(y) // len(noise) + 1)[:len(y)]
    elif len(noise) > len(y):
        off = rng.integers(0, len(noise) - len(y) + 1)
        noise = noise[off:off + len(y)]
    if not np.ptp(noise) > 0:
        raise FeatureError("the noise has no variation (silence or a constant), "
                           "so there is nothing to mix in at any SNR")
    noise = noise - np.mean(noise)
    sig_pwr = np.mean(y ** 2)
    noise_pwr = np.mean(noise ** 2) + 1e-12
    scale = np.sqrt(sig_pwr / (noise_pwr * 10 ** (snr_db / 10)))
    return (y + scale * noise).astype(y.dtype)


def reverb(y: np.ndarray, sr: int, decay: float = 0.3,
           delay_ms: float = 50.0) -> np.ndarray:
    """Cheap synthetic reverb via an exponentially-decaying comb filter: an
    echo every *delay_ms*, each *decay* times the one before. For a room that
    behaves like a room, use :func:`kudio.rir` with :func:`kudio.apply_rir`.

    A ``(frames, channels)`` array gets the same echoes on every channel.
    """
    delay = max(1, int(sr * delay_ms / 1000.0))
    ir_len = delay * 8
    impulse = np.zeros(ir_len, dtype=np.float32)
    t = 0
    amp = 1.0
    while t < ir_len:
        impulse[t] += amp
        amp *= decay
        t += delay
    y = np.asarray(y)
    wet = _each_channel(y, lambda c: fftconvolve(c, impulse)[:len(c)])
    peak = np.max(np.abs(wet)) + 1e-12
    return (wet / peak * np.max(np.abs(y))).astype(y.dtype)


def spec_augment(spec: np.ndarray, n_freq_masks: int = 1, n_time_masks: int = 1,
                 freq_mask_width: int = 8, time_mask_width: int = 16,
                 mask_value: Optional[float] = None,
                 seed: Optional[int] = None) -> np.ndarray:
    """SpecAugment: mask random frequency bands and time spans of a spectrogram.

    *spec* is ``(frames, bins)``, the layout every spectrogram in kudio comes
    in (or squeezable to it). Returns a masked copy.

    :param mask_value: what a masked cell becomes. ``None`` (the default) uses
        the spectrogram's mean, which is SpecAugment's choice: 0 is the mean of
        a standardised spectrogram, but on kudio's log10 power it is a power of
        1, louder than most of the cells it would hide.
    """
    rng = np.random.default_rng(seed)
    out = np.array(spec, copy=True)
    work = out if out.ndim == 2 else out.squeeze()
    if work.ndim != 2:
        raise ValueError(f"spec must be 2-D after squeeze, got shape {spec.shape}")
    n_time, n_freq = work.shape
    fill = float(np.mean(work)) if mask_value is None else mask_value

    for _ in range(n_freq_masks):
        w = int(rng.integers(0, min(freq_mask_width, n_freq) + 1))
        if w:
            f0 = int(rng.integers(0, n_freq - w + 1))
            work[:, f0:f0 + w] = fill
    for _ in range(n_time_masks):
        w = int(rng.integers(0, min(time_mask_width, n_time) + 1))
        if w:
            t0 = int(rng.integers(0, n_time - w + 1))
            work[t0:t0 + w, :] = fill
    return work.reshape(spec.shape)
