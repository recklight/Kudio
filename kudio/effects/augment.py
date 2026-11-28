# -*- coding: utf-8 -*-
"""Waveform / spectrogram data augmentation.

Every function is pure (returns a new array) and, where randomness is involved,
accepts a ``seed`` for reproducibility.
"""
from __future__ import annotations

import math
from typing import Callable, Optional, Tuple

import numpy as np

from kudio._numbers import generator, real, whole
from kudio._samples import channels_last, finite, real_samples, to_float
from kudio.core._policy import sample_rate, warn
from kudio.effects import _vocoder
from kudio.exceptions import FeatureError

__all__ = [
    'time_stretch', 'pitch_shift', 'gain', 'random_gain',
    'normalize', 'normalize_db',
    'add_noise_snr', 'reverb', 'spec_augment',
]


def _each_channel(y: np.ndarray, fn: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """*fn* on a mono clip, or on every channel of a ``(frames, channels)`` one."""
    y = np.asarray(y)
    if y.ndim == 1:
        return fn(y)
    if y.ndim != 2 or not y.shape[1]:
        raise FeatureError(f"expected (frames,) or (frames, channels), got shape {y.shape}")
    return np.stack([fn(y[:, c]) for c in range(y.shape[1])], axis=1)


def _read(y, name: str):
    """*y* in kudio's layout and as float, and what puts a result back."""
    return to_float(channels_last(y, name, FeatureError), name, FeatureError)


def _vocoded(y, name: str, fn: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """*fn* on every channel of *y* that has samples in it, read and put back as
    :func:`_read` does: finite ones, and enough for a frame of the phase vocoder or
    a warning."""
    x, back = _read(y, name)
    x = finite(np.asarray(x), name, FeatureError)
    if x.ndim in (1, 2) and x.size and len(x) < _vocoder.SIZE:
        warn(f"{name}(): the clip has {len(x)} sample(s), fewer than one frame of "
             f"{_vocoder.SIZE}; what it works on is mostly the zeros around it")
    out = _each_channel(x, lambda c: fn(c) if len(c) else c)
    if not np.isfinite(out).all():
        raise FeatureError(f"{name}(): the samples go as high as {np.max(np.abs(x)):.3g}, and "
                           f"what comes of them goes past what {out.dtype} holds")
    return back(out)


def time_stretch(y: np.ndarray, rate: float) -> np.ndarray:
    """Speed up (``rate>1``) or slow down (``rate<1``) without changing pitch.

    A phase vocoder, in frames of 2048 samples. *n* samples come back as
    ``round(n / rate)``, the nearest whole number. Tones keep their frequency;
    a click is smeared over a frame, and noise comes back about 3 dB down.

    A ``(frames, channels)`` array is stretched channel by channel, and
    integer audio comes back in its own type, as :func:`gain` does it.

    :raises FeatureError: for a *rate* that is not a finite number above 0, and
        for samples that are not finite.
    """
    number = real(rate)
    if number is None or number <= 0:
        raise FeatureError(f"time_stretch(): rate has to be a finite number above 0, got {rate!r}")

    def stretch(c):
        if number == 1:
            return c
        length = len(c) / number
        if not math.isfinite(length):
            raise FeatureError(f"time_stretch(): {len(c)} sample(s) at a rate of {number:g} are "
                               f"more than there can be")
        return _vocoder.stretch(c, number, round(length))

    return _vocoded(y, 'time_stretch', stretch)


def pitch_shift(y: np.ndarray, sr: int, n_steps: float) -> np.ndarray:
    """Shift pitch by *n_steps* semitones without changing duration: the clip
    stretched as :func:`time_stretch` does it, and resampled to its length.

    A ``(frames, channels)`` array is shifted channel by channel, and integer
    audio comes back in its own type, as :func:`gain` does it.

    :param n_steps: up, or down below 0, and not necessarily whole: 0.5 is a
        quarter tone. As far as 120, ten octaves.
    :raises FeatureError: for *n_steps* that are not a number from -120 to 120, and
        for samples that are not finite.
    :raises ValueError: for an *sr* that is not a positive number.
    """
    from kudio.core.io import _at_rate

    sr = sample_rate(sr, 'pitch_shift')
    steps = real(n_steps)
    if steps is None or abs(steps) > 120:
        raise FeatureError(f"pitch_shift(): n_steps has to be a number of semitones from -120 to "
                           f"120, got {n_steps!r}")
    rate = 2.0 ** (-steps / 12)

    def shift(c):
        if not steps:
            return c
        # up to a whole number, for the resampler to have the whole clip's worth
        stretched = _vocoder.stretch(c, rate, math.ceil(len(c) / rate))
        return _at_rate(stretched, sr / rate, sr, 'soxr_hq')[:len(c)]

    return _vocoded(y, 'pitch_shift', shift)


def gain(y: np.ndarray, db: float) -> np.ndarray:
    """Apply a fixed gain in decibels.

    Integer audio stays in its type, rounded and clipped at full scale rather
    than wrapped round, which would turn an int16 20000 at +6 dB into -25631.
    """
    factor = _factor(db)
    if factor is None:
        raise FeatureError(f"gain(): db has to be a finite number of decibels, got {db!r}")
    x, back = _read(y, 'gain')
    return back(x * factor)


def _factor(db, per: float = 20.0) -> Optional[float]:
    """What *db* decibels multiply an amplitude by, or at 10 *per* decade a power;
    None for what is not a finite number of them, or makes a factor past float64."""
    if real(db) is None:
        return None
    try:
        with np.errstate(over='ignore', under='ignore'):
            factor = 10.0 ** (db / per)
    except OverflowError:
        return None
    return factor if np.isfinite(factor) else None


def normalize(y: np.ndarray, peak: float = 0.99) -> np.ndarray:
    """Scale so the loudest sample sits at *peak*.

    Silence is returned unchanged rather than divided by zero — the guard
    everyone writes inline and someone eventually forgets. For integer audio
    *peak* is a fraction of the type's full scale, as it is for float, and the
    result keeps the input's type.

    :raises FeatureError: for a clip with NaN or inf in it. One of them was the
        peak, and the rest of the clip NaN or 0.
    """
    if real(peak) is None or peak <= 0:
        raise ValueError(f"peak must be > 0, a finite number, got {peak!r}")
    y = channels_last(y, 'normalize', FeatureError)
    x, back = _read(y, 'normalize')
    finite(np.asarray(x), 'normalize', FeatureError)
    current = float(np.max(np.abs(x))) if x.size else 0.0
    if current == 0.0:
        return y.copy()
    return back(x * (peak / current))


def normalize_db(y: np.ndarray, dbfs: float = -1.0) -> np.ndarray:
    """Scale so the peak sits at *dbfs* decibels below full scale.

    ``normalize_db(y, -1.0)`` is ``normalize(y, 10 ** (-1 / 20))``; use
    whichever unit the rest of the pipeline speaks.
    """
    factor = _factor(dbfs)
    if factor is None or dbfs > 0:
        raise ValueError(f"dbfs must be <= 0 (full scale), a finite number, got {dbfs!r}")
    return normalize(y, peak=factor)


def random_gain(y: np.ndarray, db_range: Tuple[float, float] = (-6.0, 6.0),
                seed: Optional[int] = None) -> np.ndarray:
    """Apply a random gain sampled uniformly from *db_range* dB, lowest first.

    :raises FeatureError: for a range that is not two finite numbers, lowest
        first, and as :func:`gain` does.
    """
    rng = generator(seed, 'random_gain', FeatureError)
    try:
        low, high = (real(v) for v in db_range)
    except (TypeError, ValueError):
        low = high = None
    if low is None or high is None or low > high:
        raise FeatureError(f"random_gain(): db_range is two finite numbers of dB, lowest first, "
                           f"got {db_range!r}")
    return gain(y, rng.uniform(low, high))


def add_noise_snr(y: np.ndarray, noise: np.ndarray, snr_db: float,
                  seed: Optional[int] = None) -> np.ndarray:
    """Mix *noise* into *y* at the target signal-to-noise ratio (dB).

    *noise* is tiled/cropped (with a random offset) to match ``len(y)``. Both
    are mono.

    :raises FeatureError: for noise with no variation in the stretch used --
        silence or a constant. Its mean is removed first, so nothing would be
        left to scale, and the result would be *y* back with no noise in it at
        any SNR. Synthesizer refuses the same file for the same reason. And for
        NaN or inf in either clip, or an *snr_db* that is not a finite number.
    """
    ratio = _factor(snr_db, per=10)
    if not ratio:
        raise FeatureError(f"add_noise_snr(): snr_db has to be a finite number of decibels, got "
                           f"{snr_db!r}")
    # the noise as the numbers it is: its mean goes and its level is set, so
    # its scale never comes into it
    y = channels_last(real_samples(y, 'add_noise_snr', FeatureError),
                      'add_noise_snr', FeatureError)
    noise = channels_last(real_samples(noise, 'add_noise_snr', FeatureError),
                          'add_noise_snr', FeatureError)
    y, noise = (a[:, 0] if a.ndim == 2 and a.shape[1] == 1 else a for a in (y, noise))
    if y.ndim != 1 or noise.ndim != 1:
        raise FeatureError(f"add_noise_snr mixes mono signals, got shapes "
                           f"{y.shape} and {noise.shape}")
    if not noise.size:
        raise FeatureError("noise is empty")
    finite(y, 'add_noise_snr', FeatureError)
    finite(noise, 'add_noise_snr', FeatureError)
    if not y.size:
        return y.copy()
    rng = generator(seed, 'add_noise_snr', FeatureError)
    if len(noise) < len(y):
        noise = np.tile(noise, len(y) // len(noise) + 1)[:len(y)]
    elif len(noise) > len(y):
        off = rng.integers(0, len(noise) - len(y) + 1)
        noise = noise[off:off + len(y)]
    # integers in float before any arithmetic: int16 squared wraps round in
    # int16, and a quarter-scale int16 tone mixed at 10 dB came out at 55.9
    if np.issubdtype(noise.dtype, np.integer):
        noise = noise.astype(np.float64)
    if not np.ptp(noise) > 0:
        raise FeatureError("the noise has no variation (silence or a constant), "
                           "so there is nothing to mix in at any SNR")
    x, back = to_float(y, 'add_noise_snr', FeatureError)
    noise = noise - np.mean(noise)
    sig_pwr = np.mean(x ** 2)
    noise_pwr = np.mean(noise ** 2) + 1e-12
    scale = np.sqrt(sig_pwr / (noise_pwr * ratio))
    return back(x + scale * noise)


def reverb(y: np.ndarray, sr: int, decay: float = 0.3,
           delay_ms: float = 50.0) -> np.ndarray:
    """Cheap synthetic reverb via an exponentially-decaying comb filter: an
    echo every *delay_ms*, each *decay* times the one before. For a room that
    behaves like a room, use :func:`kudio.rir` with :func:`kudio.apply_rir`.

    A ``(frames, channels)`` array gets the same echoes on every channel.

    :raises FeatureError: for a *decay* that is not a finite number, a *delay_ms*
        that is not one above 0, and a clip with NaN or inf in it.
    """
    from scipy.signal import fftconvolve

    sr = sample_rate(sr, 'reverb', FeatureError)
    if real(decay) is None:
        raise FeatureError(f"reverb(): decay has to be a finite number, got {decay!r}")
    if real(delay_ms) is None or delay_ms <= 0:
        raise FeatureError(f"reverb(): delay_ms has to be a finite number of milliseconds above 0, "
                           f"got {delay_ms!r}")
    delay = max(1, int(sr * delay_ms / 1000.0))
    ir_len = delay * 8
    impulse = np.zeros(ir_len, dtype=np.float32)
    t = 0
    amp = 1.0
    while t < ir_len:
        impulse[t] += amp
        amp *= decay
        t += delay
    # integer audio in float first: abs(-32768) stays -32768 in int16, so a
    # sample pinned there was missed as the peak (20000 came back, not 32768)
    x, back = _read(y, 'reverb')
    finite(np.asarray(x), 'reverb', FeatureError)
    if not x.size:
        return back(x)
    wet = _each_channel(x, lambda c: fftconvolve(c, impulse)[:len(c)])
    peak = np.max(np.abs(wet)) + 1e-12
    return back(wet / peak * np.max(np.abs(x)))


def spec_augment(spec: np.ndarray, n_freq_masks: int = 1, n_time_masks: int = 1,
                 freq_mask_width: int = 8, time_mask_width: int = 16,
                 mask_value: Optional[float] = None,
                 seed: Optional[int] = None) -> np.ndarray:
    """SpecAugment: mask random frequency bands and time spans of a spectrogram.

    *spec* is ``(frames, bins)``, the layout every spectrogram in kudio comes
    in (or squeezable to it). Returns a masked copy.

    :param mask_value: what a masked cell becomes. ``None`` (the default) uses
        the spectrogram's mean, which is SpecAugment's choice (D. S. Park et
        al., "SpecAugment: a simple data augmentation method for automatic
        speech recognition", Proc. Interspeech 2019, pp. 2613-2617, Sec. 2,
        doi:10.21437/Interspeech.2019-2680): 0 is the mean of a standardised
        spectrogram, but on kudio's log10 power it is a power of 1, louder
        than most of the cells it would hide.
    """
    rng = generator(seed, 'spec_augment', FeatureError)
    out = np.array(spec, copy=True)
    if out.dtype.kind not in 'biuf':
        raise FeatureError(f"spec_augment() takes real numbers, got {out.dtype}")
    work = out if out.ndim == 2 else out.squeeze()
    if work.ndim != 2:
        raise FeatureError(f"spec must be 2-D after squeeze, got shape {np.shape(spec)}")
    n_time, n_freq = work.shape
    fill = float(np.mean(work)) if mask_value is None else mask_value

    settings = dict(n_freq_masks=n_freq_masks, n_time_masks=n_time_masks,
                    freq_mask_width=freq_mask_width, time_mask_width=time_mask_width)
    for name, value in settings.items():
        settings[name] = whole(value)
        if settings[name] is None or settings[name] < 0:
            raise FeatureError(f"spec_augment(): {name} is a whole number, 0 or more, got "
                               f"{value!r}")
    n_freq_masks, n_time_masks, freq_mask_width, time_mask_width = settings.values()
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
    return work.reshape(np.shape(spec))
