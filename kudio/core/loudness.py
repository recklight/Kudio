# -*- coding: utf-8 -*-
"""Perceptual loudness to ITU-R BS.1770-4, and normalisation based on it.

``normalize`` and ``normalize_db`` scale by the *peak* sample, which says
nothing about how loud something sounds: a clip with one stray transient
normalises to a whisper. BS.1770 answers the question peak normalisation
cannot -- how loud is this to a listener -- by K-weighting the signal,
measuring mean square over 400 ms blocks, and gating away the silence between
words so a pause does not drag the number down.

Why this matters beyond metering: **an A/B comparison between two versions of
a clip is not valid unless they are loudness matched.** The louder one wins,
whatever its quality. :func:`match_loudness` is the primitive for that.

The standard publishes its K-weighting coefficients at 48 kHz only. Speech work
mostly happens at 8 or 16 kHz, so the filters here are re-discretised to the
requested rate through a prewarped bilinear transform: exact at 48 kHz by
construction, and within 0.06 dB of the reference curve at 16 kHz.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional, Tuple

import numpy as np
from scipy.signal import freqz, lfilter, tf2zpk, zpk2tf

from kudio.exceptions import FeatureError

__all__ = ['loudness', 'normalize_lufs', 'match_loudness']

log = logging.getLogger(__name__)

#: The rate the standard's coefficients are published at.
_REF_SR = 48000

#: Stage 1 -- the high shelf standing in for the head's acoustic effect.
#: Prewarped at its own transition frequency when adapted to another rate.
_SHELF = ([1.53512485958697, -2.69169618940638, 1.19839281085285],
          [1.0, -1.69065929318241, 0.73248077421585], 1681.974, 100.0)

#: Stage 2 -- the RLB high pass that discards rumble below hearing.
_RLB = ([1.0, -2.0, 1.0],
        [1.0, -1.99004745483398, 0.99007225036621], 38.135, 1000.0)

#: Channel weights from the standard: surrounds count for more, and the LFE
#: not at all. Mono and stereo -- everything kudio produces -- are all 1.0.
_CHANNEL_WEIGHTS = (1.0, 1.0, 1.0, 1.41, 1.41)

#: Block length and hop of the gating measurement, in seconds (75% overlap).
BLOCK_SIZE = 0.400
_OVERLAP = 0.75

#: The absolute gate, in LUFS, and the relative gate's offset, in LU.
_ABSOLUTE_GATE = -70.0
_RELATIVE_GATE = -10.0

#: The offset that puts a 0 dBFS sine at 0 LUFS.
_OFFSET = -0.691


def _adapt(b, a, f0: float, norm_hz: float, sr: int):
    """Re-discretise a 48 kHz biquad at *sr*, prewarped at *f0*.

    The published filters exist only at 48 kHz. Redesigning them from the
    textbook shelf/high-pass equations does *not* reproduce them, so instead
    the poles and zeros are carried back to the s-plane and forward again at
    the target rate. *norm_hz* is a frequency in the filter's flat region,
    where the gain is pinned to the reference filter's.
    """
    if sr == _REF_SR:
        return np.asarray(b, dtype=float), np.asarray(a, dtype=float)

    zeros, poles, _ = tf2zpk(b, a)
    w0 = 2.0 * np.pi * f0
    k_ref = w0 / np.tan(w0 / (2.0 * _REF_SR))
    k_new = w0 / np.tan(w0 / (2.0 * sr))

    s_zeros = k_ref * (zeros - 1.0) / (zeros + 1.0)
    s_poles = k_ref * (poles - 1.0) / (poles + 1.0)
    z_zeros = (k_new + s_zeros) / (k_new - s_zeros)
    z_poles = (k_new + s_poles) / (k_new - s_poles)
    # the bilinear transform sends s = infinity to z = -1; the zeros lost at
    # infinity come back there, keeping the biquad proper
    z_zeros = np.append(z_zeros, -np.ones(len(s_poles) - len(s_zeros)))

    nb, na = zpk2tf(z_zeros, z_poles, 1.0)
    nb, na = np.real(nb), np.real(na)
    reference = abs(freqz(b, a, worN=[2 * np.pi * norm_hz / _REF_SR])[1][0])
    adapted = abs(freqz(nb, na, worN=[2 * np.pi * norm_hz / sr])[1][0])
    return nb * (reference / adapted), na


@lru_cache(maxsize=16)
def _k_weighting(sr: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The two K-weighting stages as ``(b1, a1, b2, a2)`` for *sr*."""
    b1, a1 = _adapt(*_SHELF, sr=sr)
    b2, a2 = _adapt(*_RLB, sr=sr)
    return b1, a1, b2, a2


def _as_channels(y: np.ndarray) -> np.ndarray:
    """``(samples,)`` or ``(samples, channels)`` -> ``(samples, channels)``."""
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        return y[:, np.newaxis]
    if y.ndim == 2:
        return y
    raise FeatureError(f"loudness() takes mono or (samples, channels), got {y.shape}")


def loudness(y: np.ndarray, sr: int, block_size: float = BLOCK_SIZE) -> float:
    """Gated integrated loudness of *y*, in LUFS.

    Mono, or ``(samples, channels)`` -- **channels last**, the layout
    ``soundfile`` returns. Digital silence measures ``-inf``.

    *block_size* is the gating block in seconds. The standard says 400 ms and
    that is what you want; lowering it is the only way to measure something
    shorter than a block, and it is then no longer a BS.1770 measurement.

    >>> kudio.loudness(y, sr)                       # doctest: +SKIP
    -19.4
    """
    if sr <= 0:
        raise FeatureError(f"loudness() needs a positive sample rate, got {sr}")
    data = _as_channels(y)
    n_channels = data.shape[1]
    if n_channels > len(_CHANNEL_WEIGHTS):
        raise FeatureError(f"loudness() handles up to {len(_CHANNEL_WEIGHTS)} "
                           f"channels, got {n_channels}")

    block = int(round(block_size * sr))
    if block < 1 or data.shape[0] < block:
        raise FeatureError(
            f"loudness() needs at least {block_size:.3f} s "
            f"({block} samples at {sr} Hz) but got {data.shape[0]} samples; "
            f"pass a smaller block_size to measure a shorter clip")

    b1, a1, b2, a2 = _k_weighting(int(sr))
    weighted = lfilter(b2, a2, lfilter(b1, a1, data, axis=0), axis=0)

    step = max(1, int(round(block * (1.0 - _OVERLAP))))
    starts = range(0, data.shape[0] - block + 1, step)
    # mean square per block per channel -- the z_ij of the standard
    power = np.array([np.mean(weighted[s:s + block] ** 2, axis=0) for s in starts])

    weights = np.array(_CHANNEL_WEIGHTS[:n_channels])

    def _level(blocks: np.ndarray) -> float:
        if blocks.size == 0:
            return float('-inf')
        total = float(np.sum(weights * np.mean(blocks, axis=0)))
        return _OFFSET + 10.0 * np.log10(total) if total > 0 else float('-inf')

    with np.errstate(divide='ignore'):
        per_block = _OFFSET + 10.0 * np.log10(power @ weights)

    above_absolute = power[per_block > _ABSOLUTE_GATE]
    if above_absolute.size == 0:
        return float('-inf')
    relative = _level(above_absolute) + _RELATIVE_GATE
    return _level(power[(per_block > _ABSOLUTE_GATE) & (per_block > relative)])


def normalize_lufs(y: np.ndarray, sr: int, lufs: float = -23.0,
                   block_size: float = BLOCK_SIZE) -> np.ndarray:
    """Scale *y* to *lufs* integrated loudness. Silence is returned unchanged.

    -23 LUFS is the EBU R 128 broadcast target. This applies a gain and nothing
    else, so the result can exceed full scale; that is logged rather than
    silently limited, because clamping here would change the loudness you just
    asked for.

    >>> quiet = kudio.normalize_lufs(y, sr, lufs=-23.0)      # doctest: +SKIP
    """
    current = loudness(y, sr, block_size=block_size)
    if not np.isfinite(current):
        return np.asarray(y)
    out = np.asarray(y) * float(10 ** ((lufs - current) / 20.0))
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 1.0:
        log.warning("normalize_lufs: %.1f LUFS puts the peak at %.2f "
                    "(%.1f dBFS over); the signal will clip on output",
                    lufs, peak, 20 * np.log10(peak))
    return out


def match_loudness(y: np.ndarray, reference: np.ndarray, sr: int,
                   reference_sr: Optional[int] = None,
                   block_size: float = BLOCK_SIZE) -> np.ndarray:
    """Scale *y* until it is as loud as *reference*.

    The primitive behind a fair A/B: comparing a processed clip against its
    source tells you about the processing only once the level difference is
    out of the way, since the louder of two clips is reliably preferred
    regardless of quality. Returns *y* unchanged when either side is silent.

    *reference_sr* defaults to *sr*. Pass it when the two were captured at
    different rates -- loudness is a property of the sound, not of the
    sampling, so the two are still comparable.

    >>> fair = kudio.match_loudness(enhanced, original, sr)  # doctest: +SKIP
    """
    target = loudness(reference, reference_sr or sr, block_size=block_size)
    current = loudness(y, sr, block_size=block_size)
    if not (np.isfinite(target) and np.isfinite(current)):
        return np.asarray(y)
    return np.asarray(y) * float(10 ** ((target - current) / 20.0))
