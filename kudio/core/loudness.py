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
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional, Tuple

import numpy as np
from scipy.signal import freqz, lfilter, tf2zpk, zpk2tf

from kudio.exceptions import FeatureError

__all__ = ['loudness', 'normalize_lufs', 'match_loudness',
           'loudness_over_time', 'LoudnessCurve', 'loudness_range',
           'true_peak', 'MOMENTARY', 'SHORT_TERM']

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

#: The two EBU Tech 3341 meter windows, in seconds. Momentary follows
#: syllables; short-term follows how loud a passage is.
MOMENTARY = 0.400
SHORT_TERM = 3.0

#: Seconds between measurements on a curve -- what both meters step by.
CURVE_HOP = 0.100

#: EBU Tech 3342: the loudness range gates 20 LU below the mean rather than
#: the integrated measurement's 10, and spans the 10th to 95th percentile.
_LRA_RELATIVE_GATE = -20.0
_LRA_LOW, _LRA_HIGH = 10.0, 95.0

#: BS.1770-4 Annex 2 asks for at least this rate before the peak is read.
_TRUE_PEAK_SR = 192000

#: Samples per chunk when oversampling, and the context kept either side so
#: the interpolation filter never sees an edge that is not really there.
_CHUNK = 1 << 20
_PAD = 1024


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


def _block_power(data: np.ndarray, sr: int, block: int, step: int) -> np.ndarray:
    """K-weighted mean square per block per channel -- the ``z_ij`` of the
    standard, ``(blocks, channels)``.

    Everything that measures loudness starts here: the integrated figure gates
    these and averages them, the momentary and short-term curves report them
    one at a time. Written once so the two cannot disagree about what a block
    contains.
    """
    from kudio.core._framing import frame_view

    b1, a1, b2, a2 = _k_weighting(int(sr))
    weighted = lfilter(b2, a2, lfilter(b1, a1, data, axis=0), axis=0)
    # Square once, then average strided views of the result. A short-term
    # window is thirty hops long, so squaring inside the loop squares every
    # sample thirty times -- five seconds on a ten-minute file, which is long
    # enough to feel in a GUI that remeasures after every edit.
    squared = np.ascontiguousarray(weighted ** 2)
    columns = [frame_view(squared[:, c], block, step).mean(axis=1)
               for c in range(squared.shape[1])]
    return np.stack(columns, axis=1)


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

    step = max(1, int(round(block * (1.0 - _OVERLAP))))
    power = _block_power(data, int(sr), block, step)
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


@dataclass(frozen=True)
class LoudnessCurve:
    """Loudness measured over a sliding window, in LUFS.

    :param lufs: one value per hop; ``-inf`` for a silent window.
    :param times: **centre** time of each window, in seconds. A meter reports
        at the end of its window; a picture drawn beside a waveform has to line
        up with the audio that produced it, and the centre is where that is.
    """

    lufs: np.ndarray
    times: np.ndarray
    sr: int
    window: float
    hop: float

    def __len__(self) -> int:
        return int(len(self.lufs))

    @property
    def finite(self) -> np.ndarray:
        """The measurements that are not digital silence."""
        return self.lufs[np.isfinite(self.lufs)]

    @property
    def max(self) -> float:
        values = self.finite
        return float(np.max(values)) if values.size else float('-inf')

    @property
    def min(self) -> float:
        values = self.finite
        return float(np.min(values)) if values.size else float('-inf')

    def loudest(self) -> Tuple[float, float]:
        """``(seconds, lufs)`` of the loudest window; ``nan`` if all silent."""
        return self._extreme(np.argmax)

    def quietest(self) -> Tuple[float, float]:
        """``(seconds, lufs)`` of the quietest window that is not silence.

        Silence is excluded deliberately: the quietest moment of a take is
        almost always a pause, and "your quietest moment is the gap between
        two words" is not a finding.
        """
        return self._extreme(np.argmin)

    def _extreme(self, pick) -> Tuple[float, float]:
        mask = np.isfinite(self.lufs)
        if not mask.any():
            return (float('nan'), float('-inf'))
        index = np.flatnonzero(mask)[pick(self.lufs[mask])]
        return (float(self.times[index]), float(self.lufs[index]))

    def spans_below(self, lufs: float,
                    min_seconds: float = 0.0) -> List[Tuple[float, float]]:
        """Where the curve sits below *lufs*, as ``(start, end)`` seconds."""
        return self._spans(self.lufs < lufs, min_seconds)

    def spans_above(self, lufs: float,
                    min_seconds: float = 0.0) -> List[Tuple[float, float]]:
        """Where the curve sits above *lufs*, as ``(start, end)`` seconds."""
        return self._spans(np.isfinite(self.lufs) & (self.lufs > lufs),
                           min_seconds)

    def _spans(self, mask: np.ndarray,
               min_seconds: float) -> List[Tuple[float, float]]:
        if not len(self):
            return []
        spans: List[Tuple[float, float]] = []
        start: Optional[int] = None
        for i, on in enumerate(np.append(mask, False)):
            if on and start is None:
                start = i
            elif not on and start is not None:
                a, b = self.times[start], self.times[i - 1] + self.hop
                if b - a >= min_seconds:
                    spans.append((float(a), float(b)))
                start = None
        return spans

    @property
    def range_lu(self) -> float:
        """The EBU Tech 3342 statistic over this curve, in LU.

        It is the **loudness range** when the curve is short-term, which is
        what :func:`loudness_range` builds. On a momentary curve the same
        arithmetic still runs and means something narrower -- how much the
        syllables vary -- so it is not called LRA here.

        Two gates: an absolute one at -70 LUFS, and a relative one 20 LU below
        the power-mean of what survived it (not the integrated measurement's
        10 LU). What remains is measured from its 10th to its 95th percentile,
        which is what stops a single door slam from setting the answer.

        ``0.0`` when nothing survives the gates: a silent clip has no range
        rather than an undefined one.
        """
        values = self.lufs[self.lufs > _ABSOLUTE_GATE]
        if values.size == 0:
            return 0.0
        # the mean is taken in the power domain, as everywhere else in BS.1770
        # -- averaging decibels would weight a quiet window as heavily as a
        # loud one
        mean = _OFFSET + 10.0 * np.log10(
            float(np.mean(10.0 ** ((values - _OFFSET) / 10.0))))
        kept = values[values > mean + _LRA_RELATIVE_GATE]
        if kept.size == 0:
            return 0.0
        low, high = np.percentile(kept, (_LRA_LOW, _LRA_HIGH))
        return float(high - low)

    def summary(self) -> dict:
        """Everything above as plain numbers, for a report or a CSV row."""
        loud_at, loud = self.loudest()
        quiet_at, quiet = self.quietest()
        return {
            'windows': len(self),
            'range_lu': self.range_lu,
            'window': self.window,
            'max_lufs': loud,
            'max_at': loud_at,
            'min_lufs': quiet,
            'min_at': quiet_at,
            'spread_lu': (loud - quiet
                          if np.isfinite(loud) and np.isfinite(quiet)
                          else float('nan')),
        }

    def __str__(self) -> str:
        if not self.finite.size:
            return f"silent over {len(self)} window(s)"
        loud_at, loud = self.loudest()
        quiet_at, quiet = self.quietest()
        return (f"{loud:.1f} LUFS at {loud_at:.2f}s, "
                f"{quiet:.1f} LUFS at {quiet_at:.2f}s "
                f"({loud - quiet:.1f} LU apart)")


def loudness_over_time(y: np.ndarray, sr: int, *,
                       window: float = SHORT_TERM,
                       hop: float = CURVE_HOP) -> LoudnessCurve:
    """Loudness measured over a sliding window -- where a take drifted.

    :func:`loudness` returns one number for a whole recording, which is what
    you normalise against and useless for finding the moment somebody turned
    away from the microphone.

    :param window: :data:`MOMENTARY` (400 ms) or :data:`SHORT_TERM` (3 s), the
        two EBU Tech 3341 meters. Momentary follows syllables; short-term
        follows how loud a passage *is*.
    :param hop: seconds between measurements; 100 ms is what the meters use.

    **These are ungated.** Gating belongs to the integrated figure, where its
    job is to stop pauses dragging the average down. A curve is a continuous
    quantity, and one with holes punched in it where the gate fired would
    describe a recording that stops existing between words.

    >>> curve = kudio.loudness_over_time(y, sr)              # doctest: +SKIP
    >>> curve.quietest()                                     # doctest: +SKIP
    (12.4, -38.2)
    """
    if sr <= 0:
        raise FeatureError(f"loudness_over_time() needs a positive rate, got {sr}")
    if window <= 0 or hop <= 0:
        raise FeatureError(f"window and hop must be positive, got "
                           f"window={window}, hop={hop}")
    data = _as_channels(y)
    n_channels = data.shape[1]
    if n_channels > len(_CHANNEL_WEIGHTS):
        raise FeatureError(f"loudness_over_time() handles up to "
                           f"{len(_CHANNEL_WEIGHTS)} channels, got {n_channels}")

    block = int(round(window * sr))
    if block < 1 or data.shape[0] < block:
        raise FeatureError(
            f"loudness_over_time() needs at least {window:g} s "
            f"({block} samples at {sr} Hz) but got {data.shape[0]}; "
            f"pass a shorter window to measure a shorter clip")

    step = max(1, int(round(hop * sr)))
    power = _block_power(data, int(sr), block, step)
    weights = np.array(_CHANNEL_WEIGHTS[:n_channels])
    with np.errstate(divide='ignore'):
        values = _OFFSET + 10.0 * np.log10(power @ weights)
    starts = np.arange(len(values)) * step
    times = (starts + block / 2.0) / sr
    return LoudnessCurve(values, times, int(sr), float(window), float(hop))


def loudness_range(y: np.ndarray, sr: int) -> float:
    """Loudness range in LU, to EBU Tech 3342.

    How far the loud parts sit above the quiet parts once the silence and the
    outliers are out of the way. A voice recording with a consistent delivery
    lands near 3-6 LU; one where somebody walked away from the microphone does
    not, and the integrated figure cannot tell you which you have.

    It is the 10th-to-95th percentile spread of the **short-term** loudness,
    after two gates: an absolute one at -70 LUFS, and a relative one 20 LU
    below the power-mean of what survived it. The percentiles are what keep a
    single door slam from setting the answer, and the -20 LU gate -- not the
    integrated measurement's -10 -- is the one Tech 3342 specifies.

    Returns ``0.0`` when nothing survives the gates; a silent or very short
    clip has no range rather than an undefined one.

    If you already have the short-term curve, read
    :attr:`LoudnessCurve.range_lu` instead -- this builds one and asks it, and
    measuring the curve twice is the expensive half of both.

    >>> kudio.loudness_range(y, sr)                          # doctest: +SKIP
    5.7
    """
    return loudness_over_time(y, sr, window=SHORT_TERM, hop=CURVE_HOP).range_lu


def true_peak(y: np.ndarray, sr: int, *,
              oversample: Optional[int] = None) -> float:
    """Peak of the *reconstructed* waveform, in dBTP (BS.1770-4 Annex 2).

    The largest sample is not the largest value the signal reaches. Samples are
    points on a curve that passes between them, and the curve overshoots: a
    full-scale sine at a quarter of the sample rate, sampled at the zero
    crossings and the points between, reads far below the peak it actually
    has. Anything that reconstructs the waveform -- a converter, a lossy
    encoder, a resampler -- meets the real value and clips on it, which is why
    broadcast specifications are written in dBTP and not dBFS.

    Measured by oversampling and taking the maximum. *oversample* defaults to
    whatever brings the rate to at least 192 kHz, which is what the standard
    asks for; at 48 kHz that is the familiar 4x.

    >>> kudio.true_peak(y, sr)                               # doctest: +SKIP
    0.8
    """
    if sr <= 0:
        raise FeatureError(f"true_peak() needs a positive sample rate, got {sr}")
    data = _as_channels(y)
    if data.shape[0] == 0:
        return float('-inf')
    factor = int(oversample) if oversample else max(
        1, int(np.ceil(_TRUE_PEAK_SR / sr)))
    if factor < 1:
        raise FeatureError(f"oversample must be >= 1, got {oversample}")

    peak = _oversampled_peak(data, factor)
    return 20.0 * float(np.log10(peak)) if peak > 0 else float('-inf')


def _oversampled_peak(data: np.ndarray, factor: int) -> float:
    """Largest absolute value after upsampling, in chunks.

    Chunked because the whole point is long files: a ten-minute stereo take at
    48 kHz upsampled 4x is nearly a gigabyte held at once for the sake of one
    number. Each chunk is resampled with real audio either side of it and the
    context is then dropped, so the interpolation filter never sees an edge
    that is not really there.
    """
    if factor == 1:
        return float(np.max(np.abs(data)))
    from scipy.signal import resample_poly

    peak = 0.0
    for start in range(0, data.shape[0], _CHUNK):
        stop = min(start + _CHUNK, data.shape[0])
        lo, hi = max(0, start - _PAD), min(data.shape[0], stop + _PAD)
        up = resample_poly(data[lo:hi], factor, 1, axis=0)
        keep = up[(start - lo) * factor: (stop - lo) * factor]
        if keep.size:
            peak = max(peak, float(np.max(np.abs(keep))))
    return peak


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
