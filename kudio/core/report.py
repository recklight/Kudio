# -*- coding: utf-8 -*-
"""What can be said about a recording with no clean reference to compare it to.

Every quality metric in :mod:`kudio.core.evaluator` -- SNR, SI-SDR, PESQ,
STOI -- needs the clean signal alongside the degraded one. Synthetic datasets
have that. **Real recordings never do**, and those are the ones you actually
need to judge: score a denoised field recording against the noisy original and
you measure how much you changed it, not whether it got better.

This module is the half of that gap which costs nothing: level, clipping, DC
offset, how much of the file is silence, where the noise floor sits, a crude
SNR estimate, and how much bandwidth the content really occupies. None of it
predicts a listener's opinion -- that needs a trained model such as DNSMOS or
NISQA -- but all of it catches the mistakes that make a dataset quietly
useless before anyone gets as far as an opinion.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from kudio.core._framing import frame_view as _frame
from kudio.core.loudness import loudness
from kudio.exceptions import FeatureError

__all__ = ['AudioReport', 'audio_report']

log = logging.getLogger(__name__)

#: Analysis frame and hop, in seconds. 30 ms is long enough for a stable level
#: reading and short enough to sit inside a single phoneme.
_FRAME = 0.030
_HOP = 0.015

#: A sample this close to full scale is treated as pinned...
_CLIP_LEVEL = 0.999
#: ...but only a run of them is clipping. One sample touching 1.0 is luck.
_CLIP_RUN = 3
#: ...and the run has to be flat. A loud sine still traces its own shape.
_CLIP_FLATNESS = 1e-4

#: A frame this far below the loud frames counts as silence.
_SILENCE_REL_DB = 40.0

#: Where the spectrum is considered to have stopped, relative to its peak.
_BANDWIDTH_REL_DB = -50.0
#: Content reaching less than this fraction of Nyquist did not start life here.
_BAND_LIMITED_RATIO = 0.9

#: Rates real files are recorded at, for naming the one a clip came from.
_COMMON_RATES = (8000, 11025, 16000, 22050, 32000, 44100, 48000)


def _nearest_common_rate(hz: float) -> Optional[int]:
    """The standard rate *hz* is probably a measurement of, if any is close."""
    if hz <= 0:
        return None
    rate = min(_COMMON_RATES, key=lambda r: abs(r - hz))
    return rate if abs(rate - hz) <= 0.15 * rate else None


@dataclass(frozen=True)
class AudioReport:
    """Reference-free measurements of one clip. See :func:`audio_report`."""

    sr: int
    duration: float
    #: dBFS of the loudest sample; ``-inf`` for digital silence
    peak_dbfs: float
    rms_dbfs: float
    #: BS.1770 integrated loudness, or ``nan`` when the clip is too short
    lufs: float
    #: samples inside a run of at least three pinned to full scale
    clipped_samples: int
    clipped_ratio: float
    #: mean sample value; a healthy recording sits at zero
    dc_offset: float
    silence_ratio: float
    noise_floor_dbfs: float
    #: loud frames against quiet frames -- an estimate, not a measurement
    estimated_snr_db: float
    #: highest frequency still carrying content
    bandwidth_hz: float
    #: content stops well below Nyquist: upsampled, or through a narrow codec
    band_limited: bool

    def problems(self) -> List[str]:
        """The findings worth acting on, in plain words. Empty is good news."""
        found = []
        if self.peak_dbfs == float('-inf'):
            return ["the clip is digital silence"]
        if self.clipped_samples:
            found.append(
                f"{self.clipped_samples} samples clipped "
                f"({self.clipped_ratio * 100:.2f}% of the clip) — the recording "
                f"was already too hot; no processing gets that back")
        elif self.peak_dbfs > 0.0:
            found.append(f"peaks at {self.peak_dbfs:+.1f} dBFS, past full scale — "
                         f"it will clip the moment it is written to an integer "
                         f"format")
        elif self.peak_dbfs > -0.5:
            found.append(f"peaks at {self.peak_dbfs:.1f} dBFS, right against the "
                         f"ceiling — one more gain stage will clip it")
        if self.peak_dbfs < -30.0:
            found.append(f"peaks at only {self.peak_dbfs:.1f} dBFS — very quiet, "
                         f"so the usable bit depth is small")
        if abs(self.dc_offset) > 0.01:
            found.append(f"DC offset of {self.dc_offset:+.3f} — the waveform is "
                         f"not centred on zero, which wastes headroom")
        if self.band_limited:
            source = _nearest_common_rate(2 * self.bandwidth_hz)
            came_from = (f"upsampled from {source} Hz" if source
                         else f"band-limited to about {2 * self.bandwidth_hz:.0f} Hz")
            found.append(
                f"content stops at {self.bandwidth_hz:.0f} Hz although the file "
                f"claims {self.sr} Hz — it was very likely {came_from}, and the "
                f"extra rate is empty")
        if self.silence_ratio > 0.6:
            found.append(f"{self.silence_ratio * 100:.0f}% of the clip is silence")
        if np.isfinite(self.estimated_snr_db) and self.estimated_snr_db < 10.0:
            found.append(f"estimated SNR around {self.estimated_snr_db:.0f} dB — "
                         f"the noise floor is close to the content")
        return found

    def __str__(self) -> str:      # pragma: no cover - display helper
        lufs = "n/a" if not np.isfinite(self.lufs) else f"{self.lufs:.1f} LUFS"
        return (f"{self.duration:.2f}s @ {self.sr} Hz · peak "
                f"{self.peak_dbfs:.1f} dBFS · {lufs} · "
                f"SNR~{self.estimated_snr_db:.0f} dB · "
                f"bandwidth {self.bandwidth_hz:.0f} Hz")


def _clipped_run_count(y: np.ndarray) -> int:
    """Samples sitting on a flat top at full scale.

    Clipping is a *plateau*, not merely a large value. Float audio that has
    been amplified past 1.0 still traces the waveform and has lost nothing yet;
    it is heading for trouble on the way to an integer format, which is a
    different warning. Only a run of near-identical samples pinned at the rail
    is information that has actually been destroyed.
    """
    pinned = np.abs(y) >= _CLIP_LEVEL
    if not pinned.any():
        return 0
    edges = np.diff(np.concatenate(([0], pinned.view(np.int8), [0])))
    starts = np.nonzero(edges == 1)[0]
    stops = np.nonzero(edges == -1)[0]

    total = 0
    for start, stop in zip(starts, stops):
        if stop - start < _CLIP_RUN:
            continue
        run = np.abs(y[start:stop])
        if float(run.max() - run.min()) <= _CLIP_FLATNESS:
            total += stop - start
    return total


def _bandwidth(y: np.ndarray, sr: int) -> float:
    """Highest frequency still above :data:`_BANDWIDTH_REL_DB` under the peak.

    A heuristic, and a reliable one: a recording made at its stated rate has a
    noise floor spread over the whole band, so its spectrum reaches Nyquist.
    Audio upsampled from a lower rate has nothing above the old Nyquist at all.
    """
    size = min(1024, len(y))
    if size < 32:
        return 0.0
    windowed = _frame(y, size, max(1, size // 2)) * np.hanning(size)
    power = np.mean(np.abs(np.fft.rfft(windowed, axis=-1)) ** 2, axis=0)
    if not np.any(power > 0):
        return 0.0
    db = 10.0 * np.log10(np.maximum(power, 1e-30))
    above = np.nonzero(db - db.max() > _BANDWIDTH_REL_DB)[0]
    if above.size == 0:
        return 0.0
    return float(np.fft.rfftfreq(size, 1.0 / sr)[above[-1]])


def _db(value: float) -> float:
    return 20.0 * np.log10(value) if value > 0 else float('-inf')


def audio_report(y: np.ndarray, sr: int) -> AudioReport:
    """Inspect *y* without needing a clean reference to compare it against.

    Mono only -- mix down first if you have channels; the questions this
    answers are about the recording, not the mix.

    >>> report = kudio.audio_report(y, sr)               # doctest: +SKIP
    >>> for problem in report.problems():                # doctest: +SKIP
    ...     print(problem)
    content stops at 3844 Hz although the file claims 16000 Hz — ...

    :raises FeatureError: on a non-mono array or a non-positive rate.
    """
    if sr <= 0:
        raise FeatureError(f"audio_report() needs a positive sample rate, got {sr}")
    y = np.ascontiguousarray(np.asarray(y, dtype=np.float64).squeeze())
    if y.ndim != 1:
        raise FeatureError(f"audio_report() takes mono audio, got shape {y.shape}")
    if y.size == 0:
        raise FeatureError("audio_report() got an empty waveform")

    peak = float(np.max(np.abs(y)))
    rms = float(np.sqrt(np.mean(y ** 2)))
    clipped = _clipped_run_count(y)

    size = max(16, int(round(_FRAME * sr)))
    frame_rms = np.sqrt(np.mean(_frame(y, size, max(1, int(round(_HOP * sr)))) ** 2,
                                axis=-1))
    with np.errstate(divide='ignore'):
        frame_db = 20.0 * np.log10(np.maximum(frame_rms, 0.0))

    finite = frame_db[np.isfinite(frame_db)]
    if finite.size:
        loud = float(np.percentile(finite, 95))
        floor = float(np.percentile(finite, 10))
        silence_ratio = float(np.mean(frame_db < loud - _SILENCE_REL_DB))
        snr = loud - floor
    else:                                       # every frame is exact silence
        loud = floor = float('-inf')
        silence_ratio, snr = 1.0, float('-inf')

    try:
        lufs = loudness(y, sr)
    except FeatureError:                        # shorter than a BS.1770 block
        lufs = float('nan')

    bandwidth = _bandwidth(y, sr)
    nyquist = sr / 2.0

    return AudioReport(
        sr=int(sr),
        duration=len(y) / sr,
        peak_dbfs=_db(peak),
        rms_dbfs=_db(rms),
        lufs=float(lufs),
        clipped_samples=clipped,
        clipped_ratio=clipped / len(y),
        dc_offset=float(np.mean(y)),
        silence_ratio=silence_ratio,
        noise_floor_dbfs=floor,
        estimated_snr_db=snr,
        bandwidth_hz=bandwidth,
        band_limited=bool(0.0 < bandwidth < _BAND_LIMITED_RATIO * nyquist),
    )
