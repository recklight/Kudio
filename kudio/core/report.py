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
from typing import List, Tuple

import numpy as np

from kudio._samples import one_channel
from kudio.core._framing import frame_view as _frame
from kudio.core._policy import sample_rate
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

#: The rates an upsampled file is most often made from. Their Nyquist
#: frequencies are where :func:`_band_limit` looks for what upsampling leaves:
#: a 192 kHz file made from 96 kHz audio is the same mistake as a 16 kHz file
#: made from 8 kHz.
_SOURCE_RATES = (8000, 11025, 16000, 22050, 24000, 32000, 44100, 48000, 88200, 96000)

#: A band limit is an edge where the band just under it stands this far above
#: the band just over it (medians over 0.70-0.88 and 1.05-1.25 of the edge)...
_CLIFF_DB = 20.0
#: ...and the median of everything over it, to Nyquist, stays this far under.
#: A native recording's own floor goes on above its content, which is what
#: tells the two apart; docs/validation.md has how far apart they are.
_DEPTH_DB = 40.0
#: Linear interpolation leaves no edge but a mirror image of the band under
#: the old Nyquist frequency, as loud as the interpolator lets through. By a
#: whole factor (2, 3, 4) its level against the band it mirrors follows that
#: response to within this (median absolute deviation, in dB)...
_IMAGE_SPREAD_DB = 0.3
#: ...and this on the median. A native band that rolls off smoothly above the
#: edge can match the curve's shape, but it stands above it, not on it.
_IMAGE_OFFSET_DB = 1.0
#: Shorter than this, the spectrum is too rough to judge an edge by.
_BAND_LIMIT_MIN_SAMPLES = 2048


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
    #: samples inside a flat run pinned to full scale, ``clip_run`` long or
    #: more (three unless :func:`audio_report` was told otherwise)
    clipped_samples: int
    clipped_ratio: float
    #: mean sample value; a healthy recording sits at zero
    dc_offset: float
    silence_ratio: float
    noise_floor_dbfs: float
    #: loud frames against quiet frames -- an estimate, not a measurement
    estimated_snr_db: float
    #: highest frequency still within 50 dB of the spectrum's peak
    bandwidth_hz: float
    #: the content stops at an edge with nothing but an empty band above it,
    #: or is mirrored above one: upsampled, or through a narrow codec
    band_limited: bool
    #: samples that are NaN or inf; with any of these the rest is meaningless
    non_finite_samples: int = 0
    #: the rates the clip was very likely upsampled from, when its band limit
    #: sits at the Nyquist frequency of a common rate: one, or two neighbours
    #: 9% apart (11025 and 12000, 22050 and 24000, 44100 and 48000, 88200 and
    #: 96000) when resamplers put the edge where either could have left it
    source_rates: Tuple[int, ...] = ()
    #: the band limit is a mirror image of the band under it, as linear
    #: interpolation leaves, rather than an edge
    mirrored: bool = False
    #: where the content stops by the edge test: the last frequency within
    #: 10 dB of the band under the edge, or the old Nyquist frequency for a
    #: mirror image; nan with no band limit. ``bandwidth_hz`` reads a 1024-point
    #: spectrum 50 dB under its peak, and a resampler's stopband or a line
    #: above the edge can carry it past the rate the file is named after.
    band_edge_hz: float = float('nan')
    #: samples at or past 0.999 of full scale, each counted whether or not it
    #: is part of a run: the other common way of counting clipping
    full_scale_samples: int = 0
    #: the step of the grid every sample sits on, as a fraction of full scale
    #: (2/65536 for 16-bit audio); 0.0 when there is no grid as coarse as 24
    #: bits, as in float audio that has been processed
    sample_step: float = 0.0
    #: how many bits that grid amounts to, ``log2(2 / sample_step)``; nan with
    #: no grid. Not ENOB, which is read from a sine's SINAD
    grid_bits: float = float('nan')

    def problems(self) -> List[str]:
        """The findings worth acting on, in plain words. Empty is good news."""
        found = []
        if self.non_finite_samples:
            # First, and on its own: every number in this report came out of
            # the same samples, so with NaN among them there is nothing else
            # here worth reading. It has to be its own test rather than
            # something the tests below catch, because a comparison against
            # nan is false whichever way round it is written -- which is how a
            # corrupt clip came to be reported as a muted microphone, the one
            # answer that stops anybody looking any further.
            return [f"{self.non_finite_samples} samples are not finite (NaN "
                    f"or inf) — the clip is corrupt, not quiet, and every "
                    f"measurement here is meaningless. Repair it or drop it "
                    f"from the set"]
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
            # + 0.0 so a peak of 32767/32768 prints as 0.0, not -0.0
            found.append(f"peaks at {round(self.peak_dbfs, 1) + 0.0:.1f} dBFS, "
                         f"right against the "
                         f"ceiling — one more gain stage will clip it")
        if self.peak_dbfs < -30.0:
            found.append(f"peaks at only {self.peak_dbfs:.1f} dBFS — very quiet, "
                         f"so the usable bit depth is small")
        if abs(self.dc_offset) > 0.01:
            found.append(f"DC offset of {self.dc_offset:+.3f} — the waveform is "
                         f"not centred on zero, which wastes headroom")
        if self.band_limited and self.mirrored and self.source_rates:
            rate = self.source_rates[0]
            found.append(
                f"the band above {rate / 2:g} Hz mirrors the band under it — the "
                f"file was very likely upsampled from {rate} Hz by linear "
                f"interpolation, and that band holds nothing the original did not")
        elif self.band_limited:
            if len(self.source_rates) == 1:
                came_from = (f"upsampled from {self.source_rates[0]} Hz (a codec "
                             f"that cuts there looks the same)")
            elif self.source_rates:
                came_from = (f"upsampled from {self.source_rates[0]} or "
                             f"{self.source_rates[1]} Hz (a codec that cuts there "
                             f"looks the same)")
            else:
                came_from = ("upsampled from a lower rate, or passed through a "
                             "codec or a filter that cuts there")
            stops = (self.band_edge_hz if np.isfinite(self.band_edge_hz)
                     else self.bandwidth_hz)
            found.append(
                f"content stops at {stops:.0f} Hz, although a "
                f"{self.sr} Hz file holds up to {self.sr / 2:g} Hz — it was very "
                f"likely {came_from}, and the band above is empty")
        if 0.0 < self.sample_step and self.grid_bits < 15.95:
            found.append(
                f"every sample sits on a grid of {self.grid_bits:.1f} bits — "
                f"coarser than 16-bit audio, from a low-resolution converter or "
                f"gain raised after the audio was quantised")
        if self.silence_ratio > 0.6:
            found.append(f"{self.silence_ratio * 100:.0f}% of the clip is silence")
        if np.isfinite(self.estimated_snr_db) and self.estimated_snr_db < 10.0:
            # Noisy speech and a steady sound give the same frame levels: with
            # noise 0.4 under the speech of the tests, loud and quiet frames
            # differ by 2.3 dB; white noise reads 0.8, a 50 Hz hum with its
            # harmonics 2.2, and noise modulated 30% at 20 Hz 2.7. Saying "the
            # noise is close to the content" about a machine is a guess, so
            # the finding names both.
            found.append(
                f"estimated SNR around {self.estimated_snr_db:.0f} dB — noise "
                f"close to the content gives that, and so does a steady sound "
                f"such as hum or a running machine, whose loud and quiet frames "
                f"hardly differ; the estimate cannot tell the two apart")
        return found

    def __str__(self) -> str:      # pragma: no cover - display helper
        lufs = "n/a" if not np.isfinite(self.lufs) else f"{self.lufs:.1f} LUFS"
        return (f"{self.duration:.2f}s @ {self.sr} Hz · peak "
                f"{self.peak_dbfs:.1f} dBFS · {lufs} · "
                f"SNR~{self.estimated_snr_db:.0f} dB · "
                f"bandwidth {self.bandwidth_hz:.0f} Hz")


def _clipped_run_count(y: np.ndarray, shortest: int = _CLIP_RUN) -> int:
    """Samples sitting on a flat top at full scale, *shortest* or more in a row.

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
        if stop - start < shortest:
            continue
        run = np.abs(y[start:stop])
        if not np.all(np.isfinite(run)):
            # inf counts as pinned to the rail and it is not clipping: it is a
            # broken sample, which `audio_report` counts and reports on its
            # own. Measuring its flatness means `inf - inf`, and all that
            # produces is a warning about the arithmetic.
            continue
        # the plateau is the part at the run's own level: a waveform reaching
        # it leaves a sample at 0.9993 or so in front, and taking the run as a
        # whole lost every plateau that had one, however long
        level = float(run.max())
        flat = (run >= level - _CLIP_FLATNESS).view(np.int8)
        bounds = np.diff(np.concatenate(([0], flat, [0])))
        lengths = np.nonzero(bounds == -1)[0] - np.nonzero(bounds == 1)[0]
        # ...but a smooth peak has a top within 1e-4 too: near level m it
        # falls as a parabola, so sqrt(1e-4 / (m - 0.999)) of its run above
        # 0.999 stays within 1e-4 of m, 0.32 of it at full scale, and below
        # about 100 Hz at 48 kHz that is three samples or more. A plateau
        # counts when it holds twice the share a smooth peak would.
        rise = level - _CLIP_LEVEL
        if rise <= _CLIP_FLATNESS:
            continue            # the whole run is within 1e-4: no telling
        smooth = np.sqrt(_CLIP_FLATNESS / rise)
        kept = (lengths >= shortest) & (lengths >= 2.0 * smooth * len(run))
        total += int(lengths[kept].sum())
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
    if not np.all(np.isfinite(y)):
        # No spectrum: the transform of a clip with NaN or inf in it is NaN
        # throughout, which the power test below reads as no content anyway.
        # Taken here so the transform is not asked for in the first place.
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


def _welch_db(y: np.ndarray, sr: int):
    """Welch power spectrum in dB: 8192-point Hann segments, half overlapped.

    Averaged over the whole clip because an edge is a property of the file,
    not of one moment in it; 8192 points put 23 Hz between bins at 192 kHz.
    """
    from scipy.signal import welch
    size = min(len(y), 8192)
    freqs, power = welch(y, sr, nperseg=size, noverlap=size // 2)
    return freqs, 10.0 * np.log10(np.maximum(power, 1e-30))


class _Edges:
    """The spectrum of one clip, asked about edges at one frequency at a time.

    The spectrum is median-smoothed over 2% of the frequency being asked about
    before anything is read from it, so a tone or a hum line does not count as
    band. Smoothed copies are kept by window length, which repeats often in a
    scan of 200 frequencies.
    """

    def __init__(self, freqs: np.ndarray, db: np.ndarray, sr: int):
        self.freqs, self.db, self.nyquist = freqs, db, sr / 2.0
        self._step = float(freqs[1] - freqs[0])
        self._smoothed = {}

    def _smooth(self, fc: float) -> np.ndarray:
        from scipy.ndimage import median_filter
        size = max(3, int(round(0.02 * fc / self._step)) | 1)
        if size not in self._smoothed:
            self._smoothed[size] = median_filter(self.db, size, mode='nearest')
        return self._smoothed[size]

    def _median(self, smoothed, lo, hi) -> float:
        inside = (self.freqs >= lo) & (self.freqs <= hi)
        return float(np.median(smoothed[inside])) if inside.any() else float('nan')

    def cliff_and_depth(self, fc: float):
        """``(cliff, depth)`` in dB at *fc*.

        cliff: the band at 0.70-0.88 fc against the band at 1.05-1.25 fc, the
        drop right at the edge. depth: the same band against everything from
        1.05 fc to 0.97 of Nyquist, which is where a native recording's floor
        shows itself by going on. The gaps either side of fc leave room for a
        resampler's transition band.
        """
        smoothed = self._smooth(fc)
        below = self._median(smoothed, 0.70 * fc, 0.88 * fc)
        near = self._median(smoothed, 1.05 * fc, min(1.25 * fc, 0.98 * self.nyquist))
        above = self._median(smoothed, 1.05 * fc, 0.97 * self.nyquist)
        return below - near, below - above

    def is_edge(self, fc: float) -> bool:
        cliff, depth = self.cliff_and_depth(fc)
        # False when either is nan: no band on one side to compare
        return bool(cliff >= _CLIFF_DB and depth >= _DEPTH_DB)

    def edge_frequency(self, lowest: float, highest: float) -> float:
        """Where the content stops, for edges found between *lowest* and
        *highest*: the last frequency up to 1.25 *highest* still within 10 dB
        of the band at 0.70-0.88 *lowest*.

        Needed because the edge test cannot tell neighbouring rates apart:
        11025 and 12000 Hz are 9% apart, and the bands either side of one
        reach past the other.
        """
        smoothed = self._smooth(lowest)
        level = self._median(smoothed, 0.70 * lowest, 0.88 * lowest)
        inside = ((self.freqs >= 0.70 * lowest) & (self.freqs <= 1.25 * highest)
                  & (smoothed >= level - 10.0))
        return float(self.freqs[inside][-1]) if inside.any() else 0.70 * lowest

    def image_fit(self, fc: float):
        """How closely the band above *fc* follows a linear-interpolation image.

        Linear interpolation from rate 2 fc is a triangle filter; it leaves the
        band under fc mirrored above it, weakened by the filter's response,
        ``(sin(pi f / 2fc) / (L sin(pi f / sr)))**2`` in amplitude. At fc - x and
        fc + x the first sines are equal, so the image stands
        ``40 log10 |sin(pi (fc - x) / sr) / sin(pi (fc + x) / sr)|`` dB under the
        band it mirrors -- a fixed curve, whatever the content. Returns
        ``(spread, offset)``: the median absolute deviation of the measured
        difference from that curve, and its median. None when too little of the
        image stands above the floor to judge.
        """
        top = min(0.9 * fc, 0.98 * self.nyquist - fc)
        if top <= 0.3 * fc:
            return None
        sr = 2.0 * self.nyquist
        x = np.linspace(0.05 * fc, top, 300)
        under = np.interp(fc - x, self.freqs, self.db)
        over = np.interp(fc + x, self.freqs, self.db)
        floor = float(np.percentile(self.db[self.freqs > 0.05 * self.nyquist], 2))
        usable = over > floor + 10.0
        if usable.sum() < 0.3 * len(x):
            return None
        expected = 40.0 * np.log10(np.abs(np.sin(np.pi * (fc - x) / sr)
                                          / np.sin(np.pi * (fc + x) / sr)))
        residual = (over - under - expected)[usable]
        offset = float(np.median(residual))
        return float(np.median(np.abs(residual - offset))), offset

    def is_image(self, fc: float) -> bool:
        fit = self.image_fit(fc)
        return (fit is not None and fit[0] <= _IMAGE_SPREAD_DB
                and abs(fit[1]) <= _IMAGE_OFFSET_DB)


def _band_limit(y: np.ndarray, sr: int):
    """``(band_limited, source_rates, mirrored, band_edge_hz)`` for one clip.

    In order: an edge at the Nyquist frequency of a rate in
    :data:`_SOURCE_RATES` (upsampled; the rate or rates the edge fits are
    named); a linear-interpolation image above one (upsampled from that rate
    by a whole factor, mirrored); an edge anywhere from 0.1 to 0.9 of Nyquist
    (through a codec or a filter that cut there). Edges above 0.9 of Nyquist
    are not judged: an anti-aliasing filter of the file's own rate can sit
    there.
    """
    if len(y) < _BAND_LIMIT_MIN_SAMPLES or not np.all(np.isfinite(y)) or not np.any(y):
        return False, (), False, float('nan')
    edges = _Edges(*_welch_db(y, sr), sr)
    nyquist = sr / 2.0
    sources = [rate for rate in _SOURCE_RATES if rate / 2.0 <= 0.9 * nyquist]

    found = [rate for rate in sources if edges.is_edge(rate / 2.0)]
    if found:
        # The edge test alone cannot name the rate: its bands start 5% either
        # side of the edge, and neighbouring rates are 9% apart. Where the
        # content stops can. soxr's VHQ, HQ and MQ put that at 0.959-0.965 of
        # the old Nyquist frequency, FFT resampling at 1.000 and scipy's
        # resample_poly at 1.024-1.052 (docs/validation.md), so every rate it
        # falls 0.95-1.06 of is named, which is sometimes two. soxr's LQ stops
        # at 0.86-0.88, and content that stopped short of its own Nyquist
        # frequency anywhere else; neither is named, rather than named wrong.
        stops = edges.edge_frequency(found[0] / 2.0, found[-1] / 2.0)
        named = tuple(rate for rate in found if 0.95 <= stops / (rate / 2.0) <= 1.06)
        return True, named, False, stops
    for rate in sources:
        if edges.is_image(rate / 2.0):
            return True, (rate,), True, rate / 2.0
    for fc in np.geomspace(0.1 * nyquist, 0.9 * nyquist, 200):
        if edges.is_edge(float(fc)):
            return True, (), False, edges.edge_frequency(float(fc), float(fc))
    return False, (), False, float('nan')


def _sample_grid(y: np.ndarray):
    """``(step, bits)``: the grid every sample sits on, or ``(0.0, nan)``.

    Read at 24-bit resolution. The two values at full scale are left out, so a
    clipped take is not read as a grid with one huge step; fewer than 16
    distinct values, as in a square wave or a near-silent take, is no grid
    worth naming.
    """
    finite = y[np.isfinite(y)]
    scaled = finite * float(1 << 23)
    if scaled.size == 0 or not np.array_equal(scaled, np.round(scaled)):
        return 0.0, float('nan')
    levels = np.unique(scaled[np.abs(scaled) < 0.999 * (1 << 23)].astype(np.int64))
    if levels.size < 16:
        return 0.0, float('nan')
    step = int(np.gcd.reduce(np.diff(levels)))
    return step / float(1 << 23), float(24.0 - np.log2(step))


def _db(value: float) -> float:
    if not np.isfinite(value):
        # -inf is reserved for a measurement that came out at zero. A peak of
        # nan has no dB value at all, and reporting it as -inf is how a
        # corrupt clip came to look like digital silence.
        return float('nan')
    return 20.0 * np.log10(value) if value > 0 else float('-inf')


def audio_report(y: np.ndarray, sr: int, *, clip_run: int = _CLIP_RUN) -> AudioReport:
    """Inspect *y* without needing a clean reference to compare it against.

    Mono only -- mix down first if you have channels; the questions this
    answers are about the recording, not the mix.

    >>> report = kudio.audio_report(y, sr)               # doctest: +SKIP
    >>> for problem in report.problems():                # doctest: +SKIP
    ...     print(problem)
    content stops at 3844 Hz, although a 16000 Hz file holds up to 8000 Hz — ...

    A clip carrying NaN or inf is measured and counted rather than refused:
    the count is the finding, reported first by :meth:`AudioReport.problems`.
    The peak, RMS and loudness come out nan; the noise floor, the SNR estimate
    and the silence share are taken over the frames that are finite. Inspecting
    a corpus is exactly when you want to be told which files are broken, not
    stopped at the first one.

    The noise floor is read from frames with the DC offset taken out, so an
    offset is reported as an offset rather than as noise. When at least a
    tenth of the frames are digital silence, the floor *is* digital silence:
    ``-inf``, with an SNR estimate of ``inf``.

    Clipping is counted in flat runs of *clip_run* or more pinned samples,
    three by default; a run of one or two is indistinguishable from a peak
    that touched full scale. Tools differ here, so the count of every sample
    at full scale, run or not, is in ``full_scale_samples`` beside it.

    A band limit is an edge with an empty band above it, or the mirror image
    linear interpolation leaves. A recording made at its own rate keeps its
    converter's floor going above the content, and is not band-limited
    however low the content stops; docs/validation.md has the resamplers it
    was tried on and the margins.

    :raises FeatureError: on a non-mono array, a non-positive rate, or a
        *clip_run* that is not a whole number of 1 or more.
    """
    sr = sample_rate(sr, 'audio_report', FeatureError)
    if int(clip_run) != clip_run or clip_run < 1:
        raise FeatureError(f"audio_report() needs a whole clip_run of 1 or more, "
                           f"got {clip_run}")
    y = one_channel(y, 'audio_report', FeatureError)
    y = np.ascontiguousarray(np.asarray(y, dtype=np.float64).squeeze())
    if y.ndim != 1:
        raise FeatureError(f"audio_report() takes mono audio, got shape {y.shape}")
    if y.size == 0:
        raise FeatureError("audio_report() got an empty waveform")

    non_finite = int(np.count_nonzero(~np.isfinite(y)))
    peak = float(np.max(np.abs(y)))
    rms = float(np.sqrt(np.mean(y ** 2)))
    clipped = _clipped_run_count(y, clip_run)

    size = max(16, int(round(_FRAME * sr)))
    # the offset out first: left in, a 0.03 offset read as a noise floor of
    # -30 dBFS on a take whose real floor was -57
    usable = np.isfinite(y)
    centred = y - (float(np.mean(y[usable])) if usable.any() else 0.0)
    frame_rms = np.sqrt(np.mean(_frame(centred, size, max(1, int(round(_HOP * sr))))
                                ** 2, axis=-1))
    with np.errstate(divide='ignore', invalid='ignore'):
        frame_db = 20.0 * np.log10(np.maximum(frame_rms, 0.0))

    finite = frame_db[np.isfinite(frame_db)]
    # exact zeros are found on the samples as they are: with the mean taken
    # out, a zeroed pause is a tiny constant rather than nothing
    hop = max(1, int(round(_HOP * sr)))
    zero_frames = np.all(_frame(y, size, hop) == 0.0, axis=-1)
    digital_silence = float(np.mean(zero_frames)) if zero_frames.size else 0.0
    if finite.size:
        loud = float(np.percentile(finite, 95))
        floor = float(np.percentile(finite, 10))
        if digital_silence >= 0.1:
            # pauses of exact zeros: the floor is silence; the 10th
            # percentile would land on the quietest stretch of speech instead
            floor = float('-inf')
        silence_ratio = float(np.mean((frame_db < loud - _SILENCE_REL_DB)
                                      | zero_frames))
        snr = loud - floor
    else:                                       # every frame is exact silence
        loud = floor = float('-inf')
        silence_ratio, snr = 1.0, float('-inf')

    if non_finite:
        # undefined, and loudness() would warn about the samples this report
        # already names as its first finding
        lufs = float('nan')
    else:
        try:
            lufs = loudness(y, sr)
        except FeatureError:                    # shorter than a BS.1770 block
            lufs = float('nan')

    bandwidth = _bandwidth(y, sr)
    band_limited, source_rates, mirrored, band_edge = _band_limit(y, sr)
    step, bits = _sample_grid(y)

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
        band_limited=band_limited,
        non_finite_samples=non_finite,
        source_rates=source_rates,
        mirrored=mirrored,
        band_edge_hz=band_edge,
        full_scale_samples=int(np.count_nonzero(np.abs(y[usable]) >= _CLIP_LEVEL)),
        sample_step=step,
        grid_bits=bits,
    )
