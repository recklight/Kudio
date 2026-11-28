# -*- coding: utf-8 -*-
"""Voice activity detection: where in this recording is somebody talking.

:func:`kudio.split_on_silence` cuts wherever the level drops, which is the
right answer for a clean studio take and the wrong one for anything with a
noise floor — a fan, a road outside, a hiss. Above a certain noise level a
fixed ``top_db`` either keeps the whole file or throws away the quiet
consonants, and there is no value in between that works.

This decides per frame instead, on three cheap signals:

* **level against the recording's own noise floor**, rather than against a
  fixed threshold, so it adapts to how noisy the file happens to be;
* **spectral flatness** — speech is harmonic and therefore peaky, broadband
  noise is flat. This is what separates "a loud hiss" from "a quiet voice".
  Flatness is taken over the whole band, so noise confined to part of it (a
  hum, a rumble, a fan with a strong tone) is peaky too and can pass as voice
  on this test alone;
* **zero-crossing rate**, which keeps the unvoiced fricatives (*s*, *f*, *sh*)
  that carry almost no energy and would otherwise be cut off the ends of words
  — but only where they sit **next to** something voiced, because on its own a
  fricative and a hiss are the same measurement.

The consequence of that last rule is worth knowing: a whisper, which has no
voiced frames at all, is not detected. Admitting it would mean admitting every
noise burst in every file, and a detector that fires on the air conditioning is
not a detector.

Then the frame decisions are smoothed: short gaps inside speech are bridged,
segments too short to be a word are dropped, and what survives is padded so the
cut does not land on the first consonant. Pure numpy — no webrtcvad, no torch.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

from kudio._numbers import count, flag, real, within
from kudio._samples import channels_last, one_channel
from kudio.core._framing import convolve_same, frame_view
from kudio.core._policy import sample_rate
from kudio.exceptions import FeatureError

__all__ = ['vad', 'vad_split', 'vad_trim', 'vad_frames', 'speech_ratio']

log = logging.getLogger(__name__)

#: Analysis frame / hop in seconds. 30 ms sits inside a single phoneme.
FRAME = 0.030
#: Seconds between frames.
HOP = 0.010

#: How far above the estimated noise floor a frame has to be, in dB.
THRESHOLD_DB = 8.0
#: Spectral flatness above this is noise-like, not voice. A perfectly even
#: spectrum is 1.0; white noise on these 30 ms frames measures about 0.56,
#: because one frame's spectrum is not flat, and noise confined to a band
#: measures lower still.
FLATNESS_MAX = 0.40
#: A frame this far under the loudest one is silence whatever else says.
FLOOR_REL_DB = 45.0
#: Zero-crossing rate above this rescues unvoiced fricatives.
ZCR_MIN = 0.25
#: ...but only within this many seconds of something actually voiced. A
#: fricative on its own is indistinguishable from broadband noise; a fricative
#: at the edge of a vowel is a consonant.
FRICATIVE_REACH = 0.20

_NOISE_PERCENTILE = 15.0


def _flatness(frames: np.ndarray, keep: Optional[np.ndarray] = None) -> np.ndarray:
    """Wiener entropy per frame: geometric over arithmetic mean of the power,
    over the bins *keep* marks (all but DC when it is None)."""
    window = np.hanning(frames.shape[-1])
    power = np.abs(np.fft.rfft(frames * window, axis=-1)) ** 2
    power = power[:, 1:] if keep is None else power[:, keep]
    power = np.maximum(power, 1e-20)                # keep the log sane
    geometric = np.exp(np.mean(np.log(power), axis=-1))
    arithmetic = np.mean(power, axis=-1)
    return geometric / np.maximum(arithmetic, 1e-20)


def _band_bins(size: int, sr: int, low: float, high: float
               ) -> Tuple[np.ndarray, float]:
    """Which bins of a *size*-sample frame lie in the band, DC left out, and
    how many hertz apart they are."""
    freqs = np.fft.rfftfreq(size, 1.0 / sr)
    return (freqs >= max(low, freqs[1])) & (freqs <= high), float(freqs[1])


def _zcr(frames: np.ndarray) -> np.ndarray:
    return np.mean(np.diff(np.signbit(frames), axis=-1), axis=-1)


def _merge(spans: List[Tuple[float, float]], gap: float
           ) -> List[Tuple[float, float]]:
    """Join spans separated by less than *gap* seconds."""
    if not spans:
        return []
    merged = [list(spans[0])]
    for start, end in spans[1:]:
        if start - merged[-1][1] <= gap:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    return [(a, b) for a, b in merged]


def vad(y: np.ndarray, sr: int, *,
        threshold_db: float = THRESHOLD_DB,
        flatness_max: float = FLATNESS_MAX,
        min_speech: float = 0.10,
        min_silence: float = 0.15,
        pad: float = 0.05,
        frame: float = FRAME,
        hop: float = HOP,
        band: Optional[Tuple[float, float]] = None) -> List[Tuple[float, float]]:
    """Find the spans of *y* that contain speech, in **seconds**.

    ```
    >>> spans = kudio.vad(y, sr)
    >>> spans[:2]
    [(0.31, 1.84), (2.42, 3.97)]
    ```

    A clip of any length is judged, and a short one by the same rules, which
    leave it less to go by. The noise floor is the level its own quietest 15%
    of frames stay under, so speech is found where the clip holds something
    quieter as well: one cut to the word, speech from end to end, comes back
    empty. So does a clip shorter than *min_speech*, and one that holds a
    single frame (under 40 ms by default), which is its own floor.

    :param threshold_db: how far above the estimated noise floor a frame must
        sit. Lower it for a very quiet talker, raise it for a noisy room. ``-inf``
        lets every frame through on its level, and ``inf`` none.
    :param flatness_max: spectral flatness above which a loud frame is judged
        to be noise rather than voice. Raise it towards 1.0 to disable that
        test — useful for non-speech audio, where "harmonic" means nothing.
        ``inf`` disables it as well, and ``-inf`` judges every frame noise.
    :param min_speech: segments shorter than this are dropped (a click, a door).
    :param min_silence: gaps shorter than this do not split a segment; a stop
        consonant is a real silence in the middle of a word. ``inf`` splits at
        no gap: the speech is one segment from the first frame to the last.
    :param pad: seconds added to each end, so the cut misses the consonant.
    :param band: ``(low, high)`` in Hz to judge the frames on, band-passed to
        it. In a recording made at 192 kHz a floor above the speech band can
        hold most of each frame's energy, and a quiet voice no longer stands
        the *threshold_db* above it a frame needs; ``(100, 4000)`` judges it
        where speech is. ``None``, the default, takes the whole band.
    :returns: non-overlapping ``(start, end)`` pairs, in order. Empty when
        nothing in the clip looks like speech.

    :raises FeatureError: on non-mono audio, samples that are not finite, a
        rate that is not a positive number, a setting that is not a number, a
        duration below 0, or a *band* that is not ``low < high`` with *low*
        under Nyquist. A clip with NaN or inf in it is
        refused, as :func:`kudio.f0` and :func:`kudio.spectral_enhance` refuse
        it, rather than judged with those frames left out.
    """
    sr = sample_rate(sr, 'vad', FeatureError)
    threshold_db = within(threshold_db, 'threshold_db', 'vad', FeatureError, of=' of dB', inf=True)
    flatness_max = within(flatness_max, 'flatness_max', 'vad', FeatureError, inf=True)
    min_speech = within(min_speech, 'min_speech', 'vad', FeatureError, low=0, of=' of seconds')
    min_silence = within(min_silence, 'min_silence', 'vad', FeatureError, low=0,
                         of=' of seconds', inf=True)
    pad = within(pad, 'pad', 'vad', FeatureError, low=0, of=' of seconds')
    frame = within(frame, 'frame', 'vad', FeatureError, above=0, of=' of seconds')
    hop = within(hop, 'hop', 'vad', FeatureError, above=0, of=' of seconds')
    y = one_channel(y, 'vad', FeatureError)
    y = np.ascontiguousarray(np.asarray(y, dtype=np.float64).squeeze())
    if y.ndim != 1:
        raise FeatureError(f"vad() takes mono audio, got shape {y.shape}")
    if y.size == 0:
        return []
    if not np.all(np.isfinite(y)):
        raise FeatureError(f"vad() got {int(np.count_nonzero(~np.isfinite(y)))} "
                           f"sample(s) that are not finite (NaN or inf); repair "
                           f"the clip first")
    if y.size == 1:                            # DC alone: no spectrum to judge
        return []

    keep = None
    if band is not None:
        try:
            low, high = (real(v) for v in band)
        except (TypeError, ValueError):                    # not two, or not any
            low = high = None
        if low is None or high is None or not 0.0 <= low < high or low >= sr / 2.0:
            raise FeatureError(f"vad() needs a band with 0 <= low < high and low under "
                               f"{sr / 2.0:g} Hz, got {band}")
        from kudio.effects.filters import bandpass, highpass, lowpass
        if high >= sr / 2.0:
            y = highpass(y, sr, cutoff=low) if low > 0 else y
        elif low > 0:
            y = bandpass(y, sr, low=low, high=high)
        else:
            y = lowpass(y, sr, cutoff=high)

    size = max(16, int(round(frame * sr)))
    step = max(1, int(round(hop * sr)))
    frames = frame_view(y, size, step)
    if band is not None:
        keep, apart = _band_bins(size, sr, low, high)
        if np.count_nonzero(keep) < 2:
            raise FeatureError(f"vad() band {band} holds {np.count_nonzero(keep)} of the "
                               f"bins of a {size}-sample frame, {apart:.1f} Hz apart; "
                               f"widen it")
        if frames.shape[1] < size:             # a clip under one frame has its own bins
            keep, _ = _band_bins(frames.shape[1], sr, low, high)
            if np.count_nonzero(keep) < 2:     # nothing to be flat or peaky across
                return []
    n_frames = frames.shape[0]
    duration = len(y) / sr

    rms = np.sqrt(np.mean(frames ** 2, axis=-1))
    with np.errstate(divide='ignore'):
        db = 20.0 * np.log10(np.maximum(rms, 0.0))
    finite = db[np.isfinite(db)]
    if finite.size == 0:                       # digital silence
        return []

    noise_floor = float(np.percentile(finite, _NOISE_PERCENTILE))
    loudest = float(finite.max())
    # two gates: above the room, and not absurdly far under the loudest moment.
    # The second stops a dead-quiet file from having its own hiss promoted to
    # speech simply because the hiss is 8 dB above the quietest hiss.
    active = (db > noise_floor + threshold_db) & (db > loudest - FLOOR_REL_DB)

    # what is loud *and* harmonic is the only thing trusted on its own
    core = active & (_flatness(frames, keep) < flatness_max)
    # a hissy frame joins only if it is next to one of those. Zero-crossing
    # rate cannot tell an /s/ from a burst of white noise -- position can.
    reach = int(min(max(1, round(FRICATIVE_REACH / hop)), n_frames))
    near_core = convolve_same(core.astype(np.float64), np.ones(2 * reach + 1)) > 0
    speech = core | (active & (_zcr(frames) > ZCR_MIN) & near_core)

    if not speech.any():
        return []

    # frame index -> the seconds it covers
    edges = np.diff(np.concatenate(([0], speech.view(np.int8), [0])))
    starts = np.nonzero(edges == 1)[0]
    stops = np.nonzero(edges == -1)[0]
    spans = [(float(a * step) / sr,
              float(min((b - 1) * step + size, len(y))) / sr)
             for a, b in zip(starts, stops)]
    if n_frames == 1:                          # clip shorter than one frame
        spans = [(0.0, duration)] if speech[0] else []

    spans = _merge(spans, min_silence)
    spans = [s for s in spans if s[1] - s[0] >= min_speech]
    if pad > 0:
        spans = [(max(0.0, a - pad), min(duration, b + pad)) for a, b in spans]
        spans = _merge(spans, 0.0)
    return spans


def vad_split(y: np.ndarray, sr: int, **kwargs) -> List[np.ndarray]:
    """The speech segments themselves, as waveforms.

    The noise-adaptive counterpart of :func:`kudio.split_on_silence`; takes the
    same keyword arguments as :func:`vad`. Each segment is a copy.
    """
    sr = sample_rate(sr, 'vad_split', FeatureError)
    y = channels_last(y, 'vad_split', FeatureError)
    return [y[int(round(a * sr)):int(round(b * sr))].copy()
            for a, b in vad(y, sr, **kwargs)]


def vad_trim(y: np.ndarray, sr: int, **kwargs
             ) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Trim to the first and last speech, keeping whatever is between them.

    Returns ``(trimmed, (start_sample, end_sample))``, the waveform a copy, as
    :func:`kudio.trim_silence` does. A clip with no detected speech comes back
    whole with its full span, rather than as an empty array — deleting the
    audio is never the safer reading of "nothing found".
    """
    sr = sample_rate(sr, 'vad_trim', FeatureError)
    y = channels_last(y, 'vad_trim', FeatureError)
    spans = vad(y, sr, **kwargs)
    if not spans:
        log.info("vad_trim: no speech detected, returning the clip unchanged")
        return y.copy(), (0, int(y.shape[0]))
    start = int(round(spans[0][0] * sr))
    end = min(int(round(spans[-1][1] * sr)), y.shape[0])
    return y[start:end].copy(), (start, end)


def vad_frames(y: np.ndarray, sr: int, *, hop_length: int = 256,
               n_fft: int = 512, center: bool = True, pad: float = 0.0,
               **kwargs) -> np.ndarray:
    """:func:`vad` as one decision per STFT frame, ``True`` for speech.

    A frame is speech when its centre falls inside a span :func:`vad` found.
    With the default ``center=True`` the frames are the ones
    :func:`kudio.waveform_to_spectrogram` and :class:`kudio.STFT` make with
    the same *hop_length* and *n_fft*, both of which centre their frames, so
    the mask indexes their rows directly::

        >>> spec = kudio.waveform_to_spectrogram(y)       # (frames, bins)
        >>> speech = kudio.vad_frames(y, sr)              # (frames,) bool
        >>> noise_only = spec[~speech]

    :param pad: seconds added to each end of a span, as in :func:`vad` --
        but 0 by default here rather than 0.05. Padding keeps a cut from
        clipping a consonant; on a frame label it only marks the frames next
        to the speech as speech too.
    :param center: ``True`` (librosa's default and kudio's) centres frame *t*
        on sample ``t * hop_length``; ``False`` starts it there.

    Other keyword arguments go to :func:`vad`.
    """
    center = flag(center, 'center', 'vad_frames', FeatureError)
    hop_length = count(hop_length, 'hop_length', 'vad_frames', FeatureError, ' of samples')
    n_fft = count(n_fft, 'n_fft', 'vad_frames', FeatureError, ' of samples')
    sr = sample_rate(sr, 'vad_frames', FeatureError)
    spans = vad(y, sr, pad=pad, **kwargs)
    n = np.asarray(y).size                     # vad takes one channel, whichever way up
    if center:
        # the count librosa gives after padding n_fft // 2 each side, which is
        # one frame short of 1 + n // hop for an odd n_fft
        n_frames = 1 + (n + 2 * (n_fft // 2) - n_fft) // hop_length
        centres = np.arange(n_frames) * hop_length
    else:
        n_frames = 1 + (n - n_fft) // hop_length if n >= n_fft else 0
        centres = np.arange(n_frames) * hop_length + n_fft / 2
    seconds = centres / float(sr)
    speech = np.zeros(len(centres), dtype=bool)
    for start, end in spans:
        speech |= (seconds >= start) & (seconds < end)
    return speech


def speech_ratio(y: np.ndarray, sr: int, **kwargs) -> float:
    """Fraction of the clip that :func:`vad` calls speech, in ``[0, 1]``.

    Useful as a dataset filter: a "speech" file at 0.02 is usually a
    mislabelled recording of a room. The floor :func:`vad` measures against
    comes from the quietest 15% of the clip, though, so a clip cut tight
    around speech, with no room tone left in it, reads near 0 as well.
    """
    sr = sample_rate(sr, 'speech_ratio', FeatureError)
    y = channels_last(y, 'speech_ratio', FeatureError)
    if y.size == 0:
        return 0.0
    total = sum(b - a for a, b in vad(y, sr, **kwargs))
    return float(min(1.0, total / (y.shape[0] / sr)))
