# -*- coding: utf-8 -*-
"""What the plots are made of: a waveform, a spectrogram, an axis of time, and the
curves of a spectrum's centre, its roll-off and the zero crossings.

Private, and for :mod:`kudio.util.visual`, which needs matplotlib. The spectrograms
and the curves are the numbers librosa gives for them, frames of 2048 samples and
512 from one to the next unless a plot says otherwise.
"""
from __future__ import annotations

import numpy as np

from kudio.core import _mel, _spectral
from kudio.exceptions import DependencyError

try:
    from matplotlib.ticker import Formatter, MaxNLocator, ScalarFormatter, SymmetricalLogLocator
except ImportError as _e:  # pragma: no cover - exercised only without [viz]
    raise DependencyError('matplotlib', extra='viz') from _e

__all__ = ['waveform', 'linear_spectrogram', 'mel_spectrogram', 'level', 'mel_level',
           'frame_times', 'centroid', 'rolloff', 'crossings']

#: samples in a frame of the curves and the mel bands, and from one to the next
SIZE, HOP = 2048, 512
#: how many columns an envelope is drawn in: more than a figure is pixels wide
_COLUMNS = 5000


# -- axes ----------------------------------------------------------------------------

class Clock(Formatter):
    """Seconds, with as many decimals as tell the ticks apart; from a minute on show
    ``m:ss``, and from an hour ``h:mm:ss``."""

    def __call__(self, x, pos=None):
        low, high = self.axis.get_view_interval()
        span = max(abs(low), abs(high))
        if span <= 60:
            ticks = self.axis.get_majorticklocs()
            step = np.min(np.diff(ticks)) if len(ticks) > 1 else 1.0
            places = max(0, int(np.ceil(-np.log10(step) - 1e-6)) + 1) if step > 0 else 0
            text = f"{x:.{places}f}"
            text = text.rstrip('0').rstrip('.') if '.' in text else text
            return '0' if text in ('-0', '') else text
        sign, whole = '-' if x < 0 else '', int(round(abs(x)))
        minutes, seconds = divmod(whole, 60)
        if span <= 3600:
            return f"{sign}{minutes}:{seconds:02d}"
        return f"{sign}{minutes // 60}:{minutes % 60:02d}:{seconds:02d}"


def _time_axis(ax, seconds: float) -> None:
    # ticks every 1, 1.5, 5 or 6 of a decade: 10, 15, 30 and 60 seconds among them
    ax.xaxis.set_major_locator(MaxNLocator(prune=None, steps=[1, 1.5, 5, 6, 10]))
    ax.xaxis.set_major_formatter(Clock())
    ax.set_xlabel('Time')
    if seconds > 0:
        ax.set_xlim(0, seconds)


# -- a waveform ----------------------------------------------------------------------

def waveform(ax, y: np.ndarray, sr: float, **style) -> None:
    """*y*, mono, on *ax* from 0 to its end: the samples as steps where there are few
    enough of them on show, and where there are not a stroke from the lowest to the
    highest of every few. Which it is goes by what the axes show, so zooming in comes
    to the samples.

    :param style: the line's: ``alpha``, ``color``, ``label``.
    """
    seconds = len(y) / sr
    line, = ax.plot([], [], **style)
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    ax.update_datalim([(0, -peak), (seconds, peak)])
    ax.autoscale_view()
    _time_axis(ax, seconds)

    def show(axes):
        low, high = sorted(axes.get_xlim())
        first = int(np.clip(np.floor(low * sr), 0, len(y)))
        stop = int(np.clip(np.ceil(high * sr) + 1, first, len(y)))
        if stop - first <= 2 * _COLUMNS:
            line.set_drawstyle('steps-post')
            line.set_data(np.arange(first, stop) / sr, y[first:stop])
        else:
            at, lowest, highest = envelope(y, first, stop, _COLUMNS)
            line.set_drawstyle('default')
            line.set_data(np.repeat(at, 2) / sr, np.stack([lowest, highest], axis=1).ravel())

    show(ax)
    ax.callbacks.connect('xlim_changed', show)


def envelope(y: np.ndarray, first: int, stop: int, columns: int):
    """``(at, lowest, highest)`` of *y* from *first* to *stop*, in so many columns at
    most: where each begins, and the two samples it has between."""
    size = max(1, -(-(stop - first) // columns))
    at = np.arange(first, stop, size)
    part = y[:stop]
    return at, np.minimum.reduceat(part, at), np.maximum.reduceat(part, at)


# -- spectrograms ----------------------------------------------------------------------

def _mesh(ax, db: np.ndarray, hertz: np.ndarray, sr: float, hop: int, seconds: float,
          cmap: str):
    """*db*, ``(bands, frames)``, between the edges *hertz*, one more than there are
    bands, each frame a hop wide around its time and the first and last to the ends of
    the clip."""
    at = np.clip((np.arange(db.shape[1] + 1) - 0.5) * hop / sr, 0, seconds)
    at[-1] = seconds
    mesh = ax.pcolormesh(at, hertz, db, cmap=cmap, shading='flat', rasterized=True)
    ax.set_ylim(hertz[0], hertz[-1])
    ax.set_ylabel('Hz')
    _time_axis(ax, seconds)
    return mesh


def linear_spectrogram(ax, db: np.ndarray, sr: float, hop: int, seconds: float,
                       cmap: str = 'magma'):
    """*db*, ``(bins, frames)``, over hertz up to half the rate."""
    edges = (np.arange(len(db) + 1) - 0.5) * sr / (2 * (len(db) - 1))
    return _mesh(ax, db, np.clip(edges, 0, sr / 2), sr, hop, seconds, cmap)


def mel_spectrogram(ax, db: np.ndarray, sr: float, fmax: float, seconds: float,
                    cmap: str = 'magma'):
    """*db*, ``(n_mels, frames)``, over hertz on the mel scale: linear up to 1 kHz and
    logarithmic above, every band from halfway to the one below to halfway to the one
    above."""
    centres = _mel.centres(len(db), 0.0, fmax)
    edges = np.concatenate([[0.0], (centres[1:] + centres[:-1]) / 2, [fmax]])
    mesh = _mesh(ax, db, edges, sr, HOP, seconds, cmap)
    ax.set_yscale('symlog', linthresh=1000.0, base=2)
    ax.yaxis.set_major_locator(SymmetricalLogLocator(ax.yaxis.get_transform()))
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.set_ylim(0, fmax)
    return mesh


def level(y: np.ndarray, n_fft: int = SIZE, hop: int = HOP, window: str = 'hann',
          ref=None, who: str = 'level') -> np.ndarray:
    """The magnitude spectrogram of *y* in dB, ``(bins, frames)``, over *ref* or over
    its own highest, down to 80 dB under that."""
    magnitude = np.abs(_spectral.stft(y, n_fft, hop, n_fft, window, who))
    top = magnitude.max() if ref is None else ref
    return _mel.decibels(np.square(magnitude), ref=np.square(top), range_db=80.0)


def mel_level(y: np.ndarray, sr: float, n_mels: int, fmax: float, who: str) -> np.ndarray:
    """The mel spectrogram of *y* in dB under its highest, down to -80."""
    bands = _mel.power(y, sr, SIZE, HOP, n_mels, fmax=fmax, who=who)
    return _mel.decibels(bands, ref=bands.max(), range_db=80.0)


# -- curves ----------------------------------------------------------------------------

def frame_times(n_frames: int, sr: float) -> np.ndarray:
    """The middle of each frame, in seconds."""
    return np.arange(n_frames) * HOP / sr


def _magnitudes(y: np.ndarray, who: str) -> np.ndarray:
    return np.abs(_spectral.stft(y, SIZE, HOP, SIZE, 'hann', who))


def centroid(y: np.ndarray, sr: float, who: str = 'centroid') -> np.ndarray:
    """Where the middle of each frame's spectrum is, in hertz: the mean of the bins by
    their magnitude. 0 in silence."""
    magnitude = _magnitudes(y, who)
    whole = np.sum(magnitude.astype(np.float64), axis=0)
    whole[whole < np.finfo(magnitude.dtype).tiny] = 1.0
    share = np.asarray(magnitude / whole, dtype=magnitude.dtype)
    return np.sum(np.fft.rfftfreq(SIZE, 1.0 / sr)[:, np.newaxis] * share, axis=0)


def rolloff(y: np.ndarray, sr: float, who: str = 'rolloff') -> np.ndarray:
    """The frequency under which 85 % of each frame's magnitude lies, in hertz."""
    total = np.cumsum(_magnitudes(y, who), axis=0)
    return np.fft.rfftfreq(SIZE, 1.0 / sr)[np.argmax(total >= 0.85 * total[-1], axis=0)]


def crossings(y: np.ndarray) -> np.ndarray:
    """How many of each frame's samples are on the other side of zero from the one
    before, as a share of the frame. Past its ends the clip stays where it was."""
    below = np.signbit(np.where(np.abs(y) <= 1e-10, 0, y))
    below = np.pad(below, SIZE // 2, mode='edge')
    counted = np.concatenate([[0, 0], np.cumsum(below[1:] != below[:-1])])
    first = np.arange(0, len(below) - SIZE + 1, HOP)
    return (counted[first + SIZE] - counted[first + 1]) / SIZE
