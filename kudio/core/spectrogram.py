# -*- coding: utf-8 -*-
"""A spectrogram for audio that has not finished arriving.

:class:`kudio.STFT` and :func:`kudio.waveform_to_spectrogram` want the whole
clip. That is right for a file and impossible for a microphone, and the usual
workaround -- keep the last few seconds in a list and re-run the STFT over all
of it several times a second -- is wasteful in a way that shows: at 16 kHz with
a 3-second window and the default geometry, twenty redraws a second recompute
375 columns to gain about six, and 98% of that work redraws what was already
on screen.

:class:`SpectrogramStream` computes each column **once**, when the samples for
it have arrived, and keeps the last *N* in a ring buffer. Feeding it a stream
in whatever block sizes the device happens to deliver produces exactly the
columns the offline transform would.

>>> spec = kudio.SpectrogramStream(sr=16000, seconds=3.0)
>>> while recorder.recording:                            # doctest: +SKIP
...     spec.push(recorder.drain())
...     draw(spec.columns())                             # (columns, bins), dB

**Columns are not centred.** A file transform pads the start so that column
*k* is centred on sample *k x hop*; a stream cannot pad audio it has not heard
yet, so column *k* here covers samples ``[k*hop, k*hop + n_fft)`` -- the
``center=False`` convention. The first column therefore needs a whole frame
before it exists, which is :attr:`~SpectrogramStream.latency_seconds`.

**Values are dBFS** on a linear axis. Amplitude is normalised by the window,
so a full-scale sine reads 0 dB whatever the frame size and window are. A mel
axis sums bins through filters of area 1, so there the number is band power in
dB, not dBFS: a full-scale sine reads about -18 dB at 1 kHz. The number on screen
then means the same thing after someone changes ``n_fft`` -- which it does not
if the scaling is left to the FFT.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import numpy as np

from kudio._numbers import count, real, whole
from kudio._samples import finite, one_channel
from kudio.core import _mel
from kudio.core._policy import sample_rate
from kudio.exceptions import FeatureError

if TYPE_CHECKING:
    from kudio.core.stft import STFT

__all__ = ['SpectrogramStream']

log = logging.getLogger(__name__)

_EPS = 1e-20


class SpectrogramStream:
    """A rolling log-magnitude spectrogram, one column per hop.

    :param sr: sample rate of the audio that will be pushed.
    :param n_fft: analysis frame, in samples. Also the latency.
    :param hop_length: samples between columns; ``n_fft // 4`` when omitted. One
        longer than *n_fft* leaves samples out between columns.
    :param window: any window name :func:`scipy.signal.get_window` accepts.
    :param seconds: how much history to keep on screen. Rounded to a whole
        number of columns, at least one.
    :param top_db: dynamic range. Anything quieter than *top_db* below the
        reference is clamped to the floor, so an empty room draws as one flat
        colour instead of amplified dither.
    :param ref: amplitude that reads 0 dB. ``1.0`` is dBFS.
    :param n_mels: 0 for a linear frequency axis, or a band count for a mel
        one. Speech at 16 kHz spends most of its energy in the bottom eighth
        of a linear axis; 64 mel bands spend about a third of the picture
        below 1 kHz.
    :param fmin: where the lowest mel band starts, ignored when *n_mels* is 0.
    :param fmax: where the highest one ends; Nyquist when omitted.
    :raises FeatureError: for a setting that is not a number, or out of its range.
    """

    def __init__(self, sr: int, *, n_fft: int = 512,
                 hop_length: Optional[int] = None, window: str = 'hann',
                 seconds: float = 3.0, top_db: float = 80.0, ref: float = 1.0,
                 n_mels: int = 0, fmin: float = 0.0,
                 fmax: Optional[float] = None):
        where = 'SpectrogramStream'
        sr = sample_rate(sr, where, FeatureError)
        n_fft = count(n_fft, 'n_fft', where, FeatureError, ' of samples')
        if n_fft < 2:
            raise FeatureError(f"{where}(): n_fft must be >= 2, got {n_fft}")
        hop = max(1, n_fft // 4) if hop_length is None else \
            count(hop_length, 'hop_length', where, FeatureError, ' of samples')
        for name, value in (('seconds', seconds), ('top_db', top_db), ('ref', ref)):
            number = real(value)
            if number is None or number <= 0:
                raise FeatureError(f"{where}(): {name} must be positive, a finite number, got "
                                   f"{value!r}")

        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop
        self.window = window
        self.top_db = float(top_db)
        self.ref = float(ref)
        # 0 and None are a linear axis
        bands = 0 if n_mels is None else whole(n_mels)
        if bands is None or bands < 0:
            raise FeatureError(f"{where}(): n_mels is a whole number of mel bands, or 0 for a "
                               f"linear axis, got {n_mels!r}")
        self.n_mels = bands
        if self.n_mels:
            from kudio.core.feature import _limits
            self.fmin, self.fmax = _limits(fmin, fmax, self.sr, where)
        else:
            self.fmin = float(fmin)
            self.fmax = float(fmax) if fmax is not None else self.sr / 2.0

        self._window = _get_window(window, self.n_fft)
        wsum = float(np.sum(self._window)) or 1.0
        # Single-sided amplitude scaling. Every bin except DC and Nyquist has a
        # mirror holding half the energy, so those get the factor of two and
        # those two do not. Without it the level printed beside a peak depends
        # on n_fft and on the window, which makes it unreadable *as a level* --
        # and a meter you cannot read is worse than no meter.
        self._scale = np.full(self.n_fft // 2 + 1, 2.0 / wsum)
        self._scale[0] = 1.0 / wsum
        if self.n_fft % 2 == 0:
            self._scale[-1] = 1.0 / wsum

        self._mel_fb = None
        if self.n_mels:
            self._mel_fb = _mel.filters(self.sr, self.n_fft, self.n_mels, self.fmin, self.fmax,
                                        where).astype(np.float64)
            self._freqs = _mel.centres(self.n_mels, self.fmin, self.fmax)
        else:
            self._freqs = np.fft.rfftfreq(self.n_fft, 1.0 / self.sr)

        self.n_columns = max(1, int(round(seconds * self.sr / self.hop_length)))
        self.reset()

    # -------------------------------------------------------------- geometry

    @property
    def n_bins(self) -> int:
        """Rows in a column: mel bands, or linear frequency bins."""
        return self.n_mels or (self.n_fft // 2 + 1)

    @property
    def freqs(self) -> np.ndarray:
        """Centre frequency of each row, in Hz. Mel bands included."""
        return self._freqs

    @property
    def floor_db(self) -> float:
        """What an empty column holds, and the clamp applied to every other."""
        return -self.top_db

    @property
    def latency_seconds(self) -> float:
        """How long before the first column exists: one analysis frame."""
        return self.n_fft / self.sr

    @property
    def column_seconds(self) -> float:
        """Time between columns."""
        return self.hop_length / self.sr

    @property
    def window_seconds(self) -> float:
        """Span of :meth:`columns` -- the width of the picture, in seconds."""
        return self.n_columns * self.hop_length / self.sr

    @property
    def frames_seen(self) -> int:
        """Columns computed since the last :meth:`reset`, including the ones
        that have already scrolled off."""
        return self._frames

    @property
    def filled(self) -> int:
        """Columns of real audio in the window. Below :attr:`n_columns` only
        while the display is still filling up."""
        return self._filled

    def to_stft(self) -> STFT:
        """The equivalent :class:`kudio.STFT` geometry.

        Useful for recording what a live view was computed with, or for
        reproducing a moment from the file afterwards -- with the caveat at the
        top of this module: the offline transform centres its frames and this
        does not.
        """
        from kudio.core.stft import STFT
        return STFT(sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length,
                    win_length=self.n_fft, window=self.window)

    # ------------------------------------------------------------------ feed

    def reset(self) -> None:
        """Empty the window and the part-filled frame, keeping the settings."""
        self._buffer = np.full((self.n_columns, self.n_bins), self.floor_db,
                               dtype=np.float32)
        self._write = 0
        self._filled = 0
        self._pending = np.zeros(0, dtype=np.float64)
        self._skip = 0
        self._frames = 0

    def push(self, block: np.ndarray) -> np.ndarray:
        """Feed newly-arrived audio in; get the columns it completed back.

        Blocks may be any size, including zero: samples that do not fill a
        frame are held until the rest of it arrives, so the answer never
        depends on how the device chose to chop the stream up.

        Feed this :meth:`kudio.StreamRecorder.drain`, **not**
        :meth:`~kudio.StreamRecorder.tail`. The tail overlaps between calls,
        and every overlapping sample would be drawn twice, at the wrong time.

        :returns: ``(new_columns, bins)`` in dB, oldest first. Empty when the
            block did not complete a frame. More than :attr:`n_columns` of them
            when the block was longer than the window -- all are returned, only
            the last :attr:`n_columns` are kept.
        :raises FeatureError: for a block that is not mono, or holds NaN or inf.
            The window is as it was before it.
        """
        block = one_channel(block, 'SpectrogramStream.push', FeatureError)
        if block.ndim > 1:
            raise FeatureError(
                f"SpectrogramStream takes mono audio, got shape {block.shape}. "
                f"Mix down first (y.mean(axis=1)) -- flattening interleaved "
                f"channels would analyse a signal that does not exist.")
        block = finite(block, 'SpectrogramStream.push', FeatureError)
        block = block.astype(np.float64, copy=False).reshape(-1)
        # a hop longer than the frame: what lies between two columns and had not
        # arrived with the first of them
        between = min(self._skip, block.size)
        self._skip -= between
        if block.size > between:
            self._pending = np.concatenate([self._pending, block[between:]])

        columns = []
        while len(self._pending) >= self.n_fft:
            columns.append(self._column(self._pending[:self.n_fft]))
            self._skip = max(0, self.hop_length - len(self._pending))
            self._pending = self._pending[self.hop_length:]
        if not columns:
            return np.zeros((0, self.n_bins), dtype=np.float32)

        new = np.asarray(columns, dtype=np.float32)
        self._store(new)
        self._frames += len(new)
        return new

    def columns(self) -> np.ndarray:
        """The window as an image: ``(n_columns, bins)``, newest on the right.

        Always the full width, floor-filled on the left until enough audio has
        arrived. A display that grows from nothing rescales its own time axis
        on every frame, which reads as the picture jittering rather than as the
        recording starting.
        """
        if self._filled >= self.n_columns:
            return np.concatenate([self._buffer[self._write:],
                                   self._buffer[:self._write]])
        out = np.full_like(self._buffer, self.floor_db)
        if self._filled:
            out[-self._filled:] = self._buffer[:self._filled]
        return out

    def latest(self) -> np.ndarray:
        """The most recent column, floor-filled if there is not one yet.

        A single column is a spectrum: this is what an instantaneous frequency
        readout draws.
        """
        if not self._filled:
            return np.full(self.n_bins, self.floor_db, dtype=np.float32)
        return self._buffer[(self._write - 1) % self.n_columns].copy()

    # -------------------------------------------------------------- internal

    def _column(self, samples: np.ndarray) -> np.ndarray:
        spectrum = np.fft.rfft(samples * self._window, n=self.n_fft)
        power = (np.abs(spectrum) * self._scale) ** 2
        if self._mel_fb is not None:
            power = self._mel_fb @ power
        db = 10.0 * np.log10(np.maximum(power, _EPS) / (self.ref ** 2))
        return np.maximum(db, self.floor_db)

    def _store(self, new: np.ndarray) -> None:
        """Write *new* into the ring, wrapping at most once."""
        if len(new) >= self.n_columns:
            self._buffer[:] = new[-self.n_columns:]
            self._write = 0
            self._filled = self.n_columns
            return
        end = self._write + len(new)
        if end <= self.n_columns:
            self._buffer[self._write:end] = new
        else:
            split = self.n_columns - self._write
            self._buffer[self._write:] = new[:split]
            self._buffer[:end - self.n_columns] = new[split:]
        self._write = end % self.n_columns
        self._filled = min(self.n_columns, self._filled + len(new))

    def __repr__(self) -> str:                        # pragma: no cover - repr
        axis = f"{self.n_mels} mel" if self.n_mels else f"{self.n_bins} bins"
        return (f"SpectrogramStream(sr={self.sr}, n_fft={self.n_fft}, "
                f"hop_length={self.hop_length}, {axis}, "
                f"{self.window_seconds:.2f}s window)")


def _get_window(name: str, size: int) -> np.ndarray:
    from scipy.signal import get_window
    return np.asarray(get_window(name, size, fftbins=True), dtype=np.float64)
