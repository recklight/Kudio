# -*- coding: utf-8 -*-
"""A spectrogram for audio that has not finished arriving.

:class:`kudio.STFT` and :func:`kudio.waveform_to_spectrogram` want the whole
clip. That is right for a file and impossible for a microphone, and the usual
workaround -- keep the last few seconds in a list and re-run the STFT over all
of it several times a second -- is wasteful in a way that shows: at 16 kHz with
a 3-second window, twenty redraws a second recompute about 240 columns to gain
three, and 99% of that work redraws what was already on screen.

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

**Values are dBFS.** Amplitude is normalised by the window, so a full-scale
sine reads 0 dB whatever the frame size and window are. The number on screen
then means the same thing after someone changes ``n_fft`` -- which it does not
if the scaling is left to the FFT.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from kudio.exceptions import FeatureError

__all__ = ['SpectrogramStream']

log = logging.getLogger(__name__)

_EPS = 1e-20


class SpectrogramStream:
    """A rolling log-magnitude spectrogram, one column per hop.

    :param sr: sample rate of the audio that will be pushed.
    :param n_fft: analysis frame, in samples. Also the latency.
    :param hop_length: samples between columns; defaults to ``n_fft // 4``.
    :param window: any window name :func:`scipy.signal.get_window` accepts.
    :param seconds: how much history to keep on screen. Rounded to a whole
        number of columns, at least one.
    :param top_db: dynamic range. Anything quieter than *top_db* below the
        reference is clamped to the floor, so an empty room draws as one flat
        colour instead of amplified dither.
    :param ref: amplitude that reads 0 dB. ``1.0`` is dBFS.
    :param n_mels: 0 for a linear frequency axis, or a band count for a mel
        one. Speech at 16 kHz spends most of its energy in the bottom eighth
        of a linear axis; a mel axis spends half the picture there.
    :param fmin: lowest mel band, ignored when *n_mels* is 0.
    :param fmax: highest mel band; Nyquist when omitted.
    """

    def __init__(self, sr: int, *, n_fft: int = 512,
                 hop_length: Optional[int] = None, window: str = 'hann',
                 seconds: float = 3.0, top_db: float = 80.0, ref: float = 1.0,
                 n_mels: int = 0, fmin: float = 0.0,
                 fmax: Optional[float] = None):
        if sr <= 0:
            raise FeatureError(f"SpectrogramStream needs a positive rate, got {sr}")
        if n_fft < 2:
            raise FeatureError(f"n_fft must be >= 2, got {n_fft}")
        hop = max(1, int(n_fft) // 4) if hop_length is None else int(hop_length)
        if hop < 1:
            raise FeatureError(f"hop_length must be >= 1, got {hop_length}")
        if seconds <= 0:
            raise FeatureError(f"seconds must be positive, got {seconds}")
        if top_db <= 0:
            raise FeatureError(f"top_db must be positive, got {top_db}")
        if ref <= 0:
            raise FeatureError(f"ref must be positive, got {ref}")

        self.sr = int(sr)
        self.n_fft = int(n_fft)
        self.hop_length = hop
        self.window = window
        self.top_db = float(top_db)
        self.ref = float(ref)
        self.n_mels = int(n_mels or 0)
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
            import librosa
            self._mel_fb = np.asarray(librosa.filters.mel(
                sr=self.sr, n_fft=self.n_fft, n_mels=self.n_mels,
                fmin=self.fmin, fmax=self.fmax), dtype=np.float64)
            self._freqs = np.asarray(librosa.mel_frequencies(
                n_mels=self.n_mels, fmin=self.fmin, fmax=self.fmax),
                dtype=np.float64)
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

    def to_stft(self):
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
        """
        block = np.asarray(block)
        if block.ndim > 1:
            raise FeatureError(
                f"SpectrogramStream takes mono audio, got shape {block.shape}. "
                f"Mix down first (y.mean(axis=1)) -- flattening interleaved "
                f"channels would analyse a signal that does not exist.")
        block = block.astype(np.float64, copy=False).reshape(-1)
        if block.size:
            self._pending = np.concatenate([self._pending, block])

        columns = []
        while len(self._pending) >= self.n_fft:
            columns.append(self._column(self._pending[:self.n_fft]))
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
