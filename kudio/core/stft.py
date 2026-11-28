# -*- coding: utf-8 -*-
"""STFT geometry as a value you can pass around and save.

A spectrogram and the waveform reconstructed from it must be computed with the
same ``n_fft`` / ``hop_length`` / ``win_length`` / ``window``. Passing those
four to two separate functions and keeping them in sync by hand is how a
producer and a consumer end up quietly disagreeing — the geometry belongs
*with* the data that was made from it.

>>> stft = kudio.STFT(sr=16000, n_fft=512, hop_length=256)
>>> spec = stft.forward(y)                 # (frames, bins)
>>> back = stft.inverse(y, spec)           # phase taken from y
>>> stft.to_dict()                         # store next to a model checkpoint
{'sr': 16000, 'n_fft': 512, 'hop_length': 256, 'win_length': 512, 'window': 'hamming'}

A thin wrapper: :func:`waveform_to_spectrogram` and
:func:`spectrogram_to_waveform` do the work.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from kudio._numbers import whole
from kudio.core import _spectral
from kudio.core._framing import overlapped
from kudio.core._policy import sample_rate
from kudio.exceptions import FeatureError

__all__ = ['STFT']


@dataclass(frozen=True)
class STFT:
    """Immutable STFT settings, with the two transforms attached.

    :param sr: sample rate the settings were chosen for. Carried so a stored
        geometry can be checked against the audio it is applied to; the
        transforms themselves do not resample.
    :param win_length: defaults to *n_fft*.
    :raises ValueError: for an *sr* that is not a positive rate, and for a
        size that is not a whole number of 1 or more.
    """

    sr: int = 16000
    n_fft: int = 512
    hop_length: int = 256
    win_length: Optional[int] = None
    window: str = 'hamming'

    def __post_init__(self) -> None:
        object.__setattr__(self, 'sr', sample_rate(self.sr, 'STFT'))
        if self.win_length is None:
            object.__setattr__(self, 'win_length', self.n_fft)
        for name in ('n_fft', 'hop_length', 'win_length'):
            size = whole(getattr(self, name))
            if size is None or size < 1:
                raise ValueError(f"STFT(): {name} must be >= 1, a whole number of "
                                 f"samples, got {getattr(self, name)!r}")
            object.__setattr__(self, name, size)
        if self.win_length > self.n_fft:
            raise ValueError(f"win_length ({self.win_length}) cannot exceed "
                             f"n_fft ({self.n_fft})")

    # --------------------------------------------------------------- shape

    @property
    def n_bins(self) -> int:
        """Frequency bins produced by this geometry."""
        return self.n_fft // 2 + 1

    def n_frames(self, n_samples: int) -> int:
        """Frames a waveform of *n_samples* will produce (centred STFT).

        There are ``n_fft // 2`` zeros on each side of it, so an odd *n_fft* gives
        one frame fewer than ``n // hop + 1`` whenever *n* is a multiple of the
        hop.
        """
        padded = int(n_samples) + 2 * (self.n_fft // 2)
        return 1 + (padded - self.n_fft) // self.hop_length

    def frame_times(self, n_frames: int) -> np.ndarray:
        """Centre time, in seconds, of each frame."""
        return np.arange(n_frames) * self.hop_length / self.sr

    # ---------------------------------------------------------- transforms

    def forward(self, y: np.ndarray, *, norm: bool = False,
                sequence: bool = False, context: int = 0,
                desired_length: Optional[int] = None) -> np.ndarray:
        """Waveform -> log-power spectrogram ``(frames, bins)``."""
        from kudio.core.feature import waveform_to_spectrogram
        return waveform_to_spectrogram(
            y, sequence=sequence, forward_backward=context, norm=norm,
            hop_length=self.hop_length, n_fft=self.n_fft,
            win_length=self.win_length, window=self.window,
            desired_length=desired_length)

    def inverse(self, y_ref: np.ndarray, spec: np.ndarray, *,
                sequence: bool = False) -> np.ndarray:
        """Log-power spectrogram -> waveform, reusing the phase of *y_ref*."""
        from kudio.core.feature import spectrogram_to_waveform
        return spectrogram_to_waveform(
            y_ref, spec, sequence=sequence, hop_length=self.hop_length,
            n_fft=self.n_fft, win_length=self.win_length, window=self.window)

    def analyse(self, y: np.ndarray) -> np.ndarray:
        """Waveform -> **complex** spectrogram ``(frames, bins)``.

        :meth:`forward` throws the phase away, which is right for a feature and
        wrong for anything that has to reconstruct the signal. Enhancement
        multiplies the complex spectrum by a real gain and inverts it, so it
        needs the phase kept — see :func:`kudio.spectral_enhance`.

        :raises FeatureError: for anything but a mono waveform of finite samples.
        """
        return self._analyse(y, "STFT.analyse")

    def _analyse(self, y: np.ndarray, where: str) -> np.ndarray:
        """:meth:`analyse`, under the name of the function it is done for."""
        from kudio.core.feature import _clip
        y = np.asarray(_clip(y, where), dtype=np.float32)
        return _spectral.stft(y, self.n_fft, self.hop_length, self.win_length, self.window,
                              where).T                # (frames, bins), as kudio speaks

    def synthesise(self, spec: np.ndarray,
                   length: Optional[int] = None) -> np.ndarray:
        """Complex spectrogram -> waveform, the inverse of :meth:`analyse`.

        *length* trims (or pads) to an exact sample count. Pass the input
        length and the round trip is sample-aligned, which is what an A/B or a
        training target needs.

        :raises FeatureError: for anything but a complex ``(frames, bins)`` of
            at least one frame and finite values, for a geometry whose hop steps
            over samples no window covers, and for a *length* that is not a whole
            number of samples.
        """
        spec = np.asarray(spec)
        if spec.ndim != 2 or spec.shape[1] != self.n_bins or not len(spec):
            raise FeatureError(f"STFT.synthesise() takes (frames, {self.n_bins}), the "
                               f"layout analyse() gives, and got shape {spec.shape}")
        if not np.iscomplexobj(spec):
            raise FeatureError(f"STFT.synthesise() takes the complex spectrum analyse() "
                               f"gives, and got {spec.dtype}: a magnitude has no phase "
                               f"to rebuild a waveform from")
        if not np.all(self._overlap() > 1e-10):     # scipy.signal.check_NOLA's bound
            raise FeatureError(f"STFT.synthesise(): with hop_length={self.hop_length} some "
                               f"samples are under no part of a {self.win_length}-sample "
                               f"{self.window!r} window, so nothing is left to rebuild "
                               f"them from")
        if length is not None:
            samples = whole(length)
            if samples is None or samples < 0:
                raise FeatureError(f"STFT.synthesise(): length is a whole number of samples, 0 "
                                   f"or more, got {length!r}")
            length = samples
        bad = int(np.count_nonzero(~np.isfinite(spec)))
        if bad:
            raise FeatureError(f"STFT.synthesise() got {bad} value(s) that are not finite (NaN "
                               f"or inf)")
        out = _spectral.istft(spec.T, self.n_fft, self.hop_length, self.win_length, self.window,
                              length, "STFT.synthesise")
        return np.asarray(out, dtype=np.float32)

    def _window_energy(self) -> np.ndarray:
        """The window squared, centred in the frame."""
        return _spectral._centred(
            _spectral.window(self.window, self.win_length, "STFT") ** 2, self.n_fft)

    def _overlap(self) -> np.ndarray:
        """The window energy over a sample in the middle of a clip, which the
        inverse divides by."""
        return overlapped(self._window_energy(), self.hop_length)

    # ------------------------------------------------------------ storage

    def to_dict(self) -> Dict[str, Any]:
        """Plain dict, ready for JSON/YAML next to a checkpoint."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "STFT":
        """Rebuild the geometry :meth:`to_dict` stored.

        Every key has to be there. A missing one taking its default is how a
        stored geometry and the code reading it end up disagreeing -- a config
        without ``hop_length`` would quietly analyse at 256.
        """
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(values) - known
        if unknown:
            raise ValueError(
                f"unknown STFT key(s): {', '.join(sorted(unknown))}. "
                f"Valid keys: {', '.join(sorted(known))}")
        missing = known - set(values)
        if missing:
            raise ValueError(
                f"STFT key(s) missing: {', '.join(sorted(missing))}. A stored "
                f"geometry has to name all of {', '.join(sorted(known))}")
        return cls(**values)

    def matches(self, sr: int) -> bool:
        """Is *sr* the rate this geometry was configured for?"""
        return sample_rate(sr, 'STFT.matches') == self.sr
