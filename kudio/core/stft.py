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

This is a thin wrapper: :func:`waveform_to_spectrogram` and
:func:`spectrogram_to_waveform` remain the underlying implementation and are
unchanged.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

__all__ = ['STFT']


@dataclass(frozen=True)
class STFT:
    """Immutable STFT settings, with the two transforms attached.

    :param sr: sample rate the settings were chosen for. Carried so a stored
        geometry can be checked against the audio it is applied to; the
        transforms themselves do not resample.
    :param win_length: defaults to *n_fft*.
    """

    sr: int = 16000
    n_fft: int = 512
    hop_length: int = 256
    win_length: Optional[int] = None
    window: str = 'hamming'

    def __post_init__(self) -> None:
        if self.win_length is None:
            object.__setattr__(self, 'win_length', self.n_fft)
        if self.win_length > self.n_fft:
            raise ValueError(f"win_length ({self.win_length}) cannot exceed "
                             f"n_fft ({self.n_fft})")
        if self.hop_length < 1:
            raise ValueError(f"hop_length must be >= 1, got {self.hop_length}")

    # --------------------------------------------------------------- shape

    @property
    def n_bins(self) -> int:
        """Frequency bins produced by this geometry."""
        return self.n_fft // 2 + 1

    def n_frames(self, n_samples: int) -> int:
        """Frames a waveform of *n_samples* will produce (centred STFT)."""
        return int(n_samples // self.hop_length) + 1

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

    # ------------------------------------------------------------ storage

    def to_dict(self) -> Dict[str, Any]:
        """Plain dict, ready for JSON/YAML next to a checkpoint."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "STFT":
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(values) - known
        if unknown:
            raise ValueError(
                f"unknown STFT key(s): {', '.join(sorted(unknown))}. "
                f"Valid keys: {', '.join(sorted(known))}")
        return cls(**values)

    def matches(self, sr: int) -> bool:
        """Is *sr* the rate this geometry was configured for?"""
        return int(sr) == int(self.sr)
