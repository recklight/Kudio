# -*- coding: utf-8 -*-
"""Exception hierarchy for kudio.

All library-raised errors derive from :class:`KudioError`, so callers can::

    try:
        kudio.file_load(path)
    except kudio.KudioError:
        ...
"""
from __future__ import annotations

__all__ = [
    'KudioError',
    'AudioIOError',
    'UnsupportedFormatError',
    'DeviceError',
    'FeatureError',
    'SynthesisError',
    'DependencyError',
]


class KudioError(Exception):
    """Base class for every error raised by kudio."""


class AudioIOError(KudioError):
    """Reading or writing an audio file failed."""


class UnsupportedFormatError(AudioIOError):
    """The file type / dtype is not supported."""


class DeviceError(KudioError):
    """An audio input/output device could not be opened or configured."""


class FeatureError(KudioError):
    """Feature extraction or a spectrogram transform failed."""


class SynthesisError(KudioError):
    """Noisy-data synthesis failed."""


class DependencyError(KudioError, ImportError):
    """An optional dependency is required for this operation but not installed."""

    def __init__(self, package: str, extra: str | None = None):
        hint = f"pip install kudio[{extra}]" if extra else f"pip install {package}"
        super().__init__(f"'{package}' is required for this feature. Install it with: {hint}")
        self.package = package
        self.extra = extra
