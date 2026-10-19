# -*- coding: utf-8 -*-
"""Exception hierarchy for kudio.

What goes wrong with a file, a device, a feature, a synthesis run or a missing
extra derives from :class:`KudioError`, so callers can::

    try:
        kudio.file_load(path)
    except kudio.KudioError:
        ...

A wrong argument -- a negative rate, an unknown metric, two signals too
different in length to compare -- raises ``ValueError`` or ``TypeError`` as the
standard library would, and a buffer that waited too long raises
``TimeoutError``.
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
    """The file type / dtype is not supported.

    Not raised by kudio itself at present: a file soundfile cannot read comes
    back as a plain :class:`AudioIOError`. Catch that one.
    """


class DeviceError(KudioError, RuntimeError):
    """An audio input/output device could not be opened or configured.

    Also a ``RuntimeError``: :meth:`kudio.CheckDevice.input_device` raised
    plain ``RuntimeError`` for the same failures, and code written against
    that still catches them.
    """


class FeatureError(KudioError):
    """A feature, effect, room, channel or enhancement step refused its input
    or its settings: the wrong shape of audio, a parameter out of range, a
    clip too short to frame."""


class SynthesisError(KudioError):
    """Noisy-data synthesis failed."""


class DependencyError(KudioError, ImportError):
    """An optional dependency is required for this operation but not installed."""

    def __init__(self, package: str, extra: str | None = None):
        hint = f"pip install kudio[{extra}]" if extra else f"pip install {package}"
        super().__init__(f"'{package}' is required for this feature. Install it with: {hint}")
        self.package = package
        self.extra = extra
