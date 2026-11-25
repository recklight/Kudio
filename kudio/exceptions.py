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
``TimeoutError``. :class:`FeatureError` is a ``ValueError`` as well, so
``except ValueError`` catches what an effect or a measurement refuses along
with what a metric does; ``except KudioError`` catches the first alone.
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

    Not raised by kudio itself at present: a file that cannot be read or
    decoded comes back as a plain :class:`AudioIOError`. Catch that one.
    """


class DeviceError(KudioError, RuntimeError):
    """An audio input/output device could not be opened or configured.

    Also a ``RuntimeError``: :meth:`kudio.CheckDevice.input_device` raised
    plain ``RuntimeError`` for the same failures, and code written against
    that still catches them.
    """


class FeatureError(KudioError, ValueError):
    """A feature, effect, room, channel or enhancement step refused its input
    or its settings: the wrong shape of audio, a parameter out of range, a
    clip too short to frame.

    Also a ``ValueError``, the one a metric raises for input it refuses: which
    of the two a function raises depends on the module it lives in, and a
    caller should not have to know that.
    """


class SynthesisError(KudioError):
    """Noisy-data synthesis failed."""


class DependencyError(KudioError, ImportError):
    """An optional dependency is required for this operation but not installed."""

    def __init__(self, package: str, extra: str | None = None,
                 failed: str | None = None):
        if failed:
            # installed, but it will not import: installing it again is not
            # the cure, what it failed on is
            message = (f"'{package}' is required for this feature, and importing "
                       f"it failed: {failed}")
        else:
            # quoted, or zsh reads the brackets as a file pattern
            hint = f'pip install "kudio[{extra}]"' if extra else f"pip install {package}"
            message = f"'{package}' is required for this feature. Install it with: {hint}"
        super().__init__(message)
        self.package = package
        self.extra = extra
        self.failed = failed

    def __reduce__(self):
        # Rebuilt from its arguments, so one raised in a worker process
        # (Synthesizer, map_waves) reaches the parent as it was raised.
        # Exception's own pickling hands the message back to __init__, which
        # would take it for the package's name.
        return type(self), (self.package, self.extra, self.failed)
