# -*- coding: utf-8 -*-
"""kudio — a lightweight, composable audio toolkit for acoustic inspection.

Audio I/O, streaming/recording, feature extraction, noisy-data synthesis,
effects/augmentation, enhancement, and evaluation metrics.

>>> import kudio
>>> y, sr = kudio.file_load('some.wav')
>>> spec = kudio.waveform_to_spectrogram(y)
>>> clean = kudio.trim_silence(y)[0]

Heavy/niche features are optional extras and are *not* imported here:
plotting (`kudio.util.visual`, needs ``kudio[viz]``), Excel export
(`kudio.util.conv`, needs ``kudio[data]``), and live audio
(`kudio[audio]`).
"""
import logging as _logging

from kudio._version import __author__, __version__
from kudio.config import KC, KudioConfig
from kudio.exceptions import (
    AudioIOError,
    DependencyError,
    DeviceError,
    FeatureError,
    KudioError,
    SynthesisError,
    UnsupportedFormatError,
)

from kudio import core, effects, enhance, util  # noqa: E402
from kudio.core import *  # noqa: F401,F403  (curated __all__)
from kudio.effects import *  # noqa: F401,F403
from kudio.enhance import *  # noqa: F401,F403  (curated __all__)
from kudio.util import *  # noqa: F401,F403  (curated __all__)

_logging.getLogger(__name__).addHandler(_logging.NullHandler())

__all__ = [
    '__version__', '__author__', 'KudioConfig', 'KC',
    # exceptions
    'KudioError', 'AudioIOError', 'UnsupportedFormatError', 'DeviceError',
    'FeatureError', 'SynthesisError', 'DependencyError',
]
__all__ += core.__all__
__all__ += effects.__all__
__all__ += enhance.__all__
__all__ += util.__all__
