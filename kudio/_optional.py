# -*- coding: utf-8 -*-
"""librosa, imported by the features that use it rather than by ``import kudio``.

The metrics, loudness, the room tools and the filters run on numpy and scipy
alone, so a machine without librosa, or with one that will not import, can
still score and inspect audio. What librosa does for kudio -- the features,
resampling, pitch, silence trimming, time stretch and pitch shift -- raises
:class:`~kudio.exceptions.DependencyError` when it is called there instead.
"""
from __future__ import annotations

from types import ModuleType

from kudio.exceptions import DependencyError


def librosa() -> ModuleType:
    """The librosa module, or DependencyError saying why there is none.

    librosa 0.10 and later load their parts on first use, so a numba that
    will not import with the installed numpy ("Numba needs NumPy 2.0 or
    less") gets past ``import librosa`` and fails at the first call into it,
    inside librosa.pyin say, as a plain ImportError. So numba is imported
    here too: the parts of librosa kudio calls (stft, resample, pyin, the
    features, trim, the effects) all need it. A package that is missing gets
    the message that says how to install it; one that is there and fails to
    import gets the message it failed with, and the ImportError is chained,
    so the traceback still shows where.
    """
    try:
        import librosa
        import numba  # noqa: F401
    except ImportError as e:
        missing = isinstance(e, ModuleNotFoundError) and e.name in ('librosa', 'numba')
        raise DependencyError('librosa', failed=None if missing else str(e)) from e
    return librosa
