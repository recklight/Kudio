# -*- coding: utf-8 -*-
"""What every metric takes as input, and how it says it could not score it,
or that part of the input was not measured the way the rest was.

Malformed input raises ValueError, its message starting ``<function>(): ``.
Input that is well formed but cannot be scored gives nan and one
UserWarning, ``<function>(): <reason>; scored nan``. Every warning is
attributed to the first line outside kudio. A measure can be called from a
script or through kudio.score's dispatcher, and the step that warns can sit
one or two calls below the measure, so a fixed stacklevel would name the
caller's line on some of those paths and a line of kudio's on the others.
"""
from __future__ import annotations

import contextlib
import contextvars
import math
import numbers
import sys
import warnings
from typing import Iterator, Optional, Sequence, Tuple, Type, Union

import numpy as np

from kudio._samples import as_float

#: One step of 16-bit audio, 2**-15 of full scale: -90.3 dBFS. A signal that
#: peaks under it is quieter than anything a 16-bit file can hold. int16
#: samples held in an int32 array land there, read on int32's full scale
#: (32767 becomes -96.3 dBFS), and so does float audio scaled twice.
ONE_STEP = 2.0 ** -15

#: Set while score, eval_metrics, AudioEvaluate and compare_enhancers run the
#: metrics on a pair they have already called faint, if it is, and warned
#: about for its lengths where anything they run cuts it, so that neither
#: faint() nor evaluator._cut() says it again. Inside, _cut does not refuse a
#: length past MAX_LENGTH_MISMATCH either, so only that pair, cut or whole,
#: may be scored there.
_CHECKED = contextvars.ContextVar('kudio_pair_checked', default=False)


def _inside_kudio(frame) -> bool:
    name = frame.f_globals.get('__name__', '')
    return name == 'kudio' or name.startswith('kudio.')


def warn(message: str) -> None:
    """*message* as a UserWarning at the first line outside kudio."""
    frame, level = sys._getframe(1), 2
    while frame.f_back is not None and _inside_kudio(frame):
        frame, level = frame.f_back, level + 1
    warnings.warn(message, UserWarning, stacklevel=level)


def scored_nan(function: str, reason: str) -> float:
    """Warn ``<function>(): <reason>; scored nan`` and return nan."""
    warn(f"{function}(): {reason}; scored nan")
    return float('nan')


def signal(y, label: str, name: Optional[str]) -> np.ndarray:
    """*y* as one channel of float64 samples, or the ValueError that says why
    it is not one.

    A vector, a column ``(n, 1)`` and a row ``(1, n)`` are one channel.
    Integer samples are read on their type's full scale, as soundfile reads a
    file: int16 over 32768, int32 over 2**31. int64 and uint64 are refused. No
    audio file holds them, numpy gives them for a list of ints or for
    ``astype(int)``, and on their full scale 16-bit samples sit 288 dB down,
    where the fixed floors of several measures decide the score.

    :param label: what to call *y* in a message: ``'ref'``, ``'deg'``, a
        path.
    :param name: the metric, which starts the message as ``lsd(): ...``;
        None for a message with no function to name.
    """
    who = f"{name}(): " if name else ""
    if isinstance(y, np.ma.MaskedArray) and np.ma.getmaskarray(y).any():
        # np.asarray drops the mask, and the hidden samples would be scored
        masked = int(np.count_nonzero(np.ma.getmaskarray(y)))
        raise ValueError(f"{who}{label} is a masked array with {masked} masked "
                         f"sample(s), and a metric compares every sample; fill "
                         f"them (np.ma.filled) or cut them out of both signals")
    try:
        y = np.asarray(y)
    except (ValueError, TypeError) as e:        # rows of different lengths
        raise ValueError(f"{who}{label} is not an array of samples: {e}") from None
    if y.dtype.kind == 'c':
        # cast to float, it would lose its imaginary part with only numpy's
        # warning to say so
        raise ValueError(f"{who}{label} is complex ({y.dtype}); pass the real "
                         f"signal, and of an analytic signal its real part")
    if y.dtype.kind not in 'iuf':
        raise ValueError(f"{who}{label} holds {y.dtype} values, not audio "
                         f"samples; pass an array of real numbers, float or "
                         f"integer PCM")
    if y.dtype.kind in 'iu' and y.dtype.itemsize > 4:
        raise ValueError(f"{who}{label} is {y.dtype}, and no audio file holds "
                         f"64-bit integer samples: read on that type's full "
                         f"scale, 16-bit samples sit 288 dB down. A list of "
                         f"ints and astype(int) usually give int64; pass the "
                         f"samples as the type they were read as (int16 for "
                         f"16-bit PCM), or as float")
    if y.ndim == 0:
        raise ValueError(f"{who}{label} is a single number, not a signal; pass "
                         f"its samples as a 1-D array")
    # one axis at most longer than 1: (n,), (n, 1), (1, n), and (0, 1), which
    # is as empty as (0,)
    if sum(side != 1 for side in y.shape) > 1:
        raise ValueError(f"{who}{label} has shape {y.shape}, more than one "
                         f"channel, and the metrics take mono signals; score "
                         f"the channels one at a time, or mix down first")
    y = np.asarray(as_float(y), dtype=np.float64).reshape(-1)
    bad = int(np.count_nonzero(~np.isfinite(y)))
    if bad:
        raise ValueError(f"{who}{label} has {bad} sample(s) that are not finite "
                         f"(NaN or inf); repair the clip first")
    return y


def sample_rate(sr, name: str, error: Type[Exception] = ValueError,
                what: str = 'sr') -> Union[int, float]:
    """*sr* as a sample rate in hertz: an int where it is a whole number,
    16000.0 and ``np.array(16000)`` included, and a float where it is not.
    Anything but one positive number is refused; which rates a measure is
    defined at is for the measure to say.

    :param error: what to raise; a module whose refusals are FeatureErrors
        passes that.
    :param what: what the argument is called, where it is not ``sr``.
    """
    value = sr.item() if isinstance(sr, np.ndarray) and sr.ndim == 0 else sr
    if isinstance(value, numbers.Real) and not isinstance(value, (bool, np.bool_)):
        try:
            value = float(value)
        except (OverflowError, ValueError):          # an int past float64
            value = math.inf
        if math.isfinite(value) and value > 0:
            return int(value) if value.is_integer() else value
    raise error(f"{name}(): {what} has to be a positive sample rate in hertz, "
                f"a single number, got {sr!r}")


def faint(name: Optional[str], signals: Sequence[Tuple[str, np.ndarray]]) -> None:
    """Warn once if a signal that is not digital silence peaks under
    :data:`ONE_STEP`, naming each such signal and its peak in dBFS.

    The score is still given. Most measures do not change down there, but the
    1e-12 guards of snr, si_sdr, sdi and segmental_snr, the eps of lsd and of
    itakura_saito, and the -100 dB band floor of wss are absolute, and from
    about -100 dBFS they decide the number.
    """
    if _CHECKED.get():
        return
    low = []
    for label, y in signals:
        peak = float(np.max(np.abs(y))) if y.size else 0.0
        if 0.0 < peak < ONE_STEP:
            low.append((label, 20.0 * math.log10(peak)))
    if not low:
        return
    who = f"{name}(): " if name else ""
    if len(low) == 1:
        which = f"{low[0][0]} peaks at {low[0][1]:.1f} dBFS"
    else:
        which = (f"{' and '.join(label for label, _ in low)} peak at "
                 f"{' and '.join(f'{db:.1f}' for _, db in low)} dBFS")
    warn(f"{who}{which}, under one step of 16-bit audio (-90.3 dBFS). That is "
         f"where int16 samples land when they are held in an int32 array; pass "
         f"integer samples as the type they were read as, and check how float "
         f"ones were scaled. The fixed floors some measures add (the 1e-12 "
         f"guards of snr and si_sdr, the eps of lsd, the band floor of wss) "
         f"change their scores this far down")


@contextlib.contextmanager
def checked() -> Iterator[None]:
    """Run the metrics of a pair already checked, by :func:`faint` and for
    its lengths, without each of them saying it again."""
    token = _CHECKED.set(True)
    try:
        yield
    finally:
        _CHECKED.reset(token)
