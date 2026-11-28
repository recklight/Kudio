# -*- coding: utf-8 -*-
"""The discrete wavelet transform with periodization, as many levels down as the
clip allows, and back.

Every sum runs in the order PyWavelets' C code runs it, so the coefficients and
the reconstruction are PyWavelets' own to the bit, in float32 and in float64:
each output starts at zero and takes its taps in order, except where the filter
reaches past the end of the clip, whose taps come first, the one at the end
leading. An odd length is padded with its last sample before each level, as
PyWavelets pads it. The filters are in ``_wavelet_filters``, copied from
PyWavelets (MIT licence, reproduced there).
"""
from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Tuple

import numpy as np

Bank = Dict[np.dtype, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]

#: the names, by family, for the messages and the docstring
FAMILIES = ('haar', 'db1-db38', 'sym2-sym20', 'coif1-coif17', 'bior1.1-bior6.8',
            'rbio1.1-rbio6.8', 'dmey')

_F64, _F32 = np.dtype(np.float64), np.dtype(np.float32)


def _from_low_pass(dec_lo, rec_lo, dtype) -> tuple:
    dec_lo = np.asarray(dec_lo, dtype)
    rec_lo = dec_lo[::-1].copy() if rec_lo is None else np.asarray(rec_lo, dtype)
    sign = np.where(np.arange(len(dec_lo)) % 2, -1, 1).astype(dtype)
    return dec_lo, -sign * rec_lo, rec_lo, sign * dec_lo


@lru_cache(maxsize=None)
def _named(name: str) -> Bank:
    from kudio.enhance._wavelet_filters import LOW_PASS, RECONSTRUCTION, SINGLE
    low, rec = LOW_PASS[name], RECONSTRUCTION.get(name)
    bank = {_F64: _from_low_pass(low, rec, _F64)}
    if name in SINGLE:
        bank[_F32] = _from_low_pass(*SINGLE[name], _F32)
    else:
        bank[_F32] = _from_low_pass(np.asarray(low).astype(_F32),
                                    None if rec is None else np.asarray(rec).astype(_F32), _F32)
    return bank


def names() -> List[str]:
    """Every wavelet name the filters are there for."""
    from kudio.enhance._wavelet_filters import LOW_PASS
    return list(LOW_PASS)


def filter_bank(wavelet, where: str) -> Bank:
    """The four filters of *wavelet*, a name (any case) or an object with a
    ``filter_bank`` of four filters, as ``pywt.Wavelet`` has, in float64 and in
    float32."""
    if isinstance(wavelet, str) and wavelet:
        from kudio.enhance._wavelet_filters import LOW_PASS
        if wavelet.lower() in LOW_PASS:
            return _named(wavelet.lower())
        raise ValueError(f"{where}(): wavelet {wavelet!r} is not a discrete wavelet this "
                         f"filter knows; the names are {', '.join(FAMILIES)}")
    if wavelet is None or isinstance(wavelet, str):
        raise TypeError(f"{where}(): wavelet has to be the name of one, such as 'db2', "
                        f"got {wavelet!r}")
    if isinstance(wavelet, (bytes, bytearray)):
        raise ValueError(f"{where}(): wavelet {wavelet!r} is bytes; the names are text: "
                         f"{', '.join(FAMILIES)}")
    bank = getattr(wavelet, 'filter_bank', None)
    if bank is None and hasattr(wavelet, 'family_name'):
        raise ValueError(f"{where}(): wavelet is a {type(wavelet).__name__}; this filter takes "
                         f"a discrete one: {', '.join(FAMILIES)}")
    if bank is None:
        raise TypeError(f"{where}(): wavelet has to be the name of one, such as 'db2', "
                        f"or a pywt.Wavelet, got {type(wavelet).__name__}")
    try:
        four = [np.asarray(f, dtype=_F64) for f in bank]
    except (TypeError, ValueError) as e:
        raise ValueError(f"{where}(): the filters of wavelet are not numbers: {e}") from e
    if (len(four) != 4 or any(f.ndim != 1 for f in four)
            or len({len(f) for f in four}) != 1 or len(four[0]) % 2 or not len(four[0])):
        raise ValueError(f"{where}(): wavelet has to have four filters of one even length, "
                         f"got lengths {[np.size(f) for f in four]}")
    # a builtin passed as a pywt.Wavelet computes as its name does
    name = getattr(wavelet, 'name', None)
    if isinstance(name, str) and name.lower() in names():
        mine = _named(name.lower())[_F64]
        if all(np.array_equal(a, b) for a, b in zip(four, mine)):
            return _named(name.lower())
    return {_F64: tuple(four), _F32: tuple(f.astype(_F32) for f in four)}


def max_level(n: int, taps: int) -> int:
    """The levels a clip of *n* samples is taken down: log2(n / (taps - 1)), rounded down."""
    if taps <= 1 or n < taps - 1:
        return 0
    return (n // (taps - 1)).bit_length() - 1


def working_type(dtype) -> np.dtype:
    """float32 and float64 compute as they are, float16 in float32, anything else in
    float64. A big-endian float32 is not float32 here."""
    if dtype == _F64 or dtype == _F32:
        return np.dtype(dtype)
    if dtype == np.float16:
        return _F32
    return _F64


@lru_cache(maxsize=512)
def _past_the_end(first: int, count: int, step: int, n: int, taps: int) -> np.ndarray:
    """For the outputs centred at first, first + step, ... that reach past the end of
    the clip (centre i >= n): which tap each step of the sum takes. The taps that land
    at n, n + 1, ..., i come first, from the one at n down to tap 0, then the rest in
    order."""
    order = np.empty((count, taps), np.intp)
    for k in range(count):
        c = first + step * k - n + 1
        order[k] = [c - 1 - s if s < c else s for s in range(taps)]
    return order


def _periodic(x: np.ndarray, before: int) -> np.ndarray:
    """*x* with *before* samples of its own periodic extension ahead of it."""
    return x[np.arange(-before, len(x)) % len(x)]


def dwt(x: np.ndarray, lo: np.ndarray, hi: np.ndarray):
    """One level down: (approximation, detail), each ceil(len(x) / 2) long."""
    n, taps = len(x), len(lo)
    xp = np.concatenate([x, x[-1:]]) if n % 2 else x
    p, m = len(xp), (n + 1) // 2
    centre = taps // 2 + 2 * np.arange(m)
    inside = int(np.searchsorted(centre, n))
    xe = _periodic(xp, taps)
    order = _past_the_end(int(centre[inside]), m - inside, 2, n, taps) if inside < m else None
    out = []
    for h in (lo, hi):
        acc = np.zeros(m, xp.dtype)
        for j in range(taps):
            start = taps + taps // 2 - j
            acc[:inside] += h[j] * xe[start:start + 2 * inside:2]
        if order is not None:
            ci = centre[inside:]
            for s in range(taps):
                acc[inside:] += h[order[:, s]] * xp[(ci - order[:, s]) % p]
        out.append(acc)
    return out[0], out[1]


def idwt(a: np.ndarray, d: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """One level up from approximation *a* and detail *d* of one length n: 2n samples,
    the approximation's part summed first and the detail's added onto it."""
    n, taps = len(a), len(lo) // 2
    start = len(lo) // 4
    shift = 1 - taps % 2                   # an even half-length moves the output on by one
    out = np.zeros(2 * n, a.dtype)
    centre = start + np.arange(n)
    inside = int(np.searchsorted(centre, n))
    order = _past_the_end(int(centre[inside]), n - inside, 1, n, taps) if inside < n else None
    for x, h in ((a, lo), (d, hi)):
        even, odd = h[0::2], h[1::2]
        xe = _periodic(x, taps)
        for j in range(taps):
            part = xe[taps + start - j:taps + start - j + inside]
            out[shift:shift + 2 * inside:2] += even[j] * part
            out[shift + 1:shift + 2 * inside:2] += odd[j] * part
        if order is not None:
            ci = centre[inside:]
            first = (shift + 2 * (ci - start)) % (2 * n)
            second = (first + 1) % (2 * n)
            for s in range(taps):
                v = x[(ci - order[:, s]) % n]
                out[first] += even[order[:, s]] * v
                out[second] += odd[order[:, s]] * v
    return out


def wavedec(y: np.ndarray, bank: Bank) -> list:
    """[approximation, coarsest detail, ..., finest detail]. A clip too short for one
    level comes back alone, as it went in."""
    a, details = y, []
    for _ in range(max_level(len(y), len(bank[_F64][0]))):
        a = np.asarray(a, dtype=working_type(a.dtype), order='C')
        a, d = dwt(a, *bank[a.dtype][:2])
        details.append(d)
    return [a] + details[::-1]


def waverec(coeffs: list, bank: Bank) -> np.ndarray:
    """Back up from what :func:`wavedec` gives. An approximation one longer than the
    detail it meets, as an odd length leaves it, loses its last sample; where the two
    differ in type both go to float64."""
    a = coeffs[0]
    for d in coeffs[1:]:
        if len(a) == len(d) + 1:
            a = a[:len(d)]
        a = np.asarray(a, dtype=working_type(a.dtype), order='C')
        d = np.asarray(d, dtype=working_type(d.dtype), order='C')
        if a.dtype != d.dtype:
            a, d = a.astype(_F64), d.astype(_F64)
        a = idwt(a, d, *bank[a.dtype][2:])
    return a
