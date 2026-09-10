# -*- coding: utf-8 -*-
"""Enhancement for audio that has not finished arriving.

`spectral_enhance` needs the whole clip. That is fine for a file and useless
for a microphone: you cannot monitor a take through a denoiser that only runs
once the take is over.

Everything needed was already recursive. The decision-directed a priori SNR
feeds the previous frame's estimate into this one, and MCRA updates its noise
estimate a frame at a time — both are *state plus an update*, and
:class:`StreamEnhancer` is the overlap-add machinery around them. It shares
those objects with the offline path rather than reimplementing the maths, so
the two cannot give different answers to the same question.

>>> enhancer = kudio.StreamEnhancer(sr=16000, method='logmmse')
>>> for block in microphone:                             # doctest: +SKIP
...     play(enhancer.process(block))
>>> tail = enhancer.flush()                              # doctest: +SKIP

**What it costs.** One analysis frame of latency —
:attr:`~StreamEnhancer.latency_seconds`, 32 ms at the default 512-point frame
and 16 kHz. And it is not free of the offline version's judgement: the noise
estimator has only heard what has arrived so far, so the first second is
always the worst second.

**What cannot stream.** `quantile` is a statistic over the whole file;
`spectral_gate`, `specsub` and `multiband` in kudio look at the spectrogram as
a block. :data:`STREAMABLE_METHODS` is the list that does, and asking for
anything else raises rather than quietly giving you something different from
what the same name does offline.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from kudio.enhance.spectral import (
    METHODS,
    RECURSIVE_METHODS,
    DecisionDirected,
    make_gain_rule,
    make_noise_tracker,
)
from kudio.exceptions import FeatureError

__all__ = ['StreamEnhancer', 'STREAMABLE_METHODS']

log = logging.getLogger(__name__)

#: The methods whose gain depends only on this frame's (xi, gamma).
STREAMABLE_METHODS = RECURSIVE_METHODS

_EPS = 1e-12


class StreamEnhancer:
    """Denoise a stream block by block, with a one-frame delay.

    :param sr: sample rate.
    :param method: one of :data:`STREAMABLE_METHODS`.
    :param noise: ``'mcra'``, ``'initial'`` or ``'minimum'`` — see
        :func:`kudio.make_noise_tracker`. ``'quantile'`` cannot stream.
    :param n_fft: analysis frame; also the latency.
    :param hop_length: defaults to ``n_fft // 4``.
    :param params: method parameters, defaulted from ``METHODS[method]``.
    """

    def __init__(self, sr: int, method: str = 'logmmse', *,
                 noise: str = 'mcra', n_fft: int = 512,
                 hop_length: Optional[int] = None,
                 window: str = 'hann',
                 noise_kwargs: Optional[dict] = None,
                 **params):
        if sr <= 0:
            raise FeatureError(f"StreamEnhancer needs a positive rate, got {sr}")
        if method not in STREAMABLE_METHODS:
            offline = "" if method not in METHODS else \
                f" It works offline: kudio.spectral_enhance(y, sr, {method!r})."
            raise FeatureError(
                f"{method!r} cannot run on a stream — its gain depends on the "
                f"whole spectrogram. Choose from "
                f"{', '.join(STREAMABLE_METHODS)}.{offline}")

        settings = METHODS[method].defaults()
        unknown = set(params) - set(settings)
        if unknown:
            raise FeatureError(
                f"{method!r} does not take {', '.join(sorted(unknown))}. "
                f"It takes: {', '.join(sorted(settings))}")
        settings.update(params)

        self.sr = int(sr)
        self.method = method
        self.noise = noise
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length or n_fft // 4)
        self.settings = settings
        self._noise_kwargs = dict(noise_kwargs or {})
        self._g_min = 10.0 ** (settings.get('gain_floor_db', -25.0) / 20.0)

        self._window = _get_window(window, self.n_fft)
        # Analysis and synthesis both apply the window (WOLA), so what has to
        # sum to a constant is w^2, not w. Measuring it beats assuming it: an
        # untested COLA assumption shows up as a periodic amplitude ripple at
        # the hop rate, which is easy to hear and hard to attribute.
        self._wola_norm = _overlap_sum(self._window ** 2, self.hop_length)
        self.reset()

    def _normalise(self, values: np.ndarray, norm: np.ndarray) -> np.ndarray:
        """Undo the windowing, **never amplifying**.

        At the very first and very last samples only one frame has been seen,
        so the accumulated window energy is a fraction of the steady-state
        value. Dividing by it scales those samples up by whatever that fraction
        happens to be — measured at the end of a take: an output peak of 6.5
        from an input peak of 0.37, which clips the moment it is written.

        Clamping the divisor at the steady state turns that into a short fade
        instead. The edges genuinely carry less information than the middle;
        attenuating them says so, amplifying them lies about it.
        """
        return values / np.maximum(norm, self._wola_norm)

    # ------------------------------------------------------------- properties

    @property
    def latency_seconds(self) -> float:
        """How far behind the output runs: one analysis frame."""
        return self.n_fft / self.sr

    @property
    def frames_seen(self) -> int:
        return self._frames

    # ---------------------------------------------------------------- process

    def reset(self) -> None:
        """Forget every buffer and estimate, keeping the configuration."""
        self._tracker = make_noise_tracker(self.noise, **self._noise_kwargs)
        self._state = DecisionDirected(
            make_gain_rule(self.method, self.settings),
            self.settings.get('alpha', 0.98))
        self._input = np.zeros(0, dtype=np.float64)
        #: one frame wide; index 0 is always the next sample to hand out, and
        #: the buffer slides by one hop per frame. Every frame therefore lands
        #: at offset 0 *of the slid buffer*, which is what makes successive
        #: frames overlap correctly instead of stacking on top of each other.
        self._acc = np.zeros(self.n_fft, dtype=np.float64)
        self._norm = np.zeros(self.n_fft, dtype=np.float64)
        self._pending: list = []
        self._frames = 0

    def process(self, block: np.ndarray) -> np.ndarray:
        """Feed a block in; get the enhanced audio that is finished back.

        The returned block is **not** the same length as the input — nothing
        comes out until a whole frame has arrived, and then it emerges in
        hop-sized pieces. Concatenate what comes back and call :meth:`flush`
        at the end; sample *i* of that concatenation lines up with sample *i*
        of the input.
        """
        block = np.asarray(block, dtype=np.float64).reshape(-1)
        self._input = np.concatenate([self._input, block])

        while len(self._input) >= self.n_fft:
            self._consume_frame(self._input[:self.n_fft])
            self._input = self._input[self.hop_length:]
        return self._emit()

    def flush(self) -> np.ndarray:
        """Everything still in flight, once no more audio is coming.

        Pads the last partial frame with zeros so the tail is processed rather
        than dropped — a stream that quietly loses its final 30 ms is the kind
        of bug that only shows up on short takes.
        """
        while len(self._input):
            padded = np.zeros(self.n_fft, dtype=np.float64)
            take = min(len(self._input), self.n_fft)
            padded[:take] = self._input[:take]
            self._consume_frame(padded)
            self._input = self._input[self.hop_length:] \
                if len(self._input) > self.hop_length else self._input[:0]

        # drain whatever the accumulator still holds, up to the last sample any
        # frame actually reached
        covered = int(np.count_nonzero(self._norm > 1e-8))
        if covered:
            tail = self._normalise(self._acc[:covered], self._norm[:covered])
            self._pending.append(tail.astype(np.float32))
        self._acc[:] = 0.0
        self._norm[:] = 0.0
        return self._emit()

    # ---------------------------------------------------------------- internal

    def _consume_frame(self, samples: np.ndarray) -> None:
        windowed = samples * self._window
        spectrum = np.fft.rfft(windowed, n=self.n_fft)
        power = np.maximum(np.abs(spectrum) ** 2, _EPS)

        noise = np.maximum(self._tracker.update(power), _EPS)
        gain = np.clip(self._state.update(power, noise), self._g_min, 1.0)

        frame = np.fft.irfft(spectrum * gain, n=self.n_fft) * self._window
        self._acc += frame
        self._norm += self._window ** 2
        self._frames += 1

        # the leading hop can receive nothing further -- the next frame starts
        # one hop later -- so it is final, and the buffer slides past it
        hop = self.hop_length
        ready = self._normalise(self._acc[:hop], self._norm[:hop])
        self._pending.append(ready.astype(np.float32))
        self._acc = np.concatenate([self._acc[hop:], np.zeros(hop)])
        self._norm = np.concatenate([self._norm[hop:], np.zeros(hop)])

    def _emit(self) -> np.ndarray:
        if not self._pending:
            return np.zeros(0, dtype=np.float32)
        out = np.concatenate(self._pending)
        self._pending = []
        return out


def _get_window(name: str, size: int) -> np.ndarray:
    from scipy.signal import get_window
    return np.asarray(get_window(name, size, fftbins=True), dtype=np.float64)


def _overlap_sum(window: np.ndarray, hop: int) -> float:
    """The constant a windowed overlap-add converges to in the steady state."""
    total = np.zeros(len(window))
    for offset in range(-len(window), len(window) + 1, hop):
        lo, hi = max(0, offset), min(len(window), offset + len(window))
        if hi > lo:
            total[lo:hi] += window[lo - offset:hi - offset]
    middle = total[len(window) // 4: 3 * len(window) // 4]
    return float(np.median(middle)) or 1.0
