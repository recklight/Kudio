# -*- coding: utf-8 -*-
"""Statistical single-channel speech enhancement, as a set of interchangeable parts.

Every method here is the same three decisions:

1. **estimate the noise** power spectrum, frame by frame;
2. from that, estimate the **a priori SNR** — how much speech is in this bin;
3. turn that into a **gain** between 0 and 1 and multiply the spectrum by it.

`trad_enhance` fuses all three into one function with no parameters, so there
is no way to ask what any of them contributed. Splitting them apart is what
makes the methods comparable: swap the gain rule and hold the noise estimator
still, and the difference you hear is the gain rule.

What separates them in practice:

============  ===============================================================
`specsub`     Subtract the noise spectrum. Cheapest, and the reason
              "musical noise" has a name: isolated surviving bins warble.
`multiband`   Spectral subtraction with a per-band over-subtraction factor.
              Colored noise is not equally loud everywhere, and one global
              factor over-suppresses where it is quiet.
`wiener`      Gain ``xi / (1 + xi)``. Smooth, so much less musical noise, at
              the cost of muffling the speech at low SNR.
`mmse_stsa`   Ephraim & Malah (1984): MMSE estimate of the spectral
              *amplitude*, assuming Gaussian coefficients. Keeps more speech
              than Wiener at the same suppression.
`logmmse`     Ephraim & Malah (1985): MMSE in the *log* domain, which is
              closer to how loudness is perceived. The usual default.
`omlsa`       `logmmse` weighted by a speech-presence probability
              (Cohen 2001). Suppresses hard where there is probably no
              speech at all, gently where there might be.
`spectral_gate` A threshold mask rather than a statistical estimator: bins
              below (noise mean + n sigma) are attenuated. Not principled,
              very predictable, and good on steady broadband noise.
============  ===============================================================

The decision-directed a priori SNR (Ephraim & Malah) is shared by
`wiener` / `mmse_stsa` / `logmmse` / `omlsa`::

    xi(k, l) = alpha * A(k, l-1)^2 / lambda_d + (1 - alpha) * max(gamma - 1, 0)

with ``alpha = 0.98``: it is the smoothing that removes musical noise, and
also what makes these methods lag a sudden onset by a frame or two.

**Two things to know before reaching for any of them.**

*They are for noisy audio, and they damage clean audio.* Every one trades
speech distortion for noise removal. Below about 5 dB SNR that is a good trade;
above about 10 dB it is not, and all seven measure **worse than doing nothing**
— measured on this repo's own fixtures, +8.6 dB SNR at 2 dB in, −1.8 dB at
10 dB in. `compare_enhancers(reference=…)` reports `delta_snr_db` precisely so
that shows up rather than being assumed away.

*They need to hear the noise on its own.* Noise estimation works from the quiet
parts, so a clip that is wall-to-wall speech gives them almost nothing to
measure. On the same clip at the same SNR, one with pauses scored +5.9 dB and
one without −1.8 dB. If a recording has no pauses, `noise='initial'` with a
leading second of room tone beats anything adaptive.

>>> clean = kudio.spectral_enhance(y, sr)                    # logmmse
>>> clean = kudio.spectral_enhance(y, sr, method='omlsa', noise='quantile')
>>> for r in kudio.compare_enhancers(y, sr, reference=truth):
...     print(r.method, r.si_sdr_db)
"""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path, PurePath
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from kudio.core.io import ConvertResult
from kudio.exceptions import FeatureError

__all__ = [
    'Param', 'Method', 'METHODS', 'NOISE_ESTIMATORS',
    'spectral_enhance', 'estimate_noise_psd',
    'NoiseTracker', 'make_noise_tracker', 'make_gain_rule',
    'EnhanceResult', 'compare_enhancers', 'CustomEnhancer', 'CUSTOM_NOISE',
    'EnhanceFolderResult', 'enhance_folder',
]

log = logging.getLogger(__name__)

_EPS = 1e-12
#: A posteriori SNR is clamped here (40 dB). Left unbounded, one loud frame
#: sends the decision-directed estimate somewhere it takes seconds to leave.
_GAMMA_MAX = 10.0 ** 4
#: Floor on the a priori SNR, -25 dB, as in the reference implementations.
_XI_MIN = 10.0 ** (-25.0 / 10.0)
#: What a custom enhancer's `noise` column says: it brought its own, and
#: naming one of kudio's estimators beside it would be a lie.
CUSTOM_NOISE = 'own'


# --------------------------------------------------------------------- schema

@dataclass(frozen=True)
class Param:
    """One tunable knob, described well enough to build a control from it."""
    name: str
    label: str
    default: float
    low: float
    high: float
    step: float = 0.01
    decimals: int = 2
    suffix: str = ""
    help: str = ""


@dataclass(frozen=True)
class Method:
    """An enhancement method: what it is, and what it lets you turn."""
    name: str
    label: str
    summary: str
    params: Tuple[Param, ...] = ()
    #: cited source, for the ones that have one
    reference: str = ""

    def defaults(self) -> Dict[str, float]:
        return {p.name: p.default for p in self.params}


# Parameters shared by the decision-directed family.
_ALPHA = Param('alpha', 'alpha (SNR smoothing)', 0.98, 0.50, 0.999, 0.005, 3,
               help="Decision-directed smoothing. Higher removes more musical "
                    "noise and lags speech onsets more.")
_GAIN_FLOOR = Param('gain_floor_db', 'gain floor', -25.0, -80.0, 0.0, 1.0, 1, " dB",
                    help="How far a bin may be attenuated. A hard zero sounds "
                         "worse than a quiet noise bed: total suppression is "
                         "what makes the residue audible as warbling.")
_OVERSUB = Param('over_subtraction', 'over-subtraction', 2.0, 1.0, 8.0, 0.1, 1,
                 help="Subtract this multiple of the noise estimate. Above ~3 "
                      "the speech starts going with it.")
_FLOOR = Param('spectral_floor', 'spectral floor', 0.002, 0.0, 0.2, 0.001, 3,
               help="What is left where the subtraction went negative, as a "
                    "fraction of the noisy spectrum.")

METHODS: Dict[str, Method] = {m.name: m for m in (
    Method('specsub', 'Spectral subtraction',
           "Subtract the noise spectrum. Cheapest, and the origin of musical "
           "noise: isolated surviving bins warble between frames.",
           (_OVERSUB, _FLOOR, _GAIN_FLOOR),
           "Boll 1979"),
    Method('multiband', 'Multi-band spectral subtraction',
           "Spectral subtraction with a per-band over-subtraction factor set "
           "from each band's own SNR — colored noise is not equally loud "
           "everywhere, and one global factor over-suppresses where it is not.",
           (Param('bands', 'bands', 6, 2, 32, 1, 0,
                  help="Linearly spaced. More bands track colored noise more "
                       "closely and are noisier to estimate."),
            _FLOOR, _GAIN_FLOOR),
           "Kamath & Loizou 2002"),
    Method('wiener', 'Wiener (decision-directed)',
           "Gain xi/(1+xi). Smooth, so far less musical noise than subtraction "
           "— it muffles the speech at low SNR instead.",
           (_ALPHA, _GAIN_FLOOR)),
    Method('mmse_stsa', 'MMSE-STSA',
           "MMSE estimate of the spectral amplitude under a Gaussian model. "
           "Keeps more speech than Wiener at the same suppression.",
           (_ALPHA, _GAIN_FLOOR),
           "Ephraim & Malah 1984"),
    Method('logmmse', 'log-MMSE',
           "MMSE in the log-spectral domain, which is closer to how loudness "
           "is perceived. The usual default, and hard to beat without a model.",
           (_ALPHA, _GAIN_FLOOR),
           "Ephraim & Malah 1985"),
    Method('omlsa', 'OM-LSA',
           "log-MMSE weighted by the probability that there is any speech in "
           "the bin at all: suppress hard where there is probably none, gently "
           "where there might be.",
           (_ALPHA, _GAIN_FLOOR,
            Param('q', 'P(no speech)', 0.30, 0.01, 0.95, 0.05, 2,
                  help="A priori probability that a bin holds no speech. "
                       "Higher suppresses more and risks clipping onsets.")),
           "Cohen 2001"),
    Method('spectral_gate', 'Spectral gate',
           "A threshold mask, not a statistical estimator: attenuate every bin "
           "below (noise mean + n sigma), then smooth the mask so the edges do "
           "not click. Unprincipled, very predictable.",
           (Param('n_std_thresh', 'threshold', 1.5, 0.0, 6.0, 0.1, 1, " sigma",
                  help="How far above the noise floor a bin has to be to be "
                       "kept."),
            Param('prop_decrease', 'strength', 1.0, 0.0, 1.0, 0.05, 2,
                  help="1.0 applies the mask in full; lower mixes the original "
                       "back in, which trades noise for artefacts."),
            Param('freq_smooth_hz', 'smooth (freq)', 500.0, 0.0, 4000.0, 50.0, 0, " Hz",
                  help="Blur the mask across frequency. Unsmoothed masks are "
                       "what musical noise sounds like."),
            Param('time_smooth_ms', 'smooth (time)', 50.0, 0.0, 500.0, 10.0, 0, " ms",
                  help="Blur the mask across time.")),
           "Spectral gating, after noisereduce"),
)}

NOISE_ESTIMATORS: Dict[str, str] = {
    'mcra': "Minima-controlled recursive averaging: tracks the noise while "
            "speech is present, by watching each bin's running minimum. The "
            "general-purpose choice, and what trad_enhance uses.",
    'quantile': "A low quantile of each bin over the whole file. Offline only "
                "— it needs to see the future — and usually the most accurate "
                "of these when the file is not one continuous shout.",
    'initial': "Average of the first few frames, assumed to be room tone. "
               "Exact when that is true and badly wrong when it is not.",
    'minimum': "Running minimum over a sliding window, bias-compensated. "
               "Follows non-stationary noise faster than MCRA and is noisier.",
}


# ------------------------------------------------------------ noise estimation

class NoiseTracker:
    """A noise estimator as **state plus one update**, not a whole-file pass.

    Written as a batch function, a tracker cannot be used on a live stream;
    written twice — once batch, once streaming — the two drift. So the update
    rule lives here once, and the offline path is a loop over it.

    ``quantile`` is the exception and is deliberately not a tracker: it needs
    to see the future.
    """
    #: can this run on audio that has not finished arriving?
    streamable = True

    def update(self, power: np.ndarray) -> np.ndarray:
        """One frame in, that frame's noise PSD out."""
        raise NotImplementedError

    @staticmethod
    def _as_float64(power: np.ndarray) -> np.ndarray:
        """Recursive estimators must not inherit the caller's precision.

        These accumulate over hundreds of frames. Fed float32 — which is what
        an STFT hands back — the estimate drifts ~2% from the same arithmetic
        in float64 over five seconds of audio, so the offline and streaming
        paths would disagree purely on the dtype the caller happened to have.
        """
        return np.asarray(power, dtype=np.float64)


class _MCRATracker(NoiseTracker):
    """Minima-controlled recursive averaging."""

    def __init__(self, ass: float = 0.8, ad: float = 0.95, ap: float = 0.2,
                 delta: float = 5.0, window: int = 100):
        self.ass, self.ad, self.ap = ass, ad, ap
        self.delta, self.window = delta, window
        self._n = 0
        self._smoothed = None

    def update(self, power: np.ndarray) -> np.ndarray:
        power = self._as_float64(power)
        if self._smoothed is None:
            self._smoothed = power.copy()
            self._p_min = power.copy()
            self._p_tmp = power.copy()
            self._speech_p = np.zeros_like(power)
            self._estimate = power.copy()

        self._smoothed = self.ass * self._smoothed + (1.0 - self.ass) * power
        if self._n and self._n % self.window == 0:
            self._p_min = np.minimum(self._p_tmp, self._smoothed)
            self._p_tmp = self._smoothed.copy()
        else:
            self._p_min = np.minimum(self._p_min, self._smoothed)
            self._p_tmp = np.minimum(self._p_tmp, self._smoothed)
        # a bin far above its own running minimum probably holds speech, so
        # stop adapting there -- otherwise the noise estimate eats the speech
        indicator = (self._smoothed / np.maximum(self._p_min, _EPS)
                     > self.delta).astype(float)
        self._speech_p = self.ap * self._speech_p + (1.0 - self.ap) * indicator
        adapt = self.ad + (1.0 - self.ad) * self._speech_p
        self._estimate = adapt * self._estimate + (1.0 - adapt) * power
        self._n += 1
        return self._estimate.copy()


class _InitialTracker(NoiseTracker):
    """Mean of the opening frames, then held. Exact when they really are noise."""

    def __init__(self, n_frames: int = 6):
        self.n_frames = max(1, int(n_frames))
        self._seen = 0
        self._total = None
        self._level = None

    def update(self, power: np.ndarray) -> np.ndarray:
        power = self._as_float64(power)
        if self._seen < self.n_frames:
            self._total = power.copy() if self._total is None else self._total + power
            self._seen += 1
            self._level = self._total / self._seen
        return self._level.copy()


class _MinimumTracker(NoiseTracker):
    """Sliding-window minimum with a fixed bias correction.

    The minimum of a noisy quantity underestimates its mean, hence *bias*; a
    proper minimum-statistics tracker derives that correction from the window
    length rather than assuming it, which is the part left out here.
    """

    def __init__(self, window: int = 60, bias: float = 1.5,
                 smoothing: float = 0.85):
        self.window, self.bias, self.smoothing = int(window), bias, smoothing
        self._running = None
        self._history: deque = deque(maxlen=int(window))

    def update(self, power: np.ndarray) -> np.ndarray:
        power = self._as_float64(power)
        self._running = power.copy() if self._running is None else \
            self.smoothing * self._running + (1.0 - self.smoothing) * power
        self._history.append(self._running.copy())
        return np.min(np.stack(self._history), axis=0) * self.bias


def make_noise_tracker(method: str = 'mcra', **kwargs) -> NoiseTracker:
    """A stateful tracker for *method*.

    :raises FeatureError: for ``quantile``, which is not a tracker — it is a
        statistic over the whole file, and there is no honest way to compute it
        one frame at a time.
    """
    if method not in NOISE_ESTIMATORS:
        raise FeatureError(
            f"unknown noise estimator {method!r}. "
            f"Choose from {', '.join(sorted(NOISE_ESTIMATORS))}")
    if method == 'quantile':
        raise FeatureError(
            "'quantile' takes a low quantile over the whole file, so it cannot "
            "run on audio that has not finished arriving. Use 'mcra' (adaptive) "
            "or 'initial' (assumes the opening is room tone) for streaming.")
    if method == 'mcra':
        return _MCRATracker(**kwargs)
    if method == 'initial':
        return _InitialTracker(**kwargs)
    return _MinimumTracker(**kwargs)


def estimate_noise_psd(power: np.ndarray, method: str = 'mcra',
                       **kwargs) -> np.ndarray:
    """Noise power spectrum for every frame of ``power`` ``(frames, bins)``.

    See :data:`NOISE_ESTIMATORS` for what each one assumes. It is as large a
    lever as the gain rule — measured at 6.4 dB of SI-SDR against 5.8 dB on
    this repo's fixture — because a gain rule can only be as good as the noise
    it is told about.
    """
    if method not in NOISE_ESTIMATORS:
        raise FeatureError(
            f"unknown noise estimator {method!r}. "
            f"Choose from {', '.join(sorted(NOISE_ESTIMATORS))}")
    power = np.maximum(np.asarray(power, dtype=np.float64), _EPS)
    if method == 'quantile':
        level = np.quantile(power, kwargs.get('quantile', 0.15), axis=0)
        return np.tile(level, (power.shape[0], 1))

    tracker = make_noise_tracker(method, **kwargs)
    return np.stack([tracker.update(frame) for frame in power])


# ----------------------------------------------------------------- gain rules

class DecisionDirected:
    """The a priori SNR estimator of Ephraim & Malah, as state plus an update.

    ``gain_fn(xi, gamma) -> gain`` is the only thing that changes between
    Wiener, MMSE-STSA, log-MMSE and OM-LSA; everything around it — the a
    posteriori SNR, the smoothing, the feedback of the previous frame's
    estimate — is identical, and is why they can be compared at all. It is also
    already recursive, which is what makes those four streamable for free.
    """

    def __init__(self, gain_fn, alpha: float = 0.98):
        self.gain_fn, self.alpha = gain_fn, alpha
        self._previous = None             # |A(k, l-1)|^2, the fed-back estimate

    def update(self, power: np.ndarray, noise: np.ndarray) -> np.ndarray:
        gamma = np.minimum(power / noise, _GAMMA_MAX)
        instant = np.maximum(gamma - 1.0, 0.0)
        if self._previous is None:
            xi = np.maximum(instant, _XI_MIN)          # nothing to feed back yet
        else:
            xi = np.maximum(
                self.alpha * self._previous / noise + (1.0 - self.alpha) * instant,
                _XI_MIN)
        gain = self.gain_fn(xi, gamma)
        self._previous = (gain ** 2) * power
        return gain


def _decision_directed(power: np.ndarray, noise: np.ndarray, alpha: float,
                       gain_fn) -> np.ndarray:
    """The offline path: a loop over :class:`DecisionDirected`."""
    state = DecisionDirected(gain_fn, alpha)
    return np.stack([state.update(power[n], noise[n])
                     for n in range(power.shape[0])])


def _gain_wiener(xi, gamma):
    return xi / (1.0 + xi)


def _gain_mmse_stsa(xi, gamma):
    """Ephraim & Malah 1984, eq. 7.

    ``G = (sqrt(pi)/2) * (sqrt(v)/gamma) * exp(-v/2) *
          [(1+v) I0(v/2) + v I1(v/2)]``

    ``exp(-v/2) I_n(v/2)`` overflows on its own for large v, so it is evaluated
    through `scipy.special.ive`, which is exactly that product. As v grows the
    whole thing tends to the Wiener gain.
    """
    from scipy.special import ive
    v = np.minimum(xi / (1.0 + xi) * gamma, 700.0)
    half = v / 2.0
    gain = (np.sqrt(np.pi) / 2.0) * (np.sqrt(v) / np.maximum(gamma, _EPS)) * \
        ((1.0 + v) * ive(0, half) + v * ive(1, half))
    # the series is numerically hopeless for very large v; it is Wiener there
    return np.where(np.isfinite(gain) & (v < 500.0), gain, xi / (1.0 + xi))


def _gain_logmmse(xi, gamma):
    """Ephraim & Malah 1985: ``G = xi/(1+xi) * exp(0.5 * E1(v))``."""
    from scipy.special import exp1
    a = xi / (1.0 + xi)
    v = np.maximum(np.minimum(a * gamma, 700.0), _EPS)
    return a * np.exp(0.5 * exp1(v))


def _gain_omlsa_factory(q: float, g_min: float):
    """log-MMSE weighted by speech presence (Cohen 2001).

    ``p = 1 / (1 + q/(1-q) * (1+xi) * exp(-v))`` and
    ``G = G_LSA^p * G_min^(1-p)`` — a geometric mean, so a bin the detector is
    unsure about lands between "suppress" and "keep" rather than flipping.
    """
    ratio = q / max(1.0 - q, _EPS)

    def gain(xi, gamma):
        a = xi / (1.0 + xi)
        v = np.maximum(np.minimum(a * gamma, 700.0), _EPS)
        lsa = _gain_logmmse(xi, gamma)
        p = 1.0 / (1.0 + ratio * (1.0 + xi) * np.exp(-v))
        return np.clip(lsa, _EPS, None) ** p * (g_min ** (1.0 - p))
    return gain


def _subtraction_gains(power: np.ndarray, noise: np.ndarray,
                       over_subtraction: float,
                       spectral_floor: float) -> np.ndarray:
    """Power spectral subtraction, expressed as a gain so it composes."""
    estimate = power - over_subtraction * noise
    estimate = np.maximum(estimate, spectral_floor * power)
    return np.sqrt(estimate / np.maximum(power, _EPS))


def _band_alpha(snr_db: np.ndarray) -> np.ndarray:
    """Per-band over-subtraction from the band's SNR.

    Suppress hard where there is nothing to lose and barely at all where the
    band is mostly speech: 4.75 below -5 dB, sliding to 1.0 above 20 dB.
    """
    return np.clip(np.where(snr_db < -5.0, 4.75,
                            np.where(snr_db > 20.0, 1.0,
                                     4.0 - 3.0 * snr_db / 20.0)), 1.0, 4.75)


def _multiband_gains(power: np.ndarray, noise: np.ndarray, bands: int,
                     spectral_floor: float) -> np.ndarray:
    """Spectral subtraction with a separate factor per frequency band."""
    n_frames, n_bins = power.shape
    edges = np.linspace(0, n_bins, int(bands) + 1).astype(int)
    estimate = np.empty_like(power)
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi <= lo:
            continue
        band_power = power[:, lo:hi]
        band_noise = noise[:, lo:hi]
        snr_db = 10.0 * np.log10(
            np.maximum(band_power.sum(axis=1), _EPS)
            / np.maximum(band_noise.sum(axis=1), _EPS))
        alpha = _band_alpha(snr_db)[:, None]
        estimate[:, lo:hi] = band_power - alpha * band_noise
    estimate = np.maximum(estimate, spectral_floor * power)
    return np.sqrt(estimate / np.maximum(power, _EPS))


def _smooth_mask(mask: np.ndarray, freq_bins: int, time_frames: int) -> np.ndarray:
    """Separable box blur over (time, frequency). An unsmoothed mask warbles."""
    out = mask
    if time_frames > 1:
        kernel = np.ones(time_frames) / time_frames
        out = np.apply_along_axis(
            lambda col: np.convolve(col, kernel, mode='same'), 0, out)
    if freq_bins > 1:
        kernel = np.ones(freq_bins) / freq_bins
        out = np.apply_along_axis(
            lambda row: np.convolve(row, kernel, mode='same'), 1, out)
    return out


def _spectral_gate_gains(power: np.ndarray, noise: np.ndarray, *,
                         n_std_thresh: float, prop_decrease: float,
                         freq_smooth_hz: float, time_smooth_ms: float,
                         sr: int, n_fft: int, hop_length: int) -> np.ndarray:
    """Keep bins above (noise mean + n sigma); attenuate the rest."""
    db = 10.0 * np.log10(np.maximum(power, _EPS))
    noise_db = 10.0 * np.log10(np.maximum(noise, _EPS))
    spread = noise_db.std(axis=0, keepdims=True)
    if not np.any(spread > 0):                 # a held-constant noise estimate
        spread = db.std(axis=0, keepdims=True)
    threshold = noise_db + n_std_thresh * spread

    mask = (db > threshold).astype(np.float64)
    freq_width = max(1, int(round(freq_smooth_hz / (sr / n_fft))))
    time_width = max(1, int(round(time_smooth_ms / 1000.0 * sr / hop_length)))
    mask = _smooth_mask(mask, freq_width, time_width)
    return 1.0 - prop_decrease * (1.0 - np.clip(mask, 0.0, 1.0))


#: Methods whose gain is a function of (xi, gamma) alone, and which therefore
#: run inside :class:`DecisionDirected`. The other three look at the whole
#: spectrum at once and are handled separately.
RECURSIVE_METHODS = ('wiener', 'mmse_stsa', 'logmmse', 'omlsa')


def make_gain_rule(method: str, settings: Optional[dict] = None):
    """The per-frame gain function for one of :data:`RECURSIVE_METHODS`.

    Exposed so the streaming enhancer resolves a method name the same way the
    offline one does, instead of keeping its own mapping to fall out of step.
    """
    if method not in RECURSIVE_METHODS:
        raise FeatureError(
            f"{method!r} is not a per-frame gain rule. "
            f"Choose from {', '.join(RECURSIVE_METHODS)}")
    settings = {**METHODS[method].defaults(), **(settings or {})}
    if method == 'omlsa':
        g_min = 10.0 ** (settings.get('gain_floor_db', -25.0) / 20.0)
        return _gain_omlsa_factory(settings['q'], g_min)
    return {'wiener': _gain_wiener,
            'mmse_stsa': _gain_mmse_stsa,
            'logmmse': _gain_logmmse}[method]


# -------------------------------------------------------------------- the API

def spectral_enhance(y: np.ndarray, sr: int, method: str = 'logmmse', *,
                     noise: str = 'mcra',
                     n_fft: int = 512,
                     hop_length: Optional[int] = None,
                     window: str = 'hann',
                     noise_kwargs: Optional[dict] = None,
                     **params) -> np.ndarray:
    """Enhance *y* with one of the methods in :data:`METHODS`.

    >>> clean = kudio.spectral_enhance(y, sr)                       # logmmse
    >>> clean = kudio.spectral_enhance(y, sr, method='omlsa', q=0.4)
    >>> clean = kudio.spectral_enhance(y, sr, noise='quantile')

    The output is the **same length** as the input, so it lines up sample for
    sample with what it came from. (`trad_enhance` returns 128 samples fewer,
    which is a small thing until you try to A/B them.)

    :param method: see :data:`METHODS`; each entry lists its own parameters.
    :param noise: see :data:`NOISE_ESTIMATORS`. The estimator matters at least
        as much as the gain rule.
    :param params: method parameters, defaulted from ``METHODS[method]``.
    :raises FeatureError: on an unknown method, non-mono audio or a clip too
        short to frame.
    """
    if method not in METHODS:
        raise FeatureError(
            f"unknown method {method!r}. Choose from "
            f"{', '.join(sorted(METHODS))}")
    if sr <= 0:
        raise FeatureError(f"spectral_enhance() needs a positive rate, got {sr}")

    y = np.ascontiguousarray(np.asarray(y, dtype=np.float32).squeeze())
    if y.ndim != 1:
        raise FeatureError(
            f"spectral_enhance() takes mono audio, got shape {y.shape}")
    if y.size < n_fft:
        raise FeatureError(
            f"clip is shorter than one {n_fft}-sample frame ({y.size} samples); "
            f"pass a smaller n_fft")

    hop_length = hop_length or n_fft // 4
    settings = METHODS[method].defaults()
    unknown = set(params) - set(settings)
    if unknown:
        raise FeatureError(
            f"{method!r} does not take {', '.join(sorted(unknown))}. "
            f"It takes: {', '.join(sorted(settings)) or 'nothing'}")
    settings.update(params)

    from kudio.core.stft import STFT
    stft = STFT(sr=sr, n_fft=n_fft, hop_length=hop_length, window=window)
    spec = stft.analyse(y)                                  # (frames, bins)
    power = np.maximum(np.abs(spec) ** 2, _EPS)
    noise_psd = estimate_noise_psd(power, noise, **(noise_kwargs or {}))

    g_min = 10.0 ** (settings.get('gain_floor_db', -25.0) / 20.0)

    if method == 'specsub':
        gains = _subtraction_gains(power, noise_psd,
                                   settings['over_subtraction'],
                                   settings['spectral_floor'])
    elif method == 'multiband':
        gains = _multiband_gains(power, noise_psd, settings['bands'],
                                 settings['spectral_floor'])
    elif method == 'spectral_gate':
        gains = _spectral_gate_gains(
            power, noise_psd, n_std_thresh=settings['n_std_thresh'],
            prop_decrease=settings['prop_decrease'],
            freq_smooth_hz=settings['freq_smooth_hz'],
            time_smooth_ms=settings['time_smooth_ms'],
            sr=sr, n_fft=n_fft, hop_length=hop_length)
        g_min = 0.0                     # the gate's own strength is the floor
    else:
        gains = _decision_directed(power, noise_psd, settings['alpha'],
                                   make_gain_rule(method, settings))

    gains = np.clip(gains, g_min, 1.0)
    return stft.synthesise(spec * gains, length=len(y))


# ------------------------------------------------------------------ comparison

@dataclass(frozen=True)
class CustomEnhancer:
    """Anything that denoises, wrapped so it can be ranked with the rest.

    The statistical methods are named by a string; a trained model, a wrapper
    around someone else's library, or a one-off experiment is a function. Both
    belong in the same table — "is the model actually better than log-MMSE"
    is the question people have, and answering it by running two tools and
    comparing two printouts is how it goes unanswered.

    >>> from kudio_enhance.inference import Enhancer      # doctest: +SKIP
    >>> model = Enhancer.load("runs/exp1")                # doctest: +SKIP
    >>> kudio.compare_enhancers(y, sr, ['logmmse', ('model', model.enhance)])
    """
    name: str
    fn: Any = field(repr=False)
    label: str = ""

    def __call__(self, y: np.ndarray, sr: int) -> np.ndarray:
        return self.fn(y, sr)


def _as_enhancer(item):
    """Normalise one entry of ``methods`` to a name or a CustomEnhancer."""
    if isinstance(item, str):
        if item not in METHODS:
            raise FeatureError(
                f"unknown method {item!r}. Choose from "
                f"{', '.join(sorted(METHODS))}, or pass (name, callable)")
        return item
    if isinstance(item, CustomEnhancer):
        return item
    if isinstance(item, tuple) and len(item) == 2:
        name, fn = item
        if not callable(fn):
            raise FeatureError(f"{name!r}: expected (name, callable)")
        return CustomEnhancer(name=str(name), fn=fn)
    if callable(item):
        return CustomEnhancer(name=getattr(item, '__name__', 'custom'), fn=item)
    raise FeatureError(
        f"cannot compare {item!r}: give a method name, a callable, or "
        f"(name, callable)")


@dataclass(frozen=True)
class EnhanceResult:
    """One run's output, with everything needed to rank it against another.

    A run is a **(method, noise estimator) pair**, not just a method: the two
    are separate choices of comparable size, so collapsing them into one name
    would hide half of what produced the result.

    The reference metrics are ``None`` when no clean reference was given — the
    normal case for a real recording, and the reason `noise_floor_db` and
    `lufs` are always filled in.
    """
    method: str
    audio: np.ndarray = field(repr=False)
    seconds: float
    lufs: float
    noise_floor_db: float
    noise: str = 'mcra'
    snr_db: Optional[float] = None
    si_sdr_db: Optional[float] = None
    delta_snr_db: Optional[float] = None
    pesq: Optional[float] = None
    stoi: Optional[float] = None
    error: Optional[str] = None

    @property
    def label(self) -> str:
        """``method + noise`` — what actually distinguishes one run."""
        return f"{self.method} + {self.noise}"

    def __str__(self) -> str:  # pragma: no cover - presentation only
        if self.error:
            return f"{self.label}: failed — {self.error}"
        parts = [f"{self.label:<26}", f"{self.seconds * 1000:6.0f} ms",
                 f"floor {self.noise_floor_db:+6.1f} dB"]
        if self.si_sdr_db is not None:
            parts.append(f"SI-SDR {self.si_sdr_db:+6.2f} dB")
        if self.delta_snr_db is not None:
            parts.append(f"dSNR {self.delta_snr_db:+5.2f} dB")
        if self.pesq is not None:
            parts.append(f"PESQ {self.pesq:.2f}")
        return "  ".join(parts)


def compare_enhancers(y: np.ndarray, sr: int,
                      methods: Optional[Sequence[str]] = None, *,
                      noises: Optional[Sequence[str]] = None,
                      reference: Optional[np.ndarray] = None,
                      metrics: bool = True,
                      pesq_mode: str = 'auto',
                      pesq_scale: str = 'lqo',
                      **common) -> List[EnhanceResult]:
    """Run several methods over the same clip and measure each one.

    >>> for result in kudio.compare_enhancers(y, sr, reference=clean):
    ...     print(result)

    **Both axes can be swept.** `methods` chooses gain rules, `noises` chooses
    noise estimators, and giving both compares the full grid:

    >>> kudio.compare_enhancers(y, sr, ['logmmse'], noises=list(NOISE_ESTIMATORS))
    >>> kudio.compare_enhancers(y, sr, noises=['mcra', 'quantile'])   # 2 x all

    That second axis is not decoration. On this repo's fixture the two are
    **the same size** — a mean spread of 5.8 dB SI-SDR across gain rules
    against 6.4 dB across estimators — so a comparison that sweeps only
    `methods` is looking at half the problem. Which of the two dominates
    depends on the clip, which is the reason to measure rather than assume.

    With a *reference* you get SNR, SI-SDR and (where installed) PESQ / STOI,
    plus **`delta_snr_db`: the improvement over the untouched input**, which is
    the number that actually answers "did this help". Without one you get the
    residual noise floor and the loudness, which is what a real recording
    allows.

    A run that raises is reported with `error` set rather than taking the whole
    comparison down — a comparison exists to find out which ones work.

    :param common: passed to every run (`n_fft=`, `alpha=`, ...). Anything a
        given method does not accept is dropped for that method, so one call
        can hold the framing still across all of them.
    :raises FeatureError: on an unknown method or estimator, or on passing
        ``noise=`` and ``noises=`` together.
    """
    entries = [_as_enhancer(m) for m in
               (methods if methods is not None else list(METHODS))]
    names = [e for e in entries if isinstance(e, str)]

    if noises is not None and 'noise' in common:
        raise FeatureError(
            "pass noises=[...] to sweep the estimator, or noise='...' to fix "
            "it — not both")
    estimators = list(noises) if noises is not None \
        else [common.pop('noise', 'mcra')]
    unknown = [n for n in estimators if n not in NOISE_ESTIMATORS]
    if unknown:
        raise FeatureError(f"unknown noise estimator(s): {', '.join(unknown)}")

    y = np.asarray(y, dtype=np.float32)
    baseline = (_reference_metrics(reference, y, sr, metrics, pesq_mode,
                                   pesq_scale)
                if reference is not None else None)

    # Warm up before timing anything. librosa's STFT is JIT-compiled and scipy's
    # Bessel/exponential-integral functions import lazily, so whichever run
    # went first was charged ~3 s of one-time cost -- in a table whose entire
    # purpose is comparing runs against each other.
    _warm_up(y, sr, names, estimators[0])

    results: List[EnhanceResult] = []
    for entry in entries:
        if isinstance(entry, CustomEnhancer):
            # it brings its own everything, so the estimator sweep does not
            # apply and saying 'mcra' beside it would be a lie
            results.append(_run_one(entry.name, CUSTOM_NOISE,
                                    lambda: entry(y, sr),
                                    y, sr, reference, baseline, metrics,
                                    pesq_mode, pesq_scale))
            continue

        allowed = set(METHODS[entry].defaults())
        kwargs = {k: v for k, v in common.items()
                  if k in allowed or k in ('n_fft', 'hop_length', 'window',
                                           'noise_kwargs')}
        for estimator in estimators:
            results.append(_run_one(
                entry, estimator,
                lambda name=entry, est=estimator:
                    spectral_enhance(y, sr, name, noise=est, **kwargs),
                y, sr, reference, baseline, metrics,
                pesq_mode, pesq_scale))
    return results


def _run_one(name, estimator, call, y, sr, reference, baseline, metrics,
             pesq_mode='auto', pesq_scale='lqo') -> EnhanceResult:
    """Time one run and measure it, reporting a failure rather than raising."""
    started = time.perf_counter()
    try:
        out = call()
    except Exception as e:                                  # noqa: BLE001
        log.warning("%s + %s failed: %s", name, estimator, e)
        return EnhanceResult(
            method=name, noise=estimator, audio=np.zeros(0, dtype=np.float32),
            seconds=time.perf_counter() - started, lufs=float('nan'),
            noise_floor_db=float('nan'), error=f"{type(e).__name__}: {e}")
    elapsed = time.perf_counter() - started
    return _measure(name, estimator, np.asarray(out, dtype=np.float32), y, sr,
                    elapsed, reference, baseline, metrics,
                    pesq_mode, pesq_scale)


@dataclass(frozen=True)
class EnhanceFolderResult(ConvertResult):
    """What :func:`enhance_folder` did.

    A :class:`kudio.ConvertResult` — same `written` / `total` / `failed` — plus
    `floors`, the before-and-after noise floor of each file when it was asked
    for. Subclassing rather than inventing a parallel type keeps one shape for
    "I processed a folder".
    """
    #: ``(path, before_dbfs, after_dbfs)`` per file; empty unless ``report=True``
    floors: Tuple[Tuple[Any, float, float], ...] = ()

    def mean_reduction_db(self) -> Optional[float]:
        """Average drop in noise floor, or ``None`` without a report."""
        usable = [(b, a) for _, b, a in self.floors
                  if b is not None and a is not None
                  and np.isfinite(b) and np.isfinite(a)]
        if not usable:
            return None
        return float(np.mean([b - a for b, a in usable]))


def enhance_folder(src, dst, method='logmmse', *,
                   noise: str = 'mcra',
                   sr: Optional[int] = None,
                   subtype: str = 'PCM_16',
                   overwrite: bool = False,
                   progress=None,
                   on_error: str = 'collect',
                   report: bool = False,
                   **params):
    """Denoise every audio file under *src* into *dst*.

    The counterpart of :func:`kudio.convert_folder`, and it mirrors the input's
    directory structure the same way, so two files with the same name in
    different subfolders do not collide.

    >>> result = kudio.enhance_folder("noisy/", "clean/", 'logmmse')
    >>> print(result)
    412/412 written, 0 failed

    :param method: a name from :data:`METHODS`, or anything
        :class:`CustomEnhancer` accepts — a callable, or ``(name, callable)``.
        A trained model is the obvious second case.
    :param report: also measure each file's noise floor before and after, and
        return them on the result. Roughly doubles the work; worth it when the
        question is "did this help" rather than "run it".
    :param params: method parameters, defaulted from ``METHODS[method]``. A
        callable takes none — it was configured before it got here.

    :returns: an :class:`EnhanceFolderResult`.
    """
    from kudio.core.io import WAVE_SUFFIX, check_file, check_input, \
        file_load, save_wave
    from kudio.core.report import audio_report
    from kudio.exceptions import AudioIOError

    entry = _as_enhancer(method)
    if on_error not in ('collect', 'raise'):
        raise FeatureError(f"on_error must be 'collect' or 'raise', got {on_error!r}")
    if isinstance(entry, CustomEnhancer):
        if params:
            raise FeatureError(
                f"{entry.name!r} is a callable — it takes no method "
                f"parameters, but got {', '.join(sorted(params))}")
        run = entry
    else:
        # Validate the parameters here rather than letting each file trip over
        # them: with on_error='collect' a typo would otherwise come back as 412
        # identical failures instead of one sentence about the typo.
        unknown = set(params) - set(METHODS[entry].defaults())
        if unknown:
            raise FeatureError(
                f"{entry!r} does not take {', '.join(sorted(unknown))}. It "
                f"takes: {', '.join(sorted(METHODS[entry].defaults())) or 'nothing'}")

        def run(y, rate):
            return spectral_enhance(y, rate, entry, noise=noise, **params)

    files, _ = check_input(src)
    if not files:
        raise AudioIOError(f"no audio files found in {src!r}")

    root = Path(src) if isinstance(src, (str, PurePath)) and Path(src).is_dir() else None
    out_root = Path(dst)
    out_root.mkdir(parents=True, exist_ok=True)

    outputs, failed, floors = [], [], []
    for n, f in enumerate(files, 1):
        try:
            y, rate = file_load(f, sr=sr, mono=True)
            before = audio_report(y, rate).noise_floor_dbfs if report else None
            out = np.asarray(run(y, rate), dtype=np.float32)
            after = audio_report(out, rate).noise_floor_dbfs if report else None

            rel = f.relative_to(root) if root is not None else Path(f.name)
            target = (out_root / rel).with_suffix(WAVE_SUFFIX)
            target = Path(check_file(target, rename=not overwrite))
            save_wave(target, out, rate, subtype=subtype)
            outputs.append(target)
            if report:
                floors.append((target, before, after))
        except Exception as e:                              # noqa: BLE001
            if on_error == 'raise':
                raise
            failed.append((Path(f), f"{type(e).__name__}: {e}"))
        if progress is not None:
            progress(n, len(files), Path(f))

    return EnhanceFolderResult(written=len(outputs), total=len(files),
                               outputs=outputs, failed=failed,
                               floors=tuple(floors))


def _warm_up(y: np.ndarray, sr: int, names: Sequence[str], noise: str) -> None:
    """Pay every one-time import and JIT cost before the clock starts."""
    sample = y[:min(len(y), sr)] if len(y) else y
    if sample.size < 512:
        return
    for name in names:
        try:
            spectral_enhance(sample, sr, name, noise=noise)
        except Exception:                                   # noqa: BLE001
            pass          # a run that cannot work will be reported properly


def _measure(name, estimator, out, noisy, sr, elapsed, reference, baseline,
             metrics, pesq_mode='auto',
             pesq_scale='lqo') -> EnhanceResult:
    from kudio.core.report import audio_report

    try:
        report = audio_report(out, sr)
        floor, lufs = report.noise_floor_dbfs, report.lufs
    except Exception:                                       # noqa: BLE001
        floor = lufs = float('nan')

    snr = si_sdr = delta = pesq = stoi = None
    if reference is not None:
        scores = _reference_metrics(reference, out, sr, metrics,
                                    pesq_mode, pesq_scale)
        snr, si_sdr, pesq, stoi = scores
        if baseline is not None and snr is not None and baseline[0] is not None:
            delta = snr - baseline[0]
    return EnhanceResult(method=name, noise=estimator, audio=out,
                         seconds=elapsed, lufs=lufs, noise_floor_db=floor,
                         snr_db=snr, si_sdr_db=si_sdr, delta_snr_db=delta,
                         pesq=pesq, stoi=stoi)


def _reference_metrics(reference, degraded, sr, metrics,
                       pesq_mode='auto', pesq_scale='lqo'):
    """``(snr, si_sdr, pesq, stoi)`` over the overlapping samples."""
    from kudio.core.evaluator import si_sdr as _si_sdr, snr as _snr

    ref = np.asarray(reference, dtype=np.float32).squeeze()
    deg = np.asarray(degraded, dtype=np.float32).squeeze()
    n = min(len(ref), len(deg))
    if n == 0:
        return (None, None, None, None)
    ref, deg = ref[:n], deg[:n]

    try:
        snr = float(_snr(ref, deg))
        si_sdr = float(_si_sdr(ref, deg))
    except Exception:                                       # noqa: BLE001
        snr = si_sdr = None
    pesq = stoi = None
    if metrics:
        try:
            from kudio.core.evaluator import eval_metrics
            pesq, stoi, _ = eval_metrics(sr, ref, deg, pesq_mode=pesq_mode,
                                         pesq_scale=pesq_scale)
        except Exception as e:                              # noqa: BLE001
            log.debug("optional metrics unavailable: %s", e)
    return (snr, si_sdr, pesq, stoi)
