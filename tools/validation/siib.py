# -*- coding: utf-8 -*-
"""Check kudio's SIIB and SIIB^Gauss (kudio/core/siib.py) against the two
measures worked out a second way here, stage by stage, from the resampler to
the score.

Run it from anywhere; it imports the kudio of the checkout it sits in::

    python tools/validation/siib.py
    python tools/validation/siib.py --all

It needs numpy and scipy; I ran it with numpy 2.0.2 and scipy 1.13.1 on
Python 3.9.13, and the default run also with numpy 2.5.4 and scipy 1.18.1
on Python 3.13.9. With pySIIB installed (pip install
git+https://github.com/kamo-naoyuki/pySIIB.git, which compiles C code) it
also compares kudio with pySIIB; I ran that with pySIIB at commit 226c2f3 of
2022-04-04, numpy 2.2.6 and scipy 1.15.2 on Python 3.10.22. Without it,
those rows are skipped.

The second way is written to be read rather than to be fast, and shares no
code with kudio's module. The resampler is MATLAB's design, a sinc under a
Kaiser window of beta 5, ten samples of the slower rate either side, summed
tap by tap; the spectra come from an explicit DFT matrix; the masking is a
loop over its 16 lags and the stacks are filled one by one; and each
channel's information is the second estimator of Kraskov, Stögbauer and
Grassberger (their Eq. 9) over all N^2 distances, with scipy's digamma.
kudio follows the authors' MATLAB code, SIIB.m and SIIB_Gauss.m, where it
and the papers differ, and so does this. SIIB.m, SIIB_Gauss.m, MILCA's
MIxnyn and pySIIB, the Python port, are GPL-3.0, and none of their code is
here or in kudio. pySIIB runs beside kudio, through its SIIB() call alone, as
something to compare with; SIIB.m needs MATLAB and does not run here.

Every signal is made here from fixed seeds: syllables as
tests/test_siib.py makes them (a harmonic source through three moving
formants, two in five led by a burst of hiss), on a floor 54 dB under the
peak that the 40 dB voice-activity threshold drops, or 24 dB under it,
which the threshold keeps; under white noise, in a room with an RT60 of
0.5 s, low-passed at 2 kHz, made at 8, 44.1 and 48 kHz, and with a fade to
-90 dB and 3 s of digital silence a third of the way in. The default run
has four pairs of 12 s; --all has ten of 30 s.

What it checks, on each pair:

1. The resampler written out against scipy's resample_poly, which kudio
   calls, at every rate but 16 kHz.
2. On the same 16 kHz signals: the 28 band centres, widths and weights, the
   window, the masking weights, each frame's level and the voice-activity
   decision, the band powers, the masked log-spectra and the stacks.
3. The eigenvalues of the reference's covariance, and each eigenchannel of
   both signals against that channel's own spread, the signs matched first.
4. SIIB^Gauss: each channel's squared correlation, and the score.
5. SIIB: kudio's neighbour search against the all-pairs one on the same
   channels; each channel's information, with no noise added to break ties;
   and the score, against that and against the chain SIIB.m runs, noise,
   8 significant digits and the answer's 8 decimals included, drawn from
   another generator.
6. On the first pair: a copy and an inverted copy, which score the
   ceiling, an output of digital silence, and an offset of 0.5 on both
   signals.
7. With pySIIB: kudio's SIIB and SIIB^Gauss against pySIIB's on every pair
   and on the first pair's copy, inverted copy and offset, checked at
   16 kHz, SIIB^Gauss to 1e-11 and SIIB to 1e-4, the noise that breaks
   ties. At other rates pySIIB resamples with a filter of its own, and the
   differences are printed but not checked; so is an output of digital
   silence. pySIIB's default window, 'hanning', is a name scipy 1.15 no
   longer knows, so it is given 'hann', the window the name meant.

Then, as the module notes quote it, how far the scores move when the pair
goes to 16 kHz through a sharper filter than SIIB.m's, a sinc under a
Kaiser window for 60 dB (beta 0.1102 (60 - 8.7)), 365 taps from 20 kHz and
147 from 8 kHz. The 36 s of test speech of tests/test_siib.py are taken to
20 kHz and to 8 kHz by a near-ideal filter (601 taps, Kaiser beta 10), and
to 8 kHz once more by resample_poly's own filter; white noise from
default_rng(7) is added at that rate at -10 to 20 dB in 5 dB steps. With
pySIIB installed, its scores are printed beside them.

It prints both scores for each pair, then every check with its largest
difference and its tolerance, and exits with 1 if one is past it. The
default run takes about two minutes, half of it in the resampler sweep;
--all about eleven minutes, most of it in the all-pairs search.

References: S. Van Kuyk, W. B. Kleijn and R. C. Hendriks, "An instrumental
intelligibility metric based on information theory", IEEE Signal Process.
Lett. 25(1), pp. 115-119, 2018, doi:10.1109/LSP.2017.2774250; "An
evaluation of intrusive instrumental intelligibility metrics", IEEE/ACM
Trans. Audio, Speech, Lang. Process. 26(11), pp. 2153-2166, 2018,
doi:10.1109/TASLP.2018.2856374. A. Kraskov, H. Stögbauer and P.
Grassberger, "Estimating mutual information", Phys. Rev. E 69(6), 066138,
2004, doi:10.1103/PhysRevE.69.066138.
"""
from __future__ import annotations

import argparse
import importlib
import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.signal import butter, firwin, lfilter, resample_poly, sosfilt
from scipy.special import digamma, i0

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import kudio  # noqa: E402

# the module, for its stages: kudio.core binds the name siib to the function
km = importlib.import_module('kudio.core.siib')

# pySIIB, where it is installed: GPL-3.0, so it runs beside kudio as something
# to compare with, and none of it is in kudio
try:
    from pysiib import SIIB as pysiib_siib
except ImportError:
    pysiib_siib = None

EPS = 2.0 ** -52
RATE = 16000
FRAME, HOP, BINS = 400, 200, 201
FRAMES_PER_SECOND = RATE / HOP
STACK = 15
R_P = 0.75
I_MAX = -0.5 * math.log2(1 - R_P ** 2)
CEILING = FRAMES_PER_SECOND / STACK * 420 * I_MAX

#: (check, cases, largest difference, tolerance)
RESULTS = []


def report(check, cases, diff, tolerance):
    RESULTS.append((check, cases, diff, tolerance))


def relative(a, b):
    """The largest difference of *a* from *b*, over the largest of |b|."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        return math.inf
    scale = float(np.max(np.abs(b))) if b.size else 0.0
    diff = float(np.max(np.abs(a - b))) if a.size else 0.0
    return diff / scale if scale else (0.0 if diff == 0.0 else math.inf)


def absolute(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.max(np.abs(a - b))) if a.shape == b.shape else math.inf


def quietly(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return fn(*args, **kwargs)


# --- the measures, the second way --------------------------------------------------

def round_half_up(x):
    """MATLAB's round, for the positive numbers it meets here."""
    return int(math.floor(x + 0.5))


def resample(s, sr):
    """*s* at *sr* taken to 16 kHz: z[n] = sum_m h[m] u[n q + 10 M - m],
    u being *s* with p - 1 zeros after each sample, h MATLAB's design."""
    s = np.asarray(s, dtype=float)
    if sr == RATE:
        return s.copy()
    g = math.gcd(RATE, sr)
    p, q = RATE // g, sr // g
    m_most = max(p, q)
    m = np.arange(20 * m_most + 1)
    h = np.sinc((m - 10 * m_most) / m_most) / m_most
    h *= i0(5.0 * np.sqrt(1.0 - ((m - 10 * m_most) / (10 * m_most)) ** 2)) / i0(5.0)
    h *= p / np.sum(h)
    n_out = -(-len(s) * p // q)
    out = np.zeros(n_out)
    taps = np.arange(0, 20 * m_most + 1, p)
    for start in range(0, n_out, 4096):
        n = np.arange(start, min(n_out, start + 4096))
        base = n * q + 10 * m_most
        # only the stuffed samples that are multiples of p hold input
        mm = (base % p)[:, None] + taps[None, :]
        index = (base[:, None] - mm) // p
        valid = (mm <= 20 * m_most) & (index >= 0) & (index < len(s))
        out[n] = np.sum(np.where(valid, h[np.minimum(mm, 20 * m_most)], 0.0)
                        * np.where(valid, s[np.clip(index, 0, len(s) - 1)], 0.0), axis=1)
    return out


def frames(s):
    count = (len(s) - FRAME - 1) // HOP + 1 if len(s) > FRAME else 0
    return np.array([s[HOP * t:HOP * t + FRAME] for t in range(count)]).reshape(count, FRAME)


def hann():
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(FRAME) / FRAME)


def power_spectrogram(s):
    """``(201, frames)``, the DFT taken as a matrix product."""
    dft = np.exp(-2j * np.pi * np.outer(np.arange(BINS), np.arange(FRAME)) / FRAME)
    return np.abs(dft @ (frames(s) * hann()).T) ** 2


def voice_activity(x16):
    level = 10 * np.log10(np.sum((frames(x16) * hann()) ** 2, axis=1) / FRAME + EPS)
    loud = np.sort(level)[round_half_up(0.999 * len(level)) - 1]
    return level > loud - 40.0, level


def filterbank():
    """``(weights, centres, widths)``: 28 bands on equal steps of ERB rate."""
    erb = lambda f: 21.4 * np.log10(1 + 4.37 * f / 1000)   # noqa: E731
    low, high = erb(100.0), erb(6500.0)
    bands = round_half_up(high - low)
    rate = np.array([low + j * (high - low) / (bands - 1) for j in range(bands)])
    centres = 1000 * (10 ** (rate / 21.4) - 1) / 4.37
    widths = 3.2 / math.pi * 24.7 * (1 + 4.37 * centres / 1000)
    bins = 40.0 * np.arange(BINS)
    weights = np.zeros((bands, BINS))
    for j in range(bands):
        g = (widths[j] ** 2 + (bins - centres[j]) ** 2) ** -2.0
        g /= np.max(g)
        g[g < 0.001] = 0.0
        weights[j] = g
    return weights, centres, widths


def masking_weights():
    return np.array([math.log(lag + 1) / math.log(16) for lag in range(16)])


def masked(log_power, floor):
    """``(bands, frames)``: each frame the highest of the curves that reach
    it, and the last one at the floor."""
    w = masking_weights()
    bands, count = log_power.shape
    out = np.full((bands, count), -np.inf)
    for t in range(count - 1):
        for lag in range(min(15, t) + 1):
            out[:, t] = np.maximum(out[:, t], (1 - w[lag]) * log_power[:, t - lag]
                                   + w[lag] * floor)
    out[:, -1] = floor
    return out


def stacked(z):
    """``(420, frames - 15)``: element m * 28 + j of stack c is z[j, c + m]."""
    bands, count = z.shape
    out = np.empty((bands * STACK, count - STACK))
    for c in range(count - STACK):
        for m in range(STACK):
            out[m * bands:(m + 1) * bands, c] = z[:, c + m]
    return out


def deviation(v):
    v = np.asarray(v, dtype=float)
    return math.sqrt(float(np.sum((v - np.sum(v) / len(v)) ** 2)) / (len(v) - 1))


def stages(x, y, sr, gauss):
    """Every stage, from the signals as given to the eigenchannels."""
    x, y = np.asarray(x, float).ravel(), np.asarray(y, float).ravel()
    sd = deviation(x)
    divisor = sd if gauss else max(sd, EPS)
    st = {'x16': resample(x / divisor, sr), 'y16': resample(y / divisor, sr)}
    keep, st['level'] = voice_activity(st['x16'])
    st['keep'] = keep
    weights, st['centres'], st['widths'] = filterbank()
    st['weights'] = weights
    st['bx'] = (weights ** 2) @ power_spectrogram(st['x16'])
    st['by'] = (weights ** 2) @ power_spectrogram(st['y16'])
    lx = np.log(st['bx'][:, keep] + EPS)
    ly = np.log(st['by'][:, keep] + EPS)
    st['lx'], st['floor'] = lx, lx.min(axis=1)
    st['mx'], st['my'] = masked(lx, st['floor']), masked(ly, st['floor'])
    st['zx'] = st['mx'] - st['mx'].mean(axis=1, keepdims=True)
    sx = stacked(st['zx'])
    sy = stacked(st['my'] - st['my'].mean(axis=1, keepdims=True))
    st['stacks'] = sx
    st['eigenvalues'], basis = np.linalg.eigh(np.cov(sx))
    st['cx'], st['cy'] = basis.T @ sx, basis.T @ sy
    return st


def ksg2(u, v, k):
    """[KSG04] Eq. 9 in nats, from all N^2 distances in the maximum norm."""
    du = np.abs(u[:, None] - u[None, :])
    dv = np.abs(v[:, None] - v[None, :])
    d = np.maximum(du, dv)
    np.fill_diagonal(d, np.inf)
    near = np.argpartition(d, k - 1, axis=1)[:, :k]
    rows = np.arange(len(u))[:, None]
    eu = du[rows, near].max(axis=1)
    ev = dv[rows, near].max(axis=1)
    nu = (du <= eu[:, None]).sum(axis=1) - 1
    nv = (dv <= ev[:, None]).sum(axis=1) - 1
    return float(digamma(len(u)) + digamma(k) - 1.0 / k
                 - np.mean(digamma(nu) + digamma(nv)))


def standardised(a):
    a = a - np.mean(a)
    return a / (np.std(a, ddof=1) + EPS)


def eight_digits(a):
    return np.array([float(f"{value:.7e}") for value in a])


def information(cx, cy, as_siib_m=False, seed=12345):
    """Each pair of channels' information in bits, ``cx`` and ``cy`` being
    ``(420, N)``. With *as_siib_m*, the chain SIIB.m runs: Gaussian noise of
    1e-10, 8 significant digits, uniform noise of 1e-8, standardised again,
    and the answer in nats to 8 decimals; drawn from a generator of its own."""
    channels, n = cx.shape
    k = max(2, -(-n // 150))
    rng = np.random.default_rng(seed)
    bits = np.empty(channels)
    for j in range(channels):
        u, v = standardised(cx[j]), standardised(cy[j])
        if as_siib_m:
            u = eight_digits(u + 1e-10 * rng.standard_normal(n)) + 1e-8 * rng.random(n)
            v = eight_digits(v + 1e-10 * rng.standard_normal(n)) + 1e-8 * rng.random(n)
            u = (u - u.mean()) / np.std(u, ddof=1)
            v = (v - v.mean()) / np.std(v, ddof=1)
            bits[j] = float(f"{ksg2(u, v, k):.8f}") / math.log(2)
        else:
            bits[j] = ksg2(u, v, k) / math.log(2)
    return bits


def siib_of(bits):
    return max(0.0, FRAMES_PER_SECOND / STACK * float(np.sum(np.minimum(I_MAX, bits))))


def squared_correlations(cx, cy):
    n = cx.shape[1]
    return np.array([(np.sum(a * b) / n) ** 2 / ((np.sum(a * a) / n) * (np.sum(b * b) / n))
                     for a, b in zip(cx, cy)])


def siib_gauss_of(rho2):
    total = -FRAMES_PER_SECOND / (2 * STACK) * float(np.sum(np.log2(1 - R_P ** 2 * rho2)))
    return max(0.0, total)


# --- the signals ----------------------------------------------------------------------

_VOWELS = ((730, 1090, 2440), (270, 2290, 3010), (300, 870, 2240),
           (530, 1840, 2480), (570, 840, 2410))


def speech(seconds, seed, sr=RATE, floor=1e-3):
    """Syllables at *sr*, as tests/test_siib.py makes them at 16 kHz, with
    pauses of 50 to 250 ms on white noise of RMS *floor*."""
    def resonator(x, f, bandwidth):
        r = math.exp(-math.pi * bandwidth / sr)
        return lfilter([1 - r], [1, -2 * r * math.cos(2 * math.pi * f / sr), r * r], x)

    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    y = np.zeros(n)
    hiss = butter(4, [3000, min(7000, 0.45 * sr)], 'bandpass', fs=sr, output='sos')
    start = int(0.1 * sr)
    while start < n - int(0.45 * sr):
        length = int(rng.uniform(0.15, 0.32) * sr)
        f0 = np.linspace(*rng.uniform(90.0, 230.0, 2), length)
        phase = 2 * np.pi * np.cumsum(f0) / sr
        voice = sum(np.sin(h * phase) / h for h in range(1, int(0.45 * sr / f0.max())))
        first, last = _VOWELS[rng.integers(5)], _VOWELS[rng.integers(5)]
        syllable = np.zeros(length)
        for j, part in enumerate(np.array_split(np.arange(length), 4)):
            piece = voice[part]
            for f_from, f_to, bandwidth in zip(first, last, (80.0, 100.0, 140.0)):
                piece = resonator(piece, f_from + (f_to - f_from) * j / 3, bandwidth)
            syllable[part] = piece
        syllable *= np.hanning(length) / np.max(np.abs(syllable))
        if rng.random() < 0.4:
            m = int(rng.uniform(0.05, 0.12) * sr)
            burst = sosfilt(hiss, rng.standard_normal(m)) * np.hanning(m)
            syllable = np.concatenate([0.3 * burst / np.max(np.abs(burst)), syllable])
        syllable *= 10 ** (rng.uniform(-8.0, 0.0) / 20)
        stop = min(start + len(syllable), n)
        y[start:stop] = syllable[:stop - start]
        start = stop + int(rng.uniform(0.05, 0.25) * sr)
    y *= 0.5 / np.max(np.abs(y))
    return y + floor * rng.standard_normal(n)


def white(n, seed):
    return np.random.default_rng(seed).standard_normal(n)


def mix(clean, noise, snr_db):
    return clean + noise * math.sqrt(np.mean(clean ** 2)
                                     / (np.mean(noise ** 2) * 10 ** (snr_db / 10)))


def room(clean, sr, rt60=0.5, seed=5):
    """A decaying tail of white noise behind a direct path, 0.6 s long."""
    t = np.arange(int(0.6 * sr)) / sr
    h = white(len(t), seed) * np.exp(-6.9078 * t / rt60)
    h[0] = 1.0
    return np.convolve(clean, h / np.sqrt(np.sum(h ** 2)))[:len(clean)]


def low_passed(clean, sr, cutoff=2000.0):
    return sosfilt(butter(8, cutoff, 'lowpass', fs=sr, output='sos'), clean)


def faded(seconds, seed):
    """Syllables that fade to -90 dB over 2 s, stop for 3 s and fade back in
    over 1 s, a third of the way in."""
    x = speech(seconds, seed, floor=0.03)
    a, b, c, d = (int(s * RATE) for s in np.array([10, 12, 15, 16]) * seconds / 30)
    envelope = np.ones_like(x)
    envelope[a:b] = 10 ** (np.linspace(0, -90, b - a) / 20)
    envelope[b:c] = 0.0
    envelope[c:d] = 10 ** (np.linspace(-90, 0, d - c) / 20)
    return x * envelope


def pairs(full):
    """``{name: (ref, deg, sr)}``."""
    seconds = 30.0 if full else 12.0
    out = {}
    for i, snr in enumerate((-5, 0, 5, 10) if full else (0,)):
        x = speech(seconds, 10 + i)
        out[f'white noise, {snr:+d} dB'] = (x, mix(x, white(len(x), 100 + i), snr), RATE)
    x = speech(seconds, 20, floor=0.03)
    out['in a room, RT60 0.5 s'] = (x, room(x, RATE), RATE)
    if full:
        out['low-passed at 2 kHz'] = (x, low_passed(x, RATE), RATE)
    x = speech(seconds, 30, 8000)
    out['8 kHz, white noise, 0 dB'] = (x, mix(x, white(len(x), 200), 0), 8000)
    x = speech(seconds, 40, 44100)
    out['44.1 kHz, white noise, 5 dB'] = (x, mix(x, white(len(x), 201), 5), 44100)
    if full:
        x = faded(seconds, 50)
        out['a fade and 3 s of silence, 5 dB'] = (x, mix(x, white(len(x), 202), 5), RATE)
        x = speech(seconds, 60, 48000, floor=0.03)
        out['48 kHz, white noise, 10 dB'] = (x, mix(x, white(len(x), 203), 10), 48000)
    return out


# --- the comparison ----------------------------------------------------------------

#: per check, (cases, largest difference) over the pairs
_WORST = {}


def keep(check, diff):
    cases, worst = _WORST.get(check, (0, 0.0))
    _WORST[check] = (cases + 1, max(worst, diff))


def matched(kx, ky, cx):
    """kudio's channels ``(N, 420)`` with each sign set to the restated one's,
    or as they are where kudio kept another number of stacks."""
    if kx.shape != cx.T.shape:
        return kx, ky
    sign = np.sign(np.sum(kx * cx.T, axis=0))
    sign[sign == 0] = 1.0
    return kx * sign, ky * sign


def against_spread(k, c, spread):
    """The largest difference in any channel over that channel's spread."""
    if k.shape != c.shape:
        return math.inf
    return float(np.max(np.abs(k - c) / spread))


def compare(name, ref, deg, sr):
    start = time.perf_counter()
    if sr != RATE:
        g = math.gcd(RATE, sr)
        x = ref / deviation(ref)
        keep('the resampler written out, against resample_poly (relative)',
             relative(resample(x, sr), resample_poly(x, RATE // g, sr // g)))
    st = stages(ref, deg, sr, gauss=False)

    centres, widths, weights = km._filterbank()
    keep('band centres, widths and weights (relative)',
         max(relative(centres, st['centres']), relative(widths, st['widths']),
             relative(weights, st['weights'] ** 2)))
    keep('window and masking weights',
         max(absolute(km._window(), hann()), absolute(km._masking_weights(),
                                                      masking_weights())))
    level, bx, by = km._band_levels(st['x16'], st['y16'])
    keep("each frame's level, in dB", absolute(level, st['level']))
    keep('voice-activity decisions that differ',
         float(np.count_nonzero(km._active(level) != st['keep'])))
    keep('band powers (relative)', max(relative(bx, st['bx'].T), relative(by, st['by'].T)))
    keep('masked log-spectra and stacks on the same input',
         max(absolute(km._masked(st['lx'].T, st['floor']), st['mx'].T),
             absolute(km._stacked(st['zx'].T), st['stacks'].T)))

    kx, ky = matched(*km._eigenchannels(ref, deg, sr, 'siib', floor_deviation=True),
                     st['cx'])
    keep('eigenvalues (relative)', relative(np.var(kx, axis=0, ddof=1), st['eigenvalues']))
    spread = np.sqrt(np.maximum(st['eigenvalues'], 1e-300))
    keep("eigenchannels, against each channel's spread",
         max(against_spread(kx, st['cx'].T, spread), against_spread(ky, st['cy'].T, spread)))

    gauss = stages(ref, deg, sr, gauss=True)
    rho2 = squared_correlations(gauss['cx'], gauss['cy'])
    gx, gy = matched(*km._eigenchannels(ref, deg, sr, 'siib_gauss', floor_deviation=False),
                     gauss['cx'])
    keep('squared correlation of each channel', absolute(squared_correlations(gx.T, gy.T),
                                                         rho2))
    ours_gauss, theirs_gauss = kudio.siib_gauss(ref, deg, sr), siib_gauss_of(rho2)
    keep('SIIB^Gauss (relative)', abs(ours_gauss - theirs_gauss) / theirs_gauss)

    # the neighbour search: kudio's and the all-pairs one, on the same channels
    # with nothing added, every seventh channel
    n = kx.shape[0]
    k = max(2, -(-n // 150))
    worst = 0.0
    for j in range(0, kx.shape[1], 7):
        u, v = standardised(kx[:, j]), standardised(ky[:, j])
        worst = max(worst, abs(km._ksg(u, v, k) - ksg2(u, v, k)))
    keep('the neighbour search on the same channels, in nats', worst)

    plain = information(st['cx'], st['cy'])
    keep("each channel's information, no noise added, in bits",
         absolute(km._channel_bits(kx, ky), plain))
    ours = kudio.siib(ref, deg, sr)
    no_noise = siib_of(plain)
    as_siib_m = siib_of(information(st['cx'], st['cy'], as_siib_m=True))
    keep('SIIB, against no noise added (relative)', abs(ours - no_noise) / no_noise)
    keep("SIIB, against SIIB.m's chain on other draws (relative)",
         abs(ours - as_siib_m) / as_siib_m)
    print(f"  {name:<34} {len(ref) / sr:4.0f} s at {sr:5d} Hz, {int(st['keep'].sum()):4d} "
          f"of {len(st['keep']):4d} frames kept, N {n:4d}, k {k:2d}   "
          f"[{time.perf_counter() - start:.0f} s]")
    print(f"  {'':<34} SIIB {ours:10.6f}  here {no_noise:10.6f} / {as_siib_m:10.6f}   "
          f"SIIB^Gauss {ours_gauss:13.9f}  here {theirs_gauss:13.9f}")


def edges(ref, deg):
    print('\nOn the first pair:')
    for label, y in (('a copy', ref), ('an inverted copy', -ref)):
        got = (kudio.siib(ref, y, RATE), kudio.siib_gauss(ref, y, RATE))
        print(f"  {label:<36} SIIB {got[0]!r}  SIIB^Gauss {got[1]!r}  (ceiling "
              f"{CEILING!r})")
        report(f'{label} scores the ceiling', 2, max(abs(v - CEILING) for v in got), 1e-9)
    silence = np.zeros_like(ref)
    ours = (quietly(kudio.siib, ref, silence, RATE), quietly(kudio.siib_gauss, ref, silence, RATE))
    here = siib_gauss_of(squared_correlations(*_channels(ref, silence, gauss=True)))
    print(f"  {'an output of digital silence':<36} SIIB {ours[0]:.6f}  SIIB^Gauss "
          f"{ours[1]:.9f}  here {here:.9f}")
    report('SIIB^Gauss of an output of digital silence (relative)', 1,
           abs(ours[1] - here) / here, 1e-9)
    x, y = ref + 0.5, deg + 0.5
    ours = (kudio.siib(x, y, RATE), kudio.siib_gauss(x, y, RATE))
    st = stages(x, y, RATE, gauss=False)
    here = (siib_of(information(st['cx'], st['cy'])),
            siib_gauss_of(squared_correlations(*_channels(x, y, gauss=True))))
    print(f"  {'an offset of 0.5 on both':<36} SIIB {ours[0]:.6f}  here {here[0]:.6f}   "
          f"SIIB^Gauss {ours[1]:.9f}  here {here[1]:.9f}")
    report('SIIB with an offset of 0.5 on both (relative)', 1,
           abs(ours[0] - here[0]) / here[0], 1e-4)
    report('SIIB^Gauss with an offset of 0.5 on both (relative)', 1,
           abs(ours[1] - here[1]) / here[1], 1e-10)


def _channels(x, y, gauss):
    st = stages(x, y, RATE, gauss=gauss)
    return st['cx'], st['cy']


def by_pysiib(ref, deg, sr):
    """pySIIB's SIIB and SIIB^Gauss of a pair, through its SIIB() alone.
    Its default window is 'hanning', a name the scipy I ran it with, 1.15.2,
    no longer knows; 'hann' is the window the name meant."""
    with warnings.catch_warnings():
        # it warns under 20 s of speech, which the default pairs are
        warnings.simplefilter('ignore')
        return (float(pysiib_siib(ref, deg, sr, window='hann')),
                float(pysiib_siib(ref, deg, sr, gauss=True, window='hann')))


#: SIIB's tolerance against pySIIB: other draws of the noise kudio adds to
#: break ties move SIIB by up to 1.2e-5 (the module notes), and I read
#: nothing into a difference under 1e-4 of the score
PYSIIB_SIIB = 1e-4

#: SIIB^Gauss's: no random numbers enter it, and what is left is rounding,
#: up to 8e-13 of the score on the pair low-passed at 2 kHz, whose top bands
#: hold little but the floor
PYSIIB_GAUSS = 1e-11


def against_pysiib(every):
    """kudio against pySIIB on the same pairs: checked at 16 kHz, and printed
    for information at other rates, where the two resample differently."""
    print('\nkudio against pySIIB (SIIB / SIIB^Gauss):')
    if pysiib_siib is None:
        print('  pySIIB is not installed, so these rows are skipped; '
              'pip install git+https://github.com/kamo-naoyuki/pySIIB.git, which '
              'compiles C code')
        return
    worst, cases = [0.0, 0.0], 0
    for name, (ref, deg, sr) in every.items():
        ours = (kudio.siib(ref, deg, sr), kudio.siib_gauss(ref, deg, sr))
        theirs = by_pysiib(ref, deg, sr)
        rel = [abs(a - b) / b for a, b in zip(ours, theirs)]
        if sr == RATE:
            cases += 1
            worst = [max(w, r) for w, r in zip(worst, rel)]
        print(f"  {name:<34} kudio {ours[0]:10.6f} / {ours[1]:13.9f}   pySIIB "
              f"{theirs[0]:10.6f} / {theirs[1]:13.9f}   relative {rel[0]:.1e} / "
              f"{rel[1]:.1e}" + ('' if sr == RATE else '   (resampled otherwise; not checked)'))
    report('SIIB against pySIIB at 16 kHz (relative)', cases, worst[0], PYSIIB_SIIB)
    report('SIIB^Gauss against pySIIB at 16 kHz (relative)', cases, worst[1], PYSIIB_GAUSS)

    ref, deg, _ = next(iter(every.values()))
    print('  on the first pair:')
    for label, x, y in (('a copy', ref, ref), ('an inverted copy', ref, -ref),
                        ('an offset of 0.5 on both', ref + 0.5, deg + 0.5),
                        ('an output of digital silence', ref, np.zeros_like(ref))):
        ours = (quietly(kudio.siib, x, y, RATE), quietly(kudio.siib_gauss, x, y, RATE))
        theirs = by_pysiib(x, y, RATE)
        rel = [abs(a - b) / b for a, b in zip(ours, theirs)]
        print(f"    {label:<32} kudio {ours[0]:10.6f} / {ours[1]:13.9f}   pySIIB "
              f"{theirs[0]:10.6f} / {theirs[1]:13.9f}   relative {rel[0]:.1e} / {rel[1]:.1e}")
        if label == 'an output of digital silence':
            # kudio and the second way agree there (checked above) and pySIIB
            # parts from both on SIIB^Gauss; SIIB there is mostly the noise
            # that breaks ties: printed, not checked
            continue
        report(f'SIIB against pySIIB, {label} (relative)', 1, rel[0], PYSIIB_SIIB)
        report(f'SIIB^Gauss against pySIIB, {label} (relative)', 1, rel[1], PYSIIB_GAUSS)


#: what the module notes say a sharper resampler moves the scores by, in
#: bit/s: (SIIB, SIIB^Gauss) for the test speech made at 8 and 20 kHz by a
#: near-ideal filter, and made at 8 kHz by resample_poly's own
SPREAD_SAID = {'near-ideal': (5.1, 1.6), "resample_poly's own": (1.9, 0.9)}


def resampler_spread():
    """How far SIIB and SIIB^Gauss move when the pair is taken to 16 kHz by
    a sharper filter than SIIB.m's: a sinc under a Kaiser window for 60 dB,
    365 taps from 20 kHz and 147 from 8 kHz."""
    talk = speech(36.0, 1)                      # tests/test_siib.py's test speech
    beta = 0.1102 * (60 - 8.7)
    near = {up: firwin(601, 1 / up, window=('kaiser', 10.0)) for up in (5, 2)}
    sharper = {20000: firwin(365, 1 / 5, window=('kaiser', beta)),
               8000: firwin(147, 1 / 2, window=('kaiser', beta))}
    made = {
        ('near-ideal', 20000): resample_poly(talk, 5, 4, window=near[5]),
        ('near-ideal', 8000): resample_poly(talk, 1, 2, window=near[2]),
        ("resample_poly's own", 8000): resample_poly(talk, 1, 2),
    }
    print("\nThe test speech through SIIB.m's filter and through a sharper one, "
          "white noise\nadded at the rate it was made at (SIIB / SIIB^Gauss, bit/s)"
          + (", and what pySIIB gives:" if pysiib_siib else ":"))
    worst = {build: [0.0, 0.0] for build in SPREAD_SAID}
    # pySIIB against kudio and against the sharper filter, for information
    apart = {'kudio': [0.0, 0.0], 'the sharper filter': [0.0, 0.0]}
    for (build, sr), ref in made.items():
        g = math.gcd(RATE, sr)
        up, down = RATE // g, sr // g
        for snr in range(-10, 21, 5):
            deg = mix(ref, white(len(ref), 7), snr)
            ours = (kudio.siib(ref, deg, sr), kudio.siib_gauss(ref, deg, sr))
            x = resample_poly(ref, up, down, window=sharper[sr])
            y = resample_poly(deg, up, down, window=sharper[sr])
            other = (kudio.siib(x, y, RATE), kudio.siib_gauss(x, y, RATE))
            for i in (0, 1):
                worst[build][i] = max(worst[build][i], abs(other[i] - ours[i]))
            line = (f"  {sr // 1000:2d} kHz, made by {build:<19} {snr:+3d} dB   "
                    f"{ours[0]:7.2f} -> {other[0]:7.2f}   {ours[1]:7.2f} -> {other[1]:7.2f}")
            if pysiib_siib is not None:
                theirs = by_pysiib(ref, deg, sr)
                for i in (0, 1):
                    apart['kudio'][i] = max(apart['kudio'][i], abs(theirs[i] - ours[i]))
                    apart['the sharper filter'][i] = max(apart['the sharper filter'][i],
                                                         abs(theirs[i] - other[i]))
                line += f"   pySIIB {theirs[0]:7.2f}   {theirs[1]:7.2f}"
            print(line)
    for build, (siib_said, gauss_said) in SPREAD_SAID.items():
        print(f"  largest, made by {build}: SIIB {worst[build][0]:.4f}, "
              f"SIIB^Gauss {worst[build][1]:.4f}")
    if pysiib_siib is not None:
        for against, (s, gs) in apart.items():
            print(f"  pySIIB against {against}: SIIB up to {s:.2e}, SIIB^Gauss up to "
                  f"{gs:.2e} bit/s (not checked)")
    for build, (siib_said, gauss_said) in SPREAD_SAID.items():
        report(f"SIIB's spread through a sharper resampler, made by {build}: "
               f"said {siib_said}", 1, abs(worst[build][0] - siib_said), 0.05)
        report(f"SIIB^Gauss's, made by {build}: said {gauss_said}", 1,
               abs(worst[build][1] - gauss_said), 0.05)


#: each check's tolerance, absolute unless the name says relative
TOLERANCES = {
    'the resampler written out, against resample_poly (relative)': 1e-12,
    'band centres, widths and weights (relative)': 1e-12,
    'window and masking weights': 1e-15,
    "each frame's level, in dB": 1e-9,
    'voice-activity decisions that differ': 0.0,
    'band powers (relative)': 1e-12,
    'masked log-spectra and stacks on the same input': 1e-12,
    'eigenvalues (relative)': 1e-10,
    "eigenchannels, against each channel's spread": 1e-6,
    'squared correlation of each channel': 1e-9,
    'SIIB^Gauss (relative)': 1e-10,
    'the neighbour search on the same channels, in nats': 1e-12,
    "each channel's information, no noise added, in bits": 1e-3,
    'SIIB, against no noise added (relative)': 1e-4,
    "SIIB, against SIIB.m's chain on other draws (relative)": 1e-4,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--all', action='store_true',
                        help='ten pairs of 30 s instead of four of 12 s; about eleven minutes')
    args = parser.parse_args()
    start = time.perf_counter()
    import scipy
    print(f"kudio {kudio.__version__} from {Path(kudio.__file__).parent}; numpy "
          f"{np.__version__}, scipy {scipy.__version__}, Python {sys.version.split()[0]}")
    print('\nkudio against the second way ("here": no noise added / SIIB.m\'s chain):')
    every = pairs(args.all)
    for name, (ref, deg, sr) in every.items():
        compare(name, ref, deg, sr)
    for check, tolerance in TOLERANCES.items():
        if check in _WORST:
            report(check, *_WORST[check], tolerance)
    ref, deg, _ = next(iter(every.values()))
    edges(ref, deg)
    against_pysiib(every)
    resampler_spread()
    failed = print_table()
    print(f"\n{time.perf_counter() - start:.0f} s; "
          + ("SOME DIFFERENCES ARE PAST TOLERANCE" if failed
             else "every difference within its tolerance"))
    return 1 if failed else 0


def print_table():
    """The table of every check; True if one is past its limit."""
    width = max(len(r[0]) for r in RESULTS)
    print(f"\n{'check':<{width}} {'cases':>5} {'max |diff|':>11} {'limit':>7}")
    failed = False
    for check, cases, diff, tolerance in RESULTS:
        over = diff > tolerance
        failed |= over
        limit = f'{tolerance:.1e}'.replace('.0e', 'e') if tolerance else '0'
        print(f"{check:<{width}} {cases:>5} {diff:11.2e} {limit:>7}{'  PAST' if over else ''}")
    return failed


if __name__ == '__main__':
    sys.exit(main())
