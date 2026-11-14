# -*- coding: utf-8 -*-
"""Check kudio's WADA-SNR (kudio/core/wada.py) against Kim and Stern's model
computed here another way, against a Monte Carlo run, and against the Monte
Carlo table that circulates with the method, showing where that table dips.

Run it from anywhere; it imports the kudio of the checkout it sits in::

    python tools/validation/wada.py
    python tools/validation/wada.py --table Alpha0.400000.txt

It needs numpy and scipy and nothing else; I ran it with numpy 2.0.2 and
scipy 1.13.1 on Python 3.9.13, and with numpy 2.2.6 and scipy 1.15.2 on
Python 3.10.22.

The model (Kim and Stern 2008, Eqs. 1, 6, 8 and 17): clean speech, pauses
included, is two-sided gamma of shape 0.4, the noise is Gaussian, and the
SNR alone sets G = ln E|z| - E ln|z| of their sum. kudio computes both
expectations through the characteristic functions of the two parts. Here
they are computed another way: E|a + v| is the mean of a folded normal, and
E ln|a + v| that of half the log of a noncentral chi-square of one degree of
freedom, a Poisson mixture of digamma values (and for a far above the noise,
the series ln a - s^2 / 2a^2 - 3 s^4 / 4a^4 - ...); the gamma expectation
over a is a Gauss-Legendre sum after a = w^5, which turns the density's
a^-0.6 into the smooth 5 w / Gamma(0.4), on panels that halve towards
w = 0 so that a noise at 100 dB down is resolved as well as one at 0 dB.

The table is Alpha0.400000.txt, which snreval's wada_snr.m reads (Dan
Ellis, LabROSA, SNREVAL v0.54 of 2014-07-01, labrosa.ee.columbia.edu/
projects/snreval/; the file in its zip is dated 2009-05-27); a comment in
wada_snr.m says the table came from Kim's C++ code. Its 121 values of G,
one for each whole dB from -20 to 100, are written out below; with
--table FILE the script checks them against a copy of the file.

What it checks:

1. The two ends of the model in closed form: Gaussian noise alone,
   ln sqrt(2/pi) + (euler_gamma + ln 2) / 2, and gamma speech alone,
   ln(0.4) - digamma(0.4); the second route there as well.
2. kudio's curve (kudio.core.wada._g) against the second route at every
   whole dB from -20 to 110, and at -200 and 300 dB.
3. The curve against a Monte Carlo run of 4e7 samples at six SNRs, each
   within four standard errors.
4. The table against the model: where it falls instead of rising (the dip
   at -17 dB), how far each value is from the model's G, and where each
   value reads on the model's curve; then what snreval's wada_snr.m,
   restated here, reads through the dip, and on clips the module docstring
   says the ports misread.
5. The numbers tests/test_wada.py pins: five table entries read within
   0.2 dB, the 100 dB entry past the top of the curve, gamma speech in
   Gaussian noise read back within 0.2 dB (0.5 at -5 dB), and Gaussian
   noise alone scored nan in 12 of 20 clips.
6. Figures the module docstring quotes: the ends of the curve, how far it
   rises above noise alone at -15 and -10 dB, how much G scatters over 10 s
   of Gaussian noise, worked out to first order from the moments of the
   half-normal and seen again over 40 clips, and what those 40 clips of
   noise alone score, the tests' 20 among them.

The table's entries from -20 to -15 dB are where its Monte Carlo noise
shows most: the model's G rises by 0.00034 over those five decibels, and
the table's -20 dB entry alone sits 0.00031 above it, so read on the
model's curve the first three entries land 4 to 4.8 dB high.

It prints the largest differences and exits with 1 if one is past its
tolerance. The run takes about 15 seconds, most of it in the Monte Carlo
run.

Reference: C. Kim and R. M. Stern, "Robust signal-to-noise ratio estimation
based on waveform amplitude distribution analysis", Proc. Interspeech 2008,
pp. 2598-2601, doi:10.21437/Interspeech.2008-644.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy import special
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import kudio  # noqa: E402

# the module, for its curve: kudio.core may export wada_snr() under a name
# that hides the module
module = importlib.import_module('kudio.core.wada')
wada_snr = module.wada_snr

SHAPE = 0.4

#: G from Alpha0.400000.txt, at -20, -19, ..., 100 dB
TABLE = (
    0.409747739, 0.409869263, 0.409985656, 0.409690892, 0.409861864, 0.409990055,
    0.410271377, 0.410526266, 0.411010238, 0.411432644, 0.412317178, 0.413372716,
    0.415264259, 0.417819198, 0.420772515, 0.424527992, 0.429188858, 0.435103734,
    0.442341951, 0.451614855, 0.462211529, 0.474916474, 0.488838093, 0.505092356,
    0.523537093, 0.543720882, 0.565324274, 0.588475317, 0.613462118, 0.639544959,
    0.667508177, 0.695837243, 0.724547622, 0.754147993, 0.783231484, 0.81240985,
    0.842197752, 0.871664058, 0.900305039, 0.928804177, 0.95655449, 0.983534905,
    1.010471548, 1.0362095, 1.061364248, 1.085793118, 1.109481904, 1.132779949,
    1.154728256, 1.176273084, 1.197035028, 1.216716938, 1.235358982, 1.253643127,
    1.271038908, 1.287180295, 1.303028647, 1.318395272, 1.332948173, 1.347009353,
    1.360572696, 1.373455135, 1.385771224, 1.397335037, 1.408563968, 1.41959619,
    1.42983624, 1.439584667, 1.449021764, 1.458048307, 1.466695685, 1.474869384,
    1.48269965, 1.490343394, 1.49748214, 1.504351061, 1.510764265, 1.516989146,
    1.522909703, 1.528578001, 1.533898351, 1.539121095, 1.543906502, 1.54858517,
    1.553107762, 1.557443906, 1.561649273, 1.565663481, 1.569386712, 1.573077668,
    1.576547638, 1.57980083, 1.583041292, 1.586024961, 1.588806813, 1.591624771,
    1.594196895, 1.596931549, 1.599446005, 1.601850111, 1.604086681, 1.60627134,
    1.608261987, 1.610045475, 1.611924722, 1.61369656, 1.615340743, 1.616889049,
    1.618389159, 1.619853744, 1.621358779, 1.622681189, 1.623904229, 1.625131432,
    1.626324628, 1.6274027, 1.628427675, 1.629455321, 1.6303307, 1.631280263,
    1.632041021,
)
TABLE_DB = np.arange(-20, 101)

#: (check, cases, largest absolute difference, tolerance)
RESULTS = []


def report(check, cases, diff, tolerance):
    RESULTS.append((check, cases, diff, tolerance))


def quietly(fn, *args):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return fn(*args)


# --- the model, the second route ---------------------------------------------------

def _gauss_panels():
    """Nodes and weights over w in [0, 4], on panels [4 / 2^(k+1), 4 / 2^k]
    for k = 0..39 and [0, 4 / 2^40], 24 Gauss-Legendre points each. w = 4
    is a = 1024, where the gamma density is e^-1024 of its scale."""
    x, wt = np.polynomial.legendre.leggauss(24)
    edges = [4.0 / 2.0 ** k for k in range(41)] + [0.0]
    nodes, weights = [], []
    for high, low in zip(edges[:-1], edges[1:]):
        nodes.append(low + (high - low) * (x + 1.0) / 2.0)
        weights.append(wt * (high - low) / 2.0)
    return np.concatenate(nodes), np.concatenate(weights)


NODES, WEIGHTS = _gauss_panels()


def mean_log_abs(a, s):
    """E ln|a + v| for v ~ N(0, s^2), a >= 0: half of E ln((a + v)^2), whose
    (a + v)^2 / s^2 is noncentral chi-square with one degree of freedom and
    noncentrality a^2 / s^2, so ln s + (ln 2 + sum_j P(j; lam) psi(1/2 + j))
    / 2 with lam = a^2 / 2s^2. Far above the noise, the asymptotic series
    ln a - sum_m (2m - 1)!! x^m / 2m, x = s^2 / a^2, to six terms."""
    lam = a * a / (2.0 * s * s)
    out = np.empty_like(a)
    near = lam <= 300.0
    if near.any():
        j = np.arange(int(300 + 12 * math.sqrt(300) + 60))
        lam_n = lam[near][:, None]
        with np.errstate(divide='ignore'):
            log_pmf = j * np.log(lam_n) - lam_n - special.gammaln(j + 1.0)
        log_pmf[:, 0] = -lam_n[:, 0]                 # 0 * log 0
        mix = np.exp(log_pmf) @ special.digamma(0.5 + j)
        out[near] = math.log(s) + 0.5 * (math.log(2.0) + mix)
    far = ~near
    if far.any():
        x = s * s / (a[far] * a[far])
        series = np.zeros_like(x)
        for m in range(1, 7):
            double_factorial = float(np.prod(np.arange(1, 2 * m, 2)))
            series += double_factorial * x ** m / (2 * m)
        out[far] = np.log(a[far]) - series
    return out


def folded_mean(a, s):
    """E|a + v| for v ~ N(0, s^2)."""
    return (s * math.sqrt(2.0 / math.pi) * np.exp(-a * a / (2.0 * s * s))
            + a * special.erf(a / (s * math.sqrt(2.0))))


def model_g(snr_db):
    """G of gamma speech (shape 0.4, rate 1) in Gaussian noise at *snr_db*.
    The SNR is shape (shape + 1) / sigma^2 (Eq. 6 with the rate at 1); the
    expectations over the speech run over a = w^5, where the density times
    da is 5 w e^(-w^5) dw / Gamma(0.4)."""
    s = math.sqrt(SHAPE * (SHAPE + 1.0) / 10.0 ** (snr_db / 10.0))
    a = NODES ** 5
    weight = WEIGHTS * 5.0 * NODES * np.exp(-a) / math.gamma(SHAPE)
    return (math.log(float(np.sum(weight * folded_mean(a, s))))
            - float(np.sum(weight * mean_log_abs(a, s))))


NOISE_ALONE = 0.5 * math.log(2.0 / math.pi) + 0.5 * (np.euler_gamma + math.log(2.0))
SPEECH_ALONE = math.log(SHAPE) - float(special.digamma(SHAPE))


def read_on(g, curve, low=-60.0, high=200.0):
    """The SNR at which *curve* gives *g*, on -60 to 200 dB."""
    return brentq(lambda d: curve(d) - g, low, high, xtol=1e-10)


# --- 1 to 3: the curve -----------------------------------------------------------

def check_curve():
    weight_total = float(np.sum(WEIGHTS * 5.0 * NODES * np.exp(-NODES ** 5))) / math.gamma(SHAPE)
    report('the quadrature over the gamma density sums to 1', 1, abs(weight_total - 1.0), 1e-13)
    report('second route at -200 dB vs Gaussian noise alone (closed form)', 1,
           abs(model_g(-200.0) - NOISE_ALONE), 1e-12)
    report('kudio at -200 dB vs Gaussian noise alone (closed form)', 1,
           abs(module._g(-200.0) - NOISE_ALONE), 1e-10)
    # at +300 dB the noise still reaches the samples nearest zero
    report('second route vs kudio at +300 dB', 1, abs(model_g(300.0) - module._g(300.0)), 1e-10)
    print(f"\nG of Gaussian noise alone {NOISE_ALONE:.6f}, of gamma speech alone "
          f"{SPEECH_ALONE:.6f}; the model at +300 dB is {SPEECH_ALONE - model_g(300.0):.1e} "
          f"short of the latter")
    grid = np.arange(-20, 111)
    ours = np.array([module._g(float(d)) for d in grid])
    theirs = np.array([model_g(float(d)) for d in grid])
    worst = int(np.argmax(np.abs(ours - theirs)))
    report(f'kudio vs the second route, -20 to 110 dB (worst at {grid[worst]} dB)', len(grid),
           float(np.max(np.abs(ours - theirs))), 1e-9)
    return dict(zip(grid.tolist(), theirs.tolist()))


def check_monte_carlo():
    rng = np.random.default_rng(2008)
    print('\nMonte Carlo, 4e7 samples at each SNR in 10 batches:')
    worst = 0.0
    for snr_db in (-10.0, 0.0, 5.0, 10.0, 20.0, 40.0):
        s = math.sqrt(SHAPE * (SHAPE + 1.0) / 10.0 ** (snr_db / 10.0))
        batches = []
        for _ in range(10):
            speech = rng.gamma(SHAPE, 1.0, 4_000_000) * rng.choice([-1.0, 1.0], 4_000_000)
            z = np.abs(speech + s * rng.standard_normal(4_000_000))
            batches.append((float(np.mean(z)), float(np.mean(np.log(z)))))
        means = np.array(batches)
        g_all = math.log(means[:, 0].mean()) - means[:, 1].mean()
        g_each = np.log(means[:, 0]) - means[:, 1]
        se = float(np.std(g_each, ddof=1) / math.sqrt(len(g_each)))
        z_score = (g_all - module._g(snr_db)) / se
        worst = max(worst, abs(z_score))
        print(f"  {snr_db:+5.0f} dB  Monte Carlo {g_all:.6f} +/- {se:.1e}, kudio "
              f"{module._g(snr_db):.6f}, {z_score:+.2f} standard errors")
    report('kudio vs Monte Carlo, in standard errors', 6, worst, 4.0)


# --- 4: the table -------------------------------------------------------------------

def check_table(path, model):
    if path is not None:
        data = Path(path).read_bytes()
        rows = [line.split() for line in data.decode('ascii').splitlines() if line.strip()]
        listed = [(float(r[0]), float(r[3])) for r in rows]
        apart = max(abs(d - want_d) + abs(g - want_g)
                    for (d, g), want_d, want_g in zip(listed, TABLE_DB, TABLE))
        apart = apart if len(listed) == len(TABLE) else math.inf
        print(f"\n{path}: sha256 {hashlib.sha256(data).hexdigest()[:16]}..., {len(listed)} rows")
        report('the table written out here vs the file given', len(listed), apart, 0.0)
    else:
        report('the table written out here vs a file: skipped, no --table', 0, 0.0, math.inf)

    table = np.array(TABLE)
    falls = np.flatnonzero(np.diff(table) <= 0) + 1
    print('\nWhere the table falls instead of rising: '
          + ', '.join(f"{TABLE_DB[i]} dB ({table[i - 1]:.6f} -> {table[i]:.6f})" for i in falls))
    report('dB values where the table falls (said: -17)', 1,
           float(np.max(np.abs(TABLE_DB[falls] + 17))) if len(falls) == 1 else math.inf, 0.0)

    gap = table - np.array([model[int(d)] for d in TABLE_DB])
    worst = int(np.argmax(np.abs(gap)))
    print(f"Table minus the model's G: largest {gap[worst]:+.4f} at {TABLE_DB[worst]} dB; "
          f"{gap[TABLE_DB.tolist().index(-20)]:+.5f} at -20 dB, where the model rises "
          f"{model[-15] - model[-20]:.5f} up to -15 dB")
    report(f"table above the model at its farthest: said 0.0053 at 97 dB, got "
           f"{gap[worst]:.4f} at {TABLE_DB[worst]} dB", 1,
           abs(gap[worst] - 0.0053) + abs(TABLE_DB[worst] - 97), 0.00005)

    read = np.array([read_on(g, module._g) for g in table]) - TABLE_DB
    print('Each entry read on kudio\'s curve, minus where the table puts it, in dB:')
    for start in range(0, 121, 10):
        part = slice(start, min(start + 10, 121))
        print(f"  {TABLE_DB[part][0]:+4d} to {TABLE_DB[part][-1]:+4d}: "
              + ' '.join(f'{v:+6.2f}' for v in read[part]))
    middle = (TABLE_DB >= -10) & (TABLE_DB <= 30)
    at = {int(d): float(r) for d, r in zip(TABLE_DB, read)}
    print(f"  largest from -10 to 30 dB {np.max(np.abs(read[middle])):.3f}; at 50 dB "
          f"{at[50]:+.3f}; at 100 dB {at[100]:+.3f} (said: at most 0.25, 0.47 and 7.3)")
    report('entries -10 to 30 dB read on the curve: said at most 0.25 dB', 41,
           max(0.0, np.max(np.abs(read[middle])) - 0.25), 0.005)
    report(f'the 50 dB entry: said 0.47 dB, got {abs(at[50]):.3f}', 1, abs(abs(at[50]) - 0.47),
           0.005)
    report(f'the 100 dB entry: said 7.3 dB, got {abs(at[100]):.2f}', 1,
           abs(abs(at[100]) - 7.3), 0.05)
    below = TABLE_DB[(TABLE_DB <= -10) & (read + TABLE_DB < -20)]
    print(f"  entries that read below -20 dB, the bottom of the table itself: "
          f"{', '.join(f'{d}' for d in below) or 'none'}")
    return table


def snreval_reading(y, table):
    """snreval's wada_snr.m, restated: |y| floored at 1e-10 with no mean
    taken out, G from it, and the SNR read off the table by straight lines
    from the last entry below G, the ends of the table where G is past
    them."""
    magnitude = np.abs(y)
    magnitude[magnitude < 1e-10] = 1e-10
    g = math.log(float(np.mean(magnitude))) - float(np.mean(np.log(magnitude)))
    return table_reading(g, table)


def table_reading(g, table):
    below = np.flatnonzero(table < g)
    if below.size == 0:
        return float(TABLE_DB[0])
    k = int(below[-1])
    if k == len(table) - 1:
        return float(TABLE_DB[-1])
    return float(TABLE_DB[k] + (g - table[k]) / (table[k + 1] - table[k]))


def check_ports(table):
    print("\nThrough the dip, snreval's reading of the table against kudio's curve:")
    print(f"  {'G':>9} {'snreval':>8} {'kudio':>8}")
    for g in np.linspace(0.40966, 0.41004, 9):
        port = table_reading(g, table)
        ours = quietly(module._snr_of, float(g))
        print(f"  {g:9.6f} {port:8.2f} {ours:8.2f}")

    rng = np.random.default_rng(6)
    sr = 16000
    clean = rng.gamma(SHAPE, 1.0, 9 * sr) * rng.choice([-1.0, 1.0], 9 * sr)
    hiss = rng.standard_normal(9 * sr)
    zero_db = clean + hiss * math.sqrt(np.mean(clean ** 2) / np.mean(hiss ** 2))
    padded = np.concatenate([np.zeros(sr // 2), zero_db, np.zeros(sr // 2)])
    hum = np.sin(2 * np.pi * 50 * np.arange(5 * sr) / sr + 0.3)
    clicks = 1e-4 * np.random.default_rng(7).standard_normal(5 * sr)
    clicks[::2000] = 1.0
    clicks[1000::2000] = -1.0
    print('\nWhat snreval reports and what kudio reports:')
    for name, y in (('gamma speech at 0 dB', zero_db),
                    ('the same with 10% zero padding', padded),
                    ('a 50 Hz hum', hum), ('clicks over a floor 80 dB down', clicks)):
        print(f"  {name:<32} snreval {snreval_reading(y.copy(), table):7.2f} dB, "
              f"kudio {quietly(wada_snr, y, sr):7.2f} dB")


# --- 5 and 6: the tests' numbers and the docstring's -----------------------------

def check_pinned(table):
    for snr_db in (-10, 0, 10, 20, 30):
        g = table[snr_db + 20]
        read = quietly(module._snr_of, float(g))
        report(f'table entry at {snr_db:+d} dB read within 0.2 dB (got {read - snr_db:+.3f})',
               1, max(0.0, abs(read - snr_db) - 0.2), 0.0)
    top = module._curve()[1][-1]
    report(f"the 100 dB entry {table[-1]:.6f} past the curve's top {top:.4f}", 1,
           max(0.0, top - table[-1]), 0.0)

    worst = 0.0
    for snr_db in (-5.0, 0.0, 10.0, 20.0, 30.0):
        estimates = []
        for seed in range(5):
            rng = np.random.default_rng(1000 + 10 * int(snr_db + 5) + seed)
            clean = rng.gamma(SHAPE, 1.0, 10 * 16000) * rng.choice([-1.0, 1.0], 10 * 16000)
            hiss = rng.standard_normal(len(clean))
            noisy = clean + hiss * math.sqrt(np.mean(clean ** 2)
                                             / (np.mean(hiss ** 2) * 10 ** (snr_db / 10)))
            estimates.append(wada_snr(noisy, 16000))
        allowed = 0.5 if snr_db < 0 else 0.2
        worst = max(worst, abs(np.mean(estimates) - snr_db) - allowed)
    report('gamma speech read back, mean of 5 clips (0.2 dB; 0.5 at -5 dB)', 25,
           max(0.0, worst), 0.0)

    # the tests' 20 clips are the first half of the docstring's 40
    values = [quietly(wada_snr, np.random.default_rng(500 + seed).standard_normal(10 * 16000),
                      16000) for seed in range(40)]
    print()
    for clips, said_nan, said, places in ((20, 12, (-13.9, -10.3), 1), (40, 23, (-16, -10), 0)):
        read = [v for v in values[:clips] if not math.isnan(v)]
        lost = clips - len(read)
        ends = (round(min(read), places), round(max(read), places))
        print(f"Gaussian noise alone, {clips} clips of 10 s: {lost} nan, the rest read "
              f"{min(read):.2f} to {max(read):.2f} dB (said: {said_nan} nan, "
              f"{said[0]:g} to {said[1]:g})")
        report(f'Gaussian noise alone: {said_nan} of {clips} nan, the rest {said[0]:g} to '
               f'{said[1]:g} dB', clips,
               abs(lost - said_nan) + max(abs(e - s) for e, s in zip(ends, said)), 0)

    model_at = {d: module._g(float(d)) for d in (-20, -15, -10, 100)}
    print(f"\nThe curve at -20 dB is {model_at[-20] - NOISE_ALONE:.1e} above noise alone, "
          f"at 100 dB {SPEECH_ALONE - model_at[100]:.4f} below speech alone; at -15 dB "
          f"{model_at[-15] - NOISE_ALONE:.4f} and at -10 dB {model_at[-10] - NOISE_ALONE:.4f} "
          f"above noise alone")
    report('noise alone 0.4094 and speech alone 1.6451 (docstring)', 2,
           max(abs(NOISE_ALONE - 0.4094), abs(SPEECH_ALONE - 1.6451)), 0.00005)
    report('the curve at -20 dB within 5e-5 of noise alone', 1,
           max(0.0, model_at[-20] - NOISE_ALONE - 5e-5), 0.0)
    report('the curve at 100 dB within 0.019 of speech alone', 1,
           max(0.0, SPEECH_ALONE - model_at[100] - 0.019), 0.0)
    report('above noise alone: 0.0004 at -15 dB, 0.0028 at -10 dB', 2,
           max(abs(model_at[-15] - NOISE_ALONE - 0.0004),
               abs(model_at[-10] - NOISE_ALONE - 0.0028)), 0.00005)

    # G = ln m1 - m2 over n samples of |z|, z ~ N(0, 1): to first order its
    # variance is Var(|z| / E|z| - ln|z|) / n, from the half-normal's moments:
    # E|z| = sqrt(2 / pi), Var|z| = 1 - 2 / pi, Var ln|z| = pi^2 / 8,
    # E ln|z| = (digamma(1/2) + ln 2) / 2, E |z| ln|z| = sqrt(2 / pi) (ln 2 - gamma) / 2
    m1 = math.sqrt(2.0 / math.pi)
    mean_log = 0.5 * (float(special.digamma(0.5)) + math.log(2.0))
    covariance = m1 * 0.5 * (math.log(2.0) - np.euler_gamma) - m1 * mean_log
    per_sample = (1.0 - 2.0 / math.pi) / m1 ** 2 + math.pi ** 2 / 8.0 - 2.0 * covariance / m1
    sd = math.sqrt(per_sample / (10 * 16000))
    spread = []
    for seed in range(40):
        z = np.abs(np.random.default_rng(900 + seed).standard_normal(10 * 16000))
        spread.append(math.log(np.mean(z)) - np.mean(np.log(z)))
    print(f"G over 10 s of Gaussian noise at 16 kHz: SD {sd:.5f} to first order, "
          f"{np.std(spread, ddof=1):.5f} over 40 clips (said 0.0016)")
    report(f'how much G scatters over 10 s of noise: said 0.0016, got {sd:.5f}', 1,
           abs(sd - 0.0016), 0.00005)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--table', metavar='FILE',
                        help="a copy of snreval's Alpha0.400000.txt, to check the values "
                             "written out here against")
    args = parser.parse_args()
    start = time.perf_counter()
    import scipy
    print(f"kudio {kudio.__version__} from {Path(kudio.__file__).parent}; numpy "
          f"{np.__version__}, scipy {scipy.__version__}, Python {sys.version.split()[0]}")
    model = check_curve()
    check_monte_carlo()
    table = check_table(args.table, model)
    check_ports(table)
    check_pinned(table)
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
        limit = '-' if math.isinf(tolerance) else f'{tolerance:.1e}'.replace('.0e', 'e')
        print(f"{check:<{width}} {cases:>5} {diff:11.2e} {limit:>7}{'  PAST' if over else ''}")
    return failed


if __name__ == '__main__':
    sys.exit(main())
