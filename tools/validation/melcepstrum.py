# -*- coding: utf-8 -*-
"""Check kudio's MCD (kudio/core/melcepstrum.py) against URGENT 2025's,
restated on pysptk and fastdtw.

Run it from anywhere; it imports the kudio of the checkout it sits in::

    python tools/validation/melcepstrum.py

It needs pysptk (I ran 1.0.1, built from source; MIT, wrapping SPTK) and
fastdtw (0.3.4, its pure-Python version; MIT). URGENT 2025's
``evaluation_metrics/mcd_utils.py`` (Apache-2.0, commit c726558) and the
``mcd_metric`` of ``calculate_intrusive_se_metrics.py`` (commit 93e16aa)
are restated here as they read: the gain match, SPTK's power-normalized
Hamming window, ``pysptk.mcep(frame, order, alpha, eps=1e-6, etype=1)``,
``fastdtw(..., dist=scipy.spatial.distance.euclidean)`` and the mean of
``10 / ln 10 sqrt(2 sum d^2)``, with mcd_metric passing the reference where
calculate() expects the generated signal, as the script does.

Every signal is made here from fixed seeds: the synthetic speech of
tools/validation/quality.py, 3 s, resampled to 8, 16, 22.05, 24, 32, 44.1
and 48 kHz, under white noise at 5 dB, at half the level with a faint
floor, low-passed at a quarter of the rate, in a 0.4 s room, and with 1 s of
digital silence on either side of the noisy pair.

What it checks:

1. The mel-cepstrum of every frame against pysptk.mcep, as the largest
   difference in any coefficient.
2. The FastDTW path against fastdtw's, on the two sequences of
   mel-cepstra.
3. mcd against URGENT's mcd_metric, and with match_gain=False against its
   calculate().
4. How far URGENT's own number moves when the pair is read as float32, as
   its script reads files.

It prints the largest differences and exits with 1 if one is past its
tolerance. The run takes about two minutes.
"""
from __future__ import annotations

import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import kudio  # noqa: E402
from kudio.core import melcepstrum as module  # noqa: E402

try:
    import pysptk
    from fastdtw import fastdtw
    from scipy.spatial.distance import euclidean
except ImportError:
    pysptk = None

RESULTS = []


def report(check, cases, diff, tolerance):
    RESULTS.append((check, cases, diff, tolerance))


# --- URGENT 2025, restated ----------------------------------------------------------

def sptk_extract(x, fs, n_fft=1024, n_shift=256):
    mcep_dim, mcep_alpha = module.MCD_SETTINGS[fs]
    n_frame = (len(x) - n_fft) // n_shift + 1
    win = pysptk.sptk.hamming(n_fft)
    return np.stack([pysptk.mcep(x[n_shift * i: n_shift * i + n_fft] * win, mcep_dim,
                                 mcep_alpha, eps=1e-6, etype=1) for i in range(n_frame)])


def calculate(inf_audio, ref_audio, fs):
    gen_mcep = sptk_extract(inf_audio, fs)
    gt_mcep = sptk_extract(ref_audio, fs)
    _, path = fastdtw(gen_mcep, gt_mcep, dist=euclidean)
    twf = np.array(path).T
    diff2sum = np.sum((gen_mcep[twf[0]] - gt_mcep[twf[1]]) ** 2, 1)
    return np.mean(10.0 / np.log(10.0) * np.sqrt(2 * diff2sum), 0)


def mcd_metric(ref, inf, fs, eps=1.0e-08):
    scaling_factor = np.sum(ref * inf) / (np.sum(inf ** 2) + eps)
    return calculate(ref, inf * scaling_factor, fs)


# --- signals ----------------------------------------------------------------------

def pairs():
    from quality import mix, speech
    rng = np.random.default_rng(4)
    for sr in sorted(module.MCD_SETTINGS):
        base = speech(16000, 3.0, seed=1)
        clean = resample_poly(base, sr, 16000) if sr != 16000 else base
        n = clean.size
        yield sr, 'white 5 dB', clean, mix(clean, rng.standard_normal(n), 5)
        yield sr, 'half level', clean, 0.5 * clean + 1e-3 * rng.standard_normal(n)
        yield sr, 'low-passed', clean, sosfiltfilt(butter(8, sr / 4, fs=sr, output='sos'), clean)
        room = np.asarray(kudio.apply_rir(clean, kudio.rir(sr, rt60=0.4, seed=3)), float)[:n]
        yield sr, 'room 0.4 s', clean, room
        pad = np.zeros(sr)
        noisy = mix(clean, rng.standard_normal(n), 10)
        yield sr, 'silence either side', np.r_[pad, clean, pad], np.r_[pad, noisy, pad]


def main():
    start = time.perf_counter()
    if pysptk is None:
        print("pysptk and fastdtw are needed: there is nothing to compare with")
        return 1
    import scipy
    print(f"kudio {kudio.__version__} from {Path(kudio.__file__).parent}; pysptk "
          f"{pysptk.__version__}; numpy {np.__version__}, scipy {scipy.__version__}, "
          f"Python {sys.version.split()[0]}")
    worst = {'mcep': 0.0, 'path': 0, 'mcd': 0.0, 'calculate': 0.0, 'float32': 0.0}
    frames = cases = 0
    for sr, label, ref, deg in pairs():
        cases += 1
        order, alpha = module.MCD_SETTINGS[sr]
        for y in (ref, deg):
            ours = module._mel_cepstra(module._frames(y, 1024, 256), order, alpha)
            theirs = sptk_extract(y, sr)
            frames += len(theirs)
            worst['mcep'] = max(worst['mcep'], float(np.max(np.abs(ours - theirs))))
        a, b = sptk_extract(ref, sr), sptk_extract(deg, sr)
        _, path = fastdtw(a, b, dist=euclidean)
        worst['path'] += module._fastdtw(a, b) != [tuple(p) for p in path]
        got, want = kudio.mcd(ref, deg, sr), mcd_metric(ref, deg, sr)
        worst['mcd'] = max(worst['mcd'], abs(got - want))
        got = kudio.mcd(ref, deg, sr, match_gain=False)
        worst['calculate'] = max(worst['calculate'], abs(got - calculate(ref, deg, sr)))
        read = mcd_metric(ref.astype(np.float32), deg.astype(np.float32), sr)
        worst['float32'] = max(worst['float32'], abs(read - want))
        print(f"  {sr:5d} Hz  {label:20s} MCD {want:8.4f} dB")
    report('mel-cepstra vs pysptk.mcep, largest coefficient difference', frames,
           worst['mcep'], 1e-12)
    report('FastDTW paths that differ from fastdtw', cases, float(worst['path']), 0.0)
    report("mcd vs URGENT's mcd_metric (dB)", cases, worst['mcd'], 1e-10)
    report("mcd(match_gain=False) vs URGENT's calculate() (dB)", cases, worst['calculate'], 1e-10)
    report("  URGENT's own MCD with the pair read as float32, moved by (dB)", cases,
           worst['float32'], math.inf)
    width = max(len(r[0]) for r in RESULTS)
    print(f"\n{'check':<{width}} {'cases':>6} {'max |diff|':>11} {'limit':>8}")
    failed = False
    for check, n, diff, tolerance in RESULTS:
        over = diff > tolerance
        failed |= over
        limit = '-' if math.isinf(tolerance) else f"{tolerance:.0e}"
        print(f"{check:<{width}} {n:>6} {diff:>11.2e} {limit:>8}{'  PAST' if over else ''}")
    print(f"\n{time.perf_counter() - start:.0f} s; "
          + ("SOME DIFFERENCES ARE PAST TOLERANCE" if failed
             else "every difference within its tolerance"))
    return 1 if failed else 0


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)
        sys.exit(main())
