# -*- coding: utf-8 -*-
"""Check kudio's separation measures (kudio/core/separation.py): SI-SDR and
its permutation against torchmetrics and an exhaustive search, BSS Eval
against mir_eval, and the improvement over the mixture.

Run it from anywhere; it imports the kudio of the checkout it sits in::

    python tools/validation/separation.py

The oracles are optional and each row says which one it needs: torchmetrics
(I ran 1.9.0 on torch 2.14.1) for SI-SDR and its speaker-wise PIT, and
mir_eval (0.8.2) for BSS Eval. A row whose package is missing is skipped and
says so. torchmetrics is Apache-2.0 and mir_eval MIT; they run here beside
kudio as something to compare with.

Every signal is made here from fixed seeds: k talkers of harmonic bursts at
different pitches over their own noise floors, mixed into each estimate with
random leaks of 10% to 90% of the other talkers and noise, shuffled into a
random order; 2, 3 and 4 sources, 40 draws each.

What it checks:

1. si_sdr against torchmetrics' scale_invariant_signal_distortion_ratio with
   zero_mean=True, in float64, on every pairing of the draws; and how far
   torchmetrics' default, zero_mean=False, parts from it on the same pairs
   with a DC offset of 1% of the peak added to the estimate.
2. pit_si_sdr against torchmetrics' permutation_invariant_training with that
   metric, mode='speaker-wise' and eval_func='max': the mean SI-SDR and the
   permutation; and against every permutation tried in turn.
3. bss_eval against mir_eval.separation.bss_eval_sources, with its own
   permutation and with the one pit_si_sdr found, and the SDRi against the
   mixture passed as every estimate.
4. si_sdr_improvement and the SI-SDRi of pit_si_sdr against si_sdr of the
   estimate minus si_sdr of the mixture.

It prints the largest differences and exits with 1 if one is past its
tolerance, in dB where the figure is one. The run takes one to three
minutes, most of it in BSS Eval, which decomposes every pairing.
"""
from __future__ import annotations

import itertools
import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import kudio  # noqa: E402

try:
    import torch
    from torchmetrics.functional.audio import (permutation_invariant_training,
                                               scale_invariant_signal_distortion_ratio)
except ImportError:
    torch = None

try:
    import mir_eval
except ImportError:
    mir_eval = None

SR = 8000
DRAWS = 40

RESULTS = []


def report(check, cases, diff, rel, tolerance, on='abs'):
    RESULTS.append((check, cases, diff, rel, tolerance, on))


def skipped(what, why):
    report(f'{what}: skipped, {why}', 0, 0.0, 0.0, math.inf)


def talkers(k, n, seed):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / SR
    out = []
    for i in range(k):
        f0 = rng.uniform(90.0, 260.0)
        y = sum(np.sin(2 * np.pi * f0 * h * t + rng.uniform(0, 6)) / h for h in range(1, 9))
        y *= 0.5 + 0.5 * np.sin(2 * np.pi * rng.uniform(1.5, 5.0) * t + rng.uniform(0, 6))
        out.append(0.1 * y + 0.003 * rng.standard_normal(n))
    return np.array(out)


def draws():
    """``(refs, ests, mixture)`` for every draw."""
    for k in (2, 3, 4):
        for seed in range(DRAWS):
            rng = np.random.default_rng(1000 * k + seed)
            refs = talkers(k, SR, 1000 * k + seed)
            leak = rng.uniform(0.1, 0.9, (k, k))
            np.fill_diagonal(leak, 1.0)
            ests = leak @ refs + 0.05 * rng.standard_normal(refs.shape)
            yield refs, ests[rng.permutation(k)], refs.sum(axis=0)


def largest(got, want):
    got, want = np.ravel(got).astype(float), np.ravel(want).astype(float)
    diff = np.abs(got - want)
    rel = diff / np.maximum(np.abs(want), 1e-300)
    return float(np.max(diff)), float(np.max(rel))


def tm_si_sdr(ref, est, zero_mean=True):
    return float(scale_invariant_signal_distortion_ratio(
        torch.from_numpy(est), torch.from_numpy(ref), zero_mean=zero_mean))


def check_si_sdr():
    if torch is None:
        skipped('si_sdr vs torchmetrics', 'no torchmetrics')
        return
    got, want, plain, dc_want = [], [], [], []
    for refs, ests, _ in draws():
        for ref in refs:
            for est in ests:
                got.append(kudio.si_sdr(ref, est))
                want.append(tm_si_sdr(ref, est))
                offset = est + 0.01 * np.max(np.abs(est))
                plain.append(kudio.si_sdr(ref, offset))
                dc_want.append(tm_si_sdr(ref, offset, zero_mean=False))
    report('si_sdr vs torchmetrics, zero_mean=True (dB)', len(got), *largest(got, want), 1e-9)
    diff, rel = largest(plain, dc_want)
    report("  the same with 1% DC on the estimate vs zero_mean=False (they part)",
           len(plain), diff, rel, math.inf)


def check_pit():
    exhaustive_diff = 0.0
    same_order = 0
    cases = 0
    tm_diff = []
    tm_orders = 0
    for refs, ests, _ in draws():
        k = len(refs)
        score = kudio.pit_si_sdr(refs, ests)
        table = np.array([[kudio.si_sdr(r, e) for e in ests] for r in refs])
        best = max(itertools.permutations(range(k)),
                   key=lambda p: sum(table[i, p[i]] for i in range(k)))
        exhaustive_diff = max(exhaustive_diff, abs(
            np.mean(score.si_sdr) - np.mean([table[i, best[i]] for i in range(k)])))
        same_order += tuple(best) == score.permutation
        cases += 1
        if torch is not None:
            value, order = permutation_invariant_training(
                torch.from_numpy(ests)[None], torch.from_numpy(refs)[None],
                lambda p, t: scale_invariant_signal_distortion_ratio(p, t, zero_mean=True),
                mode='speaker-wise', eval_func='max')
            tm_diff.append((float(np.mean(score.si_sdr)), float(value[0])))
            tm_orders += tuple(int(i) for i in order[0]) == score.permutation
    report('pit_si_sdr mean vs every permutation tried', cases, exhaustive_diff,
           exhaustive_diff, 1e-12)
    report(f'  same permutation as the exhaustive search: {same_order} of {cases}', cases,
           float(cases - same_order), float(cases - same_order), 0.0)
    if torch is None:
        skipped('pit_si_sdr vs torchmetrics PIT', 'no torchmetrics')
        return
    got, want = zip(*tm_diff)
    report('pit_si_sdr mean vs torchmetrics PIT, speaker-wise (dB)', len(got),
           *largest(got, want), 1e-9)
    report(f'  same permutation as torchmetrics: {tm_orders} of {cases}', cases,
           float(cases - tm_orders), float(cases - tm_orders), 0.0)


def check_bss_eval():
    if mir_eval is None:
        skipped('bss_eval vs mir_eval', 'no mir_eval')
        return
    worst = [0.0, 0.0, 0.0]
    orders = 0
    cases = 0
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)
        for n, (refs, ests, mixture) in enumerate(draws()):
            if n % 4:                  # BSS Eval is slow; every fourth draw
                continue
            cases += 1
            got = kudio.bss_eval(refs, ests, mixture)
            sdr, sir, sar, perm = mir_eval.separation.bss_eval_sources(refs, ests)
            for i, (a, b) in enumerate(((got.sdr, sdr), (got.sir, sir), (got.sar, sar))):
                worst[i] = max(worst[i], largest(a, b)[0])
            orders += got.permutation == tuple(int(i) for i in perm)
            k = len(refs)
            of_mixture = mir_eval.separation.bss_eval_sources(
                refs, np.tile(mixture, (k, 1)), compute_permutation=False)[0]
            worst[0] = max(worst[0], largest(got.sdri, np.asarray(sdr) - of_mixture)[0])
            order = kudio.pit_si_sdr(refs, ests).permutation
            fixed = kudio.bss_eval(refs, ests, permutation=order)
            sdr2 = mir_eval.separation.bss_eval_sources(refs, ests[list(order)],
                                                        compute_permutation=False)[0]
            worst[0] = max(worst[0], largest(fixed.sdr, sdr2)[0])
    for name, value in zip(('SDR and SDRi', 'SIR', 'SAR'), worst):
        report(f'bss_eval {name} vs mir_eval', cases, value, value, 0.0)
    report(f'  same permutation as mir_eval: {orders} of {cases}', cases,
           float(cases - orders), float(cases - orders), 0.0)


def check_improvement():
    worst = 0.0
    cases = 0
    for refs, ests, mixture in draws():
        score = kudio.pit_si_sdr(refs, ests, mixture)
        for i, ref in enumerate(refs):
            est = ests[score.permutation[i]]
            want = kudio.si_sdr(ref, est) - kudio.si_sdr(ref, mixture)
            worst = max(worst, abs(score.si_sdri[i] - want),
                        abs(kudio.si_sdr_improvement(ref, est, mixture) - want))
            cases += 1
    report('SI-SDRi vs si_sdr(est) - si_sdr(mixture)', cases, worst, worst, 0.0)


def main():
    start = time.perf_counter()
    import scipy
    print(f"kudio {kudio.__version__} from {Path(kudio.__file__).parent}; torch "
          f"{getattr(torch, '__version__', 'none')}, mir_eval "
          f"{getattr(mir_eval, '__version__', 'none')}; numpy {np.__version__}, "
          f"scipy {scipy.__version__}, Python {sys.version.split()[0]}")
    check_si_sdr()
    check_pit()
    check_bss_eval()
    check_improvement()
    failed = print_table()
    print(f"\n{time.perf_counter() - start:.0f} s; "
          + ("SOME DIFFERENCES ARE PAST TOLERANCE" if failed
             else "every difference within its tolerance"))
    return 1 if failed else 0


def print_table():
    width = max(len(r[0]) for r in RESULTS)
    print(f"\n{'check':<{width}} {'cases':>5} {'max |diff|':>11} {'max rel':>9} "
          f"{'limit':>11}")
    failed = False
    for check, cases, diff, rel, tolerance, on in RESULTS:
        over = (rel if on == 'rel' else diff) > tolerance
        failed |= over
        limit = ('-' if math.isinf(tolerance)
                 else f"{tolerance:.1e}".replace('.0e', 'e') + f" {on}")
        print(f"{check:<{width}} {cases:>5} {diff:>11.2e} {rel:9.2e} {limit:>11}"
              f"{'  PAST' if over else ''}")
    return failed


if __name__ == '__main__':
    sys.exit(main())
