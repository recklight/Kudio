# -*- coding: utf-8 -*-
"""Check kudio.core.f0eval against mir_eval, and against a frame-by-frame
implementation written here.

Needs numpy, kudio (this checkout is found on its own) and mir_eval; I ran it
with mir_eval 0.8.2. From the top of the checkout::

    python tools/validation/f0eval.py

mir_eval scores melody extraction and defines its measures its own way: raw
pitch accuracy takes one tolerance in cents both ways and counts over the
reference's voiced frames, and overall accuracy counts every frame. So the
comparison is made where its definitions and kudio's meet:

* VDE is 1 - overall_accuracy with an unlimited cent tolerance
  (cent_tolerance=np.inf), and it also follows from voicing_recall and
  voicing_false_alarm.
* GPE is 1 - raw_pitch_accuracy over the frames voiced in both, with those
  frames split by the sign of their error and each half given the tolerance
  where 20% falls on its side, 315.6 cents above and 386.3 below.
* FFE is 1 - overall_accuracy at 350 cents on the pairs built with no error
  between 300 and 400 cents, where one tolerance draws the line the 20% rule
  draws. On the other pairs it is VDE plus the gross errors over all frames.
* The RMSE is checked against cents from mir_eval.melody.hz2cents.

FPE has no counterpart in mir_eval. It, and every other measure, is also
checked against frame_by_frame() below, which walks the frames one at a time
in plain Python: the four cases of Rabiner et al. (1976), Chu and Alwan's
ratio |est / ref - 1| > 0.2, and statistics.pstdev.

The pairs: the one tests/test_f0eval.py stores mir_eval's numbers for; 7 at
the ends of what can be scored, from a single frame to tracks never voiced
together; 200 random pairs with no error between 300 and 400 cents; and 200
with errors anywhere within two octaves, the random ones 1 to 3000 frames
long. The script prints the largest differences, the numbers the tests
store, and the figures the module docstring quotes that no test pins, and
exits with 1 when any difference is past its tolerance.
"""
from __future__ import annotations

import math
import statistics
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mir_eval  # noqa: E402

from kudio.core.f0eval import f0_errors  # noqa: E402

MEASURES = ('gpe', 'vde', 'ffe', 'fpe', 'rmse_cents')
COUNTS = ('frames', 'voiced_both', 'gross', 'voiced_to_unvoiced', 'unvoiced_to_voiced')

# 20% of the reference, in cents above it and below it
UP, DOWN = 1200.0 * math.log2(1.2), -1200.0 * math.log2(0.8)

# how far apart the two sides may be: the rates are counts over counts, and
# FPE and the RMSE go through a mean, a log and a square root
TOLERANCE = {'gpe': 1e-12, 'vde': 1e-12, 'ffe': 1e-12, 'fpe': 1e-12,
             'rmse_cents': 1e-9}


# --- the pairs ---------------------------------------------------------------

def voice(n: int = 1000) -> np.ndarray:
    """The reference track of tests/test_f0eval.py: syllables of 26 voiced
    frames with 11 unvoiced between them, on an F0 between 115 and 171 Hz."""
    i = np.arange(n)
    hz = 140.0 * 2.0 ** ((300.0 * np.sin(2 * np.pi * i / 611.0)
                          + 40.0 * np.sin(2 * np.pi * i / 37.0)) / 1200.0)
    hz[i % 37 >= 26] = 0.0
    return hz


def tracked(ref: np.ndarray) -> np.ndarray:
    """The estimate of tests/test_f0eval.py, each kind of error on frames
    known in advance and none between 42 and 420 cents."""
    i = np.arange(ref.size)
    cents = 30.0 * np.sin(2 * np.pi * i / 23.0) + 12.0 * np.cos(2 * np.pi * i / 7.0)
    est = ref * 2.0 ** (cents / 1200.0)
    est[i % 53 == 7] *= 2.0
    est[i % 71 == 30] *= 0.5
    est[i % 97 == 50] *= 2.0 ** (450.0 / 1200.0)
    est[i % 37 < 2] = 0.0
    tail = np.flatnonzero(i % 37 == 26)
    est[tail] = ref[tail - 1]
    return est


def edge_pairs():
    """The pairs at the ends of what can be scored, none with an error
    between 300 and 400 cents."""
    glide = np.linspace(100.0, 200.0, 200)
    return [
        ("one frame, voiced in both", np.array([150.0]), np.array([151.0])),
        ("one frame, an octave off", np.array([150.0]), np.array([300.0])),
        ("nothing voiced in est", voice(200), np.zeros(200)),
        ("nothing voiced in ref", np.zeros(200), voice(200)),
        ("never voiced together", voice(200), np.where(voice(200) > 0, 0.0, 120.0)),
        ("every frame right", glide, glide.copy()),
        ("every frame an octave off", glide, 2.0 * glide),
    ]


def random_pair(seed: int, banded: bool):
    """A reference between 70 and 396 Hz with runs of voiced and unvoiced
    frames, and an estimate with fine errors of 40 cents (one SD), voicing
    errors, and errors anywhere within two octaves on about one frame in
    eight. *banded* moves every error lying between 300 and 400 cents out of
    that band, by 100 cents away from the reference."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 3001))
    # voicing in runs: a two-state chain that changes state one frame in 12
    flips = rng.random(n) < 1.0 / 12.0
    ref_on = (np.cumsum(flips) + int(rng.integers(0, 2))) % 2 == 1
    ref = np.where(ref_on, 70.0 * 2.0 ** (2.5 * rng.random(n)), 0.0)
    cents = 40.0 * rng.standard_normal(n)
    wild = rng.random(n) < 0.125
    cents[wild] = 2400.0 * (2.0 * rng.random(int(wild.sum())) - 1.0)
    if banded:
        band = (np.abs(cents) > 300.0) & (np.abs(cents) < 400.0)
        cents[band] += 100.0 * np.sign(cents[band])
    base = np.where(ref_on, ref, 70.0 * 2.0 ** (2.5 * rng.random(n)))
    est = base * 2.0 ** (cents / 1200.0)
    miss = rng.random(n) < rng.uniform(0.0, 0.4)
    est[miss] = 0.0
    return ref, est


# --- the oracles -------------------------------------------------------------

def frame_by_frame(ref, est, threshold: float = 0.2) -> dict:
    """Every measure, one frame at a time: nan where it has nothing to count,
    as kudio's."""
    n = vu = uv = vv = gross = 0
    fine, cents = [], []
    for r, e in zip(np.asarray(ref, float).tolist(), np.asarray(est, float).tolist()):
        n += 1
        r_on, e_on = r > 0.0, e > 0.0
        if r_on and not e_on:
            vu += 1
        elif e_on and not r_on:
            uv += 1
        elif r_on and e_on:
            vv += 1
            cents.append(1200.0 * math.log2(e / r))
            if abs(e / r - 1.0) > threshold:
                gross += 1
            else:
                fine.append((e - r) / r)
    nan = float('nan')
    return {
        'gpe': gross / vv if vv else nan,
        'vde': (vu + uv) / n,
        'ffe': (vu + uv + gross) / n,
        'fpe': statistics.pstdev(fine) if len(fine) >= 2 else nan,
        'rmse_cents': (math.sqrt(math.fsum(c * c for c in cents) / len(cents))
                       if cents else nan),
        'frames': n, 'voiced_both': vv, 'gross': gross,
        'voiced_to_unvoiced': vu, 'unvoiced_to_voiced': uv,
    }


def mir_eval_scores(ref, est, banded: bool) -> dict:
    """GPE, VDE, FFE and the RMSE out of mir_eval's melody measures, keyed
    ``(measure, how)``; a measure is left out where mir_eval has nothing to
    say about it."""
    melody = mir_eval.melody
    ref_v, est_v = (ref > 0).astype(float), (est > 0).astype(float)
    ref_c, est_c = melody.hz2cents(ref), melody.hz2cents(est)
    both = (ref > 0) & (est > 0)
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")      # its notes on tracks with no voicing
        vde = 1.0 - melody.overall_accuracy(ref_v, ref_c, est_v, est_c,
                                            cent_tolerance=np.inf)
        out['vde', '1 - overall_accuracy, unlimited tolerance'] = vde
        recall, false_alarm = melody.voicing_measures(ref_v, est_v)
        voiced = ref_v.sum()
        out['vde', 'voicing_recall and voicing_false_alarm'] = (
            ((1.0 - recall) * voiced + false_alarm * (ref.size - voiced)) / ref.size)
        if banded:
            out['ffe', '1 - overall_accuracy, 350 cents'] = 1.0 - melody.overall_accuracy(
                ref_v, ref_c, est_v, est_c, cent_tolerance=350.0)
        if both.any():
            gross = 0
            for side, tolerance in ((est >= ref, UP), (est < ref, DOWN)):
                pick = both & side
                if pick.any():
                    ones = np.ones(int(pick.sum()))
                    rpa = melody.raw_pitch_accuracy(ones, ref_c[pick], ones, est_c[pick],
                                                    cent_tolerance=tolerance)
                    gross += int(round(pick.sum() * (1.0 - rpa)))
            out['gpe', '1 - raw_pitch_accuracy, +315.6/-386.3'] = gross / both.sum()
            out['ffe', 'VDE + gross errors from RPA, over N'] = vde + gross / ref.size
            if banded:
                ones = np.ones(int(both.sum()))
                out['gpe', '1 - raw_pitch_accuracy, 350 cents'] = (
                    1.0 - melody.raw_pitch_accuracy(ones, ref_c[both], ones, est_c[both],
                                                    cent_tolerance=350.0))
            out['rmse_cents', 'cents from hz2cents'] = math.sqrt(math.fsum(
                c * c for c in (est_c[both] - ref_c[both]).tolist()) / both.sum())
    return out


# --- the comparison ------------------------------------------------------------

class Largest:
    """The largest difference seen for each measure and oracle."""

    def __init__(self):
        self.rows = {}

    def add(self, measure: str, oracle: str, got: float, want: float, case: str):
        row = self.rows.setdefault((measure, oracle), {'pairs': 0, 'nan': 0, 'abs': 0.0,
                                                       'rel': 0.0, 'case': '-'})
        row['pairs'] += 1
        if math.isnan(got) or math.isnan(want):
            if math.isnan(got) and math.isnan(want):
                row['nan'] += 1
            else:
                row['abs'] = row['rel'] = math.inf
                row['case'] = f"{case}: {got!r} against {want!r}"
            return
        diff = abs(got - want)
        if diff > row['abs']:
            row['abs'], row['case'] = diff, case
        if want:
            row['rel'] = max(row['rel'], diff / abs(want))

    def table(self) -> bool:
        """Print the table; True when every row is within its tolerance. The
        rates go by their absolute difference, the RMSE by its relative one,
        and a count has to agree exactly."""
        print(f"{'measure':<18} {'oracle':<52} {'pairs':>5} {'both nan':>8} "
              f"{'largest abs':>11} {'largest rel':>11}  where")
        ok = True
        for (measure, oracle), row in self.rows.items():
            size = row['rel'] if measure == 'rmse_cents' else row['abs']
            past = size > TOLERANCE.get(measure, 0.0)
            ok = ok and not past
            print(f"{measure:<18} {oracle:<52} {row['pairs']:>5} {row['nan']:>8} "
                  f"{row['abs']:>11.2g} {row['rel']:>11.2g}  {row['case']}"
                  f"{'   PAST TOLERANCE' if past else ''}")
        return ok


def compare(largest: Largest, ref, est, case: str, banded: bool) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")      # kudio's own warnings, for the nan
        got = f0_errors(ref, est)._asdict()
    walked = frame_by_frame(ref, est)
    for name in COUNTS:
        largest.add(name, 'frame_by_frame()', float(got[name]), float(walked[name]), case)
    for name in MEASURES:
        largest.add(name, 'frame_by_frame()', got[name], walked[name], case)
    for (name, how), want in mir_eval_scores(ref, est, banded).items():
        largest.add(name, f"mir_eval: {how}", got[name], float(want), case)


def stored_numbers() -> None:
    """The numbers tests/test_f0eval.py keeps for the built pair."""
    melody = mir_eval.melody
    ref, est = voice(), tracked(voice())
    ref_v, est_v = (ref > 0).astype(float), (est > 0).astype(float)
    ref_c, est_c = melody.hz2cents(ref), melody.hz2cents(est)
    both = (ref > 0) & (est > 0)
    ones = np.ones(int(both.sum()))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        recall, false_alarm = melody.voicing_measures(ref_v, est_v)
        rows = [
            ('overall_accuracy, unlimited tolerance',
             melody.overall_accuracy(ref_v, ref_c, est_v, est_c, cent_tolerance=np.inf)),
            ('overall_accuracy, 350 cents',
             melody.overall_accuracy(ref_v, ref_c, est_v, est_c, cent_tolerance=350.0)),
            ('voicing_recall', recall),
            ('voicing_false_alarm', false_alarm),
            ('raw_pitch_accuracy, voiced in both, 350 cents',
             melody.raw_pitch_accuracy(ones, ref_c[both], ones, est_c[both],
                                       cent_tolerance=350.0)),
        ]
    walked = frame_by_frame(ref, est)
    print(f"mir_eval {mir_eval.__version__} on the built pair (_MIR_EVAL_0_8_2 in the tests):")
    for name, value in rows:
        print(f"    {name:<48} {float(value)!r}")
    print("frame_by_frame() on it (_FRAME_BY_FRAME in the tests):")
    for name in ('fpe', 'rmse_cents'):
        print(f"    {name:<48} {walked[name]!r}")
    print("    counts: " + ", ".join(f"{k}={walked[k]}" for k in COUNTS))


def quoted_figures() -> None:
    """Figures in the module docstring that come from arithmetic, not from
    kudio's code, and that no test pins."""
    print("Rabiner et al.'s gross error, a pitch period 1 ms off:")
    for hz in (100.0, 200.0):
        period = 1000.0 / hz                         # ms
        low = 1000.0 / (period + 1.0) / hz - 1.0
        high = 1000.0 / (period - 1.0) / hz - 1.0
        print(f"    at {hz:.0f} Hz: {100 * low:+.1f}% or {100 * high:+.1f}%")
    print(f"20% in cents: {UP:+.1f} above, {-DOWN:+.1f} below; 50 cents is "
          f"{100 * (2 ** (50 / 1200) - 1):+.1f}% and {100 * (2 ** (-50 / 1200) - 1):+.1f}%")


def main() -> int:
    largest = Largest()
    compare(largest, voice(), tracked(voice()), "the built pair", banded=True)
    edges = edge_pairs()
    for case, ref, est in edges:
        compare(largest, ref, est, case, banded=True)
    for seed in range(400):
        banded = seed < 200
        ref, est = random_pair(seed, banded)
        compare(largest, ref, est, f"seed {seed}, {ref.size} frames", banded)
    print(f"kudio.core.f0eval against mir_eval {mir_eval.__version__} and "
          f"frame_by_frame(): the built pair, {len(edges)} pairs at the edges and "
          f"400 random ones (numpy {np.__version__}, Python "
          f"{sys.version.split()[0]})\n")
    ok = largest.table()
    print()
    stored_numbers()
    print()
    quoted_figures()
    print()
    print("every difference within its tolerance" if ok
          else "SOME DIFFERENCES ARE PAST TOLERANCE")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
