# -*- coding: utf-8 -*-
"""Check kudio's PESQ against the conformance data ITU-T published with the
PESQ reference implementation, and measure the 2018 wideband correction in
ITU's own table.

Run it from anywhere; it imports the kudio of the checkout it sits in, and
checks 1 and 2 need the pesq package (kudio[pesq])::

    python tools/validation/pesq_conformance.py --annex-a ZIP [--corrigendum-2 ZIP]

The two ZIPs are files ITU-T gives out free, read as they come, without
unpacking:

- --annex-a: P.862 (2001) Amendment 2 (11/2005), "Revised Annex A --
  Reference implementations and conformance testing for ITU-T Recs P.862,
  P.862.1 and P.862.2", the English software item,
  T-REC-P.862-200511-W!Amd2!SOFT-ZST-E.zip, 5,675,673 bytes, SHA-256
  811992a9b71a7acbb38c69fb023b827a2b5b90ad38c28db8331f8008a2126346. It
  holds the Annex's text, the C code (version 2.0, October 2005), the
  conformance tables and the 8 kHz recordings of conformance test 2(b).
- --corrigendum-2: P.862 (2001) Corrigendum 2 (03/2018),
  T-REC-P.862-201803-W!Cor2!ZIP-E.zip, 277,432 bytes, SHA-256
  052e6637883b28a868ab7ec45c70e4eb6bf923135842862d3b347d76bafc1e08, with the
  wideband conformance table as the corrected code gives it.

What it checks:

1. Conformance test 2(b) of the Annex (A.3.2.2(b)): its 40 VoIP pairs at
   8 kHz, 22 of them with sides of two lengths, scored with
   kudio.pesq(mode='nb', scale='raw') and set against the raw scores of the
   C code, which the table gives to three decimals. The Annex passes an
   implementation whose difference exceeds 0.05 in at most one pair and 0.5
   in none. Here every difference has to stay within the table's rounding,
   0.0005, plus 1e-5 for the float32 MOS-LQO the pesq package returns and
   kudio maps back.
2. The same 40 pairs with the longer side cut to the shorter one's length
   first, which shows what cutting does to a PESQ score: PESQ finds the
   delay between the two itself and works over both as they are, so the
   score moves even though what the cut drops here is silence or low noise.
3. With --corrigendum-2: how far the corrected wideband scores in ITU's
   table are from the 2005 ones, over Supplement 23's 1,736 pairs. Nothing
   runs here, since neither ZIP has those recordings; the 2005 scores are
   the ones the pesq package, and so kudio, gives.

Tests 1(a), 1(b), 2(a) and 4 of the Annex need the Supplement 23 recordings
or ITU's G.191 resampler, which neither ZIP has.

It prints the largest differences and exits with 1 if one is past its
tolerance. Without the pesq package it skips checks 1 and 2 above and still
measures the correction.
"""
from __future__ import annotations

import argparse
import io
import math
import statistics
import sys
import time
import wave
import zipfile
from pathlib import Path

import numpy as np

SR = 8000
#: the CD inside the Annex A ZIP, named by ITU with two spaces
CD = 'Software/P862_annex_A_2005_CD  wav final.zip'
CONFORM = 'P862_annex_A_2005_CD/conform/'
#: half a unit in the table's third decimal, and 1e-5 for the float32 MOS-LQO
ROUNDING = 0.0005 + 1e-5

RESULTS = []


def report(check, n, diff, tolerance):
    RESULTS.append((check, n, diff, tolerance))
    print(f"{check}: {n} cases, max |diff| {diff:.2e}")


def table(text, column):
    """(reference, degraded) -> the score in *column*, from one of ITU's
    tab-separated conformance tables."""
    scores = {}
    for line in text.splitlines()[1:]:
        fields = [f.strip() for f in line.split('\t')]
        if len(fields) > column:
            scores[(fields[0], fields[1])] = float(fields[column])
    return scores


def recording(cd, name):
    """One 8 kHz recording of test 2(b), as 16-bit samples."""
    data = cd.read(CONFORM + name.rsplit('/', 1)[-1])
    with wave.open(io.BytesIO(data)) as w:
        shape = (w.getnchannels(), w.getsampwidth(), w.getframerate())
        if shape != (1, 2, SR):
            raise ValueError(f"{name}: {shape}, not mono 16-bit at {SR} Hz")
        frames = w.readframes(w.getnframes())
    # the C code skips a 44-byte header and reads the rest as samples, so a
    # file with more in its header would not be scored on the same samples
    if len(data) - 44 != len(frames):
        raise ValueError(f"{name}: the header is not 44 bytes")
    return np.frombuffer(frames, dtype='<i2')


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--annex-a', required=True, metavar='ZIP',
                        help="T-REC-P.862-200511-W!Amd2!SOFT-ZST-E.zip, as ITU gives it out")
    parser.add_argument('--corrigendum-2', metavar='ZIP',
                        help="T-REC-P.862-201803-W!Cor2!ZIP-E.zip, as ITU gives it out")
    args = parser.parse_args()

    start = time.perf_counter()
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import kudio
    from importlib.metadata import PackageNotFoundError, version
    try:
        package = f"pesq {version('pesq')}"
    except PackageNotFoundError:
        package = None
    print(f"kudio {kudio.__version__} from {Path(kudio.__file__).parent}; "
          f"{package or 'no pesq'}, numpy {np.__version__}")

    with zipfile.ZipFile(args.annex_a) as outer:
        cd = zipfile.ZipFile(io.BytesIO(outer.read(CD)))
    if package is None:
        print("\ntest 2(b): skipped, since the pesq package is not installed")
    else:
        conformance(kudio, cd)

    if args.corrigendum_2:
        before = table(cd.read(CONFORM + 'supp23_wb.txt').decode('latin-1'), 3)
        with zipfile.ZipFile(args.corrigendum_2) as corrigendum:
            after = table(corrigendum.read('conform/supp23_wb.txt').decode('latin-1'), 3)
        shift = [after[k] - before[k] for k in before if k in after]
        print(f"\nwideband MOS-LQO in ITU's Supplement 23 table, corrected minus 2005, "
              f"{len(shift)} pairs: mean {statistics.mean(shift):+.3f}, median "
              f"{statistics.median(shift):+.3f}, from {min(shift):+.3f} to "
              f"{max(shift):+.3f}; {sum(s < 0 for s in shift)} pairs lower")
    else:
        print("\nno --corrigendum-2: the wideband correction is not measured")

    if not RESULTS:
        print(f"\n{time.perf_counter() - start:.0f} s; test 2(b) skipped"
              + (", only the correction measured" if args.corrigendum_2 else ""))
        return 0
    width = max(len(r[0]) for r in RESULTS)
    print(f"\n{'check':<{width}} {'cases':>6} {'max |diff|':>11} {'limit':>8}")
    failed = False
    for check, n, diff, tolerance in RESULTS:
        past = diff > tolerance
        failed |= past
        limit = '-' if math.isinf(tolerance) else f"{tolerance:.1e}"
        print(f"{check:<{width}} {n:>6} {diff:>11.2e} {limit:>8}{'  PAST' if past else ''}")
    print(f"\n{time.perf_counter() - start:.0f} s; "
          + ("SOME DIFFERENCES ARE PAST TOLERANCE" if failed
             else "every difference within its tolerance"))
    return 1 if failed else 0


def conformance(kudio, cd):
    """Checks 1 and 2: test 2(b) on the pairs whole, then cut."""
    expected = table(cd.read(CONFORM + 'voipref_8k.txt').decode('latin-1'), 3)
    whole, cut, lengths = [], [], 0
    for (ref_name, deg_name), score in expected.items():
        ref, deg = recording(cd, ref_name), recording(cd, deg_name)
        lengths += len(ref) != len(deg)
        ours = kudio.pesq(ref, deg, SR, mode='nb', scale='raw')
        n = min(len(ref), len(deg))
        shorter = kudio.pesq(ref[:n], deg[:n], SR, mode='nb', scale='raw')
        whole.append((abs(ours - score), deg_name, score, ours))
        cut.append((abs(shorter - score), deg_name, score, shorter))
    whole.sort(reverse=True)
    cut.sort(reverse=True)
    over = [d for d, *_ in whole if d > 0.05]
    print(f"\ntest 2(b): {len(whole)} pairs, {lengths} of them with sides of two lengths")
    for diff, name, score, ours in whole[:3]:
        print(f"  {name}: table {score:.3f}, kudio {ours:.5f}")
    print(f"  the Annex's own limits: {len(over)} past 0.05 (one allowed), "
          f"{sum(d > 0.5 for d, *_ in whole)} past 0.5 (none allowed)")
    report("test 2(b), raw P.862 vs the table", len(whole), whole[0][0], ROUNDING)

    print("\nthe same pairs cut to the shorter side first:")
    for diff, name, score, shorter in cut[:3]:
        print(f"  {name}: table {score:.3f}, cut {shorter:.5f}")
    print(f"  {sum(d > 0.05 for d, *_ in cut)} past 0.05, "
          f"{sum(d > 0.5 for d, *_ in cut)} past 0.5")
    report("test 2(b), cut to one length first", len(cut), cut[0][0], math.inf)


if __name__ == '__main__':
    sys.exit(main())
