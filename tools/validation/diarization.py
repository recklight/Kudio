# -*- coding: utf-8 -*-
"""Check kudio's diarization measures (kudio/core/diarization.py): der,
diarization_errors, jer and detection_error_rate, against pyannote.metrics,
dscore's JER restated, and a count on 1 ms frames.

Run it from anywhere; it imports the kudio of the checkout it sits in::

    python tools/validation/diarization.py

pyannote.metrics (MIT; I ran 4.2.0 with pyannote.core 6.1.0) is the oracle
for DER and the detection error rate; its rows are skipped without it.
md-eval.pl maps the speakers on all the time evaluated and only then adds
the collars and drops the overlap (score_speaker_diarization in
md-eval-22.pl), where pyannote.metrics maps on the time it scores. So with a
collar or without the overlap, the oracle is pyannote.metrics' identification
error rate on the system speakers renamed by md-eval.pl's mapping, which
pyannote.metrics' own Hungarian mapper finds on the uncut annotations.
dscore (github.com/nryant/dscore, BSD-2-Clause, commit e02f949), the DIHARD
scorer, computes JER on 10 ms frames from a contingency matrix; that
computation is restated here from its scorelib/score.py and
scorelib/metrics.py, and needs scipy. md-eval.pl, which dscore calls for
DER, is not run here: pyannote.metrics is.

Every recording is made here from fixed seeds: 1 to 4 reference speakers and
0 to 5 system speakers, up to 25 turns each over 60 s, each turn 0.2 to 6 s,
boundaries on a 10 ms grid, one speaker's turns merged where they overlap,
as an RTTM holds them; 300 recordings.

What it checks:

1. der against pyannote.metrics' DiarizationErrorRate with no collar and
   the overlap scored, where the two map alike; with 0.25 s on each side
   (pyannote's collar=0.5) and without the overlap, against its
   IdentificationErrorRate after md-eval.pl's mapping, and how often its
   DiarizationErrorRate parts from it; the components; and the corpus
   figure of a dict of recordings against the pooled components.
2. der against a count on 1 ms frames: the mapping found from the frames,
   and every frame counted with the RT-09 formula.
3. jer against dscore's rule restated on 10 ms frames, and how often
   pyannote.metrics' JaccardErrorRate, which maps by shared time, parts
   from it.
4. detection_error_rate against pyannote.metrics' DetectionErrorRate.
5. What pyannote.metrics does with two overlapping turns of one speaker,
   which it counts twice.

It prints the largest differences and exits with 1 if one is past its
tolerance. The run takes a few seconds.
"""
from __future__ import annotations

import math
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import kudio  # noqa: E402
from kudio.core import diarization as module  # noqa: E402

try:
    from pyannote.core import Annotation, Segment
    from pyannote.metrics.detection import DetectionErrorRate
    from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate
except ImportError:
    Annotation = None

RESULTS = []
COUNT = 300


def report(check, cases, diff, tolerance):
    RESULTS.append((check, cases, diff, tolerance))


def skipped(what, why):
    report(f'{what}: skipped, {why}', 0, 0.0, math.inf)


def turns(rng, speakers, count, prefix, duration=60.0):
    raw = []
    for _ in range(count):
        start = round(rng.uniform(0, duration - 1) * 100) / 100
        end = start + round(rng.uniform(0.2, 6.0) * 100) / 100
        raw.append(module.Turn(round(start, 2), round(end, 2), f'{prefix}{rng.randrange(speakers)}'))
    merged = []
    for speaker in sorted({t.speaker for t in raw}):
        for start, end in module._speakers([t for t in raw if t.speaker == speaker])[speaker]:
            merged.append((round(start, 2), round(end, 2), speaker))
    rng.shuffle(merged)
    return merged


def recordings():
    rng = random.Random(0)
    out = []
    while len(out) < COUNT:
        ref = turns(rng, rng.randint(1, 4), rng.randint(1, 25), 'r')
        hyp = turns(rng, rng.randint(1, 5), rng.randint(0, 25), 'h') if rng.random() > 0.05 else []
        out.append((ref, hyp))
    return out


def annotation(items):
    a = Annotation()
    for k, (start, end, speaker) in enumerate(items):
        a[Segment(start, end), k] = speaker
    return a


def quietly(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return fn(*args, **kwargs)


# --- the oracles ------------------------------------------------------------------

def frames(items, step, n):
    """``(frames, speakers)`` 0/1 labels, frame i covering [i step, (i+1) step)."""
    speakers = sorted({s for _, _, s in items})
    x = np.zeros((n, len(speakers)), dtype=np.int64)
    for start, end, speaker in items:
        x[int(round(start / step)):int(round(end / step)), speakers.index(speaker)] = 1
    return x, speakers


def frame_der(ref, hyp, step=0.001, duration=70.0):
    """DER counted frame by frame, the mapping found from the frames."""
    from scipy.optimize import linear_sum_assignment
    n = int(round(duration / step))
    r, _ = frames(ref, step, n)
    h, _ = frames(hyp, step, n)
    together = r.T @ h
    mapping = {}
    if together.size:
        rows, cols = linear_sum_assignment(together, maximize=True)
        mapping = dict(zip(rows.tolist(), cols.tolist()))
    n_ref, n_hyp = r.sum(axis=1), h.sum(axis=1)
    correct = np.zeros(n, dtype=np.int64)
    for i, j in mapping.items():
        correct += r[:, i] * h[:, j]
    errors = (np.maximum(0, n_ref - n_hyp) + np.maximum(0, n_hyp - n_ref)
              + np.minimum(n_ref, n_hyp) - correct).sum()
    return errors / n_ref.sum()


def dscore_jer(ref, hyp, step=0.010, duration=70.0):
    """dscore's JER of one recording, restated: turns_to_frames on 10 ms
    frames (frame i starts at i step and is labelled if a turn covers that
    instant), speaker durations and the contingency matrix in frames, the
    Jaccard error of every pair, and the Hungarian assignment with the
    smallest sum. A reference speaker with no system speaker scores 1."""
    from scipy.optimize import linear_sum_assignment
    n = int(duration / step)
    times = step * np.arange(n)

    def labels(items):
        names = sorted({s for _, _, s in items})
        x = np.zeros((n, len(names)), dtype=np.int64)
        for start, end, speaker in items:
            x[np.searchsorted(times, start):np.searchsorted(times, end), names.index(speaker)] = 1
        return x
    r, h = labels(ref), labels(hyp)
    ref_durs, sys_durs = r.sum(axis=0), h.sum(axis=0)
    if h.shape[1] == 0:
        return [1.0] * r.shape[1]
    cm = r.T @ h
    union = ref_durs[:, None] + sys_durs[None, :] - cm
    table = 1 - cm / union
    rows, cols = linear_sum_assignment(table)
    jers = np.ones(r.shape[1])
    jers[rows] = table[rows, cols]
    return jers.tolist()


# --- the checks -------------------------------------------------------------------

def md_eval_mapped(ref, hyp):
    """The system's annotation with its speakers renamed by md-eval.pl's
    mapping: pyannote.metrics' Hungarian mapper on the uncut annotations."""
    from pyannote.metrics.matcher import HungarianMapper
    reference, hypothesis = annotation(ref), annotation(hyp)
    mapping = HungarianMapper()(hypothesis, reference)
    renamed = {label: mapping.get(label, f'unmapped {label}') for label in hypothesis.labels()}
    return reference, hypothesis.rename_labels(mapping=renamed)


def check_der(data):
    if Annotation is None:
        skipped('der vs pyannote.metrics', 'no pyannote.metrics')
        return
    from pyannote.metrics.identification import IdentificationErrorRate
    worst = 0.0
    for ref, hyp in data:
        got = module.der(ref, hyp)
        want = quietly(DiarizationErrorRate(), annotation(ref), annotation(hyp))
        worst = max(worst, abs(got - want))
    report('der vs pyannote.metrics DER, no collar, overlap scored', len(data), worst, 1e-12)
    for collar, skip in ((0.25, False), (0.0, True), (0.25, True)):
        worst = 0.0
        parted = []
        for ref, hyp in data:
            got = module.der(ref, hyp, collar=collar, skip_overlap=skip)
            reference, mapped = md_eval_mapped(ref, hyp)
            want = quietly(IdentificationErrorRate(collar=2 * collar, skip_overlap=skip),
                           reference, mapped)
            worst = max(worst, abs(got - want))
            theirs = quietly(DiarizationErrorRate(collar=2 * collar, skip_overlap=skip),
                             annotation(ref), annotation(hyp))
            parted.append(abs(got - theirs))
        setting = (f'collar {collar} s a side, '
                   f'{"overlap left out" if skip else "overlap scored"}')
        report(f"der vs pyannote.metrics IER on md-eval's mapping, {setting}", len(data),
               worst, 1e-12)
        report(f"  pyannote.metrics DER parts in {sum(d > 1e-9 for d in parted)} of "
               f"{len(data)}, by up to", len(data), max(parted), math.inf)
    worst = 0.0
    for ref, hyp in data:
        got = module.diarization_errors(ref, hyp)
        want = quietly(DiarizationErrorRate().compute_components,
                       annotation(ref), annotation(hyp))
        worst = max(worst, abs(got.missed - want['missed detection']),
                    abs(got.false_alarm - want['false alarm']),
                    abs(got.confusion - want['confusion']), abs(got.total - want['total']))
    report('diarization_errors vs pyannote.metrics components (s)', len(data), worst, 1e-9)
    corpus_ref = {f'rec{i}': ref for i, (ref, _) in enumerate(data)}
    corpus_hyp = {f'rec{i}': hyp for i, (_, hyp) in enumerate(data)}
    metric = DiarizationErrorRate()
    for ref, hyp in data:
        quietly(metric, annotation(ref), annotation(hyp))
    report('der of a dict of recordings vs pyannote.metrics pooled', len(data),
           abs(module.der(corpus_ref, corpus_hyp) - abs(metric)), 1e-12)


def check_frames(data):
    worst = 0.0
    for ref, hyp in data[:60]:
        worst = max(worst, abs(module.der(ref, hyp) - frame_der(ref, hyp)))
    report('der vs a count on 1 ms frames', 60, worst, 1e-9)


def check_jer(data):
    try:
        import scipy  # noqa: F401
    except ImportError:
        skipped("jer vs dscore's rule", 'no scipy')
        return
    errors = []
    for ref, hyp in data:
        errors.extend(dscore_jer(ref, hyp))
    single = max(abs(module.jer(ref, hyp) - float(np.mean(dscore_jer(ref, hyp))))
                 for ref, hyp in data)
    report("jer vs dscore's rule on 10 ms frames, each recording", len(data), single, 1e-12)
    corpus_ref = {f'rec{i}': ref for i, (ref, _) in enumerate(data)}
    corpus_hyp = {f'rec{i}': hyp for i, (_, hyp) in enumerate(data)}
    report("jer of a dict of recordings vs dscore's pooled JER", len(data),
           abs(module.jer(corpus_ref, corpus_hyp) - float(np.mean(errors))), 1e-12)
    if Annotation is None:
        skipped('jer vs pyannote.metrics', 'no pyannote.metrics')
        return
    apart = [abs(module.jer(ref, hyp) - quietly(JaccardErrorRate(), annotation(ref),
                                                 annotation(hyp))) for ref, hyp in data]
    parted = sum(d > 1e-9 for d in apart)
    report(f'  pyannote.metrics JaccardErrorRate parts in {parted} of {len(data)}, '
           f'by up to', len(data), max(apart), math.inf)


def check_detection(data):
    if Annotation is None:
        skipped('detection_error_rate vs pyannote.metrics', 'no pyannote.metrics')
        return
    worst = 0.0
    for collar in (0.0, 0.25):
        for ref, hyp in data:
            got = module.detection_error_rate(ref, hyp, collar=collar)
            want = quietly(DetectionErrorRate(collar=2 * collar), annotation(ref),
                           annotation(hyp))
            worst = max(worst, abs(got - want))
    report('detection_error_rate vs pyannote.metrics, collar 0 and 0.25 s', 2 * len(data),
           worst, 1e-12)


def check_self_overlap():
    ref = [(0.0, 4.0, 'a'), (2.0, 6.0, 'a'), (6.0, 8.0, 'b')]
    hyp = [(0.0, 8.0, 'x')]
    ours = module.diarization_errors(ref, hyp)
    print(f"\nTwo overlapping turns of one speaker, 0-4 s and 2-6 s, then another "
          f"speaker 6-8 s, against one system speaker 0-8 s:\n"
          f"  kudio:    total {ours.total:g} s, DER {ours.der:.4f}")
    if Annotation is not None:
        theirs = quietly(DiarizationErrorRate().compute_components, annotation(ref),
                         annotation(hyp))
        rate = (theirs['missed detection'] + theirs['false alarm'] + theirs['confusion']) \
            / theirs['total']
        print(f"  pyannote: total {theirs['total']:g} s, DER {rate:.4f}")


def main():
    start = time.perf_counter()
    try:
        from importlib.metadata import version
        oracle = f"pyannote.metrics {version('pyannote.metrics')}"
    except Exception:                                   # noqa: BLE001
        oracle = 'no pyannote.metrics'
    print(f"kudio {kudio.__version__} from {Path(kudio.__file__).parent}; {oracle}; "
          f"numpy {np.__version__}, Python {sys.version.split()[0]}")
    data = recordings()
    check_der(data)
    check_frames(data)
    check_jer(data)
    check_detection(data)
    check_self_overlap()
    width = max(len(r[0]) for r in RESULTS)
    print(f"\n{'check':<{width}} {'cases':>5} {'max |diff|':>11} {'limit':>8}")
    failed = False
    for check, cases, diff, tolerance in RESULTS:
        over = diff > tolerance
        failed |= over
        limit = '-' if math.isinf(tolerance) else f"{tolerance:.0e}"
        print(f"{check:<{width}} {cases:>5} {diff:>11.2e} {limit:>8}{'  PAST' if over else ''}")
    print(f"\n{time.perf_counter() - start:.0f} s; "
          + ("SOME DIFFERENCES ARE PAST TOLERANCE" if failed
             else "every difference within its tolerance"))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
