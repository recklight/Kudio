# -*- coding: utf-8 -*-
"""Diarization error rate, Jaccard error rate and the detection error rate
of speech activity."""
from __future__ import annotations

import math
import warnings

import pytest

import kudio
from kudio import der, detection_error_rate, diarization_errors, jer
from kudio.core.diarization import Turn

#: A 0-10 s and B 10-20 s; the system moves the change to 12 s
REF = [(0.0, 10.0, 'A'), (10.0, 20.0, 'B')]
HYP = [(0.0, 12.0, 'x'), (12.0, 20.0, 'y')]

#: two recordings of tools/validation/diarization.py, with pyannote.metrics
#: 4.2.0's DER with no collar and the overlap scored; its identification
#: error rate with 0.25 s a side and the overlap left out, on the system
#: speakers renamed by md-eval.pl's mapping; its detection error rate; and
#: dscore's JER on 10 ms frames
REF0 = [(52.52, 52.95, 'r3'), (36.1, 41.1, 'r2'), (14.39, 16.48, 'r0'), (3.67, 10.57, 'r0'),
        (29.78, 31.61, 'r0'), (18.66, 22.57, 'r1'), (5.38, 11.34, 'r3'), (26.25, 29.91, 'r3'),
        (11.27, 14.76, 'r1'), (23.89, 28.64, 'r2'), (28.12, 28.84, 'r1'), (11.17, 12.45, 'r2'),
        (18.3, 22.73, 'r2'), (51.73, 57.29, 'r0'), (56.87, 60.57, 'r2'), (32.3, 38.13, 'r1'),
        (52.99, 58.54, 'r3'), (57.1, 59.38, 'r1'), (47.4, 48.43, 'r3')]
HYP0 = [(35.53, 36.16, 'h0'), (41.61, 43.97, 'h0'), (46.2, 51.36, 'h0'), (14.34, 18.78, 'h0'),
        (19.62, 24.55, 'h0'), (2.67, 6.2, 'h2')]
REF2 = [(26.28, 31.67, 'r0'), (51.55, 53.37, 'r0'), (22.34, 27.85, 'r1'), (6.79, 7.89, 'r0'),
        (11.09, 17.09, 'r0'), (18.31, 22.85, 'r0'), (47.19, 48.06, 'r1'), (52.15, 55.22, 'r1'),
        (57.83, 61.95, 'r0'), (37.03, 37.36, 'r1'), (0.2, 0.48, 'r0'), (1.86, 11.09, 'r1'),
        (49.43, 49.92, 'r1'), (45.56, 50.37, 'r0'), (0.09, 1.41, 'r1')]
HYP2 = [(42.39, 44.55, 'h0'), (8.91, 11.04, 'h0'), (2.43, 3.58, 'h0'), (21.29, 27.23, 'h0')]
STORED = [
    (REF0, HYP0, 1.0530701754385965, 1.4813559322033891, 0.9974787989915196, 0.9288455294625273),
    (REF2, HYP2, 0.8770458265139115, 0.8712805081912404, 0.8352007469654528, 0.8300041614648357),
]


def _quiet(fn, *args, **kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = fn(*args, **kwargs)
    return value, [str(w.message) for w in caught]


def test_a_relabelled_copy_scores_zero():
    copy = [(s, e, {'A': 7, 'B': 3}[spk]) for s, e, spk in REF]
    assert der(REF, copy) == 0.0
    assert jer(REF, copy) == 0.0


def test_a_moved_change_point_is_speaker_confusion():
    errors = diarization_errors(REF, HYP)
    assert errors == (0.0, 0.0, 2.0, 20.0)
    assert der(REF, HYP) == pytest.approx(0.1, abs=1e-15)
    # A shares 10 s of a 12 s union with x, B 8 s of a 10 s union with y
    assert jer(REF, HYP) == pytest.approx(((1 - 10 / 12) + (1 - 8 / 10)) / 2, abs=1e-15)


def test_missed_and_false_alarm_speech():
    errors = diarization_errors(REF, [(2.0, 10.0, 'x'), (10.0, 23.0, 'y')])
    assert errors.missed == pytest.approx(2.0)
    assert errors.false_alarm == pytest.approx(3.0)
    assert errors.confusion == 0.0


def test_overlapping_speech_counts_once_for_each_speaker():
    ref = [(0.0, 10.0, 'A'), (5.0, 15.0, 'B')]
    errors = diarization_errors(ref, [(0.0, 15.0, 'x')])
    # 5-10 s has two speakers and one system speaker: 5 s missed; B's
    # last 5 s go to x, which is A's
    assert errors == (5.0, 0.0, 5.0, 20.0)
    left_out = diarization_errors(ref, [(0.0, 15.0, 'x')], skip_overlap=True)
    assert left_out == (0.0, 0.0, 5.0, 10.0)


def test_the_collar_is_seconds_on_each_side_of_a_reference_boundary():
    errors = diarization_errors([(0.0, 10.0, 'A')], [(0.2, 10.0, 'x')], collar=0.25)
    assert errors.missed == 0.0
    assert errors.total == pytest.approx(10.0 - 0.25 - 0.25)
    assert der([(0.0, 10.0, 'A')], [(0.2, 10.0, 'x')], collar=0.1) == pytest.approx(0.1 / 9.8)


def test_the_uem_limits_what_is_scored():
    errors = diarization_errors(REF, HYP, uem=[(0.0, 11.0)])
    assert errors == (0.0, 0.0, pytest.approx(1.0), pytest.approx(11.0))


def test_a_corpus_is_pooled_over_its_recordings():
    reference = {'a': REF, 'b': [(0.0, 5.0, 'C')]}
    hypothesis = {'a': HYP, 'b': []}
    assert der(reference, hypothesis) == pytest.approx((2.0 + 5.0) / (20.0 + 5.0))
    assert jer(reference, hypothesis) == pytest.approx(
        ((1 - 10 / 12) + (1 - 8 / 10) + 1.0) / 3)


@pytest.mark.parametrize("case", range(len(STORED)))
def test_the_numbers_of_pyannote_metrics_and_dscore(case):
    ref, hyp, plain, collared, detection, dscore_jer = STORED[case]
    assert der(ref, hyp) == pytest.approx(plain, abs=1e-12)
    assert der(ref, hyp, collar=0.25, skip_overlap=True) == pytest.approx(collared, abs=1e-12)
    assert detection_error_rate(ref, hyp) == pytest.approx(detection, abs=1e-12)
    assert jer(ref, hyp) == pytest.approx(dscore_jer, abs=1e-12)


def test_the_speakers_are_mapped_before_the_overlap_is_left_out():
    """As md-eval.pl maps them. pyannote.metrics maps on the scored time
    alone, and gives this pair 0: with the overlap gone, only C and x are
    left, and it pairs them."""
    ref = [(0.0, 10.0, 'A'), (0.0, 10.0, 'B'), (10.0, 12.0, 'C')]
    hyp = [(0.0, 12.0, 'x')]
    assert diarization_errors(ref, hyp, skip_overlap=True) == (0.0, 0.0, 2.0, 2.0)


def test_turns_of_one_speaker_that_overlap_are_one_stretch():
    """pyannote.metrics counts such turns twice: 0.4 here, with a total of
    10 s."""
    ref = [(0.0, 4.0, 'a'), (2.0, 6.0, 'a'), (6.0, 8.0, 'b')]
    errors = diarization_errors(ref, [(0.0, 8.0, 'x')])
    assert errors.total == 8.0 and errors.der == 0.25


def test_detection_error_rate_merges_the_speakers():
    ref = [(0.0, 10.0, 'A'), (5.0, 15.0, 'B')]
    assert detection_error_rate(ref, [(1.0, 15.0, 'anyone')]) == pytest.approx(1 / 15)
    labels = [kudio.Label(1.0, 15.0, '')]
    assert detection_error_rate(ref, labels) == pytest.approx(1 / 15)


def test_turns_can_be_labels_turns_or_tuples():
    as_labels = [kudio.Label(s, e, spk) for s, e, spk in REF]
    as_turns = [Turn(s, e, spk) for s, e, spk in HYP]
    assert der(as_labels, as_turns) == der(REF, HYP)


def test_rttm_is_read_by_recording(tmp_path):
    path = tmp_path / "ref.rttm"
    path.write_text(";; a comment\n"
                    "SPEAKER rec1 1 0.00 10.00 <NA> <NA> A <NA> <NA>\n"
                    "SPEAKER rec1 1 10.00 10.00 <NA> <NA> B <NA> <NA>\n"
                    "SPKR-INFO rec1 1 <NA> <NA> <NA> unknown A <NA> <NA>\n"
                    "SPEAKER rec2 1 1.50 2.25 <NA> <NA> C <NA> <NA>\n", encoding='utf-8')
    turns = kudio.load_rttm(path)
    assert list(turns) == ['rec1', 'rec2']
    assert turns['rec1'] == [Turn(0.0, 10.0, 'A'), Turn(10.0, 20.0, 'B')]
    assert turns['rec2'] == [Turn(1.5, 3.75, 'C')]
    path.write_text("SPEAKER rec1 1 zero 10.00 <NA> <NA> A <NA> <NA>\n", encoding='utf-8')
    with pytest.raises(ValueError, match=r"^load_rttm\(\): line 1 "):
        kudio.load_rttm(path)


@pytest.mark.parametrize("args, kwargs, said", [
    (([(5.0, 4.0, 'A')], []), {}, r"ends at 4 s, before it starts"),
    (([(0.0, 'one', 'A')], []), {}, r"the end of reference\[0\] has to be a number"),
    (([(0.0, 1.0)], []), {}, r"reference\[0\] is not a turn"),
    (([(0.0, 1.0, ['A'])], []), {}, r"has to be a label"),
    (({'a': REF}, HYP), {}, r"both turn lists or both dicts"),
    (({'a': REF}, {'b': HYP}), {}, r"have no hypothesis"),
    ((REF, HYP), {'collar': -0.1}, r"collar has to be 0 or more"),
    ((REF, HYP), {'skip_overlap': 'yes'}, r"skip_overlap has to be True or False"),
    ((REF, HYP), {'uem': [(3.0, 1.0)]}, r"uem\[0\] ends before it starts"),
    (("A 0 10", HYP), {}, r"reference has to be a sequence of turns"),
])
def test_malformed_input_is_refused_by_name(args, kwargs, said):
    with pytest.raises(ValueError, match=r"^der\(\): .*" + said):
        der(*args, **kwargs)


@pytest.mark.parametrize("fn", [der, jer, detection_error_rate])
def test_a_reference_without_speech_is_nan_with_one_warning(fn):
    value, said = _quiet(fn, [], [(0.0, 1.0, 'x')])
    assert math.isnan(value)
    assert len(said) == 1 and said[0].startswith(f"{fn.__name__}(): ")
    assert said[0].endswith("; scored nan")
