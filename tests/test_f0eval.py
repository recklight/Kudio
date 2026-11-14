# -*- coding: utf-8 -*-
"""Pitch-tracking errors: the definitions on tracks small enough to count by
hand, mir_eval's melody measures where its definitions and these meet, and
the tracks that are refused or cannot be scored."""
from __future__ import annotations

import math
import statistics
import warnings
from fractions import Fraction

import numpy as np
import pytest

from kudio.core.f0eval import (F0Errors, f0_errors, f0_frame_error, f0_rmse,
                               fine_pitch_error, gross_pitch_error,
                               voicing_decision_error)
from kudio.core.pitch import PitchTrack

MEASURES = [gross_pitch_error, voicing_decision_error, f0_frame_error,
            fine_pitch_error, f0_rmse, f0_errors]

# 20% of the reference, in cents above it and below it
UP, DOWN = 1200.0 * math.log2(1.2), -1200.0 * math.log2(0.8)


def _quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return fn(*args, **kwargs)


def _voice(n: int = 1000) -> np.ndarray:
    """A reference track: syllables of 26 voiced frames with 11 unvoiced
    frames between them, on an F0 wandering between 115 and 171 Hz."""
    i = np.arange(n)
    hz = 140.0 * 2.0 ** ((300.0 * np.sin(2 * np.pi * i / 611.0)
                          + 40.0 * np.sin(2 * np.pi * i / 37.0)) / 1200.0)
    hz[i % 37 >= 26] = 0.0
    return hz


def _tracked(ref: np.ndarray) -> np.ndarray:
    """An estimate of *ref* with each kind of error on frames known in
    advance: within 42 cents on most frames voiced in both; an octave up where
    i % 53 == 7, an octave down where i % 71 == 30 and 450 cents up where
    i % 97 == 50; the first two frames of every syllable missed, and one frame
    voiced after its end. No error lies between 42 and 420 cents, so a cent
    tolerance anywhere in there sorts the frames as the 20% rule does."""
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


def _random_pair(seed: int, n: int = 2000):
    """Tracks with errors of every size: 30% of the reference unvoiced, 30%
    of the estimate, fine errors of 40 cents (one SD) and a tenth of the
    frames anywhere within two octaves."""
    rng = np.random.default_rng(seed)
    ref = 70.0 * 2.0 ** (2.5 * rng.random(n))
    ref[rng.random(n) < 0.3] = 0.0
    cents = 40.0 * rng.standard_normal(n)
    wild = rng.random(n) < 0.1
    cents[wild] = 2400.0 * (2.0 * rng.random(int(wild.sum())) - 1.0)
    base = np.where(ref > 0, ref, 70.0 * 2.0 ** (2.5 * rng.random(n)))
    est = base * 2.0 ** (cents / 1200.0)
    est[rng.random(n) < 0.3] = 0.0
    return ref, est


def _as_track(hz, sr: int = 16000, hop: int = 160) -> PitchTrack:
    """*hz* as kudio.f0 would hand it back: nan where unvoiced, frame i at
    i * hop / sr seconds."""
    hz = np.asarray(hz, dtype=float)
    voiced = hz > 0
    return PitchTrack(np.where(voiced, hz, np.nan), voiced, voiced.astype(float),
                      np.arange(hz.size) * hop / sr, sr, 65.0, 400.0, hop)


def _frame_by_frame(ref, est, threshold=0.2):
    """The measures one frame at a time, in plain Python: [R76]'s four cases,
    [CA09]'s ratio |est / ref - 1|, and statistics.pstdev for FPE."""
    n = vu = uv = vv = gross = 0
    fine, cents = [], []
    for r, e in zip(np.asarray(ref, float).tolist(), np.asarray(est, float).tolist()):
        n += 1
        r_on, e_on = r > 0, e > 0
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
    return dict(gpe=gross / vv, vde=(vu + uv) / n, ffe=(vu + uv + gross) / n,
                fpe=statistics.pstdev(fine),
                rmse_cents=math.sqrt(math.fsum(c * c for c in cents) / len(cents)),
                frames=n, voiced_both=vv, gross=gross, voiced_to_unvoiced=vu,
                unvoiced_to_voiced=uv)


# ------------------------------------------------------------ the definitions

def test_the_example_in_the_module_docstring():
    ref = [0, 100, 102, 104, 0, 0, 120, 121]
    est = [0, 101, 204, 0, 0, 98, 119, 0]
    assert _quiet(gross_pitch_error, ref, est) == 1 / 3
    assert _quiet(voicing_decision_error, ref, est) == 0.375
    assert _quiet(f0_frame_error, ref, est) == 0.5
    got = _quiet(f0_errors, ref, est)
    assert isinstance(got, F0Errors)
    assert got[5:] == (8, 3, 1, 2, 1)
    # f0_errors' docstring: 1% and -0.83% left for FPE, 1200 cents in the RMSE
    assert got.fpe == pytest.approx(0.009166666666666667, rel=1e-12)
    assert got.fpe == pytest.approx(statistics.pstdev([0.01, -1.0 / 120.0]), rel=1e-12)
    assert got.rmse_cents == pytest.approx(692.9421890055355, rel=1e-12)
    assert got.rmse_cents == pytest.approx(
        math.sqrt((1200.0 ** 2 + (1200 * math.log2(1.01)) ** 2
                   + (1200 * math.log2(119 / 120)) ** 2) / 3), rel=1e-12)


def test_the_test_pair_counts_what_it_was_built_with():
    """648 frames voiced in both, 28 of them gross errors; 55 V/U errors,
    the first two frames of 27 syllables and the one frame of a 28th that
    the track cuts short; and 27 U/V errors, one after each full syllable."""
    got = _quiet(f0_errors, _voice(), _tracked(_voice()))
    assert got[5:] == (1000, 648, 28, 55, 27)
    assert got.gpe == 28 / 648
    assert got.vde == 82 / 1000
    assert got.ffe == 110 / 1000
    # [CA09] eq. 5
    assert got.ffe == pytest.approx(got.vde + got.voiced_both / got.frames * got.gpe,
                                    rel=1e-15)
    for name, value in _frame_by_frame(_voice(), _tracked(_voice())).items():
        assert getattr(got, name) == pytest.approx(value, rel=1e-12), name


def test_each_function_gives_the_number_f0_errors_gives():
    ref, est = _random_pair(3)
    together = _quiet(f0_errors, ref, est)
    assert _quiet(gross_pitch_error, ref, est) == together.gpe
    assert _quiet(voicing_decision_error, ref, est) == together.vde
    assert _quiet(f0_frame_error, ref, est) == together.ffe
    assert _quiet(fine_pitch_error, ref, est) == together.fpe
    assert _quiet(f0_rmse, ref, est) == together.rmse_cents


@pytest.mark.parametrize("ref, est, gross", [
    (100.0, 120.0, False),    # exactly 20% up
    (100.0, 80.0, False),     # exactly 20% down
    (125.0, 100.0, False),    # 20% under 125
    (100.0, 125.0, True),     # 25% over 100: the same pair the other way round
    (100.0, 120.001, True),
    (100.0, 79.999, True),
    (200.0, 100.0, True),     # an octave down
    (100.0, 200.0, True),     # an octave up
])
def test_a_gross_error_is_more_than_20_percent_off_the_reference(ref, est, gross):
    """Strict and measured against the reference, as [CA09]'s eq. 3 writes
    it: 125 Hz against 100 is a gross error, 100 against 125 is not."""
    assert _quiet(gross_pitch_error, [ref], [est]) == (1.0 if gross else 0.0)


def test_exactly_20_percent_means_in_the_binary_values_the_arrays_hold():
    """The module docstring: 120 Hz against 100 is not a gross error. 120.12
    against 100.1 is one, because the doubles nearest those two decimals lie
    a little more than 20% apart, and 120.24 against 100.2 is not."""
    assert _quiet(gross_pitch_error, [100.0], [120.0]) == 0.0
    assert _quiet(gross_pitch_error, [100.1], [120.12]) == 1.0
    assert Fraction(120.12) - Fraction(100.1) > Fraction(100.1) / 5
    assert _quiet(gross_pitch_error, [100.2], [120.24]) == 0.0
    assert Fraction(120.24) - Fraction(100.2) <= Fraction(100.2) / 5


@pytest.mark.parametrize("power", [-20, -3, 0, 7, 40])
def test_a_tie_in_binary_is_no_gross_error_at_any_size_and_one_step_past_it_is(power):
    """References 5 m 2^power with estimates 6 m and 4 m 2^power, all exact
    in binary and exactly 20% off, for m = 1 to 3000; then each estimate one
    double further out."""
    m = np.arange(1, 3001, dtype=np.float64)
    ref = np.ldexp(5.0 * m, power)
    for est, outward in ((np.ldexp(6.0 * m, power), np.inf), (np.ldexp(4.0 * m, power), 0.0)):
        assert _quiet(gross_pitch_error, ref, est) == 0.0
        assert _quiet(gross_pitch_error, ref, np.nextafter(est, outward)) == 1.0


def test_20_percent_is_315_6_cents_up_and_386_3_down():
    assert UP == pytest.approx(315.6, abs=0.05)
    assert DOWN == pytest.approx(386.3, abs=0.05)
    ref = np.full(4, 150.0)
    est = ref * 2.0 ** (np.array([315.5, 315.7, -386.2, -386.4]) / 1200.0)
    got = _quiet(f0_errors, ref, est)
    assert got.gross == 2
    assert _quiet(gross_pitch_error, ref[[0, 2]], est[[0, 2]]) == 0.0
    assert _quiet(gross_pitch_error, ref[[1, 3]], est[[1, 3]]) == 1.0


def test_mir_evals_default_tolerance_is_2_9_percent_up_and_2_8_down():
    assert 2.0 ** (50 / 1200) - 1 == pytest.approx(0.029, abs=5e-4)
    assert 1 - 2.0 ** (-50 / 1200) == pytest.approx(0.028, abs=5e-4)


def test_a_tighter_threshold_counts_more_gross_errors():
    """At 2% some of the test pair's fine errors, which reach 42 cents,
    count as gross ones; FPE then covers only what is left."""
    ref, est = _voice(), _tracked(_voice())
    loose = _quiet(f0_errors, ref, est)
    tight = _quiet(f0_errors, ref, est, threshold=0.02)
    assert tight.gross > loose.gross
    assert tight.voiced_both == loose.voiced_both
    assert tight.vde == loose.vde
    assert tight.fpe < loose.fpe
    assert tight.rmse_cents == loose.rmse_cents
    want = _frame_by_frame(ref, est, threshold=0.02)
    assert tight.gross == want['gross']
    assert tight.fpe == pytest.approx(want['fpe'], rel=1e-12)


def test_fpe_is_the_standard_deviation_over_the_frames_without_a_gross_error():
    """It divides by the number of those frames, as [R76]'s eq. 3 does.
    MATLAB's std divides by one fewer, which reads 0.5% higher at 100
    frames."""
    rng = np.random.default_rng(7)
    ref = 100.0 + 100.0 * rng.random(100)
    rel = 0.02 * rng.standard_normal(100)
    est = ref * (1.0 + rel)
    est[[3, 50]] = 2.0 * ref[[3, 50]]          # two octave errors, left out
    fine = [x for k, x in enumerate(rel.tolist()) if k not in (3, 50)]
    got = _quiet(fine_pitch_error, ref, est)
    assert got == pytest.approx(statistics.pstdev(fine), rel=1e-12)
    assert got == pytest.approx(np.std((est - ref)[ref * 1.2 > est] / ref[ref * 1.2 > est]),
                                rel=1e-12)
    assert math.sqrt(100 / 99) - 1 == pytest.approx(0.005, abs=5e-5)


def test_a_constant_offset_is_no_fpe_but_is_rmse():
    """An estimate 1% sharp on every frame: FPE, a standard deviation, is 0,
    and the RMSE is 1200 log2(1.01) = 17.2 cents."""
    ref = _voice()
    est = 1.01 * ref
    got = _quiet(f0_errors, ref, est)
    assert got.fpe == pytest.approx(0.0, abs=1e-15)
    assert got.rmse_cents == pytest.approx(17.2, abs=0.05)
    assert got.rmse_cents == pytest.approx(1200 * math.log2(1.01), rel=1e-9)
    assert got.gpe == 0.0


def test_one_octave_error_in_100_frames_is_120_cents_of_rmse():
    ref = np.linspace(100.0, 200.0, 100)
    est = ref.copy()
    est[40] *= 2.0
    assert _quiet(f0_rmse, ref, est) == pytest.approx(120.0, rel=1e-12)
    assert _quiet(gross_pitch_error, ref, est) == 0.01
    assert _quiet(fine_pitch_error, ref, est) == 0.0


def test_an_exact_octave_is_exactly_1200_cents():
    for hz in (100.0, 150.0, 333.3):
        assert _quiet(f0_rmse, [hz], [2.0 * hz]) == 1200.0
        assert _quiet(f0_rmse, [hz], [0.5 * hz]) == 1200.0


def test_a_ratio_past_the_range_of_a_double_still_has_its_cents():
    """1e300 Hz against 1e-10 is no F0 anyone has, but it is two finite
    numbers. Their ratio overflows to inf, and the other way round it falls
    below the smallest normal double."""
    got = _quiet(f0_rmse, [1e-10, 1e10], [1e300, 1e-300])
    want = 1200.0 * math.sqrt(((math.log2(1e300) - math.log2(1e-10)) ** 2
                               + (math.log2(1e-300) - math.log2(1e10)) ** 2) / 2)
    assert got == pytest.approx(want, rel=1e-12)


def test_a_ratio_that_underflows_to_0_has_its_cents_and_no_numpy_warning():
    """1e-300 Hz against 1e30: the ratio is 0 as a double, and its log is
    taken from the two sides, with nothing for numpy to warn about."""
    want = 1200.0 * (math.log2(1e30) - math.log2(1e-300))
    assert _quiet(f0_rmse, [1e30], [1e-300]) == pytest.approx(want, rel=1e-12)
    got = _quiet(f0_errors, [1e30, 100.0, 100.0], [1e-300, 101.0, 99.0])
    assert got.gross == 1
    assert got.rmse_cents == pytest.approx(math.sqrt(
        (want ** 2 + (1200 * math.log2(1.01)) ** 2
         + (1200 * math.log2(0.99)) ** 2) / 3), rel=1e-12)


def test_an_rmse_of_ln_f0_is_the_rmse_in_cents_over_1731_2():
    ref, est = _random_pair(5)
    both = (ref > 0) & (est > 0)
    ln_rmse = math.sqrt(np.mean(np.log(est[both] / ref[both]) ** 2))
    assert 1200 / math.log(2) == pytest.approx(1731.2, abs=0.05)
    assert _quiet(f0_rmse, ref, est) / ln_rmse == pytest.approx(1200 / math.log(2), rel=1e-12)


def test_a_semitone_is_5_9_hz_at_100_hz_and_11_9_at_200():
    assert 100 * (2 ** (1 / 12) - 1) == pytest.approx(5.9, abs=0.05)
    assert 200 * (2 ** (1 / 12) - 1) == pytest.approx(11.9, abs=0.05)


def test_calling_a_gross_error_unvoiced_moves_it_from_gpe_to_vde():
    """[CA09]'s reason for FFE: the frame leaves N_VV, GPE drops, VDE rises
    by 1/N, and FFE stays where it was."""
    ref, est = _voice(), _tracked(_voice())
    before = _quiet(f0_errors, ref, est)
    k = int(np.flatnonzero((ref > 0) & (est > 1.25 * ref))[0])
    given_up = est.copy()
    given_up[k] = 0.0
    after = _quiet(f0_errors, ref, given_up)
    assert after.gpe < before.gpe
    assert after.gpe == (before.gross - 1) / (before.voiced_both - 1)
    assert after.vde == pytest.approx(before.vde + 1 / before.frames, rel=1e-12)
    assert after.ffe == before.ffe


def test_silence_both_tracks_call_unvoiced_dilutes_vde_and_ffe_only():
    """300 frames with 30 voicing errors read 0.10; with 100 frames of
    silence at each end they read 0.06, while GPE, FPE and the RMSE, which
    count only frames voiced in both, stay as they were."""
    ref = np.full(300, 150.0)
    est = 151.0 + np.arange(300) % 3
    est[::10] = 0.0                                   # 30 V/U errors
    padded = [np.concatenate((np.zeros(100), x, np.zeros(100))) for x in (ref, est)]
    short, long_ = _quiet(f0_errors, ref, est), _quiet(f0_errors, *padded)
    assert short.vde == pytest.approx(0.10, rel=1e-15)
    assert long_.vde == pytest.approx(0.06, rel=1e-15)
    assert long_.ffe == pytest.approx(0.06, rel=1e-15)
    assert (long_.gpe, long_.fpe, long_.rmse_cents) == (short.gpe, short.fpe, short.rmse_cents)


def test_pooling_a_test_set_is_not_averaging_its_files():
    """5 gross errors in 10 frames and 10 in 1000 average to 0.255 and pool,
    by scoring the concatenated tracks once, to 15 / 1010 = 0.0149."""
    files = []
    for n, gross in ((10, 5), (1000, 10)):
        ref = np.full(n, 120.0)
        est = ref.copy()
        est[:gross] *= 2.0
        files.append((ref, est))
    per_file = [_quiet(gross_pitch_error, r, e) for r, e in files]
    assert per_file == [0.5, 0.01]
    assert np.mean(per_file) == pytest.approx(0.255, rel=1e-12)
    pooled = _quiet(gross_pitch_error, np.concatenate([r for r, _ in files]),
                    np.concatenate([e for _, e in files]))
    assert pooled == 15 / 1010
    assert pooled == pytest.approx(0.0149, abs=5e-5)


def test_a_one_frame_lag_on_a_glide_of_2_octaves_a_second_is_24_cents():
    """Nothing is shifted to line the tracks up: 10 ms frames, the estimate
    one frame late, every frame 0.02 octave off."""
    t = np.arange(200) * 0.01
    ref = 100.0 * 2.0 ** (2.0 * t)
    late = 100.0 * 2.0 ** (2.0 * (t - 0.01))
    assert _quiet(f0_rmse, ref, late) == pytest.approx(24.0, rel=1e-9)
    assert _quiet(gross_pitch_error, ref, late) == 0.0


def test_the_frame_times_of_kudio_f0_at_16_and_22_05_khz_part_by_2_3_ms():
    """The comment on _TIME_TOLERANCE: hops of 160 and 220 samples, frame
    100."""
    assert 100 * 160 / 16000 - 100 * 220 / 22050 == pytest.approx(0.0023, abs=5e-5)


# ------------------------------------------------------------- against mir_eval

# mir_eval 0.8.2's melody measures (Python 3.13.9, numpy 2.3.5) on _voice()
# and _tracked(_voice()), voicing 1 where the track is above 0 Hz and pitch
# from mir_eval.melody.hz2cents. raw_pitch_accuracy was given only the frames
# voiced in both, with their voicing set to 1.
_MIR_EVAL_0_8_2 = {
    'overall_accuracy, unlimited tolerance': 0.918,
    'overall_accuracy, 350 cents': 0.89,
    'voicing_recall': 0.9217638691322901,
    'voicing_false_alarm': 0.09090909090909091,
    'raw_pitch_accuracy, voiced in both, 350 cents': 0.9567901234567902,
}

# the frame-by-frame implementation above (_frame_by_frame: statistics.pstdev
# and math.fsum on Python 3.13.9) on the same pair
_FRAME_BY_FRAME = {'fpe': 0.013233767756113972, 'rmse_cents': 227.57761161915008}


def test_vde_and_ffe_are_one_minus_mir_evals_overall_accuracy():
    """With an unlimited cent tolerance (cent_tolerance=np.inf), overall
    accuracy is the share of frames whose voicing is right, so 1 - OA is
    VDE. With 350 cents, which on this pair draws the line the 20% rule
    draws, it is FFE."""
    got = _quiet(f0_errors, _voice(), _tracked(_voice()))
    assert got.vde == pytest.approx(1 - _MIR_EVAL_0_8_2['overall_accuracy, unlimited tolerance'],
                                    abs=1e-15)
    assert got.ffe == pytest.approx(1 - _MIR_EVAL_0_8_2['overall_accuracy, 350 cents'],
                                    abs=1e-15)


def test_vde_follows_from_mir_evals_voicing_recall_and_false_alarm():
    """Recall is 1 - N_VU / (voiced in ref), the false alarm rate
    N_UV / (unvoiced in ref); 703 and 297 of the pair's 1000 frames."""
    ref = _voice()
    voiced = int(np.count_nonzero(ref > 0))
    assert voiced == 703
    recall = _MIR_EVAL_0_8_2['voicing_recall']
    false_alarm = _MIR_EVAL_0_8_2['voicing_false_alarm']
    vde = ((1 - recall) * voiced + false_alarm * (ref.size - voiced)) / ref.size
    assert _quiet(voicing_decision_error, ref, _tracked(ref)) == pytest.approx(vde, abs=1e-15)


def test_gpe_is_one_minus_raw_pitch_accuracy_over_the_frames_voiced_in_both():
    got = _quiet(gross_pitch_error, _voice(), _tracked(_voice()))
    rpa = _MIR_EVAL_0_8_2['raw_pitch_accuracy, voiced in both, 350 cents']
    assert got == pytest.approx(1 - rpa, abs=1e-15)


def test_fpe_and_rmse_match_the_frame_by_frame_implementation():
    got = _quiet(f0_errors, _voice(), _tracked(_voice()))
    assert got.fpe == pytest.approx(_FRAME_BY_FRAME['fpe'], rel=1e-12)
    assert got.rmse_cents == pytest.approx(_FRAME_BY_FRAME['rmse_cents'], rel=1e-12)


def _mir_eval_scores(mir_eval, ref, est):
    """GPE, VDE and FFE out of mir_eval's measures, on any pair. Raw pitch
    accuracy draws one line in cents both ways and the 20% rule two, so the
    frames voiced in both are split by the sign of their error and each half
    gets the tolerance where 20% falls on its side: 315.6 cents above,
    386.3 below. Only an error lying exactly on the line is counted
    differently, and a random pair has none."""
    melody = mir_eval.melody
    both = (ref > 0) & (est > 0)
    gross = 0
    for side, tolerance in ((est >= ref, UP), (est < ref, DOWN)):
        pick = both & side
        if pick.any():
            ones = np.ones(int(pick.sum()))
            rpa = melody.raw_pitch_accuracy(ones, melody.hz2cents(ref[pick]), ones,
                                            melody.hz2cents(est[pick]),
                                            cent_tolerance=tolerance)
            gross += int(round(pick.sum() * (1.0 - rpa)))
    ref_v, est_v = (ref > 0).astype(float), (est > 0).astype(float)
    accuracy = melody.overall_accuracy(ref_v, melody.hz2cents(ref), est_v,
                                       melody.hz2cents(est), cent_tolerance=np.inf)
    vde = 1.0 - accuracy
    return gross / both.sum(), vde, vde + gross / ref.size


@pytest.mark.parametrize("seed", range(6))
def test_against_mir_eval_itself(seed):
    mir_eval = pytest.importorskip("mir_eval")
    ref, est = _random_pair(seed)
    got = _quiet(f0_errors, ref, est)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gpe, vde, ffe = _mir_eval_scores(mir_eval, ref, est)
    assert got.gpe == pytest.approx(gpe, abs=1e-14)
    assert got.vde == pytest.approx(vde, abs=1e-14)
    assert got.ffe == pytest.approx(ffe, abs=1e-14)
    both = (ref > 0) & (est > 0)
    cents = mir_eval.melody.hz2cents(est[both]) - mir_eval.melody.hz2cents(ref[both])
    assert got.rmse_cents == pytest.approx(math.sqrt(np.mean(cents ** 2)), rel=1e-12)


def test_the_test_pairs_goldens_against_mir_eval_itself():
    mir_eval = pytest.importorskip("mir_eval")
    melody = mir_eval.melody
    ref, est = _voice(), _tracked(_voice())
    ref_v, est_v = (ref > 0).astype(float), (est > 0).astype(float)
    ref_c, est_c = melody.hz2cents(ref), melody.hz2cents(est)
    both = (ref > 0) & (est > 0)
    ones = np.ones(int(both.sum()))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        recall, false_alarm = melody.voicing_measures(ref_v, est_v)
        live = {
            'overall_accuracy, unlimited tolerance': melody.overall_accuracy(
                ref_v, ref_c, est_v, est_c, cent_tolerance=np.inf),
            'overall_accuracy, 350 cents': melody.overall_accuracy(
                ref_v, ref_c, est_v, est_c, cent_tolerance=350),
            'voicing_recall': recall,
            'voicing_false_alarm': false_alarm,
            'raw_pitch_accuracy, voiced in both, 350 cents': melody.raw_pitch_accuracy(
                ones, ref_c[both], ones, est_c[both], cent_tolerance=350),
        }
    for name, value in _MIR_EVAL_0_8_2.items():
        assert live[name] == pytest.approx(value, rel=1e-12), name


@pytest.mark.parametrize("seed", range(6))
def test_against_the_frame_by_frame_implementation(seed):
    ref, est = _random_pair(100 + seed)
    got = _quiet(f0_errors, ref, est)
    want = _frame_by_frame(ref, est)
    for name in ('frames', 'voiced_both', 'gross', 'voiced_to_unvoiced',
                 'unvoiced_to_voiced'):
        assert getattr(got, name) == want[name], name
    for name in ('gpe', 'vde', 'ffe', 'fpe', 'rmse_cents'):
        assert getattr(got, name) == pytest.approx(want[name], rel=1e-12), name


# ----------------------------------------------------------------- the inputs

def test_nan_and_zero_both_mean_unvoiced():
    ref, est = _voice(), _tracked(_voice())
    with_nan = [np.where(x > 0, x, np.nan) for x in (ref, est)]
    assert _quiet(f0_errors, *with_nan) == _quiet(f0_errors, ref, est)
    assert _quiet(f0_errors, with_nan[0], est) == _quiet(f0_errors, ref, est)


def test_pitch_tracks_score_as_their_f0():
    ref, est = _voice(), _tracked(_voice())
    want = _quiet(f0_errors, ref, est)
    assert _quiet(f0_errors, _as_track(ref), _as_track(est)) == want
    assert _quiet(f0_errors, _as_track(ref), est) == want
    assert _quiet(f0_errors, ref, _as_track(est)) == want


def test_a_track_from_kudio_f0_scores_against_the_f0_it_was_made_with():
    """pYIN's track goes in as kudio.f0 hands it back, nan and all: a second
    of a harmonic tone gliding 2 semitones either side of 150 Hz, against
    that glide read at the track's frame times."""
    pytest.importorskip("librosa")
    from kudio.core.pitch import f0
    sr = 16000
    t = np.arange(sr) / sr
    glide = 150.0 * 2.0 ** (2.0 / 12.0 * np.sin(2 * np.pi * 1.5 * t))
    phase = 2 * np.pi * np.cumsum(glide) / sr
    track = f0(0.3 * sum(np.sin(k * phase) / k for k in range(1, 11)), sr)
    got = _quiet(f0_errors, np.interp(track.times, t, glide), track)
    assert got.frames == len(track)
    assert got.gpe == 0.0
    assert got.vde <= 0.05
    assert got.rmse_cents < 30.0


def test_a_pitch_track_reads_its_voicing_decision_not_its_leftover_f0():
    """A frame called unvoiced is unvoiced, whatever number sits in f0."""
    ref = _as_track([0, 100, 110, 0])
    est = PitchTrack(np.array([90.0, 101.0, 109.0, 95.0]),
                     np.array([False, True, True, False]), np.ones(4),
                     np.arange(4) * 0.01, 16000, 65.0, 400.0, 160)
    assert _quiet(voicing_decision_error, ref, est) == 0.0


def test_a_pitch_track_voiced_without_an_f0_is_refused():
    broken = PitchTrack(np.array([100.0, np.nan, 0.0]), np.array([True, True, True]),
                        np.ones(3), np.arange(3) * 0.01, 16000, 65.0, 400.0, 160)
    with pytest.raises(ValueError, match=r"^f0_rmse\(\): est is a PitchTrack with 2 "
                                         r"frame\(s\) marked voiced but no F0 .* frame 1"):
        f0_rmse([100.0, 100.0, 100.0], broken)


@pytest.mark.parametrize("times, match", [
    (np.array([0.0]), r"est is a PitchTrack whose times holds 1 value\(s\) for 2 frames"),
    (np.array([0.0, np.nan]), r"est is a PitchTrack with 1 frame time\(s\) that are not "
                              r"finite, the first at frame 1"),
], ids=["times of another length", "nan among the times"])
def test_a_pitch_track_with_broken_frame_times_is_refused(times, match):
    """Without one finite time per frame there is no telling whether the two
    tracks share a grid."""
    broken = PitchTrack(np.array([100.0, 110.0]), np.array([True, True]), np.ones(2),
                        times, 16000, 65.0, 400.0, 160)
    for fn in MEASURES:
        with pytest.raises(ValueError, match=rf"^{fn.__name__}\(\): {match}"):
            fn(_as_track([100.0, 110.0]), broken)
    with pytest.raises(ValueError, match=rf"^f0_rmse\(\): {match}"):
        f0_rmse([100.0, 110.0], broken)


def test_two_pitch_tracks_on_different_frame_times_are_refused():
    """kudio.f0 at 16 and at 22.05 kHz: 10 ms and 9.98 ms hops, the same
    number of frames, and frame 100 2.3 ms apart."""
    hz = np.full(101, 150.0)
    at_16k = _as_track(hz, sr=16000, hop=160)
    at_22k = _as_track(hz, sr=22050, hop=220)
    with pytest.raises(ValueError, match=r"^gross_pitch_error\(\): ref and est have 101 "
                                         r"frames each, but their frame times differ "
                                         r"by up to 2\.27 ms"):
        gross_pitch_error(at_16k, at_22k)
    # the same grid reached another way is one grid
    assert _quiet(gross_pitch_error, at_16k, _as_track(hz, sr=48000, hop=480)) == 0.0


def test_frame_times_half_a_microsecond_apart_are_one_grid_and_two_are_not():
    """_TIME_TOLERANCE is 1 microsecond. kudio.f0 at 16 and at 22.05 kHz is
    already 0.0227 ms apart at frame 1."""
    base = _as_track([150.0, 150.0, 150.0])

    def moved(seconds):
        return PitchTrack(base.f0, base.voiced, base.voiced_prob,
                          base.times + np.array([0.0, 0.0, seconds]), 16000, 65.0, 400.0, 160)

    assert _quiet(gross_pitch_error, base, moved(5e-7)) == 0.0
    with pytest.raises(ValueError, match=r"^gross_pitch_error\(\): ref and est have 3 "
                                         r"frames each, but their frame times differ "
                                         r"by up to 0\.002 ms"):
        gross_pitch_error(base, moved(2e-6))
    with pytest.raises(ValueError, match=r"^f0_errors\(\): ref and est have 2 frames "
                                         r"each, but their frame times differ by up to "
                                         r"0\.0227 ms"):
        f0_errors(_as_track([150.0, 150.0], sr=16000, hop=160),
                  _as_track([150.0, 150.0], sr=22050, hop=220))


def test_tracks_of_different_lengths_are_refused_with_what_to_do():
    ref = _voice(300)
    for fn in MEASURES:
        with pytest.raises(ValueError, match=rf"^{fn.__name__}\(\): ref has 300 frames "
                                             rf"and est 299\. .*one frame grid.*"
                                             rf"resample_melody_series"):
            fn(ref, _tracked(ref)[:299])


@pytest.mark.parametrize("which", ["ref", "est"])
def test_negative_values_are_refused_and_both_meanings_named(which):
    ref, est = _voice(50), _tracked(_voice(50))
    bad = (-ref if which == "ref" else -est)
    pair = (bad, est) if which == "ref" else (ref, bad)
    count = int(np.count_nonzero(bad < 0))
    for fn in MEASURES:
        with pytest.raises(ValueError, match=rf"^{fn.__name__}\(\): {which} has {count} "
                                             rf"negative value\(s\).*Keele.*"
                                             rf"keep = ref >= 0.*MIREX.*np\.maximum"):
            fn(*pair)


def test_a_masked_array_with_masked_frames_is_refused_and_both_meanings_named():
    """np.asarray drops the mask, which would score a frame masked for low
    confidence as voiced."""
    masked = np.ma.masked_where([False, True, False], [150.0, 150.0, 0.0])
    for fn in MEASURES:
        with pytest.raises(ValueError, match=rf"^{fn.__name__}\(\): est is a masked array "
                                             rf"with 1 masked frame\(s\).*"
                                             rf"np\.ma\.filled\(est, 0\).*unvoiced.*"
                                             rf"np\.ma\.getmaskarray\(est\)"):
            fn([150.0, 0.0, 0.0], masked)
    # with nothing masked, it is the array it holds
    assert _quiet(voicing_decision_error, np.ma.array([150.0, 0.0]), [150.0, 0.0]) == 0.0


def test_infinite_values_are_refused():
    for value in (np.inf, -np.inf):
        with pytest.raises(ValueError, match=r"^f0_frame_error\(\): est has 1 infinite "
                                             r"value\(s\)"):
            f0_frame_error([100.0, 100.0], [100.0, value])


@pytest.mark.parametrize("bad, match", [
    (np.array([True, False]), "is a boolean array, a voicing decision"),
    (np.array([100.0 + 0j, 0j]), "has to hold F0 values in Hz, got an array of complex"),
    (np.array([100.0, None], dtype=object),
     "has to hold F0 values in Hz, got an array of object"),
    ("100 120", "has to be F0 values in Hz, got str"),
    ([[100.0, 120.0], [100.0]],          # older numpy made an object array of it
     "(is not an array of F0 values|has to hold F0 values in Hz, got an array of object)"),
    (120.0, "is a single number, not a track"),
    (np.full((2, 2), 100.0), r"has shape \(2, 2\); score one track at a time"),
    (np.zeros((0, 2)), r"has shape \(0, 2\); score one track at a time"),
])
def test_what_is_not_one_track_of_numbers_is_refused(bad, match):
    for fn in MEASURES:
        with pytest.raises(ValueError, match=rf"^{fn.__name__}\(\): est {match}"):
            fn([100.0, 0.0], bad)


def test_a_column_or_a_row_is_read_as_one_track():
    ref, est = _voice(), _tracked(_voice())
    want = _quiet(f0_errors, ref, est)
    assert _quiet(f0_errors, ref[:, None], est[None, :]) == want


def test_integer_tracks_are_read_as_hz():
    """Not as audio on the type's full scale: an int16 120 is 120 Hz."""
    ref = np.array([0, 100, 120, 200, 150], dtype=np.int16)
    est = np.array([0, 101, 0, 400, 147], dtype=np.int16)
    got = _quiet(f0_errors, ref, est)
    assert got == _quiet(f0_errors, ref.astype(float), est.astype(float))
    assert got.rmse_cents == pytest.approx(math.sqrt(
        ((1200 * math.log2(1.01)) ** 2 + 1200.0 ** 2
         + (1200 * math.log2(147 / 150)) ** 2) / 3), rel=1e-12)
    assert got.fpe == pytest.approx(statistics.pstdev([0.01, -0.02]), rel=1e-12)


@pytest.mark.parametrize("threshold", [0, 0.0, 1, 1.0, -0.1, 20, float('nan'),
                                       float('inf')])
def test_a_threshold_outside_0_to_1_is_refused(threshold):
    for fn in (gross_pitch_error, f0_frame_error, fine_pitch_error, f0_errors):
        with pytest.raises(ValueError, match=rf"^{fn.__name__}\(\): threshold is a "
                                             rf"fraction of the reference F0, 0\.2"):
            fn([100.0], [100.0], threshold=threshold)


@pytest.mark.parametrize("threshold", [True, "0.2", None, [0.2]])
def test_a_threshold_that_is_not_a_number_is_refused(threshold):
    for fn in (gross_pitch_error, f0_frame_error, fine_pitch_error, f0_errors):
        with pytest.raises(ValueError, match=rf"^{fn.__name__}\(\): threshold has to "
                                             rf"be a number"):
            fn([100.0], [100.0], threshold=threshold)


def test_a_numpy_threshold_is_a_number():
    assert _quiet(gross_pitch_error, [100.0], [111.0], threshold=np.float32(0.1)) == 1.0
    assert _quiet(gross_pitch_error, [100.0], [109.0], threshold=np.float64(0.1)) == 0.0


# ---------------------------------------------------- what cannot be scored

def _empty(kind):
    if kind == 'PitchTrack':
        from kudio.core.pitch import f0
        return f0(np.zeros(0), 16000)        # what kudio.f0 hands back for no audio
    return np.zeros(kind) if isinstance(kind, tuple) else []


@pytest.mark.parametrize("kind", ['list', 'PitchTrack', (0, 1), (1, 0)])
def test_empty_tracks_score_nan_with_one_warning_each(kind):
    """An empty column or row is an empty track, as (n, 1) and (1, n) are
    tracks of n frames."""
    for fn in MEASURES:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            got = fn(_empty(kind), _empty(kind))
        assert len(caught) == 1, fn.__name__
        assert caught[0].category is UserWarning
        assert str(caught[0].message) == f"{fn.__name__}(): got two empty tracks; scored nan"
        if fn is f0_errors:
            assert all(math.isnan(x) for x in got[:5]) and got[5:] == (0, 0, 0, 0, 0)
        else:
            assert math.isnan(got)


@pytest.mark.parametrize("ref, est, why", [
    ([0.0, 0.0, 0.0], [100.0, 0.0, 120.0], "ref has no voiced frame"),
    ([100.0, 0.0, 120.0], [0.0, np.nan, 0.0], "est has no voiced frame"),
    ([100.0, 0.0, 120.0], [0.0, 110.0, 0.0], "ref and est are never voiced in the same frame"),
])
def test_no_frame_voiced_in_both_leaves_gpe_fpe_and_rmse_nan(ref, est, why):
    """VDE and FFE count every frame, so they still have their number."""
    for fn, needs in ((gross_pitch_error, "GPE needs at least 1 frame voiced in both"),
                      (fine_pitch_error, "FPE needs at least 2 frames voiced in both "
                                         "without a gross error"),
                      (f0_rmse, "the RMSE needs at least 1 frame voiced in both")):
        with pytest.warns(UserWarning) as caught:
            assert math.isnan(fn(ref, est))
        assert len(caught) == 1
        assert str(caught[0].message) == f"{fn.__name__}(): {why}, and {needs}; scored nan"
    vde = _quiet(voicing_decision_error, ref, est)
    assert _quiet(f0_frame_error, ref, est) == vde
    with pytest.warns(UserWarning) as caught:
        got = f0_errors(ref, est)
    assert len(caught) == 1
    assert str(caught[0].message) == (f"f0_errors(): {why}, so gpe, fpe and rmse_cents "
                                      f"have no frame to measure; scored nan")
    assert math.isnan(got.gpe) and math.isnan(got.fpe) and math.isnan(got.rmse_cents)
    assert got.vde == vde and got.ffe == vde and got.voiced_both == 0


def test_fpe_needs_two_frames_voiced_in_both_without_a_gross_error():
    one_fine = ([100.0, 100.0, 100.0], [101.0, 200.0, 0.0])
    with pytest.warns(UserWarning) as caught:
        assert math.isnan(fine_pitch_error(*one_fine))
    assert len(caught) == 1
    assert str(caught[0].message) == (
        "fine_pitch_error(): FPE needs at least 2 frames voiced in both without a "
        "gross error to take a standard deviation over, and the pair has 1; scored nan")
    with pytest.warns(UserWarning) as caught:
        got = f0_errors(*one_fine)
    assert len(caught) == 1
    assert str(caught[0].message) == (
        "f0_errors(): fpe needs at least 2 frames voiced in both without a gross "
        "error, and the pair has 1; scored nan")
    assert math.isnan(got.fpe)
    assert got.gpe == 0.5 and got.vde == 1 / 3 and got.ffe == 2 / 3
    assert _quiet(fine_pitch_error, [100.0, 100.0], [101.0, 99.0]) == pytest.approx(0.01)


def test_every_warning_points_at_the_line_that_called():
    for fn in MEASURES:
        with pytest.warns(UserWarning) as caught:
            fn([], [])
        assert caught[0].filename == __file__, fn.__name__
    for fn in (gross_pitch_error, fine_pitch_error, f0_rmse, f0_errors):
        with pytest.warns(UserWarning) as caught:
            fn([0.0, 100.0], [100.0, 0.0])
        assert caught[0].filename == __file__, fn.__name__
    for fn in (fine_pitch_error, f0_errors):
        with pytest.warns(UserWarning) as caught:
            fn([100.0, 100.0], [101.0, 0.0])
        assert caught[0].filename == __file__, fn.__name__


def test_the_scoring_leaves_its_inputs_alone():
    ref, est = _voice(), np.where(_tracked(_voice()) > 0, _tracked(_voice()), np.nan)
    ref_copy, est_copy = ref.copy(), est.copy()
    _quiet(f0_errors, ref, est)
    np.testing.assert_array_equal(ref, ref_copy)
    np.testing.assert_array_equal(est, est_copy)
