# -*- coding: utf-8 -*-
"""PESQ gets the two signals whole, whatever their lengths, the way the ITU
program reads two files: it finds the delay between them itself and works over
both as they are, so cutting the longer one moves the score (by up to 0.56 on
the pairs of ITU's conformance test 2(b), where the cut drops only silence).
Everything that compares the pair sample by sample still gets the length the
two share.

CI does not build pesq, so a stand-in for the package keeps the lengths it was
handed."""
import sys
import types
import warnings

import numpy as np
import pytest

import kudio
from kudio import AudioEvaluate
from kudio.core import evaluator

from conftest import make_speech, write_wav

SR = 16000


class _TooShort(RuntimeError):
    """pesq.BufferTooShortError, for the stand-in below."""


class _Package:
    """What the stand-in was handed, and what it answers: a number, or an
    exception to raise."""

    def __init__(self):
        self.lengths = []
        self.answer = 2.5


@pytest.fixture
def package(monkeypatch):
    seen = _Package()
    module = types.ModuleType('pesq')
    module.BufferTooShortError = _TooShort
    module.NoUtterancesError = type('NoUtterancesError', (RuntimeError,), {})

    def score(fs, ref, deg, mode):
        seen.lengths.append((len(ref), len(deg)))
        if isinstance(seen.answer, Exception):
            raise seen.answer
        return seen.answer

    module.pesq = score
    monkeypatch.setitem(sys.modules, 'pesq', module)
    monkeypatch.setattr(evaluator, '_has_module', lambda name: name == 'pesq')
    return seen


def _pair():
    """Six seconds of reference and an output 0.2 s shorter, well inside
    MAX_LENGTH_MISMATCH."""
    ref = make_speech(SR)
    return ref, 0.9 * ref[:int(5.8 * SR)]


def _length_warnings(caught):
    return [str(w.message) for w in caught if "samples and" in str(w.message)]


def test_pesq_gets_both_signals_whole_and_warns_nothing_about_their_lengths(package):
    ref, deg = _pair()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert kudio.pesq(ref, deg, SR) == 2.5
    assert package.lengths == [(len(ref), len(deg))]


def test_pesq_takes_a_pair_past_the_limit_that_guards_the_cutting_metrics(package):
    ref = make_speech(SR)
    assert kudio.pesq(ref, ref[:SR], SR) == 2.5
    assert package.lengths == [(6 * SR, SR)]


def test_score_hands_pesq_the_whole_pair_cuts_it_for_the_rest_and_warns_once(package):
    ref, deg = _pair()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got = kudio.score(ref, deg, SR, ['pesq', 'snr', 'csig'])
    # pesq, then the composite for csig
    assert package.lengths == [(len(ref), len(deg))] * 2
    assert got['snr'] == kudio.snr(ref[:len(deg)], deg)
    assert got['csig'] == kudio.composite(ref[:len(deg)], deg, SR, pesq=2.5).csig
    said = _length_warnings(caught)
    assert len(said) == 1 and said[0].startswith("score(): "), said
    assert "except by PESQ, which gets both whole" in said[0]


def test_score_for_pesq_alone_checks_no_length(package):
    ref, deg = _pair()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert kudio.score(ref, deg, SR, ['pesq']) == {'pesq': 2.5}
        # past the limit that guards the metrics that cut
        assert kudio.score(ref, ref[:SR], SR, ['pesq']) == {'pesq': 2.5}
    assert package.lengths == [(len(ref), len(deg)), (len(ref), SR)]


def test_the_metrics_that_cut_still_refuse_a_lopsided_pair(package):
    """The composite gets the whole pair inside score's check, and skips its
    own; score has refused the pair before that, and a metric called after
    score still refuses on its own."""
    ref = make_speech(SR)
    with pytest.raises(ValueError, match="unmeasured"):
        kudio.score(ref, ref[:SR], SR, ['csig'])
    kudio.score(ref, ref[:int(5.8 * SR)], SR, ['pesq', 'snr'])
    with pytest.raises(ValueError, match="unmeasured"):
        kudio.snr(ref, ref[:SR])
    assert package.lengths == [(len(ref), int(5.8 * SR))]


def test_composite_scores_pesq_on_the_whole_pair_as_composite_m_does(package):
    """composite.m cuts the two to one length for LLR, WSS and segSNR, and
    hands pesq() the two files."""
    ref, deg = _pair()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got = kudio.composite(ref, deg, SR)
    assert package.lengths == [(len(ref), len(deg))]
    assert got == kudio.composite(ref[:len(deg)], deg, SR, pesq=2.5)
    said = _length_warnings(caught)
    assert len(said) == 1 and said[0].startswith("composite(): "), said


def test_eval_metrics_hands_pesq_the_whole_pair(package):
    """Lengths are checked whatever is installed, so the call does not
    change with the back-ends; the warning says PESQ got both whole."""
    ref, deg = _pair()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert kudio.eval_metrics(SR, ref, deg)[0] == 2.5
    assert package.lengths == [(len(ref), len(deg))]
    said = _length_warnings(caught)
    assert len(said) == 1 and "except by PESQ, which gets both whole" in said[0]


def test_audio_evaluate_hands_pesq_each_pair_of_files_whole(package, tmp_path):
    ref, deg = _pair()
    write_wav(tmp_path / "clean.wav", ref)
    write_wav(tmp_path / "out.wav", deg)
    ev = AudioEvaluate([tmp_path / "clean.wav"], [tmp_path / "out.wav"])
    ev.enhanced_list = [tmp_path / "out.wav"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert ev.pesq_score() == (2.5, 2.5)
    # the noisy file as the baseline, then the same file as the output
    assert package.lengths == [(len(ref), len(deg))] * 2


def test_audio_evaluate_scores_pesq_alone_whatever_the_lengths(package, tmp_path):
    ref = make_speech(SR)
    write_wav(tmp_path / "clean.wav", ref)
    write_wav(tmp_path / "out.wav", ref[:SR])
    ev = AudioEvaluate([tmp_path / "clean.wav"], [tmp_path / "out.wav"])
    ev.enhanced_list = [tmp_path / "out.wav"]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert ev.pesq_score() == (2.5, 2.5)
    assert package.lengths == [(6 * SR, SR)] * 2


def test_compare_enhancers_hands_pesq_the_whole_pair(package):
    ref, noisy = _pair()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = kudio.compare_enhancers(noisy, SR, ['specsub'], reference=ref)
    # the untouched input, then the method's output, against the whole reference
    assert package.lengths == [(len(ref), len(noisy))] * 2
    assert [r.pesq for r in results] == [2.5]


def test_compare_enhancers_calls_a_faint_pair_faint_once_a_scoring(package):
    ref, noisy = _pair()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        kudio.compare_enhancers(1e-5 * noisy, SR, ['specsub'], reference=1e-5 * ref)
    faint = [str(w.message) for w in caught if "peak at" in str(w.message)]
    # the untouched input, then the method's output
    assert len(faint) == 2, faint


def test_an_empty_side_scores_nan_and_says_which_side(package):
    ref = make_speech(SR)
    with pytest.warns(UserWarning, match=r"^pesq\(\): deg is empty; scored nan$"):
        assert np.isnan(kudio.pesq(ref, ref[:0], SR))
    with pytest.warns(UserWarning, match=r"^pesq\(\): the pair is empty; scored nan$"):
        assert np.isnan(kudio.pesq(ref[:0], ref[:0], SR))
    assert package.lengths == []


def test_too_short_gives_both_lengths_when_they_differ(package):
    ref = make_speech(SR)
    package.answer = _TooShort()
    with pytest.warns(UserWarning, match=r"at least 0\.25 s, 4000 samples, on each "
                                         r"side, and ref has 96000 and deg 3200; scored nan"):
        assert np.isnan(kudio.pesq(ref, ref[:SR // 5], SR))
    with pytest.warns(UserWarning, match=r"at least 0\.25 s, 4000 samples, and the "
                                         r"pair has 3200; scored nan"):
        assert np.isnan(kudio.pesq(ref[:SR // 5], ref[:SR // 5], SR))
    # one sample short, which a time to the millisecond would round up to 0.250 s
    with pytest.warns(UserWarning, match=r"and ref has 96000 and deg 3999;"):
        assert np.isnan(kudio.pesq(ref, ref[:SR // 4 - 1], SR))


@pytest.fixture
def with_stoi(package, monkeypatch):
    """PESQ installed and a stand-in STOI that keeps what it is given: the pair
    is cut for STOI and PESQ still gets it whole."""
    from kudio.core import _stoi
    lengths = []

    def stoi(ref, deg, fs, extended=False):
        lengths.append((len(ref), len(deg)))
        return 0.5

    # what score() calls; stoi() is pystoi's interface around it
    monkeypatch.setattr(_stoi, 'stoi_or_none', stoi)
    monkeypatch.setattr(evaluator, '_has_module', lambda name: name == 'pesq')
    return lengths


def test_eval_metrics_cuts_for_stoi_and_hands_pesq_the_whole_pair(package, with_stoi):
    ref, deg = _pair()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert kudio.eval_metrics(SR, ref, deg)[:2] == (2.5, 0.5)
    assert package.lengths == [(len(ref), len(deg))]
    assert with_stoi == [(len(deg), len(deg))]
    said = _length_warnings(caught)
    assert len(said) == 1 and "except by PESQ, which gets both whole" in said[0]


def test_eval_metrics_refuses_a_lopsided_pair_whatever_is_installed(package, with_stoi):
    ref = make_speech(SR)
    with pytest.raises(ValueError, match="unmeasured"):
        kudio.eval_metrics(SR, ref, ref[:SR])
    assert package.lengths == [] and with_stoi == []


def test_audio_evaluate_eval_cuts_for_stoi_and_hands_pesq_each_pair_whole(package, with_stoi,
                                                                         tmp_path):
    ref, deg = _pair()
    write_wav(tmp_path / "clean.wav", ref)
    write_wav(tmp_path / "out.wav", deg)
    ev = AudioEvaluate([tmp_path / "clean.wav"], [tmp_path / "out.wav"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ev.eval("same", [tmp_path / "out.wav"])
    # the noisy file as the baseline, then the same file as the output
    assert package.lengths == [(len(ref), len(deg))] * 2
    assert with_stoi == [(len(deg), len(deg))] * 2
    said = _length_warnings(caught)
    assert said and all("except by PESQ, which gets both whole" in s for s in said)


@pytest.mark.parametrize("given, says", [(None, True), (2.0, False)])
def test_composite_says_pesq_got_both_whole_only_when_it_computes_it(package, given, says):
    ref, deg = _pair()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        kudio.composite(ref, deg, SR, pesq=given)
    said = _length_warnings(caught)
    assert len(said) == 1 and ("except by PESQ" in said[0]) is says


@pytest.mark.parametrize("names, says", [(['csig'], True), (['snr'], False)])
def test_score_says_pesq_got_both_whole_only_when_pesq_runs(package, names, says):
    ref, deg = _pair()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        kudio.score(ref, deg, SR, names)
    said = _length_warnings(caught)
    assert len(said) == 1 and ("except by PESQ" in said[0]) is says


def test_score_for_pesq_alone_with_an_empty_side_says_which(package):
    ref = make_speech(SR)
    with pytest.warns(UserWarning, match=r"^pesq\(\): deg is empty; scored nan$"):
        got = kudio.score(ref, ref[:0], SR, ['pesq'])
    assert np.isnan(got['pesq']) and package.lengths == []


def test_compare_enhancers_says_pesq_got_both_whole(package):
    ref, noisy = _pair()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        kudio.compare_enhancers(noisy, SR, ['specsub'], reference=ref)
    said = _length_warnings(caught)
    assert said and all("except by PESQ, which gets both whole" in s for s in said)


def test_a_raw_score_never_comes_back_over_4_5(package):
    """4.548638... is the float32 MOS-LQO of a near-perfect pair; mapped
    back it is 4.5 and a hair, which composite would refuse as no raw score."""
    ref = make_speech(8000)
    package.answer = np.float32(4.548638343811035)
    raw = kudio.pesq(ref, ref, 8000, mode='nb', scale='raw')
    assert raw == 4.5
    assert kudio.composite(ref, ref, 8000, pesq=raw).pesq == 4.5
