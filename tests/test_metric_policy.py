# -*- coding: utf-8 -*-
"""What every metric does with input it cannot score, said the same way.

Malformed input raises ValueError, its message starting ``<function>(): ``.
Input that is well formed but cannot be scored gives nan and one UserWarning,
``<function>(): <reason>; scored nan``, pointed at the line that called kudio
however the call got there. A new metric adds its cases to REFUSED and NAN.
"""
from __future__ import annotations

import math
import re
import sys
import types
import warnings

import numpy as np
import pytest

import kudio

from conftest import make_speech

SR = 16000
#: float64, so that 1e160 of it is still finite
SPEECH = make_speech(SR, 6.0, seed=0).astype(np.float64)
CLIP = SPEECH[SR:3 * SR]
STEREO = np.stack([CLIP, CLIP], axis=1)
QUALITY = ('llr', 'itakura_saito', 'cepstral_distance', 'wss', 'fw_segmental_snr')


def _corrupt(y):
    y = np.array(y, dtype=np.float64)
    y[100] = np.nan
    return y


def _gated():
    """Ten seconds of noise with 0.2 s of it muted to zeros, 2% of the clip."""
    y = np.random.default_rng(3).standard_normal(10 * SR)
    y[5 * SR:5 * SR + SR // 5] = 0.0
    return y


def _clicks():
    y = 1e-4 * np.random.default_rng(7).standard_normal(5 * SR)
    y[::2000], y[1000::2000] = 1.0, -1.0
    return y


#: malformed input: name -> (function, args, kwargs)
REFUSED = {
    'lsd: two channels': (kudio.lsd, (STEREO, STEREO, SR), {}),
    'lsd: not finite': (kudio.lsd, (CLIP, _corrupt(CLIP), SR), {}),
    'lsd: no rate': (kudio.lsd, (CLIP, CLIP, 0), {}),
    'lsd: scale': (kudio.lsd, (CLIP, CLIP, SR), {'scale': 'dB'}),
    'lsd: hop': (kudio.lsd, (CLIP, CLIP, SR), {'hop_length': 0}),
    'lsd: eps': (kudio.lsd, (CLIP, CLIP, SR), {'eps': 0.0}),
    'lsd: frame': (kudio.lsd, (CLIP, CLIP, SR), {'n_fft': 1}),
    'lsd: lengths': (kudio.lsd, (CLIP, CLIP[:SR // 4], SR), {}),
    'composite: rate': (kudio.composite, (CLIP, CLIP, 44100), {'pesq': 2.0}),
    'composite: two channels': (kudio.composite, (STEREO, STEREO, SR), {'pesq': 2.0}),
    'composite: not finite': (kudio.composite, (CLIP, _corrupt(CLIP), SR), {'pesq': 2.0}),
    'composite: pesq': (kudio.composite, (CLIP, CLIP, SR), {'pesq': 45.0}),
    'composite: lengths': (kudio.composite, (CLIP, CLIP[:SR // 4], SR), {'pesq': 2.0}),
    'composite: scale': (kudio.composite, (CLIP, CLIP, SR), {'scale': 'mos'}),
    'composite: no raw wideband': (kudio.composite, (CLIP, CLIP, SR),
                                   {'mode': 'wb', 'scale': 'raw'}),
    'srmr: rate': (kudio.srmr, (CLIP, 44100), {}),
    'srmr: two channels': (kudio.srmr, (STEREO, SR), {}),
    'srmr: not finite': (kudio.srmr, (_corrupt(CLIP), SR), {}),
    'srmr: bands': (kudio.srmr, (CLIP, SR), {'min_cf': 16.0}),
    'wada_snr: no rate': (kudio.wada_snr, (CLIP, 0), {}),
    'wada_snr: two channels': (kudio.wada_snr, (STEREO, SR), {}),
    'wada_snr: not finite': (kudio.wada_snr, (_corrupt(CLIP), SR), {}),
}
for _name in QUALITY:
    _fn = getattr(kudio, _name)
    REFUSED.update({
        f'{_name}: rate': (_fn, (CLIP, CLIP, 44100), {}),
        f'{_name}: two channels': (_fn, (STEREO, STEREO, SR), {}),
        f'{_name}: not finite': (_fn, (CLIP, _corrupt(CLIP), SR), {}),
        f'{_name}: lengths': (_fn, (CLIP, CLIP[:SR // 4], SR), {}),
    })
for _fn, _rate in ((kudio.ncm, 44100), (kudio.csii, 4000)):
    REFUSED.update({
        f'{_fn.__name__}: rate': (_fn, (CLIP, CLIP, _rate), {}),
        f'{_fn.__name__}: two channels': (_fn, (STEREO, STEREO, SR), {}),
        f'{_fn.__name__}: not finite': (_fn, (CLIP, _corrupt(CLIP), SR), {}),
        f'{_fn.__name__}: lengths': (_fn, (CLIP, CLIP[:SR // 4], SR), {}),
    })
for _fn in (kudio.siib, kudio.siib_gauss):
    REFUSED.update({
        f'{_fn.__name__}: rate': (_fn, (CLIP, CLIP, 16000.5), {}),
        f'{_fn.__name__}: rate in kHz': (_fn, (CLIP, CLIP, 16), {}),
        f'{_fn.__name__}: two channels': (_fn, (STEREO, STEREO, SR), {}),
        f'{_fn.__name__}: not finite': (_fn, (CLIP, _corrupt(CLIP), SR), {}),
        f'{_fn.__name__}: lengths': (_fn, (CLIP, CLIP[:SR // 4], SR), {}),
    })
for _fn in (kudio.snr, kudio.si_sdr, kudio.segmental_snr, kudio.sdi):
    REFUSED.update({
        f'{_fn.__name__}: two channels': (_fn, (STEREO, STEREO), {}),
        f'{_fn.__name__}: not finite': (_fn, (CLIP, _corrupt(CLIP)), {}),
        f'{_fn.__name__}: lengths': (_fn, (CLIP, CLIP[:SR // 4]), {}),
    })
REFUSED.update({
    'segmental_snr: frame': (kudio.segmental_snr, (CLIP, CLIP), {'frame_length': 0}),
    'segmental_snr: hop': (kudio.segmental_snr, (CLIP, CLIP), {'hop_length': 0}),
    'segmental_snr: hop past the frame': (kudio.segmental_snr, (CLIP, CLIP),
                                          {'frame_length': 256, 'hop_length': 512}),
    'pesq: rate': (kudio.pesq, (CLIP, CLIP, 44100), {}),
    'pesq: mode': (kudio.pesq, (CLIP, CLIP, SR), {'mode': 'x'}),
    'pesq: no raw wideband': (kudio.pesq, (CLIP, CLIP, SR), {'mode': 'wb', 'scale': 'raw'}),
    'score: two channels': (kudio.score, (STEREO, STEREO, SR, ['snr']), {}),
    'score: not finite': (kudio.score, (CLIP, _corrupt(CLIP), SR, ['snr']), {}),
    'score: lengths': (kudio.score, (CLIP, CLIP[:SR // 4], SR, ['snr']), {}),
    'eval_metrics: not finite': (kudio.eval_metrics, (SR, _corrupt(CLIP), CLIP), {}),
    'si_sdr_improvement: two channels': (kudio.si_sdr_improvement, (CLIP, STEREO, CLIP), {}),
    'si_sdr_improvement: not finite': (kudio.si_sdr_improvement,
                                       (CLIP, CLIP, _corrupt(CLIP)), {}),
    'si_sdr_improvement: lengths': (kudio.si_sdr_improvement,
                                    (CLIP, CLIP, CLIP[:SR // 4]), {}),
    'pit_si_sdr: samples by sources': (kudio.pit_si_sdr, (STEREO, STEREO), {}),
    'pit_si_sdr: sources and estimates': (kudio.pit_si_sdr, (STEREO.T, CLIP), {}),
    'pit_si_sdr: not finite': (kudio.pit_si_sdr, (CLIP, _corrupt(CLIP)), {}),
})
REFUSED.update({
    'wer: not text': (kudio.wer, (3, 'a'), {}),
    'cer: unpaired': (kudio.cer, (['a b', 'c'], ['a b']), {}),
    'gross_pitch_error: frames': (kudio.gross_pitch_error, (np.ones(3), np.ones(4)), {}),
    'f0_rmse: negative': (kudio.f0_rmse, (np.full(3, -1.0), np.ones(3)), {}),
    'eer: not finite': (kudio.eer, (np.r_[np.nan, 0.2], np.r_[0, 1]), {}),
    'eer: labels': (kudio.eer, (np.r_[0.1, 0.2], np.r_[-1, 1]), {}),
    'min_dcf: prior': (kudio.min_dcf, (np.r_[0.1, 0.2], np.r_[0, 1]), {'p_target': 1.5}),
    'cllr: not a number': (kudio.cllr, (np.r_[np.nan, 0.2], np.r_[0, 1]), {}),
    'min_cllr: labels': (kudio.min_cllr, (np.r_[0.1, 0.2], np.r_[-1, 1]), {}),
    'act_dcf: prior': (kudio.act_dcf, (np.r_[0.1, 0.2], np.r_[0, 1]), {'p_target': 0.0}),
    'c_primary: unpaired': (kudio.c_primary, (np.r_[0.1, 0.2, 0.3], np.r_[0, 1]), {}),
    'der: a turn backwards': (kudio.der, ([(2.0, 1.0, 'A')], []), {}),
    'mcd: rate': (kudio.mcd, (CLIP, CLIP, 11025), {}),
    'mcd: lengths': (kudio.mcd, (CLIP, CLIP[:SR // 4], SR), {}),
    'jer: not turns': (kudio.jer, ('A 0 1', []), {}),
    'detection_error_rate: collar': (kudio.detection_error_rate, ([(0.0, 1.0, 'A')], []),
                                     {'collar': -1.0}),
})

#: well formed, but nothing to score: name -> (function, args, kwargs)
NAN = {
    'lsd: empty': (kudio.lsd, (np.zeros(0), np.zeros(0), SR), {}),
    'lsd: shorter than a frame': (kudio.lsd, (CLIP[:100], CLIP[:100], SR), {}),
    'lsd: silent reference': (kudio.lsd, (np.zeros(SR), CLIP[:SR], SR), {}),
    'composite: shorter than a frame': (kudio.composite, (CLIP[:100], CLIP[:100], SR),
                                        {'pesq': 2.0}),
    'composite: silent reference': (kudio.composite, (np.zeros(SR), CLIP[:SR], SR),
                                    {'pesq': 2.0}),
    'srmr: empty': (kudio.srmr, (np.zeros(0), SR), {}),
    'srmr: silent': (kudio.srmr, (np.zeros(SR), SR), {}),
    'srmr: shorter than a window': (kudio.srmr, (CLIP[:2000], SR), {'drop_silence': False}),
    'wada_snr: empty': (kudio.wada_snr, (np.zeros(0), SR), {}),
    'wada_snr: all zeros': (kudio.wada_snr, (np.zeros(SR), SR), {}),
    'wada_snr: one value': (kudio.wada_snr, (np.full(SR, 0.25), SR), {}),
    'wada_snr: gated': (kudio.wada_snr, (_gated(), SR), {}),
    'wada_snr: shorter than a second': (kudio.wada_snr, (CLIP[:SR // 2], SR), {}),
    'wada_snr: a hum': (kudio.wada_snr,
                        (np.sin(2 * np.pi * 50 * np.arange(5 * SR) / SR + 0.3), SR), {}),
    'wada_snr: clicks': (kudio.wada_snr, (_clicks(), SR), {}),
}
for _name in QUALITY:
    _fn = getattr(kudio, _name)
    NAN.update({
        f'{_name}: shorter than a frame': (_fn, (CLIP[:100], CLIP[:100], SR), {}),
        f'{_name}: silent reference': (_fn, (np.zeros(SR), CLIP[:SR], SR), {}),
    })
for _name in ('llr', 'itakura_saito', 'cepstral_distance', 'wss'):
    NAN[f'{_name}: too loud'] = (getattr(kudio, _name),
                                 (CLIP * 1e160, CLIP * 0.9e160, SR), {})
for _fn in (kudio.ncm, kudio.csii):
    NAN.update({
        f'{_fn.__name__}: empty': (_fn, (np.zeros(0), np.zeros(0), SR), {}),
        f'{_fn.__name__}: shorter than a frame': (_fn, (CLIP[:100], CLIP[:100], SR), {}),
        f'{_fn.__name__}: silent reference': (_fn, (np.zeros(SR), CLIP[:SR], SR), {}),
    })
# SIIB needs 5.45 s of speech, which of these signals only SPEECH (6 s) holds
_GAPPED = np.concatenate([CLIP, np.zeros(4 * SR)])
for _fn in (kudio.siib, kudio.siib_gauss):
    NAN.update({
        f'{_fn.__name__}: empty': (_fn, (np.zeros(0), np.zeros(0), SR), {}),
        f'{_fn.__name__}: shorter than it needs': (_fn, (CLIP, CLIP, SR), {}),
        f'{_fn.__name__}: too little speech': (_fn, (_GAPPED, _GAPPED, SR), {}),
        f'{_fn.__name__}: silent reference': (_fn, (np.zeros_like(SPEECH), SPEECH, SR), {}),
        f'{_fn.__name__}: one value': (_fn, (np.full_like(SPEECH, 0.25), SPEECH, SR), {}),
        f'{_fn.__name__}: too loud': (_fn, (SPEECH, SPEECH * 1e160, SR), {}),
    })
for _fn in (kudio.snr, kudio.si_sdr, kudio.segmental_snr, kudio.sdi):
    NAN.update({
        f'{_fn.__name__}: empty': (_fn, (np.zeros(0), np.zeros(0)), {}),
        f'{_fn.__name__}: silent reference': (_fn, (np.zeros(SR), CLIP[:SR]), {}),
        f'{_fn.__name__}: too loud': (_fn, (CLIP * 1e160, CLIP * 0.9e160), {}),
    })
NAN['lsd: too loud'] = (kudio.lsd, (CLIP * 1e160, CLIP * 0.9e160, SR), {})
NAN.update({
    'si_sdr: constant reference': (kudio.si_sdr, (np.full(SR, 0.25), CLIP[:SR]), {}),
    'si_sdr: one sample': (kudio.si_sdr, (CLIP[SR:SR + 1], CLIP[SR:SR + 1]), {}),
    'segmental_snr: shorter than a frame': (kudio.segmental_snr, (CLIP[:100], CLIP[:100]), {}),
    'score: empty': (kudio.score, (np.zeros(0), np.zeros(0), SR), {}),
    'si_sdr_improvement: silent reference': (kudio.si_sdr_improvement,
                                             (np.zeros(SR), CLIP[:SR], CLIP[:SR]), {}),
    'si_sdr_improvement: too loud': (kudio.si_sdr_improvement,
                                     (CLIP * 1e160, CLIP * 0.9e160, CLIP * 1e160), {}),
    'pit_si_sdr: empty': (kudio.pit_si_sdr, (np.zeros(0), np.zeros(0)), {}),
    'pit_si_sdr: a silent reference': (kudio.pit_si_sdr,
                                       (np.stack([CLIP, 0 * CLIP]), STEREO.T), {}),
})
NAN.update({
    'wer: empty reference': (kudio.wer, ('', 'a b'), {}),
    'cer: empty reference': (kudio.cer, ('', 'ab'), {}),
    'f0_rmse: empty': (kudio.f0_rmse, (np.zeros(0), np.zeros(0)), {}),
    'gross_pitch_error: nothing voiced': (kudio.gross_pitch_error,
                                          (np.zeros(10), np.zeros(10)), {}),
    'eer: empty': (kudio.eer, (np.zeros(0), np.zeros(0)), {}),
    'eer: one class': (kudio.eer, (np.r_[0.1, 0.2], np.r_[1, 1]), {}),
    'min_dcf: one class': (kudio.min_dcf, (np.r_[0.1, 0.2], np.r_[0, 0]), {'p_target': 0.01}),
    'cllr: one class': (kudio.cllr, (np.r_[0.1, 0.2], np.r_[1, 1]), {}),
    'min_cllr: empty': (kudio.min_cllr, (np.zeros(0), np.zeros(0)), {}),
    'act_dcf: one class': (kudio.act_dcf, (np.r_[0.1, 0.2], np.r_[0, 0]), {'p_target': 0.01}),
    'c_primary: one class': (kudio.c_primary, (np.r_[0.1, 0.2], np.r_[0, 0]), {}),
    'der: no reference speech': (kudio.der, ([], [(0.0, 1.0, 'x')]), {}),
    'mcd: shorter than a frame': (kudio.mcd, (CLIP[:500], CLIP[:500], SR), {}),
    'mcd: too loud': (kudio.mcd, (CLIP * 1e160, CLIP * 0.9e160, SR), {}),
    'jer: no reference speech': (kudio.jer, ([], [(0.0, 1.0, 'x')]), {}),
})

#: KudioStudio's Metrics panel gives a missing figure a short reason, from
#: the first of these phrases the warning holds, tried in this order
PANEL_PHRASES = ("reference is digital silence", "ref is digital silence",
                 "deg is digital silence throughout", "digital silence", "needs at least",
                 "no more peaked than Gaussian", "sparser than", "is silent",
                 "samples are zero", "empty")

#: the phrase the panel has to find first in each case's warning; None where
#: it finds none and shows a plain n/a
PANEL_READS = {
    'lsd: empty': 'empty',
    'lsd: shorter than a frame': 'needs at least',
    'lsd: silent reference': 'ref is digital silence',
    'srmr: empty': 'empty',
    'srmr: silent': 'is silent',
    'srmr: shorter than a window': 'needs at least',
    'wada_snr: empty': 'empty',
    'wada_snr: all zeros': 'samples are zero',
    'wada_snr: one value': None,
    'wada_snr: gated': 'digital silence',
    'wada_snr: shorter than a second': 'needs at least',
    'wada_snr: a hum': 'no more peaked than Gaussian',
    'wada_snr: clicks': 'sparser than',
}
for _name in QUALITY:
    PANEL_READS[f'{_name}: shorter than a frame'] = 'needs at least'
    PANEL_READS[f'{_name}: silent reference'] = 'reference is digital silence'
for _name in ('llr', 'itakura_saito', 'cepstral_distance', 'wss'):
    PANEL_READS[f'{_name}: too loud'] = None
for _name in ('snr', 'si_sdr', 'segmental_snr', 'sdi'):
    PANEL_READS[f'{_name}: empty'] = 'empty'
    PANEL_READS[f'{_name}: silent reference'] = 'ref is digital silence'
    PANEL_READS[f'{_name}: too loud'] = None
PANEL_READS['lsd: too loud'] = None
PANEL_READS.update({'si_sdr: constant reference': None, 'si_sdr: one sample': None,
                    'segmental_snr: shorter than a frame': 'needs at least',
                    'score: empty': 'empty'})


def _caught(fn, args, kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = fn(*args, **kwargs)
    return value, caught


@pytest.mark.parametrize("case", REFUSED)
def test_malformed_input_is_refused_by_name(case):
    fn, args, kwargs = REFUSED[case]
    with pytest.raises(ValueError) as refused:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fn(*args, **kwargs)
    assert str(refused.value).startswith(f"{fn.__name__}(): "), str(refused.value)


@pytest.mark.parametrize("fn", [getattr(kudio, n) for n in QUALITY]
                         + [kudio.composite, kudio.srmr], ids=lambda fn: fn.__name__)
def test_a_rate_the_measure_is_not_defined_at_names_the_resampler(fn):
    args = (CLIP, 44100) if fn is kudio.srmr else (CLIP, CLIP, 44100)
    with pytest.raises(ValueError) as refused:
        fn(*args, **({'pesq': 2.0} if fn is kudio.composite else {}))
    assert "kudio.resample(y, 44100, 16000)" in str(refused.value)


@pytest.mark.parametrize("case", NAN)
def test_what_cannot_be_scored_is_nan_with_one_warning_at_the_callers_line(case):
    fn, args, kwargs = NAN[case]
    value, caught = _caught(fn, args, kwargs)
    if fn is kudio.composite:       # field 3 is the pesq passed in
        nan = all(math.isnan(v) for i, v in enumerate(value) if i != 3)
    elif isinstance(value, kudio.PitScore):
        nan = all(math.isnan(v) for v in value.si_sdr)
    elif isinstance(value, tuple):
        nan = all(math.isnan(v) for v in value)
    elif isinstance(value, dict):
        nan = len(value) > 0 and all(math.isnan(v) for v in value.values())
    else:
        nan = math.isnan(value)
    assert nan, value
    assert [w.category for w in caught] == [UserWarning], [str(w.message) for w in caught]
    said = str(caught[0].message)
    assert re.fullmatch(rf"{fn.__name__}\(\): .+; scored nan", said, re.S), said
    assert caught[0].filename == __file__


class _TooShort(RuntimeError):
    """pesq.BufferTooShortError, for the stand-in package below."""


class _NoUtterance(RuntimeError):
    """pesq.NoUtterancesError, for the stand-in package below."""


@pytest.fixture
def stand_in_pesq(monkeypatch):
    """A module where the pesq package would be, with the two error classes
    pesq 0.0.4 exports. Its pesq() raises what is put in ``answer[0]`` if that
    is an exception and returns it otherwise. CI does not build pesq, so this
    is how the rule is held there."""
    answer = [2.5]
    module = types.ModuleType('pesq')
    module.BufferTooShortError = _TooShort
    module.NoUtterancesError = _NoUtterance

    def score(fs, ref, deg, mode):
        if isinstance(answer[0], Exception):
            raise answer[0]
        return answer[0]

    module.pesq = score
    monkeypatch.setitem(sys.modules, 'pesq', module)
    return answer


# two lengths are not malformed for PESQ, which finds the delay itself:
# test_pesq_whole.py holds that
@pytest.mark.parametrize("args", [(STEREO, STEREO), (CLIP, _corrupt(CLIP))],
                         ids=['two channels', 'not finite'])
def test_pesq_refuses_malformed_input_by_name(stand_in_pesq, args):
    with pytest.raises(ValueError) as refused:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kudio.pesq(*args, SR)
    assert str(refused.value).startswith("pesq(): "), str(refused.value)


@pytest.mark.parametrize("case, phrase", [
    ("empty", "the pair is empty"),
    ("silent reference", "ref is digital silence throughout"),
    ("under 0.25 s", "0.25 s"),
    ("no utterance", "no utterance"),
    ("silent output", "deg is digital silence throughout"),
    ("nan returned", "came back nan"),
])
def test_pesq_gives_nan_with_one_warning_where_the_package_cannot_score(
        stand_in_pesq, case, phrase):
    ref, deg, stand_in_pesq[0] = {
        "empty": (np.zeros(0), np.zeros(0), 2.5),
        "silent reference": (np.zeros(SR), CLIP[:SR], 2.5),
        "under 0.25 s": (CLIP[:SR // 5], CLIP[:SR // 5], _TooShort()),
        "no utterance": (CLIP[:SR // 4], CLIP[:SR // 4], _NoUtterance()),
        "silent output": (CLIP, np.zeros_like(CLIP),
                          ValueError("cannot convert float NaN to integer")),
        "nan returned": (CLIP, CLIP, float('nan')),
    }[case]
    value, caught = _caught(kudio.pesq, (ref, deg, SR), {})
    assert math.isnan(value)
    assert [w.category for w in caught] == [UserWarning], [str(w.message) for w in caught]
    said = str(caught[0].message)
    assert re.fullmatch(r"pesq\(\): .+; scored nan", said, re.S), said
    assert phrase in said, said
    assert caught[0].filename == __file__


def test_composite_says_once_why_its_pesq_is_missing(stand_in_pesq):
    """pesq and composite share the reasons, and composite gives one warning
    for everything it could not score, without a second one from pesq."""
    stand_in_pesq[0] = _NoUtterance()
    value, caught = _caught(kudio.composite, (CLIP, CLIP, SR), {})
    assert math.isnan(value.pesq) and math.isnan(value.covl)
    said = [str(w.message) for w in caught]
    assert len(said) == 1 and said[0].startswith("composite(): "), said
    assert "no utterance" in said[0], said


def test_a_warning_from_inside_score_names_the_line_that_called_score():
    """kudio.score reaches each metric through a dispatcher, two calls deeper
    than calling it straight; the warning still names this file."""
    value, caught = _caught(kudio.score, (CLIP[:100], CLIP[:100], SR, ['lsd', 'llr', 'wss']), {})
    assert all(math.isnan(v) for v in value.values())
    assert [w.filename for w in caught] == [__file__] * 3
    assert [str(w.message).split('(')[0] for w in caught] == ['lsd', 'llr', 'wss']


@pytest.mark.parametrize("case", PANEL_READS)
def test_the_phrase_kudiostudio_reads_a_reason_from_comes_first(case):
    fn, args, kwargs = NAN[case]
    _, caught = _caught(fn, args, kwargs)
    said = " ".join(str(w.message) for w in caught)
    first = next((p for p in PANEL_PHRASES if p in said), None)
    assert first == PANEL_READS[case], said
