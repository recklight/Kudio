# -*- coding: utf-8 -*-
"""What every metric takes as a signal and as a rate.

One rule for all of them, so that a pair scores the same through any of them
and through kudio.score:

* A signal is one channel of real numbers: a vector, a column ``(n, 1)`` or a
  row ``(1, n)``. Integer samples are read on their type's full scale.
* int64 and uint64 are refused. No audio file holds them, and on their full
  scale 16-bit samples sit 288 dB down, where the fixed floors of several
  measures decide the score.
* A signal that peaks under one step of 16-bit audio warns once: that is
  where int16 samples land when they are held in an int32 array.
* A rate is one positive number of hertz, a 0-d array or a numpy scalar
  included; each measure then refuses the rates it is not defined at.
"""
from __future__ import annotations

import functools
import importlib.util
import math
import warnings

import numpy as np
import pytest

import kudio

from conftest import make_speech

SR = 16000
REF = make_speech(SR, 6.0, seed=0).astype(np.float64)[SR:3 * SR]
DEG = REF + 0.05 * np.random.default_rng(1).standard_normal(REF.size)
MIX = REF + 0.3 * np.random.default_rng(2).standard_normal(REF.size)

#: name -> what to call with (ref, deg, sr)
PAIRED = {
    'snr': lambda r, d, sr: kudio.snr(r, d),
    'si_sdr': lambda r, d, sr: kudio.si_sdr(r, d),
    'segmental_snr': lambda r, d, sr: kudio.segmental_snr(r, d),
    'sdi': lambda r, d, sr: kudio.sdi(r, d),
    'lsd': kudio.lsd,
    'llr': kudio.llr,
    'itakura_saito': kudio.itakura_saito,
    'cepstral_distance': kudio.cepstral_distance,
    'wss': kudio.wss,
    'fw_segmental_snr': kudio.fw_segmental_snr,
    'composite': lambda r, d, sr: kudio.composite(r, d, sr, pesq=2.0),
    'ncm': kudio.ncm,
    'csii': kudio.csii,
    'siib': kudio.siib,
    'siib_gauss': kudio.siib_gauss,
    'score': lambda r, d, sr: kudio.score(r, d, sr, ['snr', 'lsd', 'llr', 'ncm']),
    'si_sdr_improvement': lambda r, d, sr: kudio.si_sdr_improvement(r, d, MIX),
    'pit_si_sdr': lambda r, d, sr: kudio.pit_si_sdr(r, d),
    'mcd': kudio.mcd,
}
PAIRED.update({
    'sdr_improvement': lambda r, d, sr: kudio.sdr_improvement(r, d, MIX),
    'bss_eval': lambda r, d, sr: kudio.bss_eval(r, d, MIX),
})
#: name -> what to call with (y, sr)
SINGLE = {'srmr': kudio.srmr, 'wada_snr': kudio.wada_snr}
#: the ones that take a rate
RATED = [name for name in PAIRED if name not in (
    'snr', 'si_sdr', 'segmental_snr', 'sdi', 'si_sdr_improvement', 'pit_si_sdr',
    'sdr_improvement', 'bss_eval')]

#: SIIB scores nan with a warning under 5.45 s of speech, which REF does not
#: hold, so where a test needs a number it gets the whole 6 s REF is cut from
LONG = ('siib', 'siib_gauss')
_LONG_REF = make_speech(SR, 6.0, seed=0).astype(np.float64)
_LONG_DEG = _LONG_REF + 0.05 * np.random.default_rng(1).standard_normal(_LONG_REF.size)


def _pair(name):
    """The pair *name* is scored on where the test needs a number."""
    return (_LONG_REF, _LONG_DEG) if name in LONG else (REF, DEG)


def _run(name, ref, deg, sr=SR):
    if name in SINGLE:
        return SINGLE[name](deg, sr)
    return PAIRED[name](ref, deg, sr)


def _numbers(value):
    if isinstance(value, dict):
        return list(value.values())
    if isinstance(value, tuple):
        return [n for v in value if v is not None for n in _numbers(v)]
    return [value]


def _same(a, b):
    return all(x == y or (math.isnan(x) and math.isnan(y))
               for x, y in zip(_numbers(a), _numbers(b)))


def _quiet(name, *args, **kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = _run(name, *args, **kwargs)
    return value, [str(w.message) for w in caught]


@functools.lru_cache(maxsize=None)
def _vector(name):
    """``_quiet(name, *_pair(name))``, the score every other way of giving the pair is held
    to: worked out once a measure, not once a case."""
    return _quiet(name, *_pair(name))


ALL = list(PAIRED) + list(SINGLE)


@pytest.mark.parametrize("name", ALL)
@pytest.mark.parametrize("shape", ['column', 'row'])
def test_one_channel_as_a_column_or_a_row_scores_as_the_vector(name, shape):
    shaped = (lambda y: y[:, None]) if shape == 'column' else (lambda y: y[None, :])
    ref, deg = _pair(name)
    vector, said = _vector(name)
    assert said == []
    value, said = _quiet(name, shaped(ref), shaped(deg))
    assert said == []
    assert _same(value, vector), (value, vector)


NOT_SAMPLES = {
    'complex': DEG + 0.1j,
    'bool': DEG > 0,
    'text': np.array(['0.1'] * DEG.size),
    'datetime': np.zeros(DEG.size, dtype='datetime64[ms]'),
    'None among them': [0.1, None] * (DEG.size // 2),
    'rows of different lengths': [[0.1, 0.2], [0.1]],
    'a single number': np.float64(0.5),
    'masked samples': np.ma.masked_array(DEG, mask=np.arange(DEG.size) == 7),
    'two channels': np.stack([DEG, DEG], axis=1),
}


@pytest.mark.parametrize("name", ALL)
@pytest.mark.parametrize("case", NOT_SAMPLES)
def test_input_that_is_not_one_channel_of_samples_is_refused_by_name(name, case):
    with pytest.raises(ValueError) as refused:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _run(name, REF, NOT_SAMPLES[case])
    said = str(refused.value)
    label = 'y' if name in SINGLE else 'ests' if name in ('pit_si_sdr', 'bss_eval') else 'deg'
    assert said.startswith(f"{name}(): {label} "), said


def test_a_mask_with_nothing_masked_is_read_as_the_samples():
    value = kudio.snr(REF, np.ma.masked_array(DEG))
    assert value == kudio.snr(REF, DEG)


def _int16(ref, deg):
    """int16 samples of a pair, at half of full scale."""
    peak = np.abs(ref).max()
    return (np.round(ref / peak * 0.5 * 32767).astype(np.int16),
            np.round(deg / peak * 0.5 * 32767).astype(np.int16))


REF16, DEG16 = _int16(REF, DEG)


@pytest.mark.parametrize("name", ALL)
@pytest.mark.parametrize("dtype", [np.int64, np.uint64])
def test_64_bit_integers_are_refused_with_the_type_to_use(name, dtype):
    with pytest.raises(ValueError) as refused:
        _run(name, REF16.astype(dtype), DEG16.astype(dtype))
    said = str(refused.value)
    assert said.startswith(f"{name}(): ") and "int16" in said and "288 dB" in said, said


@pytest.mark.parametrize("name", ALL)
@pytest.mark.parametrize("dtype", [np.int16, np.int32, np.uint8, np.float32])
def test_integer_and_float_pcm_score_with_no_warning(name, dtype):
    ref16, deg16 = _int16(*_pair(name))
    if dtype is np.uint8:
        ref = (ref16 // 256 + 128).astype(np.uint8)
        deg = (deg16 // 256 + 128).astype(np.uint8)
    elif dtype is np.float32:
        ref, deg = (y.astype(np.float32) for y in _pair(name))
    else:
        ref, deg = ref16.astype(dtype), deg16.astype(dtype)
        if dtype is np.int32:           # 24-bit PCM, left-justified as soundfile reads it
            ref, deg = ref * 65536, deg * 65536
    value, said = _quiet(name, ref, deg)
    assert said == []
    if name == 'bss_eval':           # one source has no interference: SIR is inf
        value = value._replace(sir=())
    assert all(math.isfinite(v) for v in _numbers(value)), value


#: what makes a pair faint, as a function of the pair
FAINT = {
    'int16 samples held as int32': lambda ref, deg: tuple(y.astype(np.int32)
                                                          for y in _int16(ref, deg)),
    'float scaled by 1e-9': lambda ref, deg: (ref * 1e-9, deg * 1e-9),
}


@pytest.mark.parametrize("name", ALL)
@pytest.mark.parametrize("case", FAINT)
def test_a_signal_under_one_16_bit_step_warns_once_and_still_scores(name, case):
    ref, deg = FAINT[case](*_pair(name))
    value, said = _quiet(name, ref, deg)
    assert len(said) == 1, said
    assert said[0].startswith(f"{name}(): ") and "dBFS" in said[0], said
    assert "16-bit" in said[0] and "int32" in said[0], said
    assert not said[0].endswith("; scored nan"), said


def _dbfs(y16):
    return f"{20 * math.log10(np.abs(y16.astype(np.int64)).max() / 2 ** 31):.1f}"


def test_the_warning_names_the_faint_side_and_its_level():
    _, said = _quiet('snr', REF, DEG16.astype(np.int32))
    assert len(said) == 1 and f"deg peaks at {_dbfs(DEG16)} dBFS" in said[0], said
    _, said = _quiet('snr', REF16.astype(np.int32), DEG16.astype(np.int32))
    assert f"ref and deg peak at {_dbfs(REF16)} and {_dbfs(DEG16)} dBFS" in said[0], said


def test_digital_silence_is_not_called_faint():
    """An output of zeros has its own warnings where a measure needs one; it
    is not a signal under one step."""
    _, said = _quiet('snr', REF, np.zeros_like(DEG))
    assert said == []


RATES = {
    'text': '16000',
    'a bool': True,
    'nan': float('nan'),
    'inf': float('inf'),
    'zero': 0,
    'negative': -16000,
    'a list': [16000],
    'an array of one': np.array([16000]),
    'None': None,
}


@pytest.mark.parametrize("name", RATED + list(SINGLE))
@pytest.mark.parametrize("case", RATES)
def test_a_rate_that_is_not_one_positive_number_is_refused_by_name(name, case):
    with pytest.raises(ValueError) as refused:
        _run(name, REF, DEG, RATES[case])
    said = str(refused.value)
    assert said.startswith(f"{name}(): sr "), said


@pytest.mark.parametrize("name", RATED + list(SINGLE))
@pytest.mark.parametrize("sr", [np.array(SR), np.int64(SR), np.float32(SR), float(SR)],
                         ids=['0-d array', 'numpy int', 'numpy float', 'float'])
def test_a_rate_given_another_way_scores_as_the_int(name, sr):
    ref, deg = _pair(name)
    assert _same(_run(name, ref, deg, sr), _vector(name)[0])


def test_score_passes_the_rate_on_as_it_was_given():
    """score() used to take int(sr) first, so 16000.5 scored llr there while
    llr itself refused it."""
    with pytest.raises(ValueError, match=r"^llr\(\): defined at 8 and 16 kHz only, not 16000.5 Hz"):
        kudio.score(REF, DEG, 16000.5, ['llr'])
    with pytest.raises(ValueError, match=r"^score\(\): sr "):
        kudio.score(REF, DEG, float('nan'), ['snr'])


def test_score_says_once_that_a_pair_is_faint():
    ref, deg = FAINT['int16 samples held as int32'](REF, DEG)
    _, said = _quiet('score', ref, deg)
    assert len(said) == 1 and said[0].startswith("score(): "), said


@pytest.mark.parametrize("name, package", [('stoi', None), ('estoi', None), ('sdr', None)])
def test_a_back_end_that_would_overflow_says_so_instead_of_scoring(name, package):
    """STOI gives 0.0, the worst there is, from 1e154 of full scale,
    and BSS Eval nan from 1e152."""
    if package:
        pytest.importorskip(package)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = kudio.score(REF * 1e160, DEG * 1e160, SR, [name])[name]
    said = [str(w.message) for w in caught]
    assert math.isnan(value)
    assert len(said) == 1 and said[0].startswith(f"{name}(): the pair is too loud"), said


def test_score_refuses_an_unknown_name_by_name():
    with pytest.raises(ValueError, match=r"^score\(\): unknown metric"):
        kudio.score(REF, DEG, SR, ['snrr'])
