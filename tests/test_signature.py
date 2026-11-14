# -*- coding: utf-8 -*-
"""The signature of a result: every setting that changes a metric's number
is in it, as it was resolved; nothing that leaves the number alone is; and
the text reads back into what it was written from."""
import functools
import hashlib
import importlib
import inspect
import json
import math
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

import kudio
from conftest import make_speech
from kudio.core import evaluator
from kudio.core.registry import METRIC_INFO
from kudio.core.signature import parse_signature, signature

# the module, whatever kudio's flat API binds to the name
module = importlib.import_module('kudio.core.signature')

ROOT = Path(__file__).resolve().parents[1]
SR = 16000
VERSION = kudio.__version__

#: the arguments that are what a metric scores rather than how: positional,
#: ahead of every setting
INPUT = {'ref', 'deg', 'y', 'ir', 'reference', 'hypothesis', 'scores', 'labels',
         'llrs', 'refs', 'ests', 'mixture', 'est'}

#: the two metrics without a default for something they need
REQUIRED = {'min_dcf': {'p_target': 0.01}, 'act_dcf': {'p_target': 0.01}}


def _every_metric():
    return {name: dict(REQUIRED.get(name, {})) for name in METRIC_INFO}


def _function(name):
    obj = kudio
    for part in METRIC_INFO[name].function.split('.')[1:]:
        obj = getattr(obj, part)
    return obj


@pytest.fixture
def versions(monkeypatch):
    """Fixed back-end versions, so a signature can be written out in full."""
    known = {'pesq': '0.0.4', 'pystoi': '0.4.1', 'mir_eval': '0.8.2',
             'onnxruntime': '1.19.2', 'librosa': '0.11.0', 'soxr': '1.1.0'}
    monkeypatch.setattr(module, '_installed', lambda package: known[package])


# --- every parameter is accounted for ---------------------------------------------

def _unaccounted(spec, parameters):
    """The parameters of a metric function its spec says nothing about, and
    what the spec says about parameters the function does not have."""
    problems = []
    for key, parameter in parameters.items():
        if key in spec.takes or key in spec.ignores:
            continue
        if key == 'sr':
            if not spec.rate:
                problems.append("sr, though the spec takes no rate")
            continue
        if key in INPUT and parameter.kind is parameter.POSITIONAL_OR_KEYWORD:
            continue
        problems.append(f"{key}: neither recorded nor ruled out")
    if spec.rate and 'sr' not in parameters:
        problems.append("a rate, which the function does not take")
    problems += [f"{key}, which the function does not take"
                 for key in spec.takes + tuple(spec.ignores) if key not in parameters]
    if [key for key in parameters if key in spec.takes] != list(spec.takes):
        problems.append(f"takes {spec.takes} out of the function's order")
    return problems


def test_every_metric_has_a_spec_and_every_spec_a_metric():
    assert {module._own(name) for name in METRIC_INFO} == set(module._SPECS)
    assert {own for own, spec in module._SPECS.items() if spec.takes} == \
        set(module._RESOLVE)


def test_every_parameter_is_recorded_ruled_out_or_the_input():
    """A parameter added to a metric later fails here until someone decides
    whether it changes the number."""
    problems = []
    for name in METRIC_INFO:
        own = module._own(name)
        parameters = inspect.signature(_function(name)).parameters
        problems += [f"{own}: {p}" for p in _unaccounted(module._SPECS[own], parameters)]
    assert not problems, "\n".join(sorted(set(problems)))


def test_the_check_catches_a_new_parameter():
    spec = module._SPECS['pesq']

    def pesq(ref, deg, sr, *, mode='auto', scale='lqo', progress=False):
        pass

    def no_rate(ref, deg, *, mode='auto', scale='lqo'):
        pass

    def reordered(ref, deg, sr, *, scale='lqo', mode='auto'):
        pass

    assert _unaccounted(spec, inspect.signature(pesq).parameters) == \
        ["progress: neither recorded nor ruled out"]
    assert _unaccounted(spec, inspect.signature(no_rate).parameters) == \
        ["a rate, which the function does not take"]
    assert _unaccounted(spec, inspect.signature(reordered).parameters) == \
        ["takes ('mode', 'scale') out of the function's order"]


def test_what_does_not_change_a_number_says_why():
    ruled_out = {(own, key): why for own, spec in module._SPECS.items()
                 for key, why in spec.ignores.items()}
    assert set(ruled_out) == {(m, k) for m in ('stoi', 'estoi', 'sdr')
                              for k in ('metrics', 'pesq_mode', 'pesq_scale')} | \
        {('sdr', 'sr')}
    for why in ruled_out.values():
        assert len(why) > 20 and why.endswith('.'), why


# --- changing a setting changes the number and the signature -----------------------

@functools.lru_cache(maxsize=None)
def _pair():
    clean = make_speech(SR).astype(np.float64)[SR:SR * 5 // 2]
    noise = 0.05 * np.random.default_rng(1).standard_normal(clean.size)
    return clean, 0.6 * (clean + noise)


@functools.lru_cache(maxsize=None)
def _srmr_input():
    """SRMR's input: 8 kHz, with half a second of digital silence first, so
    that cutting the silence out changes something."""
    y = make_speech(8000, seconds=5).astype(np.float64)
    y[:4000] = 0.0
    return y


@functools.lru_cache(maxsize=None)
def _trials():
    """Gaussian scores of 100 target and 400 non-target trials, and the LLRs
    they are for unit-variance classes 2.5 apart."""
    rng = np.random.default_rng(1)
    scores = np.r_[rng.normal(2.5, 1.0, 100), rng.normal(0.0, 1.0, 400)]
    labels = np.r_[np.ones(100, int), np.zeros(400, int)]
    return scores, labels, 2.5 * scores - 2.5 ** 2 / 2


def _turns():
    return [(0.0, 6.0, 'a'), (5.0, 10.0, 'b')], [(0.2, 5.5, 'x'), (5.5, 9.8, 'y')]


def _tracks():
    ref = np.r_[np.full(50, 100.0), np.zeros(10)]
    est = ref.copy()
    est[:10] *= 1.15
    est[10:20] *= 1.3
    est[20:30] *= 1.05
    return ref, est


class _Session:
    """Stands in for an onnxruntime session: the P.835 model's three raw
    outputs, fixed, or the P.808 model's score, the mean of its input."""

    def __init__(self, p808=False):
        self.p808 = p808

    def get_inputs(self):
        class _Input:
            name = 'input_1'
            shape = [1, 900, 120] if self.p808 else [1, 144160]
        return [_Input()]

    def run(self, outputs, feed):
        if self.p808:
            return [np.array([[float(feed['input_1'].mean())]], dtype=np.float32)]
        return [np.array([[2.0, 3.0, 2.5]], dtype=np.float32)]


def _quiet(function, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return function(*args, **kwargs)


def _two_sources():
    clean, _ = _pair()
    other = np.roll(clean, 4000)
    refs = np.vstack([clean, other])
    rng = np.random.default_rng(2)
    ests = np.vstack([0.8 * clean + 0.3 * other, 0.5 * other + 0.4 * clean])
    return refs, ests + 0.01 * rng.standard_normal(ests.shape)


def _ratings(score):
    return score.sig, score.bak, score.ovrl


def _call(name):
    """*name*'s number, or the record of numbers it returns, as a function
    of its options."""
    clean, deg = _pair()
    scores, labels, llrs = _trials()
    ref_turns, hyp_turns = _turns()
    f0_ref, f0_est = _tracks()
    one_second = clean[:SR]
    return {
        'segsnr': lambda **o: kudio.segmental_snr(clean, deg, **o),
        'lsd': lambda **o: kudio.lsd(clean, deg, SR, **o),
        'mcd': lambda **o: kudio.mcd(clean, deg, SR, **o),
        'pesq': lambda **o: kudio.pesq(clean, deg, SR, **o),
        'composite': lambda **o: tuple(kudio.composite(clean, deg, SR, **o)),
        'bss_eval': lambda **o: kudio.bss_eval(*_two_sources(), **o).sdr,
        'srmr': lambda **o: kudio.srmr(_srmr_input(), 8000, **o),
        'dnsmos': lambda **o: _ratings(kudio.dnsmos(one_second, SR, session=_Session(), **o)),
        'dnsmos_p808': lambda **o: kudio.dnsmos_p808(one_second, SR,
                                                     session=_Session(p808=True), **o),
        'loudness': lambda **o: kudio.loudness(clean, SR, **o),
        'true_peak': lambda **o: kudio.true_peak(clean, SR, **o),
        'cer': lambda **o: kudio.cer("the cat sat", "the hat sat", **o),
        'eer': lambda **o: kudio.eer(scores, labels, **o),
        'min_dcf': lambda **o: kudio.min_dcf(scores, labels, **o),
        'act_dcf': lambda **o: kudio.act_dcf(llrs, labels, **o),
        'c_primary': lambda **o: kudio.c_primary(llrs, labels, **o),
        'der': lambda **o: kudio.der(ref_turns, hyp_turns, **o),
        'jer': lambda **o: kudio.jer(ref_turns, hyp_turns, **o),
        'detection_error_rate': lambda **o: kudio.detection_error_rate(ref_turns, hyp_turns,
                                                                      **o),
        'gross_pitch_error': lambda **o: kudio.gross_pitch_error(f0_ref, f0_est, **o),
        'f0_frame_error': lambda **o: kudio.f0_frame_error(f0_ref, f0_est, **o),
        'fine_pitch_error': lambda **o: kudio.fine_pitch_error(f0_ref, f0_est, **o),
    }[name]


#: (metric, the setting changed, its options before, after, a package the
#: metric needs). Every setting a signature records is changed here at least
#: once, but the DNSMOS model, which test_the_model_is_named_by_its_file
#: changes instead, having no weights to run.
MUTATIONS = [
    ('segsnr', 'frame_length', {}, {'frame_length': 256}, None),
    ('segsnr', 'hop_length', {}, {'hop_length': 128}, None),
    ('lsd', 'n_fft', {}, {'n_fft': 1024}, None),
    ('lsd', 'hop_length', {}, {'hop_length': 128}, None),
    ('lsd', 'scale', {}, {'scale': 'db'}, None),
    ('lsd', 'match_gain', {}, {'match_gain': False}, None),
    ('lsd', 'eps', {}, {'eps': 1e-3}, None),
    ('mcd', 'match_gain', {}, {'match_gain': False}, None),
    ('mcd', 'order', {}, {'order': 13, 'alpha': 0.42}, None),
    ('mcd', 'alpha', {}, {'order': 23, 'alpha': 0.3}, None),
    ('mcd', 'n_fft', {}, {'n_fft': 512}, None),
    ('mcd', 'hop_length', {}, {'hop_length': 128}, None),
    ('pesq', 'mode', {}, {'mode': 'nb'}, 'pesq'),
    ('pesq', 'scale', {'mode': 'nb'}, {'mode': 'nb', 'scale': 'raw'}, 'pesq'),
    ('composite', 'mode', {}, {'mode': 'nb'}, 'pesq'),
    ('composite', 'scale', {}, {'scale': 'raw', 'mode': 'nb'}, 'pesq'),
    ('composite', 'pesq', {}, {'pesq': 2.0}, 'pesq'),
    ('bss_eval', 'permutation', {}, {'permutation': (1, 0)}, 'mir_eval'),
    ('srmr', 'fast', {}, {'fast': True}, None),
    ('srmr', 'norm', {}, {'norm': True}, None),
    ('srmr', 'min_cf', {}, {'min_cf': 3.5}, None),
    ('srmr', 'max_cf', {}, {'max_cf': 100.0}, None),
    ('srmr', 'drop_silence', {}, {'drop_silence': False}, None),
    ('dnsmos', 'polyfit', {}, {'polyfit': False}, None),
    ('dnsmos', 'personalized', {}, {'personalized': True}, None),
    ('dnsmos_p808', 'librosa', {}, {'librosa': '0.9'}, None),
    ('loudness', 'block_size', {}, {'block_size': 0.2}, None),
    ('true_peak', 'oversample', {}, {'oversample': 1}, None),
    ('cer', 'spaces', {}, {'spaces': True}, None),
    ('eer', 'method', {}, {'method': 'rocch'}, None),
    ('min_dcf', 'p_target', {'p_target': 0.05}, {'p_target': 0.2}, None),
    ('min_dcf', 'c_miss', {'p_target': 0.05}, {'p_target': 0.05, 'c_miss': 10.0}, None),
    ('min_dcf', 'c_fa', {'p_target': 0.05}, {'p_target': 0.05, 'c_fa': 10.0}, None),
    ('act_dcf', 'p_target', {'p_target': 0.05}, {'p_target': 0.2}, None),
    ('act_dcf', 'c_miss', {'p_target': 0.05}, {'p_target': 0.05, 'c_miss': 10.0}, None),
    ('act_dcf', 'c_fa', {'p_target': 0.05}, {'p_target': 0.05, 'c_fa': 10.0}, None),
    ('c_primary', 'minimum', {}, {'minimum': True}, None),
    ('der', 'collar', {}, {'collar': 0.25}, None),
    ('der', 'skip_overlap', {}, {'skip_overlap': True}, None),
    ('der', 'uem', {}, {'uem': [(0.0, 8.0)]}, None),
    ('jer', 'uem', {}, {'uem': [(0.0, 8.0)]}, None),
    ('detection_error_rate', 'collar', {}, {'collar': 0.25}, None),
    ('detection_error_rate', 'uem', {}, {'uem': [(0.0, 8.0)]}, None),
    ('gross_pitch_error', 'threshold', {}, {'threshold': 0.1}, None),
    ('f0_frame_error', 'threshold', {}, {'threshold': 0.1}, None),
    ('fine_pitch_error', 'threshold', {}, {'threshold': 0.1}, None),
]

#: settings changed elsewhere than in MUTATIONS, and where
CHANGED_ELSEWHERE = {('dnsmos', 'model_dir'), ('dnsmos', 'session'),
                     ('dnsmos_p808', 'model_dir'), ('dnsmos_p808', 'session')}


def test_every_recorded_setting_is_changed_by_some_case():
    recorded = {(own, key) for own, spec in module._SPECS.items() for key in spec.takes}
    changed = {(module._own(name), key) for name, key, *_ in MUTATIONS}
    assert changed | CHANGED_ELSEWHERE == recorded, recorded ^ (changed | CHANGED_ELSEWHERE)


@pytest.mark.parametrize('name, key, before, after, package', MUTATIONS,
                         ids=[f"{m[0]}-{m[1]}" for m in MUTATIONS])
def test_a_setting_that_moves_the_number_moves_the_signature(name, key, before, after,
                                                             package):
    if package is not None:
        pytest.importorskip(package)
    call = _call(name)
    first, second = _quiet(call, **before), _quiet(call, **after)
    assert np.all(np.isfinite(first)) and np.all(np.isfinite(second))
    assert not np.array_equal(first, second), f"{name}: {key} left the number at {first}"
    signed = (signature({name: before}, SR), signature({name: after}, SR))
    assert signed[0] != signed[1]
    # and the field that changed is the metric's own
    fields = [parse_signature(s)[name] for s in signed]
    assert fields[0] != fields[1]


def test_the_rate_moves_the_number_and_the_signature():
    clean, deg = _pair()
    assert kudio.ncm(clean, deg, 8000) != kudio.ncm(clean, deg, 16000)
    assert signature(['ncm'], 8000) != signature(['ncm'], 16000)
    # and a default that follows the rate is written as what it came to
    assert parse_signature(signature(['lsd'], 44100))['lsd']['n_fft'] == '1411'
    assert parse_signature(signature(['mcd'], 8000))['mcd'] == {
        'match_gain': 'yes', 'order': '13', 'alpha': '0.31', 'n_fft': '1024',
        'hop_length': '256'}
    for rate, factor in ((48000, '4'), (44100, '5'), (22050, '9'), (16000, '12')):
        assert parse_signature(signature(['true_peak'], rate))['true_peak'] == \
            {'oversample': factor}, rate
    for rate, order, alpha in ((22050, '34', '0.45'), (24000, '34', '0.46'),
                               (44100, '39', '0.53'), (48000, '39', '0.55')):
        mcd = parse_signature(signature(['mcd'], rate))['mcd']
        assert (mcd['order'], mcd['alpha']) == (order, alpha), rate


def _at(own, sr):
    """The metric *own* run at *sr* on the test pair, as a call."""
    clean, deg = _pair()
    pair = {'pesq': lambda: kudio.pesq(clean, deg, sr),
            'stoi': lambda: kudio.score(clean, deg, sr, ['stoi']),
            'estoi': lambda: kudio.score(clean, deg, sr, ['estoi']),
            'lsd': lambda: kudio.lsd(clean, deg, sr),
            'composite': lambda: kudio.composite(clean, deg, sr, pesq=2.0),
            'mcd': lambda: kudio.mcd(clean, deg, sr),
            'dnsmos': lambda: kudio.dnsmos(clean, sr, session=_Session()),
            'dnsmos_p808': lambda: kudio.dnsmos_p808(clean, sr, session=_Session(p808=True))}
    if own in pair:
        return pair[own]
    if own == 'loudness_range':
        # three seconds of short-term loudness at the least, at any of the rates
        return lambda: kudio.loudness_range(np.tile(clean, 8), sr)
    function = getattr(kudio, own)
    if own in ('srmr', 'wada_snr', 'loudness', 'loudness_range', 'true_peak', 'rt60'):
        return lambda: function(clean, sr)
    return lambda: function(clean, deg, sr)


@pytest.mark.parametrize('sr', [4000, 11025, 44100])
def test_the_signature_refuses_the_rates_the_metric_refuses(sr):
    """No signature describes a number the metric would not have given."""
    from kudio.exceptions import FeatureError
    options = {'composite': {'pesq': 2.0}, 'dnsmos': {'session': object()},
               'dnsmos_p808': {'session': object()}}
    for own, spec in module._SPECS.items():
        if not spec.rate:
            continue
        package = {'pesq': 'pesq', 'stoi': 'pystoi', 'estoi': 'pystoi'}.get(own)
        if package is not None and not evaluator._has_module(package):
            continue
        try:
            _quiet(_at(own, sr))
            refused = False
        except (ValueError, FeatureError):
            refused = True
        try:
            signature({own: options.get(own, {})}, sr)
            signed = True
        except ValueError:
            signed = False
        assert signed != refused, f"{own} at {sr} Hz: refused {refused}, signed {signed}"


def test_a_pair_at_another_rate_is_signed_with_its_rate():
    """si_sdr and segmental_snr take no rate, but the same speech at 8 kHz is
    other samples: 512 of them are 64 ms there and 32 ms at 16 kHz."""
    from scipy.signal import resample_poly
    clean, deg = _pair()
    low = resample_poly(clean, 1, 2), resample_poly(deg, 1, 2)
    assert kudio.si_sdr(*low) - kudio.si_sdr(clean, deg) > 3.0
    assert kudio.segmental_snr(*low) != kudio.segmental_snr(clean, deg)
    names = ['si_sdr', 'segsnr', 'sdr']
    assert signature(names, 8000) != signature(names, 16000)
    assert parse_signature(signature(names, 8000))['sr'] == '8000'
    # without a rate there is nothing to write, and nothing is refused
    assert 'sr' not in parse_signature(signature(names))


def test_aligning_the_pair_moves_the_number_and_the_signature():
    clean, _ = _pair()
    late = np.r_[np.zeros(80), clean[:-80]] + 1e-3 * np.random.default_rng(3).standard_normal(
        clean.size)
    lined_up = kudio.align(clean, late, SR)
    assert _quiet(kudio.si_sdr, *lined_up) > _quiet(kudio.si_sdr, clean, late) + 10
    assert parse_signature(signature(['si_sdr'], aligned=True))['pair'] == \
        {'aligned': 'yes', 'cut': 'shorter', 'limit': '0.5'}
    assert signature(['si_sdr'], aligned=True) != signature(['si_sdr'])


def test_the_length_limit_decides_what_a_mean_is_over(monkeypatch):
    """A pair past MAX_LENGTH_MISMATCH is refused, and drops out of a mean
    over files, so the limit is part of how a mean was made."""
    clean, deg = _pair()
    short = deg[:int(0.4 * deg.size)]
    with pytest.raises(ValueError, match=r"past the 50% limit"):
        kudio.snr(clean, short)
    before = signature(['snr'])
    monkeypatch.setattr(evaluator, 'MAX_LENGTH_MISMATCH', 0.9)
    assert math.isfinite(_quiet(kudio.snr, clean, short))
    assert parse_signature(before)['pair']['limit'] == '0.5'
    assert parse_signature(signature(['snr']))['pair']['limit'] == '0.9'
    # PESQ alone is never cut, so no limit applies to it
    assert 'limit' not in parse_signature(signature(['pesq'], SR))['pair']


# --- what does not change the number does not change the signature ----------------

#: (metric, rate, its options with a default written by hand, the same
#: options without it)
BY_HAND = [
    ('lsd', 16000, {'n_fft': 512, 'hop_length': 256, 'scale': 'ln', 'match_gain': True,
                    'eps': 1e-8}, {}),
    ('lsd', 16000, {'n_fft': 512.0}, {}),
    ('segsnr', 16000, {'frame_length': 512, 'hop_length': 256}, {}),
    ('mcd', 16000, {'order': 23, 'alpha': 0.42}, {}),
    ('srmr', 8000, {'max_cf': 128}, {}),
    ('srmr', 8000, {'norm': True, 'max_cf': 30.0}, {'norm': True}),
    ('true_peak', 16000, {'oversample': 12}, {}),
    ('min_dcf', 16000, {'p_target': 0.05, 'c_miss': 1, 'c_fa': np.float64(1.0)},
     {'p_target': 0.05}),
    ('der', 16000, {'collar': 0, 'skip_overlap': np.bool_(False)}, {}),
    ('eer', 16000, {'method': 'linear'}, {}),
]


@pytest.mark.parametrize('name, sr, explicit, implicit', BY_HAND,
                         ids=[f"{case[0]}-{i}" for i, case in enumerate(BY_HAND)])
def test_a_default_given_by_hand_is_the_default(name, sr, explicit, implicit):
    """The number is the same, so the signature is too."""
    assert signature({name: explicit}, sr) == signature({name: implicit}, sr)
    call = _call(name)
    assert _quiet(call, **explicit) == _quiet(call, **implicit)


def test_pesq_auto_is_written_as_the_bandwidth_it_became():
    def variant(sr, **o):
        return parse_signature(signature({'pesq': o}, sr))['pesq']
    assert variant(16000) == variant(16000, mode='wb') == {'mode': 'wb', 'scale': 'lqo'}
    assert variant(8000) == variant(8000, mode='nb') == {'mode': 'nb', 'scale': 'lqo'}
    assert variant(16000, scale='raw') == {'mode': 'nb', 'scale': 'raw'}
    assert signature(['pesq'], 16000, pesq_mode='auto') == signature(['pesq'], 16000,
                                                                     pesq_mode='wb')
    pytest.importorskip('pesq')
    clean, deg = _pair()
    assert kudio.pesq(clean, deg, SR) == kudio.pesq(clean, deg, SR, mode='wb')


def test_the_composite_writes_the_pesq_that_went_in():
    def field(sr, **o):
        return parse_signature(signature({'csig': o}, sr))['csig']
    assert field(16000) == {'mode': 'wb', 'scale': 'lqo', 'pesq': 'computed'}
    assert field(8000) == {'mode': 'nb', 'scale': 'raw', 'pesq': 'computed'}
    assert field(16000, mode='nb', scale='raw') == {'mode': 'nb', 'scale': 'raw',
                                                    'pesq': 'computed'}
    # mode='nb' alone at 16 kHz takes the raw score, as composite() does
    assert field(16000, mode='nb') == {'mode': 'nb', 'scale': 'raw', 'pesq': 'computed'}
    assert field(16000, mode='nb', scale='lqo') == {'mode': 'nb', 'scale': 'lqo',
                                                    'pesq': 'computed'}
    assert field(8000, scale='lqo') == {'mode': 'nb', 'scale': 'lqo', 'pesq': 'computed'}
    given = parse_signature(signature({'csig': {'pesq': 2.31}}, 16000))
    assert given['csig']['pesq'] == 'given'
    assert 'pkg' not in given and given['pair'] == {'aligned': 'no', 'cut': 'shorter',
                                                    'limit': '0.5'}


def test_score_s_own_arguments_leave_stoi_and_the_composite_alone():
    """score() hands pesq_mode to its pesq entry and to nothing else."""
    assert signature({'stoi': {'pesq_mode': 'nb', 'pesq_scale': 'raw'}}, SR) == \
        signature(['stoi'], SR)
    assert signature(['csig', 'covl'], SR, pesq_mode='nb') == signature(['csig', 'covl'], SR)
    clean, deg = _pair()
    if evaluator._has_module('pystoi'):
        assert kudio.score(clean, deg, SR, ['stoi'], pesq_mode='nb', pesq_scale='raw') == \
            kudio.score(clean, deg, SR, ['stoi'])
    if evaluator._has_module('pesq'):
        assert kudio.score(clean, deg, SR, ['csig'], pesq_mode='nb') == \
            kudio.score(clean, deg, SR, ['csig'])


def test_sdr_needs_no_rate_and_writes_one_given():
    """mir_eval's BSS Eval takes no rate: the same samples score the same
    whatever rate score() is told. A rate that is given is written, for the
    pair."""
    assert 'sr' not in parse_signature(signature(['sdr']))
    assert parse_signature(signature(['sdr'], 16000))['sr'] == '16000'
    assert signature(['sdr'], 8000) != signature(['sdr'], 16000)
    if evaluator._has_module('mir_eval'):
        clean, deg = _pair()
        assert kudio.score(clean, deg, 8000, ['sdr']) == kudio.score(clean, deg, 16000, ['sdr'])


def test_order_aliases_and_repeats_do_not_change_it():
    assert signature(['stoi', 'pesq', 'snr'], SR) == signature(['snr', 'pesq', 'stoi'], SR) \
        == signature({'snr', 'stoi', 'pesq'}, SR) == signature(['pesq', 'snr', 'stoi', 'snr'],
                                                               SR)
    assert signature(dict.fromkeys(['pesq', 'stoi']), SR) == signature(['pesq', 'stoi'], SR)
    # one metric under score()'s name and its function's: one setting, two fields
    fields = parse_signature(signature(['segmental_snr', 'segsnr'], SR))
    assert fields['segsnr'] == fields['segmental_snr']
    assert list(fields).index('segsnr') < list(fields).index('segmental_snr')


# --- what it says --------------------------------------------------------------------

def test_pesq_and_stoi_at_16_khz(versions):
    assert signature(['pesq', 'stoi'], 16000) == (
        f"kudio:{VERSION}|sr:16000|pair:aligned=no,cut=shorter,limit=0.5,whole=pesq|"
        f"pesq:mode=wb,scale=lqo|stoi|pkg:pesq=0.0.4,pystoi=0.4.1")
    assert signature(['stoi', 'pesq'], 8000, pesq_mode='nb', pesq_scale='raw', n=824,
                     ci=dict(kind='paired_ci', method='bca', confidence=0.95,
                             n_resamples=10000, seed=0, groups='speaker')) == (
        f"kudio:{VERSION}|sr:8000|pair:aligned=no,cut=shorter,limit=0.5,whole=pesq|"
        f"pesq:mode=nb,scale=raw|stoi|pkg:pesq=0.0.4,pystoi=0.4.1|n:824|"
        f"ci:kind=paired_ci,method=bca,confidence=0.95,n_resamples=10000,seed=0,"
        f"groups=speaker")


def test_each_back_end_is_named_and_the_list_is_sorted(versions):
    def pkg(names, sr=SR):
        return parse_signature(signature(names, sr)).get('pkg')
    assert pkg(['estoi']) == {'pystoi': '0.4.1'}
    assert pkg(['sdr_improvement']) == pkg(['bss_eval']) == {'mir_eval': '0.8.2'}
    assert pkg(['csig']) == {'pesq': '0.0.4'}
    assert pkg({'dnsmos_p808': {'session': object()}}) == {'onnxruntime': '1.19.2'}
    assert pkg({'dnsmos_p808': {'session': object()}}, 48000) == \
        {'librosa': '0.11.0', 'onnxruntime': '1.19.2', 'soxr': '1.1.0'}
    for names in (['snr', 'lsd', 'mcd', 'srmr', 'eer'], ['wada_snr', 'loudness']):
        assert pkg(names) is None
    every = pkg({**_every_metric(), 'dnsmos': {'session': object()},
                 'dnsmos_p808': {'session': object()}})
    assert list(every) == ['mir_eval', 'onnxruntime', 'pesq', 'pystoi']
    resampled = pkg({'stoi': {}, 'sdr': {}, 'dnsmos': {'session': object()}}, 48000)
    assert list(resampled) == ['librosa', 'mir_eval', 'onnxruntime', 'pystoi', 'soxr']


def test_settings_are_written_to_the_last_digit(versions):
    """Rounding a setting to print it would let two settings sign alike."""
    fields = parse_signature(signature({'lsd': {'eps': 1e-12},
                                        'min_dcf': {'p_target': 1e-05},
                                        'act_dcf': {'p_target': 0.123456789012345},
                                        'gross_pitch_error': {'threshold': 1 / 3}}, SR))
    assert fields['lsd']['eps'] == '1e-12'
    assert fields['min_dcf']['p_target'] == '1e-05'
    assert fields['act_dcf']['p_target'] == '0.123456789012345'
    assert fields['gross_pitch_error']['threshold'] == '0.3333333333333333'
    assert signature({'act_dcf': {'p_target': 0.1234567890123}}) != \
        signature({'act_dcf': {'p_target': 0.1234567890124}})
    # -0.0 is the number 0.0
    assert signature({'der': {'collar': -0.0}}) == signature(['der'])


def test_bss_eval_s_permutation_is_signed_without_mir_eval(versions):
    fields = [parse_signature(signature({'bss_eval': o}))
              for o in ({}, {'permutation': (1, 0)})]
    assert fields[0]['bss_eval'] == {'permutation': 'best_sir'}
    assert fields[1]['bss_eval'] == {'permutation': 'given'}
    assert fields[0]['pkg'] == fields[1]['pkg'] == {'mir_eval': '0.8.2'}


def test_eer_and_min_dcf_need_no_rate(versions):
    assert signature({'eer': {}, 'min_dcf': {'p_target': 0.01}}, 16000) == (
        f"kudio:{VERSION}|eer:method=linear|min_dcf:p_target=0.01,c_miss=1.0,c_fa=1.0")


def test_the_pair_field():
    def pair(names, **o):
        return parse_signature(signature(names, SR, **o)).get('pair')
    cut = {'aligned': 'no', 'cut': 'shorter', 'limit': '0.5'}
    assert pair(['pesq']) == {'aligned': 'no', 'whole': 'pesq'}
    assert pair(['snr', 'pesq']) == dict(cut, whole='pesq')
    assert pair({'mcd': {'match_gain': False}}) == {'aligned': 'no', 'whole': 'mcd'}
    assert pair({'mcd': {'match_gain': False}, 'csig': {}}) == dict(cut, whole='mcd+pesq')
    assert pair(['bss_eval']) == cut
    assert pair(['dnsmos', 'eer'], aligned=True) is None
    for name in ('srmr', 'wada_snr', 'loudness'):
        fields = parse_signature(signature([name], 8000, aligned=True))
        assert 'pair' not in fields and fields['sr'] == '8000', name


def test_no_rate_no_pair_for_metrics_that_take_neither():
    fields = parse_signature(signature(['wer', 'cer', 'eer', 'f0_rmse'], 16000))
    assert list(fields) == ['kudio', 'wer', 'cer', 'eer', 'f0_rmse']
    assert fields['wer'] == {} and fields['cer'] == {'spaces': 'no'}


def test_every_metric_signs_in_ascii_and_reads_back():
    text = signature(_every_metric(), 16000, aligned=True, n=12,
                     ci=dict(kind='bootstrap_ci', method='percentile', confidence=0.9,
                             n_resamples=2000, seed=None))
    assert text.isascii() and text == text.strip() and ' ' not in text
    fields = parse_signature(text)
    names = [k for k in fields if k not in module._BEFORE + module._AFTER]
    assert names == list(METRIC_INFO)
    assert list(fields)[:3] == ['kudio', 'sr', 'pair'] and list(fields)[-3:] == \
        ['pkg', 'n', 'ci']
    assert fields['ci']['seed'] == 'none' and fields['ci']['groups'] == 'none'
    assert module._format(fields) == text
    assert parse_signature(module._format(fields)) == fields


def test_field_names_and_metric_names_do_not_collide():
    assert not set(module._BEFORE + module._AFTER) & set(METRIC_INFO)
    assert all(module._KEY.fullmatch(name) for name in METRIC_INFO)


# --- back-ends and models --------------------------------------------------------------

def test_a_package_is_named_with_its_installed_version_or_missing(monkeypatch):
    from importlib import metadata
    for package in ('pesq', 'pystoi', 'mir_eval', 'librosa', 'soxr'):
        try:
            want = metadata.version(package)
        except metadata.PackageNotFoundError:
            want = 'missing'
        assert module._installed(package) == want

    def version(dist):
        if dist == 'onnxruntime-gpu':
            return '1.17.1'
        raise metadata.PackageNotFoundError(dist)
    monkeypatch.setattr(metadata, 'version', version)
    assert module._installed('onnxruntime') == '1.17.1'
    assert module._installed('pesq') == 'missing'
    assert parse_signature(signature(['sdr'], SR))['pkg'] == {'mir_eval': 'missing'}
    for dist in ('onnxruntime-openvino', 'onnxruntime-qnn'):
        monkeypatch.setattr(metadata, 'version', lambda d, dist=dist: (
            '1.20.0' if d == dist else version('none')))
        assert module._installed('onnxruntime') == '1.20.0', dist


def test_dnsmos_names_the_packages_that_resample_its_input(versions):
    at = {sr: parse_signature(signature({'dnsmos': {'session': object()}}, sr))['pkg']
          for sr in (16000, 48000)}
    assert at[16000] == {'onnxruntime': '1.19.2'}
    assert at[48000] == {'librosa': '0.11.0', 'onnxruntime': '1.19.2', 'soxr': '1.1.0'}


def _sha8(data):
    return hashlib.sha256(data).hexdigest()[:8]


@pytest.fixture
def models(tmp_path, monkeypatch):
    """A DNS-Challenge layout of stand-in model files."""
    from kudio.core.dnsmos import ENV_VAR
    monkeypatch.delenv(ENV_VAR, raising=False)
    files = {('DNSMOS', 'sig_bak_ovr.onnx'): b'p835', ('pDNSMOS', 'sig_bak_ovr.onnx'): b'pp835',
             ('DNSMOS', 'model_v8.onnx'): b'p808'}
    for (folder, name), data in files.items():
        (tmp_path / folder).mkdir(exist_ok=True)
        (tmp_path / folder / name).write_bytes(data)
    return tmp_path


def test_the_model_is_named_by_its_file(models):
    def model(name, **o):
        return parse_signature(signature({name: o}, SR))[name]['model']
    assert model('dnsmos', model_dir=models) == f"sig_bak_ovr.onnx@{_sha8(b'p835')}"
    assert model('dnsmos', model_dir=models, personalized=True) == \
        f"sig_bak_ovr.onnx@{_sha8(b'pp835')}"
    assert model('dnsmos_p808', model_dir=models / 'DNSMOS') == \
        f"model_v8.onnx@{_sha8(b'p808')}"
    assert model('dnsmos_p808', model_dir=models / 'DNSMOS' / 'model_v8.onnx') == \
        f"model_v8.onnx@{_sha8(b'p808')}"
    (models / 'DNSMOS' / 'sig_bak_ovr.onnx').write_bytes(b'other weights')
    assert model('dnsmos', model_dir=models) == f"sig_bak_ovr.onnx@{_sha8(b'other weights')}"


def test_a_model_kudio_cannot_find(models, monkeypatch):
    from kudio.core.dnsmos import ENV_VAR
    # nowhere to look: the signature can still be written, elsewhere
    assert parse_signature(signature(['dnsmos'], SR))['dnsmos']['model'] == 'unknown'
    monkeypatch.setenv(ENV_VAR, str(models))
    assert parse_signature(signature(['dnsmos'], SR))['dnsmos']['model'] == \
        f"sig_bak_ovr.onnx@{_sha8(b'p835')}"
    monkeypatch.delenv(ENV_VAR)
    empty = models / 'empty'
    empty.mkdir()
    with pytest.raises(ValueError, match=r"^signature\(\): dnsmos: sig_bak_ovr\.onnx not "
                                         r"found"):
        signature({'dnsmos': {'model_dir': empty}}, SR)


def test_with_a_session_only_a_model_file_names_the_model(models, monkeypatch):
    """dnsmos reads nothing from model_dir when it is given a session, so a
    folder, which holds both P.835 models, says nothing of what the session
    holds, and a folder that is not there is no mistake."""
    from kudio.core.dnsmos import ENV_VAR
    monkeypatch.setenv(ENV_VAR, str(models))

    def model(**o):
        return parse_signature(signature({'dnsmos': dict(o, session=object())},
                                         SR))['dnsmos']['model']
    assert model() == 'unknown'
    assert model(model_dir=models) == model(model_dir=models / 'pDNSMOS') == 'unknown'
    assert model(model_dir=models / 'nowhere') == 'unknown'
    assert model(model_dir=models / 'pDNSMOS' / 'sig_bak_ovr.onnx') == \
        f"sig_bak_ovr.onnx@{_sha8(b'pp835')}"


def _published(coefficients):
    """The digest of a set of polynomials, worked out here a second way."""
    text = ';'.join(key + ':' + ','.join(repr(float(c)) for c in coefficients[key])
                    for key in ('sig', 'bak', 'ovrl'))
    return hashlib.sha256(text.encode()).hexdigest()[:8]


def test_the_polynomials_are_named_by_their_coefficients(models, monkeypatch):
    from kudio.core.dnsmos import POLYFIT, POLYFIT_PERSONALIZED

    def polyfit(**o):
        return parse_signature(signature({'dnsmos': o}, SR))['dnsmos']['polyfit']
    published = polyfit(model_dir=models / 'DNSMOS')
    assert published == _published(POLYFIT) == polyfit(session=object())
    assert polyfit(model_dir=models / 'pDNSMOS', personalized=True) == \
        _published(POLYFIT_PERSONALIZED) != published
    assert polyfit(model_dir=models / 'DNSMOS', polyfit=False) == 'no'
    # a polyfit.json beside the model replaces them, unless there is a session
    replaced = {'sig': [1, 0], 'bak': [1, 0], 'ovrl': [0.5, 1, 0]}
    (models / 'DNSMOS' / 'polyfit.json').write_text(json.dumps(replaced))
    assert polyfit(model_dir=models / 'DNSMOS') == _published(replaced)
    assert polyfit(model_dir=models / 'DNSMOS', session=object()) == published
    (models / 'DNSMOS' / 'polyfit.json').write_text('{"sig": [1, 0]}')
    with pytest.raises(ValueError, match=r"^signature\(\): dnsmos: .*missing coefficients"):
        polyfit(model_dir=models / 'DNSMOS')
    # and kudio's own, changed in place, are no longer the published ones
    monkeypatch.setitem(POLYFIT, 'sig', (-0.08, 1.22, 0.005))
    assert polyfit(session=object()) != published


def test_working_out_a_signature_imports_no_back_end():
    code = (
        "import sys, kudio\n"
        "from kudio.core.signature import signature\n"
        "metrics = {name: {} for name in kudio.METRIC_INFO}\n"
        "metrics['min_dcf'] = metrics['act_dcf'] = {'p_target': 0.01}\n"
        "signature(metrics, 16000)\n"
        "print(sorted(m for m in ('pesq', 'pystoi', 'mir_eval', 'onnxruntime')"
        " if m in sys.modules))\n")
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                         cwd=str(ROOT), timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == '[]'


# --- the interval ----------------------------------------------------------------------

#: an interval's settings, as kudio.paired_ci takes them by default
INTERVAL = dict(kind='paired_ci', method='bca', confidence=0.95, n_resamples=10000, seed=0)


def _ci(**o):
    return parse_signature(signature(['snr'], ci={**INTERVAL, **o}))['ci']


def test_the_interval_takes_the_settings_the_interval_ran_with():
    assert _ci() == {'kind': 'paired_ci', 'method': 'bca', 'confidence': '0.95',
                     'n_resamples': '10000', 'seed': '0', 'groups': 'none'}
    assert _ci(groups='speaker')['groups'] == 'speaker'
    assert _ci(seed=None)['seed'] == 'none'
    assert _ci(seed=np.random.default_rng(0))['seed'] == 'generator'
    assert parse_signature(signature(['snr'], n=40))['n'] == '40'


def test_the_interval_says_which_interval():
    """A mean of per-utterance WERs and the pooled WER over the same
    utterances are two intervals with the same settings."""
    kinds = {_ci(kind=kind)['kind'] for kind in module._INTERVALS}
    assert kinds == {'bootstrap_ci', 'paired_ci', 'rate_ci', 'paired_rate_ci'}
    assert signature(['wer'], ci=dict(INTERVAL, kind='bootstrap_ci')) != \
        signature(['wer'], ci=dict(INTERVAL, kind='rate_ci'))


def test_the_method_is_the_one_the_interval_reports():
    """BCa with no interval to give falls back to the percentile ends, and
    says so in the result; the signature takes what the result says."""
    stats = importlib.import_module('kudio.core.stats')
    # one large value among twenty, at a level BCa cannot reach
    values = np.r_[np.zeros(19), 1000.0]
    result = _quiet(stats.bootstrap_ci, values, confidence=1 - 1e-12, n_resamples=2000)
    assert result.method == 'percentile'
    assert _ci(kind='bootstrap_ci', method=result.method)['method'] == 'percentile'
    assert signature(['snr'], ci=dict(INTERVAL, method=result.method)) != \
        signature(['snr'], ci=INTERVAL)


@pytest.mark.parametrize('labels', [
    ['s1', 's1', 's2', 's3', 's2', 's3', 's4'],
    [1, 1.0, 2, 2, 3, 3],
    np.array([1, '1', 2], dtype=object),
    np.array([3, 3, 1]),
    np.array([[3], [3], [1], [2]]),
    ('a', ('b', 1), ('b', 1)),
])
def test_groups_are_counted_as_the_interval_counts_them(labels):
    stats = importlib.import_module('kudio.core.stats')
    values = np.arange(len(labels), dtype=float)
    result = _quiet(stats.bootstrap_ci, values, groups=labels, n_resamples=200)
    assert _ci(groups=labels)['groups'] == str(result.n_groups)


@pytest.mark.parametrize('ci, match', [
    ({k: v for k, v in INTERVAL.items() if k != 'seed'}, r"ci: \['seed'\] missing"),
    ({k: v for k, v in INTERVAL.items() if k != 'kind'}, r"ci: \['kind'\] missing"),
    (dict(INTERVAL, kind='permutation_test'), r"ci: kind has to be one of"),
    (dict(INTERVAL, level=0.9), r"ci: unknown setting\(s\) \['level'\]; choose from"),
    (dict(INTERVAL, confidence=95), r"ci: confidence has to be above 0 and below 1, got 95"),
    (dict(INTERVAL, n_resamples=0), r"ci: n_resamples has to be a whole number, 1 or more"),
    (dict(INTERVAL, seed=-1), r"ci: seed has to be a whole number, 0 or more"),
    (dict(INTERVAL, method='BCa'), r"ci: method has to be the name of the method"),
    (dict(INTERVAL, groups='by speaker'), r"ci: groups names what the items were grouped by"),
    (dict(INTERVAL, groups=['a', None, 'b']), r"ci: item 1 has None for its group"),
    (dict(INTERVAL, groups=[1.0, float('nan')]), r"ci: item 1 has nan for its group"),
    (dict(INTERVAL, groups=3), r"ci: groups has to be a name or the items' group labels"),
    ([0.95], r"ci has to be a dict"),
])
def test_the_interval_refuses(ci, match):
    with pytest.raises(ValueError, match=r"^signature\(\): " + match):
        signature(['snr'], ci=ci)


# --- what it refuses -------------------------------------------------------------------

@pytest.mark.parametrize('call, match', [
    (lambda: signature(['PESQ', 'pesq', 'nope']),
     r"unknown metric\(s\) \['PESQ', 'nope'\]; choose from \('snr'"),
    (lambda: signature({'lsd': {'nfft': 512}}, SR),
     r"lsd: unknown option\(s\) \['nfft'\]; choose from \('n_fft', 'hop_length'"),
    (lambda: signature({'snr': {'mode': 'nb'}}),
     r"snr: nothing but its input changes it, so it takes no options"),
    (lambda: signature({'lsd': {'ref': [0.0]}}, SR), r"lsd: unknown option\(s\) \['ref'\]"),
    (lambda: signature({'lsd': {'sr': 16000}}, SR), r"lsd: unknown option\(s\) \['sr'\]"),
    (lambda: signature(['pesq', 'stoi']), r"pass sr, the rate pesq, stoi ran at"),
    (lambda: signature(['min_dcf']), r"min_dcf: p_target has no default"),
    (lambda: signature(['pesq'], 44100), r"pesq: PESQ is defined at 8 and 16 kHz"),
    (lambda: signature({'pesq': {'mode': 'wb'}}, 8000), r"pesq: wideband PESQ"),
    (lambda: signature({'pesq': {'mode': 'wb', 'scale': 'raw'}}, SR),
     r"pesq: P\.862\.2 reports no raw score"),
    (lambda: signature({'pesq': {'mode': 'nb'}}, SR, pesq_mode='nb'),
     r"PESQ's mode given twice"),
    (lambda: signature({'csig': {'scale': 'mos'}}, SR), r"csig: scale must be"),
    (lambda: signature({'lsd': {'scale': 'dB'}}, SR), r"lsd: scale has to be one of"),
    (lambda: signature({'lsd': {'hop_length': 1024}}, SR),
     r"lsd: hop_length has to be a whole number, from 1 to 512"),
    (lambda: signature({'mcd': {'order': 13}}, SR), r"mcd: pass order and alpha together"),
    (lambda: signature(['mcd'], 11025), r"mcd: URGENT's MCD has settings for"),
    (lambda: signature({'eer': {'method': 'convex'}}), r"eer: method has to be one of"),
    (lambda: signature({'min_dcf': {'p_target': 1.5}}),
     r"min_dcf: p_target has to be above 0 and below 1"),
    (lambda: signature({'der': {'collar': -0.25}}), r"der: collar has to be from 0"),
    (lambda: signature({'cer': {'spaces': 1}}), r"cer: spaces has to be True or False"),
    (lambda: signature({'dnsmos_p808': {'librosa': '0.10'}}, SR),
     r"dnsmos_p808: librosa has to be one of \('0\.8', '0\.9'\)"),
    (lambda: signature(['snr'], aligned='yes'), r"aligned has to be True or False"),
    (lambda: signature(['snr'], n=0), r"n is how many items were scored"),
    (lambda: signature(['pesq'], 0), r"sr has to be a positive sample rate"),
    # a rate is checked whenever it is given, even where it is not written
    (lambda: signature(['eer'], -1), r"sr has to be a positive sample rate"),
    (lambda: signature(['snr'], 'fast'), r"sr has to be a positive sample rate"),
    # rates the metrics themselves refuse
    (lambda: signature(['srmr'], 44100), r"srmr: defined at 8 and 16 kHz only"),
    (lambda: signature(['ncm'], 44100), r"ncm: NCM is defined at 8 and 16 kHz"),
    (lambda: signature(['llr'], 44100), r"llr: defined at 8 and 16 kHz only"),
    (lambda: signature(['is'], 22050), r"is: defined at 8 and 16 kHz only"),
    (lambda: signature(['wss'], 44100), r"wss: defined at 8 and 16 kHz only"),
    (lambda: signature(['fwsegsnr'], 44100), r"fwsegsnr: defined at 8 and 16 kHz only"),
    (lambda: signature(['csii_high'], 4000), r"csii_high: CSII needs a rate of 8 kHz"),
    (lambda: signature(['siib'], 16000.5), r"siib: SIIB takes the pair to 16 kHz by a "
                                           r"ratio of whole numbers"),
    (lambda: signature(['siib_gauss'], 44100.25), r"siib_gauss: SIIB takes the pair"),
    (lambda: signature(['siib'], 16), r"siib: sr is in hertz, got 16; SIIB refuses a rate "
                                      r"under 4000 Hz"),
    (lambda: signature(['siib_gauss'], 44.1), r"siib_gauss: sr is in hertz, got 44\.1"),
    (lambda: signature([]), r"no metrics given"),
    (lambda: signature(b'pesq'), r"a metric name is text"),
    (lambda: signature([3]), r"metric names have to be strings"),
    (lambda: signature({'pesq': 'nb'}, SR), r"the options of 'pesq' have to be a dict"),
    (lambda: signature(3), r"metrics has to be a name"),
])
def test_it_refuses(call, match):
    with pytest.raises(ValueError, match=r"^signature\(\): " + match):
        call()


# --- reading it back ---------------------------------------------------------------------

def test_parse_reads_what_signature_writes():
    text = f"kudio:{VERSION}|sr:16000|pair:aligned=no,cut=shorter,limit=0.5,whole=pesq|" \
           f"pesq:mode=wb,scale=lqo|stoi|pkg:pesq=0.0.4,pystoi=0.4.1|n:824"
    fields = parse_signature(text)
    assert fields == {'kudio': VERSION, 'sr': '16000',
                      'pair': {'aligned': 'no', 'cut': 'shorter', 'limit': '0.5',
                               'whole': 'pesq'},
                      'pesq': {'mode': 'wb', 'scale': 'lqo'}, 'stoi': {},
                      'pkg': {'pesq': '0.0.4', 'pystoi': '0.4.1'}, 'n': '824'}
    assert module._format(fields) == text
    # broken over lines, as a PDF column breaks it
    assert parse_signature(text.replace('|', '|\n').replace(',', ', ')) == fields


def test_parse_reads_fields_it_does_not_know():
    """A signature from a later kudio, with a metric this one lacks."""
    fields = parse_signature("kudio:9.0.0|sr:16000|newmetric:alpha=1.5e-3,mode=x|other")
    assert fields == {'kudio': '9.0.0', 'sr': '16000',
                      'newmetric': {'alpha': '1.5e-3', 'mode': 'x'}, 'other': {}}


@pytest.mark.parametrize('text, match', [
    ("sr:16000|kudio:4.0.0", r"a kudio signature starts with kudio:<version>"),
    ("BLEU|nrefs:1|case:mixed", r"'BLEU' does not start with a field name"),
    ("kudio:4.0.0|pesq:mode=wb|pesq:mode=nb", r"pesq appears twice"),
    ("kudio:4.0.0|pesq:mode=wb,mode=nb", r"pesq: mode appears twice"),
    ("kudio:4.0.0|pesq:mode=wb,scale", r"pesq: 'scale' is not setting=value"),
    ("kudio:4.0.0|sr:16000;8000", r"sr: '16000;8000' is not a value"),
    ("kudio:4.0.0||snr", r"'' does not start with a field name"),
    ("kudio", r"a kudio signature starts with kudio:<version>"),
    (b"kudio:4.0.0", r"a signature is text"),
])
def test_parse_refuses(text, match):
    with pytest.raises(ValueError, match=r"^parse_signature\(\): " + match):
        parse_signature(text)


def test_the_text_is_ascii_whatever_the_machine_says(monkeypatch, tmp_path):
    """A version or a file name outside the characters a value may hold is
    written with '_' for each of them."""
    from importlib import metadata
    monkeypatch.setattr(metadata, 'version', lambda dist: '1.0 (local, build|3)')
    assert module._installed('pesq') == '1.0__local__build_3_'
    folder = tmp_path / 'm'
    folder.mkdir()
    odd = folder / 'my model,v2.onnx'
    odd.write_bytes(b'x')
    text = signature({'dnsmos': {'model_dir': odd}}, SR)
    assert text.isascii()
    assert parse_signature(text)['dnsmos']['model'] == f"my_model_v2.onnx@{_sha8(b'x')}"


def test_sample_rates_are_written_as_numbers():
    assert parse_signature(signature(['lsd'], 16000.0))['sr'] == '16000'
    assert parse_signature(signature(['lsd'], np.int64(22050)))['sr'] == '22050'
    fields = parse_signature(signature(['lsd'], 22050.5))
    assert (fields['sr'], fields['lsd']['n_fft']) == ('22050.5', '705')


# --- what score_dataset needs to say about a run -------------------------------------

def test_cut_says_how_the_pairs_of_the_whole_run_were_treated():
    """A line that names PESQ alone, from a run that also scored SNR on the
    same pairs, has to say the pairs were held to the length limit."""
    assert parse_signature(signature(['pesq'], SR))['pair'] == {'aligned': 'no',
                                                                'whole': 'pesq'}
    within = parse_signature(signature(['pesq'], SR, cut=True))['pair']
    assert within == parse_signature(signature(['pesq', 'snr'], SR))['pair']
    assert parse_signature(signature(['snr'], SR, cut=False))['pair'] == {'aligned': 'no'}
    with pytest.raises(ValueError, match=r"cut has to be True, False or None"):
        signature(['snr'], SR, cut='yes')


def test_resampled_names_the_resampler_and_the_packages_behind_it():
    fields = parse_signature(signature(['snr'], SR, resampled=True))
    assert list(fields)[:4] == ['kudio', 'sr', 'resample', 'pair']
    assert fields['resample'] == 'soxr_hq'
    assert set(fields['pkg']) == {'librosa', 'soxr'}
    assert 'resample' not in parse_signature(signature(['snr'], SR))
    assert parse_signature(signature(['wer'], SR, resampled=True))['sr'] == str(SR)
    with pytest.raises(ValueError, match=r"resampled needs sr"):
        signature(['wer'], resampled=True)
