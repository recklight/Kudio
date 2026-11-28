# -*- coding: utf-8 -*-
"""A signature for a result: one line that says how its numbers were made.

Two papers that both report "PESQ 2.45" have not necessarily computed the
same thing. PESQ can be raw P.862 or a MOS-LQO, narrowband or wideband, and
the ``pesq`` package's wideband score is the 2005 code's, before ITU's 2018
correction; CSIG, CBAK and COVL move with whichever PESQ went into them; an
LSD depends on its log, its frames and its gain match, a min DCF on its
prior, a CER on whether spaces count, and DNSMOS P.808 on whose mel
spectrogram the model was given. Post [P18] found the same of BLEU, with
differences as high as 1.8 between configurations in common use, and wrote
sacreBLEU, which prints with every score a short string that names the
settings behind it. :func:`signature` is that string for kudio's metrics::

    >>> kudio.signature(['pesq'], 16000)                    # doctest: +SKIP
    'kudio:4.0.0|sr:16000|pair:aligned=no,whole=pesq|pesq:mode=wb,scale=lqo|pkg:pesq=0.0.4'
    >>> kudio.signature({'eer': {}, 'min_dcf': {'p_target': 0.01}})  # doctest: +SKIP
    'kudio:4.0.0|eer:method=linear|min_dcf:p_target=0.01,c_miss=1.0,c_fa=1.0'

It goes under the table, or in a footnote, and :func:`parse_signature` reads
it back into a dict, so two of them can be compared field by field.

What it records:

* kudio's version, first.
* ``sr``, the rate, when one of the metrics takes a rate; and, when it is
  given, for any metric that scores a pair, rate or no rate, since speech
  at another rate is other samples. The test pair in kudio's tests, taken
  from 16 kHz down to 8 kHz, scores 3.2 dB higher on SI-SDR, and
  segmental_snr's 512-sample frames are 32 ms long at the one rate and
  64 ms at the other.
* ``resample``, when files at other rates were brought to ``sr`` as they
  were read: :func:`kudio.resample`'s method, with soxr under ``pkg``, since
  another resampler gives other samples.
* ``pair``, for the metrics that score a signal against a reference: whether
  the two were lined up with :func:`kudio.align` first, which only you can
  say, and what was done about their lengths. ``cut=shorter`` is kudio's
  rule, the pair compared over the length the two share, and ``limit`` the
  :data:`~kudio.core.evaluator.MAX_LENGTH_MISMATCH` it ran under, which
  decides which pairs are refused and so which ones a mean is over;
  ``whole=`` names what got both signals whole instead: PESQ, alone or
  inside the composite, and the MCD without its gain match.
* One field per metric, under the name it was given, with every setting that
  changes its number as it was resolved: PESQ's ``mode='auto'`` is written
  as the ``wb`` or ``nb`` it became at this rate, the composite's PESQ as the
  mode and scale it ran at, an LSD frame left at its default as the number
  of samples it came to. A default is written out like anything else, so a
  signature made today still says what ran after the default changes. A
  metric that nothing changes is its name alone.
* ``pkg``, the version of every package that computes one of the metrics, as
  installed here, or ``missing``: pesq and onnxruntime,
  and the soxr that resamples DNSMOS's input at any rate but 16 kHz. A
  DNSMOS field also names the model file, with the first 8 hex
  digits of its SHA-256, where it can tell which file ran, and the
  polynomials that mapped the network's outputs onto the MOS scale by 8 hex
  digits of a SHA-256 of their coefficients, so that kudio's own, a
  polyfit.json's and ones changed in place sign apart.
* ``n``, the number of items scored, and ``ci``, which confidence interval
  over them was computed and how, when they are passed.

What it leaves out. numpy and scipy: every metric runs on them and none is
defined by them, and listing their versions would make two signatures
differ between machines that computed the same thing. And how
:func:`kudio.align` searched: *aligned* says the pair was lined up, not
where it looked.

The text. Fields are separated by ``|`` and always come in the order above,
the metrics in the order of :data:`~kudio.core.registry.METRIC_INFO`
whatever order they were given in, so one set of metrics and settings always
gives one string. A field is a name; a name, a colon and a value; or a name,
a colon and ``setting=value`` pairs separated by commas. True and False are
written ``yes`` and ``no``, numbers as Python writes them (``0.01``,
``1e-08``, ``512``), and nothing is outside ASCII.

References
----------
[P18] M. Post, "A call for clarity in reporting BLEU scores", Proc. Third
      Conference on Machine Translation: Research Papers, Brussels, October
      2018, pp. 186-191, doi:10.18653/v1/W18-6319.
"""
from __future__ import annotations

import hashlib
import inspect
import math
import numbers
import re
from importlib import metadata
from pathlib import Path
from types import MappingProxyType
from typing import (Any, Callable, Dict, Iterable, List, Mapping, NamedTuple, Optional,
                    Tuple, Union)

from kudio._version import __version__
from kudio.core.registry import METRIC_INFO

__all__ = ['signature', 'parse_signature']

#: the fields that are not metrics, in the order they are written; the
#: metrics go between ``pair`` and ``pkg``
_BEFORE = ('kudio', 'sr', 'resample', 'pair')
_AFTER = ('pkg', 'n', 'ci')

#: a field or setting name, and a value
_KEY = re.compile(r'[a-z][a-z0-9_]*')
_TOKEN = re.compile(r'[A-Za-z0-9._+~!@-]+')
#: what a file name or a version keeps; anything else becomes '_'
_UNSAFE = re.compile(r'[^A-Za-z0-9._+~!-]')

#: the distributions that install a package, where there is more than one
_DISTRIBUTIONS = {'onnxruntime': ('onnxruntime', 'onnxruntime-gpu',
                                  'onnxruntime-directml', 'onnxruntime-openvino',
                                  'onnxruntime-qnn')}


# --- what each metric's number depends on ----------------------------------------

class _Spec(NamedTuple):
    """What changes one metric function's number, besides its input."""

    #: its number depends on the rate given with the samples, so the
    #: signature needs sr
    rate: bool
    #: it scores a signal against a reference, so ``pair`` applies to it
    pair: bool
    #: the keyword arguments that change the number, in the function's order
    takes: Tuple[str, ...] = ()
    #: the packages that compute it whatever its settings
    packages: Tuple[str, ...] = ()
    #: keyword arguments that do not change its number, and why
    ignores: Mapping[str, str] = MappingProxyType({})


class _Run(NamedTuple):
    """One metric's settings as they ran."""

    settings: Tuple[Tuple[str, Any], ...] = ()
    #: whether a pair is cut to the length the two share
    cut: bool = True
    #: what took the pair whole instead
    whole: Tuple[str, ...] = ()
    #: the packages that computed it, beyond the spec's own
    packages: Tuple[str, ...] = ()


#: why score()'s own arguments do not change STOI, ESTOI or SDR
_SCORE_ONLY = MappingProxyType({
    'metrics': "which metrics score() runs; this one's name is in the signature.",
    'pesq_mode': "read by score()'s pesq entry alone.",
    'pesq_scale': "read by score()'s pesq entry alone.",
})
_SDR_ONLY = MappingProxyType(dict(
    _SCORE_ONLY, sr="BSS Eval takes no rate: its 512-tap filter is counted in "
                    "samples, so the same samples score the same whatever rate is "
                    "named; a rate that is given is still written, for the pair."))

#: a pair scored at its own rate, with nothing else to set
_AT_RATE = _Spec(rate=True, pair=True)

#: by the metric function's own name, or by score()'s name for the three
#: metrics only score() computes
_SPECS: Mapping[str, _Spec] = MappingProxyType({
    # against the clean reference, sample for sample
    'snr': _Spec(rate=False, pair=True),
    'si_sdr': _Spec(rate=False, pair=True),
    'segmental_snr': _Spec(rate=False, pair=True, takes=('frame_length', 'hop_length')),
    'sdi': _Spec(rate=False, pair=True),
    'pesq': _Spec(rate=True, pair=True, takes=('mode', 'scale'), packages=('pesq',)),
    'stoi': _Spec(rate=True, pair=True, ignores=_SCORE_ONLY),
    'estoi': _Spec(rate=True, pair=True, ignores=_SCORE_ONLY),
    'sdr': _Spec(rate=False, pair=True, ignores=_SDR_ONLY),
    'lsd': _Spec(rate=True, pair=True,
                 takes=('n_fft', 'hop_length', 'scale', 'match_gain', 'eps')),
    # Hu and Loizou's measures and the composite
    'llr': _AT_RATE,
    'itakura_saito': _AT_RATE,
    'cepstral_distance': _AT_RATE,
    'wss': _AT_RATE,
    'fw_segmental_snr': _AT_RATE,
    'composite': _Spec(rate=True, pair=True, takes=('mode', 'scale', 'pesq')),
    'mcd': _Spec(rate=True, pair=True,
                 takes=('match_gain', 'order', 'alpha', 'n_fft', 'hop_length')),
    # intelligibility
    'ncm': _AT_RATE,
    'csii': _AT_RATE,
    'siib': _AT_RATE,
    'siib_gauss': _AT_RATE,
    # several sources
    'si_sdr_improvement': _Spec(rate=False, pair=True),
    'sdr_improvement': _Spec(rate=False, pair=True),
    'pit_si_sdr': _Spec(rate=False, pair=True),
    'bss_eval': _Spec(rate=False, pair=True, takes=('permutation',)),
    # no reference
    'srmr': _Spec(rate=True, pair=False,
                  takes=('fast', 'norm', 'min_cf', 'max_cf', 'drop_silence')),
    'wada_snr': _Spec(rate=True, pair=False),
    'dnsmos': _Spec(rate=True, pair=False,
                    takes=('model_dir', 'polyfit', 'personalized', 'session'),
                    packages=('onnxruntime',)),
    'dnsmos_p808': _Spec(rate=True, pair=False, takes=('model_dir', 'librosa', 'session'),
                         packages=('onnxruntime',)),
    'loudness': _Spec(rate=True, pair=False, takes=('block_size',)),
    'loudness_range': _Spec(rate=True, pair=False),
    'true_peak': _Spec(rate=True, pair=False, takes=('oversample',)),
    'rt60': _Spec(rate=True, pair=False),
    # recognizers
    'wer': _Spec(rate=False, pair=False),
    'cer': _Spec(rate=False, pair=False, takes=('spaces',)),
    # speaker verification
    'eer': _Spec(rate=False, pair=False, takes=('method',)),
    'min_dcf': _Spec(rate=False, pair=False, takes=('p_target', 'c_miss', 'c_fa')),
    'det_curve': _Spec(rate=False, pair=False),
    'cllr': _Spec(rate=False, pair=False),
    'min_cllr': _Spec(rate=False, pair=False),
    'act_dcf': _Spec(rate=False, pair=False, takes=('p_target', 'c_miss', 'c_fa')),
    'c_primary': _Spec(rate=False, pair=False, takes=('minimum',)),
    # diarization
    'der': _Spec(rate=False, pair=False, takes=('collar', 'skip_overlap', 'uem')),
    'jer': _Spec(rate=False, pair=False, takes=('uem',)),
    'detection_error_rate': _Spec(rate=False, pair=False, takes=('collar', 'uem')),
    # pitch trackers
    'gross_pitch_error': _Spec(rate=False, pair=False, takes=('threshold',)),
    'voicing_decision_error': _Spec(rate=False, pair=False),
    'f0_frame_error': _Spec(rate=False, pair=False, takes=('threshold',)),
    'fine_pitch_error': _Spec(rate=False, pair=False, takes=('threshold',)),
    'f0_rmse': _Spec(rate=False, pair=False),
})


def _own(name: str) -> str:
    """The name *name*'s spec is kept under: the function's, or score()'s
    name for a metric only score() computes."""
    function = METRIC_INFO[name].function
    return name if function == 'kudio.score' else function.rsplit('.', 1)[-1]


def _function(name: str) -> Callable:
    import kudio
    obj: Any = kudio
    for part in METRIC_INFO[name].function.split('.')[1:]:
        obj = getattr(obj, part)
    return obj


# --- reading the settings --------------------------------------------------------

def _refuse(name: str, message: str) -> ValueError:
    return ValueError(f"signature(): {name}: {message}")


def _reworded(name: str, error: Exception) -> ValueError:
    """*error* from one of kudio's own checks, said as the signature's."""
    message = str(error)
    head, sep, rest = message.partition('(): ')
    if sep and _KEY.fullmatch(head):
        message = rest
    return _refuse(name, message)


def _is_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    import numpy as np
    return isinstance(value, np.bool_)


def _flag(o: Mapping[str, Any], key: str, name: str) -> bool:
    value = o[key]
    if not _is_flag(value):
        raise _refuse(name, f"{key} has to be True or False, got {value!r}")
    return bool(value)


def _finite(value: Any) -> bool:
    """Whether *value* is a real number, not a bool, that a float holds."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False


def _number(o: Mapping[str, Any], key: str, name: str, low: Optional[float] = None,
            high: Optional[float] = None, *, low_open: bool = False,
            high_open: bool = True) -> float:
    value = o[key]
    if not _finite(value):
        raise _refuse(name, f"{key} has to be a number, got {value!r}")
    value = float(value)
    too_low = low is not None and (value <= low if low_open else value < low)
    too_high = high is not None and (value >= high if high_open else value > high)
    if too_low or too_high:
        lo = '' if low is None else f"{'above' if low_open else 'from'} {low:g}"
        hi = '' if high is None else f"{'below' if high_open else 'up to'} {high:g}"
        raise _refuse(name, f"{key} has to be {' and '.join(p for p in (lo, hi) if p)}, "
                            f"got {value:g}")
    # -0.0 is the same number as 0.0, and would be written '-0.0'
    return value + 0.0


def _count(o: Mapping[str, Any], key: str, name: str, low: int = 1,
           high: Optional[int] = None) -> int:
    value = o[key]
    whole = _finite(value) and float(value).is_integer()
    if not whole or value < low or (high is not None and value > high):
        span = f"{low} or more" if high is None else f"from {low} to {high}"
        raise _refuse(name, f"{key} has to be a whole number, {span}, got {value!r}")
    return int(value)


def _one_of(o: Mapping[str, Any], key: str, name: str, choices: Tuple[str, ...]) -> str:
    value = o[key]
    if not isinstance(value, str) or value not in choices:
        raise _refuse(name, f"{key} has to be one of {choices}, got {value!r}")
    return value


def _given(o: Mapping[str, Any], key: str) -> str:
    """``'given'`` or ``'none'``, for an argument whose value is input rather
    than a setting, such as a UEM."""
    return 'none' if o[key] is None else 'given'


def _clean(text: str) -> str:
    """*text*, a file name or a version from this machine, as a value."""
    return _UNSAFE.sub('_', str(text)) or '_'


def _digest(path: Path) -> str:
    """The first 8 hex digits of the SHA-256 of a file: enough to tell the
    models DNS-Challenge has released apart, and short enough to print."""
    sha = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(1 << 20), b''):
            sha.update(block)
    return sha.hexdigest()[:8]


def _installed(package: str) -> str:
    """The version of *package* installed here, or ``'missing'``."""
    for dist in _DISTRIBUTIONS.get(package, (package,)):
        try:
            return _clean(metadata.version(dist))
        except metadata.PackageNotFoundError:
            continue
    return 'missing'


# --- resolving each metric's settings --------------------------------------------

def _pesq_variant(name: str, sr, mode, scale) -> str:
    """The bandwidth *mode* runs at, as :func:`kudio.pesq` resolves it."""
    from kudio.core.evaluator import _pesq_mode
    try:
        return _pesq_mode(sr, mode, scale)
    except ValueError as e:
        raise _reworded(name, e) from None


def _segmental_snr(o, sr, name) -> _Run:
    frame = _count(o, 'frame_length', name, 1)
    return _Run((('frame_length', frame),
                 ('hop_length', _count(o, 'hop_length', name, 1, frame))))


def _pesq(o, sr, name) -> _Run:
    mode = _pesq_variant(name, sr, o['mode'], o['scale'])
    return _Run((('mode', mode), ('scale', o['scale'])), cut=False, whole=('pesq',))


def _lsd(o, sr, name) -> _Run:
    from kudio.core.evaluator import _LSD_FRAME, _LSD_SCALES
    n_fft = int(sr * _LSD_FRAME) if o['n_fft'] is None else _count(o, 'n_fft', name, 2)
    if n_fft < 2:
        raise _refuse(name, f"at {sr} Hz the default frame, int(0.032 * sr), is "
                            f"{n_fft} sample(s), and it needs 2; pass n_fft")
    hop = n_fft // 2 if o['hop_length'] is None else \
        _count(o, 'hop_length', name, 1, n_fft)
    return _Run((('n_fft', n_fft), ('hop_length', hop),
                 ('scale', _one_of(o, 'scale', name, tuple(_LSD_SCALES))),
                 ('match_gain', _flag(o, 'match_gain', name)),
                 ('eps', _number(o, 'eps', name, 0.0, low_open=True))))


def _composite(o, sr, name) -> _Run:
    from kudio.core.quality import _composite_scale
    try:
        scale = _composite_scale(sr, o['mode'], o['scale'])
    except ValueError as e:
        raise _reworded(name, e) from None
    mode = _pesq_variant(name, sr, o['mode'], scale)
    # a PESQ passed in is a number per item, so what the signature can say
    # is that one was, and which variant mode and scale call it
    given = o['pesq'] is not None
    return _Run((('mode', mode), ('scale', scale),
                 ('pesq', 'given' if given else 'computed')),
                whole=() if given else ('pesq',),
                packages=() if given else ('pesq',))


def _mcd(o, sr, name) -> _Run:
    from kudio.core.melcepstrum import MCD_SETTINGS
    gain = _flag(o, 'match_gain', name)
    if (o['order'] is None) != (o['alpha'] is None):
        raise _refuse(name, "pass order and alpha together, or neither for the "
                            "rate's own")
    if o['order'] is None:
        if sr not in MCD_SETTINGS:
            raise _refuse(name, f"URGENT's MCD has settings for "
                                f"{', '.join(str(r) for r in MCD_SETTINGS)} Hz, not "
                                f"{sr} Hz; pass order and alpha")
        order, alpha = MCD_SETTINGS[sr]
    else:
        order = _count(o, 'order', name, 1)
        alpha = _number(o, 'alpha', name, -1.0, 1.0, low_open=True)
    return _Run((('match_gain', gain), ('order', int(order)), ('alpha', float(alpha)),
                 ('n_fft', _count(o, 'n_fft', name, 4)),
                 ('hop_length', _count(o, 'hop_length', name, 1))),
                cut=gain, whole=() if gain else ('mcd',))


def _bss_eval(o, sr, name) -> _Run:
    # without one, BSS Eval keeps the permutation with the highest mean SIR
    return _Run((('permutation', 'best_sir' if o['permutation'] is None else 'given'),))


def _srmr(o, sr, name) -> _Run:
    from kudio.core.srmr import _MAX_CF, _MAX_CF_NORM
    norm = _flag(o, 'norm', name)
    top = (_MAX_CF_NORM if norm else _MAX_CF) if o['max_cf'] is None else \
        _number(o, 'max_cf', name, 0.0, low_open=True)
    return _Run((('fast', _flag(o, 'fast', name)), ('norm', norm),
                 ('min_cf', _number(o, 'min_cf', name, 0.0, low_open=True)),
                 ('max_cf', float(top)),
                 ('drop_silence', _flag(o, 'drop_silence', name))))


def _resampled(sr) -> Tuple[str, ...]:
    """What DNSMOS's input goes through on its way to 16 kHz:
    :func:`kudio.resample`, which is soxr's."""
    from kudio.core.dnsmos import MODEL_SR
    return () if sr == MODEL_SR else ('soxr',)


def _model(kind: str, o, name) -> Tuple[str, Optional[Path]]:
    """The model file as the signature names it, and its path where known.

    With a session of your own, dnsmos reads nothing from *model_dir*, so
    only a *model_dir* that is the model file itself says what the session
    holds; a folder could hold either P.835 model, and anything else is
    ``unknown``. Without a session, the model is the one
    :func:`kudio.find_dnsmos_model` finds, the one the metric ran; a
    *model_dir* where it finds none is refused, since the metric would have
    refused it too, and with no *model_dir* and nothing in the environment
    the model is ``unknown``, so that a signature can still be written.
    """
    from kudio.core.dnsmos import find_dnsmos_model
    from kudio.exceptions import FeatureError
    if o['session'] is not None:
        try:
            path = None if o['model_dir'] is None else Path(o['model_dir'])
        except TypeError:
            path = None
        if path is None or not path.is_file():
            return 'unknown', None
    else:
        try:
            path = find_dnsmos_model(o['model_dir'], model=kind)
        except FeatureError as e:
            if o['model_dir'] is None:
                return 'unknown', None
            raise _reworded(name, e) from None
    return f"{_clean(path.name)}@{_digest(path)}", path


def _polyfit_digest(coefficients: Mapping[str, Any]) -> str:
    """The first 8 hex digits of the SHA-256 of the polynomials, written out
    coefficient by coefficient: the same numbers give the same digits
    whether they are kudio's own or came from a polyfit.json."""
    from kudio.core.dnsmos import POLYFIT
    text = ';'.join(f"{key}:{','.join(repr(float(c)) for c in coefficients[key])}"
                    for key in POLYFIT)
    return hashlib.sha256(text.encode('ascii')).hexdigest()[:8]


def _dnsmos(o, sr, name) -> _Run:
    from kudio.core.dnsmos import POLYFIT, POLYFIT_PERSONALIZED, _load_polyfit
    from kudio.exceptions import FeatureError
    personalized = _flag(o, 'personalized', name)
    polyfit = _flag(o, 'polyfit', name)
    model, path = _model('personalized' if personalized else 'p835', o, name)
    coefficients = dict(POLYFIT_PERSONALIZED if personalized else POLYFIT)
    # a polyfit.json beside the model replaces them, unless a session was
    # passed, when dnsmos never reads the folder
    if polyfit and path is not None and o['session'] is None:
        try:
            coefficients, _ = _load_polyfit(path, coefficients)
        except (FeatureError, ValueError) as e:
            raise _reworded(name, e) from None
    mapping = _polyfit_digest(coefficients) if polyfit else 'no'
    return _Run((('model', model), ('polyfit', mapping), ('personalized', personalized)),
                packages=_resampled(sr))


def _dnsmos_p808(o, sr, name) -> _Run:
    model, _ = _model('p808', o, name)
    # the two spectrograms dnsmos_p808 computes, as it checks them
    return _Run((('model', model), ('librosa', _one_of(o, 'librosa', name, ('0.8', '0.9')))),
                packages=_resampled(sr))


def _true_peak(o, sr, name) -> _Run:
    from kudio.core.loudness import _TRUE_PEAK_SR
    factor = max(1, int(math.ceil(_TRUE_PEAK_SR / sr))) if o['oversample'] is None else \
        _count(o, 'oversample', name, 1)
    return _Run((('oversample', factor),))


def _eer(o, sr, name) -> _Run:
    from kudio.core.verification import _EER_METHODS
    return _Run((('method', _one_of(o, 'method', name, tuple(_EER_METHODS))),))


def _dcf(o, sr, name) -> _Run:
    return _Run((('p_target', _number(o, 'p_target', name, 0.0, 1.0, low_open=True)),
                 ('c_miss', _number(o, 'c_miss', name, 0.0, low_open=True)),
                 ('c_fa', _number(o, 'c_fa', name, 0.0, low_open=True))))


def _collar(o, name) -> Tuple[str, float]:
    return 'collar', _number(o, 'collar', name, 0.0)


def _threshold(o, sr, name) -> _Run:
    return _Run((('threshold', _number(o, 'threshold', name, 0.0, 1.0, low_open=True)),))


#: how each metric with settings resolves them: (options, rate, name) -> _Run
_RESOLVE: Mapping[str, Callable[[Dict[str, Any], Any, str], _Run]] = MappingProxyType({
    'segmental_snr': _segmental_snr,
    'pesq': _pesq,
    'lsd': _lsd,
    'composite': _composite,
    'mcd': _mcd,
    'bss_eval': _bss_eval,
    'srmr': _srmr,
    'dnsmos': _dnsmos,
    'dnsmos_p808': _dnsmos_p808,
    'loudness': lambda o, sr, name: _Run(
        (('block_size', _number(o, 'block_size', name, 0.0, low_open=True)),)),
    'true_peak': _true_peak,
    'cer': lambda o, sr, name: _Run((('spaces', _flag(o, 'spaces', name)),)),
    'eer': _eer,
    'min_dcf': _dcf,
    'act_dcf': _dcf,
    'c_primary': lambda o, sr, name: _Run((('minimum', _flag(o, 'minimum', name)),)),
    'der': lambda o, sr, name: _Run((_collar(o, name),
                                     ('skip_overlap', _flag(o, 'skip_overlap', name)),
                                     ('uem', _given(o, 'uem')))),
    'jer': lambda o, sr, name: _Run((('uem', _given(o, 'uem')),)),
    'detection_error_rate': lambda o, sr, name: _Run((_collar(o, name),
                                                      ('uem', _given(o, 'uem')))),
    'gross_pitch_error': _threshold,
    'f0_frame_error': _threshold,
    'fine_pitch_error': _threshold,
})


def _lpc_rate(own: str) -> Callable[[Any], None]:
    def check(sr) -> None:
        from kudio.core.quality import _lpc_order
        _lpc_order(sr, own)
    return check


def _ncm_rate(sr) -> None:
    from kudio.core.intelligibility import _NCM_RATES
    if sr not in _NCM_RATES:
        raise ValueError(f"ncm(): NCM is defined at 8 and 16 kHz, not {sr} Hz")


def _csii_rate(sr) -> None:
    from kudio.core.intelligibility import _csii_rate as check
    check(sr)


def _srmr_rate(sr) -> None:
    from kudio.core.srmr import _RATES
    if sr not in _RATES:
        raise ValueError(f"srmr(): defined at 8 and 16 kHz only, not {sr} Hz")


def _siib_rate(own: str) -> Callable[[Any], None]:
    def check(sr) -> None:
        from kudio.core.siib import _whole_rate
        _whole_rate(sr, own)
    return check


#: the rates the metrics refuse, checked as each metric checks them, so that
#: no signature describes a number that could not have been computed; PESQ,
#: the composite and the MCD are checked as their settings are resolved
_RATE_CHECKS: Mapping[str, Callable[[Any], None]] = MappingProxyType({
    **{own: _lpc_rate(own) for own in ('llr', 'itakura_saito', 'cepstral_distance',
                                      'wss', 'fw_segmental_snr')},
    'ncm': _ncm_rate,
    'csii': _csii_rate,
    **{own: _siib_rate(own) for own in ('siib', 'siib_gauss')},
    'srmr': _srmr_rate,
})


def _resolved(name: str, given: Mapping[str, Any], sr) -> _Run:
    """*name*'s settings: the function's defaults, as it declares them now,
    with *given* over them, resolved."""
    spec = _SPECS[_own(name)]
    check = _RATE_CHECKS.get(_own(name))
    if check is not None:
        try:
            check(sr)
        except ValueError as e:
            raise _reworded(name, e) from None
    run = _Run()
    if spec.takes:
        parameters = inspect.signature(_function(name)).parameters
        options: Dict[str, Any] = {}
        for key in spec.takes:
            if key in given:
                options[key] = given[key]
            elif parameters[key].default is inspect.Parameter.empty:
                raise _refuse(name, f"{key} has no default, so the signature needs "
                                    f"the value it ran with")
            else:
                options[key] = parameters[key].default
        run = _RESOLVE[_own(name)](options, sr, name)
    return run._replace(cut=run.cut and spec.pair,
                        packages=tuple(spec.packages) + tuple(run.packages))


# --- the signature ---------------------------------------------------------------

def _requested(metrics) -> Dict[str, Dict[str, Any]]:
    """``{name: options}`` from a name, a list of names, or a mapping of
    names to the keyword arguments each ran with."""
    if isinstance(metrics, str):
        requested = {metrics: {}}
    elif isinstance(metrics, (bytes, bytearray)):
        raise ValueError(f"signature(): a metric name is text, got bytes {metrics!r}")
    elif isinstance(metrics, Mapping):
        requested = {}
        for name, options in metrics.items():
            if options is not None and not isinstance(options, Mapping):
                raise ValueError(f"signature(): the options of {name!r} have to be a "
                                 f"dict of the keyword arguments it ran with, got "
                                 f"{options!r}")
            requested[name] = dict(options or {})
    else:
        try:
            requested = {name: {} for name in metrics}
        except TypeError:
            raise ValueError(f"signature(): metrics has to be a name, a list of names "
                             f"or a dict of names to their options, got "
                             f"{metrics!r}") from None
    odd = [n for n in requested if not isinstance(n, str)]
    if odd:
        raise ValueError(f"signature(): metric names have to be strings, got {odd}")
    unknown = [n for n in requested if n not in METRIC_INFO]
    if unknown:
        raise ValueError(f"signature(): unknown metric(s) {unknown}; choose from "
                         f"{tuple(METRIC_INFO)}")
    if not requested:
        raise ValueError("signature(): no metrics given; a signature describes how "
                         "some metric was computed")
    for name, options in requested.items():
        spec = _SPECS.get(_own(name))
        if spec is None:
            raise ValueError(f"signature(): {name} has no signature yet: which of "
                             f"its settings change its number has not been decided")
        allowed = spec.takes + tuple(spec.ignores)
        unknown = [k for k in options if k not in allowed]
        if unknown:
            if not allowed:
                raise _refuse(name, f"nothing but its input changes it, so it takes "
                                    f"no options; got {unknown}")
            raise _refuse(name, f"unknown option(s) {unknown}; choose from {allowed}")
    return requested


def _interval(ci: Mapping[str, Any]) -> Dict[str, str]:
    """The ``ci`` field: which interval, and the settings it ran with."""
    required = ('kind', 'method', 'confidence', 'n_resamples', 'seed')
    if not isinstance(ci, Mapping):
        raise ValueError(f"signature(): ci has to be a dict of the interval's "
                         f"settings, {required} and groups, got {ci!r}")
    unknown = [k for k in ci if k not in required + ('groups',)]
    if unknown:
        raise ValueError(f"signature(): ci: unknown setting(s) {unknown}; choose from "
                         f"{required + ('groups',)}")
    missing = [k for k in required if k not in ci]
    if missing:
        raise ValueError(f"signature(): ci: {missing} missing; an interval can be "
                         f"told apart from another, and drawn again, only with all "
                         f"of {required}")
    # a mean of per-item rates and a pooled rate over the same items are two
    # intervals with the same settings
    kind = _one_of(ci, 'kind', 'ci', _INTERVALS)
    method = ci['method']
    if not isinstance(method, str) or not _KEY.fullmatch(method):
        raise ValueError(f"signature(): ci: method has to be the name of the method "
                         f"the interval reports, such as 'bca', got {method!r}")
    import numpy as np
    seed = ci['seed']
    if seed is None:
        drawn = 'none'
    elif isinstance(seed, np.random.Generator):
        # a generator carries on from wherever earlier draws left it, which
        # nothing here can write down
        drawn = 'generator'
    else:
        drawn = _token(_count(ci, 'seed', 'ci', 0))
    return {'kind': kind, 'method': method,
            'confidence': _token(_number(ci, 'confidence', 'ci', 0.0, 1.0, low_open=True)),
            'n_resamples': _token(_count(ci, 'n_resamples', 'ci', 1)),
            'seed': drawn, 'groups': _groups(ci.get('groups'))}


#: the functions of kudio.core.stats that give a ConfidenceInterval
_INTERVALS = ('bootstrap_ci', 'paired_ci', 'rate_ci', 'paired_rate_ci')


def _groups(groups: Any) -> str:
    """What the ``ci`` field says of the groups: ``none``, the name given for
    them, or how many there are, counted from the labels as the interval
    counts them."""
    if groups is None:
        return 'none'
    if isinstance(groups, str):
        if not _TOKEN.fullmatch(groups):
            raise ValueError(f"signature(): ci: groups names what the items were "
                             f"grouped by, in letters, digits and ._+~!@-, got "
                             f"{groups!r}")
        return groups
    import numpy as np
    from kudio.core.stats import _group_index
    if not isinstance(groups, np.ndarray):
        try:
            groups = list(groups)
        except TypeError:
            raise ValueError(f"signature(): ci: groups has to be a name or the items' "
                             f"group labels, got {groups!r}") from None
    count = int(groups.size) if isinstance(groups, np.ndarray) else len(groups)
    try:
        _, count = _group_index(groups, count, 'signature')
    except ValueError as e:
        raise _reworded('ci', e) from None
    return str(count)


def _token(value: Any) -> str:
    if isinstance(value, bool):
        text = 'yes' if value else 'no'
    elif isinstance(value, numbers.Integral):
        text = str(int(value))
    elif isinstance(value, numbers.Real):
        text = repr(float(value))
    else:
        text = str(value)
    if not _TOKEN.fullmatch(text):
        raise ValueError(f"signature(): {text!r} cannot be written into a signature")
    return text


def signature(metrics: Union[str, Iterable[str], Mapping[str, Optional[Mapping[str, Any]]]],
              sr: Optional[float] = None, *, pesq_mode: Optional[str] = None,
              pesq_scale: Optional[str] = None, aligned: bool = False,
              n: Optional[int] = None, ci: Optional[Mapping[str, Any]] = None,
              cut: Optional[bool] = None, resampled: bool = False) -> str:
    """One line that says how a set of metrics was computed, to print beside
    their numbers.

    >>> kudio.signature(['pesq', 'stoi'], 16000, pesq_mode='nb', n=824)  # doctest: +SKIP
    >>> kudio.signature({'lsd': dict(n_fft=2048, hop_length=441, scale='log10',
    ...                              match_gain=False, eps=1e-12)}, 44100)  # doctest: +SKIP
    >>> kudio.signature({'min_dcf': {'p_target': 0.01}, 'eer': {}})     # doctest: +SKIP

    The module notes say what goes in and how it is written. The arguments
    follow the calls the numbers came from: a list of names and *pesq_mode*
    and *pesq_scale* as :func:`kudio.score` takes them, and a dict for
    metrics called on their own, each name with the keyword arguments it ran
    with, so the same dict can be passed to both. The package versions and
    the DNSMOS model are read from this machine, so I write the signature in
    the run that computed the numbers, not later on another computer.

    :param metrics: names as :func:`kudio.score` takes them or as the
        functions are called; a list, a set or a dict's keys, in any order.
        As a dict, each name maps to its keyword arguments, ``{}`` or None
        for its defaults. An option the metric does not take is refused,
        and so is its input, such as *ref*, which is not a setting.
    :param sr: the rate the metrics ran at, checked whenever it is given.
        Needed when one of them takes a rate. Written as well when one of
        them scores a pair, whether or not it takes a rate, since the same
        pair at another rate is other samples. Left out when none does
        either, as for WER or EER. For DNSMOS, the rate of the input, before
        kudio takes it to 16 kHz.
    :param pesq_mode, pesq_scale: as :func:`kudio.score` takes them, for its
        pesq entry. Like score, they leave csig, cbak and covl alone.
    :param aligned: whether each pair was lined up with :func:`kudio.align`
        before it was scored. Recorded for the metrics that score a pair.
    :param n: how many items the numbers are over.
    :param ci: which confidence interval over the items was computed, and
        how, as a dict: ``kind``, the function that computed it
        (``'bootstrap_ci'``, ``'paired_ci'``, ``'rate_ci'`` or
        ``'paired_rate_ci'``); ``method``, the one the result reports,
        ``result.method``, which is ``'percentile'`` where BCa had no
        interval to give; ``confidence``, ``n_resamples`` and ``seed`` as
        passed to it; and ``groups``, a name for what the items were
        grouped by (``'speaker'``) or the labels themselves, which are
        counted as the interval counts them. All but ``groups`` are needed.
        A seed of None is written ``none`` and a numpy Generator
        ``generator``: neither interval can be drawn again from the
        signature.
    :param cut: whether the pairs were cut to the length the two share and
        held to :data:`~kudio.core.evaluator.MAX_LENGTH_MISMATCH`. None works
        it out from *metrics*, as :func:`kudio.score` runs them together.
        Pass it when the line names some of the metrics that ran on the same
        pairs: :func:`kudio.score_dataset` signs a column that lost items on
        a line of its own, and PESQ there still had its pairs checked for
        length, because the other metrics of the run cut them.
    :param resampled: whether files not at *sr* were brought to it with
        :func:`kudio.resample` as they were read, which then names its
        method and the packages that ran it.
    :raises ValueError: for an unknown metric or option, a setting the metric
        would refuse, a rate it is not defined at (SRMR or NCM at 44.1 kHz,
        CSII under 8 kHz), a combination it cannot run (wideband PESQ at
        8 kHz), a metric that takes a rate without *sr*, an *sr* that is
        not a rate, and a min DCF without its *p_target*. Each message
        starts ``signature(): ``.
    :returns: the signature, ASCII, the same for the same metrics and
        settings in any order.
    """
    requested = _requested(metrics)
    if 'pesq' in requested:
        for value, key in ((pesq_mode, 'mode'), (pesq_scale, 'scale')):
            if value is None:
                continue
            if key in requested['pesq']:
                raise ValueError(f"signature(): PESQ's {key} given twice, as pesq_{key} "
                                 f"and in pesq's options; pass it once")
            requested['pesq'][key] = value
    if not _is_flag(aligned):
        raise ValueError(f"signature(): aligned has to be True or False, got {aligned!r}")
    if cut is not None and not _is_flag(cut):
        raise ValueError(f"signature(): cut has to be True, False or None, got {cut!r}")
    if not _is_flag(resampled):
        raise ValueError(f"signature(): resampled has to be True or False, got "
                         f"{resampled!r}")
    if resampled and sr is None:
        raise ValueError("signature(): resampled needs sr, the rate the files were "
                         "brought to")

    if sr is not None:
        from kudio.core._policy import sample_rate
        sr = sample_rate(sr, 'signature')
    names = [name for name in METRIC_INFO if name in requested]
    specs = {name: _SPECS[_own(name)] for name in names}
    rated = [name for name in names if specs[name].rate]
    if rated and sr is None:
        raise ValueError(f"signature(): pass sr, the rate {', '.join(rated)} ran at")
    paired = [name for name in names if specs[name].pair]
    runs = {name: _resolved(name, requested[name], sr) for name in names}

    fields: Dict[str, Union[str, Dict[str, str]]] = {'kudio': _clean(__version__)}
    if sr is not None and (rated or paired or resampled):
        fields['sr'] = _token(sr)
    if resampled:
        from kudio.core.io import resample
        fields['resample'] = _token(inspect.signature(resample).parameters['res_type'].default)
    if paired:
        pair = {'aligned': _token(bool(aligned))}
        cuts = any(runs[name].cut for name in paired) if cut is None else bool(cut)
        if cuts:
            from kudio.core.evaluator import MAX_LENGTH_MISMATCH
            pair['cut'] = 'shorter'
            pair['limit'] = _token(float(MAX_LENGTH_MISMATCH))
        whole = sorted({part for name in paired for part in runs[name].whole})
        if whole:
            pair['whole'] = '+'.join(whole)
        fields['pair'] = pair
    for name in names:
        fields[name] = {key: _token(value) for key, value in runs[name].settings}
    packages = sorted({p for run in runs.values() for p in run.packages} |
                      ({'soxr'} if resampled else set()))
    if packages:
        fields['pkg'] = {p: _installed(p) for p in packages}
    if n is not None:
        if isinstance(n, bool) or not isinstance(n, numbers.Integral) or n < 1:
            raise ValueError(f"signature(): n is how many items were scored, a whole "
                             f"number from 1, got {n!r}")
        fields['n'] = _token(n)
    if ci is not None:
        fields['ci'] = _interval(ci)
    return _format(fields)


def _format(fields: Mapping[str, Union[str, Mapping[str, str]]]) -> str:
    """The text of *fields*: :func:`parse_signature` the other way round."""
    parts: List[str] = []
    for key, value in fields.items():
        if isinstance(value, str):
            parts.append(f"{key}:{value}")
        elif value:
            parts.append(key + ':' + ','.join(f"{k}={v}" for k, v in value.items()))
        else:
            parts.append(key)
    return '|'.join(parts)


def parse_signature(text: str) -> Dict[str, Union[str, Dict[str, str]]]:
    """A signature read back: ``{field: value}`` in the order written.

    >>> text = 'kudio:4.0.0|sr:16000|pesq:mode=wb,scale=lqo|stoi'
    >>> kudio.parse_signature(text)                          # doctest: +SKIP
    {'kudio': '4.0.0', 'sr': '16000', 'pesq': {'mode': 'wb', 'scale': 'lqo'}, 'stoi': {}}

    A field with settings is a dict of them, a field with one value is that
    value, and a metric with no settings is an empty dict. Every value stays
    the text it was, ``'16000'`` and ``'yes'`` included, so a signature
    written by another version of kudio, with fields this one does not know,
    still reads. Whitespace is dropped first, so a signature broken across
    lines, as a PDF column breaks it, can be pasted as it is.

    :raises ValueError: for text that is not a signature, one that does not
        start with kudio's version, and a field or a setting given twice.
    """
    if not isinstance(text, str):
        raise ValueError(f"parse_signature(): a signature is text, got "
                         f"{type(text).__name__} {text!r}")
    fields: Dict[str, Union[str, Dict[str, str]]] = {}
    for part in ''.join(text.split()).split('|'):
        key, colon, value = part.partition(':')
        if not _KEY.fullmatch(key):
            raise ValueError(f"parse_signature(): {part!r} does not start with a field "
                             f"name")
        if key in fields:
            raise ValueError(f"parse_signature(): {key} appears twice")
        if not colon:
            fields[key] = {}
        elif '=' not in value:
            if not _TOKEN.fullmatch(value):
                raise ValueError(f"parse_signature(): {key}: {value!r} is not a value")
            fields[key] = value
        else:
            settings: Dict[str, str] = {}
            for item in value.split(','):
                setting, equals, token = item.partition('=')
                if not (_KEY.fullmatch(setting) and equals and _TOKEN.fullmatch(token)):
                    raise ValueError(f"parse_signature(): {key}: {item!r} is not "
                                     f"setting=value")
                if setting in settings:
                    raise ValueError(f"parse_signature(): {key}: {setting} appears twice")
                settings[setting] = token
            fields[key] = settings
    if not isinstance(fields.get('kudio'), str) or next(iter(fields)) != 'kudio':
        raise ValueError(f"parse_signature(): a kudio signature starts with "
                         f"kudio:<version>, got {text[:40]!r}")
    return fields
