# -*- coding: utf-8 -*-
"""What every public name takes, written down once, and held to.

Each public callable is in CONTRACT, with the clips it reads and the range of each
number it is given, or in EXEMPT with the reason it is not. A new function that is
in neither turns this file red. The tests then put the same inputs to all of them:
what is not audio, a NaN, every length down to none, each number just outside its
range. A refusal has to be kudio's own ``raise``, not an error of numpy's or
Python's that surfaced on the way; an answer has to leave its input as it was.

The rate, the layout and the integers have their tests in test_input_rules.py,
the metrics in test_metric_input.py.
"""
from __future__ import annotations

import ast
import functools
import inspect
import math
import pathlib
import re
import warnings
from typing import Any, Callable, Dict, Tuple

import numpy as np
import pytest

import kudio
from kudio.core.stream import wave_decode, wave_encode

SR = 16000
KUDIO = pathlib.Path(kudio.__file__).resolve().parent
TESTS = pathlib.Path(__file__).resolve().parent
NAN, INF = float('nan'), float('inf')


def _take(seconds, level=0.4, seed=0):
    """A floor, and a buzz in the middle of it."""
    n = int(seconds * SR)
    t = np.arange(n) / SR
    y = 0.002 * np.random.default_rng(seed).standard_normal(n)
    middle = slice(n // 4, 3 * n // 4)
    y[middle] += level * sum(np.sin(2 * np.pi * 140 * k * t[middle]) / k for k in range(1, 6))
    return y.astype(np.float32)


Y = _take(0.5)
OTHER = _take(0.5, 0.2, seed=1)
LONG = _take(4.0)
IR = kudio.rir(SR, rt60=0.2, seed=0)
SPEC = kudio.waveform_to_spectrogram(Y)
POWER = np.abs(kudio.STFT(sr=SR).analyse(Y)) ** 2
DENOISED = kudio.denoise_wav2spec(Y)


class C:
    """One entry: *fn* called with *base*, and what of it is checked.

    :param waves: the arguments that are clips; ``'reference?'`` is one that may
        be None.
    :param numbers: argument -> its range, one of RANGES; a ``?`` after it lets
        it be None.
    :param nan: what one NaN or inf sample is: ``'refuse'``, ``'warn'`` (an answer
        and a warning of kudio's), or ``'passes'`` (an answer).
    :param reads: False for what moves samples about without reading them.
    :param takes: what of :func:`_not_samples` is taken, and why.
    """

    def __init__(self, fn: Callable, base: Dict[str, Any], waves: Tuple[str, ...] = (),
                 numbers: Dict[str, str] = None, nan: str = 'refuse', reads: bool = True,
                 takes: Dict[str, str] = None):
        self.fn, self.base, self.waves = fn, base, waves
        self.numbers, self.nan, self.reads = numbers or {}, nan, reads
        self.takes = takes or {}

    def __call__(self, **changed):
        return self.fn(**{**self.base, **changed})


def _stream(y, **kwargs):
    return kudio.SpectrogramStream(**kwargs).push(y)


def _enhancer(y, **kwargs):
    return kudio.StreamEnhancer(**kwargs).process(y)


def _saved(wave, sr, tmp, **kwargs):
    kudio.save_wave(tmp / "a.wav", wave, sr, **kwargs)
    return (tmp / "a.wav").read_bytes()


def _info(source, **kwargs):
    info = kudio.audio_info(source, **kwargs)
    return [info.frames, info.channels, info.peak]


def _fit(frames):
    return kudio.Standardizer().fit(frames).mean


def _corpus(utterances, **kwargs):
    return kudio.ContextFrames(utterances, **kwargs)[:]


def _tracker(**kwargs):
    return kudio.make_noise_tracker(**kwargs).update(POWER[0])


P = dict(y=Y, sr=SR)
CONTRACT = {
    # -- effects --------------------------------------------------------------------------
    'gain': C(kudio.gain, dict(y=Y, db=3.0), ('y',), dict(db='num'), nan='passes'),
    'random_gain': C(kudio.random_gain, dict(y=Y, seed=0), ('y',), dict(seed='seed'), nan='passes'),
    'normalize': C(kudio.normalize, dict(y=Y, peak=0.9), ('y',), dict(peak='pos')),
    'normalize_db': C(kudio.normalize_db, dict(y=Y, dbfs=-3.0), ('y',), dict(dbfs='nonpos')),
    'add_noise_snr': C(kudio.add_noise_snr, dict(y=Y, noise=OTHER, snr_db=5.0, seed=0),
                       ('y', 'noise'), dict(snr_db='num', seed='seed')),
    'reverb': C(kudio.reverb, dict(P, decay=0.3, delay_ms=50.0), ('y',),
                dict(decay='num', delay_ms='pos')),
    'spec_augment': C(kudio.spec_augment, dict(spec=SPEC, seed=0), (),
                      dict(n_freq_masks='count0', n_time_masks='count0', freq_mask_width='count0',
                           time_mask_width='count0', seed='seed')),
    'fade_in': C(kudio.fade_in, dict(P, duration=0.05), ('y',), dict(duration='nonneg'),
                 nan='passes'),
    'fade_out': C(kudio.fade_out, dict(P, duration=0.05), ('y',), dict(duration='nonneg'),
                  nan='passes'),
    'fade': C(kudio.fade, dict(P, fade_in_s=0.05, fade_out_s=0.05), ('y',),
              dict(fade_in_s='nonneg', fade_out_s='nonneg'), nan='passes'),
    'crossfade': C(kudio.crossfade, dict(first=Y, second=OTHER, sr=SR, duration=0.02),
                   ('first', 'second'), dict(duration='nonneg'), nan='passes'),
    'splice': C(kudio.splice, dict(parts=[Y, OTHER], sr=SR, duration=0.02), (),
                dict(duration='nonneg')),
    'reverse': C(kudio.reverse, dict(y=Y), ('y',), nan='passes', reads=False),
    'remove_dc': C(kudio.remove_dc, dict(y=Y), ('y',)),
    'band_filter': C(kudio.band_filter, dict(P, cutoff=300.0, btype='highpass'), ('y',),
                     dict(cutoff='pos', order='count')),
    'highpass': C(kudio.highpass, dict(P), ('y',), dict(cutoff='pos', order='count')),
    'lowpass': C(kudio.lowpass, dict(P, cutoff=3000.0), ('y',), dict(cutoff='pos', order='count')),
    'bandpass': C(kudio.bandpass, dict(P), ('y',), dict(low='pos', high='pos', order='count')),
    'bandstop': C(kudio.bandstop, dict(P), ('y',), dict(low='pos', high='pos', order='count')),
    'trim_silence': C(kudio.trim_silence, dict(y=Y), ('y',),
                      dict(top_db='pos', frame_length='count', hop_length='count')),
    'split_on_silence': C(kudio.split_on_silence, dict(y=Y), ('y',),
                          dict(top_db='pos', frame_length='count', hop_length='count')),
    'mu_law': C(kudio.mu_law, dict(y=Y), ('y',), dict(bits='count')),
    'a_law': C(kudio.a_law, dict(y=Y), ('y',), dict(bits='count')),
    'bit_depth': C(kudio.bit_depth, dict(y=Y, dither=True, seed=0), ('y',),
                   dict(bits='count', seed='seed')),
    'dropouts': C(kudio.dropouts, dict(P, loss=0.2, burst=2.0, seed=0), ('y',),
                  dict(loss='frac', packet_ms='pos', burst='pos?', seed='seed'), nan='passes'),
    'telephone': C(kudio.telephone, dict(P), ('y',)),
    'apply_channel': C(kudio.apply_channel, dict(P, spec='voip', seed=0), ('y',),
                       dict(seed='seed')),
    'independent_burst': C(kudio.independent_burst, dict(loss=0.05), (), dict(loss='frac<1')),
    'time_stretch': C(kudio.time_stretch, dict(y=Y, rate=1.25), ('y',), dict(rate='pos')),
    'pitch_shift': C(kudio.pitch_shift, dict(P, n_steps=2.0), ('y',), dict(n_steps='num')),
    # -- rooms and alignment ----------------------------------------------------------------
    'rir': C(kudio.rir, dict(sr=SR, rt60=0.2, seed=0), (),
             dict(rt60='pos', drr_db='num', seconds='pos?', seed='seed')),
    'apply_rir': C(kudio.apply_rir, dict(y=Y, ir=IR), ('y', 'ir')),
    'rt60': C(kudio.rt60, dict(ir=IR, sr=SR), ('ir',)),
    'schroeder_curve': C(kudio.schroeder_curve, dict(ir=IR), ('ir',)),
    'find_delay': C(kudio.find_delay, dict(ref=Y, deg=Y[100:], sr=SR), ('ref', 'deg'),
                    dict(max_seconds='nonneg+inf?')),
    'align': C(kudio.align, dict(ref=Y, deg=Y[100:], sr=SR), ('ref', 'deg'),
               dict(max_seconds='nonneg+inf?', min_correlation='num?')),
    # -- loudness and what is in a clip -------------------------------------------------------
    'loudness': C(kudio.loudness, dict(P), ('y',), dict(block_size='pos'), nan='warn'),
    'match_loudness': C(kudio.match_loudness, dict(y=Y, reference=OTHER, sr=SR), ('y', 'reference'),
                        dict(block_size='pos')),
    'normalize_lufs': C(kudio.normalize_lufs, dict(P, lufs=-30.0), ('y',),
                        dict(lufs='num', block_size='pos')),
    'loudness_over_time': C(kudio.loudness_over_time, dict(P, window=0.4), ('y',),
                            dict(window='pos', hop='pos')),
    'loudness_range': C(kudio.loudness_range, dict(y=LONG, sr=SR), ('y',)),
    'true_peak': C(kudio.true_peak, dict(P), ('y',), dict(oversample='count?'), nan='warn'),
    'audio_report': C(kudio.audio_report, dict(P), ('y',), dict(clip_run='count'),
                      nan='passes'),
    'tonal_lines': C(kudio.tonal_lines, dict(P), ('y',),
                     dict(min_prominence_db='num', resolution_hz='pos', max_lines='count0')),
    'vad': C(kudio.vad, dict(P), ('y',),
             dict(threshold_db='num+inf', flatness_max='num+inf', min_speech='nonneg',
                  min_silence='nonneg+inf', pad='nonneg', frame='pos', hop='pos')),
    'vad_split': C(kudio.vad_split, dict(P), ('y',)),
    'vad_trim': C(kudio.vad_trim, dict(P), ('y',)),
    'vad_frames': C(kudio.vad_frames, dict(P), ('y',),
                    dict(hop_length='count', n_fft='count', pad='nonneg')),
    'speech_ratio': C(kudio.speech_ratio, dict(P), ('y',)),
    'f0': C(kudio.f0, dict(P), ('y',),
            dict(fmin='pos', fmax='pos', frame_length='count?', hop_length='count?')),
    # -- features ----------------------------------------------------------------------------
    'waveform_to_spectrogram': C(kudio.waveform_to_spectrogram, dict(y=Y), ('y',),
                                 dict(forward_backward='count0?', hop_length='count',
                                      n_fft='count', win_length='count?',
                                      desired_length='pos?')),
    'spectrogram_to_waveform': C(kudio.spectrogram_to_waveform, dict(y=Y, enhanced_spec=SPEC),
                                 ('y',), dict(hop_length='count', n_fft='count')),
    'mfcc': C(kudio.mfcc, dict(P), ('y',),
              dict(n_mfcc='count', n_fft='count?', hop_length='count?', n_mels='count?',
                   fmax='pos?')),
    'melspectrogram': C(kudio.melspectrogram, dict(source=Y, sr=SR), ('source',),
                        dict(n_mels='count', n_frames='count', n_fft='count', hop_length='count',
                             power='pos')),
    'mel_band_hz': C(kudio.mel_band_hz, dict(n_mels=40, sr=SR), (),
                     dict(n_mels='count', fmin='nonneg', fmax='pos?')),
    'denoise_wav2spec': C(kudio.denoise_wav2spec, dict(waveform=Y), ('waveform',)),
    'denoise_spec2wav': C(kudio.denoise_spec2wav, dict(spec=DENOISED[0], phase=DENOISED[1])),
    'stack_context': C(kudio.stack_context, dict(frames=SPEC, context=2), (),
                       dict(context='count0')),
    'ContextFrames': C(_corpus, dict(utterances=[SPEC], context=2), (), dict(context='count0')),
    'frame_windows': C(kudio.frame_windows, dict(frames=SPEC, n_frames=16), (),
                       dict(n_frames='count')),
    'Standardizer': C(_fit, dict(frames=SPEC)),
    'features2matrix': C(kudio.features2matrix, dict(features=[SPEC, SPEC[:3]])),
    'wave_slicing': C(kudio.wave_slicing, dict(y=Y), ('y',),
                      dict(frame_length='count', hop_length='count'), nan='passes', reads=False),
    'wave_separate': C(kudio.wave_separate, dict(wave=np.stack([Y, OTHER], 1).ravel(), channels=2),
                       (), dict(channels='count')),
    # -- in and out ----------------------------------------------------------------------------
    'resample': C(kudio.resample, dict(y=Y, orig_sr=SR, target_sr=8000), ('y',)),
    'save_wave': C(_saved, dict(wave=Y, sr=SR), ('wave',)),
    'audio_info': C(_info, dict(source=Y, sr=SR), ('source',), nan='passes'),
    'wave_encode': C(wave_encode, dict(signal=Y), ('signal',),
                     takes={'int64': "16-bit numbers in any integer are their own bytes"}),
    'wave_decode': C(wave_decode, dict(in_data=wave_encode(Y), channels=1), (),
                     dict(channels='count')),
    'SpectrogramStream': C(_stream, dict(y=Y, sr=SR), ('y',),
                           dict(n_fft='count', hop_length='count?', seconds='pos', top_db='pos',
                                ref='pos', n_mels='count0?')),
    # -- enhancement ---------------------------------------------------------------------------
    'spectral_enhance': C(kudio.spectral_enhance, dict(P), ('y',),
                          dict(n_fft='count', hop_length='count0?')),
    'compare_enhancers': C(kudio.compare_enhancers, dict(P, methods=['wiener'], metrics=False,
                                                         reference=OTHER),
                           ('y', 'reference?')),
    'estimate_noise_psd': C(kudio.estimate_noise_psd, dict(power=POWER)),
    'make_noise_tracker': C(_tracker, dict(method='mcra'), (),
                            dict(ass='frac', ad='frac', ap='frac', delta='pos', window='count')),
    'StreamEnhancer': C(_enhancer, dict(y=Y, sr=SR), ('y',),
                        dict(n_fft='count', hop_length='count0?')),
    'trad_enhance': C(kudio.trad_enhance, dict(P), ('y',)),
    'wavelet_low_pass_filter': C(kudio.wavelet_low_pass_filter, dict(y=Y, thres=0.1), ('y',),
                                 dict(thres='nonneg')),
}

#: modules whose metrics take their inputs as test_metric_input.py says
_METRIC = "the metrics' own rule, test_metric_input.py"
EXEMPT = {
    **{name: _METRIC for name in (
        'si_sdr', 'snr', 'segmental_snr', 'sdi', 'lsd', 'llr', 'itakura_saito',
        'cepstral_distance', 'wss', 'fw_segmental_snr', 'composite', 'ncm', 'csii', 'siib',
        'siib_gauss', 'mcd', 'si_sdr_improvement', 'sdr_improvement', 'pit_si_sdr', 'bss_eval',
        'score', 'srmr', 'wada_snr')},
    'pesq': "needs the pesq package; its rate and length rules are test_pesq_whole.py's",
    'eval_metrics': "pesq, stoi and si-sdr of a pair: test_evaluator.py",
    'dnsmos': "needs onnxruntime and the model files: test_dnsmos.py",
    'dnsmos_p808': "needs onnxruntime and the model files: test_dnsmos.py",
    'find_dnsmos_model': "a folder, not samples: test_dnsmos.py",
    'AudioEvaluate': "folders of files: test_evaluator.py",
    'score_dataset': "folders of files: test_score_dataset.py",
    'check_metrics_install': "takes nothing",
    'metric_info': "a name, refused when unknown: test_registry.py",
    'cite': "names, refused when unknown: test_registry.py",
    'signature': "names and settings, not samples: test_signature.py",
    'parse_signature': "text: test_signature.py",
    # text, scores, labels and tracks, each refused in test files of their own
    **{name: "text, not samples: test_wer.py" for name in (
        'wer', 'cer', 'word_errors', 'char_errors')},
    **{name: "scores and labels, not samples: test_verification.py" for name in (
        'eer', 'min_dcf', 'det_curve')},
    **{name: "scores and labels, not samples: test_calibration.py" for name in (
        'cllr', 'min_cllr', 'act_dcf', 'c_primary')},
    **{name: "turns, not samples: test_diarization.py" for name in (
        'der', 'diarization_errors', 'jer', 'detection_error_rate', 'load_rttm')},
    **{name: "pitch tracks, not samples: test_f0eval.py" for name in (
        'gross_pitch_error', 'voicing_decision_error', 'f0_frame_error', 'fine_pitch_error',
        'f0_rmse', 'f0_errors')},
    **{name: "values and groups, not samples: test_stats.py" for name in (
        'bootstrap_ci', 'paired_ci', 'rate_ci', 'paired_rate_ci', 'permutation_test', 'holm')},
    # files and folders
    **{name: "paths: test_io.py" for name in (
        'LoadAudio', 'check_file', 'check_input', 'check_path', 'copy_waves', 'file_load',
        'load_wave', 'load_waves', 'convert_folder')},
    **{name: "paths, and the features of what is in them: test_feature.py" for name in (
        'file_to_spectrogram', 'logspec_from_files', 'mfcc_from_files',
        'save_spectrogram_as_wave')},
    **{name: "paths and records: test_dataset.py" for name in (
        'save_manifest', 'load_manifest', 'split_pairs', 'save_labels', 'load_labels')},
    'KudioConfig': "a settings file: test_config.py",
    'KC': "KudioConfig by a shorter name",
    'Synthesizer': "folders of files: test_synth.py and test_synth_folder.py",
    'enhance_folder': "folders of files: test_spectral_enhance.py",
    'AudioDataManager': "named takes, interleaved: test_manager.py",
    # devices: their settings are refused before one is opened, in test_settings.py
    **{name: "opens a device; its settings are refused before: test_settings.py" for name in (
        'LocalStreamReader', 'Recorder', 'StreamRecorder', 'record')},
    'RemoteStreamReader': "a socket, and the blocks that come down it as they come",
    'play_audio': "says what is wrong instead of raising: test_stream.py",
    'CheckDevice': "asks the device: test_stream.py",
    **{name: "asks the device, and takes nothing that is not a name: test_lightweight.py" for name
       in ('check_device', 'is_available', 'list_devices')},
    'AudioBuffer': "blocks handed on as they are: test_buffer.py",
    'FetchBuffer': "blocks handed on as they are: test_buffer.py",
    # the noise and gain rules a frame at a time, inside StreamEnhancer
    'NoiseTracker': "what every tracker is; make_noise_tracker is in CONTRACT",
    'DecisionDirected': "a frame at a time inside StreamEnhancer, which is in CONTRACT",
    'make_gain_rule': "a name and the settings METHODS checks: test_streaming_enhance.py",
    'channel_spec': "a preset or a dict of settings: test_channel.py and test_settings.py",
    'channel_tag': "a preset or a dict of settings: test_channel.py",
    # neither samples nor settings
    'Timer': "runs a function later", 'timer': "a decorator", 'map_waves': "the caller's function",
    'find_duplicate': "any names", 'lowercase': "text", 'uppercase': "text",
}


def _kind(name):
    found = getattr(kudio, name)
    if inspect.isclass(found) and issubclass(found, BaseException):
        return 'error'
    if inspect.isclass(found) and name in RECORDS:
        return 'record'
    return 'function'


#: what a function gives back, and the settings the functions in CONTRACT take
RECORDS = {
    'Alignment', 'CompositeScore', 'CsiiScore', 'PitScore', 'BssEval', 'ErrorCounts', 'DetCurve',
    'DiarizationErrors', 'Turn', 'F0Errors', 'AudioInfo', 'ConvertResult', 'LoudnessCurve',
    'AudioReport', 'TonalLines', 'DnsmosScore', 'MetricInfo', 'ConfidenceInterval',
    'PermutationResult', 'DatasetScore', 'MetricSummary', 'PitchTrack', 'Reverberation', 'Pair',
    'Label', 'EnhanceFolderResult', 'CustomEnhancer', 'Method', 'Param', 'EnhanceResult', 'STFT'}


def test_every_public_name_is_written_down_once():
    public = [n for n in dict.fromkeys(kudio.__all__) if callable(getattr(kudio, n))]
    functions = {n for n in public if _kind(n) == 'function'}
    assert sorted(functions - set(CONTRACT) - set(EXEMPT)) == [], "add it to CONTRACT or EXEMPT"
    assert sorted(set(CONTRACT) & set(EXEMPT)) == []
    assert sorted((set(CONTRACT) | set(EXEMPT)) - functions) == []
    assert sorted(RECORDS - set(public)) == []


@pytest.mark.parametrize("name", sorted(n for n, why in EXEMPT.items() if '.py' in why))
def test_what_is_not_here_is_tested_where_it_says(name):
    for file in re.findall(r"test_\w+\.py", EXEMPT[name]):
        assert name in (TESTS / file).read_text(encoding="utf-8"), (name, file)


# -- what a refusal is ------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def _raises(path: str):
    """The lines of each ``raise X`` in a file: a bare ``raise`` passes on an error
    that is not kudio's."""
    lines = set()
    for node in ast.walk(ast.parse(pathlib.Path(path).read_text(encoding="utf-8"))):
        if isinstance(node, ast.Raise) and node.exc is not None:
            lines.update(range(node.lineno, node.end_lineno + 1))
    return lines


def _own(error: BaseException) -> bool:
    tb = error.__traceback__
    while tb.tb_next is not None:
        tb = tb.tb_next
    path = pathlib.Path(tb.tb_frame.f_code.co_filename).resolve()
    return (isinstance(error, (ValueError, TypeError, kudio.KudioError))
            and KUDIO in path.parents and tb.tb_lineno in _raises(str(path)))


def _outcome(call):
    """``('answer', value)`` or ``('refused', error)``, warnings aside; an error that
    is not kudio's own is raised."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return 'answer', call()
        except Exception as e:                                        # noqa: BLE001
            if not _own(e):
                raise AssertionError(f"not kudio's own refusal: {type(e).__name__}: {e}") from e
            return 'refused', e


def _refused(call):
    kind, value = _outcome(call)
    assert kind == 'refused', f"answered {_shown(value)}"


def _shown(value):
    return f"{type(value).__name__} {np.shape(value)}" if isinstance(value, np.ndarray) else \
        repr(value)[:80]


def _numbers(value):
    if isinstance(value, bytes):
        return [float(b) for b in value]
    if isinstance(value, dict):
        return [x for v in value.values() for x in _numbers(v)]
    if isinstance(value, (list, tuple)):
        return [x for v in value for x in _numbers(v)]
    if isinstance(value, np.ndarray):
        return value.astype(np.float64).ravel().tolist() if value.dtype.kind in 'fiub' else \
            np.abs(value).ravel().tolist()
    if hasattr(value, '__dataclass_fields__'):
        return [x for k in value.__dataclass_fields__ for x in _numbers(getattr(value, k))]
    if isinstance(value, (int, float, np.number)):
        return [float(value)]
    return []


def _same(a, b):
    a, b = _numbers(a), _numbers(b)
    return len(a) == len(b) and all(x == y or (math.isnan(x) and math.isnan(y))
                                    for x, y in zip(a, b))


def _with_tmp(name, tmp_path):
    entry = CONTRACT[name]
    return (lambda **k: entry(tmp=tmp_path, **k)) if name == 'save_wave' else entry


# -- samples ------------------------------------------------------------------------------------

def _not_samples(y):
    held = y.copy()
    masked = np.ma.masked_array(held, mask=np.arange(len(y)) == len(y) // 3)
    return {
        'complex': y + 0.1j,
        'bool': y > 0,
        'int64': np.round(y * 32767).astype(np.int64),
        'masked': masked,
        'text': np.array([f"{v:.3f}" for v in y[:400]]),
        'ragged': [[0.1, 0.2], [0.3]],
        'None': None,
        # a clip is (frames,) or (frames, channels), a channel or more
        'a number': np.float32(0.25),
        'three axes': np.zeros((4, 8, 2), np.float32),
        'no channel': np.zeros((len(y), 0), np.float32),
    }


WAVES = [(name, wave) for name, entry in CONTRACT.items() for wave in entry.waves]
#: what is refused whatever the function does with the samples
SHAPES = ('a number', 'three axes', 'no channel')


@pytest.mark.parametrize("name, wave", WAVES)
def test_what_is_not_samples_is_refused(name, wave, tmp_path):
    entry, wave = _with_tmp(name, tmp_path), wave.rstrip('?')
    for label, value in _not_samples(CONTRACT[name].base[wave]).items():
        if (label == 'None' and CONTRACT[name].waves.count(wave + '?')) or \
                label in CONTRACT[name].takes:
            continue
        if not CONTRACT[name].reads and label not in SHAPES:
            continue                       # what moves samples about takes any of them
        try:
            _refused(lambda: entry(**{wave: value}))
        except AssertionError as e:
            raise AssertionError(f"{label}: {e}") from None


@pytest.mark.parametrize("name, wave", WAVES)
def test_one_bad_sample(name, wave, tmp_path):
    """One NaN or one inf, and what happens to it is what the entry says."""
    entry, wave = _with_tmp(name, tmp_path), wave.rstrip('?')
    rule = CONTRACT[name].nan
    for bad in (NAN, INF):
        y = np.array(CONTRACT[name].base[wave], dtype=np.float64)
        y[len(y) // 3] = bad
        if rule == 'refuse':
            _refused(lambda: entry(**{wave: y}))
            continue
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            entry(**{wave: y})
        if rule == 'warn':
            assert any(issubclass(w.category, UserWarning) and "finite" in str(w.message)
                       for w in caught), [str(w.message) for w in caught]


@pytest.mark.parametrize("name, wave", WAVES)
@pytest.mark.parametrize("n", [0, 1, 2, 10, 511, 2000])
def test_any_length_is_an_answer_or_a_refusal(name, wave, n, tmp_path):
    """Not an error of numpy's: a clip of 0.04 to 0.43 s is where vad would fail
    on a broadcast."""
    entry, wave = _with_tmp(name, tmp_path), wave.rstrip('?')
    _outcome(lambda: entry(**{wave: CONTRACT[name].base[wave][:n]}))


@pytest.mark.parametrize("name", CONTRACT)
def test_the_input_is_left_as_it_was(name, tmp_path):
    entry = _with_tmp(name, tmp_path)
    given = {k: v for k, v in CONTRACT[name].base.items() if isinstance(v, np.ndarray)}
    before = {k: v.copy() for k, v in given.items()}
    _outcome(lambda: entry())
    for k, v in given.items():
        assert np.array_equal(v, before[k], equal_nan=True), k


# -- numbers --------------------------------------------------------------------------------------

#: what is outside each range, beside NaN, text, True, a list and complex
RANGES = {
    'num': [INF, -INF], 'num+inf': [], 'pos': [0, -1, INF], 'nonneg': [-1, INF],
    'nonneg+inf': [-1], 'nonpos': [1, INF, -INF], 'frac': [-0.1, 1.5, INF],
    'frac<1': [-0.1, 1, INF],
    'count': [0, -1, 2.5, INF], 'count0': [-1, 2.5, INF], 'seed': [-1, 1.5],
}
NUMBERS = [(name, arg) for name, entry in CONTRACT.items() for arg in entry.numbers]


def _range(code):
    optional = code.endswith('?')
    return code.rstrip('?'), optional


@pytest.mark.parametrize("name, arg", NUMBERS)
def test_a_number_outside_its_range_is_refused(name, arg, tmp_path):
    entry = _with_tmp(name, tmp_path)
    code, optional = _range(CONTRACT[name].numbers[arg])
    outside = [NAN, '1', [1.0], 1 + 1j, *RANGES[code]] + ([] if code == 'seed' else [True])
    if not optional and code != 'seed':
        outside.append(None)
    for value in outside:
        try:
            _refused(lambda: entry(**{arg: value}))
        except AssertionError as e:
            raise AssertionError(f"{arg}={value!r}: {e}") from None


@pytest.mark.parametrize("name, arg", NUMBERS)
def test_a_number_inside_its_range_is_taken(name, arg, tmp_path):
    entry = _with_tmp(name, tmp_path)
    code, optional = _range(CONTRACT[name].numbers[arg])
    given = CONTRACT[name].base.get(arg, inspect.signature(
        getattr(kudio, name, None) or CONTRACT[name].fn).parameters.get(arg, None))
    if isinstance(given, inspect.Parameter):
        given = given.default
    whole = code in ('count', 'count0', 'seed')
    inside = [] if given is None or given is inspect.Parameter.empty else \
        [(np.int64 if whole else np.float64)(given)]
    # the far ends of some ranges are answered by a refusal of the clip: an alignment
    # sure of itself, a loss that leaves no packet
    inside += {'num+inf': [INF, -INF], 'nonneg+inf': [INF], 'nonneg': [0], 'count0': [0],
               'frac': [0], 'frac<1': [0]}.get(code, [])
    inside += [None] if optional else []
    for value in inside:
        kind, out = _outcome(lambda: entry(**{arg: value}))
        assert kind == 'answer', f"{arg}={value!r}: {out}"


# -- flags -----------------------------------------------------------------------------------------

#: True and False, numpy's, and the whole numbers that are 0 and 1: 'no', 2 and NaN are True
#: to Python, and a check that asks Python would take every one of them for it
TRUE_OR_FALSE = [True, False, np.True_, np.False_, 1, 0]
NO_FLAG = ['no', 'False', 2, -1, NAN, 1.0, None, [True]]

#: what takes True and False only, stricter than the rule above: a signature keeps
#: what it was given, and these tests say so
STRICT = {
    'dnsmos.polyfit': 'test_dnsmos.py', 'dnsmos.personalized': 'test_dnsmos.py',
    'signature.aligned': 'test_signature.py', 'signature.cut': 'test_signature.py',
    'signature.resampled': 'test_signature.py', 'score_dataset.align': 'test_score_dataset.py',
    'cer.spaces': 'test_wer.py', 'char_errors.spaces': 'test_wer.py',
    'c_primary.minimum': 'test_verification.py', 'der.skip_overlap': 'test_diarization.py',
    'diarization_errors.skip_overlap': 'test_diarization.py',
    'mcd.match_gain': 'test_melcepstrum.py',
}


def _public_flags():
    """``name.parameter`` of every flag of the public functions and classes, by the
    annotation ``bool`` or a default of True or False."""
    found, seen = [], set()                       # KC is KudioConfig, Recorder.write save
    for name in dict.fromkeys(kudio.__all__):
        thing = getattr(kudio, name)
        if id(thing) in seen:
            continue
        seen.add(id(thing))
        if not callable(thing) or _kind(name) == 'error' or \
                (inspect.isclass(thing) and name in RECORDS and name != 'STFT'):
            continue
        targets = [(name, thing.__init__ if inspect.isclass(thing) else thing)]
        if inspect.isclass(thing):
            targets += [(f"{name}.{k}", v) for k, v in vars(thing).items()
                        if not k.startswith('_') and inspect.isfunction(v)]
        for called, function in targets:
            if called != name and id(function) in seen:
                continue
            seen.add(id(function))
            try:
                parameters = inspect.signature(function).parameters.values()
            except (TypeError, ValueError):
                continue
            found += [f"{called}.{p.name}" for p in parameters
                      if 'bool' in str(p.annotation) or isinstance(p.default, bool)]
    return sorted(set(found))


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    folder = tmp_path_factory.mktemp("flags")
    (folder / "in").mkdir()
    for k in range(2):
        kudio.save_wave(folder / "in" / f"{k}.wav", Y, SR)
    return folder


def _synthesizer(folder):
    return kudio.Synthesizer(str(folder / "in"), str(folder / "in"), out_path=str(folder / "syn"),
                             snr_ratio=0)


#: what is not in CONTRACT, as a call of the flag alone
ELSEWHERE = {
    'KudioConfig.write_config.exist_ok':
        lambda v, f: kudio.KudioConfig().write_config(str(f / "c.yaml"), exist_ok=v),
    'AudioBuffer.add.drop_full': lambda v, f: kudio.AudioBuffer(4).add(np.ones(3), drop_full=v),
    'lsd.match_gain': lambda v, f: kudio.lsd(Y, Y, SR, match_gain=v),
    'STFT.forward.norm': lambda v, f: kudio.STFT(sr=SR).forward(Y, norm=v),
    'STFT.forward.sequence': lambda v, f: kudio.STFT(sr=SR).forward(Y, sequence=v),
    'STFT.inverse.sequence': lambda v, f: kudio.STFT(sr=SR).inverse(Y, SPEC, sequence=v),
    'file_to_spectrogram.sequence': lambda v, f: kudio.file_to_spectrogram(f / "in" / "0.wav",
                                                                         sequence=v),
    'file_to_spectrogram.norm': lambda v, f: kudio.file_to_spectrogram(f / "in" / "0.wav", norm=v),
    'logspec_from_files.sequence': lambda v, f: kudio.logspec_from_files([f / "in" / "0.wav"], v,
                                                                       0, False),
    'logspec_from_files.norm': lambda v, f: kudio.logspec_from_files([f / "in" / "0.wav"], False,
                                                                   0, v),
    'save_spectrogram_as_wave.squeeze': lambda v, f: kudio.save_spectrogram_as_wave(
        f / "spec", f / "in" / "0.wav", SPEC, squeeze=v),
    'check_file.rename': lambda v, f: kudio.check_file(f / "in" / "0.wav", rename=v),
    'check_path.exist_ok': lambda v, f: kudio.check_path(f / "in", exist_ok=v),
    'file_load.mono': lambda v, f: kudio.file_load(f / "in" / "0.wav", mono=v),
    'load_waves.use_pool': lambda v, f: kudio.load_waves(f / "in", use_pool=v),
    'load_waves.std_len': lambda v, f: kudio.load_waves(f / "in", std_len=v),
    'convert_folder.mono': lambda v, f: kudio.convert_folder(f / "in", f / "conv", mono=v),
    'convert_folder.overwrite': lambda v, f: kudio.convert_folder(f / "in", f / "conv",
                                                                  overwrite=v),
    'srmr.fast': lambda v, f: kudio.srmr(Y, SR, fast=v),
    'srmr.norm': lambda v, f: kudio.srmr(Y, SR, norm=v),
    'srmr.drop_silence': lambda v, f: kudio.srmr(Y, SR, drop_silence=v),
    'AudioDataManager.add_new_file.override':
        lambda v, f: kudio.AudioDataManager().add_new_file("a", wave=[Y], sr=SR, override=v),
    'Recorder.save.rename': lambda v, f: kudio.Recorder.save(object.__new__(kudio.Recorder),
                                                             f / "r.wav", rename=v),
    'StreamRecorder.monitor': lambda v, f: kudio.StreamRecorder(monitor=v),
    'Synthesizer.write_targets': lambda v, f: kudio.Synthesizer(
        str(f / "in"), str(f / "in"), out_path=str(f / "syn"), snr_ratio=0, write_targets=v),
    'Synthesizer.waveform.is_pool': lambda v, f: _synthesizer(f).waveform(is_pool=v),
    'Synthesizer.syn.mkdir_parents': lambda v, f: _synthesizer(f).syn(seed=0, mkdir_parents=v),
    'Synthesizer.syn.is_add_clean': lambda v, f: _synthesizer(f).syn(seed=0, is_add_clean=v),
    'Synthesizer.syn.overwrite': lambda v, f: _synthesizer(f).syn(seed=0, overwrite=v),
    'Synthesizer.syn.is_pool': lambda v, f: _synthesizer(f).syn(seed=0, is_pool=v),
    'Synthesizer.syn.is_silence': lambda v, f: _synthesizer(f).syn(seed=0, is_silence=v),
    'Synthesizer.syn_extra_mode.is_pool': lambda v, f: _synthesizer(f).syn_extra_mode(
        str(f / "extra"), (5,), seed=0, is_pool=v),
    'enhance_folder.overwrite': lambda v, f: kudio.enhance_folder(f / "in", f / "enh",
                                                                  overwrite=v),
    'enhance_folder.report': lambda v, f: kudio.enhance_folder(f / "in", f / "enh", report=v),
}
#: what asks a device or a process for each, and so is not called with the flags it takes
NOT_TAKEN_HERE = {'StreamRecorder.monitor', 'Recorder.save.rename', 'Synthesizer.syn.is_pool',
                  'Synthesizer.syn_extra_mode.is_pool', 'load_waves.use_pool'}


def _flag_call(key):
    """*key*'s call of a value and the folder of files, from CONTRACT or ELSEWHERE."""
    if key in ELSEWHERE:
        return ELSEWHERE[key]
    name, parameter = key.rsplit('.', 1)
    return lambda v, f: CONTRACT[name](**{parameter: v})


def test_every_flag_is_held_to_true_or_false():
    flags = _public_flags()
    assert len(flags) > 50
    held = {f for f in flags if f.rsplit('.', 1)[0] in CONTRACT} | set(ELSEWHERE) | set(STRICT)
    assert [f for f in flags if f not in held] == [], "add it to ELSEWHERE, or CONTRACT"
    assert sorted((set(ELSEWHERE) | set(STRICT)) - set(flags)) == []
    for key, file in STRICT.items():
        assert key.rsplit('.', 1)[1] in (TESTS / file).read_text(encoding="utf-8"), key


FLAGS = [f for f in _public_flags() if f not in STRICT]


@pytest.mark.parametrize("key", FLAGS)
def test_a_flag_is_true_or_false(key, files):
    call = _flag_call(key)
    name = key.rsplit('.', 1)[1]
    for value in NO_FLAG:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError, match=rf"{name} is True or False, got "
                                                 rf"{re.escape(repr(value))}") as caught:
                call(value, files)
        assert _own(caught.value), (key, value)
    if key in NOT_TAKEN_HERE:
        return
    for value in TRUE_OR_FALSE:
        kind, out = _outcome(lambda: call(value, files))
        assert kind == 'answer' or "is True or False" not in str(out), (key, value, out)


# -- in one place ----------------------------------------------------------------------------------

def test_one_place_reads_audio():
    """A reader of its own read past what io.py refuses: an empty file, a folder."""
    where = {'io.py': "every reader", 'synth.py': "a header, to know whether it is seekable",
             'dataset_score.py': "headers, for the rates"}
    found = sorted({p.name for p in KUDIO.rglob("*.py")
                    if re.search(r"\bsf\.(read|info|SoundFile|blocks)\(|soundfile\.read\(",
                                 p.read_text(encoding="utf-8"))})
    assert [name for name in found if name not in where] == []
