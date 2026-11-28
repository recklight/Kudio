# -*- coding: utf-8 -*-
"""kudio with none of the packages it can do without: ``import kudio`` works, and
every function that needs none of them runs.

Everything runs once in a subprocess, which blocks them before kudio is imported; in
this process other test modules have loaded them already. A new metric module adds its
name to MODULES and one line per measure to SCORES, and any other new function one to
the rest of the tables.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import kudio
from kudio.core.evaluator import _BACKEND, METRICS


def _path(*folders) -> str:
    """PYTHONPATH with *folders* in front of what it was: an environment that finds its
    packages through it keeps them."""
    there = os.environ.get("PYTHONPATH")
    return os.pathsep.join([str(f) for f in folders] + ([there] if there else []))

#: put in sys.modules as None before kudio is imported, which makes importing
#: any of them fail the way a missing package does: kudio's extras, and librosa and
#: PyWavelets, which the tests that compare with them may have brought, with what
#: comes with librosa
BLOCKED = ('audioread', 'pesq', 'pystoi', 'mir_eval', 'pandas', 'openpyxl', 'matplotlib',
           'onnxruntime', 'pyaudio', 'sounddevice', 'pyloudnorm', 'librosa', 'numba',
           'llvmlite', 'sklearn', 'joblib', 'pooch', 'lazy_loader', 'msgpack', 'decorator',
           'resampy', 'samplerate', 'pywt')

#: every module a metric lives in; each has to import with BLOCKED missing
MODULES = ('kudio', 'kudio.core.evaluator', 'kudio.core.quality', 'kudio.core.srmr',
           'kudio.core.wada', 'kudio.core.loudness', 'kudio.core.report',
           'kudio.core.room', 'kudio.core.align', 'kudio.core.intelligibility',
           'kudio.core.wer', 'kudio.core.f0eval', 'kudio.core.verification',
           'kudio.core.separation', 'kudio.core.diarization', 'kudio.core.melcepstrum',
           'kudio.core.dnsmos', 'kudio.core.siib', 'kudio.core.lines')

#: Each metric that needs no optional package, as an expression that has to
#: give a finite number with every warning raised as an error. In scope:
#: ``ref``, 1 s of harmonic syllables over a faint floor at 16 kHz, and
#: ``deg``, the same under white noise at 10 dB; ``clip``, 2 s of the
#: syllables under noise at 15 dB; ``talk`` and ``heard``, 6 s of them over a
#: floor 30 dB under their peak and the same under noise at 10 dB, for SIIB,
#: which needs 5.45 s of speech; ``rir``, a synthetic room of 0.3 s; ``sr``; ``kudio``
#: and ``np``.
SCORES = {
    'snr': 'kudio.snr(ref, deg)',
    'si_sdr': 'kudio.si_sdr(ref, deg)',
    'segmental_snr': 'kudio.segmental_snr(ref, deg)',
    'sdi': 'kudio.sdi(ref, deg)',
    'lsd': 'kudio.lsd(ref, deg, sr)',
    'llr': 'kudio.llr(ref, deg, sr)',
    'itakura_saito': 'kudio.itakura_saito(ref, deg, sr)',
    'cepstral_distance': 'kudio.cepstral_distance(ref, deg, sr)',
    'wss': 'kudio.wss(ref, deg, sr)',
    'fw_segmental_snr': 'kudio.fw_segmental_snr(ref, deg, sr)',
    'composite': 'kudio.composite(ref, deg, sr, pesq=2.5).covl',
    'srmr': 'kudio.srmr(clip, sr)',
    'srmr fast': 'kudio.srmr(clip, sr, fast=True)',
    'wada_snr': 'kudio.wada_snr(clip, sr)',
    'loudness': 'kudio.loudness(clip, sr)',
    'true_peak': 'kudio.true_peak(clip, sr)',
    'audio_report': 'kudio.audio_report(clip, sr).estimated_snr_db',
    'tonal_lines': 'len(kudio.tonal_lines(clip, sr).lines)',
    'rt60': 'kudio.rt60(rir, sr).t30',
    'find_delay': 'kudio.find_delay(ref, deg, sr).delay',
    'ncm': 'kudio.ncm(ref, deg, sr)',
    'csii': 'kudio.csii(ref, deg, sr).low',
    'siib': 'kudio.siib(talk, heard, sr)',
    'siib_gauss': 'kudio.siib_gauss(talk, heard, sr)',
    'wer': "kudio.wer('the cat sat', 'the cat sat down')",
    'cer': "kudio.cer('the cat', 'the bat')",
    'f0_errors': ('kudio.f0_errors(np.r_[np.zeros(20), np.full(80, 120.0)], '
                  'np.r_[np.zeros(25), np.full(75, 123.0)]).ffe'),
    'eer': 'kudio.eer(np.r_[0.1, 0.4, 0.35, 0.8], np.r_[0, 0, 1, 1])',
    'min_dcf': 'kudio.min_dcf(np.r_[0.1, 0.4, 0.35, 0.8], np.r_[0, 0, 1, 1], p_target=0.01)',
    'si_sdr_improvement': 'kudio.si_sdr_improvement(ref, deg, ref + 2.0 * (deg - ref))',
    'sdr_improvement': 'kudio.sdr_improvement(ref, deg, ref + 2.0 * (deg - ref))',
    'bss_eval': 'kudio.bss_eval([ref], [deg]).sdr[0]',
    'cllr': 'kudio.cllr(np.r_[-2.0, 0.5, 1.0, 3.0], np.r_[0, 0, 1, 1])',
    'min_cllr': 'kudio.min_cllr(np.r_[0.1, 0.4, 0.35, 0.8], np.r_[0, 0, 1, 1])',
    'act_dcf': 'kudio.act_dcf(np.r_[-6.0, 5.0, 4.0, 6.0], np.r_[0, 0, 1, 1], p_target=0.01)',
    'c_primary': 'kudio.c_primary(np.r_[-6.0, 5.0, 4.0, 6.0], np.r_[0, 0, 1, 1])',
    'mcd': 'kudio.mcd(ref, deg, sr)',
    'der': "kudio.der([(0, 10, 'A'), (10, 20, 'B')], [(0, 12, 'x'), (12, 20, 'y')])",
    'jer': "kudio.jer([(0, 10, 'A'), (10, 20, 'B')], [(0, 12, 'x'), (12, 20, 'y')])",
    'pit_si_sdr': ('min(kudio.pit_si_sdr(np.stack([ref, deg - ref]), '
                   'np.stack([deg - ref, deg])).si_sdr)'),
}

#: the names kudio.score takes that need nothing installed
NO_PACKAGE = tuple(m for m in METRICS if m not in _BACKEND)

#: the ones of them that need more speech than ``ref`` holds, which score()
#: runs on ``talk`` and ``heard`` instead
LONG = ('siib', 'siib_gauss')

#: every function over the short-time transform, and over the pitch and the
#: stretching, as an expression that has to give a finite number
TRANSFORMS = {
    'waveform_to_spectrogram': 'kudio.waveform_to_spectrogram(clip).sum()',
    'file_to_spectrogram': 'kudio.file_to_spectrogram(wav).sum()',
    'logspec_from_files': 'kudio.logspec_from_files([wav], False, 0, False).sum()',
    'spectrogram_to_waveform': ('kudio.spectrogram_to_waveform(clip, '
                                'kudio.waveform_to_spectrogram(clip)).sum()'),
    'save_spectrogram_as_wave': ('kudio.save_spectrogram_as_wave(dst + "/back.wav", wav, '
                                 'kudio.file_to_spectrogram(wav)) or '
                                 'kudio.audio_info(dst + "/back.wav").peak'),
    'STFT': 'kudio.STFT(sr=sr).analyse(clip).real.sum()',
    'STFT both ways': ('kudio.STFT(sr=sr).synthesise(kudio.STFT(sr=sr).analyse(clip), '
                       'length=len(clip)).sum()'),
    'denoise both ways': 'kudio.denoise_spec2wav(*kudio.denoise_wav2spec(clip)).sum()',
    'wave_slicing': 'kudio.wave_slicing(clip).sum()',
    'spectral_enhance': 'kudio.spectral_enhance(clip, sr).sum()',
    'compare_enhancers': 'kudio.compare_enhancers(clip, sr, ["wiener"])[0].seconds',
    'enhance_folder': 'float(kudio.enhance_folder(wav, dst).written)',
    'StreamEnhancer': 'kudio.StreamEnhancer(sr).process(clip).sum()',
    'vad': 'kudio.speech_ratio(clip, sr)',
    'melspectrogram': 'kudio.melspectrogram(clip, sr=sr).sum()',
    'melspectrogram of a file': 'kudio.melspectrogram(wav).sum()',
    'mfcc': 'kudio.mfcc(clip, sr).sum()',
    'mfcc_from_files': 'kudio.mfcc_from_files([wav]).sum()',
    'mel_band_hz': 'kudio.mel_band_hz(40, sr).sum()',
    'SpectrogramStream with mel bands': 'kudio.SpectrogramStream(sr, n_mels=40).push(clip).sum()',
    'trim_silence': 'float(kudio.trim_silence(clip)[1][1])',
    'split_on_silence': 'float(len(kudio.split_on_silence(clip)))',
    'convert_folder trimming': 'float(kudio.convert_folder(wav, dst, trim_db=40).written)',
    'f0': 'kudio.f0(clip, sr).median_hz',
    'time_stretch': 'kudio.time_stretch(clip, 1.1).sum()',
}

#: every way to resample, as an expression that has to give the 16000 samples
#: of 2 s at 8 kHz
RESAMPLES = {
    'resample': 'kudio.resample(clip, sr, 8000).size',
    'resample (fft)': 'kudio.resample(clip, sr, 8000, res_type="fft").size',
    'resample (polyphase)': 'kudio.resample(clip, sr, 8000, res_type="polyphase").size',
    'file_load at another rate': 'kudio.file_load(wav, sr=8000)[0].size',
    'save_wave resampling': ('kudio.save_wave(dst + "/half.wav", clip, sr, d_sample=8000) or '
                             'kudio.audio_info(dst + "/half.wav").frames'),
    'Synthesizer resampling a file': 'kudio.core.synth._load(wav, 8000, "clean")[0].size',
    'convert_folder at another rate': '16000 * kudio.convert_folder(wav, dst, sr=8000).written',
    'telephone': 'kudio.telephone(clip, sr).size // 2',
    'pitch_shift': 'kudio.pitch_shift(clip, sr, 2).size // 2',
}

#: every way to read ``wav`` that resamples nothing, as an expression that has
#: to give its 32000 samples
READS = {
    'file_load': 'kudio.file_load(wav)[0].size',
    'file_load at its own rate': 'kudio.file_load(wav, sr=sr)[0].size',
    'audio_info': 'kudio.audio_info(wav).frames',
    'Synthesizer': 'kudio.core.synth._load(wav, None, "clean")[0].size',
    'Synthesizer at its own rate': 'kudio.core.synth._load(wav, sr, "clean")[0].size',
    'convert_folder at its own rate': '32000 * kudio.convert_folder(wav, dst, sr=sr).written',
}

#: everything else that needs no package, as an expression that has to give a finite
#: number or None; in scope besides, ``several``, 30 files of it under noise, and ``pairs``
REST = {
    'a_law': 'kudio.a_law(clip).sum()',
    'mu_law': 'kudio.mu_law(clip).sum()',
    'bit_depth': 'kudio.bit_depth(clip, 6, dither=True, seed=0).sum()',
    'dropouts': 'kudio.dropouts(clip, sr, seed=0).sum()',
    'apply_channel': 'kudio.apply_channel(clip, sr, "voip", seed=0).sum()',
    'channel_spec': 'float(len(kudio.channel_spec("telephone")))',
    'channel_tag': 'float(len(kudio.channel_tag("telephone")))',
    'independent_burst': 'kudio.independent_burst(0.05)',
    'gain': 'kudio.gain(clip, -3.0).sum()',
    'random_gain': 'kudio.random_gain(clip, seed=0).sum()',
    'normalize': 'kudio.normalize(clip).sum()',
    'normalize_db': 'kudio.normalize_db(clip).sum()',
    'normalize_lufs': 'kudio.normalize_lufs(clip, sr).sum()',
    'match_loudness': 'kudio.match_loudness(clip, 0.5 * clip, sr).sum()',
    'loudness_range': 'kudio.loudness_range(talk, sr)',
    'loudness_over_time': 'kudio.loudness_over_time(talk, sr, window=0.4).lufs.max()',
    'add_noise_snr': 'kudio.add_noise_snr(clip, noise[0], 5.0, seed=0).sum()',
    'reverb': 'kudio.reverb(clip, sr).sum()',
    'spec_augment': 'kudio.spec_augment(kudio.waveform_to_spectrogram(clip), seed=0).sum()',
    'band_filter': 'kudio.band_filter(clip, sr, 100.0).sum()',
    'highpass': 'kudio.highpass(clip, sr).sum()',
    'lowpass': 'kudio.lowpass(clip, sr, 4000.0).sum()',
    'bandpass': 'kudio.bandpass(clip, sr).sum()',
    'bandstop': 'kudio.bandstop(clip, sr).sum()',
    'fade': 'kudio.fade(clip, sr).sum()',
    'fade_in': 'kudio.fade_in(clip, sr).sum()',
    'fade_out': 'kudio.fade_out(clip, sr).sum()',
    'crossfade': 'kudio.crossfade(clip, ref, sr).sum()',
    'splice': 'kudio.splice([clip, ref], sr).sum()',
    'reverse': 'kudio.reverse(clip).sum()',
    'remove_dc': 'kudio.remove_dc(clip).sum()',
    'apply_rir': 'kudio.apply_rir(clip, rir).sum()',
    'schroeder_curve': 'kudio.schroeder_curve(rir)[:100].sum()',
    'align': 'float(kudio.align(ref, deg, sr)[0].size)',
    'vad': 'float(len(kudio.vad(clip, sr)))',
    'vad_frames': 'kudio.vad_frames(clip, sr).sum()',
    'vad_split': 'float(len(kudio.vad_split(clip, sr)))',
    'vad_trim': 'float(kudio.vad_trim(clip, sr)[1][1])',
    'estimate_noise_psd': ('kudio.estimate_noise_psd(np.abs(kudio.STFT(sr=sr).analyse(clip)) '
                           '** 2).sum()'),
    'make_noise_tracker': 'float(kudio.make_noise_tracker() is not None)',
    'make_gain_rule': 'kudio.make_gain_rule("wiener")(np.ones(4), np.full(4, 2.0)).sum()',
    'trad_enhance': 'kudio.trad_enhance(clip, sr).sum()',
    'wavelet_low_pass_filter': 'kudio.wavelet_low_pass_filter(clip).sum()',
    'frame_windows': 'kudio.frame_windows(kudio.mfcc(clip, sr)[0], 5).sum()',
    'stack_context': 'kudio.stack_context(kudio.mfcc(clip, sr)[0], 2).sum()',
    'features2matrix': 'kudio.features2matrix([kudio.mfcc(clip, sr)[0]] * 2)[0].sum()',
    'wave_separate': 'float(len(kudio.wave_separate(np.stack([clip, clip], 1), 2)))',
    'wave_encode': 'float(len(kudio.wave_encode(clip.astype(np.float32))))',
    'wave_decode': ('kudio.wave_decode(kudio.wave_encode(clip.astype(np.float32)), 1)'
                    '.astype(float).sum()'),
    'load_wave': 'kudio.load_wave(wav)[0].sum()',
    'load_waves': 'float(len(kudio.load_waves([wav, wav])))',
    'map_waves': 'float(len(kudio.map_waves(len, [clip, ref])))',
    'copy_waves': 'float(len(kudio.copy_waves(dst + "/copies", [wav])))',
    'check_file': 'float(len(kudio.check_file(wav)))',
    'check_input': 'float(len(kudio.check_input(wav)[0]))',
    'check_path': 'float(len(kudio.check_path(dst + "/new")))',
    'check_metrics_install': 'float(sum(kudio.check_metrics_install()))',
    'find_duplicate': 'float(len(kudio.find_duplicate([1, 1, 2])))',
    'lowercase': 'float(len(kudio.util.lowercase(["A"])))',
    'timer': 'float(kudio.timer(len)([1, 2]))',
    'save_manifest': 'kudio.save_manifest(dst + "/m.json", pairs) and None',
    'load_manifest': 'float(len(kudio.load_manifest(kudio.save_manifest(dst + "/n.json", pairs))))',
    'split_pairs': ('float(len(kudio.split_pairs([kudio.Pair(str(k), str(k)) for k in range(9)], '
                    'seed=0)[0]))'),
    'save_labels': 'kudio.save_labels(dst + "/l.txt", [kudio.Label(0.0, 1.0, "a")]) and None',
    'load_labels': ('float(len(kudio.load_labels(kudio.save_labels(dst + "/k.txt", '
                    '[kudio.Label(0.0, 1.0, "a")]))))'),
    'load_rttm': 'float(len(kudio.load_rttm(rttm)))',
    'diarization_errors': "kudio.diarization_errors([(0, 9, 'A')], [(0, 8, 'x')]).der",
    'detection_error_rate': "kudio.detection_error_rate([(0, 9, 'A')], [(0, 8, 'x')])",
    'word_errors': "float(kudio.word_errors('a b c', 'a c').errors)",
    'char_errors': "float(kudio.char_errors('a b c', 'a c').errors)",
    'f0_frame_error': 'kudio.f0_frame_error(track, track)',
    'f0_rmse': 'kudio.f0_rmse(track, track)',
    'fine_pitch_error': 'kudio.fine_pitch_error(track, track)',
    'gross_pitch_error': 'kudio.gross_pitch_error(track, track)',
    'voicing_decision_error': 'kudio.voicing_decision_error(track, track)',
    'det_curve': 'kudio.det_curve(np.r_[0.1, 0.4, 0.35, 0.8], np.r_[0, 0, 1, 1]).p_miss.sum()',
    'bootstrap_ci': 'kudio.bootstrap_ci(values, n_resamples=1000).estimate',
    'paired_ci': 'kudio.paired_ci(values, values[::-1] + 1, n_resamples=1000).estimate',
    'rate_ci': 'kudio.rate_ci(np.arange(40) % 5, np.full(40, 9), n_resamples=1000).estimate',
    'paired_rate_ci': ('kudio.paired_rate_ci(np.arange(40) % 5, np.arange(40) % 7, np.full(40, 9), '
                       'n_resamples=1000).estimate'),
    'permutation_test': 'kudio.permutation_test(values, values[::-1] + 1, n_resamples=1000).pvalue',
    'holm': 'float(np.sum(kudio.holm([0.01, 0.04])))',
    'cite': 'float(len(kudio.cite("snr")))',
    'metric_info': 'float(len(kudio.metric_info("snr").name))',
    'signature': 'float(len(kudio.signature(["snr"], sr)))',
    'parse_signature': 'float(len(kudio.parse_signature(kudio.signature(["snr"], sr))))',
    'score_dataset': ('float(len(kudio.score_dataset(several, reference={k: wav for k in several}, '
                      'metrics=["lsd"], n_resamples=1000).rows))'),
    'eval_metrics': 'kudio.eval_metrics(sr, ref, deg)[0] and None',
}

#: what runs nothing here, and why
NOT_RUN = {
    'a package of its own': ('pesq', 'dnsmos', 'dnsmos_p808', 'find_dnsmos_model'),
    'a sound device': ('is_available', 'list_devices', 'check_device', 'play_audio', 'record'),
    'another name, with a warning, for one that is run': (
        'w2s', 'f2s', 's2w', 'wav2spec', 'spec2wav', 'wavform2spec', 'spec2wavform', 'wav2mel',
        'w2mfcc', 'wav2mfcc', 'concat_logspec', 'concat_logspec_', 'concat_mfcc',
        'concat_mfcc_', 'contextual_LogSpectrogram', 'uppercase', 'upercase'),
    'colours on a console': ('blu', 'cya', 'gre', 'pur', 'red', 'yel', 'dkblu', 'dkcya',
                             'dkgre', 'dkpur', 'dkred', 'dkyel', 'fGre', 'fRed', 'h1',
                             'info', 'warn', 'error', 'color_test'),
}

#: the signals the tables above are given, built from numpy alone
_SIGNALS = r'''
import numpy as np
sr = 16000
rng = np.random.default_rng(0)
t = np.arange(2 * sr) / sr
phase = 2 * np.pi * np.cumsum(120.0 + 20.0 * np.sin(2 * np.pi * 0.5 * t)) / sr
voiced = sum(np.sin(k * phase) / k for k in range(1, 30))
syllables = (0.3 * voiced / np.max(np.abs(voiced))
             * np.clip(np.sin(2 * np.pi * 2.0 * t), 0, None) ** 2)
noise = rng.standard_normal((3, len(t)))
def at(signal, snr_db, n):
    return signal + n * np.sqrt(np.mean(signal ** 2) / 10 ** (snr_db / 10))
clip = at(syllables, 15.0, noise[0])
ref = syllables[:sr] + 1e-3 * noise[1, :sr]
deg = at(ref, 10.0, noise[2, :sr])
talk = np.tile(syllables, 3) + 0.01 * rng.standard_normal(6 * sr)
heard = at(talk, 10.0, rng.standard_normal(6 * sr))
track = np.r_[np.zeros(20), np.full(80, 120.0)]
values = rng.standard_normal(40)
'''

_SCRIPT = r'''
import contextlib, importlib, io, json, math, numbers, os, pkgutil, sys, warnings
config = json.load(sys.stdin)
for name in config["blocked"]:
    sys.modules[name] = None
out = {}
try:
    import kudio
except Exception as e:
    out["import"] = f"{type(e).__name__}: {e}"
    print(json.dumps(out))
    sys.exit(0)
out["import"] = "ok"
out["loaded"] = [name for name in config["blocked"] if sys.modules.get(name) is not None]

def number(value):
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        value = float(value)
        return value if math.isfinite(value) else repr(value)
    return "ok" if value is None else type(value).__name__

def outcome(fn, each=False):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            value = fn()
    except Exception as e:
        return f"{type(e).__name__}: {e}"
    return {k: number(v) for k, v in value.items()} if each else number(value)

out["modules"] = {name: outcome(lambda: importlib.import_module(name) and None)
                  for name in config["modules"]}

out["walked"] = {}
for found in pkgutil.walk_packages(kudio.__path__, "kudio."):
    try:
        importlib.import_module(found.name)
        out["walked"][found.name] = "ok"
    except kudio.DependencyError as e:
        out["walked"][found.name] = f"DependencyError: {e.package}, {e.extra}"
    except BaseException as e:
        out["walked"][found.name] = f"{type(e).__name__}: {e}"

scope = {}
exec(config["signals"], scope)
import soundfile as sf
wav = os.path.join(config["folder"], "clip.wav")
sf.write(wav, scope["clip"], scope["sr"], subtype="FLOAT")
rttm = os.path.join(config["folder"], "who.rttm")
with open(rttm, "w") as f:
    f.write("SPEAKER clip 1 0.00 1.50 <NA> <NA> A <NA> <NA>\\n")
several = {str(k): os.path.join(config["folder"], f"{k}.wav") for k in range(30)}
for k, path in several.items():
    sf.write(path, scope["at"](scope["clip"], float(k), scope["noise"][0]), scope["sr"],
             subtype="FLOAT")
scope.update(kudio=kudio, rir=kudio.rir(scope["sr"], rt60=0.3, seed=0), wav=wav, rttm=rttm,
             several=several,
             pairs=[kudio.Pair(noisy=wav, clean=wav, noise=wav, snr_db=0.0)],
             dst=os.path.join(config["folder"], "dst"))
ref, deg, sr = scope["ref"], scope["deg"], scope["sr"]

out["scores"] = {name: outcome(lambda: eval(expr, scope))
                 for name, expr in config["scores"].items()}
out["score"] = outcome(lambda: {
    **kudio.score(ref, deg, sr, config["no_package"]),
    **kudio.score(scope["talk"], scope["heard"], sr, config["long"])}, each=True)
out["reads"] = {name: outcome(lambda: eval(expr, scope))
                for name, expr in config["reads"].items()}
out["resamples"] = {name: outcome(lambda: eval(expr, scope))
                    for name, expr in config["resamples"].items()}
out["transforms"] = {name: outcome(lambda: eval(expr, scope))
                     for name, expr in config["transforms"].items()}
out["rest"] = {name: outcome(lambda: eval(expr, scope))
               for name, expr in config["rest"].items()}
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    from kudio.cli import main
    out["cli_report"] = outcome(lambda: float(main(["report", wav])))
    out["cli_pitch"] = outcome(lambda: float(main(["pitch", wav])))
    out["cli_convert_folder"] = outcome(lambda: float(main(
        ["convert", wav, os.path.join(config["folder"], "cli"), "--recursive",
         "--rate", "8000"])))
out["loaded_after"] = [name for name in config["blocked"] if sys.modules.get(name) is not None]
print(json.dumps(out))
'''


def _outcome(result, *keys):
    """The entry under *keys*, or a failure saying kudio did not import."""
    if result.get("import") != "ok":
        pytest.fail(f"import kudio failed with what is optional missing: {result.get('import')}")
    for key in keys:
        result = result[key]
    return result


@pytest.fixture(scope="module")
def blocked(tmp_path_factory):
    """What the subprocess found, as a dict."""
    return _run(tmp_path_factory.mktemp("lightweight"))


def _run(folder, *also_blocked):
    (folder / "dst").mkdir()
    config = dict(blocked=BLOCKED + also_blocked, modules=MODULES, scores=SCORES,
                  no_package=[m for m in NO_PACKAGE if m not in LONG], long=LONG,
                  reads=READS, resamples=RESAMPLES, transforms=TRANSFORMS, rest=REST,
                  signals=_SIGNALS, folder=str(folder))
    done = subprocess.run([sys.executable, "-c", _SCRIPT], input=json.dumps(config),
                          capture_output=True, text=True,
                          cwd=str(Path(__file__).resolve().parents[1]), timeout=600)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def no_soxr(tmp_path_factory):
    """The same with soxr missing too, as it is where it has no wheel and was
    not built."""
    return _run(tmp_path_factory.mktemp("no_soxr"), 'soxr')


def test_import_kudio_needs_none_of_it(blocked):
    assert _outcome(blocked, "import") == "ok"
    assert blocked["loaded"] == []


def test_every_function_is_run_here_or_is_known_not_to_be():
    import inspect
    import re

    tables = (SCORES, TRANSFORMS, RESAMPLES, READS, REST)
    run = {name for table in tables for expression in table.values()
           for name in re.findall(r'kudio\.(?:util\.)?(\w+)', expression)}
    left_out = {name for names in NOT_RUN.values() for name in names}
    functions = {name for name in kudio.__all__ if callable(getattr(kudio, name))
                 and not inspect.isclass(getattr(kudio, name))}
    assert functions <= run | left_out | {'score', 'rir'}, "a table above lacks it"
    assert left_out <= functions and not left_out & run


@pytest.mark.parametrize("name", REST)
def test_it_runs_without_any_optional_package(blocked, name):
    value = _outcome(blocked, "rest", name)
    assert isinstance(value, float) or value == "ok", f"{REST[name]} gave {value}"


def test_every_module_imports_or_names_the_extra_it_needs(blocked):
    """The rule a release is held to before it is uploaded, when what breaks it is in the
    tag already."""
    walked = _outcome(blocked, "walked")
    assert len(walked) > 60
    gated = {name for name, said in walked.items() if said != "ok"}
    assert gated == {'kudio.util.visual', 'kudio.util._draw'}, {n: walked[n] for n in gated}
    assert {walked[name] for name in gated} == {"DependencyError: matplotlib, viz"}


@pytest.mark.parametrize("name", MODULES)
def test_every_metric_module_imports_without_any_optional_package(blocked, name):
    assert _outcome(blocked, "modules", name) == "ok"


@pytest.mark.parametrize("name", SCORES)
def test_each_metric_scores_without_any_optional_package(blocked, name):
    value = _outcome(blocked, "scores", name)
    assert isinstance(value, float), f"{SCORES[name]} gave {value}"


@pytest.mark.parametrize("name", NO_PACKAGE)
def test_score_runs_every_metric_that_needs_no_package(blocked, name):
    scores = _outcome(blocked, "score")
    assert isinstance(scores, dict), f"kudio.score raised {scores}"
    assert isinstance(scores[name], float), f"{name} scored {scores[name]}"


@pytest.mark.parametrize("name", READS)
def test_a_file_at_its_own_rate_is_read(blocked, name):
    assert _outcome(blocked, "reads", name) == 32000


@pytest.mark.parametrize("name", RESAMPLES)
def test_resampling_needs_nothing_but_soxr(blocked, name):
    assert _outcome(blocked, "resamples", name) == 16000


@pytest.mark.parametrize("name", TRANSFORMS)
def test_a_spectrogram_needs_no_soxr_either(blocked, no_soxr, name):
    value = _outcome(blocked, "transforms", name)
    assert isinstance(value, float), f"{TRANSFORMS[name]} gave {value}"
    if name != 'compare_enhancers':                      # a time, never twice the same
        assert _outcome(no_soxr, "transforms", name) == value


@pytest.mark.parametrize("name", RESAMPLES)
def test_resampling_without_soxr_says_to_install_it(no_soxr, name):
    """Or goes without it, where scipy is what was asked for."""
    said = _outcome(no_soxr, "resamples", name)
    if name.startswith("resample ("):
        assert said == 16000
    else:
        assert said == ("DependencyError: 'soxr' is required for this feature. Install it "
                        "with: pip install soxr"), f"{RESAMPLES[name]}: {said}"


def test_everything_that_resamples_nothing_goes_without_soxr(blocked, no_soxr):
    """The metrics, reading and the modules: what they gave with soxr there."""
    for group in ("modules", "scores", "score", "reads"):
        assert _outcome(no_soxr, group) == _outcome(blocked, group)


def test_the_command_line_reports_and_tracks_pitch(blocked):
    assert _outcome(blocked, "cli_report") == 0.0
    assert _outcome(blocked, "cli_pitch") == 0.0


def test_converting_a_folder_without_soxr_stops_at_the_first_file(blocked, no_soxr):
    """Resampling needs soxr for every file in the folder, so `kudio convert
    --recursive --rate` says so once, with the exit status of an error, rather
    than listing each file as failed and exiting 1."""
    assert _outcome(blocked, "cli_convert_folder") == 0.0
    assert _outcome(no_soxr, "cli_convert_folder") == 2.0


def test_nothing_above_imported_one_behind_the_block(blocked):
    assert _outcome(blocked, "loaded_after") == []


def test_the_metrics_score_the_same_with_all_of_it_blocked(blocked):
    """Blocking changes what is imported, not what is computed, so the
    subprocess's numbers are this process's."""
    import kudio
    scope = {}
    exec(_SIGNALS, scope)
    ref, deg, clip, sr = scope["ref"], scope["deg"], scope["clip"], scope["sr"]
    here = {'llr': kudio.llr(ref, deg, sr), 'lsd': kudio.lsd(ref, deg, sr),
            'srmr': kudio.srmr(clip, sr), 'wada_snr': kudio.wada_snr(clip, sr),
            'loudness': kudio.loudness(clip, sr)}
    for name, value in here.items():
        assert math.isfinite(value) and _outcome(blocked, "scores", name) == value, name


_BROKEN_SOXR = r'''
import json, sys
import kudio
from kudio import cli
out = {}
try:
    kudio.file_load(sys.argv[1], sr=8000)
    out["error"] = "read"
except Exception as e:
    out["error"] = [type(e).__name__, isinstance(e, kudio.DependencyError), str(e)]
out["own rate"] = kudio.file_load(sys.argv[1])[0].size
out["cli"] = cli.main(["convert", sys.argv[1], sys.argv[2], "--rate", "8000"])
print(json.dumps(out))
'''


def test_a_soxr_that_will_not_import_is_a_dependency_error(tmp_path):
    """One built for another numpy, say. Installing it again is not the cure, so what it
    failed on is what is said, and what resamples nothing goes on."""
    fake = tmp_path / "fake" / "soxr"
    fake.mkdir(parents=True)
    (fake / "__init__.py").write_text(
        'raise ImportError("numpy.core.multiarray failed to import")\n')
    wav = tmp_path / "tone.wav"
    kudio.save_wave(wav, 0.1 * np.sin(2 * np.pi * 220 * np.arange(16000) / 16000), 16000)
    env = dict(os.environ, PYTHONPATH=_path(tmp_path / "fake", Path(kudio.__file__).parents[1]))
    done = subprocess.run([sys.executable, "-c", _BROKEN_SOXR, str(wav),
                           str(tmp_path / "out.wav")],
                          capture_output=True, text=True, env=env, timeout=300)
    assert done.stdout.strip(), done.stderr
    out = json.loads(done.stdout.strip().splitlines()[-1])
    name, is_dependency_error, said = out["error"]
    assert is_dependency_error, out
    assert said == ("'soxr' is required for this feature, and importing it failed: "
                    "numpy.core.multiarray failed to import")
    assert out["own rate"] == 16000 and out["cli"] == 2
    assert "Traceback" not in done.stderr, done.stderr
