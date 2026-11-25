# -*- coding: utf-8 -*-
"""kudio with librosa missing: ``import kudio`` works, every metric that needs
no optional package still scores, and a feature built on librosa says what to
install.

Everything runs once in a subprocess, which blocks librosa and the optional
packages before kudio is imported; in this process other test modules have
loaded them already. A new metric module adds its name to MODULES and one
line per measure to SCORES.
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

#: put in sys.modules as None before kudio is imported, which makes importing
#: any of them fail the way a missing package does
BLOCKED = ('librosa', 'pesq', 'pystoi', 'mir_eval', 'pandas', 'matplotlib',
           'onnxruntime', 'pyaudio', 'sounddevice')

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

#: features built on librosa, each of which has to raise DependencyError
#: naming librosa; ``wav`` is a 16 kHz file holding ``clip`` and ``dst`` a
#: folder to write to. The two folder functions raise it as well rather than
#: record it once for every file, since it would fail each of them the same way.
#: Reading a file is not one of them, unless it has to be resampled.
NEEDS_LIBROSA = {
    'melspectrogram': 'kudio.melspectrogram(clip, sr=sr)',
    'mfcc': 'kudio.mfcc(clip, sr)',
    'waveform_to_spectrogram': 'kudio.waveform_to_spectrogram(clip)',
    'resample': 'kudio.resample(clip, sr, 8000)',
    'file_load at another rate': 'kudio.file_load(wav, sr=8000)',
    'f0': 'kudio.f0(clip, sr)',
    'trim_silence': 'kudio.trim_silence(clip)',
    'time_stretch': 'kudio.time_stretch(clip, 1.1)',
    'STFT': 'kudio.STFT(sr=sr).analyse(clip)',
    'SpectrogramStream with mel bands': 'kudio.SpectrogramStream(sr, n_mels=40)',
    'Synthesizer resampling a file': 'kudio.core.synth._load(wav, 8000, "clean")',
    'convert_folder at another rate': 'kudio.convert_folder(wav, dst, sr=8000)',
    'convert_folder trimming': 'kudio.convert_folder(wav, dst, trim_db=40)',
    'enhance_folder': 'kudio.enhance_folder(wav, dst)',
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

#: the signals SCORES and NEEDS_LIBROSA are given, built from numpy alone
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
'''

_SCRIPT = r'''
import contextlib, importlib, io, json, math, numbers, os, sys, warnings
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
out["librosa_loaded"] = sys.modules.get("librosa") is not None

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

scope = {}
exec(config["signals"], scope)
import soundfile as sf
wav = os.path.join(config["folder"], "clip.wav")
sf.write(wav, scope["clip"], scope["sr"], subtype="FLOAT")
scope.update(kudio=kudio, rir=kudio.rir(scope["sr"], rt60=0.3, seed=0), wav=wav,
             dst=os.path.join(config["folder"], "dst"))
ref, deg, sr = scope["ref"], scope["deg"], scope["sr"]

out["scores"] = {name: outcome(lambda: eval(expr, scope))
                 for name, expr in config["scores"].items()}
out["score"] = outcome(lambda: {
    **kudio.score(ref, deg, sr, config["no_package"]),
    **kudio.score(scope["talk"], scope["heard"], sr, config["long"])}, each=True)
out["reads"] = {name: outcome(lambda: eval(expr, scope))
                for name, expr in config["reads"].items()}
out["needs_librosa"] = {name: outcome(lambda: eval(expr, scope))
                        for name, expr in config["needs_librosa"].items()}
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    from kudio.cli import main
    out["cli_report"] = outcome(lambda: float(main(["report", wav])))
    out["cli_pitch"] = outcome(lambda: float(main(["pitch", wav])))
    out["cli_convert_folder"] = outcome(lambda: float(main(
        ["convert", wav, os.path.join(config["folder"], "cli"), "--recursive",
         "--rate", "8000"])))
out["librosa_loaded_after"] = sys.modules.get("librosa") is not None
print(json.dumps(out))
'''


def _outcome(result, *keys):
    """The entry under *keys*, or a failure saying kudio did not import."""
    if result.get("import") != "ok":
        pytest.fail(f"import kudio failed with librosa missing: {result.get('import')}")
    for key in keys:
        result = result[key]
    return result


@pytest.fixture(scope="module")
def blocked(tmp_path_factory):
    """What the subprocess found, as a dict."""
    config = dict(blocked=BLOCKED, modules=MODULES, scores=SCORES,
                  no_package=[m for m in NO_PACKAGE if m not in LONG], long=LONG,
                  needs_librosa=NEEDS_LIBROSA, reads=READS,
                  signals=_SIGNALS, folder=str(tmp_path_factory.mktemp("lightweight")))
    done = subprocess.run([sys.executable, "-c", _SCRIPT], input=json.dumps(config),
                          capture_output=True, text=True,
                          cwd=str(Path(__file__).resolve().parents[1]), timeout=600)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.strip().splitlines()[-1])


def test_import_kudio_needs_no_librosa(blocked):
    assert _outcome(blocked, "import") == "ok"
    assert blocked["librosa_loaded"] is False


@pytest.mark.parametrize("name", MODULES)
def test_every_metric_module_imports_without_librosa(blocked, name):
    assert _outcome(blocked, "modules", name) == "ok"


@pytest.mark.parametrize("name", SCORES)
def test_each_metric_scores_without_librosa_or_any_optional_package(blocked, name):
    value = _outcome(blocked, "scores", name)
    assert isinstance(value, float), f"{SCORES[name]} gave {value}"


@pytest.mark.parametrize("name", NO_PACKAGE)
def test_score_runs_every_metric_that_needs_no_package(blocked, name):
    scores = _outcome(blocked, "score")
    assert isinstance(scores, dict), f"kudio.score raised {scores}"
    assert isinstance(scores[name], float), f"{name} scored {scores[name]}"


@pytest.mark.parametrize("name", READS)
def test_a_file_at_its_own_rate_is_read_without_librosa(blocked, name):
    assert _outcome(blocked, "reads", name) == 32000


@pytest.mark.parametrize("name", NEEDS_LIBROSA)
def test_a_feature_built_on_librosa_says_to_install_it(blocked, name):
    said = _outcome(blocked, "needs_librosa", name)
    assert said == ("DependencyError: 'librosa' is required for this feature. "
                    "Install it with: pip install librosa"), f"{NEEDS_LIBROSA[name]}: {said}"


def test_the_command_line_reports_without_librosa_and_says_what_pitch_needs(blocked):
    """`kudio report` reads with soundfile and measures with numpy; `kudio
    pitch` needs pYIN, and says so on one line with the exit status of an
    error rather than a traceback."""
    assert _outcome(blocked, "cli_report") == 0.0
    assert _outcome(blocked, "cli_pitch") == 2.0


def test_converting_a_folder_without_librosa_stops_at_the_first_file(blocked):
    """Resampling needs librosa for every file in the folder, so `kudio
    convert --recursive --rate` says so once, with the exit status of an
    error, rather than listing each file as failed and exiting 1."""
    assert _outcome(blocked, "cli_convert_folder") == 2.0


def test_nothing_above_imported_librosa_behind_the_block(blocked):
    assert _outcome(blocked, "librosa_loaded_after") is False


def test_the_metrics_score_the_same_with_librosa_blocked(blocked):
    """Blocking librosa changes what is imported, not what is computed, so
    the subprocess's numbers are this process's."""
    import kudio
    scope = {}
    exec(_SIGNALS, scope)
    ref, deg, clip, sr = scope["ref"], scope["deg"], scope["clip"], scope["sr"]
    here = {'llr': kudio.llr(ref, deg, sr), 'lsd': kudio.lsd(ref, deg, sr),
            'srmr': kudio.srmr(clip, sr), 'wada_snr': kudio.wada_snr(clip, sr),
            'loudness': kudio.loudness(clip, sr)}
    for name, value in here.items():
        assert math.isfinite(value) and _outcome(blocked, "scores", name) == value, name


_BROKEN_NUMBA = r'''
import json, sys
import numpy as np
import kudio
from kudio import cli
out = {}
try:
    kudio.f0(np.zeros(16000), 16000)
    out["f0"] = "scored"
except Exception as e:
    out["f0"] = [type(e).__name__, isinstance(e, kudio.DependencyError), str(e)]
out["cli_pitch"] = cli.main(["pitch", sys.argv[1]])
print(json.dumps(out))
'''


def test_a_numba_that_will_not_import_is_a_dependency_error(tmp_path):
    """librosa 0.10 and later load their parts on first use, so a numba that
    refuses the installed numpy got past ``import librosa`` and failed inside
    librosa.pyin as a plain ImportError: kudio.f0 raised that, and ``kudio
    pitch`` printed a traceback and exited 1, its code for nothing voiced."""
    pytest.importorskip("librosa")
    fake = tmp_path / "fake" / "numba"
    fake.mkdir(parents=True)
    (fake / "__init__.py").write_text(
        'raise ImportError("Numba needs NumPy 2.0 or less. Got NumPy 2.5.")\n')
    wav = tmp_path / "tone.wav"
    kudio.save_wave(wav, 0.1 * np.sin(2 * np.pi * 220 * np.arange(16000) / 16000), 16000)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        [str(tmp_path / "fake"), str(Path(kudio.__file__).parents[1])]))
    done = subprocess.run([sys.executable, "-c", _BROKEN_NUMBA, str(wav)],
                          capture_output=True, text=True, env=env, timeout=300)
    assert done.stdout.strip(), done.stderr
    out = json.loads(done.stdout.strip().splitlines()[-1])
    name, is_dependency_error, said = out["f0"]
    assert is_dependency_error, out
    assert "librosa" in said and "Numba needs NumPy 2.0 or less" in said, said
    assert out["cli_pitch"] == 2
    assert "Traceback" not in done.stderr, done.stderr
