# -*- coding: utf-8 -*-
"""Check kudio's DNSMOS (kudio/core/dnsmos.py) against DNS-Challenge's
dnsmos_local.py, run in the environment its requirements.txt pins.

Run it from anywhere; it imports the kudio of the checkout it sits in::

    python tools/validation/dnsmos.py --reference-python PY [--models DIR]

The reference half runs in a second interpreter, PY, because the script's
pins (numpy 1.22.4, librosa 0.8.1, onnxruntime 1.13.1) are older than what
kudio installs with. I made it with Python 3.9::

    python -m venv dnsmos-ref
    dnsmos-ref/Scripts/python -m pip install numpy==1.22.4 librosa==0.8.1
        onnxruntime==1.13.1 soundfile numba==0.56.4 "scipy<1.10" "resampy<0.4.3"

numba, scipy and resampy are pinned only so that pip picks releases that
still work with numpy 1.22. DIR is the DNSMOS folder of a DNS-Challenge
checkout, which holds DNSMOS/sig_bak_ovr.onnx, DNSMOS/model_v8.onnx and
pDNSMOS/sig_bak_ovr.onnx. Without it only the P.808 model's input is
compared, since that is all kudio computes without the weights.

dnsmos_local.py (MIT, commit 6412bc5 of 2023-01-10) is restated below as it
reads: sf.read, the doubling with np.append, the windows whose end is
int((idx + 9.01) * fs) and which are skipped when that comes out short,
audio_melspec on audio_seg[:-160], np.poly1d for the mapping and np.mean
over the windows. Every clip here is at 16 kHz, where the script does not
resample, and under 17 s, where its windows and kudio's are the same ones.

Every signal is made here from fixed seeds, from the synthetic speech of
tools/validation/quality.py: 2, 9.5, 12.3 and 15.5 s of it, clean, under
white noise at 5 dB and pink noise at 10 dB, under a second talker at 0 dB,
in a 0.4 s room, low-passed at 4 kHz, and with 0.5 s of digital silence on
either side. They are written as 64-bit float WAV files, which the script
reads back exactly.

What it checks:

1. The P.808 model's input, window by window, against audio_melspec under
   librosa 0.8.1.
2. With the models: SIG, BAK and OVRL, raw and mapped, the personalised
   model's three, and the P.808 score, clip by clip.
3. With a librosa of 0.9 or later in this interpreter, dnsmos_p808's
   librosa='0.9' input against audio_melspec under that librosa, where the
   ends are padded with zeros and the FFT bins sit at their true
   frequencies, as torchmetrics 1.9.0 gets them; with the models, its score
   against the script's computation on that input, and how far it is from
   the score on 0.8.1's input.
4. With the models and torchmetrics in this interpreter (I ran 1.9.0, on
   torch 2.14.1 with numpy 2.3.5, librosa 1.0.0 and onnxruntime 1.32.0):
   dnsmos, dnsmos(personalized=True) and dnsmos_p808(librosa='0.9') against
   its deep_noise_suppression_mean_opinion_score, which is pointed at DIR
   so that it downloads nothing.

It prints the largest differences and exits with 1 if one is past its
tolerance.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np

SR = 16000
REFERENCE_FLAG = '--serve-reference'


# --- the reference half: dnsmos_local.py, restated, in its own interpreter -------

def serve_reference(folder: Path, models) -> None:
    """dnsmos_local.py's ComputeScore on every clip listed in clips.json, as
    the script reads; the P.808 input of each window saved beside it."""
    import librosa
    import soundfile as sf

    SAMPLING_RATE = 16000
    INPUT_LENGTH = 9.01

    def audio_melspec(audio, n_mels=120, frame_size=320, hop_length=160, sr=16000,
                      to_db=True):
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=frame_size + 1,
                                                  hop_length=hop_length, n_mels=n_mels)
        if to_db:
            mel_spec = (librosa.power_to_db(mel_spec, ref=np.max) + 40) / 40
        return mel_spec.T

    def get_polyfit_val(sig, bak, ovr, is_personalized_MOS):
        if is_personalized_MOS:
            p_ovr = np.poly1d([-0.00533021, 0.005101, 1.18058466, -0.11236046])
            p_sig = np.poly1d([-0.01019296, 0.02751166, 1.19576786, -0.24348726])
            p_bak = np.poly1d([-0.04976499, 0.44276479, -0.1644611, 0.96883132])
        else:
            p_ovr = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
            p_sig = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
            p_bak = np.poly1d([-0.13166888, 1.60915514, -0.39604546])
        return p_sig(sig), p_bak(bak), p_ovr(ovr)

    sessions = {}
    if models:
        import onnxruntime as ort
        root = Path(models)
        sessions = {'plain': ort.InferenceSession(str(root / 'DNSMOS' / 'sig_bak_ovr.onnx')),
                    'personal': ort.InferenceSession(str(root / 'pDNSMOS' / 'sig_bak_ovr.onnx')),
                    'p808': ort.InferenceSession(str(root / 'DNSMOS' / 'model_v8.onnx'))}

    results = {}
    for name in json.loads((folder / 'clips.json').read_text(encoding='utf-8')):
        aud, input_fs = sf.read(str(folder / f'{name}.wav'))
        assert input_fs == SAMPLING_RATE
        audio = aud
        len_samples = int(INPUT_LENGTH * SAMPLING_RATE)
        while len(audio) < len_samples:
            audio = np.append(audio, audio)
        num_hops = int(np.floor(len(audio) / SAMPLING_RATE) - INPUT_LENGTH) + 1
        hop_len_samples = SAMPLING_RATE
        rows = {key: [] for key in ('SIG_raw', 'BAK_raw', 'OVRL_raw', 'SIG', 'BAK', 'OVRL',
                                    'pSIG', 'pBAK', 'pOVRL', 'P808_MOS')}
        inputs = []
        for idx in range(num_hops):
            audio_seg = audio[int(idx * hop_len_samples): int((idx + INPUT_LENGTH) * hop_len_samples)]
            if len(audio_seg) < len_samples:
                continue
            input_features = np.array(audio_seg).astype('float32')[np.newaxis, :]
            p808_input_features = np.array(audio_melspec(audio=audio_seg[:-160])).astype(
                'float32')[np.newaxis, :, :]
            inputs.append(p808_input_features[0])
            if not sessions:
                continue
            oi = {'input_1': input_features}
            p808_oi = {'input_1': p808_input_features}
            rows['P808_MOS'].append(sessions['p808'].run(None, p808_oi)[0][0][0])
            raw = sessions['plain'].run(None, oi)[0][0]
            mapped = get_polyfit_val(*raw, False)
            for key, value in zip(('SIG_raw', 'BAK_raw', 'OVRL_raw', 'SIG', 'BAK', 'OVRL'),
                                  (*raw, *mapped)):
                rows[key].append(value)
            personal = get_polyfit_val(*sessions['personal'].run(None, oi)[0][0], True)
            for key, value in zip(('pSIG', 'pBAK', 'pOVRL'), personal):
                rows[key].append(value)
        np.save(str(folder / f'{name}.p808in.npy'), np.stack(inputs))
        results[name] = {'windows': len(inputs)}
        if sessions:
            results[name].update({key: float(np.mean(values)) for key, values in rows.items()})
    import onnxruntime
    versions = {'python': sys.version.split()[0], 'numpy': np.__version__,
                'librosa': librosa.__version__, 'onnxruntime': onnxruntime.__version__}
    (folder / 'reference.json').write_text(json.dumps({'versions': versions, 'clips': results}),
                                           encoding='utf-8')


# --- the kudio half --------------------------------------------------------------

RESULTS = []


def report(check, cases, diff, tolerance):
    RESULTS.append((check, cases, diff, tolerance))


def clips():
    """``(name, y)`` at 16 kHz, every one under 17 s."""
    import kudio
    from quality import mix, noise, speech
    from scipy.signal import butter, sosfiltfilt
    for seconds in (2.0, 9.5, 12.3, 15.5):
        clean = speech(SR, seconds, seed=1)
        n = clean.size
        yield f'{seconds:g}s clean', clean
        yield f'{seconds:g}s white 5 dB', mix(clean, noise(n, 2), 5)
        yield f'{seconds:g}s pink 10 dB', mix(clean, noise(n, 3, 'pink'), 10)
        yield f'{seconds:g}s second talker 0 dB', mix(clean, speech(SR, seconds, seed=7, floor=0), 0)
        room = np.asarray(kudio.apply_rir(clean, kudio.rir(SR, rt60=0.4, seed=3)), float)[:n]
        yield f'{seconds:g}s room 0.4 s', room
        yield f'{seconds:g}s low-passed 4 kHz', sosfiltfilt(butter(8, 4000, fs=SR, output='sos'),
                                                            clean)
        pad = np.zeros(SR // 2)
        yield f'{seconds:g}s silence either side', np.r_[pad, mix(clean, noise(n, 4), 10), pad]


def modern_features(window):
    """dnsmos_local.py's audio_melspec under the librosa of this interpreter."""
    import librosa
    mel_spec = librosa.feature.melspectrogram(y=window[:-160], sr=SR, n_fft=321,
                                              hop_length=160, n_mels=120)
    return ((librosa.power_to_db(mel_spec, ref=np.max) + 40) / 40).T.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--reference-python', required=True,
                        help="an interpreter with dnsmos_local.py's pinned packages")
    parser.add_argument('--models', help="a DNS-Challenge checkout's DNSMOS folder")
    args = parser.parse_args()

    start = time.perf_counter()
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import kudio
    import soundfile as sf
    module = sys.modules['kudio.core.dnsmos']

    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        signals = dict(clips())
        for name, y in signals.items():
            sf.write(str(folder / f'{name}.wav'), y, SR, subtype='DOUBLE')
        (folder / 'clips.json').write_text(json.dumps(list(signals)), encoding='utf-8')
        command = [args.reference_python, '-I', str(Path(__file__).resolve()), REFERENCE_FLAG,
                   str(folder)] + (['--models', str(Path(args.models).resolve())]
                                   if args.models else [])
        run = subprocess.run(command, capture_output=True, text=True)
        if run.returncode:
            print(run.stdout + run.stderr)
            print("the reference interpreter failed")
            return 1
        reference = json.loads((folder / 'reference.json').read_text(encoding='utf-8'))
        theirs_in = {name: np.load(str(folder / f'{name}.p808in.npy')) for name in signals}

    import onnxruntime
    import scipy
    print(f"kudio {kudio.__version__} from {Path(kudio.__file__).parent}; numpy "
          f"{np.__version__}, scipy {scipy.__version__}, onnxruntime "
          f"{onnxruntime.__version__}, Python {sys.version.split()[0]}")
    versions = reference['versions']
    print(f"dnsmos_local.py under Python {versions['python']}, numpy {versions['numpy']}, "
          f"librosa {versions['librosa']}, onnxruntime {versions['onnxruntime']}\n")

    worst_in, windows, layouts = 0.0, 0, 0
    for name, y in signals.items():
        ours = np.stack([module._p808_features(w) for w in module._segments(y, SR)])
        theirs = theirs_in[name]
        layouts += ours.shape != theirs.shape
        if ours.shape == theirs.shape:
            windows += len(ours)
            worst_in = max(worst_in, float(np.max(np.abs(ours - theirs))))
    report("clips whose windows differ in number", len(signals), float(layouts), 0.0)
    report("P.808 input vs audio_melspec under librosa 0.8.1", windows, worst_in, 1e-6)

    modern = None
    try:
        import librosa
        if tuple(int(part) for part in librosa.__version__.split('.')[:2]) >= (0, 9):
            modern = librosa.__version__
    except ImportError:
        pass
    if modern:
        worst_new = 0.0
        for y in signals.values():
            for w in module._segments(np.asarray(y, float), SR):
                worst_new = max(worst_new, float(np.max(np.abs(
                    module._p808_features(w, True) - modern_features(w)))))
        report(f"P.808 input, librosa='0.9', vs audio_melspec under librosa {modern}",
               windows, worst_new, 1e-6)
    else:
        print("no librosa 0.9 or later here: librosa='0.9' is not compared")

    if not args.models:
        print("no --models: the scores are not compared")
    else:
        paths = {kind: kudio.find_dnsmos_model(args.models, model=kind)
                 for kind in ('p835', 'personalized', 'p808')}
        sessions = {kind: module._session(path) for kind, path in paths.items()}
        keys = {'sig': ('SIG', 'pSIG', 'SIG_raw'), 'bak': ('BAK', 'pBAK', 'BAK_raw'),
                'ovrl': ('OVRL', 'pOVRL', 'OVRL_raw')}
        worst = {'mapped': 0.0, 'raw': 0.0, 'personal': 0.0, 'p808': 0.0, 'modern': 0.0,
                 'apart': 0.0, 'torchmetrics': 0.0}
        try:
            import torch
            import torchmetrics
            import torchmetrics.functional.audio.dnsmos as tm
            tm.DNSMOS_DIR = str(Path(args.models).resolve())
        except ImportError:
            tm = None
        width = max(map(len, signals))
        print(f"{'clip':<{width}}  {'SIG':>5} {'BAK':>5} {'OVRL':>5} {'pOVRL':>5} {'P.808':>5}")
        for name, y in signals.items():
            want = reference['clips'][name]
            plain = kudio.dnsmos(y, SR, session=sessions['p835'])
            raw = kudio.dnsmos(y, SR, session=sessions['p835'], polyfit=False)
            personal = kudio.dnsmos(y, SR, session=sessions['personalized'], personalized=True)
            p808 = kudio.dnsmos_p808(y, SR, session=sessions['p808'])
            for attr, (mapped_key, personal_key, raw_key) in keys.items():
                worst['mapped'] = max(worst['mapped'], abs(getattr(plain, attr) - want[mapped_key]))
                worst['raw'] = max(worst['raw'], abs(getattr(raw, attr) - want[raw_key]))
                worst['personal'] = max(worst['personal'],
                                        abs(getattr(personal, attr) - want[personal_key]))
            worst['p808'] = max(worst['p808'], abs(p808 - want['P808_MOS']))
            if modern:
                name_in = sessions['p808'].get_inputs()[0].name
                scores = [float(sessions['p808'].run(None, {name_in: modern_features(w)[None]})[0]
                                .reshape(-1)[0]) for w in module._segments(
                                    np.asarray(y, float), SR)]
                newer = kudio.dnsmos_p808(y, SR, session=sessions['p808'], librosa='0.9')
                worst['modern'] = max(worst['modern'], abs(newer - float(np.mean(scores))))
                worst['apart'] = max(worst['apart'], abs(newer - p808))
            if tm is not None and modern:
                for personalized, score in ((False, plain), (True, personal)):
                    theirs = tm.deep_noise_suppression_mean_opinion_score(
                        torch.from_numpy(np.array(y, float)), SR, personalized).numpy()
                    ours = np.array([newer, score.sig, score.bak, score.ovrl])
                    worst['torchmetrics'] = max(worst['torchmetrics'],
                                                float(np.max(np.abs(ours - theirs))))
            print(f"{name:<{width}}  {plain.sig:5.2f} {plain.bak:5.2f} {plain.ovrl:5.2f} "
                  f"{personal.ovrl:5.2f} {p808:5.2f}")
        n = len(signals)
        report("SIG, BAK, OVRL vs the script's", n, worst['mapped'], 1e-4)
        report("raw outputs (polyfit=False) vs the script's", n, worst['raw'], 1e-4)
        report("personalised SIG, BAK, OVRL vs the script's -p", n, worst['personal'], 1e-4)
        report("P.808 vs the script's P808_MOS", n, worst['p808'], 1e-4)
        if modern:
            report(f"P.808, librosa='0.9', vs the script under librosa {modern}", n,
                   worst['modern'], 1e-4)
            report("  P.808 on librosa 0.9's input vs on 0.8.1's", n, worst['apart'],
                   math.inf)
        if tm is not None and modern:
            report(f"P.808 (librosa='0.9'), SIG, BAK, OVRL and personalised vs "
                   f"torchmetrics {torchmetrics.__version__}", n, worst['torchmetrics'], 1e-4)
        else:
            print("no torchmetrics, or no librosa 0.9 or later, here: torchmetrics is "
                  "not compared")

    width = max(len(r[0]) for r in RESULTS)
    print(f"\n{'check':<{width}} {'cases':>6} {'max |diff|':>11} {'limit':>8}")
    failed = False
    for check, n, diff, tolerance in RESULTS:
        over = diff > tolerance
        failed |= over
        limit = '-' if math.isinf(tolerance) else f"{tolerance:.0e}"
        print(f"{check:<{width}} {n:>6} {diff:>11.2e} {limit:>8}{'  PAST' if over else ''}")
    print(f"\n{time.perf_counter() - start:.0f} s; "
          + ("SOME DIFFERENCES ARE PAST TOLERANCE" if failed
             else "every difference within its tolerance"))
    return 1 if failed else 0


if __name__ == '__main__':
    if len(sys.argv) > 2 and sys.argv[1] == REFERENCE_FLAG:
        rest = sys.argv[3:]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            serve_reference(Path(sys.argv[2]), rest[1] if rest[:1] == ['--models'] else None)
        sys.exit(0)
    sys.exit(main())
