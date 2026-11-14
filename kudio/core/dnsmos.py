# -*- coding: utf-8 -*-
"""DNSMOS: a predicted listener opinion, with no clean reference.

:func:`kudio.audio_report` says whether a recording is technically sound.
Whether it sounds good is a judgement, and with no reference and no listening
panel the only source of one is a model trained on a panel's ratings.
Microsoft's DNS-Challenge distributes three, and kudio runs each as the
challenge's ``DNSMOS/dnsmos_local.py`` does:

* DNSMOS P.835 [RGC22], ``sig_bak_ovr.onnx``, trained on ratings collected
  to ITU-T P.835: three scores, each a mean on one of its 1-5 scales, SIG
  for the speech alone, BAK for how unobtrusive the background is and OVRL
  overall, the last on the usual MOS scale. Under
  denoising SIG and BAK move apart, since aggressive suppression takes some
  of the speech with the noise, and one number would hide that.
  :func:`dnsmos`.
* The personalised P.835 model, ``pDNSMOS/sig_bak_ovr.onnx``, which also
  marks down a second talker. It was released with the ICASSP 2023 challenge
  [DAG23]; its mapping onto the MOS scale is in dnsmos_local.py, and I know
  of no paper on how it was trained. ``dnsmos(..., personalized=True)``.
* DNSMOS P.808 [RGC21], ``model_v8.onnx``, trained on ratings collected with
  ITU-T P.808's crowdsourcing method: one overall score, read off a log mel
  spectrogram rather than the waveform. :func:`dnsmos_p808`.

kudio does not ship the models. The weights are Microsoft's and are
distributed with DNS-Challenge, and I would rather not bundle someone else's
binary in a library that is otherwise source only. Point *model_dir* at a
checkout's DNSMOS folder::

    kudio.dnsmos(y, sr, model_dir="DNS-Challenge/DNSMOS")
    kudio.dnsmos_p808(y, sr, model_dir="DNS-Challenge/DNSMOS")

Needs ``pip install kudio[dnsmos]`` for onnxruntime.

References
----------
[RGC21] C. K. A. Reddy, V. Gopal and R. Cutler, "DNSMOS: A non-intrusive
        perceptual objective speech quality metric to evaluate noise
        suppressors", Proc. ICASSP 2021, pp. 6493-6497,
        doi:10.1109/ICASSP39728.2021.9414878.
[RGC22] C. K. A. Reddy, V. Gopal and R. Cutler, "DNSMOS P.835: A
        non-intrusive perceptual objective speech quality metric to evaluate
        noise suppressors", Proc. ICASSP 2022, pp. 886-890,
        doi:10.1109/ICASSP43922.2022.9746108.
[DAG23] H. Dubey, A. Aazami, V. Gopal, B. Naderi, S. Braun, R. Cutler,
        A. Ju, M. Zohourian, M. Tang, H. Gamper, M. Golestaneh and
        R. Aichner, "ICASSP 2023 Deep Noise Suppression Challenge",
        arXiv:2303.11510, 2023.
"""
from __future__ import annotations

import functools
import os
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np

from kudio.core._policy import faint, sample_rate, scored_nan, signal, warn
from kudio.exceptions import DependencyError, FeatureError

__all__ = ['DnsmosScore', 'dnsmos', 'dnsmos_p808', 'find_dnsmos_model']

PathLike = Union[str, PurePath]

#: The models were trained at 16 kHz and on this segment length, both of
#: which are part of the models rather than choices we are free to make.
MODEL_SR = 16000
SEGMENT_SECONDS = 9.01
HOP_SECONDS = 1.0

#: Each model's file and the folder DNS-Challenge keeps it in, under its
#: DNSMOS folder. The two P.835 models share a file name.
MODELS = {
    'p835': ('DNSMOS', 'sig_bak_ovr.onnx'),
    'personalized': ('pDNSMOS', 'sig_bak_ovr.onnx'),
    'p808': ('DNSMOS', 'model_v8.onnx'),
}
#: The environment variable :func:`find_dnsmos_model` reads when no
#: *model_dir* is given: the folder holding the models, or a model itself.
ENV_VAR = "KUDIO_DNSMOS_DIR"

#: dnsmos_local.py's mapping from the P.835 network's raw outputs onto the
#: MOS scale, highest power first. It is part of the released model, not a
#: tuning of mine. ``polyfit=False`` returns the raw outputs instead, and a
#: ``polyfit.json`` beside the model replaces it.
POLYFIT = {
    "sig": (-0.08397278, 1.22083953, 0.0052439),
    "bak": (-0.13166888, 1.60915514, -0.39604546),
    "ovrl": (-0.06766283, 1.11546468, 0.04602535),
}
#: The same for the personalised model, from the same script: cubics.
POLYFIT_PERSONALIZED = {
    "sig": (-0.01019296, 0.02751166, 1.19576786, -0.24348726),
    "bak": (-0.04976499, 0.44276479, -0.1644611, 0.96883132),
    "ovrl": (-0.00533021, 0.005101, 1.18058466, -0.11236046),
}

# dnsmos_local.py's audio_melspec for the P.808 model: 20 ms frames, as the
# paper has them, taken as a 321-point FFT (frame_size + 1), a 10 ms hop and
# 120 mel bands
_P808_N_FFT = 321
_P808_HOP = 160
_P808_MELS = 120


@dataclass(frozen=True)
class DnsmosScore:
    """Three P.835 opinion scores, each on one of its 1-5 scales."""

    #: quality of the speech signal alone
    sig: float
    #: how unobtrusive the background is
    bak: float
    #: overall quality
    ovrl: float
    #: how many 9.01-second windows were averaged
    segments: int

    def __str__(self) -> str:  # pragma: no cover - presentation only
        return (f"SIG {self.sig:.2f} · BAK {self.bak:.2f} · "
                f"OVRL {self.ovrl:.2f}  ({self.segments} window(s))")

    def summary(self) -> str:
        """One line of plain words about what the numbers mean."""
        if self.ovrl >= 4.0:
            verdict = "clean"
        elif self.ovrl >= 3.0:
            verdict = "usable"
        elif self.ovrl >= 2.0:
            verdict = "poor"
        else:
            verdict = "bad"
        note = ""
        if self.bak - self.sig > 0.7:
            note = " — the background is cleaner than the speech, which is what "\
                   "over-aggressive denoising looks like"
        elif self.sig - self.bak > 0.7:
            note = " — the speech survived but the background is still intrusive"
        return f"{verdict} (OVRL {self.ovrl:.2f}){note}"


def find_dnsmos_model(model_dir: Optional[PathLike] = None, *,
                      model: str = 'p835') -> Path:
    """Locate a DNSMOS model.

    Looks at *model_dir*, then at ``$KUDIO_DNSMOS_DIR``. Either may be the
    model file itself or a folder. In a folder the file directly inside wins;
    otherwise its subfolders are searched, and since the two P.835 models
    share a file name, the one in a ``pDNSMOS`` folder comes last for
    ``'p835'`` and first for ``'personalized'``.

    :param model: ``'p835'`` for ``sig_bak_ovr.onnx``, ``'personalized'`` for
        the one in ``pDNSMOS/``, ``'p808'`` for ``model_v8.onnx``.
    :raises FeatureError: with instructions, when nothing is found. Being told
        where to get the model beats a FileNotFoundError on a path you did not
        choose.
    """
    if not isinstance(model, str) or model not in MODELS:
        raise ValueError(f"find_dnsmos_model(): model has to be one of "
                         f"{', '.join(map(repr, MODELS))}, got {model!r}")
    name = MODELS[model][1]
    for candidate in (model_dir, os.environ.get(ENV_VAR)):
        if not candidate:
            continue
        path = Path(candidate)
        if path.is_file():
            return path
        if path.is_dir():
            found = path / name
            if found.is_file():
                return found
            personal = model == 'personalized'
            deeper = sorted(path.rglob(name),
                            key=lambda p: (("pDNSMOS" in p.parts) != personal, str(p)))
            if deeper:
                return deeper[0]
    raise FeatureError(
        f"find_dnsmos_model(): {name} not found. kudio does not ship the DNSMOS "
        f"models; clone https://github.com/microsoft/DNS-Challenge and pass "
        f"model_dir='DNS-Challenge/DNSMOS', or set {ENV_VAR} to that folder")


def _load_polyfit(model_path: Path, default: Dict[str, tuple]) -> Tuple[dict, bool]:
    """*default*, unless a ``polyfit.json`` beside the model replaces it; and
    whether it did."""
    override = model_path.parent / "polyfit.json"
    if not override.is_file():
        return dict(default), False
    import json
    with open(override, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    missing = set(POLYFIT) - set(data)
    if missing:
        raise FeatureError(
            f"{override} is missing coefficients for {sorted(missing)}")
    return {k: tuple(data[k]) for k in POLYFIT}, True


def _session(model_path: Path) -> Any:
    try:
        import onnxruntime as ort
    except ImportError as e:
        raise DependencyError('onnxruntime', extra='dnsmos') from e
    return ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])


def _input_name(session: Any, rank: int, function: str) -> str:
    """The name of the session's input, after checking the model takes what
    *function* feeds it: a waveform (rank 2) for the P.835 models, a mel
    spectrogram (rank 3) for the P.808 one. A session whose input declares
    no shape is taken on trust."""
    first = session.get_inputs()[0]
    shape = getattr(first, 'shape', None)
    if isinstance(shape, (list, tuple)) and shape and len(shape) != rank:
        if rank == 2:
            other = ("the P.835 models take the waveform, (1, 144160); "
                     "model_v8.onnx is the P.808 model, which dnsmos_p808 runs")
        else:
            other = ("the P.808 model takes a mel spectrogram, (1, 900, 120); "
                     "sig_bak_ovr.onnx is a P.835 model, which dnsmos runs")
        raise FeatureError(f"{function}(): the session's model takes input of "
                           f"shape {list(shape)}, and {other}")
    return first.name


def _segments(y: np.ndarray, sr: int) -> Sequence[np.ndarray]:
    """9.01-second windows at a 1-second hop, laid out as DNS-Challenge's
    ``dnsmos_local.py`` lays them out, so a score here is the one that script
    gives the same audio, but for the long clips the last paragraph names.

    A clip shorter than one window is doubled until it holds one -- repeated,
    not zero-padded: silence scores as ruined speech -- and every window of
    the doubled clip is scored, so 2 s becomes 16 s and seven windows.
    Windows start on whole seconds and end inside the clip's last whole
    second, ``int(floor(seconds) - 9.01) + 1`` of them: 10.5 s gets one.

    The script works out each window's end as ``int((i + 9.01) * sr)``, and
    for the windows starting at 7 to 23 s that rounds one sample short, so it
    skips them: 7 of a 20 s clip's 11. Here the end is the start plus the
    window, so a clip that reaches 17 s, after any doubling, gets all of its
    windows; below that the two agree window for window.
    """
    n = int(SEGMENT_SECONDS * sr)
    hop = int(HOP_SECONDS * sr)
    while y.size < n:
        y = np.concatenate([y, y])
    hops = int(np.floor(len(y) / sr) - SEGMENT_SECONDS) + 1
    return [y[i * hop:i * hop + n] for i in range(hops)]


def _prepared(y: Any, sr: Any, function: str) -> Optional[np.ndarray]:
    """*y* checked by the metrics' input rule and at 16 kHz, float64; None
    when it is empty."""
    sr = sample_rate(sr, function)
    y = signal(y, 'y', function)
    if not y.size:
        return None
    faint(function, [('y', y)])
    if sr != MODEL_SR:
        from kudio.core.io import resample
        y = resample(y, sr, MODEL_SR)
    return y


def _flag(value: Any, label: str, function: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{function}(): {label} has to be True or False, got {value!r}")
    return bool(value)


def dnsmos(y: np.ndarray, sr: int, model_dir: Optional[PathLike] = None, *,
           polyfit: bool = True, personalized: bool = False,
           session: Optional[Any] = None) -> DnsmosScore:
    """Predict P.835 opinion scores for *y*.

    Audio at any rate is resampled to the 16 kHz the model was trained at
    and cut into 9.01 s windows at a 1 s hop, laid out as dnsmos_local.py
    lays them out. Each window is scored, its raw outputs are mapped onto
    the MOS scale, and the windows are averaged: map first, then average, as
    the script does, and with a quadratic the two orders differ.

    :param model_dir: folder (or file) holding the model. See
        :func:`find_dnsmos_model` for the search order.
    :param polyfit: map the network's raw outputs onto the MOS scale with
        :data:`POLYFIT`, or with :data:`POLYFIT_PERSONALIZED` for the
        personalised model, or with a ``polyfit.json`` beside the model.
        ``False`` returns the raw outputs, which rise with quality but are
        not on the 1-5 scale.
    :param personalized: run the personalised model, which also marks down a
        second talker, and map it with its own polynomials.
    :param session: a pre-built inference session, so a batch job opens the
        model once instead of per file. The model directory is then never
        read, so neither is a ``polyfit.json`` in it.

    :returns: a :class:`DnsmosScore`; nan in each score, with a warning, for
        an empty clip.
    :raises ValueError: when *y* is not one channel of finite samples, or
        *sr* is not a positive rate.
    :raises FeatureError: when the model is not found, or the session's model
        takes a mel spectrogram, as the P.808 one does.
    :raises DependencyError: when onnxruntime is not installed.
    """
    polyfit = _flag(polyfit, 'polyfit', 'dnsmos')
    personalized = _flag(personalized, 'personalized', 'dnsmos')
    y = _prepared(y, sr, 'dnsmos')
    if y is None:
        nan = scored_nan('dnsmos', "the clip is empty")
        return DnsmosScore(sig=nan, bak=nan, ovrl=nan, segments=0)

    coefficients = dict(POLYFIT_PERSONALIZED if personalized else POLYFIT)
    if session is None:
        model_path = find_dnsmos_model(
            model_dir, model='personalized' if personalized else 'p835')
        coefficients, from_file = _load_polyfit(model_path, coefficients)
        if polyfit and not from_file and ("pDNSMOS" in model_path.parts) != personalized:
            if personalized:
                warn(f"dnsmos(): {model_path} is not in a pDNSMOS folder, so it "
                     f"may be the non-personalised model, which the personalised "
                     f"mapping was not fitted for; point model_dir at the "
                     f"pDNSMOS folder")
            else:
                warn(f"dnsmos(): {model_path} is the personalised model, which "
                     f"the non-personalised mapping was not fitted for; pass "
                     f"personalized=True for its own")
        session = _session(model_path)

    input_name = _input_name(session, 2, 'dnsmos')
    raw = []
    for segment in _segments(y, MODEL_SR):
        out = session.run(None, {input_name: segment.astype(np.float32)[np.newaxis, :]})
        raw.append(np.asarray(out[0], dtype=np.float64).reshape(-1)[:3])
    windows = np.vstack(raw)
    if polyfit:
        windows = np.column_stack([
            np.polyval(coefficients[key], windows[:, i])
            for i, key in enumerate(("sig", "bak", "ovrl"))])
    sig, bak, ovrl = np.mean(windows, axis=0)
    return DnsmosScore(sig=float(sig), bak=float(bak), ovrl=float(ovrl),
                       segments=len(raw))


# -- P.808 ----------------------------------------------------------------------
# dnsmos_local.py gets the P.808 model's input from librosa, and its
# requirements.txt pins librosa 0.8.1. Two of 0.9.0's changes move that input:
# stft pads the signal's ends with zeros where 0.8.1 reflected it, which
# changes the first and last frame, and the FFT bins sit at their true
# frequencies (np.fft.rfftfreq), 16000/321 = 49.84 Hz apart, where 0.8.1 put
# them at linspace(0, 8000, 161), 50 Hz apart, for an odd n_fft. On 0.8.1's
# grid the lowest of the 120 mel bands holds no bin and is always empty; on
# the newer one it is not, and the triangles weigh every bin differently.
# The pin went in on 2023-01-12, four weeks after the P.808 model and this
# part of the script, and it is the only record of the librosa the script
# was meant to run with, so 0.8.1's input is the default. The script under a
# current librosa, and torchmetrics, give the other, and the score moves by
# up to 0.19 between the two (tools/validation/dnsmos.py), so dnsmos_p808
# computes either. Both are restated in numpy, which also means no librosa is
# needed at 16 kHz.

def _hz_to_mel(hz: float) -> float:
    """Slaney's mel scale as librosa 0.8.1's hz_to_mel computes it: linear
    to 1 kHz, logarithmic above, 200/3 Hz a mel."""
    mel = hz / (200.0 / 3)
    if hz >= 1000.0:
        mel = 1000.0 / (200.0 / 3) + np.log(hz / 1000.0) / (np.log(6.4) / 27.0)
    return mel


def _mel_to_hz(mels: np.ndarray) -> np.ndarray:
    hz = 0.0 + (200.0 / 3) * mels
    above = mels >= 1000.0 / (200.0 / 3)
    hz[above] = 1000.0 * np.exp(np.log(6.4) / 27.0 * (mels[above] - 1000.0 / (200.0 / 3)))
    return hz


@functools.lru_cache(maxsize=2)
def _p808_mel_basis(true_bins: bool = False) -> np.ndarray:
    """librosa's ``filters.mel(sr=16000, n_fft=321, n_mels=120)``: triangles
    on Slaney's mel scale from 0 to 8 kHz, each scaled to an area of about
    one (``norm='slaney'``), on 0.8.1's FFT grid, or with *true_bins* on the
    grid of 0.9 and later. Built in a float32 array, so rounded where librosa
    rounds it: once for the triangles, once more for the scaling. Shape
    (120, 161)."""
    fft_hz = (np.fft.rfftfreq(_P808_N_FFT, 1.0 / MODEL_SR) if true_bins else
              np.linspace(0, MODEL_SR / 2, 1 + _P808_N_FFT // 2, endpoint=True))
    mel_hz = _mel_to_hz(np.linspace(_hz_to_mel(0.0), _hz_to_mel(MODEL_SR / 2.0),
                                    _P808_MELS + 2))
    widths = np.diff(mel_hz)
    ramps = np.subtract.outer(mel_hz, fft_hz)
    basis = np.zeros((_P808_MELS, fft_hz.size), dtype=np.float32)
    for i in range(_P808_MELS):
        rising = -ramps[i] / widths[i]
        falling = ramps[i + 2] / widths[i + 1]
        basis[i] = np.maximum(0, np.minimum(rising, falling))
    basis *= (2.0 / (mel_hz[2:_P808_MELS + 2] - mel_hz[:_P808_MELS]))[:, np.newaxis]
    basis.setflags(write=False)
    return basis


@functools.lru_cache(maxsize=1)
def _p808_window() -> np.ndarray:
    from scipy.signal import get_window
    window = get_window('hann', _P808_N_FFT, fftbins=True)
    window.setflags(write=False)
    return window


def _p808_features(window: np.ndarray, newer: bool = False) -> np.ndarray:
    """The P.808 model's input for one 9.01 s window of float64 samples, as
    dnsmos_local.py's audio_melspec computes it under librosa 0.8.1, or with
    *newer* under librosa 0.9 and later.

    The window loses its last 160 samples, is reflected 160 samples out at
    either end (padded with zeros, *newer*), and is cut into 900 frames of
    321 samples at a 160-sample hop under a periodic Hann window. Their
    power spectra go through the mel basis, into dB below the loudest value
    (librosa's power_to_db with ref=np.max: a 1e-10 floor before the
    logarithm and nothing kept more than 80 dB down), and then onto
    (dB + 40) / 40. Shape (900, 120), float32.
    """
    x = np.pad(window[:-_P808_HOP], _P808_N_FFT // 2,
               mode='constant' if newer else 'reflect')
    count = 1 + (x.size - _P808_N_FFT) // _P808_HOP
    frames = x[np.arange(count)[:, np.newaxis] * _P808_HOP + np.arange(_P808_N_FFT)]
    power = np.abs(np.fft.rfft(_p808_window() * frames, axis=-1)) ** 2
    mel = _p808_mel_basis(newer) @ power.T
    db = 10.0 * np.log10(np.maximum(1e-10, mel))
    db -= 10.0 * np.log10(np.maximum(1e-10, mel.max()))
    db = np.maximum(db, db.max() - 80.0)
    return ((db + 40.0) / 40.0).T.astype(np.float32)


def dnsmos_p808(y: np.ndarray, sr: int, model_dir: Optional[PathLike] = None, *,
                librosa: str = '0.8', session: Optional[Any] = None) -> float:
    """Predict the P.808 opinion score of *y*: one overall score on the 1-5
    MOS scale.

    The resampling and the windows are :func:`dnsmos`'s. Each window is
    scored on its log mel spectrogram, 900 frames of 120 bands, and the
    scores are averaged. The network's output is the score; dnsmos_local.py
    maps nothing for this model.

    :param model_dir: folder (or file) holding ``model_v8.onnx``. See
        :func:`find_dnsmos_model` for the search order.
    :param librosa: whose mel spectrogram the model is given. ``'0.8'`` is
        librosa 0.8.1's, the version DNS-Challenge's requirements.txt pins.
        ``'0.9'`` is that of librosa 0.9 and later, the signal padded with
        zeros and the FFT bins at their true frequencies: what dnsmos_local.py
        gives under a current librosa, and what torchmetrics computes. On the
        clips of tools/validation/dnsmos.py the two scores are up to 0.19
        apart.
    :param session: a pre-built inference session of that model, so a batch
        job opens it once instead of per file.

    :returns: the mean over the windows; nan, with a warning, for an empty
        clip.
    :raises ValueError: when *y* is not one channel of finite samples, or
        *sr* is not a positive rate.
    :raises FeatureError: when the model is not found, or the session's model
        takes the waveform, as the P.835 ones do.
    :raises DependencyError: when onnxruntime is not installed.
    """
    if librosa not in ('0.8', '0.9'):
        raise ValueError(f"dnsmos_p808(): librosa has to be '0.8' or '0.9', got {librosa!r}")
    newer = librosa == '0.9'
    y = _prepared(y, sr, 'dnsmos_p808')
    if y is None:
        return scored_nan('dnsmos_p808', "the clip is empty")
    if session is None:
        session = _session(find_dnsmos_model(model_dir, model='p808'))
    input_name = _input_name(session, 3, 'dnsmos_p808')
    scores = []
    for segment in _segments(y, MODEL_SR):
        out = session.run(None, {input_name: _p808_features(segment, newer)[np.newaxis]})
        scores.append(float(np.asarray(out[0], dtype=np.float64).reshape(-1)[0]))
    return float(np.mean(scores))
