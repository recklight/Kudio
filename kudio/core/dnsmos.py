# -*- coding: utf-8 -*-
"""DNSMOS: a predicted listener opinion, with no clean reference.

:func:`kudio.audio_report` answers "is this recording technically sound".
It cannot answer "does it sound good" — that is a judgement, and the only
reference-free way to get one without a listening panel is a model trained on
one. DNSMOS (ITU-T P.835) is that model: it returns three scores on the usual
1–5 MOS scale.

* **SIG** — the speech itself, ignoring the background.
* **BAK** — how unobtrusive the background is.
* **OVRL** — overall quality.

The interesting part for enhancement work is that SIG and BAK move in opposite
directions: aggressive denoising raises BAK and lowers SIG, because it eats the
speech along with the noise. A single number hides that trade; three do not.

**kudio does not ship the model.** The weights belong to Microsoft's
DNS-Challenge and are distributed there; bundling ~2 MB of someone else's
binary in a library that is otherwise pure source is not a thing to do quietly.
Point *model_dir* at a checkout:

>>> kudio.dnsmos(y, sr, model_dir="DNS-Challenge/DNSMOS")   # doctest: +SKIP
DnsmosScore(sig=3.42, bak=3.91, ovrl=3.15, segments=4)

Needs ``pip install kudio[dnsmos]`` for onnxruntime.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Any, Optional, Sequence, Union

import numpy as np

from kudio.exceptions import DependencyError, FeatureError

__all__ = ['DnsmosScore', 'dnsmos', 'find_dnsmos_model']

log = logging.getLogger(__name__)

PathLike = Union[str, PurePath]

#: The model was trained at 16 kHz and on this segment length, both of which
#: are part of the model rather than choices we are free to make.
MODEL_SR = 16000
SEGMENT_SECONDS = 9.01
HOP_SECONDS = 1.0

#: File names used by the DNS-Challenge distribution.
MODEL_NAMES = ("sig_bak_ovr.onnx", "model_v8.onnx")
#: Where people usually leave a checkout, tried in order.
ENV_VAR = "KUDIO_DNSMOS_DIR"

#: Published P.835 mapping from the network's raw outputs onto the MOS scale
#: (DNSMOS P.835, non-personalised). They are part of the released model, not a
#: tuning of ours -- which also means kudio cannot verify them without the
#: weights. ``polyfit=False`` returns the raw outputs if you would rather not
#: rely on them, and a ``polyfit.json`` in *model_dir* overrides them.
POLYFIT = {
    "sig": (-0.08397278, 1.22083953, 0.0052439),
    "bak": (-0.13166888, 1.60915514, -0.39604546),
    "ovrl": (-0.06766283, 1.11546468, 0.04602535),
}


@dataclass(frozen=True)
class DnsmosScore:
    """Three P.835 opinion scores, each on the 1–5 MOS scale."""

    #: quality of the speech signal alone
    sig: float
    #: how unobtrusive the background is
    bak: float
    #: overall quality
    ovrl: float
    #: how many 9-second windows were averaged
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


def find_dnsmos_model(model_dir: Optional[PathLike] = None) -> Path:
    """Locate ``sig_bak_ovr.onnx``.

    Looks at *model_dir*, then at ``$KUDIO_DNSMOS_DIR``. Either may be the
    directory holding the file or the file itself.

    :raises FeatureError: with instructions, when nothing is found. Being told
        where to get the model beats a FileNotFoundError on a path you did not
        choose.
    """
    candidates = [model_dir, os.environ.get(ENV_VAR)]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if path.is_file():
            return path
        if path.is_dir():
            for name in MODEL_NAMES:
                found = path / name
                if found.is_file():
                    return found
            deeper = sorted(path.rglob("sig_bak_ovr.onnx"))
            if deeper:
                return deeper[0]
    raise FeatureError(
        "DNSMOS model not found. kudio does not ship it — clone "
        "https://github.com/microsoft/DNS-Challenge and pass "
        "model_dir='DNS-Challenge/DNSMOS', or set "
        f"{ENV_VAR} to that directory.")


def _load_polyfit(model_path: Path) -> dict:
    """Published coefficients, unless the model directory overrides them."""
    override = model_path.parent / "polyfit.json"
    if not override.is_file():
        return dict(POLYFIT)
    import json
    with open(override, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    missing = set(POLYFIT) - set(data)
    if missing:
        raise FeatureError(
            f"{override} is missing coefficients for {sorted(missing)}")
    return {k: tuple(data[k]) for k in POLYFIT}


def _session(model_path: Path) -> Any:
    try:
        import onnxruntime as ort
    except ImportError as e:
        raise DependencyError('onnxruntime', extra='dnsmos') from e
    return ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])


def _segments(y: np.ndarray, sr: int) -> Sequence[np.ndarray]:
    """9.01-second windows at a 1-second hop, as the model expects.

    A clip shorter than one window is tiled rather than zero-padded: silence
    scores as ruined speech, and padding a two-second answer with seven seconds
    of nothing would report a problem that is not there.
    """
    n = int(SEGMENT_SECONDS * sr)
    if y.size < n:
        y = np.tile(y, int(np.ceil(n / max(y.size, 1))))[:n]
    hop = int(HOP_SECONDS * sr)
    return [y[start:start + n] for start in range(0, len(y) - n + 1, hop)]


def dnsmos(y: np.ndarray, sr: int, model_dir: Optional[PathLike] = None, *,
           polyfit: bool = True, session: Optional[Any] = None) -> DnsmosScore:
    """Predict P.835 opinion scores for *y*.

    Mono only. Audio at any rate is resampled to the 16 kHz the model was
    trained at — the rate is a property of the model, not a preference.

    :param model_dir: directory (or file) holding ``sig_bak_ovr.onnx``. See
        :func:`find_dnsmos_model` for the search order.
    :param polyfit: map the network's raw outputs onto the MOS scale using the
        published coefficients. ``False`` returns the raw outputs, which are
        monotonic in quality but not on the 1–5 scale.
    :param session: a pre-built inference session, so a batch job opens the
        model once instead of per file.

    :raises DependencyError: when onnxruntime is not installed.
    :raises FeatureError: on non-mono audio, an empty clip, or a missing model.
    """
    if sr <= 0:
        raise FeatureError(f"dnsmos() needs a positive sample rate, got {sr}")
    y = np.ascontiguousarray(np.asarray(y, dtype=np.float32).squeeze())
    if y.ndim != 1:
        raise FeatureError(f"dnsmos() takes mono audio, got shape {y.shape}")
    if y.size == 0:
        raise FeatureError("dnsmos() got an empty waveform")

    if sr != MODEL_SR:
        from kudio.core.io import resample
        y = resample(y, sr, MODEL_SR).astype(np.float32, copy=False)

    coefficients = dict(POLYFIT)
    if session is None:
        model_path = find_dnsmos_model(model_dir)
        session = _session(model_path)
        coefficients = _load_polyfit(model_path)

    input_name = session.get_inputs()[0].name
    raw = []
    for segment in _segments(y, MODEL_SR):
        features = segment.astype(np.float32)[np.newaxis, :]
        out = session.run(None, {input_name: features})
        raw.append(np.asarray(out[0], dtype=np.float64).reshape(-1)[:3])
    if not raw:                                  # unreachable: _segments tiles
        raise FeatureError("dnsmos() produced no windows")

    sig, bak, ovrl = np.mean(np.vstack(raw), axis=0)
    if polyfit:
        sig = float(np.polyval(coefficients["sig"], sig))
        bak = float(np.polyval(coefficients["bak"], bak))
        ovrl = float(np.polyval(coefficients["ovrl"], ovrl))
    return DnsmosScore(sig=float(sig), bak=float(bak), ovrl=float(ovrl),
                       segments=len(raw))
