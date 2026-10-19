# -*- coding: utf-8 -*-
"""DNSMOS wiring.

kudio does not ship the ONNX weights, so what is tested here is everything
around them: windowing, resampling, aggregation, the MOS mapping and the error
messages. The model itself is a stub whose outputs are known, which is the only
way to assert the arithmetic at all.
"""
import json

import numpy as np
import pytest

import kudio
# `from kudio.core import dnsmos` gives the *function* -- kudio's flat API
# rebinds the submodule name. Reach for the module through its own path.
from kudio.core.dnsmos import (
    ENV_VAR,
    POLYFIT,
    SEGMENT_SECONDS,
    _load_polyfit,
)
from kudio.exceptions import FeatureError

SR = 16000


class StubSession:
    """Stands in for onnxruntime.InferenceSession."""

    def __init__(self, raw=(2.0, 3.0, 2.5)):
        self.raw = raw
        self.calls = []

    class _Input:
        name = "input_1"

    def get_inputs(self):
        return [StubSession._Input()]

    def run(self, outputs, feed):
        self.calls.append(feed["input_1"].shape)
        return [np.array([self.raw], dtype=np.float32)]


def _noise(seconds, sr=SR, seed=0):
    rng = np.random.default_rng(seed)
    return (0.1 * rng.standard_normal(int(seconds * sr))).astype(np.float32)


def test_windows_are_the_length_the_model_expects():
    session = StubSession()
    kudio.dnsmos(_noise(12), SR, session=session)
    expected = int(SEGMENT_SECONDS * SR)
    assert all(shape == (1, expected) for shape in session.calls)


def test_hop_is_one_second():
    session = StubSession()
    score = kudio.dnsmos(_noise(12.01), SR, session=session)
    # windows start on whole seconds and end by the 12th: at 0, 1 and 2 s
    assert score.segments == 3
    assert len(session.calls) == 3


def test_short_clip_is_doubled_not_padded():
    """Zero-padding a two-second clip would score seven seconds of silence;
    doubled to 16 s, as dnsmos_local.py does it, it gives seven windows."""
    session = StubSession()
    score = kudio.dnsmos(_noise(2), SR, session=session)
    assert score.segments == 7
    assert all(shape == (1, int(SEGMENT_SECONDS * SR)) for shape in session.calls)


def test_raw_mode_returns_the_model_output_untouched():
    score = kudio.dnsmos(_noise(10), SR, session=StubSession((2.0, 3.0, 2.5)),
                         polyfit=False)
    assert (score.sig, score.bak, score.ovrl) == (2.0, 3.0, 2.5)


def test_polyfit_maps_onto_the_mos_scale():
    score = kudio.dnsmos(_noise(10), SR, session=StubSession((2.0, 3.0, 2.5)))
    expected = {k: float(np.polyval(v, raw)) for (k, v), raw
                in zip(POLYFIT.items(), (2.0, 3.0, 2.5))}
    assert score.sig == pytest.approx(expected["sig"])
    assert score.bak == pytest.approx(expected["bak"])
    assert score.ovrl == pytest.approx(expected["ovrl"])


def test_audio_is_resampled_to_the_model_rate():
    session = StubSession()
    kudio.dnsmos(_noise(12, sr=44100), 44100, session=session)
    assert session.calls[0] == (1, int(SEGMENT_SECONDS * SR))


def test_summary_names_the_denoising_trade():
    over = kudio.DnsmosScore(sig=2.0, bak=4.0, ovrl=2.5, segments=1)
    assert "over-aggressive" in over.summary()
    under = kudio.DnsmosScore(sig=4.0, bak=2.0, ovrl=2.5, segments=1)
    assert "intrusive" in under.summary()
    fine = kudio.DnsmosScore(sig=4.2, bak=4.3, ovrl=4.2, segments=1)
    assert fine.summary().startswith("clean")


def test_missing_model_says_where_to_get_it(tmp_path, monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    with pytest.raises(FeatureError) as excinfo:
        kudio.find_dnsmos_model(tmp_path)
    message = str(excinfo.value)
    assert "does not ship" in message
    assert "DNS-Challenge" in message


def test_model_is_found_in_a_directory(tmp_path):
    (tmp_path / "sig_bak_ovr.onnx").write_bytes(b"not really a model")
    assert kudio.find_dnsmos_model(tmp_path).name == "sig_bak_ovr.onnx"


def test_model_is_found_by_file_path(tmp_path):
    model = tmp_path / "sig_bak_ovr.onnx"
    model.write_bytes(b"")
    assert kudio.find_dnsmos_model(model) == model


def test_environment_variable_is_the_fallback(tmp_path, monkeypatch):
    (tmp_path / "sig_bak_ovr.onnx").write_bytes(b"")
    monkeypatch.setenv(ENV_VAR, str(tmp_path))
    assert kudio.find_dnsmos_model(None).parent == tmp_path


def test_polyfit_override_is_read(tmp_path):
    model = tmp_path / "sig_bak_ovr.onnx"
    model.write_bytes(b"")
    (tmp_path / "polyfit.json").write_text(
        json.dumps({"sig": [0, 1, 0], "bak": [0, 1, 0], "ovrl": [0, 1, 0]}),
        encoding="utf-8")
    assert _load_polyfit(model)["sig"] == (0, 1, 0)


def test_incomplete_polyfit_override_is_an_error(tmp_path):
    model = tmp_path / "sig_bak_ovr.onnx"
    model.write_bytes(b"")
    (tmp_path / "polyfit.json").write_text(json.dumps({"sig": [0, 1, 0]}),
                                           encoding="utf-8")
    with pytest.raises(FeatureError, match="missing coefficients"):
        _load_polyfit(model)


def test_rejects_stereo_and_empty_audio():
    with pytest.raises(FeatureError, match="mono"):
        kudio.dnsmos(np.zeros((SR, 2), dtype=np.float32), SR,
                     session=StubSession())
    with pytest.raises(FeatureError, match="empty"):
        kudio.dnsmos(np.zeros(0, dtype=np.float32), SR, session=StubSession())


class _Alternating(StubSession):
    """Two windows that disagree, so the order of mean and mapping shows."""

    def __init__(self):
        super().__init__()
        self._outputs = [(1.0, 1.0, 1.0), (4.0, 4.0, 4.0)]

    def run(self, outputs, feed):
        self.calls.append(feed["input_1"].shape)
        return [np.array([self._outputs[(len(self.calls) - 1) % 2]], dtype=np.float32)]


def test_each_window_is_mapped_before_the_mean():
    """DNS-Challenge's dnsmos_local.py maps every window and then averages;
    with a quadratic mapping the other order is a different number."""
    score = kudio.dnsmos(_noise(11), SR, session=_Alternating())
    expected = (np.polyval(POLYFIT["ovrl"], 1.0) + np.polyval(POLYFIT["ovrl"], 4.0)) / 2
    assert score.ovrl == pytest.approx(expected)


def test_the_p808_model_is_not_taken_for_the_p835_one(tmp_path):
    (tmp_path / "model_v8.onnx").write_bytes(b"")
    with pytest.raises(FeatureError, match="not found"):
        kudio.find_dnsmos_model(tmp_path)


# -- the same windows as the reference script -----------------------------------
def _reference_windows(audio, fs=SR):
    """DNS-Challenge, DNSMOS/dnsmos_local.py, ComputeScore.__call__, as written."""
    INPUT_LENGTH = 9.01
    len_samples = int(INPUT_LENGTH * fs)
    while len(audio) < len_samples:
        audio = np.append(audio, audio)
    num_hops = int(np.floor(len(audio) / fs) - INPUT_LENGTH) + 1
    hop_len_samples = fs
    out = []
    for idx in range(num_hops):
        audio_seg = audio[int(idx * hop_len_samples): int((idx + INPUT_LENGTH) * hop_len_samples)]
        if len(audio_seg) < len_samples:
            continue
        out.append(audio_seg)
    return out


class _Recording(StubSession):
    def __init__(self):
        super().__init__()
        self.windows = []

    def run(self, outputs, feed):
        self.windows.append(np.array(feed["input_1"][0]))
        return super().run(outputs, feed)


@pytest.mark.parametrize("seconds", [0.3, 2, 4.6, 9.01, 9.5, 10.5, 12.01, 16.5])
def test_the_windows_are_the_reference_scripts(seconds):
    y = _noise(seconds)
    session = _Recording()
    kudio.dnsmos(y, SR, session=session, polyfit=False)
    expected = _reference_windows(y.astype(np.float64))
    assert len(session.windows) == len(expected)
    for got, want in zip(session.windows, expected):
        np.testing.assert_array_equal(got, want.astype(np.float32))


def test_a_long_clip_keeps_the_windows_the_script_rounds_away():
    """The script's int((idx + 9.01) * fs) comes out one sample short for the
    windows starting at 7 to 10 s of a 20 s clip, and it skips them."""
    y = _noise(20)
    session = _Recording()
    kudio.dnsmos(y, SR, session=session, polyfit=False)
    assert len(_reference_windows(y.astype(np.float64))) == 7
    assert len(session.windows) == 11
    n = int(SEGMENT_SECONDS * SR)
    for i, window in enumerate(session.windows):
        np.testing.assert_array_equal(window, y[i * SR:i * SR + n])
