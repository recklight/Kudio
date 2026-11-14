# -*- coding: utf-8 -*-
"""DNSMOS wiring.

kudio does not ship the ONNX weights, so what is tested here is everything
around them: windowing, resampling, aggregation, the MOS mapping and the error
messages. The model itself is a stub whose outputs are known, which is the only
way to assert the arithmetic at all.
"""
import importlib
import json
import math
import warnings

import numpy as np
import pytest

import kudio
from kudio.core.dnsmos import (
    ENV_VAR,
    POLYFIT,
    POLYFIT_PERSONALIZED,
    SEGMENT_SECONDS,
    _load_polyfit,
    _p808_features,
)
from kudio.exceptions import FeatureError

# `from kudio.core import dnsmos` gives the *function*: kudio's flat API
# rebinds the submodule name. importlib reaches the module itself.
module = importlib.import_module("kudio.core.dnsmos")

SR = 16000


class StubSession:
    """Stands in for onnxruntime.InferenceSession."""

    def __init__(self, raw=(2.0, 3.0, 2.5), shape=None):
        self.raw = raw
        self.shape = shape
        self.calls = []

    def get_inputs(self):
        class _Input:
            name = "input_1"
        first = _Input()
        if self.shape is not None:
            first.shape = self.shape
        return [first]

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
    coefficients, from_file = _load_polyfit(model, POLYFIT)
    assert coefficients["sig"] == (0, 1, 0) and from_file


def test_incomplete_polyfit_override_is_an_error(tmp_path):
    model = tmp_path / "sig_bak_ovr.onnx"
    model.write_bytes(b"")
    (tmp_path / "polyfit.json").write_text(json.dumps({"sig": [0, 1, 0]}),
                                           encoding="utf-8")
    with pytest.raises(FeatureError, match="missing coefficients"):
        _load_polyfit(model, POLYFIT)


@pytest.mark.parametrize("function, session", [
    (kudio.dnsmos, StubSession()), (kudio.dnsmos_p808, StubSession((3.1,)))])
def test_input_is_held_to_the_metrics_rule(function, session):
    name = function.__name__
    with pytest.raises(ValueError, match=rf"^{name}\(\): y has shape \(16000, 2\), more than one channel"):
        function(np.zeros((SR, 2), dtype=np.float32), SR, session=session)
    with pytest.raises(ValueError, match=rf"^{name}\(\): y has 1 sample\(s\) that are not finite"):
        function(np.r_[_noise(1), np.nan], SR, session=session)
    with pytest.raises(ValueError, match=rf"^{name}\(\): y is int64"):
        function(np.ones(SR, dtype=np.int64), SR, session=session)
    with pytest.raises(ValueError, match=rf"^{name}\(\): sr has to be a positive sample rate"):
        function(_noise(1), 0, session=session)
    assert session.calls == []


@pytest.mark.parametrize("function", [kudio.dnsmos, kudio.dnsmos_p808])
def test_an_empty_clip_is_nan_with_one_warning(function):
    session = StubSession()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = function(np.zeros(0, dtype=np.float32), SR, session=session)
    said = [str(w.message) for w in caught]
    assert said == [f"{function.__name__}(): the clip is empty; scored nan"]
    assert caught[0].filename == __file__
    if function is kudio.dnsmos:
        assert math.isnan(result.sig) and math.isnan(result.bak) and math.isnan(result.ovrl)
        assert result.segments == 0
    else:
        assert math.isnan(result)
    assert session.calls == []


def test_int16_held_as_int32_is_scored_with_a_warning():
    y = (_noise(10) * 32767).astype(np.int16).astype(np.int32)
    with pytest.warns(UserWarning, match=r"^dnsmos\(\): y peaks at -\d+\.\d dBFS"):
        score = kudio.dnsmos(y, SR, session=StubSession())
    assert score.segments == 1


@pytest.mark.parametrize("kwargs, said", [
    ({'polyfit': 'yes'}, "polyfit has to be True or False"),
    ({'personalized': 1}, "personalized has to be True or False")])
def test_flags_have_to_be_booleans(kwargs, said):
    with pytest.raises(ValueError, match=rf"^dnsmos\(\): {said}"):
        kudio.dnsmos(_noise(10), SR, session=StubSession(), **kwargs)


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


# -- the personalised model -----------------------------------------------------
def test_the_personalised_model_has_its_own_mapping():
    score = kudio.dnsmos(_noise(10), SR, session=StubSession((2.0, 3.0, 2.5)),
                         personalized=True)
    for key, raw in zip(("sig", "bak", "ovrl"), (2.0, 3.0, 2.5)):
        assert getattr(score, key) == pytest.approx(
            float(np.polyval(POLYFIT_PERSONALIZED[key], raw)), rel=1e-12)
    assert score.ovrl != pytest.approx(float(np.polyval(POLYFIT["ovrl"], 2.5)))


def _checkout(tmp_path):
    """DNS-Challenge's layout: DNSMOS/DNSMOS holds the P.835 and P.808
    models, DNSMOS/pDNSMOS the personalised one."""
    for folder, name in (("DNSMOS", "sig_bak_ovr.onnx"), ("DNSMOS", "model_v8.onnx"),
                         ("pDNSMOS", "sig_bak_ovr.onnx")):
        (tmp_path / "DNSMOS" / folder).mkdir(parents=True, exist_ok=True)
        (tmp_path / "DNSMOS" / folder / name).write_bytes(b"")
    return tmp_path / "DNSMOS"


def test_each_model_is_found_in_a_checkout(tmp_path):
    checkout = _checkout(tmp_path)
    assert kudio.find_dnsmos_model(checkout) == checkout / "DNSMOS" / "sig_bak_ovr.onnx"
    assert (kudio.find_dnsmos_model(checkout, model="personalized")
            == checkout / "pDNSMOS" / "sig_bak_ovr.onnx")
    assert kudio.find_dnsmos_model(checkout, model="p808") == checkout / "DNSMOS" / "model_v8.onnx"
    with pytest.raises(ValueError, match=r"^find_dnsmos_model\(\): model has to be one of"):
        kudio.find_dnsmos_model(checkout, model="p862")


def test_a_model_from_the_other_folder_is_run_with_a_warning(tmp_path, monkeypatch):
    """The two P.835 models share a file name, so a folder named directly can
    hold the wrong one for the mapping asked for."""
    checkout = _checkout(tmp_path)
    monkeypatch.setattr(module, "_session", lambda path: StubSession())
    with pytest.warns(UserWarning, match=r"^dnsmos\(\): .*pDNSMOS.* is the personalised model"):
        kudio.dnsmos(_noise(10), SR, model_dir=checkout / "pDNSMOS")
    with pytest.warns(UserWarning, match=r"^dnsmos\(\): .* is not in a pDNSMOS folder"):
        kudio.dnsmos(_noise(10), SR, model_dir=checkout / "DNSMOS", personalized=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        kudio.dnsmos(_noise(10), SR, model_dir=checkout, personalized=True)
        kudio.dnsmos(_noise(10), SR, model_dir=checkout)
        kudio.dnsmos(_noise(10), SR, model_dir=checkout / "pDNSMOS", polyfit=False)


# -- P.808 ------------------------------------------------------------------------
def _harmonic_window():
    """9.01 s of 60 harmonics on a 110 Hz fundamental with a vibrato, under a
    2.5 Hz tremolo: numpy alone, so the same in any environment."""
    t = np.arange(int(SEGMENT_SECONDS * SR)) / SR
    phase = 2 * np.pi * (110 * t - 10 / np.pi * np.cos(np.pi * t))
    return 0.02 * sum(np.sin(k * phase + k * k) / k for k in range(1, 61)) * (
        1.2 + np.sin(2 * np.pi * 2.5 * t))


#: dnsmos_local.py's audio_melspec(audio_seg[:-160]) on _harmonic_window(),
#: cast to float32 as the script casts it, computed under the environment its
#: requirements.txt pins: librosa 0.8.1, numpy 1.22.4
LIBROSA_081 = {
    0: [-1.0, 0.8372141718864441, 0.2771497368812561, 0.5575698018074036,
        0.07338902354240417, 0.6905775666236877, 0.2510494589805603, 0.4246240258216858],
    -1: [-1.0, 0.6652032732963562, 0.10513883829116821, 0.8904053568840027,
         0.4062246084213257, 0.7667784690856934, 0.32725033164024353, 0.6879033446311951],
}
LIBROSA_081_FRAME_450 = [-1.0, 0.688685953617096, 0.30795931816101074, 0.347517728805542,
                         0.32827216386795044, 0.21663421392440796, 0.12698712944984436,
                         0.037435147911310196]
LIBROSA_081_MEAN = 0.04180118817195744
#: the sums over the 120 bands of frames 0, 1, 898 and 899, where the edges
#: are padded
LIBROSA_081_EDGE_SUMS = {0: 5.733226543292403, 1: 14.086659304564819,
                         898: 16.8007192799123, 899: 13.179412352619693}


def test_the_p808_input_is_librosa_081s():
    """The first and last frames are where 0.8.1's reflection and the zeros
    of librosa 0.9 and later part ways; frame 450 is on neither edge."""
    features = _p808_features(_harmonic_window())
    assert features.shape == (900, 120) and features.dtype == np.float32
    for frame, want in LIBROSA_081.items():
        np.testing.assert_allclose(features[frame, :8], want, rtol=0, atol=1e-6)
    np.testing.assert_allclose(features[450, ::15], LIBROSA_081_FRAME_450, rtol=0, atol=1e-6)
    for frame, want in LIBROSA_081_EDGE_SUMS.items():
        assert float(features[frame].astype(np.float64).sum()) == pytest.approx(want, abs=2e-6)
    assert float(features.astype(np.float64).mean()) == pytest.approx(LIBROSA_081_MEAN, abs=1e-6)


def test_the_lowest_mel_band_is_empty_on_librosa_081s_grid():
    """On 0.8.1's 50 Hz grid no FFT bin falls inside the lowest band, so its
    row is the 80 dB floor, (-80 + 40) / 40, whatever the signal."""
    for window in (_harmonic_window(), _noise(SEGMENT_SECONDS).astype(np.float64)):
        features = _p808_features(window)
        assert np.all(features[:, 0] == -1.0)
        assert np.all(features.max(axis=0)[1:] > -1.0)


def test_digital_silence_is_one_everywhere():
    """Every band at the 1e-10 floor and the reference at it too: 0 dB, so
    (0 + 40) / 40."""
    assert np.all(_p808_features(np.zeros(int(SEGMENT_SECONDS * SR))) == 1.0)


class _P808Recording(StubSession):
    def __init__(self, outputs):
        super().__init__(shape=["N", 900, 120])
        self.outputs = outputs
        self.inputs = []

    def run(self, outputs, feed):
        self.inputs.append(feed["input_1"])
        return [np.array([[self.outputs[(len(self.inputs) - 1) % len(self.outputs)]]],
                         dtype=np.float32)]


def test_p808_scores_every_window_and_averages_them():
    y = _noise(12.01)
    session = _P808Recording([3.0, 3.5, 2.0])
    score = kudio.dnsmos_p808(y, SR, session=session)
    assert score == pytest.approx(np.mean([3.0, 3.5, 2.0]), abs=1e-6)
    assert len(session.inputs) == 3
    for i, fed in enumerate(session.inputs):
        assert fed.shape == (1, 900, 120) and fed.dtype == np.float32
        window = y[i * SR:i * SR + int(SEGMENT_SECONDS * SR)].astype(np.float64)
        np.testing.assert_array_equal(fed[0], _p808_features(window))


def test_p808_resamples_like_the_p835_model():
    session = _P808Recording([3.0])
    kudio.dnsmos_p808(_noise(12, sr=44100), 44100, session=session)
    assert len(session.inputs) == 3 and session.inputs[0].shape == (1, 900, 120)


def test_p808_finds_its_own_model(tmp_path, monkeypatch):
    checkout = _checkout(tmp_path)
    opened = []
    monkeypatch.setattr(module, "_session",
                        lambda path: opened.append(path) or _P808Recording([3.0]))
    kudio.dnsmos_p808(_noise(10), SR, model_dir=checkout)
    assert opened == [checkout / "DNSMOS" / "model_v8.onnx"]


def test_a_session_of_the_other_model_is_refused():
    """The P.835 models take the waveform and the P.808 one a mel
    spectrogram; the session's declared input says which it is."""
    with pytest.raises(FeatureError, match=r"^dnsmos_p808\(\): the session's model takes "
                                           r"input of shape \['N', 144160\]"):
        kudio.dnsmos_p808(_noise(10), SR, session=StubSession(shape=["N", 144160]))
    with pytest.raises(FeatureError, match=r"^dnsmos\(\): .*model_v8.onnx is the P.808 model"):
        kudio.dnsmos(_noise(10), SR, session=StubSession(shape=[1, 900, 120]))


#: the same, under librosa 1.0.1, whose defaults are those of 0.9 onwards
LIBROSA_09_EDGE_SUMS = {0: 0.4049201915040612, 1: 9.60272886895109,
                        898: 12.312999225672684, 899: 8.735369139700197}
LIBROSA_09_MEAN = 0.006319285354098459


def test_the_newer_librosa_input_is_also_computed():
    """librosa 0.9 pads with zeros and puts the FFT bins at their true
    frequencies, where the lowest band is no longer empty."""
    features = _p808_features(_harmonic_window(), True)
    for frame, want in LIBROSA_09_EDGE_SUMS.items():
        assert float(features[frame].astype(np.float64).sum()) == pytest.approx(want, abs=2e-6)
    assert float(features.astype(np.float64).mean()) == pytest.approx(LIBROSA_09_MEAN, abs=1e-6)
    assert features[:, 0].max() > -1.0


def test_p808_takes_the_librosa_it_is_asked_for():
    y = _noise(10)
    pinned, newer = _P808Recording([3.0]), _P808Recording([3.0])
    kudio.dnsmos_p808(y, SR, session=pinned)
    kudio.dnsmos_p808(y, SR, session=newer, librosa='0.9')
    window = y[:int(SEGMENT_SECONDS * SR)].astype(np.float64)
    np.testing.assert_array_equal(pinned.inputs[0][0], _p808_features(window))
    np.testing.assert_array_equal(newer.inputs[0][0], _p808_features(window, True))
    with pytest.raises(ValueError, match=r"^dnsmos_p808\(\): librosa has to be '0.8' or '0.9', "
                                         r"got '0.10'"):
        kudio.dnsmos_p808(y, SR, session=_P808Recording([3.0]), librosa='0.10')

