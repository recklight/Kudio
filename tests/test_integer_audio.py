# -*- coding: utf-8 -*-
"""An integer array is audio on its type's full scale, wherever it goes in.

Every function here gets one clip twice: as int16, and as the floats
soundfile would read from a 16-bit file of it. The measurements have to agree
and the processing has to give the same audio. Read as plain numbers, an int16
clip measured 90.3 dB louder than the file it came from, 20 * log10(32768),
and every report called it past full scale.
"""
import numpy as np
import pytest

import kudio

SR = 16000


@pytest.fixture(scope="module")
def pair():
    t = np.arange(SR * 4) / SR
    tone = 0.25 * np.sin(2 * np.pi * 220 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 0.7 * t))
    tone += 0.003 * np.random.default_rng(0).standard_normal(t.size)
    as_int = np.round(tone * 32768).astype(np.int16)
    return as_int, as_int / 32768.0          # the same samples, read both ways


def _same(a, b):
    np.testing.assert_allclose(np.asarray(a, dtype=np.float64),
                               np.asarray(b, dtype=np.float64), rtol=1e-6, atol=1e-9)


# -- measurements -----------------------------------------------------------------
@pytest.mark.parametrize("measure", [
    lambda y: kudio.loudness(y, SR),
    lambda y: kudio.true_peak(y, SR),
    lambda y: kudio.loudness_range(y, SR),
    lambda y: kudio.loudness_over_time(y, SR).lufs,
    lambda y: kudio.audio_info(y, SR).peak,
    lambda y: [getattr(kudio.audio_report(y, SR), k) for k in
               ("peak_dbfs", "rms_dbfs", "lufs", "noise_floor_dbfs", "estimated_snr_db")],
    lambda y: kudio.f0(y, SR).f0,
    lambda y: np.ravel(kudio.vad(y, SR)),
    lambda y: kudio.waveform_to_spectrogram(y),
    lambda y: kudio.mfcc(y, SR),
    lambda y: kudio.melspectrogram(y, sr=SR),
    lambda y: np.abs(kudio.STFT(sr=SR).analyse(y)),
    lambda y: kudio.SpectrogramStream(SR).push(y),
], ids=["loudness", "true_peak", "loudness_range", "loudness_over_time",
        "audio_info", "audio_report", "f0", "vad", "spectrogram", "mfcc",
        "melspectrogram", "STFT.analyse", "SpectrogramStream"])
def test_a_measurement_reads_int16_as_the_file_it_came_from(pair, measure):
    as_int, as_float = pair
    _same(measure(as_int), measure(as_float))


def test_a_room_measured_from_an_int16_response():
    ir = kudio.rir(SR, rt60=0.6, drr_db=6.0, seed=0)
    as_int = np.round(ir / np.max(np.abs(ir)) * 32767).astype(np.int16)
    a, b = kudio.rt60(as_int, SR), kudio.rt60(as_int / 32768.0, SR)
    _same([a.t30, a.edt, a.c50, a.drr], [b.t30, b.edt, b.c50, b.drr])


def test_the_report_no_longer_calls_int16_past_full_scale(pair):
    as_int, _ = pair
    report = kudio.audio_report(as_int, SR)
    assert -13 < report.peak_dbfs < -11
    assert not any("past full scale" in p for p in report.problems())


@pytest.mark.parametrize("metric", [kudio.snr, kudio.si_sdr, kudio.segmental_snr,
                                    kudio.sdi])
def test_a_metric_scores_int16_against_float(pair, metric):
    as_int, as_float = pair
    noisy = as_float + 0.02 * np.random.default_rng(1).standard_normal(len(as_float))
    _same(metric(as_int, noisy), metric(as_float, noisy))


def test_dnsmos_hands_the_model_the_float_reading(pair):
    as_int, as_float = pair

    class Session:
        def __init__(self):
            self.windows = []

        def get_inputs(self):
            return [type("Input", (), {"name": "input_1"})()]

        def run(self, outputs, feed):
            self.windows.append(feed["input_1"].copy())
            return [np.array([[3.0, 3.0, 3.0]], dtype=np.float32)]

    a, b = Session(), Session()
    kudio.dnsmos(as_int, SR, session=a)
    kudio.dnsmos(as_float, SR, session=b)
    assert len(a.windows) == len(b.windows) > 0
    for x, y in zip(a.windows, b.windows):
        np.testing.assert_array_equal(x, y)


# -- processing that returns float32 ------------------------------------------------
@pytest.mark.parametrize("process", [
    lambda y: kudio.spectral_enhance(y, SR),
    lambda y: kudio.trad_enhance(y, SR),
    lambda y: kudio.wavelet_low_pass_filter(y),
    lambda y: kudio.bit_depth(y, 8),
    lambda y: kudio.mu_law(y),
    lambda y: kudio.a_law(y),
    lambda y: kudio.telephone(y, SR),
    lambda y: kudio.dropouts(y, SR, loss=0.1, seed=0),
    lambda y: kudio.apply_rir(y, kudio.rir(SR, rt60=0.3, seed=0)),
    lambda y: np.concatenate([kudio.StreamEnhancer(SR).process(y)]),
], ids=["spectral_enhance", "trad_enhance", "wavelet", "bit_depth", "mu_law",
        "a_law", "telephone", "dropouts", "apply_rir", "StreamEnhancer"])
def test_processing_gives_int16_the_audio_it_gives_float(pair, process):
    as_int, as_float = pair
    _same(process(as_int), process(as_float))


def test_compare_enhancers_scores_an_int16_reference(pair):
    as_int, as_float = pair
    noisy = (as_float + 0.02 * np.random.default_rng(2).standard_normal(len(as_float)))
    a = kudio.compare_enhancers(noisy, SR, ["wiener"], reference=as_int, metrics=False)
    b = kudio.compare_enhancers(noisy, SR, ["wiener"], reference=as_float, metrics=False)
    _same(a[0].si_sdr_db, b[0].si_sdr_db)


# -- processing that keeps the input's type ------------------------------------------
def test_levelling_keeps_int16(pair):
    as_int, as_float = pair
    for level in (lambda y: kudio.normalize_lufs(y, SR, lufs=-20.0),
                  lambda y: kudio.match_loudness(y, as_float * 0.5, SR)):
        out = level(as_int)
        assert out.dtype == np.int16
        expected = np.clip(np.round(level(as_float) * 32768), -32768, 32767)
        np.testing.assert_array_equal(out, expected)


def test_trimming_cuts_int16_where_it_cuts_float(pair):
    as_int, as_float = pair
    padded_int = np.concatenate([np.zeros(8000, np.int16), as_int, np.zeros(8000, np.int16)])
    padded_float = padded_int / 32768.0
    trimmed, span = kudio.trim_silence(padded_int)
    assert trimmed.dtype == np.int16 and span == kudio.trim_silence(padded_float)[1]
    parts = kudio.split_on_silence(padded_int)
    assert [len(p) for p in parts] == [len(p) for p in kudio.split_on_silence(padded_float)]
    assert all(p.dtype == np.int16 for p in parts)


def test_uint8_silence_is_its_midpoint():
    """8-bit WAV is unsigned, with silence at 128: read as numbers, the
    silence around a tone was not quiet and trim_silence kept all of it."""
    t = np.arange(SR) / SR
    y = np.concatenate([np.zeros(SR // 2), 0.5 * np.sin(2 * np.pi * 300 * t), np.zeros(SR // 2)])
    as_u8 = np.clip(np.round(y * 128 + 128), 0, 255).astype(np.uint8)
    trimmed, (start, end) = kudio.trim_silence(as_u8)
    assert trimmed.dtype == np.uint8
    # within a frame (2048) of the tone; read as numbers it was (0, 32000)
    assert SR // 2 - 2048 <= start <= SR // 2 and SR * 3 // 2 <= end <= SR * 3 // 2 + 2048
