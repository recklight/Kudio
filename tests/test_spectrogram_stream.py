# -*- coding: utf-8 -*-
"""The rolling spectrogram: does it draw the same thing the file transform
would, and does it stop depending on how the stream was chopped up?"""
from __future__ import annotations

import numpy as np
import pytest

import kudio
from kudio.exceptions import FeatureError

from conftest import SR


def sine(freq: float, seconds: float, sr: int = SR, amp: float = 1.0):
    t = np.arange(int(sr * seconds)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def voice(seconds: float, sr: int = SR, seed: int = 3) -> np.ndarray:
    """Something with structure in both axes, at whatever length is wanted.

    `conftest.make_speech` places its bursts at fixed times and needs a clip
    long enough to hold them; these tests care about lengths shorter than that.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(int(sr * seconds)) / sr
    harmonics = sum(np.sin(2 * np.pi * 140.0 * k * t) / k for k in range(1, 10))
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 3.0 * t)
    return (harmonics * envelope / 6 + 0.01 * rng.standard_normal(len(t))
            ).astype(np.float32)


# ------------------------------------------------------------------ geometry

def test_geometry_is_reported_before_any_audio_arrives():
    spec = kudio.SpectrogramStream(SR, n_fft=512, hop_length=256, seconds=2.0)

    assert spec.n_bins == 257
    assert spec.n_columns == 125                    # 2 s / 16 ms
    assert spec.window_seconds == pytest.approx(2.0)
    assert spec.column_seconds == pytest.approx(256 / SR)
    assert spec.latency_seconds == pytest.approx(512 / SR)
    assert spec.freqs[0] == 0.0
    assert spec.freqs[-1] == pytest.approx(SR / 2)


def test_the_window_is_full_width_from_the_start():
    """Otherwise the time axis rescales on every redraw, which reads as the
    picture jittering rather than as the recording starting."""
    spec = kudio.SpectrogramStream(SR, seconds=1.0)
    assert spec.columns().shape == (spec.n_columns, spec.n_bins)
    assert np.all(spec.columns() == spec.floor_db)
    assert spec.filled == 0


def test_to_stft_reports_the_geometry_it_used():
    spec = kudio.SpectrogramStream(8000, n_fft=256, hop_length=64)
    stft = spec.to_stft()
    assert (stft.sr, stft.n_fft, stft.hop_length) == (8000, 256, 64)


# ------------------------------------------------------------- what it draws

def test_columns_match_an_uncentred_offline_stft():
    """The whole promise: live and offline see the same audio the same way.

    Compared against ``center=False`` -- a stream cannot reflect-pad samples it
    has not heard yet, so it is that convention and not the file default.
    """
    librosa = pytest.importorskip("librosa")
    y = voice(2.0)
    spec = kudio.SpectrogramStream(SR, n_fft=512, hop_length=256, seconds=10.0)

    live = spec.push(y)
    offline = np.abs(librosa.stft(y, n_fft=512, hop_length=256,
                                  window='hann', center=False)).T
    scale = 2.0 / np.sum(np.hanning(513)[:-1])      # scipy's periodic hann
    expected = np.maximum(20 * np.log10(np.maximum(offline * scale, 1e-10)),
                          spec.floor_db)

    assert live.shape == expected.shape
    # DC and Nyquist are scaled single-sided differently on purpose; the rest
    # is the comparison that matters.
    assert np.allclose(live[:, 1:-1], expected[:, 1:-1], atol=0.05)


def test_block_size_does_not_change_the_answer():
    """A device hands over whatever block size it feels like. If that changed
    the picture, two identical takes would draw differently."""
    y = voice(1.5)
    reference = kudio.SpectrogramStream(SR, seconds=5.0).push(y)

    for size in (1, 97, 128, 512, 4096):
        spec = kudio.SpectrogramStream(SR, seconds=5.0)
        chunks = [spec.push(y[i:i + size]) for i in range(0, len(y), size)]
        chunks = [c for c in chunks if len(c)]
        assert np.array_equal(np.concatenate(chunks), reference), size


def test_a_full_scale_sine_reads_zero_dbfs():
    """The level beside a peak has to survive someone changing n_fft."""
    for n_fft in (256, 512, 1024):
        spec = kudio.SpectrogramStream(SR, n_fft=n_fft, hop_length=n_fft // 2)
        columns = spec.push(sine(1000.0, 0.5))
        assert columns.max() == pytest.approx(0.0, abs=0.2), n_fft


def test_half_scale_reads_six_db_down():
    spec = kudio.SpectrogramStream(SR)
    full = spec.push(sine(1000.0, 0.3, amp=1.0)).max()
    spec.reset()
    half = spec.push(sine(1000.0, 0.3, amp=0.5)).max()
    assert full - half == pytest.approx(6.02, abs=0.1)


def test_the_peak_lands_on_the_right_frequency():
    spec = kudio.SpectrogramStream(SR, n_fft=1024, hop_length=512)
    columns = spec.push(sine(2000.0, 0.5))
    assert spec.freqs[int(np.argmax(columns[0]))] == pytest.approx(2000.0, abs=20)


def test_silence_sits_on_the_floor_instead_of_amplified_dither():
    spec = kudio.SpectrogramStream(SR, top_db=60.0)
    columns = spec.push(np.zeros(SR // 2, dtype=np.float32))
    assert columns.size
    assert np.all(columns == -60.0)


# ---------------------------------------------------------------- the buffer

def test_frames_are_held_until_they_are_complete():
    """A partial frame is not dropped and not guessed at -- it waits."""
    spec = kudio.SpectrogramStream(SR, n_fft=512, hop_length=256)
    assert spec.push(np.zeros(511, dtype=np.float32)).shape == (0, spec.n_bins)
    assert spec.frames_seen == 0
    assert len(spec.push(np.zeros(1, dtype=np.float32))) == 1
    assert spec.frames_seen == 1


def test_pushing_nothing_is_allowed():
    spec = kudio.SpectrogramStream(SR)
    assert spec.push(np.zeros(0, dtype=np.float32)).shape == (0, spec.n_bins)


def test_the_newest_column_is_on_the_right_while_filling():
    spec = kudio.SpectrogramStream(SR, n_fft=512, hop_length=256, seconds=2.0)
    new = spec.push(sine(1000.0, 0.5))

    window = spec.columns()
    assert np.array_equal(window[-1], new[-1])
    assert np.array_equal(window[-len(new):], new)
    assert np.all(window[:-len(new)] == spec.floor_db)


def test_old_columns_scroll_off_once_the_window_is_full():
    spec = kudio.SpectrogramStream(SR, n_fft=512, hop_length=256, seconds=0.5)
    spec.push(sine(1000.0, 0.4))                 # quiet-ish half, fills part
    late = spec.push(sine(3000.0, 1.0))          # more than the whole window

    window = spec.columns()
    assert spec.filled == spec.n_columns
    assert np.array_equal(window, late[-spec.n_columns:])
    # nothing of the 1 kHz tone survives: it scrolled off
    assert spec.freqs[int(np.argmax(window[0]))] == pytest.approx(3000.0, abs=50)


def test_the_ring_wraps_without_reordering_columns():
    """Push in many small pieces so the write index passes the end repeatedly:
    the picture must still read oldest-to-newest left-to-right."""
    spec = kudio.SpectrogramStream(SR, n_fft=256, hop_length=128, seconds=0.2)
    n = spec.n_columns
    rising = np.concatenate([sine(f, 0.15) for f in (500, 1500, 2500, 3500)])

    seen = []
    for i in range(0, len(rising), 333):
        got = spec.push(rising[i:i + 333])
        if len(got):
            seen.append(got)
    seen = np.concatenate(seen)

    assert spec.filled == n
    assert np.array_equal(spec.columns(), seen[-n:])


def test_latest_is_the_rightmost_column():
    spec = kudio.SpectrogramStream(SR, seconds=0.3)
    assert np.all(spec.latest() == spec.floor_db)
    spec.push(voice(1.0))
    assert np.array_equal(spec.latest(), spec.columns()[-1])


def test_reset_forgets_the_audio_and_keeps_the_settings():
    spec = kudio.SpectrogramStream(SR, n_fft=256, seconds=1.0)
    spec.push(voice(1.0))
    columns, bins = spec.n_columns, spec.n_bins

    spec.reset()
    assert spec.frames_seen == 0 and spec.filled == 0
    assert (spec.n_columns, spec.n_bins) == (columns, bins)
    assert np.all(spec.columns() == spec.floor_db)
    # the part-filled frame went too: a fresh push starts frame-aligned
    assert len(spec.push(np.zeros(spec.n_fft - 1, dtype=np.float32))) == 0


# ------------------------------------------------------------------- mel axis

def test_mel_mode_returns_one_row_per_band():
    spec = kudio.SpectrogramStream(SR, n_mels=64, seconds=1.0)
    columns = spec.push(voice(1.0))

    assert spec.n_bins == 64
    assert columns.shape[1] == 64
    assert len(spec.freqs) == 64
    assert np.all(np.diff(spec.freqs) > 0)
    assert spec.freqs[-1] <= SR / 2 + 1


def test_mel_bands_follow_the_tone_that_is_playing():
    spec = kudio.SpectrogramStream(SR, n_mels=40, seconds=1.0)
    low = int(np.argmax(spec.push(sine(300.0, 0.3))[-1]))
    spec.reset()
    high = int(np.argmax(spec.push(sine(4000.0, 0.3))[-1]))
    assert low < high


def test_fmin_and_fmax_bound_the_mel_axis():
    spec = kudio.SpectrogramStream(SR, n_mels=32, fmin=80.0, fmax=6000.0)
    assert spec.freqs[0] >= 80.0
    assert spec.freqs[-1] <= 6000.0


# -------------------------------------------------------------- refused input

def test_multichannel_is_refused_rather_than_flattened():
    """Flattening interleaved channels analyses a signal that does not exist."""
    spec = kudio.SpectrogramStream(SR)
    with pytest.raises(FeatureError, match="mono"):
        spec.push(np.zeros((1024, 2), dtype=np.float32))


@pytest.mark.parametrize("kwargs", [
    {"n_fft": 1},
    {"hop_length": 0},
    {"seconds": 0},
    {"top_db": 0},
    {"ref": 0},
])
def test_impossible_settings_are_refused_at_construction(kwargs):
    with pytest.raises(FeatureError):
        kudio.SpectrogramStream(SR, **kwargs)


def test_a_bad_rate_is_refused():
    with pytest.raises(FeatureError, match="positive rate"):
        kudio.SpectrogramStream(0)
