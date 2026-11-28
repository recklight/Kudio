# -*- coding: utf-8 -*-
"""What the plots are made of, read back off the axes: the numbers, where they stand
and what the ticks say. Not the pixels, which go by matplotlib's version and fonts."""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import kudio

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from kudio.exceptions import FeatureError  # noqa: E402
from kudio.util import _draw, visual  # noqa: E402

SR = 16000


@pytest.fixture
def ax():
    figure, axes = plt.subplots()
    yield axes
    plt.close(figure)


@pytest.fixture
def figures(monkeypatch):
    """The figures a plot made, which it closes."""
    made = []
    monkeypatch.setattr(visual.plt, "show", lambda *a, **k: None)
    monkeypatch.setattr(visual.plt, "close", lambda *a, **k: made.append(plt.gcf()))
    yield made
    monkeypatch.undo()
    plt.close("all")


def tone(hz, seconds, sr=SR, amplitude=0.3):
    return (amplitude * np.sin(2 * np.pi * hz * np.arange(int(seconds * sr)) / sr)
            ).astype(np.float32)


# -- the curves -------------------------------------------------------------------------

@pytest.mark.parametrize("hz", [440.0, 1000.0, 5000.0])
def test_the_middle_of_a_tones_spectrum_is_the_tone(hz):
    values = _draw.centroid(tone(hz, 1.0), SR)
    assert len(values) == 32
    assert values[3:-3] == pytest.approx(hz, rel=0.01)


def test_a_middle_is_by_the_magnitudes():
    """1000 Hz three times as high as 3000: at 1500 Hz, and by their powers at 1200."""
    values = _draw.centroid(tone(1000, 1.0, amplitude=0.3) + tone(3000, 1.0, amplitude=0.1), SR)
    assert values[3:-3] == pytest.approx(1500, abs=15)


def test_a_level_is_20_db_a_factor_of_ten():
    y = tone(1000, 1.0, amplitude=0.3) + tone(3000, 1.0, amplitude=0.03)
    db = _draw.level(y)[:, 3:-3].mean(axis=1)
    assert db[1000 * 2048 // SR] - db[3000 * 2048 // SR] == pytest.approx(20, abs=0.1)
    assert _draw.level(y, ref=1.0).max() == pytest.approx(20 * np.log10(0.3 * 512), abs=0.1)


def test_the_middle_and_the_roll_off_of_noise():
    """Flat to half the rate: the middle is in the middle, and 85 % of the way up 85 %
    of it lies below. Of silence both are 0."""
    noise = 0.1 * np.random.default_rng(0).standard_normal(10 * SR).astype(np.float32)
    assert np.mean(_draw.centroid(noise, SR)) == pytest.approx(4000, rel=0.01)
    assert np.mean(_draw.rolloff(noise, SR)) == pytest.approx(0.85 * 8000, rel=0.01)
    assert not _draw.centroid(np.zeros(4000, np.float32), SR).any()
    assert not _draw.rolloff(np.zeros(4000, np.float32), SR).any()


def test_the_roll_off_is_the_first_bin_with_the_share_below_it():
    y = tone(1000, 1.0) + tone(3000, 1.0)
    values = _draw.rolloff(y, SR)[3:-3]
    assert values == pytest.approx(3000, abs=8)                # half is under 2 kHz
    assert _draw.rolloff(tone(1000, 1.0) + 0.1 * tone(3000, 1.0), SR)[3:-3] == pytest.approx(
        1000, abs=24)


def test_a_tone_crosses_zero_twice_a_period():
    values = _draw.crossings(tone(250, 1.0) + 1e-4)
    assert len(values) == 32
    assert values[3:-3] == pytest.approx(2 * 250 / SR, abs=1 / 2048)
    # frame by frame, the clip's ends held past them
    y = np.random.default_rng(0).standard_normal(5000)
    padded = np.pad(y, 1024, mode='edge')
    for k, value in enumerate(_draw.crossings(y)):
        frame = padded[k * 512:k * 512 + 2048]
        assert value == np.sum(np.signbit(frame[1:]) != np.signbit(frame[:-1])) / 2048
    assert not _draw.crossings(np.zeros(3000)).any()


def test_a_frame_is_timed_by_its_middle():
    assert _draw.frame_times(3, 16000).tolist() == [0, 0.032, 0.064]


# -- a waveform -------------------------------------------------------------------------

def test_an_envelope_is_the_lowest_and_the_highest_of_each_column():
    y = np.random.default_rng(0).standard_normal(1003)
    for first, stop, columns in ((0, 1003, 100), (17, 900, 33), (5, 6, 10), (0, 1003, 2000)):
        at, low, high = _draw.envelope(y, first, stop, columns)
        assert len(at) <= columns and at[0] == first
        ends = np.r_[at[1:], stop]
        assert [y[a:b].min() for a, b in zip(at, ends)] == low.tolist()
        assert [y[a:b].max() for a, b in zip(at, ends)] == high.tolist()


def test_a_long_clip_is_drawn_in_strokes_and_a_short_one_as_its_samples(ax):
    y = 0.5 * np.random.default_rng(0).standard_normal(100000) + 0.2
    _draw.waveform(ax, y, SR)
    line, = ax.lines
    assert line.get_drawstyle() == 'default' and len(line.get_ydata()) == 10000
    assert line.get_ydata().max() == y.max() and line.get_ydata().min() == y.min()
    assert line.get_xdata()[0] == 0 and line.get_xdata()[-1] <= len(y) / SR
    low, high = line.get_ydata()[0::2], line.get_ydata()[1::2]
    assert np.array_equal(line.get_xdata()[0::2], line.get_xdata()[1::2]) and np.all(low <= high)
    assert ax.get_xlim() == (0, len(y) / SR)
    figure, other = plt.subplots()
    _draw.waveform(other, y[:3000], SR)
    line, = other.lines
    assert line.get_drawstyle() == 'steps-post'
    assert np.array_equal(line.get_ydata(), y[:3000])
    assert line.get_xdata() == pytest.approx(np.arange(3000) / SR)
    plt.close(figure)


def test_zooming_in_comes_to_the_samples_and_out_again_to_the_strokes(ax):
    y = np.random.default_rng(0).standard_normal(200000).astype(np.float32)
    _draw.waveform(ax, y, SR)
    line, = ax.lines
    ax.set_xlim(3.0, 3.1)
    assert line.get_drawstyle() == 'steps-post'
    assert line.get_xdata()[0] <= 3.0 and line.get_xdata()[-1] >= 3.1
    assert np.array_equal(line.get_ydata(), y[int(3.0 * SR):int(3.1 * SR) + 1])
    ax.set_xlim(2.0, 8.0)
    assert line.get_drawstyle() == 'default'
    assert line.get_xdata()[0] == 2.0 and line.get_ydata().max() == y[2 * SR:8 * SR + 1].max()
    ax.set_xlim(8.0, 2.0)                                              # turned round
    assert line.get_drawstyle() == 'default' and line.get_xdata()[0] == 2.0
    ax.set_xlim(50, 60)                                                # past the end
    assert len(line.get_xdata()) == 0


def test_the_axis_of_a_waveform_is_even_about_zero(ax):
    y = np.r_[0.5 * np.ones(10), -0.1 * np.ones(20000)]
    _draw.waveform(ax, y, SR, alpha=0.4, label='it')
    low, high = ax.get_ylim()
    assert low == -high and high > 0.5
    line, = ax.lines
    assert line.get_alpha() == 0.4 and line.get_label() == 'it'


def test_a_waveform_covers_what_it_is_drawn_over(ax):
    """Drawn as a filled band of nine samples a column, a voice would be a tint,
    with most of each column outside it."""
    t = np.arange(45000) / SR
    ax.figure.set_size_inches(8, 2)
    _draw.waveform(ax, 0.5 * np.sin(2 * np.pi * 1500 * t), SR, color='k')
    ax.set_axis_off()
    ax.figure.canvas.draw()
    pixels = np.asarray(ax.figure.canvas.buffer_rgba())[..., 0]
    rows, columns = pixels.shape
    inside = pixels[int(0.4 * rows):int(0.6 * rows), int(0.3 * columns):int(0.7 * columns)]
    assert inside.mean() < 40                                          # 0 is black


# -- time -------------------------------------------------------------------------------

@pytest.mark.parametrize("span, ticks, want", [
    (0.5, [0, 0.1, 0.25], ['0', '0.1', '0.25']),
    (1.3, [0, 0.15, 1.05, 1.2], ['0', '0.15', '1.05', '1.2']),
    (60, [0, 15, 45], ['0', '15', '45']), (61, [0, 10, 61, -10], ['0:00', '0:10', '1:01', '-0:10']),
    (3600, [0, 600, 3599.6], ['0:00', '10:00', '60:00']),
    (3700, [0, 600, 3661], ['0:00:00', '0:10:00', '1:01:01'])])
def test_a_tick_says_its_time(ax, span, ticks, want):
    """1.05 s is not 1.1, and ten seconds before the start is not 0:50."""
    _draw._time_axis(ax, span)
    ax.set_xticks(ticks)
    ax.set_xlim(0, span)
    assert [ax.xaxis.get_major_formatter()(t) for t in ticks] == want


# -- spectrograms -----------------------------------------------------------------------

def test_a_linear_spectrogram_stands_on_its_bins_and_frames(ax):
    y = tone(1000, 1.0)
    db = _draw.level(y)
    assert db.shape == (1025, 32) and db.max() == 0 and db.min() == -80
    mesh = _draw.linear_spectrogram(ax, db, SR, 512, 1.0)
    assert np.array_equal(mesh.get_array().reshape(db.shape), db)
    edges = np.asarray(mesh.get_coordinates())
    assert edges[0, 1:-1, 0] == pytest.approx((np.arange(31) + 0.5) * 512 / SR)
    assert edges[0, 0, 0] == 0 and edges[0, -1, 0] == 1.0
    assert edges[1:-1, 0, 1] == pytest.approx((np.arange(1024) + 0.5) * SR / 2048)
    assert ax.get_xlim() == (0, 1.0) and ax.get_ylim() == (0, 8000)
    # 1 kHz is in the row that has the most in it
    row = int(np.argmax(db.mean(axis=1)))
    assert edges[row, 0, 1] <= 1000 <= edges[row + 1, 0, 1]


def test_a_mel_spectrogram_stands_on_the_middles_of_its_bands(ax):
    """librosa.display puts the lowest of 40 bands at 0 Hz and the highest at half the
    rate, where neither has anything: they peak at 91 and 7377."""
    centres = kudio.mel_band_hz(40, SR)
    for k in (3, 20, 38):
        db = _draw.mel_level(tone(centres[k], 1.0), SR, 40, 8000, 'test')
        mesh = _draw.mel_spectrogram(ax, db, SR, 8000, 1.0)
        assert np.array_equal(mesh.get_array().reshape(db.shape), db)
        edges = mesh.get_coordinates()[:, 0, 1]
        assert len(edges) == 41 and edges[0] == 0 and edges[-1] == 8000
        row = int(np.argmax(db.mean(axis=1)))
        assert row == k and edges[row] < centres[k] < edges[row + 1]
    assert ax.get_yscale() == 'symlog' and ax.get_ylim() == (0, 8000)
    ax.figure.canvas.draw()
    assert [t.get_text() for t in ax.get_yticklabels()] == ['0', '512', '1024', '2048', '4096']


def test_a_spectrogram_of_a_few_samples_fills_its_axes(ax):
    for n in (1, 100, 511, 513, 600):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")                            # fewer than a frame
            db = _draw.level(np.ones(n, np.float32), 512, 256, 'hamming')
        mesh = _draw.linear_spectrogram(ax, db, SR, 256, n / SR)
        at = mesh.get_coordinates()[0, :, 0]
        assert at[0] == 0 and at[-1] == n / SR and np.all(np.diff(at) >= 0)


# -- the plots --------------------------------------------------------------------------

@pytest.fixture
def clip(tmp_path):
    path = tmp_path / "clip.wav"
    t = np.arange(int(1.3 * SR)) / SR
    kudio.save_wave(path, tone(440, 1.3) * (t > 0.5) + 0.001, SR)
    return path


def test_the_curves_stand_at_the_times_of_what_they_measure(clip, figures):
    """The axis of a clip of 1.3 s reaches 1.3 s, not 1 s, and each curve stands
    at the times it measures."""
    visual.plot_3_features(clip)
    axes = figures[0].axes[0]
    assert axes.get_xlim() == (0, 1.3)
    curves = {line.get_label(): line for line in axes.lines}
    assert list(curves) == ['waveform', 'spectral_centroid', 'spectral_rolloff', 'zcr']
    for name in ('spectral_centroid', 'spectral_rolloff', 'zcr'):
        assert curves[name].get_xdata() == pytest.approx(np.arange(41) * 512 / SR)
    zcr = curves['zcr']
    assert zcr.get_ydata()[zcr.get_xdata() < 0.4].max() == 0          # the tone is at 0.5
    assert zcr.get_ydata()[zcr.get_xdata() > 0.6][:-2] == pytest.approx(0.055, abs=0.001)


def test_the_overview_is_a_waveform_a_spectrogram_and_the_crossings(clip, figures):
    visual.basic_analyze(clip)
    wave, spectrum, zcr = figures[0].axes
    assert [a.get_xlim() for a in (wave, spectrum, zcr)] == [(0, 1.3)] * 3
    assert [a.get_title() for a in (wave, spectrum, zcr)] == ['clip', 'STFT', 'ZCR']
    mesh = spectrum.collections[0]
    assert mesh.get_array().reshape(1025, 41).max() > 0 and mesh.get_cmap().name == 'jet'
    assert spectrum.get_ylim() == (0, 8000)


def test_two_mel_spectrograms_of_a_file(clip, figures):
    visual.Common.plot_stft_w_2mfe(clip, n_mels=32, sub_mfe_fq=4000)
    _, whole, sub = figures[0].axes
    assert whole.get_ylim() == (0, 8000) and sub.get_ylim() == (0, 4000)
    assert whole.get_title() == 'Mel Spectrogram 16000Hz n:32'
    assert sub.get_title() == 'Mel Spectrogram 4000Hz n:32'
    for a in (whole, sub):
        data = a.collections[0].get_array()
        assert data.size == 32 * 41 and data.max() == 0
        assert a.collections[0].get_cmap().name == 'magma'


def test_a_band_past_half_the_rate_is_said_at_the_line_that_asked(clip, figures):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        visual.Common.plot_stft_w_2mfe(clip)                           # 48 kHz of 16
    said = [w for w in caught if "plot_stft_w_2mfe()" in str(w.message)]
    assert len(said) == 1 and said[0].filename == __file__
    assert "hold no FFT bin" in str(said[0].message)


PLOTS = {'plot_stft_w_2mfe': visual.Common.plot_stft_w_2mfe,
         'basic_analyze': visual.basic_analyze,
         'plot_3_features': visual.plot_3_features,
         'plot_4_features': visual.plot_4_features}


@pytest.mark.parametrize("name", [*PLOTS, 'WaveVisualizer.plot'])
@pytest.mark.parametrize("what", ["nan", "inf", "empty"])
def test_a_file_that_is_no_sound_is_refused_and_not_drawn(name, what, tmp_path, figures):
    import soundfile as sf
    y = tone(440, 0.5)
    y[100] = {'nan': np.nan, 'inf': np.inf, 'empty': 0}[what]
    sf.write(tmp_path / "bad.wav", y[:0] if what == 'empty' else y, SR, subtype='FLOAT')
    call = PLOTS.get(name) or (lambda f: visual.WaveVisualizer(subplots=2).plot([f, f]))
    before = len(plt.get_fignums())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                                # the reading warns too
        with pytest.raises(FeatureError, match=r"no samples" if what == 'empty' else r"1 sample\(s\) "
                           "that are not finite") as refused:
            call(tmp_path / "bad.wav")
    assert str(refused.value).startswith(f"{name}()")
    assert name == 'WaveVisualizer.plot' or len(plt.get_fignums()) == before


@pytest.mark.parametrize("curve", ['plot_spectral_centroid', 'plot_spectral_rolloff', 'plot_zcr'])
def test_a_curve_is_of_one_channel_of_sound(curve, ax):
    y = tone(440, 0.5)
    broken = y.copy()
    broken[7] = np.nan
    for bad in (broken, y[:0], np.stack([y, y], axis=1), np.stack([y, y]), y + 0j, y > 0,
                (32768 * y).astype(np.int64), np.float32(0.1), y.astype(object)):
        with pytest.raises(FeatureError, match=curve):
            getattr(visual, curve)(ax, bad, SR)
    assert not ax.lines and not ax.get_title()


@pytest.mark.parametrize("curve", ['plot_spectral_centroid', 'plot_spectral_rolloff', 'plot_zcr'])
def test_a_curve_is_the_same_of_a_row_a_column_a_list_and_integers(curve, ax):
    y = np.round(32768 * tone(440, 0.5)) / 32768
    y = y.astype(np.float64)
    for same in (y, y[None], y[:, None], list(y), (32768 * y).astype(np.int16)):
        getattr(visual, curve)(ax, same, SR)
    for line in ax.lines[1:]:
        assert np.array_equal(line.get_ydata(), ax.lines[0].get_ydata())


def test_files_side_by_side(clip, tmp_path, figures):
    other = tmp_path / "other.wav"
    kudio.save_wave(other, tone(300, 2.0), SR)
    drawn = visual.WaveVisualizer(subplots=2)
    drawn.plot([clip, other])
    for row, seconds in zip(drawn.ax, (1.3, 2.0)):
        assert row[0].get_xlim() == row[1].get_xlim() == (0, seconds)
        mesh = row[1].collections[0]
        assert mesh.get_array().reshape(257, -1).max() == 0 and mesh.get_cmap().name == 'jet'
    assert drawn.ax[0][0].get_xlabel() == '' and drawn.ax[1][0].get_xlabel() == 'Time'
    plt.close(drawn.fig)
