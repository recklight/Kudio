# -*- coding: utf-8 -*-
"""audio_report: what can be measured with no clean reference to compare to."""
import numpy as np
import pytest

import kudio
from kudio import audio_report, resample
from kudio.exceptions import FeatureError

SR = 16000


def _speech(sr=SR, seconds=3.0, noise=0.02, seed=0):
    """Harmonic stack under a syllable envelope, with a little broadband bed."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(sr * seconds)) / sr
    y = sum(0.4 / k * np.sin(2 * np.pi * 180 * k * t) for k in range(1, 12))
    y *= 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)
    y = y / np.max(np.abs(y)) * 0.6 + noise * rng.standard_normal(len(t))
    return y.astype(np.float32)


def test_a_healthy_clip_reports_nothing_to_fix():
    report = audio_report(_speech(), SR)
    assert report.problems() == []
    assert report.sr == SR
    assert report.duration == pytest.approx(3.0)
    assert -10 < report.peak_dbfs < 0
    assert report.clipped_samples == 0
    assert abs(report.dc_offset) < 1e-3


def test_clipping_is_counted_only_as_runs():
    """One sample touching full scale is luck; a run of them is clipping."""
    y = _speech()
    y[100] = 1.0                                    # a lone spike
    assert audio_report(y, SR).clipped_samples == 0

    y[500:520] = 1.0                                # a real flat top
    report = audio_report(y, SR)
    assert report.clipped_samples == 20
    assert report.clipped_ratio == pytest.approx(20 / len(y))
    assert any("clipped" in p for p in report.problems())


def test_a_signal_past_full_scale_is_not_the_same_as_a_clipped_one():
    """Amplified float audio still traces the waveform -- nothing is lost yet.
    It is heading for trouble on the way to an integer format, which is a
    different, weaker warning than "you already destroyed this"."""
    report = audio_report(_speech() * 2.0, SR)

    assert report.peak_dbfs > 0
    assert report.clipped_samples == 0, "loud is not clipped"
    problems = report.problems()
    assert any("past full scale" in p for p in problems)
    assert not any("clipped" in p for p in problems)

    # ...whereas actually flattening it is
    flattened = audio_report(np.clip(_speech() * 2.0, -1.0, 1.0), SR)
    assert flattened.clipped_samples > 0
    assert any("clipped" in p for p in flattened.problems())


def test_a_quiet_clip_is_flagged_without_being_called_broken():
    report = audio_report(_speech() * 0.01, SR)
    assert report.peak_dbfs < -30
    assert any("very quiet" in p for p in report.problems())
    assert not any("clipped" in p for p in report.problems())


def test_dc_offset_is_measured_and_reported():
    report = audio_report(_speech() * 0.5 + 0.05, SR)
    assert report.dc_offset == pytest.approx(0.05, abs=0.005)
    assert any("DC offset" in p for p in report.problems())


@pytest.mark.parametrize("source_sr,file_sr", [(8000, 16000), (16000, 44100)])
def test_upsampled_audio_is_caught_and_its_real_rate_named(source_sr, file_sr):
    """The failure that quietly wastes half a dataset: a file labelled 16 kHz
    that is really 8 kHz content with nothing above 4 kHz."""
    native = _speech(source_sr)
    upsampled = resample(native, source_sr, file_sr)

    report = audio_report(upsampled, file_sr)
    assert report.band_limited is True
    assert report.bandwidth_hz == pytest.approx(source_sr / 2, rel=0.1)
    assert any(f"upsampled from {source_sr} Hz" in p for p in report.problems())


@pytest.mark.parametrize("sr", [8000, 16000, 44100])
def test_audio_recorded_at_its_stated_rate_is_not_flagged(sr):
    report = audio_report(_speech(sr), sr)
    assert report.band_limited is False
    assert report.bandwidth_hz > 0.9 * (sr / 2)


def test_the_snr_estimate_tracks_the_noise_it_is_given():
    clean = audio_report(_speech(noise=0.001), SR).estimated_snr_db
    noisy = audio_report(_speech(noise=0.2), SR).estimated_snr_db
    assert clean > noisy + 10
    assert any("estimated SNR" in p for p in audio_report(
        _speech(noise=0.4), SR).problems())


def test_silence_is_measured_as_a_fraction_of_the_clip():
    y = np.concatenate([_speech(seconds=1.0), np.zeros(4 * SR, dtype=np.float32)])
    report = audio_report(y, SR)
    assert report.silence_ratio == pytest.approx(0.8, abs=0.05)
    assert any("silence" in p for p in report.problems())


def test_digital_silence_says_so_and_stops_there():
    report = audio_report(np.zeros(SR, dtype=np.float32), SR)
    assert report.peak_dbfs == float("-inf")
    assert np.isnan(report.lufs) or report.lufs == float("-inf")
    assert report.problems() == ["the clip is digital silence"]


def test_a_clip_too_short_for_bs1770_still_reports_everything_else():
    report = audio_report(_speech(seconds=0.2), SR)
    assert np.isnan(report.lufs)
    assert np.isfinite(report.peak_dbfs)
    assert report.bandwidth_hz > 0


def test_bad_input_is_rejected_clearly():
    with pytest.raises(FeatureError, match="sample rate"):
        audio_report(_speech(), 0)
    with pytest.raises(FeatureError, match="mono"):
        audio_report(np.zeros((1000, 2)), SR)
    with pytest.raises(FeatureError, match="empty"):
        audio_report(np.zeros(0), SR)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_a_non_finite_clip_is_not_called_digital_silence(bad):
    """Ten samples of a one-second tone replaced by something that is not a
    number: a peak of nan is not a peak of nothing.

    Every measurement here is computed from those samples, so all of them come
    out nan, and the dBFS of nan used to be -inf -- which is exactly what
    digital silence reads as. A corrupt file was reported as a muted
    microphone, the one answer that stops anybody looking at the file.
    """
    y = 0.3 * np.sin(2 * np.pi * 220 * np.arange(SR) / SR)
    y[1000:1010] = bad

    report = audio_report(y, SR)
    problems = report.problems()

    assert len(problems) == 1
    assert "not finite" in problems[0]
    assert "10 samples" in problems[0]
    assert "silence" not in problems[0]
    assert report.non_finite_samples == 10
    assert report.peak_dbfs != float("-inf")


def test_a_healthy_clip_counts_no_non_finite_samples():
    assert audio_report(_speech(), SR).non_finite_samples == 0
    assert audio_report(np.zeros(SR, dtype=np.float32), SR).non_finite_samples == 0


# -- offsets, zeroed pauses and the claimed rate -------------------------------

def test_a_dc_offset_is_not_read_as_noise():
    import kudio
    sr = 16000
    rng = np.random.default_rng(0)
    t = np.arange(4 * sr) / sr
    voice = 0.3 * np.sin(2 * np.pi * 200 * t) * (np.sin(2 * np.pi * 0.5 * t) > 0)
    y = voice + 0.001 * rng.standard_normal(len(t))
    plain = kudio.audio_report(y, sr).noise_floor_dbfs
    offset = kudio.audio_report(y + 0.03, sr).noise_floor_dbfs
    assert abs(offset - plain) < 1.0


def test_zeroed_pauses_are_a_floor_of_silence():
    import kudio
    sr = 16000
    t = np.arange(4 * sr) / sr
    y = 0.3 * np.sin(2 * np.pi * 200 * t) * (np.sin(2 * np.pi * 0.5 * t) > 0)
    report = kudio.audio_report(y, sr)
    assert report.noise_floor_dbfs == float('-inf')
    assert report.estimated_snr_db == float('inf')


def test_a_file_is_not_said_to_be_upsampled_from_its_own_rate():
    import kudio
    rng = np.random.default_rng(0)
    y = kudio.lowpass(rng.standard_normal(16000 * 3) * 0.1, 16000, cutoff=7000)
    for line in kudio.audio_report(y, 16000).problems():
        assert "upsampled from 16000" not in line


# -- a corrupt clip, said once -------------------------------------------------
def test_a_corrupt_clip_is_reported_with_one_warning_and_said_once_on_the_command_line(
        tmp_path, capsys):
    """The report names the bad samples itself, and a warning says how many to
    whoever reads no report; loudness() repeating it as a warning printed the same
    finding twice on `kudio report`, which leaves the warning out."""
    import soundfile as sf
    from kudio.cli import main
    y = 0.1 * np.random.default_rng(0).standard_normal(16000)
    y[100] = np.nan
    with pytest.warns(UserWarning) as said:
        report = kudio.audio_report(y, 16000)
    assert [str(w.message) for w in said] == [
        "audio_report(): 1 of the 16000 samples are not finite (NaN or inf); the peak, RMS "
        "and loudness are nan, the rest is of the finite frames"]
    assert np.isnan(report.lufs) and report.non_finite_samples == 1
    sf.write(tmp_path / "bad.wav", y, 16000, subtype="FLOAT")
    assert main(["report", str(tmp_path / "bad.wav")]) == 1
    out, err = capsys.readouterr()
    assert "audio_report()" not in err and "1 samples are not finite (NaN or inf)" in out


# -- recordings at a high rate, and sounds that do not pause -------------------

def _native_high_rate(sr=192000, seconds=1.0, content_hz=50000, seed=0):
    """What a 192 kHz measurement microphone hands over: a strong tone low
    down, broadband content that stops well short of Nyquist, and the
    converter's own floor above it, all the way up."""
    rng = np.random.default_rng(seed)
    n = int(sr * seconds)
    t = np.arange(n) / sr
    band = kudio.lowpass(rng.standard_normal(n), sr, cutoff=content_hz) * 0.05
    floor = 0.003 * rng.standard_normal(n)
    return (0.3 * np.sin(2 * np.pi * 300 * t) + band + floor).astype(np.float32)


def test_a_recording_made_at_a_high_rate_is_not_said_to_be_upsampled():
    """Content that stops at 50 kHz in a 192 kHz file, with the converter's
    noise above it, is a native recording: an upsampled file has nothing at
    all above its old Nyquist frequency. Taking the first for the second
    told a whole folder of measurements that its extra rate was empty."""
    report = audio_report(_native_high_rate(), 192000)
    assert report.bandwidth_hz < 0.9 * 96000          # the content does stop
    assert report.band_limited is False
    assert report.source_rates == ()
    assert not any("upsampled" in p or "is empty" in p for p in report.problems())


def test_a_file_upsampled_from_96_khz_is_named_with_its_neighbour():
    """soxr ends the content at 0.96 of the old Nyquist frequency: 46125 Hz
    here, 1.046 of 44100 and 0.961 of 48000. Resamplers end it anywhere from
    0.959 to 1.052 of it, so either rate could have left this edge, and the
    finding names both rather than guess."""
    report = audio_report(resample(_speech(96000), 96000, 192000), 192000)
    assert report.source_rates == (88200, 96000)
    assert any("upsampled from 88200 or 96000 Hz" in p for p in report.problems())


def test_the_band_limit_finding_names_the_frequency_the_rate_allows():
    """It said "although the file claims 44100 Hz": that is the rate, and the
    content is compared with half of it."""
    report = audio_report(resample(_speech(16000), 16000, 44100), 44100)
    line = next(p for p in report.problems() if "content stops" in p)
    assert "a 44100 Hz file holds up to 22050 Hz" in line


def test_a_steady_sound_is_not_said_to_bury_its_content():
    """Loud and quiet frames of a steady hiss differ by a decibel or two, and
    the estimate reads that as content drowned in noise. It cannot tell a
    steady sound from noisy speech -- the two give the same frame levels --
    so the finding says so instead of picking one."""
    rng = np.random.default_rng(0)
    report = audio_report(0.1 * rng.standard_normal(3 * SR).astype(np.float32), SR)
    line = next(p for p in report.problems() if "estimated SNR" in p)
    assert "steady" in line and "cannot tell" in line
    assert "the noise floor is close to the content" not in line


# -- resamplers, codecs, and the recordings they must not be taken for -------
# tools/validation/band_limit.py runs the full set: eight resamplers, ten
# pairs of rates, three kinds of content at three levels, and 470 recordings
# made at their own rate. These are its cases nearest each limit, and a few of
# the rest.

def _q16(y):
    """As it comes off a 16-bit WAV file."""
    y = np.round(np.clip(y, -1.0, 1.0 - 2.0 ** -15) * 32768.0) / 32768.0
    return y.astype(np.float32)


def _upsample(y, a, b, how):
    import math

    from scipy import signal
    g = math.gcd(a, b)
    if how == "poly":
        return signal.resample_poly(y, b // g, a // g)
    if how == "poly_kaiser":
        return signal.resample_poly(y, b // g, a // g, window=("kaiser", 8.6))
    if how == "fft":
        return signal.resample(y, len(y) * b // a)
    if how == "linear":
        n = len(y) * b // a
        return np.interp(np.arange(n) * a / b, np.arange(len(y)), y)
    soxr = pytest.importorskip("soxr")
    return soxr.resample(y, a, b, quality=how)


@pytest.mark.parametrize("how", ["poly", "poly_kaiser", "fft", "VHQ", "HQ", "MQ"])
@pytest.mark.parametrize("a,b", [(8000, 16000), (22050, 44100), (48000, 192000)])
def test_each_resampler_leaves_an_edge_that_is_found_and_named(a, b, how):
    """The content ends at 0.959-1.052 of the old Nyquist frequency, from
    soxr's VHQ to resample_poly with a Kaiser window, and each rate it ends
    0.95-1.06 of is named: the source alone, or the source and a neighbour
    9% away."""
    y = _q16(_upsample(_speech(a).astype(np.float64), a, b, how))
    report = audio_report(y, b)
    assert report.band_limited and not report.mirrored
    assert a in report.source_rates and len(report.source_rates) <= 2


def test_a_resampler_that_ends_the_content_early_is_not_named():
    """soxr's LQ ends it at 0.86-0.88 of the old Nyquist frequency, outside
    the window the naming takes, so no rate is named rather than a wrong one."""
    y = _q16(_upsample(_speech(8000).astype(np.float64), 8000, 16000, "LQ"))
    report = audio_report(y, 16000)
    assert report.band_limited and report.source_rates == ()
    assert any("upsampled from a lower rate, or passed through a codec" in p
               for p in report.problems())


def test_neighbouring_rates_are_both_named_when_the_edge_fits_both():
    y = _q16(_upsample(_speech(24000).astype(np.float64), 24000, 48000, "HQ"))
    report = audio_report(y, 48000)
    assert report.source_rates == (22050, 24000)
    assert any("upsampled from 22050 or 24000 Hz" in p for p in report.problems())


@pytest.mark.parametrize("a,b", [(8000, 16000), (16000, 48000), (48000, 192000)])
def test_linear_interpolation_by_a_whole_factor_is_found_by_its_mirror_image(a, b):
    """It leaves no edge: the band under the old Nyquist frequency comes back
    above it, weakened by the interpolator's response. Over the validation
    set that image follows the response to 0.20 dB, against a limit of 0.3;
    the nearest recording made at its own rate is 0.60 dB off it."""
    y = _q16(_upsample(_speech(a).astype(np.float64), a, b, "linear"))
    report = audio_report(y, b)
    assert report.band_limited and report.mirrored and report.source_rates == (a,)
    line = next(p for p in report.problems() if "mirrors" in p)
    assert f"the band above {a / 2:g} Hz mirrors the band under it" in line
    assert f"upsampled from {a} Hz by linear interpolation" in line


def test_a_codec_cut_away_from_any_common_rate_is_flagged_without_a_rate():
    y = _speech(44100).astype(np.float64)
    spectrum = np.fft.rfft(y)
    spectrum[np.fft.rfftfreq(len(y), 1 / 44100) > 19500] = 0
    report = audio_report(_q16(np.fft.irfft(spectrum, len(y))), 44100)
    assert report.band_limited and report.source_rates == ()
    assert any("passed through a codec or a filter that cuts there" in p
               for p in report.problems())


def _deepest_edge(y, sr):
    """The largest depth at any frequency the cliff test passes, and the
    spectrum, asked the questions audio_report asks."""
    from kudio.core.report import _CLIFF_DB, _SOURCE_RATES, _Edges, _welch_db
    edges = _Edges(*_welch_db(np.asarray(y, dtype=np.float64), sr), sr)
    nyquist = sr / 2
    asked = [rate / 2 for rate in _SOURCE_RATES if rate / 2 <= 0.9 * nyquist]
    asked += list(np.geomspace(0.1 * nyquist, 0.9 * nyquist, 200))
    found = [edges.cliff_and_depth(float(fc)) for fc in asked]
    return max((depth for cliff, depth in found if cliff >= _CLIFF_DB),
               default=float("-inf")), edges


def _band_over_a_rising_floor(cutoff, sr=192000, seed=0):
    """Content that stops at *cutoff*, over a converter floor that rises 6 dB
    an octave towards Nyquist, the way first-order noise shaping leaves it."""
    from scipy import signal
    rng = np.random.default_rng(seed)
    band = kudio.lowpass(rng.standard_normal(sr), sr, cutoff=cutoff) * 0.05
    floor = signal.lfilter([1, -0.95], [1], rng.standard_normal(sr)) * 0.0005
    return (band + floor).astype(np.float32)


def test_a_floor_rising_towards_nyquist_is_not_an_edge():
    from scipy import signal
    rng = np.random.default_rng(0)
    y = (signal.lfilter([1, -0.9], [1], rng.standard_normal(192000)) * 0.01).astype(np.float32)
    report = audio_report(y, 192000)
    assert not report.band_limited and report.source_rates == ()
    assert _deepest_edge(y, 192000)[0] == float("-inf"), "no cliff of 20 dB anywhere"


@pytest.mark.parametrize("cutoff,depth", [(30000, 32.54), (80000, 34.18)])
def test_content_that_stops_over_a_rising_floor_is_not_band_limited(cutoff, depth):
    """Two cases like the validation set's nearest an edge: over its 470
    recordings the floor keeps the band above the content no more than
    34.3 dB under it, where an edge needs 40."""
    y = _band_over_a_rising_floor(cutoff)
    report = audio_report(y, 192000)
    assert not report.band_limited and not report.mirrored
    deepest, _ = _deepest_edge(y, 192000)
    assert deepest == pytest.approx(depth, abs=0.05)


def test_a_smooth_roll_off_is_not_taken_for_a_mirror_image():
    """Content rolling off above 30 kHz in a 192 kHz file follows the curve of
    a linear-interpolation image at 24 kHz to 0.69 dB, but 1.79 dB above it:
    the image of a whole-factor interpolation lies on the curve, within 0.09."""
    y = _band_over_a_rising_floor(30000)
    spread, offset = _deepest_edge(y, 192000)[1].image_fit(24000)
    assert spread == pytest.approx(0.692, abs=0.005)
    assert offset == pytest.approx(1.792, abs=0.005)
    assert not audio_report(y, 192000).mirrored


# -- two ways to count clipping, and the grid the samples sit on --------------

def test_clipping_can_be_counted_in_shorter_runs_and_sample_by_sample():
    """Tools disagree on what clipping is: three samples at full scale in a
    row, two, or any one at all. The run length is a parameter, and every
    sample at full scale is counted beside it."""
    y = _speech()
    y[100:102] = 1.0                                # two in a row
    y[5000] = -1.0                                  # one on its own
    report = audio_report(y, SR)
    assert report.clipped_samples == 0
    assert report.full_scale_samples == 3
    assert audio_report(y, SR, clip_run=2).clipped_samples == 2
    assert audio_report(y, SR, clip_run=1).clipped_samples == 3


def test_a_plateau_is_counted_when_the_sample_before_it_is_not_on_it():
    """A waveform reaching the rail leaves a sample at 0.9993 or so in front of
    the plateau. Judged as one run with it, five samples at full scale were
    not flat enough to count."""
    y = _speech()
    y[500:506] = [0.9993, 1.0, 1.0, 1.0, 1.0, 1.0]
    y[900:904] = [-1.0, -1.0, -1.0, -0.9995]
    report = audio_report(y, SR)
    assert report.clipped_samples == 8
    assert report.full_scale_samples == 10


@pytest.mark.parametrize("hz,sr", [(30, 16000), (50, 48000), (100, 48000), (80, 44100),
                                   (50, 192000), (300, 192000)])
@pytest.mark.parametrize("form", ["float", "16-bit"])
def test_a_sine_normalised_to_full_scale_is_not_clipped(hz, sr, form):
    """Low enough, a smooth peak keeps three or more samples within 1e-4 of its
    top, and in 16 bits they are one code. Taken for a plateau, a 50 Hz sine
    at 48 kHz had 1000 samples clipped. A smooth peak at level m keeps
    sqrt(1e-4 / (m - 0.999)) of its run above 0.999 within 1e-4 of the top,
    0.32 at full scale; a plateau has to hold twice that share."""
    t = np.arange(sr) / sr
    y = np.sin(2 * np.pi * hz * t + 0.3)
    y = y / np.max(np.abs(y))
    if form == "16-bit":
        y = _q16(y)
    assert audio_report(y, sr).clipped_samples == 0


@pytest.mark.parametrize("amplitude", [1.02, 1.002])
def test_a_low_tone_driven_just_past_full_scale_is_clipped(amplitude):
    """50 Hz at 192 kHz, 1.02 cut to 1.0: about 245 samples on each plateau
    and three on each ramp. 1.002 cuts 0.2% off the top and still counts."""
    sr = 192000
    y = np.clip(amplitude * np.sin(2 * np.pi * 50 * np.arange(sr) / sr + 0.3), -1.0, 1.0)
    on_top = int(np.count_nonzero(np.abs(y) >= 1.0 - 1e-4))
    report = audio_report(y, sr)
    assert report.clipped_samples == on_top > 0


@pytest.mark.parametrize("bad", [0, -1, 2.5])
def test_a_clip_run_that_is_not_a_whole_number_of_one_or_more_is_refused(bad):
    with pytest.raises(FeatureError, match="clip_run"):
        audio_report(_speech(), SR, clip_run=bad)


def test_16_bit_audio_sits_on_a_16_bit_grid():
    report = audio_report(_q16(_speech()), SR)
    assert report.sample_step == 2 / 65536
    assert report.grid_bits == 16.0
    assert not any("grid" in p for p in report.problems())


def test_a_coarser_grid_is_found_and_named_in_bits():
    """A 12-bit converter written to a 16-bit file: every sample a multiple
    of 16 of the file's steps."""
    report = audio_report((np.round(_speech() * 2048) / 2048).astype(np.float32), SR)
    assert report.sample_step == 2 / 4096
    assert report.grid_bits == 12.0
    assert any("every sample sits on a grid of 12.0 bits" in p for p in report.problems())


def test_clipping_does_not_hide_the_grid():
    """The two values at full scale are left out of the grid: a clipped take
    would otherwise sit on a grid with one step from 32767 to 32768."""
    report = audio_report(_q16(np.clip(_speech() * 3, -1, 1)), SR)
    assert report.clipped_samples > 0 and report.grid_bits == 16.0


@pytest.mark.parametrize("make", [
    lambda: _speech(),                              # float, with a noise bed
    lambda: 0.5 * np.sign(np.sin(2 * np.pi * 100 * np.arange(SR) / SR)),
], ids=["processed float", "square wave"])
def test_no_grid_is_named_where_there_is_none_worth_naming(make):
    report = audio_report(make(), SR)
    assert report.sample_step == 0.0 and np.isnan(report.grid_bits)
    assert not any("grid" in p for p in report.problems())


# -- the frequency the finding gives ------------------------------------------

def _stops_at(report):
    line = next(p for p in report.problems() if "content stops" in p)
    return float(line.split("content stops at ")[1].split(" Hz")[0])


@pytest.mark.parametrize("a,b,tone", [(48000, 192000, False), (16000, 48000, False),
                                      (48000, 192000, True)],
                         ids=["48 to 192 kHz", "16 to 48 kHz", "with a line at 0.7 Nyquist"])
def test_the_finding_does_not_put_the_content_past_the_rate_it_names(a, b, tone):
    """It quoted bandwidth_hz, read 50 dB under the peak of a 1024-point
    spectrum: white noise taken from 48 to 192 kHz by resample_poly stopped
    at 27750 Hz, past the 24 kHz of the rate it was said to come from, and a
    line at 67.2 kHz put it at 67500. The edge test's own end is used."""
    from scipy import signal
    white = 0.1 * np.random.default_rng(0).standard_normal(2 * a)
    y = signal.resample_poly(white, b // a, 1)
    if tone:
        y = y + 0.01 * np.sin(2 * np.pi * 0.7 * b / 2 * np.arange(len(y)) / b)
    report = audio_report(_q16(y), b)
    assert report.source_rates == (a,)
    assert report.bandwidth_hz > 1.06 * a / 2, "the old reading, for the record"
    assert a / 2 <= _stops_at(report) <= 1.06 * a / 2
    assert abs(report.band_edge_hz - _stops_at(report)) < 0.5


def test_a_codec_cut_is_given_where_it_cuts():
    y = 0.1 * np.random.default_rng(0).standard_normal(2 * 44100)
    spectrum = np.fft.rfft(y)
    spectrum[np.fft.rfftfreq(len(y), 1 / 44100) > 19500] = 0
    report = audio_report(_q16(np.fft.irfft(spectrum, len(y))), 44100)
    assert report.source_rates == ()
    assert _stops_at(report) == pytest.approx(19500, rel=0.01)


def test_a_clip_with_no_band_limit_has_no_edge():
    assert np.isnan(audio_report(_speech(), SR).band_edge_hz)
