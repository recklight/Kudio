# -*- coding: utf-8 -*-
"""SRMR: rooms whose order is known, filters held to the toolbox's formulas,
and the clips the measure refuses or cannot score."""
from __future__ import annotations

import math
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

import kudio
from kudio.core.srmr import srmr

# kudio.core may import srmr() under the module's own name, which hides the
# module behind the function; sys.modules still has the module
module = sys.modules["kudio.core.srmr"]

#: the default modulation band centres, 4 Hz times (128 / 4) ** (k / 7)
CENTRES = 4.0 * 32.0 ** (np.arange(8) / 7.0)

MODES = [{}, {"fast": True}, {"norm": True}, {"fast": True, "norm": True}]


def _voice(sr: int, seconds: float = 3.0, seed: int = 0) -> np.ndarray:
    """Voiced syllables of 120-280 ms with pauses of 40-200 ms between them.

    Each syllable is a harmonic complex whose pitch starts between 220 and
    300 Hz and glides by up to 15%, with random harmonic phases, three
    formant-like peaks and a Hann envelope. The pauses are digital silence.
    """
    rng = np.random.default_rng(seed)
    out = np.zeros(int(seconds * sr))
    start = int(0.1 * sr)
    while True:
        n = int(rng.uniform(0.12, 0.28) * sr)
        if start + n > out.size - int(0.1 * sr):
            break
        t = np.arange(n) / sr
        f0 = rng.uniform(220.0, 300.0) * (1.0 + rng.uniform(-0.15, 0.15) * t / t[-1])
        phase = 2.0 * np.pi * np.cumsum(f0) / sr
        k = np.arange(1, int(sr / 2.0 / f0.max()))
        peaks = np.zeros(k.size)
        for low, high, width in ((300, 800, 90), (900, 2200, 120), (2400, 3200, 160)):
            offset = (k * f0.mean() - rng.uniform(low, high)) / width
            peaks += 1.0 / (1.0 + offset ** 2)
        phases = np.outer(phase, k) + rng.uniform(0, 2 * np.pi, k.size)
        tone = (peaks / np.sqrt(k) * np.sin(phases)).sum(axis=1)
        out[start:start + n] += tone / np.max(np.abs(tone)) * np.hanning(n)
        start += n + int(rng.uniform(0.04, 0.20) * sr)
    return out / np.max(np.abs(out))


def _band_energy(y: np.ndarray, sr: int, fast: bool = False) -> np.ndarray:
    """Modulation energy per band, summed over channels and averaged over
    windows, with the default bands."""
    rate = 400.0 if fast else float(sr)
    centres, _ = module._modulation_bands(4.0, 128.0, rate, sr)
    return module._modulation_energy(y / np.max(np.abs(y)), sr, centres, rate,
                                     fast).mean(axis=2).sum(axis=0)


# ------------------------------------------------------------ rooms

@pytest.mark.parametrize("sr, kwargs", [(16000, mode) for mode in MODES]
                         + [(8000, {})])
def test_the_score_falls_as_the_room_gets_longer(sr, kwargs):
    """What the measure is for: the same speech in rooms of 0.3, 0.6, 1.0 and
    1.5 s, each scoring below the one before and all below the dry take.

    Two things would make the takes differ in more than the room. One is the
    silence cut: with its pauses at digital zero, the dry take would lose
    them to the cut while the rooms' tails keep theirs, and it would be
    scored over less of the speech. So the voice sits on a noise floor 40 dB
    under its peak, above the cut's 50 dB line, in every take. The other is
    pitch, and the voice is pitched between 220 and 300 Hz on purpose. A
    voice's envelope also beats at its pitch, and a lower one puts that
    beating in the two fastest bands, where a short room weakens it; the dry
    take and the 0.3 s room can then come out the wrong way round.
    """
    dry = np.concatenate([_voice(sr, 3.0), np.zeros(int(1.6 * sr))])
    dry = dry + 0.01 * np.random.default_rng(1).standard_normal(dry.size)
    scores = [srmr(dry, sr, **kwargs)]
    for rt60 in (0.3, 0.6, 1.0, 1.5):
        wet = kudio.apply_rir(dry, kudio.rir(sr, rt60=rt60, seed=0))
        scores.append(srmr(wet, sr, **kwargs))
    assert all(a > b for a, b in zip(scores, scores[1:])), scores


def test_the_level_does_not_change_the_score():
    """A ratio of energies: a gain cancels, and integer audio is read on its
    own full scale, so an int16 clip scores as its float twin does. The clip
    is scaled to its peak before anything is squared; at a gain of 1e-200
    the squares would otherwise all be 0."""
    y = _voice(16000, 2.0)
    base = srmr(y, 16000)
    for gain in (1e-200, 1e-4, 0.5, 30.0):
        assert srmr(gain * y, 16000) == pytest.approx(base, rel=1e-12)
    pcm = np.round(0.5 * y * 32767).astype(np.int16)
    assert srmr(pcm, 16000) == srmr(pcm / 32768.0, 16000)


def test_both_rates_score_and_score_differently():
    """The 23 channels run from 125 Hz to half the rate, so the same speech
    at 8 and at 16 kHz is measured through two different filterbanks."""
    from scipy.signal import resample_poly
    wide = _voice(16000, 2.0)
    narrow = resample_poly(wide, 1, 2)
    a, b = srmr(wide, 16000), srmr(narrow, 8000)
    assert np.isfinite(a) and np.isfinite(b)
    assert abs(a - b) > 0.01 * a


# ------------------------------------------------------------ the filters

def test_the_auditory_channels_sit_where_slaney_puts_them():
    """23 centres at equal steps of ln(f + EarQ * minBW), the lowest exactly
    at 125 Hz and the highest one step under half the rate."""
    for sr in (8000, 16000):
        centres = module._channel_centres(sr)
        assert len(centres) == 23
        assert centres[-1] == pytest.approx(125.0, rel=1e-12)
        steps = np.diff(np.log(centres + 9.26449 * 24.7))
        assert np.allclose(steps, steps[0], rtol=1e-9)
        assert centres[0] == pytest.approx(
            math.exp(math.log(sr / 2.0 + 9.26449 * 24.7) + steps[0]) - 9.26449 * 24.7)


def test_the_filterbank_matches_the_gammatone_package():
    """Golden values from Gammatone 1.0.3 (``make_erb_filters`` on
    ``centre_freqs(16000, 23, 125)``), the Python port of Slaney's
    MakeERBFilters: the shared denominator, the four zero coefficients and
    the gain of the 125 Hz channel and of the top one."""
    sos = module._gammatone(16000)
    golden = {
        22: ((-6.876937633500417e-05, -5.418647904647575e-05,
              -6.272894249940186e-05, -6.022691288207805e-05),
             -1.9672936861036787, 0.9698962696648762, 1.4423547188624494e-10),
        0: ((-2.4532503250790663e-06, 8.642057640979885e-05,
             3.435949404850394e-05, 4.9607832036215846e-05),
            1.3434772173555167, 0.5379641598804756, 1.4861002117267011e-15),
    }
    for channel, (zeros, b1, b2, gain) in golden.items():
        sections = sos[channel]
        assert sections[0, 0] == pytest.approx(1.0 / 16000 / gain, rel=1e-12)
        assert sections[0, 1] * gain == pytest.approx(zeros[0], rel=1e-12)
        for k in (1, 2, 3):
            assert sections[k, 1] == pytest.approx(zeros[k], rel=1e-12)
        assert np.allclose(sections[:, 4], b1, rtol=1e-12)
        assert np.allclose(sections[:, 5], b2, rtol=1e-12)


@pytest.mark.parametrize("sr", [8000, 16000])
def test_every_channel_passes_its_centre_at_unit_gain_and_is_one_erb_wide(sr):
    """The gain is normalised at the centre (Slaney 1993, sec. 4.2), and a
    fourth-order gammatone with b = 1.019 ERB is one ERB wide. Below a
    quarter of the rate every channel lands within 0.3% of Glasberg and
    Moore's ERB; the top channels, close to half the rate, come out up to
    6% wider, where the impulse-invariant design folds over."""
    sos = module._gammatone(sr)
    centres = module._channel_centres(sr)
    at_centre = module._response(sos, (2 * np.pi * centres / sr)[:, None])[:, 0]
    assert np.allclose(np.abs(at_centre), 1.0, rtol=1e-12)

    f = np.linspace(0.0, sr / 2.0, 2 ** 16 + 1)
    power = np.abs(module._response(sos, 2 * np.pi * f[None, :] / sr)) ** 2
    width = (power.sum(axis=1) - 0.5 * (power[:, 0] + power[:, -1])) * (f[1] - f[0])
    ratio = width / (centres / 9.26449 + 24.7)
    low = centres < sr / 4.0
    assert np.all(np.abs(ratio[low] - 1.0) < 0.003), ratio[low]
    assert np.all(ratio[~low] < 1.06), ratio[~low]


@pytest.mark.parametrize("n", [10891, 10892])
def test_the_envelope_is_the_magnitude_of_the_analytic_signal(n):
    """Taken over the whole clip with an FFT of its own length, odd or even,
    as MATLAB's ``hilbert`` and ``scipy.signal.hilbert`` both take it."""
    from scipy.signal import hilbert
    x = np.random.default_rng(n).standard_normal(n)
    assert np.allclose(module._hilbert_envelope(x), np.abs(hilbert(x)),
                       rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("sr, length, hop, nfft", [(16000, 160, 40, 512),
                                                   (8000, 80, 20, 256)])
def test_the_fast_envelopes_are_the_gammatonegram_srmr_m_asks_for(sr, length, hop, nfft):
    """gammatonegram(s, fs, 0.010, 0.0025, ...): 10 ms windows every 2.5 ms
    from the first sample, only those that fit, MATLAB's hanning, which has
    no zeros at its ends, and an FFT of the next power of two at or above
    twice the window. Each channel's envelope is its filter's magnitude
    response times the magnitude spectrum, summed over the bins and divided
    by the FFT length.

    The clip is one sample short of a 101st window, so there are 100."""
    y = np.random.default_rng(sr).standard_normal(length + 100 * hop - 1)
    sos = module._gammatone(sr)
    envelopes = module._spectral_envelopes(y, sr, sos)
    assert envelopes.shape == (23, 100)
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(1, length + 1) / (length + 1))
    bins = 2 * np.pi * np.arange(nfft // 2 + 1) / nfft
    gains = np.abs(module._response(sos, bins[None, :]))
    for i in range(100):
        spectrum = np.abs(np.fft.rfft(y[i * hop:i * hop + length] * window, nfft))
        assert np.allclose(envelopes[:, i], gains @ spectrum / nfft, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("rate", [16000.0, 8000.0, 400.0])
def test_the_modulation_bands_follow_the_toolbox_formulas(rate):
    """Eight centres from 4 to 128 Hz a constant ratio apart, each filter's
    peak of 1 exactly on its centre, and the lower edges the K* rule reads,
    ``cf - tan(pi cf / sr) * sr / (2 pi Q)`` with Q = 2 at the audio rate."""
    from scipy.signal import freqz
    sr = 16000
    centres, lower = module._modulation_bands(4.0, 128.0, rate, sr)
    assert np.allclose(centres, CENTRES, rtol=1e-12)
    assert np.allclose(lower, CENTRES - np.tan(np.pi * CENTRES / sr) * sr / (4 * np.pi),
                       rtol=1e-12)
    for cf in centres:
        b, a = module._modulation_filter(cf, rate)
        peak = abs(freqz(b, a, worN=[cf], fs=rate)[1][0])
        assert peak == pytest.approx(1.0, abs=1e-12)
        if rate > 400.0:
            # well under half the rate, the -3 dB width is the cf / Q of Q = 2
            f = np.linspace(0.5 * cf, 1.5 * cf, 20001)
            passed = f[np.abs(freqz(b, a, worN=f, fs=rate)[1]) >= 1 / np.sqrt(2)]
            assert passed[-1] - passed[0] == pytest.approx(cf / 2.0, rel=0.005)


def test_in_the_fast_mode_the_top_band_is_28_hz_wide():
    """At the 400 Hz envelope rate the bilinear transform squeezes the 128 Hz
    band: its -3 dB points sit 28.3 Hz apart, not the 64 Hz of Q = 2."""
    from scipy.signal import freqz
    b, a = module._modulation_filter(128.0, 400.0)
    f = np.linspace(80.0, 199.0, 119001)
    passed = f[np.abs(freqz(b, a, worN=f, fs=400.0)[1]) >= 1 / np.sqrt(2)]
    assert passed[-1] - passed[0] == pytest.approx(28.3, abs=0.05)


@pytest.mark.parametrize("band", range(8))
@pytest.mark.parametrize("sr, fast", [(8000, False), (16000, True)])
def test_modulating_noise_at_a_band_centre_raises_that_band_most(band, sr, fast):
    """Noise has a modulation spectrum of its own; multiplying it by
    ``1 + cos(2 pi fm t)`` adds energy at fm, and the band centred there is
    the one that gains the most over the same noise unmodulated."""
    noise = np.random.default_rng(band).standard_normal(2 * sr)
    t = np.arange(noise.size) / sr
    modulated = noise * (1.0 + np.cos(2 * np.pi * CENTRES[band] * t))
    gain = _band_energy(modulated, sr, fast) / _band_energy(noise, sr, fast)
    assert int(np.argmax(gain)) == band, np.round(10 * np.log10(gain), 1)


def test_the_windows_fall_where_matlab_buffer_puts_them():
    """The case MATLAB's documentation draws for buffer, x = 1:30 in windows
    of 7 overlapping by 3: each window repeats the last 3 samples of the one
    before, the first starts with 3 zeros, and there are ceil(30 / 4) = 8,
    the last padded with 2 zeros. The weights multiply the squared samples
    place by place, so unequal weights also check the order."""
    windows = np.array([[0, 0, 0, 1, 2, 3, 4],
                        [2, 3, 4, 5, 6, 7, 8],
                        [6, 7, 8, 9, 10, 11, 12],
                        [10, 11, 12, 13, 14, 15, 16],
                        [14, 15, 16, 17, 18, 19, 20],
                        [18, 19, 20, 21, 22, 23, 24],
                        [22, 23, 24, 25, 26, 27, 28],
                        [26, 27, 28, 29, 30, 0, 0]], dtype=float)
    weights = np.arange(1.0, 8.0)
    energy = module._framed_energy(np.arange(1.0, 31.0), 7, 4, weights)
    assert np.array_equal(energy, windows ** 2 @ weights)


@pytest.mark.parametrize("rate, length, hop", [(16000.0, 4096, 1024),
                                               (8000.0, 2048, 512),
                                               (400.0, 103, 26)])
def test_the_windows_are_256_ms_of_symmetric_hamming_every_64_ms(rate, length, hop):
    """SRMR.m rounds both lengths up, so at the fast mode's 400 Hz 102.4
    samples become 103 and 25.6 become 26. MATLAB's hamming is the symmetric
    window, 0.54 - 0.46 cos(2 pi n / (N - 1)), and the weights are its
    square."""
    got_length, got_hop, weights = module._modulation_windows(rate)
    assert (got_length, got_hop) == (length, hop)
    hamming = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(length) / (length - 1))
    assert np.allclose(weights, hamming ** 2, rtol=0.0, atol=1e-15)


def test_k_star_follows_the_channel_that_holds_90_percent_of_the_energy():
    """BW is the ERB of the channel where the energy, counted up from 125 Hz,
    first passes 90%; K* is the last band whose lower edge is at or below
    it. The edges at 16 kHz sit at 35.7, 58.5 and 96.0 Hz for bands 6 to 8.

    85% in the 125 Hz channel and 15% in the 472 Hz one passes 80% at
    125 Hz but 90% only at 472 Hz: a share of 80% would give K* = 6 and a
    ratio of 7.5, where 90% gives K* = 7 and 5. Exactly 90% at 125 Hz has
    not passed 90%. An edge exactly at BW counts, as SRMR.m's
    ``BW >= cutoffs(k)`` counts it.
    """
    _, lower = module._modulation_bands(4.0, 128.0, 16000.0, 16000)
    bands = np.array([8.0, 4.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    lowest = np.zeros((23, 8))
    lowest[-1] = bands                   # all in the 125 Hz channel: BW 38.2 Hz
    assert module._ratio(lowest, lower, 16000) == pytest.approx(15.0 / 2.0)

    middle = np.zeros((23, 8))
    middle[-6] = bands                   # the 472 Hz channel: BW 75.7 Hz
    assert module._ratio(middle, lower, 16000) == pytest.approx(15.0 / 3.0)

    spread = np.tile(bands, (23, 1))     # 90% first passed at the 21st
    assert module._ratio(spread, lower, 16000) == pytest.approx(15.0 / 4.0)

    for low, high in ((17.0, 3.0), (9.0, 1.0)):     # 85/15%, then 90/10%
        split = np.zeros((23, 8))
        split[-1], split[-6] = low * bands, high * bands
        assert module._ratio(split, lower, 16000) == pytest.approx(15.0 / 3.0)

    tie = lower.copy()
    tie[6] = module._erb(module._channel_centres(16000)[-1])   # band 7 at 38.2 Hz
    assert module._ratio(lowest, tie, 16000) == pytest.approx(15.0 / 3.0)


# ------------------------------------------------------------ silence

def test_silence_around_the_speech_is_cut_before_scoring():
    """The toolbox cuts what is more than 50 dB under the peak sample before
    it scores, so a second of digital silence either side changes nothing;
    kept, it does."""
    y = _voice(16000, 2.0)
    pad = np.zeros(16000)
    padded = np.concatenate([pad, y, pad])
    for kwargs in MODES:
        assert srmr(padded, 16000, **kwargs) == srmr(y, 16000, **kwargs)
    assert srmr(padded, 16000, drop_silence=False) != srmr(y, 16000, drop_silence=False)


@pytest.mark.parametrize("gaps", [1, 2])
def test_a_long_silence_inside_the_clip_is_cut_as_well(gaps):
    """300 ms of digital silence spliced in between two loud samples is cut
    back out, as one stretch or as two.

    The voice sits on a noise floor 40 dB under its peak, so its own pauses
    are not silence and the spliced stretches are the only ones there are.
    With exactly one, the toolbox's ``vad_s.m`` keeps it, because its branch
    for that case repeats ``idx2(1)`` where ``idx2(2)`` belongs; this cuts
    it all the same.
    """
    y = _voice(16000, 3.0) + 0.01 * np.random.default_rng(1).standard_normal(48000)
    level = np.abs(y)
    cuts = []
    for lo, hi in ((0.25, 0.45), (0.55, 0.75))[:gaps]:
        a, b = int(lo * y.size), int(hi * y.size)
        loud = np.flatnonzero(np.minimum(level[a:b - 1], level[a + 1:b]) > 0.1)
        cuts.append(a + int(loud[0]) + 1)
    pieces = np.split(y, cuts)
    spliced = np.concatenate([np.concatenate([p, np.zeros(4800)]) for p in pieces[:-1]]
                             + [pieces[-1]])
    assert srmr(spliced, 16000) == srmr(y, 16000)
    assert srmr(spliced, 16000, drop_silence=False) != srmr(y, 16000)


def test_the_cut_draws_its_lines_50_db_under_the_peak_and_at_50_ms():
    """vad_s.m's two numbers. Spliced into the voice, 300 ms of a 1 kHz tone
    peaking 45 dB under the voice's peak is kept and the same tone 55 dB
    under it is cut; 40 ms of digital silence is kept and 70 ms is cut.

    The voice sits on a noise floor 40 dB under its peak and starts and ends
    on samples above the line, so what is spliced in is the only thing the
    cut could take out."""
    sr = 16000
    y = _voice(sr, 3.0) + 0.01 * np.random.default_rng(1).standard_normal(3 * sr)
    y = y / np.max(np.abs(y))
    above = np.flatnonzero(np.abs(y) > 10 ** (-50 / 20))
    y = y[above[0]:above[-1] + 1]
    bare = srmr(y, sr)
    assert bare == srmr(y, sr, drop_silence=False)

    # between two samples far above the line, half way in
    half = y.size // 2
    loud = np.minimum(np.abs(y[half - 1:-1]), np.abs(y[half:])) > 0.1
    at = half + int(np.flatnonzero(loud)[0])
    tone = np.sin(2 * np.pi * 1000 * np.arange(int(0.3 * sr)) / sr)
    for piece, kept in ((10 ** (-45 / 20) * tone, True), (10 ** (-55 / 20) * tone, False),
                        (np.zeros(int(0.04 * sr)), True), (np.zeros(int(0.07 * sr)), False)):
        spliced = np.concatenate([y[:at], piece, y[at:]])
        want = srmr(spliced, sr, drop_silence=False) if kept else bare
        assert srmr(spliced, sr) == want, (piece.size, np.max(np.abs(piece)))


def test_a_room_fills_the_pauses_the_cut_takes_out_of_a_dry_take():
    """The voice's pauses and ends are digital zero. The cut keeps 26877 of
    the dry take's 48000 samples, and its score falls by 8%, from 52.77 to
    48.32. In a 0.3 s room the tail fills the pauses: the cut keeps 46260
    samples, and the score moves by 0.4%, from 12.49 to 12.54."""
    sr = 16000
    dry = _voice(sr, 3.0)
    wet = kudio.apply_rir(dry, kudio.rir(sr, rt60=0.3, seed=0))
    assert module._drop_silence(dry, sr).size < 0.6 * dry.size
    assert module._drop_silence(wet, sr).size > 0.95 * wet.size
    assert srmr(dry, sr) < 0.95 * srmr(dry, sr, drop_silence=False)
    assert srmr(wet, sr) == pytest.approx(srmr(wet, sr, drop_silence=False), rel=0.01)


def test_a_silent_or_empty_clip_scores_nan_with_a_warning():
    with pytest.warns(UserWarning, match="the clip is silent"):
        assert math.isnan(srmr(np.zeros(16000), 16000))
    with pytest.warns(UserWarning, match=r"^srmr\(\): the clip is empty; scored nan$"):
        assert math.isnan(srmr(np.zeros(0), 16000))


def test_a_clip_shorter_than_one_window_scores_nan_with_a_warning():
    """One 256 ms window is 4096 samples at 16 kHz. A 0.2 s clip is short of
    it as it stands, and the same 0.2 s between two 1.5 s stretches of
    digital silence is short of it once the silence is cut; kept, the
    silence makes the clip long enough to score."""
    voice = _voice(16000, 3.0)
    short = voice[np.argmax(np.abs(voice) > 0.1):][:3200]
    with pytest.warns(UserWarning, match="4096 samples.*has 3200;"):
        assert math.isnan(srmr(short, 16000, drop_silence=False))
    buried = np.concatenate([np.zeros(24000), short, np.zeros(24000)])
    with pytest.warns(UserWarning, match="once its silence is cut"):
        assert math.isnan(srmr(buried, 16000))
    assert np.isfinite(srmr(buried, 16000, drop_silence=False))


# ------------------------------------------------------------ options

def test_norm_stops_the_bands_at_30_hz_unless_told_otherwise():
    y = _voice(16000, 2.0)
    assert srmr(y, 16000, norm=True) == srmr(y, 16000, norm=True, max_cf=30.0)
    assert srmr(y, 16000, norm=True, max_cf=128.0) != srmr(y, 16000, norm=True)
    assert srmr(y, 16000, max_cf=128.0) == srmr(y, 16000)


@pytest.mark.parametrize("sr, fast", [(16000, False), (16000, True), (8000, False)])
def test_norm_holds_every_energy_within_30_db_of_the_peak_channel_average(sr, fast):
    """SRMR.m's normalisation written out: the peak is the largest of the
    window energies averaged over the 23 channels, and every channel's
    energy in every band and window is held between that peak and a
    thousandth of it, 30 dB down, before the average over the clip. On this
    voice some energies sit above the peak and some below the floor, so both
    limits are in play."""
    y = _voice(sr, 2.0)
    rate = 400.0 if fast else float(sr)
    centres, lower = module._modulation_bands(4.0, 30.0, rate, sr)
    energy = module._modulation_energy(module._drop_silence(y, sr), sr, centres, rate, fast)
    peak = energy.mean(axis=0).max()
    assert np.any(energy > peak) and np.any(energy < peak * 1e-3)
    held = np.clip(energy, peak * 1e-3, peak)
    want = module._ratio(held.mean(axis=2), lower, sr)
    assert srmr(y, sr, fast=fast, norm=True) == pytest.approx(want, rel=1e-12)


# ------------------------------------------------------------ refused input

@pytest.mark.parametrize("sr", [11025, 22050, 44100, 48000])
def test_rates_the_measure_was_not_validated_at_are_refused(sr):
    """The way out names a resampler that needs no librosa as well."""
    with pytest.raises(ValueError, match=rf"8 and 16 kHz.*kudio\.resample.*"
                                         rf"scipy\.signal\.resample_poly\(y, 16000, {sr}\)"):
        srmr(np.random.default_rng(0).standard_normal(sr), sr)


@pytest.mark.parametrize("shape", [(16000, 2), (2, 16000)])
def test_more_than_one_channel_is_refused_with_a_way_out(shape):
    with pytest.raises(ValueError, match="one at a time, or mix down"):
        srmr(np.zeros(shape), 16000)


def test_a_single_channel_in_a_column_or_a_row_scores_as_itself():
    y = _voice(16000, 1.0)
    assert srmr(y[:, None], 16000) == srmr(y[None, :], 16000) == srmr(y, 16000)


def test_samples_that_are_not_finite_are_refused():
    y = _voice(16000, 1.0)
    for bad in (np.nan, np.inf):
        corrupt = y.copy()
        corrupt[100] = bad
        with pytest.raises(ValueError, match="not finite"):
            srmr(corrupt, 16000)


@pytest.mark.parametrize("kwargs, match", [
    ({"min_cf": 30.0, "max_cf": 30.0}, "min_cf < max_cf"),
    ({"min_cf": 0.0}, "min_cf < max_cf"),
    ({"max_cf": float("nan")}, "min_cf < max_cf"),
    ({"fast": True, "max_cf": 250.0}, "under 200 Hz"),
    ({"min_cf": 16.0}, "fifth modulation band"),
])
def test_bands_the_measure_cannot_use_are_refused(kwargs, match):
    """16 to 128 Hz puts the fifth band's lower edge at 39.4 Hz, above the
    38.2 Hz bandwidth of the 125 Hz channel, which is the narrowest BW the
    90% rule can return."""
    with pytest.raises(ValueError, match=match):
        srmr(_voice(16000, 1.0), 16000, **kwargs)


def test_the_measure_never_reaches_into_librosa():
    """numpy and scipy only: with every attribute of librosa raising, from
    before kudio is imported, all four modes still score."""
    code = textwrap.dedent("""
        import sys, types
        class Refuse(types.ModuleType):
            def __getattr__(self, name):
                raise AssertionError("srmr() reached for librosa." + name)
        sys.modules["librosa"] = Refuse("librosa")
        import numpy as np
        from kudio.core.srmr import srmr
        y = np.random.default_rng(0).standard_normal(8000)
        for fast in (False, True):
            for norm in (False, True):
                assert np.isfinite(srmr(y, 8000, fast=fast, norm=norm))
    """)
    subprocess.check_call([sys.executable, "-c", code],
                          cwd=str(Path(__file__).resolve().parents[1]))
