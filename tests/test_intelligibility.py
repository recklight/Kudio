# -*- coding: utf-8 -*-
"""NCM and CSII: pysepm's numbers and the MATLAB code written out again,
what an exact copy, a gain, a delay and a filter score, the numbers the
module docstring quotes, and the input each one refuses or cannot score."""
from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
from scipy.signal import (butter, freqz_zpk, hilbert, lfilter, resample_poly,
                          sosfilt, sosfiltfilt, upfirdn)

import kudio
from kudio.core import evaluator
from kudio.core import intelligibility
from kudio.core.intelligibility import CsiiScore, csii, ncm

#: rough F1, F2, F3 of five vowels, in Hz
_VOWELS = ((730, 1090, 2440), (270, 2290, 3010), (300, 870, 2240),
           (530, 1840, 2480), (570, 840, 2410))

EPS = np.finfo(float).eps


def _speech(sr: int, seconds: float = 3.0, seed: int = 1,
            floor: float = 5e-4) -> np.ndarray:
    """Syllables of a harmonic source through three formant resonators, the
    pitch gliding inside each and the formants moving from one vowel to the
    next, with pauses over a noise floor 60 dB under the peak. With
    ``floor=0`` the pauses are digital silence and the syllables are the same
    samples as with the floor."""
    rng = np.random.default_rng(seed)
    n = int(sr * seconds)
    y = np.zeros(n)
    start, k = int(0.1 * sr), 0
    while start < n - int(0.35 * sr):
        length = int(rng.uniform(0.15, 0.3) * sr)
        f0 = np.linspace(*rng.uniform(100.0, 220.0, 2), length)
        phase = 2 * np.pi * np.cumsum(f0) / sr
        source = sum(np.sin(h * phase) / h
                     for h in range(1, int(0.45 * sr / f0.max())))
        syllable = np.zeros(length)
        for j, part in enumerate(np.array_split(np.arange(length), 4)):
            piece = source[part]
            for f_from, f_to, bw in zip(_VOWELS[k % 5], _VOWELS[(k + 1) % 5],
                                        (80.0, 100.0, 140.0)):
                f = f_from + (f_to - f_from) * j / 3
                r = math.exp(-math.pi * bw / sr)
                piece = lfilter([1 - r], [1, -2 * r * math.cos(2 * math.pi * f / sr),
                                          r * r], piece)
            syllable[part] = piece
        y[start:start + length] = syllable * np.hanning(length)
        start += length + int(rng.uniform(0.05, 0.2) * sr)
        k += 1
    y *= 0.5 / np.max(np.abs(y))
    if floor:
        y = y + floor * rng.standard_normal(n)
    return y


def _noise(n: int, seed: int, colour: str = 'white', sr: int = 16000) -> np.ndarray:
    """White, pink (1/f power) or babble (six talkers of :func:`_speech`,
    staggered by 50 ms), at unit RMS."""
    if colour == 'babble':
        talkers = [_speech(sr, n / sr + 0.5, seed=seed * 10 + i, floor=0)
                   [int(0.05 * sr * i):][:n] for i in range(6)]
        w = np.sum(talkers, axis=0)
    else:
        w = np.random.default_rng(seed).standard_normal(n)
        if colour == 'pink':
            spectrum = np.fft.rfft(w)
            f = np.arange(len(spectrum), dtype=float)
            f[0] = 1.0
            w = np.fft.irfft(spectrum / np.sqrt(f), n)
    return w / np.std(w)


def _mix(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    scale = np.sqrt(np.mean(clean ** 2) / (np.mean(noise ** 2) * 10 ** (snr_db / 10)))
    return clean + scale * noise


def _lowpass(y: np.ndarray, sr: int, cutoff: float) -> np.ndarray:
    return sosfiltfilt(butter(8, cutoff, fs=sr, output='sos'), y)


def _late(y: np.ndarray, ms: float, sr: int) -> np.ndarray:
    k = int(ms * sr / 1000)
    return np.concatenate([np.zeros(k), y[:-k]])


def _pair(sr: int, colour: str = 'white', snr_db: float = 5.0):
    """The test pair of the module docstring, the test speech in white noise
    at 5 dB SNR, or the same speech in other noise."""
    clean = _speech(sr)
    return clean, _mix(clean, _noise(len(clean), 7, colour, sr), snr_db)


def _quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return fn(*args, **kwargs)


# -- against pysepm --------------------------------------------------------------

def _degraded(kind: str, sr: int):
    """``(ref, deg)`` for each kind of damage the goldens cover."""
    clean = _speech(sr)
    n = len(clean)
    if kind in ('white 5', 'pink 0', 'babble 0', 'white -5', 'babble 10'):
        colour, snr = kind.split()
        return clean, _mix(clean, _noise(n, 7, colour, sr), float(snr))
    if kind == 'room 0.6 s':
        return clean, kudio.apply_rir(clean, kudio.rir(sr, rt60=0.6, seed=3)).astype(float)
    if kind == 'log-MMSE':
        noisy = _mix(clean, _noise(n, 7, 'white', sr), 5.0)
        return clean, kudio.spectral_enhance(noisy, sr).astype(float)
    if kind == 'low-passed':
        return clean, _mix(_lowpass(clean, sr, 2500), _noise(n, 7, 'white', sr), 20.0)
    raise ValueError(kind)


#: pysepm_evo 0.1.1's ncm (Python 3.10, numpy 2.2.6, scipy 1.15.2, with
#: scipy.signal.kaiser put back for it) on ``_degraded(kind, sr)``.
PYSEPM_EVO_NCM = {
    (8000, 'babble 0'): 0.5599930977568368,
    (8000, 'log-MMSE'): 0.5869775555887952,
    (8000, 'room 0.6 s'): 0.6316677206624838,
    (8000, 'white 5'): 0.6073652091597762,
    (16000, 'babble 0'): 0.533924022983477,
    (16000, 'log-MMSE'): 0.5133131489864252,
    (16000, 'low-passed'): 0.6869890709768699,
    (16000, 'pink 0'): 0.49699221584869047,
    (16000, 'room 0.6 s'): 0.5644579590076472,
    (16000, 'white 5'): 0.5455569555289761,
}


@pytest.mark.parametrize("sr, kind", sorted(PYSEPM_EVO_NCM))
def test_ncm_matches_pysepm_evo(sr, kind):
    """pysepm filters each band as one transfer function and ncm() in
    second-order sections, and the transfer function of the lowest band
    drifts (see the next test). That is the whole difference, held here to
    3e-6 at 16 kHz and 2e-8 at 8 kHz; tools/validation/intelligibility.py
    shows the rest of the chain agreeing to 1e-15 once pysepm's filters are
    put in."""
    ref, deg = _degraded(kind, sr)
    tolerance = 3e-6 if sr == 16000 else 2e-8
    assert ncm(ref, deg, sr) == pytest.approx(PYSEPM_EVO_NCM[(sr, kind)], abs=tolerance)


#: pysepm_evo 0.1.1's csii, same versions, on ``ref + eps`` and ``deg + eps``:
#: CSII.m adds eps to both signals and pysepm does not.
PYSEPM_EVO_CSII = {
    (8000, 'babble 10'): (0.9762311580864237, 0.8257331838581885, 0.18255850128080686),
    (8000, 'room 0.6 s'): (0.5715047286518399, 0.4783787944362463, 0.09235287894435046),
    (8000, 'white 5'): (0.5944579957623013, 0.43166887692046657, 0.021211862847588227),
    (11025, 'white 5'): (0.6072603237650724, 0.44411880756170724, 0.03437965393987766),
    (16000, 'babble 0'): (0.6564802180262543, 0.4815676577863519, 0.03524701264623607),
    (16000, 'log-MMSE'): (0.6316837861358272, 0.48486875722712836, 0.18942147808884652),
    (16000, 'pink 0'): (0.5503266625857444, 0.35035883998191425, 0.006035798237596821),
    (16000, 'room 0.6 s'): (0.5617693140052493, 0.43514172190389416, 0.08315494519041453),
    (16000, 'white -5'): (0.4491681231261403, 0.2648636342164179, 0.005139265581685541),
    (44100, 'pink 0'): (0.5496356132394096, 0.35751664356411644, 0.018552486177899986),
}


@pytest.mark.parametrize("sr, kind", sorted(PYSEPM_EVO_CSII))
def test_csii_matches_pysepm_evo(sr, kind):
    """To 1e-12, but for the log-MMSE output: kudio's enhancer returns
    float32, whose last bits move with the numpy and scipy releases. Every
    sample moved by one of them, up or down at random, moves CSII by 4e-9,
    and by 1.5e-8 at the most in 20 draws."""
    ref, deg = _degraded(kind, sr)
    tolerance = 2e-8 if kind == 'log-MMSE' else 1e-12
    got = csii(ref, deg, sr)
    for value, want in zip(got, PYSEPM_EVO_CSII[(sr, kind)]):
        assert value == pytest.approx(want, rel=tolerance, abs=tolerance), (got, kind)


def _exact_band(y: np.ndarray, sr: int, band: int) -> np.ndarray:
    """One NCM band, filtered in the frequency domain from the filter's poles
    and zeros, on a signal padded far past the filter's decay."""
    edges = intelligibility._ncm_edges(sr)
    z, p, k = butter(4, edges[band:band + 2] * (2.0 / sr), btype='bandpass',
                     output='zpk')
    size = 1 << int(math.ceil(math.log2(len(y) + 20 * sr)))
    _, h = freqz_zpk(z, p, k, worN=2 * np.pi * np.arange(size // 2 + 1) / size)
    return np.fft.irfft(np.fft.rfft(y, size) * h, size)[:len(y)]


@pytest.mark.parametrize("sr", [8000, 16000])
def test_the_bands_are_filtered_in_sections_because_the_transfer_function_drifts(sr):
    """The lowest band is 300 to 350 Hz at 8 kHz and 300 to 370 Hz at 16 kHz.
    In sections it follows the exact filter to 1e-12 of its peak; as the
    single transfer function NCM.m and pysepm use, it strays by more than
    1e-7 at 8 kHz and 1e-5 at 16 kHz."""
    top = {8000: 350.0, 16000: 370.0}[sr]
    assert intelligibility._ncm_edges(sr)[:2] == pytest.approx((300.0, top), abs=0.5)
    y = _speech(sr)
    exact = _exact_band(y, sr, 0)
    peak = np.max(np.abs(exact))
    sections = intelligibility._band(y, intelligibility._ncm_filterbank(sr)[2][0])
    assert np.max(np.abs(sections - exact)) < 1e-12 * peak
    b, a = butter(4, intelligibility._ncm_edges(sr)[:2] * (2.0 / sr), btype='bandpass')
    stray = np.max(np.abs(lfilter(b, a, y) - exact)) / peak
    assert stray > (1e-5 if sr == 16000 else 1e-7)


# -- the MATLAB code, written out again ------------------------------------------

def _matlab_resample(x: np.ndarray, q: int) -> np.ndarray:
    """MATLAB's ``resample(x, 1, q)`` step by step: firls's ideal low-pass
    (a sinc, in closed form for one band and no weights) under a Kaiser
    window of beta 5, scaled to a gain of 1, the delay padding and the cut."""
    half_length = 10 * q
    length = 2 * half_length + 1
    taps = np.sinc(np.arange(-half_length, half_length + 1) / q) / q
    taps = taps * np.kaiser(length, 5.0)
    taps = taps / taps.sum()
    lead = q - half_length % q
    taps = np.concatenate([np.zeros(lead), taps])
    delay = (half_length + lead) // q
    out_length = -(-len(x) // q)
    tail = 0
    while math.ceil((len(x) - 1 + len(taps) + tail) / q) - delay < out_length:
        tail += 1
    taps = np.concatenate([taps, np.zeros(tail)])
    return upfirdn(taps, x, 1, q)[delay:delay + out_length]


def _ncm_m(ref: np.ndarray, deg: np.ndarray, sr: int) -> float:
    """NCM.m, line by line, with its filters in sections and np.corrcoef for
    the normalized covariance."""
    edges = intelligibility._ncm_edges(sr)
    centres = (edges[:-1] + edges[1:]) / 2
    weights = np.interp(centres, intelligibility._ANSI_CENTRES,
                        intelligibility._ANSI_IMPORTANCE)
    ti = []
    for k in range(20):
        sos = butter(4, edges[k:k + 2] * (2.0 / sr), btype='bandpass', output='sos')
        x = _matlab_resample(np.abs(hilbert(sosfilt(sos, ref))), sr // 32)
        y = _matlab_resample(np.abs(hilbert(sosfilt(sos, deg))), sr // 32)
        r2 = np.corrcoef(x, y)[0, 1] ** 2
        snr = np.clip(10 * np.log10((r2 + EPS) / (1 - r2 + EPS)), -15, 15)
        ti.append((snr + 15) / 30)
    return float(weights @ np.array(ti) / weights.sum())


@pytest.mark.parametrize("sr", [8000, 16000])
def test_ncm_is_ncm_m_written_out(sr):
    for kind in ('white 5', 'room 0.6 s'):
        ref, deg = _degraded(kind, sr)
        assert ncm(ref, deg, sr) == pytest.approx(_ncm_m(ref, deg, sr), rel=1e-11), kind


@pytest.mark.parametrize("sr", [8000, 16000])
def test_the_envelopes_come_down_to_32_hz_as_matlabs_resample_brings_them(sr):
    """scipy's resample_poly, against MATLAB's resample written out: the same
    filter, the same padding and the same alignment, sample for sample."""
    y = np.abs(hilbert(_speech(sr, 2.0)))
    ours = resample_poly(y, 32, sr)
    assert len(ours) == math.ceil(len(y) / (sr // 32))
    assert np.max(np.abs(ours - _matlab_resample(y, sr // 32))) < 1e-14 * np.max(y)


def _csii_m(ref: np.ndarray, deg: np.ndarray, sr: int) -> tuple:
    """CSII.m, line by line: a frame at a time, two passes, eps added."""
    clean, processed = ref + EPS, deg + EPS
    size = int(math.floor(30 * sr / 1000 + 0.5))
    hop = size // 4
    nfft = int(2 ** math.ceil(math.log2(2 * size)))
    half = nfft // 2
    centre = np.array(intelligibility._ANSI_CENTRES[:16])
    width = np.array(intelligibility._CSII_WIDTHS)
    weight = np.array(intelligibility._ANSI_IMPORTANCE[:16])
    p = 4 * centre / width
    g = np.abs(1 - np.arange(half)[None, :] * (sr / nfft) / centre[:, None])
    filters = (1 + p[:, None] * g) * np.exp(-p[:, None] * g)
    count = int(math.floor(len(clean) / hop - size / hop))
    window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(1, size + 1) / (size + 1)))
    overall = np.linalg.norm(clean) / math.sqrt(len(processed))
    level, sums = [], {g_: [0, 0, 0] for g_ in range(3)}
    for m in range(count):
        c = clean[m * hop:m * hop + size]
        d = processed[m * hop:m * hop + size]
        db = 20 * math.log10(np.linalg.norm(c) / math.sqrt(size) / overall)
        group = 0 if db >= 0 else (1 if db >= -10 else 2)
        level.append(group)
        x = np.fft.fft(c * window, nfft)[:half]
        y = np.fft.fft(d * window, nfft)[:half]
        sums[group][0] = sums[group][0] + x * np.conj(y)
        sums[group][1] = sums[group][1] + np.abs(x) ** 2
        sums[group][2] = sums[group][2] + np.abs(y) ** 2
    r2 = {k: np.abs(v[0]) ** 2 / (v[1] * v[2]) for k, v in sums.items()
          if not np.isscalar(v[1])}
    scores = {0: [], 1: [], 2: []}
    for m in range(count):
        d = processed[m * hop:m * hop + size]
        y = np.abs(np.fft.fft(d * window, nfft)[:half]) ** 2
        msc = r2[level[m]]
        sdr = (filters @ (y * msc)) / (filters @ (y * (1 - msc)))
        t = (np.clip(10 * np.log10(sdr), -15, 15) + 15) / 30
        scores[level[m]].append(max(0.0, weight @ t / weight.sum()))
    return tuple(float(np.mean(scores[k])) for k in range(3))


@pytest.mark.parametrize("sr", [8000, 16000, 22050])
def test_csii_is_csii_m_written_out(sr):
    for kind in ('pink 0', 'room 0.6 s'):
        ref, deg = _degraded(kind, sr)
        got = csii(ref, deg, sr)
        want = _csii_m(ref, deg, sr)
        assert got == pytest.approx(want, rel=1e-11, abs=1e-13), kind


#: pysepm_evo 0.1.1's csii, versions as above, on the first 5251 samples of
#: ``_degraded('white 5', 11025)``, eps added to both.
PYSEPM_EVO_CSII_5251 = (0.5035479314359125, 0.4000818145481788, 0.08873180911727141)


def test_the_frame_count_rounds_where_csii_m_rounds_it(monkeypatch):
    """At 11.025 kHz a frame is 331 samples and the hop 82, so 5251 samples
    hold 60 hops after the first frame; CSII.m's floor(n / hop - size / hop)
    comes to 59.99999999999999 there and scores 59 frames, as pysepm does.
    The 60th frame would move the low level from 0.0887 to 0.0846."""
    size, hop = intelligibility._csii_setup(11025.0)[:2]
    assert (size, hop) == (331, 82) and (5251 - size) / hop == 60
    assert 5251 / hop - size / hop == 59.99999999999999
    assert intelligibility._frame_count(5251, size, hop) == 59
    ref, deg = _degraded('white 5', 11025)
    got = csii(ref[:5251], deg[:5251], 11025)
    assert got == pytest.approx(PYSEPM_EVO_CSII_5251, rel=1e-12, abs=1e-14)
    assert got == pytest.approx(_csii_m(ref[:5251], deg[:5251], 11025), rel=1e-11, abs=1e-13)
    assert got.low == pytest.approx(0.0887, abs=1e-4)
    monkeypatch.setattr(intelligibility, '_frame_count', lambda n, size, hop: (n - size) // hop)
    assert csii(ref[:5251], deg[:5251], 11025).low == pytest.approx(0.0846, abs=1e-4)


def test_the_ncm_bands_are_those_of_ma_hu_and_loizou_at_8_khz():
    """Their Table 2: the 20 centres in kHz and the weights for sentences,
    each within 1e-4 of the four places printed there."""
    centres_khz = (0.3249, 0.3775, 0.4356, 0.5000, 0.5713, 0.6502, 0.7376,
                   0.8344, 0.9416, 1.0602, 1.1915, 1.3370, 1.4980, 1.6763,
                   1.8737, 2.0922, 2.3342, 2.6022, 2.8989, 3.2274)
    sentences = (0.0772, 0.0955, 0.1016, 0.0908, 0.0734, 0.0659, 0.0580,
                 0.0500, 0.0460, 0.0440, 0.0445, 0.0482, 0.0488, 0.0488,
                 0.0493, 0.0491, 0.0520, 0.0549, 0.0555, 0.0514)
    edges, weights, _ = intelligibility._ncm_filterbank(8000)
    centres = (edges[:-1] + edges[1:]) / 2000
    assert np.max(np.abs(centres - centres_khz)) < 1e-4
    assert np.max(np.abs(weights - sentences)) < 1e-4
    assert (edges[0], edges[-1]) == (pytest.approx(300.0), pytest.approx(3400.0))
    edges16 = intelligibility._ncm_filterbank(16000)[0]
    assert (edges16[0], edges16[-1]) == (pytest.approx(300.0), pytest.approx(7400.0))


def test_the_csii_filters_peak_at_their_centres_and_the_table_sums_to_1():
    _, _, nfft, _, filters, weights = intelligibility._csii_setup(8000.0)
    peaks = np.argmax(filters, axis=1) * 8000 / nfft
    assert np.allclose(peaks, intelligibility._ANSI_CENTRES[:16], atol=8000 / nfft / 2)
    assert np.max(filters) <= 1.0
    assert sum(intelligibility._ANSI_IMPORTANCE) == pytest.approx(1.0)
    assert weights.sum() == pytest.approx(0.8736)


# -- copies, gains, delays and filters ---------------------------------------------

@pytest.mark.parametrize("sr", [8000, 16000])
def test_an_exact_scaled_or_inverted_copy_scores_1(sr):
    """r^2 and the coherence can round a little over 1 for these, which is
    where pysepm returns nan; both are held at 1 here."""
    clean = _speech(sr)
    for deg in (clean.copy(), 0.3 * clean, -clean, 7.0 * clean):
        assert _quiet(ncm, clean, deg, sr) == 1.0
        assert _quiet(csii, clean, deg, sr) == (1.0, 1.0, 1.0)


@pytest.mark.parametrize("sr", [8000, 16000])
def test_a_gain_on_either_signal_changes_nothing(sr):
    ref, deg = _pair(sr)
    assert ncm(0.01 * ref, 3.0 * deg, sr) == pytest.approx(ncm(ref, deg, sr), rel=1e-12)
    for a, b in zip(csii(ref, 0.1 * deg, sr), csii(ref, deg, sr)):
        assert a == pytest.approx(b, rel=1e-9)


@pytest.mark.parametrize("sr", [8000, 16000])
def test_more_noise_scores_lower(sr):
    clean = _speech(sr)
    noise = _noise(len(clean), 7, 'babble', sr)
    ncms, csiis = [], []
    for snr in (20, 10, 5, 0, -5):
        deg = _mix(clean, noise, snr)
        ncms.append(ncm(clean, deg, sr))
        csiis.append(csii(clean, deg, sr))
    assert all(a > b for a, b in zip(ncms, ncms[1:])), ncms
    for level in range(3):
        values = [c[level] for c in csiis]
        assert all(a > b for a, b in zip(values, values[1:])), (level, values)


def test_a_longer_room_scores_lower():
    clean = _speech(16000)
    rooms = [kudio.apply_rir(clean, kudio.rir(16000, rt60=t, seed=3)) for t in (0.3, 0.8, 1.5)]
    scores = [ncm(clean, wet, 16000) for wet in rooms]
    assert scores[0] > scores[1] > scores[2], scores
    mids = [csii(clean, wet, 16000).mid for wet in rooms]
    assert mids[0] > mids[2], mids


def test_an_output_of_digital_silence_scores_0_without_a_warning():
    """NCM.m and pysepm divide 0 by 0 here. CSII.m's eps leaves the output a
    spectrum, all of it at 0 Hz, and the test speech has no offset, so the
    coherence there is next to nothing (the next test adds one)."""
    clean = _speech(16000)
    silence = np.zeros_like(clean)
    assert _quiet(ncm, clean, silence, 16000) == 0.0
    assert _quiet(csii, clean, silence, 16000) == (0.0, 0.0, 0.0)


def test_an_offset_on_the_reference_credits_an_output_of_digital_silence():
    """A frame of digital silence has its power at 0 Hz alone, so CSII scores
    it by the coherence there, and an offset on the reference makes that
    high: 0.01% of the peak lifts the muted output's low level to 0.46, 0.1%
    gives 0.055, 0.542 and 0.831. With 0.3% on both signals, the output gated
    between the syllables scores 0.136 at the low level instead of 0.083.
    Taking the reference's mean out is no cure: the syllables of the test
    speech carry a little DC, and the pauses then carry it instead."""
    ref = _speech(16000)
    peak = np.max(np.abs(ref))
    muted = np.zeros_like(ref)
    assert _quiet(csii, ref + 1e-4 * peak, muted, 16000) == pytest.approx(
        (0.0, 0.0455, 0.4639), abs=1e-4)
    assert _quiet(csii, ref + 1e-3 * peak, muted, 16000) == pytest.approx(
        (0.0545, 0.5418, 0.8313), abs=1e-4)
    assert _quiet(csii, ref - ref.mean(), muted, 16000) == pytest.approx(
        (0.0, 0.0, 0.5245), abs=1e-4)
    assert _quiet(ncm, ref + 1e-3 * peak, muted, 16000) == 0.0
    clean, noisy = _pair(16000)
    gate = _speech(16000, floor=0) != 0
    offset = 3e-3 * peak
    assert csii(clean, np.where(gate, noisy, 0.0), 16000).low == pytest.approx(0.0829, abs=1e-4)
    shifted = csii(clean + offset, np.where(gate, noisy + offset, 0.0), 16000)
    assert shifted.low == pytest.approx(0.1361, abs=1e-4)


def _highpass_20(y: np.ndarray, sr: int) -> np.ndarray:
    """The cure the module docstring gives for an offset on the reference."""
    return sosfiltfilt(butter(2, 20, 'highpass', fs=sr, output='sos'), y)


def test_the_reference_high_passed_at_20_hz_takes_that_credit_away():
    """With the reference high-passed and the output's mean taken out, the
    muted output scores 0 again, the gated one 0.082 and a copy carrying an
    offset of 1% of its peak 0.999 at the low level, where it scored 0.937.
    Filtering the output as well rings into its digital silence: the gated
    output would score 0.109."""
    ref = _speech(16000)
    peak = np.max(np.abs(ref))
    muted = np.zeros_like(ref)
    assert _quiet(csii, _highpass_20(ref + 1e-3 * peak, 16000), muted, 16000) == (0.0, 0.0, 0.0)
    copy = ref + 0.01 * peak
    assert csii(_highpass_20(ref, 16000), copy - copy.mean(), 16000) == pytest.approx(
        (1.0, 1.0, 0.9987), abs=1e-4)
    clean, noisy = _pair(16000)
    gate = _speech(16000, floor=0) != 0
    offset = 3e-3 * peak
    gated = np.where(gate, noisy + offset, 0.0)
    reference = _highpass_20(clean + offset, 16000)
    assert csii(reference, gated - gated.mean(), 16000).low == pytest.approx(0.0815, abs=1e-4)
    assert csii(reference, _highpass_20(gated, 16000), 16000).low == pytest.approx(0.1086,
                                                                                abs=1e-4)


# -- the numbers in the module docstring -------------------------------------------

def test_the_test_pair_scores_what_the_docstring_says():
    """3 s of the test speech in white noise at 5 dB SNR: NCM 0.546 at
    16 kHz and 0.635 for the same pair taken down to 8 kHz, CSII 0.617,
    0.485 and 0.084 at either rate."""
    ref, deg = _pair(16000)
    assert ncm(ref, deg, 16000) == pytest.approx(0.5456, abs=1e-4)
    ref8, deg8 = resample_poly(ref, 1, 2), resample_poly(deg, 1, 2)
    assert ncm(ref8, deg8, 8000) == pytest.approx(0.6352, abs=1e-4)
    for scores in (csii(ref, deg, 16000), csii(ref8, deg8, 8000)):
        assert scores == pytest.approx((0.6173, 0.4845, 0.0839), abs=2e-4)


def test_25_hz_envelopes_would_score_the_test_pair_0_556(monkeypatch):
    """Ma, Hu and Loizou take the envelopes down to 25 Hz, NCM.m to 32."""
    ref, deg = _pair(16000)
    monkeypatch.setattr(intelligibility, '_ENVELOPE_RATE', 25)
    assert ncm(ref, deg, 16000) == pytest.approx(0.5560, abs=1e-4)


def _levels(ref: np.ndarray, sr: int) -> np.ndarray:
    """Each frame's level in dB against the whole, as csii() measures it."""
    size, hop = int(0.03 * sr), int(0.03 * sr) // 4
    count = (len(ref) - size) // hop
    clean = ref + EPS
    frames = np.array([clean[m * hop:m * hop + size] for m in range(count)])
    whole = np.sqrt(np.mean(clean ** 2))
    return 20 * np.log10(np.sqrt(np.mean(frames ** 2, axis=1)) / whole)


def test_the_low_level_takes_the_pauses_and_a_30_db_floor_would_not():
    """396 frames: 92 high, 78 mid and 226 low, of which 166 lie more than
    30 dB down. Leaving those out, as Ma, Hu and Loizou describe the low
    level, takes its CSII in white noise at 5 dB SNR from 0.084 to 0.198."""
    ref, deg = _pair(16000)
    level = _levels(ref, 16000)
    assert len(level) == 396
    assert (np.sum(level >= 0), np.sum((level < 0) & (level >= -10)),
            np.sum(level < -10), np.sum(level < -30)) == (92, 78, 226, 166)
    size, hop, nfft, window, filters, weights = intelligibility._csii_setup(16000.0)
    kept = (level < -10) & (level >= -30)
    frames_c = np.array([(ref + EPS)[m * hop:m * hop + size] for m in np.flatnonzero(kept)])
    frames_p = np.array([(deg + EPS)[m * hop:m * hop + size] for m in np.flatnonzero(kept)])
    x = np.fft.rfft(frames_c * window, nfft)[:, :nfft // 2]
    y = np.fft.rfft(frames_p * window, nfft)[:, :nfft // 2]
    msc = (np.abs(np.sum(x * np.conj(y), axis=0)) ** 2
           / (np.sum(np.abs(x) ** 2, axis=0) * np.sum(np.abs(y) ** 2, axis=0)))
    power = np.abs(y) ** 2
    sdr = ((power * msc) @ filters.T) / ((power * (1 - msc)) @ filters.T)
    floored = np.mean(((np.clip(10 * np.log10(sdr), -15, 15) + 15) / 30) @ weights
                      / weights.sum())
    assert csii(ref, deg, 16000).low == pytest.approx(0.0839, abs=1e-4)
    assert floored == pytest.approx(0.1981, abs=1e-4)


def test_a_copy_low_passed_at_2_khz_loses_no_csii_and_some_ncm():
    ref = _speech(16000)
    low = _lowpass(ref, 16000, 2000)
    assert csii(ref, low, 16000) == pytest.approx((1.0, 1.0, 1.0), abs=1e-6)
    assert ncm(ref, low, 16000) == pytest.approx(0.7818, abs=1e-4)
    ref8, low8 = resample_poly(ref, 1, 2), resample_poly(low, 1, 2)
    assert ncm(ref8, low8, 8000) == pytest.approx(0.9392, abs=1e-4)


def test_a_tilt_costs_neither_measure_anything():
    """Pre-emphasis, 1 - 0.95 z^-1: a two-sample filter, so the copy stays
    coherent in every bin, and each band's envelope keeps its shape."""
    ref = _speech(16000)
    tilted = lfilter([1.0, -0.95], [1.0], ref)
    assert csii(ref, tilted, 16000) == pytest.approx((1.0, 1.0, 1.0), abs=1e-9)
    assert ncm(ref, tilted, 16000) == pytest.approx(1.0, abs=1e-9)


def test_a_copy_5_and_10_ms_late():
    """CSII's coherence is over 30 ms frames, and a delay takes the frames
    apart: 0.63 at 5 ms and 0.36 at 10 ms on the high level. NCM's 32 Hz
    envelopes feel it less: 0.98 and 0.80."""
    ref = _speech(16000)
    five, ten = _late(ref, 5, 16000), _late(ref, 10, 16000)
    assert csii(ref, five, 16000).high == pytest.approx(0.6339, abs=1e-4)
    assert csii(ref, ten, 16000).high == pytest.approx(0.3588, abs=1e-4)
    assert ncm(ref, five, 16000) == pytest.approx(0.9808, abs=1e-4)
    assert ncm(ref, ten, 16000) == pytest.approx(0.7994, abs=1e-4)


def test_zeros_on_both_raise_ncm_and_lower_the_low_level_of_csii():
    """Half a second of zeros either side of both signals: quiet stretches
    in both count as agreement for NCM, and as more low-level frames, each
    scoring next to nothing, for CSII."""
    ref, deg = _pair(16000)
    zeros = np.zeros(8000)
    padded_ref = np.concatenate([zeros, ref, zeros])
    padded_deg = np.concatenate([zeros, deg, zeros])
    assert ncm(padded_ref, padded_deg, 16000) == pytest.approx(0.6509, abs=1e-4)
    assert csii(padded_ref, padded_deg, 16000).low == pytest.approx(0.0449, abs=1e-4)


def test_a_gate_between_the_syllables_raises_ncm_and_leaves_csii():
    """CSII moves by 0.001 at most, the test speech having no offset (see the
    offset tests above for what one does)."""
    ref, deg = _pair(16000)
    syllables = _speech(16000, floor=0) != 0
    gated = np.where(syllables, deg, 0.0)
    assert ncm(ref, gated, 16000) == pytest.approx(0.6422, abs=1e-4)
    moved = max(abs(a - b) for a, b in zip(csii(ref, gated, 16000), csii(ref, deg, 16000)))
    assert round(moved, 3) == 0.001


def test_noise_above_5_khz_costs_csii_little_and_ncm_more():
    ref = _speech(16000)
    hiss = sosfilt(butter(8, 5000, 'highpass', fs=16000, output='sos'),
                   _noise(len(ref), 9, 'white', 16000))
    deg = _mix(ref, hiss, 10)
    assert csii(ref, deg, 16000) == pytest.approx((0.9875, 0.9799, 0.9702), abs=1e-4)
    assert ncm(ref, deg, 16000) == pytest.approx(0.8835, abs=1e-4)


def test_a_dc_offset_costs_csii_at_the_low_level_and_not_ncm():
    ref = _speech(16000)
    peak = np.max(np.abs(ref))
    assert csii(ref, ref + 0.01 * peak, 16000).low == pytest.approx(0.9371, abs=1e-4)
    assert csii(ref, ref + 0.05 * peak, 16000).low == pytest.approx(0.8527, abs=1e-4)
    assert ncm(ref, ref + 0.05 * peak, 16000) == pytest.approx(1.0, abs=1e-6)


def test_short_clips_read_high_for_an_unrelated_output():
    """The first of the eight draws behind each average the docstring gives
    (the validation script runs all eight): an unrelated stretch of the same
    kind of speech scores NCM 0.375 against 1 s and 0 against 10 s, and white
    noise scores CSII 0.146 and 0.070 at the high and mid levels against 1 s
    and under 0.003 against 3 s."""
    assert ncm(_speech(16000, 1.0, seed=100), _speech(16000, 1.0, seed=200),
               16000) == pytest.approx(0.3754, abs=1e-4)
    assert ncm(_speech(16000, 10.0, seed=100), _speech(16000, 10.0, seed=200),
               16000) == pytest.approx(0.0, abs=1e-4)
    one = _speech(16000, 1.0, seed=100)
    noise = csii(one, _noise(len(one), 300, 'white', 16000), 16000)
    assert (noise.high, noise.mid) == pytest.approx((0.1459, 0.0702), abs=1e-4)
    three = _speech(16000, 3.0, seed=100)
    assert max(csii(three, _noise(len(three), 300, 'white', 16000), 16000)) < 0.003


@pytest.mark.parametrize("sr", [8000, 12000, 24000, 32000, 48000])
def test_csii_does_not_depend_on_the_rate_when_30_ms_is_whole(sr):
    """The pair made at 48 kHz and taken to each rate; the bands are fixed in
    hertz, so the score stays within 1e-4 of the one at 16 kHz."""
    from fractions import Fraction
    ref, deg = _pair(48000)
    at16 = csii(resample_poly(ref, 1, 3), resample_poly(deg, 1, 3), 16000)
    step = Fraction(sr, 48000)
    here = csii(resample_poly(ref, step.numerator, step.denominator),
                resample_poly(deg, step.numerator, step.denominator), sr)
    assert here == pytest.approx(at16, abs=1e-4)


# -- what is refused, and what scores nan ------------------------------------------

@pytest.mark.parametrize("sr", [11025, 22050, 44100, 48000, 16000.5])
def test_ncm_refuses_other_rates_with_the_way_out(sr):
    y = _speech(8000, 1.0)
    with pytest.raises(ValueError, match=r"^ncm\(\): .*kudio\.resample\(y, sr, 16000\)"):
        ncm(y, y, sr)


@pytest.mark.parametrize("sr", [7999, 4000])
def test_csii_refuses_rates_under_8_khz_with_the_way_out(sr):
    y = _speech(8000, 1.0)
    with pytest.raises(ValueError, match=r"^csii\(\): .*kudio\.resample\(y, sr, 16000\)"):
        csii(y, y, sr)


@pytest.mark.parametrize("measure", [ncm, csii])
@pytest.mark.parametrize("sr", [0, -16000, float('nan'), float('inf'), True, '16000', None])
def test_a_rate_that_is_not_a_positive_number_is_refused(measure, sr):
    y = _speech(8000, 1.0)
    with pytest.raises(ValueError, match=rf"^{measure.__name__}\(\): sr has to be a "
                                         r"positive sample rate"):
        measure(y, y, sr)


def test_a_rate_given_as_a_float_or_a_numpy_integer_is_a_rate():
    ref, deg = _pair(16000)
    assert ncm(ref, deg, 16000.0) == ncm(ref, deg, np.int64(16000)) == ncm(ref, deg, 16000)
    assert csii(ref, deg, 16000.0) == csii(ref, deg, np.int32(16000)) == csii(ref, deg, 16000)
    with pytest.warns(UserWarning, match=r"\(62\.6 ms at 16000 Hz\)"):
        ncm(ref[:1000], deg[:1000], 16000.0)


@pytest.mark.parametrize("measure", [ncm, csii])
def test_a_rate_in_a_0_d_array_is_a_rate_and_a_longer_array_is_not(measure):
    ref, deg = _pair(16000)
    assert measure(ref, deg, np.array(16000)) == measure(ref, deg, 16000)
    for sr in (np.array([16000]), np.array([8000, 16000])):
        with pytest.raises(ValueError, match=rf"^{measure.__name__}\(\): sr has to be"):
            measure(ref, deg, sr)


@pytest.mark.parametrize("measure", [ncm, csii])
@pytest.mark.parametrize("shape", [(16000, 2), (2, 16000), (2, 8000, 1)])
def test_more_than_one_channel_is_refused_with_a_way_out(measure, shape):
    y = np.zeros(shape) + 0.1
    with pytest.raises(ValueError, match=rf"^{measure.__name__}\(\): ref has shape "
                                         r".*one at a time, or mix down"):
        measure(y, y, 16000)


@pytest.mark.parametrize("measure", [ncm, csii])
def test_one_channel_as_a_column_or_a_row_is_read_as_mono(measure):
    ref, deg = _pair(8000)
    assert measure(ref[:, None], deg[None, :], 8000) == measure(ref, deg, 8000)
    assert measure(ref[None, :, None], deg[:, None], 8000) == measure(ref, deg, 8000)


@pytest.mark.parametrize("measure", [ncm, csii])
@pytest.mark.parametrize("shape", [(0,), (0, 1), (1, 0), (1, 1)])
def test_an_empty_column_is_an_empty_clip_and_scores_nan(measure, shape):
    with pytest.warns(UserWarning, match=r"needs at least .* has [01]; scored nan"):
        value = measure(np.zeros(shape), np.zeros(shape), 16000)
    assert np.all(np.isnan(value))


@pytest.mark.parametrize("measure", [ncm, csii])
def test_a_single_number_is_refused(measure):
    y = _speech(16000, 1.0)
    with pytest.raises(ValueError, match=rf"^{measure.__name__}\(\): ref is a single "
                                         r"number, not a signal"):
        measure(np.float64(0.25), y, 16000)
    with pytest.raises(ValueError, match=r"deg is a single number"):
        measure(y, np.array(0.25), 16000)


@pytest.mark.parametrize("measure", [ncm, csii])
def test_samples_that_are_not_real_numbers_are_refused(measure):
    """Complex samples would lose their imaginary part without a word, and
    text, flags or objects are not audio."""
    y = _speech(8000, 1.0)
    with pytest.raises(ValueError, match=rf"^{measure.__name__}\(\): deg is complex "
                                         r"\(complex128\); pass the real signal"):
        measure(y, y + 0j, 8000)
    for bad in (y.astype(str), y > 0, y.astype(object)):
        with pytest.raises(ValueError, match=rf"^{measure.__name__}\(\): ref holds "
                                             rf"{bad.dtype} values, not audio samples"):
            measure(bad, y, 8000)


@pytest.mark.parametrize("measure", [ncm, csii])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_samples_that_are_not_finite_are_refused(measure, bad):
    ref, deg = _pair(8000)
    corrupt = deg.copy()
    corrupt[123] = bad
    with pytest.raises(ValueError, match=rf"^{measure.__name__}\(\): deg has 1 "
                                         r"sample\(s\) that are not finite"):
        measure(ref, corrupt, 8000)
    with pytest.raises(ValueError, match=r"ref has 1 sample"):
        measure(corrupt, ref, 8000)


@pytest.mark.parametrize("measure", [ncm, csii])
def test_a_length_mismatch_warns_at_the_callers_line_and_a_large_one_is_refused(measure):
    ref, deg = _pair(8000)
    with pytest.warns(UserWarning, match=rf"^{measure.__name__}\(\): ref is 24000 "
                                         r"samples and deg 23900: only the "
                                         r"first") as caught:
        value = measure(ref, deg[:-100], 8000)
    assert caught[0].filename == __file__
    assert np.all(np.isfinite(value))
    with pytest.raises(ValueError, match=rf"^{measure.__name__}\(\): .*unmeasured"):
        measure(ref, deg[:10000], 8000)


@pytest.mark.parametrize("measure", [ncm, csii])
def test_a_mismatch_keeps_the_first_samples_of_the_longer_signal(measure):
    """Whichever one is longer, the pair is scored over its first samples, as
    the warning says: an output 10 ms too long loses its end, and so does a
    reference."""
    ref, deg = _pair(8000)
    tail = _noise(80, 11, 'white', 8000)
    whole = measure(ref, deg, 8000)
    with pytest.warns(UserWarning, match="only the first 24000 samples"):
        assert measure(ref, np.concatenate([deg, tail]), 8000) == whole
    with pytest.warns(UserWarning, match="only the first 24000 samples"):
        assert measure(np.concatenate([ref, tail]), deg, 8000) == whole


@pytest.mark.parametrize("measure", [ncm, csii])
def test_a_pair_right_at_the_length_limit_is_scored_and_one_sample_past_is_not(measure):
    """Half of the longer one unmeasured is still within the 50% limit, as in
    the evaluator; one more sample is past it."""
    assert evaluator.MAX_LENGTH_MISMATCH == 0.5
    ref, deg = _pair(8000)
    with pytest.warns(UserWarning, match="only the first 12000 samples"):
        value = measure(ref, deg[:12000], 8000)
    assert np.all(np.isfinite(value))
    with pytest.raises(ValueError, match=r"leaves 12001 of the longer one's 24000 samples "
                                         r"unmeasured"):
        measure(ref, deg[:11999], 8000)


@pytest.mark.parametrize("measure", [ncm, csii])
def test_the_length_limit_is_kudios_and_can_be_raised(measure, monkeypatch):
    ref, deg = _pair(8000)
    monkeypatch.setattr(evaluator, 'MAX_LENGTH_MISMATCH', 0.7)
    with pytest.warns(UserWarning, match="only the first 10000 samples"):
        value = measure(ref, deg[:10000], 8000)
    assert np.all(np.isfinite(value))


@pytest.mark.parametrize("sr, need", [(8000, 501), (16000, 1001)])
def test_ncm_needs_three_envelope_samples(sr, need):
    """Through two envelope samples any two envelopes correlate perfectly: a
    pair of 2 * sr / 32 samples would score 1 against anything."""
    assert intelligibility._transmission_index(np.array([0.2, 0.9]),
                                               np.array([0.5, 0.1])) == 1.0
    ref = _speech(sr, 1.0)
    noise = _noise(len(ref), 5, 'white', sr)
    with pytest.warns(UserWarning, match=rf"^ncm\(\): the pair needs at least {need} samples "
                                         rf"\(62\.6 ms at {sr} Hz\).*has {need - 1}; "
                                         r"scored nan") as caught:
        assert math.isnan(ncm(ref[:need - 1], noise[:need - 1], sr))
    assert caught[0].filename == __file__
    assert math.isfinite(_quiet(ncm, ref[:need], noise[:need], sr))


@pytest.mark.parametrize("sr, need", [(8000, 360), (16000, 720), (44100, 1983)])
def test_csii_needs_two_frames(sr, need):
    """Over one frame the coherence is 1 in every bin, whatever deg holds, so
    two frames are the fewest a level can be scored over. Whether two are
    enough depends on where they fall; the pair is not refused for length."""
    ref = _speech(sr, 1.0)
    deg = _mix(ref, _noise(len(ref), 5, 'white', sr), 0)
    with pytest.warns(UserWarning, match=rf"^csii\(\): the pair needs at least {need} samples "
                                         rf"\(45\.0 ms .*has {need - 1}; scored "
                                         r"nan") as caught:
        assert np.isnan(csii(ref[:need - 1], deg[:need - 1], sr)).all()
    assert len(caught) == 1 and caught[0].filename == __file__
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        csii(ref[:need], deg[:need], sr)
    assert not [w for w in caught if "needs at least" in str(w.message)]


@pytest.mark.parametrize("measure", [ncm, csii])
def test_an_empty_pair_scores_nan_with_a_warning(measure):
    with pytest.warns(UserWarning, match="needs at least .* has 0; scored nan"):
        value = measure(np.zeros(0), np.zeros(0), 16000)
    assert np.all(np.isnan(value))


@pytest.mark.parametrize("measure", [ncm, csii])
def test_a_reference_of_digital_silence_scores_nan_with_a_warning(measure):
    deg = _speech(16000, 1.0)
    with pytest.warns(UserWarning, match=rf"^{measure.__name__}\(\): ref is digital "
                                         r"silence throughout.*scored nan") as caught:
        value = measure(np.zeros_like(deg), deg, 16000)
    assert np.all(np.isnan(value))
    assert len(caught) == 1 and caught[0].filename == __file__


def test_a_level_without_frames_scores_nan_with_one_warning():
    """Steady noise never falls 10 dB under its own RMS over 30 ms."""
    ref = _noise(32000, 5, 'white', 16000)
    deg = ref + 0.1 * _noise(32000, 6, 'white', 16000)
    with pytest.warns(UserWarning, match=r"^csii\(\): no frame of ref is at the low "
                                         r"level.*; scored nan$") as caught:
        got = csii(ref, deg, 16000)
    assert len(caught) == 1 and caught[0].filename == __file__
    assert np.isnan(got.low) and np.isfinite(got.high) and np.isfinite(got.mid)


def test_a_level_of_one_frame_scores_nan_and_both_reasons_share_one_warning():
    """A click in the first 7.5 ms is in the first frame alone, which puts
    that frame, and no other, at the high level; every other frame holds
    only noise 60 dB under the click, and falls below the mid level."""
    rng = np.random.default_rng(0)
    ref = 1e-3 * rng.standard_normal(16000)
    ref[10] = 1.0
    deg = _noise(16000, 4, 'white', 16000)
    with pytest.warns(UserWarning) as caught:
        got = csii(ref, deg, 16000)
    assert len(caught) == 1
    message = str(caught[0].message)
    assert "only one frame of ref is at the high level" in message
    assert "no frame of ref is at the mid level" in message
    assert np.isnan(got.high) and np.isnan(got.mid) and np.isfinite(got.low)


def test_csii_of_a_signal_too_loud_to_square_is_nan_with_a_warning():
    """At 1e160 of full scale a frame's power overflows. NCM scales both
    signals to a peak of 1 first and is not troubled."""
    ref, deg = _pair(8000)
    with pytest.warns(UserWarning, match=r"^csii\(\): ref is too loud to "
                                         r"square") as caught:
        assert np.isnan(csii(ref * 1e160, deg, 8000)).all()
    assert len(caught) == 1
    with pytest.warns(UserWarning, match=r"could not be measured, deg being too loud"):
        assert np.isnan(csii(ref, deg * 1e160, 8000)).all()
    assert ncm(ref * 1e160, deg * 1e160, 8000) == pytest.approx(ncm(ref, deg, 8000),
                                                                rel=1e-12)


def test_integer_audio_is_read_on_its_own_full_scale():
    """An int16 reference against its float copy is a copy, and CSII's eps
    sits where it would for the float file."""
    ref = _speech(8000)
    as_int = np.round(ref * 32767).astype(np.int16)
    back = as_int / 32768.0
    assert ncm(as_int, back, 8000) == pytest.approx(1.0)
    assert csii(as_int, back, 8000) == pytest.approx((1.0, 1.0, 1.0))
    deg = _mix(ref, _noise(len(ref), 7, 'white', 8000), 5)
    deg_int = np.round(deg / np.max(np.abs(deg)) * 32767).astype(np.int16)
    assert csii(as_int, deg_int, 8000) == csii(as_int / 32768.0, deg_int / 32768.0, 8000)


def test_the_score_is_a_named_tuple_of_three_floats():
    ref, deg = _pair(8000)
    got = csii(ref, deg, 8000)
    assert isinstance(got, CsiiScore) and got._fields == ('high', 'mid', 'low')
    assert all(type(v) is float for v in got)
    assert type(ncm(ref, deg, 8000)) is float


def test_long_pairs_are_framed_in_blocks_with_the_same_result(monkeypatch):
    """Ten minutes at 16 kHz is 79,996 frames, 655 MB of 512-bin spectra a
    signal if they were taken all at once."""
    assert intelligibility._frame_count(600 * 16000, 480, 120) == 79996
    assert round(79996 * 512 * 16 / 1e6) == 655
    ref, deg = _pair(8000)
    whole = csii(ref, deg, 8000)
    monkeypatch.setattr(intelligibility, '_BLOCK', 7)
    assert csii(ref, deg, 8000) == pytest.approx(whole, rel=1e-12)


def test_csii_holds_one_block_of_frames_at_a_time(monkeypatch):
    """The frame levels are taken a block at a time too: over all frames at
    once, squaring them would take four times the memory of a signal, the
    frames overlapping by three quarters, on top of the two signals with eps
    added. With small blocks, csii() needs those two and little more."""
    import tracemalloc
    monkeypatch.setattr(intelligibility, '_BLOCK', 64)
    ref, deg = _pair(16000)
    ref, deg = np.tile(ref, 10), np.tile(deg, 10)          # 30 s
    csii(ref[:16000], deg[:16000], 16000)                 # the setup, cached
    tracemalloc.start()
    try:
        csii(ref, deg, 16000)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 4 * ref.nbytes, peak / ref.nbytes


def test_the_module_reads_as_python_3_9():
    import ast
    with open(intelligibility.__file__, encoding='utf-8') as f:
        ast.parse(f.read(), feature_version=(3, 9))
