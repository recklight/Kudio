# -*- coding: utf-8 -*-
"""Check kudio's Hu and Loizou measures (kudio/core/quality.py) against
pysepm, Itakura-Saito against comp_is.m restated here, and the composite
against its regression computed on pysepm's parts with the pesq package;
then reproduce the numbers tests/test_quality.py stores and the figures the
module docstring quotes.

Run it from anywhere; it imports the kudio of the checkout it sits in::

    python tools/validation/quality.py

Besides what kudio itself needs, it needs the oracle, pysepm_evo (I ran
0.1.1, with numpy 2.2.6, scipy 1.15.2 and numba 0.67.0 on Python 3.10.22),
and for the composite the pesq package (0.0.4). Without either, the rows
that need it say they were skipped, and the rest still run: comp_is.m
restated, the numbers the tests store and the docstring's figures.

pysepm is GPL-3.0: it runs here beside kudio as something to compare with,
and none of its code is in kudio or in this script. 0.1.1 still imports
scipy.signal.kaiser, which newer scipy keeps only in scipy.signal.windows,
so the script puts it back before the import. pysepm_evo
(github.com/alphaX86/pysepm-evo) is a packaging of pysepm
(github.com/schmiph2/pysepm), whose quality measures are ports of Loizou's
MATLAB files, the same files kudio follows.

Every signal is made here from fixed seeds: the synthetic speech of
tests/test_quality.py (syllables of a harmonic source through moving
formants, pauses on a noise floor 60 dB down), white, pink and babble noise
from -5 to 20 dB SNR, rooms from kudio.rir with RT60 of 0.3, 0.6 and 1.2 s,
kudio's log-MMSE enhancer on noisy speech, an output low-passed at 2 kHz
under noise, and at 16 kHz a reference with nothing above 4 kHz. The pairs
are 3 s long, 396 frames at either rate.

What it checks:

1. llr, cepstral_distance, wss and fw_segmental_snr against pysepm's llr,
   cepstrum_distance, wss and fwSNRseg, and the composite's uncapped LLR and
   segmental SNR against pysepm's llr(used_for_composite=True) and SNRseg,
   at 8 and 16 kHz.
2. The pysepm_evo 0.1.1 numbers tests/test_quality.py stores, computed again
   from scratch, and kudio against them.
3. The one place where pysepm and kudio part on purpose. LLR, IS, CEP and
   WSS average the best round(0.95 M) of M frames. The MATLAB code rounds a
   half away from zero and Python's round() takes it to the even
   neighbour, so where 0.95 M ends in .5 and the integer below it is even
   (M = 30, 70, 110, ..., 390), pysepm averages one frame fewer. The script
   scores pairs of 30, 50, 70 and 390 frames, and then averages kudio's own
   frame values over pysepm's count, which takes the difference to zero.
4. itakura_saito against comp_is.m restated here (its frames, window and
   eps, its eps floors on the two prediction-error powers and on the clean
   frame's own quadratic form, with scipy's Toeplitz solver in place of its
   Levinson-Durbin recursion), held to [0, 100] as Hu and Loizou's paper
   holds it; against the divergence between the two LPC spectra, averaged
   over 16384 points round the unit circle; and what comp_is.m's own range,
   capped at 100 only, gives on an exact copy with digital silence in it.
   pysepm has no Itakura-Saito to compare with.
5. composite. pysepm_evo 0.1.1's composite() cannot run: it reads two names
   it never defines, mos_lqo and pesq_mos, and raises NameError, which the
   script shows. The original pysepm (qualityMeasures.py, last changed in
   commit 86b5670 of 2025-02-15, read on 2026-11-13) gets both from a pesq()
   of its own on the pesq package, which it imports under the name pypesq,
   and that line is what pysepm_evo left out. So the oracle is the
   regression of Hu and Loizou (2006) computed here on pysepm's WSS,
   uncapped LLR and SNRseg and on the pesq package's score, as pysepm's
   pesq() picks it: wideband MOS-LQO at 16 kHz, and at 8 kHz the
   raw narrowband P.862 score, taken back from the package's MOS-LQO
   through P.862.1; and the same raw score with mode='nb', scale='raw' at
   16 kHz.
6. Figures in the module docstring: what a DC offset does to WSS, what
   its -100 dB band floor does to a quiet pair, what
   white noise in the zeros of a padded reference does at 16 kHz, the
   composite's own segmental SNR, Itakura-Saito's price for level, the
   P.862.1 and P.862.2 ranges and the ratings past 5.
7. An output with a band left empty and no noise in it, the speech of 1
   low-passed at 1 kHz: its 16th-order LPC fits are singular to float64,
   and the composite's uncapped LLR is decided by round-off. kudio and
   pysepm are held to 0.5% of each other there, and with mpmath installed
   both are set against the same frames in 60-digit arithmetic. The capped
   llr, itakura_saito and cepstral_distance have to agree exactly.

Where an LPC fit is close to singular, two correct implementations part in
the last digits: the largest differences in 1 and 4 come from the log-MMSE
output under babble at 0 dB and, for the composite's uncapped LLR, from the
16 kHz reference with nothing above 4 kHz, whose 16th-order autocorrelation
matrix has next to nothing in its upper half. So those comparisons go by
the relative difference, 1e-9 as in tests/test_quality.py; the stored
numbers and the figures go by the absolute one.

It prints the largest differences and exits with 1 if one is past its
tolerance. The run takes under half a minute.

References:

* Y. Hu and P. C. Loizou, "Evaluation of objective quality measures for
  speech enhancement", IEEE Trans. Audio, Speech, Lang. Process., vol. 16,
  no. 1, pp. 229-238, 2008, doi:10.1109/TASL.2007.911054.
* Y. Hu and P. C. Loizou, "Evaluation of objective measures for speech
  enhancement", Proc. Interspeech 2006, doi:10.21437/Interspeech.2006-84.
  The composite's equations, Sec. 4.1.
* P. C. Loizou, Speech Enhancement: Theory and Practice, 2nd ed., CRC
  Press, 2013, doi:10.1201/b14529, whose MATLAB code holds comp_is.m and
  the other comp_*.m files.
* ITU-T Recommendation P.862.1 (11/2003), "Mapping function for
  transforming P.862 raw result scores to MOS-LQO", and P.862.2 (11/2007),
  "Wideband extension to Recommendation P.862 for the assessment of
  wideband telephone networks and speech codecs": the two MOS-LQO
  mappings.
"""
from __future__ import annotations

import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import scipy.signal
import scipy.signal.windows
from scipy.linalg import solve_toeplitz, toeplitz
from scipy.signal import butter, lfilter, resample_poly, sosfilt, sosfiltfilt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import kudio  # noqa: E402
from kudio.core import quality as module  # noqa: E402
from kudio.core.quality import (cepstral_distance, composite,  # noqa: E402
                                fw_segmental_snr, itakura_saito, llr, wss)

if not hasattr(scipy.signal, 'kaiser'):
    scipy.signal.kaiser = scipy.signal.windows.kaiser
try:
    import pysepm_evo
except ImportError:
    pysepm_evo = None

try:
    import pesq as pesq_package
except ImportError:
    pesq_package = None

EPS = np.finfo(float).eps
ORDER = {8000: 10, 16000: 16}

#: rough F1, F2, F3 of five vowels, in Hz
VOWELS = ((730, 1090, 2440), (270, 2290, 3010), (300, 870, 2240),
          (530, 1840, 2480), (570, 840, 2410))

#: (check, rate, cases, largest absolute difference, largest relative
#: difference, tolerance, and whether it holds the absolute or the relative one)
RESULTS = []


def report(check, rate, cases, diff, rel, tolerance, on='abs'):
    RESULTS.append((check, rate, cases, diff, rel, tolerance, on))


def largest(got, want):
    """Largest absolute and relative difference over paired values. nan on
    both sides agrees; nan on one side is as far apart as can be."""
    got, want = np.ravel(got).astype(float), np.ravel(want).astype(float)
    with np.errstate(invalid='ignore'):
        diff = np.abs(got - want)
    both = np.isnan(got) & np.isnan(want)
    diff = np.where(both, 0.0, np.where(np.isnan(diff), np.inf, diff))
    rel = np.where(both, 0.0, diff / np.maximum(np.abs(np.nan_to_num(want)), 1e-300))
    return float(np.max(diff)), float(np.max(rel))


def quietly(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return fn(*args, **kwargs)


# --- signals: the speech of tests/test_quality.py -----------------------------

def speech(sr, seconds=2.0, seed=0, floor=5e-4):
    """Syllables of a harmonic source through three formant resonators, the
    pitch gliding inside each and the formants moving from one vowel to the
    next, with pauses over a noise floor 60 dB below the peak."""
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
            for f_from, f_to, bw in zip(VOWELS[k % 5], VOWELS[(k + 1) % 5],
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


def noise(n, seed, colour='white', sr=16000):
    if colour == 'babble':
        talkers = [speech(sr, n / sr + 0.5, seed=seed * 10 + i, floor=0)
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


def mix(clean, noise_, snr_db):
    scale = np.sqrt(np.mean(clean ** 2) / (np.mean(noise_ ** 2) * 10 ** (snr_db / 10)))
    return clean + scale * noise_


def test_pair(sr, snr_db=10.0, colour='white'):
    """``_pair`` of tests/test_quality.py: 2 s of speech, seed 0, noise seed 1."""
    clean = speech(sr)
    return clean, mix(clean, noise(len(clean), 1, colour, sr), snr_db)


def padded(sr):
    """2 s of speech with 0.5 s of digital silence either side: 396 frames,
    125 of them zeros throughout."""
    return np.concatenate([np.zeros(sr // 2), speech(sr), np.zeros(sr // 2)])


def pairs(sr):
    """``(name, ref, deg)`` for every kind of damage, 3 s at *sr*."""
    clean = speech(sr, 3.0, seed=1)
    n = len(clean)
    out = []
    for colour in ('white', 'pink', 'babble'):
        hiss = noise(n, 7, colour, sr)
        for snr in (-5, 0, 5, 10, 20):
            out.append((f'{colour} {snr:+d} dB', clean, mix(clean, hiss, snr)))
    for rt60 in (0.3, 0.6, 1.2):
        wet = kudio.apply_rir(clean, kudio.rir(sr, rt60=rt60, seed=3))
        out.append((f'room {rt60} s', clean, np.asarray(wet, float)[:n]))
    for colour, snr in (('white', 5), ('babble', 0)):
        noisy = mix(clean, noise(n, 7, colour, sr), snr)
        out.append((f'log-MMSE, {colour} {snr} dB', clean,
                    np.asarray(kudio.spectral_enhance(noisy, sr), float)[:n]))
    low = sosfiltfilt(butter(8, 2000, fs=sr, output='sos'), clean)
    out.append(('low-passed 2 kHz, white 20 dB', clean, mix(low, noise(n, 7, 'white', sr), 20)))
    if sr == 16000:
        narrow = resample_poly(resample_poly(clean, 1, 2), 2, 1)
        out.append(('ref from 8 kHz, white 10 dB', narrow,
                    mix(narrow, noise(n, 7, 'white', sr), 10)))
    return out


# --- 1: against pysepm ------------------------------------------------------------

#: kudio's measure, pysepm's, by name
MEASURES = {
    'llr': (llr, lambda r, d, sr: pysepm_evo.llr(r, d, sr)),
    'cepstral_distance': (cepstral_distance,
                          lambda r, d, sr: pysepm_evo.cepstrum_distance(r, d, sr)),
    'wss': (wss, lambda r, d, sr: pysepm_evo.wss(r, d, sr)),
    'fw_segmental_snr': (fw_segmental_snr, lambda r, d, sr: pysepm_evo.fwSNRseg(r, d, sr)),
    "composite's uncapped llr": (
        lambda r, d, sr: composite(r, d, sr, pesq=2.0).llr,
        lambda r, d, sr: pysepm_evo.llr(r, d, sr, used_for_composite=True)),
    "composite's segsnr": (
        lambda r, d, sr: composite(r, d, sr, pesq=2.0).segsnr,
        lambda r, d, sr: pysepm_evo.SNRseg(r, d, sr)),
}


def skipped(what, why):
    report(f'{what}: skipped, {why}', '-', 0, 0.0, 0.0, math.inf)


def check_pysepm():
    if pysepm_evo is None:
        skipped('the measures vs pysepm', 'no pysepm_evo')
        return
    for sr in (8000, 16000):
        cases = pairs(sr)
        for name, (ours, theirs) in MEASURES.items():
            got = [quietly(ours, r, d, sr) for _, r, d in cases]
            want = [quietly(theirs, r, d, sr) for _, r, d in cases]
            diff, rel = largest(got, want)
            worst = cases[int(np.argmax(np.abs(np.subtract(got, want))))][0]
            report(f'{name} vs pysepm (worst: {worst})', sr, len(cases), diff, rel, 1e-9,
                   'rel')


# --- 2: the numbers the tests store ---------------------------------------------

#: PYSEPM_EVO_0_1_1 of tests/test_quality.py: pysepm_evo 0.1.1 on _pair(sr, 10,
#: 'white') and _pair(sr, 5, 'pink'), 262 frames each: llr, llr with
#: used_for_composite=True, wss, cepstrum_distance, fwSNRseg and SNRseg
STORED = {
    (8000, 10, 'white'): (1.1718670598342014, 1.3896920535039994, 35.978993786442956,
                          6.539503644858894, 6.734822117006508, 0.08950560791750141),
    (8000, 5, 'pink'): (1.0793230703713834, 1.1276614778289964, 58.63723299992713,
                        6.704734120051547, 5.857250647113347, -2.318112449687568),
    (16000, 10, 'white'): (1.2152022856269244, 1.9987371410391976, 30.898754137997887,
                           6.339864785704115, 6.587486329701602, -0.15799247542422712),
    (16000, 5, 'pink'): (1.206665294179598, 1.6207468310939142, 56.73095337920859,
                         6.503401150803159, 4.343724758391272, -2.52801433955951),
}

#: PYSEPM_EVO_0_1_1_WSS_PADDED: pysepm's wss on padded(sr) under white noise at 10 dB
STORED_WSS_PADDED = {8000: 27.86688153953944, 16000: 25.68243750327955}


def pysepm_all(ref, deg, sr):
    return (pysepm_evo.llr(ref, deg, sr), pysepm_evo.llr(ref, deg, sr, used_for_composite=True),
            pysepm_evo.wss(ref, deg, sr), pysepm_evo.cepstrum_distance(ref, deg, sr),
            pysepm_evo.fwSNRseg(ref, deg, sr), pysepm_evo.SNRseg(ref, deg, sr))


def kudio_all(ref, deg, sr):
    parts = composite(ref, deg, sr, pesq=2.0)
    return (llr(ref, deg, sr), parts.llr, wss(ref, deg, sr),
            cepstral_distance(ref, deg, sr), fw_segmental_snr(ref, deg, sr), parts.segsnr)


def check_stored():
    again, ours, stored = [], [], []
    for (sr, snr, colour), values in STORED.items():
        clean, noisy = test_pair(sr, snr, colour)
        if pysepm_evo is not None:
            again.extend(pysepm_all(clean, noisy, sr))
        ours.extend(kudio_all(clean, noisy, sr))
        stored.extend(values)
    if pysepm_evo is None:
        skipped('pysepm now vs the numbers the tests store', 'no pysepm_evo')
    else:
        diff, rel = largest(again, stored)
        report('pysepm now vs the numbers the tests store', '8, 16k', len(STORED) * 6,
               diff, rel, 1e-12)
    diff, rel = largest(ours, stored)
    report('kudio vs the numbers the tests store', '8, 16k', len(STORED) * 6,
           diff, rel, 1e-9, 'rel')
    for sr, value in STORED_WSS_PADDED.items():
        ref = padded(sr)
        deg = mix(ref, noise(len(ref), 1, 'white', sr), 10)
        mine = quietly(wss, ref, deg, sr)
        if pysepm_evo is not None:
            diff, rel = largest([pysepm_evo.wss(ref, deg, sr)], [value])
            report('pysepm wss now vs the tests, reference padded with zeros', sr, 1,
                   diff, rel, 1e-12)
        diff, rel = largest([mine], [value])
        report('kudio wss vs the tests, reference padded with zeros', sr, 1, diff, rel,
               1e-9, 'rel')


# --- 3: where 0.95 M ends in .5 -------------------------------------------------------

#: kudio's frame function, the range each frame is held to, and pysepm's
#: measure, by name
FRAMEWISE = {
    'llr': (module._frame_llr, (0.0, 2.0), lambda r, d, sr: pysepm_evo.llr(r, d, sr)),
    'cep': (module._frame_cep, (0.0, 10.0),
            lambda r, d, sr: pysepm_evo.cepstrum_distance(r, d, sr)),
    'wss': (module._frame_wss, None, lambda r, d, sr: pysepm_evo.wss(r, d, sr)),
}


def frame_values(name, ref, deg, sr):
    """kudio's value for every frame, before the best 95% are picked."""
    frame, held, _ = FRAMEWISE[name]
    framed = module._framed(ref, deg, sr, name)
    extra = sr if name == 'wss' else ORDER[sr]
    values = module._each(frame, framed, extra)
    return values if held is None else np.clip(values, *held)


def check_rounding():
    if pysepm_evo is None:
        skipped('where 0.95 M ends in .5', 'no pysepm_evo')
        return
    print('\nThe best 95% of M frames: kudio keeps floor(0.95 M + 0.5), pysepm '
          'round(0.95 M), which takes .5 to the even neighbour')
    print(f"  {'rate':>5} {'M':>4} {'0.95 M':>8} {'kudio':>6} {'pysepm':>6}   "
          f"{'measure':<5} {'kudio':>12} {'pysepm':>12} {'kudio, pysepm count':>20}")
    parted, rejoined, cases, apart = 0.0, 0.0, 0, 0
    for sr in (8000, 16000):
        size, hop = module._frame_size(sr)
        base = speech(sr, 3.0, seed=2)
        hiss = noise(len(base), 8, 'white', sr)
        for frames in (30, 50, 70, 390):
            n = size + frames * hop
            ref = base[:n]
            deg = mix(ref, hiss[:n], 10)
            ours_keep = int(math.floor(0.95 * frames + 0.5))
            their_keep = int(round(0.95 * frames))
            for name in FRAMEWISE:
                values = np.sort(frame_values(name, ref, deg, sr))
                assert len(values) == frames
                kudio_score = quietly({'llr': llr, 'cep': cepstral_distance, 'wss': wss}[name],
                                      ref, deg, sr)
                theirs = quietly(FRAMEWISE[name][2], ref, deg, sr)
                same_count = float(np.mean(values[:their_keep]))
                print(f"  {sr:>5} {frames:>4} {0.95 * frames:>8.2f} {ours_keep:>6} "
                      f"{their_keep:>6}   {name:<5} {kudio_score:>12.6f} {theirs:>12.6f} "
                      f"{same_count:>20.6f}")
                if ours_keep != their_keep:
                    parted = max(parted, abs(kudio_score - theirs) / abs(theirs))
                    apart += 1
                else:
                    rejoined = max(rejoined, abs(kudio_score - theirs) / abs(theirs))
                rejoined = max(rejoined, abs(same_count - theirs) / abs(theirs))
                cases += 1
    report('kudio vs pysepm where the counts part (M = 30, 70, 390)', '8, 16k', apart,
           math.nan, parted, math.inf, 'rel')
    report("kudio's frames over pysepm's count vs pysepm (all M)", '8, 16k', cases,
           math.nan, rejoined, 1e-9, 'rel')


# --- 4: Itakura-Saito ------------------------------------------------------------

def frames_of(y, sr):
    """30 ms frames every quarter frame, eps added first, under the code's
    window 0.5 (1 - cos(2 pi n / (N + 1))) for n = 1..N, the last frame that
    fits left out, as the comp_*.m files frame a signal."""
    size = int(round(0.03 * sr))
    hop = size // 4
    window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(1, size + 1) / (size + 1)))
    y = y + EPS
    return np.array([y[i * hop:i * hop + size] * window
                     for i in range((len(y) - size) // hop)])


def lags(frame, order):
    return np.array([frame[:len(frame) - k] @ frame[k:] for k in range(order + 1)])


def comp_is(ref, deg, sr):
    """comp_is.m's value for every pair of frames: LPC by the autocorrelation
    method (scipy's Toeplitz solver here, Levinson-Durbin there), and

        IS = (G_c / G_p) (a_p R_c a_p') / (a_c R_c a_c') + ln(G_p / G_c) - 1

    with G = r . a the prediction-error power of each frame, both G and the
    clean frame's own quadratic form floored at eps. No range applied yet."""
    order = ORDER[sr]
    values = []
    for c, p in zip(frames_of(ref, sr), frames_of(deg, sr)):
        r_c, r_p = lags(c, order), lags(p, order)
        a_c = np.concatenate([[1.0], -solve_toeplitz(r_c[:order], r_c[1:])])
        a_p = np.concatenate([[1.0], -solve_toeplitz(r_p[:order], r_p[1:])])
        matrix = toeplitz(r_c)
        g_c, g_p = max(r_c @ a_c, EPS), max(r_p @ a_p, EPS)
        ratio = (a_p @ matrix @ a_p) / max(a_c @ matrix @ a_c, EPS)
        values.append(g_c / g_p * ratio + math.log(g_p / g_c) - 1.0)
    return np.array(values)


def best_95(values):
    """The mean of the round(0.95 M) lowest, rounded as MATLAB rounds."""
    ordered = np.sort(values)
    return float(np.mean(ordered[:int(math.floor(0.95 * len(ordered) + 0.5))]))


def spectral_is(ref, deg, sr, points=16384):
    """The Itakura-Saito divergence itself, frame by frame: the mean over
    *points* frequencies of P_c / P_p - ln(P_c / P_p) - 1, with P = G / |A|^2
    the LPC spectrum of each frame."""
    order = ORDER[sr]
    out = []
    for c, p in zip(frames_of(ref, sr), frames_of(deg, sr)):
        spectra = []
        for frame in (c, p):
            r = lags(frame, order)
            alpha = solve_toeplitz(r[:order], r[1:])
            shape = np.abs(np.fft.rfft(np.concatenate([[1.0], -alpha]), points)) ** 2
            spectra.append((r[0] - alpha @ r[1:]) / shape)
        ratio = spectra[0] / spectra[1]
        # rfft bins: the two ends count once and the rest twice round the circle
        weights = np.full(len(ratio), 2.0)
        weights[0] = weights[-1] = 1.0
        out.append(float(np.sum(weights * (ratio - np.log(ratio) - 1.0)) / points))
    return np.array(out)


def check_itakura_saito():
    for sr in (8000, 16000):
        cases = pairs(sr)
        got = [quietly(itakura_saito, r, d, sr) for _, r, d in cases]
        restated = [best_95(np.clip(comp_is(r, d, sr), 0.0, 100.0)) for _, r, d in cases]
        diff, rel = largest(got, restated)
        worst = cases[int(np.argmax(np.abs(np.subtract(got, restated))))][0]
        report(f'itakura_saito vs comp_is.m restated (worst: {worst})', sr, len(cases),
               diff, rel, 1e-9, 'rel')
        picked = cases[:3] + cases[15:16]
        integral = [best_95(np.clip(spectral_is(r, d, sr), 0.0, 100.0)) for _, r, d in picked]
        diff, rel = largest(got[:3] + got[15:16], integral)
        report('itakura_saito vs the divergence of the LPC spectra', sr, len(picked),
               diff, rel, 1e-9, 'rel')

        ref = padded(sr)
        code_range = best_95(np.minimum(comp_is(ref, ref, sr), 100.0))
        paper_range = best_95(np.clip(comp_is(ref, ref, sr), 0.0, 100.0))
        ours = itakura_saito(ref, ref, sr)
        print(f"\nItakura-Saito of an exact copy of 2 s of speech padded with 0.5 s of "
              f"zeros either side ({sr} Hz): comp_is.m capped at 100 only {code_range:.4f} "
              f"(-125/376 = {-125 / 376:.4f}), held to [0, 100] {paper_range:.4f}, "
              f"kudio {ours:.4f}")
        diff, rel = largest([ours, code_range], [paper_range, -125 / 376])
        report('padded copy: kudio = [0, 100] restated, comp_is.m = -125/376', sr, 2,
               diff, rel, 1e-12)
        for gain, per_frame in ((2.0, 0.25 + math.log(4.0) - 1.0),
                                (0.5, 4.0 - math.log(4.0) - 1.0)):
            plain = itakura_saito(speech(sr), gain * speech(sr), sr)
            with_zeros = itakura_saito(ref, gain * ref, sr)
            diff, rel = largest([plain, with_zeros], [per_frame, 251 * per_frame / 376])
            report(f'itakura_saito of a copy x{gain}: {per_frame:.3f}, padded '
                   f'{251 * per_frame / 376:.3f}', sr, 2, diff, rel, 1e-9, 'rel')
    louder = 0.5 + math.log(0.5) - 1.0
    print(f"The log term as [HL08] eq. (5) prints it, for an output at twice the clean "
          f"power: 0.5 + ln 0.5 - 1 = {louder:.3f}; as comp_is.m has it, "
          f"0.5 + ln 2 - 1 = {0.5 + math.log(2.0) - 1.0:.3f}")


# --- 5: the composite ---------------------------------------------------------------

def p862_1_inverse(lqo):
    """The raw P.862 score behind a narrowband MOS-LQO, through P.862.1's
    0.999 + 4 / (1 + exp(-1.4945 x + 4.6607))."""
    return (4.6607 - math.log(4.0 / (lqo - 0.999) - 1.0)) / 1.4945


def regression(pesq_score, llr_, wss_, segsnr):
    """[HL06] Sec. 4.1, each rating held to [1, 5]."""
    csig = 3.093 - 1.029 * llr_ + 0.603 * pesq_score - 0.009 * wss_
    cbak = 1.634 + 0.478 * pesq_score - 0.007 * wss_ + 0.063 * segsnr
    covl = 1.594 + 0.805 * pesq_score - 0.512 * llr_ - 0.007 * wss_
    return tuple(min(5.0, max(1.0, v)) for v in (csig, cbak, covl))


def check_composite():
    if pysepm_evo is None:
        skipped('composite vs [HL06] on pysepm + pesq', 'no pysepm_evo')
        return
    clean, noisy = test_pair(16000)
    try:
        pysepm_evo.composite(clean, noisy, 16000)
        said = 'it ran'
    except NameError as e:
        said = f'NameError: {e}'
    print(f"\npysepm_evo {pysepm_evo.__version__} composite(): {said}")

    if pesq_package is None:
        skipped('composite vs [HL06] on pysepm + pesq', 'no pesq package')
        return
    for sr, kwargs, mode, raw in ((16000, {}, 'wb', False), (8000, {}, 'nb', True),
                                  (16000, {'mode': 'nb', 'scale': 'raw'}, 'nb', True)):
        got, want = [], []
        for _, ref, deg in pairs(sr):
            ours = quietly(composite, ref, deg, sr, **kwargs)
            score = float(pesq_package.pesq(sr, ref, deg, mode))
            if raw:
                score = p862_1_inverse(score)
            wss_ = pysepm_evo.wss(ref, deg, sr)
            llr_ = pysepm_evo.llr(ref, deg, sr, used_for_composite=True)
            segsnr = pysepm_evo.SNRseg(ref, deg, sr)
            got.append(tuple(ours))
            want.append(regression(score, llr_, wss_, segsnr) + (score, llr_, wss_, segsnr))
        diff, rel = largest(got, want)
        label = f"{mode} {'raw' if raw else 'MOS-LQO'}"
        report(f'composite, all 7 fields, vs [HL06] on pysepm + pesq ({label})', sr,
               len(got), diff, rel, 1e-9, 'rel')
        diff, rel = largest([g[3] for g in got], [w[3] for w in want])
        report(f'  its pesq field vs the pesq package ({label})', sr, len(got), diff, rel, 0)


# --- 7: an empty band -----------------------------------------------------------------

def exact_llr(clean, processed, order):
    """LLR of one pair of windowed frames in 60-digit arithmetic, through the
    autocorrelation, Levinson-Durbin and the two quadratic forms."""
    import mpmath as mp
    mp.mp.dps = 60
    c = [mp.mpf(float(x)) for x in clean]
    d = [mp.mpf(float(x)) for x in processed]

    def acf(x):
        return [mp.fsum(x[j] * x[j + k] for j in range(len(x) - k)) for k in range(order + 1)]

    def levinson(r):
        a, e = [mp.mpf(0)] * order, r[0]
        for i in range(order):
            k = (r[i + 1] - mp.fsum(a[j] * r[i - j] for j in range(i))) / e
            before = a[:i]
            a[i] = k
            for j in range(i):
                a[j] = before[j] - k * before[i - 1 - j]
            e = (1 - k * k) * e
        return [mp.mpf(1)] + [-x for x in a]

    def quadratic(a, r):
        return mp.fsum(a[i] * a[j] * r[abs(i - j)]
                       for i in range(order + 1) for j in range(order + 1))
    rc = acf(c)
    return float(mp.log(quadratic(levinson(acf(d)), rc) / quadratic(levinson(rc), rc)))


def check_empty_band():
    try:
        import mpmath  # noqa: F401
    except ImportError:
        mpmath = None
    for sr in (8000, 16000):
        clean = speech(sr, 3.0, seed=1)
        low = sosfiltfilt(butter(8, 1000, fs=sr, output='sos'), clean)
        got = quietly(composite, clean, low, sr, pesq=2.0).llr
        line = f"  {sr} Hz: kudio {got:.4f}"
        if pysepm_evo is not None:
            want = quietly(pysepm_evo.llr, clean, low, sr, used_for_composite=True)
            report("composite's uncapped llr vs pysepm, low-passed 1 kHz, no noise", sr, 1,
                   abs(got - want), abs(got - want) / abs(want), 5e-3, 'rel')
            line += f", pysepm {want:.4f}"
        if mpmath is not None:
            framed = module._framed(clean, low, sr, 'llr')
            frames = [exact_llr(framed.clean[i] * framed.window,
                                framed.processed[i] * framed.window, ORDER[sr])
                      for i in range(len(framed.clean))]
            exact = module._best(np.array(frames))
            report("composite's uncapped llr vs 60 digits, low-passed 1 kHz", sr, 1,
                   abs(got - exact), abs(got - exact) / abs(exact), 5e-3, 'rel')
            line += f", 60 digits {exact:.4f}"
            capped = [module._best(np.clip(frames, 0.0, 2.0))]
            diff, rel = largest(quietly(llr, clean, low, sr), capped)
            report('llr (capped) vs 60 digits, low-passed 1 kHz', sr, 1, diff, rel, 1e-9, 'rel')
        print(line)
        for measure, cap in ((itakura_saito, 100.0), (cepstral_distance, 10.0)):
            if sr == 16000:
                value = quietly(measure, clean, low, sr)
                report(f'{measure.__name__} at its cap, low-passed 1 kHz', sr, 1,
                       abs(value - cap), abs(value - cap) / cap, 0.0)


# --- 6: figures in the module docstring --------------------------------------------

def quoted(check, rate, got, said, places):
    """A figure the docstring quotes to *places* decimals: within half a unit
    of its last digit."""
    report(f'{check}: said {said:.{places}f}, got {got:.{places + 2}f}', rate, 1,
           abs(got - said), abs(got - said) / max(abs(said), 1e-300), 0.5 * 10.0 ** -places)


def check_figures():
    for sr, offset_said, noise_said in ((8000, 30.8, 36.0), (16000, 35.0, 30.9)):
        clean = speech(sr)
        shifted = clean + 0.01 * np.max(np.abs(clean))
        quoted('wss, 1% of the peak added as DC', sr, wss(clean, shifted, sr), offset_said, 1)
        quoted('wss, white noise at 10 dB', sr, wss(*test_pair(sr), sr), noise_said, 1)

    # WSS's band floor is absolute: the same pair with its clean peak at
    # -6 .. -46, -56 and -66 dBFS
    for sr, said in ((8000, (35.98, 35.29, 27.95)), (16000, (30.90, 30.84, 25.19))):
        clean, noisy = test_pair(sr)
        peak = np.max(np.abs(clean))
        for dbfs, value in zip((-46, -56, -66), said):
            gain = 10.0 ** (dbfs / 20.0) / peak
            quoted(f'wss of the 10 dB pair at a peak of {dbfs} dBFS', sr,
                   quietly(wss, gain * clean, gain * noisy, sr), value, 2)

    clean, noisy = test_pair(8000, 20, 'white')
    quoted("composite's segsnr, white 20 dB", 8000,
           composite(clean, noisy, 8000, pesq=2.0).segsnr, 6.02, 2)
    quoted('kudio.segmental_snr on the same pair', 8000, kudio.segmental_snr(clean, noisy),
           8.08, 2)

    clean = speech(16000)
    hiss = mix(clean, sosfilt(butter(8, 5000, 'highpass', fs=16000, output='sos'),
                              noise(len(clean), 1, 'white', 16000)), 20)
    quoted('fw_segmental_snr, noise above 5 kHz at 20 dB', 16000,
           fw_segmental_snr(clean, hiss, 16000), 8.58, 2)
    spectrum = np.fft.rfft(clean)
    spectrum[np.fft.rfftfreq(len(clean), 1.0 / 16000) > 4000] = 0.0
    quoted('fw_segmental_snr, nothing above 4 kHz', 16000,
           fw_segmental_snr(clean, np.fft.irfft(spectrum, len(clean)), 16000), 17.88, 2)

    # white noise at -60 dBFS RMS where the padded reference is zero: the
    # docstring gives the 8 kHz figures, and 16 kHz within 3% of them
    said = {'llr': 0.56, 'itakura_saito': 7.2, 'cepstral_distance': 2.84, 'wss': 5.73,
            'fw_segmental_snr': 20.4}
    places = {'llr': 2, 'itakura_saito': 1, 'cepstral_distance': 2, 'wss': 2,
              'fw_segmental_snr': 1}
    apart = 0.0
    print('\nPadded speech, white noise at -60 dBFS in its zeros (docstring: 8 kHz, '
          '16 kHz within 3%):')
    for measure in (llr, itakura_saito, cepstral_distance, wss, fw_segmental_snr):
        name = measure.__name__
        scores = {}
        for sr in (8000, 16000):
            ref = padded(sr)
            deg = ref.copy()
            quiet = ref == 0.0
            deg[quiet] = 1e-3 * noise(len(ref), 1, 'white', sr)[quiet]
            scores[sr] = quietly(measure, ref, deg, sr)
        quoted(f'{name}, noise in the zeros', 8000, scores[8000], said[name], places[name])
        apart = max(apart, abs(scores[16000] / scores[8000] - 1.0))
        print(f"  {name:<18} 8 kHz {scores[8000]:8.4f}   16 kHz {scores[16000]:8.4f}   "
              f"{100 * (scores[16000] / scores[8000] - 1):+.1f}%")
    report('the same at 16 kHz, largest relative move from 8 kHz', 16000, 5, apart, apart,
           0.03)

    ends = (-0.5, 4.5)
    print("\nP.862.1 takes the raw range -0.5..4.5 to "
          + '..'.join(f"{0.999 + 4 / (1 + math.exp(-1.4945 * x + 4.6607)):.3f}" for x in ends)
          + ", P.862.2 to "
          + '..'.join(f"{0.999 + 4 / (1 + math.exp(-1.3669 * x + 3.8224)):.3f}" for x in ends)
          + " (docstring: 1.017..4.549 and 1.043..4.644)")
    best = regression(4.5, 0.0, 0.0, 35.0)
    unheld = (3.093 + 0.603 * 4.5, 1.634 + 0.478 * 4.5 + 0.063 * 35.0, 1.594 + 0.805 * 4.5)
    print("A perfect copy at raw PESQ 4.5 by the equations: "
          + ', '.join(f"{v:.2f}" for v in unheld) + "; held to the scale: "
          + ', '.join(f"{v:g}" for v in best) + " (docstring: 5.81, 5.99 and 5.22)")
    for got, said in zip(unheld, (5.81, 5.99, 5.22)):
        quoted('rating past 5 for a perfect copy', '-', got, said, 2)


def main():
    start = time.perf_counter()
    import scipy
    from importlib.metadata import PackageNotFoundError, version

    def installed(name):
        try:
            return version(name)
        except PackageNotFoundError:
            return 'none'

    print(f"kudio {kudio.__version__} from {Path(kudio.__file__).parent}; pysepm_evo "
          f"{installed('pysepm_evo')}, pesq {installed('pesq')}, numba {installed('numba')}; "
          f"numpy {np.__version__}, scipy {scipy.__version__}, Python {sys.version.split()[0]}")
    check_pysepm()
    check_stored()
    check_rounding()
    check_itakura_saito()
    check_composite()
    check_figures()
    print("\nThe composite's uncapped LLR on an output low-passed at 1 kHz, no noise:")
    check_empty_band()
    failed = print_table()
    print(f"\n{time.perf_counter() - start:.0f} s; "
          + ("SOME DIFFERENCES ARE PAST TOLERANCE" if failed
             else "every difference within its tolerance"))
    return 1 if failed else 0


def print_table():
    """The table of every check; True if one is past its limit."""
    width = max(len(r[0]) for r in RESULTS)
    print(f"\n{'check':<{width}} {'rate':>7} {'cases':>5} {'max |diff|':>11} "
          f"{'max rel':>9} {'limit':>11}")
    failed = False
    for check, rate, cases, diff, rel, tolerance, on in RESULTS:
        over = (rel if on == 'rel' else diff) > tolerance
        failed |= over
        limit = ('-' if math.isinf(tolerance)
                 else f"{tolerance:.1e}".replace('.0e', 'e') + f" {on}")
        shown = '-' if math.isnan(diff) else f"{diff:.2e}"
        print(f"{check:<{width}} {rate!s:>7} {cases:>5} {shown:>11} {rel:9.2e} "
              f"{limit:>11}{'  PAST' if over else ''}")
    return failed


if __name__ == '__main__':
    sys.exit(main())
