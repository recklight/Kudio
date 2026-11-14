# -*- coding: utf-8 -*-
"""Check kudio's log-spectral distance (lsd in kudio/core/evaluator.py)
against the two definitions it is set to reproduce, URGENT 2025's and
NVSR's ssr_eval, each restated here from the code its authors published and
run on librosa's STFT as theirs is.

Run it from anywhere; it imports the kudio of the checkout it sits in::

    python tools/validation/lsd.py

Besides what kudio itself needs, it needs librosa, which both scoring
scripts compute their STFT with; without it the rows that need it say they
were skipped, and kudio is still held to the numbers the tests store and
to the arithmetic. I ran it with librosa 0.11.0 (numpy 2.2.6,
scipy 1.15.2, Python 3.10.22), the versions the numbers stored in
tests/test_lsd.py were made with, and again with librosa 1.0.0 on Python
3.13 and with librosa 0.11.0 on Python 3.9.

The two definitions, as I read them on GitHub on 2026-11-13:

* URGENT 2025: lsd_metric() in evaluation_metrics/
  calculate_intrusive_se_metrics.py of github.com/urgent-challenge/
  urgent2025_challenge. The file was last changed in commit 93e16aa
  (2024-11-19) and is the same at the head of the repository, daf1730
  (2025-05-21). The output is first scaled by its least-squares gain
  against the reference, sum(ref * inf) / (sum(inf ** 2) + eps); then
  librosa.stft with n_fft = int(fs * 0.032) and hop_length =
  int(fs * 0.016), its other arguments left at their defaults; then the
  natural log of ref_spec ** 2 / (inf_spec + eps) ** 2 + eps, squared,
  averaged over frequency, square-rooted and averaged over frames, eps
  1e-8. process_one_pair() in the same file reads both files with
  soundfile as float32, so the challenge's own numbers come out of a
  single-precision STFT.
* NVSR (Liu et al., 2022): AudioMetrics in ssr_eval/metrics.py of
  github.com/haoheliu/ssr_eval, last changed in commit 907395c
  (2022-03-28) and the same at the head, 7129ee8 (2022-06-17). At a rate R
  it takes hop_length = int(R / 100) and n_fft = int(2048 / (44100 / R)),
  2048 and 441 at 44.1 kHz; librosa.stft with the rest at its defaults;
  no gain on the estimate; and in torch, log10(target ** 2 / (est + EPS)
  ** 2 + EPS) squared, averaged over frequency, square-rooted and averaged
  over frames, EPS 1e-12. A file is read with librosa.load, as float32;
  an array is taken in whatever precision it comes.

Every signal is made here from fixed seeds: the harmonic syllables of
tests/test_lsd.py over a faint noise floor, under white noise from -5 to
20 dB SNR, low-passed at a quarter of the rate and at 4 kHz, at half the
level under noise, 1 ms late, in a 0.4 s room from kudio.rir, and with half
a second dropped out to zeros.

What it checks:

1. lsd() with its defaults against URGENT's lsd_metric restated, on
   float64 signals, at 8, 11.025, 16, 22.05, 24, 32, 44.1 and 48 kHz, and
   at 16 kHz on pairs one to two frames long, where librosa's centred
   padding decides how many frames there are.
2. lsd(scale='log10', match_gain=False, eps=1e-12) with ssr_eval's n_fft
   and hop at 16, 22.05, 44.1 and 48 kHz against ssr_eval's formula
   restated.
3. The numbers tests/test_lsd.py stores for both, computed again from
   scratch with the restated definitions, and kudio against them.
4. What single precision does: URGENT's script and ssr_eval on float32
   copies, as each reads a file, against kudio in double. On full-band
   output they agree to about 1e-6; where the output has an empty band the
   two part, because that band sits on float32's rounding floor, and a
   16-bit rounding of both signals puts the rounding noise there instead.
5. Arithmetic the module docstring relies on: URGENT's int(fs * 0.016) is
   kudio's int(fs * 0.032) // 2 at every whole rate from 50 Hz to 384 kHz;
   the three scales are one distance; a frame where the reference is
   digital silence scores -ln(eps); and the dropout figures.

It prints the largest differences and exits with 1 if one is past its
tolerance. The run takes a few seconds.

References:

* K. Saijo, W. Zhang, S. Cornell, R. Scheibler, C. Li, Z. Ni, A. Kumar,
  M. Sach, Y. Fu, W. Wang, T. Fingscheidt and S. Watanabe, "Interspeech
  2025 URGENT speech enhancement challenge", Proc. Interspeech 2025,
  pp. 858-862, doi:10.21437/Interspeech.2025-1363.
* H. Liu, W. Choi, X. Liu, Q. Kong, Q. Tian and D. Wang, "Neural vocoder is
  all you need for speech super-resolution", Proc. Interspeech 2022,
  pp. 4227-4231, doi:10.21437/Interspeech.2022-11017. The distance is
  their Eq. 7, scored with ssr_eval.
"""
from __future__ import annotations

import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    import librosa
except ImportError:
    librosa = None

import kudio  # noqa: E402
from kudio.core import evaluator  # noqa: E402
from kudio.core.evaluator import lsd  # noqa: E402

RATES = (8000, 11025, 16000, 22050, 24000, 32000, 44100, 48000)

#: (check, rate, cases, largest absolute difference, largest relative
#: difference, tolerance, and whether it holds the absolute or the relative one)
RESULTS = []


def report(check, rate, cases, diff, rel, tolerance, on='abs'):
    RESULTS.append((check, rate, cases, diff, rel, tolerance, on))


def skipped(what):
    report(f'{what}: skipped, no librosa', '-', 0, 0.0, 0.0, math.inf)


def largest(got, want):
    """Largest absolute and relative difference over paired values."""
    got, want = np.ravel(got).astype(float), np.ravel(want).astype(float)
    diff = np.abs(got - want)
    return float(np.max(diff)), float(np.max(diff / np.maximum(np.abs(want), 1e-300)))


def quietly(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return fn(*args, **kwargs)


# --- signals, the same as in tests/test_lsd.py ----------------------------------

def talk(sr, seconds=3.0, seed=0):
    """Harmonic syllables, two a second, on an f0 wandering between 90 and
    150 Hz, over a faint noise floor that keeps every frame from being
    digital silence."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(sr * seconds)) / sr
    f0 = 120.0 + 30.0 * np.sin(2 * np.pi * 0.7 * t)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    voiced = sum(np.sin(k * phase) / k for k in range(1, 40) if k * 150 < sr / 2)
    envelope = np.clip(np.sin(2 * np.pi * 2.0 * t), 0.0, None) ** 2
    return 0.3 * voiced * envelope + 0.003 * rng.standard_normal(len(t))


def noisy(ref, snr_db, seed):
    hiss = np.random.default_rng(seed).standard_normal(len(ref))
    return ref + hiss * math.sqrt(np.mean(ref ** 2) / 10 ** (snr_db / 10))


def lowpassed(ref, sr, cutoff):
    return sosfiltfilt(butter(8, cutoff, fs=sr, output='sos'), ref)


def pairs(sr):
    """``(name, ref, deg)`` at *sr*."""
    ref = talk(sr, seed=sr)
    out = [(f'white {snr:+d} dB', ref, noisy(ref, snr, sr + 1)) for snr in (-5, 0, 10, 20)]
    out.append(('low-passed at sr / 4', ref, lowpassed(ref, sr, sr / 4)))
    if sr > 8000:
        out.append(('low-passed at 4 kHz', ref, lowpassed(ref, sr, 4000)))
    out.append(('half level, white 20 dB', ref, 0.5 * noisy(ref, 20, sr + 2)))
    late = int(round(0.001 * sr))
    out.append(('1 ms late', ref, np.concatenate([np.zeros(late), ref])[:len(ref)]))
    room = kudio.apply_rir(ref, kudio.rir(sr, rt60=0.4, seed=5))
    out.append(('room 0.4 s', ref, np.asarray(room, float)[:len(ref)]))
    dropout = ref.copy()
    dropout[sr:sr + sr // 2] = 0.0
    out.append(('0.5 s dropped out', ref, dropout))
    return out


# --- the two definitions, restated from their code ----------------------------------

def magnitudes(y, n_fft, hop):
    """|STFT|, frames by bins, from librosa.stft with its defaults otherwise:
    a periodic Hann window and n_fft // 2 zeros padded at each end."""
    return np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop)).T


def urgent_lsd(ref, inf, fs, eps=1e-8):
    """URGENT 2025's lsd_metric, in the precision of its inputs."""
    gain = np.sum(ref * inf) / (np.sum(inf ** 2) + eps)
    inf = inf * gain
    n_fft, hop = int(fs * 0.032), int(fs * 0.016)
    r, d = magnitudes(ref, n_fft, hop), magnitudes(inf, n_fft, hop)
    per_bin = np.log(r ** 2 / (d + eps) ** 2 + eps) ** 2
    return float(np.mean(np.mean(per_bin, axis=1) ** 0.5))


def ssr_eval_settings(rate):
    """AudioMetrics' n_fft and hop_length at *rate*."""
    return int(2048 / (44100 / rate)), int(rate / 100)


def ssr_eval_lsd(est, target, rate, eps=1e-12):
    """ssr_eval's AudioMetrics.lsd on AudioMetrics' spectrogram, in numpy, in
    the precision of its inputs."""
    n_fft, hop = ssr_eval_settings(rate)
    t, e = magnitudes(target, n_fft, hop), magnitudes(est, n_fft, hop)
    per_bin = np.log10(t ** 2 / (e + eps) ** 2 + eps) ** 2
    return float(np.mean(np.mean(per_bin, axis=1) ** 0.5))


def kudio_ssr_eval(ref, deg, rate):
    n_fft, hop = ssr_eval_settings(rate)
    return lsd(ref, deg, rate, n_fft=n_fft, hop_length=hop, scale='log10',
               match_gain=False, eps=1e-12)


# --- 1 and 2: against the definitions ----------------------------------------------

def check_urgent():
    if librosa is None:
        skipped('defaults vs URGENT restated')
        return
    for sr in RATES:
        cases = pairs(sr)
        got = [quietly(lsd, r, d, sr) for _, r, d in cases]
        want = [urgent_lsd(r, d, sr) for _, r, d in cases]
        diff, rel = largest(got, want)
        worst = cases[int(np.argmax(np.abs(np.subtract(got, want)) / np.abs(want)))][0]
        report(f'defaults vs URGENT restated (worst: {worst})', sr, len(cases), diff, rel,
               1e-9, 'rel')
    ref = talk(16000, seed=3)
    deg = noisy(ref, 10.0, 4)
    lengths = (512, 513, 600, 767, 768, 769, 1023, 1024, 1025, 4800)
    got = [quietly(lsd, ref[:n], deg[:n], 16000) for n in lengths]
    want = [urgent_lsd(ref[:n], deg[:n], 16000) for n in lengths]
    diff, rel = largest(got, want)
    report('defaults vs URGENT, pairs of 512 to 4800 samples', 16000, len(lengths),
           diff, rel, 1e-9, 'rel')


def check_ssr_eval():
    if librosa is None:
        skipped('ssr_eval setting vs ssr_eval restated')
        return
    for sr in (16000, 22050, 44100, 48000):
        cases = pairs(sr)
        got = [quietly(kudio_ssr_eval, r, d, sr) for _, r, d in cases]
        want = [ssr_eval_lsd(d, r, sr) for _, r, d in cases]
        diff, rel = largest(got, want)
        n_fft, hop = ssr_eval_settings(sr)
        worst = cases[int(np.argmax(np.abs(np.subtract(got, want)) / np.abs(want)))][0]
        report(f'ssr_eval setting (n_fft {n_fft}, hop {hop}) vs ssr_eval restated '
               f'(worst: {worst})', sr, len(cases), diff, rel, 1e-9, 'rel')


# --- 3: the numbers the tests store ---------------------------------------------

#: _URGENT_2025 of tests/test_lsd.py: URGENT's lsd_metric on float64 copies,
#: with librosa 0.11.0, numpy 2.2.6 and scipy 1.15.2
STORED_URGENT = [
    (8000, 'noisy', 4.12792757488937),
    (16000, 'noisy', 4.472351638883378),
    (22050, 'noisy', 4.654353236598868),
    (44100, 'noisy', 4.919388486238227),
    (48000, 'noisy', 4.9373865419269425),
    (16000, 'lowpassed', 14.624145480159434),
    (48000, 'lowpassed', 16.486048293188183),
]

#: _SSR_EVAL_44K of tests/test_lsd.py: ssr_eval's formula at 44.1 kHz on the
#: same STFT
STORED_SSR_EVAL = [('noisy', 2.215475495172847), ('lowpassed', 11.339808821938234)]


def stored_urgent_pair(sr, kind):
    ref = talk(sr, seed=sr)
    deg = noisy(ref, 10.0, sr + 1) if kind == 'noisy' else lowpassed(ref, sr, sr / 4)
    return ref, deg


def stored_ssr_eval_pair(kind):
    ref = talk(44100, seed=7)
    deg = noisy(ref, 10.0, 8) if kind == 'noisy' else lowpassed(ref, 44100, 4000)
    return ref, deg


def check_stored():
    again, ours, stored = [], [], []
    for sr, kind, value in STORED_URGENT:
        ref, deg = stored_urgent_pair(sr, kind)
        if librosa is not None:
            again.append(urgent_lsd(ref, deg, sr))
        ours.append(quietly(lsd, ref, deg, sr))
        stored.append(value)
    if librosa is None:
        skipped('URGENT restated now vs the numbers the tests store')
    else:
        diff, rel = largest(again, stored)
        report('URGENT restated now vs the numbers the tests store', 'all', len(stored),
               diff, rel, 1e-9, 'rel')
    diff, rel = largest(ours, stored)
    report('kudio vs those numbers (tests: rel 1e-7)', 'all', len(stored), diff, rel,
           1e-7, 'rel')
    again, ours, stored = [], [], []
    for kind, value in STORED_SSR_EVAL:
        ref, deg = stored_ssr_eval_pair(kind)
        if librosa is not None:
            again.append(ssr_eval_lsd(deg, ref, 44100))
        ours.append(quietly(kudio_ssr_eval, ref, deg, 44100))
        stored.append(value)
    if librosa is None:
        skipped('ssr_eval restated now vs the numbers the tests store')
    else:
        diff, rel = largest(again, stored)
        report('ssr_eval restated now vs the numbers the tests store', 44100, 2, diff, rel,
               1e-9, 'rel')
    diff, rel = largest(ours, stored)
    report('kudio vs those numbers (tests: rel 1e-7)', 44100, 2, diff, rel, 1e-7, 'rel')


# --- 4: single precision ------------------------------------------------------------

def to_16_bits(y, peak):
    return np.round(y / peak * 32767.0) / 32768.0


def check_single_precision():
    if librosa is None:
        skipped('single precision, URGENT and ssr_eval')
        return
    print('\nIn float32, as the challenge reads its files (URGENT) and as librosa.load '
          'reads them (ssr_eval):')
    full = []
    for sr in RATES:
        for name, ref, deg in pairs(sr):
            if name.startswith('white') or name.startswith('half'):
                single = urgent_lsd(ref.astype(np.float32), deg.astype(np.float32), sr)
                full.append((abs(single - quietly(lsd, ref, deg, sr)), sr, name))
    worst = max(full)
    print(f"  URGENT, full-band outputs at every rate: largest difference {worst[0]:.1e} "
          f"({worst[2]}, {worst[1]} Hz)")
    report('URGENT in float32 vs kudio, full-band outputs (said: about 1e-6)', 'all',
           len(full), worst[0], math.nan, 5e-6)

    ref, deg = stored_urgent_pair(48000, 'lowpassed')
    double, single = quietly(lsd, ref, deg, 48000), urgent_lsd(
        ref.astype(np.float32), deg.astype(np.float32), 48000)
    print(f"  URGENT, 48 kHz low-passed at 12 kHz: float32 {single:.4f}, kudio {double:.4f}, "
          f"{single - double:+.4f} (said: 0.38 apart)")
    quoted('URGENT float32 vs kudio, 48 kHz low-passed at 12 kHz', 48000,
           abs(single - double), 0.38, 2)

    ref, deg = stored_ssr_eval_pair('lowpassed')
    double = quietly(kudio_ssr_eval, ref, deg, 44100)
    single = ssr_eval_lsd(deg.astype(np.float32), ref.astype(np.float32), 44100)
    print(f"  ssr_eval, 44.1 kHz low-passed at 4 kHz: float32 {single:.4f}, kudio "
          f"{double:.4f}, {single - double:+.4f} (said: 0.42 apart)")
    quoted('ssr_eval float32 vs kudio, 44.1 kHz low-passed at 4 kHz', 44100,
           abs(single - double), 0.42, 2)

    peak = max(np.max(np.abs(ref)), np.max(np.abs(deg)))
    ref16, deg16 = to_16_bits(ref, peak), to_16_bits(deg, peak)
    double = quietly(kudio_ssr_eval, ref16, deg16, 44100)
    single = ssr_eval_lsd(deg16.astype(np.float32), ref16.astype(np.float32), 44100)
    print(f"  the same pair rounded to 16 bits: float32 {single:.8f}, kudio {double:.8f}, "
          f"{abs(single - double):.1e} apart (said: 4e-8)")
    report('ssr_eval float32 vs kudio, that pair in 16 bits (said: 4e-8)', 44100, 1,
           abs(single - double), abs(single - double) / double, 5e-8)


# --- 5: arithmetic and quoted figures ---------------------------------------------

def quoted(check, rate, got, said, places):
    """A figure quoted to *places* decimals: within half a unit of its last digit."""
    report(f'{check}: said {said:.{places}f}, got {got:.{places + 2}f}', rate, 1,
           abs(got - said), abs(got - said) / max(abs(said), 1e-300), 0.5 * 10.0 ** -places)


def check_arithmetic():
    rates = np.arange(50, 384001)
    frame = evaluator._LSD_FRAME
    ours = np.array([int(int(r * frame) // 2) for r in rates.tolist()])
    theirs = np.array([int(r * 0.016) for r in rates.tolist()])
    report("kudio's default hop int(0.032 sr) // 2 vs URGENT's int(0.016 sr)", '50-384k',
           len(rates), float(np.max(np.abs(ours - theirs))), math.nan, 0)

    ref = talk(16000, seed=11)
    deg = noisy(ref, 5.0, 12)
    ln, base10, db = (lsd(ref, deg, 16000, scale=s) for s in ('ln', 'log10', 'db'))
    diff, rel = largest([ln / base10, db / ln], [math.log(10.0), 10.0 / math.log(10.0)])
    report('the three scales: ln / log10 = ln 10, db / ln = 10 / ln 10', 16000, 2,
           diff, rel, 1e-12, 'rel')

    speech = talk(16000, seed=21)
    x = np.concatenate([np.zeros(8000), speech[:40000]])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        got = lsd(x, x, 16000)
    said = str(caught[0].message) if caught else ''
    silent = int(said.split(' of ')[0].split()[-1]) if ' of ref' in said else -1
    frames = 1 + len(x) // 256
    want = silent * -math.log(1e-8) / frames
    print(f"\nHalf a second of zeros before 2.5 s of speech: {silent} of {frames} frames "
          f"silent, each scoring -ln(1e-8) = {-math.log(1e-8):.2f}; the copy scores "
          f"{got:.4f}, {silent} x 18.42 / {frames} = {want:.4f}")
    report("frames where ref is digital silence score -ln(eps)", 16000, 1,
           abs(got - want), abs(got - want) / want, 1e-4)
    # centred frames of 512 every 256: frame k spans k * 256 - 256 to
    # k * 256 + 256, all of it inside the 8000 zeros up to k = 30
    report('  how many there are, against what librosa frames', 16000, 1,
           float(abs(silent - ((8000 - 256) // 256 + 1))), math.nan, 0)

    ref = talk(16000, seed=15)
    dropout = ref.copy()
    dropout[16000:24000] = 0.0
    quoted('0.5 s dropout in 3 s, eps 1e-8', 16000, quietly(lsd, ref, dropout, 16000), 4.90, 2)
    quoted('the same, eps 1e-12', 16000, quietly(lsd, ref, dropout, 16000, eps=1e-12), 7.74, 2)
    hiss = noisy(ref, 10.0, 16)
    moved = lsd(ref, hiss, 16000, eps=1e-12) - lsd(ref, hiss, 16000)
    quoted('noise in every bin, eps 1e-8 to 1e-12', 16000, moved, 0.001, 3)


def main():
    start = time.perf_counter()
    import scipy
    print(f"kudio {kudio.__version__} from {Path(kudio.__file__).parent}; librosa "
          f"{'none' if librosa is None else librosa.__version__}, numpy {np.__version__}, "
          f"scipy {scipy.__version__}, Python {sys.version.split()[0]}")
    check_urgent()
    check_ssr_eval()
    check_stored()
    check_single_precision()
    check_arithmetic()
    failed = print_table()
    print(f"\n{time.perf_counter() - start:.0f} s; "
          + ("SOME DIFFERENCES ARE PAST TOLERANCE" if failed
             else "every difference within its tolerance"))
    return 1 if failed else 0


def print_table():
    """The table of every check; True if one is past its limit."""
    width = max(len(r[0]) for r in RESULTS)
    print(f"\n{'check':<{width}} {'rate':>8} {'cases':>6} {'max |diff|':>11} "
          f"{'max rel':>9} {'limit':>11}")
    failed = False
    for check, rate, cases, diff, rel, tolerance, on in RESULTS:
        over = (rel if on == 'rel' else diff) > tolerance
        failed |= over
        limit = ('-' if math.isinf(tolerance)
                 else f"{tolerance:.1e}".replace('.0e', 'e') + f" {on}")
        shown_rel = '-' if math.isnan(rel) else f"{rel:.2e}"
        print(f"{check:<{width}} {rate!s:>8} {cases:>6} {diff:11.2e} {shown_rel:>9} "
              f"{limit:>11}{'  PAST' if over else ''}")
    return failed


if __name__ == '__main__':
    sys.exit(main())
