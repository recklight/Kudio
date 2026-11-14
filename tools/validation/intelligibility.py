# -*- coding: utf-8 -*-
"""Check kudio's NCM and CSII (kudio/core/intelligibility.py) against pysepm,
and reproduce the figures the module docstring quotes that no test pins.

Run it from anywhere; it imports the kudio of the checkout it sits in::

    python tools/validation/intelligibility.py
    python tools/validation/intelligibility.py --loizou-examples DIR

Besides what kudio itself needs, it needs the oracle, pysepm_evo (I ran
0.1.1, with numpy 2.2.6 and scipy 1.15.2), which brings numba and pystoi
with it. pysepm is GPL-3.0: it runs here beside kudio as something to compare
with, and none of it is in kudio. 0.1.1 still imports scipy.signal.kaiser,
which newer scipy keeps only in scipy.signal.windows, so the script puts it
back before the import. pysepm's ncm and csii are ports of NCM.m and CSII.m,
the MATLAB files kudio follows.

Every signal is made here from fixed seeds: the test speech of the module
docstring and of tests/test_intelligibility.py (syllables of a harmonic
source through moving formants, with pauses on a noise floor 60 dB down);
white, pink and babble noise from -5 to 20 dB SNR; rooms from kudio.rir with
RT60 of 0.3, 0.6 and 1.2 s; kudio's log-MMSE enhancer on noisy speech;
an output low-passed at 2 kHz under noise; and at 16 kHz a reference that
holds nothing above 4 kHz, as if it came from an 8 kHz recording.

What it checks:

1. ncm() against pysepm's ncm at 8 and 16 kHz. pysepm filters each band as
   one transfer function and kudio in second-order sections, so the run is
   repeated with pysepm's own transfer functions put in place of kudio's
   sections, on signals scaled to a peak of 1 (kudio scales them so, and the
   transfer functions are sensitive enough for the scaling to show): that
   isolates the filter form as the whole difference.
2. How far the lowest band strays from the exact filter, computed in the
   frequency domain from the filter's poles and zeros, as one transfer
   function and as sections; and how far pysepm's own ncm moves when its
   inputs are only scaled, which NCM cannot feel.
3. csii() against pysepm's csii at 8, 11.025, 16, 22.05, 44.1 and 48 kHz,
   with float64 eps added to both signals for pysepm, as CSII.m adds it, and
   then without, which parts only on digital silence.
4. What pysepm returns for an exact, a scaled and an inverted copy, for an
   output of digital silence and for a level with no frames.
5. The figures the docstring quotes from many draws: the average score of
   an unrelated output against short clips, and CSII of one pair at every
   rate.
6. What an offset on the reference does to CSII of a muted output, with
   offsets of 0.001% to 5% of the peak and with pink floors, at 8 and
   16 kHz, and that the cure the docstring gives takes it to 0: the
   reference high-passed at 20 Hz, forwards and backwards. Then how far
   that cure, with the output's mean taken out as well, moves CSII of every
   validation pair.
7. With --loizou-examples DIR, the example pair in the folder of NCM.m and
   CSII.m (S_03_01.wav and S_03_01_babble_sn0_klt.wav, from the MATLAB code
   that comes with Loizou's "Speech Enhancement: Theory and Practice"),
   against the values its readme prints: 0.4162 for NCM, and 0.5556,
   0.2480 and 0.0457 for CSII.

It prints the largest differences and exits with 1 if one is past its
tolerance. The whole run takes about four minutes, most of it in pysepm's
ncm, which designs its resampling filter afresh on every call.
"""
from __future__ import annotations

import argparse
import math
import sys
import warnings
from fractions import Fraction
from pathlib import Path

import numpy as np
import scipy.signal
import scipy.signal.windows
from scipy.signal import (butter, freqz_zpk, lfilter, resample_poly, sosfilt,
                          sosfiltfilt)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import kudio  # noqa: E402
from kudio.core import intelligibility as module  # noqa: E402
from kudio.core.intelligibility import csii, ncm  # noqa: E402

if not hasattr(scipy.signal, 'kaiser'):
    scipy.signal.kaiser = scipy.signal.windows.kaiser
import pysepm_evo  # noqa: E402

EPS = np.finfo(float).eps

#: rough F1, F2, F3 of five vowels, in Hz
VOWELS = ((730, 1090, 2440), (270, 2290, 3010), (300, 870, 2240),
          (530, 1840, 2480), (570, 840, 2410))

#: (check, rate, cases, largest absolute difference, largest relative
#: difference, tolerance on the absolute one)
RESULTS = []


def report(check, rate, cases, diff, rel, tolerance):
    RESULTS.append((check, rate, cases, diff, rel, tolerance))


# --- signals, the same as in tests/test_intelligibility.py -----------------------

def speech(sr, seconds=3.0, seed=1, floor=5e-4):
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


def lowpass(y, sr, cutoff):
    return sosfiltfilt(butter(8, cutoff, fs=sr, output='sos'), y)


def pairs(sr):
    """``(name, ref, deg)`` for every kind of damage, at *sr*."""
    clean = speech(sr)
    n = len(clean)
    out = []
    for colour in ('white', 'pink', 'babble'):
        hiss = noise(n, 7, colour, sr)
        for snr in (-5, 0, 5, 10, 20):
            out.append((f'{colour} {snr:+d} dB', clean, mix(clean, hiss, snr)))
    for rt60 in (0.3, 0.6, 1.2):
        wet = kudio.apply_rir(clean, kudio.rir(sr, rt60=rt60, seed=3))
        out.append((f'room {rt60} s', clean, wet.astype(float)))
    for colour, snr in (('white', 5), ('babble', 0)):
        noisy = mix(clean, noise(n, 7, colour, sr), snr)
        out.append((f'log-MMSE, {colour} {snr} dB', clean,
                    kudio.spectral_enhance(noisy, sr).astype(float)))
    out.append(('low-passed 2 kHz, white 20 dB', clean,
                mix(lowpass(clean, sr, 2000), noise(n, 7, 'white', sr), 20)))
    if sr == 16000:
        narrow = resample_poly(resample_poly(clean, 1, 2), 2, 1)
        out.append(('ref from 8 kHz, white 10 dB', narrow,
                    mix(narrow, noise(n, 7, 'white', sr), 10)))
    return out


def quietly(fn, *args):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return fn(*args)


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


# --- 1 and 2: NCM ------------------------------------------------------------------

def pysepm_band_filters(sr):
    """kudio's _band, with pysepm's own (b, a) for each band: the edges from
    its get_band, butter(4, ...) as a transfer function, and lfilter."""
    from pysepm_evo.intelligibilityMeasures import get_band
    edges = get_band(20, sr)
    tfs = [butter(4, np.array([edges[k], edges[k + 1]]) * (2 / sr), btype='bandpass')
           for k in range(20)]
    by_section = {id(s): tf for s, tf in zip(module._ncm_filterbank(sr)[2], tfs)}
    return lambda y, sos: lfilter(*by_section[id(sos)], y)


def check_ncm():
    for sr in (8000, 16000):
        cases = pairs(sr)
        ours = [quietly(ncm, ref, deg, sr) for _, ref, deg in cases]
        theirs = [quietly(pysepm_evo.ncm, ref, deg, sr) for _, ref, deg in cases]
        diff, rel = largest(ours, theirs)
        worst = cases[int(np.argmax(np.abs(np.subtract(ours, theirs))))][0]
        report(f'ncm vs pysepm (worst: {worst})', sr, len(cases), diff, rel,
               3e-6 if sr == 16000 else 2e-8)

        scaled = [(name, ref / np.max(np.abs(ref)), deg / np.max(np.abs(deg)))
                  for name, ref, deg in cases]
        sections = module._band
        module._band = pysepm_band_filters(sr)
        try:
            swapped = [quietly(ncm, ref, deg, sr) for _, ref, deg in scaled]
        finally:
            module._band = sections
        theirs = [quietly(pysepm_evo.ncm, ref, deg, sr) for _, ref, deg in scaled]
        diff, rel = largest(swapped, theirs)
        report("ncm with pysepm's band filters vs pysepm", sr, len(cases), diff, rel, 1e-13)

        moved = [quietly(pysepm_evo.ncm, 0.3 * ref, 0.7 * deg, sr) for _, ref, deg in cases]
        diff, rel = largest(moved, [quietly(pysepm_evo.ncm, ref, deg, sr)
                                    for _, ref, deg in cases])
        report('pysepm ncm vs itself, inputs x0.3 and x0.7', sr, len(cases), diff, rel,
               math.inf)


def exact_band(y, sr, band):
    """One band of NCM's filterbank, filtered in the frequency domain from the
    filter's poles and zeros, on a signal padded far past its decay."""
    edges = module._ncm_edges(sr)
    z, p, k = butter(4, edges[band:band + 2] * (2.0 / sr), btype='bandpass', output='zpk')
    size = 1 << int(math.ceil(math.log2(len(y) + 20 * sr)))
    _, h = freqz_zpk(z, p, k, worN=2 * np.pi * np.arange(size // 2 + 1) / size)
    return np.fft.irfft(np.fft.rfft(y, size) * h, size)[:len(y)]


def check_filters():
    for sr in (8000, 16000):
        y = speech(sr)
        exact = exact_band(y, sr, 0)
        peak = np.max(np.abs(exact))
        b, a = butter(4, module._ncm_edges(sr)[:2] * (2.0 / sr), btype='bandpass')
        stray = np.max(np.abs(lfilter(b, a, y) - exact)) / peak
        ours = np.max(np.abs(sosfilt(module._ncm_filterbank(sr)[2][0], y) - exact)) / peak
        report('lowest band, transfer function vs exact (of its peak)', sr, 1,
               float(stray), float(stray), math.inf)
        report('lowest band, kudio sections vs exact (of its peak)', sr, 1,
               float(ours), float(ours), 1e-12)


# --- 3: CSII -----------------------------------------------------------------------

def check_csii():
    for sr in (8000, 11025, 16000, 22050, 44100, 48000):
        cases = pairs(sr)
        ours = [quietly(csii, ref, deg, sr) for _, ref, deg in cases]
        theirs = [quietly(pysepm_evo.csii, ref + EPS, deg + EPS, sr)
                  for _, ref, deg in cases]
        diff, rel = largest(ours, theirs)
        report('csii vs pysepm, eps added as CSII.m adds it', sr, len(cases),
               diff, rel, 1e-13)
        plain = [quietly(pysepm_evo.csii, ref, deg, sr) for _, ref, deg in cases]
        diff, rel = largest(ours, plain)
        report('csii vs pysepm as it stands', sr, len(cases), diff, rel, 1e-13)


# --- 4: copies, silence, empty levels ------------------------------------------------

def check_edges():
    print('\nWhere pysepm has no number (16 kHz, the test speech):')
    print(f"  {'output':<28} {'ncm':>8} {'pysepm':>8}   {'csii':<22} {'pysepm':<22}")
    clean = speech(16000)
    rows = [('exact copy', clean.copy()), ('copy x0.3', 0.3 * clean),
            ('inverted copy', -clean), ('digital silence', np.zeros_like(clean))]
    failed = False
    for name, deg in rows:
        ours = (quietly(ncm, clean, deg, 16000), quietly(csii, clean, deg, 16000))
        theirs = (quietly(pysepm_evo.ncm, clean, deg, 16000),
                  quietly(pysepm_evo.csii, clean, deg, 16000))
        print(f"  {name:<28} {ours[0]:8.4f} {theirs[0]:8.4f}   "
              f"{fmt(ours[1]):<22} {fmt(theirs[1]):<22}")
        want = 0.0 if name == 'digital silence' else 1.0
        failed |= ours[0] != want or tuple(ours[1]) != (want, want, want)
    steady = noise(32000, 5, 'white', 16000)
    deg = steady + 0.1 * noise(32000, 6, 'white', 16000)
    ours = quietly(csii, steady, deg, 16000)
    theirs = quietly(pysepm_evo.csii, steady + EPS, deg + EPS, 16000)
    print(f"  {'steady noise as ref, csii':<28} {'':8} {'':8}   "
          f"{fmt(ours):<22} {fmt(theirs):<22}")
    failed |= not (math.isnan(ours[2]) and math.isnan(theirs[2]))
    report('the values in the table above', 16000, len(rows) + 1, float(failed),
           float(failed), 0)


def fmt(values):
    return ' '.join('nan' if math.isnan(v) else f'{v:.4f}' for v in values)


# --- 5: figures from many draws -----------------------------------------------------

def check_figures():
    print('\nAn unrelated output against short clips of the test speech (16 kHz, '
          '8 draws each):')
    for seconds in (1.0, 3.0, 10.0):
        other, white = [], []
        for draw in range(8):
            ref = speech(16000, seconds, seed=100 + draw)
            other.append(ncm(ref, speech(16000, seconds, seed=200 + draw), 16000))
            hiss = noise(len(ref), 300 + draw, 'white', 16000)
            white.append(quietly(csii, ref, hiss, 16000))
        white = np.array(white)
        print(f"  {seconds:4.0f} s: NCM of other speech {np.mean(other):.3f} on "
              f"average ({min(other):.3f} to {max(other):.3f}); CSII of white "
              f"noise {fmt(np.nanmean(white, axis=0))} on average")

    print('\nOne pair, made at 48 kHz and taken to each rate (white noise, 5 dB):')
    ref48 = speech(48000)
    deg48 = mix(ref48, noise(len(ref48), 7, 'white', 48000), 5)
    at16 = np.array(csii(resample_poly(ref48, 1, 3), resample_poly(deg48, 1, 3), 16000))
    whole, broken = 0.0, 0.0
    for sr in (8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000):
        step = Fraction(sr, 48000)
        up, down = step.numerator, step.denominator
        here = np.array(csii(resample_poly(ref48, up, down),
                             resample_poly(deg48, up, down), sr))
        apart = float(np.max(np.abs(here - at16)))
        if (30 * sr) % 1000 == 0 and (30 * sr // 1000) % 4 == 0:
            whole = max(whole, apart)
        else:
            broken = max(broken, apart)
        print(f"  {sr:6d} Hz: CSII {fmt(here)}, {apart:.1e} from 16 kHz")
    print(f"  largest move where 30 ms is a whole number of hops: {whole:.1e}; "
          f"elsewhere: {broken:.3f}")
    report('csii at 8, 12, 24, 32, 48 kHz vs 16 kHz', 'all', 5, whole, whole, 1e-4)


# --- 6: an offset on the reference, and what takes it away -------------------------

def highpass_20(y, sr):
    """Second-order Butterworth high-pass at 20 Hz, forwards and backwards."""
    return sosfiltfilt(butter(2, 20, 'highpass', fs=sr, output='sos'), y)


def check_offsets():
    """A muted output scored against the test speech with an offset added, or
    with a pink floor in place of the white one, as it stands and with the
    reference high-passed at 20 Hz; then what that cure, with the output's
    mean taken out as well, does to CSII of every validation pair."""
    print('\nA muted output, CSII as it stands -> with the reference high-passed '
          'at 20 Hz:')
    worst, count = 0.0, 0
    for sr in (8000, 16000):
        clean = speech(sr)
        peak = np.max(np.abs(clean))
        muted = np.zeros_like(clean)
        cases = [(f'offset {dc:.3%} of the peak', clean + dc * peak)
                 for dc in (1e-5, 1e-4, 1e-3, 1e-2, 5e-2)]
        quiet = speech(sr, floor=0)
        cases += [(f'pink floor, seed {seed}', quiet + 5e-4 * noise(len(quiet), seed, 'pink', sr))
                  for seed in range(3, 8)]
        for name, ref in cases:
            before = quietly(csii, ref, muted, sr)
            after = quietly(csii, highpass_20(ref, sr), muted, sr)
            worst = max(worst, float(np.max(after)))
            count += 1
            print(f'  {sr:5d} Hz  {name:<26} {fmt(before):<22} -> {fmt(after)}')
    report('muted output, reference high-passed (largest score)', '8, 16k', count,
           worst, worst, 0)
    for sr in (8000, 16000):
        cases = pairs(sr)
        ours = [quietly(csii, ref, deg, sr) for _, ref, deg in cases]
        cured = [quietly(csii, highpass_20(ref, sr), deg - deg.mean(), sr)
                 for _, ref, deg in cases]
        diff, rel = largest(cured, ours)
        report('csii, ref high-passed and deg mean out, vs as it stands', sr,
               len(cases), diff, rel, 1.5e-4)


# --- 7: the example pair of the MATLAB code ------------------------------------------

def check_examples(folder):
    from scipy.io import wavfile
    sr, ref = wavfile.read(Path(folder) / 'S_03_01.wav')
    sr_deg, deg = wavfile.read(Path(folder) / 'S_03_01_babble_sn0_klt.wav')
    assert sr == sr_deg == 8000, (sr, sr_deg)
    ours = (quietly(ncm, ref, deg, sr),) + tuple(quietly(csii, ref, deg, sr))
    ref, deg = ref / 32768.0, deg / 32768.0            # wavread's scale
    n = min(len(ref), len(deg))
    theirs = ((quietly(pysepm_evo.ncm, ref[:n], deg[:n], sr),)
              + tuple(quietly(pysepm_evo.csii, ref[:n] + EPS, deg[:n] + EPS, sr)))
    readme = (0.4162, 0.5556, 0.2480, 0.0457)
    print('\nThe example pair of NCM.m and CSII.m (8 kHz):')
    for name, a, b, c in zip(('NCM', 'CSII high', 'CSII mid', 'CSII low'),
                             ours, theirs, readme):
        print(f"  {name:<10} kudio {a:.6f}  pysepm {b:.6f}  readme {c:.4f}")
    diff, rel = largest(ours[1:], readme[1:])
    report('csii vs the readme of CSII.m (4 places)', 8000, 1, diff, rel, 5e-5)
    diff, rel = largest(ours[1:], theirs[1:])
    report('csii vs pysepm on the example pair', 8000, 1, diff, rel, 1e-13)
    diff, rel = largest(ours[:1], theirs[:1])
    report('ncm vs pysepm on the example pair', 8000, 1, diff, rel, 2e-8)
    diff, rel = largest(ours[:1], readme[:1])
    report('ncm vs the readme of NCM.m (open)', 8000, 1, diff, rel, math.inf)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--loizou-examples', metavar='DIR',
                        help='folder holding S_03_01.wav and S_03_01_babble_sn0_klt.wav')
    args = parser.parse_args()
    check_ncm()
    check_filters()
    check_csii()
    check_edges()
    check_figures()
    check_offsets()
    if args.loizou_examples:
        check_examples(args.loizou_examples)
    return 1 if print_table() else 0


def print_table():
    """The table of every check; True if one is past its limit."""
    print(f"\n{'check':<62} {'rate':>6} {'cases':>5} {'max |diff|':>11} "
          f"{'max rel':>9} {'limit':>7}")
    failed = False
    for check, rate, cases, diff, rel, tolerance in RESULTS:
        over = diff > tolerance
        failed |= over
        # 1.5e-04 keeps its digit, 2.0e-08 prints as 2e-08
        limit = '-' if math.isinf(tolerance) else f'{tolerance:.1e}'.replace('.0e', 'e')
        print(f"{check:<62} {rate!s:>6} {cases:>5} {diff:11.2e} {rel:9.2e} "
              f"{limit:>7}{'  PAST' if over else ''}")
    return failed


if __name__ == '__main__':
    sys.exit(main())
