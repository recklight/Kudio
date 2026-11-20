# -*- coding: utf-8 -*-
"""Check how audio_report (kudio/core/report.py) tells an upsampled or
band-limited file from one recorded at its own rate, and which rate it names.

Run it from anywhere; it imports the kudio of the checkout it sits in::

    python tools/validation/band_limit.py

It needs numpy, scipy and soxr, which librosa installs with itself; I ran it
with numpy 2.0.2, scipy 1.13.1 and soxr 1.1.0 on Python 3.9.13.

There is no reference implementation to compare with, so the check is on
signals whose history is known. Every one is 2 seconds long, and every one
is checked both as float and quantised to 16 bits, the way it would come off
a WAV file, except the upsampled ones, which are only quantised.

What it checks:

1. Upsampled files. Three kinds of content: the harmonic speech of
   tests/test_report.py with its broadband bed, white noise, and noise
   low-passed at 0.84 of its own Nyquist frequency, so that it stops short of
   it. Ten rate pairs, 8000 -> 16000 to 96000 -> 192000, at 0, -20 and
   -40 dB. Eight resamplers: scipy's resample_poly with its default window
   and with ("kaiser", 8.6), scipy's FFT resample, soxr at VHQ, HQ, MQ and
   LQ, and linear interpolation with np.interp. At 0 and -20 dB every file
   must be flagged, except linear interpolation by a ratio that is not a
   whole number, which the mirror test does not model. No file may be named
   after a rate it was not made from, nor said to hold content past 1.06 of
   the Nyquist frequency of the highest rate it is named after.
2. Where each resampler puts the end of the content, as a fraction of the
   old Nyquist frequency: the naming takes each rate the end falls 0.95-1.06
   of, so every resampler but soxr LQ has to land in that window.
3. Codec cuts: speech at 44100 Hz cut by a brick wall at 15500, 16000, 17500,
   19500 and 20000 Hz, at 0 and -20 dB, all flagged.
4. Recordings made at their own rate, five seeds of each: speech, white
   noise, a spectrum falling 6 dB an octave and a random walk at the eight
   rates from 8000 to 192000 Hz; at 192000 Hz a floor rising towards
   Nyquist, and a tone with a band that stops at 10, 20, 30, 50 or 80 kHz,
   over a flat floor and over a rising one; at 96000 Hz bands stopping at 5,
   10 and 20 kHz over a floor; and mains hum at 48000 Hz. None may be
   flagged. The script prints how close the closest come to each limit.

It prints each count and exits with 1 if one is past its limit. The run
takes about 2 minutes, most of it in the margins of step 4, which ask every
native signal about all 200 frequencies of the scan.
"""
from __future__ import annotations

import collections
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import kudio  # noqa: E402
from kudio.core import report  # noqa: E402

PAIRS = ((8000, 16000), (11025, 44100), (16000, 44100), (16000, 48000), (22050, 44100),
         (24000, 48000), (44100, 96000), (48000, 96000), (48000, 192000), (96000, 192000))
RESAMPLERS = ('poly', 'poly_kaiser', 'fft', 'VHQ', 'HQ', 'MQ', 'LQ', 'linear')
LEVELS_DB = (0, -20, -40)
RESULTS = []


def result(check, cases, wrong, allowed=0):
    RESULTS.append((check, cases, wrong, allowed))


def q16(y):
    return np.round(np.clip(y, -1.0, 1.0 - 2.0 ** -15) * 32768.0) / 32768.0


def speech(sr, rng, seconds=2.0):
    t = np.arange(int(sr * seconds)) / sr
    y = sum(0.4 / k * np.sin(2 * np.pi * 180 * k * t) for k in range(1, 12))
    y *= 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)
    return y / np.max(np.abs(y)) * 0.6 + 0.02 * rng.standard_normal(len(t))


def content(kind, sr, rng):
    if kind == 'speech':
        return speech(sr, rng)
    if kind == 'white':
        return 0.1 * rng.standard_normal(2 * sr)
    return kudio.lowpass(rng.standard_normal(2 * sr), sr, cutoff=0.42 * sr) * 0.1


def upsample(y, a, b, how):
    import soxr
    g = math.gcd(a, b)
    if how == 'poly':
        return signal.resample_poly(y, b // g, a // g)
    if how == 'poly_kaiser':
        return signal.resample_poly(y, b // g, a // g, window=('kaiser', 8.6))
    if how == 'fft':
        return signal.resample(y, int(len(y) * b / a))
    if how == 'linear':
        n = int(len(y) * b / a)
        return np.interp(np.arange(n) * a / b, np.arange(len(y)), y)
    return soxr.resample(y, a, b, quality=how)


def check_upsampled():
    rng = np.random.default_rng(5)
    tally = collections.Counter()
    ratios = collections.defaultdict(list)
    missed = []
    fits, smeared = [], []
    for kind in ('speech', 'white', 'low-passed'):
        for a, b in PAIRS:
            base = content(kind, a, rng)
            for level in LEVELS_DB:
                for how in RESAMPLERS:
                    y = q16(upsample(base * 10 ** (level / 20), a, b, how))
                    r = kudio.audio_report(y, b)
                    modelled = how != 'linear' or b % a == 0
                    key = how, level
                    tally[key, 'cases'] += 1
                    tally[key, 'flagged'] += r.band_limited
                    tally[key, 'alone'] += r.source_rates == (a,)
                    tally[key, 'pair'] += len(r.source_rates) == 2 and a in r.source_rates
                    tally[key, 'unnamed'] += r.band_limited and not r.source_rates
                    tally[key, 'wrong'] += bool(r.source_rates) and a not in r.source_rates
                    if r.band_limited and r.source_rates:
                        tally['said', 'cases'] += 1
                        tally['said', 'past'] += r.band_edge_hz > 1.06 * max(r.source_rates) / 2
                    tally[key, 'mirrored'] += r.mirrored
                    if level > -40 and modelled:
                        tally['must', 'cases'] += 1
                        tally['must', 'missed'] += not r.band_limited
                    if modelled and not r.band_limited:
                        edges = report._Edges(*report._welch_db(y, b), b)
                        missed.append((edges.cliff_and_depth(a / 2), f"{kind} {how}"))
                    if how == 'linear' and modelled and level > -40:
                        tally['mirror', 'cases'] += 1
                        tally['mirror', 'missed'] += not r.mirrored
                    if how == 'linear':
                        edges = report._Edges(*report._welch_db(y, b), b)
                        (fits if modelled else smeared).append(edges.image_fit(a / 2))
                    if kind != 'low-passed' and how != 'linear' and level > -40:
                        edges = report._Edges(*report._welch_db(y, b), b)
                        stops = edges.edge_frequency(a / 2, a / 2)
                        ratios[how].append(stops / (a / 2))
    print('Upsampled, quantised to 16 bits: flagged / named alone / named with its '
          'neighbour / unnamed / wrong, of 30 at each level')
    for how in RESAMPLERS:
        cells = []
        for level in LEVELS_DB:
            t = [tally[(how, level), k] for k in ('flagged', 'alone', 'pair', 'unnamed', 'wrong')]
            cells.append(f"{level:+3d} dB {t[0]:2d}/{t[1]:2d}/{t[2]:2d}/{t[3]:2d}/{t[4]}")
        mirrored = sum(tally[(how, level), 'mirrored'] for level in LEVELS_DB)
        print(f"  {how:<12} " + '   '.join(cells)
              + (f"   mirrored {mirrored}" if how == 'linear' else ''))
    if missed:
        depths = [depth for (_, depth), _ in missed]
        kinds = collections.Counter(name.split()[0] for _, name in missed)
        print(f"  not flagged: {len(missed)} ("
              + ', '.join(f'{kind} {count}' for kind, count in kinds.items())
              + f"), with a depth of {min(depths):.1f} to {max(depths):.1f} dB at the old "
              f"Nyquist frequency")
    print(f"  linear by a whole factor: image off the curve by at most "
          f"{max(f[0] for f in fits):.2f} dB, median at most {max(abs(f[1]) for f in fits):.2f} "
          f"dB from it")
    smeared = [f for f in smeared if f is not None]
    print(f"  linear by other ratios: off the curve by {min(f[0] for f in smeared):.2f} to "
          f"{max(f[0] for f in smeared):.2f} dB, median {min(f[1] for f in smeared):+.2f} to "
          f"{max(f[1] for f in smeared):+.2f} dB from it")
    wrong = sum(tally[(how, level), 'wrong'] for how in RESAMPLERS for level in LEVELS_DB)
    result('upsampled at 0 and -20 dB, not flagged', tally['must', 'cases'],
           tally['must', 'missed'])
    result('linear by a whole factor at 0 and -20 dB, not mirrored',
           tally['mirror', 'cases'], tally['mirror', 'missed'])
    result('upsampled, named after a rate it was not made from',
           len(RESAMPLERS) * len(LEVELS_DB) * 30, wrong)
    result('named, the content said to go past 1.06 of its Nyquist',
           tally['said', 'cases'], tally['said', 'past'])

    print('\nWhere the content ends, over the old Nyquist frequency (speech and white '
          'noise at 0 and -20 dB):')
    outside = cases = 0
    for how in RESAMPLERS[:-1]:
        lo, hi = min(ratios[how]), max(ratios[how])
        print(f"  {how:<12} {lo:.3f} to {hi:.3f}")
        if how != 'LQ':
            cases += len(ratios[how])
            outside += sum(not 0.95 <= v <= 1.06 for v in ratios[how])
    result('end of the content outside 0.95-1.06, LQ aside', cases, outside)


def check_codec_cuts():
    rng = np.random.default_rng(6)
    missed, off = 0, 0.0
    print('\nCodec cuts at 44100 Hz:')
    for cut in (15500, 16000, 17500, 19500, 20000):
        for level in (0, -20):
            y = speech(44100, rng) * 10 ** (level / 20)
            spectrum = np.fft.rfft(y)
            spectrum[np.fft.rfftfreq(len(y), 1 / 44100) > cut] = 0
            r = kudio.audio_report(q16(np.fft.irfft(spectrum, len(y))), 44100)
            missed += not r.band_limited
            off = max(off, abs(r.band_edge_hz / cut - 1))
            print(f"  {cut} Hz {level:+3d} dB: flagged {r.band_limited}, named {r.source_rates}, "
                  f"content said to stop at {r.band_edge_hz:.0f} Hz")
    result('codec cuts, not flagged', 10, missed)
    print(f"  the stop given is at most {100 * off:.2f}% from the cut")


def natives(seed):
    rng = np.random.default_rng(seed)
    out = []
    for sr in (8000, 16000, 22050, 24000, 44100, 48000, 96000, 192000):
        n = 2 * sr
        out.append((f'speech {sr}', speech(sr, rng), sr))
        out.append((f'white {sr}', 0.1 * rng.standard_normal(n), sr))
        out.append((f'falling {sr}', signal.lfilter([1], [1, -0.9], rng.standard_normal(n))
                    * 0.02 + 1e-4 * rng.standard_normal(n), sr))
        out.append((f'random walk {sr}', np.cumsum(rng.standard_normal(n)) * 0.001, sr))
    n = 192000
    t = np.arange(n) / 192000
    out.append(('rising floor 192000',
                signal.lfilter([1, -0.9], [1], rng.standard_normal(n)) * 0.01, 192000))
    for hz in (10000, 20000, 30000, 50000, 80000):
        band = kudio.lowpass(rng.standard_normal(n), 192000, cutoff=hz) * 0.05
        out.append((f'tone, band to {hz}, flat floor 192000',
                    0.3 * np.sin(2 * np.pi * 300 * t) + band
                    + 0.003 * rng.standard_normal(n), 192000))
        out.append((f'band to {hz}, rising floor 192000',
                    band + signal.lfilter([1, -0.95], [1], rng.standard_normal(n)) * 0.0005,
                    192000))
    for hz in (5000, 10000, 20000):
        band = kudio.lowpass(rng.standard_normal(96000), 96000, cutoff=hz) * 0.05
        out.append((f'band to {hz}, floor 96000', band + 0.002 * rng.standard_normal(96000),
                    96000))
    t = np.arange(96000) / 48000
    hum = sum(0.02 / k * np.sin(2 * np.pi * 50 * k * t) for k in range(1, 480))
    out.append(('hum 48000', hum + 0.001 * rng.standard_normal(96000), 48000))
    return out


def check_natives():
    flagged, cases = [], 0
    deepest = (-math.inf, '')
    closest = (math.inf, '')         # how near an image fit comes to passing; 1 passes
    for seed in range(5):
        for name, y, sr in natives(seed):
            for form, z in (('float', y), ('16-bit', q16(y))):
                cases += 1
                r = kudio.audio_report(z, sr)
                if r.band_limited:
                    flagged.append(f"{name} {form}: {r.source_rates} mirrored={r.mirrored}")
                edges = report._Edges(*report._welch_db(z, sr), sr)
                nyquist = sr / 2
                common = [rate / 2 for rate in report._SOURCE_RATES if rate / 2 <= 0.9 * nyquist]
                for fc in common + list(np.geomspace(0.1 * nyquist, 0.9 * nyquist, 200)):
                    cliff, depth = edges.cliff_and_depth(float(fc))
                    if cliff >= report._CLIFF_DB and depth > deepest[0]:
                        deepest = (depth, f"{name} {form} at {fc:.0f} Hz")
                for fc in common:
                    fit = edges.image_fit(fc)
                    if fit is None:
                        continue
                    score = max(fit[0] / report._IMAGE_SPREAD_DB,
                                abs(fit[1]) / report._IMAGE_OFFSET_DB)
                    if score < closest[0]:
                        closest = (score, f"spread {fit[0]:.2f} dB, offset {fit[1]:+.2f} dB "
                                          f"({name} {form} at {fc:g} Hz)")
    print(f"\nRecordings at their own rate: {cases}, flagged {len(flagged)}")
    for line in flagged:
        print(f"  {line}")
    print(f"  deepest edge with a cliff of {report._CLIFF_DB:g} dB or more: "
          f"{deepest[0]:.1f} dB ({deepest[1]}); the limit is {report._DEPTH_DB:g}")
    print(f"  closest to an image: {closest[1]}; the limits are "
          f"{report._IMAGE_SPREAD_DB:g} and {report._IMAGE_OFFSET_DB:g} dB")
    result('recordings at their own rate, flagged', cases, len(flagged))


def main():
    import scipy
    import soxr
    start = time.perf_counter()
    print(f"kudio {kudio.__version__} from {Path(kudio.__file__).parent}; numpy "
          f"{np.__version__}, scipy {scipy.__version__}, soxr {soxr.__version__}, "
          f"Python {sys.version.split()[0]}\n")
    check_upsampled()
    check_codec_cuts()
    check_natives()
    width = max(len(r[0]) for r in RESULTS)
    print(f"\n{'check':<{width}} {'cases':>5} {'count':>5} {'limit':>5}")
    failed = False
    for check, cases, wrong, allowed in RESULTS:
        over = wrong > allowed
        failed |= over
        print(f"{check:<{width}} {cases:>5} {wrong:>5} {allowed:>5}{'  PAST' if over else ''}")
    print(f"\n{time.perf_counter() - start:.0f} s; "
          + ("SOME COUNTS ARE PAST THEIR LIMIT" if failed else "every count within its limit"))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
