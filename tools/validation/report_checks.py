# -*- coding: utf-8 -*-
"""Check tonal_lines (kudio/core/lines.py), and the clipping count and the
sample grid of audio_report (kudio/core/report.py), on signals whose answer
is known.

Run it from anywhere; it imports the kudio of the checkout it sits in::

    python tools/validation/report_checks.py

It needs numpy and scipy and nothing else; I ran it with numpy 2.0.2 and
scipy 1.13.1 on Python 3.9.13.

What it checks:

1. tonal_lines on sines at 120 frequencies spread evenly in log frequency
   from 50 Hz to 0.45 of the rate, at 16, 48 and 192 kHz, at -20 and
   -40 dBFS over white noise at -60 dBFS, 3 s each: how far the frequency
   and the level are from the sine's, and how much the prominence grows when
   the resolution is halved, from 10 to 5 to 2.5 Hz. Then what it finds where
   there is no line: white, pink and brown noise, 1 s and 10 s, at the three
   rates, five seeds each; and a voice whose pitch glides or wavers. Last,
   two combs on one clip, a hum of 10 harmonics and a comb of 3 with an
   unrelated line beside it, at 40 settings of fundamentals, levels and
   seeds: is each grouped right.
2. The clipping count. Smooth peaks, which must not count: sines normalised
   to 1.0, 0.9999, 0.9995 and 0.9992 of full scale, at 20 Hz to 2 kHz and
   16 to 192 kHz, as float and as 16 bits. Plateaus, which must: the same
   sines driven to 1.0005 to 2.0 times full scale and cut back to it. How
   near each comes to the threshold, as the plateau's share of its run over
   the share that makes it count, among plateaus clip_run long or more.
   Then clip_run and full_scale_samples
   against a count of every sample at full scale.
3. The sample grid: steps of 1, 16 and 256 at 16 bits (16, 12 and 8 bits),
   12 and 3 at 16 bits, the same with an offset, 1 at 24 bits; and the
   clips that must have none: filtered float audio, a square wave, a take
   within a few steps of silence, digital silence.

It prints each count and exits with 1 if one is past its limit. The run
takes about two minutes.
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import kudio  # noqa: E402
from kudio.core import report  # noqa: E402

RESULTS = []


def result(check, cases, wrong, allowed=0):
    RESULTS.append((check, cases, wrong, allowed))


def q16(y):
    return np.round(np.clip(y, -1.0, 1.0 - 2.0 ** -15) * 32768.0) / 32768.0


# ------------------------------------------------------------- tonal_lines

def check_lines_on_sines():
    rng = np.random.default_rng(1)
    worst_hz = worst_bins = worst_level = 0.0
    missed = extra = cases = 0
    for sr in (16000, 48000, 192000):
        seconds = 3
        t = np.arange(seconds * sr) / sr
        for hz in np.geomspace(50, 0.45 * sr, 120):
            for level in (-20.0, -40.0):
                amplitude = 10 ** (level / 20)
                y = (amplitude * np.sin(2 * np.pi * hz * t + rng.uniform(0, 2 * np.pi))
                     + 10 ** (-60 / 20) * rng.standard_normal(t.size))
                found = kudio.tonal_lines(y, sr)
                cases += 1
                near = [line for line in found.lines if abs(line.hz - hz) < 3 * found.resolution_hz]
                if not near:
                    missed += 1
                    continue
                extra += len(found.lines) - 1
                line = near[0]
                worst_hz = max(worst_hz, abs(line.hz - hz))
                worst_bins = max(worst_bins, abs(line.hz - hz) / found.resolution_hz)
                worst_level = max(worst_level, abs(line.level_dbfs - level))
    print(f"Sines, {cases}: frequency off by at most {worst_hz:.3f} Hz "
          f"({worst_bins:.3f} of a bin), level by at most {worst_level:.2f} dB")
    result('tonal_lines: sine not found', cases, missed)
    result('tonal_lines: lines besides the sine', cases, extra)
    result('tonal_lines: level more than 1.42 dB off', cases, int(worst_level > 1.42))
    return worst_hz, worst_bins, worst_level


def check_prominence_per_resolution():
    rng = np.random.default_rng(2)
    sr = 48000
    t = np.arange(10 * sr) / sr
    steps = []
    for hz in (440.0, 3150.0, 9100.0):
        y = 0.01 * np.sin(2 * np.pi * hz * t) + 0.01 * rng.standard_normal(t.size)
        p = [max(line.prominence_db for line in kudio.tonal_lines(y, sr, resolution_hz=r).lines)
             for r in (10.0, 5.0, 2.5)]
        steps += [p[1] - p[0], p[2] - p[1]]
    print(f"Prominence gained for each halving of resolution_hz: "
          f"{min(steps):.2f} to {max(steps):.2f} dB")
    return min(steps), max(steps)


def _glide(sr, seconds, rng, vibrato):
    t = np.arange(int(seconds * sr)) / sr
    if vibrato:
        pitch = 140 + 15 * np.sin(2 * np.pi * 5.5 * t)
    else:
        pitch = np.interp(t, [0, seconds / 2, seconds], [110, 170, 120])
    phase = 2 * np.pi * np.cumsum(pitch) / sr
    voice = sum(np.sin(k * phase) / k for k in range(1, 30))
    words = np.clip(np.sin(2 * np.pi * 1.5 * t), 0, None) ** 2
    return 0.05 * voice * words + 0.002 * rng.standard_normal(t.size)


def check_no_lines():
    flagged = cases = 0
    for sr in (16000, 48000, 192000):
        for seconds in (1, 10):
            for seed in range(5):
                rng = np.random.default_rng(seed)
                white = rng.standard_normal(seconds * sr)
                for y in (white, signal.lfilter([1], [1, -0.99], white),
                          np.cumsum(white)):
                    y = 0.1 * y / np.max(np.abs(y))
                    cases += 1
                    flagged += len(kudio.tonal_lines(y, sr).lines)
    print(f"Noise without lines: {cases} clips, {flagged} lines found")
    result('tonal_lines: lines in noise', cases, flagged)
    voices = {}
    for vibrato in (False, True):
        lines = combs = 0
        for seed in range(5):
            found = kudio.tonal_lines(_glide(16000, 6, np.random.default_rng(seed), vibrato), 16000)
            lines += len(found.lines)
            combs += len(found.combs)
        name = "wavering 125-155 Hz" if vibrato else "gliding 110-170 Hz"
        voices[name] = (lines, combs)
        print(f"  a voice {name}, 5 takes of 6 s: {lines} lines, {combs} combs")
    return cases, voices


def check_two_combs():
    right = cases = 0
    t = np.arange(2 * 48000) / 48000
    for hum in (50.0, 60.0):
        for comb in (1000.0, 1300.0):
            for level in (0.02, 0.005):
                for seed in range(5):
                    rng = np.random.default_rng(seed)
                    y = sum(0.05 / k * np.sin(2 * np.pi * hum * k * t + rng.uniform(0, 6))
                            for k in range(1, 11))
                    y = y + sum(level * np.sin(2 * np.pi * comb * k * t + rng.uniform(0, 6))
                                for k in (1, 2, 3))
                    y = y + level * np.sin(2 * np.pi * 2.5 * comb * t)
                    y = y + 0.003 * rng.standard_normal(t.size)
                    got = {round(c.fundamental_hz): c.harmonics
                           for c in kudio.tonal_lines(y, 48000).combs}
                    cases += 1
                    right += got == {round(hum): tuple(range(1, 11)),
                                     round(comb): (1, 2, 3)}
    print(f"Two combs on one clip: {right} of {cases} grouped right")
    result('tonal_lines: two combs grouped wrong', cases, cases - right)
    return right, cases


# ---------------------------------------------------------------- clipping

def _share_over_threshold(y):
    """For each pinned run, the plateau's share of the run over the share
    that makes it count; the largest, and whether any counted."""
    pinned = np.abs(y) >= report._CLIP_LEVEL
    edges = np.diff(np.concatenate(([0], pinned.view(np.int8), [0])))
    best = 0.0
    for a, b in zip(np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0]):
        run = np.abs(y[a:b])
        level = run.max()
        rise = level - report._CLIP_LEVEL
        if rise <= report._CLIP_FLATNESS or b - a < report._CLIP_RUN:
            continue
        flat = run >= level - report._CLIP_FLATNESS
        lengths = [len(s) for s in np.split(flat, np.nonzero(np.diff(flat))[0] + 1)
                   if s[0] and len(s) >= report._CLIP_RUN]
        if lengths:
            best = max(best, max(lengths) / len(run)
                       / (2 * math.sqrt(report._CLIP_FLATNESS / rise)))
    return best


def _plateau_samples(y):
    """Samples in runs at least clip_run long within 1e-4 of full scale: what
    a count of plateaus alone would give."""
    top = (np.abs(y) >= 1.0 - report._CLIP_FLATNESS).astype(np.int8)
    edges = np.diff(np.concatenate(([0], top, [0])))
    lengths = np.nonzero(edges == -1)[0] - np.nonzero(edges == 1)[0]
    return int(lengths[lengths >= report._CLIP_RUN].sum())


def check_smooth_peaks():
    counted = cases = 0
    closest = 0.0
    for sr in (16000, 44100, 48000, 96000, 192000):
        t = np.arange(sr) / sr
        for hz in np.geomspace(20, 2000, 25):
            for top in (1.0, 0.9999, 0.9995, 0.9992):
                y = np.sin(2 * np.pi * hz * t + 0.3)
                y = top * y / np.max(np.abs(y))
                for z in (y, q16(y)):
                    cases += 1
                    counted += kudio.audio_report(z, sr).clipped_samples > 0
                    closest = max(closest, _share_over_threshold(z))
    print(f"Smooth peaks: {cases} sines, {counted} counted as clipped; the nearest "
          f"holds {closest:.2f} of the share that counts")
    result('clipping: a smooth peak counted', cases, counted)
    return cases, closest


def check_plateaus():
    rows = {}
    nearest_counted = math.inf
    for drive in (1.0005, 1.001, 1.002, 1.005, 1.01, 1.1, 2.0):
        full = part = cases = 0
        for sr in (16000, 48000, 192000):
            t = np.arange(sr) / sr
            for hz in np.geomspace(20, 2000, 25):
                y = np.clip(drive * np.sin(2 * np.pi * hz * t + 0.3), -1.0, 1.0)
                on_top = _plateau_samples(y)
                if not on_top:
                    continue        # plateaus shorter than clip_run: nothing to count
                got = kudio.audio_report(y, sr).clipped_samples
                cases += 1
                full += got == on_top
                part += 0 < got < on_top
                if got:
                    nearest_counted = min(nearest_counted, _share_over_threshold(y))
        rows[drive] = (full, part, cases)
        if drive >= 1.01:
            result(f'clipping: cut {100 * (1 - 1 / drive):.2g}%, not counted in full',
                   cases, cases - full)
        cut = 100 * (1 - 1 / drive)
        print(f"  driven {drive:g}x ({cut:.2f}% cut off the top): {full} of {cases} "
              f"counted in full, {part} in part")
    return rows, nearest_counted


def check_clip_definitions():
    rng = np.random.default_rng(3)
    sr = 16000
    t = np.arange(10 * sr) / sr
    phase = 2 * np.pi * np.cumsum(120 + 30 * np.sin(2 * np.pi * 0.3 * t)) / sr
    voice = sum(np.sin(k * phase) / k for k in range(1, 20)) * (np.sin(2 * np.pi * 2 * t) > 0)
    voice = voice / np.max(np.abs(voice))
    rows = []
    for gain in (1.2, 2.0, 5.0):
        y = np.clip(gain * voice + 0.002 * rng.standard_normal(t.size), -1, 1)
        every = int(np.count_nonzero(np.abs(y) >= report._CLIP_LEVEL))
        r1 = kudio.audio_report(y, sr, clip_run=1)
        rows.append((gain, every, r1.full_scale_samples, r1.clipped_samples,
                     kudio.audio_report(y, sr, clip_run=2).clipped_samples,
                     kudio.audio_report(y, sr).clipped_samples))
    print("Overdriven voice: every sample at full scale / full_scale_samples / "
          "clip_run 1, 2, 3")
    for gain, every, fs, c1, c2, c3 in rows:
        print(f"  x{gain:g}: {every} / {fs} / {c1}, {c2}, {c3}")
    result('clipping: full_scale_samples off the count', len(rows),
           sum(fs != every for _, every, fs, *_ in rows))
    return rows


# ------------------------------------------------------------------- grid

def check_grid():
    rng = np.random.default_rng(4)
    sr = 16000
    t = np.arange(2 * sr) / sr
    voice = 0.3 * np.sin(2 * np.pi * 220 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))
    voice = voice + 0.01 * rng.standard_normal(t.size)
    wrong = cases = 0
    print("Known grids: step, offset -> sample_step x 32768 and grid_bits")
    for unit, step, offset, bits in ((16, 1, 0, 16.0), (16, 16, 0, 12.0), (16, 256, 0, 8.0),
                                     (16, 12, 0, 16 - math.log2(12)),
                                     (16, 3, 0, 16 - math.log2(3)),
                                     (16, 12, 4, 16 - math.log2(12)),
                                     (24, 1, 0, 24.0)):
        scale = 2.0 ** (unit - 1)
        codes = np.round((voice * scale - offset) / step) * step + offset
        r = kudio.audio_report(codes / scale, sr)
        cases += 1
        ok = abs(r.grid_bits - bits) < 1e-9 and abs(r.sample_step - step / scale) < 1e-15
        wrong += not ok
        print(f"  {unit}-bit step {step}, offset {offset}: {r.sample_step * 32768:g}, "
              f"{r.grid_bits:.4f} (expected {bits:.4f})")
    result('grid: known grid measured wrong', cases, wrong)
    none = {
        "filtered float": signal.lfilter([0.3, 0.3], [1, -0.2], voice),
        "square wave": 0.5 * np.sign(np.sin(2 * np.pi * 100 * t)),
        "within 7 steps of silence": np.round(rng.uniform(-7, 7, t.size)) / 32768,
        "digital silence": np.zeros(t.size),
    }
    named = 0
    for name, y in none.items():
        r = kudio.audio_report(y, sr)
        named += not (r.sample_step == 0.0 and math.isnan(r.grid_bits))
        print(f"  {name}: step {r.sample_step:g}, grid_bits {r.grid_bits}")
    result('grid: a grid named where there is none', len(none), named)


def main():
    import scipy
    start = time.perf_counter()
    print(f"kudio {kudio.__version__} from {Path(kudio.__file__).parent}; numpy "
          f"{np.__version__}, scipy {scipy.__version__}, Python {sys.version.split()[0]}\n")
    check_lines_on_sines()
    check_prominence_per_resolution()
    check_no_lines()
    check_two_combs()
    print()
    check_smooth_peaks()
    print("Plateaus, sines at 20 Hz-2 kHz and 16, 48 and 192 kHz cut back to full scale:")
    _, nearest = check_plateaus()
    print(f"  the nearest counted plateau holds {nearest:.2f} of the share that counts")
    check_clip_definitions()
    print()
    check_grid()
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
