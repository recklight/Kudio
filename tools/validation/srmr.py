# -*- coding: utf-8 -*-
"""Check kudio's SRMR (kudio/core/srmr.py) against the four ratios the
authors' SRMRToolbox stores for its test recording, and against SRMRpy,
with the places where SRMRpy lays out its windows and builds its envelopes
differently taken one at a time.

Run it from anywhere; it imports the kudio of the checkout it sits in::

    python tools/validation/srmr.py
    python tools/validation/srmr.py --toolbox-test DIR

Besides what kudio itself needs, it needs SRMRpy (I ran 1.0, which brings
Gammatone 1.0.3, with numpy 2.2.6 and scipy 1.15.2 on Python 3.10.22).
Without it the SRMRpy rows say they were skipped.

The toolbox (github.com/MuSAELab/SRMRToolbox, head 690bdd1 of 2023-05-23)
keeps a test recording, test/test.wav (0.68 s of speech at 16 kHz), and in
test/test.mat the four ratios its own test script, test/test_SRMR.m,
expects SRMR.m to give for it: correct_ratios, in the order full, fast,
fast with norm, norm. test.mat was last changed in commit 8be3f23 of
2014-11-07, and the four are written out below to every digit it holds.
No audio comes with kudio, so the recording is read only when DIR, the
test folder of a checkout of the toolbox, is given; without it those rows
say they were skipped.

SRMRpy (github.com/jfsantos/SRMRpy) is the Python port by J. F. Santos,
first author of the norm variant. Its srmr() runs the fast mode unless told
fast=False, and it differs from the toolbox, and so from kudio, in six
places:

* the norm variant's bands. SRMR.m moves the top modulation band from
  128 Hz to 30 Hz when it normalises; SRMRpy keeps 128 Hz unless it is
  given max_cf=30. Called with norm=True alone, it gives 3.68 for the
  toolbox's test recording where the toolbox stores 2.30, and parts from
  kudio by as much as 78% on the clips below. Everywhere else here
  SRMRpy's norm is called with max_cf=30;
* where the 256 ms windows start. SRMR.m frames each modulation band with
  MATLAB's buffer(x, 256 ms, 192 ms), which puts the first window 192 ms
  before the first sample, padded with zeros, and keeps going until a
  window has passed the last sample: ceil(N / 64 ms) windows. SRMRpy starts
  at the first sample and keeps only whole windows, 1 + (N - 256 ms) //
  64 ms of them. On 3 s at 16 kHz that is 47 windows against 43: the four
  SRMRpy does without are the ones hanging over the ends of the clip,
  three at the start and one at the end;
* the window: MATLAB's hamming is symmetric, SRMRpy takes the periodic one;
* the Hilbert transform: SRMRpy pads its FFT to a multiple of 16 samples;
* the fast mode's short-time spectrum: SRMRpy takes it from the Gammatone
  package's fft_gtgram, whose frames are as long as the FFT and whose
  window sits in their middle, where SRMR.m takes MATLAB's specgram on the
  10 ms window itself;
* the silence cut: SRMRpy scores the clip as it is, as kudio does with
  drop_silence=False.

Every signal is made here from fixed seeds: the voiced syllables of
tests/test_srmr.py (pitch 220 to 300 Hz, three formant-like peaks, pauses
at digital zero), on a noise floor 40 dB under the peak, dry, in rooms from
kudio.rir with RT60 of 0.3, 0.6, 1.0 and 1.5 s, under white noise at 10 dB,
and low-passed at 1 kHz, at 8 and 16 kHz.

What it checks:

1. With DIR, the four stored ratios against kudio, on the recording as
   int16 and as float; that DIR's test.mat holds the numbers written out
   here; SRMRpy's numbers for the same file; and what a second of digital
   silence either side does when the cut is turned off.
2. kudio against SRMRpy as each stands, in all four modes.
3. kudio with SRMRpy's ways put in place of its own, through the module's
   private functions, one more at a time, until the two agree to rounding:
   that is what shows the six places above are the whole difference, and
   how much of it the window start is. In the full mode the window start
   is nearly all of it: 1.7% at 8 kHz and 4.0% at 16 kHz at most, down to
   3e-5 once SRMRpy's windows are in, and the padded Hilbert transform is
   the rest. In the fast mode most of it is the short-time spectrum.

It prints the largest differences and exits with 1 if one is past its
tolerance. The run takes under a minute.

References:

* T. H. Falk, C. Zheng and W.-Y. Chan, "A non-intrusive quality and
  intelligibility measure of reverberant and dereverberated speech", IEEE
  Trans. Audio, Speech, Lang. Process., vol. 18, no. 7, pp. 1766-1774,
  2010, doi:10.1109/TASL.2010.2052247.
* J. F. Santos, M. Senoussaoui and T. H. Falk, "An improved non-intrusive
  intelligibility metric for noisy and reverberant speech", Proc. 14th
  Int. Workshop on Acoustic Signal Enhancement (IWAENC), 2014, pp. 55-59,
  doi:10.1109/IWAENC.2014.6953337. The norm variant.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib
import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import kudio  # noqa: E402

# kudio.core may import srmr() under the module's own name, which hides the
# module behind the function
module = importlib.import_module('kudio.core.srmr')
srmr = module.srmr

try:
    import srmrpy
    from gammatone.fftweight import fft_gtgram
except ImportError:
    srmrpy = None

#: correct_ratios of SRMRToolbox's test/test.mat: full, fast, fast with norm, norm
STORED = (6.116783821833282, 5.941239326243227, 3.525434591882688, 2.300403317120358)
MODES = (('full', {}), ('fast', {'fast': True}), ('fast, norm', {'fast': True, 'norm': True}),
         ('norm', {'norm': True}))

#: (check, rate, cases, largest absolute difference, largest relative
#: difference, tolerance on the relative one)
RESULTS = []


def report(check, rate, cases, diff, rel, tolerance):
    RESULTS.append((check, rate, cases, diff, rel, tolerance))


def largest(got, want):
    got, want = np.ravel(got).astype(float), np.ravel(want).astype(float)
    diff = np.abs(got - want)
    return float(np.max(diff)), float(np.max(diff / np.abs(want)))


# --- signals, the voice of tests/test_srmr.py -------------------------------------

def voice(sr, seconds=3.0, seed=0):
    """Voiced syllables of 120-280 ms with pauses of 40-200 ms between them,
    each a harmonic complex whose pitch starts between 220 and 300 Hz and
    glides by up to 15%, with random harmonic phases, three formant-like
    peaks and a Hann envelope. The pauses are digital silence."""
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


def clips(sr):
    """``(name, clip)`` at *sr*, 3 s and 7 samples, so that the length is
    not a multiple of 16 and SRMRpy's padded Hilbert transform shows."""
    n = 3 * sr + 7
    dry = np.concatenate([voice(sr, 3.0), np.zeros(7)])
    dry = dry + 0.01 * np.random.default_rng(1).standard_normal(n)
    out = [('dry', dry)]
    for rt60 in (0.3, 0.6, 1.0, 1.5):
        wet = kudio.apply_rir(dry, kudio.rir(sr, rt60=rt60, seed=0))
        out.append((f'room {rt60} s', np.asarray(wet, float)[:n]))
    hiss = np.random.default_rng(2).standard_normal(n)
    out.append(('white 10 dB', dry + hiss * np.sqrt(np.mean(dry ** 2) / 10.0)))
    out.append(('low-passed 1 kHz', sosfiltfilt(butter(8, 1000, fs=sr, output='sos'), dry)))
    return out


# --- SRMRpy's ways, written here from its code ------------------------------------

def windows_from_the_start(x, length, hop, weights):
    """Window energies laid out as SRMRpy lays them out: the first window on
    the first sample, and only the 1 + (N - length) // hop that fit."""
    count = 1 + (len(x) - length) // hop
    starts = np.arange(count) * hop
    frames = (x * x)[starts[:, None] + np.arange(length)]
    return frames @ weights


def periodic_hamming_windows(rate):
    """SRMR.m's lengths, with the periodic Hamming SRMRpy takes."""
    length = int(math.ceil(0.256 * rate))
    hop = int(math.ceil(0.064 * rate))
    return length, hop, np.hamming(length + 1)[:-1] ** 2


def hilbert_padded_to_16(x):
    """|analytic signal| with the FFT padded to a multiple of 16 samples and
    the result cut back to len(x), as SRMRpy's hilbert does."""
    n = len(x)
    size = -(-n // 16) * 16
    spectrum = np.fft.fft(x, size)
    gain = np.zeros(size)
    gain[0] = 1.0
    gain[1:(size + 1) // 2] = 2.0
    if size % 2 == 0:
        gain[size // 2] = 1.0
    return np.abs(np.fft.ifft(spectrum * gain)[:n])


def gammatone_package_envelopes(y, sr, sos):
    """The fast mode's envelopes from the Gammatone package's fft_gtgram,
    with SRMRpy's arguments."""
    return fft_gtgram(y, sr, 0.010, 0.0025, 23, 125)


#: the swaps, in the order they are put in
SWAPS = (('window start', '_framed_energy', windows_from_the_start),
         ('periodic Hamming', '_modulation_windows', periodic_hamming_windows),
         ('Hilbert padded to 16', '_hilbert_envelope', hilbert_padded_to_16),
         ("fft_gtgram's spectrum", '_spectral_envelopes', gammatone_package_envelopes))


@contextlib.contextmanager
def swapped(count):
    """The first *count* of SWAPS in place of kudio's own functions."""
    kept = {name: getattr(module, name) for _, name, _ in SWAPS}
    try:
        for _, name, function in SWAPS[:count]:
            setattr(module, name, function)
        yield
    finally:
        for name, function in kept.items():
            setattr(module, name, function)


def srmrpy_ratio(y, sr, fast=False, norm=False, as_toolbox=True):
    """SRMRpy's ratio. With *as_toolbox*, norm comes with max_cf=30, the top
    band SRMR.m moves to for it; without, with SRMRpy's own default, 128."""
    top = 30.0 if norm and as_toolbox else 128.0
    return float(srmrpy.srmr(y, sr, fast=fast, norm=norm, max_cf=top)[0])


# --- 1: the toolbox's test recording ----------------------------------------------

def check_toolbox(folder):
    if folder is None:
        report('the four stored ratios: skipped, no --toolbox-test DIR', '-', 0,
               0.0, 0.0, math.inf)
        return
    from scipy.io import loadmat, wavfile
    folder = Path(folder)
    sr, pcm = wavfile.read(folder / 'test.wav')
    print(f"\n{folder / 'test.wav'}: {pcm.dtype}, {len(pcm)} samples at {sr} Hz "
          f"({len(pcm) / sr:.3f} s)")
    held = loadmat(str(folder / 'test.mat'))['correct_ratios'].ravel()
    diff, rel = largest(held, STORED)
    report("the four ratios in DIR's test.mat vs the ones written out here", sr, 4,
           diff, rel, 0.0)
    got = []
    for (name, kwargs), want in zip(MODES, STORED):
        as_int, as_float = srmr(pcm, sr, **kwargs), srmr(pcm / 32768.0, sr, **kwargs)
        got += [as_int, as_float]
        print(f"  {name:<11} stored {want:.10f}  kudio {as_int:.10f} (int16), "
              f"{as_float:.10f} (float)  {abs(as_int - want) / want:.1e} apart")
    diff, rel = largest(got, np.repeat(STORED, 2))
    report('kudio vs the four stored ratios (said: within 5e-13)', sr, 8, diff, rel, 5e-13)

    pad = np.zeros(sr, dtype=pcm.dtype)
    moves = []
    for (name, kwargs), want in zip(MODES, STORED):
        kept = srmr(np.concatenate([pad, pcm, pad]), sr, drop_silence=False, **kwargs)
        moves.append(abs(kept / want - 1.0))
        print(f"  {name:<11} a second of zeros either side, kept: {kept:.4f}, "
              f"{100 * (kept / want - 1):+.1f}%")
    print(f"  (said: 1 to 4%, depending on the mode; got {100 * min(moves):.1f} to "
          f"{100 * max(moves):.1f}%)")
    report('a second of zeros either side, kept: smallest move, said 1%', sr, 4,
           abs(100 * min(moves) - 1), math.nan, 0.5)
    report('the same, largest move, said 4%', sr, 4, abs(100 * max(moves) - 4), math.nan, 0.5)

    if srmrpy is None:
        report("SRMRpy on the test recording: skipped, no srmrpy", sr, 0, 0.0, 0.0, math.inf)
        return
    y = pcm / 32768.0
    theirs = [srmrpy_ratio(y, sr, **kwargs) for _, kwargs in MODES]
    own_top = [srmrpy_ratio(y, sr, as_toolbox=False, **kwargs) for _, kwargs in MODES]
    for (name, kwargs), want, value, plain in zip(MODES, STORED, theirs, own_top):
        extra = (f"; with its own max_cf of 128, {plain:.4f}"
                 if kwargs.get('norm') else '')
        print(f"  SRMRpy {name:<11} {value:.4f}, where the toolbox stores {want:.4f}{extra}")
    report('SRMRpy full on the test recording: said 6.055', sr, 1, abs(theirs[0] - 6.055),
           math.nan, 0.0005)
    report('SRMRpy fast on the test recording: said 6.737', sr, 1, abs(theirs[1] - 6.737),
           math.nan, 0.0005)
    with swapped(len(SWAPS)):
        ours = [srmr(y, sr, drop_silence=False, **kwargs) for _, kwargs in MODES]
    diff, rel = largest(ours, theirs)
    report("kudio with SRMRpy's ways vs SRMRpy, test recording", sr, 4, diff, rel, 1e-9)


# --- 2 and 3: SRMRpy ---------------------------------------------------------------

def check_srmrpy():
    if srmrpy is None:
        report('SRMRpy: skipped, no srmrpy', '-', 0, 0.0, 0.0, math.inf)
        return
    print("\nThe windows on a clip of 3 s at 16 kHz: SRMR.m's buffer() gives "
          f"{-(-48000 // 1024)}, the first starting {4096 - 1024} samples early; "
          f"SRMRpy gives {1 + (48000 - 4096) // 1024}, from the first sample")
    print("\nkudio against SRMRpy, with SRMRpy's ways put in one more at a time "
          "(largest relative difference over the clips):")
    header = (['as each stands', 'max_cf 30 for norm', 'cut off']
              + [f'+ {label}' for label, _, _ in SWAPS])
    print(f"  {'rate':>5} {'mode':<11} " + ' '.join(f'{h:>21}' for h in header))
    for sr in (8000, 16000):
        cases = clips(sr)
        for name, kwargs in MODES:
            as_they_are = [srmr(y, sr, **kwargs) for _, y in cases]
            plain = [srmrpy_ratio(y, sr, as_toolbox=False, **kwargs) for _, y in cases]
            theirs = [srmrpy_ratio(y, sr, **kwargs) for _, y in cases]
            row = [largest(as_they_are, plain)[1], largest(as_they_are, theirs)[1]]
            for count in range(len(SWAPS) + 1):
                with swapped(count):
                    ours = [srmr(y, sr, drop_silence=False, **kwargs) for _, y in cases]
                row.append(largest(ours, theirs)[1])
            print(f"  {sr:>5} {name:<11} " + ' '.join(f'{v:>21.2e}' for v in row))
            report(f'kudio vs SRMRpy as each stands, {name}', sr, len(cases), math.nan,
                   row[0], math.inf)
            report(f"kudio with SRMRpy's ways vs SRMRpy, {name}", sr, len(cases), math.nan,
                   row[-1], 1e-9)
    print("  The fast modes do not use the Hilbert transform, nor the full ones the "
          "short-time spectrum, so those columns repeat the one before; max_cf only "
          "moves the norm modes.")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--toolbox-test', metavar='DIR',
                        help="the test folder of SRMRToolbox, holding test.wav and test.mat")
    args = parser.parse_args()
    start = time.perf_counter()
    import scipy
    versions = f"numpy {np.__version__}, scipy {scipy.__version__}, Python {sys.version.split()[0]}"
    if srmrpy is not None:
        from importlib.metadata import version
        versions = f"SRMRpy {version('srmrpy')}, Gammatone {version('gammatone')}; " + versions
    print(f"kudio {kudio.__version__} from {Path(kudio.__file__).parent}; {versions}")
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        check_toolbox(args.toolbox_test)
        check_srmrpy()
    failed = print_table()
    print(f"\n{time.perf_counter() - start:.0f} s; "
          + ("SOME DIFFERENCES ARE PAST TOLERANCE" if failed
             else "every difference within its tolerance"))
    return 1 if failed else 0


def print_table():
    """The table of every check; True if one is past its limit. The limit
    holds the relative difference where there is one, else the absolute."""
    width = max(len(r[0]) for r in RESULTS)
    print(f"\n{'check':<{width}} {'rate':>5} {'cases':>5} {'max |diff|':>11} "
          f"{'max rel':>9} {'limit':>7}")
    failed = False
    for check, rate, cases, diff, rel, tolerance in RESULTS:
        over = (diff if math.isnan(rel) else rel) > tolerance
        failed |= over
        limit = '-' if math.isinf(tolerance) else f'{tolerance:.1e}'.replace('.0e', 'e')
        shown = ['-' if math.isnan(v) else f'{v:.2e}' for v in (diff, rel)]
        print(f"{check:<{width}} {rate!s:>5} {cases:>5} {shown[0]:>11} {shown[1]:>9} "
              f"{limit:>7}{'  PAST' if over else ''}")
    return failed


if __name__ == '__main__':
    sys.exit(main())
