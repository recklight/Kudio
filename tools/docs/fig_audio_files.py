# -*- coding: utf-8 -*-
"""Figure for the audio-files page: what a 16-bit file does to a sample.

Left, a 220 Hz tone peaking at 1.25, written three ways and read back:
``FLOAT`` and ``PCM_16`` through kudio.save_wave, and ``PCM_16`` through
libsndfile with its clipping switched off, which is what save_wave guards
against. Right, the rounding error of 20,000 uniform random samples written to
16 bits by soundfile.write and by kudio.save_wave.

soundfile switches clipping on whenever it opens a file, so the wrapped trace
turns it off again through soundfile's own libsndfile binding. That is not
public API; if a soundfile release moves it, the trace is left out and the
rest of the figure is still drawn.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

import figstyle
import kudio

SR = 16000
TONE_HZ, PEAK = 220.0, 1.25
#: how much of the tone the left panel shows, in milliseconds
SHOW_MS = 10.0
N_RANDOM, SEED = 20000, 0
STEP = 32768.0


def _without_clipping(path: Path, y: np.ndarray) -> bool:
    """Write *y* to 16-bit with libsndfile's clipping off; False if the
    installed soundfile keeps its binding somewhere else."""
    try:
        from soundfile import _ffi, _snd
        with sf.SoundFile(str(path), "w", SR, 1, subtype="PCM_16") as fh:
            _snd.sf_command(fh._file, _snd.SFC_SET_CLIPPING, _ffi.NULL,
                            _snd.SF_FALSE)
            fh.write(y)
    except (ImportError, AttributeError):
        return False
    return True


def full_scale(folder: Path):
    t = np.arange(int(SHOW_MS / 1000 * SR)) / SR
    y = (PEAK * np.sin(2 * np.pi * TONE_HZ * t)).astype(np.float32)
    kudio.save_wave(folder / "float.wav", y, SR, subtype="FLOAT")
    kudio.save_wave(folder / "pcm16.wav", y, SR)
    traces = {"FLOAT from save_wave": kudio.file_load(folder / "float.wav")[0],
              "PCM_16 from save_wave": kudio.file_load(folder / "pcm16.wav")[0]}
    if _without_clipping(folder / "wrapped.wav", y):
        traces["PCM_16, libsndfile clipping off"] = \
            kudio.file_load(folder / "wrapped.wav")[0]
    return t * 1000, traces


def rounding(folder: Path):
    x = np.random.default_rng(SEED).uniform(-0.99, 0.99, N_RANDOM)
    x = x.astype(np.float32)
    sf.write(str(folder / "direct.wav"), x, SR, subtype="PCM_16")
    kudio.save_wave(folder / "kudio.wav", x, SR)
    back = {name: sf.read(str(folder / f"{name}.wav"), dtype="float64")[0]
            for name in ("direct", "kudio")}
    return {"soundfile.write": (back["direct"] - x) * STEP,
            "kudio.save_wave": (back["kudio"] - x) * STEP}


def draw(fig, p) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        ms, traces = full_scale(Path(tmp))
        errors = rounding(Path(tmp))
    left, right = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1.15, 1]})

    # FLOAT drawn wide and faint underneath, so PCM_16 shows where they agree
    styles = {"FLOAT from save_wave": {"color": p["c1"], "lw": 3.2, "alpha": 0.35},
              "PCM_16 from save_wave": {"color": p["c2"], "lw": 1.5},
              "PCM_16, libsndfile clipping off": {"color": p["c4"], "lw": 1.0,
                                                  "ls": "--"}}
    for name, y in traces.items():
        left.plot(ms, y, label=name, **styles[name])
    for level in (-1.0, 1.0):
        left.axhline(level, color=p["muted"], lw=0.8, ls=":")
    left.set_xlim(0, SHOW_MS)
    left.set_ylim(-1.45, 2.6)
    left.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    left.set_xlabel("Time (ms)")
    left.set_ylabel("Amplitude (full scale = 1)")
    left.set_title(f"A {TONE_HZ:.0f} Hz tone peaking at {PEAK}, read back")
    left.legend(loc="upper left", fontsize=9)

    # Each histogram's bins span exactly the range its errors can take, so an
    # error of exactly 0 or +0.5 step lands in the last bin (closed on the
    # right) instead of a sliver past the end that looks like a larger error.
    # The edges are rounded because arange alone puts -0.5 at -0.4999999...
    ranges = ((-1.0, 0.0), (-0.5, 0.5))
    colours = (p["c4"], p["c3"])
    for (name, err), colour, (lo, hi) in zip(errors.items(), colours, ranges):
        if err.min() < lo or err.max() > hi:
            raise ValueError(f"{name}: errors outside [{lo}, {hi}] would be dropped")
        bins = np.round(np.arange(lo, hi + 0.025, 0.05), 2)
        mean = round(float(err.mean()), 3) + 0.0      # -0.0 + 0.0 is 0.0
        right.hist(err, bins=bins, histtype="stepfilled", alpha=0.3,
                   color=colour)
        right.hist(err, bins=bins, histtype="step", lw=1.4, color=colour,
                   label=f"{name}: mean {mean:.3f}")
    right.set_xlim(-1.1, 0.6)
    right.set_ylim(0, 1500)
    right.set_xlabel("Rounding error (16-bit steps)")
    right.set_ylabel("Samples per 0.05-step bin")
    right.set_title(f"{N_RANDOM:,} random samples to PCM_16")
    right.legend(loc="upper left", fontsize=9)


def main() -> None:
    figstyle.save("audio-files-full-scale", draw, size=(8.0, 3.6))


if __name__ == "__main__":
    main()
