# -*- coding: utf-8 -*-
"""Figure for the editing page: one span deleted from a voiced tone, joined
twice, once butted together and once through kudio.splice.

The tone is synthetic: a 150 Hz fundamental with harmonics up to 3 kHz,
falling 12 dB per octave. The cut starts near a crest and the span removed is
6.5 periods long, so the two sides meet half a period out of step, which is
about as bad as an edit gets.
"""
from __future__ import annotations

import numpy as np

import figstyle
import kudio

SR = 16000
F0 = 150.0
#: where the deleted span starts and how long it is, in samples
CUT, GAP = 6427, round(6.5 * SR / F0)
#: the window each row shows around the seam, in milliseconds
ZOOM_MS, WIDE_MS = 12.0, 150.0
N_FFT, HOP = 256, 16
FLOOR_DB = 90.0


def tone(seconds: float = 1.0) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    out = np.zeros_like(t)
    for k in range(1, int(3000 // F0) + 1):
        out += np.sin(2 * np.pi * F0 * k * t) / k ** 2
    return (0.3 * out / np.max(np.abs(out))).astype(np.float32)


def joins():
    y = tone()
    before, after = y[:CUT], y[CUT + GAP:]
    hard = np.concatenate([before, after])
    spliced = kudio.splice([before, after], SR)          # 5 ms, linear
    return hard, spliced


def _spectrogram(y: np.ndarray) -> np.ndarray:
    stft = kudio.STFT(sr=SR, n_fft=N_FFT, hop_length=HOP, window="hann")
    db = 10.0 * stft.forward(y)                          # log10 power -> dB
    return np.maximum(db - db.max(), -FLOOR_DB)


def draw(fig, p) -> None:
    hard, spliced = joins()
    overlap = round(kudio.DEFAULT_CROSSFADE * SR)
    # Both seams end at sample CUT: splice blends the 5 ms *before* it.
    seam = CUT
    axes = fig.subplots(2, 2, sharex="row", sharey="row")
    titles = ("Hard cut: np.concatenate",
              "kudio.splice: 5 ms linear crossfade")

    for col, (y, title) in enumerate(zip((hard, spliced), titles)):
        top, bottom = axes[0, col], axes[1, col]

        n = int(ZOOM_MS / 1000 * SR)
        idx = np.arange(seam - n, seam + n)
        ms = (idx - seam) / SR * 1000
        top.plot(ms, y[idx], color=p["c1"], lw=1.2)
        if col == 0:
            # only above the curve: a full-height line would hide the
            # one-sample jump it is there to point at
            top.axvline(0, ymin=0.83, ymax=1.0, color=p["c4"], lw=1.2,
                        ls="--", label="seam")
        else:
            top.axvspan(-overlap / SR * 1000, 0, color=p["c2"], alpha=0.22,
                        lw=0, label="crossfade")
        # headroom above the tone's 0.3 peak keeps the legend off the curve
        top.set_ylim(-0.36, 0.46)
        top.legend(loc="upper left", fontsize=9)
        top.set_title(title)
        top.set_xlabel("Time from the seam (ms)")
        top.set_xlim(-ZOOM_MS, ZOOM_MS)

        spec = _spectrogram(y)
        frames = np.arange(spec.shape[0]) * HOP
        keep = np.abs(frames - seam) <= WIDE_MS / 1000 * SR
        extent = ((frames[keep][0] - seam) / SR * 1000,
                  (frames[keep][-1] - seam) / SR * 1000, 0, SR / 2000)
        image = bottom.imshow(spec[keep].T, origin="lower", aspect="auto",
                              extent=extent, vmin=-FLOOR_DB, vmax=0,
                              interpolation="nearest", cmap=p["cmap"])
        bottom.set_xlabel("Time from the seam (ms)")
        bottom.set_xticks(np.arange(-WIDE_MS, WIDE_MS + 1, 50))
        bottom.grid(False)

    axes[0, 0].set_ylabel("Amplitude")
    axes[1, 0].set_ylabel("Frequency (kHz)")
    bar = fig.colorbar(image, ax=axes[1, :], shrink=0.95, pad=0.02)
    bar.set_label("Level (dB re. peak)")
    bar.outline.set_visible(False)


def main() -> None:
    figstyle.save("editing-seam", draw, size=(8.0, 5.4))


if __name__ == "__main__":
    main()
