# -*- coding: utf-8 -*-
"""Figure for the getting-started page: the frame levels behind the first
session's report, before and after denoising.

The clip is the one the page writes in its setup block (two harmonic voices
with pauses, white noise at 2 dB SNR, seeded) and goes through ``save_wave``
and ``file_load`` as it does there, so the levels are the ones
``audio_report`` measured. The frames are the report's own: 30 ms long at a
15 ms hop, and its SNR estimate is the 95th percentile of their levels minus
the 10th.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

import figstyle
import kudio

SR = 16000
FRAME, HOP = 0.030, 0.015
SPANS = ((0.5, 1.75), (2.5, 4.0))


def clip():
    """``(clean, noisy)`` as the page writes and reads them."""
    t = np.arange(int(4.5 * SR)) / SR

    def syllables(start, stop, f0):
        inside = (t >= start) & (t < stop)
        tone = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 12))
        envelope = 0.5 - 0.5 * np.cos(2 * np.pi * 4 * (t - start))
        return np.where(inside, tone * envelope / 5, 0.0)

    clean = (syllables(*SPANS[0], 120) + syllables(*SPANS[1], 150)).astype(np.float32)
    noise = np.random.default_rng(0).standard_normal(len(t)).astype(np.float32)
    noisy = kudio.add_noise_snr(clean, noise, snr_db=2, seed=0)
    with tempfile.TemporaryDirectory() as tmp:
        kudio.save_wave(Path(tmp) / "clean.wav", clean, SR)
        kudio.save_wave(Path(tmp) / "noisy.wav", noisy, SR)
        clean, _ = kudio.file_load(Path(tmp) / "clean.wav")
        noisy, _ = kudio.file_load(Path(tmp) / "noisy.wav")
    return clean, noisy


def frame_levels(y):
    """``(centres in s, level in dBFS)`` of the report's 30 ms frames."""
    size, hop = int(round(FRAME * SR)), int(round(HOP * SR))
    starts = np.arange(0, len(y) - size + 1, hop)
    rms = np.array([np.sqrt(np.mean(np.square(y[s:s + size], dtype=np.float64)))
                    for s in starts])
    return (starts + size / 2) / SR, 20 * np.log10(np.maximum(rms, 1e-10))


def draw(fig, p) -> None:
    _, noisy = clip()
    denoised = kudio.spectral_enhance(noisy, SR)
    report = kudio.audio_report(noisy, SR)

    ax = fig.subplots()
    bottom, top = -52.0, -4.0
    for a, b in SPANS:
        ax.axvspan(a, b, color=p["grid"], alpha=0.6, lw=0)
        ax.text((a + b) / 2, bottom + 1.5, "speech", ha="center", va="bottom",
                color=p["muted"], fontsize=10)

    t, before = frame_levels(noisy)
    _, after = frame_levels(denoised)
    ax.plot(t, before, color=p["c2"], lw=1.4, label="noisy.wav")
    ax.plot(t, after, color=p["c1"], lw=1.4, label="denoised.wav")

    # the report's two reference levels, both taken from the noisy file
    loud, floor = np.percentile(before, 95), np.percentile(before, 10)
    for level in (loud, floor):
        ax.axhline(level, color=p["muted"], lw=0.9, ls=(0, (4, 3)))
    ax.text(1.80, loud + 0.7, "95th percentile", color=p["muted"], fontsize=10,
            va="bottom")
    ax.text(1.80, floor - 1.4, "10th percentile", color=p["muted"], fontsize=10,
            va="top")

    x = 4.25
    ax.annotate("", xy=(x, loud), xytext=(x, floor),
                arrowprops=dict(arrowstyle="<->", color=p["fg"], lw=1.0,
                                shrinkA=0, shrinkB=0))
    ax.text(x + 0.2, top - 3.0,
            f"{report.estimated_snr_db:.1f} dB apart: the report's SNR estimate",
            ha="right", va="top", color=p["fg"], fontsize=10)

    ax.set_xlim(0, len(noisy) / SR)
    ax.set_ylim(bottom, top)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frame level (dBFS)")
    ax.legend(loc="upper left", ncol=2, borderaxespad=0.2)


def main() -> None:
    figstyle.save("start-levels", draw, size=(8.0, 3.2))


if __name__ == "__main__":
    main()
