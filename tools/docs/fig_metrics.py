# -*- coding: utf-8 -*-
"""Figure for docs/guide/metrics.md: what a few samples of delay do to SI-SDR.

The clip is the one the page builds in its first example (two harmonic
phrases over room tone, seeded), so the curve and the numbers in the text
come from the same signal.
"""
from __future__ import annotations

import numpy as np

import figstyle
import kudio

SR = 16000
DELAYS = np.arange(0, 17)            # 0 to 1 ms at 16 kHz


def _clip():
    rng = np.random.default_rng(0)

    def burst(seconds, f0):
        t = np.arange(int(seconds * SR)) / SR
        tone = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 12))
        return tone * (0.5 + 0.5 * np.sin(2 * np.pi * 4 * t)) / 6

    clean = 0.01 * rng.standard_normal(4 * SR)
    clean[8000:24000] += burst(1.0, 120)
    clean[32000:52800] += burst(1.3, 150)
    clean = clean.astype(np.float32)
    noise = rng.standard_normal(4 * SR).astype(np.float32)
    noisy = kudio.add_noise_snr(clean, noise, snr_db=5, seed=0)
    return clean, noisy, kudio.spectral_enhance(noisy, SR)


def _late(y, n):
    """*y* starting *n* samples later, cut back to its own length, so the
    length check has nothing to see."""
    return np.concatenate([np.zeros(n, dtype=y.dtype), y])[:len(y)]


def main() -> None:
    clean, noisy, enhanced = _clip()
    copy = np.array([kudio.si_sdr(clean, _late(clean, n)) for n in DELAYS])
    denoised = np.array([kudio.si_sdr(clean, _late(enhanced, n)) for n in DELAYS])
    baseline = kudio.si_sdr(clean, noisy)

    def draw(fig, p):
        ax = fig.add_subplot()
        ax.axhline(baseline, color=p["muted"], linestyle="--", linewidth=1.2)
        ax.text(16, baseline + 0.6, f"noisy input, not processed: {baseline:+.1f} dB",
                color=p["muted"], ha="right", va="bottom", fontsize=9)
        ax.plot(DELAYS[1:], copy[1:], color=p["c1"], marker="o", markersize=4,
                linewidth=1.8, label="perfect copy, n samples late")
        ax.plot(DELAYS, denoised, color=p["c2"], marker="s", markersize=3.5,
                linewidth=1.8, label="log-MMSE output, n samples late")
        # the copy at zero delay is far off the scale: mark where it went
        ax.plot([0], [15.6], color=p["c1"], marker="^", markersize=7,
                linestyle="none", clip_on=False)
        ax.text(0.4, 15.6, f"perfect copy at 0 samples: {copy[0]:+.1f} dB, "
                f"off the scale", color=p["c1"], fontsize=9, va="center")
        ax.set_xlim(-0.3, 16.3)
        ax.set_ylim(-12, 17)
        ax.set_xticks(range(0, 17, 2))
        ax.set_xlabel("Delay (samples at 16 kHz)")
        ax.set_ylabel("SI-SDR (dB)")
        top = ax.secondary_xaxis("top", functions=(lambda s: s / SR * 1000,
                                                   lambda ms: ms * SR / 1000))
        top.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        top.set_xlabel("Delay (ms)")
        ax.legend(loc="lower left")

    figstyle.save("metrics-delay", draw, size=(8.0, 3.6))


if __name__ == "__main__":
    main()
