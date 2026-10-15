# -*- coding: utf-8 -*-
"""Figure for the denoising page: the page's noisy clip before and after
log-MMSE, with the SI-SDR of each in the panel titles.

The clip is the one the page builds in its first code block: two harmonic
voices with pauses around them, white noise mixed in at 5 dB SNR, seeded.
"""
from __future__ import annotations

import numpy as np

import figstyle
import kudio

SR = 16000
#: framing of the picture only; at 256 the WebP stays near 120 kB
N_FFT, HOP = 512, 256
#: shown range; the harmonics stop at 2.2 kHz and the noise is flat above
TOP_KHZ = 4.0
FLOOR_DB = 70.0


def clip():
    """``(clean, noisy)``, built exactly as the page builds them."""
    t = np.arange(SR * 5) / SR
    rng = np.random.default_rng(0)

    def voice(start, stop, f0):
        """Thirteen harmonics of f0, rising and falling 3.5 times a second."""
        span = (t >= start) & (t < stop)
        tone = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 14))
        swell = 0.5 + 0.5 * np.sin(2 * np.pi * 3.5 * t)
        return np.where(span, tone * swell / 7, 0.0)

    clean = (voice(0.5, 2.0, 140) + voice(2.8, 4.5, 170)).astype(np.float32)
    noisy = kudio.add_noise_snr(clean, rng.standard_normal(len(clean)),
                                snr_db=5, seed=0)
    return clean, noisy


def draw(fig, p) -> None:
    clean, noisy = clip()
    enhanced = kudio.spectral_enhance(noisy, SR)

    stft = kudio.STFT(sr=SR, n_fft=N_FFT, hop_length=HOP)
    db = [10.0 * np.log10(np.maximum(np.abs(stft.analyse(y)) ** 2, 1e-12))
          for y in (noisy, enhanced)]
    peak = db[0].max()              # one reference, so the panels compare
    seconds = len(noisy) / SR
    keep = round(TOP_KHZ * 1000 / (SR / N_FFT)) + 1

    axes = fig.subplots(1, 2, sharex=True, sharey=True)
    titles = (f"Noisy: SI-SDR {kudio.si_sdr(clean, noisy):.2f} dB",
              f"log-MMSE + MCRA: SI-SDR {kudio.si_sdr(clean, enhanced):.2f} dB")
    for ax, level, title in zip(axes, db, titles):
        image = ax.imshow(np.maximum(level[:, :keep] - peak, -FLOOR_DB).T,
                          origin="lower", aspect="auto", interpolation="antialiased",
                          extent=(0, seconds, 0, (keep - 1) * SR / N_FFT / 1000),
                          vmin=-FLOOR_DB, vmax=0, cmap=p["cmap"])
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.grid(False)
    axes[0].set_ylabel("Frequency (kHz)")

    bar = fig.colorbar(image, ax=list(axes), shrink=0.9, pad=0.02)
    bar.set_label("Level (dB re. noisy peak)")
    bar.outline.set_visible(False)


def main() -> None:
    figstyle.save("enhancement-before-after", draw, size=(8.0, 3.0))


if __name__ == "__main__":
    main()
