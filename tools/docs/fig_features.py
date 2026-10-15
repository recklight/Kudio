# -*- coding: utf-8 -*-
"""Figure for the STFT and features page: one clip on a linear and a mel axis.

The clip is synthetic and seeded: two voiced syllables with a gliding pitch
and a hiss between them, over a quiet noise floor.
"""
from __future__ import annotations

import librosa
import numpy as np

import figstyle
import kudio

SR = 16000
#: frame geometry shared by both panels, so their time axes agree
N_FFT, HOP = 512, 256
N_MELS = 64
FLOOR_DB = 70.0


def _syllable(seconds: float, f0_from: float, f0_to: float) -> np.ndarray:
    """Harmonics of a gliding f0 up to 6 kHz, under one half-sine swell."""
    n = int(seconds * SR)
    f0 = np.linspace(f0_from, f0_to, n)
    phase = 2 * np.pi * np.cumsum(f0) / SR
    out = np.zeros(n)
    for k in range(1, 50):
        audible = k * f0 < 6000.0
        out += audible * np.sin(k * phase) / k
    envelope = np.sin(np.pi * np.arange(n) / n) ** 0.5
    return (0.25 * out * envelope).astype(np.float32)


def clip() -> np.ndarray:
    rng = np.random.default_rng(7)
    y = np.zeros(2 * SR, np.float32)
    y[int(0.15 * SR):int(0.85 * SR)] += _syllable(0.70, 120.0, 180.0)
    hiss = kudio.highpass(rng.standard_normal(int(0.25 * SR)).astype(np.float32),
                          SR, cutoff=3500.0)
    y[int(0.95 * SR):int(1.20 * SR)] += 0.15 * kudio.fade(hiss, SR, 0.03, 0.03)
    y[int(1.30 * SR):int(1.85 * SR)] += _syllable(0.55, 210.0, 140.0)
    floor = rng.standard_normal(len(y)).astype(np.float32)
    return kudio.add_noise_snr(y, floor, snr_db=50.0, seed=0)


def _relative(db: np.ndarray) -> np.ndarray:
    return np.maximum(db - db.max(), -FLOOR_DB)


def draw(fig, p) -> None:
    y = clip()
    stft = kudio.STFT(sr=SR, n_fft=N_FFT, hop_length=HOP)
    linear = _relative(10.0 * stft.forward(y))              # log10 power -> dB
    mel = _relative(kudio.melspectrogram(y, sr=SR, n_mels=N_MELS, n_frames=1,
                                         n_fft=N_FFT, hop_length=HOP))
    seconds = len(y) / SR

    left, right = fig.subplots(1, 2, sharex=True)
    kwargs = dict(origin="lower", aspect="auto", vmin=-FLOOR_DB, vmax=0,
                  interpolation="nearest", cmap=p["cmap"])
    left.imshow(linear.T, extent=(0, seconds, 0, SR / 2000), **kwargs)
    image = right.imshow(mel.T, extent=(0, seconds, -0.5, N_MELS - 0.5), **kwargs)

    left.set_title(f"waveform_to_spectrogram: {stft.n_bins} linear bins")
    left.set_ylabel("Frequency (kHz)")
    right.set_title(f"melspectrogram(n_frames=1): {N_MELS} mel bands")
    right.set_ylabel("Frequency (kHz, mel-spaced)")

    # label mel bands by the frequency at their centre, in kHz like the left
    centres = librosa.mel_frequencies(N_MELS + 2, fmax=SR / 2)[1:-1] / 1000
    marks = [0.25, 0.5, 1, 2, 4, 8]
    right.set_yticks([np.interp(f, centres, np.arange(N_MELS)) for f in marks[:-1]]
                     + [N_MELS - 0.5])
    right.set_yticklabels([f"{f:g}" for f in marks])

    # how much of each picture lies below 1 kHz
    left.axhline(1.0, color="#ffffff", lw=0.8, ls="--", alpha=0.75)
    right.axhline(np.interp(1.0, centres, np.arange(N_MELS)), color="#ffffff",
                  lw=0.8, ls="--", alpha=0.75)

    for ax in (left, right):
        ax.set_xlabel("Time (s)")
        ax.grid(False)
    bar = fig.colorbar(image, ax=[left, right], shrink=0.9, pad=0.02)
    bar.set_label("Level (dB re. peak)")
    bar.outline.set_visible(False)


def main() -> None:
    figstyle.save("features-linear-mel", draw, size=(8.0, 3.3))


if __name__ == "__main__":
    main()
