# -*- coding: utf-8 -*-
"""Figures for the speech activity and pitch page.

Both draw the recording the page builds in its first example: three voiced
words, an /s/ and a broadband clatter, once in a quiet room and once in front
of a fan. The two files go through ``save_wave`` and ``file_load`` here as
they do on the page, so every span and every pitch value in the pictures is
the one the page prints.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import librosa
import numpy as np
from matplotlib import patheffects

import figstyle
import kudio

SR = 16000
SECONDS = 6
#: the frame geometry split_on_silence hands to librosa, so the level curve
#: is the one its top_db is compared against
FRAME, HOP = 2048, 512
TOP_DB = 30.0
#: a finer grid than the library default, so single harmonics show
N_FFT, SPEC_HOP = 2048, 160
FLOOR_DB = 80.0


def recording() -> dict:
    """The page's example, line for line: name -> waveform as read back."""
    n = SECONDS * SR
    rng = np.random.default_rng(7)

    def at(start, x):
        out = np.zeros(n)
        out[int(start * SR):int(start * SR) + len(x)] = x
        return out

    def vowel(seconds, f_from, f_to, level=1.0):
        m = int(seconds * SR)
        phase = 2 * np.pi * np.cumsum(np.linspace(f_from, f_to, m)) / SR
        tone = sum(np.sin(k * phase) / k for k in range(1, 16))
        return 0.08 * level * tone * np.sin(np.pi * np.arange(m) / m) ** 0.5

    def hiss(seconds, level, cutoff=None):
        m = int(seconds * SR)
        x = rng.standard_normal(m)
        if cutoff:
            x = kudio.highpass(x, SR, cutoff=cutoff)
        return level * x * np.hanning(m)

    voice = (at(0.8, vowel(0.8, 140, 110))
             + at(3.0, hiss(0.15, 0.05, cutoff=3500))
             + at(3.12, vowel(0.85, 120, 165))
             + at(4.6, vowel(0.8, 130, 100, level=0.5)))
    clatter = at(2.2, hiss(0.25, 0.25))
    room = 0.0005 * rng.standard_normal(n)
    fan = 0.03 * kudio.lowpass(rng.standard_normal(n), SR, cutoff=1000)

    # through a 16-bit file and back, exactly as the page does it
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        for name, wave in (("quiet", voice + clatter + room),
                           ("fan", voice + clatter + fan)):
            path = Path(tmp) / f"{name}.wav"
            kudio.save_wave(path, wave, SR)
            out[name], _ = kudio.file_load(path)
    return out


def _level_db(y: np.ndarray) -> np.ndarray:
    """Frame level relative to the loudest frame, as librosa's split sees it."""
    rms = librosa.feature.rms(y=y, frame_length=FRAME, hop_length=HOP)[0]
    return librosa.power_to_db(rms ** 2, ref=np.max, top_db=None)


def draw_level(fig, p) -> None:
    clips = recording()
    axes = fig.subplots(2, 1, sharex=True, sharey=True)
    for ax, (name, title) in zip(axes, (("quiet", "Quiet room"),
                                        ("fan", "Same voice in front of a fan"))):
        y = clips[name]
        db = _level_db(y)
        t = librosa.frames_to_time(np.arange(len(db)), sr=SR, hop_length=HOP)
        for n, (a, b) in enumerate(kudio.vad(y, SR)):
            ax.axvspan(a, b, color=p["c3"], alpha=0.22, lw=0,
                       label="vad() span" if n == 0 else None)
        ax.plot(t, db, color=p["c1"], lw=1.4, label="frame level")
        ax.axhline(-TOP_DB, color=p["c4"], lw=1.1, ls="--",
                   label=f"top_db = {TOP_DB:g}: quieter frames are silence")
        ax.set_title(title, loc="left")
        ax.set_ylabel("dB re loudest frame")
        ax.set_ylim(-72, 6)
        ax.set_xlim(0, SECONDS)
        ax.annotate("clatter", xy=(2.33, -1.5), xytext=(2.6, -12),
                    color=p["muted"], fontsize=9,
                    arrowprops=dict(arrowstyle="-", color=p["muted"], lw=0.8))
    axes[-1].set_xlabel("Time (s)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center",
               ncols=3, fontsize=9)


def draw_pitch(fig, p) -> None:
    y = recording()["fan"]
    spec = 10.0 * kudio.waveform_to_spectrogram(y, n_fft=N_FFT,
                                                hop_length=SPEC_HOP)
    spec = np.maximum(spec - spec.max(), -FLOOR_DB)          # dB re peak
    freqs = np.arange(spec.shape[1]) * SR / N_FFT
    times = np.arange(spec.shape[0]) * SPEC_HOP / SR
    track = kudio.f0(y, SR)

    ax = fig.subplots()
    keep = freqs >= 40
    mesh = ax.pcolormesh(times, freqs[keep], spec[:, keep].T, shading="nearest",
                         cmap=p["cmap"], vmin=-FLOOR_DB, vmax=0, rasterized=True)
    line, = ax.plot(track.times, track.f0, color="#4dd9ec", lw=1.5,
                    label="f0(): NaN where unvoiced")
    line.set_path_effects([patheffects.Stroke(linewidth=2.8, foreground="#101216"),
                           patheffects.Normal()])
    ax.set_yscale("log")
    ax.set_ylim(50, SR / 2)
    ticks = [100, 200, 500, 1000, 2000, 5000]
    ax.set_yticks(ticks)
    ax.set_yticklabels(["100", "200", "500", "1k", "2k", "5k"])
    ax.minorticks_off()
    ax.set_xlim(0, SECONDS)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.grid(False)
    for x, f, text in ((2.52, 4200, "clatter"), (3.2, 5000, "/s/")):
        ax.text(x, f, text, color="#ffffff", fontsize=9, va="center")
    ax.legend(loc="lower right", fontsize=9, labelcolor="#ffffff",
              facecolor="#101216", framealpha=0.6, frameon=True,
              edgecolor="none")
    bar = fig.colorbar(mesh, ax=ax, pad=0.015, aspect=25)
    bar.set_label("Level (dB re peak)")
    bar.outline.set_visible(False)


def main() -> None:
    figstyle.save("speech-level", draw_level, size=(8.0, 4.4))
    figstyle.save("speech-pitch", draw_pitch, size=(8.0, 3.4))


if __name__ == "__main__":
    main()
