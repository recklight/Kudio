# -*- coding: utf-8 -*-
"""Figure for the recording and live views page: what SpectrogramStream.columns()
returns at three moments of one stream.

The stream is synthetic and seeded: four voiced syllables with gliding pitch
over a quiet noise floor, pushed in blocks of 1024 samples, which is what
StreamRecorder hands over per chunk by default.
"""
from __future__ import annotations

import numpy as np

import figstyle
import kudio

SR = 16000
BLOCK = 1024
#: when each snapshot is taken, in seconds of audio pushed so far
SNAPSHOTS = (1.0, 2.5, 5.0)
TOP_KHZ = 4.0


def _syllable(seconds: float, f0_from: float, f0_to: float) -> np.ndarray:
    """Harmonics of a gliding f0 up to 6 kHz under a half-sine envelope."""
    n = int(seconds * SR)
    f0 = np.linspace(f0_from, f0_to, n)
    phase = 2 * np.pi * np.cumsum(f0) / SR
    out = np.zeros(n)
    for k in range(1, 50):
        out += (k * f0 < 6000.0) * np.sin(k * phase) / k
    envelope = np.sin(np.pi * np.arange(n) / n) ** 0.5
    return (0.25 * out * envelope).astype(np.float32)


def stream() -> np.ndarray:
    rng = np.random.default_rng(11)
    y = np.zeros(5 * SR, np.float32)
    for start, length, f0_from, f0_to in ((0.30, 0.60, 120.0, 180.0),
                                          (1.40, 0.70, 210.0, 140.0),
                                          (2.60, 0.70, 150.0, 230.0),
                                          (3.90, 0.80, 180.0, 120.0)):
        a = int(start * SR)
        part = _syllable(length, f0_from, f0_to)
        y[a:a + len(part)] += part
    floor = rng.standard_normal(len(y)).astype(np.float32)
    return kudio.add_noise_snr(y, floor, snr_db=30.0, seed=0)


def snapshots():
    """(seconds, columns, filled) at each moment in SNAPSHOTS, and the view."""
    y = stream()
    view = kudio.SpectrogramStream(SR, seconds=3.0)
    taken, pushed = [], 0
    for when in SNAPSHOTS:
        until = int(when * SR)
        while pushed < until:
            view.push(y[pushed:min(pushed + BLOCK, until)])
            pushed = min(pushed + BLOCK, until)
        taken.append((when, view.columns(), view.filled))
    return taken, view


def draw(fig, p) -> None:
    taken, view = snapshots()
    axes = fig.subplots(1, len(taken), sharey=True)
    kwargs = dict(origin="lower", aspect="auto", interpolation="nearest",
                  vmin=view.floor_db, vmax=0.0, cmap=p["cmap"],
                  extent=(-view.window_seconds, 0.0, 0.0, SR / 2000))
    for ax, (when, columns, filled) in zip(axes, taken):
        image = ax.imshow(columns.T, **kwargs)
        ax.set_ylim(0, TOP_KHZ)
        ax.set_title(f"after {when:.1f} s\n{filled} of {view.n_columns} columns",
                     fontsize=10)
        ax.set_xlabel("Time before now (s)")
        ax.set_xticks([-3, -2, -1, 0])
        ax.grid(False)
    # the left of the first snapshot has not been heard yet
    empty = view.window_seconds * (1 - taken[0][2] / view.n_columns)
    axes[0].text(-view.window_seconds + empty / 2, TOP_KHZ / 2,
                 "not heard yet:\nfloor_db", ha="center", va="center",
                 color="#e6e6e6", fontsize=9)
    axes[0].set_ylabel("Frequency (kHz)")
    bar = fig.colorbar(image, ax=list(axes), shrink=0.9, pad=0.02)
    bar.set_label("Level (dBFS)")
    bar.outline.set_visible(False)


def main() -> None:
    figstyle.save("live-audio-filling", draw, size=(8.0, 3.0))


if __name__ == "__main__":
    main()
