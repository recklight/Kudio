# -*- coding: utf-8 -*-
"""Figures for guide/degradation.md: how T30 is read off a decay curve, and
what burst loss does to the same number of lost packets."""
from __future__ import annotations

import numpy as np

import figstyle
import kudio

SR = 16000


def _fit(curve: np.ndarray, top: float, bottom: float):
    """The straight line rt60() fits between *top* and *bottom* dB, as
    (start, stop, slope, intercept) with times in seconds."""
    start = int(np.argmax(curve <= top))
    stop = int(np.argmax(curve <= bottom))
    x = np.arange(start, stop) / SR
    slope, intercept = np.polyfit(x, curve[start:stop], 1)
    return start, stop, slope, intercept


def _decay_panel(ax, ir: np.ndarray, p, title: str) -> None:
    # the loudest sample in each millisecond: the raw trace at 16 kHz is a
    # solid block at this width, and its peaks are what a reader looks for
    per_ms = SR // 1000
    n = len(ir) // per_ms * per_ms
    peaks = np.max(np.abs(ir[:n]).reshape(-1, per_ms), axis=1)
    level = 20 * np.log10(peaks / np.max(np.abs(ir)) + 1e-12)
    t_ms = (np.arange(len(peaks)) * per_ms) / SR

    t = np.arange(len(ir)) / SR
    curve = kudio.schroeder_curve(ir)
    room = kudio.rt60(ir, SR)
    start, stop, slope, intercept = _fit(curve, -5.0, -35.0)

    ax.axhspan(-35, -5, color=p["c2"], alpha=0.10, linewidth=0,
               label="T30 range, -5 to -35 dB")
    ax.plot(t_ms, level, color=p["muted"], linewidth=0.7, alpha=0.8,
            label="impulse response, peak per ms")
    ax.plot(t, curve, color=p["c1"], linewidth=1.8, label="Schroeder decay")
    # the fitted line from where the fit starts, carried on until it has
    # fallen 60 dB: the time that takes is T30. Drawn under the fitted span,
    # so only the part past -35 dB shows.
    line_t = np.array([t[start], room.t30])
    ax.plot(line_t, slope * line_t + intercept, color=p["fg"], linewidth=1.0,
            linestyle="--", label="fit, carried on for 60 dB", zorder=3)
    ax.plot(t[start:stop], curve[start:stop], color=p["c2"], linewidth=2.6,
            label="fitted span", zorder=4)
    if room.t30 <= t[-1]:
        bottom = slope * room.t30 + intercept
        ax.plot([room.t30], [bottom], "o", color=p["fg"], markersize=4,
                zorder=5)
        ax.annotate(f"T30 = {room.t30:.2f} s", (room.t30, bottom),
                    xytext=(8, 2), textcoords="offset points",
                    color=p["fg"], fontsize=10, va="bottom")

    ax.set_xlim(0, t[-1])
    ax.set_ylim(-80, 3)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Level (dB)")
    ax.set_title(title, loc="left")
    ax.text(0.98, 0.95, f"T30 {room.t30:.2f} s, fit {room.fit:.3f}\n"
                        f"reliable(): {room.reliable()}",
            transform=ax.transAxes, ha="right", va="top", color=p["fg"],
            fontsize=10)


def draw_schroeder(fig, p) -> None:
    clean = kudio.rir(SR, rt60=0.6, seed=0)
    # the same room over white noise 50 dB below the direct sound, cut at
    # 0.45 s, about twice as late as the point where the tail meets the floor
    rng = np.random.default_rng(1)
    long = kudio.rir(SR, rt60=0.6, seed=0, seconds=2.0).astype(np.float64)
    floored = (long + 10 ** (-50 / 20) * rng.standard_normal(long.size))
    floored = floored[:int(0.45 * SR)]

    left, right = fig.subplots(1, 2, sharey=True)
    _decay_panel(left, clean, p, "rir(16000, rt60=0.6, seed=0)")
    _decay_panel(right, floored, p, "Same room, noise floor 50 dB down")
    right.set_ylabel("")
    # five entries do not fit in either panel without covering a trace
    handles, labels = left.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=3,
               fontsize=9)


def _lost_packets(y: np.ndarray, burst) -> np.ndarray:
    _, spans = kudio.dropouts(y, SR, loss=0.05, burst=burst, seed=0,
                              return_spans=True)
    return np.array([start for start, _ in spans])


def draw_loss(fig, p) -> None:
    seconds = 30.0
    y = np.zeros(int(seconds * SR), dtype=np.float32)
    rows = [(1, "burst=None (independent)", None, p["c1"]),
            (0, "burst=6", 6, p["c2"])]
    ax = fig.subplots()
    for row, label, burst, colour in rows:
        starts = _lost_packets(y, burst)
        ax.broken_barh([(s, 0.02) for s in starts], (row - 0.3, 0.6),
                       facecolors=colour)
        # a run is a stretch of lost packets with no survivor between them
        breaks = np.flatnonzero(np.diff(starts) > 0.021)
        lengths = np.diff(np.concatenate(([-1], breaks, [len(starts) - 1])))
        ax.text(0.0, row + 0.36,
                f"{label}: {len(starts)} packets lost, "
                f"longest gap {int(lengths.max()) * 20} ms",
                color=p["fg"], fontsize=10, va="bottom")
    ax.set_xlim(0, seconds)
    ax.set_ylim(-0.4, 1.85)
    ax.set_yticks([])
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Time (s), 20 ms packets, loss=0.05")


def main() -> None:
    figstyle.save("degradation-schroeder", draw_schroeder, size=(8.0, 3.9))
    figstyle.save("degradation-loss", draw_loss, size=(8.0, 1.9))


if __name__ == "__main__":
    main()
