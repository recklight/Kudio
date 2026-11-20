# -*- coding: utf-8 -*-
"""Figures for docs/guide/inspection.md.

Both are drawn from the synthetic take the page builds in its first example,
so the numbers printed on the page and the pictures beside them agree.
"""
from __future__ import annotations

import numpy as np

import figstyle
import kudio
from kudio.core import report

SR = 16000


def _take() -> np.ndarray:
    """The page's take: a 140 Hz voice with pauses over a -54 dBFS hiss."""
    rng = np.random.default_rng(0)
    t = np.arange(20 * SR) / SR
    voice = sum(np.cos(2 * np.pi * 140 * k * t) / k for k in range(1, 55))
    words = np.sin(2 * np.pi * 0.7 * t) > -0.3
    return 0.05 * voice * words + 0.002 * rng.standard_normal(t.size)


def _edges(y: np.ndarray):
    """The spectrum audio_report looks for an edge in, and its peak in dB."""
    edges = report._Edges(*report._welch_db(np.asarray(y, dtype=np.float64), SR), SR)
    return edges, float(edges.db.max())


def draw_bandwidth(fig, p) -> None:
    take = _take()
    narrow = kudio.resample(kudio.resample(take, SR, 8000), 8000, SR)
    fc = 4000.0                       # the Nyquist frequency of 8 kHz audio
    ax = fig.add_subplot()
    # the upsampled copy underneath and wider: below 3.8 kHz the two are the
    # same audio, and both lines should stay visible there
    for y, colour, width, name in [
            (narrow, p["c2"], 2.0, "8 kHz audio in a 16 kHz file"),
            (take, p["c1"], 0.8, "recorded at 16 kHz")]:
        edges, peak = _edges(y)
        depth = edges.cliff_and_depth(fc)[1]
        ax.plot(edges.freqs / 1000, edges.db - peak, color=colour, lw=width,
                label=f"{name}: depth {depth:.0f} dB at 4 kHz")

    # the two bands the depth compares, read off the smoothed spectrum
    edges, peak = _edges(narrow)
    smoothed = edges._smooth(fc) - peak
    below = edges._median(smoothed, 0.70 * fc, 0.88 * fc)
    above = edges._median(smoothed, 1.05 * fc, 0.97 * SR / 2)
    ax.hlines(below, 0.70 * fc / 1000, 0.88 * fc / 1000, color=p["c4"], lw=3.0, zorder=3)
    ax.hlines(above, 1.05 * fc / 1000, 0.97 * SR / 2 / 1000, color=p["c4"], lw=3.0,
              zorder=3)
    ax.annotate("", xy=(fc / 1000, above), xytext=(fc / 1000, below),
                arrowprops=dict(arrowstyle="<->", color=p["c4"], lw=1.2))
    ax.text(fc / 1000 + 0.12, (below + above) / 2,
            f"depth {below - above:.0f} dB\n(an edge needs {report._DEPTH_DB:.0f})",
            color=p["c4"], fontsize=9, va="center")
    ax.text(0.79 * fc / 1000, below - 5, "0.70-0.88\nof the edge", color=p["c4"],
            fontsize=9, ha="center", va="top")
    ax.text(0.97 * SR / 2 / 1000, above + 14, "1.05 of the edge to 0.97 of Nyquist",
            color=p["c4"], fontsize=9, ha="right", va="bottom")
    ax.set_xlim(0, SR / 2 / 1000 + 0.15)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc="upper right", fontsize=9)
    ax.set_ylim(min(-120.0, above - 15), 5)
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("Level under the peak (dB)")


def draw_loudness(fig, p) -> None:
    drift = _take()
    drift[8 * SR:13 * SR] *= 10 ** (-12 / 20)
    short = kudio.loudness_over_time(drift, SR)
    moment = kudio.loudness_over_time(drift, SR, window=kudio.MOMENTARY)
    integrated = kudio.loudness(drift, SR)
    dips = short.spans_below(short.max - 6, min_seconds=1.0)

    top, bottom = fig.subplots(2, 1, sharex=True,
                               gridspec_kw={"height_ratios": [1, 2.6]})
    time = np.arange(len(drift)) / SR
    top.plot(time, drift, color=p["muted"], lw=0.4)
    top.set_ylim(-0.3, 0.3)
    top.set_ylabel("Amplitude")
    top.axvspan(8, 13, color=p["c4"], alpha=0.10, lw=0)
    top.text(10.5, 0.27, "12 dB down", color=p["c4"], fontsize=9,
             ha="center", va="top")

    bottom.plot(moment.times, moment.lufs, color=p["muted"], lw=0.7,
                label="momentary (400 ms)")
    bottom.plot(short.times, short.lufs, color=p["c1"], lw=2.0,
                label="short-term (3 s)")
    bottom.axhline(integrated, color=p["c2"], lw=1.1, ls="--",
                   label=f"integrated, {integrated:.1f} LUFS")
    for a, b in dips:
        bottom.axvspan(a, b, color=p["c4"], alpha=0.12, lw=0)
    if dips:
        a, b = dips[0]
        bottom.text((a + b) / 2, -24.5, "spans_below(max - 6)", color=p["c4"],
                    fontsize=9, ha="center", va="top")
    bottom.set_xlim(0, len(drift) / SR)
    bottom.set_ylim(-70, -22)
    bottom.set_xlabel("Time (s)")
    bottom.set_ylabel("Loudness (LUFS)")
    bottom.legend(loc="lower left", fontsize=9, ncol=3)


def main() -> None:
    figstyle.save("inspection-bandwidth", draw_bandwidth, size=(8.0, 3.4))
    figstyle.save("inspection-loudness", draw_loudness, size=(8.0, 4.4))


if __name__ == "__main__":
    main()
