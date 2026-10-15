# -*- coding: utf-8 -*-
"""Figures for the documentation site, drawn once per colour scheme.

The site follows the reader's light or dark setting, and a plot drawn for one
is hard to read on the other, so every figure is rendered twice and the page
shows the matching one (Material's ``#only-light`` / ``#only-dark``)::

    ![Loudness over time](../images/inspection-loudness-light.webp#only-light)
    ![Loudness over time](../images/inspection-loudness-dark.webp#only-dark)

Each ``fig_<page>.py`` next to this file draws the figures for one page of the
guide; ``make_figures.py`` runs them all. Every signal in them is synthetic and
seeded, so running the scripts again gives the same pictures.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

#: Where the pages look for images.
IMAGES = Path(__file__).resolve().parents[2] / "docs" / "images"

#: Colours matched to Material's default (light) and slate (dark) schemes.
PALETTES: Dict[str, Dict[str, str]] = {
    "light": {"bg": "#ffffff", "fg": "#1f2329", "muted": "#5f6672",
              "grid": "#dde1e8", "c1": "#3f51b5", "c2": "#e8710a",
              "c3": "#188038", "c4": "#c5221f", "cmap": "magma"},
    "dark": {"bg": "#1e2129", "fg": "#d5d9e0", "muted": "#9aa0ab",
             "grid": "#363b46", "c1": "#8c9eff", "c2": "#ffab40",
             "c3": "#5bd68a", "c4": "#ff7a70", "cmap": "magma"},
}

#: 150 dpi on an 8-inch figure is 1200 px: sharp in an 800 px column.
DPI = 150


def _rc(p: Dict[str, str]) -> dict:
    return {
        "figure.facecolor": p["bg"], "axes.facecolor": p["bg"],
        "savefig.facecolor": p["bg"], "text.color": p["fg"],
        "axes.labelcolor": p["fg"], "axes.edgecolor": p["muted"],
        "xtick.color": p["muted"], "ytick.color": p["muted"],
        "axes.grid": True, "grid.color": p["grid"], "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.prop_cycle": matplotlib.cycler(color=[p["c1"], p["c2"], p["c3"], p["c4"]]),
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "regular",
        "legend.frameon": False, "image.cmap": p["cmap"],
    }


def save(name: str, draw: Callable[[plt.Figure, Dict[str, str]], None],
         size: Tuple[float, float] = (8.0, 3.2)) -> None:
    """Draw *name* for both schemes: ``draw(fig, palette)`` fills the figure,
    and the result goes to ``docs/images/<name>-light.webp`` and ``-dark.webp``.

    :param name: ``<page>-<what>``, e.g. ``"inspection-loudness"``.
    :param size: figure size in inches; 8 wide fills the text column.
    """
    IMAGES.mkdir(parents=True, exist_ok=True)
    for scheme, palette in PALETTES.items():
        with plt.rc_context(_rc(palette)):
            fig = plt.figure(figsize=size, dpi=DPI, layout="constrained")
            draw(fig, palette)
            png = IMAGES / f"{name}-{scheme}.png"
            fig.savefig(png, dpi=DPI)
            plt.close(fig)
        _to_webp(png)


def _to_webp(png: Path) -> None:
    """Re-encode as WebP. A spectrogram of anything noisy is a poor fit for
    PNG (300 kB a picture); WebP at quality 88 keeps the text sharp at about
    a third of that."""
    from PIL import Image                    # Pillow comes with matplotlib
    with Image.open(png) as image:
        image.convert("RGB").save(png.with_suffix(".webp"), "WEBP",
                                  quality=88, method=6)
    png.unlink()
