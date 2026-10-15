# -*- coding: utf-8 -*-
"""Redraw every figure on the documentation site.

    python tools/docs/make_figures.py            # all of them
    python tools/docs/make_figures.py speech     # only fig_speech.py

Needs kudio with the [viz] extra. Each ``fig_<page>.py`` here draws the
figures for one page of the guide; the output lands in ``docs/images``.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main(only=None) -> None:
    scripts = sorted(HERE.glob("fig_*.py"))
    if only:
        scripts = [s for s in scripts if s.stem[len("fig_"):] in only]
    for script in scripts:
        spec = importlib.util.spec_from_file_location(script.stem, script)
        module = importlib.util.module_from_spec(spec)
        sys.path.insert(0, str(HERE))
        try:
            spec.loader.exec_module(module)
            module.main()
        finally:
            sys.path.remove(str(HERE))
        print(f"drew {script.name}")


if __name__ == "__main__":
    main(sys.argv[1:] or None)
