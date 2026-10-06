# -*- coding: utf-8 -*-
"""Dataset manifests: record which mixture came from which clean file.

`Synthesizer` already returns the pairing it wrote. Writing that down beside
the audio is what stops the pairing from being re-derived later by pattern
matching on filenames — the point at which one renamed directory silently
trains a model against the wrong targets.

The manifest is a plain JSON list of objects, so it stays readable and
diffable, and any other tool can consume it without importing kudio.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, fields
from pathlib import Path, PurePath
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = ['Pair', 'save_manifest', 'load_manifest', 'split_pairs',
           'Label', 'save_labels', 'load_labels']

log = logging.getLogger(__name__)

PathLike = Union[str, PurePath]


@dataclass(frozen=True)
class Pair:
    """One training example: a degraded file and the clean signal behind it.

    Frozen, therefore hashable — splits are checked for overlap by putting the
    pairs in a set, which silently fails to catch anything if they are not.
    """
    noisy: str
    clean: str
    noise: str = ""
    #: an int when the SNR is a whole number of dB, as it always was; a float
    #: when it is not, rather than that float cut down to an int
    snr_db: Optional[float] = None
    #: reverberation time of the room the clean signal was put through, in
    #: seconds, or ``None`` when it was not. Optional and last, so a manifest
    #: written before rooms existed still loads and a manifest written with
    #: them still loads in an older kudio -- see :func:`load_manifest`.
    rt60: Optional[float] = None
    #: the link the mixture travelled down, as :func:`kudio.channel_tag`
    #: writes it (``'telephone'``, ``'mu_law+loss5%x4'``, ...), or ``None``.
    #: A string rather than the spec dict it came from: this dataclass is
    #: frozen so that a split can be checked for overlap with a set, and a
    #: dict field would quietly make it unhashable again.
    channel: Optional[str] = None
    #: the reverberant-but-clean signal, when one was written. ``clean`` is
    #: still the dry target; this is the same speech with the room left on,
    #: for training that should remove the noise and leave the room.
    target: Optional[str] = None

    def exists(self) -> bool:
        """True when both sides are actually on disk."""
        return Path(self.noisy).is_file() and Path(self.clean).is_file()


def save_manifest(path: PathLike, pairs: Sequence[Pair]) -> Path:
    """Write *pairs* to *path* as JSON. Parent directories are created."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump([asdict(p) for p in pairs], fh, indent=2, ensure_ascii=False)
    return path


def load_manifest(path: PathLike) -> List[Pair]:
    """Read a manifest written by :func:`save_manifest`.

    Extra keys are dropped with a warning rather than raising — a manifest
    written by a later version should still load — but a missing ``noisy`` or
    ``clean`` is an error, since there is no example without them.
    """
    with open(path, "r", encoding="utf-8") as fh:
        rows = json.load(fh)
    if not isinstance(rows, list):
        raise ValueError(f"manifest is not a JSON list: {path}")

    known = {f.name for f in fields(Pair)}
    out: List[Pair] = []
    dropped: Dict[str, int] = {}
    for row in rows:
        unknown = set(row) - known
        for key in unknown:
            dropped[key] = dropped.get(key, 0) + 1
        out.append(Pair(**{k: v for k, v in row.items() if k in known}))
    if dropped:
        log.warning("manifest %s: ignored unknown key(s) %s",
                    path, ", ".join(sorted(dropped)))
    return out


# ------------------------------------------------------------------- labels

@dataclass(frozen=True)
class Label:
    """A named span of a recording, in seconds. ``end == start`` is a point.

    What :func:`kudio.vad` finds, what an annotator marks, what a dataset
    builder cuts on — all the same shape, so they get one.
    """
    start: float
    end: float
    text: str = ""

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def is_point(self) -> bool:
        return self.end <= self.start


def _label_path(path: PathLike, fmt: Optional[str]) -> Tuple[Path, str]:
    path = Path(path)
    if fmt is None:
        fmt = 'audacity' if path.suffix.lower() in ('.txt', '.tsv') else 'json'
    if fmt not in ('json', 'audacity'):
        raise ValueError(f"format must be 'json' or 'audacity', got {fmt!r}")
    return path, fmt


def save_labels(path: PathLike, labels: Sequence[Label],
                fmt: Optional[str] = None) -> Path:
    """Write *labels* as JSON, or as an **Audacity label track**.

    >>> kudio.save_labels("clip.labels.json", labels)
    >>> kudio.save_labels("clip.txt", labels)          # Audacity, ELAN, ...

    The format follows the suffix unless you say otherwise: ``.txt`` / ``.tsv``
    means Audacity's tab-separated ``start end text``, anything else means
    JSON. Audacity's format is three columns of plain text that half the audio
    world can already read, which is worth more than a private format that only
    kudio understands.
    """
    path, fmt = _label_path(path, fmt)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == 'json':
        with open(path, "w", encoding="utf-8") as fh:
            json.dump([asdict(x) for x in labels], fh, indent=2,
                      ensure_ascii=False)
        return path

    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for label in labels:
            # tabs are the separator, so a tab inside a label would split the
            # row -- Audacity has the same problem and also just strips them
            text = label.text.replace("\t", " ").replace("\n", " ")
            # microseconds: 60x finer than a sample at 16 kHz and still 5x
            # finer at 192 kHz, so the rounding cannot move a boundary
            fh.write(f"{label.start:.6f}\t{label.end:.6f}\t{text}\n")
    return path


def load_labels(path: PathLike, fmt: Optional[str] = None) -> List[Label]:
    """Read labels written by :func:`save_labels`, or by Audacity.

    :raises ValueError: on a malformed row, naming the line — a label file is
        usually hand-edited at some point, and "line 12 has 2 columns" is worth
        more than an IndexError.
    """
    path, fmt = _label_path(path, fmt)
    if fmt == 'json':
        with open(path, "r", encoding="utf-8") as fh:
            rows = json.load(fh)
        if not isinstance(rows, list):
            raise ValueError(f"label file is not a JSON list: {path}")
        known = {f.name for f in fields(Label)}
        return [Label(**{k: v for k, v in row.items() if k in known})
                for row in rows]

    labels = []
    with open(path, "r", encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                raise ValueError(
                    f"{path} line {n}: expected 'start<TAB>end<TAB>text', "
                    f"found {len(parts)} column(s)")
            try:
                start, end = float(parts[0]), float(parts[1])
            except ValueError:
                raise ValueError(
                    f"{path} line {n}: {parts[0]!r} and {parts[1]!r} are not "
                    f"times in seconds") from None
            labels.append(Label(start=start, end=end,
                                text="\t".join(parts[2:]).strip()))
    return labels


def split_pairs(pairs: Sequence[Pair], val_split: float = 0.1,
                test_split: float = 0.1, seed: Optional[int] = None
                ) -> Tuple[List[Pair], List[Pair], List[Pair]]:
    """Shuffle once, then cut into ``(train, validation, test)``.

    Splitting a tiny set by percentage can leave nothing to train on, so when
    the three shares would starve training the validation and test sets are
    given up instead — an empty validation set is visible, a one-example
    training set looks like it worked.
    """
    if not 0.0 <= val_split < 1.0 or not 0.0 <= test_split < 1.0:
        raise ValueError("splits must be in [0, 1)")
    if val_split + test_split >= 1.0:
        raise ValueError(
            f"val_split + test_split must leave something to train on, "
            f"got {val_split} + {test_split}")

    items = list(pairs)
    rng = np.random.default_rng(seed)
    rng.shuffle(items)

    n = len(items)
    n_test = int(round(n * test_split))
    n_val = int(round(n * val_split))
    if n - n_val - n_test < 1:
        n_val = n_test = 0
    return (items[n_val + n_test:], items[:n_val], items[n_val:n_val + n_test])
