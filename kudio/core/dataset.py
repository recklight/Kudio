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
import math
import warnings
from dataclasses import asdict, dataclass, fields
from pathlib import Path, PurePath
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from kudio import _textfile
from kudio._textfile import plain

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


def _portable(row: dict) -> dict:
    """Paths with forward slashes. Windows reads them as well as its own, and
    a manifest written there with backslashes named no file on Linux or
    macOS. Relative paths stay relative, to the working directory of
    whoever reads them, as they always have."""
    for key in ('noisy', 'clean', 'target'):
        value = plain(row.get(key))
        if isinstance(value, str) and value:
            row[key] = value.replace('\\', '/')
    return row


def _json_value(value: Any) -> Any:
    """What :func:`json.dumps` asks for when it has no form for *value*."""
    simple = plain(value)
    if simple is value:
        raise TypeError(f"{value!r}, and JSON has no form for a {type(value).__name__}")
    return simple


def _read_json(path: PathLike, what: str) -> Any:
    """What *path* holds, or the error that stopped it with the file's name on:
    a JSONDecodeError says which line, but not of which of fifty label files.
    Both it and a UnicodeDecodeError are ValueErrors."""
    try:
        # -sig: Notepad's "UTF-8 with BOM" puts a mark in front, which is not JSON
        with open(path, "r", encoding="utf-8-sig") as fh:
            return json.load(fh)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"{what} {path} is not JSON: {e.msg}", e.doc, e.pos) from None
    except UnicodeDecodeError as e:
        raise ValueError(f"{what} {path} is not UTF-8 text: {e}") from None


def _json_text(rows: List[dict], name: str, what: str) -> str:
    """*rows* as the JSON list a file will hold, or a ValueError naming the
    first row that has no JSON."""
    def text(part):
        return json.dumps(part, indent=2, ensure_ascii=False, default=_json_value)

    try:
        return text(rows)
    except (TypeError, ValueError) as e:
        for n, row in enumerate(rows, 1):
            try:
                text(row)
            except (TypeError, ValueError) as why:
                raise ValueError(f"{name}(): {what} {n} holds {why}") from None
        raise ValueError(f"{name}(): {e}") from e


def save_manifest(path: PathLike, pairs: Sequence[Pair]) -> Path:
    """Write *pairs* to *path* as JSON. Parent directories are created.

    Paths are written with forward slashes, so a manifest made on Windows
    loads on Linux and macOS too. A ``Path`` goes in as its text and a numpy
    number as the number it is.

    :raises ValueError: naming the row, for a value JSON has no form for. A
        manifest already at *path* is left as it was.
    """
    path = Path(path)
    text = _json_text([_portable(asdict(p)) for p in pairs], 'save_manifest', 'row')
    _textfile.write(path, text, parents=True)
    return path


def load_manifest(path: PathLike) -> List[Pair]:
    """Read a manifest written by :func:`save_manifest`.

    Extra keys are dropped with a ``UserWarning`` rather than raising — a
    manifest written by a later version should still load — but a missing
    ``noisy`` or ``clean`` is an error, since there is no example without them.

    :raises ValueError: naming the file, when it is not JSON, and the row.
    """
    rows = _read_json(path, 'manifest')
    if not isinstance(rows, list):
        raise ValueError(f"manifest is not a JSON list: {path}")

    known = {f.name for f in fields(Pair)}
    out: List[Pair] = []
    dropped: Dict[str, int] = {}
    for n, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise ValueError(f"manifest {path}: row {n} is not an object: {row!r}")
        missing = [k for k in ('noisy', 'clean') if not row.get(k)]
        if missing:
            raise ValueError(f"manifest {path}: row {n} has no "
                             f"{' or '.join(missing)}; every row needs both "
                             f"'noisy' and 'clean'")
        unknown = set(row) - known
        for key in unknown:
            dropped[key] = dropped.get(key, 0) + 1
        out.append(Pair(**{k: v for k, v in row.items() if k in known}))
    if dropped:
        warnings.warn(f"manifest {path}: ignored unknown key(s) "
                      f"{', '.join(sorted(dropped))}", UserWarning, stacklevel=2)
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


def _seconds(value: Any, where: str) -> float:
    """*value* as a Python number, or a ValueError that says *where* it was."""
    value = plain(value)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where} has to be a number of seconds, got {value!r}")
    if not math.isfinite(value):
        raise ValueError(f"{where} has to be a finite number of seconds, got {value!r}")
    return value


def _span(start: Any, end: Any, where: str) -> Tuple[float, float]:
    """``(start, end)`` as Python numbers, or a ValueError for a time that is
    none or a label that ends before it starts."""
    start = _seconds(start, f"{where}: start")
    end = _seconds(end, f"{where}: end")
    if end < start:
        raise ValueError(f"{where} ends at {end:g} s, before it starts at {start:g} s")
    return start, end


def _microseconds(value: Any) -> float:
    """*value* rounded to 6 decimals, as a Python number. A np.float64 is a
    float and rounds as it is; a np.float32 is made one first, or 0.1 is
    written as 0.10000000149011612."""
    return plain(round(value if isinstance(value, float) else plain(value), 6))


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

    :raises ValueError: naming the label, for a time that is not a finite
        number, an end before the start, and a text that is not a string or has
        no form in JSON. A label file already at *path* is left as it was.
    """
    path, fmt = _label_path(path, fmt)
    labels = list(labels)
    times = [_span(x.start, x.end, f"save_labels(): label {n}")
             for n, x in enumerate(labels, 1)]
    if fmt == 'json':
        # to the microsecond, as the text format below writes them: a time
        # worked out by subtraction, 3.14 as 3.1399999999999997, is the same
        # boundary, and the file is read by people too
        rows = [{**asdict(x), 'start': _microseconds(x.start),
                 'end': _microseconds(x.end)} for x in labels]
        text, newline = _json_text(rows, 'save_labels', 'label'), None
    else:
        lines = []
        for n, (label, (start, end)) in enumerate(zip(labels, times), 1):
            if not isinstance(label.text, str):
                raise ValueError(f"save_labels(): the text of label {n} has to be "
                                 f"a string, got {label.text!r}")
            # tabs are the separator, so a tab inside a label would split the
            # row -- Audacity has the same problem and also just strips them.
            # A carriage return ends a line as surely as a newline does
            words = label.text.replace("\t", " ").replace("\r\n", " ")
            words = words.replace("\r", " ").replace("\n", " ")
            # microseconds: 60x finer than a sample at 16 kHz and still 5x
            # finer at 192 kHz, so the rounding cannot move a boundary
            lines.append(f"{start:.6f}\t{end:.6f}\t{words}\n")
        text, newline = "".join(lines), "\n"
    _textfile.write(path, text, newline=newline, parents=True)
    return path


def load_labels(path: PathLike, fmt: Optional[str] = None) -> List[Label]:
    """Read labels written by :func:`save_labels`, or by Audacity.

    Audacity follows a label that has a frequency range with a second row,
    ``\\<TAB>low<TAB>high``. Those rows are skipped: a :class:`Label` holds
    times, not frequencies.

    :raises ValueError: on a malformed row, naming the line, or in JSON which
        row of the list it is — a label file is usually hand-edited at some
        point, and "line 12 has 2 columns" is worth more than an IndexError. A
        time that is not finite and an end before its start are malformed too.
    """
    path, fmt = _label_path(path, fmt)
    labels = []
    if fmt == 'json':
        rows = _read_json(path, 'label file')
        if not isinstance(rows, list):
            raise ValueError(f"label file is not a JSON list: {path}")
        known = {f.name for f in fields(Label)}
        for n, row in enumerate(rows, 1):
            where = f"label file {path}: row {n}"
            if not isinstance(row, dict):
                raise ValueError(f"{where} is not an object: {row!r}")
            missing = [k for k in ('start', 'end') if k not in row]
            if missing:
                raise ValueError(f"{where} has no {' or '.join(missing)}; every "
                                 f"row needs both 'start' and 'end'")
            _span(row['start'], row['end'], where)
            labels.append(Label(**{k: v for k, v in row.items() if k in known}))
        return labels

    try:
        with open(path, "r", encoding="utf-8-sig") as fh:
            lines = fh.readlines()
    except UnicodeDecodeError as e:
        raise ValueError(f"label file {path} is not UTF-8 text: {e}") from None
    for n, line in enumerate(lines, 1):
        if not line.strip() or line.startswith("\\"):
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
        _span(start, end, f"{path} line {n}")
        labels.append(Label(start=start, end=end,
                            text="\t".join(parts[2:]).strip()))
    return labels


def _distinct_examples(pairs: Sequence[Pair]) -> List[Pair]:
    """*pairs* with each example once, in the order it first turned up.

    Two rows equal in every field are one example written down twice: the
    same mixture, from the same clean file and noise at the same SNR. A
    repeated value in ``snr_ratio`` produces a manifest like that, and so
    does gluing two runs over one output folder together. Dropping the repeat
    loses nothing, which is why this de-duplicates instead of refusing:
    refusing would send the caller off to hand-edit rows that say nothing,
    when the split they asked for is perfectly well defined. It warns anyway,
    because a row the caller counted on has gone.

    Rows that disagree and still name the same mixture are the other case.
    Only one of them can be true of that file and nothing here can tell
    which, so there is nothing to de-duplicate and that one raises.
    """
    counted: Dict[Pair, int] = {}
    for pair in pairs:
        counted[pair] = counted.get(pair, 0) + 1

    claimed: Dict[str, List[Pair]] = {}
    for pair in counted:
        claimed.setdefault(pair.noisy, []).append(pair)
    clashes = [rows for rows in claimed.values() if len(rows) > 1]
    if clashes:
        rows = clashes[0]
        differ = ", ".join(f.name for f in fields(Pair)
                           if len({getattr(r, f.name) for r in rows}) > 1)
        more = (f" ({len(clashes)} mixtures are listed like this)"
                if len(clashes) > 1 else "")
        raise ValueError(
            f"{rows[0].noisy} is listed {len(rows)} times, by rows that "
            f"disagree about {differ}{more}. Only one of those rows can be "
            f"true of that file and nothing here can tell which, so "
            f"splitting would train on it under one answer and then score "
            f"it under another. Give each mixture a name of its own "
            f"(mkdir_parents=True keeps the folder layout apart) and rebuild "
            f"the manifest, or drop the rows that are out of date.")

    repeats = sum(n - 1 for n in counted.values())
    if repeats:
        worst = max(counted, key=lambda p: counted[p])
        # count the rows we tallied, not len(pairs): the argument has already
        # been walked once by now, and an iterator has no length left to ask for
        total = sum(counted.values())
        warnings.warn(
            f"{repeats} of {total} row(s) repeat an earlier one and were "
            f"dropped; {len(counted)} distinct example(s) were split. "
            f"{worst.noisy} is listed {counted[worst]} times. A repeated row "
            f"lands in two splits at once and is then scored as if it had "
            f"been held out, which inflates every number; a value repeated "
            f"in snr_ratio= is the usual cause.",
            UserWarning, stacklevel=3)
    return list(counted)


def split_pairs(pairs: Sequence[Pair], val_split: float = 0.1,
                test_split: float = 0.1, seed: Optional[int] = None
                ) -> Tuple[List[Pair], List[Pair], List[Pair]]:
    """Shuffle once, then cut into ``(train, validation, test)``.

    An example listed more than once is cut down to one first, with a
    warning, since dealing the copies out would put the same file in two
    splits and then score it as if it had been held out. Rows that disagree
    about one mixture file raise instead — see :func:`_distinct_examples`.

    Splitting a tiny set by percentage can leave nothing to train on, so when
    the shares would leave training empty, the validation and test sets are
    given up instead: an empty validation set is visible, and a run with no
    training data at all is not a run. Anything above that is split as asked,
    even when training ends up with a single example.
    """
    if not 0.0 <= val_split < 1.0 or not 0.0 <= test_split < 1.0:
        raise ValueError("splits must be in [0, 1)")
    if val_split + test_split >= 1.0:
        raise ValueError(
            f"val_split + test_split must leave something to train on, "
            f"got {val_split} + {test_split}")

    items = _distinct_examples(pairs)
    rng = np.random.default_rng(seed)
    rng.shuffle(items)

    n = len(items)
    n_test = int(round(n * test_split))
    n_val = int(round(n * val_split))
    if n - n_val - n_test < 1:
        n_val = n_test = 0
    return (items[n_val + n_test:], items[:n_val], items[n_val:n_val + n_test])
