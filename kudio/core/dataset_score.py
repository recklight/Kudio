# -*- coding: utf-8 -*-
"""Score a whole test set: every utterance, then the summary a results table
needs.

:func:`kudio.score` scores one pair. A paper reports a test set, and what
goes in its table is a mean over the utterances with an interval around it,
how many utterances that mean is over, a line that says how the numbers were
computed, and what to cite for them. :func:`score_dataset` does all of that
in one call, and ``kudio score`` is the same thing from a shell::

    >>> result = kudio.score_dataset("enhanced/", reference="clean/",
    ...                              baseline="noisy/",
    ...                              groups=re.compile(r"^(p\\d+)_"))  # doctest: +SKIP
    >>> print(result.to_markdown())                                 # doctest: +SKIP
    >>> result.save_rows("scores.csv")                              # doctest: +SKIP

The name follows :func:`kudio.score`: that one scores a pair, this one the
data set the pairs come from. It returns a :class:`DatasetScore`, which holds
one row per utterance as a plain dict, one :class:`MetricSummary` per
column of the table, and the signatures; it prints itself as Markdown, LaTeX
or JSON, and needs no pandas.

**Which file is which.** Each side, the files scored, the references and an
optional baseline, is a folder, a list, or in Python a dict from id to path.
In a folder the id of a file is its path below the folder without the
extension, with forward slashes: ``clean/p232/001.wav`` is ``p232/001``, so
two folders laid out alike pair up file for file. Every file in a format
soundfile reads is taken, by its suffix (``.wav``, ``.flac``, ``.ogg``,
``.mp3``, ``.aiff`` and the rest of ``soundfile.available_formats()``), in
every subfolder and whatever the case of the suffix; raw PCM is not, since
it cannot be read without its rate and encoding. m4a, aac and webm, which
:func:`kudio.file_load` decodes with ffmpeg, are not looked for either, and can
be named in a list or a dict. Any other file whose name
is no id of the folder is named in a warning, so that no utterance leaves
the set unsaid. Two files that come to one id, ``001.wav`` and
``001.flac``, are refused, since either could be the one meant. A list is a
Kaldi-style ``wav.scp``, one ``id path`` per line, or a CSV file with ``id``
and ``path`` columns, told apart by the ``.csv`` suffix. A relative path in a
list is read from the working directory, as Kaldi reads a ``wav.scp`` and
kudio its manifests. Every side has to hold the same ids, each once: an id
missing from one side, an id given twice and a file that does not exist are
refused before anything is scored, with the first few named and the rest
counted. Pairing up by whatever is left would score a different test set
from the one the table names. The items are scored in the order of the first
side, a folder's sorted by id. A file with several channels is averaged to
one, as kudio's commands read audio, and the item's notes say so.

**Rates.** The set is scored at one rate, the one the signature records.
Without *sr* that is the rate of the references, or of the files scored when
no metric takes a reference, read from the file headers before anything is
scored; references at more than one rate are refused, because a mean over
both is a mean over two measurements. A metric that is not defined at the
rate (PESQ, SRMR or NCM at 44.1 kHz, wideband PESQ at 8 kHz) is refused then
too. A file scored, or a baseline, at another rate than its reference is not
resampled behind your back: that item gets no score on that side, and the
reason says so. With *sr* every file is resampled to it as it is read, with
:func:`kudio.resample`, nothing is refused for its rate, and the signature
names the resampler.

**Metrics without a reference.** dnsmos, dnsmos_p808, srmr and wada_snr score
the file alone, so they run with no *reference* at all, and they can be mixed
with reference metrics in one call. Mixed, each runs as it would on its own:
the reference metrics on the pair, lined up first with *align*, and the
reference-free ones on the whole file as it was read, aligned or not, so that
a reference that cannot be read costs the reference metrics their score and
not the others. A reference metric with no *reference* is refused. dnsmos
gives three columns, ``dnsmos_sig``, ``dnsmos_bak`` and ``dnsmos_ovrl``.
The DNSMOS models are found as :func:`kudio.find_dnsmos_model` finds them,
from *dnsmos_dir* or ``$KUDIO_DNSMOS_DIR``, each is opened before anything is
scored, so that a model that takes the other input is refused then, and
each process opens a model once. The personalised P.835 model, mapped with
the polynomials of the other one, gets the warning :func:`kudio.dnsmos`
gives for it.

**What is not scored.** An item a metric cannot score, a clip too short for
PESQ or a silent reference, has None in that column, never nan, and the
reason in ``reasons``: the warning :func:`kudio.score` gives for the pair,
word for word, with the baseline called ``baseline`` on its side, or the
error that stopped the pair, such as a file that cannot be read or a length
mismatch past :data:`~kudio.core.evaluator.MAX_LENGTH_MISMATCH`. Warnings
given for an item that was still scored, a length difference cut away or a
signal close to silence, go in ``notes``. After the run one warning counts
each kind. An item is left out of the mean of a column it has no score in,
and the table says how many were.

**The summary.** For each column, the mean over the items scored and its
interval from :func:`kudio.bootstrap_ci`: BCa from 10,000 resamples with seed
0 unless told otherwise, over whole speakers when *groups* gives the speaker
of each item. With a *baseline* scored against the same references, the
three numbers of a column, baseline, mean and difference, are over the items
both sides scored, so that the difference is the mean minus the baseline
exactly; the difference gets :func:`kudio.paired_ci` and the two-sided
:func:`kudio.permutation_test`, grouped the same way, and :func:`kudio.holm`
adjusts the p-values over every column of the table that has one. That is
the family because the table is read as one claim, that the system and its
baseline differ, and the claim stands if any row says so: what has to stay
at 5% is the chance that some row shows a difference that is not there.
Whether a difference is for the better is read from its sign and the column
of which way is better. The columns are scored on the same items and move
together, and Holm's procedure holds under any such dependence. A column
with no p-value is left out of the family, and the table says which.
Several systems against one baseline make a larger family, one run each;
adjust their p-values together with :func:`kudio.holm`.

A signature, :func:`kudio.signature`, comes with the table, one line for the
intervals that share their kind (the means' and baselines'
:func:`kudio.bootstrap_ci`, the differences' :func:`kudio.paired_ci`), their
count of items, the method each result reports and the number of groups,
each line keyed by what it covers. Usually that is one line, two with a
baseline; a column that lost items to an unscorable clip gets a line of its
own, since its *n* is not the others', and so does an interval where BCa
gave way to the percentile one. An interval that does not exist, over one
item, is signed without ``ci``. Every line says how the run treated its
pairs, so a line that names PESQ alone still says the pairs were held to the
length limit when other metrics cut them. The references are
:func:`kudio.cite`'s for the metrics asked for.

**Several processes.** *jobs* scores that many items at a time, each in a
process of its own started with ``spawn``, as Windows and macOS start them
anyway, so that a worker inherits nothing from the process that called it
but the length limit, which goes with the work. Each worker gets an equal
share of the CPU's threads for BLAS (``OPENBLAS_NUM_THREADS``,
``OMP_NUM_THREADS`` and ``MKL_NUM_THREADS``, unless they are set already).
Without that share, four workers on a 20-thread CPU each start 20: SDR of
24 files of 2.45 s then took 95 s, against 2.6 s in one process and 1.9 s
with four workers of 5 threads. The rows come back in the order of
the items. For a given number of BLAS threads they hold the same numbers as
one process, to the last bit; across thread counts the dot products of
SI-SDR and SDR can differ by about 1e-14. I seed numpy's global generator
with 0 before each side of each item and put its state back afterwards, so
an item scores the same in any process and in any order. A script that passes *jobs* above 1 needs the
``if __name__ == '__main__':`` guard any spawned pool needs.
"""
from __future__ import annotations

import contextlib
import csv
import importlib
import json
import math
import numbers
import os
import re
import warnings
from pathlib import Path, PurePath
from typing import (Any, Dict, Iterable, Iterator, List, Mapping, NamedTuple, Optional,
                    Sequence, Tuple, Union)

import numpy as np

from kudio.core._policy import checked, sample_rate, warn
from kudio.core.registry import METRIC_INFO
from kudio.core.registry import cite as _cite
from kudio.core.signature import signature
from kudio.core.stats import (ConfidenceInterval, PermutationResult, bootstrap_ci, holm,
                              paired_ci, permutation_test)

__all__ = ['score_dataset', 'DatasetScore', 'MetricSummary', 'REFERENCE_FREE']

PathLike = Union[str, PurePath]

#: The metrics :func:`score_dataset` runs on the file scored alone, with or
#: without a reference.
REFERENCE_FREE = ('dnsmos', 'dnsmos_p808', 'srmr', 'wada_snr')

#: soundfile's formats that a folder is not searched for: raw PCM cannot be
#: read without its rate and encoding, WAVEX files end in .wav, and the MAT
#: and HTK formats share their suffixes with feature files
_NOT_SEARCHED = ('raw', 'wavex', 'mat4', 'mat5', 'htk')

#: the other suffixes files of those formats carry
_ALSO = {'aiff': ('.aif', '.aifc'), 'au': ('.snd',), 'nist': ('.sph',),
         'ogg': ('.oga', '.opus')}

#: how many ids, or items, a message names before it counts the rest
_NAMED = 5

#: the three numbers of :func:`kudio.dnsmos`, each a column
_DNSMOS_PARTS = ('sig', 'bak', 'ovrl')

#: the environment variables that set a BLAS library's threads
_THREADS = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')

#: the three intervals of a column, as the summary words them
_ROLES = (('mean', 'means'), ('baseline', 'baselines'), ('difference', 'differences'))

_CALLER = 'score_dataset'


def _refuse(message: str) -> ValueError:
    return ValueError(f"{_CALLER}(): {message}")


def _named(ids: Sequence[Any]) -> str:
    """The first few of *ids*, and how many more there are."""
    shown = ", ".join(str(i) for i in ids[:_NAMED])
    if len(ids) > _NAMED:
        shown += f" and {len(ids) - _NAMED} more"
    return shown


def _plural(count: int, word: str) -> str:
    return f"{count} {word if count == 1 else word + 's'}"


def _and(items: Sequence[str]) -> str:
    items = list(items)
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} and {items[-1]}"


# --- the sides ---------------------------------------------------------------------

def _audio_suffixes() -> frozenset:
    """The suffixes of the formats soundfile reads here, as a folder is
    searched for them."""
    import soundfile as sf
    formats = {name.lower() for name in sf.available_formats()} - set(_NOT_SEARCHED)
    return frozenset({f'.{name}' for name in formats} |
                     {suffix for name in formats for suffix in _ALSO.get(name, ())})


def _folder(root: Path, what: str) -> Dict[str, Path]:
    suffixes = _audio_suffixes()
    found: Dict[str, List[Path]] = {}
    others: List[Tuple[str, Path]] = []
    for path in root.rglob('*'):
        if not path.is_file() or path.name.startswith('.'):
            continue
        key = path.relative_to(root).with_suffix('').as_posix()
        if path.suffix.lower() in suffixes:
            found.setdefault(key, []).append(path)
        else:
            others.append((key, path))
    if not found:
        raise _refuse(f"{what} {root} holds no audio file in a format soundfile reads")
    clashes = [f"{key} ({', '.join(sorted(p.name for p in paths))})"
               for key, paths in sorted(found.items()) if len(paths) > 1]
    if clashes:
        raise _refuse(f"{_plural(len(clashes), 'id')} of {what} {root} "
                      f"{'comes' if len(clashes) == 1 else 'come'} from two files, "
                      f"and either could be the one meant: {_named(clashes)}. Keep one "
                      f"file per id")
    # a transcript beside its audio file is no utterance gone missing
    left = sorted(path.relative_to(root).as_posix() for key, path in others
                  if key not in found)
    if left:
        warn(f"{_CALLER}(): {_plural(len(left), 'file')} of {what} {root} "
             f"{'is' if len(left) == 1 else 'are'} not audio in a format soundfile "
             f"reads, going by the suffix, and left out: {_named(left)}")
    return {key: paths[0] for key, paths in sorted(found.items())}


def _entries(entries: List[Tuple[str, str, int]], source: Path,
             what: str) -> Dict[str, Path]:
    """id -> path from the (id, path, line) of a list, refusing an id given
    twice."""
    if not entries:
        raise _refuse(f"{what} {source} lists no file")
    lines: Dict[str, List[int]] = {}
    for key, _, number in entries:
        lines.setdefault(key, []).append(number)
    twice = [f"{key} (lines {', '.join(str(n) for n in numbers_)})"
             for key, numbers_ in lines.items() if len(numbers_) > 1]
    if twice:
        raise _refuse(f"{what} {source} gives {_plural(len(twice), 'id')} more than "
                      f"once: {_named(twice)}. Each id names one file")
    return {key: Path(path) for key, path, _ in entries}


def _scp(path: Path, what: str) -> Dict[str, Path]:
    entries = []
    with open(path, encoding='utf-8-sig') as fh:
        for number, line in enumerate(fh, 1):
            text = line.strip()
            if not text:
                continue
            parts = text.split(None, 1)
            if len(parts) < 2:
                raise _refuse(f"{what} {path}, line {number}: {text!r} is an id with no "
                              f"path; each line is `id path`")
            key, file = parts[0], parts[1].strip()
            if file.endswith('|'):
                raise _refuse(f"{what} {path}, line {number}: {file!r} is a command "
                              f"for Kaldi to run, and I only read files; write the "
                              f"audio out and list the files")
            entries.append((key, file, number))
    return _entries(entries, path, what)


def _csv(path: Path, what: str) -> Dict[str, Path]:
    with open(path, encoding='utf-8-sig', newline='') as fh:
        reader = csv.DictReader(fh)
        columns = reader.fieldnames or []
        if 'id' not in columns or 'path' not in columns:
            raise _refuse(f"{what} {path} needs an `id` and a `path` column, and has "
                          f"{columns}")
        entries = []
        for number, row in enumerate(reader, 2):
            key, file = (row.get('id') or '').strip(), (row.get('path') or '').strip()
            if not key or not file:
                raise _refuse(f"{what} {path}, line {number}: every row needs an id "
                              f"and a path, got {row.get('id')!r} and "
                              f"{row.get('path')!r}")
            entries.append((key, file, number))
    return _entries(entries, path, what)


def _side(source: Any, what: str) -> Dict[str, Path]:
    """id -> file for one side: a folder, a list file, or a dict."""
    if isinstance(source, Mapping):
        odd = [k for k in source if not isinstance(k, str) or not k]
        if odd:
            raise _refuse(f"the ids of {what} have to be text, got {_named(odd)}")
        if not source:
            raise _refuse(f"{what} is an empty dict")
        pathless = [key for key, value in source.items()
                    if not isinstance(value, (str, PurePath)) or not str(value)]
        if pathless:
            raise _refuse(f"{what} gives no path for {_plural(len(pathless), 'id')}: "
                          f"{_named(pathless)}. Each id maps to the path of its file")
        return {key: Path(value) for key, value in source.items()}
    if not isinstance(source, (str, PurePath)):
        raise _refuse(f"{what} has to be a folder, a list file or a dict from id to "
                      f"path, got {type(source).__name__}")
    path = Path(source)
    if path.is_dir():
        return _folder(path, what)
    if not path.exists():
        where = "" if path.is_absolute() else f", looked for in {Path.cwd()}"
        raise _refuse(f"{what} {path} does not exist{where}")
    if not path.is_file():
        raise _refuse(f"{what} {path} is neither a folder nor a list file")
    if path.suffix.lower() in _audio_suffixes():
        raise _refuse(f"{what} {path} is one audio file; kudio.score scores a single "
                      f"pair, and a test set is a folder or a list")
    try:
        return _csv(path, what) if path.suffix.lower() == '.csv' else _scp(path, what)
    except UnicodeDecodeError:
        raise _refuse(f"{what} {path} is not a text file, so it is no list of files; "
                      f"lists are UTF-8") from None


def _paired(sides: Sequence[Tuple[str, Dict[str, Path]]]) -> List[str]:
    """The ids, in the order of the first side, once every side is known to
    hold the same ones and every file to exist."""
    first_what, first = sides[0]
    order = list(first)
    problems = []
    for what, side in sides[1:]:
        lacking = [key for key in order if key not in side]
        extra = [key for key in side if key not in first]
        if lacking:
            problems.append(f"{_plural(len(lacking), 'id')} of {first_what} "
                            f"{'has' if len(lacking) == 1 else 'have'} no {what} file: "
                            f"{_named(lacking)}")
        if extra:
            problems.append(f"{what} has {_plural(len(extra), 'id')} that {first_what} "
                            f"has not: {_named(extra)}")
    if problems:
        raise _refuse(f"the sides do not pair up: {'; '.join(problems)}. Every id has "
                      f"to be on every side")
    for what, side in sides:
        absent = [f"{key} ({path})" for key, path in side.items() if not path.is_file()]
        if absent:
            raise _refuse(f"{_plural(len(absent), 'file')} of {what} "
                          f"{'does' if len(absent) == 1 else 'do'} not exist: "
                          f"{_named(absent)}")
    return order


# --- the groups --------------------------------------------------------------------

def _group_file(path: Path) -> Dict[str, str]:
    pairs: List[Tuple[str, str, int]] = []
    if path.suffix.lower() == '.csv':
        with open(path, encoding='utf-8-sig', newline='') as fh:
            reader = csv.DictReader(fh)
            columns = reader.fieldnames or []
            column = next((c for c in ('speaker', 'group') if c in columns), None)
            if 'id' not in columns or column is None:
                raise _refuse(f"groups {path} needs an `id` column and a `speaker` (or "
                              f"`group`) column, and has {columns}")
            for number, row in enumerate(reader, 2):
                pairs.append(((row.get('id') or '').strip(),
                              (row.get(column) or '').strip(), number))
    else:
        with open(path, encoding='utf-8-sig') as fh:
            for number, line in enumerate(fh, 1):
                parts = line.split()
                if not parts:
                    continue
                if len(parts) != 2:
                    raise _refuse(f"groups {path}, line {number}: {line.strip()!r} is "
                                  f"not `id speaker`, as a Kaldi utt2spk has it")
                pairs.append((parts[0], parts[1], number))
    labels: Dict[str, str] = {}
    for key, label, number in pairs:
        if not key or not label:
            raise _refuse(f"groups {path}, line {number}: every row needs an id and a "
                          f"speaker")
        if labels.setdefault(key, label) != label:
            raise _refuse(f"groups {path} puts {key} with two speakers, "
                          f"{labels[key]} and {label} (line {number})")
    return labels


def _group_labels(groups: Any, ids: Sequence[str]) -> Optional[List[Any]]:
    """One label per item, from what *groups* is, or None without groups;
    every label checked as the intervals will take it, before anything is
    scored."""
    if groups is None:
        return None
    if isinstance(groups, re.Pattern):
        if groups.groups < 1:
            raise _refuse(f"the pattern {groups.pattern!r} has no group in "
                          f"parentheses to take the speaker from")

        def find(key: str) -> Optional[str]:
            found = groups.search(key)
            return None if found is None else found.group(1)
        how = f"the pattern {groups.pattern!r} finds no speaker in"
    elif isinstance(groups, (str, PurePath)):
        path = Path(groups)
        if not path.is_file():
            raise _refuse(f"groups {groups!r} is not a file; pass a compiled pattern "
                          f"(re.compile) to take the speaker from the id")
        find = _group_file(path).get
        how = f"groups {path} has no speaker for"
    elif isinstance(groups, Mapping):
        find = groups.get
        how = "groups has no speaker for"
    elif callable(groups):
        find = groups
        how = "groups gave no speaker for"
    else:
        raise _refuse(f"groups has to be a file of speakers, a compiled pattern, a "
                      f"dict or a function of the id, got {type(groups).__name__}")
    labels = [find(key) for key in ids]
    missing = [key for key, label in zip(ids, labels)
               if label is None or (isinstance(label, str) and not label)]
    if missing:
        raise _refuse(f"{how} {_plural(len(missing), 'id')}: {_named(missing)}. Every "
                      f"item needs the speaker it is resampled with")
    from kudio.core.stats import _group_index
    for key, label in zip(ids, labels):
        try:
            _group_index([label], 1, _CALLER)
        except ValueError:
            raise _refuse(f"groups gives {key} the speaker {label!r}, which cannot "
                          f"label a group: a speaker is a number, text or a tuple of "
                          f"them, and not NaN") from None
    return labels


# --- the metrics -------------------------------------------------------------------

def _columns(name: str) -> Tuple[str, ...]:
    return tuple(f'dnsmos_{part}' for part in _DNSMOS_PARTS) if name == 'dnsmos' \
        else (name,)


def _default_metrics(sr: int) -> List[str]:
    """What :func:`kudio.score` runs when it is not given a list, at *sr*."""
    from kudio.core import evaluator
    return [m for m in evaluator._DEFAULT
            if (m not in evaluator._BACKEND or evaluator._has_module(evaluator._BACKEND[m][0]))
            and (m != 'pesq' or sr in (8000, 16000))]


def _asked(metrics: Any) -> Optional[List[str]]:
    if metrics is None:
        return None
    from kudio.core.evaluator import METRICS
    names = [metrics] if isinstance(metrics, str) else metrics
    try:
        names = list(names)
    except TypeError:
        raise _refuse(f"metrics has to be a name or a list of names, got "
                      f"{metrics!r}") from None
    known = tuple(METRICS) + REFERENCE_FREE
    unknown = [m for m in names if m not in known]
    if unknown:
        raise _refuse(f"unknown metric(s) {unknown}; choose from {known}")
    twice = sorted({m for m in names if names.count(m) > 1})
    if twice:
        raise _refuse(f"{', '.join(twice)} named twice")
    if not names:
        raise _refuse("metrics is empty; leave it out for the default set")
    return names


def _installed(names: Sequence[str]) -> None:
    from kudio.core import evaluator
    from kudio.exceptions import DependencyError
    for m in names:
        if m in evaluator._BACKEND and not evaluator._has_module(evaluator._BACKEND[m][0]):
            raise DependencyError(*evaluator._BACKEND[m])
        if m in ('dnsmos', 'dnsmos_p808') and not evaluator._has_module('onnxruntime'):
            raise DependencyError('onnxruntime', extra='dnsmos')


def _set_rate(files: Iterable[Path], what: str) -> int:
    """The one rate the readable files of *what* share, from their headers."""
    from kudio.core.io import audio_info
    counts: Dict[int, int] = {}
    for path in files:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                rate = audio_info(path, peak=False).sr
        except Exception:                       # noqa: BLE001 -- said per item later
            continue
        counts[rate] = counts.get(rate, 0) + 1
    if not counts:
        raise _refuse(f"none of the {what} files can be read, so there is no rate to "
                      f"score them at")
    if len(counts) > 1:
        spread = ", ".join(f"{rate} Hz: {_plural(n, 'file')}"
                           for rate, n in sorted(counts.items()))
        raise _refuse(f"the {what} files are at {len(counts)} rates ({spread}). A mean "
                      f"over them mixes measurements made at different rates, and one "
                      f"signature cannot say how they were made; pass sr to resample "
                      f"every file to one rate as it is read")
    return next(iter(counts))


def _options(names: Sequence[str], models: Mapping[str, str]) -> Dict[str, Dict[str, Any]]:
    """Each metric's keyword arguments, as the signature takes them."""
    return {name: ({'model_dir': models[name]} if name in models else {})
            for name in names}


def _check_rates(names: Sequence[str], sr: int, pesq_mode: str, pesq_scale: str,
                 models: Mapping[str, str], resampled: bool) -> None:
    """Refuse a metric that cannot run at *sr*, as :func:`kudio.signature`
    would refuse to describe it."""
    for name in names:
        try:
            signature(_options([name], models), sr, pesq_mode=pesq_mode,
                      pesq_scale=pesq_scale)
        except ValueError as e:
            text = str(e)
            for head in ('signature(): ', f'{name}: '):
                if text.startswith(head):
                    text = text[len(head):]
            way = "pass an sr it is defined at" if resampled else \
                "pass sr to resample every file as it is read"
            raise _refuse(f"{name} cannot score this set at {sr} Hz: {text}. Leave it "
                          f"out, or {way}") from None


#: DNSMOS sessions this process has opened, by model file
_SESSIONS: Dict[str, Any] = {}


def _dnsmos_module():
    # kudio.core.dnsmos as an attribute is the function, so the module is
    # looked up by its name
    return importlib.import_module('kudio.core.dnsmos')


def _open_models(models: Mapping[str, str], keep: bool) -> None:
    """Open each DNSMOS model once, here, and refuse one that takes the other
    input before any item is scored; *keep* holds the sessions for this
    process to score with."""
    from kudio.exceptions import FeatureError
    backend = _dnsmos_module()
    for name, model in models.items():
        path = Path(model)
        session = backend._session(path)
        try:
            backend._input_name(session, 2 if name == 'dnsmos' else 3, name)
        except FeatureError as e:
            text = str(e).split('(): ', 1)[-1].replace("the session's model", "the model")
            raise FeatureError(f"{_CALLER}(): {name} cannot run on {path}: "
                               f"{text}") from None
        own_mapping = (path.parent / 'polyfit.json').is_file()
        if name == 'dnsmos' and not own_mapping and 'pDNSMOS' in path.parts:
            # what kudio.dnsmos says of it, which it skips when given a session
            warn(f"{_CALLER}(): {path} is the personalised DNSMOS model, which the "
                 f"non-personalised mapping dnsmos applies was not fitted for; point "
                 f"dnsmos_dir at the P.835 model, DNSMOS/sig_bak_ovr.onnx")
        if keep and not (name == 'dnsmos' and own_mapping):
            _SESSIONS[model] = session


def _dnsmos(kind: str, y: np.ndarray, sr: int, model: str):
    """dnsmos or dnsmos_p808 of *y*, on a session opened once per process.
    A polyfit.json beside the P.835 model is only read without a session, so
    with one there the model is opened for every call, as dnsmos opens it."""
    backend = _dnsmos_module()
    if kind == 'dnsmos' and (Path(model).parent / 'polyfit.json').is_file():
        return backend.dnsmos(y, sr, model_dir=model)
    session = _SESSIONS.get(model)
    if session is None:
        session = _SESSIONS[model] = backend._session(Path(model))
    if kind == 'dnsmos':
        return backend.dnsmos(y, sr, model_dir=model, session=session)
    return backend.dnsmos_p808(y, sr, model_dir=model, session=session)


# --- one item, in whichever process ------------------------------------------------

class _Plan(NamedTuple):
    """What every item is scored with; sent to each worker, which would
    otherwise read the length limit afresh from its own import of kudio."""

    sr: int
    resample: bool
    reference: Tuple[str, ...]
    free: Tuple[str, ...]
    pesq_mode: str
    pesq_scale: str
    align: bool
    #: (metric, model file) for dnsmos and dnsmos_p808
    models: Tuple[Tuple[str, str], ...]
    #: :data:`~kudio.core.evaluator.MAX_LENGTH_MISMATCH` as the caller had it
    limit: float


class _Side:
    """One side of one item as it is scored: a value or a reason per column,
    and the notes."""

    def __init__(self) -> None:
        self.values: Dict[str, Optional[float]] = {}
        self.reasons: Dict[str, str] = {}
        self.notes: List[str] = []

    def fail(self, columns: Iterable[str], reason: str) -> None:
        for column in columns:
            self.values[column] = None
            self.reasons[column] = reason

    def record(self, column: str, value: Any, said: List[str],
               error: Optional[str]) -> None:
        number = None
        if error is None and value is not None:
            number = float(value)
        if number is not None and math.isfinite(number):
            self.values[column] = number
            self.notes.extend(said)
            return
        self.values[column] = None
        if error is not None:
            self.reasons[column] = error
            self.notes.extend(said)
            return
        why = [text for text in said if text.endswith('; scored nan')]
        self.reasons[column] = " ".join(why or said) or \
            "the metric gave no number and no warning to say why"
        self.notes.extend(text for text in said if text not in why)

    def result(self) -> Dict[str, Any]:
        # a warning that is some column's reason is not a note as well, which
        # it would be for the composite's other two columns
        reasons = set(self.reasons.values())
        return {'values': self.values, 'reasons': self.reasons,
                'notes': [text for text in dict.fromkeys(self.notes) if text not in reasons]}


def _load(path: Optional[str], plan: _Plan, label: str
          ) -> Tuple[Optional[np.ndarray], Optional[str], List[str]]:
    from kudio.core.evaluator import _recorded
    from kudio.core.io import file_load, resample

    def read():
        y, rate = file_load(path, sr=None, mono=False)
        if y.ndim > 1:
            # file_load's own mono mix, said rather than done in silence
            warn(f"{label} has {y.shape[1]} channels, averaged to one before it was "
                 f"scored")
            y = y.mean(axis=1).astype(np.float32)
        if rate != plan.sr:
            if not plan.resample:
                raise ValueError(f"{label} is at {rate} Hz and the set is scored at "
                                 f"{plan.sr} Hz; pass sr to resample every file as it "
                                 f"is read")
            y = resample(y, rate, plan.sr)
        return y
    y, said, error = _recorded(read)
    return y, error, said


def _reference_side(ref: np.ndarray, deg: np.ndarray, plan: _Plan, side: _Side,
                    label: str) -> None:
    """The reference metrics of one pair, run by the code :func:`kudio.score`
    runs them with."""
    from kudio.core import evaluator
    labels = ('ref', label)

    def scored():
        a, b = ref, deg
        if plan.align:
            from kudio.core.align import align
            # refused for NaN as score refuses it, before the correlation
            # can call it an unrelated recording; called faint further on
            with checked():
                a, b = evaluator._sides(a, b, names=labels, caller='score')
            a, b = align(a, b, plan.sr)
        return evaluator._score_pair(a, b, plan.sr, plan.reference, plan.pesq_mode,
                                     plan.pesq_scale, labels=labels, tolerance=plan.limit,
                                     recorded=True)
    results, said, error = evaluator._recorded(scored)
    side.notes.extend(said)
    if error is not None:
        side.fail(plan.reference, error)
        return
    for m in plan.reference:
        side.record(m, *results[m])


def _free_side(deg: np.ndarray, plan: _Plan, side: _Side) -> None:
    """The reference-free metrics of the file scored, whole."""
    from kudio.core.evaluator import _recorded
    models = dict(plan.models)
    for m in plan.free:
        if m == 'srmr':
            from kudio.core.srmr import srmr
            side.record(m, *_recorded(lambda: srmr(deg, plan.sr)))
        elif m == 'wada_snr':
            from kudio.core.wada import wada_snr
            side.record(m, *_recorded(lambda: wada_snr(deg, plan.sr)))
        elif m == 'dnsmos':
            rated, said, error = _recorded(lambda: _dnsmos(m, deg, plan.sr, models[m]))
            for part in _DNSMOS_PARTS:
                side.record(f'dnsmos_{part}', None if error else getattr(rated, part),
                            said, error)
        else:
            side.record(m, *_recorded(lambda: _dnsmos(m, deg, plan.sr, models[m])))


def _score_side(ref: Optional[np.ndarray], ref_error: Optional[str], path: str,
                plan: _Plan, label: str) -> Dict[str, Any]:
    side = _Side()
    np.random.seed(0)
    deg, error, said = _load(path, plan, label)
    side.notes.extend(said)
    free_columns = [c for m in plan.free for c in _columns(m)]
    if error is not None:
        side.fail(list(plan.reference) + free_columns, error)
    else:
        if plan.reference:
            if ref_error is not None:
                side.fail(plan.reference, ref_error)
            else:
                _reference_side(ref, deg, plan, side, label)
        if plan.free:
            _free_side(deg, plan, side)
    return side.result()


def _score_item(task: Tuple[str, Optional[str], str, Optional[str], _Plan]
                ) -> Dict[str, Any]:
    """Every metric of one item, on each side; a top-level function, so that
    a spawned worker can import it."""
    key, ref_path, deg_path, base_path, plan = task
    state = np.random.get_state()
    try:
        ref, ref_error, ref_notes = None, None, []
        if plan.reference:
            ref, ref_error, ref_notes = _load(ref_path, plan, 'ref')
        scored = {'deg': _score_side(ref, ref_error, deg_path, plan, 'deg')}
        scored['deg']['notes'] = list(dict.fromkeys(ref_notes + scored['deg']['notes']))
        if base_path is not None:
            scored['baseline'] = _score_side(ref, ref_error, base_path, plan, 'baseline')
    finally:
        np.random.set_state(state)
    return scored


# --- the result --------------------------------------------------------------------

class MetricSummary(NamedTuple):
    """One column of the table: one metric, or one of DNSMOS's three scores."""

    #: the column's name in the rows: the metric's, or ``dnsmos_sig`` and so on
    column: str
    #: the metric's name, as :func:`kudio.cite` and :data:`kudio.METRIC_INFO`
    #: take it
    metric: str
    #: what the table calls it
    label: str
    #: ``'higher'``, ``'lower'``, or None
    better: Optional[str]
    #: ``'dB'``, ``'MOS'`` and so on; ``''`` for none
    unit: str
    #: items in the mean, the baseline's and the difference's too
    n: int
    #: items left out of it, for want of a score on either side
    not_scored: int
    #: the mean and its interval, from :func:`kudio.bootstrap_ci`
    mean: ConfidenceInterval
    #: the baseline's mean and interval over the same items, with a baseline
    baseline: Optional[ConfidenceInterval] = None
    #: mean minus baseline, from :func:`kudio.paired_ci`
    difference: Optional[ConfidenceInterval] = None
    #: the two-sided :func:`kudio.permutation_test` of that difference
    test: Optional[PermutationResult] = None
    #: its p-value adjusted by :func:`kudio.holm` over every column with one
    p_holm: Optional[float] = None


def _decimals(metric: str) -> int:
    """Places to print a metric's numbers at: three on a scale that ends at 1
    (STOI, NCM, CSII) and for SDI, whose values sit under 1; two otherwise,
    which is where dB, MOS and the distances are read."""
    high = METRIC_INFO[metric].range[1]
    return 3 if (high is not None and high <= 1.0) or metric == 'sdi' else 2


def _interval_dict(ci: Optional[ConfidenceInterval]) -> Optional[Dict[str, Any]]:
    if ci is None:
        return None
    return {key: (None if isinstance(value, float) and not math.isfinite(value) else value)
            for key, value in ci._asdict().items()}


def _finite(value: Optional[float]) -> Optional[float]:
    return value if value is not None and math.isfinite(value) else None


#: p-values a reader holds a result against, which a rounded p must not land
#: on unless it is one: 410/8192 and 500/10001 both print 0.0500 at four places
_LEVELS = (0.05, 0.01, 0.001)


def _p_text(p: Optional[float]) -> str:
    if p is None or not math.isfinite(p):
        return 'n/a'
    if p < 0.00005:
        return '<0.0001'
    text = f"{p:.4f}"
    places = 4
    while float(text) in _LEVELS and p not in _LEVELS and places < 10:
        places += 1
        text = f"{p:.{places}f}"
    return text


class DatasetScore:
    """What :func:`score_dataset` returns: every item's scores, the summary
    of each column, and the signatures.

    :ivar rows: one dict per item, in the order the items were scored:
        ``id``, ``group`` when there are groups, one value per column (None
        where there is no score), ``baseline_<column>`` with a baseline,
        ``reasons`` (column -> why it has no score) and ``notes`` (warnings
        given while it was still scored), and ``baseline_reasons`` and
        ``baseline_notes``.
    :ivar summary: one :class:`MetricSummary` per column, in table order.
    :ivar signatures: the signature lines, each under what it covers, such
        as ``'the means and the baselines'``; see the module notes for why
        there can be more than one.
    :ivar holm_over: the columns whose p-values Holm adjusted together, with
        a baseline; the others had no p-value.
    :ivar columns: the columns, one per metric and three for dnsmos.
    :ivar metrics: the metrics, as asked for or as the default resolved.
    :ivar sr: the rate the set was scored at.
    """

    def __init__(self, *, rows: List[Dict[str, Any]], summary: List[MetricSummary],
                 signatures: Dict[str, str], columns: Tuple[str, ...],
                 metrics: Tuple[str, ...], sr: int, aligned: bool, baseline: bool,
                 grouped: bool, confidence: float, pesq_label: Optional[str]):
        self.rows = rows
        self.summary = summary
        self.signatures = signatures
        self.columns = columns
        self.metrics = metrics
        self.sr = sr
        self.aligned = aligned
        self.has_baseline = baseline
        self.grouped = grouped
        self.confidence = confidence
        self._pesq_label = pesq_label
        self.holm_over = [s.column for s in summary if s.p_holm is not None]

    def __repr__(self) -> str:
        return (f"DatasetScore({_plural(len(self.rows), 'item')}, "
                f"{', '.join(self.columns)}; {len(self.unscored)} with no score in some "
                f"column)")

    @property
    def unscored(self) -> List[str]:
        """The ids with no score in some column, on either side."""
        return [row['id'] for row in self.rows
                if row['reasons'] or row.get('baseline_reasons')]

    def cite(self) -> str:
        """:func:`kudio.cite` for the metrics, ready to save as a ``.bib``."""
        return _cite(list(self.metrics))

    @property
    def references(self) -> List[str]:
        """The BibTeX keys :meth:`cite` gives, kudio's first."""
        keys = ['kudio']
        for name in self.metrics:
            keys.extend(k for k in METRIC_INFO[name].refs if k not in keys)
        return keys

    # --- the table ---------------------------------------------------------------

    def _label(self, s: MetricSummary) -> str:
        if s.metric == 'pesq' and self._pesq_label:
            return self._pesq_label
        return f"{s.label} ({s.unit})" if s.unit else s.label

    def _level(self) -> str:
        return f"{self.confidence * 100:g}%"

    def _holm_note(self) -> Optional[str]:
        """Which columns Holm's family left out, when some were."""
        if not self.has_baseline:
            return None
        left = [self._label(s) for s in self.summary if s.p_holm is None]
        if not left:
            return None
        over = [self._label(s) for s in self.summary if s.p_holm is not None]
        if not over:
            return "No column has a p-value, so Holm's adjustment has nothing to adjust"
        one = len(left) == 1
        return (f"Holm's adjustment is over {_and(over)}; {_and(left)} "
                f"{'has' if one else 'have'} no p-value and {'is' if one else 'are'} "
                f"left out of it")

    def _signature_lines(self, lead: str) -> List[str]:
        if len(self.signatures) == 1:
            return [f"{lead}: {text}" for text in self.signatures.values()]
        return [f"{lead} for {scope}: {text}" for scope, text in self.signatures.items()]

    def to_markdown(self) -> str:
        """The summary as a Markdown table, the signatures and the references
        under it."""
        level = self._level()
        if self.has_baseline:
            head = ['Metric', 'Better', f'Baseline [{level} CI]', f'Mean [{level} CI]',
                    f'Difference [{level} CI]', 'p', 'p (Holm)', 'n', 'Not scored']
        else:
            head = ['Metric', 'Better', f'Mean [{level} CI]', 'n', 'Not scored']
        lines = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
        for s in self.summary:
            d = _decimals(s.metric)
            row = [self._label(s), s.better or '']
            if self.has_baseline:
                row += [_cell(s.baseline, d), _cell(s.mean, d),
                        _cell(s.difference, d, True),
                        _p_text(None if s.test is None else s.test.pvalue),
                        _p_text(s.p_holm)]
            else:
                row += [_cell(s.mean, d)]
            row += [str(s.n), str(s.not_scored)]
            lines.append('| ' + ' | '.join(row) + ' |')
        lines.append('')
        note = self._holm_note()
        if note:
            lines.append(f"{note}.")
        lines.extend(self._signature_lines('Signature'))
        lines.append(f"References: {', '.join(self.references)}")
        return '\n'.join(lines)

    def to_latex(self) -> str:
        """The summary as a LaTeX tabular with booktabs rules; the signatures
        and the keys to cite follow as comments."""
        level = self._level().replace('%', r'\%')
        if self.has_baseline:
            head = ['Metric', f'Baseline [{level} CI]', f'Mean [{level} CI]',
                    f'Difference [{level} CI]', '$p$', '$p$ (Holm)', '$n$']
        else:
            head = ['Metric', f'Mean [{level} CI]', '$n$']
        lines = [r'\begin{tabular}{l' + 'r' * (len(head) - 1) + '}', r'\toprule',
                 ' & '.join(head) + r' \\', r'\midrule']
        for s in self.summary:
            d = _decimals(s.metric)
            arrow = {'higher': r' $\uparrow$', 'lower': r' $\downarrow$'}.get(s.better, '')
            row = [_tex_escape(self._label(s)) + arrow]
            if self.has_baseline:
                row += [_cell(s.baseline, d, tex=True), _cell(s.mean, d, tex=True),
                        _cell(s.difference, d, True, tex=True),
                        _tex_p(None if s.test is None else s.test.pvalue),
                        _tex_p(s.p_holm)]
            else:
                row += [_cell(s.mean, d, tex=True)]
            row.append(str(s.n))
            lines.append(' & '.join(row) + r' \\')
        lines += [r'\bottomrule', r'\end{tabular}']
        note = self._holm_note()
        if note:
            lines.append(f"% {note}")
        lines.extend(f"% {line}" for line in self._signature_lines('signature'))
        lines.append(f"% \\cite{{{','.join(self.references)}}}")
        return '\n'.join(lines)

    def to_json(self) -> str:
        """The summary, the signatures and the references as JSON; numbers
        that are nan are null."""
        columns = []
        for s in self.summary:
            columns.append({
                'column': s.column, 'metric': s.metric, 'label': self._label(s),
                'better': s.better, 'unit': s.unit, 'n': s.n,
                'not_scored': s.not_scored, 'mean': _interval_dict(s.mean),
                'baseline': _interval_dict(s.baseline),
                'difference': _interval_dict(s.difference),
                'p': None if s.test is None else _finite(s.test.pvalue),
                'p_holm': _finite(s.p_holm),
                'test': None if s.test is None else {
                    key: (_finite(value) if isinstance(value, float) else value)
                    for key, value in s.test._asdict().items()},
            })
        holm_family = None
        if self.has_baseline:
            holm_family = {'over': self.holm_over,
                           'left_out': [s.column for s in self.summary
                                        if s.p_holm is None]}
        return json.dumps({'items': len(self.rows), 'sr': self.sr,
                           'aligned': self.aligned, 'baseline': self.has_baseline,
                           'grouped': self.grouped, 'columns': columns,
                           'holm': holm_family,
                           'signatures': [{'for': scope, 'signature': text}
                                          for scope, text in self.signatures.items()],
                           'references': self.references}, indent=2)

    # --- the rows ----------------------------------------------------------------

    def _fields(self) -> List[str]:
        fields = ['id'] + (['group'] if self.grouped else []) + list(self.columns)
        if self.has_baseline:
            fields += [f'baseline_{c}' for c in self.columns]
        fields += ['reason', 'note']
        if self.has_baseline:
            fields += ['baseline_reason', 'baseline_note']
        return fields

    def save_rows(self, path: PathLike) -> Path:
        """Write :attr:`rows` to *path*: CSV for ``.csv``, one JSON object a
        line for ``.jsonl``. Parent folders are created.

        In the JSON lines a missing score is null, and ``reasons`` and
        ``notes`` are a dict and a list as in :attr:`rows`. In the CSV a
        missing score is an empty cell, and ``reason`` and ``note`` hold the
        distinct texts joined by `` | ``; kudio's warnings start with the
        function that gave them, so each says which metric it is about.
        Numbers are written as Python writes them, which reads back to the
        same float.
        """
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix not in ('.csv', '.jsonl'):
            raise _refuse(f"the rows go to a .csv or a .jsonl file, not {path.name}")
        path.parent.mkdir(parents=True, exist_ok=True)
        if suffix == '.jsonl':
            with open(path, 'w', encoding='utf-8', newline='\n') as fh:
                for row in self.rows:
                    fh.write(json.dumps(row, ensure_ascii=False, default=str) + '\n')
            return path
        with open(path, 'w', encoding='utf-8', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=self._fields())
            writer.writeheader()
            for row in self.rows:
                writer.writerow(self._flat(row))
        return path

    def _flat(self, row: Mapping[str, Any]) -> Dict[str, Any]:
        flat = {key: ('' if row.get(key) is None else row[key])
                for key in self._fields() if key not in ('reason', 'note', 'baseline_reason',
                                                         'baseline_note')}
        for prefix in ('',) + (('baseline_',) if self.has_baseline else ()):
            flat[f'{prefix}reason'] = ' | '.join(dict.fromkeys(row[f'{prefix}reasons'].values()))
            flat[f'{prefix}note'] = ' | '.join(row[f'{prefix}notes'])
        return flat

    def to_dataframe(self):
        """:attr:`rows` as a pandas DataFrame, one row per item, with
        ``reason`` and ``note`` as in the CSV. Needs pandas (``kudio[data]``)."""
        from kudio.core.evaluator import _pandas
        numeric = list(self.columns) + ([f'baseline_{c}' for c in self.columns]
                                        if self.has_baseline else [])
        table = [{**self._flat(row), **{c: row[c] for c in numeric}} for row in self.rows]
        frame = _pandas().DataFrame(table, columns=self._fields())
        frame[numeric] = frame[numeric].astype(float)
        return frame


def _cell(ci: Optional[ConfidenceInterval], decimals: int, sign: bool = False, *,
          tex: bool = False) -> str:
    """An estimate and its interval, ``n/a`` for either part that is not
    there, in Markdown or, with *tex*, in LaTeX."""
    def number(value: float) -> str:
        text = f"{value:{'+' if sign else ''}.{decimals}f}"
        return text.replace('-', '$-$', 1) if tex and text.startswith('-') else text
    if ci is None or not math.isfinite(ci.estimate):
        return 'n/a'
    if not (math.isfinite(ci.low) and math.isfinite(ci.high)):
        return f"{number(ci.estimate)} [n/a]"
    return f"{number(ci.estimate)} [{number(ci.low)}, {number(ci.high)}]"


def _tex_p(p: Optional[float]) -> str:
    return _p_text(p).replace('<', '$<$')


_TEX_SPECIAL = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#',
                '_': r'\_', '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}',
                '^': r'\textasciicircum{}'}


def _tex_escape(text: str) -> str:
    return ''.join(_TEX_SPECIAL.get(ch, ch) for ch in text)


# --- the summary -------------------------------------------------------------------

def _scope(where: Iterable[Tuple[str, str]], everyone: Mapping[str, Sequence[str]]) -> str:
    """What a set of (role, column label) pairs covers, in words: ``the means
    and the baselines`` when it is every column, ``the baseline of PESQ``
    when it is one."""
    where = list(where)
    parts = []
    for role, plural in _ROLES:
        labels = list(dict.fromkeys(label for r, label in where if r == role))
        if not labels:
            continue
        if set(labels) == set(everyone.get(role, ())):
            parts.append(f"the {plural}")
        else:
            parts.append(f"the {plural if len(labels) > 1 else role} of {_and(labels)}")
    return _and(parts)


def _summarise(rows: List[Dict[str, Any]], columns: Sequence[str], owner: Mapping[str, str],
               labels: Optional[List[Any]], baseline: bool, confidence: float,
               n_resamples: int, seed: Any, label_of
               ) -> Tuple[List[MetricSummary], Dict[str, Optional[List[Any]]]]:
    """One :class:`MetricSummary` per column, and the group labels of the
    items each one is over. What the intervals and tests warn about is said
    once per warning, with the columns it is about."""
    settings = dict(confidence=confidence, n_resamples=n_resamples, seed=seed)
    summaries, scored_labels = [], {}
    said: Dict[str, List[Tuple[str, str]]] = {}

    def heard(role: str, column_label: str, compute):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            value = compute()
        for w in caught:
            said.setdefault(str(w.message), []).append((role, column_label))
        return value

    for column in columns:
        keep = [i for i, row in enumerate(rows) if row[column] is not None
                and (not baseline or row[f'baseline_{column}'] is not None)]
        values = [rows[i][column] for i in keep]
        groups = None if labels is None else [labels[i] for i in keep]
        scored_labels[column] = groups
        metric = owner[column]
        info = METRIC_INFO[metric]
        label = info.label
        if metric == 'dnsmos':
            label = f"DNSMOS {column.rsplit('_', 1)[1].upper()}"
        summary = dict(column=column, metric=metric, label=label, better=info.better,
                       unit=info.unit, n=len(keep), not_scored=len(rows) - len(keep))
        shown = label_of(metric, label, info.unit)
        summary['mean'] = heard('mean', shown,
                                lambda: bootstrap_ci(values, groups=groups, **settings))
        if baseline:
            base = [rows[i][f'baseline_{column}'] for i in keep]
            summary['baseline'] = heard('baseline', shown, lambda: bootstrap_ci(
                base, groups=groups, **settings))
            summary['difference'] = heard('difference', shown, lambda: paired_ci(
                values, base, groups=groups, **settings))
            summary['test'] = heard('difference', shown, lambda: permutation_test(
                values, base, groups=groups, n_resamples=n_resamples, seed=seed))
        summaries.append(MetricSummary(**summary))
    if baseline:
        tested = {s.column: s.test.pvalue for s in summaries
                  if s.test is not None and math.isfinite(s.test.pvalue)}
        adjusted = holm(tested) if tested else {}
        summaries = [s._replace(p_holm=adjusted.get(s.column)) for s in summaries]
    everyone = {role: [label_of(s.metric, s.label, s.unit) for s in summaries]
                for role, _ in _ROLES}
    for text, where in said.items():
        warn(f"{_CALLER}(): for {_scope(where, everyone)}: {text}")
    return summaries, scored_labels


def _signatures(summaries: Sequence[MetricSummary],
                scored_labels: Mapping[str, Optional[List[Any]]],
                options: Mapping[str, Dict[str, Any]], label_of, *, sr: int,
                pesq_mode: str, pesq_scale: str, aligned: bool, cut: bool,
                resampled: bool, confidence: float, n_resamples: int,
                seed: Any) -> Dict[str, str]:
    """One signature for each set of intervals that share their kind, their
    n, their method and their groups, under what it covers. An interval that
    does not exist, over fewer than two items or groups, is signed without
    ``ci``."""
    batches: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    everyone: Dict[str, List[str]] = {role: [] for role, _ in _ROLES}
    for s in summaries:
        if s.n < 1:
            continue
        shown = label_of(s.metric, s.label, s.unit)
        intervals = [('mean', 'bootstrap_ci', s.mean)]
        if s.baseline is not None:
            intervals += [('baseline', 'bootstrap_ci', s.baseline),
                          ('difference', 'paired_ci', s.difference)]
        for role, kind, ci in intervals:
            everyone[role].append(shown)
            exists = math.isfinite(ci.low) and math.isfinite(ci.high)
            key = (kind, ci.n, ci.method, ci.n_groups) if exists else ('', ci.n)
            batch = batches.setdefault(key, {'names': [], 'where': [],
                                             'groups': scored_labels[s.column]})
            if s.metric not in batch['names']:
                batch['names'].append(s.metric)
            batch['where'].append((role, shown))
    order = sorted(batches, key=lambda key: ('bootstrap_ci', 'paired_ci', '').index(key[0]))
    lines: Dict[str, str] = {}
    for key in order:
        batch = batches[key]
        ci = None
        if key[0]:
            ci = dict(kind=key[0], method=key[2], confidence=confidence,
                      n_resamples=n_resamples, seed=seed, groups=batch['groups'])
        lines[_scope(batch['where'], everyone)] = signature(
            {name: options[name] for name in batch['names']}, sr, pesq_mode=pesq_mode,
            pesq_scale=pesq_scale, aligned=aligned, n=key[1], ci=ci, cut=cut,
            resampled=resampled)
    return lines


def _say_what_was_missed(rows: List[Dict[str, Any]], columns: Sequence[str]) -> None:
    """One warning for the items left out of some mean, and one for the
    items scored with a warning."""
    # (columns, reason) -> the ids it holds for: one reason is said once,
    # however many items share it, as a clip too short for one metric often is
    missed: Dict[Tuple[str, str], List[str]] = {}
    for row in rows:
        for prefix in ('', 'baseline_'):
            by_text: Dict[str, List[str]] = {}
            for column, text in (row.get(f'{prefix}reasons') or {}).items():
                by_text.setdefault(text, []).append(column)
            for text, which in by_text.items():
                names = 'every metric' if len(which) == len(columns) > 1 else ', '.join(which)
                if prefix:
                    names += ' of the baseline'
                missed.setdefault((names, text), []).append(row['id'])
    if missed:
        ids = {row['id'] for row in rows if row['reasons'] or row.get('baseline_reasons')}
        parts = [f"{names}, for {_named(keys)}: {text}"
                 for (names, text), keys in list(missed.items())[:_NAMED]]
        if len(missed) > _NAMED:
            parts.append(f"and {_plural(len(missed) - _NAMED, 'other reason')}")
        warn(f"{_CALLER}(): {len(ids)} of {_plural(len(rows), 'item')} have no score in "
             f"some column, and each is left out of the means it has no score for. "
             f"{' / '.join(parts)}")
    noted = [row for row in rows if row['notes'] or row.get('baseline_notes')]
    if noted:
        first = noted[0]
        text = ' | '.join(first['notes'] + (first.get('baseline_notes') or []))
        warn(f"{_CALLER}(): {len(noted)} of {_plural(len(rows), 'item')} were scored with "
             f"a warning, which their rows keep in notes; the first, {first['id']}: {text}")


# --- the call ----------------------------------------------------------------------

def score_dataset(deg: Union[PathLike, Mapping[str, PathLike]], *,
                  reference: Union[PathLike, Mapping[str, PathLike], None] = None,
                  baseline: Union[PathLike, Mapping[str, PathLike], None] = None,
                  metrics: Union[str, Sequence[str], None] = None,
                  sr: Optional[int] = None, pesq_mode: str = 'auto',
                  pesq_scale: str = 'lqo', align: bool = False, groups: Any = None,
                  jobs: int = 1, confidence: float = 0.95, n_resamples: int = 10000,
                  seed: Any = 0, dnsmos_dir: Optional[PathLike] = None) -> DatasetScore:
    """Score every item of a test set, and summarise each metric over them.

    >>> result = kudio.score_dataset("enhanced/", reference="clean/")   # doctest: +SKIP
    >>> print(result.to_markdown())                                     # doctest: +SKIP

    The module notes say how files pair up, how rates are handled, what
    happens to an item a metric cannot score, and what the summary holds.

    :param deg: the files to score: a folder, a list file (``id path`` per
        line, or a ``.csv`` with ``id`` and ``path`` columns) or a dict from
        id to path. Keyword-only from here on, so that a reference cannot be
        passed where the files to score go.
    :param reference: the clean references, in any of those forms; None for
        the reference-free metrics alone.
    :param baseline: a third side scored with the same metrics against the
        same references, the noisy inputs say; the summary then adds the
        difference, its interval and its test.
    :param metrics: names as :func:`kudio.score` takes them, and dnsmos,
        dnsmos_p808, srmr and wada_snr. None runs what :func:`kudio.score`
        runs by default at the set's rate, with what is installed; with no
        reference, srmr and wada_snr at 8 and 16 kHz, and wada_snr alone at
        any other rate, since SRMR is defined at those two.
    :param sr: resample every file to this rate as it is read. None scores
        the files at their own rate, which then has to be one rate.
    :param pesq_mode, pesq_scale: as :func:`kudio.score` takes them.
    :param align: line each pair up with :func:`kudio.align` first; a pair it
        finds unrelated is not scored, with its reason. The signature says
        ``aligned=yes``.
    :param groups: the speaker of each item, for the intervals and the test:
        a file of ``id,speaker`` (CSV with ``id`` and ``speaker`` columns, or
        ``id speaker`` per line, as a Kaldi utt2spk), a compiled regular
        expression whose first group is taken from the id, a dict from id to
        speaker, or a function of the id.
    :param jobs: items scored at once, each in a process of its own; on Windows
        before Python 3.13 31 at most, and the BLAS threads are shared out among
        the processes there are.
    :param confidence: the level of every interval, as
        :func:`kudio.bootstrap_ci` takes it.
    :param n_resamples: the resamples of every interval, and the most sign
        patterns each test draws, as :func:`kudio.bootstrap_ci` and
        :func:`kudio.permutation_test` take it.
    :param seed: for every interval and test, as :func:`kudio.bootstrap_ci`
        takes it.
    :param dnsmos_dir: where the DNSMOS models are, as
        :func:`kudio.find_dnsmos_model` takes *model_dir*.
    :raises ValueError: before anything is scored, for sides that do not pair
        up, an id given twice, a file that does not exist, references at more
        than one rate, an unknown metric, a reference metric without a
        reference, a metric that cannot run at the set's rate, groups that
        miss an item or give one a label that is not one, and settings out
        of range. Each message starts ``score_dataset(): ``.
    :raises DependencyError: for a metric whose package is not installed.
    :raises FeatureError: when a DNSMOS model is not found, or takes the
        input of the other model.
    :returns: a :class:`DatasetScore`.
    """
    _SESSIONS.clear()
    try:
        return _score_dataset(deg, reference, baseline, metrics, sr, pesq_mode, pesq_scale,
                              align, groups, jobs, confidence, n_resamples, seed,
                              dnsmos_dir)
    finally:
        # the models are let go of once the run is over, not at the next one
        _SESSIONS.clear()


def _score_dataset(deg, reference, baseline, metrics, sr, pesq_mode, pesq_scale, align,
                   groups, jobs, confidence, n_resamples, seed, dnsmos_dir) -> DatasetScore:
    from kudio.core import evaluator
    from kudio.core.stats import _confidence, _generator, _resamples
    _confidence(confidence, _CALLER)
    _resamples(n_resamples, _CALLER)
    _generator(seed if not isinstance(seed, np.random.Generator) else None, _CALLER)
    if isinstance(jobs, bool) or not isinstance(jobs, numbers.Integral) or jobs < 1:
        raise _refuse(f"jobs is how many items are scored at once, a whole number "
                      f"from 1, got {jobs!r}")
    if pesq_mode not in evaluator.PESQ_MODES:
        raise _refuse(f"pesq_mode has to be one of {evaluator.PESQ_MODES}, got "
                      f"{pesq_mode!r}")
    if pesq_scale not in evaluator.PESQ_SCALES:
        raise _refuse(f"pesq_scale has to be one of {evaluator.PESQ_SCALES}, got "
                      f"{pesq_scale!r}")
    if not isinstance(align, (bool, np.bool_)):
        raise _refuse(f"align has to be True or False, got {align!r}")
    if sr is not None:
        sr = sample_rate(sr, _CALLER)
        if not isinstance(sr, int):
            raise _refuse(f"sr is a rate to resample to, a whole number of Hz, got {sr!r}")
    limit = float(evaluator.MAX_LENGTH_MISMATCH)

    asked = _asked(metrics)
    takes_reference = (asked is None and reference is not None) or \
        (asked is not None and any(m not in REFERENCE_FREE for m in asked))
    if asked is not None and reference is None:
        needing = [m for m in asked if m not in REFERENCE_FREE]
        if needing:
            raise _refuse(f"{', '.join(needing)} score{'s' if len(needing) == 1 else ''} "
                          f"against a reference, and none was given; pass reference, or "
                          f"choose from {REFERENCE_FREE}, which score the file alone")

    sides = [('deg', _side(deg, 'deg'))]
    if reference is not None:
        sides.append(('reference', _side(reference, 'reference')))
    if baseline is not None:
        sides.append(('baseline', _side(baseline, 'baseline')))
    ids = _paired(sides)
    found = dict(sides)
    labels = _group_labels(groups, ids)

    rate = sr if sr is not None else _set_rate(
        (found['reference' if takes_reference else 'deg'][key] for key in ids),
        'reference' if takes_reference else 'deg')
    if asked is None:
        names = _default_metrics(rate) if reference is not None else \
            (['srmr'] if rate in (8000, 16000) else []) + ['wada_snr']
    else:
        names = asked
    _installed(names)

    models: Dict[str, str] = {}
    if 'dnsmos' in names or 'dnsmos_p808' in names:
        from kudio.core.dnsmos import find_dnsmos_model
        for name, kind in (('dnsmos', 'p835'), ('dnsmos_p808', 'p808')):
            if name in names:
                models[name] = str(find_dnsmos_model(dnsmos_dir, model=kind))
    _check_rates(names, rate, pesq_mode, pesq_scale, models, sr is not None)
    from kudio._workers import most
    workers = most(min(int(jobs), len(ids)))
    _open_models(models, keep=workers <= 1)

    plan = _Plan(sr=rate, resample=sr is not None,
                 reference=tuple(m for m in names if m not in REFERENCE_FREE),
                 free=tuple(m for m in names if m in REFERENCE_FREE),
                 pesq_mode=pesq_mode, pesq_scale=pesq_scale, align=bool(align),
                 models=tuple(models.items()), limit=limit)
    tasks = [(key, None if reference is None else str(found['reference'][key]),
              str(found['deg'][key]),
              None if baseline is None else str(found['baseline'][key]), plan)
             for key in ids]
    scored = _run(tasks, workers)

    columns = tuple(c for m in names for c in _columns(m))
    owner = {c: m for m in names for c in _columns(m)}
    rows = []
    for i, (key, item) in enumerate(zip(ids, scored)):
        row: Dict[str, Any] = {'id': key}
        if labels is not None:
            row['group'] = labels[i]
        row.update((c, item['deg']['values'][c]) for c in columns)
        if baseline is not None:
            row.update((f'baseline_{c}', item['baseline']['values'][c]) for c in columns)
        row['reasons'] = item['deg']['reasons']
        row['notes'] = item['deg']['notes']
        if baseline is not None:
            row['baseline_reasons'] = item['baseline']['reasons']
            row['baseline_notes'] = item['baseline']['notes']
        rows.append(row)
    _say_what_was_missed(rows, columns)

    pesq_label = None
    if 'pesq' in names:
        mode = evaluator._pesq_mode(rate, pesq_mode, pesq_scale)
        pesq_label = f"PESQ ({mode.upper()} {'MOS-LQO' if pesq_scale == 'lqo' else 'raw'})"

    def label_of(metric: str, label: str, unit: str) -> str:
        if metric == 'pesq' and pesq_label:
            return pesq_label
        return f"{label} ({unit})" if unit else label

    summary, scored_labels = _summarise(rows, columns, owner, labels, baseline is not None,
                                        confidence, n_resamples, seed, label_of)
    # the pairs were cut, and held to the limit, unless PESQ was all they had
    # to give a number to, as in kudio.score
    cut = bool(plan.reference) and set(plan.reference) != {'pesq'}
    lines = _signatures(summary, scored_labels, _options(names, models), label_of,
                        sr=rate, pesq_mode=pesq_mode, pesq_scale=pesq_scale,
                        aligned=bool(align), cut=cut, resampled=sr is not None,
                        confidence=confidence, n_resamples=n_resamples, seed=seed)
    return DatasetScore(rows=rows, summary=summary, signatures=lines, columns=columns,
                        metrics=tuple(names), sr=rate, aligned=bool(align),
                        baseline=baseline is not None, grouped=labels is not None,
                        confidence=float(confidence), pesq_label=pesq_label)


@contextlib.contextmanager
def _worker_threads(workers: int) -> Iterator[str]:
    """The environment a spawned worker starts its BLAS from: an equal share
    of the CPU's threads for each worker, unless the variables are set
    already; put back as it was afterwards. Yields the share."""
    share = str(max(1, (os.cpu_count() or 1) // workers))
    saved = {name: os.environ.get(name) for name in _THREADS}
    for name in _THREADS:
        if saved[name] is None:
            os.environ[name] = share
    try:
        yield share
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _run(tasks: List[Tuple[Any, ...]], workers: int) -> List[Dict[str, Any]]:
    """Every task's result, in the order of the tasks."""
    from kudio._progress import tqdm
    desc = "[kudio] Scoring"
    if workers > 1:
        from kudio._workers import Workers
        chunk = max(1, min(16, len(tasks) // (workers * 4)))
        # the workers read the variables as they import numpy, which they do
        # as the pool starts them
        with _worker_threads(workers), \
                Workers(workers, 'score_dataset()', method='spawn') as pool:
            return list(tqdm(pool.imap(_score_item, tasks, chunk),
                             total=len(tasks), desc=desc))
    return [_score_item(task) for task in tqdm(tasks, desc=desc)]
