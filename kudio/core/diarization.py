# -*- coding: utf-8 -*-
"""Who spoke when: the diarization error rate, the Jaccard error rate and the
detection error rate of speech activity, against a reference of speaker
turns.

**DER** is the error measure of NIST's Rich Transcription evaluations [RT09],
scored by its ``md-eval.pl`` and by dscore, the scorer of DIHARD II and III
[DIH19]. The time scored is cut into regions where the set of speakers stays
the same, and each reference speaker is mapped to at most one system speaker,
the mapping that makes the time they speak together the longest. A region of
d seconds with N_ref reference speakers, N_sys system speakers and N_correct
reference speakers whose mapped system speaker is talking there counts::

    missed       d max(0, N_ref - N_sys)
    false alarm  d max(0, N_sys - N_ref)
    confusion    d (min(N_ref, N_sys) - N_correct)
    total        d N_ref

and DER is the three errors over the total, so overlapping speech counts once
for every speaker in it. It can pass 1: a system that talks where nobody
does adds false alarm without bound.

**JER**, introduced for DIHARD II [DIH19], scores each reference speaker on
its own: the speaker's Jaccard error, 1 minus the time it shares with its
system speaker over the union of the two, averaged over the reference
speakers, so that every speaker weighs the same whatever they said. A
reference speaker left without a system speaker scores 1. The mapping is the
one dscore takes: the Hungarian assignment with the smallest summed Jaccard
error. pyannote.metrics' ``JaccardErrorRate`` maps by the longest shared time
instead, as for DER, and the two can part where a speaker's best overlap and
best Jaccard index point at different system speakers.

**Detection error rate** scores speech activity alone: the missed and
false-alarm speech over the reference speech, every speaker merged into one,
as pyannote.metrics' ``DetectionErrorRate`` takes it. The labels of a VAD are
enough.

    [RT09]  NIST, "The 2009 (RT-09) Rich Transcription Meeting Recognition
            Evaluation Plan", 2009, Sec. 6.1.
    [DIH19] N. Ryant, K. Church, C. Cieri, A. Cristia, J. Du, S. Ganapathy
            and M. Liberman, "Second DIHARD Challenge Evaluation Plan",
            version 1.1, 2019; the scorer is dscore,
            github.com/nryant/dscore.

What to watch for:

* The collar is per side, in seconds, as in md-eval.pl and dscore:
  ``collar=0.25`` leaves 0.25 s either side of every reference turn boundary
  unscored, 0.5 s in all. pyannote.metrics' ``collar`` is the total, so its
  0.5 is this 0.25. DIHARD scores with no collar and with overlapping speech;
  the older NIST evaluations and many papers use 0.25 s and drop the
  overlap, and the two settings give very different numbers, so say which.
* Without *uem* the whole of time is scored, which counts only where
  someone, in the reference or the system, is talking. Give the scoring
  regions where only part of a recording was annotated.
* dscore computes JER on 10 ms frames, so its JER and the exact one here
  agree to the last digit only where every boundary is on that grid.
* The speakers are mapped on all the time evaluated, before the collars
  and the overlap are taken out, as md-eval.pl maps them. pyannote.metrics
  maps on the time it scores, and with a collar or without the overlap the
  two can part: reference speakers A and B talking together for 10 s, then
  C alone for 2 s, against one system speaker throughout, score a DER of 1
  here and 0 there once the overlap is left out.
* Without *uem*, all of time is scored, as dscore and pyannote.metrics score
  it. md-eval.pl run without a UEM scores only from the reference's first
  turn to its last, so system speech before or after those is not counted.
* Turns of one speaker that overlap or touch are one stretch of speech here.
  md-eval.pl refuses an RTTM with overlapping turns of one speaker, and
  pyannote.metrics counts the time they share twice. Collars are cut around
  every turn's boundaries as given.
"""
from __future__ import annotations

import math
import numbers
from pathlib import Path
from typing import Dict, Hashable, Iterable, List, Mapping, NamedTuple, Optional, Tuple

import numpy as np

from kudio.core._policy import warn

__all__ = ['Turn', 'DiarizationErrors', 'load_rttm', 'der', 'diarization_errors',
           'jer', 'detection_error_rate']

Interval = Tuple[float, float]


class Turn(NamedTuple):
    """A stretch of one speaker's speech, in seconds."""
    start: float
    end: float
    speaker: Hashable


class DiarizationErrors(NamedTuple):
    """The seconds behind a DER, summed over the recordings of a corpus."""

    #: reference speech no system speaker covers, counted per speaker
    missed: float
    #: system speech beyond the reference speakers present, per speaker
    false_alarm: float
    #: reference speech given to the wrong system speaker
    confusion: float
    #: reference speech scored, counted once for each speaker in it
    total: float

    @property
    def der(self) -> float:
        """``(missed + false_alarm + confusion) / total``; nan for no total."""
        if self.total <= 0.0:
            return float('nan')
        return (self.missed + self.false_alarm + self.confusion) / self.total


# --- input ------------------------------------------------------------------------

def _number(value, what: str, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name}(): {what} has to be a number of seconds, got "
                         f"{value!r}")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name}(): {what} has to be finite, got {value!r}")
    return value


def _turns(turns, label: str, name: str) -> List[Turn]:
    """*turns* as a list of :class:`Turn`: tuples ``(start, end, speaker)``,
    Turns, or kudio Labels (their text is the speaker)."""
    if isinstance(turns, (str, bytes)) or not isinstance(turns, Iterable):
        raise ValueError(f"{name}(): {label} has to be a sequence of turns "
                         f"(start, end, speaker), got {type(turns).__name__}")
    out = []
    for i, turn in enumerate(turns):
        if hasattr(turn, 'start') and hasattr(turn, 'end'):
            speaker = getattr(turn, 'speaker', getattr(turn, 'text', None))
            start, end = turn.start, turn.end
        else:
            try:
                start, end, speaker = turn
            except (TypeError, ValueError):
                raise ValueError(f"{name}(): {label}[{i}] is not a turn "
                                 f"(start, end, speaker): {turn!r}") from None
        start = _number(start, f"the start of {label}[{i}]", name)
        end = _number(end, f"the end of {label}[{i}]", name)
        if end < start:
            raise ValueError(f"{name}(): {label}[{i}] ends at {end:g} s, before "
                             f"it starts at {start:g} s")
        try:
            hash(speaker)
        except TypeError:
            raise ValueError(f"{name}(): the speaker of {label}[{i}] has to be a "
                             f"label such as a string, got {speaker!r}") from None
        out.append(Turn(start, end, speaker))
    return out


def _regions(uem, name: str) -> Optional[List[Interval]]:
    if uem is None:
        return None
    out = []
    for i, region in enumerate(uem):
        try:
            start, end = region
        except (TypeError, ValueError):
            raise ValueError(f"{name}(): uem[{i}] is not a region (start, end): "
                             f"{region!r}") from None
        start = _number(start, f"the start of uem[{i}]", name)
        end = _number(end, f"the end of uem[{i}]", name)
        if end < start:
            raise ValueError(f"{name}(): uem[{i}] ends before it starts")
        out.append((start, end))
    return _union(out)


def _recordings(reference, hypothesis, uem, name: str):
    """``[(recording, ref turns, hyp turns, regions)]``: one for a pair of
    turn lists, one per key for a pair of dicts."""
    if isinstance(reference, Mapping) != isinstance(hypothesis, Mapping):
        raise ValueError(f"{name}(): pass reference and hypothesis alike, both "
                         f"turn lists or both dicts of them by recording")
    if not isinstance(reference, Mapping):
        if isinstance(uem, Mapping):
            raise ValueError(f"{name}(): uem is a dict, but reference is one "
                             f"recording; pass its scoring regions as a list")
        return [(None, _turns(reference, 'reference', name),
                 _turns(hypothesis, 'hypothesis', name), _regions(uem, name))]
    missing = [k for k in reference if k not in hypothesis]
    extra = [k for k in hypothesis if k not in reference]
    if missing or extra:
        said = []
        if missing:
            said.append(f"{len(missing)} recording(s) have no hypothesis "
                        f"({', '.join(map(str, missing[:3]))}"
                        f"{', ...' if len(missing) > 3 else ''})")
        if extra:
            said.append(f"{len(extra)} hypothesis recording(s) have no reference "
                        f"({', '.join(map(str, extra[:3]))}"
                        f"{', ...' if len(extra) > 3 else ''})")
        raise ValueError(f"{name}(): {'; '.join(said)}. Pair them up, with an "
                         f"empty list where a system found no speech")
    if uem is not None and not isinstance(uem, Mapping):
        raise ValueError(f"{name}(): with dicts of recordings, uem has to be a "
                         f"dict of scoring regions by recording too")
    out = []
    for key in reference:
        regions = None
        if uem is not None:
            if key not in uem:
                raise ValueError(f"{name}(): uem has no regions for {key!r}")
            regions = _regions(uem[key], name)
        out.append((key, _turns(reference[key], f"reference[{key!r}]", name),
                    _turns(hypothesis[key], f"hypothesis[{key!r}]", name), regions))
    return out


def load_rttm(path) -> Dict[str, List[Turn]]:
    """The SPEAKER lines of an RTTM file, as turns by recording.

    >>> reference = kudio.load_rttm("ref.rttm")              # doctest: +SKIP

    Each line ``SPEAKER <file> <channel> <onset> <duration> <NA> <NA>
    <speaker> ...`` becomes ``Turn(onset, onset + duration, speaker)`` under
    ``<file>``; lines of other types are skipped, and so are blank lines and
    lines starting with ``;;``. The file is UTF-8, with or without a byte-order
    mark.

    :raises ValueError: for a SPEAKER line with fewer than 8 fields, an onset
        or duration that is not a finite number, or a negative duration,
        naming the line.
    """
    out: Dict[str, List[Turn]] = {}
    # -sig: with the mark on it the first line's type reads "\ufeffSPEAKER",
    # which is a line of another type, and its turn would be skipped
    text = Path(path).read_text(encoding='utf-8-sig')
    for number, line in enumerate(text.splitlines(), 1):
        fields = line.split()
        if not fields or fields[0].startswith(';;') or fields[0] != 'SPEAKER':
            continue
        if len(fields) < 8:
            raise ValueError(f"load_rttm(): line {number} of {path} has "
                             f"{len(fields)} fields, and a SPEAKER line has 10")
        try:
            onset, duration = float(fields[3]), float(fields[4])
        except ValueError:
            raise ValueError(f"load_rttm(): line {number} of {path}: the onset "
                             f"and duration have to be numbers, got "
                             f"{fields[3]!r} and {fields[4]!r}") from None
        if not (math.isfinite(onset) and math.isfinite(duration)):
            raise ValueError(f"load_rttm(): line {number} of {path}: the onset and "
                             f"duration have to be finite, got {fields[3]!r} and "
                             f"{fields[4]!r}")
        if duration < 0:
            raise ValueError(f"load_rttm(): line {number} of {path}: the duration "
                             f"cannot be negative, got {fields[4]!r}")
        out.setdefault(fields[1], []).append(Turn(onset, onset + duration, fields[7]))
    return out


# --- intervals --------------------------------------------------------------------

def _union(intervals: Iterable[Interval]) -> List[Interval]:
    """Sorted, merged, non-empty."""
    out: List[Interval] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if out and start <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], end))
        else:
            out.append((start, end))
    return out


def _subtract(keep: List[Interval], cut: List[Interval]) -> List[Interval]:
    """*keep* less *cut*, both merged."""
    out = []
    j = 0
    for start, end in keep:
        while j < len(cut) and cut[j][1] <= start:
            j += 1
        k = j
        while k < len(cut) and cut[k][0] < end:
            if cut[k][0] > start:
                out.append((start, cut[k][0]))
            start = max(start, cut[k][1])
            k += 1
        if start < end:
            out.append((start, end))
    return out


def _speakers(turns: List[Turn]) -> Dict[Hashable, List[Interval]]:
    by: Dict[Hashable, List[Interval]] = {}
    for start, end, speaker in turns:
        by.setdefault(speaker, []).append((start, end))
    return {speaker: _union(spans) for speaker, spans in by.items()}


def _pieces(ref: Dict, hyp: Dict, scored: Optional[List[Interval]]):
    """``(duration, ref speakers, hyp speakers)`` of every piece of scored
    time over which the two sets of speakers stay the same."""
    points = set()
    for spans in list(ref.values()) + list(hyp.values()):
        for start, end in spans:
            points.add(start)
            points.add(end)
    if scored is not None:
        for start, end in scored:
            points.add(start)
            points.add(end)
    points = sorted(points)
    ref_names, hyp_names = list(ref), list(hyp)
    pieces = []
    for t0, t1 in zip(points[:-1], points[1:]):
        middle = 0.5 * (t0 + t1)
        if scored is not None and not _inside(scored, middle):
            continue
        r = tuple(i for i, s in enumerate(ref_names) if _inside(ref[s], middle))
        h = tuple(j for j, s in enumerate(hyp_names) if _inside(hyp[s], middle))
        if r or h:
            pieces.append((t1 - t0, r, h))
    return pieces, ref_names, hyp_names


def _inside(spans: List[Interval], t: float) -> bool:
    """Whether *t* falls in one of the merged, sorted *spans*."""
    lo, hi = 0, len(spans)
    while lo < hi:
        mid = (lo + hi) // 2
        if spans[mid][1] <= t:
            lo = mid + 1
        else:
            hi = mid
    return lo < len(spans) and spans[lo][0] <= t < spans[lo][1]


def _assign(table: np.ndarray, maximize: bool) -> Dict[int, int]:
    """Row -> column of the best one-to-one assignment."""
    if table.size == 0:
        return {}
    from scipy.optimize import linear_sum_assignment
    rows, cols = linear_sum_assignment(table, maximize=maximize)
    return dict(zip(rows.tolist(), cols.tolist()))


# --- DER --------------------------------------------------------------------------

def _scored(ref_turns: List[Turn], hyp_turns: List[Turn], ref: Dict,
            regions: Optional[List[Interval]], collar: float,
            skip_overlap: bool) -> Optional[List[Interval]]:
    """The time scored: the regions less the collars and, if asked, the
    reference's overlapping speech. None for all of time."""
    cut = []
    if collar > 0.0:
        for start, end, _ in ref_turns:
            cut.append((start - collar, start + collar))
            cut.append((end - collar, end + collar))
    if skip_overlap:
        points = sorted({p for spans in ref.values() for span in spans for p in span})
        for t0, t1 in zip(points[:-1], points[1:]):
            middle = 0.5 * (t0 + t1)
            if sum(_inside(spans, middle) for spans in ref.values()) >= 2:
                cut.append((t0, t1))
    if not cut:
        return regions
    if regions is None:
        # all of time, as far as anything in it can matter
        ends = [p for t in ref_turns + hyp_turns for p in (t.start, t.end)]
        ends += [p for span in cut for p in span]
        regions = [(min(ends) - 1.0, max(ends) + 1.0)]
    return _subtract(regions, _union(cut))


def _errors(ref_turns: List[Turn], hyp_turns: List[Turn],
            regions: Optional[List[Interval]], collar: float,
            skip_overlap: bool) -> Tuple[float, float, float, float]:
    """``(missed, false alarm, confusion, total)`` of one recording.

    The speakers are mapped on all the time evaluated, collars and
    overlapping speech included, and only then is the scored time cut out of
    it: md-eval.pl maps before it adds the collars and drops the overlap,
    and the RT-09 plan computes the mapping "over all speech, including
    regions of overlap". pyannote.metrics maps on the scored time alone.
    """
    ref, hyp = _speakers(ref_turns), _speakers(hyp_turns)
    evaluated, ref_names, hyp_names = _pieces(ref, hyp, regions)
    together = np.zeros((len(ref_names), len(hyp_names)))
    for d, r, h in evaluated:
        for i in r:
            for j in h:
                together[i, j] += d
    mapping = _assign(together, maximize=True)
    if collar > 0.0 or skip_overlap:
        scored = _scored(ref_turns, hyp_turns, ref, regions, collar, skip_overlap)
        pieces, ref_again, hyp_again = _pieces(ref, hyp, scored)
        assert ref_again == ref_names and hyp_again == hyp_names
    else:
        pieces = evaluated
    missed = false_alarm = confusion = total = 0.0
    for d, r, h in pieces:
        n_ref, n_hyp = len(r), len(h)
        correct = sum(1 for i in r if mapping.get(i) in h)
        missed += d * max(0, n_ref - n_hyp)
        false_alarm += d * max(0, n_hyp - n_ref)
        confusion += d * (min(n_ref, n_hyp) - correct)
        total += d * n_ref
    return missed, false_alarm, confusion, total


def _check_options(collar, skip_overlap, name):
    collar = _number(collar, 'collar', name)
    if collar < 0.0:
        raise ValueError(f"{name}(): collar has to be 0 or more seconds, got {collar:g}")
    if not isinstance(skip_overlap, (bool, np.bool_)):
        raise ValueError(f"{name}(): skip_overlap has to be True or False, got "
                         f"{skip_overlap!r}")
    return collar, bool(skip_overlap)


def diarization_errors(reference, hypothesis, *, collar: float = 0.0,
                       skip_overlap: bool = False, uem=None) -> DiarizationErrors:
    """The seconds of missed speech, false alarm and speaker confusion behind
    :func:`der`, and the reference speech they are counted against.

    >>> kudio.diarization_errors(reference, hypothesis)      # doctest: +SKIP
    DiarizationErrors(missed=1.2, false_alarm=0.8, confusion=3.1, total=60.0)

    Arguments as :func:`der`. Summed over the recordings of a corpus, each
    with its own speaker mapping.
    """
    collar, skip_overlap = _check_options(collar, skip_overlap, 'diarization_errors')
    sums = [0.0, 0.0, 0.0, 0.0]
    for _, ref, hyp, regions in _recordings(reference, hypothesis, uem,
                                            'diarization_errors'):
        for k, value in enumerate(_errors(ref, hyp, regions, collar, skip_overlap)):
            sums[k] += value
    return DiarizationErrors(*sums)


def der(reference, hypothesis, *, collar: float = 0.0, skip_overlap: bool = False,
        uem=None) -> float:
    """Diarization error rate of *hypothesis* against *reference*, as a
    fraction: missed speech, false alarm and speaker confusion over the
    reference speech. 0 is perfect; there is no upper bound.

    >>> kudio.der(reference, hypothesis)                     # DIHARD
    >>> kudio.der(reference, hypothesis, collar=0.25, skip_overlap=True)

    :param reference: what was said when: a sequence of turns
        ``(start, end, speaker)`` in seconds, :class:`Turn` or kudio
        :class:`~kudio.core.dataset.Label` (its text the speaker), or a dict
        of such sequences by recording, as :func:`load_rttm` returns.
    :param hypothesis: the system's turns, in the same form; its speaker
        labels need not match the reference's.
    :param collar: seconds left unscored on each side of every reference turn
        boundary, as md-eval.pl's ``-c`` and dscore's ``--collar`` take it.
        pyannote.metrics' collar is the total, twice this.
    :param skip_overlap: leave out the time where two or more reference
        speakers talk at once, as md-eval.pl's ``-1``.
    :param uem: the scoring regions, ``[(start, end), ...]`` for one
        recording or a dict of them by recording; all of time if None.
    :raises ValueError: for a turn that is not ``(start, end, speaker)`` of
        finite numbers with end not before start, a collar below 0, dicts
        whose recordings do not pair up, and a dict paired with a list.
    :returns: the DER, pooled over the recordings of a corpus. nan, with a
        warning, when the reference has no speech in the time scored.
    """
    collar, skip_overlap = _check_options(collar, skip_overlap, 'der')
    sums = [0.0, 0.0, 0.0, 0.0]
    for _, ref, hyp, regions in _recordings(reference, hypothesis, uem, 'der'):
        for k, value in enumerate(_errors(ref, hyp, regions, collar, skip_overlap)):
            sums[k] += value
    errors = DiarizationErrors(*sums)
    if errors.total <= 0.0:
        warn("der(): the reference has no speech in the time scored, so there "
             "is nothing to divide the errors by; scored nan")
        return float('nan')
    return errors.der


# --- JER --------------------------------------------------------------------------

def jer(reference, hypothesis, *, uem=None) -> float:
    """Jaccard error rate of *hypothesis* against *reference*, as a fraction:
    1 minus each reference speaker's Jaccard index with its system speaker,
    averaged over the reference speakers. 0 is perfect and 1 the worst.

    >>> kudio.jer(reference, hypothesis)                     # doctest: +SKIP

    Each reference speaker is paired with at most one system speaker, by the
    Hungarian assignment with the smallest summed Jaccard error, as dscore
    pairs them, and one left over scores 1. Overlapping speech counts for
    every speaker in it, and there is no collar, as in DIHARD.

    :param reference: as :func:`der`.
    :param hypothesis: as :func:`der`.
    :param uem: as :func:`der`.
    :raises ValueError: as :func:`der`.
    :returns: the JER; over a corpus, the mean over the reference speakers of
        every recording, as dscore pools it. nan, with a warning, when there
        is no reference speaker with speech in the time scored. dscore gives
        such a recording 1 when the system talks there and 0 when it does
        not, and leaves it out of the pooled figure, as this does.
    """
    errors = []
    for _, ref_turns, hyp_turns, regions in _recordings(reference, hypothesis, uem, 'jer'):
        ref, hyp = _speakers(ref_turns), _speakers(hyp_turns)
        if regions is not None:
            ref = {s: _intersect(spans, regions) for s, spans in ref.items()}
            hyp = {s: _intersect(spans, regions) for s, spans in hyp.items()}
        ref = {s: spans for s, spans in ref.items() if spans}
        hyp = {s: spans for s, spans in hyp.items() if spans}
        if not ref:
            continue
        ref_names, hyp_names = list(ref), list(hyp)
        ref_len = np.array([_length(ref[s]) for s in ref_names])
        hyp_len = np.array([_length(hyp[s]) for s in hyp_names])
        shared = np.array([[_length(_intersect(ref[a], hyp[b])) for b in hyp_names]
                           for a in ref_names]).reshape(len(ref_names), len(hyp_names))
        union = ref_len[:, None] + hyp_len[None, :] - shared
        table = 1.0 - shared / union
        mapping = _assign(table, maximize=False)
        for i in range(len(ref_names)):
            errors.append(float(table[i, mapping[i]]) if i in mapping else 1.0)
    if not errors:
        warn("jer(): the reference has no speaker with speech in the time scored, "
             "so there is no speaker to average over; scored nan")
        return float('nan')
    return float(np.mean(errors))


def _intersect(a: List[Interval], b: List[Interval]) -> List[Interval]:
    out = []
    i = j = 0
    while i < len(a) and j < len(b):
        start, end = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if start < end:
            out.append((start, end))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


def _length(spans: List[Interval]) -> float:
    return float(sum(end - start for start, end in spans))


# --- speech activity --------------------------------------------------------------

def detection_error_rate(reference, hypothesis, *, collar: float = 0.0,
                         uem=None) -> float:
    """Missed and false-alarm speech over the reference speech, every speaker
    merged into one: how well speech was found, whoever spoke.

    >>> labels = kudio.vad(y, sr)                            # doctest: +SKIP
    >>> kudio.detection_error_rate(truth, labels)            # doctest: +SKIP

    :param reference: as :func:`der`; the speaker labels are not read.
    :param hypothesis: as :func:`der`.
    :param collar: as :func:`der`, around the boundaries of the reference's
        turns.
    :param uem: as :func:`der`.
    :raises ValueError: as :func:`der`.
    :returns: the rate, from 0 up. nan, with a warning, when the reference
        has no speech in the time scored.
    """
    collar, _ = _check_options(collar, False, 'detection_error_rate')
    sums = [0.0, 0.0, 0.0, 0.0]
    for _, ref, hyp, regions in _recordings(reference, hypothesis, uem,
                                            'detection_error_rate'):
        ref = [Turn(t.start, t.end, 'speech') for t in ref]
        hyp = [Turn(t.start, t.end, 'speech') for t in hyp]
        for k, value in enumerate(_errors(ref, hyp, regions, collar, False)):
            sums[k] += value
    missed, false_alarm, _, total = sums
    if total <= 0.0:
        warn("detection_error_rate(): the reference has no speech in the time "
             "scored, so there is nothing to divide the errors by; scored nan")
        return float('nan')
    return (missed + false_alarm) / total
