# -*- coding: utf-8 -*-
"""Source separation: SI-SDR and BSS Eval's SDR, SIR and SAR over several
sources at once, the permutation that matched each estimate with its source,
and the improvement over the mixture.

A separator hands its outputs back in no particular order, so before anything
is scored each estimate has to be matched with the source it stands for.
:func:`pit_si_sdr` takes the assignment with the highest mean SI-SDR over
the sources, the criterion of utterance-level permutation invariant training
[KYTJ17] used for scoring. With the SI-SDR of every pairing in hand that is a
linear assignment problem, and :func:`scipy.optimize.linear_sum_assignment`
solves it exactly, as torchmetrics' speaker-wise PIT does. :func:`bss_eval`
runs ``bss_eval_sources`` as mir_eval 0.8.2 has it [VGF06], which tries every permutation
and keeps the one with the highest mean SIR unless it is given one. The two
rules can pick differently where a separator leaves the sources mixed, so
each returns the permutation it used, and :func:`bss_eval` takes the one
:func:`pit_si_sdr` found when a table should use one assignment throughout.

The improvement over the mixture is the score of the estimate minus the
score of the mixture itself taken as the estimate, against the same source:
SI-SNRi and SDRi as Luo and Mesgarani report Conv-TasNet [LM19], "the
additive values over the original mixture". SI-SNR there and SI-SDR here
are one quantity, both signals taken to zero mean first.

    [VGF06]  E. Vincent, R. Gribonval and C. Févotte, "Performance
             measurement in blind audio source separation", IEEE Trans.
             Audio, Speech, Lang. Process. 14(4), pp. 1462-1469, 2006,
             doi:10.1109/TSA.2005.858005.
    [YKTJ17] D. Yu, M. Kolbæk, Z.-H. Tan and J. Jensen, "Permutation
             invariant training of deep models for speaker-independent
             multi-talker speech separation", Proc. ICASSP 2017,
             pp. 241-245, doi:10.1109/ICASSP.2017.7952154.
    [KYTJ17] M. Kolbæk, D. Yu, Z.-H. Tan and J. Jensen, "Multitalker speech
             separation with utterance-level permutation invariant training
             of deep recurrent neural networks", IEEE/ACM Trans. Audio,
             Speech, Lang. Process. 25(10), pp. 1901-1913, 2017,
             doi:10.1109/TASLP.2017.2726762.
    [LR19]   J. Le Roux, S. Wisdom, H. Erdogan and J. R. Hershey, "SDR --
             half-baked or well done?", Proc. ICASSP 2019, pp. 626-630,
             doi:10.1109/ICASSP.2019.8683855.
    [LM19]   Y. Luo and N. Mesgarani, "Conv-TasNet: surpassing ideal
             time-frequency magnitude masking for speech separation",
             IEEE/ACM Trans. Audio, Speech, Lang. Process. 27(8),
             pp. 1256-1266, 2019, doi:10.1109/TASLP.2019.2915167.

Input. *refs* and *ests* are ``(sources, samples)``, a row to a source, and
a vector is one source. The mixture is one channel. A ``(samples, sources)``
array is refused rather than read the wrong way round, since no separation
task has more sources than samples. Every row is checked as a signal is
everywhere in kudio (real samples, integer PCM on its type's full scale, no
NaN), and lengths that differ are cut to the shortest with a warning, or
refused past ``kudio.core.evaluator.MAX_LENGTH_MISMATCH``, as for a pair.
"""
from __future__ import annotations

from typing import List, NamedTuple, Optional, Sequence, Tuple

import numpy as np

from kudio.core import evaluator as _evaluator
from kudio.core._policy import faint, scored_nan, signal, warn

__all__ = ['si_sdr_improvement', 'sdr_improvement', 'pit_si_sdr', 'bss_eval',
           'PitScore', 'BssEval']


class PitScore(NamedTuple):
    """What :func:`pit_si_sdr` returns. Each tuple is in the order of the
    references: entry j is about reference j and the estimate matched with
    it."""

    #: SI-SDR of each source against its estimate, dB
    si_sdr: Tuple[float, ...]
    #: the same minus the SI-SDR of the mixture, dB; None without a mixture
    si_sdri: Optional[Tuple[float, ...]]
    #: which estimate was matched with each reference
    permutation: Tuple[int, ...]


class BssEval(NamedTuple):
    """What :func:`bss_eval` returns, in the order of the references, as
    :class:`PitScore`."""

    #: source-to-distortion ratio of each source, dB
    sdr: Tuple[float, ...]
    #: source-to-interferences ratio, dB; inf for a single source, which has
    #: no other source to interfere
    sir: Tuple[float, ...]
    #: sources-to-artifacts ratio, dB
    sar: Tuple[float, ...]
    #: SDR minus the SDR of the mixture, dB; None without a mixture
    sdri: Optional[Tuple[float, ...]]
    #: which estimate was matched with each reference
    permutation: Tuple[int, ...]


# --- input ---------------------------------------------------------------------

def _sources(y, label: str, name: str) -> List[np.ndarray]:
    """*y* as a list of sources, each one channel of float64."""
    try:
        shape = np.shape(y)
    except ValueError as e:                 # rows of different lengths
        raise ValueError(f"{name}(): {label} is not an array of samples: {e}") from None
    if len(shape) == 2 and shape[1] != 1:
        if shape[0] > shape[1]:
            raise ValueError(f"{name}(): {label} has shape {tuple(shape)}: the "
                             f"sources go along the first axis, (sources, "
                             f"samples), and no separation has more sources than "
                             f"samples; pass it transposed")
        return [signal(row, f"{label}[{i}]", name) for i, row in enumerate(y)]
    if len(shape) > 2:
        raise ValueError(f"{name}(): {label} has shape {tuple(shape)}; pass "
                         f"(sources, samples), or one source as a vector")
    return [signal(y, label, name)]


def _common_length(name: str, groups: Sequence[Tuple[str, List[np.ndarray]]]
                   ) -> List[List[np.ndarray]]:
    """Every signal cut to the shortest, saying so, and refused as a pair is
    refused when that leaves too much of the longest unmeasured."""
    lengths = [len(y) for _, ys in groups for y in ys]
    n, longest = min(lengths), max(lengths)
    if n != longest:
        limit = float(_evaluator.MAX_LENGTH_MISMATCH)
        unmeasured = (longest - n) / longest
        counts = ", ".join(f"{label} {'/'.join(str(len(y)) for y in ys)}"
                           for label, ys in groups)
        if unmeasured > limit:
            raise ValueError(
                f"{name}(): the signals run to {counts} samples: cutting them "
                f"to the shortest leaves {longest - n} of the longest one's "
                f"{longest} samples unmeasured ({unmeasured * 100:.1f}%), past "
                f"the {limit * 100:.0f}% limit. Check that they belong to the "
                f"same mixture, or raise kudio.core.evaluator.MAX_LENGTH_MISMATCH "
                f"to score them as they sit")
        warn(f"{name}(): the signals run to {counts} samples: only the first "
             f"{n} of each are scored and the rest is ignored")
    return [[y[:n] for y in ys] for _, ys in groups]


def _checked(name: str, refs, ests, mixture):
    """refs, ests and the mixture, checked and cut to one length; the
    mixture None when there is none."""
    refs = _sources(refs, 'refs', name)
    ests = _sources(ests, 'ests', name)
    if len(refs) != len(ests):
        raise ValueError(f"{name}(): {len(refs)} reference source(s) and "
                         f"{len(ests)} estimate(s); each estimate is matched "
                         f"with one source, so pass as many of each")
    groups = [('refs', refs), ('ests', ests)]
    if mixture is not None:
        groups.append(('mixture', [signal(mixture, 'mixture', name)]))
    cut = _common_length(name, groups)
    signals = [(f'{label}[{i}]' if len(ys) > 1 else label, y)
               for (label, _), ys in zip(groups, cut) for i, y in enumerate(ys)]
    faint(name, signals)
    return cut[0], cut[1], (cut[2][0] if mixture is not None else None)


def _single(name: str, ref, deg, mixture):
    """ref, deg and the mixture of a one-source measure, each one channel,
    checked and cut to one length."""
    ref, deg, mixture = (signal(y, label, name) for y, label in
                         ((ref, 'ref'), (deg, 'deg'), (mixture, 'mixture')))
    (ref,), (deg,), (mixture,) = _common_length(
        name, [('ref', [ref]), ('deg', [deg]), ('mixture', [mixture])])
    faint(name, [('ref', ref), ('deg', deg), ('mixture', mixture)])
    return ref, deg, mixture


def _constant(y: np.ndarray) -> bool:
    return y.size == 0 or bool(np.all(y == y[0]))


# --- SI-SDR ----------------------------------------------------------------------

def _si_sdr_table(name: str, refs: List[np.ndarray], ests: List[np.ndarray],
                  mixture: Optional[np.ndarray]):
    """``(table, of_mixture)``, the SI-SDR of reference i against estimate j
    and of each reference against the mixture; or ``(None, reason)`` when
    there is nothing to score. A constant estimate is scored 0 dB, as
    :func:`kudio.si_sdr` scores it, and said once."""
    if refs[0].size == 0:
        return None, "the signals are empty"
    for i, ref in enumerate(refs):
        if _constant(ref):
            what = 'digital silence throughout' if not np.any(ref) else 'constant'
            label = f"refs[{i}]" if len(refs) > 1 else "ref"
            return None, (f"{label} is {what}, so once its mean is taken out "
                          f"there is nothing to project an estimate on")
    k = len(refs)
    table = np.empty((k, k))
    for i in range(k):
        for j in range(k):
            table[i, j] = _evaluator._si_sdr_value(refs[i], ests[j])
    of_mixture = (None if mixture is None else
                  np.array([_evaluator._si_sdr_value(ref, mixture) for ref in refs]))
    if not np.all(np.isfinite(table)) or (
            of_mixture is not None and not np.all(np.isfinite(of_mixture))):
        peak = max(float(np.max(np.abs(y))) for y in refs + ests
                   + ([] if mixture is None else [mixture]))
        return None, (f"the signals are too loud to square in float64 (they "
                      f"peak at {peak:.3g}); scale them all down by the same "
                      f"factor")
    flat = [j for j, est in enumerate(ests) if _constant(est)]
    if flat:
        which = (f"ests[{flat[0]}] is" if len(flat) == 1 and len(ests) > 1 else
                 "deg is" if len(ests) == 1 else
                 f"ests {', '.join(str(j) for j in flat)} are")
        warn(f"{name}(): {which} constant (digital silence or a DC offset "
             f"alone), so once the mean is taken out there is no projection and "
             f"no residual; the 0 dB is what the 1e-12 guards make of 0 / 0")
    if mixture is not None and _constant(mixture):
        warn(f"{name}(): the mixture is constant, so its SI-SDR is the 0 dB "
             f"the 1e-12 guards make of 0 / 0, and the improvement is the "
             f"estimate's SI-SDR itself")
    return table, of_mixture


def _best_assignment(table: np.ndarray) -> Tuple[int, ...]:
    """The estimate for each reference that makes the summed score largest."""
    if len(table) == 1:
        return (0,)
    from scipy.optimize import linear_sum_assignment
    rows, cols = linear_sum_assignment(table, maximize=True)
    order = np.empty(len(rows), dtype=int)
    order[rows] = cols
    return tuple(int(c) for c in order)


def pit_si_sdr(refs, ests, mixture=None) -> PitScore:
    """SI-SDR of each source, with the estimates matched to the sources so
    that the mean is the highest it can be; with the mixture, the
    improvement over it too.

    >>> s = kudio.pit_si_sdr(sources, separated, mixture)   # doctest: +SKIP
    >>> np.mean(s.si_sdri)                                   # doctest: +SKIP

    Every pairing of a reference and an estimate is scored with
    :func:`kudio.si_sdr`, both signals taken to zero mean, and the assignment
    with the largest sum is the one kept, found exactly by the Hungarian
    method. The scores of a tie are the same whichever assignment is kept.
    The SI-SDRi of a source is its SI-SDR minus the SI-SDR of the mixture
    against it, the mixture's score not depending on the assignment.

    :param refs: the sources, ``(sources, samples)``; one source can be a
        vector.
    :param ests: the separated signals, as many as *refs*, in any order.
    :param mixture: what was separated, one channel; optional.
    :raises ValueError: for input that is not ``(sources, samples)`` of real
        samples, a ``(samples, sources)`` array, unequal numbers of sources
        and estimates, NaN or inf samples, and lengths too far apart.
    :returns: a :class:`PitScore`. Every number is nan, with one warning,
        when the signals are empty, when a reference is constant (digital
        silence, or a DC offset alone), which leaves nothing to project on,
        and when they are too loud to square. A constant estimate scores
        0 dB, as :func:`kudio.si_sdr` scores it, with a warning.
    """
    refs, ests, mixture = _checked('pit_si_sdr', refs, ests, mixture)
    table, of_mixture = _si_sdr_table('pit_si_sdr', refs, ests, mixture)
    k = len(refs)
    if table is None:
        scored_nan('pit_si_sdr', of_mixture)
        nan = (float('nan'),) * k
        return PitScore(nan, None if mixture is None else nan, tuple(range(k)))
    order = _best_assignment(table)
    scores = tuple(float(table[i, order[i]]) for i in range(k))
    better = (None if of_mixture is None else
              tuple(s - float(m) for s, m in zip(scores, of_mixture)))
    return PitScore(scores, better, order)


def si_sdr_improvement(ref, deg, mixture) -> float:
    """SI-SDRi: how much *deg* gained on the *mixture* it came from, in dB of
    SI-SDR against *ref*: ``si_sdr(ref, deg) - si_sdr(ref, mixture)``.

    >>> kudio.si_sdr_improvement(clean, enhanced, noisy)    # doctest: +SKIP

    The improvement measure of enhancement and separation papers; for several
    sources at once, with the estimates matched to them, see
    :func:`pit_si_sdr`.

    :raises ValueError: as :func:`kudio.si_sdr`, for any of the three.
    :returns: the improvement in dB. nan, with one warning, as
        :func:`kudio.si_sdr` gives nan.
    """
    ref, deg, mixture = _single('si_sdr_improvement', ref, deg, mixture)
    table, of_mixture = _si_sdr_table('si_sdr_improvement', [ref], [deg], mixture)
    if table is None:
        return scored_nan('si_sdr_improvement', of_mixture)
    return float(table[0, 0]) - float(of_mixture[0])


# --- BSS Eval --------------------------------------------------------------------

def _bss_eval_sources(refs: np.ndarray, ests: np.ndarray, permute: bool):
    from kudio.core._bss import bss_eval_sources
    return bss_eval_sources(refs, ests, compute_permutation=permute)


def bss_eval(refs, ests, mixture=None, *, permutation: Optional[Sequence[int]] = None
             ) -> BssEval:
    """SDR, SIR and SAR of each source by BSS Eval [VGF06], as mir_eval's
    ``bss_eval_sources`` computes them; with the mixture, SDRi too.

    >>> s = kudio.bss_eval(sources, separated, mixture)      # doctest: +SKIP
    >>> np.mean(s.sdri)                                      # doctest: +SKIP

    Each estimate is split, by orthogonal projection, into the part a causal
    filter of 512 taps can make from its own source, what the same filters
    on all the sources add to that (interference), and the rest (artifacts).
    512 is mir_eval's length; the paper leaves it open and uses 256 at 8 kHz
    in its examples. SDR is the target's energy over everything else, SIR
    over the interference, and SAR the target and interference together over
    the artifacts, in dB (the paper's eqs. 13, 14 and 16). The numbers are
    those of mir_eval 0.8.2's ``mir_eval.separation.bss_eval_sources(refs,
    ests)`` to the bit, and mir_eval is not needed. SDRi is each source's SDR
    minus the SDR of the mixture taken as every estimate.

    :param permutation: which estimate goes with each reference, as
        :func:`pit_si_sdr` returns it. Without one, every estimate is
        decomposed against every source, ``k * k`` decompositions, and the
        permutation with the highest mean SIR is kept.
    :raises ValueError: as :func:`pit_si_sdr`, and for a *permutation* that
        is not one of the estimates' indices.
    :returns: a :class:`BssEval`. Every number is nan, with one warning,
        where BSS Eval cannot score the signals: empty ones, a reference or
        an estimate that is digital silence throughout, and signals too loud
        to square. With one source there is no interference, and its SIR is
        inf. Where the normal equations are singular, as two references of
        two samples, one twice the other, make them, they are solved by least
        squares.
    """
    refs, ests, mixture = _checked('bss_eval', refs, ests, mixture)
    return _bss('bss_eval', refs, ests, mixture, permutation)


def _bss(name: str, refs: List[np.ndarray], ests: List[np.ndarray],
         mixture: Optional[np.ndarray], permutation) -> BssEval:
    """:func:`bss_eval` on signals already checked and cut to one length."""
    k = len(refs)
    if permutation is not None:
        given = tuple(int(i) for i in permutation)
        if sorted(given) != list(range(k)):
            raise ValueError(f"{name}(): permutation has to hold each of the "
                             f"{k} estimates' indices once, got {permutation!r}")
    nan = (float('nan'),) * k
    unscored = BssEval(nan, nan, nan, None if mixture is None else nan,
                       tuple(range(k)) if permutation is None else given)
    why = None
    if refs[0].size == 0:
        why = "the signals are empty"
    else:
        for label, ys in (('refs', refs), ('ests', ests)):
            silent = [i for i, y in enumerate(ys) if not np.any(y)]
            if silent:
                which = (f"{label}[{silent[0]}]" if k > 1 else
                         'ref' if label == 'refs' else 'deg')
                why = (f"{which} is digital silence throughout, which "
                       f"bss_eval_sources refuses to score")
                break
        if why is None and mixture is not None and not np.any(mixture):
            why = "the mixture is digital silence throughout, which bss_eval_sources refuses to score"
    if why is None and _evaluator._overflows(np.concatenate(refs), np.concatenate(
            ests + ([] if mixture is None else [mixture]))):
        why = "the signals are too loud to square in float64; scale them all down by the same factor"
    if why is not None:
        scored_nan(name, why)
        return unscored
    refs_ = np.vstack(refs)
    ests_ = np.vstack(ests)
    if permutation is None:
        sdr, sir, sar, order = _bss_eval_sources(refs_, ests_, True)
        order = tuple(int(i) for i in order)
    else:
        order = given
        sdr, sir, sar, _ = _bss_eval_sources(refs_, ests_[list(order)], False)
    sdri = None
    if mixture is not None:
        of_mixture, *_ = _bss_eval_sources(refs_, np.tile(mixture, (k, 1)), False)
        sdri = tuple(float(a) - float(b) for a, b in zip(sdr, of_mixture))
    return BssEval(tuple(float(v) for v in sdr), tuple(float(v) for v in sir),
                   tuple(float(v) for v in sar), sdri, order)


def sdr_improvement(ref, deg, mixture) -> float:
    """SDRi of one source: the BSS Eval SDR of *deg* minus that of the
    *mixture*, both against *ref*, in dB.

    >>> kudio.sdr_improvement(clean, enhanced, noisy)        # doctest: +SKIP

    The SDR is the one ``kudio.score(..., ['sdr'])`` gives, mir_eval's
    ``bss_eval_sources`` with a single source. For several sources, where the
    other sources count as interference, see :func:`bss_eval`.

    :raises ValueError: as :func:`si_sdr_improvement`.
    :returns: the improvement in dB; nan, with one warning, as
        :func:`bss_eval` gives nan.
    """
    ref, deg, mixture = _single('sdr_improvement', ref, deg, mixture)
    return _bss('sdr_improvement', [ref], [deg], mixture, None).sdri[0]
