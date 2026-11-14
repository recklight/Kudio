# -*- coding: utf-8 -*-
"""Word and character error rates: how far a recogniser's transcript is from
the reference transcript, counted in words (:func:`wer`) or in characters
(:func:`cer`), with the counts behind each from :func:`word_errors` and
:func:`char_errors`.

The hypothesis, what the recogniser wrote, is aligned with the reference, what
was said, so that as few words as possible are substituted (S), deleted (D)
or inserted (I), and the rate is those edits over the length of the
reference::

    WER = (S + D + I) / N,        N = H + S + D

H being the hits, the reference words the alignment pairs with the same word.
The fewest edits is the Levenshtein distance between the two word sequences,
the minimum edit distance of [JM] Sec. 2.9 with every edit costing 1, on
which [JM] Sec. 16.6 defines WER. CER is the same count over characters. A
corpus is scored as Kaldi's compute-wer and jiwer score one: the errors of
all its pairs over the reference words of all of them. The sources::

    [JM]     D. Jurafsky and J. H. Martin, "Speech and Language Processing",
             3rd ed., draft of 19 August 2026: Ch. 2, "Words and Tokens",
             Sec. 2.9, minimum edit distance, pp. 28-32; Ch. 16, "Automatic
             Speech Recognition", Sec. 16.6, word error rate, pp. 21-24.
             https://web.stanford.edu/~jurafsky/slp3/
    [SCTK]   NIST Speech Recognition Scoring Toolkit, the sclite manual,
             doc/sclite.htm, master branch as read on 2026-11-12.
             https://github.com/usnistgov/SCTK
    [Kaldi]  src/bin/compute-wer.cc, src/util/edit-distance-inl.h,
             src/util/kaldi-holder-inl.h and
             egs/wsj/s5/steps/scoring/score_kaldi_cer.sh, master branch as
             read on 2026-11-12. https://github.com/kaldi-asr/kaldi
    [jiwer]  jiwer 2.3.0, jiwer/measures.py; jiwer 3.1.0 and 4.0.0,
             src/jiwer/process.py, transformations.py and transforms.py.
             https://github.com/jitsi/jiwer. Its alignments come from
             rapidfuzz, rapidfuzz/distance/Levenshtein_impl.hpp in
             https://github.com/rapidfuzz/rapidfuzz-cpp, main branch as read
             on 2026-11-12.
    [ESPnet] espnet2/legacy/nets/e2e_asr_common.py (ErrorCalculator),
             egs2/TEMPLATE/asr1/asr.sh and espnet2/text/char_tokenizer.py,
             master branch as read on 2026-11-12.
             https://github.com/espnet/espnet

On the worked examples of [JM] these give the published counts: 6
substitutions, 1 deletion and 3 insertions in the 13 reference words of the
CallHome utterance in Sec. 16.6, a WER of 76.9%, and the 5 edits between
"intention" and "execution" in Sec. 2.9, split as its Fig. 2.17 splits them.

**The total is fixed, the split is not.** Reference "a c" against hypothesis
"d d a" is three edits either as two substitutions and an insertion, or as
two insertions, a hit on "a" and a deletion. Of the alignments with the
fewest errors I take the one with the most hits. Since
``2H + S = N + M - (S + D + I)``, M being the length of the hypothesis, that
is also the one with the fewest substitutions, and it belongs to the pair:
the four counts are the same whichever way the table is filled. A tool that
builds one alignment instead picks among equally cheap steps by a fixed
order, and its counts follow that order. compute-wer takes the diagonal only
when it is strictly cheaper than both other steps, then a deletion only when
it is strictly cheaper than an insertion, and an insertion otherwise, so it
reports the pair above as 2 substitutions and 1 insertion where kudio reports
1 hit, 1 deletion and 2 insertions. jiwer counts the alignment rapidfuzz
recovers from its own table, with an order of preference of its own. The WER
agrees; a table of S, D and I from two tools need not. How often they part depends on how often
the text has ties: never in the 5,000 utterances of the simulated corpus in
``tools/validation/wer.py`` (100,000 words, about 10% of them wrong), and on
618 of 18,420 random pairs of up to 12 words drawn from four, for
compute-wer.

**sclite can count more errors.** It minimises a weighted distance: 0 for a
hit, 3 for an insertion or a deletion and 4 for a substitution ([SCTK],
"Dynamic Programming string alignment"), which is ``3 (S + D + I) + S``.
Among the alignments with the fewest errors that picks the one with the
fewest substitutions, so wherever sclite's alignment has the fewest errors,
its four counts are the ones here. But it trades one more error for five
fewer substitutions. Against the reference "one two three four five", the
hypothesis "four five six seven eight" is 5 substitutions here, a WER of 1.0;
sclite's weights pair up the two "four five" at a cost of 18 against 20, for
3 deletions and 3 insertions, 6 errors and a WER of 1.2. That happens on none
of the simulated utterances, and on 59 of the 18,420 short pairs, with 711
more where it depends on how sclite breaks its own ties.

**Words and characters.** A word is whatever ``str.split()`` separates: runs
of any whitespace split, tabs, line breaks and the full-width space U+3000
among them. compute-wer splits on ASCII whitespace. jiwer splits on the
space character, after it collapses runs of two or more whitespace characters
into one space, so a lone tab or line break there joins two words into one:
"hello world" against "hello\\nworld" is a WER of 1.0 in jiwer and 0 here. A
character is a Unicode code point, as Python's ``str`` holds it.

For CER the tools disagree about spaces. Kaldi's score_kaldi_cer.sh splits
each word into its characters and drops the spaces, and so does ESPnet's
ErrorCalculator, which logs CER during training. jiwer counts every character
from the first that is not whitespace to the last, spaces included, and so
does the CER that ESPnet's asr.sh reports, which turns each space into a
``<space>`` token for sclite. I leave whitespace out by default. CER is the
measure for Chinese and Japanese, which are written without spaces; there a
space is layout, and counting it would make "我 愛 你" against "我愛你" 2
errors in 5 characters where nothing was misheard. ``spaces=True`` counts one
space between each two words however wide the gap, which is what jiwer counts
on text with single spaces between its words; jiwer counts each extra space,
and a tab, as a character of its own.

What to watch for:

* Nothing is normalised. Case, punctuation, digits against numbers written
  out, contractions and filler words all count as they stand: "Hello, world."
  against "hello world" is 2 substitutions, a WER of 1.0. Normalise both sides
  the same way before scoring, and say how next to the number. Lower-casing
  and turning punctuation into spaces, for example::

      import re
      def normalise(text):
          return re.sub(r"[^\\w\\s']", " ", text.lower())
      kudio.wer([normalise(t) for t in refs], [normalise(t) for t in hyps])

* Unicode is taken as it comes, with no normalisation form applied: "é" as one
  code point and as "e" followed by a combining accent are different strings,
  and ``cer("café", "cafe\\u0301")`` is 2 errors in 4 characters, 0.5. Run
  ``unicodedata.normalize("NFC", text)`` over both sides when they may come
  from different sources; "NFKC" also folds full-width letters and digits
  (Ａ, １) into ASCII. Simplified and traditional characters stay different
  too: "我爱你" against "我愛你" is a CER of 1/3. And a code point is not
  always one letter on screen: an accent written as a combining mark, a
  Devanagari syllable or an emoji sequence counts as several.
* A corpus rate is pooled, not averaged. One error in a 2-word utterance and
  none in an 18-word one is 1/20 = 0.05 pooled, and 0.25 as the mean of the
  two utterances' rates, which gives the short one the weight of the long
  one. Pass the whole corpus in one call rather than averaging calls.
* WER is not a share of the words and has no upper bound, since insertions
  can outnumber the reference: "yes" against "yes yes yes" is 2 insertions
  over 1 word, a WER of 2.0.
* Text without spaces between its words is one word to :func:`wer`:
  "今天天氣很好" against "今天天氣不好" is a WER of 1.0 and a CER of 1/6. Use
  :func:`cer` for Chinese and Japanese, or segment the words first and join
  them with spaces.
* An empty reference has nothing to divide by. Inside a corpus that does not
  matter: its insertions count, as in compute-wer and jiwer 4.0, and
  ``wer(["a b", ""], ["a b", "c"])`` is 1/2. When the whole reference side is
  empty, :func:`wer` and :func:`cer` warn and return nan; jiwer 4.0 returns
  the number of insertions there (0 for an empty pair), jiwer 2.3 to 3.1
  raise as soon as any one reference is empty, and compute-wer prints nan or
  inf. :func:`word_errors` and :func:`char_errors` return the counts either
  way.
* A list is a corpus of transcripts, paired in order. A sentence already cut
  into words, ``["the", "cat"]``, would be read as two one-word transcripts,
  each aligned on its own; join the tokens with spaces first. A dict is read
  as utterance id to transcript and paired by key, as compute-wer pairs them.
  A pandas Series is paired by position, like a list, so two Series have to
  carry the same index in the same order or the call is refused: the same
  two transcripts with the hypotheses in the other order, their index
  following them, would score 1.0 by position and 0 by index. ``.to_dict()``
  of both pairs them by index. A DataFrame is refused as well, since it
  iterates over its column names; pass its column of transcripts.
* The work grows with the product of the two lengths of each pair. On a
  desktop CPU the simulated corpus, 100,000 words in 5,000 utterances, is
  scored in under 0.1 s, and its 530,000 characters in about 0.7 s; one pair
  of 10,000-word transcripts takes about 0.3 s, and one of 100,000 words about
  25 s.
"""
from __future__ import annotations

import itertools
from collections.abc import Iterable, Mapping
from typing import Any, Callable, List, NamedTuple, Sequence, Tuple, Union

import numpy as np

from kudio.core._policy import warn

__all__ = ['wer', 'cer', 'word_errors', 'char_errors', 'ErrorCounts']

#: Tables of this many cells or more, once the shared start and end of the
#: pair are set aside, are filled a row at a time with numpy; smaller ones in
#: plain Python. A numpy call costs a few microseconds however short its row,
#: and the two ways take about the same time at 48 tokens a side.
_NUMPY_CELLS = 48 * 48


class ErrorCounts(NamedTuple):
    """The counts behind a WER or a CER, from :func:`word_errors` or
    :func:`char_errors`, summed over every pair of a corpus.

    ``hits + substitutions + deletions`` is the length of the reference and
    ``hits + substitutions + insertions`` that of the hypothesis; the rate is
    ``errors / reference_length``. Of the alignments with the fewest errors,
    the counts are those of the one with the most hits. Why, and why
    another tool's split can differ while its total agrees:
    https://recklight.github.io/Kudio/guide/task-metrics/#which-errors-and-what-counts-as-a-word
    """

    #: reference tokens aligned with the same token of the hypothesis
    hits: int
    #: reference tokens aligned with a different token
    substitutions: int
    #: reference tokens with nothing aligned to them
    deletions: int
    #: hypothesis tokens with nothing aligned to them
    insertions: int
    #: tokens in the reference, ``hits + substitutions + deletions``
    reference_length: int

    @property
    def errors(self) -> int:
        """``substitutions + deletions + insertions``, the edit distance."""
        return self.substitutions + self.deletions + self.insertions


# --- reading the input --------------------------------------------------------

def _refuse_side(value: Any, side: str, name: str) -> None:
    """Refuse a whole side that is neither text nor a collection of texts."""
    if isinstance(value, (bytes, bytearray)):
        raise ValueError(f"{name}(): {side} is bytes; decode it first, "
                         f"e.g. {side}.decode('utf-8')")
    if isinstance(value, (set, frozenset)):
        raise ValueError(f"{name}(): {side} is a set, which has no order to "
                         f"pair transcripts by; pass a list, or a dict from "
                         f"utterance id to transcript")
    if not isinstance(value, Iterable):
        raise ValueError(f"{name}(): {side} has to be a str, a sequence of "
                         f"str or a dict from utterance id to str, got "
                         f"{type(value).__name__}")
    ndim = getattr(value, 'ndim', 1)
    if isinstance(ndim, int) and ndim > 1:
        # a DataFrame iterates over its column names, which would be scored
        raise ValueError(f"{name}(): {side} is a {ndim}-d table; pass the "
                         f"one column that holds the transcripts, e.g. "
                         f"table['text']")


def _labels(value: Any) -> Any:
    """The index of a pandas Series, or None for a side without one. A list
    or a tuple has an ``index`` too, but it is a method."""
    labels = getattr(value, 'index', None)
    return None if labels is None or callable(labels) else labels


def _same_labels(a: Any, b: Any) -> bool:
    """Same labels in the same order, whatever the dtype of either index
    (in pandas 2.3, ``Index.equals`` tells an object index from a string one
    with the same labels)."""
    try:
        return bool(list(a) == list(b))
    except TypeError:       # pandas NA against a label has no truth value
        return False


def _refuse_item(text: Any, where: str, side: str, name: str) -> None:
    """Refuse one transcript that is not a str, saying what to pass instead."""
    if isinstance(text, (bytes, bytearray)):
        hint = "decode it first, e.g. .decode('utf-8')"
    elif ((isinstance(text, float) and text != text)
          or type(text).__name__ == 'NAType'):
        hint = ("pass '' for an empty transcript; pandas reads an empty cell "
                "as NaN or NA, so fillna('') first")
    elif text is None:
        hint = "pass '' for an empty transcript"
    elif isinstance(text, (list, tuple)):
        hint = "join a transcript's tokens with spaces first: ' '.join(tokens)"
    else:
        hint = "pass the transcript as text"
    shown = repr(text)
    if len(shown) > 60:
        shown = shown[:57] + "..."
    raise ValueError(f"{name}(): {side}{where} is {type(text).__name__} "
                     f"{shown}, not a str; {hint}")


def _examples(keys: List[Any]) -> str:
    shown = ", ".join(repr(k) for k in keys[:3])
    return shown + (", ..." if len(keys) > 3 else "")


def _listed(value: Any, side: str, name: str) -> List[Any]:
    try:
        return list(value)
    except TypeError as e:          # a 0-d numpy array, for one
        raise ValueError(f"{name}(): {side} cannot be read as a sequence of "
                         f"transcripts ({e}); pass a str or a list of str"
                         ) from e


def _pairs(reference: Any, hypothesis: Any, name: str) -> List[Tuple[str, str]]:
    """The input as ``(reference, hypothesis)`` text pairs, once it is clear
    that the two sides pair up."""
    if isinstance(reference, str) and isinstance(hypothesis, str):
        return [(reference, hypothesis)]
    _refuse_side(reference, 'reference', name)
    _refuse_side(hypothesis, 'hypothesis', name)
    if isinstance(reference, str) or isinstance(hypothesis, str):
        one, other = (('reference', 'hypothesis') if isinstance(reference, str)
                      else ('hypothesis', 'reference'))
        raise ValueError(f"{name}(): {one} is one transcript and {other} a "
                         f"collection of them; pass both as str, or both as "
                         f"lists in the same order")
    if isinstance(reference, Mapping) != isinstance(hypothesis, Mapping):
        raise ValueError(f"{name}(): one side is a dict and the other is not, "
                         f"so the transcripts cannot be paired by key; pass "
                         f"both as dicts from utterance id to transcript, or "
                         f"both as lists in the same order")
    if isinstance(reference, Mapping):
        unheard = [k for k in reference if k not in hypothesis]
        unasked = [k for k in hypothesis if k not in reference]
        if unheard or unasked:
            parts = []
            if unheard:
                parts.append(f"{len(unheard)} reference key(s) have no "
                             f"hypothesis ({_examples(unheard)})")
            if unasked:
                parts.append(f"{len(unasked)} hypothesis key(s) have no "
                             f"reference ({_examples(unasked)})")
            raise ValueError(f"{name}(): {' and '.join(parts)}; give both "
                             f"the same keys, with '' for an utterance the "
                             f"recogniser returned nothing for")
        keys = list(reference)
        refs = [reference[k] for k in keys]
        hyps = [hypothesis[k] for k in keys]
        where = [f"[{k!r}]" for k in keys]
    else:
        refs = _listed(reference, 'reference', name)
        hyps = _listed(hypothesis, 'hypothesis', name)
        if len(refs) != len(hyps):
            raise ValueError(f"{name}(): {len(refs)} reference transcript(s) "
                             f"and {len(hyps)} hypothesis transcript(s); they "
                             f"are paired in order, so both sides need the "
                             f"same number, with '' for an utterance the "
                             f"recogniser returned nothing for")
        ref_labels, hyp_labels = _labels(reference), _labels(hypothesis)
        if (ref_labels is not None and hyp_labels is not None
                and not _same_labels(ref_labels, hyp_labels)):
            raise ValueError(f"{name}(): the two sides carry different "
                             f"indexes, and a Series is paired by position, "
                             f"not by its index; pass .to_dict() of both to "
                             f"pair them by utterance id, or .tolist() of "
                             f"both to pair them by position")
        where = [f"[{i}]" for i in range(len(refs))]
    for side, texts in (('reference', refs), ('hypothesis', hyps)):
        for text, at in zip(texts, where):
            if not isinstance(text, str):
                _refuse_item(text, at, side, name)
    return list(zip(refs, hyps))


def _spaces(spaces: Any, name: str) -> bool:
    if not isinstance(spaces, (bool, np.bool_)):
        raise ValueError(f"{name}(): spaces has to be True or False, got "
                         f"{spaces!r}")
    return bool(spaces)


def _words(text: str) -> List[str]:
    return text.split()


def _characters(spaces: bool) -> Callable[[str], str]:
    """The text as the characters CER counts: whitespace left out, or, with
    *spaces*, one space between each two words and none at the ends."""
    if spaces:
        return lambda text: ' '.join(text.split())
    return lambda text: ''.join(text.split())


# --- the alignment ------------------------------------------------------------

def _cost_python(a: Sequence, b: Sequence, weight: int) -> int:
    """Lowest cost of aligning *a* with *b*, a hit costing -1 and every
    substitution, deletion and insertion *weight*, filled one cell at a time.

    With *weight* above the most hits the pair can have, one error more
    always outweighs any number of hits, and the cost is
    ``weight * errors - hits`` of the alignment with the fewest errors and,
    among those, the most hits.

    Where the two tokens match, the diagonal is never worse than the other
    two steps, so it is taken without comparing. The step from above starts
    from the best alignment of ``a[:i - 1]`` with ``b[:j]``. Dropping
    ``b[j - 1]`` from it leaves an alignment of ``a[:i - 1]`` with
    ``b[:j - 1]`` that costs at most ``weight + 1`` more, a hit or a
    substitution of ``b[j - 1]`` turning into a deletion of its partner, so
    the diagonal, at -1 on top of the best such alignment, is no dearer. The
    step from the left goes the same way with the two sides swapped.
    """
    prev = list(range(0, (len(b) + 1) * weight, weight))
    for i, x in enumerate(a, 1):
        left = i * weight
        row = [left]
        diag = prev[0]
        for y, up in zip(b, itertools.islice(prev, 1, None)):
            if x == y:
                left = diag - 1
            else:
                if up < diag:
                    diag = up
                if left < diag:
                    diag = left
                left = diag + weight
            row.append(left)
            diag = up
        prev = row
    return prev[-1]


def _cost_rows(a: Sequence, b: Sequence, weight: int) -> int:
    """:func:`_cost_python`, a row at a time with numpy.

    The row is kept as ``g[j] = cur[j] - weight * j``. In that frame the
    step from the left, ``cur[j - 1] + weight``, is just ``g[j - 1]``, so a
    row is a running minimum, one ``np.minimum.accumulate``, of
    ``min(h[j - 1], h[j] + weight)``, h being the row above in the same
    frame; a substitution on the diagonal costs nothing extra there. Where
    the tokens match, the diagonal wins outright at ``h[j - 1] - weight - 1``
    (see :func:`_cost_python`), and only those cells are written on their
    own, at positions found once by sorting *b*.
    """
    ids: dict = {}
    a_ids = np.array([ids.setdefault(t, len(ids)) for t in a], dtype=np.int64)
    b_ids = np.array([ids.setdefault(t, len(ids)) for t in b], dtype=np.int64)
    # the positions in b of the token in row i: order[first[i]:last[i]]
    order = np.argsort(b_ids, kind='stable')
    ranked = b_ids[order]
    first = np.searchsorted(ranked, a_ids, 'left').tolist()
    last = np.searchsorted(ranked, a_ids, 'right').tolist()
    above = np.zeros(len(b_ids) + 1, dtype=np.int64)   # row 0: cur[j] = weight * j
    row = np.empty_like(above)
    down = np.empty(len(b_ids), dtype=np.int64)
    hit = -(weight + 1)
    for i in range(len(a_ids)):
        np.add(above[1:], weight, out=down)
        np.minimum(above[:-1], down, out=row[1:])
        if last[i] > first[i]:
            where = order[first[i]:last[i]]
            row[where + 1] = above[where] + hit
        row[0] = (i + 1) * weight
        np.minimum.accumulate(row, out=row)
        above, row = row, above
    return int(above[-1]) + weight * len(b_ids)


def _align(ref: Sequence, hyp: Sequence) -> Tuple[int, int, int, int]:
    """``(hits, substitutions, deletions, insertions)`` of the alignment of
    *hyp* with *ref* that has the fewest errors and, among those, the most
    hits.

    A start or an end the two share is all hits in such an alignment (the
    same argument as for a matching cell in :func:`_cost_python`), so it is
    counted and set aside before the table is filled; a good transcript
    leaves a small table.
    """
    n, m = len(ref), len(hyp)
    limit = min(n, m)
    start = 0
    while start < limit and ref[start] == hyp[start]:
        start += 1
    end = 0
    while end < limit - start and ref[n - 1 - end] == hyp[m - 1 - end]:
        end += 1
    a, b = ref[start:n - end], hyp[start:m - end]
    shared = start + end
    if not a or not b:
        return shared, 0, len(a), len(b)
    weight = min(len(a), len(b)) + 1
    # rows over the shorter side: fewer, longer rows. The cost is the same
    # either way round, since deletions and insertions weigh the same.
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    fill = _cost_rows if len(a) * len(b) >= _NUMPY_CELLS else _cost_python
    cost = fill(short, long, weight)
    errors = -(-cost // weight)
    hits = weight * errors - cost
    # S + D = N - H, D - I = N - M and S + D + I = errors give the rest
    insertions = errors - len(a) + hits
    deletions = insertions + len(a) - len(b)
    substitutions = len(a) - hits - deletions
    return shared + hits, substitutions, deletions, insertions


def _tally(pairs: List[Tuple[str, str]],
           tokens: Callable[[str], Sequence]) -> ErrorCounts:
    hits = substitutions = deletions = insertions = 0
    for ref_text, hyp_text in pairs:
        h, s, d, i = _align(tokens(ref_text), tokens(hyp_text))
        hits += h
        substitutions += s
        deletions += d
        insertions += i
    return ErrorCounts(hits, substitutions, deletions, insertions,
                       hits + substitutions + deletions)


def _number(count: int, unit: str) -> str:
    return f"{count} {unit}" + ("" if count == 1 else "s")


def _rate(counts: ErrorCounts, pairs: int, name: str, unit: str,
          kind: str) -> float:
    """``errors / reference_length``, or nan with a warning when the
    reference side holds no token to divide by.

    :param kind: the tokens as the message names them, e.g. "words".
    """
    if counts.reference_length:
        return counts.errors / counts.reference_length
    if pairs == 0:
        why = f"got no transcripts, so there is no reference {unit}"
    elif pairs == 1:
        why = f"the reference has no {kind}"
    else:
        why = f"none of the {pairs} references has any {kind}"
    if counts.insertions == 1:
        why += f" (the one {unit} on the hypothesis side would be an insertion)"
    elif counts.insertions:
        why += (f" (the {_number(counts.insertions, unit)} on the hypothesis "
                f"side would all be insertions)")
    warn(f"{name}(): {why}, and a rate needs at least 1 to divide "
         f"by; scored nan")
    return float('nan')


# --- the measures -------------------------------------------------------------

def word_errors(reference: Union[str, Sequence[str], Mapping[Any, str]],
                hypothesis: Union[str, Sequence[str], Mapping[Any, str]]
                ) -> ErrorCounts:
    """Hits, substitutions, deletions and insertions of words in
    *hypothesis* against *reference*, summed over the pairs of a corpus.

    >>> kudio.word_errors("the cat sat on the mat", "the cat sat on mat")
    ErrorCounts(hits=5, substitutions=0, deletions=1, insertions=0, reference_length=6)

    Words are split on whitespace and compared as they stand, case and
    punctuation included. Of the alignments with the fewest errors, the
    counts are those of the one with the most hits. Why, and how another
    tool can split the same total differently:
    https://recklight.github.io/Kudio/guide/task-metrics/#which-errors-and-what-counts-as-a-word

    :param reference: what was said: one transcript, a sequence of them, or a
        dict from utterance id to transcript.
    :param hypothesis: what the recogniser wrote, in the same form as
        *reference*: paired in order for sequences, by key for dicts.
    :raises ValueError: for a side that is not text, a sequence of text or a
        dict of text (bytes, None, a set, a number, a DataFrame or another
        2-d table); for a transcript inside one that is not a str (None,
        NaN, bytes, a list of tokens); for one side a single str and the
        other a collection; and for sides that do not pair up, sequences of
        different lengths, dicts with different keys or two pandas Series
        with different indexes.
    :returns: an :class:`ErrorCounts`. A corpus with no reference words, or
        no pairs at all, has its counts returned like any other; it is the
        rate that cannot be taken from them.
    """
    return _tally(_pairs(reference, hypothesis, 'word_errors'), _words)


def char_errors(reference: Union[str, Sequence[str], Mapping[Any, str]],
                hypothesis: Union[str, Sequence[str], Mapping[Any, str]], *,
                spaces: bool = False) -> ErrorCounts:
    """Hits, substitutions, deletions and insertions of characters in
    *hypothesis* against *reference*, summed over the pairs of a corpus.

    >>> kudio.char_errors("我愛你", "我恨你")
    ErrorCounts(hits=2, substitutions=1, deletions=0, insertions=0, reference_length=3)

    A character is a Unicode code point, and nothing is normalised: an
    accent written as a combining mark is a character of its own, so
    ``cer("café", "cafe\\u0301")`` is 0.5; a full-width letter is not its
    ASCII one; and "我爱你" against "我愛你" is 1/3. ``unicodedata.normalize``
    over both sides takes care of the first with "NFC" and of the second as
    well with "NFKC"; nothing folds simplified into traditional.

    :param reference: what was said, in any form :func:`word_errors` takes.
    :param hypothesis: what the recogniser wrote, in the same form.
    :param spaces: ``False`` (default) leaves all whitespace out, as Kaldi's
        CER recipe and ESPnet's training-time CER do. ``True`` counts one
        space between each two words, however the gap is written, and none
        at either end, as jiwer counts text with single spaces.
    :raises ValueError: as :func:`word_errors`, and for *spaces* other than
        True or False.
    :returns: an :class:`ErrorCounts` of characters.
    """
    tokens = _characters(_spaces(spaces, 'char_errors'))
    return _tally(_pairs(reference, hypothesis, 'char_errors'), tokens)


def wer(reference: Union[str, Sequence[str], Mapping[Any, str]],
        hypothesis: Union[str, Sequence[str], Mapping[Any, str]]) -> float:
    """Word error rate of *hypothesis* against *reference*,
    ``(S + D + I) / N``, as a fraction (0.25, not 25%). 0 is a perfect
    transcript and lower is better; insertions can take it above 1.

    >>> kudio.wer("the cat sat on the mat", "the cat sat on mat")
    0.16666666666666666
    >>> kudio.wer(["a b", "c d e"], ["a x", "c d e"])        # 1 error, 5 words
    0.2

    A sequence or a dict of transcripts is one corpus: the errors of all its
    pairs over the reference words of all of them. Words are split on
    whitespace and nothing is normalised, so lower-case and strip
    punctuation first if case and punctuation should not count; one way is at
    https://recklight.github.io/Kudio/guide/task-metrics/#which-errors-and-what-counts-as-a-word

    :param reference: what was said: one transcript, a sequence of them, or a
        dict from utterance id to transcript.
    :param hypothesis: what the recogniser wrote, in the same form: paired in
        order for sequences, by key for dicts.
    :raises ValueError: as :func:`word_errors`.
    :returns: the rate. nan, with a warning, when the reference side has no
        words at all (an empty reference, references that are all empty, or
        no pairs), since a rate needs at least 1 reference word to divide
        by. An empty reference among others is scored, its insertions
        counted.
    """
    pairs = _pairs(reference, hypothesis, 'wer')
    return _rate(_tally(pairs, _words), len(pairs), 'wer', 'word', 'words')


def cer(reference: Union[str, Sequence[str], Mapping[Any, str]],
        hypothesis: Union[str, Sequence[str], Mapping[Any, str]], *,
        spaces: bool = False) -> float:
    """Character error rate of *hypothesis* against *reference*,
    ``(S + D + I) / N`` over characters, as a fraction. 0 is a perfect
    transcript and lower is better; insertions can take it above 1.

    >>> kudio.cer("今天天氣很好", "今天天氣不好")
    0.16666666666666666
    >>> kudio.cer("the cat sat", "the hat sat")               # 1 of 9 letters
    0.1111111111111111
    >>> kudio.cer("the cat sat", "the hat sat", spaces=True)  # 1 of 11
    0.09090909090909091

    A corpus is pooled as in :func:`wer`. The two settings of *spaces* give
    different numbers on the same text, since the spaces join the
    denominator too, so say which one a table reports.

    :param reference: what was said, in any form :func:`wer` takes.
    :param hypothesis: what the recogniser wrote, in the same form.
    :param spaces: ``False`` (default) leaves all whitespace out, as Kaldi's
        CER recipe and ESPnet's training-time CER do, so a hypothesis spaced
        differently from its reference costs nothing. ``True`` counts one
        space between each two words, however the gap is written, and none
        at either end: a missing or an extra word boundary is then an error,
        as it is in jiwer.
    :raises ValueError: as :func:`word_errors`, and for *spaces* other than
        True or False.
    :returns: the rate. nan, with a warning, when the reference side has no
        characters other than whitespace, since a rate needs at least 1 to
        divide by.
    """
    tokens = _characters(_spaces(spaces, 'cer'))
    pairs = _pairs(reference, hypothesis, 'cer')
    return _rate(_tally(pairs, tokens), len(pairs), 'cer', 'character',
                 'characters other than whitespace')
