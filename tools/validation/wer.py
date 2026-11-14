# -*- coding: utf-8 -*-
"""Checks kudio's word and character error rates (kudio/core/wer.py) against
counters written here for the purpose, against the worked examples of
Jurafsky and Martin, and against the rules other tools follow, then times
them.

Run it from anywhere; it imports the kudio of the checkout it sits in::

    python tools/validation/wer.py

It needs what kudio itself needs and nothing more. If nltk is installed (I
used 3.9.2), its ``edit_distance``, written by others, is a third check on
the totals; without nltk that row says it was skipped.

What it compares:

1. The worked examples of Speech and Language Processing (Jurafsky and
   Martin, 3rd ed., draft of 19 August 2026): the CallHome utterance of
   Sec. 16.6 (6 substitutions, 3 insertions, 1 deletion in 13 reference
   words, 76.9%); the NIST example in the same section, whose four regions
   hold 2, 0, 1 and 1 errors for system A and 0, 1, 2 and 0 for system B;
   and "intention" against "execution" of Sec. 2.9, 5 edits with unit costs,
   split as Fig. 2.17 splits them, and 8 with substitutions at 2.
2. Every alignment. For small pairs, every alignment there is gets listed
   and the one with the fewest errors and then the most hits is picked out:
   all pairs of up to 4 tokens over two symbols, and random pairs of up to 6.
3. Full tables: the textbook forward table, a second one filled from the
   end, the steps that lie on some shortest path (forward cost + step +
   backward cost = distance), the most hits over those steps, and one such
   alignment traced back and checked token by token. None of it is shared
   with kudio, which keeps two rows of a single weighted cost.
4. kudio's two ways of filling the table, plain Python and numpy rows,
   against each other.
5. A corpus against the sum of its pairs, a dict against a list, and CER on
   Chinese with and without spaces.
6. Other tools' rules, on the simulated corpus below and on short random
   pairs over four symbols, where ties are common: the tie order of Kaldi's
   compute-wer (src/util/edit-distance-inl.h), sclite's weights of 3 for an
   insertion or a deletion and 4 for a substitution (SCTK doc/sclite.htm),
   and how jiwer 4.0.0 cuts text into words and characters
   (src/jiwer/transformations.py and transforms.py), each written here from
   that source, or for sclite from its manual. The pairs the module notes
   quote are among them.
7. Speed, on a simulated corpus of 100,000 words in 5,000 utterances with
   about 10% of the words wrong, on the two fills around the size where
   kudio switches between them, and on single pairs of 10,000 and of
   100,000 words (the last takes about half a minute).

Every count is an integer, so the table printed at the end should hold
nothing but zeros. Rates are compared to 1e-15.
"""
from __future__ import annotations

import importlib
import itertools
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kudio.core.wer import cer, char_errors, wer, word_errors  # noqa: E402

# the module, for its private parts: once kudio.core exports the function wer,
# ``from kudio.core import wer`` gives the function instead
module = importlib.import_module("kudio.core.wer")

#: (check, cases, largest count difference, largest rate difference)
RESULTS = []


def report(check, cases, count_diff=0, rate_diff=0.0):
    RESULTS.append((check, cases, count_diff, rate_diff))


def apart(got, want):
    """The largest difference between two tuples of counts."""
    return max((abs(x - y) for x, y in zip(got, want)), default=0)


# --- the counters written for checking --------------------------------------

def every_alignment(a, b):
    """Every alignment of *a* with *b*, as (hits, subs, dels, ins)."""
    if not a:
        yield 0, 0, 0, len(b)
        return
    if not b:
        yield 0, 0, len(a), 0
        return
    hit = a[0] == b[0]
    for h, s, d, i in every_alignment(a[1:], b[1:]):
        yield (h + 1, s, d, i) if hit else (h, s + 1, d, i)
    for h, s, d, i in every_alignment(a[1:], b):
        yield h, s, d + 1, i
    for h, s, d, i in every_alignment(a, b[1:]):
        yield h, s, d, i + 1


def by_enumeration(a, b):
    """The fewest errors, then the most hits, out of every alignment."""
    return min(every_alignment(a, b), key=lambda c: (c[1] + c[2] + c[3], -c[0]))


def by_tables(a, b):
    """The same from full tables, with one alignment traced back.

    F[i][j] is the edit distance of a[:i] and b[:j] with unit costs, G[i][j]
    that of a[i:] and b[j:]. A step from (i, j) lies on a shortest path when
    F[i][j] + its cost + G at its far end equals the distance; the most hits
    over such paths is then a longest path over those steps.
    """
    n, m = len(a), len(b)
    F = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 or j == 0:
                F[i][j] = i + j
            else:
                F[i][j] = min(F[i - 1][j] + 1, F[i][j - 1] + 1,
                              F[i - 1][j - 1] + (a[i - 1] != b[j - 1]))
    G = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n, -1, -1):
        for j in range(m, -1, -1):
            if i == n or j == m:
                G[i][j] = (n - i) + (m - j)
            else:
                G[i][j] = min(G[i + 1][j] + 1, G[i][j + 1] + 1,
                              G[i + 1][j + 1] + (a[i] != b[j]))
    distance = F[n][m]
    assert G[0][0] == distance
    best = [[None] * (m + 1) for _ in range(n + 1)]
    back = {}
    best[0][0] = 0
    for i in range(n + 1):
        for j in range(m + 1):
            here = best[i][j]
            if here is None:
                continue
            steps = []
            if i < n and j < m:
                same = a[i] == b[j]
                steps.append((i + 1, j + 1, 0 if same else 1, 'hit' if same else 'sub'))
            if i < n:
                steps.append((i + 1, j, 1, 'del'))
            if j < m:
                steps.append((i, j + 1, 1, 'ins'))
            for ti, tj, cost, op in steps:
                if F[i][j] + cost + G[ti][tj] != distance:
                    continue
                gain = here + (op == 'hit')
                if best[ti][tj] is None or gain > best[ti][tj]:
                    best[ti][tj] = gain
                    back[ti, tj] = (i, j, op)
    ops = []
    i, j = n, m
    while (i, j) != (0, 0):
        pi, pj, op = back[i, j]
        if op == 'hit':
            assert a[pi] == b[pj]
        elif op == 'sub':
            assert a[pi] != b[pj]
        ops.append(op)
        i, j = pi, pj
    counts = (ops.count('hit'), ops.count('sub'), ops.count('del'), ops.count('ins'))
    assert counts[0] == best[n][m]
    assert sum(counts[1:]) == distance
    assert counts[0] + counts[1] + counts[2] == n
    assert counts[0] + counts[1] + counts[3] == m
    return counts


# --- other tools' rules, from their sources -----------------------------------

def kaldi_counts(ref, hyp):
    """compute-wer's counts, by the tie order of Kaldi's
    LevenshteinEditDistance: rows over the hypothesis, columns over the
    reference, each cell carrying (cost, ins, del, sub) from the step it
    took. The diagonal only when strictly cheaper than both other steps,
    then a deletion only when strictly cheaper than an insertion, and an
    insertion otherwise."""
    n = len(ref)
    prev = [(j, 0, j, 0) for j in range(n + 1)]
    for y in hyp:
        cost, ins, dele, sub = prev[0]
        row = [(cost + 1, ins + 1, dele, sub)]
        for j in range(1, n + 1):
            differ = y != ref[j - 1]
            ins_err = prev[j][0] + 1
            del_err = row[j - 1][0] + 1
            sub_err = prev[j - 1][0] + differ
            if sub_err < ins_err and sub_err < del_err:
                cost, ins, dele, sub = prev[j - 1]
                row.append((sub_err, ins, dele, sub + differ))
            elif del_err < ins_err:
                cost, ins, dele, sub = row[j - 1]
                row.append((del_err, ins, dele + 1, sub))
            else:
                cost, ins, dele, sub = prev[j]
                row.append((ins_err, ins + 1, dele, sub))
        prev = row
    cost, ins, dele, sub = prev[n]
    return n - sub - dele, sub, dele, ins


def sclite_errors(ref, hyp):
    """The fewest and the most errors over the alignments that minimise
    sclite's distance, 3 for an insertion or a deletion and 4 for a
    substitution.

    That distance is 3E + S, E the errors. At an equal distance more hits
    mean more errors (2E = 2H + distance - N - M), so the fewest errors come
    with the fewest hits and the most with the most hits. Hits ride along in
    the cost with a factor larger than any hit count, once counted against
    and once in favour."""
    n, m = len(ref), len(hyp)
    big = n + m + 1

    def cheapest(sign):
        prev = [3 * j * big for j in range(m + 1)]
        for i in range(1, n + 1):
            row = [3 * i * big]
            for j in range(1, m + 1):
                if ref[i - 1] == hyp[j - 1]:
                    diag = prev[j - 1] + sign
                else:
                    diag = prev[j - 1] + 4 * big
                row.append(min(diag, prev[j] + 3 * big, row[j - 1] + 3 * big))
            prev = row
        return prev[m]

    fewest_hits = cheapest(+1)
    most_hits = cheapest(-1)
    distance = -(-most_hits // big)
    hits_low = fewest_hits - distance * big
    hits_high = distance * big - most_hits
    return ((2 * hits_low + distance - n - m) // 2,
            (2 * hits_high + distance - n - m) // 2)


def sclite_distance(ref, hyp):
    """sclite's weighted distance alone."""
    prev = [3 * j for j in range(len(hyp) + 1)]
    for i in range(1, len(ref) + 1):
        row = [3 * i]
        for j in range(1, len(hyp) + 1):
            row.append(min(prev[j - 1] + (0 if ref[i - 1] == hyp[j - 1] else 4),
                           prev[j] + 3, row[j - 1] + 3))
        prev = row
    return prev[-1]


def weighted_distance(a, b, substitution):
    """Edit distance with insertions and deletions at 1 and substitutions at
    *substitution*: the algorithm of Fig. 2.21 of Jurafsky and Martin."""
    prev = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        row = [i]
        for j in range(1, len(b) + 1):
            row.append(min(prev[j] + 1, row[j - 1] + 1,
                           prev[j - 1] + (0 if a[i - 1] == b[j - 1] else substitution)))
        prev = row
    return prev[-1]


def jiwer_words(text):
    """jiwer 4.0.0's wer_default as its source states it: runs of two or more
    whitespace characters to one space, strip, split on the space."""
    return [w for w in re.sub(r"\s\s+", " ", text).strip().split(" ") if w]


def jiwer_characters(text):
    """jiwer 4.0.0's cer_default: strip, then every character."""
    return list(text.strip())


def kudio_counts(ref, hyp):
    """kudio's counts for two token sequences."""
    return module._align(list(ref), list(hyp))


def rate_of(counts):
    return sum(counts[1:]) / (counts[0] + counts[1] + counts[2])


# --- inputs -------------------------------------------------------------------

def random_pair(rng, longest, symbols):
    alphabet = [f"w{k}" for k in range(symbols)]
    a = [rng.choice(alphabet) for _ in range(rng.randint(0, longest))]
    b = [rng.choice(alphabet) for _ in range(rng.randint(0, longest))]
    return a, b


def edited(rng, tokens, alphabet, rate):
    """*tokens* with about *rate* of them substituted, deleted or inserted."""
    out = []
    for t in tokens:
        u = rng.random()
        if u < 0.6 * rate:
            out.append(rng.choice(alphabet))
        elif u >= 0.8 * rate:
            out.append(t)
        if rng.random() < 0.2 * rate:
            out.append(rng.choice(alphabet))
    return out


def simulated_corpus(seed=2026):
    """100,000 reference words in 5,000 utterances of 5 to 35 words, drawn
    from 5,000 made-up words with Zipf frequencies. The hypotheses
    substitute 6% of the words, delete 2% and insert 2%."""
    rng = random.Random(seed)
    letters = "abcdefghijklmnopqrstuvwxyz"
    vocabulary = set()
    while len(vocabulary) < 5000:
        vocabulary.add("".join(rng.choice(letters) for _ in range(rng.randint(2, 9))))
    vocabulary = sorted(vocabulary)
    rng.shuffle(vocabulary)
    weights = [1.0 / (rank + 1) for rank in range(len(vocabulary))]
    refs, hyps = [], []
    for _ in range(2500):
        spread = rng.randint(0, 15)
        for length in (20 + spread, 20 - spread):
            ref = rng.choices(vocabulary, weights, k=length)
            hyp = []
            for word in ref:
                u = rng.random()
                if u < 0.06:
                    hyp.append(rng.choices(vocabulary, weights)[0])
                elif u >= 0.08:
                    hyp.append(word)
                if rng.random() < 0.02:
                    hyp.append(rng.choices(vocabulary, weights)[0])
            refs.append(" ".join(ref))
            hyps.append(" ".join(hyp))
    return refs, hyps


# --- the checks -----------------------------------------------------------------

def check_worked_examples():
    # Sec. 16.6: REF "i *** ** UM the PHONE IS i LEFT THE portable **** PHONE
    # UPSTAIRS last night", lower-cased and without the gaps of the alignment
    ref = "i um the phone is i left the portable phone upstairs last night"
    hyp = "i got it to the fullest i love to portable form of stores last night"
    report("SLP3 16.6, CallHome: H, S, D, I, N = 6, 6, 1, 3, 13", 1,
           apart(word_errors(ref, hyp), (6, 6, 1, 3, 13)))
    report("SLP3 16.6, CallHome: 76.9%", 1, 0, abs(round(100 * wer(ref, hyp), 1) - 76.9))

    regions = [("it was", "its", "it was"),
               ("the best of times it", "the best of times it", "the best times it"),
               ("was the worst", "is the worst", "won the test"),
               ("of times it was", "of times or it was", "of times it was")]
    worst = max(apart([word_errors(r, a).errors for r, a, _ in regions], (2, 0, 1, 1)),
                apart([word_errors(r, b).errors for r, _, b in regions], (0, 1, 2, 0)))
    whole = " ".join(r for r, _, _ in regions)
    worst = max(worst,
                abs(word_errors(whole, " ".join(a for _, a, _ in regions)).errors - 4),
                abs(word_errors(whole, " ".join(b for _, _, b in regions)).errors - 3))
    report("SLP3 16.6, NIST regions: errors per region", 10, worst)

    # Sec. 2.9 and Fig. 2.17: d s s i s, and the other five letters hits
    worst = max(apart(char_errors("intention", "execution"), (5, 3, 1, 1, 9)),
                abs(weighted_distance("intention", "execution", 1) - 5),
                abs(weighted_distance("intention", "execution", 2) - 8))
    report("SLP3 2.9, intention/execution: 5 and 8", 3, worst)


def check_enumeration():
    small = [[]] + [list(p) for n in range(1, 5) for p in itertools.product("xy", repeat=n)]
    worst = 0
    cases = 0
    for a in small:
        for b in small:
            worst = max(worst, apart(kudio_counts(a, b), by_enumeration(a, b)))
            cases += 1
    rng = random.Random(1)
    for _ in range(3000):
        a, b = random_pair(rng, 6, rng.choice((2, 3, 4)))
        worst = max(worst, apart(kudio_counts(a, b), by_enumeration(a, b)))
        cases += 1
    report("every alignment, pairs of up to 6 tokens", cases, worst)


def check_tables():
    rng = random.Random(2)
    worst = 0
    cases = 0
    for _ in range(2000):
        a, b = random_pair(rng, 40, rng.choice((2, 3, 5, 20)))
        worst = max(worst, apart(kudio_counts(a, b), by_tables(a, b)))
        cases += 1
    report("full tables, pairs of up to 40 tokens", cases, worst)

    # long enough for the numpy rows, edited as a recogniser would and unrelated
    worst = 0
    cases = 0
    for k in range(60):
        alphabet = [f"w{t}" for t in range(rng.choice((3, 10, 200)))]
        ref = [rng.choice(alphabet) for _ in range(rng.randint(64, 300))]
        if k % 3 == 0:
            hyp = [rng.choice(alphabet) for _ in range(rng.randint(64, 300))]
        else:
            hyp = edited(rng, ref, alphabet, rng.choice((0.1, 0.3, 0.6)))
        worst = max(worst, apart(kudio_counts(ref, hyp), by_tables(ref, hyp)))
        cases += 1
    report("full tables, pairs of 64 to 300 tokens", cases, worst)


def check_fills():
    rng = random.Random(3)
    worst = 0
    cases = 0
    for _ in range(500):
        a, b = random_pair(rng, 120, rng.choice((2, 5, 50)))
        if not a or not b:
            continue
        weight = min(len(a), len(b)) + 1
        cost = module._cost_python(a, b, weight)
        worst = max(worst, abs(module._cost_rows(a, b, weight) - cost),
                    abs(module._cost_rows(b, a, weight) - cost))
        cases += 1
    report("plain Python against numpy rows (cost)", cases, worst)


def check_corpus():
    rng = random.Random(4)
    worst = 0
    worst_rate = 0.0
    cases = 0
    for _ in range(300):
        refs, hyps = [], []
        for _ in range(rng.randint(1, 30)):
            a, b = random_pair(rng, 12, rng.choice((3, 8)))
            refs.append(" ".join(a))
            hyps.append(" ".join(b))
        total = [0, 0, 0, 0]
        for r, h in zip(refs, hyps):
            for k, value in enumerate(by_tables(r.split(), h.split())):
                total[k] += value
        got = word_errors(refs, hyps)
        keyed = word_errors({f"u{k}": r for k, r in enumerate(refs)},
                            {f"u{k}": h for k, h in reversed(list(enumerate(hyps)))})
        worst = max(worst, apart(got[:4], total), apart(keyed, got))
        words = total[0] + total[1] + total[2]
        if words:
            worst_rate = max(worst_rate, abs(wer(refs, hyps) - sum(total[1:]) / words))
        cases += 1
    report("a corpus is the sum of its pairs; dict = list", cases, worst, worst_rate)

    worst = 0
    cases = 0
    han = "的一是不了人我在有他這為之大來以個中上們"
    for _ in range(500):
        a = "".join(rng.choice(han) for _ in range(rng.randint(1, 30)))
        b = "".join(rng.choice(han) for _ in range(rng.randint(0, 30)))
        spaced = " ".join(b)
        worst = max(worst, apart(char_errors(a, b)[:4], by_tables(a, b)),
                    apart(char_errors(a, spaced), char_errors(a, b)),
                    apart(char_errors(a, spaced, spaces=True)[:4], by_tables(a, spaced)))
        cases += 1
    report("CER on Chinese, with and without spaces", cases, worst)


def check_nltk():
    try:
        import nltk
    except ImportError:
        report("nltk edit_distance: skipped, no nltk", 0)
        return
    rng = random.Random(5)
    worst = 0
    for _ in range(2000):
        a, b = random_pair(rng, 30, rng.choice((2, 5, 30)))
        worst = max(worst, abs(nltk.edit_distance(a, b) - sum(kudio_counts(a, b)[1:])))
    report(f"nltk {nltk.__version__} edit_distance, totals", 2000, worst)


def check_other_tools(refs, hyps):
    notes = []
    rng = random.Random(6)
    short = [random_pair(rng, 12, 4) for _ in range(20000)]
    short = [(a, b) for a, b in short if a]
    for label, pairs in (("simulated corpus", [(r.split(), h.split()) for r, h in zip(refs, hyps)]),
                         ("short pairs over four symbols", short)):
        split_apart = 0
        totals_apart = 0
        sclite_more = 0
        sclite_maybe = 0
        for a, b in pairs:
            mine = kudio_counts(a, b)
            errors = sum(mine[1:])
            kaldi = kaldi_counts(a, b)
            totals_apart = max(totals_apart, abs(sum(kaldi[1:]) - errors))
            split_apart += kaldi != mine
            fewest, most = sclite_errors(a, b)
            if fewest > errors:
                sclite_more += 1
            elif most > errors:
                sclite_maybe += 1
        report(f"compute-wer totals, {label}", len(pairs), totals_apart)
        notes.append(f"  {label}, {len(pairs)} pairs: compute-wer splits S, D and I "
                     f"differently on {split_apart}; sclite's weights give more "
                     f"errors than the fewest on {sclite_more} for certain, and on "
                     f"{sclite_maybe} more depending on how it breaks its own ties")

    # where sclite's alignment has the fewest errors, its counts are kudio's
    rng = random.Random(7)
    worst = 0
    cases = 0
    for _ in range(3000):
        a, b = random_pair(rng, 6, rng.choice((2, 3, 4)))
        mine = kudio_counts(a, b)
        if sclite_errors(a, b)[1] == sum(mine[1:]):
            cheapest = min(every_alignment(a, b), key=lambda c: 3 * (c[1] + c[2] + c[3]) + c[1])
            worst = max(worst, apart(cheapest, mine))
            cases += 1
    report("sclite's counts where it has the fewest errors", cases, worst)

    # the pairs the module notes quote
    a_c, d_d_a = "a c".split(), "d d a".split()
    report("'a c' / 'd d a': compute-wer 0 H, 2 S, 0 D, 1 I", 1,
           apart(kaldi_counts(a_c, d_d_a), (0, 2, 0, 1)))
    report("'a c' / 'd d a': kudio 1 H, 0 S, 1 D, 2 I", 1,
           apart(kudio_counts(a_c, d_d_a), (1, 0, 1, 2)))
    one, four = "one two three four five".split(), "four five six seven eight".split()
    mine = kudio_counts(one, four)
    report("one..five / four..eight: kudio 5 S, WER 1.0", 1,
           apart(mine, (0, 5, 0, 0)), abs(rate_of(mine) - 1.0))
    fewest, most = sclite_errors(one, four)
    report("one..five / four..eight: sclite 6 errors, WER 1.2", 1,
           apart((fewest, most), (6, 6)), abs(fewest / 5 - 1.2))
    report("one..five / four..eight: sclite's cost 18, 20 for 5 S", 1,
           max(abs(sclite_distance(one, four) - 18), abs(4 * 5 - 20)))

    split = kudio_counts(jiwer_words("hello world"), jiwer_words("hello\nworld"))
    report("jiwer's words, 'hello world' / 'hello\\nworld': 1.0", 1, 0,
           abs(rate_of(split) - 1.0))
    report("kudio, 'hello world' / 'hello\\nworld': 0", 1, 0,
           abs(wer("hello world", "hello\nworld")))
    split = kudio_counts(jiwer_characters("a  b"), jiwer_characters("a b"))
    report("jiwer's characters, 'a  b' / 'a b': 0.25", 1, 0, abs(rate_of(split) - 0.25))
    report("kudio spaces=True, 'a  b' / 'a b': 0", 1, 0, abs(cer("a  b", "a b", spaces=True)))
    report("jiwer's characters = spaces=True on single spaces", 1000, 0,
           max(abs(rate_of(kudio_counts(jiwer_characters(r), jiwer_characters(h)))
                   - cer(r, h, spaces=True))
               for r, h in zip(refs[:1000], hyps[:1000])))
    return notes


def timed(fn, *args, repeats=1):
    start = time.perf_counter()
    for _ in range(repeats):
        out = fn(*args)
    return out, (time.perf_counter() - start) / repeats


def check_speed(refs, hyps):
    lines = []
    counts, took = timed(word_errors, refs, hyps)
    lines.append(f"  word_errors, {len(refs)} utterances, {counts.reference_length} words: "
                 f"{took:.2f} s (WER {counts.errors / counts.reference_length:.4f})")
    counts, took = timed(char_errors, refs, hyps)
    lines.append(f"  char_errors, the same corpus, {counts.reference_length} characters: "
                 f"{took:.2f} s (CER {counts.errors / counts.reference_length:.4f})")

    rng = random.Random(8)
    for side in (32, 48, 64, 128):
        a = [rng.randrange(side) for _ in range(side)]
        b = [t if rng.random() > 0.1 else rng.randrange(side) for t in a]
        weight = side + 1
        repeats = max(1, 40000 // (side * side))
        _, python = timed(module._cost_python, a, b, weight, repeats=repeats)
        _, rows = timed(module._cost_rows, a, b, weight, repeats=repeats)
        lines.append(f"  one table of {side} x {side} tokens: plain Python {1e3 * python:.3f} ms, "
                     f"numpy rows {1e3 * rows:.3f} ms")

    words_ref = " ".join(refs).split()
    words_hyp = " ".join(hyps).split()
    for size in (10000, 100000):
        ref = " ".join(words_ref[:size])
        hyp = " ".join(words_hyp[:round(size * len(words_hyp) / len(words_ref))])
        counts, took = timed(word_errors, ref, hyp)
        lines.append(f"  word_errors, one pair of {counts.reference_length} words: {took:.2f} s")
    return lines


def main():
    check_worked_examples()
    check_enumeration()
    check_tables()
    check_fills()
    check_corpus()
    check_nltk()
    refs, hyps = simulated_corpus()
    notes = check_other_tools(refs, hyps)
    speed = check_speed(refs, hyps)

    width = max(len(r[0]) for r in RESULTS)
    print(f"{'check':{width}s}  {'cases':>6s}  {'counts':>6s}  {'rate':>8s}")
    for check, cases, count_diff, rate_diff in RESULTS:
        print(f"{check:{width}s}  {cases:6d}  {count_diff:6d}  {rate_diff:8.1e}")
    print("\nOther tools' rules:")
    print("\n".join(notes))
    print("\nSpeed:")
    print("\n".join(speed))
    bad = [r for r in RESULTS if r[2] != 0 or r[3] > 1e-15]
    if bad:
        print(f"\n{len(bad)} check(s) differ")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
