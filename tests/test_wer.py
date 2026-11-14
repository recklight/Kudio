# -*- coding: utf-8 -*-
"""Word and character error rates: the worked examples of Jurafsky and
Martin, every alignment of small pairs, both ways of filling the table,
corpora, the numbers the module notes quote, and the input the four
functions refuse or cannot score."""
from __future__ import annotations

import ast
import importlib.util
import itertools
import random
import re
import sys
import unicodedata
import warnings
from pathlib import Path

import numpy as np
import pytest

from kudio.core.wer import ErrorCounts, cer, char_errors, wer, word_errors

# the module, for its private parts: once kudio.core exports the function wer,
# ``from kudio.core import wer`` gives the function instead
module = importlib.import_module("kudio.core.wer")
ROOT = Path(__file__).resolve().parents[1]


def _quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return fn(*args, **kwargs)


def _every_alignment(a, b):
    """Every alignment of a with b, as (hits, subs, dels, ins)."""
    if not a:
        yield 0, 0, 0, len(b)
        return
    if not b:
        yield 0, 0, len(a), 0
        return
    hit = a[0] == b[0]
    for h, s, d, i in _every_alignment(a[1:], b[1:]):
        yield (h + 1, s, d, i) if hit else (h, s + 1, d, i)
    for h, s, d, i in _every_alignment(a[1:], b):
        yield h, s, d + 1, i
    for h, s, d, i in _every_alignment(a, b[1:]):
        yield h, s, d, i + 1


def _best_of_all(a, b):
    """The fewest errors, then the most hits, by trying every alignment."""
    return min(_every_alignment(a, b), key=lambda c: (c[1] + c[2] + c[3], -c[0]))


def _random_tokens(rng, longest, symbols):
    return [rng.choice(symbols) for _ in range(rng.randint(0, longest))]


# --------------------------------------------------- the worked examples

def test_the_callhome_utterance_of_slp3_16_6():
    """Jurafsky and Martin, Speech and Language Processing, 3rd ed. draft of
    19 August 2026, Sec. 16.6: six substitutions, three insertions and one
    deletion in 13 reference words, 76.9%. The alignment printed there leaves
    6 hits, and so does this one."""
    ref = "i um the phone is i left the portable phone upstairs last night"
    hyp = "i got it to the fullest i love to portable form of stores last night"
    assert word_errors(ref, hyp) == (6, 6, 1, 3, 13)
    assert round(100 * wer(ref, hyp), 1) == 76.9


def test_the_nist_regions_of_slp3_16_6():
    """The MAPSSWE example in the same section, from NIST: per region,
    system A makes 2, 0, 1 and 1 errors and system B 0, 1, 2 and 0."""
    regions = [("it was", "its", "it was"),
               ("the best of times it", "the best of times it", "the best times it"),
               ("was the worst", "is the worst", "won the test"),
               ("of times it was", "of times or it was", "of times it was")]
    assert [word_errors(r, a).errors for r, a, _ in regions] == [2, 0, 1, 1]
    assert [word_errors(r, b).errors for r, _, b in regions] == [0, 1, 2, 0]
    whole = " ".join(r for r, _, _ in regions)
    assert word_errors(whole, " ".join(a for _, a, _ in regions)).errors == 4
    assert word_errors(whole, " ".join(b for _, _, b in regions)).errors == 3


def test_intention_and_execution_of_slp3_2_9():
    """5 edits, split in Fig. 2.17 as a deletion, three substitutions and an
    insertion, with the other 5 letters hits."""
    assert char_errors("intention", "execution") == (5, 3, 1, 1, 9)


# Totals from nltk 3.9.2's edit_distance, run on these pairs (as word lists,
# or as strings for the characters): an implementation written by others.
_NLTK_3_9_2 = [
    ("the quick brown fox jumps over the lazy dog",
     "the quick brown fox jumped over a lazy dog", 2),
    ("a b a b a b a b", "b a b a b a b a", 2),
    ("to be or not to be that is the question",
     "to be or to be not that is a question", 3),
    ("one two three four five", "four five six seven eight", 5),
    ("we will we will rock you", "will we rock you we will", 4),
]
_NLTK_3_9_2_CHARS = [
    ("今天天氣很好我們去公園散步", "今天天汽很好我門去公圓散步吧", 4),
    ("intention", "execution", 5),
    ("kitten", "sitting", 3),
]


@pytest.mark.parametrize("ref, hyp, total", _NLTK_3_9_2)
def test_word_totals_agree_with_nltk(ref, hyp, total):
    assert word_errors(ref, hyp).errors == total


@pytest.mark.parametrize("ref, hyp, total", _NLTK_3_9_2_CHARS)
def test_character_totals_agree_with_nltk(ref, hyp, total):
    assert char_errors(ref, hyp).errors == total


# Counts from the full-table counter of tools/validation/wer.py (forward and
# backward tables, the most hits over the shortest paths, one alignment traced
# back and recounted), on random pairs of 50 to 70 tokens: long enough for the
# numpy rows, and over three or eight symbols, where ties abound.
_FULL_TABLES = [
    ("c a b c a c a a b b a b c a a b b b a c b a c c a c a b c c a a c b a b b b a a b c c c b a "
     "b a a c c c a b c b",
     "a b a c c a c a a c c a c b b b c a b c c a b c a c a b a b a a c c b c a c a c a a c c a a "
     "a b b b b a b a b a a c",
     (33, 19, 4, 6)),
    ("a a c a b b b c b b c c c b a c a b a c a a b b b c b c b c b a a c b b c c c a c a c a b b "
     "a b b c b b b b c b c c b c b c b b a c c c a b",
     "b c a a c b c b b a a a b c b a b c a a c a a b a b b a b a a c c b c c a b c c c c a b c a "
     "a b c b b c a a a a a c c a a",
     (40, 17, 13, 4)),
    ("c a b c c c c b c a b b b c c a a c c b a c a c c a b c b b c a b c b a c a c a a b b a b c "
     "b a a a c c a c b a b c a b",
     "b a a a c a c b b c a a a a b a b b c b a b a a c c c a a a b b c c c c b b a a b c b b a a "
     "b b c c b a c c b b a b b a c a",
     (39, 12, 9, 11)),
    ("h a g a d d f d h b c g f a h h b f g a d h f a h d a e e a d a f f e a a g c b g g e h f b "
     "a a f h h f c f f f g g h b h g d d h d a e a e",
     "d f a d g c a b f d d f b f d e f g a f h a g a d h b a c e e a a e f c a f h h d a c f h a "
     "d d b b b d a b d d f b b d b",
     (32, 18, 20, 11)),
    ("b b c c d g d a e b e h b e e e a e d g g a g a b g b b e h g e g b c f e h h d b c g f d a "
     "c f g e e f f a e f a",
     "h e f a d g g e d c f h d c e d b h f h h e g a c h f c f f e b d g e a g c d c b h g c f a "
     "c b f e g a d b d c d",
     (20, 30, 7, 7)),
    ("f g g d g h b f e f g f h d e h a a h g b e f h c b e b g d a a c f h b b d b g f d d b f a "
     "e e c d d e d g e g c a h d b f a d",
     "h g h g h b e a b e a d d d f h f f c c f c h a b b a d g a g d c a f e e g d d b b h e h d "
     "c e b g g",
     (26, 19, 19, 6)),
]


@pytest.mark.parametrize("ref, hyp, counts", _FULL_TABLES)
def test_long_tie_heavy_pairs_match_the_full_table_counter(ref, hyp, counts):
    got = word_errors(ref, hyp)
    assert got[:4] == counts
    assert got.reference_length == len(ref.split())


# ------------------------------------------- the best of every alignment

def test_the_counts_are_the_fewest_errors_then_the_most_hits_of_every_alignment():
    """Every pair of up to 4 tokens over two symbols, 961 of them, and 600
    random pairs of up to 5 over three or four: the counts are those of the
    alignment with the fewest errors and, among those, the most hits."""
    small = [[]] + [list(p) for n in range(1, 5) for p in itertools.product("xy", repeat=n)]
    for a in small:
        for b in small:
            assert module._align(a, b) == _best_of_all(a, b), (a, b)
    rng = random.Random(11)
    for _ in range(600):
        symbols = rng.choice(("abc", "abcd"))
        a, b = _random_tokens(rng, 5, symbols), _random_tokens(rng, 5, symbols)
        assert module._align(a, b) == _best_of_all(a, b), (a, b)


def test_a_shared_start_and_end_are_hits():
    """They are set aside before the table is filled; the counts still have
    to be the best of every alignment of the whole pair."""
    rng = random.Random(12)
    for _ in range(300):
        common = _random_tokens(rng, 3, "ab")
        tail = _random_tokens(rng, 2, "ab")
        a = common + _random_tokens(rng, 3, "abc") + tail
        b = common + _random_tokens(rng, 3, "abc") + tail
        assert module._align(a, b) == _best_of_all(a, b), (a, b)


def test_the_total_is_fixed_but_the_split_follows_the_tie_rule():
    """"a c" against "d d a": three errors either way. compute-wer's tie order
    makes it two substitutions and an insertion (tools/validation/wer.py);
    the most hits make it one hit, one deletion and two insertions."""
    assert word_errors("a c", "d d a") == (1, 0, 1, 2, 2)
    assert set(c for c in _every_alignment("a c".split(), "d d a".split())
               if sum(c[1:]) == 3) >= {(1, 0, 1, 2), (0, 2, 0, 1)}


def test_a_shifted_hypothesis_is_all_substitutions_with_the_fewest_errors():
    """5 substitutions, a WER of 1.0. sclite's weights (3, 3, 4) would pair
    the two "four five" and count 6 errors instead; that comparison is in
    tools/validation/wer.py."""
    ref, hyp = "one two three four five", "four five six seven eight"
    assert word_errors(ref, hyp) == (0, 5, 0, 0, 5)
    assert wer(ref, hyp) == 1.0
    keeps_the_pair = (2, 0, 3, 3)
    assert keeps_the_pair in set(_every_alignment(ref.split(), hyp.split()))


# ----------------------------------------------- the two ways of filling

def test_the_numpy_rows_cost_the_same_as_plain_python():
    rng = random.Random(13)
    for _ in range(300):
        symbols = rng.choice(("ab", "abcde", [f"w{k}" for k in range(40)]))
        a = [rng.choice(symbols) for _ in range(rng.randint(1, 90))]
        b = [rng.choice(symbols) for _ in range(rng.randint(1, 90))]
        weight = min(len(a), len(b)) + 1
        want = module._cost_python(a, b, weight)
        assert module._cost_rows(a, b, weight) == want
        assert module._cost_rows(b, a, weight) == want


def test_both_ways_of_filling_give_the_same_counts(monkeypatch):
    """Pairs of 1 to 60 tokens, each counted once with every table filled in
    plain Python and once with every table filled by numpy rows."""
    rng = random.Random(14)
    pairs = []
    for _ in range(200):
        n, m = rng.randint(1, 60), rng.randint(1, 60)
        pairs.append(([rng.choice("abc") for _ in range(n)],
                      [rng.choice("abc") for _ in range(m)]))
    monkeypatch.setattr(module, "_NUMPY_CELLS", 0)
    by_rows = [module._align(a, b) for a, b in pairs]
    monkeypatch.setattr(module, "_NUMPY_CELLS", float("inf"))
    by_python = [module._align(a, b) for a, b in pairs]
    assert by_rows == by_python


def test_known_edits_far_apart_are_counted_exactly():
    """400 distinct words with 7 substituted, 5 deleted and 3 inserted, each
    edit between untouched words, so the best alignment is the edits
    themselves. Long enough for the numpy rows."""
    ref = [f"w{k}" for k in range(400)]
    hyp = list(ref)
    for k in (370, 330, 290, 250, 210, 170, 130):
        hyp[k] = f"new{k}"
    for k in (350, 310, 230, 150, 90):
        del hyp[k]
    for k in (270, 190, 50):
        hyp.insert(k, f"extra{k}")
    assert word_errors(" ".join(ref), " ".join(hyp)) == (388, 7, 5, 3, 400)


# ------------------------------------------------------------ corpora

def test_a_corpus_is_pooled_not_averaged():
    """One error in a 2-word utterance, none in an 18-word one: 1/20 pooled,
    where the mean of the two rates would be 0.25."""
    long = " ".join(f"w{k}" for k in range(18))
    refs, hyps = ["a b", long], ["a c", long]
    assert wer(refs, hyps) == 0.05
    assert np.mean([wer(r, h) for r, h in zip(refs, hyps)]) == 0.25


def test_a_corpus_is_the_sum_of_its_pairs():
    rng = random.Random(15)
    refs = [" ".join(_random_tokens(rng, 10, "abcdef")) for _ in range(50)]
    hyps = [" ".join(_random_tokens(rng, 10, "abcdef")) for _ in range(50)]
    summed = np.sum([word_errors(r, h) for r, h in zip(refs, hyps)], axis=0)
    assert tuple(word_errors(refs, hyps)) == tuple(int(x) for x in summed)
    assert wer(refs, hyps) == (summed[1] + summed[2] + summed[3]) / summed[4]
    letters = char_errors(refs, hyps)
    assert cer(refs, hyps) == letters.errors / letters.reference_length


def test_dicts_are_paired_by_key_whatever_their_order():
    refs = {"u1": "a b c", "u2": "d e", "u3": "f"}
    hyps = {"u3": "g", "u1": "a c", "u2": "d e"}
    assert word_errors(refs, hyps) == word_errors(["a b c", "d e", "f"], ["a c", "d e", "g"])
    assert wer(refs, hyps) == 2 / 6


def test_tuples_generators_and_numpy_string_arrays_are_sequences():
    refs, hyps = ["a b", "c d"], ["a b", "c x"]
    want = word_errors(refs, hyps)
    assert word_errors(tuple(refs), tuple(hyps)) == want
    assert word_errors(iter(refs), (h for h in hyps)) == want
    assert word_errors(np.array(refs), np.array(hyps)) == want


def test_an_empty_reference_inside_a_corpus_counts_its_insertions():
    assert _quiet(wer, ["a b", ""], ["a b", "c"]) == 0.5
    assert word_errors(["a b", ""], ["a b", "c"]) == (2, 0, 0, 1, 2)


def test_counts_have_errors_and_both_lengths():
    counts = word_errors("a b c d", "a x c d e")
    assert counts == ErrorCounts(hits=3, substitutions=1, deletions=0,
                                 insertions=1, reference_length=4)
    assert counts.errors == 2
    assert counts.hits + counts.substitutions + counts.insertions == 5


# -------------------------------------------- the numbers the notes quote

def test_the_docstring_examples():
    assert word_errors("the cat sat on the mat", "the cat sat on mat") == (5, 0, 1, 0, 6)
    assert char_errors("我愛你", "我恨你") == (2, 1, 0, 0, 3)
    assert wer("the cat sat on the mat", "the cat sat on mat") == 1 / 6
    assert wer(["a b", "c d e"], ["a x", "c d e"]) == 0.2
    assert cer("今天天氣很好", "今天天氣不好") == 1 / 6
    assert cer("the cat sat", "the hat sat") == 1 / 9
    assert cer("the cat sat", "the hat sat", spaces=True) == 1 / 11


def test_case_and_punctuation_count_as_they_stand():
    assert word_errors("Hello, world.", "hello world") == (0, 2, 0, 0, 2)
    assert wer("Hello, world.", "hello world") == 1.0

    def normalise(text):
        return re.sub(r"[^\w\s']", " ", text.lower())
    assert wer(normalise("Hello, world."), normalise("hello world")) == 0.0
    assert wer([normalise("Don't stop.")], [normalise("don't stop")]) == 0.0


def test_unicode_is_taken_as_it_comes():
    """A composed é against e and a combining accent is 2 errors in 4 code
    points; NFC makes them one. NFKC folds full-width forms. Simplified and
    traditional characters stay different."""
    assert char_errors("café", "café") == (3, 1, 0, 1, 4)
    assert cer("café", "café") == 0.5
    assert cer(unicodedata.normalize("NFC", "café"),
               unicodedata.normalize("NFC", "café")) == 0.0
    assert cer("ＡＢＣ１", "ABC1") == 1.0
    assert cer(unicodedata.normalize("NFKC", "ＡＢＣ１"), "ABC1") == 0.0
    assert cer("我爱你", "我愛你") == 1 / 3


def test_wer_has_no_upper_bound():
    assert word_errors("yes", "yes yes yes") == (1, 0, 0, 2, 1)
    assert wer("yes", "yes yes yes") == 2.0


def test_text_without_spaces_is_one_word():
    assert wer("今天天氣很好", "今天天氣不好") == 1.0
    assert cer("今天天氣很好", "今天天氣不好") == 1 / 6


def test_words_split_on_any_whitespace():
    """jiwer would read "hello\\nworld" as one word and score 1.0 against
    "hello world" (tools/validation/wer.py shows it from its source)."""
    assert wer("hello world", "hello\nworld") == 0.0
    assert wer("hello world", "  hello\t\tworld \r\n") == 0.0
    assert wer("你好 世界", "你好　世界") == 0.0


def test_cer_leaves_whitespace_out_unless_asked():
    assert cer("我 愛 你", "我愛你") == 0.0
    assert char_errors("我 愛 你", "我愛你", spaces=True) == (3, 0, 2, 0, 5)
    assert cer("我 愛 你", "我愛你", spaces=True) == 0.4
    assert cer("a cat", "acat") == 0.0
    assert cer("a cat", "acat", spaces=True) == 1 / 5


def test_spaces_count_once_per_gap_and_not_at_the_ends():
    """jiwer would count the second space of "a  b" as a character of its
    own, 1/4; here a gap is one space however it is written."""
    assert cer("a  b", "a b", spaces=True) == 0.0
    assert cer(" a\tb\n", "a b", spaces=True) == 0.0
    assert char_errors("a  b", "a b", spaces=True).reference_length == 3


# ------------------------------------------- what cannot be scored: nan

@pytest.mark.parametrize("fn, ref, hyp, match", [
    (wer, "", "", r"^wer\(\): the reference has no words, and a rate needs "
                  r"at least 1 to divide by; scored nan$"),
    (wer, "", "a b", r"the 2 words on the hypothesis side would all be insertions"),
    (wer, "  ", "a", r"the one word on the hypothesis side would be an insertion"),
    (wer, [], [], r"^wer\(\): got no transcripts"),
    (wer, ["", " "], ["a", ""], r"none of the 2 references has any words"),
    (cer, " \t", "x", r"^cer\(\): the reference has no characters other than whitespace"),
    (cer, "", "", r"scored nan$"),
])
def test_a_reference_side_with_nothing_in_it_is_nan_with_a_warning(fn, ref, hyp, match):
    with pytest.warns(UserWarning, match=match):
        assert np.isnan(fn(ref, hyp))


def test_the_warning_comes_once_and_points_at_the_caller():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        wer(["", "", ""], ["a", "b", "c"])
        cer("", "abc", spaces=True)
    assert len(caught) == 2
    assert all(w.filename == __file__ for w in caught)


def test_the_counts_of_an_empty_reference_come_back_without_a_warning():
    assert _quiet(word_errors, "", "a b") == (0, 0, 0, 2, 0)
    assert _quiet(char_errors, [], []) == (0, 0, 0, 0, 0)


# ------------------------------------------------ what is refused: errors

@pytest.mark.parametrize("ref, hyp, match", [
    (None, "a", r"reference has to be a str, a sequence of str or a dict"),
    ("a", 3, r"hypothesis has to be a str"),
    (b"a b", "a b", r"reference is bytes; decode it first"),
    ({"a b"}, ["a b"], r"reference is a set, which has no order"),
    ("a b", ["a b"], r"reference is one transcript and hypothesis a collection"),
    (["a b"], "a b", r"hypothesis is one transcript and reference a collection"),
    (["a", "b"], ["a"], r"2 reference transcript\(s\) and 1 hypothesis transcript\(s\)"),
    ({"u1": "a"}, ["a"], r"one side is a dict and the other is not"),
    ({"u1": "a", "u2": "b"}, {"u1": "a", "u3": "b"},
     r"1 reference key\(s\) have no hypothesis \('u2'\) and 1 hypothesis key\(s\) "
     r"have no reference \('u3'\)"),
    ({"u1": "a"}, {"u1": "a", "u2": "b"},
     r"1 hypothesis key\(s\) have no reference \('u2'\); give"),
    ({"u1": "a", "u2": "b"}, {"u1": "a"},
     r"1 reference key\(s\) have no hypothesis \('u2'\); give"),
    (np.array([["a b"], ["c d"]]), np.array([["a b"], ["c d"]]), r"reference is a 2-d table"),
    (["a", None], ["a", "b"], r"reference\[1\] is NoneType None, not a str; pass ''"),
    (["a", "b"], ["a", float("nan")], r"hypothesis\[1\] is float nan.*fillna\(''\)"),
    (["a"], [b"a"], r"hypothesis\[0\] is bytes .*decode it first"),
    ([["the", "cat"]], ["the cat"], r"join a transcript's tokens with spaces"),
    ({"u1": 5}, {"u1": "a"}, r"reference\['u1'\] is int 5"),
    (np.array("a b"), np.array("a b"), r"cannot be read as a sequence"),
])
def test_input_that_does_not_pair_up_is_refused(ref, hyp, match):
    with pytest.raises(ValueError, match=r"^wer\(\): .*" + match):
        wer(ref, hyp)


@pytest.mark.parametrize("fn", [wer, cer, word_errors, char_errors])
def test_each_function_names_itself(fn):
    name = fn.__name__
    with pytest.raises(ValueError, match=rf"^{name}\(\): "):
        fn(["a"], ["a", "b"])


@pytest.mark.parametrize("fn", [cer, char_errors])
@pytest.mark.parametrize("spaces", ["yes", 1, None])
def test_spaces_has_to_be_true_or_false(fn, spaces):
    with pytest.raises(ValueError, match=rf"^{fn.__name__}\(\): spaces has to be "
                                         rf"True or False"):
        fn("a b", "a b", spaces=spaces)


def test_numpy_bools_are_taken_for_spaces():
    assert cer("a cat", "acat", spaces=np.bool_(True)) == 1 / 5


def test_a_pandas_na_gets_the_pandas_hint():
    pd = pytest.importorskip("pandas")
    with pytest.raises(ValueError, match=r"fillna\(''\)"):
        wer(pd.Series(["a", "b"], dtype="string"),
            pd.Series(["a", None], dtype="string"))


def test_two_series_are_paired_by_position_only_when_their_indexes_agree():
    """The same two transcripts, the hypotheses in the other order with their
    index following them: paired by position they would score 1.0, which is
    why the call is refused. .to_dict() of both pairs them by index, at 0."""
    pd = pytest.importorskip("pandas")
    refs = pd.Series(["a b", "c d"], index=["u1", "u2"])
    hyps = pd.Series(["c d", "a b"], index=["u2", "u1"])
    with pytest.raises(ValueError, match=r"^wer\(\): the two sides carry different "
                                         r"indexes.*\.to_dict\(\).*\.tolist\(\)"):
        wer(refs, hyps)
    assert wer(refs.to_dict(), hyps.to_dict()) == 0.0
    assert wer(refs.tolist(), hyps.tolist()) == 1.0
    assert wer(refs, hyps.sort_index()) == 0.0
    assert word_errors(refs, ["a b", "c x"]) == word_errors(["a b", "c d"], ["a b", "c x"])
    # the labels decide, whatever the dtype of the index; a NA label differs
    # from any other label, without the comparison itself raising
    as_strings = pd.Series(["a b", "c d"], index=pd.Index(["u1", "u2"], dtype="string"))
    assert wer(refs, as_strings) == 0.0
    assert wer(pd.Series(["a b", "c d"]), pd.Series(["a b", "c x"], index=[0, 1])) == 0.25
    with_na = pd.Series(["a b", "c d"], index=pd.Index(["u1", pd.NA], dtype="string"))
    with pytest.raises(ValueError, match=r"different indexes"):
        wer(refs, with_na)


def test_a_table_is_refused_rather_than_read_by_its_column_names():
    """Iterating a DataFrame gives its column names, so two frames with a
    "text" column would otherwise score 0 whatever their rows hold."""
    pd = pytest.importorskip("pandas")
    refs = pd.DataFrame({"text": ["a b", "c d"]})
    hyps = pd.DataFrame({"text": ["x y", "z w"]})
    with pytest.raises(ValueError, match=r"^wer\(\): reference is a 2-d table.*\['text'\]"):
        wer(refs, hyps)
    assert wer(refs["text"], hyps["text"]) == 1.0


# ------------------------------------------------------- the files

@pytest.mark.parametrize("path", ["kudio/core/wer.py", "tools/validation/wer.py"])
def test_the_files_parse_as_python_3_9(path):
    ast.parse((ROOT / path).read_text(encoding="utf-8"), feature_version=(3, 9))


def test_the_validation_script_reaches_the_module_once_kudio_core_exports_wer(monkeypatch):
    """With the function exported from kudio.core, ``from kudio.core import
    wer`` gives the function, which has none of the module's private parts."""
    import kudio.core
    monkeypatch.setattr(kudio.core, "wer", wer, raising=False)
    monkeypatch.setattr(sys, "path", list(sys.path))
    spec = importlib.util.spec_from_file_location("wer_validation_script",
                                                  ROOT / "tools/validation/wer.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    assert script.module is sys.modules["kudio.core.wer"]
    assert script.kudio_counts("a c".split(), "d d a".split()) == (1, 0, 1, 2)
