# -*- coding: utf-8 -*-
"""Dataset manifests and splits."""
import json
import warnings

import pytest

import kudio


def _pairs(n=10):
    return [kudio.Pair(noisy=f"mix/{i}.wav", clean=f"clean/{i}.wav",
                       noise="fan", snr_db=i - 5) for i in range(n)]


def test_manifest_round_trips(tmp_path):
    pairs = _pairs()
    path = kudio.save_manifest(tmp_path / "m.json", pairs)
    assert kudio.load_manifest(path) == pairs


def test_manifest_is_plain_readable_json(tmp_path):
    """Any other tool has to be able to read it without importing kudio."""
    path = kudio.save_manifest(tmp_path / "m.json", _pairs(2))
    rows = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(rows, list)
    assert rows[0]["noisy"] == "mix/0.wav"


def test_save_manifest_creates_parents(tmp_path):
    path = kudio.save_manifest(tmp_path / "runs" / "exp" / "m.json", _pairs(1))
    assert path.is_file()


def test_pairs_are_hashable():
    """Split overlap is checked with sets; unhashable pairs would pass silently."""
    assert len(set(_pairs())) == 10


def test_unknown_keys_are_dropped_not_fatal(tmp_path):
    """And said out loud: a log line behind the kudio NullHandler reached nobody."""
    path = tmp_path / "m.json"
    path.write_text(json.dumps([
        {"noisy": "a.wav", "clean": "b.wav", "future_field": 1}]),
        encoding="utf-8")
    with pytest.warns(UserWarning, match="future_field"):
        loaded = kudio.load_manifest(path)
    assert loaded == [kudio.Pair(noisy="a.wav", clean="b.wav")]


def test_missing_required_key_is_an_error(tmp_path):
    path = tmp_path / "m.json"
    path.write_text(json.dumps([{"noisy": "a.wav"}]), encoding="utf-8")
    with pytest.raises(TypeError):
        kudio.load_manifest(path)


def test_manifest_must_be_a_list(tmp_path):
    path = tmp_path / "m.json"
    path.write_text(json.dumps({"noisy": "a.wav"}), encoding="utf-8")
    with pytest.raises(ValueError, match="not a JSON list"):
        kudio.load_manifest(path)


def test_exists_reports_what_is_on_disk(tmp_path):
    noisy, clean = tmp_path / "n.wav", tmp_path / "c.wav"
    noisy.write_bytes(b"")
    assert not kudio.Pair(str(noisy), str(clean)).exists()
    clean.write_bytes(b"")
    assert kudio.Pair(str(noisy), str(clean)).exists()


def test_split_sizes_and_disjointness():
    pairs = _pairs(100)
    train, val, test = kudio.split_pairs(pairs, 0.1, 0.2, seed=0)
    assert (len(train), len(val), len(test)) == (70, 10, 20)
    assert not (set(train) & set(val))
    assert not (set(train) & set(test))
    assert not (set(val) & set(test))
    assert set(train) | set(val) | set(test) == set(pairs)


def test_split_is_reproducible():
    pairs = _pairs(50)
    assert kudio.split_pairs(pairs, 0.2, 0.2, seed=7) == \
        kudio.split_pairs(pairs, 0.2, 0.2, seed=7)


def test_split_shuffles():
    pairs = _pairs(50)
    assert kudio.split_pairs(pairs, 0.0, 0.0, seed=1)[0] != list(pairs)


def test_tiny_set_keeps_something_to_train_on():
    """Percentages on two examples would otherwise leave an empty training set."""
    train, val, test = kudio.split_pairs(_pairs(2), 0.4, 0.4, seed=0)
    assert len(train) == 2 and not val and not test


def test_splits_that_leave_nothing_are_rejected():
    with pytest.raises(ValueError, match="leave something"):
        kudio.split_pairs(_pairs(10), 0.6, 0.5)


def test_out_of_range_splits_are_rejected():
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        kudio.split_pairs(_pairs(10), -0.1, 0.2)


def _repeated(n_distinct=10, times=2):
    """A manifest that lists every example more than once, which is what a
    repeated value in snr_ratio= leaves behind."""
    return _pairs(n_distinct) * times


def test_a_repeated_example_cannot_land_in_two_splits():
    """It would be trained on and then scored as if it had been held out,
    which inflates every number the run reports. The warning is checked
    elsewhere; what matters here is where the examples ended up.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        train, val, test = kudio.split_pairs(_repeated(), 0.2, 0.2, seed=0)
    assert not (set(train) & set(val))
    assert not (set(train) & set(test))
    assert not (set(val) & set(test))
    assert len(train) + len(val) + len(test) == 10
    assert set(train) | set(val) | set(test) == set(_pairs(10))
    mixtures = [p.noisy for p in train + val + test]
    assert len(set(mixtures)) == len(mixtures)


def test_the_dropped_rows_are_named_and_counted():
    """A row the caller counted on is gone, so the warning has to be enough
    to find the cause: how many went, and one that went."""
    pairs = _repeated() + [_pairs(10)[4]]
    with pytest.warns(UserWarning) as caught:
        kudio.split_pairs(pairs, 0.2, 0.2, seed=0)
    said = str(caught[0].message)
    assert "11 of 21 row(s)" in said
    assert "mix/4.wav is listed 3 times" in said
    said.encode("ascii")        # the console these land on


def test_rows_that_disagree_about_one_mixture_are_refused():
    """Two clean files cannot both be the answer for one mixture. Dropping
    either one would be a guess, so this is the case that raises."""
    pairs = _pairs(5) + [kudio.Pair(noisy="mix/2.wav", clean="clean/9.wav",
                                    noise="fan", snr_db=4)]
    with pytest.raises(ValueError, match="listed 2 times") as refused:
        kudio.split_pairs(pairs, 0.2, 0.2, seed=0)
    said = str(refused.value)
    assert said.startswith("mix/2.wav is listed 2 times")
    assert "disagree about clean, snr_db" in said
    said.encode("ascii")


def test_a_split_still_returns_three_lists_when_rows_repeat():
    """Callers unpack three lists; nothing about the rows may change that."""
    with pytest.warns(UserWarning):
        got = kudio.split_pairs(_repeated(), 0.2, 0.2, seed=0)
    assert len(got) == 3 and all(isinstance(part, list) for part in got)


def test_the_manifest_keeps_every_row_and_the_split_drops_the_repeats(tmp_path):
    """The manifest is the record of what was written, repeats included --
    that is where a name collision is still visible afterwards. The split is
    where a repeat does damage, so that is where it is dealt with.
    """
    path = kudio.save_manifest(tmp_path / "m.json", _repeated())
    rows = kudio.load_manifest(path)
    assert len(rows) == 20
    with pytest.warns(UserWarning):
        train, val, test = kudio.split_pairs(rows, 0.3, 0.0, seed=0)
    assert (len(train), len(val), len(test)) == (7, 3, 0)


# ------------------------------------------------------------------- labels

def _labels():
    return [kudio.Label(0.5, 1.5, "speech 1"),
            kudio.Label(2.0, 2.0, "click"),
            kudio.Label(3.0, 4.2, "speech 2")]


def test_labels_round_trip_as_json(tmp_path):
    path = kudio.save_labels(tmp_path / "a.labels.json", _labels())
    assert kudio.load_labels(path) == _labels()


def test_labels_round_trip_as_an_audacity_track(tmp_path):
    """Three columns of plain text that half the audio world can already read."""
    path = kudio.save_labels(tmp_path / "a.txt", _labels())
    assert path.read_text(encoding="utf-8").splitlines()[0] == \
        "0.500000\t1.500000\tspeech 1"
    assert kudio.load_labels(path) == _labels()


def test_the_format_follows_the_suffix(tmp_path):
    text = kudio.save_labels(tmp_path / "a.txt", _labels()).read_text("utf-8")
    assert "\t" in text and "{" not in text
    js = kudio.save_labels(tmp_path / "a.json", _labels()).read_text("utf-8")
    assert js.lstrip().startswith("[")


def test_the_suffix_can_be_overridden(tmp_path):
    path = kudio.save_labels(tmp_path / "a.json", _labels(), fmt="audacity")
    assert "\t" in path.read_text(encoding="utf-8")
    assert kudio.load_labels(path, fmt="audacity") == _labels()


def test_a_point_and_a_range_are_distinguishable():
    assert kudio.Label(2.0, 2.0).is_point is True
    assert kudio.Label(2.0, 3.0).is_point is False
    assert kudio.Label(2.0, 3.5).duration == pytest.approx(1.5)


def test_tabs_inside_a_label_do_not_break_the_row(tmp_path):
    """They are the column separator; Audacity has the same problem."""
    path = kudio.save_labels(tmp_path / "a.txt",
                             [kudio.Label(0.0, 1.0, "two\tparts")])
    assert kudio.load_labels(path) == [kudio.Label(0.0, 1.0, "two parts")]


def test_a_label_with_no_text_survives(tmp_path):
    path = kudio.save_labels(tmp_path / "a.txt", [kudio.Label(0.0, 1.0)])
    assert kudio.load_labels(path) == [kudio.Label(0.0, 1.0, "")]


def test_a_malformed_row_names_the_line(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("0.0\t1.0\tfine\n2.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        kudio.load_labels(path)


def test_a_row_whose_times_are_not_numbers_says_so(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("start\tend\theader row\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not.*times in seconds"):
        kudio.load_labels(path)


def test_blank_lines_are_skipped(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("\n0.0\t1.0\ta\n\n2.0\t3.0\tb\n\n", encoding="utf-8")
    assert len(kudio.load_labels(path)) == 2


def test_an_unknown_format_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="json.*audacity"):
        kudio.save_labels(tmp_path / "a.txt", _labels(), fmt="textgrid")


def test_save_labels_creates_parents(tmp_path):
    path = kudio.save_labels(tmp_path / "deep" / "a.txt", _labels())
    assert path.is_file()


def test_vad_output_becomes_labels(tmp_path, speech):
    """The obvious pipeline: find the speech, write it down, open it elsewhere."""
    spans = kudio.vad(speech, 16000)
    labels = [kudio.Label(a, b, f"speech {n}")
              for n, (a, b) in enumerate(spans, 1)]
    path = kudio.save_labels(tmp_path / "found.txt", labels)

    back = kudio.load_labels(path)
    assert [x.text for x in back] == [x.text for x in labels]
    # the text format keeps microseconds, so times come back rounded there --
    # 60x finer than a sample at 16 kHz, and 5x finer at 192 kHz
    for written, read in zip(labels, back):
        assert read.start == pytest.approx(written.start, abs=1e-6)
        assert read.end == pytest.approx(written.end, abs=1e-6)


def test_a_manifest_written_before_the_link_existed_still_loads(tmp_path):
    """Every field after `clean` is optional and appended, so an older run's
    manifest keeps working and a newer one degrades to what it understands.
    """
    import json
    path = tmp_path / "old.json"
    path.write_text(json.dumps([
        {"noisy": "a.wav", "clean": "b.wav", "noise": "n", "snr_db": 0},
    ]), encoding="utf-8")
    pair, = kudio.load_manifest(path)
    assert pair.rt60 is None and pair.channel is None and pair.target is None


def test_the_link_and_the_target_survive_a_round_trip(tmp_path):
    pairs = [kudio.Pair(noisy="a.wav", clean="b.wav", noise="n", snr_db=-5,
                        rt60=0.4, channel="mu_law+loss5%x4",
                        target="TARGETS/a.wav")]
    path = kudio.save_manifest(tmp_path / "m.json", pairs)
    assert kudio.load_manifest(path) == pairs


def test_a_pair_is_still_hashable_with_a_link_on_it():
    """`split_pairs` checks for overlap by putting pairs in a set, which
    silently catches nothing if they stop being hashable. A dict-valued
    `channel` would have done exactly that, which is why it is a string.
    """
    pair = kudio.Pair(noisy="a.wav", clean="b.wav", channel="telephone",
                      target="t.wav", rt60=0.3)
    assert len({pair, pair}) == 1


def test_a_fractional_snr_survives_the_manifest(tmp_path):
    """`int()` turned 2.5 dB into 2 on the way into the manifest."""
    pairs = [kudio.Pair(noisy="m.wav", clean="c.wav", snr_db=2.5),
             kudio.Pair(noisy="n.wav", clean="c.wav", snr_db=-5)]
    path = kudio.save_manifest(tmp_path / "m.json", pairs)
    back = kudio.load_manifest(path)
    assert [p.snr_db for p in back] == [2.5, -5]
    assert isinstance(back[1].snr_db, int)


# -- portable manifests and editor labels --------------------------------------

def test_a_manifest_names_its_files_with_forward_slashes(tmp_path):
    """Written on Windows with backslashes, it named no file on Linux."""
    windows = kudio.Pair(noisy=r"a\b\n.wav", clean=r"a\c.wav")
    path = kudio.save_manifest(tmp_path / "m.json", [windows])
    assert kudio.load_manifest(path) == [kudio.Pair(noisy="a/b/n.wav", clean="a/c.wav")]


def test_audacity_frequency_rows_are_skipped(tmp_path):
    """A label with a frequency range gets a second row starting with a
    backslash; it is not a time."""
    path = tmp_path / "labels.txt"
    tab, backslash = "\t", "\\"
    rows = [tab.join(["0.5", "1.5", "speech"]),
            tab.join([backslash, "100.000000", "3400.000000"]),
            tab.join(["2.0", "2.5", "click"])]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    labels = kudio.load_labels(path)
    assert [(lab.start, lab.end, lab.text) for lab in labels] == [
        (0.5, 1.5, "speech"), (2.0, 2.5, "click")]
