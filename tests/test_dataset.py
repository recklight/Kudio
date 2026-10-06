# -*- coding: utf-8 -*-
"""Dataset manifests and splits."""
import json

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


def test_unknown_keys_are_dropped_not_fatal(tmp_path, caplog):
    path = tmp_path / "m.json"
    path.write_text(json.dumps([
        {"noisy": "a.wav", "clean": "b.wav", "future_field": 1}]),
        encoding="utf-8")
    loaded = kudio.load_manifest(path)
    assert loaded == [kudio.Pair(noisy="a.wav", clean="b.wav")]
    assert "future_field" in caplog.text


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
