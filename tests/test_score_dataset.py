# -*- coding: utf-8 -*-
"""kudio.score_dataset and ``kudio score``: a whole test set, paired by id,
scored as kudio.score scores each pair, and summarised for a table."""
import csv
import json
import re
import warnings

import numpy as np
import pytest

import kudio
from kudio.cli import main
from kudio.core.evaluator import _has_module

SR = 16000
SPEAKERS = 10


def _utterance(seed, sr=SR, seconds=1.2):
    """A voiced stretch on a pitch of its own over a little noise: enough for
    every metric here, and short enough to score twenty times over."""
    rng = np.random.default_rng(seed)
    n = int(sr * seconds)
    t = np.arange(n) / sr
    f0 = 100 + 80 * rng.random()
    voice = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 10))
    envelope = 0.5 - 0.5 * np.cos(2 * np.pi * 3 * t)
    return (0.1 * voice * envelope + 0.003 * rng.standard_normal(n)).astype(np.float32)


def _make_set(root, speakers=SPEAKERS, utterances=2, sr=SR):
    """clean/, noisy/ and enhanced/ laid out alike, spkNN/uN.wav; the
    'enhanced' side is the noisy one with half its noise taken out."""
    for k in range(speakers):
        for u in range(utterances):
            seed = 100 * k + u
            clean = _utterance(seed, sr)
            noise = np.random.default_rng(seed + 1).standard_normal(clean.size)
            noisy = kudio.add_noise_snr(clean, noise.astype(np.float32), snr_db=5, seed=0)
            enhanced = (0.5 * clean + 0.5 * noisy).astype(np.float32)
            for side, y in (("clean", clean), ("noisy", noisy), ("enhanced", enhanced)):
                path = root / side / f"spk{k:02d}" / f"u{u}.wav"
                path.parent.mkdir(parents=True, exist_ok=True)
                kudio.save_wave(str(path), y, sr)
    return root / "clean", root / "noisy", root / "enhanced"


@pytest.fixture
def testset(tmp_path):
    return _make_set(tmp_path)


def _quiet(**kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return kudio.score_dataset(**kwargs)


def _score(path_ref, path_deg, sr=SR, metrics=None):
    """kudio.score of one pair, from numpy's global generator seeded with 0
    as score_dataset seeds it: pystoi's ESTOI draws from it."""
    ref, _ = kudio.file_load(path_ref, sr=None)
    deg, _ = kudio.file_load(path_deg, sr=None)
    np.random.seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return kudio.score(ref, deg, sr, metrics)


# --- pairing ----------------------------------------------------------------------

def test_folders_pair_by_relative_path_and_score_as_score_does(testset):
    clean, _, enhanced = testset
    result = _quiet(deg=enhanced, reference=clean)
    ids = [row["id"] for row in result.rows]
    assert ids == sorted(f"spk{k:02d}/u{u}" for k in range(SPEAKERS) for u in range(2))
    # score's default set: these four always, then whichever of pesq, stoi,
    # estoi and sdr have their package installed
    assert result.columns[:4] == ("snr", "si_sdr", "segsnr", "sdi")
    for row in result.rows:
        expected = _score(clean / f"{row['id']}.wav", enhanced / f"{row['id']}.wav")
        assert {c: row[c] for c in result.columns} == expected
        assert row["reasons"] == {} and row["notes"] == []
    assert result.unscored == []


def test_lists_pair_up_as_folders_do(testset, tmp_path, monkeypatch):
    clean, _, enhanced = testset
    folders = _quiet(deg=enhanced, reference=clean, metrics=["si_sdr"])
    ids = [row["id"] for row in folders.rows]
    # an scp with paths relative to the working directory, as Kaldi reads it
    monkeypatch.chdir(tmp_path)
    (tmp_path / "deg.scp").write_text(
        "".join(f"{key} enhanced/{key}.wav\n" for key in reversed(ids)), encoding="utf-8")
    with open(tmp_path / "ref.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["path", "id", "speaker"])
        for key in ids:
            writer.writerow([str(clean / f"{key}.wav"), key, key.split("/")[0]])
    listed = _quiet(deg="deg.scp", reference=tmp_path / "ref.csv", metrics=["si_sdr"])
    assert [row["id"] for row in listed.rows] == ids[::-1]      # the first side's order
    assert sorted(listed.rows, key=lambda r: r["id"]) == folders.rows
    as_dict = _quiet(deg={key: enhanced / f"{key}.wav" for key in ids},
                     reference=clean, metrics=["si_sdr"])
    assert as_dict.rows == folders.rows


def test_sides_that_do_not_pair_up_are_refused_with_the_ids(testset):
    clean, _, enhanced = testset
    for k in range(7):
        (clean / f"spk{k:02d}" / "u1.wav").unlink()
    kudio.save_wave(str(clean / "extra.wav"), _utterance(5), SR)
    with pytest.raises(ValueError) as caught:
        kudio.score_dataset(enhanced, reference=clean)
    message = str(caught.value)
    assert message.startswith("score_dataset(): the sides do not pair up")
    assert "7 ids of deg have no reference file: spk00/u1, spk01/u1" in message
    assert "and 2 more" in message
    assert "reference has 1 id that deg has not: extra" in message


def test_an_id_twice_and_two_files_for_one_id_are_refused(testset, tmp_path):
    clean, _, enhanced = testset
    scp = tmp_path / "deg.scp"
    scp.write_text(f"a {enhanced / 'spk00/u0.wav'}\nb {enhanced / 'spk00/u1.wav'}\n"
                   f"a {enhanced / 'spk01/u0.wav'}\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"gives 1 id more than once: a \(lines 1, 3\)"):
        kudio.score_dataset(scp, reference=clean)
    import soundfile as sf
    sf.write(str(enhanced / "spk00" / "u0.flac"), _utterance(0), SR)
    with pytest.raises(ValueError, match=r"spk00/u0 \(u0\.flac, u0\.wav\)"):
        kudio.score_dataset(enhanced, reference=clean)


def test_a_listed_file_that_does_not_exist_is_refused(testset, tmp_path):
    clean, _, enhanced = testset
    scp = tmp_path / "deg.scp"
    scp.write_text("".join(f"spk{k:02d}/u{u} {enhanced}/spk{k:02d}/u{u}.wav\n"
                           for k in range(SPEAKERS) for u in range(2))
                   .replace("spk03/u1.wav", "spk03/u9.wav"), encoding="utf-8")
    with pytest.raises(ValueError, match=r"1 file of deg does not exist: spk03/u1"):
        kudio.score_dataset(scp, reference=clean)
    with pytest.raises(ValueError, match=r"is one audio file; kudio\.score scores"):
        kudio.score_dataset(enhanced / "spk00" / "u0.wav", reference=clean)
    (tmp_path / "deg.lst").write_bytes(b"\xff\xfe\x00binary")
    with pytest.raises(ValueError, match=r"is not a text file"):
        kudio.score_dataset(tmp_path / "deg.lst", reference=clean)


def test_metric_names_are_checked_before_anything_is_read(testset):
    clean, _, enhanced = testset
    with pytest.raises(ValueError, match=r"si_sdr scores against a reference, and none"):
        kudio.score_dataset(enhanced, metrics=["si_sdr", "wada_snr"])
    with pytest.raises(ValueError, match=r"unknown metric\(s\) \['segmental_snr'\]"):
        kudio.score_dataset(enhanced, reference=clean, metrics=["segmental_snr"])
    with pytest.raises(ValueError, match=r"snr named twice"):
        kudio.score_dataset(enhanced, reference=clean, metrics=["snr", "snr"])
    for bad in (dict(jobs=0), dict(confidence=1.5), dict(n_resamples=0),
                dict(pesq_mode="xx"), dict(sr=0)):
        with pytest.raises(ValueError, match=r"^score_dataset\(\): "):
            kudio.score_dataset(enhanced, reference=clean, **bad)


# --- rates ------------------------------------------------------------------------

def test_a_metric_that_cannot_run_at_the_rate_is_refused_up_front(tmp_path):
    clean, _, enhanced = _make_set(tmp_path, speakers=2, utterances=1, sr=22050)
    with pytest.raises(ValueError, match=r"ncm cannot score this set at 22050 Hz: .*"
                                         r"pass sr to resample"):
        kudio.score_dataset(enhanced, reference=clean, metrics=["si_sdr", "ncm"])
    result = _quiet(deg=enhanced, reference=clean, metrics=["si_sdr", "ncm"], sr=16000)
    assert result.sr == 16000 and not result.unscored
    ref = kudio.resample(kudio.file_load(clean / "spk00/u0.wav")[0], 22050, 16000)
    deg = kudio.resample(kudio.file_load(enhanced / "spk00/u0.wav")[0], 22050, 16000)
    row = result.rows[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert {c: row[c] for c in result.columns} == kudio.score(ref, deg, 16000,
                                                                 ["si_sdr", "ncm"])


def test_references_at_two_rates_are_refused(testset):
    clean, _, enhanced = testset
    kudio.save_wave(str(clean / "spk00" / "u0.wav"), _utterance(0, 8000), 8000)
    with pytest.raises(ValueError, match=r"the reference files are at 2 rates "
                                         r"\(8000 Hz: 1 file, 16000 Hz: 19 files\)"):
        kudio.score_dataset(enhanced, reference=clean)


def test_a_pair_at_two_rates_is_reported_on_its_own_item(testset):
    clean, _, enhanced = testset
    kudio.save_wave(str(enhanced / "spk04" / "u1.wav"), _utterance(401, 8000), 8000)
    result = _quiet(deg=enhanced, reference=clean, metrics=["snr", "si_sdr"])
    assert result.unscored == ["spk04/u1"]
    row = next(r for r in result.rows if r["id"] == "spk04/u1")
    assert row["snr"] is None and row["si_sdr"] is None
    assert row["reasons"]["snr"] == ("deg is at 8000 Hz and the set is scored at 16000 "
                                     "Hz; pass sr to resample every file as it is read")
    assert [s.n for s in result.summary] == [19, 19]
    assert [s.not_scored for s in result.summary] == [1, 1]
    resampled = _quiet(deg=enhanced, reference=clean, metrics=["snr"], sr=16000)
    assert not resampled.unscored


# --- what is not scored -----------------------------------------------------------

def test_an_item_a_metric_cannot_score_has_none_and_kudios_own_warning(testset, tmp_path):
    clean, _, enhanced = testset
    kudio.save_wave(str(clean / "spk02" / "u0.wav"), np.zeros(int(1.2 * SR), np.float32), SR)
    with pytest.warns(UserWarning, match=r"1 of 20 items have no score in some column"):
        result = kudio.score_dataset(enhanced, reference=clean, metrics=["snr", "sdi"])
    row = next(r for r in result.rows if r["id"] == "spk02/u0")
    assert row["snr"] is None and row["sdi"] is None
    with pytest.warns(UserWarning) as said:
        kudio.snr(np.zeros(int(1.2 * SR)), kudio.file_load(enhanced / "spk02/u0.wav")[0])
    assert row["reasons"]["snr"] == str(said[0].message)
    assert row["reasons"]["snr"].endswith("; scored nan")

    rows_csv = result.save_rows(tmp_path / "out" / "rows.csv")
    with open(rows_csv, newline="", encoding="utf-8") as fh:
        table = list(csv.DictReader(fh))
    assert list(table[0]) == ["id", "snr", "sdi", "reason", "note"]
    written = next(r for r in table if r["id"] == "spk02/u0")
    assert written["snr"] == "" and written["sdi"] == ""
    assert written["reason"] == f"{row['reasons']['snr']} | {row['reasons']['sdi']}"
    scored = next(r for r in table if r["id"] == "spk00/u0")
    assert float(scored["snr"]) == result.rows[0]["snr"]           # to the last bit

    rows_jsonl = result.save_rows(tmp_path / "rows.jsonl")
    lines = [json.loads(line) for line in rows_jsonl.read_text(encoding="utf-8").splitlines()]
    assert lines == [json.loads(json.dumps(r)) for r in result.rows]
    assert next(r for r in lines if r["id"] == "spk02/u0")["snr"] is None
    with pytest.raises(ValueError, match=r"a \.csv or a \.jsonl file"):
        result.save_rows(tmp_path / "rows.xlsx")


def test_a_length_cut_is_a_note_and_a_length_past_the_limit_a_reason(testset):
    clean, _, enhanced = testset
    y = kudio.file_load(enhanced / "spk01/u0.wav")[0]
    kudio.save_wave(str(enhanced / "spk01" / "u0.wav"), y[:-160], SR)
    kudio.save_wave(str(enhanced / "spk01" / "u1.wav"), y[:2000], SR)
    with warnings.catch_warnings(record=True) as said:
        warnings.simplefilter("always")
        result = kudio.score_dataset(enhanced, reference=clean, metrics=["si_sdr"])
    said = [str(w.message) for w in said]
    assert any(text.startswith("score_dataset(): 1 of 20 items have no score in some "
                               "column") and "si_sdr, for spk01/u1: score(): ref is 19200" in text
               for text in said)
    assert any("1 of 20 items were scored with a warning" in text for text in said)
    cut, short = result.rows[2], result.rows[3]
    assert cut["si_sdr"] is not None
    assert cut["notes"] and "only the first 19040 samples are scored" in cut["notes"][0]
    assert short["si_sdr"] is None and "unmeasured" in short["reasons"]["si_sdr"]


# --- reference-free metrics -------------------------------------------------------

def test_reference_free_metrics_score_the_file_alone(testset):
    _, _, enhanced = testset
    result = _quiet(deg=enhanced)
    assert result.columns == ("srmr", "wada_snr")
    y, _ = kudio.file_load(enhanced / "spk03/u1.wav")
    row = next(r for r in result.rows if r["id"] == "spk03/u1")
    assert row["srmr"] == kudio.srmr(y, SR)
    assert row["wada_snr"] == kudio.wada_snr(y, SR)


def test_mixed_reference_free_ones_survive_a_reference_that_cannot_be_read(testset):
    clean, _, enhanced = testset
    (clean / "spk05" / "u0.wav").write_bytes(b"not audio")
    result = _quiet(deg=enhanced, reference=clean, metrics=["si_sdr", "wada_snr"])
    row = next(r for r in result.rows if r["id"] == "spk05/u0")
    assert row["si_sdr"] is None and "Could not load" in row["reasons"]["si_sdr"]
    y, _ = kudio.file_load(enhanced / "spk05/u0.wav")
    assert row["wada_snr"] == kudio.wada_snr(y, SR)
    assert "wada_snr" not in row["reasons"]


class _Session:
    """A stand-in for an onnxruntime session: kudio.dnsmos reads its input's
    shape and feeds it windows."""

    opened = 0
    shape = [1, 144160]

    def __init__(self):
        type(self).opened += 1
        self.calls = 0

    def get_inputs(self):
        return [type("Input", (), {"name": "input_1", "shape": self.shape})()]

    def run(self, _, feed):
        self.calls += 1
        return [np.array([[2.0, 3.0, 2.5]], dtype=np.float32)]


@pytest.fixture
def dnsmos_model(tmp_path, monkeypatch):
    """A model file where DNS-Challenge keeps it, opened as the stand-in."""
    import importlib

    from kudio.core import evaluator
    backend = importlib.import_module("kudio.core.dnsmos")

    class Session(_Session):
        opened = 0
    monkeypatch.setattr(backend, "_session", lambda path: Session())
    real = evaluator._has_module
    monkeypatch.setattr(evaluator, "_has_module",
                        lambda name: True if name == "onnxruntime" else real(name))

    def place(folder, name="sig_bak_ovr.onnx"):
        model = tmp_path / "models" / folder / name
        model.parent.mkdir(parents=True, exist_ok=True)
        model.write_bytes(b"stand-in")
        return model
    return Session, place


def test_dnsmos_gives_three_columns_on_one_session(testset, dnsmos_model):
    from kudio.core import dataset_score
    Session, place = dnsmos_model
    model = place("DNSMOS")
    _, _, enhanced = testset
    result = _quiet(deg=enhanced, metrics=["dnsmos"], dnsmos_dir=model.parent)
    assert result.columns == ("dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl")
    assert Session.opened == 1                     # checked, then scored with
    assert dataset_score._SESSIONS == {}           # and let go of afterwards
    expected = kudio.dnsmos(kudio.file_load(enhanced / "spk00/u0.wav")[0], SR,
                            session=Session())
    assert result.rows[0]["dnsmos_ovrl"] == expected.ovrl
    assert [s.label for s in result.summary] == ["DNSMOS SIG", "DNSMOS BAK", "DNSMOS OVRL"]
    assert "|dnsmos:model=sig_bak_ovr.onnx@" in result.signatures["the means"]


def test_the_personalised_model_gets_the_warning_dnsmos_gives(testset, dnsmos_model):
    _, place = dnsmos_model
    model = place("pDNSMOS")
    _, _, enhanced = testset
    with warnings.catch_warnings(record=True) as said:
        warnings.simplefilter("always")
        kudio.score_dataset(enhanced, metrics=["dnsmos"], dnsmos_dir=model.parent)
    assert sum("is the personalised DNSMOS model, which the non-personalised mapping"
               in str(w.message) for w in said) == 1


def test_a_model_that_takes_the_other_input_is_refused_before_scoring(testset,
                                                                       dnsmos_model):
    from kudio.exceptions import FeatureError
    Session, place = dnsmos_model
    model = place("DNSMOS")                        # sig_bak_ovr.onnx, a P.835 model
    _, _, enhanced = testset
    with pytest.raises(FeatureError, match=r"^score_dataset\(\): dnsmos_p808 cannot run "
                                           r"on .*sig_bak_ovr\.onnx: the model takes "
                                           r"input of shape \[1, 144160\]"):
        kudio.score_dataset(enhanced, metrics=["dnsmos_p808"], dnsmos_dir=model)
    assert Session.opened == 1


# --- align, groups, baseline ------------------------------------------------------

def test_align_lines_each_pair_up_first_and_says_so(testset):
    clean, _, enhanced = testset
    for k in range(SPEAKERS):
        path = enhanced / f"spk{k:02d}" / "u0.wav"
        y = kudio.file_load(path)[0]
        kudio.save_wave(str(path), np.concatenate([np.zeros(40, np.float32), y]), SR)
    as_they_sit = _quiet(deg=enhanced, reference=clean, metrics=["si_sdr"])
    aligned = _quiet(deg=enhanced, reference=clean, metrics=["si_sdr"], align=True)
    ref, _ = kudio.file_load(clean / "spk00/u0.wav")
    deg, _ = kudio.file_load(enhanced / "spk00/u0.wav")
    assert aligned.rows[0]["si_sdr"] == kudio.si_sdr(*kudio.align(ref, deg, SR))
    assert aligned.rows[0]["si_sdr"] > as_they_sit.rows[0]["si_sdr"] + 10
    for run, said in ((aligned, "yes"), (as_they_sit, "no")):
        assert kudio.parse_signature(run.signatures["the means"])["pair"]["aligned"] == said


def test_groups_from_a_file_a_pattern_or_a_dict(testset, tmp_path):
    clean, _, enhanced = testset
    utt2spk = tmp_path / "utt2spk"
    utt2spk.write_text("".join(f"spk{k:02d}/u{u} S{k}\n" for k in range(SPEAKERS)
                               for u in range(2)), encoding="utf-8")
    table = tmp_path / "speakers.csv"
    table.write_text("id,speaker\n" + "".join(f"spk{k:02d}/u{u},S{k}\n"
                                               for k in range(SPEAKERS) for u in range(2)),
                     encoding="utf-8")
    runs = [_quiet(deg=enhanced, reference=clean, metrics=["si_sdr"], groups=groups)
            for groups in (utt2spk, table, re.compile(r"^spk(\d+)/"),
                           {f"spk{k:02d}/u{u}": k for k in range(SPEAKERS) for u in range(2)})]
    for run in runs:
        assert run.summary[0].mean.n_groups == SPEAKERS
        assert run.summary[0].mean == runs[0].summary[0].mean
    values = [row["si_sdr"] for row in runs[0].rows]
    speakers = [row["id"].split("/")[0] for row in runs[0].rows]
    assert runs[0].summary[0].mean == kudio.bootstrap_ci(values, groups=speakers)
    assert runs[0].rows[0]["group"] == "S0"
    fields = kudio.parse_signature(runs[0].signatures["the means"])
    assert fields["ci"]["groups"] == str(SPEAKERS)
    with pytest.raises(ValueError, match=r"finds no speaker in 20 ids"):
        kudio.score_dataset(enhanced, reference=clean, groups=re.compile(r"^p(\d+)_"))
    with pytest.raises(ValueError, match=r"has no group in parentheses"):
        kudio.score_dataset(enhanced, reference=clean, groups=re.compile(r"^spk"))


def test_a_baseline_adds_the_difference_its_test_and_holm(testset):
    clean, noisy, enhanced = testset
    result = _quiet(deg=enhanced, reference=clean, baseline=noisy,
                    metrics=["si_sdr", "segsnr", "sdi"], groups=re.compile(r"^(spk\d+)/"))
    speakers = [row["group"] for row in result.rows]
    p = {}
    for s in result.summary:
        mine = [row[s.column] for row in result.rows]
        base = [row[f"baseline_{s.column}"] for row in result.rows]
        assert s.baseline == kudio.bootstrap_ci(base, groups=speakers)
        assert s.difference == kudio.paired_ci(mine, base, groups=speakers)
        assert s.test == kudio.permutation_test(mine, base, groups=speakers)
        assert s.mean.estimate - s.baseline.estimate == pytest.approx(s.difference.estimate)
        p[s.column] = s.test.pvalue
    assert {s.column: s.p_holm for s in result.summary} == kudio.holm(p)
    assert list(result.signatures) == ["the means and the baselines", "the differences"]
    kinds = [kudio.parse_signature(text)["ci"]["kind"]
             for text in result.signatures.values()]
    assert kinds == ["bootstrap_ci", "paired_ci"]
    assert "Difference [95% CI]" in result.to_markdown()


def test_a_column_that_lost_items_signs_on_a_line_of_its_own(testset):
    clean, _, enhanced = testset
    kudio.save_wave(str(clean / "spk06" / "u1.wav"), np.zeros(int(1.2 * SR), np.float32), SR)
    result = _quiet(deg=enhanced, reference=clean, metrics=["si_sdr", "wada_snr"])
    assert [s.n for s in result.summary] == [19, 20]
    assert list(result.signatures) == ["the mean of SI-SDR (dB)",
                                       "the mean of WADA-SNR (dB)"]
    by_n = {kudio.parse_signature(text)["n"]: text for text in result.signatures.values()}
    assert set(by_n) == {"19", "20"}
    assert "|si_sdr|" in by_n["19"] and "wada_snr" not in by_n["19"]
    assert "|wada_snr|" in by_n["20"] and "si_sdr" not in by_n["20"]


# --- several processes ------------------------------------------------------------

def test_two_processes_give_what_one_gives(testset):
    """The same rows, reasons and notes in the same order. A worker runs its
    BLAS on its share of the threads, and the dot products of SI-SDR then
    land within 1e-12 rather than on the same last bit; segsnr, which has
    none, does."""
    clean, noisy, enhanced = testset
    metrics = ["si_sdr", "segsnr", "srmr"] + (["estoi"] if _has_module("pystoi") else [])
    one = _quiet(deg=enhanced, reference=clean, baseline=noisy, metrics=metrics)
    two = _quiet(deg=enhanced, reference=clean, baseline=noisy, metrics=metrics, jobs=2)
    assert [row["id"] for row in two.rows] == [row["id"] for row in one.rows]
    for mine, theirs in zip(one.rows, two.rows):
        assert set(theirs) == set(mine)
        for key, value in mine.items():
            if isinstance(value, float):
                assert theirs[key] == pytest.approx(value, rel=1e-12, abs=1e-12), key
            else:
                assert theirs[key] == value, key
        assert (theirs["segsnr"], theirs["baseline_segsnr"]) == \
            (mine["segsnr"], mine["baseline_segsnr"])
    assert list(two.signatures) == list(one.signatures)
    for a, b in zip(one.summary, two.summary):
        assert b.mean.estimate == pytest.approx(a.mean.estimate, rel=1e-12)


def test_a_raised_length_limit_reaches_the_workers(tmp_path, monkeypatch):
    """A spawned worker imports kudio afresh, where the limit is 0.5 again."""
    from kudio.core import evaluator
    clean, _, enhanced = _make_set(tmp_path, speakers=2, utterances=2)
    y = kudio.file_load(enhanced / "spk01/u1.wav")[0]
    kudio.save_wave(str(enhanced / "spk01" / "u1.wav"), y[:int(0.375 * y.size)], SR)
    monkeypatch.setattr(evaluator, "MAX_LENGTH_MISMATCH", 0.9)
    one = _quiet(deg=enhanced, reference=clean, metrics=["snr"])
    two = _quiet(deg=enhanced, reference=clean, metrics=["snr"], jobs=2)
    assert one.summary[0].n == two.summary[0].n == 4
    assert one.rows == two.rows
    assert kudio.parse_signature(two.signatures["the means"])["pair"]["limit"] == "0.9"


def test_each_worker_gets_a_share_of_the_blas_threads(monkeypatch):
    import os

    from kudio.core import dataset_score
    for name in dataset_score._THREADS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MKL_NUM_THREADS", "3")
    monkeypatch.setattr(os, "cpu_count", lambda: 20)
    with dataset_score._worker_threads(4) as share:
        assert share == "5"
        assert os.environ["OPENBLAS_NUM_THREADS"] == os.environ["OMP_NUM_THREADS"] == "5"
        assert os.environ["MKL_NUM_THREADS"] == "3"            # set already, so kept
    assert "OPENBLAS_NUM_THREADS" not in os.environ and "OMP_NUM_THREADS" not in os.environ
    assert os.environ["MKL_NUM_THREADS"] == "3"
    with dataset_score._worker_threads(64) as share:
        assert share == "1"


@pytest.mark.skipif(not _has_module("pesq"), reason="needs the pesq package")
def test_pesq_runs_whole_as_score_runs_it(testset):
    clean, _, enhanced = testset
    result = _quiet(deg=enhanced, reference=clean, metrics=["pesq", "si_sdr"],
                    pesq_mode="nb", pesq_scale="raw")
    for row in result.rows[:4]:
        expected = _score(clean / f"{row['id']}.wav", enhanced / f"{row['id']}.wav",
                          metrics=["pesq", "si_sdr"])
        ref, _ = kudio.file_load(clean / f"{row['id']}.wav")
        deg, _ = kudio.file_load(enhanced / f"{row['id']}.wav")
        assert row["pesq"] == kudio.pesq(ref, deg, SR, mode="nb", scale="raw")
        assert row["si_sdr"] == expected["si_sdr"]
    assert result.summary[0].label == "PESQ" and "PESQ (NB raw)" in result.to_markdown()


# --- the summary as text ----------------------------------------------------------

def test_the_three_formats_and_the_references(testset):
    clean, _, enhanced = testset
    result = _quiet(deg=enhanced, reference=clean, metrics=["si_sdr", "sdi"],
                    groups=re.compile(r"^(spk\d+)/"))
    markdown = result.to_markdown().splitlines()
    assert markdown[0] == "| Metric | Better | Mean [95% CI] | n | Not scored |"
    s = result.summary[1]
    assert markdown[3] == (f"| SDI | lower | {s.mean.estimate:.3f} [{s.mean.low:.3f}, "
                           f"{s.mean.high:.3f}] | 20 | 0 |")
    assert markdown[-2] == f"Signature: {result.signatures['the means']}"
    assert markdown[-1] == "References: kudio, leroux2019sdr, luo2019convtasnet"

    latex = result.to_latex().splitlines()
    assert latex[:4] == [r"\begin{tabular}{lrr}", r"\toprule",
                         r"Metric & Mean [95\% CI] & $n$ \\", r"\midrule"]
    assert latex[4].startswith(r"SI-SDR (dB) $\uparrow$ & ")
    assert latex[5].startswith(r"SDI $\downarrow$ & ")
    assert latex[6:8] == [r"\bottomrule", r"\end{tabular}"]
    assert latex[8] == f"% signature: {result.signatures['the means']}"

    summary = json.loads(result.to_json())
    assert summary["items"] == 20 and summary["grouped"] is True
    assert summary["columns"][0]["mean"]["estimate"] == result.summary[0].mean.estimate
    assert summary["columns"][0]["mean"]["n_groups"] == SPEAKERS
    assert summary["signatures"] == [{"for": "the means",
                                      "signature": result.signatures["the means"]}]
    assert result.cite() == kudio.cite(["si_sdr", "sdi"])


def test_a_dataframe_when_pandas_is_there(testset):
    pytest.importorskip("pandas")
    clean, _, enhanced = testset
    frame = _quiet(deg=enhanced, reference=clean, metrics=["si_sdr"]).to_dataframe()
    assert list(frame.columns) == ["id", "si_sdr", "reason", "note"]
    assert frame["si_sdr"].dtype == float and len(frame) == 20


# --- the command ------------------------------------------------------------------

def test_the_command_prints_the_table_and_writes_rows_and_bibtex(testset, tmp_path, capsys):
    clean, noisy, enhanced = testset
    rows, bib = tmp_path / "rows.csv", tmp_path / "refs.bib"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        code = main(["score", str(enhanced), "--reference", str(clean),
                     "--baseline", str(noisy), "--metrics", "si_sdr", "sdi",
                     "--groups", r"^(spk\d+)/", "--per-item", str(rows),
                     "--bib", str(bib)])
    out, err = capsys.readouterr()
    assert code == 0
    assert out.startswith("| Metric | Better | Baseline [95% CI] | Mean [95% CI] |")
    assert "Signature for the differences: kudio:" in out and "groups=10" in out
    assert f"wrote 20 row(s) to {rows}" in err
    assert bib.read_text(encoding="utf-8") == kudio.cite(["si_sdr", "sdi"])
    with open(rows, newline="", encoding="utf-8") as fh:
        assert next(csv.reader(fh))[:4] == ["id", "group", "si_sdr", "sdi"]


def test_the_command_exits_1_when_an_item_has_no_score(testset, capsys):
    clean, _, enhanced = testset
    kudio.save_wave(str(clean / "spk02" / "u0.wav"), np.zeros(int(1.2 * SR), np.float32), SR)
    assert main(["score", str(enhanced), "--reference", str(clean),
                 "--metrics", "snr", "--format", "json"]) == 1
    out, err = capsys.readouterr()
    summary = json.loads(out)                            # standard output is the JSON alone
    assert summary["columns"][0]["not_scored"] == 1
    assert "warning: score_dataset(): 1 of 20 items have no score" in err


def test_the_command_refuses_bad_input_on_one_line(testset, capsys):
    clean, _, enhanced = testset
    (clean / "spk00" / "u0.wav").unlink()
    assert main(["score", str(enhanced), "--reference", str(clean)]) == 2
    err = capsys.readouterr().err
    assert err.startswith("kudio score: score_dataset(): the sides do not pair up")
    assert "Traceback" not in err and len(err.strip().splitlines()) == 1
    assert main(["score", str(enhanced), "--groups", "speakers.csv"]) == 2
    assert "is not a file, and as a regular expression it has no group" in \
        capsys.readouterr().err
    assert main(["score", str(enhanced), "--per-item", "rows.txt"]) == 2
    assert "--per-item writes a .csv or a .jsonl file" in capsys.readouterr().err


# --- checked before anything is scored --------------------------------------------

def test_a_group_label_that_is_none_is_refused_with_its_id(testset):
    clean, _, enhanced = testset
    labels = {f"spk{k:02d}/u{u}": f"S{k}" for k in range(SPEAKERS) for u in range(2)}
    labels["spk03/u1"] = float("nan")
    with pytest.raises(ValueError, match=r"^score_dataset\(\): groups gives spk03/u1 the "
                                         r"speaker nan, which cannot label a group"):
        kudio.score_dataset(enhanced, reference=clean, groups=labels)
    with pytest.raises(ValueError, match=r"groups gives spk00/u0 the speaker \['S', 0\]"):
        kudio.score_dataset(enhanced, reference=clean,
                            groups=lambda key: ["S", int(key[3:5])])


def test_a_dict_side_without_a_path_is_refused_with_its_id(testset):
    clean, _, enhanced = testset
    side = {f"spk{k:02d}/u{u}": enhanced / f"spk{k:02d}/u{u}.wav"
            for k in range(SPEAKERS) for u in range(2)}
    side["spk02/u0"] = None
    with pytest.raises(ValueError, match=r"deg gives no path for 1 id: spk02/u0"):
        kudio.score_dataset(side, reference=clean)


def test_the_rate_hint_does_not_ask_for_sr_when_sr_was_given(tmp_path):
    clean, _, enhanced = _make_set(tmp_path, speakers=1, utterances=2)
    with pytest.raises(ValueError) as caught:
        kudio.score_dataset(enhanced, reference=clean, metrics=["ncm"], sr=22050)
    assert "pass an sr it is defined at" in str(caught.value)
    assert "pass sr to resample" not in str(caught.value)


# --- files and their channels -----------------------------------------------------

def test_every_format_soundfile_reads_is_taken_and_the_rest_is_named(tmp_path):
    import soundfile as sf
    clean, _, enhanced = _make_set(tmp_path, speakers=1, utterances=2)
    for side in (clean, enhanced):
        y = kudio.file_load(side / "spk00/u1.wav")[0]
        (side / "spk00" / "u1.wav").unlink()
        sf.write(str(side / "spk00" / "u1.ogg"), y, SR)
        sf.write(str(side / "spk00" / "u2.aiff"), y, SR)
    (enhanced / "spk00" / "u0.txt").write_text("a transcript", encoding="utf-8")
    (enhanced / "notes.md").write_text("not an utterance", encoding="utf-8")
    with pytest.warns(UserWarning) as said:
        result = kudio.score_dataset(enhanced, reference=clean, metrics=["snr"])
    assert [row["id"] for row in result.rows] == ["spk00/u0", "spk00/u1", "spk00/u2"]
    left = [str(w.message) for w in said if "not audio" in str(w.message)]
    assert len(left) == 1 and left[0].endswith("and left out: notes.md")


def test_a_file_of_several_channels_is_averaged_and_said(testset):
    clean, _, enhanced = testset
    y = kudio.file_load(enhanced / "spk00/u0.wav")[0]
    kudio.save_wave(str(enhanced / "spk00" / "u0.wav"), np.stack([y, 0.5 * y], axis=1), SR)
    result = _quiet(deg=enhanced, reference=clean, metrics=["si_sdr"])
    row = result.rows[0]
    assert row["notes"] == ["deg has 2 channels, averaged to one before it was scored"]
    ref, _ = kudio.file_load(clean / "spk00/u0.wav")
    mixed, _ = kudio.file_load(enhanced / "spk00/u0.wav")       # mono by default
    assert row["si_sdr"] == kudio.si_sdr(ref, mixed)


# --- the text of a reason ---------------------------------------------------------

def test_reasons_and_notes_are_what_score_says_with_the_baseline_named(testset):
    clean, noisy, enhanced = testset
    y = kudio.file_load(noisy / "spk01/u0.wav")[0]
    kudio.save_wave(str(noisy / "spk01" / "u0.wav"), y[:-160], SR)
    kudio.save_wave(str(enhanced / "spk01" / "u0.wav"), y[:-160], SR)
    result = _quiet(deg=enhanced, reference=clean, baseline=noisy, metrics=["si_sdr"])
    row = result.rows[2]
    ref, _ = kudio.file_load(clean / "spk01/u0.wav")
    with warnings.catch_warnings(record=True) as said:
        warnings.simplefilter("always")
        kudio.score(ref, y[:-160], SR, ["si_sdr"])
    assert row["notes"] == [str(said[0].message)]
    assert row["baseline_notes"] == [str(said[0].message).replace("and deg", "and baseline")]


def test_a_nan_baseline_is_refused_as_such_before_it_is_aligned(testset):
    import soundfile as sf
    clean, noisy, enhanced = testset
    y = kudio.file_load(noisy / "spk00/u0.wav")[0]
    y[100] = np.nan
    sf.write(str(noisy / "spk00" / "u0.wav"), y, SR, subtype="FLOAT")
    result = _quiet(deg=enhanced, reference=clean, baseline=noisy, metrics=["si_sdr"],
                    align=True)
    reason = result.rows[0]["baseline_reasons"]["si_sdr"]
    assert reason == ("score(): baseline has 1 sample(s) that are not finite (NaN or "
                      "inf); repair the clip first")


def test_a_reason_is_not_repeated_as_a_note():
    """The composite's warning is given once for its three columns; the column
    it left without a score keeps it as its reason, and a sibling scored all
    the same does not keep it as a note."""
    from kudio.core.dataset_score import _Side
    side = _Side()
    said = ["composite(): something about cbak; scored nan"]
    side.record("csig", 2.5, said, None)
    side.record("cbak", float("nan"), said, None)
    result = side.result()
    assert result["reasons"] == {"cbak": said[0]} and result["notes"] == []


# --- the summary, signed ----------------------------------------------------------

def test_one_item_signs_no_interval_and_prints_none(tmp_path):
    clean, _, enhanced = _make_set(tmp_path, speakers=1, utterances=1)
    result = _quiet(deg=enhanced, reference=clean, metrics=["si_sdr"])
    fields = kudio.parse_signature(result.signatures["the means"])
    assert fields["n"] == "1" and "ci" not in fields
    assert " [n/a] | 1 | 0 |" in result.to_markdown()
    assert " [n/a] & 1 \\\\" in result.to_latex()


def test_a_method_that_differs_on_one_side_is_signed_for_that_side():
    from kudio.core.dataset_score import MetricSummary, _signatures
    from kudio.core.stats import ConfidenceInterval, PermutationResult

    def interval(method):
        return ConfidenceInterval(1.0, 0.5, 1.5, 0.95, 20, None, 10000, method)
    test = PermutationResult(0.5, 0.01, 20, None, 10000, False, "two-sided")
    summaries = [MetricSummary("snr", "snr", "SNR", "higher", "dB", 20, 0, interval("bca"),
                               interval(method), interval("bca"), test, 0.02)
                 for method in ("bca", "percentile")]
    summaries[1] = summaries[1]._replace(column="sdi", metric="sdi", label="SDI", unit="")
    lines = _signatures(summaries, {"snr": None, "sdi": None}, {"snr": {}, "sdi": {}},
                        lambda metric, label, unit: label, sr=SR, pesq_mode="auto",
                        pesq_scale="lqo", aligned=False, cut=True, resampled=False,
                        confidence=0.95, n_resamples=10000, seed=0)
    assert list(lines) == ["the means and the baseline of SNR", "the baseline of SDI",
                           "the differences"]
    assert kudio.parse_signature(lines["the baseline of SDI"])["ci"]["method"] == "percentile"


@pytest.mark.skipif(not _has_module("pesq"), reason="needs the pesq package")
def test_a_line_for_pesq_alone_says_the_run_held_its_pairs_to_the_limit(testset):
    clean, _, enhanced = testset
    kudio.save_wave(str(enhanced / "spk04" / "u0.wav"), np.zeros(int(1.2 * SR), np.float32),
                    SR)
    result = _quiet(deg=enhanced, reference=clean, metrics=["pesq", "si_sdr"])
    assert [s.n for s in result.summary] == [19, 20]       # PESQ gets nan for silence
    pair = kudio.parse_signature(result.signatures["the mean of PESQ (WB MOS-LQO)"])["pair"]
    assert pair == {"aligned": "no", "cut": "shorter", "limit": "0.5", "whole": "pesq"}


def test_resampling_is_signed(tmp_path):
    clean, _, enhanced = _make_set(tmp_path, speakers=1, utterances=2)
    resampled = _quiet(deg=enhanced, reference=clean, metrics=["snr"], sr=8000)
    as_read = _quiet(deg=enhanced, reference=clean, metrics=["snr"])
    assert kudio.parse_signature(resampled.signatures["the means"])["resample"] == "soxr_hq"
    assert "resample" not in kudio.parse_signature(as_read.signatures["the means"])


def test_a_warning_of_the_summary_names_the_columns_it_is_about(tmp_path):
    clean, noisy, enhanced = _make_set(tmp_path, speakers=4, utterances=3)
    with warnings.catch_warnings(record=True) as said:
        warnings.simplefilter("always")
        kudio.score_dataset(enhanced, reference=clean, baseline=noisy,
                            metrics=["snr", "sdi"], groups=re.compile(r"^(spk\d+)/"))
    said = [str(w.message) for w in said]
    assert not [text for text in said if text.startswith(("bootstrap_ci", "paired_ci",
                                                          "permutation_test"))]
    assert "score_dataset(): for the means and the baselines: bootstrap_ci(): 4 groups " \
           "are too few" in " ".join(said)
    assert any(text.startswith("score_dataset(): for the differences: permutation_test()")
               for text in said)


def test_a_column_without_a_p_value_is_named_outside_holms_family(testset):
    clean, noisy, enhanced = testset
    result = _quiet(deg=enhanced, reference=clean, baseline=noisy, metrics=["si_sdr", "siib"])
    assert result.holm_over == ["si_sdr"]
    siib = f"SIIB ({kudio.metric_info('siib').unit})"
    assert (f"Holm's adjustment is over SI-SDR (dB); {siib} has no p-value and is left "
            f"out of it.") in result.to_markdown().splitlines()
    assert json.loads(result.to_json())["holm"] == {"over": ["si_sdr"], "left_out": ["siib"]}


def test_a_p_value_near_a_level_shows_which_side_of_it_it_is_on():
    from kudio.core.dataset_score import _p_text
    assert _p_text(410 / 8192) == "0.05005"
    assert _p_text(500 / 10001) == "0.049995"
    assert _p_text(0.05) == "0.0500"
    assert _p_text(0.0123) == "0.0123"
    assert _p_text(1 / 10001) == "0.0001" and _p_text(2 ** -20) == "<0.0001"
