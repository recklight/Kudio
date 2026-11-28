# -*- coding: utf-8 -*-
import numpy as np

from kudio import AudioEvaluate
from kudio.core.evaluator import check_metrics_install, eval_metrics


def test_check_metrics_install_no_side_effects():
    status = check_metrics_install()
    assert len(status) == 3
    assert all(isinstance(s, bool) for s in status)


def test_eval_metrics_handles_missing_backends(sine):
    ref = sine
    deg = sine + 0.01 * np.random.default_rng(0).standard_normal(len(sine)).astype(np.float32)
    pesq, stoi, sdr = eval_metrics(16000, ref, deg)
    for value in (pesq, stoi, sdr):
        assert value is None or np.isfinite(value)


def test_audio_evaluate_constructs(wav_dir):
    files = sorted(wav_dir.glob("*.wav"))
    ev = AudioEvaluate(files, files)
    df = ev.get_df()        # nothing evaluated yet: an empty frame, not a list
    assert df.empty and "Enh_method" in df.columns


# -- the dependency-free additions ----------------------------------------------
import pytest

import kudio
from kudio.core import evaluator
from kudio.exceptions import DependencyError


def _pair(seconds=5.0, noise=0.05, seed=1):
    from conftest import make_speech
    ref = make_speech(16000, max(seconds, 5.0))
    deg = ref + noise * np.random.default_rng(seed).standard_normal(len(ref)
                                                                    ).astype(np.float32)
    return ref, deg


def test_sdi_is_snr_on_a_linear_scale():
    ref, deg = _pair()
    assert kudio.sdi(ref, ref) == pytest.approx(0.0, abs=1e-12)
    assert kudio.sdi(ref, deg) == pytest.approx(10 ** (-kudio.snr(ref, deg) / 10),
                                                rel=1e-6)
    assert kudio.sdi(ref, 2 * ref) == pytest.approx(1.0)


def test_score_runs_what_can_run_and_names_it():
    ref, deg = _pair()
    got = kudio.score(ref, deg, 16000)
    assert {'snr', 'si_sdr', 'segsnr', 'sdi'} <= set(got)
    assert all(isinstance(v, float) for v in got.values())
    assert 'pesq' not in kudio.score(ref, deg, 44100), "PESQ is 8/16 kHz only"
    assert kudio.score(ref, deg, 16000, ['sdi', 'snr']).keys() == {'sdi', 'snr'}
    assert kudio.score(ref, deg, 16000, 'snr') == {'snr': kudio.snr(ref, deg)}


def test_score_refuses_what_it_cannot_do(monkeypatch):
    ref, deg = _pair()
    with pytest.raises(ValueError, match="unknown metric"):
        kudio.score(ref, deg, 16000, ['pesq_wb'])
    monkeypatch.setattr(evaluator, "_has_module", lambda name: False)
    with pytest.raises(DependencyError, match=r"kudio\[pesq\]"):
        kudio.score(ref, deg, 16000, ['pesq'])
    assert set(kudio.score(ref, deg, 16000)) == {'snr', 'si_sdr', 'segsnr', 'sdi', 'sdr',
                                                 'stoi', 'estoi'}


_ASKED_FOR_ONLY = ('llr', 'is', 'cep', 'wss', 'fwsegsnr', 'lsd', 'csig', 'cbak',
                   'covl', 'ncm', 'csii_high', 'csii_mid', 'csii_low', 'siib',
                   'siib_gauss')


def test_score_runs_the_newer_measures_when_they_are_named():
    """Named, each is the number its own function gives. Not named, none of
    them runs: the default list is the eight score() has always run."""
    ref, deg = _pair()
    got = kudio.score(ref, deg, 16000, ['lsd', 'llr', 'is', 'cep', 'wss', 'fwsegsnr',
                                        'ncm', 'csii_high', 'csii_mid', 'csii_low'])
    levels = kudio.csii(ref, deg, 16000)
    assert got == {'lsd': kudio.lsd(ref, deg, 16000),
                   'llr': kudio.llr(ref, deg, 16000),
                   'is': kudio.itakura_saito(ref, deg, 16000),
                   'cep': kudio.cepstral_distance(ref, deg, 16000),
                   'wss': kudio.wss(ref, deg, 16000),
                   'fwsegsnr': kudio.fw_segmental_snr(ref, deg, 16000),
                   'ncm': kudio.ncm(ref, deg, 16000),
                   'csii_high': levels.high, 'csii_mid': levels.mid,
                   'csii_low': levels.low}
    assert set(kudio.score(ref, deg, 16000)).isdisjoint(_ASKED_FOR_ONLY)
    assert set(_ASKED_FOR_ONLY) <= set(kudio.core.evaluator.METRICS)


def test_score_runs_siib_by_name_and_says_when_the_pair_is_too_short():
    """6 s of the test signal hold the 5.45 s of speech SIIB needs, and 5 s
    do not; through score the warning is siib's own."""
    ref, deg = _pair(6.0)
    assert kudio.score(ref, deg, 16000, ['siib', 'siib_gauss']) == {
        'siib': kudio.siib(ref, deg, 16000),
        'siib_gauss': kudio.siib_gauss(ref, deg, 16000)}
    with pytest.warns(UserWarning, match=r"^siib_gauss\(\): the pair is 80000 samples"):
        got = kudio.score(ref[:80000], deg[:80000], 16000, ['siib_gauss'])
    assert np.isnan(got['siib_gauss'])


def test_score_takes_the_three_csii_levels_from_one_call(monkeypatch):
    ref, deg = _pair()
    calls = []

    def stand_in(r, d, sr):
        calls.append(sr)
        return kudio.CsiiScore(0.9, 0.7, 0.4)

    monkeypatch.setattr(kudio.core.intelligibility, "csii", stand_in)
    got = kudio.score(ref, deg, 16000, ['csii_low', 'snr', 'csii_high', 'csii_mid'])
    assert got == {'csii_low': 0.4, 'snr': kudio.snr(ref, deg), 'csii_high': 0.9,
                   'csii_mid': 0.7}
    assert calls == [16000]


def test_score_takes_the_three_ratings_from_one_composite(monkeypatch):
    ref, deg = _pair()
    calls = []

    def stand_in(r, d, sr):
        calls.append(sr)
        return kudio.CompositeScore(3.1, 2.2, 2.6, 2.0, 0.5, 30.0, 5.0)

    monkeypatch.setattr(evaluator, "_has_module", lambda name: True)
    monkeypatch.setattr(kudio.core.quality, "composite", stand_in)
    got = kudio.score(ref, deg, 16000, ['covl', 'snr', 'csig', 'cbak'])
    assert got == {'covl': 2.6, 'snr': kudio.snr(ref, deg), 'csig': 3.1, 'cbak': 2.2}
    assert calls == [16000]


def test_score_refuses_the_newer_measures_where_they_cannot_run(monkeypatch):
    ref, deg = _pair()
    with pytest.raises(ValueError, match="8 and 16 kHz"):
        kudio.score(ref, deg, 44100, ['wss'])
    with pytest.raises(ValueError, match="8 and 16 kHz"):
        kudio.score(ref, deg, 44100, ['ncm'])
    with pytest.raises(ValueError, match="8 kHz or more"):
        kudio.score(ref[::4], deg[::4], 4000, ['csii_mid'])
    monkeypatch.setattr(evaluator, "_has_module", lambda name: False)
    with pytest.raises(DependencyError, match=r"kudio\[pesq\]"):
        kudio.score(ref, deg, 16000, ['cbak'])


@pytest.mark.parametrize("kwargs, match", [
    (dict(mode='wb', scale='raw'), "no raw score"),
    (dict(mode='x'), "mode"),
    (dict(scale='mos'), "scale"),
])
def test_pesq_refuses_what_the_standards_do_not_define(kwargs, match):
    ref, deg = _pair()
    with pytest.raises(ValueError, match=match):
        kudio.pesq(ref, deg, 16000, **kwargs)


def test_pesq_refuses_rates_it_is_not_defined_at():
    ref, deg = _pair()
    with pytest.raises(ValueError, match="8 and 16 kHz"):
        kudio.pesq(ref, deg, 44100)
    with pytest.raises(ValueError, match="16 kHz"):
        kudio.pesq(ref[::2], deg[::2], 8000, mode='wb')


def test_missing_backends_are_reported_once(monkeypatch, caplog):
    """Per-file scoring would otherwise log the same install hint for every file."""
    monkeypatch.setattr(evaluator, "_has_module", lambda name: False)
    monkeypatch.setattr(evaluator, "_REPORTED", set())
    with caplog.at_level("WARNING", logger="kudio.core.evaluator"):
        evaluator.check_metrics_install()
        first = len(caplog.records)
        for _ in range(5):
            evaluator.check_metrics_install()
    assert first == 1 and len(caplog.records) == 1


def test_audio_evaluate_refuses_lists_that_do_not_pair_up(wav_dir):
    files = sorted(wav_dir.glob("*.wav"))
    with pytest.raises(ValueError, match="pair up"):
        AudioEvaluate(files, files[:-1])
    with pytest.raises(ValueError, match="no raw score"):
        AudioEvaluate(files, files, pesq_mode='wb', pesq_scale='raw')
    ev = AudioEvaluate(files, files)
    with pytest.raises(ValueError, match="clean references"):
        ev.eval('short', files[:1])


# -- with the back-ends installed ----------------------------------------------

def test_pesq_raw_is_the_p862_1_inverse_of_narrowband_lqo():
    pytest.importorskip("pesq")
    import math
    ref, deg = _pair()
    lqo = kudio.pesq(ref, deg, 16000, mode='nb')
    raw = kudio.pesq(ref, deg, 16000, mode='nb', scale='raw')
    assert 0.999 + 4 / (1 + math.exp(-1.4945 * raw + 4.6607)) == pytest.approx(lqo)
    assert raw > lqo, "in the low range the mapping pulls the score down"


def test_pesq_auto_is_what_kudio_always_reported():
    """pysepm gave wideband MOS-LQO at 16 kHz and narrowband at 8 kHz."""
    pytest.importorskip("pesq")
    ref, deg = _pair()
    assert kudio.pesq(ref, deg, 16000) == kudio.pesq(ref, deg, 16000, mode='wb')
    assert kudio.pesq(ref, deg, 16000, scale='raw') == \
        kudio.pesq(ref, deg, 16000, mode='nb', scale='raw')
    r8, d8 = ref[::2], deg[::2]
    assert kudio.pesq(r8, d8, 8000) == kudio.pesq(r8, d8, 8000, mode='nb')
    pesq_, _, _ = eval_metrics(16000, ref, deg)
    assert pesq_ == kudio.pesq(ref, deg, 16000)


def test_audio_evaluate_keeps_every_file_and_reads_any_format(tmp_path):
    """Means alone cannot be broken down by noise or SNR; scipy's reader
    could not open TIMIT's NIST SPHERE either."""
    import soundfile as sf
    from conftest import make_speech
    rng = np.random.default_rng(0)
    clean, noisy, enhanced = [], [], []
    for i in range(3):
        ref = make_speech(16000, 5.0, seed=i)
        sf.write(tmp_path / f"c{i}.sph", ref, 16000, format='NIST',
                 subtype='PCM_16')
        sf.write(tmp_path / f"n{i}.wav", ref + 0.1 * rng.standard_normal(
            len(ref)).astype(np.float32), 16000)
        sf.write(tmp_path / f"e{i}.wav", ref + 0.01 * rng.standard_normal(
            len(ref)).astype(np.float32), 16000)
        clean.append(tmp_path / f"c{i}.sph")
        noisy.append(tmp_path / f"n{i}.wav")
        enhanced.append(tmp_path / f"e{i}.wav")

    ev = AudioEvaluate(clean, noisy)
    ev.eval('good', enhanced)
    ev.eval('nothing', noisy)
    table = ev.per_file()
    assert len(table) == 9                       # 3 baseline + 2 methods x 3
    assert list(table['method']).count('Baseline') == 3
    assert set(table['noisy']) == {str(p) for p in noisy}
    # each row is the metric on that pair of files, read as file_load reads them
    good = table[table['method'] == 'good']
    for (_, row), ref_path, deg_path in zip(good.iterrows(), clean, enhanced):
        ref, sr = kudio.file_load(ref_path)
        deg, _ = kudio.file_load(deg_path)
        assert row['file'] == str(deg_path)
        assert row['STOI'] == pytest.approx(
            kudio.score(ref, deg, sr, ['stoi'])['stoi'])
    # scoring the noisy files as a "method" reproduces the baseline exactly
    same = table[table['method'] == 'nothing']['STOI'].values
    assert np.array_equal(same, table[table['method'] == 'Baseline']['STOI'].values)

    df = ev.get_df()
    assert list(df['Enh_method']) == ['good', 'nothing', 'Baseline']
    assert ev.get_df().shape == df.shape, "get_df grows with each call"
    assert df['STOI'].iloc[1] == pytest.approx(df['STOI'].iloc[2])



# -- what the metrics refuse, and files that cannot be scored ------------------

def test_stereo_is_refused_by_every_metric_alike():
    """Left alone, snr, sdi and segmental_snr would score both channels at
    once, and si_sdr would fail with numpy's shape error."""
    import numpy as np
    pair = np.zeros((1600, 2)) + 0.1
    for fn in (kudio.snr, kudio.si_sdr, kudio.segmental_snr, kudio.sdi):
        with pytest.raises(ValueError, match="mono"):
            fn(pair, pair)


@pytest.mark.parametrize("name", ["stoi", "estoi"])
def test_stoi_on_a_clip_too_short_for_it_is_nan(name):
    """STOI's 1e-5 placeholder is not a score to average in."""
    import numpy as np
    y = np.sin(2 * np.pi * 300 * np.arange(4000) / 16000)       # 0.25 s
    with pytest.warns(UserWarning, match="0.4 s"):
        assert np.isnan(kudio.score(y, y, 16000, [name])[name])


def _stoi_near_zero():
    """A tone against itself in white noise at 0.05: no envelope is left to share."""
    t = np.arange(16000) / 16000
    x = 0.3 * np.sin(2 * np.pi * 220 * t)
    return x, x + 0.05 * np.random.default_rng(0).standard_normal(16000)


def _envelopes_swapped():
    """500 Hz and 3 kHz swelling in turn at 4 Hz, and the other way round: every
    band's envelope and every frame's spectrum run against the reference's."""
    t = np.arange(16000) / 16000
    swell = np.sin(2 * np.pi * 4 * t)
    low, high = np.sin(2 * np.pi * 500 * t), np.sin(2 * np.pi * 3000 * t)
    return (0.2 * (low * (1.2 + swell) + high * (1.2 - swell)),
            0.2 * (low * (1.2 - swell) + high * (1.2 + swell)))


@pytest.mark.parametrize("name, pair", [("stoi", _stoi_near_zero), ("stoi", _envelopes_swapped),
                                        ("estoi", _envelopes_swapped)],
                         ids=["stoi near 0", "stoi swapped", "estoi swapped"])
def test_a_score_of_1e5_or_under_is_a_score(name, pair):
    """STOI and ESTOI are means of correlations, and one at 1e-5 or under 0 is a
    result: the worst items of a set, which a mean has to keep. Too short is said by
    how many frames are left, not by the number."""
    import warnings
    from kudio.core import _stoi
    ref, deg = pair()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        got = kudio.score(ref, deg, 16000, [name])[name]
    assert got < 1e-5
    assert got == float(_stoi.stoi(ref, deg, 16000, extended=(name == "estoi")))


def test_the_placeholder_stays_as_pystoi_gives_it():
    """kudio's STOI answers a clip too short for it as pystoi does, 1e-5 and a
    RuntimeWarning, and says so to score() with None."""
    from kudio.core import _stoi
    y = np.sin(2 * np.pi * 300 * np.arange(4000) / 16000)
    for extended in (False, True):
        assert _stoi.stoi_or_none(y, y, 16000, extended=extended) is None
        with pytest.warns(RuntimeWarning, match="fewer than 30 frames"):
            assert _stoi.stoi(y, y, 16000, extended=extended) == _stoi.TOO_SHORT == 1e-5


@pytest.mark.parametrize("name", ["stoi", "estoi"])
@pytest.mark.parametrize("silent", ["ref", "deg"])
def test_stoi_of_digital_silence_is_nan_on_every_call(name, silent):
    """Either side silent has no envelope to correlate, and it is nan with one
    warning each time."""
    import warnings
    from conftest import make_speech
    speech = make_speech(16000, 6.0).astype(np.float64)
    ref, deg = (np.zeros_like(speech), speech) if silent == "ref" else (speech,
                                                                        np.zeros_like(speech))
    for _ in range(6):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            value = kudio.score(ref, deg, 16000, [name])[name]
        assert np.isnan(value)
        assert [str(w.message) for w in caught] == [
            f"{name}(): {silent} is digital silence throughout, so there is no "
            f"envelope to correlate; scored nan"]
        assert caught[0].filename == __file__


def _snr_family_written_out(ref, deg):
    """snr, si_sdr, segmental_snr and sdi written out from their definitions,
    on a pair of one length."""
    snr = 10 * np.log10((np.sum(ref ** 2) + 1e-12) / (np.sum((ref - deg) ** 2) + 1e-12))
    r, d = ref - ref.mean(), deg - deg.mean()
    proj = (np.dot(d, r) / (np.dot(r, r) + 1e-12)) * r
    si_sdr = 10 * np.log10((np.sum(proj ** 2) + 1e-12) / (np.sum((d - proj) ** 2) + 1e-12))
    frames = []
    for start in range(0, len(ref) - 512 + 1, 256):
        a, b = ref[start:start + 512], deg[start:start + 512]
        value = 10 * np.log10((np.sum(a ** 2) + 1e-12) / (np.sum((a - b) ** 2) + 1e-12))
        frames.append(np.clip(value, -10, 35))
    sdi = np.sum((ref - deg) ** 2) / (np.sum(ref ** 2) + 1e-12)
    return float(snr), float(si_sdr), float(np.mean(frames)), float(sdi)


def test_the_snr_family_scores_an_ordinary_pair_by_its_definitions():
    """The input rule changes only what happens to input with nothing to
    score; a pair that scores gets the same four numbers to the last bit."""
    rng = np.random.default_rng(11)
    for trial in range(40):
        n = int(rng.integers(600, 40000))
        ref = rng.standard_normal(n) * 10 ** rng.uniform(-4, 0)
        deg = ref * rng.uniform(0.2, 2.0) + rng.standard_normal(n) * 10 ** rng.uniform(-5, 0)
        got = (kudio.snr(ref, deg), kudio.si_sdr(ref, deg),
               kudio.segmental_snr(ref, deg), kudio.sdi(ref, deg))
        assert got == _snr_family_written_out(ref, deg), trial


def test_a_constant_output_scores_0_db_si_sdr_and_says_where_that_comes_from():
    """Silence, or a DC offset alone, leaves nothing once its mean is out: no
    projection on the reference and no residual, and 0 dB is what the 1e-12
    guards make of 0 / 0. It stays a number, so an output that collapses
    still pulls a mean down instead of dropping out of it."""
    import warnings
    ref, _ = _pair()
    for deg in (np.zeros_like(ref), np.full_like(ref, 0.3)):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert kudio.si_sdr(ref, deg) == 0.0
        said = [str(w.message) for w in caught]
        assert len(said) == 1 and said[0].startswith("si_sdr(): deg is constant"), said
        assert caught[0].filename == __file__


def test_segmental_snr_says_how_many_frames_of_ref_are_digital_silence():
    """A frame that is silent in ref scores -10 dB, or 0 dB where deg is
    silent there too, so a copy padded with zeros scores under 35."""
    speech = np.random.default_rng(2).standard_normal(32000)
    padded = np.concatenate([np.zeros(8000), speech, np.zeros(8000)])
    with pytest.warns(UserWarning, match=r"^segmental_snr\(\): 59 of ref's 186 frames "
                                         r"are digital silence") as caught:
        value = kudio.segmental_snr(padded, padded)
    assert len(caught) == 1
    assert value == pytest.approx(35.0 * (186 - 59) / 186, abs=1e-12)


def test_one_sample_is_a_signal_rather_than_a_shape_error():
    """squeeze() makes a one-sample array 0-d, and the length check takes it as
    one sample, not as a shape it cannot score."""
    got = kudio.score(np.r_[0.5], np.r_[0.25], 16000, ['snr', 'sdi'])
    assert got['snr'] == pytest.approx(10 * np.log10(4.0))
    assert got['sdi'] == pytest.approx(0.25)


@pytest.mark.parametrize("name, case", [
    ("stoi", "under 25.6 ms"), ("stoi", "silent ref"), ("stoi", "silent deg"),
    ("sdr", "one sample"), ("sdr", "silent ref"), ("sdr", "silent deg")])
def test_score_gives_nan_where_pystoi_or_bss_eval_would_raise(name, case):
    """pystoi raises AxisError under 25.6 ms, and BSS Eval refuses a silent
    side and gives inf for one sample; through score each is nan with one
    warning, as every metric gives."""
    import warnings
    from conftest import make_speech
    speech = make_speech(16000, 6.0).astype(np.float64)
    ref, deg = {"under 25.6 ms": (speech[:300], speech[:300]),
                "one sample": (speech[8000:8001], speech[8000:8001] * 0.5),
                "silent ref": (np.zeros_like(speech), speech),
                "silent deg": (speech, np.zeros_like(speech))}[case]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = kudio.score(ref, deg, 16000, [name])[name]
    assert np.isnan(value)
    said = [str(w.message) for w in caught]
    assert len(said) == 1 and said[0].startswith(f"{name}(): "), said
    assert said[0].endswith("; scored nan"), said
    assert caught[0].filename == __file__


def test_audio_evaluate_names_once_per_pass_the_files_a_metric_could_not_score(
        tmp_path, monkeypatch):
    """A file a metric has nothing to score in gets nan in that column and
    drops out of both means beside it. One warning per column and pass says
    which files and why, instead of one unnamed warning per file."""
    import warnings
    from conftest import make_speech
    speech = make_speech(16000, 5.0)
    clean, noisy, out = [], [], []
    for i in range(3):
        for kind, wave, into in (("clean", speech, clean), ("noisy", speech * 0.5, noisy),
                                 ("out", np.zeros_like(speech) if i == 1 else speech, out)):
            path = tmp_path / f"{kind}{i}.wav"
            kudio.save_wave(str(path), wave, 16000)
            into.append(path)

    def stand_in(name, ref, deg, sr, *args):
        if not np.any(deg):
            return evaluator.scored_nan(name, "deg is digital silence throughout")
        return 0.5

    monkeypatch.setattr(evaluator, "check_metrics_install", lambda: (False, True, False))
    monkeypatch.setattr(evaluator, "_compute", stand_in)
    ev = AudioEvaluate(clean, noisy)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ev.eval("m", out)
    said = [str(w.message) for w in caught]
    assert len(said) == 1, said
    assert said[0].startswith("AudioEvaluate.eval(): STOI has no score for 1 of 3 files"), said
    assert "out1.wav (deg is digital silence throughout)" in said[0], said
    assert caught[0].filename == __file__
    assert ev.score_dict["STOI"] == [0.5] and ev.base_stoi == 0.5


def test_one_bad_file_does_not_stop_the_rest(wav_dir, tmp_path):
    import warnings
    import numpy as np
    files = sorted(wav_dir.glob("*.wav"))
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"not audio")
    ev = AudioEvaluate(files, files)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rows = ev._score_rows("m", [bad] + files[1:], ["SDR"])
    assert len(rows) == len(files) and np.isnan(rows[0]["SDR"])
    assert any("was not scored" in str(w.message) for w in caught)


# -- AudioEvaluate before eval(), and pesq.exe ---------------------------------
def test_a_score_before_eval_says_what_to_call(wav_dir):
    files = sorted(wav_dir.glob("*.wav"))
    with pytest.raises(ValueError, match="call eval"):
        AudioEvaluate(files, files).sdr_score()


def test_a_pesq_binary_that_is_not_there_is_named(wav_dir):
    files = sorted(wav_dir.glob("*.wav"))
    ev = AudioEvaluate(files, files)
    ev.enhanced_list = files
    with pytest.raises(FileNotFoundError, match="no-such-pesq-binary"):
        ev.pesq_score_exe("no-such-pesq-binary.exe")


import os
import subprocess


def _printed_by_version_2(raw):
    """What version 2.0 (October 2005) of the ITU reference program prints in
    narrowband mode: a last line with the raw score, a tab, and the MOS-LQO
    that P.862.1 maps it to."""
    lqo = 0.999 + 4.0 / (1.0 + np.exp(-1.4945 * raw + 4.6607))
    return ("\nPerceptual Evaluation of Speech Quality (PESQ)\n\n"
            f"\nP.862 Prediction (Raw MOS, MOS-LQO):  = {raw:.3f}\t{lqo:.3f}\n")


def _stand_in_pesq_exe(monkeypatch, printed):
    """Answer for pesq.exe with ``printed(reference, degraded)``, so the
    output is read where no binary is installed. Returns the calls made."""
    calls = []

    def run(args, **kwargs):
        calls.append(list(args))
        return subprocess.CompletedProcess(args, 0, stdout=printed(*args[2:]), stderr="")

    monkeypatch.setattr(evaluator.shutil, "which", lambda name: name)
    monkeypatch.setattr(evaluator.subprocess, "run", run)
    return calls


def test_pesq_exe_takes_the_raw_score_where_version_2_prints_two(wav_dir, tmp_path,
                                                                 monkeypatch):
    pytest.importorskip("pandas")
    clean = sorted(wav_dir.glob("*.wav"))
    enhanced = []
    for path in clean:      # under another name, so the stand-in can tell them apart
        enhanced.append(tmp_path / f"enhanced_{path.name}")
        enhanced[-1].write_bytes(path.read_bytes())
    ev = AudioEvaluate(clean, clean)
    ev.enhanced_list = enhanced

    def printed(ref, deg):
        return _printed_by_version_2(2.6 if os.path.basename(deg).startswith("enhanced_")
                                     else 1.8)

    calls = _stand_in_pesq_exe(monkeypatch, printed)
    assert ev.pesq_score_exe() == (pytest.approx(1.8), pytest.approx(2.6))
    # narrowband at the rate asked for: +wb would print no raw score at all
    assert len(calls) == 6
    assert all(args[1] == "+16000" and "+wb" not in args for args in calls)


#: what the ITU program ends with where it scores nothing: a file it cannot
#: read, and no utterance found, which version 2.0 reports and exits on
PESQ_EXE_ERRORS = {
    "file": "An error of type 1  (Could not open source file) occurred during processing.",
    "utterance": "Processing error!",
}


@pytest.mark.parametrize("error", sorted(PESQ_EXE_ERRORS))
def test_pesq_exe_drops_a_file_the_program_scores_nothing_for_and_names_it(wav_dir,
                                                                           monkeypatch,
                                                                           error):
    pytest.importorskip("pandas")
    files = sorted(wav_dir.glob("*.wav"))
    ev = AudioEvaluate(files, files)
    ev.enhanced_list = files

    def printed(ref, deg):
        if os.path.basename(deg) == files[0].name:
            return ("\nPerceptual Evaluation of Speech Quality (PESQ)\n\n"
                    + PESQ_EXE_ERRORS[error] + "\n")
        return _printed_by_version_2(3.0)

    _stand_in_pesq_exe(monkeypatch, printed)
    with pytest.warns(UserWarning, match=r"^AudioEvaluate\.pesq_score_exe\(\): pesq\.exe "
                                         r"gave no score in 2 of 6 runs") as said:
        assert ev.pesq_score_exe() == (pytest.approx(3.0), pytest.approx(3.0))
    message = str(said[0].message)
    assert str(files[0]) in message and PESQ_EXE_ERRORS[error] in message


def test_pesq_exe_reads_no_score_from_a_last_line_without_an_equals_sign(wav_dir,
                                                                         monkeypatch):
    """Even one that is a number: only the prediction line carries a score."""
    pytest.importorskip("pandas")
    files = sorted(wav_dir.glob("*.wav"))
    ev = AudioEvaluate(files, files)
    ev.enhanced_list = files
    _stand_in_pesq_exe(monkeypatch, lambda ref, deg: "\n3.000\n")
    with pytest.warns(UserWarning, match="gave no score in 6 of 6 runs"):
        baseline, enhanced = ev.pesq_score_exe()
    assert np.isnan(baseline) and np.isnan(enhanced)


@pytest.mark.parametrize("rate", [44100, 16000.5, True, "16000"])
def test_pesq_exe_refuses_a_rate_the_program_does_not_run_at(wav_dir, monkeypatch, rate):
    files = sorted(wav_dir.glob("*.wav"))
    ev = AudioEvaluate(files, files)
    ev.enhanced_list = files
    calls = _stand_in_pesq_exe(monkeypatch, lambda ref, deg: _printed_by_version_2(3.0))
    with pytest.raises(ValueError, match=r"^AudioEvaluate\.pesq_score_exe\(\): the ITU "
                                         r"program runs at 8000 or 16000 Hz"):
        ev.pesq_score_exe(rate=rate)
    assert calls == []
