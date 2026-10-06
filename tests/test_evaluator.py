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
    assert ev.get_df() == []  # nothing evaluated yet


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
    with pytest.raises(DependencyError, match=r"kudio\[eval\]"):
        kudio.score(ref, deg, 16000, ['stoi'])
    assert set(kudio.score(ref, deg, 16000)) == {'snr', 'si_sdr', 'segsnr', 'sdi'}


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
    """Per-file scoring used to log the same install hint for every file."""
    monkeypatch.setattr(evaluator, "_has_module", lambda name: False)
    monkeypatch.setattr(evaluator, "_REPORTED", set())
    with caplog.at_level("WARNING", logger="kudio.core.evaluator"):
        evaluator.check_metrics_install()
        first = len(caplog.records)
        for _ in range(5):
            evaluator.check_metrics_install()
    assert first == 3 and len(caplog.records) == 3


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
    pytest.importorskip("pystoi")
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
    assert ev.get_df().shape == df.shape, "get_df no longer grows each call"
    assert df['STOI'].iloc[1] == pytest.approx(df['STOI'].iloc[2])
