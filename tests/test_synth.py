# -*- coding: utf-8 -*-
import numpy as np
import pytest

from kudio import Synthesizer, file_load


def test_syn_increment_mode(clean_noise_dirs, tmp_path):
    clean, noise = clean_noise_dirs
    out = tmp_path / "mixed"
    syx = Synthesizer(clean, noise, out_path=str(out), snr_ratio=(-5, 0, 5))
    result = syx.syn(mode='inc')
    # 2 clean x 1 noise x 3 SNR
    assert len(result) == 6
    files = list(out.rglob("*.wav"))
    assert len(files) == 6
    for f in files:
        y, sr = file_load(f)
        assert len(y) > 0


def test_syn_regular_mode(clean_noise_dirs, tmp_path):
    clean, noise = clean_noise_dirs
    out = tmp_path / "mixed_reg"
    syx = Synthesizer(clean, noise, out_path=str(out), snr_ratio=[0])
    result = syx.syn(mode='reg')
    assert len(result) == 2  # equals number of clean files


def test_syn_snr_is_respected(clean_noise_dirs, tmp_path):
    """Mixed at high SNR should stay closer to clean than mixed at low SNR."""
    clean, noise = clean_noise_dirs
    out_hi = tmp_path / "hi"
    out_lo = tmp_path / "lo"
    Synthesizer(clean, noise, out_path=str(out_hi), snr_ratio=[20]).syn(mode='reg')
    Synthesizer(clean, noise, out_path=str(out_lo), snr_ratio=[-5]).syn(mode='reg')

    def err_vs_clean(out_dir):
        errs = []
        for f in sorted(out_dir.rglob("*.wav")):
            stem = f.stem.split('_white')[0]
            y_mix, _ = file_load(f)
            y_cln, _ = file_load(clean / f"{stem}.wav")
            n = min(len(y_mix), len(y_cln))
            errs.append(np.mean((y_mix[:n] - y_cln[:n]) ** 2))
        return np.mean(errs)

    assert err_vs_clean(out_hi) < err_vs_clean(out_lo)


def test_syn_overwrite_false_aborts(clean_noise_dirs, tmp_path):
    clean, noise = clean_noise_dirs
    out = tmp_path / "mixed"
    syx = Synthesizer(clean, noise, out_path=str(out), snr_ratio=[0])
    assert syx.syn(mode='reg') is not None
    assert syx.syn(mode='reg', overwrite=False) is None


def test_bad_inputs(clean_noise_dirs, tmp_path):
    clean, noise = clean_noise_dirs
    with pytest.raises(TypeError):
        Synthesizer(clean, noise, snr_ratio="5")
    with pytest.raises(NotADirectoryError):
        Synthesizer(tmp_path / "nope", noise)
    syx = Synthesizer(clean, noise, out_path=str(tmp_path / "x"), snr_ratio=[0])
    with pytest.raises(ValueError):
        syx.syn(mode='bogus')


# -- sample-rate handling ------------------------------------------------------

def _corpus(tmp_path, clean_sr: int, noise_sr: int):
    """Clean and noise folders written at explicitly different rates."""
    from conftest import make_sine, write_wav

    clean = tmp_path / "c"
    noise = tmp_path / "n"
    clean.mkdir()
    noise.mkdir()
    write_wav(clean / "a.wav", make_sine(220, 1.0, clean_sr), clean_sr)
    write_wav(noise / "w.wav", make_sine(1000, 1.0, noise_sr), noise_sr)
    return clean, noise


def _rates(out_dir):
    import soundfile as sf
    return {sf.info(str(f)).samplerate for f in out_dir.rglob("*.wav")}


def test_syn_keeps_the_source_sample_rate(tmp_path):
    """Default output rate follows the clean file, not a hard-coded 16 kHz."""
    clean, noise = _corpus(tmp_path, clean_sr=8000, noise_sr=8000)
    out = tmp_path / "mixed"
    Synthesizer(clean, noise, out_path=str(out), snr_ratio=[0]).syn(mode='reg')

    assert _rates(out) == {8000}


def test_syn_target_sr_resamples(tmp_path):
    clean, noise = _corpus(tmp_path, clean_sr=8000, noise_sr=8000)
    out = tmp_path / "mixed"
    Synthesizer(clean, noise, out_path=str(out), snr_ratio=[0]).syn(
        mode='reg', target_sr=16000)

    assert _rates(out) == {16000}


def test_syn_resamples_noise_to_match_clean(tmp_path):
    """A noise file at a different rate must not be mixed in as-is."""
    clean, noise = _corpus(tmp_path, clean_sr=8000, noise_sr=44100)
    out = tmp_path / "mixed"
    result = Synthesizer(clean, noise, out_path=str(out), snr_ratio=[0]).syn(mode='reg')

    assert result
    assert _rates(out) == {8000}
    y, sr = file_load(next(out.rglob("*.wav")))
    assert sr == 8000
    assert np.isfinite(y).all()


def test_desired_sample_still_works_but_warns(tmp_path):
    clean, noise = _corpus(tmp_path, clean_sr=8000, noise_sr=8000)
    out = tmp_path / "mixed"
    syx = Synthesizer(clean, noise, out_path=str(out), snr_ratio=[0])

    with pytest.deprecated_call():
        syx.syn(mode='reg', desired_sample=16000)
    assert _rates(out) == {16000}
