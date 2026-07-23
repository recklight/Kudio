# -*- coding: utf-8 -*-
import numpy as np
import pytest

from kudio import check_input, file_load, load_waves, save_wave
from kudio.core.io import check_file, check_path, copy_waves, LoadAudio


def test_file_load_roundtrip(wav_file):
    y, sr = file_load(wav_file)
    assert sr == 16000
    assert len(y) == 16000
    assert y.dtype == np.float32


def test_file_load_missing(tmp_path):
    with pytest.raises(Exception):
        file_load(tmp_path / "nope.wav")


def test_save_wave_roundtrip(tmp_path, sine):
    out = tmp_path / "out.wav"
    save_wave(out, sine, 16000)
    y, sr = file_load(out)
    assert sr == 16000
    assert np.corrcoef(y, sine)[0, 1] > 0.999


def test_save_wave_resample(tmp_path, sine):
    out = tmp_path / "out8k.wav"
    save_wave(out, sine, 16000, d_sample=8000)
    y, sr = file_load(out)
    assert sr == 8000
    assert len(y) == 8000


def test_check_input_dir(wav_dir):
    right, wrong = check_input(wav_dir)
    assert len(right) == 3 and not wrong


def test_check_input_txt_manifest(wav_dir, tmp_path):
    manifest = tmp_path / "list.txt"
    files = sorted(wav_dir.glob("*.wav"))
    manifest.write_text("\n".join(str(f) for f in files[:2]))
    right, wrong = check_input(manifest)
    assert len(right) == 2 and not wrong


def test_check_input_mixed_list(wav_dir, tmp_path):
    bogus = tmp_path / "not_here.wav"
    right, wrong = check_input([wav_dir, str(bogus)])
    assert len(right) == 3
    assert wrong == [str(bogus)]


def test_check_path_increments(tmp_path):
    p = tmp_path / "exp"
    assert check_path(p) == str(p)
    p.mkdir()
    assert check_path(p, exist_ok=True) == str(p)
    assert check_path(p) == f"{p}2"
    (tmp_path / "exp5").mkdir()
    assert check_path(p) == f"{p}6"


def test_check_file_renames(tmp_path, sine):
    target = tmp_path / "rec.wav"
    save_wave(target, sine, 16000)
    renamed = check_file(target, rename=True)
    assert renamed.endswith("rec2.wav")


def test_load_waves_std_len(wav_dir):
    w2d, sr, min_len = load_waves(wav_dir, std_len=True)
    assert sr == 16000
    assert all(len(w) == min_len for w in w2d)


def test_copy_waves_and_load_audio(wav_dir, tmp_path):
    dst = tmp_path / "copied"
    copy_waves(dst, sorted(wav_dir.glob("*.wav")))
    la = LoadAudio(dst)
    assert len(la) == 3
