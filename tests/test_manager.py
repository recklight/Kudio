# -*- coding: utf-8 -*-
import numpy as np

from kudio import AudioDataManager, file_load


def test_add_local_file_and_lookup(wav_file):
    mgr = AudioDataManager()
    mgr.add_local_file(wav_file)
    assert len(mgr) == 1
    assert 'sine' in mgr.get_names()
    assert mgr.get_sample('sine') == 16000
    assert len(mgr.get_wave('sine')[0]) == 16000
    # int and call access
    assert mgr[0] is mgr('sine')


def test_missing_item_returns_none():
    mgr = AudioDataManager()
    assert mgr.get_item('nope') is None
    assert mgr.get_wave('nope') is None


def test_add_new_file_and_save(tmp_path, sine):
    mgr = AudioDataManager()
    mgr.set_params(sr=16000, ch=1)
    mgr.add_new_file(name='tone', wave=[sine])
    assert len(mgr) == 1

    assert mgr.save_all(dst_path=tmp_path) is True
    y, sr = file_load(tmp_path / "tone.wav")
    assert sr == 16000
    assert np.corrcoef(y, sine)[0, 1] > 0.999


def test_duplicate_name_ignored(wav_file):
    mgr = AudioDataManager()
    mgr.add_local_file(wav_file)
    mgr.add_local_file(wav_file)  # ignored, no exception
    assert len(mgr) == 1


def test_del_element(wav_file):
    mgr = AudioDataManager()
    mgr.add_local_file(wav_file)
    mgr.del_element('sine')
    assert len(mgr) == 0
    mgr.del_element('sine')  # no-op
