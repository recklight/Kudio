# -*- coding: utf-8 -*-
import logging

import numpy as np

import kudio
from kudio import AudioDataManager, file_load

from conftest import make_sine


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


# -- one clip, one file ---------------------------------------------------------

def test_clips_with_dotted_names_keep_their_own_files(tmp_path):
    """'take' and 'take.v2' are two recordings, and they need two files.

    ``with_suffix`` read the dot as the start of an extension and sent both to
    'take.wav': one clip's audio was gone, ``save_all`` still said True, and
    nothing in the log mentioned it.
    """
    mgr = AudioDataManager()
    mgr.set_params(sr=16000, ch=1)
    take, take_v2 = make_sine(freq=220.0), make_sine(freq=880.0)
    mgr.add_new_file(name='take', wave=[take])
    mgr.add_new_file(name='take.v2', wave=[take_v2])

    assert mgr.save_all(dst_path=tmp_path) is True

    written = sorted(p.name for p in tmp_path.glob("*.wav"))
    assert written == ["take.v2.wav", "take.wav"]
    y, _ = file_load(tmp_path / "take.wav")
    assert np.corrcoef(y, take)[0, 1] > 0.999
    y_v2, _ = file_load(tmp_path / "take.v2.wav")
    assert np.corrcoef(y_v2, take_v2)[0, 1] > 0.999


def test_both_channels_of_a_dotted_name_survive(tmp_path):
    """'take.v2_ch0' and 'take.v2_ch1' both came out as 'take.wav', so channel
    one landed on channel zero under a name identifying neither."""
    mgr = AudioDataManager()
    mgr.set_params(sr=16000, ch=2)
    left, right = make_sine(freq=220.0), make_sine(freq=880.0)
    interleaved = np.empty(len(left) + len(right), dtype=np.float32)
    interleaved[0::2], interleaved[1::2] = left, right
    mgr.add_new_file(name='take.v2', wave=[interleaved])

    assert mgr.save_(name='take.v2', dst_path=tmp_path) is True

    written = sorted(p.name for p in tmp_path.glob("*.wav"))
    assert written == ["take.v2_ch0.wav", "take.v2_ch1.wav"]
    ch0, _ = file_load(tmp_path / "take.v2_ch0.wav")
    assert np.corrcoef(ch0, left)[0, 1] > 0.999
    ch1, _ = file_load(tmp_path / "take.v2_ch1.wav")
    assert np.corrcoef(ch1, right)[0, 1] > 0.999


def test_two_clips_that_want_one_file_are_refused_before_anything_is_written(
        tmp_path, caplog):
    """A clip called 'take' and one called 'take.wav' ask for the same file.
    Which of them is renamed is not a decision to make quietly."""
    mgr = AudioDataManager()
    mgr.set_params(sr=16000, ch=1)
    mgr.add_new_file(name='take', wave=[make_sine(freq=220.0)])
    mgr.add_new_file(name='take.wav', wave=[make_sine(freq=880.0)])

    with caplog.at_level(logging.ERROR, logger="kudio.core.manager"):
        saved = mgr.save_all(dst_path=tmp_path)

    assert saved is False
    assert list(tmp_path.glob("*.wav")) == [], "not one of them was written"
    assert "take.wav" in caplog.text and "rename" in caplog.text


# -- a clip's own channel count ------------------------------------------------
def test_a_mono_clip_goes_back_to_its_own_file_when_the_manager_is_stereo(tmp_path):
    path = tmp_path / "take.wav"
    kudio.save_wave(path, 0.1 * np.ones(1600), 16000)
    manager = AudioDataManager()
    manager.set_params(16000, 2)
    manager.add_local_file(path)
    assert manager.save_all()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["take.wav"]


# -- volume on integer audio ------------------------------------------------------
def test_play_volume_clips_integer_audio_rather_than_wrapping(monkeypatch):
    played = {}
    monkeypatch.setattr("kudio.core.manager.play_audio",
                        lambda wav, rate: played.update(wav=wav, rate=rate))
    m = AudioDataManager()
    m.add_new_file("take", wave=[np.array([20000, -20000, 1000], np.int16)],
                   sr=16000, ch=1)
    m.play_audio("take", volume=2.0)
    assert played["wav"].dtype == np.int16
    assert played["wav"].tolist() == [32767, -32768, 2000]
