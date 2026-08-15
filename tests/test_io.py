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


# -- in-memory resampling -------------------------------------------------------

def test_resample_changes_the_length_proportionally():
    from kudio import resample
    from conftest import make_sine

    y = make_sine(seconds=1.0, sr=8000)
    up = resample(y, 8000, 16000)

    assert abs(len(up) - 2 * len(y)) <= 1


def test_resample_is_a_noop_at_the_same_rate():
    from kudio import resample
    from conftest import make_sine

    y = make_sine(seconds=0.1)
    assert resample(y, 16000, 16000) is y


def test_resample_preserves_the_tone():
    """A 440 Hz sine stays 440 Hz after a rate change."""
    import numpy as np
    from kudio import resample
    from conftest import make_sine

    y = make_sine(freq=440.0, seconds=0.5, sr=16000)
    out = resample(y, 16000, 8000)

    spectrum = np.abs(np.fft.rfft(out))
    peak_hz = np.fft.rfftfreq(len(out), 1 / 8000)[int(np.argmax(spectrum))]
    assert abs(peak_hz - 440.0) < 10.0


def test_resample_matches_load_time_resampling(tmp_path):
    import numpy as np
    from kudio import file_load, resample
    from conftest import make_sine, write_wav

    p = tmp_path / "tone.wav"
    write_wav(p, make_sine(seconds=0.5, sr=16000), sr=16000)

    native, sr = file_load(p)
    on_load, _ = file_load(p, sr=8000)
    in_memory = resample(native, sr, 8000)

    n = min(len(on_load), len(in_memory))
    assert np.corrcoef(on_load[:n], in_memory[:n])[0, 1] > 0.99


def test_resample_rejects_bad_rates():
    import numpy as np
    import pytest
    from kudio import resample

    with pytest.raises(ValueError, match="sample rates must be > 0"):
        resample(np.zeros(10, dtype=np.float32), 0, 16000)


def test_stereo_comes_back_channels_last_on_both_paths(tmp_path):
    """The two branches of file_load must not disagree about the layout.

    ``sr=None`` reads through soundfile, a requested rate goes through librosa,
    and the two libraries have opposite conventions. Picking the branch by an
    unrelated argument must not transpose the audio.
    """
    import numpy as np
    import soundfile as sf
    from kudio import file_load

    sr = 16000
    t = np.arange(sr, dtype=np.float32) / sr
    left = 0.5 * np.sin(2 * np.pi * 220 * t)
    right = 0.5 * np.sin(2 * np.pi * 880 * t)          # a different tone
    path = tmp_path / "stereo.wav"
    sf.write(str(path), np.stack([left, right], axis=1), sr)

    native, native_sr = file_load(path, mono=False)
    resampled, out_sr = file_load(path, sr=8000, mono=False)

    assert native.shape == (sr, 2)
    assert resampled.shape == (8000, 2)
    assert native_sr == sr and out_sr == 8000

    # and the channels really are in the same order, not swapped by the
    # transpose: channel 0 is the low tone on both paths
    for data, rate in ((native, native_sr), (resampled, out_sr)):
        spectrum = np.abs(np.fft.rfft(data, axis=0))
        peaks = np.fft.rfftfreq(data.shape[0], 1 / rate)[np.argmax(spectrum, axis=0)]
        assert peaks[0] < peaks[1]
        assert abs(peaks[0] - 220) < 5


def test_mono_downmix_is_unaffected_by_the_layout_fix(tmp_path):
    import numpy as np
    import soundfile as sf
    from kudio import file_load

    sr = 16000
    data = np.stack([np.ones(sr, dtype=np.float32) * 0.5,
                     np.ones(sr, dtype=np.float32) * 0.1], axis=1)
    path = tmp_path / "stereo.wav"
    sf.write(str(path), data, sr)

    for kwargs in ({}, {"sr": 8000}):
        y, _ = file_load(path, mono=True, **kwargs)
        assert y.ndim == 1
        assert np.allclose(y.mean(), 0.3, atol=0.01)
