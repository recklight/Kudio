# -*- coding: utf-8 -*-
import numpy as np
import pytest

import kudio
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


# --------------------------------------------------------------- audio_info

def test_audio_info_reads_the_header(wav_file):
    info = kudio.audio_info(wav_file)
    assert info.sr == 16000
    assert info.channels == 1
    assert info.frames == 16000
    assert info.duration == pytest.approx(1.0)
    assert info.format == "WAV"
    assert info.subtype == "PCM_16"
    assert info.path == wav_file


def test_audio_info_measures_the_peak(wav_file):
    assert kudio.audio_info(wav_file).peak == pytest.approx(0.5, abs=0.01)


def test_audio_info_can_skip_the_samples(wav_file):
    """peak=False never touches the data, so it stays instant on a big file."""
    info = kudio.audio_info(wav_file, peak=False)
    assert info.peak == 0.0
    assert info.duration == pytest.approx(1.0)


def test_audio_info_agrees_with_file_load(wav_file):
    y, sr = file_load(wav_file, sr=None)
    info = kudio.audio_info(wav_file)
    assert (info.sr, info.frames) == (sr, len(y))
    assert info.peak == pytest.approx(float(np.max(np.abs(y))), abs=1e-4)


def test_audio_info_accepts_an_array(sine):
    info = kudio.audio_info(sine, sr=16000)
    assert info.channels == 1 and info.frames == len(sine)
    assert info.path is None and info.subtype is None
    stereo = np.stack([sine, sine], axis=1)
    assert kudio.audio_info(stereo, sr=16000).channels == 2


def test_audio_info_on_an_array_needs_a_rate(sine):
    with pytest.raises(ValueError, match="needs sr"):
        kudio.audio_info(sine)


def test_audio_info_on_a_missing_file(tmp_path):
    from kudio.exceptions import AudioIOError
    with pytest.raises(AudioIOError):
        kudio.audio_info(tmp_path / "nope.wav")


# ------------------------------------------------------------ convert_folder

def test_convert_folder_writes_every_file(wav_dir, tmp_path):
    result = kudio.convert_folder(wav_dir, tmp_path / "out", sr=8000)
    assert (result.written, result.total) == (3, 3)
    assert not result.failed and bool(result) is True
    for path in result.outputs:
        assert kudio.audio_info(path, peak=False).sr == 8000


def test_convert_folder_mirrors_subdirectories(wav_dir, tmp_path, sine):
    nested = wav_dir / "deep" / "deeper"
    nested.mkdir(parents=True)
    save_wave(nested / "tone_0.wav", sine, 16000)     # same name as a top-level file
    result = kudio.convert_folder(wav_dir, tmp_path / "out")
    written = {p.relative_to(tmp_path / "out").as_posix() for p in result.outputs}
    assert "tone_0.wav" in written
    assert "deep/deeper/tone_0.wav" in written        # not renamed, not clobbered


def test_convert_folder_normalizes_loudness(wav_dir, tmp_path):
    result = kudio.convert_folder(wav_dir, tmp_path / "out", lufs=-23.0)
    for path in result.outputs:
        y, sr = file_load(path, sr=None)
        assert kudio.loudness(y, sr) == pytest.approx(-23.0, abs=0.5)


def test_convert_folder_peak_and_lufs_are_exclusive(wav_dir, tmp_path):
    with pytest.raises(ValueError, match="not both"):
        kudio.convert_folder(wav_dir, tmp_path / "out", peak=0.99, lufs=-23.0)


def test_convert_folder_collects_failures(wav_dir, tmp_path):
    (wav_dir / "broken.wav").write_bytes(b"this is not a wave file")
    result = kudio.convert_folder(wav_dir, tmp_path / "out")
    assert result.written == 3
    assert [p.name for p, _ in result.failed] == ["broken.wav"]
    assert bool(result) is False


def test_convert_folder_can_stop_at_the_first_failure(wav_dir, tmp_path):
    from kudio.exceptions import AudioIOError
    (wav_dir / "aaa_broken.wav").write_bytes(b"nope")
    with pytest.raises(AudioIOError):
        kudio.convert_folder(wav_dir, tmp_path / "out", on_error="raise")


def test_convert_folder_reports_progress(wav_dir, tmp_path):
    seen = []
    kudio.convert_folder(wav_dir, tmp_path / "out",
                         progress=lambda n, total, path: seen.append((n, total)))
    assert seen == [(1, 3), (2, 3), (3, 3)]


def test_convert_folder_on_an_empty_folder(tmp_path):
    from kudio.exceptions import AudioIOError
    (tmp_path / "empty").mkdir()
    with pytest.raises(AudioIOError, match="no audio files"):
        kudio.convert_folder(tmp_path / "empty", tmp_path / "out")


def test_convert_folder_does_not_overwrite_by_default(wav_dir, tmp_path):
    out = tmp_path / "out"
    kudio.convert_folder(wav_dir, out)
    kudio.convert_folder(wav_dir, out)
    assert len(list(out.glob("*.wav"))) == 6      # renamed, not replaced


def test_convert_folder_can_overwrite(wav_dir, tmp_path):
    out = tmp_path / "out"
    kudio.convert_folder(wav_dir, out)
    kudio.convert_folder(wav_dir, out, overwrite=True)
    assert len(list(out.glob("*.wav"))) == 3
