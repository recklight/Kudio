# -*- coding: utf-8 -*-
import contextlib
import logging
import os
import struct
import subprocess
import sys
import types
import warnings
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import kudio
from kudio import check_input, file_load, load_waves, save_wave
from kudio.core.io import check_file, check_path, copy_waves, LoadAudio
from kudio.exceptions import AudioIOError


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


@pytest.mark.parametrize("name", ["out.aif", "out.AIF", "out.aiff"])
def test_save_wave_writes_aiff_under_either_extension(tmp_path, sine, name):
    # soundfile reads the container from ".aiff" and not from ".aif"
    out = tmp_path / name
    save_wave(out, sine, 16000)
    assert sf.info(str(out)).format == "AIFF"
    y, sr = file_load(out)
    assert sr == 16000 and np.corrcoef(y, sine)[0, 1] > 0.999


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

    with pytest.raises(ValueError, match="orig_sr must be a positive rate"):
        resample(np.zeros(10, dtype=np.float32), 0, 16000)


def test_stereo_comes_back_channels_last_on_both_paths(tmp_path):
    """The two branches of file_load must not disagree about the layout.

    ``sr=None`` reads the file, a requested rate resamples what was read, and a
    resampler can just as well be one for (channels, frames). Picking the branch
    by an unrelated argument must not transpose the audio.
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


def test_mono_downmix_does_not_depend_on_the_layout(tmp_path):
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
    """peak=False never touches the data, so it stays instant on a big file.
    The peak it did not measure is None: a 0.0 there read as digital silence."""
    info = kudio.audio_info(wav_file, peak=False)
    assert info.peak is None
    assert "not measured" in str(info)
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


# -- what a fixed-point file cannot hold ---------------------------------------

def test_save_wave_clips_past_full_scale_instead_of_wrapping(tmp_path):
    out = tmp_path / "hot.wav"
    save_wave(out, np.array([1.2, -1.3, 1.0, -1.0, 0.5], np.float32), 16000)
    import soundfile as sf
    assert sf.read(out, dtype="int16")[0].tolist() == [32767, -32768, 32767,
                                                       -32768, 16384]


def test_save_wave_rounds_to_the_nearest_step(tmp_path):
    """libsndfile floors, which leaves every sample half a step low."""
    import soundfile as sf
    x = np.random.default_rng(0).uniform(-0.99, 0.99, 20000).astype(np.float32)
    out = tmp_path / "r.wav"
    save_wave(out, x, 16000)
    err = (sf.read(out, dtype="float64")[0] - x) * 32768
    assert np.abs(err).max() <= 0.5 + 1e-6
    assert abs(err.mean()) < 0.02


def test_save_wave_writes_int16_unchanged(tmp_path):
    import soundfile as sf
    v = np.random.default_rng(1).integers(-32768, 32768, 5000).astype(np.int16)
    out = tmp_path / "i.wav"
    save_wave(out, v, 16000)
    assert np.array_equal(sf.read(out, dtype="int16")[0], v)


def test_save_wave_float_keeps_what_is_past_full_scale(tmp_path):
    out = tmp_path / "f.wav"
    save_wave(out, np.array([1.5, -2.0], np.float32), 16000, subtype="FLOAT")
    y, _ = file_load(out)
    assert y.tolist() == [1.5, -2.0]


# -- convert_folder on a list --------------------------------------------------

def _two_talkers(tmp_path, sine):
    for folder in ("dr1/fcjf0", "dr2/fdml0"):
        (tmp_path / "timit" / folder).mkdir(parents=True)
        save_wave(tmp_path / "timit" / folder / "sx133.wav", sine, 16000)
    return sorted((tmp_path / "timit").rglob("*.wav"))


def test_convert_folder_mirrors_a_list_below_what_it_shares(tmp_path, sine):
    """A list landing flat would make two talkers' sx133 collide and the second
    come out as sx1332 -- which reads like another sentence."""
    files = _two_talkers(tmp_path, sine)
    result = kudio.convert_folder(files, tmp_path / "out")
    assert sorted(p.relative_to(tmp_path / "out").as_posix()
                  for p in result.outputs) == ["dr1/fcjf0/sx133.wav",
                                               "dr2/fdml0/sx133.wav"]


def test_convert_folder_takes_a_root(tmp_path, sine):
    files = _two_talkers(tmp_path, sine)
    result = kudio.convert_folder(files, tmp_path / "out",
                                  root=tmp_path / "timit" / "dr1" / "..")
    assert sorted(p.relative_to(tmp_path / "out").as_posix()
                  for p in result.outputs) == ["dr1/fcjf0/sx133.wav",
                                               "dr2/fdml0/sx133.wav"]
    with pytest.raises(ValueError, match="not inside root"):
        kudio.convert_folder(files, tmp_path / "out2",
                             root=tmp_path / "timit" / "dr1")


def test_convert_folder_of_files_in_one_folder_stays_flat(wav_dir, tmp_path):
    result = kudio.convert_folder(sorted(wav_dir.glob("*.wav")), tmp_path / "o")
    assert all(p.parent == tmp_path / "o" for p in result.outputs)


# -- names that are free, and names that only look free -------------------------

def test_check_file_never_hands_back_an_occupied_name(tmp_path, sine):
    """A dotted name is a name, not a name plus an extension.

    Numbered through ``with_suffix``, 'take.v2.wav' would become 'take.wav' --
    the file belonging to the take next to it -- and the caller would write over
    it in the one mode that exists to leave existing files alone.
    """
    save_wave(tmp_path / "take.wav", sine, 16000)
    save_wave(tmp_path / "take.v2.wav", sine, 16000)

    chosen = Path(check_file(tmp_path / "take.v2.wav", rename=True))

    assert not chosen.exists(), "the name handed back has to be free"
    assert chosen != tmp_path / "take.wav", "that file is another recording"
    assert chosen.name.startswith("take.v2"), "and it has to be this one's name"


def test_check_file_logs_the_name_it_actually_chose(tmp_path, sine, caplog):
    """The INFO line is the only record of the rename, so it has to be true."""
    save_wave(tmp_path / "take.wav", sine, 16000)
    save_wave(tmp_path / "take.v2.wav", sine, 16000)

    with caplog.at_level(logging.INFO, logger="kudio.core.io"):
        chosen = check_file(tmp_path / "take.v2.wav", rename=True)

    renamed_to = [m.split("Renamed to: ")[1] for m in caplog.messages
                  if m.startswith("Renamed to: ")]
    assert renamed_to == [chosen]
    assert not Path(renamed_to[0]).exists()


def test_check_file_keeps_a_dotted_name_when_it_adds_the_suffix(tmp_path):
    """'take.v2' with no suffix is the clip's whole name; '.flac' is not."""
    assert check_file(tmp_path / "take.v2") == str(tmp_path / "take.v2.wav")
    assert check_file(tmp_path / "take.flac") == str(tmp_path / "take.wav")


def test_convert_folder_without_overwrite_leaves_the_existing_files_alone(tmp_path):
    """The whole point of ``overwrite=False``: nothing already there is lost.

    Two takes whose names differ only by a dotted part, converted into a
    folder that already holds both, must not end with the second one's
    audio under the first one's name.
    """
    from conftest import make_sine

    src = tmp_path / "raw"
    src.mkdir()
    out = tmp_path / "16k"
    out.mkdir()
    take, take_v2 = make_sine(freq=220.0), make_sine(freq=880.0)
    for folder in (src, out):                   # out/ is left over from a run
        save_wave(folder / "take.wav", take, 16000)
        save_wave(folder / "take.v2.wav", take_v2, 16000)

    result = kudio.convert_folder(src, out, overwrite=False)

    assert result.written == 2 and not result.failed
    kept, _ = file_load(out / "take.wav")
    assert np.corrcoef(kept, take)[0, 1] > 0.999, "take.wav is still take"
    kept_v2, _ = file_load(out / "take.v2.wav")
    assert np.corrcoef(kept_v2, take_v2)[0, 1] > 0.999
    assert len(list(out.glob("*.wav"))) == 4, "both conversions were written too"


# -- samples that are not numbers -----------------------------------------------

def _write_non_finite(path, bad=np.nan, count=10, sr=16000):
    """A one-second 220 Hz tone at 0.3 with *count* samples replaced by *bad*.

    Written straight through soundfile: ``save_wave`` refuses it, which is the
    point, and such files are on disks all the same.
    """
    y = 0.3 * np.sin(2 * np.pi * 220 * np.arange(sr) / sr)
    y[1000:1000 + count] = bad
    sf.write(str(path), y.astype(np.float32), sr, subtype='FLOAT')
    return y


def test_save_wave_refuses_to_write_non_finite_samples(tmp_path, sine):
    """FLOAT keeps a NaN and an integer subtype turns it into a zero, which
    reads back as silence. Neither is a file anybody can use."""
    broken = sine.copy()
    broken[1000:1010] = np.nan
    broken[2000] = np.inf

    for subtype in ('FLOAT', 'PCM_16'):
        out = tmp_path / f"broken_{subtype}.wav"
        with pytest.raises(AudioIOError, match="not finite"):
            save_wave(out, broken, 16000, subtype=subtype)
        assert not out.exists(), "and nothing is left behind"


def test_save_wave_names_the_file_and_the_count(tmp_path, sine):
    broken = sine.copy()
    broken[:3] = np.nan

    with pytest.raises(AudioIOError) as excinfo:
        save_wave(tmp_path / "mix.wav", broken, 16000, subtype='FLOAT')

    message = str(excinfo.value)
    assert "mix.wav" in message
    assert "3 of" in message


def test_file_load_says_so_when_a_file_holds_nan(tmp_path):
    """Loading one is allowed -- you may be reading it to repair it -- but it
    is the last place anything can be said about it, so it is a warning that
    shows without any logging set up."""
    path = tmp_path / "take.wav"
    _write_non_finite(path)

    with pytest.warns(UserWarning, match="not finite") as caught:
        y, _ = file_load(path)

    assert np.count_nonzero(~np.isfinite(y)) == 10, "the samples come back as they are"
    assert "take.wav" in str(caught[0].message)


def test_audio_info_does_not_report_a_nan_file_as_silent(tmp_path):
    """``max(0.0, nan)`` keeps the 0.0, so the peak of a 0.3 tone with ten NaN
    samples in it would read as digital silence."""
    path = tmp_path / "take.wav"
    _write_non_finite(path)

    with pytest.warns(UserWarning, match="not finite"):
        info = kudio.audio_info(path)

    assert not np.isfinite(info.peak), "there is no largest sample here"
    assert info.frames == 16000, "the header is still readable"


# -- channels last, integer types, every PCM width -----------------------------

def _stereo(sr=16000, seconds=1.0):
    t = np.arange(int(sr * seconds)) / sr
    return np.stack([0.5 * np.sin(2 * np.pi * 220 * t),
                     0.25 * np.sin(2 * np.pi * 330 * t)], axis=1).astype(np.float32)


def test_resample_runs_along_the_frames_of_a_stereo_clip():
    """Along the last axis, where a resampler for (channels, frames) works, a
    (16000, 2) clip comes back (16000, 1)."""
    out = kudio.resample(_stereo(), 16000, 8000)
    assert out.shape == (8000, 2)
    assert np.max(np.abs(out[:, 1])) == pytest.approx(0.25, abs=0.01), "channels kept apart"


def test_resample_rounds_integer_audio_instead_of_truncating():
    y = np.round(10000 * np.sin(2 * np.pi * 50 * np.arange(800) / 8000)).astype(np.int16)
    out = kudio.resample(y, 8000, 16000)
    assert out.dtype == np.int16
    exact = kudio.resample(y.astype(np.float32), 8000, 16000)
    assert np.array_equal(out, np.round(exact).astype(np.int16))
    assert not np.array_equal(out, exact.astype(np.int16)), "nothing here to round"


#: what resample's res_type can name: soxr's five settings and scipy's two
#: resamplers, one of them under two names
RESAMPLERS = ('soxr_vhq', 'soxr_hq', 'soxr_mq', 'soxr_lq', 'soxr_qq', 'fft', 'scipy',
              'polyphase')


@pytest.mark.parametrize("res_type, floor_db", [
    ('soxr_vhq', -140), ('soxr_hq', -130), ('soxr_mq', -108), ('soxr_lq', -106),
    ('soxr_qq', -64), ('fft', -128), ('scipy', -128), ('polyphase', -57)])
@pytest.mark.parametrize("orig_sr, target_sr", [(16000, 44100), (48000, 16000), (44100, 16000)])
def test_a_resampled_sine_is_the_sine_at_the_other_rate(res_type, floor_db, orig_sr, target_sr):
    """Sample for sample, with the phase it had: one sample late, a 1 kHz sine
    differs from itself by -17 dB at 44.1 kHz and by -8 dB at 16 kHz. Away from
    the ends, which a filter runs past."""
    tone = {sr: (0.5 * np.sin(2 * np.pi * 1000.0 * np.arange(sr) / sr + 0.3)).astype(np.float32)
            for sr in (orig_sr, target_sr)}
    out = kudio.resample(tone[orig_sr], orig_sr, target_sr, res_type=res_type)
    inside = slice(target_sr // 10, -target_sr // 10)
    error = (out[:target_sr].astype(np.float64) - tone[target_sr])[inside]
    assert 10 * np.log10(np.mean(error ** 2) / 0.125 + 1e-30) < floor_db


@pytest.mark.parametrize("res_type", RESAMPLERS)
def test_every_resampler_gives_the_length_the_rates_give(res_type):
    for orig_sr, target_sr in ((16000, 8000), (8000, 16000), (44100, 16000), (48000, 8000)):
        for n in (1, 2, 5, 6, 7, 441, 1000, 1001):
            out = kudio.resample(np.ones(n, np.float32), orig_sr, target_sr, res_type=res_type)
            assert out.shape == (-(-n * target_sr // orig_sr),), (orig_sr, target_sr, n)
            assert out.dtype == np.float32 and np.isfinite(out).all()


#: from a rate of the 11025 Hz family to one of the 48 kHz family, n * (b / a)
#: in floating point comes out a hair over a whole number: 147 samples at 44.1
#: kHz are 160.00000000000003 at 48 kHz, where they are 160
ONE_TOO_MANY = [(a, b) for a in (11025, 22050, 44100, 88200, 176400)
                for b in (12000, 24000, 48000, 96000, 192000)]
NEVER = [(8000, 16000), (16000, 8000), (48000, 44100), (44100, 16000), (16000, 44100),
         (44100, 32000), (48000, 22050), (11025, 8000), (96000, 88200), (192000, 11025)]


@pytest.mark.parametrize("orig_sr, target_sr", ONE_TOO_MANY + NEVER)
@pytest.mark.parametrize("res_type", ['soxr_hq', 'soxr_qq', 'fft', 'polyphase'])
def test_a_second_is_a_second_at_the_other_rate(res_type, orig_sr, target_sr):
    """A second of 44.1 kHz is 48000 samples at 48 kHz, not 48001 with the last
    one zero, or with the Fourier method the whole second drawn out by one."""
    y = (0.5 * np.sin(2 * np.pi * 1000.0 * np.arange(orig_sr) / orig_sr)).astype(np.float32)
    out = kudio.resample(y, orig_sr, target_sr, res_type=res_type)
    assert out.shape == (target_sr,)
    want = 0.5 * np.sin(2 * np.pi * 1000.0 * np.arange(target_sr) / target_sr)
    inside = slice(target_sr // 10, -target_sr // 10)
    error_db = 10 * np.log10(np.mean((out - want)[inside] ** 2) / 0.125 + 1e-30)
    assert error_db < {'soxr_hq': -125, 'soxr_qq': -42, 'fft': -120, 'polyphase': -55}[res_type]


@pytest.mark.parametrize("orig_sr, target_sr", ONE_TOO_MANY + NEVER)
def test_a_length_is_the_one_the_rates_give_for_any_number_of_samples(orig_sr, target_sr):
    for n in list(range(2, 40)) + [146, 147, 148, 294, 441, 1470, 4410, 4411]:
        y = np.ones((n, 2), np.float32)
        wanted = (n * target_sr + orig_sr - 1) // orig_sr
        assert kudio.resample(y, orig_sr, target_sr).shape == (wanted, 2), n
        assert len(kudio.resample(y, orig_sr, target_sr, res_type='polyphase')) == wanted, n


#: what files come at, and the rates of 48 kHz and 44.1 kHz doubled and halved
RATES = (8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000, 64000, 88200, 96000,
         176400, 192000, 352800, 384000)


@pytest.mark.parametrize("orig_sr", RATES + (22254.5,))
def test_the_length_a_clip_comes_to_is_known_before(orig_sr):
    """What a run is checked by before anything is written, from a header alone,
    is the length that comes out: from every rate to every other, in whole numbers
    however they are given and in others."""
    from kudio.core.io import _resampled_length
    for target_sr in RATES + (22254.5,):
        for n in (1, 2, 3, 100, 146, 147, 148, 160, 441, 999, 1000, 1001, 4410):
            got = _resampled_length(n, orig_sr, target_sr)
            assert isinstance(got, int)
            for kind in (float, np.int64, np.float64):
                if float(orig_sr).is_integer() and float(target_sr).is_integer():
                    assert _resampled_length(n, kind(orig_sr), kind(target_sr)) == got
            for res_type in ('soxr_hq', 'soxr_qq'):
                out = kudio.resample(np.ones(n, np.float32), orig_sr, target_sr,
                                     res_type=res_type)
                assert len(out) == got, (orig_sr, target_sr, n, res_type)


@pytest.mark.parametrize("res_type", ['kaiser_best', 'kaiser_fast', 'sinc_best', 'linear',
                                      'soxr', 'SOXR_HQ', '', None, 3, ['soxr_hq']], ids=repr)
@pytest.mark.parametrize("target_sr", [8000, 16000])
def test_a_resampler_there_is_not_is_refused_by_name(sine, res_type, target_sr):
    """At the rate the clip is at too, where nothing would have been resampled
    and the name never looked at."""
    with pytest.raises(ValueError) as caught:
        kudio.resample(sine, 16000, target_sr, res_type=res_type)
    assert str(caught.value) == (f"resample(): res_type has to be one of "
                                 f"{', '.join(RESAMPLERS)}, got {res_type!r}")


def test_polyphase_needs_whole_rates(sine):
    with pytest.raises(ValueError) as caught:
        kudio.resample(sine, 22254.5, 16000, res_type='polyphase')
    assert str(caught.value) == ("resample(): res_type='polyphase' takes whole rates, got "
                                 "22254.5 and 16000")
    assert len(kudio.resample(sine, 16000.0, 8000.0, res_type='polyphase')) == 8000


@pytest.mark.parametrize("res_type", ['soxr_hq', 'fft', 'polyphase'])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, 1e39])
def test_samples_that_are_not_finite_are_not_resampled(res_type, bad):
    """One of them becomes a whole filter length of them, or with the Fourier
    method every sample there is. 1e39 is finite in float64 and inf in the
    float32 the work is done in."""
    y = np.full((1000, 2), 0.25)
    y[10:13, 1] = bad
    with warnings.catch_warnings():
        warnings.simplefilter("error")                   # the error says it all
        with pytest.raises(ValueError) as caught:
            kudio.resample(y, 16000, 8000, res_type=res_type)
    assert str(caught.value) == ("resample(): 3 of 2000 sample(s) are not finite in float32, "
                                 "which it works in (NaN, inf or past 3.4e38); repair the clip "
                                 "first")


@pytest.mark.parametrize("y, shape", [(np.zeros((4, 4, 4)), "(4, 4, 4)"), (0.5, "()"),
                                      (np.float32(1), "()")], ids=repr)
def test_what_is_neither_a_clip_nor_channels_is_not_resampled(y, shape):
    with pytest.raises(ValueError) as caught:
        kudio.resample(y, 16000, 8000)
    assert str(caught.value) == (f"resample(): expected (frames,) or (frames, channels), got "
                                 f"shape {shape}")


def test_resampling_leaves_its_input_alone(sine):
    for y in (sine, np.stack([sine, sine], axis=1), np.asfortranarray(np.stack([sine, sine], 1)),
              (sine * 20000).astype(np.int16)):
        before = y.copy()
        for res_type in RESAMPLERS:
            out = kudio.resample(y, 16000, 8000, res_type=res_type)
            assert np.array_equal(y, before) and not np.shares_memory(out, y)


def test_save_wave_resamples_a_stereo_clip_along_its_frames(tmp_path):
    path = tmp_path / "st.wav"
    save_wave(path, _stereo(), 16000, d_sample=8000)
    info = sf.info(str(path))
    assert (info.channels, info.frames, info.samplerate) == (2, 8000, 8000)


@pytest.mark.parametrize("dtype, value, expected", [
    (np.int16, 16384, 0.5),
    (np.int32, 2 ** 30, 0.5),
    (np.uint8, 192, 0.5),
    (np.uint8, 128, 0.0),
])
def test_save_wave_scales_each_integer_type_by_its_own_full_scale(tmp_path, dtype, value, expected):
    """Divided by 32768 whatever its type, half-scale int32 would come out
    clipped at full scale and uint8 200 as 0.006."""
    path = tmp_path / "int.wav"
    save_wave(path, np.full(100, value, dtype=dtype), 16000, subtype='FLOAT')
    y, _ = sf.read(str(path))
    assert y[0] == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("subtype, bits", [("PCM_24", 24), ("PCM_U8", 8), ("PCM_32", 32)])
def test_every_pcm_width_is_rounded_not_floored(tmp_path, subtype, bits):
    """Left to libsndfile's flooring, PCM_24 and PCM_U8 would average a third
    and a half of a step low."""
    # float32, which is what save_wave works in below DOUBLE
    y = np.random.default_rng(0).uniform(-0.9, 0.9, 20000).astype(np.float32)
    path = tmp_path / f"{subtype}.wav"
    save_wave(path, y, 16000, subtype=subtype)
    back, _ = sf.read(str(path), dtype='float64')
    step = 2.0 ** -(bits - 1)
    error = back - y.astype(np.float64)
    assert abs(np.mean(error)) < 0.05 * step
    assert np.max(np.abs(error)) <= 0.5 * step + 1e-12


def test_double_keeps_float64(tmp_path):
    path = tmp_path / "d.wav"
    save_wave(path, np.full(10, 0.1, dtype=np.float64), 16000, subtype='DOUBLE')
    y, _ = sf.read(str(path), dtype='float64')
    assert y[0] == 0.1


def test_a_refused_write_leaves_no_folder_behind(tmp_path):
    target = tmp_path / "new" / "sub" / "x.wav"
    with pytest.raises(AudioIOError):
        save_wave(target, np.array([0.0, np.nan]), 16000)
    assert not (tmp_path / "new").exists()


def test_a_bad_rate_is_a_bad_call_not_a_bad_file(wav_file):
    with pytest.raises(ValueError, match="positive whole rate"):
        file_load(wav_file, sr=0)


def test_check_input_ignores_the_case_of_the_suffix(tmp_path, sine):
    folder = tmp_path / "rec"
    folder.mkdir()
    sf.write(str(folder / "ZOOM0001.WAV"), sine, 16000)
    sf.write(str(folder / "b.wav"), sine, 16000)
    found, _ = check_input(folder)
    assert sorted(p.name for p in found) == ["ZOOM0001.WAV", "b.wav"]
    found, rejected = check_input(str(folder / "ZOOM0001.WAV"))
    assert len(found) == 1 and not rejected
    # and what check_input finds, load_waves loads
    assert len(load_waves(folder)) == 2


def test_a_list_is_read_as_utf8(tmp_path, sine):
    folder = tmp_path / "錄音"
    folder.mkdir()
    sf.write(str(folder / "一.wav"), sine, 16000)
    listing = tmp_path / "list.txt"
    listing.write_bytes(("﻿" + str(folder / "一.wav") + "\n").encode("utf-8"))
    found, rejected = check_input(listing)
    assert [p.name for p in found] == ["一.wav"] and not rejected


def test_check_file_takes_a_trailing_separator_as_a_folder(tmp_path):
    out = check_file(str(tmp_path / "newdir") + "/")
    assert Path(out) == tmp_path / "newdir" / "unknown.wav"


def test_convert_folder_counts_what_it_could_not_use(tmp_path, sine):
    good = tmp_path / "a.wav"
    sf.write(str(good), sine, 16000)
    result = kudio.convert_folder([good, tmp_path / "typo.wav"], tmp_path / "out")
    assert (result.written, result.total, len(result.failed)) == (1, 2, 1)
    assert "not an existing .wav file" in result.failed[0][1]


def test_convert_folder_does_not_feed_on_its_own_output(tmp_path, sine):
    src = tmp_path / "src"
    src.mkdir()
    sf.write(str(src / "a.wav"), sine, 16000)
    for _ in range(2):
        result = kudio.convert_folder(src, src / "out", overwrite=True)
        assert result.total == 1
    assert not (src / "out" / "out").exists()


def test_an_empty_file_is_not_blamed_on_trimming(tmp_path):
    empty = tmp_path / "empty.wav"
    sf.write(str(empty), np.zeros(0), 16000)
    result = kudio.convert_folder(empty, tmp_path / "out")
    assert "no samples" in result.failed[0][1]


def test_copy_waves_keeps_two_files_of_one_name(tmp_path, sine):
    for talker in ("a", "b"):
        (tmp_path / talker).mkdir()
        sf.write(str(tmp_path / talker / "sx133.wav"), sine, 16000)
    copied = copy_waves(tmp_path / "flat", [tmp_path / "a", tmp_path / "b"])
    assert sorted(p.name for p in copied) == ["sx133.wav", "sx1332.wav"]


def test_load_waves_says_when_there_was_nothing_to_load(tmp_path):
    with pytest.raises(AudioIOError, match="no audio files"):
        load_waves(tmp_path, std_len=True)


def test_audio_info_treats_an_array_like_a_file():
    y = 0.3 * np.ones(100)
    assert kudio.audio_info(y, sr=16000, peak=False).peak is None
    y[5] = np.inf
    with pytest.warns(UserWarning, match="not finite"):
        assert np.isnan(kudio.audio_info(y, sr=16000).peak)


# -- names with brackets, and every folder reader alike ------------------------
def test_check_path_lists_rather_than_globs(tmp_path):
    """[1] is a glob character class: globbed, runs/exp[1] would not see its own
    runs/exp[1]2 and would hand that folder back as if it were free."""
    (tmp_path / "exp[1]").mkdir()
    (tmp_path / "exp[1]2").mkdir()
    free = Path(check_path(tmp_path / "exp[1]"))
    assert free.name == "exp[1]3" and not free.exists()


def test_load_audio_collects_what_check_input_collects(tmp_path, sine):
    sf.write(str(tmp_path / "ZOOM0001.WAV"), sine, 16000)
    sf.write(str(tmp_path / "b.wav"), sine, 16000)
    assert list(kudio.LoadAudio(tmp_path)) == check_input(tmp_path)[0]


# -- a rate is a positive number, and a file's a whole one -----------------------

#: everything in kudio.core.io that is given a rate: the call, with *f* a 1 s
#: 16 kHz file, *y* its samples and *out* a folder that is not there yet; the
#: argument; whether None is one of its values; and whether it has to be whole
RATED = {
    'file_load': (lambda f, y, out, sr: file_load(f, sr=sr), 'sr', True, True),
    # a short one: taken as 1 Hz it would come back 8000 times as long
    'resample from': (lambda f, y, out, sr: kudio.resample(y[:160], sr, 8000),
                      'orig_sr', False, False),
    'resample to': (lambda f, y, out, sr: kudio.resample(y, 8000, sr),
                    'target_sr', False, False),
    'audio_info': (lambda f, y, out, sr: kudio.audio_info(y, sr=sr), 'sr', False, True),
    'save_wave': (lambda f, y, out, sr: save_wave(out / "a.wav", y, sr), 'sr', False, True),
    'save_wave resampling': (
        lambda f, y, out, sr: save_wave(out / "a.wav", y, 8000, d_sample=sr),
        'd_sample', True, True),
    'convert_folder': (lambda f, y, out, sr: kudio.convert_folder(f, out, sr=sr),
                       'sr', True, True),
    'enhance_folder': (lambda f, y, out, sr: kudio.enhance_folder(f, out, sr=sr),
                       'sr', True, True),
}

NOT_A_RATE = [True, False, np.True_, np.array(True), '16000', b'16000', bytearray(b'16000'),
              float('nan'), float('inf'), float('-inf'), 0, 0.0, -1, 16000 + 0j,
              np.complex128(16000), [16000], np.array([16000])]

REFUSED = [(name, value) for name, (_, _, optional, whole) in RATED.items()
           for value in NOT_A_RATE + [16000.5] * whole + [None] * (not optional)]


@pytest.mark.parametrize("name, value", [
    pytest.param(n, v, id=f"{n}-{v!r}") for n, v in REFUSED])
def test_what_is_not_a_rate_is_refused(wav_file, sine, tmp_path, name, value):
    """Unchecked, True would go in as 1 Hz and come back as 1 sample a second,
    16000.5 would come back labelled 16000, and nan and inf would be blamed on
    the file."""
    call, argument, optional, whole = RATED[name]
    out = tmp_path / "out"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError) as caught:
            call(wav_file, sine, out, value)
    if name == 'audio_info' and value is None:
        assert str(caught.value) == "audio_info(array) needs sr="
    else:
        assert str(caught.value) == (f"{argument} must be a positive {'whole ' * whole}rate"
                                     f"{' or None' * optional}, got {value!r}")
    assert not out.exists(), "a mistake in the call leaves nothing behind"


@pytest.mark.parametrize("value", [16000.0, np.int32(16000), np.float64(16000.0),
                                   np.array(16000), Fraction(16000), Decimal(16000)], ids=repr)
@pytest.mark.parametrize("name", RATED)
def test_a_whole_rate_of_any_type_is_that_rate(wav_file, sine, tmp_path, name, value):
    call = RATED[name][0]

    def outcome(sr, folder):
        result = call(wav_file, sine, tmp_path / folder, sr)
        written = {p.name: p.read_bytes() for p in (tmp_path / folder).glob("*.wav")}
        if isinstance(result, tuple):                            # file_load
            return result[0].tolist(), result[1], type(result[1])
        if isinstance(result, np.ndarray):                       # resample
            return result.tolist()
        if isinstance(result, kudio.AudioInfo):
            return result, type(result.sr)
        return written

    assert outcome(value, "given") == outcome(16000, "int")


@pytest.mark.parametrize("value", [22254.5, np.float32(22254.5), Fraction(244800, 11),
                                   Decimal("22254.5")], ids=repr)
def test_an_array_can_be_at_a_rate_that_is_not_whole(sine, value):
    """The sound chip of the first Macintosh ran at 22254.5 Hz. No file can be
    at such a rate; what was recorded at it still has to be resampled from it."""
    out = kudio.resample(sine, value, 8000)
    assert len(out) == int(np.ceil(len(sine) * 8000 / float(value)))
    assert len(kudio.resample(out, 8000, value)) == pytest.approx(len(sine), abs=2)


# -- a shape no file can hold --------------------------------------------------

@pytest.mark.parametrize("shape", [(), (10, 2, 2), (1, 1, 16000), (100, 0), (1025, 1025)], ids=repr)
@pytest.mark.parametrize("d_sample", [None, 8000])
def test_a_shape_no_file_can_hold_leaves_the_file_alone(tmp_path, sine, shape, d_sample):
    """libsndfile opened the file first, which empties it, and looked at the
    samples then."""
    path = tmp_path / "a.wav"
    save_wave(path, sine, 16000)
    before = path.read_bytes()
    with pytest.raises(ValueError) as caught:
        save_wave(path, np.zeros(shape), 16000, d_sample=d_sample)
    assert str(caught.value) == (f"save_wave() writes (frames,) or (frames, channels) with 1 to "
                                 f"1024 channels, got shape {shape}")
    assert path.read_bytes() == before
    with pytest.raises(ValueError):
        save_wave(tmp_path / "new" / "a.wav", np.zeros(shape), 16000)
    assert not (tmp_path / "new").exists()


@pytest.mark.parametrize("shape", [(0,), (0, 2), (4, 1), (1024, 1024)], ids=repr)
def test_every_shape_a_file_can_hold_is_written(tmp_path, shape):
    save_wave(tmp_path / "a.wav", np.zeros(shape), 16000)
    info = sf.info(str(tmp_path / "a.wav"))
    assert (info.frames, info.channels) == (shape[0], 1 if len(shape) == 1 else shape[1])


@contextlib.contextmanager
def _held(path, share):
    """*path* open to another program on Windows, which shares it with nobody
    (0) or lets it be read (1)."""
    import ctypes
    from ctypes import wintypes
    create = ctypes.windll.kernel32.CreateFileW
    create.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
                       wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    create.restype = wintypes.HANDLE
    # read access, and the file has to exist
    handle = create(str(path), 0x80000000, share, None, 3, 0x80, None)
    try:
        yield
    finally:
        ctypes.windll.kernel32.CloseHandle(wintypes.HANDLE(handle))


# -- a write that fails --------------------------------------------------------

#: a file of each kind, and what cannot be written in its place: the extension,
#: the subtype, then the shape and the rate
REFUSED_BY_LIBSNDFILE = {
    'nine channels of FLAC': ('flac', 'PCM_16', (100, 9), 16000),
    'three channels of MP3': ('mp3', 'MPEG_LAYER_III', (100, 3), 16000),
    'FLAC at 1 MHz': ('flac', 'PCM_16', (100,), 1000000),
    'Vorbis in a WAV': ('wav', 'VORBIS', (100,), 16000),
    'float in a FLAC': ('flac', 'FLOAT', (100,), 16000),
}


def _there(tmp_path, sine, extension):
    """A file worth keeping, alone in its folder."""
    path = tmp_path / f"take.{extension}"
    save_wave(path, sine, 16000, subtype='MPEG_LAYER_III' if extension == 'mp3' else 'PCM_16')
    return path, path.read_bytes()


@pytest.mark.parametrize("name", REFUSED_BY_LIBSNDFILE)
def test_what_libsndfile_refuses_leaves_the_file_there_as_it_was(tmp_path, sine, name):
    """It empties the file first and refuses then: 9 channels of FLAC, 3 of MP3
    and FLAC at 1 MHz each left 0 bytes."""
    extension, subtype, shape, sr = REFUSED_BY_LIBSNDFILE[name]
    path, before = _there(tmp_path, sine, extension)
    with pytest.raises(AudioIOError) as caught:
        save_wave(path, np.zeros(shape), sr, subtype=subtype)
    assert str(caught.value).startswith(f"Could not write audio file: {path} (")
    assert ".tmp" not in str(caught.value)
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path], "and nothing is left beside it"


@pytest.mark.parametrize("error, raised, said", [
    (OSError(28, "No space left on device"), AudioIOError, "No space left on device"),
    (RuntimeError("the disk was pulled out"), AudioIOError, "the disk was pulled out"),
    (MemoryError(), AudioIOError, "MemoryError"),
    (KeyboardInterrupt(), KeyboardInterrupt, None),
], ids=["disk full", "an error", "no text", "Ctrl-C"])
def test_a_write_that_breaks_off_half_way_leaves_the_file_there(tmp_path, sine, monkeypatch,
                                                                error, raised, said):
    path, before = _there(tmp_path, sine, "wav")
    write, beside = sf.SoundFile.write, []

    def half(self, data):
        write(self, data[:len(data) // 2])
        self.flush()
        beside.extend(p for p in tmp_path.iterdir() if p != path)
        raise error

    with monkeypatch.context() as patched:
        patched.setattr(sf.SoundFile, "write", half)
        with pytest.raises(raised, match=said):
            save_wave(path, -sine, 16000)
        with pytest.raises(raised):
            save_wave(tmp_path / "new.wav", -sine, 16000)
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]
    assert len(beside) == 2 and {p.suffix for p in beside} == {".tmp"}, (
        "half a file is not one a folder tool would collect")


def test_a_file_that_is_replaced_is_the_new_one(tmp_path, sine):
    path, before = _there(tmp_path, sine, "wav")
    save_wave(path, 0.5 * sine, 8000)
    assert path.read_bytes() != before
    info = kudio.audio_info(path)
    assert (info.sr, info.format) == (8000, "WAV") and info.peak == pytest.approx(0.25, abs=0.01)
    assert list(tmp_path.iterdir()) == [path]


def test_a_relative_name_is_written_where_it_says(tmp_path, sine, monkeypatch):
    monkeypatch.chdir(tmp_path)
    save_wave("take.wav", sine, 16000)
    save_wave(os.path.join("sub", "take.flac"), sine, 16000)
    assert sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*.*")) == [
        "sub/take.flac", "take.wav"]


@pytest.mark.parametrize("name", ["take", "take.xyz", "take.m4a"])
def test_an_extension_that_is_no_format_writes_nothing(tmp_path, sine, name):
    with pytest.raises(AudioIOError, match="there is no format to take from the extension"):
        save_wave(tmp_path / name, sine, 16000)
    assert not list(tmp_path.iterdir())


def test_a_link_stays_a_link(tmp_path, sine):
    real = tmp_path / "real"
    real.mkdir()
    path, _ = _there(real, sine, "wav")
    link = tmp_path / "link.wav"
    try:
        link.symlink_to(path)
    except OSError as e:
        pytest.skip(f"no links here: {e}")
    save_wave(link, 0.5 * sine, 8000)
    assert link.is_symlink() and not path.is_symlink()
    assert kudio.audio_info(path).sr == 8000
    assert [p.name for p in real.iterdir()] == ["take.wav"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["link.wav", "real"]


def test_the_file_behind_a_link_is_the_one_replaced(tmp_path, sine, monkeypatch):
    """The same where links take an administrator."""
    path, _ = _there(tmp_path, sine, "wav")
    name = tmp_path / "other" / "link.wav"
    name.parent.mkdir()
    monkeypatch.setattr(os.path, "islink", lambda p: str(p) == str(name))
    monkeypatch.setattr(os.path, "realpath", lambda p, **k: str(path))
    save_wave(name, 0.5 * sine, 8000)
    assert kudio.audio_info(path).sr == 8000
    assert not list(name.parent.iterdir())


def test_what_is_no_ordinary_file_is_not_replaced():
    from kudio.core.io import _put_in_place
    seen = []
    _put_in_place(Path(os.devnull), 'WAV', seen.append)
    assert seen == [str(Path(os.devnull))]


@pytest.mark.skipif(os.name == 'nt', reason="Windows has no permission bits")
def test_the_permissions_of_the_file_are_kept(tmp_path, sine):
    path, _ = _there(tmp_path, sine, "wav")
    path.chmod(0o640)
    save_wave(path, -sine, 16000)
    assert path.stat().st_mode & 0o777 == 0o640


def test_a_file_that_may_not_be_written_is_not_replaced(tmp_path, sine):
    """Replacing a file asks for the folder's permission alone."""
    path, before = _there(tmp_path, sine, "wav")
    path.chmod(0o444)
    try:
        if os.access(path, os.W_OK):
            pytest.skip("root writes every file")
        with pytest.raises(AudioIOError, match="Permission denied"):
            save_wave(path, -sine, 16000)
    finally:
        path.chmod(0o644)
    assert path.read_bytes() == before and list(tmp_path.iterdir()) == [path]


windows = pytest.mark.skipif(os.name != 'nt', reason="only Windows keeps a file that is open")


def _as_written(folder, wave, sr):
    """What save_wave makes of *wave*, byte for byte."""
    folder.mkdir()
    save_wave(folder / "a.wav", wave, sr)
    return (folder / "a.wav").read_bytes()


@windows
@pytest.mark.parametrize("seconds", [0.25, 1.0, 3.0], ids=["shorter", "as long", "longer"])
@pytest.mark.parametrize("held", [lambda p: open(p, 'rb'), lambda p: sf.SoundFile(str(p))],
                         ids=["by Python", "by soundfile"])
def test_a_file_that_is_open_elsewhere_is_overwritten(tmp_path, sine, held, seconds):
    """Windows replaces no file that is open, and lets it be written to."""
    path, before = _there(tmp_path, sine, "wav")
    new = 0.5 * np.sin(np.arange(int(16000 * seconds)) / 3)
    with held(path):
        save_wave(path, new, 16000)
    assert path.read_bytes() == _as_written(tmp_path / "apart", new, 16000) != before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["apart", "take.wav"]


@windows
@pytest.mark.parametrize("share", [0, 1], ids=["shared with nobody", "to be read"])
def test_a_file_that_is_closed_to_writing_is_left_as_it_was(tmp_path, sine, share):
    path, before = _there(tmp_path, sine, "wav")
    with _held(path, share):
        with pytest.raises(AudioIOError, match="may be open in another program"):
            save_wave(path, -sine, 16000)
        assert [p.name for p in tmp_path.iterdir()] == ["take.wav"]
    assert path.read_bytes() == before


@windows
def test_a_file_in_another_programs_memory_is_not_cut_short(tmp_path, sine):
    """Windows does not shorten it, and the new samples on top of the old
    would be neither recording."""
    path, before = _there(tmp_path, sine, "wav")
    mapped = np.memmap(path, dtype=np.uint8, mode='r')
    try:
        with pytest.raises(AudioIOError, match="may be open in another program"):
            save_wave(path, sine[:4000], 16000)
        assert bytes(mapped) == before
        assert [p.name for p in tmp_path.iterdir()] == ["take.wav"]
    finally:
        mapped._mmap.close()


@windows
def test_no_other_error_is_answered_by_overwriting(tmp_path, sine, monkeypatch):
    path, before = _there(tmp_path, sine, "wav")

    def broken(source, target):
        raise OSError(5, "Input/output error")
    with monkeypatch.context() as patched:
        patched.setattr(os, "replace", broken)
        with pytest.raises(AudioIOError, match="Input/output error"):
            save_wave(path, -sine, 16000)
    assert path.read_bytes() == before and list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("old, new", [(b"a long one", b"short"), (b"short", b"a long one"),
                                      (b"same", b"size")])
def test_a_file_is_overwritten_to_its_new_length(tmp_path, old, new):
    from kudio.core.io import _overwrite
    (tmp_path / "old").write_bytes(old)
    (tmp_path / "new").write_bytes(new)
    assert _overwrite(str(tmp_path / "new"), str(tmp_path / "old")) is True
    assert (tmp_path / "old").read_bytes() == new


# -- files that cannot be read, and what is said about them --------------------

#: every way a file is read: the call, the error, and the words before the path
READS = {
    'file_load': (lambda p: file_load(p), AudioIOError, "Could not load audio file: "),
    'file_load at its rate': (lambda p: file_load(p, sr=16000), AudioIOError,
                              "Could not load audio file: "),
    'file_load at a rate': (lambda p: file_load(p, sr=8000), AudioIOError,
                            "Could not load audio file: "),
    'file_load in stereo': (lambda p: file_load(p, sr=8000, mono=False), AudioIOError,
                            "Could not load audio file: "),
    'audio_info': (lambda p: kudio.audio_info(p), AudioIOError,
                   "Could not read audio header: "),
    'audio_info of the header': (lambda p: kudio.audio_info(p, peak=False), AudioIOError,
                                 "Could not read audio header: "),
    'Synthesizer': (lambda p: kudio.core.synth._load(p, None, "clean"), kudio.SynthesisError,
                    "could not read clean file "),
    'Synthesizer at a rate': (lambda p: kudio.core.synth._load(p, 8000, "noise"),
                              kudio.SynthesisError, "could not read noise file "),
    'save_spectrogram_as_wave': (
        lambda p: kudio.save_spectrogram_as_wave(Path(p).parent / "out.wav", p,
                                                 np.zeros((63, 257))),
        AudioIOError, "Could not load audio file: "),
}


def _unreadable(folder, what):
    path = folder / "take.wav"
    if what == "folder":
        path.mkdir()
    elif what == "empty":
        path.write_bytes(b"")
    return path


@pytest.mark.parametrize("spelled", [str, Path])
@pytest.mark.parametrize("what, said", [("missing", "no such file"),
                                        ("folder", "it is a folder, not a file"),
                                        ("empty", "the file is empty, 0 bytes")])
@pytest.mark.parametrize("name", READS)
def test_a_path_with_nothing_to_read_is_told_apart(tmp_path, name, what, said, spelled):
    """libsndfile says "System error" for a missing file and a folder alike,
    and librosa 0.11 warns about audioread on the way to each of them."""
    call, error, words = READS[name]
    path = _unreadable(tmp_path, what)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(error) as caught:
            call(spelled(path))
    assert type(caught.value) is error, "and no second base to catch it by"
    message = str(caught.value)
    assert message.startswith(f"{words}{path}") and said in message


def test_a_null_in_the_name_is_no_file(tmp_path):
    with pytest.raises(AudioIOError, match="no such file"):
        file_load(str(tmp_path / "a.wav") + "\0")


def test_a_path_that_may_not_be_looked_at_is_not_called_missing(wav_file, monkeypatch):
    def refuse(path, *args, **kwargs):
        raise PermissionError(13, "Permission denied", str(path))

    with monkeypatch.context() as patched:
        patched.setattr(os, "stat", refuse)
        with pytest.raises(AudioIOError) as caught:
            file_load(wav_file)
    assert str(caught.value).endswith("(Permission denied)")


def test_only_a_regular_file_is_empty_at_zero_bytes():
    """A pipe has a size of 0 as well, and can still hold audio."""
    from kudio.core.io import _nothing_to_read
    assert os.stat(os.devnull).st_size == 0
    assert _nothing_to_read(os.devnull) is None


@contextlib.contextmanager
def _closed_to_us(path):
    """*path* as it is while another program holds it, or under somebody
    else's permissions."""
    if os.name == 'nt':
        with _held(path, 0):
            yield
    else:
        path.chmod(0)
        if os.access(path, os.R_OK):
            pytest.skip("root reads every file")
        try:
            yield
        finally:
            path.chmod(0o600)


@pytest.mark.parametrize("name", READS)
def test_a_file_that_will_not_open_says_what_the_system_says(wav_file, name):
    call, error, words = READS[name]
    with _closed_to_us(wav_file):
        with pytest.raises(error) as caught:
            call(wav_file)
    assert str(caught.value).startswith(f"{words}{wav_file}")
    assert "Permission denied" in str(caught.value)
    assert "System error" not in str(caught.value)


@pytest.fixture
def failing(monkeypatch):
    """Make soundfile raise what it is handed, however it is asked to read."""
    def install(error):
        def fail(*args, **kwargs):
            raise error
        for name in ("read", "info", "SoundFile"):
            monkeypatch.setattr(sf, name, fail)
    return install


@pytest.mark.parametrize("error, said", [(EOFError(), "EOFError"),
                                         (RuntimeError("the disk was pulled out"),
                                          "the disk was pulled out")], ids=repr)
@pytest.mark.parametrize("name", READS)
def test_an_error_without_text_is_named_by_its_type(wav_file, failing, name, error, said):
    """An error with a text keeps it."""
    failing(error)
    call, expected, words = READS[name]
    with pytest.raises(expected) as caught:
        call(wav_file)
    message = str(caught.value)
    assert message in (f"{words}{wav_file} ({said})", f"{words}{wav_file}: {said}")


def test_importing_kudio_and_reading_a_wav_need_no_audioread(wav_file):
    code = ("import sys, kudio; kudio.file_load(sys.argv[1]); kudio.audio_info(sys.argv[1]); "
            "kudio.file_load(sys.argv[1], sr=16000); kudio.file_load(sys.argv[1], sr=8000); "
            "kudio.core.synth._load(sys.argv[1], None, 'clean'); "
            "print([m for m in ('audioread', 'librosa', 'numba') if m in sys.modules])")
    done = subprocess.run([sys.executable, "-c", code, str(wav_file)], capture_output=True,
                          text=True, cwd=str(Path(kudio.__file__).parents[1]), timeout=300)
    assert done.stdout.strip() == "[]", done.stdout + done.stderr


# -- NaN and inf ---------------------------------------------------------------

@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("sr, mono", [(None, True), (16000, True), (16000, False),
                                      (16000.0, True)])
def test_a_file_with_nan_is_read_at_its_own_rate(tmp_path, bad, sr, mono):
    """Asking for the rate the file is at is no resampling, and repairing a
    clip starts with reading it."""
    path = tmp_path / "take.wav"
    written = _write_non_finite(path, bad=bad)
    with pytest.warns(UserWarning, match=r"take\.wav: 10 of 16000 sample\(s\)") as caught:
        y, rate = file_load(path, sr=sr, mono=mono)
    assert rate == 16000
    assert np.array_equal(y, written.astype(np.float32), equal_nan=True)
    assert caught[0].filename == __file__


@pytest.mark.parametrize("bad", [np.nan, np.inf])
@pytest.mark.parametrize("mono", [True, False])
def test_a_file_with_nan_is_not_resampled(tmp_path, bad, mono):
    """soxr's filter is hundreds of samples long, and would make that many NaN
    of each one."""
    path = tmp_path / "take.wav"
    _write_non_finite(path, bad=bad)
    with warnings.catch_warnings():
        warnings.simplefilter("error")                   # the error says it all
        with pytest.raises(AudioIOError) as caught:
            file_load(path, sr=8000, mono=mono)
    message = str(caught.value)
    assert message.startswith(f"Could not load audio file: {path} (10 of 16000 sample(s) are "
                              f"not finite (NaN or inf), so it cannot be resampled from 16000 "
                              f"to 8000 Hz")
    assert "sr=None" in message


def test_nan_in_one_channel_is_counted_where_it_is(tmp_path):
    path = tmp_path / "st.wav"
    y = np.full((1000, 2), 0.25, np.float32)
    y[10:13, 1] = np.nan
    sf.write(str(path), y, 16000, subtype="FLOAT")
    with pytest.warns(UserWarning, match=r"3 of 2000 sample\(s\)"):
        kept, _ = file_load(path, sr=16000, mono=False)
    assert np.array_equal(kept, y, equal_nan=True)
    with pytest.warns(UserWarning, match=r"3 of 1000 sample\(s\)"):
        mixed, _ = file_load(path, sr=16000)
    assert np.flatnonzero(np.isnan(mixed)).tolist() == [10, 11, 12]


def test_convert_folder_names_the_file_it_could_not_resample(tmp_path, sine):
    _write_non_finite(tmp_path / "bad.wav")
    save_wave(tmp_path / "good.wav", sine, 16000)
    result = kudio.convert_folder(tmp_path, tmp_path / "out", sr=8000)
    assert result.written == 1
    assert [p.name for p, _ in result.failed] == ["bad.wav"]
    assert "cannot be resampled from 16000 to 8000 Hz" in result.failed[0][1]


@pytest.mark.parametrize("sr", [None, 16000, 8000])
def test_a_synthesis_input_with_nan_is_named_with_the_count(tmp_path, sr):
    path = tmp_path / "take.wav"
    _write_non_finite(path)
    with pytest.raises(kudio.SynthesisError) as caught:
        kudio.core.synth._load(path, sr, "noise")
    assert str(caught.value) == (f"could not read noise file {path}: 10 of 16000 sample(s) are "
                                 f"not finite (NaN or inf)")


def test_there_is_no_phase_in_a_file_with_nan(tmp_path, sine):
    path = tmp_path / "take.wav"
    _write_non_finite(path)
    spec = kudio.waveform_to_spectrogram(sine)
    with pytest.raises(AudioIOError) as caught:
        kudio.save_spectrogram_as_wave(tmp_path / "out.wav", path, spec)
    assert str(caught.value) == (f"Could not load audio file: {path} (10 of 16000 sample(s) are "
                                 f"not finite (NaN or inf))")
    assert not (tmp_path / "out.wav").exists()


# -- a WAV that ends before its header says ------------------------------------

#: the ways a file is read, all of which have to say so
CUT_READS = {
    'file_load': lambda p: file_load(p),
    'file_load at its rate': lambda p: file_load(p, sr=16000),
    'file_load in stereo': lambda p: file_load(p, sr=16000, mono=False),
    'audio_info': lambda p: kudio.audio_info(p),
    'audio_info of the header': lambda p: kudio.audio_info(p, peak=False),
    'Synthesizer': lambda p: kudio.core.synth._load(p, None, "clean"),
    'play_audio': lambda p: kudio.play_audio(str(p)),
}


@pytest.fixture
def quiet_card(monkeypatch):
    """play_audio plays to nothing."""
    monkeypatch.setitem(sys.modules, "sounddevice",
                        types.SimpleNamespace(play=lambda y, rate: None))


def _cut(path, keep):
    """*path* with the first *keep* of its bytes, or that share of them."""
    whole = path.read_bytes()
    path.write_bytes(whole[:int(len(whole) * keep) if isinstance(keep, float) else keep])
    return path


def _riff(*chunks, form=b'WAVE', riff=b'RIFF'):
    body = b''.join(name + length.to_bytes(4, 'little') + content
                    for name, length, content in chunks)
    return riff + (4 + len(body)).to_bytes(4, 'little') + form + body


def _fmt(average=32000):
    """16 kHz, 16 bits, one channel."""
    return b'fmt ', 16, struct.pack('<HHIIHH', 1, 1, 16000, average, 2, 16)


FMT = _fmt()


def _data(length, held):
    return b'data', length, bytes(held)


def _listed(path, *chunks, **kwargs):
    path.write_bytes(_riff(*chunks, **kwargs))
    return path


def _in_silence(read, path):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return read(path)


@pytest.mark.parametrize("name", CUT_READS)
def test_a_wav_that_was_cut_short_is_said_to_be(tmp_path, sine, quiet_card, name):
    """libsndfile reads what there is, so half of a take is a shorter take."""
    path = tmp_path / "take.wav"
    save_wave(path, sine, 16000)
    _cut(path, 44 + 16000)                                # of 44 + 32000
    with pytest.warns(UserWarning) as caught:
        CUT_READS[name](path)
    assert [str(w.message) for w in caught] == [
        f"{path}: the header says 1.00 s and 0.50 s are there, 16000 bytes fewer. "
        f"The file may have been cut short."]
    assert caught[0].filename == __file__


def test_what_is_read_of_it_is_what_soundfile_reads(tmp_path, sine):
    path = tmp_path / "take.wav"
    save_wave(path, np.stack([sine, -sine], axis=1), 16000)
    _cut(path, 0.6)
    expected, _ = sf.read(str(path), dtype='float32')
    with pytest.warns(UserWarning, match="cut short"):
        y, _ = file_load(path, mono=False)
    assert np.array_equal(y, expected)
    with pytest.warns(UserWarning, match="cut short"):
        info = kudio.audio_info(path, peak=False)
    assert (info.frames, info.channels, info.peak) == (len(expected), 2, None)


@pytest.mark.parametrize("subtype, channels", [("GSM610", 1)] + [
    (subtype, channels) for channels in (1, 2) for subtype in (
        "PCM_U8", "PCM_16", "PCM_24", "PCM_32", "FLOAT", "DOUBLE", "ULAW", "ALAW",
        "IMA_ADPCM", "MS_ADPCM")])
def test_every_kind_of_wav_is_held_to_its_header(tmp_path, subtype, channels):
    path = tmp_path / "take.wav"
    y = 0.3 * np.sin(np.arange(16001 * channels) / 9).reshape(-1, channels)
    sf.write(str(path), y, 16000, subtype=subtype)
    _in_silence(file_load, path)                          # 16001 frames: an odd length
    _cut(path, 0.5)
    # at its own rate: soundfile does not seek in GSM, which the rest would need
    with pytest.warns(UserWarning, match=r"the header says 1\.0\d s and 0\.[45]\d s are there"):
        file_load(path)


@pytest.mark.parametrize("name", CUT_READS)
def test_a_whole_wav_is_read_without_a_word(tmp_path, quiet_card, name):
    for frames in (0, 1, 999, 1000):
        for subtype in ("PCM_U8", "PCM_16", "PCM_24", "FLOAT"):
            path = tmp_path / f"{frames}_{subtype}.wav"
            save_wave(path, 0.1 * np.ones(frames), 16000, subtype=subtype)
            _in_silence(CUT_READS[name], path)


@pytest.mark.parametrize("short, said", [(0, False), (1, False), (2, False), (3, True),
                                         (4, True)])
def test_a_block_is_not_counted_for_a_cut(tmp_path, short, said):
    """A frame that is missing is not a copy that broke off."""
    path = _listed(tmp_path / "a.wav", FMT, _data(32000, 32000 - short))
    if said:
        with pytest.warns(UserWarning, match=f"{short} bytes fewer"):
            kudio.audio_info(path, peak=False)
    else:
        _in_silence(lambda p: kudio.audio_info(p, peak=False), path)


@pytest.mark.parametrize("length", [0, 0x7FFFF000, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF], ids=hex)
@pytest.mark.parametrize("name", ['file_load', 'audio_info', 'audio_info of the header'])
def test_a_length_left_by_a_program_that_streams_is_none(tmp_path, name, length):
    path = _listed(tmp_path / "a.wav", FMT, _data(length, 32000))
    _in_silence(CUT_READS[name], path)


def test_the_longest_length_that_is_one_is_held_to(tmp_path):
    path = _listed(tmp_path / "a.wav", FMT, _data(0x7FFFEFFE, 32000))
    with pytest.warns(UserWarning, match="cut short"):
        kudio.audio_info(path, peak=False)


LIST = (b'LIST', 26, b'INFOISFT' + (14).to_bytes(4, 'little') + b'Lavf62.0.100\x00\x00')


@pytest.mark.parametrize("chunks", [
    (FMT, _data(32000, 32000), LIST),
    (LIST, FMT, _data(32000, 32000)),
    (FMT, LIST, (b'JUNK', 7, bytes(8)), _data(32000, 32000), LIST),
    (FMT, _data(31999, 32000)),                       # odd, with the byte that fills it up
    (FMT, _data(31999, 31999)),                       # and without
    (FMT, _data(31999, 32000), LIST),
    (FMT, _data(32000, 32000), (b'LIST', 5000, bytes(20))),      # cut in what comes after
], ids=["a list after", "a list before", "chunks all round", "odd", "odd, not filled up",
        "odd, a list after", "cut after the samples"])
def test_what_surrounds_the_samples_is_no_cut(tmp_path, chunks):
    path = _listed(tmp_path / "a.wav", *chunks)
    for name in ('file_load', 'audio_info', 'audio_info of the header'):
        _in_silence(CUT_READS[name], path)


@pytest.mark.parametrize("chunks", [
    (LIST, FMT, (b'JUNK', 7, bytes(8)), _data(32000, 16000)),
    (FMT, (b'bext', 602, bytes(602)), _data(32000, 16000)),
], ids=["an odd chunk before", "a broadcast wave"])
def test_the_samples_are_found_behind_other_chunks(tmp_path, chunks):
    path = _listed(tmp_path / "a.wav", *chunks)
    with pytest.warns(UserWarning, match=r"says 1\.00 s and 0\.50 s are there, 16000 bytes"):
        kudio.audio_info(path, peak=False)


@pytest.mark.parametrize("container, kwargs", [
    ('RF64', {}), ('W64', {}), ('AIFF', {}), ('CAF', {}), ('AU', {}),
    ('WAV', {'endian': 'BIG'})], ids=str)
def test_only_a_riff_wav_is_held_to_its_header(tmp_path, sine, container, kwargs):
    path = tmp_path / "take.wav"
    sf.write(str(path), sine, 16000, format=container, subtype='PCM_16', **kwargs)
    _cut(path, 0.5)
    try:
        expected = len(sf.read(str(path))[0])
    except sf.SoundFileError:
        return                                        # refused, so nothing is read short
    y, _ = _in_silence(file_load, path)
    assert len(y) == expected
    _in_silence(kudio.audio_info, path)


@pytest.mark.parametrize("head", [{'riff': b'RF64'}, {'riff': b'RIFX'}, {'riff': b'FORM'},
                                  {'form': b'AVI '}], ids=str)
def test_the_same_chunks_in_another_container_are_not_judged(tmp_path, head):
    from kudio.core.io import _wav_shortfall
    path = _listed(tmp_path / "a.wav", FMT, _data(32000, 16000))
    assert _wav_shortfall(path) == (1.0, 16000)
    assert _wav_shortfall(_listed(path, FMT, _data(32000, 16000), **head)) is None


@pytest.mark.parametrize("average", [0, 1, 0xFFFFFFFF])
def test_the_seconds_of_samples_are_counted(tmp_path, average):
    """Not every writer fills in the bytes a second that a player goes by."""
    path = _listed(tmp_path / "a.wav", _fmt(average), _data(32000, 16000))
    with pytest.warns(UserWarning, match=r"says 1\.00 s and 0\.50 s are there"):
        kudio.audio_info(path, peak=False)


def test_no_header_is_too_broken_to_be_looked_at(tmp_path, sine):
    """Whatever is in it, the answer is a number of bytes or none."""
    from kudio.core.io import _wav_shortfall
    path = tmp_path / "take.wav"
    save_wave(path, sine[:500], 16000)
    whole = path.read_bytes()
    rng = np.random.default_rng(0)
    cut = [whole[:keep] for keep in range(60)]
    for _ in range(300):
        broken = bytearray(whole[:int(rng.integers(60, len(whole)))])
        for _ in range(int(rng.integers(1, 6))):
            broken[int(rng.integers(0, 60))] = int(rng.integers(0, 256))
        cut.append(bytes(broken))
    for content in cut:
        path.write_bytes(content)
        short = _wav_shortfall(path)
        assert short is None or (short[0] > 0 and short[1] > 0)
    path.write_bytes(_riff(*[(b'JUNK', 0, b'')] * 10000, FMT, _data(32000, 10)))
    assert _wav_shortfall(path) is None, "given up after so many"


def test_a_cut_file_can_be_made_one_that_is_not_read(tmp_path, sine):
    """The way out for a sweep over a corpus, as inspection.md gives it."""
    save_wave(tmp_path / "take.wav", sine, 16000)
    whole, _ = file_load(tmp_path / "take.wav")
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*cut short")
        assert np.array_equal(file_load(tmp_path / "take.wav")[0], whole)
        _cut(tmp_path / "take.wav", 0.5)
        with pytest.raises(AudioIOError, match="may have been cut short"):
            file_load(tmp_path / "take.wav")


def test_a_folder_of_takes_names_the_one_that_was_cut(tmp_path, sine):
    for name in ("a", "b"):
        save_wave(tmp_path / f"{name}.wav", sine, 16000)
    _cut(tmp_path / "b.wav", 0.5)
    with pytest.warns(UserWarning) as caught:
        result = kudio.convert_folder(tmp_path, tmp_path / "out")
    assert result.written == 2
    said = [str(w.message) for w in caught]
    assert len(said) == 1 and said[0].startswith(f"{tmp_path / 'b.wav'}: the header says")


def test_a_set_that_is_scored_says_it_with_the_item(tmp_path, sine):
    """The rate of the set is read from the headers of its items before that,
    which is no place to say it as well."""
    for side in ("ref", "deg"):
        for name in ("a", "b"):
            save_wave(tmp_path / side / f"{name}.wav", 0.5 * sine + 0.01, 16000)
        _cut(tmp_path / side / "a.wav", 0.5)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = kudio.score_dataset(tmp_path / "deg", reference=tmp_path / "ref",
                                     metrics=["snr"])
    said = [str(w.message) for w in caught if "cut short" in str(w.message)]
    assert len(said) == 1 and said[0].startswith("score_dataset(): 1 of 2 items")
    notes = {row['id']: row['notes'] for row in result.rows}
    assert [note.split(": the header says")[0] for note in notes['a']] == [
        str(tmp_path / side / "a.wav") for side in ("ref", "deg")]
    assert not notes['b']
