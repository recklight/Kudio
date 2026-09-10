# -*- coding: utf-8 -*-
import numpy as np
import pytest
from scipy.io import wavfile

from kudio.cli import build_parser, main


def _write(path, freq=440, sr=16000, seconds=1.0):
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    y = (0.5 * np.sin(2 * np.pi * freq * t) * np.iinfo(np.int16).max).astype(np.int16)
    wavfile.write(str(path), sr, y)


@pytest.mark.parametrize("argv", [
    ["info", "x.wav"],
    ["convert", "a.wav", "b.wav", "--rate", "8000"],
    ["synth", "--clean", "c", "--noise", "n", "--out", "o"],
    ["trim", "a.wav", "b.wav"],
    ["devices"],
])
def test_parser_accepts_subcommands(argv):
    args = build_parser().parse_args(argv)
    assert args.command == argv[0]


def test_info_command(tmp_path, capsys):
    f = tmp_path / "t.wav"
    _write(f)
    assert main(["info", str(f)]) == 0
    out = capsys.readouterr().out
    assert "samplerate: 16000" in out
    assert "duration" in out


def test_convert_command(tmp_path):
    src = tmp_path / "s.wav"
    dst = tmp_path / "d.wav"
    _write(src)
    assert main(["convert", str(src), str(dst), "--rate", "8000"]) == 0
    import soundfile as sf
    info = sf.info(str(dst))
    assert info.samplerate == 8000


def test_trim_command(tmp_path):
    src = tmp_path / "s.wav"
    dst = tmp_path / "d.wav"
    sr = 16000
    t = np.linspace(0, 0.5, sr // 2, endpoint=False)
    tone = 0.5 * np.sin(2 * np.pi * 440 * t)
    y = np.concatenate([np.zeros(sr // 2), tone, np.zeros(sr // 2)])
    wavfile.write(str(src), sr, (y * np.iinfo(np.int16).max).astype(np.int16))
    assert main(["trim", str(src), str(dst), "--top-db", "20"]) == 0
    assert dst.is_file()


def _write_speech(path, sr=16000):
    from conftest import make_speech
    import kudio
    kudio.save_wave(str(path), make_speech(sr=sr), sr)


@pytest.mark.parametrize("argv", [
    ["report", "x.wav"],
    ["loudness", "x.wav"],
    ["normalize", "a.wav", "b.wav", "--lufs", "-23"],
    ["vad", "x.wav"],
    ["pitch", "x.wav"],
    ["convert", "a", "b", "--recursive"],
])
def test_parser_accepts_the_new_subcommands(argv):
    assert build_parser().parse_args(argv).command == argv[0]


def test_normalize_targets_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["normalize", "a.wav", "b.wav", "--lufs", "-23", "--peak", "0.9"])


def test_report_command_prints_findings(tmp_path, capsys):
    f = tmp_path / "t.wav"
    _write(f)
    code = main(["report", str(f)])
    out = capsys.readouterr().out
    assert "dBFS" in out
    assert "perceptual score" in out          # the honesty note is not optional
    assert code in (0, 1)


def test_report_exits_nonzero_when_something_is_wrong(tmp_path, capsys):
    """A pure tone has no content above its own frequency: band-limited."""
    f = tmp_path / "tone.wav"
    _write(f)
    assert main(["report", str(f)]) == 1
    assert "·" in capsys.readouterr().out


def test_loudness_command(tmp_path, capsys):
    f = tmp_path / "t.wav"
    _write(f)
    assert main(["loudness", str(f)]) == 0
    assert "LUFS" in capsys.readouterr().out


def test_normalize_command_hits_the_target(tmp_path):
    import kudio
    src, dst = tmp_path / "s.wav", tmp_path / "d.wav"
    _write(src)
    assert main(["normalize", str(src), str(dst), "--lufs", "-20"]) == 0
    y, sr = kudio.file_load(dst, sr=None)
    assert kudio.loudness(y, sr) == pytest.approx(-20.0, abs=0.5)


def test_normalize_command_peak_mode(tmp_path):
    import kudio
    src, dst = tmp_path / "s.wav", tmp_path / "d.wav"
    _write(src)
    assert main(["normalize", str(src), str(dst), "--peak", "0.5"]) == 0
    assert kudio.audio_info(dst).peak == pytest.approx(0.5, abs=0.01)


def test_vad_command_lists_spans(tmp_path, capsys):
    f = tmp_path / "speech.wav"
    _write_speech(f)
    assert main(["vad", str(f)]) == 0
    out = capsys.readouterr().out
    assert "span(s)" in out
    assert "→" in out


def test_vad_command_exits_nonzero_on_silence(tmp_path, capsys):
    import kudio
    import numpy as np
    f = tmp_path / "quiet.wav"
    kudio.save_wave(str(f), np.zeros(16000, dtype=np.float32), 16000)
    assert main(["vad", str(f)]) == 1
    assert "no speech" in capsys.readouterr().out


def test_vad_command_can_split_to_files(tmp_path):
    f = tmp_path / "speech.wav"
    _write_speech(f)
    out_dir = tmp_path / "spans"
    assert main(["vad", str(f), "--split-to", str(out_dir)]) == 0
    assert list(out_dir.glob("*.wav"))


def test_pitch_command_summarises_the_contour(tmp_path, capsys):
    f = tmp_path / "speech.wav"
    _write_speech(f)
    assert main(["pitch", str(f), "--segments"]) == 0
    out = capsys.readouterr().out
    assert "voiced" in out and "semitones" in out
    assert "→" in out                       # the span listing


def test_pitch_command_exits_nonzero_when_nothing_is_voiced(tmp_path, capsys):
    import numpy as np

    import kudio
    f = tmp_path / "quiet.wav"
    kudio.save_wave(str(f), np.zeros(16000, dtype=np.float32), 16000)
    assert main(["pitch", str(f)]) == 1
    assert "no voiced frames" in capsys.readouterr().out


def test_pitch_command_writes_a_label_track(tmp_path):
    import kudio
    f = tmp_path / "speech.wav"
    _write_speech(f)
    labels = tmp_path / "voiced.txt"
    assert main(["pitch", str(f), "--labels", str(labels)]) == 0
    assert len(kudio.load_labels(labels)) >= 1


def test_convert_recursive_walks_a_folder(tmp_path, capsys):
    import kudio
    src, dst = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    for i in range(3):
        _write(src / f"t{i}.wav")
    assert main(["convert", str(src), str(dst), "--rate", "8000",
                 "--recursive"]) == 0
    assert "3/3" in capsys.readouterr().out
    assert all(kudio.audio_info(p, peak=False).sr == 8000
               for p in dst.glob("*.wav"))


def test_info_command_reports_the_encoding(tmp_path, capsys):
    f = tmp_path / "t.wav"
    _write(f)
    assert main(["info", str(f)]) == 0
    assert "encoding  : WAV / PCM_16" in capsys.readouterr().out


def test_enhance_command(tmp_path, capsys):
    import kudio
    src, dst = tmp_path / "s.wav", tmp_path / "d.wav"
    _write_speech(src)
    assert main(["enhance", str(src), str(dst), "--method", "logmmse"]) == 0
    out = capsys.readouterr().out
    assert "log-MMSE" in out and "noise floor" in out
    y, sr = kudio.file_load(dst)
    assert len(y) > 0


def test_enhance_takes_method_parameters(tmp_path, capsys):
    src, dst = tmp_path / "s.wav", tmp_path / "d.wav"
    _write_speech(src)
    assert main(["enhance", str(src), str(dst), "--method", "omlsa",
                 "--set", "q=0.5"]) == 0
    assert dst.is_file()


def test_enhance_rejects_a_malformed_set(tmp_path):
    src, dst = tmp_path / "s.wav", tmp_path / "d.wav"
    _write_speech(src)
    assert main(["enhance", str(src), str(dst), "--set", "q"]) == 2


def test_compare_command_ranks_every_method(tmp_path, capsys):
    src = tmp_path / "s.wav"
    _write_speech(src)
    assert main(["compare", str(src), "--no-metrics"]) == 0
    out = capsys.readouterr().out
    assert "method" in out and "floor dBFS" in out
    for name in ("logmmse", "wiener", "omlsa"):
        assert name in out
    # with no reference these are not quality scores, and it has to say so
    assert "not quality scores" in out


def test_compare_with_a_reference_shows_the_quality_columns(tmp_path, capsys):
    import kudio
    import numpy as np
    from conftest import make_speech

    sr = 16000
    clean = make_speech(sr=sr)
    noisy = clean + (0.05 * np.random.default_rng(1).standard_normal(len(clean))
                     ).astype(np.float32)
    kudio.save_wave(str(tmp_path / "clean.wav"), clean, sr)
    kudio.save_wave(str(tmp_path / "noisy.wav"), noisy, sr)

    assert main(["compare", str(tmp_path / "noisy.wav"),
                 "--reference", str(tmp_path / "clean.wav"), "--no-metrics"]) == 0
    out = capsys.readouterr().out
    assert "SI-SDR" in out and "dSNR" in out


def test_compare_can_write_every_result(tmp_path):
    src = tmp_path / "s.wav"
    _write_speech(src)
    out_dir = tmp_path / "results"
    assert main(["compare", str(src), "--no-metrics", "--methods", "wiener",
                 "logmmse", "--write-to", str(out_dir)]) == 0
    assert sorted(p.stem for p in out_dir.glob("*.wav")) == ["s_logmmse", "s_wiener"]
