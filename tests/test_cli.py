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
