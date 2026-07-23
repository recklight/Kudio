# -*- coding: utf-8 -*-
import numpy as np
import pytest
from scipy.io import wavfile

SR = 16000


def make_sine(freq: float = 440.0, seconds: float = 1.0, sr: int = SR) -> np.ndarray:
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def write_wav(path, wave: np.ndarray, sr: int = SR) -> None:
    wavfile.write(str(path), sr, (wave * np.iinfo(np.int16).max).astype(np.int16))


@pytest.fixture
def sine():
    return make_sine()


@pytest.fixture
def wav_file(tmp_path):
    p = tmp_path / "sine.wav"
    write_wav(p, make_sine())
    return p


@pytest.fixture
def wav_dir(tmp_path):
    d = tmp_path / "waves"
    d.mkdir()
    for i, freq in enumerate((220, 440, 880)):
        write_wav(d / f"tone_{i}.wav", make_sine(freq))
    return d


@pytest.fixture
def clean_noise_dirs(tmp_path):
    clean = tmp_path / "clean"
    noise = tmp_path / "noise"
    clean.mkdir()
    noise.mkdir()
    for i, freq in enumerate((220, 440)):
        write_wav(clean / f"clean_{i}.wav", make_sine(freq))
    rng = np.random.default_rng(0)
    write_wav(noise / "white.wav",
              (0.3 * rng.standard_normal(SR)).astype(np.float32))
    return clean, noise
