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


def make_speech(sr: int = SR, seconds: float = 6.0, seed: int = 0) -> np.ndarray:
    """Two harmonic bursts over a noise floor — a stand-in for a recording.

    A pure sine is the wrong fixture for anything that reasons about *activity*:
    it has one constant level, so nothing in it stands out from anything else,
    and it has no noise floor to stand out from.
    """
    rng = np.random.default_rng(seed)
    y = (0.01 * rng.standard_normal(int(sr * seconds))).astype(np.float32)

    def burst(n: int, f0: float) -> np.ndarray:
        t = np.arange(n) / sr
        harmonics = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 12))
        envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 4 * t)     # syllable rate
        return (harmonics * envelope / 6).astype(np.float32)

    y[sr:2 * sr] += burst(sr, 120.0)
    y[int(3.5 * sr):int(4.8 * sr)] += burst(int(1.3 * sr), 150.0)
    return y


@pytest.fixture
def sine():
    return make_sine()


@pytest.fixture
def speech():
    """Speech-like audio: bursts at 1.0-2.0s and 3.5-4.8s over a noise floor."""
    return make_speech()


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
