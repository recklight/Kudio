# -*- coding: utf-8 -*-
import numpy as np
import pytest
from scipy.io import wavfile

SR = 16000


def make_sine(freq: float = 440.0, seconds: float = 1.0, sr: int = SR) -> np.ndarray:
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def write_wav(path, wave: np.ndarray, sr: int = SR) -> None:
    """Write a fixture through the library's own writer.

    This used to be ``(wave * iinfo(int16).max).astype(int16)`` -- the exact
    wrap-around 3.7.0 fixed in `Synthesizer`. The shared noise fixture peaks at
    1.207, so twelve of its samples came back with the sign flipped, and every
    test that mixed against it was mixing against clicks it did not ask for.
    """
    from kudio import save_wave
    save_wave(path, wave, sr)


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
    # 12 dB below make_sine's own level, so that a mixture at this fixture's
    # lowest SNR still fits in the format. The noise is renormalised to hit the
    # requested SNR, so the mixture's peak is set by the CLEAN level and the
    # SNR -- quietening the noise file would change nothing. At -5 dB a 0.5
    # peak mixes to 2.68 and every test using this fixture warned about
    # clipping; the warning was right, and a warning that fires on a dozen of
    # your own tests is one nobody reads.
    for i, freq in enumerate((220, 440)):
        write_wav(clean / f"clean_{i}.wav", make_sine(freq) * 0.25)
    rng = np.random.default_rng(0)
    write_wav(noise / "white.wav",
              (0.3 * rng.standard_normal(SR)).astype(np.float32))
    return clean, noise
