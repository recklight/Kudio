# -*- coding: utf-8 -*-
"""Plotting helpers: waveforms, spectrograms, training histories."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

import librosa
import librosa.display
import numpy as np
from scipy import signal

from kudio.core.io import copy_waves, file_load
from kudio.exceptions import DependencyError

try:
    import matplotlib.pyplot as plt
except ImportError as _e:  # pragma: no cover - exercised only without [viz]
    raise DependencyError('matplotlib', extra='viz') from _e

__all__ = [
    'Common',
    'WaveVisualizer',
    'history_plot',
]

log = logging.getLogger(__name__)


def history_plot(history, metrics: str = 'loss', save_fig: Optional[str] = None):
    """Plot every history curve whose key contains *metrics* (Keras History)."""
    for_legend = []
    color = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'b--', 'r:', 'c:', 'm:', 'k:', 'g--', 'y--']
    for index, key in enumerate(history.history.keys()):
        if metrics in key:
            plt.plot(history.history[key], color[index % len(color)], label=key)
            for_legend.append(key)
    plt.legend(for_legend, loc='upper left', fontsize='small')
    plt.title(metrics)
    plt.xlabel('Epoch')
    plt.ylabel(metrics)
    plt.grid(True)
    if save_fig is not None:
        save_fig = Path(save_fig)
        save_fig.parent.mkdir(parents=True, exist_ok=True)
        if save_fig.suffix != '.png':
            save_fig = save_fig.with_suffix('.png')
        plt.savefig(save_fig.parent / f"{metrics}_{save_fig.name}", dpi=150)
    plt.show()
    plt.close()


def hist_plt(hist, fig_dir=None, show: bool = False):
    """Plot Keras accuracy/loss curves side by side."""
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(hist.history['accuracy'])
    plt.plot(hist.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.subplot(1, 2, 2)
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout()
    if fig_dir is not None:
        plt.savefig(str(fig_dir), dpi=150)
    if show:
        plt.show()
        plt.close()


class Common:
    """Base figure wrapper with save/show/close helpers."""

    def __init__(self, fig_dir: str = 'kd_figs', is_save: bool = False):
        self.is_save = is_save
        if self.is_save:
            fig_dir = Path(fig_dir).absolute()
            fig_dir.mkdir(parents=True, exist_ok=True)

        self.plt = plt
        self.fig = self.plt.figure(figsize=(7, 5))
        self.plt.subplots_adjust(wspace=0.3, hspace=0.3)

    def loss_plot(self, loss: Sequence[float], val_loss: Sequence[float]) -> None:
        ax = self.fig.add_subplot(1, 1, 1)
        ax.cla()
        ax.plot(loss)
        ax.plot(val_loss)
        ax.set_title("Model loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(["Train", "Validation"], loc="upper right")

    def show(self) -> None:
        self.fig.show()

    def save(self, dir) -> None:
        if self.is_save:
            self.plt.savefig(dir)

    def close(self) -> None:
        self.plt.close()

    @staticmethod
    def plot_stft_w_2mfe(f, n_mels: int = 40, sub_mfe_fq: int = 48000) -> None:
        """Waveform + two mel spectrograms (full band vs. ``sub_mfe_fq``)."""
        f = Path(f)
        y, sr = librosa.load(str(f), sr=None, mono=True)

        plt.figure(figsize=(12, 10), dpi=200)

        plt.subplot(311)
        plt.title(f.stem)
        librosa.display.waveshow(y, sr=sr)

        plt.subplot(312)
        plt.title(f'Mel Spectrogram {sr}Hz n:{n_mels}')
        mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        librosa.display.specshow(librosa.power_to_db(mel_spect, ref=np.max),
                                 sr=sr, x_axis='time', y_axis='mel', fmax=sr // 2)

        plt.subplot(313)
        plt.title(f'Mel Spectrogram {sub_mfe_fq}Hz n:{n_mels}')
        mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels,
                                                   fmax=sub_mfe_fq)
        librosa.display.specshow(librosa.power_to_db(mel_spect, ref=np.max),
                                 sr=sr, x_axis='time', y_axis='mel', fmax=sub_mfe_fq)

        plt.tight_layout()
        plt.show()
        plt.close()


class WaveVisualizer(Common):
    """Plot waveform + spectrogram side by side for one or more files."""

    def __init__(self, fig_size=(7, 5), dpi: int = 200, subplots: Optional[int] = None):
        super().__init__()
        self.plt = plt
        self.row_len = None
        self.wave_length = 0
        if subplots is not None:
            self.row_len = subplots
            self.fig, self.ax = plt.subplots(self.row_len, 2, figsize=(11, 8))
        else:
            self.fig = self.plt.figure(figsize=fig_size, dpi=dpi)
        self.plt.subplots_adjust(wspace=0.3, hspace=0.3)

    def plot(self, wave_list: Sequence, title: Optional[Sequence[str]] = None) -> None:
        self.wave_length = len(wave_list)
        log.info("Plotting %d wave(s)", self.wave_length)
        if title is None or len(title) != self.wave_length:
            title = [Path(w).stem for w in wave_list]

        for ind, (w, t) in enumerate(zip(wave_list, title)):
            self.ax[ind][0].set_title(str(t))
            y, sr = file_load(w)
            librosa.display.waveshow(y, sr=sr, ax=self.ax[ind][0])

            self.ax[ind][1].set_title(str(t))
            DA = librosa.amplitude_to_db(
                np.abs(librosa.stft(y, n_fft=512, hop_length=256, win_length=512,
                                    window='hamming')), ref=np.max)
            librosa.display.specshow(DA, sr=sr, hop_length=256,
                                     x_axis='time', y_axis='linear',
                                     cmap='jet', ax=self.ax[ind][1])
        self.multi_hide_xlabel()
        self.fig.tight_layout()

    def multi_hide_xlabel(self) -> None:
        for idx in range(self.wave_length - 1):
            self.ax[idx, 0].set_xlabel("")
            self.ax[idx, 1].set_xlabel("")

    def save(self, path) -> None:
        path = Path(path)
        if path.exists():
            log.warning("File or path already exists: %s", path)
            return
        self.plt.savefig(path)

    def copy_wave(self, path, wave_list) -> None:
        copy_waves(path, wave_list)


def basic_analyze(file, show: bool = False, save_dir=None) -> None:
    """Waveform / STFT / zero-crossing-rate overview of a file."""
    x, sr = file_load(file)

    plt.figure(figsize=(10, 8), dpi=200)
    plt.subplot(311)
    plt.title(Path(file).stem)
    librosa.display.waveshow(x, sr=sr)

    X = librosa.stft(x)
    Xdb = librosa.amplitude_to_db(abs(X))
    plt.subplot(312)
    plt.title('STFT')
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz', cmap='jet')

    plt.subplot(313)
    plt.title('ZCR')
    zcrs = librosa.feature.zero_crossing_rate(x + 0.0001)
    time2 = np.linspace(0, int(len(x) / sr), len(zcrs[0]))
    plt.plot(time2, zcrs[0])
    plt.axis([0, int(len(x) / sr), 0, max(zcrs[0])])

    plt.tight_layout()

    if save_dir is not None:
        plt.savefig(str(save_dir))
    if show:
        plt.show()
    plt.close()


def plot_4_features(file, show: bool = False, save_dir=None, dpi: int = 200) -> None:
    """Waveform + spectral centroid/rolloff/ZCR and power spectral density."""
    x, sr = file_load(file)

    fig, ax = plt.subplots(2, 1, figsize=(10, 10), dpi=dpi)

    librosa.display.waveshow(x, sr=sr, alpha=0.4, ax=ax[0], label='waveform')
    plot_spectral_centroid(ax[0], x, sr, color='r', label='spectral_centroid')
    plot_spectral_rolloff(ax[0], x, sr, color='b', label='spectral_rolloff')
    plot_zcr(ax[0], x, sr, color='g', label='zcr')
    plot_power_spectral_density(ax[1], x, sr)

    ax[0].set_xlim(0, int(len(x) / sr))
    ax[0].grid(which='major', axis='both')
    ax[0].set_title('spectral centroid & spectral rolloff & zcr')

    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(save_dir)
    if show:
        plt.show()
    plt.close()


def plot_3_features(file, show: bool = False, save_dir=None, dpi: int = 200) -> None:
    """Waveform + spectral centroid/rolloff/ZCR."""
    x, sr = file_load(file)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=dpi)

    librosa.display.waveshow(x, sr=sr, alpha=0.4, ax=ax, label='waveform')
    plot_spectral_centroid(ax, x, sr, color='r', label='spectral_centroid')
    plot_spectral_rolloff(ax, x, sr, color='b', label='spectral_rolloff')
    plot_zcr(ax, x, sr, color='g', label='zcr')

    ax.set_xlim(0, int(len(x) / sr))
    ax.grid(which='major', axis='both')
    ax.set_title('spectral centroid & spectral rolloff & zcr')

    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(save_dir)
    if show:
        plt.show()
    plt.close()


def plot_PSD(file, show: bool = False, save_dir=None, dpi: int = 200) -> None:
    """Power spectral density of a file."""
    x, sr = file_load(file)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=dpi)
    plot_power_spectral_density(ax, x, sr)

    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(save_dir)
    if show:
        plt.show()
    plt.close()


def normalize(x, axis: int = 0):
    """Min-max scale to [0, 1] along *axis*."""
    x = np.asarray(x, dtype=float)
    x_min = x.min(axis=axis, keepdims=True)
    x_range = x.max(axis=axis, keepdims=True) - x_min
    x_range[x_range == 0] = 1.0
    return (x - x_min) / x_range


def plot_spectral_centroid(ax, y, sr, color='r', label=None) -> None:
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    t = np.linspace(0, int(len(y) / sr), len(spectral_centroids))
    ax.plot(t, normalize(spectral_centroids), color=color, label=label)
    ax.set_title('Spectral Centroid')
    if label is not None:
        ax.legend()


def plot_spectral_rolloff(ax, y, sr, color='r', label=None) -> None:
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    t = np.linspace(0, int(len(y) / sr), len(spectral_rolloff))
    ax.plot(t, normalize(spectral_rolloff), color=color, label=label)
    ax.set_title('Spectral Rolloff')
    if label is not None:
        ax.legend()


def plot_power_spectral_density(ax, y, sr, color='b', label=None) -> None:
    freqs, psd = signal.welch(y, sr, nfft=1024)
    ax.semilogx(freqs, psd, color=color, label=label)
    ax.set_title('Power Spectral Density')
    ax.set_xlabel('Frequency[Hz]')
    ax.set_ylabel('PSD [V**2/Hz]')
    ax.grid(which='major', axis='both')
    if label is not None:
        ax.legend()


def plot_zcr(ax, y, sr, color='r', label=None) -> None:
    ax.set_title('ZCR')
    zcrs = librosa.feature.zero_crossing_rate(y + 0.0001)
    time2 = np.linspace(0, int(len(y) / sr), len(zcrs[0]))
    ax.plot(time2, zcrs[0], color=color, label=label)
    if label is not None:
        ax.legend()
