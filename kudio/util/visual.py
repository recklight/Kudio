# -*- coding: utf-8 -*-
"""Plotting helpers: waveforms, spectrograms, training histories."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np

from kudio.core.feature import _clip
from kudio.core.io import copy_waves, file_load
from kudio.exceptions import DependencyError, FeatureError

# Every helper in here draws something, so matplotlib is needed to import the
# module at all rather than only to call it.
try:
    import matplotlib.pyplot as plt
except ImportError as _e:  # pragma: no cover - exercised only without [viz]
    raise DependencyError('matplotlib', extra='viz') from _e

from kudio.util import _draw  # noqa: E402

__all__ = [
    'Common',
    'WaveVisualizer',
    'history_plot',
]

log = logging.getLogger(__name__)


def _drawable(y, name: str) -> np.ndarray:
    """*y* as the one channel a plot is of, or a FeatureError: finite, and a sample
    at the least."""
    y = _clip(y, name)
    if not len(y):
        raise FeatureError(f"{name}(): the clip has no samples")
    return y


def _audio(file, name: str) -> Tuple[np.ndarray, int]:
    """A file's :func:`_drawable` samples, and their rate."""
    y, sr = file_load(file)
    return _drawable(y, name), sr


def history_plot(history: Any, metrics: str = 'loss',
                 save_fig: Optional[str] = None) -> None:
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
    """Plot Keras accuracy/loss curves side by side, saved as
    ``model_history.png`` in the folder *fig_dir* when one is given."""
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
        plt.savefig(_figure_path(fig_dir, 'model', 'history'), dpi=150)
    if show:
        plt.show()
        plt.close()


class Common:
    """Base figure wrapper with save/show/close helpers."""

    def __init__(self, fig_dir: str = 'kd_figs', is_save: bool = False):
        self.is_save = is_save
        self.fig_dir = Path(fig_dir).absolute()
        if self.is_save:
            self.fig_dir.mkdir(parents=True, exist_ok=True)

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

    def save(self, dir: Union[str, Path]) -> None:
        """Save the figure when *is_save* was given: a bare name, ``loss`` or
        ``loss.png``, in *fig_dir*, and a path with a folder in it where it
        points."""
        if self.is_save:
            target = Path(dir)
            if target.parent == Path('.'):
                target = self.fig_dir / target
            self.plt.savefig(target)

    def close(self) -> None:
        self.plt.close()

    @staticmethod
    def plot_stft_w_2mfe(f: Union[str, Path], n_mels: int = 40,
                         sub_mfe_fq: int = 48000) -> None:
        """Waveform + two mel spectrograms (full band vs. ``sub_mfe_fq``)."""
        f = Path(f)
        y, sr = _audio(f, 'plot_stft_w_2mfe')

        plt.figure(figsize=(12, 10), dpi=200)

        ax = plt.subplot(311)
        plt.title(f.stem)
        _draw.waveform(ax, y, sr)

        for place, hertz, fmax in ((312, sr, sr / 2), (313, sub_mfe_fq, sub_mfe_fq)):
            ax = plt.subplot(place)
            plt.title(f'Mel Spectrogram {hertz}Hz n:{n_mels}')
            bands = _draw.mel_level(y, sr, n_mels, fmax, 'plot_stft_w_2mfe')
            _draw.mel_spectrogram(ax, bands, sr, fmax, len(y) / sr)

        plt.tight_layout()
        plt.show()
        plt.close()


class WaveVisualizer(Common):
    """Plot waveform + spectrogram side by side for one or more files."""

    def __init__(self, fig_size: Tuple[float, float] = (7, 5), dpi: int = 200,
                 subplots: Optional[int] = None):
        super().__init__()
        self.plt = plt
        self.row_len = None
        self.wave_length = 0
        if subplots is not None:
            self.row_len = subplots
            self.fig, self.ax = plt.subplots(self.row_len, 2, figsize=(11, 8), squeeze=False)
        else:
            # as many rows as there are files, when there are
            self.fig, self.ax = self.plt.figure(figsize=fig_size, dpi=dpi), None
        self.plt.subplots_adjust(wspace=0.3, hspace=0.3)

    def plot(self, wave_list: Sequence, title: Optional[Sequence[str]] = None) -> None:
        self.wave_length = len(wave_list)
        if self.row_len is None and self.wave_length:
            self.fig.clear()
            self.ax = self.fig.subplots(self.wave_length, 2, squeeze=False)
        elif self.row_len is not None and self.wave_length > self.row_len:
            raise ValueError(f"WaveVisualizer.plot(): {self.wave_length} files do not go into "
                             f"{self.row_len} row(s); make it with subplots={self.wave_length}")
        log.info("Plotting %d wave(s)", self.wave_length)
        if title is None or len(title) != self.wave_length:
            title = [Path(w).stem for w in wave_list]

        for ind, (w, t) in enumerate(zip(wave_list, title)):
            y, sr = _audio(w, 'WaveVisualizer.plot')
            self.ax[ind][0].set_title(str(t))
            _draw.waveform(self.ax[ind][0], y, sr)

            self.ax[ind][1].set_title(str(t))
            db = _draw.level(y, 512, 256, 'hamming', who='WaveVisualizer.plot')
            _draw.linear_spectrogram(self.ax[ind][1], db, sr, 256, len(y) / sr, cmap='jet')
        self.multi_hide_xlabel()
        self.fig.tight_layout()

    def multi_hide_xlabel(self) -> None:
        for idx in range(self.wave_length - 1):
            self.ax[idx, 0].set_xlabel("")
            self.ax[idx, 1].set_xlabel("")

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        if path.exists():
            log.warning("File or path already exists: %s", path)
            return
        self.plt.savefig(path)

    def copy_wave(self, path: Union[str, Path], wave_list: Any) -> None:
        copy_waves(path, wave_list)


def _figure_path(save_dir, file, kind: str) -> Path:
    """Where a plot of *file* is saved: in the folder *save_dir*, made if it
    is missing, as ``<file stem>_<kind>.png``, so plotting several files or
    several kinds into one folder keeps every figure. A *save_dir* ending in
    a format matplotlib writes (``figs/clip.png``, ``.pdf``, ``.svg``) is
    the file itself."""
    target = Path(save_dir)
    formats = plt.gcf().canvas.get_supported_filetypes()
    if not target.is_dir() and target.suffix[1:].lower() in formats:
        target.parent.mkdir(parents=True, exist_ok=True)
        return target
    target.mkdir(parents=True, exist_ok=True)
    return target / f"{Path(file).stem}_{kind}.png"


def basic_analyze(file, show: bool = False, save_dir=None) -> None:
    """Waveform / STFT / zero-crossing-rate overview of a file, saved as
    ``<stem>_overview.png`` in the folder *save_dir* when one is given."""
    x, sr = _audio(file, 'basic_analyze')

    plt.figure(figsize=(10, 8), dpi=200)
    ax = plt.subplot(311)
    plt.title(Path(file).stem)
    _draw.waveform(ax, x, sr)

    ax = plt.subplot(312)
    plt.title('STFT')
    _draw.linear_spectrogram(ax, _draw.level(x, ref=1.0, who='basic_analyze'), sr, _draw.HOP,
                             len(x) / sr, cmap='jet')

    ax = plt.subplot(313)
    plot_zcr(ax, x, sr, color=None)
    ax.set_xlim(0, len(x) / sr)
    ax.set_ylim(bottom=0)

    plt.tight_layout()

    if save_dir is not None:
        plt.savefig(_figure_path(save_dir, file, "overview"))
    if show:
        plt.show()
    plt.close()


def plot_4_features(file, show: bool = False, save_dir=None, dpi: int = 200) -> None:
    """Waveform + spectral centroid/rolloff/ZCR and power spectral density,
    saved as ``<stem>_4_features.png`` in the folder *save_dir*."""
    x, sr = _audio(file, 'plot_4_features')

    fig, ax = plt.subplots(2, 1, figsize=(10, 10), dpi=dpi)

    _draw.waveform(ax[0], x, sr, alpha=0.4, label='waveform')
    plot_spectral_centroid(ax[0], x, sr, color='r', label='spectral_centroid')
    plot_spectral_rolloff(ax[0], x, sr, color='b', label='spectral_rolloff')
    plot_zcr(ax[0], x, sr, color='g', label='zcr')
    plot_power_spectral_density(ax[1], x, sr)

    ax[0].set_xlim(0, len(x) / sr)
    ax[0].grid(which='major', axis='both')
    ax[0].set_title('spectral centroid & spectral rolloff & zcr')

    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(_figure_path(save_dir, file, "4_features"))
    if show:
        plt.show()
    plt.close()


def plot_3_features(file, show: bool = False, save_dir=None, dpi: int = 200) -> None:
    """Waveform + spectral centroid/rolloff/ZCR, saved as
    ``<stem>_3_features.png`` in the folder *save_dir*."""
    x, sr = _audio(file, 'plot_3_features')

    fig, ax = plt.subplots(figsize=(7, 5), dpi=dpi)

    _draw.waveform(ax, x, sr, alpha=0.4, label='waveform')
    plot_spectral_centroid(ax, x, sr, color='r', label='spectral_centroid')
    plot_spectral_rolloff(ax, x, sr, color='b', label='spectral_rolloff')
    plot_zcr(ax, x, sr, color='g', label='zcr')

    ax.set_xlim(0, len(x) / sr)
    ax.grid(which='major', axis='both')
    ax.set_title('spectral centroid & spectral rolloff & zcr')

    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(_figure_path(save_dir, file, "3_features"))
    if show:
        plt.show()
    plt.close()


def plot_PSD(file, show: bool = False, save_dir=None, dpi: int = 200) -> None:
    """Power spectral density of a file, saved as ``<stem>_psd.png`` in the
    folder *save_dir*."""
    x, sr = _audio(file, 'plot_PSD')

    fig, ax = plt.subplots(figsize=(7, 5), dpi=dpi)
    plot_power_spectral_density(ax, x, sr)

    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(_figure_path(save_dir, file, "psd"))
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
    values = _draw.centroid(_drawable(y, 'plot_spectral_centroid'), sr, 'plot_spectral_centroid')
    ax.plot(_draw.frame_times(len(values), sr), normalize(values), color=color, label=label)
    ax.set_title('Spectral Centroid')
    if label is not None:
        ax.legend()


def plot_spectral_rolloff(ax, y, sr, color='r', label=None) -> None:
    values = _draw.rolloff(_drawable(y, 'plot_spectral_rolloff'), sr, 'plot_spectral_rolloff')
    ax.plot(_draw.frame_times(len(values), sr), normalize(values), color=color, label=label)
    ax.set_title('Spectral Rolloff')
    if label is not None:
        ax.legend()


def plot_power_spectral_density(ax, y, sr, color='b', label=None) -> None:
    from scipy.signal import welch
    freqs, psd = welch(_drawable(y, 'plot_power_spectral_density'), sr, nfft=1024)
    ax.semilogx(freqs, psd, color=color, label=label)
    ax.set_title('Power Spectral Density')
    ax.set_xlabel('Frequency[Hz]')
    ax.set_ylabel('PSD [V**2/Hz]')
    ax.grid(which='major', axis='both')
    if label is not None:
        ax.legend()


def plot_zcr(ax, y, sr, color='r', label=None) -> None:
    # a silence of zeros and least bits is on one side
    values = _draw.crossings(_drawable(y, 'plot_zcr') + 0.0001)
    ax.set_title('ZCR')
    ax.plot(_draw.frame_times(len(values), sr), values, color=color, label=label)
    if label is not None:
        ax.legend()
