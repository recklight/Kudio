# -*- coding: utf-8 -*-
"""Feature extraction and spectrogram <-> waveform conversions.

v3 canonical names
------------------
=================================  =============================================
canonical                          deprecated aliases
=================================  =============================================
``waveform_to_spectrogram``        ``w2s``, ``wavform2spec``
``spectrogram_to_waveform``        ``spec2wavform``
``file_to_spectrogram``            ``f2s``, ``wav2spec``
``save_spectrogram_as_wave``       ``s2w``, ``spec2wav``
``mfcc``                           ``w2mfcc``, ``wav2mfcc``
``mfcc_from_files``                ``concat_mfcc``, ``concat_mfcc_``
``logspec_from_files``             ``concat_logspec``, ``concat_logspec_``,
                                   ``contextual_LogSpectrogram``
``melspectrogram``                 ``wav2mel``
=================================  =============================================

The aliases still work but emit a ``DeprecationWarning``.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional, Sequence, Tuple

import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm

from kudio._deprecation import alias
from kudio.core.io import file_load
from kudio.exceptions import FeatureError

__all__ = [
    # canonical
    'wave_separate', 'wave_slicing', 'features2matrix',
    'waveform_to_spectrogram', 'spectrogram_to_waveform',
    'file_to_spectrogram', 'save_spectrogram_as_wave',
    'mfcc', 'mfcc_from_files', 'logspec_from_files', 'melspectrogram',
    'denoise_wav2spec', 'denoise_spec2wav',
    # deprecated aliases (kept for compatibility)
    'w2s', 'wavform2spec', 'spec2wavform', 'f2s', 'wav2spec', 's2w', 'spec2wav',
    'w2mfcc', 'wav2mfcc', 'concat_mfcc', 'concat_mfcc_', 'concat_logspec',
    'concat_logspec_', 'contextual_LogSpectrogram', 'wav2mel',
]

log = logging.getLogger(__name__)


def wave_separate(wave: np.ndarray, channels: int) -> list:
    """Split an interleaved multi-channel waveform into per-channel arrays."""
    if channels < 1:
        raise FeatureError(f"channels must be >= 1, got {channels}")
    return [wave[c::channels] for c in range(channels)]


def waveform_to_spectrogram(y: np.ndarray,
                            sequence: bool = False,
                            forward_backward: int = 0,
                            norm: bool = False,
                            hop_length: int = 256,
                            n_fft: int = 512,
                            win_length: int = 512,
                            window: str = 'hamming',
                            desired_length: Optional[int] = None) -> np.ndarray:
    """Waveform -> log-power spectrogram.

    Parameters
    ----------
    y : waveform, shape ``(n,)``
    sequence : return a 3-D time-series array ``(1, frames, dim)``
    forward_backward : if > 0, stack that many context frames on each side
    norm : per-frequency-bin mean/std normalization
    desired_length : truncate the waveform to this many samples first
    """
    if desired_length and desired_length < len(y):
        y = y[:desired_length]
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                     win_length=win_length, window=window)
    D = D + np.finfo(float).eps
    Sxx = np.log10(np.abs(D) ** 2)

    if norm:
        Sxx_mean = np.mean(Sxx, axis=1, keepdims=True)
        Sxx_std = np.std(Sxx, axis=1, keepdims=True) + 1e-12
        Sxx_r = (Sxx - Sxx_mean) / Sxx_std
    else:
        Sxx_r = np.array(Sxx)

    if forward_backward:
        Sxx_r = Sxx_r.T
        frames, dim = Sxx_r.shape
        return_data = np.empty(
            (frames + 50, int(forward_backward * 2) + 1, n_fft // 2 + 1),
            dtype=np.float32)
        for idx in range(frames):
            idx_start = idx - forward_backward
            idx_end = idx + forward_backward
            if idx_start < 0:
                null = np.zeros((-idx_start, dim))
                tmp = np.concatenate((null, Sxx_r[0:idx_end + 1]), axis=0)
            elif idx_end > frames - 1:
                null = np.zeros((idx_end - frames + 1, dim))
                tmp = np.concatenate((Sxx_r[idx_start:], null), axis=0)
            else:
                tmp = Sxx_r[idx_start:idx_end + 1]
            return_data[idx] = tmp
        shape = return_data.shape
        if sequence:
            return return_data.reshape(1, shape[0], shape[1] * shape[2])[:, :frames]
        return return_data.reshape(shape[0], shape[1] * shape[2])[:frames]

    Sxx_r = np.array(Sxx_r).T
    if sequence:
        return Sxx_r.reshape(1, *Sxx_r.shape)
    return Sxx_r


def spectrogram_to_waveform(y: np.ndarray, enhanced_spec: np.ndarray,
                            sequence: bool = False, hop_length: int = 256,
                            n_fft: int = 512, win_length: int = 512,
                            window: str = 'hamming') -> np.ndarray:
    """Log-power spectrogram -> waveform, reusing the phase of reference *y*."""
    if sequence:
        enhanced_spec = enhanced_spec.squeeze()
    D = librosa.stft(y.astype('float32'), hop_length=hop_length,
                     n_fft=n_fft, win_length=win_length, window=window)
    D = D + np.finfo(float).eps
    phase = np.exp(1j * np.angle(D))
    magnitude = np.sqrt(10 ** np.array(enhanced_spec)).T
    reverse = np.multiply(magnitude, phase)
    result = librosa.istft(reverse, hop_length=hop_length,
                           win_length=win_length, window=window)
    return librosa.util.fix_length(result, size=len(y), mode='edge')


def file_to_spectrogram(file,
                        sequence: bool = False,
                        forward_backward: int = 0,
                        norm: bool = False,
                        hop_length: int = 256,
                        n_fft: int = 512,
                        win_length: int = 512,
                        desired_length: Optional[int] = None,
                        resample_rate: Optional[int] = None) -> np.ndarray:
    """File -> log-power spectrogram (see :func:`waveform_to_spectrogram`)."""
    y, _ = file_load(file, sr=resample_rate, mono=True)
    return waveform_to_spectrogram(y, sequence, forward_backward, norm,
                                   hop_length, n_fft, win_length,
                                   desired_length=desired_length)


def save_spectrogram_as_wave(wave_out_dir, noisy_file, enhanced_spec,
                             squeeze: bool = False, hop_length: int = 256,
                             n_fft: int = 512, win_length: int = 512,
                             window: str = 'hamming') -> None:
    """Reconstruct a waveform from *enhanced_spec* (phase from *noisy_file*)
    and write it to *wave_out_dir* as 16-bit PCM."""
    y, rate = librosa.load(str(noisy_file), sr=None)
    y_out = spectrogram_to_waveform(y, enhanced_spec, squeeze, hop_length,
                                    n_fft, win_length, window)
    Path(wave_out_dir).parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(str(wave_out_dir), rate,
                  (y_out * np.iinfo(np.int16).max).astype(np.int16))


def mfcc(y: np.ndarray, sr: int, n_mfcc: int = 40) -> np.ndarray:
    """Waveform -> MFCC, shaped ``(1, n_frames, n_mfcc)``."""
    return librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc).T.reshape(1, -1, n_mfcc)


def mfcc_from_files(wav_list: Sequence, n_mfcc: int = 40,
                    desired_samples: Optional[int] = None,
                    desired_length: Optional[int] = None) -> np.ndarray:
    """Load each wave file and stack their MFCC matrices ``(n_files, frames, n_mfcc)``."""
    mfcc_list = []
    for wav_dir in tqdm(wav_list, desc="[kudio] MFCC"):
        y, sr = file_load(wav_dir)
        if desired_length:
            y = y[:sr * desired_length]
        if desired_samples and desired_samples < len(y):
            y = y[:desired_samples]
        m = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc).T
        mfcc_list.append(m.reshape(1, m.shape[0], -1))
    return np.vstack(mfcc_list)


def logspec_from_files(wav_list: Sequence,
                       sequence: bool,
                       forward_backward: int,
                       norm: bool,
                       hop_length: int = 256,
                       n_fft: int = 512,
                       win_length: int = 512,
                       desired_samples: Optional[int] = None) -> np.ndarray:
    """Convert each file to a log spectrogram and stack them."""
    return np.vstack([
        file_to_spectrogram(file=file, sequence=sequence,
                            forward_backward=forward_backward, norm=norm,
                            hop_length=hop_length, n_fft=n_fft,
                            win_length=win_length, desired_length=desired_samples)
        for file in tqdm(wav_list, desc='[kudio] log-spectrograms')
    ])


def melspectrogram(file_name, n_mels: int = 64, n_frames: int = 5,
                   n_fft: int = 1024, hop_length: int = 512,
                   power: float = 2.0) -> np.ndarray:
    """File -> stacked log-mel feature vectors ``(n_vectors, n_mels * n_frames)``."""
    dims = n_mels * n_frames
    y, sr = file_load(file_name, mono=True)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft,
                                         hop_length=hop_length, n_mels=n_mels,
                                         power=power)
    log_mel = 20.0 / power * np.log10(np.maximum(mel, sys.float_info.epsilon))
    n_vectors = log_mel.shape[1] - n_frames + 1
    if n_vectors < 1:
        return np.empty((0, dims))
    vectors = np.zeros((n_vectors, dims))
    for t in range(n_frames):
        vectors[:, n_mels * t: n_mels * (t + 1)] = log_mel[:, t: t + n_vectors].T
    return vectors


def denoise_wav2spec(waveform: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Peak-normalized magnitude spectrogram + phase, for denoising models."""
    D = librosa.stft(waveform / np.max(np.abs(waveform)),
                     n_fft=512, hop_length=256, win_length=512, window='hann')
    D += np.finfo(float).eps
    Sxx = np.abs(D)
    phase = np.angle(D)
    Sxx_mean = np.mean(Sxx, axis=1, keepdims=True)
    Sxx_std = np.std(Sxx, axis=1, keepdims=True) + 1e-12
    Sxx_r = (Sxx - Sxx_mean) / Sxx_std
    return np.reshape(Sxx_r.T, (1, Sxx_r.shape[1], Sxx_r.shape[0])), phase


def denoise_spec2wav(spec: np.ndarray, phase: np.ndarray) -> np.ndarray:
    """Inverse of :func:`denoise_wav2spec`; output peak-normalized."""
    spec = np.squeeze(spec).T
    pred_wav = librosa.istft(np.multiply(spec, np.exp(1j * phase)),
                             hop_length=256, win_length=512, window='hann')
    return pred_wav / np.max(np.abs(pred_wav))


def wave_slicing(y: np.ndarray, frame_length: int = 2048,
                 hop_length: int = 512, is_reshape: bool = True) -> np.ndarray:
    """Slice a waveform into overlapping frames (flattened if *is_reshape*)."""
    frame_stack = librosa.util.frame(y, frame_length=frame_length,
                                     hop_length=hop_length)
    if is_reshape:
        frame_stack = frame_stack.T.reshape(-1)
    return frame_stack


def features2matrix(features: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Concatenate feature matrices; label rows by list index."""
    feature_matrix = np.vstack(features)
    labels = np.concatenate([i * np.ones(len(f)) for i, f in enumerate(features)])
    return feature_matrix, labels


# -- deprecated aliases (v2 names) ----------------------------------------------
w2s = alias(waveform_to_spectrogram, 'w2s')
wavform2spec = alias(waveform_to_spectrogram, 'wavform2spec')
spec2wavform = alias(spectrogram_to_waveform, 'spec2wavform')
f2s = alias(file_to_spectrogram, 'f2s')
wav2spec = alias(file_to_spectrogram, 'wav2spec')
s2w = alias(save_spectrogram_as_wave, 's2w')
spec2wav = alias(save_spectrogram_as_wave, 'spec2wav')
w2mfcc = alias(mfcc, 'w2mfcc')
wav2mfcc = alias(mfcc, 'wav2mfcc')
concat_mfcc = alias(mfcc_from_files, 'concat_mfcc')
concat_mfcc_ = alias(mfcc_from_files, 'concat_mfcc_')
concat_logspec = alias(logspec_from_files, 'concat_logspec')
concat_logspec_ = alias(logspec_from_files, 'concat_logspec_')
contextual_LogSpectrogram = alias(logspec_from_files, 'contextual_LogSpectrogram')
wav2mel = alias(melspectrogram, 'wav2mel')
