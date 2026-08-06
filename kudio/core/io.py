# -*- coding: utf-8 -*-
"""Audio file I/O: path resolution, loading and saving of wave files."""
from __future__ import annotations

import logging
import re
import shutil
from multiprocessing import Pool
from pathlib import Path, PurePath
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import soundfile as sf
from tqdm import tqdm

from kudio.exceptions import AudioIOError

__all__ = [
    'check_path',
    'check_file',
    'check_input',
    'save_wave',
    'file_load',
    'load_wave',
    'load_waves',
    'copy_waves',
    'LoadAudio',
]

log = logging.getLogger(__name__)

WAVE_SUFFIX = '.wav'
PathLike = Union[str, PurePath]


def check_path(path: PathLike, exist_ok: bool = False, sep: str = '') -> str:
    """Return ``path`` if it is free (or ``exist_ok``), otherwise an incremented
    variant, e.g. ``runs/exp`` -> ``runs/exp2``, ``runs/exp3`` ...
    """
    path = Path(path)
    if exist_ok or not path.exists():
        return str(path)
    siblings = path.parent.glob(f"{path.name}{sep}*") if path.parent.exists() else []
    matches = (re.fullmatch(rf"{re.escape(path.name)}{re.escape(sep)}(\d+)", p.name)
               for p in siblings)
    indices = [int(m.group(1)) for m in matches if m]
    n = max(indices) + 1 if indices else 2
    return f"{path}{sep}{n}"


def check_file(file_dir: PathLike, rename: bool = True, sep: str = '') -> str:
    """Normalize an output wave path: force the ``.wav`` suffix, create parent
    directories, and (optionally) rename to avoid overwriting an existing file.
    """
    file_dir = Path(file_dir)
    if file_dir.is_dir():
        file_dir = file_dir / 'unknown'
    file_dir = file_dir.with_suffix(WAVE_SUFFIX)
    if file_dir.is_file():
        log.info("File already exists: %s", file_dir)
        if rename:
            pattern = rf"{re.escape(file_dir.stem)}{re.escape(sep)}(\d+)"
            existing = file_dir.parent.glob(f"{file_dir.stem}{sep}*")
            indices = [int(m.group(1)) for m in
                       (re.fullmatch(pattern, p.stem) for p in existing) if m]
            n = max(indices) + 1 if indices else 2
            file_dir = (file_dir.parent / f"{file_dir.stem}{sep}{n}").with_suffix(WAVE_SUFFIX)
            log.info("Renamed to: %s", file_dir)
        else:
            log.warning("Existing file will be overwritten: %s", file_dir)
    file_dir.parent.mkdir(exist_ok=True, parents=True)
    return str(file_dir)


def check_input(input: Any) -> Tuple[List[Path], List[Any]]:
    """Recursively collect ``.wav`` files from *input*.

    ``input`` may be a file path, a directory, a ``.txt`` file containing one
    path per line, or a (nested) list/tuple of any of those.

    Returns ``(right_list, wrong_list)`` where ``right_list`` holds resolved
    ``.wav`` paths and ``wrong_list`` everything that could not be resolved.
    """
    right_list: List[Path] = []
    wrong_list: List[Any] = []
    if isinstance(input, (list, tuple)):
        for item in input:
            r, w = check_input(item)
            right_list.extend(r)
            wrong_list.extend(w)
    elif input and isinstance(input, (str, PurePath)):
        input = Path(input)
        if input.is_file():
            if input.suffix == '.txt':
                lines = [x.strip() for x in input.read_text().strip().splitlines() if x.strip()]
                for item in lines:
                    r, w = check_input(item)
                    right_list.extend(r)
                    wrong_list.extend(w)
            elif input.suffix == WAVE_SUFFIX:
                right_list.append(input)
            else:
                wrong_list.append(str(input))
        elif input.is_dir():
            right_list.extend(sorted(input.rglob(f"*{WAVE_SUFFIX}")))
        else:
            wrong_list.append(str(input))
    else:
        wrong_list.append(input)
    return right_list, wrong_list


def file_load(wav_dir: PathLike, sr: Optional[int] = None, mono: bool = True
              ) -> Tuple[np.ndarray, int]:
    """Load a wave file; returns ``(waveform float32, sample_rate)``.

    ``sr=None`` keeps the file's native sample rate and reads via soundfile
    (fast, no librosa/numba needed). A requested ``sr`` triggers a resample
    through librosa.
    """
    try:
        if sr is None:
            y, rate = sf.read(str(wav_dir), dtype='float32', always_2d=False)
            if mono and y.ndim > 1:
                y = y.mean(axis=1).astype(np.float32)
            return np.asarray(y, dtype=np.float32), int(rate)
        # resampling path (heavier)
        import librosa
        y, rate = librosa.load(str(wav_dir), sr=sr, mono=mono)
        return np.asarray(y, dtype=np.float32), int(rate)
    except Exception as e:
        raise AudioIOError(f"Could not load audio file: {wav_dir} ({e})") from e


def resample(y: np.ndarray, orig_sr: int, target_sr: int,
             res_type: str = 'soxr_hq') -> np.ndarray:
    """Resample a waveform already in memory.

    The in-memory counterpart of ``file_load(..., sr=target_sr)``, which can
    only resample on the way in from disk. Returns *y* untouched when the rates
    already match, so it is safe to call unconditionally.

    >>> y16 = kudio.resample(y8, orig_sr=8000, target_sr=16000)
    """
    if orig_sr <= 0 or target_sr <= 0:
        raise ValueError(f"sample rates must be > 0, got {orig_sr} -> {target_sr}")
    y = np.asarray(y)
    if orig_sr == target_sr or y.size == 0:
        return y
    import librosa
    out = librosa.resample(y.astype(np.float32), orig_sr=orig_sr,
                           target_sr=target_sr, res_type=res_type)
    return out.astype(y.dtype, copy=False)


def load_wave(wave, num: Optional[int] = None):
    """Load one wave file or a list of them into ``(waveform, sr)`` tuples."""
    info = wave if num is None else wave[num]
    if isinstance(info, (list, tuple)):
        return [load_wave(w) for w in tqdm(info, desc="[kudio] Loading audios")]
    if isinstance(info, (str, PurePath)) and Path(info).is_file() \
            and Path(info).suffix == WAVE_SUFFIX:
        return file_load(info)
    raise ValueError(f"Unexpected wave input: {info!r}")


def load_waves(wave, use_pool: bool = False, std_len: bool = False, core_: int = 8):
    """Convert every ``.wav`` in *wave* (see :func:`check_input`) to waveform data.

    Returns a list of ``(np.ndarray, int)``. With ``std_len=True`` returns
    ``(waves_2d, sr, min_len)`` where every waveform is truncated to the
    shortest length (all files must share one sample rate).
    """
    waves, _ = check_input(wave)
    if use_pool and len(waves) > 300:
        with Pool(core_) as pool:
            result = list(tqdm(pool.imap(file_load, map(str, waves)),
                               total=len(waves), desc="[kudio] Loading audios (pool)"))
    else:
        result = load_wave(waves)

    if std_len:
        w2d = [y for y, _ in result]
        srs = {sr for _, sr in result}
        if len(srs) != 1:
            raise ValueError(f"Different audio sample rates found: {sorted(srs)}")
        min_len = min(len(y) for y in w2d)
        w2d = [y[:min_len] for y in w2d]
        return w2d, srs.pop(), min_len

    return result


def save_wave(path: PathLike, wave: np.ndarray, sr: int,
              d_sample: Optional[int] = None, subtype: str = 'PCM_16') -> None:
    """Write *wave* to a ``.wav`` file via soundfile.

    :param subtype: soundfile subtype — ``'PCM_16'`` (default), ``'PCM_24'``,
        ``'FLOAT'`` (32-bit float), ... see ``soundfile.available_subtypes()``.
    :param d_sample: optional resample rate before writing.

    int16 input is normalized to float [-1, 1]; float input is written as-is.
    """
    Path(path).parent.mkdir(exist_ok=True, parents=True)
    wave = np.asarray(wave)

    if np.issubdtype(wave.dtype, np.integer):
        wave = wave.astype(np.float32) / np.iinfo(np.int16).max
    else:
        wave = wave.astype(np.float32)

    if d_sample is not None and d_sample != sr:
        import librosa
        wave = librosa.resample(wave, orig_sr=sr, target_sr=d_sample)
        sr = d_sample

    try:
        sf.write(str(path), wave, sr, subtype=subtype)
    except Exception as e:
        raise AudioIOError(f"Could not write audio file: {path} ({e})") from e


def copy_waves(path: PathLike, wave_) -> None:
    """Copy one wave file (or a list of them) into directory *path*."""
    path = Path(path)
    if path.suffix:
        log.warning("Destination is not a directory: %s", path)
        return
    path.mkdir(exist_ok=True, parents=True)

    if isinstance(wave_, (list, tuple)):
        for w in tqdm(wave_, desc=f"[kudio] Copying waves to {path.resolve()}"):
            copy_waves(path, w)
    else:
        w = Path(wave_)
        shutil.copy(str(w), path / w.name)


class LoadAudio:
    """Iterate over the ``.wav`` files found (recursively) in a directory."""

    def __init__(self, waves_dirs: PathLike):
        self.sr_pth = Path(waves_dirs)
        if not self.sr_pth.is_dir():
            raise NotADirectoryError(f"Folder not found: {self.sr_pth.absolute()}")
        self.files = sorted(self.sr_pth.rglob(f"*{WAVE_SUFFIX}"))
        self.nf = len(self.files)

    def __iter__(self):
        return iter(self.files)

    def __len__(self) -> int:
        return self.nf

    check_input = staticmethod(check_input)
    file_load = staticmethod(file_load)
    load_wave = staticmethod(load_wave)
    load_waves = staticmethod(load_waves)
    copy_waves = staticmethod(copy_waves)
