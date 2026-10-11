# -*- coding: utf-8 -*-
"""Audio file I/O: path resolution, loading and saving of wave files."""
from __future__ import annotations

import logging
import os
import re
import shutil
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path, PurePath
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

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
    'AudioInfo',
    'audio_info',
    'ConvertResult',
    'convert_folder',
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


#: Containers a caller may name instead of ``.wav``. Their suffix is an
#: extension and gets replaced, because ``clip.flac`` is asking for
#: ``clip.wav``. Anything else -- ``take.v2``, ``session.2026-10-11`` -- is
#: part of the name and keeps it.
_AUDIO_SUFFIXES = frozenset((
    '.wav', '.wave', '.flac', '.mp3', '.m4a', '.aac', '.ogg', '.oga', '.opus',
    '.aiff', '.aifc', '.aif', '.au', '.snd', '.caf', '.w64', '.sph', '.pcm',
    '.raw',
))


def _as_wave_path(path: Path) -> Path:
    """*path* ending in ``.wav``, with the rest of its name left alone.

    ``with_suffix`` cannot tell an extension from a dotted name, so it reads
    ``take.v2`` as a stem plus an extension and hands back ``take.wav`` --
    which is a *different* recording. Only a real container suffix is
    replaced; everything else gets ``.wav`` appended.
    """
    if path.suffix.lower() in _AUDIO_SUFFIXES:
        return path.with_suffix(WAVE_SUFFIX)
    return path.parent / (path.name + WAVE_SUFFIX)


def _free_wave_path(path: Path, sep: str = '') -> Path:
    """A sibling of *path* that nothing occupies: ``rec.wav`` -> ``rec2.wav``.

    The counter is joined onto the stem and the suffix put back by hand.
    Numbering through ``with_suffix`` instead turned the candidate
    ``take.v22`` into ``take.wav``, so a folder holding ``take.wav`` and
    ``take.v2.wav`` lost the first of them to the second -- in the mode whose
    whole job is to leave existing files alone.

    The highest index already in the folder is where the search starts, not
    where it ends: it says nothing about the names between it and *path*, and
    a sibling may have appeared since the glob. Every candidate is tested
    before it is handed back, so the name the caller gets really is free.
    """
    stem, suffix = path.stem, path.suffix
    pattern = re.compile(rf"{re.escape(stem)}{re.escape(sep)}(\d+)")
    existing = path.parent.glob(f"{stem}{sep}*")
    indices = [int(m.group(1)) for m in
               (pattern.fullmatch(p.stem) for p in existing) if m]
    n = max(indices) + 1 if indices else 2
    while True:
        candidate = path.parent / f"{stem}{sep}{n}{suffix}"
        if not candidate.exists():
            return candidate
        n += 1


def check_file(file_dir: PathLike, rename: bool = True, sep: str = '') -> str:
    """Normalize an output wave path: force the ``.wav`` suffix, create parent
    directories, and (optionally) rename to avoid overwriting an existing file.
    """
    file_dir = Path(file_dir)
    if file_dir.is_dir():
        file_dir = file_dir / 'unknown'
    file_dir = _as_wave_path(file_dir)
    if file_dir.exists():
        log.info("File already exists: %s", file_dir)
        if rename:
            file_dir = _free_wave_path(file_dir, sep)
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


def _abspath(path: PathLike) -> Path:
    # textual, like os.path.abspath: '..' is folded away rather than kept, so
    # two spellings of one folder compare equal without touching the disk
    return Path(os.path.abspath(str(path)))


def _common_root(files: Sequence[PathLike]) -> Optional[Path]:
    """The deepest folder holding every one of *files*, or ``None`` if none
    does (files on two Windows drives)."""
    if not files:
        return None
    try:
        return Path(os.path.commonpath([str(_abspath(f).parent) for f in files]))
    except ValueError:
        return None


def _source_root(src: Any, files: Sequence[PathLike],
                 root: Optional[PathLike] = None) -> Optional[Path]:
    """Where the folder layout of *files* is measured from.

    An explicit *root* wins and must hold every file. Otherwise a folder input
    is its own root, and anything else -- a list, a ``.txt`` manifest -- is
    measured from the folder the files have in common, so that two files
    called ``sx133.wav`` by two different talkers stay two files.
    """
    if root is not None:
        base = _abspath(root)
        for f in files:
            try:
                _abspath(f).relative_to(base)
            except ValueError:
                raise ValueError(f"{f} is not inside root={str(root)!r}") from None
        return Path(root)
    if isinstance(src, (str, PurePath)) and Path(src).is_dir():
        return Path(src)
    return _common_root(files)


def _relative(path: PathLike, root: Optional[PathLike]) -> Path:
    """*path* below *root*, or just its name when there is no root."""
    if root is None:
        return Path(Path(path).name)
    return _abspath(path).relative_to(_abspath(root))


def _count_non_finite(y: np.ndarray) -> int:
    """How many samples of *y* are NaN or inf."""
    y = np.asarray(y)
    if not y.size or not np.issubdtype(y.dtype, np.floating):
        return 0
    return int(np.count_nonzero(~np.isfinite(y)))


def _warn_non_finite(y: np.ndarray, wav_dir: PathLike) -> None:
    """Say so when a file comes back with NaN or inf samples in it.

    Not an error: repairing a corpus means reading it first. But this is the
    last point at which anyone can be told what is actually wrong with the
    clip, because every measurement taken from it afterwards comes back nan
    or -inf, and that reads like a microphone nobody switched on. kudio up to
    3.7.0 wrote FLOAT files without checking, so such files are sitting in
    corpora right now.
    """
    bad = _count_non_finite(y)
    if bad:
        log.warning("%s: %d of %d sample(s) are not finite (NaN or inf). This "
                    "is a corrupt file, not a quiet one -- anything measured "
                    "from it comes back nan or -inf. Repair the clip or drop "
                    "it from the set.", wav_dir, bad, y.size)


def file_load(wav_dir: PathLike, sr: Optional[int] = None, mono: bool = True
              ) -> Tuple[np.ndarray, int]:
    """Load a wave file; returns ``(waveform float32, sample_rate)``.

    ``sr=None`` keeps the file's native sample rate and reads via soundfile
    (fast, no librosa/numba needed). A requested ``sr`` triggers a resample
    through librosa.

    Multi-channel audio (``mono=False``) comes back **channels last**, shaped
    ``(frames, channels)`` -- the layout soundfile uses and the one
    :func:`kudio.loudness` expects.
    """
    try:
        if sr is None:
            y, rate = sf.read(str(wav_dir), dtype='float32', always_2d=False)
            if mono and y.ndim > 1:
                y = y.mean(axis=1).astype(np.float32)
            y = np.asarray(y, dtype=np.float32)
            _warn_non_finite(y, wav_dir)
            return y, int(rate)
        # resampling path (heavier)
        import librosa
        y, rate = librosa.load(str(wav_dir), sr=sr, mono=mono)
        if y.ndim > 1:
            # librosa hands back (channels, frames); the rest of kudio -- and
            # the sr=None branch above -- speak (frames, channels). Two layouts
            # from one function, chosen by an unrelated argument, is exactly
            # the kind of thing nobody notices until a model is training on
            # transposed audio.
            y = y.T
        y = np.ascontiguousarray(y, dtype=np.float32)
        _warn_non_finite(y, wav_dir)
        return y, int(rate)
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


@dataclass(frozen=True)
class AudioInfo:
    """What a file is, before you decide whether to load it.

    ``duration`` is in seconds and ``peak`` is the largest absolute sample.
    ``subtype`` / ``format`` are soundfile's (``'PCM_16'``, ``'WAV'``) and are
    ``None`` when the info was taken from an array rather than a file.
    """
    sr: int
    channels: int
    frames: int
    duration: float
    peak: float
    path: Optional[Path] = None
    subtype: Optional[str] = None
    format: Optional[str] = None

    def __str__(self) -> str:  # pragma: no cover - presentation only
        name = self.path.name if self.path is not None else "<array>"
        enc = f", {self.subtype}" if self.subtype else ""
        return (f"{name}: {self.duration:.3f}s, {self.sr} Hz, "
                f"{self.channels}ch{enc}, peak {self.peak:.4f}")


def audio_info(source: Union[PathLike, np.ndarray], sr: Optional[int] = None,
               peak: bool = True) -> AudioInfo:
    """Describe an audio file (or an in-memory waveform) without interpreting it.

    >>> info = kudio.audio_info("clip.wav")
    >>> info.duration, info.sr, info.channels
    (3.5, 16000, 1)

    The header alone answers rate, channels, frames and duration, so
    ``peak=False`` never touches the samples and stays instant on a large file.
    The default reads them, because every caller so far wanted the peak too.

    An array needs its *sr* passed in — a waveform does not carry one — and is
    read **channels last**, matching :func:`file_load`.
    """
    if isinstance(source, np.ndarray):
        if sr is None:
            raise ValueError("audio_info(array) needs sr=")
        y = source
        frames = int(y.shape[0]) if y.size else 0
        channels = 1 if y.ndim == 1 else int(y.shape[1])
        return AudioInfo(sr=int(sr), channels=channels, frames=frames,
                         duration=frames / float(sr) if sr else 0.0,
                         peak=float(np.max(np.abs(y))) if y.size else 0.0)

    path = Path(source)
    try:
        meta = sf.info(str(path))
    except Exception as e:
        raise AudioIOError(f"Could not read audio header: {path} ({e})") from e

    top = 0.0
    if peak:
        bad = 0
        try:
            with sf.SoundFile(str(path)) as fh:
                for block in fh.blocks(blocksize=1 << 20, dtype='float32'):
                    if block.size:
                        bad += _count_non_finite(block)
                        top = max(top, float(np.max(np.abs(block))))
        except Exception as e:
            raise AudioIOError(f"Could not read audio file: {path} ({e})") from e
        if bad:
            # `max(0.0, nan)` keeps the 0.0, so a file with NaN in it reported
            # a peak of 0.0 -- digital silence, for a clip that is anything
            # but. There is no largest sample when some of them are not
            # numbers, so say nan and let callers ask `isfinite(info.peak)`.
            log.warning("%s: %d sample(s) are not finite (NaN or inf), so the "
                        "peak is undefined. Repair the clip or drop it from "
                        "the set.", path, bad)
            top = float('nan')

    return AudioInfo(sr=int(meta.samplerate), channels=int(meta.channels),
                     frames=int(meta.frames),
                     duration=float(meta.frames) / float(meta.samplerate)
                     if meta.samplerate else 0.0,
                     peak=top, path=path,
                     subtype=meta.subtype, format=meta.format)


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


def _is_fixed_point(subtype: str) -> bool:
    """Does *subtype* store integers, so that a sample past ±1.0 cannot be
    written as it is? True for ``PCM_*`` and the G.711 codes; false for
    ``FLOAT`` / ``DOUBLE``, which keep it."""
    name = str(subtype).upper()
    return name.startswith('PCM_') or name in ('ULAW', 'ALAW')


def save_wave(path: PathLike, wave: np.ndarray, sr: int,
              d_sample: Optional[int] = None, subtype: str = 'PCM_16') -> None:
    """Write *wave* to a ``.wav`` file via soundfile.

    :param subtype: soundfile subtype — ``'PCM_16'`` (default), ``'PCM_24'``,
        ``'FLOAT'`` (32-bit float), ... see ``soundfile.available_subtypes()``.
    :param d_sample: optional resample rate before writing.

    int16 input is scaled the way soundfile reads a 16-bit file back,
    ``v / 32768``, so a 16-bit array is written unchanged; float input is
    written as-is.

    A fixed-point *subtype* clips anything beyond ±1.0 to full scale, and
    ``PCM_16`` is rounded to the nearest step. Both are done here rather than
    left to libsndfile, whose conversion floors -- every sample half a step
    low -- and wraps around unless clipping has been switched on: 1.2 comes
    back as -0.8. Whether it has been is up to the soundfile release.

    :raises AudioIOError: if any sample is NaN or inf. Neither subtype family
        survives one: ``FLOAT`` keeps it, and an integer one turns it into a
        zero that reads back as silence.
    """
    Path(path).parent.mkdir(exist_ok=True, parents=True)
    wave = np.asarray(wave)

    if np.issubdtype(wave.dtype, np.integer):
        wave = wave.astype(np.float32) / 32768.0
    else:
        wave = wave.astype(np.float32)

    bad = _count_non_finite(wave)
    if bad:
        # Neither subtype family survives this. `FLOAT` writes the NaN out,
        # and every reader of that file gets nan back for the peak, the RMS
        # and the loudness; an integer subtype converts it to a zero, which is
        # worse, because the clip then reads as plain silence and nothing ever
        # says otherwise. One truncated file or one divide by a silent frame
        # upstream is enough, so it stops here with the name of the file.
        raise AudioIOError(
            f"refusing to write {path}: {bad} of {wave.size} sample(s) are "
            f"not finite (NaN or inf). Written as FLOAT they stay, and as an "
            f"integer subtype they become zeros that read back as silence. "
            f"Find what produced them -- a gain computed from a silent frame "
            f"is the usual one -- or drop the clip.")

    if d_sample is not None and d_sample != sr:
        import librosa
        wave = librosa.resample(wave, orig_sr=sr, target_sr=d_sample)
        sr = d_sample

    if _is_fixed_point(subtype):
        wave = np.clip(wave, -1.0, 1.0)
        if str(subtype).upper() == 'PCM_16':
            # soundfile writes int16 as it stands, so the rounding is ours
            wave = np.clip(np.round(wave * 32768.0), -32768, 32767).astype(np.int16)

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


@dataclass(frozen=True)
class ConvertResult:
    """What :func:`convert_folder` did. ``failed`` is ``(path, reason)`` pairs."""
    written: int
    total: int
    outputs: List[Path]
    failed: List[Tuple[Path, str]]

    def __bool__(self) -> bool:
        return not self.failed

    def __str__(self) -> str:  # pragma: no cover - presentation only
        return f"{self.written}/{self.total} written, {len(self.failed)} failed"


def convert_folder(src: Any, dst: PathLike, *,
                   sr: Optional[int] = None,
                   mono: bool = True,
                   subtype: str = 'PCM_16',
                   trim_db: Optional[float] = None,
                   peak: Optional[float] = None,
                   lufs: Optional[float] = None,
                   overwrite: bool = False,
                   progress: Optional[Callable[[int, int, Path], None]] = None,
                   on_error: str = 'collect',
                   root: Optional[PathLike] = None) -> ConvertResult:
    """Resample / re-encode / trim / normalise every audio file under *src*.

    *src* is anything :func:`check_input` accepts — a folder, a file, a ``.txt``
    manifest, or a list of those. The output mirrors the input's directory
    structure below *dst*, so two files with the same name in different
    subfolders do not collide. A folder is mirrored from itself; a list or a
    manifest from the deepest folder its files share, or from *root*.

    >>> kudio.convert_folder("raw/", "16k/", sr=16000, lufs=-23.0)
    ConvertResult(written=412, total=412, ...)

    :param trim_db: ``top_db`` for :func:`kudio.trim_silence`; ``None`` skips it.
    :param peak: peak-normalise target (e.g. ``0.99``).
    :param lufs: loudness-normalise target in LUFS (e.g. ``-23.0``). Mutually
        exclusive with *peak* — normalising twice would undo the first one.
    :param overwrite: by default an existing output is renamed rather than
        replaced (:func:`check_file`).
    :param progress: optional ``callback(done, total, path)``.
    :param on_error: ``'collect'`` (default) records the failure and carries on;
        ``'raise'`` stops at the first one.
    :param root: the folder the output layout is measured from. Every input
        has to be inside it.
    """
    if peak is not None and lufs is not None:
        raise ValueError("pass peak= or lufs=, not both")
    if on_error not in ('collect', 'raise'):
        raise ValueError(f"on_error must be 'collect' or 'raise', got {on_error!r}")

    files, _ = check_input(src)
    if not files:
        raise AudioIOError(f"no audio files found in {src!r}")

    root = _source_root(src, files, root)
    out_root = Path(dst)
    out_root.mkdir(parents=True, exist_ok=True)

    outputs: List[Path] = []
    failed: List[Tuple[Path, str]] = []

    for n, f in enumerate(files, 1):
        try:
            y, rate = file_load(f, sr=sr, mono=mono)
            if trim_db is not None:
                from kudio.effects.silence import trim_silence
                y = trim_silence(y, top_db=trim_db)[0]
            if y.size == 0:
                raise AudioIOError("empty after trimming")
            if peak is not None:
                from kudio.effects.augment import normalize
                y = normalize(y, peak=peak)
            elif lufs is not None:
                from kudio.core.loudness import normalize_lufs
                y = normalize_lufs(y, rate, lufs=lufs)

            target = (out_root / _relative(f, root)).with_suffix(WAVE_SUFFIX)
            target = Path(check_file(target, rename=not overwrite))
            save_wave(target, y, rate, subtype=subtype)
            outputs.append(target)
        except Exception as e:
            if on_error == 'raise':
                raise
            failed.append((Path(f), f"{type(e).__name__}: {e}"))
        if progress is not None:
            progress(n, len(files), Path(f))

    return ConvertResult(written=len(outputs), total=len(files),
                         outputs=outputs, failed=failed)


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
