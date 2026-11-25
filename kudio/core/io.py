# -*- coding: utf-8 -*-
"""Audio file I/O: path resolution, loading and saving of wave files."""
from __future__ import annotations

import contextlib
import errno
import logging
import math
import os
import re
import secrets
import shutil
import stat
import struct
import time
import warnings
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path, PurePath
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
import soundfile as sf

from kudio import _optional
from kudio._progress import tqdm
from kudio._samples import as_float, channels_last, layout_known, refuse_64_bit
from kudio.core._policy import warn
from kudio.exceptions import AudioIOError, DependencyError

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
    # listed rather than globbed: a name holding [ or * is a glob pattern,
    # and runs/exp[1] missed its own runs/exp[1]2 and handed it back
    pattern = re.compile(rf"{re.escape(path.name)}{re.escape(sep)}(\d+)")
    siblings = path.parent.iterdir() if path.parent.exists() else []
    indices = [int(m.group(1)) for m in (pattern.fullmatch(p.name) for p in siblings) if m]
    n = max(indices) + 1 if indices else 2
    while Path(f"{path}{sep}{n}").exists():
        n += 1
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
    existing = path.parent.iterdir() if path.parent.exists() else []
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

    A folder becomes ``<folder>/unknown.wav``. A path ending in a separator is
    taken as a folder even before it exists: ``Path`` drops the trailing
    slash, so ``"newdir/"`` would otherwise come back as ``newdir.wav``.
    """
    spelled = str(file_dir)
    file_dir = Path(file_dir)
    if file_dir.is_dir() or spelled.endswith(('/', '\\')):
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

    The suffix test ignores case, so a recorder's ``ZOOM0001.WAV`` counts
    whether it is named directly, listed, or found in a folder -- and a folder
    collects the same files on Windows as on Linux, where a plain
    ``rglob("*.wav")`` would not. A ``.txt`` list is read as UTF-8 (a BOM is
    fine), and relative paths in it resolve against the working directory.
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
            if input.suffix.lower() == '.txt':
                text = input.read_text(encoding='utf-8-sig')
                lines = [x.strip() for x in text.strip().splitlines() if x.strip()]
                for item in lines:
                    r, w = check_input(item)
                    right_list.extend(r)
                    wrong_list.extend(w)
            elif input.suffix.lower() == WAVE_SUFFIX:
                right_list.append(input)
            else:
                wrong_list.append(str(input))
        elif input.is_dir():
            right_list.extend(sorted(p for p in input.rglob('*')
                                     if p.suffix.lower() == WAVE_SUFFIX and p.is_file()))
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


def _check_rate(value: Any, name: str = 'sr', optional: bool = False,
                whole: bool = True) -> Any:
    """*value* as an int -- or, for a rate that need not be *whole*, as the
    float it is -- or a ValueError naming the argument it was passed as.

    Headers, configs and numpy hand rates out as 16000.0, np.int32(16000) or a
    0-d array, so a number of any type passes, and None where the argument is
    *optional*. ``True`` equals 1 and is refused by type: taken as a rate of
    1 Hz it makes two samples of two seconds of audio.
    """
    if value is None and optional:
        return None
    number = value.item() if isinstance(value, np.ndarray) and value.ndim == 0 else value
    # float() would read '16000' too, and text has no __float__; a complex
    # number of numpy's does, and drops its imaginary part
    if hasattr(number, '__float__') and not isinstance(
            number, (bool, np.bool_, complex, np.complexfloating, np.ndarray)):
        try:
            rate = float(number)
        except (TypeError, ValueError, OverflowError):
            rate = math.nan
        if 0 < rate < math.inf:
            if rate.is_integer():
                return int(rate)
            if not whole:
                return rate
    raise ValueError(f"{name} must be a positive {'whole ' if whole else ''}rate"
                     f"{' or None' if optional else ''}, got {value!r}")


class _Unread(Exception):
    """A file that could not be read. The text is the reason, which the caller
    puts in an error of its own together with the path."""


def _nothing_to_read(path: PathLike) -> Optional[str]:
    """Why there is nothing at *path* to decode, or None when there is.

    Asked before any decoder is: libsndfile says "System error" for a missing
    file and for a folder alike.
    """
    try:
        found = os.stat(str(path))
    except (FileNotFoundError, NotADirectoryError, ValueError):
        return "no such file"              # ValueError: a null in the name
    except OSError as e:
        return e.strerror or type(e).__name__
    if stat.S_ISDIR(found.st_mode):
        return "it is a folder, not a file"
    # only a regular file says how long it is; a pipe says 0 and may hold audio
    if stat.S_ISREG(found.st_mode) and not found.st_size:
        return "the file is empty, 0 bytes"
    return None


#: libsndfile's SF_ERR_UNRECOGNISED_FORMAT, SF_ERR_SYSTEM,
#: SF_ERR_UNSUPPORTED_ENCODING and SFE_BAD_FILE
_UNRECOGNISED, _SYSTEM, _UNSUPPORTED, _BAD_FILE = 1, 2, 4, 7


def _left_to_ffmpeg(error: BaseException, path: PathLike) -> Optional[str]:
    """How a message puts it that soundfile does not read *path*, when that is
    for ffmpeg to try; None when soundfile's refusal stands.

    Two refusals are not about the file: the format is not known, or the
    encoding not supported. Any other one is: from a damaged FLAC ffmpeg hands
    back as much as it can, and does not say how much that was.

    An extension of .mp3 is all libsndfile needs to take a file it does not
    recognise for MP3. Finding no frame in it, it says that the file does not
    exist or is a pipe.
    """
    code = error.code if isinstance(error, sf.LibsndfileError) else None
    if code == _UNSUPPORTED:
        return "soundfile does not support the encoding"
    if code == _UNRECOGNISED or (code == _BAD_FILE and Path(path).suffix.lower() == '.mp3'):
        return "soundfile does not recognise the format"
    return None


def _reason(error: BaseException, path: PathLike) -> str:
    """What stopped *path* from being read, for the brackets at the end of the
    message: the text of *error*, or its type when it has none.

    libsndfile's "System error" is all it says of a file that is there and
    will not open, so the operating system is asked what it has against it.
    """
    if isinstance(error, sf.LibsndfileError) and error.code == _SYSTEM:
        try:
            with open(str(path), 'rb'):
                pass
        except OSError as e:
            return e.strerror or type(e).__name__
    return str(error) or type(error).__name__


# What libsndfile leaves -- AAC in m4a, mp4, 3gp or on its own, WebM, AMR, WMA,
# ALAC -- goes through audioread, which runs ffmpeg or, on a Mac, Core Audio. It
# is imported here and not at the top: a machine that reads WAV needs neither,
# and librosa 1.0 no longer brings it along.

def _decoders(refused: str) -> list:
    """What audioread can decode with on this machine."""
    try:
        import audioread
        for fresh in (False, True):
            # its own WAV, AIFF and AU reader is left out: those are
            # libsndfile's, and it has already turned the file down
            found = [backend for backend in audioread.available_backends(flush_cache=fresh)
                     if backend.__name__ != 'RawAudioFile']
            if found:
                return found
            # audioread looks once and remembers; ffmpeg may be on the PATH by now
    except ImportError as e:
        if isinstance(e, ModuleNotFoundError) and e.name == 'audioread':
            raise _Unread(f"{refused}. Such a file is decoded with audioread and ffmpeg, "
                          f"and audioread is not installed: pip install audioread") from e
        raise _Unread(f"{refused}. Such a file is decoded with audioread, which cannot be "
                      f"imported: {e}") from e
    raise _Unread(f"{refused}. Such a file is decoded with ffmpeg, and audioread found "
                  f"none: install ffmpeg and put it on the PATH")


def _decode(path: PathLike, error: BaseException,
            take: Optional[Callable[[bytes], Any]] = None) -> Tuple[int, int, float]:
    """``(rate, channels, seconds)`` of a file soundfile refused with *error*,
    the seconds being the header's. *take* gets the 16-bit samples, block by
    block; without it nothing but the header is read.
    """
    refused = _left_to_ffmpeg(error, path)
    decoders = _decoders(refused)
    import audioread
    cause: Optional[BaseException] = None
    try:
        # the whole path: ffmpeg takes what stands before a colon in a
        # relative one for a protocol
        with audioread.audio_open(os.path.abspath(str(path)), backends=decoders) as reader:
            rate, channels = int(reader.samplerate), int(reader.channels)
            got = take is None
            if take is not None:
                for block in reader:
                    take(block)
                    got = True
            # a header with nothing behind it opens, and decodes to nothing
            if got and rate > 0 and channels > 0:
                return rate, channels, float(reader.duration)
    except (audioread.DecodeError, EOFError, OSError) as e:
        cause = e
    raise _Unread(f"{refused}, and it could not be decoded either: the file is damaged, "
                  f"cut short or holds no audio") from cause


def _warn_decoded(path: PathLike, error: BaseException, seconds: float,
                  header: float, counted: bool = True) -> None:
    """Say so when a file soundfile refused with *error* may not be what it
    seems, now that audioread has read it.

    One named as a format soundfile reads, and not recognised by it, is damaged
    or has the wrong extension; ffmpeg reads both, and of a FLAC that lost its
    beginning it hands back the rest. And ffmpeg decodes as far as a file goes
    and reports no error: half of an m4a made for streaming, or of a WebM, is
    half the clip. Only containers that store how long they are can be held to
    it. For a bare AAC or AMR stream ffmpeg guesses from the bit rate.
    """
    from kudio.core.dataset_score import _audio_suffixes
    suffix = Path(path).suffix.lower()
    if error.code != _UNSUPPORTED and suffix in _audio_suffixes():
        made = (f"ffmpeg decoded {seconds:.2f} s of it" if counted else
                f"ffmpeg reads {seconds:.1f} s off its header")
        warn(f"{path}: soundfile does not recognise it as the {suffix} file its name says, "
             f"and {made}. The file is damaged, or its extension is wrong.")
        return
    with open(str(path), 'rb') as f:
        head = f.read(8)
    stored = head[4:8] == b'ftyp' or head[:4] == b'\x1aE\xdf\xa3'     # MP4, Matroska
    # the header's figure comes rounded down to a tenth of a second
    if counted and stored and header - seconds > 0.1:
        warn(f"{path}: {seconds:.2f} s could be decoded of the {header:.1f} s in its "
             f"header. The file is cut short or damaged, unless it is a video whose "
             f"sound ends early.")


#: The least length of samples that is not one. A program that streams its
#: output writes a header before it knows the length and cannot go back to it:
#: sox leaves 0x7FFFF000 there, ffmpeg 0xFFFFFFFF, and others 0.
_STREAMED = 0x7FFFF000

#: the chunks looked through for the samples of a WAV, which has a dozen
_MOST_CHUNKS = 4096


def _wav_shortfall(path: PathLike) -> Optional[Tuple[float, int]]:
    """``(seconds, bytes)`` of a RIFF WAV that ends before its samples do: how
    long its header says it is, and how much of that is missing. None for one
    that is whole, or short by a block at most, and for any other file.
    """
    found = os.stat(str(path))
    if not stat.S_ISREG(found.st_mode):              # a pipe is read by now
        return None
    with open(str(path), 'rb') as f:
        head = f.read(12)
        if head[:4] != b'RIFF' or head[8:] != b'WAVE':
            return None
        block = per_second = 0
        for _ in range(_MOST_CHUNKS):
            chunk = f.read(8)
            if len(chunk) < 8:
                return None
            name, length = struct.unpack('<4sI', chunk)
            if name == b'data':
                missing = length - (found.st_size - f.tell())
                if not 0 < length < _STREAMED or not per_second or missing <= block:
                    return None
                return length / per_second, missing
            start = f.tell()
            if name == b'fmt ':
                described = f.read(16)
                if len(described) < 16:
                    return None
                tag, _, rate, average, block, _ = struct.unpack('<HHIIHH', described)
                # a block is a frame in PCM, float, the G.711 codes and the
                # extensible form of them; in a codec it is a packet
                per_second = rate * block if tag in (1, 3, 6, 7, 0xFFFE) else average
            f.seek(start + length + length % 2)      # chunks begin at even bytes
    return None


def _warn_cut_wav(path: PathLike, frames: int, sr: int) -> None:
    """Say so when a WAV ends before its header says, which libsndfile reads as
    a shorter recording. Only the header and the size of the file are looked
    at, so it is said of a file whose samples nobody asked for as well."""
    try:
        short = _wav_shortfall(path)
    except OSError:
        return
    if short:
        seconds, missing = short
        warn(f"{path}: the header says {seconds:.2f} s and {frames / sr:.2f} s are there, "
             f"{missing} bytes fewer. The file may have been cut short.")


def _read_native(path: PathLike, rewind: bool = True) -> Tuple[np.ndarray, int]:
    """*path* at its own rate: float32, 1-D or ``(frames, channels)``, and the
    rate. Every function that reads audio from a file reads it here.

    soundfile first, and audioread for what soundfile leaves to it
    (:func:`_left_to_ffmpeg`).

    *rewind* is what ``soundfile.read`` does before it reads, seek to the first
    frame. In a low-rate MP3 that moves some samples by a step of float32, so
    ``file_load`` keeps to it without *sr* and does without it with one.

    :raises _Unread: with the reason, when the file is not there or cannot be
        decoded. soundfile's own errors are passed on.
    """
    gone = _nothing_to_read(path)
    if gone:
        raise _Unread(gone)
    try:
        if rewind:
            y, rate = sf.read(str(path), dtype='float32', always_2d=False)
        else:
            with sf.SoundFile(str(path)) as f:
                y, rate = f.read(dtype='float32', always_2d=False), f.samplerate
        _warn_cut_wav(path, len(y), rate)
        return y, int(rate)
    except sf.LibsndfileError as e:
        if _left_to_ffmpeg(e, path) is None:
            raise
        blocks: List[bytes] = []
        rate, channels, header = _decode(path, e, blocks.append)
        samples = np.frombuffer(b''.join(blocks), dtype='<i2')
        y = samples[:samples.size - samples.size % channels].astype(np.float32) / 32768.0
        _warn_decoded(path, e, y.size / channels / rate, header)
        return (y if channels == 1 else y.reshape(-1, channels)), rate


def _read_mono(path: PathLike) -> Tuple[np.ndarray, int]:
    """:func:`_read_native` without the rewind, its channels averaged, for what
    cannot be made from a clip with NaN or inf in it: a mixture, a phase."""
    y, rate = _read_native(path, rewind=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    bad = _count_non_finite(y)
    if bad:
        raise _Unread(f"{bad} of {y.size} sample(s) are not finite (NaN or inf)")
    return y, rate


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
        # a warning rather than a log line: the kudio logger has a
        # NullHandler, so a log.warning here reached nobody who had not set
        # up logging first, which is everybody this message is for
        warnings.warn(f"{wav_dir}: {bad} of {y.size} sample(s) are not finite "
                      f"(NaN or inf). This is a corrupt file, not a quiet one "
                      f"-- anything measured from it comes back nan or -inf. "
                      f"Repair the clip or drop it from the set.",
                      UserWarning, stacklevel=3)


def file_load(wav_dir: PathLike, sr: Optional[int] = None, mono: bool = True
              ) -> Tuple[np.ndarray, int]:
    """Load a wave file; returns ``(waveform float32, sample_rate)``.

    ``sr=None`` keeps the file's native sample rate. Another *sr* resamples,
    which takes librosa; a file already at *sr* comes back as it is.

    The file is read with soundfile. What soundfile does not read -- m4a, mp4,
    aac, webm, 3gp, amr -- is decoded to 16 bits with audioread and ffmpeg,
    whichever *sr* is asked for; neither comes with kudio. A file that gives
    less than its header promises, or is not recognised as what its extension
    says, is read with a warning.

    Multi-channel audio (``mono=False``) comes back **channels last**, shaped
    ``(frames, channels)`` -- the layout soundfile uses and the one
    :func:`kudio.loudness` expects. A mono file stays 1-D with ``mono=False``;
    it does not become ``(frames, 1)``.

    A file with NaN or inf in it is read with a warning too, at its own rate;
    it is not resampled.

    :raises ValueError: for an *sr* that is not a positive whole rate. That is
        a mistake in the call, so it is not dressed up as a bad file.
    :raises AudioIOError: when there is no such file, it is a folder or empty,
        it cannot be read, or it holds NaN or inf and would have to be
        resampled.
    """
    sr = _check_rate(sr, optional=True)
    try:
        y, rate = _read_native(wav_dir, rewind=sr is None)
        if mono and y.ndim > 1:
            y = y.mean(axis=1)
        y = np.ascontiguousarray(y, dtype=np.float32)
        if sr is None or sr == rate:
            _warn_non_finite(y, wav_dir)
            return y, rate
        bad = _count_non_finite(y)
        if bad:
            # a resampler's filter is long, and would turn one NaN into a
            # whole filter length of them
            raise _Unread(f"{bad} of {y.size} sample(s) are not finite (NaN or inf), so "
                          f"it cannot be resampled from {rate} to {sr} Hz; read it with "
                          f"sr=None, repair it and resample that")
        with layout_known():
            y = resample(y, rate, sr)
        return np.ascontiguousarray(y, dtype=np.float32), sr
    except DependencyError:
        # a missing librosa is not a file that cannot be read
        raise
    except Exception as e:
        raise AudioIOError(f"Could not load audio file: {wav_dir} "
                           f"({_reason(e, wav_dir)})") from e


def resample(y: np.ndarray, orig_sr: float, target_sr: float,
             res_type: str = 'soxr_hq') -> np.ndarray:
    """Resample a waveform already in memory.

    The in-memory counterpart of ``file_load(..., sr=target_sr)``, which can
    only resample on the way in from disk. Returns *y* untouched when the rates
    already match, so it is safe to call unconditionally.

    >>> y16 = kudio.resample(y8, orig_sr=8000, target_sr=16000)

    A ``(frames, channels)`` array is resampled along its frames, the layout
    :func:`file_load` returns, and a row ``(1, n)`` is one channel. Integer
    input comes back in its own dtype, rounded and clipped to its range rather
    than truncated; unsigned types are read around their midpoint.

    :param orig_sr: the rate *y* is at. It need not be whole, unlike a file's:
        22254.5 Hz is a rate an array can be at.
    :param target_sr: the rate to take it to, the same.
    :raises ValueError: for a rate that is not a positive number, 64-bit
        integers, or an array that is channels first.
    """
    orig_sr = _check_rate(orig_sr, 'orig_sr', whole=False)
    target_sr = _check_rate(target_sr, 'target_sr', whole=False)
    y = channels_last(refuse_64_bit(y, 'resample'), 'resample')
    if orig_sr == target_sr or y.size == 0:
        return y
    if y.ndim > 2:
        raise ValueError(f"expected (frames,) or (frames, channels), got shape {y.shape}")
    librosa = _optional.librosa()
    limits = np.iinfo(y.dtype) if np.issubdtype(y.dtype, np.integer) else None
    middle = (int(limits.max) + 1) / 2 if limits is not None and limits.min == 0 else 0
    x = y.astype(np.float64) - middle if middle else y
    # librosa resamples its last axis by default, which on a channels-last
    # array is the channel axis: a (16000, 2) clip came back (16000, 1)
    out = librosa.resample(x.astype(np.float32), orig_sr=orig_sr,
                           target_sr=target_sr, res_type=res_type, axis=0)
    if limits is None:
        return out.astype(y.dtype, copy=False)
    # in float64: the largest int32 is not a float32, and 2**31 cast back to
    # int32 is full scale the other way
    return np.clip(np.round(out.astype(np.float64) + middle),
                   limits.min, limits.max).astype(y.dtype)


def _resampled_length(n: int, orig_sr: float, target_sr: float) -> int:
    """How many frames :func:`resample` makes of *n*. librosa pads or cuts what
    the resampler gives to this."""
    return n if orig_sr == target_sr else int(np.ceil(n * (float(target_sr) / orig_sr)))


@dataclass(frozen=True)
class AudioInfo:
    """What a file is, before you decide whether to load it.

    ``duration`` is in seconds and ``peak`` is the largest absolute sample:
    ``nan`` when some samples are not finite, ``None`` when it was not
    measured (``peak=False``).
    ``subtype`` / ``format`` are soundfile's (``'PCM_16'``, ``'WAV'``) and are
    ``None`` when the info was taken from an array, or from a file soundfile
    does not read.
    """
    sr: int
    channels: int
    frames: int
    duration: float
    peak: Optional[float]
    path: Optional[Path] = None
    subtype: Optional[str] = None
    format: Optional[str] = None

    def __str__(self) -> str:  # pragma: no cover - presentation only
        name = self.path.name if self.path is not None else "<array>"
        enc = f", {self.subtype}" if self.subtype else ""
        level = "peak not measured" if self.peak is None else f"peak {self.peak:.4f}"
        return (f"{name}: {self.duration:.3f}s, {self.sr} Hz, "
                f"{self.channels}ch{enc}, {level}")


def audio_info(source: Union[PathLike, np.ndarray], sr: Optional[int] = None,
               peak: bool = True) -> AudioInfo:
    """Describe an audio file (or an in-memory waveform) without interpreting it.

    >>> info = kudio.audio_info("clip.wav")
    >>> info.duration, info.sr, info.channels
    (3.5, 16000, 1)

    The header alone answers rate, channels, frames and duration, so
    ``peak=False`` never touches the samples and stays instant on a large file.
    The default reads them, because every caller so far wanted the peak too.

    What soundfile does not read (m4a, aac, webm) goes through audioread and
    ffmpeg, as in :func:`file_load`. ``frames`` and ``duration``
    are then counted from what was decoded; with ``peak=False`` they are the
    header's, which is read to a tenth of a second, unless it gives no
    duration.

    A WAV that ends before its header says is described as the shorter
    recording it holds, with a warning. That takes no samples either.

    An array needs its *sr* passed in — a waveform does not carry one — and is
    read **channels last**, matching :func:`file_load`. A row ``(1, n)`` is one
    channel.

    :raises ValueError: for an array without *sr*, or with one that is not a
        positive whole rate, and for an array that is channels first or holds
        64-bit integers.
    :raises AudioIOError: when there is no such file, it is a folder or empty,
        or its header cannot be read.
    """
    if isinstance(source, np.ndarray):
        if sr is None:
            raise ValueError("audio_info(array) needs sr=")
        sr = _check_rate(sr)
        y = channels_last(refuse_64_bit(source, 'audio_info'), 'audio_info')
        frames = int(y.shape[0]) if y.size else 0
        channels = 1 if y.ndim == 1 else int(y.shape[1])
        top: Optional[float] = None
        if peak:
            # on the full scale a file's peak is read on; abs(-32768) is
            # -32768 in int16 besides
            top = _peak(as_float(y), "<array>")
        return AudioInfo(sr=sr, channels=channels, frames=frames,
                         duration=frames / float(sr), peak=top)

    path = Path(source)
    try:
        gone = _nothing_to_read(path)
        if gone:
            raise _Unread(gone)
        try:
            meta = sf.info(str(path))
        except sf.LibsndfileError as e:
            if _left_to_ffmpeg(e, path) is None:
                raise
            return _decoded_info(path, e, peak)
    except Exception as e:
        raise AudioIOError(f"Could not read audio header: {path} "
                           f"({_reason(e, path)})") from e

    top: Optional[float] = None
    if peak:
        top, bad = 0.0, 0
        try:
            with sf.SoundFile(str(path)) as fh:
                for block in fh.blocks(blocksize=1 << 20, dtype='float32'):
                    if block.size:
                        bad += _count_non_finite(block)
                        top = max(top, float(np.max(np.abs(block))))
        except Exception as e:
            raise AudioIOError(f"Could not read audio file: {path} "
                               f"({_reason(e, path)})") from e
        if bad:
            top = _undefined_peak(path, bad)

    _warn_cut_wav(path, meta.frames, meta.samplerate)
    return AudioInfo(sr=int(meta.samplerate), channels=int(meta.channels),
                     frames=int(meta.frames),
                     duration=float(meta.frames) / float(meta.samplerate)
                     if meta.samplerate else 0.0,
                     peak=top, path=path,
                     subtype=meta.subtype, format=meta.format)


def _decoded_info(path: Path, error: BaseException, peak: bool) -> AudioInfo:
    """:func:`audio_info` of a file soundfile refused with *error*. Decoded
    block by block, so that a long file is not held whole."""
    samples = loudest = 0

    def take(block: bytes) -> None:
        nonlocal samples, loudest
        x = np.frombuffer(block, dtype='<i2')
        if x.size:
            samples += x.size
            # int32 first: the absolute value of -32768 does not fit int16
            loudest = max(loudest, int(np.max(np.abs(x.astype(np.int32)))))

    rate, channels, header = _decode(path, error, take if peak else None)
    if not peak and header <= 0:             # a live recording's header has none
        _decode(path, error, take)
    counted = peak or header <= 0
    frames = samples // channels if counted else int(round(header * rate))
    seconds = frames / rate if counted else float(header)
    _warn_decoded(path, error, seconds, header, counted)
    return AudioInfo(sr=rate, channels=channels, frames=frames, duration=seconds,
                     peak=loudest / 32768.0 if peak else None, path=path)


def _undefined_peak(name: Any, bad: int) -> float:
    """`max(0.0, nan)` keeps the 0.0, so a clip with NaN in it once reported a
    peak of 0.0 -- digital silence, for a clip that is anything but. There is
    no largest sample when some of them are not numbers, so the answer is nan,
    and callers can ask ``isfinite(info.peak)``."""
    warnings.warn(f"{name}: {bad} sample(s) are not finite (NaN or inf), so the "
                  f"peak is undefined. Repair the clip or drop it from the set.",
                  UserWarning, stacklevel=4)
    return float('nan')


def _peak(y: np.ndarray, name: Any) -> float:
    """The largest absolute sample, or nan (with a warning) when some are not
    finite -- the same answer audio_info gives for a file."""
    if not y.size:
        return 0.0
    bad = _count_non_finite(y)
    if bad:
        return _undefined_peak(name, bad)
    return float(np.max(np.abs(y)))


def load_wave(wave: Any, num: Optional[int] = None
              ) -> Union[Tuple[np.ndarray, int], List[Tuple[np.ndarray, int]]]:
    """Load one wave file or a list of them into ``(waveform, sr)`` tuples."""
    info = wave if num is None else wave[num]
    if isinstance(info, (list, tuple)):
        return [load_wave(w) for w in tqdm(info, desc="[kudio] Loading audios")]
    if isinstance(info, (str, PurePath)) and Path(info).is_file() \
            and Path(info).suffix.lower() == WAVE_SUFFIX:
        return file_load(info)
    raise ValueError(f"Unexpected wave input: {info!r}")


def load_waves(wave: Any, use_pool: bool = False, std_len: bool = False, core_: int = 8
               ) -> Union[List[Tuple[np.ndarray, int]], Tuple[List[np.ndarray], int, int]]:
    """Convert every ``.wav`` in *wave* (see :func:`check_input`) to waveform data.

    Returns a list of ``(np.ndarray, int)``. With ``std_len=True`` returns
    ``(waves_2d, sr, min_len)`` where every waveform is truncated to the
    shortest length (all files must share one sample rate).
    """
    waves, skipped = check_input(wave)
    for item in skipped:
        log.warning("Skipped, not an existing .wav file: %s", item)
    if std_len and not waves:
        raise AudioIOError(f"no audio files found in {wave!r}")
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


#: Bits per sample of the fixed-point subtypes save_wave rounds itself.
_PCM_BITS = {'PCM_S8': 8, 'PCM_U8': 8, 'PCM_16': 16, 'PCM_24': 24, 'PCM_32': 32}


#: extensions soundfile cannot read the container from, and the container
_CONTAINERS = {'.aif': 'AIFF'}

#: containers libsndfile puts the file's own name in, or writes as two files
#: under one name. They are written where they go.
_BY_NAME = ('MPC2K', 'SD2', 'SVX')


def _replace(source: str, target: str) -> None:
    """``os.replace``, again for a moment on Windows, where a virus scanner or
    an indexer holds a file it has just seen."""
    for wait in (0.01, 0.05, 0.2, None):
        try:
            os.replace(source, target)
            return
        except PermissionError as e:
            if os.name != 'nt':
                raise
            if wait is None:
                raise PermissionError(e.errno, "Permission denied; it may be open in "
                                               "another program") from e
            time.sleep(wait)


def _overwrite(source: str, target: str) -> bool:
    """Copy *source* over what is in *target*. False when that cannot begin,
    and then *target* is as it was: it does not open for writing, or it has to
    become shorter and will not, being mapped into another program's memory.
    """
    size = os.path.getsize(source)
    try:
        f = open(target, 'r+b')                     # 'w' would empty it first
    except OSError:
        return False
    with f:
        if size < os.path.getsize(target):
            try:
                f.truncate(size)
            except OSError:
                return False
        with open(source, 'rb') as whole:
            shutil.copyfileobj(whole, f)
    return True


def _put_in_place(path: Path, container: str, write: Callable[[str], Any]) -> None:
    """Have *write* make *path*: in a temporary file beside it, which takes its
    place when it is whole.

    Until then the file that is at *path* is not touched. libsndfile empties a
    file as it opens it, and only then finds that FLAC takes 8 channels, that
    it has no bitrate for the rate, that the disk is full.

    Windows does not let a file be replaced while another program has it open.
    The finished one is then copied over it, if it opens for writing; a disk
    that fails in the middle of that does damage the file.

    A link stays one, and the file it points to is replaced. What is no
    ordinary file, /dev/null for one, is written as it is.
    """
    target = os.path.realpath(path) if os.path.islink(path) else str(path)
    try:
        there: Optional[os.stat_result] = os.stat(target)
    except OSError:
        there = None
    if container in _BY_NAME or (there is not None and not stat.S_ISREG(there.st_mode)):
        write(target)
        return
    if there is not None and not os.access(target, os.W_OK):
        # replacing it would only ask for the folder's permission
        raise PermissionError(errno.EACCES, os.strerror(errno.EACCES), str(path))
    temporary = os.path.join(os.path.dirname(target), f".{secrets.token_hex(4)}.tmp")
    try:
        write(temporary)
        if there is not None:
            with contextlib.suppress(OSError):
                os.chmod(temporary, stat.S_IMODE(there.st_mode))
        try:
            _replace(temporary, target)
        except PermissionError:
            if os.name != 'nt' or there is None or not _overwrite(temporary, target):
                raise
            os.unlink(temporary)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temporary)
        raise


def save_wave(path: PathLike, wave: np.ndarray, sr: int,
              d_sample: Optional[int] = None, subtype: str = 'PCM_16') -> None:
    """Write *wave* to an audio file via soundfile.

    The container follows the extension, the way soundfile picks it: a
    ``.wav`` path writes WAV, ``.flac`` writes FLAC, and ``.aiff`` or
    ``.aif`` AIFF.

    The file is written beside *path* and replaces it when it is whole, so a
    write that fails leaves the file that was there as it was. The one exception
    is a file that another program has open on Windows, which is overwritten
    with the finished one: if the disk fails in the middle of that, the file is
    damaged.

    :param subtype: soundfile subtype — ``'PCM_16'`` (default), ``'PCM_24'``,
        ``'FLOAT'`` (32-bit float), ... see ``soundfile.available_subtypes()``.
    :param d_sample: optional resample rate before writing.

    Integer input is scaled by its own type's full scale, the way soundfile
    reads such a file back: int16 by ``1 / 32768``, int32 by ``1 / 2**31``,
    and unsigned types around their midpoint, so a 16-bit array is written
    unchanged. Float input is written as it is; ``DOUBLE`` keeps float64. A row
    ``(1, n)`` is one channel.

    A fixed-point *subtype* clips anything beyond ±1.0 to full scale, and every
    ``PCM_*`` subtype is rounded to the nearest step. Both are done here rather
    than left to libsndfile, whose conversion floors -- every sample half a
    step low -- and wraps around unless clipping has been switched on: 1.2
    comes back as -0.8. Whether it has been is up to the soundfile release.

    :raises ValueError: for an *sr* or *d_sample* that is not a positive whole
        rate, and for a *wave* that is neither ``(frames,)`` nor ``(frames,
        channels)`` with 1 to 1024 channels, that is channels first, or that
        holds 64-bit integers.
    :raises AudioIOError: if any sample is NaN or inf, before anything is
        written or any folder created. Neither subtype family survives one:
        ``FLOAT`` keeps it, and an integer subtype writes NaN as a zero that
        reads back as silence, and inf as a full-scale sample.
    """
    sr = _check_rate(sr)
    d_sample = _check_rate(d_sample, 'd_sample', optional=True)
    wave = channels_last(refuse_64_bit(wave, 'save_wave'), 'save_wave')
    if wave.ndim not in (1, 2) or (wave.ndim == 2 and not 1 <= wave.shape[1] <= 1024):
        raise ValueError(f"save_wave() writes (frames,) or (frames, channels) with 1 to "
                         f"1024 channels, got shape {wave.shape}")
    precision = np.float64 if str(subtype).upper() == 'DOUBLE' else np.float32

    if np.issubdtype(wave.dtype, np.integer):
        limits = np.iinfo(wave.dtype)
        if limits.min == 0:                      # unsigned: silence is the middle
            middle = (int(limits.max) + 1) / 2.0
            wave = ((wave.astype(np.float64) - middle) / middle).astype(precision)
        else:
            wave = (wave.astype(np.float64) / -float(limits.min)).astype(precision)
    else:
        wave = wave.astype(precision)

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
        wave = resample(wave, sr, d_sample)        # along the frames, not the channels
        sr = d_sample

    if _is_fixed_point(subtype):
        wave = np.clip(wave, -1.0, 1.0)
        bits = _PCM_BITS.get(str(subtype).upper())
        if bits:
            # Rounded here to the subtype's own step and handed over as the
            # integer type that holds it, shifted to the top bits. libsndfile
            # keeps the top bits of an integer, so nothing is floored on the
            # way out -- PCM_24 and PCM_U8 used to average a third and a half
            # of a step low.
            container = 16 if bits <= 16 else 32
            full = 2.0 ** (bits - 1)
            steps = np.clip(np.round(wave.astype(np.float64) * full), -full, full - 1)
            wave = (steps * 2.0 ** (container - bits)).astype(
                np.int16 if container == 16 else np.int32)

    Path(path).parent.mkdir(exist_ok=True, parents=True)
    # soundfile takes the container from the extension spelled as libsndfile
    # names it, so .aiff writes and .aif, AIFF's other usual extension, did not
    suffix = Path(path).suffix.lower()
    container = _CONTAINERS.get(suffix, suffix[1:].upper())
    try:
        if container not in sf.available_formats():
            raise TypeError(f"there is no format to take from the extension {suffix!r}")
        _put_in_place(Path(path), container, lambda name: sf.write(
            name, wave, sr, subtype=subtype, format=container))
    except Exception as e:
        # the operating system's words and libsndfile's, without the name of
        # the temporary file
        why = getattr(e, 'error_string', None) or getattr(e, 'strerror', None) or str(e)
        raise AudioIOError(f"Could not write audio file: {path} "
                           f"({why or type(e).__name__})") from e


def copy_waves(path: PathLike, wave_: Any) -> List[Path]:
    """Copy wave files into directory *path* and return where they went.

    Note the order: the destination comes first. *wave_* is anything
    :func:`check_input` accepts -- a file, a folder, a ``.txt`` list, or a list
    of those. Files land flat in *path*; a name already taken there gets a free
    one (:func:`check_file`), so two talkers' ``sx133.wav`` both arrive.
    """
    path = Path(path)
    if path.suffix:
        log.warning("Destination is not a directory: %s", path)
        return []
    path.mkdir(exist_ok=True, parents=True)

    files, skipped = check_input(wave_)
    for item in skipped:
        log.warning("Not copied, not an existing .wav file: %s", item)
    copied = []
    for w in tqdm(files, desc=f"[kudio] Copying waves to {path.resolve()}",
                  disable=len(files) < 2):
        target = Path(check_file(path / w.name, rename=True))
        shutil.copy(str(w), target)
        copied.append(target)
    return copied


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
    """Resample / re-encode / trim / normalise every ``.wav`` file under *src*.

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
        ``'raise'`` stops at the first one. A missing package
        (:class:`~kudio.exceptions.DependencyError`, librosa for *sr* or
        *trim_db*) is raised either way: it would fail every file the same way.
    :param root: the folder the output layout is measured from. Every input
        has to be inside it.
    :raises ValueError: before the first file is read, for an *sr* that is not
        a positive whole rate, for *peak* and *lufs* together, or for an
        unknown *on_error*. Met file by file it would fail every file the same
        way.
    """
    sr = _check_rate(sr, optional=True)
    if peak is not None and lufs is not None:
        raise ValueError("pass peak= or lufs=, not both")
    if on_error not in ('collect', 'raise'):
        raise ValueError(f"on_error must be 'collect' or 'raise', got {on_error!r}")

    files, skipped = check_input(src)
    # a dst inside src would be collected on the next run and converted again,
    # one folder deeper each time
    out_abs = _abspath(dst)
    files = [f for f in files if out_abs not in _abspath(f).parents]
    if not files:
        raise AudioIOError(f"no audio files found in {src!r}")

    root = _source_root(src, files, root)
    out_root = Path(dst)
    out_root.mkdir(parents=True, exist_ok=True)

    outputs: List[Path] = []
    # an entry check_input could not use is a failure like any other, not
    # something to drop: a typo in a list used to leave "1/1 written"
    failed: List[Tuple[Path, str]] = [
        (Path(str(item)), "not an existing .wav file") for item in skipped]

    for n, f in enumerate(files, 1):
        try:
            with layout_known():
                y, rate = file_load(f, sr=sr, mono=mono)
                if trim_db is not None:
                    from kudio.effects.silence import trim_silence
                    y = trim_silence(y, top_db=trim_db)[0]
                if y.size == 0:
                    raise AudioIOError("empty after trimming" if trim_db is not None
                                       else "the file holds no samples")
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
        except DependencyError:
            raise
        except Exception as e:
            if on_error == 'raise':
                raise
            failed.append((Path(f), f"{type(e).__name__}: {e}"))
        if progress is not None:
            progress(n, len(files), Path(f))

    return ConvertResult(written=len(outputs), total=len(files) + len(skipped),
                         outputs=outputs, failed=failed)


class LoadAudio:
    """Iterate over the ``.wav`` files found (recursively) in a directory,
    collected the way :func:`check_input` collects them."""

    def __init__(self, waves_dirs: PathLike):
        self.sr_pth = Path(waves_dirs)
        if not self.sr_pth.is_dir():
            raise NotADirectoryError(f"Folder not found: {self.sr_pth.absolute()}")
        self.files = check_input(self.sr_pth)[0]
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
