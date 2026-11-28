# -*- coding: utf-8 -*-
"""Noisy-speech synthesis: the whole chain between a talker and a file.

A recording is degraded three ways and this class puts all three in.
Noise is *additive*, a room is *convolutive*, and the channel is neither, so
they are applied in the order the signal actually meets them::

    talker -> room (rt60=) -> + noise (snr_ratio=) -> link (channel=) -> file

Passing ``rt60=`` reverberates the clean signal **before** the noise is mixed
in, which is the order a microphone meets them: the room acts on the talker,
and the noise arrives at the microphone alongside the result. The SNR is
therefore measured against the **reverberant** speech, because that is what is
actually competing with the noise. Passing ``channel=`` runs the finished
mixture through a link -- the wire carries whatever the microphone produced,
noise included, so it comes last.

**The channel is not a fourth axis.** ``snr_ratio`` and ``rt60`` are swept;
``channel`` is not, and it does not appear in the output names. A corpus is
recorded over a phone line or it is not -- that is a property of the corpus,
not a dimension to cross with the others, and four axes already multiply
enough. To compare two links, run the Synthesizer twice into two directories.

The clean file stays the training target, so a model trained on this data is
asked to undo all of it. ``write_targets=True`` also writes the
reverberant-but-clean signal -- the room and nothing else -- so a model can be
asked to remove only the noise and leave the room alone, which is a different
and easier problem. :meth:`manifest` records which is which.
"""
from __future__ import annotations

import contextlib
import itertools
import logging
import math
import operator
import os
import secrets
import shutil
import stat
import tempfile
import time
import zlib
import warnings
from functools import lru_cache, partial
from pathlib import Path, PurePath
from typing import (Any, Dict, Iterable, Iterator, List, NamedTuple, Optional, Sequence, Set,
                    Tuple, Union)

import numpy as np

from kudio._numbers import flag, whole, within
from kudio._progress import tqdm
from kudio._workers import Workers, chunk_of
from kudio.core.dataset import Pair
from kudio.core.io import (PathLike, _abspath, _check_rate, _is_fixed_point, _read_mono,
                           _reason, _relative, _resampled_length, _source_root, _soxr,
                           audio_info, check_input, resample, save_wave)
from kudio.exceptions import AudioIOError, DependencyError, FeatureError, SynthesisError

__all__ = ['Synthesizer']

log = logging.getLogger(__name__)

#: keeps a file's noise crop out of step with its room, which is drawn from
#: the same per-file seed
_CROP_STREAM = 0x63726F70          # 'crop'

#: The SNRs a mixture can be made at, in dB. Under the first the noise is 10^15
#: times the speech and on its way out of what a 32-bit float holds; over the
#: second 10 ** (snr / 10) is out of what a 64-bit one does.
_SNR_RANGE = (-300.0, 3000.0)

#: More frames than a file has. libsndfile gives the largest number there is
#: for a FLAC that was written to a pipe, whose header has none.
_ENDLESS = 2 ** 62

#: An impulse response up to this many samples, 350 s of it at 48 kHz, is not
#: made for the sake of finding out whether it fits in memory.
_LONG_ROOM = 2 ** 24

#: more outputs than this go through a process pool where one is asked for
_POOL_FROM = 200


def _snr_text(snr) -> str:
    """``5`` -> ``'5'``, ``-2.5`` -> ``'2p5'``. No decimal point goes into a
    file name, where it would read as the start of the suffix."""
    value = abs(float(snr))
    return str(int(value)) if value.is_integer() else repr(value).replace('.', 'p')


def _snr_value(snr) -> Union[int, float]:
    """An SNR as the manifest records it: an int when it is one."""
    value = float(snr)
    return int(value) if value.is_integer() else value


def _reg_pairing(n_out: int, n_noise: int, n_snr: int
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """Noise and SNR index for each output of ``'reg'`` mode.

    Cycling the two lists side by side only ever meets ``lcm(n_noise, n_snr)``
    of the ``n_noise * n_snr`` combinations -- 14 noises against 21 SNRs give
    each noise 3 SNRs. Shifting the SNR by one at the end of every such cycle
    walks through the rest, so any ``n_noise * n_snr`` consecutive outputs hold
    every combination exactly once. With coprime lengths the shift is always
    zero and this is the plain cycle it has always been.
    """
    i = np.arange(n_out)
    g = math.gcd(n_noise, n_snr)
    cycle = n_noise * n_snr // g
    return i % n_noise, (i + (i // cycle) % g) % n_snr


def _file_seed(base: int, out, root) -> int:
    """The number an output file draws everything random from: the run's seed
    and where the file goes. The path below *root* rather than the name, since
    a mirrored tree repeats names -- every TIMIT talker reads ``sa1``. For a
    file directly in *root* the two are the same."""
    rel = Path(out).relative_to(root).as_posix()
    return int(base) ^ (zlib.crc32(rel.encode('utf-8')) & 0x7FFFFFFF)


def _load(path, sr: Optional[int], what: str) -> Tuple[np.ndarray, int]:
    """One input file, or a SynthesisError that names it and says what is
    wrong with it: unreadable, or NaN or inf in the samples. A missing soxr
    is not a bad file, so it is passed on as it is."""
    try:
        y, rate = _read_mono(path)
        return (y, rate) if sr is None else (resample(y, rate, sr), sr)
    except DependencyError:
        raise
    except Exception as e:                              # noqa: BLE001
        raise SynthesisError(f"could not read {what} file {path}: "
                             f"{_reason(e, path)}") from e


class _Header(NamedTuple):
    """What an input says about itself: its rate, and its frames, which are None
    where it takes decoding to count them."""
    sr: int
    frames: Optional[int]


def _header(path, what: str) -> _Header:
    """The header of an input file, or the SynthesisError that :func:`_load`
    would raise for it.

    What there is to warn about is said when the samples are read. The caller's
    filters are left as they are, so a warning that is an error for the caller
    is one here, where it can still keep the output folder from being emptied.
    """
    import soundfile as sf
    try:
        with warnings.catch_warnings(record=True):
            info = audio_info(path, peak=False)
            if info.format is None:                     # it is for ffmpeg
                return _Header(info.sr, None)
            with sf.SoundFile(str(path)) as f:
                readable = f.seekable() and f.frames < _ENDLESS
            if not readable:
                # GSM in a WAV, a FLAC written to a pipe: libsndfile opens them
                # and soundfile does not read them to the end
                _read_mono(path)
    except DependencyError:
        raise
    except Exception as e:                              # noqa: BLE001
        cause = e.__cause__ if isinstance(e, AudioIOError) and e.__cause__ else e
        raise SynthesisError(f"could not read {what} file {path}: "
                             f"{_reason(cause, path)}") from e
    # of an MP3 that was cut the header still tells the whole length
    return _Header(info.sr, None if info.format == 'MP3' else info.frames)


def _silent_clean(path) -> SynthesisError:
    return SynthesisError(f"clean file has no signal to set an SNR against: {path}. "
                          f"It is empty or digital silence. Drop it from the clean set.")


def _empty_noise(path) -> SynthesisError:
    return SynthesisError(f"noise file is empty: {path}. Drop it from the noise set.")


def _check_lengths(clean: _Header, noise: _Header, sr: int, cln, nos) -> None:
    """Refuse a pair that the headers show to be too short to mix at *sr*: a
    single sample has no level apart from its mean."""
    for header, what, path in ((clean, 'clean', cln), (noise, 'noise', nos)):
        if header.frames is None:
            continue
        if not header.frames:
            raise _silent_clean(path) if what == 'clean' else _empty_noise(path)
        if _resampled_length(header.frames, header.sr, sr) < 2:
            raise SynthesisError(f"{what} file is a single sample long at {sr} Hz, which "
                                 f"no SNR can be set with: {path}. Drop it from the "
                                 f"{what} set.")


def _check_snr(values: Sequence, name: str) -> None:
    """Refuse an SNR that no mixture has."""
    if not all(np.isfinite(float(v)) for v in values):
        raise ValueError(f"{name} values must be finite, got {list(values)}")
    low, high = _SNR_RANGE
    if not all(low <= float(v) <= high for v in values):
        raise ValueError(f"{name} values must lie between {low:g} and {high:g} dB, got "
                         f"{list(values)}; past that the gain of the noise is no number "
                         f"its samples hold")


def _is_link(path) -> bool:
    """A symbolic link, or a junction of Windows."""
    try:
        found = os.lstat(str(path))
    except OSError:
        return False
    return stat.S_ISLNK(found.st_mode) or (
        os.name == 'nt' and found.st_reparse_tag == stat.IO_REPARSE_TAG_MOUNT_POINT)


def _held(path: str, folder: bool) -> Optional[str]:
    """What keeps Windows from deleting *path*, if anything does."""
    for wait in (0.01, 0.05, 0.2, None):
        try:
            if os.lstat(path).st_file_attributes & stat.FILE_ATTRIBUTE_READONLY:
                return "it is read-only"
            # to the name it has: nothing happens to it, and it asks for the
            # access that deleting it will
            os.rename(path, path)
            return None
        except PermissionError as e:
            # 32 is a sharing violation. A folder is refused for what is open
            # below it as well, which comes up under its own name if it matters.
            if folder and e.winerror != 32:
                return None
            if wait is None:
                return f"{e.strerror or type(e).__name__}; it may be open in another program"
            time.sleep(wait)        # for a virus scanner or an indexer to be done
        except OSError:                           # it is gone, or has no such thing
            return None
    return None


@contextlib.contextmanager
def _scratch_beside(folder: Path) -> Iterator[Path]:
    """An empty folder in the place of *folder* and with a name as long, so that
    a path that is too long below the one is too long below the other."""
    size = len(folder.name)
    for attempt in itertools.count(1):
        scratch = folder.parent / secrets.token_hex(size)[:size]
        try:
            scratch.mkdir()
            break
        except FileExistsError:
            # a name of one letter runs out
            size += attempt > 20
        except OSError as e:
            raise SynthesisError(f"refusing to empty {folder} before synthesising: nothing "
                                 f"can be written beside it ({e.strerror}).") from e
    try:
        yield scratch
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _in_the_way(folder: Path) -> Optional[str]:
    """Why *folder* cannot be deleted as a whole, or None: the first thing in it
    that would stop the deleting half-way."""
    unlisted: List[OSError] = []
    for where, folders, files in os.walk(str(folder), onerror=unlisted.append):
        folders[:] = [f for f in folders if not _is_link(os.path.join(where, f))]
        if os.name == 'nt':
            for path in [os.path.join(where, f) for f in files] + [where]:
                why = _held(path, folder=path == where)
                if why:
                    return f"{path} cannot be deleted ({why})"
        elif os.path.ismount(where):
            return f"{where} is a mount point"
        elif not os.access(where, os.W_OK | os.X_OK):
            return f"what is in {where} cannot be deleted (no permission to write there)"
    if unlisted:
        return f"{unlisted[0].filename} cannot be opened ({unlisted[0].strerror})"
    return None


def _room_fits(sr: int, rt60: float, drr_db: float, seed: int) -> bool:
    """Whether the opening of the room ``rir`` draws already shows that it can
    have *drr_db*. ``rir`` has the last word where it does not: what comes
    later only adds to the reverberant side, so this never says yes to a room
    that ``rir`` refuses, and takes a tenth of the time on a long one.
    """
    from kudio.core.room import _DIRECT_MS
    half = int(round(_DIRECT_MS / 1000.0 * sr))
    whole = int(round(1.5 * rt60 * sr))
    n = min(whole, 16 * (half + 1))
    if n < 2 or whole > _LONG_ROOM:
        return False
    tail = np.random.default_rng(seed).standard_normal(n) * \
        10.0 ** (-3.0 * (np.arange(n) / sr) / rt60)
    inside = float(np.sum(tail[1:half + 1] ** 2))
    outside = float(np.sum(tail[half + 1:] ** 2))
    return outside * 10.0 ** (drr_db / 10.0) > inside


class Synthesizer:
    """Mix clean audio with noise at the requested SNRs.

    >>> syx = Synthesizer('data/train/clean', 'data/train/noise',
    ...                   out_path='data/train/mixed', snr_ratio=(-5, 0, 5))
    >>> syn_list = syx.syn(mode='inc')     # every clean x noise x SNR
    >>> syn_list = syx.syn(mode='reg')     # one output per clean file
    >>> syn_list = syx.syn_extra_mode('data/backgroundnoise', background_snr=(5,))

    ``syn`` returns ``[(out_path, clean, noise, snr), ...]``.

    :param clean: any input :func:`kudio.check_input` accepts -- a directory,
        a ``.wav`` file, a ``.txt`` list with one path per line, or a list of
        those. A list is how a subset is synthesised without copying it out
        first.
    :param noise: the same, for the noise.
    :param out_path: output directory for the mixed files (default
        ``mixed``). With ``overwrite=True`` :meth:`syn` empties it first, and
        refuses to when it is the root of a filesystem or a link, or holds an
        input file, the working directory, the home directory or something the
        system will not delete.
    :param snr_ratio: one or more SNR values in dB, from -300 to 3000.
    :param rt60: one or more reverberation times in seconds. ``None`` (the
        default) puts no room on the speech and leaves it out of the output
        names. Several values are another axis: every output the mode would
        write is written once per room, in ``reg`` mode as much as in ``inc``.
    :param drr_db: direct-to-reverberant ratio for the rooms, in dB. Distance
        in a readable unit -- +10 and above is a close microphone, 0 is a
        metre or two, negative is the far end of a hall.
    :param channel: a link for the finished mixture to travel down: a preset
        name (``'telephone'``, ``'voip'``, ``'mu_law'``, ...), a dict of
        :data:`kudio.CHANNEL_DEFAULTS`, or ``None`` for none. Constant across
        the run -- see the module docstring on why it is not an axis. Packet
        loss is seeded per output file, so a rerun drops the same packets.
    :param write_targets: also write the reverberant-but-clean signal to a
        ``TARGETS`` folder inside *out_path*, and record it in
        :meth:`manifest`. Only meaningful with ``rt60=``: with no room the
        clean file already *is* that signal. Anything that later collects
        ``.wav`` files from *out_path* by folder collects these too.
    :param root: the folder the clean files' layout is measured from; the
        output mirrors that layout. Defaults to *clean* when it is a directory
        and otherwise to the deepest folder the clean files share, so two
        talkers' ``sx133.wav`` stay two files.
    :param subtype: how the files are written: ``'PCM_16'`` (the default),
        ``'PCM_24'``, ``'FLOAT'``, ... A mixture is a sum and can land past
        full scale. A fixed-point subtype clips it there (every ``PCM_*`` width
        rounded to its nearest step), and :meth:`syn` warns with the count --
        the files are in :attr:`clipped`. ``'FLOAT'`` keeps the samples as
        they are.
    """

    def __init__(self, clean: Any, noise: Any, out_path: str = '',
                 snr_ratio: Union[float, Sequence[float]] = (-5, 0, 5),
                 rt60: Optional[Union[float, Sequence[float]]] = None,
                 drr_db: float = 0.0,
                 channel: Optional[Union[str, dict]] = None,
                 write_targets: bool = False,
                 root: Optional[PathLike] = None,
                 subtype: str = 'PCM_16'):
        write_targets = flag(write_targets, 'write_targets', 'Synthesizer', ValueError)
        for what, given in (('clean', clean), ('noise', noise)):
            if isinstance(given, (str, PurePath)) and not Path(given).exists():
                raise FileNotFoundError(f"{what} input not found: {str(given)!r}")
        self.cleanWaves, bad_clean = check_input(clean)
        self.noiseWaves, bad_noise = check_input(noise)
        if not self.cleanWaves:
            raise ValueError(f"No wave files found in clean path: {clean}")
        if not self.noiseWaves:
            raise ValueError(f"No wave files found in noise path: {noise}")
        skipped = bad_clean + bad_noise
        if skipped:
            # a list or a manifest names files one by one, so one that is not
            # there is a mistake worth hearing about, not a file to step over
            warnings.warn(
                f"Synthesizer skipped {len(skipped)} input(s) that are not "
                f"existing .wav files, e.g. {str(skipped[0])!r}",
                UserWarning, stacklevel=2)

        self.noisyDir = Path('mixed' if out_path == '' else out_path)

        if isinstance(snr_ratio, str):
            raise TypeError("snr_ratio can't be str")
        self.snrRatio = list(snr_ratio) if hasattr(snr_ratio, '__iter__') else [snr_ratio]
        if not self.snrRatio:
            raise ValueError("snr_ratio is empty")
        _check_snr(self.snrRatio, 'snr_ratio')

        if isinstance(rt60, str):
            raise TypeError("rt60 can't be str")
        times = [] if rt60 is None else list(rt60) if hasattr(rt60, '__iter__') else [rt60]
        if any(isinstance(v, (bool, np.bool_)) for v in times):
            raise ValueError(f"rt60 is a time in seconds, got {rt60!r}")
        self.rt60Ratio: List[float] = [float(v) for v in times]
        self.drr_db = float(drr_db)
        self._check_room()
        #: room per output file, keyed by the output path rather than held in a
        #: list beside `synWavesList`. Two lists that have to stay in step is
        #: the failure this codebase keeps meeting; a mapping cannot drift.
        self.rooms: Dict[Path, tuple] = {}

        from kudio.effects.channel import channel_spec, channel_tag
        #: the link, normalised once. Empty dict means no channel, and is
        #: falsy, so every downstream check is `if self.channel:`.
        self.channel: dict = channel_spec(channel)
        self.channel_tag: str = channel_tag(self.channel)
        #: (spec, seed) per output file, for the same reason `rooms` is a
        #: mapping: the seed has to reach the worker that writes that file.
        self.channels: Dict[Path, tuple] = {}

        self.write_targets = bool(write_targets)
        if self.write_targets and not self.rt60Ratio:
            warnings.warn(
                "Synthesizer(write_targets=True) without rt60= writes nothing: "
                "with no room the clean file already is the reverberant-clean "
                "signal. Pass rt60=, or drop write_targets.",
                UserWarning, stacklevel=2)
        #: output file -> the reverberant-clean target written beside it
        self.targets: Dict[Path, Path] = {}

        import soundfile as sf
        if not sf.check_format('WAV', str(subtype).upper()):
            raise ValueError(f"subtype {subtype!r} cannot be written to a .wav "
                             f"file; see soundfile.available_subtypes('WAV')")
        self.subtype = str(subtype).upper()
        #: output file -> its peak, for every file of the last run that went
        #: past full scale and was clipped there when written
        self.clipped: Dict[Path, float] = {}
        #: the base every per-file draw of the last run came from. Passed back
        #: as ``syn(seed=...)`` it rebuilds that run, even one given no seed.
        self.seed: Optional[int] = None

        self.synWavesList: Optional[list] = None
        self.clean_dir_ = clean
        #: where the output layout is measured from; ``None`` puts every
        #: output directly in `noisyDir`
        self.clean_root: Optional[Path] = _source_root(clean, self.cleanWaves, root)
        #: target output rate; ``None`` keeps each clean file's own rate
        self.target_sr: Optional[int] = None

    def wave_input(self) -> Tuple[List[Path], List[Path]]:
        return self.cleanWaves, self.noiseWaves

    def wave_syn(self) -> Optional[list]:
        return self.synWavesList

    def waveform(self, is_pool: bool = False
                 ) -> Tuple[List[Tuple[np.ndarray, int]], List[Tuple[np.ndarray, int]]]:
        """Load the clean and noise file lists into waveform data."""
        is_pool = flag(is_pool, 'is_pool', 'Synthesizer.waveform', ValueError)
        from kudio.core.io import load_waves
        return (load_waves(self.cleanWaves, use_pool=is_pool),
                load_waves(self.noiseWaves, use_pool=is_pool))

    def syn(self, mode: str = 'regular',
            mkdir_parents: bool = True,
            is_add_clean: bool = False,
            rdn_choice_num: int = 0,
            overwrite: bool = True,
            is_pool: bool = False,
            is_silence: bool = False,
            p_silence: float = 0.02,
            target_sr: Optional[int] = None,
            desired_sample: Optional[int] = None,
            seed: Optional[int] = None) -> Optional[list]:
        """Generate the noisy dataset.

        mode ``'reg'``/``'regular'``: one output per clean file, the noises and
        SNRs taken in turn -- and every noise x SNR combination comes up once
        in each ``len(noise) * len(snr)`` outputs, whatever the two lengths
        are. mode ``'inc'``/``'increment'``: full cartesian product
        clean x noise x SNR.

        With ``overwrite=True`` the output folder is emptied, and before that
        the run is tried as far as it goes without the samples: the header and
        the length of every input it reads, every room at the rate of its file,
        the packets the link loses in every output, and a sample written beside
        the folder at every rate and under the longest names. Whatever fails
        there raises, and the folder, :attr:`seed`, :attr:`target_sr`,
        :attr:`rooms` and :meth:`manifest` stay as they were.

        What only the samples or the writing can show comes up after the folder
        was emptied: a clean file of digital silence, a noise without variation
        where it was cropped, NaN or inf, damage behind a header, a disk or a
        memory that is full, an input that changes or goes away meanwhile, an
        interrupted run. So does a warning that the caller's filters make an
        error of, unless it is about a header.

        :param rdn_choice_num: synthesise this many clean files, drawn at
            random without replacement, instead of all of them.
        :param is_silence: also write a *p_silence* share of the outputs again
            as noise alone, named ``..._n00.wav``. Only that share: the
            mixtures keep their speech.
        :param target_sr: resample the mixture to this rate. ``None`` (the
            default) keeps each clean file's own rate; the noise is always
            resampled to match its clean partner.
        :param desired_sample: deprecated alias for *target_sr*.
        :param seed: the base of everything random in the run. Each output
            file draws its noise crop, its room and its packet loss from the
            base and its own path, so a rerun rebuilds the same files whether
            or not they went through a process pool. ``None`` draws a base
            from numpy's global RNG -- so seeding that reproduces the run too
            -- and records it in :attr:`seed`. The global RNG is never
            reseeded.
        """
        mkdir_parents = flag(mkdir_parents, 'mkdir_parents', 'Synthesizer.syn', ValueError)
        is_add_clean = flag(is_add_clean, 'is_add_clean', 'Synthesizer.syn', ValueError)
        overwrite = flag(overwrite, 'overwrite', 'Synthesizer.syn', ValueError)
        is_pool = flag(is_pool, 'is_pool', 'Synthesizer.syn', ValueError)
        is_silence = flag(is_silence, 'is_silence', 'Synthesizer.syn', ValueError)
        if mode in ('max', 'extra'):
            raise ValueError("Use 'syn_extra_mode' to run extra mode.")
        if mode not in ('reg', 'regular', 'inc', 'increment'):
            raise ValueError(f"Unknown mode: {mode!r}")
        if desired_sample is not None:
            warnings.warn(
                "Synthesizer.syn(desired_sample=...) is deprecated and will be "
                "removed in a future release; use target_sr=... instead.",
                DeprecationWarning, stacklevel=2)
        # checked before the output folder is emptied, not on the first file
        if target_sr is None:
            target_sr = _check_rate(desired_sample, 'desired_sample', optional=True)
        else:
            target_sr = _check_rate(target_sr, 'target_sr', optional=True)
        self._check_room()
        _check_snr(self.snrRatio, 'snr_ratio')
        self._check_any(self.cleanWaves, self.noiseWaves, self.snrRatio)
        log.info("==== Audio synthesizing ====")
        occupied = self.noisyDir.is_dir() and any(self.noisyDir.iterdir())
        if occupied and not overwrite:
            # a warning rather than a log line: returning None from a
            # call that looked like it ran is easy to miss
            warnings.warn(f"{self.noisyDir} is not empty, so nothing was "
                          f"synthesised. Pass overwrite=True to replace it, "
                          f"or point out_path= somewhere else.",
                          UserWarning, stacklevel=2)
            return None
        if occupied:
            self._check_clearable()

        # From here to the folder being emptied nothing is written, neither to
        # the disk nor to `self`: `seed`, `target_sr` and the rooms name the
        # corpus that is in the folder, and a call refused on the way has to
        # leave all of it as it is.
        base = self._base_seed(seed)
        rng = np.random.default_rng(base)

        if is_silence:
            p_silence = within(p_silence, 'p_silence', 'Synthesizer.syn', low=0, high=1)
        wanted = whole(rdn_choice_num)
        if wanted is None or wanted < 0:
            raise ValueError(f"Synthesizer.syn(): rdn_choice_num is a whole number of clean files, "
                             f"or 0 for all of them, got {rdn_choice_num!r}")
        rdn_choice_num = wanted
        # optionally sample N clean files at random
        if rdn_choice_num > len(self.cleanWaves):
            warnings.warn(f"rdn_choice_num={rdn_choice_num} asks for more clean "
                          f"files than there are ({len(self.cleanWaves)}), so "
                          f"all of them are used.", UserWarning, stacklevel=2)
        if len(self.cleanWaves) >= rdn_choice_num > 0:
            picks = rng.choice(len(self.cleanWaves), size=rdn_choice_num,
                               replace=False)
            clnWaves = [self.cleanWaves[k] for k in picks]
        else:
            clnWaves = list(self.cleanWaves)
        chosen = list(clnWaves)

        clnLen, nosLen, snrLen = len(clnWaves), len(self.noiseWaves), len(self.snrRatio)
        log.info("Clean: %d, Noise: %d, SNR: %s", clnLen, nosLen, self.snrRatio)

        if mode in ('reg', 'regular'):
            log.info("%d SNRs x %d noises -> %d noisy files", snrLen, nosLen, clnLen)
            noise_idx, snr_idx = _reg_pairing(clnLen, nosLen, snrLen)
            nosWaves = [self.noiseWaves[k] for k in noise_idx]
            snrSeqs = [self.snrRatio[k] for k in snr_idx]
            if clnLen < nosLen * snrLen:
                log.info("%d outputs hold %d of the %d noise x SNR combinations",
                         clnLen, clnLen, nosLen * snrLen)
        else:
            log.info("%d noisy files from %d SNRs, %d noises and %d room(s)",
                     clnLen * nosLen * snrLen * max(1, len(self.rt60Ratio)),
                     snrLen, nosLen, max(1, len(self.rt60Ratio)))
            clnWaves = [c for c in clnWaves for _ in range(nosLen * snrLen)]
            nosWaves = [n for n in self.noiseWaves for _ in range(snrLen)] * clnLen
            snrSeqs = list(self.snrRatio) * (clnLen * nosLen)

        # The room is a fourth axis. Repeating the three existing sequences
        # rather than regenerating them keeps `reg` and `inc` meaning exactly
        # what they meant before, with or without rooms.
        rooms = self.rt60Ratio or [None]
        if len(rooms) > 1:
            repeat = len(rooms)
            clnWaves = [c for c in clnWaves for _ in range(repeat)]
            nosWaves = [n for n in nosWaves for _ in range(repeat)]
            snrSeqs = [s for s in snrSeqs for _ in range(repeat)]
            rt60Seqs = list(rooms) * (len(clnWaves) // repeat)
        else:
            rt60Seqs = [rooms[0]] * len(clnWaves)

        layout: Dict[Path, Path] = {}
        outWaves = []
        for cln, nos, snr, rt in zip(clnWaves, nosWaves, snrSeqs, rt60Seqs):
            snr_tag = f"n{_snr_text(snr)}" if float(snr) < 0 else f"{_snr_text(snr)}dB"
            parts = [cln.stem, nos.stem, snr_tag]
            if rt is not None:
                parts.append(f"rt{int(round(rt * 1000))}ms")
            if mkdir_parents:
                if cln not in layout:
                    layout[cln] = _relative(cln, self.clean_root).parent
                outDir = self.noisyDir / layout[cln]
            else:
                outDir = self.noisyDir
            # joined rather than `with_suffix`, which would take '.b_noise_5dB'
            # off 'a.b_noise_5dB' and leave every output of 'a.b.wav' as 'a.wav'
            outWaves.append(outDir / ('_'.join(parts) + '.wav'))

        # Normalised here as well as in the manifest: an SNR list given as
        # np.arange carries np.int64, which `json` cannot serialise, so a
        # caller writing `syn()`'s own tuples out hit what the manifest path
        # was already protected from.
        entries = list(zip(outWaves, clnWaves, nosWaves,
                           [_snr_value(s) for s in snrSeqs]))
        rt_of = dict(zip(outWaves, rt60Seqs))
        entries = self._one_clean_per_name(entries, mkdir_parents)
        self._warn_collisions([out for out, *_ in entries], mkdir_parents)
        silent: Set[Path] = set()
        if is_silence and p_silence > 0:
            extra = self._silence_share(entries, rng, p_silence)
            silent = {out for out, *_ in extra}
            entries += extra

        # Each file is seeded from the base seed and its own path, so the same
        # request rebuilds the same dataset and two files that share an RT60
        # still get different rooms.
        rooms_of, channels_of, targets_of = {}, {}, {}
        seeds = self._seeds(entries, base, self.noisyDir)
        targets_dir = self.noisyDir / "TARGETS"
        for out, file_seed in seeds.items():
            rt = rt_of.get(out)
            if rt is not None:
                rooms_of[out] = (float(rt), self.drr_db, file_seed)
                if self.write_targets:
                    # mirror the mixture's own layout, so the two names line up
                    targets_of[out] = targets_dir / out.relative_to(self.noisyDir)
            if self.channel:
                # a different derivation from the room's, so a file does not
                # get the same number twice and lose the packets where the
                # room happens to be loudest
                channels_of[out] = (self.channel,
                                    (file_seed * 2654435761) & 0x7FFFFFFF)
        # optionally copy the raw clean files alongside the output, mirrored
        # like the mixtures, or two talkers' sa1.wav would be one copy
        raw = [(cln, self.noisyDir / "RAW_DATA" /
                (_relative(cln, self.clean_root).parent if mkdir_parents else Path()) /
                f"{cln.stem}_raw.wav") for cln in (chosen if is_add_clean else ())]

        rates = self._try_settings(entries, rooms_of, channels_of, target_sr)
        with self._workers(is_pool and len(entries) > _POOL_FROM) as pool:
            if occupied:
                self._try_writing([*seeds, *targets_of.values(), *(to for _, to in raw)],
                                  rates, [cln for cln, _ in raw])
                log.warning("Output path exists, overwriting -> %s", self.noisyDir)
                shutil.rmtree(self.noisyDir)
            self.noisyDir.mkdir(parents=True, exist_ok=True)
            self.seed, self.target_sr = base, target_sr
            self.rooms, self.channels, self.targets = rooms_of, channels_of, targets_of

            for cln, to in raw:
                to.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(cln, to)

            self.synWavesList = entries
            self._noise_only = silent
            peaks = self.__syn_pool(entries, pool=pool, sr=target_sr, subtype=self.subtype,
                                    seeds=seeds, silent=silent, rooms=self.rooms,
                                    channels=self.channels, targets=self.targets)
        message = self._clipping(entries, peaks)
        if message:
            warnings.warn(message, UserWarning, stacklevel=2)
        return self.synWavesList

    def manifest(self) -> List[Pair]:
        """The last run as :class:`kudio.Pair` records, rooms included.

        :meth:`syn` returns four-tuples and always has; a fifth element that
        appeared only when ``rt60=`` was passed would be a return shape that
        depends on a keyword, which is the bug this codebase has met most
        often. The room is carried here instead, where it travels with the
        output path it belongs to.

        Each row is a file this run wrote and the clean signal it was built
        from. A clean file that lost its output name to another one is not
        here -- there is no mixture of it to point at -- and :meth:`syn` warns
        when that happens. Nor are the noise-only files ``is_silence=True``
        adds: there is no speech in them, so the clean file is not their
        reference, and they are in what :meth:`syn` returns instead.

        >>> pairs = syx.manifest()                           # doctest: +SKIP
        >>> kudio.save_manifest("runs/exp/manifest.json", pairs)  # doctest: +SKIP
        """
        if not self.synWavesList:
            return []
        tag = self.channel_tag or None
        # forward slashes, as save_manifest writes them, so a manifest read
        # back compares equal to the one that was saved
        return [Pair(noisy=Path(out).as_posix(), clean=Path(cln).as_posix(),
                     noise=Path(nos).stem,
                     snr_db=_snr_value(snr),
                     rt60=self.rooms.get(Path(out), (None,))[0],
                     channel=tag if Path(out) in self.channels else None,
                     target=(self.targets[Path(out)].as_posix()
                             if Path(out) in self.targets else None))
                for out, cln, nos, snr in self.synWavesList
                if Path(out) not in getattr(self, '_noise_only', ())]

    def syn_extra_mode(self, background_path: Any = '../data/backgroundnoise',
                       background_snr: Union[float, Sequence[float]] = (5,),
                       seed: Optional[int] = None,
                       is_pool: bool = True) -> list:
        """Two-stage mixing: noise x background first, then everything x clean.

        Stage one mixes every noise with every background at *background_snr*
        in a temporary folder. Stage two mixes every clean file with each of
        those mixtures, noises and backgrounds at every SNR, and writes a 2%
        share again as noise alone. The output folder is emptied between the
        two, under the same guard as :meth:`syn` and once stage two has been
        tried as there; *seed* is as there too. Rooms, the link and resampling
        are not applied in this mode, and the output is written flat, one
        folder for everything.

        Names carry ``dB`` on both signs, ``n5dB`` and ``5dB``, where
        :meth:`syn` writes ``n5``. Corpora on disk already use both spellings,
        so neither changes; match on the manifest, which holds each SNR as a
        number, rather than on the name.

        :param is_pool: write stage two through a process pool once it holds
            more than 200 files. A pool needs the calling script's top level
            behind ``if __name__ == '__main__':`` on Windows and macOS.
        """
        is_pool = flag(is_pool, 'is_pool', 'Synthesizer.syn_extra_mode', ValueError)
        if isinstance(background_snr, str):
            raise TypeError("background_snr can't be str")
        bgSNR = list(background_snr) if hasattr(background_snr, '__iter__') else [background_snr]
        _check_snr(bgSNR, 'background_snr')
        _check_snr(self.snrRatio, 'snr_ratio')
        bgNoiseWaves, _ = check_input(background_path)
        if not bgNoiseWaves:
            raise ValueError(f"No wave files found in: {background_path}")
        occupied = self.noisyDir.is_dir() and any(self.noisyDir.iterdir())
        if occupied:
            self._check_clearable(bgNoiseWaves)
        base = self._base_seed(seed)
        rng = np.random.default_rng(base)
        cleanWaves = self.cleanWaves
        copies = [(cln, self.noisyDir / f"{cln.stem}_n0{cln.suffix}") for cln in cleanWaves]
        # every clean file meets every noise and every background
        rates = {_header(f, what).sr
                 for what, files in (('clean', cleanWaves), ('noise', self.noiseWaves),
                                     ('background', bgNoiseWaves)) for f in files}
        if len(rates) > 1:
            _soxr()                              # asked for before anything is emptied

        def name(cln, nos, snr) -> str:
            tag = f"n{_snr_text(snr)}" if float(snr) < 0 else _snr_text(snr)
            return f"{cln.stem}_{nos.stem}_{tag}dB.wav"

        with tempfile.TemporaryDirectory(prefix="kudio-extra-") as tmp:
            # stage 1: noise x background noise into a temporary directory,
            # written as float -- an intermediate has no business clipping. It
            # comes before the output folder is emptied, and so do its errors.
            temporaryPath = Path(tmp)
            fullCleanWaves = [n for n in self.noiseWaves
                              for _ in range(len(bgNoiseWaves) * len(bgSNR))]
            fullNoiseWaves = [b for b in bgNoiseWaves
                              for _ in range(len(bgSNR))] * len(self.noiseWaves)
            fullSnrSequence = bgSNR * (len(self.noiseWaves) * len(bgNoiseWaves))
            fullOutPaths = [temporaryPath / name(cln, nos, snr) for cln, nos, snr
                            in zip(fullCleanWaves, fullNoiseWaves, fullSnrSequence)]
            stage1 = list(zip(fullOutPaths, fullCleanWaves, fullNoiseWaves,
                              fullSnrSequence))
            self._try_settings(stage1, {}, {}, None)
            # no resampling in this mode, whatever an earlier syn() asked for
            self.__syn_pool(stage1, pool=None, sr=None, subtype='FLOAT',
                            seeds=self._seeds(stage1, base, temporaryPath))

            # stage 2: clean x (stage-1 output + noise + background noise)
            noiseWaves = fullOutPaths + list(self.noiseWaves) + list(bgNoiseWaves)
            fullCleanWaves = [c for c in cleanWaves
                              for _ in range(len(noiseWaves) * len(self.snrRatio))]
            fullNoiseWaves = [n for n in noiseWaves
                              for _ in range(len(self.snrRatio))] * len(cleanWaves)
            fullSnrSequence = list(self.snrRatio) * (len(cleanWaves) * len(noiseWaves))
            fullOutPaths = [self.noisyDir / name(cln, nos, snr) for cln, nos, snr
                            in zip(fullCleanWaves, fullNoiseWaves, fullSnrSequence)]
            entries = list(zip(fullOutPaths, fullCleanWaves, fullNoiseWaves,
                               fullSnrSequence))
            self._check_any(entries)
            self._warn_collisions([out for out, *_ in entries], False)
            extra = self._silence_share(entries, rng, 0.02, flat=True)
            entries += extra
            seeds = self._seeds(entries, base, self.noisyDir)
            rates = self._try_settings(entries, {}, {}, None)
            with self._workers(is_pool and len(entries) > _POOL_FROM) as pool:
                if occupied:
                    self._try_writing([*seeds, *(to for _, to in copies)], rates, cleanWaves)
                    shutil.rmtree(self.noisyDir)
                self.noisyDir.mkdir(parents=True, exist_ok=True)
                # after the guards, like `syn`: a run that refuses to clear the
                # folder must not take the seed that names the corpus in it
                self.seed, self.target_sr = base, None
                self.rooms, self.channels, self.targets = {}, {}, {}
                for cln, to in copies:
                    shutil.copy(cln, to)
                self.synWavesList = entries
                self._noise_only = {out for out, *_ in extra}
                peaks = self.__syn_pool(entries, pool=pool, sr=None, subtype=self.subtype,
                                        seeds=seeds, silent=self._noise_only)
        message = self._clipping(entries, peaks)
        if message:
            warnings.warn(message, UserWarning, stacklevel=2)
        return self.synWavesList

    # ----------------------------------------------------------- internals

    @staticmethod
    def _base_seed(seed: Optional[int]) -> int:
        # drawn from, never written to: a caller's own np.random.seed() still
        # decides it, and nothing the caller draws afterwards moves
        if seed is None:
            return int(np.random.randint(0, 2 ** 31 - 1))
        try:
            base = int(operator.index(seed))          # a long one keeps all its digits
        except TypeError:
            base = whole(seed)
        if base is None or base < 0:
            raise ValueError(f"seed must be >= 0, a whole number, got {seed!r}")
        return base

    @staticmethod
    def _check_any(*lists: Sequence) -> None:
        """Refuse a run that would empty the folder and write nothing. Nothing
        was empty when it was passed in, and an attribute can be since."""
        if not all(len(given) for given in lists):
            raise SynthesisError("there is nothing to synthesise: cleanWaves, noiseWaves "
                                 "or snrRatio is empty")

    def _check_room(self) -> None:
        """Refuse a room that is none at any rate."""
        if not all(0 < float(v) < math.inf for v in self.rt60Ratio):
            raise ValueError(f"rt60 values must be positive and finite, got {self.rt60Ratio}")
        if not math.isfinite(float(self.drr_db)):
            raise ValueError(f"drr_db must be a finite number, got {self.drr_db}")

    def _try_settings(self, entries: list, rooms: Dict[Path, tuple],
                      channels: Dict[Path, tuple], target_sr: Optional[int]) -> Set[int]:
        """Go through a plan before its folder is emptied, as far as that is
        possible without the samples; returns the rates it writes.

        Which settings work is for the functions that use them to say, and
        some of it comes down to the rate of a file, its length and the seed of
        an output: whether the direct sound can be that far below the tail,
        whether the telephone band fits under Nyquist, whether the packets that
        survive keep the runs of lost ones apart.
        """
        from kudio.core.room import rir
        from kudio.effects.channel import _check_loss, apply_channel, channel_spec
        headers: Dict[str, _Header] = {}
        for _, cln, nos, *_ in entries:
            for f, what in ((cln, 'clean'), (nos, 'noise')):
                if str(f) not in headers:
                    headers[str(f)] = _header(f, what)

        mixed: Set[int] = set()
        resampled = False
        for out, cln, nos, *_ in entries:
            own, noise = headers[str(cln)], headers[str(nos)]
            sr = own.sr if target_sr is None else target_sr
            _check_lengths(own, noise, sr, cln, nos)
            mixed.add(sr)
            resampled = resampled or {own.sr, noise.sr} != {sr}
            room = rooms.get(Path(out))
            if room is not None:
                rt60_s, drr_db, room_seed = room
                if not _room_fits(sr, rt60_s, drr_db, room_seed):
                    rir(sr, rt60=rt60_s, drr_db=drr_db, seed=room_seed)
        if resampled:
            _soxr()                              # asked for before anything is emptied

        spec = channel_spec(self.channel) if channels else {}
        if spec:
            # the band, the codec and the bits go by the rate
            for sr in sorted(mixed):
                noise = 0.1 * np.random.default_rng(0).standard_normal(max(sr, 1000))
                apply_channel(noise, sr, dict(spec, loss=0.0), seed=0)
        if spec and float(spec['loss']) > 0.5 and spec['burst'] is not None:
            # The packets go by the length and the seed of every output. As
            # long as a half of them or less are lost, any runs can be kept
            # apart.
            @lru_cache(maxsize=None)
            def frames(cln: str) -> int:
                own = headers[cln]
                if own.frames is None:
                    with warnings.catch_warnings(record=True):
                        return len(_load(cln, target_sr, 'clean')[0])
                return _resampled_length(own.frames, own.sr, target_sr or own.sr)

            for out, cln, *_ in entries:
                try:
                    _check_loss(frames(str(cln)), target_sr or headers[str(cln)].sr,
                                spec, channels[Path(out)][1])
                except FeatureError as e:
                    raise FeatureError(f"{out}: {e}") from e
        return mixed

    def _try_writing(self, files: Sequence[Path], rates: Iterable[int],
                     copied: Sequence[Path] = ()) -> None:
        """Write a sample at every one of *rates*, and one under the longest
        names of *files*, as the run will write them but in a folder beside the
        output one; then see that the output folder can be deleted. *copied*
        are the inputs that are copied as they are.
        """
        taken = set(files)
        for path in files:
            for folder in path.parents:
                if folder in taken:
                    raise SynthesisError(f"{folder} would be a file and the folder of "
                                         f"{path}. Give the clean files other names.")
        for path in copied:
            try:
                with open(str(path), 'rb'):
                    pass
            except OSError as e:
                raise SynthesisError(f"could not read clean file {path}: "
                                     f"{_reason(e, path)}") from e

        def long(text: str) -> int:
            # Windows counts characters, the others bytes
            return len(text) if os.name == 'nt' else len(os.fsencode(text))

        target = _abspath(self.noisyDir)
        names = [path.relative_to(self.noisyDir) for path in files]
        longest = max(names, key=lambda p: long(str(p)))
        widest = max(names, key=lambda p: max(long(part) for part in p.parts))
        with _scratch_beside(target) as scratch:
            for name, sr in [(widest, min(rates))] + [(longest, sr) for sr in sorted(rates)]:
                try:
                    save_wave(scratch / name, np.zeros(1), sr, subtype=self.subtype)
                except AudioIOError as e:
                    raise AudioIOError(str(e).replace(str(scratch),
                                                      str(self.noisyDir))) from e
                except OSError as e:                # of a folder on the way to it
                    raise AudioIOError(f"Could not write audio file: {self.noisyDir / name} "
                                       f"({e.strerror or type(e).__name__})") from e

        why = _in_the_way(target)
        if why is not None:
            raise SynthesisError(f"refusing to empty {self.noisyDir} before synthesising: "
                                 f"{why}.")

    @staticmethod
    def _seeds(entries, base: int, root) -> Dict[Path, int]:
        return {Path(out): _file_seed(base, out, root) for out, *_ in entries}

    @staticmethod
    def _silence_share(entries: list, rng, p_silence: float,
                       flat: bool = False) -> list:
        """A *p_silence* share of *entries* again, to be written as noise alone.

        The name follows the same ``clean_noise_snr`` shape as every other
        output, with ``n00`` standing in for the SNR, so ``ab`` + ``c`` and
        ``a`` + ``bc`` stay two files. At least one is written: asked for, a
        share that rounds to none on a small run is a share nobody gets.
        """
        n = min(len(entries), max(1, int(round(len(entries) * p_silence)))) \
            if entries else 0
        chosen: Dict[tuple, tuple] = {}
        for k in rng.choice(len(entries), size=n, replace=False):
            out, cln, nos, _ = entries[k]
            path = Path(out).parent / \
                f"{Path(cln).stem}_{Path(nos).stem}_n00.wav"
            # Two SNRs of one clean x noise pair are the same noise-only file,
            # and one of them is the right answer. Keyed on the pair rather
            # than on the path so that this collapses only what it means to.
            chosen.setdefault((str(cln), str(nos)), (path, cln, nos, 0))

        # ...which leaves the case it never meant to collapse: two different
        # pairs whose stems land on one name. Possible with mkdir_parents=False
        # over a tree, and worth saying rather than dropping one of them.
        by_path: Dict[Path, list] = {}
        for entry in chosen.values():
            by_path.setdefault(entry[0], []).append(entry)
        clashes = {path: found for path, found in by_path.items()
                   if len(found) > 1}
        if clashes:
            first = next(iter(clashes))
            advice = ("Give the inputs distinct stems: this mode writes "
                      "everything into one folder." if flat else
                      "Keep mkdir_parents on so the folder layout separates "
                      "them, or give the inputs distinct stems.")
            warnings.warn(
                f"{sum(len(v) - 1 for v in clashes.values())} noise-only "
                f"file(s) share a name with another and only one of each is "
                f"written, e.g. {first}. {advice}",
                UserWarning, stacklevel=3)
        return [found[0] for found in by_path.values()]

    def _check_clearable(self, *more: Sequence) -> None:
        """Refuse to empty the output folder when that would delete more than
        an earlier run of this class put there."""
        @lru_cache(maxsize=None)
        def real(folder: str) -> Path:
            return Path(os.path.realpath(folder))

        target = _abspath(self.noisyDir)
        # the two names of it, and of what may be in it: as they are spelled,
        # and with every link on the way followed
        targets = {target, real(str(target))}

        def held(*places: Path) -> bool:
            return any(t == p or t in p.parents for t in targets for p in places)

        why = None
        if any(t == Path(t.anchor) for t in targets):
            why = "it is the root of a filesystem"
        elif _is_link(target):
            why = f"it is a link to {real(str(target))}"
        for label, where in (("the working directory", Path.cwd),
                             ("the home directory", Path.home)):
            try:
                place = _abspath(where())
            except (OSError, RuntimeError, KeyError):   # no home to find
                continue
            if why is None and held(place, real(str(place))):
                why = f"it holds {label}"
        if why is None:
            for f in itertools.chain(self.cleanWaves, self.noiseWaves, *more):
                path = _abspath(f)
                places = [path.parent, real(str(path.parent))]
                if os.path.islink(path):
                    places.append(real(str(path)).parent)
                if held(*places):
                    why = f"it holds the input {f}"
                    break
        if why is not None:
            raise SynthesisError(
                f"refusing to empty {self.noisyDir} before synthesising: {why}. "
                f"Point out_path= at a folder of its own.")

    @staticmethod
    def _one_clean_per_name(entries: List, mkdir_parents: bool) -> List:
        """*entries*, with every output name left to a single clean file.

        Flatten a tree and two talkers' ``sx133.wav`` ask for one output
        name. One mixture is all that can be at that name, so a row for each
        clean file would leave :meth:`manifest` offering one talker's
        recording as the clean reference for the other talker's mixture --
        a pair that was never written, and one a training loop has no way of
        telling from a real one. The first clean file to ask for a name keeps
        it and the rest are dropped here, before anything is synthesised, so
        the manifest says exactly what is on disk.

        Rows that ask for the same name with the *same* clean file -- an SNR
        given twice over, two noise files sharing a stem -- all describe the
        file that lands correctly, and stay. The dataset is a row short per
        clean file dropped, which is why this is loud about it.
        """
        claimed: Dict[Path, str] = {}
        kept, lost = [], []
        for entry in entries:
            out, cln = Path(entry[0]), str(entry[1])
            if claimed.setdefault(out, cln) == cln:
                kept.append(entry)
            else:
                lost.append(entry)
        if lost:
            out, cln = Path(lost[0][0]), str(lost[0][1])
            hint = ("" if mkdir_parents else
                    " -- with mkdir_parents=False, clean files of the same name "
                    "in different folders land on one output name")
            warnings.warn(
                f"{len(lost)} clean file(s) are not synthesised: another clean "
                f"file asked for the same output name first, and only one "
                f"mixture can be written there, so these are left out of the "
                f"run and out of manifest(). {cln} wanted {out}, which belongs "
                f"to {claimed[out]}{hint}. Keep mkdir_parents on, or give the "
                f"clean files distinct stems, to get a file for each of them.",
                UserWarning, stacklevel=3)
        return kept

    @staticmethod
    def _warn_collisions(outWaves: List[Path], mkdir_parents: bool) -> None:
        """Names still asked for twice after `_one_clean_per_name` has run.

        What reaches here is one clean file written twice over, so the file
        that lands is the right one for it either way -- worth saying, not
        worth dropping.
        """
        seen: Set[Path] = set()
        dupes = [p for p in outWaves if p in seen or seen.add(p)]
        if dupes:
            hint = ("" if mkdir_parents else
                    " -- with mkdir_parents=False, noise files of the same name "
                    "in different folders land on one output name")
            warnings.warn(
                f"{len(dupes)} output(s) share a name with another and will "
                f"overwrite it, e.g. {dupes[0]}{hint}", UserWarning, stacklevel=3)

    def _clipping(self, entries: list, peaks: Sequence[float]) -> Optional[str]:
        """Record the files that went past full scale; the warning, if any."""
        self.clipped = {}
        if _is_fixed_point(self.subtype):
            self.clipped = {Path(out): float(peak)
                            for (out, *_), peak in zip(entries, peaks) if peak > 1.0}
        if not self.clipped:
            return None
        worst = max(self.clipped.values())
        return (f"{len(self.clipped)} of {len(entries)} file(s) went past full "
                f"scale and were clipped when written as {self.subtype} (worst "
                f"peak {worst:.2f}, {20 * math.log10(worst):+.1f} dBFS). Lower "
                f"the clean level or write subtype='FLOAT'; they are listed in "
                f".clipped.")

    @staticmethod
    @contextlib.contextmanager
    def _workers(wanted: bool) -> Iterator[Any]:
        """A process pool if one is *wanted*, else None. It has answered once by
        the time it is handed out: a process that may not have children, or a
        script without its ``if __name__ == '__main__':``, gets no further."""
        if not wanted:
            yield None
            return
        with Workers(8, 'Synthesizer', SynthesisError) as pool:
            pool.answer()
            yield pool

    @staticmethod
    def __syn_pool(entries: list, *, pool: Any, sr: Optional[int], subtype: str,
                   seeds: Optional[Dict[Path, int]] = None,
                   silent: Optional[Set[Path]] = None,
                   rooms: Optional[Dict[Path, tuple]] = None,
                   channels: Optional[Dict[Path, tuple]] = None,
                   targets: Optional[Dict[Path, Path]] = None) -> List[float]:
        """Write *entries*; returns each one's peak, in order. With *sr* None
        each clean file keeps its own rate."""
        kwargs = dict(rooms=rooms or None, channels=channels or None,
                      targets=targets or None, seeds=seeds or None,
                      silent=silent or None, subtype=subtype)
        if pool is not None:
            fcn = partial(Synthesizer.syn_waves, entries, False, sr, **kwargs)
            n = len(entries)
            return list(tqdm(pool.imap(fcn, range(n), chunk_of(n, pool.n)), total=n,
                             desc="[kudio] Audio synthesis"))
        return [Synthesizer.syn_waves(wave, False, sr, **kwargs)
                for wave in tqdm(entries, desc="[kudio] Audio synthesis")]

    @staticmethod
    def syn_waves(synWavesList: Union[tuple, Sequence[tuple]], isSilence: bool,
                  target_sr: Optional[int],
                  num: Optional[int] = None,
                  rooms: Optional[Dict[Path, tuple]] = None,
                  channels: Optional[Dict[Path, tuple]] = None,
                  targets: Optional[Dict[Path, Path]] = None,
                  seeds: Optional[Dict[Path, int]] = None,
                  silent: Optional[Set[Path]] = None,
                  subtype: str = 'PCM_16') -> float:
        """Mix one ``(save_dir, clean_file, noise_file, snr)`` entry to disk.

        With ``num`` given, ``synWavesList`` is the full list and ``num`` its
        index (multiprocessing form); otherwise it is a single entry.

        *rooms* maps an output path to ``(rt60, drr_db, seed)``. The room is
        applied to the clean signal **before** the noise, and the SNR is then
        measured against the reverberant result -- the order a microphone meets
        them, and the only one in which the requested SNR is the SNR you get.

        *channels* maps an output path to ``(spec, seed)`` and is applied to
        the finished mixture, since the wire carries the noise too. *targets*
        maps an output path to where the reverberant-clean signal goes; it is
        written before the noise arrives and before the channel, so it is the
        speech as the room left it and nothing else.

        *seeds* maps an output path to the seed its noise crop is drawn from,
        so the crop does not depend on which worker wrote the file; a path
        missing from it draws from numpy's global RNG. Outputs in *silent* are
        written as the noise alone; ``isSilence=True`` does that to all.

        Returns the largest absolute sample written, mixture or target. Past
        1.0, a fixed-point *subtype* clipped it.
        """
        save_dir, clean_file, noise_file, snr = \
            synWavesList if num is None else synWavesList[num]
        out = Path(save_dir)
        # target_sr=None keeps the clean file's native rate; the noise is then
        # resampled to whatever the clean file turned out to be, so the two are
        # never mixed at different rates
        y_clean, sr_clean = _load(clean_file, target_sr, "clean")
        y_noise, _ = _load(noise_file, sr_clean, "noise")
        if not len(y_noise):
            raise _empty_noise(noise_file)

        room = (rooms or {}).get(out)
        if room is not None:
            from kudio.core.room import apply_rir, rir
            rt60_s, drr_db, room_seed = room
            # trim=True keeps the length, so the clean file stays a
            # sample-aligned target; the direct path sits at sample 0 of the
            # impulse response, so nothing is delayed either
            y_clean = apply_rir(
                y_clean, rir(int(sr_clean), rt60=rt60_s, drr_db=drr_db,
                             seed=room_seed))

        peak = 0.0
        target_path = (targets or {}).get(out)
        if target_path is not None:
            peak = float(np.max(np.abs(y_clean))) if y_clean.size else 0.0
            save_wave(target_path, y_clean, int(sr_clean), subtype=subtype)

        clean_pwr = np.sum(np.abs(y_clean) ** 2) / len(y_clean) if len(y_clean) else 0.0
        if not clean_pwr > 0.0:
            # the noise is scaled against the speech, so silent speech made a
            # silent mixture: a file of zeros labelled with an SNR
            raise _silent_clean(clean_file)
        if len(y_noise) < len(y_clean):
            repeats = len(y_clean) // len(y_noise) + 1
            y_noise = np.tile(y_noise, repeats)[:len(y_clean)]
        else:  # random crop (no-op when both are the same length)
            span = len(y_noise) - len(y_clean) + 1
            crop_seed = (seeds or {}).get(out)
            if crop_seed is None:
                ind = np.random.randint(0, span)
            else:
                ind = int(np.random.default_rng([crop_seed, _CROP_STREAM])
                          .integers(0, span))
            y_noise = y_noise[ind:ind + len(y_clean)]

        y_noise = y_noise - np.mean(y_noise)
        spread = float(np.std(y_noise))
        if not spread > 0.0:
            # Digital silence, or a constant: there is no scaling that puts
            # this at the requested SNR, and dividing by it gave every sample
            # NaN, which int16 wrote as zero -- a mixture with no speech in it
            # and nothing said. One truncated download in a noise corpus was
            # enough.
            raise SynthesisError(
                f"noise file has no variation and cannot be scaled to "
                f"{_snr_value(snr)} dB: {noise_file}. It is digital silence or "
                f"a constant level (std 0 after removing the mean), so there "
                f"is no gain that makes it a noise floor. Drop the file from "
                f"the noise set, or trim it to the part that has signal in it.")
        noise_variance = clean_pwr / (10 ** (snr / 10))
        noise = np.sqrt(noise_variance) * y_noise / spread
        if not np.all(np.isfinite(noise)):
            # belt and braces: a near-silent file scales by a vast factor, and
            # a mixture of inf or NaN must never reach the disk as zeros
            raise SynthesisError(
                f"scaling {noise_file} to {_snr_value(snr)} dB against "
                f"{clean_file} produced values that are not finite. The noise "
                f"file is almost certainly degenerate -- check its level.")

        y_noisy = noise if isSilence or out in (silent or ()) else y_clean + noise

        link = (channels or {}).get(out)
        if link is not None:
            from kudio.effects.channel import apply_channel
            spec, link_seed = link
            # the mixture, not the speech: a wire carries whatever reached the
            # microphone, and the noise reached it too
            y_noisy = apply_channel(y_noisy, int(sr_clean), spec,
                                    seed=link_seed)

        if y_noisy.size:
            peak = max(peak, float(np.max(np.abs(y_noisy))))
        save_wave(out, y_noisy, int(sr_clean), subtype=subtype)
        return peak
