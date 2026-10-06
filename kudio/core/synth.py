# -*- coding: utf-8 -*-
"""Noisy-speech synthesis: the whole chain between a talker and a file.

A recording is degraded three ways and this class can now put all three in.
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

import itertools
import logging
import math
import shutil
import tempfile
import zlib
import warnings
from functools import partial
from multiprocessing import Pool
from pathlib import Path, PurePath
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

import librosa
import numpy as np
from tqdm import tqdm, trange

from kudio.core.dataset import Pair
from kudio.core.io import (_abspath, _is_fixed_point, _relative, _source_root,
                           check_input, save_wave)
from kudio.exceptions import SynthesisError

__all__ = ['Synthesizer']

log = logging.getLogger(__name__)

#: keeps a file's noise crop out of step with its room, which is drawn from
#: the same per-file seed
_CROP_STREAM = 0x63726F70          # 'crop'


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


class Synthesizer:
    """Mix clean audio with noise at the requested SNRs.

    Parameters
    ----------
    clean, noise :
        Any input accepted by :func:`kudio.check_input` -- a directory, a
        ``.wav`` file, a ``.txt`` manifest with one path per line, or a list
        of those. A list is how a subset is synthesised without copying it
        out first.
    out_path :
        Output directory for the mixed files (default ``mixed``). With
        ``overwrite=True`` :meth:`syn` empties it first -- and refuses to when
        it holds an input file, the working directory or the home directory.
    snr_ratio :
        One or more SNR values in dB.
    rt60 :
        One or more reverberation times in seconds. ``None`` (the default)
        keeps the existing behaviour exactly -- no room, no change to the
        output names, no change to what :meth:`syn` returns. Given, it becomes
        a fourth axis: in ``inc`` mode the output count is multiplied by it.
    drr_db :
        Direct-to-reverberant ratio for the rooms, in dB. Distance in a
        readable unit -- +10 and above is a close microphone, 0 is a metre or
        two, negative is the far end of a hall.
    channel :
        A link for the finished mixture to travel down: a preset name
        (``'telephone'``, ``'voip'``, ``'mu_law'``, ...), a dict of
        :data:`kudio.CHANNEL_DEFAULTS`, or ``None`` for none. Constant across
        the run -- see the module docstring on why it is not an axis. Packet
        loss is seeded per output file, so a rerun drops the same packets.
    write_targets :
        Also write the reverberant-but-clean signal to a ``TARGETS``
        directory beside the mixtures, and record it in :meth:`manifest`.
        Only meaningful with ``rt60=``: with no room the clean file already
        *is* that signal, and a copy of it would be a second name for one
        file.
    root :
        The folder the clean files' layout is measured from; the output
        mirrors that layout. Defaults to *clean* when it is a directory and
        otherwise to the deepest folder the clean files share, so two talkers'
        ``sx133.wav`` stay two files.
    subtype :
        How the files are written: ``'PCM_16'`` (the default), ``'PCM_24'``,
        ``'FLOAT'``, ... A mixture is a sum and can land past full scale. A
        fixed-point subtype clips it there, rounding to the nearest step, and
        :meth:`syn` warns with the count -- the files are in :attr:`clipped`.
        ``'FLOAT'`` keeps the samples as they are.

    Examples
    --------
    >>> syx = Synthesizer('data/train/clean', 'data/train/noise',
    ...                   out_path='data/train/mixed', snr_ratio=(-5, 0, 5))
    >>> syn_list = syx.syn(mode='inc')          # 每筆 clean x noise x SNR
    >>> syn_list = syx.syn(mode='reg')          # 輸出數量與 clean 相同
    >>> syn_list = syx.syn_extra_mode('data/backgroundnoise', background_snr=(5,))

    ``syn`` returns ``[(out_path, clean, noise, snr), ...]``.
    """

    def __init__(self, clean, noise, out_path: str = '',
                 snr_ratio: Union[float, Sequence[float]] = (-5, 0, 5),
                 rt60: Optional[Union[float, Sequence[float]]] = None,
                 drr_db: float = 0.0,
                 channel=None,
                 write_targets: bool = False,
                 root=None,
                 subtype: str = 'PCM_16'):
        for what, given in (('clean', clean), ('noise', noise)):
            if isinstance(given, (str, PurePath)) and not Path(given).exists():
                raise NotADirectoryError(f"{what} input not found: {str(given)!r}")
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
        if not all(np.isfinite(float(v)) for v in self.snrRatio):
            raise ValueError(f"snr_ratio values must be finite, got {self.snrRatio}")

        if isinstance(rt60, str):
            raise TypeError("rt60 can't be str")
        if rt60 is None:
            self.rt60Ratio: List[float] = []
        elif hasattr(rt60, '__iter__'):
            self.rt60Ratio = [float(v) for v in rt60]
        else:
            self.rt60Ratio = [float(rt60)]
        if any(v <= 0 for v in self.rt60Ratio):
            raise ValueError(f"rt60 values must be positive, got {self.rt60Ratio}")
        self.drr_db = float(drr_db)
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

    def wave_input(self):
        return self.cleanWaves, self.noiseWaves

    def wave_syn(self) -> Optional[list]:
        return self.synWavesList

    def waveform(self, is_pool: bool = False):
        """Load the clean and noise file lists into waveform data."""
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
        if mode in ('max', 'extra'):
            raise ValueError("Use 'syn_extra_mode' to run extra mode.")
        if mode not in ('reg', 'regular', 'inc', 'increment'):
            raise ValueError(f"Unknown mode: {mode!r}")
        base = self._base_seed(seed)
        rng = np.random.default_rng(base)
        if desired_sample is not None:
            warnings.warn(
                "Synthesizer.syn(desired_sample=...) is deprecated and will be "
                "removed in a future release; use target_sr=... instead.",
                DeprecationWarning, stacklevel=2)
            if target_sr is None:
                target_sr = desired_sample
        self.target_sr = target_sr
        log.info("==== Audio synthesizing ====")
        if self.noisyDir.is_dir() and any(self.noisyDir.iterdir()):
            if overwrite:
                self._check_clearable()
                log.warning("Output path exists, overwriting -> %s", self.noisyDir)
                shutil.rmtree(self.noisyDir)
            else:
                log.warning("Output path exists -> %s, aborted", self.noisyDir)
                return None
        self.noisyDir.mkdir(parents=True, exist_ok=True)

        # optionally sample N clean files at random
        if len(self.cleanWaves) >= rdn_choice_num > 0:
            picks = rng.choice(len(self.cleanWaves), size=rdn_choice_num,
                               replace=False)
            clnWaves = [self.cleanWaves[k] for k in picks]
        else:
            clnWaves = list(self.cleanWaves)

        # optionally copy the raw clean files alongside the output
        if is_add_clean:
            dir_ = self.noisyDir / "RAW_DATA"
            dir_.mkdir(parents=True, exist_ok=True)
            for cln in clnWaves:
                shutil.copy(cln, dir_ / f"{cln.stem}_raw.wav")

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
            log.info("%d noisy files from %d SNRs and %d noises",
                     clnLen * nosLen * snrLen, snrLen, nosLen)
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
        self._warn_collisions(outWaves, mkdir_parents)

        # Normalised here as well as in the manifest: an SNR list given as
        # np.arange carries np.int64, which `json` cannot serialise, so a
        # caller writing `syn()`'s own tuples out hit what the manifest path
        # was already protected from.
        entries = list(zip(outWaves, clnWaves, nosWaves,
                           [_snr_value(s) for s in snrSeqs]))
        rt_of = dict(zip(outWaves, rt60Seqs))
        silent: Set[Path] = set()
        if is_silence and 0 < p_silence < 1:
            extra = self._silence_share(entries, rng, p_silence)
            silent = {out for out, *_ in extra}
            entries += extra

        # Each file is seeded from the base seed and its own path, so the same
        # request rebuilds the same dataset and two files that share an RT60
        # still get different rooms.
        self.rooms, self.channels, self.targets = {}, {}, {}
        seeds = self._seeds(entries, base, self.noisyDir)
        targets_dir = self.noisyDir / "TARGETS"
        for out, file_seed in seeds.items():
            rt = rt_of.get(out)
            if rt is not None:
                self.rooms[out] = (float(rt), self.drr_db, file_seed)
                if self.write_targets:
                    # mirror the mixture's own layout, so the two names line up
                    self.targets[out] = targets_dir / out.relative_to(self.noisyDir)
            if self.channel:
                # a different derivation from the room's, so a file does not
                # get the same number twice and lose the packets where the
                # room happens to be loudest
                self.channels[out] = (self.channel,
                                      (file_seed * 2654435761) & 0x7FFFFFFF)

        self.synWavesList = entries
        peaks = self.__syn_pool(entries, is_pool=is_pool, subtype=self.subtype,
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

        >>> pairs = syx.manifest()                           # doctest: +SKIP
        >>> kudio.save_manifest("runs/exp/manifest.json", pairs)  # doctest: +SKIP
        """
        if not self.synWavesList:
            return []
        tag = self.channel_tag or None
        return [Pair(noisy=str(out), clean=str(cln), noise=Path(nos).stem,
                     snr_db=_snr_value(snr),
                     rt60=self.rooms.get(Path(out), (None,))[0],
                     channel=tag if Path(out) in self.channels else None,
                     target=(str(self.targets[Path(out)])
                             if Path(out) in self.targets else None))
                for out, cln, nos, snr in self.synWavesList]

    def syn_extra_mode(self, background_path='../data/backgroundnoise',
                       background_snr: Union[float, Sequence[float]] = (5,),
                       seed: Optional[int] = None) -> list:
        """Two-stage mixing: noise x background first, then everything x clean.

        Stage one mixes every noise with every background at *background_snr*
        in a temporary folder. Stage two mixes every clean file with each of
        those mixtures, noises and backgrounds at every SNR, and writes a 2%
        share again as noise alone. The output folder is emptied first, under
        the same guard as :meth:`syn`; *seed* is as there. Rooms and the link
        are not applied in this mode.
        """
        if isinstance(background_snr, str):
            raise TypeError("background_snr can't be str")
        bgSNR = list(background_snr) if hasattr(background_snr, '__iter__') else [background_snr]
        bgNoiseWaves, _ = check_input(background_path)
        if not bgNoiseWaves:
            raise ValueError(f"No wave files found in: {background_path}")
        base = self._base_seed(seed)
        rng = np.random.default_rng(base)
        if self.noisyDir.is_dir() and any(self.noisyDir.iterdir()):
            self._check_clearable(bgNoiseWaves)
            shutil.rmtree(self.noisyDir)
        self.noisyDir.mkdir(parents=True, exist_ok=True)
        self.rooms, self.channels, self.targets = {}, {}, {}
        cleanWaves = self.cleanWaves
        for cln in cleanWaves:
            shutil.copy(cln, self.noisyDir / f"{cln.stem}_n0{cln.suffix}")

        def name(cln, nos, snr) -> str:
            tag = f"n{_snr_text(snr)}" if float(snr) < 0 else _snr_text(snr)
            return f"{cln.stem}_{nos.stem}_{tag}dB.wav"

        with tempfile.TemporaryDirectory(prefix="kudio-extra-") as tmp:
            # stage 1: noise x background noise into a temporary directory,
            # written as float -- an intermediate has no business clipping
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
            self.__syn_pool(stage1, is_pool=False, subtype='FLOAT',
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
            extra = self._silence_share(entries, rng, 0.02)
            entries += extra
            self.synWavesList = entries
            peaks = self.__syn_pool(entries, is_pool=True, subtype=self.subtype,
                                    seeds=self._seeds(entries, base, self.noisyDir),
                                    silent={out for out, *_ in extra})
        message = self._clipping(entries, peaks)
        if message:
            warnings.warn(message, UserWarning, stacklevel=2)
        return self.synWavesList

    # ----------------------------------------------------------- internals

    def _base_seed(self, seed: Optional[int]) -> int:
        if seed is not None and int(seed) < 0:
            raise ValueError(f"seed must be >= 0, got {seed}")
        # drawn from, never written to: a caller's own np.random.seed() still
        # decides it, and nothing the caller draws afterwards moves
        self.seed = int(seed) if seed is not None else \
            int(np.random.randint(0, 2 ** 31 - 1))
        return self.seed

    @staticmethod
    def _seeds(entries, base: int, root) -> Dict[Path, int]:
        return {Path(out): _file_seed(base, out, root) for out, *_ in entries}

    @staticmethod
    def _silence_share(entries: list, rng, p_silence: float) -> list:
        """A *p_silence* share of *entries* again, to be written as noise alone.

        The name follows the same ``clean_noise_snr`` shape as every other
        output, with ``n00`` standing in for the SNR. It used to join the two
        stems with nothing between them, which made ``ab`` + ``c`` and ``a`` +
        ``bc`` one file.
        """
        n = min(len(entries), int(round(len(entries) * p_silence)))
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
            warnings.warn(
                f"{sum(len(v) - 1 for v in clashes.values())} noise-only "
                f"file(s) share a name with another and only one of each is "
                f"written, e.g. {first}. Keep mkdir_parents on so the folder "
                f"layout separates them, or give the inputs distinct stems.",
                UserWarning, stacklevel=3)
        return [found[0] for found in by_path.values()]

    def _check_clearable(self, *more: Sequence) -> None:
        """Refuse to empty the output folder when that would delete more than
        an earlier run of this class put there."""
        target = _abspath(self.noisyDir)
        why = None
        if target == Path(target.anchor):
            why = "it is the root of a filesystem"
        for label, where in (("the working directory", Path.cwd),
                             ("the home directory", Path.home)):
            try:
                place = _abspath(where())
            except (OSError, RuntimeError, KeyError):   # no home to find
                continue
            if why is None and (place == target or target in place.parents):
                why = f"it holds {label}"
        if why is None:
            for f in itertools.chain(self.cleanWaves, self.noiseWaves, *more):
                if target in _abspath(f).parents:
                    why = f"it holds the input {f}"
                    break
        if why is not None:
            raise SynthesisError(
                f"refusing to empty {self.noisyDir} before synthesising: {why}. "
                f"Point out_path= at a folder of its own.")

    @staticmethod
    def _warn_collisions(outWaves: List[Path], mkdir_parents: bool) -> None:
        seen: Set[Path] = set()
        dupes = [p for p in outWaves if p in seen or seen.add(p)]
        if dupes:
            hint = ("" if mkdir_parents else
                    " -- with mkdir_parents=False, clean files of the same name "
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

    def __syn_pool(self, entries: list, *, is_pool: bool, subtype: str,
                   seeds: Optional[Dict[Path, int]] = None,
                   silent: Optional[Set[Path]] = None,
                   rooms: Optional[Dict[Path, tuple]] = None,
                   channels: Optional[Dict[Path, tuple]] = None,
                   targets: Optional[Dict[Path, Path]] = None) -> List[float]:
        """Write *entries*; returns each one's peak, in order."""
        kwargs = dict(rooms=rooms or None, channels=channels or None,
                      targets=targets or None, seeds=seeds or None,
                      silent=silent or None, subtype=subtype)
        sr_ = self.target_sr          # None -> keep each clean file's own rate
        if is_pool and len(entries) > 200:
            fcn = partial(Synthesizer.syn_waves, entries, False, sr_, **kwargs)
            with Pool(8) as pool:
                return pool.map(fcn, trange(len(entries), desc="[kudio] Audio synthesis"))
        return [Synthesizer.syn_waves(wave, False, sr_, **kwargs)
                for wave in tqdm(entries, desc="[kudio] Audio synthesis")]

    @staticmethod
    def syn_waves(synWavesList, isSilence: bool, target_sr: Optional[int],
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
        y_clean, sr_clean = librosa.load(str(clean_file), sr=target_sr)
        y_noise, _ = librosa.load(str(noise_file), sr=sr_clean)

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

        clean_pwr = np.sum(np.abs(y_clean) ** 2) / len(y_clean)
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
