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

import logging
import shutil
import zlib
import warnings
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm, trange

from kudio.core.dataset import Pair
from kudio.core.io import check_input

__all__ = ['Synthesizer']

log = logging.getLogger(__name__)


class Synthesizer:
    """Mix clean audio with noise at the requested SNRs.

    Parameters
    ----------
    clean, noise :
        Any input accepted by :func:`kudio.check_input` (directory, file list,
        ``.txt`` manifest, ...).
    out_path :
        Output directory for the mixed files (default ``mixed``).
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
                 snr_ratio: Union[int, Sequence[int]] = (-5, 0, 5),
                 rt60: Optional[Union[float, Sequence[float]]] = None,
                 drr_db: float = 0.0,
                 channel=None,
                 write_targets: bool = False):
        if not (Path(clean).is_dir() and Path(noise).is_dir()):
            raise NotADirectoryError(f"Input path error: {clean!r} / {noise!r}")
        self.cleanWaves, _ = check_input(clean)
        self.noiseWaves, _ = check_input(noise)
        if not self.cleanWaves:
            raise ValueError(f"No wave files found in clean path: {clean}")
        if not self.noiseWaves:
            raise ValueError(f"No wave files found in noise path: {noise}")

        self.noisyDir = Path('mixed' if out_path == '' else out_path)

        if isinstance(snr_ratio, str):
            raise TypeError("snr_ratio can't be str")
        self.snrRatio = list(snr_ratio) if hasattr(snr_ratio, '__iter__') else [snr_ratio]

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

        self.synWavesList: Optional[list] = None
        self.clean_dir_ = clean
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

        mode ``'reg'``/``'regular'``: cycle noises & SNRs over the clean list
        (output count == clean count). mode ``'inc'``/``'increment'``: full
        cartesian product clean x noise x SNR.

        :param target_sr: resample the mixture to this rate. ``None`` (the
            default) keeps each clean file's own rate; the noise is always
            resampled to match its clean partner.
        :param desired_sample: deprecated alias for *target_sr*.
        :param seed: seed the RNG for reproducible noise cropping / sampling,
            and the base from which each room is derived. Rooms are seeded per
            output file rather than from the global RNG, so they come out the
            same whether or not the work went through a process pool -- which
            the noise cropping, drawn from the global RNG, does not.
        """
        if seed is not None:
            np.random.seed(seed)
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
                log.warning("Output path exists, overwriting -> %s", self.noisyDir)
                shutil.rmtree(self.noisyDir)
            else:
                log.warning("Output path exists -> %s, aborted", self.noisyDir)
                return None
        self.noisyDir.mkdir(parents=True, exist_ok=True)

        # optionally sample N clean files at random
        clnWaves = [
            np.random.choice(self.cleanWaves) for _ in range(rdn_choice_num)
        ] if len(self.cleanWaves) >= rdn_choice_num > 0 else self.cleanWaves

        # optionally copy the raw clean files alongside the output
        if is_add_clean:
            dir_ = self.noisyDir / "RAW_DATA"
            dir_.mkdir(parents=True, exist_ok=True)
            for cln in clnWaves:
                shutil.copy(cln, (dir_ / f"{cln.stem}_raw").with_suffix('.wav'))

        clnLen, nosLen, snrLen = len(clnWaves), len(self.noiseWaves), len(self.snrRatio)
        log.info("Clean: %d, Noise: %d, SNR: %s", clnLen, nosLen, self.snrRatio)

        if mode in ('reg', 'regular'):
            log.info("%d SNRs x %d noises -> %d noisy files", snrLen, nosLen, clnLen)
            nosWaves = np.tile(self.noiseWaves, clnLen // nosLen + 1)[:clnLen]
            snrSeqs = np.tile(self.snrRatio, clnLen // snrLen + 1)[:clnLen]
        elif mode in ('inc', 'increment'):
            log.info("%d noisy files from %d SNRs and %d noises",
                     clnLen * nosLen * snrLen, snrLen, nosLen)
            clnWaves = np.repeat(clnWaves, nosLen * snrLen)
            nosWaves = np.tile(np.repeat(self.noiseWaves, snrLen), clnLen)
            snrSeqs = np.tile(self.snrRatio, clnLen * nosLen)
        elif mode in ('max', 'extra'):
            raise ValueError("Use 'syn_extra_mode' to run extra mode.")
        else:
            raise ValueError(f"Unknown mode: {mode!r}")

        # The room is a fourth axis. Repeating the three existing sequences
        # rather than regenerating them keeps `reg` and `inc` meaning exactly
        # what they meant before, with or without rooms.
        rooms = self.rt60Ratio or [None]
        if len(rooms) > 1:
            repeat = len(rooms)
            clnWaves = np.repeat(clnWaves, repeat)
            nosWaves = np.repeat(nosWaves, repeat)
            snrSeqs = np.repeat(snrSeqs, repeat)
            rt60Seqs = np.tile(rooms, len(clnWaves) // repeat)
        else:
            rt60Seqs = [rooms[0]] * len(clnWaves)

        outWaves = []
        for cln, nos, snr, rt in zip(clnWaves, nosWaves, snrSeqs, rt60Seqs):
            snr_tag = f'n{abs(snr)}' if snr < 0 else f"{snr}dB"
            parts = [cln.stem, nos.stem, snr_tag]
            if rt is not None:
                parts.append(f"rt{int(round(rt * 1000))}ms")
            outDir = self.noisyDir / cln.relative_to(self.clean_dir_).parent \
                if mkdir_parents else self.noisyDir
            outWaves.append((outDir / '_'.join(parts)).with_suffix('.wav'))

        # Each room is seeded from the base seed and the output name, so the
        # same request rebuilds the same dataset and two files that share an
        # RT60 still get different rooms.
        self.rooms = {}
        self.channels = {}
        self.targets = {}
        targets_dir = self.noisyDir / "TARGETS"
        for out, rt in zip(outWaves, rt60Seqs):
            file_seed = (0 if seed is None else int(seed)) ^ \
                (zlib.crc32(out.name.encode('utf-8')) & 0x7FFFFFFF)
            if rt is not None:
                self.rooms[Path(out)] = (float(rt), self.drr_db, file_seed)
                if self.write_targets:
                    # mirror the mixture's own layout, so the two names line up
                    self.targets[Path(out)] = \
                        targets_dir / Path(out).relative_to(self.noisyDir)
            if self.channel:
                # a different derivation from the room's, so a file does not
                # get the same number twice and lose the packets where the
                # room happens to be loudest
                self.channels[Path(out)] = (self.channel,
                                            (file_seed * 2654435761) & 0x7FFFFFFF)

        self.synWavesList = list(zip(outWaves, clnWaves, nosWaves, snrSeqs))
        self.__syn_pool(is_pool=is_pool, is_silence=is_silence, p_silence=p_silence)
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
                     snr_db=int(snr),
                     rt60=self.rooms.get(Path(out), (None,))[0],
                     channel=tag,
                     target=(str(self.targets[Path(out)])
                             if Path(out) in self.targets else None))
                for out, cln, nos, snr in self.synWavesList]

    def syn_extra_mode(self, background_path='../data/backgroundnoise',
                       background_snr: Union[int, Sequence[int]] = (5,)) -> list:
        """Two-stage mixing: noise x background first, then everything x clean."""
        if isinstance(background_snr, str):
            raise TypeError("background_snr can't be str")
        bgSNR = list(background_snr) if hasattr(background_snr, '__iter__') else [background_snr]
        bgNoiseWaves, _ = check_input(background_path)
        if not bgNoiseWaves:
            raise ValueError(f"No wave files found in: {background_path}")
        if self.noisyDir.is_dir() and any(self.noisyDir.iterdir()):
            shutil.rmtree(self.noisyDir)
        self.noisyDir.mkdir(parents=True, exist_ok=True)
        cleanWaves = self.cleanWaves
        for cln in cleanWaves:
            shutil.copy(cln, (self.noisyDir / f"{cln.stem}_n0").with_suffix(cln.suffix))

        # stage 1: noise x background noise into a temporary directory
        temporaryPath = Path('tmp_mixed')
        if temporaryPath.is_dir():
            shutil.rmtree(temporaryPath)
        temporaryPath.mkdir()

        fullCleanWaves = np.repeat(self.noiseWaves, len(bgNoiseWaves) * len(bgSNR))
        fullNoiseWaves = np.tile(np.repeat(bgNoiseWaves, len(bgSNR)), len(self.noiseWaves))
        fullSnrSequence = np.tile(bgSNR, len(self.noiseWaves) * len(bgNoiseWaves))
        fullOutPaths = [
            (temporaryPath / '_'.join(
                [cln.stem, nos.stem, f"{f'n{abs(snr)}' if snr < 0 else snr}dB"])
             ).with_suffix(".wav")
            for cln, nos, snr in zip(fullCleanWaves, fullNoiseWaves, fullSnrSequence)]
        self.synWavesList = list(zip(fullOutPaths, fullCleanWaves,
                                     fullNoiseWaves, fullSnrSequence))
        self.__syn_pool(is_pool=False, is_silence=False)

        # stage 2: clean x (stage-1 output + noise + background noise)
        noiseWaves = fullOutPaths + self.noiseWaves + bgNoiseWaves
        fullCleanWaves = np.repeat(self.cleanWaves, len(noiseWaves) * len(self.snrRatio))
        fullNoiseWaves = np.tile(np.repeat(noiseWaves, len(self.snrRatio)), len(cleanWaves))
        fullSnrSequence = np.tile(self.snrRatio, len(cleanWaves) * len(noiseWaves))
        fullOutPaths = [
            (self.noisyDir / '_'.join(
                [cln.stem, nos.stem, f"{f'n{abs(snr)}' if snr < 0 else snr}dB"])
             ).with_suffix(".wav")
            for cln, nos, snr in zip(fullCleanWaves, fullNoiseWaves, fullSnrSequence)]
        self.synWavesList = list(zip(fullOutPaths, fullCleanWaves,
                                     fullNoiseWaves, fullSnrSequence))

        self.__syn_pool(is_pool=True, is_silence=True, p_silence=0.02)
        shutil.rmtree(temporaryPath)
        return self.synWavesList

    def __syn_pool(self, is_pool: bool, is_silence: bool, p_silence: float = 0.02) -> None:
        sr_ = self.target_sr          # None -> keep each clean file's own rate

        if is_silence and 0 < p_silence < 1:
            # duplicate a random p_silence share as noise-only "silence" class
            out0, cln0, nos0, snr0 = zip(*self.synWavesList)
            index = np.random.choice(
                range(len(self.synWavesList)),
                size=round(len(self.synWavesList) * p_silence))
            out, cln, nos, snr = zip(*[self.synWavesList[i] for i in index])
            snr = tuple(0 for _ in snr)
            out = tuple(o.parent / f"{c.stem}{n.stem}_n00.wav"
                        for o, c, n in zip(out, cln, nos))
            self.synWavesList = list(zip(out0 + out, cln0 + cln, nos0 + nos, snr0 + snr))

        rooms = self.rooms or None
        channels = self.channels or None
        targets = self.targets or None
        if is_pool and len(self.synWavesList) > 200:
            fcn = partial(Synthesizer.syn_waves, self.synWavesList, is_silence,
                          sr_, rooms=rooms, channels=channels, targets=targets)
            with Pool(8) as pool:
                pool.map(fcn, trange(len(self.synWavesList), desc="[kudio] Audio synthesis"))
        else:
            for wave in tqdm(self.synWavesList, desc="[kudio] Audio synthesis"):
                Synthesizer.syn_waves(wave, is_silence, sr_, rooms=rooms,
                                      channels=channels, targets=targets)

    @staticmethod
    def syn_waves(synWavesList, isSilence: bool, target_sr: Optional[int],
                  num: Optional[int] = None,
                  rooms: Optional[Dict[Path, tuple]] = None,
                  channels: Optional[Dict[Path, tuple]] = None,
                  targets: Optional[Dict[Path, Path]] = None) -> None:
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
        """
        save_dir, clean_file, noise_file, snr = \
            synWavesList if num is None else synWavesList[num]
        # target_sr=None keeps the clean file's native rate; the noise is then
        # resampled to whatever the clean file turned out to be, so the two are
        # never mixed at different rates
        y_clean, sr_clean = librosa.load(str(clean_file), sr=target_sr)
        y_noise, _ = librosa.load(str(noise_file), sr=sr_clean)

        room = (rooms or {}).get(Path(save_dir))
        if room is not None:
            from kudio.core.room import apply_rir, rir
            rt60_s, drr_db, room_seed = room
            # trim=True keeps the length, so the clean file stays a
            # sample-aligned target; the direct path sits at sample 0 of the
            # impulse response, so nothing is delayed either
            y_clean = apply_rir(
                y_clean, rir(int(sr_clean), rt60=rt60_s, drr_db=drr_db,
                             seed=room_seed))

        target_path = (targets or {}).get(Path(save_dir))
        if target_path is not None:
            Path(target_path).parent.mkdir(exist_ok=True, parents=True)
            wavfile.write(str(target_path), int(sr_clean),
                          (np.clip(y_clean, -1.0, 1.0)
                           * np.iinfo(np.int16).max).astype(np.int16))

        clean_pwr = np.sum(np.abs(y_clean) ** 2) / len(y_clean)
        if len(y_noise) < len(y_clean):
            repeats = len(y_clean) // len(y_noise) + 1
            y_noise = np.tile(y_noise, repeats)[:len(y_clean)]
        else:  # random crop (no-op when both are the same length)
            ind = np.random.randint(0, len(y_noise) - len(y_clean) + 1)
            y_noise = y_noise[ind:ind + len(y_clean)]

        y_noise = y_noise - np.mean(y_noise)
        noise_variance = clean_pwr / (10 ** (snr / 10))
        noise = np.sqrt(noise_variance) * y_noise / np.std(y_noise)

        y_noisy = noise if isSilence else y_clean + noise

        link = (channels or {}).get(Path(save_dir))
        if link is not None:
            from kudio.effects.channel import apply_channel
            spec, link_seed = link
            # the mixture, not the speech: a wire carries whatever reached the
            # microphone, and the noise reached it too
            y_noisy = apply_channel(y_noisy, int(sr_clean), spec,
                                    seed=link_seed)

        Path(save_dir).parent.mkdir(exist_ok=True, parents=True)
        wavfile.write(str(save_dir), int(sr_clean),
                      (y_noisy * np.iinfo(np.int16).max).astype(np.int16))
