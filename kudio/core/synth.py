# -*- coding: utf-8 -*-
"""Noisy-speech synthesis: mix clean waves with noise at given SNRs."""
from __future__ import annotations

import logging
import shutil
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import List, Optional, Sequence, Union

import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm, trange

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
                 snr_ratio: Union[int, Sequence[int]] = (-5, 0, 5)):
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

        self.synWavesList: Optional[list] = None
        self.clean_dir_ = clean
        self.desired_sample: Optional[int] = None

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
            desired_sample: Optional[int] = None,
            seed: Optional[int] = None) -> Optional[list]:
        """Generate the noisy dataset.

        mode ``'reg'``/``'regular'``: cycle noises & SNRs over the clean list
        (output count == clean count). mode ``'inc'``/``'increment'``: full
        cartesian product clean x noise x SNR.

        :param seed: seed the RNG for reproducible noise cropping / sampling.
        """
        if seed is not None:
            np.random.seed(seed)
        self.desired_sample = desired_sample
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

        outWaves = []
        for cln, nos, snr in zip(clnWaves, nosWaves, snrSeqs):
            snr_tag = f'n{abs(snr)}' if snr < 0 else f"{snr}dB"
            outDir = self.noisyDir / cln.relative_to(self.clean_dir_).parent \
                if mkdir_parents else self.noisyDir
            outWaves.append(
                (outDir / '_'.join([cln.stem, nos.stem, snr_tag])).with_suffix('.wav'))

        self.synWavesList = list(zip(outWaves, clnWaves, nosWaves, snrSeqs))
        self.__syn_pool(is_pool=is_pool, is_silence=is_silence, p_silence=p_silence)
        return self.synWavesList

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
        sr_ = self.desired_sample if self.desired_sample else 16000

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

        if is_pool and len(self.synWavesList) > 200:
            fcn = partial(Synthesizer.syn_waves, self.synWavesList, is_silence, sr_)
            with Pool(8) as pool:
                pool.map(fcn, trange(len(self.synWavesList), desc="[kudio] Audio synthesis"))
        else:
            for wave in tqdm(self.synWavesList, desc="[kudio] Audio synthesis"):
                Synthesizer.syn_waves(wave, is_silence, sr_)

    @staticmethod
    def syn_waves(synWavesList, isSilence: bool, d_sr_: int,
                  num: Optional[int] = None) -> None:
        """Mix one ``(save_dir, clean_file, noise_file, snr)`` entry to disk.

        With ``num`` given, ``synWavesList`` is the full list and ``num`` its
        index (multiprocessing form); otherwise it is a single entry.
        """
        save_dir, clean_file, noise_file, snr = \
            synWavesList if num is None else synWavesList[num]
        y_clean, sr_clean = librosa.load(str(clean_file), sr=d_sr_)
        y_noise, _ = librosa.load(str(noise_file), sr=d_sr_)

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
        Path(save_dir).parent.mkdir(exist_ok=True, parents=True)
        wavfile.write(str(save_dir), int(sr_clean),
                      (y_noisy * np.iinfo(np.int16).max).astype(np.int16))
