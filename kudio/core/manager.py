# -*- coding: utf-8 -*-
"""In-memory registry of loaded/recorded audio clips used by the GUI."""
from __future__ import annotations

import logging
import os
from copy import deepcopy
from pathlib import Path
from typing import KeysView, List, Optional, Union, ValuesView

import numpy as np

from kudio._numbers import flag, whole
from kudio._samples import to_float
from kudio.core._policy import sample_rate
from kudio.core.feature import wave_separate
from kudio.core.io import PathLike, file_load, save_wave
from kudio.core.stream import play_audio

__all__ = ['AudioDataManager']

log = logging.getLogger(__name__)


class AudioDataManager:
    """Keyed store of audio clips: ``{name: {dir, wave, sr, ch}}``.

    ``wave`` is a list with one waveform per channel.
    """

    def __init__(self):
        self.suffix = '.wav'
        self.rate_: Optional[int] = None
        self.ch_ = 1
        self.default_dict = {"dir": None, "wave": [], "sr": None, "ch": self.ch_}
        self.reset_defaults()

    def reset_defaults(self) -> None:
        self.audios_ = {}

    reset = reset_defaults

    def set_params(self, sr: int, ch: int) -> None:
        """The rate and the channels, 1 or 2, of what is added from now on.

        :raises ValueError: for another number of channels. Taken for 1, six of
            them were saved one after the other, six times as long.
        """
        sr = sample_rate(sr, 'set_params')
        if whole(ch) not in (1, 2):
            raise ValueError(f"set_params(): ch has to be 1 or 2 channels, got {ch!r}")
        self.ch_ = whole(ch)
        self.rate_ = sr
        self.default_dict = {"dir": None, "wave": [], "sr": sr, "ch": self.ch_}

    def __call__(self, wave_name: str):
        return self.get_item(wave_name)

    def __getitem__(self, wave_name: Union[str, int]):
        if isinstance(wave_name, int):
            wave_name = list(self.get_names())[wave_name]
        return self.get_item(wave_name)

    def __len__(self) -> int:
        return len(self.audios_)

    def get_item(self, name: str) -> Optional[dict]:
        item = self.audios_.get(name)
        if item is None:
            log.warning("Manager: audio %r not found", name)
        return item

    def get_wave(self, name: str) -> Optional[List[np.ndarray]]:
        item = self.get_item(name)
        return item['wave'] if item is not None else None

    def get_sample(self, name: str) -> Optional[int]:
        item = self.get_item(name)
        return item['sr'] if item is not None else None

    def add_local_file(self, dir: PathLike, name: Optional[str] = None) -> None:
        """Load *dir* into the manager as a mono downmix (``ch`` 1), whatever
        :meth:`set_params` says, so :meth:`save_all` writes it back as mono.
        Any format :func:`kudio.file_load` opens is fine."""
        dir = Path(dir)
        if not dir.is_file():
            raise FileNotFoundError(f"File not found: {dir}")
        name = dir.stem if name is None else name
        if name in self.get_names():
            log.warning("Manager: name %r already exists", name)
            return
        # read first: a file that cannot be read leaves no empty clip behind
        # it, which the next call would take for one already loaded
        wave_, rate_ = file_load(dir, mono=True)
        item = deepcopy(self.default_dict)
        item["dir"] = str(dir)
        item["wave"].append(wave_)
        item["sr"] = rate_
        item["ch"] = 1
        self.audios_[name] = item

    def add_new_file(self, name: Optional[str] = None, dir: Optional[PathLike] = None,
                     wave: Optional[List[np.ndarray]] = None,
                     sr: Optional[int] = None, ch: Optional[int] = None,
                     override: bool = False) -> None:
        """Add a clip. *wave* is a list holding one waveform, the *ch* channels
        of it interleaved; left out, the clip starts empty.

        A name already taken keeps its clip unless *override* is set, and then
        only a clip that can be built takes its place.

        :raises ValueError: for a *wave* that is not a list of one waveform, a
            bare array among them.
        """
        override = flag(override, 'override', 'AudioDataManager.add_new_file', ValueError)
        if name is None:
            if dir is None:
                log.warning("Manager: please provide a name or dir")
                return
            name = Path(dir).stem
        if name in self.get_names() and not override:
            log.warning("Manager: name %r already exists", name)
            return
        sr = self.rate_ if sr is None else sample_rate(sr, 'add_new_file')
        ch = self.ch_ if ch is None else ch
        item = deepcopy(self.default_dict)
        if dir is not None:
            item["dir"] = str(dir)
        if wave is not None:
            try:
                waveform, = wave
            except (TypeError, ValueError):      # not a sequence, or not of one
                waveform = None
            if np.ndim(waveform) == 0:           # one number is not a waveform
                got = (f"an array of shape {wave.shape}" if isinstance(wave, np.ndarray)
                       else f"{type(wave).__name__} {wave!r}"[:60])
                raise ValueError(f"add_new_file(): wave has to be a list holding one "
                                 f"waveform, got {got}; the manager is as it was")
            item["wave"].extend(wave_separate(waveform, ch))
        item["sr"] = sr
        item["ch"] = ch
        self.audios_[name] = item

    def add_name(self, name: str) -> None:
        self.audios_[name] = deepcopy(self.default_dict)

    def add_dir(self, name: str, dir: str) -> None:
        self.audios_[name]["dir"] = dir

    def add_wave(self, name: str, wave: np.ndarray) -> None:
        self.audios_[name]["wave"].append(wave)

    def add_sr(self, name: str, sr: int) -> None:
        self.audios_[name]["sr"] = sample_rate(sr, 'add_sr')

    def add_ch(self, name: str, ch: int) -> None:
        self.audios_[name]["ch"] = ch

    def del_element(self, name: str) -> None:
        self.audios_.pop(name, None)

    def del_from_path(self, pth: PathLike, key: Optional[str] = None) -> None:
        pth = Path(pth)
        files = pth.glob(key if key else f"*{self.suffix}")
        for f in files:
            if f.is_file():
                f.unlink(missing_ok=True)

    def get_names(self) -> KeysView[str]:
        return self.audios_.keys()

    def get_values(self) -> ValuesView[dict]:
        return self.audios_.values()

    def _file_name(self, name: str) -> str:
        """A clip name as a file name: the whole name, plus ``.wav``.

        ``with_suffix`` used to do this, and it reads the dot in 'take.v2' as
        the start of an extension, so the clip was written to 'take.wav' --
        the file belonging to the clip called 'take'. Load both from a folder
        and whichever was saved second kept the file. The channel names were
        worse: 'take.v2_ch0' and 'take.v2_ch1' both came out as 'take.wav',
        one overwriting the other under a name identifying neither.
        """
        name = str(name)
        if name.lower().endswith(self.suffix):
            return name
        return name + self.suffix

    def _targets(self, name: str, item: dict,
                 dst_path: Optional[PathLike] = None) -> List[Path]:
        """Every file :meth:`save_` would write for the clip *name*.

        Empty when the clip has no destination at all: it was never loaded
        from a file and no *dst_path* was given.
        """
        if dst_path is None:
            if item['dir'] is None:
                return []
            dir_ = Path(item['dir'])
        else:
            dir_ = Path(dst_path) / self._file_name(name)
        # the clip's own channel count, not the manager's: a mono file loaded
        # while the manager was set to two channels goes back where it came
        # from, not to a new <name>_ch0.wav beside it
        if len(item['wave']) > 1:
            return [dir_.parent / self._file_name(f"{name}_ch{i}")
                    for i in range(len(item['wave']))]
        return [dir_]

    def save_(self, name: str, dst_path: Optional[PathLike] = None,
              down_sample: Optional[int] = None) -> bool:
        if not len(self):
            log.warning("Manager: nothing to save")
            return False
        item = self.get_item(name)
        if item is None:
            return False
        if not item['wave']:
            log.warning("Manager: %r holds no audio yet, nothing written", name)
            return False
        targets = self._targets(name, item, dst_path)
        if not targets:
            log.warning("Manager: %r did not come from a file, so there is "
                        "nowhere to put it; pass dst_path= to say where", name)
            return False
        for d, w in zip(targets, item['wave']):
            save_wave(d, w, item['sr'], d_sample=down_sample)
        return True

    def save_all(self, dst_path: Optional[PathLike] = None,
                 down_sample: Optional[int] = None) -> bool:
        if not len(self):
            log.warning("Manager: nothing to save")
            return False
        # Checked for the whole store before a single file is written: two
        # clips landing on one name used to mean one of them simply vanished,
        # with `True` coming back and nothing in the log to say so. `normcase`
        # because Windows would collide on names that compare different.
        claimed = {}
        for name in self.get_names():
            for target in self._targets(name, self.audios_[name], dst_path):
                key = os.path.normcase(str(target))
                owner = claimed.setdefault(key, name)
                if owner != name:
                    log.error("Manager: %r and %r would both be written to "
                              "%s, so one would lose its audio. Nothing was "
                              "saved -- rename one of them first.",
                              owner, name, target)
                    return False
        for n in self.get_names():
            if not self.save_(n, dst_path=dst_path, down_sample=down_sample):
                return False
        log.info("Manager: saved all waves")
        return True

    def play_audio(self, name: str, volume: float = 1.0) -> None:
        item = self.get_item(name)
        if item is None:
            log.warning("Manager: cannot play %r", name)
            return
        # scaled in float and put back: int16 times 1.5, cast straight back,
        # wraps round to the other sign
        w, back = to_float(item['wave'][0])
        play_audio(wav=back(w * volume), rate=item['sr'])
