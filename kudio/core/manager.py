# -*- coding: utf-8 -*-
"""In-memory registry of loaded/recorded audio clips used by the GUI."""
from __future__ import annotations

import logging
import os
from copy import deepcopy
from pathlib import Path
from typing import List, Optional, Union

from kudio.core.feature import wave_separate
from kudio.core.io import file_load, save_wave
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
        if sr <= 0:
            raise ValueError(f"sample rate must be > 0, got {sr}")
        self.ch_ = ch if 0 < ch < 3 else 1
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

    def get_item(self, name: str):
        item = self.audios_.get(name)
        if item is None:
            log.warning("Manager: audio %r not found", name)
        return item

    def get_wave(self, name: str):
        item = self.get_item(name)
        return item['wave'] if item is not None else None

    def get_sample(self, name: str):
        item = self.get_item(name)
        return item['sr'] if item is not None else None

    def add_local_file(self, dir, name: Optional[str] = None) -> None:
        """Load *dir* into the manager as a mono downmix (``ch`` 1), whatever
        :meth:`set_params` says, so :meth:`save_all` writes it back as mono.
        Any format soundfile reads is fine."""
        dir = Path(dir)
        if not dir.is_file():
            raise FileNotFoundError(f"File not found: {dir}")
        name = dir.stem if name is None else name
        if name in self.get_names():
            log.warning("Manager: name %r already exists", name)
            return
        self.audios_[name] = deepcopy(self.default_dict)
        self.audios_[name]["dir"] = str(dir)
        wave_, rate_ = file_load(dir, mono=True)
        self.audios_[name]["wave"].append(wave_)
        self.audios_[name]["sr"] = rate_
        self.audios_[name]["ch"] = 1

    def add_new_file(self, name: Optional[str] = None, dir=None, wave=None,
                     sr: Optional[int] = None, ch: Optional[int] = None,
                     override: bool = False) -> None:
        if name is None:
            if dir is None:
                log.warning("Manager: please provide a name or dir")
                return
            name = Path(dir).stem
        if name in self.get_names() and not override:
            log.warning("Manager: name %r already exists", name)
            return
        sr = self.rate_ if sr is None else sr
        ch = self.ch_ if ch is None else ch
        self.audios_[name] = deepcopy(self.default_dict)
        if dir is not None:
            self.audios_[name]["dir"] = str(dir)
        if wave is not None:
            if len(wave) == 1 and ch > 0:
                for channel_wave in wave_separate(wave[0], ch):
                    self.audios_[name]["wave"].append(channel_wave)
            else:
                self.del_element(name)
                log.warning("Manager: unexpected wave or channel count")
                return
        self.audios_[name]["sr"] = sr
        self.audios_[name]["ch"] = ch

    def add_name(self, name: str) -> None:
        self.audios_[name] = deepcopy(self.default_dict)

    def add_dir(self, name: str, dir: str) -> None:
        self.audios_[name]["dir"] = dir

    def add_wave(self, name: str, wave) -> None:
        self.audios_[name]["wave"].append(wave)

    def add_sr(self, name: str, sr: int) -> None:
        self.audios_[name]["sr"] = sr

    def add_ch(self, name: str, ch: int) -> None:
        self.audios_[name]["ch"] = ch

    def del_element(self, name: str) -> None:
        self.audios_.pop(name, None)

    def del_from_path(self, pth, key: Optional[str] = None) -> None:
        pth = Path(pth)
        files = pth.glob(key if key else f"*{self.suffix}")
        for f in files:
            if f.is_file():
                f.unlink(missing_ok=True)

    def get_names(self):
        return self.audios_.keys()

    def get_values(self):
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

    def _targets(self, name: str, item, dst_path=None) -> List[Path]:
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
        if self.ch_ > 1:
            return [dir_.parent / self._file_name(f"{name}_ch{i}")
                    for i in range(len(item['wave']))]
        return [dir_]

    def save_(self, name: str, dst_path=None, down_sample=None) -> bool:
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
        if self.ch_ > 1:
            for d, w in zip(targets, item['wave']):
                save_wave(d, w, item['sr'], d_sample=down_sample)
        else:
            save_wave(targets[0], item['wave'][0], item['sr'],
                      d_sample=down_sample)
        return True

    def save_all(self, dst_path=None, down_sample=None) -> bool:
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
        w = item['wave'][0]
        play_audio(wav=(w * volume).astype(w.dtype), rate=item['sr'])
