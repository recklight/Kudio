# -*- coding: utf-8 -*-
"""YAML-backed configuration with attribute access.

>>> from kudio import KudioConfig
>>> cfg = KudioConfig.load_config('TrainConfig.yaml')
>>> cfg.train.clean
'data/train/clean'
"""
from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Optional

import yaml

__all__ = ['KudioConfig', 'KC']

log = logging.getLogger(__name__)


class Dict(dict):
    """dict subclass with attribute access and recursive conversion.

    Vendored from the `addict <https://github.com/mewwts/addict>`_ project
    (MIT license), trimmed to what kudio needs.
    """

    def __init__(__self, *args, **kwargs):
        object.__setattr__(__self, '__parent', kwargs.pop('__parent', None))
        object.__setattr__(__self, '__key', kwargs.pop('__key', None))
        object.__setattr__(__self, '__frozen', False)
        for arg in args:
            if not arg:
                continue
            if isinstance(arg, dict):
                for key, val in arg.items():
                    __self[key] = __self._hook(val)
            elif isinstance(arg, tuple) and (not isinstance(arg[0], tuple)):
                __self[arg[0]] = __self._hook(arg[1])
            else:
                for key, val in iter(arg):
                    __self[key] = __self._hook(val)

        for key, val in kwargs.items():
            __self[key] = __self._hook(val)

    def __setattr__(self, name, value):
        if hasattr(self.__class__, name):
            raise AttributeError(
                f"'Dict' object attribute {name!r} is read-only")
        self[name] = value

    def __setitem__(self, name, value):
        is_frozen = (hasattr(self, '__frozen') and
                     object.__getattribute__(self, '__frozen'))
        if is_frozen and name not in super().keys():
            raise KeyError(name)
        super().__setitem__(name, value)
        try:
            p = object.__getattribute__(self, '__parent')
            key = object.__getattribute__(self, '__key')
        except AttributeError:
            p = None
            key = None
        if p is not None:
            p[key] = self
            object.__delattr__(self, '__parent')
            object.__delattr__(self, '__key')

    @classmethod
    def _hook(cls, item):
        if isinstance(item, dict):
            return cls(item)
        if isinstance(item, (list, tuple)):
            return type(item)(cls._hook(elem) for elem in item)
        return item

    def __getattr__(self, item):
        return self.__getitem__(item)

    def __missing__(self, name):
        if object.__getattribute__(self, '__frozen'):
            raise KeyError(name)
        return self.__class__(__parent=self, __key=name)

    def __delattr__(self, name):
        del self[name]

    def to_dict(self) -> dict:
        base = {}
        for key, value in self.items():
            if isinstance(value, type(self)):
                base[key] = value.to_dict()
            elif isinstance(value, (list, tuple)):
                base[key] = type(value)(
                    item.to_dict() if isinstance(item, type(self)) else item
                    for item in value)
            else:
                base[key] = value
        return base

    def copy(self):
        return copy.copy(self)

    def deepcopy(self):
        return copy.deepcopy(self)

    def __deepcopy__(self, memo):
        other = self.__class__()
        memo[id(self)] = other
        for key, value in self.items():
            other[copy.deepcopy(key, memo)] = copy.deepcopy(value, memo)
        return other

    def update(self, *args, **kwargs):
        other = {}
        if args:
            if len(args) > 1:
                raise TypeError("update expected at most 1 argument")
            other.update(args[0])
        other.update(kwargs)
        for k, v in other.items():
            if ((k not in self) or
                    (not isinstance(self[k], dict)) or
                    (not isinstance(v, dict))):
                self[k] = v
            else:
                self[k].update(v)

    def __getnewargs__(self):
        return tuple(self.items())

    def __getstate__(self):
        return self

    def __setstate__(self, state):
        self.update(state)

    def __or__(self, other):
        if not isinstance(other, (Dict, dict)):
            return NotImplemented
        new = Dict(self)
        new.update(other)
        return new

    def __ror__(self, other):
        if not isinstance(other, (Dict, dict)):
            return NotImplemented
        new = Dict(other)
        new.update(self)
        return new

    def __ior__(self, other):
        self.update(other)
        return self

    def setdefault(self, key, default=None):
        if key in self:
            return self[key]
        self[key] = default
        return default

    def freeze(self, should_freeze: bool = True):
        object.__setattr__(self, '__frozen', should_freeze)
        for val in self.values():
            if isinstance(val, Dict):
                val.freeze(should_freeze)

    def unfreeze(self):
        self.freeze(False)


class KudioConfig(Dict):
    """Attribute-style config with YAML load/save and audio-centric defaults."""

    def write_config(self, file: str = 'cfg.yaml', exist_ok: bool = False) -> bool:
        """Dump the config to *file*; refuses to overwrite unless ``exist_ok``."""
        if Path(file).exists() and not exist_ok:
            log.warning("Config exists, not overwritten: %s", file)
            return False
        with open(str(file), 'w', encoding='utf-8') as f:
            yaml.safe_dump(self.to_dict(), f, allow_unicode=True)
        log.info("Config written to %s", file)
        return True

    @classmethod
    def load_config(cls, cfg_dir: str = '') -> Optional['KudioConfig']:
        """Load a YAML config file.

        *cfg_dir* may be a file, or a directory containing exactly one
        ``cfg.yaml``. Anything else falls back to :meth:`default_config`.
        """

        def _load(f_: Path):
            log.info("Load config -> %s", f_.absolute())
            with open(str(f_.absolute()), encoding="utf-8") as f:
                return yaml.safe_load(f)

        _file = Path(cfg_dir)
        if _file.is_file():
            return cls(_load(_file))
        if _file.is_dir():
            candidates = list(_file.rglob("cfg.yaml"))
            if len(candidates) == 1:
                return cls(_load(candidates[0]))
        log.info("No config file found, using defaults")
        return cls(**cls.default_config())

    @staticmethod
    def default_config() -> dict:
        """Audio-only defaults. Training/model settings live in the
        application layer (e.g. chptrain), not in the audio library."""
        return {
            'waveform': {
                'type': '.wav',
                'default_rate': 16000,
                'common_rate': [192000, 96000, 88200, 64000, 48000, 44100,
                                32000, 22050, 16000, 11025, 8000, 6000],
                'resample_rate': 16000,
                'channels': 'mono',
                'format': '16bit'},
            'device': {
                'pool_max': 200, 'cpu_cores': 8},
            'f_mfe': {
                'n_mels': 40,
                'bank': 1,
                'f0': 0,
                'f1': 48000},
        }

    def show(self, key: Optional[str] = None) -> None:
        """Print one key (or the whole config) to the log."""
        if key is None:
            for k, value in self.items():
                log.info("%s: %s", k, value)
        elif key in self:
            log.info("%s: %s", key, self.get(key))
        else:
            log.warning("Unknown config key: %s", key)


KC = KudioConfig
