# -*- coding: utf-8 -*-
"""Lightweight utilities (device checks, colored console output, timing).

Heavy helpers live in submodules that are *not* imported here, so ``import
kudio`` stays free of matplotlib / pandas:

* plotting  -> ``kudio.util.visual``   (needs ``kudio[viz]``)
* Excel/DF  -> ``kudio.util.conv``     (needs ``kudio[data]``)
"""
from kudio.util.check import CheckDevice, check_device, is_available
from kudio.util.colors import (
    blu,
    color_test,
    cya,
    dkblu,
    dkcya,
    dkgre,
    dkpur,
    dkred,
    dkyel,
    error,
    fGre,
    fRed,
    gre,
    h1,
    info,
    pur,
    red,
    warn,
    yel,
)
from kudio.util.conv import lowercase, upercase, uppercase  # dep-free helpers
from kudio.util.parallel import map_waves
from kudio.util.tools import Timer, find_duplicate, timer

__all__ = [
    # check
    'CheckDevice', 'check_device', 'is_available',
    # colors
    'blu', 'color_test', 'cya', 'dkblu', 'dkcya', 'dkgre', 'dkpur', 'dkred',
    'dkyel', 'error', 'fGre', 'fRed', 'gre', 'h1', 'info', 'pur', 'red',
    'warn', 'yel',
    # tools
    'Timer', 'find_duplicate', 'timer', 'map_waves',
    # conv (list helpers; data2xlsx stays in kudio.util.conv, needs kudio[data])
    'lowercase', 'uppercase', 'upercase',
]
