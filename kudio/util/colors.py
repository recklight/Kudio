# -*- coding: utf-8 -*-
"""ANSI-colored console helpers used by the CLI tools and the GUI."""
from __future__ import annotations

__all__ = [
    'fRed',
    'fGre',
    'red',
    'gre',
    'blu',
    'yel',
    'pur',
    'cya',
    'dkred',
    'dkgre',
    'dkblu',
    'dkyel',
    'dkpur',
    'dkcya',
    'h1',
    'info',
    'warn',
    'error',
]


class Colors:
    """ANSI escape sequences (SGR codes)."""
    END = "\033[0m"

    BRIGHT = "\033[1m"
    DIM = "\033[2m"
    UNDERSCORE = "\033[4m"
    BLINK = "\033[5m"

    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    BACK_RED = "\033[41m"
    BACK_GREEN = "\033[42m"
    BACK_YELLOW = "\033[43m"
    BACK_BLUE = "\033[44m"
    BACK_MAGENTA = "\033[45m"
    BACK_CYAN = "\033[46m"
    BACK_WHITE = "\033[47m"


def fmt(iterable) -> str:
    return " ".join(str(i) for i in iterable)


def fRed(*args):
    print(Colors.BLINK + Colors.WHITE + Colors.BACK_RED, fmt(args), Colors.END)


def fGre(*args):
    print(Colors.BLINK + Colors.WHITE + Colors.BACK_GREEN, fmt(args), Colors.END)


def red(*args):
    print(Colors.RED, fmt(args), Colors.END)


def gre(*args):
    print(Colors.GREEN, fmt(args), Colors.END)


def blu(*args):
    print(Colors.BLUE, fmt(args), Colors.END)


def yel(*args):
    print(Colors.YELLOW, fmt(args), Colors.END)


def pur(*args):
    print(Colors.MAGENTA, fmt(args), Colors.END)


def cya(*args):
    print(Colors.CYAN, fmt(args), Colors.END)


def dkred(*args):
    print(Colors.BACK_RED, fmt(args), Colors.END)


def dkgre(*args):
    print(Colors.BACK_GREEN, fmt(args), Colors.END)


def dkblu(*args):
    print(Colors.BACK_BLUE, fmt(args), Colors.END)


def dkyel(*args):
    print(Colors.BACK_YELLOW, fmt(args), Colors.END)


def dkpur(*args):
    print(Colors.BACK_MAGENTA, fmt(args), Colors.END)


def dkcya(*args):
    print(Colors.BACK_CYAN, fmt(args), Colors.END)


def h1(*args):
    print(Colors.BRIGHT, fmt(args), Colors.END)


def info(*args):
    print(Colors.DIM + "\t", fmt(args), Colors.END)


def warn(*args):
    print(Colors.BACK_CYAN + "WARN:" + Colors.END + Colors.CYAN, fmt(args), Colors.END)


def error(*args):
    print(Colors.BACK_RED + Colors.BLINK + "ERROR:" + Colors.END + Colors.RED,
          fmt(args), Colors.END)


def wait(*args) -> str:
    return str.lower(input(Colors.BLUE + fmt(args) + Colors.END))


_ALL_FUNCS = (fRed, fGre, red, gre, blu, yel, pur, cya,
              dkred, dkgre, dkblu, dkyel, dkpur, dkcya, h1, info, warn, error)


def color_test(text: str = 'Thanks for using kudio packages!') -> bool:
    try:
        for f in _ALL_FUNCS:
            f(text)
        return True
    except Exception as e:
        print(e)
        return False
