# -*- coding: utf-8 -*-
"""Speech-enhancement algorithms (traditional signal-processing methods)."""
from kudio.enhance.traditional import trad_enhance, wavelet_low_pass_filter

__all__ = ['trad_enhance', 'wavelet_low_pass_filter']
