# -*- coding: utf-8 -*-
"""Waveform effects: editing, filtering, silence handling and augmentation."""
from kudio.effects.augment import (
    add_noise_snr,
    gain,
    normalize,
    normalize_db,
    pitch_shift,
    random_gain,
    reverb,
    spec_augment,
    time_stretch,
)
from kudio.effects.edit import (
    DEFAULT_CROSSFADE,
    crossfade,
    fade,
    fade_in,
    fade_out,
    remove_dc,
    reverse,
    splice,
)
from kudio.effects.filters import (
    band_filter,
    bandpass,
    bandstop,
    highpass,
    lowpass,
)
from kudio.effects.silence import split_on_silence, trim_silence

__all__ = [
    # silence
    'trim_silence', 'split_on_silence',
    # editing
    'fade', 'fade_in', 'fade_out', 'reverse', 'remove_dc',
    'crossfade', 'splice', 'DEFAULT_CROSSFADE',
    # filters
    'band_filter', 'highpass', 'lowpass', 'bandpass', 'bandstop',
    # augment
    'time_stretch', 'pitch_shift', 'gain', 'random_gain',
    'normalize', 'normalize_db',
    'add_noise_snr', 'reverb', 'spec_augment',
]
