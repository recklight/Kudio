# -*- coding: utf-8 -*-
"""Waveform effects: silence handling and data augmentation."""
from kudio.effects.augment import (
    add_noise_snr,
    gain,
    pitch_shift,
    random_gain,
    reverb,
    spec_augment,
    time_stretch,
)
from kudio.effects.silence import split_on_silence, trim_silence

__all__ = [
    # silence
    'trim_silence', 'split_on_silence',
    # augment
    'time_stretch', 'pitch_shift', 'gain', 'random_gain',
    'add_noise_snr', 'reverb', 'spec_augment',
]
