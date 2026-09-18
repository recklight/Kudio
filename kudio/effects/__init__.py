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
from kudio.effects.channel import (
    A_LAW_A,
    CHANNEL_DEFAULTS,
    CHANNEL_PRESETS,
    MU_LAW_MU,
    PACKET_MS,
    TELEPHONE_BAND,
    a_law,
    apply_channel,
    bit_depth,
    channel_spec,
    channel_tag,
    dropouts,
    independent_burst,
    mu_law,
    telephone,
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
    # the channel: what the transmission did to it
    'mu_law', 'a_law', 'bit_depth', 'dropouts', 'telephone',
    'independent_burst', 'channel_spec', 'channel_tag', 'apply_channel',
    'MU_LAW_MU', 'A_LAW_A', 'PACKET_MS', 'TELEPHONE_BAND',
    'CHANNEL_DEFAULTS', 'CHANNEL_PRESETS',
    # augment
    'time_stretch', 'pitch_shift', 'gain', 'random_gain',
    'normalize', 'normalize_db',
    'add_noise_snr', 'reverb', 'spec_augment',
]
