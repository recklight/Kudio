# -*- coding: utf-8 -*-
"""Core audio functionality: I/O, buffering, streaming, features, synthesis,
and evaluation."""
from kudio.core.buffer import AudioBuffer, FetchBuffer
from kudio.core.evaluator import (
    AudioEvaluate,
    eval_metrics,
    segmental_snr,
    si_sdr,
    snr,
)
from kudio.core.feature import (
    # canonical (v3)
    denoise_spec2wav,
    denoise_wav2spec,
    features2matrix,
    file_to_spectrogram,
    logspec_from_files,
    melspectrogram,
    mfcc,
    mfcc_from_files,
    save_spectrogram_as_wave,
    spectrogram_to_waveform,
    waveform_to_spectrogram,
    wave_separate,
    wave_slicing,
    # deprecated aliases
    concat_logspec,
    concat_logspec_,
    concat_mfcc,
    concat_mfcc_,
    contextual_LogSpectrogram,
    f2s,
    s2w,
    spec2wav,
    spec2wavform,
    w2mfcc,
    w2s,
    wav2mel,
    wav2mfcc,
    wav2spec,
    wavform2spec,
)
from kudio.core.io import (
    LoadAudio,
    check_file,
    check_input,
    check_path,
    copy_waves,
    file_load,
    load_wave,
    load_waves,
    save_wave,
)
from kudio.core.manager import AudioDataManager
from kudio.core.stream import (
    LocalStreamReader,
    Recorder,
    RemoteStreamReader,
    play_audio,
    record,
    wave_decode,
    wave_encode,
)
from kudio.core.synth import Synthesizer

__all__ = [
    # buffer
    'AudioBuffer', 'FetchBuffer',
    # evaluator + lightweight metrics
    'AudioEvaluate', 'eval_metrics', 'si_sdr', 'snr', 'segmental_snr',
    # feature (canonical)
    'denoise_spec2wav', 'denoise_wav2spec', 'features2matrix',
    'file_to_spectrogram', 'logspec_from_files', 'melspectrogram', 'mfcc',
    'mfcc_from_files', 'save_spectrogram_as_wave', 'spectrogram_to_waveform',
    'waveform_to_spectrogram', 'wave_separate', 'wave_slicing',
    # feature (deprecated aliases)
    'concat_logspec', 'concat_logspec_', 'concat_mfcc', 'concat_mfcc_',
    'contextual_LogSpectrogram', 'f2s', 's2w', 'spec2wav', 'spec2wavform',
    'w2mfcc', 'w2s', 'wav2mel', 'wav2mfcc', 'wav2spec', 'wavform2spec',
    # io
    'LoadAudio', 'check_file', 'check_input', 'check_path', 'copy_waves',
    'file_load', 'load_wave', 'load_waves', 'save_wave',
    # manager
    'AudioDataManager',
    # stream
    'LocalStreamReader', 'Recorder', 'RemoteStreamReader', 'play_audio',
    'record', 'wave_decode', 'wave_encode',
    # synth
    'Synthesizer',
]
