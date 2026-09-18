# -*- coding: utf-8 -*-
"""Core audio functionality: I/O, buffering, streaming, features, synthesis,
and evaluation."""
from kudio.core.align import MIN_CORRELATION, Alignment, align, find_delay
from kudio.core.buffer import AudioBuffer, FetchBuffer
from kudio.core.evaluator import (
    AudioEvaluate,
    check_metrics_install,
    eval_metrics,
    segmental_snr,
    si_sdr,
    snr,
)
from kudio.core.feature import (
    # canonical (v3)
    Standardizer,
    denoise_spec2wav,
    denoise_wav2spec,
    features2matrix,
    file_to_spectrogram,
    frame_windows,
    logspec_from_files,
    melspectrogram,
    mfcc,
    mfcc_from_files,
    save_spectrogram_as_wave,
    spectrogram_to_waveform,
    stack_context,
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
from kudio.core.dataset import (
    Label,
    Pair,
    load_labels,
    load_manifest,
    save_labels,
    save_manifest,
    split_pairs,
)
from kudio.core.dnsmos import DnsmosScore, dnsmos, find_dnsmos_model
from kudio.core.io import (
    AudioInfo,
    ConvertResult,
    LoadAudio,
    audio_info,
    check_file,
    check_input,
    check_path,
    convert_folder,
    copy_waves,
    file_load,
    load_wave,
    resample,
    load_waves,
    save_wave,
)
from kudio.core.loudness import (
    MOMENTARY,
    SHORT_TERM,
    LoudnessCurve,
    loudness,
    loudness_over_time,
    loudness_range,
    match_loudness,
    normalize_lufs,
    true_peak,
)
from kudio.core.manager import AudioDataManager
from kudio.core.report import AudioReport, audio_report
from kudio.core.room import (
    Reverberation,
    apply_rir,
    rir,
    rt60,
    schroeder_curve,
)
from kudio.core.pitch import PitchTrack, f0
from kudio.core.spectrogram import SpectrogramStream
from kudio.core.stft import STFT
from kudio.core.stream import (
    LocalStreamReader,
    Recorder,
    RemoteStreamReader,
    StreamRecorder,
    play_audio,
    record,
    wave_decode,
    wave_encode,
)
from kudio.core.synth import Synthesizer
from kudio.core.vad import speech_ratio, vad, vad_split, vad_trim

__all__ = [
    # buffer
    'AudioBuffer', 'FetchBuffer',
    # alignment, which the reference metrics assume and cannot check
    'Alignment', 'find_delay', 'align', 'MIN_CORRELATION',
    # evaluator + lightweight metrics
    'AudioEvaluate', 'check_metrics_install', 'eval_metrics', 'si_sdr', 'snr',
    'segmental_snr',
    # feature (canonical)
    'STFT', 'Standardizer', 'stack_context', 'frame_windows',
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
    'file_load', 'resample', 'load_wave', 'load_waves', 'save_wave',
    'AudioInfo', 'audio_info', 'ConvertResult', 'convert_folder',
    # loudness (BS.1770 / EBU R 128)
    'loudness', 'match_loudness', 'normalize_lufs',
    'loudness_over_time', 'LoudnessCurve', 'loudness_range', 'true_peak',
    'MOMENTARY', 'SHORT_TERM',
    # reference-free inspection
    'AudioReport', 'audio_report',
    'DnsmosScore', 'dnsmos', 'find_dnsmos_model',
    # pitch
    'PitchTrack', 'f0',
    # rooms: describe one, apply it, measure what it did
    'rir', 'apply_rir', 'rt60', 'Reverberation', 'schroeder_curve',
    # voice activity
    'vad', 'vad_split', 'vad_trim', 'speech_ratio',
    # dataset manifests and labels
    'Pair', 'save_manifest', 'load_manifest', 'split_pairs',
    'Label', 'save_labels', 'load_labels',
    # manager
    'AudioDataManager',
    # live analysis
    'SpectrogramStream',
    # stream
    'LocalStreamReader', 'Recorder', 'RemoteStreamReader', 'StreamRecorder',
    'play_audio', 'record', 'wave_decode', 'wave_encode',
    # synth
    'Synthesizer',
]
