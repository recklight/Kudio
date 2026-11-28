# -*- coding: utf-8 -*-
"""Core audio functionality: I/O, buffering, streaming, features, synthesis,
and evaluation."""
from kudio.core.align import MIN_CORRELATION, Alignment, align, find_delay
from kudio.core.buffer import AudioBuffer, FetchBuffer
from kudio.core.evaluator import (
    AudioEvaluate,
    check_metrics_install,
    eval_metrics,
    lsd,
    pesq,
    score,
    sdi,
    segmental_snr,
    si_sdr,
    snr,
)
from kudio.core.feature import (
    # canonical (v3)
    ContextFrames,
    Standardizer,
    denoise_spec2wav,
    denoise_wav2spec,
    features2matrix,
    file_to_spectrogram,
    frame_windows,
    logspec_from_files,
    mel_band_hz,
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
from kudio.core.diarization import (
    DiarizationErrors,
    Turn,
    der,
    detection_error_rate,
    diarization_errors,
    jer,
    load_rttm,
)
from kudio.core.dnsmos import DnsmosScore, dnsmos, dnsmos_p808, find_dnsmos_model
from kudio.core.f0eval import (
    F0Errors,
    f0_errors,
    f0_frame_error,
    f0_rmse,
    fine_pitch_error,
    gross_pitch_error,
    voicing_decision_error,
)
from kudio.core.intelligibility import CsiiScore, csii, ncm
from kudio.core.siib import siib, siib_gauss
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
from kudio.core.lines import TonalLines, tonal_lines
from kudio.core.manager import AudioDataManager
from kudio.core.melcepstrum import MCD_SETTINGS, mcd
from kudio.core.report import AudioReport, audio_report
from kudio.core.separation import (
    BssEval,
    PitScore,
    bss_eval,
    pit_si_sdr,
    sdr_improvement,
    si_sdr_improvement,
)
from kudio.core.room import (
    Reverberation,
    apply_rir,
    rir,
    rt60,
    schroeder_curve,
)
from kudio.core.pitch import PitchTrack, f0
from kudio.core.registry import BIBTEX, METRIC_INFO, MetricInfo, cite, metric_info
from kudio.core.dataset_score import (
    REFERENCE_FREE,
    DatasetScore,
    MetricSummary,
    score_dataset,
)
from kudio.core.signature import parse_signature, signature
from kudio.core.stats import (
    ConfidenceInterval,
    PermutationResult,
    bootstrap_ci,
    holm,
    paired_ci,
    paired_rate_ci,
    permutation_test,
    rate_ci,
)
from kudio.core.quality import (
    CompositeScore,
    cepstral_distance,
    composite,
    fw_segmental_snr,
    itakura_saito,
    llr,
    wss,
)
from kudio.core.spectrogram import SpectrogramStream
from kudio.core.srmr import srmr
from kudio.core.stft import STFT
from kudio.core.stream import (
    PA_FLOAT32,
    PA_INT16,
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
from kudio.core.vad import speech_ratio, vad, vad_frames, vad_split, vad_trim
from kudio.core.verification import (
    C_PRIMARY_PRIORS,
    DetCurve,
    act_dcf,
    c_primary,
    cllr,
    det_curve,
    eer,
    min_cllr,
    min_dcf,
)
from kudio.core.wada import wada_snr
from kudio.core.wer import ErrorCounts, cer, char_errors, wer, word_errors

__all__ = [
    # buffer
    'AudioBuffer', 'FetchBuffer',
    # alignment, which the reference metrics assume and cannot check
    'Alignment', 'find_delay', 'align', 'MIN_CORRELATION',
    # evaluator + lightweight metrics
    'AudioEvaluate', 'check_metrics_install', 'eval_metrics', 'si_sdr', 'snr',
    'segmental_snr', 'sdi', 'pesq', 'score', 'lsd',
    # Hu and Loizou's measures for enhanced speech, and their composite
    'llr', 'itakura_saito', 'cepstral_distance', 'wss', 'fw_segmental_snr',
    'composite', 'CompositeScore',
    # intelligibility against the clean reference
    'ncm', 'csii', 'CsiiScore', 'siib', 'siib_gauss',
    # mel-cepstral distortion, as URGENT 2025 scores it
    'mcd', 'MCD_SETTINGS',
    # separation: several sources, matched to their estimates
    'si_sdr_improvement', 'sdr_improvement', 'pit_si_sdr', 'PitScore',
    'bss_eval', 'BssEval',
    # what a recognizer, a speaker verifier or a pitch tracker got wrong
    'wer', 'cer', 'word_errors', 'char_errors', 'ErrorCounts',
    'eer', 'min_dcf', 'det_curve', 'DetCurve',
    'cllr', 'min_cllr', 'act_dcf', 'c_primary', 'C_PRIMARY_PRIORS',
    'der', 'diarization_errors', 'DiarizationErrors', 'jer',
    'detection_error_rate', 'load_rttm', 'Turn',
    'gross_pitch_error', 'voicing_decision_error', 'f0_frame_error',
    'fine_pitch_error', 'f0_rmse', 'f0_errors', 'F0Errors',
    # feature (canonical)
    'STFT', 'Standardizer', 'stack_context', 'ContextFrames', 'frame_windows',
    'denoise_spec2wav', 'denoise_wav2spec', 'features2matrix',
    'file_to_spectrogram', 'logspec_from_files', 'mel_band_hz', 'melspectrogram',
    'mfcc', 'mfcc_from_files', 'save_spectrogram_as_wave', 'spectrogram_to_waveform',
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
    'AudioReport', 'audio_report', 'TonalLines', 'tonal_lines',
    'DnsmosScore', 'dnsmos', 'dnsmos_p808', 'find_dnsmos_model',
    # what each metric is, and what to cite for it
    'METRIC_INFO', 'MetricInfo', 'metric_info', 'BIBTEX', 'cite',
    # how a result was computed, in one line beside it
    'signature', 'parse_signature',
    # how far a result over a test set could move, and whether a gap is real
    'bootstrap_ci', 'paired_ci', 'rate_ci', 'paired_rate_ci', 'ConfidenceInterval',
    'permutation_test', 'PermutationResult', 'holm',
    # a whole test set, from the files to the table under a paper
    'score_dataset', 'DatasetScore', 'MetricSummary', 'REFERENCE_FREE',
    'srmr', 'wada_snr',
    # pitch
    'PitchTrack', 'f0',
    # rooms: describe one, apply it, measure what it did
    'rir', 'apply_rir', 'rt60', 'Reverberation', 'schroeder_curve',
    # voice activity
    'vad', 'vad_split', 'vad_trim', 'vad_frames', 'speech_ratio',
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
    'PA_FLOAT32', 'PA_INT16',
    # synth
    'Synthesizer',
]
