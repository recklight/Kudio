# -*- coding: utf-8 -*-
"""Package surface: version, public names, and deprecation aliases."""
import warnings

import numpy as np
import pytest

import kudio

# names the app / training layer import from the top level (must stay working)
TOP_LEVEL_NAMES = [
    'AudioBuffer', 'AudioDataManager', 'AudioEvaluate', 'CheckDevice',
    'FetchBuffer', 'LocalStreamReader', 'Recorder', 'RemoteStreamReader',
    'Synthesizer', 'Timer', 'check_input', 'file_load', 'save_wave',
    'play_audio', 'record', 'trad_enhance', 'wavelet_low_pass_filter',
    'warn', 'gre', 'pur', 'fGre',
    # v3 canonical feature names
    'waveform_to_spectrogram', 'spectrogram_to_waveform', 'file_to_spectrogram',
    'save_spectrogram_as_wave', 'mfcc', 'mfcc_from_files', 'logspec_from_files',
    'melspectrogram',
    # v3 effects
    'trim_silence', 'split_on_silence', 'time_stretch', 'pitch_shift',
    'add_noise_snr', 'spec_augment',
    # v3 metrics + exceptions
    'si_sdr', 'snr', 'segmental_snr', 'KudioError', 'AudioIOError',
]


def test_version_is_v3():
    assert kudio.__version__.startswith("3.")


def test_top_level_names_present():
    missing = [n for n in TOP_LEVEL_NAMES if not hasattr(kudio, n)]
    assert not missing, f"missing top-level names: {missing}"


def test_all_is_curated():
    for name in kudio.__all__:
        assert hasattr(kudio, name), name


def test_import_is_lightweight():
    """`import kudio` must not drag in matplotlib / pandas.

    Uses a subprocess so other test modules (which import pandas at collection
    time) don't pollute the check.
    """
    import subprocess
    import sys
    code = ("import kudio, sys; "
            "assert 'matplotlib' not in sys.modules, 'matplotlib leaked'; "
            "assert 'pandas' not in sys.modules, 'pandas leaked'")
    subprocess.check_call([sys.executable, "-c", code])


def test_deprecated_aliases_warn():
    y = np.zeros(4096, dtype=np.float32)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        kudio.w2s(y)                      # deprecated alias of waveform_to_spectrogram
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)


def test_alias_result_matches_canonical():
    y = np.random.RandomState(0).randn(4096).astype(np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        np.testing.assert_array_equal(kudio.w2s(y), kudio.waveform_to_spectrogram(y))


def test_optional_dep_error_is_typed():
    # data2xlsx lives in an extra-gated submodule, not the top level
    assert not hasattr(kudio, 'data2xlsx')
    from kudio.exceptions import DependencyError
    assert issubclass(DependencyError, kudio.KudioError)
