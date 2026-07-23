# -*- coding: utf-8 -*-
import numpy as np

from kudio import AudioEvaluate
from kudio.core.evaluator import check_metrics_install, eval_metrics


def test_check_metrics_install_no_side_effects():
    status = check_metrics_install()
    assert len(status) == 3
    assert all(isinstance(s, bool) for s in status)


def test_eval_metrics_handles_missing_backends(sine):
    ref = sine
    deg = sine + 0.01 * np.random.default_rng(0).standard_normal(len(sine)).astype(np.float32)
    pesq, stoi, sdr = eval_metrics(16000, ref, deg)
    for value in (pesq, stoi, sdr):
        assert value is None or np.isfinite(value)


def test_audio_evaluate_constructs(wav_dir):
    files = sorted(wav_dir.glob("*.wav"))
    ev = AudioEvaluate(files, files)
    assert ev.get_df() == []  # nothing evaluated yet
