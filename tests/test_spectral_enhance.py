# -*- coding: utf-8 -*-
"""Statistical speech enhancement.

The gain rules are textbook formulas, so the strongest checks available are
their *known relationships* — MMSE-STSA collapses to Wiener at high SNR, and
the three sit in a fixed order at low SNR. Those hold or the implementation is
wrong, whatever a listening test says.
"""
import numpy as np
import pytest

import kudio
from kudio.enhance.spectral import (
    _gain_logmmse,
    _gain_mmse_stsa,
    _gain_omlsa_factory,
    _gain_wiener,
)
from kudio.exceptions import FeatureError

SR = 16000


@pytest.fixture
def noisy_and_clean():
    """5 s: two harmonic bursts over white noise, with silence to open with."""
    rng = np.random.default_rng(0)

    def burst(n, f0):
        t = np.arange(n) / SR
        harmonics = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 14))
        return (harmonics * (0.5 + 0.5 * np.sin(2 * np.pi * 3.5 * t)) / 7)

    clean = np.zeros(SR * 5, dtype=np.float32)
    clean[SR // 2:SR * 2] = burst(int(1.5 * SR), 140.0)
    clean[int(2.8 * SR):int(4.5 * SR)] = burst(int(1.7 * SR), 170.0)
    noisy = (clean + 0.05 * rng.standard_normal(len(clean))).astype(np.float32)
    return noisy, clean


# ----------------------------------------------------------------- gain rules

@pytest.fixture
def snr_grid():
    xi = np.array([1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1e4])
    return xi, 1.0 + xi                      # gamma = 1 + xi, the mean case


def test_the_three_gains_sit_in_their_known_order(snr_grid):
    """Wiener <= log-MMSE <= MMSE-STSA. Get one formula wrong and this breaks."""
    xi, gamma = snr_grid
    wiener = _gain_wiener(xi, gamma)
    lsa = _gain_logmmse(xi, gamma)
    stsa = _gain_mmse_stsa(xi, gamma)
    assert np.all(wiener <= lsa + 1e-12)
    assert np.all(lsa <= stsa + 1e-12)


def test_mmse_stsa_becomes_wiener_at_high_snr(snr_grid):
    """The Bessel series has a known limit; it is also where it goes unstable."""
    xi, gamma = snr_grid
    assert _gain_mmse_stsa(xi, gamma)[-1] == pytest.approx(
        _gain_wiener(xi, gamma)[-1], abs=1e-6)

    huge = np.array([1e6, 1e8])
    gains = _gain_mmse_stsa(huge, 1.0 + huge)
    assert np.all(np.isfinite(gains))
    assert gains == pytest.approx(_gain_wiener(huge, 1.0 + huge), abs=1e-6)


@pytest.mark.parametrize("rule", [_gain_wiener, _gain_logmmse, _gain_mmse_stsa])
def test_gains_stay_between_zero_and_one(snr_grid, rule):
    xi, gamma = snr_grid
    gains = rule(xi, gamma)
    assert np.all(gains > 0)
    assert np.all(gains <= 1.0 + 1e-9)


def test_gains_rise_with_the_a_priori_snr(snr_grid):
    """More speech in the bin must never mean less of it kept."""
    xi, gamma = snr_grid
    for rule in (_gain_wiener, _gain_logmmse, _gain_mmse_stsa):
        assert np.all(np.diff(rule(xi, gamma)) > 0)


def test_omlsa_never_exceeds_the_lsa_gain_or_the_floor(snr_grid):
    """It is a geometric mean of the LSA gain and the floor, so it lies between."""
    xi, gamma = snr_grid
    g_min = 10.0 ** (-25.0 / 20.0)
    lsa = _gain_logmmse(xi, gamma)
    omlsa = _gain_omlsa_factory(0.3, g_min)(xi, gamma)
    assert np.all(omlsa <= np.maximum(lsa, g_min) + 1e-12)
    assert np.all(omlsa >= np.minimum(lsa, g_min) - 1e-12)


def test_a_higher_speech_absence_probability_suppresses_more(snr_grid):
    """...above the floor. Below it, mixing toward G_min raises the gain, and
    `spectral_enhance` clips to the floor anyway — so that region is not where
    the parameter means anything."""
    xi, gamma = snr_grid
    g_min = 10.0 ** (-25.0 / 20.0)
    lsa = _gain_logmmse(xi, gamma)
    above = lsa > g_min
    gentle = _gain_omlsa_factory(0.1, g_min)(xi, gamma)[above]
    harsh = _gain_omlsa_factory(0.8, g_min)(xi, gamma)[above]
    assert above.sum() >= 4
    assert np.all(harsh <= gentle + 1e-12)


def test_a_higher_speech_absence_probability_leaves_a_quieter_residue(
        noisy_and_clean):
    """The same thing measured where it matters: on a passage with no speech."""
    noisy, _ = noisy_and_clean
    silence = slice(0, SR // 4)
    gentle = kudio.spectral_enhance(noisy, SR, 'omlsa', q=0.05)
    harsh = kudio.spectral_enhance(noisy, SR, 'omlsa', q=0.9)
    assert np.std(harsh[silence]) < np.std(gentle[silence])


# ------------------------------------------------------------------- the loop

@pytest.mark.parametrize("method", sorted(kudio.METHODS))
def test_every_method_runs_and_preserves_length(noisy_and_clean, method):
    noisy, _ = noisy_and_clean
    out = kudio.spectral_enhance(noisy, SR, method)
    assert out.shape == noisy.shape
    assert out.dtype == np.float32
    assert np.all(np.isfinite(out))


@pytest.mark.parametrize("method", sorted(kudio.METHODS))
def test_every_method_improves_the_snr(noisy_and_clean, method):
    noisy, clean = noisy_and_clean
    before = kudio.si_sdr(clean, noisy)
    after = kudio.si_sdr(clean, kudio.spectral_enhance(noisy, SR, method))
    assert after > before + 1.0, f"{method}: {before:.2f} -> {after:.2f} dB"


@pytest.mark.parametrize("method", sorted(kudio.METHODS))
def test_no_method_amplifies(noisy_and_clean, method):
    """These are gains in [0, 1]; an output louder than its input is a bug."""
    noisy, _ = noisy_and_clean
    out = kudio.spectral_enhance(noisy, SR, method)
    assert np.max(np.abs(out)) <= np.max(np.abs(noisy)) * 1.05


def test_output_lines_up_with_the_input(noisy_and_clean):
    """trad_enhance returns 128 samples short; this must not."""
    noisy, clean = noisy_and_clean
    out = kudio.spectral_enhance(noisy, SR)
    assert len(out) == len(noisy)
    # a time shift would wreck the correlation with the clean signal
    lag = int(np.argmax(np.correlate(out, clean, mode="same")) - len(out) // 2)
    assert abs(lag) <= 1


@pytest.mark.parametrize("estimator", sorted(kudio.NOISE_ESTIMATORS))
def test_every_noise_estimator_works(noisy_and_clean, estimator):
    noisy, clean = noisy_and_clean
    out = kudio.spectral_enhance(noisy, SR, 'logmmse', noise=estimator)
    assert kudio.si_sdr(clean, out) > kudio.si_sdr(clean, noisy)


def test_the_noise_estimator_matters_as_much_as_the_gain_rule(noisy_and_clean):
    """The clip opens with half a second of noise, which `initial` assumes and
    `quantile` cannot exploit. A comparison that varies only the gain rule is
    measuring the smaller of the two knobs."""
    noisy, clean = noisy_and_clean
    scores = {est: kudio.si_sdr(
        clean, kudio.spectral_enhance(noisy, SR, 'logmmse', noise=est))
        for est in kudio.NOISE_ESTIMATORS}
    assert max(scores.values()) - min(scores.values()) > 2.0, scores


def test_estimate_noise_psd_shape_and_sign(noisy_and_clean):
    noisy, _ = noisy_and_clean
    stft = kudio.STFT(sr=SR, n_fft=512, hop_length=128, window='hann')
    power = np.abs(stft.analyse(noisy)) ** 2
    for estimator in kudio.NOISE_ESTIMATORS:
        psd = kudio.estimate_noise_psd(power, estimator)
        assert psd.shape == power.shape
        assert np.all(psd > 0)


def test_the_gain_floor_controls_the_residual(noisy_and_clean):
    """A hard zero is not the goal: total suppression is what warbles."""
    noisy, _ = noisy_and_clean
    quiet = kudio.spectral_enhance(noisy, SR, 'logmmse', gain_floor_db=-60.0)
    loud = kudio.spectral_enhance(noisy, SR, 'logmmse', gain_floor_db=-6.0)
    silence = slice(0, SR // 4)              # the clip opens with noise only
    assert np.std(quiet[silence]) < np.std(loud[silence])


def test_stronger_over_subtraction_removes_more(noisy_and_clean):
    noisy, _ = noisy_and_clean
    silence = slice(0, SR // 4)
    gentle = kudio.spectral_enhance(noisy, SR, 'specsub', over_subtraction=1.0)
    strong = kudio.spectral_enhance(noisy, SR, 'specsub', over_subtraction=6.0)
    assert np.std(strong[silence]) < np.std(gentle[silence])


def test_spectral_gate_strength_of_zero_is_a_no_op(noisy_and_clean):
    noisy, _ = noisy_and_clean
    out = kudio.spectral_enhance(noisy, SR, 'spectral_gate', prop_decrease=0.0)
    assert out == pytest.approx(noisy, abs=1e-4)


# ---------------------------------------------------------------- the schema

def test_every_method_describes_itself():
    """The Studio builds its controls from this; a bare name is not enough."""
    for name, method in kudio.METHODS.items():
        assert method.name == name
        assert method.label and method.summary
        for param in method.params:
            assert param.low <= param.default <= param.high, (name, param.name)
            assert param.help, (name, param.name)


def test_declared_parameters_are_the_ones_accepted(noisy_and_clean):
    noisy, _ = noisy_and_clean
    short = noisy[:SR]
    for name, method in kudio.METHODS.items():
        kudio.spectral_enhance(short, SR, name, **method.defaults())


def test_an_unknown_method_lists_the_real_ones(noisy_and_clean):
    noisy, _ = noisy_and_clean
    with pytest.raises(FeatureError, match="logmmse"):
        kudio.spectral_enhance(noisy, SR, "magic")


def test_a_parameter_the_method_does_not_take_is_an_error(noisy_and_clean):
    """Silently ignoring `q=0.4` on Wiener would look like it did something."""
    noisy, _ = noisy_and_clean
    with pytest.raises(FeatureError, match="does not take"):
        kudio.spectral_enhance(noisy, SR, "wiener", q=0.4)


def test_rejects_stereo_and_clips_shorter_than_a_frame():
    with pytest.raises(FeatureError, match="mono"):
        kudio.spectral_enhance(np.zeros((SR, 2), dtype=np.float32), SR)
    with pytest.raises(FeatureError, match="shorter than"):
        kudio.spectral_enhance(np.zeros(100, dtype=np.float32), SR)


# ------------------------------------------------------------------ comparison

def test_compare_runs_everything_and_ranks_it(noisy_and_clean):
    noisy, clean = noisy_and_clean
    results = kudio.compare_enhancers(noisy, SR, reference=clean, metrics=False)
    assert len(results) == len(kudio.METHODS)
    assert all(r.error is None for r in results)
    assert all(r.si_sdr_db is not None for r in results)
    assert all(r.delta_snr_db is not None and r.delta_snr_db > 0 for r in results)


def test_compare_without_a_reference_still_measures_something(noisy_and_clean):
    """The real-recording case: no clean signal exists anywhere."""
    noisy, _ = noisy_and_clean
    results = kudio.compare_enhancers(noisy, SR, ['wiener', 'logmmse'])
    assert all(r.si_sdr_db is None for r in results)
    assert all(np.isfinite(r.noise_floor_db) for r in results)


def test_compare_reports_a_failure_instead_of_dying(noisy_and_clean, monkeypatch):
    """A comparison exists to find out which methods work."""
    import kudio.enhance.spectral as module

    real = module.spectral_enhance

    def explode(y, sr, method='logmmse', **kwargs):
        if method == 'wiener':
            raise RuntimeError("boom")
        return real(y, sr, method, **kwargs)

    monkeypatch.setattr(module, "spectral_enhance", explode)
    noisy, _ = noisy_and_clean
    results = module.compare_enhancers(noisy, SR, ['wiener', 'logmmse'])
    assert results[0].error and "boom" in results[0].error
    assert results[1].error is None


def test_compare_times_every_method_fairly(noisy_and_clean):
    """Whichever ran first used to absorb librosa's JIT — in a timing table."""
    noisy, _ = noisy_and_clean
    results = kudio.compare_enhancers(noisy, SR, ['wiener', 'logmmse', 'wiener'],
                                      metrics=False)
    first, _, again = (r.seconds for r in results)
    assert max(first, again) < 10 * min(first, again) + 0.05


def test_compare_rejects_an_unknown_method(noisy_and_clean):
    noisy, _ = noisy_and_clean
    with pytest.raises(FeatureError, match="unknown method"):
        kudio.compare_enhancers(noisy, SR, ['logmmse', 'magic'])


# ----------------------------------------------------------------- STFT round trip

def test_complex_stft_round_trips(noisy_and_clean):
    noisy, _ = noisy_and_clean
    stft = kudio.STFT(sr=SR, n_fft=512, hop_length=128, window='hann')
    spec = stft.analyse(noisy)
    assert spec.shape[1] == stft.n_bins
    assert np.iscomplexobj(spec)
    back = stft.synthesise(spec, length=len(noisy))
    assert len(back) == len(noisy)
    assert back == pytest.approx(noisy, abs=1e-4)


# ------------------------------------------------------------- the second axis

def test_sweeping_both_axes_gives_the_full_grid(noisy_and_clean):
    noisy, clean = noisy_and_clean
    results = kudio.compare_enhancers(
        noisy, SR, reference=clean, metrics=False,
        noises=list(kudio.NOISE_ESTIMATORS))
    assert len(results) == len(kudio.METHODS) * len(kudio.NOISE_ESTIMATORS)
    assert {(r.method, r.noise) for r in results} == {
        (m, n) for m in kudio.METHODS for n in kudio.NOISE_ESTIMATORS}


def test_the_estimator_axis_is_the_wider_one(noisy_and_clean):
    """The reason `noises=` exists: sweeping only methods measures the smaller
    knob. On this clip the grid spans far more than any single estimator's
    column does."""
    noisy, clean = noisy_and_clean
    results = kudio.compare_enhancers(
        noisy, SR, reference=clean, metrics=False,
        noises=list(kudio.NOISE_ESTIMATORS))

    grid = max(r.si_sdr_db for r in results) - min(r.si_sdr_db for r in results)
    within = max(
        max(r.si_sdr_db for r in results if r.noise == est)
        - min(r.si_sdr_db for r in results if r.noise == est)
        for est in kudio.NOISE_ESTIMATORS)
    assert grid > within


def test_a_result_is_labelled_by_both_choices(noisy_and_clean):
    noisy, _ = noisy_and_clean
    result = kudio.compare_enhancers(noisy, SR, ['logmmse'],
                                     noises=['quantile'])[0]
    assert result.label == "logmmse + quantile"
    assert "logmmse + quantile" in str(result)


def test_a_fixed_estimator_still_appears_on_every_row(noisy_and_clean):
    noisy, _ = noisy_and_clean
    results = kudio.compare_enhancers(noisy, SR, ['wiener', 'logmmse'],
                                      noise='initial')
    assert {r.noise for r in results} == {"initial"}


def test_the_default_estimator_is_recorded(noisy_and_clean):
    noisy, _ = noisy_and_clean
    assert kudio.compare_enhancers(noisy, SR, ['wiener'])[0].noise == "mcra"


def test_fixing_and_sweeping_the_estimator_together_is_an_error(noisy_and_clean):
    """`noise='mcra', noises=[...]` has no sensible reading."""
    noisy, _ = noisy_and_clean
    with pytest.raises(FeatureError, match="not both"):
        kudio.compare_enhancers(noisy, SR, ['wiener'], noise='mcra',
                                noises=['quantile'])


def test_an_unknown_estimator_is_rejected(noisy_and_clean):
    noisy, _ = noisy_and_clean
    with pytest.raises(FeatureError, match="unknown noise estimator"):
        kudio.compare_enhancers(noisy, SR, ['wiener'], noises=['telepathy'])


def test_a_failure_in_the_grid_names_both_choices(noisy_and_clean, monkeypatch):
    import kudio.enhance.spectral as module

    real = module.spectral_enhance

    def explode(y, sr, method='logmmse', **kwargs):
        if kwargs.get('noise') == 'quantile':
            raise RuntimeError("boom")
        return real(y, sr, method, **kwargs)

    monkeypatch.setattr(module, "spectral_enhance", explode)
    noisy, _ = noisy_and_clean
    results = module.compare_enhancers(noisy, SR, ['wiener'],
                                       noises=['mcra', 'quantile'])
    failed = [r for r in results if r.error]
    assert [r.label for r in failed] == ["wiener + quantile"]


# ---------------------------------------------------------------- folders

@pytest.fixture
def noisy_folder(tmp_path, noisy_and_clean):
    noisy, _ = noisy_and_clean
    src = tmp_path / "in"
    (src / "sub").mkdir(parents=True)
    for name in ("a.wav", "b.wav", "sub/a.wav"):
        kudio.save_wave(src / name, noisy, SR)
    return src


def test_enhance_folder_writes_everything(noisy_folder, tmp_path):
    result = kudio.enhance_folder(noisy_folder, tmp_path / "out")
    assert (result.written, result.total) == (3, 3)
    assert not result.failed


def test_enhance_folder_mirrors_subdirectories(noisy_folder, tmp_path):
    """Two files named a.wav in different folders must not collide."""
    out = tmp_path / "out"
    result = kudio.enhance_folder(noisy_folder, out)
    written = {p.relative_to(out).as_posix() for p in result.outputs}
    assert written == {"a.wav", "b.wav", "sub/a.wav"}


def test_enhance_folder_actually_denoises(noisy_folder, tmp_path):
    out = tmp_path / "out"
    kudio.enhance_folder(noisy_folder, out, 'logmmse')
    before = kudio.audio_report(*kudio.file_load(noisy_folder / "a.wav")[::1])
    after = kudio.audio_report(*kudio.file_load(out / "a.wav")[::1])
    assert after.noise_floor_dbfs < before.noise_floor_dbfs - 5.0


def test_the_report_says_how_much_came_off(noisy_folder, tmp_path):
    result = kudio.enhance_folder(noisy_folder, tmp_path / "out", report=True)
    assert len(result.floors) == 3
    assert result.mean_reduction_db() > 5.0
    for _, before, after in result.floors:
        assert after < before


def test_without_a_report_there_is_nothing_to_report(noisy_folder, tmp_path):
    result = kudio.enhance_folder(noisy_folder, tmp_path / "out")
    assert result.floors == ()
    assert result.mean_reduction_db() is None


def test_the_result_is_still_a_convert_result(noisy_folder, tmp_path):
    """One shape for 'I processed a folder', however it was processed."""
    result = kudio.enhance_folder(noisy_folder, tmp_path / "out")
    assert isinstance(result, kudio.ConvertResult)
    assert bool(result) is True


def test_enhance_folder_collects_failures(noisy_folder, tmp_path):
    (noisy_folder / "broken.wav").write_bytes(b"not a wave file")
    result = kudio.enhance_folder(noisy_folder, tmp_path / "out")
    assert result.written == 3
    assert [p.name for p, _ in result.failed] == ["broken.wav"]
    assert bool(result) is False


def test_enhance_folder_can_stop_at_the_first_failure(noisy_folder, tmp_path):
    (noisy_folder / "aaa_broken.wav").write_bytes(b"nope")
    with pytest.raises(Exception):
        kudio.enhance_folder(noisy_folder, tmp_path / "out", on_error="raise")


def test_enhance_folder_reports_progress(noisy_folder, tmp_path):
    seen = []
    kudio.enhance_folder(noisy_folder, tmp_path / "out",
                         progress=lambda n, total, path: seen.append((n, total)))
    assert seen == [(1, 3), (2, 3), (3, 3)]


def test_enhance_folder_takes_method_parameters(noisy_folder, tmp_path):
    kudio.enhance_folder(noisy_folder, tmp_path / "out", 'omlsa', q=0.5)
    with pytest.raises(FeatureError, match="does not take"):
        kudio.enhance_folder(noisy_folder, tmp_path / "out2", 'wiener', q=0.5)


def test_enhance_folder_rejects_an_unknown_method(noisy_folder, tmp_path):
    with pytest.raises(FeatureError, match="unknown method"):
        kudio.enhance_folder(noisy_folder, tmp_path / "out", 'magic')


def test_enhance_folder_on_an_empty_folder(tmp_path):
    from kudio.exceptions import AudioIOError
    (tmp_path / "empty").mkdir()
    with pytest.raises(AudioIOError, match="no audio files"):
        kudio.enhance_folder(tmp_path / "empty", tmp_path / "out")


def test_enhance_folder_does_not_overwrite_by_default(noisy_folder, tmp_path):
    out = tmp_path / "out"
    kudio.enhance_folder(noisy_folder, out)
    kudio.enhance_folder(noisy_folder, out)
    assert len(list(out.rglob("*.wav"))) == 6
    kudio.enhance_folder(noisy_folder, out, overwrite=True)
    assert len(list(out.rglob("*.wav"))) == 6


# ------------------------------------------------------- custom enhancers

def test_a_callable_can_be_compared_with_the_built_ins(noisy_and_clean):
    """'Is the model actually better than log-MMSE' is the question people
    have, and two separate printouts is how it goes unanswered."""
    noisy, clean = noisy_and_clean

    def mine(y, sr):
        return kudio.lowpass(kudio.spectral_enhance(y, sr, 'logmmse'), sr, 4000)

    results = kudio.compare_enhancers(
        noisy, SR, ['logmmse', ('mine', mine)], reference=clean, metrics=False)
    assert [r.method for r in results] == ['logmmse', 'mine']
    assert all(r.si_sdr_db is not None for r in results)
    assert all(r.audio.size == len(noisy) for r in results)


def test_a_custom_enhancer_says_it_brought_its_own_noise_estimate(noisy_and_clean):
    noisy, _ = noisy_and_clean
    result = kudio.compare_enhancers(noisy, SR, [('mine', lambda y, s: y)])[0]
    assert result.noise == kudio.CUSTOM_NOISE
    assert result.label == f"mine + {kudio.CUSTOM_NOISE}"


def test_a_bare_function_is_named_after_itself(noisy_and_clean):
    noisy, _ = noisy_and_clean
    results = kudio.compare_enhancers(noisy, SR, [kudio.trad_enhance])
    assert results[0].method == "trad_enhance"


def test_a_custom_enhancer_is_not_swept_over_estimators(noisy_and_clean):
    """It brings its own; running it four times would produce four identical
    rows claiming to differ."""
    noisy, _ = noisy_and_clean
    results = kudio.compare_enhancers(
        noisy, SR, ['wiener', ('mine', lambda y, s: y)],
        noises=list(kudio.NOISE_ESTIMATORS))
    assert len([r for r in results if r.method == 'mine']) == 1
    assert len([r for r in results if r.method == 'wiener']) == 4


def test_a_custom_enhancer_that_raises_is_a_row(noisy_and_clean):
    noisy, _ = noisy_and_clean
    results = kudio.compare_enhancers(
        noisy, SR, ['wiener', ('bad', lambda y, s: 1 / 0)])
    assert results[1].error and "ZeroDivisionError" in results[1].error
    assert results[0].error is None


def test_a_custom_enhancer_object_can_be_passed_directly(noisy_and_clean):
    noisy, _ = noisy_and_clean
    entry = kudio.CustomEnhancer(name="wrapped", fn=lambda y, s: y * 0.5)
    result = kudio.compare_enhancers(noisy, SR, [entry])[0]
    assert result.method == "wrapped"
    assert np.allclose(result.audio, noisy * 0.5)


def test_something_that_is_not_an_enhancer_is_refused(noisy_and_clean):
    noisy, _ = noisy_and_clean
    with pytest.raises(FeatureError, match="cannot compare"):
        kudio.compare_enhancers(noisy, SR, [42])
    with pytest.raises(FeatureError, match="expected"):
        kudio.compare_enhancers(noisy, SR, [("name", "not callable")])
    with pytest.raises(FeatureError, match="unknown method"):
        kudio.compare_enhancers(noisy, SR, ['magic'])


def test_enhance_folder_takes_a_callable(noisy_folder, tmp_path):
    """A trained model is the obvious second case: it configures itself, then
    runs over a tree the same way the built-in methods do."""
    def halve(y, sr):
        return (np.asarray(y) * 0.5).astype(np.float32)

    result = kudio.enhance_folder(noisy_folder, tmp_path / "out",
                                  ("my model", halve))
    assert result.written == 3
    original, _ = kudio.file_load(noisy_folder / "a.wav")
    written, _ = kudio.file_load(next((tmp_path / "out").glob("a.wav")))
    assert np.allclose(written, original * 0.5, atol=1e-4)


def test_a_callable_takes_no_method_parameters(noisy_folder, tmp_path):
    with pytest.raises(FeatureError, match="takes no method parameters"):
        kudio.enhance_folder(noisy_folder, tmp_path / "out",
                             ("m", lambda y, s: y), alpha=0.9)
