# -*- coding: utf-8 -*-
"""What goes in, the same way everywhere: the rate, the layout, the integers.

A rate that is not a positive number, channels first and 64-bit integers are
each refused by one function of kudio's, and everything here is a function
that has to go through it. The metrics have the same tests of their own in
test_metric_input.py.
"""
import importlib
import inspect
import pathlib
import pkgutil
import re
import tempfile
import warnings

import numpy as np
import pytest
import soundfile as sf

import kudio
from kudio import cli
from kudio.core.stream import wave_encode

SR = 16000


def _take(seconds=1.2, level=0.4):
    """Hiss, a tone, hiss."""
    n = int(seconds * SR)
    t = np.arange(n) / SR
    y = 0.001 * np.random.default_rng(0).standard_normal(n)
    y[n // 4:3 * n // 4] += level * np.sin(2 * np.pi * 220 * t[n // 4:3 * n // 4])
    return y.astype(np.float32)


Y = _take()
OTHER = _take(level=0.2)
LONG = np.tile(Y, 3)
IR = kudio.rir(SR, rt60=0.3, drr_db=3.0, seed=0)
NOISE = (0.05 * np.random.default_rng(1).standard_normal(4000)).astype(np.float32)
SPEC = kudio.waveform_to_spectrogram(Y)


def _manager(**kwargs):
    manager = kudio.AudioDataManager()
    manager.add_new_file("take", wave=[Y], **kwargs)
    return manager.get_sample("take")


def _added(sr):
    manager = kudio.AudioDataManager()
    manager.add_name("take")
    manager.add_sr("take", sr)
    return manager.get_sample("take")


def _set(sr):
    manager = kudio.AudioDataManager()
    manager.set_params(sr, 1)
    return manager.rate_


def _written(y, **kwargs):
    with tempfile.TemporaryDirectory() as folder:
        path = pathlib.Path(folder) / "a.wav"
        kudio.save_wave(path, y, SR, **kwargs)
        return sf.read(str(path), dtype='int16')[0]


def _info(y, **kwargs):
    info = kudio.audio_info(y, sr=SR, **kwargs)
    return [x for x in (info.frames, info.channels, info.duration, info.peak) if x is not None]


def _three(y):
    return np.concatenate([y, y, y], axis=-1 if y.shape[0] <= 2 else 0)


#: every function that takes a rate, as one of the rate alone
RATED = {
    'vad': lambda r: kudio.vad(Y, r),
    'vad_split': lambda r: kudio.vad_split(Y, r),
    'vad_trim': lambda r: kudio.vad_trim(Y, r),
    'vad_frames': lambda r: kudio.vad_frames(Y, r),
    'speech_ratio': lambda r: kudio.speech_ratio(Y, r),
    'rir': lambda r: kudio.rir(r, rt60=0.3, seed=0),
    'rt60': lambda r: kudio.rt60(IR, r).summary(),
    'loudness': lambda r: kudio.loudness(Y, r),
    'loudness_over_time': lambda r: kudio.loudness_over_time(Y, r, window=0.4).lufs,
    'loudness_range': lambda r: kudio.loudness_range(LONG, r),
    'true_peak': lambda r: kudio.true_peak(Y, r),
    'normalize_lufs': lambda r: kudio.normalize_lufs(Y, r, -30.0),
    'match_loudness': lambda r: kudio.match_loudness(Y, OTHER, r),
    'match_loudness(reference_sr)':
        lambda r: kudio.match_loudness(Y, OTHER, SR, reference_sr=r),
    'audio_report': lambda r: vars(kudio.audio_report(Y, r)),
    'tonal_lines': lambda r: [line.hz for line in kudio.tonal_lines(Y, r).lines],
    'f0': lambda r: kudio.f0(Y, r).f0,
    'find_delay': lambda r: vars(kudio.find_delay(Y, Y[100:], r, max_seconds=0.1)),
    'align': lambda r: kudio.align(Y, Y[100:], r),
    'fade_in': lambda r: kudio.fade_in(Y, r),
    'fade_out': lambda r: kudio.fade_out(Y, r),
    'fade': lambda r: kudio.fade(Y, r),
    'crossfade': lambda r: kudio.crossfade(Y, OTHER, r),
    'crossfade(0)': lambda r: kudio.crossfade(Y, OTHER, r, 0.0),
    'splice': lambda r: kudio.splice([Y, OTHER], r),
    'highpass': lambda r: kudio.highpass(Y, r),
    'lowpass': lambda r: kudio.lowpass(Y, r, cutoff=3000),
    'lowpass(default)': lambda r: kudio.lowpass(Y, r),
    'bandpass': lambda r: kudio.bandpass(Y, r),
    'bandstop': lambda r: kudio.bandstop(Y, r),
    'band_filter': lambda r: kudio.band_filter(Y, r, 500.0, 'lowpass'),
    'pitch_shift': lambda r: kudio.pitch_shift(Y, r, 2),
    'reverb': lambda r: kudio.reverb(Y, r),
    'mfcc': lambda r: kudio.mfcc(Y, r),
    'melspectrogram': lambda r: kudio.melspectrogram(Y, sr=r),
    'STFT': lambda r: kudio.STFT(sr=r).frame_times(4),
    'STFT.matches': lambda r: kudio.STFT(sr=SR).matches(r),
    'dropouts': lambda r: kudio.dropouts(Y, r, loss=0.1, seed=0),
    'telephone': lambda r: kudio.telephone(Y, r),
    'apply_channel': lambda r: kudio.apply_channel(Y, r, 'voip', seed=0),
    'apply_channel(None)': lambda r: kudio.apply_channel(Y, r, None),
    'SpectrogramStream': lambda r: kudio.SpectrogramStream(r, seconds=0.5).push(Y),
    'StreamEnhancer': lambda r: kudio.StreamEnhancer(r).process(Y),
    'spectral_enhance': lambda r: kudio.spectral_enhance(Y, r),
    'compare_enhancers': lambda r: kudio.compare_enhancers(Y, r, ['wiener'])[0].audio,
    'trad_enhance': lambda r: kudio.trad_enhance(Y, r),
    'AudioDataManager.set_params': _set,
    'AudioDataManager.add_new_file': lambda r: _manager(sr=r),
    'AudioDataManager.add_sr': _added,
}
#: where None is "not given" rather than a rate
OPTIONAL = {'match_loudness(reference_sr)', 'AudioDataManager.add_new_file', 'melspectrogram'}

NO_RATE = ['16000', b'16000', None, [SR], np.array([SR]), 16000 + 0j, float('nan'),
           np.float32('nan'), float('inf'), float('-inf'), 0, 0.0, -1, -16000.0, False, True,
           np.True_, 10 ** 400]


@pytest.mark.parametrize("name", RATED)
def test_what_is_not_a_rate_is_refused(name):
    """True is 1 to Python, and a rate of 1 Hz makes two seconds of audio two
    samples: rt60 reads 6424 s off a room of 0.4."""
    for value in NO_RATE:
        if value is None and name in OPTIONAL:
            continue
        with pytest.raises(ValueError) as refused:
            RATED[name](value)
        said = str(refused.value)
        assert "positive sample rate" in said and said.endswith(f"got {value!r}"), (value, said)
        assert re.match(r"[A-Za-z_0-9.]+\(\): ", said), said


def _flat(value):
    if isinstance(value, dict):
        return [x for v in value.values() for x in _flat(v)]
    if isinstance(value, (list, tuple)):
        return [x for v in value for x in _flat(v)]
    if isinstance(value, np.ndarray):
        return value.astype(np.float64).ravel().tolist()
    return [value]


def _as_text(value):
    """What str() writes of every number _flat gives, in a form compared at once: the
    samples of an array as their float64 bits, every NaN the same one, the rest as str()
    writes them. Two that are the same here write the same there, and writing out 19 200
    samples seven times was most of what the test took."""
    if isinstance(value, dict):
        return tuple(_as_text(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return tuple(_as_text(v) for v in value)
    if isinstance(value, np.ndarray):
        x = value.astype(np.float64).ravel()
        x[np.isnan(x)] = np.nan
        return ('samples', x.tobytes())
    return str(value)


@pytest.mark.parametrize("name", RATED)
def test_one_rate_of_any_type_is_the_same_answer(name):
    """Headers, configs and numpy hand it out as all of these."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = RATED[name](SR)
        expected, text = _flat(out), _as_text(out)
        for value in (16000.0, np.int32(SR), np.uint16(SR), np.float32(SR), np.float64(SR),
                      np.array(SR)):
            got = RATED[name](value)
            if _as_text(got) == text:
                continue
            np.testing.assert_array_equal(np.array(_flat(got), dtype=object).astype(str),
                                          np.array(expected, dtype=object).astype(str), repr(value))


@pytest.mark.parametrize("name", ['vad', 'f0', 'loudness', 'true_peak', 'audio_report',
                                  'tonal_lines', 'rir', 'rt60', 'find_delay', 'dropouts',
                                  'telephone', 'highpass', 'SpectrogramStream',
                                  'StreamEnhancer', 'spectral_enhance'])
def test_a_rate_is_refused_with_the_error_its_module_raises(name):
    with pytest.raises(kudio.FeatureError):
        RATED[name](-1)


def test_a_rate_need_not_be_whole():
    assert kudio.STFT(sr=22254.5).sr == 22254.5
    assert not kudio.STFT(sr=16000).matches(16000.9)
    assert len(kudio.fade_in(Y, 22254.5)) == len(Y)


def test_one_place_checks_rates():
    """io.py has its own, for a file's rate, which has to be whole."""
    root = pathlib.Path(kudio.__file__).parent
    own = [f"{path.name}:{n}" for path in sorted(root.rglob("*.py")) if path.name != "io.py"
           for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
           if re.search(r"\bsr\b *<= *0|sr\) *<= *0", line)]
    assert own == []


# -- the layout -----------------------------------------------------------------------
#: functions of one channel, as one of the audio alone
MONO = {
    'vad': lambda y: kudio.vad(y, SR),
    'vad_frames': lambda y: kudio.vad_frames(y, SR),
    'speech_ratio': lambda y: kudio.speech_ratio(y, SR),
    'f0': lambda y: kudio.f0(y, SR).f0,
    'tonal_lines': lambda y: [line.hz for line in kudio.tonal_lines(y, SR).lines],
    'audio_report': lambda y: vars(kudio.audio_report(y, SR)),
    'apply_rir': lambda y: kudio.apply_rir(y, IR),
    'apply_rir(ir)': lambda y: kudio.apply_rir(Y, y[..., :800] if y.shape[0] == 1 else y[:800]),
    'schroeder_curve': kudio.schroeder_curve,
    'rt60': lambda y: kudio.rt60(y, SR).summary(),
    'find_delay(ref)': lambda y: vars(kudio.find_delay(y, Y[100:], SR)),
    'find_delay(deg)': lambda y: vars(kudio.find_delay(Y, y, SR)),
    'align(ref)': lambda y: kudio.align(y, Y[100:], SR),
    'align(deg)': lambda y: kudio.align(Y, y, SR),
    'mu_law': kudio.mu_law,
    'a_law': kudio.a_law,
    'bit_depth': kudio.bit_depth,
    'dropouts': lambda y: kudio.dropouts(y, SR, seed=0),
    'telephone': lambda y: kudio.telephone(y, SR),
    'apply_channel': lambda y: kudio.apply_channel(y, SR, 'voip', seed=0),
    'waveform_to_spectrogram': kudio.waveform_to_spectrogram,
    'spectrogram_to_waveform': lambda y: kudio.spectrogram_to_waveform(y, SPEC),
    'mfcc': lambda y: kudio.mfcc(y, SR),
    'melspectrogram': lambda y: kudio.melspectrogram(y, sr=SR),
    'denoise_wav2spec': lambda y: kudio.denoise_wav2spec(y)[0],
    'STFT.analyse': lambda y: np.abs(kudio.STFT().analyse(y)),
    'STFT.forward': lambda y: kudio.STFT().forward(y),
    'STFT.inverse': lambda y: kudio.STFT().inverse(y, SPEC),
    'SpectrogramStream': lambda y: kudio.SpectrogramStream(SR).push(y),
    'StreamEnhancer': lambda y: kudio.StreamEnhancer(SR).process(y),
    'spectral_enhance': lambda y: kudio.spectral_enhance(y, SR),
    'compare_enhancers': lambda y: kudio.compare_enhancers(y, SR, ['wiener'])[0].audio,
    'trad_enhance': lambda y: kudio.trad_enhance(y, SR),
    'wavelet_low_pass_filter': kudio.wavelet_low_pass_filter,
    'add_noise_snr': lambda y: kudio.add_noise_snr(y, NOISE, 10, seed=0),
    'add_noise_snr(noise)': lambda y: kudio.add_noise_snr(Y, y, 10, seed=0),
}
#: functions that keep the channels
MULTI = {
    'vad_split': lambda y: kudio.vad_split(y, SR),
    'vad_trim': lambda y: kudio.vad_trim(y, SR),
    'loudness': lambda y: kudio.loudness(y, SR),
    'loudness_over_time': lambda y: kudio.loudness_over_time(y, SR, window=0.4).lufs,
    'loudness_range': lambda y: kudio.loudness_range(_three(y), SR),
    'true_peak': lambda y: kudio.true_peak(y, SR),
    'normalize_lufs': lambda y: kudio.normalize_lufs(y, SR, -30.0),
    'match_loudness': lambda y: kudio.match_loudness(y, OTHER, SR),
    'match_loudness(reference)': lambda y: kudio.match_loudness(OTHER, y, SR),
    'fade_in': lambda y: kudio.fade_in(y, SR),
    'fade_out': lambda y: kudio.fade_out(y, SR),
    'fade': lambda y: kudio.fade(y, SR),
    'crossfade': lambda y: kudio.crossfade(y, y, SR),
    'crossfade(second)': lambda y: kudio.crossfade(Y, y, SR),
    'splice': lambda y: kudio.splice([y, y], SR),
    'splice(second)': lambda y: kudio.splice([Y, y], SR),
    'reverse': kudio.reverse,
    'remove_dc': kudio.remove_dc,
    'highpass': lambda y: kudio.highpass(y, SR),
    'lowpass': lambda y: kudio.lowpass(y, SR, cutoff=2000),
    'lowpass(default)': lambda y: kudio.lowpass(y, SR),
    'bandpass': lambda y: kudio.bandpass(y, SR),
    'bandstop': lambda y: kudio.bandstop(y, SR),
    'band_filter': lambda y: kudio.band_filter(y, SR, 500.0, 'lowpass'),
    'time_stretch': lambda y: kudio.time_stretch(y, 1.25),
    'pitch_shift': lambda y: kudio.pitch_shift(y, SR, 2),
    'gain': lambda y: kudio.gain(y, 3.0),
    'random_gain': lambda y: kudio.random_gain(y, seed=0),
    'normalize': kudio.normalize,
    'normalize_db': kudio.normalize_db,
    'reverb': lambda y: kudio.reverb(y, SR),
    'trim_silence': kudio.trim_silence,
    'split_on_silence': kudio.split_on_silence,
    'Alignment.apply': lambda y: kudio.find_delay(Y, Y[100:], SR).apply(y, y),
    'wave_slicing': kudio.wave_slicing,
    'resample': lambda y: kudio.resample(y, SR, 8000),
    'resample(same)': lambda y: kudio.resample(y, SR, SR),
    'save_wave': _written,
    'save_wave(d_sample)': lambda y: _written(y, d_sample=8000),
    'audio_info': _info,
    'audio_info(peak=False)': lambda y: _info(y, peak=False),
}
EVERY = {**MONO, **MULTI}

#: the modules whose functions have their tests in test_metric_input.py and
#: files of their own
METRICS = {'evaluator', 'quality', 'intelligibility', 'siib', 'srmr', 'wada', 'dnsmos',
           'melcepstrum', 'separation', 'stats'}
#: what takes an array that is not audio, or that is audio of another layout
ELSEWHERE = {
    # spectra and feature matrices
    'DecisionDirected.update', 'NoiseTracker.update', 'STFT.synthesise', 'Standardizer.fit',
    'Standardizer.fit_transform', 'Standardizer.inverse', 'Standardizer.transform',
    'denoise_spec2wav', 'estimate_noise_psd', 'features2matrix', 'frame_windows',
    'save_spectrogram_as_wave', 'spec_augment', 'stack_context',
    # interleaved channels, blocks handed on, the caller's own function
    'AudioBuffer.add', 'AudioDataManager.add_new_file', 'AudioDataManager.add_wave',
    'CustomEnhancer.__call__', 'wave_encode', 'wave_separate',
    # returns what is wrong
    'play_audio',
}


def _takes_an_array():
    """Every public function and method with a parameter annotated as an array,
    and the module it is in."""
    for info in pkgutil.walk_packages(kudio.__path__, 'kudio.'):
        if any(part.startswith('_') for part in info.name.split('.')):
            continue
        try:
            module = importlib.import_module(info.name)
        except ImportError:                          # the plots need matplotlib
            continue
        for name in getattr(module, '__all__', []):
            found = getattr(module, name)
            if inspect.isclass(found):
                members = [(f"{found.__name__}.{k}", getattr(v, '__func__', v))
                           for k, v in vars(found).items() if k == '__call__' or k[0] != '_']
            else:
                members = [(name, found)]
            for called, function in members:
                # what is wrapped is an old name for a function listed here
                if inspect.isfunction(function) and not hasattr(function, '__wrapped__') and any(
                        'ndarray' in str(p.annotation)
                        for p in inspect.signature(function).parameters.values()):
                    yield called, function.__module__.split('.')[-1]


def test_every_function_that_takes_an_array_is_here():
    listed = {name.split('(')[0] for name in EVERY} | {
        'SpectrogramStream.push', 'StreamEnhancer.process'}
    found = set(_takes_an_array())
    assert len(found) > 100
    assert sorted(name for name, module in found
                  if module not in METRICS and name not in listed | ELSEWHERE) == []


@pytest.mark.parametrize("name", EVERY)
def test_channels_first_is_refused(name):
    """librosa's layout. A filter run along its first axis filters across the
    channels, and remove_dc takes half the difference of the two."""
    stereo = np.stack([Y, OTHER])
    with pytest.raises(ValueError, match=r"^[A-Za-z_0-9.]+\(\): an array of shape \(2, \d+\) is "
                                         r"channels first, and kudio takes \(frames, channels\); "
                                         r"transpose it"):
        EVERY[name](stereo)


@pytest.mark.parametrize("name", EVERY)
def test_a_row_is_one_channel(name):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _assert_close(_flat(EVERY[name](Y[None])), _flat(EVERY[name](Y)), nan_ok=True,
                      abs=0, rel=0)


@pytest.mark.parametrize("name", MONO)
def test_a_column_is_one_channel(name):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _assert_close(_flat(MONO[name](Y[:, None])), _flat(MONO[name](Y)), nan_ok=True,
                      abs=0, rel=0)


def test_what_is_taken_for_channels_first():
    from kudio._samples import channels_last
    for shape in ((2, 3), (2, 16000), (8, 9), (100, 101)):
        with pytest.raises(ValueError, match="channels first"):
            channels_last(np.zeros(shape), 'f')
    for shape in ((2, 2), (3, 2), (16000, 2), (16, 16), (0, 2), (0,), (5,), (5, 1)):
        assert channels_last(np.zeros(shape), 'f').shape == shape
    # a number, three axes and no channel are no clip
    for shape in ((), (2, 3, 4), (5, 0)):
        with pytest.raises(ValueError, match=r"f\(\) takes \(frames,\) or \(frames, channels\)"):
            channels_last(np.zeros(shape), 'f')
    for shape in ((1, 1), (1, 2), (1, 16000), (1, 0)):
        assert channels_last(np.zeros(shape), 'f').shape == (shape[1],)


def test_the_bytes_of_an_array_are_interleaved():
    """Flattened channels first, an array is one channel after the other."""
    stereo = np.arange(8, dtype=np.int16).reshape(4, 2)
    assert np.frombuffer(wave_encode(stereo), dtype=np.int16).tolist() == list(range(8))
    for one in (np.arange(8), np.arange(8)[None], np.arange(8)[:, None]):
        assert wave_encode(one) == np.arange(8, dtype=np.int16).tobytes()
    with pytest.raises(ValueError, match=r"^wave_encode\(\): an array of shape \(2, 4\) is "
                                         r"channels first"):
        wave_encode(stereo.T)


def test_a_row_is_split_as_the_waveform_it_holds():
    """Every second row of it is none."""
    wave = np.arange(8, dtype=np.int16)
    left, right = kudio.wave_separate(wave[None], 2)
    assert (left.tolist(), right.tolist()) == ([0, 2, 4, 6], [1, 3, 5, 7])
    assert np.shares_memory(left, wave)
    with pytest.raises(kudio.FeatureError, match="wave_separate.*channels first"):
        kudio.wave_separate(np.zeros((2, 100)), 2)
    manager = kudio.AudioDataManager()
    with pytest.raises(ValueError, match="channels first"):
        manager.add_new_file("take", wave=[np.zeros((2, 100))], sr=SR, ch=1)
    assert len(manager) == 0


# -- a file's own channels ---------------------------------------------------------------
TINY = [(1, 2), (1, 6), (2, 3), (2, 6), (3, 4)]


def _tiny(folder, shape):
    """A file of fewer frames than channels."""
    x = np.round(np.random.default_rng(0).uniform(-0.5, 0.5, shape) * 32768) / 32768
    folder.mkdir()
    sf.write(str(folder / "a.wav"), x, SR, subtype='PCM_16')
    return x.astype(np.float32)


@pytest.mark.parametrize("shape", TINY, ids=repr)
def test_a_file_with_fewer_frames_than_channels_keeps_them(tmp_path, shape):
    """A file says how many channels it has, so nothing about it is guessed."""
    x = _tiny(tmp_path / "in", shape)
    y, _ = kudio.file_load(tmp_path / "in" / "a.wav", mono=False)
    assert np.array_equal(y, x)
    assert kudio.file_load(tmp_path / "in" / "a.wav", sr=8000, mono=False)[0].shape == (
        (shape[0] + 1) // 2, shape[1])
    info = kudio.audio_info(tmp_path / "in" / "a.wav")
    assert (info.frames, info.channels) == shape
    for n, (settings, expected) in enumerate([
            ({}, x), ({'trim_db': 200.0}, x), ({'peak': 0.9}, x / np.abs(x).max() * 0.9)]):
        out = tmp_path / str(n)
        result = kudio.convert_folder(tmp_path / "in", out, mono=False, **settings)
        assert result.failed == []
        assert sf.read(str(out / "a.wav"))[0] == pytest.approx(expected, abs=1 / 32768)
    result = kudio.convert_folder(tmp_path / "in", tmp_path / "lufs", mono=False, lufs=-23.0)
    assert "loudness() " in result.failed[0][1] and "channels first" not in result.failed[0][1]


@pytest.mark.parametrize("shape", TINY, ids=repr)
def test_the_command_line_keeps_them_too(tmp_path, capsys, shape):
    x = _tiny(tmp_path / "in", shape)
    path, out = str(tmp_path / "in" / "a.wav"), tmp_path / "out.wav"
    assert cli.main(["normalize", path, str(out), "--peak", "0.9"]) == 0
    assert sf.read(str(out))[0] == pytest.approx(x / np.abs(x).max() * 0.9, abs=1 / 32768)
    assert cli.main(["loudness", path]) == 2
    assert "channels first" not in capsys.readouterr().err
    with pytest.raises(ValueError, match="channels first"):      # until it is over
        kudio.normalize(np.zeros((2, 3)))


def test_a_callback_of_convert_folder_is_held_to_the_layout(tmp_path):
    _tiny(tmp_path / "in", (2, 3))
    refused = []

    def progress(done, total, path):
        with pytest.raises(ValueError, match="channels first"):
            kudio.reverse(np.zeros((2, 100)))
        refused.append(path)
    kudio.convert_folder(tmp_path / "in", tmp_path / "out", mono=False, progress=progress)
    assert len(refused) == 1


# -- integers -----------------------------------------------------------------------
@pytest.mark.parametrize("name", [n for n in EVERY if n not in (
    'reverse', 'Alignment.apply', 'wave_slicing')])          # they only move samples about
def test_64_bit_integers_are_refused(name):
    """astype(int) and a list of ints give them, and 16-bit samples read on
    their scale are 288 dB down: vad found no speech in them, loudness
    measured -inf, and mu_law came back all zeros."""
    for kind in (np.int64, np.uint64):
        with pytest.raises(ValueError, match=r"^[A-Za-z_0-9.]+\(\): the samples are u?int64, and "
                                             r"no audio file holds 64-bit integers"):
            EVERY[name](np.abs(np.round(Y * 32767)).astype(kind))


def test_a_list_of_ints_is_what_numpy_makes_of_it():
    samples = np.round(Y * 32767).astype(int).tolist()
    if np.asarray(samples).dtype != np.int64:
        pytest.skip("numpy makes 32-bit integers of it here")
    with pytest.raises(ValueError, match="int64"):
        kudio.loudness(samples, SR)
    with pytest.raises(ValueError, match="int64"):
        kudio.gain(samples, 3.0)


def test_every_other_integer_is_read_on_its_own_scale():
    from kudio._samples import as_float
    for kind in (np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32):
        limits = np.iinfo(kind)
        ends = as_float(np.array([limits.min, limits.max], dtype=kind))
        assert ends[0] == -1.0 and 0.99 < ends[1] < 1.0


def _u8(y):
    return np.clip(np.round(y * 128 + 128), 0, 255).astype(np.uint8)


@pytest.mark.parametrize("name", EVERY)
def test_unsigned_audio_is_read_around_its_midpoint(name):
    """128 is silence in an 8-bit file. Faded as a number it goes to 0, which
    is full scale the other way."""
    as_int = _u8(Y)
    as_float = (as_int.astype(np.float64) - 128.0) / 128.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _same_audio(EVERY[name](as_int), EVERY[name](as_float))


@pytest.mark.parametrize("kind", [np.uint8, np.uint16, np.uint32])
def test_unsigned_silence_is_resampled_to_silence(kind):
    """Taken as a number, a silence of 128 swings a quarter of full scale at
    both ends, where the filter runs past the clip."""
    middle = (int(np.iinfo(kind).max) + 1) // 2
    out = kudio.resample(np.full(1600, middle, dtype=kind), SR, 22050)
    assert out.dtype == kind and len(out) == 2205 and (out == middle).all()


@pytest.mark.parametrize("kind", [np.int8, np.int16, np.int32, np.uint8, np.uint16, np.uint32])
def test_full_scale_is_held_to_the_type_when_it_is_resampled(kind):
    """The largest int32 is not a float32, and held to it as one, a square wave
    changes the sign of every peak."""
    limits = np.iinfo(kind)
    middle = (int(limits.max) + 1) // 2 if limits.min == 0 else 0
    full = middle or int(limits.max) + 1
    square = np.sin(2 * np.pi * 50 * np.arange(SR) / SR + 0.1) > 0
    out = kudio.resample(np.where(square, limits.max, limits.min).astype(kind), SR, 22050)
    plain = kudio.resample(np.where(square, 1.0, -1.0).astype(np.float32), SR, 22050)
    assert out.dtype == kind
    _assert_close((out.astype(np.float64) - middle) / full, np.clip(plain, -1.0, 1.0),
                  abs=3 * 2.0 ** -7 if limits.bits == 8 else 1e-3)


def _floats(value):
    """*value* as float64, if it is an array of floats or a list of nothing but floats."""
    if isinstance(value, np.ndarray):
        return value.astype(np.float64) if value.dtype.kind == 'f' else None
    if isinstance(value, list) and all(type(x) is float for x in value):
        return np.array(value, dtype=np.float64)
    return None


def _assert_close(got, expected, rel=None, abs=None, nan_ok=False):
    """``assert got == pytest.approx(expected, rel=rel, abs=abs, nan_ok=nan_ok)``. approx
    goes one number at a time in Python, 0.09 s for the 38 400 samples of a seam, so numpy
    looks first, with approx's tolerance: where every number is within it, approx would say
    so too. Whatever else is left to approx, and so is the message."""
    a, e = _floats(got), _floats(expected)
    if a is not None and e is not None and type(got) is type(expected) and a.shape == e.shape:
        with np.errstate(invalid='ignore', over='ignore'):
            if rel is None and abs is not None:
                tolerance = abs
            else:
                tolerance = np.maximum((1e-6 if rel is None else rel) * np.abs(e),
                                       1e-12 if abs is None else abs)
            near = (a == e) | (np.isfinite(e) & (np.abs(e - a) <= tolerance))
        if nan_ok:
            near |= np.isnan(a) & np.isnan(e)
        if near.all():
            return
    assert got == pytest.approx(expected, rel=rel, abs=abs, nan_ok=nan_ok)


def _same_audio(got, expected):
    if isinstance(got, (list, tuple)):
        assert len(got) == len(expected)
        for a, b in zip(got, expected):
            _same_audio(a, b)
    elif isinstance(got, np.ndarray) and got.dtype == np.uint8:
        # rounded to a step of 1/128 and held to the type's range
        _assert_close((got.astype(np.float64) - 128.0) / 128.0,
                      np.clip(expected, -1.0, 127 / 128), abs=2 / 128)
    else:
        _assert_close(_flat(got), _flat(expected), rel=1e-4, abs=2e-5, nan_ok=True)


SEAM_TYPES = {
    'int16': lambda y: np.round(y * 32767).astype(np.int16),
    'int32': lambda y: np.round(y * 2147483000).astype(np.int32),
    'uint8': _u8,
    'float32': lambda y: y.astype(np.float32),
    'float64': lambda y: y.astype(np.float64),
}


@pytest.mark.parametrize("duration", [0.005, 0.0])
@pytest.mark.parametrize("first", SEAM_TYPES)
@pytest.mark.parametrize("second", SEAM_TYPES)
def test_a_seam_joins_two_types_at_the_level_each_was_recorded_at(first, second, duration):
    """A recorder gives int16 and a file float: end to end as numbers, the
    second half of that is either silence or 32767 times too loud."""
    a, b = SEAM_TYPES[first](Y), SEAM_TYPES[second](OTHER)
    plain = kudio.crossfade(Y.astype(np.float64), OTHER.astype(np.float64), SR, duration)
    for joined in (kudio.crossfade(a, b, SR, duration),
                   kudio.splice([a, b], SR, duration, shape='equal_power')):
        assert joined.dtype == a.dtype
        scale = {'int16': 32768.0, 'int32': 2.0 ** 31, 'uint8': 128.0}.get(first, 1.0)
        heard = (joined.astype(np.float64) - (128 if first == 'uint8' else 0)) / scale
        _assert_close(heard, plain, abs=2 / 128)
