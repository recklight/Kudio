# -*- coding: utf-8 -*-
"""What is in the output folder of a Synthesizer when a call to it raises.

A run with ``overwrite=True`` empties its folder, and the folder may hold the
only copy of a corpus that took a night to make. So whatever can be known
without the samples is found out first, and a call that raises for it leaves
every file where it was.
"""
import contextlib
import io
import os
import shutil
import stat
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import kudio
from kudio import Synthesizer
from kudio.core import synth
from kudio.core.io import _resampled_length
from kudio.effects import channel as link

NAN, INF = float('nan'), float('inf')

#: how many random calls are made; a few thousand take minutes
CASES = int(os.environ.get("KUDIO_SYNTH_CASES", 240))


def _tone(n, sr=16000, channels=1):
    y = 0.2 * np.sin(2 * np.pi * min(220.0, sr / 5) * np.arange(n) / sr)
    y = y + 0.02 * np.random.default_rng(n).standard_normal(n)
    return y if channels == 1 else np.stack([y * (1 - 0.3 * c) for c in range(channels)], axis=1)


def _put(path, y, sr=16000, **kwargs):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), y, sr, **{'format': 'WAV', 'subtype': 'PCM_16', **kwargs})
    return path


def _endless_flac(path):
    """As an encoder writes FLAC to a pipe: it does not know the length yet, and
    cannot go back to the header when it does."""
    _put(path, _tone(4000), format='FLAC')
    data = bytearray(path.read_bytes())
    data[21] &= 0xF0                             # the 36 bits of the total
    data[22:26] = bytes(4)
    path.write_bytes(bytes(data))
    return path


def _cut(path, share=0.5):
    data = path.read_bytes()
    path.write_bytes(data[:int(len(data) * share)])
    return path


class Folder:
    """An output folder with something in it, and inputs beside it."""

    def __init__(self, root):
        self.root, self.out = root, root / "out"
        self.clean = _put(root / "c" / "a.wav", _tone(3200))
        self.noise = _put(root / "n" / "x.wav", _tone(4800) * 0.5)
        (self.out / "sub").mkdir(parents=True)
        (self.out / "keep.txt").write_text("from an earlier run")
        _put(self.out / "old.wav", _tone(800, 8000), 8000)
        _put(self.out / "sub" / "older.wav", _tone(800, 8000), 8000)
        self.before = self.now()

    def now(self):
        return {p.relative_to(self.root).as_posix(): p.read_bytes()
                for p in sorted(self.root.rglob("*")) if p.is_file()
                and p.relative_to(self.root).parts[0] not in ("c", "n", "b")}

    def build(self, **kwargs):
        return Synthesizer(self.root / "c", self.root / "n", self.out,
                           **{'snr_ratio': 0, **kwargs})

    @contextlib.contextmanager
    def refused(self, *errors, match=None):
        """The call raises, and everything is where it was, with nothing beside
        it."""
        with pytest.raises(errors, match=match):
            yield
        assert self.now() == self.before


@pytest.fixture
def folder(tmp_path):
    return Folder(tmp_path)


# -- the packets of every output ------------------------------------------------

@pytest.mark.parametrize("channel", [
    {'loss': 0.9, 'burst': 1}, {'loss': 0.9, 'burst': 4}, {'loss': 1.0, 'burst': 4},
    {'loss': 0.6, 'burst': 1}, {'loss': 0.8, 'burst': 1.5, 'packet_ms': 5},
    {'preset': 'telephone', 'loss': 0.9, 'burst': 2}, {'bits': 8, 'loss': 0.95, 'burst': 1}],
    ids=str)
@pytest.mark.parametrize("called", [{}, {'mode': 'inc', 'target_sr': 8000, 'is_silence': True}],
                         ids=["reg", "inc at 8 kHz"])
def test_a_loss_that_no_packets_are_left_for_empties_nothing(folder, channel, called):
    """A second of noise with half of its packets gone had stood for all of
    it."""
    _put(folder.clean, _tone(32000))
    syx = folder.build(channel=channel, snr_ratio=(0, 5))
    with folder.refused(kudio.FeatureError, match="surviving packets to keep the runs apart"):
        syx.syn(seed=0, **called)
    assert syx.seed is None and syx.channels == {} and syx.manifest() == []


def test_the_output_that_fails_is_named(folder):
    """Each has its own length and its own seed, and here most of them fit."""
    for k in range(12):
        _put(folder.root / "c" / f"f{k}.wav", _tone(3200 + 1600 * k))
    channel = {'loss': 0.62, 'burst': 1.7}
    syx = folder.build(channel=channel, snr_ratio=(0, 5))
    fails = None
    for seed in range(40):
        plan = synth._reg_pairing(13, 1, 2)[1]
        outs = [folder.out / f"{c.stem}_x_{(0, 5)[plan[k]]}dB.wav"
                for k, c in enumerate(syx.cleanWaves)]
        bad = []
        for out, cln in zip(outs, syx.cleanWaves):
            own = (synth._file_seed(seed, out, folder.out) * 2654435761) & 0x7FFFFFFF
            try:
                kudio.dropouts(np.zeros(sf.info(str(cln)).frames), 16000, seed=own, **channel)
            except kudio.FeatureError:
                bad.append(out)
        if 0 < len(bad) < 5:
            fails = seed, bad
        with contextlib.ExitStack() as stack:
            if bad:
                stack.enter_context(folder.refused(kudio.FeatureError))
                syx.syn(seed=seed)
    assert fails, "a run of which some outputs fail, or this tests nothing"
    seed, bad = fails
    with pytest.raises(kudio.FeatureError) as caught:
        syx.syn(seed=seed)
    assert str(caught.value).startswith(f"{bad[0]}: loss=")


@pytest.mark.parametrize("target_sr", [None, 8000, 11025, 48000])
@pytest.mark.parametrize("clean_sr", [16000, 22050])
def test_the_packets_are_those_of_the_file_that_is_written(folder, clean_sr, target_sr):
    """As long as that is at the rate it is written at."""
    pytest.importorskip("librosa")
    _put(folder.clean, _tone(30001, clean_sr), clean_sr)
    out, sr = folder.out / "a_x_0dB.wav", target_sr or clean_sr
    # 7.3 ms is a whole number of samples at none of them
    channel = {'loss': 0.9, 'burst': 1.5, 'packet_ms': 7.3}
    samples = np.zeros(len(kudio.resample(np.zeros(30001, np.float32), clean_sr, sr)))
    own = (synth._file_seed(3, out, folder.out) * 2654435761) & 0x7FFFFFFF
    with pytest.raises(kudio.FeatureError) as expected:
        kudio.dropouts(samples, sr, seed=own, **channel)
    syx = folder.build(channel=channel)
    with pytest.raises(kudio.FeatureError) as caught:
        syx.syn(seed=3, target_sr=target_sr)
    assert str(caught.value) == f"{out}: {expected.value}"
    assert folder.now() == folder.before


def test_a_link_that_every_output_survives_runs(folder):
    syx = folder.build(channel={'loss': 0.6, 'burst': 8}, snr_ratio=(0, 5))
    for seed in range(200):
        with contextlib.suppress(kudio.FeatureError):
            assert len(syx.syn(seed=seed)) == 1
            break
    assert not (folder.out / "keep.txt").exists(), "and then the folder is its own"


@pytest.mark.parametrize("block", range(4))
def test_what_is_found_out_of_the_packets_is_what_dropouts_does(block):
    rng = np.random.default_rng(block)
    raised = 0
    for _ in range(150):
        n, sr = int(rng.integers(0, 40000)), int(rng.choice([8000, 16000, 44100]))
        spec = {'loss': float(rng.choice([0.0, 0.05, 0.5, 0.55, 0.7, 0.9, 1.0])),
                'burst': [None, 1, 1.3, 2.5, 20, 1e300][int(rng.integers(6))],
                'packet_ms': float(rng.choice([1, 5, 20, 100, 5000]))}
        seed = int(rng.integers(0, 2 ** 31))

        def error(call):
            try:
                call()
            except kudio.FeatureError as e:
                return str(e)

        said = error(lambda: kudio.dropouts(np.zeros(n), sr, seed=seed, **spec))
        assert error(lambda: link._check_loss(n, sr, spec, seed)) == said, (n, sr, spec, seed)
        raised += said is not None
    assert 20 < raised < 140, "both answers came up"


def test_half_of_the_packets_can_always_be_lost_in_runs():
    """Below that nothing has to be tried, which is most runs."""
    for packets in range(0, 400):
        for loss in (0.05, 0.3, 0.499, 0.5):
            for burst in (1, 1.2, 4):
                link._check_loss(packets * 320, 16000,
                                 {'loss': loss, 'burst': burst, 'packet_ms': 20}, packets)


@pytest.mark.parametrize("orig, target", [(16000, 8000), (8000, 16000), (44100, 16000),
                                          (16000, 44100), (48000, 11025), (22050, 7),
                                          (16000, 16000), (384000, 8000)])
def test_the_length_of_a_resampled_clip_is_known(orig, target):
    pytest.importorskip("librosa")
    for n in [*range(1, 40), 999, 1000, 1001, 4410]:
        assert len(kudio.resample(np.zeros(n, np.float32), orig, target)) \
            == _resampled_length(n, orig, target)


def test_an_mp3_that_was_cut_is_counted(folder):
    """Its header still tells the whole length, so the packets are those of
    what can be decoded."""
    if 'MP3' not in sf.available_formats():
        pytest.skip("this libsndfile writes no MP3")
    _put(folder.root / "c" / "a.wav", _tone(48000), format='MP3', subtype='MPEG_LAYER_III')
    _cut(folder.root / "c" / "a.wav")
    info = sf.info(str(folder.root / "c" / "a.wav"))
    with sf.SoundFile(str(folder.root / "c" / "a.wav")) as f:
        assert len(f.read()) < info.frames, "shorter than its header, or this tests nothing"
    for seed in range(12):
        syx = folder.build(channel={'loss': 0.7, 'burst': 2.2}, snr_ratio=0)
        try:
            syx.syn(seed=seed)
        except kudio.FeatureError as e:
            assert str(e).startswith(str(folder.out / "a_x_0dB.wav"))
            assert folder.now() == folder.before
        else:
            shutil.rmtree(folder.out)
            folder.__init__(folder.root)
            _cut(_put(folder.root / "c" / "a.wav", _tone(48000), format='MP3',
                      subtype='MPEG_LAYER_III'))


# -- the inputs -----------------------------------------------------------------

BAD_HEADERS = {
    'no wave file': (lambda p: p.write_bytes(bytes(range(256)) * 8), "could not read"),
    'nothing in it': (lambda p: p.write_bytes(b''), "the file is empty, 0 bytes"),
    'half a header': (lambda p: _cut(_put(p, _tone(100)), 0.1), "could not read"),
    'GSM, which is not read at a rate': (
        lambda p: _put(p, _tone(2400), 8000, subtype='GSM610'), "non-seekable"),
    'a FLAC without a length': (_endless_flac, "could not read"),
    'no frames': (lambda p: _put(p, np.zeros(0)), "(clean file has no signal|noise file is "
                                                   "empty)"),
    'one frame': (lambda p: _put(p, _tone(1)), "a single sample long at 16000 Hz"),
}


@pytest.mark.parametrize("what", ["clean", "noise"])
@pytest.mark.parametrize("name", BAD_HEADERS)
def test_an_input_that_is_bad_by_its_header_empties_nothing(folder, name, what):
    make, said = BAD_HEADERS[name]
    bad = folder.root / what[0] / "bad.wav"
    make(bad)
    syx = folder.build(snr_ratio=(0, 5))
    with folder.refused(kudio.SynthesisError, match=said):
        syx.syn(mode='inc', seed=0)
    with pytest.raises(kudio.SynthesisError, match="bad.wav"):
        syx.syn(mode='inc', seed=0)


@pytest.mark.parametrize("what", ["clean", "noise", "background"])
@pytest.mark.parametrize("name", BAD_HEADERS)
def test_extra_mode_looks_at_the_headers_too(folder, name, what):
    make, said = BAD_HEADERS[name]
    shutil.copytree(folder.root / "n", folder.root / "b")
    make(folder.root / what[0] / "bad.wav")
    syx = folder.build()
    with folder.refused(kudio.SynthesisError):
        syx.syn_extra_mode(folder.root / "b", (5,), seed=0, is_pool=False)
    with pytest.raises(kudio.SynthesisError, match="bad.wav"):
        syx.syn_extra_mode(folder.root / "b", (5,), seed=0, is_pool=False)
    assert syx.seed is None and syx.synWavesList is None


def test_extra_mode_looks_at_all_of_it_before_it_mixes_any(folder, monkeypatch):
    """The noises and the backgrounds may take an hour to mix."""
    def save(*args, **kwargs):
        pytest.fail("something was mixed")

    monkeypatch.setattr(synth, "save_wave", save)
    (folder.root / "c" / "bad.wav").write_bytes(b"not a wave file")
    with folder.refused(kudio.SynthesisError, match="could not read clean file .*bad.wav"):
        folder.build().syn_extra_mode(folder.root / "n", (5,), seed=0, is_pool=False)
    (folder.root / "c" / "bad.wav").unlink()

    def missing():
        raise kudio.DependencyError('librosa')

    monkeypatch.setattr(kudio._optional, "librosa", missing)
    _put(folder.clean, _tone(3200, 8000), 8000)
    with folder.refused(kudio.DependencyError, match="librosa"):
        folder.build().syn_extra_mode(folder.root / "n", (5,), seed=0, is_pool=False)


@pytest.mark.parametrize("attribute", ["cleanWaves", "noiseWaves"])
def test_an_input_that_is_gone_empties_nothing(folder, attribute):
    syx = folder.build()
    getattr(syx, attribute).append(folder.root / "c" / "gone.wav")
    with folder.refused(kudio.SynthesisError, match="gone.wav: no such file"):
        syx.syn(mode='inc', seed=0)
    with folder.refused(kudio.SynthesisError, match="gone.wav: no such file"):
        syx.syn_extra_mode(folder.root / "n", (5,), seed=0, is_pool=False)


def test_a_clip_that_one_sample_is_left_of_empties_nothing(folder):
    """0.15 s at 1 Hz is one sample."""
    pytest.importorskip("librosa")
    syx = folder.build()
    with folder.refused(kudio.SynthesisError, match="a single sample long at 1 Hz"):
        syx.syn(seed=0, target_sr=1)


def test_a_clean_file_nothing_is_made_of_need_not_be_read(folder):
    """rdn_choice_num leaves it out, and it is no business of this run."""
    (folder.root / "c" / "bad.wav").write_bytes(b"not a wave file")
    syx = folder.build()
    seeds = [s for s in range(40) if np.random.default_rng(s).choice(2, 1, False)[0] == 0]
    assert len(syx.syn(seed=seeds[0], rdn_choice_num=1)) == 1
    assert (folder.out / "a_x_0dB.wav").exists()


def test_a_warning_about_a_header_that_is_an_error_empties_nothing(folder):
    _cut(folder.clean)
    syx = folder.build()
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*cut short")
        with folder.refused(kudio.SynthesisError, match="may have been cut short"):
            syx.syn(seed=0)
    with pytest.warns(UserWarning) as caught:
        syx.syn(seed=0)
    assert ["cut short" in str(w.message) for w in caught] == [True], "said once"


# -- the settings ---------------------------------------------------------------

@pytest.mark.parametrize("value", [-300.1, -400, -1e5, 3000.5, 1e5, 1e308])
def test_an_snr_no_float_can_hold_the_noise_at_is_refused(folder, value):
    with pytest.raises(ValueError, match="snr_ratio values must lie between -300 and 3000 dB"):
        folder.build(snr_ratio=(0, value))
    syx = folder.build()
    with folder.refused(ValueError, match="background_snr values must lie between"):
        syx.syn_extra_mode(folder.root / "n", (0, value), seed=0, is_pool=False)
    syx.snrRatio = [value]
    with folder.refused(ValueError, match="snr_ratio values must lie between"):
        syx.syn(seed=0)
    with folder.refused(ValueError, match="snr_ratio values must lie between"):
        syx.syn_extra_mode(folder.root / "n", (0,), seed=0, is_pool=False)


@pytest.mark.parametrize("value", [-300, -100, 100, 3000])
def test_the_snrs_within_reach_are_made(folder, value):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")              # what is clipped at -300 dB
        assert len(folder.build(snr_ratio=value).syn(seed=0)) == 1


@pytest.mark.parametrize("name, value, errors", [
    ('snrRatio', [NAN], ValueError), ('rt60Ratio', [-1.0], ValueError),
    ('subtype', 'FOO', kudio.AudioIOError), ('channel', {'loss': 0.9, 'burst': 1},
                                             kudio.FeatureError),
    ('channel', {'loss': 3}, kudio.FeatureError), ('snrRatio', [], kudio.SynthesisError),
    ('cleanWaves', [], kudio.SynthesisError), ('noiseWaves', [], kudio.SynthesisError),
], ids=str)
@pytest.mark.parametrize("mode", ["reg", "inc"])
def test_an_attribute_that_was_changed_empties_nothing(folder, name, value, errors, mode):
    _put(folder.clean, _tone(32000))
    syx = folder.build()
    setattr(syx, name, value)
    with folder.refused(errors):
        syx.syn(mode=mode, seed=0)


@pytest.mark.parametrize("name", ["cleanWaves", "snrRatio"])
def test_extra_mode_with_nothing_to_make_empties_nothing(folder, name):
    syx = folder.build()
    setattr(syx, name, [])
    with folder.refused(kudio.SynthesisError, match="there is nothing to synthesise"):
        syx.syn_extra_mode(folder.root / "n", (5,), seed=0, is_pool=False)


def test_a_rate_the_subtype_is_not_written_at_empties_nothing(folder, monkeypatch):
    """Which ones those are is for libsndfile to say, when it is asked to
    write."""
    def write(path, wave, sr, **kwargs):
        if sr == 8000:
            raise sf.LibsndfileError(25)
        return real(path, wave, sr, **kwargs)

    pytest.importorskip("librosa")
    _put(folder.root / "c" / "b.wav", _tone(2000, 8000), 8000)
    real = sf.write
    monkeypatch.setattr(sf, "write", write)
    with folder.refused(kudio.AudioIOError):
        folder.build().syn(seed=0)
    with pytest.raises(kudio.AudioIOError) as caught:
        folder.build().syn(seed=0)
    assert str(caught.value).startswith(f"Could not write audio file: {folder.out}{os.sep}")
    assert str(caught.value).count("Could not write") == 1


def test_a_rate_no_header_holds_empties_nothing(folder):
    pytest.importorskip("librosa")
    with folder.refused(kudio.AudioIOError, match="Could not write audio file"):
        folder.build().syn(seed=0, target_sr=10 ** 12)


def test_a_room_that_does_not_fit_in_memory_empties_nothing(folder):
    with folder.refused(MemoryError):
        folder.build(rt60=1e12).syn(seed=0)


# -- the names ------------------------------------------------------------------

@pytest.mark.parametrize("called", [{}, {'is_add_clean': True}], ids=str)
def test_a_name_that_is_too_long_empties_nothing(folder, called):
    """Two names of 130 letters are a name of more than 255."""
    folder.clean.rename(folder.root / "c" / f"{'a' * 130}.wav")
    folder.noise.rename(folder.root / "n" / f"{'b' * 130}.wav")
    syx = folder.build()
    with folder.refused(kudio.AudioIOError, match="Could not write audio file: .*aaaa.*bbbb"):
        syx.syn(seed=0, **called)
    with folder.refused(kudio.AudioIOError, match="Could not write audio file"):
        syx.syn_extra_mode(folder.root / "n", (5,), seed=0, is_pool=False)


def test_the_widest_name_is_tried_as_well_as_the_longest_path(folder):
    """The path is longer deep in the tree, where the name is shorter."""
    syx = folder.build()
    tried = []

    def save(path, wave, sr, subtype):
        tried.append(path.name)
        real(path, wave, sr, subtype=subtype)

    real, synth.save_wave = synth.save_wave, save
    try:
        deep = folder.out / "dr1" / "fcjf0" / "sentences" / "read" / "sa1.wav"
        wide = folder.out / "a_name_of_its_own_length.wav"
        syx._try_writing([folder.out / "a.wav", deep, wide, folder.out / "b.wav"], {16000, 8000})
    finally:
        synth.save_wave = real
    assert tried == [wide.name, deep.name, deep.name]
    assert folder.now() == folder.before


def test_a_name_that_would_be_a_file_and_a_folder_empties_nothing(folder):
    _put(folder.root / "c" / "a_x_0dB.wav" / "deep.wav", _tone(1600))
    with folder.refused(kudio.SynthesisError, match="would be a file and the folder of"):
        folder.build().syn(seed=0)


def test_a_clean_file_that_is_copied_and_not_mixed_has_to_be_there(folder):
    """The name it asks for belongs to another, and is_add_clean copies it all
    the same."""
    _put(folder.root / "c" / "d" / "a.wav", _tone(3000))
    syx = folder.build()
    (folder.root / "c" / "d" / "a.wav").unlink()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with folder.refused(kudio.SynthesisError, match="could not read clean file .*a.wav"):
            syx.syn(seed=0, mkdir_parents=False, is_add_clean=True)
        assert len(syx.syn(seed=0, mkdir_parents=False)) == 1, "and is not needed otherwise"


@pytest.mark.parametrize("name", ["o", "mixed", "a much longer name than that"])
def test_the_folder_the_names_are_tried_in_has_a_name_as_long(tmp_path, name):
    """Or a path would be too long below the one and not below the other."""
    (tmp_path / name).mkdir()
    with synth._scratch_beside(tmp_path / name) as scratch:
        assert len(scratch.name) == len(name) and scratch.parent == tmp_path
        assert scratch.is_dir() and scratch != tmp_path / name
    assert [p.name for p in tmp_path.iterdir()] == [name]


def test_a_folder_whose_name_is_one_letter_has_room_beside_it(tmp_path):
    """Sixteen letters, and then a name that is longer."""
    (tmp_path / "o").mkdir()
    with contextlib.ExitStack() as stack:
        names = [stack.enter_context(synth._scratch_beside(tmp_path / "o")).name
                 for _ in range(20)]
    assert len(set(names)) == 20
    assert [p.name for p in tmp_path.iterdir()] == ["o"]


@pytest.mark.parametrize("name", ["o", "ab", "a name with spaces", "長檔名"])
def test_the_folder_may_be_called_anything(tmp_path, name):
    folder = Folder(tmp_path)
    (tmp_path / name).mkdir()
    (tmp_path / name / "keep.txt").write_text("x")
    syx = Synthesizer(tmp_path / "c", tmp_path / "n", tmp_path / name, snr_ratio=0)
    assert len(syx.syn(seed=0)) == 1
    assert [p.name for p in (tmp_path / name).iterdir()] == ["a_x_0dB.wav"]
    assert folder.now().keys() - folder.before.keys() == {f"{name}/a_x_0dB.wav"}


# -- the folder itself ----------------------------------------------------------

@pytest.mark.skipif(os.name != 'nt', reason="elsewhere an open file can be deleted")
@pytest.mark.parametrize("where", ["keep.txt", "sub/older.wav"])
def test_a_file_that_is_open_elsewhere_empties_nothing(folder, where):
    """Deleted one by one, the files before it are gone when it is refused."""
    with open(folder.out / where, 'rb'):
        with folder.refused(kudio.SynthesisError, match="cannot be deleted"):
            folder.build().syn(seed=0)
        with folder.refused(kudio.SynthesisError, match="cannot be deleted"):
            folder.build().syn_extra_mode(folder.root / "n", (5,), seed=0, is_pool=False)
    assert len(folder.build().syn(seed=0)) == 1


@pytest.mark.skipif(os.name != 'nt', reason="there is no such handle elsewhere")
@pytest.mark.parametrize("where", [".", "sub", "old.wav"])
def test_what_shares_its_handle_is_not_in_the_way(folder, where):
    """A window that shows the folder, an indexer that reads a file: Windows
    deletes what they have open."""
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                   wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD,
                                   wintypes.HANDLE]
    # to read, sharing everything, something that exists, folders included
    handle = kernel.CreateFileW(str(folder.out / where), 1, 7, None, 3, 0x02000000, None)
    assert handle not in (None, wintypes.HANDLE(-1).value)
    try:
        assert synth._in_the_way(folder.out) is None
        assert folder.now() == folder.before, "and nothing came of looking"
    finally:
        kernel.CloseHandle(wintypes.HANDLE(handle))


@pytest.mark.skipif(os.name != 'nt', reason="elsewhere a read-only file can be deleted")
def test_a_read_only_file_empties_nothing(folder):
    """As a copy of a corpus from a disc is."""
    os.chmod(folder.out / "sub" / "older.wav", stat.S_IREAD)
    try:
        with folder.refused(kudio.SynthesisError, match="older.wav cannot be deleted .*read-only"):
            folder.build().syn(seed=0)
    finally:
        os.chmod(folder.out / "sub" / "older.wav", stat.S_IWRITE | stat.S_IREAD)


@pytest.mark.skipif(os.name == 'nt' or not hasattr(os, 'geteuid') or os.geteuid() == 0,
                    reason="Windows has no such permission, and root does not mind")
@pytest.mark.parametrize("where", ["out/sub", "out", "."])
def test_a_folder_that_may_not_be_written_empties_nothing(folder, where):
    os.chmod(folder.root / where, 0o555)
    try:
        with folder.refused(kudio.SynthesisError, match="refusing to empty"):
            folder.build().syn(seed=0)
    finally:
        os.chmod(folder.root / where, 0o755)


def _link(target, name):
    """*name*, a second way to *target*."""
    if os.name == 'nt':
        import _winapi
        _winapi.CreateJunction(str(target), str(name))
    else:
        name.symlink_to(target, target_is_directory=True)
    return name


def test_an_input_is_found_by_another_way_to_the_folder(folder):
    """The output folder spelled through a link holds the inputs all the same."""
    (folder.root / "real").mkdir()
    for name in ("c", "n"):
        shutil.copytree(folder.root / name, folder.root / "real" / "mixed" / name)
    _link(folder.root / "real", folder.root / "way")
    real, way = folder.root / "real" / "mixed", folder.root / "way" / "mixed"
    for inputs, out in ((real, way), (way, real)):
        with pytest.raises(kudio.SynthesisError, match="it holds the input"):
            Synthesizer(inputs / "c", inputs / "n", out, snr_ratio=0).syn(seed=0)
        assert (real / "c" / "a.wav").exists() and (real / "n" / "x.wav").exists()
    if os.name == 'nt':
        os.rmdir(folder.root / "way")


def test_an_input_that_is_a_link_into_the_folder_is_found(folder):
    try:
        (folder.root / "c" / "b.wav").symlink_to(folder.out / "old.wav")
    except OSError as e:
        pytest.skip(f"no links here: {e}")
    with folder.refused(kudio.SynthesisError, match="it holds the input"):
        folder.build().syn(seed=0)


def test_a_link_to_a_folder_is_not_emptied(folder):
    way = _link(folder.out, folder.root / "way")
    try:
        syx = Synthesizer(folder.root / "c", folder.root / "n", way, snr_ratio=0)
        with pytest.raises(kudio.SynthesisError, match="it is a link to"):
            syx.syn(seed=0)
        assert (folder.out / "keep.txt").exists() and (folder.out / "sub" / "older.wav").exists()
    finally:
        os.rmdir(way) if os.name == 'nt' else way.unlink()


def test_a_link_in_the_folder_is_deleted_and_not_what_it_points_to(folder):
    (folder.root / "elsewhere").mkdir()
    (folder.root / "elsewhere" / "theirs.txt").write_text("not yours")
    _link(folder.root / "elsewhere", folder.out / "way")
    with open(folder.root / "elsewhere" / "theirs.txt"):
        assert len(folder.build().syn(seed=0)) == 1, "and is not in the way when it is open"
    assert [p.name for p in folder.out.iterdir()] == ["a_x_0dB.wav"]
    assert (folder.root / "elsewhere" / "theirs.txt").read_text() == "not yours"


def test_a_pool_that_does_not_start_empties_nothing(folder, monkeypatch):
    """A daemon may not have children, and a script without its ``__main__``
    never hears from them."""
    def pool(n):
        raise AssertionError("daemonic processes are not allowed to have children")

    monkeypatch.setattr(synth, "Pool", pool)
    syx = folder.build(snr_ratio=[0.5 * k for k in range(210)])     # it takes 200
    with folder.refused(AssertionError):
        syx.syn(mode='inc', seed=0, is_pool=True)
    with folder.refused(AssertionError):
        syx.syn_extra_mode(folder.root / "n", (5,), seed=0, is_pool=True)


@pytest.mark.parametrize("make", [
    lambda p: _put(p, np.zeros(2400)), lambda p: _put(p, np.full(2400, 0.1)),
    lambda p: _put(p, np.where(np.arange(2400) == 7, NAN, 0.1), subtype='FLOAT'),
    lambda p: _cut(_put(p, _tone(4000), format='FLAC'))],
    ids=["silent", "constant", "NaN", "damaged"])
def test_what_stage_one_of_extra_mode_finds_empties_nothing(folder, make):
    """It mixes the noises with the backgrounds into a folder of its own."""
    make(folder.root / "b" / "bad.wav")
    with folder.refused(kudio.SynthesisError, match="bad.wav"):
        folder.build().syn_extra_mode(folder.root / "b", (5,), seed=0, is_pool=False)


def test_a_call_that_returns_none_leaves_the_folder(folder):
    with pytest.warns(UserWarning, match="nothing was synthesised"):
        assert folder.build().syn(seed=0, overwrite=False) is None
    assert folder.now() == folder.before


# -- anything at all ------------------------------------------------------------

LEGAL = {
    'snr_ratio': [(-5, 0, 5), 0, [10], (-5, 5), np.arange(-5, 6, 5), (0, 0), 40, np.float32(3)],
    'rt60': [None, None, 0.3, (0.2, 0.5), 0.01, 0.004],
    'drr_db': [0.0, 6, -5, 12, -12, 30],
    'channel': [None, None, 'telephone', 'voip', 'mu_law', {'loss': 0.5, 'burst': 1},
                {'bits': 8}, {'preset': 'voip', 'loss': 0.3}],
    'write_targets': [False, True],
    'subtype': ['PCM_16', 'PCM_24', 'FLOAT', 'ULAW'],
    'mode': ['reg', 'inc'],
    'mkdir_parents': [True, False],
    'is_add_clean': [False, True],
    'rdn_choice_num': [0, 1, 2],
    'is_silence': [False, True],
    'p_silence': [0.02, 0.5],
    'overwrite': [True, True, True, False],
    'target_sr': [None, None, 8000, 22050.0, np.int32(11025)],
    'seed': [0, 5, None, 2 ** 40],
    'background_snr': [(5,), 0, (0, 10), np.arange(0, 6, 5)],
}
HARD = {
    'snr_ratio': [(), NAN, (0, INF), '5', (None,), 1e5, -1e5, 400, -400, -1000, 3000, 3083],
    'rt60': [0, -1, NAN, True, 'x', 1e-5, (0.3, None), 1e12, 1e-300, 0.001, ()],
    'drr_db': [NAN, 'a', 1e4, -1e4, -60, 300],
    'channel': ['phone', 3, {'loss': 2}, {'loss': NAN}, {'loss': 0.9, 'burst': 1},
                {'loss': 0.9, 'burst': 4}, {'loss': 1.0, 'burst': 4}, {'loss': 1.0},
                {'loss': 0.6, 'burst': 1}, {'loss': 0.7, 'burst': 1.5}, {'loss': 0.8, 'burst': 3},
                {'loss': 0.55, 'burst': 1.1}, {'loss': 0.9, 'burst': 1e300},
                {'loss': 0.1, 'burst': 0.5}, {'loss': 0.1, 'packet_ms': 0},
                {'loss': 0.1, 'packet_ms': 1e300}, {'loss': 0.9, 'packet_ms': 1e308, 'burst': 2},
                {'loss': 0.6, 'packet_ms': 1000, 'burst': 1}, {'bits': 1}, {'bits': 8.5},
                {'bits': 65}, {'codec': 'x'}, {'foo': 1},
                {'band': True, 'loss': 0.8, 'burst': 1.2}],
    'subtype': ['FOO', '', None, 'VORBIS', 'GSM610', 'IMA_ADPCM', 'NMS_ADPCM_16',
                'MPEG_LAYER_III'],
    'mode': ['max', 'foo', None],
    'rdn_choice_num': [-1, 1.5, 'a', None, 100],
    'p_silence': [0, 1, NAN, 2],
    'target_sr': ['16000', -1, 0, True, 8000.5, NAN, 1, 7, 100, 599, 601, 1000000, 10 ** 12],
    'seed': [-1, 1.5, 'a', NAN, 2 ** 70, (1, 2)],
    'background_snr': [(), NAN, '5', (None,), 1e5, -1e5, -400, 3083],
    'attribute': [('snrRatio', [NAN]), ('snrRatio', [-1e5]), ('subtype', 'FOO'),
                  ('rt60Ratio', [NAN]), ('drr_db', NAN), ('channel', {'loss': 0.9, 'burst': 1}),
                  ('cleanWaves', []), ('noiseWaves', []), ('snrRatio', [])],
}
EXTRA = ('snr_ratio', 'subtype', 'seed', 'background_snr', 'attribute')
CTOR = ('snr_ratio', 'rt60', 'drr_db', 'channel', 'write_targets', 'subtype')
CALL = ('mode', 'mkdir_parents', 'is_add_clean', 'rdn_choice_num', 'is_silence', 'p_silence',
        'overwrite', 'target_sr', 'seed')


@pytest.fixture(scope="module")
def pool(tmp_path_factory):
    """Inputs of any kind that shows what is wrong with it in its header."""
    root = tmp_path_factory.mktemp("pool")
    for name, sr, n, channels in (('g16', 16000, 2400, 1), ('g8', 8000, 1300, 1),
                                  ('g44', 44100, 6000, 2), ('g48', 48000, 5000, 1),
                                  ('hz7', 7, 20, 1), ('hz1', 1, 9, 1), ('hz600', 600, 300, 1),
                                  ('k384', 384000, 30000, 1)):
        _put(root / f"{name}.wav", _tone(n, sr, channels), sr)
    for n in (0, 1, 2, 3, 40):
        _put(root / f"n{n}.wav", _tone(n))
    _put(root / "float.wav", _tone(2500), subtype='FLOAT')
    _put(root / "gsm.wav", _tone(2400), 8000, subtype='GSM610')
    _put(root / "flac.wav", _tone(4000), format='FLAC')
    _endless_flac(root / "endless.wav")
    _put(root / "whole.wav", _tone(2400))
    (root / "cut.wav").write_bytes((root / "whole.wav").read_bytes()[:2000])
    (root / "garbage.wav").write_bytes(bytes(range(256)) * 8)
    (root / "zero.wav").write_bytes(b'')
    for letter in "ab":
        shutil.copy(root / "g16.wav", root / f"{letter * 130}.wav")
    return root


def _case(k, pool):
    """What call *k* is made with, drawn from its number alone."""
    rng = np.random.default_rng([k, 43])

    def one_of(options):
        return options[int(rng.integers(len(options)))]

    extra = rng.random() < 0.25
    case = {name: one_of(options) for name, options in LEGAL.items() if rng.random() < 0.55}
    names = [n for n in HARD if not extra or n in EXTRA]
    for name in rng.choice(names, size=int(rng.choice([0, 1, 2], p=[0.35, 0.5, 0.15])),
                           replace=False):
        case[str(name)] = one_of(HARD[str(name)])
    good, odd = ['g16', 'g8', 'g44', 'g48', 'float'], sorted(p.stem for p in pool.iterdir())
    inputs = []
    for _ in range(3):
        stems = [one_of(good) for _ in range(int(rng.integers(1, 4)))]
        if rng.random() < 0.2:
            stems[0] = one_of(odd)
        inputs.append([pool / f"{stem}.wav" for stem in stems])
    return extra, case, inputs


@pytest.mark.parametrize("block", range(8))
def test_no_call_that_raises_touches_the_folder(pool, tmp_path, block):
    pytest.importorskip("librosa")
    raised = ran = 0
    for k in range(block, CASES, 8):
        extra, case, (clean, noise, background) = _case(k, pool)
        root = tmp_path / str(k)
        folder = Folder(root)
        np.random.seed(k)
        syx, kept = None, None
        try:
            with warnings.catch_warnings(), contextlib.redirect_stderr(io.StringIO()):
                warnings.simplefilter("ignore")
                syx = Synthesizer(clean, noise, folder.out,
                                  **{n: case[n] for n in CTOR if n in case})
                if 'attribute' in case:
                    setattr(syx, *case['attribute'])
                kept = syx.seed, syx.target_sr, syx.rooms, syx.channels, syx.synWavesList
                if extra:
                    syx.syn_extra_mode(background, is_pool=False, **{
                        n: case[n] for n in ('background_snr', 'seed') if n in case})
                else:
                    syx.syn(**{n: case[n] for n in CALL if n in case})
                ran += 1
        except Exception as e:                              # noqa: BLE001
            raised += 1
            assert folder.now() == folder.before, (k, case, repr(e))
            if kept is not None:
                assert kept == (syx.seed, syx.target_sr, syx.rooms, syx.channels,
                                syx.synWavesList), (k, case, repr(e))
        shutil.rmtree(root)
    assert raised and ran, "both came up, or this tests nothing"


def test_a_script_without_its_main_guard_empties_nothing(folder, tmp_path):
    """Windows and macOS start a pool by running the script again, which runs
    into the pool again. Nothing comes of it but errors, and the folder is not
    emptied on the way to them."""
    if sys.platform.startswith('linux'):
        pytest.skip("a pool is forked here")
    script = tmp_path / "script.py"
    script.write_text(
        "import kudio\n"
        "syx = kudio.Synthesizer('c', 'n', 'out', snr_ratio=[0.5 * k for k in range(210)])\n"
        "syx.syn(mode='inc', seed=0, is_pool=True)\n", encoding='utf-8')
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
    process = subprocess.Popen([sys.executable, str(script)], cwd=str(tmp_path), env=env,
                               stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        for line in process.stderr:
            if b"freeze_support" in line or b"bootstrapping phase" in line:
                break
        assert folder.now() == {**folder.before, 'script.py': script.read_bytes()}
    finally:
        process.kill()
        process.wait()
        process.stderr.close()
