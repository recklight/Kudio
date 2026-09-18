# -*- coding: utf-8 -*-
import pathlib
from pathlib import Path

import numpy as np
import pytest

from kudio import Synthesizer, file_load


def test_syn_increment_mode(clean_noise_dirs, tmp_path):
    clean, noise = clean_noise_dirs
    out = tmp_path / "mixed"
    syx = Synthesizer(clean, noise, out_path=str(out), snr_ratio=(-5, 0, 5))
    result = syx.syn(mode='inc')
    # 2 clean x 1 noise x 3 SNR
    assert len(result) == 6
    files = list(out.rglob("*.wav"))
    assert len(files) == 6
    for f in files:
        y, sr = file_load(f)
        assert len(y) > 0


def test_syn_regular_mode(clean_noise_dirs, tmp_path):
    clean, noise = clean_noise_dirs
    out = tmp_path / "mixed_reg"
    syx = Synthesizer(clean, noise, out_path=str(out), snr_ratio=[0])
    result = syx.syn(mode='reg')
    assert len(result) == 2  # equals number of clean files


def test_syn_snr_is_respected(clean_noise_dirs, tmp_path):
    """Mixed at high SNR should stay closer to clean than mixed at low SNR."""
    clean, noise = clean_noise_dirs
    out_hi = tmp_path / "hi"
    out_lo = tmp_path / "lo"
    Synthesizer(clean, noise, out_path=str(out_hi), snr_ratio=[20]).syn(mode='reg')
    Synthesizer(clean, noise, out_path=str(out_lo), snr_ratio=[-5]).syn(mode='reg')

    def err_vs_clean(out_dir):
        errs = []
        for f in sorted(out_dir.rglob("*.wav")):
            stem = f.stem.split('_white')[0]
            y_mix, _ = file_load(f)
            y_cln, _ = file_load(clean / f"{stem}.wav")
            n = min(len(y_mix), len(y_cln))
            errs.append(np.mean((y_mix[:n] - y_cln[:n]) ** 2))
        return np.mean(errs)

    assert err_vs_clean(out_hi) < err_vs_clean(out_lo)


def test_syn_overwrite_false_aborts(clean_noise_dirs, tmp_path):
    clean, noise = clean_noise_dirs
    out = tmp_path / "mixed"
    syx = Synthesizer(clean, noise, out_path=str(out), snr_ratio=[0])
    assert syx.syn(mode='reg') is not None
    assert syx.syn(mode='reg', overwrite=False) is None


def test_bad_inputs(clean_noise_dirs, tmp_path):
    clean, noise = clean_noise_dirs
    with pytest.raises(TypeError):
        Synthesizer(clean, noise, snr_ratio="5")
    with pytest.raises(NotADirectoryError):
        Synthesizer(tmp_path / "nope", noise)
    syx = Synthesizer(clean, noise, out_path=str(tmp_path / "x"), snr_ratio=[0])
    with pytest.raises(ValueError):
        syx.syn(mode='bogus')


# -- sample-rate handling ------------------------------------------------------

def _corpus(tmp_path, clean_sr: int, noise_sr: int):
    """Clean and noise folders written at explicitly different rates."""
    from conftest import make_sine, write_wav

    clean = tmp_path / "c"
    noise = tmp_path / "n"
    clean.mkdir()
    noise.mkdir()
    write_wav(clean / "a.wav", make_sine(220, 1.0, clean_sr), clean_sr)
    write_wav(noise / "w.wav", make_sine(1000, 1.0, noise_sr), noise_sr)
    return clean, noise


def _rates(out_dir):
    import soundfile as sf
    return {sf.info(str(f)).samplerate for f in out_dir.rglob("*.wav")}


def test_syn_keeps_the_source_sample_rate(tmp_path):
    """Default output rate follows the clean file, not a hard-coded 16 kHz."""
    clean, noise = _corpus(tmp_path, clean_sr=8000, noise_sr=8000)
    out = tmp_path / "mixed"
    Synthesizer(clean, noise, out_path=str(out), snr_ratio=[0]).syn(mode='reg')

    assert _rates(out) == {8000}


def test_syn_target_sr_resamples(tmp_path):
    clean, noise = _corpus(tmp_path, clean_sr=8000, noise_sr=8000)
    out = tmp_path / "mixed"
    Synthesizer(clean, noise, out_path=str(out), snr_ratio=[0]).syn(
        mode='reg', target_sr=16000)

    assert _rates(out) == {16000}


def test_syn_resamples_noise_to_match_clean(tmp_path):
    """A noise file at a different rate must not be mixed in as-is."""
    clean, noise = _corpus(tmp_path, clean_sr=8000, noise_sr=44100)
    out = tmp_path / "mixed"
    result = Synthesizer(clean, noise, out_path=str(out), snr_ratio=[0]).syn(mode='reg')

    assert result
    assert _rates(out) == {8000}
    y, sr = file_load(next(out.rglob("*.wav")))
    assert sr == 8000
    assert np.isfinite(y).all()


def test_desired_sample_still_works_but_warns(tmp_path):
    clean, noise = _corpus(tmp_path, clean_sr=8000, noise_sr=8000)
    out = tmp_path / "mixed"
    syx = Synthesizer(clean, noise, out_path=str(out), snr_ratio=[0])

    with pytest.deprecated_call():
        syx.syn(mode='reg', desired_sample=16000)
    assert _rates(out) == {16000}


# ===================================================== clean x noise x room

@pytest.fixture
def dry_corpus(tmp_path):
    """One second of speech then real silence, and near-silent noise.

    A reverberant tail is only visible against silence. `clean_noise_dirs`
    carries its own noise floor, which sits above the tail of anything under a
    second and hides exactly what these tests are about.
    """
    import kudio
    sr = 16000
    clean, noise = tmp_path / "dry_clean", tmp_path / "dry_noise"
    clean.mkdir()
    noise.mkdir()

    t = np.arange(sr) / sr
    burst = sum(np.sin(2 * np.pi * 150 * k * t) / k for k in range(1, 12)) / 6
    burst = (burst * np.hanning(sr)).astype(np.float32)
    kudio.save_wave(clean / "c.wav",
                    np.concatenate([burst, np.zeros(sr, np.float32)]), sr)
    kudio.save_wave(noise / "n.wav",
                    (1e-5 * np.random.default_rng(0)
                     .standard_normal(sr * 3)).astype(np.float32), sr)
    return str(clean), str(noise), sr


def _tail_below_speech(path, sr):
    """Energy in the half-second after the speech stops, relative to it."""
    from kudio import file_load
    y, _ = file_load(path, sr=None, mono=True)
    tail = y[int(1.0 * sr):int(1.5 * sr)]
    speech = y[int(0.2 * sr):int(0.8 * sr)]
    return 20 * np.log10((np.sqrt(np.mean(tail ** 2)) + 1e-20) /
                         (np.sqrt(np.mean(speech ** 2)) + 1e-20))


def test_no_rt60_changes_nothing_at_all(clean_noise_dirs, tmp_path):
    """The room is opt-in, down to the output names and the return shape."""
    clean, noise = clean_noise_dirs
    out = tmp_path / "plain"
    syx = Synthesizer(clean, noise, out_path=str(out), snr_ratio=[0])
    result = syx.syn(mode="reg", seed=1)

    assert all(len(entry) == 4 for entry in result)
    assert syx.rooms == {}
    assert all(p.rt60 is None for p in syx.manifest())
    assert not any("rt" in f.name for f in out.rglob("*.wav"))


def test_a_longer_room_leaves_more_behind(dry_corpus, tmp_path):
    """The whole point, measured: the tail after the speech grows with rt60."""
    clean, noise, sr = dry_corpus
    tails = {}
    for rt60 in (None, 0.3, 0.9, 2.0):
        out = tmp_path / f"room_{rt60}"
        Synthesizer(clean, noise, out_path=str(out), snr_ratio=[60],
                    rt60=rt60).syn(mode="reg", seed=1)
        tails[rt60] = _tail_below_speech(sorted(out.rglob("*.wav"))[0], sr)

    assert tails[None] < tails[0.3] < tails[0.9] < tails[2.0]
    assert tails[None] < -60, "no room means no tail"
    assert tails[2.0] > -30, "a two-second room is unmistakable"


def test_the_room_goes_in_the_name_and_the_manifest(dry_corpus, tmp_path):
    clean, noise, _ = dry_corpus
    out = tmp_path / "named"
    syx = Synthesizer(clean, noise, out_path=str(out), snr_ratio=[0],
                      rt60=[0.3, 0.9])
    result = syx.syn(mode="reg", seed=1)

    names = sorted(f.name for f in out.rglob("*.wav"))
    assert any("rt300ms" in n for n in names)
    assert any("rt900ms" in n for n in names)

    pairs = syx.manifest()
    assert len(pairs) == len(result)
    assert sorted(p.rt60 for p in pairs) == [0.3, 0.9]
    for pair in pairs:
        assert f"rt{int(pair.rt60 * 1000)}ms" in pair.noisy


def test_the_return_shape_does_not_depend_on_a_keyword(dry_corpus, tmp_path):
    """A four-tuple with no room and a five-tuple with one is the bug this
    codebase has met most often; the room travels in `manifest()` instead."""
    clean, noise, _ = dry_corpus
    syx = Synthesizer(clean, noise, out_path=str(tmp_path / "shape"),
                      snr_ratio=[0], rt60=[0.5])
    for entry in syx.syn(mode="reg", seed=1):
        assert len(entry) == 4
        out, cln, nos, snr = entry              # the documented unpack
        assert Path(out).is_file()


def test_the_room_is_a_fourth_axis_in_inc_mode(clean_noise_dirs, tmp_path):
    clean, noise = clean_noise_dirs             # 2 clean x 1 noise
    plain = Synthesizer(clean, noise, out_path=str(tmp_path / "a"),
                        snr_ratio=[-5, 0]).syn(mode="inc", seed=2)
    three = Synthesizer(clean, noise, out_path=str(tmp_path / "b"),
                        snr_ratio=[-5, 0],
                        rt60=[0.3, 0.6, 0.9]).syn(mode="inc", seed=2)
    assert len(plain) == 4
    assert len(three) == 12


def test_the_same_request_rebuilds_the_same_dataset(dry_corpus, tmp_path):
    """Rooms are seeded per output file, so a process pool cannot change the
    answer the way the globally-seeded noise cropping can."""
    from kudio import file_load
    clean, noise, _ = dry_corpus
    made = []
    for run in ("first", "second"):
        out = tmp_path / run
        Synthesizer(clean, noise, out_path=str(out), snr_ratio=[10],
                    rt60=[0.4, 0.8]).syn(mode="reg", seed=7)
        made.append([file_load(f, sr=None, mono=True)[0]
                     for f in sorted(out.rglob("*.wav"))])

    assert len(made[0]) == len(made[1]) > 0
    for a, b in zip(*made):
        assert np.array_equal(a, b)


def test_two_files_at_the_same_rt60_get_different_rooms(dry_corpus, tmp_path):
    """One room reused across a corpus would teach a model that room."""
    clean, noise, _ = dry_corpus
    out = tmp_path / "varied"
    syx = Synthesizer(clean, noise, out_path=str(out), snr_ratio=[0, 5],
                      rt60=[0.5])
    syx.syn(mode="inc", seed=3)

    seeds = {room[2] for room in syx.rooms.values()}
    assert len(syx.rooms) >= 2
    assert len(seeds) == len(syx.rooms)


def test_the_mixture_stays_the_same_length_as_its_target(dry_corpus, tmp_path):
    """The clean file is the training target; a mixture that grew by the
    length of the tail would no longer line up with it."""
    import kudio
    clean, noise, _ = dry_corpus
    out = tmp_path / "aligned"
    Synthesizer(clean, noise, out_path=str(out), snr_ratio=[20],
                rt60=[1.0]).syn(mode="reg", seed=1)

    source = sorted(Path(clean).rglob("*.wav"))[0]
    mixed = sorted(out.rglob("*.wav"))[0]
    assert kudio.audio_info(mixed).frames == kudio.audio_info(source).frames


def test_the_snr_is_measured_against_what_the_microphone_hears(dry_corpus,
                                                               tmp_path):
    """The room acts on the talker and the noise arrives at the microphone,
    so the SNR is against the reverberant speech. Measured against the dry
    signal instead, a long room would quietly raise every SNR."""
    from kudio import file_load
    clean, noise, sr = dry_corpus
    levels = {}
    for rt60 in (None, 1.5):
        out = tmp_path / f"snr_{rt60}"
        Synthesizer(clean, noise, out_path=str(out), snr_ratio=[0],
                    rt60=rt60).syn(mode="reg", seed=1)
        y, _ = file_load(sorted(out.rglob("*.wav"))[0], sr=None, mono=True)
        levels[rt60] = float(np.sqrt(np.mean(y ** 2)))

    # the mixture's overall level should not jump just because a room was added
    assert levels[1.5] == pytest.approx(levels[None], rel=0.4), levels


def test_an_impossible_room_is_refused_at_construction(clean_noise_dirs,
                                                       tmp_path):
    clean, noise = clean_noise_dirs
    with pytest.raises(ValueError, match="rt60"):
        Synthesizer(clean, noise, out_path=str(tmp_path / "x"), rt60=0.0)
    with pytest.raises(ValueError, match="rt60"):
        Synthesizer(clean, noise, out_path=str(tmp_path / "x"), rt60=[0.5, -1])
    with pytest.raises(TypeError, match="rt60"):
        Synthesizer(clean, noise, out_path=str(tmp_path / "x"), rt60="0.5")


def test_a_manifest_with_a_room_round_trips(dry_corpus, tmp_path):
    import kudio
    clean, noise, _ = dry_corpus
    syx = Synthesizer(clean, noise, out_path=str(tmp_path / "m"),
                      snr_ratio=[0], rt60=[0.6])
    syx.syn(mode="reg", seed=1)

    path = kudio.save_manifest(tmp_path / "manifest.json", syx.manifest())
    assert kudio.load_manifest(path) == syx.manifest()
    assert all(p.rt60 == 0.6 for p in kudio.load_manifest(path))


# ================================================ the link, and what it is not

def test_no_channel_changes_nothing_at_all(clean_noise_dirs, tmp_path):
    """Opt-in, the same way the room is: no link, no tag, no extra files."""
    clean, noise = clean_noise_dirs
    syx = Synthesizer(clean, noise, out_path=str(tmp_path / "plain"),
                      snr_ratio=[0])
    syx.syn(mode="reg", seed=1)
    assert syx.channel == {} and syx.channel_tag == ""
    assert syx.channels == {}
    assert all(p.channel is None for p in syx.manifest())


def test_the_link_is_measurable_on_the_output(dry_corpus, tmp_path):
    """A telephone line band-limits, and `audio_report` says so afterwards.
    That is the property worth asserting: the file really did come off a
    narrowband link, not merely that a keyword was accepted.
    """
    import kudio
    clean, noise, sr = dry_corpus
    out = tmp_path / "phone"
    Synthesizer(clean, noise, out_path=str(out), snr_ratio=[20],
                channel="telephone").syn(mode="reg", seed=1)
    plain = tmp_path / "wide"
    Synthesizer(clean, noise, out_path=str(plain), snr_ratio=[20]).syn(
        mode="reg", seed=1)

    narrow, _ = kudio.file_load(sorted(out.rglob("*.wav"))[0], sr=None)
    wide, _ = kudio.file_load(sorted(plain.rglob("*.wav"))[0], sr=None)
    assert kudio.audio_report(narrow, sr).bandwidth_hz < 4000
    assert kudio.audio_report(wide, sr).bandwidth_hz > 5000
    assert len(narrow) == len(wide)


def test_the_link_is_not_an_axis_and_stays_out_of_the_names(clean_noise_dirs,
                                                            tmp_path):
    """A corpus is recorded over a phone line or it is not. Four axes already
    multiply; making the link a fifth would square the directory for a
    setting that is constant across the run, and it is in the manifest.
    """
    clean, noise = clean_noise_dirs
    out = tmp_path / "voip"
    syx = Synthesizer(clean, noise, out_path=str(out), snr_ratio=[-5, 0, 5],
                      channel="voip")
    result = syx.syn(mode="inc", seed=1)
    assert len(result) == 6                       # 2 clean x 1 noise x 3 SNR
    assert not any("voip" in Path(p).name or "loss" in Path(p).name
                   for p, *_ in result)
    assert all(p.channel == "mu_law+loss5%x4-hold" for p in syx.manifest())


def test_the_link_lands_on_the_mixture_rather_than_the_speech(dry_corpus,
                                                              tmp_path):
    """The wire carries whatever reached the microphone, noise and all. If
    the codec ran on the speech alone the noise would come through clean and
    the file would not be a recording of anything.
    """
    import kudio
    clean, noise, sr = dry_corpus
    out = tmp_path / "banded"
    Synthesizer(clean, noise, out_path=str(out), snr_ratio=[0],
                channel="telephone").syn(mode="reg", seed=1)
    y, _ = kudio.file_load(sorted(out.rglob("*.wav"))[0], sr=None)
    # nothing above the passband survives -- including the noise, which is
    # broadband and would still be there had it been added afterwards
    spectrum = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    freqs = np.fft.rfftfreq(len(y), 1 / sr)
    above = spectrum[freqs > 4500].max()
    inside = spectrum[(freqs > 300) & (freqs < 3000)].max()
    assert 20 * np.log10(above / inside + 1e-20) < -40


def test_each_file_loses_its_own_packets(clean_noise_dirs, tmp_path):
    """One loss pattern reused across a corpus would teach a model where the
    holes are, which is the same mistake as reusing one room."""
    clean, noise = clean_noise_dirs
    syx = Synthesizer(clean, noise, out_path=str(tmp_path / "lossy"),
                      snr_ratio=[0], channel={"preset": "voip", "loss": 0.2})
    syx.syn(mode="reg", seed=3)
    seeds = {seed for _, seed in syx.channels.values()}
    assert len(syx.channels) >= 2
    assert len(seeds) == len(syx.channels)


def test_a_rerun_rebuilds_the_same_link(clean_noise_dirs, tmp_path):
    clean, noise = clean_noise_dirs
    seeds = []
    for name in ("a", "b"):
        syx = Synthesizer(clean, noise, out_path=str(tmp_path / name),
                          snr_ratio=[0], channel="voip")
        syx.syn(mode="reg", seed=9)
        seeds.append(sorted(s for _, s in syx.channels.values()))
    assert seeds[0] == seeds[1]


def test_an_impossible_link_is_refused_before_anything_is_written(
        clean_noise_dirs, tmp_path):
    from kudio.exceptions import FeatureError
    clean, noise = clean_noise_dirs
    with pytest.raises(FeatureError, match="unknown channel"):
        Synthesizer(clean, noise, out_path=str(tmp_path / "no"),
                    snr_ratio=[0], channel="opus")
    assert not (tmp_path / "no").exists()


# ===================================================== the reverberant target

def test_the_target_is_the_room_and_nothing_else(dry_corpus, tmp_path):
    """The clean file is the dry target and stays it. This is the same
    speech with the room left on, so that a model can be asked to remove the
    noise and leave the room -- a different and easier problem.
    """
    import kudio
    clean, noise, sr = dry_corpus
    out = tmp_path / "targets"
    syx = Synthesizer(clean, noise, out_path=str(out), snr_ratio=[0],
                      rt60=[0.6], write_targets=True)
    syx.syn(mode="reg", seed=4)

    pairs = syx.manifest()
    assert all(p.target is not None for p in pairs)
    target = Path(pairs[0].target)
    assert target.is_file() and target.parent.name == "TARGETS"

    dry, _ = kudio.file_load(pairs[0].clean, sr=None)
    wet, _ = kudio.file_load(target, sr=None)
    mix, _ = kudio.file_load(pairs[0].noisy, sr=None)
    assert len(wet) == len(mix)
    # it has the room on it...
    assert _tail_below_speech(target, sr) > _tail_below_speech(
        Path(pairs[0].clean), sr) + 20
    # ...and it is quieter, per sample, than the mixture it came out of
    assert np.mean(wet ** 2) < np.mean(mix ** 2)
    assert len(dry) > 0


def test_the_target_matches_the_speech_inside_the_mixture(dry_corpus,
                                                          tmp_path):
    """The one thing that makes it a target: mixture minus target is the
    noise, so a model trained on the pair is being asked for something that
    is actually in the file.
    """
    import kudio
    clean, noise, sr = dry_corpus
    out = tmp_path / "exact"
    syx = Synthesizer(clean, noise, out_path=str(out), snr_ratio=[20],
                      rt60=[0.5], write_targets=True)
    syx.syn(mode="reg", seed=6)
    pair = syx.manifest()[0]
    wet, _ = kudio.file_load(pair.target, sr=None)
    mix, _ = kudio.file_load(pair.noisy, sr=None)
    # 20 dB SNR: the target should account for almost all of the mixture
    assert kudio.si_sdr(wet, mix) > 15.0


def test_targets_without_a_room_would_be_a_second_name_for_the_clean_file(
        clean_noise_dirs, tmp_path):
    clean, noise = clean_noise_dirs
    with pytest.warns(UserWarning, match="writes nothing"):
        syx = Synthesizer(clean, noise, out_path=str(tmp_path / "pointless"),
                          snr_ratio=[0], write_targets=True)
    syx.syn(mode="reg", seed=1)
    assert syx.targets == {}
    assert all(p.target is None for p in syx.manifest())


def test_a_target_per_room_rather_than_per_clean_file(dry_corpus, tmp_path):
    """Two rooms over one talker are two different targets, and the names
    have to line up with the mixtures or the pairing is guesswork."""
    clean, noise, _ = dry_corpus
    out = tmp_path / "two_rooms"
    syx = Synthesizer(clean, noise, out_path=str(out), snr_ratio=[0],
                      rt60=[0.3, 0.9], write_targets=True)
    syx.syn(mode="inc", seed=5)
    pairs = syx.manifest()
    assert len(pairs) == 2
    assert len({p.target for p in pairs}) == 2
    for pair in pairs:
        assert Path(pair.target).name == Path(pair.noisy).name


def test_the_link_does_not_reach_the_target(dry_corpus, tmp_path):
    """The target is the speech as the room left it. Putting the codec on it
    too would ask a model to reproduce the codec, which is not denoising.

    Said by running the same request twice: the mixtures differ because one
    went down a phone line, and the targets are byte for byte the same file.
    """
    import kudio
    clean, noise, _ = dry_corpus

    def build(link, name):
        syx = Synthesizer(clean, noise, out_path=str(tmp_path / name),
                          snr_ratio=[10], rt60=[0.4], channel=link,
                          write_targets=True)
        syx.syn(mode="reg", seed=8)
        return syx.manifest()[0]

    phoned, plain = build("telephone", "both"), build(None, "room_only")
    assert phoned.channel == "telephone" and plain.channel is None
    assert phoned.rt60 == plain.rt60 == 0.4

    assert (pathlib.Path(phoned.target).read_bytes()
            == pathlib.Path(plain.target).read_bytes())
    assert (pathlib.Path(phoned.noisy).read_bytes()
            != pathlib.Path(plain.noisy).read_bytes())
