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


# ================================================== what goes past full scale

@pytest.fixture
def loud_corpus(tmp_path):
    """A talker near full scale and a noise that will push it over at -5 dB."""
    import kudio
    sr = 16000
    clean, noise = tmp_path / "loud_clean", tmp_path / "loud_noise"
    clean.mkdir()
    noise.mkdir()
    t = np.arange(sr) / sr
    kudio.save_wave(clean / "c.wav",
                    (0.8 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), sr)
    kudio.save_wave(noise / "n.wav",
                    (0.2 * np.random.default_rng(0).standard_normal(2 * sr))
                    .astype(np.float32), sr, subtype="FLOAT")
    return clean, noise


def test_past_full_scale_is_clipped_and_rounded_never_wrapped(loud_corpus,
                                                              tmp_path):
    """`astype(int16)` turned 1.2 into -0.8 without a word. The float run is
    the mixture before it was written, sample for sample -- same seed, same
    name, same crop -- so the 16-bit file can be checked against it."""
    import soundfile as sf
    clean, noise = loud_corpus
    exact = Synthesizer(clean, noise, out_path=str(tmp_path / "f"),
                        snr_ratio=[-5], subtype="FLOAT")
    exact.syn(mode="reg", seed=1)
    with pytest.warns(UserWarning, match="past full scale"):
        pcm = Synthesizer(clean, noise, out_path=str(tmp_path / "p"),
                          snr_ratio=[-5])
        pcm.syn(mode="reg", seed=1)

    y = sf.read(next((tmp_path / "f").rglob("*.wav")))[0]
    q = sf.read(next((tmp_path / "p").rglob("*.wav")))[0]
    assert np.abs(y).max() > 1.2, "the float file keeps what is past 1.0"
    assert exact.clipped == {}
    assert list(pcm.clipped.values()) == [pytest.approx(np.abs(y).max(), rel=1e-4)]

    assert np.all(q[y > 1.0] > 0.999) and np.all(q[y < -1.0] <= -0.999)
    inside = np.abs(y) < 0.999
    step = 1 / 32768
    assert np.abs(q[inside] - y[inside]).max() <= step / 2 + 1e-7
    # rounding is unbiased; truncation towards zero would sit half a step low
    bias = np.mean((q[inside] - y[inside]) * np.sign(y[inside])) / step
    assert abs(bias) < 0.1


def test_a_subtype_that_wav_cannot_hold_is_refused(clean_noise_dirs, tmp_path):
    clean, noise = clean_noise_dirs
    with pytest.raises(ValueError, match="subtype"):
        Synthesizer(clean, noise, out_path=str(tmp_path / "x"), subtype="NOPE")


# ============================================== reg mode meets every pairing

def _numbered_corpus(tmp_path, n_clean, n_noise, seconds=0.1):
    import kudio
    sr = 16000
    clean, noise = tmp_path / "nc", tmp_path / "nn"
    clean.mkdir()
    noise.mkdir()
    rng = np.random.default_rng(0)
    for i in range(n_clean):
        kudio.save_wave(clean / f"u{i:03d}.wav", (0.1 * rng.standard_normal(
            int(sr * seconds))).astype(np.float32), sr)
    for k in range(n_noise):
        kudio.save_wave(noise / f"noise{k}.wav", (0.1 * rng.standard_normal(
            int(sr * seconds * 2))).astype(np.float32), sr)
    return clean, noise


def test_reg_mode_meets_every_noise_snr_combination(tmp_path):
    """Cycled side by side, 2 noises and 4 SNRs only ever meet in 4 of their 8
    combinations: noise0 always at the even SNRs, noise1 at the odd ones."""
    clean, noise = _numbered_corpus(tmp_path, n_clean=8, n_noise=2)
    result = Synthesizer(clean, noise, out_path=str(tmp_path / "o"),
                         snr_ratio=[0, 5, 10, 15]).syn(mode="reg", seed=0)
    combos = {(Path(n).stem, snr) for _, _, n, snr in result}
    assert len(combos) == 8


def test_reg_mode_is_unchanged_when_the_counts_share_no_factor(tmp_path):
    """Coprime counts already met every combination; the assignment stays the
    plain cycle it has always been, so those datasets come out the same."""
    clean, noise = _numbered_corpus(tmp_path, n_clean=12, n_noise=2)
    snrs = [0, 5, 10]
    result = Synthesizer(clean, noise, out_path=str(tmp_path / "o"),
                         snr_ratio=snrs).syn(mode="reg", seed=0)
    assert [Path(n).stem for _, _, n, _ in result] == \
        [f"noise{i % 2}" for i in range(12)]
    assert [snr for *_, snr in result] == [snrs[i % 3] for i in range(12)]


# ================================================== any input, mirrored right

@pytest.fixture
def two_talkers(tmp_path):
    """Two talkers reading the same sentence: same file name, two folders."""
    import kudio
    sr = 16000
    tree = tmp_path / "timit"
    t = np.arange(sr // 4) / sr
    for i, folder in enumerate(("dr1/fcjf0", "dr2/fdml0")):
        (tree / folder).mkdir(parents=True)
        for name in ("sa1", "sx133"):
            kudio.save_wave(tree / folder / f"{name}.wav",
                            (0.3 * np.sin(2 * np.pi * (200 + 50 * i) * t))
                            .astype(np.float32), sr)
    return tree


def test_a_file_list_is_mirrored_below_the_folder_it_shares(two_talkers,
                                                           clean_noise_dirs,
                                                           tmp_path):
    """The point of a list is a subset -- SI and SX without SA -- and two
    talkers' sx133 have to stay two files."""
    _, noise = clean_noise_dirs
    subset = sorted(two_talkers.rglob("sx*.wav"))
    out = tmp_path / "subset"
    result = Synthesizer(subset, noise, out_path=str(out), snr_ratio=[5]).syn(
        mode="reg", seed=0)
    written = sorted(p.relative_to(out).as_posix() for p in out.rglob("*.wav"))
    assert written == ["dr1/fcjf0/sx133_white_5dB.wav",
                       "dr2/fdml0/sx133_white_5dB.wav"]
    assert len(result) == 2


def test_a_txt_manifest_with_an_explicit_root(two_talkers, clean_noise_dirs,
                                              tmp_path):
    _, noise = clean_noise_dirs
    listing = tmp_path / "subset.txt"
    listing.write_text("\n".join(str(p) for p in
                                 sorted(two_talkers.rglob("sa1.wav"))))
    out = tmp_path / "rooted"
    Synthesizer(listing, noise, out_path=str(out), snr_ratio=[0],
                root=two_talkers).syn(mode="reg", seed=0)
    assert sorted(p.relative_to(out).as_posix() for p in out.rglob("*.wav")) \
        == ["dr1/fcjf0/sa1_white_0dB.wav", "dr2/fdml0/sa1_white_0dB.wav"]


def test_a_root_that_does_not_hold_the_files_is_refused(two_talkers,
                                                        clean_noise_dirs):
    _, noise = clean_noise_dirs
    with pytest.raises(ValueError, match="not inside root"):
        Synthesizer(sorted(two_talkers.rglob("*.wav")), noise,
                    root=two_talkers / "dr1")


def test_a_list_entry_that_is_not_there_is_reported(two_talkers,
                                                    clean_noise_dirs):
    _, noise = clean_noise_dirs
    files = sorted(two_talkers.rglob("*.wav")) + [two_talkers / "gone.wav"]
    with pytest.warns(UserWarning, match="skipped 1 input"):
        syx = Synthesizer(files, noise)
    assert len(syx.cleanWaves) == 4


def test_colliding_output_names_are_reported(two_talkers, clean_noise_dirs,
                                             tmp_path):
    """Flattened, two talkers' sx133 land on one name and one overwrites the
    other -- worth a warning rather than a quietly smaller dataset."""
    _, noise = clean_noise_dirs
    syx = Synthesizer(sorted(two_talkers.rglob("sx*.wav")), noise,
                      out_path=str(tmp_path / "flat"), snr_ratio=[0])
    with pytest.warns(UserWarning, match="mkdir_parents=False"):
        syx.syn(mode="reg", seed=0, mkdir_parents=False)


# ========================================================= seeds and the pool

def test_the_pool_does_not_change_a_single_file(tmp_path):
    """The noise crop used to come from each worker's global RNG, so a seeded
    run through the pool differed from itself and from a serial run."""
    import soundfile as sf
    clean, noise = _numbered_corpus(tmp_path, n_clean=21, n_noise=1)
    made = {}
    for tag, pool in (("pool", True), ("serial", False)):
        out = tmp_path / tag
        Synthesizer(clean, noise, out_path=str(out),
                    snr_ratio=list(range(10))).syn(mode="inc", seed=3,
                                                   is_pool=pool)
        made[tag] = {p.name: sf.read(p)[0] for p in out.rglob("*.wav")}
    assert len(made["pool"]) == 210 > 200          # past the pool threshold
    assert made["pool"].keys() == made["serial"].keys()
    for name, y in made["pool"].items():
        assert np.array_equal(y, made["serial"][name]), name


def test_a_file_does_not_depend_on_what_was_written_before_it(tmp_path):
    """What the pool changes is the order and the worker; neither can matter
    once each file draws from its own seed. Written in reverse, after the
    global RNG has been disturbed, every file comes out the same."""
    import soundfile as sf
    clean, noise = _numbered_corpus(tmp_path, n_clean=3, n_noise=1)
    syx = Synthesizer(clean, noise, out_path=str(tmp_path / "a"),
                      snr_ratio=[0, 5])
    syx.syn(mode="inc", seed=11)
    first = {p.name: sf.read(p)[0] for p in (tmp_path / "a").rglob("*.wav")}

    seeds = Synthesizer._seeds(syx.synWavesList, 11, syx.noisyDir)
    np.random.seed(12345)
    for entry in reversed(syx.synWavesList):
        Synthesizer.syn_waves(entry, False, None, seeds=seeds)
    again = {p.name: sf.read(p)[0] for p in (tmp_path / "a").rglob("*.wav")}
    assert first.keys() == again.keys()
    for name in first:
        assert np.array_equal(first[name], again[name])


def test_syn_leaves_the_callers_rng_alone(clean_noise_dirs, tmp_path):
    clean, noise = clean_noise_dirs
    np.random.seed(2024)
    expected = np.random.random(3)
    np.random.seed(2024)
    Synthesizer(clean, noise, out_path=str(tmp_path / "x"),
                snr_ratio=[0]).syn(mode="reg", seed=5, rdn_choice_num=1)
    assert np.array_equal(np.random.random(3), expected)


def test_a_run_without_a_seed_can_be_rebuilt(clean_noise_dirs, tmp_path):
    import soundfile as sf
    clean, noise = clean_noise_dirs
    syx = Synthesizer(clean, noise, out_path=str(tmp_path / "a"), snr_ratio=[0])
    syx.syn(mode="reg")
    assert isinstance(syx.seed, int)
    first = {p.name: sf.read(p)[0] for p in (tmp_path / "a").rglob("*.wav")}
    again = Synthesizer(clean, noise, out_path=str(tmp_path / "b"),
                        snr_ratio=[0])
    again.syn(mode="reg", seed=syx.seed)
    for p in (tmp_path / "b").rglob("*.wav"):
        assert np.array_equal(sf.read(p)[0], first[p.name])


def test_a_random_subset_has_no_repeats(tmp_path):
    clean, noise = _numbered_corpus(tmp_path, n_clean=10, n_noise=1)
    result = Synthesizer(clean, noise, out_path=str(tmp_path / "o"),
                         snr_ratio=[0]).syn(mode="reg", seed=0,
                                            rdn_choice_num=10)
    assert len({c for _, c, _, _ in result}) == 10


def test_a_mirrored_tree_gives_same_named_files_their_own_draws(two_talkers,
                                                                clean_noise_dirs,
                                                                tmp_path):
    """Seeds come from the path below the output folder, not the name: every
    TIMIT talker reads sa1, and they should not all share a room."""
    _, noise = clean_noise_dirs
    syx = Synthesizer(sorted(two_talkers.rglob("sa1.wav")), noise,
                      out_path=str(tmp_path / "o"), snr_ratio=[0], rt60=[0.4])
    syx.syn(mode="reg", seed=0)
    assert len({room[2] for room in syx.rooms.values()}) == 2


# ======================================================== the silence share

def _correlation(path, clean_file):
    from kudio import file_load
    y, _ = file_load(path)
    c, _ = file_load(clean_file)
    n = min(len(y), len(c))
    return float(np.corrcoef(y[:n], c[:n])[0, 1])


def test_is_silence_leaves_the_mixtures_their_speech(tmp_path):
    """The flag was handed to every file, so `is_silence=True` wrote a whole
    dataset of noise with not one word in it. Only the share is noise now."""
    import kudio
    sr = 16000
    clean, noise = tmp_path / "sc", tmp_path / "sn"
    clean.mkdir()
    noise.mkdir()
    t = np.arange(sr // 4) / sr
    for i in range(20):
        kudio.save_wave(clean / f"u{i:02d}.wav",
                        (0.3 * np.sin(2 * np.pi * (200 + 10 * i) * t))
                        .astype(np.float32), sr)
    kudio.save_wave(noise / "w.wav", (0.1 * np.random.default_rng(0)
                                      .standard_normal(sr)).astype(np.float32), sr)
    out = tmp_path / "o"
    result = Synthesizer(clean, noise, out_path=str(out), snr_ratio=[20]).syn(
        mode="reg", seed=0, is_silence=True, p_silence=0.1)

    silent = [(o, c) for o, c, _, _ in result if o.name.endswith("_n00.wav")]
    mixtures = [(o, c) for o, c, _, _ in result if not o.name.endswith("_n00.wav")]
    assert len(silent) == 2 and len(mixtures) == 20
    assert len(list(out.rglob("*.wav"))) == 22
    assert all(_correlation(o, c) > 0.9 for o, c in mixtures)
    assert all(abs(_correlation(o, c)) < 0.2 for o, c in silent)


def test_extra_mode_keeps_the_speech_and_the_working_directory(
        clean_noise_dirs, tmp_path, monkeypatch):
    """Stage two ran with the silence flag on for every file, so the whole of
    extra mode came out as noise; and its scratch folder sat in the working
    directory, under a name anything else could have been using."""
    clean, noise = clean_noise_dirs
    background = tmp_path / "bg"
    background.mkdir()
    from conftest import write_wav
    write_wav(background / "hum.wav", (0.05 * np.sin(
        2 * np.pi * 50 * np.arange(16000) / 16000)).astype(np.float32))
    work = tmp_path / "cwd"
    (work / "tmp_mixed").mkdir(parents=True)
    (work / "tmp_mixed" / "keep.txt").write_text("not yours")
    monkeypatch.chdir(work)

    out = tmp_path / "extra"
    result = Synthesizer(clean, noise, out_path=str(out), snr_ratio=[20]) \
        .syn_extra_mode(background, background_snr=(5,), seed=0)
    # 2 clean x (1 stage-one mixture + 1 noise + 1 background) x 1 SNR
    assert len(result) == 6
    for path, clean_file, _, _ in result:
        assert _correlation(path, clean_file) > 0.9
    assert (work / "tmp_mixed" / "keep.txt").read_text() == "not yours"


# ============================================== what overwrite may not delete

def test_overwrite_will_not_delete_the_inputs(clean_noise_dirs, tmp_path):
    """`out_path` one level too high used to take the corpus with it."""
    from kudio.exceptions import SynthesisError
    clean, noise = clean_noise_dirs
    syx = Synthesizer(clean, noise, out_path=str(tmp_path), snr_ratio=[0])
    with pytest.raises(SynthesisError, match="holds the input"):
        syx.syn(mode="reg", seed=0)
    assert any(Path(clean).iterdir()) and any(Path(noise).iterdir())


def test_overwrite_will_not_delete_the_working_directory(clean_noise_dirs,
                                                         tmp_path, monkeypatch):
    from kudio.exceptions import SynthesisError
    clean, noise = clean_noise_dirs
    work = tmp_path / "project"
    work.mkdir()
    (work / "train.py").write_text("print('mine')")
    monkeypatch.chdir(work)
    syx = Synthesizer(clean, noise, out_path=".", snr_ratio=[0])
    with pytest.raises(SynthesisError, match="working directory"):
        syx.syn(mode="reg", seed=0)
    assert (work / "train.py").exists()


def test_a_bad_mode_is_refused_before_anything_is_deleted(clean_noise_dirs,
                                                          tmp_path):
    clean, noise = clean_noise_dirs
    out = tmp_path / "keep"
    syx = Synthesizer(clean, noise, out_path=str(out), snr_ratio=[0])
    syx.syn(mode="reg", seed=0)
    before = sorted(out.rglob("*.wav"))
    with pytest.raises(ValueError):
        syx.syn(mode="bogus")
    assert sorted(out.rglob("*.wav")) == before


# ======================================================== names and values

def test_a_fractional_snr_keeps_its_own_file_and_value(clean_noise_dirs,
                                                       tmp_path):
    """`with_suffix` read '.5dB' as a suffix: 2.5 and 2.7 dB both became
    `..._2.wav` and one overwrote the other."""
    clean, noise = clean_noise_dirs
    out = tmp_path / "frac"
    syx = Synthesizer(clean, noise, out_path=str(out),
                      snr_ratio=[-5, -2.5, 2.5, 2.7, 5])
    syx.syn(mode="inc", seed=0)
    names = sorted(p.name for p in out.glob("clean_0_*.wav"))
    assert names == ["clean_0_white_2p5dB.wav", "clean_0_white_2p7dB.wav",
                     "clean_0_white_5dB.wav", "clean_0_white_n2p5.wav",
                     "clean_0_white_n5.wav"]
    assert sorted({p.snr_db for p in syx.manifest()}) == [-5, -2.5, 2.5, 2.7, 5]
    assert all(isinstance(p.snr_db, int) for p in syx.manifest()
               if p.snr_db in (-5, 5))


def test_a_dotted_file_name_keeps_its_whole_stem(tmp_path, clean_noise_dirs):
    import kudio
    _, noise = clean_noise_dirs
    clean = tmp_path / "dotted"
    clean.mkdir()
    kudio.save_wave(clean / "take.v2.wav", np.zeros(1600, np.float32) + 0.1,
                    16000)
    out = tmp_path / "o"
    Synthesizer(clean, noise, out_path=str(out), snr_ratio=[0, 5]).syn(
        mode="inc", seed=0)
    assert sorted(p.name for p in out.rglob("*.wav")) == \
        ["take.v2_white_0dB.wav", "take.v2_white_5dB.wav"]
