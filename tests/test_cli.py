import pathlib
# -*- coding: utf-8 -*-
import numpy as np
import pytest
from scipy.io import wavfile

from kudio.cli import build_parser, main


def _write(path, freq=440, sr=16000, seconds=1.0):
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    y = (0.5 * np.sin(2 * np.pi * freq * t) * np.iinfo(np.int16).max).astype(np.int16)
    wavfile.write(str(path), sr, y)


@pytest.mark.parametrize("argv", [
    ["info", "x.wav"],
    ["convert", "a.wav", "b.wav", "--rate", "8000"],
    ["synth", "--clean", "c", "--noise", "n", "--out", "o"],
    ["trim", "a.wav", "b.wav"],
    ["devices"],
])
def test_parser_accepts_subcommands(argv):
    args = build_parser().parse_args(argv)
    assert args.command == argv[0]


def test_info_command(tmp_path, capsys):
    f = tmp_path / "t.wav"
    _write(f)
    assert main(["info", str(f)]) == 0
    out = capsys.readouterr().out
    assert "samplerate: 16000" in out
    assert "duration" in out


def test_convert_command(tmp_path):
    src = tmp_path / "s.wav"
    dst = tmp_path / "d.wav"
    _write(src)
    assert main(["convert", str(src), str(dst), "--rate", "8000"]) == 0
    import soundfile as sf
    info = sf.info(str(dst))
    assert info.samplerate == 8000


def test_trim_command(tmp_path):
    src = tmp_path / "s.wav"
    dst = tmp_path / "d.wav"
    sr = 16000
    t = np.linspace(0, 0.5, sr // 2, endpoint=False)
    tone = 0.5 * np.sin(2 * np.pi * 440 * t)
    y = np.concatenate([np.zeros(sr // 2), tone, np.zeros(sr // 2)])
    wavfile.write(str(src), sr, (y * np.iinfo(np.int16).max).astype(np.int16))
    assert main(["trim", str(src), str(dst), "--top-db", "20"]) == 0
    assert dst.is_file()


def _write_speech(path, sr=16000, seconds=6.0, seed=0):
    from conftest import make_speech
    import kudio
    kudio.save_wave(str(path), make_speech(sr=sr, seconds=seconds, seed=seed), sr)


@pytest.mark.parametrize("argv", [
    ["report", "x.wav"],
    ["loudness", "x.wav"],
    ["normalize", "a.wav", "b.wav", "--lufs", "-23"],
    ["vad", "x.wav"],
    ["pitch", "x.wav"],
    ["align", "a.wav", "b.wav"],
    ["room", "ir.wav"],
    ["channel", "a.wav", "b.wav"],
    ["convert", "a", "b", "--recursive"],
])
def test_parser_accepts_the_new_subcommands(argv):
    assert build_parser().parse_args(argv).command == argv[0]


def test_normalize_targets_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["normalize", "a.wav", "b.wav", "--lufs", "-23", "--peak", "0.9"])


def test_report_command_prints_findings(tmp_path, capsys):
    f = tmp_path / "t.wav"
    _write(f)
    code = main(["report", str(f)])
    out = capsys.readouterr().out
    assert "dBFS" in out
    assert "perceptual score" in out          # the honesty note is not optional
    assert code in (0, 1)


def test_report_exits_nonzero_when_something_is_wrong(tmp_path, capsys):
    """A pure tone has no content above its own frequency: band-limited."""
    f = tmp_path / "tone.wav"
    _write(f)
    assert main(["report", str(f)]) == 1
    assert "·" in capsys.readouterr().out


def test_loudness_command(tmp_path, capsys):
    f = tmp_path / "t.wav"
    _write(f)
    assert main(["loudness", str(f)]) == 0
    assert "LUFS" in capsys.readouterr().out


def test_normalize_command_hits_the_target(tmp_path):
    import kudio
    src, dst = tmp_path / "s.wav", tmp_path / "d.wav"
    _write(src)
    assert main(["normalize", str(src), str(dst), "--lufs", "-20"]) == 0
    y, sr = kudio.file_load(dst, sr=None)
    assert kudio.loudness(y, sr) == pytest.approx(-20.0, abs=0.5)


def test_normalize_command_peak_mode(tmp_path):
    import kudio
    src, dst = tmp_path / "s.wav", tmp_path / "d.wav"
    _write(src)
    assert main(["normalize", str(src), str(dst), "--peak", "0.5"]) == 0
    assert kudio.audio_info(dst).peak == pytest.approx(0.5, abs=0.01)


def test_vad_command_lists_spans(tmp_path, capsys):
    f = tmp_path / "speech.wav"
    _write_speech(f)
    assert main(["vad", str(f)]) == 0
    out = capsys.readouterr().out
    assert "span(s)" in out
    assert "→" in out


def test_vad_command_exits_nonzero_on_silence(tmp_path, capsys):
    import kudio
    import numpy as np
    f = tmp_path / "quiet.wav"
    kudio.save_wave(str(f), np.zeros(16000, dtype=np.float32), 16000)
    assert main(["vad", str(f)]) == 1
    assert "no speech" in capsys.readouterr().out


def test_vad_command_can_split_to_files(tmp_path):
    f = tmp_path / "speech.wav"
    _write_speech(f)
    out_dir = tmp_path / "spans"
    assert main(["vad", str(f), "--split-to", str(out_dir)]) == 0
    assert list(out_dir.glob("*.wav"))


def test_pitch_command_summarises_the_contour(tmp_path, capsys):
    f = tmp_path / "speech.wav"
    _write_speech(f)
    assert main(["pitch", str(f), "--segments"]) == 0
    out = capsys.readouterr().out
    assert "voiced" in out and "semitones" in out
    assert "→" in out                       # the span listing


def test_pitch_command_exits_nonzero_when_nothing_is_voiced(tmp_path, capsys):
    import numpy as np

    import kudio
    f = tmp_path / "quiet.wav"
    kudio.save_wave(str(f), np.zeros(16000, dtype=np.float32), 16000)
    assert main(["pitch", str(f)]) == 1
    assert "no voiced frames" in capsys.readouterr().out


def test_pitch_command_writes_a_label_track(tmp_path):
    import kudio
    f = tmp_path / "speech.wav"
    _write_speech(f)
    labels = tmp_path / "voiced.txt"
    assert main(["pitch", str(f), "--labels", str(labels)]) == 0
    assert len(kudio.load_labels(labels)) >= 1


def test_convert_recursive_walks_a_folder(tmp_path, capsys):
    import kudio
    src, dst = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    for i in range(3):
        _write(src / f"t{i}.wav")
    assert main(["convert", str(src), str(dst), "--rate", "8000",
                 "--recursive"]) == 0
    assert "3/3" in capsys.readouterr().out
    assert all(kudio.audio_info(p, peak=False).sr == 8000
               for p in dst.glob("*.wav"))


def test_info_command_reports_the_encoding(tmp_path, capsys):
    f = tmp_path / "t.wav"
    _write(f)
    assert main(["info", str(f)]) == 0
    assert "encoding  : WAV / PCM_16" in capsys.readouterr().out


def test_enhance_command(tmp_path, capsys):
    import kudio
    src, dst = tmp_path / "s.wav", tmp_path / "d.wav"
    _write_speech(src)
    assert main(["enhance", str(src), str(dst), "--method", "logmmse"]) == 0
    out = capsys.readouterr().out
    assert "log-MMSE" in out and "noise floor" in out
    y, sr = kudio.file_load(dst)
    assert len(y) > 0


def test_enhance_takes_method_parameters(tmp_path, capsys):
    src, dst = tmp_path / "s.wav", tmp_path / "d.wav"
    _write_speech(src)
    assert main(["enhance", str(src), str(dst), "--method", "omlsa",
                 "--set", "q=0.5"]) == 0
    assert dst.is_file()


def test_enhance_rejects_a_malformed_set(tmp_path):
    src, dst = tmp_path / "s.wav", tmp_path / "d.wav"
    _write_speech(src)
    assert main(["enhance", str(src), str(dst), "--set", "q"]) == 2


def test_compare_command_ranks_every_method(tmp_path, capsys):
    src = tmp_path / "s.wav"
    _write_speech(src)
    assert main(["compare", str(src), "--no-metrics"]) == 0
    out = capsys.readouterr().out
    assert "method" in out and "floor dBFS" in out
    for name in ("logmmse", "wiener", "omlsa"):
        assert name in out
    # with no reference these are not quality scores, and it has to say so
    assert "not quality scores" in out


def test_compare_with_a_reference_shows_the_quality_columns(tmp_path, capsys):
    import kudio
    import numpy as np
    from conftest import make_speech

    sr = 16000
    clean = make_speech(sr=sr)
    noisy = clean + (0.05 * np.random.default_rng(1).standard_normal(len(clean))
                     ).astype(np.float32)
    kudio.save_wave(str(tmp_path / "clean.wav"), clean, sr)
    kudio.save_wave(str(tmp_path / "noisy.wav"), noisy, sr)

    assert main(["compare", str(tmp_path / "noisy.wav"),
                 "--reference", str(tmp_path / "clean.wav"), "--no-metrics"]) == 0
    out = capsys.readouterr().out
    assert "SI-SDR" in out and "dSNR" in out


def test_compare_can_write_every_result(tmp_path):
    src = tmp_path / "s.wav"
    _write_speech(src)
    out_dir = tmp_path / "results"
    assert main(["compare", str(src), "--no-metrics", "--methods", "wiener",
                 "logmmse", "--write-to", str(out_dir)]) == 0
    assert sorted(p.stem for p in out_dir.glob("*.wav")) == ["s_logmmse", "s_wiener"]


def _write_varying(path, sr=16000, seconds=12.0):
    """A take whose level moves, so range and the extremes mean something."""
    import numpy as np

    import kudio
    t = np.arange(int(sr * seconds), dtype=np.float64) / sr
    gate = np.where((t % 4.0) < 2.0, 1.0, 10 ** (-14 / 20))
    y = 0.4 * np.sin(2 * np.pi * 220 * t) * gate
    kudio.save_wave(str(path), y.astype("float32"), sr)


def test_loudness_command_reports_range_and_true_peak(tmp_path, capsys):
    f = tmp_path / "take.wav"
    _write_varying(f)
    assert main(["loudness", str(f)]) == 0

    out = capsys.readouterr().out
    assert "integrated" in out and "LUFS" in out
    assert "range" in out and "LU" in out
    assert "true peak" in out and "dBTP" in out
    assert "short-term" in out and "loudest" in out and "quietest" in out


def test_loudness_command_can_draw_the_curve(tmp_path, capsys):
    f = tmp_path / "take.wav"
    _write_varying(f)
    assert main(["loudness", str(f), "--momentary", "--over-time"]) == 0

    out = capsys.readouterr().out
    assert "momentary" in out
    assert out.count("#") > 50          # one row per 100 ms of a 12 s take


def test_loudness_command_checks_a_delivery_target(tmp_path, capsys):
    f = tmp_path / "take.wav"
    _write_varying(f)

    assert main(["loudness", str(f), "--target", "-23"]) == 1
    assert "over the -23 LUFS target" in capsys.readouterr().out

    import kudio
    y, sr = kudio.file_load(str(f), sr=None, mono=True)
    on_target = tmp_path / "on_target.wav"
    kudio.save_wave(str(on_target), kudio.normalize_lufs(y, sr, -23.0), sr)
    assert main(["loudness", str(on_target), "--target", "-23"]) == 0


def test_loudness_command_survives_a_clip_shorter_than_the_window(tmp_path, capsys):
    """The integrated figure and the true peak still work; only the curve does
    not, and it says which window it could not fill."""
    f = tmp_path / "short.wav"
    _write(f)                                     # the 1 s fixture
    assert main(["loudness", str(f)]) == 0

    out = capsys.readouterr().out
    assert "shorter than the 3 s window" in out
    assert "true peak" in out


def _write_pair(tmp_path, shift=128):
    """A reference and a delayed, denoised copy of it."""
    import numpy as np

    import kudio
    from conftest import make_speech
    clean = make_speech()
    rng = np.random.default_rng(0)
    noisy = kudio.add_noise_snr(
        clean, rng.standard_normal(len(clean)).astype("float32"),
        snr_db=5, seed=0)
    deg = np.concatenate([np.zeros(shift, dtype="float32"),
                          kudio.spectral_enhance(noisy, 16000, method="logmmse")])
    ref_path, deg_path = tmp_path / "ref.wav", tmp_path / "deg.wav"
    kudio.save_wave(str(ref_path), clean, 16000)
    kudio.save_wave(str(deg_path), deg, 16000)
    return ref_path, deg_path


def test_align_command_reports_the_delay(tmp_path, capsys):
    ref, deg = _write_pair(tmp_path, shift=128)
    assert main(["align", str(ref), str(deg)]) == 0

    out = capsys.readouterr().out
    assert "lags by 128 samples" in out
    assert "correlation" in out
    assert "overlap" in out


def test_align_command_shows_what_the_misalignment_cost(tmp_path, capsys):
    """The point of the whole thing: the same two files score 20-odd dB apart
    depending only on whether anyone lined them up."""
    ref, deg = _write_pair(tmp_path, shift=128)
    assert main(["align", str(ref), str(deg), "--metrics"]) == 0

    import re
    out = capsys.readouterr().out
    values = [float(m) for m in re.findall(r"(-?\d+\.\d+) dB", out)]

    assert len(values) == 2, out
    assert "as the files sit" in out and "aligned" in out
    assert values[1] > values[0] + 10.0, out


def test_align_command_writes_the_aligned_pair(tmp_path):
    import kudio
    ref, deg = _write_pair(tmp_path, shift=128)
    out_dir = tmp_path / "aligned"
    assert main(["align", str(ref), str(deg), "--write-to", str(out_dir)]) == 0

    written = sorted(p.name for p in out_dir.glob("*.wav"))
    assert written == ["deg_aligned.wav", "ref_aligned.wav"]
    lengths = {kudio.audio_info(p).frames for p in out_dir.glob("*.wav")}
    assert len(lengths) == 1, "the pair has to come back the same length"


def test_align_command_refuses_two_unrelated_files(tmp_path, capsys):
    import numpy as np

    import kudio
    from conftest import make_speech
    a, b = tmp_path / "a.wav", tmp_path / "b.wav"
    kudio.save_wave(str(a), make_speech(), 16000)
    kudio.save_wave(str(b), (np.random.default_rng(3).standard_normal(96000)
                             * 0.1).astype("float32"), 16000)

    assert main(["align", str(a), str(b)]) == 1
    assert "do not look like the same recording" in capsys.readouterr().out


def test_align_command_resamples_before_measuring(tmp_path, capsys):
    """A sample offset is not a quantity yet if the two are on different
    clocks."""
    import kudio
    from conftest import make_speech
    clean = make_speech()
    a, b = tmp_path / "a.wav", tmp_path / "b.wav"
    kudio.save_wave(str(a), clean, 16000)
    kudio.save_wave(str(b), kudio.resample(clean, 16000, 8000), 8000)

    assert main(["align", str(a), str(b)]) == 0
    out = capsys.readouterr().out
    assert "resampling" in out and "8000 to 16000" in out


def test_room_command_synthesises_and_measures_the_same_room(tmp_path, capsys):
    """The knob and the measurement have to agree, or one of them is wrong."""
    import kudio
    out = tmp_path / "room.wav"
    assert main(["room", "--make", str(out), "--rt60", "0.7",
                 "--drr", "6", "--seed", "0"]) == 0
    assert out.exists()

    capsys.readouterr()
    assert main(["room", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "T30" in printed and "EDT" in printed and "DRR" in printed

    measured = kudio.rt60(*reversed(kudio.file_load(str(out), sr=None,
                                                    mono=True)[::-1]))
    assert measured.t30 == pytest.approx(0.7, rel=0.06)
    assert measured.drr == pytest.approx(6.0, abs=0.2)


def test_room_command_puts_audio_through_a_room(tmp_path, capsys):
    import kudio
    speech = tmp_path / "speech.wav"
    _write_speech(speech)
    wet = tmp_path / "wet.wav"

    assert main(["room", str(speech), "--apply", str(wet),
                 "--rt60", "0.8", "--seed", "1"]) == 0
    assert "0.8s room" in capsys.readouterr().out

    dry_len = kudio.audio_info(speech).frames
    assert kudio.audio_info(wet).frames == dry_len, "length is kept"

    dry, sr = kudio.file_load(str(speech), sr=None, mono=True)
    reverberant, _ = kudio.file_load(str(wet), sr=None, mono=True)
    assert not np.allclose(dry, reverberant, atol=1e-3)


def test_room_command_refuses_a_file_that_is_not_an_impulse_response(tmp_path,
                                                                    capsys):
    speech = tmp_path / "speech.wav"
    _write_speech(speech)
    assert main(["room", str(speech)]) == 1
    assert "may not be an impulse response" in capsys.readouterr().out


def test_room_command_needs_something_to_do(capsys):
    assert main(["room"]) == 1
    assert "--make" in capsys.readouterr().out


def test_channel_command_puts_a_file_through_a_phone_line(tmp_path, capsys):
    import kudio
    src, dst = tmp_path / "s.wav", tmp_path / "phone.wav"
    _write_speech(src)
    assert main(["channel", str(src), str(dst), "--telephone"]) == 0

    out = capsys.readouterr().out
    assert "telephone (mu_law)" in out
    assert "bandwidth" in out and "si_sdr" in out

    y, sr = kudio.file_load(dst, sr=None, mono=True)
    report = kudio.audio_report(y, sr)
    assert report.band_limited
    assert 3000 < report.bandwidth_hz < 4200
    assert kudio.audio_info(dst).frames == kudio.audio_info(src).frames


def test_channel_command_drops_packets_reproducibly(tmp_path, capsys):
    import kudio
    src = tmp_path / "s.wav"
    _write_speech(src)
    for name in ("a.wav", "b.wav"):
        assert main(["channel", str(src), str(tmp_path / name),
                     "--loss", "0.1", "--seed", "3"]) == 0
        capsys.readouterr()

    first, sr = kudio.file_load(tmp_path / "a.wav", sr=None, mono=True)
    second, _ = kudio.file_load(tmp_path / "b.wav", sr=None, mono=True)
    assert np.array_equal(first, second)
    assert np.count_nonzero(first == 0.0) > 0


def test_channel_command_stacks_what_it_is_asked_for(tmp_path, capsys):
    src, dst = tmp_path / "s.wav", tmp_path / "d.wav"
    _write_speech(src)
    assert main(["channel", str(src), str(dst), "--codec", "a_law",
                 "--bits", "8", "--loss", "0.02", "--seed", "0"]) == 0

    out = capsys.readouterr().out
    assert "a_law" in out and "8-bit" in out and "packet(s) lost" in out


def test_channel_command_needs_something_to_do(tmp_path, capsys):
    src, dst = tmp_path / "s.wav", tmp_path / "d.wav"
    _write_speech(src)
    assert main(["channel", str(src), str(dst)]) == 1
    assert "nothing asked for" in capsys.readouterr().out


def test_channel_command_loses_packets_in_runs_when_asked(tmp_path, capsys):
    import kudio
    src = tmp_path / "s.wav"
    _write_speech(src, seconds=20.0)

    def longest_hole(path):
        y, _ = kudio.file_load(path, sr=None, mono=True)
        zeros, longest, run = y == 0.0, 0, 0
        for value in zeros:
            run = run + 1 if value else 0
            longest = max(longest, run)
        return longest

    assert main(["channel", str(src), str(tmp_path / "scattered.wav"),
                 "--loss", "0.05", "--seed", "1"]) == 0
    scattered = capsys.readouterr().out
    assert main(["channel", str(src), str(tmp_path / "bursty.wav"),
                 "--loss", "0.05", "--burst", "8", "--seed", "1"]) == 0
    bursty = capsys.readouterr().out

    assert "run(s) of" in scattered and "run(s) of" in bursty
    assert longest_hole(tmp_path / "bursty.wav") > \
        longest_hole(tmp_path / "scattered.wav")


def test_synth_command_sends_the_corpus_down_a_link(tmp_path, capsys):
    import kudio
    clean, noise, out = tmp_path / "c", tmp_path / "n", tmp_path / "m"
    clean.mkdir()
    noise.mkdir()
    _write_speech(clean / "a.wav")
    _write_speech(noise / "n.wav", seed=3)

    manifest = tmp_path / "m.json"
    assert main(["synth", "--clean", str(clean), "--noise", str(noise),
                 "--out", str(out), "--snr", "10", "--mode", "reg",
                 "--channel", "telephone", "--seed", "1",
                 "--manifest", str(manifest)]) == 0
    printed = capsys.readouterr().out
    assert "link: telephone" in printed

    pairs = kudio.load_manifest(manifest)
    assert pairs and all(p.channel == "telephone" for p in pairs)
    y, sr = kudio.file_load(pairs[0].noisy, sr=None, mono=True)
    assert kudio.audio_report(y, sr).band_limited


def test_synth_command_overrides_a_preset_without_restating_it(tmp_path,
                                                               capsys):
    import kudio
    clean, noise, out = tmp_path / "c", tmp_path / "n", tmp_path / "m"
    clean.mkdir()
    noise.mkdir()
    _write_speech(clean / "a.wav")
    _write_speech(noise / "n.wav", seed=3)

    manifest = tmp_path / "m.json"
    assert main(["synth", "--clean", str(clean), "--noise", str(noise),
                 "--out", str(out), "--snr", "10", "--mode", "reg",
                 "--channel", "voip", "--channel-loss", "0.1",
                 "--channel-burst", "6", "--seed", "1",
                 "--manifest", str(manifest)]) == 0
    capsys.readouterr()
    pairs = kudio.load_manifest(manifest)
    assert pairs[0].channel == "mu_law+loss10%x6-hold"


def test_synth_command_writes_reverberant_targets(tmp_path, capsys):
    import kudio
    clean, noise, out = tmp_path / "c", tmp_path / "n", tmp_path / "m"
    clean.mkdir()
    noise.mkdir()
    _write_speech(clean / "a.wav")
    _write_speech(noise / "n.wav", seed=3)

    manifest = tmp_path / "m.json"
    assert main(["synth", "--clean", str(clean), "--noise", str(noise),
                 "--out", str(out), "--snr", "10", "--mode", "reg",
                 "--rt60", "0.5", "--targets", "--seed", "1",
                 "--manifest", str(manifest)]) == 0
    printed = capsys.readouterr().out
    assert "TARGETS" in printed

    pairs = kudio.load_manifest(manifest)
    assert all(pathlib.Path(p.target).is_file() for p in pairs)
    assert all(pathlib.Path(p.target).name == pathlib.Path(p.noisy).name
               for p in pairs)
