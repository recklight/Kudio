# -*- coding: utf-8 -*-
"""What soundfile does not read: m4a, mp4, aac, webm, 3gp, amr, ALAC in a CAF.

Twice over. With a stand-in for audioread, so that the path from its blocks of
samples to each of kudio's readers runs on a machine with neither audioread
nor ffmpeg; and with the real two on files made here, which is skipped where
one is missing unless KUDIO_REQUIRE_FFMPEG=1 says that it must not be.
"""
import importlib.util
import os
import shutil
import struct
import subprocess
import sys
import types
import warnings
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import kudio
from kudio.core import dataset_score, stream, synth
from kudio.exceptions import AudioIOError

#: the first bytes of an MP4 and of a Matroska file, which store how long they
#: are, and of an AMR stream, which does not
MP4 = b'\x00\x00\x00\x18ftypM4A '
MATROSKA = b'\x1aE\xdf\xa3\x01\x00\x00\x00'
AMR = b'#!AMR\n\x3c\x00'


def _tone(sr, channels, seconds=1.5):
    t = np.arange(int(sr * seconds)) / sr
    y = np.stack([0.4 * np.sin(2 * np.pi * 440 * (c + 1) * t) for c in range(channels)], axis=1)
    return np.round(y * 32767).astype('<i2')


def _encoded(folder, head=MP4, name="take.m4a", code=1):
    path = folder / name
    path.write_bytes(head + bytes(200))
    with pytest.raises(sf.LibsndfileError) as refused:
        sf.info(str(path))
    assert refused.value.code == code, "refused for that reason, or this tests nothing"
    return path


def _mp3_with_no_mp3(folder, name="take.mp3"):
    """libsndfile takes the extension for the format here, and finding no frame
    says that the file does not exist."""
    return _encoded(folder, name=name, code=7)


def _alac_caf():
    """A CAF file that says it holds Apple Lossless: libsndfile knows the
    container, and turns the encoding down without its description."""
    describes = struct.pack('>d4sIIIII', 44100.0, b'alac', 0, 0, 4096, 2, 16)
    return (b'caff' + struct.pack('>HH', 1, 0) + b'desc' + struct.pack('>q', len(describes))
            + describes + b'data' + struct.pack('>q', 4) + bytes(4))


class DecodeError(Exception):
    pass


class NoBackendError(DecodeError):
    pass


class RawAudioFile:
    pass


class FFmpegAudioFile:
    pass


class Reader:
    """What audioread hands out: the header's figures and an iterator over
    blocks of 16-bit samples, here of 5 so that they split frames."""

    def __init__(self, pcm, samplerate, duration=None, fail_after=None):
        self.data, self.samplerate = pcm.tobytes(), samplerate
        self.channels = 1 if pcm.ndim == 1 else pcm.shape[1]
        self.duration = pcm.shape[0] / (samplerate or 1) if duration is None else duration
        self.fail_after = fail_after

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def __iter__(self):
        for n, start in enumerate(range(0, len(self.data), 10)):
            if n == self.fail_after:
                raise DecodeError()
            yield self.data[start:start + 10]


@pytest.fixture
def audioread(monkeypatch):
    """Install a stand-in for audioread. *opens* is what opening a file gives,
    a Reader or an error; *ffmpeg* whether it finds ffmpeg, and "later" for not
    before its memory is cleared."""
    def install(opens, ffmpeg=True):
        module = types.ModuleType("audioread")
        module.DecodeError, module.NoBackendError = DecodeError, NoBackendError
        module.calls = []

        def available_backends(flush_cache=False):
            found = ffmpeg is True or (ffmpeg == "later" and flush_cache)
            return [RawAudioFile] + [FFmpegAudioFile] * found

        def audio_open(path, backends=None):
            module.calls.append((path, backends))
            if isinstance(opens, BaseException):
                raise opens
            return opens

        module.available_backends, module.audio_open = available_backends, audio_open
        monkeypatch.setitem(sys.modules, "audioread", module)
        return module
    return install


@pytest.fixture
def played(monkeypatch):
    """What play_audio hands to the sound card."""
    out = []
    monkeypatch.setitem(sys.modules, "sounddevice",
                        types.SimpleNamespace(play=lambda y, rate: out.append((y, rate))))
    return out


# -- every reader, from the same blocks ----------------------------------------

@pytest.mark.parametrize("sr, channels", [(44100, 1), (44100, 2), (48000, 6), (8000, 1)])
def test_file_load_reads_it_with_and_without_a_rate(tmp_path, audioread, sr, channels):
    pcm = _tone(sr, channels)
    expected = pcm.astype(np.float32) / 32768
    path = _encoded(tmp_path)
    audioread(Reader(pcm, sr))

    y, rate = kudio.file_load(path, mono=False)
    assert (rate, y.dtype) == (sr, np.float32)
    assert np.array_equal(y, expected if channels > 1 else expected[:, 0])
    assert y.flags.c_contiguous and y.flags.writeable

    mixed, _ = kudio.file_load(path)
    assert np.array_equal(mixed, expected.mean(axis=1) if channels > 1 else expected[:, 0])
    again, rate = kudio.file_load(path, sr=sr)
    assert rate == sr and np.array_equal(again, mixed)
    at16, rate = kudio.file_load(path, sr=16000, mono=False)
    assert rate == 16000 and np.array_equal(at16, kudio.resample(y, sr, 16000))


def test_every_other_reader_takes_it_too(tmp_path, audioread, played):
    """file_load opened an m4a at a rate and nothing else did: not without the
    rate, not audio_info, not play_audio."""
    pcm = _tone(44100, 2)
    expected = pcm.astype(np.float32) / 32768
    path = _encoded(tmp_path)
    audioread(Reader(pcm, 44100))

    assert stream.play_audio(str(path)) is None
    assert played[0][1] == 44100 and np.array_equal(played[0][0], expected)

    y, rate = synth._load(path, None, "clean")
    assert rate == 44100 and np.array_equal(y, expected.mean(axis=1))
    at8, rate = synth._load(path, 8000, "noise")
    assert rate == 8000 and np.array_equal(at8, kudio.resample(y, 44100, 8000))

    assert dataset_score._set_rate([path], "reference") == 44100

    kudio.save_spectrogram_as_wave(tmp_path / "out.wav", path, kudio.waveform_to_spectrogram(y))
    assert kudio.audio_info(tmp_path / "out.wav").sr == 44100


def test_the_plot_of_a_file_takes_it_too(tmp_path, audioread, monkeypatch):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from kudio.util import visual
    shown = []
    monkeypatch.setattr(visual.plt, "show", lambda: shown.append(True))
    audioread(Reader(_tone(16000, 1), 16000))
    visual.Common.plot_stft_w_2mfe(_encoded(tmp_path), n_mels=16, sub_mfe_fq=8000)
    assert shown


def test_audio_info_counts_what_was_decoded(tmp_path, audioread):
    pcm = _tone(44100, 2)                              # 1.5 s, and a header that says 1.4
    path = _encoded(tmp_path)
    audioread(Reader(pcm, 44100, duration=1.4))
    info = kudio.audio_info(path)
    assert (info.sr, info.channels, info.frames, info.duration) == (44100, 2, 66150, 1.5)
    assert info.peak == np.max(np.abs(pcm.astype(np.int32))) / 32768
    assert (info.path, info.subtype, info.format) == (path, None, None)
    assert "1.500s, 44100 Hz, 2ch, peak" in str(info)
    assert info.frames == len(kudio.file_load(path)[0])


def test_audio_info_of_the_header_decodes_nothing(tmp_path, audioread):
    reader = Reader(_tone(44100, 2), 44100, duration=1.4, fail_after=0)
    audioread(reader)
    info = kudio.audio_info(_encoded(tmp_path), peak=False)
    assert (info.sr, info.channels, info.frames, info.duration) == (44100, 2, 61740, 1.4)
    assert info.peak is None


def test_a_header_without_a_duration_is_counted(tmp_path, audioread):
    """A live recording's WebM says nothing about its length."""
    audioread(Reader(_tone(8000, 1), 8000, duration=0))
    info = kudio.audio_info(_encoded(tmp_path, MATROSKA), peak=False)
    assert (info.frames, info.duration, info.peak) == (12000, 1.5, None)


def test_the_full_negative_scale_is_a_peak_of_one(tmp_path, audioread):
    audioread(Reader(np.array([0, -32768, 100], dtype='<i2'), 8000))
    assert kudio.audio_info(_encoded(tmp_path)).peak == 1.0


def test_the_command_line_describes_it(tmp_path, audioread, capsys):
    from kudio.cli import main
    audioread(Reader(_tone(44100, 2), 44100))
    assert main(["info", str(_encoded(tmp_path))]) == 0
    out = capsys.readouterr().out
    assert "44100 Hz" in out and "not read by soundfile, decoded with audioread" in out


# -- how audioread is asked ----------------------------------------------------

def test_the_decoder_gets_the_whole_path_and_no_wav_reader(tmp_path, audioread, monkeypatch):
    """ffmpeg takes what stands before a colon in a relative path for a
    protocol, and audioread's own WAV reader leaves its file open when it
    turns one down."""
    _encoded(tmp_path)
    monkeypatch.chdir(tmp_path)
    module = audioread(Reader(_tone(8000, 1), 8000))
    kudio.file_load("take.m4a")
    (path, backends), = module.calls
    assert path == str(tmp_path / "take.m4a") and backends == [FFmpegAudioFile]


def test_ffmpeg_installed_in_the_meantime_is_found(tmp_path, audioread):
    """audioread looks for ffmpeg once and remembers that there was none."""
    audioread(Reader(_tone(8000, 1), 8000), ffmpeg="later")
    assert kudio.file_load(_encoded(tmp_path))[1] == 8000


def test_what_soundfile_reads_never_reaches_audioread(tmp_path, audioread, sine):
    module = audioread(AssertionError("not for a WAV"))
    kudio.save_wave(tmp_path / "a.wav", sine, 16000)
    for sr in (None, 8000):
        kudio.file_load(tmp_path / "a.wav", sr=sr)
    kudio.audio_info(tmp_path / "a.wav")
    assert not module.calls


def test_an_encoding_soundfile_does_not_support_is_decoded_without_a_word(tmp_path, audioread):
    """Apple Lossless in a CAF: there is nothing wrong with the file."""
    path = _encoded(tmp_path, _alac_caf(), "take.caf", code=4)
    pcm = _tone(44100, 2)
    audioread(Reader(pcm, 44100))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        y, rate = kudio.file_load(path, mono=False)
        assert kudio.audio_info(path).frames == kudio.audio_info(path, peak=False).frames == 66150
        assert synth._load(path, None, "clean")[1] == 44100
    assert rate == 44100 and np.array_equal(y, pcm.astype(np.float32) / 32768)


def test_what_is_missing_for_such_an_encoding_is_said_of_it(tmp_path, audioread, monkeypatch):
    path = _encoded(tmp_path, _alac_caf(), "take.caf", code=4)
    audioread(NoBackendError())
    with pytest.raises(AudioIOError, match="does not support the encoding, and it could not be "
                                           "decoded either"):
        kudio.file_load(path)
    monkeypatch.setitem(sys.modules, "audioread", None)
    with pytest.raises(AudioIOError, match=r"does not support the encoding\. .*pip install"):
        kudio.file_load(path)


@pytest.mark.parametrize("name, head", [("bad.flac", b"fLaC"), ("bad.ogg", b"OggS")])
@pytest.mark.parametrize("sr", [None, 16000])
def test_a_format_soundfile_knows_is_not_decoded_another_way(tmp_path, audioread, name, head, sr):
    """Of a damaged FLAC ffmpeg hands back what it can and calls that the
    file. soundfile says what is wrong with it."""
    path = tmp_path / name
    path.write_bytes(head + bytes(60))
    module = audioread(Reader(_tone(8000, 1), 8000))
    with pytest.raises(AudioIOError) as caught:
        kudio.file_load(path, sr=sr)
    assert "Error opening" in str(caught.value), "soundfile's own words"
    assert not module.calls


# -- a name soundfile knows, on a file it does not -----------------------------

@pytest.mark.parametrize("name", ["take.flac", "take.OGG", "take.wav", "take.opus", "take.aif",
                                  "take.mp3", "take.MP3"])
@pytest.mark.parametrize("reader", ['file_load', 'file_load at a rate', 'audio_info',
                                    'Synthesizer', 'save_spectrogram_as_wave'])
def test_a_file_that_is_not_what_its_name_says_is_read_with_a_warning(tmp_path, audioread,
                                                                      reader, name):
    """Of a FLAC that lost its beginning ffmpeg hands back the rest, and an m4a
    that somebody called .mp3 it simply reads."""
    audioread(Reader(_tone(8000, 1), 8000, duration=3.0))
    path = _mp3_with_no_mp3(tmp_path, name) if name[-3:].lower() == "mp3" else \
        _encoded(tmp_path, name=name)
    with pytest.warns(UserWarning) as caught:
        READS[reader][0](path)
    said, = [str(w.message) for w in caught]
    assert said == (f"{path}: soundfile does not recognise it as the {path.suffix.lower()} file "
                    f"its name says, and ffmpeg decoded 1.50 s of it. The file is damaged, or "
                    f"its extension is wrong.")
    assert caught[0].filename == __file__


@pytest.mark.parametrize("reader", ['file_load', 'file_load at a rate', 'audio_info',
                                    'audio_info of the header', 'Synthesizer'])
def test_an_mp3_with_no_mp3_in_it_is_not_said_to_be_missing(tmp_path, audioread, monkeypatch,
                                                           reader):
    path = _mp3_with_no_mp3(tmp_path)
    audioread(NoBackendError())
    said = _said(*READS[reader], path)
    assert "could not be decoded either" in said and "does not exist" not in said
    monkeypatch.setitem(sys.modules, "audioread", None)
    assert "pip install audioread" in _said(*READS[reader], path)


def test_the_header_of_such_a_file_is_not_taken_at_its_word(tmp_path, audioread):
    audioread(Reader(_tone(8000, 1), 8000, duration=3.0))
    with pytest.warns(UserWarning, match=r"and ffmpeg reads 3\.0 s off its header"):
        kudio.audio_info(_encoded(tmp_path, name="take.flac"), peak=False)


@pytest.mark.parametrize("name", ["take.m4a", "take.webm", "take", "take.dat"])
def test_a_name_soundfile_does_not_know_is_no_reason_for_a_warning(tmp_path, audioread, name):
    audioread(Reader(_tone(8000, 1), 8000))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        kudio.file_load(_encoded(tmp_path, name=name))


# -- what is missing -----------------------------------------------------------

#: every reader: the call and the error
READS = {
    'file_load': (lambda p: kudio.file_load(p), AudioIOError),
    'file_load at a rate': (lambda p: kudio.file_load(p, sr=8000), AudioIOError),
    'audio_info': (lambda p: kudio.audio_info(p), AudioIOError),
    'audio_info of the header': (lambda p: kudio.audio_info(p, peak=False), AudioIOError),
    'Synthesizer': (lambda p: synth._load(p, None, "clean"), kudio.SynthesisError),
    'save_spectrogram_as_wave': (
        # as many frames as 1.5 s at 8 kHz have
        lambda p: kudio.save_spectrogram_as_wave(p.parent / "o.wav", p, np.zeros((47, 257))),
        AudioIOError),
}


def _said(call, error, path):
    with pytest.raises(error) as caught:
        call(path)
    assert type(caught.value) is error
    message = str(caught.value)
    assert str(path) in message and "soundfile does not recognise the format" in message
    return message


@pytest.mark.parametrize("name", READS)
def test_without_audioread_it_says_to_install_it(tmp_path, monkeypatch, name):
    monkeypatch.setitem(sys.modules, "audioread", None)
    said = _said(*READS[name], _encoded(tmp_path))
    assert "audioread is not installed: pip install audioread" in said and "ffmpeg" in said


@pytest.mark.parametrize("name", READS)
def test_without_ffmpeg_it_says_to_install_that(tmp_path, audioread, name):
    audioread(AssertionError("there is nothing to open it with"), ffmpeg=False)
    said = _said(*READS[name], _encoded(tmp_path))
    assert "audioread found none: install ffmpeg and put it on the PATH" in said
    assert "pip install" not in said


def test_an_audioread_that_will_not_import_says_why(tmp_path, audioread):
    """Python 3.13 has no aifc, which audioread imports."""
    module = audioread(None)

    def broken(flush_cache=False):
        raise ModuleNotFoundError("No module named 'aifc'", name='aifc')
    module.available_backends = broken
    said = _said(*READS['file_load'], _encoded(tmp_path))
    assert "which cannot be imported: No module named 'aifc'" in said
    assert "pip install" not in said


@pytest.mark.parametrize("opens", [NoBackendError(), OSError("file not found"), EOFError(),
                                   Reader(_tone(8000, 1), 8000, fail_after=3),
                                   Reader(np.zeros(0, dtype='<i2'), 8000),
                                   Reader(_tone(8000, 1), 0)],
                         ids=["no backend", "OSError", "EOFError", "fails half way",
                              "nothing in it", "no rate"])
@pytest.mark.parametrize("name", ['file_load', 'file_load at a rate', 'audio_info', 'Synthesizer',
                                  'save_spectrogram_as_wave'])
def test_a_file_that_will_not_decode_is_told_from_a_missing_decoder(tmp_path, audioread, name,
                                                                    opens):
    audioread(opens)
    said = _said(*READS[name], _encoded(tmp_path))
    assert ("could not be decoded either: the file is damaged, cut short or holds no audio"
            in said)


def test_play_audio_keeps_its_answer_and_logs_the_reason(tmp_path, audioread, played, caplog):
    audioread(NoBackendError())
    path = _encoded(tmp_path)
    with caplog.at_level("ERROR", logger="kudio.core.stream"):
        assert stream.play_audio(str(path)) == "input file can't be played"
    assert not played
    assert str(path) in caplog.text and "could not be decoded either" in caplog.text


# -- files that end early ------------------------------------------------------

@pytest.mark.parametrize("head", [MP4, MATROSKA])
@pytest.mark.parametrize("name", ['file_load', 'file_load at a rate', 'audio_info',
                                  'Synthesizer', 'save_spectrogram_as_wave'])
def test_less_than_the_header_promises_is_said(tmp_path, audioread, name, head):
    """ffmpeg decodes half an m4a without an error. The warning points at the
    line that asked for the file."""
    audioread(Reader(_tone(8000, 1), 8000, duration=3.0))
    with pytest.warns(UserWarning, match=r"1\.50 s could be decoded of the 3\.0 s") as caught:
        READS[name][0](_encoded(tmp_path, head))
    assert caught[0].filename == __file__


@pytest.mark.parametrize("head, duration", [(MP4, 1.5), (MP4, 1.55), (MP4, 1.4), (MP4, 0),
                                            (AMR, 3.0)])
def test_a_figure_that_is_close_or_a_guess_is_not_held_against_it(tmp_path, audioread, head,
                                                                  duration):
    """The header's figure comes to a tenth of a second, and for a bare stream
    it is a guess from the bit rate."""
    audioread(Reader(_tone(8000, 1), 8000, duration=duration))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        kudio.file_load(_encoded(tmp_path, head))


# -- the real decoder ----------------------------------------------------------

#: extension -> rate, channels, what ffmpeg is asked for. Not every ffmpeg comes
#: with an AMR encoder, so AMR's frames are written by hand further down.
FORMATS = {
    'm4a': (44100, 2, ['-c:a', 'aac']),
    'mp4': (48000, 2, ['-c:a', 'aac']),
    'aac': (44100, 2, ['-c:a', 'aac', '-f', 'adts']),
    'webm': (48000, 2, ['-c:a', 'libopus']),
    '3gp': (16000, 1, ['-c:a', 'aac']),
    'mka': (16000, 1, ['-c:a', 'aac']),
}


@pytest.fixture(scope="module")
def ffmpeg():
    missing = [name for name, there in (("ffmpeg", shutil.which("ffmpeg")),
                                        ("audioread", importlib.util.find_spec("audioread")))
               if not there]
    if missing:
        said = f"{' and '.join(missing)} not installed"
        if os.environ.get("KUDIO_REQUIRE_FFMPEG") == "1":
            pytest.fail(f"{said}, and KUDIO_REQUIRE_FFMPEG=1")
        pytest.skip(said)
    return shutil.which("ffmpeg")


def _encode(ffmpeg, folder, extension, *extra):
    sr, channels, args = FORMATS[extension]
    pcm = _tone(sr, channels, seconds=2.0)
    path = folder / f"取 樣.{extension}"
    subprocess.run([ffmpeg, '-y', '-loglevel', 'error', '-f', 's16le', '-ar', str(sr),
                    '-ac', str(channels), '-i', '-', *args, *extra, str(path)],
                   input=pcm.tobytes(), check=True)
    return path, sr, channels


def _pitch(y, sr):
    return float(np.fft.rfftfreq(len(y), 1 / sr)[np.argmax(np.abs(np.fft.rfft(y)))])


@pytest.mark.parametrize("extension", FORMATS)
def test_every_reader_agrees_on_a_real_file(tmp_path, ffmpeg, played, extension):
    path, sr, channels = _encode(ffmpeg, tmp_path, extension)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        native, rate = kudio.file_load(path, mono=False)
        at16, rate16 = kudio.file_load(path, sr=16000, mono=False)
        info = kudio.audio_info(path)
        header = kudio.audio_info(path, peak=False)
        assert stream.play_audio(str(path)) is None
        mixed, synth_rate = synth._load(path, None, "clean")

    assert (rate, synth_rate, played[0][1], info.sr, header.sr) == (sr,) * 5
    assert dataset_score._set_rate([path], "reference") == sr
    columns = native.reshape(len(native), -1)
    assert columns.shape[1] == channels == info.channels == header.channels
    assert np.array_equal(played[0][0], native)
    assert np.array_equal(mixed, columns.mean(axis=1) if channels > 1 else native)
    assert rate16 == 16000
    assert np.array_equal(at16, native if sr == 16000 else kudio.resample(native, sr, 16000))

    # AAC and Opus pad, and by how much depends on the ffmpeg that encoded it:
    # 0.112 s in a Matroska file from the ffmpeg 8.0.1 of Ubuntu 26.04. The sound
    # itself is there, channel for channel
    assert len(native) / sr == pytest.approx(2.0, abs=0.15)
    assert (info.frames, info.duration) == (len(native), len(native) / sr)
    assert info.peak == float(np.max(np.abs(native)))
    assert header.duration == pytest.approx(2.0, abs=0.15)
    for c in range(channels):
        assert _pitch(columns[:, c], sr) == pytest.approx(440 * (c + 1), abs=2)


def test_a_real_amr_stream(tmp_path, ffmpeg):
    """100 frames of AMR's 12.2 kbit/s mode, 20 ms each."""
    path = tmp_path / "take.amr"
    path.write_bytes(b"#!AMR\n" + (b"\x3c" + bytes(31)) * 100)
    y, rate = kudio.file_load(path)
    assert (rate, y.ndim) == (8000, 1) and len(y) / rate == pytest.approx(2.0, abs=0.1)
    assert kudio.audio_info(path).frames == len(y)


def test_a_real_file_cut_short(tmp_path, ffmpeg):
    """Where the index is at the end of the file, half of it cannot be opened;
    where it is in front, half of it is half the clip and a warning."""
    for extra, opens in (((), False), (('-movflags', '+faststart'), True)):
        path, sr, _ = _encode(ffmpeg, tmp_path, 'm4a', *extra)
        path.write_bytes(path.read_bytes()[:path.stat().st_size // 2])
        for read in (lambda: kudio.file_load(path), lambda: kudio.file_load(path, sr=16000),
                     lambda: kudio.audio_info(path)):
            if opens:
                with pytest.warns(UserWarning, match="s could be decoded of the 2.0 s"):
                    read()
            else:
                with pytest.raises(AudioIOError, match="could not be decoded either"):
                    read()


@pytest.mark.parametrize("extension", ["mp3", "wav", "flac", "ogg"])
def test_a_real_m4a_under_another_name(tmp_path, ffmpeg, extension):
    path, sr, _ = _encode(ffmpeg, tmp_path, 'm4a')
    whole, _ = kudio.file_load(path, mono=False)
    renamed = path.rename(tmp_path / f"take.{extension}")
    for rate in (None, sr):
        with pytest.warns(UserWarning, match=f"as the .{extension} file its name says"):
            y, native = kudio.file_load(renamed, sr=rate, mono=False)
        assert native == sr and np.array_equal(y, whole)


def test_the_synthesizer_looks_at_the_header_of_one_first(tmp_path, ffmpeg):
    """An m4a called .wav gets that far, and one that lost its header empties no
    folder."""
    made, rate, _ = _encode(ffmpeg, tmp_path, "m4a")
    shutil.copy(made, tmp_path / "take.wav")
    noise = tmp_path / "noise.wav"
    sf.write(str(noise), 0.1 * np.random.default_rng(0).standard_normal(20000), 16000)
    out = tmp_path / "out"
    out.mkdir()
    (out / "keep.txt").write_text("from an earlier run")
    whole = (tmp_path / "take.wav").read_bytes()
    (tmp_path / "take.wav").write_bytes(whole[:len(whole) // 3])
    syx = kudio.Synthesizer(str(tmp_path / "take.wav"), str(noise), out, snr_ratio=0)
    with pytest.raises(kudio.SynthesisError, match="could not read clean file .*take.wav"):
        syx.syn(seed=0)
    assert [p.name for p in out.iterdir()] == ["keep.txt"]
    (tmp_path / "take.wav").write_bytes(whole)
    # the packets are those of what is decoded; the header has tenths of a second
    link, mixture = {'loss': 0.9, 'burst': 1.5}, out / "take_noise_0dB.wav"
    with pytest.warns(UserWarning, match="not recognise it as the .wav"):
        samples, _ = kudio.file_load(tmp_path / "take.wav")
    own = (synth._file_seed(0, mixture, out) * 2654435761) & 0x7FFFFFFF
    with pytest.raises(kudio.FeatureError) as expected:
        kudio.dropouts(np.zeros(len(samples)), rate, seed=own, **link)
    syx = kudio.Synthesizer(str(tmp_path / "take.wav"), str(noise), out, snr_ratio=0,
                            channel=link)
    with warnings.catch_warnings(record=True) as said:
        warnings.simplefilter("always")
        with pytest.raises(kudio.FeatureError) as caught:
            syx.syn(seed=0)
    assert str(caught.value) == f"{mixture}: {expected.value}"
    assert not said, "what there is to warn about is said when the file is mixed"
    assert [p.name for p in out.iterdir()] == ["keep.txt"]
    syx = kudio.Synthesizer(str(tmp_path / "take.wav"), str(noise), out, snr_ratio=0)
    with pytest.warns(UserWarning, match="not recognise it as the .wav"):
        assert len(syx.syn(seed=0)) == 1


def test_a_real_alac_caf(tmp_path, ffmpeg):
    path = tmp_path / "take.caf"
    pcm = _tone(44100, 2, seconds=2.0)
    subprocess.run([ffmpeg, '-y', '-loglevel', 'error', '-f', 's16le', '-ar', '44100', '-ac', '2',
                    '-i', '-', '-c:a', 'alac', str(path)], input=pcm.tobytes(), check=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        y, rate = kudio.file_load(path, mono=False)
    # lossless, but for the frames ffmpeg's own muxer leaves off the end
    assert rate == 44100 and 0 < len(y) <= len(pcm)
    assert np.array_equal(y, pcm[:len(y)].astype(np.float32) / 32768)


def _damaged(whole, how, rng):
    if how == "head cut off":
        return whole[int(rng.integers(1, 3000)):]
    if how == "junk in front":
        return rng.bytes(int(rng.integers(1, 4000))) + whole
    if how == "zeros in front":
        return bytes(int(rng.integers(1, 4000))) + whole
    if how == "cut to half":
        return whole[:len(whole) // 2]
    if how == "cut short":
        return whole[:int(rng.integers(20, len(whole)))]
    hole = bytearray(whole)
    at = int(rng.integers(0, 200))
    size = int(rng.integers(1, 60))
    hole[at:at + size] = bytes(size)
    return bytes(hole)


@pytest.mark.parametrize("how", ["head cut off", "junk in front", "zeros in front",
                                 "cut to half", "cut short", "a hole in the header"])
@pytest.mark.parametrize("name, subtype", [("a.flac", "PCM_16"), ("a.ogg", "VORBIS"),
                                           ("a.wav", "PCM_16"), ("a.aiff", "PCM_16"),
                                           ("a.mp3", "MPEG_LAYER_III")])
def test_ffmpeg_never_reads_a_damaged_file_in_silence(tmp_path, ffmpeg, name, subtype, how):
    """Ten of each kind of damage: an error, what soundfile itself makes of the
    file, or a warning that ffmpeg had to."""
    sf.write(str(tmp_path / name), _tone(44100, 2, 0.5), 44100, subtype=subtype)
    whole = (tmp_path / name).read_bytes()
    rng = np.random.default_rng(len(name) * 100 + len(how))
    for trial in range(10):
        path = tmp_path / f"{trial}_{name}"
        path.write_bytes(_damaged(whole, how, rng))
        try:
            by_soundfile = len(sf.read(str(path))[0])
        except sf.SoundFileError:
            by_soundfile = None
        for sr in (None, 44100):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    read = len(kudio.file_load(path, sr=sr, mono=False)[0])
                except AudioIOError:
                    assert by_soundfile is None
                    continue
            said = [str(w.message) for w in caught if w.category is UserWarning]
            if by_soundfile is None:
                assert len(said) == 1 and "soundfile does not recognise it as the" in said[0]
            else:
                # soundfile reads it, and how long it was meant to be is in the
                # header of a WAV alone
                assert read == by_soundfile
                assert len(said) == (name == "a.wav" and how.startswith("cut"))
                assert not said or "may have been cut short" in said[0]


def test_a_real_file_that_is_not_audio(tmp_path, ffmpeg):
    path = tmp_path / "notes.m4a"
    path.write_text("not audio, only text\n" * 50)
    with pytest.raises(AudioIOError, match="could not be decoded either"):
        kudio.file_load(path)


def test_the_real_audioread_without_ffmpeg(tmp_path, ffmpeg, monkeypatch):
    path, _, _ = _encode(ffmpeg, tmp_path, 'm4a')
    import audioread
    if [b for b in audioread.available_backends()
            if b.__name__ not in ('RawAudioFile', 'FFmpegAudioFile')]:
        pytest.skip("this machine decodes without ffmpeg")
    monkeypatch.setenv("PATH", str(Path(sys.executable).parent))
    monkeypatch.setattr(audioread, "BACKENDS", [])
    with pytest.raises(AudioIOError, match="audioread found none: install ffmpeg"):
        kudio.file_load(path)
