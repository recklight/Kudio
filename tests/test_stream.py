# -*- coding: utf-8 -*-
"""StreamRecorder, and the device layer under it.

No sound card is needed. For StreamRecorder the device layer is replaced by a
fake reader that hands frames over on demand, and what is under test is
everything StreamRecorder owns: accumulation without loss, the level and
elapsed readings, the rolling tail, the memory cap, and the channel layout of
the finished take. Below it, a fake sounddevice keeps what each class opened
and wrote: the device index, the sample type, the frames.
"""
import logging
import sys
import threading
import types
import wave

import numpy as np
import pytest

import kudio
from kudio.core import stream as stream_module

SR = 16000


class FakeReader:
    """Stands in for LocalStreamReader, minus PortAudio."""

    def __init__(self, device=None, rate=None, channels=1, frame_size=1024,
                 wav_format=1, buffer=None, on_frame=None):
        self.rate_ = rate or SR
        self.channels_ = channels
        self.frame_size_ = frame_size
        self.on_frame = on_frame
        self.isListening = threading.Event()
        self.monitoring = False
        self.terminated = False

    def get_status(self):
        return True

    def play_audio(self):
        self.monitoring = True

    def stream_start(self):
        self.isListening.set()

    def terminate(self):
        self.isListening.clear()
        self.terminated = True

    # -- test helper: deliver one chunk exactly as the callback would
    def feed(self, data):
        self.on_frame(np.asarray(data, dtype=np.float32))


@pytest.fixture
def fake_reader(monkeypatch):
    made = []

    def factory(**kwargs):
        reader = FakeReader(**kwargs)
        made.append(reader)
        return reader

    monkeypatch.setattr(stream_module, "LocalStreamReader", factory)
    return made


def _recorder(fake_reader, **kwargs):
    rec = kudio.StreamRecorder(sr=SR, **kwargs).start()
    return rec, fake_reader[0]


def test_start_opens_the_stream(fake_reader):
    rec, reader = _recorder(fake_reader)
    assert rec.recording is True
    assert reader.isListening.is_set()


def test_start_is_idempotent(fake_reader):
    rec, reader = _recorder(fake_reader)
    rec.start()
    assert len(fake_reader) == 1


def test_captured_audio_is_returned_in_order(fake_reader):
    rec, reader = _recorder(fake_reader)
    reader.feed(np.arange(100, dtype=np.float32))
    reader.feed(np.arange(100, 250, dtype=np.float32))
    take = rec.stop()
    assert np.array_equal(take, np.arange(250, dtype=np.float32))


def test_nothing_is_dropped_under_load(fake_reader):
    """An AudioBuffer discards when full; a recording must not."""
    rec, reader = _recorder(fake_reader)
    for _ in range(500):
        reader.feed(np.ones(1024, dtype=np.float32))
    assert len(rec.stop()) == 500 * 1024


def test_stop_can_be_called_twice(fake_reader):
    rec, reader = _recorder(fake_reader)
    reader.feed(np.ones(100, dtype=np.float32))
    first = rec.stop()
    assert np.array_equal(rec.stop(), first)


def test_level_tracks_the_latest_chunk(fake_reader):
    rec, reader = _recorder(fake_reader)
    reader.feed(np.full(100, 0.5, dtype=np.float32))
    assert rec.level() == pytest.approx(0.5)
    assert rec.level_db() == pytest.approx(-6.02, abs=0.01)
    reader.feed(np.full(100, 0.1, dtype=np.float32))
    assert rec.level() == pytest.approx(0.1)


def test_level_of_silence_is_minus_infinity(fake_reader):
    rec, _ = _recorder(fake_reader)
    assert rec.level_db() == float("-inf")


def test_elapsed_counts_seconds(fake_reader):
    rec, reader = _recorder(fake_reader)
    reader.feed(np.zeros(SR // 2, dtype=np.float32))
    assert rec.elapsed() == pytest.approx(0.5)


def test_tail_returns_only_the_recent_window(fake_reader):
    rec, reader = _recorder(fake_reader)
    for _ in range(10):
        reader.feed(np.arange(SR, dtype=np.float32))
    tail = rec.tail(0.25)
    assert len(tail) == SR // 4
    assert tail[-1] == pytest.approx(SR - 1)


def test_tail_is_short_when_little_has_been_recorded(fake_reader):
    rec, reader = _recorder(fake_reader)
    reader.feed(np.zeros(100, dtype=np.float32))
    assert len(rec.tail(5.0)) == 100


def test_tail_before_anything_arrives_is_empty(fake_reader):
    rec, _ = _recorder(fake_reader)
    assert rec.tail(1.0).size == 0


def test_take_does_not_stop_the_stream(fake_reader):
    rec, reader = _recorder(fake_reader)
    reader.feed(np.ones(100, dtype=np.float32))
    assert len(rec.take()) == 100
    assert rec.recording is True


def test_max_seconds_caps_the_take(fake_reader):
    rec, reader = _recorder(fake_reader, max_seconds=1.0)
    for _ in range(4):
        reader.feed(np.ones(SR // 2, dtype=np.float32))
    assert rec.capped is True
    assert len(rec.stop()) <= SR
    # the level keeps updating so the meter does not freeze
    reader.feed(np.full(10, 0.25, dtype=np.float32))
    assert rec.level() == pytest.approx(0.25)


def test_reset_throws_the_take_away(fake_reader):
    rec, reader = _recorder(fake_reader)
    reader.feed(np.ones(100, dtype=np.float32))
    rec.reset()
    assert rec.take().size == 0
    assert rec.elapsed() == 0.0
    assert rec.recording is True


def test_stereo_comes_back_channels_last(fake_reader):
    rec = kudio.StreamRecorder(sr=SR, channels=2).start()
    reader = fake_reader[0]
    reader.feed(np.array([1, 2, 3, 4, 5, 6], dtype=np.float32))  # interleaved
    take = rec.stop()
    assert take.shape == (3, 2)
    assert np.array_equal(take[:, 0], [1, 3, 5])
    assert rec.elapsed() == pytest.approx(3 / SR)


def test_monitor_is_off_unless_asked(fake_reader):
    rec, reader = _recorder(fake_reader)
    assert reader.monitoring is False
    rec2 = kudio.StreamRecorder(sr=SR, monitor=True).start()
    assert fake_reader[1].monitoring is True
    rec2.stop()


def test_context_manager_stops_the_stream(fake_reader):
    with kudio.StreamRecorder(sr=SR) as rec:
        fake_reader[0].feed(np.ones(50, dtype=np.float32))
        assert rec.recording is True
    assert fake_reader[0].terminated is True


def test_no_input_device_is_a_device_error(monkeypatch):
    class Dead(FakeReader):
        def get_status(self):
            return False

    monkeypatch.setattr(stream_module, "LocalStreamReader",
                        lambda **kwargs: Dead(**kwargs))
    from kudio.exceptions import DeviceError
    with pytest.raises(DeviceError):
        kudio.StreamRecorder()


def test_a_raising_callback_does_not_kill_the_audio_thread():
    """PortAudio's callback runs on its own thread; an exception there would
    just stop the stream delivering, with nothing said."""
    reader = stream_module.LocalStreamReader.__new__(stream_module.LocalStreamReader)
    reader.audio_buffer = None
    reader.data_type = np.float32
    reader.isListening = threading.Event()
    reader.isListening.set()
    reader.isBroadcasting = threading.Event()
    reader.on_frame = lambda data: (_ for _ in ()).throw(RuntimeError("boom"))
    reader._callback(np.zeros(8, dtype=np.float32).tobytes(), 8, None, None)


def test_no_playback_backend_is_a_dependency_error(monkeypatch):
    """A plain ImportError would escape `except kudio.KudioError`."""
    import builtins
    import kudio
    real_import = builtins.__import__

    def without_audio(name, *args, **kwargs):
        if name in ("sounddevice", "pyaudio"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_audio)
    with pytest.raises(kudio.DependencyError, match="audio"):
        kudio.play_audio(np.zeros(160, dtype=np.float32), 16000)


# -- no device, and a device that comes back different -------------------------
def test_a_reader_with_no_device_says_so():
    reader = object.__new__(stream_module.LocalStreamReader)
    reader.isInitial = threading.Event()
    with pytest.raises(kudio.DeviceError, match="no input device"):
        reader.stream_start()
    with pytest.raises(kudio.DeviceError, match="no input device"):
        reader.play_audio()


class ReopeningReader(FakeReader):
    """Closes on terminate(), and comes back at 44.1 kHz when reopened."""

    def get_status(self):
        return not self.terminated

    def set_stream(self, *args):
        self.terminated = False
        self.rate_ = 44100
        return True


def test_a_take_is_not_continued_at_another_rate(monkeypatch):
    monkeypatch.setattr(stream_module, "LocalStreamReader",
                        lambda **kwargs: ReopeningReader(**kwargs))
    rec = kudio.StreamRecorder(sr=SR).start()
    rec._reader.feed(np.ones(1024))
    rec.stop()
    with pytest.raises(kudio.DeviceError, match="44100 Hz"):
        rec.start()
    rec.reset()
    rec.start()
    assert rec.sr == 44100


# -- format codes, and a missing microphone named as a device error -----------
def test_the_portaudio_format_codes_are_public():
    assert (kudio.PA_FLOAT32, kudio.PA_INT16) == (1, 8)


class _HostApis:
    """PortAudio as one Windows 11 machine with no default recording device has
    it, in the dictionaries CheckDevice.p answers with: MME holds two outputs,
    and the microphones are on WDM-KS alone."""

    _apis = ["MME", "Windows WDM-KS"]
    _devices = [(0, 0, 2, "Speakers"), (0, 0, 2, "Monitor"),
                (1, 2, 0, "Microphone (USB Audio)"), (1, 0, 2, "Speakers (USB Audio)")]

    def get_device_count(self):
        return len(self._devices)

    def get_device_info_by_index(self, i):
        api, inputs, outputs, name = self._devices[i]
        return {"index": i, "hostApi": api, "name": name, "maxInputChannels": inputs,
                "maxOutputChannels": outputs, "defaultSampleRate": 44100.0}

    def get_host_api_info_by_index(self, api):
        return {"index": api, "name": self._apis[api],
                "deviceCount": sum(d[0] == api for d in self._devices)}

    def get_device_info_by_host_api_device_index(self, api, i):
        return self.get_device_info_by_index(
            [n for n, d in enumerate(self._devices) if d[0] == api][i])


def test_system_devices_lists_the_inputs_of_every_host_api(caplog):
    check = kudio.CheckDevice.__new__(kudio.CheckDevice)
    check.p = _HostApis()
    with caplog.at_level(logging.INFO, logger="kudio"):
        devices = check.system_devices()
    # `kudio devices` prints the list; an INFO line per device repeated it on stderr
    assert not caplog.records
    assert [d["index"] for d in devices] == [0, 1, 2, 3]
    assert [(d["Device"], d["api"]) for d in devices if d["in"]] == [
        ("Microphone (USB Audio)", "Windows WDM-KS")]


def test_input_device_without_a_microphone_is_a_device_error():
    class NoInputs:
        def get_device_count(self):
            return 0

    check = kudio.CheckDevice.__new__(kudio.CheckDevice)
    check.p = NoInputs()
    from kudio.exceptions import DeviceError
    with pytest.raises(DeviceError, match="No microphone") as caught:
        check.input_device()
    # and a RuntimeError, for code that catches that
    assert isinstance(caught.value, RuntimeError)


# -- on sounddevice: one numbering, the streams it opens -------------------------
class _Stream:
    """A sounddevice stream that keeps what it was opened with and given."""

    def __init__(self, card, kind, **settings):
        self.card, self.kind, self.settings = card, kind, settings
        self.active = False
        self.closed = False
        self.written = []
        card.opened.append(self)

    def start(self):
        if self.settings.get('device') in self.card.dead:
            raise self.card.PortAudioError("Error starting stream: Unanticipated host error "
                                           "[PaErrorCode -9999]: 'WdmSyncIoctl: DeviceIoControl "
                                           "GLE = 0x0000048F'")
        self.active = True
        callback = self.settings.get('callback')
        for block, status in self.card.feed if callback else ():
            callback(block, len(block) // (2 * self.settings['channels']), None, status)

    def stop(self):
        self.active = False

    def close(self):
        self.closed = True
        self.active = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
        self.close()
        return False

    def write(self, data):
        self.written.append(data)
        return False



class FakeCard(types.ModuleType):
    """sounddevice on the Windows 11 machine of _HostApis, whose only input is on
    WDM-KS and which has no default input; *refuse* is the rates no device opens at.
    WDM-KS opens no stream without a callback, as PortAudio's does not. A callback
    stream is handed *feed*, (bytes, status) pairs, when it starts. The devices in
    *dead* open and do not start, as a Bluetooth headset that is not connected."""

    class PortAudioError(Exception):
        pass

    def __init__(self, default_input=None, refuse=()):
        super().__init__("sounddevice")
        self.devices = [{"name": name, "hostapi": api, "max_input_channels": inputs,
                         "max_output_channels": outputs, "default_samplerate": 44100.0,
                         "default_low_input_latency": 0.01, "default_low_output_latency": 0.01,
                         "default_high_input_latency": 0.1, "default_high_output_latency": 0.1}
                        for api, inputs, outputs, name in _HostApis._devices]
        self.apis = [{"name": name, "devices": [i for i, d in enumerate(self.devices)
                                                if d["hostapi"] == n],
                      "default_input_device": -1, "default_output_device": -1}
                     for n, name in enumerate(_HostApis._apis)]
        self.default_input = default_input
        self.refuse = set(refuse)
        self.opened = []
        self.played = []
        self.feed = []
        self.dead = set()
        self.recs = []
        card = self

        def opener(kind):
            def open_(**settings):
                device = settings.get('device')
                if device is not None and _HostApis._apis[card.devices[device]["hostapi"]] \
                        == "Windows WDM-KS" and settings.get('callback') is None:
                    raise card.PortAudioError("Error opening RawInputStream: Unanticipated host "
                                              "error [PaErrorCode -9999]: 'Blocking API not "
                                              "supported yet'")
                if kind == 'RawInputStream':
                    if card.devices[device]["max_input_channels"] < settings['channels'] \
                            or settings['samplerate'] in card.refuse:
                        raise card.PortAudioError("Invalid sample rate")
                return _Stream(card, kind, **settings)
            return open_

        self.RawInputStream = opener('RawInputStream')
        self.RawOutputStream = opener('RawOutputStream')
        self.OutputStream = opener('OutputStream')

    def query_devices(self, device=None, kind=None):
        if kind == 'input':
            if self.default_input is None:
                raise self.PortAudioError("Error querying device -1")
            device = self.default_input
        if device is None:
            return [dict(d, index=i) for i, d in enumerate(self.devices)]
        if not 0 <= device < len(self.devices):
            raise self.PortAudioError(f"Error querying device {device}")
        return dict(self.devices[device], index=device)

    def query_hostapis(self, index=None):
        return self.apis if index is None else self.apis[index]

    def play(self, data, samplerate):
        self.played.append((data, samplerate))

    def rec(self, frames, samplerate, channels, dtype, device):
        if device is None and self.default_input is None or device in self.dead:
            raise self.PortAudioError(f"Error starting stream on device {device}")
        if samplerate in self.refuse:
            raise self.PortAudioError("Error opening InputStream: Invalid device [PaErrorCode -9996]")
        self.recs.append((device, samplerate, channels, dtype))
        return np.zeros((frames, channels), dtype)

    def wait(self):
        pass


@pytest.fixture
def card(monkeypatch):
    fake = FakeCard()
    monkeypatch.setitem(sys.modules, "sounddevice", fake)
    monkeypatch.setitem(sys.modules, "pyaudio", None)        # nothing reaches for it
    return fake


def test_check_device_numbers_the_devices_as_list_devices_does(card):
    """The index CheckDevice gives is the one record() and the streams take."""
    with kudio.CheckDevice() as check:
        listed = [(d["index"], d["Device"], d["api"], d["in"], d["out"])
                  for d in check.system_devices()]
    assert listed == [(d["index"], d["name"], d["api"], d["max_input_channels"],
                       d["max_output_channels"]) for d in kudio.list_devices()]
    assert [d["index"] for d in kudio.list_devices("input")] == [2]
    assert kudio.list_devices("input")[0]["api"] == "Windows WDM-KS"


def test_the_last_input_that_opens_is_the_one_taken(card):
    """Probed with a callback, which is how WDM-KS opens a microphone at all."""
    card.devices[1]["max_input_channels"] = 1                  # an MME input as well
    with kudio.CheckDevice() as check:
        assert check.input_device() == (2, 16000, 1)
    card.refuse = {16000}
    with kudio.CheckDevice() as check:
        assert check.input_device(1) == (1, 22050, 1)
        assert check.check_available_rate(2) == [r for r in check.standard_rate if r != 16000]


def test_the_default_input_is_asked_of_portaudio(card):
    """What code built on CheckDevice asks for its default input: OSError where the
    system has none."""
    with kudio.CheckDevice() as check:
        with pytest.raises(OSError, match="No Default Input Device Available"):
            check.p.get_default_input_device_info()
    card.default_input = 2
    with kudio.CheckDevice() as check:
        info = check.p.get_default_input_device_info()
    assert (info["index"], info["name"], info["maxInputChannels"]) == (2, "Microphone (USB Audio)", 2)


def test_a_stream_recorder_opens_the_device_it_is_given(card):
    """StreamRecorder by an index from CheckDevice: the device opened is that one,
    float32, and each chunk is a copy of its own."""
    with kudio.CheckDevice() as check:
        index = next(d["index"] for d in check.system_devices() if d["in"])
    rec = kudio.StreamRecorder(device=index, sr=16000, frame_size=256).start()
    stream = [s for s in card.opened if s.kind == 'RawInputStream'][-1]
    assert stream.settings == dict(samplerate=16000, channels=1, dtype='float32', device=index,
                                   blocksize=256, callback=rec._reader._callback)
    assert stream.active
    chunk = bytearray(np.linspace(-0.5, 0.5, 256, dtype=np.float32).tobytes())
    rec._reader._callback(chunk, 256, None, None)
    chunk[:] = bytes(len(chunk))                           # PortAudio reuses its buffer
    y = rec.stop()
    assert y.dtype == np.float32 and np.array_equal(y, np.linspace(-0.5, 0.5, 256, dtype=np.float32))
    assert stream.closed and not stream.active


def test_monitoring_echoes_the_input_to_an_output_stream(card):
    reader = kudio.LocalStreamReader(device=2, rate=16000, frame_size=128, wav_format=kudio.PA_INT16)
    reader.play_audio()
    reader.stream_start()
    out = [s for s in card.opened if s.kind == 'RawOutputStream'][-1]
    assert out.settings == dict(samplerate=16000, channels=1, dtype='int16', blocksize=128)
    reader._callback(np.arange(128, dtype=np.int16).tobytes(), 128, None, None)
    assert out.written == [np.arange(128, dtype=np.int16).tobytes()]
    reader.terminate()
    assert out.closed and all(s.closed for s in card.opened if s.kind == 'RawInputStream'
                              and s.settings.get('callback'))


def test_a_device_that_opens_and_will_not_start_is_closed_and_named(card, monkeypatch):
    """Windows lists a Bluetooth headset that is not connected; it opens, and
    starting it is a host error."""
    def dead(self):
        raise card.PortAudioError("Error starting stream: Unanticipated host error "
                                  "[PaErrorCode -9999]: 'WdmSyncIoctl: DeviceIoControl "
                                  "GLE = 0x0000048F'")

    rec = kudio.StreamRecorder(device=2, sr=16000)
    monkeypatch.setattr(_Stream, "start", dead)
    with pytest.raises(kudio.DeviceError, match=r"device 2 opened and did not start: .*0x0000048F"):
        rec.start()
    stream = [s for s in card.opened if s.kind == 'RawInputStream'][-1]
    assert stream.settings.get('callback') and stream.closed
    assert not rec.recording and rec._reader.rec_stream is None


def test_a_chunk_handed_on_is_its_own(card):
    """PortAudio fills the same buffer for the next chunk; on_frame=chunks.append,
    as the guide has it, must keep what it was given."""
    chunks = []
    reader = kudio.LocalStreamReader(device=2, rate=16000, frame_size=4, on_frame=chunks.append)
    reader.stream_start()
    block = bytearray(np.array([1, 2, 3, 4], np.int16).tobytes())
    reader._callback(block, 4, None, None)
    block[:] = bytes(len(block))
    reader.terminate()
    assert np.array_equal(chunks[0], [1, 2, 3, 4])


def test_the_callback_returns_nothing():
    """sounddevice's callback hands nothing back."""
    reader = stream_module.LocalStreamReader.__new__(stream_module.LocalStreamReader)
    reader.audio_buffer, reader.on_frame = None, None
    reader.data_type = np.int16
    reader.isListening, reader.isBroadcasting = threading.Event(), threading.Event()
    reader.isListening.set()
    assert reader._callback(np.zeros(8, dtype=np.int16).tobytes(), 8, None, None) is None


def test_recorder_takes_whole_buffers_of_int16(card, caplog):
    """From a callback stream, which WDM-KS opens; what comes after the last buffer
    asked for is left out."""
    every = np.arange(4200, dtype=np.int16)
    late = types.SimpleNamespace(input_overflow=True)
    # blocks of 350 frames: the fifth runs past the 1600 asked for
    card.feed = [(every[i:i + 700].tobytes(), late if i == 700 else None)
                 for i in range(0, 4200, 700)]
    with kudio.Recorder(device=2, record_times=0.1, rate=16000, channels=2,
                        frames_per_buffer=400) as r:
        with caplog.at_level(logging.WARNING, logger="kudio"):
            y = r.start_record()
        stream = [s for s in card.opened if s.kind == 'RawInputStream'][-1]
        assert stream.settings["callback"] is not None
        assert {k: v for k, v in stream.settings.items() if k != "callback"} == dict(
            samplerate=16000, channels=2, dtype='int16', device=2, blocksize=400)
    # int(16000 / 400 * 0.1) = 4 buffers of 400 frames, two channels
    assert y.dtype == np.int16 and y.shape == (1600, 2)
    assert np.array_equal(y.ravel(), every[:3200])
    assert "overflow" in caplog.text
    assert stream.closed


def test_recorder_says_what_the_device_did_not_give(card, monkeypatch):
    """A device that stops delivering is an error after ten seconds more than the
    take, not a wait for ever."""
    card.feed = [(np.zeros(800, np.int16).tobytes(), None)]
    monkeypatch.setattr(threading.Event, "wait", lambda self, timeout=None: self.is_set())
    with kudio.Recorder(device=2, record_times=0.1, rate=16000, channels=2,
                        frames_per_buffer=400) as r:
        with pytest.raises(kudio.DeviceError, match="the device gave 400 of 1600 frames"):
            r.start_record()


@pytest.mark.parametrize("why", ["refused", "dead"])
def test_recorder_names_the_device_and_the_rate_it_does_not_record_at(card, why):
    """As record() says it: a rate the device does not take, or a device that opens
    and does not start, is a DeviceError with the device and the rate, the error of
    sounddevice as its cause, and no stream left open."""
    if why == "refused":
        card.refuse = {16000}
    else:
        card.dead = {2}
    with kudio.Recorder(device=2, record_times=0.1, rate=16000) as r:
        with pytest.raises(kudio.DeviceError, match=r"^Recorder: device 2 did not record at "
                                                    r"16000 Hz: .+; `kudio devices` lists the "
                                                    r"devices, and device= picks one$") as caught:
            r.start_record()
        assert isinstance(caught.value.__cause__, card.PortAudioError)
        assert r.stream is None
    assert all(s.closed for s in card.opened if s.kind == 'RawInputStream')


@pytest.mark.parametrize("width", [1, 2, 3, 4])
def test_an_open_wave_file_plays_to_its_end(card, tmp_path, width):
    """Every sample width the wave module writes, as integers of that width; 24 bits
    in the top of 32."""
    path = tmp_path / "x.wav"
    rng = np.random.default_rng(width)
    lo, hi = (0, 256) if width == 1 else (-(1 << (8 * width - 1)), 1 << (8 * width - 1))
    samples = rng.integers(lo, hi, size=(2500, 2), dtype=np.int64)
    raw = b"".join(int(v).to_bytes(width, "little", signed=width > 1)
                   for v in samples.ravel())
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(width)
        w.setframerate(22050)
        w.writeframes(raw)
    with wave.open(str(path), "rb") as w:
        assert kudio.play_audio(w) is None
    out = [s for s in card.opened if s.kind == 'OutputStream'][-1]
    assert out.settings == dict(samplerate=22050, channels=2,
                                dtype={1: 'uint8', 2: 'int16', 3: 'int32', 4: 'int32'}[width])
    played = np.concatenate(out.written)
    want = samples << 8 if width == 3 else samples
    assert played.shape == (2500, 2) and np.array_equal(played.astype(np.int64), want)
    assert out.closed


def test_an_array_is_queued_and_a_failure_is_said(card, monkeypatch):
    y = np.zeros(160, np.float32)
    assert kudio.play_audio(y, 16000) is None and card.played[-1][1] == 16000

    def refuse(data, samplerate):
        raise card.PortAudioError("Error opening OutputStream: Invalid device")

    monkeypatch.setattr(card, "play", refuse)
    assert kudio.play_audio(y, 16000) == "can't be played: Error opening OutputStream: Invalid device"


def test_a_remote_stream_plays_on_an_output_stream(card):
    reader = kudio.RemoteStreamReader(None)
    reader.initialize_params(frame_size=512, wav_format='32bit')
    reader.play_remote_audio(rate=8000, channels=2)
    out = [s for s in card.opened if s.kind == 'RawOutputStream'][-1]
    assert out.settings == dict(samplerate=8000, channels=2, dtype='float32', blocksize=512)
    assert reader.isPlaying.is_set()
    reader.terminate()
    assert out.closed and not reader.isPlaying.is_set()


# -- the input taken when none is given: one rule everywhere ------------------------
def _two_inputs(card):
    card.devices[1]["max_input_channels"] = 1                  # an MME input too
    return card


def test_a_default_input_is_the_one_taken(card, caplog):
    _two_inputs(card).default_input = 1
    with caplog.at_level(logging.INFO, logger="kudio"):
        with kudio.CheckDevice() as check:
            assert check.input_device() == (1, 16000, 1)
    assert "No device given: input 1, Monitor" in caplog.text
    rec = kudio.StreamRecorder(sr=16000)
    assert rec._reader.device_ == 1
    kudio.record(0.1)
    assert card.recs[-1] == (1, 16000, 1, "float32")
    with kudio.Recorder(record_times=0) as r:
        r.start_record()
    card.feed = [(np.zeros(1024, np.int16).tobytes(), None)] * 20
    with kudio.Recorder(record_times=0.1, frames_per_buffer=512) as r:
        r.start_record()
    assert [s for s in card.opened if s.kind == 'RawInputStream'][-1].settings['device'] == 1


def test_a_default_input_that_does_not_start_is_an_error_and_not_another(card):
    """Another microphone in its place would record what nobody chose."""
    _two_inputs(card).default_input = 1
    card.dead = {1}
    for call in (lambda: kudio.CheckDevice().input_device(), lambda: kudio.record(0.1),
                 lambda: kudio.Recorder(record_times=0.1).start_record()):
        with pytest.raises(kudio.DeviceError, match=r"the system's default input, device 1, "
                                                    r"does not start: .*kudio devices"):
            call()
    with pytest.raises(kudio.DeviceError, match="no usable input device"):
        kudio.StreamRecorder(sr=16000)


def test_without_a_default_the_last_input_that_starts_is_taken(card, caplog):
    """A headset Windows lists while it is not connected opens and does not start:
    it is passed over, from the end of the list back."""
    _two_inputs(card).dead = {2}
    with caplog.at_level(logging.INFO, logger="kudio"):
        with kudio.CheckDevice() as check:
            assert check.input_device() == (1, 16000, 1)
    assert "No device given: input 1, Monitor (the system names no default input)" in caplog.text
    started = [s for s in card.opened if s.kind == 'RawInputStream' and s.settings['device'] == 1]
    assert started and all(s.closed for s in started)
    assert kudio.record(0.1).shape == (1600,) and card.recs[-1][0] == 1


def test_no_input_that_starts_is_a_device_error(card):
    _two_inputs(card).dead = {1, 2}
    with pytest.raises(kudio.DeviceError, match="No microphone found"):
        kudio.CheckDevice().input_device()
    with pytest.raises(kudio.DeviceError, match="No microphone found"):
        kudio.record(0.1)


def test_a_device_that_does_not_record_is_named_with_where_to_look(card):
    card.dead = {2}
    with pytest.raises(kudio.DeviceError, match=r"^record\(\): device 2 did not record at 16000 Hz: .*"
                                                r"`kudio devices` lists the devices, and device= "
                                                r"picks one") as caught:
        kudio.record(0.1, device=2)
    assert isinstance(caught.value.__cause__, card.PortAudioError)


def test_a_rate_the_device_does_not_take_is_named():
    """PortAudio calls a rate a device does not take an invalid device, which sends one
    looking at the wrong thing."""
    card = FakeCard()
    card.refuse = {16000}
    import sys as _sys
    old = _sys.modules.get("sounddevice")
    _sys.modules["sounddevice"] = card
    try:
        with pytest.raises(kudio.DeviceError, match=r"device 2 did not record at 16000 Hz: "
                                                    r"Error opening InputStream: Invalid device"):
            kudio.record(0.1, device=2)
        assert kudio.record(0.1, sr=44100, device=2).shape == (4410,)
    finally:
        if old is None:
            del _sys.modules["sounddevice"]
        else:
            _sys.modules["sounddevice"] = old
