# -*- coding: utf-8 -*-
"""StreamRecorder.

PyAudio is deliberately absent from this project's venv, so the device layer is
replaced by a fake reader that hands frames over on demand. What is under test
is everything StreamRecorder owns: accumulation without loss, the level and
elapsed readings, the rolling tail, the memory cap, and the channel layout of
the finished take.
"""
import threading

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


def test_a_raising_callback_does_not_kill_the_audio_thread(monkeypatch):
    """PortAudio's callback runs on its own thread; an exception there would
    just stop the stream delivering, with nothing said."""
    reader = stream_module.LocalStreamReader.__new__(stream_module.LocalStreamReader)
    reader.audio_buffer = None
    reader.data_type = np.float32
    reader.isListening = threading.Event()
    reader.isListening.set()
    reader.isBroadcasting = threading.Event()
    reader.on_frame = lambda data: (_ for _ in ()).throw(RuntimeError("boom"))

    fake_pyaudio = type("M", (), {"paContinue": 0})
    monkeypatch.setitem(__import__("sys").modules, "pyaudio", fake_pyaudio)
    data, flag = reader._callback(np.zeros(8, dtype=np.float32).tobytes(),
                                  8, None, None)
    assert flag == 0
