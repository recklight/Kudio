# -*- coding: utf-8 -*-
import threading
import time

import numpy as np
import pytest

from kudio import AudioBuffer, FetchBuffer


def test_add_get_fifo():
    buf = AudioBuffer(size=4)
    for i in range(3):
        buf.add(np.full(4, i))
    assert buf.size() == 3
    np.testing.assert_array_equal(buf.get(), np.full(4, 0))
    np.testing.assert_array_equal(buf.get(), np.full(4, 1))
    assert buf.size() == 1


def test_drop_oldest_when_full():
    buf = AudioBuffer(size=2)
    for i in range(5):
        buf.add(np.full(2, i))
    assert buf.size() == 2
    assert buf.is_full()
    np.testing.assert_array_equal(buf.get(), np.full(2, 3))
    np.testing.assert_array_equal(buf.get(), np.full(2, 4))


def test_get_timeout():
    buf = AudioBuffer(size=2)
    with pytest.raises(TimeoutError):
        buf.get(timeout=0.05)


def test_blocking_get_receives_from_producer():
    buf = AudioBuffer(size=2)

    def producer():
        time.sleep(0.05)
        buf.add(np.array([42]))

    threading.Thread(target=producer, daemon=True).start()
    np.testing.assert_array_equal(buf.get(timeout=2), np.array([42]))


def test_clear():
    buf = AudioBuffer(size=3)
    assert buf.clear() is False
    buf.add(np.array([1]))
    assert buf.clear() is True
    assert buf.is_empty()


def test_get_fixed_size_data():
    buf = AudioBuffer(size=2)
    n_frames = 5

    def producer():
        for i in range(n_frames + 3):
            buf.add(np.full(10, i))
            time.sleep(0.005)

    threading.Thread(target=producer, daemon=True).start()
    data = buf.get_fixed_size_data(n_frames)
    assert data.shape == (n_frames * 10,)


def test_fetch_buffer_thread():
    buf = AudioBuffer(size=2)

    def producer():
        for i in range(4):
            buf.add(np.full(8, i))
            time.sleep(0.005)

    threading.Thread(target=producer, daemon=True).start()
    data = FetchBuffer(buf).get_data(3)
    assert data.shape == (24,)


def _call_with_deadline(fn, *args, **kwargs):
    # A frame count below 1 used to block the caller forever. Running the call
    # on a throwaway thread turns that into a thread that is still alive after
    # the deadline, so the test reports a hang instead of becoming one.
    outcome = {}

    def target():
        try:
            outcome["value"] = fn(*args, **kwargs)
        except BaseException as exc:
            outcome["error"] = exc

    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    worker.join(3)
    return worker, outcome


@pytest.mark.parametrize("queue_size", [0, -5])
def test_fixed_size_data_refuses_non_positive_count(queue_size):
    buf = AudioBuffer(size=2)
    buf.add(np.full(4, 1.0))

    worker, outcome = _call_with_deadline(buf.get_fixed_size_data, queue_size)

    assert not worker.is_alive(), "get_fixed_size_data is still waiting for frames"
    assert isinstance(outcome.get("error"), ValueError)
    assert "at least 1 frame" in str(outcome["error"])


@pytest.mark.parametrize("queue_size", [0, -5])
def test_fetch_buffer_refuses_non_positive_count(queue_size):
    buf = AudioBuffer(size=2)

    def producer():
        for i in range(20):
            buf.add(np.full(8, i))
            time.sleep(0.005)

    threading.Thread(target=producer, daemon=True).start()
    worker, outcome = _call_with_deadline(FetchBuffer(buf).get_data, queue_size)

    assert not worker.is_alive(), "get_data is still waiting for frames"
    assert isinstance(outcome.get("error"), ValueError)
    assert "at least 1 frame" in str(outcome["error"])


def test_fetch_buffer_timeout_is_an_error_not_a_none():
    buf = AudioBuffer(size=2)
    # nothing feeds the buffer, so the capture can only run out of time
    with pytest.raises(TimeoutError, match="timed out"):
        FetchBuffer(buf).get_data(5, timeout=0.1)


def test_fetch_buffer_says_why_a_second_capture_is_refused():
    buf = AudioBuffer(size=2)
    fetch = FetchBuffer(buf)
    with pytest.raises(TimeoutError):
        fetch.get_data(5, timeout=0.1)
    with pytest.raises(RuntimeError, match="once per instance"):
        fetch.get_data(5, timeout=0.1)


# -- frame counts and timeouts -------------------------------------------------

@pytest.mark.parametrize("count", [2.5, "3", True])
def test_a_frame_count_that_is_not_whole_is_refused(count):
    """int() used to truncate 2.5 to 2 and accept the string '3'."""
    with pytest.raises(TypeError, match="whole number"):
        AudioBuffer(size=2).get_fixed_size_data(count, timeout=0.01)


def test_a_timed_out_capture_leaves_nothing_behind():
    """The timed-out thread used to stay waiting on the buffer and take the
    frames of the next capture."""
    buf = AudioBuffer(size=8)
    with pytest.raises(TimeoutError):
        FetchBuffer(buf).get_data(5, timeout=0.05)

    def producer():
        for i in range(3):
            time.sleep(0.01)
            buf.add(np.full(4, i))

    threading.Thread(target=producer, daemon=True).start()
    data = FetchBuffer(buf).get_data(3, timeout=2)
    np.testing.assert_array_equal(data, np.repeat([0, 1, 2], 4))
