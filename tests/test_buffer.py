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
