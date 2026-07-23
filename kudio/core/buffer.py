# -*- coding: utf-8 -*-
"""Thread-safe frame buffers shared between audio producers and consumers."""
from __future__ import annotations

import logging
import threading
from collections import deque
from typing import List, Optional

import numpy as np

__all__ = ['AudioBuffer', 'FetchBuffer']

log = logging.getLogger(__name__)


class AudioBuffer:
    """Bounded FIFO of audio frames shared between producer/consumer threads.

    A stream reader calls :meth:`add` for every incoming frame; consumers either
    poll frames with :meth:`get` (e.g. for real-time plotting) or collect a fixed
    number of consecutive frames with :meth:`get_fixed_size_data` (e.g. timed
    recording). All methods are thread-safe.
    """

    def __init__(self, size: int = 2):
        if size < 1:
            raise ValueError(f"size must be >= 1, got {size}")
        self.buffer_size = int(size)
        self._frames: deque = deque()
        self._cond = threading.Condition()
        # fixed-size capture state (timed recording)
        self._capture: Optional[List[np.ndarray]] = None
        self._capture_left = 0
        self._capture_done = threading.Event()

    def add(self, data, drop_full: bool = True) -> None:
        """Append one frame.

        With ``drop_full=True`` (default) the oldest frame is dropped when the
        buffer is full; otherwise the call blocks until a slot is free.
        """
        with self._cond:
            # feed an active fixed-size capture first so timed recordings
            # do not lose frames to a slow consumer
            if self._capture is not None and self._capture_left > 0:
                self._capture.append(data)
                self._capture_left -= 1
                if self._capture_left == 0:
                    self._capture_done.set()
            if drop_full:
                if len(self._frames) >= self.buffer_size:
                    self._frames.popleft()
            else:
                while len(self._frames) >= self.buffer_size:
                    self._cond.wait()
            self._frames.append(data)
            self._cond.notify_all()

    def get(self, timeout: Optional[float] = None):
        """Pop the oldest frame, blocking until one is available.

        Raises ``TimeoutError`` if *timeout* (seconds) elapses first.
        """
        with self._cond:
            while not self._frames:
                if not self._cond.wait(timeout=timeout):
                    raise TimeoutError("AudioBuffer.get timed out")
            data = self._frames.popleft()
            self._cond.notify_all()
            return data

    def get_fixed_size_data(self, queue_size: int) -> np.ndarray:
        """Block until *queue_size* frames have arrived, then return them
        concatenated with ``np.hstack``."""
        with self._cond:
            self._capture = []
            self._capture_left = int(queue_size)
            self._capture_done.clear()
        log.info("Buffer: capturing %d frames", queue_size)
        self._capture_done.wait()
        with self._cond:
            frames, self._capture = self._capture, None
        return np.hstack(frames)

    def clear(self) -> bool:
        """Drop all buffered frames. Returns True if anything was dropped."""
        with self._cond:
            had_data = bool(self._frames)
            self._frames.clear()
            self._cond.notify_all()
        if had_data:
            log.info("Buffer: cleared")
        return had_data

    def size(self) -> int:
        return len(self._frames)

    def max_size(self) -> int:
        return self.buffer_size

    def is_full(self) -> bool:
        return len(self._frames) >= self.buffer_size

    def is_empty(self) -> bool:
        return not self._frames


class FetchBuffer(threading.Thread):
    """Collect a fixed number of frames from an :class:`AudioBuffer` in a
    background thread.

    >>> data = FetchBuffer(buffer).get_data(desired_queue_size=10)
    """

    def __init__(self, buffer: AudioBuffer):
        super().__init__(daemon=True)
        if not isinstance(buffer, AudioBuffer):
            raise TypeError("buffer must be an AudioBuffer")
        self.buffer = buffer
        self.queue_size: Optional[int] = None
        self.data: Optional[np.ndarray] = None

    def get_data(self, desired_queue_size: int, timeout: Optional[float] = None):
        self.queue_size = desired_queue_size
        self.start()
        return self.join(timeout)

    def run(self) -> None:
        if isinstance(self.queue_size, int):
            self.data = self.buffer.get_fixed_size_data(self.queue_size)

    def join(self, timeout: Optional[float] = None):
        super().join(timeout)
        return self.data
