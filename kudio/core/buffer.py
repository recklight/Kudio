# -*- coding: utf-8 -*-
"""Thread-safe frame buffers shared between audio producers and consumers."""
from __future__ import annotations

import logging
import operator
import threading
from collections import deque
from typing import List, Optional

import numpy as np

__all__ = ['AudioBuffer', 'FetchBuffer']

log = logging.getLogger(__name__)


def _checked_frame_count(value, caller: str) -> int:
    """Validate a frame count before anything starts waiting for it.

    Callers rarely type the number: it comes out of arithmetic like
    ``int(seconds * sr / chunk)``, which rounds down to zero as soon as the
    request is shorter than one chunk. Waiting for zero frames can never
    finish, and a silent hang in a capture gives nobody anything to act on.
    """
    whole = None
    if not isinstance(value, (bool, str, bytes)):
        try:
            whole = operator.index(value)              # int, numpy integers
        except TypeError:
            # a float only when it is whole: int() used to truncate 2.5 to 2,
            # and float('inf') -- arithmetic on a zero chunk size -- is not
            if isinstance(value, (float, np.floating)) and np.isfinite(value) \
                    and float(value).is_integer():
                whole = int(value)
    if whole is None:
        raise TypeError(
            "{0} needs a whole number of frames, got {1!r}.".format(caller, value))
    count = whole
    if count < 1:
        raise ValueError(
            "{0} needs at least 1 frame, got {1}. A frame count computed from a "
            "duration rounds down to zero when the duration is shorter than one "
            "chunk, so check the duration and the chunk size.".format(caller, count)
        )
    return count


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

    def add(self, data: np.ndarray, drop_full: bool = True) -> None:
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

    def get(self, timeout: Optional[float] = None) -> np.ndarray:
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

    def get_fixed_size_data(self, queue_size: int,
                            timeout: Optional[float] = None) -> np.ndarray:
        """Block until *queue_size* frames have arrived, then return them
        concatenated with ``np.hstack``.

        Raises ``ValueError`` when *queue_size* is below 1, because that wait
        would never end, and ``TypeError`` when it is not a whole number. A
        *timeout* that expires raises ``TimeoutError`` and ends the capture,
        so nothing is left waiting on the buffer to take the next one's
        frames. One capture runs at a time: a second one started while the
        first waits raises ``RuntimeError``, since the two would share one
        set of frames.
        """
        count = _checked_frame_count(queue_size, "AudioBuffer.get_fixed_size_data")
        with self._cond:
            if self._capture is not None:
                raise RuntimeError("AudioBuffer: a capture is already running on "
                                   "this buffer; wait for it, or capture from "
                                   "another AudioBuffer")
            self._capture = []
            self._capture_left = count
            self._capture_done.clear()
        log.info("Buffer: capturing %d frames", count)
        try:
            finished = self._capture_done.wait(timeout)
        finally:
            # also when the wait is interrupted (Ctrl-C in a notebook), or the
            # buffer would refuse every capture after this one
            with self._cond:
                frames, self._capture = self._capture, None
                self._capture_left = 0
        if not finished:
            raise TimeoutError(
                "AudioBuffer.get_fixed_size_data timed out after {0} s with {1} "
                "of {2} frames".format(timeout, len(frames or ()), count))
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

    One instance runs one capture, because a thread cannot be started twice.
    """

    def __init__(self, buffer: AudioBuffer):
        super().__init__(daemon=True)
        if not isinstance(buffer, AudioBuffer):
            raise TypeError("buffer must be an AudioBuffer")
        self.buffer = buffer
        self.queue_size: Optional[int] = None
        self.data: Optional[np.ndarray] = None
        self._timeout: Optional[float] = None
        self._error: Optional[BaseException] = None

    def get_data(self, desired_queue_size: int, timeout: Optional[float] = None) -> np.ndarray:
        """Run the capture and return the frames.

        Always returns an array; a *timeout* that expires raises
        ``TimeoutError`` instead, so the caller can tell a stream that stopped
        delivering from a capture that came back short.
        """
        if self.queue_size is not None:
            raise RuntimeError(
                "FetchBuffer.get_data runs once per instance and this one has "
                "already run. Build a new FetchBuffer for the next capture."
            )
        self.queue_size = _checked_frame_count(
            desired_queue_size, "FetchBuffer.get_data"
        )
        self._timeout = timeout
        self.start()
        # the capture ends itself at the timeout, so this join returns; the
        # thread used to be left waiting on the buffer and took the next
        # capture's frames
        data = self.join()
        if isinstance(self._error, TimeoutError):
            raise TimeoutError(
                "FetchBuffer.get_data timed out after {0} s waiting for {1} "
                "frames. Check that the stream is running, or allow more "
                "time.".format(timeout, self.queue_size))
        if data is None:
            raise RuntimeError(
                "FetchBuffer.get_data collected nothing for {0} frames. The "
                "capture thread ended early; its traceback is on "
                "stderr.".format(self.queue_size)
            )
        return data

    def run(self) -> None:
        if isinstance(self.queue_size, int):
            try:
                self.data = self.buffer.get_fixed_size_data(
                    self.queue_size, timeout=self._timeout)
            except TimeoutError as e:
                self._error = e

    def join(self, timeout: Optional[float] = None) -> Optional[np.ndarray]:
        super().join(timeout)
        return self.data
