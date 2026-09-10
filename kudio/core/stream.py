# -*- coding: utf-8 -*-
"""Audio streaming: playback, local recording, and remote (TCP) streams.

PyAudio is an optional dependency (``pip install kudio[audio]``); it is only
imported when a stream/recorder is actually created, so the package can be
used for offline processing without any audio hardware.
"""
from __future__ import annotations

import logging
import socket
import threading
import wave
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

from kudio.core.buffer import AudioBuffer
from kudio.core.io import check_file
from kudio.util.check import CheckDevice

__all__ = [
    'wave_decode',
    'wave_encode',
    'play_audio',
    'record',
    'Recorder',
    'RemoteStreamReader',
    'LocalStreamReader',
    'StreamRecorder',
]

log = logging.getLogger(__name__)


def record(seconds: float, sr: int = 16000, channels: int = 1,
           device=None) -> np.ndarray:
    """Blocking record for *seconds* via sounddevice (modern backend).

    Returns float32 samples, shape ``(n,)`` for mono else ``(n, channels)``.
    Requires ``kudio[audio]``.

    *device* is a **sounddevice** index — get one from
    :func:`kudio.list_devices`, not from :class:`kudio.CheckDevice`, which
    numbers devices through PyAudio for the streaming classes.
    """
    try:
        import sounddevice as sd
    except ImportError as e:
        from kudio.exceptions import DependencyError
        raise DependencyError('sounddevice', extra='audio') from e
    frames = int(seconds * sr)
    data = sd.rec(frames, samplerate=sr, channels=channels,
                  dtype='float32', device=device)
    sd.wait()
    return data.squeeze()

# stable PortAudio format codes (== pyaudio.paFloat32 / pyaudio.paInt16)
PA_FLOAT32 = 1
PA_INT16 = 8


def _pyaudio():
    try:
        import pyaudio
        return pyaudio
    except ImportError as e:
        raise ImportError(
            "PyAudio is required for audio streaming: pip install kudio[audio]"
        ) from e


def wave_decode(in_data: bytes, channels: int) -> np.ndarray:
    """Raw int16 bytes -> array of shape ``(chunk_length, channels)``."""
    result = np.frombuffer(in_data, dtype=np.int16)
    chunk_length = len(result) // channels
    return np.reshape(result, (chunk_length, channels))


def wave_encode(signal: np.ndarray) -> bytes:
    """Multi-channel array -> interleaved int16 bytes."""
    return signal.flatten().astype(np.int16).tobytes()


def play_audio(wav, rate: Optional[int] = None, channels: int = 1):
    """Play audio from a file path, an opened ``wave.Wave_read``, or an ndarray.

    ndarray input must be int16 or float32; *rate* is required in that case.
    Returns an error message string on failure, else ``None``.
    """
    if isinstance(wav, np.ndarray):
        if wav.dtype not in (np.int16, np.float32):
            log.error("play_audio: unsupported dtype %s", wav.dtype)
            return f"unsupported dtype: {wav.dtype}"
        try:
            import sounddevice as sd
            sd.play(wav, rate)
            return None
        except Exception:
            pass  # fall back to pyaudio
        pyaudio = _pyaudio()
        pya = pyaudio.PyAudio()
        try:
            fmt = pyaudio.paInt16 if wav.dtype == np.int16 else pyaudio.paFloat32
            stream = pya.open(format=fmt, channels=channels, rate=rate, output=True)
            stream.write(wav.tobytes())
            stream.stop_stream()
            stream.close()
            return None
        except Exception as e:
            log.error("play_audio failed: %s", e)
            return "can't be played by pyaudio"
        finally:
            pya.terminate()

    if isinstance(wav, (str, Path)):
        if not Path(wav).is_file():
            return "input file can't be played"
        wav = wave.open(str(wav), "rb")

    if isinstance(wav, wave.Wave_read):
        pyaudio = _pyaudio()
        pya = pyaudio.PyAudio()
        try:
            stream = pya.open(format=pya.get_format_from_width(wav.getsampwidth()),
                              channels=wav.getnchannels(),
                              rate=wav.getframerate(),
                              output=True)
            data = wav.readframes(1024)
            while data != b'':
                stream.write(data)
                data = wav.readframes(1024)
            stream.stop_stream()
            stream.close()
            return None
        except Exception as e:
            log.error("play_audio failed: %s", e)
            return "can't be played"
        finally:
            pya.terminate()

    return "unsupported input"


class Recorder:
    """Blocking recorder for a fixed duration.

    Examples
    --------
    >>> with Recorder(device=11, record_times=3, channels=6, rate=16000) as r:
    ...     wave_data = r.start_record()
    ...     r.save('kudio_test.wav')
    """

    def __init__(self, device: Optional[int] = None, record_times: int = 3,
                 rate: int = 16000, channels: int = 1,
                 frames_per_buffer: int = 512):
        pyaudio = _pyaudio()
        self.format = pyaudio.paInt16
        self.input_device_index = device
        self.record_times = record_times
        self.rate = rate
        self.channels = channels
        self.frames_per_buffer = frames_per_buffer
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.frames: Optional[np.ndarray] = None  # interleaved int16 samples

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.terminate()

    def terminate(self) -> None:
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        self.p.terminate()

    close = terminate

    def start_record(self) -> np.ndarray:
        """Record for ``record_times`` seconds; returns the waveform, with one
        column per channel (squeezed for mono)."""
        self.stream = self.p.open(
            format=self.format,
            channels=self.channels,
            input_device_index=self.input_device_index,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.frames_per_buffer)

        log.info("Recording %d second(s)...", self.record_times)
        chunks = []
        n_reads = int(self.rate / self.frames_per_buffer * self.record_times)
        for _ in range(n_reads):
            rec = self.stream.read(self.frames_per_buffer)
            chunks.append(np.frombuffer(rec, dtype=np.int16))
        self.frames = np.concatenate(chunks) if chunks else np.empty(0, np.int16)
        per_channel = [self.frames[c::self.channels] for c in range(self.channels)]
        return np.array(per_channel).T.squeeze()

    def save(self, file_dir, rename: bool = False,
             resample: Optional[int] = None) -> bool:
        """Write the last recording to *file_dir* (16-bit PCM)."""
        if self.frames is None:
            log.error("Recorder: nothing recorded yet")
            return False
        file_dir = check_file(file_dir, rename=rename)
        try:
            data = self.frames
            rate = self.rate
            if resample is not None and resample != self.rate:
                import scipy.signal
                data = scipy.signal.resample(
                    data, resample * self.record_times * self.channels)
                data = np.clip(data, np.iinfo(np.int16).min,
                               np.iinfo(np.int16).max).astype(np.int16)
                rate = resample
            with wave.open(file_dir, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(2)  # int16
                wf.setframerate(rate)
                wf.writeframes(data.astype(np.int16).tobytes())
            log.info("Recorder: saved -> %s", file_dir)
            return True
        except Exception as e:
            log.error("Recorder: save failed: %s", e)
            return False

    write = save


class RemoteStreamReader(threading.Thread):
    """Receive an audio stream over TCP and feed it into an
    :class:`AudioBuffer` (optionally playing it back locally)."""

    def __init__(self, buffer: Optional[AudioBuffer]):
        super().__init__(daemon=True)
        self.audio_buffer = buffer
        self.frame_size_: Optional[int] = None
        self.receive_buffer_size_: Optional[int] = None

        self.p = None
        self.isReceiving = threading.Event()
        self.isPlaying = threading.Event()

        self.play_stream = None
        self.stream_dtype: Optional[int] = None
        self.data_type = None

        self.sk: Optional[socket.socket] = None
        self.break_count = 10

    def connect_remote_device(self, ip: str, port: int) -> bool:
        try:
            self.sk = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sk.connect((ip, port))
            self.sk.settimeout(5)
            log.info("Stream: connect success")
            return True
        except Exception as e:
            log.error("Stream: %s", e)
            return False

    def sock_send(self, input_data: str, recv_buffer_size: int):
        """Send a text command; returns decoded waveform when the reply is
        binary audio data."""
        if input_data == '':
            log.error("Stream: input data error")
            return None
        self.sk.send(input_data.encode("utf-8"))
        data = self.sk.recv(recv_buffer_size)
        try:
            log.info("Stream: %s", data.decode('utf-8'))
            return None
        except UnicodeDecodeError:
            while len(data) < recv_buffer_size:
                data += self.sk.recv(recv_buffer_size)
            waves = np.frombuffer(data, dtype=np.int16)
            log.info("Stream: wave shape %s", waves.shape)
            return waves

    def initialize_params(self, frame_size: int = 2048,
                          wav_format: str = '16bit') -> None:
        _format = PA_FLOAT32 if wav_format.lower() == '32bit' else PA_INT16
        pyaudio = _pyaudio()
        self.p = pyaudio.PyAudio()
        self.frame_size_ = frame_size
        self.receive_buffer_size_ = frame_size * (2 if _format == PA_INT16 else 4)
        self.stream_dtype = _format
        self.data_type = np.int16 if _format == PA_INT16 else np.float32

    def play_remote_audio(self, rate: int, channels: int = 1) -> None:
        self.play_stream = self.p.open(format=self.stream_dtype,
                                       channels=channels,
                                       rate=rate,
                                       output=True,
                                       frames_per_buffer=self.frame_size_)
        if self.play_stream.is_active():
            self.isPlaying.set()
            log.info("Stream: play stream opened")

    def run(self) -> None:
        self.isReceiving.set()
        while self.isReceiving.is_set():
            try:
                data = self.sk.recv(self.receive_buffer_size_)
                cnt = 0
                while len(data) < self.receive_buffer_size_:
                    cnt += 1
                    data += self.sk.recv(self.receive_buffer_size_)
                    if cnt > self.break_count:
                        log.warning("Stream: auto break by break_count")
                        self.terminate()

                if len(data) != self.receive_buffer_size_:
                    log.warning("Stream: unexpected data size (%d)", len(data))
                    continue
                if self.audio_buffer is not None:
                    self.audio_buffer.add(np.frombuffer(data, dtype=self.data_type))

                if self.isPlaying.is_set():
                    self.play_stream.write(data)
            except Exception:
                pass
        log.info("Stream: stopped receiving")

    def terminate(self) -> None:
        if self.sk is not None:
            try:
                self.sk.send(b'stop')
                log.info("Stream: sent stop command to server")
            except OSError:
                pass
            self.sk.close()
            self.sk = None
            log.info("Stream: closed local socket")
        if self.is_alive() or self.isReceiving.is_set():
            self.isReceiving.clear()
            log.info("Stream: receiving stopped")
        if self.play_stream is not None and self.play_stream.is_active():
            self.isPlaying.clear()
            self.play_stream.stop_stream()
            self.play_stream.close()
            self.play_stream = None
            log.info("Stream: play stream stopped")
        if self.p is not None:
            self.p.terminate()


class LocalStreamReader:
    """Stream from a local microphone into an :class:`AudioBuffer`, with
    optional live monitoring through the default output device.

    >>> lsr = LocalStreamReader(channels=1, frame_size=2048, wav_format=PA_INT16)
    >>> if lsr.get_status():
    ...     lsr.play_audio()     # optional monitoring
    ...     lsr.stream_start()
    """

    def __init__(self,
                 device: Optional[int] = None,
                 rate: Optional[int] = None,
                 channels: int = 1,
                 frame_size: int = 2048,
                 wav_format: int = PA_INT16,
                 buffer: Optional[AudioBuffer] = None,
                 on_frame: Optional[Callable[[np.ndarray], None]] = None):
        """
        :param device: input device index (default: system default)
        :param rate: sample rate (default: device default)
        :param channels: channel count, default 1
        :param frame_size: frames per buffer, default 2048
        :param wav_format: ``PA_FLOAT32`` (1) or ``PA_INT16`` (8)
        :param buffer: target AudioBuffer (optional)
        :param on_frame: called with each captured chunk. An
            :class:`AudioBuffer` **drops** frames once it is full, which is
            correct for a monitor and wrong for a recording; a callback lets
            the caller decide. It runs on PortAudio's thread, so it must
            return promptly — append, do not process.
        """
        self.audio_buffer = buffer
        self.on_frame = on_frame
        self.device_: Optional[int] = None
        self.rate_: Optional[int] = None
        self.channels_: Optional[int] = None
        self.frame_size_: Optional[int] = None
        self.receive_buffer_size_: Optional[int] = None

        self.isInitial = threading.Event()
        self.isBroadcasting = threading.Event()
        self.isListening = threading.Event()
        self.play_stream = None
        self.rec_stream = None
        self.p = None

        self.set_stream(device, rate, channels, frame_size, wav_format)

    def set_stream(self, device: Optional[int] = None, rate: Optional[int] = None,
                   channels: int = 1, frame_size: int = 2048,
                   wav_format: int = PA_INT16) -> bool:
        pyaudio = _pyaudio()
        self.p = pyaudio.PyAudio()
        try:
            with CheckDevice() as check:
                self.device_, self.rate_, self.channels_ = \
                    check.input_device(device, rate, channels)
        except Exception as e:
            # no usable input device yet — stay uninitialized so the caller can
            # retry later (see get_status()) instead of crashing at construction
            log.warning("Stream: no input device available (%s)", e)
            self.isInitial.clear()
            return False

        self.frame_size_ = frame_size
        self.receive_buffer_size_ = frame_size * (2 if wav_format == PA_INT16 else 4)
        if wav_format in (PA_FLOAT32, PA_INT16):
            self.format = wav_format
            self.data_type = np.int16 if wav_format == PA_INT16 else np.float32
            self.isInitial.set()
            return True
        log.error("Stream: unsupported wav_format %r", wav_format)
        return False

    def get_status(self) -> bool:
        return self.isInitial.is_set()

    def get_setup(self):
        return (self.device_, self.rate_, self.channels_) if self.get_status() else None

    def play_audio(self) -> None:
        """Open a playback stream that echoes the microphone input."""
        self.play_stream = self.p.open(
            format=self.format,
            channels=self.channels_,
            rate=self.rate_,
            output=True,
            frames_per_buffer=self.frame_size_)
        if self.play_stream.is_active():
            self.isBroadcasting.set()
            log.info("Stream: broadcasting started")
        else:
            log.error("Stream: broadcasting failed")

    def _callback(self, in_data, frame_count, time_info, status):
        import pyaudio
        if self.isListening.is_set():
            if self.isBroadcasting.is_set():
                self.play_stream.write(in_data)
            data = np.frombuffer(in_data, dtype=self.data_type)
            if self.audio_buffer is not None:
                self.audio_buffer.add(data)
            if self.on_frame is not None:
                try:
                    self.on_frame(data)
                except Exception:
                    # an exception here would silently kill the audio thread
                    # and the stream would just stop delivering
                    log.exception("Stream: on_frame callback raised")
        return in_data, pyaudio.paContinue

    def stream_start(self) -> None:
        self.rec_stream = self.p.open(
            format=self.format,
            channels=self.channels_,
            input=True,
            input_device_index=self.device_,
            rate=self.rate_,
            frames_per_buffer=self.frame_size_,
            stream_callback=self._callback)
        self.rec_stream.start_stream()
        self.isListening.set()

    def terminate(self) -> None:
        self.isInitial.clear()
        self.isBroadcasting.clear()
        if self.play_stream is not None and self.play_stream.is_active():
            self.play_stream.stop_stream()
            self.play_stream.close()
            self.play_stream = None
            log.info("Stream: playing stream stopped")
        if self.rec_stream is not None and self.isListening.is_set():
            self.isListening.clear()
            self.rec_stream.stop_stream()
            self.rec_stream.close()
            self.rec_stream = None
            log.info("Stream: recording stream stopped")
        if self.p is not None:
            self.p.terminate()


class StreamRecorder:
    """Record for as long as you like, and watch the level while it happens.

    :func:`kudio.record` is one blocking call: you commit to a duration up
    front and nothing is observable until it returns. That is fine for a script
    and useless for a person at a microphone, who needs to see the level before
    committing and stop when the sentence ends.

    >>> rec = kudio.StreamRecorder(sr=16000)
    >>> rec.start()
    >>> rec.level_db()                     # doctest: +SKIP
    -23.4
    >>> y = rec.stop()                     # everything captured, float32

    Capture runs on PortAudio's own thread; :meth:`level`, :meth:`tail` and
    :meth:`elapsed` are safe to poll from a UI timer. Nothing is dropped —
    unlike :class:`AudioBuffer`, which is a monitor and discards when full.

    Requires ``kudio[audio]`` (PyAudio).
    """

    def __init__(self, device: Optional[int] = None, sr: Optional[int] = None,
                 channels: int = 1, frame_size: int = 1024,
                 max_seconds: Optional[float] = 3600.0,
                 monitor: bool = False):
        """
        :param device: **PyAudio** input index (:class:`kudio.CheckDevice`),
            not a sounddevice one — the two number devices independently.
        :param sr: sample rate; the device default when omitted.
        :param max_seconds: hard cap on the take, so a recorder left running
            cannot fill memory. ``None`` removes the cap.
        :param monitor: echo the input to the default output while recording.
        """
        self._chunks: List[np.ndarray] = []
        self._lock = threading.Lock()
        self._level = 0.0
        self._frames = 0
        self._drained = 0                     # chunks already handed to drain()
        self._capped = False
        self._channels = max(1, int(channels))
        self._max_seconds = max_seconds
        self._monitor = monitor

        self._reader = LocalStreamReader(
            device=device, rate=sr, channels=self._channels,
            frame_size=frame_size, wav_format=PA_FLOAT32,
            on_frame=self._on_frame)
        if not self._reader.get_status():
            from kudio.exceptions import DeviceError
            raise DeviceError("no usable input device for StreamRecorder")
        self._sr = int(self._reader.rate_ or sr or 16000)
        self._channels = int(self._reader.channels_ or self._channels)

    # ------------------------------------------------------------- properties

    @property
    def sr(self) -> int:
        """The rate actually negotiated with the device."""
        return self._sr

    @property
    def channels(self) -> int:
        return self._channels

    @property
    def recording(self) -> bool:
        return self._reader.isListening.is_set()

    @property
    def capped(self) -> bool:
        """True once ``max_seconds`` was reached and capture stopped growing."""
        return self._capped

    # ------------------------------------------------------------- capture

    def _on_frame(self, data: np.ndarray) -> None:
        peak = float(np.max(np.abs(data))) if data.size else 0.0
        with self._lock:
            self._level = peak
            if self._capped:
                return
            frames = data.size // self._channels
            if self._max_seconds is not None and \
                    (self._frames + frames) / self._sr > self._max_seconds:
                self._capped = True
                log.warning("StreamRecorder: hit max_seconds=%.0f, "
                            "capture stopped", self._max_seconds)
                return
            self._chunks.append(data.copy())
            self._frames += frames

    def start(self) -> "StreamRecorder":
        """Open the stream and begin capturing. Idempotent."""
        if self.recording:
            return self
        if self._monitor:
            self._reader.play_audio()
        self._reader.stream_start()
        return self

    def stop(self) -> np.ndarray:
        """Close the stream and return everything captured.

        Mono comes back shaped ``(n,)`` and multi-channel ``(n, channels)`` —
        the layout :func:`kudio.file_load` and :func:`kudio.save_wave` use.
        Calling this twice returns the same take rather than an empty array.
        """
        if self.recording:
            self._reader.terminate()
        return self.take()

    def take(self) -> np.ndarray:
        """Everything captured so far, without stopping."""
        with self._lock:
            chunks = list(self._chunks)
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        flat = np.concatenate(chunks).astype(np.float32, copy=False)
        if self._channels > 1:
            usable = (flat.size // self._channels) * self._channels
            return flat[:usable].reshape(-1, self._channels)
        return flat

    # ------------------------------------------------------------- monitoring

    def level(self) -> float:
        """Peak of the most recent chunk, linear ``0..1``."""
        with self._lock:
            return self._level

    def level_db(self) -> float:
        """The same reading in dBFS; ``-inf`` for silence."""
        peak = self.level()
        return 20.0 * float(np.log10(peak)) if peak > 0 else float('-inf')

    def elapsed(self) -> float:
        """Seconds captured so far."""
        with self._lock:
            return self._frames / self._sr if self._sr else 0.0

    def tail(self, seconds: float) -> np.ndarray:
        """The most recent *seconds* of audio, for a rolling display.

        Cheap enough to call from a repeating timer: it copies only the window
        asked for, not the whole take.
        """
        want = int(round(seconds * self._sr)) * self._channels
        with self._lock:
            chunks, total = list(self._chunks), self._frames * self._channels
        if not chunks or want <= 0:
            return np.zeros(0, dtype=np.float32)
        collected: List[np.ndarray] = []
        got = 0
        for chunk in reversed(chunks):
            collected.append(chunk)
            got += chunk.size
            if got >= want:
                break
        flat = np.concatenate(list(reversed(collected)))[-min(want, total):]
        flat = flat.astype(np.float32, copy=False)
        if self._channels > 1:
            usable = (flat.size // self._channels) * self._channels
            return flat[:usable].reshape(-1, self._channels)
        return flat

    def drain(self) -> np.ndarray:
        """Audio captured **since the last call**, and never twice.

        :meth:`tail` returns a sliding window, which overlaps between calls —
        right for a display, wrong for anything stateful. Feeding overlapping
        audio to a streaming denoiser processes the same samples repeatedly and
        corrupts every recursive estimate it holds.

        >>> while rec.recording:                            # doctest: +SKIP
        ...     monitor(enhancer.process(rec.drain()))
        """
        with self._lock:
            new, self._drained = self._chunks[self._drained:], len(self._chunks)
        if not new:
            return np.zeros(0, dtype=np.float32)
        flat = np.concatenate(new).astype(np.float32, copy=False)
        if self._channels > 1:
            usable = (flat.size // self._channels) * self._channels
            return flat[:usable].reshape(-1, self._channels)
        return flat

    def reset(self) -> None:
        """Throw the take away and start counting again, stream untouched."""
        with self._lock:
            self._chunks.clear()
            self._frames = 0
            self._drained = 0
            self._capped = False
            self._level = 0.0

    # ------------------------------------------------------------- lifecycle

    def __enter__(self) -> "StreamRecorder":
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.recording:
            self._reader.terminate()
