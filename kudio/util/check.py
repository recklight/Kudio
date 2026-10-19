# -*- coding: utf-8 -*-
"""Audio device discovery and validation (requires PyAudio)."""
from __future__ import annotations

import logging
from pprint import pformat
from typing import List, Optional, Tuple, Union

from kudio.exceptions import DeviceError
from kudio.util.tools import timer

__all__ = [
    'is_available',
    'check_device',
    'list_devices',
    'CheckDevice',
]

log = logging.getLogger(__name__)


def list_devices(kind: Optional[str] = None) -> List[dict]:
    """Devices as **sounddevice** sees them.

    This is the index space :func:`kudio.record` accepts. It is *not* the same
    numbering as :class:`CheckDevice`, which enumerates through PyAudio for the
    streaming/recorder classes — PortAudio's two Python bindings index devices
    independently, so an index from one must never be passed to the other.

    :param kind: ``'input'``, ``'output'``, or ``None`` for every device.
    :returns: ``[{'index', 'name', 'max_input_channels',
        'max_output_channels', 'default_samplerate'}, ...]``

    >>> [d['name'] for d in kudio.list_devices('input')]        # doctest: +SKIP
    """
    if kind not in (None, 'input', 'output'):
        raise ValueError(f"kind must be 'input', 'output' or None, got {kind!r}")
    sd = _sounddevice()

    devices = []
    for index, info_ in enumerate(sd.query_devices()):
        inputs = int(info_.get('max_input_channels', 0))
        outputs = int(info_.get('max_output_channels', 0))
        if kind == 'input' and inputs < 1:
            continue
        if kind == 'output' and outputs < 1:
            continue
        devices.append({
            'index': index,
            'name': info_.get('name', ''),
            'max_input_channels': inputs,
            'max_output_channels': outputs,
            'default_samplerate': float(info_.get('default_samplerate', 0.0)),
        })
    return devices


def _sounddevice():
    """sounddevice, or the error that says what is missing.

    Installed but unable to find the PortAudio library -- Linux without
    libportaudio2 -- it raises OSError on import, which is not a missing
    extra: pip cannot fix it, the system package can.
    """
    try:
        import sounddevice as sd
        return sd
    except ImportError as e:
        from kudio.exceptions import DependencyError
        raise DependencyError('sounddevice', extra='audio') from e
    except OSError as e:
        from kudio.exceptions import DeviceError
        raise DeviceError(
            f"sounddevice is installed but could not load PortAudio ({e}). "
            f"Install the PortAudio library: libportaudio2 on Debian and "
            f"Ubuntu, portaudio on Homebrew.") from e


def _pyaudio():
    try:
        import pyaudio
        return pyaudio
    except ImportError as e:
        from kudio.exceptions import DependencyError
        raise DependencyError('PyAudio', extra='audio') from e


def is_available() -> bool:
    """Smoke-test the terminal colors and the audio input stack."""
    from kudio.util.colors import color_test
    return color_test('[kudio] Color test') and check_device()


@timer
def check_device() -> bool:
    """List system devices and probe an input; True on success.

    The input probed is the one :meth:`CheckDevice.input_device` picks with no
    arguments: the last input-capable device PyAudio lists, at the first of
    16000, 22050 and 44100 Hz it opens at.
    """
    try:
        with CheckDevice() as cd:
            cd.system_devices()
            cd.input_device()
        log.info("Audio test passed")
        return True
    except Exception as e:
        log.error("Audio test failed: %s", e)
        return False


class CheckDevice:
    """Query/validate PortAudio devices.

    >>> with CheckDevice() as cd:
    ...     devices = cd.system_devices()
    ...     device, rate, channels = cd.input_device()
    """

    standard_rate = [192000, 96000, 88200, 64000,
                     48000, 44100, 32000, 22050,
                     16000, 11025, 8000, 6000]

    def __init__(self, frame_size: int = 1024):
        self.p = _pyaudio().PyAudio()
        self.frame_size = frame_size
        self.device: Optional[int] = None
        self.rate: Optional[int] = None
        self.channels: Optional[int] = None

    def __call__(self, m: str = 'device'):
        log.info(m)
        return self.system_devices()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.terminate()

    def terminate(self) -> None:
        self.p.terminate()

    def system_devices(self) -> List[dict]:
        """Return `[{'Device', 'index', 'fs', 'out', 'in'}, ...]` for host API 0."""
        device_list = []
        info = self.p.get_host_api_info_by_index(0)
        for i in range(info.get('deviceCount')):
            dev = self.p.get_device_info_by_host_api_device_index(0, i)
            device_list.append({
                'Device': dev.get('name'),
                'index': dev.get('index'),
                'fs': dev.get('defaultSampleRate'),
                'out': dev.get('maxOutputChannels'),
                'in': dev.get('maxInputChannels'),
            })
            log.info("Device %s: in=%s out=%s (%s)",
                     dev.get('index'), dev.get('maxInputChannels'),
                     dev.get('maxOutputChannels'), dev.get('name'))
        return device_list

    def _describe(self, device) -> str:
        """The device's info for a message, or just its index when PyAudio
        cannot describe it -- an index out of range used to raise from inside
        the message meant to explain it."""
        try:
            return str(self.get_device_info(device))
        except Exception:                                   # noqa: BLE001
            return f"device {device!r}"

    def get_device_info(self, device_index: int) -> dict:
        return self.p.get_device_info_by_index(device_index)

    def _is_input_device(self, device: int, sr: Optional[int] = None,
                         channels: Optional[int] = None) -> bool:
        pyaudio = _pyaudio()
        try:
            info = self.get_device_info(device)
            if not info["maxInputChannels"] > 0:
                return False
            sr = int(info["defaultSampleRate"]) if sr is None else sr
            stream = self.p.open(
                format=pyaudio.paInt16,
                channels=channels or 1,
                input=True,
                input_device_index=device,
                rate=sr,
                frames_per_buffer=self.frame_size)
            stream.close()
            return True
        except Exception as e:
            log.debug("Device probe failed (device=%s, sr=%s): %s", device, sr, e)
            return False

    def check_available_rate(self, device: int) -> Union[List[int], bool]:
        """Return the standard sample rates the device accepts, or False."""
        if not self._is_input_device(device):
            log.warning("Device unusable: %s", self._describe(device))
            return False
        return [rt for rt in self.standard_rate if self._is_input_device(device, rt)]

    def input_device(self, device: Optional[int] = None,
                     rate: Optional[int] = None,
                     channels: Optional[int] = None
                     ) -> Tuple[int, int, int]:
        """Resolve and validate an input device; returns (device, rate, channels).

        Missing arguments are auto-detected (last input-capable device, first
        working rate of 16000/22050/44100, mono).

        :raises DeviceError: when there is no input, or the device, rate or
            channel count will not open.
        """
        if device is None:
            mics = [d for d in range(self.p.get_device_count())
                    if self._is_input_device(d)]
            if not mics:
                raise DeviceError("No microphone found on this system")
            log.info("Found %d microphone(s)", len(mics))
            device = mics[-1]
        if not self._is_input_device(device):
            raise DeviceError(f"Device unusable: {self._describe(device)}")

        # sample rate
        if not (rate is not None and self._is_input_device(device, rate)):
            rate = next((r for r in (16000, 22050, 44100)
                         if self._is_input_device(device, r)), None)
            if rate is None:
                rate = int(self.get_device_info(device)["defaultSampleRate"])
        if not self._is_input_device(device, rate):
            raise DeviceError(
                f"Bad sample rate for device: {self._describe(device)}")
        log.info("Device %s, sample rate %s", device, rate)

        # channels
        if channels is not None:
            if self.get_device_info(device)["maxInputChannels"] < channels:
                log.warning("Requested channels exceed device maximum: %s",
                            self._describe(device))
                channels = 1
        else:
            channels = 1
        if not self._is_input_device(device, rate, channels):
            raise DeviceError(
                f"Bad channel setting ({channels}): {self._describe(device)}")

        self.device, self.rate, self.channels = device, rate, channels
        self.show_current_device_info()
        return self.device, self.rate, self.channels

    def show_current_device_info(self) -> None:
        if self.device is None or self.rate is None:
            self.input_device()
            return
        _device = self.get_device_info(self.device)
        log.info(
            "Selected device:\n%s\n"
            "  name: %s | index: %s | maxIn: %s | maxOut: %s\n"
            "  default fs: %s | fs: %s Hz | channels: %s | frame: %s | update: %.2f/s",
            pformat(_device), _device["name"], _device["index"],
            _device["maxInputChannels"], _device["maxOutputChannels"],
            _device["defaultSampleRate"], self.rate, self.channels,
            self.frame_size, self.rate / self.frame_size)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    is_available()
