# -*- coding: utf-8 -*-
"""Audio device discovery and validation (requires PyAudio)."""
from __future__ import annotations

import logging
from pprint import pformat
from typing import List, Optional, Tuple, Union

from kudio.util.tools import timer

__all__ = [
    'is_available',
    'check_device',
    'CheckDevice',
]

log = logging.getLogger(__name__)


def _pyaudio():
    try:
        import pyaudio
        return pyaudio
    except ImportError as e:
        raise ImportError(
            "PyAudio is required for device checks: pip install kudio[audio]"
        ) from e


def is_available() -> bool:
    """Smoke-test the terminal colors and the audio input stack."""
    from kudio.util.colors import color_test
    return color_test('[kudio] Color test') and check_device()


@timer
def check_device() -> bool:
    """List system devices and probe the default input; True on success."""
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
            log.warning("Device unusable: %s", self.get_device_info(device))
            return False
        return [rt for rt in self.standard_rate if self._is_input_device(device, rt)]

    def input_device(self, device: Optional[int] = None,
                     rate: Optional[int] = None,
                     channels: Optional[int] = None
                     ) -> Tuple[int, int, int]:
        """Resolve and validate an input device; returns (device, rate, channels).

        Missing arguments are auto-detected (last input-capable device, first
        working rate of 16000/22050/44100, mono).
        """
        if device is None:
            mics = [d for d in range(self.p.get_device_count())
                    if self._is_input_device(d)]
            if not mics:
                raise RuntimeError("No microphone found on this system")
            log.info("Found %d microphone(s)", len(mics))
            device = mics[-1]
        if not self._is_input_device(device):
            raise RuntimeError(f"Device unusable: {self.get_device_info(device)}")

        # sample rate
        if not (rate is not None and self._is_input_device(device, rate)):
            rate = next((r for r in (16000, 22050, 44100)
                         if self._is_input_device(device, r)), None)
            if rate is None:
                rate = int(self.get_device_info(device)["defaultSampleRate"])
        if not self._is_input_device(device, rate):
            raise RuntimeError(
                f"Bad sample rate for device: {self.get_device_info(device)}")
        log.info("Device %s, sample rate %s", device, rate)

        # channels
        if channels is not None:
            if self.get_device_info(device)["maxInputChannels"] < channels:
                log.warning("Requested channels exceed device maximum: %s",
                            self.get_device_info(device))
                channels = 1
        else:
            channels = 1
        if not self._is_input_device(device, rate, channels):
            raise RuntimeError(
                f"Bad channel setting ({channels}): {self.get_device_info(device)}")

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
