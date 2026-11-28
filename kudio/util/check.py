# -*- coding: utf-8 -*-
"""Audio device discovery and validation, through sounddevice (``kudio[audio]``).

One numbering for every device: sounddevice's, which is PortAudio's. The index
:func:`list_devices` and :meth:`CheckDevice.system_devices` print is the one
:func:`kudio.record`, :class:`kudio.Recorder`, :class:`kudio.StreamRecorder` and
:class:`kudio.LocalStreamReader` take. Given no device, every one of them takes
the input :meth:`CheckDevice.default_input` names.
"""
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
    """Every device sounddevice knows, by the index all of kudio's recording and
    playback takes.

    :param kind: ``'input'``, ``'output'``, or ``None`` for every device.
    :returns: ``[{'index', 'name', 'max_input_channels',
        'max_output_channels', 'default_samplerate', 'api'}, ...]``, ``api``
        naming the host API: the same microphone is a device under MME, another
        under WASAPI and another under WDM-KS.

    >>> [d['name'] for d in kudio.list_devices('input')]        # doctest: +SKIP
    """
    if kind not in (None, 'input', 'output'):
        raise ValueError(f"kind must be 'input', 'output' or None, got {kind!r}")
    sd = _sounddevice()
    apis = [api.get('name', '') for api in sd.query_hostapis()]

    devices = []
    for index, info_ in enumerate(sd.query_devices()):
        inputs = int(info_.get('max_input_channels', 0))
        outputs = int(info_.get('max_output_channels', 0))
        if kind == 'input' and inputs < 1:
            continue
        if kind == 'output' and outputs < 1:
            continue
        api = info_.get('hostapi')
        devices.append({
            'index': index,
            'name': info_.get('name', ''),
            'max_input_channels': inputs,
            'max_output_channels': outputs,
            'default_samplerate': float(info_.get('default_samplerate', 0.0)),
            'api': apis[api] if isinstance(api, int) and 0 <= api < len(apis) else '',
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
        raise DeviceError(
            f"sounddevice is installed but could not load PortAudio ({e}). "
            f"Install the PortAudio library: libportaudio2 on Debian and "
            f"Ubuntu, portaudio on Homebrew.") from e


def _nothing(*args) -> None:
    """A callback for a stream that is opened and closed and never started."""


class _PortAudio:
    """sounddevice asked what CheckDevice asks. The answers are dictionaries with
    the keys code built on CheckDevice reads: ``index``, ``name``, ``hostApi``,
    ``maxInputChannels``, ``maxOutputChannels``, ``defaultSampleRate``
    (``check.p.get_default_input_device_info()['index']``, say)."""

    def __init__(self):
        self.sd = _sounddevice()

    def get_device_count(self) -> int:
        return len(self.sd.query_devices())

    def get_device_info_by_index(self, index: int) -> dict:
        d = self.sd.query_devices(index)
        return {'index': index, 'name': d.get('name', ''), 'hostApi': d.get('hostapi'),
                'maxInputChannels': d.get('max_input_channels', 0),
                'maxOutputChannels': d.get('max_output_channels', 0),
                'defaultLowInputLatency': d.get('default_low_input_latency'),
                'defaultLowOutputLatency': d.get('default_low_output_latency'),
                'defaultHighInputLatency': d.get('default_high_input_latency'),
                'defaultHighOutputLatency': d.get('default_high_output_latency'),
                'defaultSampleRate': d.get('default_samplerate')}

    def get_host_api_info_by_index(self, api: int) -> dict:
        h = self.sd.query_hostapis(api)
        return {'index': api, 'name': h.get('name', ''), 'deviceCount': len(h.get('devices', ())),
                'defaultInputDevice': h.get('default_input_device', -1),
                'defaultOutputDevice': h.get('default_output_device', -1)}

    def get_default_input_device_info(self) -> dict:
        """The system's default input; OSError where there is none, as on a
        Windows machine whose MME lists no input."""
        try:
            d = self.sd.query_devices(kind='input')
        except Exception as e:                              # noqa: BLE001
            raise OSError("No Default Input Device Available") from e
        index = d.get('index')
        if index is None:
            index = next(i for i, x in enumerate(self.sd.query_devices())
                         if x.get('name') == d.get('name') and x.get('hostapi') == d.get('hostapi'))
        return self.get_device_info_by_index(int(index))

    def opens(self, device: int, rate: int, channels: int, frames: int) -> None:
        """Open an input stream on *device* and close it again; raises where the
        device does not take these settings. With a callback, as the streams
        that record are opened: WDM-KS, where a Windows machine may have every
        microphone it has, opens no stream without one ("Blocking API not
        supported")."""
        stream = self.sd.RawInputStream(device=device, samplerate=rate, channels=channels,
                                        dtype='int16', blocksize=frames, callback=_nothing)
        stream.close()

    def starts(self, device: int, rate: int, channels: int, frames: int) -> None:
        """Open an input stream on *device*, start it, stop it and close it; raises
        where any of that fails. A Bluetooth headset Windows lists while it is not
        connected opens, and does not start."""
        stream = self.sd.RawInputStream(device=device, samplerate=rate, channels=channels,
                                        dtype='int16', blocksize=frames, callback=_nothing)
        try:
            stream.start()
            stream.stop()
        finally:
            stream.close()

    def terminate(self) -> None:
        """Nothing to close: sounddevice keeps PortAudio for the life of the process."""


def is_available() -> bool:
    """Smoke-test the terminal colors and the audio input stack."""
    from kudio.util.colors import color_test
    return color_test('[kudio] Color test') and check_device()


@timer
def check_device() -> bool:
    """List system devices and probe an input; True on success.

    The input probed is the one :meth:`CheckDevice.input_device` picks with no
    arguments (:meth:`CheckDevice.default_input`), at the first of 16000, 22050
    and 44100 Hz it opens at.
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
    """Query and validate PortAudio devices, by sounddevice's index.

    >>> with CheckDevice() as cd:
    ...     devices = cd.system_devices()
    ...     device, rate, channels = cd.input_device()
    """

    standard_rate = [192000, 96000, 88200, 64000,
                     48000, 44100, 32000, 22050,
                     16000, 11025, 8000, 6000]

    def __init__(self, frame_size: int = 1024):
        self.p = _PortAudio()
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
        """Return `[{'Device', 'index', 'fs', 'out', 'in', 'api'}, ...]`, every
        device of every host API in index order, `api` naming the host API.

        All of them, because input_device() picks from all of them. On a
        Windows 11 machine with no default recording device, MME, DirectSound
        and WASAPI list no input at all and WDM-KS lists five.
        """
        device_list = []
        apis = {}
        for i in range(self.p.get_device_count()):
            dev = self.p.get_device_info_by_index(i)
            api = dev.get('hostApi')
            if api not in apis:
                apis[api] = self.p.get_host_api_info_by_index(api).get('name')
            device_list.append({
                'Device': dev.get('name'),
                'index': dev.get('index'),
                'fs': dev.get('defaultSampleRate'),
                'out': dev.get('maxOutputChannels'),
                'in': dev.get('maxInputChannels'),
                'api': apis[api],
            })
            log.debug("Device %s: in=%s out=%s (%s, %s)",
                      dev.get('index'), dev.get('maxInputChannels'),
                      dev.get('maxOutputChannels'), dev.get('name'), apis[api])
        return device_list

    def _describe(self, device) -> str:
        """The device's info for a message, or just its index when PortAudio
        cannot describe it: an index out of range raised from inside the
        message meant to explain it."""
        try:
            return str(self.get_device_info(device))
        except Exception:                                   # noqa: BLE001
            return f"device {device!r}"

    def get_device_info(self, device_index: int) -> dict:
        return self.p.get_device_info_by_index(device_index)

    def _is_input_device(self, device: int, sr: Optional[int] = None,
                         channels: Optional[int] = None) -> bool:
        try:
            info = self.get_device_info(device)
            if not info["maxInputChannels"] > 0:
                return False
            sr = int(info["defaultSampleRate"]) if sr is None else sr
            self.p.opens(device, sr, channels or 1, self.frame_size)
            return True
        except Exception as e:
            log.debug("Device probe failed (device=%s, sr=%s): %s", device, sr, e)
            return False

    def _starts(self, device: int) -> bool:
        try:
            info = self.get_device_info(device)
            if not info["maxInputChannels"] > 0:
                return False
            self.p.starts(device, int(info["defaultSampleRate"]), 1, self.frame_size)
            return True
        except Exception as e:
            log.debug("Device does not start (device=%s): %s", device, e)
            return False

    def default_input(self) -> int:
        """The input a recording takes when it is given none.

        The system's default input, when there is one; when it does not start,
        that is an error, and another microphone is not taken in its place.
        With no default input, as on a Windows machine whose microphones are
        all on WDM-KS, the last input that opens and starts, going back from
        the end of the list. The device taken is logged at INFO.

        :raises DeviceError: when the default input does not start, or no
            input starts.
        """
        try:
            default = int(self.p.get_default_input_device_info()["index"])
        except Exception:                                   # noqa: BLE001 - none there
            default = None
        if default is not None:
            if not self._starts(default):
                raise DeviceError(f"the system's default input, device {default}, does not "
                                  f"start: {self._describe(default)}; `kudio devices` lists "
                                  f"the others, and device= picks one")
            device = default
        else:
            device = next((d for d in reversed(range(self.p.get_device_count()))
                           if self._starts(d)), None)
            if device is None:
                raise DeviceError("No microphone found on this system")
        name = self._describe(device)
        try:
            name = self.get_device_info(device)["name"]
        except Exception:                                   # noqa: BLE001
            pass
        log.info("No device given: input %s, %s%s", device, " ".join(str(name).split()),
                 "" if default is not None else " (the system names no default input)")
        return device

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

        Missing arguments are auto-detected: the device by
        :meth:`default_input`, the first working rate of 16000/22050/44100,
        mono.

        :raises DeviceError: when there is no input, or the device, rate or
            channel count will not open.
        """
        if device is None:
            device = self.default_input()
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
