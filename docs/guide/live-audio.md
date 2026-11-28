# Recording and live views

Audio that is still arriving: taken from a microphone or from a device on the
network, drawn or denoised while the take runs, and played back.

## What needs the audio extra

```bash
pip install "kudio[audio]"     # sounddevice
```

kudio reaches the hardware through sounddevice, a binding of PortAudio, and
imports it only when needed. Its Windows and macOS wheels carry PortAudio; on
Linux it uses the system's, `libportaudio2` on Debian and Ubuntu.

| Backend | Used by |
|---|---|
| sounddevice | [`record`][kudio.core.stream.record], [`list_devices`][kudio.util.check.list_devices], [`play_audio`][kudio.core.stream.play_audio], [`StreamRecorder`][kudio.core.stream.StreamRecorder], [`Recorder`][kudio.core.stream.Recorder], [`LocalStreamReader`][kudio.core.stream.LocalStreamReader], [`RemoteStreamReader`][kudio.core.stream.RemoteStreamReader], [`CheckDevice`][kudio.util.check.CheckDevice] |
| none | [`SpectrogramStream`][kudio.core.spectrogram.SpectrogramStream], [`StreamEnhancer`][kudio.enhance.streaming.StreamEnhancer], [`AudioBuffer`][kudio.core.buffer.AudioBuffer], [`FetchBuffer`][kudio.core.buffer.FetchBuffer] |

Without the extra these raise [`DependencyError`][kudio.exceptions.DependencyError]
naming `kudio[audio]`; sounddevice on a Linux machine without the PortAudio
library raises [`DeviceError`][kudio.exceptions.DeviceError] naming the system
package.

## A fixed-length take with `record`

```python
import kudio

y = kudio.record(seconds=3, sr=16000)          # blocks for 3 s
kudio.save_wave("take.wav", y, 16000)
```

[`record`][kudio.core.stream.record] returns float32 with `int(seconds * sr)`
samples, `(n,)` for mono and `(n, channels)` otherwise. A device that refuses
the rate raises a `DeviceError` that names the device and the rate.

## Recording until you stop: `StreamRecorder`

```python
import time

with kudio.StreamRecorder(sr=16000) as rec:
    print(f"{rec.sr} Hz, {rec.channels} channel(s)")   # what the device agreed to
    while rec.elapsed() < 5.0:                         # or until someone clicks Stop
        print(f"\r{rec.level_db():6.1f} dBFS", end="")
        time.sleep(0.1)
y = rec.stop()                                         # float32, (n,) for mono
kudio.save_wave("take.wav", y, rec.sr)
```

[`StreamRecorder`][kudio.core.stream.StreamRecorder] captures on PortAudio's
thread and keeps every sample; in a GUI, call
[`start()`][kudio.core.stream.StreamRecorder.start] and
[`stop()`][kudio.core.stream.StreamRecorder.stop] from the buttons. `sr=` is a
request: if the device refuses it, 16000, 22050 and 44100 Hz are tried, then the
device's default, and the choice is logged at INFO. `rec.sr` and `rec.channels`
are what the take is in. An input that will not open raises `DeviceError`.

| Safe from any thread | Returns |
|---|---|
| [`level()`][kudio.core.stream.StreamRecorder.level] | peak of the latest chunk, 0 to 1 |
| [`level_db()`][kudio.core.stream.StreamRecorder.level_db] | the same in dBFS, `-inf` for digital silence |
| [`elapsed()`][kudio.core.stream.StreamRecorder.elapsed] | seconds captured so far |
| [`tail(seconds)`][kudio.core.stream.StreamRecorder.tail] | the most recent *seconds*, for a display |
| [`drain()`][kudio.core.stream.StreamRecorder.drain] | everything since the previous `drain()`, each sample once |
| [`take()`][kudio.core.stream.StreamRecorder.take] | everything so far, without stopping |

A chunk is `frame_size` (1024) samples. A `start()` after `stop()` carries on the
same take, like a Pause button, and raises `DeviceError` if the device comes back
at another rate; [`reset()`][kudio.core.stream.StreamRecorder.reset] throws the
take away. The take stops growing at `max_seconds` (3600, `None` for no cap) and
[`capped`][kudio.core.stream.StreamRecorder.capped] turns true while the level
keeps updating. `monitor=True` plays the input through the default output. Feed
anything that keeps state, a spectrogram, a denoiser or a file, from `drain()`;
`tail()` returns overlapping windows.

## Choosing a device

Every device has one index, sounddevice's, which
[`list_devices()`][kudio.util.check.list_devices],
[`CheckDevice().system_devices()`][kudio.util.check.CheckDevice.system_devices]
and `kudio devices` print with each host API, and which `record`, `Recorder`,
`StreamRecorder` and `LocalStreamReader` take as `device`. Left out, the device
is [`CheckDevice.default_input`][kudio.util.check.CheckDevice.default_input]:
the system's default input, or a `DeviceError` when it does not start; with no
default input, as on a Windows machine whose microphones are all on WDM-KS, the
last input that opens and starts. Pass `device` whenever it matters which input
records. [`check_available_rate(device)`][kudio.util.check.CheckDevice.check_available_rate]
returns which of twelve standard rates open.

## A live spectrogram: `SpectrogramStream`

```python
import numpy as np
import kudio

sr = 16000
y = (0.5 * np.sin(2 * np.pi * 1000 * np.arange(2 * sr) / sr)).astype(np.float32)
view = kudio.SpectrogramStream(sr, seconds=3.0)
print(view)
for start in range(0, len(y), 700):        # whatever block size the device uses
    new = view.push(y[start:start + 700])  # the columns this block completed
print(view.columns().shape, view.filled, view.frames_seen)
print(view.latency_seconds, view.column_seconds, round(float(view.latest().max()), 2))
```

```text
SpectrogramStream(sr=16000, n_fft=512, hop_length=128, 257 bins, 3.00s window)
(375, 257) 247 247
0.032 0.008 -6.02
```

[`SpectrogramStream`][kudio.core.spectrogram.SpectrogramStream] computes each
column once, when its samples have arrived, and keeps the last `seconds` in a
ring buffer. [`push()`][kudio.core.spectrogram.SpectrogramStream.push] takes a
mono block of any length and returns the columns it completed, in dB, oldest
first; the block size does not change a single value.
[`columns()`][kudio.core.spectrogram.SpectrogramStream.columns] is the whole
window, full width from the first call with
[`floor_db`][kudio.core.spectrogram.SpectrogramStream.floor_db] where nothing
has arrived yet, and [`latest()`][kudio.core.spectrogram.SpectrogramStream.latest]
the newest column. The first column waits one frame, `n_fft / sr`; the rest
follow every `hop_length` (`n_fft // 4`).

![Three snapshots of SpectrogramStream.columns() from one stream, after 1, 2.5 and 5 seconds: the window is full width throughout, the part not heard yet sits at the floor, and old syllables scroll off the left](../images/live-audio-filling-light.webp#only-light)
![Three snapshots of SpectrogramStream.columns() from one stream, after 1, 2.5 and 5 seconds: the window is full width throughout, the part not heard yet sits at the floor, and old syllables scroll off the left](../images/live-audio-filling-dark.webp#only-dark)

- Levels are dBFS on the linear axis: a full-scale sine reads 0 dB, and
  anything `top_db` (80) below `ref` is clamped to `floor_db`. Draw with a fixed
  colour range, `vmin=view.floor_db, vmax=0`.
- `n_mels=64` puts the rows on a mel scale, which shows speech better but is not
  dBFS: a full-scale 1 kHz sine reads about -18 dB, so set `vmax` near -15.
- Column *k* covers samples `[k*hop, k*hop + n_fft)`, not centred as the offline
  `STFT` is, so it is half a frame later than the offline column of the same
  index; [`to_stft()`][kudio.core.spectrogram.SpectrogramStream.to_stft] gives
  the matching geometry.
- It holds no lock: push and draw from one thread, and leave the crossing to
  `StreamRecorder.drain()`. A 2-D block is refused with `FeatureError`.

With matplotlib (`kudio[viz]`) and a microphone, the live view is this; for a
denoised view, push `enhancer.process(rec.drain())` from a
[`StreamEnhancer`][kudio.enhance.streaming.StreamEnhancer] instead
([Denoising](enhancement.md)):

```python
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

rec = kudio.StreamRecorder(sr=16000).start()
view = kudio.SpectrogramStream(rec.sr, seconds=3.0)
fig, ax = plt.subplots()
image = ax.imshow(view.columns().T, origin="lower", aspect="auto",
                  vmin=view.floor_db, vmax=0.0,
                  extent=(-view.window_seconds, 0.0, 0.0, rec.sr / 2))


def update(_):
    view.push(rec.drain())
    image.set_data(view.columns().T)
    return [image]


anim = FuncAnimation(fig, update, interval=50, blit=True, cache_frame_data=False)
plt.show()                                 # returns when the window is closed
y = rec.stop()
```

## Playing audio

```python
tone = np.sin(2 * np.pi * 440 * np.arange(16000) / 16000)    # float64
print(kudio.play_audio(tone, 16000))
```

```text
unsupported dtype: float64
```

[`play_audio`][kudio.core.stream.play_audio] takes an array with its rate, a
path, or an open `wave.Wave_read`, and returns a message instead of raising for
most failures, so check what comes back; `None` means it played. Arrays must be
int16 or float32. An array plays in the background, so a script that ends on the
next line cuts it off; `sounddevice.wait()` blocks until it is done. A path is
read as `file_load` reads it. A missing sounddevice raises `DependencyError`.

## The lower-level pieces

[`Recorder`][kudio.core.stream.Recorder] is a blocking, fixed-length recorder that
returns int16, with no rate fallback: a refused rate, or a device that does not
start, raises a `DeviceError` that names the device and the rate, as `record()`
does, and a device that stops delivering raises one ten seconds after the take
should have ended.

```python
with kudio.Recorder(record_times=3, rate=16000) as r:   # the system's default input
    y = r.start_record()                       # int16, (n,) for mono
    r.save("take.wav")                         # 16-bit WAV; True or False
```

[`LocalStreamReader`][kudio.core.stream.LocalStreamReader] opens an input and
hands every chunk, one-dimensional with the channels interleaved (int16, or
float32 with `wav_format=kudio.PA_FLOAT32`), to an `AudioBuffer`, an `on_frame`
callback, or both. A device that will not open leaves it uninitialised, and
`get_status()` says so; `get_setup()` returns what was opened. `on_frame` runs on
PortAudio's thread, so store the chunk and return; an exception inside it is
logged and the stream goes on.

An [`AudioBuffer`][kudio.core.buffer.AudioBuffer] is a bounded FIFO of `size`
chunks (2) that drops the oldest when full, which suits a display.
[`get()`][kudio.core.buffer.AudioBuffer.get] waits for a chunk and raises
`TimeoutError` after `timeout` seconds;
[`get_fixed_size_data(n)`][kudio.core.buffer.AudioBuffer.get_fixed_size_data]
waits for the next `n` chunks and refuses `n = 0`.
[`FetchBuffer(buf).get_data(n, timeout=...)`][kudio.core.buffer.FetchBuffer.get_data]
runs the same wait on a thread of its own.

[`RemoteStreamReader`][kudio.core.stream.RemoteStreamReader] is a TCP client for
a device that streams raw PCM, `"16bit"` or `"32bit"` float:

```python
buf = kudio.AudioBuffer(size=16)
remote = kudio.RemoteStreamReader(buf)
if remote.connect_remote_device("192.168.1.50", 5000):
    remote.initialize_params(frame_size=2048, wav_format="16bit")
    remote.start()
    chunk = buf.get(timeout=5.0)               # 2048 int16 samples
    remote.terminate()                         # sends "stop", closes the socket
```

The server sends samples once the connection is open and should stop on the text
`stop`. Send other commands with
[`sock_send()`][kudio.core.stream.RemoteStreamReader.sock_send] before
`start()`. A chunk that arrives in pieces is put back together, up to
`break_count` (10) further reads. The parameters are under
[Live audio](../reference/live-audio.md) in the API reference.
