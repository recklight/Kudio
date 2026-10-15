# User guide

The guide follows the job in front of you rather than the module layout. Each
page explains what the functions are for, which one to reach for and where they
go wrong, with examples that run as written. Every parameter of every function
is in the [API reference](../reference/index.md).

## Pick a page by the job

| You want to | Read |
|---|---|
| read, write or convert audio files, a whole folder at a time | [Audio files](audio-files.md) |
| turn audio into the matrices a model trains on | [STFT and features](features.md) |
| know whether a recording is usable before doing anything with it | [Inspecting a recording](inspection.md) |
| find where somebody is talking, and on what pitch | [Speech activity and pitch](speech.md) |
| cut, join, filter, fade or augment | [Editing and effects](editing.md) |
| put a clip in a room or down a phone line | [Rooms and channels](degradation.md) |
| make degraded/clean training pairs from a folder of speech and one of noise | [Building a noisy dataset](synthesis.md) |
| take noise out, or choose between denoisers | [Denoising](enhancement.md) |
| score processed audio against a clean reference | [Measuring quality](metrics.md) |
| record, monitor or draw audio while it is still arriving | [Recording and live views](live-audio.md) |

## The order a recording is damaged in

Speech reaches a file through a chain, and several pages are about one link of
it:

```text
talker -> room -> + noise -> link -> file
```

The room is convolutive, the noise is added, and the link (a phone line, a
codec, a network that drops packets) acts on whatever reached the microphone.
[Rooms and channels](degradation.md) covers the room and the link,
[Building a noisy dataset](synthesis.md) applies the whole chain in that order,
and [Denoising](enhancement.md) and [Measuring quality](metrics.md) work back
from the file.

## Conventions on every page

- Code imports `kudio` and calls everything from the top level:
  `kudio.file_load`, not `kudio.core.io.file_load`.
- A waveform is a float32 NumPy array with ±1.0 as full scale. Mono is 1-D;
  more than one channel is `(frames, channels)`, the layout soundfile uses.
- A spectrogram has one row per frame: `(frames, bins)`.
- Anything random takes a seed, and the examples pass one, so the output shown
  on a page is the output you get.
