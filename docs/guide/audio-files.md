# Audio files

Every waveform kudio loads is a float32 NumPy array in which ±1.0 is full
scale: 1-D for mono, `(frames, channels)` for more channels. Files go through
[soundfile](https://python-soundfile.readthedocs.io/), with ffmpeg for
[what it does not read](#formats-soundfile-does-not-read). The examples read
the files this block makes; run it in an empty folder.

??? example "Making the files the examples read"

    ```python
    from pathlib import Path

    import numpy as np

    import kudio


    def tone(freq, seconds, sr=16000, amp=0.3):
        t = np.arange(int(seconds * sr)) / sr
        return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


    kudio.save_wave("clip.wav", tone(220, 3.5, amp=0.45), 16000)
    kudio.save_wave("stereo.wav", np.stack([tone(220, 1), tone(880, 1)], axis=1), 16000)
    for name, amp in [("spk1/a", 0.05), ("spk1/b", 0.2), ("spk2/a", 0.6)]:
        y = tone(330, 2.0, 44100, amp)
        kudio.save_wave(f"raw/{name}.wav", np.stack([y, 0.8 * y], axis=1), 44100)
    for name in ["one", "two"]:
        kudio.save_wave(f"incoming/{name}.wav", tone(440, 1.0), 16000)
    Path("incoming/broken.wav").write_bytes(b"this is not a wave file")
    for name in ["rec.wav", "take.wav", "take.v2.wav"]:
        kudio.save_wave(f"takes/{name}", tone(250, 0.5), 16000)
    ```

## Looking before loading

```python
info = kudio.audio_info("clip.wav")
print(info)
print(info.frames, info.duration, info.subtype, info.format)
```

```text
clip.wav: 3.500s, 16000 Hz, 1ch, PCM_16, peak 0.4500
56000 3.5 PCM_16 WAV
```

[`audio_info`][kudio.core.io.audio_info] reads everything but the peak from the
header. The peak is read 2^20 frames at a time; `peak=False` skips it. An
array works too, with `sr=`. `kudio info` prints the same from a shell.

## Loading a file

```python
y, sr = kudio.file_load("clip.wav")                       # the file's own rate
y8, sr8 = kudio.file_load("clip.wav", sr=8000)            # resampled with soxr_hq
st, _ = kudio.file_load("stereo.wav", mono=False)         # channels kept
print(y.shape, sr, y8.shape, sr8, st.shape)
```

```text
(56000,) 16000 (28000,) 8000 (16000, 2)
```

[`file_load`][kudio.core.io.file_load] reads what libsndfile opens: WAV, FLAC,
OGG, MP3, AIFF and NIST SPHERE among them. `mono=True`, the default, averages
the channels. A file it cannot read raises an
[`AudioIOError`][kudio.exceptions.AudioIOError] naming the path and the reason;
a rate of 0, NaN, `True` or 16000.5 is a `ValueError`.

!!! warning "Channels last"
    `librosa.load(..., mono=False)` returns `(channels, frames)`. In kudio the
    left channel is `y[:, 0]`. An array of two rows or more with more columns
    than rows is refused with a `ValueError` that says to transpose it.

### Formats soundfile does not read

m4a, mp4, aac, webm, 3gp, amr and wma are decoded with
[audioread](https://github.com/beetbox/audioread) and
[ffmpeg](https://ffmpeg.org/) wherever kudio reads a file. Install the extra and
put ffmpeg on the PATH; without either, the `AudioIOError` says which is
missing:

```bash
pip install "kudio[compressed]"
```

An encoder pads its last block, so 2 s of sound comes back as 2.043 s from a
bare AAC stream and 2.000 s from an m4a: [line such files
up](metrics.md#finding-the-delay) before comparing them sample by sample.

### A file that was cut short

A WAV that ends before its header says is read as far as it goes, with a
warning such as `take.wav: the header says 6.00 s and 3.00 s are there, 96022
bytes fewer. The file may have been cut short.` A length of 0 or 0x7FFFF000 and
up, which streaming writers leave, is not held against the file. A FLAC that was
cut raises.

## Writing a file

```python
import numpy as np

hot = np.array([1.2, -1.3, 0.5], dtype=np.float32)
kudio.save_wave("out/hot16.wav", hot, 16000)                  # 16-bit PCM
kudio.save_wave("out/hotf.wav", hot, 16000, subtype="FLOAT")  # 32-bit float
kudio.save_wave("out/hot.flac", hot, 16000)                   # FLAC, by extension
print(kudio.file_load("out/hot16.wav")[0])
print(kudio.file_load("out/hotf.wav")[0])
```

```text
[ 0.9999695 -1.         0.5      ]
[ 1.2 -1.3  0.5]
```

| `subtype` | Stores | Past ±1.0 |
|---|---|---|
| `PCM_16` (default) | integers, 65,536 steps | clipped to full scale |
| `PCM_24` | integers, 16,777,216 steps | clipped to full scale |
| `FLOAT` | 32-bit floats | kept as written |

[`save_wave`][kudio.core.io.save_wave] creates missing folders, replaces an
existing file only once the new one is whole, and resamples first with
`d_sample=`. The subtype has to suit the container: FLAC takes `PCM_16` and
`PCM_24`, not `FLOAT`. kudio clips and rounds every integer subtype itself;
libsndfile's own conversion floors, half a 16-bit step low on average, and with
its clipping off wraps 1.2 round to -0.8:

![Left: a 220 Hz tone peaking at 1.25, read back from a FLOAT file from save_wave (intact), a PCM_16 file from save_wave (flat at plus and minus 1) and a PCM_16 file written with libsndfile's clipping off (the tops wrapped round to the other sign). Right: rounding error of 20,000 random samples written to 16 bits, spread from -1 to 0 steps through soundfile.write and from -0.5 to +0.5 through save_wave.](../images/audio-files-full-scale-light.webp#only-light)
![Left: a 220 Hz tone peaking at 1.25, read back from a FLOAT file from save_wave (intact), a PCM_16 file from save_wave (flat at plus and minus 1) and a PCM_16 file written with libsndfile's clipping off (the tops wrapped round to the other sign). Right: rounding error of 20,000 random samples written to 16 bits, spread from -1 to 0 steps through soundfile.write and from -0.5 to +0.5 through save_wave.](../images/audio-files-full-scale-dark.webp#only-dark)

An integer array is scaled by its own type's full scale everywhere in kudio:
int16 by 1/32768, int32 by 1/2^31, uint8 around 128. int64 is refused, since on
its scale 16-bit samples sit 288 dB down.

### NaN and inf

A file holding NaN or inf loads at its own rate, with a warning that counts the
samples; at another rate it is an `AudioIOError`, since a resampler would spread
them. `save_wave` refuses such samples and writes nothing.

## Changing the rate in memory

```python
y8 = kudio.resample(y, orig_sr=sr, target_sr=8000)
print(len(y), len(y8), kudio.resample(y, orig_sr=sr, target_sr=sr) is y)
```

```text
56000 28000 True
```

[`resample`][kudio.core.io.resample] uses `soxr_hq`, as `file_load(path, sr=)`
does, and returns the array itself at equal rates. A `(frames, channels)` array
is resampled along its frames; an integer array comes back in its own type.

## Gathering files

```python
files, rejected = kudio.check_input(["raw/", "takes/rec.wav", "missing.wav"])
print([f.as_posix() for f in files], rejected)
waves, sr, length = kudio.load_waves("raw/", std_len=True)
copied = kudio.copy_waves("picked/", "raw/")       # destination first
print(len(waves), sr, length, [p.name for p in copied])
```

```text
['raw/spk1/a.wav', 'raw/spk1/b.wav', 'raw/spk2/a.wav', 'takes/rec.wav'] ['missing.wav']
3 44100 88200 ['a.wav', 'b.wav', 'a2.wav']
```

[`check_input`][kudio.core.io.check_input] is how every folder tool reads its
input: files, folders at any depth, `.txt` lists (UTF-8, lines relative to the
working directory) and lists of these, and returns the `.wav` files and what it
could not use. [`load_waves`][kudio.core.io.load_waves] loads mono at the native
rate, `std_len=True` cut to the shortest; [`copy_waves`][kudio.core.io.copy_waves]
copies flat by name, so the second `a.wav` arrives as `a2.wav`.

## Converting a whole folder

```python
result = kudio.convert_folder("raw/", "16k/", sr=16000, lufs=-23.0)
print(result)
for path in result.outputs:
    y, sr = kudio.file_load(path)
    print(path.as_posix(), sr, f"{kudio.loudness(y, sr):.1f} LUFS")
```

```text
3/3 written, 0 failed
16k/spk1/a.wav 16000 -23.0 LUFS
16k/spk1/b.wav 16000 -23.0 LUFS
16k/spk2/a.wav 16000 -23.0 LUFS
```

[`convert_folder`][kudio.core.io.convert_folder] loads each file (at `sr=`,
mixed to mono unless `mono=False`), trims silence if `trim_db=` is set,
normalises to `peak=` or `lufs=` (not both), and writes `subtype=`, `PCM_16` by
default, mirroring the tree below `dst`. A list is mirrored below the deepest
folder its files share, or below `root=`. A taken name gets a free one, so empty
`dst` or pass `overwrite=True` to redo a run. Under `lufs=` a file shorter than
0.4 s fails, and a quiet file with one loud transient can pass full scale.

```python
result = kudio.convert_folder("incoming/", "clean/")
print(result, bool(result), [path.name for path, reason in result.failed])
```

```text
2/3 written, 1 failed False ['broken.wav']
```

A file that fails goes into `failed` as `(path, reason)` and the run goes on;
the result is falsy then, and `on_error="raise"` stops at the first failure.
The folder tools see `.wav` only. `kudio convert` exits 1 when a file failed.

## Picking a name that is free

```python
from pathlib import Path

for name in ["new.wav", "rec.wav", "take.v2.wav", "take.flac", "take.v2"]:
    print(name, "->", Path(kudio.check_file(Path("takes") / name)).name)
print(Path(kudio.check_file("takes/take.v2.wav", sep="_")).name)
```

```text
new.wav -> new.wav
rec.wav -> rec2.wav
take.v2.wav -> take.v22.wav
take.flac -> take2.wav
take.v2 -> take.v22.wav
take.v2_2.wav
```

[`check_file`][kudio.core.io.check_file] hands back a `.wav` name: an audio
suffix such as `.flac` is replaced, anything else after a dot is part of the
name, and the counter goes above the highest number in the folder. It creates
the parent folders, not the file, so ask right before `save_wave`.
[`check_path`][kudio.core.io.check_path] does the same for folders.

## Clips kept by a GUI

[`AudioDataManager`][kudio.core.manager.AudioDataManager] is the registry a GUI
keeps behind its clip list. `save_all(dst_path=...)` writes every clip, and
returns `False` without writing when two clips would land on one file.
