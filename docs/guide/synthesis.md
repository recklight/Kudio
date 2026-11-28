# Building a noisy dataset

[`Synthesizer`][kudio.core.synth.Synthesizer] makes training pairs from clean
speech and noise, and adds as much of the path from a talker to a file as you
ask for, in the order the signal meets it:

```text
talker -> room (rt60=) -> + noise (snr_ratio=) -> link (channel=) -> file
```

The random parts (where the noise is cropped, the room, the lost packets) are
seeded per output file, and [`manifest()`][kudio.core.synth.Synthesizer.manifest]
writes down which mixture came from which clean file. For one clip in memory,
[`add_noise_snr`][kudio.effects.augment.add_noise_snr] is enough.

??? example "The corpus the examples on this page run on"

    Three short utterances from two talkers, a hiss and a mains hum.

    ```python
    import numpy as np
    import kudio

    sr = 16000
    rng = np.random.default_rng(0)
    t = np.arange(int(1.5 * sr)) / sr


    def talker(f0):
        """Nine harmonics of f0, sounding for 350 ms of every 500 ms."""
        voiced = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 10))
        return (0.05 * voiced * (t % 0.5 < 0.35)).astype(np.float32)


    kudio.save_wave("data/clean/f01/sa1.wav", talker(210), sr)
    kudio.save_wave("data/clean/f01/sx133.wav", talker(190), sr)
    kudio.save_wave("data/clean/m02/sx133.wav", talker(115), sr)

    kudio.save_wave("data/noise/hiss.wav",
                    (0.1 * rng.standard_normal(3 * sr)).astype(np.float32), sr)
    n = np.arange(2 * sr) / sr
    hum = sum(np.sin(2 * np.pi * 50 * k * n) / k for k in (1, 3, 5))
    kudio.save_wave("data/noise/hum.wav", (0.05 * hum).astype(np.float32), sr)
    ```

## A first run

```python
syx = kudio.Synthesizer("data/clean", "data/noise", out_path="data/mixed",
                        snr_ratio=(-5, 0, 5))
rows = syx.syn(mode="inc", seed=17)
kudio.save_manifest("data/mixed.json", syx.manifest())
print(len(rows), "files")
for out, clean, noise, snr in rows[:4]:
    print(out.as_posix(), "<-", clean.name, noise.name, snr)
```

```text
18 files
data/mixed/f01/sa1_hiss_n5.wav <- sa1.wav hiss.wav -5
data/mixed/f01/sa1_hiss_0dB.wav <- sa1.wav hiss.wav 0
data/mixed/f01/sa1_hiss_5dB.wav <- sa1.wav hiss.wav 5
data/mixed/f01/sa1_hum_n5.wav <- sa1.wav hum.wav -5
```

[`syn()`][kudio.core.synth.Synthesizer.syn] returns one `(output, clean, noise,
snr)` tuple per file; rooms, links and targets go in [the
manifest](#the-manifest). Names follow `{clean}_{noise}_{snr}.wav`: `5dB` for +5,
`n5` for −5, `2p5dB` for 2.5, and a room adds `_rt300ms`.

!!! warning "`out_path` is emptied first"
    With `overwrite=True`, the default, `syn()` deletes everything in `out_path`
    before it writes. It raises [`SynthesisError`][kudio.exceptions.SynthesisError]
    instead when that folder holds an input file, the working directory or the
    home directory, or is a link or a filesystem root. It tries the run without
    the samples before deleting anything. `overwrite=False` leaves a non-empty
    folder alone, warns and returns `None`.

| `mode=` | Files written | Use it for |
|---|---|---|
| `"regular"` / `"reg"` (default) | one per clean file, the conditions dealt out in turn | a training set the size of the clean corpus |
| `"increment"` / `"inc"` | clean × noise × SNR | a test set broken down by condition |

Both multiply by the number of rooms. `rdn_choice_num=N` synthesises N clean
files drawn at random.

## Folders, lists and `root=`

`clean` and `noise` take what [`check_input`][kudio.core.io.check_input] takes:
a folder, a `.wav` file, a `.txt` list or a list of those, so a subset such as
`sorted(Path("data/clean").rglob("sx*.wav"))` is synthesised without copying it
out. The output mirrors the clean files below their folder, below the deepest
folder a list shares, or below `root=`, so two talkers' `sx133` stay apart. A
noise is known by its stem alone, so noise files from different folders need
distinct names. `syn(mkdir_parents=False)` writes flat.

## What the SNR is measured against

```python
syx = kudio.Synthesizer("data/clean/f01/sa1.wav", "data/noise/hiss.wav",
                        out_path="data/check", snr_ratio=5, subtype="FLOAT")
[(out, clean, noise, snr)] = syx.syn(seed=17)
speech, sr = kudio.file_load(clean)
mix, sr = kudio.file_load(out)
talking = speech != 0                     # the pauses are digital zero
print(f"whole file {kudio.snr(speech, mix):.2f} dB, "
      f"while talking {kudio.snr(speech[talking], mix[talking]):.2f} dB")
```

```text
whole file 5.00 dB, while talking 6.54 dB
```

The noise, mean removed, is scaled to the clean file's mean power over the whole
file, pauses included, so a talker silent for 30% of the file sits 1.5 dB above
the label while talking; trim the clean files first if the label should describe
the speech. SNRs lie between -300 and 3000 dB. A shorter noise is looped, a
longer one cropped at a seeded offset, and it is resampled to the clean file's
rate. A noise or a clean file without variation raises `SynthesisError`.

## Rooms and links

```python
from pathlib import Path

syx = kudio.Synthesizer("data/clean", "data/noise", out_path="data/rooms",
                        snr_ratio=(0, 5), rt60=(0.3, 0.8), drr_db=3.0,
                        write_targets=True)
syx.syn(mode="reg", seed=17)
kudio.save_manifest("data/rooms.json", syx.manifest())
for p in syx.manifest()[:3]:
    dry, wet, mix = (kudio.file_load(f)[0] for f in (p.clean, p.target, p.noisy))
    noise = mix - wet
    print(f"{Path(p.noisy).name:26} against wet {kudio.snr(wet, wet + noise):5.2f} dB,"
          f" against dry {kudio.snr(dry, dry + noise):5.2f} dB")
```

```text
sa1_hiss_0dB_rt300ms.wav   against wet  0.00 dB, against dry  1.92 dB
sa1_hiss_0dB_rt800ms.wav   against wet  0.00 dB, against dry  0.94 dB
sx133_hum_5dB_rt300ms.wav  against wet  5.00 dB, against dry  5.35 dB
```

`rt60=` takes one or more reverberation times and puts each clean file through a
room of its own ([`rir`][kudio.core.room.rir], [`apply_rir`][kudio.core.room.apply_rir]),
`drr_db=` (0 dB) away, before the noise. The noise competes with what the room
left, so the SNR is set against the reverberant speech. The manifest's target is
the clean file; `write_targets=True` also writes the reverberant clean signal to
`out_path/TARGETS/`, for a model that should not undo the room.

`channel=` sends every finished mixture down a link, a preset from
[`CHANNEL_PRESETS`][kudio.effects.channel.CHANNEL_PRESETS] or a dict
([Rooms and channels](degradation.md)), and leaves `TARGETS/` alone. It is one
setting for the run, not an axis, and the manifest tags each row with it.

## Seeds, and the pool

`seed=` is a base: each file draws its crop, room and packet loss from the base
and its own path below `out_path`, so the same request rebuilds the same files
byte for byte, in any order and in any process. With no seed the base is drawn
from numpy's global generator and kept in [`.seed`][kudio.core.synth.Synthesizer.seed]:

```python
np.random.seed(2026)                       # only so the number below repeats
first = kudio.Synthesizer("data/clean", "data/noise", out_path="data/run1",
                          rt60=0.5, channel="voip")
first.syn(mode="inc")                      # no seed
again = kudio.Synthesizer("data/clean", "data/noise", out_path="data/run2",
                          rt60=0.5, channel="voip")
again.syn(mode="inc", seed=first.seed)
run1, run2 = (sorted(Path(d).rglob("*.wav")) for d in ("data/run1", "data/run2"))
print("drew", first.seed, len(run1), "files, identical:",
      [f.read_bytes() for f in run1] == [f.read_bytes() for f in run2])
```

```text
drew 942082305 18 files, identical: True
```

`is_pool=True` writes through 8 worker processes once a run has more than 200
files, with the same bytes as a serial run. On Windows and macOS the call has to
sit under `if __name__ == "__main__":`.

## Writing the files

```python
y, sr = kudio.file_load("data/clean/f01/sa1.wav")
kudio.save_wave("data/loud/sa1.wav", kudio.normalize(y, peak=0.9), sr)
for subtype in ("PCM_16", "FLOAT"):
    syx = kudio.Synthesizer("data/loud", "data/noise", out_path=f"data/loud_{subtype}",
                            snr_ratio=(0, 10), subtype=subtype)
    syx.syn(mode="inc", seed=17)
    print(subtype, sorted(round(peak, 2) for peak in syx.clipped.values()))
```

```text
PCM_16 [1.05, 1.26, 1.37, 2.07]
FLOAT []
```

`subtype=` is `"PCM_16"` by default. A mixture past full scale is clipped in a
fixed-point subtype, and `syn()` warns with the count and the worst peak;
[`.clipped`][kudio.core.synth.Synthesizer.clipped] maps each such file to its
peak. `"FLOAT"` keeps the samples. `target_sr=` on `syn()` resamples the output;
a stereo clean file is mixed to mono.

## The manifest

[`save_manifest`][kudio.core.dataset.save_manifest] writes one
[`Pair`][kudio.core.dataset.Pair] per file as plain JSON: `noisy`, `clean`,
`noise`, `snr_db`, `rt60`, `channel` and `target`, and
[`load_manifest`][kudio.core.dataset.load_manifest] reads it back. Only `noisy`
and `clean` are required and unknown keys are dropped with a warning. Paths are
written with forward slashes, relative where the inputs were, read from the
folder the script ran in. `is_silence=True` also writes about `p_silence` (2%)
of the outputs as noise alone, `..._n00.wav`, which `syn()` returns and the
manifest leaves out.

## Train, validation and test

```python
pairs = kudio.load_manifest("data/mixed.json")
train, val, test = kudio.split_pairs(pairs, 0.2, 0.2, seed=17)
shared = {p.clean for p in train} & {p.clean for p in test}
print(len(train), len(val), len(test), len(shared), "clean files in train and test")
```

```text
10 4 4 2 clean files in train and test
```

[`split_pairs`][kudio.core.dataset.split_pairs] shuffles with `seed=` (the
default `None` splits differently each call) and deals out rows, not talkers or
noises, so one clean file at three SNRs can land in train and test. For a test
of talkers or noises the model has never heard, split the clean or noise files
first and synthesise each split into its own folder, with `root=`. Exact repeat
rows are dropped with a warning; two rows naming one mixture that disagree raise
`ValueError`.

## Noise over a background, and the command line

[`syn_extra_mode(background_path, background_snr)`][kudio.core.synth.Synthesizer.syn_extra_mode]
mixes in two stages: every noise with every background at each `background_snr`,
then every clean file with each of those mixtures, each noise and each
background at every SNR. It applies no rooms or links, writes flat, writes
`n5dB` for −5, and takes no `target_sr=`. `kudio synth --clean data/clean
--noise data/noise --out data/cli --snr -5 0 5 --seed 17` does a run from a
shell, `--mode inc` by default, and prints the seed it used. [Denoising](enhancement.md) and
[Measuring quality](metrics.md) take the mixtures and their clean references from
there.
