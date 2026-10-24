# Building a noisy dataset

A denoising model learns from pairs: a degraded recording and the clean signal
behind it. [`Synthesizer`][kudio.core.synth.Synthesizer] makes those pairs from
clean speech and noise, and adds as much of the path between a talker and a
file as you ask for, in the order the signal meets it:

```text
talker -> room (rt60=) -> + noise (snr_ratio=) -> link (channel=) -> file
```

The room acts on the talker, the noise reaches the microphone alongside what
the room left, and the link carries whatever the microphone picked up, noise
included. The random parts of that chain (where the noise is cropped, the
room, the lost packets) are seeded per output file, and
[`manifest()`][kudio.core.synth.Synthesizer.manifest] writes down which
mixture came from which clean file, so nobody has to work that out from file
names later. For one clip in memory,
[`add_noise_snr`][kudio.effects.augment.add_noise_snr] is enough
([Editing and effects](editing.md)); `Synthesizer` is for a corpus on disk.

??? example "The corpus the examples on this page run on"

    Three short utterances from two talkers, a hiss and a mains hum. All of it
    synthetic, so every number on this page can be rebuilt from this block.

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
import kudio

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

[`syn()`][kudio.core.synth.Synthesizer.syn] returns one
`(output, clean, noise, snr)` tuple per file it wrote, and it stays a
four-tuple whatever else you switch on: a return shape that depends on a
keyword breaks every caller that unpacks it, so rooms, links and targets go in
[the manifest](#the-manifest) instead. The names follow
`{clean}_{noise}_{snr}.wav`: `5dB` for +5, and `n5`, with no `dB`, for −5. A
fraction is written with a `p` (`2p5dB`, `n2p5`), since a decimal point in a
file name reads as the start of the suffix, and a room adds `_rt300ms`.

!!! warning "`out_path` is emptied first"
    With `overwrite=True`, the default, `syn()` deletes everything in
    `out_path` before it writes. It refuses, with
    [`SynthesisError`][kudio.exceptions.SynthesisError], when that folder
    holds one of the input files, the working directory or the home
    directory, or is the root of a filesystem, which catches an `out_path=`
    one level too high. Anything else in there is gone. `overwrite=False`
    leaves a non-empty folder alone, warns, and returns `None`, so check the
    return value in a script that carries on.

## Which mode: `reg` or `inc`

| `mode=` | Files written | Reach for it when |
|---|---|---|
| `"regular"` / `"reg"` (default) | one per clean file | a training set the size of the clean corpus, with the conditions spread over it |
| `"increment"` / `"inc"` | clean × noise × SNR | every utterance should be heard in every condition, e.g. a test set you will break down by SNR |

Both multiply by the number of rooms when `rt60=` is given. `reg` deals the
noises and SNRs out in turn, so that any noises × SNRs consecutive files hold
every combination exactly once, whatever the two counts are. A corpus
smaller than that leaves some out: the three clean files here, against 2
noises and 3 SNRs, get hiss at −5 and 5 dB and hum at 0 dB.

`rdn_choice_num=N` synthesises N clean files drawn at random, without
replacement, instead of all of them; an N larger than the corpus takes all of
them, with a warning.

## Folders, lists and `root=`

`clean` and `noise` take anything [`check_input`][kudio.core.io.check_input]
does: a folder, a `.wav` file, a `.txt` list of paths, or a list of those
([Audio files](audio-files.md) has the rules). Only WAV files are read, `.wav`
or `.WAV`. A folder search passes over every other format without a word; a
list entry that is not an existing `.wav` file is skipped with a
`UserWarning`. A list is how a subset gets synthesised without copying it out
first.

The output mirrors the layout of the clean files. A folder is measured from
itself; a list or a `.txt` file from the deepest folder its files share, so
two talkers who both read `sx133` still get two files:

```python
from pathlib import Path
import kudio

# every talker's sx files, and nothing else
subset = sorted(Path("data/clean").rglob("sx*.wav"))
syx = kudio.Synthesizer(subset, "data/noise", out_path="data/mixed_sx",
                        snr_ratio=0)
for out, *_ in syx.syn(seed=17):
    print(out.as_posix())
```

```text
data/mixed_sx/f01/sx133_hiss_0dB.wav
data/mixed_sx/m02/sx133_hum_0dB.wav
```

When that shared folder is deeper than the one you mean, pass `root=`. One
talker's files share the talker's own folder, so on their own they land flat
in `out_path`; with `root="data/clean"` they land in `out_path/f01/`. A
`root=` that does not hold every clean file is refused with `ValueError`.

Only the clean side is mirrored. A noise file is known by its stem alone, in
the output name and in the manifest's `noise` field, so a noise set that
repeats names across folders needs distinct names first (DEMAND, for one, has
`ch01.wav` to `ch16.wav` in every environment's folder; copy them out as
`kitchen_ch01.wav` and so on). Otherwise `syn()` warns that outputs share a
name, the last noise written wins, and the manifest keeps a row for each with
nothing to say which noise is in the file.

[`syn(mkdir_parents=False)`][kudio.core.synth.Synthesizer.syn] writes
everything flat into `out_path`. When two clean files of one name then ask
for the same output name, the first keeps it and the other is left out of the
run and the manifest, with a warning naming it.

## What the SNR is measured against

The noise has its mean removed and is scaled so that its power is the clean
signal's mean power divided by $10^{\mathrm{SNR}/10}$. That mean is taken over
the whole clean file, pauses included. Written as `FLOAT`, the mixture minus
the clean file gives the scaled noise back sample for sample, so
[`snr`][kudio.core.evaluator.snr] can check this on disk:

```python
import kudio

syx = kudio.Synthesizer("data/clean/f01/sa1.wav", "data/noise/hiss.wav",
                        out_path="data/check", snr_ratio=5, subtype="FLOAT")
[(out, clean, noise, snr)] = syx.syn(seed=17)

speech, sr = kudio.file_load(clean)
mix, sr = kudio.file_load(out)
talking = speech != 0                     # the pauses are digital zero
print(f"whole file     {kudio.snr(speech, mix):.2f} dB")
print(f"while talking  {kudio.snr(speech[talking], mix[talking]):.2f} dB")
```

```text
whole file     5.00 dB
while talking  6.54 dB
```

This talker is silent for 30% of the file, which puts the SNR 1.54 dB above
the label while they are talking. Leading and trailing silence does the same,
so if the label should describe the speech, trim the clean files first
([Speech activity and pitch](speech.md) finds where the speech is).

A noise file shorter than its clean partner is looped to length; a longer one
is cropped at a random offset drawn from the file's seed. The noise is
resampled to the clean file's rate before mixing. A noise file with no
variation in it, digital silence or a constant level, raises
[`SynthesisError`][kudio.exceptions.SynthesisError] naming the file, since no
gain puts a constant at any SNR. A clean file of digital silence is refused
the same way: the noise is scaled against the speech, and with no speech its
mixture would be a file of zeros labelled with an SNR.

## Rooms: `rt60=` and reverberant targets

`rt60=` takes one or more reverberation times in seconds and puts the clean
signal through a room, built with [`rir`][kudio.core.room.rir] and applied
with [`apply_rir`][kudio.core.room.apply_rir], before the noise goes in.
`drr_db=` sets the direct-to-reverberant ratio, 0 dB by default, about a metre
or two away ([Rooms and channels](degradation.md) has the scale). Each file
gets a room of its own, drawn from its seed, because one room reused across a
corpus would teach a model that room. The mixture keeps the clean file's
length, so the two stay sample-aligned.

The clean file stays the target in the manifest, so a model trained on
mixture → clean is asked to undo the room as well as the noise. To ask for
less, `write_targets=True` also writes the reverberant-but-clean signal to
`out_path/TARGETS/`, under the same relative path as its mixture: the speech
as the room left it, with no noise and no link. Without `rt60=` it warns and
writes nothing, since the clean file already is that signal. `TARGETS/` sits
inside `out_path`, so anything that reads `out_path` as a folder takes the
targets for mixtures; take the mixtures from the manifest instead.

The noise competes with what the room left, so the SNR is measured against
the **reverberant** speech:

```python
from pathlib import Path
import kudio

syx = kudio.Synthesizer("data/clean", "data/noise", out_path="data/rooms",
                        snr_ratio=(0, 5), rt60=(0.3, 0.8), drr_db=3.0,
                        write_targets=True)
syx.syn(mode="reg", seed=17)
kudio.save_manifest("data/rooms.json", syx.manifest())

for p in syx.manifest():
    dry, wet, mix = (kudio.file_load(f)[0]
                     for f in (p.clean, p.target, p.noisy))
    noise = mix - wet
    name = Path(p.noisy).relative_to("data/rooms").as_posix()
    print(f"{name:30} against wet {kudio.snr(wet, wet + noise):5.2f} dB,"
          f" against dry {kudio.snr(dry, dry + noise):5.2f} dB")
```

```text
f01/sa1_hiss_0dB_rt300ms.wav   against wet  0.00 dB, against dry  1.92 dB
f01/sa1_hiss_0dB_rt800ms.wav   against wet  0.00 dB, against dry  0.94 dB
f01/sx133_hum_5dB_rt300ms.wav  against wet  5.00 dB, against dry  5.35 dB
f01/sx133_hum_5dB_rt800ms.wav  against wet  5.00 dB, against dry  8.08 dB
m02/sx133_hiss_5dB_rt300ms.wav against wet  5.00 dB, against dry  5.89 dB
m02/sx133_hiss_5dB_rt800ms.wav against wet  5.00 dB, against dry  5.66 dB
```

Measured against the dry file, the same six labels would be 0.35 to 3.08 dB
off, by an amount that changes from file to file.

## A phone line or codec: `channel=`

`channel=` sends every finished mixture down a link, given as a preset name
from [`CHANNEL_PRESETS`][kudio.effects.channel.CHANNEL_PRESETS] or a dict of
[`CHANNEL_DEFAULTS`][kudio.effects.channel.CHANNEL_DEFAULTS] keys
(`{"preset": "voip", "loss": 0.1}`). It leaves the `TARGETS/` files alone:
asking a model to reproduce a codec is not denoising.

The link is not an axis. `snr_ratio` and `rt60` are swept and multiply the
file count, while `channel` is one setting for the whole run and stays out of
the file names, since a corpus was recorded over a phone line or it was not.
To compare two links, run twice into two folders. The manifest tags every row
with the link: `"telephone"`, or `"mu_law+loss10%x4-hold"` for the dict above.

## Seeds, and why the pool cannot change a file

`seed=` is a base, not the seed of one generator shared by the whole run.
Each file draws its noise crop, its room and its packet loss from the base and
its own path below `out_path` (the path rather than the name, since both
talkers here have an `sx133.wav` and should not share a room). So the same
request rebuilds the same files byte for byte, and neither the order they are
written in nor the process that writes them can change one. With no `seed=`,
the base is drawn from numpy's global RNG, which is never reseeded, and kept
in [`.seed`][kudio.core.synth.Synthesizer.seed], so a run nobody seeded can
still be rebuilt:

```python
from pathlib import Path
import numpy as np
import kudio

# seeded only so that the number printed below is the same every time
np.random.seed(2026)
first = kudio.Synthesizer("data/clean", "data/noise", out_path="data/run1",
                          rt60=0.5, channel="voip")
first.syn(mode="inc")         # no seed
print("drew", first.seed)

again = kudio.Synthesizer("data/clean", "data/noise", out_path="data/run2",
                          rt60=0.5, channel="voip")
again.syn(mode="inc", seed=first.seed)

run1 = sorted(Path("data/run1").rglob("*.wav"))
run2 = sorted(Path("data/run2").rglob("*.wav"))
print(len(run1), "files, identical:",
      [f.read_bytes() for f in run1] == [f.read_bytes() for f in run2])
```

```text
drew 942082305
18 files, identical: True
```

`is_pool=True` writes through 8 worker processes once a run has more than 200
files, and in turn below that. On this corpus with `snr_ratio=range(-10, 26)`
in `inc` mode, a pooled run and a serial one give the same 216 files, byte for
byte. On Windows and macOS, where Python starts each worker as a fresh
interpreter, the workers begin by importing your script again, so the call
has to sit under `if __name__ == "__main__":`. Without that guard every worker
fails as it starts, the pool keeps replacing them, and the run hangs.

## Writing the files

`subtype=` sets how the files are written, `"PCM_16"` by default; one a WAV
file cannot hold is refused when the Synthesizer is built. A mixture is a sum
and can land past full scale. A fixed-point subtype clips it there
([`save_wave`][kudio.core.io.save_wave] clips before writing, so nothing wraps
around), and `PCM_16` is also rounded to the nearest step.
[`syn()`][kudio.core.synth.Synthesizer.syn] then warns with the count and the
worst peak, and [`.clipped`][kudio.core.synth.Synthesizer.clipped] maps each
of those files to its peak. `"FLOAT"` keeps the samples as they are:

```python
import kudio

y, sr = kudio.file_load("data/clean/f01/sa1.wav")
kudio.save_wave("data/loud/sa1.wav", kudio.normalize(y, peak=0.9), sr)

for subtype in ("PCM_16", "FLOAT"):
    syx = kudio.Synthesizer("data/loud", "data/noise",
                            out_path=f"data/loud_{subtype}",
                            snr_ratio=(0, 10), subtype=subtype)
    syx.syn(mode="inc", seed=17)
    print(subtype, sorted(round(peak, 2) for peak in syx.clipped.values()))
```

```text
PCM_16 [1.05, 1.26, 1.37, 2.07]
FLOAT []
```

The `PCM_16` run also warns that "4 of 4 file(s) went past full scale and
were clipped when written as PCM_16 (worst peak 2.07, +6.3 dBFS)", and
suggests a lower clean level or `subtype='FLOAT'`.

`target_sr=` on `syn()` resamples the output; the default keeps each clean
file's own rate. A stereo clean file is mixed down to mono when it is read.

## The manifest

A [`Pair`][kudio.core.dataset.Pair] per file: the mixture, the clean file
behind it, and what was done in between.
[`save_manifest`][kudio.core.dataset.save_manifest] writes them as a plain
JSON list that any tool can read without kudio, and
[`load_manifest`][kudio.core.dataset.load_manifest] reads them back. One row of
the `data/rooms.json` written above:

```json
{
  "noisy": "data/rooms/f01/sa1_hiss_0dB_rt300ms.wav",
  "clean": "data/clean/f01/sa1.wav",
  "noise": "hiss",
  "snr_db": 0,
  "rt60": 0.3,
  "channel": null,
  "target": "data/rooms/TARGETS/f01/sa1_hiss_0dB_rt300ms.wav"
}
```

Only `noisy` and `clean` are required and unknown keys are dropped, so a JSON
list written by another tool loads as long as each row has those two.

!!! note "Relative paths stay relative"
    Paths are written with forward slashes, which Windows reads as well as
    Linux and macOS, so a manifest made on one system loads on the others. A
    relative `out_path` or input still gives relative paths, read from the
    folder the script ran in rather than from the JSON file. Load the manifest
    from that folder, or pass absolute paths when it will be read from
    elsewhere; otherwise [`Pair.exists()`][kudio.core.dataset.Pair.exists] is
    `False`.

`is_silence=True` on [`syn()`][kudio.core.synth.Synthesizer.syn] also writes
about `p_silence` of the outputs again as noise alone, named `..._n00.wav`:
2% by default, rounded to whole files and never fewer than one, so the 18
outputs of the first run come with one. Those files are in what `syn()`
returns, with `snr_db` 0, and not in the manifest: there is no speech in them,
so the clean file they were drawn for is not their reference.

## Train, validation and test

[`split_pairs`][kudio.core.dataset.split_pairs] shuffles once with
`numpy.random.default_rng(seed)` and cuts the rows into
`(train, validation, test)`. Pass `seed=`: the default, `None`, gives a
different split on every call. If the shares would leave nothing to train on,
the validation and test sets are given up instead.

```python
import kudio

pairs = kudio.load_manifest("data/mixed.json")
train, val, test = kudio.split_pairs(pairs, 0.2, 0.2, seed=17)
print(len(train), len(val), len(test))
shared = {p.clean for p in train} & {p.clean for p in test}
print(len(shared), "of 3 clean files in both train and test")
```

```text
10 4 4
2 of 3 clean files in both train and test
```

That second line is the catch. `split_pairs` deals out rows, not talkers,
utterances or noises, and one clean file at three SNRs is three rows that can
land in different splits. Each test row is a mixture the model has not seen,
but of a talker and a noise it usually has: here both noises are in train and
test, and 2 of the 4 test rows have their exact noise and SNR in train as
well. Such a split asks whether a model learnt more than its training
mixtures, and says nothing about a talker or a noise it has never heard. For
those, split the clean files (or the noise files) first and synthesise each
split into its own folder; a list input and `root=` make that a loop:

```python
from pathlib import Path
import kudio

talkers = {"train": ["f01"], "test": ["m02"]}
for split, who in talkers.items():
    files = [f for t in who
             for f in sorted(Path("data/clean", t).glob("*.wav"))]
    syx = kudio.Synthesizer(files, "data/noise", out_path=f"data/{split}",
                            root="data/clean", snr_ratio=(-5, 0, 5))
    syx.syn(mode="inc", seed=17)
    kudio.save_manifest(f"data/{split}.json", syx.manifest())
    print(split, len(syx.manifest()), "rows")
```

```text
train 12 rows
test 6 rows
```

A row listed twice, which a value repeated in `snr_ratio=` produces, would be
trained on and then scored as though it had been held out, so `split_pairs`
drops exact repeats with a warning. Two rows that name one mixture and
disagree about anything else raise `ValueError`.

## Noise over a background: `syn_extra_mode`

[`syn_extra_mode(background_path, background_snr)`][kudio.core.synth.Synthesizer.syn_extra_mode]
mixes in two stages. Every noise is first mixed with every background file at
each `background_snr`; then every clean file is mixed with each of those
mixtures, each plain noise and each background, at every SNR. With 2 clean
files, 2 noises, 1 background at one `background_snr` and 2 SNRs, that is
2 × (2 + 2 + 1) × 2 = 20 rows. A 2% share, at least one file, is written
again as noise alone, so `syn_extra_mode` returns 21 rows here and the
manifest holds 20. Each clean file is also copied in unchanged as
`{stem}_n0.wav`, the speech alone where `_n00` is the noise alone; those
copies are in neither.

The output folder is emptied first, under the same guard as
[`syn()`][kudio.core.synth.Synthesizer.syn], but the rest differs:

- Rooms and links are not applied.
- Names carry `dB` on both signs, `n5dB` and `5dB`, where `syn()` writes `n5`.
  Corpora on disk already use both spellings, so neither changes; match on
  the manifest's `snr_db` rather than on the name.
- The output is flat: clean files that share a name, such as two talkers'
  `sx133.wav`, write to the same output names. It warns, and the second
  overwrites the first, so give this mode clean files whose names are
  distinct.
- It takes no `target_sr=`: each output stays at its clean file's rate,
  whatever an earlier `syn()` on the same Synthesizer asked for. The noise is
  still read at the rate of the file it is mixed into, as in `syn()`.
- Above 200 outputs it writes through the 8-process pool unless you pass
  `is_pool=False`, and then needs the `if __name__ == "__main__":` guard on
  Windows and macOS.

## From the command line

`kudio synth` builds the same corpora without a script. Its `--mode` defaults
to `inc`, where [`syn()`][kudio.core.synth.Synthesizer.syn] defaults to
`regular`. It empties `--out` first, under the same guard as `syn()`, and
prints the seed it used, so a run without `--seed` can be rebuilt with the
number it printed:

```bash
kudio synth --clean data/clean --noise data/noise --out data/cli \
    --snr -5 0 5 --seed 17 --rt60 0.3 0.8 --targets --manifest data/cli.json
```

That writes 36 mixtures, 36 targets under `data/cli/TARGETS/` and the
manifest. The [command line](../cli.md) page lists every option.

[Denoising](enhancement.md) and [Measuring quality](metrics.md) take it from
there: the mixtures are the input, and the clean files in the manifest are
the references to score against.
