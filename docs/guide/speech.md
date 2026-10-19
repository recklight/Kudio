# Speech activity and pitch

Two questions come before almost anything else I do with a speech recording:
where in it somebody is talking, and what pitch they are on.
[`vad`][kudio.core.vad.vad] (voice activity detection) and its relatives
answer the first, with nothing beyond numpy. [`f0`][kudio.core.pitch.f0]
answers the second: it tracks the fundamental frequency with librosa's pYIN
(probabilistic YIN) and keeps the voiced/unvoiced decision next to the pitch.

| To | Use |
|---|---|
| find where somebody is talking, as `(start, end)` in seconds | [`vad`][kudio.core.vad.vad] |
| cut each stretch of speech out as its own clip | [`vad_split`][kudio.core.vad.vad_split] |
| drop everything before the first word and after the last | [`vad_trim`][kudio.core.vad.vad_trim] |
| find the recordings in a folder with no speech in them | [`speech_ratio`][kudio.core.vad.speech_ratio] |
| mark each spectrogram frame as speech or not | [`vad_frames`][kudio.core.vad.vad_frames] |
| track the pitch, and where there is one | [`f0`][kudio.core.pitch.f0], which returns a [`PitchTrack`][kudio.core.pitch.PitchTrack] |
| open either result in an editor | [`Label`][kudio.core.dataset.Label], [`save_labels`][kudio.core.dataset.save_labels] |

Every example below runs on one synthetic recording. Run the code in the box
once, in an empty folder. The blocks after it continue from there, with numpy
and kudio imported and the three files in place, and print what the page
shows.

??? example "The recording used on this page"

    Three voiced words with a gliding pitch, an /s/ running into the second
    one, and a loud broadband clatter that is not speech. `quiet.wav` puts them
    in a quiet room, `fan.wav` in front of a fan, and `empty.wav` holds the
    clatter and the fan with nobody talking.

    ```python
    import numpy as np
    import kudio

    sr = 16000
    n = 6 * sr
    rng = np.random.default_rng(7)

    def at(start, x):                      # place x at `start` seconds
        out = np.zeros(n)
        out[int(start * sr):int(start * sr) + len(x)] = x
        return out

    def vowel(seconds, f_from, f_to, level=1.0):   # harmonics of a gliding f0
        m = int(seconds * sr)
        phase = 2 * np.pi * np.cumsum(np.linspace(f_from, f_to, m)) / sr
        tone = sum(np.sin(k * phase) / k for k in range(1, 16))
        return 0.08 * level * tone * np.sin(np.pi * np.arange(m) / m) ** 0.5

    def hiss(seconds, level, cutoff=None):         # a noise burst
        m = int(seconds * sr)
        x = rng.standard_normal(m)
        if cutoff:
            x = kudio.highpass(x, sr, cutoff=cutoff)
        return level * x * np.hanning(m)

    voice = (at(0.8, vowel(0.8, 140, 110))
             + at(3.0, hiss(0.15, 0.05, cutoff=3500))      # an /s/ ...
             + at(3.12, vowel(0.85, 120, 165))             # ... running into a vowel
             + at(4.6, vowel(0.8, 130, 100, level=0.5)))   # a quieter last word
    clatter = at(2.2, hiss(0.25, 0.25))                    # loud, broadband, not speech
    room = 0.0005 * rng.standard_normal(n)                 # a quiet room
    fan = 0.03 * kudio.lowpass(rng.standard_normal(n), sr, cutoff=1000)

    kudio.save_wave("quiet.wav", voice + clatter + room, sr)
    kudio.save_wave("fan.wav", voice + clatter + fan, sr)
    kudio.save_wave("empty.wav", clatter + fan, sr)       # nobody talking
    ```

## Why a silence threshold is not enough

[`split_on_silence`][kudio.effects.silence.split_on_silence] and
[`trim_silence`][kudio.effects.silence.trim_silence] (see
[Editing and effects](editing.md)) decide on level alone: a frame is silence
when it sits more than `top_db` under the loudest frame of the file, 30 dB by
default. Here is the same voice in both rooms:

```python
for name in ("quiet.wav", "fan.wav"):
    y, sr = kudio.file_load(name)
    for top_db in (30, 25, 20):
        parts = kudio.split_on_silence(y, top_db=top_db)
        print(f"{name:9}  top_db={top_db}  ", [round(len(p) / sr, 2) for p in parts])
```

```text
quiet.wav  top_db=30   [0.9, 0.32, 1.02, 0.83]
quiet.wav  top_db=25   [0.83, 0.32, 0.99, 0.77]
quiet.wav  top_db=20   [0.77, 0.29, 0.93, 0.61]
fan.wav    top_db=30   [6.0]
fan.wav    top_db=25   [0.9, 0.32, 1.02, 0.83]
fan.wav    top_db=20   [0.8, 0.29, 0.96, 0.64]
```

In the quiet room the default finds four parts. In front of the fan it finds
one, six seconds long: the whole file. The fan sits about 26 dB under the
loudest frame, so at `top_db=30` it counts as sound. `top_db=25` happens to
rescue this file, but the right number depends on how loud this fan is next
to the loudest moment of this take, and that moment is the clatter, not the
voice. Going lower to be safe eats into the words: at 20 the quiet last word
is down from 0.83 s to 0.64 s behind the fan, and to 0.61 s in the quiet
room. And wherever the file splits at all, the second part is the clatter,
because loud is all `split_on_silence` asks.

![Frame level of both files with the top_db line and the vad spans](../images/speech-level-light.webp#only-light)
![Frame level of both files with the top_db line and the vad spans](../images/speech-level-dark.webp#only-dark)

This is the level `split_on_silence` measures, each frame against the loudest
one. Behind the fan nothing drops under the dashed `top_db` line. The shaded
spans are what [`vad`][kudio.core.vad.vad] finds, with the same settings in
both rooms:

```python
for name in ("quiet.wav", "fan.wav"):
    y, sr = kudio.file_load(name)
    print(f"{name:9}", [(round(a, 2), round(b, 2)) for a, b in kudio.vad(y, sr)])
```

```text
quiet.wav [(0.73, 1.67), (2.94, 4.04), (4.53, 5.47)]
fan.wav   [(0.77, 1.64), (2.99, 4.01), (4.64, 5.36)]
```

Three spans in each room, no clatter, nothing tuned between the two runs. The
result is `(start, end)` pairs in seconds, and an empty list when nothing
looks like speech. Behind the fan the last word comes out shorter, 4.64 to
5.36 s against 4.53 to 5.47 s: its onset and decay are too close to the fan's
level for the 8 dB test described [below](#how-vad-decides).
`threshold_db=6` brings it back to 4.61 to 5.40 s with no new span; at 5 the
clatter comes back too, as 2.17 to 2.50 s.

`split_on_silence` is still the tool for something that is not speech in a
quiet room, such as a knock. For speech, use `vad`.

## Cutting, trimming and counting

[`vad_split`][kudio.core.vad.vad_split] returns the spans as waveforms,
[`vad_trim`][kudio.core.vad.vad_trim] keeps everything from the first speech
to the last, and [`speech_ratio`][kudio.core.vad.speech_ratio] says how much
of the clip is speech. All three take the same keyword arguments as `vad`.

```python
y, sr = kudio.file_load("fan.wav")

for n, part in enumerate(kudio.vad_split(y, sr), 1):
    kudio.save_wave(f"spans/fan_{n:03d}.wav", part, sr)

clip, (start, end) = kudio.vad_trim(y, sr)
print(start, end, round(len(clip) / sr, 2))
```

```text
12320 85760 4.59
```

`vad_trim` returns `(clip, (start, end))` in samples, the same shape as
[`trim_silence`][kudio.effects.silence.trim_silence], so the code that uses
the result works with either. When it finds no speech it returns the clip
untouched with its full span, `(0, 96000)` for `empty.wav`, and logs that at
INFO. Deleting the audio is never the safer reading of "nothing found".

`speech_ratio` is the check I run over a folder of recordings before it
becomes training data:

```python
from pathlib import Path

for path in sorted(Path(".").glob("*.wav")):
    y, sr = kudio.file_load(path)
    print(f"{path.name:10} {kudio.speech_ratio(y, sr):4.0%}")
```

```text
empty.wav    0%
fan.wav     43%
quiet.wav   50%
```

`empty.wav` holds a clatter louder than any word in the other two and still
reads 0 %. A whole recording near zero in a folder that should be speech is a
recording of a room, and it should not reach a
[noisy dataset](synthesis.md) as a clean target.

It needs some background in the file to measure against. `vad` takes its
noise floor from the file itself, so in a clip that is speech from end to end
the floor is the speech: the three files `vad_split` wrote above read 0 %,
35 % and 0 %. Run it on whole recordings, not on clips already cut to the
words.

From a shell, `kudio vad fan.wav --split-to spans/` prints the same spans,
writes each to its own file, and exits 1 when there is no speech; see
[Command line](../cli.md).

## One decision per spectrogram frame

A model trained on spectrogram frames (see [STFT and features](features.md)),
or a noise estimate taken from the frames where nobody talks (see
[Denoising](enhancement.md)), needs the decision per frame rather than per
span. [`vad_frames`][kudio.core.vad.vad_frames] gives one boolean per STFT
frame, `True` where the frame's centre falls inside a span:

```python
y, sr = kudio.file_load("fan.wav")
spec = kudio.waveform_to_spectrogram(y)        # (frames, bins), hop 256
speech = kudio.vad_frames(y, sr)               # (frames,), the same frames
print(spec.shape, speech.shape, speech.sum())

background = np.median(spec[~speech], axis=0)  # per bin, over what vad turned down
```

```text
(376, 257) (376,) 144
```

At its defaults it gives the frames of
[`waveform_to_spectrogram`][kudio.core.feature.waveform_to_spectrogram] and
[`STFT`][kudio.core.stft.STFT] at theirs. If you change `hop_length`, `n_fft`
or `center` there, pass the same values here.

Not speech is not the same as background. `~speech` is everything `vad`
turned down, and in `fan.wav` that includes the clatter: 18 of the 232 frames
overlap it. Leaving those 18 out moves a mean over the frames by 3.5 to 4.6 dB
above 2 kHz, where the fan is weak, and the median by 1.1 dB at most, which is
why the example takes the median.

## How vad decides

[`vad`][kudio.core.vad.vad] looks at 30 ms frames every 10 ms
([`FRAME`][kudio.core.vad.FRAME]). A frame first has to be loud for this
file: the noise floor is the 15th percentile of the frame levels, and the
frame must sit [`THRESHOLD_DB`][kudio.core.vad.THRESHOLD_DB] (8 dB) above it.
A fan that runs through the whole recording *is* that floor, which is why
`fan.wav` needed nothing changed. Anything more than
[`FLOOR_REL_DB`][kudio.core.vad.FLOOR_REL_DB] (45 dB) under the loudest frame
is never speech, so a near-silent file does not get its own hiss promoted.

A loud frame then has to look like a voice. Spectral flatness, the geometric
over the arithmetic mean of the frame's power spectrum, is under 0.001 for the
vowels here. The textbook value for white noise is 1.0, but one 30 ms frame of
it measures 0.46 to 0.66, 0.56 on average, and the clatter 0.53 to 0.61.
Under [`FLATNESS_MAX`][kudio.core.vad.FLATNESS_MAX] (0.40) a loud frame counts
by itself, and this is the test the clatter failed.

A loud frame that fails it still counts when its zero-crossing rate is above
[`ZCR_MIN`][kudio.core.vad.ZCR_MIN] (0.25: a quarter of neighbouring samples
change sign, as in a 2 kHz sine at 16 kHz) and a frame that passed lies
within [`FRICATIVE_REACH`][kudio.core.vad.FRICATIVE_REACH] (0.2 s).
Zero-crossing rate cannot tell a broadband fricative from a burst of hiss;
standing next to a vowel can. The /s/ in this recording does not need the
rule: it is band-limited, its flatness is 0.045, and it passes on its own.

Last, gaps up to `min_silence` are bridged, since a stop consonant is a real
silence inside a word, spans shorter than `min_speech` are dropped, and `pad`
is added to each end.

| Argument | Default | Change it when |
|---|---|---|
| `threshold_db` | 8.0 dB | a quiet talker is missed (lower it), or a noisy room leaks in (raise it) |
| `flatness_max` | 0.40 | the audio is not speech (1.0 switches the test off) |
| `min_speech` | 0.10 s | clicks and knocks come through as short spans (raise it) |
| `min_silence` | 0.15 s | words split at their stops (raise it), or separate phrases merge (lower it) |
| `pad` | 0.05 s | the cuts clip the first or last consonant (raise it) |

White noise sits close above the default flatness limit, so raise it with
care: on `quiet.wav`, 0.5 still finds the three words alone, and 0.6 brings
the clatter back as 2.14 to 2.51 s. The [reference][kudio.core.vad.vad] has
every argument. `vad` raises [`FeatureError`][kudio.exceptions.FeatureError]
on a multichannel array; [`file_load`][kudio.core.io.file_load] mixes down to
mono unless you pass `mono=False`.

### Where it goes wrong

I tried it on synthetic audio like the recording above, and these are the
cases to know about before a corpus finds them.

Coloured noise that starts partway through a file is taken for speech. Noise
becomes the floor only when it covers about 85 % of the file or more. In a
four-second quiet recording, a fan switched on at 0.6 s was ignored; switched
on at 0.7 s it was reported as speech from 0.63 s to the end, and at 2 s from
1.93 s. Cut the file where the background changes and run `vad` on each part.

Flatness is measured over the whole spectrum, and noise that fills only part
of it is uneven too. Alone in a quiet room, a 0.5 s rumble below 500 Hz and a
0.15 s hiss above 3.5 kHz were both reported as speech. Only noise spread
across the whole band fails the flatness test, which is also why white noise
switched on at 2 s was not reported.

Broadband noise right next to a word is joined to it, which is the consonant
rule doing its job: a burst of white noise that ended where a vowel began
became part of the vowel's span. With 0.4 s between them, it was dropped.

## Pitch with f0

[`f0`][kudio.core.pitch.f0] runs pYIN (Mauch & Dixon 2014) through librosa
and returns a [`PitchTrack`][kudio.core.pitch.PitchTrack]. It costs far more
than `vad`: on my machine a minute of 16 kHz audio takes 1.3 to 2.6 s through
`f0` and 0.04 to 0.07 s through `vad`, so to ask whether a file has a voice in
it at all, [`speech_ratio`][kudio.core.vad.speech_ratio] is the quicker
question.

```python
y, sr = kudio.file_load("fan.wav")
track = kudio.f0(y, sr)
print(track)
print(len(track), track.hop_length, track.times[75:80])
print(track.f0[75:80].round(1))
```

```text
median 125.6 Hz  ·  103.8-158.2 Hz (7.3 semitones)  ·  45% voiced
601 160 [0.75 0.76 0.77 0.78 0.79]
[  nan   nan 140.1 140.1 139.3]
```

An estimator asked for the pitch of a frame always gives one, for silence,
for the clatter and for the /s/ as well, and a contour drawn through those
answers is noise with a line through it. pYIN also decides whether each frame
is voiced, with a hidden Markov model over the whole clip, so the decision
does not flicker within a syllable. `track.f0` is in Hz and `NaN` wherever
the frame is unvoiced, which is why the first two values above are `nan` (the
first word's voicing starts at 0.77 s) and why a plot of it has gaps;
`track.voiced` holds the decision itself. The hop is 10 ms unless you set
`hop_length`.

![Spectrogram of fan.wav with the f0 contour drawn over it](../images/speech-pitch-light.webp#only-light)
![Spectrogram of fan.wav with the f0 contour drawn over it](../images/speech-pitch-dark.webp#only-dark)

Drawn over the spectrogram of `fan.wav`, the contour follows each glide and
breaks off wherever there is only the clatter, the /s/ or the fan.

`f0` takes one channel and, like `vad`, raises `FeatureError` on a
multichannel array. Read as one interleaved signal, `fan.wav` as two identical
columns would be a clip twice as long at half the pitch, 1201 frames at a
median of 65.0 Hz against the 601 frames at 125.6 Hz above.

pYIN measures periodicity, not the lowest line in a spectrogram. A 120 Hz
harmonic tone through [`bandpass`][kudio.effects.filters.bandpass]'s default
300 to 3400 Hz, the telephone band, comes out with its fundamental 68 dB down
and still reads 119.9 Hz.

### The search range is a decision

The default range is 65 to 400 Hz
([`SPEECH_FMIN`][kudio.core.pitch.SPEECH_FMIN] and `SPEECH_FMAX`), from a low
male voice to a high female one. pYIN only looks inside it, and a pitch
outside it is not reported missing. It is reported as something inside:

```python
sr = 16000
t = np.arange(sr) / sr
tone = 0.3 * sum(np.sin(2 * np.pi * 500 * k * t) / k for k in range(1, 10))

print(kudio.f0(tone, sr))                  # the speech range, 65-400 Hz
print(kudio.f0(tone, sr, fmax=1000))       # a range that holds 500 Hz
```

```text
median 249.7 Hz  ·  249.7-249.7 Hz (0.0 semitones)  ·  100% voiced
median 499.4 Hz  ·  499.4-499.4 Hz (0.0 semitones)  ·  100% voiced
```

The first line is an octave low and 100 % voiced, and nothing in it looks
wrong. Below the range the answer sticks to the floor: tones at 50 and 60 Hz
both came back as 65.0 Hz, so a median sitting exactly on `fmin` is the sign
to look for. For singing or a child's voice, set the range yourself with
`fmin` and `fmax`.

These raise a [`FeatureError`][kudio.exceptions.FeatureError] before pYIN
runs: an `fmin` that is not between 0 and `fmax`, an `fmax` above Nyquist,
and a `frame_length` that does not hold more than two periods of `fmin`,
`2 * floor(sr / fmin) + 2` samples. Left out, `frame_length` is 2048, which
holds two periods of 65 Hz up to 66.56 kHz, or the next power of two that is
long enough above that: 4096 at 88.2 and 96 kHz. `kudio pitch` picks it the
same way. A `frame_length` you pass yourself is checked, not raised.

### Reading the track

`print(track)` shows the median
([`median_hz`][kudio.core.pitch.PitchTrack.median_hz]) and the 5th to 95th
percentile range ([`range_hz()`][kudio.core.pitch.PitchTrack.range_hz]),
because one octave error or one misjudged frame would move a mean or set a
minimum by itself. [`summary()`][kudio.core.pitch.PitchTrack.summary] returns
the same numbers as a dict, ready to be a CSV row.

[`voiced_ratio`][kudio.core.pitch.PitchTrack.voiced_ratio] is the share of
frames with a pitch, which is not `speech_ratio`: an /s/ is speech without a
pitch. Even so, on `fan.wav` the voiced share is the larger, 45 % against
43 %, because `vad` lost the edges of the quiet last word behind the fan.

`quiet.wav` shows why the percentiles matter. pYIN's voicing does not look at
level, and it called 0.46 to 0.55 s of the quiet room's hiss voiced, at
exactly 65.0 Hz. That becomes the minimum, while `range_hz()` still starts at
101.5 Hz. To keep only the pitch inside the speech, ask
[`vad_frames`][kudio.core.vad.vad_frames] for the track's hop; both are
centred by default, so the two line up frame for frame:

```python
y, sr = kudio.file_load("quiet.wav")
track = kudio.f0(y, sr)
speech = kudio.vad_frames(y, sr, hop_length=track.hop_length)
inside = track.f0[track.voiced & speech]       # Hz, voiced and in a vad span
print(track.f0[track.voiced].min(), round(inside.min(), 1))
```

```text
65.0 101.4
```

## Labels: opening the result in an editor

A list of tuples is hard to check by eye. Turn the spans into
[`Label`][kudio.core.dataset.Label] objects and the pitch track's voiced runs
into labels with [`to_labels()`][kudio.core.pitch.PitchTrack.to_labels], and
[`save_labels`][kudio.core.dataset.save_labels] writes each list as an
Audacity label track:

```python
y, sr = kudio.file_load("fan.wav")
speech = [kudio.Label(a, b, f"speech {n}")
          for n, (a, b) in enumerate(kudio.vad(y, sr), 1)]
kudio.save_labels("fan.txt", speech)
kudio.save_labels("fan.voiced.txt", kudio.f0(y, sr).to_labels())
```

```text title="fan.voiced.txt"
0.770000	1.650000	voiced 1
3.080000	3.990000	voiced 2
4.570000	5.460000	voiced 3
```

In Audacity, open `fan.wav` and add both files with *File › Import › Labels*;
the speech spans and the voiced runs sit as two tracks under the waveform.
`to_labels()` and
[`voiced_segments()`][kudio.core.pitch.PitchTrack.voiced_segments] leave out
voiced runs shorter than 0.05 s; `min_seconds` changes that. The label
formats, and reading a corrected track back with
[`load_labels`][kudio.core.dataset.load_labels], are on
[Editing and effects](editing.md).

`kudio pitch fan.wav --segments --labels fan.voiced.txt` does the pitch half
in one step: it prints the summary line and the voiced spans, writes the
label track, and exits 1 when nothing is voiced; see
[Command line](../cli.md).
