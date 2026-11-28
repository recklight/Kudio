# Speech activity and pitch

[`vad`][kudio.core.vad.vad] and its relatives find where somebody is talking,
with nothing beyond numpy. [`f0`][kudio.core.pitch.f0] tracks the pitch with
pYIN and keeps the voiced/unvoiced decision beside it.

| To | Use |
|---|---|
| find where somebody is talking, as `(start, end)` in seconds | [`vad`][kudio.core.vad.vad] |
| cut each stretch of speech out as its own clip | [`vad_split`][kudio.core.vad.vad_split] |
| drop everything before the first word and after the last | [`vad_trim`][kudio.core.vad.vad_trim] |
| find the recordings in a folder with no speech in them | [`speech_ratio`][kudio.core.vad.speech_ratio] |
| mark each spectrogram frame as speech or not | [`vad_frames`][kudio.core.vad.vad_frames] |
| track the pitch, and where there is one | [`f0`][kudio.core.pitch.f0], a [`PitchTrack`][kudio.core.pitch.PitchTrack] |
| open either result in an editor | [`Label`][kudio.core.dataset.Label], [`save_labels`][kudio.core.dataset.save_labels] |

The examples run on three files this block makes. Run it in an empty folder.

??? example "The recording used on this page"

    Three voiced words with a gliding pitch, an /s/ running into the second
    one, and a loud broadband clatter that is not speech: `quiet.wav` in a
    quiet room, `fan.wav` in front of a fan, `empty.wav` the clatter and the
    fan with nobody talking.

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

## Where somebody is talking

```python
for name in ("quiet.wav", "fan.wav"):
    y, sr = kudio.file_load(name)
    spans = kudio.vad(y, sr)
    parts = kudio.split_on_silence(y)
    print(f"{name:9}", [(round(a, 2), round(b, 2)) for a, b in spans], len(parts), "parts")
```

```text
quiet.wav [(0.73, 1.67), (2.94, 4.04), (4.53, 5.47)] 4 parts
fan.wav   [(0.77, 1.64), (2.99, 4.01), (4.64, 5.36)] 1 parts
```

[`vad`][kudio.core.vad.vad] returns `(start, end)` pairs in seconds, an empty
list when nothing looks like speech, and finds the three words in both rooms
with nothing changed between them. A level threshold such as
[`split_on_silence`][kudio.effects.silence.split_on_silence] finds four parts in
the quiet room, the clatter among them, and one six-second part behind the fan,
whose level sits within 30 dB of the loudest frame. Use that for sounds that are
not speech in a quiet room.

![Frame level of both files with the top_db line and the vad spans](../images/speech-level-light.webp#only-light)
![Frame level of both files with the top_db line and the vad spans](../images/speech-level-dark.webp#only-dark)

## Cutting, trimming and counting

```python
y, sr = kudio.file_load("fan.wav")
for n, part in enumerate(kudio.vad_split(y, sr), 1):
    kudio.save_wave(f"spans/fan_{n:03d}.wav", part, sr)
clip, (start, end) = kudio.vad_trim(y, sr)
print(start, end, round(len(clip) / sr, 2))
for name in ("empty.wav", "fan.wav", "quiet.wav"):
    print(f"{name:10} {kudio.speech_ratio(kudio.file_load(name)[0], sr):4.0%}")
```

```text
12320 85760 4.59
empty.wav    0%
fan.wav     43%
quiet.wav   50%
```

[`vad_split`][kudio.core.vad.vad_split] returns the spans as waveforms,
[`vad_trim`][kudio.core.vad.vad_trim] keeps first speech to last as
`(clip, (start, end))` in samples, and [`speech_ratio`][kudio.core.vad.speech_ratio]
gives the share of speech. With no speech, `vad_trim` returns the clip untouched
and logs it at INFO. `empty.wav` holds a clatter louder than any word and still
reads 0 %. `vad` takes its noise floor from the file, so run these on whole
recordings: a clip already cut to the words is its own floor. `kudio vad
fan.wav --split-to spans/` does the same from a shell and exits 1 when there is
no speech.

## One decision per spectrogram frame

```python
spec = kudio.waveform_to_spectrogram(y)        # (frames, bins), hop 256
speech = kudio.vad_frames(y, sr)               # (frames,), the same frames
print(spec.shape, speech.shape, speech.sum())
background = np.median(spec[~speech], axis=0)  # per bin, over what vad turned down
```

```text
(376, 257) (376,) 144
```

[`vad_frames`][kudio.core.vad.vad_frames] gives one boolean per STFT frame,
`True` where the frame's centre falls inside a span, on the frames of
[`waveform_to_spectrogram`][kudio.core.feature.waveform_to_spectrogram] at its
defaults; pass the same `hop_length`, `n_fft` or `center` if you change them.
`~speech` includes the clatter too, which is why the example takes a median.

## How vad decides

[`vad`][kudio.core.vad.vad] looks at 30 ms frames every 10 ms. A frame has to
sit [`THRESHOLD_DB`][kudio.core.vad.THRESHOLD_DB] (8 dB) above the noise floor,
the 15th percentile of the frame levels, and no more than
[`FLOOR_REL_DB`][kudio.core.vad.FLOOR_REL_DB] (45 dB) under the loudest frame.
Then it has to look like a voice: spectral flatness under
[`FLATNESS_MAX`][kudio.core.vad.FLATNESS_MAX] (0.40; vowels are under 0.001,
a 30 ms frame of white noise about 0.56), or a zero-crossing rate above
[`ZCR_MIN`][kudio.core.vad.ZCR_MIN] (0.25) within
[`FRICATIVE_REACH`][kudio.core.vad.FRICATIVE_REACH] (0.2 s) of a frame that
passed, for fricatives. Gaps up to `min_silence` are bridged, spans shorter than
`min_speech` dropped, and `pad` added to each end.

| Argument | Default | Change it when |
|---|---|---|
| `threshold_db` | 8.0 dB | a quiet talker is missed (lower it), or a noisy room leaks in (raise it) |
| `flatness_max` | 0.40 | the audio is not speech (1.0 switches the test off) |
| `min_speech` | 0.10 s | clicks and knocks come through as short spans (raise it) |
| `min_silence` | 0.15 s | words split at their stops (raise it), or separate phrases merge (lower it) |
| `pad` | 0.05 s | the cuts clip the first or last consonant (raise it) |
| `band` | `None`, the whole band | a high-rate recording has a strong floor above the speech band: `(100, 4000)` |

Where it goes wrong: noise that starts partway through a file and covers less
than about 85 % of it is taken for speech, so cut the file where the background
changes; noise that fills only part of the spectrum, a rumble below 500 Hz or a
hiss above 3.5 kHz, passes the flatness test; broadband noise right next to a
word joins it; and a clip needs something quieter than the speech in 15 % of its
frames. A multichannel array raises [`FeatureError`][kudio.exceptions.FeatureError].

## Pitch with f0

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

[`f0`][kudio.core.pitch.f0] runs pYIN[^pyin] and returns a
[`PitchTrack`][kudio.core.pitch.PitchTrack]: `track.f0` in Hz, `NaN` wherever a
frame is unvoiced, and `track.voiced`, the decision itself, made with a hidden
Markov model over the clip so it does not flicker within a syllable. The hop is
10 ms unless you set `hop_length`. A minute of 16 kHz audio takes about 0.8 s,
so [`speech_ratio`][kudio.core.vad.speech_ratio] is the quicker question when
all you want is whether there is a voice.

![Spectrogram of fan.wav with the f0 contour drawn over it](../images/speech-pitch-light.webp#only-light)
![Spectrogram of fan.wav with the f0 contour drawn over it](../images/speech-pitch-dark.webp#only-dark)

The search range is 65 to 400 Hz ([`SPEECH_FMIN`][kudio.core.pitch.SPEECH_FMIN]
and `SPEECH_FMAX`), and a pitch outside it is reported as something inside:

```python
t = np.arange(sr) / sr
tone = 0.3 * sum(np.sin(2 * np.pi * 500 * k * t) / k for k in range(1, 10))
print(kudio.f0(tone, sr))                  # the speech range, 65-400 Hz
print(kudio.f0(tone, sr, fmax=1000))       # a range that holds 500 Hz
```

```text
median 249.7 Hz  ·  249.7-249.7 Hz (0.0 semitones)  ·  100% voiced
median 499.4 Hz  ·  499.4-499.4 Hz (0.0 semitones)  ·  100% voiced
```

The first line is an octave low, and a median sitting on `fmin` is the other
sign to look for; set `fmin` and `fmax` for singing or a child's voice. An
`fmin` not between 0 and `fmax`, an `fmax` above Nyquist and a `frame_length`
shorter than `2 * floor(sr / fmin) + 2` raise `FeatureError`. Left out,
`frame_length` is 2048, or 4096 at 88.2 and 96 kHz.

`print(track)` shows the median and the 5th to 95th percentile range, which one
octave error cannot move; [`summary()`][kudio.core.pitch.PitchTrack.summary]
gives them as a dict. pYIN's voicing does not look at level, so to keep only the
pitch inside the speech, combine it with `vad_frames` at the track's hop:
`track.f0[track.voiced & kudio.vad_frames(y, sr, hop_length=track.hop_length)]`.

## Labels: opening the result in an editor

```python
speech = [kudio.Label(a, b, f"speech {n}")
          for n, (a, b) in enumerate(kudio.vad(y, sr), 1)]
kudio.save_labels("fan.txt", speech)
kudio.save_labels("fan.voiced.txt", track.to_labels())
print(open("fan.voiced.txt").read())
```

```text
0.770000	1.650000	voiced 1
3.080000	3.990000	voiced 2
4.570000	5.460000	voiced 3
```

[`save_labels`][kudio.core.dataset.save_labels] writes an Audacity label track;
in Audacity, open `fan.wav` and add the files with *File › Import › Labels*.
[`to_labels()`][kudio.core.pitch.PitchTrack.to_labels] leaves out voiced runs
shorter than 0.05 s (`min_seconds`). Reading a corrected track back is on
[Editing and effects](editing.md). `kudio pitch fan.wav --segments --labels
fan.voiced.txt` prints the summary and the voiced spans, writes the label track,
and exits 1 when nothing is voiced.

[^pyin]: M. Mauch and S. Dixon, "pYIN: A fundamental frequency estimator using probabilistic threshold distributions", Proc. ICASSP 2014, pp. 659-663, doi:10.1109/ICASSP.2014.6853678.
