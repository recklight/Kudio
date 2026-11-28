# Editing and effects

`kudio.effects` is what an audio editor does to a waveform, as functions:
fades, filters, cuts and joins, trimming, level, and augmentation for training.
Each takes a NumPy array and returns a new one; none writes into the array you
pass in. The channel degradations have their own page,
[Rooms and channels](degradation.md).

| To do this | Use |
|---|---|
| stop a clip starting or ending on a click | [`fade`][kudio.effects.edit.fade] |
| take out rumble, hum, or everything outside a band | [`highpass`][kudio.effects.filters.highpass], [`lowpass`][kudio.effects.filters.lowpass], [`bandpass`][kudio.effects.filters.bandpass], [`bandstop`][kudio.effects.filters.bandstop] |
| take out a constant offset | [`remove_dc`][kudio.effects.edit.remove_dc] |
| cut a span out, or patch one in | [`splice`][kudio.effects.edit.splice] |
| drop the silence around a studio take | [`trim_silence`][kudio.effects.silence.trim_silence], [`split_on_silence`][kudio.effects.silence.split_on_silence] |
| cut a recording where a label file says | [`load_labels`][kudio.core.dataset.load_labels] |
| set the peak | [`normalize`][kudio.effects.augment.normalize], [`normalize_db`][kudio.effects.augment.normalize_db] |
| get more training data out of one clip | [`pitch_shift`][kudio.effects.augment.pitch_shift], [`time_stretch`][kudio.effects.augment.time_stretch], [`add_noise_snr`][kudio.effects.augment.add_noise_snr], [`reverb`][kudio.effects.augment.reverb], [`spec_augment`][kudio.effects.augment.spec_augment] |

??? example "The files used on this page"

    `take.wav` is six seconds of white noise at 0.01 RMS, standing in for a
    room, with two voiced phrases in it: 1.0 to 2.0 s and 3.5 to 4.8 s.
    `retake.wav` is a 0.3 s phrase to patch in, `stereo.wav` the take on two
    channels, and `take.txt` an Audacity label track marking the phrases.

    ```python
    import numpy as np
    import kudio

    sr = 16000

    def voiced(seconds, f0):
        t = np.arange(int(seconds * sr)) / sr
        return sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 12)) / 6

    y = 0.01 * np.random.default_rng(0).standard_normal(6 * sr)    # the room
    y[1 * sr:2 * sr] += voiced(1.0, 120)
    y[int(3.5 * sr):int(4.8 * sr)] += voiced(1.3, 150)
    kudio.save_wave("take.wav", y.astype(np.float32), sr)
    kudio.save_wave("retake.wav", (0.5 * voiced(0.3, 130)).astype(np.float32), sr)
    kudio.save_wave("stereo.wav", np.stack([y, 0.5 * y], axis=1).astype(np.float32), sr)
    kudio.save_labels("take.txt", [kudio.Label(0.93, 2.07, "speech"),
                                   kudio.Label(3.43, 4.87, "speech")])
    ```

## Channels and sample rates

Anything that works in seconds takes the rate as `sr`. Every function here takes
a mono `(n,)` clip or a `(frames, channels)` one. The fades, `remove_dc`,
`splice`, `crossfade` and the level functions do the same thing to every
channel, one gain for all; the filters, `pitch_shift`, `time_stretch` and
`reverb` work on each channel along its frames; the silence functions cut every
channel at the same samples. [`add_noise_snr`][kudio.effects.augment.add_noise_snr]
takes mono only. `splice` and `crossfade` do not check that the parts share a
rate, so load every part at one rate.

## Fades and DC offset

```python
y, sr = kudio.file_load("take.wav")
y = kudio.fade(y, sr, fade_in_s=0.01, fade_out_s=0.01)
```

A clip cut from a longer recording starts and stops mid-waveform, which clicks;
10 ms at each end removes it without being heard as a fade.
[`fade_in`][kudio.effects.edit.fade_in] and [`fade_out`][kudio.effects.edit.fade_out]
do one end each. `shape=` is `'linear'` (the default, for anti-click fades),
`'cosine'` (flat at both ends, for audible fades) or `'exponential'` (straight in
dB from -60 dB, for slow fade-outs). A fade longer than the clip, or a fade-in
and fade-out that overlap, raises [`FeatureError`][kudio.exceptions.FeatureError].

[`remove_dc`][kudio.effects.edit.remove_dc] subtracts each channel's mean. A
constant offset costs headroom and stops `trim_silence` trimming; an offset that
drifts is a very low frequency, which `highpass` takes out.

## Filters

```python
y = kudio.highpass(y, sr, cutoff=80)       # rumble, handling noise
y = kudio.bandstop(y, sr, 45, 55)          # 50 Hz hum; (55, 65) for 60 Hz
y = kudio.lowpass(y, sr, cutoff=7000)      # hiss
```

| Butterworth, order 4 (`order=`) | Default | Chosen for |
|---|---|---|
| [`highpass`][kudio.effects.filters.highpass] | removes below 80 Hz | rumble, handling noise, slow drift |
| [`lowpass`][kudio.effects.filters.lowpass] | removes above 8000 Hz | hiss, and anything above the band of a 16 kHz model |
| [`bandpass`][kudio.effects.filters.bandpass] | keeps 300–3400 Hz | the telephone band |
| [`bandstop`][kudio.effects.filters.bandstop] | removes 45–55 Hz | 50 Hz mains hum |

A `lowpass` at or above Nyquist has nothing to remove and returns a copy; the
other edges have to lie strictly between 0 and Nyquist, and band edges the wrong
way round are refused. [`band_filter`][kudio.effects.filters.band_filter] takes
the kind as `btype=`, for code that picks it at run time.

The filters run forwards and then backwards (`sosfiltfilt`), so nothing moves in
time and the output lines up sample for sample with the input. That squares the
response, so a tone at the cutoff comes out 6 dB down, not 3:

```python
import numpy as np

tone = np.sin(2 * np.pi * 1000 * np.arange(2 * sr) / sr)     # at the cutoff
middle = slice(sr // 2, 3 * sr // 2)                         # away from the ends
for zero_phase in (True, False):
    out = kudio.lowpass(tone, sr, cutoff=1000, zero_phase=zero_phase)
    gain_db = 20 * np.log10(np.std(out[middle]) / np.std(tone[middle]))
    print(f"zero_phase={zero_phase}: {gain_db:.2f} dB")
```

```text
zero_phase=True: -6.02 dB
zero_phase=False: -3.01 dB
```

`zero_phase=False` is the one-pass causal filter, as a device running live
filters, with a delay that depends on frequency.

## Cutting and joining

```python
y, sr = kudio.file_load("take.wav")
retake, _ = kudio.file_load("retake.wav", sr=sr)    # at the take's rate
a, b = int(1.20 * sr), int(1.45 * sr)               # a cough, say
fixed = kudio.splice([y[:a], y[b:]], sr)                 # cut it out
patched = kudio.splice([y[:a], retake, y[b:]], sr)       # or replace it
print(len(y) - (b - a) - len(fixed))
```

```text
80
```

[`splice`][kudio.effects.edit.splice] blends the last 5 ms of each part into the
first 5 ms of the next ([`DEFAULT_CROSSFADE`][kudio.effects.edit.DEFAULT_CROSSFADE]),
where `np.concatenate` leaves a jump that clicks. The result is shorter than its
parts by that overlap per seam, so labels kept in step with the audio move too.
Empty parts are skipped, and parts of different types are each read on their own
scale.

![A deleted span joined two ways: np.concatenate leaves a one-sample jump and a broadband stripe in the spectrogram; splice leaves no jump and a stripe 44 dB weaker](../images/editing-seam-light.webp#only-light)
![A deleted span joined two ways: np.concatenate leaves a one-sample jump and a broadband stripe in the spectrogram; splice leaves no jump and a stripe 44 dB weaker](../images/editing-seam-dark.webp#only-dark)

[`crossfade`][kudio.effects.edit.crossfade] joins two clips through one seam and
defaults to `shape='equal_power'`, where `splice` defaults to `'linear'`. Linear
keeps the level of one signal carrying on through the seam; equal power keeps the
level of two unrelated recordings, and adds up to 3 dB on one signal.

## Trimming and splitting on silence

```python
tone = 0.5 * np.sin(2 * np.pi * 440 * np.arange(sr // 2) / sr)
y = np.concatenate([np.zeros(sr // 2), tone, np.zeros(sr // 2)]).astype(np.float32)
for clip in (y, y + 0.02, kudio.remove_dc(y + 0.02)):
    _, (start, end) = kudio.trim_silence(clip, top_db=30)
    print(start / sr, end / sr)
```

```text
0.448 1.088
0.0 1.5
0.448 1.088
```

[`trim_silence`][kudio.effects.silence.trim_silence] cuts the silence off both
ends and returns the clip with the range it kept;
[`split_on_silence`][kudio.effects.silence.split_on_silence] cuts at every gap.
A frame is silent when its RMS is `top_db` (30) or more below the loudest
frame's; frames are 2048 samples at a hop of 512, and the cuts lean outwards. A
DC offset of 0.02 keeps every frame above that, so nothing is trimmed until
`remove_dc` takes it out. Level alone does not work over a background such as
the room in `take.wav`; for speech use [`vad`][kudio.core.vad.vad]
([Speech activity and pitch](speech.md)). `kudio trim` does the trim from a
shell.

## Cutting at labels

```python
y, sr = kudio.file_load("take.wav")
for label in kudio.load_labels("take.txt"):
    clip = y[round(label.start * sr):round(label.end * sr)]
    print(label.text, round(label.duration, 2), len(clip))
```

```text
speech 1.14 18240
speech 1.44 23040
```

[`load_labels`][kudio.core.dataset.load_labels] reads an Audacity label track or
JSON back as [`Label`][kudio.core.dataset.Label] spans in seconds; convert with
`round`, not `int`, which cuts 2.01 s one sample short. A frequency row from
Audacity is skipped; a row that is not a label, a time that is not finite and an
end before its start are a `ValueError` naming the line.

## Level

[`normalize`][kudio.effects.augment.normalize] and
[`normalize_db`][kudio.effects.augment.normalize_db] set the peak, as a fraction
of full scale or in dBFS, and leave a silent clip unchanged.
[`gain`][kudio.effects.augment.gain] and [`random_gain`][kudio.effects.augment.random_gain]
change the level by a fixed or a seeded random number of dB. Integer audio keeps
its type, scaled on its own full scale and clipped rather than wrapped. To make
clips sound equally loud, use [`normalize_lufs`][kudio.core.loudness.normalize_lufs]
([Inspecting a recording](inspection.md)).

## Augmentation

Whatever is random takes a `seed`, and the same seed gives the same array.

[`pitch_shift`][kudio.effects.augment.pitch_shift] moves the pitch by `n_steps`
semitones and keeps the length; [`time_stretch`][kudio.effects.augment.time_stretch]
changes the length by `rate` (above 1 is faster) and keeps the pitch. Both are a
phase vocoder with the phases in float64.

```python
y = (0.3 * np.sin(2 * np.pi * 220 * np.arange(2 * sr) / sr)).astype(np.float32)
noise = np.random.default_rng(0).standard_normal(10 * sr).astype(np.float32)
mix = kudio.add_noise_snr(y, noise, snr_db=5, seed=0)
print(round(kudio.snr(y, mix), 2), np.array_equal(mix, kudio.add_noise_snr(y, noise, 5, seed=0)))
```

```text
5.0 True
```

[`add_noise_snr`][kudio.effects.augment.add_noise_snr] mixes mono `noise` into
`y` at `snr_db`, measured over the whole clip, pauses included. Shorter noise is
tiled, longer noise cropped at a seeded offset; noise that is silent or constant
raises `FeatureError`. For a corpus, [`Synthesizer`][kudio.core.synth.Synthesizer]
does the mixing ([Building a noisy dataset](synthesis.md)).

[`reverb`][kudio.effects.augment.reverb] adds seven echoes `delay_ms` apart
(50 ms), each `decay` (0.3) times the last, keeps the length and the peak. It
sounds like an echo, not a room; for a room use [`rir`][kudio.core.room.rir]
and [`apply_rir`][kudio.core.room.apply_rir] ([Rooms and channels](degradation.md)).

[`spec_augment`][kudio.effects.augment.spec_augment] blanks out random bands of a
`(frames, bins)` spectrogram: up to `freq_mask_width` bins (8) and
`time_mask_width` frames (16), one of each by default, each width drawn from 0
up. A masked cell takes the spectrogram's mean, which is what masking with zero
after normalising does in the paper[^specaugment]; `mask_value=` sets another.
The parameters are under [Editing and effects](../reference/effects.md).

[^specaugment]: D. S. Park, W. Chan, Y. Zhang, C.-C. Chiu, B. Zoph, E. D. Cubuk
    and Q. V. Le, "SpecAugment: A simple data augmentation method for
    automatic speech recognition", Proc. Interspeech 2019, pp. 2613-2617,
    doi:10.21437/Interspeech.2019-2680, Sec. 2.
