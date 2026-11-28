# Rooms and channels

Noise is additive, $y + n$. A room is convolutive, $y \ast h$: it adds nothing
but smears every sound over the next few hundred milliseconds. The link a
recording travelled over, which kudio calls the channel, is neither: a codec
quantises, a converter throws bits away, a network drops packets. They happen
in one order:

```text
talker -> room -> + noise -> link -> file
```

This page makes and measures rooms and links; mixing in noise is on
[Building a noisy dataset](synthesis.md). Everything here takes mono audio and
refuses a `(frames, channels)` array with [`FeatureError`][kudio.exceptions.FeatureError].
The examples share three seconds of a stand-in for speech:

```python
import numpy as np
import kudio

sr = 16000
t = np.arange(3 * sr) / sr
# a 140 Hz voice with a 4 Hz syllable rhythm, silent between the phrases
voice = sum(np.sin(2 * np.pi * 140 * k * t) / k for k in range(1, 12))
y = 0.1 * voice * (0.5 + 0.5 * np.sin(2 * np.pi * 4 * t))
y[(t < 0.5) | ((t > 1.4) & (t < 1.9)) | (t > 2.6)] = 0.0
y += 0.003 * np.random.default_rng(0).standard_normal(t.size)   # a faint floor
y = y.astype(np.float32)
```

A denoiser takes out a noise spectrum, and a room adds none, so
[`spectral_enhance`][kudio.enhance.spectral.spectral_enhance] does nothing for
reverberation, and nothing undoes a lost packet. The functions here make these
degradations under control and measure them.

## Rooms

### Making one

```python
ir = kudio.rir(sr, rt60=0.6, drr_db=6.0, seed=0)
print(len(ir) / sr)
print(kudio.rt60(ir, sr))
```

```text
0.9
T30 0.60 s (EDT 0.55 s)  ·  C50 +11.7 dB  ·  DRR +6.0 dB  ·  fit 1.000
```

[`rir`][kudio.core.room.rir] builds an impulse response from exponentially
decaying noise, the standard model of a diffuse tail. `rt60` is the time the
envelope takes to fall 60 dB; `drr_db` is how far the direct path sits above the
reverberant energy: +10 dB or more is a close microphone, around 0 a metre or
two away. The length defaults to 1.5 × `rt60`, and the same `seed` builds the
same room. A `drr_db` below what the tail already puts in the 2.5 ms window
raises `FeatureError`. The room has no early reflections, so it will not stand
in for a measured response; [`reverb`][kudio.effects.augment.reverb] is cruder
still.

### Putting a clip in it

[`apply_rir`][kudio.core.room.apply_rir] convolves, cuts the result back to the
input's length (`trim=True`) and puts the peak back where the input's was
(`normalise=True`). It takes no sample rate, so load a recorded response at the
clip's rate, and cut it at the direct sound unless the time of flight should be
kept as a delay:

```python
wet = kudio.apply_rir(y, ir)
measured = ir[np.argmax(np.abs(ir)):]          # a recorded response, from its peak
print(len(wet) == len(y), np.max(np.abs(wet)) == np.max(np.abs(y)))
```

```text
True True
```

### Measuring one

```python
noise = np.random.default_rng(1).standard_normal(len(y)).astype(np.float32)
noisy = kudio.add_noise_snr(y, noise, snr_db=5, seed=0)
print(kudio.rt60(noisy, sr))
room = kudio.rt60(kudio.rir(sr, rt60=1.0, seed=0)[:int(0.05 * sr)], sr)  # 50 ms of 1 s
print(f"T30 {room.t30:.2f} s, fit {room.fit:.3f}, reliable: {room.reliable()}")
```

```text
T30 2.62 s (EDT 18.47 s)  ·  C50 -11.2 dB  ·  DRR -20.3 dB  ·  fit 0.924  — but EDT and T30 disagree, so this may not be an impulse response
T30 0.19 s, fit 0.802, reliable: False
```

[`rt60`][kudio.core.room.rt60] returns a [`Reverberation`][kudio.core.room.Reverberation].
T20, T30 and EDT come from the backward-integrated decay
curve[^schroeder] ([`schroeder_curve`][kudio.core.room.schroeder_curve]): the
time to fall from −5 to −25 dB, −5 to −35 dB and 0 to −10 dB, carried on to
60 dB, the ranges of ISO 3382-1.[^iso3382] C50, C80 and DRR are energy ratios
around the peak, and `fit` is the R² of the T30 line. The measurement starts at
the first sample within 20 dB of the peak, so a time of flight is not counted.

Backward integration makes any signal decay, so read
[`reliable()`][kudio.core.room.Reverberation.reliable] before the figures: it
asks for a `fit` of at least 0.9 and for EDT within a factor of
[`MAX_EDT_RATIO`][kudio.core.room.MAX_EDT_RATIO] (2) of T30. It cannot see a
noise floor: a response less than 45 dB above its noise holds no T30 to quote,
and from about +10 dB DRR there is no EDT, so `reliable()` judges on the fit
alone. `rt60` measures the whole band; for an octave, filter with
`zero_phase=False` and cut at the peak again.

![Schroeder decay of a synthetic room with its T30 fit, and the same room over a noise floor](../images/degradation-schroeder-light.webp#only-light)
![Schroeder decay of a synthetic room with its T30 fit, and the same room over a noise floor](../images/degradation-schroeder-dark.webp#only-dark)

## Channels

### A phone line

```python
phone = kudio.telephone(y, sr)
print(len(phone) == len(y))
for problem in kudio.audio_report(phone, sr).problems():
    print(problem)
```

```text
True
content stops at 3758 Hz, although a 16000 Hz file holds up to 8000 Hz — it was very likely upsampled from a lower rate, or passed through a codec or a filter that cuts there, and the band above is empty
```

[`telephone`][kudio.effects.channel.telephone] band-limits to 300–3400 Hz
([`TELEPHONE_BAND`][kudio.effects.channel.TELEPHONE_BAND]) at the original rate,
resamples to 8 kHz, compands through G.711[^g711] mu-law, and comes back at the
rate and length it went in with.

### Companding and bit depth

```python
full = y / np.max(np.abs(y))
for level in (0, -12, -24):
    x = (full * 10 ** (level / 20)).astype(np.float32)
    print(f"{level:3} dBFS   mu-law {kudio.snr(x, kudio.mu_law(x)):.1f}   "
          f"A-law {kudio.snr(x, kudio.a_law(x)):.1f}   "
          f"8-bit linear {kudio.snr(x, kudio.bit_depth(x, 8)):.1f}")
```

```text
  0 dBFS   mu-law 37.9   A-law 38.1   8-bit linear 40.2
-12 dBFS   mu-law 37.7   A-law 38.0   8-bit linear 28.2
-24 dBFS   mu-law 36.6   A-law 36.6   8-bit linear 18.5
```

[`mu_law`][kudio.effects.channel.mu_law] and [`a_law`][kudio.effects.channel.a_law]
quantise to 8-bit code words on the continuous companding curve, so the noise
follows the signal down; [`bit_depth`][kudio.effects.channel.bit_depth] spends
the bits linearly. They are the curves, not G.711's segment tables, so close
enough for augmentation and not a bit-exact decoder. `bit_depth(dither=True)`
adds TPDF dither two steps wide; pass `seed` with it.

### Packet loss

```python
lossy, gaps = kudio.dropouts(y, sr, loss=0.05, seed=0, return_spans=True)
print(len(gaps), [f"{a:.2f}-{b:.2f} s" for a, b in gaps[:3]])
```

```text
8 ['0.04-0.06 s', '0.12-0.14 s', '0.22-0.24 s']
```

[`dropouts`][kudio.effects.channel.dropouts] cuts the clip into packets of
`packet_ms` (20 ms, [`PACKET_MS`][kudio.effects.channel.PACKET_MS], G.711 over
RTP[^rtp]) and loses `round(loss * packets)` of them: three seconds is 150
packets, so 8. `conceal='silence'`, the default, leaves the holes empty;
`conceal='hold'` repeats the packet before each one. Whether `hold` scores better
on a waveform metric depends on how the pitch period meets the packet length,
so SNR or SI-SDR cannot choose between them.

`burst=` sets the mean run length in packets, geometric as in Gilbert's
model,[^gilbert] with runs kept at least one packet apart; past a loss of a
half that may be impossible, and `dropouts` raises `FeatureError`. `burst=None`,
the default, loses packets independently, which already averages
[`independent_burst`][kudio.effects.channel.independent_burst] packets a run
(1.05 at 5% loss); `burst=1` is runs of exactly one.

![Thirty seconds of 5% packet loss, independent and in runs of 6](../images/degradation-loss-light.webp#only-light)
![Thirty seconds of 5% packet loss, independent and in runs of 6](../images/degradation-loss-dark.webp#only-dark)

### The whole link at once

```python
right = kudio.apply_channel(y, sr, {"preset": "telephone", "loss": 0.1}, seed=0)
wrong = kudio.telephone(kudio.dropouts(y, sr, loss=0.1, seed=0), sr)
print(np.count_nonzero(right == 0.0), np.count_nonzero(wrong == 0.0))
print(kudio.channel_tag({"preset": "voip", "loss": 0.1}))
```

```text
4800 0
mu_law+loss10%x4-hold
```

[`apply_channel`][kudio.effects.channel.apply_channel] runs a link in the order
a link does: band and rate, codec, converter, then the network, so the lost
packets stay exact silence, where losing them first lets the filters spread the
speech into every hole. The spec is a preset from
[`CHANNEL_PRESETS`][kudio.effects.channel.CHANNEL_PRESETS] (`telephone`,
`telephone_a`, `mu_law`, `a_law`, `voip`, `none`), a dict of
[`CHANNEL_DEFAULTS`][kudio.effects.channel.CHANNEL_DEFAULTS] keys, or a preset
with overrides. [`channel_spec`][kudio.effects.channel.channel_spec] refuses an
unknown setting, a loss outside 0 to 1, bits not from 2 to 64, a `burst` under 1
and anything that is not one finite number, before any audio goes through.
[`Synthesizer`][kudio.core.synth.Synthesizer] takes the same spec as `channel=`.

## From the command line

```bash
kudio room ir.wav                                        # exit 1 if not reliable()
kudio room --make room.wav --rt60 0.6 --drr 6 --seed 0
kudio room speech.wav --apply wet.wav --rt60 0.6 --seed 0
kudio channel in.wav out.wav --telephone
kudio channel in.wav out.wav --loss 0.05 --burst 6 --conceal hold --seed 0
```

The parameters are in the [reference](../reference/degradation.md).

[^schroeder]: M. R. Schroeder, "New method of measuring reverberation time", J. Acoust. Soc. Am. 37(3), pp. 409-412, 1965, doi:10.1121/1.1909343.
[^iso3382]: ISO 3382-1:2009, "Acoustics — Measurement of room acoustic parameters — Part 1: Performance spaces".
[^g711]: ITU-T Recommendation G.711 (11/1988), "Pulse code modulation (PCM) of voice frequencies".
[^rtp]: H. Schulzrinne and S. Casner, "RTP profile for audio and video conferences with minimal control", RFC 3551, July 2003, Sec. 4.2, doi:10.17487/RFC3551.
[^gilbert]: E. N. Gilbert, "Capacity of a burst-noise channel", Bell Syst. Tech. J. 39(5), pp. 1253-1265, 1960, doi:10.1002/j.1538-7305.1960.tb03959.x.
