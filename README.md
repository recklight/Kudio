# Kudio

[![PyPI](https://img.shields.io/pypi/v/kudio.svg)](https://pypi.org/project/kudio/)
[![Python](https://img.shields.io/pypi/pyversions/kudio.svg)](https://pypi.org/project/kudio/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE.txt)

**Lightweight, composable audio toolkit for acoustic inspection.**

Kudio gives you audio I/O, streaming/recording, feature extraction, noisy-data
synthesis, effects & augmentation, enhancement, and quality metrics — as small
composable functions with a clean, typed API.

The core stays light (numpy + soundfile + librosa); heavy or niche features
(plotting, dataframes, PESQ/STOI, live audio) are opt-in extras.

## Install

```bash
pip install kudio            # core: I/O, features, effects, synthesis, metrics
pip install kudio[audio]     # + live capture / playback (PyAudio, sounddevice)
pip install kudio[eval]      # + PESQ / STOI / SDR back-ends
pip install kudio[dnsmos]    # + DNSMOS perceptual score (onnxruntime)
pip install kudio[viz]       # + plotting helpers (matplotlib)
pip install kudio[data]      # + Excel / dataframe export (pandas, openpyxl)
pip install kudio[all]       # everything
```

Requires **Python 3.9+** (tested on CPython 3.9–3.13).

> The Python badge above reflects the version currently published on PyPI; it
> updates automatically once v3.0 is released.

## Quick start

```python
import kudio

# --- I/O (soundfile-backed; float32) --------------------------------------
y, sr = kudio.file_load("clip.wav")             # native rate, no resample
kudio.save_wave("out.wav", y, sr, subtype="PCM_24")   # 16/24-bit or FLOAT

# --- STFT geometry as a value ---------------------------------------------
stft = kudio.STFT(sr=16000, n_fft=512, hop_length=256)
spec = stft.forward(y)                                # (frames, bins)
back = stft.inverse(y, spec)                          # phase taken from y
stft.to_dict()                                        # store beside a checkpoint

# --- features -------------------------------------------------------------
spec = kudio.waveform_to_spectrogram(y, norm=True)    # log-power spectrogram
feat = kudio.mfcc(y, sr, n_mfcc=40)                    # (1, frames, 40)
mel  = kudio.melspectrogram("clip.wav", n_mels=64)     # path or waveform…
mel  = kudio.melspectrogram(y, sr=sr, n_mels=64)       # …arrays need sr

# --- model-facing helpers -------------------------------------------------
x   = kudio.stack_context(spec, context=2)            # ±2 frames per row
win = kudio.frame_windows(spec, n_frames=64)          # (n, 64, bins)
std = kudio.Standardizer().fit(spec)                  # save it with the model
z   = std.transform(spec); spec_again = std.inverse(z)

# --- loudness (BS.1770 / EBU R 128) ---------------------------------------
lufs = kudio.loudness(y, sr)                          # what it sounds like,
y    = kudio.normalize_lufs(y, sr, lufs=-23.0)        # not what its peak is
fair = kudio.match_loudness(enhanced, y, sr)          # before you A/B them

curve = kudio.loudness_over_time(y, sr)               # short-term, 100 ms hop
print(curve)      # "-18.2 LUFS at 4.10s, -31.7 LUFS at 12.80s (13.5 LU apart)"
print(kudio.loudness_range(y, sr))                    # EBU Tech 3342, in LU
print(kudio.true_peak(y, sr))                         # dBTP, between the samples

# --- is this recording usable at all? no reference needed -----------------
info = kudio.audio_info("clip.wav")                   # header, without loading
for problem in kudio.audio_report(y, sr).problems():
    print(problem)     # "content stops at 3812 Hz although the file claims
                       #  16000 Hz — very likely upsampled from 8000 Hz"

# ...and what a listener would say about it (needs kudio[dnsmos] + weights)
score = kudio.dnsmos(y, sr, model_dir="DNS-Challenge/DNSMOS")
print(score.summary())      # SIG / BAK / OVRL on the 1-5 MOS scale

# --- denoising, as interchangeable parts ----------------------------------
clean = kudio.spectral_enhance(y, sr)                 # log-MMSE + MCRA
clean = kudio.spectral_enhance(y, sr, 'omlsa', noise='quantile', q=0.4)
kudio.enhance_folder("noisy/", "clean/", 'logmmse')   # a whole tree

# sweep both axes, and rank a trained model in the same table
for r in kudio.compare_enhancers(y, sr, reference=truth,
                                 noises=list(kudio.NOISE_ESTIMATORS)):
    print(r)   # wiener + initial  15 ms  floor -45.3 dB  SI-SDR +12.42 dB

# --- ...and on audio that has not finished arriving -----------------------
enhancer = kudio.StreamEnhancer(sr, 'logmmse')        # 32 ms latency
for block in microphone:                              # doctest: +SKIP
    monitor(enhancer.process(block))
tail = enhancer.flush()

# --- ...and drawn while it arrives, one column per hop --------------------
view = kudio.SpectrogramStream(sr, seconds=3.0, n_mels=64)
while recording:                                      # doctest: +SKIP
    view.push(rec.drain())                            # drain, never tail
    draw(view.columns())                              # (columns, bins), dBFS

# --- editing --------------------------------------------------------------
joined = kudio.splice([before, replacement, after], sr)   # no click at the seam
labels = [kudio.Label(a, b, "speech") for a, b in kudio.vad(y, sr)]
kudio.save_labels("clip.txt", labels)                 # Audacity label track

# --- where is somebody actually talking? ----------------------------------
spans = kudio.vad(y, sr)                              # [(0.31, 1.84), ...]
parts = kudio.vad_split(y, sr)                        # the segments themselves
clip, (a, b) = kudio.vad_trim(y, sr)                  # first speech to last

# --- ...and what note they are on -----------------------------------------
track = kudio.f0(y, sr)                               # pYIN, NaN where unvoiced
print(track)          # "median 118.3 Hz · 94-162 Hz (9.4 semitones) · 61% voiced"
kudio.save_labels("voiced.txt", track.to_labels())    # openable in an editor

# --- editing --------------------------------------------------------------
y = kudio.fade(y, sr, 0.01, 0.01)                     # anti-click both ends
y = kudio.highpass(y, sr, cutoff=80)                  # zero-phase Butterworth
y = kudio.bandstop(y, sr, 45, 55)                     # mains hum
y = kudio.remove_dc(kudio.reverse(y))

# --- effects & augmentation ----------------------------------------------
y    = kudio.normalize(y, peak=0.99)                  # or normalize_db(y, -1)
y16  = kudio.resample(y, orig_sr=8000, target_sr=16000)
clip, (a, b) = kudio.trim_silence(y, top_db=30)       # isolate the event
segments      = kudio.split_on_silence(y)
aug = kudio.pitch_shift(y, sr, n_steps=2)
aug = kudio.add_noise_snr(y, noise, snr_db=5, seed=0) # reproducible
spec = kudio.spec_augment(spec, freq_mask_width=8, time_mask_width=16, seed=0)

# --- whole folders, and the manifest that records what came from what -----
result = kudio.convert_folder("raw/", "16k/", sr=16000, lufs=-23.0)
print(result)                                         # "412/412 written, 0 failed"
pairs = kudio.load_manifest("runs/exp/manifest.json")
train, val, test = kudio.split_pairs(pairs, 0.1, 0.1, seed=17)

# --- what the transmission did to it --------------------------------------
phone = kudio.telephone(y, sr)                        # 300-3400 Hz, 8 kHz, G.711
lossy = kudio.dropouts(y, sr, loss=0.05, seed=0)      # a bad link
burst = kudio.dropouts(y, sr, loss=0.05, burst=6)     # ...that loses them in runs
rough = kudio.bit_depth(y, bits=8)                    # and a cheap converter
link  = kudio.apply_channel(y, sr, "voip", seed=0)    # or the whole lot at once

# --- rooms: the other kind of degradation ---------------------------------
ir  = kudio.rir(sr, rt60=0.6, drr_db=6.0, seed=0)     # a room you can rebuild
wet = kudio.apply_rir(y, ir)                          # put the clip in it
print(kudio.rt60(ir, sr))    # "T30 0.60 s (EDT 0.61 s) · C50 +12.7 dB · ..."

# --- noisy-data synthesis: the whole chain, in the order it happens -------
#     talker -> room (rt60) -> + noise (snr_ratio) -> link (channel) -> file
syx = kudio.Synthesizer("data/clean", "data/noise",
                        out_path="data/mixed", snr_ratio=(-5, 0, 5),
                        rt60=(0.3, 0.8),              # ...and in rooms
                        channel="telephone",          # ...down a phone line
                        write_targets=True)           # ...with a wet target
syx.syn(mode="inc", seed=17)                          # reproducible
syx.syn(mode="inc", target_sr=16000)                  # or resample the output
kudio.save_manifest("manifest.json", syx.manifest())  # noise, SNR, room, link

# --- metrics (pure numpy, no extra deps) ----------------------------------
print(kudio.si_sdr(ref, est), kudio.snr(ref, est), kudio.segmental_snr(ref, est))

# ...but only once the two line up. They compare sample i with sample i, and
# one sample of delay costs 134 dB.
print(kudio.find_delay(ref, est, sr))   # "lags by 128 samples (8.0 ms), 0.998"
a, b = kudio.align(ref, est, sr)        # trimmed to what they share
print(kudio.si_sdr(a, b))

# --- recording (needs kudio[audio]) --------------------------------------
mics = kudio.list_devices("input")                    # sounddevice indices
wave = kudio.record(seconds=3, sr=16000, device=mics[0]["index"])

with kudio.StreamRecorder(sr=16000) as rec:           # ...or stop when you like
    while still_talking:
        meter.set(rec.level_db())                     # safe to poll from a UI
wave = rec.stop()
```

> **Device indices are backend-specific.** `record()` runs on sounddevice, so
> its `device=` comes from `kudio.list_devices()`. The streaming classes
> (`Recorder`, `LocalStreamReader`) run on PyAudio and take indices from
> `CheckDevice.system_devices()`. The two are numbered independently — an index
> from one must never be handed to the other. `kudio devices` prints both,
> labelled.

## Command line

```bash
kudio devices                                  # list audio devices
kudio info clip.wav                            # sr / channels / duration / peak
kudio report clip.wav                          # health check; exit 1 if wrong
kudio loudness clip.wav                        # integrated, range, true peak
kudio loudness clip.wav --over-time            # the short-term curve
kudio loudness clip.wav --target -23           # exit 1 if it misses EBU R 128
kudio normalize in.wav out.wav --lufs -23      # or --peak 0.99
kudio vad clip.wav --split-to spans/           # where the speech is
kudio pitch clip.wav --segments                # f0 summary + the voiced spans
kudio align clean.wav processed.wav --metrics  # how far apart, and what it cost
kudio room ir.wav                              # T30 / EDT / C50 / DRR
kudio room --make room.wav --rt60 0.6          # ...or build one
kudio room speech.wav --apply wet.wav --rt60 0.6   # ...or put a clip in it
kudio channel in.wav out.wav --telephone       # what the phone line did
kudio channel in.wav out.wav --loss 0.05 --burst 6 --conceal hold
kudio synth ... --channel telephone --targets  # a corpus off a phone line
kudio enhance in.wav out.wav --method logmmse --noise mcra
kudio enhance noisy/ clean/ --recursive        # a whole tree
kudio compare in.wav --reference clean.wav     # rank every method
kudio compare in.wav --noises all              # ...over every estimator too
kudio convert in.wav out.wav --rate 16000 --subtype PCM_16
kudio convert raw/ 16k/ --rate 16000 --recursive --lufs -23
kudio synth --clean C --noise N --out O --snr -5 0 5 --seed 17
kudio synth ... --rt60 0.3 0.6 --manifest manifest.json   # ...in rooms too
kudio trim in.wav out.wav --top-db 30
```

`report`, `vad`, `pitch`, `align` and `room` exit non-zero when they find a
problem, find no speech, find nothing voiced, decide two files are not the same
recording, or decide a file is not an impulse response — and
`loudness --target` does when a file misses its delivery level. They drop
straight into a shell test:

```bash
kudio report clip.wav || echo "needs another take"
```

## Modules

| Module | What's inside |
|---|---|
| `kudio.core.io` | `file_load`, `save_wave`, `resample`, `audio_info`, `convert_folder`, `check_input`, `load_waves`, `copy_waves` |
| `kudio.core.stft` | `STFT` — geometry + `forward`/`inverse`, storable next to a model |
| `kudio.core.feature` | `waveform_to_spectrogram`, `spectrogram_to_waveform`, `mfcc`, `melspectrogram`, `stack_context`, `frame_windows`, `Standardizer`, ... |
| `kudio.effects` | `fade`, `reverse`, `remove_dc`, `highpass`/`lowpass`/`bandpass`/`bandstop`, `trim_silence`, `split_on_silence`, `time_stretch`, `pitch_shift`, `normalize`, `add_noise_snr`, `reverb`, `spec_augment` |
| `kudio.effects.channel` | `mu_law`, `a_law` (G.711 companding), `bit_depth`, `dropouts` (packet loss, independent or in Gilbert runs), `telephone`, `apply_channel`/`channel_spec`/`channel_tag` — the degradation that is neither additive nor convolutive |
| `kudio.core.loudness` | `loudness`, `normalize_lufs`, `match_loudness` — ITU-R BS.1770-4, the perceptual answer `normalize`'s peak scaling cannot give; plus `loudness_over_time` → `LoudnessCurve`, `loudness_range` (EBU Tech 3342) and `true_peak` in dBTP |
| `kudio.core.report` | `audio_report` → `AudioReport` — clipping, DC, silence, noise floor, real bandwidth, **with no clean reference needed** |
| `kudio.core.dnsmos` | `dnsmos` → `DnsmosScore` — predicted P.835 opinion (SIG/BAK/OVRL); needs `[dnsmos]` and weights you supply |
| `kudio.core.vad` | `vad`, `vad_split`, `vad_trim`, `speech_ratio` — noise-adaptive speech detection |
| `kudio.core.pitch` | `f0` → `PitchTrack` — pYIN fundamental frequency **with** the voiced/unvoiced decision, summary stats and label export |
| `kudio.core.spectrogram` | `SpectrogramStream` — ring-buffered column-wise STFT for audio still arriving; linear or mel, dBFS |
| `kudio.core.dataset` | `Pair`, `save_manifest`, `load_manifest`, `split_pairs` — what came from what, in plain JSON |
| `kudio.core.synth` | `Synthesizer` — clean × noise × SNR × room, seedable, with `manifest()` recording what came from what |
| `kudio.core.room` | `rir`, `apply_rir`, `rt60` → `Reverberation`, `schroeder_curve` — reverberation as a controllable degradation and an ISO 3382-1 measurement |
| `kudio.core.evaluator` | `si_sdr`, `snr`, `segmental_snr` (dep-free); `AudioEvaluate` (PESQ/STOI/SDR); `check_metrics_install` |
| `kudio.core.align` | `find_delay` → `Alignment`, `align` — the sample alignment every reference metric assumes and none of them can check |
| `kudio.core.stream` | `record`, `StreamRecorder`, `play_audio`, `Recorder`, `LocalStreamReader`, `RemoteStreamReader` |
| `kudio.enhance` | `spectral_enhance` (7 gain rules × 4 noise estimators), `StreamEnhancer`, `compare_enhancers`, `enhance_folder`, `trad_enhance`, `wavelet_low_pass_filter` |
| `kudio.util` | `list_devices`, `CheckDevice`, `map_waves`, colored console helpers, timers |

Everything commonly used is importable straight from the top level (`kudio.…`).

## Speech enhancement

Seven gain rules over four noise estimators, all sharing one STFT loop:

| Method | What it is | Why you would pick it |
|---|---|---|
| `specsub` | Subtract the noise spectrum | Cheapest; the origin of "musical noise" |
| `multiband` | Per-band over-subtraction | Colored noise isn't equally loud everywhere |
| `wiener` | Gain `ξ/(1+ξ)` | Smooth, so little musical noise; muffles at low SNR |
| `mmse_stsa` | MMSE on the amplitude (Ephraim & Malah 1984) | Keeps more speech than Wiener |
| `logmmse` | MMSE in the log domain (1985) | The default, and hard to beat without a model |
| `omlsa` | log-MMSE × speech-presence probability (Cohen 2001) | Hardest suppression where there is no speech at all |
| `spectral_gate` | Threshold mask, smoothed | Unprincipled, very predictable, good on steady hiss |

```python
clean = kudio.spectral_enhance(y, sr, 'logmmse', noise='mcra')
```

Every method is the same three decisions — estimate the noise, estimate the a
priori SNR, turn it into a gain — so holding one still isolates the other.

**The noise estimator is as large a lever as the gain rule.** Measured across
the full 7 x 4 grid on this repo's fixture: a mean spread of 5.8 dB SI-SDR when
the gain rule varies, 6.4 dB when the estimator does. Neither dominates, which
is exactly why both are choices rather than one being hidden inside the other —
and why `compare_enhancers` can sweep both. `kudio.NOISE_ESTIMATORS` documents
what each one assumes; `initial` is exact when the file opens with room tone and
badly wrong when it does not.

`kudio.METHODS` describes every parameter — range, default, unit, and a
sentence about what turning it costs you — so a GUI or a CLI can build its own
controls instead of restating them.

**Two caveats measured on this repo's own fixtures, not assumed:**

- These methods are for *noisy* audio. Below about 5 dB SNR the trade is good
  (+8.6 dB); above about 10 dB it is not, and all seven come out **worse than
  doing nothing** (−1.8 dB). `compare_enhancers` reports `delta_snr_db` so that
  shows up instead of being hoped away.
- They need to hear the noise on its own. Same clip, same SNR: with pauses
  +5.9 dB, wall-to-wall speech −1.8 dB. With no pauses, `noise='initial'` over a
  leading second of room tone beats anything adaptive.

### Live, on audio still arriving

```python
enhancer = kudio.StreamEnhancer(sr, 'logmmse')   # 32 ms latency
clean = enhancer.process(block)                  # call per block, then flush()
```

The four decision-directed methods were already recursive, so streaming needed
an overlap-add wrapper rather than a second implementation — the noise tracker
and the a priori SNR estimator are the same objects the offline path uses.
`quantile` and the whole-spectrogram methods are refused rather than quietly
meaning something else.

## Errors

All library errors derive from `kudio.KudioError` (`AudioIOError`,
`FeatureError`, `DeviceError`, `SynthesisError`, `DependencyError`), so you can
catch them in one place. Missing an optional extra raises a `DependencyError`
that tells you exactly what to install.

## Logging

kudio uses the standard `logging` module (loggers named `kudio.*`) and prints
nothing by default:

```python
import logging; logging.basicConfig(level=logging.INFO)
```

## Migrating from v2

v3 is a cleanup release. The short cryptic names still work but now emit a
`DeprecationWarning` — switch to the canonical names:

| v2 (deprecated) | v3 canonical |
|---|---|
| `w2s`, `wavform2spec` | `waveform_to_spectrogram` |
| `spec2wavform` | `spectrogram_to_waveform` |
| `f2s`, `wav2spec` | `file_to_spectrogram` |
| `s2w`, `spec2wav` | `save_spectrogram_as_wave` |
| `w2mfcc`, `wav2mfcc` | `mfcc` |
| `concat_mfcc_` | `mfcc_from_files` |
| `concat_logspec_`, `contextual_LogSpectrogram` | `logspec_from_files` |
| `wav2mel` | `melspectrogram` |

Also: `matplotlib` / `pandas` / `openpyxl` are no longer installed by default
(use the `[viz]` / `[data]` extras), and the deprecated `_config` / `_enh` /
`version` compatibility shims were removed.

## License

MIT — see [LICENSE.txt](LICENSE.txt).
