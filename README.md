# Kudio

[![Kudio, audio processing](https://raw.githubusercontent.com/recklight/Kudio/master/docs/assets/banner.webp)](https://recklight.github.io/Kudio/)

[![PyPI](https://img.shields.io/pypi/v/kudio.svg)](https://pypi.org/project/kudio/)
[![Python](https://img.shields.io/pypi/pyversions/kudio.svg)](https://pypi.org/project/kudio/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE.txt)
[![Docs](https://img.shields.io/badge/docs-recklight.github.io%2FKudio-blue.svg)](https://recklight.github.io/Kudio/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24219274.svg)](https://doi.org/10.5281/zenodo.24219274)

**Lightweight, composable toolkit for audio, speech most of all.**

Kudio gives you audio I/O, recording, feature extraction, noisy-data
synthesis, effects and augmentation, denoising and metrics, as small typed
functions over a numpy array and a sample rate.

The metrics are the ones speech papers report: PESQ, STOI, SI-SDR, Hu and
Loizou's measures, MCD and the log-spectral distance for enhancement; NCM,
CSII and SIIB for intelligibility; SI-SDRi, PIT and BSS Eval for separation;
DNSMOS P.835 and P.808, SRMR and WADA-SNR with no reference; WER and CER,
EER, minimum DCF and C_llr, DER and JER, and pitch-tracking errors.
`kudio score` scores a whole test set and prints the table a paper needs:
means with bootstrap confidence intervals, a sign-flip test against a
baseline, a signature line that says how the numbers were computed, and the
BibTeX keys for kudio and every work behind them.

The core install is numpy, scipy, soundfile and librosa, with PyYAML, tqdm and
PyWavelets; plotting, dataframes, PESQ, STOI and SDR, DNSMOS and live audio
are extras. WAV, FLAC, OGG, MP3 and AIFF are read with that; m4a, aac and webm
take `pip install audioread` and ffmpeg on the PATH.

The [documentation](https://recklight.github.io/Kudio/) has a guide to each
part of the library, the full API reference, and
[how each metric was checked](https://recklight.github.io/Kudio/validation/)
and how closely it agrees with its reference; this README is the short tour.

## Install

```bash
pip install kudio            # core: I/O, features, effects, synthesis, metrics
pip install "kudio[audio]"   # + live capture / playback (PyAudio, sounddevice)
pip install "kudio[eval]"    # + STOI / SDR back-ends (pystoi, mir_eval)
pip install "kudio[pesq]"    # + PESQ (pesq) -- builds from source, needs a C compiler
pip install "kudio[dnsmos]"  # + DNSMOS perceptual score (onnxruntime)
pip install "kudio[viz]"     # + plotting helpers (matplotlib)
pip install "kudio[data]"    # + Excel / dataframe export (pandas, openpyxl)
pip install "kudio[all]"     # everything but [pesq]
```

> `pesq` is published on PyPI as source only, so `kudio[pesq]` compiles it:
> it needs a C compiler -- on Windows, the MSVC Build Tools. That is why it
> is an extra of its own rather than part of `[eval]` or `[all]`.

Requires **Python 3.9+**; CI runs the tests on CPython 3.9, 3.11 and 3.13.

## Quick start

```python
import re
from pathlib import Path

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
    print(problem)     # "content stops at 3812 Hz, although a 16000 Hz file
                       #  holds up to 8000 Hz — it was very likely upsampled
                       #  from 8000 Hz (a codec that cuts there looks the
                       #  same), and the band above is empty"

print(kudio.srmr(y, sr), kudio.wada_snr(y, sr))  # reverberation (8/16 kHz), SNR

# ...and what a listener would say about it (needs kudio[dnsmos] + weights)
score = kudio.dnsmos(y, sr, model_dir="DNS-Challenge/DNSMOS")
print(score.summary())      # SIG / BAK / OVRL on the 1-5 MOS scale
print(kudio.dnsmos_p808(y, sr, model_dir="DNS-Challenge/DNSMOS"))  # P.808, one score

# --- denoising, as interchangeable parts ----------------------------------
clean = kudio.spectral_enhance(y, sr)                 # log-MMSE + MCRA
clean = kudio.spectral_enhance(y, sr, 'omlsa', noise='quantile', q=0.4)
kudio.enhance_folder("noisy/", "clean/", 'logmmse')   # a whole tree

# sweep both axes, and put a trained model in the same table
for r in kudio.compare_enhancers(y, sr, reference=truth,
                                 noises=list(kudio.NOISE_ESTIMATORS)):
    print(r)   # wiener + initial  22 ms  floor -45.7 dB  SI-SDR +15.50 dB  dSNR +10.62 dB

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

# --- joining, and labels an editor can open -------------------------------
joined = kudio.splice([before, replacement, after], sr)   # no click at the seam
labels = [kudio.Label(a, b, "speech") for a, b in kudio.vad(y, sr)]
kudio.save_labels("clip.txt", labels)                 # Audacity label track

# --- where is somebody actually talking? ----------------------------------
spans = kudio.vad(y, sr)                              # [(0.31, 1.84), ...]
parts = kudio.vad_split(y, sr)                        # the segments themselves
clip, (a, b) = kudio.vad_trim(y, sr)                  # first speech to last

# --- ...and what note they are on -----------------------------------------
track = kudio.f0(y, sr)                               # pYIN, NaN where unvoiced
print(track)   # "median 118.3 Hz  ·  94.2-161.8 Hz (9.4 semitones)  ·  61% voiced"
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
print(kudio.rt60(ir, sr))    # "T30 0.60 s (EDT 0.55 s)  ·  C50 +11.7 dB  ·  ..."

# --- noisy-data synthesis: the whole chain, in the order it happens -------
#     talker -> room (rt60) -> + noise (snr_ratio) -> link (channel) -> file
syx = kudio.Synthesizer("data/clean", "data/noise",
                        out_path="data/mixed", snr_ratio=(-5, 0, 5),
                        rt60=(0.3, 0.8),              # ...and in rooms
                        channel="telephone",          # ...down a phone line
                        write_targets=True)           # ...with a wet target
syx.syn(mode="inc", seed=17)                          # reproducible, pool or not
syx.syn(mode="inc", target_sr=16000)                  # or resample the output
kudio.save_manifest("manifest.json", syx.manifest())  # noise, SNR, room, link

# a subset straight from a list, mirrored below root=; FLOAT keeps what is
# past full scale, where 16-bit has to clip it (syn() says how many it did)
si_sx = kudio.Synthesizer(sorted(Path("TIMIT/TRAIN").rglob("s[ix]*.wav")),
                          "data/noise", out_path="data/mixed_sisx",
                          root="TIMIT/TRAIN", subtype="FLOAT")

# --- metrics (pure numpy, no extra deps) ----------------------------------
print(kudio.si_sdr(ref, est), kudio.snr(ref, est), kudio.segmental_snr(ref, est))
print(kudio.sdi(ref, est))                     # speech distortion index

# ...or any of them by name, PESQ / STOI / SDR included where installed.
# Papers differ on which PESQ they quote; say which. Raw narrowband P.862:
print(kudio.score(ref, est, sr, ["pesq", "stoi", "sdi"],
                  pesq_mode="nb", pesq_scale="raw"))
print(kudio.score(ref, est, sr, ["llr", "wss", "fwsegsnr", "lsd"]))  # run when named
print(kudio.mcd(ref, est, sr))                 # mel-cepstral distortion, as URGENT 2025
print(kudio.composite(ref, est, sr))           # CSIG/CBAK/COVL, wideband PESQ inside at 16 kHz
print(kudio.ncm(ref, est, sr), kudio.csii(ref, est, sr))  # intelligibility; CSII at 3 levels
print(kudio.si_sdr_improvement(ref, est, noisy))  # SI-SDRi over the input
s = kudio.pit_si_sdr(sources, separated, mixture)  # (sources, samples); matched by best mean
print(s.permutation, s.si_sdri)               # kudio.bss_eval for SDR/SIR/SAR, via mir_eval
ev = kudio.AudioEvaluate(clean_files, noisy_files, pesq_mode="nb", pesq_scale="raw")
ev.eval("ddae", enhanced_files)
ev.get_df()                                    # means, with a Baseline row
ev.per_file()                                  # every file: join on "noisy"

# ...but only once the two line up. They compare sample i with sample i: on
# the metrics guide's clip, a perfect copy one sample late scores +11.9 dB.
print(kudio.find_delay(ref, est, sr))   # "lags by 128 samples (8.0 ms), correlation 0.970"
a, b = kudio.align(ref, est, sr)        # trimmed to what they share
print(kudio.si_sdr(a, b))

# --- a whole test set: what goes in the table, and how it was computed ----
result = kudio.score_dataset("enhanced/", reference="clean/", baseline="noisy/",
                             metrics=["pesq", "stoi", "si_sdr"],
                             groups=re.compile(r"^(p\d+)_"))  # speaker from the id
print(result.to_markdown())   # mean [95% CI], difference, p (Holm), signature, keys
result.save_rows("scores.csv")                         # one row per utterance
print(kudio.bootstrap_ci(per_utterance, groups=speakers))  # BCa, by whole speaker

# --- what a recognizer, a speaker verifier or a pitch tracker got wrong ---
print(kudio.wer(ref_texts, hyp_texts))          # pooled over the corpus; normalise first
print(kudio.cer("今天天氣很好", "今天天氣不好"))    # 0.1667; spaces left out unless spaces=True
print(kudio.word_errors(ref_texts, hyp_texts))  # hits, substitutions, deletions, insertions
print(kudio.eer(scores, labels))                # proportion; method="rocch" for BOSARIS's
print(kudio.min_dcf(scores, labels, p_target=0.05))   # no default prior: VoxSRC's here
print(kudio.cllr(llrs, labels), kudio.act_dcf(llrs, labels, p_target=0.01))  # calibration
ref, hyp = kudio.load_rttm("ref.rttm"), kudio.load_rttm("sys.rttm")
print(kudio.der(ref, hyp), kudio.jer(ref, hyp))  # who spoke when; collar= is per side
print(kudio.f0_errors(ref_track, est_track))    # GPE, VDE, FFE, FPE, RMSE in cents

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
kudio score enhanced/ --reference clean/ --baseline noisy/   # a test set, and its table
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

`report`, `vad`, `pitch`, `align` and `room` exit 1 when they find a problem,
find no speech, find nothing voiced, decide two files are not the same
recording, or decide a file is not an impulse response; `loudness --target`
does when a file misses its delivery level, and `score` when an utterance has
no score in some column. They drop straight into a shell test:

```bash
kudio report clip.wav || echo "needs another take"
```

## Modules

| Module | What's inside |
|---|---|
| `kudio.core.io` | `file_load`, `save_wave`, `resample`, `audio_info`, `convert_folder`, `check_input`, `load_waves`, `copy_waves` |
| `kudio.core.stft` | `STFT` — geometry + `forward`/`inverse`, storable next to a model |
| `kudio.core.feature` | `waveform_to_spectrogram`, `spectrogram_to_waveform`, `mfcc`, `melspectrogram`, `stack_context`, `ContextFrames` (context windows a batch at a time), `frame_windows`, `Standardizer`, ... |
| `kudio.effects` | `fade`, `reverse`, `remove_dc`, `highpass`/`lowpass`/`bandpass`/`bandstop`, `trim_silence`, `split_on_silence`, `time_stretch`, `pitch_shift`, `normalize`, `add_noise_snr`, `reverb`, `spec_augment` |
| `kudio.effects.channel` | `mu_law`, `a_law` (G.711 companding), `bit_depth`, `dropouts` (packet loss, independent or in Gilbert runs), `telephone`, `apply_channel`/`channel_spec`/`channel_tag` — the degradation that is neither additive nor convolutive |
| `kudio.core.loudness` | `loudness`, `normalize_lufs`, `match_loudness` — ITU-R BS.1770-4, the perceptual answer `normalize`'s peak scaling cannot give; plus `loudness_over_time` → `LoudnessCurve`, `loudness_range` (EBU Tech 3342) and `true_peak` in dBTP |
| `kudio.core.report` | `audio_report` → `AudioReport` — clipping, DC, silence, noise floor, real bandwidth, **with no clean reference needed** |
| `kudio.core.dnsmos` | `dnsmos` → `DnsmosScore` — predicted P.835 opinion (SIG/BAK/OVRL), personalised with `personalized=True`; `dnsmos_p808`, the P.808 score; needs `[dnsmos]` and weights you supply |
| `kudio.core.srmr` | `srmr` — speech-to-reverberation modulation energy ratio, no reference needed; the numbers of the authors' MATLAB toolbox, 8 and 16 kHz |
| `kudio.core.wada` | `wada_snr` — the SNR of a speech recording estimated from the recording alone (Kim & Stern 2008), nan rather than a guess where it cannot read one |
| `kudio.core.vad` | `vad`, `vad_split`, `vad_trim`, `vad_frames` (one decision per STFT frame), `speech_ratio` — noise-adaptive speech detection |
| `kudio.core.pitch` | `f0` → `PitchTrack` — pYIN fundamental frequency **with** the voiced/unvoiced decision, summary stats and label export |
| `kudio.core.spectrogram` | `SpectrogramStream` — ring-buffered column-wise STFT for audio still arriving; linear or mel, dBFS |
| `kudio.core.dataset` | `Pair`, `save_manifest`, `load_manifest`, `split_pairs` — what came from what, in plain JSON |
| `kudio.core.synth` | `Synthesizer` — clean × noise × SNR × room from folders, lists or manifests; seeded per file, so the pool cannot change a result; `manifest()` records what came from what, `.clipped` names any file written past full scale, `.seed` is the base it drew so a run with no seed can still be rebuilt |
| `kudio.core.room` | `rir`, `apply_rir`, `rt60` → `Reverberation`, `schroeder_curve` — reverberation as a controllable degradation, and measured with ISO 3382-1's evaluation ranges |
| `kudio.core.evaluator` | `si_sdr`, `snr`, `segmental_snr`, `sdi`, `lsd` (dep-free; `lsd` gives URGENT 2025's number by default); `pesq` (P.862 / P.862.2, raw or MOS-LQO); `score` (any metric by name); `AudioEvaluate` (PESQ/STOI/SDR, means and per file); `check_metrics_install` |
| `kudio.core.quality` | `llr`, `itakura_saito`, `cepstral_distance`, `wss`, `fw_segmental_snr`, `composite` → `CompositeScore` (CSIG/CBAK/COVL) — Hu and Loizou's measures as Loizou's MATLAB code computes them; numpy only, 8 and 16 kHz, and the composite needs a PESQ score |
| `kudio.core.melcepstrum` | `mcd` — the mel-cepstral distortion of URGENT 2025, SPTK's analysis and FastDTW in numpy |
| `kudio.core.intelligibility` | `ncm`, `csii` → `CsiiScore` — the normalized covariance metric and the coherence SII at three levels, as Loizou's NCM.m and CSII.m compute them |
| `kudio.core.siib` | `siib`, `siib_gauss` — speech intelligibility in bits per second, SIIB and SIIB^Gauss, as the authors' SIIB.m and SIIB_Gauss.m compute them; 5.45 s of speech at the least, 20 s or more as they advise |
| `kudio.core.separation` | `si_sdr_improvement`, `sdr_improvement`, `pit_si_sdr` → `PitScore`, `bss_eval` → `BssEval` — the improvement over the mixture, and several sources matched to their estimates; BSS Eval through mir_eval |
| `kudio.core.wer` | `wer`, `cer`, `word_errors`, `char_errors` → `ErrorCounts` — error rates pooled over a corpus, with the substitutions, deletions and insertions behind them; no text normalisation |
| `kudio.core.verification` | `eer`, `min_dcf`, `det_curve` → `DetCurve` — the EER (interpolated, or BOSARIS's ROCCH-EER) and NIST's minimum detection cost, with the cost parameters yours to give; `cllr`, `min_cllr`, `act_dcf`, `c_primary` for scores that are LLRs |
| `kudio.core.diarization` | `der`, `diarization_errors` → `DiarizationErrors`, `jer`, `detection_error_rate`, `load_rttm` — who spoke when, scored as md-eval.pl and dscore score it |
| `kudio.core.f0eval` | `gross_pitch_error`, `voicing_decision_error`, `f0_frame_error`, `fine_pitch_error`, `f0_rmse`, `f0_errors` → `F0Errors` — GPE, VDE, FFE, FPE and the RMSE in cents, for two F0 tracks or two `PitchTrack`s |
| `kudio.core.align` | `find_delay` → `Alignment`, `align` — the sample alignment every reference metric assumes and none of them can check |
| `kudio.core.dataset_score` | `score_dataset` → `DatasetScore` — a whole test set paired up by id from folders, `id path` lists or CSV files, one row per utterance, and the table of means, intervals and tests as Markdown, LaTeX or JSON; `kudio score` from a shell |
| `kudio.core.stats` | `bootstrap_ci`, `paired_ci`, `rate_ci`, `paired_rate_ci` → `ConfidenceInterval`, `permutation_test` → `PermutationResult`, `holm` — BCa intervals and the paired sign-flip test, with whole speakers resampled and flipped together when `groups=` names them |
| `kudio.core.signature` | `signature`, `parse_signature` — one line, after sacreBLEU's, that says how a set of numbers was computed |
| `kudio.core.registry` | `cite`, `metric_info` → `MetricInfo`, `METRIC_INFO`, `BIBTEX` — which way each metric is better, its range and unit, the settings that change its number, and the BibTeX of the works it comes from |
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
| `omlsa` | log-MMSE × speech-presence probability (Cohen & Berdugo 2001) | Suppresses hard where speech is unlikely, gently where it might be |
| `spectral_gate` | Threshold mask, smoothed | Unprincipled, very predictable, good on steady hiss |

```python
clean = kudio.spectral_enhance(y, sr, 'logmmse', noise='mcra')
```

Every method is the same three decisions: estimate the noise, estimate the a
priori SNR, turn it into a gain. Holding one still isolates the other.

**The noise estimator is as large a lever as the gain rule.** Measured across
the full 7 x 4 grid on the test suite's 5 s clip (two harmonic bursts over
white noise, 2 dB SNR): a mean spread of 5.5 dB SI-SDR when the gain rule
varies, 5.7 dB when the estimator does. Neither dominates, which is why both
are choices rather than one being hidden inside the other, and why
`compare_enhancers` can sweep both. `kudio.NOISE_ESTIMATORS` documents
what each one assumes; `initial` is exact when the file opens with room tone and
badly wrong when it does not.

`kudio.METHODS` describes every parameter, with its range, default, unit and a
sentence about what turning it costs you, so a GUI or a CLI can build its own
controls instead of restating them.

**Two caveats measured on that clip, not assumed** (the seven gain rules over
`mcra`, scored by how much they raise the SNR):

- These methods are for *noisy* audio. At 0 dB SNR in they gain +7.8 dB on
  average, and at 5 dB +6.1. At 10 dB the average is down to +3.8 and one of the
  seven already does harm; at 20 dB all seven come out **worse than doing
  nothing** (−4.1 dB on average). `compare_enhancers` reports `delta_snr_db` so
  that shows up instead of being hoped away.
- They need to hear the noise on its own. At 5 dB SNR, `logmmse` gains +7.1 dB
  on the clip with its pauses and +0.7 dB on the same two voices back to back.
  With no pauses, give it a leading second of room tone and `noise='initial'`
  (+9.6 dB there) beats anything adaptive (`mcra` +6.1, `minimum` +1.7).

### Live, on audio still arriving

```python
enhancer = kudio.StreamEnhancer(sr, 'logmmse')   # 32 ms latency
clean = enhancer.process(block)                  # call per block, then flush()
```

The four decision-directed methods were already recursive, so streaming needed
an overlap-add wrapper rather than a second implementation: the noise tracker
and the a priori SNR estimator are the same objects the offline path uses.
`quantile` and the whole-spectrogram methods are refused rather than quietly
meaning something else.

## Errors

What goes wrong with a file, a device, a feature, a synthesis run or a
missing extra raises a `kudio.KudioError` (`AudioIOError`, `FeatureError`,
`DeviceError`, `SynthesisError`, `DependencyError`), so those can be caught in
one place. A missing optional extra raises a `DependencyError` that names what
to install; it is an `ImportError` too. A wrong argument, such as a negative
rate, an unknown metric or two signals too different in length to compare,
raises `ValueError` or `TypeError`, the way the standard library does, and a
buffer that waited too long raises `TimeoutError`. `FeatureError` is a
`ValueError` too. A metric refuses what it
cannot read as a signal, NaN samples or int64 ones, with a `ValueError` that
starts with its name, and gives nan with one warning where there is nothing
to score, such as an empty pair or a silent reference.

## Logging

kudio uses the standard `logging` module (loggers named `kudio.*`) and logs
nothing you see by default. Problems with the audio itself, such as samples
that are not numbers, a normalisation that will clip or a pair scored on only
part of its length, come as Python warnings, which do show. To see the log:

```python
import logging; logging.basicConfig(level=logging.INFO)
```

## Upgrading

From 3.x: the metrics 3.x had give the same numbers on every pair they could
score, except PESQ on a pair of two lengths, which now gets both signals whole,
the way the ITU program reads two files. Input with nothing to score, such as
an empty pair or a silent reference, gives nan with a warning where 3.x gave a
number or raised, and input that is not a signal, int64 samples among them,
raises `ValueError`. From v2: the short names (`w2s`, `wav2mfcc` and the rest)
still work behind a `DeprecationWarning`, and plotting, dataframes and live
audio need their extras. [Migrating](https://recklight.github.io/Kudio/migrating/)
lists every rename and every result that changes for the same call.

## Citing Kudio

If you use Kudio in your research, please cite it. Zenodo archives the
releases under [10.5281/zenodo.24219274](https://doi.org/10.5281/zenodo.24219274),
which resolves to the newest one. `CITATION.cff` holds what GitHub shows
under "Cite this repository", and `kudio.cite` writes the BibTeX
for Kudio and for the papers and standards behind the metrics you report:

```python
import kudio

print(kudio.cite(["pesq", "stoi", "si_sdr"]))
```

Kudio's entry comes first, then each work once. Put the settings next to the
number too: `kudio.metric_info("pesq").variants` says what changes a metric's
value, such as PESQ's mode and scale.

## License

MIT — see [LICENSE.txt](LICENSE.txt).
