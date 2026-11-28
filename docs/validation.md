# How the metrics are checked

Each metric in the table below was compared with the implementation its
published numbers come from, or, where there is none, with its definition
worked out a second way; `tonal_lines` and the band limit, clipping count and
sample grid of `audio_report` have neither and are checked on signals whose
answer is known. The sections under the table say where each definition comes
from, and the last ones how to run the scripts in `tools/validation/`.
Differences are absolute unless marked relative.

The tests store the reference implementations' numbers for pairs of their own,
so `pytest` holds kudio to them without the references installed; SRMRToolbox's
ratios, PESQ's conformance test and pySIIB's demo figures are left to the
scripts, since their recordings are not committed.

| Metric | Compared with | Largest difference |
|---|---|---|
| `llr`, `cepstral_distance`, `wss`, `fw_segmental_snr` | pysepm_evo 0.1.1 | 1.1e-11 relative |
| `itakura_saito` | comp_is.m restated; the Itakura-Saito divergence of the two LPC spectra | 4.1e-11 relative |
| `composite` | the regression on pysepm_evo 0.1.1's parts and pesq 0.0.4 | 3.3e-11 relative |
| `lsd` | URGENT 2025's and `ssr_eval`'s definitions, restated on librosa's STFT | 1.1e-10 relative |
| `mcd` | URGENT 2025's MCD on pysptk 1.0.1 and fastdtw 0.3.4 | 7.1e-15 dB |
| `srmr` | the four ratios SRMRToolbox stores for its test recording | 4.5e-13 relative |
| `wada_snr` | the model computed another way | 2.3e-11 in the curve it reads the SNR from |
| `pesq` | ITU-T P.862 Annex A, conformance test 2(b) | 5.0e-4, the table's rounding |
| `dnsmos`, `dnsmos_p808` | DNS-Challenge's `dnsmos_local.py` under the packages it pins; torchmetrics 1.9.0 | 6.8e-7; 0 against torchmetrics |
| `ncm` | pysepm_evo 0.1.1; two copies of ANSI S3.5-1997's tables | 2.2e-6 at 16 kHz, 8.3e-9 at 8 kHz; none in the tables |
| `csii` | pysepm_evo 0.1.1; two copies of ANSI S3.5-1997's tables | 4.4e-16; none in the tables |
| `siib`, `siib_gauss` | both measures worked out a second way; pySIIB at commit 226c2f3; the points pySIIB's demo figures plot | at 16 kHz, SIIB^Gauss under 1e-12 relative from both, SIIB 1.4e-5 relative, the noise that breaks ties; elsewhere pySIIB's resampler, a few bit/s; all 15 points within the figures' reading, SIIB's with the pair taken to 16 kHz by pySIIB's filter |
| `wer`, `cer` | every alignment enumerated; full edit-distance tables | 0 |
| `eer`, `min_dcf`, `det_curve` | an exact sweep in fractions; scikit-learn; the VoxSRC and NIST scorers | 0 for the EER; 2.2e-16 for min DCF |
| `cllr`, `min_cllr`, `act_dcf`, `c_primary` | 40-digit arithmetic; PYLLR's rules; scikit-learn's isotonic regression; exact fractions | 3.3e-16 bits; 2.8e-14 in the cost |
| F0 measures | mir_eval 0.8.2; a frame-by-frame count | 0 in the counts; 1.1e-16 in the rates |
| `der`, `jer`, `detection_error_rate` | pyannote.metrics 4.2.0; dscore's JER; a count on 1 ms frames | 8.9e-16 |
| `pit_si_sdr`, `si_sdr_improvement`, `bss_eval`, `sdr_improvement` | torchmetrics 1.9.0; every permutation; mir_eval 0.8.2 | 8.9e-11 dB in SI-SDR; 0 for BSS Eval |
| `bootstrap_ci`, `paired_ci`, `rate_ci`, `paired_rate_ci`, `permutation_test`, `holm` | scipy's `bootstrap` and `permutation_test` (1.13.1 and 1.18.1) on the same resamples and patterns; exact fractions; the definitions restated | 0 for percentile ends read off scipy's resamples and for exact p-values; 2.4e-14 of the width for BCa |
| `audio_report`'s `band_limited`, `source_rates`, `mirrored` | no implementation; 720 upsampled files, 10 codec cuts and 470 recordings of known history | at 0 and -20 dB 468 of 480 flagged, the 12 left linear interpolation by a ratio that is not a whole number; none named wrong; none of the 470 flagged |
| `tonal_lines` | no implementation; 720 sines, 90 noise clips, 40 pairs of combs | 0.162 Hz and 0.52 dB on the sines; no line in noise; 40 of 40 combs grouped right |
| `audio_report`'s `clipped_samples`, `full_scale_samples` | no implementation; 1000 smooth peaks, 409 cut sines, an overdriven voice | no smooth peak counted; every plateau of a sine driven 1.01 times or more counted in full |
| `audio_report`'s `sample_step`, `grid_bits` | no implementation; 7 known grids and 4 clips without one | 0 |

## Hu and Loizou's measures and the composite

Defined by
:   Hu and Loizou 2008:[^hl08] LLR eq. 4, IS eq. 5, CEP eqs. 6-7 (eq. 7 is
    the LPC cepstrum distance of Kitawaki et al.,[^kit88] their eq. 1),
    fwSegSNR eqs. 8-9 and WSS eq. 10 (after
    Klatt[^klatt82]); the composite's regressions in Hu and Loizou 2006,[^hl06]
    Sec. 4.1. Window, eps, frame count and rounding follow Loizou's MATLAB
    files (comp_llr.m, comp_is.m, comp_cep.m, comp_wss.m, comp_fwseg.m,
    composite.m).[^l13]

## Log-spectral distance

Defined by
:   `lsd_metric` in the scoring script of URGENT 2025[^urgent]
    (`evaluation_metrics/calculate_intrusive_se_metrics.py`, last changed in
    commit 93e16aa of 2024-11-19), and `AudioMetrics.lsd` in NVSR's
    `ssr_eval`[^nvsr] (`ssr_eval/metrics.py`, last changed in commit 907395c
    of 2022-03-28), both read on 2026-11-13.

## Mel-cepstral distortion

Defined by
:   `mcd_utils.py` and `mcd_metric` of URGENT 2025's scoring script[^urgent]
    (commits c726558 and 93e16aa), which port ESPnet's MCD: SPTK's `mcep`,
    the mel-cepstral analysis of Fukada et al.,[^ftki92] and FastDTW.[^sc07]

## SRMR

Defined by
:   Falk, Zheng and Chan,[^srmr] and Santos, Senoussaoui and Falk for the
    normalised variant,[^srmrnorm] as the authors' SRMRToolbox computes them
    in `SRMR.m` (github.com/MuSAELab/SRMRToolbox, head 690bdd1 of
    2023-05-23).

## WADA-SNR

Defined by
:   Kim and Stern,[^wada] Eqs. 1, 6, 8 and 17, with speech gamma-distributed
    of shape 0.4 and the noise Gaussian.

## DNSMOS

Defined by
:   The three models DNS-Challenge distributes, `DNSMOS/sig_bak_ovr.onnx`
    (P.835),[^dnsmos835] `pDNSMOS/sig_bak_ovr.onnx` (personalised) and
    `DNSMOS/model_v8.onnx` (P.808),[^dnsmos808] as of commit 785ab12, run as
    `DNSMOS/dnsmos_local.py` runs them (commit 6412bc5 of 2023-01-10) under
    the packages its `requirements.txt` pins: numpy 1.22.4, librosa 0.8.1 and
    onnxruntime 1.13.1.

## PESQ

Defined by
:   ITU-T P.862, with P.862.1's mapping to MOS-LQO and P.862.2's wideband
    extension.[^p862] The C code of P.862's Annex A as Amendment 2 revised it
    in 2005, version 2.0, is part of the Recommendation and takes precedence
    over the text where the two differ (A.1.1).

## NCM and CSII

Defined by
:   Ma, Hu and Loizou[^mhl09] for both, Kates and Arehart[^ka05] for CSII,
    as NCM.m and CSII.m compute them (F. Chen and P. C. Loizou, 2011 and
    2012, in the code that comes with Loizou's book[^l13]). The band tables
    are those of ANSI S3.5-1997,[^ansi] checked against the CRAN package SII
    1.3.0[^siir] and the C program of the standard's working group.[^siic]

## SIIB

Defined by
:   Van Kuyk, Kleijn and Hendriks: SIIB in their letter,[^vkh18a] Eqs. 2
    and 8 to 12, and SIIB^Gauss in their evaluation of intrusive
    measures,[^vkh18b] Sec. IV-C, with each SIIB channel's information taken
    by the second estimator of Kraskov, Stögbauer and Grassberger,[^ksg04]
    their Eq. 9.
    Where the papers leave a detail open or the code does it otherwise,
    kudio follows SIIB.m, SIIB_Gauss.m and MILCA's MIxnyn; the
    [API reference][kudio.core.siib] lists each place.

## WER and CER

Defined by
:   The Levenshtein distance between the two word or character sequences,
    every edit costing 1, and WER as Jurafsky and Martin define it on that
    distance.[^jm] A corpus is pooled as Kaldi's `compute-wer` and jiwer pool
    it.

## EER, minimum DCF and the DET curve

Defined by
:   The operating points of the DET curve;[^det] the EER by straight lines
    between successive operating points, as the VoxSRC 2020-2023 scorer and
    NIST's SRE 2016 scoring script (version 4.1, as WeSpeaker distributes
    it) read it, or by the convex hull, the ROCCH-EER of BOSARIS[^bosaris]
    (Sec. 2.6.2); the detection cost and its normalisation as NIST's
    evaluation plans define them, minDCF being the lowest normalised cost over
    all thresholds.[^nist]

## Calibration: C_llr, min C_llr, the actual DCF and C_Primary

Defined by
:   Brümmer and du Preez[^bd06] for C_llr and for min C_llr through the
    pool-adjacent-violators fit; actDCF, the cost at the threshold log(beta),
    and C_Primary, its mean at p_target 0.01 and 0.005, from NIST's SRE 2016
    plan, Sec. 3.1.[^nist]

## Diarization: DER, JER and the detection error rate

Defined by
:   DER as NIST's RT-09 plan, Sec. 6.1,[^rt09] defines it and md-eval.pl
    (version 22, as dscore carries it) scores it: the speakers mapped on all
    the time evaluated, collars of the given seconds on each side of every
    reference boundary and the overlap taken out afterwards. JER as dscore,
    the DIHARD scorer,[^dihard] computes it; the detection error rate as
    pyannote.metrics' `DetectionErrorRate`.

## F0 errors

Defined by
:   Chu and Alwan[^ca09] for GPE, VDE and FFE (eqs. 1-5), Drugman and
    Alwan[^da11] for FPE, the four cases of Rabiner et al.,[^r76] and the
    RMSE of `1200 log2(est / ref)` in cents.

## Separation: SI-SDRi, PIT and BSS Eval

Defined by
:   SI-SDR as `si_sdr` computes it, both signals taken to zero mean as
    Conv-TasNet's SI-SNR is (Luo and Mesgarani, eq. 15[^convtasnet]); the
    assignment with the highest mean SI-SDR over the sources, the criterion
    of utterance-level PIT;[^upit] SDR, SIR and SAR by BSS Eval;[^bss] and
    the improvement as the score of the estimate minus that of the mixture
    against the same source.

## Confidence intervals and the sign-flip test

Defined by
:   The percentile and BCa intervals of Efron 1987[^e87] and Efron and
    Tibshirani 1993[^et93] (Ch. 14), the acceleration from the jackknife
    over the units resampled; the cluster bootstrap, whole groups resampled
    with replacement, as Field and Welsh study it;[^fw07] error rates
    resampled as ratios of sums, as Bisani and Ney do for the WER;[^bn04]
    the sign-flip test, with Phipson and Smyth's (b + 1)/(B + 1) for
    patterns drawn at random;[^ps10] Holm's step-down adjustment.[^h79]

## Upsampling and band limits in `audio_report`

Defined by
:   kudio itself; no published test says whether a file was upsampled. An
    edge is a frequency where the Welch spectrum (8192-sample Hann segments,
    half overlapped, median-smoothed over 2% of the edge) stands at least
    20 dB higher at 0.70-0.88 of the edge than at 1.05-1.25 of it, and at
    least 40 dB higher than the median of everything from 1.05 of the edge to
    0.97 of Nyquist. The content ends at the last frequency within 10 dB of
    the band under the edge, and each common rate whose Nyquist frequency it
    ends 0.95-1.06 of is named. Linear interpolation is tested apart: by a
    whole factor, its image above the old Nyquist frequency fc stands
    `40 log10 |sin(pi (fc - x) / sr) / sin(pi (fc + x) / sr)|` dB under the
    band it mirrors, and the measured difference has to follow that curve to
    a median absolute deviation of 0.3 dB, with its median within 1 dB of it.

## Narrowband lines: `tonal_lines`

Defined by
:   kudio itself. A line is a local peak of the Welch spectrum (Hann
    segments, half overlapped, the longest power of two up to
    `sr / resolution_hz`, 5 Hz by default) standing `min_prominence_db`,
    20 dB, above the median of the 101 bins around it; its frequency and
    level are read off a parabola through the peak's three bins in dB. A comb
    is three lines or more on one unbroken run of whole multiples of a
    frequency, broken where three multiples in a row have no line, and is
    scored one for each line it explains and minus a half for each multiple
    from 1 to its highest member that has none.

## Clipping in `audio_report`

Defined by
:   kudio itself. Samples at 0.999 of full scale or more make runs, and the
    plateau of a run is its samples within 1e-4 of its top. A plateau counts
    when it is `clip_run` samples long or more, three by default, and holds
    at least twice the share of its run that a smooth peak of the same
    height keeps within 1e-4 of its top: `sqrt(1e-4 / (m - 0.999))` for a
    top at m, 0.32 at full scale, so 0.63 there. A run whose top is within
    1e-4 of 0.999 is not judged. `full_scale_samples` counts every sample at
    0.999 or more.

## The sample grid of `audio_report`

Defined by
:   kudio itself. Every finite sample is read at 24-bit resolution. When all
    of them are whole numbers there and 16 distinct values or more remain
    once the two values at full scale are left out, the step is the greatest
    common divisor of the gaps between neighbouring values; `sample_step` is
    the step as a fraction of full scale and `grid_bits` is
    `log2(2 / sample_step)`. Under 15.95 bits is a finding.

## SNR, SI-SDR, STOI and SDR

`snr`, `si_sdr`, `segmental_snr` and `sdi` are a few lines of numpy each,
defined on [Measuring quality](guide/metrics.md#what-each-metric-sees).
`stoi` and `estoi` are pystoi, and `sdr` is BSS Eval as mir_eval has it. kudio cuts
the pair to the length it shares and passes it on, so their numbers are those
packages' numbers.

## kudio and librosa

The resampling, the short-time transform and its inverse, the mel filters, the
mel spectrogram and MFCC, cutting at silence, pYIN and the time stretch follow
librosa[^librosa], and kudio computes them itself, on numpy, scipy and soxr. The
conventions are librosa's, down to how a frame is padded and how a mel filter is
scaled, and I owe them to its authors; librosa is under the ISC licence. The
algorithms go back to where they were published: the inverse transform is
Griffin and Lim's least-squares estimate[^gl84], the mel scale Slaney's[^s98],
the cepstrum Davis and Mermelstein's[^dm80], the pitch tracker pYIN[^pyin], which
is built on YIN[^yin], and the time stretch a phase vocoder[^fg66][^ld99].

`tools/validation/against_librosa.py` compares kudio with librosa on the cases of
`tests/test_against_librosa.py`, made from the same seeds. A difference of 0 is
to the bit: the same shape, the same type and the same bits, a NaN in the same
place.

| Compared with | Cases | Largest difference |
|---|---|---|
| `librosa.resample` and `librosa.load(sr=...)`: 13 pairs of rates, 8 resamplers, 1, 2 and 6 channels, 12 sample types, 5 layouts in memory, 6 kinds of file | 1626 | 0 |
| `librosa.stft`: 16 geometries, 8 windows, float16, float32 and float64, from one sample to 5001 | 1062 | 0 |
| `librosa.istft`: 15 geometries, 3 windows, 1 to 40 frames, complex64 and complex128 | 1644 | 0 |
| the spectrogram functions on top of the two | 128 | 0 |
| `librosa.filters.mel`: 7 rates, 6 frames, 1 to 256 bands, 4 lowest and 5 highest frequencies | 5880 | 0 |
| `librosa.mel_frequencies` | 30 | 0 |
| `librosa.feature.mfcc` and `librosa.feature.melspectrogram`: 8 kinds of clip, 2 precisions, 3 rates | 720 and 192 | 0 |
| `librosa.effects.trim` and `librosa.effects.split`: 25 clips, 2 precisions, 6 thresholds, 7 frames | 2100 each | 0 |
| `librosa.pyin`, its whole search: 528 hard signals and 1243 more over hops, frames, ranges, rates and lengths | 1771 | 0 |
| `librosa.effects.time_stretch`: two seconds of a voice at 8, 16 and 44.1 kHz, 6 rates, the end left out | 18 | 0.47 % of the peak against 0.11, 4.3 parts in a million against 1.0 |

kudio works out a resampled clip's length in whole numbers. From a rate of the
11025 Hz family to 12, 24, 48, 96 or 192 kHz, librosa's floating-point
`ceil(n * target_sr / orig_sr)` comes out a sample longer, a zero, and the
table compares the samples without it.

kudio's pYIN searches every transition, as librosa 0.11 does; librosa 1.0 skips
those under `transition_min_prob=1e-4` unless passed `None`, and the table
compares its whole search. Frames its default voices otherwise, by hop, in a
minute of a voice with pauses (16 kHz, 65-400 Hz) over white noise:

| Hop (ms) | 5 | 10 | 15 | 20 | 25 | 30 | 40 | 45 | 50 | 60 | 70 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| floor 60 dB down | 0/12001 | 0/6001 | 0/4001 | 0/3001 | 0/2401 | 0/2001 | 1/1501 | 5/1334 | 267/1201 | 63/1001 | 21/858 |
| floor 40 dB down | 0/12001 | 0/6001 | 0/4001 | 0/3001 | 0/2401 | 0/2001 | 1/1501 | 5/1334 | 267/1201 | 63/1001 | 20/858 |

The time stretch is not meant to be librosa's. kudio keeps the phases in float64
and past the last frame holds the magnitudes and turns the phases on, so at the
end a voice differs by up to 0.8 % of the peak from librosa 0.11 and 1.1 % from
1.0. A tone at 48 kHz in float32, stretched, is measured second by second as
the best sine with its frequency left free; each cell gives how far that is from
the tone at the end, in cents, and the SINAD at the end and in its worst second:

| Tone, stretched by 1.25 | librosa 0.11.0 | librosa 1.0.1 | kudio |
|---|---|---|---|
| 187.3 Hz, 10 s | -0.032 cents; 98.1 / 58.5 dB | 0.000 cents; 99.0 / 93.8 dB | 0.000 cents; 99.1 / 93.8 dB |
| 3001.7 Hz, 10 s | +0.033 cents; 70.3 / 29.1 dB | 0.000 cents; 121.4 / 113.5 dB | 0.000 cents; 140.5 / 139.3 dB |
| 187.3 Hz, 300 s | +1.314 cents; 75.1 / 20.1 dB | 0.000 cents; 93.1 / 91.6 dB | 0.000 cents; 93.1 / 91.6 dB |
| 3001.7 Hz, 300 s | -1.514 cents; 64.4 / 3.9 dB | 0.000 cents; 97.7 / 97.6 dB | 0.000 cents; 117.7 / 117.7 dB |
| 20640 Hz, 300 s | +0.868 cents; 43.0 / -4.3 dB | 0.000 cents; 68.3 / 68.1 dB | 0.000 cents; 130.6 / 129.4 dB |

## kudio and PyWavelets

`wavelet_low_pass_filter` takes a clip apart and puts it back together with a
discrete wavelet transform of kudio's own, in periodization mode, and the filters
of the 106 discrete wavelets of PyWavelets[^pywt], copied from it. PyWavelets is
under the MIT licence, whose notice is at the head of
`kudio/enhance/_wavelet_filters.py`, and I thank its authors. Every sum runs in
the order PyWavelets' C code runs it, and for the coiflets kudio keeps the float32
filters PyWavelets computes in float32 from tables of its own, which are not its
float64 filters rounded; so the results are PyWavelets' to the bit.
`tests/test_against_pywavelets.py` holds kudio to it, and the comparison job on CI
sets `KUDIO_REQUIRE_PYWAVELETS=1`, which stops the run where PyWavelets does not
import instead of letting the tests skip.

| Compared with PyWavelets | Cases | Difference |
|---|---|---|
| the four filters of each discrete wavelet, in float64 and in float32 | 106 | 0 |
| `dwt` and `idwt`, one level each: every length from 1 to 20 samples, and 31, 101 and 1001, in float32 and float64 | 4876 each | 0 |
| `wavedec`, `threshold(mode='soft')` and `waverec` against `wavelet_low_pass_filter`: 14 lengths from 1 to 4097 samples, two thresholds in turn, float32 and float64 | 2862 | 0 |
| the same with float16, big-endian float32 and float64, int16 and uint8 samples | 40 | 0 |
| a `pywt.Wavelet`, and one made from the same filters under another name | 20 | 0 |

## kudio and mir_eval

`sdr` in `score`, [`bss_eval`][kudio.core.separation.bss_eval] and
[`sdr_improvement`][kudio.core.separation.sdr_improvement] run BSS Eval[^bss]
as mir_eval[^mireval] 0.8.2's `bss_eval_sources` runs it, step for step and call
for call, so the numbers are mir_eval's to the bit and mir_eval is not needed.
mir_eval is under the MIT licence, whose notice is at the head of
`kudio/core/_bss.py`, and I thank its authors. Where the references make the
normal equations singular, two references of two samples, one twice the other,
for instance, mir_eval 0.8.2 goes on by least squares; with numpy 2.5 the name
it catches the error by is gone and it raises `AttributeError` instead, so the
comparison puts that name back first. `tests/test_against_mir_eval.py` holds kudio to it, and the
comparison job on CI sets `KUDIO_REQUIRE_MIR_EVAL=1`.

| Compared with mir_eval 0.8.2 (numpy 2.5.5) | Cases | Difference |
|---|---|---|
| `bss_eval_sources`, one source: 1 to 3000 samples, float32 and float64 | 12 | 0 |
| two sources, with and without the permutation search, 5 to 1500 samples | 12 | 0 |
| three sources, 600 samples | 4 | 0 |
| two references that make the normal equations singular, by least squares | 1 | 0 |
| `bss_eval` with the mixture, `score(['sdr'])`, `sdr_improvement` | 4 | 0 |

## kudio and pystoi

`stoi` and `estoi` in `score` are pystoi[^pystoi] 0.4.1's steps, one for one: its
resampler to 10 kHz, its removal of silent frames, its one-third-octave bands and
its 30-frame segments. pystoi is under the MIT licence, whose notice is at the
head of `kudio/core/_stoi.py`, and I thank its author. One step is left out:
pystoi's ESTOI adds noise of machine-epsilon size, drawn from numpy's global
random state, before each of its two normalisations, which moves a seeded
script's other random numbers and ESTOI's last digits from one call to the next.
`tests/test_against_pystoi.py` holds kudio to pystoi with that noise put to
zeros, and against pystoi as it is; the comparison job on CI sets
`KUDIO_REQUIRE_PYSTOI=1`.

| Compared with pystoi 0.4.1 | Cases | Difference |
|---|---|---|
| STOI and ESTOI, its noise put to zeros: 8 to 48 kHz, 0.5 to 4 s, float32 and float64 | 72 | 0 |
| the same at the edges: exactly 30 frames left, pauses 42 dB under the peak | 4 | 0 |
| STOI, pystoi as it is | 1 | 0 |
| ESTOI, pystoi as it is | 1 | under 1e-15 |
| `score(['stoi', 'estoi'])` | 2 | 0 |

## Running the scripts

From the top of a checkout:

```bash
python tools/validation/quality.py
```

Each script imports the kudio of its checkout, prints every comparison with its
largest difference and tolerance, and exits with 1 when one is past it; rows
whose reference package is missing are skipped and named. pysepm and pySIIB
are GPL-3.0 and only run beside kudio; none of either is in kudio.

| Script | Checks | Needs besides kudio | Time |
|---|---|---|---|
| `quality.py` | LLR, IS, CEP, WSS, fwSegSNR, composite | pysepm_evo, pesq; mpmath for the 60-digit LLR | 17 s; 4 s without them |
| `lsd.py` | `lsd` | librosa | 5 s |
| `melcepstrum.py` | `mcd` | pysptk, fastdtw | 2 minutes |
| `srmr.py [--toolbox-test DIR]` | `srmr` | SRMRpy; DIR is the `test` folder of an SRMRToolbox checkout | 53 s with DIR, 49 s without; 1 s without SRMRpy |
| `wada.py [--table FILE]` | `wada_snr` | nothing; FILE is snreval's `Alpha0.400000.txt` | 14 s |
| `pesq_conformance.py --annex-a ZIP [--corrigendum-2 ZIP]` | `pesq` | pesq; ZIP, ITU's P.862 Amendment 2 software file and Corrigendum 2 file, as ITU gives them out | 5 s |
| `dnsmos.py --reference-python PY [--models DIR]` | `dnsmos`, `dnsmos_p808` | onnxruntime; PY, an interpreter with the packages `dnsmos_local.py` pins; DIR, a DNS-Challenge checkout's `DNSMOS` folder; torchmetrics and a librosa of 0.9 or later, if installed | 2 minutes; 5 s without DIR |
| `intelligibility.py [--loizou-examples DIR]` | `ncm`, `csii` | pysepm_evo; DIR holds NCM.m's example pair | about 4 minutes |
| `sii_tables.py --cran TARBALL --sii-c ZIP` | the ANSI S3.5-1997 numbers in `ncm` and `csii` | xlrd; TARBALL, the CRAN package SII 1.3.0, and ZIP, the C program developer's kit from sii.to, as they are given out | 1 s |
| `siib.py [--all]` | `siib`, `siib_gauss` against the second way and against pySIIB, and how far a sharper resampler moves them; not pySIIB's figures, which need its demo recording | nothing; pySIIB if installed, `pip install git+https://github.com/kamo-naoyuki/pySIIB.git`, which compiles C code | 2 minutes; 11 minutes with `--all` |
| `wer.py` | `wer`, `cer` | nothing; nltk if installed | 32 s, 23 of them one pair of 100,000 words |
| `verification.py` | `eer`, `min_dcf`, `det_curve` | scikit-learn | 9 s |
| `calibration.py` | `cllr`, `min_cllr`, `act_dcf`, `c_primary` | scikit-learn; mpmath for the 40-digit C_llr | 1 s |
| `diarization.py` | `der`, `jer`, `detection_error_rate` | pyannote.metrics; scipy | 4 s |
| `f0eval.py` | the F0 measures | mir_eval | 2 s |
| `separation.py` | `pit_si_sdr`, `si_sdr_improvement`, `bss_eval` | torchmetrics, mir_eval | 70 s; 3 minutes on newer numpy |
| `stats.py [--repetitions N]` | `bootstrap_ci`, `paired_ci`, `rate_ci`, `paired_rate_ci`, `permutation_test`, `holm` | scipy | 85 s; 30 s with `--repetitions 500` |
| `band_limit.py` | the band limit of `audio_report` | nothing | 2 minutes |
| `report_checks.py` | `tonal_lines`, and the clipping count and sample grid of `audio_report` | nothing | 2 minutes |
| `against_librosa.py [--all] [--long]` | resampling, the transform and its inverse, mel bands and MFCC, cutting at silence, `f0`, `time_stretch` | librosa | 1 minute with librosa 0.11, 1.5 with 1.0; 6 and 4.5 with `--all --long` |

The numbers on this page come from five environments:

- Python 3.9.13, numpy 2.0.2, scipy 1.13.1, librosa 0.11.0, soxr 1.1.0 and
  scikit-learn 1.6.1: `lsd.py`, `wada.py`, `wer.py`, `verification.py`,
  `stats.py`, `siib.py`, `band_limit.py` and `report_checks.py`, and the rows
  of `quality.py` and `srmr.py` that need no reference package.
- Python 3.10.22, numpy 2.2.6, scipy 1.15.2, librosa 0.11.0, pysepm_evo 0.1.1
  (with numba 0.67.0), pesq 0.0.4, SRMRpy 1.0 (with Gammatone 1.0.3),
  pySIIB at commit 226c2f3 and scikit-learn 1.7.2: every script but
  `wer.py`, `f0eval.py`, `separation.py`, `diarization.py`, `dnsmos.py` and
  `stats.py`; for `melcepstrum.py`, pysptk 1.0.1 built from source and
  fastdtw 0.3.4.
- Python 3.13.9, numpy 2.3.5, scipy 1.16.3, librosa 1.0.0, mir_eval 0.8.2,
  scikit-learn 1.7.2, mpmath 1.3.0, onnxruntime 1.32.0 and torchmetrics 1.9.0
  on torch 2.14.1: `lsd.py`, `f0eval.py`, `separation.py`, `calibration.py`,
  `dnsmos.py` and the 60-digit rows of `quality.py`.
- Python 3.13.9, numpy 2.5.4, scipy 1.18.1 and pyannote.metrics 4.2.0:
  `diarization.py`, and `stats.py` and the default run of `siib.py` again.
- Python 3.9.13 with the pins of DNS-Challenge's `requirements.txt`, numpy
  1.22.4, librosa 0.8.1 and onnxruntime 1.13.1, and scipy 1.9.3: the
  reference half of `dnsmos.py`.

`against_librosa.py` ran with librosa 0.11.0 on Python 3.9.13 and librosa 1.0.1
on Python 3.13.9. The times are from a desktop CPU.

## The input rule and the import without optional packages

`tests/test_metric_policy.py` holds the
[input rule](guide/metrics.md#input-a-metric-cannot-score) as two lists of
cases. Every malformed case, such as a rate a measure is not defined at, two
channels, a NaN sample or labels that are not booleans, has to raise
`ValueError` with a message starting `<function>(): `. Every case that is
well formed but leaves nothing to score, such as an empty pair, a clip
shorter than one frame, a silent reference or a trial list with one kind of
trial, has to give nan and exactly one `UserWarning` of the form
`<function>(): ...; scored nan`, whose file is the test file itself: the
warning names the caller's line. One more test calls `score` and checks that
its warnings name the line that called `score`. `pesq` is held to the rule with a stand-in for the `pesq`
package, which CI does not build. `tests/test_evaluator.py` works out `snr`,
`si_sdr`, `segmental_snr` and `sdi` from their formulas on 40 random pairs
and requires the same numbers to the last bit, so the rule changes nothing
on a pair that has something to score.

[^hl08]: Y. Hu and P. C. Loizou, "Evaluation of objective quality measures for speech enhancement", IEEE Trans. Audio, Speech, Lang. Process. 16(1), pp. 229-238, 2008, doi:10.1109/TASL.2007.911054.
[^hl06]: Y. Hu and P. C. Loizou, "Evaluation of objective measures for speech enhancement", Proc. Interspeech 2006, doi:10.21437/Interspeech.2006-84.
[^kit88]: N. Kitawaki, H. Nagabuchi and K. Itoh, "Objective quality evaluation for low-bit-rate speech coding systems", IEEE J. Sel. Areas Commun. 6(2), pp. 242-248, 1988, doi:10.1109/49.601.
[^klatt82]: D. Klatt, "Prediction of perceived phonetic distance from critical-band spectra: a first step", Proc. ICASSP 1982, vol. 7, pp. 1278-1281, doi:10.1109/ICASSP.1982.1171512.
[^urgent]: K. Saijo, W. Zhang, S. Cornell, R. Scheibler, C. Li, Z. Ni, A. Kumar, M. Sach, Y. Fu, W. Wang, T. Fingscheidt and S. Watanabe, "Interspeech 2025 URGENT speech enhancement challenge", Proc. Interspeech 2025, pp. 858-862, doi:10.21437/Interspeech.2025-1363. Code: <https://github.com/urgent-challenge/urgent2025_challenge>.
[^nvsr]: H. Liu, W. Choi, X. Liu, Q. Kong, Q. Tian and D. Wang, "Neural vocoder is all you need for speech super-resolution", Proc. Interspeech 2022, pp. 4227-4231, doi:10.21437/Interspeech.2022-11017, Eq. 7. Code: <https://github.com/haoheliu/ssr_eval>.
[^srmr]: T. H. Falk, C. Zheng and W.-Y. Chan, "A non-intrusive quality and intelligibility measure of reverberant and dereverberated speech", IEEE Trans. Audio, Speech, Lang. Process. 18(7), pp. 1766-1774, 2010, doi:10.1109/TASL.2010.2052247.
[^srmrnorm]: J. F. Santos, M. Senoussaoui and T. H. Falk, "An improved non-intrusive intelligibility metric for noisy and reverberant speech", Proc. IWAENC 2014, pp. 55-59, doi:10.1109/IWAENC.2014.6953337.
[^wada]: C. Kim and R. M. Stern, "Robust signal-to-noise ratio estimation based on waveform amplitude distribution analysis", Proc. Interspeech 2008, pp. 2598-2601, doi:10.21437/Interspeech.2008-644.
[^mhl09]: J. Ma, Y. Hu and P. C. Loizou, "Objective measures for predicting speech intelligibility in noisy conditions based on new band-importance functions", J. Acoust. Soc. Am. 125(5), pp. 3387-3405, 2009, doi:10.1121/1.3097493.
[^ka05]: J. M. Kates and K. H. Arehart, "Coherence and the speech intelligibility index", J. Acoust. Soc. Am. 117(4), pp. 2224-2237, 2005, doi:10.1121/1.1862575.
[^l13]: P. C. Loizou, *Speech Enhancement: Theory and Practice*, 2nd ed., CRC Press, 2013, doi:10.1201/b14529. The MATLAB code that comes with it holds the comp_*.m files, composite.m, NCM.m and CSII.m. composite.m alone is also in composite.zip on the author's page, <https://ecs.utdallas.edu/loizou/speech/composite.zip>.
[^ansi]: ANSI S3.5-1997, "American National Standard Methods for Calculation of the Speech Intelligibility Index", American National Standards Institute, New York, 1997.
[^siir]: G. R. Warnes and M. Shaver, "SII: Calculate ANSI/ASA S3.5-1997 (R2024) Speech Intelligibility Index", R package version 1.3.0, 2026, https://CRAN.R-project.org/package=SII.
[^siic]: C. Pavlovic, SII.C, in the C program developer's kit on the site of ASA working group S3-79, http://sii.to/programs.html.
[^jm]: D. Jurafsky and J. H. Martin, *Speech and Language Processing*, 3rd ed., draft of 19 August 2026: Ch. 2, "Words and Tokens", Sec. 2.9, "Minimum Edit Distance", pp. 28-32; Ch. 16, "Automatic Speech Recognition", Sec. 16.6, "ASR Evaluation: Word Error Rate", pp. 21-24. <https://web.stanford.edu/~jurafsky/slp3/>
[^det]: A. Martin, G. Doddington, T. Kamm, M. Ordowski and M. Przybocki, "The DET curve in assessment of detection task performance", Proc. Eurospeech 1997, pp. 1895-1898, doi:10.21437/Eurospeech.1997-504.
[^bosaris]: N. Brümmer and E. de Villiers, "The BOSARIS Toolkit: Theory, Algorithms and Code for Surviving the New DCF", NIST SRE'11 Analysis Workshop, Atlanta, December 2011. arXiv:1304.2865.
[^nist]: NIST, "The NIST Year 2008 Speaker Recognition Evaluation Plan" (3 April 2008); "The NIST Year 2010 Speaker Recognition Evaluation Plan" (version r6, 21 April 2010); "NIST 2016 Speaker Recognition Evaluation Plan" (4 August 2016); "NIST 2019 Speaker Recognition Evaluation: CTS Challenge" (22 July 2019); "NIST 2019 Speaker Recognition Evaluation Plan" (16 August 2019, the audio-visual evaluation).
[^ca09]: W. Chu and A. Alwan, "Reducing F0 frame error of F0 tracking algorithms under noisy conditions with an unvoiced/voiced classification frontend", Proc. ICASSP 2009, pp. 3969-3972, doi:10.1109/ICASSP.2009.4960497.
[^da11]: T. Drugman and A. Alwan, "Joint robust voicing detection and pitch estimation based on residual harmonics", Proc. Interspeech 2011, pp. 1973-1976, doi:10.21437/Interspeech.2011-519.
[^r76]: L. R. Rabiner, M. J. Cheng, A. E. Rosenberg and C. A. McGonegal, "A comparative performance study of several pitch detection algorithms", IEEE Trans. Acoust., Speech, Signal Process. 24(5), pp. 399-418, 1976, doi:10.1109/TASSP.1976.1162846.
[^mireval]: C. Raffel, B. McFee, E. J. Humphrey, J. Salamon, O. Nieto, D. Liang and D. P. W. Ellis, "mir_eval: a transparent implementation of common MIR metrics", Proc. ISMIR 2014, pp. 367-372, doi:10.5281/zenodo.1416528.
[^bss]: E. Vincent, R. Gribonval and C. Févotte, "Performance measurement in blind audio source separation", IEEE Trans. Audio, Speech, Lang. Process. 14(4), pp. 1462-1469, 2006, doi:10.1109/TSA.2005.858005.
[^pystoi]: M. Pariente, pystoi, <https://github.com/mpariente/pystoi>, under the MIT licence; STOI after C. H. Taal, R. C. Hendriks, R. Heusdens and J. Jensen, IEEE Trans. Audio, Speech, Lang. Process. 19(7), pp. 2125-2136, 2011, and ESTOI after J. Jensen and C. H. Taal, IEEE/ACM Trans. Audio, Speech, Lang. Process. 24(11), pp. 2009-2022, 2016.
[^upit]: M. Kolbæk, D. Yu, Z.-H. Tan and J. Jensen, "Multitalker speech separation with utterance-level permutation invariant training of deep recurrent neural networks", IEEE/ACM Trans. Audio, Speech, Lang. Process. 25(10), pp. 1901-1913, 2017, doi:10.1109/TASLP.2017.2726762.
[^convtasnet]: Y. Luo and N. Mesgarani, "Conv-TasNet: surpassing ideal time-frequency magnitude masking for speech separation", IEEE/ACM Trans. Audio, Speech, Lang. Process. 27(8), pp. 1256-1266, 2019, doi:10.1109/TASLP.2019.2915167.
[^bd06]: N. Brümmer and J. du Preez, "Application-independent evaluation of speaker detection", Computer Speech and Language 20(2-3), pp. 230-275, 2006, doi:10.1016/j.csl.2005.08.001.
[^ftki92]: T. Fukada, K. Tokuda, T. Kobayashi and S. Imai, "An adaptive algorithm for mel-cepstral analysis of speech", Proc. ICASSP 1992, vol. 1, pp. 137-140, doi:10.1109/ICASSP.1992.225953.
[^sc07]: S. Salvador and P. Chan, "Toward accurate dynamic time warping in linear time and space", Intelligent Data Analysis 11(5), pp. 561-580, 2007, doi:10.3233/IDA-2007-11508.
[^p862]: ITU-T Recommendation P.862 (02/2001), "Perceptual evaluation of speech quality (PESQ): An objective method for end-to-end speech quality assessment of narrow-band telephone networks and speech codecs"; P.862.1 (11/2003), "Mapping function for transforming P.862 raw result scores to MOS-LQO"; P.862.2 (11/2007), "Wideband extension to Recommendation P.862 for the assessment of wideband telephone networks and speech codecs". P.862 Amendment 2 (11/2005), "Revised Annex A – Reference implementations and conformance testing for ITU-T Recs P.862, P.862.1 and P.862.2", whose A.2 sets the sampling rates and whose C code, version 2.0, is the reference program; P.862 Corrigendum 2 (03/2018). ITU withdrew P.862, P.862.1, P.862.2 and P.862.3 on 5 January 2024 and points to P.863 instead.
[^dnsmos835]: C. K. A. Reddy, V. Gopal and R. Cutler, "DNSMOS P.835: A non-intrusive perceptual objective speech quality metric to evaluate noise suppressors", Proc. ICASSP 2022, pp. 886-890, doi:10.1109/ICASSP43922.2022.9746108.
[^dnsmos808]: C. K. A. Reddy, V. Gopal and R. Cutler, "DNSMOS: A non-intrusive perceptual objective speech quality metric to evaluate noise suppressors", Proc. ICASSP 2021, pp. 6493-6497, doi:10.1109/ICASSP39728.2021.9414878.
[^e87]: B. Efron, "Better bootstrap confidence intervals", J. Amer. Statist. Assoc. 82(397), pp. 171-185, 1987, doi:10.1080/01621459.1987.10478410.
[^et93]: B. Efron and R. J. Tibshirani, *An Introduction to the Bootstrap*, Chapman & Hall, 1993, doi:10.1007/978-1-4899-4541-9.
[^fw07]: C. A. Field and A. H. Welsh, "Bootstrapping clustered data", J. R. Statist. Soc. B 69(3), pp. 369-390, 2007, doi:10.1111/j.1467-9868.2007.00593.x.
[^bn04]: M. Bisani and H. Ney, "Bootstrap estimates for confidence intervals in ASR performance evaluation", Proc. ICASSP 2004, vol. 1, pp. I-409-412, doi:10.1109/ICASSP.2004.1326009.
[^ps10]: B. Phipson and G. K. Smyth, "Permutation p-values should never be zero: calculating exact p-values when permutations are randomly drawn", Stat. Appl. Genet. Mol. Biol. 9(1), Article 39, 2010, doi:10.2202/1544-6115.1585.
[^h79]: S. Holm, "A simple sequentially rejective multiple test procedure", Scand. J. Statist. 6, pp. 65-70, 1979.
[^vkh18a]: S. Van Kuyk, W. B. Kleijn and R. C. Hendriks, "An instrumental intelligibility metric based on information theory", IEEE Signal Process. Lett. 25(1), pp. 115-119, 2018, doi:10.1109/LSP.2017.2774250.
[^vkh18b]: S. Van Kuyk, W. B. Kleijn and R. C. Hendriks, "An evaluation of intrusive instrumental intelligibility metrics", IEEE/ACM Trans. Audio, Speech, Lang. Process. 26(11), pp. 2153-2166, 2018, doi:10.1109/TASLP.2018.2856374.
[^ksg04]: A. Kraskov, H. Stögbauer and P. Grassberger, "Estimating mutual information", Phys. Rev. E 69(6), 066138, 2004, doi:10.1103/PhysRevE.69.066138.
[^librosa]: B. McFee, C. Raffel, D. Liang, D. P. W. Ellis, M. McVicar, E. Battenberg and O. Nieto, "librosa: Audio and music signal analysis in Python", Proc. 14th Python in Science Conference (SciPy 2015), pp. 18-24, doi:10.25080/Majora-7b98e3ed-003. Code: <https://github.com/librosa/librosa>, under the ISC licence.
[^pywt]: G. R. Lee, R. Gommers, F. Wasilewski, K. Wohlfahrt and A. O'Leary, "PyWavelets: A Python package for wavelet analysis", Journal of Open Source Software 4(36), 1237, 2019, doi:10.21105/joss.01237. Code: <https://github.com/PyWavelets/pywt>, under the MIT licence.
[^gl84]: D. W. Griffin and J. S. Lim, "Signal estimation from modified short-time Fourier transform", IEEE Trans. Acoust., Speech, Signal Process. 32(2), pp. 236-243, 1984, doi:10.1109/TASSP.1984.1164317.
[^s98]: M. Slaney, "Auditory Toolbox, version 2", Technical Report 1998-010, Interval Research Corporation, 1998.
[^dm80]: S. B. Davis and P. Mermelstein, "Comparison of parametric representations for monosyllabic word recognition in continuously spoken sentences", IEEE Trans. Acoust., Speech, Signal Process. 28(4), pp. 357-366, 1980, doi:10.1109/TASSP.1980.1163420.
[^yin]: A. de Cheveigné and H. Kawahara, "YIN, a fundamental frequency estimator for speech and music", J. Acoust. Soc. Am. 111(4), pp. 1917-1930, 2002, doi:10.1121/1.1458024.
[^pyin]: M. Mauch and S. Dixon, "pYIN: A fundamental frequency estimator using probabilistic threshold distributions", Proc. ICASSP 2014, pp. 659-663, doi:10.1109/ICASSP.2014.6853678.
[^fg66]: J. L. Flanagan and R. M. Golden, "Phase vocoder", Bell System Technical Journal 45(9), pp. 1493-1509, 1966, doi:10.1002/j.1538-7305.1966.tb01706.x.
[^ld99]: J. Laroche and M. Dolson, "Improved phase vocoder time-scale modification of audio", IEEE Trans. Speech Audio Process. 7(3), pp. 323-332, 1999, doi:10.1109/89.759041.
