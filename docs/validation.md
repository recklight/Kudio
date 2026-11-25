# How the metrics are checked

Each metric in the table below was compared with the implementation its
published numbers come from, or, where there is none, with its definition
worked out a second way. `tonal_lines`, and the band limit, the clipping count
and the sample grid of `audio_report`, have neither, and are checked on
signals whose answer is known. The comparisons are scripts in
`tools/validation/` that anyone can run again. This page says, for each
metric, where its definition comes from, what it was compared with, on what
signals, the largest difference found, and where kudio differs from the
reference on purpose.
Differences are absolute unless marked relative.

Most of the comparisons are in the test suite as well: the tests store the
reference implementations' numbers for pairs of their own, so `pytest` holds
kudio to them without the reference installed. The ratios of the SRMR
toolbox, PESQ's conformance test and the points of pySIIB's demo figures are
the exceptions, since none of the toolbox's test recording, ITU's
conformance recordings and pySIIB's demo recording is committed.

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

The sections below have the details, and the last two say how to run the
scripts and how the input rule and the import without librosa are tested.

## Hu and Loizou's measures and the composite

Defined by
:   Hu and Loizou 2008:[^hl08] LLR eq. 4, IS eq. 5, CEP eqs. 6-7 (eq. 7 is
    the LPC cepstrum distance of Kitawaki et al.,[^kit88] their eq. 1),
    fwSegSNR eqs. 8-9 and WSS eq. 10 (after
    Klatt[^klatt82]); the composite's regressions in Hu and Loizou 2006,[^hl06]
    Sec. 4.1. Window, eps, frame count and rounding follow Loizou's MATLAB
    files (comp_llr.m, comp_is.m, comp_cep.m, comp_wss.m, comp_fwseg.m,
    composite.m).[^l13]

Compared with
:   pysepm_evo 0.1.1, a packaging of pysepm, whose quality measures port
    Loizou's files: `llr`, `cepstrum_distance`, `wss`, `fwSNRseg`, and for
    the composite's parts `llr(used_for_composite=True)` and `SNRseg`.
    pysepm has no Itakura-Saito, so comp_is.m is restated in the script and
    checked a second way, as the Itakura-Saito divergence of the two LPC
    spectra over 16384 points of the unit circle. pysepm_evo 0.1.1's own
    `composite()` raises `NameError` (it reads `mos_lqo` and `pesq_mos`
    without defining them), so the ratings are checked against the 2006
    regression computed on pysepm's parts and the PESQ of the `pesq` package
    0.0.4, picked as the original pysepm picks it.

Signals
:   The synthetic speech of `tests/test_quality.py` (syllables of a harmonic
    source through moving formants, pauses on a noise floor 60 dB down), 3 s
    long: white, pink and babble noise at -5, 0, 5, 10 and 20 dB SNR; rooms
    from `kudio.rir` with RT60 of 0.3, 0.6 and 1.2 s; kudio's log-MMSE output
    for white noise at 5 dB and babble at 0 dB; an output low-passed at 2 kHz
    under noise; and at 16 kHz a reference with nothing above 4 kHz. 21 pairs
    at 8 kHz and 22 at 16 kHz, and pairs of 30, 50, 70 and 390 frames for the
    rounding.

Largest difference
:   Relative, at 8 and 16 kHz: LLR 1.9e-12 and 1.1e-11, CEP 7.5e-13 and
    3.9e-12, WSS 1.6e-16 and 2.4e-16, fwSegSNR 3.0e-15 and 7.8e-15. IS
    against comp_is.m 1.7e-12 and 4.1e-11, and against the spectral
    divergence 7.7e-13. The composite's seven fields 1.9e-12 and 3.3e-11,
    its PESQ field 0. The largest come where an LPC fit is close to singular:
    the log-MMSE output under babble, and the 16 kHz reference with nothing
    above 4 kHz.

    Those figures need some noise in every band. An output with a band left
    empty and nothing to fill it makes the fit singular to float64
    (condition number about 1e16), and then the composite's uncapped LLR is
    decided by round-off. On the 3 s speech above at 16 kHz, the output
    low-passed at 1 kHz with no noise added scores 6.5993 here, 6.5896 in
    pysepm and 6.5800 with the same frames in 60-digit arithmetic: both are
    within 0.3% of it, and single frames are off by up to 2.97. COVL takes
    0.512 of that difference (1.3140 here, 1.3238 from the exact LLR); CSIG
    is held at 1 there, and CBAK has no LLR in it. The capped `llr`,
    `itakura_saito` and `cepstral_distance` agree to the last digit, every
    frame of such a pair sitting at its cap. At 8 kHz the same low-pass
    agrees with 60 digits to 7.7e-7, relative.

    A reference with no noise at all, a harmonic source alone, makes the
    clean frame's own fit singular the same way, and its prediction error
    can come out as round-off below zero: -7.1e-10 against a frame energy of
    0.0277 on the signal in `tests/test_quality.py`. pysepm sets a ratio at
    or below zero to 1000 before the log; kudio floors both quadratic forms
    at eps times the frame's energy. Either way the frame scores the cap of
    `llr`. Uncapped, in the composite, the two part. Here one frame of the
    129 is floored and scores 34.0, the worst of them, so it drops out with
    the worst 5%. pysepm finds three, and log 1000 = 6.91 is below every
    other frame's 9.00 or more, so they stay in the mean as its best frames.
    Under white noise at 20 dB that signal scores 2 in both, and the
    composite's uncapped LLR is 12.083 here and 11.925 in pysepm, 12.139
    with those three frames left out.

On purpose
:   LLR, IS, CEP and WSS keep the best `round(0.95 M)` of M frames rounded
    the MATLAB way, a half away from zero; pysepm rounds it to even, and at
    30, 70 or 390 frames averages one frame fewer. That is the one place the
    two part, by up to 6.25% (WSS over 30 frames at 8 kHz: 27.012824 against
    25.424200); kudio's frames averaged over pysepm's count agree to 4.2e-13,
    relative. IS holds each frame to [0, 100] as the paper does, where
    comp_is.m only caps it at 100: an exact copy of speech padded with zeros
    scores 0 here and -0.3324 under comp_is.m's range. Its log term is
    comp_is.m's, the reciprocal of the printed eq. 5. The composite's PESQ is
    wideband MOS-LQO at 16 kHz and raw narrowband at 8 kHz by default; see
    [CSIG, CBAK and COVL](guide/metrics.md#csig-cbak-and-covl).

## Log-spectral distance

Defined by
:   `lsd_metric` in the scoring script of URGENT 2025[^urgent]
    (`evaluation_metrics/calculate_intrusive_se_metrics.py`, last changed in
    commit 93e16aa of 2024-11-19), and `AudioMetrics.lsd` in NVSR's
    `ssr_eval`[^nvsr] (`ssr_eval/metrics.py`, last changed in commit 907395c
    of 2022-03-28), both read on 2026-11-13.

Compared with
:   Both definitions restated in the script on `librosa.stft`, as both
    scripts compute their STFT: librosa 0.11.0, and again librosa 1.0.0.

Signals
:   Harmonic syllables, 3 s long, at 8, 11.025, 16, 22.05, 24, 32, 44.1 and
    48 kHz: under white noise from -5 to 20 dB SNR, low-passed at a quarter
    of the rate and at 4 kHz, at half the level under noise, 1 ms late, in a
    0.4 s room, and with 0.5 s dropped out to zeros; 9 or 10 pairs a rate.
    At 16 kHz, pairs of 512 to 4800 samples, where librosa's padding decides
    the number of frames.

Largest difference
:   Relative, with librosa 0.11.0: 1.0e-11 against URGENT's definition and
    1.1e-10 against `ssr_eval`'s; with librosa 1.0.0, 1.6e-11 against
    URGENT's. The last digits move with the numpy and scipy of the run:
    numpy 2.2.6 and scipy 1.15.2 give 7.8e-12 and 5.6e-11.

On purpose
:   kudio computes in float64. URGENT's script reads files as float32, and
    `ssr_eval` does when it loads a file, so their STFTs run in single
    precision. On full-band output the two agree to 6.3e-7; where the output
    has an empty band they part, by 0.38 from URGENT's number (48 kHz,
    low-passed at 12 kHz) and 0.42 from `ssr_eval`'s (44.1 kHz, low-passed
    at 4 kHz). Rounded to 16 bits first, that second pair agrees to 3.5e-8.
    Frames where the reference is digital silence are counted, as URGENT
    counts them, at `-ln(eps)` = 18.4 each, with a warning.

## Mel-cepstral distortion

Defined by
:   `mcd_utils.py` and `mcd_metric` of URGENT 2025's scoring script[^urgent]
    (commits c726558 and 93e16aa), which port ESPnet's MCD: SPTK's `mcep`,
    the mel-cepstral analysis of Fukada et al.,[^ftki92] and FastDTW.[^sc07]

Compared with
:   That code restated on pysptk 1.0.1 and fastdtw 0.3.4's pure-Python
    version: every frame's mel-cepstrum against `pysptk.mcep`, every warping
    path against `fastdtw`, and the MCD against `mcd_metric` and, without
    the gain match, `calculate`.

Signals
:   The speech of the Hu and Loizou rows, 3 s, at 8, 16, 22.05, 24, 32, 44.1
    and 48 kHz: under white noise at 5 dB, at half the level, low-passed at a
    quarter of the rate, in a 0.4 s room, and with 1 s of digital silence
    either side; 35 pairs and 25,544 frames.

Largest difference
:   6.2e-15 in a mel-cepstral coefficient; no warping path differs; the MCD
    7.1e-15 dB.

On purpose
:   kudio scores in float64. URGENT's script reads its files as float32,
    which moves its own MCD by up to 1.1e-6 dB on these pairs. The compiled
    `fastdtw` breaks ties between equal accumulated costs another way than
    the pure-Python one followed here.

## SRMR

Defined by
:   Falk, Zheng and Chan,[^srmr] and Santos, Senoussaoui and Falk for the
    normalised variant,[^srmrnorm] as the authors' SRMRToolbox computes them
    in `SRMR.m` (github.com/MuSAELab/SRMRToolbox, head 690bdd1 of
    2023-05-23).

Compared with
:   The four ratios the toolbox keeps in `test/test.mat` for its test
    recording `test/test.wav`, which its own test script expects `SRMR.m` to
    give: 6.116783821833282 (full), 5.941239326243227 (fast),
    3.525434591882688 (fast, normalised) and 2.300403317120358 (normalised).
    And SRMRpy 1.0 (<https://github.com/jfsantos/SRMRpy>), the Python port,
    with each of its own choices put into kudio one at a time.

Signals
:   `test.wav` (0.68 s at 16 kHz, read from a checkout of the toolbox), as
    int16 and as float. Voiced syllables on a noise floor 40 dB down, 3 s
    long: dry, in rooms with RT60 of 0.3, 0.6, 1.0 and 1.5 s, under white
    noise at 10 dB and low-passed at 1 kHz; 7 clips at each of 8 and 16 kHz,
    in all four modes.

Largest difference
:   4.5e-13 relative from the stored ratios (2.3e-13 with numpy 2.0.2).
    SRMRpy as it stands parts by up to 4.0% in the full mode, 4.1% in the
    fast one and 78% in the normalised ones; with its choices put in, by
    1.1e-12 relative.

On purpose
:   kudio follows `SRMR.m`, not SRMRpy: the first 256 ms window starts
    192 ms before the signal, the Hamming window is symmetric, the Hilbert
    transform is unpadded, the fast mode takes MATLAB's `specgram`, and
    `norm=True` moves the top modulation band to 30 Hz. SRMRpy with
    `norm=True` alone keeps it at 128 Hz and gives 3.680 for the test
    recording, where the toolbox stores 2.300. Where kudio parts from the
    toolbox: rates other than 8 and 16 kHz are refused, where the toolbox
    warns and scores them; a clip shorter than one window after its silence
    is cut scores nan; and a single long silence inside a clip is cut, where
    the toolbox's `vad_s.m` keeps it. The toolbox, and so kudio, parts from
    the papers' own description in three places. It filters each envelope
    into the eight bands and frames them every 64 ms, where Falk et al. take
    a DFT of 256 ms windows every 32 ms and group its bins (sec. III-A).
    With `norm=True` it stops the bands at 30 Hz, where Santos et al. report
    4 to 40 Hz (sec. 4). And it puts the 30 dB floor under the peak of the
    window energies averaged over channels, where Santos et al. take the
    peak over channels and bands of the energies averaged over frames
    (eq. 1).

## WADA-SNR

Defined by
:   Kim and Stern,[^wada] Eqs. 1, 6, 8 and 17, with speech gamma-distributed
    of shape 0.4 and the noise Gaussian.

Compared with
:   The model computed a second way in the script: `E|a + v|` as the mean of
    a folded normal and `E ln|a + v|` as a Poisson mixture of digamma values,
    integrated over the gamma density by Gauss-Legendre panels; its closed
    forms at both ends; and a Monte Carlo run of 4e7 samples at six SNRs.
    Also the Monte Carlo table that circulates with the method,
    `Alpha0.400000.txt`, which snreval's `wada_snr.m` reads (D. Ellis,
    LabROSA, SNREVAL v0.54 of 2014-07-01,
    <http://labrosa.ee.columbia.edu/projects/snreval/>).

Signals
:   No audio for the curve: every whole dB from -20 to 110, and -200 and
    300 dB. Gamma speech in Gaussian noise, 5 clips of 10 s at each of -5, 0,
    10, 20 and 30 dB, and 40 clips of 10 s of Gaussian noise alone.

Largest difference
:   2.3e-11 in the statistic the SNR is read from, from -20 to 110 dB; 4.2e-12
    from the closed form at -200 dB; the Monte Carlo run within 1.44 standard
    errors.

On purpose
:   The curve is computed from the model rather than read off the table,
    which falls at -17 dB where the model rises: read on the model's curve,
    its entries are up to 0.25 dB off from -10 to 30 dB, 0.47 dB at 50 dB
    and 7.3 dB at 100 dB. The mean is taken out first. Exact zeros are left
    out where the ports floor them at 1e-10, so a 0 dB clip with 10% zero
    padding reads -0.06 dB here and 100 dB in snreval. Outside -20 to 100 dB
    the result is nan with a warning, where the ports return the end of the
    range.

## DNSMOS

Defined by
:   The three models DNS-Challenge distributes, `DNSMOS/sig_bak_ovr.onnx`
    (P.835),[^dnsmos835] `pDNSMOS/sig_bak_ovr.onnx` (personalised) and
    `DNSMOS/model_v8.onnx` (P.808),[^dnsmos808] as of commit 785ab12, run as
    `DNSMOS/dnsmos_local.py` runs them (commit 6412bc5 of 2023-01-10) under
    the packages its `requirements.txt` pins: numpy 1.22.4, librosa 0.8.1 and
    onnxruntime 1.13.1.

Compared with
:   That script restated and run with those pins: the P.808 model's input
    window by window, and clip by clip SIG, BAK and OVRL raw and mapped, the
    personalised model's three and the P.808 score. For `librosa="0.9"`, the
    same script under librosa 1.0.0, and torchmetrics 1.9.0's
    `deep_noise_suppression_mean_opinion_score`, which computes all seven.

Signals
:   The speech of the Hu and Loizou rows at 16 kHz, 2, 9.5, 12.3 and 15.5 s
    long: clean, under white noise at 5 dB and pink noise at 10 dB, under a
    second talker at 0 dB, in a 0.4 s room, low-passed at 4 kHz, and with
    0.5 s of digital silence either side; 28 clips and 117 windows, written
    as 64-bit float WAV files for the script to read.

Largest difference
:   0 in the P.808 model's input, with either librosa. 6.2e-7 in SIG, BAK
    and OVRL, 6.8e-7 in their raw outputs, 5.5e-7 in the personalised scores
    and 3.1e-7 in P.808. Against torchmetrics, 0 in all seven.

On purpose
:   onnxruntime 1.13.1 and 1.31.1 give some windows' outputs a float32 step
    apart, and the script averages the raw outputs and the P.808 scores in
    float32, where kudio averages in float64. Every clip here is at 16 kHz
    and under 17 s: at other rates the script resamples with resampy and
    kudio with soxr, and on a longer clip the script skips the windows whose
    end its floating-point arithmetic puts one sample short, which kudio
    keeps (see
    [Inspecting a recording](guide/inspection.md#dnsmos-what-a-listener-would-say)).
    The P.808 score on librosa 0.9's input is up to 0.19 from the score on
    0.8.1's.

## PESQ

Defined by
:   ITU-T P.862, with P.862.1's mapping to MOS-LQO and P.862.2's wideband
    extension.[^p862] The C code of P.862's Annex A as Amendment 2 revised it
    in 2005, version 2.0, is part of the Recommendation and takes precedence
    over the text where the two differ (A.1.1).

Compared with
:   The conformance data ITU-T gives out with that code. Test 2(b) of the
    Annex (A.3.2.2(b)) scores 40 pairs of recordings at 8 kHz with
    `pesq(mode="nb", scale="raw")` against the raw scores the C code gives,
    which its table lists to three decimals. kudio's `pesq` is the `pesq`
    package 0.0.4, which compiles version 2.0 of the code to read the two
    signals from memory and to return its errors as codes rather than end the
    program; kudio chooses the mode and takes the raw score back from the
    package's MOS-LQO through P.862.1's mapping.

Signals
:   The Annex's own: 40 pairs from real and simulated VoIP connections whose
    delay changes along the recording, 22 of them with sides of two lengths.

Largest difference
:   5.0e-4, the table's rounding. The Annex passes an implementation that is
    within 0.05 on all but one pair and within 0.5 on every pair. Cutting the
    longer side to the shorter one's length first, as the metrics that
    compare sample by sample do, puts the score up to 0.57 off the table,
    with 8 pairs past 0.05 and 2 past 0.5, though what the cut drops is
    silence or low noise; kudio gives PESQ the two whole.

On purpose
:   Wideband scores are those of the 2005 code, whose wideband input filter
    the `pesq` package keeps as it was published. Corrigendum 2 (2018)
    scales that filter by 0.251188 and says the uncorrected code predicts
    listeners 0.8 MOS low on average; in ITU's wideband table for the 1,736
    pairs of Supplement 23 the corrected scores are 0.73 higher on average,
    from 0.08 lower to 1.40 higher. Tests 1(a), 1(b), 2(a) and 4 of the
    Annex need the Supplement 23 recordings or ITU's G.191 resampler, which
    neither of ITU's files has, and are not run; test 3 has no data set of
    its own.

## NCM and CSII

Defined by
:   Ma, Hu and Loizou[^mhl09] for both, Kates and Arehart[^ka05] for CSII,
    as NCM.m and CSII.m compute them (F. Chen and P. C. Loizou, 2011 and
    2012, in the code that comes with Loizou's book[^l13]).

Compared with
:   pysepm_evo 0.1.1's `ncm` and `csii`, ports of NCM.m and CSII.m. And the
    example pair in the folder of NCM.m and CSII.m, against the values its
    readme prints. The band centres, importance values and band widths the
    two carry from ANSI S3.5-1997,[^ansi] against two free copies of its
    tables: the CRAN package SII 1.3.0[^siir] for Tables 1 and B.1, and the
    C program from the site of the standard's working group[^siic] for
    Table 1.

Signals
:   The same kinds of pair as for Hu and Loizou's measures: 21 at 8 kHz and
    22 at 16 kHz, and for CSII also at 11.025, 22.05, 44.1 and 48 kHz. For
    the offset note on the metrics page, references with DC offsets of
    0.001% to 5% of their peak and with pink noise floors, at 8 and 16 kHz.

Largest difference
:   NCM: 8.3e-9 at 8 kHz and 2.2e-6 at 16 kHz, all of it from the form of
    the band filters: with pysepm's filters put in place of kudio's, 4.4e-16
    and 2.2e-16. CSII: 4.4e-16 at every rate, with float64 eps added for
    pysepm as CSII.m adds it. On the example pair CSII gives the readme's
    0.5556, 0.2480 and 0.0457 to all four places. NCM gives 0.4152, as
    pysepm does, where the readme prints 0.4162; I have not found where that
    last 0.001 comes from. The tables: none. The 21 centres are Table 1's,
    the 21 importance values are Table B.1's column for short passages of
    easy reading material, and CSII's 16 widths are those of Table 1's
    lowest 16 bands; the two copies agree with each other on Table 1.

On purpose
:   NCM's band filters run as second-order sections. NCM.m runs each as one
    transfer function, which for the lowest band at 16 kHz strays from the
    exact filter by 1.6e-4 of the band's peak; the sections stay within
    2.5e-14. A squared correlation that rounds above 1, as for an exact,
    scaled or inverted copy, is held at 1, where pysepm returns nan; an
    output of digital silence scores NCM 0, where NCM.m and pysepm return
    nan; a pair too short for three envelope samples scores nan. CSII adds
    eps to both signals as CSII.m does, which pysepm does not, holds the
    coherence to [0, 1], and scores nan for a level with no frame or one
    frame, where CSII.m reports 0 for an empty level.

## SIIB

Defined by
:   Van Kuyk, Kleijn and Hendriks: SIIB in their letter,[^vkh18a] Eqs. 2
    and 8 to 12, and SIIB^Gauss in their evaluation of intrusive
    measures,[^vkh18b] Sec. IV-C, with each SIIB channel's information taken
    by the second estimator of Kraskov, Stögbauer and Grassberger,[^ksg04]
    their Eq. 9.
    Where the papers leave a detail open or the authors' MATLAB code does it
    otherwise, kudio follows the code, SIIB.m and SIIB_Gauss.m, and for the
    estimator MIxnyn, the program of Kraskov's MILCA package that SIIB.m
    calls. The module notes in the [API reference][kudio.core.siib] list
    each such place and what it does to the score.

Compared with
:   Both measures worked out a second way in `tools/validation/siib.py`,
    written to be read rather than to be fast: MATLAB's resampler summed tap
    by tap, the spectra from an explicit DFT matrix, the masking as a loop
    over its lags, the stacks filled one by one, and each channel's
    information from all N^2 distances. Every stage is compared on the same
    input, and SIIB twice: with no noise added to break ties, and through the
    chain SIIB.m runs, the noise, the 8 significant digits of the file MIxnyn
    reads and the 8 decimals of its answer, on draws of another generator.
    pySIIB, the Python port (<https://github.com/kamo-naoyuki/pySIIB>,
    commit 226c2f3 of 2022-04-04), run in the same script through its
    `SIIB()` call alone, on the same pairs and on the resampler sweep of the
    module notes. Its default window is `'hanning'`, a name scipy 1.15 no
    longer knows, so it is given `'hann'`, the window the name meant. And
    the points pySIIB plots for its demo: 15 mixtures of its 20 kHz
    recording (20.94 s) with speech-shaped noise, at SNRs from -22.55 to
    +0.31 dB, scored with kudio and set against the points read off the two
    figures pySIIB publishes. Each point there is drawn on a whole pixel, so
    the figures read to ±0.067 bit/s for SIIB^Gauss and ±0.128 for SIIB.
    SIIB.m, SIIB_Gauss.m, MILCA's MIxnyn and pySIIB are GPL-3.0, and none of
    their code is in kudio. pySIIB ran beside kudio as something to compare
    with; SIIB.m needs MATLAB and was not run.

Signals
:   For the second way, syllables of a harmonic source through three moving
    formants, two in five led by a burst of hiss, on a floor 54 dB under the
    peak, which the 40 dB threshold drops, or 24 dB under it, which it
    keeps: under white noise at -5, 0, 5 and 10 dB, in a room with an RT60
    of 0.5 s, low-passed at 2 kHz, made at 8 kHz under noise at 0 dB and at
    44.1 and 48 kHz under noise at 5 and 10 dB, and with a fade to -90 dB and
    3 s of digital silence; 10 pairs of 30 s, and in the default run 4 of
    12 s. On the first pair also a copy, an inverted copy, an output of
    digital silence and an offset of 0.5 on both signals. For pySIIB, the
    same, and the 21 pairs of the resampler sweep: the 36 s of test speech of
    `tests/test_siib.py` taken to 20 and to 8 kHz by a near-ideal filter and
    to 8 kHz by `resample_poly`'s own, under white noise at -10 to 20 dB.
    For pySIIB's figures, the demo's 15 mixtures, of which the threshold
    keeps 1638 of 1674 frames, 1623 stacks.

Largest difference
:   Up to the transform, bit for bit, but the band centres, widths and
    weights at 1.9e-14 relative and the band powers at 4.9e-15; the
    resampler 1.1e-15 relative from scipy's `resample_poly`, which kudio
    calls. The eigenvalues 2.6e-14 relative and the eigenchannels 5.0e-8 of
    a channel's spread. SIIB^Gauss under 1e-12 relative, 1.5e-13 to 4e-13 in
    the runs here, moving with the number of BLAS threads. On the same
    channels kudio's neighbour search and the all-pairs one count the same,
    so the estimator gives the same number to the bit; each channel's
    information differs by up to 3.9e-4 bits and SIIB by 1.4e-5 relative
    from the run with no noise added, and by 6.3e-6 from SIIB.m's chain on
    other draws, which is the size of the noise that breaks ties (see
    below). A copy scores the ceiling, 1335.76 bit/s, one step of float64
    over it.

    Against pySIIB at 16 kHz, on the 7 pairs of the full run and the 2 of
    the default one: SIIB^Gauss 8.2e-13 relative at most, on the pair
    low-passed at 2 kHz, and 2.2e-13 or less on the others; SIIB 1.4e-5
    relative at most, the noise that breaks ties. A copy and an inverted
    copy score the same to the bit, and an offset of 0.5 on both signals to
    1.3e-13 on SIIB^Gauss and 3.5e-6 on SIIB. On an output of digital
    silence, where nothing of the output is left but the floor under the
    log, pySIIB parts from kudio and the second way, which agree there, by
    0.8% to 1.0% on SIIB^Gauss (0.682 against 0.689 bit/s over 30 s); SIIB
    there is mostly the noise that breaks ties. At the other rates the two
    resample differently, and part by up to 8.5e-3 relative on SIIB and
    1.2e-3 on SIIB^Gauss at 8 kHz, and 2.4e-3 and 1.5e-4 at 44.1 and
    48 kHz. Over the 21 pairs of the
    resampler sweep pySIIB gives the sharper filter's scores, to 8.5e-4
    bit/s on SIIB and 3.6e-10 on SIIB^Gauss, and parts from kudio by up to
    5.1 and 1.6 bit/s.

    Against pySIIB's figures, SIIB^Gauss lands within the reading on all 15 points,
    at most 0.062 bit/s off. SIIB lands up to 3.2 bit/s off: pySIIB takes
    the pair to 16 kHz with a longer filter than MATLAB's, 365 taps from
    20 kHz where MATLAB's design has 101, which moves SIIB on these mixtures
    by -3.0 to +3.3 bit/s and SIIB^Gauss by 0.03 at most. Scoring the
    mixtures taken to 16 kHz by pySIIB's filter instead, SIIB lands within
    the reading on all 15, at most 0.127 off, with the two lowest points,
    which share a pixel row, matched the other way round.

On purpose
:   The noise that breaks ties comes from numpy's PCG64 with a fixed seed,
    where SIIB.m draws it from MATLAB's global generator in whatever state
    the session left it, a fresh session repeating its first call, and
    MIxnyn seeds its own from the data; so kudio gives a pair one score
    every time and SIIB.m moves by a few parts in 1e5 from call to call; I read nothing into a difference under
    1e-4 of the score. A pair with less than 5.45 s of speech, 436 frames,
    scores nan with a warning, where SIIB.m gives a number that depends on
    the eigen-solver; so does a reference of digital silence or of one value,
    where SIIB.m returns a small random number and SIIB_Gauss.m stops at the
    eigendecomposition. A pair of two lengths is cut to the shorter with a
    warning and refused past half, as every kudio metric takes a pair, where
    SIIB.m refuses it; a rate that is not a whole number of hertz is
    refused, and so is one under 4000 Hz, more likely a rate in kHz than a
    recording, which would be taken up more than 4 times. A SIIB^Gauss channel with no power shares nothing, where
    SIIB_Gauss.m divides 0 by 0 and scores the whole pair 0. The rest is
    rounding: the masking curve written as (1 - w) L + w floor, the exact
    digamma, the uniform noise on [0, 1), each about 1e-14 of the features.
    Rates other than 16 kHz follow SIIB.m's resampler, not pySIIB's.

## WER and CER

Defined by
:   The Levenshtein distance between the two word or character sequences,
    every edit costing 1, and WER as Jurafsky and Martin define it on that
    distance.[^jm] A corpus is pooled as Kaldi's `compute-wer` and jiwer pool
    it.

Compared with
:   Counters written in the script, sharing no code with kudio: every
    alignment of a small pair listed out; full edit-distance tables filled
    from both ends, with the most hits over the shortest paths; and the
    worked examples of Jurafsky and Martin (the CallHome utterance of
    Sec. 16.6, 6 substitutions, 1 deletion and 3 insertions in 13 words,
    76.9%; its NIST example; "intention" against "execution" in Sec. 2.9).
    nltk's `edit_distance` too when nltk is installed (I used 3.9.2). The
    rules of `compute-wer`, sclite and jiwer 4.0.0, restated from their
    code or manual.

Signals
:   Every pair of up to 6 tokens enumerated (3961), tables of up to 300
    tokens, 18,420 random pairs of up to 12 words drawn from four, and a
    simulated corpus of 100,000 words in 5,000 utterances with about 10% of
    the words wrong.

Largest difference
:   0 in every count, and so in every rate.

On purpose
:   Of the alignments with the fewest errors, kudio counts the one with the
    most hits. `compute-wer` breaks ties by a fixed order of steps and splits
    substitutions, deletions and insertions differently on 618 of the 18,420
    short pairs, with the same totals, and on none of the simulated
    utterances. sclite's weights count more errors than the fewest on 59 of
    the short pairs for certain, and on 711 more depending on its own ties.
    `cer` leaves whitespace out by default; `spaces=True` counts what jiwer
    counts on text with single spaces, which the script checks on 1000
    pairs. A word is what `str.split()` separates, where jiwer splits on the
    space character, so "hello world" against "hello\nworld" is 1.0 in jiwer
    and 0 here. A reference side with no words at all gives nan, where jiwer
    4.0 returns the number of insertions.

## EER, minimum DCF and the DET curve

Defined by
:   The operating points of the DET curve;[^det] the EER by straight lines
    between successive operating points, as the VoxSRC 2020-2023 scorer and
    NIST's SRE 2016 scoring script (version 4.1, as WeSpeaker distributes
    it) read it, or by the convex hull, the ROCCH-EER of BOSARIS[^bosaris]
    (Sec. 2.6.2); the detection cost and its normalisation as NIST's
    evaluation plans define them.[^nist]

Compared with
:   A brute-force sweep that counts the errors trial by trial at every
    distinct score, every midpoint and both ends, in exact fractions; the
    ROC hull by Andrew's monotone chain in integers, and the ROCCH-EER again
    as the largest minimum cost over every prior (BOSARIS Eq. 26);
    scikit-learn's `roc_curve` and `det_curve` (1.6.1 and 1.7.2); the VoxSRC
    scorer's `brentq` on `roc_curve`; and the rules of NIST's SRE 2016
    script, Kaldi's `compute-eer` and `compute_min_dcf.py`, voxceleb_trainer
    and SpeechBrain, restated from their code.

Signals
:   600 lists of 2 to 40 trials, with continuous scores, a few integer
    levels or scores rounded to one decimal; 6 lists of 1,000 to 10,000
    trials, some with ties; lists of 40,000 trials with Gaussian scores; one
    of 22,000 trials with its scores rounded to a tenth of their standard
    deviation; and a list of 1,003,302 trials built so that its hull has 8830
    vertices.

Largest difference
:   0 for both EERs and the DET curve against the exact sweep and the hull;
    2.2e-16 for min DCF over 8 cost settings; 1.9e-12 against VoxSRC's
    `brentq`, which is its tolerance; 1.1e-16 against NIST's script and
    against `roc_curve`; 0 against scikit-learn's `det_curve`, whose points
    are a run of kudio's; 2.7e-8 against SpeechBrain, which works in float32.

On purpose
:   Trials with the same score are one threshold. Kaldi's
    `compute_min_dcf.py` and NIST's script rank the trials one at a time and
    so put operating points between tied trials, which no threshold reaches.
    On the 22,000 trials, listed with the tied non-targets first, Kaldi's
    rule gives a min DCF of 0.39965 at `p_target` 0.05 where no threshold
    does better than 0.43615, and NIST's script reads the EER as 0.05985 or
    0.07050 depending on the order of the tied trials, where kudio reads
    0.06535 in any order. The NIST comparison
    above is on the 291 lists without a target and a non-target sharing a
    score.

## Calibration: C_llr, min C_llr, the actual DCF and C_Primary

Defined by
:   Brümmer and du Preez[^bd06] for C_llr and for min C_llr through the
    pool-adjacent-violators fit; the actual cost at the threshold log(beta)
    and C_Primary, its mean at p_target 0.01 and 0.005, from NIST's SRE 2016
    plan, Sec. 3.1.[^nist]

Compared with
:   C_llr in 40-digit arithmetic (mpmath 1.3.0). min C_llr against PYLLR's
    PAV and `min_cllr` restated from its code (github.com/bsxfan/PYLLR,
    commit 442b10b), against scikit-learn 1.7.2's `IsotonicRegression`, and
    against BOSARIS's `min_cllr.m` restated. `act_dcf` against a count in
    exact fractions and PYLLR's `fast_Bayes_error_rate` restated, at
    p_target 0.001, 0.005, 0.01, 0.05 and 0.5 and with c_miss 10;
    `c_primary` against the mean of the two costs.

Signals
:   240 lists of 4 to 400 trials: LLRs drawn from the two Gaussians they are
    the LLRs of, the same shifted and scaled, rounded to whole numbers or
    tenths so that targets and non-targets share scores, and with an LLR of
    +inf and -inf in them.

Largest difference
:   C_llr 2.2e-16 bits; min C_llr 3.3e-16 bits against PYLLR's rule and the
    isotonic fit; the actual DCF 2.8e-14; C_Primary 0.

On purpose
:   Tied scores share one LLR in min C_llr, as in PYLLR. BOSARIS's
    `min_cllr.m` sorts the non-targets of a tie first and lets the fit split
    the tie, and on the 56 lists with ties between the two kinds of trial it
    reads lower in all 56, by up to 0.30 bits; without such ties the two are
    5.2e-8 bits apart, from the 1e-6 i / N BOSARIS adds to the i-th LLR.

## Diarization: DER, JER and the detection error rate

Defined by
:   DER as NIST's RT-09 plan, Sec. 6.1,[^rt09] defines it and md-eval.pl
    (version 22, as dscore carries it) scores it: the speakers mapped on all
    the time evaluated, collars of the given seconds on each side of every
    reference boundary and the overlap taken out afterwards. JER as dscore,
    the DIHARD scorer,[^dihard] computes it; the detection error rate as
    pyannote.metrics' `DetectionErrorRate`.

Compared with
:   pyannote.metrics 4.2.0: its `DiarizationErrorRate` with no collar and the
    overlap scored; with a collar or without the overlap, its
    `IdentificationErrorRate` on the system speakers renamed by md-eval.pl's
    mapping, which its Hungarian mapper finds on the uncut annotations; its
    `DetectionErrorRate`. dscore's JER restated from its code on 10 ms
    frames (commit e02f949). DER counted again on 1 ms frames.

Signals
:   300 recordings of 60 s: 1 to 4 reference speakers and 0 to 5 system
    speakers, up to 25 turns each of 0.2 to 6 s on a 10 ms grid, one
    speaker's turns merged where they overlap.

Largest difference
:   DER 8.9e-16 in all four settings; the error components 0 s; JER
    6.7e-16 against dscore's rule; the detection error rate 4.4e-16.

On purpose
:   pyannote.metrics maps the speakers on the time it scores, so with a
    collar or without the overlap its own DER parts from md-eval.pl's: on 18
    to 38 of the 300 recordings above, by up to 0.29. Its
    `JaccardErrorRate` maps by shared time where dscore minimizes the summed
    Jaccard error, and parts on 37 of them, by up to 0.037. It counts the
    time shared by two overlapping turns of one speaker twice, which
    md-eval.pl refuses and kudio merges. md-eval.pl itself was not run
    here.

## F0 errors

Defined by
:   Chu and Alwan[^ca09] for GPE, VDE and FFE (eqs. 1-5), Drugman and
    Alwan[^da11] for FPE, the four cases of Rabiner et al.,[^r76] and the
    RMSE of `1200 log2(est / ref)` in cents.

Compared with
:   mir_eval 0.8.2,[^mireval] where its definitions meet kudio's: VDE as
    1 - overall accuracy with an unlimited cent tolerance and from its
    voicing recall and false alarm; GPE as 1 - raw pitch accuracy over the
    frames voiced in both, with the tolerance put where 20% falls, +315.6
    and -386.3 cents; FFE as 1 - overall accuracy at 350 cents on pairs with
    no error between 300 and 400 cents; the RMSE from its `hz2cents`. Every
    measure, FPE included, also against a frame-by-frame count in plain
    Python.

Signals
:   The pair the tests store mir_eval's numbers for (1000 frames), 7 pairs at
    the edges of what can be scored, from a single frame to tracks never
    voiced together, and 400 random pairs of 1 to 3000 frames.

Largest difference
:   0 in every frame count on all 408 pairs; 1.1e-16 in the rates; 4.5e-13
    cents in the RMSE; 6.9e-18 in FPE.

On purpose
:   The 20% is measured against the reference and is strict. FPE divides by
    the number of frames, as Rabiner et al. do for their standard deviation;
    MATLAB's `std` divides by one fewer, 0.5% higher at 100 frames. The RMSE
    counts gross errors. Negative values are refused rather than read as
    Keele's uncertain frames or MIREX's unvoiced guesses.

## Separation: SI-SDRi, PIT and BSS Eval

Defined by
:   SI-SDR as `si_sdr` computes it, both signals taken to zero mean as
    Conv-TasNet's SI-SNR is (Luo and Mesgarani, eq. 15[^convtasnet]); the
    assignment with the highest mean SI-SDR over the sources, the criterion
    of utterance-level PIT;[^upit] SDR, SIR and SAR by BSS Eval;[^bss] and
    the improvement as the score of the estimate minus that of the mixture
    against the same source.

Compared with
:   torchmetrics 1.9.0 on torch 2.14.1: `scale_invariant_signal_distortion_ratio`
    with `zero_mean=True` in float64, and `permutation_invariant_training`
    with it, `mode="speaker-wise"`, `eval_func="max"`. Every permutation
    tried in turn. mir_eval 0.8.2's `bss_eval_sources`, with its own
    permutation and with the one `pit_si_sdr` found, and with the mixture
    passed as every estimate for SDRi.

Signals
:   2, 3 and 4 talkers of harmonic bursts at random pitches over their own
    noise floors, each estimate holding 10% to 90% of the other talkers and
    noise, shuffled into a random order: 40 draws for each number of
    sources, 1 s at 8 kHz. BSS Eval on every fourth draw.

Largest difference
:   SI-SDR 8.9e-11 dB against torchmetrics on 1160 pairings; the PIT mean
    4.0e-13 dB, with the same permutation in all 120 draws, as the
    exhaustive search also found; BSS Eval's SDR, SIR, SAR and SDRi 0, with
    mir_eval's permutation in all 30; SI-SDRi 0 against the two SI-SDRs.

On purpose
:   torchmetrics' SI-SDR keeps the mean unless `zero_mean=True`. With a DC
    offset of 1% of the peak on the estimate, its default parts from kudio
    by up to 0.035 dB on the pairings above.

## Confidence intervals and the sign-flip test

Defined by
:   The percentile and BCa intervals of Efron 1987[^e87] and Efron and
    Tibshirani 1993[^et93] (Ch. 14), the acceleration from the jackknife
    over the units resampled; the cluster bootstrap, whole groups resampled
    with replacement, as Field and Welsh study it;[^fw07] error rates
    resampled as ratios of sums, as Bisani and Ney do for the WER;[^bn04]
    the sign-flip test, with Phipson and Smyth's (b + 1)/(B + 1) for
    patterns drawn at random;[^ps10] Holm's step-down adjustment.[^h79]

Compared with
:   `scipy.stats.bootstrap` (percentile, BCa and `paired=True`) on the same
    resamples, which kudio draws with the call scipy makes, in scipy 1.13.1
    and 1.18.1; `scipy.stats.permutation_test` with
    `permutation_type='samples'`, every sign pattern enumerated and patterns
    drawn at random; an enumeration of every sign pattern in exact
    fractions; a slow restatement that loops over the same resamples, adds
    up errors and totals item by item with `math.fsum` and recomputes the
    leave-one-group-out jackknife from the items; the exact p-value over all
    2**20 sign patterns of 20 items; and Holm's step-down procedure run at
    every level at which its decisions can change.

Signals
:   Normal and lognormal values, 5 to 1000 a set, five seeds; paired
    differences of 12 to 300 items; WER-like counts (binomial errors over
    Poisson word counts, a Beta error rate per speaker) for 6 to 18
    speakers; 2 to 16 pairs for the exact test, with and without ties, 10
    speakers of 2 to 8 items for the grouped one, and 300 sets of 6
    differences in signed tenths, whose sign patterns tie the observed sum
    in exact arithmetic but not in floats. For the simulation, 2000 data
    sets a cell: i.i.d. normal and lognormal data of 2 to 200 values, and
    speakers of 10 to 30 utterances with an intra-class correlation of 0.3.

Largest difference
:   Percentile ends 0 against `np.percentile` over scipy's
    `bootstrap_distribution`; against scipy's own ends 0 with scipy 1.13.1
    and 1.0e-15 of the interval's width with 1.18.1, which reads them with
    `scipy.stats.quantile`. BCa ends 1.3e-14 of the width (2.4e-14 with
    1.18.1), paired ones 3.1e-15 (3.6e-15); grouped intervals and rates
    1.1e-15 of the width against the restatement; groups of one item each
    against no groups, 0. Exact p-values 0 against scipy on 2 to 16 items,
    grouped ones 0 against scipy on the group sums, and 0 against the exact
    fractions on all 900 tests of signed tenths; random patterns within 1.4
    standard errors of the exact p-value, and within 3.1 of scipy's own
    random p-value. Holm 0.

    By simulation, 95% intervals that resampled speakers held the true mean
    in 89% (10 speakers) and 92% (30) of the data sets, against 52% and 50%
    resampling utterances; for a WER pooled over 30 speakers, 93% against
    67%. Each of these shares is good to about 1 point (one standard error,
    printed beside each). Under a true null the sign-flip test rejected at
    0.05 in 5.3% and 4.6% of the data sets flipping speakers, 48% and 46%
    flipping utterances, and 5.1% of 10,000 i.i.d. data sets.

On purpose
:   Where some resamples hold each item once, 3.8% of them at 5 items and
    0.24% at 8, their mean is the estimate but summed in another order can
    miss it in the last bit. scipy counts each of those as below or above
    the estimate as the rounding falls; kudio counts them as equal to it,
    which moved BCa's ends by up to 7.7% of the width at 5 items, and
    agrees with the `fsum` restatement to 2.1e-16 of the width. Where the
    observed difference is 0, scipy's tie tolerance, 100 eps times that
    difference, is 0 as well, and its one-sided p-value follows the
    rounding; kudio's tolerance also allows for the rounding of the sum. On
    the 9 sets of signed tenths whose sum is 0, kudio's 27 p-values match
    the exact fractions, and 9 of scipy's one-sided ones do not. For random
    patterns the two-sided
    p-value counts the patterns at least as large in absolute value,
    (b + 1)/(B + 1); scipy doubles the smaller one-sided p-value. Over
    every pattern the two are the same.

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

Compared with
:   No implementation: files whose history is known, made by
    `tools/validation/band_limit.py`.

Signals
:   Upsampled, 720 files of 2 s: the harmonic speech of the tests, white
    noise, and noise low-passed at 0.84 of its own Nyquist frequency; ten
    pairs of rates from 8000 -> 16000 to 96000 -> 192000 Hz; 0, -20 and
    -40 dB; scipy's `resample_poly` with its default window and with
    ("kaiser", 8.6), scipy's FFT `resample`, soxr VHQ, HQ, MQ and LQ, and
    `np.interp`; quantised to 16 bits. Codec cuts: a brick wall at 15.5, 16,
    17.5, 19.5 and 20 kHz in a 44.1 kHz file, at 0 and -20 dB. Recorded at
    their own rate, 470 files, each as float and as 16-bit over five seeds:
    speech, white noise, a spectrum falling 6 dB an octave and a random walk
    at eight rates from 8000 to 192000 Hz; at 192000 Hz a floor rising
    towards Nyquist, and bands stopping at 10, 20, 30, 50 and 80 kHz, with a
    tone over a flat floor and without one over a rising floor; at 96000 Hz
    bands stopping at 5, 10 and 20 kHz over a floor; mains hum at 48000 Hz.

Result
:   At 0 and -20 dB all 468 upsampled files the test models are flagged, the
    12 by linear interpolation at 16000 -> 44100 and 44100 -> 96000 aside;
    the 48 by whole-factor linear interpolation are found as mirror images,
    0.20 dB off the curve at most, the median within 0.09 dB. None of the 720
    is named after a rate it was not made from, and none of the 372 named is
    said to hold content past 1.06 of the Nyquist frequency of the rate it is
    named after. The content ends at 1.024 to
    1.039 of the old Nyquist frequency with `resample_poly`, 1.042 to 1.052
    with its Kaiser window, 1.000 with FFT resampling, 0.959 to 0.965 with
    soxr VHQ, HQ and MQ, and 0.861 to 0.875 with soxr LQ, whose files are
    flagged and not named; the low-passed noise, ending at 0.85 on its own,
    is not named either. At -40 dB the 70 files of speech through the seven
    resamplers that leave an edge are not flagged: the speech's bed stands
    28.6 to 33.1 dB above the 16-bit floor there, under the 40 dB an edge
    needs. All ten codec cuts are flagged, the content said to stop within
    0.03% of the cut, and the cuts at 15.5 and 16 kHz are named 32000 Hz. None of the 470 recordings is flagged. The deepest
    edge among them with a cliff of 20 dB is 34.3 dB deep, a band stopping at
    80 kHz over a rising floor; the nearest to an image is 0.60 dB off the
    curve and 1.77 dB above it, a band stopping at 30 kHz over a rising
    floor, read at 24 kHz.

On purpose
:   Rates 9% apart (11025 and 12000, 22050 and 24000, 44100 and 48000, 88200
    and 96000) are both named when the end of the content fits both: soxr
    from 24, 48 or 96 kHz is named beside the rate below it. Linear
    interpolation by a ratio that is not a whole number smears its image
    (0.39 to 0.76 dB off the curve, the median 0.93 to 1.77 dB under it) and
    is not caught. Edges above 0.9 of Nyquist, where a file's own
    anti-aliasing filter can sit, are not judged, nor are clips shorter than
    2048 samples.

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

Compared with
:   No implementation: sines and noise whose answer is known, made by
    `tools/validation/report_checks.py`.

Signals
:   Sines at 120 frequencies spread evenly in log frequency from 50 Hz to
    0.45 of the rate, at 16, 48 and 192 kHz, at -20 and -40 dBFS over white
    noise at -60 dBFS, 3 s each: 720. Sines at 440, 3150 and 9100 Hz over
    white noise of the same level, 10 s at 48 kHz, read at 10, 5 and 2.5 Hz.
    White, pink and brown noise of 1 and 10 s at the three rates, five seeds:
    90. A voice of 29 harmonics whose pitch glides from 110 to 170 Hz and
    back, and one that wavers between 125 and 155 Hz five times a second,
    five takes of 6 s each. A hum at 50 or 60 Hz with 10 harmonics beside a
    comb at 1000 or 1300 Hz with 3 and a line at 2.5 times it, at two levels
    and five seeds: 40.

Result
:   Every sine is found, and found alone: its frequency within 0.162 Hz
    (0.042 of a bin), its level within 0.52 dB. Each halving of
    `resolution_hz` raises a sine's prominence over the noise by 2.83 to
    3.15 dB. No line in the 90 noise clips. The gliding voice leaves 20 lines
    over its five takes and the wavering one 10, and neither a comb: where
    a pitch turns it dwells long enough to stand out. The two combs are
    grouped right in 40 of 40.

On purpose
:   The lines are not among `audio_report`'s findings: a voice or a note
    held at one pitch is a comb, and a turning pitch leaves lines. A line is
    measured against the median around it, so content switching on and off,
    which raises that median, hides a weaker line under it.

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

Compared with
:   No implementation: sines normalised to a peak and sines cut at full
    scale, made by `tools/validation/report_checks.py`.

Signals
:   Smooth peaks: sines at 25 frequencies from 20 Hz to 2 kHz, at 16, 44.1,
    48, 96 and 192 kHz, normalised to 1.0, 0.9999, 0.9995 and 0.9992, as
    float and as 16 bits: 1000. Plateaus: sines at the same frequencies at
    16, 48 and 192 kHz driven to 1.0005, 1.001, 1.002, 1.005, 1.01, 1.1 and
    2 times full scale and cut back to it, where they leave a plateau of
    three samples or more: 409. A voice of 19 harmonics on a wavering pitch,
    10 s at 16 kHz, driven 1.2, 2 and 5 times and cut.

Result
:   No smooth peak is counted; the nearest holds 0.93 of the share that
    counts. Every plateau of a sine driven 1.01 times or more is counted in
    full, 207 of 207. Driven less, some count in part: at 0.05% cut off
    (driven 1.0005), 5 of 42 in full and 35 in part; at 0.1%, 44 of 49 in
    full; at 0.2%, 49 of 52;
    at 0.5%, 57 of 59. The nearest plateau counted holds 1.01 of the share.
    On the voice, `full_scale_samples` is the count of samples at 0.999 or
    more, 3858, 33518 and 62310; `clip_run=1` leaves out 12, 267 and 12 of
    them, the samples on the way up to a plateau, and the default three
    counts 3492, 31608 and 62298.

On purpose
:   A cut of less than about 0.05% of the top cannot be told from a smooth
    peak and is not counted: a sine driven 1.0005 times holds, on the
    parabola, the share the threshold asks for, 0.632.

## The sample grid of `audio_report`

Defined by
:   kudio itself. Every finite sample is read at 24-bit resolution. When all
    of them are whole numbers there and 16 distinct values or more remain
    once the two values at full scale are left out, the step is the greatest
    common divisor of the gaps between neighbouring values; `sample_step` is
    the step as a fraction of full scale and `grid_bits` is
    `log2(2 / sample_step)`. Under 15.95 bits is a finding.

Compared with
:   No implementation: grids built in `tools/validation/report_checks.py`.

Signals
:   A 220 Hz tone with a tremolo over a noise bed, 2 s at 16 kHz, rounded to
    steps of 1, 16, 256, 12 and 3 at 16 bits, to 12 at 16 bits with an
    offset of 4, and to 1 at 24 bits. Without a grid: the same tone filtered
    in float, a square wave, uniform noise within 7 steps of zero at 16 bits,
    and digital silence.

Result
:   Every grid is measured exactly: 16, 12, 8, 12.4150, 14.4150, 12.4150 and
    24 bits, the step to the bit. None of the four clips without a grid is
    given one.

On purpose
:   `grid_bits` is not ENOB, which is read from a sine's SINAD; it says only
    what steps the samples take. A gain raised after quantising leaves a
    grid of its own, which is the point of the finding, and a near-silent
    take with fewer than 16 values is not judged.

## SNR, SI-SDR, STOI and SDR

`snr`, `si_sdr`, `segmental_snr` and `sdi` are a few lines of numpy each,
defined on [Measuring quality](guide/metrics.md#what-each-metric-sees).
`stoi` and `estoi` are pystoi, and `sdr` is mir_eval's BSS Eval. kudio cuts
the pair to the length it shares and passes it on, so their numbers are those
packages' numbers.

## Running the scripts

From the top of a checkout:

```bash
python tools/validation/quality.py
```

Each script imports the kudio of the checkout it sits in, prints every
comparison with its largest difference and its tolerance, and exits with 1
when a difference is past its tolerance. A script whose reference package is
missing says which rows it skipped and runs the rest. pysepm and pySIIB are
GPL-3.0; they run beside kudio as something to compare with, and none of
either is in kudio.

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
| `band_limit.py` | the band limit of `audio_report` | soxr, which librosa installs | 2 minutes |
| `report_checks.py` | `tonal_lines`, and the clipping count and sample grid of `audio_report` | nothing | 2 minutes |

The numbers on this page come from five environments:

- Python 3.9.13, numpy 2.0.2, scipy 1.13.1, librosa 0.11.0, soxr 1.1.0 and
  scikit-learn 1.6.1: `lsd.py`, `wada.py`, `wer.py`, `verification.py`,
  `stats.py`, `siib.py`, `band_limit.py` and `report_checks.py`, and the rows of `quality.py` and `srmr.py` that
  need no reference package.
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

The times are from a desktop CPU.

## The input rule and the import without librosa

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
its warnings name the line that called `score`. A new metric adds its cases
to both lists. `pesq` is held to the rule with a stand-in for the `pesq`
package, which CI does not build. `tests/test_evaluator.py` works out `snr`,
`si_sdr`, `segmental_snr` and `sdi` from their formulas on 40 random pairs
and requires the same numbers to the last bit, so the rule changes nothing
on a pair that has something to score.

`tests/test_metric_input.py` holds every metric that takes audio to the same
signals and rates. A column `(n, 1)` and a row `(1, n)` have to score as the
vector; int16, 24-bit PCM in int32, uint8 and float32 have to score with no
warning; complex, boolean, text and date arrays, `None` among the samples,
ragged rows, a single number, masked samples, two channels, int64 and uint64
have to be refused by name; int16 samples held as int32 and float audio
scaled by 1e-9 have to give a number and exactly one warning; a rate as a
0-d array or a numpy scalar has to score as the int, and text, a bool, nan,
inf, zero, a negative number, a list or None has to be refused by name. With
pystoi and mir_eval installed, STOI, ESTOI and SDR at 1e160 of full scale
have to give nan and one warning that says the pair is too loud.

`tests/test_lightweight.py` starts a fresh Python and puts `None` in
`sys.modules` for librosa, pesq, pystoi, mir_eval, pandas, matplotlib,
onnxruntime, PyAudio and sounddevice, which makes importing any of them fail
the way a missing package does, before it imports kudio. In that process
every metric module has to import, every metric that needs no optional
package has to give a finite number with every warning raised as an error,
`score` has to run every name that needs no package, a file has to be read at
its own rate, every feature built on librosa, resampling among them, has to
raise `DependencyError` naming it, `kudio report` has to run,
and librosa must still be unloaded at the end. The test then computes some of
the metrics again in its own process, librosa loaded, and they have to be the
same numbers. `tests/test_package.py` checks that `import kudio` loads none
of matplotlib, pandas, librosa, scipy.signal, PyWavelets and tqdm. One more
test puts a numba that refuses to import in front of the real one, and
`kudio.f0` has to raise `DependencyError` with numba's message and
`kudio pitch` has to exit with 2 and no traceback.

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
