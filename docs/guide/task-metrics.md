# Recognizers, verifiers and pitch trackers

These measures score what a system decided rather than what it did to the
sound: a transcript against what was said, verification trials against who was
speaking, a diarization against who spoke when, an F0 track against a
reference track. They take no sample rate. [How the metrics are
checked](../validation.md) has what each one was compared with.

## Word and character error rates

```python
import kudio

refs = ["the cat sat on the mat", "a dog barked"]
hyps = ["the cat sat on mat", "a dog barked twice"]
print(kudio.wer(refs, hyps))
print(kudio.word_errors(refs, hyps))
print(kudio.cer("今天天氣很好", "今天天氣不好"), kudio.wer("今天天氣很好", "今天天氣不好"))
```

```text
0.2222222222222222
ErrorCounts(hits=8, substitutions=0, deletions=1, insertions=1, reference_length=9)
0.16666666666666666 1.0
```

[`wer`][kudio.core.wer.wer] aligns the hypothesis with the reference for the
fewest substitutions (S), deletions (D) and insertions (I), and returns
`(S + D + I) / N` over the N reference words, the Levenshtein distance over
words.[^jm] [`word_errors`][kudio.core.wer.word_errors] returns the counts; of
the alignments with the fewest errors, kudio counts the one with the most hits.

- A list is one corpus paired in order, a dict one paired by utterance id. The
  errors are pooled over all reference words, not averaged per utterance, so
  pass the whole corpus in one call. [`rate_ci`][kudio.core.stats.rate_ci] and
  [`paired_rate_ci`][kudio.core.stats.paired_rate_ci] give intervals
  ([Measuring quality](metrics.md#confidence-intervals-and-tests)).
- Rates are fractions, and insertions can take them above 1. A reference side
  with no words gives nan with a warning.
- Nothing is normalised: case, punctuation and the way numbers are written count
  as they stand, so normalise both sides the same way and say how.
- A word is what `str.split()` separates, so text without spaces is one word.
  [`cer`][kudio.core.wer.cer] counts characters, for Chinese and Japanese; it
  leaves whitespace out by default, and `spaces=True` counts one space between
  words. [`char_errors`][kudio.core.wer.char_errors] gives the counts.
- Two pandas Series are paired by position; a DataFrame is refused.

## Speaker verification: EER, minimum DCF and the DET curve

Each trial has a score, higher for a likelier target (same speaker), and a label,
`True` or 1 for a target. Distances go in as their negatives.

```python
import numpy as np

rng = np.random.default_rng(0)
labels = np.r_[np.ones(2000, bool), np.zeros(18000, bool)]
scores = np.where(labels, 2.5, 0.0) + rng.standard_normal(labels.size)
print(f"EER     {kudio.eer(scores, labels):.4f}")
print(f"minDCF  {kudio.min_dcf(scores, labels, p_target=0.05):.4f}  VoxSRC")
print(f"minDCF  {kudio.min_dcf(scores, labels, p_target=0.01, c_miss=10):.4f}  SRE 2008")
```

```text
EER     0.1069
minDCF  0.6159  VoxSRC
minDCF  0.5224  SRE 2008
```

[`eer`][kudio.core.verification.eer] reads the equal error rate off straight
lines between successive operating points, as NIST's SRE 2016 script and the
VoxSRC scorer do; `method="rocch"` takes the convex hull, BOSARIS's
ROCCH-EER,[^bosaris] which is never higher. Other toolkits use other rules and
differ on short lists, so say which rule a number came from.

[`min_dcf`][kudio.core.verification.min_dcf] is NIST's detection cost at the
best threshold, divided by the cost of deciding without the scores. `p_target`
has no default, because the evaluations differ:[^nist]

| Evaluation | `c_miss` | `c_fa` | `p_target` |
|---|---:|---:|---:|
| NIST SRE 2008, and SRE 2010's historical cost | 10 | 1 | 0.01 |
| NIST SRE 2010, core condition | 1 | 1 | 0.001 |
| NIST SRE 2016 and the SRE 2019 CTS challenge | 1 | 1 | 0.01 and 0.005 |
| NIST SRE 2019 audio-visual, VoxSRC 2020-2023 | 1 | 1 | 0.05 |

`p_target` is the prior of the application, not the share of targets in the
list. The minimum is the cost of the best threshold for this list; to set a
threshold for new trials, take it from a development list with
[`det_curve`][kudio.core.verification.det_curve], which returns every operating
point in rising order of threshold:

```python
curve = kudio.det_curve(scores, labels)
best = np.argmin(0.05 * curve.p_miss + 0.95 * curve.p_fa)   # VoxSRC's prior
print(f"threshold {curve.thresholds[best]:.3f}  P_miss {curve.p_miss[best]:.3f}  "
      f"P_fa {curve.p_fa[best]:.4f}")
```

```text
threshold 2.436  P_miss 0.485  P_fa 0.0069
```

A trial is accepted at or above the threshold, and tied scores are accepted or
rejected together, so the order of the list never changes a result. For a DET
plot, put both rates through `scipy.stats.norm.ppf`.[^det]

### Calibration: C_llr and the actual cost

```python
llrs = 2.5 * scores - 3.125        # the LLR of N(2.5, 1) against N(0, 1)
for name, x in (("scores", scores), ("llrs", llrs)):
    print(f"{name:6s}  Cllr {kudio.cllr(x, labels):.3f}  "
          f"minCllr {kudio.min_cllr(x, labels):.3f}  "
          f"actDCF {kudio.act_dcf(x, labels, p_target=0.01):.3f}")
print(f"Cprimary {kudio.c_primary(llrs, labels):.3f}")
```

```text
scores  Cllr 0.672  minCllr 0.364  actDCF 0.983
llrs    Cllr 0.372  minCllr 0.364  actDCF 0.830
Cprimary 0.850
```

NIST's evaluations since SRE 2016 ask for natural-log likelihood ratios and cost
the decisions they make at `log(beta)`, `beta = c_fa (1 - p_target) / (c_miss
p_target)`. [`act_dcf`][kudio.core.verification.act_dcf] is the cost there,
[`c_primary`][kudio.core.verification.c_primary] its mean at SRE 2016's priors
0.01 and 0.005, [`cllr`][kudio.core.verification.cllr] the cost over every
application in bits (1 for LLRs of 0 throughout), and
[`min_cllr`][kudio.core.verification.min_cllr] the cost after the best
calibration the ranking allows.[^bd06] The gap between C_llr and min C_llr is the
price of the calibration.

## Speaker diarization: DER and JER

```python
reference = [(0.0, 10.0, "A"), (10.0, 20.0, "B"), (15.0, 18.0, "C")]
hypothesis = [(0.3, 12.0, "spk1"), (12.0, 20.0, "spk2")]
e = kudio.diarization_errors(reference, hypothesis)
print(f"missed {e.missed:.2f} s  false alarm {e.false_alarm:.2f} s  "
      f"confusion {e.confusion:.2f} s  of {e.total:.2f} s")
print(f"DER {kudio.der(reference, hypothesis):.4f}  "
      f"DER {kudio.der(reference, hypothesis, collar=0.25, skip_overlap=True):.4f}  "
      f"JER {kudio.jer(reference, hypothesis):.4f}")
```

```text
missed 3.30 s  false alarm 0.00 s  confusion 2.00 s  of 23.00 s
DER 0.2304  DER 0.1161  JER 0.4639
```

Turns are `(start, end, speaker)` in seconds; [`load_rttm`][kudio.core.diarization.load_rttm]
reads them from an RTTM file, by recording. [`der`][kudio.core.diarization.der]
is NIST's diarization error rate:[^rt09] each reference speaker mapped to the
system speaker it shares most time with, missed, false-alarm and confused speech
summed over the reference speech, overlap counting once per speaker.
[`jer`][kudio.core.diarization.jer], DIHARD II's Jaccard error rate,[^dihard]
scores each reference speaker on its own and averages them. `collar=` is per
side, as in md-eval.pl and dscore, and the speakers are mapped before the collars
and overlap are taken out; a DER means little without these settings. A dict of
recordings gives the corpus figure.
[`detection_error_rate`][kudio.core.diarization.detection_error_rate] scores
speech activity alone, so [`vad`][kudio.core.vad.vad] labels go in as they are.

## Pitch trackers: GPE, VDE, FFE, FPE and the RMSE in cents

```python
ref = [0, 100, 102, 104, 0, 0, 120, 121]      # Hz, 0 where unvoiced
est = [0, 101, 204, 0, 0, 98, 119, 0]
e = kudio.f0_errors(ref, est)
print(f"GPE {e.gpe:.3f}  VDE {e.vde:.3f}  FFE {e.ffe:.3f}  FPE {e.fpe:.4f}  "
      f"RMSE {e.rmse_cents:.1f} cents  frames {e.frames}")
```

```text
GPE 0.333  VDE 0.375  FFE 0.500  FPE 0.0092  RMSE 692.9 cents  frames 8
```

Give two arrays in Hz on the same frames, 0 or NaN where unvoiced, or two
[`PitchTrack`][kudio.core.pitch.PitchTrack]s from [`f0`][kudio.core.pitch.f0]
that share their frames. [`f0_errors`][kudio.core.f0eval.f0_errors] gives all
five with the frame counts:

| Function | Measure | What it counts | Over |
|---|---|---|---|
| [`gross_pitch_error`][kudio.core.f0eval.gross_pitch_error] | GPE | frames where `|est - ref| > 0.2 ref` | frames voiced in both |
| [`voicing_decision_error`][kudio.core.f0eval.voicing_decision_error] | VDE | frames where the voicing disagrees | all frames |
| [`f0_frame_error`][kudio.core.f0eval.f0_frame_error] | FFE | both kinds of error | all frames |
| [`fine_pitch_error`][kudio.core.f0eval.fine_pitch_error] | FPE | standard deviation of `(est - ref) / ref` | voiced in both, no gross error |
| [`f0_rmse`][kudio.core.f0eval.f0_rmse] | RMSE | RMS of `1200 log2(est / ref)`, in cents | frames voiced in both |

GPE, VDE and FFE are as Chu and Alwan write them,[^ca09] FPE as Drugman and
Alwan define it.[^da11] A frame called unvoiced leaves GPE's denominator, so
report GPE with VDE, or FFE alone. Papers count over different frames, so check
the denominator before setting a number beside a published one; pool a test set
by concatenating the tracks. Raw pitch accuracy as mir_eval scores it[^mireval]
is not 1 - GPE. Negative values are refused: an uncertain frame in the Keele
reference[^keele] is dropped with `keep = ref >= 0`, a MIREX unvoiced guess
clipped with `np.maximum(est, 0)`. When no frame is voiced in both, GPE, FPE and
the RMSE are nan with a warning. The parameters are under
[Recognizers, verifiers and pitch trackers](../reference/task-metrics.md).

[^jm]: D. Jurafsky and J. H. Martin, *Speech and Language Processing*, 3rd ed., draft of 19 August 2026: Ch. 2, "Words and Tokens", Sec. 2.9, "Minimum Edit Distance", pp. 28-32; Ch. 16, "Automatic Speech Recognition", Sec. 16.6, "ASR Evaluation: Word Error Rate", pp. 21-24. <https://web.stanford.edu/~jurafsky/slp3/>
[^bosaris]: N. Brümmer and E. de Villiers, "The BOSARIS Toolkit: Theory, Algorithms and Code for Surviving the New DCF", NIST SRE'11 Analysis Workshop, Atlanta, December 2011, Sec. 2.6.2. arXiv:1304.2865.
[^nist]: NIST, "The NIST Year 2008 Speaker Recognition Evaluation Plan" (3 April 2008), Sec. 3; "The NIST Year 2010 Speaker Recognition Evaluation Plan" (version r6, 21 April 2010), Sec. 3; "NIST 2016 Speaker Recognition Evaluation Plan" (4 August 2016), Sec. 3.1; "NIST 2019 Speaker Recognition Evaluation: CTS Challenge" (22 July 2019), Sec. 3.1; "NIST 2019 Speaker Recognition Evaluation Plan" (16 August 2019, the audio-visual evaluation), Sec. 3.1. VoxSRC 2020-2023 take p_target 0.05 with unit costs, as the last does.
[^det]: A. Martin, G. Doddington, T. Kamm, M. Ordowski and M. Przybocki, "The DET curve in assessment of detection task performance", Proc. Eurospeech 1997, pp. 1895-1898, doi:10.21437/Eurospeech.1997-504.
[^ca09]: W. Chu and A. Alwan, "Reducing F0 frame error of F0 tracking algorithms under noisy conditions with an unvoiced/voiced classification frontend", Proc. ICASSP 2009, pp. 3969-3972, doi:10.1109/ICASSP.2009.4960497.
[^da11]: T. Drugman and A. Alwan, "Joint robust voicing detection and pitch estimation based on residual harmonics", Proc. Interspeech 2011, pp. 1973-1976, doi:10.21437/Interspeech.2011-519.
[^keele]: F. Plante, G. F. Meyer and W. A. Ainsworth, "A pitch extraction reference database", Proc. Eurospeech 1995, pp. 837-840, doi:10.21437/Eurospeech.1995-191.
[^mireval]: C. Raffel, B. McFee, E. J. Humphrey, J. Salamon, O. Nieto, D. Liang and D. P. W. Ellis, "mir_eval: a transparent implementation of common MIR metrics", Proc. ISMIR 2014, pp. 367-372, doi:10.5281/zenodo.1416528.
[^bd06]: N. Brümmer and J. du Preez, "Application-independent evaluation of speaker detection", Computer Speech and Language 20(2-3), pp. 230-275, 2006, doi:10.1016/j.csl.2005.08.001.
[^rt09]: NIST, "The 2009 (RT-09) Rich Transcription Meeting Recognition Evaluation Plan", 2009, Sec. 6.1; scored with NIST's `md-eval.pl`.
[^dihard]: N. Ryant, K. Church, C. Cieri, A. Cristia, J. Du, S. Ganapathy and M. Liberman, "Second DIHARD Challenge Evaluation Plan", version 1.1, 2019; scored with dscore, <https://github.com/nryant/dscore>.
