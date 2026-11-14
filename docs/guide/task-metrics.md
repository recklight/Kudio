# Recognizers, verifiers and pitch trackers

The measures on this page score what a system decided rather than what it
did to the sound: a transcript against what was said, a list of verification
trials against who was really speaking, a diarization against who spoke
when, an F0 track against a reference track. They need numpy, and scipy to
pair the speakers of a diarization, and take no sample rate. What each was
compared with, and how closely it agrees, is on
[How the metrics are checked](../validation.md).

## Word and character error rates

[`wer`][kudio.core.wer.wer] aligns the hypothesis with the reference so that
as few words as possible are substituted (S), deleted (D) or inserted (I),
and returns `(S + D + I) / N`, N being the number of reference words: the
Levenshtein distance over words, on which Jurafsky and Martin define WER.[^jm]
[`word_errors`][kudio.core.wer.word_errors] returns the counts behind it.

```python
import kudio

refs = ["the cat sat on the mat", "a dog barked"]
hyps = ["the cat sat on mat", "a dog barked twice"]
print(kudio.wer(refs, hyps))
print(kudio.word_errors(refs, hyps))
```

```text
0.2222222222222222
ErrorCounts(hits=8, substitutions=0, deletions=1, insertions=1, reference_length=9)
```

A list is one corpus, paired in order, and a dict is one corpus paired by
utterance id. The errors of every pair are added up and divided by all the
reference words, as Kaldi's `compute-wer` and jiwer do, so the result is not
the mean of the utterances' rates: one error in a 2-word utterance and none
in an 18-word one is 1/20 = 0.05 pooled, and 0.25 averaged. Pass the whole
corpus in one call. An interval around the rate, or around the gap between
two recognizers, comes from each utterance's counts through
[`rate_ci`][kudio.core.stats.rate_ci] and
[`paired_rate_ci`][kudio.core.stats.paired_rate_ci]; the example is on
[Measuring quality](metrics.md#confidence-intervals-and-tests).

Rates are fractions, not percent, and insertions can take them above 1:
`kudio.wer("yes", "yes yes yes")` is 2.0. A reference side with no words at
all gives nan with a warning; an empty reference among others is scored, its
insertions counted.

### Which errors, and what counts as a word

The total is fixed, the split into S, D and I is not: "a c" against "d d a"
is three errors either as two substitutions and an insertion or as a hit, a
deletion and two insertions. Of the alignments with the fewest errors, kudio
counts the one with the most hits. `compute-wer` breaks the tie by a fixed
order of steps and splits S, D and I differently on 618 of 18,420 random
short pairs, with the same totals. sclite weights a substitution 4 and an
insertion or deletion 3, and so can count more errors than the fewest: on 59
of those pairs for certain, and on 711 more depending on how it breaks its
own ties.

Nothing is normalised. Case, punctuation, Unicode forms and the way numbers
are written all count as they stand: "Hello, world." against "hello world"
is two substitutions, a WER of 1.0. Normalise both sides the same way first,
and say how next to the number:

```python
import re

def normalise(text):
    return re.sub(r"[^\w\s']", " ", text.lower())

print(kudio.wer("Hello, world.", "hello world"))
print(kudio.wer(normalise("Hello, world."), normalise("hello world")))
```

```text
1.0
0.0
```

A word is whatever `str.split()` separates, so text written without spaces
is one word to `wer`. [`cer`][kudio.core.wer.cer] counts characters (Unicode
code points) instead, and is the measure for Chinese and Japanese:

```python
print(kudio.wer("今天天氣很好", "今天天氣不好"))
print(kudio.cer("今天天氣很好", "今天天氣不好"))
print(kudio.cer("the cat sat", "the hat sat"))
print(kudio.cer("the cat sat", "the hat sat", spaces=True))
```

```text
1.0
0.16666666666666666
0.1111111111111111
0.09090909090909091
```

By default `cer` leaves whitespace out, as Kaldi's CER recipe does, so a
hypothesis spaced differently from its reference costs nothing.
`spaces=True` counts one space between each two words, which is what jiwer
counts on text with single spaces; the spaces join the denominator too, so
the two settings give different numbers for the same text. Say which one a
table reports. [`char_errors`][kudio.core.wer.char_errors] gives the counts.

Two pandas Series are paired by position and have to carry the same index;
pass `.to_dict()` of both to pair them by id. A DataFrame is refused, so pass
its column of transcripts.

## Speaker verification: EER, minimum DCF and the DET curve

Each trial gets a score, higher the more likely it is a target trial (the
same speaker), and a label: `True` or 1 for a target, `False` or 0 for a
non-target. Distances go in as their negatives. The eleven trials below,
ranked from the lowest score, are four non-targets, three targets, two
non-targets and two targets:

```python
scores = list(range(11))
labels = [0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1]
print(kudio.eer(scores, labels))
print(kudio.eer(scores, labels, method="rocch"))
```

```text
0.3333333333333333
0.21428571428571427
```

No threshold makes the miss and false-alarm rates of a finite list equal, so
every EER is a rule for reading between operating points, and on a short
list the rules part by whole trials:

| Rule | Eleven trials |
|---|---:|
| [`eer`][kudio.core.verification.eer], default: straight lines between successive operating points | 0.333 |
| NIST's SRE 2016 scoring script, the VoxSRC 2020-2023 scorer | 0.333 |
| `eer(method="rocch")`: the convex hull of the operating points, BOSARIS's ROCCH-EER[^bosaris] | 0.214 |
| SpeechBrain's `EER` | 0.367 |
| Kaldi's `compute-eer` | 0.4 |
| voxceleb_trainer's `tuneThresholdfromScore` | 0.6 |

I use the straight lines by default because the scorers the evaluations
published use them, and because tied scores then cannot make the result
depend on the order of the list. The NIST script agrees only while no target
and non-target share a score; with such ties its EER moves with the order of
the list. The hull is never above the default. On 40,000 trials with
Gaussian scores, half of them targets, the two read 1.090% and 1.081%, and
on six lists like it Kaldi, SpeechBrain and voxceleb_trainer stay within 0.01
percentage points of the default. Say which rule a number came from.

[`min_dcf`][kudio.core.verification.min_dcf] is NIST's detection cost at the
best threshold, divided by the cost of deciding without the scores, so 0 is
perfect and 1 is no better than always rejecting or always accepting.
`p_target` has no default, because the evaluations do not agree:[^nist]

| Evaluation | `c_miss` | `c_fa` | `p_target` |
|---|---:|---:|---:|
| NIST SRE 2008, and SRE 2010's historical cost | 10 | 1 | 0.01 |
| NIST SRE 2010, core condition | 1 | 1 | 0.001 |
| NIST SRE 2016 and the SRE 2019 CTS challenge | 1 | 1 | 0.01 and 0.005 |
| NIST SRE 2019 audio-visual, VoxSRC 2020-2023 | 1 | 1 | 0.05 |

```python
import numpy as np

rng = np.random.default_rng(0)
labels = np.r_[np.ones(2000, bool), np.zeros(18000, bool)]
scores = np.where(labels, 2.5, 0.0) + rng.standard_normal(labels.size)

print(f"EER     {kudio.eer(scores, labels):.4f}")
print(f"minDCF  {kudio.min_dcf(scores, labels, p_target=0.05):.4f}  VoxSRC")
print(f"minDCF  {kudio.min_dcf(scores, labels, p_target=0.01):.4f}  SRE 2016 at 0.01")
print(f"minDCF  {kudio.min_dcf(scores, labels, p_target=0.01, c_miss=10):.4f}  SRE 2008")
print(f"minDCF  {kudio.min_dcf(scores, labels, p_target=0.1 / 1.09):.4f}  unit costs at 0.0917")
```

```text
EER     0.1069
minDCF  0.6159  VoxSRC
minDCF  0.7975  SRE 2016 at 0.01
minDCF  0.5224  SRE 2008
minDCF  0.5224  unit costs at 0.0917
```

The last two lines are one cost spelled two ways: costs `c_miss` and `c_fa`
at a prior p are unit costs at the effective prior
`c_miss p / (c_miss p + c_fa (1 - p))`. SRE 2008's cost is therefore not the
cost at 0.01 with unit costs, which the copy of NIST's SRE 2016 script in
WeSpeaker (`wespeaker/utils/score_metrics.py`) prints under the name
minDCF08. SpeechBrain's `minDCF` returns the cost before the division by
the cost without scores.

`p_target` is the prior of the application the cost stands for, not the
share of targets in the list. At a small prior the false alarms decide: at
0.001 with unit costs, each false alarm among 10,000 non-target trials adds
0.0999, so a min DCF read off a list that size counts a handful of errors.

The minimum is taken with the labels in hand, so it is the cost of the best
threshold for this very list. To choose a threshold for new trials, take it
from a development list with [`det_curve`][kudio.core.verification.det_curve],
which returns every operating point in rising order of threshold, both ends
included:

```python
curve = kudio.det_curve(scores, labels)
cost = 0.05 * curve.p_miss + 0.95 * curve.p_fa       # VoxSRC's prior, unit costs
best = np.argmin(cost)
print(f"threshold {curve.thresholds[best]:.3f}  "
      f"P_miss {curve.p_miss[best]:.3f}  P_fa {curve.p_fa[best]:.4f}")
```

```text
threshold 2.436  P_miss 0.485  P_fa 0.0069
```

A trial is accepted when its score is at or above the threshold, and trials
with the same score are one threshold, accepted or rejected together, so the
order of the list never changes a result. For a DET plot, put both rates
through `scipy.stats.norm.ppf`.[^det] The cost parameters and the sources of
every rule above are in [`kudio.core.verification`][kudio.core.verification].

### Calibration: C_llr and the actual cost

EER and min DCF judge the ranking alone. NIST's evaluations since SRE 2016
ask for scores that are natural-log likelihood ratios, and cost the
decisions they make themselves:[^nist] a trial is accepted when its LLR is at
or above `log(beta)`, `beta = c_fa (1 - p_target) / (c_miss p_target)`, the
threshold Bayes' rule sets for the application.
[`act_dcf`][kudio.core.verification.act_dcf] is the normalized cost there,
and [`c_primary`][kudio.core.verification.c_primary] its mean at SRE 2016's
two priors, 0.01 and 0.005. [`cllr`][kudio.core.verification.cllr] costs the
LLRs over every application at once, in bits, where 1 is what LLRs of 0
throughout cost, and [`min_cllr`][kudio.core.verification.min_cllr] is what
the ranking would cost after the best calibration it allows.[^bd06] The scores
above become LLRs through the two Gaussians they were drawn from:

```python
llrs = 2.5 * scores - 3.125        # the LLR of N(2.5, 1) against N(0, 1)

for name, x in (("scores", scores), ("llrs", llrs)):
    print(f"{name:6s}  Cllr {kudio.cllr(x, labels):.3f}  "
          f"minCllr {kudio.min_cllr(x, labels):.3f}  "
          f"actDCF {kudio.act_dcf(x, labels, p_target=0.01):.3f}  "
          f"minDCF {kudio.min_dcf(x, labels, p_target=0.01):.3f}")
print(f"Cprimary {kudio.c_primary(llrs, labels):.3f}  "
      f"min {kudio.c_primary(llrs, labels, minimum=True):.3f}")
```

```text
scores  Cllr 0.672  minCllr 0.364  actDCF 0.983  minDCF 0.798
llrs    Cllr 0.372  minCllr 0.364  actDCF 0.830  minDCF 0.798
Cprimary 0.850  min 0.818
```

The two rankings are the same, so their min DCF and min C_llr agree. The raw
scores are not LLRs, and what they cost as LLRs, against what the same
ranking could cost, is the price of their calibration: 0.31 bits of C_llr,
and 0.19 of actual DCF at 0.01. The true LLRs pay 0.008 bits and 0.03.
`c_primary` pools the list as it comes, where SRE 2016 averages the cost over
partitions of the list; and `min_cllr` keeps tied scores together, where
BOSARIS's `min_cllr.m` splits them and reads lower on lists with many ties.

## Speaker diarization: DER and JER

A diarization says who spoke when, as turns `(start, end, speaker)` in
seconds; [`load_rttm`][kudio.core.diarization.load_rttm] reads them from an
RTTM file, by recording. [`der`][kudio.core.diarization.der] is the
diarization error rate of NIST's Rich Transcription evaluations:[^rt09] each
reference speaker is mapped to the system speaker it shares the most time
with, and missed speech, false-alarm speech and speaker confusion are summed
over the reference speech, overlapping speech counting once for each speaker
in it. [`jer`][kudio.core.diarization.jer], the Jaccard error rate of
DIHARD II,[^dihard] scores every reference speaker on its own and averages
them, so a speaker who said little weighs as much as one who said a lot.
Below, C talks over B from 15 to 18 s, the system starts 0.3 s late and puts
the change from A to B 2 s late, and it never finds C:

```python
reference = [(0.0, 10.0, "A"), (10.0, 20.0, "B"), (15.0, 18.0, "C")]
hypothesis = [(0.3, 12.0, "spk1"), (12.0, 20.0, "spk2")]

e = kudio.diarization_errors(reference, hypothesis)
print(f"missed {e.missed:.2f} s  false alarm {e.false_alarm:.2f} s  "
      f"confusion {e.confusion:.2f} s  of {e.total:.2f} s")
print(f"DER {kudio.der(reference, hypothesis):.4f}")
print(f"DER {kudio.der(reference, hypothesis, collar=0.25, skip_overlap=True):.4f}")
print(f"JER {kudio.jer(reference, hypothesis):.4f}")
```

```text
missed 3.30 s  false alarm 0.00 s  confusion 2.00 s  of 23.00 s
DER 0.2304
DER 0.1161
JER 0.4639
```

The second DER is the older NIST setting: 0.25 s left unscored on each side
of every reference boundary, and the overlapping speech left out. DIHARD
scores with neither, so a DER means little without its settings. The collar
here is per side, as in md-eval.pl and dscore; pyannote.metrics' collar is
the total, so its 0.5 is this 0.25. The speakers are mapped on all the
time evaluated, before the collars and the overlap are taken out, as
md-eval.pl maps them; pyannote.metrics maps on the time it scores, which
can give another DER once either is in use. `jer` pairs the speakers as dscore does,
by the smallest summed Jaccard error; pyannote.metrics pairs them by the
longest shared time, and parted from dscore on 37 of the 300 recordings of
the validation runs, by up to 0.037. A dict of recordings, as `load_rttm`
returns, gives the figure of the corpus: DER pooled over the speech of every
recording, JER averaged over every reference speaker.
[`detection_error_rate`][kudio.core.diarization.detection_error_rate]
scores speech activity alone, every speaker merged into one, so the labels
of [`vad`][kudio.core.vad.vad] can go in as they are.

## Pitch trackers: GPE, VDE, FFE, FPE and the RMSE in cents

The F0 measures compare an estimated track with a reference track frame by
frame. Give them two arrays in Hz on the same frames, with 0 or NaN where a
frame is unvoiced, or two [`PitchTrack`][kudio.core.pitch.PitchTrack]s from
[`kudio.f0`][kudio.core.pitch.f0]. [`f0_errors`][kudio.core.f0eval.f0_errors]
gives all five at once, with the frame counts behind them:

```python
ref = [0, 100, 102, 104, 0, 0, 120, 121]      # Hz, 0 where unvoiced
est = [0, 101, 204, 0, 0, 98, 119, 0]
print(kudio.f0_errors(ref, est))
```

```text
F0Errors(gpe=0.3333333333333333, vde=0.375, ffe=0.5, fpe=0.009166666666666667, rmse_cents=692.9421890055355, frames=8, voiced_both=3, gross=1, voiced_to_unvoiced=2, unvoiced_to_voiced=1)
```

| Function | Measure | What it counts | Over |
|---|---|---|---|
| [`gross_pitch_error`][kudio.core.f0eval.gross_pitch_error] | GPE | frames where `|est - ref| > 0.2 ref` | frames voiced in both |
| [`voicing_decision_error`][kudio.core.f0eval.voicing_decision_error] | VDE | frames where the two disagree on voicing | all frames |
| [`f0_frame_error`][kudio.core.f0eval.f0_frame_error] | FFE | both kinds of error | all frames |
| [`fine_pitch_error`][kudio.core.f0eval.fine_pitch_error] | FPE | standard deviation of `(est - ref) / ref` | frames voiced in both, without a gross error |
| [`f0_rmse`][kudio.core.f0eval.f0_rmse] | RMSE | RMS of `1200 log2(est / ref)`, in cents | frames voiced in both |

GPE, VDE and FFE are as Chu and Alwan write them, and FFE is their
proposal;[^ca09] FPE is as Drugman and Alwan define it.[^da11] The first
four come back as fractions; times 100 they are the percentages papers
print. In the pair above, counting frames from 0, frame 2 is an octave up,
one gross error among the three frames voiced in both, and frames 3, 5 and
7 are voicing errors, three of eight. The octave is also why the RMSE reads
693 cents: it counts gross errors at full size, where FPE leaves them out.

Two `PitchTrack`s have to share their frames, to within a microsecond. Here
the track of a clean take stands in for the reference, and the tracker runs
on the same take in noise:

```python
sr = 16000
t = np.arange(4 * sr) / sr
f0 = 120 + 20 * np.sin(2 * np.pi * 0.5 * t)               # a pitch that wanders
phase = 2 * np.pi * np.cumsum(f0) / sr
voice = sum(np.sin(k * phase) / k for k in range(1, 20))  # 19 harmonics
syllables = np.clip(np.sin(2 * np.pi * 1.5 * t), 0, None)  # sound, pause, sound...
clean = (0.1 * voice * syllables).astype(np.float32)
hiss = np.random.default_rng(7).standard_normal(len(clean))

reference = kudio.f0(clean, sr)
for snr_db in (10, 0):
    noisy = kudio.add_noise_snr(clean, hiss, snr_db=snr_db, seed=0)
    e = kudio.f0_errors(reference, kudio.f0(noisy, sr))
    print(f"{snr_db:2d} dB  GPE {e.gpe:.3f}  VDE {e.vde:.3f}  FFE {e.ffe:.3f}  "
          f"RMSE {e.rmse_cents:.1f} cents  V/U errors {e.voiced_to_unvoiced}")
```

```text
10 dB  GPE 0.000  VDE 0.022  FFE 0.022  RMSE 15.5 cents  V/U errors 8
 0 dB  GPE 0.000  VDE 0.175  FFE 0.175  RMSE 20.4 cents  V/U errors 69
```

At 0 dB the tracker gives up on 69 frames the clean take voices, and GPE
does not move, because a frame called unvoiced leaves its denominator. That
is why Chu and Alwan proposed FFE: report GPE with VDE, or FFE alone.

What else to watch for:

- Papers count over different frames. Some replace each tracker's voicing
  with the reference's and interpolate an F0 where a tracker gave none, some
  count only the frames the reference and every compared tracker call
  pitched, and some count gross errors over all frames. Check
  the denominator before setting a number beside a published one.
- Frames both tracks call unvoiced dilute VDE and FFE: 300 frames with 30
  voicing errors read 0.10, and with 100 frames of silence added at each end,
  0.06. Trim every system's tracks the same way.
- To pool a test set, concatenate the tracks and score them once. The mean of
  per-file scores weighs a short file like a long one: 5 gross errors in 10
  frames and 10 in 1000 average to a GPE of 0.255 and pool to 0.0149.
- The 20% is measured against the reference and is strict, so 120 Hz against
  100 is not a gross error. Raw pitch accuracy, as mir_eval scores
  it,[^mireval] takes a symmetric tolerance in cents and counts over the
  reference's voiced frames whatever the estimate's voicing, so 1 - RPA is
  not GPE.
- Negative values are refused, because they mean two things: an uncertain
  frame in the Keele reference,[^keele] to be dropped from both tracks
  (`keep = ref >= 0`), and an unvoiced frame with a pitch guess in MIREX
  files (`np.maximum(est, 0)`). A masked array with masked frames is refused
  for the same reason.
- Nothing is shifted in time. Synthesized speech has to be put on the
  reference's frames first, by DTW for instance, and that alignment is then
  part of the number.
- FPE is a standard deviation, so a constant offset does not reach it: an
  estimate 1% sharp on every frame has an FPE of 0 and an RMSE of 17.2 cents.
  It divides by the number of frames; MATLAB's `std` divides by one fewer.

When no frame is voiced in both tracks, GPE, FPE and the RMSE are nan with a
warning, and VDE and FFE still count every frame.

Every parameter of the functions on this page is in the API reference under
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
