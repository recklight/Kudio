# 語音辨識、語者驗證與音高追蹤 {#recognizers-verifiers-and-pitch-trackers}

這些指標評的是系統做出的判斷，而不是它對聲音做了什麼：轉寫對照實際說的話、驗證試驗對照
真正的說話者、語者分段標記對照誰在何時說話、F0 軌跡對照參考軌跡。它們不收取樣率。每個指標
拿什麼對照，見[指標驗證](../validation.md)。

## 詞錯誤率與字錯誤率 {#word-and-character-error-rates}

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

[`wer`][kudio.core.wer.wer] 讓假設和參考以最少的替換（S）、刪除（D）和插入（I）對齊，傳回
`(S + D + I) / N`，N 是參考的詞數，也就是以詞為單位的 Levenshtein 距離。[^jm]
[`word_errors`][kudio.core.wer.word_errors] 傳回這些次數；錯誤一樣少的對齊裡，kudio 取命中
最多的那個。

- 清單是依順序配對的一個語料，dict 是依語句 id 配對的一個語料。錯誤以全部參考詞數合併計算，
  不是每句平均，所以整個語料要在一次呼叫裡傳入。[`rate_ci`][kudio.core.stats.rate_ci] 和
  [`paired_rate_ci`][kudio.core.stats.paired_rate_ci] 給出信賴區間
  （[品質評估](metrics.md#confidence-intervals-and-tests)）。
- 錯誤率是比例，插入可以讓它超過 1。參考那一邊完全沒有詞時給 nan 並發出警告。
- 不做任何正規化：大小寫、標點和數字的寫法都照原樣算，所以兩邊要用同樣的方式正規化，並寫明
  怎麼做的。
- 詞是 `str.split()` 切開的東西，所以沒有空格的文字是一個詞。[`cer`][kudio.core.wer.cer] 以
  字元計，適合中文和日文；預設不算空白，`spaces=True` 會在詞之間算一個空格。
  [`char_errors`][kudio.core.wer.char_errors] 給出次數。
- 兩個 pandas Series 依位置配對；DataFrame 會被拒絕。

## 語者驗證：EER、最小 DCF 和 DET 曲線 {#speaker-verification-eer-minimum-dcf-and-the-det-curve}

每個試驗有一個分數，越可能是目標（同一個說話者）就越高，以及一個標籤，目標是 `True` 或 1。
距離要取負號再傳入。

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

[`eer`][kudio.core.verification.eer] 在相鄰操作點之間以直線讀出等錯誤率，和 NIST SRE 2016
的腳本及 VoxSRC 的評分程式一樣；`method="rocch"` 取凸包，也就是 BOSARIS 的 ROCCH-EER，[^bosaris]
它不會比預設高。其他工具用別的規則，在短的清單上結果不同，所以要寫明數字出自哪個規則。

[`min_dcf`][kudio.core.verification.min_dcf] 是最佳門檻下 NIST 的偵測成本，再除以不看分數就做
決定的成本。`p_target` 沒有預設值，因為各評測不一樣：[^nist]

| 評測 | `c_miss` | `c_fa` | `p_target` |
|---|---:|---:|---:|
| NIST SRE 2008，以及 SRE 2010 的歷史成本 | 10 | 1 | 0.01 |
| NIST SRE 2010 核心條件 | 1 | 1 | 0.001 |
| NIST SRE 2016 和 SRE 2019 CTS challenge | 1 | 1 | 0.01 和 0.005 |
| NIST SRE 2019 影音評測、VoxSRC 2020-2023 | 1 | 1 | 0.05 |

`p_target` 是應用情境的先驗機率，不是清單裡目標試驗所佔的比例。最小值是這份清單最佳門檻的
成本；要為新的試驗訂門檻，請從開發清單用 [`det_curve`][kudio.core.verification.det_curve]
取，它依門檻由小到大傳回每個操作點：

```python
curve = kudio.det_curve(scores, labels)
best = np.argmin(0.05 * curve.p_miss + 0.95 * curve.p_fa)   # VoxSRC's prior
print(f"threshold {curve.thresholds[best]:.3f}  P_miss {curve.p_miss[best]:.3f}  "
      f"P_fa {curve.p_fa[best]:.4f}")
```

```text
threshold 2.436  P_miss 0.485  P_fa 0.0069
```

分數達到門檻以上就接受，同分的試驗一起接受或一起拒絕，所以清單的順序不會改變結果。畫 DET
圖時，把兩個比率都經過 `scipy.stats.norm.ppf`。[^det]

### 校準：C_llr 和實際成本 {#calibration-c_llr-and-the-actual-cost}

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

NIST 從 SRE 2016 起要求分數是自然對數的概似比，並在 `log(beta)` 對它們做出的決定計算成本，
`beta = c_fa (1 - p_target) / (c_miss p_target)`。[`act_dcf`][kudio.core.verification.act_dcf]
是那裡的成本，[`c_primary`][kudio.core.verification.c_primary] 是它在 SRE 2016 兩個先驗 0.01 和
0.005 的平均，[`cllr`][kudio.core.verification.cllr] 是所有應用情境的成本，單位是位元（LLR 全是
0 時為 1），[`min_cllr`][kudio.core.verification.min_cllr] 是排序容許的最佳校準之後的成本。[^bd06]
C_llr 和最小 C_llr 的差，就是校準付出的代價。

## 語者分段標記：DER 和 JER {#speaker-diarization-der-and-jer}

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

每一段是以秒為單位的 `(start, end, speaker)`；[`load_rttm`][kudio.core.diarization.load_rttm]
依錄音從 RTTM 檔讀出。[`der`][kudio.core.diarization.der] 是 NIST 的語者分段錯誤率：[^rt09]
每個參考說話者對應到和它共同時間最長的系統說話者，漏掉的、誤報的和混淆的語音加總後除以參考
語音，重疊的語音每個說話者各算一次。[`jer`][kudio.core.diarization.jer] 是 DIHARD II 的 Jaccard
錯誤率，[^dihard] 每個參考說話者分開評再平均。`collar=` 是每一側的長度，和 md-eval.pl 及 dscore
一樣；說話者的對應在扣掉 collar 和重疊之前就做好。沒有這些設定，DER 數字意義不大。傳入錄音的
dict 會得到整個語料的數字。[`detection_error_rate`][kudio.core.diarization.detection_error_rate]
只評語音活動，所以 [`vad`][kudio.core.vad.vad] 的標籤可以直接傳入。

## 音高追蹤：GPE、VDE、FFE、FPE 和以音分計的 RMSE {#pitch-trackers-gpe-vde-ffe-fpe-and-the-rmse-in-cents}

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

傳入兩個以 Hz 為單位、音框相同的陣列，無聲處是 0 或 NaN；或兩個音框相同、來自
[`f0`][kudio.core.pitch.f0] 的 [`PitchTrack`][kudio.core.pitch.PitchTrack]。
[`f0_errors`][kudio.core.f0eval.f0_errors] 一次給出五個指標和背後的音框數：

| 函式 | 指標 | 算的是 | 範圍 |
|---|---|---|---|
| [`gross_pitch_error`][kudio.core.f0eval.gross_pitch_error] | GPE | `|est - ref| > 0.2 ref` 的音框 | 兩邊都有聲的音框 |
| [`voicing_decision_error`][kudio.core.f0eval.voicing_decision_error] | VDE | 有聲判斷不一致的音框 | 所有音框 |
| [`f0_frame_error`][kudio.core.f0eval.f0_frame_error] | FFE | 兩種錯誤 | 所有音框 |
| [`fine_pitch_error`][kudio.core.f0eval.fine_pitch_error] | FPE | `(est - ref) / ref` 的標準差 | 兩邊都有聲、沒有粗大誤差 |
| [`f0_rmse`][kudio.core.f0eval.f0_rmse] | RMSE | `1200 log2(est / ref)` 的均方根，單位音分 | 兩邊都有聲的音框 |

GPE、VDE 和 FFE 照 Chu 和 Alwan 的寫法，[^ca09] FPE 照 Drugman 和 Alwan 的定義。[^da11] 被判為
無聲的音框會離開 GPE 的分母，所以 GPE 要和 VDE 一起報，或只報 FFE。論文計算的音框範圍各不相同，
拿自己的數字和發表的比之前，先確認分母；整個測試集請把軌跡接起來一次算。mir_eval 算的 raw pitch
accuracy[^mireval] 不是 1 - GPE。負值會被拒絕：Keele 參考資料[^keele] 裡不確定的音框用
`keep = ref >= 0` 去掉，MIREX 的無聲猜測用 `np.maximum(est, 0)` 截掉。兩邊都沒有同時有聲的
音框時，GPE、FPE 和 RMSE 是 nan 並發出警告。參數在
[語音辨識、語者驗證與音高追蹤](../reference/task-metrics.md)。

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
