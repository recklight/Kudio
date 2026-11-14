# 語音辨識、語者驗證與音高追蹤 {#recognizers-verifiers-and-pitch-trackers}

這一頁的指標評的是系統做出的判斷，而不是它對聲音做了什麼：拿轉錄文字和實際說的話比、
拿一串驗證試驗和真正說話的人比、拿語者分段標記和誰在什麼時候說話比、拿一條 F0 軌跡
和參考軌跡比。它們需要 numpy，替分段標記配對語者時還要 scipy，也不需要取樣率。每個指標拿什麼比對過、吻合到什麼程度，寫在
[指標的驗證方式](../validation.md)。

## 詞錯誤率與字元錯誤率 {#word-and-character-error-rates}

[`wer`][kudio.core.wer.wer] 在辨識結果和參考文字之間，找出被替換（S）、刪除（D）和
插入（I）的詞最少的對齊，回傳 `(S + D + I) / N`，N 是參考文字的詞數。這就是以詞為
單位的 Levenshtein 距離，Jurafsky 和 Martin 就是用這個距離定義 WER。[^jm]
[`word_errors`][kudio.core.wer.word_errors] 回傳背後的計數。

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

一個 list 是一整個語料，依順序配對；一個 dict 也是一整個語料，依語句 ID 配對。
所有配對的錯誤加總之後，除以全部的參考詞數，和 Kaldi 的 `compute-wer`、jiwer 的做法
一樣，所以結果不是各句錯誤率的平均：2 個詞的語句錯 1 個、18 個詞的語句全對，合併算是
1/20 = 0.05，逐句平均則是 0.25。整個語料請放在同一次呼叫裡。要替錯誤率加上信賴區間，
或比較兩個辨識器的差距，就把每段語句的計數交給 [`rate_ci`][kudio.core.stats.rate_ci] 和
[`paired_rate_ci`][kudio.core.stats.paired_rate_ci]，例子在[品質評估](metrics.md#confidence-intervals-and-tests)。

錯誤率是比例，不是百分比，插入太多時還會超過 1：`kudio.wer("yes", "yes yes yes")`
是 2.0。參考那一邊完全沒有詞時，得到 nan 並發出警告；空的參考文字夾在其他語句之間時
照常評分，它的插入也算進去。

### 算成哪種錯誤，什麼算一個詞 {#which-errors-and-what-counts-as-a-word}

錯誤總數是固定的，拆成 S、D、I 的方式卻不是：「a c」對「d d a」是三個錯誤，可以算成
兩個替換加一個插入，也可以算成一個命中、一個刪除加兩個插入。在錯誤最少的幾種對齊裡，
kudio 取命中最多的那一種。`compute-wer` 照固定的步驟順序決定，在 18,420 組隨機的短
配對中，有 618 組拆出的 S、D、I 和 kudio 不同，總數相同。sclite 把替換算成 4、插入或
刪除算成 3，所以算出的錯誤可能比最少的還多：在這些配對裡，確定如此的有 59 組，另外
711 組要看它自己怎麼處理平手。

這裡不做任何正規化。大小寫、標點、Unicode 的不同寫法、數字的寫法，全都照原樣算：
「Hello, world.」對「hello world」是兩個替換，WER 為 1.0。請先用同一種方式把兩邊
正規化，並在數字旁邊註明是怎麼做的：

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

一個詞就是 `str.split()` 切出來的一段，所以不用空格分詞的文字，對 `wer` 來說整句只是
一個詞。[`cer`][kudio.core.wer.cer] 改算字元（Unicode code point），中文和日文要用它：

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

`cer` 預設不算空白，和 Kaldi 的 CER 做法相同，所以辨識結果的空格和參考文字不一樣也
不扣分。`spaces=True` 會在每兩個詞之間算一個空格，和 jiwer 在單一空格的文字上算的
一樣；這些空格也會算進分母，所以同一段文字在兩種設定下的數字不同。表格裡要註明用的是
哪一種。[`char_errors`][kudio.core.wer.char_errors] 給出計數。

兩個 pandas Series 依位置配對，索引必須相同；要依 ID 配對，請兩邊都傳 `.to_dict()`。
DataFrame 會被拒絕，請傳存放轉錄文字的那一欄。

## 語者驗證：EER、最小 DCF 和 DET 曲線 {#speaker-verification-eer-minimum-dcf-and-the-det-curve}

每筆試驗有一個分數，越可能是目標試驗（同一位語者）分數越高，還有一個標籤：目標試驗是
`True` 或 1，非目標試驗是 `False` 或 0。距離要先取負號再傳進來。下面十一筆試驗從分數
最低的排起，依序是四筆非目標、三筆目標、兩筆非目標和兩筆目標：

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

在有限的試驗清單上，沒有哪個門檻能讓漏報率和誤報率剛好相等，所以每一種 EER 都是在
操作點之間取值的一種規則；清單短的時候，不同規則會差到一整筆試驗：

| 規則 | 十一筆試驗 |
|---|---:|
| [`eer`][kudio.core.verification.eer] 預設：相鄰操作點之間連直線 | 0.333 |
| NIST SRE 2016 的評分腳本、VoxSRC 2020-2023 的評分程式 | 0.333 |
| `eer(method="rocch")`：操作點的凸包，也就是 BOSARIS 的 ROCCH-EER[^bosaris] | 0.214 |
| SpeechBrain 的 `EER` | 0.367 |
| Kaldi 的 `compute-eer` | 0.4 |
| voxceleb_trainer 的 `tuneThresholdfromScore` | 0.6 |

我預設用直線，因為各評測公布的評分程式都這樣算，而且同分的試驗不會因此讓結果隨清單
順序改變。NIST 的腳本只在沒有目標和非目標試驗同分時和它一致；有這種同分時，它的 EER
會隨清單順序變動。凸包永遠不會高於預設值。在 40,000 筆分數呈高斯分布、一半是目標的
試驗上，兩者分別是 1.090% 和 1.081%；在六份類似的清單上，Kaldi、SpeechBrain 和
voxceleb_trainer 和預設值的差距都在 0.01 個百分點以內。數字是用哪一種規則算的，要
寫出來。

[`min_dcf`][kudio.core.verification.min_dcf] 是 NIST 的偵測成本在最佳門檻下的值，再除以
不看分數直接判定的成本，所以 0 是完美，1 代表不比一律拒絕或一律接受好。`p_target`
沒有預設值，因為各評測的設定不一樣：[^nist]

| 評測 | `c_miss` | `c_fa` | `p_target` |
|---|---:|---:|---:|
| NIST SRE 2008，以及 SRE 2010 沿用的舊成本 | 10 | 1 | 0.01 |
| NIST SRE 2010，核心條件 | 1 | 1 | 0.001 |
| NIST SRE 2016 和 SRE 2019 CTS 挑戰 | 1 | 1 | 0.01 和 0.005 |
| NIST SRE 2019 影音評測、VoxSRC 2020-2023 | 1 | 1 | 0.05 |

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

最後兩行是同一個成本的兩種寫法：先驗機率 p 下的成本 `c_miss` 和 `c_fa`，等於有效先驗
`c_miss p / (c_miss p + c_fa (1 - p))` 下的單位成本。所以 SRE 2008 的成本並不是單位
成本在 0.01 的成本，WeSpeaker 收錄的 NIST SRE 2016 腳本
（`wespeaker/utils/score_metrics.py`）卻把後者印成 minDCF08。
SpeechBrain 的 `minDCF` 回傳的是還沒除以不看分數成本的值。

`p_target` 是這個成本所代表的應用場合的先驗機率，不是清單裡目標試驗所佔的比例。先驗
很小時，由誤報決定結果：單位成本、0.001 時，10,000 筆非目標試驗裡每一筆誤報都會加
0.0999，所以從這麼大的清單讀出的最小 DCF，其實只數了少數幾個錯誤。

這個最小值是拿著標籤取的，所以它是對這份清單本身最好的門檻的成本。要替新的試驗選
門檻，請在開發用的清單上用 [`det_curve`][kudio.core.verification.det_curve] 挑；它依
門檻由小到大回傳每一個操作點，兩端也包括在內：

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

分數大於或等於門檻的試驗就接受；同分的試驗算同一個門檻，一起接受或一起拒絕，所以
清單的順序永遠不會改變結果。要畫 DET 圖，把兩個比率都經過
`scipy.stats.norm.ppf`。[^det] 成本參數和上面每條規則的出處，寫在
[`kudio.core.verification`][kudio.core.verification]。

### 校準：C_llr 和實際成本 {#calibration-c_llr-and-the-actual-cost}

EER 和最小 DCF 只評排序。NIST 從 SRE 2016 起，要求分數是自然對數的概似比（LLR），
並計算這些分數自己做出的決策的成本：[^nist] LLR 大於或等於 `log(beta)` 的試驗就接受，
`beta = c_fa (1 - p_target) / (c_miss p_target)`，這是貝氏決策規則為這個應用場合
訂的門檻。[`act_dcf`][kudio.core.verification.act_dcf] 是在這個門檻上的正規化成本，
[`c_primary`][kudio.core.verification.c_primary] 是它在 SRE 2016 兩個先驗機率
0.01 和 0.005 上的平均。[`cllr`][kudio.core.verification.cllr] 一次計算 LLR 在所有
應用場合上的成本，單位是 bit，1 是 LLR 全為 0 時的成本；
[`min_cllr`][kudio.core.verification.min_cllr] 則是這個排序經過它所能允許的最佳
校準之後的成本。[^bd06] 上面的分數，可以用抽出它們的那兩個高斯分布換成 LLR：

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

兩者的排序相同，所以最小 DCF 和最小 C_llr 一樣。原始分數不是 LLR，把它們當成 LLR
的成本，和同一個排序能做到的成本之間的差，就是校準的代價：C_llr 差 0.31 bit，
0.01 時的實際 DCF 差 0.19。真正的 LLR 只差 0.008 bit 和 0.03。`c_primary` 把整份清單
合在一起算，SRE 2016 則是把成本在清單的各個分區上平均；`min_cllr` 讓同分的試驗留在
一起，BOSARIS 的 `min_cllr.m` 會把它們分開，在同分很多的清單上讀出較低的值。

## 語者分段標記：DER 和 JER {#speaker-diarization-der-and-jer}

語者分段標記（diarization）說明誰在什麼時候說話，形式是以秒為單位的段落
`(start, end, speaker)`；[`load_rttm`][kudio.core.diarization.load_rttm] 可以從
RTTM 檔讀出各段錄音的段落。[`der`][kudio.core.diarization.der] 是 NIST Rich
Transcription 評測的分段錯誤率：[^rt09] 每位參考語者對應到和它共同說話時間最長的
系統語者，再把漏掉的語音、多出來的語音和認錯的語者加總，除以參考語音的總長，重疊的
語音裡有幾位語者就算幾次。[`jer`][kudio.core.diarization.jer] 是 DIHARD II 的
Jaccard 錯誤率，[^dihard] 每位參考語者各自評分再平均，所以話少的語者和話多的語者
份量一樣。下面的例子裡，C 在 15 到 18 秒和 B 同時說話，系統晚了 0.3 秒才開始，
A 換到 B 的地方晚了 2 秒，而且完全沒找到 C：

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

第二個 DER 是 NIST 早期的設定：每個參考段落邊界的兩側各留 0.25 秒不評分，也不評
重疊的語音。DIHARD 兩者都不用，所以 DER 不寫設定就沒有意義。這裡的 collar 是單側
的長度，和 md-eval.pl、dscore 相同；pyannote.metrics 的 collar 是兩側合計，所以它的
0.5 等於這裡的 0.25。語者的配對是在去掉 collar 和重疊語音之前、用所有評估的時間算出
來的，和 md-eval.pl 相同；pyannote.metrics 則用實際評分的時間來配對，所以只要用了
其中一項設定，DER 就可能不同。`jer` 照 dscore 的做法配對語者，取 Jaccard 錯誤總和最小的
配對；pyannote.metrics 則依共同說話時間最長來配對，在驗證用的 300 段錄音中有 37 段和
dscore 不同，最多差 0.037。傳入以錄音為鍵的 dict（`load_rttm` 回傳的就是這種形式）
時，得到的是整個語料的數字：DER 是所有錄音的語音合起來算，JER 是所有參考語者的平均。
[`detection_error_rate`][kudio.core.diarization.detection_error_rate] 只評語音
偵測，把所有語者合成一個，所以 [`vad`][kudio.core.vad.vad] 的標記可以直接傳入。

## 音高追蹤：GPE、VDE、FFE、FPE 和以音分計的 RMSE {#pitch-trackers-gpe-vde-ffe-fpe-and-the-rmse-in-cents}

F0 指標逐音框比較估計的軌跡和參考軌跡。可以傳入兩個在同一組音框上、以 Hz 為單位的
陣列，無聲的音框填 0 或 NaN；也可以傳入兩個 [`kudio.f0`][kudio.core.pitch.f0] 回傳的
[`PitchTrack`][kudio.core.pitch.PitchTrack]。[`f0_errors`][kudio.core.f0eval.f0_errors]
一次給出全部五個指標，連同背後的音框數：

```python
ref = [0, 100, 102, 104, 0, 0, 120, 121]      # Hz, 0 where unvoiced
est = [0, 101, 204, 0, 0, 98, 119, 0]
print(kudio.f0_errors(ref, est))
```

```text
F0Errors(gpe=0.3333333333333333, vde=0.375, ffe=0.5, fpe=0.009166666666666667, rmse_cents=692.9421890055355, frames=8, voiced_both=3, gross=1, voiced_to_unvoiced=2, unvoiced_to_voiced=1)
```

| 函式 | 指標 | 算的是 | 範圍 |
|---|---|---|---|
| [`gross_pitch_error`][kudio.core.f0eval.gross_pitch_error] | GPE | `|est - ref| > 0.2 ref` 的音框 | 兩邊都有聲的音框 |
| [`voicing_decision_error`][kudio.core.f0eval.voicing_decision_error] | VDE | 兩邊有聲／無聲判斷不一致的音框 | 所有音框 |
| [`f0_frame_error`][kudio.core.f0eval.f0_frame_error] | FFE | 兩種錯誤都算 | 所有音框 |
| [`fine_pitch_error`][kudio.core.f0eval.fine_pitch_error] | FPE | `(est - ref) / ref` 的標準差 | 兩邊都有聲、又沒有大誤差的音框 |
| [`f0_rmse`][kudio.core.f0eval.f0_rmse] | RMSE | `1200 log2(est / ref)` 的均方根，單位是音分 | 兩邊都有聲的音框 |

GPE、VDE 和 FFE 照 Chu 和 Alwan 的寫法，其中 FFE 是他們提出的；[^ca09] FPE 照 Drugman
和 Alwan 的定義。[^da11] 前四個以比例回傳，乘以 100 就是論文裡印的百分比。上面那一對
裡（音框從 0 數起），音框 2 高了一個八度，是兩邊都有聲的三個音框中的一個大誤差
（gross error）；音框 3、5、7 是有聲判斷錯誤，八個中有三個。RMSE 讀成 693 音分也是因為那個八度：它把大誤差照實算進去，
FPE 則把它們排除。

兩個 `PitchTrack` 必須共用同一組音框，誤差在一微秒以內。這裡用一段乾淨錄音的軌跡當
參考，追蹤器則在加了雜訊的同一段錄音上跑：

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

在 0 dB 時，乾淨錄音判為有聲的音框裡，有 69 個被追蹤器放棄了，GPE 卻沒有動，因為判為
無聲的音框會離開它的分母。這就是 Chu 和 Alwan 提出 FFE 的原因：GPE 要和 VDE 一起報告，
或者只報 FFE。

其他要注意的地方：

- 各篇論文計算的音框不一樣。有的把追蹤器的有聲判斷換成參考的，追蹤器沒給 F0 的地方
  用內插補上；有的只算參考和所有被比較的追蹤器都判為有音高的音框；有的則在所有音框上
  計算大誤差。拿數字和
  發表的數字並列之前，先確認分母。
- 兩條軌跡都判為無聲的音框會稀釋 VDE 和 FFE：300 個音框中有 30 個有聲判斷錯誤時是
  0.10，前後各加上 100 個靜音音框就變成 0.06。每個系統的軌跡要用同樣的方式裁切。
- 要合併整個測試集，就把軌跡接起來算一次。逐檔分數的平均，會讓短檔案和長檔案一樣重：
  10 個音框中 5 個大誤差，加上 1000 個音框中 10 個，平均起來 GPE 是 0.255，合併計算
  則是 0.0149。
- 20% 是相對於參考值、而且是嚴格大於，所以 120 Hz 對 100 Hz 不算大誤差。
  mir_eval[^mireval] 的 raw pitch accuracy 用的是以音分計的對稱容許範圍，而且不管
  估計值判為有聲還是無聲，都在參考的所有有聲音框上計算，所以 1 - RPA 不是 GPE。
- 負值會被拒絕，因為它有兩種意思：在 Keele 的參考軌跡[^keele]裡代表不確定的音框，
  應該從兩條軌跡中一起拿掉（`keep = ref >= 0`）；在 MIREX 的檔案裡則代表附帶音高猜測的無聲音框
  （`np.maximum(est, 0)`）。有被遮罩音框的 masked array 也因為同樣的理由被拒絕。
- 時間上不做任何平移。合成語音要先放到參考的音框上，例如用 DTW，而這個對齊也就成了
  數字的一部分。
- FPE 是標準差，所以固定的偏移影響不到它：每個音框都高 1% 的估計，FPE 是 0，RMSE 是
  17.2 音分。它除以音框數；MATLAB 的 `std` 則除以音框數減一。

沒有任何音框兩條軌跡都有聲時，GPE、FPE 和 RMSE 是 nan 並發出警告，VDE 和 FFE 照樣
計算所有音框。

這一頁函式的所有參數，都在 API 參考的
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
