# 品質評估 {#measuring-quality}

這裡的每個指標都拿處理過的錄音和它來源的乾淨錄音比；沒有乾淨錄音時，請看
[檢查錄音](inspection.md)。SNR、SI-SDR、segmental SNR 和 SDI 拿第 *i* 個樣本和第 *i* 個樣本比，
所以只要晚一個樣本，完美的複本 SI-SDR 就掉到大約 +12 dB，而分數本身不會告訴你原因是延遲。
挑指標之前，先把兩段對齊。

## 第一個分數 {#a-first-score}

範例共用一段片段：兩句諧波音疊在房間底噪上，混上 5 dB SNR 的白雜訊，再用 log-MMSE 降噪。

```python
import numpy as np
import kudio

sr = 16000
rng = np.random.default_rng(0)

def burst(seconds, f0):
    """A harmonic tone that swells four times a second, like syllables."""
    t = np.arange(int(seconds * sr)) / sr
    tone = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 12))
    return tone * (0.5 + 0.5 * np.sin(2 * np.pi * 4 * t)) / 6

clean = 0.01 * rng.standard_normal(4 * sr)       # room tone
clean[8000:24000] += burst(1.0, 120)             # two phrases with
clean[32000:52800] += burst(1.3, 150)            # a pause between
clean = clean.astype(np.float32)

noise = rng.standard_normal(4 * sr).astype(np.float32)
noisy = kudio.add_noise_snr(clean, noise, snr_db=5, seed=0)
enhanced = kudio.spectral_enhance(noisy, sr)     # log-MMSE

for name, value in kudio.score(clean, enhanced, sr).items():
    print(f"{name:7s} {value:7.3f}")
```

```text
snr      12.002
si_sdr   11.725
segsnr    5.120
sdi       0.063
stoi      0.601
estoi     0.276
sdr      13.325
```

[`score`][kudio.core.evaluator.score] 先收參考訊號，再收要評的訊號，傳回 dict。不給清單時算
`snr`、`si_sdr`、`segsnr`、`sdi`、`stoi`、`estoi` 和 `sdr`，裝了 `kudio[pesq]` 且取樣率是
8 或 16 kHz 時再加 `pesq`；頁面下方的指標要指名才會算。每個指標也都是參數順序相同的獨立函式。
不認得的名稱丟出 `ValueError`，缺套件丟出寫明套件組的
[`DependencyError`][kudio.exceptions.DependencyError]。
[`check_metrics_install`][kudio.core.evaluator.check_metrics_install] 傳回
`(pesq_ok, stoi_ok, sdr_ok)`；[`eval_metrics`][kudio.core.evaluator.eval_metrics] 先收取樣率，
一律傳回 `(pesq, stoi, sdr)`，算不了的是 `None`。

## 先把兩段對齊 {#line-the-pair-up-first}

![SI-SDR 對延遲的關係：晚一個樣本的完美複本得 +11.9 dB；從晚四個樣本起，複本和 log-MMSE 輸出的分數都低於沒處理過的有雜訊輸入](../images/metrics-delay-light.webp#only-light)
![SI-SDR 對延遲的關係：晚一個樣本的完美複本得 +11.9 dB；從晚四個樣本起，複本和 log-MMSE 輸出的分數都低於沒處理過的有雜訊輸入](../images/metrics-delay-dark.webp#only-dark)

晚四個樣本時，複本和降噪後的結果都比沒處理過的有雜訊輸入分數還低。濾波器、重新取樣、會補
音框的模型和重新錄音都會造成延遲，而被裁回輸入長度的檔案不會有任何警告。

### 找出延遲 {#finding-the-delay}

```python
processed = np.concatenate([np.zeros(128, np.float32), enhanced])  # 8 ms late
found = kudio.find_delay(clean, processed, sr)
print(found)
a, b = kudio.align(clean, processed, sr)
print(len(a), len(b), round(kudio.si_sdr(a, b), 2))
print(kudio.find_delay(clean, noise, sr))
```

```text
lags by 128 samples (8.0 ms), correlation 0.968
64000 64000 11.72
leads by 5873 samples (367.1 ms), correlation 0.016, polarity inverted
```

[`find_delay`][kudio.core.align.find_delay] 用互相關找出精確到樣本的延遲，相關係數只在重疊
部分上正規化；`delay` 為正表示第二個訊號開始得較晚。[`align`][kudio.core.align.align] 傳回裁到
共同部分的一對，[`Alignment.apply`][kudio.core.align.Alignment.apply] 用已經量到的延遲裁之後的
其他對。互相關一定有最大值，所以兩段不相干的訊號也會得到一個延遲：低於
[`MIN_CORRELATION`][kudio.core.align.MIN_CORRELATION]（0.5）時 `align` 會丟出 `FeatureError`
拒絕。已知延遲很小時，用 `max_seconds=` 限制搜尋範圍。反相的複本會回報 `polarity inverted`，
`align` 記一筆，原樣傳回。兩者都收單聲道、取樣率都是你傳入的那個 `sr`。`kudio align clean.wav
processed.wav --metrics` 對一對檔案做同樣的事。

## 長度不同時 {#when-the-lengths-differ}

```python
import warnings

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    kudio.snr(clean, clean[:60000])
print(caught[0].message)
```

```text
snr(): ref is 64000 samples and deg 60000: only the first 60000 samples are scored and the rest is ignored. Use kudio.align if the short one is delayed rather than just short, or trim the pair yourself to say which part you meant.
```

`snr`、`si_sdr`、`segmental_snr`、`sdi`，以及 `score` 裡 PESQ 以外的指標，會把兩個訊號裁到
共同的長度並發出警告。較長那個有超過一半會量不到時
（[`MAX_LENGTH_MISMATCH`][kudio.core.evaluator.MAX_LENGTH_MISMATCH]，0.5），這一對會被
`ValueError` 拒絕；真的要評長度差很多的一對，請設定 `kudio.core.evaluator.MAX_LENGTH_MISMATCH`。
PESQ 會拿到完整的兩個訊號，就像 ITU 的程式讀兩個檔案那樣，因為它自己找延遲。
[`AudioEvaluate`][kudio.core.evaluator.AudioEvaluate] 對每一對都做同樣的檢查。

## 各指標看得到什麼 {#what-each-metric-sees}

```python
def late(y, n):
    return np.concatenate([np.zeros(n, np.float32), y])[:len(y)]

cases = {"noise, 5 dB SNR": noisy, "half the gain": 0.5 * clean,
         "polarity flipped": -clean, "low-pass, 2 kHz": kudio.lowpass(clean, sr, cutoff=2000),
         "1 sample late": late(clean, 1), "1 ms late": late(clean, 16)}
names = ["snr", "si_sdr", "segsnr", "sdi", "stoi", "sdr"]
print(f"{'':17}" + "".join(f"{n:>8}" for n in names))
for case, deg in cases.items():
    s = kudio.score(clean, deg, sr, names)
    print(f"{case:17}" + "".join(f"{s[n]:8.2f}" for n in names))
```

```text
                      snr  si_sdr  segsnr     sdi    stoi     sdr
noise, 5 dB SNR      5.00    4.97   -2.11    0.32    0.61    5.01
half the gain        6.02  138.88    6.02    0.25    1.00  283.00
polarity flipped    -6.02  144.90   -6.02    4.00    1.00  283.00
low-pass, 2 kHz     18.24   18.17   10.31    0.02    0.93   24.95
1 sample late       12.09   11.89    4.58    0.06    1.00   91.30
1 ms late           -1.45   -9.97   -2.17    1.40    0.99   53.57
```

超過 100 dB 表示指標找不到任何問題；那個數字來自 1e-12 的保護值或捨入誤差。
[`snr`][kudio.core.evaluator.snr] 什麼都算，音量也算。[`si_sdr`][kudio.core.evaluator.si_sdr]
先去掉平均和最適合的增益，[^sisdr] 所以音量和極性不扣分，濾波和延遲照扣：評降噪器時先看這個。

- `segsnr`、[`segmental_snr`][kudio.core.evaluator.segmental_snr]：逐音框的 SNR，限制在
  [-10, 35] dB 再平均；經由 `score` 時音框 512 個樣本、跳距 256。停頓和語音一樣算，參考訊號
  數位靜音的音框算 -10 dB 並發出警告。
- [`sdi`][kudio.core.evaluator.sdi]：`sum((ref - deg)**2) / sum(ref**2)`，完美複本是 0；算在
  整個輸出上，所以不是降噪文獻裡把雜訊排除在外的語音失真指數。[^sdi]
- `stoi`、`estoi`：短時客觀可懂度[^stoi] 和它的延伸版，是相關係數的平均，介於 -1 和 1 之間，
  語音什麼都沒留下時會略低於 0；音量、極性和一毫秒的延遲幾乎不影響。剩不到大約 0.4 s 的語音時
  給 nan 並發出警告。
- `sdr`：BSS Eval 的 SDR，[^bss] 原諒從參考到輸出的最佳 512 階因果濾波器；只能和同樣算法的 SDR
  比。

各指標對同一個降噪器的看法不一致：這裡 log-MMSE 讓 SI-SDR 多 6.8 dB，STOI 卻沒動，ESTOI 還
下降。

## PESQ：要講清楚是哪一個 {#pesq-say-which-one}

PESQ 有兩個標準、兩種刻度。[^p862] P.862 評窄頻語音，原始分數在 -0.5 到 4.5，或經 P.862.1
對映成 MOS-LQO；寬頻延伸 P.862.2 給 MOS-LQO，在 16 kHz 執行。
[`pesq`][kudio.core.evaluator.pesq] 用 `mode=` 指定頻寬、`scale=` 指定刻度，`score`、
`eval_metrics` 和 `AudioEvaluate` 以 `pesq_mode=` 和 `pesq_scale=` 轉交。它需要 `kudio[pesq]`：

| `mode` | `scale` | 8 kHz | 16 kHz |
|---|---|---|---|
| `'auto'`（預設） | `'lqo'`（預設） | P.862，MOS-LQO | P.862.2，MOS-LQO |
| `'nb'` | `'lqo'` | P.862，MOS-LQO | P.862，MOS-LQO |
| `'wb'` | `'lqo'` | 拒絕 | P.862.2，MOS-LQO |
| `'auto'` 或 `'nb'` | `'raw'` | P.862，原始分數 | P.862，原始分數 |
| `'wb'` | `'raw'` | 拒絕 | 拒絕 |

```python
kudio.pesq(clean, enhanced, 16000)                          # P.862.2 MOS-LQO
kudio.pesq(clean, enhanced, 16000, mode="nb", scale="raw")  # raw P.862
```

其他取樣率丟出 `ValueError`。原始分數 2.0 是窄頻 MOS-LQO 的 1.63，所以兩個參數都要傳，並在表裡
寫明刻度。`pesq` 套件的寬頻程式是 2005 年的參考程式；ITU 2018 年的修正，也就是 Torcoli 等人的
`pesqc2`，[^torcoli] 寬頻分數平均高 0.73。

## Hu 和 Loizou 的指標 {#hu-and-loizous-measures}

```python
names = ["llr", "is", "cep", "wss", "fwsegsnr"]
for label, deg in (("noisy", noisy), ("enhanced", enhanced)):
    s = kudio.score(clean, deg, sr, names)
    print(f"{label:8s}", "  ".join(f"{n} {s[n]:.3f}" for n in names))
```

```text
noisy    llr 0.388  is 1.648  cep 3.228  wss 23.960  fwsegsnr 9.678
enhanced llr 0.283  is 2.324  cep 3.293  wss 35.328  fwsegsnr 11.118
```

對數概似比、Itakura-Saito 距離、倒頻譜距離、加權頻譜斜率和頻率加權 segmental SNR，[^hl08] 照
Loizou 的 MATLAB 程式計算：[`llr`][kudio.core.quality.llr]、
[`itakura_saito`][kudio.core.quality.itakura_saito]、
[`cepstral_distance`][kudio.core.quality.cepstral_distance]、[`wss`][kudio.core.quality.wss] 和
[`fw_segmental_snr`][kudio.core.quality.fw_segmental_snr]。它們在 8 和 16 kHz、以 30 ms 音框
計算，所以要先對齊。前四個越低越好，fwSegSNR 越高越好。LLR、IS、CEP 和 WSS 只平均最好的 95%
音框；WSS 的 25 個頻帶在兩種取樣率下都只到 3.8 kHz，低於 -100 dB 的頻帶一律當 -100 dB，所以
很小聲的一對要先調到正常音量（`wss` 會警告）。

### CSIG、CBAK 和 COVL {#csig-cbak-and-covl}

```python
c = kudio.composite(clean, enhanced, sr, pesq=1.88)   # a PESQ score you have
print(f"CSIG {c.csig:.3f}  CBAK {c.cbak:.3f}  COVL {c.covl:.3f}  LLR {c.llr:.3f}  WSS {c.wss:.3f}")
```

```text
CSIG 3.617  CBAK 2.595  COVL 2.715  LLR 0.283  WSS 35.328
```

[`composite`][kudio.core.quality.composite] 用 PESQ、LLR、WSS 和一個 segmental SNR 預測 P.835[^p835]
的三個 1 到 5 分評分。PESQ 由 `kudio[pesq]` 算，或用 `pesq=` 傳入你自己的；經由 `score` 時是
`csig`、`cbak` 和 `covl`。擬合用的是窄頻原始 PESQ，而大多數已發表的 16 kHz 數字裡面是寬頻
MOS-LQO，`composite` 在 16 kHz 預設也用這個；`mode="nb", scale="raw"` 給另一種，兩者的 COVL
差大約 0.8，所以表裡要寫明用的是哪種 PESQ。[`CompositeScore`][kudio.core.quality.CompositeScore]
帶著它的四個輸入。

## 對數頻譜距離與梅爾倒頻譜失真 {#log-spectral-and-mel-cepstral-distance}

```python
low = kudio.lowpass(clean, sr, cutoff=2000)
print(f"LSD {kudio.lsd(clean, low, sr):.2f} ln, {kudio.lsd(clean, low, sr, scale='log10'):.2f} log10")
for label, deg in (("noisy", noisy), ("enhanced", enhanced)):
    print(f"{label:8s} MCD {kudio.mcd(clean, deg, sr):.2f} dB")
```

```text
LSD 18.00 ln, 7.82 log10
noisy    MCD 6.85 dB
enhanced MCD 5.40 dB
```

[`lsd`][kudio.core.evaluator.lsd] 是對數功率比在頻率上的均方根，再對音框平均，頻譜相同時為 0；
它看得到 SNR 家族平均掉的缺頻帶。預設給出 URGENT 2025 挑戰的數字：[^urgent] 自然對數、每 16 ms
一個 32 ms 音框、輸出先乘上最小平方增益；44.1 kHz 下 `scale="log10", n_fft=2048, hop_length=441,
match_gain=False, eps=1e-12` 給出 NVSR 的 `ssr_eval`。[^nvsr] `ln`、`log10` 和 `db` 三種刻度只差
一個常數倍，所以要先確認發表的數字是哪一種。

[`mcd`][kudio.core.melcepstrum.mcd] 是動態時間校正配對音框後的梅爾倒頻譜失真，單位 dB，照
URGENT 2025 的腳本計算：SPTK 的分析、依取樣率查表的階數、含 c0、半徑 1 的 FastDTW、匹配增益。
因為有時間校正，時間不一致的一對也能評，那時請傳 `match_gain=False`。它內建 8、16、22.05、24、
32、44.1 和 48 kHz 的設定；其他取樣率請自己給 `order` 和 `alpha`。各家實作不同，只比同一套程式
算出的 MCD。

## 可懂度：NCM、CSII 和 SIIB {#intelligibility-ncm-csii-and-siib}

```python
for label, deg in (("noisy", noisy), ("enhanced", enhanced)):
    high, mid, low = kudio.csii(clean, deg, sr)
    print(f"{label:8s} NCM {kudio.ncm(clean, deg, sr):.3f}  "
          f"CSII high {high:.3f}  mid {mid:.3f}  low {low:.3f}")
```

```text
noisy    NCM 0.574  CSII high 0.661  mid 0.473  low 0.141
enhanced NCM 0.490  CSII high 0.660  mid 0.436  low 0.116
```

[`ncm`][kudio.core.intelligibility.ncm] 和 [`csii`][kudio.core.intelligibility.csii] 照 Loizou 的
NCM.m 和 CSII.m，以 0 到 1 估計聽者還能聽懂多少語音。[^mhl09] NCM 在 20 個頻帶上計算兩個訊號
包絡的相關，只支援 8 和 16 kHz。CSII，即相干性 SII，[^ka05] 在 3.7 kHz 以下的臨界頻帶上，分別
對參考訊號響、中、靜的音框量相干的比例（`score` 裡是 `csii_high`、`csii_mid`、`csii_low`）。
兩者都聽不到音量、都需要對齊，短片段讀數偏高，參考訊號上的直流偏移會讓靜音的輸出得分。

[`siib`][kudio.core.siib.siib] 和 [`siib_gauss`][kudio.core.siib.siib_gauss] 以每秒位元數評乾淨
語音和輸出共有的資訊量，[^vkh18a] 用 Kraskov 估計器，[^ksg04] 或快 25 倍的高斯通道，[^vkh18b]
照 SIIB.m 和 SIIB_Gauss.m 計算。複本得到上限 1335.76 bit/s。它們至少需要 5.45 s 的語音（作者
建議 20 s 以上），聽得到音量，需要對齊，其他取樣率先轉成 16 kHz，所以分數要附上取樣率。

## 多個聲源：SI-SDRi、PIT 和 BSS Eval {#several-sources-si-sdri-pit-and-bss-eval}

```python
other = 0.01 * rng.standard_normal(4 * sr)       # a second talker, higher
other[16000:48000] += burst(2.0, 210)
sources = np.stack([clean, other])
mixture = sources.sum(axis=0)
hiss = 0.005 * rng.standard_normal((2, 4 * sr))
separated = np.stack([other + 0.3 * clean,       # in the wrong order, each
                      clean + 0.1 * other]) + hiss   # with some of the other
s = kudio.pit_si_sdr(sources, separated, mixture)
b = kudio.bss_eval(sources, separated, mixture, permutation=s.permutation)
print("matched:", s.permutation)
for label, values in (("SI-SDRi", s.si_sdri), ("SDR", b.sdr), ("SIR", b.sir), ("SAR", b.sar)):
    print(f"{label:8s}", "  ".join(f"{v:6.2f}" for v in values))
print(f"{kudio.si_sdr_improvement(clean, enhanced, noisy):.2f} dB SI-SDRi for the denoiser")
```

```text
matched: (1, 0)
SI-SDRi   17.83   10.13
SDR       18.63    9.68
SIR       20.67    9.92
SAR       22.93   22.76
6.76 dB SI-SDRi for the denoiser
```

[`si_sdr_improvement`][kudio.core.separation.si_sdr_improvement] 和
[`sdr_improvement`][kudio.core.separation.sdr_improvement] 給出輸出比輸入多了多少。[^convtasnet]
[`pit_si_sdr`][kudio.core.separation.pit_si_sdr] 以最高的平均 SI-SDR 把輸出配給聲源，這是語句層級
排列不變訓練的準則，[^upit] [`bss_eval`][kudio.core.separation.bss_eval] 給出 SDR、SIR 和 SAR；[^bss]
它自己配對時用最高的平均 SIR。所有數字都依聲源的順序；論文報的是它們的平均。SI-SDR 先把兩個
訊號都變成零平均，和 Conv-TasNet 的 SI-SNR 一樣。

## 指標評不了的輸入 {#input-a-metric-cannot-score}

每個指標都遵守同一條規則。格式不對的輸入丟出 `ValueError`，訊息以函式名稱開頭；格式對但沒有
東西可評的輸入給 nan 和一次 `UserWarning`，`<function>(): <reason>; scored nan`。訊號是一個聲道
的實數，可以是向量、`(n, 1)` 的欄或 `(1, n)` 的列，整數按型別的滿刻度換算；int64 和 uint64、
複數、布林和文字陣列，以及被遮罩的樣本都會被拒絕。取樣率必須是一個正數。

| 指標 | `ValueError` 的情況 | nan 加警告的情況 |
|---|---|---|
| `snr`、`si_sdr`、`segmental_snr`、`sdi` | NaN 或 inf 樣本、兩個聲道、音框設定錯誤 | 空的一對、靜音的參考（`si_sdr` 是任何常數參考）、大到無法平方的一對；`segmental_snr` 短於一個音框 |
| `pesq` | NaN 或 inf、兩個聲道、8 或 16 kHz 以外的取樣率、PESQ 給不出的模式與刻度組合 | 空的一邊、靜音的參考、短於 0.25 s 的一邊、找不到語句、靜音的輸出 |
| `lsd` | NaN 或 inf、兩個聲道、錯誤的取樣率 | 空的一對、短於一個音框、靜音的參考 |
| `llr`、`itakura_saito`、`cepstral_distance`、`wss`、`fw_segmental_snr`、`composite` | NaN 或 inf、兩個聲道、8 或 16 kHz 以外的取樣率 | 短於 37.5 ms、靜音的參考 |
| `ncm`、`csii` | NaN 或 inf、兩個聲道、`ncm` 或 `csii` 不收的取樣率 | 空的或太短的一對、靜音的參考 |
| `siib`、`siib_gauss` | NaN 或 inf、兩個聲道、低於 4000 Hz 或不是整數的取樣率 | 語音不到 436 個音框、靜音或常數的參考 |
| `stoi`、`estoi`、`sdr` | 兩個聲道 | 空的一對、靜音的一邊、語音太少（STOI） |
| [`wer`、`cer`、`eer`、F0 指標](task-metrics.md) | 配不起來的兩邊、錯誤的標籤或數值 | 沒有東西可評 |

數位靜音的輸出照樣評分，不會被拒絕，因為輸出斷掉是分數應該反映的缺陷。規則怎麼測試，見
[指標驗證](../validation.md#the-input-rule-and-the-import-without-optional-packages)。

## 整個測試集 {#a-whole-test-set}

[`score_dataset`][kudio.core.dataset_score.score_dataset]，或在 shell 裡用
[`kudio score`](../cli.md#score)，依 id 配對檔案，執行 `score` 的任何指標和不需參考訊號的指標，
給出每個平均值和它的信賴區間、語句數、[簽章](#signing-a-result)，以及要引用的文獻：

```python
import re

result = kudio.score_dataset("testset/enhanced", reference="testset/clean",
                             baseline="testset/noisy", metrics=["pesq", "stoi", "si_sdr"],
                             groups=re.compile(r"^(p\d+)_"))
print(result.to_markdown())          # or to_latex(), to_json()
result.save_rows("scores.csv")       # one row per utterance
```

`groups` 可以是第一個群組為說話者的編譯過的正規表示式、dict、函式，或 Kaldi `utt2spk` 或 CSV
檔的名稱。[命令列](../cli.md#score)有它印出的表。

[`AudioEvaluate`][kudio.core.evaluator.AudioEvaluate] 評依位置配對的檔案清單：乾淨參考、有雜訊
的輸入，以及每個方法一份清單，算 PESQ（有裝時）、STOI 和 SDR。`ev.eval(name, files)` 加入一個
方法，[`get_df`][kudio.core.evaluator.AudioEvaluate.get_df] 給出平均值，有雜訊的檔案是 Baseline
那一列（需要 `kudio[data]`），[`per_file`][kudio.core.evaluator.AudioEvaluate.per_file] 每個方法
每個檔案一列，附 `clean`、`noisy` 和 `file` 路徑，可以和 [`Synthesizer`](synthesis.md) 的
manifest 以 `noisy` 合併。某個指標評不了的檔案在那一欄是 NaN，兩邊的平均都排除它，並發出警告。

## 信賴區間與檢定 {#confidence-intervals-and-tests}

```python
rng = np.random.default_rng(23)
speakers = np.repeat([f"spk{k:02d}" for k in range(20)], 30)   # 20 speakers, 30 each
voice = np.repeat(rng.normal(0.0, 0.35, 20), 30)    # how each voice scores
gain = np.repeat(rng.normal(0.12, 0.10, 20), 30)    # how much each voice gains
pesq_base = 2.6 + voice + rng.normal(0.0, 0.3, 600)
pesq_a = pesq_base + gain + rng.normal(0.0, 0.1, 600)
ci = kudio.paired_ci(pesq_a, pesq_base, groups=speakers)
test = kudio.permutation_test(pesq_a, pesq_base, groups=speakers)
print(f"{ci.estimate:+.3f} PESQ, 95% CI [{ci.low:+.3f}, {ci.high:+.3f}], "
      f"{ci.n_groups} speakers, p = {test.pvalue:.4f}")
alone = kudio.paired_ci(pesq_a, pesq_base)
print(f"by utterance: [{alone.low:+.3f}, {alone.high:+.3f}]")
```

```text
+0.106 PESQ, 95% CI [+0.057, +0.168], 20 speakers, p = 0.0007
by utterance: [+0.094, +0.119]
```

[`paired_ci`][kudio.core.stats.paired_ci] 給出兩個系統在同一批語句上平均差異的 bootstrap 區間，
[`permutation_test`][kudio.core.stats.permutation_test] 給出符號翻轉檢定的 p 值。有 `groups` 時，
整個說話者一起重抽、一起翻轉；以語句為單位的區間窄得太多，因為 20 個人的 600 句不是 600 次獨立
抽樣。[`bootstrap_ci`][kudio.core.stats.bootstrap_ci] 給出單一系統的平均，
[`holm`][kudio.core.stats.holm] 調整多個系統對同一基準的 p 值，[`rate_ci`][kudio.core.stats.rate_ci]
和 [`paired_rate_ci`][kudio.core.stats.paired_rate_ci] 收每句的錯誤數和參考長度，所以 WER 在每次
重抽都會重新合併計算。

預設是 BCa 區間、10,000 次重抽、種子 0。群組 13 個以下時，檢定會走過每一種符號組合，p 值是精確
的。群組少於 10 個時區間會警告沒什麼意義，5 個以下時雙尾 p 值不可能低於 0.05。NaN 分數會被拒絕：
把那些語句從每一份清單裡拿掉，`groups` 也一樣。每個區間都要附上說話者人數。

## 報告與引用指標 {#reporting-and-citing-a-metric}

```python
info = kudio.metric_info("csii_high")
print(info.function, info.field, info.better, info.range, info.validation)
text = kudio.signature(["stoi", "si_sdr", "lsd"], sr)
print(text)
print(kudio.signature({"eer": {}, "min_dcf": {"p_target": 0.01}}, n=4874))
print(kudio.parse_signature(text)["lsd"])
```

```text
kudio.csii high higher (0.0, 1.0) validation.md#ncm-and-csii
kudio:5.0.0|sr:16000|pair:aligned=no,cut=shorter,limit=0.5|si_sdr|stoi|lsd:n_fft=512,hop_length=256,scale=ln,match_gain=yes,eps=1e-08
kudio:5.0.0|eer:method=linear|min_dcf:p_target=0.01,c_miss=1.0,c_fa=1.0|n:4874
{'n_fft': '512', 'hop_length': '256', 'scale': 'ln', 'match_gain': 'yes', 'eps': '1e-08'}
```

[`metric_info`][kudio.core.registry.metric_info] 給出每個指標的方向、範圍、會改變數字的設定
（`variants`）和它出自的文獻（`refs`）；[`cite`][kudio.core.registry.cite] 寫出它們的 BibTeX，
kudio 自己的條目排第一。

### 為結果簽章 {#signing-a-result}

[`signature`][kudio.core.signature.signature] 仿照 sacreBLEU[^post] 寫出一行，列出每個會改變數字
的設定解析後的值（PESQ 的 `'auto'` 在 16 kHz 寫成 `mode=wb`）、這一對怎麼處理（`aligned`、
`cut`、`limit`），以及算出數字的套件版本，包括 `pesq`。指標依登記表的順序排列，所以同一組指標和
設定只會得到一行。單獨呼叫的指標以 dict 傳入它的關鍵字參數；`n=` 和 `ci=` 記錄評了多少項、區間
怎麼算的，DNSMOS 會再加上模型 SHA-256 的 8 個十六進位字元。請在算出數字的那次執行裡寫簽章；
[`parse_signature`][kudio.core.signature.parse_signature] 把它讀回來。參數在
[指標與對齊](../reference/metrics.md)。

[^sisdr]: J. Le Roux, S. Wisdom, H. Erdogan and J. R. Hershey, "SDR – half-baked or well done?", Proc. ICASSP 2019, pp. 626-630, doi:10.1109/ICASSP.2019.8683855.
[^sdi]: J. Chen, J. Benesty, Y. Huang and S. Doclo, "New insights into the noise reduction Wiener filter", IEEE Trans. Audio, Speech, Lang. Process. 14(4), pp. 1218-1234, 2006, doi:10.1109/TSA.2005.860851. The index is their Eq. 34.
[^stoi]: C. H. Taal, R. C. Hendriks, R. Heusdens and J. Jensen, "An algorithm for intelligibility prediction of time-frequency weighted noisy speech", IEEE Trans. Audio, Speech, Lang. Process. 19(7), pp. 2125-2136, 2011, doi:10.1109/TASL.2011.2114881. ESTOI: J. Jensen and C. H. Taal, "An algorithm for predicting the intelligibility of speech masked by modulated noise maskers", IEEE/ACM Trans. Audio, Speech, Lang. Process. 24(11), pp. 2009-2022, 2016, doi:10.1109/TASLP.2016.2585878.
[^p862]: ITU-T Recommendation P.862 (02/2001), "Perceptual evaluation of speech quality (PESQ): An objective method for end-to-end speech quality assessment of narrow-band telephone networks and speech codecs"; P.862.1 (11/2003), "Mapping function for transforming P.862 raw result scores to MOS-LQO"; P.862.2 (11/2007), "Wideband extension to Recommendation P.862 for the assessment of wideband telephone networks and speech codecs". P.862 Amendment 2 (11/2005), "Revised Annex A – Reference implementations and conformance testing for ITU-T Recs P.862, P.862.1 and P.862.2", whose A.2 sets the sampling rates and whose C code, version 2.0, is the reference program; P.862 Corrigendum 2 (03/2018). ITU withdrew P.862, P.862.1, P.862.2 and P.862.3 on 5 January 2024 and points to P.863 instead.
[^torcoli]: M. Torcoli, M. M. Halimeh and E. A. P. Habets, "Navigating PESQ: Up-to-date versions and open implementations", ITG Conference on Speech Communication 2025, arXiv:2505.19760.
[^p835]: ITU-T Recommendation P.835 (11/2003), "Subjective test methodology for evaluating speech communication systems that include noise suppression algorithm".
[^hl08]: Y. Hu and P. C. Loizou, "Evaluation of objective quality measures for speech enhancement", IEEE Trans. Audio, Speech, Lang. Process. 16(1), pp. 229-238, 2008, doi:10.1109/TASL.2007.911054. The CSIG, CBAK and COVL equations: Y. Hu and P. C. Loizou, "Evaluation of objective measures for speech enhancement", Proc. Interspeech 2006, doi:10.21437/Interspeech.2006-84.
[^urgent]: K. Saijo, W. Zhang, S. Cornell, R. Scheibler, C. Li, Z. Ni, A. Kumar, M. Sach, Y. Fu, W. Wang, T. Fingscheidt and S. Watanabe, "Interspeech 2025 URGENT speech enhancement challenge", Proc. Interspeech 2025, pp. 858-862, doi:10.21437/Interspeech.2025-1363. The scoring script is `lsd_metric` in `evaluation_metrics/calculate_intrusive_se_metrics.py`, <https://github.com/urgent-challenge/urgent2025_challenge>.
[^nvsr]: H. Liu, W. Choi, X. Liu, Q. Kong, Q. Tian and D. Wang, "Neural vocoder is all you need for speech super-resolution", Proc. Interspeech 2022, pp. 4227-4231, doi:10.21437/Interspeech.2022-11017, Eq. 7; scored with `ssr_eval`, <https://github.com/haoheliu/ssr_eval>.
[^mhl09]: J. Ma, Y. Hu and P. C. Loizou, "Objective measures for predicting speech intelligibility in noisy conditions based on new band-importance functions", J. Acoust. Soc. Am. 125(5), pp. 3387-3405, 2009, doi:10.1121/1.3097493.
[^ka05]: J. M. Kates and K. H. Arehart, "Coherence and the speech intelligibility index", J. Acoust. Soc. Am. 117(4), pp. 2224-2237, 2005, doi:10.1121/1.1862575.
[^bss]: E. Vincent, R. Gribonval and C. Févotte, "Performance measurement in blind audio source separation", IEEE Trans. Audio, Speech, Lang. Process. 14(4), pp. 1462-1469, 2006, doi:10.1109/TSA.2005.858005.
[^upit]: M. Kolbæk, D. Yu, Z.-H. Tan and J. Jensen, "Multitalker speech separation with utterance-level permutation invariant training of deep recurrent neural networks", IEEE/ACM Trans. Audio, Speech, Lang. Process. 25(10), pp. 1901-1913, 2017, doi:10.1109/TASLP.2017.2726762.
[^convtasnet]: Y. Luo and N. Mesgarani, "Conv-TasNet: surpassing ideal time-frequency magnitude masking for speech separation", IEEE/ACM Trans. Audio, Speech, Lang. Process. 27(8), pp. 1256-1266, 2019, doi:10.1109/TASLP.2019.2915167. SI-SNR is their eq. 15, both signals taken to zero mean; SI-SNRi and SDRi are "the additive values over the original mixture".
[^post]: M. Post, "A call for clarity in reporting BLEU scores", Proc. Third Conference on Machine Translation: Research Papers, Brussels, October 2018, pp. 186-191, doi:10.18653/v1/W18-6319.
[^vkh18a]: S. Van Kuyk, W. B. Kleijn and R. C. Hendriks, "An instrumental intelligibility metric based on information theory", IEEE Signal Process. Lett. 25(1), pp. 115-119, 2018, doi:10.1109/LSP.2017.2774250.
[^vkh18b]: S. Van Kuyk, W. B. Kleijn and R. C. Hendriks, "An evaluation of intrusive instrumental intelligibility metrics", IEEE/ACM Trans. Audio, Speech, Lang. Process. 26(11), pp. 2153-2166, 2018, doi:10.1109/TASLP.2018.2856374.
[^ksg04]: A. Kraskov, H. Stögbauer and P. Grassberger, "Estimating mutual information", Phys. Rev. E 69(6), 066138, 2004, doi:10.1103/PhysRevE.69.066138.
